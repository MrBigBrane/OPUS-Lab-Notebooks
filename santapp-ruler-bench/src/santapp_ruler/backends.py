"""Model loading and generation backends."""

from __future__ import annotations

import gc
import os
import time
from dataclasses import dataclass
from typing import Any

import torch

from .attention.santapp import SantaPlusEngine
from .config import ModelConfig, SantaPlusConfig


def _cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def resolve_dtype(name: str) -> torch.dtype:
    values = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    try:
        return values[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype {name!r}; choose {sorted(values)}") from exc


@dataclass(slots=True)
class ModelBundle:
    model: Any
    tokenizer: Any
    device: torch.device
    eos_token_ids: set[int]

    @classmethod
    def load(cls, config: ModelConfig) -> "ModelBundle":
        if config.disable_hf_xet:
            os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is unavailable. This benchmark requires an NVIDIA GPU; "
                "run `santapp-ruler doctor` for diagnostics."
            )

        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = torch.device(config.device)
        dtype = resolve_dtype(config.dtype)
        tokenizer = AutoTokenizer.from_pretrained(
            config.name,
            revision=config.revision,
            trust_remote_code=config.trust_remote_code,
            use_fast=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            config.name,
            revision=config.revision,
            dtype=dtype,
            attn_implementation=config.attn_implementation,
            trust_remote_code=config.trust_remote_code,
            low_cpu_mem_usage=True,
        )
        model.to(device)
        model.eval()

        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        eos_ids: set[int] = set()
        for source in (
            tokenizer.eos_token_id,
            getattr(model.generation_config, "eos_token_id", None),
        ):
            if source is None:
                continue
            if isinstance(source, (list, tuple, set)):
                eos_ids.update(int(value) for value in source)
            else:
                eos_ids.add(int(source))

        return cls(model=model, tokenizer=tokenizer, device=device, eos_token_ids=eos_ids)

    def tokenize(self, prompt: str) -> torch.Tensor:
        encoded = self.tokenizer(
            prompt,
            add_special_tokens=False,
            return_tensors="pt",
        )
        return encoded.input_ids.to(self.device)

    def decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def release_example_memory(self) -> None:
        gc.collect()
        torch.cuda.empty_cache()


@dataclass(frozen=True, slots=True)
class BackendGeneration:
    token_ids: list[int]
    prediction: str
    metrics: dict[str, Any]


class SdpaBackend:
    name = "sdpa"

    def __init__(self, bundle: ModelBundle):
        self.bundle = bundle

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int,
        stop_on_eos: bool,
        random_seed: int,
    ) -> BackendGeneration:
        del random_seed  # greedy SDPA is deterministic given the model/runtime
        model = self.bundle.model
        prompt_tokens = int(input_ids.shape[1])
        attention_mask = torch.ones_like(input_ids)
        torch.cuda.reset_peak_memory_stats(input_ids.device)

        _cuda_sync()
        total_start = time.perf_counter()
        prefill_start = time.perf_counter()
        output = model(
            input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            logits_to_keep=1,
        )
        past = output.past_key_values
        _cuda_sync()
        prefill_seconds = time.perf_counter() - prefill_start

        generated: list[int] = []
        _cuda_sync()
        decode_start = time.perf_counter()
        for step in range(max_new_tokens):
            next_token = output.logits[0, -1].argmax().view(1, 1)
            token_id = int(next_token.item())
            generated.append(token_id)
            if stop_on_eos and token_id in self.bundle.eos_token_ids:
                break
            if step + 1 == max_new_tokens:
                break

            attention_mask = torch.cat(
                (attention_mask, torch.ones_like(next_token)), dim=1
            )
            output = model(
                next_token,
                attention_mask=attention_mask,
                past_key_values=past,
                use_cache=True,
                logits_to_keep=1,
            )
            past = output.past_key_values
        _cuda_sync()
        decode_seconds = time.perf_counter() - decode_start
        total_seconds = time.perf_counter() - total_start

        generated_count = len(generated)
        attention_head_calls = (
            generated_count
            * len(model.model.layers)
            * int(model.config.num_attention_heads)
        )
        # Conceptual dense decode reads for the queries producing each output.
        token_sum = sum(prompt_tokens + step for step in range(generated_count))
        dense_vectors = (
            2
            * token_sum
            * len(model.model.layers)
            * int(model.config.num_attention_heads)
        )
        metrics: dict[str, Any] = {
            "backend": self.name,
            "prompt_tokens": prompt_tokens,
            "generated_tokens": generated_count,
            "max_new_tokens": max_new_tokens,
            "prefill_seconds": prefill_seconds,
            "clustering_seconds": 0.0,
            "decode_seconds": decode_seconds,
            "total_seconds": total_seconds,
            "decode_attention_head_calls": attention_head_calls,
            "decode_dense_kv_vectors": dense_vectors,
            "decode_kv_vectors_read": dense_vectors,
            "decode_metadata_key_vectors_read": 0,
            "decode_kv_access_pct": 100.0,
            "decode_read_equivalent_pct": 100.0,
            "decode_metadata_equivalent_pct": 0.0,
            "mean_ess_over_samples": None,
            "peak_allocated_gib": torch.cuda.max_memory_allocated(input_ids.device)
            / (1024**3),
            "peak_reserved_gib": torch.cuda.max_memory_reserved(input_ids.device)
            / (1024**3),
        }
        prediction = self.bundle.decode(generated)
        del output, past, attention_mask
        return BackendGeneration(generated, prediction, metrics)


class SantaPlusBackend:
    name = "santapp"

    def __init__(self, bundle: ModelBundle, config: SantaPlusConfig):
        self.bundle = bundle
        self.engine = SantaPlusEngine(bundle.model, config)

    def generate(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int,
        stop_on_eos: bool,
        random_seed: int,
    ) -> BackendGeneration:
        result = self.engine.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            eos_token_ids=self.bundle.eos_token_ids,
            stop_on_eos=stop_on_eos,
            random_seed=random_seed,
        )
        prediction = self.bundle.decode(result.token_ids)
        return BackendGeneration(result.token_ids, prediction, result.metrics)
