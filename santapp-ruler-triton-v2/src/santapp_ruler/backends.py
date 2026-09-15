"""Model loading and the four supported generation backends."""

from __future__ import annotations

import gc
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

from .attention.hierarchical import HierarchicalWholeTeamEngine
from .attention.santapp import SantappEngine
from .attention.qwen import peak_memory_gib, reset_peak_memory
from .attention.santa import SantaEngine
from .config import HierarchicalConfig, ModelConfig, SantaConfig, SantappConfig


def _cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def resolve_dtype(name: str) -> torch.dtype:
    values = {"float16": torch.float16, "bfloat16": torch.bfloat16}
    try:
        return values[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype {name!r}; choose {sorted(values)}") from exc


def encode_prompt(tokenizer: Any, prompt: str, prompt_format: str) -> torch.Tensor:
    """Tokenize one official RULER prompt with an explicit framing policy."""

    if prompt_format == "raw":
        encoded = tokenizer(
            prompt,
            add_special_tokens=False,
            return_tensors="pt",
        )
        return encoded.input_ids
    if prompt_format == "chat_template":
        if not hasattr(tokenizer, "apply_chat_template"):
            raise TypeError("The selected tokenizer does not expose apply_chat_template.")
        encoded = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        if isinstance(encoded, Mapping):
            encoded = encoded["input_ids"]
        if not isinstance(encoded, torch.Tensor):
            encoded = torch.tensor(encoded, dtype=torch.long)
        if encoded.ndim == 1:
            encoded = encoded.unsqueeze(0)
        return encoded
    raise ValueError(f"Unknown prompt_format: {prompt_format!r}")


@dataclass(slots=True)
class ModelBundle:
    model: Any
    tokenizer: Any
    device: torch.device
    eos_token_ids: set[int]
    prompt_format: str

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

        return cls(
            model=model,
            tokenizer=tokenizer,
            device=device,
            eos_token_ids=eos_ids,
            prompt_format=config.prompt_format,
        )

    def tokenize(self, prompt: str) -> torch.Tensor:
        return encode_prompt(self.tokenizer, prompt, self.prompt_format).to(self.device)

    def decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def release_example_memory(self) -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


@dataclass(frozen=True, slots=True)
class BackendGeneration:
    token_ids: list[int]
    prediction: str
    metrics: dict[str, Any]


def _dense_access_metrics(
    model: Any,
    *,
    prompt_tokens: int,
    generated_count: int,
) -> dict[str, Any]:
    layers = len(model.model.layers)
    query_heads = int(model.config.num_attention_heads)
    kv_heads = int(model.config.num_key_value_heads)
    token_sum = sum(prompt_tokens + step for step in range(generated_count))
    head_calls = generated_count * layers * query_heads
    group_calls = generated_count * layers * kv_heads
    dense_gqa = 2 * token_sum * layers * kv_heads
    dense_naive = 2 * token_sum * layers * query_heads
    mean_total = token_sum / generated_count if generated_count else 0.0
    return {
        "decode_attention_head_calls": head_calls,
        "decode_gqa_group_calls": group_calls,
        "decode_dense_gqa_kv_vectors": dense_gqa,
        "decode_dense_naive_kv_vectors": dense_naive,
        "decode_gqa_kv_vectors_read": dense_gqa,
        "decode_naive_kv_vectors_read": dense_naive,
        "decode_gqa_routing_key_vectors_read": 0,
        "decode_naive_routing_key_vectors_read": 0,
        "decode_gqa_centroid_key_vectors_read": 0,
        "decode_naive_centroid_key_vectors_read": 0,
        "decode_gqa_total_vectors_read": dense_gqa,
        "decode_naive_total_vectors_read": dense_naive,
        "decode_gqa_kv_access_pct": 100.0 if dense_gqa else 0.0,
        "decode_gqa_routing_access_pct": 0.0,
        "decode_gqa_centroid_access_pct": 0.0,
        "decode_gqa_total_access_pct": 100.0 if dense_gqa else 0.0,
        "decode_naive_kv_access_pct": 100.0 if dense_naive else 0.0,
        "decode_naive_routing_access_pct": 0.0,
        "decode_naive_centroid_access_pct": 0.0,
        "decode_naive_total_access_pct": 100.0 if dense_naive else 0.0,
        "mean_sampled_token_draws_per_head_call": 0.0,
        "mean_sampled_token_rows_per_head_call": 0.0,
        "mean_unique_sampled_tokens_per_gqa_group_call": 0.0,
        "mean_exact_tokens_per_gqa_group_call": mean_total,
        "mean_total_tokens_per_head_call": mean_total,
    }


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
        del random_seed
        model = self.bundle.model
        prompt_tokens = int(input_ids.shape[1])
        attention_mask = torch.ones_like(input_ids)
        reset_peak_memory(input_ids.device)

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
        peak_allocated_gib, peak_reserved_gib = peak_memory_gib(input_ids.device)
        metrics: dict[str, Any] = {
            "backend": self.name,
            "mode": "dense_sdpa",
            "sampling_scheme": "none",
            "sampling_unit": "none",
            "importance_correction": "not_applicable",
            "routing_key_type": "none",
            "prompt_tokens": prompt_tokens,
            "generated_tokens": generated_count,
            "max_new_tokens": max_new_tokens,
            "prefill_seconds": prefill_seconds,
            "parent_build_seconds": 0.0,
            "clustering_seconds": 0.0,
            "decode_seconds": decode_seconds,
            "total_seconds": total_seconds,
            "prompt_exact_tail_tokens": prompt_tokens,
            "initial_generated_exact_tokens": 0,
            "exact_token_policy": "dense_all_tokens",
            "parent_policy": "not_applicable",
            "probe_policy": "not_applicable",
            "routing_summary_gib": 0.0,
            **_dense_access_metrics(
                model,
                prompt_tokens=prompt_tokens,
                generated_count=generated_count,
            ),
            "peak_allocated_gib": peak_allocated_gib,
            "peak_reserved_gib": peak_reserved_gib,
        }
        prediction = self.bundle.decode(generated)
        del output, past, attention_mask
        return BackendGeneration(generated, prediction, metrics)


class SantaBackend:
    name = "santa"

    def __init__(self, bundle: ModelBundle, config: SantaConfig):
        self.bundle = bundle
        self.engine = SantaEngine(bundle.model, config)

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
        return BackendGeneration(
            result.token_ids,
            self.bundle.decode(result.token_ids),
            {**result.metrics, "backend": self.name},
        )


class SantappBackend:
    name = "santapp"

    def __init__(self, bundle: ModelBundle, config: SantappConfig):
        self.bundle = bundle
        self.engine = SantappEngine(bundle.model, config)

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
        return BackendGeneration(
            result.token_ids,
            self.bundle.decode(result.token_ids),
            {**result.metrics, "backend": self.name},
        )


class HierarchicalBackend:
    """Contiguous-span parent variant of hierarchical whole-team sampling."""

    name = "hierarchical"

    def __init__(self, bundle: ModelBundle, config: HierarchicalConfig):
        self.bundle = bundle
        self.engine = HierarchicalWholeTeamEngine(bundle.model, config)

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
        return BackendGeneration(
            result.token_ids,
            self.bundle.decode(result.token_ids),
            {**result.metrics, "backend": self.name},
        )

