"""Shared Qwen2/Qwen2.5 cache and temporary attention patch.

This module contains model plumbing only. The sparse estimators live in
:mod:`santa`, :mod:`santapp`, and :mod:`hierarchical`.
"""

from __future__ import annotations

import contextlib
import os
import time
import types
from dataclasses import dataclass
from typing import Any, Iterator

import torch

try:
    from transformers.integrations.sdpa_attention import (
        sdpa_attention_forward as hf_sdpa_attention_forward,
    )
except ImportError:  # Unit tests that do not load a model remain importable.
    hf_sdpa_attention_forward = None


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def uses_cuda(device: torch.device | str) -> bool:
    return torch.device(device).type == "cuda" and torch.cuda.is_available()


def cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def seed_torch(seed: int, device: torch.device | str) -> None:
    torch.manual_seed(seed)
    if uses_cuda(device):
        torch.cuda.manual_seed_all(seed)


def reset_peak_memory(device: torch.device | str) -> None:
    if uses_cuda(device):
        torch.cuda.reset_peak_memory_stats(device)


def peak_memory_gib(device: torch.device | str) -> tuple[float, float]:
    if not uses_cuda(device):
        return 0.0, 0.0
    return (
        torch.cuda.max_memory_allocated(device) / (1024**3),
        torch.cuda.max_memory_reserved(device) / (1024**3),
    )


@dataclass(slots=True)
class LayerCache:
    key: torch.Tensor
    value: torch.Tensor
    length: int

    @property
    def capacity(self) -> int:
        return int(self.key.shape[1])

    def view(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.key[:, : self.length], self.value[:, : self.length]


@dataclass(frozen=True, slots=True)
class AttentionEstimate:
    output: torch.Tensor
    sampled_indices: torch.Tensor
    selected_teams: int = 0
    inclusion_probabilities: torch.Tensor | None = None


@dataclass(frozen=True, slots=True)
class AttentionGeneration:
    token_ids: list[int]
    metrics: dict[str, Any]


_MISSING = object()


class QwenSparseAttentionEngine:
    """Patch one loaded Qwen model only for the lifetime of a generation call."""

    def __init__(self, model: Any, config: Any):
        self.model = model
        self.config = config
        self.base_model = getattr(model, "model", None)
        if self.base_model is None or not hasattr(self.base_model, "layers"):
            raise TypeError("Expected a Hugging Face causal LM with model.layers.")
        model_type = getattr(model.config, "model_type", None)
        if model_type != "qwen2":
            raise TypeError(
                "The sparse attention patch currently supports Qwen2/Qwen2.5 "
                f"(model_type='qwen2'), not {model_type!r}. See PORTABILITY.md "
                "before adapting it to another model family."
            )

        cfg = model.config
        if bool(getattr(cfg, "use_sliding_window", False)):
            raise NotImplementedError(
                "The sparse patch requires full attention in every layer; "
                "use_sliding_window=True is unsupported."
            )
        layer_types = getattr(cfg, "layer_types", None)
        if layer_types is not None and any(
            layer_type != "full_attention" for layer_type in layer_types
        ):
            raise NotImplementedError(
                "The sparse patch requires every layer type to be 'full_attention'."
            )

        self.num_layers = len(self.base_model.layers)
        self.num_query_heads = int(cfg.num_attention_heads)
        self.num_kv_heads = int(cfg.num_key_value_heads)
        if self.num_query_heads % self.num_kv_heads:
            raise ValueError("Query heads must be divisible by KV heads.")
        self.query_heads_per_kv = self.num_query_heads // self.num_kv_heads
        self.head_dim = int(
            getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
        )

        self.mode = "dense"
        self.sample_end = 0
        self.cache_capacity = 0
        self.cache: dict[int, LayerCache] = {}
        self._grouped_decoders: dict[int, Any] = {}
        self._grouped_statistics: torch.Tensor | None = None
        self._grouped_sampling: dict[str, Any] | None = None
        self._grouped_extra: dict[str, Any] = {}
        self._original_forwards: list[Any] = []
        self._reset_sampling_diagnostics()

    def _make_traffic_tracker(self) -> Any:
        raise NotImplementedError

    def _reset_sampling_diagnostics(self) -> None:
        self._sampled_rows_total = 0
        self._selected_teams_total = 0
        self._selected_team_head_calls = 0
        self._selected_inclusion_probabilities: list[torch.Tensor] = []

    @contextlib.contextmanager
    def patched(self) -> Iterator[None]:
        self._original_forwards = []
        patched_attentions: list[Any] = []
        try:
            for layer_id, layer in enumerate(self.base_model.layers):
                attention = layer.self_attn
                original = attention.__dict__.get("forward", _MISSING)
                self._original_forwards.append(original)
                patched_attentions.append(attention)
                attention._sparse_attention_engine = self
                attention._sparse_attention_layer_id = layer_id
                attention.forward = types.MethodType(_patched_forward, attention)
            yield
        finally:
            for attention, original in zip(
                patched_attentions, self._original_forwards, strict=True
            ):
                if original is _MISSING:
                    attention.__dict__.pop("forward", None)
                else:
                    attention.forward = original
                attention.__dict__.pop("_sparse_attention_engine", None)
                attention.__dict__.pop("_sparse_attention_layer_id", None)
            self._original_forwards = []

    def clear(self) -> None:
        self.cache.clear()
        self._grouped_decoders.clear()
        self._grouped_statistics = None
        self._grouped_sampling = None
        self._grouped_extra = {}
        self._reset_sampling_diagnostics()

    def _append_cache(
        self,
        layer_id: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # key/value arrive as [1, n_kv, T, d].
        key0 = key[0]
        value0 = value[0]
        tokens = int(key0.shape[1])
        cached = self.cache.get(layer_id)
        if cached is None:
            capacity = max(self.cache_capacity, tokens)
            key_buffer = torch.empty(
                self.num_kv_heads,
                capacity,
                self.head_dim,
                dtype=key0.dtype,
                device=key0.device,
            )
            value_buffer = torch.empty_like(key_buffer)
            key_buffer[:, :tokens].copy_(key0)
            value_buffer[:, :tokens].copy_(value0)
            self.cache[layer_id] = LayerCache(key_buffer, value_buffer, tokens)
            return key0.contiguous(), value0.contiguous()

        start = cached.length
        end = start + tokens
        if end > cached.capacity:
            raise RuntimeError(
                f"Custom cache capacity exceeded in layer {layer_id}: "
                f"need {end}, allocated {cached.capacity}."
            )
        cached.key[:, start:end].copy_(key0)
        cached.value[:, start:end].copy_(value0)
        cached.length = end
        decoder = self._grouped_decoders.get(layer_id)
        if decoder is not None and start < self.sample_end:
            decoder.refresh(cached.key, cached.value, start, end)
        full_key, full_value = cached.view()
        if self.mode == "dense":
            return full_key.contiguous(), full_value.contiguous()
        return full_key, full_value

    def _trim_cache(self, length: int) -> None:
        for layer_id, cached in self.cache.items():
            if cached.length < length:
                raise RuntimeError(
                    f"Layer {layer_id} cache has only {cached.length} tokens; "
                    f"cannot trim to {length}."
                )
            cached.length = length

    def _attention_forward(
        self,
        attention: Any,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
        attention_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, None]:
        del attention_mask
        batch, query_tokens, _ = hidden_states.shape
        if batch != 1:
            raise NotImplementedError("The custom cache supports batch size 1.")
        if position_embeddings is None:
            raise RuntimeError("Qwen did not supply position_embeddings.")

        query = attention.q_proj(hidden_states).view(
            batch, query_tokens, self.num_query_heads, self.head_dim
        ).transpose(1, 2)
        key = attention.k_proj(hidden_states).view(
            batch, query_tokens, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)
        value = attention.v_proj(hidden_states).view(
            batch, query_tokens, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)

        cos, sin = position_embeddings

        def apply_rope(x: torch.Tensor) -> torch.Tensor:
            return x * cos[:, None, :, :] + rotate_half(x) * sin[:, None, :, :]

        query = apply_rope(query)
        key = apply_rope(key)

        layer_id = int(attention._sparse_attention_layer_id)
        full_key, full_value = self._append_cache(layer_id, key, value)
        key_tokens = int(full_key.shape[1])

        if query_tokens > 1 or self.mode == "dense":
            if query_tokens == key_tokens:
                dense_mask = None
                dense_is_causal = query_tokens > 1
            elif query_tokens == 1:
                dense_mask = None
                dense_is_causal = False
            else:
                query_absolute = torch.arange(
                    key_tokens - query_tokens,
                    key_tokens,
                    device=query.device,
                )[:, None]
                key_absolute = torch.arange(key_tokens, device=query.device)[None, :]
                dense_mask = (key_absolute <= query_absolute)[None, None, :, :]
                dense_is_causal = False

            if hf_sdpa_attention_forward is None:
                raise RuntimeError(
                    "This implementation expects transformers==5.12.1 and its "
                    "SDPA integration. Install the pinned project dependencies."
                )
            output, _ = hf_sdpa_attention_forward(
                attention,
                query,
                full_key.unsqueeze(0),
                full_value.unsqueeze(0),
                dense_mask,
                dropout=0.0,
                scaling=attention.scaling,
                is_causal=dense_is_causal,
            )
        elif self._grouped_decoders:
            decoder = self._grouped_decoders[layer_id]
            epoch = key_tokens - self.sample_end
            output = decoder.run(
                query[0, :, 0].contiguous(), full_key, full_value,
                seed=self._grouped_seed, layer=layer_id, epoch=epoch,
                statistics=self._grouped_statistics[epoch, layer_id],
            )[None, None]
        else:
            outputs: list[torch.Tensor] = []
            for kv_head in range(self.num_kv_heads):
                sampled_indices_by_head: list[torch.Tensor] = []
                for query_head in range(
                    kv_head * self.query_heads_per_kv,
                    (kv_head + 1) * self.query_heads_per_kv,
                ):
                    estimate = self._approximate_attention(
                        query[0, query_head, 0].float(),
                        full_key[kv_head],
                        full_value[kv_head],
                        layer_id,
                        kv_head,
                    )
                    outputs.append(estimate.output)
                    sampled_indices_by_head.append(estimate.sampled_indices)
                    self._sampled_rows_total += int(estimate.sampled_indices.numel())
                    if estimate.selected_teams:
                        self._selected_teams_total += estimate.selected_teams
                        self._selected_team_head_calls += 1
                    if estimate.inclusion_probabilities is not None:
                        self._selected_inclusion_probabilities.append(
                            estimate.inclusion_probabilities.detach().reshape(-1)
                        )
                self._record_decode_group(
                    layer_id=layer_id,
                    kv_head=kv_head,
                    n_total=key_tokens,
                    sampled_indices_by_head=sampled_indices_by_head,
                )
            output = torch.stack(outputs)[None, None].to(hidden_states.dtype)

        output = output.reshape(
            batch, query_tokens, self.num_query_heads * self.head_dim
        ).contiguous()
        return attention.o_proj(output), None

    def _prepare_grouped_decode(self, *, max_new_tokens: int, random_seed: int) -> None:
        """Pack and compile each layer outside decode, without changing prefill."""
        if getattr(self.config, "decode_backend", "torch") != "grouped_triton":
            return
        from .triton_decode.packing import requested_teams
        from .triton_decode.runtime import LayerDecoder
        budget = requested_teams(self.config.samples_per_head, self.config.parent_size,
                                 self.config.representatives_per_parent)
        self._grouped_seed = random_seed
        start = time.perf_counter()
        device = self.cache[0].key.device
        self._grouped_statistics = torch.empty(
            (max_new_tokens, self.num_layers, self.num_kv_heads, 6),
            dtype=torch.float64, device=device,
        )
        if os.environ.get("SANTAPP_PHASE_LOGS") == "1":
            print(f"PHASE_START=decode_prepare unix={time.time():.6f}", flush=True)
        for layer_id in range(self.num_layers):
            key, value = self.cache[layer_id].view()
            decoder = LayerDecoder(
                [self.summaries[(layer_id, h)] for h in range(self.num_kv_heads)],
                key, value, query_heads=self.num_query_heads, budget=budget,
                splits=self.config.decode_splits,
            )
            self._grouped_decoders[layer_id] = decoder
            decoder.warmup(key, value, seed=random_seed, layer=layer_id)
        cuda_sync()
        self._grouped_extra = {
            "decode_prepare_seconds": time.perf_counter() - start,
            "decode_prepare_includes_packing_and_jit_warmup": True,
            "decode_cache_layout": "team_major_prompt_plus_live_exact_suffix",
            "decode_routing": "grouped_gqa_native_mma_batched_torch_topk",
            "decode_rng": "philox_seed_layer_epoch_queryhead_team",
            "decode_topk_tie_policy": "torch_topk_unspecified_ties",
            "decode_diagnostics": "device_aggregate_then_one_host_transfer",
            "decode_attention_splits": self.config.decode_splits,
            "decode_packed_storage_gib": sum(d.packed.storage_bytes()
                for d in self._grouped_decoders.values()) / (1024**3),
            "decode_workspace_gib": sum(d.workspace_bytes()
                for d in self._grouped_decoders.values()) / (1024**3),
            "decode_diagnostics_storage_gib": self._grouped_statistics.numel()*8 / (1024**3),
        }
        if os.environ.get("SANTAPP_PHASE_LOGS") == "1":
            print(f"PHASE_END=decode_prepare unix={time.time():.6f}", flush=True)

    def _finalize_grouped_decode(self, generated_count: int) -> None:
        if not self._grouped_decoders:
            return
        # A single D2H transfer after timed decode. All remaining accounting is CPU.
        records = self._grouped_statistics[:generated_count].cpu().numpy()
        probability_sum = 0.0
        count = 0
        min_p, max_p = 1.0, 0.0
        for step in range(generated_count):
            for layer in range(self.num_layers):
                counts = self._grouped_decoders[layer].packed.counts_host
                for head in range(self.num_kv_heads):
                    naive, union, selected, p_sum, p_min, p_max = records[step, layer, head]
                    self.traffic.record_aggregated_group(
                        n_total=self.sample_end+step, exact_tokens=step,
                        sampled_token_rows=int(naive), unique_sampled_rows=int(union),
                        routing_key_vectors=counts[head],
                    )
                    count += int(selected)
                    probability_sum += float(p_sum)
                    min_p, max_p = min(min_p, float(p_min)), max(max_p, float(p_max))
        calls = generated_count * self.num_layers * self.num_query_heads
        self._grouped_sampling = {
            "mean_selected_teams_per_head_call": count/calls if calls else 0.0,
            "selected_team_inclusion_probability_count": count,
            "mean_selected_team_inclusion_probability": probability_sum/count if count else None,
            "min_selected_team_inclusion_probability": min_p if count else None,
            "max_selected_team_inclusion_probability": max_p if count else None,
        }

    def _decode_metadata(self) -> dict[str, Any]:
        return {"decode_backend": getattr(self.config, "decode_backend", "torch"),
                **self._grouped_extra}

    def _run_sparse_tokens(self, input_ids: torch.Tensor, *, max_new_tokens: int,
                           stop_on_eos: bool, eos_token_ids: set[int]) -> tuple[list[int], float]:
        """Preserve token semantics; avoid per-token .item() when EOS stopping is off."""
        prompt_tokens = int(input_ids.shape[1])
        current = input_ids[:, -1:]
        positions = torch.arange(prompt_tokens-1, prompt_tokens-1+max_new_tokens,
                                 device=input_ids.device, dtype=torch.long)
        device_tokens = torch.empty(max_new_tokens, device=input_ids.device, dtype=torch.long)
        generated: list[int] = []
        cuda_sync()
        wall_start = time.time()
        logging = os.environ.get("SANTAPP_PHASE_LOGS") == "1"
        if logging:
            print(f"PHASE_START=decode unix={wall_start:.6f} "
                  f"backend={getattr(self.config, 'decode_backend', 'torch')}", flush=True)
        decode_start = time.perf_counter()
        completed = 0
        with torch.profiler.record_function("santapp_sparse_decode"):
            for step in range(max_new_tokens):
                output = self.model(
                    current, position_ids=positions[step:step+1].view(1, 1),
                    use_cache=False, logits_to_keep=1,
                )
                current = output.logits[0, -1].argmax().view(1, 1)
                completed = step+1
                if stop_on_eos:
                    token_id = int(current.item())
                    generated.append(token_id)
                    if token_id in eos_token_ids:
                        break
                else:
                    device_tokens[step].copy_(current.reshape(()))
        cuda_sync()
        seconds = time.perf_counter() - decode_start
        wall_end = time.time()
        if logging:
            print(f"PHASE_END=decode unix={wall_end:.6f} tokens={completed} "
                  f"seconds={seconds:.6f}", flush=True)
        self._grouped_extra.update(decode_started_at_unix=wall_start,
                                   decode_finished_at_unix=wall_end)
        if not stop_on_eos:
            generated = device_tokens[:completed].cpu().tolist()
        return generated, seconds

    def _approximate_attention(
        self,
        query: torch.Tensor,
        full_key: torch.Tensor,
        full_value: torch.Tensor,
        layer_id: int,
        kv_head: int,
    ) -> AttentionEstimate:
        raise NotImplementedError

    def _record_decode_group(
        self,
        *,
        layer_id: int,
        kv_head: int,
        n_total: int,
        sampled_indices_by_head: list[torch.Tensor],
    ) -> None:
        raise NotImplementedError

    @torch.inference_mode()
    def generate_dense_reference(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int,
    ) -> list[int]:
        """Run the custom cache with dense SDPA for patch-parity tests."""

        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("Dense reference expects input_ids with shape [1, T].")
        prompt_tokens = int(input_ids.shape[1])
        self.clear()
        self.cache_capacity = prompt_tokens + max_new_tokens
        self.mode = "dense"
        generated: list[int] = []
        with self.patched():
            output = self.model(
                input_ids,
                attention_mask=torch.ones_like(input_ids),
                use_cache=False,
                logits_to_keep=1,
            )
            position = prompt_tokens
            for step in range(max_new_tokens):
                next_token = output.logits[0, -1].argmax().view(1, 1)
                generated.append(int(next_token.item()))
                if step + 1 == max_new_tokens:
                    break
                output = self.model(
                    next_token,
                    attention_mask=torch.ones_like(next_token),
                    position_ids=torch.full(
                        (1, 1),
                        position,
                        dtype=torch.long,
                        device=input_ids.device,
                    ),
                    use_cache=False,
                    logits_to_keep=1,
                )
                position += 1
        self.clear()
        return generated

    def cache_storage_bytes(self) -> int:
        return sum(
            cached.key.numel() * cached.key.element_size()
            + cached.value.numel() * cached.value.element_size()
            for cached in self.cache.values()
        )


def _patched_forward(
    attention: Any,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    attention_mask: torch.Tensor | None = None,
    *args: Any,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    del args, kwargs
    engine: QwenSparseAttentionEngine = attention._sparse_attention_engine
    return engine._attention_forward(
        attention,
        hidden_states,
        position_embeddings,
        attention_mask,
    )
