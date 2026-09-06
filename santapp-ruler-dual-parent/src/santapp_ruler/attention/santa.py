"""Reference SANTA attention for Qwen2/Qwen2.5."""

from __future__ import annotations

import math
import time
from typing import Any

import torch

from ..config import SantaConfig
from .qwen import (
    AttentionEstimate,
    AttentionGeneration,
    QwenSparseAttentionEngine,
    cuda_sync,
    peak_memory_gib,
    reset_peak_memory,
    seed_torch,
)
from .traffic import DecodeTrafficTracker


class SantaEngine(QwenSparseAttentionEngine):
    """Score every key, sample values IID from the exact attention distribution."""

    def __init__(self, model: Any, config: SantaConfig):
        super().__init__(model, config)
        self.config = config
        self.traffic = self._make_traffic_tracker()

    def _make_traffic_tracker(self) -> DecodeTrafficTracker:
        return DecodeTrafficTracker(
            query_heads_per_kv=self.query_heads_per_kv,
            samples_per_head=self.config.samples_per_head,
            access_pattern="full_k_sampled_v",
        )

    def clear(self) -> None:
        super().clear()
        self.traffic = self._make_traffic_tracker()

    def _record_decode_group(
        self,
        *,
        layer_id: int,
        kv_head: int,
        n_total: int,
        sampled_indices_by_head: list[torch.Tensor],
    ) -> None:
        del layer_id, kv_head
        self.traffic.record_group(
            n_total=n_total,
            sampled_indices_by_head=sampled_indices_by_head,
            exact_tokens=0,
            routing_key_vectors=0,
            centroid_key_vectors=0,
        )

    def _approximate_attention(
        self,
        query: torch.Tensor,
        full_key: torch.Tensor,
        full_value: torch.Tensor,
        layer_id: int,
        kv_head: int,
    ) -> AttentionEstimate:
        del layer_id, kv_head
        scores = full_key.float() @ query * (1.0 / math.sqrt(self.head_dim))
        probability = torch.softmax(scores, dim=0)
        sampled_indices = torch.multinomial(
            probability,
            self.config.samples_per_head,
            replacement=True,
        )
        output = full_value[sampled_indices].float().mean(dim=0)
        return AttentionEstimate(output=output, sampled_indices=sampled_indices)

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int,
        eos_token_ids: set[int],
        stop_on_eos: bool,
        random_seed: int,
    ) -> AttentionGeneration:
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("SANTA generation expects input_ids with shape [1, T].")
        prompt_tokens = int(input_ids.shape[1])
        if prompt_tokens < 2:
            raise ValueError("SANTA requires at least two prompt tokens.")

        self.clear()
        self.cache_capacity = prompt_tokens + max_new_tokens
        seed_torch(random_seed, input_ids.device)
        reset_peak_memory(input_ids.device)

        cuda_sync()
        total_start = time.perf_counter()
        with self.patched():
            self.mode = "dense"
            cuda_sync()
            prefill_start = time.perf_counter()
            _ = self.model(
                input_ids,
                attention_mask=torch.ones_like(input_ids),
                use_cache=False,
                logits_to_keep=1,
            )
            cuda_sync()
            prefill_seconds = time.perf_counter() - prefill_start

            # Re-feed the final prompt token so sparse decoding starts at the same
            # next-token position as stock cached generation.
            self._trim_cache(prompt_tokens - 1)
            self.mode = "santa"
            self.traffic = self._make_traffic_tracker()
            self._reset_sampling_diagnostics()
            seed_torch(random_seed, input_ids.device)

            generated: list[int] = []
            current = input_ids[:, -1:]
            position = prompt_tokens - 1
            cuda_sync()
            decode_start = time.perf_counter()
            for _ in range(max_new_tokens):
                output = self.model(
                    current,
                    position_ids=torch.tensor(
                        [[position]], dtype=torch.long, device=input_ids.device
                    ),
                    use_cache=False,
                    logits_to_keep=1,
                )
                current = output.logits[0, -1].argmax().view(1, 1)
                token_id = int(current.item())
                generated.append(token_id)
                position += 1
                if stop_on_eos and token_id in eos_token_ids:
                    break
            cuda_sync()
            decode_seconds = time.perf_counter() - decode_start
            cache_bytes = self.cache_storage_bytes()

        cuda_sync()
        total_seconds = time.perf_counter() - total_start
        peak_allocated_gib, peak_reserved_gib = peak_memory_gib(input_ids.device)
        metrics: dict[str, Any] = {
            "backend": "santa",
            "mode": "santa",
            "sampling_scheme": "token_iid_with_replacement_from_exact_attention",
            "sampling_unit": "token",
            "importance_correction": "not_required_exact_proposal",
            "routing_key_type": "none",
            "prompt_tokens": prompt_tokens,
            "generated_tokens": len(generated),
            "max_new_tokens": max_new_tokens,
            "prefill_seconds": prefill_seconds,
            "routing_setup_seconds": 0.0,
            "decode_seconds": decode_seconds,
            "total_seconds": total_seconds,
            "samples_per_head": self.config.samples_per_head,
            "nominal_sample_budget_per_head": self.config.samples_per_head,
            "prompt_exact_tail_tokens": 0,
            "initial_generated_exact_tokens": 0,
            "exact_token_policy": "none",
            "parent_policy": "not_applicable",
            "probe_policy": "not_applicable",
            "custom_cache_gib": cache_bytes / (1024**3),
            "routing_summary_gib": 0.0,
            "peak_allocated_gib": peak_allocated_gib,
            "peak_reserved_gib": peak_reserved_gib,
            **self.traffic.as_dict(),
        }
        self.clear()
        return AttentionGeneration(token_ids=generated, metrics=metrics)
