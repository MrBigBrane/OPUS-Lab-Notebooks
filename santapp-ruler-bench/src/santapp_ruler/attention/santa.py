"""Reference SANTA attention backend for Qwen2/Qwen2.5.

For every decode query head, SANTA computes the full ``qK^T`` score vector,
samples ``S`` token indices IID with replacement from the resulting softmax,
and returns the arithmetic mean of the gathered value vectors. There is no
fixed or growing exact-token window and no clustering path.
"""

from __future__ import annotations

import math
import time
from typing import Any

import torch

from ..config import SantaConfig
from .santapp import (
    AttentionGeneration,
    SantaPlusEngine,
    _cuda_sync,
    _peak_memory_gib,
    _reset_peak_memory,
    _seed_torch,
)
from .traffic import DecodeTrafficTracker


class SantaEngine(SantaPlusEngine):
    """Use the shared Qwen cache/patch machinery with the SANTA estimator."""

    def __init__(self, model: Any, config: SantaConfig):
        # SantaPlusEngine's common cache/patch code only requires a
        # samples_per_head attribute from the algorithm config.
        super().__init__(model, config)  # type: ignore[arg-type]
        self.config: SantaConfig = config

    def _make_traffic_tracker(self) -> DecodeTrafficTracker:
        return DecodeTrafficTracker(
            query_heads_per_kv=self.query_heads_per_kv,
            samples_per_head=self.config.samples_per_head,
            access_pattern="full_k_sampled_v",
        )

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
            centroid_key_vectors=0,
        )

    def _approximate_attention(
        self,
        query: torch.Tensor,
        full_key: torch.Tensor,
        full_value: torch.Tensor,
        layer_id: int,
        kv_head: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del layer_id, kv_head
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = full_key.float() @ query * scale
        probability = torch.softmax(scores, dim=0)
        sampled_indices = torch.multinomial(
            probability,
            self.config.samples_per_head,
            replacement=True,
        )
        # Because the proposal is the exact attention distribution, the
        # self-normalized estimator reduces to a simple mean of sampled values.
        output = full_value[sampled_indices].float().mean(dim=0)
        return output, sampled_indices

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
        _seed_torch(random_seed, input_ids.device)
        _reset_peak_memory(input_ids.device)

        _cuda_sync()
        total_start = time.perf_counter()
        with self.patched():
            self.mode = "dense"
            _cuda_sync()
            prefill_start = time.perf_counter()
            _ = self.model(
                input_ids,
                attention_mask=torch.ones_like(input_ids),
                use_cache=False,
                logits_to_keep=1,
            )
            _cuda_sync()
            prefill_seconds = time.perf_counter() - prefill_start

            # Re-feed the final prompt token so the first approximate call
            # produces the same next-token position as the stock cached path.
            self._trim_cache(prompt_tokens - 1)
            self.mode = "sparse"
            self.traffic = self._make_traffic_tracker()
            _seed_torch(random_seed, input_ids.device)

            generated: list[int] = []
            current = input_ids[:, -1:]
            position = prompt_tokens - 1
            _cuda_sync()
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
            _cuda_sync()
            decode_seconds = time.perf_counter() - decode_start
            cache_bytes, _ = self._storage_bytes()

        _cuda_sync()
        total_seconds = time.perf_counter() - total_start
        peak_allocated_gib, peak_reserved_gib = _peak_memory_gib(input_ids.device)

        traffic = self.traffic.as_dict()
        metrics: dict[str, Any] = {
            "backend": "santa",
            "prompt_tokens": prompt_tokens,
            "generated_tokens": len(generated),
            "max_new_tokens": max_new_tokens,
            "prefill_seconds": prefill_seconds,
            "clustering_seconds": 0.0,
            "decode_seconds": decode_seconds,
            "total_seconds": total_seconds,
            "samples_per_head": self.config.samples_per_head,
            "exact_window_tokens": 0,
            "custom_cache_gib": cache_bytes / (1024**3),
            "cluster_summary_gib": 0.0,
            "peak_allocated_gib": peak_allocated_gib,
            "peak_reserved_gib": peak_reserved_gib,
            **traffic,
        }
        self.clear()
        return AttentionGeneration(token_ids=generated, metrics=metrics)
