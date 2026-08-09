"""Theoretical decode-time memory-access accounting.

The main convention is GQA-aware: dense K/V rows are counted once per KV head,
and sampled rows are unioned across the query heads that share that KV head.
A second, explicitly naive convention retains the older per-query-head denominator.
These are logical vector-equivalent estimates, not measured DRAM transactions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import torch

AccessPattern = Literal["sampled_kv", "full_k_sampled_v"]


@dataclass(frozen=True, slots=True)
class _GroupAccess:
    n_total: int
    exact_tokens: int
    centroid_key_vectors: int


@dataclass(slots=True)
class DecodeTrafficTracker:
    """Accumulate access counts over GQA groups without perturbing decode timing.

    Sample indices are retained during generation and unioned after timed decode.
    This avoids a synchronization or ``unique`` kernel in every attention call.
    """

    query_heads_per_kv: int
    samples_per_head: int
    access_pattern: AccessPattern
    head_calls: int = 0
    gqa_group_calls: int = 0
    dense_gqa_kv_vectors: int = 0
    dense_naive_kv_vectors: int = 0
    naive_kv_vectors_read: int = 0
    gqa_centroid_key_vectors_read: int = 0
    naive_centroid_key_vectors_read: int = 0
    sampled_token_draws: int = 0
    exact_token_reads_naive: int = 0
    total_tokens_summed_naive: int = 0
    _groups: list[_GroupAccess] = field(default_factory=list)
    _sampled_indices: list[torch.Tensor] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.query_heads_per_kv <= 0:
            raise ValueError("query_heads_per_kv must be positive.")
        if self.samples_per_head <= 0:
            raise ValueError("samples_per_head must be positive.")
        if self.access_pattern not in {"sampled_kv", "full_k_sampled_v"}:
            raise ValueError(f"Unknown access pattern: {self.access_pattern!r}")

    def record_group(
        self,
        *,
        n_total: int,
        sampled_indices_by_head: list[torch.Tensor],
        exact_tokens: int,
        centroid_key_vectors: int,
    ) -> None:
        """Record one layer/KV-head/decode-token GQA group.

        ``sampled_indices_by_head`` retains draw multiplicity for the naive metric.
        The GQA metric unions those indices across all sharing query heads later.
        """

        if n_total <= 0:
            raise ValueError("n_total must be positive.")
        if not 0 <= exact_tokens <= n_total:
            raise ValueError("exact_tokens must be between zero and n_total.")
        if centroid_key_vectors < 0:
            raise ValueError("centroid_key_vectors cannot be negative.")
        if len(sampled_indices_by_head) != self.query_heads_per_kv:
            raise ValueError(
                "Expected one sampled-index tensor per query head sharing the KV head: "
                f"{len(sampled_indices_by_head)} != {self.query_heads_per_kv}."
            )
        for indices in sampled_indices_by_head:
            if int(indices.numel()) != self.samples_per_head:
                raise ValueError(
                    "Every query head must contribute exactly samples_per_head draws: "
                    f"{int(indices.numel())} != {self.samples_per_head}."
                )
            self._sampled_indices.append(indices.detach().reshape(-1))

        heads = self.query_heads_per_kv
        samples = self.samples_per_head
        self.head_calls += heads
        self.gqa_group_calls += 1
        self.dense_gqa_kv_vectors += 2 * n_total
        self.dense_naive_kv_vectors += 2 * n_total * heads
        self.gqa_centroid_key_vectors_read += centroid_key_vectors
        self.naive_centroid_key_vectors_read += centroid_key_vectors * heads
        self.sampled_token_draws += samples * heads
        self.exact_token_reads_naive += exact_tokens * heads
        self.total_tokens_summed_naive += n_total * heads

        if self.access_pattern == "sampled_kv":
            # Each draw/exact token requires both K and V in the per-query-head
            # convention. Draw duplicates are intentionally retained here.
            self.naive_kv_vectors_read += 2 * (samples + exact_tokens) * heads
        else:
            if exact_tokens:
                raise ValueError(
                    "full_k_sampled_v accounting does not use an exact-token window."
                )
            # SANTA scores every K and gathers one V per sampled draw.
            self.naive_kv_vectors_read += (n_total + samples) * heads

        self._groups.append(
            _GroupAccess(
                n_total=n_total,
                exact_tokens=exact_tokens,
                centroid_key_vectors=centroid_key_vectors,
            )
        )

    def _unique_sample_counts(self) -> np.ndarray:
        if not self._groups:
            return np.empty(0, dtype=np.int64)
        expected_tensors = self.gqa_group_calls * self.query_heads_per_kv
        if len(self._sampled_indices) != expected_tensors:
            raise RuntimeError(
                "Traffic tracker sampled-index log is incomplete: "
                f"{len(self._sampled_indices)} != {expected_tensors}."
            )

        # One device concatenation and one host transfer occur after timed decode.
        flattened = torch.cat(self._sampled_indices, dim=0).cpu().numpy()
        rows = flattened.reshape(
            self.gqa_group_calls,
            self.query_heads_per_kv * self.samples_per_head,
        )
        rows.sort(axis=1)
        if rows.shape[1] == 1:
            return np.ones(rows.shape[0], dtype=np.int64)
        return 1 + np.count_nonzero(rows[:, 1:] != rows[:, :-1], axis=1)

    @staticmethod
    def _pct(numerator: int, denominator: int) -> float:
        return 100.0 * numerator / denominator if denominator else 0.0

    def as_dict(self) -> dict[str, float | int]:
        unique_samples = self._unique_sample_counts()
        if len(unique_samples) != len(self._groups):
            raise RuntimeError("Traffic tracker group/union lengths do not match.")

        gqa_kv_vectors_read = 0
        for group, unique_count in zip(self._groups, unique_samples, strict=True):
            unique = int(unique_count)
            if self.access_pattern == "sampled_kv":
                # Sampled prefix and exact suffix are disjoint. K and V are read
                # once for each row in their GQA-group union.
                gqa_kv_vectors_read += 2 * (unique + group.exact_tokens)
            else:
                # Every K row is scanned once for the GQA group; sampled V rows
                # are unioned across sharing query heads.
                gqa_kv_vectors_read += group.n_total + unique

        gqa_centroids = self.gqa_centroid_key_vectors_read
        naive_centroids = self.naive_centroid_key_vectors_read
        dense_gqa = self.dense_gqa_kv_vectors
        dense_naive = self.dense_naive_kv_vectors
        gqa_total = gqa_kv_vectors_read + gqa_centroids
        naive_total = self.naive_kv_vectors_read + naive_centroids
        calls = self.head_calls
        groups = self.gqa_group_calls

        return {
            "decode_attention_head_calls": calls,
            "decode_gqa_group_calls": groups,
            "decode_dense_gqa_kv_vectors": dense_gqa,
            "decode_dense_naive_kv_vectors": dense_naive,
            "decode_gqa_kv_vectors_read": gqa_kv_vectors_read,
            "decode_naive_kv_vectors_read": self.naive_kv_vectors_read,
            "decode_gqa_centroid_key_vectors_read": gqa_centroids,
            "decode_naive_centroid_key_vectors_read": naive_centroids,
            "decode_gqa_total_vectors_read": gqa_total,
            "decode_naive_total_vectors_read": naive_total,
            "decode_gqa_kv_access_pct": self._pct(gqa_kv_vectors_read, dense_gqa),
            "decode_gqa_centroid_access_pct": self._pct(gqa_centroids, dense_gqa),
            "decode_gqa_total_access_pct": self._pct(gqa_total, dense_gqa),
            "decode_naive_kv_access_pct": self._pct(
                self.naive_kv_vectors_read, dense_naive
            ),
            "decode_naive_centroid_access_pct": self._pct(
                naive_centroids, dense_naive
            ),
            "decode_naive_total_access_pct": self._pct(naive_total, dense_naive),
            "mean_sampled_token_draws_per_head_call": (
                self.sampled_token_draws / calls if calls else 0.0
            ),
            "mean_unique_sampled_tokens_per_gqa_group_call": (
                float(unique_samples.mean()) if groups else 0.0
            ),
            "mean_exact_tokens_per_gqa_group_call": (
                sum(group.exact_tokens for group in self._groups) / groups
                if groups
                else 0.0
            ),
            "mean_total_tokens_per_head_call": (
                self.total_tokens_summed_naive / calls if calls else 0.0
            ),
        }
