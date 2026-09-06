"""SANTAPP hierarchical whole-team attention with post-RoPE key k-means.

This is the cleaned implementation of the former "hierarchical whole-team
sampling, raw post-RoPE key parents" path.  Dense prefill stores the actual
RoPE-applied key cache.  For every layer and KV head, global GPU
MiniBatchKMeans clusters all prompt keys directly in that unnormalized key
space.  Each parent is split into actual-key teams using mean-nearest followed
by farthest-first leaders.  Decode performs Gumbel Top-K without replacement
over the globally flattened team table and reads every member of selected
teams.

All prompt tokens remain sampled.  The first sparse decode query has no exact
KV suffix; only tokens generated after the prompt form an exact suffix that
grows by one row per decode step.
"""

from __future__ import annotations

import math
import time
from typing import Any

import torch

from ..config import SantappConfig
from .minibatch_kmeans import SklearnLikeTorchMiniBatchKMeans
from .qwen import (
    AttentionEstimate,
    AttentionGeneration,
    QwenSparseAttentionEngine,
    cuda_sync,
    peak_memory_gib,
    reset_peak_memory,
    seed_torch,
)
from .sampling import expand_selected_packed_members, gumbel_topk_without_replacement
from .teams import TeamSummary, build_team_summary
from .traffic import DecodeTrafficTracker


def generated_exact_tokens_at_decode_step(step: int) -> int:
    """Return exact generated-token KV rows before zero-indexed decode step."""

    if step < 0:
        raise ValueError("step cannot be negative.")
    return step


def parent_cluster_count(token_count: int, parent_size: int) -> int:
    """Match the former global parent-k-means cluster-count rule exactly."""

    if token_count < 2:
        raise ValueError("token_count must be at least two.")
    if parent_size <= 0:
        raise ValueError("parent_size must be positive.")
    return min(token_count, max(2, token_count // parent_size))


class SantappEngine(QwenSparseAttentionEngine):
    """Post-RoPE-key k-means parents with whole-team Gumbel Top-K sampling."""

    def __init__(self, model: Any, config: SantappConfig):
        super().__init__(model, config)
        self.config = config
        self.summaries: dict[tuple[int, int], TeamSummary] = {}
        self.parent_cluster_count = 0
        self.traffic = self._make_traffic_tracker()

    def _make_traffic_tracker(self) -> DecodeTrafficTracker:
        # Whole-team selection has a variable actual row count, so the nominal
        # token budget must not be enforced as a fixed row count.
        return DecodeTrafficTracker(
            query_heads_per_kv=self.query_heads_per_kv,
            samples_per_head=None,
            access_pattern="sampled_kv",
        )

    def clear(self) -> None:
        super().clear()
        self.summaries.clear()
        self.parent_cluster_count = 0
        self.traffic = self._make_traffic_tracker()

    def _record_decode_group(
        self,
        *,
        layer_id: int,
        kv_head: int,
        n_total: int,
        sampled_indices_by_head: list[torch.Tensor],
    ) -> None:
        exact_tokens = n_total - self.sample_end
        if exact_tokens < 0:
            raise RuntimeError(
                f"Cache length {n_total} is shorter than sampled prompt boundary "
                f"{self.sample_end}."
            )
        summary = self.summaries[(layer_id, kv_head)]
        self.traffic.record_group(
            n_total=n_total,
            sampled_indices_by_head=sampled_indices_by_head,
            exact_tokens=exact_tokens,
            routing_key_vectors=summary.num_teams,
            centroid_key_vectors=0,
        )

    def _exact_generated_suffix(
        self,
        query: torch.Tensor,
        full_key: torch.Tensor,
        full_value: torch.Tensor,
        scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if int(full_key.shape[0]) > self.sample_end:
            return (
                full_key[self.sample_end :].float() @ query * scale,
                full_value[self.sample_end :].float(),
            )
        return (
            torch.empty(0, dtype=torch.float32, device=query.device),
            torch.empty(0, self.head_dim, dtype=torch.float32, device=query.device),
        )

    @staticmethod
    def _combine_ht_and_exact(
        *,
        sampled_values: torch.Tensor,
        sampled_log_weights: torch.Tensor,
        exact_values: torch.Tensor,
        exact_scores: torch.Tensor,
    ) -> torch.Tensor:
        maximum = (
            torch.cat((sampled_log_weights, exact_scores)).max()
            if exact_scores.numel()
            else sampled_log_weights.max()
        )
        sampled_weights = torch.exp(sampled_log_weights - maximum)
        exact_weights = torch.exp(exact_scores - maximum)
        numerator = sampled_weights @ sampled_values
        if exact_scores.numel():
            numerator = numerator + exact_weights @ exact_values
        denominator = sampled_weights.sum() + exact_weights.sum()
        return numerator / denominator

    def _approximate_attention(
        self,
        query: torch.Tensor,
        full_key: torch.Tensor,
        full_value: torch.Tensor,
        layer_id: int,
        kv_head: int,
    ) -> AttentionEstimate:
        n_total = int(full_key.shape[0])
        if not 0 < self.sample_end <= n_total:
            raise RuntimeError(
                f"Invalid sampled prompt boundary {self.sample_end} for cache "
                f"length {n_total}."
            )
        summary = self.summaries[(layer_id, kv_head)]
        scale = 1.0 / math.sqrt(self.head_dim)
        exact_scores, exact_values = self._exact_generated_suffix(
            query, full_key, full_value, scale
        )

        nominal_team_size = (
            self.config.parent_size // self.config.representatives_per_parent
        )
        requested_teams = self.config.samples_per_head // nominal_team_size
        selected_team_count = min(requested_teams, summary.num_teams)
        team_logits = torch.log(summary.lengths_float) + summary.leader_keys @ query * scale
        selection = gumbel_topk_without_replacement(
            team_logits,
            selected_team_count,
        )
        sampled_indices, selected_slot = expand_selected_packed_members(
            summary.members,
            summary.starts,
            summary.lengths_long,
            selection.selected_indices,
        )
        sampled_scores = full_key[sampled_indices].float() @ query * scale
        sampled_log_weights = (
            sampled_scores - selection.log_inclusion_probabilities[selected_slot]
        )
        output = self._combine_ht_and_exact(
            sampled_values=full_value[sampled_indices].float(),
            sampled_log_weights=sampled_log_weights,
            exact_values=exact_values,
            exact_scores=exact_scores,
        )
        return AttentionEstimate(
            output=output,
            sampled_indices=sampled_indices,
            selected_teams=selected_team_count,
            inclusion_probabilities=torch.exp(selection.log_inclusion_probabilities),
        )

    @torch.inference_mode()
    def _dense_prefill_and_cluster(
        self,
        input_ids: torch.Tensor,
    ) -> tuple[float, float, int]:
        prompt_tokens = int(input_ids.shape[1])
        self.sample_end = prompt_tokens
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

        cluster_count = parent_cluster_count(prompt_tokens, self.config.parent_size)
        self.parent_cluster_count = cluster_count
        km = self.config.kmeans

        cuda_sync()
        clustering_start = time.perf_counter()
        for layer_id in range(self.num_layers):
            cached_key, _ = self.cache[layer_id].view()
            for kv_head in range(self.num_kv_heads):
                # _append_cache stores keys only after Qwen RoPE is applied.  Use
                # those raw float32 coordinates directly: no probe projection,
                # standardization, normalization, sorting, or cache permutation.
                post_rope_keys = cached_key[kv_head, :prompt_tokens].float().contiguous()
                labels = SklearnLikeTorchMiniBatchKMeans(
                    n_clusters=cluster_count,
                    batch_size=km.batch_size,
                    n_init=km.n_init,
                    max_iter=km.max_iter,
                    tol=km.tol,
                    max_no_improvement=km.max_no_improvement,
                    init_size=km.init_size,
                    reassignment_ratio=km.reassignment_ratio,
                    random_state=km.random_state,
                ).fit_predict(post_rope_keys)
                self.summaries[(layer_id, kv_head)] = build_team_summary(
                    post_rope_keys,
                    labels,
                    representatives_per_parent=self.config.representatives_per_parent,
                    parent_cluster_count=cluster_count,
                )
                del labels, post_rope_keys
        cuda_sync()
        clustering_seconds = time.perf_counter() - clustering_start
        return prefill_seconds, clustering_seconds, cluster_count

    def _summary_storage_bytes(self) -> int:
        total = 0
        for summary in self.summaries.values():
            for tensor in (
                summary.members,
                summary.starts,
                summary.lengths_long,
                summary.lengths_float,
                summary.leader_keys,
                summary.leader_token_indices,
                summary.parent_ids,
                summary.parent_lengths_long,
            ):
                total += tensor.numel() * tensor.element_size()
        return total

    @staticmethod
    def _size_statistics(
        all_lengths: torch.Tensor,
        *,
        prefix: str,
    ) -> dict[str, float | int]:
        if all_lengths.numel() == 0:
            raise RuntimeError(f"Cannot summarize an empty {prefix} table.")
        values = all_lengths.float()
        mean = float(values.mean().item())
        std = float(values.std(unbiased=False).item())
        return {
            f"{prefix}_size_mean": mean,
            f"{prefix}_size_std": std,
            f"{prefix}_size_cv": std / mean if mean else 0.0,
            f"{prefix}_size_min": int(values.min().item()),
            f"{prefix}_size_median": float(torch.quantile(values, 0.5).item()),
            f"{prefix}_size_p90": float(torch.quantile(values, 0.9).item()),
            f"{prefix}_size_max": int(values.max().item()),
        }

    def _summary_statistics(self) -> dict[str, float | int | bool | str]:
        summaries = list(self.summaries.values())
        if not summaries:
            raise RuntimeError("No SANTAPP team summaries were constructed.")
        team_lengths = torch.cat([summary.lengths_float for summary in summaries])
        parent_lengths = torch.cat(
            [summary.parent_lengths_long.float() for summary in summaries]
        )
        active_parents = sum(summary.active_parent_count for summary in summaries)
        configured_parents = sum(
            summary.configured_parent_count for summary in summaries
        )
        return {
            "active_parent_count_total": active_parents,
            "configured_parent_count_total": configured_parents,
            "empty_parent_count_total": configured_parents - active_parents,
            "mean_active_parents_per_layer_kv_head": active_parents / len(summaries),
            "active_team_count_total": int(team_lengths.numel()),
            "mean_active_teams_per_layer_kv_head": (
                int(team_lengths.numel()) / len(summaries)
            ),
            "representative_selection": "mean_nearest_then_farthest_first",
            "team_assignment_space": "actual_post_rope_key_l2",
            "team_assignment_rope_applied": True,
            **self._size_statistics(parent_lengths, prefix="parent"),
            **self._size_statistics(team_lengths, prefix="team"),
        }

    def _sampling_statistics(self) -> dict[str, float | int | None]:
        inclusion_mean: float | None = None
        inclusion_min: float | None = None
        inclusion_max: float | None = None
        inclusion_count = 0
        if self._selected_inclusion_probabilities:
            probabilities = torch.cat(self._selected_inclusion_probabilities).float().cpu()
            inclusion_count = int(probabilities.numel())
            inclusion_mean = float(probabilities.mean().item())
            inclusion_min = float(probabilities.min().item())
            inclusion_max = float(probabilities.max().item())
        return {
            "mean_selected_teams_per_head_call": (
                self._selected_teams_total / self._selected_team_head_calls
                if self._selected_team_head_calls
                else 0.0
            ),
            "selected_team_inclusion_probability_count": inclusion_count,
            "mean_selected_team_inclusion_probability": inclusion_mean,
            "min_selected_team_inclusion_probability": inclusion_min,
            "max_selected_team_inclusion_probability": inclusion_max,
        }

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
            raise ValueError("SANTAPP generation expects input_ids with shape [1, T].")
        prompt_tokens = int(input_ids.shape[1])
        if prompt_tokens < 3:
            raise ValueError("SANTAPP requires at least three prompt tokens.")

        self.clear()
        self.cache_capacity = prompt_tokens + max_new_tokens
        seed_torch(random_seed, input_ids.device)
        reset_peak_memory(input_ids.device)

        cuda_sync()
        total_start = time.perf_counter()
        with self.patched():
            prefill_seconds, clustering_seconds, cluster_count = (
                self._dense_prefill_and_cluster(input_ids)
            )
            # The parent summaries cover every prompt row. Re-feed the final
            # prompt token as the first sparse query so the first call has zero
            # exact rows while preserving the standard next-token position.
            self._trim_cache(prompt_tokens - 1)
            self.mode = "santapp"
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
            summary_bytes = self._summary_storage_bytes()
            summary_statistics = self._summary_statistics()

        cuda_sync()
        total_seconds = time.perf_counter() - total_start
        peak_allocated_gib, peak_reserved_gib = peak_memory_gib(input_ids.device)
        nominal_team_size = (
            self.config.parent_size // self.config.representatives_per_parent
        )
        requested_teams = self.config.samples_per_head // nominal_team_size
        metrics: dict[str, Any] = {
            "backend": "santapp",
            "mode": "hierarchical_whole_team_post_rope_key_kmeans",
            "sampling_scheme": "team_gumbel_topk_without_replacement_whole_team",
            "sampling_unit": "team",
            "importance_correction": "conditional_team_inclusion_probability",
            "routing_key_type": "actual_team_leader",
            "prompt_tokens": prompt_tokens,
            "generated_tokens": len(generated),
            "max_new_tokens": max_new_tokens,
            "prefill_seconds": prefill_seconds,
            # Compatibility name used by the current report table. It includes
            # both MiniBatchKMeans and team-summary construction.
            "parent_build_seconds": clustering_seconds,
            "clustering_seconds": clustering_seconds,
            "decode_seconds": decode_seconds,
            "total_seconds": total_seconds,
            "sampled_prompt_tokens": prompt_tokens,
            "prompt_exact_tail_tokens": 0,
            "initial_generated_exact_tokens": 0,
            "exact_token_policy": "generated_suffix_only_grows_from_zero",
            "parent_policy": "global_minibatch_kmeans_raw_post_rope_keys",
            "parent_clustering_used": True,
            "parent_clustering_scope": "global",
            "parent_clustering_representation": "post_rope_key_space",
            "parent_clustering_space": "post_rope_key_l2",
            "parent_clustering_feature_dim": self.head_dim,
            "parent_clustering_uses_probe_queries": False,
            "key_space_normalization": "none",
            "key_space_rope_applied": True,
            "key_space_stage": "post_rope",
            "parent_size": self.config.parent_size,
            "parent_size_semantics": "nominal_tokens_per_parent_for_cluster_count",
            "parent_cluster_count_per_layer_kv_head": cluster_count,
            "representatives_per_parent": self.config.representatives_per_parent,
            "samples_per_head": self.config.samples_per_head,
            "nominal_sample_budget_per_head": self.config.samples_per_head,
            "nominal_team_size": nominal_team_size,
            "requested_teams_per_head": requested_teams,
            "partial_parent_policy": "not_applicable_kmeans_parents_are_variable_size",
            "probe_policy": "not_used_for_parent_clustering",
            "team_routing_scope": "global_flattened_teams",
            "kmeans_batch_size": self.config.kmeans.batch_size,
            "kmeans_n_init": self.config.kmeans.n_init,
            "kmeans_max_iter": self.config.kmeans.max_iter,
            "kmeans_tol": self.config.kmeans.tol,
            "kmeans_max_no_improvement": self.config.kmeans.max_no_improvement,
            "kmeans_init_size": self.config.kmeans.init_size,
            "kmeans_reassignment_ratio": self.config.kmeans.reassignment_ratio,
            "kmeans_random_state": self.config.kmeans.random_state,
            "num_query_heads": self.num_query_heads,
            "num_kv_heads": self.num_kv_heads,
            "query_heads_per_kv": self.query_heads_per_kv,
            "head_dim": self.head_dim,
            "custom_cache_gib": cache_bytes / (1024**3),
            "routing_summary_gib": summary_bytes / (1024**3),
            "peak_allocated_gib": peak_allocated_gib,
            "peak_reserved_gib": peak_reserved_gib,
            **summary_statistics,
            **self._sampling_statistics(),
            **self.traffic.as_dict(),
        }
        self.clear()
        return AttentionGeneration(token_ids=generated, metrics=metrics)
