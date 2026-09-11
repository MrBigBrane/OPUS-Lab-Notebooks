"""Hierarchical whole-team attention with fixed contiguous parent spans."""

from __future__ import annotations

import math
import time
from typing import Any

import torch

from ..config import HierarchicalConfig
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
from .contiguous_teams import TeamSummary, build_contiguous_team_summary
from .traffic import DecodeTrafficTracker


def generated_exact_tokens_at_decode_step(step: int) -> int:
    """Exact generated-token suffix before the zero-indexed decode step.

    Step 0 (the first sparse query) has no generated KV rows yet. Step 1 has one,
    step 2 has two, and so on. Prompt rows—including a short final parent—are not
    part of this exact suffix.
    """

    if step < 0:
        raise ValueError("step cannot be negative.")
    return step


class HierarchicalWholeTeamEngine(QwenSparseAttentionEngine):
    """Gumbel Top-K over teams; selected teams contribute all their members."""

    def __init__(self, model: Any, config: HierarchicalConfig):
        super().__init__(model, config)
        self.config = config
        self.summaries: dict[tuple[int, int], TeamSummary] = {}
        self.traffic = self._make_traffic_tracker()

    def _make_traffic_tracker(self) -> DecodeTrafficTracker:
        # Whole-team selection has a variable actual row count, so the nominal
        # budget must not be enforced as a fixed per-head row count here.
        return DecodeTrafficTracker(
            query_heads_per_kv=self.query_heads_per_kv,
            samples_per_head=None,
            access_pattern="sampled_kv",
        )

    def clear(self) -> None:
        super().clear()
        self.summaries.clear()
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
        team_logits = (
            torch.log(summary.lengths_float)
            + summary.leader_keys @ query * scale
        )
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
            sampled_scores
            - selection.log_inclusion_probabilities[selected_slot]
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
            inclusion_probabilities=torch.exp(
                selection.log_inclusion_probabilities
            ),
        )

    @torch.inference_mode()
    def _dense_prefill_and_build_teams(
        self,
        input_ids: torch.Tensor,
    ) -> tuple[float, float]:
        prompt_tokens = int(input_ids.shape[1])
        self.sample_end = prompt_tokens
        self.mode = "dense"
        team_builder = build_contiguous_team_summary
        if self.config.prefill_backend == "triton":
            from .prefill_runtime import require_triton_environment
            from .triton_prefill.prefill_teams import build_contiguous_team_summary_triton
            require_triton_environment(input_ids.device)
            team_builder = build_contiguous_team_summary_triton
        elif self.config.prefill_backend != "torch":
            raise ValueError("prefill_backend must be torch or triton")

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

        cuda_sync()
        build_start = time.perf_counter()
        for layer_id in range(self.num_layers):
            cached_key, _ = self.cache[layer_id].view()
            for kv_head in range(self.num_kv_heads):
                self.summaries[(layer_id, kv_head)] = (
                    team_builder(
                        cached_key[kv_head, :prompt_tokens].float(),
                        parent_size=self.config.parent_size,
                        representatives_per_parent=(
                            self.config.representatives_per_parent
                        ),
                    )
                )
        cuda_sync()
        build_seconds = time.perf_counter() - build_start
        return prefill_seconds, build_seconds

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
            raise RuntimeError("No hierarchical summaries were constructed.")
        team_lengths = torch.cat([summary.lengths_float for summary in summaries])
        parent_lengths = torch.cat(
            [summary.parent_lengths_long.float() for summary in summaries]
        )
        active_parents = sum(summary.active_parent_count for summary in summaries)
        partial_sizes = {summary.partial_parent_size for summary in summaries}
        if len(partial_sizes) != 1:
            raise RuntimeError("Layer/KV-head summaries disagree on the tail span size.")
        partial_size = partial_sizes.pop()
        return {
            "active_parent_count_total": active_parents,
            "mean_active_parents_per_layer_kv_head": (
                active_parents / len(summaries)
            ),
            "active_team_count_total": int(team_lengths.numel()),
            "mean_active_teams_per_layer_kv_head": (
                int(team_lengths.numel()) / len(summaries)
            ),
            "representative_selection": "mean_nearest_then_farthest_first",
            "team_assignment_space": "actual_post_rope_key_l2",
            "team_assignment_rope_applied": True,
            "partial_parent_size": partial_size,
            "partial_parent_present": partial_size > 0,
            "partial_parent_is_exact": False,
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
            raise ValueError(
                "Hierarchical generation expects input_ids with shape [1, T]."
            )
        prompt_tokens = int(input_ids.shape[1])
        if prompt_tokens < 2:
            raise ValueError("Hierarchical attention requires at least two prompt tokens.")

        self.clear()
        self.cache_capacity = prompt_tokens + max_new_tokens
        seed_torch(random_seed, input_ids.device)
        reset_peak_memory(input_ids.device)

        cuda_sync()
        total_start = time.perf_counter()
        with self.patched():
            prefill_seconds, parent_build_seconds = (
                self._dense_prefill_and_build_teams(input_ids)
            )
            # Re-feed the final prompt token as the first sparse query. Its KV row
            # remains in its positional parent summary, so exact prompt rows = 0.
            self._trim_cache(prompt_tokens - 1)
            self.mode = "hierarchical"
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
        parent_count = math.ceil(prompt_tokens / self.config.parent_size)
        metrics: dict[str, Any] = {
            "backend": "hierarchical",
            "prefill_backend": self.config.prefill_backend,
            "kmeans_backend": "not_used",
            "team_builder_backend": self.config.prefill_backend,
            "prefill_reference_replay": False,
            "mode": "hierarchical_whole_team",
            "sampling_scheme": "team_gumbel_topk_without_replacement_whole_team",
            "sampling_unit": "team",
            "importance_correction": "conditional_team_inclusion_probability",
            "routing_key_type": "actual_team_leader",
            "prompt_tokens": prompt_tokens,
            "generated_tokens": len(generated),
            "max_new_tokens": max_new_tokens,
            "prefill_seconds": prefill_seconds,
            "parent_build_seconds": parent_build_seconds,
            "clustering_seconds": 0.0,
            "decode_seconds": decode_seconds,
            "total_seconds": total_seconds,
            "sampled_prompt_tokens": prompt_tokens,
            "prompt_exact_tail_tokens": 0,
            "initial_generated_exact_tokens": 0,
            "exact_token_policy": "generated_suffix_only_grows_from_zero",
            "parent_policy": "fixed_contiguous_spans_in_original_token_order",
            "parent_membership_formula": (
                "parent_j=[j*P,min((j+1)*P,prompt_tokens))"
            ),
            "parent_clustering_used": False,
            "parent_sorting_used": False,
            "parent_size": self.config.parent_size,
            "parent_count_per_layer_kv_head": parent_count,
            "representatives_per_parent": self.config.representatives_per_parent,
            "samples_per_head": self.config.samples_per_head,
            "nominal_sample_budget_per_head": self.config.samples_per_head,
            "nominal_team_size": nominal_team_size,
            "requested_teams_per_head": requested_teams,
            "partial_parent_policy": "ordinary_short_sampled_parent",
            "probe_policy": "not_used",
            "team_routing_scope": "global_flattened_teams",
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
