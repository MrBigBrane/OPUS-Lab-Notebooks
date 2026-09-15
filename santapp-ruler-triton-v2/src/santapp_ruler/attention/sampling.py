"""Sampling helpers shared by both SANTA++ whole-team variants."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class GumbelTopKSample:
    selected_indices: torch.Tensor
    log_inclusion_probabilities: torch.Tensor
    threshold: torch.Tensor | None
    log_leave_one_out_inclusion_probabilities: torch.Tensor


def log_gumbel_tail_probability(delta: torch.Tensor) -> torch.Tensor:
    """Compute ``log(1 - exp(-exp(delta)))`` stably in float32."""

    value = delta.float()
    middle_delta = value.clamp(min=-20.0, max=20.0)
    middle = torch.log(-torch.expm1(-torch.exp(middle_delta)))
    return torch.where(
        value < -20.0,
        value,
        torch.where(value > 20.0, torch.zeros_like(value), middle),
    )


def gumbel_topk_without_replacement(
    logits: torch.Tensor,
    k: int,
    *,
    generator: torch.Generator | None = None,
) -> GumbelTopKSample:
    """Select weighted distinct teams and return conditional HT terms.

    For a selected team, the observed global ``(k+1)``-st priority is its
    leave-one-out threshold. Conditional on all other priorities, the returned
    inclusion probability is the probability that this team's Gumbel priority
    exceeds that threshold. Whole-team contributions are divided by this
    probability and are not divided by ``k``.
    """

    if logits.ndim != 1:
        raise ValueError(f"Expected one-dimensional logits, got {logits.shape}.")
    item_count = int(logits.numel())
    if item_count == 0:
        raise ValueError("Cannot sample from an empty set.")
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}.")

    selected_count = min(k, item_count)
    logits32 = logits.float()
    if not bool(torch.isfinite(logits32).all()):
        raise ValueError("Gumbel top-k logits must be finite.")

    uniform = torch.rand(
        logits32.shape,
        dtype=torch.float32,
        device=logits32.device,
        generator=generator,
    ).clamp_min_(torch.finfo(torch.float32).tiny)
    priorities = logits32 - torch.log(-torch.log(uniform))

    if selected_count == item_count:
        selected = torch.topk(priorities, k=item_count, sorted=True).indices
        zeros = torch.zeros_like(logits32)
        return GumbelTopKSample(selected, zeros, None, zeros)

    top_values, top_indices = torch.topk(
        priorities,
        k=selected_count + 1,
        largest=True,
        sorted=True,
    )
    selected = top_indices[:selected_count]
    selected_threshold = top_values[selected_count]
    unselected_threshold = top_values[selected_count - 1]
    thresholds = torch.full_like(logits32, unselected_threshold)
    thresholds[selected] = selected_threshold
    all_log_probabilities = log_gumbel_tail_probability(logits32 - thresholds)
    return GumbelTopKSample(
        selected_indices=selected,
        log_inclusion_probabilities=all_log_probabilities[selected],
        threshold=selected_threshold,
        log_leave_one_out_inclusion_probabilities=all_log_probabilities,
    )


def expand_selected_packed_members(
    members: torch.Tensor,
    starts: torch.Tensor,
    lengths: torch.Tensor,
    selected_groups: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand selected packed teams into token rows and selected-team slots."""

    if selected_groups.ndim != 1:
        raise ValueError("selected_groups must be one-dimensional.")
    if selected_groups.numel() == 0:
        empty = torch.empty(0, dtype=torch.long, device=members.device)
        return empty, empty

    chosen_lengths = lengths[selected_groups].long()
    if bool((chosen_lengths <= 0).any()):
        raise ValueError("Selected groups must be non-empty.")
    chosen_starts = starts[selected_groups].long()
    slots = torch.arange(selected_groups.numel(), dtype=torch.long, device=members.device)
    repeated_slots = torch.repeat_interleave(slots, chosen_lengths)
    prefix = torch.cumsum(chosen_lengths, dim=0) - chosen_lengths
    repeated_prefix = torch.repeat_interleave(prefix, chosen_lengths)
    total = int(chosen_lengths.sum().item())
    offsets = torch.arange(total, device=members.device) - repeated_prefix
    packed_positions = chosen_starts[repeated_slots] + offsets
    return members[packed_positions], repeated_slots
