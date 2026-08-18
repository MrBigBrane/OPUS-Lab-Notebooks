"""Sampling helpers for the reference attention implementations.

The probabilistic core is kept free of Hugging Face dependencies so it can be
unit-tested on CPU. Whole-cluster SANTA++ uses Gumbel top-k sampling without
replacement. The observed leave-one-out threshold supplies the conditional
inclusion probability used by the Horvitz--Thompson correction.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class GumbelTopKSample:
    selected_indices: torch.Tensor
    log_inclusion_probabilities: torch.Tensor
    threshold: torch.Tensor | None
    log_leave_one_out_inclusion_probabilities: torch.Tensor


@dataclass(frozen=True, slots=True)
class TeamIIDSample:
    """One-token-per-team IID proposal, retaining sampled-team identity."""

    sampled_team_indices: torch.Tensor
    sampled_token_indices: torch.Tensor
    log_token_proposal_probabilities: torch.Tensor
    team_probabilities: torch.Tensor


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


def iid_inclusion_probabilities(
    per_draw_probabilities: torch.Tensor,
    draws: int,
) -> torch.Tensor:
    """Return the probability each item appears at least once in iid draws."""

    if draws <= 0:
        raise ValueError(f"draws must be positive, got {draws}.")
    probabilities = per_draw_probabilities.float()
    if not bool(torch.isfinite(probabilities).all()):
        raise ValueError("Per-draw probabilities must be finite.")
    if bool(((probabilities < 0.0) | (probabilities > 1.0)).any()):
        raise ValueError("Per-draw probabilities must lie in [0, 1].")
    return -torch.expm1(draws * torch.log1p(-probabilities))


def gumbel_topk_without_replacement(
    logits: torch.Tensor,
    k: int,
    *,
    generator: torch.Generator | None = None,
) -> GumbelTopKSample:
    """Draw ``k`` distinct weighted items and return HT correction terms.

    For a selected item, the leave-one-out threshold is the observed
    ``(k+1)``-st global priority. Conditional on all other priorities, the
    item's inclusion probability is ``P(phi_i + G_i > threshold)``. Dividing
    the selected contribution by that probability yields an unbiased estimate
    of a population sum. The corrected sum must not be divided by ``k``.
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
    )
    uniform.clamp_min_(torch.finfo(uniform.dtype).tiny)
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

    # For selected i, the k-th largest priority among all other items is the
    # global (k+1)-st priority. For unselected i, it is the global k-th.
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


def sample_team_members_iid(
    team_logits: torch.Tensor,
    members: torch.Tensor,
    starts: torch.Tensor,
    lengths: torch.Tensor,
    samples: int,
    *,
    generator: torch.Generator | None = None,
) -> TeamIIDSample:
    """Sample teams with replacement, then one uniform member from each team."""

    if team_logits.ndim != 1:
        raise ValueError("team_logits must be one-dimensional.")
    if samples <= 0:
        raise ValueError("samples must be positive.")
    if not (
        starts.ndim == lengths.ndim == 1
        and starts.numel() == lengths.numel() == team_logits.numel()
    ):
        raise ValueError("starts and lengths must contain one entry per team.")
    lengths_long = lengths.long()
    if bool((lengths_long <= 0).any()):
        raise ValueError("Every team must be non-empty.")

    probabilities = torch.softmax(team_logits.float(), dim=0)
    sampled_teams = torch.multinomial(
        probabilities,
        samples,
        replacement=True,
        generator=generator,
    )
    sampled_lengths = lengths_long[sampled_teams]
    within_team = torch.floor(
        torch.rand(
            samples,
            dtype=torch.float32,
            device=team_logits.device,
            generator=generator,
        )
        * sampled_lengths.float()
    ).long()
    packed_positions = starts.long()[sampled_teams] + within_team
    sampled_tokens = members[packed_positions]
    log_proposal = (
        torch.log(probabilities[sampled_teams])
        - torch.log(sampled_lengths.float())
    )
    return TeamIIDSample(
        sampled_team_indices=sampled_teams,
        sampled_token_indices=sampled_tokens,
        log_token_proposal_probabilities=log_proposal,
        team_probabilities=probabilities,
    )


def expand_selected_packed_members(
    members: torch.Tensor,
    starts: torch.Tensor,
    lengths: torch.Tensor,
    selected_groups: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand packed group members without synchronizing once per group.

    Returns token indices and a parallel vector identifying which selected
    group produced each token.
    """

    if selected_groups.ndim != 1:
        raise ValueError("selected_groups must be one-dimensional.")
    if selected_groups.numel() == 0:
        empty = torch.empty(0, dtype=torch.long, device=members.device)
        return empty, empty

    chosen_lengths = lengths[selected_groups].long()
    if bool((chosen_lengths <= 0).any()):
        raise ValueError("Selected groups must be non-empty.")
    chosen_starts = starts[selected_groups].long()
    slots = torch.arange(
        selected_groups.numel(), dtype=torch.long, device=members.device
    )
    repeated_slots = torch.repeat_interleave(slots, chosen_lengths)
    prefix = torch.cumsum(chosen_lengths, dim=0) - chosen_lengths
    repeated_prefix = torch.repeat_interleave(prefix, chosen_lengths)
    total = int(chosen_lengths.sum().item())
    offsets = torch.arange(total, device=members.device) - repeated_prefix
    packed_positions = chosen_starts[repeated_slots] + offsets
    return members[packed_positions], repeated_slots


# Backwards-compatible public name retained for existing cluster code/tests.
expand_selected_cluster_members = expand_selected_packed_members


def largest_cluster_token_sum(lengths: torch.Tensor, cluster_count: int) -> int:
    """Return the maximum rows obtainable by selecting distinct clusters."""

    if cluster_count <= 0 or lengths.numel() == 0:
        return 0
    count = min(cluster_count, int(lengths.numel()))
    return int(torch.topk(lengths.long(), k=count).values.sum().item())
