"""Hierarchical actual-key team construction for SANTA++.

Parent labels are supplied by the benchmark's existing probe-fingerprint
MiniBatchKMeans pipeline.  This module deliberately knows nothing about that
clustering step: once labels exist, representatives and memberships are built
only from RoPE-applied cached keys in key space.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class TeamSummary:
    """Compact decode-time table for all teams in one layer/KV head."""

    members: torch.Tensor
    starts: torch.Tensor
    lengths_long: torch.Tensor
    lengths_float: torch.Tensor
    leader_keys: torch.Tensor
    leader_token_indices: torch.Tensor
    parent_ids: torch.Tensor
    active_parent_count: int

    @property
    def num_groups(self) -> int:
        """Return the number of flattened teams (legacy summary compatibility)."""

        return int(self.lengths_long.numel())

    @property
    def num_teams(self) -> int:
        return self.num_groups


def select_actual_key_representatives(
    parent_keys: torch.Tensor,
    representatives_per_parent: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select actual keys and assign the parent by mean-nearest/farthest-first.

    The first representative is the cached key nearest the arithmetic mean.
    Each later representative maximizes its distance to the nearest selected
    key.  ``torch.argmin``/``argmax`` supply deterministic first-index tie
    behavior.  In the duplicate-key edge case, each selected representative is
    assigned its own row so no flattened team is empty; this only resolves an
    exact nearest-representative tie.
    """

    if parent_keys.ndim != 2:
        raise ValueError(f"parent_keys must have shape [tokens, dim], got {parent_keys.shape}.")
    member_count = int(parent_keys.shape[0])
    if member_count == 0:
        raise ValueError("Cannot select representatives from an empty parent.")
    if representatives_per_parent <= 0:
        raise ValueError("representatives_per_parent must be positive.")

    keys = parent_keys.float()
    requested = min(representatives_per_parent, member_count)
    mean = keys.mean(dim=0)
    first = (keys - mean).square().sum(dim=1).argmin()
    chosen = torch.empty(requested, dtype=torch.long, device=parent_keys.device)
    chosen[0] = first

    nearest_distance = (keys - keys[first]).square().sum(dim=1)
    selected_mask = torch.zeros(member_count, dtype=torch.bool, device=parent_keys.device)
    selected_mask[first] = True
    for position in range(1, requested):
        candidate_distance = nearest_distance.masked_fill(selected_mask, -1.0)
        next_position = candidate_distance.argmax()
        chosen[position] = next_position
        selected_mask[next_position] = True
        distance_to_new = (keys - keys[next_position]).square().sum(dim=1)
        nearest_distance = torch.minimum(nearest_distance, distance_to_new)

    leaders = keys[chosen]
    distances = (
        keys.square().sum(dim=1, keepdim=True)
        + leaders.square().sum(dim=1).unsqueeze(0)
        - 2.0 * keys @ leaders.T
    ).clamp_min_(0.0)
    assignment = distances.argmin(dim=1)

    # A later representative can duplicate an earlier key exactly.  Default
    # argmin would then give its own row to the earlier representative and
    # create an invalid empty team.  Owning its row is an equally-near,
    # deterministic tie resolution and guarantees every sampling unit is valid.
    assignment[chosen] = torch.arange(
        requested, dtype=torch.long, device=parent_keys.device
    )
    return chosen, assignment


def build_team_summary(
    keys: torch.Tensor,
    parent_labels: torch.Tensor,
    *,
    representatives_per_parent: int,
    parent_cluster_count: int | None = None,
) -> TeamSummary:
    """Split supplied parents in actual-key space and flatten all nonempty teams."""

    if keys.ndim != 2:
        raise ValueError(f"keys must have shape [tokens, dim], got {keys.shape}.")
    if parent_labels.ndim != 1 or int(parent_labels.numel()) != int(keys.shape[0]):
        raise ValueError("parent_labels must contain one label per key.")
    if int(keys.shape[0]) == 0:
        raise ValueError("Cannot build teams for an empty key prefix.")
    if representatives_per_parent <= 0:
        raise ValueError("representatives_per_parent must be positive.")

    labels = parent_labels.long()
    if bool((labels < 0).any()):
        raise ValueError("parent_labels cannot be negative.")
    inferred_count = int(labels.max().item()) + 1
    cluster_count = inferred_count if parent_cluster_count is None else parent_cluster_count
    if cluster_count < inferred_count:
        raise ValueError("parent_cluster_count does not cover every supplied label.")

    team_members: list[torch.Tensor] = []
    leader_indices: list[torch.Tensor] = []
    parent_ids: list[int] = []
    active_parents = 0
    for parent_id in range(cluster_count):
        parent_members = torch.nonzero(labels == parent_id, as_tuple=False).squeeze(1)
        if parent_members.numel() == 0:
            continue
        active_parents += 1
        local_leaders, assignment = select_actual_key_representatives(
            keys[parent_members], representatives_per_parent
        )
        global_leaders = parent_members[local_leaders]
        for local_team in range(int(global_leaders.numel())):
            members = parent_members[assignment == local_team]
            if members.numel() == 0:  # Defensive: representative ownership prevents this.
                raise RuntimeError("Actual-key construction produced an empty team.")
            team_members.append(members)
            leader_indices.append(global_leaders[local_team])
            parent_ids.append(parent_id)

    if not team_members:
        raise RuntimeError("No active team was constructed from a nonempty key prefix.")

    lengths_long = torch.tensor(
        [int(members.numel()) for members in team_members],
        dtype=torch.long,
        device=keys.device,
    )
    starts = torch.cat(
        (
            torch.zeros(1, dtype=torch.long, device=keys.device),
            torch.cumsum(lengths_long[:-1], dim=0),
        )
    )
    packed_members = torch.cat(team_members)
    leader_token_indices = torch.stack(leader_indices).long()

    # Construction is a partition, not a filter: every prefix row appears once.
    sorted_members = torch.sort(packed_members).values
    expected = torch.arange(int(keys.shape[0]), device=keys.device)
    if not torch.equal(sorted_members, expected):
        raise RuntimeError("Flattened teams do not partition every prefix token exactly once.")

    return TeamSummary(
        members=packed_members,
        starts=starts,
        lengths_long=lengths_long,
        lengths_float=lengths_long.float(),
        leader_keys=keys[leader_token_indices].float(),
        leader_token_indices=leader_token_indices,
        parent_ids=torch.tensor(parent_ids, dtype=torch.long, device=keys.device),
        active_parent_count=active_parents,
    )
