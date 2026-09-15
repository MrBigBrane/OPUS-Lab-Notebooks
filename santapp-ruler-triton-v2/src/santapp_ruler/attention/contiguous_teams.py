"""Contiguous-parent, actual-key team construction.

Parent membership is purely positional: with parent size ``P``, parent ``j`` is
``[j*P, min((j+1)*P, N))``. The final parent is allowed to be shorter and is
handled by the same sampled-team machinery; it is never converted to an exact
attention window.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class TeamSummary:
    """Flat decode-time table for all teams in one layer/KV head."""

    members: torch.Tensor
    starts: torch.Tensor
    lengths_long: torch.Tensor
    lengths_float: torch.Tensor
    leader_keys: torch.Tensor
    leader_token_indices: torch.Tensor
    parent_ids: torch.Tensor
    parent_lengths_long: torch.Tensor
    active_parent_count: int
    partial_parent_size: int

    @property
    def num_groups(self) -> int:
        return int(self.lengths_long.numel())

    @property
    def num_teams(self) -> int:
        return self.num_groups


def contiguous_parent_spans(token_count: int, parent_size: int) -> tuple[tuple[int, int], ...]:
    """Return half-open positional parent spans covering ``[0, token_count)``."""

    if token_count <= 0:
        raise ValueError("token_count must be positive.")
    if parent_size <= 0:
        raise ValueError("parent_size must be positive.")
    return tuple(
        (start, min(start + parent_size, token_count))
        for start in range(0, token_count, parent_size)
    )


def contiguous_parent_ids(
    token_count: int,
    parent_size: int,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return the positional parent id for every token without clustering."""

    if token_count <= 0:
        raise ValueError("token_count must be positive.")
    if parent_size <= 0:
        raise ValueError("parent_size must be positive.")
    return torch.arange(token_count, device=device, dtype=torch.long) // parent_size


def select_actual_key_representatives(
    parent_keys: torch.Tensor,
    representatives_per_parent: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select actual keys by mean-nearest followed by farthest-first traversal.

    The first representative is the key nearest the arithmetic mean. Every
    later representative maximizes distance to its nearest already-selected
    representative. Tokens are then assigned to the nearest representative.
    Representative rows own themselves to resolve duplicate-key ties and keep
    all teams nonempty.
    """

    if parent_keys.ndim != 2:
        raise ValueError(
            f"parent_keys must have shape [tokens, dim], got {parent_keys.shape}."
        )
    member_count = int(parent_keys.shape[0])
    if member_count == 0:
        raise ValueError("Cannot select representatives from an empty parent.")
    if representatives_per_parent <= 0:
        raise ValueError("representatives_per_parent must be positive.")

    keys = parent_keys.float()
    requested = min(representatives_per_parent, member_count)
    first = (keys - keys.mean(dim=0)).square().sum(dim=1).argmin()
    chosen = torch.empty(requested, dtype=torch.long, device=keys.device)
    chosen[0] = first

    nearest_distance = (keys - keys[first]).square().sum(dim=1)
    selected = torch.zeros(member_count, dtype=torch.bool, device=keys.device)
    selected[first] = True
    for position in range(1, requested):
        next_index = nearest_distance.masked_fill(selected, -1.0).argmax()
        chosen[position] = next_index
        selected[next_index] = True
        distance_to_new = (keys - keys[next_index]).square().sum(dim=1)
        nearest_distance = torch.minimum(nearest_distance, distance_to_new)

    leaders = keys[chosen]
    distances = (
        keys.square().sum(dim=1, keepdim=True)
        + leaders.square().sum(dim=1).unsqueeze(0)
        - 2.0 * keys @ leaders.T
    ).clamp_min_(0.0)
    assignment = distances.argmin(dim=1)
    assignment[chosen] = torch.arange(requested, dtype=torch.long, device=keys.device)
    return chosen, assignment


def _build_full_parent_teams(
    full_keys: torch.Tensor,
    *,
    token_offset: int,
    parent_id_offset: int,
    representatives_per_parent: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Vectorized construction for equal-sized full parents.

    Returns packed members, team lengths, leader indices, parent ids, and leader
    keys. No k-means and no parent-membership sorting is performed.
    """

    parent_count, parent_size, head_dim = full_keys.shape
    if parent_count == 0:
        empty_long = torch.empty(0, dtype=torch.long, device=full_keys.device)
        empty_keys = torch.empty(0, head_dim, dtype=torch.float32, device=full_keys.device)
        return empty_long, empty_long, empty_long, empty_long, empty_keys

    keys = full_keys.float()
    reps = representatives_per_parent
    means = keys.mean(dim=1)
    first = (keys - means[:, None, :]).square().sum(dim=2).argmin(dim=1)
    chosen = torch.empty(parent_count, reps, dtype=torch.long, device=keys.device)
    chosen[:, 0] = first

    rows = torch.arange(parent_count, device=keys.device)
    first_keys = keys[rows, first]
    nearest_distance = (keys - first_keys[:, None, :]).square().sum(dim=2)
    selected = torch.zeros(
        parent_count, parent_size, dtype=torch.bool, device=keys.device
    )
    selected.scatter_(1, first[:, None], True)

    for position in range(1, reps):
        next_index = nearest_distance.masked_fill(selected, -1.0).argmax(dim=1)
        chosen[:, position] = next_index
        selected.scatter_(1, next_index[:, None], True)
        next_keys = keys[rows, next_index]
        distance_to_new = (keys - next_keys[:, None, :]).square().sum(dim=2)
        nearest_distance = torch.minimum(nearest_distance, distance_to_new)

    gather_index = chosen[:, :, None].expand(parent_count, reps, head_dim)
    leaders = keys.gather(1, gather_index)
    distances = (keys[:, :, None, :] - leaders[:, None, :, :]).square().sum(dim=3)
    assignment = distances.argmin(dim=2)
    assignment.scatter_(
        1,
        chosen,
        torch.arange(reps, device=keys.device, dtype=torch.long)[None, :].expand(
            parent_count, reps
        ),
    )

    # Boolean traversal is parent-major, team-major, token-major, so each team is
    # contiguous in ``packed_members`` without sorting tokens into parents.
    local_tokens = torch.arange(parent_size, device=keys.device, dtype=torch.long)
    global_tokens = (
        token_offset
        + torch.arange(parent_count, device=keys.device, dtype=torch.long)[:, None]
        * parent_size
        + local_tokens[None, :]
    )
    team_numbers = torch.arange(reps, device=keys.device, dtype=torch.long)
    membership = assignment[:, None, :] == team_numbers[None, :, None]
    expanded_tokens = global_tokens[:, None, :].expand(parent_count, reps, parent_size)
    packed_members = expanded_tokens[membership]
    lengths = membership.sum(dim=2).reshape(-1).long()
    if bool((lengths <= 0).any()):
        raise RuntimeError("Representative ownership failed to keep every team nonempty.")

    leader_indices = (
        token_offset
        + torch.arange(parent_count, device=keys.device, dtype=torch.long)[:, None]
        * parent_size
        + chosen
    ).reshape(-1)
    parent_ids = torch.arange(
        parent_id_offset,
        parent_id_offset + parent_count,
        dtype=torch.long,
        device=keys.device,
    ).repeat_interleave(reps)
    return packed_members, lengths, leader_indices, parent_ids, leaders.reshape(-1, head_dim)


def build_contiguous_team_summary(
    keys: torch.Tensor,
    *,
    parent_size: int,
    representatives_per_parent: int,
) -> TeamSummary:
    """Build flattened teams from fixed contiguous parent spans.

    A non-multiple tail is an ordinary shorter sampled parent. For example,
    ``N=38, P=16`` yields parents ``[0,16)``, ``[16,32)``, and ``[32,38)``.
    """

    if keys.ndim != 2:
        raise ValueError(f"keys must have shape [tokens, dim], got {keys.shape}.")
    token_count, head_dim = map(int, keys.shape)
    if token_count == 0:
        raise ValueError("Cannot build teams for an empty key prefix.")
    if parent_size <= 0:
        raise ValueError("parent_size must be positive.")
    if representatives_per_parent <= 0:
        raise ValueError("representatives_per_parent must be positive.")
    if representatives_per_parent > parent_size:
        raise ValueError("representatives_per_parent must be <= parent_size.")

    full_parent_count, remainder = divmod(token_count, parent_size)
    members_parts: list[torch.Tensor] = []
    lengths_parts: list[torch.Tensor] = []
    leader_index_parts: list[torch.Tensor] = []
    parent_id_parts: list[torch.Tensor] = []
    leader_key_parts: list[torch.Tensor] = []

    if full_parent_count:
        full_keys = keys[: full_parent_count * parent_size].reshape(
            full_parent_count, parent_size, head_dim
        )
        built = _build_full_parent_teams(
            full_keys,
            token_offset=0,
            parent_id_offset=0,
            representatives_per_parent=representatives_per_parent,
        )
        members_parts.append(built[0])
        lengths_parts.append(built[1])
        leader_index_parts.append(built[2])
        parent_id_parts.append(built[3])
        leader_key_parts.append(built[4])

    if remainder:
        tail_start = full_parent_count * parent_size
        local_leaders, assignment = select_actual_key_representatives(
            keys[tail_start:], representatives_per_parent
        )
        tail_team_count = int(local_leaders.numel())
        tail_tokens = torch.arange(
            tail_start, token_count, dtype=torch.long, device=keys.device
        )
        for team in range(tail_team_count):
            members_parts.append(tail_tokens[assignment == team])
        lengths_parts.append(
            torch.bincount(assignment, minlength=tail_team_count).long()
        )
        leader_index_parts.append(tail_start + local_leaders)
        parent_id_parts.append(
            torch.full(
                (tail_team_count,),
                full_parent_count,
                dtype=torch.long,
                device=keys.device,
            )
        )
        leader_key_parts.append(keys[tail_start + local_leaders].float())

    packed_members = torch.cat(members_parts)
    lengths_long = torch.cat(lengths_parts)
    leader_token_indices = torch.cat(leader_index_parts)
    parent_ids = torch.cat(parent_id_parts)
    leader_keys = torch.cat(leader_key_parts)
    starts = torch.cat(
        (
            torch.zeros(1, dtype=torch.long, device=keys.device),
            torch.cumsum(lengths_long[:-1], dim=0),
        )
    )

    # Parent/team construction is a partition, not a filter.
    expected = torch.arange(token_count, dtype=torch.long, device=keys.device)
    if not torch.equal(torch.sort(packed_members).values, expected):
        raise RuntimeError("Flattened teams do not partition every prompt token once.")

    parent_count = full_parent_count + int(remainder > 0)
    parent_lengths = torch.full(
        (parent_count,), parent_size, dtype=torch.long, device=keys.device
    )
    if remainder:
        parent_lengths[-1] = remainder

    return TeamSummary(
        members=packed_members,
        starts=starts,
        lengths_long=lengths_long,
        lengths_float=lengths_long.float(),
        leader_keys=leader_keys,
        leader_token_indices=leader_token_indices,
        parent_ids=parent_ids,
        parent_lengths_long=parent_lengths,
        active_parent_count=parent_count,
        partial_parent_size=remainder,
    )
