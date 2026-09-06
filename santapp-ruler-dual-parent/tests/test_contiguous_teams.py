import torch
import pytest

from santapp_ruler.attention.contiguous_teams import (
    build_contiguous_team_summary,
    contiguous_parent_ids,
    contiguous_parent_spans,
    select_actual_key_representatives,
)


def _team_members(summary, team: int) -> torch.Tensor:
    start = int(summary.starts[team])
    length = int(summary.lengths_long[team])
    return summary.members[start : start + length]


def test_contiguous_parent_spans_with_tail() -> None:
    assert contiguous_parent_spans(38, 16) == ((0, 16), (16, 32), (32, 38))


def test_contiguous_parent_spans_exact_multiple() -> None:
    assert contiguous_parent_spans(32, 16) == ((0, 16), (16, 32))


def test_contiguous_parent_spans_validate_inputs() -> None:
    with pytest.raises(ValueError):
        contiguous_parent_spans(0, 16)
    with pytest.raises(ValueError):
        contiguous_parent_spans(8, 0)


def test_contiguous_parent_ids_are_positional() -> None:
    ids = contiguous_parent_ids(38, 16)
    assert ids.tolist() == [0] * 16 + [1] * 16 + [2] * 6


def test_first_leader_is_nearest_arithmetic_mean() -> None:
    keys = torch.tensor([[0.0], [1.0], [2.0], [10.0]])
    leaders, _ = select_actual_key_representatives(keys, 1)
    # Mean is 3.25; key 2 is nearest.
    assert leaders.tolist() == [2]


def test_second_leader_is_farthest_from_first() -> None:
    keys = torch.tensor([[0.0], [1.0], [2.0], [10.0]])
    leaders, _ = select_actual_key_representatives(keys, 2)
    assert leaders.tolist() == [2, 3]


def test_farthest_first_uses_distance_to_nearest_selected_leader() -> None:
    keys = torch.tensor([[0.0], [2.0], [4.0], [10.0], [11.0]])
    leaders, _ = select_actual_key_representatives(keys, 3)
    assert leaders.tolist() == [2, 4, 0]


def test_duplicate_keys_still_produce_nonempty_teams() -> None:
    keys = torch.zeros(4, 2)
    leaders, assignment = select_actual_key_representatives(keys, 4)
    assert sorted(leaders.tolist()) == [0, 1, 2, 3]
    assert torch.bincount(assignment, minlength=4).tolist() == [1, 1, 1, 1]


def test_summary_tail_is_sampled_parent_not_exact_window() -> None:
    torch.manual_seed(0)
    summary = build_contiguous_team_summary(
        torch.randn(38, 8), parent_size=16, representatives_per_parent=4
    )
    assert summary.parent_lengths_long.tolist() == [16, 16, 6]
    assert summary.active_parent_count == 3
    assert summary.partial_parent_size == 6
    assert summary.parent_ids.tolist() == [0] * 4 + [1] * 4 + [2] * 4
    assert sorted(summary.members.tolist()) == list(range(38))
    tail_members = torch.cat(
        [_team_members(summary, team) for team in range(8, 12)]
    )
    assert sorted(tail_members.tolist()) == list(range(32, 38))


def test_summary_exact_multiple_has_no_partial_parent() -> None:
    summary = build_contiguous_team_summary(
        torch.randn(32, 4), parent_size=16, representatives_per_parent=4
    )
    assert summary.parent_lengths_long.tolist() == [16, 16]
    assert summary.partial_parent_size == 0
    assert summary.num_teams == 8


def test_tail_shorter_than_representative_count_uses_one_team_per_token() -> None:
    summary = build_contiguous_team_summary(
        torch.randn(18, 4), parent_size=16, representatives_per_parent=4
    )
    assert summary.parent_lengths_long.tolist() == [16, 2]
    assert summary.num_teams == 6
    assert summary.lengths_long[-2:].tolist() == [1, 1]
    assert sorted(summary.members[-2:].tolist()) == [16, 17]


def test_every_team_stays_inside_its_positional_parent() -> None:
    summary = build_contiguous_team_summary(
        torch.randn(51, 7), parent_size=16, representatives_per_parent=4
    )
    for team, parent_id in enumerate(summary.parent_ids.tolist()):
        members = _team_members(summary, team)
        assert bool((members // 16 == parent_id).all())


def test_leader_key_matches_actual_key_row() -> None:
    keys = torch.randn(38, 5)
    summary = build_contiguous_team_summary(
        keys, parent_size=16, representatives_per_parent=4
    )
    assert torch.equal(
        summary.leader_keys,
        keys[summary.leader_token_indices].float(),
    )


def test_all_teams_are_nonempty() -> None:
    summary = build_contiguous_team_summary(
        torch.randn(101, 3), parent_size=16, representatives_per_parent=4
    )
    assert bool((summary.lengths_long > 0).all())
    assert int(summary.lengths_long.sum()) == 101
