import pytest
import torch

from santapp_ruler.attention.teams import (
    build_team_summary,
    select_actual_key_representatives,
)


def _team_members(summary, team: int) -> torch.Tensor:
    start = int(summary.starts[team])
    length = int(summary.lengths_long[team])
    return summary.members[start : start + length]


def test_first_leader_is_nearest_arithmetic_mean() -> None:
    keys = torch.tensor([[0.0], [1.0], [2.0], [10.0]])
    leaders, _ = select_actual_key_representatives(keys, 1)
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


def test_kmeans_parent_labels_define_membership_not_position() -> None:
    keys = torch.arange(8, dtype=torch.float32).unsqueeze(1)
    labels = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    summary = build_team_summary(
        keys,
        labels,
        representatives_per_parent=2,
        parent_cluster_count=2,
    )
    assert summary.parent_lengths_long.tolist() == [4, 4]
    assert summary.parent_ids.tolist() == [0, 0, 1, 1]
    parent0 = torch.cat([_team_members(summary, 0), _team_members(summary, 1)])
    parent1 = torch.cat([_team_members(summary, 2), _team_members(summary, 3)])
    assert sorted(parent0.tolist()) == [0, 2, 4, 6]
    assert sorted(parent1.tolist()) == [1, 3, 5, 7]


def test_empty_kmeans_parent_is_skipped_but_recorded() -> None:
    keys = torch.randn(6, 3)
    labels = torch.tensor([0, 0, 2, 2, 2, 0])
    summary = build_team_summary(
        keys,
        labels,
        representatives_per_parent=2,
        parent_cluster_count=4,
    )
    assert summary.active_parent_count == 2
    assert summary.configured_parent_count == 4
    assert summary.parent_lengths_long.tolist() == [3, 3]
    assert set(summary.parent_ids.tolist()) == {0, 2}


def test_small_parent_uses_one_team_per_available_token() -> None:
    keys = torch.randn(5, 4)
    labels = torch.tensor([0, 0, 0, 0, 1])
    summary = build_team_summary(
        keys,
        labels,
        representatives_per_parent=4,
        parent_cluster_count=2,
    )
    parent1_teams = [i for i, parent in enumerate(summary.parent_ids.tolist()) if parent == 1]
    assert len(parent1_teams) == 1
    assert _team_members(summary, parent1_teams[0]).tolist() == [4]


def test_team_summary_partitions_every_prompt_row_once() -> None:
    torch.manual_seed(0)
    keys = torch.randn(51, 7)
    labels = torch.randint(0, 5, (51,))
    summary = build_team_summary(
        keys,
        labels,
        representatives_per_parent=4,
        parent_cluster_count=5,
    )
    assert sorted(summary.members.tolist()) == list(range(51))
    assert int(summary.lengths_long.sum()) == 51
    assert bool((summary.lengths_long > 0).all())


def test_leader_key_matches_actual_post_rope_key_row() -> None:
    keys = torch.randn(38, 5)
    labels = torch.arange(38) % 3
    summary = build_team_summary(
        keys,
        labels,
        representatives_per_parent=4,
        parent_cluster_count=3,
    )
    assert torch.equal(summary.leader_keys, keys[summary.leader_token_indices].float())


def test_summary_rejects_invalid_labels() -> None:
    keys = torch.randn(4, 2)
    with pytest.raises(ValueError, match="one label per key"):
        build_team_summary(keys, torch.tensor([0, 1]), representatives_per_parent=2)
    with pytest.raises(ValueError, match="negative"):
        build_team_summary(
            keys,
            torch.tensor([0, 0, 1, -1]),
            representatives_per_parent=2,
        )
