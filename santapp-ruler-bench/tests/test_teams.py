import torch

from santapp_ruler.attention.teams import (
    build_team_summary,
    select_actual_key_representatives,
)


def test_first_representative_is_actual_key_nearest_parent_mean():
    keys = torch.tensor([[0.0], [2.0], [3.0], [10.0]])
    representatives, _ = select_actual_key_representatives(keys, 3)
    assert representatives[0].item() == 2
    torch.testing.assert_close(keys[representatives[0]], torch.tensor([3.0]))


def test_later_representatives_are_distinct_farthest_first_max_min():
    keys = torch.tensor([[0.0], [2.0], [3.0], [10.0]])
    representatives, _ = select_actual_key_representatives(keys, 4)
    assert representatives.tolist() == [2, 3, 0, 1]
    assert torch.unique(representatives).numel() == 4


def test_small_parent_uses_at_most_its_member_count_representatives():
    keys = torch.tensor([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
    representatives, assignment = select_actual_key_representatives(keys, 4)
    assert representatives.numel() == 3
    assert set(assignment.tolist()) == {0, 1, 2}


def test_flattened_teams_partition_prefix_and_store_actual_member_keys():
    keys = torch.tensor(
        [[0.0], [1.0], [4.0], [10.0], [11.0], [15.0], [15.0]]
    )
    labels = torch.tensor([0, 0, 0, 1, 1, 1, 1])
    summary = build_team_summary(
        keys,
        labels,
        representatives_per_parent=4,
        parent_cluster_count=3,
    )

    assert summary.active_parent_count == 2
    assert torch.equal(
        torch.sort(summary.members).values,
        torch.arange(keys.shape[0]),
    )
    assert torch.all(summary.lengths_long > 0)
    assert summary.lengths_long.sum().item() == keys.shape[0]
    torch.testing.assert_close(
        summary.leader_keys,
        keys[summary.leader_token_indices].float(),
    )
    for team, leader_index in enumerate(summary.leader_token_indices):
        start = summary.starts[team]
        end = start + summary.lengths_long[team]
        assert int(leader_index) in summary.members[start:end].tolist()


def test_team_construction_uses_key_distances_after_parent_labels_are_supplied():
    # Labels deliberately mix distant regions.  Local representatives and
    # assignments must still be derived from these actual keys, not fingerprints.
    keys = torch.tensor([[0.0], [100.0], [49.0], [51.0], [2.0], [98.0]])
    labels = torch.tensor([0, 0, 1, 1, 0, 0])
    summary = build_team_summary(
        keys,
        labels,
        representatives_per_parent=2,
        parent_cluster_count=2,
    )
    parent_zero_leaders = summary.leader_token_indices[summary.parent_ids == 0]
    assert parent_zero_leaders.tolist() == [4, 1]
    assert set(summary.leader_token_indices[summary.parent_ids == 1].tolist()) == {2, 3}


def test_duplicate_keys_do_not_create_an_empty_team():
    keys = torch.zeros(4, 2)
    summary = build_team_summary(
        keys,
        torch.zeros(4, dtype=torch.long),
        representatives_per_parent=4,
        parent_cluster_count=1,
    )
    assert summary.lengths_long.tolist() == [1, 1, 1, 1]
    assert torch.unique(summary.leader_token_indices).numel() == 4
