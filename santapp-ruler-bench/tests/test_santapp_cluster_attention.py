import math

import torch

from santapp_ruler.attention.santapp import ClusterSummary, SantaPlusEngine
from santapp_ruler.attention.teams import TeamSummary
from santapp_ruler.config import SantaPlusConfig


def test_whole_cluster_path_recovers_dense_attention_when_all_clusters_selected():
    engine = object.__new__(SantaPlusEngine)
    engine.sample_end = 4
    engine.head_dim = 2
    engine.mode = "gumbel_cluster"
    engine.config = SantaPlusConfig(
        mode="gumbel_cluster",
        group_size=2,
        samples_per_head=4,
    )
    engine.summaries = {
        (0, 0): ClusterSummary(
            members=torch.tensor([0, 2, 1, 3]),
            starts=torch.tensor([0, 2]),
            lengths_long=torch.tensor([2, 2]),
            lengths_float=torch.tensor([2.0, 2.0]),
            key_centroids=torch.tensor([[0.5, 0.0], [0.0, 0.5]]),
        )
    }
    query = torch.tensor([0.7, -0.2])
    keys = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [-0.5, 0.2], [0.2, -0.7]]
    )
    values = torch.tensor(
        [[1.0, 2.0], [2.0, -1.0], [0.5, 4.0], [-2.0, 0.3], [3.0, 1.0]]
    )

    estimate = engine._approximate_attention(query, keys, values, 0, 0)
    scores = keys @ query / math.sqrt(2.0)
    expected = torch.softmax(scores, dim=0) @ values
    torch.testing.assert_close(estimate.output, expected)
    assert estimate.selected_clusters == 2
    assert set(estimate.sampled_indices.tolist()) == {0, 1, 2, 3}
    assert torch.equal(estimate.inclusion_probabilities, torch.ones(2))


def test_whole_team_path_recovers_dense_attention_with_exact_generated_suffix():
    engine = object.__new__(SantaPlusEngine)
    engine.sample_end = 4
    engine.head_dim = 2
    engine.mode = "team_gumbel_topk"
    engine.config = SantaPlusConfig(
        mode="team_gumbel_topk",
        parent_size=4,
        representatives_per_parent=2,
        samples_per_head=4,
    )
    leaders = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    engine.summaries = {
        (0, 0): TeamSummary(
            members=torch.tensor([0, 2, 1, 3]),
            starts=torch.tensor([0, 2]),
            lengths_long=torch.tensor([2, 2]),
            lengths_float=torch.tensor([2.0, 2.0]),
            leader_keys=leaders,
            leader_token_indices=torch.tensor([0, 1]),
            parent_ids=torch.tensor([0, 0]),
            active_parent_count=1,
        )
    }
    query = torch.tensor([0.7, -0.2])
    keys = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [-0.5, 0.2], [0.2, -0.7]]
    )
    values = torch.tensor(
        [[1.0, 2.0], [2.0, -1.0], [0.5, 4.0], [-2.0, 0.3], [3.0, 1.0]]
    )

    estimate = engine._approximate_attention(query, keys, values, 0, 0)
    expected = torch.softmax(keys @ query / math.sqrt(2.0), dim=0) @ values
    torch.testing.assert_close(estimate.output, expected)
    assert estimate.selected_teams == 2
    assert set(estimate.sampled_indices.tolist()) == {0, 1, 2, 3}
    assert torch.equal(estimate.inclusion_probabilities, torch.ones(2))


def test_team_sampler_returns_exactly_s_rows_with_multiplicity():
    engine = object.__new__(SantaPlusEngine)
    engine.sample_end = 2
    engine.head_dim = 1
    engine.mode = "team_sampler"
    engine.config = SantaPlusConfig(
        mode="team_sampler",
        parent_size=2,
        representatives_per_parent=1,
        samples_per_head=7,
    )
    engine.summaries = {
        (0, 0): TeamSummary(
            members=torch.tensor([0, 1]),
            starts=torch.tensor([0]),
            lengths_long=torch.tensor([2]),
            lengths_float=torch.tensor([2.0]),
            leader_keys=torch.tensor([[0.0]]),
            leader_token_indices=torch.tensor([0]),
            parent_ids=torch.tensor([0]),
            active_parent_count=1,
        )
    }
    estimate = engine._approximate_attention(
        torch.tensor([0.0]),
        torch.tensor([[0.0], [0.0]]),
        torch.tensor([[1.0], [2.0]]),
        0,
        0,
    )
    assert estimate.sampled_indices.numel() == 7
    assert torch.unique(estimate.sampled_indices).numel() <= 2
