import torch

from santapp_ruler.attention.sampling import (
    expand_selected_cluster_members,
    expand_selected_packed_members,
    gumbel_topk_without_replacement,
    iid_inclusion_probabilities,
    log_gumbel_tail_probability,
    sample_team_members_iid,
)


def test_gumbel_topk_is_distinct_and_probabilities_are_valid():
    generator = torch.Generator().manual_seed(123)
    result = gumbel_topk_without_replacement(
        torch.tensor([-2.0, -0.5, 0.0, 1.0, 2.0]),
        3,
        generator=generator,
    )
    assert result.selected_indices.numel() == 3
    assert torch.unique(result.selected_indices).numel() == 3
    probabilities = torch.exp(result.log_inclusion_probabilities)
    assert torch.all(probabilities > 0.0)
    assert torch.all(probabilities <= 1.0)
    assert result.threshold is not None


def test_selecting_every_cluster_has_unit_inclusion_and_recovers_dense_sum():
    logits = torch.tensor([0.3, -1.2, 2.1])
    result = gumbel_topk_without_replacement(
        logits, 3, generator=torch.Generator().manual_seed(9)
    )
    assert result.threshold is None
    assert torch.equal(result.log_inclusion_probabilities, torch.zeros(3))

    members = torch.tensor([2, 0, 3, 1])
    starts = torch.tensor([0, 2, 3])
    lengths = torch.tensor([2, 1, 1])
    token_indices, selected_slots = expand_selected_cluster_members(
        members, starts, lengths, result.selected_indices
    )
    assert set(token_indices.tolist()) == {0, 1, 2, 3}

    scores = torch.tensor([0.2, -0.4, 1.1, 0.6])
    values = torch.tensor([[1.0], [2.0], [4.0], [-1.0]])
    corrected_scores = (
        scores[token_indices] - result.log_inclusion_probabilities[selected_slots]
    )
    sampled_output = torch.softmax(corrected_scores, dim=0) @ values[token_indices]
    dense_output = torch.softmax(scores, dim=0) @ values
    torch.testing.assert_close(sampled_output, dense_output)


def test_threshold_conditioned_ht_sum_is_monte_carlo_unbiased():
    logits = torch.log(torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]))
    values = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    generator = torch.Generator().manual_seed(7)
    estimates = []
    for _ in range(3000):
        result = gumbel_topk_without_replacement(logits, 2, generator=generator)
        probabilities = torch.exp(result.log_inclusion_probabilities)
        estimates.append(
            float((values[result.selected_indices] / probabilities).sum().item())
        )
    assert abs(sum(estimates) / len(estimates) - float(values.sum())) < 0.5


def test_threshold_conditioned_ht_population_sum_at_team_level():
    team_logits = torch.log(torch.tensor([1.0, 3.0, 2.0, 5.0]))
    complete_team_contributions = torch.tensor([2.0, 7.0, 3.0, 11.0])
    generator = torch.Generator().manual_seed(17)
    estimates = []
    for _ in range(3000):
        result = gumbel_topk_without_replacement(
            team_logits,
            2,
            generator=generator,
        )
        inclusion = torch.exp(result.log_inclusion_probabilities)
        estimates.append(
            float(
                (
                    complete_team_contributions[result.selected_indices]
                    / inclusion
                ).sum()
            )
        )
    assert abs(sum(estimates) / len(estimates) - 23.0) < 0.7


def test_probability_helpers_are_stable_at_extremes():
    log_probabilities = log_gumbel_tail_probability(
        torch.tensor([-100.0, 0.0, 100.0])
    )
    assert torch.isfinite(log_probabilities).all()
    assert log_probabilities[0].item() == -100.0
    assert log_probabilities[2].item() == 0.0

    inclusion = iid_inclusion_probabilities(torch.tensor([0.0, 0.25, 1.0]), 4)
    torch.testing.assert_close(inclusion, torch.tensor([0.0, 1.0 - 0.75**4, 1.0]))


def test_team_iid_proposal_is_team_probability_divided_by_population():
    result = sample_team_members_iid(
        torch.tensor([0.0, 1.0]),
        members=torch.tensor([0, 1, 2, 3]),
        starts=torch.tensor([0, 1]),
        lengths=torch.tensor([1, 3]),
        samples=40,
        generator=torch.Generator().manual_seed(4),
    )
    expected = (
        torch.log(result.team_probabilities[result.sampled_team_indices])
        - torch.log(
            torch.tensor([1.0, 3.0])[result.sampled_team_indices]
        )
    )
    torch.testing.assert_close(result.log_token_proposal_probabilities, expected)
    assert result.sampled_token_indices.numel() == 40


def test_team_iid_retains_exact_sample_count_and_multiplicity():
    result = sample_team_members_iid(
        torch.tensor([0.0]),
        members=torch.tensor([7]),
        starts=torch.tensor([0]),
        lengths=torch.tensor([1]),
        samples=8,
        generator=torch.Generator().manual_seed(8),
    )
    assert result.sampled_team_indices.tolist() == [0] * 8
    assert result.sampled_token_indices.tolist() == [7] * 8


def test_generic_packed_expansion_matches_legacy_cluster_alias():
    args = (
        torch.tensor([4, 2, 1, 3]),
        torch.tensor([0, 2, 3]),
        torch.tensor([2, 1, 1]),
        torch.tensor([2, 0]),
    )
    generic = expand_selected_packed_members(*args)
    legacy = expand_selected_cluster_members(*args)
    assert all(torch.equal(left, right) for left, right in zip(generic, legacy))
