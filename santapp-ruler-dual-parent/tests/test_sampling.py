import torch
import pytest

from santapp_ruler.attention.sampling import (
    expand_selected_packed_members,
    gumbel_topk_without_replacement,
    log_gumbel_tail_probability,
)


def test_log_gumbel_tail_probability_is_finite_at_extremes() -> None:
    values = log_gumbel_tail_probability(torch.tensor([-100.0, 0.0, 100.0]))
    assert bool(torch.isfinite(values).all())
    assert values[0].item() == pytest.approx(-100.0)
    assert values[2].item() == pytest.approx(0.0)


def test_gumbel_topk_is_deterministic_with_generator() -> None:
    logits = torch.tensor([0.0, 1.0, 2.0, 3.0])
    g1 = torch.Generator().manual_seed(123)
    g2 = torch.Generator().manual_seed(123)
    a = gumbel_topk_without_replacement(logits, 2, generator=g1)
    b = gumbel_topk_without_replacement(logits, 2, generator=g2)
    assert torch.equal(a.selected_indices, b.selected_indices)
    assert len(set(a.selected_indices.tolist())) == 2


def test_selecting_all_items_has_unit_inclusion_probability() -> None:
    sample = gumbel_topk_without_replacement(torch.tensor([0.0, 1.0, 2.0]), 5)
    assert sorted(sample.selected_indices.tolist()) == [0, 1, 2]
    assert sample.threshold is None
    assert torch.equal(sample.log_inclusion_probabilities, torch.zeros(3))


def test_gumbel_topk_validates_inputs() -> None:
    with pytest.raises(ValueError):
        gumbel_topk_without_replacement(torch.empty(0), 1)
    with pytest.raises(ValueError):
        gumbel_topk_without_replacement(torch.ones(2, 2), 1)
    with pytest.raises(ValueError):
        gumbel_topk_without_replacement(torch.ones(2), 0)


def test_expand_selected_packed_members() -> None:
    members = torch.tensor([10, 11, 20, 30, 31, 32])
    starts = torch.tensor([0, 2, 3])
    lengths = torch.tensor([2, 1, 3])
    rows, slots = expand_selected_packed_members(
        members, starts, lengths, torch.tensor([2, 0])
    )
    assert rows.tolist() == [30, 31, 32, 10, 11]
    assert slots.tolist() == [0, 0, 0, 1, 1]


def test_expand_empty_selection() -> None:
    rows, slots = expand_selected_packed_members(
        torch.tensor([1]), torch.tensor([0]), torch.tensor([1]), torch.empty(0, dtype=torch.long)
    )
    assert rows.numel() == slots.numel() == 0
