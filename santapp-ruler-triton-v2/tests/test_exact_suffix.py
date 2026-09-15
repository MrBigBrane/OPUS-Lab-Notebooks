import pytest
import torch

from santapp_ruler.attention.hierarchical import (
    HierarchicalWholeTeamEngine,
    generated_exact_tokens_at_decode_step as contiguous_exact_tokens,
)
from santapp_ruler.attention.santapp import (
    SantappEngine,
    generated_exact_tokens_at_decode_step as kmeans_exact_tokens,
)


@pytest.mark.parametrize("function", [kmeans_exact_tokens, contiguous_exact_tokens])
def test_generated_exact_suffix_starts_at_zero_and_grows(function) -> None:
    assert [function(i) for i in range(5)] == [0, 1, 2, 3, 4]


@pytest.mark.parametrize("function", [kmeans_exact_tokens, contiguous_exact_tokens])
def test_generated_exact_suffix_rejects_negative_step(function) -> None:
    with pytest.raises(ValueError, match="negative"):
        function(-1)


@pytest.mark.parametrize("engine_class", [SantappEngine, HierarchicalWholeTeamEngine])
def test_prompt_tail_is_not_returned_as_exact(engine_class) -> None:
    engine = object.__new__(engine_class)
    engine.sample_end = 38
    engine.head_dim = 3
    query = torch.ones(3)
    values = torch.arange(38 * 3, dtype=torch.float32).reshape(38, 3)
    scores, exact_values = engine._exact_generated_suffix(
        query, torch.ones(38, 3), values, 1.0
    )
    assert scores.numel() == 0
    assert exact_values.shape == (0, 3)


@pytest.mark.parametrize("engine_class", [SantappEngine, HierarchicalWholeTeamEngine])
def test_only_generated_rows_after_prompt_boundary_are_exact(engine_class) -> None:
    engine = object.__new__(engine_class)
    engine.sample_end = 38
    engine.head_dim = 2
    keys = torch.arange(42 * 2, dtype=torch.float32).reshape(42, 2)
    values = keys + 1000
    query = torch.tensor([1.0, 0.0])
    scores, exact_values = engine._exact_generated_suffix(query, keys, values, 1.0)
    assert scores.tolist() == keys[38:, 0].tolist()
    assert torch.equal(exact_values, values[38:])


@pytest.mark.parametrize("engine_class", [SantappEngine, HierarchicalWholeTeamEngine])
def test_ht_and_exact_combination_is_finite(engine_class) -> None:
    output = engine_class._combine_ht_and_exact(
        sampled_values=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        sampled_log_weights=torch.tensor([1000.0, 999.0]),
        exact_values=torch.tensor([[2.0, 2.0]]),
        exact_scores=torch.tensor([998.0]),
    )
    assert bool(torch.isfinite(output).all())
    assert output.shape == (2,)
