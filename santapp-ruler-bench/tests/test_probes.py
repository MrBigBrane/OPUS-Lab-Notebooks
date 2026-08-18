import pytest
import torch

from santapp_ruler.attention.probes import last_prompt_probe_positions


def test_last_prompt_probe_positions_are_contiguous_and_most_recent():
    positions = last_prompt_probe_positions(100, 8)
    assert torch.equal(positions, torch.arange(92, 100))


def test_all_prompt_tokens_can_be_probes():
    assert torch.equal(last_prompt_probe_positions(4, 4), torch.arange(4))


def test_probe_count_cannot_exceed_prompt():
    with pytest.raises(ValueError, match="exceeds"):
        last_prompt_probe_positions(4, 5)
