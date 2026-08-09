import pytest

from santapp_ruler.attention.probes import last_prompt_probe_positions


def test_probe_policy_uses_exactly_the_last_prompt_tokens():
    positions = last_prompt_probe_positions(100, 4)
    assert positions.tolist() == [96, 97, 98, 99]


def test_probe_count_must_fit_prompt():
    with pytest.raises(ValueError, match="exceeds prompt_tokens"):
        last_prompt_probe_positions(3, 4)
