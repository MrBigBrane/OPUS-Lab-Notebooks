import torch

from santapp_ruler.attention.santa import SantaEngine
from santapp_ruler.config import SantaConfig


def test_santa_estimator_is_mean_of_values_drawn_from_full_distribution(monkeypatch):
    engine = object.__new__(SantaEngine)
    engine.head_dim = 2
    engine.config = SantaConfig(samples_per_head=3)

    chosen = torch.tensor([0, 2, 2])

    def fake_multinomial(probability, num_samples, replacement):
        assert probability.shape == (3,)
        assert num_samples == 3
        assert replacement is True
        return chosen

    monkeypatch.setattr(torch, "multinomial", fake_multinomial)
    query = torch.tensor([1.0, 0.0])
    keys = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    values = torch.tensor([[1.0, 2.0], [10.0, 20.0], [4.0, 8.0]])

    estimate = engine._approximate_attention(query, keys, values, 0, 0)

    assert estimate.sampled_indices.tolist() == [0, 2, 2]
    assert torch.allclose(estimate.output, torch.tensor([3.0, 6.0]))
