import torch

from santapp_ruler.attention.minibatch_kmeans import (
    SklearnLikeTorchMiniBatchKMeans,
)


def test_minibatch_kmeans_returns_valid_labels_on_cpu() -> None:
    x = torch.tensor(
        [[0.0, 0.0], [0.1, 0.0], [10.0, 10.0], [10.1, 10.0]],
        dtype=torch.float32,
    )
    labels = SklearnLikeTorchMiniBatchKMeans(
        n_clusters=2,
        batch_size=4,
        max_iter=5,
        random_state=0,
    ).fit_predict(x)
    assert labels.shape == (4,)
    assert labels.dtype == torch.long
    assert len(torch.unique(labels)) == 2


def test_minibatch_kmeans_is_deterministic_for_fixed_seed() -> None:
    generator = torch.Generator().manual_seed(17)
    x = torch.randn(40, 5, generator=generator)
    kwargs = dict(
        n_clusters=4,
        batch_size=16,
        n_init=1,
        max_iter=5,
        random_state=9,
    )
    first = SklearnLikeTorchMiniBatchKMeans(**kwargs).fit_predict(x)
    second = SklearnLikeTorchMiniBatchKMeans(**kwargs).fit_predict(x)
    assert torch.equal(first, second)
