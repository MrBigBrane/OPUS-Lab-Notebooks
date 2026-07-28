import pytest


torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

from santapp_ruler.attention.minibatch_kmeans import (  # noqa: E402
    SklearnLikeTorchMiniBatchKMeans,
)


def test_minibatch_kmeans_returns_valid_labels_on_cpu():
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
