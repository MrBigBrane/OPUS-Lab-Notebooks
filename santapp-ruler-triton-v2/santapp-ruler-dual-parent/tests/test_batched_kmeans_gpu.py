"""Actual CUDA tests. CPU controller tests do not substitute for these."""
import numpy as np
import pytest
import torch

from santapp_ruler.attention.triton_prefill.batched_kmeans import BatchedTritonMiniBatchKMeans

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')]


@pytest.fixture(autouse=True)
def cuda_environment():
    from santapp_ruler.attention.prefill_runtime import require_triton_environment
    require_triton_environment()
    old = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision('highest')
    yield
    torch.cuda.synchronize()
    torch.set_float32_matmul_precision(old)


def fixture(b=3, n=137, d=128):
    return torch.randn(b, n, d, generator=torch.Generator().manual_seed(8721)).cuda()


def test_gather_and_greedy_seed_streams_are_independent():
    from santapp_ruler.attention.triton_prefill.batched_kmeans_ops import BatchedTritonKMeansOps
    from santapp_ruler.attention.triton_prefill.prefill_kmeans_ops import TritonKMeansOps
    x = fixture()
    ids = torch.tensor([[136, 0, 17, 17], [0, 0, 1, 37], [3, 79, 37, 16]], device='cuda')
    ops = BatchedTritonKMeansOps()
    torch.testing.assert_close(ops.gather(x, ids), x[torch.arange(3, device='cuda')[:, None], ids],
                               rtol=0, atol=0)
    rngs = [np.random.RandomState(27 + i) for i in range(3)]
    centers = ops.greedy_initialize(x, 7, rngs)
    for f in range(3):
        rng = np.random.RandomState(27 + f)
        single = TritonKMeansOps().greedy_initialize(x[f], 7, rng)
        # Small continuous-valued fixture has no ambiguous trial-score ties.
        torch.testing.assert_close(centers[f], single, rtol=0, atol=0)
        assert np.array_equal(rng.randint(0, 10000, 20), rngs[f].randint(0, 10000, 20))
        assert bool(((centers[f, :, None] == x[f, None]).all(-1).any(-1)).all())
    assert ops.launch_counts['scan_closest'] == 6  # NOT 3*6 launches.


def test_assignment_update_and_stopped_fit_isolation():
    from santapp_ruler.attention.triton_prefill.batched_kmeans_ops import BatchedTritonKMeansOps
    x = fixture(n=113)
    centers = x[:, :35].clone()
    counts = torch.arange(35, device='cuda').float().expand(3, 35).clone()
    active = torch.tensor([True, False, True], device='cuda')
    ops = BatchedTritonKMeansOps()
    labels, inertia = ops.assign(x, centers, active)
    new, new_counts = ops.update(x, labels, centers, counts, active)
    torch.testing.assert_close(new[1], centers[1], rtol=0, atol=0)
    torch.testing.assert_close(new_counts[1], counts[1], rtol=0, atol=0)
    assert float(inertia[1]) == 0
    for f in (0, 2):
        dist = (x[f, :, None].double() - centers[f, None].double()).square().sum(-1)
        costs = dist[torch.arange(113, device='cuda'), labels[f]]
        best = dist.min(1).values
        assert bool((costs <= best + 2e-4 + 2e-5*best).all())
        batch_counts = torch.bincount(labels[f], minlength=35).float()
        sums = torch.zeros_like(centers[f]).index_add_(0, labels[f], x[f])
        expected = centers[f].clone()
        changed = batch_counts > 0
        expected[changed] = (centers[f, changed]*counts[f, changed, None] + sums[changed]) / (counts[f] + batch_counts)[changed, None]
        torch.testing.assert_close(new[f], expected, rtol=3e-6, atol=3e-6)
        torch.testing.assert_close(new_counts[f], counts[f]+batch_counts, rtol=0, atol=0)


def test_reassignment_ragged_row_counts_and_fit_offsets():
    from santapp_ruler.attention.triton_prefill.batched_kmeans_ops import BatchedTritonKMeansOps
    x = fixture(n=33)
    centers = x[:, :7].clone()
    counts = torch.arange(7, device='cuda').float().expand(3, 7).clone()
    mask = torch.tensor([[1, 0, 1, 0, 0, 0, 0], [0]*7, [0, 1, 1, 1, 0, 0, 0]],
                        dtype=torch.bool, device='cuda')
    rows = torch.tensor([[32, 13, 0], [0, 0, 0], [11, 12, 13]], device='cuda')
    expected, expected_counts = centers.clone(), counts.clone()
    for f, n in enumerate((2, 0, 3)):
        if n:
            expected[f, mask[f]] = x[f, rows[f, :n]]
            expected_counts[f, mask[f]] = counts[f, ~mask[f]].min()
    BatchedTritonKMeansOps().reassign_(centers, counts, x, mask, rows)
    torch.testing.assert_close(centers, expected, rtol=0, atol=0)
    torch.testing.assert_close(counts, expected_counts, rtol=0, atol=0)


@pytest.mark.parametrize('fits', [1, 3, 8])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_complete_fits_with_divergent_stopping_and_precision(fits, dtype):
    from santapp_ruler.attention.triton_prefill.batched_validation import validate_fitted_batch
    x = fixture(fits, n=257).to(dtype).float()
    if fits > 1:
        x[1].zero_()
    estimator = BatchedTritonMiniBatchKMeans(16, batch_size=64, n_init=2,
        init_size=97, max_iter=4, max_no_improvement=2, random_state=17)
    labels = estimator.fit_predict(x)
    validate_fitted_batch(x, estimator, labels)
    assert estimator._ops is None
    assert estimator.launch_counts_['scan_closest'] == 2 * 15
    if fits > 1:
        assert estimator.stopping_reason_[1] == 'max_no_improvement'
        assert estimator.inertia_[1] == 0
        assert all(a >= b for a, b in zip(estimator.active_fits_per_step_,
                                         estimator.active_fits_per_step_[1:]))


@pytest.mark.parametrize('extra', [{}, {'n_init': 2, 'init_size': 23}, {'tol': 0.1}])
def test_separated_prototypes_recover_exact_zero_in_all_fits(extra):
    x = (torch.eye(128, device='cuda')[:4]*8).repeat_interleave(24, dim=0)
    x = torch.stack([x, x.roll(24, dims=0), x.roll(48, dims=0)])
    params = dict(n_clusters=4, batch_size=24, max_iter=4,
                  random_state=14, max_no_improvement=3)
    params.update(extra)
    km = BatchedTritonMiniBatchKMeans(**params)
    labels = km.fit_predict(x)
    assert bool(np.all(km.inertia_ == 0))
    for f in range(3):
        assert bool((x[f] == km.cluster_centers_[f, labels[f]]).all())


@pytest.mark.parametrize('budget', [32, 1024])
def test_actual_batched_summaries_feed_grouped_decoder_without_repair(budget):
    from santapp_ruler.attention.triton_decode.validation import validate_case
    from santapp_ruler.attention.triton_prefill.prefill_teams import build_team_summary_triton
    from santapp_ruler.attention.triton_prefill.prefill_algorithmic_validation import assert_team_algorithm
    def build(keys):
        x = keys.float().contiguous()
        km = BatchedTritonMiniBatchKMeans(32, batch_size=128, max_iter=2, random_state=13)
        labels = km.fit_predict(x)
        results = []
        for h in range(x.shape[0]):
            summary = build_team_summary_triton(x[h], labels[h], parent_cluster_count=32)
            assert_team_algorithm(summary, x[h], labels[h], configured_parent_count=32)
            results.append(summary)
        return results
    validate_case(513, budget, suffixes=(0, 1, 17), kv_heads=3, group=7,
                  summaries_override=build)
