"""Native GPU tests: explicit opt-in; never repair output or require random partition parity."""
import os

import numpy as np
import pytest
import torch

from santapp_ruler.attention.minibatch_kmeans import SklearnLikeTorchMiniBatchKMeans as Reference
from santapp_ruler.attention.triton_prefill.prefill_kmeans import TritonMiniBatchKMeans
from santapp_ruler.attention.triton_prefill.prefill_teams import (
    build_team_summary_triton, build_team_summary_from_csr,
    build_contiguous_team_summary_triton, parents_from_labels)
from santapp_ruler.attention.triton_prefill.prefill_algorithmic_validation import (
    assert_team_algorithm, assert_kmeans_health)

pytestmark = pytest.mark.skipif(os.environ.get("SANTAPP_RUN_PREFILL_GPU_TESTS") != "1",
                                reason="GPU execution requires SANTAPP_RUN_PREFILL_GPU_TESTS=1")


@pytest.fixture(autouse=True)
def gpu_environment():
    from santapp_ruler.attention.prefill_runtime import require_triton_environment
    # Explicitly requested but unavailable CUDA is a FAILURE, not a successful skip.
    require_triton_environment()
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    yield
    torch.cuda.synchronize()
    torch.set_float32_matmul_precision(previous)


def keys(n, d, dtype=torch.float32):
    return torch.randn(n, d, generator=torch.Generator().manual_seed(n+d)).to(device="cuda",dtype=dtype)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("d", [7, 128, 256])
def test_generic_gapped_labels_and_independent_csr(dtype, d):
    x = keys(89, d, dtype)
    labels = torch.tensor([0]+[2]*2+[4]*3+[6]*10+[8]*73, device="cuda")
    labels = labels[torch.randperm(89, generator=torch.Generator().manual_seed(113)).cuda()]
    summary = build_team_summary_triton(x,labels,parent_cluster_count=11)
    assert_team_algorithm(summary,x,labels,configured_parent_count=11)
    csr = parents_from_labels(labels,11)
    summary = build_team_summary_from_csr(x,csr,representatives_per_parent=4)
    assert_team_algorithm(summary,x,labels,configured_parent_count=11)


@pytest.mark.parametrize("n", [1,2,3,4,15,16,17,31,33,38,257])
def test_contiguous_full_and_short_tail(n):
    x = keys(n,128,torch.float16)
    summary = build_contiguous_team_summary_triton(x)
    assert_team_algorithm(summary,x,torch.arange(n,device="cuda")//16,
                          configured_parent_count=(n+15)//16)
    assert summary.partial_parent_size == n%16


@pytest.mark.parametrize("n", [7,129,513])
def test_literal_ties_have_distinct_owned_leaders(n):
    x = torch.zeros(n,128,device="cuda")
    labels = torch.full((n,),3,device="cuda",dtype=torch.int64)
    summary = build_team_summary_triton(x,labels,parent_cluster_count=6)
    assert_team_algorithm(summary,x,labels,configured_parent_count=6)
    assert summary.leader_token_indices.tolist() == [0,1,2,3]


@pytest.mark.parametrize("n", [129,513,1025])
def test_large_random_parent_streams_without_truncation(n):
    x = keys(n,128)
    labels = torch.zeros(n,device="cuda",dtype=torch.long)
    summary = build_team_summary_triton(x,labels)
    assert_team_algorithm(summary,x,labels,configured_parent_count=1)


def test_large_contiguous_parents_with_tail():
    x = keys(515,128)
    summary = build_contiguous_team_summary_triton(x,parent_size=256)
    assert_team_algorithm(summary,x,torch.arange(515,device="cuda")//256,configured_parent_count=3)


@pytest.mark.parametrize("r", [1,4,16])
def test_strided_features_and_representative_counts(r):
    x = keys(49,16)[:,::2]
    labels = torch.arange(49,device="cuda")%3
    summary = build_team_summary_triton(x,labels,representatives_per_parent=r)
    assert_team_algorithm(summary,x,labels,representatives_per_parent=r,configured_parent_count=3)


def test_assignment_online_update_gather_and_reassignment():
    from santapp_ruler.attention.triton_prefill.prefill_kmeans_ops import TritonKMeansOps
    ops = TritonKMeansOps("ieee")
    # Production head dimension; all gather/update/reassignment checks stay mandatory.
    x,c = keys(67,128), keys(35,128)
    c[3] = c[0]
    ids = torch.tensor([66,0,1,1,2,45],device="cuda")
    torch.testing.assert_close(ops.gather(x,ids),x[ids],rtol=0,atol=0)
    labels,inertia = ops.assign(x,c)
    distances = ((x[:,None,:].double()-c[None,:,:].double())**2).sum(-1)
    best = distances.min(1).values
    cost = distances[torch.arange(len(x),device="cuda"),labels]
    assert bool((cost <= best + 2e-4 + 2e-5*best).all())
    torch.testing.assert_close(inertia.double(),cost.sum(),rtol=2e-5,atol=2e-4)
    tied,_ = ops.assign(torch.zeros_like(x),torch.ones_like(c))
    assert not bool(tied.any())
    counts = torch.arange(35,device="cuda",dtype=torch.float32)
    new,new_counts = ops.update(x,labels,c,counts)
    batch_counts = torch.bincount(labels,minlength=35).float()
    sums = torch.zeros_like(c).index_add_(0,labels,x)
    expected = c.clone(); active = batch_counts > 0
    expected[active] = (c[active]*counts[active,None]+sums[active])/(counts+batch_counts)[active,None]
    torch.testing.assert_close(new,expected,rtol=2e-6,atol=2e-6)
    torch.testing.assert_close(new_counts,counts+batch_counts,rtol=0,atol=0)
    mask = torch.arange(35,device="cuda")%9 == 0
    rows = torch.tensor([1,5,6,8],device="cuda")
    expected[mask] = x[rows]
    expected_counts = new_counts.clone(); expected_counts[mask] = new_counts[~mask].min()
    ops.reassign_(new,new_counts,x,mask,rows)
    torch.testing.assert_close(new,expected,rtol=2e-6,atol=2e-6)
    torch.testing.assert_close(new_counts,expected_counts,rtol=0,atol=0)


@pytest.mark.parametrize("n", [31,513,1025])
def test_greedy_init_uses_cached_rows_and_preserves_random_stream(n):
    from santapp_ruler.attention.triton_prefill.prefill_kmeans_ops import TritonKMeansOps
    x = keys(n,7)
    a,b = np.random.RandomState(217),np.random.RandomState(217)
    Reference(5)._greedy_kmeans_plus_plus(x,a)
    centers = TritonKMeansOps().greedy_initialize(x,5,b)
    assert centers.shape == (5,7) and bool(torch.isfinite(centers).all())
    assert bool(((centers[:,None,:] == x[None,:,:]).all(-1).any(-1)).all())
    assert np.array_equal(a.randint(0,10000,20),b.randint(0,10000,20))


@pytest.mark.parametrize("extra", [{},{"n_init":2,"init_size":23},{"tol":0.1}])
def test_separated_prototypes_recover_zero_cost(extra):
    x = (torch.eye(8,device="cuda")[:4]*8).repeat_interleave(24,dim=0)
    kwargs = dict(n_clusters=4,batch_size=24,max_iter=4,random_state=14,max_no_improvement=3)
    kwargs.update(extra)
    km = TritonMiniBatchKMeans(**kwargs)
    labels = km.fit_predict(x)
    assert_kmeans_health(km,labels,tokens=96,dimensions=8,clusters=4)
    assert km.inertia_ == 0
    assert bool(((x-km.cluster_centers_[labels])**2).sum(-1).eq(0).all())


def test_zero_potential_and_empty_center_reassignment():
    x = torch.zeros(37,7,device="cuda")
    km = TritonMiniBatchKMeans(4,batch_size=7,max_iter=3,max_no_improvement=2)
    labels = km.fit_predict(x)
    assert_kmeans_health(km,labels,tokens=37,dimensions=7,clusters=4)
    assert km.reassignment_events_ and km.inertia_ == 0


@pytest.mark.parametrize("policy", ["santapp","hierarchical"])
def test_unchanged_sampler_small_large_and_take_all_budgets(policy):
    from santapp_ruler.attention.triton_prefill.smoke import check_decode_budgets
    x = keys(513,128)
    summary = (build_team_summary_triton(x,torch.arange(len(x),device="cuda")//19)
               if policy == "santapp" else build_contiguous_team_summary_triton(x))
    rows = check_decode_budgets(summary,x,x.sin(),policy=policy,
        budgets=[4,128,256,512,1024,2048,4096,8192,16384,32768])
    assert len(rows) == 20 and rows[-1]["take_all"]


def _check_assignment_cost(x, centers):
    from santapp_ruler.attention.triton_prefill.prefill_kmeans_ops import TritonKMeansOps
    labels, inertia = TritonKMeansOps("ieee").assign(x, centers)
    distances = ((x[:, None, :].double() - centers[None, :, :].double()) ** 2).sum(-1)
    best = distances.min(1).values
    cost = distances[torch.arange(len(x), device=x.device), labels]
    allowance = 2e-4 + 2e-5 * best
    excess = cost - best - allowance
    assert bool((excess <= 0).all()), (
        f"non-nearest labels: {int((excess > 0).sum())}; "
        f"max excess={float(excess.max())}; D={x.shape[1]}"
    )
    torch.testing.assert_close(inertia.double(), cost.sum(), rtol=2e-5, atol=2e-4)


def test_production_d128_multitile_assignment():
    # Cross multiple point/center tiles and non-power-of-two tails.
    _check_assignment_cost(keys(513, 128), keys(257, 128))


def test_known_a10_d7_assignment_diagnostic(request):
    """Keep the exact failing synthetic fixture visible; never mask D=128 failures."""
    import triton
    from santapp_ruler.attention.triton_prefill.known_issues import is_known_a10_d7_assignment_case
    if is_known_a10_d7_assignment_case(torch.cuda.get_device_name(),
                                      torch.cuda.get_device_capability(), triton.__version__, 7):
        request.node.add_marker(pytest.mark.xfail(
            reason="Known R004 A10/sm86 + Triton 3.6.0 D=7 assignment defect; not production D=128. See docs/KNOWN_ISSUES.md",
            raises=AssertionError, strict=False))
    x, centers = keys(67, 7), keys(35, 7)
    centers[3] = centers[0]
    _check_assignment_cost(x, centers)
