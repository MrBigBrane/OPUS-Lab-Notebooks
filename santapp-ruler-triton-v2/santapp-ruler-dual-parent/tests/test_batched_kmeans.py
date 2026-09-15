"""CPU controller oracle; deliberately NOT evidence of CUDA correctness."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from santapp_ruler.attention.minibatch_kmeans import SklearnLikeTorchMiniBatchKMeans
from santapp_ruler.attention.triton_prefill.batched_kmeans import (
    BatchedTritonMiniBatchKMeans, fit_chunks, reassignment_mask,
)
from test_triton_prefill import CPUControllerOps


class CPUOracleBatchedOps:
    """Independent CPU numerical oracles injected only by tests, never production."""
    def __init__(self):
        self.single = CPUControllerOps()
        self.assign_active_history = []

    def gather(self, x, ids):
        return x[torch.arange(x.shape[0])[:, None], ids]

    def greedy_initialize(self, x, clusters, rngs):
        return torch.stack([self.single.greedy_initialize(a, clusters, rng)
                            for a, rng in zip(x, rngs)])

    def assign(self, x, centers, active=None):
        self.assign_active_history.append(None if active is None else active.clone())
        results = [self.single.assign(a, c) for a, c in zip(x, centers)]
        return torch.stack([r[0] for r in results]), torch.stack([r[1] for r in results])

    def update(self, x, labels, centers, counts, active):
        results = [(self.single.update(x[f], labels[f], centers[f], counts[f])
                    if active[f] else (centers[f].clone(), counts[f].clone()))
                   for f in range(len(x))]
        return torch.stack([r[0] for r in results]), torch.stack([r[1] for r in results])

    def reassign_(self, centers, counts, x_batch, mask, row_ids):
        for f in range(len(centers)):
            size = int(mask[f].sum())
            if size:
                self.single.reassign_(centers[f], counts[f], x_batch[f], mask[f],
                                      row_ids[f, :size])


@pytest.mark.parametrize("extra", [
    {},
    {"n_init": 3, "init_size": 12},
    {"tol": 0.01, "max_no_improvement": None},
    {"batch_size": 4, "n_clusters": 12, "init_size": 6, "reassignment_ratio": 0.7},
    {"batch_size": 4096, "n_clusters": 1, "max_iter": 1},
    {"max_no_improvement": 1, "reassignment_ratio": 0.0},
    {"batch_size": 1, "n_clusters": 4, "max_iter": 1},
])
def test_batched_controller_preserves_independent_reference_fits(monkeypatch, extra):
    import santapp_ruler.attention.triton_prefill.batched_kmeans as module
    oracle = CPUOracleBatchedOps()
    monkeypatch.setattr(module, "_make_batched_ops", lambda x, precision: oracle)
    torch.set_num_threads(1)
    x = torch.randn(4, 83, 7, generator=torch.Generator().manual_seed(91))
    x[1].zero_()  # Different stop/reassignment schedule in the same batch.
    params = dict(n_clusters=4, batch_size=16, max_iter=3, random_state=17,
                  max_no_improvement=3)
    params.update(extra)
    batched = BatchedTritonMiniBatchKMeans(**params)
    labels = batched.fit_predict(x)
    for f in range(len(x)):
        old = SklearnLikeTorchMiniBatchKMeans(**params)
        expected = old.fit_predict(x[f])
        assert torch.equal(labels[f], expected)
        torch.testing.assert_close(batched.cluster_centers_[f], old.cluster_centers_,
                                   atol=0, rtol=0)
        torch.testing.assert_close(batched.counts_[f], old.counts_, atol=0, rtol=0)
        assert batched.n_steps_[f] == old.n_steps_
        assert batched.n_iter_[f] == old.n_iter_
        assert batched.inertia_[f] == old.inertia_
    assert batched._ops is None
    assert oracle.assign_active_history[-1] is None  # Final labels for stopped fits too.
    assert batched.minibatch_host_control_transfers_ <= 2 * max(batched.n_steps_)
    if not extra:
        assert len(set(batched.n_steps_)) > 1
        assert any(not bool(a.all()) for a in oracle.assign_active_history if a is not None)


def test_chunking_never_drops_or_reorders_fits():
    for fits, batch in [(112, 8), (112, 16), (11, 4), (3, 8), (1, 1)]:
        chunks = fit_chunks(fits, batch)
        assert [i for c in chunks for i in range(c.first, c.stop)] == list(range(fits))
        assert max(c.size for c in chunks) <= batch
    assert len(fit_chunks(112, 8)) == 14


@pytest.mark.parametrize("batch", [0, -1, 1.5, True, 65536])
def test_chunk_size_is_explicitly_validated(batch):
    with pytest.raises(ValueError):
        fit_chunks(112, batch)


@pytest.mark.parametrize("shape", [(2, 3), (0, 8, 4), (2, 2, 4), (2, 8, 257)])
def test_invalid_batch_shapes_fail_without_triton(shape):
    with pytest.raises(ValueError):
        BatchedTritonMiniBatchKMeans(4).fit_predict(torch.ones(shape))


def test_invalid_precision_and_nonfinite_fail_without_triton():
    with pytest.raises(TypeError, match="float64"):
        BatchedTritonMiniBatchKMeans(2).fit_predict(torch.ones(2, 8, 4, dtype=torch.float64))
    with pytest.raises(ValueError, match="finite"):
        BatchedTritonMiniBatchKMeans(2).fit_predict(torch.full((2, 8, 4), float('nan')))


def test_reassignment_half_batch_cap_and_inactive_fit():
    counts = torch.tensor([[0., 0., 0., 1., 1., 9.], [0., 0., 0., 1., 1., 9.]])
    mask = reassignment_mask(counts, torch.tensor([True, False]), 4, 0.7)
    assert mask.sum(dim=1).tolist() == [2, 0]
    assert not bool(reassignment_mask(counts, torch.ones(2, dtype=torch.bool), 1, 0.7).any())


def test_batch_partition_does_not_change_rng_or_fit_state(monkeypatch):
    import santapp_ruler.attention.triton_prefill.batched_kmeans as module
    monkeypatch.setattr(module, "_make_batched_ops", lambda x, precision: CPUOracleBatchedOps())
    x = torch.randn(7, 39, 5, generator=torch.Generator().manual_seed(42))
    params = dict(n_clusters=5, batch_size=7, max_iter=2, random_state=77)
    whole = BatchedTritonMiniBatchKMeans(**params).fit_predict(x)
    chunks = [BatchedTritonMiniBatchKMeans(**params).fit_predict(x[c.first:c.stop])
              for c in fit_chunks(len(x), 3)]
    assert torch.equal(whole, torch.cat(chunks))


def test_compile_plan_covers_all_wrapper_signatures_without_importing_triton():
    import ast
    from pathlib import Path
    from santapp_ruler.attention.triton_prefill.batched_compile_check import compile_plan
    path = (Path(__file__).resolve().parents[1] / 'src/santapp_ruler/attention/'
            'triton_prefill/kernels/p007_batched_kmeans.py')
    source = ast.parse(path.read_text())
    signatures = {n.name: {a.arg for a in n.args.args} for n in source.body
                  if isinstance(n, ast.FunctionDef)}
    cases = compile_plan()
    assert {c.kernel for c in cases} == set(signatures)
    for c in cases:
        assert set(c.signature) | set(c.constants) == signatures[c.kernel], c
        assert not set(c.signature) & set(c.constants)
    # All independent fits live in grid axis 2, no fit-count specialization.
    functions = [n for n in source.body if isinstance(n, ast.FunctionDef)]
    assert all('tl.program_id(2)' in ast.unparse(f) for f in functions)
    assert all('single.' in ast.unparse(f) for f in functions)


def test_new_config_and_cli_separate_fit_batch_from_token_minibatch():
    from santapp_ruler.cli import build_parser, _config_with_cli
    args = build_parser().parse_args(['run', '--kmeans-fit-batch-size', '16'])
    config = _config_with_cli(args)
    assert config.santapp.kmeans.fit_batch_size == 16
    assert config.santapp.kmeans.batch_size == 4096


def test_legacy_native_results_cannot_be_silently_resumed_as_batched(tmp_path):
    import yaml
    from santapp_ruler.config import load_config
    from santapp_ruler.run_state import prepare_run_directory
    config = load_config(overrides=['generation.backends=[santapp]'])
    old = config.to_dict()
    old['santapp']['kmeans'].pop('fit_batch_size')
    path = tmp_path / 'config.resolved.yaml'
    path.write_text(yaml.safe_dump(old))
    with pytest.raises(ValueError, match='predate batched fits'):
        prepare_run_directory(config, tmp_path)
    assert 'fit_batch_size' not in path.read_text()


def test_engine_batches_all_layer_head_inputs_in_order(monkeypatch):
    from types import SimpleNamespace
    from santapp_ruler.attention.qwen import LayerCache
    from santapp_ruler.attention.santapp import SantappEngine
    from santapp_ruler.config import SantappConfig
    import santapp_ruler.attention.triton_prefill.batched_kmeans as module
    calls = []
    class FakeEstimator:
        def __init__(self, **kwargs):
            pass
        def fit_predict(self, x):
            calls.append((tuple(x.shape), x[:, 0, 0].tolist()))
            b, n, _ = x.shape
            self.n_steps_ = np.ones(b, dtype=np.int64)
            self.inertia_ = np.zeros(b)
            self.stopping_reason_ = ['max_iter'] * b
            self.active_fits_per_step_ = [b]
            self.launch_counts_ = {'first_center': 1}
            self.minibatch_host_control_transfers_ = 1
            return torch.zeros((b, n), dtype=torch.int64)
    monkeypatch.setattr(module, 'BatchedTritonMiniBatchKMeans', FakeEstimator)
    engine = object.__new__(SantappEngine)
    engine.config = SantappConfig()
    engine.num_layers, engine.num_kv_heads, engine.head_dim = 28, 4, 2
    engine.summaries, engine._prefill_metrics = {}, {'kmeans_fit_chunks': [], 'team_build_chunks': []}
    engine.cache = {}
    for layer in range(28):
        key = torch.arange(4).view(4, 1, 1).expand(4, 17, 2).float() + 4 * layer
        engine.cache[layer] = LayerCache(key, key, 17)
    engine._build_batched_kmeans_teams(17, {'n_clusters': 2},
        lambda keys, labels, **kw: float(keys[0, 0]))
    assert len(calls) == 14
    assert all(shape == (8, 17, 2) for shape, _ in calls)
    assert [value for _, group in calls for value in group] == list(range(112))
    assert all(engine.summaries[(l, h)] == 4*l + h for l in range(28) for h in range(4))


def test_phase_telemetry_retains_zero_boundaries_and_all_fit_windows(tmp_path):
    import json
    from santapp_ruler.gpu_telemetry import summarize_phase_telemetry, summarize_windows
    path = tmp_path/'gpu-utilization.csv'
    path.write_text('unix_time,gpu_uuid,gpu_util_pct\n1,g,0\n2,g,90\n3,g,0\n4,g,90\n5,g,90\n6,g,90\n')
    (tmp_path/'predictions').mkdir()
    metrics = {'generation_started_at_unix': 1., 'generation_finished_at_unix': 6.,
               'decode_started_at_unix': 4., 'decode_finished_at_unix': 6.,
               'kmeans_fit_chunks': [{'started_at_unix': 1., 'finished_at_unix': 2.},
                                     {'started_at_unix': 3., 'finished_at_unix': 4.}]}
    (tmp_path/'predictions/x.jsonl').write_text(json.dumps({'metrics': metrics})+'\n')
    (tmp_path/'console.log').write_text('PHASE_START=decode_prepare unix=3.0\nPHASE_END=decode_prepare unix=4.0\n')
    report = summarize_phase_telemetry(tmp_path)
    phases = report['cases'][0]['phases']
    assert phases['generation']['mean_device_gpu_util_pct'] == 60
    assert phases['kmeans_fit']['mean_device_gpu_util_pct'] == 45
    assert phases['decode']['mean_device_gpu_util_pct'] == 90
    assert phases['decode_prepare']['mean_device_gpu_util_pct'] == 45
    assert not report['boundary_samples_removed']
    assert summarize_windows(path, [(10, 20)])['mean_device_gpu_util_pct'] is None
    path.write_text(path.read_text()+'4,other,10\n')
    value = summarize_windows(path, [(1, 6)])
    assert value['ambiguous_multiple_devices'] and value['mean_device_gpu_util_pct'] is None
