"""CPU algorithm-controller/dispatch/deployment tests; no GPU execution claim."""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch
import yaml

from santapp_ruler.attention.minibatch_kmeans import SklearnLikeTorchMiniBatchKMeans as Reference
from santapp_ruler.attention.triton_prefill.prefill_kmeans import TritonMiniBatchKMeans
from santapp_ruler.attention.triton_prefill.prefill_teams import (
    ParentCSR, contiguous_parents, parents_from_labels, parent_buckets, validate_parent_csr)
from santapp_ruler.attention.prefill_runtime import device_compatibility_error
from santapp_ruler.attention.triton_prefill.compile_check import compile_plan
from santapp_ruler.config import load_config

ROOT = Path(__file__).resolve().parents[1]
class CPUControllerOps:
    """A torch oracle for the host controller, NOT an emulated GPU correctness claim."""
    def gather(self, x, ids):
        return x[ids]

    def greedy_initialize(self, x, clusters, rng):
        return Reference(clusters)._greedy_kmeans_plus_plus(x, rng)

    def assign(self, x, centers):
        return Reference(centers.shape[0])._assign(x, centers)

    def update(self, x, labels, centers, counts):
        batch_counts = torch.bincount(labels, minlength=len(centers)).to(x.dtype)
        sums = torch.zeros_like(centers)
        sums.index_add_(0, labels, x)
        new_counts = counts + batch_counts
        result = centers.clone()
        active = batch_counts > 0
        result[active] = (centers[active] * counts[active, None] + sums[active]) / new_counts[active, None]
        return result, new_counts

    def reassign_(self, centers, counts, x_batch, mask, row_ids):
        centers[mask] = x_batch[row_ids]
        counts[mask] = counts[~mask].min()


@pytest.mark.parametrize("extra", [
    {},
    {"n_init": 3, "init_size": 12},
    {"tol": 0.01, "max_no_improvement": None},
    {"batch_size": 4, "n_clusters": 12, "init_size": 6, "reassignment_ratio": 0.7},
    {"batch_size": 4096, "n_clusters": 1, "max_iter": 1},
    {"max_no_improvement": 1, "reassignment_ratio": 0.0},
])
def test_host_controller_matches_unmodified_reference(monkeypatch, extra):
    import santapp_ruler.attention.triton_prefill.prefill_kmeans as module
    monkeypatch.setattr(module, "_make_ops", lambda x, precision: CPUControllerOps())
    torch.set_num_threads(1)
    x = torch.randn(83, 7, generator=torch.Generator().manual_seed(91))
    params = dict(n_clusters=4, batch_size=16, max_iter=3, random_state=17,
                  max_no_improvement=3)
    params.update(extra)
    old, new = Reference(**params), TritonMiniBatchKMeans(**params)
    a, b = old.fit_predict(x), new.fit_predict(x)
    assert torch.equal(a, b)
    for name in ("cluster_centers_", "counts_"):
        torch.testing.assert_close(getattr(new, name), getattr(old, name), atol=0, rtol=0)
    assert (new.n_steps_, new.n_iter_, new.inertia_) == (old.n_steps_, old.n_iter_, old.inertia_)
    assert new._ops is None  # Workspaces released, not retained per model head.


def test_zero_potential_and_early_stopping_controller(monkeypatch):
    import santapp_ruler.attention.triton_prefill.prefill_kmeans as module
    monkeypatch.setattr(module, "_make_ops", lambda x, precision: CPUControllerOps())
    kwargs = dict(n_clusters=4, batch_size=7, max_iter=4, max_no_improvement=2)
    old, new = Reference(**kwargs), TritonMiniBatchKMeans(**kwargs)
    x = torch.zeros(31, 5)
    assert torch.equal(old.fit_predict(x), new.fit_predict(x))
    assert new.n_steps_ == old.n_steps_
    assert new.stopping_reason_ == "max_no_improvement"
    assert new.reassignment_events_


@pytest.mark.parametrize("kwargs", [
    {"n_clusters": 0}, {"batch_size": 0}, {"n_init": 0}, {"max_iter": 0},
    {"tol": -1}, {"tol": float("nan")}, {"init_size": 0},
    {"max_no_improvement": 0}, {"reassignment_ratio": 1.1}, {"dot_precision": "tf32"},
])
def test_bad_kmeans_settings_fail_without_cuda(kwargs):
    params = {"n_clusters": 4, **kwargs}
    with pytest.raises(ValueError):
        TritonMiniBatchKMeans(**params)


def test_float64_is_not_silently_demoted():
    with pytest.raises(TypeError, match="float64"):
        TritonMiniBatchKMeans(1).fit_predict(torch.ones(2, 3, dtype=torch.float64))


def test_label_csr_gaps_and_order():
    labels = torch.tensor([5, 0, 5, 2, 0, 5, 2, 2], dtype=torch.int32)
    csr = parents_from_labels(labels, 8)
    assert csr.members.tolist() == [1, 4, 3, 6, 7, 0, 2, 5]
    assert csr.indptr.tolist() == [0, 2, 2, 5, 5, 5, 8, 8, 8]
    assert validate_parent_csr(csr, len(labels), "cpu").tolist() == [2, 0, 3, 0, 0, 3, 0, 0]


@pytest.mark.parametrize("n,p", [(1, 16), (15, 16), (16, 16), (17, 16), (38, 16), (513, 1024)])
def test_contiguous_csr_keeps_entire_tail(n, p):
    csr = contiguous_parents(n, p)
    sizes = validate_parent_csr(csr, n, "cpu")
    assert sizes.sum() == n
    assert csr.members.tolist() == list(range(n))
    assert sizes[-1] == (n % p or p)


@pytest.mark.parametrize("labels,count", [([-1, 0], None), ([1, 2], 2)])
def test_bad_labels_rejected(labels, count):
    with pytest.raises(ValueError):
        parents_from_labels(torch.tensor(labels), count)


@pytest.mark.parametrize("members,ptr", [
    ([0, 0, 2], [0, 3]), ([0, 1, 2], [0, 2]),
    ([1, 0, 2], [0, 3]), ([0, 1, 2], [1, 3]),
    ([0, 1, 2], [0, 3, 2, 3]),
])
def test_invalid_csr_fails_before_kernel_access(members, ptr):
    with pytest.raises(ValueError):
        validate_parent_csr(ParentCSR(torch.tensor(members), torch.tensor(ptr)), 3, "cpu")


def test_buckets_cover_every_active_parent_including_huge_parent():
    sizes = np.array([0, 1, 2, 3, 8, 9, 16, 128, 129, 32768, 0])
    buckets = parent_buckets(sizes)
    selected = np.concatenate([ids for _, ids in buckets])
    assert sorted(selected.tolist()) == np.flatnonzero(sizes).tolist()
    assert buckets[-1][0] == 0
    assert buckets[-1][1].tolist() == [8, 9]
    for tile, ids in buckets[:-1]:
        assert np.all(sizes[ids] <= tile)



def test_original_decode_and_ported_source_hashes():
    manifest = json.loads((ROOT / "docs/TRITON_PREFILL_PROVENANCE.json").read_text())
    integration = json.loads((ROOT / "docs/GROUPED_DECODE_PROVENANCE.json").read_text())
    approved = integration["modified_original_files"]
    for filename, digest in manifest["unchanged_original_files"].items():
        actual = hashlib.sha256((ROOT / filename).read_bytes()).hexdigest()
        if filename in approved:
            assert approved[filename]["previous_sha256"] in {digest, manifest.get("allowed_metadata_lf_sha256", {}).get(filename)}
            assert actual == approved[filename]["sha256"], filename
        else:
            assert actual in {digest, manifest.get("allowed_metadata_lf_sha256", {}).get(filename)}, filename
    for item in manifest["files"]:
        assert hashlib.sha256((ROOT / item["destination"]).read_bytes()).hexdigest() == item["ported_sha256"]
    for filename, methods in manifest["unchanged_decode_methods"].items():
        source = (ROOT / "src/santapp_ruler/attention" / filename).read_text()
        tree = ast.parse(source)
        found = {node.name: node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
        for name, digest in methods.items():
            value = ast.get_source_segment(source, found[name])
            assert hashlib.sha256(value.encode()).hexdigest() == digest, name


def test_reference_config_changes_only_prefill_and_decode_dispatch():
    from dataclasses import asdict
    a = asdict(load_config(ROOT / "configs/default.yaml"))
    b = asdict(load_config(ROOT / "configs/torch.yaml"))
    for name in ("santapp", "hierarchical"):
        assert a[name]["prefill_backend"] == "triton"
        assert b[name]["prefill_backend"] == "torch"
        b[name]["prefill_backend"] = "triton"
        assert a[name]["decode_backend"] == "grouped_triton"
        assert b[name]["decode_backend"] == "torch"
        b[name]["decode_backend"] = "grouped_triton"
    assert a == b


@pytest.mark.parametrize("context", [8192, 32768])
@pytest.mark.parametrize("budget", [4, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768])
def test_cli_budget_independent_of_context_and_prefill(context, budget):
    from santapp_ruler.cli import build_parser, _config_with_cli
    args = build_parser().parse_args(["run", "--config", str(ROOT / "configs/default.yaml"),
        "--context-length", str(context), "--samples-per-head", str(budget),
        "--prefill-backend", "triton", "--decode-backend", "torch", "--backends", "santapp", "hierarchical"])
    config = _config_with_cli(args)
    for name in ("santapp", "hierarchical"):
        assert getattr(config, name).samples_per_head == budget
        assert getattr(config, name).prefill_backend == "triton"
    assert config.benchmark.context_length == context


@pytest.mark.parametrize("budget", [0, -4, 3, 129])
def test_bad_budget_is_not_silently_rounded(budget):
    with pytest.raises(ValueError):
        load_config(ROOT / "configs/default.yaml", overrides=[f"santapp.samples_per_head={budget}"])


@pytest.mark.parametrize("cap,cuda,valid", [((8,0),"12.6",True), ((8,6),"12.6",True),
    ((8,9),"12.6",True), ((9,0),"12.6",True), ((7,5),"12.6",False),
    ((12,0),"12.6",False), ((12,0),"12.8",True), ((12,0),"13.0",True), ((8,0),None,False)])
def test_explicit_architecture_profiles(cap, cuda, valid):
    assert (device_compatibility_error(cap, cuda) is None) == valid


def test_torch_imports_do_not_import_triton():
    command = [sys.executable, "-c", "import sys; import santapp_ruler.attention.santapp; "
               "import santapp_ruler.attention.hierarchical; assert 'triton' not in sys.modules"]
    subprocess.run(command, env=dict(os.environ, PYTHONPATH=str(ROOT / "src")), check=True)


def test_all_kernel_compiler_signatures_match_source():
    directory = ROOT / "src/santapp_ruler/attention/triton_prefill/kernels"
    functions = {}
    for kind in ("kmeans", "teams"):
        tree = ast.parse((directory / f"p001_prefill_{kind}.py").read_text())
        functions[kind] = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    plan = compile_plan()
    assert len({(row.module, row.kernel) for row in plan}) == 16
    assert {x.kernel for x in plan if x.module == "kmeans"} == set(functions["kmeans"])
    for row in plan:
        node = functions[row.module][row.kernel]
        names = {arg.arg for arg in node.args.args}
        const = {arg.arg for arg in node.args.args if arg.annotation is not None}
        assert set(row.signature) | set(row.constants) == names
        assert set(row.constants) == const
        assert not (set(row.signature) & const)
        assert "samples_per_head" not in names


@pytest.mark.parametrize("policy", ["santapp", "hierarchical"])
def test_original_decode_smoke_with_cpu_reference_summary(policy):
    from santapp_ruler.attention.teams import build_team_summary
    from santapp_ruler.attention.contiguous_teams import build_contiguous_team_summary
    from santapp_ruler.attention.triton_prefill.smoke import check_decode_budgets
    keys = torch.randn(65, 8, generator=torch.Generator().manual_seed(88))
    if policy == "santapp":
        summary = build_team_summary(keys, torch.arange(65) // 16, parent_cluster_count=5, representatives_per_parent=4)
    else:
        summary = build_contiguous_team_summary(keys, parent_size=16, representatives_per_parent=4)
    rows = check_decode_budgets(summary, keys, keys.sin(), policy=policy,
                               budgets=[4,128,256,512,1024,2048,4096,8192,16384,32768])
    assert len(rows) == 20
    assert all(r["status"] == "passed" for r in rows)
    assert rows[-1]["take_all"] and rows[-1]["generated_exact_rows"] == 1


def test_triton_dispatch_preserves_original_generation_contract(monkeypatch):
    from test_santapp import _TinyQwen, _test_sdpa_forward
    from santapp_ruler.attention import qwen as qwen_module, prefill_runtime
    from santapp_ruler.attention.santapp import SantappEngine
    from santapp_ruler.attention.hierarchical import HierarchicalWholeTeamEngine
    from santapp_ruler.attention.teams import build_team_summary
    from santapp_ruler.attention.contiguous_teams import build_contiguous_team_summary
    from santapp_ruler.attention.triton_prefill import batched_kmeans, prefill_teams
    from santapp_ruler.config import SantappConfig, HierarchicalConfig, MiniBatchKMeansConfig
    from test_batched_kmeans import CPUOracleBatchedOps
    calls = []
    def mock_ops(x, precision):
        calls.append("kmeans")
        return CPUOracleBatchedOps()
    def generic(*args, **kwargs):
        calls.append("generic")
        return build_team_summary(*args, **kwargs)
    def contiguous(*args, **kwargs):
        calls.append("contiguous")
        return build_contiguous_team_summary(*args, **kwargs)
    monkeypatch.setattr(qwen_module, "hf_sdpa_attention_forward", _test_sdpa_forward)
    monkeypatch.setattr(prefill_runtime, "require_triton_environment", lambda *_: {})
    monkeypatch.setattr(batched_kmeans, "_make_batched_ops", mock_ops)
    monkeypatch.setattr(prefill_teams, "build_team_summary_triton", generic)
    monkeypatch.setattr(prefill_teams, "build_contiguous_team_summary_triton", contiguous)
    for klass, config in [(SantappEngine, SantappConfig(parent_size=4, representatives_per_parent=2,
                         samples_per_head=8, prefill_backend="triton",
                         kmeans=MiniBatchKMeansConfig(batch_size=8,max_iter=1))),
                        (HierarchicalWholeTeamEngine, HierarchicalConfig(parent_size=4,
                         representatives_per_parent=2,samples_per_head=8,prefill_backend="triton"))]:
        result = klass(_TinyQwen(), config).generate(torch.tensor([[1,2,3,4,5]]),
            max_new_tokens=2, eos_token_ids=set(),stop_on_eos=False,random_seed=11)
        assert result.metrics["team_builder_backend"] == "triton"
        assert result.metrics["mean_exact_tokens_per_gqa_group_call"] == 0.5
        assert result.metrics["requested_teams_per_head"] == 4
    assert calls == ["kmeans", "generic", "contiguous"]


def test_k8s_sweep_and_portable_gpu_filter(tmp_path):
    from test_k8s import _render, _documents, _env_values
    job = _documents(_render(tmp_path, "--prefill-backend", "triton", "--samples-per-head", "128", "512",
                            "--backends", "santapp", "hierarchical", "--max-new-tokens", "2"))[2]
    assert job["spec"]["completions"] == 8
    env = _env_values(job)
    assert env["SMOKE_SAMPLE_BUDGETS"] == "128,512"
    assert env["SMOKE_PREFILL_BACKEND"] == "triton" and env["SMOKE_MAX_NEW_TOKENS"] == "2"
    pod = job["spec"]["template"]["spec"]
    assert any(v["name"] == "jit-scratch" and "emptyDir" in v for v in pod["volumes"])
    products = pod["affinity"]["nodeAffinity"]["requiredDuringSchedulingIgnoredDuringExecution"]["nodeSelectorTerms"][0]["matchExpressions"][0]["values"]
    assert "NVIDIA-A10" in products
    assert "NVIDIA-TITAN-RTX" not in products and "Quadro-RTX-6000" not in products


def test_index_mapping_preserves_all_budgets_and_distinct_paths(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("r004_k8s", ROOT / "scripts/k8s_run_index.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    monkeypatch.setenv("RESULTS_ROOT", str(tmp_path))
    monkeypatch.setenv("SMOKE_CONTEXTS", "8192,32768")
    monkeypatch.setenv("SMOKE_BACKENDS", "santapp,hierarchical")
    monkeypatch.setenv("SMOKE_SAMPLE_BUDGETS", "128,512")
    monkeypatch.setenv("SMOKE_PREFILL_BACKEND", "triton")
    rows = []
    monkeypatch.setattr(module, "run_benchmark", lambda config, explicit_run_dir: rows.append((config, explicit_run_dir)))
    for i in range(8):
        monkeypatch.setattr(sys, "argv", ["script", "--index", str(i), "--config", str(ROOT / "configs/smoke.yaml")])
        assert module.main() == 0
    assert len({str(path) for _, path in rows}) == 8
    assert {(c.benchmark.context_length,c.santapp.samples_per_head,c.generation.backends[0]) for c,_ in rows} == {
        (n,s,p) for n in (8192,32768) for s in (128,512) for p in ("santapp","hierarchical")}
    assert all("prefill-triton" in str(p) for _, p in rows)


@pytest.mark.parametrize("action,profile,mode", [("build","cluster",None),
    ("build","local-5090",None),("run","cluster","compile"),("run","local-5090","quick")])
def test_docker_helper_argument_contract_without_docker(tmp_path,action,profile,mode):
    # This is deliberately a CLI stub, NOT a Docker build/GPU validation claim.
    bin_dir = tmp_path / "bin"; bin_dir.mkdir()
    stub = bin_dir / "docker"
    stub.write_text("#!/usr/bin/env python3\nimport json,os,sys\n"
        "with open(os.environ['DOCKER_STUB_LOG'],'a') as f: f.write(json.dumps(sys.argv[1:])+'\\n')\n"
        "if sys.argv[1:3] == ['image','inspect']: print('[]')\n")
    stub.chmod(0o755)
    log = tmp_path / "calls.jsonl"
    env = dict(os.environ, PATH=str(bin_dir)+os.pathsep+os.environ['PATH'],DOCKER_STUB_LOG=str(log))
    command = ["bash",str(ROOT / "scripts/docker_triton_smoke.sh"),action,"--profile",profile,
               "--artifacts",str(tmp_path / "artifacts"),"--hf-cache",str(tmp_path / "hf")]
    if mode: command += ["--","--mode",mode]
    subprocess.run(command,env=env,check=True,capture_output=True,text=True)
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    primary = next(c for c in calls if c[0] == action)
    if action == "build":
        assert ("LOCAL_CUDA128=1" if profile=="local-5090" else "LOCAL_CUDA128=0") in primary
    else:
        assert ("--gpus" in primary) == (mode != "compile")
        assert "--user" in primary
        assert "/workspace/scripts/smoke_triton_prefill.py" in primary
        mounts = [c for c in primary if c.startswith("type=bind")]
        assert len(mounts) == 2 and not any("dst=/workspace" in c for c in mounts)


def test_backend_and_budget_changes_protect_resume_identity():
    import copy
    from santapp_ruler.run_state import _result_identity
    original = load_config(ROOT / "configs/torch.yaml")
    native = copy.deepcopy(original); native.santapp.prefill_backend = "triton"
    other_budget = copy.deepcopy(native); other_budget.santapp.samples_per_head = 512
    assert _result_identity(original) != _result_identity(native)
    assert _result_identity(native) != _result_identity(other_budget)
