"""R005 packaging/deployment regression checks; these do not execute a GPU."""
from __future__ import annotations

from dataclasses import asdict
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from santapp_ruler.config import load_config
from santapp_ruler.runtime_defaults import configure_allocator
from santapp_ruler.attention.triton_prefill.known_issues import is_known_a10_d7_assignment_case

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("filename", [None, "default.yaml", "smoke.yaml", "triton.yaml"])
def test_all_entry_configs_default_native(filename):
    c = load_config(ROOT / "configs" / filename if filename else None)
    assert c.santapp.prefill_backend == c.hierarchical.prefill_backend == "triton"


def test_explicit_torch_reference_and_cli_override():
    from santapp_ruler.cli import build_parser, _config_with_cli
    expected = load_config(ROOT / "configs/torch.yaml")
    actual = _config_with_cli(build_parser().parse_args(["run", "--config", str(ROOT / "configs/default.yaml"),
                                                       "--prefill-backend", "torch", "--decode-backend", "torch"]))
    assert asdict(actual) == asdict(expected)
    assert asdict(load_config(ROOT / "configs/default.yaml")) == asdict(load_config(ROOT / "configs/triton.yaml"))


@pytest.mark.parametrize("env", [{}, {"OTHER": "yes"}, {"PYTORCH_CUDA_ALLOC_CONF": "backend:cudaMallocAsync"},
    {"PYTORCH_ALLOC_CONF": "expandable_segments:False"}, {"PYTORCH_CUDA_ALLOC_CONF": ""},
    {"PYTORCH_ALLOC_CONF": "x", "PYTORCH_CUDA_ALLOC_CONF": "y"}])
def test_allocator_defaults_preserve_explicit_environment(env):
    original = dict(env)
    result = configure_allocator(env)
    if any(k in original for k in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF")):
        assert env == original
    else:
        assert env["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
    assert set(result) == {"PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF"}


def test_cli_sets_allocator_before_torch_and_library_import_has_no_side_effect():
    env = {k:v for k,v in os.environ.items() if k not in {"PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF"}}
    env["PYTHONPATH"] = str(ROOT / "src")
    code = ("import os,sys; import santapp_ruler; "
            "assert 'PYTORCH_CUDA_ALLOC_CONF' not in os.environ; "
            "from santapp_ruler.cli import main; main(['list-tasks']); "
            "assert os.environ['PYTORCH_CUDA_ALLOC_CONF']=='expandable_segments:True'; "
            "assert 'torch' not in sys.modules")
    subprocess.run([sys.executable,"-c",code],env=env,check=True,capture_output=True,text=True)


@pytest.mark.parametrize("name,cap,version,d,expected", [
    ("NVIDIA A10",(8,6),"3.6.0",7,True), ("NVIDIA-A10",(8,6),"3.6.0+abc",7,True),
    ("NVIDIA A10",(8,6),"3.6.0",128,False), ("NVIDIA A10",(8,6),"3.7.0",7,False),
    ("NVIDIA GeForce RTX 3090",(8,6),"3.6.0",7,False), ("NVIDIA A100",(8,0),"3.6.0",7,False),
    ("NVIDIA GeForce RTX 5090 Laptop GPU",(12,0),"3.6.0",7,False),
])
def test_known_issue_is_narrow_not_an_ampere_wide_test_bypass(name,cap,version,d,expected):
    assert is_known_a10_d7_assignment_case(name,cap,version,d) == expected


def _pods(doc):
    if doc["kind"] == "Job":
        yield doc["spec"]["template"]["spec"]
    elif doc["kind"] == "Pod":
        yield doc["spec"]


def _check_resources(docs):
    for doc in docs:
        for pod in _pods(doc):
            for c in pod.get("initContainers",[]) + pod.get("containers",[]):
                res=c["resources"]
                assert res["requests"] == res["limits"], (doc["metadata"]["name"], c["name"])
                assert set(res["requests"]) >= {"cpu","memory"}


def test_renderer_equal_resources_allocator_and_native_default(tmp_path):
    from test_k8s import _render,_documents,_env_values
    docs = _documents(_render(tmp_path))
    _check_resources(docs)
    job=docs[2];env=_env_values(job)
    assert job["spec"]["parallelism"] == 4
    assert env["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
    assert env["SMOKE_PREFILL_BACKEND"] == "triton"
    assert job["spec"]["template"]["spec"]["containers"][0]["resources"]["requests"] == {
        "cpu":"4", "memory":"32Gi", "nvidia.com/gpu":1}


def test_tunable_resources_stay_equal(tmp_path):
    from test_k8s import _render,_documents
    docs=_documents(_render(tmp_path,"--cpu","2","--memory","48Gi"))
    _check_resources(docs)
    assert docs[2]["spec"]["template"]["spec"]["containers"][0]["resources"]["limits"]["memory"] == "48Gi"


def test_staged_smoke_is_generated_not_stale_and_has_correct_resources(tmp_path):
    from test_k8s import _render
    d=_render(tmp_path,"--staged-smoke","--backends","santapp","hierarchical",
              "--samples-per-head","128","512","--tasks","niah_single_1",
              "--max-new-tokens","2","--gpu-product","NVIDIA-A10")
    docs={p.name:yaml.safe_load(p.read_text()) for p in d.glob("*.yaml")}
    assert len(docs)==6 and not (d/"all.yaml").exists()
    _check_resources(docs.values())
    job=docs["04-ruler-smoke-job.yaml"]
    assert job["spec"]["completions"]==8 and job["spec"]["parallelism"]==4
    kernel=docs["02-triton-kernel-smoke-job.yaml"]["spec"]["template"]["spec"]
    assert "--run-tests" in kernel["containers"][0]["args"]
    assert kernel["affinity"]["nodeAffinity"]["requiredDuringSchedulingIgnoredDuringExecution"]["nodeSelectorTerms"][0]["matchExpressions"][0]["values"]==["NVIDIA-A10"]
    prefetch=docs["03-hf-prefetch-job.yaml"]["spec"]["template"]["spec"]
    assert "affinity" not in prefetch
    assert "nvidia.com/gpu" not in prefetch["containers"][0]["resources"]["requests"]
    assert prefetch["containers"][0]["args"][0]=="/workspace/scripts/prefetch_hf.py"
    for pod in (kernel,prefetch,job["spec"]["template"]["spec"]):
        volumes={v["name"] for v in pod["volumes"]}
        for c in pod["initContainers"]+pod["containers"]:
            assert {m["name"] for m in c["volumeMounts"]} <= volumes


def test_checked_in_example_has_no_legacy_resources():
    docs=list(yaml.safe_load_all((ROOT/"k8s/example/all.yaml").read_text()))
    _check_resources(docs)
    from test_k8s import _env_values
    assert _env_values(docs[2])["PYTORCH_CUDA_ALLOC_CONF"]=="expandable_segments:True"
    assert _env_values(docs[2])["SMOKE_PREFILL_BACKEND"]=="triton"


def test_image_contains_allocator_default():
    assert "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" in (ROOT/"Dockerfile").read_text()
    assert "santapp-ruler:r005-$suffix" in (ROOT/"scripts/docker_triton_smoke.sh").read_text()


def test_legacy_saved_config_without_dispatch_fields_is_not_reinterpreted_as_triton(tmp_path):
    from santapp_ruler.run_state import prepare_run_directory
    c=load_config(ROOT/"configs/torch.yaml")
    old=asdict(c)
    for p in ("santapp","hierarchical"):
        old[p].pop("prefill_backend")
    path=tmp_path/"config.resolved.yaml"
    path.write_text(yaml.safe_dump(old))
    prepare_run_directory(c,tmp_path)
    with pytest.raises(ValueError,match="different resolved"):
        prepare_run_directory(load_config(ROOT/"configs/default.yaml"),tmp_path)
    assert "prefill_backend" not in path.read_text()  # No mutation of old experiment metadata.


def test_native_resume_still_succeeds(tmp_path):
    from santapp_ruler.run_state import prepare_run_directory
    c=load_config(ROOT/"configs/default.yaml")
    prepare_run_directory(c,tmp_path)
    prepare_run_directory(c,tmp_path)


def test_prefetch_uses_configured_model_and_each_context_without_gpu(tmp_path,monkeypatch):
    from types import SimpleNamespace
    path=ROOT/"scripts/prefetch_hf.py"
    spec=importlib.util.spec_from_file_location("release_prefetch",path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    calls=[]
    monkeypatch.setitem(sys.modules,"huggingface_hub",SimpleNamespace(snapshot_download=lambda **kw:calls.append(("model",kw))))
    def dataset(repo,**kw):
        calls.append(("data",repo,kw));return [0]*2
    monkeypatch.setitem(sys.modules,"datasets",SimpleNamespace(load_dataset=dataset))
    monkeypatch.setattr(sys,"argv",[str(path),"--config",str(ROOT/"configs/smoke.yaml"),
                                  "--contexts","8192","32768","--tasks","niah_single_1"])
    assert module.main()==0
    assert calls[0][1]==dict(repo_id="Qwen/Qwen2.5-7B-Instruct",revision="a09a35458c702b33eeacc393d103063234e8bc28")
    assert len(calls)==3 and "8192" in calls[1][1] and "32768" in calls[2][1]


def test_default_index_script_no_flag_selects_native(tmp_path,monkeypatch):
    spec=importlib.util.spec_from_file_location("release_index",ROOT/"scripts/k8s_run_index.py")
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    for key in list(os.environ):
        if key.startswith("SMOKE_") or key=="JOB_COMPLETION_INDEX":monkeypatch.delenv(key,raising=False)
    monkeypatch.setenv("RESULTS_ROOT",str(tmp_path));monkeypatch.setenv("OWNER_SLUG","test")
    rows=[]
    monkeypatch.setattr(mod,"run_benchmark",lambda c,explicit_run_dir:rows.append((c,explicit_run_dir)))
    monkeypatch.setattr(sys,"argv",["script","--index","2","--config",str(ROOT/"configs/smoke.yaml")])
    assert mod.main()==0
    c,d=rows[0]
    assert c.santapp.prefill_backend=="triton"
    assert "prefill-triton" in str(d)
    assert json.loads((d/"shard.json").read_text())["prefill_backend"]=="triton"


def test_doctor_checks_default_native_or_explicit_reference():
    from santapp_ruler.cli import build_parser
    assert build_parser().parse_args(["doctor"]).prefill_backend == "triton"
    assert build_parser().parse_args(["doctor", "--prefill-backend", "torch"]).prefill_backend == "torch"
