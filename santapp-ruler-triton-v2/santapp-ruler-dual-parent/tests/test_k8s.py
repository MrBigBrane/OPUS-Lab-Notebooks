from pathlib import Path
import os
import subprocess
import sys

import yaml

from santapp_ruler.ruler.tasks import DEFAULT_TASKS


ROOT = Path(__file__).resolve().parents[1]
RENDERER = ROOT / "scripts" / "render_nautilus_manifests.py"


def _render(tmp_path: Path, *extra: str) -> Path:
    output = tmp_path / "rendered"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    command = [
        sys.executable,
        str(RENDERER),
        "--owner",
        "researcher-one",
        "--image",
        "registry.example/project/image:immutable",
        "--namespace",
        "shared-namespace",
        "--output",
        str(output),
        *extra,
    ]
    subprocess.run(command, cwd=ROOT, env=env, check=True, capture_output=True, text=True)
    return output


def _documents(output: Path):
    return list(yaml.safe_load_all((output / "all.yaml").read_text(encoding="utf-8")))


def _env_values(job: dict) -> dict[str, str]:
    env = job["spec"]["template"]["spec"]["containers"][0]["env"]
    return {item["name"]: item.get("value", "") for item in env}


def test_renderer_creates_four_resource_types(tmp_path: Path) -> None:
    docs = _documents(_render(tmp_path))
    assert [doc["kind"] for doc in docs] == [
        "PersistentVolumeClaim",
        "ConfigMap",
        "Job",
        "Pod",
    ]


def test_resource_names_and_labels_are_owner_specific(tmp_path: Path) -> None:
    docs = _documents(_render(tmp_path))
    for doc in docs:
        assert doc["metadata"]["name"].startswith("researcher-one-santa-ruler-")
        assert doc["metadata"]["namespace"] == "shared-namespace"
        assert doc["metadata"]["labels"]["benchmark-owner"] == "researcher-one"


def test_default_job_is_two_contexts_by_four_backends(tmp_path: Path) -> None:
    job = _documents(_render(tmp_path))[2]
    assert job["spec"]["completionMode"] == "Indexed"
    assert job["spec"]["completions"] == 8
    env = _env_values(job)
    assert env["SMOKE_CONTEXTS"] == "8192,32768"
    assert env["SMOKE_BACKENDS"] == "sdpa,santa,santapp,hierarchical"


def test_each_shard_receives_all_thirteen_tasks(tmp_path: Path) -> None:
    job = _documents(_render(tmp_path))[2]
    tasks = _env_values(job)["SMOKE_TASKS"].split(",")
    assert tasks == list(DEFAULT_TASKS)
    assert len(tasks) == 13


def test_gpu_allow_list_drops_sub_24gb_classes(tmp_path: Path) -> None:
    job = _documents(_render(tmp_path))[2]
    expression = job["spec"]["template"]["spec"]["affinity"]["nodeAffinity"][
        "requiredDuringSchedulingIgnoredDuringExecution"
    ]["nodeSelectorTerms"][0]["matchExpressions"][0]
    products = expression["values"]
    assert "NVIDIA-A10" in products
    assert "NVIDIA-RTX-A5000" in products
    assert "NVIDIA-RTX-A4000" not in products
    assert "NVIDIA-L4" not in products
    assert "Tesla-V100-PCIE-16GB" not in products
    assert "NVIDIA-RTX-4000-Ada-Generation" not in products



def test_job_prepares_cephfs_writable_roots_as_root(tmp_path: Path) -> None:
    job = _documents(_render(tmp_path))[2]
    pod_spec = job["spec"]["template"]["spec"]
    assert pod_spec["securityContext"]["runAsUser"] == 10001
    assert pod_spec["securityContext"]["fsGroupChangePolicy"] == "OnRootMismatch"
    init = pod_spec["initContainers"][0]
    assert init["name"] == "prepare-shared-permissions"
    assert init["securityContext"]["runAsUser"] == 0
    command = init["args"][0]
    assert "mkdir -p /shared/results /shared/hf" in command
    assert "chown 10001:10001 /shared/results /shared/hf" in command

def test_copy_pod_requests_no_gpu(tmp_path: Path) -> None:
    copy_pod = _documents(_render(tmp_path))[3]
    resources = copy_pod["spec"]["containers"][0]["resources"]
    assert "nvidia.com/gpu" not in resources["requests"]
    assert "nvidia.com/gpu" not in resources["limits"]


def test_custom_matrix_changes_completion_count(tmp_path: Path) -> None:
    job = _documents(
        _render(
            tmp_path,
            "--contexts",
            "8192",
            "--backends",
            "santapp",
            "hierarchical",
        )
    )[2]
    assert job["spec"]["completions"] == 2
    assert _env_values(job)["SMOKE_BACKENDS"] == "santapp,hierarchical"


def test_invalid_owner_slug_fails(tmp_path: Path) -> None:
    output = tmp_path / "rendered"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    result = subprocess.run(
        [
            sys.executable,
            str(RENDERER),
            "--owner",
            "!!!",
            "--image",
            "example/image:tag",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
