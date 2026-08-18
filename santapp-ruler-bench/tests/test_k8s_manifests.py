import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
K8S_ROOT = REPO_ROOT / "k8s"
TARGET_PRODUCTS = [
    "Tesla-V100-SXM2-16GB",
    "Tesla-V100-PCIE-16GB",
    "NVIDIA-RTX-A4000",
]
FILES = [
    "00-santapp-ruler-pvc.yaml",
    "01-santapp-ruler-prefetch-job.yaml",
    "02-santapp-ruler-smoke-job.yaml",
    "03-santapp-ruler-indexed-job.yaml",
    "04-santapp-ruler-results-copy-pod.yaml",
]


def _load(root: Path, name: str) -> dict[str, Any]:
    value = yaml.safe_load((root / name).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _pod_spec(document: dict[str, Any]) -> dict[str, Any]:
    if document["kind"] == "Job":
        return document["spec"]["template"]["spec"]
    return document["spec"]


def _expression(spec: dict[str, Any], key: str) -> dict[str, Any]:
    terms = spec["affinity"]["nodeAffinity"][
        "requiredDuringSchedulingIgnoredDuringExecution"
    ]["nodeSelectorTerms"]
    matches = [
        expression
        for term in terms
        for expression in term.get("matchExpressions", [])
        if expression.get("key") == key
    ]
    assert len(matches) == 1
    return matches[0]


def _all_containers(document: dict[str, Any]) -> list[dict[str, Any]]:
    spec = _pod_spec(document)
    return [*spec.get("initContainers", []), *spec.get("containers", [])]


def _assert_no_hostname_exclusion(document: dict[str, Any]) -> None:
    if document["kind"] == "PersistentVolumeClaim":
        return
    spec = _pod_spec(document)
    affinity = spec.get("affinity", {}).get("nodeAffinity", {})
    required = affinity.get("requiredDuringSchedulingIgnoredDuringExecution", {})
    for term in required.get("nodeSelectorTerms", []):
        for expression in term.get("matchExpressions", []):
            assert expression.get("key") != "kubernetes.io/hostname"


def test_static_manifests_are_shareable_and_match_default_matrix():
    pvc, prefetch, smoke, job, copy_pod = [
        _load(K8S_ROOT, filename) for filename in FILES
    ]
    documents = [pvc, prefetch, smoke, job, copy_pod]

    for document in documents:
        assert document["metadata"]["namespace"] == "ucsb-opus-lab"
        assert document["metadata"]["labels"]["owner"] == "yourname"
        assert document["metadata"]["name"].startswith("yourname-santapp-ruler")
        _assert_no_hostname_exclusion(document)

    assert pvc["spec"]["storageClassName"] == "rook-cephfs"
    assert pvc["spec"]["accessModes"] == ["ReadWriteMany"]
    assert pvc["spec"]["resources"]["requests"]["storage"] == "50Gi"

    prefetch_container = _pod_spec(prefetch)["containers"][0]
    assert prefetch_container["resources"]["requests"] == prefetch_container[
        "resources"
    ]["limits"]
    assert not any(
        key.startswith("nvidia.com/")
        for key in prefetch_container["resources"]["requests"]
    )
    assert (
        "/workspace/k8s/matrices/default-2tasks-6methods-100p-8k.yaml"
        in prefetch_container["args"]
    )

    smoke_spec = _pod_spec(smoke)
    smoke_container = smoke_spec["containers"][0]
    assert smoke_container["resources"]["requests"]["nvidia.com/gpu"] == "1"
    assert "/workspace/configs/smoke_all_backends_8k.yaml" in smoke_container["args"]
    assert _expression(smoke_spec, "nvidia.com/gpu.product")["values"] == (
        TARGET_PRODUCTS
    )

    assert job["spec"]["completionMode"] == "Indexed"
    assert job["spec"]["completions"] == 12
    assert job["spec"]["parallelism"] == 5
    assert job["spec"]["backoffLimitPerIndex"] == 2
    assert job["spec"]["backoffLimit"] == 20
    assert job["spec"]["maxFailedIndexes"] == 3
    worker_spec = _pod_spec(job)
    worker = worker_spec["containers"][0]
    assert worker["resources"]["requests"] == worker["resources"]["limits"]
    assert worker["resources"]["requests"]["nvidia.com/gpu"] == "1"
    assert _expression(worker_spec, "nvidia.com/gpu.product")["values"] == (
        TARGET_PRODUCTS
    )
    assert _expression(worker_spec, "nvidia.com/gpu.sharing-strategy")["values"] == [
        "none"
    ]

    copy_container = _pod_spec(copy_pod)["containers"][0]
    assert not any(
        key.startswith("nvidia.com/")
        for key in copy_container["resources"]["requests"]
    )
    pvc_volume = next(
        volume
        for volume in _pod_spec(copy_pod)["volumes"]
        if volume["name"] == "results-vol"
    )
    assert pvc_volume["persistentVolumeClaim"] == {
        "claimName": "yourname-santapp-ruler-rwx",
        "readOnly": True,
    }

    for document in documents:
        containers = (
            _all_containers(document)
            if document["kind"] != "PersistentVolumeClaim"
            else []
        )
        for container in containers:
            assert container["image"] == "your-registry/santapp-ruler:latest"

    dockerfile = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "2.11.0-cuda12.6-cudnn9-runtime" in dockerfile


def test_renderer_updates_all_resources_and_derives_completion_count(tmp_path: Path):
    output = tmp_path / "rendered"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/render_nautilus_manifests.py",
            "--matrix",
            "k8s/matrices/default-2tasks-6methods-100p-8k.yaml",
            "--output-dir",
            str(output),
            "--resource-prefix",
            "example-test-render",
            "--owner",
            "example",
            "--image",
            "example.invalid/santapp:test",
            "--parallelism",
            "3",
            "--storage",
            "20Gi",
            "--gpu-product",
            "NVIDIA-RTX-A4000",
        ],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert sorted(path.name for path in output.glob("*.yaml")) == FILES

    pvc, prefetch, smoke, job, copy_pod = [
        _load(output, filename) for filename in FILES
    ]
    assert pvc["metadata"]["name"] == "example-test-render-rwx"
    assert pvc["spec"]["resources"]["requests"]["storage"] == "20Gi"
    assert prefetch["metadata"]["name"] == "example-test-render-prefetch"
    assert smoke["metadata"]["name"] == "example-test-render-smoke"
    assert job["metadata"]["name"] == "example-test-render-sweep"
    assert copy_pod["metadata"]["name"] == "example-test-render-results-copy"
    assert job["spec"]["completions"] == 12
    assert job["spec"]["parallelism"] == 3

    for document in (pvc, prefetch, smoke, job, copy_pod):
        assert document["metadata"]["labels"]["owner"] == "example"
        assert document["metadata"]["labels"]["app.kubernetes.io/name"] == (
            "example-test-render"
        )
        _assert_no_hostname_exclusion(document)

    for document in (prefetch, smoke, job, copy_pod):
        for container in _all_containers(document):
            assert container["image"] == "example.invalid/santapp:test"
            rendered = yaml.safe_dump(container)
            assert "/mnt/results/example-test-render" in rendered or (
                container.get("name") == "copy"
                and "/mnt/results/example-test-render" in rendered
            )
    assert _expression(_pod_spec(job), "nvidia.com/gpu.product")["values"] == [
        "NVIDIA-RTX-A4000"
    ]


def test_default_matrix_and_smoke_config_have_six_methods_and_parent_size_16():
    matrix = yaml.safe_load(
        (K8S_ROOT / "matrices/default-2tasks-6methods-100p-8k.yaml").read_text(
            encoding="utf-8"
        )
    )
    smoke = yaml.safe_load(
        (REPO_ROOT / "configs/smoke_all_backends_8k.yaml").read_text(
            encoding="utf-8"
        )
    )
    expected = [
        "sdpa",
        "santa",
        "santapp",
        "santapp_gumbel_topk",
        "team_sampler",
        "team_gumbel_topk",
    ]
    assert matrix["tasks"] == ["niah_multiquery", "niah_multivalue"]
    assert matrix["prompts_per_task"] == 100
    assert [setting["backend"] for setting in matrix["settings"]] == expected
    assert smoke["generation"]["backends"] == expected
    assert smoke["benchmark"]["prompts_per_task"] == 1
    assert smoke["generation"]["max_new_tokens"] == 4
    parent_sizes = [
        setting["group_size"]
        for setting in matrix["settings"]
        if "group_size" in setting
    ]
    assert parent_sizes == [16, 16]
