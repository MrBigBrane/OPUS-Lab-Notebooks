#!/usr/bin/env python3
"""Render owner-specific PVC, ConfigMap, indexed Job, and copy-pod YAML."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import yaml

from santapp_ruler.config import SUPPORTED_BACKENDS
from santapp_ruler.ruler.tasks import DEFAULT_TASKS

DEFAULT_GPU_PRODUCTS = (
    "NVIDIA-A10",
    "NVIDIA-RTX-A5000",
    "NVIDIA-GeForce-RTX-3090",
    "NVIDIA-GeForce-RTX-4090",
    "NVIDIA-TITAN-RTX",
    "Quadro-RTX-6000",
    "NVIDIA-RTX-A6000",
    "NVIDIA-A40",
    "NVIDIA-L40",
    "NVIDIA-L40S",
    "NVIDIA-A100-PCIE-40GB",
    "NVIDIA-A100-SXM4-40GB",
    "NVIDIA-A100-80GB-PCIe",
    "NVIDIA-H100-80GB-HBM3",
)


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9-]+", "-", value.lower()).strip("-")
    cleaned = re.sub(r"-+", "-", cleaned)
    if not cleaned or len(cleaned) > 35:
        raise ValueError("--owner must produce a nonempty DNS slug of at most 35 characters.")
    return cleaned


def _dump(path: Path, document: Any) -> None:
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--owner", required=True, help="Unique colleague slug, e.g. alice.")
    parser.add_argument("--image", required=True, help="Built image, including immutable tag.")
    parser.add_argument("--namespace", default="ucsb-opus-lab")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--storage-class", default="rook-cephfs")
    parser.add_argument("--pvc-size", default="50Gi")
    parser.add_argument("--parallelism", type=int, default=1)
    parser.add_argument("--contexts", nargs="+", type=int, default=[8192, 32768])
    parser.add_argument(
        "--backends", nargs="+", choices=list(SUPPORTED_BACKENDS), default=list(SUPPORTED_BACKENDS)
    )
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument(
        "--gpu-product",
        action="append",
        dest="gpu_products",
        help="Allowed nvidia.com/gpu.product value; repeat to replace the default 24GB+ list.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.parallelism <= 0:
        raise ValueError("--parallelism must be positive.")
    owner = _slug(args.owner)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    prefix = f"{owner}-santa-ruler"
    pvc_name = f"{prefix}-pvc"
    config_name = f"{prefix}-config"
    job_name = f"{prefix}-smoke"
    copy_name = f"{prefix}-copy"
    labels = {"app.kubernetes.io/name": "santa-ruler", "benchmark-owner": owner}
    gpu_products = args.gpu_products or list(DEFAULT_GPU_PRODUCTS)
    completion_count = len(args.contexts) * len(args.backends)

    smoke_config = (Path(__file__).resolve().parents[1] / "configs" / "smoke.yaml").read_text(
        encoding="utf-8"
    )
    pvc = {
        "apiVersion": "v1",
        "kind": "PersistentVolumeClaim",
        "metadata": {"name": pvc_name, "namespace": args.namespace, "labels": labels},
        "spec": {
            "accessModes": ["ReadWriteMany"],
            "storageClassName": args.storage_class,
            "resources": {"requests": {"storage": args.pvc_size}},
        },
    }
    config_map = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {"name": config_name, "namespace": args.namespace, "labels": labels},
        "data": {"smoke.yaml": smoke_config},
    }
    env = [
        {"name": "OWNER_SLUG", "value": owner},
        {"name": "RESULTS_ROOT", "value": "/shared/results"},
        {"name": "SMOKE_CONTEXTS", "value": ",".join(map(str, args.contexts))},
        {"name": "SMOKE_BACKENDS", "value": ",".join(args.backends)},
        {"name": "SMOKE_TASKS", "value": ",".join(args.tasks)},
        {"name": "HF_HOME", "value": "/shared/hf"},
        {"name": "HF_DATASETS_CACHE", "value": "/shared/hf/datasets"},
        # Indexed Jobs provide JOB_COMPLETION_INDEX directly to the container.
    ]
    volume_mounts = [
        {"name": "shared", "mountPath": "/shared"},
        {"name": "config", "mountPath": "/config", "readOnly": True},
    ]
    pod_spec = {
        "restartPolicy": "Never",
        "terminationGracePeriodSeconds": 30,
        "securityContext": {
            "runAsNonRoot": True,
            "runAsUser": 10001,
            "runAsGroup": 10001,
            "fsGroup": 10001,
            "fsGroupChangePolicy": "OnRootMismatch",
        },
        # CephFS volume roots are commonly root-owned. Prepare only the shared
        # writable roots, then run the benchmark itself as the unprivileged UID.
        "initContainers": [
            {
                "name": "prepare-shared-permissions",
                "image": args.image,
                "imagePullPolicy": "IfNotPresent",
                "command": ["bash", "-lc"],
                "args": [
                    "set -euo pipefail; "
                    "mkdir -p /shared/results /shared/hf; "
                    "chown 10001:10001 /shared/results /shared/hf; "
                    "chmod 2775 /shared/results /shared/hf; "
                    "ls -ld /shared /shared/results /shared/hf"
                ],
                "securityContext": {
                    "runAsNonRoot": False,
                    "runAsUser": 0,
                    "runAsGroup": 0,
                    "allowPrivilegeEscalation": False,
                },
                "volumeMounts": [{"name": "shared", "mountPath": "/shared"}],
            }
        ],
        "affinity": {
            "nodeAffinity": {
                "requiredDuringSchedulingIgnoredDuringExecution": {
                    "nodeSelectorTerms": [
                        {"matchExpressions": [{"key": "nvidia.com/gpu.product", "operator": "In", "values": gpu_products}]}
                    ]
                }
            }
        },
        "containers": [
            {
                "name": "benchmark",
                "image": args.image,
                "imagePullPolicy": "IfNotPresent",
                "workingDir": "/workspace",
                "command": ["python"],
                "args": ["/workspace/scripts/k8s_run_index.py", "--config", "/config/smoke.yaml"],
                "env": env,
                "resources": {
                    "requests": {"cpu": "4", "memory": "32Gi", "nvidia.com/gpu": 1},
                    "limits": {"cpu": "12", "memory": "64Gi", "nvidia.com/gpu": 1},
                },
                "volumeMounts": volume_mounts,
            }
        ],
        "volumes": [
            {"name": "shared", "persistentVolumeClaim": {"claimName": pvc_name}},
            {"name": "config", "configMap": {"name": config_name}},
        ],
    }
    job = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {"name": job_name, "namespace": args.namespace, "labels": labels},
        "spec": {
            "completionMode": "Indexed",
            "completions": completion_count,
            "parallelism": min(args.parallelism, completion_count),
            "backoffLimit": 1,
            "template": {"metadata": {"labels": labels}, "spec": pod_spec},
        },
    }
    copy_pod = {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {"name": copy_name, "namespace": args.namespace, "labels": labels},
        "spec": {
            "restartPolicy": "Never",
            "securityContext": {
                "runAsNonRoot": True,
                "runAsUser": 10001,
                "runAsGroup": 10001,
                "fsGroup": 10001,
                "fsGroupChangePolicy": "OnRootMismatch",
            },
            "containers": [
                {
                    "name": "copy",
                    "image": args.image,
                    "imagePullPolicy": "IfNotPresent",
                    "command": ["bash", "-lc"],
                    "args": ["echo 'Results mounted at /shared/results'; sleep infinity"],
                    "resources": {"requests": {"cpu": "100m", "memory": "256Mi"}, "limits": {"cpu": "1", "memory": "1Gi"}},
                    "volumeMounts": [{"name": "shared", "mountPath": "/shared"}],
                }
            ],
            "volumes": [{"name": "shared", "persistentVolumeClaim": {"claimName": pvc_name}}],
        },
    }

    documents = [
        ("00-pvc.yaml", pvc),
        ("01-configmap.yaml", config_map),
        ("02-smoke-job.yaml", job),
        ("03-copy-pod.yaml", copy_pod),
    ]
    for filename, document in documents:
        _dump(output / filename, document)
    (output / "all.yaml").write_text(
        "---\n".join(yaml.safe_dump(document, sort_keys=False) for _, document in documents),
        encoding="utf-8",
    )
    (output / "RUN_INFO.txt").write_text(
        f"owner={owner}\nnamespace={args.namespace}\njob={job_name}\ncopy_pod={copy_name}\n"
        f"pvc={pvc_name}\ncompletions={completion_count}\n",
        encoding="utf-8",
    )
    print(f"Rendered {len(documents)} manifests plus all.yaml under {output}")
    print(f"Indexed smoke shards: {completion_count} ({len(args.contexts)} contexts x {len(args.backends)} backends)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
