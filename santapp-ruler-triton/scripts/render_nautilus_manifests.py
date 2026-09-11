#!/usr/bin/env python3
"""Render owner-specific PVC, ConfigMap, indexed Job, and copy-pod YAML."""

from __future__ import annotations

import argparse
import copy
import re
import sys
from pathlib import Path
from typing import Any

import yaml

# Rendering is CPU-only and can use this checkout without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

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
    parser.add_argument("--parallelism", type=int, default=4)
    parser.add_argument("--cpu", default="4", help="Benchmark CPU request AND limit (equal).")
    parser.add_argument("--memory", default="32Gi", help="Benchmark host RAM request AND limit, not VRAM.")
    parser.add_argument("--staged-smoke", action="store_true",
                        help="Also generate separate kernel and HF-prefetch Jobs; apply in documented order.")
    parser.add_argument("--contexts", nargs="+", type=int, default=[8192, 32768])
    parser.add_argument(
        "--backends", nargs="+", choices=list(SUPPORTED_BACKENDS), default=list(SUPPORTED_BACKENDS)
    )
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--prefill-backend", choices=["torch", "triton"], default="triton")
    parser.add_argument("--samples-per-head", nargs="+", type=int, default=None,
                        help="Optional nominal S sweep; each budget gets separate result paths")
    parser.add_argument("--prompts-per-task", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=None,
                        help="Software-smoke override only; omit for official task output budgets")
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
    if args.prompts_per_task <= 0:
        raise ValueError("--prompts-per-task must be positive")
    if args.samples_per_head is not None:
        if len(set(args.samples_per_head)) != len(args.samples_per_head):
            raise ValueError("Sample budgets cannot contain duplicates")
        if any(s <= 0 or s % 4 for s in args.samples_per_head):
            raise ValueError("P=16/R=4 nominal budgets must be positive multiples of four")
    if args.max_new_tokens is not None and args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    # Validate the user-tunable resource quantities; equal limits prevent ratio drift.
    if not re.fullmatch(r"(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+)m?", args.cpu) or float(args.cpu.rstrip("m")) <= 0:
        raise ValueError("--cpu must be a positive CPU quantity, e.g. 4 or 500m")
    if not re.fullmatch(r"[1-9][0-9]*(?:Ki|Mi|Gi|Ti|K|M|G|T)?", args.memory):
        raise ValueError("--memory must be a positive RAM quantity, e.g. 32Gi")
    if args.staged_smoke and args.prefill_backend != "triton":
        raise ValueError("--staged-smoke includes native GPU gates; use --prefill-backend triton")
    owner = _slug(args.owner)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Do not leave stale all.yaml / stage manifests that could launch the wrong experiment.
    if any(output.iterdir()):
        raise ValueError("--output must be an empty/new directory; do not mix release manifests")
    prefix = f"{owner}-santa-ruler"
    pvc_name = f"{prefix}-pvc"
    config_name = f"{prefix}-config"
    job_name = f"{prefix}-smoke"
    copy_name = f"{prefix}-copy"
    labels = {"app.kubernetes.io/name": "santa-ruler", "benchmark-owner": owner}
    gpu_products = args.gpu_products or list(DEFAULT_GPU_PRODUCTS)
    if args.prefill_backend == "triton":
        # Original list contains two Turing cards, which are not Triton targets.
        unsupported = {"NVIDIA-TITAN-RTX", "Quadro-RTX-6000"}
        if args.gpu_products and unsupported.intersection(args.gpu_products):
            raise ValueError("Turing GPUs are not supported by native Triton; choose Ampere or newer")
        gpu_products = [p for p in gpu_products if p not in unsupported]
    completion_count = len(args.contexts) * len(args.backends) * len(args.samples_per_head or [None])

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
        {"name": "SMOKE_PREFILL_BACKEND", "value": args.prefill_backend},
        {"name": "SMOKE_PROMPTS_PER_TASK", "value": str(args.prompts_per_task)},
        {"name": "TRITON_CACHE_DIR", "value": "/tmp/santapp-runtime/triton"},
        {"name": "CUDA_CACHE_PATH", "value": "/tmp/santapp-runtime/cuda"},
        {"name": "TMPDIR", "value": "/tmp/santapp-runtime"},
        {"name": "PYTORCH_CUDA_ALLOC_CONF", "value": "expandable_segments:True"},
        # Indexed Jobs provide JOB_COMPLETION_INDEX directly to the container.
    ]
    if args.samples_per_head is not None:
        env.append({"name": "SMOKE_SAMPLE_BUDGETS", "value": ",".join(map(str, args.samples_per_head))})
    if args.max_new_tokens is not None:
        env.append({"name": "SMOKE_MAX_NEW_TOKENS", "value": str(args.max_new_tokens)})
    volume_mounts = [
        {"name": "jit-scratch", "mountPath": "/tmp/santapp-runtime"},
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
                "resources": {
                    "requests": {"cpu": "50m", "memory": "32Mi"},
                    "limits": {"cpu": "50m", "memory": "32Mi"},
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
                    "requests": {"cpu": args.cpu, "memory": args.memory, "nvidia.com/gpu": 1},
                    "limits": {"cpu": args.cpu, "memory": args.memory, "nvidia.com/gpu": 1},
                },
                "volumeMounts": volume_mounts,
            }
        ],
        "volumes": [
            {"name": "jit-scratch", "emptyDir": {}},
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
                    "resources": {"requests": {"cpu": "100m", "memory": "256Mi"}, "limits": {"cpu": "100m", "memory": "256Mi"}},
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
    if args.staged_smoke:
        # Clone the same policy-compliant pod setup; no separate hand-maintained YAML.
        kernel_job = copy.deepcopy(job)
        kernel_job["metadata"]["name"] = f"{prefix}-kernels"
        kernel_job["spec"] = {
            "backoffLimit": 0,
            "template": copy.deepcopy(job["spec"]["template"]),
        }
        kernel_pod = kernel_job["spec"]["template"]["spec"]
        kernel_container = kernel_pod["containers"][0]
        kernel_container["name"] = "smoke"
        kernel_container["args"] = [
            "/workspace/scripts/smoke_triton_prefill.py", "--mode", "matrix", "--run-tests",
            "--contexts", *map(str, args.contexts), "--sample-budgets",
            *map(str, args.samples_per_head or [128, 512]), "--dtypes", "float16",
            "--output-root", f"/shared/results/{owner}/triton-kernel-smoke",
        ]
        kernel_container["env"] = [e for e in env if e["name"] in {
            "TRITON_CACHE_DIR", "CUDA_CACHE_PATH", "TMPDIR", "PYTORCH_CUDA_ALLOC_CONF"}]
        kernel_container["resources"] = {
            "requests": {"cpu": "4", "memory": "16Gi", "nvidia.com/gpu": 1},
            "limits": {"cpu": "4", "memory": "16Gi", "nvidia.com/gpu": 1},
        }
        kernel_container["volumeMounts"] = [v for v in kernel_container["volumeMounts"] if v["name"] != "config"]
        kernel_pod["volumes"] = [v for v in kernel_pod["volumes"] if v["name"] != "config"]

        prefetch_job = copy.deepcopy(kernel_job)
        prefetch_job["metadata"]["name"] = f"{prefix}-prefetch"
        prefetch_job["spec"]["backoffLimit"] = 1
        prefetch_pod = prefetch_job["spec"]["template"]["spec"]
        prefetch_pod.pop("affinity", None)
        prefetch_container = prefetch_pod["containers"][0]
        prefetch_container["name"] = "prefetch"
        prefetch_container["args"] = [
            "/workspace/scripts/prefetch_hf.py", "--config", "/config/smoke.yaml",
            "--contexts", *map(str, args.contexts), "--tasks", *args.tasks,
        ]
        prefetch_container["env"] = [e for e in env if e["name"] in {"HF_HOME", "HF_DATASETS_CACHE"}]
        prefetch_container["resources"] = {
            "requests": {"cpu": "2", "memory": "4Gi"},
            "limits": {"cpu": "2", "memory": "4Gi"},
        }
        prefetch_container["volumeMounts"] = [v for v in volume_mounts if v["name"] != "jit-scratch"]
        prefetch_pod["volumes"] = [v for v in pod_spec["volumes"] if v["name"] != "jit-scratch"]
        documents = [
            ("00-pvc.yaml", pvc), ("01-configmap.yaml", config_map),
            ("02-triton-kernel-smoke-job.yaml", kernel_job),
            ("03-hf-prefetch-job.yaml", prefetch_job),
            ("04-ruler-smoke-job.yaml", job), ("05-copy-pod.yaml", copy_pod),
        ]
        (output / "APPLY_ORDER.txt").write_text(
            "Apply 00; wait for PVC Bound. Apply 01.\n"
            "Apply 02 and 03 (may run concurrently). Require both Jobs Complete.\n"
            "Then apply 04; require all completion indexes succeed. 05 is optional for copying.\n"
            "No all.yaml is generated: Kubernetes does not order independent Jobs automatically.\n"
            "Do not delete a PVC that holds earlier results. Use a fresh owner for distinct reruns.\n"
        )
    for filename, document in documents:
        _dump(output / filename, document)
    if not args.staged_smoke:
        (output / "all.yaml").write_text(
            "---\n".join(yaml.safe_dump(document, sort_keys=False) for _, document in documents),
            encoding="utf-8",
        )
    (output / "RUN_INFO.txt").write_text(
        f"owner={owner}\nnamespace={args.namespace}\njob={job_name}\ncopy_pod={copy_name}\n"
        f"pvc={pvc_name}\ncompletions={completion_count}\n"
        f"prefill_backend={args.prefill_backend}\nsample_budgets={args.samples_per_head}\n"
        f"prompts_per_task={args.prompts_per_task}\nmax_new_tokens_override={args.max_new_tokens}\n"
        f"image={args.image}\nparallelism={min(args.parallelism, completion_count)}\n"
        f"cpu_request_and_limit={args.cpu}\nhost_memory_request_and_limit={args.memory}\n"
        "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True\n"
        f"staged_smoke={args.staged_smoke}\n",
        encoding="utf-8",
    )
    print(f"Rendered {len(documents)} manifests under {output}")
    print(f"Indexed shards: {completion_count} ({len(args.contexts)} contexts x {len(args.samples_per_head or [None])} budgets x {len(args.backends)} backends)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
