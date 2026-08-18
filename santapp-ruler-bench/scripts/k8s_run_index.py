#!/usr/bin/env python3
"""Run one resumable setting/task shard selected by a Kubernetes Job index."""

from __future__ import annotations

import argparse
import hashlib
import os
import platform
import socket
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

from santapp_ruler.config import save_config
from santapp_ruler.io_utils import atomic_write_json, repair_trailing_partial_jsonl
from santapp_ruler.k8s_matrix import (
    MatrixError,
    apply_work_item,
    cache_fingerprint,
    load_matrix,
)
from santapp_ruler.k8s_runtime import (
    CacheNotReadyError,
    KubernetesPaths,
    compact_utc_now,
    configure_huggingface_environment,
    exclusive_lock,
    execution_fingerprint,
    read_json_object,
    require_cache_marker,
    utc_now,
)
from santapp_ruler.reporting import read_jsonl
from santapp_ruler.runner import run_benchmark


class PermanentWorkerError(RuntimeError):
    """An unchanged retry cannot repair this shard."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix",
        type=Path,
        default=Path("k8s/matrices/default-2tasks-6methods-100p-8k.yaml"),
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path(os.environ.get("RESULTS_ROOT", "/mnt/results/yourname-santapp-ruler")),
    )
    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help="Defaults to JOB_COMPLETION_INDEX from the Indexed Job.",
    )
    return parser


def _completion_index(explicit: int | None) -> int:
    if explicit is not None:
        return explicit
    raw = os.environ.get("JOB_COMPLETION_INDEX")
    if raw is None:
        raise PermanentWorkerError(
            "JOB_COMPLETION_INDEX is unset; pass --index outside Kubernetes."
        )
    try:
        return int(raw)
    except ValueError as exc:
        raise PermanentWorkerError(
            f"Invalid JOB_COMPLETION_INDEX={raw!r}."
        ) from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _runtime_metadata() -> dict[str, Any]:
    result: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "pod_name": os.environ.get("POD_NAME"),
        "pod_uid": os.environ.get("POD_UID"),
        "node_name": os.environ.get("NODE_NAME"),
        "namespace": os.environ.get("POD_NAMESPACE"),
        "job_name": os.environ.get("JOB_NAME"),
        "container_image": os.environ.get("CONTAINER_IMAGE"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    try:
        import torch

        result.update(
            {
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(),
                "cuda_device_count": torch.cuda.device_count(),
                "compiled_cuda_arches": torch.cuda.get_arch_list(),
            }
        )
        if torch.cuda.is_available():
            properties = torch.cuda.get_device_properties(0)
            result.update(
                {
                    "gpu_name": properties.name,
                    "gpu_total_gib": properties.total_memory / (1024**3),
                    "cuda_capability": list(torch.cuda.get_device_capability(0)),
                }
            )
            try:
                probe = torch.arange(256, dtype=torch.float16, device="cuda")
                probe = (probe + probe).sum()
                torch.cuda.synchronize()
                result["cuda_kernel_probe"] = float(probe.item())
            except Exception as exc:
                result["cuda_kernel_probe_error"] = (
                    f"{type(exc).__name__}: {exc}"
                )
    except Exception as exc:
        result["torch_probe_error"] = f"{type(exc).__name__}: {exc}"
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version,uuid",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
        output = completed.stdout.strip()
        result["nvidia_smi"] = output or completed.stderr.strip()
        if completed.returncode == 0 and output:
            fields = [value.strip() for value in output.splitlines()[0].split(",")]
            if len(fields) == 4:
                result.update(
                    {
                        "nvidia_smi_gpu_name": fields[0],
                        "nvidia_smi_memory_total_mib": fields[1],
                        "nvidia_driver_version": fields[2],
                        "gpu_uuid": fields[3],
                    }
                )
    except Exception as exc:
        result["nvidia_smi_error"] = f"{type(exc).__name__}: {exc}"
    return result


def _validate_cuda_runtime(runtime: dict[str, Any]) -> None:
    """Fail one index immediately when the allocated GPU cannot run this image."""

    if not runtime.get("cuda_available"):
        raise PermanentWorkerError(
            "CUDA is unavailable inside the GPU worker container."
        )
    if runtime.get("cuda_device_count", 0) < 1:
        raise PermanentWorkerError("No CUDA device is visible to the worker.")
    if runtime.get("cuda_kernel_probe_error"):
        raise PermanentWorkerError(
            "The PyTorch/CUDA image cannot execute a kernel on the allocated GPU: "
            + str(runtime["cuda_kernel_probe_error"])
        )
    capability = runtime.get("cuda_capability")
    arches = runtime.get("compiled_cuda_arches")
    if isinstance(capability, list) and len(capability) == 2 and isinstance(arches, list):
        current = f"sm_{int(capability[0])}{int(capability[1])}"
        # The actual kernel probe is authoritative. This explicit check makes a
        # missing native target visible in provenance without rejecting newer
        # GPUs that execute successfully through forward-compatible PTX.
        runtime["native_cuda_arch_present"] = current in arches


def _prediction_path(run_dir: Path, backend: str, task: str) -> Path:
    return run_dir / "predictions" / backend / f"{task}.jsonl"


def _validate_records(
    *,
    prediction_path: Path,
    selected_manifest: Path,
    expected_uids: list[str],
    expected_count: int,
    setting: Any,
    task: str,
) -> list[dict[str, Any]]:
    repair_trailing_partial_jsonl(prediction_path)
    records = read_jsonl(prediction_path)
    if len(records) != expected_count:
        raise PermanentWorkerError(
            f"Expected {expected_count} records at {prediction_path}, found "
            f"{len(records)} after benchmark completion."
        )
    actual_uids = [str(record.get("uid")) for record in records]
    if actual_uids != expected_uids:
        raise PermanentWorkerError(
            f"Prediction UID order differs from the pinned selection for {task}."
        )
    if len(set(actual_uids)) != len(actual_uids):
        raise PermanentWorkerError(f"Duplicate UIDs in {prediction_path}.")
    if any("input" in record for record in records):
        raise PermanentWorkerError(
            f"Compact Kubernetes output unexpectedly contains prompt text: "
            f"{prediction_path}"
        )
    if not selected_manifest.is_file():
        raise PermanentWorkerError(f"Missing selection manifest: {selected_manifest}")
    manifest = read_jsonl(selected_manifest)
    if [str(row.get("uid")) for row in manifest] != expected_uids:
        raise PermanentWorkerError(
            "Run selection manifest differs from cache-ready selected UIDs."
        )
    if any("input" in row for row in manifest):
        raise PermanentWorkerError(
            "Kubernetes selection manifest unexpectedly contains prompt text."
        )

    expected_budget = setting.samples_per_head
    for record in records:
        if record.get("task") != task:
            raise PermanentWorkerError("Prediction record has the wrong task.")
        metrics = record.get("metrics")
        if not isinstance(metrics, dict):
            raise PermanentWorkerError("Prediction record has no metrics mapping.")
        if metrics.get("prompt_exact_tail_tokens") != 0:
            raise PermanentWorkerError("A fixed exact prompt tail reappeared.")
        if expected_budget is not None and metrics.get(
            "nominal_sample_budget_per_head"
        ) != expected_budget:
            raise PermanentWorkerError(
                f"Expected sample budget {expected_budget}, found "
                f"{metrics.get('nominal_sample_budget_per_head')}."
            )
        expected_clusters = setting.selected_clusters_per_head
        if (
            expected_clusters is not None
            and metrics.get("clusters_per_head") != expected_clusters
        ):
            raise PermanentWorkerError(
                f"Expected {expected_clusters} selected clusters, found "
                f"{metrics.get('clusters_per_head')}."
            )
        expected_teams = setting.selected_teams_per_head
        if (
            expected_teams is not None
            and metrics.get("teams_per_head") != expected_teams
        ):
            raise PermanentWorkerError(
                f"Expected {expected_teams} selected teams, found "
                f"{metrics.get('teams_per_head')}."
            )
        if (
            setting.parent_size is not None
            and metrics.get("parent_size") != setting.parent_size
        ):
            raise PermanentWorkerError(
                f"Expected parent_size={setting.parent_size}, found "
                f"{metrics.get('parent_size')}."
            )
        if (
            setting.representatives_per_parent is not None
            and metrics.get("representatives_per_parent")
            != setting.representatives_per_parent
        ):
            raise PermanentWorkerError(
                "Expected representatives_per_parent="
                f"{setting.representatives_per_parent}, found "
                f"{metrics.get('representatives_per_parent')}."
            )
    return records


def _success_is_valid(
    success_path: Path,
    *,
    matrix_hash: str,
    execution_hash: str,
    work_index: int,
    prediction_path: Path,
    selected_manifest: Path,
    expected_uids: list[str],
    expected_count: int,
    setting: Any,
    task: str,
) -> bool:
    if not success_path.is_file():
        return False
    success = read_json_object(success_path)
    if success.get("matrix_hash") != matrix_hash or success.get("work_index") != work_index:
        raise PermanentWorkerError(
            f"Existing success marker belongs to another matrix/index: {success_path}"
        )
    if success.get("execution_fingerprint") != execution_hash:
        raise PermanentWorkerError(
            "Existing success marker was produced by different repository code. "
            "Use a new matrix experiment name, or deliberately remove this shard."
        )
    records = _validate_records(
        prediction_path=prediction_path,
        selected_manifest=selected_manifest,
        expected_uids=expected_uids,
        expected_count=expected_count,
        setting=setting,
        task=task,
    )
    if success.get("prediction_sha256") != _sha256(prediction_path):
        raise PermanentWorkerError(
            f"Prediction file changed after success marker was written: {prediction_path}"
        )
    return len(records) == expected_count


def run(args: argparse.Namespace) -> int:
    matrix = load_matrix(args.matrix)
    base = matrix.load_base_run_config()
    index = _completion_index(args.index)
    item = matrix.work_item(index)
    execution_hash = execution_fingerprint(matrix)
    paths = KubernetesPaths.create(args.results_root, matrix)
    configure_huggingface_environment(paths, offline=True)
    marker = require_cache_marker(paths, matrix, base)
    expected_uids = marker.get("selected_uids", {}).get(item.task)
    if not isinstance(expected_uids, list) or len(expected_uids) != matrix.prompts_per_task:
        raise PermanentWorkerError(
            f"Cache marker lacks {matrix.prompts_per_task} selected UIDs for {item.task}."
        )
    expected_uids = [str(uid) for uid in expected_uids]

    shard_dir = paths.shards_root / item.shard_name
    run_dir = shard_dir / "run"
    success_path = shard_dir / "success.json"
    prediction_path = _prediction_path(run_dir, item.setting.backend, item.task)
    selected_manifest = run_dir / "selected_prompts.jsonl"
    shard_dir.mkdir(parents=True, exist_ok=True)

    # Duplicate Indexed-Job pods for the same completion serialize here. The
    # later pod validates the first pod's durable success marker and exits 0.
    with exclusive_lock(paths.locks_root / f"{item.shard_name}.lock"):
        if _success_is_valid(
            success_path,
            matrix_hash=matrix.matrix_hash,
            execution_hash=execution_hash,
            work_index=index,
            prediction_path=prediction_path,
            selected_manifest=selected_manifest,
            expected_uids=expected_uids,
            expected_count=matrix.prompts_per_task,
            setting=item.setting,
            task=item.task,
        ):
            print(f"Shard already complete and verified: {item.shard_name}")
            return 0

        runtime = _runtime_metadata()
        attempt_id = f"{compact_utc_now()}-{runtime.get('pod_uid') or socket.gethostname()}"
        attempts_dir = shard_dir / "attempts"
        attempts_dir.mkdir(parents=True, exist_ok=True)
        attempt_path = attempts_dir / f"{attempt_id}.json"
        attempt = {
            "schema_version": 1,
            "status": "running",
            "started_at_utc": utc_now(),
            "attempt_id": attempt_id,
            "work_index": index,
            "matrix_hash": matrix.matrix_hash,
            "execution_fingerprint": execution_hash,
            "cache_fingerprint": cache_fingerprint(matrix, base),
            "work_item": item.to_dict(),
            "runtime": runtime,
        }
        atomic_write_json(attempt_path, attempt)
        atomic_write_json(shard_dir / "work-item.json", attempt)

        config = apply_work_item(
            base,
            matrix,
            item,
            local_data_root=paths.local_data_root,
        )
        save_config(config, shard_dir / "config.worker.yaml")
        print(
            f"Running index {index}/{matrix.completion_count - 1}: "
            f"{item.setting.name} x {item.task} ({matrix.prompts_per_task} prompts)"
        )
        try:
            _validate_cuda_runtime(runtime)
            run_benchmark(config, explicit_run_dir=run_dir)
            records = _validate_records(
                prediction_path=prediction_path,
                selected_manifest=selected_manifest,
                expected_uids=expected_uids,
                expected_count=matrix.prompts_per_task,
                setting=item.setting,
                task=item.task,
            )
            success = {
                **attempt,
                "status": "complete",
                "finished_at_utc": utc_now(),
                "record_count": len(records),
                "prediction_path": str(prediction_path),
                "prediction_sha256": _sha256(prediction_path),
            }
            atomic_write_json(success_path, success)
            atomic_write_json(attempt_path, success)
            print(f"Shard complete: {item.shard_name}")
            return 0
        except Exception as exc:
            failure = {
                **attempt,
                "status": "failed",
                "finished_at_utc": utc_now(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            atomic_write_json(attempt_path, failure)
            atomic_write_json(shard_dir / "latest-failure.json", failure)
            raise


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except (PermanentWorkerError, CacheNotReadyError, MatrixError, FileNotFoundError) as exc:
        print(f"PERMANENT WORKER ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 42
    except Exception as exc:
        print(f"RETRYABLE WORKER ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
