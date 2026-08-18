#!/usr/bin/env python3
"""Create a compact, prompt-free archive from all Kubernetes result shards."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Mapping

from santapp_ruler.io_utils import atomic_write_json, atomic_write_text
from santapp_ruler.k8s_matrix import load_matrix
from santapp_ruler.k8s_runtime import (
    KubernetesPaths,
    configure_huggingface_environment,
    read_json_object,
    require_cache_marker,
    utc_now,
)
from santapp_ruler.reporting import build_reports, read_jsonl


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
        "--output-dir",
        type=Path,
        default=Path(os.environ.get("EXPORT_ROOT", "/export")),
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Write a status/archive even when some shards are unfinished.",
    )
    parser.add_argument("--no-archive", action="store_true")
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    rendered = "\n".join(
        json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        for row in rows
    )
    atomic_write_text(path, rendered + ("\n" if rendered else ""))


def _flatten_record(record: Mapping[str, Any]) -> dict[str, Any]:
    row = {
        "experiment": record.get("experiment"),
        "matrix_hash": record.get("matrix_hash"),
        "work_index": record.get("work_index"),
        "setting": record.get("setting"),
        "backend": record.get("backend"),
        "task": record.get("task"),
        "index": record.get("index"),
        "uid": record.get("uid"),
        "example_ruler_score": record.get("example_ruler_score"),
        "prediction": record.get("prediction"),
        "references": json.dumps(record.get("references", []), ensure_ascii=False),
    }
    runtime = record.get("runtime", {})
    if isinstance(runtime, Mapping):
        for key in (
            "pod_name",
            "pod_uid",
            "node_name",
            "gpu_name",
            "gpu_total_gib",
            "cuda_capability",
            "torch",
            "torch_cuda",
            "container_image",
            "nvidia_driver_version",
            "gpu_uuid",
        ):
            row[key] = runtime.get(key)
    metrics = record.get("metrics", {})
    if isinstance(metrics, Mapping):
        for key, value in metrics.items():
            row[f"metric_{key}"] = value
    return row


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not fieldnames:
            return
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _hardware_provenance(
    shard_statuses: list[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for status in shard_statuses:
        if status.get("status") != "complete":
            continue
        rows.append(
            {
                "work_index": status.get("work_index"),
                "setting": status.get("setting"),
                "backend": status.get("backend"),
                "task": status.get("task"),
                "pod_name": status.get("pod_name"),
                "node_name": status.get("node_name"),
                "gpu_name": status.get("gpu_name"),
                "gpu_total_gib": status.get("gpu_total_gib"),
                "cuda_capability": status.get("cuda_capability"),
                "torch": status.get("torch"),
                "torch_cuda": status.get("torch_cuda"),
                "container_image": status.get("container_image"),
                "nvidia_driver_version": status.get("nvidia_driver_version"),
                "gpu_uuid": status.get("gpu_uuid"),
            }
        )
    gpu_names = sorted(
        {str(row["gpu_name"]) for row in rows if row.get("gpu_name")}
    )
    node_names = sorted(
        {str(row["node_name"]) for row in rows if row.get("node_name")}
    )
    counts_by_gpu = {
        name: sum(row.get("gpu_name") == name for row in rows) for name in gpu_names
    }
    single_product = bool(rows) and len(gpu_names) == 1
    if single_product:
        note = (
            "All completed shards report one GPU product. Timing is product-controlled "
            "but can still vary with node load and software/runtime effects."
        )
    elif gpu_names:
        note = (
            "Completed shards span multiple GPU products. Wall-clock timing and speedup "
            "fields are not hardware-controlled comparisons; seeded stochastic paths "
            "also need not be bitwise identical across GPU architectures. Stratify by "
            "gpu_name or rerun with one product affinity for strict control."
        )
    else:
        note = (
            "No GPU product provenance is available yet, so timing-comparison validity "
            "cannot be assessed."
        )
    return rows, {
        "completed_shards_with_provenance": len(rows),
        "gpu_products": gpu_names,
        "nodes": node_names,
        "shard_counts_by_gpu_product": counts_by_gpu,
        "single_gpu_product": single_product,
        "runtime_comparisons_hardware_controlled": single_product,
        "note": note,
    }


def _assert_prompt_free(rows: list[Mapping[str, Any]]) -> None:
    for row in rows:
        if "input" in row:
            raise ValueError("Compact export contains a top-level input field.")
        serialized = json.dumps(row, ensure_ascii=False)
        # The literal field names catch accidental embedding without rejecting a
        # harmless metric or prose containing the English word "input".
        if '"input":' in serialized or '"prompt":' in serialized:
            raise ValueError("Compact export contains prompt text fields.")


def _collect_shard(
    *,
    item: Any,
    matrix: Any,
    paths: KubernetesPaths,
    cache_marker: Mapping[str, Any],
) -> tuple[list[dict[str, Any]] | None, dict[str, Any]]:
    shard_dir = paths.shards_root / item.shard_name
    success_path = shard_dir / "success.json"
    prediction_path = (
        shard_dir
        / "run"
        / "predictions"
        / item.setting.backend
        / f"{item.task}.jsonl"
    )
    status: dict[str, Any] = {
        "work_index": item.index,
        "shard_name": item.shard_name,
        "setting": item.setting.name,
        "backend": item.setting.backend,
        "task": item.task,
        "expected_records": matrix.prompts_per_task,
        "success_marker": str(success_path),
        "prediction_path": str(prediction_path),
    }
    if not success_path.is_file():
        status.update({"status": "incomplete", "reason": "missing success.json"})
        return None, status
    try:
        success = read_json_object(success_path)
        if success.get("matrix_hash") != matrix.matrix_hash:
            raise ValueError("success marker matrix_hash mismatch")
        if success.get("work_index") != item.index:
            raise ValueError("success marker work_index mismatch")
        records = read_jsonl(prediction_path)
        expected_uids = cache_marker.get("selected_uids", {}).get(item.task)
        if not isinstance(expected_uids, list):
            raise ValueError("cache marker selected_uids missing")
        actual_uids = [str(record.get("uid")) for record in records]
        if actual_uids != [str(uid) for uid in expected_uids]:
            raise ValueError("prediction UIDs differ from pinned selection")
        if len(records) != matrix.prompts_per_task:
            raise ValueError(
                f"expected {matrix.prompts_per_task} records, found {len(records)}"
            )
        if any("input" in record for record in records):
            raise ValueError("prediction JSONL contains input prompt text")
        checksum = _sha256(prediction_path)
        if success.get("prediction_sha256") != checksum:
            raise ValueError("prediction checksum differs from success marker")

        runtime = success.get("runtime", {})
        runtime_mapping = runtime if isinstance(runtime, Mapping) else {}
        compact_runtime = {
            key: runtime_mapping.get(key)
            for key in (
                "pod_name",
                "pod_uid",
                "node_name",
                "gpu_name",
                "gpu_total_gib",
                "cuda_capability",
                "torch",
                "torch_cuda",
                "container_image",
                "nvidia_driver_version",
                "gpu_uuid",
            )
        }
        enriched: list[dict[str, Any]] = []
        for record in records:
            metrics = record.get("metrics", {})
            if not isinstance(metrics, Mapping):
                raise ValueError("record metrics is not a mapping")
            enriched.append(
                {
                    "schema_version": 1,
                    "experiment": matrix.experiment,
                    "matrix_hash": matrix.matrix_hash,
                    "work_index": item.index,
                    "setting": item.setting.name,
                    "backend": item.setting.backend,
                    "task": item.task,
                    "index": record.get("index"),
                    "uid": record.get("uid"),
                    "references": record.get("outputs", []),
                    "prediction": record.get("pred", ""),
                    "example_ruler_score": metrics.get("example_ruler_score"),
                    "metrics": dict(metrics),
                    "runtime": compact_runtime,
                }
            )
        _assert_prompt_free(enriched)
        status.update(
            {
                "status": "complete",
                "record_count": len(records),
                "prediction_sha256": checksum,
                "pod_name": runtime_mapping.get("pod_name"),
                "node_name": runtime_mapping.get("node_name"),
                "gpu_name": runtime_mapping.get("gpu_name"),
                "gpu_total_gib": runtime_mapping.get("gpu_total_gib"),
                "cuda_capability": runtime_mapping.get("cuda_capability"),
                "torch": runtime_mapping.get("torch"),
                "torch_cuda": runtime_mapping.get("torch_cuda"),
                "container_image": runtime_mapping.get("container_image"),
                "nvidia_driver_version": runtime_mapping.get(
                    "nvidia_driver_version"
                ),
                "gpu_uuid": runtime_mapping.get("gpu_uuid"),
                "execution_fingerprint": success.get("execution_fingerprint"),
            }
        )
        return enriched, status
    except Exception as exc:
        status.update(
            {
                "status": "invalid",
                "reason": f"{type(exc).__name__}: {exc}",
            }
        )
        return None, status


def _report_record(compact: Mapping[str, Any]) -> dict[str, Any]:
    runtime = compact.get("runtime", {})
    runtime_mapping = runtime if isinstance(runtime, Mapping) else {}
    return {
        "index": compact.get("index"),
        "uid": compact.get("uid"),
        "task": compact.get("task"),
        "outputs": compact.get("references", []),
        "pred": compact.get("prediction", ""),
        "others": {
            "setting": compact.get("setting"),
            "backend": compact.get("backend"),
            "work_index": compact.get("work_index"),
            "pod_name": runtime_mapping.get("pod_name"),
            "node_name": runtime_mapping.get("node_name"),
            "gpu_name": runtime_mapping.get("gpu_name"),
        },
        "metrics": compact.get("metrics", {}),
    }


def _replace_directory(staging: Path, destination: Path) -> None:
    backup = destination.with_name(destination.name + ".old")
    if backup.exists():
        shutil.rmtree(backup)
    if destination.exists():
        os.replace(destination, backup)
    os.replace(staging, destination)
    if backup.exists():
        shutil.rmtree(backup)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    matrix = load_matrix(args.matrix)
    base = matrix.load_base_run_config()
    paths = KubernetesPaths.create(
        args.results_root, matrix, create_directories=False
    )
    configure_huggingface_environment(paths, offline=True)
    cache_marker = require_cache_marker(paths, matrix, base)

    output_root = args.output_dir.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{matrix.experiment}-", dir=output_root)
    )
    final_dir = output_root / matrix.experiment
    compact_rows: list[dict[str, Any]] = []
    shard_statuses: list[dict[str, Any]] = []
    report_rows: dict[tuple[str, str], list[dict[str, Any]]] = {}

    try:
        for item in matrix.work_items():
            rows, status = _collect_shard(
                item=item,
                matrix=matrix,
                paths=paths,
                cache_marker=cache_marker,
            )
            shard_statuses.append(status)
            if rows is None:
                continue
            compact_rows.extend(rows)
            report_rows[(item.setting.name, item.task)] = [
                _report_record(row) for row in rows
            ]

        completed = sum(status["status"] == "complete" for status in shard_statuses)
        execution_hashes = {
            status.get("execution_fingerprint")
            for status in shard_statuses
            if status.get("execution_fingerprint")
        }
        if len(execution_hashes) > 1:
            raise RuntimeError(
                "Completed shards were produced by more than one repository code "
                f"fingerprint: {sorted(execution_hashes)}"
            )
        complete = completed == matrix.completion_count
        hardware_rows, hardware_summary = _hardware_provenance(shard_statuses)
        status_payload = {
            "schema_version": 1,
            "experiment": matrix.experiment,
            "matrix_hash": matrix.matrix_hash,
            "generated_at_utc": utc_now(),
            "status": "complete" if complete else "partial",
            "expected_shards": matrix.completion_count,
            "completed_shards": completed,
            "expected_records": matrix.completion_count * matrix.prompts_per_task,
            "exported_records": len(compact_rows),
            "tasks": list(matrix.tasks),
            "settings": [setting.name for setting in matrix.settings],
            "execution_fingerprint": next(iter(execution_hashes), None),
            "hardware": hardware_summary,
            "shards": shard_statuses,
        }
        atomic_write_json(staging / "export-status.json", status_payload)
        atomic_write_json(staging / "matrix.json", matrix.canonical_payload())
        atomic_write_json(staging / "timing-validity.json", hardware_summary)
        _write_csv(staging / "hardware-provenance.csv", hardware_rows)
        _write_jsonl(staging / "compact-results.jsonl", compact_rows)
        _write_csv(
            staging / "compact-results.csv",
            [_flatten_record(row) for row in compact_rows],
        )

        if complete:
            report_root = staging / "report"
            for setting in matrix.settings:
                for task in matrix.tasks:
                    rows = report_rows[(setting.name, task)]
                    destination = (
                        report_root
                        / "predictions"
                        / setting.name
                        / f"{task}.jsonl"
                    )
                    _write_jsonl(destination, rows)
            build_reports(
                report_root,
                backends=[setting.name for setting in matrix.settings],
                tasks=list(matrix.tasks),
                bootstrap_resamples=base.grading.bootstrap_resamples,
                confidence_level=base.grading.confidence_level,
                bootstrap_seed=base.grading.bootstrap_seed,
            )
            for name in (
                "summary.json",
                "summary.csv",
                "summary.md",
                "per_example.csv",
                "comparisons.csv",
            ):
                shutil.copy2(report_root / name, staging / name)
        elif not args.allow_partial:
            incomplete = [
                f"{row['work_index']}:{row['setting']}:{row['task']}:{row['status']}"
                for row in shard_statuses
                if row["status"] != "complete"
            ]
            raise RuntimeError(
                "Experiment is incomplete; unfinished/invalid shards: "
                + ", ".join(incomplete)
            )

        _assert_prompt_free(compact_rows)
        atomic_write_text(
            staging / "README.txt",
            (
                f"Experiment: {matrix.experiment}\n"
                f"Status: {'complete' if complete else 'partial'}\n"
                f"Compact records: {len(compact_rows)}\n"
                "Prompt text and the shared Hugging Face cache are intentionally "
                "excluded.\n"
                f"Timing validity: {hardware_summary['note']}\n"
            ),
        )
        _replace_directory(staging, final_dir)

        archive_path: Path | None = None
        if not args.no_archive:
            suffix = "results" if complete else "partial-results"
            archive_path = output_root / f"{matrix.experiment}-{suffix}.tar.gz"
            temporary_archive = archive_path.with_suffix(archive_path.suffix + ".tmp")
            temporary_archive.unlink(missing_ok=True)
            with tarfile.open(temporary_archive, "w:gz") as archive:
                archive.add(final_dir, arcname=final_dir.name, recursive=True)
            os.replace(temporary_archive, archive_path)
            atomic_write_text(
                output_root / f"{archive_path.name}.sha256",
                f"{_sha256(archive_path)}  {archive_path.name}\n",
            )

        print(json.dumps(status_payload, indent=2))
        print(f"Compact export directory: {final_dir}")
        if archive_path is not None:
            print(f"Archive: {archive_path}")
        return 0 if complete or args.allow_partial else 2
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
