"""Prediction loading, official RULER grading, and benchmark reports."""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Iterable, Mapping
from pathlib import Path
from statistics import fmean
from typing import Any

from .ruler.grader import grade_task


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    records: list[dict[str, Any]] = []
    if not path.is_file():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            records.append(value)
    return records


def _safe_mean(values: Iterable[float | int | None]) -> float | None:
    cleaned = [float(value) for value in values if value is not None]
    return fmean(cleaned) if cleaned else None


def _weighted_mean(pairs: Iterable[tuple[float | None, int]]) -> float | None:
    numerator = 0.0
    denominator = 0
    for value, weight in pairs:
        if value is None or weight <= 0:
            continue
        numerator += float(value) * weight
        denominator += weight
    return numerator / denominator if denominator else None


def _aggregate_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = [record.get("metrics", {}) for record in records]
    generated = sum(int(value.get("generated_tokens", 0)) for value in metrics)
    total_seconds = sum(float(value.get("total_seconds", 0.0)) for value in metrics)
    prefill_seconds = sum(float(value.get("prefill_seconds", 0.0)) for value in metrics)
    cluster_seconds = sum(
        float(value.get("clustering_seconds", 0.0)) for value in metrics
    )
    decode_seconds = sum(float(value.get("decode_seconds", 0.0)) for value in metrics)
    dense_vectors = sum(int(value.get("decode_dense_kv_vectors", 0)) for value in metrics)
    kv_vectors = sum(int(value.get("decode_kv_vectors_read", 0)) for value in metrics)
    metadata_vectors = sum(
        int(value.get("decode_metadata_key_vectors_read", 0)) for value in metrics
    )
    head_calls = sum(int(value.get("decode_attention_head_calls", 0)) for value in metrics)
    return {
        "num_examples": len(records),
        "generated_tokens": generated,
        "mean_prompt_tokens": _safe_mean(
            value.get("prompt_tokens") for value in metrics
        ),
        "mean_generated_tokens": _safe_mean(
            value.get("generated_tokens") for value in metrics
        ),
        "mean_prefill_seconds": prefill_seconds / len(records) if records else None,
        "mean_clustering_seconds": cluster_seconds / len(records) if records else None,
        "mean_decode_seconds": decode_seconds / len(records) if records else None,
        "mean_total_seconds": total_seconds / len(records) if records else None,
        "decode_tokens_per_second": (
            generated / decode_seconds if decode_seconds > 0 else None
        ),
        "end_to_end_output_tokens_per_second": (
            generated / total_seconds if total_seconds > 0 else None
        ),
        "decode_kv_access_pct": (
            100.0 * kv_vectors / dense_vectors if dense_vectors else None
        ),
        "decode_read_equivalent_pct": (
            100.0 * (kv_vectors + metadata_vectors) / dense_vectors
            if dense_vectors
            else None
        ),
        "mean_ess_over_samples": _weighted_mean(
            (
                value.get("mean_ess_over_samples"),
                int(value.get("decode_attention_head_calls", 0)),
            )
            for value in metrics
        ),
        "peak_allocated_gib": max(
            (float(value.get("peak_allocated_gib", 0.0)) for value in metrics),
            default=0.0,
        ),
        "decode_attention_head_calls": head_calls,
        "decode_dense_kv_vectors": dense_vectors,
        "decode_kv_vectors_read": kv_vectors,
        "decode_metadata_key_vectors_read": metadata_vectors,
    }


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_official_style(
    backend_dir: Path,
    task_rows: list[dict[str, Any]],
    records_by_task: Mapping[str, list[dict[str, Any]]],
) -> None:
    tasks = [row["task"] for row in task_rows]
    scores = [row["score"] for row in task_rows]
    nulls = [row["nulls"] for row in task_rows]
    with (backend_dir / "summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["Tasks", *tasks])
        writer.writerow(["Score", *scores])
        writer.writerow(["Nulls", *nulls])

    with (backend_dir / "submission.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=["Task", "ID", "Prediction"])
        writer.writeheader()
        for task in tasks:
            for record in records_by_task[task]:
                writer.writerow(
                    {
                        "Task": task,
                        "ID": record.get("index"),
                        "Prediction": record.get("pred", ""),
                    }
                )


def build_reports(
    run_dir: str | Path,
    *,
    backends: Iterable[str],
    tasks: Iterable[str],
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    backends = list(backends)
    tasks = list(tasks)
    report_rows: list[dict[str, Any]] = []
    per_example_rows: list[dict[str, Any]] = []
    structured: dict[str, Any] = {"backends": {}}

    for backend in backends:
        backend_dir = run_dir / "predictions" / backend
        records_by_task: dict[str, list[dict[str, Any]]] = {}
        task_rows: list[dict[str, Any]] = []
        all_records: list[dict[str, Any]] = []
        task_scores: list[float] = []

        for task in tasks:
            records = read_jsonl(backend_dir / f"{task}.jsonl")
            if not records:
                raise ValueError(
                    f"No predictions found for backend={backend!r}, task={task!r}."
                )
            records_by_task[task] = records
            all_records.extend(records)
            predictions = [str(record.get("pred", "")) for record in records]
            references = [list(map(str, record["outputs"])) for record in records]
            grade = grade_task(task, predictions, references)
            task_scores.append(grade.score)
            aggregate = _aggregate_metrics(records)
            row = {
                "backend": backend,
                "task": task,
                "score": grade.score,
                "nulls": grade.nulls_label,
                **aggregate,
            }
            report_rows.append(row)
            task_rows.append(row)

            for record in records:
                single_grade = grade_task(
                    task,
                    [str(record.get("pred", ""))],
                    [list(map(str, record["outputs"]))],
                )
                flat = {
                    "backend": backend,
                    "task": task,
                    "index": record.get("index"),
                    "uid": record.get("uid"),
                    "example_score": single_grade.score,
                    "prediction": record.get("pred", ""),
                    "references": json.dumps(record.get("outputs", []), ensure_ascii=False),
                }
                for key, value in record.get("metrics", {}).items():
                    flat[f"metric_{key}"] = value
                per_example_rows.append(flat)

        mean_score = round(fmean(task_scores), 2)
        aggregate_all = _aggregate_metrics(all_records)
        aggregate_row = {
            "backend": backend,
            "task": "__selected_task_mean__",
            "score": mean_score,
            "nulls": f"{sum(not str(r.get('pred', '')).strip() for r in all_records)}/{len(all_records)}",
            **aggregate_all,
        }
        report_rows.append(aggregate_row)
        structured["backends"][backend] = {
            "selected_task_unweighted_mean_score": mean_score,
            "tasks": {row["task"]: row for row in task_rows},
            "aggregate": aggregate_row,
        }
        _write_official_style(backend_dir, task_rows, records_by_task)

    comparisons: dict[str, Any] = {}
    if "sdpa" in structured["backends"] and "santapp" in structured["backends"]:
        sdpa = structured["backends"]["sdpa"]["aggregate"]
        santa = structured["backends"]["santapp"]["aggregate"]
        comparisons = {
            "selected_task_mean_score_delta_santapp_minus_sdpa": (
                santa["score"] - sdpa["score"]
            ),
            "santapp_end_to_end_speedup_vs_sdpa": (
                sdpa["mean_total_seconds"] / santa["mean_total_seconds"]
                if santa.get("mean_total_seconds")
                else None
            ),
            "santapp_decode_speedup_vs_sdpa": (
                sdpa["mean_decode_seconds"] / santa["mean_decode_seconds"]
                if santa.get("mean_decode_seconds")
                else None
            ),
            "santapp_decode_kv_access_pct": santa.get("decode_kv_access_pct"),
            "santapp_decode_read_equivalent_pct": santa.get(
                "decode_read_equivalent_pct"
            ),
        }
    structured["comparisons"] = comparisons

    _write_csv(run_dir / "summary.csv", report_rows)
    _write_csv(run_dir / "per_example.csv", per_example_rows)
    with (run_dir / "summary.json").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        json.dump(structured, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")

    lines = [
        "# SANTA++ RULER benchmark summary",
        "",
        "Task scores below use RULER's task-family grader. The selected-task mean is an unweighted harness summary, not an additional RULER metric.",
        "",
        "| Backend | Task | Score | Mean total s | Mean decode s | Decode KV access | Read-equivalent |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in report_rows:
        def fmt(value: Any, digits: int = 2) -> str:
            if value is None:
                return "—"
            if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                return "—"
            return f"{float(value):.{digits}f}"

        lines.append(
            "| {backend} | {task} | {score} | {total} | {decode} | {kv}% | {equiv}% |".format(
                backend=row["backend"],
                task=row["task"],
                score=fmt(row.get("score")),
                total=fmt(row.get("mean_total_seconds"), 3),
                decode=fmt(row.get("mean_decode_seconds"), 3),
                kv=fmt(row.get("decode_kv_access_pct"), 1),
                equiv=fmt(row.get("decode_read_equivalent_pct"), 1),
            )
        )
    if comparisons:
        lines.extend(
            [
                "",
                "## Paired aggregate comparison",
                "",
                f"- SANTA++ minus SDPA selected-task mean score: {comparisons['selected_task_mean_score_delta_santapp_minus_sdpa']:.2f} points",
                f"- SANTA++ decode KV access: {comparisons['santapp_decode_kv_access_pct']:.1f}%",
                f"- SANTA++ metadata-inclusive read-equivalent: {comparisons['santapp_decode_read_equivalent_pct']:.1f}%",
                f"- SANTA++ end-to-end speedup vs SDPA: {comparisons['santapp_end_to_end_speedup_vs_sdpa']:.3f}x",
                f"- SANTA++ decode speedup vs SDPA: {comparisons['santapp_decode_speedup_vs_sdpa']:.3f}x",
            ]
        )
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return structured
