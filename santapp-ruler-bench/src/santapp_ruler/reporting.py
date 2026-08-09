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


def _pct(numerator: int, denominator: int) -> float | None:
    return 100.0 * numerator / denominator if denominator else None


def _aggregate_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = [record.get("metrics", {}) for record in records]
    generated = sum(int(value.get("generated_tokens", 0)) for value in metrics)
    total_seconds = sum(float(value.get("total_seconds", 0.0)) for value in metrics)
    prefill_seconds = sum(float(value.get("prefill_seconds", 0.0)) for value in metrics)
    cluster_seconds = sum(
        float(value.get("clustering_seconds", 0.0)) for value in metrics
    )
    decode_seconds = sum(float(value.get("decode_seconds", 0.0)) for value in metrics)

    head_calls = sum(int(value.get("decode_attention_head_calls", 0)) for value in metrics)
    group_calls = sum(int(value.get("decode_gqa_group_calls", 0)) for value in metrics)
    dense_gqa = sum(
        int(value.get("decode_dense_gqa_kv_vectors", 0)) for value in metrics
    )
    dense_naive = sum(
        int(value.get("decode_dense_naive_kv_vectors", 0)) for value in metrics
    )
    gqa_kv = sum(int(value.get("decode_gqa_kv_vectors_read", 0)) for value in metrics)
    naive_kv = sum(
        int(value.get("decode_naive_kv_vectors_read", 0)) for value in metrics
    )
    gqa_centroids = sum(
        int(value.get("decode_gqa_centroid_key_vectors_read", 0))
        for value in metrics
    )
    naive_centroids = sum(
        int(value.get("decode_naive_centroid_key_vectors_read", 0))
        for value in metrics
    )

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
        # Main reported access estimate: GQA-row union plus centroids.
        "decode_gqa_total_access_pct": _pct(gqa_kv + gqa_centroids, dense_gqa),
        "decode_gqa_kv_access_pct": _pct(gqa_kv, dense_gqa),
        "decode_gqa_centroid_access_pct": _pct(gqa_centroids, dense_gqa),
        # Explicitly optimistic legacy convention with a per-query-head dense
        # denominator and draw-count numerator.
        "decode_naive_total_access_pct": _pct(
            naive_kv + naive_centroids, dense_naive
        ),
        "decode_naive_kv_access_pct": _pct(naive_kv, dense_naive),
        "decode_naive_centroid_access_pct": _pct(naive_centroids, dense_naive),
        "mean_sampled_token_draws_per_head_call": _weighted_mean(
            (
                value.get("mean_sampled_token_draws_per_head_call"),
                int(value.get("decode_attention_head_calls", 0)),
            )
            for value in metrics
        ),
        "mean_unique_sampled_tokens_per_gqa_group_call": _weighted_mean(
            (
                value.get("mean_unique_sampled_tokens_per_gqa_group_call"),
                int(value.get("decode_gqa_group_calls", 0)),
            )
            for value in metrics
        ),
        "mean_exact_tokens_per_gqa_group_call": _weighted_mean(
            (
                value.get("mean_exact_tokens_per_gqa_group_call"),
                int(value.get("decode_gqa_group_calls", 0)),
            )
            for value in metrics
        ),
        "peak_allocated_gib": max(
            (float(value.get("peak_allocated_gib", 0.0)) for value in metrics),
            default=0.0,
        ),
        "decode_attention_head_calls": head_calls,
        "decode_gqa_group_calls": group_calls,
        "decode_dense_gqa_kv_vectors": dense_gqa,
        "decode_dense_naive_kv_vectors": dense_naive,
        "decode_gqa_kv_vectors_read": gqa_kv,
        "decode_naive_kv_vectors_read": naive_kv,
        "decode_gqa_centroid_key_vectors_read": gqa_centroids,
        "decode_naive_centroid_key_vectors_read": naive_centroids,
        "decode_gqa_total_vectors_read": gqa_kv + gqa_centroids,
        "decode_naive_total_vectors_read": naive_kv + naive_centroids,
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


def _speedup(reference: Mapping[str, Any], candidate: Mapping[str, Any], key: str):
    candidate_value = candidate.get(key)
    reference_value = reference.get(key)
    if not candidate_value or reference_value is None:
        return None
    return reference_value / candidate_value


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
                    "references": json.dumps(
                        record.get("outputs", []), ensure_ascii=False
                    ),
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
            "nulls": (
                f"{sum(not str(r.get('pred', '')).strip() for r in all_records)}"
                f"/{len(all_records)}"
            ),
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
    sdpa_entry = structured["backends"].get("sdpa")
    if sdpa_entry is not None:
        sdpa = sdpa_entry["aggregate"]
        for backend in ("santa", "santapp"):
            candidate_entry = structured["backends"].get(backend)
            if candidate_entry is None:
                continue
            candidate = candidate_entry["aggregate"]
            comparisons[f"{backend}_vs_sdpa"] = {
                "selected_task_mean_score_delta": candidate["score"] - sdpa["score"],
                "end_to_end_speedup": _speedup(
                    sdpa, candidate, "mean_total_seconds"
                ),
                "decode_speedup": _speedup(sdpa, candidate, "mean_decode_seconds"),
                "decode_gqa_total_access_pct": candidate.get(
                    "decode_gqa_total_access_pct"
                ),
                "decode_gqa_kv_access_pct": candidate.get(
                    "decode_gqa_kv_access_pct"
                ),
                "decode_gqa_centroid_access_pct": candidate.get(
                    "decode_gqa_centroid_access_pct"
                ),
                "decode_naive_total_access_pct": candidate.get(
                    "decode_naive_total_access_pct"
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

    def fmt(value: Any, digits: int = 2) -> str:
        if value is None:
            return "—"
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return "—"
        return f"{float(value):.{digits}f}"

    lines = [
        "# SANTA / SANTA++ RULER benchmark summary",
        "",
        "Task scores use RULER's task-family grader. The selected-task mean is an unweighted harness summary, not an additional RULER metric.",
        "",
        "| Backend | Task | Score | Mean total s | Mean decode s | GQA + centroid | GQA KV | GQA centroid | Naive + centroid |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report_rows:
        lines.append(
            "| {backend} | {task} | {score} | {total} | {decode} | {gqa_total}% | "
            "{gqa_kv}% | {centroid}% | {naive}% |".format(
                backend=row["backend"],
                task=row["task"],
                score=fmt(row.get("score")),
                total=fmt(row.get("mean_total_seconds"), 3),
                decode=fmt(row.get("mean_decode_seconds"), 3),
                gqa_total=fmt(row.get("decode_gqa_total_access_pct"), 1),
                gqa_kv=fmt(row.get("decode_gqa_kv_access_pct"), 1),
                centroid=fmt(row.get("decode_gqa_centroid_access_pct"), 1),
                naive=fmt(row.get("decode_naive_total_access_pct"), 1),
            )
        )

    if comparisons:
        lines.extend(["", "## Aggregate comparisons against SDPA", ""])
        for backend, label in (("santa", "SANTA"), ("santapp", "SANTA++")):
            comparison = comparisons.get(f"{backend}_vs_sdpa")
            if comparison is None:
                continue
            lines.extend(
                [
                    f"### {label}",
                    "",
                    f"- Selected-task mean score delta: {fmt(comparison['selected_task_mean_score_delta'])} points",
                    f"- GQA + centroid access: {fmt(comparison['decode_gqa_total_access_pct'], 1)}%",
                    f"- GQA KV access: {fmt(comparison['decode_gqa_kv_access_pct'], 1)}%",
                    f"- GQA centroid access: {fmt(comparison['decode_gqa_centroid_access_pct'], 1)}%",
                    f"- Naive + centroid access: {fmt(comparison['decode_naive_total_access_pct'], 1)}%",
                    f"- End-to-end speedup: {fmt(comparison['end_to_end_speedup'], 3)}x",
                    f"- Decode speedup: {fmt(comparison['decode_speedup'], 3)}x",
                    "",
                ]
            )

    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return structured
