"""RULER grading, metric aggregation, and compact CSV/Markdown reports."""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import fmean
from typing import Any

from .ruler.grader import grade_task, stratified_bootstrap_mean_confidence_interval


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.is_file():
        return []
    records: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {source}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object at {source}:{line_number}")
            records.append(value)
    return records


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
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


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _mean(metrics: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = [number for metric in metrics if (number := _finite(metric.get(key))) is not None]
    return fmean(values) if values else None


def _sum(metrics: Sequence[Mapping[str, Any]], key: str) -> float:
    values = [_finite(metric.get(key)) for metric in metrics]
    return sum(value for value in values if value is not None)


def _constant(metrics: Sequence[Mapping[str, Any]], key: str) -> Any:
    values = [metric[key] for metric in metrics if key in metric]
    if not values:
        return None
    first = values[0]
    return first if all(value == first for value in values[1:]) else "mixed"


def _pct(numerator: float, denominator: float) -> float | None:
    return 100.0 * numerator / denominator if denominator else None


IDENTITY_KEYS = (
    "mode",
    "sampling_scheme",
    "sampling_unit",
    "importance_correction",
    "routing_key_type",
    "prompt_format",
    "exact_token_policy",
    "prompt_exact_tail_tokens",
    "initial_generated_exact_tokens",
    "parent_policy",
    "parent_clustering_scope",
    "parent_clustering_representation",
    "parent_clustering_space",
    "parent_clustering_uses_probe_queries",
    "key_space_normalization",
    "key_space_rope_applied",
    "key_space_stage",
    "parent_size",
    "parent_size_semantics",
    "parent_cluster_count_per_layer_kv_head",
    "representatives_per_parent",
    "samples_per_head",
    "nominal_sample_budget_per_head",
    "nominal_team_size",
    "requested_teams_per_head",
    "probe_policy",
    "kmeans_batch_size",
    "kmeans_n_init",
    "kmeans_max_iter",
    "kmeans_tol",
    "kmeans_max_no_improvement",
    "kmeans_init_size",
    "kmeans_reassignment_ratio",
    "kmeans_random_state",
    "num_query_heads",
    "num_kv_heads",
    "query_heads_per_kv",
    "head_dim",
)

ACCESS_KEYS = (
    "decode_attention_head_calls",
    "decode_gqa_group_calls",
    "decode_dense_gqa_kv_vectors",
    "decode_dense_naive_kv_vectors",
    "decode_gqa_kv_vectors_read",
    "decode_naive_kv_vectors_read",
    "decode_gqa_routing_key_vectors_read",
    "decode_naive_routing_key_vectors_read",
    "decode_gqa_centroid_key_vectors_read",
    "decode_naive_centroid_key_vectors_read",
)


def aggregate_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("Cannot aggregate an empty prediction set.")
    metrics = [record.get("metrics", {}) for record in records]
    if not all(isinstance(metric, Mapping) for metric in metrics):
        raise ValueError("Every prediction record must contain a metrics mapping.")
    typed = [metric for metric in metrics if isinstance(metric, Mapping)]

    result = {key: _constant(typed, key) for key in IDENTITY_KEYS}
    generated = int(_sum(typed, "generated_tokens"))
    prefill = _sum(typed, "prefill_seconds")
    parent_build = _sum(typed, "parent_build_seconds")
    decode = _sum(typed, "decode_seconds")
    total = _sum(typed, "total_seconds")
    result.update(
        {
            "num_examples": len(records),
            "total_generated_tokens": generated,
            "mean_prompt_tokens": _mean(typed, "prompt_tokens"),
            "mean_generated_tokens": generated / len(records),
            "mean_prefill_seconds": prefill / len(records),
            "mean_parent_build_seconds": parent_build / len(records),
            "mean_decode_seconds": decode / len(records),
            "mean_total_seconds": total / len(records),
            "decode_tokens_per_second": generated / decode if decode else None,
            "end_to_end_output_tokens_per_second": generated / total if total else None,
            "peak_allocated_gib": max(
                (_finite(metric.get("peak_allocated_gib")) or 0.0 for metric in typed),
                default=0.0,
            ),
            "peak_reserved_gib": max(
                (_finite(metric.get("peak_reserved_gib")) or 0.0 for metric in typed),
                default=0.0,
            ),
            "mean_custom_cache_gib": _mean(typed, "custom_cache_gib"),
            "mean_routing_summary_gib": _mean(typed, "routing_summary_gib"),
            "mean_actual_sampled_rows_per_head": _mean(
                typed, "mean_sampled_token_rows_per_head_call"
            ),
            "mean_selected_teams_per_head": _mean(
                typed, "mean_selected_teams_per_head_call"
            ),
            "mean_exact_tokens_per_gqa_group": _mean(
                typed, "mean_exact_tokens_per_gqa_group_call"
            ),
            "mean_team_size": _mean(typed, "team_size_mean"),
            "max_team_size": max(
                (_finite(metric.get("team_size_max")) or 0.0 for metric in typed),
                default=0.0,
            ),
            "mean_parent_size": _mean(typed, "parent_size_mean"),
            "max_parent_size": max(
                (_finite(metric.get("parent_size_max")) or 0.0 for metric in typed),
                default=0.0,
            ),
            "mean_active_parents_per_layer_kv_head": _mean(
                typed, "mean_active_parents_per_layer_kv_head"
            ),
        }
    )
    for key in ACCESS_KEYS:
        result[key] = int(_sum(typed, key))

    dense_gqa = float(result["decode_dense_gqa_kv_vectors"])
    dense_naive = float(result["decode_dense_naive_kv_vectors"])
    gqa_kv = float(result["decode_gqa_kv_vectors_read"])
    naive_kv = float(result["decode_naive_kv_vectors_read"])
    gqa_routing = float(result["decode_gqa_routing_key_vectors_read"])
    naive_routing = float(result["decode_naive_routing_key_vectors_read"])
    result.update(
        {
            "decode_gqa_total_vectors_read": int(gqa_kv + gqa_routing),
            "decode_naive_total_vectors_read": int(naive_kv + naive_routing),
            "decode_gqa_kv_access_pct": _pct(gqa_kv, dense_gqa),
            "decode_gqa_routing_access_pct": _pct(gqa_routing, dense_gqa),
            "decode_gqa_total_access_pct": _pct(gqa_kv + gqa_routing, dense_gqa),
            "decode_naive_kv_access_pct": _pct(naive_kv, dense_naive),
            "decode_naive_routing_access_pct": _pct(naive_routing, dense_naive),
            "decode_naive_total_access_pct": _pct(naive_kv + naive_routing, dense_naive),
        }
    )
    return result


def _is_pareto(row: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> bool:
    score = _finite(row.get("selected_task_mean_score"))
    access = _finite(row.get("decode_gqa_total_access_pct"))
    if score is None or access is None:
        return False
    for other in rows:
        if other is row:
            continue
        other_score = _finite(other.get("selected_task_mean_score"))
        other_access = _finite(other.get("decode_gqa_total_access_pct"))
        if other_score is None or other_access is None:
            continue
        if (
            other_score >= score
            and other_access <= access
            and (other_score > score or other_access < access)
        ):
            return False
    return True


def build_reports(
    run_dir: str | Path,
    *,
    backends: Sequence[str],
    tasks: Sequence[str],
    bootstrap_resamples: int,
    confidence_level: float,
    bootstrap_seed: int,
) -> dict[str, Any]:
    root = Path(run_dir)
    task_rows: list[dict[str, Any]] = []
    backend_rows: list[dict[str, Any]] = []

    for backend_index, backend in enumerate(backends):
        task_scores: dict[str, Sequence[float]] = {}
        backend_records: list[dict[str, Any]] = []
        for task_index, task in enumerate(tasks):
            records = read_jsonl(root / "predictions" / backend / f"{task}.jsonl")
            if not records:
                raise FileNotFoundError(
                    f"No predictions found for backend={backend!r}, task={task!r}."
                )
            grade = grade_task(
                task,
                [str(record.get("pred", "")) for record in records],
                [record.get("outputs", []) for record in records],
                bootstrap_resamples=bootstrap_resamples,
                confidence_level=confidence_level,
                bootstrap_seed=bootstrap_seed + backend_index * 1000 + task_index,
            )
            task_scores[task] = grade.example_scores
            metrics = aggregate_metrics(records)
            row = {
                "backend": backend,
                "task": task,
                "score": grade.score,
                "score_ci_lower": grade.confidence_interval.lower,
                "score_ci_upper": grade.confidence_interval.upper,
                "null_predictions": grade.null_predictions,
                **metrics,
            }
            task_rows.append(row)
            backend_records.extend(records)

        interval = stratified_bootstrap_mean_confidence_interval(
            task_scores,
            resamples=bootstrap_resamples,
            confidence_level=confidence_level,
            seed=bootstrap_seed + backend_index * 10_000,
        )
        backend_metrics = aggregate_metrics(backend_records)
        backend_rows.append(
            {
                "backend": backend,
                "selected_task_mean_score": interval.point_estimate,
                "score_ci_lower": interval.lower,
                "score_ci_upper": interval.upper,
                "task_count": len(tasks),
                **backend_metrics,
            }
        )

    for row in backend_rows:
        row["pareto_accuracy_vs_gqa_access"] = _is_pareto(row, backend_rows)

    _write_csv(root / "task_summary.csv", task_rows)
    _write_csv(root / "backend_summary.csv", backend_rows)
    _write_csv(
        root / "pareto.csv",
        sorted(
            backend_rows,
            key=lambda row: (_finite(row.get("decode_gqa_total_access_pct")) or math.inf),
        ),
    )

    lines = [
        "# RULER summary",
        "",
        "| Backend | Mean score | 95% CI | GQA total access | Decode tok/s | Pareto |",
        "|---|---:|---:|---:|---:|:---:|",
    ]
    for row in backend_rows:
        access = _finite(row.get("decode_gqa_total_access_pct"))
        speed = _finite(row.get("decode_tokens_per_second"))
        lines.append(
            "| {backend} | {score:.2f} | [{lo:.2f}, {hi:.2f}] | {access} | "
            "{speed} | {pareto} |".format(
                backend=row["backend"],
                score=float(row["selected_task_mean_score"]),
                lo=float(row["score_ci_lower"]),
                hi=float(row["score_ci_upper"]),
                access=f"{access:.2f}%" if access is not None else "n/a",
                speed=f"{speed:.2f}" if speed is not None else "n/a",
                pareto="yes" if row["pareto_accuracy_vs_gqa_access"] else "no",
            )
        )
    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"tasks": task_rows, "backends": backend_rows}
