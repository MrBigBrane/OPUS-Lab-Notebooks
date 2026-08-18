"""Prediction loading, RULER grading, bootstrap intervals, and reports."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from statistics import fmean
from typing import Any

from .ruler.grader import (
    BootstrapInterval,
    grade_task,
    stratified_bootstrap_mean_confidence_interval,
)


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


def _derived_seed(base_seed: int, *parts: object) -> int:
    digest = hashlib.sha256(
        ":".join([str(base_seed), *(str(part) for part in parts)]).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _finite_float(value: object) -> float | None:
    if not _is_number(value):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _safe_mean(values: Iterable[object]) -> float | None:
    cleaned = [number for value in values if (number := _finite_float(value)) is not None]
    return fmean(cleaned) if cleaned else None


def _weighted_mean(pairs: Iterable[tuple[object, object]]) -> float | None:
    numerator = 0.0
    denominator = 0.0
    for raw_value, raw_weight in pairs:
        value = _finite_float(raw_value)
        weight = _finite_float(raw_weight)
        if value is None or weight is None or weight <= 0.0:
            continue
        numerator += value * weight
        denominator += weight
    return numerator / denominator if denominator else None


def _pct(numerator: object, denominator: object) -> float | None:
    top = _finite_float(numerator)
    bottom = _finite_float(denominator)
    if top is None or bottom is None or bottom == 0.0:
        return None
    return 100.0 * top / bottom


def _constant_metric(metrics: Sequence[Mapping[str, Any]], key: str) -> Any:
    present = [metric[key] for metric in metrics if key in metric]
    if not present:
        return None
    first = present[0]
    if any(value != first for value in present[1:]):
        raise ValueError(f"Metric {key!r} changed within one report aggregation.")
    return first


def _sum_metric(metrics: Sequence[Mapping[str, Any]], key: str) -> int | float:
    values = [metric.get(key) for metric in metrics]
    numeric = [_finite_float(value) for value in values]
    if any(value is None for value in numeric):
        raise ValueError(f"Additive metric {key!r} is missing or non-finite.")
    total = sum(value for value in numeric if value is not None)
    if all(isinstance(value, int) and not isinstance(value, bool) for value in values):
        return int(total)
    return float(total)


_IDENTITY_KEYS = (
    "mode",
    "sampling_scheme",
    "importance_correction",
    "sampling_unit",
    "routing_key_type",
    "samples_per_head",
    "nominal_sample_budget_per_head",
    "clusters_per_head",
    "teams_per_head",
    "group_size",
    "parent_size",
    "representatives_per_parent",
    "nominal_team_size",
    "representative_selection",
    "parent_clustering_space",
    "team_assignment_space",
    "probe_queries",
    "probe_policy",
    "exact_token_policy",
    "prompt_exact_tail_tokens",
    "initial_growing_exact_tokens",
    "num_query_heads",
    "num_kv_heads",
    "query_heads_per_kv",
    "head_dim",
)

_ADDITIVE_ACCESS_KEYS = (
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

_ROUTING_FALLBACK_KEYS = (
    (
        "decode_gqa_routing_key_vectors_read",
        "decode_gqa_centroid_key_vectors_read",
    ),
    (
        "decode_naive_routing_key_vectors_read",
        "decode_naive_centroid_key_vectors_read",
    ),
)


def _normalize_legacy_routing_metrics(metric: Mapping[str, Any]) -> dict[str, Any]:
    """Add generic routing counts to legacy centroid-only metric records."""

    normalized = dict(metric)
    for generic_key, centroid_key in _ROUTING_FALLBACK_KEYS:
        if generic_key not in normalized:
            if centroid_key not in normalized:
                raise ValueError(
                    f"Metric {generic_key!r} and legacy fallback "
                    f"{centroid_key!r} are both missing."
                )
            normalized[generic_key] = normalized[centroid_key]
    return normalized


def _optional_sum_metric(metrics: Sequence[Mapping[str, Any]], key: str) -> int | float | None:
    present = [metric.get(key) is not None for metric in metrics]
    if not any(present):
        return None
    if not all(present):
        raise ValueError(f"Metric {key!r} is missing from part of one aggregation.")
    return _sum_metric(metrics, key)


def _pool_size_statistics(
    result: dict[str, Any],
    metrics: Sequence[Mapping[str, Any]],
    *,
    unit: str,
) -> None:
    count_key = f"active_{unit}_count_total"
    mean_key = f"{unit}_size_mean"
    std_key = f"{unit}_size_std"
    parts: list[tuple[float, float, float]] = []
    for metric in metrics:
        count = _finite_float(metric.get(count_key))
        mean = _finite_float(metric.get(mean_key))
        std = _finite_float(metric.get(std_key))
        if count is not None and count > 0.0 and mean is not None and std is not None:
            parts.append((count, mean, std))

    if not parts:
        return

    total = sum(count for count, _, _ in parts)
    pooled_mean = sum(count * mean for count, mean, _ in parts) / total
    second = sum(count * (std * std + mean * mean) for count, mean, std in parts) / total
    pooled_std = math.sqrt(max(0.0, second - pooled_mean * pooled_mean))
    minimums = [
        int(value)
        for metric in metrics
        if (value := _finite_float(metric.get(f"{unit}_size_min"))) is not None
    ]
    maximums = [
        int(value)
        for metric in metrics
        if (value := _finite_float(metric.get(f"{unit}_size_max"))) is not None
    ]
    median = _safe_mean(metric.get(f"{unit}_size_median") for metric in metrics)
    p90 = _safe_mean(metric.get(f"{unit}_size_p90") for metric in metrics)
    result.update(
        {
            count_key: int(total),
            f"pooled_active_{unit}_count": int(total),
            mean_key: pooled_mean,
            std_key: pooled_std,
            f"{unit}_size_cv": pooled_std / pooled_mean if pooled_mean else 0.0,
            f"{unit}_size_min": min(minimums) if minimums else None,
            f"{unit}_size_max": max(maximums) if maximums else None,
            f"{unit}_size_median": median,
            f"{unit}_size_p90": p90,
            f"mean_{unit}_size_median_across_examples": median,
            f"mean_{unit}_size_p90_across_examples": p90,
            f"mean_active_{unit}s_per_layer_kv_head": _safe_mean(
                metric.get(f"mean_active_{unit}s_per_layer_kv_head") for metric in metrics
            ),
        }
    )


def _aggregate_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("Cannot aggregate an empty prediction set.")
    raw_metrics = [record.get("metrics", {}) for record in records]
    if not all(isinstance(metric, Mapping) for metric in raw_metrics):
        raise ValueError("Every prediction record must contain a metrics mapping.")
    metrics = [
        _normalize_legacy_routing_metrics(metric)
        for metric in raw_metrics
        if isinstance(metric, Mapping)
    ]

    result: dict[str, Any] = {key: _constant_metric(metrics, key) for key in _IDENTITY_KEYS}
    generated = sum(int(metric.get("generated_tokens", 0)) for metric in metrics)
    prefill_seconds = sum(float(metric.get("prefill_seconds", 0.0)) for metric in metrics)
    clustering_seconds = sum(float(metric.get("clustering_seconds", 0.0)) for metric in metrics)
    decode_seconds = sum(float(metric.get("decode_seconds", 0.0)) for metric in metrics)
    total_seconds = sum(float(metric.get("total_seconds", 0.0)) for metric in metrics)
    result.update(
        {
            "num_examples": len(records),
            "total_generated_tokens": generated,
            "mean_prompt_tokens": _safe_mean(metric.get("prompt_tokens") for metric in metrics),
            "mean_generated_tokens": generated / len(records),
            # Prompt lengths vary within a RULER task, so the number of
            # parent clusters is a per-example diagnostic rather than a
            # backend identity.  Keep the original metric name as the pooled
            # per-example mean and expose an explicit alias for readers.
            "nominal_parent_clusters_per_kv_head": _safe_mean(
                metric.get("nominal_parent_clusters_per_kv_head")
                for metric in metrics
            ),
            "mean_nominal_parent_clusters_per_kv_head": _safe_mean(
                metric.get("nominal_parent_clusters_per_kv_head")
                for metric in metrics
            ),
            "mean_prefill_seconds": prefill_seconds / len(records),
            "mean_clustering_seconds": clustering_seconds / len(records),
            "mean_decode_seconds": decode_seconds / len(records),
            "mean_total_seconds": total_seconds / len(records),
            "decode_tokens_per_second": (
                generated / decode_seconds if decode_seconds > 0.0 else None
            ),
            "end_to_end_output_tokens_per_second": (
                generated / total_seconds if total_seconds > 0.0 else None
            ),
            "peak_allocated_gib": max(
                (float(metric.get("peak_allocated_gib", 0.0)) for metric in metrics),
                default=0.0,
            ),
            "peak_reserved_gib": max(
                (float(metric.get("peak_reserved_gib", 0.0)) for metric in metrics),
                default=0.0,
            ),
            "mean_custom_cache_gib": _safe_mean(
                metric.get("custom_cache_gib") for metric in metrics
            ),
            "mean_cluster_summary_gib": _safe_mean(
                metric.get("cluster_summary_gib") for metric in metrics
            ),
            "mean_routing_summary_gib": _safe_mean(
                metric.get("routing_summary_gib", metric.get("cluster_summary_gib"))
                for metric in metrics
            ),
        }
    )

    for key in _ADDITIVE_ACCESS_KEYS:
        result[key] = _sum_metric(metrics, key)

    dense_gqa = result["decode_dense_gqa_kv_vectors"]
    dense_naive = result["decode_dense_naive_kv_vectors"]
    gqa_kv = result["decode_gqa_kv_vectors_read"]
    naive_kv = result["decode_naive_kv_vectors_read"]
    gqa_routing = result["decode_gqa_routing_key_vectors_read"]
    naive_routing = result["decode_naive_routing_key_vectors_read"]
    gqa_centroids = result["decode_gqa_centroid_key_vectors_read"]
    naive_centroids = result["decode_naive_centroid_key_vectors_read"]
    # Recompute these from pooled raw counts instead of averaging percentages.
    result.update(
        {
            "decode_gqa_total_vectors_read": gqa_kv + gqa_routing,
            "decode_naive_total_vectors_read": naive_kv + naive_routing,
            "decode_gqa_kv_access_pct": _pct(gqa_kv, dense_gqa),
            "decode_gqa_routing_access_pct": _pct(gqa_routing, dense_gqa),
            "decode_gqa_centroid_access_pct": _pct(gqa_centroids, dense_gqa),
            "decode_gqa_total_access_pct": _pct(gqa_kv + gqa_routing, dense_gqa),
            "decode_naive_kv_access_pct": _pct(naive_kv, dense_naive),
            "decode_naive_routing_access_pct": _pct(naive_routing, dense_naive),
            "decode_naive_centroid_access_pct": _pct(naive_centroids, dense_naive),
            "decode_naive_total_access_pct": _pct(naive_kv + naive_routing, dense_naive),
        }
    )

    head_calls = result["decode_attention_head_calls"]
    group_calls = result["decode_gqa_group_calls"]
    for key in (
        "mean_sampled_token_draws_per_head_call",
        "mean_sampled_token_rows_per_head_call",
        "mean_selected_clusters_per_head_call",
        "mean_selected_teams_per_head_call",
    ):
        result[key] = _weighted_mean(
            (metric.get(key), metric.get("decode_attention_head_calls", 0))
            for metric in metrics
        )
    for key in (
        "mean_unique_sampled_tokens_per_gqa_group_call",
        "mean_exact_tokens_per_gqa_group_call",
    ):
        result[key] = _weighted_mean(
            (metric.get(key), metric.get("decode_gqa_group_calls", 0)) for metric in metrics
        )
    result["mean_total_tokens_per_head_call"] = _weighted_mean(
        (
            metric.get("mean_total_tokens_per_head_call"),
            metric.get("decode_attention_head_calls", 0),
        )
        for metric in metrics
    )
    for unit in ("cluster", "team"):
        probability_count_key = f"selected_{unit}_inclusion_probability_count"
        probability_mean_key = f"mean_selected_{unit}_inclusion_probability"
        result[probability_mean_key] = _weighted_mean(
            (
                metric.get(probability_mean_key),
                metric.get(probability_count_key, 0),
            )
            for metric in metrics
        )
        result[probability_count_key] = sum(
            int(metric.get(probability_count_key, 0)) for metric in metrics
        )
        for bound in ("min", "max"):
            key = f"{bound}_selected_{unit}_inclusion_probability"
            values = [
                value
                for metric in metrics
                if (value := _finite_float(metric.get(key))) is not None
            ]
            if not values:
                result[key] = None
            elif bound == "min":
                result[key] = min(values)
            else:
                result[key] = max(values)

    _pool_size_statistics(result, metrics, unit="cluster")
    _pool_size_statistics(result, metrics, unit="team")

    parent_count = _optional_sum_metric(metrics, "active_parent_count_total")
    if parent_count is not None:
        result["active_parent_count_total"] = parent_count
        result["pooled_active_parent_count"] = parent_count
        result["mean_active_parents_per_layer_kv_head"] = _safe_mean(
            metric.get("mean_active_parents_per_layer_kv_head") for metric in metrics
        )

    if head_calls and group_calls:
        result["observed_query_heads_per_gqa_group"] = head_calls / group_calls
    return result


def _ci_fields(interval: BootstrapInterval, *, prefix: str = "score_ci") -> dict[str, Any]:
    return {
        f"{prefix}_point_estimate": interval.point_estimate,
        f"{prefix}_lower": interval.lower,
        f"{prefix}_upper": interval.upper,
        f"{prefix}_confidence_level": interval.confidence_level,
        f"{prefix}_resamples": interval.resamples,
        f"{prefix}_seed": interval.seed,
        f"{prefix}_method": interval.method,
    }


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


def _write_official_style(
    backend_dir: Path,
    task_rows: Sequence[Mapping[str, Any]],
    records_by_task: Mapping[str, Sequence[Mapping[str, Any]]],
) -> None:
    tasks = [str(row["task"]) for row in task_rows]
    with (backend_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Tasks", *tasks])
        writer.writerow(["Score", *(row["score"] for row in task_rows)])
        writer.writerow(["Nulls", *(row["nulls"] for row in task_rows)])

    with (backend_dir / "submission.csv").open("w", encoding="utf-8", newline="") as handle:
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


def _uid_score_map(
    records: Sequence[Mapping[str, Any]], scores: Sequence[float]
) -> dict[str, float]:
    result: dict[str, float] = {}
    for record, score in zip(records, scores, strict=True):
        uid = str(record.get("uid") or f"{record.get('task')}:{record.get('index')}")
        if uid in result:
            raise ValueError(f"Duplicate prompt UID in predictions: {uid}")
        result[uid] = float(score)
    return result


def _paired_score_comparison(
    *,
    candidate: str,
    reference: str,
    tasks: Sequence[str],
    scores: Mapping[str, Mapping[str, Mapping[str, float]]],
    bootstrap_resamples: int,
    confidence_level: float,
    bootstrap_seed: int,
) -> dict[str, Any]:
    differences: dict[str, list[float]] = {}
    candidate_task_means: list[float] = []
    reference_task_means: list[float] = []
    for task in tasks:
        candidate_map = scores[candidate][task]
        reference_map = scores[reference][task]
        if set(candidate_map) != set(reference_map):
            raise ValueError(
                f"Paired comparison requires identical prompt UIDs for "
                f"{candidate!r} and {reference!r} on task {task!r}."
            )
        uids = sorted(candidate_map)
        differences[task] = [candidate_map[uid] - reference_map[uid] for uid in uids]
        candidate_task_means.append(fmean(candidate_map[uid] for uid in uids))
        reference_task_means.append(fmean(reference_map[uid] for uid in uids))

    interval = stratified_bootstrap_mean_confidence_interval(
        differences,
        resamples=bootstrap_resamples,
        confidence_level=confidence_level,
        seed=_derived_seed(bootstrap_seed, candidate, reference, "paired"),
    )
    return {
        "candidate": candidate,
        "reference": reference,
        "candidate_selected_task_mean_score": fmean(candidate_task_means),
        "reference_selected_task_mean_score": fmean(reference_task_means),
        "score_delta_candidate_minus_reference": interval.point_estimate,
        "confidence_interval": interval.to_dict(),
        "pairing": "prompt UID within task; tasks equally weighted",
    }


def _speedup(
    reference: Mapping[str, Any], candidate: Mapping[str, Any], key: str
) -> float | None:
    reference_value = _finite_float(reference.get(key))
    candidate_value = _finite_float(candidate.get(key))
    if reference_value is None or candidate_value is None or candidate_value == 0.0:
        return None
    return reference_value / candidate_value


def _memory_snapshot(aggregate: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "decode_gqa_total_access_pct",
        "decode_gqa_kv_access_pct",
        "decode_gqa_routing_access_pct",
        "decode_gqa_centroid_access_pct",
        "decode_naive_total_access_pct",
        "decode_naive_kv_access_pct",
        "decode_naive_routing_access_pct",
        "decode_naive_centroid_access_pct",
        "decode_gqa_total_vectors_read",
        "decode_gqa_kv_vectors_read",
        "decode_gqa_routing_key_vectors_read",
        "decode_gqa_centroid_key_vectors_read",
        "decode_dense_gqa_kv_vectors",
        "mean_sampled_token_rows_per_head_call",
        "mean_unique_sampled_tokens_per_gqa_group_call",
        "mean_exact_tokens_per_gqa_group_call",
    )
    return {key: aggregate.get(key) for key in keys}


def _comparison_row(
    comparison_type: str,
    comparison: Mapping[str, Any],
    *,
    nominal_budget: int | None,
) -> dict[str, Any]:
    interval = comparison["confidence_interval"]
    performance = comparison.get("performance", {})
    memory = comparison.get("candidate_memory", {})
    return {
        "comparison_type": comparison_type,
        "candidate": comparison["candidate"],
        "reference": comparison["reference"],
        "nominal_budget": nominal_budget,
        "score_delta": comparison["score_delta_candidate_minus_reference"],
        "score_delta_ci_lower": interval["lower"],
        "score_delta_ci_upper": interval["upper"],
        "confidence_level": interval["confidence_level"],
        **performance,
        **memory,
    }


def _comparison_with_performance(
    *,
    candidate: str,
    reference: str,
    tasks: Sequence[str],
    scores: Mapping[str, Mapping[str, Mapping[str, float]]],
    structured_backends: Mapping[str, Mapping[str, Any]],
    bootstrap_resamples: int,
    confidence_level: float,
    bootstrap_seed: int,
) -> dict[str, Any]:
    comparison = _paired_score_comparison(
        candidate=candidate,
        reference=reference,
        tasks=tasks,
        scores=scores,
        bootstrap_resamples=bootstrap_resamples,
        confidence_level=confidence_level,
        bootstrap_seed=bootstrap_seed,
    )
    reference_aggregate = structured_backends[reference]["aggregate"]
    candidate_aggregate = structured_backends[candidate]["aggregate"]
    comparison["performance"] = {
        "end_to_end_speedup_candidate_vs_reference": _speedup(
            reference_aggregate, candidate_aggregate, "mean_total_seconds"
        ),
        "decode_speedup_candidate_vs_reference": _speedup(
            reference_aggregate, candidate_aggregate, "mean_decode_seconds"
        ),
    }
    comparison["candidate_memory"] = _memory_snapshot(candidate_aggregate)
    return comparison


def build_reports(
    run_dir: str | Path,
    *,
    backends: Iterable[str],
    tasks: Iterable[str],
    bootstrap_resamples: int = 10_000,
    confidence_level: float = 0.95,
    bootstrap_seed: int = 20_260_805,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    backends = list(backends)
    tasks = list(tasks)
    if bootstrap_resamples <= 0:
        raise ValueError("bootstrap_resamples must be positive.")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be strictly between zero and one.")

    report_rows: list[dict[str, Any]] = []
    per_example_rows: list[dict[str, Any]] = []
    score_maps: dict[str, dict[str, dict[str, float]]] = {}
    structured: dict[str, Any] = {
        "schema_version": 3,
        "grading": {
            "confidence_level": confidence_level,
            "bootstrap_resamples": bootstrap_resamples,
            "bootstrap_seed": bootstrap_seed,
            "task_method": "percentile bootstrap over prompts",
            "aggregate_method": (
                "stratified percentile bootstrap within task; tasks equally weighted"
            ),
            "comparison_method": "paired by prompt UID within task",
        },
        "access_accounting": {
            "scope": "conceptual logical decode vector-row reads, not hardware transactions",
            "primary": (
                "GQA-aware K/V union across query heads sharing each KV head, plus "
                "generic routing-key reads, divided by dense GQA K/V reads"
            ),
            "gqa_kv": "sampled K/V rows unioned across sharing query heads",
            "gqa_routing": (
                "centroid or actual-team-leader key-vector reads; legacy records "
                "fall back to their centroid counts"
            ),
            "gqa_centroid": "legacy centroid subset reported separately",
            "naive": (
                "older per-query-head convention that counts dense K/V separately "
                "for every query head"
            ),
        },
        "backends": {},
    }

    for backend in backends:
        backend_dir = run_dir / "predictions" / backend
        records_by_task: dict[str, list[dict[str, Any]]] = {}
        task_rows: list[dict[str, Any]] = []
        all_records: list[dict[str, Any]] = []
        raw_scores_by_task: dict[str, Sequence[float]] = {}
        score_maps[backend] = {}

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
            grade = grade_task(
                task,
                predictions,
                references,
                bootstrap_resamples=bootstrap_resamples,
                confidence_level=confidence_level,
                bootstrap_seed=_derived_seed(bootstrap_seed, backend, task),
            )
            if grade.confidence_interval is None:
                raise RuntimeError("Task grading did not produce a confidence interval.")
            raw_scores_by_task[task] = grade.example_scores
            score_maps[backend][task] = _uid_score_map(records, grade.example_scores)
            aggregate = _aggregate_metrics(records)
            row = {
                "backend": backend,
                "task": task,
                "score": grade.score,
                "nulls": grade.nulls_label,
                **_ci_fields(grade.confidence_interval),
                **aggregate,
            }
            report_rows.append(row)
            task_rows.append(row)

            for record, example_score in zip(records, grade.example_scores, strict=True):
                example_row = {
                    "backend": backend,
                    "task": task,
                    "index": record.get("index"),
                    "uid": record.get("uid"),
                    "example_score": example_score,
                    "prediction": record.get("pred", ""),
                    "references": json.dumps(record.get("outputs", []), ensure_ascii=False),
                }
                for key, value in record.get("metrics", {}).items():
                    example_row[f"metric_{key}"] = value
                per_example_rows.append(example_row)

        overall_interval = stratified_bootstrap_mean_confidence_interval(
            raw_scores_by_task,
            resamples=bootstrap_resamples,
            confidence_level=confidence_level,
            seed=_derived_seed(bootstrap_seed, backend, "all_tasks"),
        )
        aggregate_all = _aggregate_metrics(all_records)
        aggregate_row = {
            "backend": backend,
            "task": "__selected_task_mean__",
            "score": round(overall_interval.point_estimate, 2),
            "nulls": (
                f"{sum(not str(record.get('pred', '')).strip() for record in all_records)}"
                f"/{len(all_records)}"
            ),
            **_ci_fields(overall_interval),
            **aggregate_all,
        }
        report_rows.append(aggregate_row)
        structured["backends"][backend] = {
            "selected_task_unweighted_mean_score": aggregate_row["score"],
            "score_confidence_interval": overall_interval.to_dict(),
            "algorithm": {key: aggregate_row.get(key) for key in _IDENTITY_KEYS},
            "tasks": {row["task"]: row for row in task_rows},
            "aggregate": aggregate_row,
        }
        _write_official_style(backend_dir, task_rows, records_by_task)

    comparisons: dict[str, Any] = {"vs_sdpa": {}}
    comparison_rows: list[dict[str, Any]] = []
    if "sdpa" in structured["backends"]:
        for backend in backends:
            if backend == "sdpa":
                continue
            comparison = _comparison_with_performance(
                candidate=backend,
                reference="sdpa",
                tasks=tasks,
                scores=score_maps,
                structured_backends=structured["backends"],
                bootstrap_resamples=bootstrap_resamples,
                confidence_level=confidence_level,
                bootstrap_seed=bootstrap_seed,
            )
            candidate_aggregate = structured["backends"][backend]["aggregate"]
            comparisons["vs_sdpa"][backend] = comparison
            comparison_rows.append(
                _comparison_row(
                    "vs_sdpa",
                    comparison,
                    nominal_budget=candidate_aggregate.get(
                        "nominal_sample_budget_per_head"
                    ),
                )
            )

    structured["comparisons"] = comparisons
    _write_csv(run_dir / "summary.csv", report_rows)
    _write_csv(run_dir / "per_example.csv", per_example_rows)
    _write_csv(run_dir / "comparisons.csv", comparison_rows)
    with (run_dir / "summary.json").open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(structured, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")

    def fmt(value: Any, digits: int = 2) -> str:
        number = _finite_float(value)
        return "—" if number is None else f"{number:.{digits}f}"

    lines = [
        "# SANTA-family RULER benchmark summary",
        "",
        (
            "The primary access percentage is GQA-aware K/V union plus generic "
            "routing-key reads, relative to dense GQA K/V reads. The centroid "
            "column is the legacy centroid subset of routing, not an extra read. "
            "Values are theoretical logical vector-row estimates, not measured "
            "DRAM traffic."
        ),
        "",
        "| Backend | Task | Score | 95% CI | Mean total s | Mean decode s | GQA total | GQA KV | GQA routing K | GQA centroid subset | Naive total |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report_rows:
        lines.append(
            "| {backend} | {task} | {score} | [{lower}, {upper}] | {total} | "
            "{decode} | {gqa_total}% | {gqa_kv}% | {routing}% | {centroid}% | "
            "{naive}% |".format(
                backend=row["backend"],
                task=row["task"],
                score=fmt(row.get("score")),
                lower=fmt(row.get("score_ci_lower")),
                upper=fmt(row.get("score_ci_upper")),
                total=fmt(row.get("mean_total_seconds"), 3),
                decode=fmt(row.get("mean_decode_seconds"), 3),
                gqa_total=fmt(row.get("decode_gqa_total_access_pct"), 1),
                gqa_kv=fmt(row.get("decode_gqa_kv_access_pct"), 1),
                routing=fmt(row.get("decode_gqa_routing_access_pct"), 1),
                centroid=fmt(row.get("decode_gqa_centroid_access_pct"), 1),
                naive=fmt(row.get("decode_naive_total_access_pct"), 1),
            )
        )

    lines.extend(
        [
            "",
            "Task scores use the vendored RULER task-family grader. The selected-task mean is an unweighted harness summary.",
            "",
        ]
    )
    (run_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return structured
