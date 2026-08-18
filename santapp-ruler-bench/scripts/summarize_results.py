#!/usr/bin/env python3
"""Summarize one or more compact Kubernetes exports.

The script discovers any export directory containing ``matrix.json`` and
``compact-results.csv``. Inputs may be export directories, parent directories,
or ``.tar.gz``/``.tgz`` archives produced by ``k8s_export_results.py``.

Unlike an experiment-specific plotting script, this module derives tasks,
settings, labels, and hyperparameters from each export. It writes one analysis
directory containing accuracy tables, logical-access tables, paired score
comparisons, Pareto membership, and accuracy-versus-access plots.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import tarfile
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

DEFAULT_OUTPUT_DIR = "analysis"
SUMMARY_TASK = "__selected_task_mean__"


@dataclass(frozen=True, slots=True)
class ExportSource:
    root: Path
    matrix: dict[str, Any]
    rows: tuple[dict[str, str], ...]


@dataclass(frozen=True, slots=True)
class Interval:
    point: float
    lower: float
    upper: float
    sample_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "point_estimate": self.point,
            "lower": self.lower,
            "upper": self.upper,
            "sample_count": self.sample_count,
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        action="append",
        required=True,
        help=(
            "Export directory, parent directory, or compact-results .tar.gz/.tgz "
            "archive. Repeat to merge multiple compatible runs."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(DEFAULT_OUTPUT_DIR),
        help=f"Single analysis directory to create (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument(
        "--reference-setting",
        default="auto",
        help=(
            "Generate paired comparisons against this setting. 'auto' uses an exact "
            "'sdpa' setting when present; 'none' disables automatic comparisons."
        ),
    )
    parser.add_argument(
        "--comparison",
        action="append",
        default=[],
        metavar="CANDIDATE:REFERENCE",
        help="Additional paired comparison. Repeat as needed.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow exports whose export-status.json reports partial completion.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output directory instead of failing.",
    )
    return parser


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _integer(value: Any) -> int | None:
    number = _finite_float(value)
    if number is None or not number.is_integer():
        return None
    return int(number)


def _safe_extract(archive_path: Path, destination: Path) -> None:
    destination = destination.resolve()
    with tarfile.open(archive_path, "r:*") as archive:
        members = archive.getmembers()
        for member in members:
            if member.issym() or member.islnk():
                raise ValueError(f"Archive links are not allowed: {member.name}")
            target = (destination / member.name).resolve()
            try:
                target.relative_to(destination)
            except ValueError as exc:
                raise ValueError(
                    f"Archive member escapes extraction root: {member.name}"
                ) from exc
        try:
            archive.extractall(destination, members=members, filter="data")
        except TypeError:  # Python versions without extraction filters.
            archive.extractall(destination, members=members)


def _candidate_export_roots(path: Path) -> list[Path]:
    roots: list[Path] = []
    if (path / "matrix.json").is_file() and (path / "compact-results.csv").is_file():
        roots.append(path)
    if path.is_dir():
        for result_path in path.rglob("compact-results.csv"):
            parent = result_path.parent
            if (parent / "matrix.json").is_file():
                roots.append(parent)
    return sorted({root.resolve() for root in roots})


def _load_json_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _load_export(root: Path, *, allow_partial: bool) -> ExportSource:
    matrix = _load_json_object(root / "matrix.json")
    status_path = root / "export-status.json"
    if status_path.is_file():
        status = _load_json_object(status_path)
        if status.get("status") != "complete" and not allow_partial:
            raise ValueError(
                f"Export is not complete: {root}. Pass --allow-partial to analyze it."
            )

    with (root / "compact-results.csv").open(encoding="utf-8", newline="") as handle:
        rows = tuple(dict(row) for row in csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No compact result rows found in {root}")

    required = {"setting", "task", "uid"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"{root} is missing required columns: {sorted(missing)}")
    return ExportSource(root=root, matrix=matrix, rows=rows)


def discover_exports(
    inputs: Sequence[Path], *, allow_partial: bool, scratch: Path
) -> list[ExportSource]:
    roots: list[Path] = []
    for index, raw_path in enumerate(inputs):
        path = raw_path.expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        if path.is_file():
            if not (path.name.endswith(".tar.gz") or path.suffix == ".tgz"):
                raise ValueError(f"Unsupported input file type: {path}")
            extracted = scratch / f"archive-{index:03d}"
            extracted.mkdir(parents=True)
            _safe_extract(path, extracted)
            roots.extend(_candidate_export_roots(extracted))
        else:
            roots.extend(_candidate_export_roots(path))

    unique_roots = sorted(set(roots))
    if not unique_roots:
        raise ValueError(
            "No export containing matrix.json and compact-results.csv was discovered."
        )
    return [_load_export(root, allow_partial=allow_partial) for root in unique_roots]


def _matrix_settings(matrix: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = matrix.get("settings", [])
    if not isinstance(raw, list):
        return []
    settings: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, str):
            settings.append({"name": item})
        elif isinstance(item, Mapping) and item.get("name"):
            settings.append(dict(item))
    return settings


def merge_exports(
    exports: Sequence[ExportSource],
) -> tuple[list[dict[str, str]], dict[str, dict[str, Any]], list[str], list[str]]:
    merged: dict[tuple[str, str, str], dict[str, str]] = {}
    setting_metadata: dict[str, dict[str, Any]] = {}
    task_order: list[str] = []
    setting_order: list[str] = []

    for source in exports:
        matrix_tasks = source.matrix.get("tasks", [])
        if isinstance(matrix_tasks, list):
            for task in matrix_tasks:
                name = str(task)
                if name not in task_order:
                    task_order.append(name)
        for metadata in _matrix_settings(source.matrix):
            name = str(metadata["name"])
            if name not in setting_order:
                setting_order.append(name)
            previous = setting_metadata.get(name)
            if previous is None:
                setting_metadata[name] = metadata
            else:
                common = set(previous) & set(metadata)
                conflicts = {
                    key: (previous[key], metadata[key])
                    for key in common
                    if previous[key] != metadata[key]
                    and key not in {"description"}
                }
                if conflicts:
                    raise ValueError(
                        f"Conflicting metadata for setting {name!r}: {conflicts}"
                    )
                previous.update({k: v for k, v in metadata.items() if k not in previous})

        for row in source.rows:
            setting = str(row.get("setting", "")).strip()
            task = str(row.get("task", "")).strip()
            uid = str(row.get("uid", "")).strip()
            if not setting or not task or not uid:
                raise ValueError(f"Empty setting/task/uid in {source.root}")
            key = (setting, task, uid)
            previous = merged.get(key)
            if previous is not None and previous != row:
                raise ValueError(
                    "Conflicting duplicate compact result for "
                    f"setting={setting!r}, task={task!r}, uid={uid!r}."
                )
            merged[key] = row
            if setting not in setting_order:
                setting_order.append(setting)
            if task not in task_order:
                task_order.append(task)
            setting_metadata.setdefault(
                setting,
                {
                    "name": setting,
                    "backend": row.get("backend") or setting,
                },
            )

    return list(merged.values()), setting_metadata, task_order, setting_order


def _score(row: Mapping[str, Any]) -> float:
    value = _finite_float(
        row.get("example_ruler_score", row.get("metric_example_ruler_score"))
    )
    if value is None:
        raise ValueError(
            f"Missing finite example_ruler_score for {row.get('setting')}/{row.get('task')}/"
            f"{row.get('uid')}"
        )
    return value


def _percentile_bounds(
    samples: np.ndarray, confidence_level: float
) -> tuple[float, float]:
    alpha = (1.0 - confidence_level) / 2.0
    lower, upper = np.quantile(samples, [alpha, 1.0 - alpha])
    return float(lower), float(upper)


def _bootstrap_mean(
    values: Sequence[float],
    *,
    resamples: int,
    confidence_level: float,
    seed: int,
) -> Interval:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("Cannot bootstrap an empty sample.")
    point = float(array.mean())
    if array.size == 1:
        return Interval(point, point, point, 1)
    rng = np.random.default_rng(seed)
    draws = rng.choice(array, size=(resamples, array.size), replace=True).mean(axis=1)
    lower, upper = _percentile_bounds(draws, confidence_level)
    return Interval(point, lower, upper, int(array.size))


def _bootstrap_stratified_mean(
    values_by_task: Mapping[str, Sequence[float]],
    *,
    resamples: int,
    confidence_level: float,
    seed: int,
) -> Interval:
    arrays = [
        np.asarray(values_by_task[task], dtype=np.float64)
        for task in values_by_task
        if values_by_task[task]
    ]
    if not arrays:
        raise ValueError("Cannot bootstrap an empty stratified sample.")
    task_points = np.asarray([array.mean() for array in arrays])
    point = float(task_points.mean())
    if all(array.size == 1 for array in arrays):
        return Interval(point, point, point, int(sum(array.size for array in arrays)))
    rng = np.random.default_rng(seed)
    task_draws: list[np.ndarray] = []
    for array in arrays:
        sampled = rng.choice(array, size=(resamples, array.size), replace=True)
        task_draws.append(sampled.mean(axis=1))
    aggregate = np.vstack(task_draws).mean(axis=0)
    lower, upper = _percentile_bounds(aggregate, confidence_level)
    return Interval(point, lower, upper, int(sum(array.size for array in arrays)))


def compute_accuracy(
    rows: Sequence[Mapping[str, Any]],
    *,
    settings: Sequence[str],
    tasks: Sequence[str],
    resamples: int,
    confidence_level: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["setting"]), str(row["task"]))].append(_score(row))

    by_task: list[dict[str, Any]] = []
    overall: list[dict[str, Any]] = []
    for setting_index, setting in enumerate(settings):
        values_by_task: dict[str, Sequence[float]] = {}
        for task_index, task in enumerate(tasks):
            values = grouped.get((setting, task), [])
            if not values:
                continue
            interval = _bootstrap_mean(
                values,
                resamples=resamples,
                confidence_level=confidence_level,
                seed=seed + setting_index * 10_003 + task_index,
            )
            values_by_task[task] = values
            by_task.append(
                {
                    "setting": setting,
                    "task": task,
                    "score": interval.point,
                    "score_ci_lower": interval.lower,
                    "score_ci_upper": interval.upper,
                    "sample_count": interval.sample_count,
                }
            )
        if values_by_task:
            interval = _bootstrap_stratified_mean(
                values_by_task,
                resamples=resamples,
                confidence_level=confidence_level,
                seed=seed + setting_index * 10_003 + 9_991,
            )
            overall.append(
                {
                    "setting": setting,
                    "task": SUMMARY_TASK,
                    "score": interval.point,
                    "score_ci_lower": interval.lower,
                    "score_ci_upper": interval.upper,
                    "sample_count": interval.sample_count,
                    "task_count": len(values_by_task),
                }
            )
    return by_task, overall


def _metric(row: Mapping[str, Any], *names: str) -> float | None:
    for name in names:
        key = name if name.startswith("metric_") else f"metric_{name}"
        value = _finite_float(row.get(key))
        if value is not None:
            return value
    return None


def _sum_metric(rows: Sequence[Mapping[str, Any]], *names: str) -> float | None:
    values = [_metric(row, *names) for row in rows]
    if not values or any(value is None for value in values):
        return None
    return float(sum(value for value in values if value is not None))


def _mean_metric(rows: Sequence[Mapping[str, Any]], *names: str) -> float | None:
    values = [_metric(row, *names) for row in rows]
    if not values or any(value is None for value in values):
        return None
    return float(sum(value for value in values if value is not None) / len(values))


def _pct(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0.0:
        return None
    return 100.0 * numerator / denominator


def pooled_access(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    dense_gqa = _sum_metric(rows, "decode_dense_gqa_kv_vectors")
    dense_naive = _sum_metric(rows, "decode_dense_naive_kv_vectors")
    gqa_kv = _sum_metric(rows, "decode_gqa_kv_vectors_read")
    naive_kv = _sum_metric(rows, "decode_naive_kv_vectors_read")
    gqa_routing = _sum_metric(
        rows,
        "decode_gqa_routing_key_vectors_read",
        "decode_gqa_centroid_key_vectors_read",
    )
    naive_routing = _sum_metric(
        rows,
        "decode_naive_routing_key_vectors_read",
        "decode_naive_centroid_key_vectors_read",
    )
    gqa_centroid = _sum_metric(rows, "decode_gqa_centroid_key_vectors_read")
    naive_centroid = _sum_metric(rows, "decode_naive_centroid_key_vectors_read")

    gqa_total = (
        None if gqa_kv is None or gqa_routing is None else gqa_kv + gqa_routing
    )
    naive_total = (
        None
        if naive_kv is None or naive_routing is None
        else naive_kv + naive_routing
    )

    accounting = "pooled_vector_counts"
    if dense_gqa is None or gqa_total is None:
        gqa_total_pct = _mean_metric(rows, "decode_gqa_total_access_pct")
        gqa_kv_pct = _mean_metric(rows, "decode_gqa_kv_access_pct")
        gqa_routing_pct = _mean_metric(
            rows,
            "decode_gqa_routing_access_pct",
            "decode_gqa_centroid_access_pct",
        )
        gqa_centroid_pct = _mean_metric(rows, "decode_gqa_centroid_access_pct")
        accounting = "mean_record_percentages"
    else:
        gqa_total_pct = _pct(gqa_total, dense_gqa)
        gqa_kv_pct = _pct(gqa_kv, dense_gqa)
        gqa_routing_pct = _pct(gqa_routing, dense_gqa)
        gqa_centroid_pct = _pct(gqa_centroid, dense_gqa)

    if dense_naive is None or naive_total is None:
        naive_total_pct = _mean_metric(rows, "decode_naive_total_access_pct")
        naive_kv_pct = _mean_metric(rows, "decode_naive_kv_access_pct")
        naive_routing_pct = _mean_metric(
            rows,
            "decode_naive_routing_access_pct",
            "decode_naive_centroid_access_pct",
        )
        naive_centroid_pct = _mean_metric(rows, "decode_naive_centroid_access_pct")
    else:
        naive_total_pct = _pct(naive_total, dense_naive)
        naive_kv_pct = _pct(naive_kv, dense_naive)
        naive_routing_pct = _pct(naive_routing, dense_naive)
        naive_centroid_pct = _pct(naive_centroid, dense_naive)

    return {
        "record_count": len(rows),
        "accounting_method": accounting,
        "decode_dense_gqa_kv_vectors": dense_gqa,
        "decode_gqa_kv_vectors_read": gqa_kv,
        "decode_gqa_routing_key_vectors_read": gqa_routing,
        "decode_gqa_centroid_key_vectors_read": gqa_centroid,
        "decode_gqa_total_vectors_read": gqa_total,
        "decode_gqa_kv_access_pct": gqa_kv_pct,
        "decode_gqa_routing_access_pct": gqa_routing_pct,
        "decode_gqa_centroid_access_pct": gqa_centroid_pct,
        "decode_gqa_total_access_pct": gqa_total_pct,
        "decode_dense_naive_kv_vectors": dense_naive,
        "decode_naive_kv_vectors_read": naive_kv,
        "decode_naive_routing_key_vectors_read": naive_routing,
        "decode_naive_centroid_key_vectors_read": naive_centroid,
        "decode_naive_total_vectors_read": naive_total,
        "decode_naive_kv_access_pct": naive_kv_pct,
        "decode_naive_routing_access_pct": naive_routing_pct,
        "decode_naive_centroid_access_pct": naive_centroid_pct,
        "decode_naive_total_access_pct": naive_total_pct,
    }


def compute_access_tables(
    rows: Sequence[Mapping[str, Any]],
    *,
    settings: Sequence[str],
    tasks: Sequence[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    by_setting: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        setting = str(row["setting"])
        task = str(row["task"])
        grouped[(setting, task)].append(row)
        by_setting[setting].append(row)

    task_rows: list[dict[str, Any]] = []
    setting_rows: list[dict[str, Any]] = []
    for setting in settings:
        for task in tasks:
            records = grouped.get((setting, task), [])
            if records:
                task_rows.append({"setting": setting, "task": task, **pooled_access(records)})
        records = by_setting.get(setting, [])
        if records:
            setting_rows.append(
                {"setting": setting, "task": SUMMARY_TASK, **pooled_access(records)}
            )
    return task_rows, setting_rows


def _metadata_number(metadata: Mapping[str, Any], key: str) -> int | None:
    value = metadata.get(key)
    if value is None and isinstance(metadata.get("parameters"), Mapping):
        value = metadata["parameters"].get(key)
    return _integer(value)


def setting_label(setting: str, metadata: Mapping[str, Any]) -> str:
    backend = str(metadata.get("backend") or setting)
    mode = str(metadata.get("mode") or "")
    samples = _metadata_number(metadata, "samples_per_head")
    group_size = _metadata_number(metadata, "group_size")
    parent_size = _metadata_number(metadata, "parent_size")
    representatives = _metadata_number(metadata, "representatives_per_parent")

    suffix: list[str] = []
    if group_size is not None:
        suffix.append(f"B={group_size}")
    if parent_size is not None and mode.startswith("team_"):
        suffix.append(f"P={parent_size}")
    if representatives is not None and mode.startswith("team_"):
        suffix.append(f"R={representatives}")
    if samples is not None:
        suffix.append(f"S={samples}")
    params = f" ({', '.join(suffix)})" if suffix else ""

    if backend == "sdpa" or mode == "dense_sdpa":
        return "SDPA"
    if backend == "santa":
        return f"SANTA{params}"
    if mode == "guided" or (backend == "santapp" and not mode):
        return f"SANTA++ IID{params}"
    if mode == "gumbel_cluster":
        return f"SANTA++ whole-parent sampling{params}"
    if mode == "team_sampler":
        return f"Hierarchical team sampling{params}"
    if mode == "team_gumbel_topk":
        return f"Whole-team sampling{params}"
    if backend == "santapp" and mode in {"uniform", "topk", "oracle_token"}:
        method = {
            "uniform": "uniform sampling",
            "topk": "top-k selection",
            "oracle_token": "oracle-token selection",
        }[mode]
        return f"SANTA++ {method}{params}"
    return setting


def _parse_comparisons(
    raw: Sequence[str], settings: Sequence[str], reference_setting: str
) -> list[tuple[str, str]]:
    available = set(settings)
    pairs: list[tuple[str, str]] = []
    normalized_reference = reference_setting.strip()
    if normalized_reference.lower() == "auto":
        normalized_reference = "sdpa" if "sdpa" in available else ""
    elif normalized_reference.lower() == "none":
        normalized_reference = ""
    if normalized_reference:
        if normalized_reference not in available:
            raise ValueError(
                f"Reference setting {normalized_reference!r} is not present. "
                f"Available settings: {sorted(available)}"
            )
        pairs.extend(
            (candidate, normalized_reference)
            for candidate in settings
            if candidate != normalized_reference
        )
    for expression in raw:
        if ":" not in expression:
            raise ValueError(
                f"Invalid --comparison {expression!r}; expected CANDIDATE:REFERENCE."
            )
        candidate, reference = (part.strip() for part in expression.split(":", 1))
        if candidate not in available or reference not in available:
            raise ValueError(
                f"Comparison {expression!r} names an unavailable setting. "
                f"Available settings: {sorted(available)}"
            )
        pairs.append((candidate, reference))
    return list(dict.fromkeys(pairs))


def _paired_comparison(
    rows: Sequence[Mapping[str, Any]],
    *,
    candidate: str,
    reference: str,
    tasks: Sequence[str],
    resamples: int,
    confidence_level: float,
    seed: int,
) -> dict[str, Any]:
    score_maps: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for row in rows:
        setting = str(row["setting"])
        if setting not in {candidate, reference}:
            continue
        score_maps[(setting, str(row["task"]))][str(row["uid"])] = _score(row)

    diffs_by_task: dict[str, list[float]] = {}
    task_rows: list[dict[str, Any]] = []
    for task_index, task in enumerate(tasks):
        candidate_scores = score_maps.get((candidate, task), {})
        reference_scores = score_maps.get((reference, task), {})
        if not candidate_scores and not reference_scores:
            continue
        if set(candidate_scores) != set(reference_scores):
            missing_candidate = sorted(set(reference_scores) - set(candidate_scores))[:5]
            missing_reference = sorted(set(candidate_scores) - set(reference_scores))[:5]
            raise ValueError(
                f"Paired comparison requires identical UIDs for {candidate} vs "
                f"{reference} on {task}; missing from candidate={missing_candidate}, "
                f"missing from reference={missing_reference}."
            )
        diffs = [
            candidate_scores[uid] - reference_scores[uid]
            for uid in sorted(candidate_scores)
        ]
        diffs_by_task[task] = diffs
        interval = _bootstrap_mean(
            diffs,
            resamples=resamples,
            confidence_level=confidence_level,
            seed=seed + task_index,
        )
        task_rows.append(
            {
                "task": task,
                "score_delta_candidate_minus_reference": interval.point,
                "ci_lower": interval.lower,
                "ci_upper": interval.upper,
                "paired_prompt_count": interval.sample_count,
            }
        )

    if not diffs_by_task:
        raise ValueError(f"No paired rows found for {candidate} vs {reference}.")
    overall = _bootstrap_stratified_mean(
        diffs_by_task,
        resamples=resamples,
        confidence_level=confidence_level,
        seed=seed + 9_999,
    )
    return {
        "candidate": candidate,
        "reference": reference,
        "score_delta_candidate_minus_reference": overall.point,
        "ci_lower": overall.lower,
        "ci_upper": overall.upper,
        "paired_prompt_count": overall.sample_count,
        "task_count": len(diffs_by_task),
        "tasks": task_rows,
    }


def pareto_membership(
    accuracy_rows: Sequence[Mapping[str, Any]],
    access_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    accuracy = {
        (str(row["setting"]), str(row["task"])): _finite_float(row.get("score"))
        for row in accuracy_rows
    }
    access = {
        (str(row["setting"]), str(row["task"])): _finite_float(
            row.get("decode_gqa_total_access_pct")
        )
        for row in access_rows
    }
    tasks = sorted({task for _, task in set(accuracy) | set(access)})
    output: list[dict[str, Any]] = []
    for task in tasks:
        points = [
            (setting, access_value, score)
            for (setting, point_task), score in accuracy.items()
            if point_task == task
            and score is not None
            and (access_value := access.get((setting, task))) is not None
        ]
        for setting, x_value, y_value in points:
            dominated_by = [
                other
                for other, other_x, other_y in points
                if other != setting
                and other_x <= x_value
                and other_y >= y_value
                and (other_x < x_value or other_y > y_value)
            ]
            output.append(
                {
                    "setting": setting,
                    "task": task,
                    "score": y_value,
                    "decode_gqa_total_access_pct": x_value,
                    "on_pareto_frontier": not dominated_by,
                    "dominated_by": ";".join(sorted(dominated_by)),
                }
            )
    return output


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
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


def _write_accuracy_matrix(
    path: Path,
    by_task: Sequence[Mapping[str, Any]],
    overall: Sequence[Mapping[str, Any]],
    settings: Sequence[str],
    tasks: Sequence[str],
) -> None:
    lookup = {
        (str(row["setting"]), str(row["task"])): row.get("score")
        for row in [*by_task, *overall]
    }
    rows = []
    for setting in settings:
        row: dict[str, Any] = {"setting": setting}
        for task in tasks:
            row[task] = lookup.get((setting, task))
        row[SUMMARY_TASK] = lookup.get((setting, SUMMARY_TASK))
        rows.append(row)
    _write_csv(path, rows)


def _plot(
    *,
    output_path: Path,
    title: str,
    accuracy_rows: Sequence[Mapping[str, Any]],
    access_rows: Sequence[Mapping[str, Any]],
    settings: Sequence[str],
    labels: Mapping[str, str],
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Plotting requires matplotlib. Install with `pip install -e .[analysis]`."
        ) from exc

    accuracy = {str(row["setting"]): row for row in accuracy_rows}
    access = {str(row["setting"]): row for row in access_rows}
    fig, axis = plt.subplots(figsize=(10.5, 6.4), constrained_layout=True)
    plotted = 0
    for setting in settings:
        score_row = accuracy.get(setting)
        access_row = access.get(setting)
        if score_row is None or access_row is None:
            continue
        x_value = _finite_float(access_row.get("decode_gqa_total_access_pct"))
        y_value = _finite_float(score_row.get("score"))
        lower = _finite_float(score_row.get("score_ci_lower"))
        upper = _finite_float(score_row.get("score_ci_upper"))
        if x_value is None or y_value is None:
            continue
        yerr = None
        if lower is not None and upper is not None:
            yerr = [[max(0.0, y_value - lower)], [max(0.0, upper - y_value)]]
        axis.errorbar(
            x_value,
            y_value,
            yerr=yerr,
            marker="o",
            linestyle="none",
            capsize=3,
            markersize=7,
            label=labels[setting],
        )
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        return
    axis.set_title(title)
    axis.set_xlabel("GQA-aware logical decode access (% of dense K/V rows)")
    axis.set_ylabel("RULER score")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best", fontsize=8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path.with_suffix(".png"), dpi=200)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def create_plots(
    output_dir: Path,
    *,
    by_task_accuracy: Sequence[Mapping[str, Any]],
    overall_accuracy: Sequence[Mapping[str, Any]],
    by_task_access: Sequence[Mapping[str, Any]],
    overall_access: Sequence[Mapping[str, Any]],
    settings: Sequence[str],
    tasks: Sequence[str],
    labels: Mapping[str, str],
) -> None:
    plots = output_dir / "plots"
    _plot(
        output_path=plots / "pareto_selected_task_mean",
        title="Selected-task mean accuracy versus logical access",
        accuracy_rows=overall_accuracy,
        access_rows=overall_access,
        settings=settings,
        labels=labels,
    )
    for task in tasks:
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", task).strip("-").lower()
        _plot(
            output_path=plots / f"pareto_{slug}",
            title=f"{task}: accuracy versus logical access",
            accuracy_rows=[row for row in by_task_accuracy if row["task"] == task],
            access_rows=[row for row in by_task_access if row["task"] == task],
            settings=settings,
            labels=labels,
        )


def _fmt(value: Any, digits: int = 2) -> str:
    number = _finite_float(value)
    return "—" if number is None else f"{number:.{digits}f}"


def render_markdown(
    *,
    exports: Sequence[ExportSource],
    settings: Sequence[str],
    labels: Mapping[str, str],
    tasks: Sequence[str],
    overall_accuracy: Sequence[Mapping[str, Any]],
    overall_access: Sequence[Mapping[str, Any]],
    comparisons: Sequence[Mapping[str, Any]],
) -> str:
    accuracy = {str(row["setting"]): row for row in overall_accuracy}
    access = {str(row["setting"]): row for row in overall_access}
    lines = [
        "# RULER result summary",
        "",
        f"Merged {len(exports)} compact export(s), {len(settings)} setting(s), and "
        f"{len(tasks)} task(s).",
        "",
        "Logical access is recomputed from pooled vector-row counts when available. "
        "It is an accounting estimate, not measured memory traffic.",
        "",
        "| Setting | Display label | Mean score | Confidence interval | GQA total access | Records |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for setting in settings:
        score = accuracy.get(setting, {})
        memory = access.get(setting, {})
        lines.append(
            "| {setting} | {label} | {score} | [{lower}, {upper}] | {access}% | {count} |".format(
                setting=setting,
                label=labels[setting],
                score=_fmt(score.get("score")),
                lower=_fmt(score.get("score_ci_lower")),
                upper=_fmt(score.get("score_ci_upper")),
                access=_fmt(memory.get("decode_gqa_total_access_pct"), 1),
                count=score.get("sample_count", "—"),
            )
        )

    if comparisons:
        lines.extend(
            [
                "",
                "## Paired score comparisons",
                "",
                "| Candidate | Reference | Mean score delta | Confidence interval | Paired prompts |",
                "|---|---|---:|---:|---:|",
            ]
        )
        for row in comparisons:
            lines.append(
                "| {candidate} | {reference} | {delta} | [{lower}, {upper}] | {count} |".format(
                    candidate=row["candidate"],
                    reference=row["reference"],
                    delta=_fmt(row.get("score_delta_candidate_minus_reference")),
                    lower=_fmt(row.get("ci_lower")),
                    upper=_fmt(row.get("ci_upper")),
                    count=row.get("paired_prompt_count", "—"),
                )
            )

    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `accuracy_by_task.csv` and `accuracy_overall.csv`: score estimates and bootstrap intervals.",
            "- `kv_access_by_setting_task.csv` and `kv_access_by_setting.csv`: pooled logical-access accounting.",
            "- `paired_comparisons.csv`: prompt-paired score deltas.",
            "- `pareto_frontier_membership.csv`: nondominance by score and GQA-aware total access.",
            "- `plots/`: one selected-task-mean plot and one plot per task, each as PNG and PDF.",
            "",
        ]
    )
    return "\n".join(lines)


def _prepare_output(path: Path, *, overwrite: bool) -> Path:
    destination = path.expanduser().resolve()
    if destination.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {destination}. Pass --overwrite."
            )
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    return destination


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.bootstrap_resamples <= 0:
        raise ValueError("--bootstrap-resamples must be positive.")
    if not 0.0 < args.confidence_level < 1.0:
        raise ValueError("--confidence-level must be strictly between zero and one.")

    with tempfile.TemporaryDirectory(prefix="santapp-ruler-analysis-") as temp_dir:
        exports = discover_exports(
            args.input,
            allow_partial=args.allow_partial,
            scratch=Path(temp_dir),
        )
        rows, metadata, tasks, settings = merge_exports(exports)

        by_task_accuracy, overall_accuracy = compute_accuracy(
            rows,
            settings=settings,
            tasks=tasks,
            resamples=args.bootstrap_resamples,
            confidence_level=args.confidence_level,
            seed=args.bootstrap_seed,
        )
        by_task_access, overall_access = compute_access_tables(
            rows,
            settings=settings,
            tasks=tasks,
        )
        pairs = _parse_comparisons(
            args.comparison, settings, args.reference_setting
        )
        comparisons = [
            _paired_comparison(
                rows,
                candidate=candidate,
                reference=reference,
                tasks=tasks,
                resamples=args.bootstrap_resamples,
                confidence_level=args.confidence_level,
                seed=args.bootstrap_seed + pair_index * 100_003,
            )
            for pair_index, (candidate, reference) in enumerate(pairs)
        ]
        comparison_rows = [
            {key: value for key, value in row.items() if key != "tasks"}
            for row in comparisons
        ]
        frontier = pareto_membership(
            [*by_task_accuracy, *overall_accuracy],
            [*by_task_access, *overall_access],
        )
        labels = {
            setting: setting_label(setting, metadata.get(setting, {}))
            for setting in settings
        }

        output_dir = _prepare_output(args.output_dir, overwrite=args.overwrite)
        _write_csv(output_dir / "accuracy_by_task.csv", by_task_accuracy)
        _write_csv(output_dir / "accuracy_overall.csv", overall_accuracy)
        _write_accuracy_matrix(
            output_dir / "accuracy_matrix.csv",
            by_task_accuracy,
            overall_accuracy,
            settings,
            tasks,
        )
        _write_csv(output_dir / "kv_access_by_setting_task.csv", by_task_access)
        _write_csv(output_dir / "kv_access_by_setting.csv", overall_access)
        _write_csv(output_dir / "paired_comparisons.csv", comparison_rows)
        _write_csv(output_dir / "pareto_frontier_membership.csv", frontier)
        create_plots(
            output_dir,
            by_task_accuracy=by_task_accuracy,
            overall_accuracy=overall_accuracy,
            by_task_access=by_task_access,
            overall_access=overall_access,
            settings=settings,
            tasks=tasks,
            labels=labels,
        )

        summary = {
            "schema_version": 1,
            "sources": [str(source.root) for source in exports],
            "tasks": tasks,
            "settings": [
                {
                    "name": setting,
                    "display_label": labels[setting],
                    "metadata": metadata.get(setting, {}),
                }
                for setting in settings
            ],
            "bootstrap": {
                "resamples": args.bootstrap_resamples,
                "confidence_level": args.confidence_level,
                "seed": args.bootstrap_seed,
                "overall_method": "stratified within task; tasks equally weighted",
            },
            "accuracy_by_task": by_task_accuracy,
            "accuracy_overall": overall_accuracy,
            "access_by_setting_task": by_task_access,
            "access_by_setting": overall_access,
            "paired_comparisons": comparisons,
            "pareto_frontier_membership": frontier,
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        (output_dir / "summary.md").write_text(
            render_markdown(
                exports=exports,
                settings=settings,
                labels=labels,
                tasks=tasks,
                overall_accuracy=overall_accuracy,
                overall_access=overall_access,
                comparisons=comparison_rows,
            ),
            encoding="utf-8",
        )

    print(f"Wrote analysis to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
