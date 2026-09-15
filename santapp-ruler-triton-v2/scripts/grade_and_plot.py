#!/usr/bin/env python3
"""Re-grade copied smoke shards and create accuracy/access Pareto curves."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from santapp_ruler.config import load_config
from santapp_ruler.reporting import build_reports


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _pareto(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        score = _number(row.get("selected_task_mean_score"))
        access = _number(row.get("decode_gqa_total_access_pct"))
        keep = score is not None and access is not None
        if keep:
            for other in rows:
                if other is row:
                    continue
                other_score = _number(other.get("selected_task_mean_score"))
                other_access = _number(other.get("decode_gqa_total_access_pct"))
                if other_score is None or other_access is None:
                    continue
                if (
                    other_score >= score
                    and other_access <= access
                    and (other_score > score or other_access < access)
                ):
                    keep = False
                    break
        row["pareto_accuracy_vs_gqa_access"] = keep


def _plot(rows: list[dict[str, Any]], context: int, output: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Install plotting support with `pip install -e .[plot]`.") from exc

    usable = [
        row
        for row in rows
        if _number(row.get("selected_task_mean_score")) is not None
        and _number(row.get("decode_gqa_total_access_pct")) is not None
    ]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for row in usable:
        x = float(row["decode_gqa_total_access_pct"])
        y = float(row["selected_task_mean_score"])
        ax.scatter([x], [y], s=55)
        label = str(row["backend"])
        if row.get("sample_budget") is not None:
            label += f" S={row['sample_budget']}"
        if row.get("prefill_backend") not in (None, "not_used"):
            label += f" ({row['prefill_backend']})"
        ax.annotate(label, (x, y), xytext=(5, 5), textcoords="offset points")
    frontier = sorted(
        [row for row in usable if str(row.get("pareto_accuracy_vs_gqa_access")).lower() == "true"],
        key=lambda row: float(row["decode_gqa_total_access_pct"]),
    )
    if frontier:
        ax.plot(
            [float(row["decode_gqa_total_access_pct"]) for row in frontier],
            [float(row["selected_task_mean_score"]) for row in frontier],
            marker="o",
        )
    ax.set_xlabel("GQA-aware logical KV + routing access (% of dense KV)")
    ax.set_ylabel("Mean selected-task RULER score")
    ax.set_title(f"RULER Pareto frontier — context {context}")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / f"pareto_{context}.png", dpi=180)
    fig.savefig(output / f"pareto_{context}.pdf")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_root", type=Path, help="Copied PVC results directory.")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    root = args.results_root.expanduser().resolve()
    output = (args.output or (root / "local_analysis")).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    shard_files = sorted(root.rglob("shard.json"))
    if not shard_files:
        raise FileNotFoundError(f"No shard.json files found under {root}")

    combined_backends: list[dict[str, Any]] = []
    combined_tasks: list[dict[str, Any]] = []
    for shard_file in shard_files:
        run_dir = shard_file.parent
        shard = json.loads(shard_file.read_text(encoding="utf-8"))
        if shard.get("status") != "complete":
            message = f"Incomplete shard: {shard_file}"
            if args.allow_incomplete:
                print(f"WARNING: {message}; skipped")
                continue
            raise RuntimeError(message)
        config = load_config(run_dir / "config.resolved.yaml")
        build_reports(
            run_dir,
            backends=config.generation.backends,
            tasks=config.benchmark.tasks,
            bootstrap_resamples=config.grading.bootstrap_resamples,
            confidence_level=config.grading.confidence_level,
            bootstrap_seed=config.grading.bootstrap_seed,
        )
        backend_rows = _read_csv(run_dir / "backend_summary.csv")
        task_rows = _read_csv(run_dir / "task_summary.csv")
        context = int(shard["context_length"])
        for row in backend_rows:
            combined_backends.append(
                {"context_length": context, "shard_path": str(run_dir),
                 "sample_budget": shard.get("sample_budget"),
                 "prefill_backend": shard.get("prefill_backend"), **row}
            )
        for row in task_rows:
            combined_tasks.append(
                {"context_length": context, "shard_path": str(run_dir),
                 "sample_budget": shard.get("sample_budget"),
                 "prefill_backend": shard.get("prefill_backend"), **row}
            )

    contexts = sorted({int(row["context_length"]) for row in combined_backends})
    for context in contexts:
        rows = [row for row in combined_backends if int(row["context_length"]) == context]
        _pareto(rows)
        _plot(rows, context, output)

    combined_backends.sort(key=lambda row: (int(row["context_length"]), str(row["backend"])))
    combined_tasks.sort(
        key=lambda row: (int(row["context_length"]), str(row["backend"]), str(row["task"]))
    )
    _write_csv(output / "combined_backend_summary.csv", combined_backends)
    _write_csv(output / "combined_task_summary.csv", combined_tasks)
    print(f"Wrote regraded summaries and Pareto plots to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
