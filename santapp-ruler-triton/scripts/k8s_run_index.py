#!/usr/bin/env python3
"""Run one context/backend shard of the indexed Nautilus smoke matrix."""

from __future__ import annotations

import argparse
import json
import os
import socket
from datetime import datetime, timezone
from pathlib import Path

from santapp_ruler.runtime_defaults import configure_allocator

configure_allocator()  # Before runner imports torch.

from santapp_ruler.config import SUPPORTED_BACKENDS, load_config
from santapp_ruler.io_utils import atomic_write_json
from santapp_ruler.runner import run_benchmark
from santapp_ruler.ruler.tasks import DEFAULT_TASKS


def _csv(name: str, default: str) -> list[str]:
    return [item.strip() for item in os.environ.get(name, default).split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/smoke.yaml"))
    parser.add_argument("--index", type=int, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    contexts = [int(value) for value in _csv("SMOKE_CONTEXTS", "8192,32768")]
    backends = _csv("SMOKE_BACKENDS", ",".join(SUPPORTED_BACKENDS))
    tasks = _csv("SMOKE_TASKS", ",".join(DEFAULT_TASKS))
    unknown = set(backends) - set(SUPPORTED_BACKENDS)
    if unknown:
        raise ValueError(f"Unknown backends in SMOKE_BACKENDS: {sorted(unknown)}")

    raw_index = args.index
    if raw_index is None:
        raw_index = int(os.environ.get("JOB_COMPLETION_INDEX", "0"))
    budgets = [int(value) for value in _csv("SMOKE_SAMPLE_BUDGETS", "")] or [None]
    prefill = os.environ.get("SMOKE_PREFILL_BACKEND", "triton")
    if prefill not in {"torch", "triton"}:
        raise ValueError("SMOKE_PREFILL_BACKEND must be torch or triton")
    if any(s is not None and (s <= 0 or s % 4) for s in budgets):
        raise ValueError("P=16/R=4 nominal budgets must be positive multiples of four")
    completion_count = len(contexts) * len(budgets) * len(backends)
    if not 0 <= raw_index < completion_count:
        raise IndexError(f"index {raw_index} is outside [0, {completion_count}).")

    context_index, remainder = divmod(raw_index, len(budgets) * len(backends))
    budget_index, backend_index = divmod(remainder, len(backends))
    budget = budgets[budget_index]
    context = contexts[context_index]
    backend = backends[backend_index]
    owner = os.environ.get("OWNER_SLUG", "researcher")
    results_root = Path(os.environ.get("RESULTS_ROOT", "/shared/results"))
    # Keep original paths for unchanged Torch/no-sweep invocations.
    matrix_root = results_root / owner / "smoke"
    if prefill != "torch" or budget is not None:
        budget_label = str(budget) if budget is not None else "config"
        matrix_root = matrix_root / f"prefill-{prefill}" / f"S{budget_label}"
    run_dir = matrix_root / str(context) / backend
    run_dir.mkdir(parents=True, exist_ok=True)

    overrides = [
        f"benchmark.context_length={context}",
        f"benchmark.tasks={json.dumps(tasks)}",
        f"benchmark.prompts_per_task={int(os.environ.get('SMOKE_PROMPTS_PER_TASK', '1'))}",
        f"generation.backends=[{backend}]",
        f"output.root={json.dumps(str(run_dir.parent))}",
        f"output.run_name={json.dumps(run_dir.name)}",
        "output.resume=true",
        "output.save_full_prompts=false",
        "output.save_prediction_inputs=false",
    ]
    for name in ("santapp", "hierarchical"):
        overrides.append(f"{name}.prefill_backend={prefill}")
    if budget is not None:
        for name in ("santa", "santapp", "hierarchical"):
            overrides.append(f"{name}.samples_per_head={budget}")
    if os.environ.get("SMOKE_MAX_NEW_TOKENS"):
        overrides.append(f"generation.max_new_tokens={int(os.environ['SMOKE_MAX_NEW_TOKENS'])}")
    config = load_config(args.config, overrides=overrides)
    metadata = {
        "index": raw_index,
        "completion_count": completion_count,
        "context_length": context,
        "backend": backend,
        "prefill_backend": prefill if backend in {"santapp", "hierarchical"} else "not_used",
        "sample_budget": getattr(config, backend).samples_per_head if backend != "sdpa" else None,
        "requested_sample_budget": budget,
        "prompts_per_task": config.benchmark.prompts_per_task,
        "tasks": tasks,
        "owner_slug": owner,
        "allocator_environment": configure_allocator(),
        "hostname": socket.gethostname(),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(run_dir / "shard.json", metadata)
    print(json.dumps(metadata, indent=2))
    run_benchmark(config, explicit_run_dir=run_dir)
    metadata["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    metadata["status"] = "complete"
    atomic_write_json(run_dir / "shard.json", metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
