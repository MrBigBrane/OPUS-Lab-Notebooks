#!/usr/bin/env python3
"""Run one context/backend shard of the indexed Nautilus smoke matrix."""

from __future__ import annotations

import argparse
import contextlib
import sys
import traceback
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
    decode = os.environ.get("SMOKE_DECODE_BACKEND", "torch")
    profile = os.environ.get("SMOKE_RUN_PROFILE", "")
    if decode not in {"torch", "grouped_triton"}:
        raise ValueError("SMOKE_DECODE_BACKEND must be torch or grouped_triton")
    if decode == "grouped_triton" and os.environ.get("SANTAPP_REQUIRE_DECODE_GATE"):
        from santapp_ruler.attention.triton_decode.identity import source_digest
        gate = json.loads(Path(os.environ["SANTAPP_REQUIRE_DECODE_GATE"]).read_text())
        if gate.get("status") != "passed" or gate.get("mode") != "gpu" or gate.get("source_digest") != source_digest():
            raise RuntimeError("The grouped CUDA correctness gate has not passed for this source/image")
        if not set(contexts) <= set(gate.get("contexts", [])):
            raise RuntimeError("Requested contexts are not covered by the grouped CUDA gate")
        if not {s for s in budgets if s is not None} <= set(gate.get("nominal_sample_budgets", [])):
            raise RuntimeError("Requested sample budgets are not covered by the grouped CUDA gate")
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
    if decode != "torch" or profile:
        matrix_root = matrix_root / f"decode-{decode}"
    if profile:
        if profile not in {"canary", "matrix", "reference"}:
            raise ValueError("Unknown SMOKE_RUN_PROFILE")
        matrix_root = matrix_root / profile
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
        overrides.append(f"{name}.decode_backend={decode}")
    if budget is not None:
        for name in ("santa", "santapp", "hierarchical"):
            overrides.append(f"{name}.samples_per_head={budget}")
    if os.environ.get("SMOKE_MAX_NEW_TOKENS"):
        overrides.append(f"generation.max_new_tokens={int(os.environ['SMOKE_MAX_NEW_TOKENS'])}")
    if os.environ.get("SMOKE_STOP_ON_EOS"):
        stop = os.environ["SMOKE_STOP_ON_EOS"].lower()
        if stop not in {"true", "false"}:
            raise ValueError("SMOKE_STOP_ON_EOS must be true or false")
        overrides.append(f"generation.stop_on_eos={stop}")
    if os.environ.get("SMOKE_KMEANS_FIT_BATCH_SIZE"):
        overrides.append("santapp.kmeans.fit_batch_size=" + os.environ["SMOKE_KMEANS_FIT_BATCH_SIZE"])
    config = load_config(args.config, overrides=overrides)
    if os.environ.get("SANTAPP_REQUIRE_GPU_PRODUCT"):
        import torch
        required = os.environ["SANTAPP_REQUIRE_GPU_PRODUCT"]
        actual = torch.cuda.get_device_name(0)
        normalize = lambda name: name.lower().replace("-", "").replace(" ", "")
        if normalize(required) != normalize(actual):
            raise RuntimeError(f"This smoke requires {required}, got {actual}")
    if backend == "santapp" and prefill == "triton" and os.environ.get("SANTAPP_REQUIRE_KMEANS_GATE"):
        from santapp_ruler.attention.triton_decode.identity import source_digest
        gate = json.loads(Path(os.environ["SANTAPP_REQUIRE_KMEANS_GATE"]).read_text())
        if (gate.get("status") != "passed" or gate.get("mode") != "gpu"
                or gate.get("source_digest") != source_digest()
                or context not in gate.get("contexts", [])
                or config.santapp.kmeans.fit_batch_size > gate.get("fit_batch_size", 0)):
            raise RuntimeError("The batched K-means GPU gate has not passed for this source/context/fit batch size")
    metadata = {
        "index": raw_index,
        "completion_count": completion_count,
        "context_length": context,
        "backend": backend,
        "prefill_backend": prefill if backend in {"santapp", "hierarchical"} else "not_used",
        "kmeans_fit_batch_size": config.santapp.kmeans.fit_batch_size if backend == "santapp" else None,
        "decode_backend": decode if backend in {"santapp", "hierarchical"} else "not_used",
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
    telemetry_enabled = os.environ.get("SANTAPP_GPU_TELEMETRY") == "1"
    from santapp_ruler.gpu_telemetry import GpuTelemetry, summarize_decode_telemetry, summarize_phase_telemetry
    class Tee:
        def __init__(self, stream, log):
            self.stream, self.log = stream, log
        def write(self, text):
            self.stream.write(text)
            self.log.write(text)
            self.log.flush()
            return len(text)
        def flush(self):
            self.stream.flush()
            self.log.flush()
        def isatty(self):
            return False
    try:
        with (run_dir / "console.log").open("a") as log:
            with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
                try:
                    monitor = GpuTelemetry(run_dir / "gpu-utilization.csv") if telemetry_enabled else contextlib.nullcontext()
                    with monitor:
                        run_benchmark(config, explicit_run_dir=run_dir)
                except Exception:
                    traceback.print_exc()
                    raise
        metadata["status"] = "complete"
    except Exception as exc:
        metadata.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if telemetry_enabled:
            summarize_decode_telemetry(run_dir)
            summarize_phase_telemetry(run_dir)
        metadata["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_write_json(run_dir / "shard.json", metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
