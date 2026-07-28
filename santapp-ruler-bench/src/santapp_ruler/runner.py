"""End-to-end benchmark orchestration."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .backends import ModelBundle, SantaPlusBackend, SdpaBackend
from .config import RunConfig
from .data import RulerExample, select_examples
from .reporting import build_reports, read_jsonl
from .run_state import prepare_run_directory, validate_or_write_selection_manifest
from .ruler.grader import grade_task
from .ruler.provenance import RULER_COMMIT, RULER_REPOSITORY
from .ruler.tasks import require_task


def _example_seed(base_seed: int, example: RulerExample) -> int:
    digest = hashlib.sha256(
        f"{base_seed}:{example.uid}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)


def _default_run_name(config: RunConfig) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model = config.model.name.rsplit("/", 1)[-1].replace(" ", "-")
    return f"{timestamp}-{model}-{config.benchmark.context_length}"


def resolve_run_dir(
    config: RunConfig, explicit_run_dir: str | Path | None
) -> Path:
    if explicit_run_dir is not None:
        return Path(explicit_run_dir).expanduser().resolve()
    root = Path(config.output.root).expanduser().resolve()
    name = config.output.run_name or _default_run_name(config)
    return root / name


def _runtime_info(config: RunConfig, bundle: ModelBundle | None = None) -> dict[str, Any]:
    info: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "numpy": np.__version__,
        "ruler_repository": RULER_REPOSITORY,
        "ruler_commit": RULER_COMMIT,
        "model": config.model.name,
        "model_revision_requested": config.model.revision,
        "data_source": config.benchmark.data.source,
        "dataset_repository": config.benchmark.data.repository,
        "dataset_revision": config.benchmark.data.revision,
    }
    try:
        import transformers

        info["transformers"] = transformers.__version__
    except ImportError:
        info["transformers"] = None
    try:
        import datasets

        info["datasets"] = datasets.__version__
    except ImportError:
        info["datasets"] = None
    if torch.cuda.is_available():
        info.update(
            {
                "gpu": torch.cuda.get_device_name(0),
                "gpu_total_gib": torch.cuda.get_device_properties(0).total_memory
                / (1024**3),
                "cuda_capability": list(torch.cuda.get_device_capability(0)),
            }
        )
    if bundle is not None:
        info["model_type"] = getattr(bundle.model.config, "model_type", None)
        info["model_dtype"] = str(next(bundle.model.parameters()).dtype)
        info["model_commit_hash"] = getattr(
            bundle.model.config, "_commit_hash", None
        )
    return info


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def _completed_uids(path: Path) -> set[str]:
    completed: set[str] = set()
    for record in read_jsonl(path):
        uid = record.get("uid")
        if uid is None:
            uid = f"{record.get('task')}:{record.get('index')}"
        completed.add(str(uid))
    return completed


def _append_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()


def _max_new_tokens(config: RunConfig, task: str) -> int:
    official = require_task(task).max_new_tokens
    cap = config.generation.max_new_tokens_cap
    return min(official, cap) if cap is not None else official


def _validate_token_length(
    *,
    example: RulerExample,
    prompt_tokens: int,
    max_new_tokens: int,
    context_length: int,
) -> None:
    if prompt_tokens > context_length:
        raise ValueError(
            f"{example.uid} has {prompt_tokens} prompt tokens, exceeding the "
            f"configured context length {context_length}."
        )
    if prompt_tokens + max_new_tokens > context_length:
        raise ValueError(
            f"{example.uid}: prompt_tokens ({prompt_tokens}) + the official "
            f"generation budget ({max_new_tokens}) exceeds context_length "
            f"({context_length}). This usually means the data was generated with "
            "a different tokenizer or RULER max sequence length."
        )


def run_benchmark(
    config: RunConfig,
    *,
    explicit_run_dir: str | Path | None = None,
) -> Path:
    config.validate()
    run_dir = prepare_run_directory(
        config, resolve_run_dir(config, explicit_run_dir)
    )
    _write_json(run_dir / "runtime.pre_model.json", _runtime_info(config))

    print(f"Run directory: {run_dir}")
    print(
        f"Selecting {config.benchmark.prompts_per_task} prompt(s) for each of "
        f"{len(config.benchmark.tasks)} task(s)..."
    )
    selected = select_examples(config.benchmark)
    validate_or_write_selection_manifest(
        selected,
        config.benchmark.tasks,
        run_dir / "selected_prompts.jsonl",
        include_prompts=config.output.save_full_prompts,
    )

    print(f"Loading {config.model.name} with stock SDPA enabled...")
    bundle = ModelBundle.load(config.model)
    _write_json(run_dir / "runtime.json", _runtime_info(config, bundle))
    print(
        f"GPU: {torch.cuda.get_device_name(0)} | model dtype: "
        f"{next(bundle.model.parameters()).dtype}"
    )

    backend_objects = {
        "sdpa": SdpaBackend(bundle),
        "santapp": SantaPlusBackend(bundle, config.santapp),
    }
    total_expected = (
        len(config.generation.backends)
        * len(config.benchmark.tasks)
        * config.benchmark.prompts_per_task
    )
    completed_this_process = 0
    run_start = time.perf_counter()

    for backend_name in config.generation.backends:
        backend = backend_objects[backend_name]
        print(f"\n=== Backend: {backend_name} ===")
        for task in config.benchmark.tasks:
            prediction_path = run_dir / "predictions" / backend_name / f"{task}.jsonl"
            completed = (
                _completed_uids(prediction_path) if config.output.resume else set()
            )
            examples = selected[task]
            task_budget = _max_new_tokens(config, task)
            print(
                f"[{backend_name}] {task}: {len(examples)} prompt(s), "
                f"max_new_tokens={task_budget}"
            )
            for ordinal, example in enumerate(examples, start=1):
                if example.uid in completed:
                    print(f"  {ordinal}/{len(examples)} {example.uid}: resume skip")
                    continue

                input_ids = bundle.tokenize(example.input)
                prompt_tokens = int(input_ids.shape[1])
                _validate_token_length(
                    example=example,
                    prompt_tokens=prompt_tokens,
                    max_new_tokens=task_budget,
                    context_length=config.benchmark.context_length,
                )
                seed = _example_seed(config.santapp.sample_seed, example)
                result = backend.generate(
                    input_ids,
                    max_new_tokens=task_budget,
                    stop_on_eos=config.generation.stop_on_eos,
                    random_seed=seed,
                )
                example_grade = grade_task(
                    task, [result.prediction], [example.outputs]
                ).score
                record = {
                    "index": example.index,
                    "uid": example.uid,
                    "task": task,
                    "input": example.input,
                    "outputs": example.outputs,
                    "pred": result.prediction,
                    "others": {
                        "id": example.index,
                        "task": task,
                        "backend": backend_name,
                        "source_position": example.source_position,
                    },
                    "metrics": {
                        **result.metrics,
                        "example_ruler_score": example_grade,
                        "random_seed": seed,
                        "reported_dataset_length": example.reported_length,
                    },
                }
                _append_record(prediction_path, record)
                completed.add(example.uid)
                completed_this_process += 1

                access = result.metrics.get("decode_kv_access_pct")
                equivalent = result.metrics.get("decode_read_equivalent_pct")
                access_label = (
                    f"KV {access:.1f}% / equiv {equivalent:.1f}%"
                    if access is not None and equivalent is not None
                    else "KV n/a"
                )
                preview = result.prediction.replace("\n", " ")[:80]
                print(
                    f"  {ordinal}/{len(examples)} {example.uid} | "
                    f"score {example_grade:5.1f} | "
                    f"{result.metrics['total_seconds']:7.2f}s | {access_label} | "
                    f"{preview!r}"
                )
                del input_ids, result
                bundle.release_example_memory()

    elapsed = time.perf_counter() - run_start
    _write_json(
        run_dir / "run_status.json",
        {
            "status": "complete",
            "expected_records": total_expected,
            "records_generated_this_process": completed_this_process,
            "wall_seconds_this_process": elapsed,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    print("\nGrading with the vendored RULER synthetic grader...")
    build_reports(
        run_dir,
        backends=config.generation.backends,
        tasks=config.benchmark.tasks,
    )
    print(f"Completed. Summary: {run_dir / 'summary.md'}")
    return run_dir
