#!/usr/bin/env python3
"""Populate the shared pinned model cache and a read-only local RULER mirror."""

from __future__ import annotations

import argparse
import json
import os
import socket
import traceback
from dataclasses import asdict
from pathlib import Path

from santapp_ruler.config import BenchmarkConfig
from santapp_ruler.data import load_task_examples, select_examples
from santapp_ruler.io_utils import atomic_write_json, atomic_write_text
from santapp_ruler.k8s_matrix import cache_fingerprint, load_matrix
from santapp_ruler.k8s_runtime import (
    KubernetesPaths,
    compact_utc_now,
    configure_huggingface_environment,
    exclusive_lock,
    require_cache_marker,
    utc_now,
)
from santapp_ruler.ruler.tasks import require_task


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
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    return parser


def _dataset_payload(examples) -> str:
    lines: list[str] = []
    for example in examples:
        row = {
            "index": example.index,
            "input": example.input,
            "outputs": example.outputs,
        }
        if example.reported_length is not None:
            row["length"] = example.reported_length
        lines.append(json.dumps(row, ensure_ascii=False, allow_nan=False))
    return "\n".join(lines) + ("\n" if lines else "")


def _existing_marker_is_ready(paths, matrix, base) -> bool:
    try:
        require_cache_marker(paths, matrix, base)
    except Exception:
        return False
    return True


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    matrix = load_matrix(args.matrix)
    base = matrix.load_base_run_config()
    paths = KubernetesPaths.create(args.results_root, matrix)
    expected_fingerprint = cache_fingerprint(matrix, base)

    if args.check_only:
        ready = _existing_marker_is_ready(paths, matrix, base)
        print("ready" if ready else "not ready")
        return 0 if ready else 2

    with exclusive_lock(paths.locks_root / "prefetch.lock"):
        if not args.force and _existing_marker_is_ready(paths, matrix, base):
            print(f"Shared cache already ready: {paths.cache_marker}")
            return 0

        configure_huggingface_environment(paths, offline=False)
        build_id = compact_utc_now()
        build_status = paths.cache_root / f"cache-build-{build_id}.json"
        common = {
            "schema_version": 1,
            "status": "building",
            "started_at_utc": utc_now(),
            "host": socket.gethostname(),
            "pod_name": os.environ.get("POD_NAME"),
            "node_name": os.environ.get("NODE_NAME"),
            "matrix": str(matrix.source_path),
            "matrix_hash": matrix.matrix_hash,
            "cache_fingerprint": expected_fingerprint,
        }
        atomic_write_json(build_status, common)

        try:
            from huggingface_hub import snapshot_download
            from transformers import AutoTokenizer

            token = os.environ.get("HF_TOKEN") or os.environ.get(
                "HUGGING_FACE_HUB_TOKEN"
            )
            print(
                f"Downloading pinned model snapshot {base.model.name}@"
                f"{base.model.revision} into {paths.hub_cache}"
            )
            snapshot_path = snapshot_download(
                repo_id=base.model.name,
                revision=base.model.revision,
                cache_dir=paths.hub_cache,
                token=token or None,
            )
            tokenizer = AutoTokenizer.from_pretrained(
                base.model.name,
                revision=base.model.revision,
                trust_remote_code=base.model.trust_remote_code,
                use_fast=True,
                cache_dir=paths.hub_cache,
                token=token or None,
            )

            dataset_counts: dict[str, int] = {}
            for task in matrix.tasks:
                print(
                    f"Downloading pinned RULER task {task} from "
                    f"{base.benchmark.data.repository}@"
                    f"{base.benchmark.data.revision}"
                )
                source_config = BenchmarkConfig(
                    context_length=base.benchmark.context_length,
                    tasks=[task],
                    prompts_per_task=base.benchmark.prompts_per_task,
                    selection_seed=base.benchmark.selection_seed,
                    data=base.benchmark.data,
                )
                examples = load_task_examples(source_config, task)
                destination = paths.local_data_root / task / "validation.jsonl"
                atomic_write_text(destination, _dataset_payload(examples))
                dataset_counts[task] = len(examples)

            # Validate that the mirror reproduces deterministic prompt selection
            # and token budgets before any GPU is requested.
            local_config = BenchmarkConfig(
                context_length=base.benchmark.context_length,
                tasks=list(matrix.tasks),
                prompts_per_task=matrix.prompts_per_task,
                selection_seed=base.benchmark.selection_seed,
                data=type(base.benchmark.data)(
                    source="local",
                    repository=base.benchmark.data.repository,
                    revision=base.benchmark.data.revision,
                    local_root=str(paths.local_data_root),
                    subset="validation",
                ),
            )
            selected = select_examples(local_config)
            selected_uids: dict[str, list[str]] = {}
            token_ranges: dict[str, dict[str, int]] = {}
            for task, examples in selected.items():
                budget = (
                    base.generation.max_new_tokens
                    if base.generation.max_new_tokens is not None
                    else require_task(task).max_new_tokens
                )
                lengths: list[int] = []
                for example in examples:
                    encoded = tokenizer(
                        example.input,
                        add_special_tokens=False,
                        return_length=True,
                    )
                    length = int(encoded["length"][0])
                    if length + budget > base.benchmark.context_length:
                        raise ValueError(
                            f"{example.uid}: {length}+{budget} exceeds "
                            f"context_length={base.benchmark.context_length}"
                        )
                    lengths.append(length)
                selected_uids[task] = [example.uid for example in examples]
                token_ranges[task] = {
                    "minimum_prompt_tokens": min(lengths),
                    "maximum_prompt_tokens": max(lengths),
                    "max_new_tokens": budget,
                }

            marker = {
                **common,
                "status": "ready",
                "finished_at_utc": utc_now(),
                "model": asdict(base.model),
                "model_snapshot_path": str(snapshot_path),
                "dataset": asdict(base.benchmark.data),
                "dataset_rows": dataset_counts,
                "selected_uids": selected_uids,
                "token_ranges": token_ranges,
                "local_data_root": str(paths.local_data_root),
                "hf_home": str(paths.hf_home),
            }
            atomic_write_json(paths.cache_marker, marker)
            atomic_write_json(build_status, marker)
            print(f"Shared cache is ready: {paths.cache_marker}")
            return 0
        except Exception as exc:
            failure = {
                **common,
                "status": "failed",
                "finished_at_utc": utc_now(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            atomic_write_json(build_status, failure)
            atomic_write_json(paths.cache_root / "cache-latest-failure.json", failure)
            raise


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
