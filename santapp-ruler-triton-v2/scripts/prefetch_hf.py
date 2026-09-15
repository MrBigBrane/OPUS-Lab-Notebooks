#!/usr/bin/env python3
"""Prefetch configured model/data once, without a GPU, before parallel RULER Jobs."""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/smoke.yaml"))
    parser.add_argument("--contexts", type=int, nargs="+", default=[8192, 32768])
    parser.add_argument("--tasks", nargs="+", default=["niah_single_1"])
    args = parser.parse_args()
    from santapp_ruler.config import load_config
    from santapp_ruler.ruler.tasks import validate_tasks
    from huggingface_hub import snapshot_download
    from datasets import load_dataset
    config = load_config(args.config)
    validate_tasks(args.tasks)
    if config.benchmark.data.source != "huggingface":
        raise ValueError("Prefetch requires the existing Hugging Face data source")
    print(f"Prefetching {config.model.name} revision={config.model.revision}", flush=True)
    snapshot_download(repo_id=config.model.name, revision=config.model.revision)
    for context in args.contexts:
        config.benchmark.context_length = context
        repo = config.resolved_dataset_repository()
        for task in args.tasks:
            print(f"Prefetching {repo} / {task}", flush=True)
            ds = load_dataset(repo, split=task, revision=config.benchmark.data.revision)
            print(f"{repo} / {task}: {len(ds)} examples", flush=True)
    print("HF PREFETCH PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
