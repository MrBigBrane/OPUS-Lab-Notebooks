"""Call NVIDIA/RULER's pinned classic generator for selected local JSONL.

Run ``scripts/bootstrap_ruler.py`` first. Some task families require the
upstream source-data download steps documented by NVIDIA/RULER.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from santapp_ruler.ruler.provenance import (
    DEFAULT_MODEL_REPOSITORY,
    DEFAULT_MODEL_REVISION,
)

DEFAULT_TASKS = (
    "niah_single_1",
    "niah_multikey_1",
    "niah_multiquery",
    "vt",
    "fwe",
    "qa_1",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ruler-root", type=Path, default=Path("third_party/RULER"))
    parser.add_argument("--output", type=Path, default=Path("data/ruler_8k"))
    parser.add_argument("--model", default=DEFAULT_MODEL_REPOSITORY)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        help="Optional local tokenizer snapshot; otherwise download the pinned revision.",
    )
    parser.add_argument("--context-length", type=int, default=8192)
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    parser.add_argument("--subset", default="validation")
    args = parser.parse_args()

    ruler_root = args.ruler_root.resolve()
    prepare = ruler_root / "scripts" / "data" / "prepare.py"
    if not prepare.is_file():
        raise FileNotFoundError(
            f"Missing {prepare}. Run scripts/bootstrap_ruler.py first."
        )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    tasks = [value.strip() for value in args.tasks.split(",") if value.strip()]

    if args.tokenizer_path is not None:
        tokenizer_path = args.tokenizer_path.expanduser().resolve()
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"Tokenizer path does not exist: {tokenizer_path}")
    else:
        from huggingface_hub import snapshot_download

        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        tokenizer_path = Path(
            snapshot_download(
                repo_id=args.model,
                revision=args.model_revision,
                allow_patterns=["*.json", "*.txt", "*.model"],
            )
        )
    print(f"Tokenizer snapshot: {tokenizer_path}")

    for task in tasks:
        command = [
            sys.executable,
            str(prepare),
            "--save_dir",
            str(output),
            "--benchmark",
            "synthetic",
            "--task",
            task,
            "--subset",
            args.subset,
            "--tokenizer_path",
            str(tokenizer_path),
            "--tokenizer_type",
            "hf",
            "--max_seq_length",
            str(args.context_length),
            "--num_samples",
            str(args.num_samples),
            "--random_seed",
            str(args.seed),
            "--model_template_type",
            "base",
        ]
        print("+", " ".join(command))
        try:
            subprocess.run(command, cwd=prepare.parent, check=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"Upstream preparation failed for {task}. Check NVIDIA/RULER's "
                "source-data setup for this task family."
            ) from exc
    print(f"Wrote local RULER data under {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
