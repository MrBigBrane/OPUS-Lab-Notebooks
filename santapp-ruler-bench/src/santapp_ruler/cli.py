"""Command-line interface for the benchmark harness."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _comma_list(value: str) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected a non-empty comma-separated list.")
    return values


def _add_config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "Optional YAML config. Without it, use the built-in defaults "
            "equivalent to configs/default_8k.yaml."
        ),
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="PATH=VALUE",
        help="Repeatable dotted config override, parsed as YAML.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="santapp-ruler",
        description="Benchmark SANTA++ against stock SDPA on RULER prompts.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Run selected RULER tasks.")
    _add_config_argument(run)
    run.add_argument("--tasks", type=_comma_list)
    run.add_argument("--prompts-per-task", type=int)
    run.add_argument("--context-length", type=int)
    run.add_argument("--backends", type=_comma_list)
    run.add_argument("--run-dir", type=Path, help="Exact output directory.")
    run.add_argument(
        "--no-resume", action="store_true", help="Do not skip existing UIDs."
    )

    validate = subparsers.add_parser(
        "validate-data",
        help="Download/select prompts and validate token budgets without loading the LM.",
    )
    _add_config_argument(validate)
    validate.add_argument("--tasks", type=_comma_list)
    validate.add_argument("--prompts-per-task", type=int)

    grade = subparsers.add_parser(
        "grade", help="Re-run the vendored RULER grader on an existing run."
    )
    grade.add_argument("--run-dir", type=Path, required=True)
    grade.add_argument(
        "--config",
        type=Path,
        help="Config to use; defaults to RUN_DIR/config.resolved.yaml.",
    )

    fidelity = subparsers.add_parser(
        "fidelity",
        help="Compare the custom dense cache against stock cached SDPA.",
    )
    _add_config_argument(fidelity)
    fidelity.add_argument("--tokens", type=int, default=16)
    fidelity.add_argument(
        "--prompt",
        default=(
            "The SANTA++ dense-cache fidelity check should reproduce stock "
            "greedy generation exactly. Continue this sentence:"
        ),
    )

    subparsers.add_parser("list-tasks", help="List supported RULER task names.")
    subparsers.add_parser("doctor", help="Check Python, package, CUDA, and GPU setup.")
    return parser


def _config_with_cli(args: argparse.Namespace):
    from .config import load_config

    overrides = list(args.overrides)
    if getattr(args, "tasks", None) is not None:
        overrides.append(f"benchmark.tasks={json.dumps(args.tasks)}")
    if getattr(args, "prompts_per_task", None) is not None:
        overrides.append(
            f"benchmark.prompts_per_task={args.prompts_per_task}"
        )
    if getattr(args, "context_length", None) is not None:
        overrides.append(f"benchmark.context_length={args.context_length}")
    if getattr(args, "backends", None) is not None:
        overrides.append(f"generation.backends={json.dumps(args.backends)}")
    if getattr(args, "no_resume", False):
        overrides.append("output.resume=false")
    return load_config(args.config, overrides=overrides)


def command_run(args: argparse.Namespace) -> int:
    from .runner import run_benchmark

    config = _config_with_cli(args)
    run_benchmark(config, explicit_run_dir=args.run_dir)
    return 0


def command_validate_data(args: argparse.Namespace) -> int:
    from transformers import AutoTokenizer

    from .data import select_examples
    from .ruler.tasks import require_task

    config = _config_with_cli(args)
    if config.model.disable_hf_xet:
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    selected = select_examples(config.benchmark)
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.name,
        revision=config.model.revision,
        trust_remote_code=config.model.trust_remote_code,
        use_fast=True,
    )
    print(
        f"Validated selection count: {len(config.benchmark.tasks)} task(s) x "
        f"{config.benchmark.prompts_per_task} prompt(s)."
    )
    for task in config.benchmark.tasks:
        budget = require_task(task).max_new_tokens
        if config.generation.max_new_tokens_cap is not None:
            budget = min(budget, config.generation.max_new_tokens_cap)
        lengths = []
        for example in selected[task]:
            tokens = tokenizer(
                example.input, add_special_tokens=False, return_length=True
            )
            length = int(tokens["length"][0])
            if length + budget > config.benchmark.context_length:
                raise ValueError(
                    f"{example.uid}: {length} + {budget} > "
                    f"{config.benchmark.context_length}"
                )
            lengths.append(length)
        print(
            f"  {task:20s} prompts={len(lengths):3d} "
            f"prompt_tokens=[{min(lengths)}, {max(lengths)}] "
            f"generation_budget={budget}"
        )
    return 0


def command_grade(args: argparse.Namespace) -> int:
    from .config import load_config
    from .reporting import build_reports

    run_dir = args.run_dir.expanduser().resolve()
    config_path = args.config or (run_dir / "config.resolved.yaml")
    config = load_config(config_path)
    build_reports(
        run_dir,
        backends=config.generation.backends,
        tasks=config.benchmark.tasks,
    )
    print(f"Wrote {run_dir / 'summary.md'}")
    return 0


def command_fidelity(args: argparse.Namespace) -> int:
    from .attention.santapp import SantaPlusEngine
    from .backends import ModelBundle, SdpaBackend

    config = _config_with_cli(args)
    bundle = ModelBundle.load(config.model)
    input_ids = bundle.tokenize(args.prompt)
    stock = SdpaBackend(bundle).generate(
        input_ids,
        max_new_tokens=args.tokens,
        stop_on_eos=False,
        random_seed=0,
    )
    custom_ids = SantaPlusEngine(bundle.model, config.santapp).generate_dense_reference(
        input_ids, max_new_tokens=args.tokens
    )
    print("stock token IDs :", stock.token_ids)
    print("custom token IDs:", custom_ids)
    print("stock text       :", repr(bundle.decode(stock.token_ids)))
    print("custom text      :", repr(bundle.decode(custom_ids)))
    if stock.token_ids == custom_ids:
        print(f"PASS: all {args.tokens} generated token IDs match.")
        return 0
    for index, (left, right) in enumerate(
        zip(stock.token_ids, custom_ids, strict=False)
    ):
        if left != right:
            print(f"FAIL: first mismatch at generated token {index}: {left} != {right}")
            break
    return 1


def command_list_tasks() -> int:
    from .ruler.tasks import DEFAULT_TASKS, TASKS

    defaults = set(DEFAULT_TASKS)
    print("Task                  Family                    Max new  Default  Description")
    print("-" * 100)
    for task in TASKS.values():
        print(
            f"{task.name:21s} {task.family:25s} {task.max_new_tokens:7d}  "
            f"{'yes' if task.name in defaults else '':7s}  {task.description}"
        )
    return 0


def command_doctor() -> int:
    import numpy as np
    import torch

    from . import __version__

    print(f"santapp-ruler: {__version__}")
    print(f"Python:          {sys.version.split()[0]}")
    print(f"PyTorch:         {torch.__version__}")
    print(f"PyTorch CUDA:    {torch.version.cuda}")
    print(f"NumPy:           {np.__version__}")
    try:
        import transformers

        print(f"Transformers:    {transformers.__version__}")
        from transformers.integrations.sdpa_attention import sdpa_attention_forward

        if not callable(sdpa_attention_forward):
            raise TypeError("HF SDPA wrapper is not callable")
        print("HF SDPA wrapper: available")
    except Exception as exc:
        print(f"Transformers/SDPA: FAIL ({type(exc).__name__}: {exc})")
    try:
        import datasets

        print(f"Datasets:        {datasets.__version__}")
    except Exception as exc:
        print(f"Datasets:        FAIL ({type(exc).__name__}: {exc})")
    print(f"CUDA usable:     {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU:             {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"GPU memory:      {props.total_memory / (1024**3):.1f} GiB")
        a = torch.randn(64, 64, device="cuda")
        b = a @ a
        torch.cuda.synchronize()
        print(f"CUDA matmul:     PASS ({float(b[0, 0]):.4f})")
    else:
        print("CUDA matmul:     SKIPPED")
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    commands = {
        "run": command_run,
        "validate-data": command_validate_data,
        "grade": command_grade,
        "fidelity": command_fidelity,
        "list-tasks": lambda _args: command_list_tasks(),
        "doctor": lambda _args: command_doctor(),
    }
    try:
        return commands[args.command](args)
    except KeyboardInterrupt:
        print("Interrupted. Completed JSONL rows are resumable.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        if os.environ.get("SANTAPP_RULER_TRACEBACK") == "1":
            raise
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
