"""Command-line interface for local and Kubernetes runs."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _add_config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="DOTTED.PATH=VALUE",
        help="Override a YAML field; may be repeated.",
    )


def _add_selection_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--prompts-per-task", type=int)
    parser.add_argument("--context-length", type=int)
    parser.add_argument("--backends", nargs="+", choices=["sdpa", "santa", "santapp", "hierarchical"])
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--random-seed", type=int)
    parser.add_argument("--prefill-backend", choices=["torch", "triton"],
                        help="Choose preparation independently of decode.")
    parser.add_argument("--decode-backend", choices=["torch", "grouped_triton"],
                        help="Decoder for both whole-team methods; grouped supports up to 1024 teams.")
    parser.add_argument("--kmeans-fit-batch-size", type=int,
                        help="Independent Triton K-means fits per GPU batch (default 8), not token minibatch size.")
    parser.add_argument("--samples-per-head", type=int,
                        help="Set S for SANTA and the nominal token budget for both team backends.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="santapp-ruler")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Run and grade a RULER configuration.")
    _add_config_argument(run)
    _add_selection_arguments(run)
    run.add_argument("--run-dir", type=Path)
    run.add_argument("--no-resume", action="store_true")

    validate = subparsers.add_parser(
        "validate-data",
        help="Select prompts and verify post-framing token budgets without loading the model.",
    )
    _add_config_argument(validate)
    _add_selection_arguments(validate)

    grade = subparsers.add_parser("grade", help="Rebuild reports for one run directory.")
    grade.add_argument("run_dir", type=Path)
    grade.add_argument("--config", type=Path)
    grade.add_argument("--bootstrap-resamples", type=int)
    grade.add_argument("--confidence-level", type=float)
    grade.add_argument("--bootstrap-seed", type=int)

    fidelity = subparsers.add_parser(
        "fidelity",
        help="Compare stock SDPA with the temporary custom-cache patch in dense mode.",
    )
    _add_config_argument(fidelity)
    fidelity.add_argument("--tokens", type=int, default=8)
    fidelity.add_argument(
        "--prompt",
        default="Verify that the dense custom-cache path matches stock greedy decoding:",
    )

    subparsers.add_parser("list-tasks", help="List the 13 supported RULER tasks.")
    doctor = subparsers.add_parser("doctor", help="Check packages, CUDA, GPU memory, and default prefill stack.")
    doctor.add_argument("--prefill-backend", choices=["torch", "triton"], default="triton")
    return parser


def _config_with_cli(args: argparse.Namespace):
    from .config import load_config

    overrides = list(getattr(args, "overrides", []))
    if getattr(args, "tasks", None) is not None:
        overrides.append(f"benchmark.tasks={json.dumps(args.tasks)}")
    if getattr(args, "prompts_per_task", None) is not None:
        overrides.append(f"benchmark.prompts_per_task={args.prompts_per_task}")
    if getattr(args, "context_length", None) is not None:
        overrides.append(f"benchmark.context_length={args.context_length}")
    if getattr(args, "backends", None) is not None:
        overrides.append(f"generation.backends={json.dumps(args.backends)}")
    if getattr(args, "max_new_tokens", None) is not None:
        overrides.append(f"generation.max_new_tokens={args.max_new_tokens}")
    if getattr(args, "random_seed", None) is not None:
        overrides.append(f"generation.random_seed={args.random_seed}")
    if getattr(args, "prefill_backend", None) is not None:
        for name in ("santapp", "hierarchical"):
            overrides.append(f"{name}.prefill_backend={args.prefill_backend}")
    if getattr(args, "decode_backend", None) is not None:
        for name in ("santapp", "hierarchical"):
            overrides.append(f"{name}.decode_backend={args.decode_backend}")
    if getattr(args, "kmeans_fit_batch_size", None) is not None:
        overrides.append(f"santapp.kmeans.fit_batch_size={args.kmeans_fit_batch_size}")
    if getattr(args, "samples_per_head", None) is not None:
        for name in ("santa", "santapp", "hierarchical"):
            overrides.append(f"{name}.samples_per_head={args.samples_per_head}")
    if getattr(args, "no_resume", False):
        overrides.append("output.resume=false")
    return load_config(getattr(args, "config", None), overrides=overrides)


def command_run(args: argparse.Namespace) -> int:
    from .runner import run_benchmark

    run_benchmark(_config_with_cli(args), explicit_run_dir=args.run_dir)
    return 0


def command_validate_data(args: argparse.Namespace) -> int:
    from transformers import AutoTokenizer

    from .backends import encode_prompt
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
    print(f"Prompt framing: {config.model.prompt_format}")
    for task in config.benchmark.tasks:
        budget = config.generation.max_new_tokens or require_task(task).max_new_tokens
        lengths = []
        for example in selected[task]:
            length = int(
                encode_prompt(tokenizer, example.input, config.model.prompt_format).shape[1]
            )
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
    config = load_config(args.config or (run_dir / "config.resolved.yaml"))
    build_reports(
        run_dir,
        backends=config.generation.backends,
        tasks=config.benchmark.tasks,
        bootstrap_resamples=(
            args.bootstrap_resamples
            if args.bootstrap_resamples is not None
            else config.grading.bootstrap_resamples
        ),
        confidence_level=(
            args.confidence_level
            if args.confidence_level is not None
            else config.grading.confidence_level
        ),
        bootstrap_seed=(
            args.bootstrap_seed
            if args.bootstrap_seed is not None
            else config.grading.bootstrap_seed
        ),
    )
    print(f"Wrote {run_dir / 'summary.md'}")
    return 0


def command_fidelity(args: argparse.Namespace) -> int:
    from .attention.santapp import SantappEngine
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
    custom_ids = SantappEngine(
        bundle.model, config.santapp
    ).generate_dense_reference(input_ids, max_new_tokens=args.tokens)
    print("stock token IDs :", stock.token_ids)
    print("custom token IDs:", custom_ids)
    if stock.token_ids == custom_ids:
        print(f"PASS: all {args.tokens} generated token IDs match.")
        return 0
    print("FAIL: token IDs differ.")
    return 1


def command_list_tasks() -> int:
    from .ruler.tasks import TASKS

    print("Task                  Family                    Max new  Description")
    print("-" * 94)
    for task in TASKS.values():
        print(
            f"{task.name:21s} {task.family:25s} {task.max_new_tokens:7d}  "
            f"{task.description}"
        )
    return 0


def command_doctor(prefill_backend: str = "triton") -> int:
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
        from transformers.integrations.sdpa_attention import sdpa_attention_forward

        print(f"Transformers:    {transformers.__version__}")
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
    from .runtime_defaults import configure_allocator
    print(f"Prefill backend: {prefill_backend}")
    print(f"Allocator env:   {configure_allocator()}")
    print(f"CUDA usable:     {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("CUDA matmul:     SKIPPED")
        return 1
    if prefill_backend == "triton":
        from .attention.prefill_runtime import require_triton_environment
        info = require_triton_environment()
        print(f"Triton:          {info['triton']} (package/device check; run GPU smoke for kernels)")
    print(f"GPU:             {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    print(f"GPU memory:      {props.total_memory / (1024**3):.1f} GiB")
    a = torch.randn(64, 64, device="cuda")
    b = a @ a
    torch.cuda.synchronize()
    print(f"CUDA matmul:     PASS ({float(b[0, 0]):.4f})")
    if props.total_memory < 23 * 1024**3:
        print("WARNING: less than the intended 24 GB GPU class.")
    return 0


def main(argv: list[str] | None = None) -> int:
    from .runtime_defaults import configure_allocator

    configure_allocator()  # Before command handlers import torch or load weights.
    args = build_parser().parse_args(argv)
    commands = {
        "run": command_run,
        "validate-data": command_validate_data,
        "grade": command_grade,
        "fidelity": command_fidelity,
        "list-tasks": lambda _args: command_list_tasks(),
        "doctor": lambda args: command_doctor(args.prefill_backend),
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
