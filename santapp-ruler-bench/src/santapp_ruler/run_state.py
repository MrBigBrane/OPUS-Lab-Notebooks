"""Run-directory safety and deterministic selection-manifest handling."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

from .config import RunConfig, load_config, save_config
from .data import RulerExample, flatten_selected, write_selection_manifest
from .reporting import read_jsonl


def _result_identity(config: RunConfig) -> dict:
    """Return fields that must not change within one prediction directory."""
    value = config.to_dict()
    output = value.pop("output")
    # Storage location and resume behavior do not change generated results.
    # Whether prompts are present in the manifest does change its schema.
    value["output"] = {"save_full_prompts": output["save_full_prompts"]}
    return value


def prediction_files(run_dir: str | Path) -> list[Path]:
    root = Path(run_dir) / "predictions"
    return sorted(root.glob("*/*.jsonl")) if root.is_dir() else []


def prepare_run_directory(config: RunConfig, run_dir: str | Path) -> Path:
    """Create or validate a run directory before any output is modified.

    Resuming with a changed model, task selection, generation budget, or
    SANTA++ parameter would silently mix incompatible rows. This guard refuses
    that state. ``resume=false`` also refuses an existing prediction directory
    rather than deleting or duplicating rows.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.resolved.yaml"
    existing_predictions = prediction_files(run_dir)

    if config_path.is_file():
        existing = load_config(config_path)
        if _result_identity(existing) != _result_identity(config):
            raise ValueError(
                "The run directory already contains a different resolved "
                "benchmark configuration. Use a new --run-dir, or remove the "
                f"existing directory deliberately: {run_dir}"
            )
    elif existing_predictions:
        raise ValueError(
            "Prediction files exist without config.resolved.yaml, so safe resume "
            f"is impossible: {run_dir}"
        )
    else:
        save_config(config, config_path)

    if existing_predictions and not config.output.resume:
        raise FileExistsError(
            "output.resume=false was requested, but prediction JSONL already "
            f"exists under {run_dir}. Use a new --run-dir instead of mixing rows."
        )
    return run_dir


def validate_or_write_selection_manifest(
    selected: Mapping[str, list[RulerExample]],
    tasks: Iterable[str],
    path: str | Path,
    *,
    include_prompts: bool,
) -> None:
    """Ensure a resumed run uses exactly the original selected prompt rows."""
    path = Path(path)
    task_order = list(tasks)
    expected = [
        example.to_manifest_record(include_prompt=include_prompts)
        for example in flatten_selected(selected, task_order)
    ]
    if path.is_file():
        existing = read_jsonl(path)
        if existing != expected:
            raise ValueError(
                "The newly selected prompt manifest differs from the existing "
                f"run manifest: {path}. The data source may have changed; use a "
                "new run directory."
            )
        return
    write_selection_manifest(
        selected,
        task_order,
        path,
        include_prompts=include_prompts,
    )
