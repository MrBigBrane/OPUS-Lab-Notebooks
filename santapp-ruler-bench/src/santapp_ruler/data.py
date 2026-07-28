"""RULER dataset loading and deterministic per-task prompt selection."""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .config import BenchmarkConfig
from .ruler.grader import normalize_references
from .ruler.tasks import require_task


@dataclass(frozen=True, slots=True)
class RulerExample:
    task: str
    index: int | str
    input: str
    outputs: list[str]
    source_position: int
    reported_length: int | None = None

    @property
    def uid(self) -> str:
        return f"{self.task}:{self.source_position}:{self.index}"

    def to_manifest_record(self, *, include_prompt: bool = True) -> dict[str, Any]:
        record = asdict(self)
        record["uid"] = self.uid
        if not include_prompt:
            record.pop("input", None)
        return record


def _stable_task_seed(base_seed: int, task: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{task}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _normalize_row(task: str, row: Mapping[str, Any], position: int) -> RulerExample:
    if "input" not in row:
        raise ValueError(f"RULER row {task}[{position}] has no 'input' field.")
    prompt = str(row["input"])
    answer_prefix = row.get("answer_prefix")
    if answer_prefix:
        answer_prefix = str(answer_prefix)
        if not prompt.endswith(answer_prefix):
            prompt += answer_prefix

    raw_outputs = row.get("outputs", row.get("output"))
    if raw_outputs is None:
        raise ValueError(f"RULER row {task}[{position}] has no outputs field.")
    if isinstance(raw_outputs, str):
        outputs = [raw_outputs]
    else:
        outputs = normalize_references(raw_outputs)

    raw_index = row.get("index", position)
    if isinstance(raw_index, bool):
        raw_index = int(raw_index)
    elif not isinstance(raw_index, (int, str)):
        raw_index = str(raw_index)

    reported_length: int | None = None
    if row.get("length") is not None:
        try:
            reported_length = int(row["length"])
        except (TypeError, ValueError):
            reported_length = None

    return RulerExample(
        task=task,
        index=raw_index,
        input=prompt,
        outputs=outputs,
        source_position=position,
        reported_length=reported_length,
    )


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_number}")
            yield value


def _find_local_task_file(root: Path, task: str, subset: str) -> Path:
    candidates = (
        root / task / f"{subset}.jsonl",
        root / task / "validation.jsonl",
        root / f"{task}.jsonl",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    shown = "\n  - ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"No local RULER JSONL found for task {task!r}. Checked:\n  - {shown}"
    )


def _load_local_rows(config: BenchmarkConfig, task: str) -> list[Mapping[str, Any]]:
    assert config.data.local_root is not None
    path = _find_local_task_file(
        Path(config.data.local_root).expanduser().resolve(),
        task,
        config.data.subset,
    )
    return list(_iter_jsonl(path))


def _load_huggingface_rows(
    config: BenchmarkConfig, task: str
) -> Iterable[Mapping[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The 'datasets' package is required for benchmark.data.source=huggingface."
        ) from exc

    common = {
        "path": config.data.repository,
        "revision": config.data.revision,
    }
    errors: list[Exception] = []
    # The pinned convenience mirror exposes tasks as splits. The fallback also
    # supports repositories that expose one configuration per task.
    try:
        return load_dataset(split=task, **common)
    except Exception as exc:  # datasets raises several backend-specific types
        errors.append(exc)
    try:
        return load_dataset(name=task, split="train", **common)
    except Exception as exc:
        errors.append(exc)

    detail = "\n".join(f"  {type(err).__name__}: {err}" for err in errors)
    raise RuntimeError(
        f"Could not load task {task!r} from {config.data.repository!r} at "
        f"revision {config.data.revision!r}. Attempts failed:\n{detail}"
    )


def load_task_examples(config: BenchmarkConfig, task: str) -> list[RulerExample]:
    require_task(task)
    if config.data.source == "local":
        rows = _load_local_rows(config, task)
    elif config.data.source == "huggingface":
        rows = _load_huggingface_rows(config, task)
    else:  # validated earlier; defensive for direct callers
        raise ValueError(f"Unsupported data source: {config.data.source!r}")

    examples = [_normalize_row(task, row, i) for i, row in enumerate(rows)]
    if not examples:
        raise ValueError(f"Task {task!r} contains no examples.")
    return examples


def select_examples(config: BenchmarkConfig) -> dict[str, list[RulerExample]]:
    """Select exactly ``prompts_per_task`` examples for every chosen task."""
    selected: dict[str, list[RulerExample]] = {}
    requested = config.prompts_per_task
    for task in config.tasks:
        examples = load_task_examples(config, task)
        if len(examples) < requested:
            raise ValueError(
                f"Task {task!r} has {len(examples)} examples, fewer than the "
                f"requested prompts_per_task={requested}."
            )
        rng = random.Random(_stable_task_seed(config.selection_seed, task))
        positions = rng.sample(range(len(examples)), requested)
        selected[task] = [examples[position] for position in positions]
    return selected


def flatten_selected(
    selected: Mapping[str, list[RulerExample]], tasks: Iterable[str]
) -> list[RulerExample]:
    return [example for task in tasks for example in selected[task]]


def write_selection_manifest(
    selected: Mapping[str, list[RulerExample]],
    tasks: Iterable[str],
    path: str | Path,
    *,
    include_prompts: bool,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for example in flatten_selected(selected, tasks):
            record = example.to_manifest_record(include_prompt=include_prompts)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
