"""Shared path, lock, and environment helpers for Nautilus workers."""

from __future__ import annotations

import fcntl
import json
import os
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .io_utils import atomic_write_json
from .k8s_matrix import ExperimentMatrix, cache_fingerprint, experiment_root


class CacheNotReadyError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def compact_utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")


@dataclass(frozen=True, slots=True)
class KubernetesPaths:
    results_root: Path
    experiment_root: Path
    cache_root: Path
    hf_home: Path
    hub_cache: Path
    datasets_cache: Path
    local_data_root: Path
    cache_marker: Path
    locks_root: Path
    shards_root: Path

    @classmethod
    def create(
        cls,
        results_root: str | Path,
        matrix: ExperimentMatrix,
        *,
        create_directories: bool = True,
    ) -> "KubernetesPaths":
        root = Path(results_root).expanduser().resolve()
        experiment = experiment_root(root, matrix)
        cache = experiment / "cache"
        result = cls(
            results_root=root,
            experiment_root=experiment,
            cache_root=cache,
            hf_home=cache / "huggingface",
            hub_cache=cache / "huggingface" / "hub",
            datasets_cache=cache / "huggingface" / "datasets",
            local_data_root=cache / "ruler_data",
            cache_marker=cache / "cache-ready.json",
            locks_root=experiment / "locks",
            shards_root=experiment / "shards",
        )
        if create_directories:
            for directory in (
                result.experiment_root,
                result.cache_root,
                result.hf_home,
                result.hub_cache,
                result.datasets_cache,
                result.local_data_root,
                result.locks_root,
                result.shards_root,
            ):
                directory.mkdir(parents=True, exist_ok=True)
        return result


def configure_huggingface_environment(
    paths: KubernetesPaths, *, offline: bool
) -> None:
    os.environ["HF_HOME"] = str(paths.hf_home)
    os.environ["HF_HUB_CACHE"] = str(paths.hub_cache)
    os.environ["HF_DATASETS_CACHE"] = str(paths.datasets_cache)
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
    else:
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)
        os.environ.pop("HF_DATASETS_OFFLINE", None)


@contextmanager
def exclusive_lock(path: str | Path) -> Iterator[None]:
    lock_path = Path(path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def read_json_object(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def model_snapshot_ready(marker: dict[str, Any]) -> tuple[bool, str]:
    """Check that the pinned Hugging Face model snapshot is actually materialized."""

    raw_path = marker.get("model_snapshot_path")
    if not isinstance(raw_path, str) or not raw_path:
        return False, "cache marker has no model_snapshot_path"
    snapshot = Path(raw_path)
    if not snapshot.is_dir():
        return False, f"model snapshot directory is missing: {snapshot}"
    if not (snapshot / "config.json").is_file():
        return False, f"model snapshot config.json is missing: {snapshot}"
    weights = [
        path
        for pattern in ("*.safetensors", "pytorch_model*.bin")
        for path in snapshot.glob(pattern)
        if path.is_file()
    ]
    if not weights:
        return False, f"model snapshot has no weight files: {snapshot}"
    tokenizer_files = (
        snapshot / "tokenizer.json",
        snapshot / "tokenizer_config.json",
        snapshot / "tokenizer.model",
    )
    if not any(path.is_file() for path in tokenizer_files):
        return False, f"model snapshot has no tokenizer files: {snapshot}"
    return True, "ready"


def require_cache_marker(
    paths: KubernetesPaths,
    matrix: ExperimentMatrix,
    base_config: Any,
) -> dict[str, Any]:
    if not paths.cache_marker.is_file():
        raise CacheNotReadyError(
            f"Shared cache marker is missing: {paths.cache_marker}. Run the CPU "
            "prefetch Job before the GPU Job."
        )
    marker = read_json_object(paths.cache_marker)
    if marker.get("status") != "ready":
        raise CacheNotReadyError(
            f"Shared cache marker is not ready: status={marker.get('status')!r}."
        )
    expected = cache_fingerprint(matrix, base_config)
    actual = marker.get("cache_fingerprint")
    if actual != expected:
        raise CacheNotReadyError(
            "Shared cache fingerprint does not match this matrix/base config: "
            f"expected {expected}, found {actual}. Re-run the prefetch Job."
        )
    model_ready, model_reason = model_snapshot_ready(marker)
    if not model_ready:
        raise CacheNotReadyError(model_reason)
    missing = [
        str(paths.local_data_root / task / "validation.jsonl")
        for task in matrix.tasks
        if not (paths.local_data_root / task / "validation.jsonl").is_file()
        or (paths.local_data_root / task / "validation.jsonl").stat().st_size == 0
    ]
    if missing:
        raise CacheNotReadyError(
            "Shared local RULER mirror is incomplete: " + ", ".join(missing)
        )
    selected = marker.get("selected_uids")
    if not isinstance(selected, dict):
        raise CacheNotReadyError("Shared cache marker has no selected_uids mapping.")
    invalid_tasks = [
        task
        for task in matrix.tasks
        if not isinstance(selected.get(task), list)
        or len(selected[task]) != matrix.prompts_per_task
    ]
    if invalid_tasks:
        raise CacheNotReadyError(
            "Shared cache marker has invalid selected UIDs for: "
            + ", ".join(invalid_tasks)
        )
    return marker


def write_event(path: str | Path, value: dict[str, Any]) -> None:
    atomic_write_json(path, value)


def execution_fingerprint(matrix: ExperimentMatrix) -> str:
    """Hash the repository code/config that defines one worker's semantics."""

    import hashlib

    repo_root = None
    for candidate in (matrix.source_path, *matrix.source_path.parents):
        if (candidate / "pyproject.toml").is_file():
            repo_root = candidate
            break
    if repo_root is None:
        raise FileNotFoundError(
            f"Could not locate repository root above {matrix.source_path}."
        )
    files: list[Path] = [
        repo_root / "pyproject.toml",
        matrix.source_path,
        matrix.resolve_base_config(),
    ]
    files.extend(sorted((repo_root / "src" / "santapp_ruler").rglob("*.py")))
    files.extend(
        path
        for path in sorted((repo_root / "scripts").glob("k8s_*.py"))
        if path.is_file()
    )
    digest = hashlib.sha256()
    for path in sorted(set(files), key=lambda value: str(value)):
        relative = path.relative_to(repo_root) if path.is_relative_to(repo_root) else path
        digest.update(str(relative).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()
