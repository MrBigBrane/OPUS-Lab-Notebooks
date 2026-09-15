"""Small crash-safe filesystem helpers used by local and Kubernetes runs."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def _fsync_directory(path: Path) -> None:
    """Best-effort directory sync after replacing a file."""

    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_bytes(path: str | Path, payload: bytes) -> None:
    """Replace ``path`` atomically with fully flushed bytes.

    The temporary file is created beside the destination so ``os.replace`` stays
    on the same filesystem, including CephFS-backed Kubernetes volumes.
    """

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_text(path: str | Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: str | Path, value: Any, *, indent: int = 2) -> None:
    text = json.dumps(
        value,
        indent=indent,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=False,
    )
    atomic_write_text(path, text + "\n")


def append_jsonl_fsync(path: str | Path, record: dict[str, Any]) -> None:
    """Append one complete JSON object and flush it to stable storage."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    descriptor = os.open(
        destination,
        os.O_WRONLY | os.O_CREAT | os.O_APPEND,
        0o644,
    )
    try:
        # A single os.write avoids interleaving within one process. Kubernetes
        # shards never intentionally share a prediction file; a shard lock also
        # guards rare duplicate Indexed-Job pods.
        written = os.write(descriptor, encoded)
        if written != len(encoded):
            raise OSError(
                f"Short JSONL append to {destination}: {written}/{len(encoded)} bytes"
            )
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def repair_trailing_partial_jsonl(path: str | Path) -> bool:
    """Normalize the final JSONL record after an interrupted append.

    A valid final object that is missing only its newline is completed in place.
    An invalid, unterminated final fragment is removed. Completed lines and any
    corruption before the last physical line are never modified. Returns ``True``
    when the file was changed.
    """

    destination = Path(path)
    if not destination.is_file() or destination.stat().st_size == 0:
        return False
    payload = destination.read_bytes()
    if payload.endswith(b"\n"):
        return False

    last_newline = payload.rfind(b"\n")
    fragment_start = last_newline + 1
    fragment = payload[fragment_start:]
    try:
        decoded = fragment.decode("utf-8")
        value = json.loads(decoded)
        if not isinstance(value, dict):
            raise ValueError("final JSONL value is not an object")
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        with destination.open("r+b") as handle:
            handle.truncate(fragment_start)
            handle.flush()
            os.fsync(handle.fileno())
        return True

    # A crash or short write can leave a complete JSON object without the final
    # newline. Add it before any future O_APPEND write so records cannot merge.
    with destination.open("ab") as handle:
        handle.write(b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    return True
