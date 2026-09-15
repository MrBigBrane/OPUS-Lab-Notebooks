"""Executable defaults. Importing the library alone does not mutate the environment."""
from __future__ import annotations

import os
from collections.abc import MutableMapping

DEFAULT_ALLOCATOR_CONFIG = "expandable_segments:True"


def configure_allocator(environ: MutableMapping[str, str] | None = None) -> dict[str, str | None]:
    """Call at executable startup, before importing torch; preserve either user spelling.

    This is a fragmentation mitigation, not a VRAM capacity guarantee. Library
    callers that have already imported torch should set the environment before
    starting their Python process, rather than relying on this helper.
    """
    env = os.environ if environ is None else environ
    if "PYTORCH_ALLOC_CONF" not in env and "PYTORCH_CUDA_ALLOC_CONF" not in env:
        env["PYTORCH_CUDA_ALLOC_CONF"] = DEFAULT_ALLOCATOR_CONFIG
    return {name: env.get(name) for name in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF")}
