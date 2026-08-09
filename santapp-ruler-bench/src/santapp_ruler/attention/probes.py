"""Probe-query position policy for SANTA++ clustering."""

from __future__ import annotations

import torch


def last_prompt_probe_positions(
    prompt_tokens: int,
    probe_queries: int,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return the final ``probe_queries`` prompt positions in chronological order."""

    if prompt_tokens <= 0:
        raise ValueError("prompt_tokens must be positive.")
    if probe_queries <= 0:
        raise ValueError("probe_queries must be positive.")
    if probe_queries > prompt_tokens:
        raise ValueError(
            f"probe_queries={probe_queries} exceeds prompt_tokens={prompt_tokens}."
        )
    return torch.arange(
        prompt_tokens - probe_queries,
        prompt_tokens,
        dtype=torch.long,
        device=device,
    )
