"""Layer-wide native-precision leaders and team-major prompt K/V.

Adapted from SANTA-Triton's v016-era packing contract (Apache-2.0).
Packing is setup work; validation and host copies must not run in decode.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch


MAX_REQUESTED_TEAMS = 1024


def requested_teams(samples_per_head: int, parent_size: int, representatives: int) -> int:
    if parent_size <= 0 or representatives <= 0 or parent_size % representatives:
        raise ValueError("An integral positive nominal team size is required")
    nominal = parent_size // representatives
    if samples_per_head <= 0 or samples_per_head % nominal:
        raise ValueError("The nominal sample budget must be a positive multiple of team size")
    count = samples_per_head // nominal
    if count > MAX_REQUESTED_TEAMS:
        raise ValueError(f"grouped_triton supports at most {MAX_REQUESTED_TEAMS} teams")
    return count


@dataclass(slots=True)
class PackedTeams:
    key: torch.Tensor                    # [Hkv, prompt, D], team-major
    value: torch.Tensor
    leaders: torch.Tensor                # [Hkv, padded_teams, D], native FP16/BF16
    starts: torch.Tensor                 # [Hkv, padded_teams], int32
    lengths: torch.Tensor
    counts: torch.Tensor                 # [Hkv], int32
    inverse: torch.Tensor                # [Hkv, prompt], original -> packed
    counts_host: tuple[int, ...]
    prompt_tokens: int

    @property
    def max_teams(self) -> int:
        return int(self.leaders.shape[1])

    def storage_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (
            self.key, self.value, self.leaders, self.starts, self.lengths,
            self.counts, self.inverse,
        ))


def pack_teams(summaries: Sequence[Any], key: torch.Tensor, value: torch.Tensor,
               budget: int) -> PackedTeams:
    """Pack existing summaries without changing parents, leaders, or membership.

    Native leaders are cast from the *frozen summaries*, not rebuilt from live K.
    The lossless-cast gate is intentionally setup-only. Empty teams are forbidden.
    Extra masked columns guarantee that one batched top-(K+1) also handles K>=N.
    """
    if not 1 <= budget <= MAX_REQUESTED_TEAMS:
        raise ValueError("requested teams must lie in [1, 1024]")
    if key.ndim != 3 or value.shape != key.shape:
        raise ValueError("K/V must have matching [Hkv, prompt, D] shapes")
    if key.dtype != value.dtype or key.device != value.device:
        raise ValueError("K/V dtype and device must match")
    if key.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("grouped_triton requires FP16 or BF16 cached keys")
    heads, tokens, dim = map(int, key.shape)
    if len(summaries) != heads or tokens <= 0:
        raise ValueError("One nonempty summary is required per KV head")
    counts_host = tuple(int(s.num_teams) for s in summaries)
    if min(counts_host) <= 0:
        raise ValueError("Each KV head must contain at least one team")
    padded = ((max(max(counts_host), budget + 1) + 63) // 64) * 64
    device = key.device
    pk = torch.empty((heads, tokens, dim), dtype=key.dtype, device=device)
    pv = torch.empty_like(pk)
    leaders = torch.zeros((heads, padded, dim), dtype=key.dtype, device=device)
    starts = torch.zeros((heads, padded), dtype=torch.int32, device=device)
    lengths = torch.zeros_like(starts)
    inverse = torch.empty((heads, tokens), dtype=torch.int32, device=device)
    expected = torch.arange(tokens, device=device)
    for h, summary in enumerate(summaries):
        if summary.members.device != device:
            raise ValueError("All summaries must reside on the K/V device")
        members = summary.members.long()
        lens = summary.lengths_long.long()
        n = counts_host[h]
        if members.numel() != tokens or not torch.equal(members.sort().values, expected):
            raise ValueError("Team members must partition every prompt row exactly once")
        if lens.numel() != n or not bool((lens > 0).all()) or int(lens.sum()) != tokens:
            raise ValueError("Team lengths must be positive and sum to prompt length")
        offsets = torch.cumsum(lens, 0) - lens
        if not torch.equal(offsets, summary.starts.long()):
            raise ValueError("Team starts must describe a contiguous member permutation")
        frozen = summary.leader_keys
        native = frozen.to(key.dtype)
        if frozen.shape != (n, dim) or not torch.equal(native.to(frozen.dtype), frozen):
            raise ValueError("Frozen actual-key leaders must round-trip losslessly to cache dtype")
        pk[h].copy_(key[h].index_select(0, members))
        pv[h].copy_(value[h].index_select(0, members))
        leaders[h, :n].copy_(native)
        starts[h, :n].copy_(offsets)
        lengths[h, :n].copy_(lens)
        inverse[h].scatter_(0, members, expected.to(torch.int32))
    return PackedTeams(pk, pv, leaders, starts, lengths,
                       torch.tensor(counts_host, dtype=torch.int32, device=device),
                       inverse, counts_host, tokens)


def refresh_prompt_rows_reference(packed: PackedTeams, key: torch.Tensor,
                                  value: torch.Tensor, start: int, end: int) -> None:
    """CPU/test oracle for packed refresh; not used by the CUDA decode hot path."""
    end = min(end, packed.prompt_tokens)
    for h in range(int(key.shape[0])):
        rows = packed.inverse[h, start:end].long()
        packed.key[h, rows] = key[h, start:end]
        packed.value[h, rows] = value[h, start:end]
