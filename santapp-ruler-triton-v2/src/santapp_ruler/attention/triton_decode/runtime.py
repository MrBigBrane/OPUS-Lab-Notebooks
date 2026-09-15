"""Preallocated, layer-wide CUDA dispatch for nominal budgets through 4096.

Only setup validates device contents. The steady-state path has no host readback,
per-head Python loop, dynamic selected-row allocation, or K-dependent launch loop.
"""
from __future__ import annotations

import math

import torch

from .packing import PackedTeams, pack_teams


class LayerDecoder:
    def __init__(self, summaries, key: torch.Tensor, value: torch.Tensor,
                 *, query_heads: int, budget: int, splits: int = 16):
        if key.device.type != "cuda":
            raise RuntimeError("grouped_triton requires CUDA; select decode_backend=torch on CPU")
        from ..prefill_runtime import require_triton_environment
        require_triton_environment(key.device)
        if key.shape[-1] not in (32, 64, 128, 256):
            raise ValueError("grouped_triton supports head dimensions 32, 64, 128, 256")
        kv = int(key.shape[0])
        if query_heads % kv or not 1 <= query_heads // kv <= 16:
            raise ValueError("grouped_triton requires 1..16 query heads per KV head")
        if not 1 <= splits <= 64:
            raise ValueError("decode_splits must be in [1, 64]")
        self.packed: PackedTeams = pack_teams(summaries, key, value, budget)
        self.heads = query_heads
        self.kv_heads = kv
        self.group = query_heads // kv
        self.dim = int(key.shape[-1])
        self.budget = budget
        self.splits = splits
        p = self.packed
        opts = dict(device=key.device, dtype=torch.float32)
        self.logits = torch.empty((query_heads, p.max_teams), **opts)
        self.priorities = torch.empty_like(self.logits)
        self.top_values = torch.empty((query_heads, budget+1), **opts)
        self.top_indices = torch.empty((query_heads, budget+1), device=key.device, dtype=torch.long)
        self.selected = torch.empty((query_heads, budget), device=key.device, dtype=torch.int32)
        self.logp = torch.empty((query_heads, budget), **opts)
        self.prefix = torch.empty((query_heads, budget+1), device=key.device, dtype=torch.int32)
        self.rows = torch.empty(query_heads, device=key.device, dtype=torch.int32)
        self.stamps = torch.zeros((kv, p.max_teams), device=key.device, dtype=torch.int32)
        self.pm = torch.empty((query_heads, splits), **opts)
        self.pl = torch.empty_like(self.pm)
        self.pa = torch.empty((query_heads, splits, self.dim), **opts)
        self.output = torch.empty((query_heads, self.dim), dtype=key.dtype, device=key.device)

    def workspace_bytes(self) -> int:
        return sum(t.numel()*t.element_size() for t in vars(self).values()
                   if isinstance(t, torch.Tensor))

    def route(self, query: torch.Tensor, *, seed: int, layer: int, epoch: int,
              uniforms: torch.Tensor | None = None) -> None:
        from .kernels import grouped_route
        p = self.packed
        grouped_route[(self.kv_heads, p.max_teams // 64)](
            query, p.leaders, p.lengths, p.counts, self.logits, self.priorities,
            self.priorities if uniforms is None else uniforms, seed, layer, epoch,
            T=p.max_teams, D=self.dim, G=self.group, BD=self.dim, BM=16, BN=64,
            USE_UNIFORMS=uniforms is not None, num_warps=4, num_stages=2,
        )

    def select(self, *, epoch: int) -> None:
        import triton
        from .kernels import compact_plan
        # GPU batched top-K deliberately replaces v16's local top32/k-way merge.
        # K is uniform for the layer; active team counts may differ by KV head.
        torch.topk(self.priorities, self.budget+1, dim=-1, largest=True,
                   sorted=True, out=(self.top_values, self.top_indices))
        p = self.packed
        compact_plan[(self.heads,)](
            self.top_values, self.top_indices, self.logits, p.lengths, p.counts,
            self.selected, self.logp, self.prefix, self.rows, self.stamps, epoch,
            T=p.max_teams, G=self.group, K=self.budget,
            BK=triton.next_power_of_2(self.budget), num_warps=4,
        )

    def attend(self, query: torch.Tensor, full_key: torch.Tensor,
               full_value: torch.Tensor) -> torch.Tensor:
        import triton
        from .kernels import reduce_partials, selected_attention
        p = self.packed
        selected_attention[(self.heads, self.splits)](
            query, p.key, p.value, full_key, full_value, p.starts, p.counts,
            self.selected, self.logp, self.prefix, self.rows, self.pm, self.pl, self.pa,
            int(full_key.shape[1]), *full_key.stride()[:2], *full_value.stride()[:2],
            N=p.prompt_tokens, T=p.max_teams, D=self.dim, G=self.group,
            K=self.budget, SPLITS=self.splits, BD=self.dim, BR=32,
            SEARCH=math.ceil(math.log2(self.budget+1)), num_warps=4, num_stages=2,
        )
        reduce_partials[(self.heads,)](
            self.pm, self.pl, self.pa, self.output, D=self.dim, BD=self.dim,
            SPLITS=self.splits, BS=triton.next_power_of_2(self.splits), num_warps=4,
        )
        return self.output

    def record(self, statistics: torch.Tensor, *, epoch: int) -> None:
        import triton
        from .kernels import diagnostics
        p = self.packed
        diagnostics[(self.kv_heads,)](
            p.lengths, p.counts, self.stamps, self.rows, self.logp, statistics, epoch,
            T=p.max_teams, K=self.budget, G=self.group,
            BT=triton.next_power_of_2(p.max_teams),
            BSEL=triton.next_power_of_2(self.group*self.budget),
            BG=triton.next_power_of_2(self.group), num_warps=4,
        )

    def run(self, query: torch.Tensor, full_key: torch.Tensor, full_value: torch.Tensor,
            *, seed: int, layer: int, epoch: int,
            statistics: torch.Tensor | None = None,
            uniforms: torch.Tensor | None = None) -> torch.Tensor:
        # Python metadata checks only; no reading CUDA tensor values here.
        if query.shape != (self.heads, self.dim) or not query.is_contiguous():
            raise ValueError("Query must be contiguous [Hq,D]")
        if query.dtype != self.packed.key.dtype or query.device != self.packed.key.device:
            raise ValueError("Query must match cached-key native dtype and device")
        if full_key.shape != full_value.shape or full_key.shape[0] != self.kv_heads:
            raise ValueError("K/V shapes must match the packed layer")
        if full_key.shape[1] < self.packed.prompt_tokens or full_key.shape[2] != self.dim:
            raise ValueError("Live K/V must include the entire prompt and match head dimension")
        if (full_key.dtype != self.packed.key.dtype or full_value.dtype != full_key.dtype
                or full_key.device != query.device or full_value.device != query.device
                or full_key.stride(-1) != 1 or full_value.stride(-1) != 1):
            raise ValueError("Live K/V must match dtype/device and have contiguous D")
        if not 0 <= epoch < 2**31-1 or not 0 <= layer < 2**32 or not 0 <= seed < 2**64:
            raise ValueError("Invalid Philox layer/epoch/seed identity")
        if uniforms is not None and (uniforms.shape != self.priorities.shape
                or uniforms.dtype != torch.float32 or uniforms.device != query.device
                or not uniforms.is_contiguous()):
            raise ValueError("Debug uniforms must be contiguous float32 [Hq,padded_teams]")
        if statistics is not None and (statistics.shape != (self.kv_heads, 6)
                or statistics.device != query.device or not statistics.is_contiguous()):
            raise ValueError("Statistics must be contiguous [Hkv,6] on the CUDA device")
        self.route(query, seed=seed, layer=layer, epoch=epoch, uniforms=uniforms)
        self.select(epoch=epoch)
        result = self.attend(query, full_key, full_value)
        if statistics is not None:
            self.record(statistics, epoch=epoch)
        return result

    def refresh(self, key: torch.Tensor, value: torch.Tensor, start: int, end: int) -> None:
        from .kernels import refresh_packed
        p = self.packed
        end = min(end, p.prompt_tokens)
        if end <= start:
            return
        refresh_packed[(self.kv_heads, end-start)](
            p.key, p.value, key, value, p.inverse, start, end,
            *key.stride()[:2], *value.stride()[:2],
            N=p.prompt_tokens, D=self.dim, BD=self.dim, num_warps=4,
        )

    def warmup(self, key: torch.Tensor, value: torch.Tensor, *, seed: int, layer: int) -> None:
        # Compile against actual native cache shapes, outside measured decode.
        query = key[:, -1].repeat_interleave(self.group, dim=0).contiguous()
        scratch = torch.empty((self.kv_heads, 6), dtype=torch.float64, device=key.device)
        self.run(query, key, value, seed=seed, layer=layer, epoch=0, statistics=scratch)
        # Compile the one-row refresh used when the final prompt token is re-fed.
        self.refresh(key, value, self.packed.prompt_tokens-1, self.packed.prompt_tokens)
        self.stamps.zero_()
