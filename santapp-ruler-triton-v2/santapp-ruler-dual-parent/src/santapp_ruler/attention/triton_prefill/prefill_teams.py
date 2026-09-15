"""Native label/CSR/contiguous parent -> team preparation, ported from P001.

Only Python metadata construction and stable grouping remain in PyTorch. There
is no reference-replay guard, decode kernel, model loader, or second repository.
Small parents are fused; large parents use a bounded-tile streaming kernel.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..teams import TeamSummary


@dataclass(slots=True)
class ParentCSR:
    """Parent-major members, ascending original token indices within each parent.

    indptr has configured_parent_count+1 entries, including empty parents.
    These tensors index the ORIGINAL key/cache order; keys are not permuted.
    """
    members: torch.Tensor
    indptr: torch.Tensor

    @property
    def configured_parent_count(self) -> int:
        return self.indptr.numel() - 1


@dataclass(slots=True)
class PrefillTeamSummary(TeamSummary):
    """Decode-compatible summary with both supplied RULER metadata contracts."""
    partial_parent_size: int = 0


def parents_from_labels(labels: torch.Tensor, parent_cluster_count: int | None = None) -> ParentCSR:
    """One stable global sort, not one N-element scan for every parent.

    Works on CPU for tests/metadata and on CUDA for actual accelerated builds.
    Label IDs and ascending original token order are preserved, including gaps.
    """
    if labels.ndim != 1 or labels.numel() == 0:
        raise ValueError("parent_labels must be a nonempty one-dimensional tensor.")
    if labels.dtype not in {torch.int32, torch.int64}:
        raise TypeError("parent_labels must be int32 or int64.")
    labels = labels.to(torch.int64).contiguous()
    low, high = torch.aminmax(labels)
    low, high = int(low.item()), int(high.item())
    if low < 0:
        raise ValueError("parent_labels cannot be negative.")
    count = high + 1 if parent_cluster_count is None else int(parent_cluster_count)
    if count <= high:
        raise ValueError("parent_cluster_count does not cover every supplied label.")
    if count > 65536:
        raise ValueError("P001 supports at most 65536 configured parents.")
    members = torch.argsort(labels, stable=True)
    lengths = torch.bincount(labels, minlength=count)
    indptr = torch.cat((torch.zeros(1, dtype=torch.int64, device=labels.device),
                        lengths.cumsum(0)))
    return ParentCSR(members, indptr)


def contiguous_parents(token_count: int, parent_size: int, *, device=None) -> ParentCSR:
    """CSR directly from geometry; no sort and no membership search."""
    if token_count <= 0 or parent_size <= 0:
        raise ValueError("token_count and parent_size must be positive.")
    count = (token_count + parent_size - 1) // parent_size
    if count > 65536:
        raise ValueError("P001 supports at most 65536 configured parents.")
    members = torch.arange(token_count, device=device, dtype=torch.int64)
    indptr = torch.arange(count + 1, device=device, dtype=torch.int64) * parent_size
    indptr.clamp_max_(token_count)
    return ParentCSR(members, indptr)


def parent_buckets(lengths: np.ndarray, small_limit: int = 128) -> list[tuple[int, np.ndarray]]:
    """Host-only launch planning: (tile size, parent IDs); tile=0 is streaming."""
    if small_limit < 1 or small_limit & (small_limit - 1):
        raise ValueError("small_limit must be a positive power of two.")
    lengths = np.asarray(lengths, dtype=np.int64)
    result = []
    lower, upper = 0, 1
    while upper <= small_limit:
        ids = np.flatnonzero((lengths > lower) & (lengths <= upper)).astype(np.int32)
        if ids.size:
            result.append((upper, ids))
        lower, upper = upper, upper * 2
    ids = np.flatnonzero(lengths > small_limit).astype(np.int32)
    if ids.size:
        result.append((0, ids))
    return result


def validate_parent_csr(parents: ParentCSR, token_count: int, device) -> np.ndarray:
    """Validate caller-owned CSR before launch; return host parent lengths."""
    if parents.members.ndim != 1 or parents.indptr.ndim != 1:
        raise ValueError("CSR members and indptr must be one-dimensional.")
    if parents.members.dtype != torch.int64 or parents.indptr.dtype != torch.int64:
        raise TypeError("CSR members and indptr must be int64.")
    if parents.members.device != torch.device(device) or parents.indptr.device != torch.device(device):
        raise ValueError("CSR tensors and keys must be on the same device.")
    if not parents.members.is_contiguous() or not parents.indptr.is_contiguous():
        raise ValueError("CSR tensors must be contiguous.")
    ptr = parents.indptr.detach().cpu().numpy()
    if len(ptr) < 2 or ptr[0] != 0 or ptr[-1] != token_count or np.any(np.diff(ptr) < 0):
        raise ValueError("indptr must be monotone, start at zero and end at token_count.")
    if len(ptr) - 1 > 65536 or parents.members.numel() != token_count:
        raise ValueError("Invalid CSR parent count or member count.")
    expected = torch.arange(token_count, device=device, dtype=torch.int64)
    if not torch.equal(torch.sort(parents.members).values, expected):
        raise ValueError("CSR members must partition every original token exactly once.")
    if token_count > 1:
        # A descent is allowed ONLY at a parent boundary. Equal members were
        # already rejected by the permutation check.
        bad = parents.members[1:] <= parents.members[:-1]
        boundaries = parents.indptr[1:-1]
        boundaries = boundaries[(boundaries > 0) & (boundaries < token_count)]
        bad[boundaries - 1] = False
        if bool(bad.any()):
            raise ValueError("Members must be ascending within each parent (left-most tie contract).")
    return np.diff(ptr)


def _check_keys(keys: torch.Tensor, representatives_per_parent: int) -> None:
    if keys.ndim != 2 or keys.shape[0] == 0:
        raise ValueError("keys must have nonempty shape [tokens, features].")
    if keys.shape[0] >= 2**31 or not 1 <= keys.shape[1] <= 256:
        raise ValueError("P001 supports fewer than 2^31 tokens and feature dimensions 1..256.")
    if not 1 <= representatives_per_parent <= 16:
        raise ValueError("P001 supports 1..16 representatives per parent.")
    if keys.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        raise TypeError("Accelerated teams require float16/bfloat16/float32 keys.")
    if keys.device.type != "cuda":
        raise ValueError("Accelerated team construction requires CUDA keys.")


@torch.inference_mode()
def _build(keys, parents, representatives_per_parent, *, validate,
           assignment_distance="gram", full_span=0, partial_parent_size=0):
    _check_keys(keys, representatives_per_parent)
    with torch.cuda.device(keys.device):
        return _build_on_device(
            keys, parents, representatives_per_parent, validate=validate,
            assignment_distance=assignment_distance, full_span=full_span,
            partial_parent_size=partial_parent_size,
        )


def _build_on_device(keys, parents, representatives_per_parent, *, validate,
                     assignment_distance, full_span, partial_parent_size):
    if assignment_distance not in {"gram", "direct"}:
        raise ValueError("assignment_distance must be gram or direct.")
    if not bool(torch.isfinite(keys).all()):
        raise ValueError("Keys must be finite.")
    if validate:
        lengths_cpu = validate_parent_csr(parents, keys.shape[0], keys.device)
    else:
        # Used internally only for CSR just constructed from validated labels or
        # geometry. Public users selecting validate=False own those invariants.
        lengths_cpu = np.diff(parents.indptr.detach().cpu().numpy())
    try:
        import triton
        from .kernels import p001_prefill_teams as kernels
    except ImportError as exc:
        raise RuntimeError("Triton is required for accelerated CUDA prefill.") from exc
    keys = keys.float().contiguous()
    n, d = keys.shape
    r = representatives_per_parent
    team_counts = np.minimum(lengths_cpu, r)
    team_offsets_cpu = np.concatenate(([0], np.cumsum(team_counts, dtype=np.int64)))
    teams = int(team_offsets_cpu[-1])
    if teams == 0:
        raise ValueError("No active parent exists.")
    team_offsets = torch.as_tensor(team_offsets_cpu, dtype=torch.int64, device=keys.device)
    def long(size):
        return torch.empty(size, dtype=torch.int64, device=keys.device)

    members, starts, lengths, leader_ids, team_parents = (
        long(n), long(teams), long(teams), long(teams), long(teams),
    )
    leader_keys = torch.empty((teams, d), dtype=torch.float32, device=keys.device)
    scratch = None
    for tile, ids_cpu in parent_buckets(lengths_cpu):
        ids = torch.as_tensor(ids_cpu, dtype=torch.int32, device=keys.device)
        common = (keys, parents.members, parents.indptr, ids, team_offsets)
        outputs = (members, starts, lengths, leader_ids, leader_keys, team_parents)
        if tile:
            kernels.small_parent_teams[(len(ids_cpu),)](
                *common, *outputs, d, r, tile, triton.next_power_of_2(d),
                assignment_distance == "direct", full_span,
                num_warps=4, enable_fp_fusion=False,
            )
        else:
            scratch = torch.empty(n, dtype=torch.int32, device=keys.device)
            kernels.streaming_parent_teams[(len(ids_cpu),)](
                *common, scratch, *outputs, d, r, triton.next_power_of_2(r),
                64, triton.next_power_of_2(d), assignment_distance == "direct", full_span,
                num_warps=4, enable_fp_fusion=False,
            )
    summary = PrefillTeamSummary(
        members=members, starts=starts, lengths_long=lengths, lengths_float=lengths.float(),
        leader_keys=leader_keys, leader_token_indices=leader_ids, parent_ids=team_parents,
        active_parent_count=int(np.count_nonzero(lengths_cpu)),
        parent_lengths_long=torch.as_tensor(lengths_cpu[lengths_cpu > 0],
                                            dtype=torch.int64, device=keys.device),
        configured_parent_count=parents.configured_parent_count,
        partial_parent_size=partial_parent_size,
    )
    return summary


@torch.inference_mode()
def build_team_summary_from_csr(keys: torch.Tensor, parents: ParentCSR, *,
                                representatives_per_parent: int = 4,
                                validate: bool = True) -> PrefillTeamSummary:
    """Build teams from caller-supplied parents, irrespective of their origin.

    Uses the generic reference's norm/dot final assignment. Validation protects
    range/partition/order before a kernel can follow user-provided row indices.
    """
    return _build(keys, parents, representatives_per_parent, validate=validate)


@torch.inference_mode()
def build_team_summary_triton(keys: torch.Tensor, parent_labels: torch.Tensor, *,
                               representatives_per_parent: int = 4,
                               parent_cluster_count: int | None = None) -> PrefillTeamSummary:
    """Drop-in replacement for attention.teams.build_team_summary."""
    _check_keys(keys, representatives_per_parent)
    if parent_labels.ndim != 1 or parent_labels.numel() != keys.shape[0]:
        raise ValueError("parent_labels must contain one label per key.")
    if parent_labels.device != keys.device:
        raise ValueError("parent_labels and keys must be on the same device.")
    parents = parents_from_labels(parent_labels, parent_cluster_count)
    return _build(keys, parents, representatives_per_parent, validate=False)


@torch.inference_mode()
def build_contiguous_team_summary_triton(keys: torch.Tensor, *, parent_size: int = 16,
                                         representatives_per_parent: int = 4) -> PrefillTeamSummary:
    """Drop-in for the supplied RULER contiguous builder, including its tail.

    Full parents use direct squared distances for final assignment. The short
    tail uses norm/dot assignment, exactly as that reference's two paths do.
    """
    _check_keys(keys, representatives_per_parent)
    if parent_size <= 0 or representatives_per_parent > parent_size:
        raise ValueError("Require 0 < representatives_per_parent <= parent_size.")
    parents = contiguous_parents(keys.shape[0], parent_size, device=keys.device)
    return _build(keys, parents, representatives_per_parent, validate=False,
                  full_span=parent_size, partial_parent_size=keys.shape[0] % parent_size)
