"""P001 prefill kernels. No decode kernel or protected reference is modified.

Greedy initialization is sequential in the center index, but the selected trial
stays on device. Assignment is tiled: never materialize the N-by-K distance
matrix. Explicit IEEE FP32 dots are the default in the Python launcher.
"""

import triton as tr
import triton.language as tl


@tr.jit
def gather_rows(X, Ids, Y, M: tl.constexpr, D: tl.constexpr, B: tl.constexpr):
    off = tl.program_id(0) * B + tl.arange(0, B)
    row = off // D
    col = off % D
    src = tl.load(Ids + row, row < M, 0)
    val = tl.load(X + src * D + col, row < M, 0)
    tl.store(Y + off, val, row < M)


@tr.jit
def first_center(X, Centers, Closest, N: tl.constexpr, D: tl.constexpr,
                 first_id, BN: tl.constexpr, BD: tl.constexpr):
    rows = tl.program_id(0) * BN + tl.arange(0, BN)
    ds = tl.arange(0, BD)
    x = tl.load(X + rows[:, None] * D + ds[None, :],
                (rows[:, None] < N) & (ds[None, :] < D), 0)
    c = tl.load(X + first_id * D + ds, ds < D, 0)
    # Match the norm/dot formula used by the reference initializer.
    dist = tl.maximum(tl.sum(x * x, 1) + tl.sum(c * c, 0)
                      - 2.0 * tl.sum(x * c[None, :], 1), 0.0)
    tl.store(Closest + rows, dist, rows < N)
    if tl.program_id(0) == 0:
        tl.store(Centers + ds, c, ds < D)


@tr.jit
def scan_closest(Closest, Prefix, Totals, N: tl.constexpr, B: tl.constexpr):
    block = tl.program_id(0)
    i = block * B + tl.arange(0, B)
    x = tl.load(Closest + i, i < N, 0)
    tl.store(Prefix + i, tl.cumsum(x, 0), i < N)
    tl.store(Totals + block, tl.sum(x, 0))


@tr.jit(do_not_specialize=["center_index"])
def sample_candidates(Prefix, Totals, Potential, Uniforms, CandidateIds,
                      N: tl.constexpr, NB: tl.constexpr, SB: tl.constexpr,
                      TB: tl.constexpr, TRIALS: tl.constexpr, center_index):
    # One tiny CTA per trial. NumPy's float64 uniforms are copied once for the
    # complete initialization; no .item() / D2H round trip per new center.
    trial = tl.program_id(0)
    b = tl.arange(0, TB)
    sums = tl.load(Totals + b, b < NB, 0)
    cumulative = tl.cumsum(sums, 0)
    # Exclusive scan via shifted inclusive values (NOT cumulative - sums).
    prev = tl.gather(cumulative, tl.maximum(b - 1, 0), 0)
    offsets = tl.where(b == 0, 0.0, prev)
    potential = tl.load(Potential)
    if center_index == 1:
        potential = tl.sum(sums, 0)
    u = tl.load(Uniforms + (center_index - 1) * TRIALS + trial)
    target = u * potential.to(tl.float64)
    # Search the exact chunk-offset + local-prefix array used below. The last
    # prefix in a chunk need not round identically to that chunk's tl.sum.
    last_i = tl.minimum((b + 1) * SB, N) - 1
    last = tl.load(Prefix + last_i, b < NB, 0) + offsets
    block = tl.minimum(tl.min(tl.where((b < NB) & (last.to(tl.float64) >= target),
                                     b, NB), 0), NB - 1)
    offset = tl.sum(tl.where(b == block, offsets, 0.0), 0)
    lo = block * SB
    hi = tl.minimum((block + 1) * SB, N)
    while lo < hi:
        mid = (lo + hi) // 2
        value = tl.load(Prefix + mid) + offset
        below = value.to(tl.float64) < target
        lo = tl.where(below, mid + 1, lo)
        hi = tl.where(below, hi, mid)
    tl.store(CandidateIds + trial, tl.minimum(lo, N - 1))


@tr.jit
def trial_distances(X, Closest, CandidateIds, Distances, Partials,
                    N: tl.constexpr, D: tl.constexpr, TRIALS: tl.constexpr,
                    BN: tl.constexpr, BT: tl.constexpr, BD: tl.constexpr,
                    PRECISION: tl.constexpr):
    block = tl.program_id(0)
    rows = block * BN + tl.arange(0, BN)
    ts = tl.arange(0, BT)
    ds = tl.arange(0, BD)
    candidates = tl.load(CandidateIds + ts, ts < TRIALS, 0)
    x = tl.load(X + rows[:, None] * D + ds[None, :],
                (rows[:, None] < N) & (ds[None, :] < D), 0)
    c = tl.load(X + candidates[None, :] * D + ds[:, None],
                (ts[None, :] < TRIALS) & (ds[:, None] < D), 0)
    dot = tl.dot(x, c, input_precision=PRECISION)
    dist = tl.maximum(tl.sum(x * x, 1)[:, None] + tl.sum(c * c, 0)[None, :]
                      - 2.0 * dot, 0.0)
    old = tl.load(Closest + rows, rows < N, 0)
    dist = tl.minimum(dist, old[:, None])
    tl.store(Distances + rows[:, None] * TRIALS + ts[None, :], dist,
             (rows[:, None] < N) & (ts[None, :] < TRIALS))
    sums = tl.sum(tl.where(rows[:, None] < N, dist, 0.0), 0)
    tl.store(Partials + block * TRIALS + ts, sums, ts < TRIALS)


@tr.jit(do_not_specialize=["center_index"])
def choose_and_commit(X, Centers, Closest, CandidateIds, Distances, Partials, Potential,
                      N: tl.constexpr, D: tl.constexpr, TRIALS: tl.constexpr,
                      NB: tl.constexpr, BP: tl.constexpr, BT: tl.constexpr,
                      BN: tl.constexpr, BD: tl.constexpr, center_index):
    # All CTAs reduce the same small trial table and select the same left-most
    # winner; the remaining N-element copy is parallel. No cross-CTA barrier.
    bs = tl.arange(0, BP)
    ts = tl.arange(0, BT)
    partial = tl.load(Partials + bs[:, None] * TRIALS + ts[None, :],
                      (bs[:, None] < NB) & (ts[None, :] < TRIALS), 0)
    potentials = tl.where(ts < TRIALS, tl.sum(partial, 0), float('inf'))
    best = tl.argmin(potentials, 0, tie_break_left=True)
    rows = tl.program_id(0) * BN + tl.arange(0, BN)
    distance = tl.load(Distances + rows * TRIALS + best, rows < N, 0)
    tl.store(Closest + rows, distance, rows < N)
    if tl.program_id(0) == 0:
        best_id = tl.load(CandidateIds + best)
        ds = tl.arange(0, BD)
        center = tl.load(X + best_id * D + ds, ds < D, 0)
        tl.store(Centers + center_index * D + ds, center, ds < D)
        tl.store(Potential, tl.min(potentials, 0))


@tr.jit
def assignment_partials(X, Centers, Minima, Ids,
                        N: tl.constexpr, K: tl.constexpr, D: tl.constexpr,
                        NC: tl.constexpr, BM: tl.constexpr, BC: tl.constexpr,
                        BD: tl.constexpr, PRECISION: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    group = tl.program_id(1)
    cs = group * BC + tl.arange(0, BC)
    ds = tl.arange(0, BD)
    x = tl.load(X + rows[:, None] * D + ds[None, :],
                (rows[:, None] < N) & (ds[None, :] < D), 0)
    c = tl.load(Centers + cs[None, :] * D + ds[:, None],
                (cs[None, :] < K) & (ds[:, None] < D), 0)
    dot = tl.dot(x, c, input_precision=PRECISION)
    dist = tl.maximum(tl.sum(x * x, 1)[:, None] + tl.sum(c * c, 0)[None, :]
                      - 2.0 * dot, 0.0)
    dist = tl.where(cs[None, :] < K, dist, float('inf'))
    best = tl.min(dist, 1)
    best_id = tl.min(tl.where(dist == best[:, None], cs[None, :], 2147483647), 1)
    tl.store(Minima + rows * NC + group, best, rows < N)
    tl.store(Ids + rows * NC + group, best_id, rows < N)


@tr.jit
def finish_assignment(Minima, Ids, Labels, InertiaParts,
                      N: tl.constexpr, NC: tl.constexpr, BG: tl.constexpr,
                      BM: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    gs = tl.arange(0, BG)
    dist = tl.load(Minima + rows[:, None] * NC + gs[None, :],
                   (rows[:, None] < N) & (gs[None, :] < NC), float('inf'))
    ids = tl.load(Ids + rows[:, None] * NC + gs[None, :],
                  (rows[:, None] < N) & (gs[None, :] < NC), 2147483647)
    best = tl.min(dist, 1)
    label = tl.min(tl.where(dist == best[:, None], ids, 2147483647), 1)
    tl.store(Labels + rows, label.to(tl.int64), rows < N)
    tl.store(InertiaParts + tl.program_id(0), tl.sum(tl.where(rows < N, best, 0.0), 0))


@tr.jit
def sum_scalar(X, Out, N: tl.constexpr, B: tl.constexpr):
    i = tl.arange(0, B)
    tl.store(Out, tl.sum(tl.load(X + i, i < N, 0), 0))


@tr.jit
def zero_update(Sums, BatchCounts, K: tl.constexpr, D: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    tl.store(Sums + i, 0.0, i < K * D)
    tl.store(BatchCounts + i, 0, i < K)


@tr.jit
def accumulate_batch(X, Labels, Sums, BatchCounts, N: tl.constexpr, D: tl.constexpr,
                     BM: tl.constexpr, BD: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    ds = tl.arange(0, BD)
    label = tl.load(Labels + rows, rows < N, 0)
    x = tl.load(X + rows[:, None] * D + ds[None, :],
                (rows[:, None] < N) & (ds[None, :] < D), 0)
    # Like the reference's CUDA index_add_, FP32 atomics are not a promise of
    # bitwise deterministic summation. Counts are integer and exact.
    tl.atomic_add(Sums + label[:, None] * D + ds[None, :], x,
                  (rows[:, None] < N) & (ds[None, :] < D), sem='relaxed')
    tl.atomic_add(BatchCounts + label, 1, rows < N, sem='relaxed')


@tr.jit
def update_centers(Centers, Counts, Sums, BatchCounts, NewCenters, NewCounts,
                   K: tl.constexpr, D: tl.constexpr, BC: tl.constexpr, BD: tl.constexpr):
    cs = tl.program_id(0) * BC + tl.arange(0, BC)
    ds = tl.arange(0, BD)
    old_count = tl.load(Counts + cs, cs < K, 0)
    batch_count = tl.load(BatchCounts + cs, cs < K, 0).to(tl.float32)
    new_count = old_count + batch_count
    old = tl.load(Centers + cs[:, None] * D + ds[None, :],
                  (cs[:, None] < K) & (ds[None, :] < D), 0)
    sums = tl.load(Sums + cs[:, None] * D + ds[None, :],
                   (cs[:, None] < K) & (ds[None, :] < D), 0)
    value = tl.div_rn(old * old_count[:, None] + sums,
                      tl.maximum(new_count[:, None], 1.0))
    value = tl.where(batch_count[:, None] > 0, value, old)
    tl.store(NewCenters + cs[:, None] * D + ds[None, :], value,
             (cs[:, None] < K) & (ds[None, :] < D))
    tl.store(NewCounts + cs, new_count, cs < K)


@tr.jit
def reassignment_plan(Counts, Mask, Ranks, Floor, K: tl.constexpr, B: tl.constexpr):
    i = tl.arange(0, B)
    mask = tl.load(Mask + i, i < K, 0)
    counts = tl.load(Counts + i, i < K, float('inf'))
    tl.store(Ranks + i, tl.cumsum(mask.to(tl.int32), 0) - 1, i < K)
    tl.store(Floor, tl.min(tl.where(mask, float('inf'), counts), 0))


@tr.jit
def apply_reassignment(Centers, Counts, XBatch, Mask, Ranks, RowIds, Floor,
                       K: tl.constexpr, D: tl.constexpr, BC: tl.constexpr, BD: tl.constexpr):
    cs = tl.program_id(0) * BC + tl.arange(0, BC)
    ds = tl.arange(0, BD)
    mask = (cs < K) & tl.load(Mask + cs, cs < K, 0)
    rank = tl.load(Ranks + cs, mask, 0)
    row = tl.load(RowIds + rank, mask, 0)
    x = tl.load(XBatch + row[:, None] * D + ds[None, :],
                mask[:, None] & (ds[None, :] < D), 0)
    tl.store(Centers + cs[:, None] * D + ds[None, :], x,
             mask[:, None] & (ds[None, :] < D))
    tl.store(Counts + cs, tl.load(Floor), mask)
