"""Independent CSR-parent -> actual-key teams kernels (P001 prefill).

Small parents use one fused CTA per parent, bucketed by power-of-two length.
Large parents use a bounded-tile streaming path: no truncation, fixed capacity,
per-parent Python loop, or fallback to the slow PyTorch representative builder.
"""

import triton as tr
import triton.language as tl


@tr.jit
def small_parent_teams(Keys, ParentMembers, Indptr, ParentIds, TeamOffsets,
                       Members, Starts, Lengths, LeaderIds, LeaderKeys, TeamParents,
                       D: tl.constexpr, R: tl.constexpr, BN: tl.constexpr,
                       BD: tl.constexpr, DIRECT: tl.constexpr, FULL_SPAN: tl.constexpr):
    parent = tl.load(ParentIds + tl.program_id(0))
    start = tl.load(Indptr + parent)
    count = (tl.load(Indptr + parent + 1) - start).to(tl.int32)
    base = tl.load(TeamOffsets + parent)
    i = tl.arange(0, BN)
    ds = tl.arange(0, BD)
    rows = tl.load(ParentMembers + start + i, i < count, 0)
    x = tl.load(Keys + rows[:, None] * D + ds[None, :],
                (i[:, None] < count) & (ds[None, :] < D), 0).to(tl.float32)
    mean = tl.div_rn(tl.sum(x, 0), count.to(tl.float32))
    delta = x - mean[None, :]
    first = tl.argmin(tl.where(i < count, tl.sum(delta * delta, 1), float('inf')),
                      0, tie_break_left=True)
    selected = tl.full((BN,), False, tl.int1)
    owner = tl.full((BN,), -1, tl.int32)
    nearest = tl.full((BN,), float('inf'), tl.float32)
    best_distance = tl.full((BN,), float('inf'), tl.float32)
    assignment = tl.full((BN,), 0, tl.int32)
    x_norm = tl.sum(x * x, 1)
    for r in tl.static_range(0, R):
        if r < count:
            if r == 0:
                chosen = first
            else:
                score = tl.where((i < count) & ~selected, nearest, -1.0)
                chosen = tl.argmax(score, 0, tie_break_left=True)
            global_row = tl.load(ParentMembers + start + chosen)
            leader = tl.load(Keys + global_row * D + ds, ds < D, 0).to(tl.float32)
            delta = x - leader[None, :]
            direct_distance = tl.sum(delta * delta, 1)
            nearest = tl.minimum(nearest, direct_distance)
            selected = selected | (i == chosen)
            owner = tl.where(i == chosen, r, owner)
            if DIRECT or ((FULL_SPAN > 0) & (count == FULL_SPAN)):
                dist = direct_distance
            else:
                dist = tl.maximum(x_norm + tl.sum(leader * leader, 0)
                                  - 2.0 * tl.sum(x * leader[None, :], 1), 0.0)
            better = dist < best_distance
            assignment = tl.where(better, r, assignment)
            best_distance = tl.minimum(best_distance, dist)
            tl.store(LeaderIds + base + r, global_row)
            tl.store(LeaderKeys + (base + r) * D + ds, leader, ds < D)
    assignment = tl.where(owner >= 0, owner, assignment)
    cursor = start
    for r in tl.static_range(0, R):
        if r < count:
            take = (i < count) & (assignment == r)
            length = tl.sum(take.to(tl.int32), 0)
            rank = tl.cumsum(take.to(tl.int32), 0) - 1
            tl.store(Members + cursor + rank, rows, take)
            tl.store(Starts + base + r, cursor)
            tl.store(Lengths + base + r, length)
            tl.store(TeamParents + base + r, parent)
            cursor += length


@tr.jit
def streaming_parent_teams(Keys, ParentMembers, Indptr, ParentIds, TeamOffsets, Scratch,
                           Members, Starts, Lengths, LeaderIds, LeaderKeys, TeamParents,
                           D: tl.constexpr, R: tl.constexpr, BR: tl.constexpr,
                           BN: tl.constexpr, BD: tl.constexpr,
                           DIRECT: tl.constexpr, FULL_SPAN: tl.constexpr):
    parent = tl.load(ParentIds + tl.program_id(0))
    start = tl.load(Indptr + parent)
    count = (tl.load(Indptr + parent + 1) - start).to(tl.int32)
    base = tl.load(TeamOffsets + parent)
    local = tl.arange(0, BN)
    ds = tl.arange(0, BD)
    rs = tl.arange(0, BR)
    blocks = tl.cdiv(count, BN)
    mean_sum = tl.full((BD,), 0.0, tl.float32)
    for block in range(blocks):
        i = block * BN + local
        rows = tl.load(ParentMembers + start + i, i < count, 0)
        x = tl.load(Keys + rows[:, None] * D + ds[None, :],
                    (i[:, None] < count) & (ds[None, :] < D), 0).to(tl.float32)
        mean_sum += tl.sum(x, 0)
    mean = tl.div_rn(mean_sum, count.to(tl.float32))
    first_distance = tl.full((), float('inf'), tl.float32)
    first = tl.full((), 2147483647, tl.int32)
    for block in range(blocks):
        i = block * BN + local
        rows = tl.load(ParentMembers + start + i, i < count, 0)
        x = tl.load(Keys + rows[:, None] * D + ds[None, :],
                    (i[:, None] < count) & (ds[None, :] < D), 0).to(tl.float32)
        delta = x - mean[None, :]
        distance = tl.where(i < count, tl.sum(delta * delta, 1), float('inf'))
        value = tl.min(distance, 0)
        index = tl.min(tl.where((i < count) & (distance == value), i, 2147483647), 0)
        better = (value < first_distance) | ((value == first_distance) & (index < first))
        first = tl.where(better, index, first)
        first_distance = tl.minimum(first_distance, value)
    chosen = tl.where(rs == 0, first, 0)
    # Recompute nearest distances for each new leader instead of allocating a
    # parent_size-by-D register tile. R is small (normally four); no upper bound
    # is imposed on a parent's membership by this fallback.
    for r in tl.static_range(1, R):
        if r < count:
            best_score = tl.full((), -1.0, tl.float32)
            best_index = tl.full((), 2147483647, tl.int32)
            for block in range(blocks):
                i = block * BN + local
                rows = tl.load(ParentMembers + start + i, i < count, 0)
                x = tl.load(Keys + rows[:, None] * D + ds[None, :],
                            (i[:, None] < count) & (ds[None, :] < D), 0).to(tl.float32)
                nearest = tl.full((BN,), float('inf'), tl.float32)
                selected = tl.full((BN,), False, tl.int1)
                for s in tl.static_range(0, R):
                    if s < r:
                        pos = tl.sum(tl.where(rs == s, chosen, 0), 0)
                        row = tl.load(ParentMembers + start + pos)
                        leader = tl.load(Keys + row * D + ds, ds < D, 0).to(tl.float32)
                        delta = x - leader[None, :]
                        nearest = tl.minimum(nearest, tl.sum(delta * delta, 1))
                        selected = selected | (i == pos)
                valid = (i < count) & ~selected
                score = tl.where(valid, nearest, -1.0)
                value = tl.max(score, 0)
                index = tl.min(tl.where(valid & (score == value), i, 2147483647), 0)
                better = (value > best_score) | ((value == best_score) & (index < best_index))
                best_index = tl.where(better, index, best_index)
                best_score = tl.maximum(best_score, value)
            chosen = tl.where(rs == r, best_index, chosen)
    totals = tl.full((BR,), 0, tl.int32)
    for block in range(blocks):
        i = block * BN + local
        rows = tl.load(ParentMembers + start + i, i < count, 0)
        x = tl.load(Keys + rows[:, None] * D + ds[None, :],
                    (i[:, None] < count) & (ds[None, :] < D), 0).to(tl.float32)
        x_norm = tl.sum(x * x, 1)
        best = tl.full((BN,), float('inf'), tl.float32)
        assignment = tl.full((BN,), 0, tl.int32)
        owner = tl.full((BN,), -1, tl.int32)
        for r in tl.static_range(0, R):
            if r < count:
                pos = tl.sum(tl.where(rs == r, chosen, 0), 0)
                row = tl.load(ParentMembers + start + pos)
                leader = tl.load(Keys + row * D + ds, ds < D, 0).to(tl.float32)
                if DIRECT or ((FULL_SPAN > 0) & (count == FULL_SPAN)):
                    delta = x - leader[None, :]
                    dist = tl.sum(delta * delta, 1)
                else:
                    dist = tl.maximum(x_norm + tl.sum(leader * leader, 0)
                                      - 2.0 * tl.sum(x * leader[None, :], 1), 0.0)
                assignment = tl.where(dist < best, r, assignment)
                best = tl.minimum(best, dist)
                owner = tl.where(i == pos, r, owner)
        assignment = tl.where(owner >= 0, owner, assignment)
        tl.store(Scratch + start + i, assignment, i < count)
        for r in tl.static_range(0, R):
            length = tl.sum(((i < count) & (assignment == r)).to(tl.int32), 0)
            totals += tl.where(rs == r, length, 0)
    offsets = tl.cumsum(totals, 0) - totals
    for r in tl.static_range(0, R):
        if r < count:
            pos = tl.sum(tl.where(rs == r, chosen, 0), 0)
            row = tl.load(ParentMembers + start + pos)
            leader = tl.load(Keys + row * D + ds, ds < D, 0).to(tl.float32)
            offset = tl.sum(tl.where(rs == r, offsets, 0), 0)
            length = tl.sum(tl.where(rs == r, totals, 0), 0)
            tl.store(Starts + base + r, start + offset)
            tl.store(Lengths + base + r, length)
            tl.store(LeaderIds + base + r, row)
            tl.store(LeaderKeys + (base + r) * D + ds, leader, ds < D)
            tl.store(TeamParents + base + r, parent)
    # All scratch rows are owned by this CTA; synchronize its threads before
    # the second traversal reads those writes.
    tl.debug_barrier()
    written = tl.full((BR,), 0, tl.int32)
    for block in range(blocks):
        i = block * BN + local
        rows = tl.load(ParentMembers + start + i, i < count, 0)
        assignment = tl.load(Scratch + start + i, i < count, -1)
        for r in tl.static_range(0, R):
            if r < count:
                take = (i < count) & (assignment == r)
                rank = tl.cumsum(take.to(tl.int32), 0) - 1
                offset = tl.sum(tl.where(rs == r, offsets + written, 0), 0)
                tl.store(Members + start + offset + rank, rows, take)
                length = tl.sum(take.to(tl.int32), 0)
                written += tl.where(rs == r, length, 0)
