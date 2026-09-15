"""Grouped route, flexible compact plan, split-KV attention, and bounded diagnostics.

Apache-2.0; derived from SANTA-Triton's v010/v012/v016 execution contracts.
_counter_uniform is retained from v010. The stable Gumbel-tail polynomial and
online-softmax merge follow that lineage. Route/top-K and span traversal are new:
no K31 specialization, no Python head loop, no O(K) serial span walk per split.
"""
from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def _counter_uniform(seed, stream_id, epoch, query_head, team):
    counter_zero = team.to(tl.uint32)*0 + query_head.to(tl.uint32)*0
    counter_team = counter_zero + team.to(tl.uint32)
    counter_query_head = counter_zero + query_head.to(tl.uint32)
    counter_epoch = (counter_zero + epoch).to(tl.uint32)
    counter_stream = (counter_zero + stream_id).to(tl.uint32)
    random_bits, _, _, _ = tl.philox(
        seed, counter_team, counter_query_head, counter_epoch, counter_stream,
    )
    return tl.uint_to_uniform_float(random_bits)


@triton.jit
def _log_tail(delta):
    # Clamp each inactive branch too: tl.where is not short-circuit evaluation.
    x = tl.exp(tl.minimum(delta, -1.0))
    ratio = (1.0 - 0.5*x + x*x/6.0 - x*x*x/24.0 + x*x*x*x/120.0
             - x*x*x*x*x/720.0 + x*x*x*x*x*x/5040.0)
    series = delta + tl.log(ratio)
    x_direct = tl.exp(tl.minimum(tl.maximum(delta, -1.0), 20.0))
    direct = tl.log(1.0 - tl.exp(-x_direct))
    return tl.where(delta < -1.0, series, tl.where(delta > 20.0, 0.0, direct))


@triton.jit(do_not_specialize=['seed', 'layer', 'epoch'])
def grouped_route(Q, Leaders, Lengths, Counts, Logits, Priorities, Uniforms,
                  seed, layer, epoch,
                  T: tl.constexpr, D: tl.constexpr, G: tl.constexpr,
                  BD: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                  USE_UNIFORMS: tl.constexpr):
    """One KV-head/leader tile reuses native leaders across its GQA queries."""
    kv = tl.program_id(0)
    teams = tl.program_id(1)*BN + tl.arange(0, BN)
    g = tl.arange(0, BM)
    dims = tl.arange(0, BD)
    qh = kv*G + g
    q = tl.load(Q + qh[:, None]*D + dims[None, :],
                (g < G)[:, None] & (dims < D)[None, :], 0)
    k = tl.load(Leaders + kv*T*D + teams[None, :]*D + dims[:, None],
                (teams < T)[None, :] & (dims < D)[:, None], 0)
    dot = tl.dot(q, k, out_dtype=tl.float32)
    count = tl.load(Counts + kv)
    lens = tl.load(Lengths + kv*T + teams, teams < count, 1).to(tl.float32)
    logits = dot * (D ** -0.5) + tl.log(lens)[None, :]
    if USE_UNIFORMS:
        u = tl.load(Uniforms + qh[:, None]*T + teams[None, :],
                    (g < G)[:, None] & (teams < T)[None, :], 0.5)
    else:
        # Identical logical counter identity across budgets, tiles and launches.
        u = _counter_uniform(seed, layer, epoch, qh[:, None], teams[None, :])
    u = tl.minimum(tl.maximum(u, 1.1754943508222875e-38), 0.9999999403953552)
    valid = (g < G)[:, None] & (teams < count)[None, :]
    priority = logits - tl.log(-tl.log(u))
    offsets = qh[:, None]*T + teams[None, :]
    store = (g < G)[:, None] & (teams < T)[None, :]
    tl.store(Logits + offsets, tl.where(valid, logits, -float('inf')), store)
    tl.store(Priorities + offsets, tl.where(valid, priority, -float('inf')), store)


@triton.jit(do_not_specialize=['epoch'])
def compact_plan(TopValues, TopIndices, Logits, Lengths, Counts,
                 Selected, LogP, Prefix, RowCounts, Stamps, epoch,
                 T: tl.constexpr, G: tl.constexpr, K: tl.constexpr, BK: tl.constexpr):
    h = tl.program_id(0)
    kv = h // G
    slots = tl.arange(0, BK)
    active = tl.load(Counts + kv)
    count = tl.minimum(active, K)
    all_selected = active <= K
    ranked = tl.load(TopIndices + h*(K+1) + slots, slots < K, 0).to(tl.int32)
    team = tl.where(all_selected, slots, ranked)
    valid = slots < count
    safe = tl.where(valid, team, 0)
    chosen_logits = tl.load(Logits + h*T + safe, valid, 0.0)
    threshold = tl.load(TopValues + h*(K+1) + K)
    delta = chosen_logits - tl.where(all_selected, chosen_logits, threshold)
    logp = tl.where(all_selected | ~valid, 0.0, _log_tail(delta))
    lens = tl.load(Lengths + kv*T + safe, valid, 0).to(tl.int32)
    inclusive = tl.cumsum(lens, 0)
    tl.store(Selected + h*K + slots, tl.where(valid, team, -1), slots < K)
    tl.store(LogP + h*K + slots, logp, slots < K)
    tl.store(Prefix + h*(K+1), 0)
    tl.store(Prefix + h*(K+1) + slots + 1, inclusive, slots < K)
    tl.store(RowCounts + h, tl.sum(lens, 0))
    # Epoch-stamped union membership. No clearing kernel or retained token logs.
    tl.atomic_max(Stamps + kv*T + safe, epoch + 1, sem='relaxed', mask=valid)


@triton.jit(do_not_specialize=['total_tokens'])
def selected_attention(Q, PK, PV, FullK, FullV, Starts, Counts,
                       Selected, LogP, Prefix, RowCounts, PM, PL, PA,
                       total_tokens, full_k_head_stride, full_k_token_stride,
                       full_v_head_stride, full_v_token_stride,
                       N: tl.constexpr, T: tl.constexpr, D: tl.constexpr,
                       G: tl.constexpr, K: tl.constexpr, SPLITS: tl.constexpr,
                       BD: tl.constexpr, BR: tl.constexpr, SEARCH: tl.constexpr):
    """Split the selected logical row stream; binary-search compact team prefixes.

    All query heads launch together. Attention CTAs remain per query head/split,
    as in v16; this does NOT claim that selected K/V are GQA-unioned physically.
    """
    h = tl.program_id(0)
    split = tl.program_id(1)
    kv = h // G
    dims = tl.arange(0, BD)
    lanes = tl.arange(0, BR)
    query = tl.load(Q + h*D + dims, dims < D, 0).to(tl.float32)
    selected_rows = tl.load(RowCounts + h)
    n_selected = tl.minimum(tl.load(Counts + kv), K)
    total = selected_rows + total_tokens - N
    begin = total*split // SPLITS
    end = total*(split+1) // SPLITS
    running_m = -float('inf')
    running_l = 0.0
    acc = tl.zeros((BD,), tl.float32)
    for start in tl.range(begin, end, BR):
        rows = start + lanes
        valid = rows < end
        sampled = valid & (rows < selected_rows)
        # Search ends, not starts. Handles repeated padding prefixes and K>Nteams.
        target = tl.minimum(rows, selected_rows - 1)
        low = tl.full((BR,), 0, tl.int32)
        high = tl.zeros((BR,), tl.int32) + n_selected
        for _ in tl.static_range(SEARCH):
            mid = (low + high) // 2
            safe_mid = tl.minimum(mid, n_selected - 1)
            p_end = tl.load(Prefix + h*(K+1) + safe_mid + 1)
            right = (low < high) & (p_end <= target)
            high = tl.where((low < high) & ~right, mid, high)
            low = tl.where(right, mid+1, low)
        slot = tl.minimum(low, n_selected-1)
        p_start = tl.load(Prefix + h*(K+1) + slot)
        team = tl.load(Selected + h*K + slot)
        team_start = tl.load(Starts + kv*T + team)
        packed_rows = team_start + rows - p_start
        logp = tl.load(LogP + h*K + slot)
        exact_rows = N + rows - selected_rows
        packed_offsets = kv*N*D + packed_rows[:, None]*D + dims[None, :]
        full_k_offsets = kv*full_k_head_stride + exact_rows[:, None]*full_k_token_stride + dims[None, :]
        mask_s = sampled[:, None] & (dims < D)[None, :]
        mask_e = (valid & ~sampled)[:, None] & (dims < D)[None, :]
        ks = tl.load(PK + packed_offsets, mask_s, 0).to(tl.float32)
        ke = tl.load(FullK + full_k_offsets, mask_e, 0).to(tl.float32)
        keys = tl.where(sampled[:, None], ks, ke)
        scores = tl.sum(keys * query[None, :], 1)*(D ** -0.5)
        scores -= tl.where(sampled, logp, 0.0)
        scores = tl.where(valid, scores, -float('inf'))
        tile_m = tl.max(scores, 0)
        weights = tl.where(valid, tl.exp(scores - tile_m), 0.0)
        full_v_offsets = kv*full_v_head_stride + exact_rows[:, None]*full_v_token_stride + dims[None, :]
        vs = tl.load(PV + packed_offsets, mask_s, 0).to(tl.float32)
        ve = tl.load(FullV + full_v_offsets, mask_e, 0).to(tl.float32)
        values = tl.where(sampled[:, None], vs, ve)
        tile_acc = tl.sum(values * weights[:, None], 0)
        tile_l = tl.sum(weights, 0)
        merged_m = tl.maximum(running_m, tile_m)
        old_factor = tl.where(running_l > 0, tl.exp(running_m-merged_m), 0.0)
        new_factor = tl.exp(tile_m-merged_m)
        acc = acc*old_factor + tile_acc*new_factor
        running_l = running_l*old_factor + tile_l*new_factor
        running_m = merged_m
    record = h*SPLITS + split
    tl.store(PM + record, running_m)
    tl.store(PL + record, running_l)
    tl.store(PA + record*D + dims, acc, dims < D)


@triton.jit
def reduce_partials(PM, PL, PA, Out, D: tl.constexpr, BD: tl.constexpr,
                    SPLITS: tl.constexpr, BS: tl.constexpr):
    h = tl.program_id(0)
    split = tl.arange(0, BS)
    dims = tl.arange(0, BD)
    rec = h*SPLITS + split
    m = tl.load(PM + rec, split < SPLITS, -float('inf'))
    l = tl.load(PL + rec, split < SPLITS, 0.0)
    nonempty = l > 0
    max_m = tl.max(tl.where(nonempty, m, -float('inf')), 0)
    max_m = tl.where(tl.sum(nonempty.to(tl.int32), 0) > 0, max_m, 0.0)
    weight = tl.where(nonempty, tl.exp(m - max_m), 0.0)
    a = tl.load(PA + rec[:, None]*D + dims[None, :],
                (split < SPLITS)[:, None] & (dims < D)[None, :], 0.0)
    den = tl.sum(l*weight, 0)
    numerator = tl.sum(a*weight[:, None], 0)
    tl.store(Out + h*D + dims, tl.where(den > 0, numerator/den, 0.0), dims < D)


@triton.jit(do_not_specialize=['start', 'end'])
def refresh_packed(PK, PV, FullK, FullV, Inverse, start, end,
                   kh, kt, vh, vt,
                   N: tl.constexpr, D: tl.constexpr, BD: tl.constexpr):
    kv = tl.program_id(0)
    original = start + tl.program_id(1)
    dims = tl.arange(0, BD)
    row = tl.load(Inverse + kv*N + original, original < end, 0)
    valid = (original < end) & (dims < D)
    k = tl.load(FullK + kv*kh + original*kt + dims, valid, 0)
    v = tl.load(FullV + kv*vh + original*vt + dims, valid, 0)
    tl.store(PK + kv*N*D + row*D + dims, k, valid)
    tl.store(PV + kv*N*D + row*D + dims, v, valid)


@triton.jit(do_not_specialize=['epoch'])
def diagnostics(Lengths, Counts, Stamps, Rows, LogP, Out, epoch,
                T: tl.constexpr, K: tl.constexpr, G: tl.constexpr,
                BT: tl.constexpr, BSEL: tl.constexpr, BG: tl.constexpr):
    """Six numbers/KV head/call, not O(tokens x heads x steps) retained logs."""
    kv = tl.program_id(0)
    teams = tl.arange(0, BT)
    count = tl.load(Counts + kv)
    lens = tl.load(Lengths + kv*T + teams, teams < count, 0)
    stamp = tl.load(Stamps + kv*T + teams, teams < count, 0)
    union = tl.sum(tl.where(stamp == epoch+1, lens, 0), 0)
    gs = tl.arange(0, BG)
    rows = tl.load(Rows + kv*G + gs, gs < G, 0)
    naive = tl.sum(rows, 0)
    slots = tl.arange(0, BSEL)
    selected = tl.minimum(K, count)
    valid = (slots < G*K) & (slots % K < selected)
    lp = tl.load(LogP + kv*G*K + slots, slots < G*K, 0.0)
    p = tl.exp(lp)
    p_sum = tl.sum(tl.where(valid, p, 0.0), 0)
    p_min = tl.min(tl.where(valid, p, float('inf')), 0)
    p_max = tl.max(tl.where(valid, p, 0.0), 0)
    tl.store(Out + kv*6 + 0, naive)
    tl.store(Out + kv*6 + 1, union)
    tl.store(Out + kv*6 + 2, selected*G)
    tl.store(Out + kv*6 + 3, p_sum)
    tl.store(Out + kv*6 + 4, p_min)
    tl.store(Out + kv*6 + 5, p_max)
