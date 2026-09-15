"""R007 independent-fit batching of the unchanged P001 numerical kernels.

Grid axes 0/1 still tile rows/centers. Axis 2 selects an independent fit.
The single-fit JIT bodies are inlined after pointer rebasing; there is NO
Python loop launching one complete fit at a time and no cross-fit reduction.
The greedy center index remains sequential (an algorithm dependency).
"""

import triton as tr
import triton.language as tl

from . import p001_prefill_kmeans as single


@tr.jit
def gather_rows(X, Ids, Y, N: tl.constexpr, M: tl.constexpr, D: tl.constexpr,
                B: tl.constexpr):
    f = tl.program_id(2).to(tl.int64)
    single.gather_rows(X + f * N * D, Ids + f * M, Y + f * M * D, M, D, B)


@tr.jit
def first_center(X, Centers, Closest, FirstIds, N: tl.constexpr, K: tl.constexpr,
                 D: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr):
    f = tl.program_id(2).to(tl.int64)
    first_id = tl.load(FirstIds + f)
    single.first_center(X + f * N * D, Centers + f * K * D, Closest + f * N,
                        N, D, first_id, BN, BD)


@tr.jit
def scan_closest(Closest, Prefix, Totals, N: tl.constexpr, NB: tl.constexpr,
                 B: tl.constexpr):
    f = tl.program_id(2).to(tl.int64)
    single.scan_closest(Closest + f * N, Prefix + f * N, Totals + f * NB, N, B)


@tr.jit(do_not_specialize=["center_index"])
def sample_candidates(Prefix, Totals, Potential, Uniforms, CandidateIds,
                      N: tl.constexpr, K: tl.constexpr, NB: tl.constexpr,
                      SB: tl.constexpr, TB: tl.constexpr, TRIALS: tl.constexpr,
                      center_index):
    f = tl.program_id(2).to(tl.int64)
    single.sample_candidates(Prefix + f * N, Totals + f * NB, Potential + f,
                             Uniforms + f * (K - 1) * TRIALS,
                             CandidateIds + f * TRIALS,
                             N, NB, SB, TB, TRIALS, center_index)


@tr.jit
def trial_distances(X, Closest, CandidateIds, Distances, Partials,
                    N: tl.constexpr, D: tl.constexpr, TRIALS: tl.constexpr,
                    NB: tl.constexpr, BN: tl.constexpr, BT: tl.constexpr,
                    BD: tl.constexpr, PRECISION: tl.constexpr):
    f = tl.program_id(2).to(tl.int64)
    single.trial_distances(X + f * N * D, Closest + f * N,
                           CandidateIds + f * TRIALS, Distances + f * N * TRIALS,
                           Partials + f * NB * TRIALS, N, D, TRIALS,
                           BN, BT, BD, PRECISION)


@tr.jit(do_not_specialize=["center_index"])
def choose_and_commit(X, Centers, Closest, CandidateIds, Distances, Partials, Potential,
                      N: tl.constexpr, K: tl.constexpr, D: tl.constexpr,
                      TRIALS: tl.constexpr, NB: tl.constexpr, BP: tl.constexpr,
                      BT: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr,
                      center_index):
    f = tl.program_id(2).to(tl.int64)
    single.choose_and_commit(X + f * N * D, Centers + f * K * D, Closest + f * N,
                             CandidateIds + f * TRIALS, Distances + f * N * TRIALS,
                             Partials + f * NB * TRIALS, Potential + f,
                             N, D, TRIALS, NB, BP, BT, BN, BD, center_index)


@tr.jit
def assignment_partials(X, Centers, Minima, Ids, Active,
                        N: tl.constexpr, K: tl.constexpr, D: tl.constexpr,
                        NC: tl.constexpr, BM: tl.constexpr, BC: tl.constexpr,
                        BD: tl.constexpr, PRECISION: tl.constexpr):
    f = tl.program_id(2).to(tl.int64)
    if tl.load(Active + f):
        single.assignment_partials(X + f * N * D, Centers + f * K * D,
                                   Minima + f * N * NC, Ids + f * N * NC,
                                   N, K, D, NC, BM, BC, BD, PRECISION)


@tr.jit
def finish_assignment(Minima, Ids, Labels, InertiaParts, Active,
                      N: tl.constexpr, NC: tl.constexpr, BG: tl.constexpr,
                      BM: tl.constexpr, NP: tl.constexpr):
    f = tl.program_id(2).to(tl.int64)
    if tl.load(Active + f):
        single.finish_assignment(Minima + f * N * NC, Ids + f * N * NC,
                                 Labels + f * N, InertiaParts + f * NP,
                                 N, NC, BG, BM)


@tr.jit
def sum_scalar(X, Out, Active, N: tl.constexpr, B: tl.constexpr):
    f = tl.program_id(2).to(tl.int64)
    if tl.load(Active + f):
        single.sum_scalar(X + f * N, Out + f, N, B)
    else:
        tl.store(Out + f, 0.0)


@tr.jit
def zero_update(Sums, BatchCounts, K: tl.constexpr, D: tl.constexpr, B: tl.constexpr):
    f = tl.program_id(2).to(tl.int64)
    single.zero_update(Sums + f * K * D, BatchCounts + f * K, K, D, B)


@tr.jit
def accumulate_batch(X, Labels, Sums, BatchCounts, Active,
                     N: tl.constexpr, K: tl.constexpr, D: tl.constexpr,
                     BM: tl.constexpr, BD: tl.constexpr):
    f = tl.program_id(2).to(tl.int64)
    if tl.load(Active + f):
        single.accumulate_batch(X + f * N * D, Labels + f * N,
                                Sums + f * K * D, BatchCounts + f * K,
                                N, D, BM, BD)


@tr.jit
def update_centers(Centers, Counts, Sums, BatchCounts, NewCenters, NewCounts,
                   K: tl.constexpr, D: tl.constexpr, BC: tl.constexpr,
                   BD: tl.constexpr):
    f = tl.program_id(2).to(tl.int64)
    # Inactive fits have zero batch sums/counts, so P001 copies their old state.
    # This matters: no uninitialized output or changed state after early stop.
    single.update_centers(Centers + f * K * D, Counts + f * K,
                          Sums + f * K * D, BatchCounts + f * K,
                          NewCenters + f * K * D, NewCounts + f * K,
                          K, D, BC, BD)


@tr.jit
def reassignment_plan(Counts, Mask, Ranks, Floor, K: tl.constexpr, B: tl.constexpr):
    f = tl.program_id(2).to(tl.int64)
    single.reassignment_plan(Counts + f * K, Mask + f * K, Ranks + f * K,
                             Floor + f, K, B)


@tr.jit
def apply_reassignment(Centers, Counts, XBatch, Mask, Ranks, RowIds, Floor,
                       K: tl.constexpr, D: tl.constexpr, M: tl.constexpr,
                       ROW_STRIDE: tl.constexpr, BC: tl.constexpr, BD: tl.constexpr):
    f = tl.program_id(2).to(tl.int64)
    single.apply_reassignment(Centers + f * K * D, Counts + f * K,
                              XBatch + f * M * D, Mask + f * K, Ranks + f * K,
                              RowIds + f * ROW_STRIDE, Floor + f,
                              K, D, BC, BD)
