"""Untimed P003b validation of the algorithm, not reference tensor equality.

The numerical kernels are unchanged. This CPU/FP64 checker accepts alternate
near-tied winners, but rejects malformed partitions and clearly incorrect
mean-nearest, farthest-first or nearest-leader decisions. Later decisions are
checked against the candidate's OWN preceding leaders, not the reference's.

The score budget is atol + rtol * (4 * max squared key norm in the parent).
It is deliberately scaled by the operands, not the possibly cancelled Gram
result. This is an engineering tolerance, NOT a formal error certificate or an
end-task accuracy test. It never repairs, substitutes or mutates any output.
"""
from __future__ import annotations

import math

import numpy as np

DEFAULT_SCORE_RTOL = 1.0e-4
DEFAULT_SCORE_ATOL = 1.0e-6


def _numpy(tensor, *, floating=False):
    value = tensor.detach().cpu()
    if floating:
        value = value.double()
    return value.numpy()


def _require(condition, message):
    if not condition:
        raise AssertionError("Algorithmic team validation: " + message)


def validate_score_tolerances(rtol, atol):
    if not all(math.isfinite(v) and v >= 0 for v in (rtol, atol)):
        raise ValueError("Score tolerances must be finite and nonnegative.")


def assert_team_algorithm(summary, keys, labels, *, representatives_per_parent=4,
                          configured_parent_count=None, rtol=DEFAULT_SCORE_RTOL,
                          atol=DEFAULT_SCORE_ATOL):
    """Validate every token/parent; no GPU kernel, reference replay or mutation.

    Transfers and checks belong ONLY in the untimed real-workload gate. The
    result records actual score gaps/slack rather than claiming bitwise parity.
    """
    validate_score_tolerances(rtol, atol)
    x = _numpy(keys, floating=True)
    lab = _numpy(labels)
    _require(x.ndim == 2 and len(x) > 0, "keys must be nonempty [N,D]")
    n, d = x.shape
    _require(np.isfinite(x).all(), "nonfinite keys")
    _require(lab.shape == (n,) and np.issubdtype(lab.dtype, np.integer),
             "one integer parent label per token is required")
    _require((lab >= 0).all(), "negative parent label")
    _require(representatives_per_parent > 0, "representative count must be positive")
    inferred = int(lab.max()) + 1
    count = inferred if configured_parent_count is None else int(configured_parent_count)
    _require(count >= inferred, "configured parents do not cover labels")

    names = ("members", "starts", "lengths_long", "leader_token_indices",
             "parent_ids", "parent_lengths_long")
    arrays = {name: _numpy(getattr(summary, name)) for name in names}
    for name, array in arrays.items():
        _require(array.ndim == 1 and np.issubdtype(array.dtype, np.integer),
                 f"{name} must be a one-dimensional integer table")
    members, starts, lengths, leaders, parents, parent_lengths = (
        arrays[name] for name in names)
    teams = len(lengths)
    _require(teams > 0 and len(starts) == len(leaders) == len(parents) == teams,
             "inconsistent team-table shapes")
    _require(len(members) == n and np.array_equal(np.sort(members), np.arange(n)),
             "members must partition every token exactly once")
    _require((lengths > 0).all() and int(lengths.sum()) == n, "empty/incorrect team lengths")
    _require(np.array_equal(starts, np.cumsum(lengths) - lengths), "invalid starts")
    _require(((leaders >= 0) & (leaders < n)).all(), "leader index out of range")
    _require(len(np.unique(leaders)) == teams, "duplicate leader token index")
    _require(np.array_equal(_numpy(summary.lengths_float, floating=True), lengths),
             "float lengths disagree with integer lengths")
    _require(np.array_equal(_numpy(summary.leader_keys, floating=True), x[leaders]),
             "leader keys do not equal their cached token keys")

    active, counts = np.unique(lab, return_counts=True)
    team_counts = np.minimum(counts, representatives_per_parent)
    _require(int(summary.active_parent_count) == len(active), "active-parent count disagrees")
    _require(np.array_equal(parent_lengths, counts), "parent lengths disagree with labels")
    if hasattr(summary, "configured_parent_count"):
        _require(int(summary.configured_parent_count) == count, "configured-parent count disagrees")
    _require(np.array_equal(parents, np.repeat(active, team_counts)),
             "wrong teams per parent or parent order")
    owner = np.empty(n, dtype=np.int64)
    owner[members] = np.repeat(np.arange(teams), lengths)
    _require(np.array_equal(parents[owner], lab), "a token crossed its supplied parent")
    _require(np.array_equal(owner[leaders], np.arange(teams)), "leader does not own its row")
    if n > 1:
        same_team = np.repeat(np.arange(teams), lengths)
        _require(not np.any((members[1:] <= members[:-1]) &
                            (same_team[1:] == same_team[:-1])), "unsorted members within a team")

    # Grouping uses original indices so arbitrary/gapped parent labels work.
    grouped = np.argsort(lab, kind="stable")
    token_offsets = np.concatenate(([0], np.cumsum(counts)))
    team_offsets = np.concatenate(([0], np.cumsum(team_counts)))
    report = {
        "status": "passed", "scope": "all_tokens_all_parents_untimed_cpu_fp64",
        "active_parents": len(active), "tokens": n, "teams": teams,
        "score_rtol": float(rtol), "score_atol": float(atol),
        "score_scale": "4*max_squared_key_norm_within_parent",
        "max_score_gap_over_budget": 0.0,
        "mean_nearest_positive_gap_decisions": 0,
        "farthest_first_positive_gap_decisions": 0,
        "assignment_positive_gap_decisions": 0,
        "max_mean_nearest_score_gap": 0.0,
        "max_farthest_first_score_gap": 0.0,
        "max_assignment_score_gap": 0.0,
        "assignment_squared_distance_sum": 0.0,
        "reference_replay_performed": False,
    }

    def check_gap(gaps, budget, stage, parent):
        gaps = np.maximum(np.asarray(gaps, dtype=np.float64), 0.0)
        gap = float(gaps.max(initial=0.0))
        _require(np.isfinite(gaps).all(), f"nonfinite {stage} score in parent {parent}")
        ratio = gap / budget if budget > 0 else (0.0 if gap == 0 else float("inf"))
        report["max_score_gap_over_budget"] = max(report["max_score_gap_over_budget"], ratio)
        report[f"{stage}_positive_gap_decisions"] += int(np.count_nonzero(gaps > 0))
        report[f"max_{stage}_score_gap"] = max(report[f"max_{stage}_score_gap"], gap)
        _require(gap <= budget, f"{stage} rule violated in parent {parent}: "
                 f"score gap {gap:.9g} > budget {budget:.9g}")

    for i, parent in enumerate(active):
        ids = grouped[token_offsets[i]:token_offsets[i + 1]]
        z = x[ids]
        t0, t1 = team_offsets[i:i + 2]
        local_leaders = np.searchsorted(ids, leaders[t0:t1])
        # Parent/ownership checks above ensure these indices are valid members.
        norms = np.einsum("ij,ij->i", z, z)
        budget = float(atol + rtol * 4.0 * norms.max())
        _require(math.isfinite(budget), f"nonfinite score budget in parent {parent}")
        delta = z - z.mean(axis=0)
        mean_dist = np.einsum("ij,ij->i", delta, delta)
        check_gap(mean_dist[local_leaders[0]] - mean_dist.min(), budget,
                  "mean_nearest", parent)
        nearest = np.full(len(z), np.inf)
        selected = np.zeros(len(z), dtype=bool)
        distances = []
        for position, chosen in enumerate(local_leaders):
            if position:
                check_gap(nearest[~selected].max() - nearest[chosen], budget,
                          "farthest_first", parent)
            delta = z - z[chosen]
            distance = np.einsum("ij,ij->i", delta, delta)
            distances.append(distance)
            nearest = np.minimum(nearest, distance)
            selected[chosen] = True
        distances = np.stack(distances, axis=1)
        assigned = distances[np.arange(len(z)), owner[ids] - t0]
        check_gap(assigned - distances.min(axis=1), budget, "assignment", parent)
        report["assignment_squared_distance_sum"] += float(assigned.sum())
    return report


def reference_difference(actual, expected, keys):
    """Descriptive only: team IDs and leader choices may legitimately differ."""
    from .prefill_failure import describe_team_difference
    result = describe_team_difference(actual, expected)
    x = _numpy(keys, floating=True)
    maps = []
    costs = []
    for summary in (actual, expected):
        members = _numpy(summary.members)
        lengths = _numpy(summary.lengths_long)
        leaders = _numpy(summary.leader_token_indices)
        token_leader = np.empty(len(x), dtype=np.int64)
        token_leader[members] = np.repeat(leaders, lengths)
        maps.append(token_leader)
        delta = x - x[token_leader]
        costs.append(float(np.einsum("ij,ij->", delta, delta)))
    result.update(
        diagnostic_only=True,
        exact_tables_equal=all(field["equal"] for field in result["fields"].values()),
        changed_token_leader_indices=int(np.count_nonzero(maps[0] != maps[1])),
        actual_assignment_squared_distance_sum=costs[0],
        reference_assignment_squared_distance_sum=costs[1],
        assignment_cost_ratio=costs[0] / costs[1] if costs[1] > 0 else None,
    )
    return result


def assert_kmeans_health(estimator, labels, *, tokens, dimensions, clusters):
    """Structural/numerical health, not an identical-partition requirement."""
    lab = _numpy(labels)
    centers = _numpy(estimator.cluster_centers_, floating=True)
    _require(lab.shape == (tokens,) and np.issubdtype(lab.dtype, np.integer),
             "k-means must return one integer label per token")
    _require(((lab >= 0) & (lab < clusters)).all(), "k-means labels out of range")
    _require(centers.shape == (clusters, dimensions) and np.isfinite(centers).all(),
             "k-means centers have invalid shape or nonfinite values")
    _require(math.isfinite(float(estimator.inertia_)) and float(estimator.inertia_) >= 0,
             "k-means inertia is invalid")
    _require(int(estimator.n_steps_) > 0, "k-means did not perform an update")
