"""Untimed R007 checks; no repair/replay in production fitting."""
from __future__ import annotations

from types import SimpleNamespace

import torch

from .prefill_algorithmic_validation import assert_kmeans_health


@torch.inference_mode()
def validate_fitted_batch(x, estimator, labels):
    """Check all labels against the original single-fit assignment on fixed centers.

    Full-cost checks use FP64 selected-row distances. A separate FP64 nearest-
    center oracle checks up to 256 spread-out rows/fit, with a bounded [rows,K]
    distance tile, never an [N,K,D] broadcast. This is not partition equality.
    """
    from .prefill_kmeans_ops import TritonKMeansOps
    fits, n, d = x.shape
    k = estimator.cluster_centers_.shape[1]
    reports = []
    for f in range(fits):
        centers = estimator.cluster_centers_[f]
        lab = labels[f]
        assert_kmeans_health(SimpleNamespace(
            cluster_centers_=centers, inertia_=estimator.inertia_[f],
            n_steps_=estimator.n_steps_[f]), lab, tokens=n, dimensions=d, clusters=k)
        assert bool(torch.isfinite(estimator.counts_[f]).all())
        assert bool((estimator.counts_[f] >= 0).all())
        expected, _ = TritonKMeansOps('ieee').assign(x[f], centers)
        costs = (x[f].double() - centers[lab].double()).square().sum(-1)
        ref_cost = (x[f].double() - centers[expected].double()).square().sum(-1)
        allowance = 2e-4 + 2e-5 * ref_cost
        assert bool((costs <= ref_cost + allowance).all()), 'Batched assignment crossed fit/tile boundaries'
        torch.testing.assert_close(torch.tensor(estimator.inertia_[f], dtype=torch.float64,
                                                device=x.device), costs.sum(), rtol=2e-5, atol=2e-4)
        rows = torch.linspace(0, n-1, min(256, n), device=x.device).long().unique()
        points, c64 = x[f, rows].double(), centers.double()
        distances = (points.square().sum(1)[:, None] + c64.square().sum(1)[None, :]
                     - 2.0 * (points @ c64.T)).clamp_min_(0)
        best = distances.min(1).values
        assert bool((costs[rows] <= best + 2e-4 + 2e-5 * best).all()), 'Non-nearest labels in FP64 oracle'
        reports.append({
            'fit': f, 'tokens': n, 'clusters': k, 'n_steps': int(estimator.n_steps_[f]),
            'inertia': float(estimator.inertia_[f]), 'fp64_selected_cost': float(costs.sum()),
            'label_agreement_with_single_assignment': float((expected == lab).float().mean()),
            'fp64_nearest_rows_checked': int(rows.numel()), 'status': 'passed',
        })
    return reports
