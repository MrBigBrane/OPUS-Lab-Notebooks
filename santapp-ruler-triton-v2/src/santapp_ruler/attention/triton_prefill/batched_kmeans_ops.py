"""Batched launchers. Every numerical launch includes all independent fits."""
from __future__ import annotations

import numpy as np
import torch


class BatchedTritonKMeansOps:
    def __init__(self, dot_precision: str = "ieee"):
        if dot_precision not in {"ieee", "tf32x3"}:
            raise ValueError("dot_precision must be ieee or tf32x3")
        try:
            import triton
            from .kernels import p007_batched_kmeans
        except ImportError as exc:
            raise RuntimeError("Triton is required for batched CUDA K-means") from exc
        self.tr = triton
        self.k = p007_batched_kmeans
        self.dot_precision = dot_precision
        self._assignment_scratch: dict[tuple, tuple] = {}
        self._update_scratch: dict[tuple, tuple] = {}
        self.launch_counts: dict[str, int] = {}

    def _launch(self, name, grid, *args, **kwargs):
        self.launch_counts[name] = self.launch_counts.get(name, 0) + 1
        getattr(self.k, name)[grid](*args, num_warps=4,
                                  enable_fp_fusion=False, **kwargs)

    def gather(self, x, ids):
        fits, n, d = x.shape
        m = ids.shape[1]
        y = torch.empty((fits, m, d), dtype=x.dtype, device=x.device)
        self._launch("gather_rows", (self.tr.cdiv(m * d, 256), 1, fits),
                     x, ids, y, n, m, d, 256)
        return y

    def greedy_initialize(self, x, clusters, rngs):
        fits, n, d = x.shape
        trials = 2 + int(np.log(clusters))
        p = np.ones(n, dtype=np.float32)
        p /= p.sum(dtype=np.float32)
        # Each fit owns its RandomState, just as in the old sequential path.
        first_ids = torch.as_tensor(np.array([rng.choice(n, p=p) for rng in rngs]),
                                    dtype=torch.int64, device=x.device)
        centers = torch.empty((fits, clusters, d), dtype=torch.float32, device=x.device)
        closest = torch.empty((fits, n), dtype=torch.float32, device=x.device)
        bd = self.tr.next_power_of_2(max(32, d))
        nb, ns = self.tr.cdiv(n, 64), self.tr.cdiv(n, 512)
        self._launch("first_center", (nb, 1, fits),
                     x, centers, closest, first_ids, n, clusters, d, 64, bd)
        if clusters == 1:
            return centers
        uniforms = torch.as_tensor(
            np.stack([rng.uniform(size=(clusters - 1, trials)) for rng in rngs]),
            dtype=torch.float64, device=x.device,
        )
        prefix = torch.empty_like(closest)
        totals = torch.empty((fits, ns), dtype=torch.float32, device=x.device)
        potential = torch.zeros(fits, dtype=torch.float32, device=x.device)
        candidate_ids = torch.empty((fits, trials), dtype=torch.int64, device=x.device)
        distances = torch.empty((fits, n, trials), dtype=torch.float32, device=x.device)
        partials = torch.empty((fits, nb, trials), dtype=torch.float32, device=x.device)
        # One common center loop, four launches per center across the WHOLE batch.
        # Greedy center i depends on center i-1; independent fit f does not.
        for center_index in range(1, clusters):
            self._launch("scan_closest", (ns, 1, fits),
                         closest, prefix, totals, n, ns, 512)
            self._launch("sample_candidates", (trials, 1, fits),
                         prefix, totals, potential, uniforms, candidate_ids,
                         n, clusters, ns, 512, self.tr.next_power_of_2(ns), trials,
                         center_index)
            self._launch("trial_distances", (nb, 1, fits),
                         x, closest, candidate_ids, distances, partials,
                         n, d, trials, nb, 64, self.tr.next_power_of_2(max(16, trials)),
                         bd, self.dot_precision)
            self._launch("choose_and_commit", (nb, 1, fits),
                         x, centers, closest, candidate_ids, distances, partials, potential,
                         n, clusters, d, trials, nb, self.tr.next_power_of_2(nb),
                         self.tr.next_power_of_2(trials), 64, bd, center_index)
        return centers

    def assign(self, x, centers, active=None):
        fits, n, d = x.shape
        clusters = centers.shape[1]
        bm, bc, br = 32, 32, 64
        nc, nparts = self.tr.cdiv(clusters, bc), self.tr.cdiv(n, br)
        key = (fits, n, clusters, d, x.device)
        if key not in self._assignment_scratch:
            self._assignment_scratch[key] = (
                torch.empty((fits, n, nc), dtype=torch.float32, device=x.device),
                torch.empty((fits, n, nc), dtype=torch.int32, device=x.device),
                torch.empty((fits, nparts), dtype=torch.float32, device=x.device),
            )
        minima, ids, parts = self._assignment_scratch[key]
        if active is None:
            active = torch.ones(fits, dtype=torch.bool, device=x.device)
        labels = torch.empty((fits, n), dtype=torch.int64, device=x.device)
        inertia = torch.empty(fits, dtype=torch.float32, device=x.device)
        self._launch("assignment_partials", (self.tr.cdiv(n, bm), nc, fits),
                     x, centers, minima, ids, active, n, clusters, d, nc, bm, bc,
                     self.tr.next_power_of_2(max(32, d)), self.dot_precision)
        self._launch("finish_assignment", (nparts, 1, fits),
                     minima, ids, labels, parts, active, n, nc,
                     self.tr.next_power_of_2(nc), br, nparts)
        self._launch("sum_scalar", (1, 1, fits),
                     parts, inertia, active, nparts, self.tr.next_power_of_2(nparts))
        return labels, inertia

    def update(self, x, labels, centers, counts, active):
        fits, clusters, d = centers.shape
        n = x.shape[1]
        key = (fits, clusters, d, x.device)
        if key not in self._update_scratch:
            self._update_scratch[key] = (
                torch.empty_like(centers),
                torch.empty((fits, clusters), dtype=torch.int32, device=x.device),
            )
        sums, batch_counts = self._update_scratch[key]
        new_centers, new_counts = torch.empty_like(centers), torch.empty_like(counts)
        self._launch("zero_update", (self.tr.cdiv(clusters * d, 256), 1, fits),
                     sums, batch_counts, clusters, d, 256)
        self._launch("accumulate_batch", (self.tr.cdiv(n, 16), 1, fits),
                     x, labels, sums, batch_counts, active, n, clusters, d,
                     16, self.tr.next_power_of_2(d))
        self._launch("update_centers", (self.tr.cdiv(clusters, 16), 1, fits),
                     centers, counts, sums, batch_counts, new_centers, new_counts,
                     clusters, d, 16, self.tr.next_power_of_2(d))
        return new_centers, new_counts

    def reassign_(self, centers, counts, x_batch, mask, row_ids):
        fits, clusters, d = centers.shape
        ranks = torch.empty((fits, clusters), dtype=torch.int32, device=centers.device)
        floor = torch.empty(fits, dtype=torch.float32, device=centers.device)
        self._launch("reassignment_plan", (1, 1, fits),
                     counts, mask, ranks, floor, clusters,
                     self.tr.next_power_of_2(clusters))
        self._launch("apply_reassignment", (self.tr.cdiv(clusters, 16), 1, fits),
                     centers, counts, x_batch, mask, ranks, row_ids, floor,
                     clusters, d, x_batch.shape[1], row_ids.shape[1],
                     16, self.tr.next_power_of_2(d))
