"""Lazy launchers for the P001 MiniBatchKMeans numerical work."""

from __future__ import annotations

import numpy as np
import torch


class TritonKMeansOps:
    def __init__(self, dot_precision: str = "ieee"):
        if dot_precision not in {"ieee", "tf32x3"}:
            raise ValueError("dot_precision must be ieee or the explicit tf32x3 opt-in.")
        try:
            import triton
            from .kernels import p001_prefill_kmeans
        except ImportError as exc:
            raise RuntimeError("Triton is required for accelerated CUDA prefill.") from exc
        self.tr = triton
        self.k = p001_prefill_kmeans
        self.dot_precision = dot_precision
        self._assignment_scratch: dict[tuple, tuple] = {}
        self._update_scratch: dict[tuple, tuple] = {}

    def gather(self, x: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        y = torch.empty((ids.numel(), x.shape[1]), device=x.device, dtype=x.dtype)
        self.k.gather_rows[(self.tr.cdiv(y.numel(), 256),)](
            x, ids, y, ids.numel(), x.shape[1], 256, num_warps=4,
        )
        return y

    def greedy_initialize(self, x: torch.Tensor, clusters: int, rng) -> torch.Tensor:
        n, d = x.shape
        trials = 2 + int(np.log(clusters))
        # Preserve RandomState.choice's float32 probability-vector behavior,
        # including its consumption of the random stream.
        p = np.ones(n, dtype=np.float32)
        p /= p.sum(dtype=np.float32)
        first = int(rng.choice(n, p=p))
        centers = torch.empty((clusters, d), dtype=torch.float32, device=x.device)
        closest = torch.empty(n, dtype=torch.float32, device=x.device)
        bd = self.tr.next_power_of_2(max(32, d))
        self.k.first_center[(self.tr.cdiv(n, 64),)](
            x, centers, closest, n, d, first, 64, bd,
            num_warps=4, enable_fp_fusion=False,
        )
        if clusters == 1:
            return centers
        # One bulk H2D transfer. This consumes exactly the same sequence of
        # uniform draws as (K-1) rng.uniform(size=trials) calls.
        uniforms = torch.as_tensor(
            rng.uniform(size=(clusters - 1, trials)), dtype=torch.float64, device=x.device,
        )
        scan_block, distance_block = 512, 64
        ns = self.tr.cdiv(n, scan_block)
        nb = self.tr.cdiv(n, distance_block)
        prefix = torch.empty_like(closest)
        totals = torch.empty(ns, dtype=torch.float32, device=x.device)
        potential = torch.zeros((), dtype=torch.float32, device=x.device)
        candidate_ids = torch.empty(trials, dtype=torch.int64, device=x.device)
        distances = torch.empty((n, trials), dtype=torch.float32, device=x.device)
        partials = torch.empty((nb, trials), dtype=torch.float32, device=x.device)
        for center_index in range(1, clusters):
            # center_index is a RUN-TIME argument: no distinct JIT per center.
            self.k.scan_closest[(ns,)](
                closest, prefix, totals, n, scan_block, num_warps=4,
            )
            self.k.sample_candidates[(trials,)](
                prefix, totals, potential, uniforms, candidate_ids,
                n, ns, scan_block, self.tr.next_power_of_2(ns), trials, center_index,
                num_warps=4, enable_fp_fusion=False,
            )
            self.k.trial_distances[(nb,)](
                x, closest, candidate_ids, distances, partials, n, d, trials,
                distance_block, self.tr.next_power_of_2(max(16, trials)), bd,
                self.dot_precision, num_warps=4, enable_fp_fusion=False,
            )
            self.k.choose_and_commit[(nb,)](
                x, centers, closest, candidate_ids, distances, partials, potential,
                n, d, trials, nb, self.tr.next_power_of_2(nb),
                self.tr.next_power_of_2(trials), distance_block, bd, center_index,
                num_warps=4, enable_fp_fusion=False,
            )
        return centers

    def assign(self, x: torch.Tensor, centers: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n, d = x.shape
        clusters = centers.shape[0]
        bm, bc, br = 32, 32, 64
        nc = self.tr.cdiv(clusters, bc)
        key = (n, clusters, d, x.device)
        scratch = self._assignment_scratch.get(key)
        if scratch is None:
            scratch = (
                torch.empty((n, nc), dtype=torch.float32, device=x.device),
                torch.empty((n, nc), dtype=torch.int32, device=x.device),
                torch.empty(self.tr.cdiv(n, br), dtype=torch.float32, device=x.device),
            )
            self._assignment_scratch[key] = scratch
        minima, ids, parts = scratch
        labels = torch.empty(n, dtype=torch.int64, device=x.device)
        inertia = torch.empty((), dtype=torch.float32, device=x.device)
        self.k.assignment_partials[(self.tr.cdiv(n, bm), nc)](
            x, centers, minima, ids, n, clusters, d, nc, bm, bc,
            self.tr.next_power_of_2(max(32, d)), self.dot_precision,
            num_warps=4, enable_fp_fusion=False,
        )
        self.k.finish_assignment[(parts.numel(),)](
            minima, ids, labels, parts, n, nc, self.tr.next_power_of_2(nc), br,
            num_warps=4, enable_fp_fusion=False,
        )
        self.k.sum_scalar[(1,)](
            parts, inertia, parts.numel(), self.tr.next_power_of_2(parts.numel()),
            num_warps=4,
        )
        return labels, inertia

    def update(self, x, labels, centers, counts):
        clusters, d = centers.shape
        key = (clusters, d, x.device)
        scratch = self._update_scratch.get(key)
        if scratch is None:
            scratch = (
                torch.empty_like(centers),
                torch.empty(clusters, dtype=torch.int32, device=x.device),
            )
            self._update_scratch[key] = scratch
        sums, batch_counts = scratch
        new_centers, new_counts = torch.empty_like(centers), torch.empty_like(counts)
        self.k.zero_update[(self.tr.cdiv(clusters * d, 256),)](
            sums, batch_counts, clusters, d, 256, num_warps=4,
        )
        self.k.accumulate_batch[(self.tr.cdiv(x.shape[0], 16),)](
            x, labels, sums, batch_counts, x.shape[0], d, 16,
            self.tr.next_power_of_2(d), num_warps=4,
        )
        self.k.update_centers[(self.tr.cdiv(clusters, 16),)](
            centers, counts, sums, batch_counts, new_centers, new_counts,
            clusters, d, 16, self.tr.next_power_of_2(d), num_warps=4,
            enable_fp_fusion=False,
        )
        return new_centers, new_counts

    def reassign_(self, centers, counts, x_batch, mask, row_ids):
        clusters, d = centers.shape
        ranks = torch.empty(clusters, dtype=torch.int32, device=centers.device)
        floor = torch.empty((), dtype=torch.float32, device=centers.device)
        self.k.reassignment_plan[(1,)](
            counts, mask, ranks, floor, clusters, self.tr.next_power_of_2(clusters),
            num_warps=4,
        )
        self.k.apply_reassignment[(self.tr.cdiv(clusters, 16),)](
            centers, counts, x_batch, mask, ranks, row_ids, floor,
            clusters, d, 16, self.tr.next_power_of_2(d), num_warps=4,
        )
