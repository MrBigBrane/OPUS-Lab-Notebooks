"""Independent-fit batched MiniBatchKMeans, not a Lloyd/replay replacement.

Numerical state has a leading fit dimension. Each fit retains its own NumPy
RandomState, online counts, reassignment schedule, EWA and stopping decision.
Small batched host control readbacks remain (at most two per minibatch round),
but initialization has no per-center readback and every numerical kernel covers
all fits in the chunk. Stopped fits are masked, not trained until the slowest fit.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np
import torch

from .prefill_kmeans import TritonMiniBatchKMeans


def _make_batched_ops(x, precision):
    if x.device.type != "cuda":
        raise ValueError("BatchedTritonMiniBatchKMeans requires CUDA; no CPU fallback")
    from .batched_kmeans_ops import BatchedTritonKMeansOps
    return BatchedTritonKMeansOps(precision)


@dataclass(frozen=True)
class FitChunk:
    first: int
    stop: int

    @property
    def size(self):
        return self.stop - self.first


def fit_chunks(num_fits: int, fit_batch_size: int) -> list[FitChunk]:
    if isinstance(fit_batch_size, bool) or not isinstance(fit_batch_size, int):
        raise ValueError("fit_batch_size must be a positive integer")
    if num_fits < 1 or not 1 <= fit_batch_size <= 65535:
        raise ValueError("num_fits must be positive; fit_batch_size must be in 1..65535")
    return [FitChunk(i, min(i + fit_batch_size, num_fits))
            for i in range(0, num_fits, fit_batch_size)]


def reassignment_mask(counts, flags, batch_size, ratio):
    """Same low-count rule and floor(batch_size/2) cap, independently per fit."""
    mask = (counts < ratio * counts.amax(dim=1, keepdim=True)) & flags[:, None]
    half = batch_size // 2
    if counts.shape[1] > half:
        over_cap = mask.sum(dim=1) > half
        order = torch.argsort(counts, dim=1)
        ranks = torch.empty_like(order)
        ranks.scatter_(1, order, torch.arange(counts.shape[1], device=counts.device)
                       .expand_as(order))
        mask &= (~over_cap[:, None]) | (ranks < half)
    return mask


class BatchedTritonMiniBatchKMeans(TritonMiniBatchKMeans):
    """fit_predict([B,N,D]) -> labels[B,N], independent greedy minibatch fits.

    Uses the same configured seed separately for every fit as R006. Numerical
    reductions and tie behavior need not be bitwise equal across GPU scheduling.
    n_steps_, n_iter_, inertia_, stopping_reason_, and reassignment_events_ have
    one entry per fit; numerical output tensors also keep their fit dimension.
    """

    @torch.inference_mode()
    def fit_predict(self, x):
        if x.ndim != 3:
            raise ValueError(f"Expected [fits,samples,features], got {tuple(x.shape)}")
        if x.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
            raise TypeError("Batched K-means supports float16/bfloat16/float32, not float64")
        fits, n, d = x.shape
        if not 1 <= fits <= 65535 or not 1 <= d <= 256:
            raise ValueError("Batched K-means requires 1..65535 fits and D in 1..256")
        if n < self.n_clusters or n >= 2**31 or self.n_clusters > 65536:
            raise ValueError("Require n_clusters <= samples < 2^31 and clusters <= 65536")
        if n * max(d, (self.n_clusters + 31) // 32) >= 2**31:
            raise ValueError("Per-fit key/assignment scratch exceeds 32-bit indexing capacity")
        if not bool(torch.isfinite(x).all()):
            raise ValueError("Keys must be finite")
        x = x.float().contiguous()
        self._ops = _make_batched_ops(x, self.dot_precision)
        guard = torch.cuda.device(x.device) if x.device.type == "cuda" else nullcontext()
        with guard:
            try:
                return self._fit_prepared_batch(x)
            finally:
                # Scratch must not remain attached to each of 112 fitted heads.
                self.launch_counts_ = dict(getattr(self._ops, "launch_counts", {}))
                self._ops = None

    def _fit_prepared_batch(self, x):
        fits, n, _ = x.shape
        k = self.n_clusters
        batch = min(self.batch_size, n)
        init = self.init_size
        if init is None:
            init = 3 * batch
            if init < k:
                init = 3 * k
        elif init < k:
            init = 3 * k
        init = min(int(init), n)
        rngs = [np.random.RandomState(self.random_state) for _ in range(fits)]

        def on_device(value, dtype=None):
            return torch.as_tensor(value, dtype=dtype, device=x.device)

        validation_ids = on_device(np.stack([r.randint(0, n, init) for r in rngs]),
                                   torch.int64)
        x_valid = self._ops.gather(x, validation_ids) if self.n_init > 1 else None
        best_centers = best_inertia = None
        for _ in range(self.n_init):
            if init < n:
                init_ids = on_device(np.stack([r.randint(0, n, init) for r in rngs]),
                                     torch.int64)
                x_init = self._ops.gather(x, init_ids)
            else:
                x_init = x
            centers = self._ops.greedy_initialize(x_init, k, rngs)
            if self.n_init == 1:
                best_centers = centers
            else:
                _, inertia = self._ops.assign(x_valid, centers)
                if best_centers is None:
                    best_centers, best_inertia = centers.clone(), inertia
                else:
                    better = inertia < best_inertia
                    best_centers = torch.where(better[:, None, None], centers, best_centers)
                    best_inertia = torch.where(better, inertia, best_inertia)
        del x_init, x_valid, validation_ids
        centers = best_centers
        self.initial_centers_ = centers.clone()
        counts = torch.zeros((fits, k), dtype=x.dtype, device=x.device)
        scaled_tol = (x.var(dim=1, unbiased=False).mean(dim=1).cpu().numpy()
                      .astype(np.float64) * self.tol) if self.tol > 0 else np.zeros(fits)
        n_steps = (self.max_iter * n) // batch
        active = np.ones(fits, dtype=bool)
        active_device = on_device(active.copy(), torch.bool)
        empty_counts = np.ones(fits, dtype=bool)
        since_reassign = np.zeros(fits, dtype=np.int64)
        ewa = np.full(fits, np.nan)
        ewa_min = np.full(fits, np.nan)
        no_improvement = np.zeros(fits, dtype=np.int64)
        self.n_steps_ = np.zeros(fits, dtype=np.int64)
        self.reassignment_events_ = [[] for _ in range(fits)]
        self.stopping_reason_ = ["max_iter"] * fits
        self.active_fits_per_step_ = []
        self.minibatch_host_control_transfers_ = 0
        alpha = min(batch * 2.0 / (n + 1), 1.0)
        for step in range(n_steps):
            self.active_fits_per_step_.append(int(active.sum()))
            since_reassign[active] += batch
            random_reassign = active & (empty_counts | (since_reassign >= 10 * k))
            since_reassign[random_reassign] = 0
            # Only host RNG bookkeeping loops over fits; GPU work never does.
            ids = np.zeros((fits, batch), dtype=np.int64)
            for f in np.flatnonzero(active):
                ids[f] = rngs[f].randint(0, n, batch)
            x_batch = self._ops.gather(x, on_device(ids, torch.int64))
            labels, batch_inertia = self._ops.assign(x_batch, centers, active_device)
            centers_new, counts = self._ops.update(x_batch, labels, centers, counts,
                                                   active_device)
            if self.reassignment_ratio > 0 and random_reassign.any():
                mask = reassignment_mask(counts, on_device(random_reassign, torch.bool),
                                         batch, self.reassignment_ratio)
                n_reassign = mask.sum(dim=1).cpu().numpy()
                self.minibatch_host_control_transfers_ += 1
                row_stride = int(n_reassign.max())
                if row_stride:
                    # Fixed stride avoids one JIT specialization per reassignment count.
                    rows = np.zeros((fits, min(k, batch // 2)), dtype=np.int64)
                    for f in np.flatnonzero(n_reassign):
                        size = int(n_reassign[f])
                        rows[f, :size] = rngs[f].choice(batch, replace=False, size=size)
                        self.reassignment_events_[f].append({"step": step, "centers": size})
                    self._ops.reassign_(centers_new, counts, x_batch, mask,
                                         on_device(rows, torch.int64))
            difference = ((centers_new - centers).square().sum(dim=(1, 2))
                          if np.any(scaled_tol > 0) else torch.zeros_like(batch_inertia))
            # One compact [B,3] control readback, not .item() per head/center.
            stats = torch.stack((batch_inertia, difference,
                                 (counts == 0).any(dim=1).to(x.dtype)), dim=1).cpu().numpy()
            self.minibatch_host_control_transfers_ += 1
            empty_counts = stats[:, 2].astype(bool)
            centers = centers_new
            self.n_steps_[active] = step + 1
            if step == 0:
                continue
            indices = np.flatnonzero(active)
            mean_inertia = stats[:, 0].astype(np.float64) / batch
            unset = np.isnan(ewa)
            ewa[active & unset] = mean_inertia[active & unset]
            old = active & ~unset
            ewa[old] = ewa[old] * (1.0 - alpha) + mean_inertia[old] * alpha
            for f in indices:
                if scaled_tol[f] > 0 and stats[f, 1] <= scaled_tol[f]:
                    self.stopping_reason_[f] = "tol"
                    active[f] = False
                    continue
                if np.isnan(ewa_min[f]) or ewa[f] < ewa_min[f]:
                    ewa_min[f] = ewa[f]
                    no_improvement[f] = 0
                else:
                    no_improvement[f] += 1
                if (self.max_no_improvement is not None and
                        no_improvement[f] >= self.max_no_improvement):
                    self.stopping_reason_[f] = "max_no_improvement"
                    active[f] = False
            if not active.any():
                break
            if len(indices) != int(active.sum()):
                active_device = on_device(active.copy(), torch.bool)
        self.cluster_centers_, self.counts_ = centers, counts
        self.n_iter_ = np.ceil(self.n_steps_ * batch / n).astype(np.int64)
        # ALL fits must get final labels, including fits stopped early.
        labels, inertia = self._ops.assign(x, centers)
        self.labels_, self.inertia_ = labels, inertia.cpu().numpy().astype(np.float64)
        if self.verbose:
            print(f"Batched Triton MiniBatchKMeans: fits={fits} "
                  f"steps={self.n_steps_.tolist()} precision={self.dot_precision}", flush=True)
        return labels
