"""Algorithm-preserving, opt-in Triton MiniBatchKMeans.

This is NOT Lloyd k-means. NumPy RandomState and the original Python convergence
controller are retained; initialization, assignment, online updates and center
replacement are Triton numerical kernels. No bitwise partition guarantee is
made across different GPU reduction orders. See docs/TRITON_PREFILL.md.
"""

from __future__ import annotations

import math
from contextlib import nullcontext

import numpy as np
import torch

from ..minibatch_kmeans import SklearnLikeTorchMiniBatchKMeans


def _make_ops(x: torch.Tensor, precision: str):
    if x.device.type != "cuda":
        raise ValueError("TritonMiniBatchKMeans requires CUDA keys; use the PyTorch reference on CPU.")
    from .prefill_kmeans_ops import TritonKMeansOps
    return TritonKMeansOps(precision)


class TritonMiniBatchKMeans(SklearnLikeTorchMiniBatchKMeans):
    """Same constructor/fit_predict contract as the supplied FP32 reference.

    Additional keyword: dot_precision='ieee' (default) or explicit 'tf32x3'.
    Float16/bfloat16 are promoted to float32 as in the reference. Float64 is
    rejected rather than silently demoted. Feature dimensions 1..256 supported.
    Instances are not safe for concurrent fits on multiple streams/threads.
    """

    def __init__(self, n_clusters, *, batch_size=4096, n_init=1, max_iter=100,
                 tol=0.0, max_no_improvement=10, init_size=None,
                 reassignment_ratio=0.01, random_state=0, verbose=False,
                 dot_precision="ieee"):
        super().__init__(
            n_clusters, batch_size=batch_size, n_init=n_init, max_iter=max_iter,
            tol=tol, max_no_improvement=max_no_improvement, init_size=init_size,
            reassignment_ratio=reassignment_ratio, random_state=random_state, verbose=verbose,
        )
        if min(self.n_clusters, self.batch_size, self.n_init, self.max_iter) <= 0:
            raise ValueError("n_clusters, batch_size, n_init and max_iter must be positive.")
        if not math.isfinite(self.tol) or self.tol < 0:
            raise ValueError("tol must be finite and nonnegative.")
        if self.init_size is not None and int(self.init_size) <= 0:
            raise ValueError("init_size must be positive or None.")
        if self.max_no_improvement is not None and self.max_no_improvement <= 0:
            raise ValueError("max_no_improvement must be positive or None.")
        if not 0 <= self.reassignment_ratio <= 1:
            raise ValueError("reassignment_ratio must be in [0, 1].")
        if dot_precision not in {"ieee", "tf32x3"}:
            raise ValueError("dot_precision must be ieee or tf32x3 (no implicit TF32 truncation).")
        self.dot_precision = dot_precision
        self._ops = None

    def _assign(self, x, centers):
        return self._ops.assign(x, centers)

    def _greedy_kmeans_plus_plus(self, x, rng):
        return self._ops.greedy_initialize(x, self.n_clusters, rng)

    @torch.inference_mode()
    def fit_predict(self, x):
        if x.ndim != 2:
            raise ValueError(f"Expected [samples, features], got {tuple(x.shape)}")
        if x.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
            raise TypeError("The accelerated path supports float16/bfloat16/float32, not float64.")
        n_samples, features = x.shape
        if not 1 <= features <= 256:
            raise ValueError("The accelerated path supports feature dimensions 1..256.")
        if n_samples < self.n_clusters:
            raise ValueError(f"n_samples={n_samples} must be >= n_clusters={self.n_clusters}")
        if n_samples >= 2**31 or self.n_clusters > 65536:
            raise ValueError("P001 supports fewer than 2^31 samples and at most 65536 clusters.")
        # Kernels use 32-bit tile offsets for dense scratch. Reject overflow
        # before allocating or launching (normal 8K/32K workloads are far below).
        center_tiles = (self.n_clusters + 31) // 32
        if n_samples * max(features, center_tiles) >= 2**31:
            raise ValueError("P001 key/assignment scratch exceeds 32-bit indexing capacity.")
        # This is one prefill-time validation sync, not a sync per parent/center.
        if not bool(torch.isfinite(x).all()):
            raise ValueError("Keys must be finite.")
        x = x.float().contiguous()
        self._ops = _make_ops(x, self.dot_precision)
        guard = torch.cuda.device(x.device) if x.device.type == "cuda" else nullcontext()
        with guard:
            try:
                return self._fit_prepared(x)
            finally:
                self._ops = None

    def _fit_prepared(self, x):
        n_samples = x.shape[0]
        batch_size = min(self.batch_size, n_samples)
        init_size = self.init_size
        if init_size is None:
            init_size = 3 * batch_size
            if init_size < self.n_clusters:
                init_size = 3 * self.n_clusters
        elif init_size < self.n_clusters:
            init_size = 3 * self.n_clusters
        init_size = min(int(init_size), n_samples)
        rng = np.random.RandomState(self.random_state)

        # Draw even with n_init=1. Moving/removing this RNG consumption changes
        # initialization and every subsequent minibatch.
        validation_ids = torch.as_tensor(
            rng.randint(0, n_samples, init_size), dtype=torch.long, device=x.device,
        )
        x_valid = self._ops.gather(x, validation_ids) if self.n_init > 1 else None
        best_centers, best_validation_inertia = None, None
        for _ in range(self.n_init):
            if init_size < n_samples:
                init_ids = torch.as_tensor(
                    rng.randint(0, n_samples, init_size), dtype=torch.long, device=x.device,
                )
                x_init = self._ops.gather(x, init_ids)
            else:
                x_init = x
            centers = self._greedy_kmeans_plus_plus(x_init, rng)
            if self.n_init == 1:
                best_centers = centers
            else:
                _, validation_inertia = self._assign(x_valid, centers)
                validation_inertia = float(validation_inertia.item())
                if best_validation_inertia is None or validation_inertia < best_validation_inertia:
                    best_validation_inertia = validation_inertia
                    best_centers = centers.clone()

        centers = best_centers
        self.initial_centers_ = centers.clone()
        counts = torch.zeros(self.n_clusters, dtype=x.dtype, device=x.device)
        scaled_tol = float(x.var(dim=0, unbiased=False).mean().item()) * self.tol if self.tol > 0 else 0.0
        n_steps = (self.max_iter * n_samples) // batch_size
        ewa_inertia = ewa_inertia_min = None
        no_improvement = n_since_last_reassign = 0
        self.reassignment_events_ = []
        self.stopping_reason_ = "max_iter"
        for step in range(n_steps):
            n_since_last_reassign += batch_size
            random_reassign = bool(
                (counts == 0).any().item() or n_since_last_reassign >= 10 * self.n_clusters
            )
            if random_reassign:
                n_since_last_reassign = 0
            batch_ids = torch.as_tensor(
                rng.randint(0, n_samples, batch_size), dtype=torch.long, device=x.device,
            )
            x_batch = self._ops.gather(x, batch_ids)
            labels, batch_inertia = self._assign(x_batch, centers)
            centers_new, counts = self._ops.update(x_batch, labels, centers, counts)
            if random_reassign and self.reassignment_ratio > 0:
                # Tiny O(K) control operations deliberately retain the supplied
                # reference's argsort convention in the half-batch cap case.
                to_reassign = counts < self.reassignment_ratio * counts.max()
                n_reassign = int(to_reassign.sum().item())
                if n_reassign > 0.5 * batch_size:
                    keep_ids = torch.argsort(counts)[int(0.5 * batch_size):]
                    to_reassign[keep_ids] = False
                    n_reassign = int(to_reassign.sum().item())
                if n_reassign:
                    rows_np = rng.choice(batch_size, replace=False, size=n_reassign)
                    rows = torch.as_tensor(rows_np, dtype=torch.long, device=x.device)
                    self._ops.reassign_(centers_new, counts, x_batch, to_reassign, rows)
                    self.reassignment_events_.append({"step": step, "centers": n_reassign})
            centers_squared_diff = (
                float((centers_new - centers).square().sum().item()) if scaled_tol > 0 else 0.0
            )
            centers = centers_new
            if step == 0:
                continue
            mean_batch_inertia = float(batch_inertia.item()) / batch_size
            if ewa_inertia is None:
                ewa_inertia = mean_batch_inertia
            else:
                alpha = min(batch_size * 2.0 / (n_samples + 1), 1.0)
                ewa_inertia = ewa_inertia * (1.0 - alpha) + mean_batch_inertia * alpha
            if scaled_tol > 0 and centers_squared_diff <= scaled_tol:
                self.stopping_reason_ = "tol"
                break
            if ewa_inertia_min is None or ewa_inertia < ewa_inertia_min:
                ewa_inertia_min = ewa_inertia
                no_improvement = 0
            else:
                no_improvement += 1
            if self.max_no_improvement is not None and no_improvement >= self.max_no_improvement:
                self.stopping_reason_ = "max_no_improvement"
                break
        self.cluster_centers_, self.counts_ = centers, counts
        self.n_steps_ = step + 1
        self.n_iter_ = int(np.ceil(self.n_steps_ * batch_size / n_samples))
        labels, inertia = self._assign(x, centers)
        self.labels_, self.inertia_ = labels, float(inertia.item())
        # Scratch belongs to this fit only. Do not retain N-by-ceil(K/32) memory
        # for every layer/head when a caller retains fitted estimators.
        if self.verbose:
            print(f"Triton MiniBatchKMeans: {self.n_steps_} mini-batches, "
                  f"{self.n_iter_} effective epochs ({self.dot_precision})")
        return labels
