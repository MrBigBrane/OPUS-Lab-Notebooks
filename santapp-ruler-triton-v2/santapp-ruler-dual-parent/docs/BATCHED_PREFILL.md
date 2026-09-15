# R007: independent-fit batched MiniBatchKMeans

R007 / package 1.5.0 changes **the scheduling of independent K-means fits**, not
SANTA's team sampler, nominal budgets, or the grouped decoder. The default
native `santapp` path now fits eight `(layer, KV-head)` problems per GPU batch.
For the pinned 28-layer/four-KV-head Qwen model, 112 fits become 14 batched calls.
The `hierarchical` contiguous path and all R006 grouped decode kernels are unchanged.

## Two different batch sizes

```yaml
santapp:
  prefill_backend: triton
  decode_backend: grouped_triton
  kmeans:
    fit_batch_size: 8  # independent clustering problems per GPU batch
    batch_size: 4096  # sampled tokens in EACH fit's minibatch
```

`--kmeans-fit-batch-size 8` overrides the first value. `--prefill-backend torch`
selects the old explicit reference; there is no automatic CPU/Torch fallback.
The fitting batch is unrelated to Kubernetes matrix parallelism. Eight workers
can each run their own model on one A10, with eight fits per worker/GPU batch.
Changing the fitting batch does not change requested teams or nominal sample budgets.

## Numerical execution

`p007_batched_kmeans.py` adds a third launch-grid axis to the existing numerical
P001 routines. Axes zero and one retain row/center tiling; axis two identifies the
fit and rebases all pointers into its private state. Those unchanged single-fit
JIT bodies are inlined, **not launched once from Python per fit**. All numerical
primitives cover the batch: gathering, K-means++ distance scans/candidate search,
trial scoring/commit, nearest-center assignment, cumulative online update,
reassignment, and final labels/inertia.

Greedy center selection is still sequential over center index because that
algorithm has a dependency. There is now one such loop for a batch, not one full
loop per fit. For K parents, one initialization uses `1 + 4*(K-1)` Triton launches
for the whole batch. Independent fits do not exchange centers, labels, counts,
statistics, or random samples. Assignment scratch is tiled, not a full N-by-K
pairwise distance tensor and never N-by-K-by-D.

## Algorithmic fidelity

Each fit retains its own NumPy RandomState initialized with the same configured
seed as the old independent fit. Validation/init subset draws, greedy K-means++,
replacement minibatch draws, `n_init` selection, cumulative counts and means,
low-count reassignment and its half-minibatch cap, EWA/no-improvement stopping,
and tolerance stopping remain independent. A fit that stops is masked and frozen;
it is **not forced to keep training until the last fit stops**.

FP32 atomics, reduction scheduling, and ties do not promise bitwise fitted labels
or identical generated text. This is not an algorithm switch to Lloyd, balanced
K-means, normalized keys, reduced cluster count, shared centers, or reference
replay. Original post-RoPE cached key coordinates are copied without normalization.
The native team builder and original whole-team conditional inclusion estimator
are preserved. Exact final assignment is checked with fixed fitted centers;
cluster-label numbering itself is not an accuracy metric.

## Bounded memory and remaining host work

Only one chunk of FP32 keys is materialized. Eight 32768-by-128 key matrices use
128 MiB; that is **input staging only, not total clustering scratch or model peak**.
Assignment scratch, labels, centers, initialization workspaces and allocator state
also consume memory. Fit scratch is released after each fit call, and the staged
chunk is released after its teams are constructed, before grouped decode-cache
packing. No attempt is made to fit all 112 heads concurrently on a 24 GB device.

Small host loops still advance independent RNG/control state. Each common
minibatch iteration has at most two compact batched control readbacks rather than
head-by-head scalar readbacks. There is no per-center D2H readback. Copying cached
heads into a chunk and building the existing team tables afterward still uses
per-fit orchestration. Decode packing/preparation is not refactored in this release.
These are residual costs, not claims of complete GPU residency.

Eight fits is an explicit default, not a measured optimal or guaranteed-safe A10
setting. The strict 32k/S4096 K-means model canary must pass before the matrix.
No hidden fitting-batch reduction, budget reduction, or backend fallback occurs
on OOM. New config identity prevents resuming an R006 native run as if it used
batched fitting.

## Tests and measurement

CPU oracles exercise independent RNG/control, divergent convergence, `n_init`,
reassignment, chunk boundaries, and engine dispatch. These are not CUDA execution.
GPU tests exercise batched pointer isolation, inactive fits, exact primitives,
FP16/BF16-origin inputs at D128, prototype clustering, native team construction,
and grouped decode with budgets through 1024 teams. Fixed-center checks compare
to the original single-fit Triton primitive and sampled FP64 nearest-center costs.

The GPU gate additionally times original sequential fits versus the new batch on
the **same A10**, at both contexts, with alternating order and explicit timing
windows. It uses real-shaped synthetic keys and no resident model; it does not
prove model memory or task accuracy. The later full-model matrix performs the
integration, utilization, and OOM check using one real NIAH prompt per case.

`phase-utilization.json` retains all raw boundary samples and separates full
`generation`, dense prefill, parent build, K-means fits, team construction, decode
preparation, decode, and monitored process lifetime. Short unsampled intervals
are unknown, not zero. Multiple device UUIDs are not pooled into a misleading
single-device mean. `MATRIX_ACCEPTANCE.json` reports software and >40% generation
utilization separately. These samples do not reproduce Nautilus's warning window.
One-prompt NIAH success is not full-RULER statistical-equivalence evidence.

See `NAUTILUS_BATCHED_PREFILL.md` and `VALIDATION.md` for the rollout gates.
