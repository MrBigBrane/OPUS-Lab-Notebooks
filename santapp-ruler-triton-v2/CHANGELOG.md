# Changelog

## R007 / 1.5.0 — independent-fit batched K-means

- Native GPU fitting batches eight independent layer/KV-head problems per numerical launch.
- Reuses unchanged P001 numerical routines with fit-axis pointer rebasing; preserves per-fit RNG, counts, reassignment, and independent early stopping. No Lloyd replacement or reference repair.
- Limits staging/scratch to one chunk; preserves existing team builders, grouped decoder, all budgets, and contiguous prefill.
- Adds true-CUDA primitive/fit/native-summary tests, SM86 offline compilation, and paired same-A10 sequential/batched fitting telemetry.
- Adds A10-only manifests, a 32k/S4096 K-means canary, 24 model cases with parallelism eight, and optional native single-fit control.
- Adds full-generation and fit/team phase telemetry and a matrix acceptance summary. Transfer Pod now has one CPU; results-only packaging is uncompressed/progress-reporting by default.
- Local CPU/static validation only; new GPU acceptance and utilization remain to be measured.

## R006 / 1.4.0 — grouped decode utilization backend

- Added all-head CUDA route/top-(K+1)/compact-plan/split-KV decode for K=1..1024;
  smoke nominal budgets 128 through 4096 at P16/R4.
- Kept both parent constructors and reference estimators; separate native packed
  K/V refreshes the re-fed prompt row without changing frozen leaders.
- Moved head-level accounting to bounded GPU diagnostics and deferred readback.
- Added source-bound offline compile and real-GPU correctness gates, 24 indexed
  model cases, matched Torch reference, and decode-window utilization telemetry.
- Kept legacy configs without the new field on Torch; bundled current configs
  explicitly select grouped decode. No new GPU success/performance claim.


## 1.3.0 / R005 — 2026-09-10

Native Triton preparation becomes the default for both existing team backends.
The original Torch implementation remains explicitly selectable, with no hidden
fallback. The four backend names, benchmark protocol and PyTorch decoder remain.

The A10 smoke findings are incorporated into runtime/Docker/Kubernetes defaults:
expandable segments, equal CPU/RAM requests and limits, init-container resources,
and writable pod-local JIT scratch. Default model-worker parallelism is four.
Resource quantities are tunable without violating the request/limit ratio.

The old failed A10 D=7 k-means assignment fixture is retained as an isolated,
platform/version-specific expected-failure diagnostic. Production D=128 primitive
and multiple-tile tests remain mandatory. The R004a small-D workaround is not
included. Numerical prefill kernels/host controller/team builder are R004-identical.

The manifest renderer retains its original four-resource mode and adds a staged
kernel/prefetch/model smoke mode. Configs, README and deployment notes are updated
for colleague handoff, with generic identifiers. Old implicit-Torch saved run
configs retain their historical identity to prevent mixed resumes.

Package version and image tag are advanced; build/push a new image. Predecessor
R004 GPU evidence is recorded separately from release-local CPU/package checks.
See `VALIDATION.md`, `docs/VALIDATION_HISTORY.md`, and `docs/KNOWN_ISSUES.md`.
