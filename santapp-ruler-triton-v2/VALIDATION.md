# R007 validation status

Package version: 1.5.0. This introduces batched independent-fit K-means. The R006
decode handoff and earlier single-fit GPU results do NOT validate these new GPU
entry points. Existing decoder and numerical P001 files remain byte-identical;
see `docs/R007_SOURCE_PRESERVATION.json`.

## Executed locally

The final counts/environment are recorded in `docs/R007_LOCAL_CHECKS.txt`.
CPU tests cover independent-fit control with a CPU numerical oracle, early-stop
isolation, RNG/chunk semantics, engine dispatch, batch indexing compile-plan
signatures, manifest gates, and result collection. These do not execute Triton.
Python 3.13.5 and torch 2.10.0+cpu were available. No CUDA, Triton compiler, Docker,
PowerShell interpreter, or Kubernetes runtime was available for execution here.
The production Docker image's own CPU tests are the next acceptance check.

## Mandatory target-stack checks

1. Build image and pass its CPU tests.
2. CPU Job: offline SM86 compile of decoder, new batched wrappers and existing prefill.
3. A10 Job: existing decoder CUDA gate, all new batched-fit GPU tests, paired original
   sequential/new batched microbenchmark at 8k and 32k. GPU tests validate primitives,
   fit isolation, native team summaries and selected attention; no reference replay.
4. A10 full-model K-means canary at 32k / nominal 4096 / fit batch eight.
5. All 24 A10 matrix cases (both methods, both contexts, all six budgets), parallelism eight.

The mandatory kernel benchmark provides a same-GPU original sequential baseline;
the optional two-case native fit-batch-one model Job is not required. A smoke pass
is not a full-RULER equivalence study. Floating-point/tie changes can alter fitted
labels and generation without changing the algorithm.

Utilization is an empirical acceptance criterion, not a claimed result. Compare
raw, untrimmed full-generation and fitting windows against >40%; retain process
setup/preparation intervals in the handoff. CUDA memory safety must be established
with the resident model; the kernel-only benchmark cannot establish that.

## Historical R006 authoring status (not R007 evidence)

# R006 validation status

Package version: 1.4.0. This adds a new grouped decoder; predecessor GPU results
are NOT validation of this decoder.

## Executed in the authoring environment

- 232 CPU/static tests passed; 55 GPU-dependent tests skipped (42 existing prefill,
  13 new decode). Final packaging checks are recorded in `docs/R006_LOCAL_CHECKS.txt`.
- Python 3.13.5, PyTorch 2.10.0+cpu; no CUDA device, Triton compiler, Docker daemon,
  or Kubernetes execution was available here. This is not the production image.
- Python 3.11 syntax parsing, manifest checks, and nine-document/24-index deployment
  validation are included in the final local packaging check.

The CPU tests cover packing, inverse permutations, frozen leaders, prompt-row
refresh, all-selected behavior, nominal-to-team budget conversion, aggregated
traffic accounting, backend selection, tiny-model integration with a CPU test
stand-in, source preservation, and the source-bound deployment gates. The CPU
stand-in is NOT Triton execution and is never enabled in production.

## Required acceptance on the actual stack

Run the supplied stages in order: offline SM86/SM89 compilation in a CPU-only
container; real-CUDA kernel correctness; an 8k/S4096 model canary; then the 24-case
model sweep. A failed gate prevents later grouped model stages from loading the
model. Gates are bound to the exact Python source digest. Do not bypass them.

GPU checks compare discrete selection, inclusion probabilities, compact prefixes,
cache refresh, original-cache attention outputs, suffixes 0/1/17, variable team
lengths, all-selected cases, empty attention splits, RNG reproducibility, and
native prefill summaries. Default kernel matrix: both contexts and all six nominal
budgets in FP16; a separate GPU test covers BF16. Compilation is not execution.

No GPU utilization, throughput, 24 GB peak-memory fit, full-model numerical result,
Triton compilation, Docker build, or cluster deployment has been measured here.
The included smoke uses one NIAH prompt and 128 forced decode steps; it is not an
official RULER accuracy evaluation. Do not interpret the passing CPU suite as a
GPU correctness or speed claim.

R005's historical report is retained in `docs/VALIDATION_R005.md`.
