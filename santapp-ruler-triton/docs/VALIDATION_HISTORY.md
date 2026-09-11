# Validation history and evidence boundaries

## Source lineage

The colleague base is the supplied dual-parent repository. R004 vendored native
prefill implementations from `SANTA-Triton-main (21).zip`. R005 is a packaging,
defaults, test-scope, and deployment update on R004, not the R004a small-D hotfix.
`TRITON_PREFILL_PROVENANCE.json` retains source hashes and unchanged decode-method
hashes. Kernel sources, k-means launcher/controller, and native team builder are
unchanged from the R004 archive used for the successful full-model smoke.

## Recipient-side evidence from 2026-09-10 (predecessor R004)

| Test | Observed result | Scope |
|---|---|---|
| Cluster Docker build | Passed; Torch 2.11.0+cu126 / Triton 3.6.0 | Build and dependency checks, not GPU execution |
| Cluster-stack sm80/sm86 cross-compilation | Passed | 8k/32k D=128, all numerical entry points; not hardware execution |
| Local 5090 Docker regression | 161 CPU + 40 GPU passed | CUDA 12.8 local profile |
| Local synthetic matrix | Passed | 8k/32k, FP16/BF16, 10 S budgets, both policies, exact suffix 0/1 |
| Local real-Qwen smoke | Passed | Both contexts/policies, S128/S512, 2 generated tokens |
| First A10 GPU regression | 39 passed, 1 failed | Synthetic D=7 assignment; known issue retained, not relabeled as passed |
| First A10 full-model smoke | 8k cases completed; 32k CUDA OOM shown | Failure in dense MLP, before Triton preprocessing |
| A10 allocator-config rerun | **8/8 completions reported by maintainer** | R004 image, A10-only, 4 CPUs/32Gi host RAM/one GPU, expandable segments; 8k/32k x S128/S512 x both policies; one NIAH prompt each, 2 output tokens |

The successful A10 completion count is from the maintainer's message and supplied
working YAML; no new full A10 handoff was attached for independent per-row
inspection. Do not represent this as a full accuracy study, a multi-prompt soak,
a 3090 result, or GPU execution of the newly packaged R005 image.

The two-token NIAH output can score zero because it is incomplete. Its purpose
was environment and integration execution, not a reportable task score.

## R005 release-local validation

See `../VALIDATION.md`. CPU/static tests and packaging checks are rerun during
release preparation. GPU tests are explicitly skipped in the release preparation
environment. Rebuild/push a distinct R005 image and rerun the staged smoke before
rolling it out to colleagues. Keep the R004 image/results for comparison.
