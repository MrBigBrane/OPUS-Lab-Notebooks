# Native Triton preparation: R005

## Integration contract

`default.yaml`, `smoke.yaml`, no-file CLI defaults, and Kubernetes renderer/shard
runner defaults select native Triton preparation for `santapp` and `hierarchical`.
Existing command shapes and backend names do not change. `configs/triton.yaml`
is an explicit alias of the default. `configs/torch.yaml` and
`--prefill-backend torch` select the original Torch numerical implementations.
Unavailable CUDA/Triton causes a clear failure, not silent slow fallback.

Only preparation is accelerated. The original PyTorch attention patch, cache
layout, post-RoPE keys, team sampler, Gumbel Top-K, inclusion correction, generated
suffix, task outputs and grading are retained. No optimized decode kernels or
additional packed-cache stage from the Triton development repo are imported.
No bitwise-output parity or PyTorch replay is required. The validation checks
partition integrity and algorithmic decision quality rather than demanding
identical random partitions.

K-means remains the original mini-batch algorithm with greedy k-means++,
replacement/reassignment and host convergence control, not Lloyd's algorithm.
NumPy random draws, small host metadata, and convergence checks remain. Teams
can take labels from k-means or use direct contiguous parent geometry. Source
lineage and protected decode-method hashes are in `TRITON_PREFILL_PROVENANCE.json`.
All R004 numerical source hashes are retained in R005.

## Budget/shape scope

Production Qwen2.5-7B uses D=128. Nominal S is not an input to the prefill kernels.
It is passed to the unchanged PyTorch estimator afterward. With P=16/R=4,
S/4 teams are requested, capped by available teams; S must be a positive multiple
of four. Whole-team membership makes actual rows read variable. No S=128 or 32k
specialization from a separate decode kernel restricts this port.

Synthetic 8k/32k rows and full model 8k/32k *context budgets* are distinct. The
original prompt framing/budget checks stay active; generation room is included.
See `KNOWN_ISSUES.md` for the A10 D=7 assignment diagnostic and long-context memory
limits. An accepted shape is not a guarantee of correctness on every platform.

## Environment and Docker profiles

| Profile | Environment | Use |
|---|---|---|
| `cluster` | `pytorch/pytorch:2.11.0-cuda12.6-cudnn9-runtime`; inherited matched Torch/Triton | Ampere deployment, including A10/sm86 |
| `local-5090` | Same Torch version; matched CUDA 12.8 Torch/vision/audio wheels | Local 5090/WSL execution |

The image includes the C compiler and Python headers needed for Triton's host
launcher, runs `pip check` and compiler-package import checks, and saves
`/workspace/BUILD_ENVIRONMENT.txt`. It does not install a host GPU driver or push
to a registry. The two profiles are not identical CUDA-binary environments.

The validated predecessor cluster stack was Torch 2.11.0+cu126 / Triton 3.6.0.
Dependency installation is performed at build time; transitive packages can
resolve differently in a new build. Inspect the saved environment and rerun the
release smoke rather than claiming an untested image is identical to R004.
For a host environment use an already matched CUDA Torch/Triton pair, not an
independent arbitrary Triton upgrade. Unsupported architectures fail explicitly.

## Local acceptance commands

Run from the repository root; Docker GPU/WSL integration must already be set up.
The helper mounts artifacts and HF cache only, not host source or CUDA binaries.

```bash
# Actual production compiler stack; no GPU is attached for this mode.
bash scripts/docker_triton_smoke.sh build --profile cluster
bash scripts/docker_triton_smoke.sh run --profile cluster -- \
  --mode compile --targets 80 86 --contexts 8192 32768

# Actual local GPU execution; separate CUDA 12.8 profile.
bash scripts/docker_triton_smoke.sh build --profile local-5090
bash scripts/docker_triton_smoke.sh run --profile local-5090 -- \
  --mode quick --run-tests

bash scripts/docker_triton_smoke.sh run --profile local-5090 -- \
  --mode matrix --contexts 8192 32768 \
  --sample-budgets 4 128 256 512 1024 2048 4096 8192 16384 32768 \
  --dtypes float16 bfloat16

bash scripts/docker_triton_smoke.sh run --profile local-5090 -- \
  --mode model --contexts 8192 32768 --sample-budgets 128 512 --max-new-tokens 2
```

Compiler-only tests cover all 16 numerical entry points with sm80/sm86 codegen;
they do not emulate or execute Ampere on a 5090. The default GPU suite now has
42 cases. One isolated D=7 accuracy diagnostic can be an expected failure on the
specific known A10/Triton 3.6.0 stack; 41 other cases remain mandatory there.
An XPASS is reported when that fixture succeeds. No expected-failure exemption
applies to missing CUDA, compilation failures, D=128, or the matrix/model worker.

The matrix runs native k-means/team construction, algorithmic checks, and the
original estimator at every S with zero/one generated exact row. Take-all budgets
are checked against dense attention. The quick fixture reduces iterations for
control-flow coverage; the matrix uses production k-means settings. Times are
single-KV-head preparation API timings, not full-model speedups. Optional
`--compare-torch` measures the original reference separately.

The model smoke uses an actual NIAH example selected by the original code and
runs the two parent policies. Two tokens check the plumbing, not task accuracy.
The new model/GPU suite still needs execution in the recipient's rebuilt R005
image; predecessor R004 successes are separately recorded in `VALIDATION_HISTORY.md`.

## Allocator and artifacts

The image and generated GPU YAMLs set:

```text
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

CLI and smoke executable startup set this default before importing torch, unless
either user allocator variable is explicitly present. External Python/notebook
callers must set their allocator configuration before importing PyTorch. There is
no library-import global side effect. Both spellings are recorded in environment
metadata; expandable segments is not a universal OOM cure.

Host outputs default to `.local-smoke-artifacts/`, configurable with
`--artifacts DIR` before the helper's `--`. HF cache is separate. Build logs and
image inspection metadata are saved. Every Python smoke produces a unique
`r005-...-handoff.zip` and prints `UPLOAD_ARCHIVE=...` or `FAILURE_ARCHIVE=...`.
The archive contains package/source/protocol/test/compiler diagnostics, not
model weights, HF tokens, or full prompt text. Compilation cache is pod-local.

For deployment instructions see `NAUTILUS.md`; source preservation and validation
are recorded in `../VALIDATION.md` and `TRITON_PREFILL_PROVENANCE.json`.

## External implementation references

These explain the mechanisms; the concrete A10 findings come from the supplied
run logs and maintainer report, not these references.

* PyTorch CUDA allocator configuration:
  https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf
* Kubernetes Job behavior:
  https://kubernetes.io/docs/concepts/workloads/controllers/job/
* Docker image publishing:
  https://docs.docker.com/docker-hub/repos/manage/hub-images/push/
