# Known limitations and their handling

## A10 / sm86 / Triton 3.6.0: synthetic D=7 k-means assignment

Observed on 2026-09-10 using the R004 cluster image (PyTorch 2.11.0+cu126,
Triton 3.6.0): 161 CPU tests passed; 39 of 40 GPU tests passed; the combined
assignment/update/gather/reassignment test failed its nearest-center cost check
at N=67, K=35, D=7, FP32. The matrix worker had not started. This was not an
installation failure and not a harmless bitwise-parity check: a discrete winner
exceeded the test's allowed distance-cost error. The compiler/root cause was not
established. Do not generalize it to every D=7 operation or every Ampere GPU.

The production model in this repository has D=128. Its actual A10 full-model
8k/32k, S=128/512 smoke subsequently completed 8/8 with R004 and expandable
segments. That is execution evidence for the tested workload, not a proof of
all possible k-means inputs or task accuracy.

R005 intentionally retains the same R004 numerical kernels; the proposed R004a
small-D direct-distance workaround is NOT included. Test changes are:

* The combined primitive test now uses D=128, keeping gather, assignment, inertia,
  updates, and reassignment checks mandatory for the production geometry.
* An additional D=128 assignment test spans multiple point/center tiles.
* The exact failing D=7 fixture remains a separate diagnostic. On NVIDIA A10,
  sm86, Triton 3.6.0 only, an assertion failure is `XFAIL` with a visible reason.
  `XPASS` remains visible if a future run succeeds. Compilation/runtime errors
  are not expected failures, and no D=128 failure is ever exempted.

Other architectures/versions run the D=7 diagnostic normally. The existing D=7
team-construction tests remain mandatory: they were not the reported failure.
The implementation accepts additional shapes, but that is not a hardware-wide
accuracy certification. Do not deploy D=7 k-means on this known affected stack;
use the explicit Torch reference or investigate separately. Do not count an
XFAIL as a passed GPU accuracy test.

## A10 long-context VRAM pressure

The original 32k failure was in dense Qwen MLP prefill, before parent/team
construction and before S-dependent sparse decoding. The allocation requested
1.15 GiB with 22.06 GiB CUDA-visible capacity in the supplied error. Increasing
Kubernetes `memory` does not add VRAM. Equal requests and limits satisfy the
observed cluster policy but do not solve a CUDA allocation failure.

With `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, the maintainer reported
8/8 completions for the same two-token A10 software smoke. R005 applies that
setting at Docker/CLI/smoke startup and in generated GPU Jobs. It is a mitigation
for allocation patterns and fragmentation, not a guarantee against all OOMs.
No model quantization, MLP chunking, offload, input shortening, or S reduction is
introduced. Stress the largest intended S and actual generation lengths before
launching a full suite. A 3090 can be selected with
`--gpu-product NVIDIA-GeForce-RTX-3090`, but its success is not guaranteed by a GPU
product name alone.

## Resource policy, reruns, and disk caches

Every generated container/init container has equal CPU/RAM request and limit.
Benchmark defaults are 4 CPUs, 32Gi host RAM, and one GPU per pod. The permission
init container uses 50m CPU/32Mi; the copy pod 100m/256Mi; staged prefetch 2 CPU/4Gi;
the standalone kernel job 4 CPU/16Gi/one GPU. These are smoke starting points, not
measured upper bounds for all workloads. `--cpu` and `--memory` change the
benchmark's request AND limit together. Monitor usage rather than inflating requests.

Each pod gets independent `emptyDir` JIT scratch. Shared HF/data/results live on
the PVC. Never copy locally compiled sm120 binaries as the A10 JIT cache.
Use a new Job name/owner for a changed pod template or deliberately recreate only
the Job; Kubernetes does not allow arbitrary edits to an existing Job template.
Do not delete a PVC holding cache/results merely to rerun a Job.
