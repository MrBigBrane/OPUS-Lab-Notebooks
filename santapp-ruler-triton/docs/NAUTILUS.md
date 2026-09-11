# Nautilus / Lens deployment

## 1. Build and publish the release image

From this repository root, with your normal Docker registry permissions:

```bash
bash scripts/docker_triton_smoke.sh build --profile cluster \
  --image REGISTRY/PROJECT/santapp-ruler:r005-ampere

docker login
docker push REGISTRY/PROJECT/santapp-ruler:r005-ampere
docker buildx imagetools inspect REGISTRY/PROJECT/santapp-ruler:r005-ampere
```

The image/tag must exist in a registry accessible to Nautilus. A local Docker
Desktop image is not automatically uploaded. For a local image built under the
helper default, tag and push that **existing image** instead of rebuilding:

```bash
docker tag santapp-ruler:r005-ampere REGISTRY/PROJECT/santapp-ruler:r005-ampere
docker push REGISTRY/PROJECT/santapp-ruler:r005-ampere
```

Public repositories can be pulled without private credentials. A private image
requires the usual namespace `imagePullSecrets`; never embed passwords/tokens in
these YAMLs. Use a new tag/digest, not `latest` or an overwritten R004 tag. The
old R004 image does not contain the new defaults/tests/scripts. Do not label a
retagged R004 image as the new R005 release. The `local-5090` CUDA 12.8 image is for
local testing; the Ampere deployment profile remains CUDA 12.6.

## 2. Generate a colleague's normal manifest set

```bash
python scripts/render_nautilus_manifests.py \
  --owner REPLACE_WITH_YOUR_SLUG \
  --image REGISTRY/PROJECT/santapp-ruler:r005-ampere \
  --namespace ucsb-opus-lab \
  --output k8s/generated/REPLACE_WITH_YOUR_SLUG \
  --parallelism 4
```

This keeps the original four resource types and filenames:
`00-pvc.yaml`, `01-configmap.yaml`, `02-smoke-job.yaml`, `03-copy-pod.yaml`, plus
`all.yaml`. Default matrix: two contexts x four original backends = eight indexed
shards; all 13 original tasks; one prompt per task; original task-specific output
budgets. Native prefill is enabled for the two team backends. `sdpa`/`santa` remain
unchanged. At most four independent GPU pods run; each requests one GPU.

Optional flags: `--samples-per-head 128 512 2048`, `--contexts 8192 32768`,
`--backends santapp hierarchical`, `--tasks niah_single_1`, `--prompts-per-task 2`,
`--gpu-product NVIDIA-A10`, `--cpu 4`, `--memory 32Gi`. Repeated `--gpu-product`
allows several GPU types. For an original-reference ablation use
`--prefill-backend torch` and a distinct owner/run path.

Generated output directories must be new/empty. This prevents leftover `all.yaml`
files from an earlier matrix launching unintended workloads.

## 3. Targeted A10 acceptance: staged smoke

For the shape of the successful A10 experiment (not an accuracy benchmark):

```bash
python scripts/render_nautilus_manifests.py \
  --owner REPLACE_WITH_YOUR_SLUG-a10 \
  --image REGISTRY/PROJECT/santapp-ruler:r005-ampere \
  --namespace ucsb-opus-lab \
  --output k8s/generated/REPLACE_WITH_YOUR_SLUG-a10 \
  --staged-smoke --gpu-product NVIDIA-A10 \
  --contexts 8192 32768 --samples-per-head 128 512 \
  --backends santapp hierarchical \
  --tasks niah_single_1 --prompts-per-task 1 --max-new-tokens 2 \
  --parallelism 4
```

In Lens, select `ucsb-opus-lab` and apply the files in this order:

1. `00-pvc.yaml`: wait for Bound; then `01-configmap.yaml`.
2. `02-triton-kernel-smoke-job.yaml` and `03-hf-prefetch-job.yaml` may run together.
   The kernel job requests one GPU; prefetch requests no GPU. Require both Complete.
3. `04-ruler-smoke-job.yaml`: require 8/8 completions. `05-copy-pod.yaml` is optional
   for reading/copying the shared results.

The stage files are generated from the same resource policy as ordinary Jobs, not
separately maintained YAML. A staged set deliberately has **no all.yaml**: applying
all Jobs at once would not establish dependencies or prevent a cold HF-cache race.
The kernel gate runs all GPU regressions, then the synthetic context/budget matrix.
On the exact known A10/Triton 3.6.0 stack, its isolated D=7 diagnostic may report
XFAIL; production D=128 tests still must pass. Read `KNOWN_ISSUES.md`.

The eight real-model indexes are ordered context, then S, then backend:

```text
0  8192   128  santapp        1  8192   128  hierarchical
2  8192   512  santapp        3  8192   512  hierarchical
4  32768  128  santapp        5  32768  128  hierarchical
6  32768  512  santapp        7  32768  512  hierarchical
```

A new owner generates a separate PVC/cache/results; it does not delete/reuse the
old smoke volume. To reuse the existing cache manually, point all generated
`claimName` fields to that existing PVC and do not apply the new PVC file. Keep
OWNER_SLUG/new Job names distinct. Preserve existing result/cache volumes.

## 4. Deployment defaults reflecting the A10 findings

Every GPU job has:

```yaml
- name: PYTORCH_CUDA_ALLOC_CONF
  value: "expandable_segments:True"
```

Benchmark resources (host RAM, not VRAM):

```yaml
resources:
  requests: {cpu: "4", memory: 32Gi, nvidia.com/gpu: 1}
  limits:   {cpu: "4", memory: 32Gi, nvidia.com/gpu: 1}
```

The root permission init container has equal 50m CPU/32Mi limits and requests.
The benchmark remains UID/GID 10001. Only the two top-level writable directories
are initialized; no recursive chown of a shared model cache. Each pod has its own
`emptyDir` JIT cache; HF/data/results use the RWX shared PVC. Copy/prefetch/init
containers all have requests and limits; none requests a GPU.

The equal ratios address the observed maximum-1.2 request/limit policy. Right-size
host resources from actual measurements. The 32Gi default worked for the reported
software smoke but is not a hard capacity proof for every task or generation length.
Parallelism 4 reserves up to 16 CPUs/128Gi host RAM/4 GPUs for model workers, not four
processes sharing a single GPU's VRAM.

The A10 OOM was CUDA VRAM, not host RAM. Expandable segments enabled the reported
32k run; raising `memory` in Kubernetes cannot increase GPU VRAM. Larger S/longer
outputs still require validation. Nothing here silently reduces the workload.

## 5. Monitor, retrieve, and rerun

```bash
kubectl -n ucsb-opus-lab get jobs,pods -l benchmark-owner=REPLACE_WITH_YOUR_SLUG
kubectl -n ucsb-opus-lab logs POD_NAME -c benchmark
kubectl -n ucsb-opus-lab cp \
  REPLACE_WITH_YOUR_SLUG-santa-ruler-copy:/shared/results ./downloaded-results
python scripts/grade_and_plot.py ./downloaded-results/REPLACE_WITH_YOUR_SLUG/smoke
```

Native results are separated by mode/budget:
`/shared/results/OWNER/smoke/prefill-triton/S128/32768/santapp/`.
Without an explicit S sweep the budget folder is `Sconfig`. Original Torch/no-sweep
paths are retained. R005 refuses to mix older implicit-Torch results with new
native results during resume. A fresh owner also prevents an old completed case
from being mistaken for a new allocator/image test.

Changed container resources, image, or allocator env require a **new Job name**
or deliberate Job deletion/recreation, not editing an immutable pod template.
Do not delete the PVC/ConfigMap merely to replace a Job. Do not leave duplicate
active reruns reserving GPUs. Copy diagnostics before deleting failed pods.

A software smoke is not a memory/accuracy acceptance test for the full suite.
Before broad deployment, remove `--max-new-tokens 2`, test the largest intended S,
use several prompts, and verify no CUDA/host OOM and valid task outputs. Treat any
accuracy comparison as a separate experiment from these deployment checks.
