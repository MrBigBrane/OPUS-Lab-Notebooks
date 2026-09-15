# R006 staged Nautilus smoke

Use the separate personalized deployment bundle supplied with this release. The
colleague repository remains generic. Do not use the older renderer as the R006
decode acceptance workflow: its legacy indexed runner defaults to Torch decode.

## Build and render

From a clean checkout, with Docker already authenticated:

```bash
docker build --platform linux/amd64 --pull -t REGISTRY/PROJECT/santapp-ruler:r006-grouped .
docker run --rm --platform linux/amd64 REGISTRY/PROJECT/santapp-ruler:r006-grouped \
  python -m pytest -q -m 'not gpu'
docker push REGISTRY/PROJECT/santapp-ruler:r006-grouped
python scripts/render_grouped_decode_smoke.py \
  --owner YOUR-UNIQUE-SLUG --namespace ucsb-opus-lab \
  --image REGISTRY/PROJECT/santapp-ruler:r006-grouped \
  --output ../r006-deployment
```

Use a new image tag and owner/results path for changed code. The manifests assume
a publicly pullable image; local Docker login does not configure Kubernetes image
pull credentials. Docker Desktop must use Linux containers. The inherited image
base and package pins are retained; no independent Triton upgrade is requested.

## Default experiment

Qwen2.5-7B-Instruct, pinned existing revision, FP16, one NIAH single-key example,
contexts 8192 and 32768, both contiguous and K-means parents, and nominal token
budgets 128/256/512/1024/2048/4096. That is 24 indexed cases. Actual framed prompt
lengths depend on the existing data pipeline and leave generation room. Each case
runs 128 real decode steps with EOS stopping disabled. This is a software and
utilization smoke, not RULER accuracy evaluation.

Matrix parallelism is one: one GPU case at a time. Default node-label allowlist is
A10, RTX A5000, RTX 3090 and RTX 4090. Repeat `--gpu-product` to narrow it. Each
model worker requests/limits one GPU, four CPUs and 32 GiB HOST RAM; that is not a
32 GiB VRAM request. PVC is new, 50 GiB RWX rook-cephfs. Backend selection is
explicitly `grouped_triton`; numerical prefill remains `triton`.

## Apply in this order, waiting for completion at each gate

| File | Purpose |
|---|---|
| 00-pvc.yaml | Persistent HF cache and results; wait Bound |
| 01-configmap.yaml | Explicit model/algorithm configuration |
| 02-offline-compile-job.yaml | CPU tests and offline SM86/SM89 Triton compilation; no GPU |
| 03-hf-prefetch-job.yaml | Download pinned model and both datasets; no GPU |
| 04-decode-kernel-job.yaml | Real-CUDA correctness tests and full budget/context kernel matrix |
| 05-ruler-canary-job.yaml | One 8k/S4096 contiguous-parent full-model run |
| 06-ruler-matrix-job.yaml | 24 cases, one GPU worker at a time |
| 07-copy-pod.yaml | Temporary CPU-only results access, automatic 1800-second deadline |
| 08-torch-reference-job.yaml | Optional matched 8k contiguous S128/S4096 reference cases |

Only stages 02 and 03 may be launched together. Wait for BOTH before stage 04.
Do not apply the entire directory at once. The canary is an operator acceptance
gate before the matrix; source-bound compile/GPU success files additionally
prevent accidental execution against a stale successful image. Do not bypass a
failed gate or increase matrix parallelism while investigating utilization.
The reference job is optional and should not overlap the measured grouped run.

## Inspect and recover

Use the owner's namespace and prefix from `RUN_INFO.json` with `kubectl get jobs`,
`kubectl get pods`, and `kubectl logs -f job/PREFIX-STAGE`. For the Indexed Job,
inspect every completion in `INDEX_MAP.json`, not only the first Pod's logs.
Use the provided Windows script for fail-fast waits that also detect failed Jobs.

On failure, stop the rollout and retain diagnostics. CPU and GPU kernel gates
write summary JSON, test XML, hashes and a small archive under the results owner.
Model shards save `console.log`, `shard.json`, environment/configuration, normal
benchmark output, GPU telemetry and `decode-utilization.json` on the PVC.

After applying the temporary copy Pod, run inside it:

```bash
python /workspace/scripts/package_smoke_results.py \
  --root /shared/results/YOUR-UNIQUE-SLUG \
  --output /shared/results/YOUR-UNIQUE-SLUG-handoff.zip
```

Copy that ZIP with `kubectl cp`. The packager omits weights, model caches and nested
archives; it includes the small underlying diagnostics with a SHA256 manifest.
The CPU-only copy Pod expires after 1800 seconds; delete it after copying. Do not
delete the PVC unless intentionally discarding cached weights and results.

Logs from a crashed container may only exist in Kubernetes; save those with
`kubectl logs` before deleting failed Jobs. An OOM kill can interrupt the runner
before it writes a final shard report. A failed high-budget memory case is not
permission to silently lower its sample budget.

The full-model peak VRAM and achieved utilization remain unknown until these
stages run. Prefill remains unmodified and may still create low-utilization
intervals. Compare measured decode windows separately from whole-Pod startup,
compilation, model loading and prefill.
