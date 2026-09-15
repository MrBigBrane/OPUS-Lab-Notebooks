# R007 A10 smoke deployment

Use `scripts/render_batched_prefill_smoke.py`, not the historical R006 renderer.
It renders only A10 GPU affinities and defaults to matrix parallelism eight.
It has no cluster side effects. Replace the generic owner/image below:

```powershell
python scripts/render_batched_prefill_smoke.py --owner researcher-r007 --image account/santapp-ruler:r007 --namespace ucsb-opus-lab --parallelism 8 --fit-batch-size 8 --output .\nautilus-r007
```

Use a new image tag and a fresh owner/results directory. Do not overwrite/resume
R006 outputs. All six nominal budgets (128 through 4096) and both contexts (8192,
32768) are retained. Both parent methods share unchanged grouped decode. Every
model worker generates 128 autoregressive steps without EOS stopping solely for
stable utilization measurement. This is not an official accuracy configuration.

## Ordered acceptance

Apply `00-pvc.yaml` and wait Bound, then `01-configmap.yaml`. Apply the CPU-only
`02-offline-compile-job.yaml` and `03-hf-prefetch-job.yaml`; wait for BOTH Complete.
The compiler runs CPU tests plus SM86 compilation of the old decoder, the batched
fitter and unchanged numerical prefill kernels. No model weights or GPU are
required by the compiler Job.

Next apply `04-gpu-kernel-job.yaml`. It runs actual CUDA decode correctness,
batched fitting correctness, and paired sequential/batched fit timing. Source-bound
success JSONs authorize the model stages; a compile pass is not a CUDA pass.

Then apply `05-ruler-canary-job.yaml`: **K-means, 32768 target context, nominal
S=4096, fit batch eight**, one A10. This tests the actual model plus clustering
and decode-cache memory, unlike the kernel-only gate.

Only after canary success, apply `06-ruler-matrix-job.yaml`. All 24 indexed cases
must complete. Parallelism eight is eight separate one-A10 Pods, not eight GPUs
used by one fit or one prompt. Each requests four CPUs and 32 GiB of host RAM;
eight active model Pods therefore request 32 CPUs/256 GiB host RAM/eight A10s.
The GPU itself has its own memory capacity, separate from the host RAM request.
There are no retries or automatic budget reductions after a failed shard.

`08-single-fit-reference-job.yaml` is optional. It uses the same new native fitter
with fit batch one, and grouped decode, for two 8k K-means cases. It does not invoke
the old Torch decoder. The mandatory kernel gate already compares the original
single-fit fitter and new batch on the same A10, so this extra model run is not
required for the default rollout.

Do not apply the directory en masse. Kubernetes does not enforce file-order
execution dependencies. A10-only node affinity is mandatory in every GPU manifest;
runtime checks also reject another device model rather than report mixed-GPU results.

## Collection

Use the supplied customized `run-r007.ps1 -Stage Collect` helper when present. It
collects Kubernetes diagnostics, creates a temporary CPU-only transfer Pod,
generates `MATRIX_SUMMARY.csv`/`MATRIX_ACCEPTANCE.json`, and downloads a small
results-only ZIP. The remote ZIP is stored without compression and prints progress;
Windows performs final compression. Weights, caches and binary tensors are excluded.

For manual collection with a running copy Pod, substitute your actual owner:

```powershell
kubectl -n ucsb-opus-lab exec researcher-r007-santa-ruler-copy -- python /workspace/scripts/summarize_r007_results.py --root /shared/results/researcher-r007
kubectl -n ucsb-opus-lab exec researcher-r007-santa-ruler-copy -- python /workspace/scripts/package_smoke_results.py --root /shared/results/researcher-r007 --output /shared/results/researcher-r007-handoff.zip --compression stored
kubectl cp "ucsb-opus-lab/researcher-r007-santa-ruler-copy:/shared/results/researcher-r007-handoff.zip" ".\researcher-r007-handoff.zip"
```

A relative local destination avoids Windows drive-letter/remote-colon ambiguity.
The copy Pod has one CPU, 512 MiB host RAM, and a two-hour deadline; delete it after
successful transfer. Keep failed Jobs until their logs are collected. Never delete
the PVC as part of routine transfer cleanup. A successful matrix is software
acceptance; inspect full-generation and fitting utilization before concluding the
>40% operational target is met. Retain zeros/boundaries rather than trimming them.
