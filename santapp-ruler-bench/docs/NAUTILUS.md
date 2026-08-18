# Kubernetes / Nautilus runbook

The files under `k8s/` are shareable templates for a ReadWriteMany PVC, a CPU
prefetch Job, a one-GPU all-method smoke Job, a five-way indexed benchmark Job,
and a CPU result-export pod. The default namespace and scheduling labels match
the cluster for which this repository was prepared; render or edit them for
another cluster.

## 1. Build and push the image

Set an image name owned by your group or registry account:

```bash
IMAGE=registry.example.org/group/santapp-ruler:latest \
  bash scripts/build_and_push_nautilus.sh
```

The image installs the package at build time. Pods do not install packages on
shared storage. The CUDA 12.6 PyTorch base is deliberate because the default GPU
pool includes Volta V100 and RTX A4000 devices.

## 2. Render unique manifests

Use a short DNS-safe prefix that will not collide with colleagues in the shared
namespace:

```bash
PYTHONPATH=src python scripts/render_nautilus_manifests.py \
  --matrix k8s/matrices/default-2tasks-6methods-100p-8k.yaml \
  --output-dir k8s/generated/example-run \
  --resource-prefix example-santapp-ruler \
  --owner example \
  --image registry.example.org/group/santapp-ruler:latest \
  --parallelism 5
```

Useful renderer options include `--namespace`, `--storage-class`,
`--storage-region`, repeated `--gpu-product`, `--gpu-resource`,
`--results-root`, and retry limits. When omitted, `--results-root` becomes
`/mnt/results/<resource-prefix>`. The renderer derives Job completions from
the matrix and uses the same image, PVC, experiment, and result root throughout.
It does not add hostname exclusions.

## 3. Create storage and prefetch immutable inputs

```bash
kubectl apply -f k8s/generated/example-run/00-santapp-ruler-pvc.yaml
kubectl get pvc example-santapp-ruler-rwx -w

kubectl apply -f k8s/generated/example-run/01-santapp-ruler-prefetch-job.yaml
kubectl logs -f job/example-santapp-ruler-prefetch
kubectl wait --for=condition=complete \
  job/example-santapp-ruler-prefetch --timeout=2h
```

The prefetch Job downloads the pinned model/tokenizer and dataset, selects the
matrix prompt UIDs once, and writes a cache marker under the experiment root.
Workers run in Hugging Face offline mode and reject a cache marker that does not
match the matrix or base config.

An optional `hf-token` Secret may provide an `HF_TOKEN` key. It is not required
for the public default model and dataset.

## 4. Run the all-method smoke Job

```bash
kubectl apply -f k8s/generated/example-run/02-santapp-ruler-smoke-job.yaml
kubectl logs -f job/example-santapp-ruler-smoke
kubectl wait --for=condition=complete \
  job/example-santapp-ruler-smoke --timeout=2h
```

The smoke config runs one `niah_multiquery` prompt with four generated tokens
through `sdpa`, `santa`, `santapp`, `santapp_gumbel_topk`, `team_sampler`, and
`team_gumbel_topk`. Its purpose is execution coverage, not benchmark signal.
Inspect failures before launching the full matrix.

## 5. Launch the indexed matrix

```bash
kubectl apply -f k8s/generated/example-run/03-santapp-ruler-indexed-job.yaml
kubectl get pods -l app.kubernetes.io/component=benchmark -w
```

The default matrix contains 12 indexes: two tasks times six settings. Up to five
pods run concurrently. Each index writes into its own shard directory and
creates `success.json` only after validating record count, pinned UIDs, matrix
hash, code fingerprint, and prediction checksum.

Useful commands:

```bash
kubectl get job example-santapp-ruler-sweep -o wide
kubectl logs -f job/example-santapp-ruler-sweep
kubectl get pods -l job-name=example-santapp-ruler-sweep
```

Indexed Jobs can be re-applied after deleting the old Job. Valid completed
shards are detected and skipped. Do not delete the PVC unless the cache and all
results are no longer needed.

## 6. Export compact results

```bash
kubectl apply -f k8s/generated/example-run/04-santapp-ruler-results-copy-pod.yaml
kubectl logs -f pod/example-santapp-ruler-results-copy
```

The copy pod runs `scripts/k8s_export_results.py`, writes a compact directory and
archive under `/export`, then sleeps so `kubectl cp` can retrieve the files.
For the default matrix:

```bash
kubectl cp \
  example-santapp-ruler-results-copy:/export/santapp-ruler-default-2tasks-6methods-100p-8k-results.tar.gz \
  ./santapp-ruler-results.tar.gz
kubectl cp \
  example-santapp-ruler-results-copy:/export/santapp-ruler-default-2tasks-6methods-100p-8k-results.tar.gz.sha256 \
  ./santapp-ruler-results.tar.gz.sha256
```

The archive excludes prompt inputs and the Hugging Face cache. Use
`--allow-partial` only when intentionally exporting an incomplete matrix; the
status file records every missing or invalid shard.

## 7. Analyze locally

```bash
python -m pip install -e .[analysis]
python scripts/summarize_results.py \
  --input santapp-ruler-results.tar.gz \
  --output-dir analysis
```

## Editing the template

Copy the matrix and base config rather than changing algorithm code for ordinary
sweeps. Setting names only need to be unique and DNS-safe enough for generated
shard paths. Each setting identifies a backend and may override its budget,
mode, parent size, group size, number of representatives, or probe count.

When the number of settings or tasks changes, rerun the manifest renderer. Do
not manually leave a stale `completions` value in an indexed Job.

## Cleanup

Delete Jobs and the copy pod without deleting shared results:

```bash
kubectl delete job example-santapp-ruler-smoke --ignore-not-found
kubectl delete job example-santapp-ruler-sweep --ignore-not-found
kubectl delete job example-santapp-ruler-prefetch --ignore-not-found
kubectl delete pod example-santapp-ruler-results-copy --ignore-not-found
```

Delete the PVC only after confirming that no experiment or colleague depends on
it:

```bash
kubectl delete pvc example-santapp-ruler-rwx
```
