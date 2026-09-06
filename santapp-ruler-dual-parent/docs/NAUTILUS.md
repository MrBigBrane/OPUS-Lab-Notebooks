# Nautilus / Kubernetes workflow

## Build and push

```bash
docker build -t REGISTRY/PROJECT/santapp-ruler:dual-parent-v1 .
docker push REGISTRY/PROJECT/santapp-ruler:dual-parent-v1
```

The Dockerfile removes irrelevant inherited `spin` tooling before `pip check` and installs dependencies at image-build time. Pods do not install Python environments on CephFS.

## Render collision-free resources

```bash
python scripts/render_nautilus_manifests.py \
  --owner REPLACE_WITH_YOUR_SLUG \
  --image REGISTRY/PROJECT/santapp-ruler:dual-parent-v1 \
  --namespace ucsb-opus-lab \
  --output k8s/generated/REPLACE_WITH_YOUR_SLUG \
  --contexts 8192 32768 \
  --backends sdpa santa santapp hierarchical \
  --parallelism 1
```

Useful overrides include `--tasks`, repeated `--gpu-product`, `--storage-class`, and `--pvc-size`.

The default eight indexed shards are:

```text
index 0:  8192 / sdpa
index 1:  8192 / santa
index 2:  8192 / santapp       (post-RoPE-key K-means parents)
index 3:  8192 / hierarchical  (contiguous-span parents)
index 4: 32768 / sdpa
index 5: 32768 / santa
index 6: 32768 / santapp
index 7: 32768 / hierarchical
```

Each shard evaluates every task named in `SMOKE_TASKS`. The renderer supplies all 13 classic RULER tasks by default; a quick software smoke can use `--tasks niah_single_1`.

## Lens workflow

In the selected namespace, create the generated resources in this order:

1. `00-pvc.yaml`; wait until the PVC is `Bound`.
2. `01-configmap.yaml`.
3. `03-copy-pod.yaml`; confirm it reaches `Running` and mounts `/shared`.
4. `02-smoke-job.yaml`; inspect indexed pod logs until all completions succeed.

The Job includes a short root init container that creates `/shared/results` and `/shared/hf`, assigns those directories to UID/GID 10001, and exits. The benchmark itself remains non-root. The copy pod requests no GPU.

## Command-line alternative

```bash
kubectl apply -f k8s/generated/REPLACE_WITH_YOUR_SLUG/all.yaml
kubectl -n ucsb-opus-lab get jobs,pods \
  -l benchmark-owner=REPLACE_WITH_YOUR_SLUG
```

Copy completed results:

```bash
kubectl -n ucsb-opus-lab cp \
  REPLACE_WITH_YOUR_SLUG-santa-ruler-copy:/shared/results \
  ./downloaded-results
```

Regrade and plot locally:

```bash
python scripts/grade_and_plot.py \
  ./downloaded-results/REPLACE_WITH_YOUR_SLUG/smoke
```

The output contains combined backend/task CSVs and a PNG/PDF Pareto plot for each completed context.

## Cleanup

After results are safely copied, delete the Job, copy pod, and ConfigMap. Delete the PVC only after confirming the local files. The K-means `santapp` shards include a preprocessing stage for every layer/KV head; keep `parallelism=1` unless the namespace has sufficient independent GPU and shared-storage capacity.
