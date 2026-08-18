# Lens quick start

Render a uniquely named manifest set before using Lens. Do not apply the
checked-in `yourname`/`your-registry` placeholders directly.

```bash
PYTHONPATH=src python scripts/render_nautilus_manifests.py \
  --matrix k8s/matrices/default-2tasks-6methods-100p-8k.yaml \
  --output-dir k8s/generated/example-run \
  --resource-prefix example-santapp-ruler \
  --owner example \
  --image registry.example.org/group/santapp-ruler:latest \
  --parallelism 5
```

Create the rendered resources in this order:

1. `00-santapp-ruler-pvc.yaml`
2. `01-santapp-ruler-prefetch-job.yaml`; wait for **Complete**
3. `02-santapp-ruler-smoke-job.yaml`; inspect logs and wait for **Complete**
4. `03-santapp-ruler-indexed-job.yaml`
5. `04-santapp-ruler-results-copy-pod.yaml` after the indexed Job finishes

The default indexed Job has 12 completions and parallelism 5. The smoke Job runs
one prompt through all six standard methods. The manifests contain no specific
hostname exclusions.

From a terminal, copy the generated archive from `/export` in the result-copy
pod. The exact archive name is the matrix `experiment` followed by
`-results.tar.gz`.
