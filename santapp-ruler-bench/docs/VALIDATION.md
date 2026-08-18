# Validation guide

Validation is intentionally split into CPU/static checks and GPU execution
checks. Passing the CPU suite does not prove that a model-sized CUDA smoke will
fit or execute on every GPU product.

## CPU and static checks

From an editable development install:

```bash
python -m compileall -q src scripts tests
python -m ruff check src scripts tests
pytest -q
```

The tests cover configuration merging and validation, task grading, sampling
identities, Gumbel inclusion probabilities, team construction, traffic
accounting, report generation, matrix expansion, Kubernetes manifest policy,
compact export behavior, and general result analysis with synthetic exports.

Validate all YAML files independently:

```bash
python - <<'PY'
from pathlib import Path
import yaml

for path in sorted(Path('.').rglob('*.yaml')):
    with path.open(encoding='utf-8') as handle:
        list(yaml.safe_load_all(handle))
    print(path)
PY
```

Inspect the default matrix and resolved method configs:

```bash
PYTHONPATH=src python - <<'PY'
from santapp_ruler.k8s_matrix import apply_work_item, load_matrix

matrix = load_matrix('k8s/matrices/default-2tasks-6methods-100p-8k.yaml')
base = matrix.load_base_run_config()
print(matrix.completion_count)
for item in matrix.work_items():
    config = apply_work_item(base, matrix, item)
    print(item.index, item.setting.name, item.task, config.generation.backends)
PY
```

Expected defaults are 12 work items, two tasks, six unique settings,
`prompts_per_task=100`, and five-way Job parallelism. The standard SANTA++
configuration and whole-parent variant use group size 16.

## Manifest rendering check

Render into a temporary directory and inspect names, images, completion count,
PVC references, GPU affinity, and absence of hostname exclusions:

```bash
PYTHONPATH=src python scripts/render_nautilus_manifests.py \
  --matrix k8s/matrices/default-2tasks-6methods-100p-8k.yaml \
  --output-dir /tmp/santapp-manifests \
  --resource-prefix validation-santapp \
  --owner validation \
  --image example.invalid/santapp:test \
  --parallelism 5
```

## Data-only validation

`validate-data` downloads/selects prompts and checks tokenizer budgets without
loading the language model:

```bash
python -m santapp_ruler validate-data \
  --config configs/default_8k.yaml \
  --prompts-per-task 1
```

This still needs network access and the pinned tokenizer/dataset.

## Local GPU execution smoke

On a CUDA machine with the pinned model available:

```bash
python -m santapp_ruler doctor
python -m santapp_ruler run \
  --config configs/smoke_all_backends_8k.yaml \
  --run-dir runs/smoke-all-backends
```

Confirm that each of the six backend directories contains one
`niah_multiquery.jsonl` record and that the final report has six backends. This
smoke uses only four generated tokens and tests code-path execution rather than
accuracy.

## Kubernetes smoke gate

After prefetch completes, run `02-santapp-ruler-smoke-job.yaml` before the
indexed Job. Check logs for all six backend headings and inspect the smoke run's
`summary.json`. Only then launch the full matrix.

The checked-in scheduling policy contains a GPU-product allow-list and no
specific node/hostname exclusions. Adjust product labels through the renderer
when a cluster uses different label values.

## Result-analysis smoke

The test suite creates a synthetic compact export and invokes
`scripts/summarize_results.py`. A manual check can use any real compact archive:

```bash
python scripts/summarize_results.py \
  --input path/to/results.tar.gz \
  --output-dir /tmp/santapp-analysis
```

Verify that there is one output directory, every setting from `matrix.json`
appears in `summary.json`, and no method is silently omitted.

## Release hygiene check

Before sharing an archive or repository snapshot:

```bash
find . -type d \( -name runs -o -name results -o -name analysis -o -name cache \)
find . -type f \( -name '*.jsonl' -o -name '*.csv' -o -name '*.png' -o -name '*.pdf' \)
```

The source tree should not contain generated benchmark records, model caches,
prompt data, plots, or local environment files. Also search for personal names,
usernames, registry accounts, absolute home paths, and one-off cluster hostnames.
