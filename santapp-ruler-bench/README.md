# SANTA-family RULER benchmark harness

This repository evaluates dense PyTorch SDPA and five SANTA-family sparse
attention backends on the 8k RULER benchmark. It includes local execution,
resume-safe reporting, an indexed Kubernetes workflow, compact prompt-free
exports, and data-driven postprocessing.

The checked-in defaults are examples rather than a fixed experiment. Edit a
YAML config or matrix to change tasks, budgets, clustering parameters, or the
set of methods.

## Standard attention backends

| Backend ID | Method | Checked-in default |
|---|---|---|
| `sdpa` | Dense PyTorch scaled-dot-product attention | dense reference |
| `santa` | Standalone SANTA token sampling | `S=128` samples per query head |
| `santapp` | Centroid-guided SANTA++ with IID token draws inside selected parents | `B=16`, `S=128`, 64 probes |
| `santapp_gumbel_topk` | SANTA++ whole-parent sampling without replacement | `B=16`, `S=128`, 64 probes |
| `team_sampler` | Hierarchical parent/team proposal using actual-key representatives | `P=16`, `R=4`, `S=128`, 64 probes |
| `team_gumbel_topk` | Whole-team sampling without replacement | `P=16`, `R=4`, `S=128`, 64 probes |

`generation.backends` may contain any subset of these names. Additional named
SANTA++ variants can be defined under `santapp_variants` and selected by name.
The checked-in SANTA++ parent size is **16** in both the local defaults and
the Kubernetes matrix.

## Repository layout

```text
configs/                     Local, smoke, and Kubernetes base configs
docs/                        Algorithm, output, validation, and cluster notes
k8s/                         Shareable Kubernetes templates
k8s/matrices/                Indexed experiment matrices
scripts/                     Setup, sweep, cluster, export, and analysis tools
src/santapp_ruler/           Installable benchmark package
tests/                       CPU-friendly unit and configuration tests
```

Generated runs, caches, exports, archives, and analysis folders are ignored by
Git and are not included in the repository.

## Installation

Python 3.11 or 3.12 is supported. Install a PyTorch build appropriate for the
machine first, then install the harness.

```bash
python -m pip install -r requirements-torch-cu128.txt   # example: newer local GPU
python -m pip install -e .[dev,analysis]
python -m santapp_ruler doctor
pytest -q
```

For the CUDA 12.6 environment used by the default Docker image:

```bash
python -m pip install -r requirements-torch-cu126.txt
python -m pip install -e .[dev,analysis]
```

The reference SANTA-family backends require CUDA. Most unit tests use synthetic
CPU tensors and do not download the model or RULER data.

Helper installers are provided for Linux (`scripts/setup_linux.sh`), WSL
(`scripts/setup_wsl.sh`), and Anaconda Prompt on Windows
(`scripts/setup_windows.bat`). For optional local generation with the pinned
official RULER code, use `scripts/bootstrap_ruler.py` followed by
`scripts/prepare_official.py`; generated datasets remain ignored by Git.

## Local use

Run the small two-task default:

```bash
python -m santapp_ruler run --config configs/default_8k.yaml
```

Run one prompt through all six standard backends with a four-token generation
budget:

```bash
python -m santapp_ruler run --config configs/smoke_all_backends_8k.yaml
# Or run the environment check, data validation, and smoke in sequence:
./scripts/run_local_smokes.sh
```

Useful overrides can be supplied without editing YAML:

```bash
python -m santapp_ruler run \
  --config configs/default_8k.yaml \
  --tasks niah_multiquery \
  --prompts-per-task 10 \
  --backends sdpa,santapp,team_sampler \
  --set santapp.samples_per_head=256 \
  --set santapp_variants.team_sampler.samples_per_head=256
```

The local grid runner reads `configs/example_sweep.yaml`:

```bash
python scripts/sweep.py configs/example_sweep.yaml
```

## Default Kubernetes template

The checked-in matrix is intentionally modest and easy to replace:

- tasks: `niah_multiquery`, `niah_multivalue`;
- prompts: 100 per task;
- settings: one run of each of the six standard backends;
- indexed work items: `2 tasks × 6 settings = 12`;
- default concurrency: 5 pods;
- SANTA++ parent size: `B=16`.

Before applying the manifests, replace these placeholders consistently:

- `yourname` with a short unique cluster user/project prefix;
- `your-registry/santapp-ruler:latest` with the pushed image;
- `ucsb-opus-lab`, storage class, region, and GPU products when using a
  different cluster.

A safer alternative is to render a personalized copy while leaving the source
templates unchanged. The renderer also derives the PVC results path from the
resource prefix unless `--results-root` is supplied explicitly:

```bash
PYTHONPATH=src python scripts/render_nautilus_manifests.py \
  --matrix k8s/matrices/default-2tasks-6methods-100p-8k.yaml \
  --output-dir k8s/generated/example-run \
  --resource-prefix example-santapp-ruler \
  --owner example \
  --image registry.example.org/group/santapp-ruler:latest \
  --parallelism 5
```

Apply the rendered files in order:

```bash
kubectl apply -f k8s/generated/example-run/00-santapp-ruler-pvc.yaml
kubectl apply -f k8s/generated/example-run/01-santapp-ruler-prefetch-job.yaml
kubectl wait --for=condition=complete job/example-santapp-ruler-prefetch --timeout=2h

kubectl apply -f k8s/generated/example-run/02-santapp-ruler-smoke-job.yaml
kubectl wait --for=condition=complete job/example-santapp-ruler-smoke --timeout=2h

kubectl apply -f k8s/generated/example-run/03-santapp-ruler-indexed-job.yaml
```

The smoke Job runs one `niah_multiquery` prompt through every standard backend.
The indexed Job then creates one shard per `(setting, task)` pair. Completed
indexes are resume-safe and can be re-run without duplicating valid records.
See `docs/NAUTILUS.md` and `k8s/LENS_QUICKSTART.md` for the full workflow.

### Creating another matrix

Copy `k8s/matrices/default-2tasks-6methods-100p-8k.yaml`, then change any of:

- `tasks` and `prompts_per_task`;
- setting names and backend IDs;
- `samples_per_head`, `group_size`, `parent_size`,
  `representatives_per_parent`, or `probe_queries`;
- the base config path.

The completion count is inferred as `len(tasks) × len(settings)`. The renderer
updates `completions` automatically and caps `parallelism` at that count.

## Compact exports

After the indexed Job finishes, apply the copy pod:

```bash
kubectl apply -f k8s/generated/example-run/04-santapp-ruler-results-copy-pod.yaml
kubectl logs -f pod/example-santapp-ruler-results-copy
kubectl cp \
  example-santapp-ruler-results-copy:/export/santapp-ruler-default-2tasks-6methods-100p-8k-results.tar.gz \
  ./results.tar.gz
```

The exporter intentionally excludes prompt text, selected prompt inputs, model
and dataset caches, and large run-state artifacts. It includes compact
per-example scores, predictions/references, metric fields, matrix metadata,
checksums, shard status, hardware provenance, and summary reports. Predictions
and references may still contain benchmark content, so review them before a
public release.

## General result analysis

`scripts/summarize_results.py` accepts any number of compact export directories,
parent directories, or exported archives. It discovers settings and tasks from
`matrix.json`; it does not require a historical experiment name or a fixed
method list, and it writes one output directory.

```bash
python scripts/summarize_results.py \
  --input results.tar.gz \
  --output-dir analysis \
  --overwrite
```

Multiple exports can be merged:

```bash
python scripts/summarize_results.py \
  --input first-results.tar.gz \
  --input second-results.tar.gz \
  --output-dir combined-analysis
```

When an exact `sdpa` setting is present, the analysis creates prompt-paired
comparisons against it. Disable that behavior with
`--reference-setting none`, choose another reference with
`--reference-setting SETTING`, or add explicit pairs with
`--comparison CANDIDATE:REFERENCE`.

The analysis writes accuracy tables and bootstrap intervals, pooled GQA-aware
and naive logical-access tables, paired comparisons, Pareto membership, a JSON
summary, Markdown summary, and PNG/PDF plots. Plot labels are derived from
matrix metadata and fall back to the setting name for custom methods.

## Metrics and interpretation

The primary reported access estimate is the union of sampled K/V rows across
query heads that share a GQA K/V head, plus routing-key reads, divided by dense
GQA K/V rows. Centroid reads are also retained as a routing subset. Naive
per-query-head accounting is reported separately.

These are logical vector-row counts, not measured DRAM transactions or a claim
of end-to-end speedup. Timing fields include runtime provenance because mixed
GPU products or software environments make direct speed comparisons invalid.
See `docs/OUTPUTS.md` for field definitions.

## Reproducibility and safety

Model and dataset revisions are pinned in the default configs. Each Kubernetes
matrix has a canonical hash; each shard records its work index, code
fingerprint, environment, selected UIDs, and prediction checksum. The exporter
rejects inconsistent matrix hashes, UID selections, or output checksums.

The default manifests contain no hostname exclusions. GPU product affinity is a
portable allow-list and can be changed by the renderer. Resource names use
obvious placeholders to prevent collisions in a shared namespace.

## License and attribution

The harness is Apache-2.0. See `NOTICE` and the pinned RULER provenance in
`src/santapp_ruler/ruler/provenance.py`.
