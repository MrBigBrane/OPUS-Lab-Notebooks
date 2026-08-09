# SANTA / SANTA++ × RULER benchmark harness

A batch-1 research harness for comparing three decode-attention paths on the 13 classic synthetic RULER tasks:

- **`sdpa`** — stock Hugging Face cached generation with PyTorch SDPA.
- **`santa`** — compute the full `qK^T` distribution for each decode query head, draw `S` token indices IID with replacement, gather the corresponding values, and return their arithmetic mean.
- **`santapp`** — dense prefill, prompt clustering, centroid-guided IID sampling with importance correction, and an exact suffix containing only tokens that have not yet been assigned to clusters.

The package selects the same prompt rows for every method, writes each completed prediction immediately, resumes interrupted runs safely, applies the vendored RULER task-family grader, and reports GQA-aware theoretical memory-access estimates alongside quality and timing.

## Current group policy

The reference paths implement these policies directly:

| Policy | Behavior |
|---|---|
| SANTA++ fixed prompt-tail window | **Always zero.** There is no configurable 64-token exact tail. |
| SANTA++ growing exact suffix | Every prompt token, including the final token that is re-fed for the first approximate decode call, is covered by the frozen prompt clusters. The exact suffix starts at zero and contains only generated tokens that have not been assigned to clusters. |
| SANTA exact window | **None.** SANTA samples from the entire current KV cache. |
| SANTA++ probe queries | Exactly the final `P` prompt-query positions in chronological order; default `P=64`. No random draw from the last quarter. |
| SANTA++ defaults | Group size 16, 128 samples per query head, 64 probe queries, and the existing MiniBatchKMeans defaults. |
| SANTA default | 128 IID samples per query head. The sample budget is its only algorithm parameter. |
| Primary access metric | `decode_gqa_total_access_pct`: GQA-aware KV-row union plus shared centroid-vector reads. |

Prompt clusters are frozen after prefill. A policy for periodically assigning generated tokens to clusters is intentionally not assumed in this repository.

## Included run configurations

| Config | Tasks and instances | Methods | Generation limit |
|---|---|---|---|
| `configs/default_8k.yaml` | 6 representative tasks × 5 prompts | `sdpa`, `santapp` | Per-task RULER defaults |
| `configs/simple_8k.yaml` | 20 `niah_multivalue` + 20 `qa_1` | all three | Per-task RULER defaults |
| `configs/smoke_8k.yaml` | 2 tasks × 1 prompt | all three | 8 tokens, diagnostic only |
| `configs/smoke_all_tasks_8k.yaml` | 13 tasks × 1 prompt = 39 rows | all three | Per-task RULER defaults |
| `configs/all_tasks_8k.yaml` | 13 tasks × 500 prompts | all three | Per-task RULER defaults |

The ordinary `santapp-ruler run` command still resolves to one configuration; sweep behavior is separate and explicit.

## Linux and WSL installation

### Requirements

- Linux or WSL2 with an NVIDIA GPU visible to PyTorch.
- Python 3.11 or 3.12.
- A compatible NVIDIA driver.
- Enough GPU memory for the pinned Qwen2.5-3B model and the selected context length.

From the repository root:

```bash
chmod +x scripts/setup_linux.sh
./scripts/setup_linux.sh --dev
source .venv/bin/activate
```

The setup script:

1. chooses `python3.12` or `python3.11` unless `--python`/`PYTHON_BIN` is supplied;
2. creates or reuses `.venv`;
3. installs the CUDA 12.8 PyTorch requirement file;
4. installs this package; and
5. runs the environment doctor.

Useful alternatives:

```bash
# Reuse a PyTorch installation managed by the user or site administrator.
./scripts/setup_linux.sh --skip-torch --dev

# Use another virtual-environment location or interpreter.
./scripts/setup_linux.sh --python /usr/bin/python3.11 --venv .venv-ruler --dev

# Point at a different PyTorch requirement file.
TORCH_REQUIREMENTS=/path/to/requirements-torch.txt ./scripts/setup_linux.sh --dev
```

The default PyTorch wheel is isolated in `requirements-torch-cu128.txt`; the package itself does not force a particular torch wheel. This makes the same repository usable on WSL workstations and Linux servers with centrally managed CUDA environments.

Check the environment at any time:

```bash
santapp-ruler doctor
```

## First-run checks

### 1. List the supported tasks and their default generation limits

```bash
santapp-ruler list-tasks
```

### 2. Validate prompt selection and token budgets

```bash
santapp-ruler validate-data --config configs/simple_8k.yaml
```

This loads the tokenizer and selected RULER rows without loading model weights.

### 3. Verify the custom dense cache against stock SDPA

```bash
santapp-ruler fidelity --config configs/default_8k.yaml --tokens 16
```

This should match every generated token ID before sparse results are trusted on a new Transformers/PyTorch environment.

### 4. Run the fast two-task plumbing smoke

```bash
santapp-ruler run --config configs/smoke_8k.yaml
```

The 8-token override makes this a diagnostic run rather than a standard RULER score.

### 5. Run the all-method, all-task handoff smoke

```bash
santapp-ruler run --config configs/smoke_all_tasks_8k.yaml
```

This produces 39 prediction rows: `sdpa`, `santa`, and `santapp` on one prompt from each of the 13 tasks, using each task's normal generation budget.

## Routine colleague benchmark

The group-policy comparison requested for routine use is already encoded:

```bash
santapp-ruler run --config configs/simple_8k.yaml
```

It runs the same deterministically selected 20 `niah_multivalue` prompts and 20 `qa_1` prompts for all three methods. Completed rows are resumable.

Use an explicit path when several configurations may coexist:

```bash
santapp-ruler run \
  --config configs/simple_8k.yaml \
  --run-dir runs/simple-8k-S128
```

Repeating the same command skips completed UIDs. The resume guard refuses to mix results if the model, data revision, task selection, generation settings, SANTA parameters, or SANTA++ parameters have changed.

## Selecting tasks, instances, methods, and generation length

Every task name in `santapp-ruler list-tasks` can be selected in any subset. `--prompts-per-task` applies the requested count independently to every selected task.

```bash
santapp-ruler run \
  --config configs/default_8k.yaml \
  --tasks niah_single_2,niah_multivalue,qa_1 \
  --prompts-per-task 7 \
  --backends sdpa,santa,santapp \
  --run-dir runs/custom-subset
```

By default, each task uses the generation limit registered in `src/santapp_ruler/ruler/tasks.py`. Set one exact common limit only when needed:

```bash
santapp-ruler run \
  --config configs/default_8k.yaml \
  --max-new-tokens 16 \
  --run-dir runs/diagnostic-16-tokens
```

A nonstandard limit changes the benchmark protocol and should be treated as a diagnostic unless it matches the intended evaluation policy.

## Algorithm-parameter overrides

Dotted overrides are parsed as YAML and may be repeated.

```bash
# SANTA: its only algorithm hyperparameter.
santapp-ruler run \
  --config configs/simple_8k.yaml \
  --set santa.samples_per_head=256 \
  --run-dir runs/simple-santa-S256

# SANTA++ sample budget and clustering granularity.
santapp-ruler run \
  --config configs/simple_8k.yaml \
  --set santapp.samples_per_head=256 \
  --set santapp.group_size=8 \
  --run-dir runs/simple-santapp-S256-B8

# Use the final 32 prompt tokens rather than the final 64 as probes.
santapp-ruler run \
  --config configs/simple_8k.yaml \
  --set santapp.probe_queries=32 \
  --run-dir runs/simple-P32
```

There is deliberately no SANTA or SANTA++ fixed-recent-window setting.

## Full RULER matrix

`configs/all_tasks_8k.yaml` requests 500 prompts for all 13 classic synthetic tasks. Run only the methods needed when the full three-method matrix would be excessive:

```bash
santapp-ruler run \
  --config configs/all_tasks_8k.yaml \
  --backends sdpa,santapp \
  --run-dir runs/ruler-all-8k-sdpa-santapp
```

Prompt availability and tokenizer budgets can be checked before model loading:

```bash
santapp-ruler validate-data --config configs/all_tasks_8k.yaml
```

## GQA-aware memory-access accounting

The percentages are decode-time **logical head-vector equivalents**, not measured HBM transactions or a claim about a particular fused-kernel implementation.

For one layer, one decode token, and one KV head, define:

- `N`: current KV-cache token count;
- `G`: number of query heads sharing that KV head;
- `S`: samples drawn per query head;
- `U`: number of distinct sampled token rows in the union across those `G` query heads, including deduplication of repeated draws;
- `R`: SANTA++ growing exact-suffix length, which is zero on the first approximate call and then contains only previously generated tokens;
- `M`: number of active key-centroid vectors.

The GQA-aware dense baseline is `2N` vectors: each K row and each V row once for the shared KV head.

### SANTA++

```text
GQA KV vectors       = 2(U + R)
GQA centroid vectors = M
GQA total percentage = [2(U + R) + M] / (2N) × 100
```

Centroid metadata is shared at the KV-group level rather than replicated for each query head. The primary reported field is the total percentage; KV and centroid components are also reported separately.

The explicit naive convention treats every query head as if SDPA read an independent copy of the KV cache and retains draw multiplicity:

```text
naive dense vectors    = 2NG
naive KV vectors       = 2G(S + R)
naive centroid vectors = MG
```

Because the denominator is incorrectly replicated per query head, this percentage is generally more optimistic.

### SANTA

SANTA reads all K rows to form the exact sampling distribution and reads only sampled V rows:

```text
GQA KV vectors    = N + U
naive KV vectors  = G(N + S)
centroid vectors  = 0
```

### SDPA

Both GQA-aware and naive percentages are 100% by definition.

The main fields are:

- `decode_gqa_total_access_pct`
- `decode_gqa_kv_access_pct`
- `decode_gqa_centroid_access_pct`
- `decode_naive_total_access_pct`
- `decode_naive_kv_access_pct`
- `decode_naive_centroid_access_pct`

Raw vector counts and GQA-group call counts are retained so aggregate percentages are recomputed from summed numerators and denominators rather than averaging percentages.

## SANTA++ probe and suffix details

For a prompt of length `T`:

```text
clustered prompt = token positions [0, T - 1]
probe positions  = [T - P, ..., T - 1]
fixed exact tail = 0
initial growing exact suffix = 0
```

Dense prefill captures the prompt query projections. The final `P` post-RoPE query vectors are used to fingerprint every prompt key, including token `T-1`. After clustering, the cache is trimmed to `T-1` and token `T-1` is re-fed as the first approximate decode query. Its re-created KV row occupies the same cluster-covered position. Therefore:

- the first SANTA++ decision has zero exact tokens outside the clusters;
- the next decision has one exact token: the first generated token;
- generated tokens are never silently inserted into the frozen clusters.

## Data provenance

The default config pins:

```text
Model:   Qwen/Qwen2.5-3B-Instruct
Revision: aa8e72537993ba99e69dfaafa59ed015b17504d1

Dataset: SaylorTwift/RULER-8192-Qwen2.5-3B-tokenizer
Revision: 6ee2d0f4e9b8983361da35204ead8931c3f65ad4
```

To generate local JSONL from the pinned NVIDIA/RULER source instead of using the convenience mirror:

```bash
python -m pip install -e ".[official-data]"
python scripts/bootstrap_ruler.py
python scripts/prepare_official.py \
  --context-length 8192 \
  --num-samples 500 \
  --tasks niah_single_1,niah_single_2,niah_single_3,niah_multikey_1,niah_multikey_2,niah_multikey_3,niah_multivalue,niah_multiquery,vt,cwe,fwe,qa_1,qa_2
```

Then set:

```bash
--set benchmark.data.source=local \
--set benchmark.data.local_root=data/ruler_8k
```

## Outputs

Each run directory contains:

```text
config.resolved.yaml
runtime.pre_model.json
runtime.json
selected_prompts.jsonl
run_status.json
predictions/<backend>/<task>.jsonl
predictions/<backend>/summary.csv
predictions/<backend>/submission.csv
summary.md
summary.csv
summary.json
per_example.csv
```

`summary.md` puts the GQA-aware total access percentage first, followed by the GQA KV and centroid components and the naive total. See `docs/OUTPUTS.md` for the complete schema.

## Timing scope

Timed regions are CUDA-synchronized at their boundaries. Reports include:

- dense prefill;
- SANTA++ clustering;
- decode;
- end-to-end per-example time;
- output-token throughput; and
- peak allocated/reserved CUDA memory.

The SANTA and SANTA++ paths are readable PyTorch references with Python-level per-head dispatch, not final fused kernels. Use them for algorithmic correctness, quality, theoretical access comparisons, and reproducible RULER evaluation—not as a ceiling on attainable kernel speed.

## Optional sweep

Sweep execution is isolated from normal runs:

```bash
python scripts/sweep.py configs/example_sweep.yaml --dry-run
python scripts/sweep.py configs/example_sweep.yaml
```

Each point gets its own resumable run directory; the script does not launch competing model processes on one GPU.

## Repository map

```text
src/santapp_ruler/
├── attention/
│   ├── minibatch_kmeans.py
│   ├── probes.py
│   ├── santa.py
│   ├── santapp.py
│   └── traffic.py
├── backends.py
├── cli.py
├── config.py
├── data.py
├── reporting.py
├── runner.py
├── run_state.py
└── ruler/
    ├── grader.py
    ├── provenance.py
    └── tasks.py
```

Algorithm details are in `docs/ALGORITHM.md`; output fields are in `docs/OUTPUTS.md`; the checks executed for this handoff are recorded in `SMOKE_TEST_STATUS.md`.

## Scope

- Qwen2/Qwen2.5 architecture (`model_type == "qwen2"`).
- Batch size 1 for the reference SANTA/SANTA++ cache.
- One GPU; no tensor parallelism.
- Dense prefill and approximate decode.
- Frozen prefill clusters during generation.
- No fused sparse CUDA kernel in this repository.
- The included 8k mirror is tokenizer-specific; generate local RULER data for other tokenizers or context lengths.

## License and attribution

The harness is Apache-2.0. See `NOTICE` and `CITATION.cff` for attribution to NVIDIA/RULER and the SANTA/SANTA++ research implementation. This project is not endorsed by NVIDIA.
