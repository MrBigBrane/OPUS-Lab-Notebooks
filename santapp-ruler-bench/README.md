# SANTA++ × RULER benchmark harness

A clean, batch-1 research harness for comparing the supplied **SANTA++ centroid-guided sparse attention** implementation against **stock Hugging Face / PyTorch SDPA** on selected RULER tasks.

The repository is deliberately a command-line package rather than a notebook. It selects the same prompts for both backends, writes one prediction row at a time, resumes interrupted runs, applies RULER's own synthetic-task grader, and reports decode-time KV access alongside quality and timing.

## Default run at a glance

| Setting | Default |
|---|---|
| Model | `Qwen/Qwen2.5-3B-Instruct` at revision `aa8e72537993ba99e69dfaafa59ed015b17504d1` |
| Context length | 8,192 total tokens, using RULER's prompt-plus-generation convention |
| Tasks | `niah_single_1`, `niah_multikey_1`, `niah_multiquery`, `vt`, `fwe`, `qa_1` |
| Prompts per task | 5 |
| Backends | stock `sdpa`, then `santapp` |
| Total prediction rows | 6 tasks × 5 prompts × 2 backends = 60 |
| SANTA++ group size | 16 |
| Samples per query head | 128 |
| Probe queries | 64, drawn from the final quarter of the prompt |
| Exact recent window | 64 tokens |
| Clustering | CUDA MiniBatchKMeans with the notebook's scikit-learn-like defaults |
| Generation | greedy; official RULER token budget for each task |
| Sweep behavior | none—the normal command runs exactly one configuration |

This default is intended to fit a roughly couple-of-hours local experiment budget on the 24 GB RTX 5090 laptop used for development. Actual time depends heavily on how many tokens the model emits; run the smoke test first on every new machine.

## What is kept faithful to the notebook

The extracted path preserves the current algorithmic defaults:

1. Dense SDPA prefill.
2. Capture pre-RoPE query projections for all layers.
3. Apply Qwen RoPE and build per-token query-response fingerprints.
4. Cluster the old prefix independently for every layer and KV head with GPU MiniBatchKMeans.
5. Represent each group with its size and mean key.
6. During one-token decode, score group metadata, sample groups, sample a token uniformly inside each selected group, and apply importance correction.
7. Read the recent window and generated history exactly.

The implementation improves one important engineering detail without changing the estimator: it converts only sampled/recent K and V rows to float32 rather than converting an entire KV head before indexing. It also preallocates the custom cache instead of concatenating the full cache every token.

The SDPA baseline is not the notebook's handwritten dense branch. It is the unpatched model using its normal Hugging Face cache and `attn_implementation="sdpa"`.

## Installation

### Windows / RTX 5090

Use **Anaconda Prompt**, from the repository root:

```bat
scripts\setup_windows.bat
```

Equivalent manual commands:

```bat
conda create -n santapp-ruler python=3.11 -y
conda activate santapp-ruler
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-torch-cu128.txt
python -m pip install -r requirements.txt
python -m santapp_ruler doctor
```

The pinned core versions match the saved successful notebook environment:

- Python 3.11
- PyTorch 2.11.0 from the CUDA 12.8 wheel index
- Transformers 5.12.1

A separately installed CUDA Toolkit is not required for the prebuilt PyTorch wheel. A compatible NVIDIA driver is required. On an ASUS ROG laptop, do not use Armoury Crate's **Eco** GPU mode.

### Linux

With Python 3.11 available:

```bash
scripts/setup_linux.sh
```

The package intentionally does not depend on FlashAttention, vLLM, SGLang, NeMo, or `accelerate` for the normal benchmark path.

## First-run sequence

### 1. Check the machine

```bash
santapp-ruler doctor
```

### 2. Validate the pinned 8k prompts and tokenizer budgets

```bash
santapp-ruler validate-data --config configs/default_8k.yaml
```

This downloads only the tokenizer and selected dataset rows. Every chosen task must contribute exactly `prompts_per_task` valid prompts.

### 3. Verify that the extracted custom dense cache matches stock cached SDPA

```bash
santapp-ruler fidelity --config configs/default_8k.yaml --tokens 16
```

This is the repository-level analogue of the notebook's dense-patch fidelity cell. Do not trust sparse results on a new Transformers version if this check fails.

### 4. Run a short pipeline smoke test

```bash
santapp-ruler run --config configs/smoke_8k.yaml
```

The smoke config caps generation at eight tokens. Its scores are **not** a standard RULER result; it exists only to exercise data loading, both backends, persistence, and grading.

### 5. Run the default 8k benchmark

```bash
santapp-ruler run --config configs/default_8k.yaml
```

The shorter `santapp-ruler run` command uses the same built-in defaults, so it also works when invoked outside the repository root.

An interrupted run is resumable because each completed prediction is flushed immediately. Resume is guarded: the harness refuses to reuse a prediction directory if the model, selected tasks, prompt count, data revision, generation settings, or SANTA++ parameters changed. For an easily reusable run path:

```bash
santapp-ruler run \
  --config configs/default_8k.yaml \
  --run-dir runs/8k-default
```

Run the same command again to skip completed task/index pairs.

## Select a task subset and prompt count

Every selected task receives exactly the requested count:

```bash
santapp-ruler run \
  --config configs/default_8k.yaml \
  --tasks niah_single_1,vt,qa_1 \
  --prompts-per-task 3 \
  --run-dir runs/8k-small
```

List all supported task names:

```bash
santapp-ruler list-tasks
```

The full classic synthetic set is also listed in `configs/all_tasks_8k.yaml`.

## Override one algorithm parameter without making a sweep

Use repeatable dotted overrides:

```bash
santapp-ruler run \
  --config configs/default_8k.yaml \
  --run-dir runs/S256 \
  --set santapp.samples_per_head=256
```

Other common fields:

```text
santapp.group_size
santapp.probe_queries
santapp.recent_window
santapp.kmeans.random_state
benchmark.selection_seed
generation.backends
```

The ordinary command still runs one resolved configuration. The exact resolved YAML is copied into the run directory.

## Optional explicit sweep

A separate sequential script is provided so sweep logic cannot accidentally affect a normal run:

```bash
python scripts/sweep.py configs/example_sweep.yaml --dry-run
python scripts/sweep.py configs/example_sweep.yaml
```

It launches independent, resumable run directories. It does not parallelize multiple model processes onto one GPU.

## RULER data and grader provenance

### Default data

For easy setup, the default config pins the Hugging Face dataset mirror:

```text
SaylorTwift/RULER-8192-Qwen2.5-3B-tokenizer
revision 6ee2d0f4e9b8983361da35204ead8931c3f65ad4
```

It contains 8k prompts generated with the Qwen2.5-3B tokenizer. Prompt selection is deterministic per task and seed, without replacement. The model/tokenizer itself is pinned to revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`.

### Official local generation

To remove the convenience mirror from the trust path, generate JSONL from a pinned NVIDIA/RULER checkout:

```bash
python -m pip install -e .[official-data]
python scripts/bootstrap_ruler.py
```

Then complete any source-data download steps required by the chosen upstream task families and run. The helper downloads only tokenizer/configuration files from the pinned model revision, not the model weights:

```bash
python scripts/prepare_official.py \
  --context-length 8192 \
  --num-samples 500 \
  --tasks niah_single_1,niah_multikey_1,niah_multiquery,vt,fwe,qa_1
```

Use those files with:

```bash
santapp-ruler run \
  --config configs/default_8k.yaml \
  --run-dir runs/8k-official-data \
  --set benchmark.data.source=local \
  --set benchmark.data.local_root=data/ruler_8k
```

The optional generator is pinned to NVIDIA/RULER commit:

```text
38da79d79519ef87aa46ae804f838e1eab7f86d7
```

The classic synthetic definitions are pinned because this custom attention backend runs inside a Hugging Face model process; it is not a vLLM/SGLang/TRT-LLM server backend.

### Grading

The vendored grader keeps RULER's task-family decisions:

- NIAH, variable tracking, common-word extraction, and frequent-word extraction use `string_match_all`.
- QA uses `string_match_part`.
- Matching is case-insensitive substring matching after RULER-style prediction cleanup.

No model-specific answer parser, regex, or hand-tuned normalization is added. Each backend directory also receives a RULER-style `summary.csv` and `submission.csv`.

The harness additionally reports an **unweighted mean over the selected tasks**. That aggregate is clearly labeled as a harness reporting choice, not another official RULER metric.

Re-grade existing predictions without rerunning the model:

```bash
santapp-ruler grade --run-dir runs/8k-default
```

## Output layout

```text
runs/8k-default/
├── config.resolved.yaml
├── runtime.pre_model.json
├── runtime.json
├── run_status.json
├── selected_prompts.jsonl
├── predictions/
│   ├── sdpa/
│   │   ├── niah_single_1.jsonl
│   │   ├── ...
│   │   ├── summary.csv
│   │   └── submission.csv
│   └── santapp/
│       ├── niah_single_1.jsonl
│       ├── ...
│       ├── summary.csv
│       └── submission.csv
├── per_example.csv
├── summary.csv
├── summary.json
└── summary.md
```

Prediction JSONL keeps RULER's core fields:

```json
{
  "index": 17,
  "uid": "niah_single_1:203:17",
  "task": "niah_single_1",
  "input": "...",
  "outputs": ["..."],
  "pred": "...",
  "others": {"id": 17, "task": "niah_single_1", "backend": "santapp"},
  "metrics": {"decode_kv_access_pct": 2.4, "total_seconds": 12.3}
}
```

## KV access metrics

All access percentages refer to **decode attention**, not dense prefill or the one-time clustering pass.

### `decode_kv_access_pct`

Draw-count K/V row accesses divided by the corresponding dense K/V row accesses:

```text
2 × (sampled token draws + exact recent tokens)
------------------------------------------------
             2 × total cached tokens
```

The counters are accumulated over every generated query head and layer before taking the ratio. Sampling is with replacement, matching the estimator; repeated draws therefore count repeatedly.

### `decode_read_equivalent_pct`

The notebook's metadata-inclusive convention:

```text
sampled/recent K+V rows + key-centroid rows
-------------------------------------------
             dense K+V rows
```

A centroid counts as one K-like vector, or half of a K/V pair. For the optional `santa` oracle proposal, every old K row used to construct the proposal is counted as KV-cache access, and sampled K rows are not counted twice when their V rows are fetched. The default `guided` mode reads only group metadata before fetching sampled K/V rows.

Both values are reported because “KV access” and “total attention-side read equivalent” answer different questions.

## Timing interpretation

The harness reports wall-clock, CUDA-synchronized:

- SDPA prefill, decode, and total time.
- SANTA++ dense prefill, clustering, sparse decode, and total time.
- Generated-token throughput.
- Peak allocated and reserved CUDA memory.

SANTA++ clustering is charged once per prompt. This is correct for an ordinary RULER example, where every prompt has a distinct prefill. The implementation is a readable PyTorch reference with Python-level per-head dispatch, not a fused kernel; use the harness for correctness, quality, access, and stable end-to-end comparisons, not as evidence that the current reference code is the final attainable kernel speed.

## Reproducibility

- Dataset repository and revision are pinned.
- Model weights and tokenizer revision are pinned.
- Official RULER source revision is recorded.
- Task selection is deterministic per task.
- The same selected prompt rows are used by both backends.
- SANTA++ sampling receives a deterministic seed derived from the example UID.
- MiniBatchKMeans uses the notebook's fixed `random_state=0` by default.
- Every run saves its fully resolved configuration and runtime package versions.
- Resume validates both the resolved result configuration and the exact selected-prompt manifest before appending.
- Greedy generation is used for both backends.

GPU reductions may still vary slightly across GPU architectures, drivers, and library builds.

## Current scope and known constraints

- Qwen2/Qwen2.5 architecture only (`model_type == "qwen2"`).
- Batch size 1 only for the SANTA++ custom cache.
- One GPU; no tensor parallelism.
- Dense prefill, sparse decode.
- No padding inside a prompt batch because prompts are processed individually.
- No fused sparse kernel yet.
- The default 8k mirror is tokenizer-specific. For another context length or model tokenizer, generate local official JSONL.
- Setting `generation.max_new_tokens_cap` below RULER's task budget creates a diagnostic run, not a standard benchmark result.

## Repository map

```text
src/santapp_ruler/
├── attention/
│   ├── minibatch_kmeans.py   # CUDA, scikit-learn-like MiniBatchKMeans
│   └── santapp.py            # patched Qwen attention and traffic counters
├── backends.py               # stock cached SDPA and SANTA++ generation
├── config.py                 # typed config and dotted overrides
├── data.py                   # task loading and deterministic selection
├── reporting.py              # official grading and summaries
├── runner.py                 # resumable benchmark loop
├── run_state.py              # config/manifest resume safety
└── ruler/
    ├── grader.py             # RULER metric functions/post-processing
    ├── provenance.py
    └── tasks.py
```

Algorithm and measurement details are expanded in [`docs/ALGORITHM.md`](docs/ALGORITHM.md). The output schema is described in [`docs/OUTPUTS.md`](docs/OUTPUTS.md).

## Validation status

This source release passes Python bytecode compilation, 14 unit tests, Ruff static checks, and an isolated wheel/sdist build. The release-building environment did not have the target RTX 5090/model runtime, so a full 8k CUDA benchmark was not fabricated. The `doctor`, `validate-data`, `fidelity`, and smoke commands are intentional target-machine gates before a long run.

## License and attribution

The harness is Apache-2.0. See `NOTICE` for attribution to NVIDIA/RULER and the supplied SANTA++ notebook. This project is not endorsed by NVIDIA.
