# SANTAPP RULER benchmark harness

A compact research repository for comparing four decode-attention backends on the 13 classic synthetic RULER tasks:

- `sdpa`: dense PyTorch/Hugging Face SDPA reference.
- `santa`: token-IID SANTA sampling from the exact attention distribution.
- `santapp`: SANTA++ hierarchical whole-team Gumbel Top-K sampling with global MiniBatchKMeans parents in raw post-RoPE key space.
- `hierarchical`: the same hierarchical whole-team sampler with fixed contiguous positional parents.

The two SANTA++ variants differ only in how parent membership is constructed. They share actual-key leader selection, team assignment, global team routing, Gumbel Top-K selection, whole-team expansion, inclusion correction, and generated-only exact-suffix behavior.

The default model is the pinned `Qwen/Qwen2.5-7B-Instruct` snapshot `a09a35458c702b33eeacc393d103063234e8bc28`, framed through its tokenizer-owned instruction template. Default target contexts are 8,192 and 32,768 tokens.

## Parent policies

### `santapp`: global post-RoPE-key K-means

For every layer and KV head, dense prefill stores the actual RoPE-applied prompt keys. With prompt length `N` and nominal parent size `P`, the parent count is:

```text
min(N, max(2, floor(N / P)))
```

The bundled GPU MiniBatchKMeans implementation clusters all prompt keys globally and directly in unnormalized post-RoPE key coordinates. It uses no probe queries, projection, standardization, L2 normalization, positional windows, cache sorting, or cache permutation during parent construction. Grouped decode later packs a separate team-major cache without changing parent membership.

### `hierarchical`: contiguous spans

Parent `j` is the original-order span:

```text
[j * P, min((j + 1) * P, N))
```

With `P=16`, the parents are `0..15`, `16..31`, and so on. A short final span is an ordinary sampled parent, never an exact window.

### Shared team sampler

Inside either kind of parent:

1. Compute the arithmetic mean of its actual post-RoPE keys.
2. Choose the actual key nearest the mean as leader 1.
3. Choose later leaders by max-min farthest-first traversal.
4. Assign every parent member to its nearest leader, with each leader forced to own itself.
5. Flatten all teams globally. At decode, score team `g` by `log(team_size) + <leader, query>/sqrt(d)`, sample distinct teams with weighted Gumbel Top-K, read every member of selected teams, and apply the selected-team inclusion correction.

All prompt rows remain sampled. The first sparse decode query has zero exact KV rows; after one generated token there is one exact generated row, after two there are two, and so on. No prompt tail is exact. See [docs/ALGORITHM.md](docs/ALGORITHM.md).

## Prompting policy

The complete prepared RULER example is supplied as one `user` turn through the pinned Qwen tokenizer's `apply_chat_template(..., add_generation_prompt=True)`. The harness adds no custom task hints. Every backend receives identical rendered token IDs, and over-budget examples fail rather than being silently truncated. See [docs/PROMPTING.md](docs/PROMPTING.md).

The convenience RULER mirrors were generated for approximate target lengths using a Qwen2.5-3B tokenizer. They contain raw text and answers, not pre-tokenized inputs. This harness always re-tokenizes with the configured pinned Qwen2.5-7B tokenizer after final chat framing.

## R007: batched independent K-means fits (current release)

The native K-means path now processes **eight independent layer/KV-head fits per
GPU batch** (`santapp.kmeans.fit_batch_size: 8`), instead of fitting all heads
sequentially. Each fit keeps its own greedy initialization, minibatch updates,
counts, RNG and stopping state. The minibatch token count remains 4096. Both
parent methods retain the R006 grouped decoder and all nominal sample budgets
through 4096. Numerical prefill routines and decoder kernels are unchanged.

Use [BATCHED_PREFILL](docs/BATCHED_PREFILL.md) for scheduling/algorithm details and
[NAUTILUS_BATCHED_PREFILL](docs/NAUTILUS_BATCHED_PREFILL.md) for the **A10-only,
parallelism-eight** rollout. The full-model canary is now K-means at 32k/S4096.
Local CPU validation is not a CUDA pass or a guarantee of >40% utilization or
24 GB memory safety; the included GPU/model gates must pass on the target stack.

The R006 section below is historical decode documentation; use the R007 renderer
and runbook for current deployment.

## R006: grouped Triton decode through 4096 nominal tokens

Both team methods now explicitly select `decode_backend: grouped_triton` in the
bundled default/smoke/Triton configs. Native prefill is unchanged. The new decoder
uses grouped GQA routing, GPU batched top-(K+1), team-major packed K/V, compact
prefixes and all-head split-KV attention. It accepts K through 1024 rather than
v16's fixed K31. The six smoke budgets are 128, 256, 512, 1024, 2048 and 4096
nominal tokens per query head (32 through 1024 teams at P16/R4).

**CPU checks passed; the new CUDA kernels and full model have not executed in the
authoring environment. Run the supplied ordered compile/GPU/model gates before
calling this a GPU-validated release. No utilization threshold is guaranteed.**
See [VALIDATION.md](VALIDATION.md).

Use [docs/GROUPED_DECODE.md](docs/GROUPED_DECODE.md) for layout, RNG/numerical
semantics, memory cost and diagnostics, and
[docs/NAUTILUS_GROUPED_DECODE.md](docs/NAUTILUS_GROUPED_DECODE.md) for the 24-case
one-GPU-at-a-time rollout. The separate personalized ZIP contains ready-to-apply
YAMLs and Windows commands. No second source repository is required.

`--decode-backend torch` restores only the old decoder while retaining accelerated
prefill. `configs/torch.yaml` selects both Torch preparation and Torch decode.
Legacy configs omitting the new decode field resolve to Torch rather than being
silently reinterpreted. New current configs explicitly request grouped decode.
Dense SDPA and token-IID SANTA are unchanged.

## Local setup and ordinary runs

Use Python 3.11/3.12 and a working, matched CUDA PyTorch/Triton installation
(Ampere or newer; Linux/WSL2). The deployment image below supplies that stack.
Do not independently upgrade Triton to an arbitrary latest version.

```bash
python -m pip install -e '.[dev,plot]'
pytest -q
santapp-ruler doctor

santapp-ruler run \
  --config configs/default.yaml \
  --context-length 8192 \
  --tasks niah_single_1 --prompts-per-task 1 \
  --samples-per-head 512 \
  --backends sdpa santa santapp hierarchical \
  --run-dir runs/example-native
```

The two team backends use Triton without an extra flag. `--samples-per-head`
remains general; it is not fixed to 128. With default P=16/R=4, use a positive
multiple of four through 4096 for grouped decode. Both 8k and 32k context budgets remain supported. A 32k RULER
budget includes generation room; actual framed prompts are not forcibly 32768
input tokens. The original per-task generation lengths are unchanged.

The CLI, local smoke scripts, image, and generated GPU Jobs default to
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. For an external Python/notebook
caller, set it **before importing PyTorch**. Importing this library alone does not
change your environment. An explicitly supplied allocator configuration is respected.

## Older prefill-only Docker/Nautilus workflow (reference)

For this release's grouped decoder, use the new staged instructions above. The
older renderer below intentionally retains its original Torch-decode behavior.

```bash
bash scripts/docker_triton_smoke.sh build --profile cluster \
  --image REGISTRY/PROJECT/santapp-ruler:r005-ampere
# Building is local. Upload the image before any cluster submission:
docker push REGISTRY/PROJECT/santapp-ruler:r005-ampere

python scripts/render_nautilus_manifests.py \
  --owner REPLACE_WITH_YOUR_SLUG \
  --image REGISTRY/PROJECT/santapp-ruler:r005-ampere \
  --namespace ucsb-opus-lab \
  --output k8s/generated/REPLACE_WITH_YOUR_SLUG \
  --parallelism 4
```

Default output filenames and matrix shape are retained: PVC, ConfigMap, an
8-completion Job (two contexts x four backends), and a no-GPU copy pod. The
renderer includes equal CPU/RAM requests and limits, init-container resources,
per-pod JIT scratch, and expandable segments. Do not reuse another colleague's
owner/PVC/results or overwrite an old image tag.

For the targeted A10 acceptance matrix, add:

```text
--staged-smoke --gpu-product NVIDIA-A10
--backends santapp hierarchical --samples-per-head 128 512
--tasks niah_single_1 --prompts-per-task 1 --max-new-tokens 2
```

That generates separate kernel and HF-prefetch Jobs followed by eight real-model
cases, with four GPU workers at once. Apply them in `APPLY_ORDER.txt` order.
**Two generated tokens are only a software smoke; remove that override for RULER
accuracy evaluation.** Details, Lens instructions, resource tuning, image publishing,
and result recovery are in [docs/NAUTILUS.md](docs/NAUTILUS.md).

For local RTX 5090/WSL testing, use `--profile local-5090` instead of the cluster
profile. See [docs/TRITON_PREFILL.md](docs/TRITON_PREFILL.md).

After copying results locally:

```bash
python scripts/grade_and_plot.py ./downloaded-results/REPLACE_WITH_YOUR_SLUG/smoke
```

## Validation and known limits

The predecessor R004 numerical implementation was reported to complete the A10
8k/32k x S128/S512 x both-parent-policy smoke **8/8** after enabling expandable
segments. This release preserves those numerical kernels and records the exact
scope in [docs/VALIDATION_HISTORY.md](docs/VALIDATION_HISTORY.md).

**A10 + Triton 3.6.0 has a known synthetic D=7 k-means assignment failure.** It is
not the Qwen D=128 workload. The isolated D=7 accuracy diagnostic is an explicit
expected failure only on that reported platform; mandatory D=128 tests and all
other checks still fail normally. No kernel arithmetic was changed to hide it.
See [docs/KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md).

Expandable segments is not a no-OOM guarantee. Larger S, longer generation,
other prompts, and long-running multi-prompt workloads need their own stress test.
Do not infer full-suite accuracy or memory safety from the two-token smoke.

## Repository layout

```text
configs/                  Default/native, smoke, and explicit Torch reference
src/santapp_ruler/         Same four backends; native prefill is bundled
scripts/                  Original CLI helpers plus Docker/compiler/GPU smoke tools
k8s/example/              One generic example; regenerate per colleague
docs/                    Algorithm, deployment, known issues, and provenance
tests/                   CPU checks plus opt-in native GPU checks
```

[VALIDATION.md](VALIDATION.md) records release-local checks separately from the
user-reported A10 run. A new image build still requires recipient-side validation.
