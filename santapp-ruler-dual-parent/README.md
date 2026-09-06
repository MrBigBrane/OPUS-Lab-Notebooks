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

The bundled GPU MiniBatchKMeans implementation clusters all prompt keys globally and directly in unnormalized post-RoPE key coordinates. It uses no probe queries, projection, standardization, L2 normalization, positional windows, cache sorting, or cache permutation.

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

## Local setup

Install a CUDA-enabled PyTorch build appropriate for the machine first, then install the repository:

```bash
conda create -n santapp-ruler python=3.12 -y
conda activate santapp-ruler
python -m pip install --upgrade pip wheel "setuptools<82"
# Install the desired CUDA PyTorch wheel here.
python -m pip install -e '.[dev,plot]'
pytest -q
santapp-ruler doctor
```

Validate one prompt without loading model weights:

```bash
santapp-ruler validate-data \
  --config configs/smoke.yaml \
  --context-length 8192 \
  --tasks niah_single_1 \
  --prompts-per-task 1
```

Run all four backends on one example:

```bash
santapp-ruler run \
  --config configs/default.yaml \
  --context-length 8192 \
  --tasks niah_single_1 \
  --prompts-per-task 1 \
  --backends sdpa santa santapp hierarchical \
  --run-dir runs/example
```

## Docker and Nautilus

```bash
docker build -t REGISTRY/PROJECT/santapp-ruler:dual-parent-v1 .
docker push REGISTRY/PROJECT/santapp-ruler:dual-parent-v1
```

Render owner-specific resources:

```bash
python scripts/render_nautilus_manifests.py \
  --owner REPLACE_WITH_YOUR_SLUG \
  --image REGISTRY/PROJECT/santapp-ruler:dual-parent-v1 \
  --namespace ucsb-opus-lab \
  --output k8s/generated/REPLACE_WITH_YOUR_SLUG
```

The default indexed Job has eight shards: two contexts by four backends. Each shard loads one model and evaluates one deterministic example from each selected task. Tasks can be reduced to `niah_single_1` for a software smoke test. Full instructions are in [docs/NAUTILUS.md](docs/NAUTILUS.md).

After copying results locally, rebuild reports and Pareto plots with:

```bash
python scripts/grade_and_plot.py \
  ./downloaded-results/REPLACE_WITH_YOUR_SLUG/smoke
```

## Repository layout

```text
configs/                  Default and smoke configurations
docs/                     Algorithm, prompting, cluster, and portability notes
k8s/example/              Generic rendered example; regenerate per colleague
scripts/                  Cross-platform manifest, shard, and report tools
src/santapp_ruler/        Benchmark package and four supported backends
tests/                    CPU/static regression tests
```

## Validation boundary

CPU/static tests cover both parent policies, exact-suffix semantics, post-RoPE K-means input, MiniBatchKMeans behavior, contiguous-span partitioning, leader/team construction, sampling utilities, prompt framing, reporting, Kubernetes rendering, and repository hygiene. A real GPU smoke run is still required on each target software/hardware environment before large experiments.
