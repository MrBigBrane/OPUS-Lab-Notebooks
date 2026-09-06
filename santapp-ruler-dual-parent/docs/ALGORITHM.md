# SANTA++ hierarchical whole-team sampling with two parent policies

The repository exposes two variants of the same downstream whole-team estimator:

```text
santapp      = global MiniBatchKMeans parents in raw post-RoPE key space
hierarchical = fixed contiguous positional parents
```

## Shared dense prefill and key representation

Dense prefill runs through the temporary Qwen attention patch and stores every layer's actual cached prompt keys after RoPE has been applied. Both parent policies and all leader/team geometry use these cached post-RoPE keys cast to float32.

## Parent policy A: `santapp`

For each layer and KV head, all `N` prompt keys are clustered globally. With nominal parent size `P`:

```text
K_parent = min(N, max(2, floor(N / P)))
```

Thus `P=16` targets an average of roughly 16 tokens per learned parent; actual parent sizes vary. The feature for token `i` is exactly its unnormalized cached post-RoPE key `k_i`.

The bundled `SklearnLikeTorchMiniBatchKMeans` defaults are:

```yaml
batch_size: 4096
n_init: 1
max_iter: 100
tol: 0.0
max_no_improvement: 10
init_size: null
reassignment_ratio: 0.01
random_state: 0
```

This path uses no probe queries, probe fingerprints, feature standardization, per-key normalization, local/windowed K-means, token sorting, or KV-cache permutation.

## Parent policy B: `hierarchical`

For prompt length `N`, parent size `P`, and parent index `j`:

```text
parent_j = [j * P, min((j + 1) * P, N))
```

The original token order is retained. If `N` is not divisible by `P`, the final parent is shorter and is handled by the same sampled-team machinery. It is not an exact tail.

## Shared leaders and teams

For a nonempty parent containing actual post-RoPE keys `K = {k_1, ..., k_m}`:

1. Compute `mu = (1/m) sum_i k_i`.
2. Select the actual key minimizing `||k_i - mu||_2^2` as the first leader.
3. Repeatedly select the unchosen key maximizing its distance to its nearest already selected leader.
4. Stop after `min(R, m)` leaders.
5. Assign every parent token to its nearest selected leader in post-RoPE key space.
6. Force every selected leader to own its own row, a deterministic equal-distance tie resolution that prevents empty teams.

Each flattened team stores its actual leader key, cardinality, and original KV-cache member indices. In both variants, the flattened teams partition every prompt row exactly once without reordering the cache.

## Shared whole-team Gumbel Top-K

For decode query `q`, team `g` with leader `l_g` and size `n_g` receives routing logit:

```text
phi_g = log(n_g) + <l_g, q> / sqrt(d)
```

The nominal team size is `P / R`, and the requested team count is:

```text
samples_per_head / (P / R)
```

capped by the number of active teams. Weighted Gumbel Top-K samples distinct teams globally. Every token belonging to a selected team is read. The selected team's conditional inclusion probability corrects its token numerator and denominator contributions. Because team sizes vary, `samples_per_head` is a nominal token budget rather than an exact row count.

## Shared exact suffix

For both variants, `sample_end` is the full prompt length `N`; all prompt rows remain sampled. After dense prefill, the custom cache is trimmed to `N-1` and the final prompt token is re-fed as the first sparse query.

```text
first sparse decode query  -> 0 exact generated KV rows
second sparse decode query -> 1 exact generated KV row
third sparse decode query  -> 2 exact generated KV rows
...
```

At zero-indexed decode step `t`, exactly the `t` previously generated rows after `sample_end` are evaluated exactly. There is no exact prompt window in either variant.

## Runtime-identifying metadata

The K-means variant records:

```text
backend: santapp
parent_policy: global_minibatch_kmeans_raw_post_rope_keys
parent_clustering_used: true
key_space_stage: post_rope
```

The contiguous variant records:

```text
backend: hierarchical
parent_policy: fixed_contiguous_spans_in_original_token_order
parent_clustering_used: false
partial_parent_policy: ordinary_short_sampled_parent
```

Both record:

```text
prompt_exact_tail_tokens: 0
initial_generated_exact_tokens: 0
exact_token_policy: generated_suffix_only_grows_from_zero
```
