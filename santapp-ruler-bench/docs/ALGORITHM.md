# SANTA and SANTA++ reference algorithms

## Shared decode setup

Both reference methods patch Qwen2/Qwen2.5 attention only while their generation call is active. Dense prefill uses the Hugging Face SDPA integration. The custom cache stores post-RoPE K and unmodified V with shape `[n_kv_heads, tokens, head_dim]`.

The final prompt token is re-fed so the first approximate attention call computes the same next-token position as stock cached generation. For SANTA++, re-feeding does not make it exact: its K/V row is already part of the frozen prompt clusters.

## SANTA

For each decode query head and the entire current cache:

```text
scores       = qK^T / sqrt(d)
probability  = softmax(scores)
indices      ~ IID Categorical(probability), with replacement, S draws
output       = mean(V[indices])
```

Sampling from the exact attention distribution makes the sample mean an unbiased Monte Carlo estimator of the dense attention output.

SANTA has no fixed exact window, no growing exact window, no clusters, and no centroid metadata. Generated tokens simply become part of the cache sampled on later steps. Its only algorithm parameter is `santa.samples_per_head`.

## SANTA++ prompt decomposition

For a prompt with `T` tokens:

```text
sample_end = T
clustered token positions = [0, sample_end - 1] = [0, T - 1]
fixed exact prompt tail = 0
initial growing exact suffix = 0
```

After dense prefill and clustering, the cache is trimmed to `T-1` tokens and the final prompt token is re-fed. Its KV row returns at position `T-1`, which is already represented by the frozen prompt clusters. The exact region at the first approximate decision therefore has length zero. It grows by one after every generated token while `sample_end` remains fixed.

This intentionally leaves generated tokens unclustered until a future cluster-update policy is defined.

## SANTA++ probe queries

The probe policy is deterministic:

```text
probe positions = [T - P, T - P + 1, ..., T - 1]
```

`P=santapp.probe_queries`, default 64. These are the final `P` prompt query tokens, including the final prompt token. There is no random selection from a larger region.

For each layer and KV head:

1. collect all query heads sharing that KV head;
2. apply Qwen RoPE to the captured query projections;
3. compute scaled dot products between every prompt key and every selected probe query;
4. concatenate features across probe positions and sharing query heads;
5. standardize each feature with population mean and standard deviation; and
6. cluster the token fingerprints with the CUDA-capable MiniBatchKMeans reference implementation.

The nominal cluster count is:

```text
min(sample_end, max(2, sample_end // group_size))
```

Only active clusters are retained. Each summary stores membership layout, cluster length, and the arithmetic mean key.

## SANTA++ decode proposal

For a cluster `g`, cluster size `n_g`, mean key `k_bar_g`, and query `q`:

```text
p(g | q) ∝ n_g exp(k_bar_g · q / sqrt(d))
```

For each of `S` draws:

1. draw a cluster IID with replacement from `p(g | q)`;
2. draw one token uniformly from that cluster; and
3. use token proposal probability

```text
r(j | q) = p(g(j) | q) / n_g(j).
```

The sampled-prompt score is corrected with:

```text
log_weight_j = q · k_j / sqrt(d) - log r(j | q).
```

After a shared numerical-stability shift, the clustered-prompt Monte Carlo numerator and denominator use the usual `1/S` factor. Exact generated-suffix exponentials are added to the same numerator and denominator.

## MiniBatchKMeans defaults

The implementation retains the existing defaults:

- `batch_size=4096`
- `n_init=1`
- `max_iter=100`
- `tol=0`
- `max_no_improvement=10`
- `init_size=null`
- `reassignment_ratio=0.01`
- `random_state=0`

It uses NumPy `RandomState` for stochastic choices, greedy k-means++ initialization, with-replacement mini-batches, cumulative-count online means, low-count center reassignment, and exponentially weighted inertia stopping. Reduction order can still differ across devices and software builds.

## GQA-aware access accounting

The accounting unit is one head-dimensional K-like or V-like vector. It covers decode attention only; prefill clustering time is reported separately and its one-time data movement is not folded into the decode percentage.

For one layer, decode token, and KV head:

- `N`: current cache length;
- `G`: query heads sharing the KV head;
- `S`: draws per query head;
- `U`: unique sampled token indices in the union across the `G` query heads;
- `R`: SANTA++ exact growing-suffix length, which is zero on the first approximate call and then contains only generated tokens appended beyond the prompt boundary;
- `M`: active centroid count.

### Dense denominator

```text
GQA-aware dense denominator = 2N
naive dense denominator     = 2NG
```

The GQA-aware denominator reflects one shared K cache and one shared V cache for the KV head. The naive denominator incorrectly treats each query head as owning an independent KV cache.

### SANTA++ numerator

```text
GQA KV       = 2(U + R)
GQA centroid = M
GQA total    = 2(U + R) + M

naive KV       = 2G(S + R)
naive centroid = MG
naive total    = 2G(S + R) + MG
```

The GQA metric deduplicates repeated samples within a query head and overlapping samples across all query heads sharing the KV head. The exact suffix and centroid summaries are shared once at the KV-group level.

### SANTA numerator

```text
GQA KV   = N + U
naive KV = G(N + S)
centroid = 0
```

All K rows are needed to form the exact categorical proposal. Only sampled V rows are needed for the estimator.

### Aggregation

Per-example trackers retain raw vector counts. Task and run summaries sum numerators and denominators first and then compute percentages. They do not average per-example percentages.

These values are theoretical logical-access estimates. Cache-line effects, tensor-core scheduling, L2 reuse, coalescing, and actual HBM transactions require kernel profiling and are outside this metric.
