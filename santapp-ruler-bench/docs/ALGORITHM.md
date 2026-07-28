# SANTA++ reference path in this harness

## Prompt decomposition

For a prompt of length `T` and exact recent window `W`, the sampled old prefix ends at:

```text
sample_end = T - 1 - W
```

The final prompt token is temporarily removed from the custom cache after clustering and re-fed as the first sparse query, matching the supplied notebook. At the first generated-token decision, the exact region therefore contains `W + 1` prompt tokens. It grows by one token at every later step.

## Fingerprint construction

For each layer and KV head:

1. Select `P` query positions from the final quarter of the prompt, excluding the last token.
2. Include every query head that shares the KV head.
3. For each old key, concatenate scaled key/query dot products over those probe queries.
4. Normalize every fingerprint feature by population mean and standard deviation.
5. Cluster token fingerprints with GPU MiniBatchKMeans.

The number of requested clusters is:

```text
max(2, sample_end // group_size)
```

and is capped by the number of old-prefix tokens.

## Guided proposal

For group `g`, size `n_g`, centroid key `k_bar_g`, and decode query `q`:

```text
p(g | q) ∝ n_g exp(k_bar_g · q / sqrt(d))
```

A group is sampled with replacement, then a token is sampled uniformly inside that group. The token proposal probability is:

```text
r(j | q) = p(g(j) | q) / n_g(j)
```

The sampled old-prefix contribution uses self-normalized importance weights:

```text
w_j = exp(k_j · q / sqrt(d)) / r(j | q)
```

with the usual `1/S` Monte Carlo factor. Exact recent-token exponentials are added to the same numerator and denominator after a shared numerical-stability shift.

## MiniBatchKMeans fidelity choices

The CUDA implementation keeps the supplied notebook defaults:

- NumPy `RandomState` for stochastic choices.
- Greedy k-means++ initialization with `2 + floor(log(K))` local trials.
- Mini-batches sampled with replacement.
- Cumulative-count online centroid means.
- Low-count center reassignment.
- Exponentially weighted inertia early stopping.
- `batch_size=4096`, `n_init=1`, `max_iter=100`, `tol=0`, `max_no_improvement=10`, and `reassignment_ratio=0.01`.

CPU/GPU reduction order can still produce small numerical differences from scikit-learn.

## Access accounting boundary

The reported percentages cover decode attention rows only. They do not pretend that clustering is free: clustering time is reported separately, but its one-time fingerprint and k-means memory traffic is not folded into the decode KV percentage.

This separation matches the intended claim: clustering is a prefill-time structure that is amortized across subsequent decode tokens.

## Notebook traffic equivalence

For the default guided mode, `decode_read_equivalent_pct` is the exact,
per-head/per-step accumulated version of the notebook estimate

```text
(number of key centroids / 2 + sampled K/V pairs + exact K/V pairs)
-------------------------------------------------------------------
                         dense K/V pairs
```

Unlike the notebook's closed-form average over a fixed generation length, the
harness counts the actual cache length at every generated token and naturally
handles early EOS. `decode_kv_access_pct` excludes centroid metadata and reports
only K/V-cache vector accesses.
