# Attention methods and accounting

This document describes the reference algorithms implemented in
`src/santapp_ruler/attention/`. The implementation is intended for controlled
RULER experiments and metric collection; it is not a fused production kernel.

## Shared model and cache behavior

The harness loads Qwen2.5-3B-Instruct with stock SDPA for prefill. RoPE is
applied before cached keys enter the custom decode cache. Values are cached
without RoPE. Greedy generation is used by default.

For every SANTA++ method, the full prompt prefix is assigned to the frozen
sampling structure. There is no exact prompt-tail window. Tokens generated
after prefill form a deterministic exact suffix, so the sampled prefix remains
fixed while the suffix grows during decode.

The default model has grouped-query attention. Sampling decisions are made per
query head, but logical K/V traffic is unioned across query heads sharing a K/V
head before the primary access percentage is computed.

## Dense SDPA (`sdpa`)

The dense baseline uses the model's stock cached scaled-dot-product attention.
Its logical K/V access is 100% by definition. It supplies a correctness and
runtime reference, not a custom-cache approximation.

## Standalone SANTA (`santa`)

For each query head, standalone SANTA evaluates exact attention probabilities
for the sampled prefix, draws `S` token indices IID with replacement from that
exact distribution, and averages sampled values. Because the proposal equals
the target attention distribution, no importance correction is required.

This is an algorithmic reference: computing the exact proposal is not itself a
sparse routing operation.

## Parent construction for SANTA++

SANTA++ clusters prompt keys independently for each layer and K/V head.
Clustering features are probe-query fingerprints: the final
`probe_queries` prompt queries score every cached key, and those score vectors
are clustered with MiniBatchKMeans. The nominal number of parents is derived
from the prompt length and configured `group_size` or `parent_size`.

A parent summary stores packed member indices, lengths, starts, and a key
centroid. Empty nominal clusters are skipped. Parent probability logits use

```text
log(parent_size) + <routing_key, query> / sqrt(head_dim)
```

where the routing key is a centroid for non-hierarchical methods.

## Centroid-guided SANTA++ (`santapp`, mode `guided`)

For each of `S` draws, the method samples a parent with replacement from the
centroid-guided parent distribution and then samples one member uniformly from
that parent. The token proposal probability is

```text
p(parent | query) / actual_parent_size.
```

The sampled softmax numerator and denominator are self-normalized with the
inverse token proposal probability. The checked-in default uses nominal parent
size `B=16`.

## Whole-parent sampling (`santapp_gumbel_topk`, mode `gumbel_cluster`)

This method selects distinct parents using weighted Gumbel top-k without
replacement. The requested number of parents is
`samples_per_head / group_size`, capped by the number of active parents. Every
token in each selected parent is fetched.

For a selected parent, the observed leave-one-out threshold gives the
conditional inclusion probability

```text
P(parent selected | other priorities)
  = 1 - exp(-exp(parent_logit - threshold)).
```

Each selected parent's token contributions are divided by that inclusion
probability. This is a Horvitz-Thompson population-sum correction, so the
corrected sum is **not** divided by the number of selected parents.

## Hierarchical team construction

The hierarchical methods reuse the parent labels produced in probe-fingerprint
space, then split each nonempty parent in actual RoPE-applied key space.

For a parent containing `n` keys, up to `R` actual-key representatives are
chosen:

1. choose the key nearest the parent arithmetic mean;
2. repeatedly choose the key maximizing distance to its nearest selected
   representative;
3. assign every parent member to its nearest representative.

When `n < R`, every key becomes a representative. Exact duplicate-key ties are
resolved deterministically so every selected representative owns a nonempty
team. The resulting teams partition the prompt prefix exactly once. Team
routing logits use team cardinality and the actual representative key:

```text
log(team_size) + <leader_key, query> / sqrt(head_dim).
```

The default hierarchy uses nominal parent size `P=16` and `R=4`
representatives, giving nominal team size four while allowing natural variation
from parent and Voronoi assignment sizes.

## Hierarchical team sampling (`team_sampler`, mode `team_sampler`)

For each of `S` draws, this method samples a team with replacement from the
leader-key distribution, then samples one team member uniformly. Its token
proposal probability is

```text
p(team | query) / actual_team_size.
```

The estimator applies the corresponding token-level importance correction.
This method reads actual leader keys for routing and sampled token K/V rows for
the estimator.

## Whole-team sampling (`team_gumbel_topk`, mode `team_gumbel_topk`)

This method selects distinct teams with weighted Gumbel top-k and fetches every
member of each selected team. The requested team count is

```text
samples_per_head / (parent_size / representatives_per_parent)
```

and is capped by the number of active teams. Team contributions use the same
leave-one-out conditional inclusion-probability correction as whole-parent
sampling. The corrected population sum is not divided by the selected-team
count.

## Importance-sampled softmax combination

For IID token proposals, sampled terms are divided by their token proposal
probability and by the number of IID draws. For without-replacement whole-unit
proposals, sampled terms are divided by the selected unit's conditional
inclusion probability and are not divided by the number of selected units.
Exact generated-suffix terms are included without sampling correction. A common
maximum is subtracted before exponentiation for numerical stability.

## Logical access metrics

The primary metric is

```text
(GQA-unioned sampled K/V rows + GQA-unioned routing-key rows)
----------------------------------------------------------- × 100
                 dense GQA K/V rows
```

Routing keys are centroids for parent methods and actual leader keys for team
methods. Centroid rows are retained as a subset field for compatibility and
inspection; they are not added a second time. The harness also reports:

- sampled K/V access alone;
- routing-key access alone;
- centroid subset access;
- naive per-query-head K/V and routing accounting;
- raw vector-row numerators and denominators;
- worst-case and expected quantities where the implementation can define them;
- cluster/team counts, size statistics, and inclusion-probability statistics.

These quantities count logical key/value vectors read by the reference
algorithm. They do not measure cache-line traffic, coalescing, on-chip reuse,
kernel-launch overhead, or end-to-end hardware bandwidth.

## Determinism and reproducibility

Prompt selection, MiniBatchKMeans, and stochastic sampling are seeded. Exact
bitwise agreement across GPU architectures is not guaranteed. Every Kubernetes
shard records the matrix hash, selected UIDs, runtime provenance, and a code
fingerprint so incompatible shards cannot be silently merged.

## Additional experimental modes

The source retains `oracle_token`, `uniform`, and deterministic `topk` modes for
research and unit testing. They are not among the six standard backends or the
default Kubernetes matrix. Custom variants can expose them under a unique name
in `santapp_variants`.
