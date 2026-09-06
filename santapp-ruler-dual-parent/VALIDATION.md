# Validation record

## Intended handoff state

The public attention backends are exactly:

```text
sdpa
santa
santapp
hierarchical
```

The two SANTA++ variants retain their previous names and implementations:

- `santapp`: global MiniBatchKMeans parents in raw post-RoPE key space.
- `hierarchical`: fixed contiguous positional parents.

They share actual post-RoPE-key leader/team construction, global whole-team Gumbel Top-K sampling, conditional selected-team inclusion correction, and the same exact-suffix policy. All prompt rows remain sampled. The first sparse decode query has zero exact rows; at zero-indexed decode step `t`, the `t` previously generated KV rows are exact.

The default model protocol remains:

```text
Qwen/Qwen2.5-7B-Instruct
a09a35458c702b33eeacc393d103063234e8bc28
chat_template with add_generation_prompt=True
```

## Source-preservation checks

The following implementation-preservation checks were performed:

- `attention/santapp.py` is byte-identical to the post-RoPE-key K-means handoff revision.
- `attention/minibatch_kmeans.py` is byte-identical to that revision; its SHA-256 is `b92649b181b3374ba415093e2e2ca4284f6f9c2df1e3311254903d157c8eac93`.
- `attention/teams.py`, the learned-parent team builder, is byte-identical to that revision.
- `attention/contiguous_teams.py` is byte-identical to the contiguous-window revision's team builder.
- `attention/hierarchical.py` is identical to the contiguous-window engine except that it imports the renamed `contiguous_teams` module, avoiding a collision with the learned-parent team builder.

## Automated suite

The merged source tree passes **87 CPU/static tests**, covering:

- the pinned model revision and Qwen instruction-template framing;
- all four backend names and configuration construction;
- both parent-policy geometries and validation rules;
- MiniBatchKMeans validity and fixed-seed determinism;
- direct post-RoPE key-space clustering input;
- contiguous parent spans, including a short sampled final parent;
- actual-key leader selection and complete prompt-row partitioning;
- generated-only exact-suffix semantics for both SANTA++ variants;
- tiny-Qwen end-to-end execution of both hierarchical variants;
- whole-team Gumbel Top-K and inclusion correction;
- reporting and Pareto logic;
- eight-shard owner-specific Kubernetes rendering and CephFS permission preparation;
- absence of personal identifiers and experiment-output bloat.

## Default indexed smoke matrix

The generated default Job contains eight shards:

```text
0:  8192 / sdpa
1:  8192 / santa
2:  8192 / santapp
3:  8192 / hierarchical
4: 32768 / sdpa
5: 32768 / santa
6: 32768 / santapp
7: 32768 / hierarchical
```

For a quick handoff smoke test, set `SMOKE_TASKS=niah_single_1`; this exercises eight generated-and-graded examples rather than all 104 context/backend/task combinations.

## Validation boundary

This build environment did not load the full pinned Qwen 7B checkpoint or execute the 8K/32K CUDA matrix. The prior two source revisions were independently GPU-smoked by the recipient, but this merged repository should still receive one eight-shard, single-prompt GPU smoke run before it is used for larger experiments. The smoke should be followed by standalone regrading and `scripts/grade_and_plot.py` to verify cross-shard aggregation and both Pareto plots.
