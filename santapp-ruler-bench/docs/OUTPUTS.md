# Output and metric schema

## Prediction rows

Each `predictions/<backend>/<task>.jsonl` row contains:

- `index`: source RULER index;
- `uid`: stable `<task>:<source_position>:<index>` resume key;
- `task`: RULER task name;
- `input`: exact prompt;
- `outputs`: reference strings;
- `pred`: decoded greedy continuation;
- `others`: compatibility metadata; and
- `metrics`: timing, access, memory, algorithm, seed, and per-example score fields.

## Core timing and size fields

- `prompt_tokens`
- `generated_tokens`
- `max_new_tokens`
- `prefill_seconds`
- `clustering_seconds`
- `decode_seconds`
- `total_seconds`
- `custom_cache_gib`
- `cluster_summary_gib`
- `peak_allocated_gib`
- `peak_reserved_gib`

CUDA timing regions synchronize at their boundaries. SANTA and SDPA report zero clustering time.

## Primary access fields

The principal result is:

- `decode_gqa_total_access_pct`

Its separately reported components are:

- `decode_gqa_kv_access_pct`
- `decode_gqa_centroid_access_pct`

The explicitly optimistic GQA-unaware convention is:

- `decode_naive_total_access_pct`
- `decode_naive_kv_access_pct`
- `decode_naive_centroid_access_pct`

A vector is one head-dimensional K row, V row, or K-like centroid row. Centroids are zero for SDPA and SANTA.

## Raw access counts

- `decode_attention_head_calls`
- `decode_gqa_group_calls`
- `decode_dense_gqa_kv_vectors`
- `decode_dense_naive_kv_vectors`
- `decode_gqa_kv_vectors_read`
- `decode_naive_kv_vectors_read`
- `decode_gqa_centroid_key_vectors_read`
- `decode_naive_centroid_key_vectors_read`
- `decode_gqa_total_vectors_read`
- `decode_naive_total_vectors_read`

These raw counts are summed before aggregate percentages are calculated.

## Supporting access diagnostics

- `mean_sampled_token_draws_per_head_call`
- `mean_unique_sampled_tokens_per_gqa_group_call`
- `mean_exact_tokens_per_gqa_group_call`
- `mean_total_tokens_per_head_call`


## SANTA fields

- `samples_per_head`
- `exact_window_tokens` — always zero

SANTA's access percentage includes the full K scan and sampled V-row union.

## SANTA++ fields

- `clustered_prompt_tokens`
- `prompt_exact_tail_tokens` — always zero
- `initial_growing_exact_tokens` — zero; the exact suffix begins only after generation starts
- `samples_per_head`
- `group_size`
- `probe_queries`
- `probe_policy` — `last_prompt_tokens`
- `nominal_clusters_per_kv_head`

## Summary files

- `summary.md`: human-readable backend/task table, with GQA total first;
- `summary.csv`: one row per backend/task plus selected-task aggregate;
- `summary.json`: structured task and aggregate results, plus SANTA/SDPA and SANTA++/SDPA comparisons;
- `per_example.csv`: flattened prediction and metric fields;
- `predictions/<backend>/summary.csv`: RULER-style task/score/null layout; and
- `predictions/<backend>/submission.csv`: task, source ID, and prediction.

The selected-task mean is an unweighted mean across the chosen tasks. It is a harness convenience rather than a new RULER metric.
