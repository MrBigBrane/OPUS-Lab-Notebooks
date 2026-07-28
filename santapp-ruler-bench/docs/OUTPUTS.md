# Output and metric schema

## Prediction row

Each `predictions/<backend>/<task>.jsonl` row contains:

- `index`: source RULER index.
- `uid`: stable `<task>:<source_position>:<index>` resume key; the source position prevents collisions in upstream files with repeated `index` values.
- `task`: selected task name.
- `input`: exact prompt passed to the tokenizer.
- `outputs`: RULER reference substring list.
- `pred`: decoded greedy continuation.
- `others`: compatibility metadata including `id`.
- `metrics`: timing, token count, memory, access, and per-example score.

## Core timing fields

- `prefill_seconds`
- `clustering_seconds`
- `decode_seconds`
- `total_seconds`
- `peak_allocated_gib`
- `peak_reserved_gib`

All timed regions synchronize CUDA at their boundaries.

## Core access fields

- `decode_dense_kv_vectors`
- `decode_kv_vectors_read`
- `decode_metadata_key_vectors_read`
- `decode_kv_access_pct`
- `decode_read_equivalent_pct`
- `mean_ess_over_samples`

A “vector” is one head-dimensional K or V row. Centroid metadata is K-like and appears only in the metadata-inclusive read-equivalent.

## Summary files

- `summary.md`: human-readable table.
- `summary.csv`: one row per backend/task plus selected-task aggregate.
- `summary.json`: structured equivalent and paired comparison.
- `per_example.csv`: flattened prediction metrics.
- `predictions/<backend>/summary.csv`: RULER-style task/score/null layout.
- `predictions/<backend>/submission.csv`: task, ID, prediction.
