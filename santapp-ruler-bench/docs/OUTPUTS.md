# Output and metric reference

## Local run directory

A normal run contains:

```text
RUN_DIR/
  config.resolved.yaml
  run-metadata.json
  predictions/<backend>/<task>.jsonl
  predictions/<backend>/summary.csv
  summary.json
  summary.csv
  summary.md
  per_example.csv
  comparisons.csv
```

Resume state is derived from prediction UIDs. A record is written only after a
prompt completes.

## Prediction records

Each JSONL record contains the task/index/UID, references, prediction, and a
`metrics` object. Local configs may retain the full input. Kubernetes configs
set `save_full_prompts=false` and `save_prediction_inputs=false` so compact
shared-storage records do not contain prompt inputs.

Important identity fields include:

- `backend`, `mode`, `sampling_scheme`, and `importance_correction`;
- `samples_per_head` and `nominal_sample_budget_per_head`;
- `group_size`, `parent_size`, `representatives_per_parent`, and
  `nominal_team_size` when applicable;
- `probe_queries`, `probe_policy`, and routing-key type;
- model head counts, head dimension, and prompt/generated token counts.

Timing fields include prefill, clustering, decode, and total seconds. They are
wall-clock measurements around the Python/PyTorch reference path. Compare them
only when runtime provenance is controlled.

## Accuracy reports

`summary.csv` contains one row per `(backend, task)` and an unweighted
`__selected_task_mean__` row for each backend. Task confidence intervals use a
percentile bootstrap over prompts. The selected-task mean uses stratified
bootstrap resampling within each task and weights tasks equally.

When an exact `sdpa` backend is present, `comparisons.csv` contains score deltas
paired by prompt UID for each other backend. No experiment-specific method pair
is assumed.

## Logical K/V and routing access

Raw counters are preferred over averaging precomputed percentages. Key fields
include:

- `decode_dense_gqa_kv_vectors`;
- `decode_gqa_kv_vectors_read`;
- `decode_gqa_routing_key_vectors_read`;
- `decode_gqa_centroid_key_vectors_read`;
- `decode_gqa_total_vectors_read`;
- the corresponding `decode_naive_*` fields;
- `decode_gqa_*_access_pct` and `decode_naive_*_access_pct`.

The GQA counters union sampled token rows and routing rows across query heads
that share a K/V head. The naive counters count each query head independently.
`decode_gqa_total_vectors_read` equals sampled K/V rows plus generic routing-key
rows. Centroid rows are a routing subset rather than an additional component.

For dense SDPA, K/V access is 100% and routing access is zero. Parent-based
SANTA++ uses centroid routing. Team methods use actual-team-leader routing and
normally report zero centroid subset rows.

## Cluster and team statistics

Depending on the method, records may include:

- active parent/team counts and size summaries;
- selected parent/team counts per head call;
- selected-unit inclusion probability count, mean, minimum, and maximum;
- mean sampled draws, rows, and unique GQA-unioned tokens;
- parent clustering space, team assignment space, and representative policy;
- estimated summary-storage size and peak CUDA allocation/reservation.

Null fields indicate that a statistic does not apply to the method, not a zero
measurement.

## Kubernetes shard layout

The indexed worker writes one directory per `(setting, task)` work item:

```text
RESULTS_ROOT/EXPERIMENT/
  cache/
  matrix.canonical.json
  work-index-map.csv
  shards/NNN-SETTING-TASK/
    run/
    status.json
    success.json
    failure.json        # only after a failed attempt
```

`success.json` pins the matrix hash, work index, selected UIDs, record count,
prediction checksum, code fingerprint, and runtime provenance. The exporter
refuses to treat a shard as complete when those values disagree.

## Compact export

`scripts/k8s_export_results.py` creates:

```text
EXPORT_DIR/EXPERIMENT/
  export-status.json
  matrix.json
  timing-validity.json
  hardware-provenance.csv
  compact-results.jsonl
  compact-results.csv
  summary.json           # complete exports
  summary.csv
  summary.md
  comparisons.csv
  per_example.csv
  report/
  README.txt
EXPORT_DIR/EXPERIMENT-results.tar.gz
EXPORT_DIR/EXPERIMENT-results.tar.gz.sha256
```

Prompt input text and the shared Hugging Face cache are excluded. Predictions
and references remain because they are needed to inspect and regrade outputs;
review them before distributing an archive beyond the intended research group.

`export-status.json` lists every expected shard and marks it complete,
incomplete, or invalid. `timing-validity.json` summarizes GPU products and warns
when timing comparisons span heterogeneous hardware.

## General analysis output

`scripts/summarize_results.py` discovers arbitrary settings/tasks from compact
exports and writes one directory:

```text
analysis/
  summary.json
  summary.md
  accuracy_by_task.csv
  accuracy_overall.csv
  accuracy_matrix.csv
  kv_access_by_setting_task.csv
  kv_access_by_setting.csv
  paired_comparisons.csv
  pareto_frontier_membership.csv
  plots/
```

Access percentages are recomputed from pooled raw vector counts whenever the
export provides them. Older exports that only contain percentages fall back to
mean record percentages and mark the accounting method accordingly.

The Pareto table treats lower logical access and higher score as better. It is a
descriptive frontier, not a statistical significance test.
