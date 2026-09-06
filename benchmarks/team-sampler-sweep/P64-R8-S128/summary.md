# SANTA-family RULER benchmark summary

The primary access percentage is GQA-aware K/V union plus generic routing-key reads, relative to dense GQA K/V reads. The centroid column is the legacy centroid subset of routing, not an extra read. Values are theoretical logical vector-row estimates, not measured DRAM traffic.

| Backend | Task | Score | 95% CI | Mean total s | Mean decode s | GQA total | GQA KV | GQA routing K | GQA centroid subset | Naive total |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| team_gumbel_topk | niah_multiquery | 93.00 | [90.25, 95.50] | 94.127 | 72.558 | 21.0% | 14.8% | 6.2% | 0.0% | 9.8% |
| team_gumbel_topk | niah_multivalue | 84.25 | [80.50, 87.75] | 102.671 | 80.268 | 22.3% | 16.1% | 6.2% | 0.0% | 10.2% |
| team_gumbel_topk | __selected_task_mean__ | 88.62 | [86.25, 90.88] | 98.399 | 76.413 | 21.7% | 15.5% | 6.2% | 0.0% | 10.0% |

Task scores use the vendored RULER task-family grader. The selected-task mean is an unweighted harness summary.
