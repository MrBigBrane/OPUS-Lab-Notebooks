# SANTA-family RULER benchmark summary

The primary access percentage is GQA-aware K/V union plus generic routing-key reads, relative to dense GQA K/V reads. The centroid column is the legacy centroid subset of routing, not an extra read. Values are theoretical logical vector-row estimates, not measured DRAM traffic.

| Backend | Task | Score | 95% CI | Mean total s | Mean decode s | GQA total | GQA KV | GQA routing K | GQA centroid subset | Naive total |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| team_gumbel_topk | niah_multiquery | 96.25 | [94.25, 98.00] | 178.494 | 110.643 | 19.7% | 7.4% | 12.2% | 0.0% | 14.1% |
| team_gumbel_topk | niah_multivalue | 90.75 | [88.00, 93.50] | 175.329 | 110.394 | 20.1% | 7.9% | 12.2% | 0.0% | 14.3% |
| team_gumbel_topk | __selected_task_mean__ | 93.50 | [91.75, 95.12] | 176.911 | 110.518 | 19.9% | 7.7% | 12.2% | 0.0% | 14.2% |

Task scores use the vendored RULER task-family grader. The selected-task mean is an unweighted harness summary.
