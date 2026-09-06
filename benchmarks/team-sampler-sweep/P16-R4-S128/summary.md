# SANTA-family RULER benchmark summary

The primary access percentage is GQA-aware K/V union plus generic routing-key reads, relative to dense GQA K/V reads. The centroid column is the legacy centroid subset of routing, not an extra read. Values are theoretical logical vector-row estimates, not measured DRAM traffic.

| Backend | Task | Score | 95% CI | Mean total s | Mean decode s | GQA total | GQA KV | GQA routing K | GQA centroid subset | Naive total |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| team_gumbel_topk | niah_multiquery | 99.25 | [98.25, 100.00] | 177.299 | 107.205 | 25.4% | 13.2% | 12.2% | 0.0% | 15.4% |
| team_gumbel_topk | niah_multivalue | 92.75 | [90.25, 95.00] | 160.540 | 96.888 | 26.3% | 14.0% | 12.2% | 0.0% | 15.6% |
| team_gumbel_topk | __selected_task_mean__ | 96.00 | [94.75, 97.25] | 168.919 | 102.046 | 25.8% | 13.6% | 12.2% | 0.0% | 15.5% |

Task scores use the vendored RULER task-family grader. The selected-task mean is an unweighted harness summary.
