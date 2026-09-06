# SANTA-family RULER benchmark summary

The primary access percentage is GQA-aware K/V union plus generic routing-key reads, relative to dense GQA K/V reads. The centroid column is the legacy centroid subset of routing, not an extra read. Values are theoretical logical vector-row estimates, not measured DRAM traffic.

| Backend | Task | Score | 95% CI | Mean total s | Mean decode s | GQA total | GQA KV | GQA routing K | GQA centroid subset | Naive total |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| team_gumbel_topk | niah_multiquery | 95.50 | [93.25, 97.50] | 120.986 | 89.180 | 19.5% | 13.3% | 6.2% | 0.0% | 9.4% |
| team_gumbel_topk | niah_multivalue | 86.25 | [82.75, 89.75] | 136.851 | 103.358 | 20.5% | 14.3% | 6.2% | 0.0% | 9.6% |
| team_gumbel_topk | __selected_task_mean__ | 90.88 | [88.75, 92.88] | 128.919 | 96.269 | 20.0% | 13.8% | 6.2% | 0.0% | 9.5% |

Task scores use the vendored RULER task-family grader. The selected-task mean is an unweighted harness summary.
