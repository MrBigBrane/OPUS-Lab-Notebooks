# SANTA-family RULER benchmark summary

The primary access percentage is GQA-aware K/V union plus generic routing-key reads, relative to dense GQA K/V reads. The centroid column is the legacy centroid subset of routing, not an extra read. Values are theoretical logical vector-row estimates, not measured DRAM traffic.

| Backend | Task | Score | 95% CI | Mean total s | Mean decode s | GQA total | GQA KV | GQA routing K | GQA centroid subset | Naive total |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| team_gumbel_topk | niah_multiquery | 99.00 | [98.00, 99.75] | 107.959 | 68.374 | 25.7% | 13.4% | 12.3% | 0.0% | 15.6% |
| team_gumbel_topk | niah_multivalue | 92.25 | [89.75, 94.50] | 112.932 | 73.618 | 26.7% | 14.4% | 12.3% | 0.0% | 15.8% |
| team_gumbel_topk | __selected_task_mean__ | 95.62 | [94.38, 96.88] | 110.446 | 70.996 | 26.2% | 13.9% | 12.3% | 0.0% | 15.7% |

Task scores use the vendored RULER task-family grader. The selected-task mean is an unweighted harness summary.
