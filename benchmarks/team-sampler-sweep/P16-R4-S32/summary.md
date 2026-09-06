# SANTA-family RULER benchmark summary

The primary access percentage is GQA-aware K/V union plus generic routing-key reads, relative to dense GQA K/V reads. The centroid column is the legacy centroid subset of routing, not an extra read. Values are theoretical logical vector-row estimates, not measured DRAM traffic.

| Backend | Task | Score | 95% CI | Mean total s | Mean decode s | GQA total | GQA KV | GQA routing K | GQA centroid subset | Naive total |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| team_gumbel_topk | niah_multiquery | 91.50 | [88.50, 94.25] | 198.557 | 126.628 | 16.5% | 4.3% | 12.2% | 0.0% | 13.6% |
| team_gumbel_topk | niah_multivalue | 84.00 | [79.75, 87.75] | 199.286 | 129.847 | 16.7% | 4.5% | 12.2% | 0.0% | 13.6% |
| team_gumbel_topk | __selected_task_mean__ | 87.75 | [85.25, 90.12] | 198.921 | 128.238 | 16.6% | 4.4% | 12.2% | 0.0% | 13.6% |

Task scores use the vendored RULER task-family grader. The selected-task mean is an unweighted harness summary.
