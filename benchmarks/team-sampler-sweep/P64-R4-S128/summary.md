# SANTA-family RULER benchmark summary

The primary access percentage is GQA-aware K/V union plus generic routing-key reads, relative to dense GQA K/V reads. The centroid column is the legacy centroid subset of routing, not an extra read. Values are theoretical logical vector-row estimates, not measured DRAM traffic.

| Backend | Task | Score | 95% CI | Mean total s | Mean decode s | GQA total | GQA KV | GQA routing K | GQA centroid subset | Naive total |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| team_gumbel_topk | niah_multiquery | 60.00 | [54.50, 65.50] | 111.628 | 94.184 | 16.7% | 13.6% | 3.1% | 0.0% | 6.4% |
| team_gumbel_topk | niah_multivalue | 50.00 | [44.50, 55.50] | 121.045 | 103.583 | 17.9% | 14.8% | 3.1% | 0.0% | 6.7% |
| team_gumbel_topk | __selected_task_mean__ | 55.00 | [51.25, 58.88] | 116.336 | 98.883 | 17.3% | 14.2% | 3.1% | 0.0% | 6.5% |

Task scores use the vendored RULER task-family grader. The selected-task mean is an unweighted harness summary.
