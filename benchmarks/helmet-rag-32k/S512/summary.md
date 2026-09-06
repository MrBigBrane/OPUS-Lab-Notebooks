# SANTA-family RULER benchmark summary

The primary access percentage is GQA-aware K/V union plus generic routing-key reads, relative to dense GQA K/V reads. The centroid column is the legacy centroid subset of routing, not an extra read. Values are theoretical logical vector-row estimates, not measured DRAM traffic.

| Backend | Task | Score | 95% CI | Mean total s | Mean decode s | GQA total | GQA KV | GQA routing K | GQA centroid subset | Naive total |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| team_gumbel_topk | kilt_triviaqa_32k | 60.00 | [30.00, 90.00] | 392.514 | 71.381 | 28.4% | 16.9% | 11.5% | 0.0% | 15.4% |
| team_gumbel_topk | kilt_nq_32k | 40.00 | [10.00, 70.00] | 371.406 | 63.975 | 28.5% | 17.0% | 11.5% | 0.0% | 15.4% |
| team_gumbel_topk | kilt_popqa_32k | 40.00 | [10.00, 70.00] | 343.258 | 58.268 | 28.2% | 16.7% | 11.5% | 0.0% | 15.4% |
| team_gumbel_topk | kilt_hotpotqa_32k | 30.00 | [0.00, 60.00] | 363.052 | 62.821 | 28.3% | 16.8% | 11.5% | 0.0% | 15.4% |
| team_gumbel_topk | __selected_task_mean__ | 42.50 | [27.50, 57.50] | 367.558 | 64.111 | 28.4% | 16.9% | 11.5% | 0.0% | 15.4% |

Task scores use the vendored RULER task-family grader. The selected-task mean is an unweighted harness summary.
