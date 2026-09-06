# SANTA-family RULER benchmark summary

The primary access percentage is GQA-aware K/V union plus generic routing-key reads, relative to dense GQA K/V reads. The centroid column is the legacy centroid subset of routing, not an extra read. Values are theoretical logical vector-row estimates, not measured DRAM traffic.

| Backend | Task | Score | 95% CI | Mean total s | Mean decode s | GQA total | GQA KV | GQA routing K | GQA centroid subset | Naive total |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| team_gumbel_topk | kilt_triviaqa_32k | 80.00 | [50.00, 100.00] | 306.923 | 50.829 | 21.5% | 10.0% | 11.5% | 0.0% | 13.5% |
| team_gumbel_topk | kilt_nq_32k | 30.00 | [0.00, 60.00] | 304.256 | 50.121 | 21.7% | 10.2% | 11.5% | 0.0% | 13.6% |
| team_gumbel_topk | kilt_popqa_32k | 40.00 | [10.00, 70.00] | 304.900 | 50.425 | 21.5% | 9.9% | 11.5% | 0.0% | 13.6% |
| team_gumbel_topk | kilt_hotpotqa_32k | 10.00 | [0.00, 30.00] | 305.511 | 50.665 | 21.5% | 10.0% | 11.5% | 0.0% | 13.5% |
| team_gumbel_topk | __selected_task_mean__ | 40.00 | [27.50, 52.50] | 305.397 | 50.510 | 21.5% | 10.0% | 11.5% | 0.0% | 13.6% |

Task scores use the vendored RULER task-family grader. The selected-task mean is an unweighted harness summary.
