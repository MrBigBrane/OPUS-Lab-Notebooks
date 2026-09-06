# SANTA-family RULER benchmark summary

The primary access percentage is GQA-aware K/V union plus generic routing-key reads, relative to dense GQA K/V reads. The centroid column is the legacy centroid subset of routing, not an extra read. Values are theoretical logical vector-row estimates, not measured DRAM traffic.

| Backend | Task | Score | 95% CI | Mean total s | Mean decode s | GQA total | GQA KV | GQA routing K | GQA centroid subset | Naive total |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sdpa | kilt_triviaqa_32k | 77.00 | [69.00, 85.00] | 12.594 | 5.526 | 100.0% | 100.0% | 0.0% | 0.0% | 100.0% |
| sdpa | kilt_nq_32k | 35.00 | [26.00, 45.00] | 12.587 | 5.522 | 100.0% | 100.0% | 0.0% | 0.0% | 100.0% |
| sdpa | kilt_popqa_32k | 41.00 | [31.00, 51.00] | 12.575 | 5.522 | 100.0% | 100.0% | 0.0% | 0.0% | 100.0% |
| sdpa | kilt_hotpotqa_32k | 29.00 | [21.00, 38.00] | 12.559 | 5.518 | 100.0% | 100.0% | 0.0% | 0.0% | 100.0% |
| sdpa | __selected_task_mean__ | 45.50 | [41.00, 50.00] | 12.579 | 5.522 | 100.0% | 100.0% | 0.0% | 0.0% | 100.0% |
| santa | kilt_triviaqa_32k | 76.00 | [67.00, 84.00] | 16.849 | 9.801 | 50.4% | 50.4% | 0.0% | 0.0% | 50.2% |
| santa | kilt_nq_32k | 41.00 | [31.00, 51.00] | 16.823 | 9.771 | 50.4% | 50.4% | 0.0% | 0.0% | 50.2% |
| santa | kilt_popqa_32k | 43.00 | [33.00, 52.00] | 16.819 | 9.784 | 50.4% | 50.4% | 0.0% | 0.0% | 50.2% |
| santa | kilt_hotpotqa_32k | 30.00 | [21.00, 39.00] | 16.805 | 9.783 | 50.4% | 50.4% | 0.0% | 0.0% | 50.2% |
| santa | __selected_task_mean__ | 47.50 | [42.75, 52.25] | 16.824 | 9.785 | 50.4% | 50.4% | 0.0% | 0.0% | 50.2% |
| team_gumbel_topk | kilt_triviaqa_32k | 76.00 | [67.00, 84.00] | 314.501 | 52.232 | 17.1% | 5.7% | 11.5% | 0.0% | 12.5% |
| team_gumbel_topk | kilt_nq_32k | 33.00 | [24.00, 43.00] | 312.830 | 51.864 | 17.2% | 5.7% | 11.5% | 0.0% | 12.6% |
| team_gumbel_topk | kilt_popqa_32k | 43.00 | [33.00, 53.00] | 313.747 | 51.951 | 17.2% | 5.7% | 11.5% | 0.0% | 12.6% |
| team_gumbel_topk | kilt_hotpotqa_32k | 25.00 | [17.00, 34.00] | 313.229 | 51.805 | 17.1% | 5.6% | 11.5% | 0.0% | 12.5% |
| team_gumbel_topk | __selected_task_mean__ | 44.25 | [39.75, 48.75] | 313.577 | 51.963 | 17.1% | 5.7% | 11.5% | 0.0% | 12.6% |

Task scores use the vendored RULER task-family grader. The selected-task mean is an unweighted harness summary.
