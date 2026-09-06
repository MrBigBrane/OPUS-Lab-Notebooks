# SANTA-family RULER benchmark summary

The primary access percentage is GQA-aware K/V union plus generic routing-key reads, relative to dense GQA K/V reads. The centroid column is the legacy centroid subset of routing, not an extra read. Values are theoretical logical vector-row estimates, not measured DRAM traffic.

| Backend | Task | Score | 95% CI | Mean total s | Mean decode s | GQA total | GQA KV | GQA routing K | GQA centroid subset | Naive total |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sdpa | kilt_nq_8k | 42.00 | [32.00, 52.00] | 2.029 | 0.978 | 100.0% | 100.0% | 0.0% | 0.0% | 100.0% |
| sdpa | kilt_popqa_8k | 53.00 | [43.00, 63.00] | 2.181 | 1.128 | 100.0% | 100.0% | 0.0% | 0.0% | 100.0% |
| sdpa | kilt_hotpotqa_8k | 37.00 | [28.00, 47.00] | 2.472 | 1.417 | 100.0% | 100.0% | 0.0% | 0.0% | 100.0% |
| sdpa | kilt_triviaqa_8k | 82.00 | [74.00, 89.00] | 2.239 | 1.184 | 100.0% | 100.0% | 0.0% | 0.0% | 100.0% |
| sdpa | __selected_task_mean__ | 53.50 | [48.75, 58.25] | 2.230 | 1.177 | 100.0% | 100.0% | 0.0% | 0.0% | 100.0% |
| santa | kilt_nq_8k | 42.00 | [32.00, 52.00] | 6.758 | 5.723 | 51.2% | 51.2% | 0.0% | 0.0% | 50.8% |
| santa | kilt_popqa_8k | 52.00 | [42.00, 62.00] | 6.506 | 5.475 | 51.1% | 51.1% | 0.0% | 0.0% | 50.8% |
| santa | kilt_hotpotqa_8k | 37.00 | [28.00, 47.00] | 7.658 | 6.627 | 51.1% | 51.1% | 0.0% | 0.0% | 50.8% |
| santa | kilt_triviaqa_8k | 84.00 | [77.00, 91.00] | 6.402 | 5.369 | 51.0% | 51.0% | 0.0% | 0.0% | 50.8% |
| santa | __selected_task_mean__ | 53.75 | [49.25, 58.25] | 6.831 | 5.798 | 51.1% | 51.1% | 0.0% | 0.0% | 50.8% |
| team_gumbel_topk | kilt_nq_8k | 38.00 | [29.00, 48.00] | 95.713 | 27.561 | 26.2% | 13.9% | 12.3% | 0.0% | 15.4% |
| team_gumbel_topk | kilt_popqa_8k | 54.00 | [44.00, 64.00] | 96.324 | 27.627 | 26.1% | 13.9% | 12.3% | 0.0% | 15.4% |
| team_gumbel_topk | kilt_hotpotqa_8k | 33.00 | [24.00, 42.00] | 106.095 | 36.140 | 25.9% | 13.7% | 12.3% | 0.0% | 15.4% |
| team_gumbel_topk | kilt_triviaqa_8k | 87.00 | [80.00, 93.00] | 101.790 | 32.327 | 25.6% | 13.4% | 12.3% | 0.0% | 15.3% |
| team_gumbel_topk | __selected_task_mean__ | 53.00 | [48.50, 57.50] | 99.981 | 30.914 | 26.0% | 13.7% | 12.3% | 0.0% | 15.4% |

Task scores use the vendored RULER task-family grader. The selected-task mean is an unweighted harness summary.
