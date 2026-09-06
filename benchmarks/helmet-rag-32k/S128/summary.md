# SANTA-family RULER benchmark summary

The primary access percentage is GQA-aware K/V union plus generic routing-key reads, relative to dense GQA K/V reads. The centroid column is the legacy centroid subset of routing, not an extra read. Values are theoretical logical vector-row estimates, not measured DRAM traffic.

| Backend | Task | Score | 95% CI | Mean total s | Mean decode s | GQA total | GQA KV | GQA routing K | GQA centroid subset | Naive total |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sdpa | kilt_triviaqa_32k | 60.00 | [30.00, 90.00] | 6.170 | 2.757 | 100.0% | 100.0% | 0.0% | 0.0% | 100.0% |
| sdpa | kilt_nq_32k | 30.00 | [0.00, 60.00] | 6.234 | 2.761 | 100.0% | 100.0% | 0.0% | 0.0% | 100.0% |
| sdpa | kilt_popqa_32k | 40.00 | [10.00, 70.00] | 6.290 | 2.764 | 100.0% | 100.0% | 0.0% | 0.0% | 100.0% |
| sdpa | kilt_hotpotqa_32k | 30.00 | [0.00, 60.00] | 6.314 | 2.763 | 100.0% | 100.0% | 0.0% | 0.0% | 100.0% |
| sdpa | __selected_task_mean__ | 40.00 | [25.00, 55.00] | 6.252 | 2.761 | 100.0% | 100.0% | 0.0% | 0.0% | 100.0% |
| santa | kilt_triviaqa_32k | 50.00 | [20.00, 80.00] | 8.678 | 5.186 | 50.4% | 50.4% | 0.0% | 0.0% | 50.2% |
| santa | kilt_nq_32k | 30.00 | [0.00, 60.00] | 8.612 | 5.163 | 50.4% | 50.4% | 0.0% | 0.0% | 50.2% |
| santa | kilt_popqa_32k | 40.00 | [10.00, 70.00] | 8.610 | 5.171 | 50.4% | 50.4% | 0.0% | 0.0% | 50.2% |
| santa | kilt_hotpotqa_32k | 40.00 | [10.00, 70.00] | 8.626 | 5.186 | 50.4% | 50.4% | 0.0% | 0.0% | 50.2% |
| santa | __selected_task_mean__ | 40.00 | [25.00, 55.00] | 8.631 | 5.177 | 50.4% | 50.4% | 0.0% | 0.0% | 50.2% |
| team_gumbel_topk | kilt_triviaqa_32k | 70.00 | [40.00, 100.00] | 155.220 | 30.161 | 17.1% | 5.7% | 11.5% | 0.0% | 12.5% |
| team_gumbel_topk | kilt_nq_32k | 40.00 | [10.00, 70.00] | 155.380 | 30.216 | 17.2% | 5.7% | 11.5% | 0.0% | 12.6% |
| team_gumbel_topk | kilt_popqa_32k | 40.00 | [10.00, 70.00] | 155.493 | 30.223 | 17.1% | 5.6% | 11.5% | 0.0% | 12.6% |
| team_gumbel_topk | kilt_hotpotqa_32k | 40.00 | [10.00, 70.00] | 155.030 | 30.153 | 17.1% | 5.6% | 11.5% | 0.0% | 12.6% |
| team_gumbel_topk | __selected_task_mean__ | 47.50 | [32.50, 62.50] | 155.281 | 30.188 | 17.1% | 5.6% | 11.5% | 0.0% | 12.6% |

Task scores use the vendored RULER task-family grader. The selected-task mean is an unweighted harness summary.
