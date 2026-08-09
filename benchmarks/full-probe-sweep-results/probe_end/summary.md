# SANTA++ RULER benchmark summary

Task scores below use RULER's task-family grader. The selected-task mean is an unweighted harness summary, not an additional RULER metric.

| Backend | Task | Score | Mean total s | Mean decode s | Decode KV access | Read-equivalent |
|---|---|---:|---:|---:|---:|---:|
| sdpa | niah_single_1 | 100.00 | 1.376 | 0.318 | 100.0% | 100.0% |
| sdpa | niah_single_2 | 100.00 | 4.716 | 3.702 | 100.0% | 100.0% |
| sdpa | niah_single_3 | 100.00 | 5.301 | 4.274 | 100.0% | 100.0% |
| sdpa | niah_multikey_1 | 100.00 | 4.774 | 3.760 | 100.0% | 100.0% |
| sdpa | niah_multikey_2 | 100.00 | 1.334 | 0.319 | 100.0% | 100.0% |
| sdpa | niah_multikey_3 | 100.00 | 2.202 | 1.178 | 100.0% | 100.0% |
| sdpa | niah_multivalue | 87.50 | 4.715 | 3.700 | 100.0% | 100.0% |
| sdpa | niah_multiquery | 100.00 | 4.802 | 3.795 | 100.0% | 100.0% |
| sdpa | vt | 100.00 | 2.069 | 1.027 | 100.0% | 100.0% |
| sdpa | cwe | 63.00 | 5.261 | 4.212 | 100.0% | 100.0% |
| sdpa | fwe | 73.33 | 2.786 | 1.764 | 100.0% | 100.0% |
| sdpa | qa_1 | 70.00 | 1.954 | 1.105 | 100.0% | 100.0% |
| sdpa | qa_2 | 50.00 | 1.780 | 0.841 | 100.0% | 100.0% |
| sdpa | __selected_task_mean__ | 87.99 | 3.313 | 2.307 | 100.0% | 100.0% |
| santapp | niah_single_1 | 80.00 | 59.220 | 41.317 | 3.1% | 6.2% |
| santapp | niah_single_2 | 60.00 | 81.666 | 64.603 | 3.4% | 6.5% |
| santapp | niah_single_3 | 10.00 | 84.368 | 67.265 | 3.4% | 6.5% |
| santapp | niah_multikey_1 | 60.00 | 84.876 | 67.716 | 3.4% | 6.4% |
| santapp | niah_multikey_2 | 20.00 | 68.590 | 51.006 | 3.2% | 6.3% |
| santapp | niah_multikey_3 | 0.00 | 70.050 | 52.044 | 3.2% | 6.2% |
| santapp | niah_multivalue | 27.50 | 84.240 | 66.876 | 3.4% | 6.4% |
| santapp | niah_multiquery | 42.50 | 71.386 | 54.075 | 3.2% | 6.3% |
| santapp | vt | 92.00 | 34.115 | 15.911 | 2.6% | 5.7% |
| santapp | cwe | 49.00 | 81.279 | 63.034 | 3.2% | 6.2% |
| santapp | fwe | 76.67 | 43.764 | 26.196 | 2.8% | 5.9% |
| santapp | qa_1 | 30.00 | 28.605 | 14.225 | 3.2% | 6.2% |
| santapp | qa_2 | 30.00 | 30.179 | 14.529 | 2.9% | 6.0% |
| santapp | __selected_task_mean__ | 44.44 | 63.257 | 46.061 | 3.2% | 6.3% |

## Paired aggregate comparison

- SANTA++ minus SDPA selected-task mean score: -43.55 points
- SANTA++ decode KV access: 3.2%
- SANTA++ metadata-inclusive read-equivalent: 6.3%
- SANTA++ end-to-end speedup vs SDPA: 0.052x
- SANTA++ decode speedup vs SDPA: 0.050x
