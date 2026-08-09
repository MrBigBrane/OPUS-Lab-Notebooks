# SANTA++ RULER benchmark summary

Task scores below use RULER's task-family grader. The selected-task mean is an unweighted harness summary, not an additional RULER metric.

| Backend | Task | Score | Mean total s | Mean decode s | Decode KV access | Read-equivalent |
|---|---|---:|---:|---:|---:|---:|
| sdpa | niah_single_1 | 100.00 | 1.371 | 0.317 | 100.0% | 100.0% |
| sdpa | niah_single_2 | 100.00 | 4.763 | 3.752 | 100.0% | 100.0% |
| sdpa | niah_single_3 | 100.00 | 5.405 | 4.369 | 100.0% | 100.0% |
| sdpa | niah_multikey_1 | 100.00 | 4.706 | 3.679 | 100.0% | 100.0% |
| sdpa | niah_multikey_2 | 100.00 | 1.336 | 0.305 | 100.0% | 100.0% |
| sdpa | niah_multikey_3 | 100.00 | 2.178 | 1.138 | 100.0% | 100.0% |
| sdpa | niah_multivalue | 87.50 | 4.572 | 3.534 | 100.0% | 100.0% |
| sdpa | niah_multiquery | 100.00 | 4.588 | 3.563 | 100.0% | 100.0% |
| sdpa | vt | 100.00 | 2.033 | 0.970 | 100.0% | 100.0% |
| sdpa | cwe | 63.00 | 5.120 | 4.051 | 100.0% | 100.0% |
| sdpa | fwe | 73.33 | 2.673 | 1.629 | 100.0% | 100.0% |
| sdpa | qa_1 | 70.00 | 1.899 | 1.030 | 100.0% | 100.0% |
| sdpa | qa_2 | 50.00 | 1.761 | 0.802 | 100.0% | 100.0% |
| sdpa | __selected_task_mean__ | 87.99 | 3.262 | 2.241 | 100.0% | 100.0% |
| santapp | niah_single_1 | 100.00 | 28.864 | 14.120 | 27.2% | 30.3% |
| santapp | niah_single_2 | 100.00 | 78.815 | 64.590 | 28.9% | 32.0% |
| santapp | niah_single_3 | 30.00 | 78.860 | 64.623 | 28.8% | 31.9% |
| santapp | niah_multikey_1 | 100.00 | 69.990 | 55.879 | 28.6% | 31.7% |
| santapp | niah_multikey_2 | 70.00 | 34.858 | 20.414 | 27.9% | 31.0% |
| santapp | niah_multikey_3 | 0.00 | 70.931 | 56.328 | 27.7% | 30.8% |
| santapp | niah_multivalue | 85.00 | 70.437 | 56.127 | 28.6% | 31.6% |
| santapp | niah_multiquery | 82.50 | 69.315 | 54.973 | 28.4% | 31.5% |
| santapp | vt | 100.00 | 30.161 | 15.275 | 26.8% | 29.9% |
| santapp | cwe | 59.00 | 75.422 | 60.613 | 27.1% | 30.2% |
| santapp | fwe | 73.33 | 39.858 | 25.294 | 27.7% | 30.8% |
| santapp | qa_1 | 70.00 | 28.758 | 16.128 | 32.7% | 35.8% |
| santapp | qa_2 | 30.00 | 24.928 | 11.490 | 30.7% | 33.8% |
| santapp | __selected_task_mean__ | 69.22 | 53.938 | 39.681 | 28.3% | 31.4% |

## Paired aggregate comparison

- SANTA++ minus SDPA selected-task mean score: -18.77 points
- SANTA++ decode KV access: 28.3%
- SANTA++ metadata-inclusive read-equivalent: 31.4%
- SANTA++ end-to-end speedup vs SDPA: 0.060x
- SANTA++ decode speedup vs SDPA: 0.056x
