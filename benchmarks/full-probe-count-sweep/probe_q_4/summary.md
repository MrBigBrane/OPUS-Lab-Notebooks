# SANTA++ RULER benchmark summary

Task scores below use RULER's task-family grader. The selected-task mean is an unweighted harness summary, not an additional RULER metric.

| Backend | Task | Score | Mean total s | Mean decode s | Decode KV access | Read-equivalent |
|---|---|---:|---:|---:|---:|---:|
| sdpa | niah_single_1 | 100.00 | 1.407 | 0.378 | 100.0% | 100.0% |
| sdpa | niah_single_2 | 100.00 | 5.733 | 4.753 | 100.0% | 100.0% |
| sdpa | niah_single_3 | 100.00 | 6.559 | 5.545 | 100.0% | 100.0% |
| sdpa | niah_multikey_1 | 100.00 | 5.703 | 4.695 | 100.0% | 100.0% |
| sdpa | niah_multikey_2 | 100.00 | 1.393 | 0.382 | 100.0% | 100.0% |
| sdpa | niah_multikey_3 | 100.00 | 2.542 | 1.524 | 100.0% | 100.0% |
| sdpa | niah_multivalue | 87.50 | 5.782 | 4.768 | 100.0% | 100.0% |
| sdpa | niah_multiquery | 100.00 | 5.815 | 4.808 | 100.0% | 100.0% |
| sdpa | vt | 100.00 | 2.463 | 1.425 | 100.0% | 100.0% |
| sdpa | cwe | 63.00 | 6.973 | 5.925 | 100.0% | 100.0% |
| sdpa | fwe | 73.33 | 3.217 | 2.195 | 100.0% | 100.0% |
| sdpa | qa_1 | 70.00 | 2.131 | 1.284 | 100.0% | 100.0% |
| sdpa | qa_2 | 50.00 | 1.931 | 0.995 | 100.0% | 100.0% |
| sdpa | __selected_task_mean__ | 87.99 | 3.973 | 2.975 | 100.0% | 100.0% |
| santapp | niah_single_1 | 70.00 | 93.072 | 75.996 | 3.2% | 6.3% |
| santapp | niah_single_2 | 20.00 | 135.575 | 111.412 | 3.4% | 6.5% |
| santapp | niah_single_3 | 10.00 | 118.841 | 100.210 | 3.4% | 6.5% |
| santapp | niah_multikey_1 | 20.00 | 109.009 | 89.498 | 3.4% | 6.4% |
| santapp | niah_multikey_2 | 0.00 | 78.979 | 62.387 | 3.2% | 6.3% |
| santapp | niah_multikey_3 | 0.00 | 91.732 | 74.828 | 3.2% | 6.3% |
| santapp | niah_multivalue | 7.50 | 87.289 | 71.701 | 3.4% | 6.4% |
| santapp | niah_multiquery | 2.50 | 80.006 | 64.187 | 3.3% | 6.4% |
| santapp | vt | 68.00 | 33.494 | 17.098 | 2.6% | 5.7% |
| santapp | cwe | 32.00 | 83.418 | 67.224 | 3.2% | 6.2% |
| santapp | fwe | 70.00 | 44.349 | 28.434 | 2.8% | 5.9% |
| santapp | qa_1 | 50.00 | 31.847 | 17.983 | 3.2% | 6.3% |
| santapp | qa_2 | 30.00 | 31.416 | 16.794 | 3.0% | 6.1% |
| santapp | __selected_task_mean__ | 29.23 | 78.387 | 61.366 | 3.2% | 6.3% |

## Paired aggregate comparison

- SANTA++ minus SDPA selected-task mean score: -58.76 points
- SANTA++ decode KV access: 3.2%
- SANTA++ metadata-inclusive read-equivalent: 6.3%
- SANTA++ end-to-end speedup vs SDPA: 0.051x
- SANTA++ decode speedup vs SDPA: 0.048x
