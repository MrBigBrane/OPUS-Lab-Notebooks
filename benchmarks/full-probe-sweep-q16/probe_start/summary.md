# SANTA++ RULER benchmark summary

Task scores below use RULER's task-family grader. The selected-task mean is an unweighted harness summary, not an additional RULER metric.

| Backend | Task | Score | Mean total s | Mean decode s | Decode KV access | Read-equivalent |
|---|---|---:|---:|---:|---:|---:|
| sdpa | niah_single_1 | 100.00 | 1.175 | 0.251 | 100.0% | 100.0% |
| sdpa | niah_single_2 | 100.00 | 3.854 | 3.001 | 100.0% | 100.0% |
| sdpa | niah_single_3 | 100.00 | 4.477 | 3.430 | 100.0% | 100.0% |
| sdpa | niah_multikey_1 | 100.00 | 4.070 | 2.973 | 100.0% | 100.0% |
| sdpa | niah_multikey_2 | 100.00 | 1.387 | 0.250 | 100.0% | 100.0% |
| sdpa | niah_multikey_3 | 100.00 | 2.065 | 0.931 | 100.0% | 100.0% |
| sdpa | niah_multivalue | 87.50 | 3.974 | 2.869 | 100.0% | 100.0% |
| sdpa | niah_multiquery | 100.00 | 3.983 | 2.866 | 100.0% | 100.0% |
| sdpa | vt | 100.00 | 2.090 | 0.793 | 100.0% | 100.0% |
| sdpa | cwe | 63.00 | 4.427 | 3.224 | 100.0% | 100.0% |
| sdpa | fwe | 70.00 | 2.478 | 1.331 | 100.0% | 100.0% |
| sdpa | qa_1 | 70.00 | 1.799 | 0.837 | 100.0% | 100.0% |
| sdpa | qa_2 | 50.00 | 1.729 | 0.645 | 100.0% | 100.0% |
| sdpa | __selected_task_mean__ | 87.73 | 2.885 | 1.800 | 100.0% | 100.0% |
| santapp | niah_single_1 | 70.00 | 54.068 | 42.286 | 3.2% | 6.2% |
| santapp | niah_single_2 | 10.00 | 60.261 | 48.901 | 3.4% | 6.5% |
| santapp | niah_single_3 | 0.00 | 59.856 | 48.517 | 3.4% | 6.5% |
| santapp | niah_multikey_1 | 30.00 | 60.034 | 48.637 | 3.4% | 6.4% |
| santapp | niah_multikey_2 | 0.00 | 53.489 | 41.894 | 3.2% | 6.3% |
| santapp | niah_multikey_3 | 0.00 | 49.609 | 37.877 | 3.2% | 6.3% |
| santapp | niah_multivalue | 10.00 | 60.126 | 48.678 | 3.4% | 6.4% |
| santapp | niah_multiquery | 10.00 | 51.619 | 40.051 | 3.2% | 6.3% |
| santapp | vt | 78.00 | 23.444 | 11.494 | 2.6% | 5.7% |
| santapp | cwe | 39.00 | 57.612 | 45.705 | 3.2% | 6.2% |
| santapp | fwe | 83.33 | 30.886 | 19.194 | 2.8% | 5.9% |
| santapp | qa_1 | 10.00 | 21.818 | 11.521 | 3.2% | 6.3% |
| santapp | qa_2 | 20.00 | 22.954 | 11.694 | 3.0% | 6.1% |
| santapp | __selected_task_mean__ | 27.72 | 46.598 | 35.111 | 3.2% | 6.3% |

## Paired aggregate comparison

- SANTA++ minus SDPA selected-task mean score: -60.01 points
- SANTA++ decode KV access: 3.2%
- SANTA++ metadata-inclusive read-equivalent: 6.3%
- SANTA++ end-to-end speedup vs SDPA: 0.062x
- SANTA++ decode speedup vs SDPA: 0.051x
