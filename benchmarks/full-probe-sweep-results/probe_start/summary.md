# SANTA++ RULER benchmark summary

Task scores below use RULER's task-family grader. The selected-task mean is an unweighted harness summary, not an additional RULER metric.

| Backend | Task | Score | Mean total s | Mean decode s | Decode KV access | Read-equivalent |
|---|---|---:|---:|---:|---:|---:|
| sdpa | niah_single_1 | 100.00 | 1.382 | 0.334 | 100.0% | 100.0% |
| sdpa | niah_single_2 | 100.00 | 4.895 | 3.906 | 100.0% | 100.0% |
| sdpa | niah_single_3 | 100.00 | 5.495 | 4.472 | 100.0% | 100.0% |
| sdpa | niah_multikey_1 | 100.00 | 4.962 | 3.953 | 100.0% | 100.0% |
| sdpa | niah_multikey_2 | 100.00 | 1.348 | 0.333 | 100.0% | 100.0% |
| sdpa | niah_multikey_3 | 100.00 | 2.267 | 1.233 | 100.0% | 100.0% |
| sdpa | niah_multivalue | 87.50 | 4.817 | 3.801 | 100.0% | 100.0% |
| sdpa | niah_multiquery | 100.00 | 4.757 | 3.748 | 100.0% | 100.0% |
| sdpa | vt | 100.00 | 2.121 | 1.066 | 100.0% | 100.0% |
| sdpa | cwe | 63.00 | 5.316 | 4.264 | 100.0% | 100.0% |
| sdpa | fwe | 73.33 | 2.788 | 1.764 | 100.0% | 100.0% |
| sdpa | qa_1 | 70.00 | 1.953 | 1.103 | 100.0% | 100.0% |
| sdpa | qa_2 | 50.00 | 1.786 | 0.848 | 100.0% | 100.0% |
| sdpa | __selected_task_mean__ | 87.99 | 3.376 | 2.371 | 100.0% | 100.0% |
| santapp | niah_single_1 | 100.00 | 78.410 | 60.163 | 3.2% | 6.3% |
| santapp | niah_single_2 | 60.00 | 86.493 | 69.200 | 3.4% | 6.5% |
| santapp | niah_single_3 | 0.00 | 85.390 | 68.160 | 3.4% | 6.5% |
| santapp | niah_multikey_1 | 80.00 | 85.763 | 68.427 | 3.4% | 6.4% |
| santapp | niah_multikey_2 | 10.00 | 69.467 | 51.581 | 3.2% | 6.3% |
| santapp | niah_multikey_3 | 0.00 | 82.597 | 64.488 | 3.2% | 6.3% |
| santapp | niah_multivalue | 40.00 | 85.328 | 67.957 | 3.4% | 6.4% |
| santapp | niah_multiquery | 37.50 | 80.070 | 62.600 | 3.3% | 6.4% |
| santapp | vt | 78.00 | 33.848 | 15.628 | 2.6% | 5.7% |
| santapp | cwe | 49.00 | 81.849 | 63.627 | 3.2% | 6.2% |
| santapp | fwe | 73.33 | 44.439 | 26.729 | 2.8% | 5.9% |
| santapp | qa_1 | 20.00 | 31.586 | 16.970 | 3.2% | 6.3% |
| santapp | qa_2 | 20.00 | 30.152 | 14.195 | 3.0% | 6.1% |
| santapp | __selected_task_mean__ | 43.68 | 67.338 | 49.979 | 3.3% | 6.3% |

## Paired aggregate comparison

- SANTA++ minus SDPA selected-task mean score: -44.31 points
- SANTA++ decode KV access: 3.3%
- SANTA++ metadata-inclusive read-equivalent: 6.3%
- SANTA++ end-to-end speedup vs SDPA: 0.050x
- SANTA++ decode speedup vs SDPA: 0.047x
