# SANTA++ RULER benchmark summary

Task scores below use RULER's task-family grader. The selected-task mean is an unweighted harness summary, not an additional RULER metric.

| Backend | Task | Score | Mean total s | Mean decode s | Decode KV access | Read-equivalent |
|---|---|---:|---:|---:|---:|---:|
| sdpa | niah_single_1 | 100.00 | 1.374 | 0.314 | 100.0% | 100.0% |
| sdpa | niah_single_2 | 100.00 | 4.779 | 3.771 | 100.0% | 100.0% |
| sdpa | niah_single_3 | 100.00 | 5.386 | 4.357 | 100.0% | 100.0% |
| sdpa | niah_multikey_1 | 100.00 | 4.866 | 3.849 | 100.0% | 100.0% |
| sdpa | niah_multikey_2 | 100.00 | 1.336 | 0.316 | 100.0% | 100.0% |
| sdpa | niah_multikey_3 | 100.00 | 2.238 | 1.210 | 100.0% | 100.0% |
| sdpa | niah_multivalue | 87.50 | 4.714 | 3.694 | 100.0% | 100.0% |
| sdpa | niah_multiquery | 100.00 | 4.614 | 3.604 | 100.0% | 100.0% |
| sdpa | vt | 100.00 | 2.033 | 0.987 | 100.0% | 100.0% |
| sdpa | cwe | 63.00 | 5.084 | 4.031 | 100.0% | 100.0% |
| sdpa | fwe | 73.33 | 2.718 | 1.691 | 100.0% | 100.0% |
| sdpa | qa_1 | 70.00 | 1.916 | 1.064 | 100.0% | 100.0% |
| sdpa | qa_2 | 50.00 | 1.764 | 0.814 | 100.0% | 100.0% |
| sdpa | __selected_task_mean__ | 87.99 | 3.294 | 2.285 | 100.0% | 100.0% |
| santapp | niah_single_1 | 90.00 | 75.251 | 57.408 | 3.2% | 6.3% |
| santapp | niah_single_2 | 40.00 | 83.686 | 66.708 | 3.4% | 6.5% |
| santapp | niah_single_3 | 10.00 | 82.445 | 65.583 | 3.4% | 6.5% |
| santapp | niah_multikey_1 | 70.00 | 81.742 | 64.879 | 3.4% | 6.4% |
| santapp | niah_multikey_2 | 0.00 | 56.749 | 39.524 | 3.2% | 6.3% |
| santapp | niah_multikey_3 | 0.00 | 61.613 | 44.231 | 3.2% | 6.2% |
| santapp | niah_multivalue | 27.50 | 82.784 | 65.727 | 3.4% | 6.4% |
| santapp | niah_multiquery | 47.50 | 72.352 | 55.198 | 3.3% | 6.3% |
| santapp | vt | 82.00 | 33.293 | 15.314 | 2.6% | 5.7% |
| santapp | cwe | 43.00 | 79.301 | 61.452 | 3.2% | 6.2% |
| santapp | fwe | 66.67 | 43.091 | 25.683 | 2.8% | 5.9% |
| santapp | qa_1 | 20.00 | 30.743 | 16.396 | 3.2% | 6.3% |
| santapp | qa_2 | 40.00 | 31.874 | 16.427 | 3.0% | 6.1% |
| santapp | __selected_task_mean__ | 41.28 | 62.686 | 45.733 | 3.2% | 6.3% |

## Paired aggregate comparison

- SANTA++ minus SDPA selected-task mean score: -46.71 points
- SANTA++ decode KV access: 3.2%
- SANTA++ metadata-inclusive read-equivalent: 6.3%
- SANTA++ end-to-end speedup vs SDPA: 0.053x
- SANTA++ decode speedup vs SDPA: 0.050x
