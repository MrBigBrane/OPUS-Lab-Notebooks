# SANTA++ RULER benchmark summary

Task scores below use RULER's task-family grader. The selected-task mean is an unweighted harness summary, not an additional RULER metric.

| Backend | Task | Score | Mean total s | Mean decode s | Decode KV access | Read-equivalent |
|---|---|---:|---:|---:|---:|---:|
| sdpa | cwe | 63.00 | 3.863 | 2.964 | 100.0% | 100.0% |
| sdpa | niah_multikey_1 | 100.00 | 3.563 | 2.722 | 100.0% | 100.0% |
| sdpa | qa_1 | 70.00 | 1.427 | 0.706 | 100.0% | 100.0% |
| sdpa | __selected_task_mean__ | 77.67 | 2.951 | 2.131 | 100.0% | 100.0% |
| santapp | cwe | 45.00 | 75.427 | 41.199 | 3.2% | 6.2% |
| santapp | niah_multikey_1 | 60.00 | 74.646 | 41.461 | 3.4% | 6.4% |
| santapp | qa_1 | 36.67 | 38.368 | 10.336 | 3.2% | 6.3% |
| santapp | __selected_task_mean__ | 47.22 | 62.814 | 30.999 | 3.3% | 6.3% |

## Paired aggregate comparison

- SANTA++ minus SDPA selected-task mean score: -30.45 points
- SANTA++ decode KV access: 3.3%
- SANTA++ metadata-inclusive read-equivalent: 6.3%
- SANTA++ end-to-end speedup vs SDPA: 0.047x
- SANTA++ decode speedup vs SDPA: 0.069x
