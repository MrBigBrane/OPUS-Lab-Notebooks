# SANTA++ RULER benchmark summary

Task scores below use RULER's task-family grader. The selected-task mean is an unweighted harness summary, not an additional RULER metric.

| Backend | Task | Score | Mean total s | Mean decode s | Decode KV access | Read-equivalent |
|---|---|---:|---:|---:|---:|---:|
| sdpa | cwe | 63.00 | 4.050 | 3.126 | 100.0% | 100.0% |
| sdpa | niah_multikey_1 | 100.00 | 3.739 | 2.872 | 100.0% | 100.0% |
| sdpa | qa_1 | 70.00 | 1.482 | 0.741 | 100.0% | 100.0% |
| sdpa | __selected_task_mean__ | 77.67 | 3.090 | 2.246 | 100.0% | 100.0% |
| santapp | cwe | 41.33 | 106.088 | 43.832 | 3.2% | 6.2% |
| santapp | niah_multikey_1 | 66.67 | 103.198 | 44.618 | 3.4% | 6.4% |
| santapp | qa_1 | 40.00 | 62.094 | 11.663 | 3.2% | 6.3% |
| santapp | __selected_task_mean__ | 49.33 | 90.460 | 33.371 | 3.2% | 6.3% |

## Paired aggregate comparison

- SANTA++ minus SDPA selected-task mean score: -28.34 points
- SANTA++ decode KV access: 3.2%
- SANTA++ metadata-inclusive read-equivalent: 6.3%
- SANTA++ end-to-end speedup vs SDPA: 0.034x
- SANTA++ decode speedup vs SDPA: 0.067x
