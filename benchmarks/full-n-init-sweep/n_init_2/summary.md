# SANTA++ RULER benchmark summary

Task scores below use RULER's task-family grader. The selected-task mean is an unweighted harness summary, not an additional RULER metric.

| Backend | Task | Score | Mean total s | Mean decode s | Decode KV access | Read-equivalent |
|---|---|---:|---:|---:|---:|---:|
| sdpa | cwe | 63.00 | 3.938 | 3.021 | 100.0% | 100.0% |
| sdpa | niah_multikey_1 | 100.00 | 3.628 | 2.768 | 100.0% | 100.0% |
| sdpa | qa_1 | 70.00 | 1.455 | 0.717 | 100.0% | 100.0% |
| sdpa | __selected_task_mean__ | 77.67 | 3.007 | 2.168 | 100.0% | 100.0% |
| santapp | cwe | 39.67 | 94.559 | 47.591 | 3.2% | 6.2% |
| santapp | niah_multikey_1 | 43.33 | 94.629 | 50.005 | 3.4% | 6.4% |
| santapp | qa_1 | 53.33 | 50.456 | 11.977 | 3.2% | 6.3% |
| santapp | __selected_task_mean__ | 45.44 | 79.881 | 36.525 | 3.2% | 6.3% |

## Paired aggregate comparison

- SANTA++ minus SDPA selected-task mean score: -32.23 points
- SANTA++ decode KV access: 3.2%
- SANTA++ metadata-inclusive read-equivalent: 6.3%
- SANTA++ end-to-end speedup vs SDPA: 0.038x
- SANTA++ decode speedup vs SDPA: 0.059x
