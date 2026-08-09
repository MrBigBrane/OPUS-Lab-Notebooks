# SANTA++ RULER benchmark summary

Task scores below use RULER's task-family grader. The selected-task mean is an unweighted harness summary, not an additional RULER metric.

| Backend | Task | Score | Mean total s | Mean decode s | Decode KV access | Read-equivalent |
|---|---|---:|---:|---:|---:|---:|
| sdpa | niah_single_1 | 100.00 | 1.175 | 0.246 | 100.0% | 100.0% |
| sdpa | niah_single_2 | 100.00 | 3.787 | 2.939 | 100.0% | 100.0% |
| sdpa | niah_single_3 | 100.00 | 4.352 | 3.396 | 100.0% | 100.0% |
| sdpa | niah_multikey_1 | 100.00 | 4.077 | 2.937 | 100.0% | 100.0% |
| sdpa | niah_multikey_2 | 100.00 | 1.410 | 0.240 | 100.0% | 100.0% |
| sdpa | niah_multikey_3 | 100.00 | 2.036 | 0.907 | 100.0% | 100.0% |
| sdpa | niah_multivalue | 87.50 | 3.893 | 2.798 | 100.0% | 100.0% |
| sdpa | niah_multiquery | 100.00 | 3.917 | 2.810 | 100.0% | 100.0% |
| sdpa | vt | 100.00 | 2.036 | 0.772 | 100.0% | 100.0% |
| sdpa | cwe | 63.00 | 4.349 | 3.160 | 100.0% | 100.0% |
| sdpa | fwe | 70.00 | 2.455 | 1.318 | 100.0% | 100.0% |
| sdpa | qa_1 | 70.00 | 1.852 | 0.831 | 100.0% | 100.0% |
| sdpa | qa_2 | 50.00 | 1.843 | 0.638 | 100.0% | 100.0% |
| sdpa | __selected_task_mean__ | 87.73 | 2.860 | 1.769 | 100.0% | 100.0% |
| santapp | niah_single_1 | 60.00 | 54.710 | 43.029 | 3.2% | 6.3% |
| santapp | niah_single_2 | 30.00 | 58.535 | 47.382 | 3.4% | 6.5% |
| santapp | niah_single_3 | 0.00 | 58.374 | 47.260 | 3.4% | 6.5% |
| santapp | niah_multikey_1 | 30.00 | 58.508 | 47.335 | 3.4% | 6.4% |
| santapp | niah_multikey_2 | 20.00 | 49.718 | 38.353 | 3.3% | 6.4% |
| santapp | niah_multikey_3 | 0.00 | 49.234 | 37.848 | 3.2% | 6.3% |
| santapp | niah_multivalue | 7.50 | 58.399 | 47.132 | 3.4% | 6.4% |
| santapp | niah_multiquery | 32.50 | 55.886 | 44.683 | 3.3% | 6.4% |
| santapp | vt | 78.00 | 22.685 | 11.045 | 2.6% | 5.7% |
| santapp | cwe | 38.00 | 55.749 | 44.169 | 3.2% | 6.2% |
| santapp | fwe | 70.00 | 29.887 | 18.493 | 2.8% | 5.9% |
| santapp | qa_1 | 30.00 | 21.935 | 11.909 | 3.2% | 6.3% |
| santapp | qa_2 | 30.00 | 21.911 | 11.375 | 2.9% | 6.0% |
| santapp | __selected_task_mean__ | 32.77 | 45.810 | 34.616 | 3.3% | 6.3% |

## Paired aggregate comparison

- SANTA++ minus SDPA selected-task mean score: -54.96 points
- SANTA++ decode KV access: 3.3%
- SANTA++ metadata-inclusive read-equivalent: 6.3%
- SANTA++ end-to-end speedup vs SDPA: 0.062x
- SANTA++ decode speedup vs SDPA: 0.051x
