# SANTA++ RULER benchmark summary

Task scores below use RULER's task-family grader. The selected-task mean is an unweighted harness summary, not an additional RULER metric.

| Backend | Task | Score | Mean total s | Mean decode s | Decode KV access | Read-equivalent |
|---|---|---:|---:|---:|---:|---:|
| sdpa | niah_single_1 | 100.00 | 1.177 | 0.249 | 100.0% | 100.0% |
| sdpa | niah_single_2 | 100.00 | 3.813 | 2.961 | 100.0% | 100.0% |
| sdpa | niah_single_3 | 100.00 | 4.392 | 3.375 | 100.0% | 100.0% |
| sdpa | niah_multikey_1 | 100.00 | 4.059 | 2.942 | 100.0% | 100.0% |
| sdpa | niah_multikey_2 | 100.00 | 1.426 | 0.242 | 100.0% | 100.0% |
| sdpa | niah_multikey_3 | 100.00 | 2.066 | 0.908 | 100.0% | 100.0% |
| sdpa | niah_multivalue | 87.50 | 3.973 | 2.827 | 100.0% | 100.0% |
| sdpa | niah_multiquery | 100.00 | 3.982 | 2.843 | 100.0% | 100.0% |
| sdpa | vt | 100.00 | 2.009 | 0.771 | 100.0% | 100.0% |
| sdpa | cwe | 63.00 | 4.400 | 3.180 | 100.0% | 100.0% |
| sdpa | fwe | 70.00 | 2.514 | 1.314 | 100.0% | 100.0% |
| sdpa | qa_1 | 70.00 | 1.838 | 0.832 | 100.0% | 100.0% |
| sdpa | qa_2 | 50.00 | 1.715 | 0.645 | 100.0% | 100.0% |
| sdpa | __selected_task_mean__ | 87.73 | 2.874 | 1.776 | 100.0% | 100.0% |
| santapp | niah_single_1 | 80.00 | 46.049 | 34.392 | 3.2% | 6.3% |
| santapp | niah_single_2 | 10.00 | 58.265 | 47.186 | 3.4% | 6.5% |
| santapp | niah_single_3 | 0.00 | 58.356 | 47.231 | 3.4% | 6.5% |
| santapp | niah_multikey_1 | 20.00 | 54.199 | 43.028 | 3.4% | 6.4% |
| santapp | niah_multikey_2 | 0.00 | 31.032 | 19.693 | 3.2% | 6.2% |
| santapp | niah_multikey_3 | 0.00 | 51.836 | 40.355 | 3.2% | 6.3% |
| santapp | niah_multivalue | 7.50 | 58.462 | 47.251 | 3.4% | 6.4% |
| santapp | niah_multiquery | 7.50 | 53.057 | 41.855 | 3.3% | 6.4% |
| santapp | vt | 70.00 | 22.727 | 11.098 | 2.6% | 5.7% |
| santapp | cwe | 33.00 | 56.124 | 44.478 | 3.2% | 6.2% |
| santapp | fwe | 66.67 | 30.061 | 18.664 | 2.8% | 5.9% |
| santapp | qa_1 | 40.00 | 21.835 | 11.879 | 3.2% | 6.3% |
| santapp | qa_2 | 20.00 | 22.380 | 11.868 | 3.0% | 6.1% |
| santapp | __selected_task_mean__ | 27.28 | 43.414 | 32.229 | 3.2% | 6.3% |

## Paired aggregate comparison

- SANTA++ minus SDPA selected-task mean score: -60.45 points
- SANTA++ decode KV access: 3.2%
- SANTA++ metadata-inclusive read-equivalent: 6.3%
- SANTA++ end-to-end speedup vs SDPA: 0.066x
- SANTA++ decode speedup vs SDPA: 0.055x
