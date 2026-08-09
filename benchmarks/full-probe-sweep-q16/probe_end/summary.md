# SANTA++ RULER benchmark summary

Task scores below use RULER's task-family grader. The selected-task mean is an unweighted harness summary, not an additional RULER metric.

| Backend | Task | Score | Mean total s | Mean decode s | Decode KV access | Read-equivalent |
|---|---|---:|---:|---:|---:|---:|
| sdpa | niah_single_1 | 100.00 | 1.175 | 0.252 | 100.0% | 100.0% |
| sdpa | niah_single_2 | 100.00 | 3.822 | 2.973 | 100.0% | 100.0% |
| sdpa | niah_single_3 | 100.00 | 4.399 | 3.421 | 100.0% | 100.0% |
| sdpa | niah_multikey_1 | 100.00 | 4.074 | 2.977 | 100.0% | 100.0% |
| sdpa | niah_multikey_2 | 100.00 | 1.402 | 0.243 | 100.0% | 100.0% |
| sdpa | niah_multikey_3 | 100.00 | 2.140 | 0.922 | 100.0% | 100.0% |
| sdpa | niah_multivalue | 87.50 | 4.023 | 2.863 | 100.0% | 100.0% |
| sdpa | niah_multiquery | 100.00 | 3.967 | 2.850 | 100.0% | 100.0% |
| sdpa | vt | 100.00 | 1.982 | 0.783 | 100.0% | 100.0% |
| sdpa | cwe | 63.00 | 4.371 | 3.196 | 100.0% | 100.0% |
| sdpa | fwe | 70.00 | 2.477 | 1.327 | 100.0% | 100.0% |
| sdpa | qa_1 | 70.00 | 1.911 | 0.855 | 100.0% | 100.0% |
| sdpa | qa_2 | 50.00 | 1.829 | 0.651 | 100.0% | 100.0% |
| sdpa | __selected_task_mean__ | 87.73 | 2.890 | 1.793 | 100.0% | 100.0% |
| santapp | niah_single_1 | 90.00 | 45.267 | 33.642 | 3.1% | 6.2% |
| santapp | niah_single_2 | 60.00 | 55.595 | 44.476 | 3.4% | 6.4% |
| santapp | niah_single_3 | 10.00 | 58.904 | 47.752 | 3.4% | 6.5% |
| santapp | niah_multikey_1 | 50.00 | 53.899 | 42.787 | 3.3% | 6.4% |
| santapp | niah_multikey_2 | 40.00 | 30.213 | 18.891 | 3.2% | 6.2% |
| santapp | niah_multikey_3 | 0.00 | 47.344 | 35.774 | 3.2% | 6.2% |
| santapp | niah_multivalue | 32.50 | 59.112 | 47.848 | 3.4% | 6.4% |
| santapp | niah_multiquery | 27.50 | 54.167 | 42.895 | 3.3% | 6.4% |
| santapp | vt | 80.00 | 22.843 | 11.221 | 2.6% | 5.7% |
| santapp | cwe | 47.00 | 56.653 | 44.931 | 3.2% | 6.2% |
| santapp | fwe | 80.00 | 30.200 | 18.722 | 2.8% | 5.9% |
| santapp | qa_1 | 20.00 | 20.353 | 10.408 | 3.2% | 6.2% |
| santapp | qa_2 | 20.00 | 19.630 | 9.064 | 2.9% | 6.0% |
| santapp | __selected_task_mean__ | 42.85 | 42.629 | 31.416 | 3.2% | 6.3% |

## Paired aggregate comparison

- SANTA++ minus SDPA selected-task mean score: -44.88 points
- SANTA++ decode KV access: 3.2%
- SANTA++ metadata-inclusive read-equivalent: 6.3%
- SANTA++ end-to-end speedup vs SDPA: 0.068x
- SANTA++ decode speedup vs SDPA: 0.057x
