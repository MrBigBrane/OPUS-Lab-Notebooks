# SANTA++ RULER benchmark summary

Task scores below use RULER's task-family grader. The selected-task mean is an unweighted harness summary, not an additional RULER metric.

| Backend | Task | Score | Mean total s | Mean decode s | Decode KV access | Read-equivalent |
|---|---|---:|---:|---:|---:|---:|
| sdpa | niah_single_1 | 100.00 | 1.375 | 0.331 | 100.0% | 100.0% |
| sdpa | niah_single_2 | 100.00 | 4.917 | 3.925 | 100.0% | 100.0% |
| sdpa | niah_single_3 | 100.00 | 5.533 | 4.520 | 100.0% | 100.0% |
| sdpa | niah_multikey_1 | 100.00 | 5.009 | 4.004 | 100.0% | 100.0% |
| sdpa | niah_multikey_2 | 100.00 | 1.331 | 0.322 | 100.0% | 100.0% |
| sdpa | niah_multikey_3 | 100.00 | 2.251 | 1.234 | 100.0% | 100.0% |
| sdpa | niah_multivalue | 87.50 | 4.749 | 3.739 | 100.0% | 100.0% |
| sdpa | niah_multiquery | 100.00 | 4.784 | 3.782 | 100.0% | 100.0% |
| sdpa | vt | 100.00 | 2.076 | 1.038 | 100.0% | 100.0% |
| sdpa | cwe | 63.00 | 5.283 | 4.237 | 100.0% | 100.0% |
| sdpa | fwe | 73.33 | 2.748 | 1.728 | 100.0% | 100.0% |
| sdpa | qa_1 | 70.00 | 1.946 | 1.098 | 100.0% | 100.0% |
| sdpa | qa_2 | 50.00 | 1.787 | 0.852 | 100.0% | 100.0% |
| sdpa | __selected_task_mean__ | 87.99 | 3.369 | 2.370 | 100.0% | 100.0% |
| santapp | niah_single_1 | 100.00 | 82.171 | 46.794 | 3.2% | 6.2% |
| santapp | niah_single_2 | 40.00 | 101.580 | 68.555 | 3.4% | 6.5% |
| santapp | niah_single_3 | 20.00 | 102.551 | 69.243 | 3.4% | 6.5% |
| santapp | niah_multikey_1 | 30.00 | 96.959 | 63.345 | 3.4% | 6.4% |
| santapp | niah_multikey_2 | 30.00 | 72.116 | 37.541 | 3.2% | 6.3% |
| santapp | niah_multikey_3 | 0.00 | 89.777 | 54.710 | 3.2% | 6.3% |
| santapp | niah_multivalue | 62.50 | 101.818 | 68.169 | 3.4% | 6.4% |
| santapp | niah_multiquery | 55.00 | 99.219 | 65.412 | 3.3% | 6.4% |
| santapp | vt | 88.00 | 52.076 | 16.133 | 2.6% | 5.7% |
| santapp | cwe | 49.00 | 102.387 | 66.151 | 3.2% | 6.2% |
| santapp | fwe | 80.00 | 62.163 | 27.528 | 2.8% | 5.9% |
| santapp | qa_1 | 40.00 | 44.832 | 17.458 | 3.2% | 6.3% |
| santapp | qa_2 | 30.00 | 46.126 | 15.818 | 3.0% | 6.1% |
| santapp | __selected_task_mean__ | 48.04 | 81.060 | 47.451 | 3.2% | 6.3% |

## Paired aggregate comparison

- SANTA++ minus SDPA selected-task mean score: -39.95 points
- SANTA++ decode KV access: 3.2%
- SANTA++ metadata-inclusive read-equivalent: 6.3%
- SANTA++ end-to-end speedup vs SDPA: 0.042x
- SANTA++ decode speedup vs SDPA: 0.050x
