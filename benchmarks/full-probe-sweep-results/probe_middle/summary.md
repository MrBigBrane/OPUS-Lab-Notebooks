# SANTA++ RULER benchmark summary

Task scores below use RULER's task-family grader. The selected-task mean is an unweighted harness summary, not an additional RULER metric.

| Backend | Task | Score | Mean total s | Mean decode s | Decode KV access | Read-equivalent |
|---|---|---:|---:|---:|---:|---:|
| sdpa | niah_single_1 | 100.00 | 1.380 | 0.317 | 100.0% | 100.0% |
| sdpa | niah_single_2 | 100.00 | 4.729 | 3.713 | 100.0% | 100.0% |
| sdpa | niah_single_3 | 100.00 | 5.303 | 4.277 | 100.0% | 100.0% |
| sdpa | niah_multikey_1 | 100.00 | 4.722 | 3.709 | 100.0% | 100.0% |
| sdpa | niah_multikey_2 | 100.00 | 1.326 | 0.309 | 100.0% | 100.0% |
| sdpa | niah_multikey_3 | 100.00 | 2.219 | 1.194 | 100.0% | 100.0% |
| sdpa | niah_multivalue | 87.50 | 4.748 | 3.734 | 100.0% | 100.0% |
| sdpa | niah_multiquery | 100.00 | 4.675 | 3.668 | 100.0% | 100.0% |
| sdpa | vt | 100.00 | 2.045 | 1.002 | 100.0% | 100.0% |
| sdpa | cwe | 63.00 | 5.186 | 4.138 | 100.0% | 100.0% |
| sdpa | fwe | 73.33 | 2.722 | 1.699 | 100.0% | 100.0% |
| sdpa | qa_1 | 70.00 | 1.917 | 1.067 | 100.0% | 100.0% |
| sdpa | qa_2 | 50.00 | 1.748 | 0.809 | 100.0% | 100.0% |
| sdpa | __selected_task_mean__ | 87.99 | 3.286 | 2.280 | 100.0% | 100.0% |
| santapp | niah_single_1 | 90.00 | 56.538 | 38.340 | 3.1% | 6.2% |
| santapp | niah_single_2 | 40.00 | 84.698 | 67.570 | 3.4% | 6.5% |
| santapp | niah_single_3 | 0.00 | 82.542 | 65.594 | 3.4% | 6.5% |
| santapp | niah_multikey_1 | 10.00 | 77.469 | 60.324 | 3.4% | 6.4% |
| santapp | niah_multikey_2 | 0.00 | 58.631 | 41.102 | 3.2% | 6.3% |
| santapp | niah_multikey_3 | 0.00 | 66.959 | 48.977 | 3.2% | 6.3% |
| santapp | niah_multivalue | 35.00 | 85.467 | 68.048 | 3.4% | 6.4% |
| santapp | niah_multiquery | 15.00 | 80.733 | 63.244 | 3.3% | 6.4% |
| santapp | vt | 88.00 | 33.690 | 15.581 | 2.6% | 5.7% |
| santapp | cwe | 26.00 | 82.100 | 63.748 | 3.2% | 6.2% |
| santapp | fwe | 66.67 | 44.007 | 26.349 | 2.8% | 5.9% |
| santapp | qa_1 | 40.00 | 30.954 | 16.508 | 3.2% | 6.3% |
| santapp | qa_2 | 10.00 | 31.932 | 16.359 | 3.0% | 6.1% |
| santapp | __selected_task_mean__ | 32.36 | 62.748 | 45.519 | 3.2% | 6.3% |

## Paired aggregate comparison

- SANTA++ minus SDPA selected-task mean score: -55.63 points
- SANTA++ decode KV access: 3.2%
- SANTA++ metadata-inclusive read-equivalent: 6.3%
- SANTA++ end-to-end speedup vs SDPA: 0.052x
- SANTA++ decode speedup vs SDPA: 0.050x
