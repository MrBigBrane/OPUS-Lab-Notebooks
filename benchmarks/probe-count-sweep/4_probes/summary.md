# SANTA / SANTA++ RULER benchmark summary

Task scores use RULER's task-family grader. The selected-task mean is an unweighted harness summary, not an additional RULER metric.

| Backend | Task | Score | Mean total s | Mean decode s | GQA + centroid | GQA KV | GQA centroid | Naive + centroid |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| santapp | niah_multivalue | 36.00 | 80.919 | 64.790 | 8.3% | 5.2% | 3.1% | 5.6% |
| santapp | niah_multiquery | 30.25 | 75.036 | 59.352 | 8.1% | 5.0% | 3.1% | 5.6% |
| santapp | __selected_task_mean__ | 33.12 | 77.977 | 62.071 | 8.2% | 5.1% | 3.1% | 5.6% |
