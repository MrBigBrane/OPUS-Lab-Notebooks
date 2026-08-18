# SANTA / SANTA++ RULER benchmark summary

Task scores use RULER's task-family grader. The selected-task mean is an unweighted harness summary, not an additional RULER metric.

| Backend | Task | Score | Mean total s | Mean decode s | GQA + centroid | GQA KV | GQA centroid | Naive + centroid |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| santapp | niah_multivalue | 53.25 | 74.475 | 59.136 | 8.6% | 5.5% | 3.1% | 5.6% |
| santapp | niah_multiquery | 42.75 | 64.753 | 50.313 | 8.3% | 5.2% | 3.1% | 5.5% |
| santapp | __selected_task_mean__ | 48.00 | 69.614 | 54.724 | 8.4% | 5.3% | 3.1% | 5.6% |
