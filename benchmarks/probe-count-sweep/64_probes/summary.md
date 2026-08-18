# SANTA / SANTA++ RULER benchmark summary

Task scores use RULER's task-family grader. The selected-task mean is an unweighted harness summary, not an additional RULER metric.

| Backend | Task | Score | Mean total s | Mean decode s | GQA + centroid | GQA KV | GQA centroid | Naive + centroid |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| santapp | niah_multivalue | 55.75 | 74.290 | 57.016 | 8.6% | 5.5% | 3.1% | 5.6% |
| santapp | niah_multiquery | 53.50 | 70.772 | 53.150 | 8.3% | 5.2% | 3.1% | 5.5% |
| santapp | __selected_task_mean__ | 54.62 | 72.531 | 55.083 | 8.5% | 5.4% | 3.1% | 5.6% |
