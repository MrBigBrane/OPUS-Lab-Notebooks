# SANTA / SANTA++ RULER benchmark summary

Task scores use RULER's task-family grader. The selected-task mean is an unweighted harness summary, not an additional RULER metric.

| Backend | Task | Score | Mean total s | Mean decode s | GQA + centroid | GQA KV | GQA centroid | Naive + centroid |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| santapp | niah_multivalue | 4.00 | 75.273 | 60.069 | 7.7% | 4.6% | 3.1% | 5.6% |
| santapp | niah_multiquery | 3.25 | 68.482 | 54.513 | 7.6% | 4.5% | 3.1% | 5.6% |
| santapp | __selected_task_mean__ | 3.62 | 71.878 | 57.291 | 7.6% | 4.5% | 3.1% | 5.6% |
