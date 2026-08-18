# SANTA / SANTA++ RULER benchmark summary

Task scores use RULER's task-family grader. The selected-task mean is an unweighted harness summary, not an additional RULER metric.

| Backend | Task | Score | Mean total s | Mean decode s | GQA + centroid | GQA KV | GQA centroid | Naive + centroid |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| santapp | niah_multivalue | 53.50 | 62.648 | 40.446 | 8.5% | 5.4% | 3.1% | 5.6% |
| santapp | niah_multiquery | 58.50 | 60.143 | 37.853 | 8.4% | 5.3% | 3.1% | 5.6% |
| santapp | __selected_task_mean__ | 56.00 | 61.395 | 39.150 | 8.5% | 5.4% | 3.1% | 5.6% |
