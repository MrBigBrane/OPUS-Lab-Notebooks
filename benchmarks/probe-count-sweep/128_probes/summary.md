# SANTA / SANTA++ RULER benchmark summary

Task scores use RULER's task-family grader. The selected-task mean is an unweighted harness summary, not an additional RULER metric.

| Backend | Task | Score | Mean total s | Mean decode s | GQA + centroid | GQA KV | GQA centroid | Naive + centroid |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| santapp | niah_multivalue | 58.25 | 52.928 | 37.844 | 8.6% | 5.5% | 3.1% | 5.6% |
| santapp | niah_multiquery | 56.50 | 50.318 | 35.164 | 8.4% | 5.3% | 3.1% | 5.5% |
| santapp | __selected_task_mean__ | 57.38 | 51.623 | 36.504 | 8.5% | 5.4% | 3.1% | 5.6% |
