# SANTA / SANTA++ RULER benchmark summary

Task scores use RULER's task-family grader. The selected-task mean is an unweighted harness summary, not an additional RULER metric.

| Backend | Task | Score | Mean total s | Mean decode s | GQA + centroid | GQA KV | GQA centroid | Naive + centroid |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| santapp | niah_multivalue | 17.00 | 72.659 | 58.000 | 7.8% | 4.7% | 3.1% | 5.6% |
| santapp | niah_multiquery | 19.00 | 74.890 | 59.550 | 7.8% | 4.7% | 3.1% | 5.6% |
| santapp | __selected_task_mean__ | 18.00 | 73.775 | 58.775 | 7.8% | 4.7% | 3.1% | 5.6% |
