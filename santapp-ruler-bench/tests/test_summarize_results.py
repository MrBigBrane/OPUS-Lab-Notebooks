import csv
import json
import tarfile
from pathlib import Path

from scripts import summarize_results as analysis

SETTINGS = [
    {
        "name": "sdpa",
        "backend": "sdpa",
        "description": "Dense reference",
    },
    {
        "name": "santa-s128",
        "backend": "santa",
        "samples_per_head": 128,
    },
    {
        "name": "santapp-b16-s128",
        "backend": "santapp",
        "mode": "guided",
        "group_size": 16,
        "samples_per_head": 128,
    },
    {
        "name": "santapp-parent-s128",
        "backend": "santapp_gumbel_topk",
        "mode": "gumbel_cluster",
        "group_size": 16,
        "samples_per_head": 128,
    },
    {
        "name": "team-sampler-s128",
        "backend": "team_sampler",
        "mode": "team_sampler",
        "parent_size": 16,
        "representatives_per_parent": 4,
        "samples_per_head": 128,
    },
    {
        "name": "team-whole-s128",
        "backend": "team_gumbel_topk",
        "mode": "team_gumbel_topk",
        "parent_size": 16,
        "representatives_per_parent": 4,
        "samples_per_head": 128,
    },
]
TASKS = ["niah_multiquery", "niah_multivalue"]


def _row(setting: dict[str, object], task: str, index: int) -> dict[str, object]:
    name = str(setting["name"])
    dense = 1000
    order = [str(value["name"]) for value in SETTINGS].index(name)
    if name == "sdpa":
        kv = 1000
        routing = 0
        score = 100.0
    else:
        kv = 80 + order * 20 + index
        routing = 10 + order
        score = 70.0 + order * 4 + index
    return {
        "experiment": "synthetic-analysis",
        "setting": name,
        "backend": setting["backend"],
        "task": task,
        "index": index,
        "uid": f"{task}:{index}",
        "example_ruler_score": score,
        "prediction": "answer",
        "references": '["answer"]',
        "metric_decode_dense_gqa_kv_vectors": dense,
        "metric_decode_gqa_kv_vectors_read": kv,
        "metric_decode_gqa_routing_key_vectors_read": routing,
        "metric_decode_gqa_centroid_key_vectors_read": (
            routing if setting.get("mode") in {"guided", "gumbel_cluster"} else 0
        ),
        "metric_decode_dense_naive_kv_vectors": 4000,
        "metric_decode_naive_kv_vectors_read": kv * 4,
        "metric_decode_naive_routing_key_vectors_read": routing * 4,
        "metric_decode_naive_centroid_key_vectors_read": (
            routing * 4
            if setting.get("mode") in {"guided", "gumbel_cluster"}
            else 0
        ),
    }


def _write_export(root: Path) -> Path:
    export = root / "synthetic-analysis"
    export.mkdir(parents=True)
    matrix = {
        "schema_version": 1,
        "experiment": "synthetic-analysis",
        "base_config": "configs/k8s_default_8k.yaml",
        "prompts_per_task": 2,
        "tasks": TASKS,
        "settings": SETTINGS,
    }
    (export / "matrix.json").write_text(json.dumps(matrix), encoding="utf-8")
    (export / "export-status.json").write_text(
        json.dumps({"status": "complete"}), encoding="utf-8"
    )
    rows = [
        _row(setting, task, index)
        for setting in SETTINGS
        for task in TASKS
        for index in range(2)
    ]
    with (export / "compact-results.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return export


def test_pooled_access_recomputes_percentages_from_raw_counts():
    records = [
        {
            "metric_decode_dense_gqa_kv_vectors": "1000",
            "metric_decode_gqa_kv_vectors_read": "100",
            "metric_decode_gqa_routing_key_vectors_read": "25",
            "metric_decode_gqa_centroid_key_vectors_read": "10",
            "metric_decode_dense_naive_kv_vectors": "4000",
            "metric_decode_naive_kv_vectors_read": "400",
            "metric_decode_naive_routing_key_vectors_read": "100",
            "metric_decode_naive_centroid_key_vectors_read": "40",
        },
        {
            "metric_decode_dense_gqa_kv_vectors": "1000",
            "metric_decode_gqa_kv_vectors_read": "200",
            "metric_decode_gqa_routing_key_vectors_read": "25",
            "metric_decode_gqa_centroid_key_vectors_read": "10",
            "metric_decode_dense_naive_kv_vectors": "4000",
            "metric_decode_naive_kv_vectors_read": "800",
            "metric_decode_naive_routing_key_vectors_read": "100",
            "metric_decode_naive_centroid_key_vectors_read": "40",
        },
    ]
    pooled = analysis.pooled_access(records)
    assert pooled["accounting_method"] == "pooled_vector_counts"
    assert pooled["decode_gqa_kv_access_pct"] == 15.0
    assert pooled["decode_gqa_routing_access_pct"] == 2.5
    assert pooled["decode_gqa_centroid_access_pct"] == 1.0
    assert pooled["decode_gqa_total_access_pct"] == 17.5
    assert pooled["decode_naive_total_access_pct"] == 17.5


def test_pooled_access_does_not_pool_partial_raw_counter_coverage():
    records = [
        {
            "metric_decode_dense_gqa_kv_vectors": "1000",
            "metric_decode_gqa_kv_vectors_read": "100",
            "metric_decode_gqa_routing_key_vectors_read": "25",
            "metric_decode_dense_naive_kv_vectors": "4000",
            "metric_decode_naive_kv_vectors_read": "400",
            "metric_decode_naive_routing_key_vectors_read": "100",
            "metric_decode_gqa_kv_access_pct": "10",
            "metric_decode_gqa_routing_access_pct": "2.5",
            "metric_decode_gqa_total_access_pct": "12.5",
            "metric_decode_naive_kv_access_pct": "10",
            "metric_decode_naive_routing_access_pct": "2.5",
            "metric_decode_naive_total_access_pct": "12.5",
        },
        {
            "metric_decode_gqa_kv_access_pct": "20",
            "metric_decode_gqa_routing_access_pct": "2.5",
            "metric_decode_gqa_total_access_pct": "22.5",
            "metric_decode_naive_kv_access_pct": "20",
            "metric_decode_naive_routing_access_pct": "2.5",
            "metric_decode_naive_total_access_pct": "22.5",
        },
    ]

    pooled = analysis.pooled_access(records)
    assert pooled["accounting_method"] == "mean_record_percentages"
    assert pooled["decode_dense_gqa_kv_vectors"] is None
    assert pooled["decode_gqa_kv_access_pct"] == 15.0
    assert pooled["decode_gqa_routing_access_pct"] == 2.5
    assert pooled["decode_gqa_total_access_pct"] == 17.5
    assert pooled["decode_naive_total_access_pct"] == 17.5


def test_pooled_access_requires_routing_counts_for_raw_total_accounting():
    records = [
        {
            "metric_decode_dense_gqa_kv_vectors": "1000",
            "metric_decode_gqa_kv_vectors_read": "100",
            "metric_decode_dense_naive_kv_vectors": "4000",
            "metric_decode_naive_kv_vectors_read": "400",
            "metric_decode_gqa_kv_access_pct": "10",
            "metric_decode_gqa_routing_access_pct": "2",
            "metric_decode_gqa_total_access_pct": "12",
            "metric_decode_naive_kv_access_pct": "10",
            "metric_decode_naive_routing_access_pct": "2",
            "metric_decode_naive_total_access_pct": "12",
        }
    ]

    pooled = analysis.pooled_access(records)
    assert pooled["accounting_method"] == "mean_record_percentages"
    assert pooled["decode_gqa_total_vectors_read"] is None
    assert pooled["decode_gqa_total_access_pct"] == 12.0
    assert pooled["decode_naive_total_vectors_read"] is None
    assert pooled["decode_naive_total_access_pct"] == 12.0


def test_setting_labels_use_method_specific_metadata():
    labels = [analysis.setting_label(str(row["name"]), row) for row in SETTINGS]
    assert labels[0] == "SDPA"
    assert "SANTA++ IID" in labels[2]
    assert "whole-parent sampling" in labels[3]
    assert "Hierarchical team sampling" in labels[4]
    assert "Whole-team sampling" in labels[5]
    assert analysis.setting_label(
        "same-backend-parent",
        {
            "backend": "santapp",
            "mode": "gumbel_cluster",
            "group_size": 16,
            "samples_per_head": 128,
        },
    ).startswith("SANTA++ whole-parent sampling")
    assert analysis.setting_label("custom-method", {"backend": "custom"}) == (
        "custom-method"
    )


def test_analysis_discovers_all_settings_and_writes_one_output_tree(tmp_path: Path):
    export = _write_export(tmp_path / "exports")
    output = tmp_path / "analysis"
    result = analysis.main(
        [
            "--input",
            str(export.parent),
            "--output-dir",
            str(output),
            "--bootstrap-resamples",
            "50",
        ]
    )
    assert result == 0
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert [entry["name"] for entry in summary["settings"]] == [
        str(setting["name"]) for setting in SETTINGS
    ]
    assert summary["tasks"] == TASKS
    assert len(summary["accuracy_overall"]) == 6
    assert len(summary["paired_comparisons"]) == 5
    assert (output / "plots" / "pareto_selected_task_mean.png").is_file()
    assert (output / "plots" / "pareto_selected_task_mean.pdf").is_file()
    markdown = (output / "summary.md").read_text(encoding="utf-8")
    assert "SANTA++ whole-parent sampling" in markdown


def test_analysis_accepts_export_archive_and_can_disable_reference(tmp_path: Path):
    export = _write_export(tmp_path / "source")
    archive = tmp_path / "results.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(export, arcname=export.name)
    output = tmp_path / "from-archive"
    analysis.main(
        [
            "--input",
            str(archive),
            "--output-dir",
            str(output),
            "--reference-setting",
            "none",
            "--bootstrap-resamples",
            "20",
        ]
    )
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["paired_comparisons"] == []
    assert (output / "accuracy_matrix.csv").is_file()
