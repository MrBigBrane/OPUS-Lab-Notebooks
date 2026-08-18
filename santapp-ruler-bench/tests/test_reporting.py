import json
from pathlib import Path

import pytest

from santapp_ruler.reporting import build_reports


def _metric_payload(backend: str, mode: str, budget: int | None, index: int):
    dense_gqa = 1000
    dense_naive = 4000
    if backend == "sdpa":
        gqa_kv = dense_gqa
        naive_kv = dense_naive
        gqa_centroid = 0
        naive_centroid = 0
        probe_policy = "not_applicable"
        exact_policy = "dense_all_tokens"
    else:
        gqa_kv = 100 + index
        naive_kv = 400 + 4 * index
        gqa_centroid = 25
        naive_centroid = 100
        probe_policy = "last_prompt_tokens"
        exact_policy = "generated_suffix_only"
    return {
        "mode": mode,
        "sampling_scheme": "test",
        "importance_correction": "test",
        "sampling_unit": "cluster" if mode == "gumbel_cluster" else "token",
        "samples_per_head": budget,
        "nominal_sample_budget_per_head": budget,
        "clusters_per_head": 8 if mode == "gumbel_cluster" else None,
        "group_size": 16 if backend != "sdpa" else None,
        "probe_queries": 64 if backend != "sdpa" else None,
        "probe_policy": probe_policy,
        "exact_token_policy": exact_policy,
        "prompt_exact_tail_tokens": 0,
        "initial_growing_exact_tokens": 0,
        "prompt_tokens": 100,
        "generated_tokens": 2,
        "prefill_seconds": 1.0,
        "clustering_seconds": 0.5 if backend != "sdpa" else 0.0,
        "decode_seconds": 2.0,
        "total_seconds": 3.5 if backend != "sdpa" else 3.0,
        "decode_attention_head_calls": 4,
        "decode_gqa_group_calls": 2,
        "decode_dense_gqa_kv_vectors": dense_gqa,
        "decode_dense_naive_kv_vectors": dense_naive,
        "decode_gqa_kv_vectors_read": gqa_kv,
        "decode_naive_kv_vectors_read": naive_kv,
        "decode_gqa_centroid_key_vectors_read": gqa_centroid,
        "decode_naive_centroid_key_vectors_read": naive_centroid,
        "decode_gqa_total_vectors_read": gqa_kv + gqa_centroid,
        "decode_naive_total_vectors_read": naive_kv + naive_centroid,
        "decode_gqa_kv_access_pct": 100.0 * gqa_kv / dense_gqa,
        "decode_gqa_centroid_access_pct": 100.0 * gqa_centroid / dense_gqa,
        "decode_gqa_total_access_pct": 100.0 * (gqa_kv + gqa_centroid) / dense_gqa,
        "decode_naive_kv_access_pct": 100.0 * naive_kv / dense_naive,
        "decode_naive_centroid_access_pct": 100.0 * naive_centroid / dense_naive,
        "decode_naive_total_access_pct": 100.0 * (naive_kv + naive_centroid) / dense_naive,
        "mean_sampled_token_draws_per_head_call": float(budget or 0),
        "mean_sampled_token_rows_per_head_call": float(budget or 0),
        "mean_unique_sampled_tokens_per_gqa_group_call": float(budget or 0),
        "mean_exact_tokens_per_gqa_group_call": 1.0,
        "mean_total_tokens_per_head_call": 100.0,
        "mean_selected_clusters_per_head_call": 8.0 if mode == "gumbel_cluster" else 0.0,
        "selected_cluster_inclusion_probability_count": 16 if mode == "gumbel_cluster" else 0,
        "mean_selected_cluster_inclusion_probability": 0.5
        if mode == "gumbel_cluster"
        else None,
        "min_selected_cluster_inclusion_probability": 0.2 if mode == "gumbel_cluster" else None,
        "max_selected_cluster_inclusion_probability": 0.8 if mode == "gumbel_cluster" else None,
        "peak_allocated_gib": 8.0,
        "peak_reserved_gib": 9.0,
        "cluster_summary_gib": 0.0 if backend == "sdpa" else 0.25,
        "routing_summary_gib": 0.0 if backend == "sdpa" else 0.25,
    }


def _write_records(
    root: Path,
    backend: str,
    task: str,
    predictions: list[str],
    *,
    mode: str,
    budget: int | None,
) -> None:
    path = root / "predictions" / backend / f"{task}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, prediction in enumerate(predictions):
        rows.append(
            {
                "index": index,
                "uid": f"{task}:{index}",
                "task": task,
                "input": "prompt",
                "outputs": ["answer"],
                "pred": prediction,
                "metrics": _metric_payload(backend, mode, budget, index),
            }
        )
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _update_metrics(
    root: Path,
    backend: str,
    task: str,
    updates: list[dict[str, object]],
) -> None:
    path = root / "predictions" / backend / f"{task}.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == len(updates)
    for record, metric_updates in zip(records, updates, strict=True):
        record["metrics"].update(metric_updates)
    path.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")


def test_report_builds_primary_gqa_metrics_and_official_outputs(tmp_path: Path):
    _write_records(tmp_path, "sdpa", "vt", ["answer"], mode="dense_sdpa", budget=None)
    _write_records(tmp_path, "santapp", "vt", ["answer"], mode="guided", budget=128)

    summary = build_reports(
        tmp_path,
        backends=["sdpa", "santapp"],
        tasks=["vt"],
        bootstrap_resamples=100,
    )
    assert summary["backends"]["sdpa"]["aggregate"]["score"] == 100.0
    assert summary["backends"]["sdpa"]["aggregate"]["mean_routing_summary_gib"] == 0.0
    sparse = summary["backends"]["santapp"]["aggregate"]
    assert sparse["decode_gqa_kv_access_pct"] == 10.0
    assert sparse["decode_gqa_routing_access_pct"] == 2.5
    assert sparse["decode_gqa_centroid_access_pct"] == 2.5
    assert sparse["decode_gqa_total_access_pct"] == 12.5
    assert sparse["decode_naive_total_access_pct"] == 12.5
    assert (tmp_path / "summary.md").is_file()
    assert (tmp_path / "summary.json").is_file()
    assert (tmp_path / "predictions" / "sdpa" / "summary.csv").is_file()
    assert "GQA routing K" in (tmp_path / "summary.md").read_text(encoding="utf-8")


def test_reports_bootstrap_and_matched_budget_paired_delta(tmp_path: Path):
    backends = ["sdpa", "guided128", "gumbel128"]
    for task in ("vt", "qa_1"):
        _write_records(
            tmp_path,
            "sdpa",
            task,
            ["answer", "answer"],
            mode="dense_sdpa",
            budget=None,
        )
        _write_records(
            tmp_path,
            "guided128",
            task,
            ["answer", "wrong"],
            mode="guided",
            budget=128,
        )
        _write_records(
            tmp_path,
            "gumbel128",
            task,
            ["answer", "answer"],
            mode="gumbel_cluster",
            budget=128,
        )

    summary = build_reports(
        tmp_path,
        backends=backends,
        tasks=["vt", "qa_1"],
        bootstrap_resamples=500,
        confidence_level=0.95,
        bootstrap_seed=11,
    )
    assert summary["grading"]["bootstrap_resamples"] == 500
    guided_ci = summary["backends"]["guided128"]["score_confidence_interval"]
    assert guided_ci["lower"] <= 50.0 <= guided_ci["upper"]
    comparisons = summary["comparisons"]["vs_sdpa"]
    assert set(comparisons) == {"guided128", "gumbel128"}
    assert comparisons["guided128"]["score_delta_candidate_minus_reference"] == -50.0
    assert comparisons["gumbel128"]["score_delta_candidate_minus_reference"] == 0.0
    assert comparisons["guided128"]["confidence_interval"]["resamples"] == 500
    assert (tmp_path / "comparisons.csv").is_file()


def test_paired_comparison_rejects_uid_mismatch(tmp_path: Path):
    _write_records(
        tmp_path,
        "sdpa",
        "vt",
        ["answer", "answer"],
        mode="dense_sdpa",
        budget=None,
    )
    _write_records(
        tmp_path,
        "guided128",
        "vt",
        ["answer", "wrong"],
        mode="guided",
        budget=128,
    )
    path = tmp_path / "predictions" / "guided128" / "vt.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    records[1]["uid"] = "vt:different"
    path.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")

    with pytest.raises(ValueError, match="identical prompt UIDs"):
        build_reports(
            tmp_path,
            backends=["sdpa", "guided128"],
            tasks=["vt"],
            bootstrap_resamples=100,
        )


def test_reports_team_rows_with_generic_routing_metrics(tmp_path: Path):
    _write_records(
        tmp_path,
        "sdpa",
        "vt",
        ["answer", "answer"],
        mode="dense_sdpa",
        budget=None,
    )

    team_common = {
        "group_size": None,
        "routing_key_type": "actual_team_leader",
        "decode_gqa_routing_key_vectors_read": 40,
        "decode_naive_routing_key_vectors_read": 160,
        "decode_gqa_centroid_key_vectors_read": 0,
        "decode_naive_centroid_key_vectors_read": 0,
        "parent_size": 16,
        "representatives_per_parent": 4,
        "nominal_team_size": 4.0,
        "nominal_parent_clusters_per_kv_head": 7,
        "representative_selection": "mean_nearest_then_farthest_first",
        "parent_clustering_space": "probe_fingerprint",
        "team_assignment_space": "actual_key_l2",
        "active_parent_count_total": 14,
        "mean_active_parents_per_layer_kv_head": 7.0,
        "mean_active_teams_per_layer_kv_head": 4.0,
        "routing_summary_gib": 0.25,
    }
    for backend, mode, predictions in (
        ("team-sampler-p16-r4-s128", "team_sampler", ["answer", "wrong"]),
        (
            "team-gumbel-p16-r4-s128",
            "team_gumbel_topk",
            ["answer", "answer"],
        ),
    ):
        _write_records(
            tmp_path,
            backend,
            "vt",
            predictions,
            mode=mode,
            budget=128,
        )
        updates = []
        for index, (count, mean, std) in enumerate(((2, 3.0, 1.0), (6, 5.0, 2.0))):
            updates.append(
                {
                    **team_common,
                    "nominal_parent_clusters_per_kv_head": 7 + index,
                    "active_team_count_total": count,
                    "team_size_mean": mean,
                    "team_size_std": std,
                    "team_size_cv": std / mean,
                    "team_size_min": 1,
                    "team_size_median": mean,
                    "team_size_p90": mean + 1.0,
                    "team_size_max": 8 + index,
                    "teams_per_head": 32 if mode == "team_gumbel_topk" else None,
                    "mean_selected_teams_per_head_call": (
                        32.0 if mode == "team_gumbel_topk" else 0.0
                    ),
                    "selected_team_inclusion_probability_count": (
                        64 if mode == "team_gumbel_topk" else 0
                    ),
                    "mean_selected_team_inclusion_probability": (
                        0.5 if mode == "team_gumbel_topk" else None
                    ),
                    "min_selected_team_inclusion_probability": (
                        0.25 if mode == "team_gumbel_topk" else None
                    ),
                    "max_selected_team_inclusion_probability": (
                        0.75 if mode == "team_gumbel_topk" else None
                    ),
                }
            )
        _update_metrics(tmp_path, backend, "vt", updates)

    summary = build_reports(
        tmp_path,
        backends=[
            "sdpa",
            "team-sampler-p16-r4-s128",
            "team-gumbel-p16-r4-s128",
        ],
        tasks=["vt"],
        bootstrap_resamples=100,
    )

    team = summary["backends"]["team-sampler-p16-r4-s128"]["aggregate"]
    assert team["decode_gqa_routing_key_vectors_read"] == 80
    assert team["decode_gqa_centroid_key_vectors_read"] == 0
    assert team["decode_gqa_total_vectors_read"] == 281
    assert team["decode_gqa_total_access_pct"] == pytest.approx(14.05)
    assert team["parent_size"] == 16
    assert team["nominal_parent_clusters_per_kv_head"] == pytest.approx(7.5)
    assert team["mean_nominal_parent_clusters_per_kv_head"] == pytest.approx(7.5)
    assert team["active_team_count_total"] == 8
    assert team["team_size_mean"] == pytest.approx(4.5)
    assert team["team_size_std"] == pytest.approx(2.0)
    assert set(summary["comparisons"]["vs_sdpa"]) == {
        "team-sampler-p16-r4-s128",
        "team-gumbel-p16-r4-s128",
    }
