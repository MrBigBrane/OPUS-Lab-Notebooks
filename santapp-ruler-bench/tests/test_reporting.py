import json
from pathlib import Path

from santapp_ruler.reporting import build_reports


def _access_metrics(backend: str) -> dict:
    common = {
        "decode_attention_head_calls": 4,
        "decode_gqa_group_calls": 1,
        "decode_dense_gqa_kv_vectors": 200,
        "decode_dense_naive_kv_vectors": 800,
        "mean_sampled_token_draws_per_head_call": 2.0,
        "mean_unique_sampled_tokens_per_gqa_group_call": 3.0,
        "mean_exact_tokens_per_gqa_group_call": 1.0,
    }
    if backend == "sdpa":
        return {
            **common,
            "decode_gqa_kv_vectors_read": 200,
            "decode_naive_kv_vectors_read": 800,
            "decode_gqa_centroid_key_vectors_read": 0,
            "decode_naive_centroid_key_vectors_read": 0,
        }
    if backend == "santa":
        return {
            **common,
            "decode_gqa_kv_vectors_read": 120,
            "decode_naive_kv_vectors_read": 440,
            "decode_gqa_centroid_key_vectors_read": 0,
            "decode_naive_centroid_key_vectors_read": 0,
        }
    return {
        **common,
        "decode_gqa_kv_vectors_read": 100,
        "decode_naive_kv_vectors_read": 160,
        "decode_gqa_centroid_key_vectors_read": 20,
        "decode_naive_centroid_key_vectors_read": 80,
    }


def _write_prediction(root: Path, backend: str, task: str, pred: str) -> None:
    path = root / "predictions" / backend / f"{task}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "index": 0,
        "uid": f"{task}:0",
        "task": task,
        "input": "prompt",
        "outputs": ["answer"],
        "pred": pred,
        "metrics": {
            "prompt_tokens": 100,
            "generated_tokens": 2,
            "prefill_seconds": 1.0,
            "clustering_seconds": 0.5 if backend == "santapp" else 0.0,
            "decode_seconds": 2.0,
            "total_seconds": 3.5 if backend == "santapp" else 3.0,
            "peak_allocated_gib": 8.0,
            **_access_metrics(backend),
        },
    }
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")


def test_report_builds_official_and_harness_outputs(tmp_path: Path):
    backends = ["sdpa", "santa", "santapp"]
    for backend in backends:
        _write_prediction(tmp_path, backend, "vt", "answer")

    summary = build_reports(tmp_path, backends=backends, tasks=["vt"])

    assert summary["backends"]["sdpa"]["aggregate"]["score"] == 100.0
    assert summary["backends"]["sdpa"]["aggregate"][
        "decode_gqa_total_access_pct"
    ] == 100.0
    assert summary["backends"]["santa"]["aggregate"][
        "decode_gqa_total_access_pct"
    ] == 60.0
    assert summary["backends"]["santapp"]["aggregate"][
        "decode_gqa_kv_access_pct"
    ] == 50.0
    assert summary["backends"]["santapp"]["aggregate"][
        "decode_gqa_centroid_access_pct"
    ] == 10.0
    assert summary["backends"]["santapp"]["aggregate"][
        "decode_gqa_total_access_pct"
    ] == 60.0
    assert summary["backends"]["santapp"]["aggregate"][
        "decode_naive_total_access_pct"
    ] == 30.0
    assert set(summary["comparisons"]) == {"santa_vs_sdpa", "santapp_vs_sdpa"}
    assert (tmp_path / "summary.md").is_file()
    assert (tmp_path / "predictions" / "sdpa" / "summary.csv").is_file()
    rendered = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "GQA + centroid" in rendered


def test_access_percentages_pool_raw_counts_instead_of_averaging_examples():
    from santapp_ruler.reporting import _aggregate_metrics

    records = [
        {
            "metrics": {
                "decode_dense_gqa_kv_vectors": 100,
                "decode_dense_naive_kv_vectors": 400,
                "decode_gqa_kv_vectors_read": 40,
                "decode_gqa_centroid_key_vectors_read": 10,
                "decode_naive_kv_vectors_read": 120,
                "decode_naive_centroid_key_vectors_read": 40,
            }
        },
        {
            "metrics": {
                "decode_dense_gqa_kv_vectors": 900,
                "decode_dense_naive_kv_vectors": 3600,
                "decode_gqa_kv_vectors_read": 810,
                "decode_gqa_centroid_key_vectors_read": 90,
                "decode_naive_kv_vectors_read": 3240,
                "decode_naive_centroid_key_vectors_read": 360,
            }
        },
    ]

    aggregate = _aggregate_metrics(records)
    # Pooled GQA total: (40+10+810+90)/(100+900) = 95%, not the
    # unweighted mean of the two per-example totals (75%).
    assert aggregate["decode_gqa_total_access_pct"] == 95.0
    assert aggregate["decode_gqa_kv_access_pct"] == 85.0
    assert aggregate["decode_gqa_centroid_access_pct"] == 10.0
    assert aggregate["decode_naive_total_access_pct"] == 94.0
