from santapp_ruler.reporting import _is_pareto, aggregate_metrics


def _record(*, backend: str, score: float, access: int) -> dict:
    return {
        "pred": "x",
        "outputs": ["x"],
        "metrics": {
            "backend": backend,
            "generated_tokens": 2,
            "prompt_tokens": 10,
            "prefill_seconds": 1.0,
            "parent_build_seconds": 0.1,
            "decode_seconds": 0.5,
            "total_seconds": 1.6,
            "decode_dense_gqa_kv_vectors": 100,
            "decode_dense_naive_kv_vectors": 200,
            "decode_gqa_kv_vectors_read": access,
            "decode_naive_kv_vectors_read": access * 2,
            "decode_gqa_routing_key_vectors_read": 5,
            "decode_naive_routing_key_vectors_read": 10,
            "decode_gqa_centroid_key_vectors_read": 0,
            "decode_naive_centroid_key_vectors_read": 0,
            "decode_attention_head_calls": 1,
            "decode_gqa_group_calls": 1,
            "example_ruler_score": score,
        },
    }


def test_aggregate_metrics_recomputes_access_percentages() -> None:
    result = aggregate_metrics([_record(backend="santapp", score=100, access=20)])
    assert result["decode_gqa_kv_access_pct"] == 20.0
    assert result["decode_gqa_routing_access_pct"] == 5.0
    assert result["decode_gqa_total_access_pct"] == 25.0
    assert result["decode_tokens_per_second"] == 4.0


def test_pareto_detects_dominated_row() -> None:
    better = {"selected_task_mean_score": 90.0, "decode_gqa_total_access_pct": 20.0}
    worse = {"selected_task_mean_score": 80.0, "decode_gqa_total_access_pct": 30.0}
    rows = [better, worse]
    assert _is_pareto(better, rows)
    assert not _is_pareto(worse, rows)
