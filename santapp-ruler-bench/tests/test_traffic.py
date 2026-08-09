import pytest
import torch

from santapp_ruler.attention.traffic import DecodeTrafficTracker


def test_santapp_gqa_union_and_naive_accounting():
    tracker = DecodeTrafficTracker(
        query_heads_per_kv=4,
        samples_per_head=2,
        access_pattern="sampled_kv",
    )
    tracker.record_group(
        n_total=10,
        sampled_indices_by_head=[
            torch.tensor([0, 0]),
            torch.tensor([0, 1]),
            torch.tensor([2, 2]),
            torch.tensor([2, 3]),
        ],
        exact_tokens=1,
        centroid_key_vectors=3,
    )

    metrics = tracker.as_dict()

    # GQA union={0,1,2,3}; K+V for four sampled rows and one exact row.
    assert metrics["decode_dense_gqa_kv_vectors"] == 20
    assert metrics["decode_gqa_kv_vectors_read"] == 10
    assert metrics["decode_gqa_centroid_key_vectors_read"] == 3
    assert metrics["decode_gqa_kv_access_pct"] == 50.0
    assert metrics["decode_gqa_centroid_access_pct"] == 15.0
    assert metrics["decode_gqa_total_access_pct"] == 65.0

    # Naive convention keeps all 2 draws/head and repeats the exact suffix and
    # centroid set for four query heads.
    assert metrics["decode_dense_naive_kv_vectors"] == 80
    assert metrics["decode_naive_kv_vectors_read"] == 24
    assert metrics["decode_naive_centroid_key_vectors_read"] == 12
    assert metrics["decode_naive_kv_access_pct"] == 30.0
    assert metrics["decode_naive_centroid_access_pct"] == 15.0
    assert metrics["decode_naive_total_access_pct"] == 45.0
    assert metrics["mean_unique_sampled_tokens_per_gqa_group_call"] == 4.0


def test_santa_counts_full_k_scan_and_union_of_sampled_values():
    tracker = DecodeTrafficTracker(
        query_heads_per_kv=2,
        samples_per_head=2,
        access_pattern="full_k_sampled_v",
    )
    tracker.record_group(
        n_total=10,
        sampled_indices_by_head=[torch.tensor([0, 1]), torch.tensor([1, 2])],
        exact_tokens=0,
        centroid_key_vectors=0,
    )

    metrics = tracker.as_dict()

    # Full K once (10) plus three distinct sampled V rows.
    assert metrics["decode_gqa_kv_vectors_read"] == 13
    assert metrics["decode_gqa_kv_access_pct"] == 65.0
    # Naive: two independent heads, each with 10 K rows + two V draws.
    assert metrics["decode_naive_kv_vectors_read"] == 24
    assert metrics["decode_naive_kv_access_pct"] == 60.0
    assert metrics["decode_gqa_total_access_pct"] == 65.0


def test_tracker_rejects_wrong_number_of_query_heads():
    tracker = DecodeTrafficTracker(2, 1, "sampled_kv")
    with pytest.raises(ValueError, match="one sampled-index tensor"):
        tracker.record_group(
            n_total=4,
            sampled_indices_by_head=[torch.tensor([0])],
            exact_tokens=0,
            centroid_key_vectors=1,
        )
