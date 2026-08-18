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
    assert metrics["decode_dense_gqa_kv_vectors"] == 20
    assert metrics["decode_gqa_kv_vectors_read"] == 10
    assert metrics["decode_gqa_routing_key_vectors_read"] == 3
    assert metrics["decode_gqa_centroid_key_vectors_read"] == 3
    assert metrics["decode_gqa_kv_access_pct"] == 50.0
    assert metrics["decode_gqa_routing_access_pct"] == 15.0
    assert metrics["decode_gqa_centroid_access_pct"] == 15.0
    assert metrics["decode_gqa_total_access_pct"] == 65.0

    assert metrics["decode_dense_naive_kv_vectors"] == 80
    assert metrics["decode_naive_kv_vectors_read"] == 24
    assert metrics["decode_naive_routing_key_vectors_read"] == 12
    assert metrics["decode_naive_centroid_key_vectors_read"] == 12
    assert metrics["decode_naive_kv_access_pct"] == 30.0
    assert metrics["decode_naive_routing_access_pct"] == 15.0
    assert metrics["decode_naive_centroid_access_pct"] == 15.0
    assert metrics["decode_naive_total_access_pct"] == 45.0
    assert metrics["mean_unique_sampled_tokens_per_gqa_group_call"] == 4.0


def test_gumbel_whole_cluster_rows_may_vary_by_query_head():
    tracker = DecodeTrafficTracker(
        query_heads_per_kv=2,
        samples_per_head=None,
        access_pattern="sampled_kv",
    )
    tracker.record_group(
        n_total=10,
        sampled_indices_by_head=[
            torch.tensor([0, 1, 2]),
            torch.tensor([2, 3, 4, 5]),
        ],
        exact_tokens=1,
        centroid_key_vectors=3,
    )
    metrics = tracker.as_dict()
    # Union {0,1,2,3,4,5}, plus one disjoint exact row, with both K and V.
    assert metrics["decode_gqa_kv_vectors_read"] == 14
    assert metrics["decode_gqa_total_access_pct"] == 85.0
    # Seven realized rows plus one exact row per query head, K and V.
    assert metrics["decode_naive_kv_vectors_read"] == 18
    assert metrics["decode_naive_total_access_pct"] == 60.0
    assert metrics["mean_sampled_token_rows_per_head_call"] == 3.5


def test_team_sampler_counts_actual_leader_routing_and_fixed_draws():
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
        routing_key_vectors=5,
        centroid_key_vectors=0,
    )

    metrics = tracker.as_dict()
    # The GQA path unions the four actual sampled rows, then reads one disjoint
    # exact row, while routing scans five leaders only once for the KV head.
    assert metrics["decode_gqa_kv_vectors_read"] == 10
    assert metrics["decode_gqa_routing_key_vectors_read"] == 5
    assert metrics["decode_gqa_centroid_key_vectors_read"] == 0
    assert metrics["decode_gqa_total_vectors_read"] == 15
    assert metrics["decode_gqa_routing_access_pct"] == 25.0
    assert metrics["decode_gqa_total_access_pct"] == 75.0

    # Naive accounting keeps all eight draws and repeats both the exact row and
    # the five-leader routing scan for each of the four query heads.
    assert metrics["decode_naive_kv_vectors_read"] == 24
    assert metrics["decode_naive_routing_key_vectors_read"] == 20
    assert metrics["decode_naive_centroid_key_vectors_read"] == 0
    assert metrics["decode_naive_total_vectors_read"] == 44
    assert metrics["decode_naive_routing_access_pct"] == 25.0
    assert metrics["decode_naive_total_access_pct"] == 55.0
    assert metrics["mean_sampled_token_rows_per_head_call"] == 2.0


def test_team_gumbel_counts_variable_rows_and_actual_leader_routing():
    tracker = DecodeTrafficTracker(
        query_heads_per_kv=2,
        samples_per_head=None,
        access_pattern="sampled_kv",
    )
    tracker.record_group(
        n_total=10,
        sampled_indices_by_head=[
            torch.tensor([0, 1, 2]),
            torch.tensor([2, 3, 4, 5]),
        ],
        exact_tokens=1,
        routing_key_vectors=6,
        centroid_key_vectors=0,
    )

    metrics = tracker.as_dict()
    # Six unique sampled rows plus one exact row require fourteen K/V vectors;
    # the six actual team leaders are a separate routing-stage read.
    assert metrics["decode_gqa_kv_vectors_read"] == 14
    assert metrics["decode_gqa_routing_key_vectors_read"] == 6
    assert metrics["decode_gqa_centroid_key_vectors_read"] == 0
    assert metrics["decode_gqa_total_vectors_read"] == 20
    assert metrics["decode_gqa_total_access_pct"] == 100.0

    # Naive accounting retains all seven realized rows, repeats the exact row,
    # and scans all six routing leaders for each query head.
    assert metrics["decode_naive_kv_vectors_read"] == 18
    assert metrics["decode_naive_routing_key_vectors_read"] == 12
    assert metrics["decode_naive_centroid_key_vectors_read"] == 0
    assert metrics["decode_naive_total_vectors_read"] == 30
    assert metrics["decode_naive_total_access_pct"] == 75.0
    assert metrics["mean_sampled_token_rows_per_head_call"] == 3.5


def test_centroid_metrics_are_a_subset_not_an_extra_total_component():
    tracker = DecodeTrafficTracker(
        query_heads_per_kv=1,
        samples_per_head=1,
        access_pattern="sampled_kv",
    )
    tracker.record_group(
        n_total=10,
        sampled_indices_by_head=[torch.tensor([0])],
        exact_tokens=0,
        routing_key_vectors=5,
        centroid_key_vectors=3,
    )

    metrics = tracker.as_dict()
    assert metrics["decode_gqa_kv_vectors_read"] == 2
    assert metrics["decode_gqa_routing_key_vectors_read"] == 5
    assert metrics["decode_gqa_centroid_key_vectors_read"] == 3
    assert metrics["decode_gqa_total_vectors_read"] == 7
    assert metrics["decode_gqa_total_access_pct"] == 35.0


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
    assert metrics["decode_gqa_kv_vectors_read"] == 13
    assert metrics["decode_gqa_kv_access_pct"] == 65.0
    assert metrics["decode_naive_kv_vectors_read"] == 24
    assert metrics["decode_naive_kv_access_pct"] == 60.0


def test_tracker_rejects_wrong_number_of_query_heads():
    tracker = DecodeTrafficTracker(
        query_heads_per_kv=2,
        samples_per_head=1,
        access_pattern="sampled_kv",
    )
    with pytest.raises(ValueError, match="one sampled-index tensor"):
        tracker.record_group(
            n_total=4,
            sampled_indices_by_head=[torch.tensor([0])],
            exact_tokens=0,
            centroid_key_vectors=1,
        )


def test_tracker_rejects_centroid_count_larger_than_routing_count():
    tracker = DecodeTrafficTracker(
        query_heads_per_kv=1,
        samples_per_head=1,
        access_pattern="sampled_kv",
    )
    with pytest.raises(ValueError, match="cannot exceed routing_key_vectors"):
        tracker.record_group(
            n_total=4,
            sampled_indices_by_head=[torch.tensor([0])],
            exact_tokens=0,
            routing_key_vectors=1,
            centroid_key_vectors=2,
        )
