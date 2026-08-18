import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

from santapp_ruler.io_utils import atomic_write_json
from santapp_ruler.k8s_matrix import cache_fingerprint, load_matrix
from santapp_ruler.k8s_runtime import KubernetesPaths


def _metrics(mode: str, budget: int | None, *, clusters: int | None = None):
    dense_gqa = 1000
    dense_naive = 4000
    sparse = mode != "dense_sdpa"
    gqa_kv = 100 if sparse else dense_gqa
    naive_kv = 400 if sparse else dense_naive
    gqa_centroid = 25 if mode in {"guided", "gumbel_cluster"} else 0
    naive_centroid = 100 if mode in {"guided", "gumbel_cluster"} else 0
    return {
        "mode": mode,
        "sampling_scheme": "test",
        "importance_correction": "test",
        "sampling_unit": "cluster" if mode == "gumbel_cluster" else "token",
        "samples_per_head": budget,
        "nominal_sample_budget_per_head": budget,
        "clusters_per_head": clusters,
        "group_size": 16 if mode in {"guided", "gumbel_cluster"} else None,
        "probe_queries": 64 if mode in {"guided", "gumbel_cluster"} else None,
        "probe_policy": (
            "last_prompt_tokens"
            if mode in {"guided", "gumbel_cluster"}
            else "not_applicable"
        ),
        "exact_token_policy": (
            "dense_all_tokens" if mode == "dense_sdpa" else "generated_suffix_only"
        ),
        "prompt_exact_tail_tokens": 0,
        "initial_growing_exact_tokens": 0,
        "prompt_tokens": 100,
        "generated_tokens": 2,
        "prefill_seconds": 1.0,
        "clustering_seconds": 0.5 if mode in {"guided", "gumbel_cluster"} else 0.0,
        "decode_seconds": 2.0,
        "total_seconds": 3.5 if sparse else 3.0,
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
        "mean_exact_tokens_per_gqa_group_call": 0.0,
        "mean_total_tokens_per_head_call": 100.0,
        "mean_selected_clusters_per_head_call": float(clusters or 0),
        "selected_cluster_inclusion_probability_count": 0,
        "mean_selected_cluster_inclusion_probability": None,
        "min_selected_cluster_inclusion_probability": None,
        "max_selected_cluster_inclusion_probability": None,
        "peak_allocated_gib": 8.0,
        "peak_reserved_gib": 9.0,
    }


def _write_jsonl(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_compact_k8s_export_excludes_prompts_and_builds_paired_reports(tmp_path: Path):
    config_path = tmp_path / "base.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "benchmark": {"tasks": ["vt", "qa_1"], "prompts_per_task": 2},
                "generation": {"backends": ["sdpa"]},
                "santapp_variants": {
                    "santapp_gumbel_topk": {
                        "mode": "gumbel_cluster",
                        "samples_per_head": 128,
                        "group_size": 16,
                    }
                },
                "grading": {"bootstrap_resamples": 50, "bootstrap_seed": 7},
                "output": {
                    "save_full_prompts": False,
                    "save_prediction_inputs": False,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    matrix_path = tmp_path / "matrix.yaml"
    matrix_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "experiment": "shareable-test-export",
                "base_config": str(config_path),
                "prompts_per_task": 2,
                "tasks": ["vt", "qa_1"],
                "settings": [
                    {"name": "sdpa", "backend": "sdpa"},
                    {
                        "name": "santapp-s128",
                        "backend": "santapp",
                        "mode": "guided",
                        "samples_per_head": 128,
                        "group_size": 16,
                    },
                    {
                        "name": "santapp-gumbel-s128",
                        "backend": "santapp_gumbel_topk",
                        "mode": "gumbel_cluster",
                        "samples_per_head": 128,
                        "group_size": 16,
                    },
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    matrix = load_matrix(matrix_path)
    base = matrix.load_base_run_config()
    results_root = tmp_path / "results"
    paths = KubernetesPaths.create(results_root, matrix)
    selected_uids = {
        task: [f"{task}:source:{index}" for index in range(2)]
        for task in matrix.tasks
    }
    for task in matrix.tasks:
        data_path = paths.local_data_root / task / "validation.jsonl"
        data_path.parent.mkdir(parents=True, exist_ok=True)
        data_path.write_text("{}\n", encoding="utf-8")
    snapshot = paths.cache_root / "fake-model-snapshot"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}\n", encoding="utf-8")
    (snapshot / "model.safetensors").write_bytes(b"test-weights")
    (snapshot / "tokenizer_config.json").write_text("{}\n", encoding="utf-8")
    atomic_write_json(
        paths.cache_marker,
        {
            "status": "ready",
            "cache_fingerprint": cache_fingerprint(matrix, base),
            "model_snapshot_path": str(snapshot),
            "selected_uids": selected_uids,
        },
    )

    for item in matrix.work_items():
        if item.setting.backend == "sdpa":
            mode, budget, clusters = "dense_sdpa", None, None
        elif item.setting.mode == "guided":
            mode, budget, clusters = "guided", 128, None
        else:
            mode, budget, clusters = "gumbel_cluster", 128, 8
        rows = []
        for index, uid in enumerate(selected_uids[item.task]):
            metrics = _metrics(mode, budget, clusters=clusters)
            metrics["example_ruler_score"] = 100.0
            rows.append(
                {
                    "index": index,
                    "uid": uid,
                    "task": item.task,
                    "outputs": ["answer"],
                    "pred": "answer",
                    "metrics": metrics,
                }
            )
        shard = paths.shards_root / item.shard_name
        prediction = (
            shard
            / "run"
            / "predictions"
            / item.setting.backend
            / f"{item.task}.jsonl"
        )
        _write_jsonl(prediction, rows)
        atomic_write_json(
            shard / "success.json",
            {
                "matrix_hash": matrix.matrix_hash,
                "work_index": item.index,
                "prediction_sha256": _sha256(prediction),
                "execution_fingerprint": "test-code-fingerprint",
                "runtime": {
                    "pod_name": f"pod-{item.index}",
                    "node_name": "node-a",
                    "gpu_name": "test-gpu",
                    "gpu_total_gib": 16.0,
                    "cuda_capability": [8, 6],
                    "torch": "2.11.0",
                    "torch_cuda": "12.6",
                    "container_image": "example:test",
                },
            },
        )

    output_root = tmp_path / "export"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path.cwd() / "src")
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/k8s_export_results.py",
            "--matrix",
            str(matrix_path),
            "--results-root",
            str(results_root),
            "--output-dir",
            str(output_root),
            "--no-archive",
        ],
        cwd=Path.cwd(),
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    export = output_root / matrix.experiment
    compact = [
        json.loads(line)
        for line in (export / "compact-results.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(compact) == 12
    assert all("input" not in row for row in compact)
    assert all('"input":' not in json.dumps(row) for row in compact)
    status = json.loads((export / "export-status.json").read_text(encoding="utf-8"))
    assert status["status"] == "complete"
    assert status["hardware"]["single_gpu_product"] is True
    assert status["hardware"]["gpu_products"] == ["test-gpu"]
    assert status["execution_fingerprint"] == "test-code-fingerprint"
    timing = json.loads(
        (export / "timing-validity.json").read_text(encoding="utf-8")
    )
    assert timing["runtime_comparisons_hardware_controlled"] is True
    assert (export / "hardware-provenance.csv").is_file()
    summary = json.loads((export / "summary.json").read_text(encoding="utf-8"))
    assert set(summary["backends"]) == {
        "sdpa",
        "santapp-s128",
        "santapp-gumbel-s128",
    }
    assert set(summary["comparisons"]["vs_sdpa"]) == {
        "santapp-s128",
        "santapp-gumbel-s128",
    }
