import json
from pathlib import Path

from santapp_ruler.reporting import build_reports


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
            "decode_attention_head_calls": 4,
            "decode_dense_kv_vectors": 800,
            "decode_kv_vectors_read": 80 if backend == "santapp" else 800,
            "decode_metadata_key_vectors_read": 40 if backend == "santapp" else 0,
            "mean_ess_over_samples": 0.5 if backend == "santapp" else None,
            "peak_allocated_gib": 8.0,
        },
    }
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")


def test_report_builds_official_and_harness_outputs(tmp_path: Path):
    for backend in ("sdpa", "santapp"):
        _write_prediction(tmp_path, backend, "vt", "answer")
    summary = build_reports(tmp_path, backends=["sdpa", "santapp"], tasks=["vt"])
    assert summary["backends"]["sdpa"]["aggregate"]["score"] == 100.0
    assert summary["backends"]["santapp"]["aggregate"]["decode_kv_access_pct"] == 10.0
    assert (tmp_path / "summary.md").is_file()
    assert (tmp_path / "predictions" / "sdpa" / "summary.csv").is_file()
