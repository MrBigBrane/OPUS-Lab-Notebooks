import json
from pathlib import Path

import pytest

from santapp_ruler.k8s_matrix import MatrixSetting
from scripts.k8s_run_index import PermanentWorkerError, _validate_records


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_index_worker_validates_team_gumbel_requested_team_count(tmp_path: Path):
    prediction = tmp_path / "predictions" / "team_gumbel_topk" / "qa_1.jsonl"
    selection = tmp_path / "selected_prompts.jsonl"
    metrics = {
        "prompt_exact_tail_tokens": 0,
        "nominal_sample_budget_per_head": 128,
        "teams_per_head": 32,
        "parent_size": 16,
        "representatives_per_parent": 4,
    }
    row = {"uid": "qa_1:7", "task": "qa_1", "metrics": metrics}
    _write_jsonl(prediction, [row])
    _write_jsonl(selection, [{"uid": "qa_1:7", "task": "qa_1"}])
    setting = MatrixSetting(
        name="team-gumbel-topk-p16-r4-s128",
        backend="team_gumbel_topk",
        mode="team_gumbel_topk",
        samples_per_head=128,
        parent_size=16,
        representatives_per_parent=4,
    )

    records = _validate_records(
        prediction_path=prediction,
        selected_manifest=selection,
        expected_uids=["qa_1:7"],
        expected_count=1,
        setting=setting,
        task="qa_1",
    )
    assert records == [row]

    row["metrics"]["teams_per_head"] = 31
    _write_jsonl(prediction, [row])
    with pytest.raises(PermanentWorkerError, match="Expected 32 selected teams"):
        _validate_records(
            prediction_path=prediction,
            selected_manifest=selection,
            expected_uids=["qa_1:7"],
            expected_count=1,
            setting=setting,
            task="qa_1",
        )


def test_index_worker_validates_recorded_hierarchy_identity(tmp_path: Path):
    prediction = tmp_path / "predictions" / "team_sampler" / "qa_1.jsonl"
    selection = tmp_path / "selected_prompts.jsonl"
    row = {
        "uid": "qa_1:8",
        "task": "qa_1",
        "metrics": {
            "prompt_exact_tail_tokens": 0,
            "nominal_sample_budget_per_head": 128,
            "parent_size": 8,
            "representatives_per_parent": 4,
        },
    }
    _write_jsonl(prediction, [row])
    _write_jsonl(selection, [{"uid": "qa_1:8", "task": "qa_1"}])
    setting = MatrixSetting(
        name="team-sampler-p16-r4-s128",
        backend="team_sampler",
        mode="team_sampler",
        samples_per_head=128,
        parent_size=16,
        representatives_per_parent=4,
    )

    with pytest.raises(PermanentWorkerError, match="Expected parent_size=16"):
        _validate_records(
            prediction_path=prediction,
            selected_manifest=selection,
            expected_uids=["qa_1:8"],
            expected_count=1,
            setting=setting,
            task="qa_1",
        )
