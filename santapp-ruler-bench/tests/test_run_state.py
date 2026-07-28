from pathlib import Path

import pytest

from santapp_ruler.config import load_config
from santapp_ruler.data import RulerExample
from santapp_ruler.run_state import (
    prepare_run_directory,
    validate_or_write_selection_manifest,
)


def test_run_directory_refuses_changed_result_config(tmp_path: Path):
    first = load_config(None)
    prepare_run_directory(first, tmp_path)
    changed = load_config(None, overrides=["santapp.samples_per_head=256"])
    with pytest.raises(ValueError, match="different resolved benchmark"):
        prepare_run_directory(changed, tmp_path)


def test_no_resume_refuses_existing_predictions(tmp_path: Path):
    first = load_config(None)
    prepare_run_directory(first, tmp_path)
    prediction = tmp_path / "predictions" / "sdpa" / "vt.jsonl"
    prediction.parent.mkdir(parents=True)
    prediction.write_text("{}\n", encoding="utf-8")
    no_resume = load_config(None, overrides=["output.resume=false"])
    with pytest.raises(FileExistsError, match="resume=false"):
        prepare_run_directory(no_resume, tmp_path)


def test_selection_manifest_is_validated_on_resume(tmp_path: Path):
    selected = {
        "vt": [
            RulerExample(
                task="vt",
                index=7,
                input="prompt",
                outputs=["answer"],
                source_position=2,
            )
        ]
    }
    path = tmp_path / "selected_prompts.jsonl"
    validate_or_write_selection_manifest(
        selected, ["vt"], path, include_prompts=True
    )
    validate_or_write_selection_manifest(
        selected, ["vt"], path, include_prompts=True
    )
    changed = {
        "vt": [
            RulerExample(
                task="vt",
                index=7,
                input="different prompt",
                outputs=["answer"],
                source_position=2,
            )
        ]
    }
    with pytest.raises(ValueError, match="manifest differs"):
        validate_or_write_selection_manifest(
            changed, ["vt"], path, include_prompts=True
        )
