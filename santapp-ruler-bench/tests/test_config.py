from pathlib import Path

import pytest

from santapp_ruler.config import load_config


def test_default_config_has_single_non_sweep_run():
    config = load_config(None)
    assert config.benchmark.context_length == 8192
    assert config.model.revision == "aa8e72537993ba99e69dfaafa59ed015b17504d1"
    assert config.benchmark.prompts_per_task == 5
    assert config.generation.backends == ["sdpa", "santapp"]
    assert config.santapp.samples_per_head == 128
    assert config.santapp.group_size == 16


def test_dotted_overrides(tmp_path: Path):
    path = tmp_path / "config.yaml"
    path.write_text("benchmark:\n  tasks: [vt]\n", encoding="utf-8")
    config = load_config(
        path,
        overrides=[
            "benchmark.prompts_per_task=3",
            "santapp.samples_per_head=256",
            "generation.backends=[santapp]",
        ],
    )
    assert config.benchmark.tasks == ["vt"]
    assert config.benchmark.prompts_per_task == 3
    assert config.santapp.samples_per_head == 256
    assert config.generation.backends == ["santapp"]


def test_unknown_task_fails():
    with pytest.raises(ValueError, match="Unknown RULER task"):
        load_config(None, overrides=["benchmark.tasks=[not_a_task]"])
