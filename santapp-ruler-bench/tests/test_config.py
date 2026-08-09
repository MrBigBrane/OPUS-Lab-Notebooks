from pathlib import Path

import pytest

from santapp_ruler.config import load_config
from santapp_ruler.ruler.tasks import TASKS


def test_default_config_has_single_non_sweep_run():
    config = load_config(None)
    assert config.benchmark.context_length == 8192
    assert config.model.revision == "aa8e72537993ba99e69dfaafa59ed015b17504d1"
    assert config.benchmark.prompts_per_task == 5
    assert config.generation.backends == ["sdpa", "santapp"]
    assert config.generation.max_new_tokens is None
    assert config.generation.random_seed == 0
    assert config.santa.samples_per_head == 128
    assert config.santapp.samples_per_head == 128
    assert config.santapp.group_size == 16
    assert config.santapp.probe_queries == 64
    assert set(config.santapp.__dataclass_fields__) == {
        "group_size",
        "samples_per_head",
        "probe_queries",
        "kmeans",
    }


def test_dotted_overrides(tmp_path: Path):
    path = tmp_path / "config.yaml"
    path.write_text("benchmark:\n  tasks: [vt]\n", encoding="utf-8")
    config = load_config(
        path,
        overrides=[
            "benchmark.prompts_per_task=3",
            "santapp.samples_per_head=256",
            "santa.samples_per_head=64",
            "generation.backends=[santa, santapp]",
            "generation.max_new_tokens=17",
        ],
    )
    assert config.benchmark.tasks == ["vt"]
    assert config.benchmark.prompts_per_task == 3
    assert config.santapp.samples_per_head == 256
    assert config.santa.samples_per_head == 64
    assert config.generation.backends == ["santa", "santapp"]
    assert config.generation.max_new_tokens == 17


def test_colleague_simple_config():
    config = load_config("configs/simple_8k.yaml")
    assert config.benchmark.tasks == ["niah_multivalue", "qa_1"]
    assert config.benchmark.prompts_per_task == 20
    assert config.generation.backends == ["sdpa", "santa", "santapp"]
    assert config.generation.max_new_tokens is None
    assert config.santa.samples_per_head == 128
    assert config.santapp.samples_per_head == 128
    assert config.santapp.group_size == 16
    assert config.santapp.probe_queries == 64


def test_all_task_smoke_config_has_39_cases():
    config = load_config("configs/smoke_all_tasks_8k.yaml")
    assert set(config.benchmark.tasks) == set(TASKS)
    assert len(config.benchmark.tasks) == 13
    assert config.benchmark.prompts_per_task == 1
    assert config.generation.backends == ["sdpa", "santa", "santapp"]
    assert (
        len(config.benchmark.tasks)
        * config.benchmark.prompts_per_task
        * len(config.generation.backends)
        == 39
    )


def test_full_config_requests_500_prompts_for_all_tasks():
    config = load_config("configs/all_tasks_8k.yaml")
    assert set(config.benchmark.tasks) == set(TASKS)
    assert config.benchmark.prompts_per_task == 500


def test_unknown_task_fails():
    with pytest.raises(ValueError, match="Unknown RULER task"):
        load_config(None, overrides=["benchmark.tasks=[not_a_task]"])


def test_nonpositive_generation_override_fails():
    with pytest.raises(ValueError, match="generation.max_new_tokens"):
        load_config(None, overrides=["generation.max_new_tokens=0"])
