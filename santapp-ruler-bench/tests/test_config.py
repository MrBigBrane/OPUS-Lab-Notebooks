from pathlib import Path

import pytest

from santapp_ruler.config import load_config
from santapp_ruler.ruler.tasks import TASKS

STANDARD_BACKENDS = [
    "sdpa",
    "santa",
    "santapp",
    "santapp_gumbel_topk",
    "team_sampler",
    "team_gumbel_topk",
]


def test_builtin_defaults_include_all_standard_backends():
    config = load_config(None)
    assert config.benchmark.context_length == 8192
    assert config.generation.backends == STANDARD_BACKENDS
    assert config.generation.max_new_tokens is None
    assert config.generation.random_seed == 0
    assert config.santa.samples_per_head == 128
    assert config.santapp.samples_per_head == 128
    assert config.santapp.group_size == 16
    assert config.santapp.parent_size == 16
    assert config.santapp.representatives_per_parent == 4
    assert config.santapp.probe_queries == 64
    assert config.santapp_config_for_backend("santapp_gumbel_topk").mode == (
        "gumbel_cluster"
    )
    assert config.santapp_config_for_backend("team_sampler").mode == "team_sampler"
    assert config.santapp_config_for_backend("team_gumbel_topk").mode == (
        "team_gumbel_topk"
    )


def test_default_file_uses_six_backends_and_b16_santapp():
    config = load_config("configs/default_8k.yaml")
    assert config.benchmark.tasks == ["niah_multiquery", "niah_multivalue"]
    assert config.benchmark.prompts_per_task == 5
    assert config.generation.backends == STANDARD_BACKENDS
    assert config.santapp.group_size == 16
    parent = config.santapp_config_for_backend("santapp_gumbel_topk")
    assert parent.mode == "gumbel_cluster"
    assert parent.group_size == 16
    assert parent.samples_per_head == 128
    teams = config.santapp_config_for_backend("team_sampler")
    assert teams.parent_size == 16
    assert teams.representatives_per_parent == 4


def test_smoke_config_runs_one_prompt_for_all_standard_backends():
    config = load_config("configs/smoke_all_backends_8k.yaml")
    assert config.benchmark.tasks == ["niah_multiquery"]
    assert config.benchmark.prompts_per_task == 1
    assert config.generation.backends == STANDARD_BACKENDS
    assert config.generation.max_new_tokens == 4
    assert config.output.save_full_prompts is False
    assert config.output.save_prediction_inputs is False


def test_all_tasks_config_covers_supported_task_set():
    config = load_config("configs/all_tasks_8k.yaml")
    assert set(config.benchmark.tasks) == set(TASKS)
    assert config.generation.backends == STANDARD_BACKENDS


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
            "generation.random_seed=9",
        ],
    )
    assert config.benchmark.tasks == ["vt"]
    assert config.benchmark.prompts_per_task == 3
    assert config.santapp.samples_per_head == 256
    assert config.santa.samples_per_head == 64
    assert config.generation.backends == ["santa", "santapp"]
    assert config.generation.max_new_tokens == 17
    assert config.generation.random_seed == 9


def test_builtin_variants_inherit_base_santapp_overrides(tmp_path: Path):
    path = tmp_path / "inherit.yaml"
    path.write_text(
        """
generation:
  backends: [santapp_gumbel_topk, team_sampler, team_gumbel_topk]
santapp:
  group_size: 32
  parent_size: 32
  representatives_per_parent: 8
  samples_per_head: 256
  probe_queries: 99
santapp_variants:
  team_sampler:
    samples_per_head: 512
""",
        encoding="utf-8",
    )

    config = load_config(path)
    parent = config.santapp_config_for_backend("santapp_gumbel_topk")
    teams = config.santapp_config_for_backend("team_sampler")
    whole_teams = config.santapp_config_for_backend("team_gumbel_topk")

    for variant in (parent, teams, whole_teams):
        assert variant.group_size == 32
        assert variant.parent_size == 32
        assert variant.representatives_per_parent == 8
        assert variant.probe_queries == 99

    assert parent.samples_per_head == 256
    assert teams.samples_per_head == 512
    assert whole_teams.samples_per_head == 256
    assert parent.mode == "gumbel_cluster"
    assert teams.mode == "team_sampler"
    assert whole_teams.mode == "team_gumbel_topk"


def test_gumbel_budget_must_be_divisible_by_group_size():
    with pytest.raises(ValueError, match="divisible by group_size"):
        load_config(
            "configs/default_8k.yaml",
            overrides=["santapp_variants.santapp_gumbel_topk.samples_per_head=130"],
        )


def test_team_configuration_and_nominal_gumbel_budget_validation(tmp_path: Path):
    path = tmp_path / "teams.yaml"
    path.write_text(
        """
generation:
  backends: [team_sampler, team_gumbel_topk]
santapp_variants:
  team_sampler:
    mode: team_sampler
    parent_size: 16
    representatives_per_parent: 4
    samples_per_head: 130
  team_gumbel_topk:
    mode: team_gumbel_topk
    parent_size: 16
    representatives_per_parent: 4
    samples_per_head: 128
""",
        encoding="utf-8",
    )
    config = load_config(path)
    assert config.santapp_config_for_backend("team_sampler").samples_per_head == 130
    whole_team = config.santapp_config_for_backend("team_gumbel_topk")
    assert whole_team.samples_per_head // (whole_team.parent_size // 4) == 32

    with pytest.raises(ValueError, match="nominal_team_size"):
        load_config(
            path,
            overrides=["santapp_variants.team_gumbel_topk.samples_per_head=130"],
        )
    with pytest.raises(ValueError, match="divisible by representatives_per_parent"):
        load_config(
            path,
            overrides=["santapp_variants.team_gumbel_topk.parent_size=15"],
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("parent_size", 0, "parent_size must be positive"),
        (
            "representatives_per_parent",
            0,
            "representatives_per_parent must be positive",
        ),
        (
            "representatives_per_parent",
            17,
            "representatives_per_parent must be <= parent_size",
        ),
    ],
)
def test_invalid_hierarchy_sizes_fail(field: str, value: int, message: str):
    with pytest.raises(ValueError, match=message):
        load_config(None, overrides=[f"santapp.{field}={value}"])


def test_unknown_task_fails():
    with pytest.raises(ValueError, match="Unknown RULER task"):
        load_config(None, overrides=["benchmark.tasks=[not_a_task]"])


def test_nonpositive_generation_override_fails():
    with pytest.raises(ValueError, match="max_new_tokens"):
        load_config(None, overrides=["generation.max_new_tokens=0"])


def test_prediction_input_storage_is_configurable_and_defaults_on():
    assert load_config(None).output.save_prediction_inputs is True
    compact = load_config(None, overrides=["output.save_prediction_inputs=false"])
    assert compact.output.save_prediction_inputs is False
