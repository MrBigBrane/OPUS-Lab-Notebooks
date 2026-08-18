from pathlib import Path

import pytest

from santapp_ruler.k8s_matrix import (
    MatrixError,
    MatrixSetting,
    apply_work_item,
    load_matrix,
)

MATRIX = Path("k8s/matrices/default-2tasks-6methods-100p-8k.yaml")


def test_default_matrix_expands_to_12_unique_shards():
    matrix = load_matrix(MATRIX)
    items = matrix.work_items()
    assert matrix.experiment == "santapp-ruler-default-2tasks-6methods-100p-8k"
    assert matrix.tasks == ("niah_multiquery", "niah_multivalue")
    assert matrix.prompts_per_task == 100
    assert matrix.completion_count == 12
    assert len(items) == 12
    assert len({item.shard_name for item in items}) == 12
    assert [item.index for item in items] == list(range(12))
    assert [item.task for item in items[:2]] == [
        "niah_multiquery",
        "niah_multivalue",
    ]
    assert all(item.setting.name == "sdpa" for item in items[:2])
    assert [setting.backend for setting in matrix.settings] == [
        "sdpa",
        "santa",
        "santapp",
        "santapp_gumbel_topk",
        "team_sampler",
        "team_gumbel_topk",
    ]


def test_default_matrix_applies_each_method_and_default_parameters(tmp_path: Path):
    matrix = load_matrix(MATRIX)
    base = matrix.load_base_run_config()
    by_name = {
        item.setting.name: item
        for item in matrix.work_items()
        if item.task == "niah_multiquery"
    }

    sdpa = apply_work_item(base, matrix, by_name["sdpa"], local_data_root=tmp_path)
    assert sdpa.generation.backends == ["sdpa"]

    santa = apply_work_item(
        base, matrix, by_name["santa-s128"], local_data_root=tmp_path
    )
    assert santa.generation.backends == ["santa"]
    assert santa.santa.samples_per_head == 128

    guided = apply_work_item(
        base, matrix, by_name["santapp-b16-s128"], local_data_root=tmp_path
    )
    assert guided.generation.backends == ["santapp"]
    assert guided.santapp.mode == "guided"
    assert guided.santapp.group_size == 16
    assert guided.santapp.samples_per_head == 128

    parent = apply_work_item(
        base,
        matrix,
        by_name["santapp-gumbel-b16-s128"],
        local_data_root=tmp_path,
    )
    parent_cfg = parent.santapp_config_for_backend("santapp_gumbel_topk")
    assert parent_cfg.mode == "gumbel_cluster"
    assert parent_cfg.group_size == 16
    assert parent_cfg.samples_per_head == 128
    assert by_name["santapp-gumbel-b16-s128"].setting.selected_clusters_per_head == 8

    team = apply_work_item(
        base,
        matrix,
        by_name["team-sampler-p16-r4-s128"],
        local_data_root=tmp_path,
    )
    team_cfg = team.santapp_config_for_backend("team_sampler")
    assert team_cfg.mode == "team_sampler"
    assert team_cfg.parent_size == 16
    assert team_cfg.representatives_per_parent == 4

    whole_team = apply_work_item(
        base,
        matrix,
        by_name["team-gumbel-p16-r4-s128"],
        local_data_root=tmp_path,
    )
    whole_team_cfg = whole_team.santapp_config_for_backend("team_gumbel_topk")
    assert whole_team_cfg.mode == "team_gumbel_topk"
    assert whole_team_cfg.parent_size == 16
    assert whole_team_cfg.representatives_per_parent == 4
    assert by_name["team-gumbel-p16-r4-s128"].setting.selected_teams_per_head == 32

    for config in (sdpa, santa, guided, parent, team, whole_team):
        assert config.output.resume is True
        assert config.output.save_full_prompts is False
        assert config.output.save_prediction_inputs is False
        assert config.benchmark.tasks == ["niah_multiquery"]
        assert config.benchmark.prompts_per_task == 100
        assert config.benchmark.data.source == "local"


def test_matrix_supports_a_custom_named_santapp_variant(tmp_path: Path):
    matrix_path = tmp_path / "matrix.yaml"
    config_path = tmp_path / "base.yaml"
    config_path.write_text(
        "benchmark:\n  tasks: [vt]\ngeneration:\n  backends: [sdpa]\n",
        encoding="utf-8",
    )
    matrix_path.write_text(
        f"""
schema_version: 1
experiment: custom-variant-test
base_config: {config_path}
prompts_per_task: 1
tasks: [vt]
settings:
  - name: custom-guided
    backend: custom_guided
    mode: guided
    samples_per_head: 64
    group_size: 16
""",
        encoding="utf-8",
    )
    matrix = load_matrix(matrix_path)
    config = apply_work_item(matrix.load_base_run_config(), matrix, matrix.work_item(0))
    assert config.generation.backends == ["custom_guided"]
    variant = config.santapp_config_for_backend("custom_guided")
    assert variant.mode == "guided"
    assert variant.samples_per_head == 64
    assert variant.group_size == 16


def test_matrix_rejects_invalid_team_gumbel_nominal_budget():
    with pytest.raises(MatrixError, match="parent_size must be divisible"):
        MatrixSetting(
            name="bad-parent",
            backend="team_gumbel_topk",
            mode="team_gumbel_topk",
            samples_per_head=128,
            parent_size=15,
            representatives_per_parent=4,
        ).validate()

    with pytest.raises(MatrixError, match="nominal team size"):
        MatrixSetting(
            name="bad-budget",
            backend="team_gumbel_topk",
            mode="team_gumbel_topk",
            samples_per_head=130,
            parent_size=16,
            representatives_per_parent=4,
        ).validate()
