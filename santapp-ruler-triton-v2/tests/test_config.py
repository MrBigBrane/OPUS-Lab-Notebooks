from pathlib import Path

import pytest

from santapp_ruler.config import (
    DEFAULT_CONFIG,
    SUPPORTED_BACKENDS,
    apply_dotted_overrides,
    load_config,
)
from santapp_ruler.ruler.provenance import default_dataset_repository
from santapp_ruler.ruler.tasks import DEFAULT_TASKS


PINNED_QWEN_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"


def test_only_four_supported_backends() -> None:
    assert SUPPORTED_BACKENDS == ("sdpa", "santa", "santapp", "hierarchical")


def test_default_model_and_sparse_geometry() -> None:
    cfg = load_config()
    assert cfg.model.name == "Qwen/Qwen2.5-7B-Instruct"
    assert cfg.model.revision == PINNED_QWEN_REVISION
    assert cfg.model.prompt_format == "chat_template"
    assert cfg.santapp.parent_size == 16
    assert cfg.santapp.representatives_per_parent == 4
    assert cfg.santapp.samples_per_head == 128
    assert cfg.santapp.kmeans.batch_size == 4096
    assert cfg.santapp.kmeans.n_init == 1
    assert cfg.santapp.kmeans.max_iter == 100
    assert cfg.santapp.kmeans.random_state == 0
    assert cfg.hierarchical.parent_size == 16
    assert cfg.hierarchical.representatives_per_parent == 4
    assert cfg.hierarchical.samples_per_head == 128


def test_all_thirteen_ruler_tasks_are_default() -> None:
    assert len(DEFAULT_TASKS) == 13
    assert DEFAULT_CONFIG.benchmark.tasks == list(DEFAULT_TASKS)


def test_unknown_backend_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown generation backend"):
        load_config(overrides=["generation.backends=[sdpa,legacy]"])


def test_duplicate_backend_is_rejected() -> None:
    with pytest.raises(ValueError, match="duplicates"):
        load_config(overrides=["generation.backends=[sdpa,sdpa]"])


def test_unknown_prompt_format_is_rejected() -> None:
    with pytest.raises(ValueError, match="prompt_format"):
        load_config(overrides=["model.prompt_format=magic"])


def test_parent_representative_geometry_is_validated() -> None:
    with pytest.raises(ValueError, match="divisible"):
        load_config(
            overrides=[
                "santapp.parent_size=16",
                "santapp.representatives_per_parent=3",
            ]
        )


def test_nominal_budget_geometry_is_validated() -> None:
    with pytest.raises(ValueError, match="samples_per_head"):
        load_config(overrides=["santapp.samples_per_head=130"])


def test_contiguous_geometry_is_validated() -> None:
    with pytest.raises(ValueError, match="divisible"):
        load_config(
            overrides=[
                "hierarchical.parent_size=16",
                "hierarchical.representatives_per_parent=3",
            ]
        )
    with pytest.raises(ValueError, match="samples_per_head"):
        load_config(overrides=["hierarchical.samples_per_head=130"])


def test_kmeans_configuration_is_validated() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        load_config(overrides=["santapp.kmeans.batch_size=0"])
    with pytest.raises(ValueError, match="random_state"):
        load_config(overrides=["santapp.kmeans.random_state=-1"])


def test_auto_dataset_mirrors_cover_target_contexts() -> None:
    assert "8192" in default_dataset_repository(8192)
    assert "32768" in default_dataset_repository(32768)


def test_auto_dataset_rejects_unconfigured_context() -> None:
    with pytest.raises(ValueError, match="Set benchmark.data.repository explicitly"):
        default_dataset_repository(16384)


def test_dotted_override_handles_lists_and_nested_kmeans_scalars() -> None:
    data = apply_dotted_overrides(
        DEFAULT_CONFIG.to_dict(),
        [
            "benchmark.context_length=32768",
            "generation.backends=[santapp,hierarchical]",
            "santapp.kmeans.max_iter=20",
            "hierarchical.parent_size=32",
            "output.resume=false",
        ],
    )
    assert data["benchmark"]["context_length"] == 32768
    assert data["generation"]["backends"] == ["santapp", "hierarchical"]
    assert data["santapp"]["kmeans"]["max_iter"] == 20
    assert data["hierarchical"]["parent_size"] == 32
    assert data["output"]["resume"] is False


def test_supplied_yaml_files_load() -> None:
    root = Path(__file__).resolve().parents[1]
    default = load_config(root / "configs" / "default.yaml")
    smoke = load_config(root / "configs" / "smoke.yaml")
    assert default.generation.backends == list(SUPPORTED_BACKENDS)
    assert smoke.generation.backends == ["sdpa"]
    for config in (default, smoke):
        assert config.model.revision == PINNED_QWEN_REVISION
        assert config.model.prompt_format == "chat_template"
        assert config.santapp.parent_size == 16
        assert config.santapp.kmeans.random_state == 0
        assert config.hierarchical.parent_size == 16
        assert config.hierarchical.representatives_per_parent == 4
