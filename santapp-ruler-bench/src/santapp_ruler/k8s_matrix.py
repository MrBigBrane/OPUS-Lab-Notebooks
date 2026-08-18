"""Declarative matrix expansion for resumable Kubernetes benchmark shards."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from .config import RunConfig, SantaPlusConfig, load_config
from .ruler.tasks import validate_tasks

_DNS_LABEL = re.compile(r"^[a-z0-9](?:[a-z0-9-]*[a-z0-9])?$")
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class MatrixError(ValueError):
    """A matrix cannot be expanded into unambiguous benchmark shards."""


@dataclass(frozen=True, slots=True)
class MatrixSetting:
    """One algorithm setting applied independently to every matrix task."""

    name: str
    backend: str
    samples_per_head: int | None = None
    mode: str | None = None
    group_size: int | None = None
    probe_queries: int | None = None
    parent_size: int | None = None
    representatives_per_parent: int | None = None
    description: str | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "MatrixSetting":
        allowed = {
            "name",
            "backend",
            "samples_per_head",
            "mode",
            "group_size",
            "probe_queries",
            "parent_size",
            "representatives_per_parent",
            "description",
        }
        unknown = set(value) - allowed
        if unknown:
            raise MatrixError(f"Unknown setting field(s): {sorted(unknown)}")
        try:
            setting = cls(**dict(value))
        except TypeError as exc:
            raise MatrixError(f"Invalid matrix setting: {value!r}") from exc
        setting.validate()
        return setting

    def validate(self) -> None:
        if _DNS_LABEL.fullmatch(self.name) is None or len(self.name) > 63:
            raise MatrixError(
                f"Setting name must be a Kubernetes-safe DNS label: {self.name!r}"
            )
        if _SAFE_NAME.fullmatch(self.backend) is None:
            raise MatrixError(f"Invalid backend name: {self.backend!r}")
        for key in (
            "samples_per_head",
            "group_size",
            "probe_queries",
            "parent_size",
            "representatives_per_parent",
        ):
            value = getattr(self, key)
            if value is not None and value <= 0:
                raise MatrixError(f"{self.name}.{key} must be positive when set.")
        if self.backend == "sdpa":
            configured = {
                "samples_per_head": self.samples_per_head,
                "mode": self.mode,
                "group_size": self.group_size,
                "probe_queries": self.probe_queries,
                "parent_size": self.parent_size,
                "representatives_per_parent": self.representatives_per_parent,
            }
            if any(value is not None for value in configured.values()):
                raise MatrixError(
                    f"Dense SDPA setting {self.name!r} cannot set sparse parameters."
                )
        elif self.backend == "santa":
            if self.samples_per_head is None:
                raise MatrixError(
                    f"Standalone SANTA setting {self.name!r} needs samples_per_head."
                )
            if any(
                value is not None
                for value in (
                    self.mode,
                    self.group_size,
                    self.probe_queries,
                    self.parent_size,
                    self.representatives_per_parent,
                )
            ):
                raise MatrixError(
                    f"Standalone SANTA setting {self.name!r} accepts only "
                    "samples_per_head."
                )
        else:
            if self.samples_per_head is None:
                raise MatrixError(
                    f"SANTA++ setting {self.name!r} needs samples_per_head."
                )
            mode = self.mode or "guided"
            if mode not in {
                "guided",
                "gumbel_cluster",
                "oracle_token",
                "uniform",
                "topk",
                "team_sampler",
                "team_gumbel_topk",
            }:
                raise MatrixError(f"Unsupported SANTA++ mode {mode!r}.")
            parent_size = self.parent_size or 16
            representatives = self.representatives_per_parent or 4
            if representatives > parent_size:
                raise MatrixError(
                    f"{self.name}: representatives_per_parent cannot exceed "
                    "parent_size."
                )
            if mode == "gumbel_cluster":
                group_size = self.group_size or 16
                if self.samples_per_head % group_size:
                    raise MatrixError(
                        f"{self.name}: samples_per_head must be divisible by "
                        "group_size for Gumbel cluster sampling."
                    )
            if mode == "team_gumbel_topk":
                if parent_size % representatives:
                    raise MatrixError(
                        f"{self.name}: parent_size must be divisible by "
                        "representatives_per_parent for team Gumbel sampling."
                    )
                nominal_team_size = parent_size // representatives
                if self.samples_per_head % nominal_team_size:
                    raise MatrixError(
                        f"{self.name}: samples_per_head must be divisible by the "
                        "nominal team size for team Gumbel sampling."
                    )

    @property
    def selected_clusters_per_head(self) -> int | None:
        if (self.mode or "guided") != "gumbel_cluster":
            return None
        assert self.samples_per_head is not None
        return self.samples_per_head // (self.group_size or 16)

    @property
    def selected_teams_per_head(self) -> int | None:
        """Requested distinct teams for flattened whole-team Gumbel sampling."""

        if (self.mode or "guided") != "team_gumbel_topk":
            return None
        assert self.samples_per_head is not None
        parent_size = self.parent_size or 16
        representatives = self.representatives_per_parent or 4
        return self.samples_per_head // (parent_size // representatives)

    @property
    def requested_teams_per_head(self) -> int | None:
        """Compatibility-friendly explicit name for the requested team count."""

        return self.selected_teams_per_head

    @property
    def teams_per_head(self) -> int | None:
        """Match the prediction metric name used by the indexed worker."""

        return self.selected_teams_per_head


@dataclass(frozen=True, slots=True)
class WorkItem:
    index: int
    setting_index: int
    task_index: int
    setting: MatrixSetting
    task: str

    @property
    def shard_name(self) -> str:
        task_slug = self.task.replace("_", "-")
        return f"{self.index:03d}-{self.setting.name}-{task_slug}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "setting_index": self.setting_index,
            "task_index": self.task_index,
            "setting": asdict(self.setting),
            "task": self.task,
            "shard_name": self.shard_name,
        }


@dataclass(frozen=True, slots=True)
class ExperimentMatrix:
    schema_version: int
    experiment: str
    base_config: str
    prompts_per_task: int
    tasks: tuple[str, ...]
    settings: tuple[MatrixSetting, ...]
    description: str | None = None
    source_path: Path = field(default=Path(), repr=False, compare=False)
    raw_payload: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @property
    def completion_count(self) -> int:
        return len(self.tasks) * len(self.settings)

    @property
    def matrix_hash(self) -> str:
        payload = self.canonical_payload()
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "experiment": self.experiment,
            "base_config": self.base_config,
            "prompts_per_task": self.prompts_per_task,
            "tasks": list(self.tasks),
            "settings": [asdict(setting) for setting in self.settings],
            "description": self.description,
        }

    def work_items(self) -> tuple[WorkItem, ...]:
        # Setting-major ordering keeps every task for one setting contiguous.
        # The ordering is part of matrix_hash.
        result: list[WorkItem] = []
        index = 0
        for setting_index, setting in enumerate(self.settings):
            for task_index, task in enumerate(self.tasks):
                result.append(
                    WorkItem(
                        index=index,
                        setting_index=setting_index,
                        task_index=task_index,
                        setting=setting,
                        task=task,
                    )
                )
                index += 1
        return tuple(result)

    def work_item(self, index: int) -> WorkItem:
        items = self.work_items()
        if index < 0 or index >= len(items):
            raise MatrixError(
                f"Completion index {index} is outside [0, {len(items) - 1}]."
            )
        return items[index]

    def resolve_base_config(self) -> Path:
        requested = Path(self.base_config)
        if requested.is_absolute():
            candidate = requested
        else:
            candidates = [self.source_path.parent / requested]
            for parent in (self.source_path, *self.source_path.parents):
                if (parent / "pyproject.toml").is_file():
                    candidates.append(parent / requested)
                    break
            candidate = next((value for value in candidates if value.is_file()), candidates[0])
        if not candidate.is_file():
            raise FileNotFoundError(
                f"Matrix base_config does not exist: {self.base_config!r} "
                f"(resolved to {candidate})"
            )
        return candidate.resolve()

    def load_base_run_config(self) -> RunConfig:
        return load_config(self.resolve_base_config())


def load_matrix(path: str | Path) -> ExperimentMatrix:
    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, Mapping):
        raise MatrixError(f"Matrix root must be a mapping: {source}")
    allowed = {
        "schema_version",
        "experiment",
        "description",
        "base_config",
        "prompts_per_task",
        "tasks",
        "settings",
    }
    unknown = set(raw) - allowed
    if unknown:
        raise MatrixError(f"Unknown matrix field(s): {sorted(unknown)}")

    schema_version = int(raw.get("schema_version", 1))
    if schema_version != 1:
        raise MatrixError(f"Unsupported matrix schema_version={schema_version}.")
    experiment = str(raw.get("experiment", ""))
    if _DNS_LABEL.fullmatch(experiment) is None or len(experiment) > 63:
        raise MatrixError(
            "experiment must be a lowercase Kubernetes-safe DNS label no longer "
            f"than 63 characters: {experiment!r}"
        )
    base_config = str(raw.get("base_config", ""))
    if not base_config:
        raise MatrixError("base_config is required.")
    prompts_per_task = int(raw.get("prompts_per_task", 0))
    if prompts_per_task <= 0:
        raise MatrixError("prompts_per_task must be positive.")

    raw_tasks = raw.get("tasks")
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise MatrixError("tasks must be a non-empty list.")
    tasks = tuple(validate_tasks(str(task) for task in raw_tasks))
    if len(set(tasks)) != len(tasks):
        raise MatrixError("tasks cannot contain duplicates.")

    raw_settings = raw.get("settings")
    if not isinstance(raw_settings, list) or not raw_settings:
        raise MatrixError("settings must be a non-empty list.")
    settings = tuple(
        MatrixSetting.from_mapping(value)
        for value in raw_settings
        if isinstance(value, Mapping)
    )
    if len(settings) != len(raw_settings):
        raise MatrixError("Every settings entry must be a mapping.")
    names = [setting.name for setting in settings]
    if len(set(names)) != len(names):
        raise MatrixError("Setting names must be unique.")

    matrix = ExperimentMatrix(
        schema_version=schema_version,
        experiment=experiment,
        description=(
            str(raw["description"]) if raw.get("description") is not None else None
        ),
        base_config=base_config,
        prompts_per_task=prompts_per_task,
        tasks=tasks,
        settings=settings,
        source_path=source,
        raw_payload=dict(raw),
    )
    matrix.resolve_base_config()
    return matrix


def apply_work_item(
    base: RunConfig,
    matrix: ExperimentMatrix,
    item: WorkItem,
    *,
    local_data_root: str | Path | None = None,
) -> RunConfig:
    """Return a one-setting, one-task config for one Indexed-Job completion."""

    config = copy.deepcopy(base)
    config.benchmark.tasks = [item.task]
    config.benchmark.prompts_per_task = matrix.prompts_per_task
    config.generation.backends = [item.setting.backend]
    config.output.resume = True
    config.output.save_full_prompts = False
    config.output.save_prediction_inputs = False
    config.output.run_name = item.shard_name

    if local_data_root is not None:
        config.benchmark.data.source = "local"
        config.benchmark.data.local_root = str(Path(local_data_root).resolve())

    setting = item.setting
    if setting.backend == "sdpa":
        pass
    elif setting.backend == "santa":
        assert setting.samples_per_head is not None
        config.santa.samples_per_head = setting.samples_per_head
    else:
        if setting.backend == "santapp":
            target = config.santapp
        else:
            if setting.backend not in config.santapp_variants:
                config.santapp_variants[setting.backend] = copy.deepcopy(config.santapp)
            target = config.santapp_variants[setting.backend]
        assert isinstance(target, SantaPlusConfig)
        assert setting.samples_per_head is not None
        target.samples_per_head = setting.samples_per_head
        if setting.mode is not None:
            target.mode = setting.mode
        if setting.group_size is not None:
            target.group_size = setting.group_size
        if setting.probe_queries is not None:
            target.probe_queries = setting.probe_queries
        if setting.parent_size is not None:
            target.parent_size = setting.parent_size
        if setting.representatives_per_parent is not None:
            target.representatives_per_parent = setting.representatives_per_parent

    config.validate()
    return config


def cache_fingerprint(matrix: ExperimentMatrix, base: RunConfig) -> str:
    """Hash all inputs that determine the shared model/data cache."""

    payload = {
        "schema": 1,
        "model": asdict(base.model),
        "context_length": base.benchmark.context_length,
        "selection_seed": base.benchmark.selection_seed,
        "data": asdict(base.benchmark.data),
        "tasks": list(matrix.tasks),
        "prompts_per_task": matrix.prompts_per_task,
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def experiment_root(results_root: str | Path, matrix: ExperimentMatrix) -> Path:
    return Path(results_root).expanduser().resolve() / matrix.experiment
