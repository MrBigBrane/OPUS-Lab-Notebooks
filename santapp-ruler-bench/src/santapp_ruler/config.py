"""Configuration loading, validation, and CLI override handling."""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .ruler.provenance import (
    DEFAULT_DATASET_REPOSITORY,
    DEFAULT_DATASET_REVISION,
    DEFAULT_MODEL_REPOSITORY,
    DEFAULT_MODEL_REVISION,
)
from .ruler.tasks import DEFAULT_TASKS, validate_tasks


@dataclass(slots=True)
class ModelConfig:
    name: str = DEFAULT_MODEL_REPOSITORY
    revision: str | None = DEFAULT_MODEL_REVISION
    dtype: str = "float16"
    device: str = "cuda"
    attn_implementation: str = "sdpa"
    trust_remote_code: bool = False
    disable_hf_xet: bool = True


@dataclass(slots=True)
class DataConfig:
    source: str = "huggingface"
    repository: str = DEFAULT_DATASET_REPOSITORY
    revision: str = DEFAULT_DATASET_REVISION
    local_root: str | None = None
    subset: str = "validation"


@dataclass(slots=True)
class BenchmarkConfig:
    context_length: int = 8192
    tasks: list[str] = field(default_factory=lambda: list(DEFAULT_TASKS))
    prompts_per_task: int = 5
    selection_seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)


@dataclass(slots=True)
class GenerationConfig:
    backends: list[str] = field(default_factory=lambda: ["sdpa", "santapp"])
    stop_on_eos: bool = True
    max_new_tokens_cap: int | None = None


@dataclass(slots=True)
class MiniBatchKMeansConfig:
    batch_size: int = 4096
    n_init: int = 1
    max_iter: int = 100
    tol: float = 0.0
    max_no_improvement: int | None = 10
    init_size: int | None = None
    reassignment_ratio: float = 0.01
    random_state: int = 0


@dataclass(slots=True)
class SantaPlusConfig:
    mode: str = "guided"
    group_size: int = 16
    samples_per_head: int = 128
    probe_queries: int = 64
    recent_window: int = 64
    probe_region_start_fraction: float = 0.75
    probe_strategy: str = "end_quarter"  # <--- ADD THIS FIELD (options: end_quarter, start, middle, end, random)
    sample_seed: int = 0
    kmeans: MiniBatchKMeansConfig = field(default_factory=MiniBatchKMeansConfig)


@dataclass(slots=True)
class OutputConfig:
    root: str = "runs"
    run_name: str | None = None
    resume: bool = True
    save_full_prompts: bool = True


@dataclass(slots=True)
class RunConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    santapp: SantaPlusConfig = field(default_factory=SantaPlusConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def validate(self) -> None:
        self.benchmark.tasks = list(validate_tasks(self.benchmark.tasks))
        if self.benchmark.context_length < 256:
            raise ValueError("benchmark.context_length must be at least 256.")
        if self.benchmark.prompts_per_task <= 0:
            raise ValueError("benchmark.prompts_per_task must be positive.")
        if self.model.device != "cuda":
            raise ValueError(
                "This reference SANTA++ implementation currently requires model.device=cuda."
            )
        if self.model.attn_implementation != "sdpa":
            raise ValueError("model.attn_implementation must be 'sdpa'.")
        if self.model.dtype not in {"float16", "bfloat16"}:
            raise ValueError("model.dtype must be float16 or bfloat16.")
        supported_backends = {"sdpa", "santapp"}
        unknown = set(self.generation.backends) - supported_backends
        if unknown:
            raise ValueError(f"Unknown generation backend(s): {sorted(unknown)}")
        if not self.generation.backends:
            raise ValueError("At least one generation backend must be selected.")
        if self.generation.max_new_tokens_cap is not None:
            if self.generation.max_new_tokens_cap <= 0:
                raise ValueError("generation.max_new_tokens_cap must be positive.")
        if self.santapp.mode not in {"guided", "santa", "uniform", "topk"}:
            raise ValueError(
                "santapp.mode must be one of guided, santa, uniform, or topk."
            )
        for name, value in {
            "group_size": self.santapp.group_size,
            "samples_per_head": self.santapp.samples_per_head,
            "probe_queries": self.santapp.probe_queries,
            "recent_window": self.santapp.recent_window,
        }.items():
            if value <= 0:
                raise ValueError(f"santapp.{name} must be positive.")
        if not 0.0 <= self.santapp.probe_region_start_fraction < 1.0:
            raise ValueError(
                "santapp.probe_region_start_fraction must be in [0, 1)."
            )
        valid_strategies = {"end_quarter", "start", "middle", "end", "random"}
        if self.santapp.probe_strategy not in valid_strategies:
            raise ValueError(
                f"santapp.probe_strategy must be one of {sorted(valid_strategies)}, "
                f"got {self.santapp.probe_strategy!r}."
            )
        if self.benchmark.data.source not in {"huggingface", "local"}:
            raise ValueError("benchmark.data.source must be huggingface or local.")
        if self.benchmark.data.source == "huggingface":
            if self.benchmark.context_length != 8192 and (
                self.benchmark.data.repository == DEFAULT_DATASET_REPOSITORY
            ):
                raise ValueError(
                    "The default Hugging Face mirror contains 8192-token RULER data. "
                    "For another context length, generate official JSONL and set "
                    "benchmark.data.source=local plus benchmark.data.local_root."
                )
        else:
            if not self.benchmark.data.local_root:
                raise ValueError(
                    "benchmark.data.local_root is required when data.source=local."
                )
        km = self.santapp.kmeans
        if km.batch_size <= 0 or km.n_init <= 0 or km.max_iter <= 0:
            raise ValueError("MiniBatchKMeans batch_size, n_init, and max_iter must be positive.")
        if not 0.0 <= km.reassignment_ratio:
            raise ValueError("MiniBatchKMeans reassignment_ratio cannot be negative.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_CONFIG = RunConfig()


def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _construct(data: dict[str, Any]) -> RunConfig:
    model = ModelConfig(**data.get("model", {}))
    benchmark_raw = dict(data.get("benchmark", {}))
    data_cfg = DataConfig(**benchmark_raw.pop("data", {}))
    benchmark = BenchmarkConfig(data=data_cfg, **benchmark_raw)
    generation = GenerationConfig(**data.get("generation", {}))
    santapp_raw = dict(data.get("santapp", {}))
    kmeans = MiniBatchKMeansConfig(**santapp_raw.pop("kmeans", {}))
    santapp = SantaPlusConfig(kmeans=kmeans, **santapp_raw)
    output = OutputConfig(**data.get("output", {}))
    config = RunConfig(
        model=model,
        benchmark=benchmark,
        generation=generation,
        santapp=santapp,
        output=output,
    )
    config.validate()
    return config


def parse_scalar(value: str) -> Any:
    """Parse a ``--set`` value using YAML scalar/list syntax."""
    return yaml.safe_load(value)


def apply_dotted_overrides(
    data: dict[str, Any], overrides: list[str] | tuple[str, ...]
) -> dict[str, Any]:
    result = copy.deepcopy(data)
    for expression in overrides:
        if "=" not in expression:
            raise ValueError(
                f"Invalid override {expression!r}; expected dotted.path=value."
            )
        dotted, raw_value = expression.split("=", 1)
        keys = [part.strip() for part in dotted.split(".") if part.strip()]
        if not keys:
            raise ValueError(f"Invalid empty override path in {expression!r}.")
        cursor: dict[str, Any] = result
        for key in keys[:-1]:
            child = cursor.setdefault(key, {})
            if not isinstance(child, dict):
                raise ValueError(
                    f"Cannot descend through non-mapping config field {key!r}."
                )
            cursor = child
        cursor[keys[-1]] = parse_scalar(raw_value)
    return result


def load_config(
    path: str | Path | None = None,
    *,
    overrides: list[str] | tuple[str, ...] = (),
) -> RunConfig:
    base = DEFAULT_CONFIG.to_dict()
    if path is not None:
        config_path = Path(path)
        with config_path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"Config root must be a mapping: {config_path}")
        base = _deep_merge(base, loaded)
    base = apply_dotted_overrides(base, overrides)
    return _construct(base)


def save_config(config: RunConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        yaml.safe_dump(config.to_dict(), handle, sort_keys=False)


def config_as_json(config: RunConfig) -> str:
    return json.dumps(config.to_dict(), indent=2, sort_keys=True)
