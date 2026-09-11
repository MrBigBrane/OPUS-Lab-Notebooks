"""Typed configuration loading, validation, and command-line overrides."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .io_utils import atomic_write_text
from .ruler.provenance import (
    DEFAULT_DATASET_REVISION,
    DEFAULT_MODEL_REPOSITORY,
    DEFAULT_MODEL_REVISION,
    default_dataset_repository,
)
from .ruler.tasks import DEFAULT_TASKS, validate_tasks

SUPPORTED_BACKENDS = ("sdpa", "santa", "santapp", "hierarchical")
SUPPORTED_PROMPT_FORMATS = ("raw", "chat_template")


@dataclass(slots=True)
class ModelConfig:
    name: str = DEFAULT_MODEL_REPOSITORY
    revision: str | None = DEFAULT_MODEL_REVISION
    dtype: str = "float16"
    device: str = "cuda"
    attn_implementation: str = "sdpa"
    trust_remote_code: bool = False
    disable_hf_xet: bool = True
    # The default checkpoint is instruction-tuned, so present the complete
    # prepared RULER string as one user turn through its tokenizer-owned chat
    # template. ``raw`` remains available as an explicitly labeled alternative.
    prompt_format: str = "chat_template"


@dataclass(slots=True)
class DataConfig:
    source: str = "huggingface"
    # ``auto`` maps 8192 and 32768 to the matching convenience mirrors.
    repository: str = "auto"
    revision: str = DEFAULT_DATASET_REVISION
    local_root: str | None = None
    subset: str = "validation"


@dataclass(slots=True)
class BenchmarkConfig:
    context_length: int = 8192
    tasks: list[str] = field(default_factory=lambda: list(DEFAULT_TASKS))
    prompts_per_task: int = 1
    selection_seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)


@dataclass(slots=True)
class GenerationConfig:
    backends: list[str] = field(default_factory=lambda: list(SUPPORTED_BACKENDS))
    stop_on_eos: bool = True
    # null uses the official per-task RULER generation budget.
    max_new_tokens: int | None = None
    random_seed: int = 0


@dataclass(slots=True)
class SantaConfig:
    """SANTA token-IID sample budget per query head."""

    samples_per_head: int = 128


@dataclass(slots=True)
class MiniBatchKMeansConfig:
    """Parameters matching the previous GPU MiniBatchKMeans parent stage."""

    batch_size: int = 4096
    n_init: int = 1
    max_iter: int = 100
    tol: float = 0.0
    max_no_improvement: int | None = 10
    init_size: int | None = None
    reassignment_ratio: float = 0.01
    random_state: int = 0


@dataclass(slots=True)
class SantappConfig:
    """Post-RoPE-key k-means plus hierarchical whole-team sampling."""

    # ``parent_size`` is the nominal target cardinality used to choose
    # floor(prompt_tokens / parent_size) global k-means parents. Actual parent
    # sizes are data dependent.
    parent_size: int = 16
    representatives_per_parent: int = 4
    # Nominal token budget. It is converted to a team count using
    # parent_size / representatives_per_parent; actual selected rows vary with
    # the learned parent/team cardinalities.
    samples_per_head: int = 128
    kmeans: MiniBatchKMeansConfig = field(default_factory=MiniBatchKMeansConfig)
    prefill_backend: str = "triton"


@dataclass(slots=True)
class HierarchicalConfig:
    """Contiguous-parent hierarchical whole-team sampling parameters."""

    prefill_backend: str = "triton"

    parent_size: int = 16
    representatives_per_parent: int = 4
    # Nominal token budget. It is converted to a team count using
    # parent_size / representatives_per_parent; actual selected rows vary with
    # data-dependent team sizes and a potentially short final parent.
    samples_per_head: int = 128


@dataclass(slots=True)
class GradingConfig:
    bootstrap_resamples: int = 10_000
    confidence_level: float = 0.95
    bootstrap_seed: int = 20_260_903


@dataclass(slots=True)
class OutputConfig:
    root: str = "runs"
    run_name: str | None = None
    resume: bool = True
    save_full_prompts: bool = True
    save_prediction_inputs: bool = True


@dataclass(slots=True)
class RunConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    santa: SantaConfig = field(default_factory=SantaConfig)
    santapp: SantappConfig = field(default_factory=SantappConfig)
    hierarchical: HierarchicalConfig = field(default_factory=HierarchicalConfig)
    grading: GradingConfig = field(default_factory=GradingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def validate(self) -> None:
        self.benchmark.tasks = list(validate_tasks(self.benchmark.tasks))
        if self.benchmark.context_length < 256:
            raise ValueError("benchmark.context_length must be at least 256.")
        if self.benchmark.prompts_per_task <= 0:
            raise ValueError("benchmark.prompts_per_task must be positive.")

        if self.model.device != "cuda":
            raise ValueError("The model-backed benchmark currently requires model.device=cuda.")
        if self.model.attn_implementation != "sdpa":
            raise ValueError("model.attn_implementation must be 'sdpa'.")
        if self.model.dtype not in {"float16", "bfloat16"}:
            raise ValueError("model.dtype must be float16 or bfloat16.")
        if self.model.prompt_format not in SUPPORTED_PROMPT_FORMATS:
            raise ValueError(
                "model.prompt_format must be one of "
                f"{', '.join(SUPPORTED_PROMPT_FORMATS)}."
            )

        if not self.generation.backends:
            raise ValueError("At least one generation backend must be selected.")
        if len(set(self.generation.backends)) != len(self.generation.backends):
            raise ValueError("generation.backends cannot contain duplicates.")
        unknown = set(self.generation.backends) - set(SUPPORTED_BACKENDS)
        if unknown:
            raise ValueError(
                f"Unknown generation backend(s): {sorted(unknown)}; supported: "
                f"{list(SUPPORTED_BACKENDS)}."
            )
        if self.generation.max_new_tokens is not None and self.generation.max_new_tokens <= 0:
            raise ValueError("generation.max_new_tokens must be positive or null.")
        if self.generation.random_seed < 0:
            raise ValueError("generation.random_seed cannot be negative.")

        if self.santa.samples_per_head <= 0:
            raise ValueError("santa.samples_per_head must be positive.")
        h = self.santapp
        if h.parent_size <= 0:
            raise ValueError("santapp.parent_size must be positive.")
        if h.representatives_per_parent <= 0:
            raise ValueError("santapp.representatives_per_parent must be positive.")
        if h.representatives_per_parent > h.parent_size:
            raise ValueError(
                "santapp.representatives_per_parent must be <= parent_size."
            )
        if h.parent_size % h.representatives_per_parent:
            raise ValueError(
                "santapp.parent_size must be divisible by representatives_per_parent "
                "so the nominal team size is integral."
            )
        if h.samples_per_head <= 0:
            raise ValueError("santapp.samples_per_head must be positive.")
        nominal_team_size = h.parent_size // h.representatives_per_parent
        if h.samples_per_head % nominal_team_size:
            raise ValueError(
                "santapp.samples_per_head must be divisible by the nominal team "
                f"size ({nominal_team_size})."
            )

        km = h.kmeans
        if km.batch_size <= 0 or km.n_init <= 0 or km.max_iter <= 0:
            raise ValueError(
                "santapp.kmeans batch_size, n_init, and max_iter must be positive."
            )
        if km.max_no_improvement is not None and km.max_no_improvement <= 0:
            raise ValueError(
                "santapp.kmeans.max_no_improvement must be positive or null."
            )
        if km.init_size is not None and km.init_size <= 0:
            raise ValueError("santapp.kmeans.init_size must be positive or null.")
        if km.tol < 0.0:
            raise ValueError("santapp.kmeans.tol cannot be negative.")
        if km.reassignment_ratio < 0.0:
            raise ValueError("santapp.kmeans.reassignment_ratio cannot be negative.")
        if km.random_state < 0:
            raise ValueError("santapp.kmeans.random_state cannot be negative.")

        c = self.hierarchical
        if c.parent_size <= 0:
            raise ValueError("hierarchical.parent_size must be positive.")
        if c.representatives_per_parent <= 0:
            raise ValueError(
                "hierarchical.representatives_per_parent must be positive."
            )
        if c.representatives_per_parent > c.parent_size:
            raise ValueError(
                "hierarchical.representatives_per_parent must be <= parent_size."
            )
        if c.parent_size % c.representatives_per_parent:
            raise ValueError(
                "hierarchical.parent_size must be divisible by "
                "representatives_per_parent so the nominal team size is integral."
            )
        if c.samples_per_head <= 0:
            raise ValueError("hierarchical.samples_per_head must be positive.")
        contiguous_nominal_team_size = (
            c.parent_size // c.representatives_per_parent
        )
        if c.samples_per_head % contiguous_nominal_team_size:
            raise ValueError(
                "hierarchical.samples_per_head must be divisible by the nominal "
                f"team size ({contiguous_nominal_team_size})."
            )

        for name in ("santapp", "hierarchical"):
            sparse = getattr(self, name)
            if sparse.prefill_backend not in {"torch", "triton"}:
                raise ValueError(f"{name}.prefill_backend must be torch or triton.")
            if sparse.prefill_backend == "triton" and sparse.representatives_per_parent > 16:
                raise ValueError(f"{name}: native Triton supports at most 16 representatives.")
        if self.santapp.prefill_backend == "triton" and km.reassignment_ratio > 1:
            raise ValueError("Triton k-means requires reassignment_ratio <= 1.")

        if self.grading.bootstrap_resamples <= 0:
            raise ValueError("grading.bootstrap_resamples must be positive.")
        if not 0.0 < self.grading.confidence_level < 1.0:
            raise ValueError("grading.confidence_level must lie strictly between 0 and 1.")
        if self.grading.bootstrap_seed < 0:
            raise ValueError("grading.bootstrap_seed cannot be negative.")

        data = self.benchmark.data
        if data.source not in {"huggingface", "local"}:
            raise ValueError("benchmark.data.source must be huggingface or local.")
        if data.source == "huggingface":
            if not data.repository:
                raise ValueError("benchmark.data.repository cannot be empty.")
            if data.repository == "auto":
                default_dataset_repository(self.benchmark.context_length)
        elif not data.local_root:
            raise ValueError(
                "benchmark.data.local_root is required when benchmark.data.source=local."
            )

    def resolved_dataset_repository(self) -> str:
        repository = self.benchmark.data.repository
        if repository == "auto":
            return default_dataset_repository(self.benchmark.context_length)
        return repository

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
    santapp_raw = dict(data.get("santapp", {}))
    kmeans = MiniBatchKMeansConfig(**santapp_raw.pop("kmeans", {}))
    config = RunConfig(
        model=model,
        benchmark=benchmark,
        generation=GenerationConfig(**data.get("generation", {})),
        santa=SantaConfig(**data.get("santa", {})),
        santapp=SantappConfig(kmeans=kmeans, **santapp_raw),
        hierarchical=HierarchicalConfig(**data.get("hierarchical", {})),
        grading=GradingConfig(**data.get("grading", {})),
        output=OutputConfig(**data.get("output", {})),
    )
    config.validate()
    return config


def parse_scalar(value: str) -> Any:
    """Parse a ``--set`` value with YAML scalar/list syntax."""

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
    return _construct(apply_dotted_overrides(base, overrides))


def save_config(config: RunConfig, path: str | Path) -> None:
    rendered = yaml.safe_dump(config.to_dict(), sort_keys=False)
    atomic_write_text(path, rendered)
