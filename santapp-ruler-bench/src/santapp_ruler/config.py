"""Configuration loading, validation, and CLI override handling."""

from __future__ import annotations

import copy
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .io_utils import atomic_write_text
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
    backends: list[str] = field(
        default_factory=lambda: [
            "sdpa",
            "santa",
            "santapp",
            "santapp_gumbel_topk",
            "team_sampler",
            "team_gumbel_topk",
        ]
    )
    stop_on_eos: bool = True
    # null uses each task's official RULER generation budget.
    max_new_tokens: int | None = None
    random_seed: int = 0


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
    """SANTA++ parameters shared by the original and Gumbel Top-K paths.

    Every prompt token is clustered.  The fixed exact prompt-tail window and the
    old random last-quarter probe policy are intentionally not configurable.
    Only subsequently generated tokens form the growing deterministic suffix.
    """

    mode: str = "guided"
    group_size: int = 16
    parent_size: int = 16
    representatives_per_parent: int = 4
    samples_per_head: int = 128
    probe_queries: int = 64
    kmeans: MiniBatchKMeansConfig = field(default_factory=MiniBatchKMeansConfig)


@dataclass(slots=True)
class SantaConfig:
    """Standalone SANTA parameters.

    The sole algorithmic hyperparameter is the per-query-head IID sample budget.
    Reproducibility is controlled globally by ``generation.random_seed``.
    """

    samples_per_head: int = 128


@dataclass(slots=True)
class GradingConfig:
    bootstrap_resamples: int = 10_000
    confidence_level: float = 0.95
    bootstrap_seed: int = 20_260_805


@dataclass(slots=True)
class OutputConfig:
    root: str = "runs"
    run_name: str | None = None
    resume: bool = True
    save_full_prompts: bool = True
    # Keep prediction JSONL compact on shared storage when false.
    save_prediction_inputs: bool = True


def _validate_kmeans(config: MiniBatchKMeansConfig, *, path: str) -> None:
    if config.batch_size <= 0 or config.n_init <= 0 or config.max_iter <= 0:
        raise ValueError(
            f"{path} batch_size, n_init, and max_iter must be positive."
        )
    if config.max_no_improvement is not None and config.max_no_improvement <= 0:
        raise ValueError(f"{path}.max_no_improvement must be positive or null.")
    if config.init_size is not None and config.init_size <= 0:
        raise ValueError(f"{path}.init_size must be positive or null.")
    if config.tol < 0.0:
        raise ValueError(f"{path}.tol cannot be negative.")
    if config.reassignment_ratio < 0.0:
        raise ValueError(f"{path}.reassignment_ratio cannot be negative.")
    if config.random_state < 0:
        raise ValueError(f"{path}.random_state cannot be negative.")


def _validate_santapp(config: SantaPlusConfig, *, path: str) -> None:
    # Existing paths remain unchanged; hierarchical modes opt into a separate
    # parent/team construction explicitly.
    modes = {
        "guided",
        "gumbel_cluster",
        "team_sampler",
        "team_gumbel_topk",
        "oracle_token",
        "uniform",
        "topk",
    }
    if config.mode not in modes:
        raise ValueError(f"{path}.mode must be one of {', '.join(sorted(modes))}.")
    for name, value in {
        "group_size": config.group_size,
        "parent_size": config.parent_size,
        "representatives_per_parent": config.representatives_per_parent,
        "samples_per_head": config.samples_per_head,
        "probe_queries": config.probe_queries,
    }.items():
        if value <= 0:
            raise ValueError(f"{path}.{name} must be positive.")
    if config.mode == "gumbel_cluster":
        if config.samples_per_head % config.group_size:
            raise ValueError(
                f"{path}.samples_per_head must be divisible by group_size in "
                "gumbel_cluster mode."
            )
        if config.samples_per_head // config.group_size < 1:
            raise ValueError(f"{path} must request at least one cluster.")
    if config.representatives_per_parent > config.parent_size:
        raise ValueError(
            f"{path}.representatives_per_parent must be <= parent_size."
        )
    if config.mode == "team_gumbel_topk":
        if config.parent_size % config.representatives_per_parent:
            raise ValueError(
                f"{path}.parent_size must be divisible by "
                "representatives_per_parent in team_gumbel_topk mode."
            )
        nominal_team_size = config.parent_size // config.representatives_per_parent
        if config.samples_per_head % nominal_team_size:
            raise ValueError(
                f"{path}.samples_per_head must be divisible by nominal_team_size "
                "in team_gumbel_topk mode."
            )
        if config.samples_per_head // nominal_team_size < 1:
            raise ValueError(f"{path} must request at least one team.")
    _validate_kmeans(config.kmeans, path=f"{path}.kmeans")


STANDARD_SANTAPP_VARIANT_MODES = {
    "santapp_gumbel_topk": "gumbel_cluster",
    "team_sampler": "team_sampler",
    "team_gumbel_topk": "team_gumbel_topk",
}


def _default_santapp_variants() -> dict[str, SantaPlusConfig]:
    """Return the standard named SANTA++ method configurations."""

    return {
        name: SantaPlusConfig(mode=mode)
        for name, mode in STANDARD_SANTAPP_VARIANT_MODES.items()
    }


@dataclass(slots=True)
class RunConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    santa: SantaConfig = field(default_factory=SantaConfig)
    santapp: SantaPlusConfig = field(default_factory=SantaPlusConfig)
    santapp_variants: dict[str, SantaPlusConfig] = field(
        default_factory=_default_santapp_variants
    )
    grading: GradingConfig = field(default_factory=GradingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def validate(self) -> None:
        self.benchmark.tasks = list(validate_tasks(self.benchmark.tasks))
        if self.benchmark.context_length < 256:
            raise ValueError("benchmark.context_length must be at least 256.")
        if self.benchmark.prompts_per_task <= 0:
            raise ValueError("benchmark.prompts_per_task must be positive.")
        if self.model.device != "cuda":
            raise ValueError(
                "The SANTA and SANTA++ reference backends currently require "
                "model.device=cuda."
            )
        if self.model.attn_implementation != "sdpa":
            raise ValueError("model.attn_implementation must be 'sdpa'.")
        if self.model.dtype not in {"float16", "bfloat16"}:
            raise ValueError("model.dtype must be float16 or bfloat16.")

        if not self.generation.backends:
            raise ValueError("At least one generation backend must be selected.")
        if len(set(self.generation.backends)) != len(self.generation.backends):
            raise ValueError("generation.backends cannot contain duplicates.")
        if self.generation.max_new_tokens is not None:
            if self.generation.max_new_tokens <= 0:
                raise ValueError("generation.max_new_tokens must be positive.")
        if self.generation.random_seed < 0:
            raise ValueError("generation.random_seed cannot be negative.")

        for name in self.santapp_variants:
            if name in {"sdpa", "santa", "santapp"}:
                raise ValueError(f"santapp_variants name {name!r} is reserved.")
            if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) is None:
                raise ValueError(
                    "SANTA++ variant names may contain only letters, digits, '.', "
                    f"'_', or '-': {name!r}."
                )
        supported = {"sdpa", "santa", "santapp", *self.santapp_variants}
        unknown = set(self.generation.backends) - supported
        if unknown:
            raise ValueError(f"Unknown generation backend(s): {sorted(unknown)}")

        if self.santa.samples_per_head <= 0:
            raise ValueError("santa.samples_per_head must be positive.")
        _validate_santapp(self.santapp, path="santapp")
        for name, variant in self.santapp_variants.items():
            _validate_santapp(variant, path=f"santapp_variants.{name}")

        if self.grading.bootstrap_resamples <= 0:
            raise ValueError("grading.bootstrap_resamples must be positive.")
        if not 0.0 < self.grading.confidence_level < 1.0:
            raise ValueError(
                "grading.confidence_level must be strictly between zero and one."
            )
        if self.grading.bootstrap_seed < 0:
            raise ValueError("grading.bootstrap_seed cannot be negative.")

        data = self.benchmark.data
        if data.source not in {"huggingface", "local"}:
            raise ValueError("benchmark.data.source must be huggingface or local.")
        # if data.source == "huggingface":
        #     if (
        #         self.benchmark.context_length != 8192
        #         and data.repository == DEFAULT_DATASET_REPOSITORY
        #     ):
        #         raise ValueError(
        #             "The default Hugging Face mirror contains 8192-token RULER "
        #             "data. For another context length, generate official JSONL and "
        #             "set benchmark.data.source=local plus benchmark.data.local_root."
        #         )
        elif not data.local_root:
            raise ValueError(
                "benchmark.data.local_root is required when data.source=local."
            )

    def santapp_config_for_backend(self, backend: str) -> SantaPlusConfig:
        if backend == "santapp":
            return self.santapp
        try:
            return self.santapp_variants[backend]
        except KeyError as exc:
            raise KeyError(f"No SANTA++ config for backend {backend!r}.") from exc

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_CONFIG = RunConfig()


def _default_config_payload() -> dict[str, Any]:
    """Return merge-ready defaults while preserving variant inheritance.

    ``RunConfig.to_dict()`` expands every named variant into a complete
    ``SantaPlusConfig``.  Using that expanded form as the merge base would pin
    all inherited fields (sample budget, group/parent size, probes, and k-means
    settings) to their original defaults.  Keep only each built-in variant's
    method-specific mode in the raw payload so changes to the base ``santapp``
    block continue to flow into variants unless explicitly overridden.
    """

    payload = DEFAULT_CONFIG.to_dict()
    payload["santapp_variants"] = {
        name: {"mode": mode}
        for name, mode in STANDARD_SANTAPP_VARIANT_MODES.items()
    }
    return payload


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
    santa = SantaConfig(**data.get("santa", {}))

    base_raw = dict(data.get("santapp", {}))

    def construct_santapp(raw: dict[str, Any]) -> SantaPlusConfig:
        value = copy.deepcopy(raw)
        kmeans = MiniBatchKMeansConfig(**value.pop("kmeans", {}))
        return SantaPlusConfig(kmeans=kmeans, **value)

    santapp = construct_santapp(base_raw)
    variants_raw = data.get("santapp_variants", {}) or {}
    if not isinstance(variants_raw, dict):
        raise ValueError("santapp_variants must be a mapping.")
    variants: dict[str, SantaPlusConfig] = {}
    for name, raw in variants_raw.items():
        if not isinstance(raw, dict):
            raise ValueError(f"santapp_variants.{name} must be a mapping.")
        variants[name] = construct_santapp(_deep_merge(base_raw, raw))

    grading = GradingConfig(**data.get("grading", {}))
    output = OutputConfig(**data.get("output", {}))
    config = RunConfig(
        model=model,
        benchmark=benchmark,
        generation=generation,
        santa=santa,
        santapp=santapp,
        santapp_variants=variants,
        grading=grading,
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
    base = _default_config_payload()
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
    rendered = yaml.safe_dump(config.to_dict(), sort_keys=False)
    atomic_write_text(path, rendered)


def config_as_json(config: RunConfig) -> str:
    return json.dumps(config.to_dict(), indent=2, sort_keys=True)
