"""RULER synthetic-task grading rules and deterministic bootstrap intervals.

The two metric functions and family-to-metric assignment are intentionally
kept equivalent to NVIDIA/RULER's ``scripts/eval/synthetic/constants.py`` at
the pinned revision. Prediction post-processing follows the corresponding
``scripts/eval/evaluate.py`` behavior with regex stop-pattern truncation.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from statistics import fmean
from typing import Any

import numpy as np

from .tasks import require_task

_CONTROL_CHARS = re.compile(r"[\x00-\x1f]")

# Patterns that signal prompt hallucination / generation overrunning answer boundary
_STOP_PATTERNS = re.compile(
    r"(?:"
    r"You are an AI assistant|"
    r"Please provide|"
    r"<\|im_start\|>|"
    r"<\|im_end\|>|"
    r"<\|endoftext\|>|"
    r"\n\n|"
    r"\nUser:|"
    r"\nQuestion:"
    r")",
    flags=re.IGNORECASE,
)


def postprocess_prediction(prediction: str) -> str:
    """Apply RULER's evaluator post-processing and regex truncation to one model prediction."""
    prediction = prediction.strip()
    prediction = _CONTROL_CHARS.sub("\n", prediction).strip()
    # Truncate string at the first occurrence of prompt leakage or template artifacts
    prediction = _STOP_PATTERNS.split(prediction)[0].strip()
    return prediction


def _validate_batch(
    predictions: Sequence[str], references: Sequence[Sequence[str]]
) -> None:
    if len(predictions) != len(references):
        raise ValueError(
            "Predictions and references must have equal length: "
            f"{len(predictions)} != {len(references)}"
        )
    if not predictions:
        raise ValueError("Cannot grade an empty task.")
    for index, refs in enumerate(references):
        if not refs:
            raise ValueError(f"Example {index} has no reference answers.")


def _part_score(prediction: str, references: Sequence[str]) -> float:
    return 100.0 * max(
        1.0 if reference.lower() in prediction.lower() else 0.0
        for reference in references
    )


def _all_score(prediction: str, references: Sequence[str]) -> float:
    return 100.0 * sum(
        1.0 if reference.lower() in prediction.lower() else 0.0
        for reference in references
    ) / len(references)


def normalize_answer(s: str) -> str:
    """Normalize text for SQuAD/HELMET exact match and token F1."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def _f1_score(prediction: str, references: Sequence[str]) -> float:
    def compute_f1(gold, pred):
        gold_toks = normalize_answer(gold).split()
        pred_toks = normalize_answer(pred).split()
        common = Counter(gold_toks) & Counter(pred_toks)
        num_same = sum(common.values())
        if not gold_toks or not pred_toks:
            return 1.0 if gold_toks == pred_toks else 0.0
        if num_same == 0:
            return 0.0
        precision = 1.0 * num_same / len(pred_toks)
        recall = 1.0 * num_same / len(gold_toks)
        return 100.0 * (2 * precision * recall) / (precision + recall)

    return max(compute_f1(ref, prediction) for ref in references)


def token_f1(predictions: Sequence[str], references: Sequence[Sequence[str]]) -> float:
    _validate_batch(predictions, references)
    return round(fmean(_f1_score(p, r) for p, r in zip(predictions, references, strict=True)), 2)


def string_match_part(
    predictions: Sequence[str], references: Sequence[Sequence[str]]
) -> float:
    """RULER partial string match: any reference substring earns full credit."""
    _validate_batch(predictions, references)
    return round(
        fmean(
            _part_score(prediction, refs)
            for prediction, refs in zip(predictions, references, strict=True)
        ),
        2,
    )


def string_match_all(
    predictions: Sequence[str], references: Sequence[Sequence[str]]
) -> float:
    """RULER all-string match: average recall over required substrings."""
    _validate_batch(predictions, references)
    return round(
        fmean(
            _all_score(prediction, refs)
            for prediction, refs in zip(predictions, references, strict=True)
        ),
        2,
    )


def metric_for_task(task_name: str):
    family = require_task(task_name).family
    if family == "qa":
        return string_match_part
    if family == "rag":
        return token_f1
    return string_match_all


@dataclass(frozen=True, slots=True)
class BootstrapInterval:
    """Deterministic percentile-bootstrap interval for a mean statistic."""

    point_estimate: float
    lower: float
    upper: float
    confidence_level: float
    resamples: int
    seed: int
    method: str = "percentile"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TaskGrade:
    task: str
    score: float
    null_predictions: int
    num_examples: int
    example_scores: tuple[float, ...]
    confidence_interval: BootstrapInterval | None = None

    @property
    def nulls_label(self) -> str:
        return f"{self.null_predictions}/{self.num_examples}"


def _validate_bootstrap(
    *, resamples: int, confidence_level: float, seed: int
) -> None:
    if resamples <= 0:
        raise ValueError("bootstrap resamples must be positive.")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be strictly between zero and one.")
    if seed < 0:
        raise ValueError("bootstrap seed cannot be negative.")


def bootstrap_mean_confidence_interval(
    values: Sequence[float] | np.ndarray,
    *,
    resamples: int = 10_000,
    confidence_level: float = 0.95,
    seed: int = 20_260_805,
) -> BootstrapInterval:
    """Bootstrap a mean by resampling observations with replacement."""
    _validate_bootstrap(
        resamples=resamples,
        confidence_level=confidence_level,
        seed=seed,
    )
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("bootstrap values must be a non-empty one-dimensional array.")
    if not np.isfinite(array).all():
        raise ValueError("bootstrap values must be finite.")

    point = float(array.mean())
    if array.size == 1:
        lower = upper = point
    else:
        rng = np.random.default_rng(seed)
        indices = rng.integers(0, array.size, size=(resamples, array.size))
        bootstrap_means = array[indices].mean(axis=1)
        alpha = (1.0 - confidence_level) / 2.0
        lower, upper = np.quantile(bootstrap_means, [alpha, 1.0 - alpha])
        lower = float(lower)
        upper = float(upper)
    return BootstrapInterval(
        point_estimate=point,
        lower=lower,
        upper=upper,
        confidence_level=confidence_level,
        resamples=resamples,
        seed=seed,
    )


def stratified_bootstrap_mean_confidence_interval(
    strata: Mapping[str, Sequence[float] | np.ndarray],
    *,
    resamples: int = 10_000,
    confidence_level: float = 0.95,
    seed: int = 20_260_805,
) -> BootstrapInterval:
    """Bootstrap an equally weighted mean while resampling within each task."""
    _validate_bootstrap(
        resamples=resamples,
        confidence_level=confidence_level,
        seed=seed,
    )
    if not strata:
        raise ValueError("stratified bootstrap requires at least one stratum.")

    arrays: list[np.ndarray] = []
    for name, values in strata.items():
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 1 or array.size == 0:
            raise ValueError(
                f"Bootstrap stratum {name!r} must be a non-empty one-dimensional array."
            )
        if not np.isfinite(array).all():
            raise ValueError(f"Bootstrap stratum {name!r} contains non-finite values.")
        arrays.append(array)

    point = float(np.mean([array.mean() for array in arrays]))
    rng = np.random.default_rng(seed)
    aggregate = np.zeros(resamples, dtype=np.float64)
    for array in arrays:
        if array.size == 1:
            aggregate += float(array[0])
        else:
            indices = rng.integers(0, array.size, size=(resamples, array.size))
            aggregate += array[indices].mean(axis=1)
    aggregate /= len(arrays)
    alpha = (1.0 - confidence_level) / 2.0
    lower, upper = np.quantile(aggregate, [alpha, 1.0 - alpha])
    return BootstrapInterval(
        point_estimate=point,
        lower=float(lower),
        upper=float(upper),
        confidence_level=confidence_level,
        resamples=resamples,
        seed=seed,
    )


def grade_task(
    task_name: str,
    predictions: Sequence[str],
    references: Sequence[Sequence[str]],
    *,
    bootstrap_resamples: int | None = None,
    confidence_level: float = 0.95,
    bootstrap_seed: int = 20_260_805,
) -> TaskGrade:
    processed = [postprocess_prediction(prediction) for prediction in predictions]
    _validate_batch(processed, references)
    family = require_task(task_name).family
    if family == "qa":
        score_function = _part_score
    elif family == "rag":
        score_function = _f1_score
    else:
        score_function = _all_score
    example_scores = tuple(
        score_function(prediction, refs)
        for prediction, refs in zip(processed, references, strict=True)
    )
    score = round(fmean(example_scores), 2)
    interval = (
        bootstrap_mean_confidence_interval(
            example_scores,
            resamples=bootstrap_resamples,
            confidence_level=confidence_level,
            seed=bootstrap_seed,
        )
        if bootstrap_resamples is not None
        else None
    )
    return TaskGrade(
        task=task_name,
        score=score,
        null_predictions=sum(not prediction for prediction in processed),
        num_examples=len(processed),
        example_scores=example_scores,
        confidence_interval=interval,
    )


def normalize_references(values: Iterable[object]) -> list[str]:
    refs = [str(value) for value in values]
    if not refs:
        raise ValueError("RULER row contains an empty outputs field.")
    return refs