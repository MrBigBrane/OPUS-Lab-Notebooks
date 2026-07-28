"""RULER's synthetic-task grading rules.

The two metric functions and family-to-metric assignment are intentionally
kept equivalent to NVIDIA/RULER's ``scripts/eval/synthetic/constants.py`` at
the pinned revision. Prediction post-processing follows the corresponding
``scripts/eval/evaluate.py`` behavior.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from .tasks import require_task

_CONTROL_CHARS = re.compile(r"[\x00-\x1f]")


def postprocess_prediction(prediction: str) -> str:
    """Apply RULER's evaluator post-processing to one model prediction."""
    prediction = prediction.strip()
    prediction = _CONTROL_CHARS.sub("\n", prediction).strip()
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


def string_match_part(
    predictions: Sequence[str], references: Sequence[Sequence[str]]
) -> float:
    """RULER partial string match: any reference substring earns full credit."""
    _validate_batch(predictions, references)
    score = sum(
        max(1.0 if ref.lower() in pred.lower() else 0.0 for ref in refs)
        for pred, refs in zip(predictions, references, strict=True)
    ) / len(predictions) * 100.0
    return round(score, 2)


def string_match_all(
    predictions: Sequence[str], references: Sequence[Sequence[str]]
) -> float:
    """RULER all-string match: average recall over required substrings."""
    _validate_batch(predictions, references)
    score = sum(
        sum(1.0 if ref.lower() in pred.lower() else 0.0 for ref in refs)
        / len(refs)
        for pred, refs in zip(predictions, references, strict=True)
    ) / len(predictions) * 100.0
    return round(score, 2)


def metric_for_task(task_name: str):
    family = require_task(task_name).family
    if family == "qa":
        return string_match_part
    return string_match_all


@dataclass(frozen=True, slots=True)
class TaskGrade:
    task: str
    score: float
    null_predictions: int
    num_examples: int

    @property
    def nulls_label(self) -> str:
        return f"{self.null_predictions}/{self.num_examples}"


def grade_task(
    task_name: str,
    predictions: Sequence[str],
    references: Sequence[Sequence[str]],
) -> TaskGrade:
    processed = [postprocess_prediction(pred) for pred in predictions]
    metric = metric_for_task(task_name)
    score = metric(processed, references)
    return TaskGrade(
        task=task_name,
        score=score,
        null_predictions=sum(not pred for pred in processed),
        num_examples=len(processed),
    )


def normalize_references(values: Iterable[object]) -> list[str]:
    refs = [str(value) for value in values]
    if not refs:
        raise ValueError("RULER row contains an empty outputs field.")
    return refs
