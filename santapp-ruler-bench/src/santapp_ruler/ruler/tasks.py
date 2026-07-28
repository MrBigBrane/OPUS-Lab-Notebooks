"""RULER synthetic task registry.

Task names, task families, and generation budgets mirror NVIDIA/RULER's
classic synthetic benchmark configuration at the pinned upstream revision in
:mod:`santapp_ruler.ruler.provenance`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True, slots=True)
class RulerTask:
    name: str
    family: str
    max_new_tokens: int
    description: str


TASKS: Final[dict[str, RulerTask]] = {
    "niah_single_1": RulerTask(
        "niah_single_1", "niah", 128, "single needle: numbers in repeated grass text"
    ),
    "niah_single_2": RulerTask(
        "niah_single_2", "niah", 128, "single needle: words in Paul Graham essays"
    ),
    "niah_single_3": RulerTask(
        "niah_single_3", "niah", 128, "single needle: UUIDs in repeated grass text"
    ),
    "niah_multikey_1": RulerTask(
        "niah_multikey_1", "niah", 128, "multiple keys associated with one value"
    ),
    "niah_multikey_2": RulerTask(
        "niah_multikey_2", "niah", 128, "multiple word keys associated with one value"
    ),
    "niah_multikey_3": RulerTask(
        "niah_multikey_3", "niah", 128, "multiple UUID keys associated with one value"
    ),
    "niah_multivalue": RulerTask(
        "niah_multivalue", "niah", 128, "one key associated with multiple values"
    ),
    "niah_multiquery": RulerTask(
        "niah_multiquery", "niah", 128, "retrieve values for multiple queried keys"
    ),
    "vt": RulerTask(
        "vt", "variable_tracking", 30, "variable-assignment chain tracking"
    ),
    "cwe": RulerTask(
        "cwe", "common_words_extraction", 120, "extract the ten most common words"
    ),
    "fwe": RulerTask(
        "fwe", "freq_words_extraction", 50, "extract the three most frequent coded words"
    ),
    "qa_1": RulerTask(
        "qa_1", "qa", 32, "single-hop question answering over long documents"
    ),
    "qa_2": RulerTask(
        "qa_2", "qa", 32, "multi-hop question answering over long documents"
    ),
}

DEFAULT_TASKS: Final[tuple[str, ...]] = (
    "niah_single_1",
    "niah_multikey_1",
    "niah_multiquery",
    "vt",
    "fwe",
    "qa_1",
)


def require_task(name: str) -> RulerTask:
    try:
        return TASKS[name]
    except KeyError as exc:
        options = ", ".join(TASKS)
        raise ValueError(f"Unknown RULER task {name!r}. Available tasks: {options}") from exc


def validate_tasks(names: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    if not names:
        raise ValueError("At least one RULER task must be selected.")
    cleaned: list[str] = []
    for raw in names:
        name = raw.strip()
        require_task(name)
        if name not in cleaned:
            cleaned.append(name)
    return tuple(cleaned)
