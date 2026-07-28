import json
from pathlib import Path

from santapp_ruler.config import load_config
from santapp_ruler.data import RulerExample, select_examples


def _write_task(root: Path, task: str, count: int = 6) -> None:
    path = root / task / "validation.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for index in range(count):
            handle.write(
                json.dumps(
                    {
                        "index": index,
                        "input": f"prompt {task} {index}",
                        "outputs": [f"answer {index}"],
                    }
                )
                + "\n"
            )


def test_local_selection_is_exact_and_deterministic(tmp_path: Path):
    _write_task(tmp_path, "vt")
    _write_task(tmp_path, "qa_1")
    overrides = [
        "benchmark.tasks=[vt, qa_1]",
        "benchmark.prompts_per_task=3",
        "benchmark.data.source=local",
        f"benchmark.data.local_root={str(tmp_path)!r}",
    ]
    config = load_config(None, overrides=overrides)
    first = select_examples(config.benchmark)
    second = select_examples(config.benchmark)
    assert [x.index for x in first["vt"]] == [x.index for x in second["vt"]]
    assert len(first["vt"]) == 3
    assert len(first["qa_1"]) == 3
    assert set(x.task for x in first["vt"]) == {"vt"}


def test_uid_includes_source_position_to_avoid_index_collisions():
    first = RulerExample("vt", 5, "a", ["x"], source_position=1)
    second = RulerExample("vt", 5, "b", ["y"], source_position=2)
    assert first.uid != second.uid
    assert first.uid == "vt:1:5"
