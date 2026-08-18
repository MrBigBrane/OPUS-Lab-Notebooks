import json
from pathlib import Path

import pytest

from santapp_ruler.io_utils import append_jsonl_fsync, repair_trailing_partial_jsonl
from santapp_ruler.reporting import read_jsonl


def test_trailing_partial_jsonl_is_repaired_without_touching_complete_rows(tmp_path: Path):
    path = tmp_path / "rows.jsonl"
    first = {"uid": "a", "value": 1}
    second = {"uid": "b", "value": 2}
    path.write_bytes(
        (json.dumps(first) + "\n" + json.dumps(second) + "\n" + '{"uid":"c"').encode()
    )
    assert repair_trailing_partial_jsonl(path) is True
    assert read_jsonl(path) == [first, second]
    assert path.read_bytes().endswith(b"\n")


def test_jsonl_repair_never_hides_middle_corruption(tmp_path: Path):
    path = tmp_path / "rows.jsonl"
    path.write_text('{"uid":"a"}\nnot-json\n{"uid":"b"', encoding="utf-8")
    assert repair_trailing_partial_jsonl(path) is True
    with pytest.raises(ValueError, match="Invalid JSON"):
        read_jsonl(path)


def test_valid_unterminated_jsonl_record_gets_newline_before_next_append(tmp_path: Path):
    path = tmp_path / "rows.jsonl"
    first = {"uid": "a", "value": 1}
    second = {"uid": "b", "value": 2}
    path.write_text(json.dumps(first), encoding="utf-8")

    assert repair_trailing_partial_jsonl(path) is True
    assert path.read_bytes().endswith(b"\n")
    append_jsonl_fsync(path, second)
    assert read_jsonl(path) == [first, second]
