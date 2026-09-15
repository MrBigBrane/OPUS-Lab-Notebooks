from collections import UserDict
from types import SimpleNamespace

import torch
import pytest

from santapp_ruler.backends import encode_prompt
from santapp_ruler.config import BenchmarkConfig, DataConfig
from santapp_ruler.data import RulerExample, _normalize_row, select_examples


class RawTokenizer:
    def __init__(self) -> None:
        self.calls = []

    def __call__(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        return SimpleNamespace(input_ids=torch.tensor([[1, 2, 3]]))


class ChatBatchEncodingTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        del messages, kwargs
        return UserDict({"input_ids": torch.tensor([[8, 9, 10]])})


class ChatTokenizer:
    def __init__(self) -> None:
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return torch.tensor([[4, 5, 6, 7]])


def test_raw_prompt_adds_no_special_tokens() -> None:
    tokenizer = RawTokenizer()
    encoded = encode_prompt(tokenizer, "official ruler text", "raw")
    assert encoded.tolist() == [[1, 2, 3]]
    assert tokenizer.calls == [
        (
            "official ruler text",
            {"add_special_tokens": False, "return_tensors": "pt"},
        )
    ]


def test_chat_template_accepts_mapping_batch_encoding() -> None:
    tokenizer = ChatBatchEncodingTokenizer()
    encoded = encode_prompt(tokenizer, "official ruler text", "chat_template")
    assert encoded.tolist() == [[8, 9, 10]]


def test_chat_template_is_explicit_single_user_message() -> None:
    tokenizer = ChatTokenizer()
    encoded = encode_prompt(tokenizer, "official ruler text", "chat_template")
    assert encoded.shape == (1, 4)
    messages, kwargs = tokenizer.calls[0]
    assert messages == [{"role": "user", "content": "official ruler text"}]
    assert kwargs["add_generation_prompt"] is True


def test_unknown_prompt_format_is_rejected() -> None:
    with pytest.raises(ValueError):
        encode_prompt(RawTokenizer(), "x", "unknown")


def test_normalize_row_preserves_prompt_and_answer_prefix() -> None:
    row = {"input": "question", "answer_prefix": " Answer:", "outputs": ["x"], "index": 7}
    example = _normalize_row("qa_1", row, 2)
    assert example.input == "question Answer:"
    assert example.outputs == ["x"]
    assert example.uid == "qa_1:2:7"


def test_selection_is_deterministic_per_task(monkeypatch) -> None:
    examples = [
        RulerExample("qa_1", i, f"p{i}", [str(i)], i) for i in range(10)
    ]

    def fake_load(config, task):
        assert task == "qa_1"
        return examples

    monkeypatch.setattr("santapp_ruler.data.load_task_examples", fake_load)
    cfg = BenchmarkConfig(
        context_length=8192,
        tasks=["qa_1"],
        prompts_per_task=3,
        selection_seed=42,
        data=DataConfig(source="local", local_root="unused"),
    )
    first = [x.index for x in select_examples(cfg)["qa_1"]]
    second = [x.index for x in select_examples(cfg)["qa_1"]]
    assert first == second
    assert len(set(first)) == 3
