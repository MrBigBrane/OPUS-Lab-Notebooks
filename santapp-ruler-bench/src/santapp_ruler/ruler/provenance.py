"""Pinned upstream references used by this harness."""

from typing import Final

RULER_REPOSITORY: Final = "https://github.com/NVIDIA/RULER"
RULER_COMMIT: Final = "38da79d79519ef87aa46ae804f838e1eab7f86d7"
RULER_LICENSE: Final = "Apache-2.0"

DEFAULT_MODEL_REPOSITORY: Final = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_MODEL_REVISION: Final = (
    "aa8e72537993ba99e69dfaafa59ed015b17504d1"
)

# Convenience 8k data mirror generated specifically with the Qwen2.5-3B
# tokenizer. The harness also accepts locally generated official RULER JSONL.
DEFAULT_DATASET_REPOSITORY: Final = (
    "SaylorTwift/RULER-8192-Qwen2.5-3B-tokenizer"
)
DEFAULT_DATASET_REVISION: Final = (
    "6ee2d0f4e9b8983361da35204ead8931c3f65ad4"
)
