"""Upstream references and default model/data identifiers."""

from typing import Final

RULER_REPOSITORY: Final = "https://github.com/NVIDIA/RULER"
RULER_COMMIT: Final = "38da79d79519ef87aa46ae804f838e1eab7f86d7"
RULER_LICENSE: Final = "Apache-2.0"

DEFAULT_MODEL_REPOSITORY: Final = "Qwen/Qwen2.5-7B-Instruct"
# Immutable Hugging Face commit used for both tokenizer and model loading.
DEFAULT_MODEL_REVISION: Final = "a09a35458c702b33eeacc393d103063234e8bc28"

# DATASET_REPOSITORIES: Final[dict[int, str]] = {
#     8192: "SaylorTwift/RULER-8192-Qwen2.5-3B-tokenizer",
#     32768: "SaylorTwift/RULER-32768-Qwen2.5-3B-tokenizer",
# }
DATASET_REPOSITORIES: Final[dict[int, str]] = {
    64000: "MrBigBrane/longbench-v2-short"
}
DEFAULT_DATASET_REVISION: Final = "main"


def default_dataset_repository(context_length: int) -> str:
    """Return the convenience RULER mirror for a supported target length."""

    try:
        return DATASET_REPOSITORIES[context_length]
    except KeyError as exc:
        supported = ", ".join(str(value) for value in sorted(DATASET_REPOSITORIES))
        raise ValueError(
            "No automatic Hugging Face RULER mirror is configured for "
            f"context_length={context_length}. Supported automatic lengths: "
            f"{supported}. Set benchmark.data.repository explicitly or use local "
            "official RULER JSONL."
        ) from exc
