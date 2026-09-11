"""Observed portability limitations, not a diagnosis of compiler root cause."""
from __future__ import annotations


def is_known_a10_d7_assignment_case(gpu_name: str, capability: tuple[int, int],
                                    triton_version: str, dimensions: int) -> bool:
    return (gpu_name.strip() in {"NVIDIA A10", "NVIDIA-A10"}
            and tuple(capability) == (8, 6)
            and triton_version.split("+")[0] == "3.6.0"
            and dimensions == 7)
