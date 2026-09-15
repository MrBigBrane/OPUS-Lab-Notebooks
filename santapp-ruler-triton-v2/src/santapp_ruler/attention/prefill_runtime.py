"""Explicit environment checks for optional, native Triton prefill acceleration.

Torch-only runs never import Triton. Requests for Triton fail loudly rather
than silently reverting to the slow reference on an unsupported GPU/image.
"""
from __future__ import annotations

import importlib.metadata as metadata
import os
import platform
import shutil
import sys
import sysconfig
from functools import lru_cache
from pathlib import Path


def package_environment() -> dict:
    versions = {}
    for name in ("torch", "triton", "transformers", "numpy", "datasets", "santapp-ruler"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    try:
        requirements = [x for x in metadata.requires("torch") or []
                        if x.lower().startswith("triton")]
    except metadata.PackageNotFoundError:
        requirements = []
    return {
        **versions, "python": sys.version, "platform": platform.platform(),
        "torch_triton_requirements": requirements,
        "cc": shutil.which(os.environ.get("CC", "gcc")),
        "python_include": sysconfig.get_path("include"),
        "triton_cache_dir": os.environ.get("TRITON_CACHE_DIR"),
        "cuda_cache_path": os.environ.get("CUDA_CACHE_PATH"),
        "native_prefill_only": True, "reference_replay": False,
        "pytorch_alloc_conf": os.environ.get("PYTORCH_ALLOC_CONF"),
        "pytorch_cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        "validated_production_head_dim": 128,
        "known_limitations": ["A10/sm86, Triton 3.6.0: synthetic D=7 k-means assignment accuracy defect; see docs/KNOWN_ISSUES.md"],
    }


def check_triton_package() -> dict:
    """GPU-independent package/compiler check; also usable at Docker build time."""
    from packaging.requirements import Requirement
    from packaging.version import Version

    info = package_environment()
    if sys.platform != "linux":
        raise RuntimeError("Triton prefill requires Linux (WSL2 Linux is supported).")
    if not info["triton"]:
        raise RuntimeError("Triton is missing. Use the supplied Docker image or the Triton "
                           "dependency matched to your CUDA PyTorch; do not install an arbitrary latest version.")
    if Version(info["triton"]) < Version("3.6"):
        raise RuntimeError("This port targets the Torch 2.11-era Triton 3.6+ API. "
                           "Use the supplied image rather than upgrading Triton independently.")
    for raw in info["torch_triton_requirements"]:
        requirement = Requirement(raw)
        if requirement.marker is None or requirement.marker.evaluate():
            if Version(info["triton"]) not in requirement.specifier:
                raise RuntimeError(f"Installed Triton {info['triton']} violates PyTorch's "
                                   f"requirement {raw}. Restore the matched environment.")
    if not info["cc"]:
        raise RuntimeError("A C compiler is required for Triton's host launcher. "
                           "Use the updated image (build-essential is included).")
    if not (Path(info["python_include"]) / "Python.h").is_file():
        raise RuntimeError("Python.h is missing for the running Python; install its matching development headers.")
    # Merely importing these checks the actual compiler package, without a GPU.
    from .triton_prefill.kernels import p001_prefill_kmeans, p001_prefill_teams
    assert callable(p001_prefill_kmeans.gather_rows)
    assert callable(p001_prefill_teams.small_parent_teams)
    return info


def device_compatibility_error(capability: tuple[int, int], cuda_version: str | None) -> str | None:
    if capability < (8, 0):
        return "Native Triton prefill requires compute capability >= 8.0 (Ampere or newer). Use prefill_backend=torch on Turing/Volta."
    if cuda_version is None:
        return "The installed PyTorch is not a CUDA build."
    version = tuple(int(part) for part in cuda_version.split(".")[:2])
    if capability >= (10, 0) and version < (12, 8):
        return ("This CUDA 12.6 PyTorch image targets pre-Blackwell GPUs. For the local 5090 "
                "use the helper's --profile local-5090 (matched CUDA 12.8 PyTorch wheels); "
                "keep --profile cluster for Ampere deployment. A newer host driver alone "
                "does not replace the image's compiled PyTorch kernels.")
    return None


@lru_cache(maxsize=16)
def _require_on_device(index: int) -> dict:
    import torch
    info = check_triton_package()
    capability = torch.cuda.get_device_capability(index)
    error = device_compatibility_error(capability, torch.version.cuda)
    if error:
        raise RuntimeError(error)
    # Resolve writable defaults without changing or cleaning an existing cache.
    for variable, default in (("TRITON_CACHE_DIR", "~/.triton/cache"),
                              ("CUDA_CACHE_PATH", "~/.nv/ComputeCache")):
        path = Path(os.environ.get(variable, default)).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        import tempfile
        with tempfile.TemporaryFile(dir=path):
            pass
        info[variable.lower()] = str(path)
    info.update(gpu=torch.cuda.get_device_name(index),
                compute_capability=list(capability), torch_cuda=torch.version.cuda,
                torch_arch_list=torch.cuda.get_arch_list())
    return info


def require_triton_environment(device="cuda") -> dict:
    import torch
    if os.environ.get("TRITON_INTERPRET") == "1":
        raise RuntimeError("Unset TRITON_INTERPRET: this backend requires compiled GPU kernels.")
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Triton prefill was explicitly selected but CUDA is unavailable. "
                           "No PyTorch fallback was performed. Use Docker --gpus all or a GPU pod.")
    index = torch.cuda.current_device() if device.index is None else device.index
    return _require_on_device(index)
