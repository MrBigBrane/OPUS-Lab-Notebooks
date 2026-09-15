"""Offline Ampere compiler checks; no CUDA device/driver or tensors required.

This is a compiler portability check, NOT an emulation, runtime correctness
check, or speed measurement. Real Ampere execution is still required.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import inspect
import json
import math
from pathlib import Path
import re


@dataclass
class CompileCase:
    module: str
    kernel: str
    signature: dict[str, str]
    constants: dict
    tag: str


def _pow2(x):
    return 1 << (int(x) - 1).bit_length()


def compile_plan(contexts=(8192, 32768), dimensions=128) -> list[CompileCase]:
    """Mirror P001 launch types/constants, including init, update and final fit."""
    cases = []
    def add(kernel, pointers, constants, tag, runtime=None, module="kmeans"):
        signature = {name: "*fp32" for name in pointers.split()}
        signature.update(runtime or {})
        # Override integer/boolean buffers. Ids' width depends on the operation.
        widths = {"Labels": "*i64", "CandidateIds": "*i64", "RowIds": "*i64",
                  "BatchCounts": "*i32", "Ranks": "*i32", "Mask": "*i1",
                  "Uniforms": "*fp64", "ParentMembers": "*i64", "Indptr": "*i64",
                  "ParentIds": "*i32", "TeamOffsets": "*i64", "Scratch": "*i32",
                  "Members": "*i64", "Starts": "*i64", "Lengths": "*i64",
                  "LeaderIds": "*i64", "TeamParents": "*i64"}
        if "Ids" in signature:
            widths["Ids"] = "*i64" if kernel == "gather_rows" else "*i32"
        signature.update({k: v for k, v in widths.items() if k in signature})
        cases.append(CompileCase(module, kernel, signature, constants, tag))

    d = dimensions
    if not 1 <= d <= 256:
        raise ValueError("dimensions must be 1..256")
    bd = _pow2(max(32, d))
    for total in contexts:
        k = min(total, max(2, total // 16))
        init = min(total, max(3 * 4096, 3 * k))
        trials = 2 + int(math.log(k))
        n = init
        nb, ns = (n + 63) // 64, (n + 511) // 512
        tag = f"N{total}_D{d}"
        add("gather_rows", "X Ids Y", dict(M=min(total, 4096), D=d, B=256), tag)
        add("first_center", "X Centers Closest", dict(N=n, D=d, BN=64, BD=bd), tag,
            {"first_id": "i32"})
        add("scan_closest", "Closest Prefix Totals", dict(N=n, B=512), tag)
        add("sample_candidates", "Prefix Totals Potential Uniforms CandidateIds",
            dict(N=n, NB=ns, SB=512, TB=_pow2(ns), TRIALS=trials), tag,
            {"center_index": "i32"})
        add("trial_distances", "X Closest CandidateIds Distances Partials",
            dict(N=n, D=d, TRIALS=trials, BN=64, BT=_pow2(max(16, trials)), BD=bd,
                 PRECISION="ieee"), tag)
        add("choose_and_commit", "X Centers Closest CandidateIds Distances Partials Potential",
            dict(N=n, D=d, TRIALS=trials, NB=nb, BP=_pow2(nb), BT=_pow2(trials),
                 BN=64, BD=bd), tag, {"center_index": "i32"})
        nc = (k + 31) // 32
        for n in sorted({min(total, 4096), total}):
            atag = tag + f"_assign{n}"
            add("assignment_partials", "X Centers Minima Ids",
                dict(N=n, K=k, D=d, NC=nc, BM=32, BC=32, BD=bd, PRECISION="ieee"), atag)
            add("finish_assignment", "Minima Ids Labels InertiaParts",
                dict(N=n, NC=nc, BG=_pow2(nc), BM=64), atag)
            parts = (n + 63) // 64
            add("sum_scalar", "X Out", dict(N=parts, B=_pow2(parts)), atag)
        add("zero_update", "Sums BatchCounts", dict(K=k, D=d, B=256), tag)
        add("accumulate_batch", "X Labels Sums BatchCounts",
            dict(N=min(total, 4096), D=d, BM=16, BD=_pow2(d)), tag)
        add("update_centers", "Centers Counts Sums BatchCounts NewCenters NewCounts",
            dict(K=k, D=d, BC=16, BD=_pow2(d)), tag)
        add("reassignment_plan", "Counts Mask Ranks Floor", dict(K=k, B=_pow2(k)), tag)
        add("apply_reassignment", "Centers Counts XBatch Mask Ranks RowIds Floor",
            dict(K=k, D=d, BC=16, BD=_pow2(d)), tag)
    common = "Keys ParentMembers Indptr ParentIds TeamOffsets"
    output = "Members Starts Lengths LeaderIds LeaderKeys TeamParents"
    for full in (0, 16):
        for tile in (1, 2, 4, 8, 16, 32, 64, 128):
            add("small_parent_teams", common + " " + output,
                dict(D=d, R=4, BN=tile, BD=_pow2(d), DIRECT=False, FULL_SPAN=full),
                f"D{d}_B{tile}_span{full}", module="teams")
        add("streaming_parent_teams", common + " Scratch " + output,
            dict(D=d, R=4, BR=4, BN=64, BD=_pow2(d), DIRECT=False, FULL_SPAN=full),
            f"D{d}_stream_span{full}", module="teams")
    return cases


def compile_portability(output: Path, targets=(80, 86), contexts=(8192, 32768),
                        dimensions=128) -> dict:
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    from .kernels import p001_prefill_kmeans, p001_prefill_teams
    output.mkdir(parents=True, exist_ok=True)
    report = {"status": "running", "triton": triton.__version__, "execution": False,
              "targets": list(targets), "cases": []}
    modules = {"kmeans": p001_prefill_kmeans, "teams": p001_prefill_teams}
    try:
        for target in targets:
            if target not in (80, 86):
                raise ValueError("Offline Ampere targets are 80 and 86; use a GPU quick/matrix smoke for the actual local device")
            for case in compile_plan(contexts, dimensions):
                fn = getattr(modules[case.module], case.kernel)
                assert set(fn.arg_names) == set(case.signature) | set(case.constants)
                kwargs = {"signature": case.signature}
                # Use the installed compiler's named constructor, not positional ABI guesses.
                if "constexprs" in inspect.signature(ASTSource).parameters:
                    kwargs["constexprs"] = case.constants
                else:
                    kwargs["constants"] = case.constants
                compiled = triton.compile(ASTSource(fn, **kwargs), target=GPUTarget("cuda", target, 32),
                                          options={"num_warps": 4, "num_ctas": 1,
                                                   "enable_fp_fusion": False})
                ptx = compiled.asm["ptx"]
                if re.search(r"wgmma\.|tcgen05\.|cp\.async\.bulk\.tensor", ptx):
                    raise AssertionError("A non-Ampere instruction appeared in a portable prefill kernel")
                if not re.search(rf"\.target\s+sm_{target}\b", ptx):
                    raise AssertionError(f"Unexpected PTX target for requested sm_{target}")
                shared = int(compiled.metadata.shared)
                # Conservative common budget for sm80/sm86; these kernels are not
                # intended to need the larger sm80-specific shared-memory limit.
                if target in (80, 86) and shared > 99 * 1024:
                    raise AssertionError(f"{case.kernel} uses {shared} shared bytes; exceeds the sm86 budget")
                row = {"target": target, "kernel": case.kernel, "case": case.tag,
                       "shared_bytes": shared, "ptx_sha256": hashlib.sha256(ptx.encode()).hexdigest(),
                       "cubin_generated": "cubin" in compiled.asm}
                report["cases"].append(row)
                (output / f"sm{target}_{case.kernel}_{case.tag}.ptx").write_text(ptx)
                print(f"COMPILE PASS sm{target} {case.kernel} {case.tag} shared={shared}", flush=True)
        report["status"] = "passed"
        return report
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        (output / "compile_report.json").write_text(json.dumps(report, indent=2) + "\n")
