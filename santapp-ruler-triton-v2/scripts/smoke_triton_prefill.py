#!/usr/bin/env python3
"""Self-contained R005 environment/compiler/runtime/model smoke with handoff logs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import traceback
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from santapp_ruler.runtime_defaults import configure_allocator

configure_allocator()  # Before workers/tests import torch; also inherited by children.


def write(path, value):
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def worker(path):
    c = json.loads(Path(path).read_text())
    out = Path(c["output_directory"])
    from santapp_ruler.attention.prefill_runtime import check_triton_package, require_triton_environment
    result = {"status": "running", "mode": c["mode"]}
    try:
        env = check_triton_package()
        write(out / "environment.json", env)
        if c["mode"] in ("compile", "all"):
            from santapp_ruler.attention.triton_prefill.compile_check import compile_portability
            result["compile"] = compile_portability(out / "compiler", c["targets"], c["contexts"])
        if c["mode"] in ("quick", "matrix", "model", "all"):
            if os.environ.get("TRITON_INTERPRET") == "1":
                raise RuntimeError("Unset TRITON_INTERPRET: this smoke must execute compiled GPU kernels")
            env = require_triton_environment()
            write(out / "environment.json", env)
            from santapp_ruler.attention.triton_prefill.smoke import quick_smoke, matrix_smoke, model_smoke
            result["quick"] = quick_smoke()
            write(out / "quick.json", result["quick"])
            if c["mode"] in ("matrix", "all"):
                result["matrix"] = matrix_smoke(
                    c["contexts"], c["sample_budgets"], c["dtypes"], c["repeats"], c["compare_torch"],
                    lambda report: write(out / "matrix.partial.json", report))
                write(out / "matrix.json", result["matrix"])
            if c["mode"] == "model":
                result["model"] = model_smoke(
                    Path(c["config"]), c["contexts"], c["sample_budgets"],
                    max_new_tokens=c["max_new_tokens"], policies=c["policies"],
                    report_callback=lambda report: write(out / "model.partial.json", report))
                write(out / "model.json", result["model"])
        result["status"] = "passed"
        return 0
    except Exception as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        return 1
    finally:
        write(out / "summary.json", result)


def run(command, log, env=None):
    print("RUN " + " ".join(map(str, command)), flush=True)
    with log.open("w") as handle:
        proc = subprocess.Popen(command, cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, bufsize=1)
        for line in proc.stdout:
            print(line, end="", flush=True)
            handle.write(line)
            handle.flush()
        status = proc.wait()
    if status:
        raise RuntimeError(f"Stage exited {status}; see {log.name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["packages", "compile", "quick", "matrix", "model", "all"], default="quick")
    parser.add_argument("--contexts", nargs="+", type=int, default=[8192, 32768])
    parser.add_argument("--sample-budgets", nargs="+", type=int)
    parser.add_argument("--dtypes", nargs="+", choices=["float16", "bfloat16", "float32"], default=["float16", "bfloat16"])
    parser.add_argument("--targets", nargs="+", type=int, choices=[80, 86], default=[80, 86])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--compare-torch", action="store_true", help="Extra single-head reference timings; never replay/correct native output")
    parser.add_argument("--run-tests", action="store_true", help="Run CPU tests and (for runtime modes) GPU regression tests before the worker")
    parser.add_argument("--max-new-tokens", type=int, default=2)
    parser.add_argument("--policies", nargs="+", choices=["santapp", "hierarchical"], default=["santapp", "hierarchical"])
    parser.add_argument("--config", type=Path, default=ROOT / "configs/default.yaml")
    parser.add_argument("--output-root", type=Path, default=ROOT / "runs/triton-prefill-smoke")
    parser.add_argument("--worker-json", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_json:
        return worker(args.worker_json)
    from santapp_ruler.attention.triton_prefill.smoke import DEFAULT_BUDGETS
    budgets = args.sample_budgets or ([128, 512] if args.mode == "model" else DEFAULT_BUDGETS)
    if any(s <= 0 or s % 4 for s in budgets):
        parser.error("P=16/R=4 smoke budgets must be positive multiples of four")
    if any(n < 16 for n in args.contexts) or args.repeats < 0 or args.max_new_tokens < 1:
        parser.error("contexts must be >=16, repeats nonnegative, output token count positive")
    root = args.output_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix="r005-", dir=root))
    config = {**vars(args), "sample_budgets": budgets, "output_directory": str(out),
              "config": str(args.config.expanduser().resolve()),
              "created_at_utc": datetime.now(timezone.utc).isoformat()}
    write(out / "config.json", config)
    # Capture only project sources/configuration, never tokens, environment secrets or model caches.
    manifest = {}
    for path in (ROOT / "src").rglob("*.py"):
        manifest[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    write(out / "source_hashes.json", manifest)
    shutil.copyfile(ROOT / "docs/TRITON_PREFILL_PROVENANCE.json", out / "provenance.json")
    for filename in ("Dockerfile", "BUILD_ENVIRONMENT.txt"):
        if (ROOT / filename).is_file():
            shutil.copyfile(ROOT / filename, out / filename)
    from santapp_ruler.attention.prefill_runtime import package_environment
    write(out / "package_environment.json", package_environment())
    status = 0
    try:
        run([sys.executable, "-m", "pip", "check"], out / "pip_check.log")
        if args.run_tests:
            env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), SANTAPP_RUN_PREFILL_GPU_TESTS="0")
            # Repository-hygiene tests intentionally forbid run artifacts and are
            # for clean source checkouts, not a running smoke's output directory.
            test_files = sorted(str(p.relative_to(ROOT)) for p in (ROOT / "tests").glob("test_*.py")
                                if p.name not in {"test_repository_hygiene.py", "test_triton_prefill_gpu.py"})
            run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *test_files], out / "cpu_tests.log", env)
            if args.mode in ("quick", "matrix", "model", "all"):
                env["SANTAPP_RUN_PREFILL_GPU_TESTS"] = "1"
                run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/test_triton_prefill_gpu.py"], out / "gpu_tests.log", env)
        run([sys.executable, str(Path(__file__).resolve()), "--worker-json", str(out / "config.json")],
            out / "smoke.log")
    except Exception as exc:
        status = 1
        print(f"FAIL {type(exc).__name__}: {exc}", flush=True)
        write(out / "failure.json", {"error": str(exc), "status": "failed"})
    finally:
        archive = out.with_name(out.name + "-handoff.zip")
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
            for path in out.rglob("*"):
                if path.is_file():
                    z.write(path, str(path.relative_to(out)))
        print(("UPLOAD_ARCHIVE=" if status == 0 else "FAILURE_ARCHIVE=") + str(archive))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
