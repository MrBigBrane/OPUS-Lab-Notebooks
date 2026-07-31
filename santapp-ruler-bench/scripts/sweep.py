"""Run an explicit, sequential parameter grid.

The main benchmark command never invokes this script automatically.
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path

import yaml


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("grid", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    spec = yaml.safe_load(args.grid.read_text(encoding="utf-8"))
    base_config = Path(spec["base_config"])
    output_root = Path(spec["output_root"])
    grid = spec["grid"]
    keys = list(grid)
    values = [grid[key] for key in keys]

    for step_idx, combination in enumerate(itertools.product(*values)):
        mapping = dict(zip(keys, combination, strict=True))
        simple_names = {key.rsplit(".", 1)[-1]: value for key, value in mapping.items()}
        run_name = spec.get("name_template", "run").format(**simple_names)
        run_dir = output_root / run_name
        command = [
            sys.executable,
            "-m",
            "santapp_ruler",
            "run",
            "--config",
            str(base_config),
            "--run-dir",
            str(run_dir),
        ]

        # On the first step run both backends (sdpa + santapp); on step 2 onwards, run santapp only.
        if step_idx == 0:
            command.extend(["--set", 'generation.backends=["sdpa", "santapp"]'])
        else:
            command.extend(["--set", 'generation.backends=["santapp"]'])

        for key, value in mapping.items():
            command.extend(["--set", f"{key}={json.dumps(value)}"])

        print(" ".join(command))
        if not args.dry_run:
            subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())