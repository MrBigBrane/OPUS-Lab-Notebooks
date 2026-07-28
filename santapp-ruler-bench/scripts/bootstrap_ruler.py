"""Clone and pin NVIDIA/RULER for optional local data generation."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

RULER_REPOSITORY = "https://github.com/NVIDIA/RULER.git"
RULER_COMMIT = "38da79d79519ef87aa46ae804f838e1eab7f86d7"


def run(*args: str, cwd: Path | None = None) -> None:
    print("+", " ".join(args))
    subprocess.run(args, cwd=cwd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--destination", type=Path, default=Path("third_party/RULER")
    )
    args = parser.parse_args()
    destination = args.destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)

    if not destination.exists():
        run("git", "clone", RULER_REPOSITORY, str(destination))
    run("git", "fetch", "--all", "--tags", cwd=destination)
    run("git", "checkout", "--detach", RULER_COMMIT, cwd=destination)
    if shutil.which("git-lfs"):
        try:
            run("git", "lfs", "pull", cwd=destination)
        except subprocess.CalledProcessError:
            print(
                "WARNING: `git lfs pull` failed. An upstream synthetic source "
                "file may remain a Git LFS pointer."
            )
    else:
        print(
            "Git LFS is not installed; skipping `git lfs pull`. Install it only "
            "if an upstream synthetic source file is checked out as a pointer."
        )
    print(f"Pinned RULER checkout: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
