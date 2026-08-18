#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if ! grep -qi microsoft /proc/version 2>/dev/null; then
  echo "WARNING: /proc/version does not identify WSL; continuing as ordinary Linux." >&2
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  cat >&2 <<'EOF'
ERROR: nvidia-smi is unavailable inside WSL.
Install/update the Windows NVIDIA driver with WSL CUDA support, run `wsl --shutdown`
from PowerShell, reopen Ubuntu, and verify `nvidia-smi` before installing PyTorch.
EOF
  exit 1
fi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

if [[ -n "${PYTHON:-}" ]]; then
  PYTHON_BIN="$PYTHON"
else
  for candidate in python3.12 python3.11 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
      PYTHON_BIN="$candidate"
      break
    fi
  done
fi
if [[ -z "${PYTHON_BIN:-}" ]]; then
  echo "ERROR: Python 3.11 or 3.12 was not found." >&2
  exit 1
fi

"$PYTHON_BIN" - <<'PY'
import sys
if not ((3, 11) <= sys.version_info[:2] < (3, 13)):
    raise SystemExit(
        f"This pinned environment requires Python 3.11 or 3.12; found {sys.version.split()[0]}."
    )
print("Using Python", sys.version.split()[0])
PY

if [[ ! -d .venv ]]; then
  if ! "$PYTHON_BIN" -m venv .venv; then
    echo "Failed to create .venv. On Ubuntu, install python3-venv (or python3.12-venv)." >&2
    exit 1
  fi
fi
# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install --upgrade pip wheel "setuptools<82"
# Retain the repository's existing CUDA/PyTorch and Python dependency pins.
python -m pip install -r requirements-torch-cu128.txt
python -m pip install -r requirements-dev.txt

export HF_HUB_DISABLE_XET=1
python -m santapp_ruler doctor
python -m pytest -q
python -m ruff check src tests scripts

cat <<'EOF'

WSL environment is ready.

Activate it later with:
  source .venv/bin/activate

Run the recommended local handoff smoke with:
  bash scripts/run_local_smokes.sh

The smoke run is resumable; rerun the same command after an interruption.
Use another first argument to choose a different run root.
EOF
