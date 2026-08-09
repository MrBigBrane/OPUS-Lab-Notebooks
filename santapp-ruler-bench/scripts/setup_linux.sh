#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'TXT'
Usage: scripts/setup_linux.sh [options]

Create or update a Linux/WSL virtual environment and install the benchmark.

Options:
  --dev                 Install test/build tooling as well as runtime packages.
  --skip-torch          Keep the PyTorch already available in the environment.
  --skip-doctor         Do not run the CUDA diagnostic after installation.
  --python PATH         Python 3.11 or 3.12 interpreter to use.
  --venv PATH           Virtual-environment path (default: .venv).
  -h, --help            Show this help.

Environment:
  PYTHON_BIN            Alternative to --python.
  TORCH_REQUIREMENTS    PyTorch requirement file
                        (default: requirements-torch-cu128.txt).
TXT
}

DEV=0
SKIP_TORCH=0
SKIP_DOCTOR=0
VENV_PATH=".venv"
PYTHON_CHOICE="${PYTHON_BIN:-}"
TORCH_REQUIREMENTS="${TORCH_REQUIREMENTS:-requirements-torch-cu128.txt}"

while (($#)); do
  case "$1" in
    --dev)
      DEV=1
      ;;
    --skip-torch)
      SKIP_TORCH=1
      ;;
    --skip-doctor)
      SKIP_DOCTOR=1
      ;;
    --python)
      shift
      [[ $# -gt 0 ]] || { echo "--python requires a path" >&2; exit 2; }
      PYTHON_CHOICE="$1"
      ;;
    --venv)
      shift
      [[ $# -gt 0 ]] || { echo "--venv requires a path" >&2; exit 2; }
      VENV_PATH="$1"
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ -z "$PYTHON_CHOICE" ]]; then
  for candidate in python3.12 python3.11 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
      if "$candidate" - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(not ((3, 11) <= sys.version_info[:2] < (3, 13)))
PY
      then
        PYTHON_CHOICE="$candidate"
        break
      fi
    fi
  done
fi

if [[ -z "$PYTHON_CHOICE" ]]; then
  echo "Python 3.11 or 3.12 was not found. Set PYTHON_BIN or pass --python." >&2
  exit 1
fi

"$PYTHON_CHOICE" - <<'PY'
import sys
if not ((3, 11) <= sys.version_info[:2] < (3, 13)):
    raise SystemExit(
        f"Expected Python 3.11 or 3.12, found {sys.version.split()[0]}"
    )
PY

if [[ ! -x "$VENV_PATH/bin/python" ]]; then
  "$PYTHON_CHOICE" -m venv "$VENV_PATH"
fi

"$VENV_PATH/bin/python" - <<'PY'
import sys
if not ((3, 11) <= sys.version_info[:2] < (3, 13)):
    raise SystemExit(
        "The existing virtual environment uses unsupported Python "
        f"{sys.version.split()[0]}. Remove it or choose another --venv path."
    )
PY

# shellcheck disable=SC1090
source "$VENV_PATH/bin/activate"

python -m pip install --upgrade pip setuptools wheel
if (( ! SKIP_TORCH )); then
  if [[ ! -f "$TORCH_REQUIREMENTS" ]]; then
    echo "PyTorch requirement file not found: $TORCH_REQUIREMENTS" >&2
    exit 1
  fi
  python -m pip install -r "$TORCH_REQUIREMENTS"
fi

if (( DEV )); then
  python -m pip install -e ".[dev]"
else
  python -m pip install -e .
fi

if (( ! SKIP_DOCTOR )); then
  python -m santapp_ruler doctor
fi

cat <<TXT
Environment ready.
Activate it later with:
  source "$VENV_PATH/bin/activate"
TXT
