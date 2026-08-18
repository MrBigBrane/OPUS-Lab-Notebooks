#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

RUN_ROOT="${1:-runs/local-smoke}"
SMOKE_CONFIG="${SMOKE_CONFIG:-configs/smoke_all_backends_8k.yaml}"

printf '\n[1/3] Environment check\n'
santapp-ruler doctor

printf '\n[2/3] Data and token-budget validation\n'
santapp-ruler validate-data \
  --config "$SMOKE_CONFIG"

printf '\n[3/3] One prompt through all six standard backends\n'
santapp-ruler run \
  --config "$SMOKE_CONFIG" \
  --run-dir "$RUN_ROOT/all-backends"

printf '\nSmoke run complete.\n'
printf '  Summary: %s\n' "$RUN_ROOT/all-backends/summary.md"
