# Validation status for the Linux/WSL handoff

Validation date: **August 6, 2026** (America/Los_Angeles)

## Completed in this build environment

The following checks passed against the packaged source:

| Check | Result |
|---|---|
| Python compilation of `src/`, `scripts/`, and `tests/` | Pass |
| Unit and integration suite | **27 passed** |
| Three-method × 13-task runner matrix with local synthetic rows and mocked model backends | **39/39 prediction rows written and reported** |
| Tiny Qwen-shaped end-to-end attention integration on CPU | Pass for SANTA and SANTA++ |
| SANTA estimator test | Pass: exact-distribution samples reduce to the arithmetic mean of gathered values |
| SANTA++ probe policy | Pass: exact final `P` prompt positions |
| GQA-aware and naive access formulas | Pass for SANTA and SANTA++ reference cases |
| Configuration loading | Pass for default, simple 40-prompt, quick smoke, 39-row handoff smoke, and full 13×500 profiles |
| Linux setup-script syntax and help path | Pass |
| CLI task registry | Pass: all 13 tasks and their configured generation limits listed |
| Python wheel build without dependency download | Pass |

Commands used for the principal local checks:

```bash
PYTHONPATH=src pytest -q
python -m compileall -q src scripts tests
bash -n scripts/setup_linux.sh
PYTHONPATH=src python -m santapp_ruler list-tasks
python -m pip wheel --no-build-isolation --no-deps --ignore-requires-python .
```

The CPU attention integration exercises the shared cache/patch path rather than only isolated formulas: dense prefill, SANTA full-score sampling, SANTA++ clustering of every prompt token, last-token probes, a zero-length exact region on the first sparse call, the generated-token-only growing exact suffix, generation, and access reporting all execute together on a small Qwen-compatible test module.

## GPU/model smoke not executable in this session

This build container has no `nvidia-smi`, reports `torch.cuda.is_available() == False`, lacks the pinned Transformers and Datasets packages, and has no network route from which to obtain the pinned model or RULER data. Consequently, the requested real Qwen2.5-3B 8k smoke—three methods × 13 tasks × one prompt—was **not** executed here. The repository should not be described as target-GPU validated until that command passes on a CUDA machine.

Run these checks on the target WSL/Linux system before distributing benchmark results:

```bash
./scripts/setup_linux.sh --dev
source .venv/bin/activate

santapp-ruler doctor
santapp-ruler validate-data --config configs/smoke_all_tasks_8k.yaml
santapp-ruler fidelity --config configs/default_8k.yaml --tokens 16
santapp-ruler run \
  --config configs/smoke_all_tasks_8k.yaml \
  --run-dir runs/smoke-all-tasks-8k

pytest -q
ruff check src tests scripts
python -m build
```

The expected benchmark smoke output is 39 completed prediction records: 13 each for `sdpa`, `santa`, and `santapp`. The run is resumable, so an interruption can be followed by the same command.
