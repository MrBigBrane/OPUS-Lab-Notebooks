# R005 release validation

Package version: 1.3.0. Numerical base: R004, not the untested R004a small-D hotfix.

## Release-local checks

**196 CPU/static tests passed; 42 GPU tests explicitly skipped.**

Test environment: Python 3.13.5 and Torch 2.10.0+cpu. The deployment
package still targets Python 3.11/3.12; its source parses with Python 3.11
syntax rules. This is not a run in the production Docker dependency stack.

CPU checks cover default-native dispatch, explicit Torch selection, no hidden
Triton fallback, original controller equivalence, preserved source/decode hashes,
complete context/budget indexing, equal resource requests/limits in all generated
containers, allocator setup before torch imports, respect for user overrides,
safe resume of legacy implicit-Torch results, staged prefetch wiring, and narrow
D=7 expected-failure classification. GPU tests are opt-in and are not represented
as executed here. Docker helper argument tests use a CLI stub, not a live daemon.

Release packaging also checks wheel/source-distribution construction, Python
3.11 syntax, shell syntax, YAML parsing, regenerated manifest equality, and actual
patch application to clean original/R004 copies. See the supplied external
`R005_VALIDATION.txt` for the concrete checks performed for this delivery.

## Preserved behavior

R005 retains byte-identical R004 numerical kernel sources, numerical launcher,
mini-batch controller and native team builder. The two original reference
builders, original PyTorch k-means, qwen cache patch, both engine source files,
sampling, grader, task definitions, prompt/data handling and reports are retained
from R004. Selected decode methods are also hash-checked against the supplied
original colleague repository. The algorithm/code that passed the reported A10
model smoke is not replaced by the proposed R004a workaround.

Changes are defaults, executable allocator bootstrap, metadata/version, legacy
resume handling, deployment machinery, tests, and documentation. The full source
manifest is `FILE_MANIFEST.sha256`; no experiment outputs, weights, cached binaries,
private tokens or person-specific identifiers are shipped in the colleague repo.

## GPU evidence is predecessor evidence

On 2026-09-10, the maintainer reported 8/8 A10 R004 full-model smoke completions
with expandable segments, 8k/32k, S=128/512, both parent policies, one NIAH example
per case, and two output tokens. Earlier logs show local-5090 validation and an
A10 synthetic D=7 failure. See `docs/VALIDATION_HISTORY.md` for the exact boundaries.

This release does not claim a new R005 Docker/GPU/A10 run, general bitwise parity,
full RULER accuracy, arbitrary-dimension portability, or a guarantee of no OOM at
larger budgets/longer generation. Build a distinct image and rerun the included
acceptance workflow before handing it out as a tested new image.
