# Changelog

## 1.3.0 / R005 — 2026-09-10

Native Triton preparation becomes the default for both existing team backends.
The original Torch implementation remains explicitly selectable, with no hidden
fallback. The four backend names, benchmark protocol and PyTorch decoder remain.

The A10 smoke findings are incorporated into runtime/Docker/Kubernetes defaults:
expandable segments, equal CPU/RAM requests and limits, init-container resources,
and writable pod-local JIT scratch. Default model-worker parallelism is four.
Resource quantities are tunable without violating the request/limit ratio.

The old failed A10 D=7 k-means assignment fixture is retained as an isolated,
platform/version-specific expected-failure diagnostic. Production D=128 primitive
and multiple-tile tests remain mandatory. The R004a small-D workaround is not
included. Numerical prefill kernels/host controller/team builder are R004-identical.

The manifest renderer retains its original four-resource mode and adds a staged
kernel/prefetch/model smoke mode. Configs, README and deployment notes are updated
for colleague handoff, with generic identifiers. Old implicit-Torch saved run
configs retain their historical identity to prevent mixed resumes.

Package version and image tag are advanced; build/push a new image. Predecessor
R004 GPU evidence is recorded separately from release-local CPU/package checks.
See `VALIDATION.md`, `docs/VALIDATION_HISTORY.md`, and `docs/KNOWN_ISSUES.md`.
