#!/usr/bin/env bash
# Build/run the actual image; do not mount a second checkout or reuse host binaries.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
usage() {
  cat <<'EOF'
Usage:
  bash scripts/docker_triton_smoke.sh build [--profile cluster|local-5090] [--image TAG]
  bash scripts/docker_triton_smoke.sh run [--profile cluster|local-5090] [--image TAG] -- [SMOKE FLAGS]
Options before --: --artifacts DIR, --hf-cache DIR, --base-image IMAGE
Default cluster profile retains the repository's CUDA 12.6 base.
local-5090 replaces only the matched Torch/vision/audio CUDA wheels with CUDA 12.8.
SMOKE FLAGS go to smoke_triton_prefill.py; use --mode compile without a GPU,
--mode quick --run-tests, --mode matrix, or --mode model. No image is pushed.
EOF
}
[[ $# -gt 0 ]] || { usage; exit 2; }
action="$1"; shift
[[ "$action" != --help && "$action" != -h ]] || { usage; exit 0; }
[[ "$action" == build || "$action" == run ]] || { usage >&2; exit 2; }
profile=cluster
image=""
artifacts="${SANTAPP_SMOKE_ARTIFACTS:-$ROOT/.local-smoke-artifacts}"
hf_cache="${SANTAPP_HF_CACHE:-$HOME/.cache/santapp-ruler-hf}"
base_image=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile|--image|--artifacts|--hf-cache|--base-image)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 2; }
      case "$1" in
        --profile) profile="$2";; --image) image="$2";; --artifacts) artifacts="$2";;
        --hf-cache) hf_cache="$2";; --base-image) base_image="$2";;
      esac
      shift 2;;
    --) shift; break;;
    --help|-h) usage; exit 0;;
    *) echo "Unknown helper flag $1; put smoke flags after --" >&2; exit 2;;
  esac
done
case "$profile" in
  cluster) suffix=ampere; local_cuda=0;;
  local-5090) suffix=local-cu128; local_cuda=1;;
  *) echo "Unknown profile: $profile" >&2; exit 2;;
esac
image="${image:-santapp-ruler:r005-$suffix}"
command -v docker >/dev/null || { echo 'Docker CLI not found. Configure Docker Engine or Docker Desktop WSL2 integration first.' >&2; exit 1; }
mkdir -p "$artifacts"
artifacts="$(cd "$artifacts" && pwd)"
stamp="$(date -u +%Y%m%dT%H%M%SZ)-$$"
if [[ "$action" == build ]]; then
  [[ $# -eq 0 ]] || { echo "build does not accept smoke flags" >&2; exit 2; }
  args=(build --progress=plain --file "$ROOT/Dockerfile" --tag "$image" --build-arg "LOCAL_CUDA128=$local_cuda")
  [[ -z "$base_image" ]] || args+=(--build-arg "PYTORCH_IMAGE=$base_image")
  # pipefail preserves a build failure; retain complete compiler/package errors.
  docker "${args[@]}" "$ROOT" 2>&1 | tee "$artifacts/build-$suffix-$stamp.log"
  docker image inspect "$image" > "$artifacts/image-$suffix-$stamp.json"
  echo "BUILT_IMAGE=$image"
  exit 0
fi
[[ -z "$base_image" ]] || { echo "--base-image is only a build option" >&2; exit 2; }
mode=quick
prev=""
for arg in "$@"; do
  [[ "$prev" != --mode ]] || mode="$arg"
  case "$arg" in --mode=*) mode="${arg#--mode=}";; esac
  prev="$arg"
done
mkdir -p "$hf_cache"
hf_cache="$(cd "$hf_cache" && pwd)"
# Do not mount source: imports and compilation must use the built image's code.
# Use the caller's UID/GID for local bind mounts only; Kubernetes keeps 10001.
args=(run --rm --user "$(id -u):$(id -g)" --shm-size=1g
  --env HOME=/tmp/santapp-local-home --env HF_HOME=/hf
  --env TRITON_CACHE_DIR=/tmp/santapp-triton --env CUDA_CACHE_PATH=/tmp/santapp-cuda
  --mount "type=bind,src=$artifacts,dst=/artifacts"
  --mount "type=bind,src=$hf_cache,dst=/hf")
case "$mode" in packages|compile) ;; *) args+=(--gpus all);; esac
docker image inspect "$image" > "$artifacts/image-$suffix-$stamp.json"
docker "${args[@]}" "$image" python /workspace/scripts/smoke_triton_prefill.py \
  --output-root /artifacts "$@" 2>&1 | tee "$artifacts/run-$suffix-$stamp.log"
