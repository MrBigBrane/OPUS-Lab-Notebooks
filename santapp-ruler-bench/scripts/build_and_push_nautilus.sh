#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-your-registry/santapp-ruler:latest}"
PLATFORM="${PLATFORM:-linux/amd64}"
PUSH="${PUSH:-1}"
PYTORCH_IMAGE="${PYTORCH_IMAGE:-pytorch/pytorch:2.11.0-cuda12.6-cudnn9-runtime}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

COMMON_ARGS=(
  --platform "${PLATFORM}"
  --tag "${IMAGE}"
  --build-arg "PYTORCH_IMAGE=${PYTORCH_IMAGE}"
  --pull
)

if [[ "${PUSH}" == "1" ]]; then
  echo "[image] building and pushing ${IMAGE} for ${PLATFORM}"
  docker buildx build "${COMMON_ARGS[@]}" --push .
else
  echo "[image] building ${IMAGE} locally for ${PLATFORM}"
  docker buildx build "${COMMON_ARGS[@]}" --load .
fi

echo "[image] base: ${PYTORCH_IMAGE}"
echo "[image] complete: ${IMAGE}"
