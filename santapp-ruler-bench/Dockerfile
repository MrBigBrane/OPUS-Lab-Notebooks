# PyTorch's CUDA 12.8 binaries do not include sm_70. CUDA 12.6 is deliberate:
# it keeps PyTorch 2.11 while supporting both Volta V100 (sm_70) and A4000
# (sm_86), the small-GPU pool used by the default Kubernetes template.
ARG PYTORCH_IMAGE=pytorch/pytorch:2.11.0-cuda12.6-cudnn9-runtime
FROM ${PYTORCH_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
ARG APP_UID=10001
ARG APP_GID=10001

LABEL org.opencontainers.image.title="SANTA-family RULER harness" \
      org.opencontainers.image.description="Fault-tolerant Kubernetes RULER benchmark worker" \
      org.opencontainers.image.authors="SANTA-family RULER contributors"

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        tar \
        tini \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid "${APP_GID}" santapp \
    && useradd --uid "${APP_UID}" --gid "${APP_GID}" \
        --create-home --shell /bin/bash santapp

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_DISABLE_XET=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    HOME=/home/santapp

WORKDIR /workspace

# Install the application into the image. NRP policy forbids pip/conda installs
# on shared CephFS, so no Kubernetes pod performs package installation.
COPY pyproject.toml README.md LICENSE NOTICE MANIFEST.in ./
COPY src ./src
RUN python - <<'PY'
import torch
assert torch.__version__.split('+', 1)[0] == '2.11.0', torch.__version__
assert torch.version.cuda is not None and torch.version.cuda.startswith('12.6'), torch.version.cuda
print(f'base torch={torch.__version__} cuda={torch.version.cuda}')
PY
RUN python -m pip install --break-system-packages --upgrade pip wheel "setuptools<82" \
    && python -m pip uninstall --break-system-packages -y spin \
    && python -m pip install --break-system-packages . \
    && python -m pip check

COPY configs ./configs
COPY k8s ./k8s
COPY scripts ./scripts
COPY docs ./docs

RUN chmod +x scripts/*.py scripts/*.sh \
    && chown -R "${APP_UID}:${APP_GID}" /workspace /home/santapp

USER ${APP_UID}:${APP_GID}

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["santapp-ruler", "doctor"]
