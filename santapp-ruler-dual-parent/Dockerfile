ARG PYTORCH_IMAGE=pytorch/pytorch:2.11.0-cuda12.6-cudnn9-runtime
FROM ${PYTORCH_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
ARG APP_UID=10001
ARG APP_GID=10001

LABEL org.opencontainers.image.title="SANTA RULER benchmark" \
      org.opencontainers.image.description="SDPA, SANTA, and K-means or contiguous-parent SANTA++ attention" \
      org.opencontainers.image.authors="Research contributors"

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git tar tini \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid "${APP_GID}" benchmark \
    && useradd --uid "${APP_UID}" --gid "${APP_GID}" \
        --create-home --shell /bin/bash benchmark

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_DISABLE_XET=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    HOME=/home/benchmark

WORKDIR /workspace
COPY pyproject.toml README.md LICENSE NOTICE CITATION.cff ./
COPY src ./src
RUN python -m pip install --break-system-packages --upgrade pip wheel "setuptools<82" \
    && python -m pip uninstall --break-system-packages -y spin \
    && python -m pip install --break-system-packages . \
    && python -m pip check

COPY configs ./configs
COPY scripts ./scripts
COPY docs ./docs
COPY k8s ./k8s
RUN chmod +x scripts/*.py \
    && chown -R "${APP_UID}:${APP_GID}" /workspace /home/benchmark

USER ${APP_UID}:${APP_GID}
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["santapp-ruler", "doctor"]
