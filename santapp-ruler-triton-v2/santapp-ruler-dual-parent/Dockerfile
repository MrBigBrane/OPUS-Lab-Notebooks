ARG PYTORCH_IMAGE=pytorch/pytorch:2.11.0-cuda12.6-cudnn9-runtime
FROM ${PYTORCH_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
ARG APP_UID=10001
ARG APP_GID=10001
# Only for WSL/Blackwell tests. Production Ampere defaults to the original image.
ARG LOCAL_CUDA128=0

LABEL org.opencontainers.image.title="SANTA RULER benchmark" \
      org.opencontainers.image.description="SDPA, SANTA, and K-means or contiguous-parent SANTA++ attention" \
      org.opencontainers.image.authors="Research contributors"

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git tar tini build-essential \
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
    HOME=/home/benchmark \
    TRITON_CACHE_DIR=/tmp/santapp-triton \
    CUDA_CACHE_PATH=/tmp/santapp-cuda \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

WORKDIR /workspace
COPY pyproject.toml README.md LICENSE NOTICE CITATION.cff .gitignore ./
COPY src ./src
COPY Dockerfile ./Dockerfile
RUN python -m pip install --break-system-packages --upgrade pip wheel "setuptools<82" \
    && python -m pip uninstall --break-system-packages -y spin \
    && if [ "${LOCAL_CUDA128}" = "1" ]; then \
         python -m pip install --break-system-packages --upgrade --force-reinstall \
           torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
           --index-url https://download.pytorch.org/whl/cu128 \
         && python -c "import torch; assert torch.version.cuda == '12.8', torch.version.cuda"; \
       elif [ "${LOCAL_CUDA128}" != "0" ]; then exit 2; fi \
    && python -m pip install --break-system-packages . "pytest>=8,<10" \
    && python -m pip check \
    && python -c "from santapp_ruler.attention.prefill_runtime import check_triton_package; print(check_triton_package())" \
    && python -m pip freeze > /workspace/BUILD_ENVIRONMENT.txt

COPY configs ./configs
COPY scripts ./scripts
COPY docs ./docs
COPY k8s ./k8s
COPY tests ./tests
RUN chmod +x scripts/*.py scripts/*.sh \
    && chown -R "${APP_UID}:${APP_GID}" /workspace /home/benchmark

USER ${APP_UID}:${APP_GID}
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["santapp-ruler", "doctor"]
