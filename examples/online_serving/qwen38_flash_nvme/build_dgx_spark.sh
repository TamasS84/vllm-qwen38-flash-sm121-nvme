#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)
DOCKER_BIN=${DOCKER_BIN:-docker}

cd "$REPO_ROOT"
"$DOCKER_BIN" buildx build \
  --load \
  --platform linux/arm64 \
  --target vllm-openai \
  --build-arg CUDA_VERSION=13.0.3 \
  --build-arg BUILD_BASE_IMAGE=pytorch/manylinuxaarch64-builder:cuda13.0 \
  --build-arg torch_cuda_arch_list=12.1a \
  --build-arg max_jobs=16 \
  --build-arg nvcc_threads=4 \
  --tag vllm-qwen38-flash-sm121:nvme \
  --file docker/Dockerfile \
  .
