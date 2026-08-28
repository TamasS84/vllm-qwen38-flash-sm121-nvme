#!/usr/bin/env bash
set -euo pipefail

QWEN38_ROOT=${QWEN38_ROOT:-"$HOME/qwen38-flash-vllm"}
DOCKER_BIN=${DOCKER_BIN:-docker}
MODEL_DIR=${MODEL_DIR:-"$QWEN38_ROOT/models/RadixArk/Qwen3.8-Flash-Next-NVFP4"}
PLE_DIR="$QWEN38_ROOT/ple"
PLE_PATH="$PLE_DIR/ple.fp8"
CACHE_DIR="$QWEN38_ROOT/cache"
LOG_DIR="$QWEN38_ROOT/logs"
CONTAINER_NAME=vllm_qwen38_flash_nvme
IMAGE_NAME=${IMAGE_NAME:-vllm-qwen38-flash-sm121:nvme}
SERVING_PROFILE=${SERVING_PROFILE:-balanced}
ENABLE_MTP=${ENABLE_MTP:-1}

case "$SERVING_PROFILE" in
  balanced)
    PROFILE_MAX_NUM_SEQS=5
    PROFILE_MAX_NUM_BATCHED_TOKENS=8192
    ;;
  throughput)
    PROFILE_MAX_NUM_SEQS=10
    PROFILE_MAX_NUM_BATCHED_TOKENS=8192
    ;;
  *)
    echo "SERVING_PROFILE must be balanced or throughput, got: $SERVING_PROFILE" >&2
    exit 1
    ;;
esac

MAX_NUM_SEQS=${MAX_NUM_SEQS:-$PROFILE_MAX_NUM_SEQS}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-$PROFILE_MAX_NUM_BATCHED_TOKENS}
KV_CACHE_MEMORY_BYTES=${KV_CACHE_MEMORY_BYTES:-12G}

require_positive_integer() {
  local name=$1
  local value=$2
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "$name must be a positive integer, got: $value" >&2
    exit 1
  fi
}

require_positive_integer MAX_NUM_SEQS "$MAX_NUM_SEQS"
require_positive_integer MAX_NUM_BATCHED_TOKENS "$MAX_NUM_BATCHED_TOKENS"
if [[ ! "$KV_CACHE_MEMORY_BYTES" =~ ^[1-9][0-9]*([KMGTP]i?B?)?$ ]]; then
  echo "KV_CACHE_MEMORY_BYTES must be a positive byte size, got: $KV_CACHE_MEMORY_BYTES" >&2
  exit 1
fi

MTP_ARGS=()
if [[ "$ENABLE_MTP" != "0" && "$ENABLE_MTP" != "1" ]]; then
  echo "ENABLE_MTP must be 0 or 1, got: $ENABLE_MTP" >&2
  exit 1
fi
if [[ "$ENABLE_MTP" == "1" ]]; then
  MTP_ARGS=(
    --speculative-config
    '{"method":"mtp","num_speculative_tokens":1,"enforce_eager":true}'
  )
fi

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Required file is missing: $1" >&2
    exit 1
  fi
}

require_file "$MODEL_DIR/config.json"
require_file "$MODEL_DIR/model.safetensors.index.json"
require_file "$PLE_PATH"
require_file "$PLE_PATH.json"
mkdir -p "$CACHE_DIR" "$LOG_DIR"

"$DOCKER_BIN" rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

"$DOCKER_BIN" run --detach \
  --name "$CONTAINER_NAME" \
  --gpus all \
  --ipc host \
  --publish 8010:8000 \
  --restart unless-stopped \
  --env VLLM_PLE_CPU_OFFLOAD=1 \
  --env VLLM_PLE_NVME_PATH=/ple/ple.fp8 \
  --volume "$MODEL_DIR:/models/qwen38-flash:ro" \
  --volume "$PLE_DIR:/ple:ro" \
  --volume "$CACHE_DIR:/root/.cache" \
  "$IMAGE_NAME" \
  /models/qwen38-flash \
  --host 0.0.0.0 \
  --port 8000 \
  --trust-remote-code \
  --language-model-only \
  --tensor-parallel-size 1 \
  --distributed-executor-backend mp \
  --kv-cache-memory-bytes "$KV_CACHE_MEMORY_BYTES" \
  --max-model-len 262144 \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --enable-chunked-prefill \
  "${MTP_ARGS[@]}" \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --no-enable-prefix-caching

nohup "$DOCKER_BIN" logs --follow "$CONTAINER_NAME" \
  >"$LOG_DIR/server.log" 2>&1 &
echo "$!" >"$LOG_DIR/server-log.pid"

echo "Started $CONTAINER_NAME on http://127.0.0.1:8010"
echo "Logs: $LOG_DIR/server.log"
