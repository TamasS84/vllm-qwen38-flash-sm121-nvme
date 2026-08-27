# Qwen3.8 Flash on DGX Spark with NVMe-backed PLE

This workflow builds a dedicated image, uses a dedicated container, and serves
on port 8010. The launcher only replaces a container named
`vllm_qwen38_flash_nvme`, so unrelated containers are left alone.

The launcher uses TP1 with the multiprocessing executor required by PLE
offload, text-only target-model CUDA graphs, the model's native one-token MTP
proposer in eager mode, the full 262,144-token context, a BF16 KV cache, and no
prefix caching. Qwen3.8 QSA requires the main KV cache to remain BF16. The
default balanced profile schedules five concurrent requests with an
8,192-token batch budget; the throughput profile schedules ten. A `0.82`
GPU-memory-utilization budget leaves enough room for the weights, graph
captures, and at least two full-length KV-cache sequences. The OpenAI server
uses the `qwen3` reasoning parser and `qwen3_coder` tool parser, so private
reasoning and structured tool calls are not mixed into visible content.

## Quick start on a new DGX Spark

Prerequisites are an Ubuntu-based DGX Spark with its NVIDIA driver and
Container Toolkit working with `docker --gpus all`, Docker Buildx, Git, `uv`,
and roughly 300 GB of free NVMe space during the first build. Stop other large
GPU models before starting this profile; the launcher itself only removes a
container named `vllm_qwen38_flash_nvme`.

The following is a complete BF16-head installation. `QWEN38_ROOT` makes the
workflow independent of the local Linux username:

```bash
export QWEN38_ROOT="${HOME}/qwen38-flash-vllm"
mkdir -p "$QWEN38_ROOT/src" "$QWEN38_ROOT/models/RadixArk" "$QWEN38_ROOT/ple"

git clone --branch qwen38-flash-sm121-nvme --single-branch \
  https://github.com/TamasS84/vllm-qwen38-flash-sm121-nvme.git \
  "$QWEN38_ROOT/src/vllm"
cd "$QWEN38_ROOT/src/vllm"

uvx --from huggingface_hub hf download \
  RadixArk/Qwen3.8-Flash-Next-NVFP4 \
  --local-dir \
  "$QWEN38_ROOT/models/RadixArk/Qwen3.8-Flash-Next-NVFP4"

uv run --no-project python tools/prepare_ple_nvme.py \
  --model-dir \
  "$QWEN38_ROOT/models/RadixArk/Qwen3.8-Flash-Next-NVFP4" \
  --output "$QWEN38_ROOT/ple/ple.fp8"

bash examples/online_serving/qwen38_flash_nvme/build_dgx_spark.sh
bash examples/online_serving/qwen38_flash_nvme/serve_dgx_spark.sh
```

The command above starts the balanced production profile. For parallel
agentic workloads, select the maximum-throughput profile:

```bash
SERVING_PROFILE=throughput \
  bash examples/online_serving/qwen38_flash_nvme/serve_dgx_spark.sh
```

Both profiles retain the 262,144-token context limit. Concurrency does not
reserve ten complete context windows up front; live requests share the KV
cache and are admitted according to their actual combined token use.

The CUDA 13 ARM64 image is built locally and can take a long time on the first
run. Model startup then takes roughly ten minutes. Follow the log, then press
Ctrl-C when you want to return to the shell:

```bash
tail -f "$QWEN38_ROOT/logs/server.log"
```

Check health separately:

```bash
curl --fail http://127.0.0.1:8010/health
```

For the faster W4A16 LM-head profile, build the image and PLE sidecar as above,
but run the converter before starting the server:

```bash
docker run --rm --gpus all --ipc=host \
  --entrypoint /usr/local/bin/uv \
  -v "$PWD/examples/online_serving/qwen38_flash_nvme/quantize_lm_head_nvfp4.py:/convert.py:ro" \
  -v "$QWEN38_ROOT/models/RadixArk:/checkpoints" \
  vllm-qwen38-flash-sm121:nvme \
  run --no-project python /convert.py \
  --source /checkpoints/Qwen3.8-Flash-Next-NVFP4 \
  --output /checkpoints/Qwen3.8-Flash-Next-NVFP4-W4A16-LMHead

MODEL_DIR="$QWEN38_ROOT/models/RadixArk/Qwen3.8-Flash-Next-NVFP4-W4A16-LMHead" \
  bash examples/online_serving/qwen38_flash_nvme/serve_dgx_spark.sh
```

Add `SERVING_PROFILE=throughput` to that command for the ten-request profile.
Always build the image from the same checkout used to run the converter. An
older image without the W4A16 LM-head loader will reject the converted
checkpoint with a tensor-shape mismatch.

The converter never modifies the downloaded checkpoint. To return to its BF16
head, launch the default checkpoint explicitly:

```bash
unset MODEL_DIR
bash examples/online_serving/qwen38_flash_nvme/serve_dgx_spark.sh
```

## Reasoning and tool-call examples

The current vLLM API names the separated field `reasoning`; older clients may
know it as `reasoning_content`. The visible answer is in `content`, without
`<think>` markup:

```bash
curl --fail http://127.0.0.1:8010/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "/models/qwen38-flash",
    "messages": [
      {"role": "user", "content": "Calculate 27 * 14 and give the final answer."}
    ],
    "temperature": 0,
    "max_tokens": 256,
    "chat_template_kwargs": {"enable_thinking": true}
  }'
```

Automatic tool choice returns a structured `message.tool_calls` array. This
request asks the model to select a function but does not execute it:

```bash
curl --fail http://127.0.0.1:8010/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "/models/qwen38-flash",
    "messages": [
      {"role": "user", "content": "What is the weather in London right now? Use the weather tool."}
    ],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {
          "type": "object",
          "properties": {"city": {"type": "string"}},
          "required": ["city"]
        }
      }
    }],
    "tool_choice": "auto",
    "temperature": 0,
    "max_tokens": 256,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

Clients may override `enable_thinking` per request. The parser flags remain
enabled at the server level in either mode.

## Runtime layout

- Workspace: `$QWEN38_ROOT`
- Model: `$QWEN38_ROOT/models/RadixArk/Qwen3.8-Flash-Next-NVFP4`
- Optional W4A16-head experiment:
  `$QWEN38_ROOT/models/RadixArk/Qwen3.8-Flash-Next-NVFP4-W4A16-LMHead`
- PLE sidecar: `$QWEN38_ROOT/ple/ple.fp8`
- Persistent compiler/autotuning cache: `$QWEN38_ROOT/cache`
- Image: `vllm-qwen38-flash-sm121:nvme`
- Container: `vllm_qwen38_flash_nvme`
- API: `http://127.0.0.1:8010`

## Prepare, build, and serve

Run the sidecar builder from this vLLM checkout:

```bash
uv run --no-project python tools/prepare_ple_nvme.py \
  --model-dir "$QWEN38_ROOT/models/RadixArk/Qwen3.8-Flash-Next-NVFP4" \
  --output "$QWEN38_ROOT/ple/ple.fp8"
```

The builder streams raw SafeTensors ranges into one flat FP8 file and writes a
validated JSON manifest. At runtime the PLE worker opens the table with a
private file mapping; only lookup results are copied through the existing
pinned CPU-to-GPU buffers.

Build native CUDA 13 code for the DGX Spark architecture and start the isolated
server:

```bash
bash examples/online_serving/qwen38_flash_nvme/build_dgx_spark.sh
bash examples/online_serving/qwen38_flash_nvme/serve_dgx_spark.sh
```

The build selects PyTorch's official ARM64 manylinux CUDA 13 builder instead of
the upstream Dockerfile's AMD64 default and preserves `12.1a` as a native CMake
target. It uses four concurrent NVCC jobs on the 20-core Spark to keep the first
native build practical without exhausting unified memory.

Follow startup and query health:

```bash
tail -f "$QWEN38_ROOT/logs/server.log"
```

After exiting `tail`, query health:

```bash
curl --fail http://127.0.0.1:8010/health
```

The first start takes roughly ten minutes to read all 206 checkpoint shards
and materialize the target plus MTP draft weights. Subsequent kernel autotuning
also runs before the health endpoint becomes ready.

Validate the exact configured context limit with 262,016 prompt tokens and
128 generated tokens:

```bash
uv run --no-project python \
  examples/online_serving/qwen38_flash_nvme/validate_context_dgx_spark.py
```

## Optional W4A16 NVFP4 LM head

The original checkpoint keeps its 1.27 GB LM head in BF16. The converter below
creates a sibling checkpoint, hard-links every unchanged file, and rewrites
only the head shard as weight-only W4A16 NVFP4. The source directory is never
modified. Run it after stopping the isolated Flash container so the converter
can use the GPU:

```bash
docker run --rm --gpus all --ipc=host \
  --entrypoint /usr/local/bin/uv \
  -v "$PWD/examples/online_serving/qwen38_flash_nvme/quantize_lm_head_nvfp4.py:/convert.py:ro" \
  -v "$QWEN38_ROOT/models/RadixArk:/checkpoints" \
  vllm-qwen38-flash-sm121:nvme \
  run --no-project python /convert.py \
  --source /checkpoints/Qwen3.8-Flash-Next-NVFP4 \
  --output /checkpoints/Qwen3.8-Flash-Next-NVFP4-W4A16-LMHead
```

An image built from this checkout includes the target and MTP quantized-head
wiring. Select the sibling checkpoint without changing the launcher defaults:

```bash
MODEL_DIR="$QWEN38_ROOT/models/RadixArk/Qwen3.8-Flash-Next-NVFP4-W4A16-LMHead" \
  bash examples/online_serving/qwen38_flash_nvme/serve_dgx_spark.sh
```

To roll back, omit `MODEL_DIR` and run the launcher again. Do not replace the
original checkpoint with the experimental directory.

A deterministic six-prompt screen covered arithmetic, factual recall, logic,
code, technical explanation, and Hungarian translation. Both checkpoints gave
semantically correct answers; the quantized head changed wording on three
prompts. That is encouraging smoke-test evidence, not proof of zero quality
loss. Quantizing shared experts is intentionally deferred: it would rewrite
four more checkpoint shards, alter the fused-expert execution path, and expose
far more quality-sensitive weights for a less certain gain. The LM-head result
is large enough to keep that work out of this first usable version.

## First-run verification

Verified on a GB10 DGX Spark on 2026-08-27:

- Image: built locally from this branch with the included DGX Spark build
  script.
- Runtime: PyTorch 2.13.0 + CUDA 13.0 detected `NVIDIA GB10` capability
  `(12, 1)`. Native `sm_121a` cubins were found in vLLM core, MoE, QUTLASS,
  and Flash-KDA extensions.
- Compiled-image regressions: the PLE functionalization and isolated
  model-parallel configuration tests both passed.
- Sidecar: exactly 51,200,245,760 bytes, shape `320001536 x 160`, FP8 E4M3,
  assembled from 128 PLE checkpoint shards.
- Full-context BF16-head profile: the target plus one draft layer used
  77.75 GiB. vLLM exposed 745,986 KV-cache tokens, or 2.85 full-length
  sequences, at `--gpu-memory-utilization 0.82`.
- W4A16-head profile: the converted head shrank from 1,271,398,400 to
  357,580,804 bytes. With the throughput profile, startup reported 79.61 GiB
  for weights and non-torch allocations, 2.10 GiB peak activation memory,
  0.30 GiB for CUDA graphs, and 18.02 GiB for KV cache. vLLM exposed 665,096
  KV-cache tokens, or 2.54 full-length sequences; shorter concurrent requests
  share that same token pool. The primary GPU process used about 103.2 GiB
  after the full-context and parallel verification runs.
- API: `/health` returned HTTP 200. Exact-limit requests with 262,016 prompt
  tokens plus 128 generated tokens succeeded with both heads; the W4A16 run
  under the throughput profile completed with exact returned usage validation.
- OpenAI chat parsing: a thinking-enabled request returned private work in
  `message.reasoning`, the final answer in `message.content` with no `<think>`
  markup, and `completion_tokens_details.reasoning_tokens: 49`. An automatic
  weather-tool request returned `finish_reason: "tool_calls"` and one valid
  `get_weather` call.
- NVMe residency after the W4A16 parallel, quality, and full-context
  request: the PLE process had one private `rw-p` mapping for `/ple/ple.fp8`.
  The 50,000,240 kB mapping had `Rss: 4,821,428 kB` after the c10 and 256K
  workloads, all private-clean, and `Anonymous: 0 kB`, so the full 51.2 GB
  table was not copied into anonymous RAM.

For a non-reasoning chat smoke test, pass this request field:

```json
"chat_template_kwargs": {"enable_thinking": false}
```
