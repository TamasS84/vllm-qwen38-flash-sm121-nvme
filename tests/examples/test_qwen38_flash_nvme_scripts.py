# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
import shlex
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_DIR = REPO_ROOT / "examples" / "online_serving" / "qwen38_flash_nvme"
BUILD_SCRIPT = EXAMPLE_DIR / "build_dgx_spark.sh"
SERVE_SCRIPT = EXAMPLE_DIR / "serve_dgx_spark.sh"
CONTEXT_SCRIPT = EXAMPLE_DIR / "validate_context_dgx_spark.py"
QUANTIZE_HEAD_SCRIPT = EXAMPLE_DIR / "quantize_lm_head_nvfp4.py"


@pytest.fixture
def fake_docker(tmp_path: Path) -> tuple[Path, Path]:
    docker_path = tmp_path / "docker"
    docker_path.write_text(
        "#!/usr/bin/env bash\n"
        'printf \'%q \' "$@" >> "$DOCKER_LOG"\n'
        "printf '\\n' >> \"$DOCKER_LOG\"\n",
        encoding="utf-8",
    )
    docker_path.chmod(0o755)
    return docker_path, tmp_path / "docker.log"


def _run_script(
    script: Path,
    fake_docker: tuple[Path, Path],
    *,
    qwen38_root: Path | None = None,
    extra_env: dict[str, str] | None = None,
) -> tuple[subprocess.CompletedProcess[str], list[list[str]]]:
    docker_path, log_path = fake_docker
    env = os.environ.copy()
    env.update({"DOCKER_BIN": str(docker_path), "DOCKER_LOG": str(log_path)})
    if qwen38_root is not None:
        env["QWEN38_ROOT"] = str(qwen38_root)
    if extra_env is not None:
        env.update(extra_env)
    result = subprocess.run(
        ["bash", str(script)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    calls = []
    if log_path.exists():
        calls = [shlex.split(line) for line in log_path.read_text().splitlines()]
    return result, calls


def _option(args: list[str], name: str) -> list[str]:
    index = args.index(name)
    return args[index : index + 2]


def _create_runtime_fixture(root: Path, *, include_manifest: bool = True) -> None:
    model_dir = root / "models" / "RadixArk" / "Qwen3.8-Flash-Next-NVFP4"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "model.safetensors.index.json").write_text("{}", encoding="utf-8")
    ple_dir = root / "ple"
    ple_dir.mkdir()
    (ple_dir / "ple.fp8").write_bytes(b"x")
    if include_manifest:
        (ple_dir / "ple.fp8.json").write_text("{}", encoding="utf-8")


def test_build_script_targets_sm121_arm64_image(fake_docker) -> None:
    result, calls = _run_script(BUILD_SCRIPT, fake_docker)

    assert result.returncode == 0, result.stderr
    assert len(calls) == 1
    args = calls[0]
    assert args[:2] == ["buildx", "build"]
    assert "--load" in args
    assert _option(args, "--platform") == ["--platform", "linux/arm64"]
    assert _option(args, "--target") == ["--target", "vllm-openai"]
    assert "CUDA_VERSION=13.0.3" in args
    assert "BUILD_BASE_IMAGE=pytorch/manylinuxaarch64-builder:cuda13.0" in args
    assert "torch_cuda_arch_list=12.1a" in args
    assert "max_jobs=16" in args
    assert "nvcc_threads=4" in args
    assert _option(args, "--tag") == [
        "--tag",
        "vllm-qwen38-flash-sm121:nvme",
    ]
    assert _option(args, "--file") == ["--file", "docker/Dockerfile"]
    assert args[-1] == "."


def test_serve_script_uses_isolated_nvme_configuration(
    tmp_path: Path,
    fake_docker,
) -> None:
    root = tmp_path / "qwen38"
    _create_runtime_fixture(root)

    result, calls = _run_script(SERVE_SCRIPT, fake_docker, qwen38_root=root)

    assert result.returncode == 0, result.stderr
    assert calls[0] == ["rm", "-f", "vllm_qwen38_flash_nvme"]
    run_args = calls[1]
    assert run_args[:2] == ["run", "--detach"]
    assert _option(run_args, "--name") == [
        "--name",
        "vllm_qwen38_flash_nvme",
    ]
    assert _option(run_args, "--publish") == ["--publish", "8010:8000"]
    assert _option(run_args, "--restart") == ["--restart", "unless-stopped"]
    assert "VLLM_PLE_CPU_OFFLOAD=1" in run_args
    assert "VLLM_PLE_NVME_PATH=/ple/ple.fp8" in run_args
    assert (
        f"{root}/models/RadixArk/Qwen3.8-Flash-Next-NVFP4:/models/qwen38-flash:ro"
        in run_args
    )
    assert f"{root}/ple:/ple:ro" in run_args
    assert f"{root}/cache:/root/.cache" in run_args
    assert "vllm-qwen38-flash-sm121:nvme" in run_args
    assert "--language-model-only" in run_args
    assert _option(run_args, "--tensor-parallel-size") == [
        "--tensor-parallel-size",
        "1",
    ]
    assert _option(run_args, "--distributed-executor-backend") == [
        "--distributed-executor-backend",
        "mp",
    ]
    assert "--enforce-eager" not in run_args
    assert "--gpu-memory-utilization" not in run_args
    assert _option(run_args, "--kv-cache-memory-bytes") == [
        "--kv-cache-memory-bytes",
        "17G",
    ]
    assert "--kv-cache-dtype" not in run_args
    assert _option(run_args, "--max-model-len") == [
        "--max-model-len",
        "262144",
    ]
    assert _option(run_args, "--max-num-seqs") == ["--max-num-seqs", "5"]
    assert _option(run_args, "--max-num-batched-tokens") == [
        "--max-num-batched-tokens",
        "8192",
    ]
    assert "--enable-chunked-prefill" in run_args
    assert _option(run_args, "--speculative-config") == [
        "--speculative-config",
        '{"method":"mtp","num_speculative_tokens":1,"enforce_eager":true}',
    ]
    assert _option(run_args, "--reasoning-parser") == [
        "--reasoning-parser",
        "qwen3",
    ]
    assert "--enable-auto-tool-choice" in run_args
    assert _option(run_args, "--tool-call-parser") == [
        "--tool-call-parser",
        "qwen3_coder",
    ]
    assert "--enable-prefix-caching" in run_args
    assert "--no-enable-prefix-caching" not in run_args
    assert _option(run_args, "--prefix-cache-retention-interval") == [
        "--prefix-cache-retention-interval",
        "12672",
    ]


def test_serve_script_default_root_follows_home(
    tmp_path: Path,
    fake_docker,
) -> None:
    home = tmp_path / "operator"
    root = home / "qwen38-flash-vllm"
    _create_runtime_fixture(root)

    result, calls = _run_script(
        SERVE_SCRIPT,
        fake_docker,
        extra_env={"HOME": str(home)},
    )

    assert result.returncode == 0, result.stderr
    run_args = calls[1]
    assert f"{root}/ple:/ple:ro" in run_args
    assert f"{root}/cache:/root/.cache" in run_args


def test_serve_script_accepts_model_and_image_overrides(
    tmp_path: Path,
    fake_docker,
) -> None:
    root = tmp_path / "qwen38"
    _create_runtime_fixture(root)
    model_dir = root / "models" / "quantized-head"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "model.safetensors.index.json").write_text("{}", encoding="utf-8")

    result, calls = _run_script(
        SERVE_SCRIPT,
        fake_docker,
        qwen38_root=root,
        extra_env={
            "MODEL_DIR": str(model_dir),
            "IMAGE_NAME": "vllm-qwen38-flash-sm121:experiment",
        },
    )

    assert result.returncode == 0, result.stderr
    run_args = calls[1]
    assert f"{model_dir}:/models/qwen38-flash:ro" in run_args
    assert "vllm-qwen38-flash-sm121:experiment" in run_args


def test_serve_script_accepts_parallelism_and_mtp_overrides(
    tmp_path: Path,
    fake_docker,
) -> None:
    root = tmp_path / "qwen38"
    _create_runtime_fixture(root)

    result, calls = _run_script(
        SERVE_SCRIPT,
        fake_docker,
        qwen38_root=root,
        extra_env={
            "MAX_NUM_SEQS": "10",
            "MAX_NUM_BATCHED_TOKENS": "8192",
            "KV_CACHE_MEMORY_BYTES": "14G",
            "PREFIX_CACHE_RETENTION_INTERVAL": "25344",
            "ENABLE_MTP": "0",
        },
    )

    assert result.returncode == 0, result.stderr
    run_args = calls[1]
    assert _option(run_args, "--max-num-seqs") == ["--max-num-seqs", "10"]
    assert _option(run_args, "--max-num-batched-tokens") == [
        "--max-num-batched-tokens",
        "8192",
    ]
    assert _option(run_args, "--kv-cache-memory-bytes") == [
        "--kv-cache-memory-bytes",
        "14G",
    ]
    assert _option(run_args, "--prefix-cache-retention-interval") == [
        "--prefix-cache-retention-interval",
        "25344",
    ]
    assert "--speculative-config" not in run_args


def test_serve_script_supports_maximum_throughput_profile(
    tmp_path: Path,
    fake_docker,
) -> None:
    root = tmp_path / "qwen38"
    _create_runtime_fixture(root)

    result, calls = _run_script(
        SERVE_SCRIPT,
        fake_docker,
        qwen38_root=root,
        extra_env={"SERVING_PROFILE": "throughput"},
    )

    assert result.returncode == 0, result.stderr
    run_args = calls[1]
    assert _option(run_args, "--max-num-seqs") == ["--max-num-seqs", "10"]
    assert _option(run_args, "--max-num-batched-tokens") == [
        "--max-num-batched-tokens",
        "8192",
    ]


def test_serve_script_rejects_unknown_serving_profile(
    tmp_path: Path,
    fake_docker,
) -> None:
    root = tmp_path / "qwen38"
    _create_runtime_fixture(root)

    result, calls = _run_script(
        SERVE_SCRIPT,
        fake_docker,
        qwen38_root=root,
        extra_env={"SERVING_PROFILE": "unbounded"},
    )

    assert result.returncode != 0
    assert "SERVING_PROFILE must be balanced or throughput" in result.stderr
    assert calls == []


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("MAX_NUM_SEQS", "0"),
        ("MAX_NUM_SEQS", "-1"),
        ("MAX_NUM_SEQS", "1.5"),
        ("MAX_NUM_BATCHED_TOKENS", "0"),
        ("MAX_NUM_BATCHED_TOKENS", "eight"),
        ("PREFIX_CACHE_RETENTION_INTERVAL", "0"),
        ("PREFIX_CACHE_RETENTION_INTERVAL", "auto"),
    ],
)
def test_serve_script_rejects_invalid_numeric_overrides_before_docker(
    tmp_path: Path,
    fake_docker,
    name: str,
    value: str,
) -> None:
    root = tmp_path / "qwen38"
    _create_runtime_fixture(root)

    result, calls = _run_script(
        SERVE_SCRIPT,
        fake_docker,
        qwen38_root=root,
        extra_env={name: value},
    )

    assert result.returncode != 0
    assert f"{name} must be a positive integer" in result.stderr
    assert calls == []


def test_serve_script_rejects_invalid_kv_cache_size_before_docker(
    tmp_path: Path,
    fake_docker,
) -> None:
    root = tmp_path / "qwen38"
    _create_runtime_fixture(root)

    result, calls = _run_script(
        SERVE_SCRIPT,
        fake_docker,
        qwen38_root=root,
        extra_env={"KV_CACHE_MEMORY_BYTES": "all-of-it"},
    )

    assert result.returncode != 0
    assert "KV_CACHE_MEMORY_BYTES must be a positive byte size" in result.stderr
    assert calls == []


def test_serve_script_rejects_misaligned_prefix_retention_before_docker(
    tmp_path: Path,
    fake_docker,
) -> None:
    root = tmp_path / "qwen38"
    _create_runtime_fixture(root)

    result, calls = _run_script(
        SERVE_SCRIPT,
        fake_docker,
        qwen38_root=root,
        extra_env={"PREFIX_CACHE_RETENTION_INTERVAL": "12673"},
    )

    assert result.returncode != 0
    assert "must be a multiple of 1584" in result.stderr
    assert calls == []


def test_serve_script_rejects_invalid_mtp_toggle(
    tmp_path: Path,
    fake_docker,
) -> None:
    root = tmp_path / "qwen38"
    _create_runtime_fixture(root)

    result, calls = _run_script(
        SERVE_SCRIPT,
        fake_docker,
        qwen38_root=root,
        extra_env={"ENABLE_MTP": "sometimes"},
    )

    assert result.returncode != 0
    assert "ENABLE_MTP must be 0 or 1" in result.stderr
    assert calls == []


def test_serve_script_fails_before_docker_when_manifest_is_missing(
    tmp_path: Path,
    fake_docker,
) -> None:
    root = tmp_path / "qwen38"
    _create_runtime_fixture(root, include_manifest=False)

    result, calls = _run_script(SERVE_SCRIPT, fake_docker, qwen38_root=root)

    assert result.returncode != 0
    assert "ple.fp8.json" in result.stderr
    assert calls == []


def test_context_script_sends_exact_token_budget() -> None:
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            assert self.path == "/v1/models"
            body = json.dumps({"data": [{"id": "test-model"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:
            length = int(self.headers["Content-Length"])
            request = json.loads(self.rfile.read(length))
            requests.append((self.path, request))
            if self.path == "/tokenize":
                response = {
                    "count": 3,
                    "max_model_len": 7,
                    "tokens": [10, 20, 30],
                }
            else:
                assert self.path == "/v1/completions"
                response = {
                    "choices": [{"text": "ok"}],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 2},
                }
            body = json.dumps(response).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        result = subprocess.run(
            [
                sys.executable,
                str(CONTEXT_SCRIPT),
                "--base-url",
                f"http://127.0.0.1:{server.server_port}",
                "--prompt-tokens",
                "5",
                "--max-tokens",
                "2",
                "--expected-max-model-len",
                "7",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        server.shutdown()
        thread.join()

    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout)
    assert output["configured_max_model_len"] == 7
    assert output["requested_prompt_tokens"] == 5
    assert output["usage"] == {"prompt_tokens": 5, "completion_tokens": 2}
    assert requests == [
        (
            "/tokenize",
            {
                "model": "test-model",
                "prompt": "Stable context marker. ",
                "add_special_tokens": False,
            },
        ),
        (
            "/v1/completions",
            {
                "model": "test-model",
                "prompt": [10, 20, 30, 10, 20],
                "max_tokens": 2,
                "temperature": 0.0,
                "ignore_eos": True,
            },
        ),
    ]


def test_context_validator_rejects_wrong_limit_and_usage() -> None:
    spec = spec_from_file_location("validate_context", CONTEXT_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)

    with pytest.raises(ValueError, match="reported max_model_len"):
        module.validate_context_contract(
            configured_max_model_len=8192,
            expected_max_model_len=262144,
            prompt_tokens=8064,
            completion_tokens=128,
        )
    with pytest.raises(ValueError, match="requested token total"):
        module.validate_context_contract(
            configured_max_model_len=262144,
            expected_max_model_len=262144,
            prompt_tokens=262015,
            completion_tokens=128,
        )
    with pytest.raises(RuntimeError, match="prompt token count"):
        module.validate_usage(
            {"prompt_tokens": 4, "completion_tokens": 2},
            prompt_tokens=5,
            completion_tokens=2,
        )


def test_quantized_lm_head_config_preserves_expert_quantization() -> None:
    spec = spec_from_file_location("quantize_lm_head_nvfp4", QUANTIZE_HEAD_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)

    weight_map = {
        "model.language_model.layers.0.mlp.experts.0.gate_proj.weight_scale": (
            "model-00001.safetensors"
        ),
        "model.language_model.layers.0.mlp.experts.1.down_proj.weight_scale": (
            "model-00002.safetensors"
        ),
        "model.language_model.layers.1.mlp.experts.0.up_proj.weight_scale": (
            "model-00003.safetensors"
        ),
    }
    layers = module.quantized_layer_map(weight_map)
    assert layers == {
        "model.language_model.layers.0.mlp.experts": {
            "group_size": 16,
            "quant_algo": "NVFP4",
        },
        "model.language_model.layers.1.mlp.experts": {
            "group_size": 16,
            "quant_algo": "NVFP4",
        },
        "lm_head": {
            "group_size": 16,
            "quant_algo": "W4A16_NVFP4",
        },
    }

    config, hf_quant_config = module.mixed_precision_configs(
        {
            "quantization_config": {
                "ignore": ["model.embed_tokens", "lm_head"],
                "producer": {"name": "modelopt", "version": "test"},
            }
        },
        {
            "producer": {"name": "modelopt", "version": "test"},
            "quantization": {
                "exclude_modules": ["model.embed_tokens", "lm_head"],
                "quant_algo": "NVFP4",
            },
        },
        layers,
    )
    assert config["quantization_config"]["quant_algo"] == "MIXED_PRECISION"
    assert config["quantization_config"]["quantized_layers"] == layers
    assert config["quantization_config"]["ignore"] == ["model.embed_tokens"]
    assert hf_quant_config["quantization"]["quant_algo"] == "MIXED_PRECISION"
    assert hf_quant_config["quantization"]["quantized_layers"] == layers
    assert hf_quant_config["quantization"]["exclude_modules"] == ["model.embed_tokens"]


def test_quantized_lm_head_failure_leaves_no_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = spec_from_file_location("quantize_lm_head_nvfp4", QUANTIZE_HEAD_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)

    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "output"
    shard_name = "model-bf16.safetensors"
    (source / shard_name).write_bytes(b"source remains intact")
    (source / "config.json").write_text(
        json.dumps({"quantization_config": {}}),
        encoding="utf-8",
    )
    (source / "hf_quant_config.json").write_text(
        json.dumps({"quantization": {"exclude_modules": ["lm_head"]}}),
        encoding="utf-8",
    )
    (source / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "lm_head.weight": shard_name,
                    "model.layers.0.mlp.experts.0.gate_proj.weight_scale": (shard_name),
                }
            }
        ),
        encoding="utf-8",
    )

    def fail_quantization(source_shard: Path, output_shard: Path) -> None:
        raise RuntimeError("simulated conversion failure")

    monkeypatch.setattr(module, "quantize_head_shard", fail_quantization)
    with pytest.raises(RuntimeError, match="simulated conversion failure"):
        module.convert_checkpoint(source, output)

    assert not output.exists()
    assert (source / shard_name).read_bytes() == b"source remains intact"
    assert list(tmp_path.glob(".output.*")) == []
