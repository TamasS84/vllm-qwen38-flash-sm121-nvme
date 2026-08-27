#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Create an isolated Qwen3.8 Flash checkpoint with a W4A16 NVFP4 LM head."""

import argparse
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

HEAD_WEIGHT = "lm_head.weight"
INDEX_NAME = "model.safetensors.index.json"
EXPERT_SCALE_PATTERN = re.compile(
    r"^(?P<root>.+\.layers\.\d+\.mlp)\.experts\.\d+\."
    r"(?:gate_proj|up_proj|down_proj)\.weight_scale$"
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def quantized_layer_map(weight_map: dict[str, str]) -> dict[str, dict[str, Any]]:
    expert_layers = set()
    for name in weight_map:
        if match := EXPERT_SCALE_PATTERN.match(name):
            expert_layers.add(f"{match.group('root')}.experts")
    if not expert_layers:
        raise ValueError("checkpoint index contains no NVFP4 expert scales")

    layers = {
        name: {"group_size": 16, "quant_algo": "NVFP4"}
        for name in sorted(expert_layers)
    }
    layers["lm_head"] = {
        "group_size": 16,
        "quant_algo": "W4A16_NVFP4",
    }
    return layers


def mixed_precision_configs(
    config: dict[str, Any],
    hf_quant_config: dict[str, Any],
    layers: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    original_quantization = hf_quant_config.get("quantization", hf_quant_config)
    exclude_modules = [
        name
        for name in original_quantization.get("exclude_modules", [])
        if name != "lm_head"
    ]
    producer = hf_quant_config.get(
        "producer",
        config.get("quantization_config", {}).get("producer", {}),
    )
    quantization: dict[str, Any] = {
        "exclude_modules": exclude_modules,
        "quant_algo": "MIXED_PRECISION",
        "quantized_layers": layers,
    }
    if kv_algo := original_quantization.get("kv_cache_quant_algo"):
        quantization["kv_cache_quant_algo"] = kv_algo
    updated_hf_quant_config = {
        "producer": producer,
        "quantization": quantization,
    }

    updated_config = dict(config)
    original_config_quant = config.get("quantization_config", {})
    config_quant: dict[str, Any] = {
        "ignore": exclude_modules,
        "producer": original_config_quant.get("producer", producer),
        "quant_algo": "MIXED_PRECISION",
        "quant_method": "modelopt",
        "quantized_layers": layers,
    }
    if kv_scheme := original_config_quant.get("kv_cache_scheme"):
        config_quant["kv_cache_scheme"] = kv_scheme
    updated_config["quantization_config"] = config_quant
    return updated_config, updated_hf_quant_config


def quantize_head_shard(source: Path, destination: Path) -> tuple[int, int]:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    from vllm._custom_ops import scaled_fp4_quant

    with safe_open(source, framework="pt", device="cpu") as checkpoint:
        if HEAD_WEIGHT not in checkpoint:
            raise ValueError(f"{source.name} does not contain {HEAD_WEIGHT}")
        metadata = checkpoint.metadata()
        tensors = {
            name: checkpoint.get_tensor(name)
            for name in checkpoint
            if name != HEAD_WEIGHT
        }
        head = checkpoint.get_tensor(HEAD_WEIGHT)

    if head.dtype != torch.bfloat16:
        raise ValueError(f"expected a BF16 LM head, found {head.dtype}")
    if head.ndim != 2 or head.shape[-1] % 16:
        raise ValueError(f"unsupported LM-head shape: {tuple(head.shape)}")
    if not torch.cuda.is_available():
        raise RuntimeError("NVFP4 LM-head conversion requires a CUDA GPU")

    original_bytes = head.numel() * head.element_size()
    with torch.inference_mode():
        cuda_head = head.to("cuda")
        amax = cuda_head.abs().amax().to(torch.float32).clamp_min(1e-8)
        global_scale = (6.0 * torch.finfo(torch.float8_e4m3fn).max) / amax
        packed, group_scales = scaled_fp4_quant(
            cuda_head,
            global_scale,
            is_sf_swizzled_layout=False,
        )
        tensors[HEAD_WEIGHT] = packed.cpu()
        tensors["lm_head.weight_scale"] = group_scales.cpu()
        tensors["lm_head.weight_scale_2"] = (1.0 / global_scale).cpu()
    del cuda_head, head, packed, group_scales
    torch.cuda.empty_cache()

    quantized_bytes = sum(
        tensors[name].numel() * tensors[name].element_size()
        for name in (
            HEAD_WEIGHT,
            "lm_head.weight_scale",
            "lm_head.weight_scale_2",
        )
    )
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    save_file(tensors, temporary, metadata=metadata)
    os.replace(temporary, destination)
    return original_bytes, quantized_bytes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def convert_checkpoint(source: Path, output: Path) -> dict[str, int | str]:
    source = source.resolve()
    output = output.resolve()
    if source == output:
        raise ValueError("source and output checkpoint directories must differ")
    if not source.is_dir():
        raise FileNotFoundError(source)
    if output.exists():
        raise FileExistsError(
            f"refusing to overwrite existing output checkpoint: {output}"
        )

    index = read_json(source / INDEX_NAME)
    weight_map = index["weight_map"]
    head_shard_name = weight_map.get(HEAD_WEIGHT)
    if head_shard_name is None:
        raise ValueError(f"checkpoint index does not contain {HEAD_WEIGHT}")
    layers = quantized_layer_map(weight_map)
    config, hf_quant_config = mixed_precision_configs(
        read_json(source / "config.json"),
        read_json(source / "hf_quant_config.json"),
        layers,
    )

    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}.",
        dir=output.parent,
    ) as temporary:
        staging = Path(temporary) / output.name
        shutil.copytree(source, staging, copy_function=os.link)
        head_shard = staging / head_shard_name
        original_bytes, quantized_bytes = quantize_head_shard(
            source / head_shard_name,
            head_shard,
        )
        weight_map["lm_head.weight_scale"] = head_shard_name
        weight_map["lm_head.weight_scale_2"] = head_shard_name
        if "metadata" in index and "total_size" in index["metadata"]:
            index["metadata"]["total_size"] += quantized_bytes - original_bytes

        write_json(staging / INDEX_NAME, index)
        write_json(staging / "config.json", config)
        write_json(staging / "hf_quant_config.json", hf_quant_config)
        if output.exists():
            raise FileExistsError(
                f"refusing to overwrite existing output checkpoint: {output}"
            )
        os.replace(staging, output)

    return {
        "expert_layer_groups": len(layers) - 1,
        "lm_head_original_bytes": original_bytes,
        "lm_head_quantized_bytes": quantized_bytes,
        "output": str(output),
        "source": str(source),
    }


def main() -> None:
    args = parse_args()
    print(
        json.dumps(
            convert_checkpoint(args.source, args.output),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
