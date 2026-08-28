# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build a flat NVMe-backed Qwen PLE table from SafeTensors shards."""

import argparse
import json
import os
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

_PLE_SHARD_RE = re.compile(r"(?:^|\.)ngram_embedding\.shard_(\d+)\.weight$")
_COPY_BUFFER_SIZE = 16 * 1024 * 1024
_FORMAT_VERSION = 1
_FP8_DTYPE = "F8_E4M3"


@dataclass(frozen=True)
class PleShard:
    index: int
    source_path: Path
    data_offset: int
    byte_length: int
    rows: int
    columns: int


def _read_json_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def read_safetensors_header(path: Path) -> tuple[int, dict[str, object]]:
    """Return the absolute data-section offset and decoded tensor header."""
    with path.open("rb") as source:
        length_bytes = source.read(8)
        if len(length_bytes) != 8:
            raise ValueError(f"SafeTensors file has no complete header length: {path}")
        (header_length,) = struct.unpack("<Q", length_bytes)
        header_bytes = source.read(header_length)
        if len(header_bytes) != header_length:
            raise ValueError(f"SafeTensors file has a truncated header: {path}")
    try:
        header = json.loads(header_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(
            f"SafeTensors file has an invalid JSON header: {path}"
        ) from error
    if not isinstance(header, dict):
        raise ValueError(f"SafeTensors header must be a JSON object: {path}")
    return 8 + header_length, header


def _expected_shard_count(model_dir: Path) -> int:
    config = _read_json_object(model_dir / "config.json")
    text_config = config.get("text_config", config)
    if not isinstance(text_config, dict):
        raise ValueError("config.json text_config must be an object")
    shard_count = text_config.get("split_ngram_parts")
    if not isinstance(shard_count, int) or shard_count <= 0:
        raise ValueError("config.json must define a positive split_ngram_parts")
    return shard_count


def discover_ple_shards(model_dir: Path) -> list[PleShard]:
    """Find and validate PLE tensors, returning them in numeric shard order."""
    model_dir = model_dir.resolve()
    index = _read_json_object(model_dir / "model.safetensors.index.json")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError("model.safetensors.index.json must contain a weight_map")

    locations: dict[int, tuple[str, str]] = {}
    for tensor_name, filename in weight_map.items():
        if not isinstance(tensor_name, str) or not isinstance(filename, str):
            raise ValueError("SafeTensors weight_map entries must be strings")
        match = _PLE_SHARD_RE.search(tensor_name)
        if match is None:
            continue
        shard_index = int(match.group(1))
        if shard_index in locations:
            raise ValueError(f"Duplicate PLE embedding shard index {shard_index}")
        locations[shard_index] = (tensor_name, filename)

    expected_count = _expected_shard_count(model_dir)
    actual_indices = sorted(locations)
    expected_indices = list(range(expected_count))
    if actual_indices != expected_indices:
        raise ValueError(
            "Expected contiguous PLE shard indices "
            f"0..{expected_count - 1}, got {actual_indices}"
        )

    header_cache: dict[Path, tuple[int, dict[str, object]]] = {}
    shards: list[PleShard] = []
    expected_columns: int | None = None
    for shard_index in actual_indices:
        tensor_name, filename = locations[shard_index]
        source_path = (model_dir / filename).resolve()
        if not source_path.is_file():
            raise ValueError(f"PLE shard file does not exist: {source_path}")
        if source_path not in header_cache:
            header_cache[source_path] = read_safetensors_header(source_path)
        data_start, header = header_cache[source_path]
        metadata = header.get(tensor_name)
        if not isinstance(metadata, dict):
            raise ValueError(f"SafeTensors header is missing {tensor_name}")
        dtype = metadata.get("dtype")
        if dtype != _FP8_DTYPE:
            raise ValueError(
                f"PLE embedding shard {shard_index} must use F8_E4M3, got {dtype}"
            )
        shape = metadata.get("shape")
        if (
            not isinstance(shape, list)
            or len(shape) != 2
            or not all(isinstance(value, int) and value > 0 for value in shape)
        ):
            raise ValueError(
                f"PLE embedding shard {shard_index} must have a positive 2D shape"
            )
        rows, columns = shape
        if expected_columns is None:
            expected_columns = columns
        elif columns != expected_columns:
            raise ValueError("All PLE embedding shards must use the same column count")

        offsets = metadata.get("data_offsets")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(value, int) for value in offsets)
        ):
            raise ValueError(
                f"PLE embedding shard {shard_index} has invalid data_offsets"
            )
        relative_start, relative_end = offsets
        if relative_start < 0 or relative_end < relative_start:
            raise ValueError(
                f"PLE embedding shard {shard_index} has an invalid byte range"
            )
        byte_length = relative_end - relative_start
        if byte_length != rows * columns:
            raise ValueError(
                f"PLE embedding shard {shard_index} byte range ({byte_length}) "
                f"does not match shape ({rows}, {columns})"
            )
        absolute_start = data_start + relative_start
        if absolute_start + byte_length > source_path.stat().st_size:
            raise ValueError(
                f"PLE embedding shard {shard_index} byte range exceeds {source_path}"
            )
        shards.append(
            PleShard(
                index=shard_index,
                source_path=source_path,
                data_offset=absolute_start,
                byte_length=byte_length,
                rows=rows,
                columns=columns,
            )
        )
    return shards


def copy_file_range(
    source: BinaryIO,
    target: BinaryIO,
    offset: int,
    size: int,
) -> None:
    """Copy one tensor byte range without materializing it as a tensor."""
    source.seek(offset)
    remaining = size
    while remaining:
        chunk = source.read(min(remaining, _COPY_BUFFER_SIZE))
        if not chunk:
            raise ValueError("Unexpected end of SafeTensors data while copying PLE")
        target.write(chunk)
        remaining -= len(chunk)


def _manifest_for(shards: list[PleShard]) -> dict[str, object]:
    return {
        "version": _FORMAT_VERSION,
        "dtype": _FP8_DTYPE,
        "rows": sum(shard.rows for shard in shards),
        "columns": shards[0].columns,
        "shard_count": len(shards),
        "byte_size": sum(shard.byte_length for shard in shards),
    }


def _manifest_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.name}.json")


def build_ple_sidecar(model_dir: Path, output_path: Path) -> dict[str, object]:
    """Build or validate a flat PLE sidecar and return its manifest."""
    shards = discover_ple_shards(model_dir)
    if not shards:
        raise ValueError("No PLE embedding shards were found")
    manifest = _manifest_for(shards)
    output_path = output_path.resolve()
    manifest_path = _manifest_path(output_path)

    if output_path.is_file() and manifest_path.is_file():
        existing = _read_json_object(manifest_path)
        if existing == manifest and output_path.stat().st_size == manifest["byte_size"]:
            return manifest

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_tmp = output_path.with_name(f"{output_path.name}.tmp")
    manifest_tmp = manifest_path.with_name(f"{manifest_path.name}.tmp")

    with output_tmp.open("wb") as target:
        for shard in shards:
            with shard.source_path.open("rb") as source:
                copy_file_range(
                    source,
                    target,
                    offset=shard.data_offset,
                    size=shard.byte_length,
                )
        target.flush()
        os.fsync(target.fileno())
    if output_tmp.stat().st_size != manifest["byte_size"]:
        raise ValueError(
            f"PLE sidecar size mismatch: expected {manifest['byte_size']}, "
            f"got {output_tmp.stat().st_size}"
        )
    os.replace(output_tmp, output_path)

    with manifest_tmp.open("w", encoding="utf-8", newline="\n") as target:
        json.dump(manifest, target, indent=2, sort_keys=True)
        target.write("\n")
        target.flush()
        os.fsync(target.fileno())
    os.replace(manifest_tmp, manifest_path)
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a flat FP8 Qwen PLE table for NVMe-backed vLLM lookup."
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    manifest = build_ple_sidecar(args.model_dir, args.output)
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
