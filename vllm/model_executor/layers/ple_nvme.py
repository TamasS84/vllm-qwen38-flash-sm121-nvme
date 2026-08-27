# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validation and file mapping for NVMe-backed FP8 PLE tables."""

import json
import os
from dataclasses import dataclass
from pathlib import Path

import torch

_FORMAT_VERSION = 1
_FP8_DTYPE = "F8_E4M3"


def _manifest_path(data_path: Path) -> Path:
    return data_path.with_name(f"{data_path.name}.json")


def _required_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"PLE NVMe manifest field {field} must be an integer")
    return value


@dataclass(frozen=True)
class PleNvmeManifest:
    version: int
    dtype: str
    rows: int
    columns: int
    shard_count: int
    byte_size: int

    @classmethod
    def load(cls, data_path: Path) -> "PleNvmeManifest":
        if not data_path.is_file():
            raise ValueError(f"PLE NVMe data file does not exist: {data_path}")
        manifest_path = _manifest_path(data_path)
        if not manifest_path.is_file():
            raise ValueError(f"PLE NVMe manifest does not exist: {manifest_path}")
        try:
            raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(
                f"PLE NVMe manifest is invalid JSON: {manifest_path}"
            ) from error
        if not isinstance(raw, dict):
            raise ValueError("PLE NVMe manifest must be a JSON object")

        version = _required_int(raw.get("version"), "version")
        if version != _FORMAT_VERSION:
            raise ValueError(
                f"Unsupported PLE NVMe format version {version}; "
                f"expected {_FORMAT_VERSION}"
            )
        dtype = raw.get("dtype")
        if dtype != _FP8_DTYPE:
            raise ValueError(f"PLE NVMe manifest must use F8_E4M3, got {dtype}")
        rows = _required_int(raw.get("rows"), "rows")
        if rows <= 0:
            raise ValueError("PLE NVMe manifest must define positive rows")
        columns = _required_int(raw.get("columns"), "columns")
        if columns <= 0:
            raise ValueError("PLE NVMe manifest must define positive columns")
        shard_count = _required_int(raw.get("shard_count"), "shard_count")
        if shard_count <= 0:
            raise ValueError("PLE NVMe manifest must define a positive shard_count")
        byte_size = _required_int(raw.get("byte_size"), "byte_size")
        if byte_size != rows * columns:
            raise ValueError(
                "PLE NVMe manifest byte_size must equal rows multiplied by columns "
                f"for FP8 data; got {byte_size} versus {rows * columns}"
            )
        actual_size = data_path.stat().st_size
        if actual_size != byte_size:
            raise ValueError(
                f"PLE NVMe file size is {actual_size} bytes, expected {byte_size}"
            )
        return cls(
            version=version,
            dtype=dtype,
            rows=rows,
            columns=columns,
            shard_count=shard_count,
            byte_size=byte_size,
        )


def map_ple_weight(
    data_path: str | os.PathLike[str],
    expected_shape: tuple[int, int],
) -> torch.Tensor:
    """Map an FP8 PLE table privately and validate its runtime shape."""
    path = Path(data_path)
    if not path.is_absolute():
        raise ValueError(f"VLLM_PLE_NVME_PATH must be absolute, got {path}")
    manifest = PleNvmeManifest.load(path)
    actual_shape = (manifest.rows, manifest.columns)
    if actual_shape != expected_shape:
        raise ValueError(
            f"PLE runtime expected {expected_shape}, but NVMe sidecar contains "
            f"{actual_shape}"
        )
    mapped_bytes = torch.from_file(
        str(path),
        shared=False,
        size=manifest.byte_size,
        dtype=torch.uint8,
    )
    return mapped_bytes.view(torch.float8_e4m3fn).reshape(actual_shape)


def validate_ple_nvme_config(
    data_path: str | None,
    cpu_offload_enabled: bool,
) -> Path | None:
    """Validate the environment combination before model initialization."""
    if not data_path:
        return None
    if not cpu_offload_enabled:
        raise ValueError(
            "VLLM_PLE_NVME_PATH requires VLLM_PLE_CPU_OFFLOAD=1 so the "
            "file-backed table is opened only by the PLE CPU worker"
        )
    path = Path(data_path)
    if not path.is_absolute():
        raise ValueError(f"VLLM_PLE_NVME_PATH must be absolute, got {path}")
    PleNvmeManifest.load(path)
    return path
