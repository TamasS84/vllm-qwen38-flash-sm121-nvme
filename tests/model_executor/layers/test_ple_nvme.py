# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from pathlib import Path

import pytest
import torch

from vllm.model_executor.layers.ple_nvme import (
    PleNvmeManifest,
    map_ple_weight,
    validate_ple_nvme_config,
)


def _write_sidecar(
    data_path: Path,
    *,
    data: bytes = b"\x00\x01\x02\x03",
    version: int = 1,
    dtype: str = "F8_E4M3",
    rows: int = 2,
    columns: int = 2,
    shard_count: int = 1,
    byte_size: int = 4,
) -> None:
    data_path.write_bytes(data)
    data_path.with_name(f"{data_path.name}.json").write_text(
        json.dumps(
            {
                "version": version,
                "dtype": dtype,
                "rows": rows,
                "columns": columns,
                "shard_count": shard_count,
                "byte_size": byte_size,
            }
        )
    )


def test_maps_private_fp8_weight_from_absolute_path(tmp_path: Path) -> None:
    data_path = tmp_path / "ple.fp8"
    _write_sidecar(data_path)

    weight = map_ple_weight(data_path, expected_shape=(2, 2))
    with data_path.open("r+b") as target:
        target.write(b"\x04\x05\x06\x07")
        target.flush()

    assert weight.shape == (2, 2)
    assert weight.dtype == torch.float8_e4m3fn
    assert weight.device.type == "cpu"
    mapped_bytes = weight.view(torch.uint8).flatten()
    assert mapped_bytes.tolist() == [4, 5, 6, 7]

    mapped_bytes[0] = 9

    assert data_path.read_bytes() == b"\x04\x05\x06\x07"


def test_loads_typed_manifest(tmp_path: Path) -> None:
    data_path = tmp_path / "ple.fp8"
    _write_sidecar(data_path)

    manifest = PleNvmeManifest.load(data_path)

    assert manifest == PleNvmeManifest(
        version=1,
        dtype="F8_E4M3",
        rows=2,
        columns=2,
        shard_count=1,
        byte_size=4,
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"version": 2}, "format version"),
        ({"dtype": "F16"}, "F8_E4M3"),
        ({"rows": 0}, "positive rows"),
        ({"byte_size": 3}, "rows multiplied by columns"),
    ],
)
def test_rejects_malformed_manifest(
    tmp_path: Path,
    overrides: dict[str, object],
    message: str,
) -> None:
    data_path = tmp_path / "ple.fp8"
    _write_sidecar(data_path, **overrides)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match=message):
        PleNvmeManifest.load(data_path)


def test_rejects_missing_manifest(tmp_path: Path) -> None:
    data_path = tmp_path / "ple.fp8"
    data_path.write_bytes(b"\x00")

    with pytest.raises(ValueError, match="manifest does not exist"):
        PleNvmeManifest.load(data_path)


def test_rejects_data_file_size_mismatch(tmp_path: Path) -> None:
    data_path = tmp_path / "ple.fp8"
    _write_sidecar(data_path, data=b"\x00\x01\x02")

    with pytest.raises(ValueError, match="file size"):
        PleNvmeManifest.load(data_path)


def test_rejects_runtime_shape_mismatch(tmp_path: Path) -> None:
    data_path = tmp_path / "ple.fp8"
    _write_sidecar(data_path)

    with pytest.raises(ValueError, match=r"expected \(3, 2\).+contains \(2, 2\)"):
        map_ple_weight(data_path, expected_shape=(3, 2))


def test_rejects_relative_runtime_path(tmp_path: Path, monkeypatch) -> None:
    data_path = tmp_path / "ple.fp8"
    _write_sidecar(data_path)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="must be absolute"):
        map_ple_weight(Path("ple.fp8"), expected_shape=(2, 2))


def test_nvme_config_requires_cpu_offload(tmp_path: Path) -> None:
    data_path = tmp_path / "ple.fp8"
    _write_sidecar(data_path)

    with pytest.raises(ValueError, match="VLLM_PLE_CPU_OFFLOAD=1"):
        validate_ple_nvme_config(str(data_path), cpu_offload_enabled=False)


def test_nvme_config_accepts_complete_sidecar(tmp_path: Path) -> None:
    data_path = tmp_path / "ple.fp8"
    _write_sidecar(data_path)

    assert (
        validate_ple_nvme_config(str(data_path), cpu_offload_enabled=True) == data_path
    )
    assert validate_ple_nvme_config(None, cpu_offload_enabled=False) is None
