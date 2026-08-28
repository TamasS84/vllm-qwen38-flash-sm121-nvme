# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import struct
from pathlib import Path

import pytest

from tools.prepare_ple_nvme import build_ple_sidecar

_KEY_PREFIX = (
    "model.language_model.model.layers.0.ple.ple_embedding.ngram_embedding.shard_"
)


def _write_safetensors(
    path: Path,
    tensors: list[tuple[str, str, list[int], bytes]],
) -> None:
    header: dict[str, object] = {}
    payload = bytearray()
    for name, dtype, shape, data in tensors:
        start = len(payload)
        payload.extend(data)
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [start, len(payload)],
        }
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


def _write_model(
    model_dir: Path,
    tensors: list[tuple[int, str, list[int], bytes]],
    *,
    split_ngram_parts: int,
) -> None:
    model_dir.mkdir()
    filename = "model-plefp8-00000-of-00001.safetensors"
    safetensor_entries = [
        (f"{_KEY_PREFIX}{index}.weight", dtype, shape, data)
        for index, dtype, shape, data in tensors
    ]
    _write_safetensors(model_dir / filename, safetensor_entries)
    weight_map = {
        name: filename
        for name, _dtype, _shape, _data in sorted(
            safetensor_entries, key=lambda item: item[0]
        )
    }
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map})
    )
    (model_dir / "config.json").write_text(
        json.dumps({"text_config": {"split_ngram_parts": split_ngram_parts}})
    )


def test_builds_sidecar_in_numeric_shard_order(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    tensors = [(index, "F8_E4M3", [1, 1], bytes([index])) for index in range(11)]
    _write_model(model_dir, tensors, split_ngram_parts=11)
    output_path = tmp_path / "ple.fp8"

    manifest = build_ple_sidecar(model_dir, output_path)

    assert output_path.read_bytes() == bytes(range(11))
    assert manifest == {
        "version": 1,
        "dtype": "F8_E4M3",
        "rows": 11,
        "columns": 1,
        "shard_count": 11,
        "byte_size": 11,
    }
    assert json.loads((tmp_path / "ple.fp8.json").read_text()) == manifest
    assert not (tmp_path / "ple.fp8.tmp").exists()


def test_reuses_complete_valid_sidecar(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    _write_model(
        model_dir,
        [(0, "F8_E4M3", [2, 1], b"\x01\x02")],
        split_ngram_parts=1,
    )
    output_path = tmp_path / "ple.fp8"
    expected = build_ple_sidecar(model_dir, output_path)
    original_mtime = output_path.stat().st_mtime_ns

    actual = build_ple_sidecar(model_dir, output_path)

    assert actual == expected
    assert output_path.read_bytes() == b"\x01\x02"
    assert output_path.stat().st_mtime_ns == original_mtime


def test_rejects_missing_numeric_shard_index(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    _write_model(
        model_dir,
        [
            (0, "F8_E4M3", [1, 1], b"\x00"),
            (2, "F8_E4M3", [1, 1], b"\x02"),
        ],
        split_ngram_parts=3,
    )

    with pytest.raises(ValueError, match="contiguous PLE shard indices"):
        build_ple_sidecar(model_dir, tmp_path / "ple.fp8")


def test_rejects_non_fp8_shard(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    _write_model(
        model_dir,
        [(0, "F16", [1, 1], b"\x00\x00")],
        split_ngram_parts=1,
    )

    with pytest.raises(ValueError, match="must use F8_E4M3"):
        build_ple_sidecar(model_dir, tmp_path / "ple.fp8")


def test_rejects_inconsistent_column_count(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    _write_model(
        model_dir,
        [
            (0, "F8_E4M3", [1, 1], b"\x00"),
            (1, "F8_E4M3", [1, 2], b"\x01\x02"),
        ],
        split_ngram_parts=2,
    )

    with pytest.raises(ValueError, match="same column count"):
        build_ple_sidecar(model_dir, tmp_path / "ple.fp8")


def test_rejects_byte_range_that_disagrees_with_shape(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    _write_model(
        model_dir,
        [(0, "F8_E4M3", [2, 2], b"\x00\x01\x02")],
        split_ngram_parts=1,
    )

    with pytest.raises(ValueError, match="byte range"):
        build_ple_sidecar(model_dir, tmp_path / "ple.fp8")
