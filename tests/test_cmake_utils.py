# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import shutil
import subprocess
import sys
from pathlib import Path

import regex as re


def _get_cmake_bin() -> str:
    cmake = shutil.which("cmake")
    if cmake:
        return cmake
    venv_cmake = Path(sys.executable).parent / "cmake"
    if venv_cmake.is_file():
        return str(venv_cmake)
    return "cmake"


def test_exact_family_arch_precedes_generic_family_fallback(tmp_path: Path):
    repo_root = Path(__file__).parents[1]
    script = tmp_path / "test_cuda_archs.cmake"
    script.write_text(
        f"""
cmake_minimum_required(VERSION 3.26)
include("{repo_root / "cmake" / "utils.cmake"}")
cuda_archs_loose_intersection(
  actual "10.0f;10.7f" "10.7")
if(NOT "${{actual}}" STREQUAL "10.7f")
  message(FATAL_ERROR "Expected 10.7f, got '${{actual}}'")
endif()
"""
    )

    subprocess.run([_get_cmake_bin(), "-P", script], check=True)


def test_extract_archs_prefers_sass_target_over_corrupted_virtual_arch(
    tmp_path: Path,
):
    """torch's autodetection can emit a bogus arch=compute_* half (e.g.
    capability 12.1 corrupted to arch=compute_20,code=sm_121); the SASS
    target must win, while PTX-only entries keep the virtual arch."""
    repo_root = Path(__file__).parents[1]
    script = tmp_path / "test_extract_archs.cmake"
    script.write_text(
        f"""
cmake_minimum_required(VERSION 3.26)
include("{repo_root / "cmake" / "utils.cmake"}")
extract_unique_cuda_archs_ascending(actual
  "-gencode arch=compute_20,code=sm_121;\
-gencode arch=compute_80,code=sm_80;\
-gencode arch=compute_80,code=compute_80")
if(NOT "${{actual}}" STREQUAL "8.0;12.1")
  message(FATAL_ERROR "Expected '8.0;12.1', got '${{actual}}'")
endif()
"""
    )

    subprocess.run([_get_cmake_bin(), "-P", script], check=True)


def test_clear_cuda_gencode_flags(tmp_path: Path):
    repo_root = Path(__file__).parents[1]
    script = tmp_path / "test_clear_flags.cmake"
    script.write_text(
        f"""
cmake_minimum_required(VERSION 3.26)
include("{repo_root / "cmake" / "utils.cmake"}")
set(CMAKE_CUDA_FLAGS "-Wall -gencode arch=compute_80,code=sm_80")
clear_cuda_gencode_flags(CUDA_ARCH_FLAGS)
if(NOT "${{CMAKE_CUDA_FLAGS}}" STREQUAL "-Wall ")
  message(FATAL_ERROR "Expected '-Wall ', got '${{CMAKE_CUDA_FLAGS}}'")
endif()
if(NOT "${{CUDA_ARCH_FLAGS}}" STREQUAL "-gencode arch=compute_80,code=sm_80")
  message(FATAL_ERROR "Expected '-gencode arch=compute_80,code=sm_80', "
    "got '${{CUDA_ARCH_FLAGS}}'")
endif()
"""
    )

    subprocess.run([_get_cmake_bin(), "-P", script], check=True)


def test_cuda13_supports_native_sm121_target(tmp_path: Path):
    repo_root = Path(__file__).parents[1]
    cmake_lists = (repo_root / "CMakeLists.txt").read_text(encoding="utf-8")
    cuda13_4_branch = cmake_lists.split(
        "CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.4",
        maxsplit=1,
    )[1].split(
        "CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.0",
        maxsplit=1,
    )[0]
    cuda13_branch = cmake_lists.split(
        "CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.0",
        maxsplit=1,
    )[1].split(
        "CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.8",
        maxsplit=1,
    )[0]

    supported_archs = [
        re.search(r'set\(CUDA_SUPPORTED_ARCHS "([^"]+)"\)', branch).group(1)
        for branch in (cuda13_4_branch, cuda13_branch)
    ]
    script = tmp_path / "test_cuda13_sm121.cmake"
    script.write_text(
        f"""
cmake_minimum_required(VERSION 3.26)
include("{repo_root / "cmake" / "utils.cmake"}")
cuda_archs_loose_intersection(cuda13_4_actual "12.1a" "{supported_archs[0]}")
cuda_archs_loose_intersection(cuda13_actual "12.1a" "{supported_archs[1]}")
if(NOT "${{cuda13_4_actual}}" STREQUAL "12.1a")
  message(FATAL_ERROR "CUDA 13.4 lost 12.1a: '${{cuda13_4_actual}}'")
endif()
if(NOT "${{cuda13_actual}}" STREQUAL "12.1a")
  message(FATAL_ERROR "CUDA 13.0 lost 12.1a: '${{cuda13_actual}}'")
endif()
"""
    )

    subprocess.run([_get_cmake_bin(), "-P", script], check=True)
