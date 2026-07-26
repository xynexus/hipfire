#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# hipfire — see LICENSE and NOTICE in the project root.
"""One source of truth for IRON/XDNA target selection.

The cost harness runs on both XDNA/NPU1 (AIE2) and XDNA2/NPU2 (AIE2P).
Keeping the IRON device, Peano target, runtime library, topology, and cache
namespace together prevents a cached AIE2 image from being reused on AIE2P.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from . import env


@dataclass(frozen=True)
class Target:
    key: str
    tile_isa: str
    target_arch: str
    compute_columns: int
    compute_cores: int

    @property
    def runtime_library_name(self) -> str:
        return self.tile_isa

    @property
    def cache_tag(self) -> str:
        return f"{self.key}-{self.target_arch}"

    def iron_device(self):
        from aie.iron.device import NPU1, NPU2

        return NPU1() if self.key == "npu1" else NPU2()


TARGETS = {
    "npu1": Target("npu1", "AIE2", "aie2", 4, 16),
    "npu2": Target("npu2", "AIE2P", "aie2p", 8, 32),
}


def infer_device_key(name: str, architecture: str = "") -> str:
    """Map XRT/pyxrt identities to the mlir-aie NPU generation."""
    identity = f"{name} {architecture}".lower()
    if any(token in identity for token in ("aie2p", "npu2", "npu4", "npu5", "npu6", "strix", "krackan")):
        return "npu2"
    if any(token in identity for token in ("npu1", "phoenix", "hawk point", "aie2")):
        return "npu1"
    raise ValueError(f"unsupported NPU identity: {name!r} architecture={architecture!r}")


def detect_device_key() -> str:
    """Detect the installed NPU and fail closed on an unknown generation."""
    env.bootstrap()
    try:
        import pyxrt

        name = pyxrt.device(0).get_info(pyxrt.xrt_info_device.name)
        return infer_device_key(str(name))
    except (ImportError, AttributeError, RuntimeError, ValueError):
        smi = env.xrt_bin("xrt-smi")
        result = subprocess.run(
            [str(smi), "examine"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        table = re.search(r"\|\[[^]]+\]\s*\|([^|]+)\|([^|]+)\|", result.stdout)
        if table:
            return infer_device_key(table.group(1).strip(), table.group(2).strip())
        raise ValueError("unable to detect a supported XDNA NPU from pyxrt or xrt-smi")


def resolve_target(device: str = "auto") -> Target:
    key = detect_device_key() if device == "auto" else device.lower()
    try:
        return TARGETS[key]
    except KeyError as error:
        raise ValueError(f"unsupported NPU target {device!r}; expected auto, npu1, or npu2") from error


def runtime_include(mlir_package: Path | None, target: Target) -> Path | None:
    if mlir_package is None:
        return None
    return mlir_package / "mlir_aie" / "aie_runtime_lib" / target.runtime_library_name


def include_dirs(mlir_package: Path | None, target: Target) -> list[str]:
    if mlir_package is None:
        return []
    return [
        str(mlir_package / "mlir_aie" / "include"),
        str(runtime_include(mlir_package, target)),
    ]


def resolve_program(program):
    """Resolve across IRON releases before/after explicit placers were removed."""
    try:
        from aie.iron.placers import SequentialPlacer
    except ImportError:
        return program.resolve_program()
    return program.resolve_program(SequentialPlacer())
