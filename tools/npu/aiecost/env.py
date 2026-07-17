#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# hipfire — see LICENSE and NOTICE in the project root.
"""Environment bootstrap for NPU work on this box.

Importing this module makes the bare interpreter able to reach the NPU. Three
things must happen, in order, before `aie.utils` is imported:

  1. libxrt_coreutil preloaded RTLD_GLOBAL (weak-vtable ordering under XRT 2.25)
  2. pyxrt on sys.path (/opt/xilinx/xrt/python) — otherwise aie.utils silently
     falls back to CPUOnlyTensor and device="npu" is rejected
  3. boost on LD_LIBRARY_PATH for xclbinutil, set before aiecc spawns

`aiecost` never reaches the network; this only touches the local toolchain.
Extracted from tools/npu/oq_gemm_design.py so the calibration suite and the
existing build scripts agree on one bootstrap.
"""

import ctypes
import os
import sys
from pathlib import Path

_XRT_ROOT = Path("/opt/xilinx/xrt")
_XRT_LIB = _XRT_ROOT / "lib"
_XRT_BIN = _XRT_ROOT / "bin"
_XRT_PY = _XRT_ROOT / "python"
_BOOST_DEP = Path.home() / ".cache" / "hipfire-npu-deps" / "lib"

_ready = False


def bootstrap() -> None:
    """Idempotently prepare the process to talk to the NPU."""
    global _ready
    if _ready:
        return

    # 3. boost + XRT libs on LD_LIBRARY_PATH before any aiecc subprocess spawns.
    extra_ld = [str(p) for p in (_BOOST_DEP, _XRT_LIB) if p.is_dir()]
    if extra_ld:
        cur = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(extra_ld + ([cur] if cur else []))

    # xclbinutil lives in the XRT bin dir, which is not on PATH by default.
    # aiecc silently skips xclbin assembly if it cannot find it — no error.
    if _XRT_BIN.is_dir() and str(_XRT_BIN) not in os.environ.get("PATH", ""):
        os.environ["PATH"] = str(_XRT_BIN) + os.pathsep + os.environ.get("PATH", "")

    # venv site-packages (mlir_aie, ml_dtypes) when launched via a bare python.
    for p in (Path.home() / ".venv" / "lib").glob("python*/site-packages"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))

    # 1. coreutil preload, then 2. pyxrt on path — both before aie.utils imports.
    coreutil = _XRT_LIB / "libxrt_coreutil.so.2"
    if coreutil.exists():
        ctypes.CDLL(str(coreutil), mode=ctypes.RTLD_GLOBAL)
    if _XRT_PY.is_dir() and str(_XRT_PY) not in sys.path:
        sys.path.insert(0, str(_XRT_PY))

    _ready = True


def xrt_bin(tool: str) -> Path:
    return _XRT_BIN / tool
