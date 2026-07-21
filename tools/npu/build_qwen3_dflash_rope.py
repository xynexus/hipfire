#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
"""Compile the FULL-rotation NPU RoPE kernel for the DFlash drafter (Qwen3).

Unlike `build_qwen35_rope.py` (Qwen3.5 partial rotary, n_rot=head_dim/4), the
DFlash draft rotates the ENTIRE head_dim (neox / half-split), matching the
runtime `rope_batched.hip` the F16 golden was validated against. Kernel source:
`dflash_rope_bf16.cc` (n_rot = head_dim).

Produces (Q and K):
  <out_dir>/dflash-rope-q-{n_heads}h{head_dim}d.xclbin  / -instr.bin
  <out_dir>/dflash-rope-k-{n_kv_heads}h{head_dim}d.xclbin / -instr.bin

cos/sin (half-split [cos_0..cos_{hd/2-1}, sin_0..sin_{hd/2-1}]) are built by the
caller at the sidecar's rope_theta (1e7). The kernel is theta-agnostic.

Usage:
  python tools/npu/build_qwen3_dflash_rope.py --n-heads 32 --n-kv-heads 8 --head-dim 128
"""
import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

_VENV = Path.home() / ".venv" / "lib"
for p in _VENV.glob("python*/site-packages"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

_XRT_BIN = Path("/opt/xilinx/xrt/bin")
if _XRT_BIN.is_dir() and str(_XRT_BIN) not in os.environ.get("PATH", ""):
    os.environ["PATH"] = str(_XRT_BIN) + os.pathsep + os.environ.get("PATH", "")

import numpy as np
from ml_dtypes import bfloat16
from aie.iron import ExternalFunction
from aie.iron.algorithms.transform import _transform_gen
from aie.iron.device import NPU1, NPU2
from aie.utils import set_current_device
from aie.utils.compile import compile_mlir_module, compile_external_kernel

SCRIPT_DIR = Path(__file__).resolve().parent
KERNEL_SRC = SCRIPT_DIR / "dflash_rope_bf16.cc"
REPO_ROOT = SCRIPT_DIR.parent.parent

_mlir_aie_pkg = next((Path(p) for p in sys.path if (Path(p) / "mlir_aie").is_dir()), None)
AIE_INCLUDE = _mlir_aie_pkg / "mlir_aie" / "include" if _mlir_aie_pkg else None

_NAME_TO_NPU = {"npu1": "npu1", "Phoenix": "npu1", "npu4": "npu2", "npu5": "npu2",
                "npu6": "npu2", "Strix": "npu2", "Krackan": "npu2"}
_NPU_DEVICES = {"npu1": (NPU1, "aie2"), "npu2": (NPU2, "aie2p")}


def detect_npu() -> str:
    import ctypes
    ctypes.CDLL("/opt/xilinx/xrt/lib/libxrt_coreutil.so.2", mode=ctypes.RTLD_GLOBAL)
    if "/opt/xilinx/xrt/python" not in sys.path:
        sys.path.insert(0, "/opt/xilinx/xrt/python")
    import pyxrt
    name = pyxrt.device(0).get_info(pyxrt.xrt_info_device.name)
    for substr, key in _NAME_TO_NPU.items():
        if substr in name:
            return key
    raise RuntimeError(f"Cannot map device {name!r}; pass --npu npu1|npu2.")


def build_one(label, n_total, n_heads_label, head_dim, out_dir, target_arch, device_cls):
    set_current_device(device_cls())
    if head_dim % 32 != 0:
        raise ValueError(f"head_dim={head_dim} must be a multiple of 32 (head_dim/2 % 16 == 0)")
    xclbin_path = out_dir / f"dflash-rope-{label}-{n_heads_label}h{head_dim}d.xclbin"
    instr_path = out_dir / f"dflash-rope-{label}-{n_heads_label}h{head_dim}d-instr.bin"
    print(f"  [{label}] n_total={n_total} head_dim={head_dim} n_rot={head_dim} "
          f"(full) N_div_n={n_total // head_dim} npu={target_arch}")

    tile_ty: Any = np.ndarray[(head_dim,), np.dtype[bfloat16]]
    cs_tile_ty: Any = np.ndarray[(head_dim,), np.dtype[bfloat16]]  # full: n_rot=head_dim
    include_dirs = [str(AIE_INCLUDE)] if AIE_INCLUDE and AIE_INCLUDE.is_dir() else []

    kernel = ExternalFunction(
        name="dflash_rope_bf16",
        source_file=str(KERNEL_SRC),
        arg_types=[tile_ty, tile_ty, cs_tile_ty, np.int32],
        include_dirs=include_dirs,
    )
    qk_buf = np.zeros(n_total, dtype=bfloat16)
    out_buf = np.zeros(n_total, dtype=bfloat16)
    cs_buf = np.zeros(head_dim, dtype=bfloat16)
    mlir_module = _transform_gen(kernel, [qk_buf], out_buf, cs_buf, tile_size=head_dim)

    with tempfile.TemporaryDirectory(prefix="hipfire_npu_build_") as tmp:
        tmp_path = Path(tmp)
        compile_external_kernel(kernel, tmp_path, target_arch=target_arch)
        compile_mlir_module(
            mlir_module=mlir_module,
            insts_path=tmp_path / "insts.bin",
            xclbin_path=tmp_path / "final.xclbin",
            work_dir=tmp_path,
        )
        shutil.copy2(tmp_path / "final.xclbin", xclbin_path)
        shutil.copy2(tmp_path / "insts.bin", instr_path)
    print(f"  xclbin: {xclbin_path.stat().st_size} bytes  instr: {instr_path.stat().st_size} bytes")


def build(n_heads, n_kv_heads, head_dim, out_dir, npu="auto"):
    key = detect_npu() if npu == "auto" else npu
    device_cls, target_arch = _NPU_DEVICES[key]
    out_dir.mkdir(parents=True, exist_ok=True)
    build_one("q", n_heads * head_dim, n_heads, head_dim, out_dir, target_arch, device_cls)
    build_one("k", n_kv_heads * head_dim, n_kv_heads, head_dim, out_dir, target_arch, device_cls)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-heads", type=int, default=32)
    p.add_argument("--n-kv-heads", type=int, default=8)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "target/npu")
    p.add_argument("--npu", choices=["auto", "npu1", "npu2"], default="auto")
    args = p.parse_args()
    build(args.n_heads, args.n_kv_heads, args.head_dim, args.out_dir, args.npu)


if __name__ == "__main__":
    main()
