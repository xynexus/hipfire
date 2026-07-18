#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
"""Build BATCHED variants of the DFlash-body primitive NPU kernels (Phase D
step-2, Lever 1: dispatch reduction).

The per-op primitive xclbins built by build_qwen35_{rmsnorm,headnorm,rope,swiglu}
process ONE row (rmsnorm/swiglu) or ONE token position (headnorm/rope) per
dispatch, so the unfused body loops the host once per row. Because _transform_gen
bakes N_div_n = num_elements/tile_size into the instruction stream, a batched
[rows, ·] buffer runs all rows in ONE dispatch — but only if the xclbin is built
with the batched num_elements. This builder emits those batched variants:

  qwen35-rmsnorm-<H>-b<rows>            weight tiled as a 2nd input (replicated
                                        per row), N_div_n = rows.
  qwen35-headnorm-<q|k>-<nh>h<hd>d-b<rows>   weight is a once-acquired param,
                                        N_div_n = rows*nh head-tiles.
  dflash-rope-<q|k>-<nh>h<hd>d-b<rows>  FULL-neox rope; cs is a 2nd TILED input
                                        (dflash_rope_batched_bf16.cc) so each
                                        head-tile carries its own position's cs.
                                        N_div_n = rows*nh.
  qwen35-swiglu-<I>-b<rows>            parallel silu(gate)*up over rows*I.
  dflash-hnrope-<q|k>-<nh>h<hd>d-b<rows>  Phase D step 1 FUSION: headnorm+rope in
                                        ONE dispatch (weight param + tiled cs);
                                        replaces the headnorm/rope pair.

Env (fork, nix1) — same as the body runner:
  PATH=/opt/xilinx/xrt/bin:$PATH
  LD_LIBRARY_PATH=~/.cache/hipfire-npu-deps/lib:/opt/xilinx/xrt/lib:$LD_LIBRARY_PATH
  PEANO_INSTALL_DIR=~/mlir-aie-312/venv312/lib/python3.12/site-packages/llvm-aie
  PYTHONPATH=~/mlir-aie-312/install/python:/opt/xilinx/xrt/python:$PYTHONPATH
  run with ~/mlir-aie-312/venv312/bin/python

Usage:
  build_dflash_body_batched.py --which rmsnorm16
  build_dflash_body_batched.py --which all    # every variant the 9B body needs
"""
import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

# ── venv bootstrap (mlir_aie headers live under ~/.venv) ─────────────────────
_VENV = Path.home() / ".venv" / "lib"
for _p in _VENV.glob("python*/site-packages"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

_XRT_BIN = Path("/opt/xilinx/xrt/bin")
if _XRT_BIN.is_dir() and str(_XRT_BIN) not in os.environ.get("PATH", ""):
    os.environ["PATH"] = str(_XRT_BIN) + os.pathsep + os.environ.get("PATH", "")

import numpy as np
from ml_dtypes import bfloat16
from aie.iron import ExternalFunction
try:
    # stock wheel (~/.venv, mlir-aie 1.3.1)
    from aie.iron.algorithms.transform import _transform_gen, transform_parallel_binary
except ModuleNotFoundError:
    # custom fork (~/mlir-aie-312) renamed the module to _transform; the
    # _transform_gen signature/semantics are the same in both.
    from aie.iron.algorithms._transform import _transform_gen, transform_parallel_binary
from aie.iron.device import NPU1, NPU2
from aie.utils import set_current_device, get_current_device
from aie.utils.compile import compile_mlir_module, compile_external_kernel

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR.parent.parent / "target" / "npu"

_mlir_aie_pkg = next((Path(p) for p in sys.path if (Path(p) / "mlir_aie").is_dir()), None)
AIE_INCLUDE = _mlir_aie_pkg / "mlir_aie" / "include" if _mlir_aie_pkg else None

_NAME_TO_NPU = {
    "Phoenix": (NPU1, "aie2", "AIE2"), "npu1": (NPU1, "aie2", "AIE2"),
    "Strix": (NPU2, "aie2p", "AIE2P"), "Krackan": (NPU2, "aie2p", "AIE2P"),
    "npu4": (NPU2, "aie2p", "AIE2P"), "npu5": (NPU2, "aie2p", "AIE2P"),
    "npu6": (NPU2, "aie2p", "AIE2P"),
}


def _detect():
    import ctypes
    ctypes.CDLL("/opt/xilinx/xrt/lib/libxrt_coreutil.so.2", mode=ctypes.RTLD_GLOBAL)
    if "/opt/xilinx/xrt/python" not in sys.path:
        sys.path.insert(0, "/opt/xilinx/xrt/python")
    import pyxrt
    name = pyxrt.device(0).get_info(pyxrt.xrt_info_device.name)
    for sub, tpl in _NAME_TO_NPU.items():
        if sub in name:
            return tpl
    return _NAME_TO_NPU["npu1"]


def _incs(extra=None):
    d = []
    if AIE_INCLUDE and AIE_INCLUDE.is_dir():
        d.append(str(AIE_INCLUDE))
    if extra and extra.is_dir():
        d.append(str(extra))
    return d


def _emit(mlir_module, kernel, xclbin_path, instr_path, target_arch):
    with tempfile.TemporaryDirectory(prefix="hipfire_npu_batch_") as tmp:
        tmp_path = Path(tmp)
        tmp_xclbin = tmp_path / "final.xclbin"
        tmp_instr = tmp_path / "insts.bin"
        compile_external_kernel(kernel, tmp_path, target_arch=target_arch)
        compile_mlir_module(mlir_module=mlir_module, insts_path=tmp_instr,
                            xclbin_path=tmp_xclbin, work_dir=tmp_path)
        shutil.copy2(tmp_xclbin, xclbin_path)
        shutil.copy2(tmp_instr, instr_path)
    print(f"  xclbin: {xclbin_path.stat().st_size} B  instr: {instr_path.stat().st_size} B")


def build_rmsnorm(H, rows, target_arch):
    stem = f"qwen35-rmsnorm-{H}-b{rows}"
    print(f"[rmsnorm] H={H} rows={rows} N_div_n={rows} -> {stem}")
    tile_ty: Any = np.ndarray[(H,), np.dtype[bfloat16]]
    kernel = ExternalFunction(name="rms_norm_weighted_bf16",
                              source_file=str(SCRIPT_DIR / "rms_norm_weighted_bf16.cc"),
                              arg_types=[tile_ty, tile_ty, tile_ty, np.int32],
                              include_dirs=_incs())
    in_buf = np.zeros(rows * H, dtype=bfloat16)
    w_buf = np.zeros(rows * H, dtype=bfloat16)
    out_buf = np.zeros(rows * H, dtype=bfloat16)
    mlir = _transform_gen(kernel, [in_buf, w_buf], out_buf, tile_size=H)
    _emit(mlir, kernel, OUT_DIR / f"{stem}.xclbin", OUT_DIR / f"{stem}-instr.bin", target_arch)


def build_headnorm(label, nh, hd, rows, target_arch):
    stem = f"qwen35-headnorm-{label}-{nh}h{hd}d-b{rows}"
    print(f"[headnorm] {label} nh={nh} hd={hd} rows={rows} N_div_n={rows*nh} -> {stem}")
    tile_ty: Any = np.ndarray[(hd,), np.dtype[bfloat16]]
    weight_ty: Any = np.ndarray[(hd,), np.dtype[bfloat16]]
    kernel = ExternalFunction(name="rms_norm_head_bf16",
                              source_file=str(SCRIPT_DIR / "rms_norm_head_bf16.cc"),
                              arg_types=[tile_ty, tile_ty, weight_ty, np.int32],
                              include_dirs=_incs())
    qk = np.zeros(rows * nh * hd, dtype=bfloat16)
    out = np.zeros(rows * nh * hd, dtype=bfloat16)
    w = np.zeros(hd, dtype=bfloat16)
    mlir = _transform_gen(kernel, [qk], out, w, tile_size=hd)
    _emit(mlir, kernel, OUT_DIR / f"{stem}.xclbin", OUT_DIR / f"{stem}-instr.bin", target_arch)


def build_rope(label, nh, hd, rows, target_arch):
    stem = f"dflash-rope-{label}-{nh}h{hd}d-b{rows}"
    print(f"[rope] {label} nh={nh} hd={hd} rows={rows} N_div_n={rows*nh} -> {stem}")
    tile_ty: Any = np.ndarray[(hd,), np.dtype[bfloat16]]
    kernel = ExternalFunction(name="dflash_rope_batched_bf16",
                              source_file=str(SCRIPT_DIR / "dflash_rope_batched_bf16.cc"),
                              arg_types=[tile_ty, tile_ty, tile_ty, np.int32],
                              include_dirs=_incs())
    qk = np.zeros(rows * nh * hd, dtype=bfloat16)
    cs = np.zeros(rows * nh * hd, dtype=bfloat16)
    out = np.zeros(rows * nh * hd, dtype=bfloat16)
    mlir = _transform_gen(kernel, [qk, cs], out, tile_size=hd)
    _emit(mlir, kernel, OUT_DIR / f"{stem}.xclbin", OUT_DIR / f"{stem}-instr.bin", target_arch)


def build_headnorm_rope(label, nh, hd, rows, target_arch):
    """Phase D step 1: FUSED per-head rmsnorm + FULL-neox rope in ONE dispatch.

    Replaces the qwen35-headnorm-* + dflash-rope-* pair for q and k.

    TWO heads per tile (tile_size = 2*hd): a core tile has only 2 input DMA
    channels, so tiled-qk + tiled-cs + a weight PARAM (3 inbound) will not
    place. Widening the tile lets the norm gamma and the row's cs share one
    stream as [gamma(hd) | cs(hd)] — valid because gamma is shared by all heads
    and both heads of a pair sit in the same row (nh even), hence same position.
    """
    if nh % 2:
        raise ValueError(f"n_heads must be even for head-pair tiling, got {nh}")
    stem = f"dflash-hnrope-{label}-{nh}h{hd}d-b{rows}"
    tile = 2 * hd
    print(f"[hnrope] {label} nh={nh} hd={hd} rows={rows} tile={tile} "
          f"N_div_n={rows*nh//2} -> {stem}")
    tile_ty: Any = np.ndarray[(tile,), np.dtype[bfloat16]]
    kernel = ExternalFunction(
        name="dflash_headnorm_rope_batched_bf16",
        source_file=str(SCRIPT_DIR / "dflash_headnorm_rope_batched_bf16.cc"),
        arg_types=[tile_ty, tile_ty, tile_ty, np.int32],
        include_dirs=_incs())
    qk = np.zeros(rows * nh * hd, dtype=bfloat16)
    coeff = np.zeros(rows * nh * hd, dtype=bfloat16)
    out = np.zeros(rows * nh * hd, dtype=bfloat16)
    mlir = _transform_gen(kernel, [qk, coeff], out, tile_size=tile)
    _emit(mlir, kernel, OUT_DIR / f"{stem}.xclbin", OUT_DIR / f"{stem}-instr.bin", target_arch)


def build_swiglu(I, rows, target_arch, runtime_subdir, tile_size=16):
    stem = f"qwen35-swiglu-{I}-b{rows}"
    dev = get_current_device()
    n = tile_size * dev.cols
    total = rows * I
    if total % n != 0:
        raise ValueError(f"rows*I={total} not divisible by tile*cols={n}")
    print(f"[swiglu] I={I} rows={rows} total={total} -> {stem}")
    runtime_lib = _mlir_aie_pkg / "mlir_aie" / "aie_runtime_lib" / runtime_subdir if _mlir_aie_pkg else None
    tile_ty: Any = np.ndarray[(tile_size,), np.dtype[bfloat16]]
    kernel = ExternalFunction(name="silu_mul_bf16",
                              source_file=str(SCRIPT_DIR / "silu_mul_bf16.cc"),
                              arg_types=[tile_ty, tile_ty, tile_ty, np.int32],
                              include_dirs=_incs(runtime_lib))
    import inspect
    if "tensor_ty" in inspect.signature(transform_parallel_binary).parameters:
        tensor_ty: Any = np.ndarray[(total,), np.dtype[bfloat16]]
        mlir = transform_parallel_binary(kernel, tensor_ty, tile_size=tile_size)
    else:
        g = np.zeros(total, dtype=bfloat16)
        u = np.zeros(total, dtype=bfloat16)
        o = np.zeros(total, dtype=bfloat16)
        mlir = transform_parallel_binary(kernel, g, u, o, tile_size=tile_size)
    _emit(mlir, kernel, OUT_DIR / f"{stem}.xclbin", OUT_DIR / f"{stem}-instr.bin", target_arch)


# The exact batched variants the z-lab Qwen3.5-9B DFlash body needs:
#   B=16 block rows, L=32 ctx (thp), tot=48, H=4096, I=12288, NH=32, NKV=8, hd=128.
VARIANTS = {
    "rmsnorm16": lambda ta, rs: build_rmsnorm(4096, 16, ta),   # xn, xn2, final
    "rmsnorm32": lambda ta, rs: build_rmsnorm(4096, 32, ta),   # thp
    "headnorm_q16": lambda ta, rs: build_headnorm("q", 32, 128, 16, ta),
    "headnorm_k48": lambda ta, rs: build_headnorm("k", 8, 128, 48, ta),
    "rope_q16": lambda ta, rs: build_rope("q", 32, 128, 16, ta),
    "rope_k48": lambda ta, rs: build_rope("k", 8, 128, 48, ta),
    "swiglu16": lambda ta, rs: build_swiglu(12288, 16, ta, rs),
    # Phase D step 1: fused headnorm+rope (supersedes headnorm_* + rope_* pairs).
    "hnrope_q16": lambda ta, rs: build_headnorm_rope("q", 32, 128, 16, ta),
    "hnrope_k48": lambda ta, rs: build_headnorm_rope("k", 8, 128, 48, ta),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--which", default="all", help="variant key or 'all': " + ", ".join(VARIANTS))
    args = ap.parse_args()

    device_cls, target_arch, runtime_subdir = _detect()
    set_current_device(device_cls())
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    keys = list(VARIANTS) if args.which == "all" else [args.which]
    for k in keys:
        if k not in VARIANTS:
            raise SystemExit(f"unknown --which {k!r}; choices: {', '.join(VARIANTS)}")
        VARIANTS[k](target_arch, runtime_subdir)
    print("[build_dflash_body_batched] done:", ", ".join(keys))


if __name__ == "__main__":
    main()
