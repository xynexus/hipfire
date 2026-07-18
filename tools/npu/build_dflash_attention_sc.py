#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
"""Build the single-core DFlash non-causal cross-attention (one head/dispatch).

Fits nix1's npu1 (4 columns) — the 8-col segmented_attention is aie2p-only.
Reuses the @iron.jit + ObjectFifo pattern proven on nix1 by oq_gemm_design.
One dispatch = one (q_head): Q[q_len,128] K[kv_len,128] V[kv_len,128] -> O[q_len,128],
plain bf16 row-major. Host loops heads (GQA: feeds K/V for q_head//groups).
Kernel: dflash_attention_sc_bf16.cc (q_len/kv_len baked via -D).

Usage (JIT + run happens in test_dflash_attention_npu.py via run_attn()).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, cast

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import oq_gemm_design as design  # noqa: E402 — sets device/pyxrt; provides iron env

import aie.iron as iron  # noqa: E402
from aie.iron import (  # noqa: E402
    CompileTime, In, ObjectFifo, Out, Program, Runtime, Worker, ExternalFunction,
)

HEAD_DIM = 128
KERNEL_SRC = SCRIPT_DIR / "dflash_attention_sc_bf16.cc"
from ml_dtypes import bfloat16  # noqa: E402

_mlir_aie_pkg = next((Path(p) for p in sys.path if (Path(p) / "mlir_aie").is_dir()), None)
AIE_INCLUDE = _mlir_aie_pkg / "mlir_aie" / "include" if _mlir_aie_pkg else None


def _attn_kernel(q_len: int, kv_len: int):
    nd = cast(Any, np.ndarray)
    dt = cast(Any, np.dtype)
    q_ty: Any = nd[(q_len * HEAD_DIM,), dt[bfloat16]]
    kv2_ty: Any = nd[(2 * kv_len * HEAD_DIM,), dt[bfloat16]]  # [K | V]
    inc = [str(AIE_INCLUDE)] if AIE_INCLUDE and AIE_INCLUDE.is_dir() else []
    return ExternalFunction(
        name="dflash_attention_sc_bf16",
        source_file=str(KERNEL_SRC),
        arg_types=[q_ty, kv2_ty, q_ty],
        include_dirs=inc,
        compile_flags=[f"-DHIPFIRE_Q_LEN={q_len}", f"-DHIPFIRE_KV_LEN={kv_len}"],
    )


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def dflash_attn_head(Q: In, KV: In, O: Out, *,
                     q_len: CompileTime[int], kv_len: CompileTime[int]):
    nd = cast(Any, np.ndarray)
    dt = cast(Any, np.dtype)
    Q_ty: Any = nd[(q_len * HEAD_DIM,), dt[bfloat16]]
    KV_ty: Any = nd[(2 * kv_len * HEAD_DIM,), dt[bfloat16]]  # [K | V]
    attn = _attn_kernel(q_len, kv_len)

    # depth=1 (single-buffer): one head/dispatch, no pipelining; keeps the big
    # KV buffer from double-buffering past the 64 KB tile budget. Stage inputs
    # through the memtile (.forward) like oq_gemm_design — direct shim->core
    # delivery of Q did not land (output was invariant to the score compute).
    inQ = ObjectFifo(Q_ty, name="inQ", depth=1)
    memQ = inQ.cons().forward(name="memQ")
    inKV = ObjectFifo(KV_ty, name="inKV", depth=1)
    memKV = inKV.cons().forward(name="memKV")
    outO = ObjectFifo(Q_ty, name="outO", depth=1)

    def core_fn(of_q, of_kv, of_o, kfn):
        q = of_q.acquire(1)
        kv = of_kv.acquire(1)
        o = of_o.acquire(1)
        kfn(q, kv, o)
        of_q.release(1)
        of_kv.release(1)
        of_o.release(1)

    worker = Worker(core_fn, [memQ.cons(), memKV.cons(), outO.prod(), attn],
                    stack_size=0x1000)

    # Contiguous 1-D taps — WITHOUT a tap, rt.fill/drain transfer no data (the
    # core then runs on uninitialised tile memory). Treat each 1-D buffer as
    # (1, N) tiled whole.
    from aie.helpers.taplib import TensorTiler2D
    qN = q_len * HEAD_DIM
    kvN = 2 * kv_len * HEAD_DIM
    q_tap = TensorTiler2D.group_tiler((1, qN), (1, qN), (1, 1), prune_step=False)[0]
    kv_tap = TensorTiler2D.group_tiler((1, kvN), (1, kvN), (1, 1), prune_step=False)[0]
    o_tap = TensorTiler2D.group_tiler((1, qN), (1, qN), (1, 1), prune_step=False)[0]

    rt = Runtime()
    with rt.sequence(Q_ty, KV_ty, Q_ty) as (Qh, KVh, Oh):
        rt.start(worker)
        rt.fill(inQ.prod(), Qh, tap=q_tap)
        rt.fill(inKV.prod(), KVh, tap=kv_tap)
        rt.drain(outO.cons(), Oh, tap=o_tap, wait=True)
    return Program(iron.get_current_device(), rt).resolve_program()


def run_attn_head(Qh_np, Kh_np, Vh_np, q_len, kv_len):
    """Run one head on the NPU. Q[q_len,128], K/V[kv_len,128] bf16 -> O[q_len,128]."""
    from aie.utils.benchmark import run_iters
    kv = np.concatenate([Kh_np.reshape(-1), Vh_np.reshape(-1)]).astype(bfloat16)  # [K|V]
    Qt = iron.tensor(Qh_np.reshape(-1).astype(bfloat16), dtype=bfloat16, device="npu")
    KVt = iron.tensor(kv, dtype=bfloat16, device="npu")
    Ot = iron.zeros(q_len * HEAD_DIM, dtype=bfloat16, device="npu")
    run_iters(dflash_attn_head, Qt, KVt, Ot, q_len=q_len, kv_len=kv_len, warmup=0, iters=1)
    return np.array(Ot.numpy().reshape(q_len, HEAD_DIM), dtype=np.float32)
