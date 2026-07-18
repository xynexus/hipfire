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
from aie.iron.controlflow import range_  # noqa: E402
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


# source_files=[KERNEL_SRC] folds the .cc mtime into the JIT cache key so that
# editing the kernel reliably invalidates ~/.npu/cache (the ExternalFunction is
# created inside the generator, after the cache hash is computed, so without this
# the design hash would not track .cc edits — the JIT would reuse a stale xclbin).
@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"], source_files=[str(KERNEL_SRC)])
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


# ── all-KV-head streaming variant: ONE dispatch for the whole layer's attention ──
# The core loops `n_iters` (= n_kv) times, acquiring one kv-head's Q-group / KV /
# O-group per iteration through the ObjectFifos, so the tile only ever holds ONE
# iteration (56 KB at groups=4, kv_len=48) while all 8 groups stream through.
# Dispatches/layer: n_kv (8) -> 1. Same kernel, same math as run_attn_head.
@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"], source_files=[str(KERNEL_SRC)])
def dflash_attn_all(Q: In, KV: In, O: Out, *,
                    q_len: CompileTime[int], kv_len: CompileTime[int],
                    n_iters: CompileTime[int]):
    nd = cast(Any, np.ndarray)
    dt = cast(Any, np.dtype)
    qN, kvN = q_len * HEAD_DIM, 2 * kv_len * HEAD_DIM
    # fifo element = ONE iteration; runtime buffers hold all n_iters back to back.
    Qt_ty: Any = nd[(qN,), dt[bfloat16]]
    KVt_ty: Any = nd[(kvN,), dt[bfloat16]]
    Q_all: Any = nd[(n_iters * qN,), dt[bfloat16]]
    KV_all: Any = nd[(n_iters * kvN,), dt[bfloat16]]
    attn = _attn_kernel(q_len, kv_len)

    inQ = ObjectFifo(Qt_ty, name="inQ", depth=1)
    memQ = inQ.cons().forward(name="memQ")
    inKV = ObjectFifo(KVt_ty, name="inKV", depth=1)
    memKV = inKV.cons().forward(name="memKV")
    outO = ObjectFifo(Qt_ty, name="outO", depth=1)

    def core_fn(of_q, of_kv, of_o, kfn):
        for _ in range_(n_iters):
            q = of_q.acquire(1)
            kv = of_kv.acquire(1)
            o = of_o.acquire(1)
            kfn(q, kv, o)
            of_q.release(1)
            of_kv.release(1)
            of_o.release(1)

    worker = Worker(core_fn, [memQ.cons(), memKV.cons(), outO.prod(), attn],
                    stack_size=0x1000)

    from aie.helpers.taplib import TensorTiler2D
    # stream n_iters tiles of one iteration each
    q_tap = TensorTiler2D.group_tiler((n_iters, qN), (1, qN), (n_iters, 1), prune_step=False)[0]
    kv_tap = TensorTiler2D.group_tiler((n_iters, kvN), (1, kvN), (n_iters, 1), prune_step=False)[0]
    o_tap = TensorTiler2D.group_tiler((n_iters, qN), (1, qN), (n_iters, 1), prune_step=False)[0]

    rt = Runtime()
    with rt.sequence(Q_all, KV_all, Q_all) as (Qh, KVh, Oh):
        rt.start(worker)
        rt.fill(inQ.prod(), Qh, tap=q_tap)
        rt.fill(inKV.prod(), KVh, tap=kv_tap)
        rt.drain(outO.cons(), Oh, tap=o_tap, wait=True)
    return Program(iron.get_current_device(), rt).resolve_program()


def run_attn_all_kv(q, k, v, groups):
    """ONE dispatch for a whole layer's attention.

    q [B,NH,HD], k/v [tot,NKV,HD] -> ctx [B, NH*HD]. Packs, per kv-head, the
    `groups` q-heads' queries stacked (q_len=groups*B) and that kv-head's [K|V],
    then streams all NKV groups through a single dispatch.
    """
    from aie.utils.benchmark import run_iters
    B, NH, HD = q.shape
    tot = k.shape[0]
    NKV = NH // groups
    q_len = groups * B
    qbuf, kvbuf = [], []
    for kvh in range(NKV):
        heads = range(kvh * groups, (kvh + 1) * groups)
        qbuf.append(np.concatenate([q[:, h, :] for h in heads], axis=0).reshape(-1))
        kvbuf.append(np.concatenate([k[:, kvh, :].reshape(-1), v[:, kvh, :].reshape(-1)]))
    Qt = iron.tensor(np.concatenate(qbuf).astype(bfloat16), dtype=bfloat16, device="npu")
    KVt = iron.tensor(np.concatenate(kvbuf).astype(bfloat16), dtype=bfloat16, device="npu")
    Ot = iron.zeros(NKV * q_len * HEAD_DIM, dtype=bfloat16, device="npu")
    run_iters(dflash_attn_all, Qt, KVt, Ot, q_len=q_len, kv_len=tot, n_iters=NKV,
              warmup=0, iters=1)
    o = np.array(Ot.numpy(), dtype=np.float32).reshape(NKV, q_len, HEAD_DIM)
    ctx = np.empty((B, NH * HD), np.float32)
    for kvh in range(NKV):
        for i, h in enumerate(range(kvh * groups, (kvh + 1) * groups)):
            ctx[:, h * HD:(h + 1) * HD] = o[kvh, i * B:(i + 1) * B, :]
    return ctx


def run_attn_head(Qh_np, Kh_np, Vh_np, q_len, kv_len):
    """Run one head on the NPU. Q[q_len,128], K/V[kv_len,128] bf16 -> O[q_len,128]."""
    from aie.utils.benchmark import run_iters
    kv = np.concatenate([Kh_np.reshape(-1), Vh_np.reshape(-1)]).astype(bfloat16)  # [K|V]
    Qt = iron.tensor(Qh_np.reshape(-1).astype(bfloat16), dtype=bfloat16, device="npu")
    KVt = iron.tensor(kv, dtype=bfloat16, device="npu")
    Ot = iron.zeros(q_len * HEAD_DIM, dtype=bfloat16, device="npu")
    run_iters(dflash_attn_head, Qt, KVt, Ot, q_len=q_len, kv_len=kv_len, warmup=0, iters=1)
    return np.array(Ot.numpy().reshape(q_len, HEAD_DIM), dtype=np.float32)
