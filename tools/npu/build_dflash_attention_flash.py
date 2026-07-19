#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
"""Flash-style (streaming-KV, mmul-tiled) DFlash cross-attention build.

Companion to `dflash_attention_flash_bf16.cc`. The original
`build_dflash_attention_sc.py` / `dflash_attention_sc_bf16.cc` pair is left
untouched as the working fallback.

Two structural differences from the sc kernel:

* **Streaming KV.** One dispatch iteration = one q-head; the kv-head's K/V is
  delivered as `n_tiles` fixed-size tiles of `kv_tile` rows, so core-tile L1
  holds one tile instead of the whole KV. That removes the `tot <= 55` cap
  (`memKV = 512*tot` against a 64 KiB tile) — core memory is now independent of
  `tot`. Softmax therefore has to run online (running max/sum, accumulator
  rescale per tile).
* **mmul tiling.** Both GEMMs go through `aie::mmul<4,8,4,bfloat16,bfloat16>`,
  so Q/K/V are pre-tiled host-side into the mm.cc block layout.

`kv_tile` must divide `tot` (no tail masking in v1) and be a multiple of 16.

MEASURED on nix1/npu1, 4 cores, block=16, 32q/8kv/128d, vs the sc kernel's
12.246 ms at tot=48 in the same session:

    config                    tot=48     tot=528    tot=4080
    q_len=16 kv_tile=48    1.214 ms    6.607 ms   48.342 ms
    ms / KV row              0.0253      0.0125      0.0118
    GFLOP/s                   10.37       20.95       22.12

sc kernel: 0.262 ms/row, 0.97 GFLOP/s, and no build at all past tot=55.
`q_len=32` and `kv_depth=2` were both measured null; `kv_tile=48` beats 16 by
~1.35x at long context. Recommended: q_len=16, kv_tile=48, kv_depth=1.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, cast

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import oq_gemm_design as design  # noqa: F401,E402 — bootstraps pyxrt/device for iron

import aie.iron as iron  # noqa: E402
from aie.iron.controlflow import range_  # noqa: E402
from aie.iron import (  # noqa: E402
    CompileTime, In, ObjectFifo, Out, Program, Runtime, Worker, ExternalFunction,
)
from ml_dtypes import bfloat16  # noqa: E402

HEAD_DIM = 128
MR, MS, MT = 4, 8, 4  # aie::mmul<r, s, t>
KERNEL_SRC = SCRIPT_DIR / "dflash_attention_flash_bf16.cc"

_mlir_aie_pkg = next((Path(p) for p in sys.path if (Path(p) / "mlir_aie").is_dir()), None)
AIE_INCLUDE = _mlir_aie_pkg / "mlir_aie" / "include" if _mlir_aie_pkg else None


# ── host-side mmul tiling helpers (mm.cc convention: tiles row-major, tiles
# themselves in row-major order) ──────────────────────────────────────────────
def pack_a(x: np.ndarray) -> np.ndarray:
    """[M, K] -> A-operand blocks of MR x MS."""
    m, k = x.shape
    return x.reshape(m // MR, MR, k // MS, MS).transpose(0, 2, 1, 3).reshape(-1)


def pack_b(x: np.ndarray) -> np.ndarray:
    """[K, N] -> B-operand blocks of MS x MT."""
    k, n = x.shape
    return x.reshape(k // MS, MS, n // MT, MT).transpose(0, 2, 1, 3).reshape(-1)


def unpack_c(v: np.ndarray, m: int, n: int) -> np.ndarray:
    """C-operand blocks of MR x MT -> [M, N]."""
    return v.reshape(m // MR, n // MT, MR, MT).transpose(0, 2, 1, 3).reshape(m, n)


def _flash_kernel(q_len: int, kv_tile: int, n_tiles: int):
    """Single entry point: two exported symbols from one .cc collide at link
    time (IRON recompiles the whole file per ExternalFunction)."""
    nd = cast(Any, np.ndarray)
    dt = cast(Any, np.dtype)
    q_ty: Any = nd[(q_len * HEAD_DIM,), dt[bfloat16]]
    kvt_ty: Any = nd[(2 * kv_tile * HEAD_DIM,), dt[bfloat16]]
    inc = [str(AIE_INCLUDE)] if AIE_INCLUDE and AIE_INCLUDE.is_dir() else []
    return ExternalFunction(
        name="dflash_flash_step",
        source_file=str(KERNEL_SRC),
        arg_types=[q_ty, kvt_ty, q_ty],
        include_dirs=inc,
        compile_flags=[f"-DHIPFIRE_Q_LEN={q_len}",
                       f"-DHIPFIRE_KV_TILE={kv_tile}",
                       f"-DHIPFIRE_N_TILES={n_tiles}"],
    )


# Multi-core: iterations (= q-heads) split across columns, one worker per column
# on row 2. Per column the shim uses 2 MM2S (Q, KV) + 1 S2MM (O), the same
# budget the sc `dflash_attn_mc` variant proved on npu1.
@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"], source_files=[str(KERNEL_SRC)])
def dflash_attn_flash_mc(Q: In, KV: In, O: Out, *,
                         q_len: CompileTime[int], kv_tile: CompileTime[int],
                         n_tiles: CompileTime[int], n_iters: CompileTime[int],
                         n_cores: CompileTime[int], kv_depth: CompileTime[int]):
    from aie.helpers.taplib import TensorAccessPattern
    from aie.iron.device import Tile

    nd = cast(Any, np.ndarray)
    dt = cast(Any, np.dtype)
    qN = q_len * HEAD_DIM
    kvtN = 2 * kv_tile * HEAD_DIM
    assert n_iters % n_cores == 0, f"n_iters={n_iters} not divisible by n_cores={n_cores}"
    per = n_iters // n_cores

    Qt_ty: Any = nd[(qN,), dt[bfloat16]]
    KVt_ty: Any = nd[(kvtN,), dt[bfloat16]]
    Q_all: Any = nd[(n_iters * qN,), dt[bfloat16]]
    KV_all: Any = nd[(n_iters * n_tiles * kvtN,), dt[bfloat16]]

    step = _flash_kernel(q_len, kv_tile, n_tiles)

    def core_fn(of_q, of_kv, of_o, kstep):
        for _ in range_(per):
            q = of_q.acquire(1)
            o = of_o.acquire(1)
            for _ in range_(n_tiles):
                kv = of_kv.acquire(1)
                kstep(q, kv, o)
                of_kv.release(1)
            of_q.release(1)
            of_o.release(1)

    workers, fills, drains = [], [], []
    for w in range(n_cores):
        inQ = ObjectFifo(Qt_ty, name=f"inQ{w}", depth=1)
        memQ = inQ.cons().forward(name=f"memQ{w}")
        inKV = ObjectFifo(KVt_ty, name=f"inKV{w}", depth=kv_depth)
        memKV = inKV.cons().forward(name=f"memKV{w}", depth=kv_depth)
        outO = ObjectFifo(Qt_ty, name=f"outO{w}", depth=1)
        workers.append(Worker(
            core_fn, [memQ.cons(), memKV.cons(), outO.prod(), step],
            tile=Tile(w % 4, 2 + w // 4), stack_size=0x1000))
        fills.append(("q", inQ.prod(), TensorAccessPattern(
            (n_iters, qN), w * per * qN, [1, 1, per, qN], [0, 0, qN, 1])))
        fills.append(("kv", inKV.prod(), TensorAccessPattern(
            (n_iters * n_tiles, kvtN), w * per * n_tiles * kvtN,
            [1, 1, per * n_tiles, kvtN], [0, 0, kvtN, 1])))
        drains.append((outO.cons(), TensorAccessPattern(
            (n_iters, qN), w * per * qN, [1, 1, per, qN], [0, 0, qN, 1])))

    rt = Runtime()
    with rt.sequence(Q_all, KV_all, Q_all) as (Qh, KVh, Oh):
        for wk in workers:
            rt.start(wk)
        for which, prod, tap in fills:
            rt.fill(prod, Qh if which == "q" else KVh, tap=tap)
        for consumer, tap in drains:
            rt.drain(consumer, Oh, tap=tap, wait=True)
    return Program(iron.get_current_device(), rt).resolve_program()


def pack_inputs(q, k, v, groups, q_len, kv_tile):
    """Pack host tensors into the flash kernel's tiled, per-iteration layout.

    q [B, NH, HD]; k/v [tot, NKV, HD]. One iteration covers `q_len` query rows
    (i.e. `q_len // B` q-heads) that share one kv-head, so the kv-head's K/V is
    repeated once per iteration that consumes it.
    """
    b, nh, hd = q.shape
    tot = k.shape[0]
    assert hd == HEAD_DIM
    assert q_len % b == 0, f"q_len={q_len} must be a multiple of block={b}"
    heads_per_iter = q_len // b
    assert groups % heads_per_iter == 0 or heads_per_iter % groups == 0
    assert tot % kv_tile == 0, f"kv_tile={kv_tile} must divide tot={tot}"
    n_tiles = tot // kv_tile

    qbuf, kvbuf = [], []
    for h0 in range(0, nh, heads_per_iter):
        heads = range(h0, h0 + heads_per_iter)
        kvh = h0 // groups
        assert all(h // groups == kvh for h in heads), "iteration spans kv-heads"
        qm = np.concatenate([q[:, h, :] for h in heads], axis=0)   # [q_len, HD]
        qbuf.append(pack_a(qm.astype(bfloat16)))
        for t in range(n_tiles):
            sl = slice(t * kv_tile, (t + 1) * kv_tile)
            kt = k[sl, kvh, :].astype(bfloat16)   # [kv_tile, HD]
            vt = v[sl, kvh, :].astype(bfloat16)
            kvbuf.append(pack_b(kt.T))            # Kᵀ: [HD, kv_tile]
            kvbuf.append(pack_b(vt))              # V : [kv_tile, HD]
    n_iters = nh // heads_per_iter
    return (np.concatenate(qbuf), np.concatenate(kvbuf), n_iters, n_tiles,
            heads_per_iter)


def unpack_output(o_flat, b, nh, q_len, heads_per_iter):
    """Tiled C-layout, per-iteration -> ctx [B, NH*HD]."""
    n_iters = nh // heads_per_iter
    o = np.array(o_flat, dtype=np.float32).reshape(n_iters, q_len * HEAD_DIM)
    ctx = np.empty((b, nh * HEAD_DIM), np.float32)
    for it in range(n_iters):
        m = unpack_c(o[it], q_len, HEAD_DIM)
        for i in range(heads_per_iter):
            h = it * heads_per_iter + i
            ctx[:, h * HEAD_DIM:(h + 1) * HEAD_DIM] = m[i * b:(i + 1) * b, :]
    return ctx


def run_attn_flash(q, k, v, groups, q_len=16, kv_tile=16, n_cores=4, kv_depth=2):
    """One whole layer's attention through the flash kernel."""
    b, nh, _ = q.shape
    qpk, kvpk, n_iters, n_tiles, hpi = pack_inputs(q, k, v, groups, q_len, kv_tile)
    Qt = iron.tensor(qpk.astype(bfloat16), dtype=bfloat16, device="npu")
    KVt = iron.tensor(kvpk.astype(bfloat16), dtype=bfloat16, device="npu")
    Ot = iron.zeros(n_iters * q_len * HEAD_DIM, dtype=bfloat16, device="npu")
    dflash_attn_flash_mc(Qt, KVt, Ot, q_len=q_len, kv_tile=kv_tile,
                         n_tiles=n_tiles, n_iters=n_iters, n_cores=n_cores,
                         kv_depth=kv_depth)
    return unpack_output(Ot.numpy(), b, nh, q_len, hpi)
