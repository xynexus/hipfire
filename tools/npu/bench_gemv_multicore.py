#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# hipfire — see LICENSE and NOTICE in the project root.
#
# Derived from the upstream mlir-aie matrix-vector IRON example
# (programming_examples/basic/matrix_multiplication/matrix_vector/matrix_vector.py,
# Copyright (C) 2023-2026 AMD, Apache-2.0 WITH LLVM-exception).
"""Decode-shape GEMV on the NPU, scaled across cores with SPREAD shim endpoints.

WHY THIS EXISTS: the open question in docs/npu/next-phase-goals.md is whether a
decode GEMV reaches the measured ~54.7 GB/s. The upstream example answers only
for ONE core (`n_cores = 1` is a local, not a parameter), and naively raising it
fails placement:

    no ShimNOCTile has sufficient DMA capacity for 0 input/1 output channels
    near centroid column 0

Read against this platform's constants — shim budget 16 in / 16 out over 8 shim
tiles, 2 each way — the device HAS the channels. The example simply lets every
endpoint bind near column 0. So this pins each core's shim endpoints to its own
column via `prod(tile=Tile(col=...))` / `cons(tile=Tile(col=...))`, which is the
whole difference between this file and upstream.

Weight bandwidth is the number that matters: a decode GEMV streams the entire
M x K weight matrix per token and reuses nothing, so it is DMA-bound by
construction and bytes/second is the ceiling the projections care about.

Usage:
    python tools/npu/bench_gemv_multicore.py --M 2048 --K 2048 --cores 1,2,4,8
"""

import ctypes
import os
import sys
from pathlib import Path

# ── Self-contained env bootstrap (mirrors oq_gemm_design.py) ─────────────────
# Without these, aie.utils silently falls back to CPUOnlyTensor and device="npu"
# is rejected at run time — which is exactly how the upstream example fails when
# invoked from a bare interpreter.
_BOOST_DEP = Path.home() / ".cache" / "hipfire-npu-deps" / "lib"
_XRT_LIB = "/opt/xilinx/xrt/lib"
_extra_ld = [str(p) for p in (_BOOST_DEP, Path(_XRT_LIB)) if p and Path(p).is_dir()]
if _extra_ld:
    _cur = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(_extra_ld + ([_cur] if _cur else []))

_XRT_BIN = "/opt/xilinx/xrt/bin"
if Path(_XRT_BIN).is_dir() and _XRT_BIN not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _XRT_BIN + os.pathsep + os.environ.get("PATH", "")

for _p in (Path.home() / ".venv" / "lib").glob("python*/site-packages"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

ctypes.CDLL(os.path.join(_XRT_LIB, "libxrt_coreutil.so.2"), mode=ctypes.RTLD_GLOBAL)
_XRT_PY = "/opt/xilinx/xrt/python"
if _XRT_PY not in sys.path:
    sys.path.insert(0, _XRT_PY)

import argparse  # noqa: E402

import numpy as np  # noqa: E402

import aie.iron as iron  # noqa: E402
from aie.iron import (  # noqa: E402
    CompileTime,
    In,
    ObjectFifo,
    Out,
    Program,
    Runtime,
    Worker,
    kernels,
)
from aie.iron.controlflow import range_  # noqa: E402
from aie.iron.device import NPU2, Tile  # noqa: E402
from aie.helpers.taplib import TensorTiler2D  # noqa: E402
from aie.utils import set_current_device  # noqa: E402
from aie.utils.benchmark import run_iters  # noqa: E402


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def gemv_spread(
    A: In,
    B: In,
    C: Out,
    *,
    M: CompileTime[int],
    K: CompileTime[int],
    m: CompileTime[int],
    k: CompileTime[int],
    n_cores: CompileTime[int] = 1,
    spread: CompileTime[bool] = True,
):
    M_div_n_cores = M // n_cores
    M_div_m_div_n_cores = M // (m * n_cores)
    K_div_k = K // k

    matvec_kernel = kernels.mv(dim_m=m, dim_k=k, vectorized=True, use_chess=False)
    zero_kernel = matvec_kernel.zero

    dtype_in = np.dtype[np.int16]
    dtype_out = np.dtype[np.int32]
    A_ty = np.ndarray[(M, K), dtype_in]
    B_ty = np.ndarray[(1, K), dtype_in]
    C_ty = np.ndarray[(1, M), dtype_out]
    inA_ty = np.ndarray[(m, k), dtype_in]
    inB_ty = np.ndarray[(k,), dtype_in]
    outC_ty = np.ndarray[(m,), dtype_out]

    a_dims_from_stream = [(m, 2), (k // 2, 2 * m), (2, 1)]

    def core_fn(of_a, of_b, of_c, zero, matvec):
        elem_out = of_c.acquire(1)
        zero(elem_out)
        for _ in range_(K_div_k):
            elem_in_a = of_a.acquire(1)
            elem_in_b = of_b.acquire(1)
            matvec(elem_in_a, elem_in_b, elem_out)
            of_a.release(1)
            of_b.release(1)
        of_c.release(1)

    memA_fifos = []
    coreA_fifos = []
    outC_fifos = []
    workers = []
    B_fifo = ObjectFifo(inB_ty)
    for i in range(n_cores):
        a_fifo = ObjectFifo(inA_ty, name=f"memA{i}")
        memA_fifos.append(a_fifo)
        coreA_fifos.append(a_fifo.cons().forward(dims_from_stream=a_dims_from_stream))
        outC_fifos.append(ObjectFifo(outC_ty, name=f"outC{i}"))
        workers.append(
            Worker(
                core_fn,
                [
                    coreA_fifos[i].cons(),
                    B_fifo.cons(),
                    outC_fifos[i].prod(),
                    zero_kernel,
                    matvec_kernel,
                ],
            )
        )

    A_taps = TensorTiler2D.group_tiler(
        (M, K), (m, k), (M_div_m_div_n_cores, K_div_k), prune_step=False
    )
    C_taps = TensorTiler2D.simple_tiler((1, M), (1, M_div_n_cores), prune_step=False)
    b_tap = TensorTiler2D.simple_tiler(
        (1, K), pattern_repeat=M_div_m_div_n_cores, prune_step=False
    )[0]

    # THE CHANGE: bind core i's weight-in and result-out shim endpoints to
    # column i. Unspread, every endpoint lands near the placer's centroid
    # (column 0) and 4 cores already exhaust that column's 2-in/2-out budget,
    # even though the device has 8 shim tiles. `spread=False` reproduces the
    # upstream failure for comparison.
    if spread:
        memA_prods = [f.prod(tile=Tile(col=i)) for i, f in enumerate(memA_fifos)]
        outC_cons = [f.cons(tile=Tile(col=i)) for i, f in enumerate(outC_fifos)]
    else:
        memA_prods = [f.prod() for f in memA_fifos]
        outC_cons = [f.cons() for f in outC_fifos]

    def sequence(a_in, b_in, c_out, b_h, memA_hs, outC_hs):
        b_h.fill(b_in, b_tap)
        for i, (a_tap, c_tap) in enumerate(zip(A_taps, C_taps)):
            memA_hs[i].fill(a_in, a_tap)
            outC_hs[i].drain(c_out, c_tap, wait=True)

    rt = Runtime(sequence, [A_ty, B_ty, C_ty, B_fifo.prod(), memA_prods, outC_cons])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


def run_one(M, K, m, k, n_cores, spread, warmup, iters):
    """Compile + run one configuration; returns (npu_us, None) or (None, reason).

    Correctness is checked before timing: a placement change must not alter the
    result, and a GEMV that silently drops a core's rows would otherwise look
    like a bandwidth win.
    """
    rng = np.random.default_rng(1726250518)
    A_np = rng.integers(-1000, 1000, size=(M, K), dtype=np.int16)
    B_np = rng.integers(-1000, 1000, size=(K,), dtype=np.int16)
    A_t = iron.tensor(A_np.reshape(-1), dtype=np.int16, device="npu")
    B_t = iron.tensor(B_np.reshape(-1), dtype=np.int16, device="npu")
    C_t = iron.zeros(M, dtype=np.int32, device="npu")

    bench = run_iters(
        gemv_spread,
        A_t,
        B_t,
        C_t,
        M=M,
        K=K,
        m=m,
        k=k,
        n_cores=n_cores,
        spread=spread,
        warmup=warmup,
        iters=iters,
    )

    expected = (A_np.astype(np.int64) @ B_np.astype(np.int64)).astype(np.int32)
    actual = C_t.numpy().reshape(M)
    if not np.array_equal(actual, expected):
        bad = int((actual != expected).sum())
        return None, f"MISMATCH in {bad}/{M} outputs"

    return (bench.npu.avg_us if bench.npu else bench.e2e.avg_us), None


def main():
    p = argparse.ArgumentParser(prog="NPU decode-shape GEMV, shim endpoints spread")
    p.add_argument("--M", type=int, default=2048)
    p.add_argument("--K", type=int, default=2048)
    p.add_argument("-m", type=int, default=32)
    p.add_argument("-k", type=int, default=32)
    p.add_argument("--cores", type=str, default="1,2,4,8")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument(
        "--no-spread",
        action="store_true",
        help="reproduce the upstream centroid-column placement failure",
    )
    a = p.parse_args()

    set_current_device(NPU2())
    wbytes = a.M * a.K * 2  # int16 weights, streamed whole per token
    print(f"[gemv] M={a.M} K={a.K} tile(m,k)=({a.m},{a.k})  weights {wbytes/1e6:.2f} MB")
    print(f"{'cores':>6} {'spread':>7} {'us':>10} {'GB/s':>8}  note")

    for nc in [int(x) for x in a.cores.split(",")]:
        if a.M % (a.m * nc) != 0:
            print(f"{nc:>6} {'-':>7} {'-':>10} {'-':>8}  skip: M % (m*cores) != 0")
            continue
        try:
            us, err = run_one(
                a.M, a.K, a.m, a.k, nc, not a.no_spread, a.warmup, a.iters
            )
        except Exception as e:  # placement/compile failures are the point here
            first = str(e).strip().split("\n")[0][:88]
            print(f"{nc:>6} {str(not a.no_spread):>7} {'-':>10} {'-':>8}  FAILED: {first}")
            continue
        if us is None:
            print(f"{nc:>6} {str(not a.no_spread):>7} {'-':>10} {'-':>8}  {err}")
            continue
        print(f"{nc:>6} {str(not a.no_spread):>7} {us:>10.1f} {wbytes/1e9/(us/1e6):>8.2f}  ok")


if __name__ == "__main__":
    main()
