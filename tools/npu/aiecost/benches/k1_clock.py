#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# hipfire — see LICENSE and NOTICE in the project root.
"""K1: measure the AIE compute clock (f_H) on npu1.

npu1 reports no clock. xrt-smi implements only {aie-partitions, all, host,
platform} and none carries npu_clk_max — halo's 1800 MHz comes from a report
this device does not have. f_H is the time base for the whole t_core term, so
without it `cyc_mmul / f_H` can only ever be fitted as a product.

Method (adapted from r0/r0b_throughput.cc, which established it on aie2p):
CHAINS independent VMAC accumulator chains run from resident L1 tiles. With
enough chains the accumulator latency is hidden and the pipe saturates at II=1
(one VMAC per cycle), so measured VMAC/s == f_H.

The II=1 premise is tested, not assumed: sweep CHAINS and require the
throughput to plateau. A plateau means latency is hidden; still-scaling means
not saturated and the estimate is not admissible.

Only the ITERS slope is used, so fixed dispatch/host cost cancels.

Build/run uses the skill-documented compile_mlir_module + XRTHostRuntime path.
The @jit path used by r0b_run.py fails on this box: aiecc asserts
`targetModel.hasProperty(IsNPU)` during CDO generation even though the module
is a correct `aie.device(npu1)`.

Usage:
    python -m aiecost.benches.k1_clock
    python -m aiecost.benches.k1_clock --iters 200000 400000 --chains 1 2 4
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aiecost import env  # noqa: E402

env.bootstrap()

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
KERNEL_SRC = HERE / "k1_vmac_chain.cc"

_mlir_pkg = next((Path(p) for p in sys.path if (Path(p) / "mlir_aie").is_dir()), None)
AIE_INCLUDE = _mlir_pkg / "mlir_aie" / "include" if _mlir_pkg else None
AIE_RUNTIME_LIB = _mlir_pkg / "mlir_aie" / "aie_runtime_lib" / "AIE2" if _mlir_pkg else None

# Resident tile sizes for aie::mmul<4,8,8,int8,int8>: A=4x8=32B, B=8x8=64B.
SA, SB, OUT_N = 32, 64, 64
A_N, B_N = 2 * SA, 2 * SB


def build(iters: int, chains: int, out_dir: Path) -> tuple[Path, Path]:
    """Compile one (iters, chains) point. Returns (xclbin, insts)."""
    from aie.iron import ObjectFifo, Program, Runtime, Worker
    from aie.iron.device import NPU1
    from aie.iron.kernel import ExternalFunction
    from aie.iron.placers import SequentialPlacer
    from aie.utils import set_current_device
    from aie.utils.compile import compile_external_kernel, compile_mlir_module

    set_current_device(NPU1())

    A_ty: object = np.ndarray[(A_N,), np.dtype[np.int8]]
    B_ty: object = np.ndarray[(B_N,), np.dtype[np.int8]]
    O_ty: object = np.ndarray[(OUT_N,), np.dtype[np.int32]]

    kern = ExternalFunction(
        "k1_vmac_chain",
        source_file=str(KERNEL_SRC),
        arg_types=[A_ty, B_ty, O_ty],
        include_dirs=[str(AIE_INCLUDE), str(AIE_RUNTIME_LIB)],
        compile_flags=["-std=c++20", "-O2", f"-DITERS={iters}", f"-DCHAINS={chains}"],
    )

    of_a = ObjectFifo(A_ty, name="a", depth=1)
    of_b = ObjectFifo(B_ty, name="b", depth=1)
    of_o = ObjectFifo(O_ty, name="o", depth=1)

    def core(a_in, b_in, o_out, kk):
        ea = a_in.acquire(1)
        eb = b_in.acquire(1)
        eo = o_out.acquire(1)
        kk(ea, eb, eo)
        a_in.release(1)
        b_in.release(1)
        o_out.release(1)

    w = Worker(core, [of_a.cons(), of_b.cons(), of_o.prod(), kern])
    rt = Runtime()
    with rt.sequence(A_ty, B_ty, O_ty) as (a, b, o):
        rt.start(w)
        rt.fill(of_a.prod(), a)
        rt.fill(of_b.prod(), b)
        rt.drain(of_o.cons(), o, wait=True)

    module = Program(NPU1(), rt).resolve_program(SequentialPlacer())

    out_dir.mkdir(parents=True, exist_ok=True)
    xclbin = out_dir / f"k1-i{iters}-c{chains}.xclbin"
    insts = out_dir / f"k1-i{iters}-c{chains}-insts.bin"
    if xclbin.exists() and insts.exists():
        return xclbin, insts

    with tempfile.TemporaryDirectory(prefix="aiecost_k1_") as tmpname:
        tmp = Path(tmpname)
        compile_external_kernel(kern, tmp, target_arch="aie2")
        compile_mlir_module(mlir_module=module, insts_path=tmp / "insts.bin", xclbin_path=tmp / "final.xclbin", work_dir=tmp)
        shutil.copy2(tmp / "final.xclbin", xclbin)
        shutil.copy2(tmp / "insts.bin", insts)
    return xclbin, insts


def run(xclbin: Path, insts: Path, reps: int) -> dict:
    """Run the kernel `reps` times; return best device + wall time and counters."""
    from aie.utils.hostruntime.xrtruntime.hostruntime import XRTHostRuntime
    from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor
    from aie.utils.npukernel import NPUKernel

    rng = np.random.default_rng(0)
    a = rng.integers(-8, 8, size=(A_N,), dtype=np.int8)
    b = rng.integers(-8, 8, size=(B_N,), dtype=np.int8)

    t_a = XRTTensor(a, dtype=np.int8, device="cpu")
    t_b = XRTTensor(b, dtype=np.int8, device="cpu")
    t_o = XRTTensor((OUT_N,), dtype=np.int32, device="cpu")

    kernel = NPUKernel(xclbin_path=str(xclbin), insts_path=str(insts), kernel_name="MLIR_AIE")
    hrt = XRTHostRuntime()
    handle = hrt.load(kernel)

    best_npu, best_wall = float("inf"), float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        result = hrt.run(handle, [t_a, t_b, t_o])
        wall = time.perf_counter() - t0
        npu_ns = getattr(result, "npu_time", None)
        if npu_ns:
            best_npu = min(best_npu, float(npu_ns) * 1e-9)
        best_wall = min(best_wall, wall)

    t_o.to("cpu")
    o = t_o.numpy()
    return {
        "npu_s": None if best_npu == float("inf") else best_npu,
        "wall_s": best_wall,
        "reported_iters": int(o[0]),
        "reported_chains": int(o[1]),
        "vmacs": int(o[2]),
        "macs_per_vmac": int(o[3]),
    }


def fit_slope(points: list[tuple[int, float]]) -> tuple[float, float]:
    """Least squares t = a + b*iters."""
    n = len(points)
    sx = sum(p[0] for p in points)
    sy = sum(p[1] for p in points)
    sxx = sum(p[0] * p[0] for p in points)
    sxy = sum(p[0] * p[1] for p in points)
    b = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    return (sy - b * sx) / n, b


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--iters", type=int, nargs="+", default=[200_000, 400_000, 800_000, 1_600_000])
    p.add_argument("--chains", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--cache", default=str(Path.home() / ".cache" / "hipfire-aiecost" / "k1"))
    p.add_argument("--json", metavar="PATH")
    args = p.parse_args()

    cache = Path(args.cache)
    print(f"K1 clock: chains={args.chains} iters={args.iters} reps={args.reps}")
    results: dict[int, dict] = {}

    for chains in args.chains:
        pts_npu: list[tuple[int, float]] = []
        pts_wall: list[tuple[int, float]] = []
        for iters in args.iters:
            xclbin, insts = build(iters, chains, cache)
            r = run(xclbin, insts, args.reps)
            if r["reported_iters"] != iters or r["reported_chains"] != chains:
                print(f"  !! kernel reported iters={r['reported_iters']} chains={r['reported_chains']}, expected {iters}/{chains}")
                return 2
            if r["npu_s"]:
                pts_npu.append((iters, r["npu_s"]))
            pts_wall.append((iters, r["wall_s"]))
            npu_ms = r["npu_s"] * 1e3 if r["npu_s"] else float("nan")
            print(f"  chains={chains} iters={iters:>9} npu={npu_ms:9.4f} ms wall={r['wall_s'] * 1e3:9.4f} ms")

        pts = pts_npu if len(pts_npu) >= 2 else pts_wall
        basis = "npu_time" if len(pts_npu) >= 2 else "wall"
        if len(pts) < 2:
            continue
        a, b = fit_slope(pts)
        vmac_s = chains / b if b > 0 else float("nan")
        results[chains] = {
            "basis": basis,
            "fixed_s": a,
            "slope_s_per_iter": b,
            "vmac_per_s": vmac_s,
            "implied_f_h_mhz": vmac_s / 1e6,
            "points": pts,
        }
        print(f"  -> chains={chains} [{basis}]: {b * 1e9:.4f} ns/iter  {vmac_s / 1e9:.3f} G VMAC/s  f_H(if II=1)={vmac_s / 1e6:.1f} MHz\n")

    if not results:
        print("no results")
        return 1

    print("=" * 78)
    print("saturation check — VMAC/s must plateau as chains grow (II=1 <=> latency hidden):")
    for c in sorted(results):
        print(f"  chains={c}: {results[c]['vmac_per_s'] / 1e9:7.3f} G VMAC/s  ->  f_H {results[c]['implied_f_h_mhz']:7.1f} MHz")

    order = sorted(results)
    verdict, admissible = "UNKNOWN (need >= 2 chain points)", False
    if len(order) >= 2:
        gain = results[order[-1]]["vmac_per_s"] / results[order[-2]]["vmac_per_s"]
        print(f"\n  chains {order[-2]} -> {order[-1]} changed throughput by {gain:.3f}x")
        if gain < 1.10:
            verdict, admissible = "PLATEAU — II=1 supported, f_H admissible", True
        else:
            verdict = "STILL SCALING — not saturated; add chains before trusting f_H"
    print(f"  verdict: {verdict}")

    f_h = results[order[-1]]["implied_f_h_mhz"]
    print(f"\n  K1: f_H ~= {f_h:.1f} MHz" + ("" if admissible else "   [NOT ADMISSIBLE]"))

    if args.json:
        Path(args.json).write_text(json.dumps({"results": {str(k): v for k, v in results.items()}, "admissible": admissible, "f_h_mhz": f_h}, indent=2))
        print(f"  wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
