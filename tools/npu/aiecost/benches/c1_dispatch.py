#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# hipfire — see LICENSE and NOTICE in the project root.
"""C1: the fixed dispatch floor — c_cmd and c_bo.

A near-null kernel: touch each input once, write one word. Whatever a dispatch
costs here is cost no real kernel can escape. Sweeping the BO count separates
per-command from per-BO cost:

    t(n_bos) = c_cmd + c_bo * n_bos

Why this is measured before anything else (plan §7, phase 2): R64 found a warm
production wrapper was 76.6% preparation/submit/sync/deblock and only 23.4%
device; R117 doubled useful work and got 9.8% faster. On npu1 — 16 cores rather
than halo's 32 — the floor should matter proportionally *more*, not less.

Reports both device (npu_time) and wall time. The gap between them is the
host/submit overhead that t_host + t_submit must cover.

Usage:
    python -m aiecost.benches.c1_dispatch
    python -m aiecost.benches.c1_dispatch --reps 200 --save
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aiecost import env  # noqa: E402
from aiecost.target import include_dirs, resolve_program, resolve_target  # noqa: E402

env.bootstrap()

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
KERNEL_SRC = HERE / "c1_null.cc"

_mlir_pkg = next((Path(p) for p in sys.path if (Path(p) / "mlir_aie").is_dir()), None)
N_ELEM = 16  # tiny: we are measuring the floor, not bandwidth


def build(nargs: int, out_dir: Path, device: str = "auto") -> tuple[Path, Path]:
    from aie.iron import ObjectFifo, Program, Runtime, Worker
    from aie.iron.kernel import ExternalFunction
    from aie.utils import set_current_device
    from aie.utils.compile import compile_external_kernel, compile_mlir_module

    target = resolve_target(device)
    iron_device = target.iron_device()
    set_current_device(iron_device)

    T: object = np.ndarray[(N_ELEM,), np.dtype[np.int32]]
    arg_types = [T] * nargs + [T]

    kern = ExternalFunction(
        "c1_null",
        source_file=str(KERNEL_SRC),
        arg_types=arg_types,
        include_dirs=include_dirs(_mlir_pkg, target),
        compile_flags=["-std=c++20", "-O2", f"-DNARGS={nargs}"],
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    xclbin = out_dir / f"c1-{target.cache_tag}-n{nargs}.xclbin"
    insts = out_dir / f"c1-{target.cache_tag}-n{nargs}-insts.bin"
    if xclbin.exists() and insts.exists():
        return xclbin, insts

    ins = [ObjectFifo(T, name=f"i{j}", depth=1) for j in range(nargs)]
    of_o = ObjectFifo(T, name="o", depth=1)

    def core(*fifos):
        kk = fifos[-1]
        prods = fifos[:-1]
        elems = [f.acquire(1) for f in prods]
        kk(*elems)
        for f in prods:
            f.release(1)

    w = Worker(core, [f.cons() for f in ins] + [of_o.prod(), kern])
    rt = Runtime()
    with rt.sequence(*arg_types) as args:
        rt.start(w)
        for f, a in zip(ins, args[:-1]):
            rt.fill(f.prod(), a)
        rt.drain(of_o.cons(), args[-1], wait=True)

    module = resolve_program(Program(iron_device, rt))
    with tempfile.TemporaryDirectory(prefix="aiecost_c1_") as tmpname:
        tmp = Path(tmpname)
        compile_external_kernel(kern, tmp, target_arch=target.target_arch)
        compile_mlir_module(mlir_module=module, insts_path=tmp / "insts.bin", xclbin_path=tmp / "final.xclbin", work_dir=tmp)
        shutil.copy2(tmp / "final.xclbin", xclbin)
        shutil.copy2(tmp / "insts.bin", insts)
    return xclbin, insts


def run(xclbin: Path, insts: Path, nargs: int, reps: int, warmup: int) -> dict:
    from aie.utils.hostruntime.xrtruntime.hostruntime import XRTHostRuntime
    from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor
    from aie.utils.npukernel import NPUKernel

    tensors = [XRTTensor(np.ones((N_ELEM,), dtype=np.int32), dtype=np.int32, device="cpu") for _ in range(nargs)]
    t_out = XRTTensor((N_ELEM,), dtype=np.int32, device="cpu")
    args = tensors + [t_out]

    kernel = NPUKernel(xclbin_path=str(xclbin), insts_path=str(insts), kernel_name="MLIR_AIE")
    hrt = XRTHostRuntime()
    handle = hrt.load(kernel)

    for _ in range(warmup):
        hrt.run(handle, args)

    npu, wall = [], []
    for _ in range(reps):
        t0 = time.perf_counter()
        r = hrt.run(handle, args)
        wall.append(time.perf_counter() - t0)
        if getattr(r, "npu_time", None):
            npu.append(float(r.npu_time) * 1e-9)

    t_out.to("cpu")
    o = t_out.numpy()
    return {
        "nargs": nargs,
        "n_bos": nargs + 1,
        "npu_med": statistics.median(npu) if npu else None,
        "npu_min": min(npu) if npu else None,
        "wall_med": statistics.median(wall),
        "wall_min": min(wall),
        "reported_nargs": int(o[1]),
        "acc": int(o[0]),
    }


def fit(points: list[tuple[int, float]]) -> tuple[float, float]:
    n = len(points)
    sx = sum(p[0] for p in points)
    sy = sum(p[1] for p in points)
    sxx = sum(p[0] * p[0] for p in points)
    sxy = sum(p[0] * p[1] for p in points)
    b = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    return (sy - b * sx) / n, b


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--nargs",
        type=int,
        nargs="+",
        default=[1, 2],
        help="inputs; one core has two input DMA channels (BOs = nargs+1)",
    )
    p.add_argument("--reps", type=int, default=100)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--device", default="auto", choices=["auto", "npu1", "npu2"])
    p.add_argument("--cache", default=str(Path.home() / ".cache" / "hipfire-aiecost" / "c1"))
    p.add_argument("--save", action="store_true", help="record c_cmd/c_bo into the calibration set")
    p.add_argument("--json", metavar="PATH")
    args = p.parse_args()

    target = resolve_target(args.device)
    print(f"C1 dispatch floor: target={target.cache_tag} nargs={args.nargs} reps={args.reps} warmup={args.warmup}")
    rows = []
    for nargs in args.nargs:
        xclbin, insts = build(nargs, Path(args.cache), target.key)
        r = run(xclbin, insts, nargs, args.reps, args.warmup)
        if r["reported_nargs"] != nargs:
            print(f"  !! kernel reported NARGS={r['reported_nargs']}, expected {nargs}")
            return 2
        rows.append(r)
        npu_us = r["npu_med"] * 1e6 if r["npu_med"] else float("nan")
        print(f"  n_bos={r['n_bos']}  npu_med={npu_us:8.3f} us  wall_med={r['wall_med'] * 1e6:9.3f} us  (wall-npu={(r['wall_med'] - (r['npu_med'] or 0)) * 1e6:8.3f} us)")

    basis = "npu" if all(r["npu_med"] for r in rows) else "wall"
    pts = [(r["n_bos"], r["npu_med"] if basis == "npu" else r["wall_med"]) for r in rows]
    c_cmd, c_bo = fit(pts)

    wall_pts = [(r["n_bos"], r["wall_med"]) for r in rows]
    w_cmd, w_bo = fit(wall_pts)

    print("=" * 78)
    print(f"  device [{basis}]: c_cmd = {c_cmd * 1e6:8.3f} us    c_bo = {c_bo * 1e6:7.3f} us/BO")
    print(f"  wall        : c_call+ = {w_cmd * 1e6:8.3f} us    c_bo = {w_bo * 1e6:7.3f} us/BO")
    print(f"  host overhead above device: {(w_cmd - c_cmd) * 1e6:.3f} us fixed")
    print("\n  This is the floor: no kernel on this device can dispatch faster.")

    if args.json:
        Path(args.json).write_text(json.dumps({"rows": rows, "c_cmd": c_cmd, "c_bo": c_bo, "basis": basis}, indent=2))
        print(f"  wrote {args.json}")

    if args.save:
        from aiecost import calib

        key = calib.current_key()
        ev = [f"n_bos {r['n_bos']}: npu_med={(r['npu_med'] or 0) * 1e6:.3f} us wall_med={r['wall_med'] * 1e6:.3f} us" for r in rows]
        cs = {
            "c_cmd_s": calib.Constant(
                name="c_cmd_s", value=c_cmd, unit="s", bench="C1",
                method=f"null-kernel BO sweep, intercept of t = c_cmd + c_bo*n_bos ({basis} basis, median of {args.reps})",
                admissible=True, evidence=ev,
                caveats=["near-null kernel: floor only", "medians; warmed — cold first-command values are much higher (cf. R63)"],
            ),
            "c_bo_s": calib.Constant(
                name="c_bo_s", value=c_bo, unit="s", bench="C1",
                method=f"null-kernel BO sweep, slope ({basis} basis)",
                admissible=True, evidence=ev,
                caveats=[
                    "H4 caps BOs at 5 data-arg slots",
                    "the null graph uses one core and therefore measures only two and three total BOs; that core has two input DMA channels",
                ],
            ),
            "c_call_s": calib.Constant(
                name="c_call_s", value=max(0.0, w_cmd - c_cmd), unit="s", bench="C1",
                method="wall minus device intercept: host-side fixed cost of one run() round trip",
                admissible=True, evidence=[f"wall intercept {w_cmd * 1e6:.3f} us - device intercept {c_cmd * 1e6:.3f} us"],
                caveats=["excludes user-side pack/deblock, which C7 measures separately"],
            ),
        }
        print(f"  saved -> {calib.save(key, cs, meta={'device': target.key, 'tile_isa': target.tile_isa})}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
