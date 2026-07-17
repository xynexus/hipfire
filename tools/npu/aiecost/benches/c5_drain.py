#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# hipfire — see LICENSE and NOTICE in the project root.
"""C5: drain bandwidth — bw_drain_bytes_s.

The mirror of C2. Cores produce tiles as cheaply as possible (one word each) and
the DMA drains them out; the slope of time-vs-bytes is 1/drain-bandwidth.

Measured separately from the feed rather than assumed symmetric, because R64
found the output DMA starved for 198 us of a 241 us device span — the drain path
was the limiter there, not the feed.

Usage:
    python -m aiecost.benches.c5_drain --save
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aiecost import env  # noqa: E402

env.bootstrap()

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
KERNEL_SRC = HERE / "c5_src.cc"

_mlir_pkg = next((Path(p) for p in sys.path if (Path(p) / "mlir_aie").is_dir()), None)
AIE_INCLUDE = _mlir_pkg / "mlir_aie" / "include" if _mlir_pkg else None
AIE_RUNTIME_LIB = _mlir_pkg / "mlir_aie" / "aie_runtime_lib" / "AIE2" if _mlir_pkg else None

TILE_ELEM = 1024  # 4 KiB


def build(n_tiles: int, columns: int, out_dir: Path) -> tuple[Path, Path]:
    from aie.iron import ObjectFifo, Program, Runtime, Worker
    from aie.iron.controlflow import range_
    from aie.iron.device import NPU1
    from aie.iron.kernel import ExternalFunction
    from aie.iron.placers import SequentialPlacer
    from aie.utils import set_current_device
    from aie.utils.compile import compile_external_kernel, compile_mlir_module

    set_current_device(NPU1())

    out_dir.mkdir(parents=True, exist_ok=True)
    xclbin = out_dir / f"c5-t{n_tiles}-c{columns}.xclbin"
    insts = out_dir / f"c5-t{n_tiles}-c{columns}-insts.bin"
    if xclbin.exists() and insts.exists():
        return xclbin, insts

    Tile: object = np.ndarray[(TILE_ELEM,), np.dtype[np.int32]]
    Stream: object = np.ndarray[(TILE_ELEM * n_tiles,), np.dtype[np.int32]]

    kern = ExternalFunction(
        "c5_src",
        source_file=str(KERNEL_SRC),
        arg_types=[Tile],
        include_dirs=[str(AIE_INCLUDE), str(AIE_RUNTIME_LIB)],
        compile_flags=["-std=c++20", "-O2"],
    )

    of_outs = [ObjectFifo(Tile, name=f"out{c}", depth=2) for c in range(columns)]

    def core(o_out, kk):
        for _ in range_(n_tiles):
            eo = o_out.acquire(1)
            kk(eo)
            o_out.release(1)

    workers = [Worker(core, [of_outs[c].prod(), kern]) for c in range(columns)]

    rt = Runtime()
    with rt.sequence(*([Stream] * columns)) as args:
        dsts = args if columns > 1 else [args]
        for w in workers:
            rt.start(w)
        for c in range(columns):
            rt.drain(of_outs[c].cons(), dsts[c], wait=True)

    module = Program(NPU1(), rt).resolve_program(SequentialPlacer())
    with tempfile.TemporaryDirectory(prefix="aiecost_c5_") as tmpname:
        tmp = Path(tmpname)
        compile_external_kernel(kern, tmp, target_arch="aie2")
        compile_mlir_module(mlir_module=module, insts_path=tmp / "insts.bin", xclbin_path=tmp / "final.xclbin", work_dir=tmp)
        shutil.copy2(tmp / "final.xclbin", xclbin)
        shutil.copy2(tmp / "insts.bin", insts)
    return xclbin, insts


def run(xclbin: Path, insts: Path, n_tiles: int, columns: int, reps: int, warmup: int) -> dict:
    from aie.utils.hostruntime.xrtruntime.hostruntime import XRTHostRuntime
    from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor
    from aie.utils.npukernel import NPUKernel

    n_elem = TILE_ELEM * n_tiles
    dsts = [XRTTensor((n_elem,), dtype=np.int32, device="cpu") for _ in range(columns)]

    kernel = NPUKernel(xclbin_path=str(xclbin), insts_path=str(insts), kernel_name="MLIR_AIE")
    hrt = XRTHostRuntime()
    handle = hrt.load(kernel)

    for _ in range(warmup):
        hrt.run(handle, dsts)
    npu = []
    for _ in range(reps):
        r = hrt.run(handle, dsts)
        if getattr(r, "npu_time", None):
            npu.append(float(r.npu_time) * 1e-9)
    return {"bytes": n_elem * 4 * columns, "npu_med": statistics.median(npu) if npu else None}


def fit(points):
    n = len(points)
    sx = sum(p[0] for p in points)
    sy = sum(p[1] for p in points)
    sxx = sum(p[0] * p[0] for p in points)
    sxy = sum(p[0] * p[1] for p in points)
    b = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    return (sy - b * sx) / n, b


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tiles", type=int, nargs="+", default=[512, 1024, 2048])
    p.add_argument("--columns", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--reps", type=int, default=12)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--cache", default=str(Path.home() / ".cache" / "hipfire-aiecost" / "c5"))
    p.add_argument("--save", action="store_true")
    p.add_argument("--json", metavar="PATH")
    args = p.parse_args()

    print(f"C5 drain bandwidth: tiles={args.tiles} columns={args.columns} (tile={TILE_ELEM * 4} B)")
    per_col: dict[int, dict] = {}
    for cols in args.columns:
        pts = []
        for nt in args.tiles:
            xclbin, insts = build(nt, cols, Path(args.cache))
            r = run(xclbin, insts, nt, cols, args.reps, args.warmup)
            if not r["npu_med"]:
                continue
            pts.append((r["bytes"], r["npu_med"]))
            print(f"  cols={cols} {r['bytes'] / 1024:8.0f} KiB  npu={r['npu_med'] * 1e6:9.2f} us  point-rate={r['bytes'] / r['npu_med'] / 1e9:6.3f} GB/s")
        if len(pts) < 2:
            continue
        fixed, slope = fit(pts)
        bw = 1.0 / slope if slope > 0 else float("nan")
        per_col[cols] = {"fixed_s": fixed, "slope_s_per_byte": slope, "bw_bytes_s": bw, "points": pts}
        print(f"  -> cols={cols}: slope-BW={bw / 1e9:6.3f} GB/s  fixed={fixed * 1e6:8.2f} us\n")

    if not per_col:
        print("no results")
        return 1
    print("=" * 78)
    for c in sorted(per_col):
        print(f"  cols={c}: {per_col[c]['bw_bytes_s'] / 1e9:6.3f} GB/s")
    best = max(per_col, key=lambda c: per_col[c]["bw_bytes_s"])
    bw = per_col[best]["bw_bytes_s"]
    print(f"\n  C5: bw_drain = {bw / 1e9:.3f} GB/s at {best} columns")

    if args.json:
        Path(args.json).write_text(json.dumps({str(k): v for k, v in per_col.items()}, indent=2))
    if args.save:
        from aiecost import calib

        key = calib.current_key()
        ev = [f"cols={c}: {per_col[c]['bw_bytes_s'] / 1e9:.3f} GB/s (fixed {per_col[c]['fixed_s'] * 1e6:.1f} us)" for c in sorted(per_col)]
        cs = {
            "bw_drain_bytes_s": calib.Constant(
                name="bw_drain_bytes_s", value=bw, unit="bytes/s", bench="C5",
                method=f"core writes 1 word/tile, DMA drains; slope of npu_time vs bytes at {best} columns",
                admissible=True, evidence=ev,
                caveats=["slope basis; C1 floor lands in the intercept",
                         "measured separately from feed because R64 found drain, not feed, was the limiter"],
            )
        }
        print(f"  saved -> {calib.save(key, cs)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
