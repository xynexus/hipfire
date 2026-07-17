#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# hipfire — see LICENSE and NOTICE in the project root.
"""C3: per-DMA-task issue cost — c_task_issue_s.

Holds total bytes CONSTANT and varies how many tiles carry them. Same payload,
same bandwidth demand, different task count — so any time that scales with tile
count is per-task overhead and nothing else.

    t(n_tiles) = (bytes / bw_feed) + c_issue * n_tiles

This is the term the aie2p corpus cared about most:
  - R68 collapsed three output objects into one and cut ~360 tasks ~3x, for 24%
    less time with identical math.
  - R119 added repeat_count=1 to a single task with outer dim 2 and got 3.54%
    over R118's two explicit tasks.
  - R117 doubled useful work and got *faster*, which only makes sense if fixed
    per-task/dispatch cost dominates.

Tile size is bounded below by L1 (M1: 64 KiB) and in practice by the BD budget
(M6: 16 BDs/core).

Usage:
    python -m aiecost.benches.c3_tasks --save
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
from aiecost.target import include_dirs, resolve_program, resolve_target  # noqa: E402

env.bootstrap()

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
KERNEL_SRC = HERE / "c2_sink.cc"  # same cheapest-possible consumer as C2

_mlir_pkg = next((Path(p) for p in sys.path if (Path(p) / "mlir_aie").is_dir()), None)
ACC_ELEM = 16


def build(
    tile_elem: int, n_tiles: int, out_dir: Path, device: str = "auto"
) -> tuple[Path, Path] | None:
    from aie.iron import ObjectFifo, Program, Runtime, Worker
    from aie.iron.controlflow import range_
    from aie.iron.kernel import ExternalFunction
    from aie.utils import set_current_device
    from aie.utils.compile import compile_external_kernel, compile_mlir_module

    target = resolve_target(device)
    iron_device = target.iron_device()
    set_current_device(iron_device)

    out_dir.mkdir(parents=True, exist_ok=True)
    xclbin = out_dir / f"c3-{target.cache_tag}-e{tile_elem}-t{n_tiles}.xclbin"
    insts = out_dir / f"c3-{target.cache_tag}-e{tile_elem}-t{n_tiles}-insts.bin"
    if xclbin.exists() and insts.exists():
        return xclbin, insts

    Tile: object = np.ndarray[(tile_elem,), np.dtype[np.int32]]
    Acc: object = np.ndarray[(ACC_ELEM,), np.dtype[np.int32]]
    Stream: object = np.ndarray[(tile_elem * n_tiles,), np.dtype[np.int32]]

    kern = ExternalFunction(
        "c2_sink",
        source_file=str(KERNEL_SRC),
        arg_types=[Tile, Acc],
        include_dirs=include_dirs(_mlir_pkg, target),
        compile_flags=["-std=c++20", "-O2"],
    )

    of_in = ObjectFifo(Tile, name="in0", depth=2)
    of_out = ObjectFifo(Acc, name="out0", depth=1)

    def core(a_in, o_out, kk):
        eo = o_out.acquire(1)
        for _ in range_(n_tiles):
            ea = a_in.acquire(1)
            kk(ea, eo)
            a_in.release(1)
        o_out.release(1)

    w = Worker(core, [of_in.cons(), of_out.prod(), kern])
    rt = Runtime()
    with rt.sequence(Stream, Acc) as (src, dst):
        rt.start(w)
        rt.fill(of_in.prod(), src)
        rt.drain(of_out.cons(), dst, wait=True)

    try:
        module = resolve_program(Program(iron_device, rt))
        with tempfile.TemporaryDirectory(prefix="aiecost_c3_") as tmpname:
            tmp = Path(tmpname)
            compile_external_kernel(kern, tmp, target_arch=target.target_arch)
            compile_mlir_module(
                mlir_module=module,
                insts_path=tmp / "insts.bin",
                xclbin_path=tmp / "final.xclbin",
                work_dir=tmp,
            )
            shutil.copy2(tmp / "final.xclbin", xclbin)
            shutil.copy2(tmp / "insts.bin", insts)
    except RuntimeError as error:
        print(
            f"  tile={tile_elem * 4} B: BUILD REJECTED ({type(error).__name__}) — "
            "depth-2 input plus output/stack exceeds tile L1"
        )
        return None
    return xclbin, insts


def run(xclbin: Path, insts: Path, tile_elem: int, n_tiles: int, reps: int, warmup: int) -> dict:
    from aie.utils.hostruntime.xrtruntime.hostruntime import XRTHostRuntime
    from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor
    from aie.utils.npukernel import NPUKernel

    n_elem = tile_elem * n_tiles
    src = XRTTensor(np.ones((n_elem,), dtype=np.int32), dtype=np.int32, device="cpu")
    dst = XRTTensor((ACC_ELEM,), dtype=np.int32, device="cpu")

    kernel = NPUKernel(xclbin_path=str(xclbin), insts_path=str(insts), kernel_name="MLIR_AIE")
    hrt = XRTHostRuntime()
    handle = hrt.load(kernel)
    for _ in range(warmup):
        hrt.run(handle, [src, dst])
    npu = []
    for _ in range(reps):
        r = hrt.run(handle, [src, dst])
        if getattr(r, "npu_time", None):
            npu.append(float(r.npu_time) * 1e-9)
    return {"tile_bytes": tile_elem * 4, "n_tiles": n_tiles, "bytes": n_elem * 4,
            "npu_med": statistics.median(npu) if npu else None}


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
    p.add_argument("--total-bytes", type=int, default=8 * 1024 * 1024)
    p.add_argument("--tile-bytes", type=int, nargs="+", default=[2048, 4096, 8192, 16384, 32768])
    p.add_argument("--reps", type=int, default=12)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--device", default="auto", choices=["auto", "npu1", "npu2"])
    p.add_argument("--cache", default=str(Path.home() / ".cache" / "hipfire-aiecost" / "c3"))
    p.add_argument("--save", action="store_true")
    p.add_argument("--json", metavar="PATH")
    args = p.parse_args()

    target = resolve_target(args.device)
    print(f"C3 task issue cost: target={target.cache_tag} total={args.total_bytes / 1024:.0f} KiB held CONSTANT, tile sizes={args.tile_bytes}")
    rows = []
    for tb in args.tile_bytes:
        tile_elem = tb // 4
        n_tiles = args.total_bytes // tb
        built = build(tile_elem, n_tiles, Path(args.cache), target.key)
        if not built:
            continue
        xclbin, insts = built
        r = run(xclbin, insts, tile_elem, n_tiles, args.reps, args.warmup)
        if not r["npu_med"]:
            continue
        rows.append(r)
        print(f"  tile={tb:>6} B  n_tiles={n_tiles:>5}  npu={r['npu_med'] * 1e6:9.2f} us  rate={r['bytes'] / r['npu_med'] / 1e9:6.3f} GB/s")

    if len(rows) < 2:
        print("no results")
        return 1

    fixed, c_issue = fit([(r["n_tiles"], r["npu_med"]) for r in rows])
    print("=" * 82)
    print(f"  c_task_issue = {c_issue * 1e9:8.1f} ns/task    fixed = {fixed * 1e6:8.2f} us")
    if c_issue > 0:
        eq = args.total_bytes / (16.1e9)  # C2 feed time at 4 cols for reference
        print(f"  at {rows[-1]['n_tiles']} tiles that is {c_issue * rows[-1]['n_tiles'] * 1e6:.2f} us of pure task overhead")
        print(f"  (reference: {args.total_bytes / 1024:.0f} KiB at the C2 4-col feed roof would take {eq * 1e6:.1f} us)")
    else:
        print("  NOTE: non-positive slope — task count has no measurable cost at these sizes.")

    if args.json:
        Path(args.json).write_text(json.dumps({"rows": rows, "c_issue": c_issue, "fixed": fixed}, indent=2))
    if args.save:
        from aiecost import calib

        key = calib.current_key()
        ev = [f"tile={r['tile_bytes']} B, n_tiles={r['n_tiles']}: {r['npu_med'] * 1e6:.2f} us" for r in rows]
        cs = {
            "c_task_issue_s": calib.Constant(
                name="c_task_issue_s", value=max(0.0, c_issue), unit="s", bench="C3",
                method=f"constant {args.total_bytes} B payload, tile-size sweep; slope of npu_time vs tile count",
                admissible=True, evidence=ev,
                caveats=[
                    "single column; per-task cost may differ when several columns issue concurrently",
                    "tile size is bounded by L1 (M1: 64 KiB) and the BD budget (M6: 16/core)",
                ],
            )
        }
        print(f"  saved -> {calib.save(key, cs, meta={'device': target.key, 'tile_isa': target.tile_isa})}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
