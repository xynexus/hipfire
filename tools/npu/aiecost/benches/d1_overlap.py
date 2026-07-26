#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# hipfire — see LICENSE and NOTICE in the project root.
"""D1: validate the model's max() composition — does feed overlap compute?

The model composes terms as `max(t_feed, t_core, t_drain)`, not a sum. That is
its central design decision and the reason it can reproduce R117 (more work,
same fixed cost, less time). Yet nothing tested it: family B is feed-only with a
trivial core, family C is compute-only with no feed. Both terms were validated
in ISOLATION and the composition never was.

This streams tiles AND computes per tile, sweeping MMULS so t_core crosses
t_feed:

    max model:  time/tile = max(t_feed, t_core)  -> flat, then rises
    sum model:  time/tile = t_feed + t_core      -> rises immediately

At the crossover the predictions differ by 2x. The sweep is committed before
measuring, like the other families.

Usage:
    python -m aiecost.benches.d1_overlap
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

from aiecost import calib, env  # noqa: E402

env.bootstrap()

import numpy as np  # noqa: E402

from aiecost.benches.c2_feed import ACC_ELEM, TILE_ELEM, _mlir_pkg, include_dirs  # noqa: E402
from aiecost.target import resolve_target  # noqa: E402

HERE = Path(__file__).resolve().parent
KERNEL = HERE / "d1_overlap.cc"
CHAINS = 4


def build(mmuls: int, n_tiles: int, out_dir: Path, target) -> tuple[Path, Path] | None:
    from aie.iron import ObjectFifo, Program, Runtime, Worker
    from aie.iron.controlflow import range_
    from aie.iron.kernel import ExternalFunction
    from aie.iron.placers import SequentialPlacer
    from aie.utils import set_current_device
    from aie.utils.compile import compile_external_kernel, compile_mlir_module

    dev = target.iron_device()
    set_current_device(dev)
    out_dir.mkdir(parents=True, exist_ok=True)
    xclbin = out_dir / f"d1-m{mmuls}-t{n_tiles}.xclbin"
    insts = out_dir / f"d1-m{mmuls}-t{n_tiles}-insts.bin"
    if xclbin.exists() and insts.exists():
        return xclbin, insts

    Tile: object = np.ndarray[(TILE_ELEM,), np.dtype[np.int32]]
    Acc: object = np.ndarray[(ACC_ELEM,), np.dtype[np.int32]]
    Stream: object = np.ndarray[(TILE_ELEM * n_tiles,), np.dtype[np.int32]]

    kern = ExternalFunction(
        "d1_overlap", source_file=str(KERNEL), arg_types=[Tile, Acc],
        include_dirs=include_dirs(_mlir_pkg, target),
        compile_flags=["-std=c++20", "-O2", f"-DMMULS={mmuls}", f"-DCHAINS={CHAINS}"],
    )
    fi = ObjectFifo(Tile, name="in0", depth=2)
    fo = ObjectFifo(Acc, name="out0", depth=1)

    def core(a_in, o_out, kk):
        eo = o_out.acquire(1)
        for _ in range_(n_tiles):
            ea = a_in.acquire(1)
            kk(ea, eo)
            a_in.release(1)
        o_out.release(1)

    w = Worker(core, [fi.cons(), fo.prod(), kern])
    rt = Runtime()
    with rt.sequence(Stream, Acc) as (src, dst):
        rt.start(w)
        rt.fill(fi.prod(), src)
        rt.drain(fo.cons(), dst, wait=True)
    try:
        module = Program(dev, rt).resolve_program(SequentialPlacer())
        with tempfile.TemporaryDirectory(prefix="aiecost_d1_") as tn:
            tmp = Path(tn)
            compile_external_kernel(kern, tmp, target_arch=target.target_arch)
            compile_mlir_module(mlir_module=module, insts_path=tmp / "insts.bin",
                                xclbin_path=tmp / "final.xclbin", work_dir=tmp)
            shutil.copy2(tmp / "final.xclbin", xclbin)
            shutil.copy2(tmp / "insts.bin", insts)
    except Exception as e:
        print(f"  MMULS={mmuls}: BUILD FAILED {type(e).__name__}: {str(e)[:70]}")
        return None
    return xclbin, insts


def run(xclbin: Path, insts: Path, n_tiles: int, reps: int) -> float | None:
    from aie.utils.hostruntime.xrtruntime.hostruntime import XRTHostRuntime
    from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor
    from aie.utils.npukernel import NPUKernel

    src = XRTTensor(np.ones((TILE_ELEM * n_tiles,), dtype=np.int32), dtype=np.int32, device="cpu")
    dst = XRTTensor((ACC_ELEM,), dtype=np.int32, device="cpu")
    hrt = XRTHostRuntime()
    h = hrt.load(NPUKernel(xclbin_path=str(xclbin), insts_path=str(insts), kernel_name="MLIR_AIE"))
    for _ in range(3):
        hrt.run(h, [src, dst])
    ts = sorted(float(hrt.run(h, [src, dst]).npu_time) * 1e-9 for _ in range(reps))
    return ts[len(ts) // 2]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mmuls", type=int, nargs="+", default=[0, 64, 128, 256, 512, 1024])
    p.add_argument("--tiles", type=int, default=1024)
    p.add_argument("--reps", type=int, default=7)
    p.add_argument("--device", default="auto", choices=["auto", "npu1", "npu2"])
    p.add_argument("--cache", default=str(Path.home() / ".cache" / "hipfire-aiecost" / "d1"))
    p.add_argument("--json", metavar="PATH")
    args = p.parse_args()

    target = resolve_target(args.device)
    consts = calib.load(calib.current_key())
    f_h = consts["f_h_hz"].value
    bw = consts["bw_feed_per_stream_bytes_s"].value
    floor = consts["c_cmd_s"].value
    tile_b = TILE_ELEM * 4
    t_feed_tile = tile_b / bw
    print(f"D1 overlap — 1 stream, {args.tiles} tiles x {tile_b} B, on {target.key}")
    print(f"  t_feed/tile = {t_feed_tile * 1e9:.0f} ns  (at {bw / 1e9:.2f} GB/s/stream)")
    print(f"  crossover at MMULS ~= {t_feed_tile * f_h / CHAINS:.0f}\n")
    print(f"  {'MMULS':>6} {'t_core/tile':>12} {'MEASURED':>11} {'max model':>10} {'sum model':>10}  verdict")

    rows = []
    for m in args.mmuls:
        built = build(m, args.tiles, Path(args.cache), target)
        if not built:
            continue
        t = run(*built, args.tiles, args.reps)
        if not t:
            continue
        per_tile = (t - floor) / args.tiles  # the C1 floor is per dispatch, not per tile
        t_core_tile = m * CHAINS / f_h
        pred_max = max(t_feed_tile, t_core_tile)
        pred_sum = t_feed_tile + t_core_tile
        e_max = (pred_max - per_tile) / per_tile
        e_sum = (pred_sum - per_tile) / per_tile
        verdict = "MAX" if abs(e_max) < abs(e_sum) else "SUM"
        rows.append({"mmuls": m, "npu_s": t, "per_tile_s": per_tile, "t_core_tile": t_core_tile,
                     "pred_max": pred_max, "pred_sum": pred_sum, "err_max": e_max, "err_sum": e_sum})
        print(f"  {m:>6} {t_core_tile * 1e9:>11.0f}n {per_tile * 1e9:>10.0f}n {pred_max * 1e9:>9.0f}n "
              f"{pred_sum * 1e9:>9.0f}n  {verdict} (max {e_max * 100:+.0f}%, sum {e_sum * 100:+.0f}%)")

    if rows:
        wins = sum(1 for r in rows if abs(r["err_max"]) < abs(r["err_sum"]))
        print(f"\n  max model closer on {wins}/{len(rows)} points")
        mape_max = sum(abs(r["err_max"]) for r in rows) / len(rows) * 100
        mape_sum = sum(abs(r["err_sum"]) for r in rows) / len(rows) * 100
        print(f"  mean |error|: max {mape_max:.1f}%   sum {mape_sum:.1f}%")
        print(f"  => the composition is {'MAX (feed and compute OVERLAP)' if mape_max < mape_sum else 'SUM (they SERIALISE — the model is wrong)'}")
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
