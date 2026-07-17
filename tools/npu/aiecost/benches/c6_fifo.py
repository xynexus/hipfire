#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# hipfire — see LICENSE and NOTICE in the project root.
"""C6: pipeline fill/drain cost vs ObjectFIFO depth — fill_drain_s.

Same payload, same tiles, same kernel; only the FIFO depth changes. Depth 1 is
strictly sequential (fetch, compute, fetch, ...); depth >= 2 lets the DMA
prefetch tile N+1 while the core works on N. The difference isolates what
buffering buys and what the pipeline costs to fill and drain.

aie2p reference points, which frame the expectation rather than predict it:
  - R58/E2: fifo_depth 2 -> 15.3 TOPS, 3 -> 15.7 (+2.5%), 4 -> build failure.
    "marginal; DMA already hidden."
  - R60: a depth-3 activation FIFO exceeded tile SRAM beside the weight double
    buffer; depth 1 was the correct sequential schedule.
Depth is bounded by L1 (M1: 64 KiB) — depth * tile_bytes must fit alongside
everything else the core holds.

Usage:
    python -m aiecost.benches.c6_fifo --save
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
KERNEL_SRC = HERE / "c2_sink.cc"

_mlir_pkg = next((Path(p) for p in sys.path if (Path(p) / "mlir_aie").is_dir()), None)
AIE_INCLUDE = _mlir_pkg / "mlir_aie" / "include" if _mlir_pkg else None
AIE_RUNTIME_LIB = _mlir_pkg / "mlir_aie" / "aie_runtime_lib" / "AIE2" if _mlir_pkg else None

TILE_ELEM = 1024  # 4 KiB
ACC_ELEM = 16


def build(depth: int, n_tiles: int, out_dir: Path) -> tuple[Path, Path] | None:
    from aie.iron import ObjectFifo, Program, Runtime, Worker
    from aie.iron.controlflow import range_
    from aie.iron.device import NPU1
    from aie.iron.kernel import ExternalFunction
    from aie.iron.placers import SequentialPlacer
    from aie.utils import set_current_device
    from aie.utils.compile import compile_external_kernel, compile_mlir_module

    set_current_device(NPU1())

    out_dir.mkdir(parents=True, exist_ok=True)
    xclbin = out_dir / f"c6-d{depth}-t{n_tiles}.xclbin"
    insts = out_dir / f"c6-d{depth}-t{n_tiles}-insts.bin"
    if xclbin.exists() and insts.exists():
        return xclbin, insts

    Tile: object = np.ndarray[(TILE_ELEM,), np.dtype[np.int32]]
    Acc: object = np.ndarray[(ACC_ELEM,), np.dtype[np.int32]]
    Stream: object = np.ndarray[(TILE_ELEM * n_tiles,), np.dtype[np.int32]]

    kern = ExternalFunction(
        "c2_sink", source_file=str(KERNEL_SRC), arg_types=[Tile, Acc],
        include_dirs=[str(AIE_INCLUDE), str(AIE_RUNTIME_LIB)], compile_flags=["-std=c++20", "-O2"],
    )

    of_in = ObjectFifo(Tile, name="in0", depth=depth)
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
        module = Program(NPU1(), rt).resolve_program(SequentialPlacer())
        with tempfile.TemporaryDirectory(prefix="aiecost_c6_") as tmpname:
            tmp = Path(tmpname)
            compile_external_kernel(kern, tmp, target_arch="aie2")
            compile_mlir_module(mlir_module=module, insts_path=tmp / "insts.bin", xclbin_path=tmp / "final.xclbin", work_dir=tmp)
            shutil.copy2(tmp / "final.xclbin", xclbin)
            shutil.copy2(tmp / "insts.bin", insts)
    except Exception as e:
        print(f"  depth={depth}: BUILD FAILED ({type(e).__name__}) — depth*tile likely exceeds L1 (M1)")
        return None
    return xclbin, insts


def run(xclbin: Path, insts: Path, n_tiles: int, reps: int, warmup: int) -> float | None:
    from aie.utils.hostruntime.xrtruntime.hostruntime import XRTHostRuntime
    from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor
    from aie.utils.npukernel import NPUKernel

    src = XRTTensor(np.ones((TILE_ELEM * n_tiles,), dtype=np.int32), dtype=np.int32, device="cpu")
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
    return statistics.median(npu) if npu else None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--depths", type=int, nargs="+", default=[1, 2, 3, 4, 8])
    p.add_argument("--tiles", type=int, default=1024)
    p.add_argument("--reps", type=int, default=12)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--cache", default=str(Path.home() / ".cache" / "hipfire-aiecost" / "c6"))
    p.add_argument("--save", action="store_true")
    p.add_argument("--json", metavar="PATH")
    args = p.parse_args()

    total = TILE_ELEM * args.tiles * 4
    print(f"C6 fifo depth: depths={args.depths} tiles={args.tiles} (tile={TILE_ELEM * 4} B, total={total / 1024:.0f} KiB)")
    rows = {}
    for d in args.depths:
        built = build(d, args.tiles, Path(args.cache))
        if not built:
            rows[d] = None
            continue
        t = run(*built, args.tiles, args.reps, args.warmup)
        rows[d] = t
        if t:
            print(f"  depth={d}: npu={t * 1e6:9.2f} us   rate={total / t / 1e9:6.3f} GB/s")

    ok = {d: t for d, t in rows.items() if t}
    if len(ok) < 2:
        print("insufficient results")
        return 1

    print("=" * 78)
    base = ok.get(1)
    if base:
        for d in sorted(ok):
            print(f"  depth={d}: {ok[d] * 1e6:9.2f} us   vs depth1: {base / ok[d]:5.3f}x")
    best_d = min(ok, key=lambda d: ok[d])
    # fill/drain cost attributable to depth: the residual over the best pipelined time
    fill = max(0.0, (ok[best_d] - min(ok.values()))) / max(1, best_d)
    print(f"\n  best depth = {best_d} at {ok[best_d] * 1e6:.2f} us")
    failed = [d for d, t in rows.items() if t is None]
    if failed:
        print(f"  build failures at depth {failed} — L1 (M1) bounds depth*tile_bytes")

    if args.json:
        Path(args.json).write_text(json.dumps({str(d): t for d, t in rows.items()}, indent=2))
    if args.save:
        from aiecost import calib

        key = calib.current_key()
        ev = [f"depth={d}: {(t * 1e6 if t else 0):.2f} us" + ("" if t else " BUILD FAILED") for d, t in sorted(rows.items())]
        if base:
            ev.append(f"depth1 -> best({best_d}): {base / ok[best_d]:.3f}x")
        cs = {
            "fill_drain_s": calib.Constant(
                name="fill_drain_s", value=fill, unit="s", bench="C6",
                method="fifo-depth sweep at constant payload; per-depth residual over the best pipelined time",
                admissible=True, evidence=ev,
                caveats=[
                    "small term on npu1: the C1 dispatch floor (~155 us) dwarfs it",
                    "L1 (M1) bounds depth*tile_bytes; aie2p saw depth-4 build failures for the same reason (R58/E2)",
                ],
            )
        }
        print(f"  saved -> {calib.save(key, cs)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
