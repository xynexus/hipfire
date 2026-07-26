#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# hipfire — see LICENSE and NOTICE in the project root.
"""E2: bfp16 (block-float / MX) sustained rate on AIE2P.

Confirms the header/ISA result on hardware: bfp16 `mac_8x8_8x8T` runs natively at
int8 rate (512 MACs/native VMAC), not the emulated cost of true bf16. Mirrors K1's
resident-chain method — sweep ITERS, fit the slope, derive VMAC/s and MACs/s. The
inputs are random bytes: this measures THROUGHPUT, not numerics (block-float
packing + a reference compare is a separate step; see the design guide's Part 5
bfp16 note and BUGS.md on `mm_bfp.cc`).

Usage:
    python -m aiecost.benches.e2_bfp16 --device npu2
"""

from __future__ import annotations

import argparse
import shutil
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
KERNEL_SRC = HERE / "bfp16_chain.cc"
_mlir_pkg = next((Path(p) for p in sys.path if (Path(p) / "mlir_aie").is_dir()), None)

CHAINS = 4
BLOCK_BYTES = 72  # bfp16ebs8: 64 mantissa + 8 shared-exponent bytes per 64-value block
A_BYTES = 2 * BLOCK_BYTES  # kernel pops 2 A blocks
B_BYTES = 2 * BLOCK_BYTES  # and 2 B blocks
OUT_F32 = 96  # pOut[0..7] guards + 64-float store at +16, padded


def build(iters: int, out_dir: Path, target) -> tuple[Path, Path] | None:
    from aie.iron import ObjectFifo, Program, Runtime, Worker
    from aie.iron.kernel import ExternalFunction
    from aie.utils import set_current_device
    from aie.utils.compile import compile_external_kernel, compile_mlir_module

    iron_device = target.iron_device()
    set_current_device(iron_device)

    A: object = np.ndarray[(A_BYTES,), np.dtype[np.int8]]
    B: object = np.ndarray[(B_BYTES,), np.dtype[np.int8]]
    O: object = np.ndarray[(OUT_F32,), np.dtype[np.float32]]

    kern = ExternalFunction(
        "bfp16_chain", source_file=str(KERNEL_SRC), arg_types=[A, B, O],
        include_dirs=include_dirs(_mlir_pkg, target),
        compile_flags=["-std=c++20", "-O2", f"-DITERS={iters}", f"-DCHAINS={CHAINS}"],
    )
    fa, fb, fo = ObjectFifo(A, name="a", depth=1), ObjectFifo(B, name="b", depth=1), ObjectFifo(O, name="o", depth=1)

    def core(a_in, b_in, o_out, kk):
        ea, eb, eo = a_in.acquire(1), b_in.acquire(1), o_out.acquire(1)
        kk(ea, eb, eo)
        a_in.release(1)
        b_in.release(1)
        o_out.release(1)

    w = Worker(core, [fa.cons(), fb.cons(), fo.prod(), kern])
    rt = Runtime()
    with rt.sequence(A, B, O) as (a, b, o):
        rt.start(w)
        rt.fill(fa.prod(), a)
        rt.fill(fb.prod(), b)
        rt.drain(fo.cons(), o, wait=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{target.cache_tag}-i{iters}"
    xclbin, insts = out_dir / f"e2bfp-{tag}.xclbin", out_dir / f"e2bfp-{tag}-insts.bin"
    if xclbin.exists() and insts.exists():
        return xclbin, insts
    try:
        module = resolve_program(Program(iron_device, rt))
        with tempfile.TemporaryDirectory(prefix="aiecost_e2_") as tn:
            tmp = Path(tn)
            compile_external_kernel(kern, tmp, target_arch=target.target_arch)
            compile_mlir_module(mlir_module=module, insts_path=tmp / "insts.bin", xclbin_path=tmp / "final.xclbin", work_dir=tmp)
            shutil.copy2(tmp / "final.xclbin", xclbin)
            shutil.copy2(tmp / "insts.bin", insts)
    except Exception as e:
        print(f"    build failed: {type(e).__name__}: {str(e)[:140]}")
        return None
    return xclbin, insts


def run(xclbin: Path, insts: Path, reps: int) -> float | None:
    from aie.utils.hostruntime.xrtruntime.hostruntime import XRTHostRuntime
    from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor
    from aie.utils.npukernel import NPUKernel

    rng = np.random.default_rng(0)
    ta = XRTTensor(rng.integers(0, 255, size=(A_BYTES,), dtype=np.uint8).view(np.int8), dtype=np.int8, device="cpu")
    tb = XRTTensor(rng.integers(0, 255, size=(B_BYTES,), dtype=np.uint8).view(np.int8), dtype=np.int8, device="cpu")
    to = XRTTensor((OUT_F32,), dtype=np.float32, device="cpu")
    kernel = NPUKernel(xclbin_path=str(xclbin), insts_path=str(insts), kernel_name="MLIR_AIE")
    hrt = XRTHostRuntime()
    h = hrt.load(kernel)
    hrt.run(h, [ta, tb, to])  # warm
    best = float("inf")
    for _ in range(reps):
        r = hrt.run(h, [ta, tb, to])
        if getattr(r, "npu_time", None):
            best = min(best, float(r.npu_time) * 1e-9)
    return None if best == float("inf") else best


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--iters", type=int, nargs="+", default=[200_000, 400_000, 800_000, 1_600_000])
    p.add_argument("--reps", type=int, default=6)
    p.add_argument("--device", default="auto", choices=["auto", "npu1", "npu2"])
    p.add_argument("--cache", default=str(Path.home() / ".cache" / "hipfire-aiecost" / "e2"))
    args = p.parse_args()
    target = resolve_target(args.device)
    print(f"E2 bfp16 rate — target={target.cache_tag} chains={CHAINS}")
    pts = []
    for it in args.iters:
        built = build(it, Path(args.cache), target)
        if not built:
            return 1
        t = run(*built, args.reps)
        if t:
            pts.append((it, t))
            print(f"  iters={it:>9} npu={t * 1e3:9.4f} ms")
    if len(pts) < 2:
        print("not enough points")
        return 1
    n = len(pts)
    sx = sum(x for x, _ in pts); sy = sum(y for _, y in pts)
    sxx = sum(x * x for x, _ in pts); sxy = sum(x * y for x, y in pts)
    slope = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    vmac_s = CHAINS / slope
    macs_s = vmac_s * 512  # bfp16 <8,8,8> = 512 MACs/native VMAC
    print(f"\n  slope {slope * 1e9:.4f} ns/iter -> {vmac_s / 1e9:.3f} G VMAC/s, {macs_s / 1e9:.1f} G MACs/s/core (bfp16)")
    print(f"  cf int8 <8,8,8> ~532 G MACs/s/core: bfp16 is {'~int8-rate' if macs_s / 1e9 > 400 else 'SLOWER'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
