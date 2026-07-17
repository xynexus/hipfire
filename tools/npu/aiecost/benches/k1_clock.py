#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# hipfire — see LICENSE and NOTICE in the project root.
"""K1: measure the AIE compute clock (f_H) on AIE2 or AIE2P.

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
from aiecost.target import include_dirs, resolve_program, resolve_target  # noqa: E402

env.bootstrap()

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
KERNEL_SRC = HERE / "k1_vmac_chain.cc"

_mlir_pkg = next((Path(p) for p in sys.path if (Path(p) / "mlir_aie").is_dir()), None)
_AIE_INCLUDE = _mlir_pkg / "mlir_aie" / "include" if _mlir_pkg else None
_PEANO = next((Path(p) / "llvm-aie" for p in sys.path if (Path(p) / "llvm-aie").is_dir()), None)


def loop_cycles(mr: int, chains: int, target, cache: Path) -> dict:
    """Compile the kernel with Peano and count bundles in the loop body.

    AIE cores are statically-scheduled VLIW: one bundle issues per cycle with no
    dynamic stalls for register-resident compute, so the loop's bundle count IS
    its cycle count. Combined with the measured ns/iter this yields f_H directly
    — no throughput plateau required, which matters on AIE2P where the
    accumulator register file spills before enough chains hide the latency, so
    the plateau (and hence the II=1 saturation test) is never reached.

    The loop is `add` (counter) + `jnz` (branch) + a fixed branch shadow. At the
    highest non-spilled chain count the shadow is fully packed with VMACs and the
    accumulator read-out epilogue follows immediately, so counting bundles from
    the .LBB0_1 loop head to the first epilogue bundle is exact. (At low chain
    counts trailing NOPs are one-time loop-exit drain, not per-iteration cost, so
    the count is taken at the packed chain count.)
    """
    import re
    import subprocess

    if _PEANO is None or _AIE_INCLUDE is None:
        return {"ok": False, "reason": "peano/include not found"}
    cache.mkdir(parents=True, exist_ok=True)
    obj = cache / f"k1cyc-{target.cache_tag}-m{mr}-c{chains}.o"
    cmd = [
        str(_PEANO / "bin" / "clang"), f"--target={target.target_arch}-none-unknown-elf",
        "-std=c++20", f"-I{_AIE_INCLUDE}", "-O2", "-DITERS=1000", f"-DCHAINS={chains}",
        f"-DMR={mr}", "-DMK=8", "-DMN=8", "-DTA=int8", "-DTB=int8", "-c", str(KERNEL_SRC), "-o", str(obj),
    ]
    subprocess.run(cmd, capture_output=True, text=True, check=False)
    if not obj.exists():
        return {"ok": False, "reason": "compile failed"}
    # No -z on purpose: the executed loop body ends at the branch shadow, after
    # which the linker aligns the epilogue with zero padding. objdump renders that
    # padding as a "..." marker (with -z it would be spelled out and miscounted as
    # loop cycles — inflating f_H above the die's clock ceiling). The "..." is the
    # exact end-of-executed-shadow signal for a packed loop.
    d = subprocess.run(
        [str(_PEANO / "bin" / "llvm-objdump"), "-d", "--no-show-raw-insn", str(obj)],
        capture_output=True, text=True, check=False,
    )
    loop: list[str] = []
    inside = False
    for line in d.stdout.splitlines():
        if re.search(r"<\.LBB0_1>:", line):
            inside = True
            continue
        if inside:
            if line.strip() == "...":  # alignment padding after the shadow — loop is done
                break
            # epilogue begins when an accumulator (bm* register) is read out, or
            # the store pointer is set up — either marks the end of the loop body.
            if re.search(r"(vmov\s+\w+,\s*bm|vst\s+bm|movs?\s+p0,\s*p2)", line):
                break
            if not line.strip():
                break
            loop.append(line)
    if not loop:
        return {"ok": False, "reason": "no loop body found"}
    vmac = sum(ln.count("vmac") for ln in loop)
    # The host counts SOURCE iterations; Peano may unroll the loop, so recover the
    # unroll factor from the induction-variable decrement (add rN, rN, #-0xU) and
    # report cycles PER SOURCE ITERATION, which is what the ITERS slope measures.
    unroll = 1
    if mo := re.search(r"add\s+r\d+,\s*r\d+,\s*#-0x([0-9a-f]+)", "\n".join(loop)):
        unroll = int(mo.group(1), 16) or 1
    return {
        "ok": True,
        "cycles": len(loop),
        "unroll": unroll,
        "cycles_per_iter": len(loop) / unroll,
        "vmac": vmac,
        "vmac_per_call": vmac / (unroll * chains) if chains else float("nan"),
        "native": abs(vmac / (unroll * chains) - 1.0) < 1e-6 if chains else False,
        "body": loop,
    }


def vmac_geometry(device: str = "auto") -> tuple[int, int, int, int]:
    """Return MR, two-A bytes, two-B bytes, and output i32 elements.

    K1 is a clock probe: at II=1 the VMAC issue rate is f_H regardless of tile
    shape, so both targets use the small MR=4 accumulator. On AIE2P the 8x8x8
    accumulator (64 int32) spills the register file before enough chains hide
    the accumulator latency — the sweep collapsed at 8 chains instead of
    plateauing. MR=4 (32 int32) leaves room to push chains to the plateau.
    """
    resolve_target(device)  # validate the target is supported
    mr, mk, mn = 4, 8, 8
    return mr, 2 * mr * mk, 2 * mk * mn, 8 + mr * mn


def build(iters: int, chains: int, out_dir: Path, device: str = "auto") -> tuple[Path, Path]:
    """Compile one (iters, chains) point. Returns (xclbin, insts)."""
    from aie.iron import ObjectFifo, Program, Runtime, Worker
    from aie.iron.kernel import ExternalFunction
    from aie.utils import set_current_device
    from aie.utils.compile import compile_external_kernel, compile_mlir_module

    target = resolve_target(device)
    iron_device = target.iron_device()
    set_current_device(iron_device)
    mr, a_n, b_n, out_n = vmac_geometry(target.key)

    A_ty: object = np.ndarray[(a_n,), np.dtype[np.int8]]
    B_ty: object = np.ndarray[(b_n,), np.dtype[np.int8]]
    O_ty: object = np.ndarray[(out_n,), np.dtype[np.int32]]

    kern = ExternalFunction(
        "k1_vmac_chain",
        source_file=str(KERNEL_SRC),
        arg_types=[A_ty, B_ty, O_ty],
        include_dirs=include_dirs(_mlir_pkg, target),
        compile_flags=[
            "-std=c++20",
            "-O2",
            f"-DITERS={iters}",
            f"-DCHAINS={chains}",
            f"-DMR={mr}",
            "-DMK=8",
            "-DMN=8",
        ],
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

    module = resolve_program(Program(iron_device, rt))

    out_dir.mkdir(parents=True, exist_ok=True)
    xclbin = out_dir / f"k1-{target.cache_tag}-meta2-m{mr}-i{iters}-c{chains}.xclbin"
    insts = out_dir / f"k1-{target.cache_tag}-meta2-m{mr}-i{iters}-c{chains}-insts.bin"
    if xclbin.exists() and insts.exists():
        return xclbin, insts

    with tempfile.TemporaryDirectory(prefix="aiecost_k1_") as tmpname:
        tmp = Path(tmpname)
        compile_external_kernel(kern, tmp, target_arch=target.target_arch)
        compile_mlir_module(mlir_module=module, insts_path=tmp / "insts.bin", xclbin_path=tmp / "final.xclbin", work_dir=tmp)
        shutil.copy2(tmp / "final.xclbin", xclbin)
        shutil.copy2(tmp / "insts.bin", insts)
    return xclbin, insts


def run(xclbin: Path, insts: Path, reps: int, device: str = "auto") -> dict:
    """Run the kernel `reps` times; return best device + wall time and counters."""
    from aie.utils.hostruntime.xrtruntime.hostruntime import XRTHostRuntime
    from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor
    from aie.utils.npukernel import NPUKernel

    _mr, a_n, b_n, out_n = vmac_geometry(device)
    rng = np.random.default_rng(0)
    a = rng.integers(-8, 8, size=(a_n,), dtype=np.int8)
    b = rng.integers(-8, 8, size=(b_n,), dtype=np.int8)

    t_a = XRTTensor(a, dtype=np.int8, device="cpu")
    t_b = XRTTensor(b, dtype=np.int8, device="cpu")
    t_o = XRTTensor((out_n,), dtype=np.int32, device="cpu")

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
        "reported_iters": int(o[4]),
        "reported_chains": int(o[5]),
        "vmacs": int(o[6]),
        "macs_per_vmac": int(o[7]),
        "output_head": [int(value) for value in o[:8]],
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


def select_plateau(results: dict[int, dict]) -> tuple[tuple[int, int] | None, float]:
    """Return an adjacent non-regressing plateau and its mean VMAC/s."""
    order = sorted(results)
    for left, right in zip(order, order[1:]):
        left_rate = results[left]["vmac_per_s"]
        right_rate = results[right]["vmac_per_s"]
        gain = right_rate / left_rate
        if 0.90 <= gain <= 1.10:
            return (left, right), (left_rate + right_rate) / 2
    return None, max((results[c]["vmac_per_s"] for c in order), default=float("nan"))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--iters", type=int, nargs="+", default=[200_000, 400_000, 800_000, 1_600_000])
    p.add_argument("--chains", type=int, nargs="+", default=[1, 2, 4, 8, 12, 16])
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--device", default="auto", choices=["auto", "npu1", "npu2"])
    p.add_argument("--cache", default=str(Path.home() / ".cache" / "hipfire-aiecost" / "k1"))
    p.add_argument("--json", metavar="PATH")
    p.add_argument("--save", action="store_true", help="record admissible f_H in the current calibration")
    args = p.parse_args()

    target = resolve_target(args.device)
    cache = Path(args.cache)
    print(f"K1 clock: target={target.cache_tag} chains={args.chains} iters={args.iters} reps={args.reps}")
    results: dict[int, dict] = {}

    for chains in args.chains:
        pts_npu: list[tuple[int, float]] = []
        pts_wall: list[tuple[int, float]] = []
        for iters in args.iters:
            xclbin, insts = build(iters, chains, cache, target.key)
            r = run(xclbin, insts, args.reps, target.key)
            if r["reported_iters"] != iters or r["reported_chains"] != chains:
                print(
                    f"  !! kernel reported iters={r['reported_iters']} chains={r['reported_chains']}, "
                    f"expected {iters}/{chains}; output_head={r['output_head']}"
                )
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
    # Saturation (plateau) check — informational. On AIE2P it never plateaus:
    # the register file spills before enough chains hide the accumulator latency.
    plateau, f_h_rate = select_plateau(results)
    if plateau:
        left, right = plateau
        gain = results[right]["vmac_per_s"] / results[left]["vmac_per_s"]
        print(f"\n  chains {left} -> {right} changed throughput by {gain:.3f}x")
        sat_verdict = "PLATEAU — II=1 directly observable"
    elif len(order) >= 2 and results[order[-1]]["vmac_per_s"] < 0.9 * results[order[-2]]["vmac_per_s"]:
        sat_verdict = "REGRESSION — highest chain count spills before saturation"
    else:
        sat_verdict = "STILL SCALING — latency not hidden at these chain counts"
    print(f"  saturation verdict: {sat_verdict}")

    # Bundle method (primary): the AIE core is statically-scheduled VLIW, so the
    # loop's bundle count is its cycle count and f_H = cycles / (ns/iter) — no
    # throughput plateau required. Count bundles at the highest non-spilled chain
    # count, where the branch shadow is packed with VMACs (trailing NOPs at low
    # chain counts are one-time loop-exit drain, not per-iteration cost).
    mr = vmac_geometry(target.key)[0]
    max_vmac_s = max(results[c]["vmac_per_s"] for c in order)
    clean = [c for c in order if results[c]["vmac_per_s"] >= 0.5 * max_vmac_s]
    best = max(clean, key=lambda c: results[c]["vmac_per_s"])
    lc = loop_cycles(mr, best, target, cache)

    admissible = False
    f_h = f_h_rate / 1e6  # fallback: the plateau lower-bound estimate, in MHz
    if lc.get("ok"):
        cps = lc["cycles_per_iter"]
        per_chain = {c: cps / results[c]["slope_s_per_iter"] for c in clean}
        fhs = list(per_chain.values())
        spread = (max(fhs) / min(fhs)) if min(fhs) > 0 else float("inf")
        native = "native 1 vmac/mac()" if lc["native"] else f"{lc['vmac_per_call']:.2f} vmac/mac()"
        print(
            f"\n  ISA loop (chains={best}, mmul<{mr},8,8> int8): {lc['cycles']} bundles, "
            f"unroll {lc['unroll']} -> {cps:.1f} cyc/iter, {lc['vmac']} vmac ({native})"
        )
        print("  f_H = cyc/iter ÷ (ns/iter), cross-checked across non-spilled chains:")
        for c in clean:
            print(
                f"    chains={c}: {cps:.1f} cyc / {results[c]['slope_s_per_iter'] * 1e9:.3f} ns "
                f"= {per_chain[c] / 1e9:.3f} GHz"
            )
        admissible = spread <= 1.15 and len(clean) >= 2
        f_h = per_chain[best] / 1e6
        print(
            f"  spread {spread:.3f}× across {len(clean)} chain counts — "
            f"{'CONSISTENT, admissible' if admissible else 'inconsistent, NOT admissible'}"
        )
        print(f"\n  K1: f_H = {f_h / 1e3:.3f} GHz" + ("" if admissible else "   [NOT ADMISSIBLE]"))
    else:
        print(f"\n  bundle method unavailable ({lc.get('reason', '?')}); f_H ~= {f_h:.1f} MHz [NOT ADMISSIBLE]")

    if args.json:
        Path(args.json).write_text(json.dumps({
            "results": {str(k): v for k, v in results.items()},
            "loop_cycles": lc.get("cycles"), "unroll": lc.get("unroll"),
            "cycles_per_iter": lc.get("cycles_per_iter"), "loop_vmac": lc.get("vmac"),
            "best_chains": best if lc.get("ok") else None,
            "admissible": admissible, "f_h_mhz": f_h,
        }, indent=2))
        print(f"  wrote {args.json}")

    if args.save:
        if not admissible:
            print("  refusing to save f_H: the ISA bundle-count cross-check is not consistent")
            return 2
        from aiecost import calib

        key = calib.current_key()
        cps = lc["cycles_per_iter"]
        evidence = [
            f"ISA: mmul<{mr},8,8> int8 loop = {lc['cycles']} bundles, unroll {lc['unroll']} "
            f"-> {cps:.1f} cyc/iter, {lc['vmac']} vmac "
            f"({'native' if lc['native'] else f'{lc['vmac_per_call']:.2f}/mac()'}), "
            f"count taken at packed chains={best}"
        ]
        evidence += [
            f"chains={c}: {cps:.1f} cyc / {results[c]['slope_s_per_iter'] * 1e9:.3f} ns "
            f"= {cps / results[c]['slope_s_per_iter'] / 1e9:.3f} GHz"
            for c in clean
        ]
        constants = {
            "f_h_hz": calib.Constant(
                name="f_h_hz",
                value=f_h * 1e6,
                unit="Hz",
                bench="K1",
                method=(
                    f"ISA bundle count ÷ ITERS slope: the statically-scheduled VLIW loop is "
                    f"{cps:.1f} cycles per source iteration; f_H = cyc/iter ÷ (ns/iter), "
                    f"cross-checked across non-spilled chain counts"
                ),
                admissible=True,
                evidence=evidence,
                caveats=[
                    "AIE2P never reaches the II=1 throughput plateau (register spill precedes "
                    "saturation); the clock is read from the ISA bundle count, not a plateau",
                    "power mode must be recorded with every use",
                ],
            )
        }
        print(
            "  saved -> "
            f"{calib.save(key, constants, meta={'device': target.key, 'tile_isa': target.tile_isa})}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
