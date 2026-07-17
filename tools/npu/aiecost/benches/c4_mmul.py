#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# hipfire — see LICENSE and NOTICE in the project root.
"""C4: MMUL shape throughput — cyc_per_vmac, and which shapes are NATIVE.

The question this answers: does a bigger aie::mmul shape buy anything on AIE2,
or does the API just emulate it with several native VMACs?

K1 established f_H = 1.015 GHz with mmul<4,8,8> issuing 1 VMAC/cycle, which
gives 16 cores x 256 MACs x 2 x 1.015 GHz = 8.3 TOPS — half the ~16 TOPS
marketing figure. That gap has exactly two explanations:
  (a) a wider native shape exists (aie2p uses 8x8x8 = 512 MACs), or
  (b) the marketing number is not dense int8 on 16 cores.
Distinguishing them is worth 2x on every GEMM-shaped kernel, so it is measured
rather than assumed.

Method — two independent checks that must agree:

  1. ISA (no hardware): disassemble and count `vmac` in the loop body. A shape
     that is native emits 1 VMAC per mac(); a virtual shape emits several.
     This is the decisive test: MACs-per-native-VMAC is the hardware's real
     throughput, and source-level shape cannot change it.
  2. Hardware: measure the ITERS slope per shape and derive MACs/s. If a shape
     is virtual, its MACs/s will match the native shape's despite doing more
     "work" per source call.

Reports MACs/cycle/core, which is the number that actually sets peak TOPS.

Usage:
    python -m aiecost.benches.c4_mmul --save
    python -m aiecost.benches.c4_mmul --isa-only     # no NPU needed
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aiecost import env  # noqa: E402
from aiecost.target import include_dirs, resolve_program, resolve_target  # noqa: E402

env.bootstrap()

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
KERNEL_SRC = HERE / "k1_vmac_chain.cc"

_mlir_pkg = next((Path(p) for p in sys.path if (Path(p) / "mlir_aie").is_dir()), None)
AIE_INCLUDE = _mlir_pkg / "mlir_aie" / "include" if _mlir_pkg else None
PEANO = next((Path(p) / "llvm-aie" for p in sys.path if (Path(p) / "llvm-aie").is_dir()), None)

CHAINS = 4
UNROLL_HINT = 4  # Peano unrolls the chain loop 4x; verified from the counter decrement

# (M, K, N, TypeA, TypeB). int8xint4 has its OWN shape family (mmul_8_4), and the
# valid shapes differ by generation because the vector width does:
#   AIE2  (npu1, 256-bit): 4x16x8, 8x16x8, 4x32x8
#   AIE2P (npu2, 512-bit): mmul_8_4 specializes ONLY <4,16,16>, and it is VIRTUAL
#     (2 native VMACs of 512 MACs each) — int4 buys NO compute density over int8
#     on AIE2P, only halved weight-DMA bytes. int4xint4 has no mmul family at all.
_INT8_SHAPES = [
    (4, 8, 8, "int8", "int8"),
    (8, 8, 8, "int8", "int8"),
    (2, 8, 8, "int8", "int8"),
]
_INT4_SHAPES = {
    "npu1": [(4, 16, 8, "int8", "int4"), (8, 16, 8, "int8", "int4"), (4, 32, 8, "int8", "int4")],
    "npu2": [(4, 16, 16, "int8", "int4")],
}


def shapes_for(target) -> list:
    return _INT8_SHAPES + _INT4_SHAPES.get(target.key, [])


_BITS = {"int8": 8, "int4": 4, "int16": 16, "uint8": 8, "uint4": 4, "uint16": 16}


def sizes(shape) -> dict:
    m, k, n = shape[0], shape[1], shape[2]
    ta, tb = (shape[3], shape[4]) if len(shape) > 3 else ("int8", "int8")
    # Sub-byte operands pack: an int4 element is half a byte on the wire.
    return {
        "size_A": m * k, "size_B": k * n, "size_C": m * n, "macs": m * k * n,
        "ta": ta, "tb": tb,
        "bytes_A": m * k * _BITS[ta] // 8, "bytes_B": k * n * _BITS[tb] // 8,
    }


def isa_probe(shape, tmpdir: Path, target) -> dict:
    """Compile + disassemble only. Returns vmac counts; no hardware needed."""
    m, k, n = shape[0], shape[1], shape[2]
    s_ = sizes(shape)
    obj = tmpdir / f"c4_{m}{k}{n}_{s_['ta']}{s_['tb']}.o"
    dis = tmpdir / f"c4_{m}{k}{n}_{s_['ta']}{s_['tb']}.dis"
    cmd = [
        str(PEANO / "bin" / "clang"), f"--target={target.target_arch}-none-unknown-elf", "-std=c++20",
        f"-I{AIE_INCLUDE}", "-O2", "-DITERS=1000", f"-DCHAINS={CHAINS}",
        f"-DMR={m}", f"-DMK={k}", f"-DMN={n}", f"-DTA={s_['ta']}", f"-DTB={s_['tb']}",
        "-c", str(KERNEL_SRC), "-o", str(obj),
    ]
    # check=False on purpose: a shape that refuses to compile is a RESULT, not an error.
    r = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if not obj.exists():
        err = [ln for ln in r.stderr.splitlines() if "error" in ln.lower()]
        return {"shape": shape, "compiles": False, "error": err[0][:120] if err else "unknown"}

    d = subprocess.run([str(PEANO / "bin" / "llvm-objdump"), "-d", "--no-show-raw-insn", str(obj)],
                       capture_output=True, text=True, check=False)
    dis.write_text(d.stdout)

    # Loop body = from the .LBB label to the blank line after it.
    body: list[str] = []
    inside = False
    for line in d.stdout.splitlines():
        if re.search(r"<\.LBB0_1>:", line):
            inside = True
            continue
        if inside:
            if not line.strip():
                break
            body.append(line)
    loop_vmac = sum(ln.count("vmac") for ln in body)

    # Unroll factor from the induction-variable decrement.
    unroll = UNROLL_HINT
    if mo := re.search(r"add\s+r\d+,\s*r\d+,\s*#-0x([0-9a-f]+)", "\n".join(body)):
        unroll = int(mo.group(1), 16)

    mac_calls = unroll * CHAINS
    per_call = loop_vmac / mac_calls if mac_calls else float("nan")
    s = sizes(shape)
    return {
        "shape": shape,
        "compiles": True,
        "loop_vmac": loop_vmac,
        "unroll": unroll,
        "mac_calls": mac_calls,
        "vmac_per_call": per_call,
        "macs_per_call": s["macs"],
        "macs_per_vmac": s["macs"] / per_call if per_call else float("nan"),
        "native": abs(per_call - 1.0) < 1e-6,
        "ta": s["ta"], "tb": s["tb"],
    }


def build(shape, iters: int, out_dir: Path, target=None, cores: int = 1) -> tuple[Path, Path] | None:
    from aie.iron import ObjectFifo, Program, Runtime, Worker
    from aie.iron.kernel import ExternalFunction
    from aie.utils import set_current_device
    from aie.utils.compile import compile_external_kernel, compile_mlir_module

    if target is None:  # match c1/c2's device="auto" convention for external callers
        target = resolve_target("auto")
    iron_device = target.iron_device()
    set_current_device(iron_device)
    m, k, n = shape[0], shape[1], shape[2]
    s = sizes(shape)
    # Buffers are declared in BYTES as int8 memrefs; the kernel reinterprets the
    # pointer as TA*/TB*. int4 packs two elements per byte, so a sub-byte operand
    # needs half the bytes its element count suggests.
    a_n, b_n = 2 * s["bytes_A"], 2 * s["bytes_B"]
    out_n = 8 + s["size_C"] + 8  # kernel stores counters at [0..3] then a size_C vector at +8

    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{target.cache_tag}-{m}{k}{n}-{s['ta']}{s['tb']}-i{iters}" + (f"-x{cores}j" if cores > 1 else "")
    xclbin, insts = out_dir / f"c4-{tag}.xclbin", out_dir / f"c4-{tag}-insts.bin"
    if xclbin.exists() and insts.exists():
        return xclbin, insts

    A: object = np.ndarray[(a_n,), np.dtype[np.int8]]
    B: object = np.ndarray[(b_n,), np.dtype[np.int8]]
    O: object = np.ndarray[(out_n,), np.dtype[np.int32]]

    # Multi-core shares one over-allocated tile type for both operands (see the
    # cores>1 branch below), so the kernel must be DECLARED with those types —
    # otherwise func.call gets memref<128xi8> where it declared memref<64xi8> and
    # MLIR verification fails.
    IN: object = np.ndarray[(max(a_n, b_n),), np.dtype[np.int8]]
    kern_args = [A, B, O] if cores <= 1 else [IN, IN, O]

    kern = ExternalFunction(
        "k1_vmac_chain", source_file=str(KERNEL_SRC), arg_types=kern_args,
        include_dirs=include_dirs(_mlir_pkg, target),
        compile_flags=["-std=c++20", "-O2", f"-DITERS={iters}", f"-DCHAINS={CHAINS}",
                       f"-DMR={m}", f"-DMK={k}", f"-DMN={n}", f"-DTA={s['ta']}", f"-DTB={s['tb']}"],
    )
    def core(a_in, b_in, o_out, kk):
        ea, eb, eo = a_in.acquire(1), b_in.acquire(1), o_out.acquire(1)
        kk(ea, eb, eo)
        a_in.release(1)
        b_in.release(1)
        o_out.release(1)

    rt = Runtime()
    if cores <= 1:
        fa, fb, fo = ObjectFifo(A, name="a", depth=1), ObjectFifo(B, name="b", depth=1), ObjectFifo(O, name="o", depth=1)
        w = Worker(core, [fa.cons(), fb.cons(), fo.prod(), kern])
        with rt.sequence(A, B, O) as (a, b, o):
            rt.start(w)
            rt.fill(fa.prod(), a)
            rt.fill(fb.prod(), b)
            rt.drain(fo.cons(), o, wait=True)
    else:
        # Multi-core, up to the full array. Two constraints shape this topology:
        #
        #  1. H4 caps DPU data-args at 5 BOs. Per-worker outputs blow that at
        #     N=4. Instead each column's cores JOIN into one fifo at that
        #     column's memtile, so outputs cost 1 BO per COLUMN, not per core:
        #     1 shared source + 4 column outputs = 5 BOs for all 16 cores.
        #  2. A join is placed on a memtile, and a memtile serves only its own
        #     column — joining all 16 cores into ONE fifo is unplaceable
        #     ("Failed to find a tile matching column 0"). Hence per-column.
        #
        # Inputs are BROADCAST: one a-fifo and one b-fifo, whose cons() every
        # worker shares (an objectfifo's consumer list is native broadcast). That
        # keeps input DMA at 2 tasks regardless of core count — R61 found ~32
        # independent shim transfers exceed shim DMA capacity.
        #
        # Both fifos carry the same over-allocated tile type and are filled from
        # one source BO; the kernel reads only size_A / size_B from each pointer.
        # Operand VALUES are irrelevant to a throughput/energy bench.
        cols = min(cores, target.compute_columns)
        per_col = -(-cores // cols)
        BIG: object = np.ndarray[(out_n * per_col,), np.dtype[np.int32]]
        fa = ObjectFifo(IN, name="a", depth=1)
        fb = ObjectFifo(IN, name="b", depth=1)
        bigs, subs = [], []
        for c in range(cols):
            b = ObjectFifo(BIG, name=f"out{c}", depth=1)
            bigs.append(b)
            subs.append(
                b.prod().join(
                    offsets=[i * out_n for i in range(per_col)],
                    obj_types=[O] * per_col,
                    names=[f"o{c}_{i}" for i in range(per_col)],
                )
            )
        ws = [
            Worker(core, [fa.cons(), fb.cons(), subs[c][i].prod(), kern])
            for c in range(cols)
            for i in range(per_col)
        ]
        with rt.sequence(IN, *([BIG] * cols)) as args:
            src, outs = args[0], args[1:]
            for w in ws:
                rt.start(w)
            rt.fill(fa.prod(), src)
            rt.fill(fb.prod(), src)
            for c in range(cols):
                rt.drain(bigs[c].cons(), outs[c], wait=True)

    try:
        module = resolve_program(Program(iron_device, rt))
        with tempfile.TemporaryDirectory(prefix="aiecost_c4_") as tn:
            tmp = Path(tn)
            compile_external_kernel(kern, tmp, target_arch=target.target_arch)
            compile_mlir_module(mlir_module=module, insts_path=tmp / "insts.bin",
                                xclbin_path=tmp / "final.xclbin", work_dir=tmp)
            shutil.copy2(tmp / "final.xclbin", xclbin)
            shutil.copy2(tmp / "insts.bin", insts)
    except Exception as e:
        print(f"    build failed: {type(e).__name__}: {str(e)[:100]}")
        return None
    return xclbin, insts


def tensors_for(shape, cores: int = 1, columns: int = 4) -> list:
    """Host buffers matching build()'s arg layout for `cores`."""
    from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor

    s = sizes(shape)
    rng = np.random.default_rng(0)
    out_n = 8 + s["size_C"] + 8
    if cores <= 1:
        return [
            XRTTensor(rng.integers(-8, 8, size=(2 * s["bytes_A"],), dtype=np.int8), dtype=np.int8, device="cpu"),
            XRTTensor(rng.integers(-8, 8, size=(2 * s["bytes_B"],), dtype=np.int8), dtype=np.int8, device="cpu"),
            XRTTensor((out_n,), dtype=np.int32, device="cpu"),
        ]
    # Multi-core: one shared source BO feeds every A and B fifo, and outputs are
    # joined per COLUMN, so there is one output BO per column (see build()).
    n = max(2 * s["bytes_A"], 2 * s["bytes_B"])
    src = XRTTensor(rng.integers(-8, 8, size=(n,), dtype=np.int8), dtype=np.int8, device="cpu")
    cols = min(cores, columns)
    per_col = -(-cores // cols)
    return [src] + [XRTTensor((out_n * per_col,), dtype=np.int32, device="cpu") for _ in range(cols)]


def run(xclbin: Path, insts: Path, shape, reps: int, cores: int = 1) -> float | None:
    from aie.utils.hostruntime.xrtruntime.hostruntime import XRTHostRuntime
    from aie.utils.npukernel import NPUKernel

    args = tensors_for(shape, cores)
    kernel = NPUKernel(xclbin_path=str(xclbin), insts_path=str(insts), kernel_name="MLIR_AIE")
    hrt = XRTHostRuntime()
    h = hrt.load(kernel)
    npu = []
    for _ in range(reps):
        r = hrt.run(h, args)
        if getattr(r, "npu_time", None):
            npu.append(float(r.npu_time) * 1e-9)
    return min(npu) if npu else None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--iters", type=int, nargs="+", default=[400_000, 800_000, 1_600_000])
    p.add_argument("--reps", type=int, default=6)
    p.add_argument("--isa-only", action="store_true", help="disassembly check only; no NPU")
    p.add_argument("--cache", default=str(Path.home() / ".cache" / "hipfire-aiecost" / "c4"))
    p.add_argument("--save", action="store_true")
    p.add_argument("--json", metavar="PATH")
    p.add_argument("--device", default="auto", choices=["auto", "npu1", "npu2"])
    args = p.parse_args()

    from aiecost import calib

    target = resolve_target(args.device)
    shapes = shapes_for(target)
    consts = calib.load(calib.current_key())
    f_h = consts["f_h_hz"].value if "f_h_hz" in consts else None

    print(f"C4 — MMUL shape: native or virtual?  target={target.cache_tag}\n")
    print("ISA check (vmac emitted per source mac() call):")
    print(f"  {'shape':<24} {'loop vmac':>9} {'calls':>6} {'vmac/call':>9} {'MACs/call':>9} {'MACs/vmac':>9}  verdict")
    isa = {}
    with tempfile.TemporaryDirectory(prefix="aiecost_c4_isa_") as tn:
        for shape in shapes:
            r = isa_probe(shape, Path(tn), target)
            isa[str(shape)] = r
            tag = f"<{shape[0]},{shape[1]},{shape[2]},{shape[3]},{shape[4]}>"
            if not r["compiles"]:
                print(f"  {tag:<24} {'—':>9} {'—':>6} {'—':>9} {'—':>9} {'—':>9}  DOES NOT COMPILE: {r['error']}")
                continue
            verdict = "NATIVE" if r["native"] else f"VIRTUAL ({r['vmac_per_call']:.0f} native VMACs each)"
            print(f"  {tag:<24} {r['loop_vmac']:>9} {r['mac_calls']:>6} {r['vmac_per_call']:>9.2f} "
                  f"{r['macs_per_call']:>9} {r['macs_per_vmac']:>9.0f}  {verdict}")

    ok = [r for r in isa.values() if r.get("compiles")]
    # The VMAC ceiling is per OPERAND-TYPE PAIR, not global: a narrower operand
    # packs more MACs into the same native instruction. Reporting one number
    # across dtypes would claim int8xint8 reaches the int4 rate, which it cannot.
    print("\n  MACs per NATIVE vmac, by operand type pair:")
    families: dict[tuple[str, str], list] = {}
    for r in ok:
        families.setdefault((r["ta"], r["tb"]), []).append(r)
    for (ta, tb), rs in sorted(families.items(), key=lambda kv: max(x["macs_per_vmac"] for x in kv[1])):
        ceiling = max(x["macs_per_vmac"] for x in rs)
        native = [x for x in rs if x["native"] and x["macs_per_vmac"] == ceiling]
        best = native[0]["shape"] if native else None
        line = f"    {ta:>4} x {tb:<4}  ceiling {ceiling:>4.0f} MACs/VMAC"
        if best:
            line += f"   optimal shape <{best[0]},{best[1]},{best[2]}> (native, fills the VMAC)"
        print(line)
        if f_h:
            cores = target.compute_cores
            print(f"{'':18}=> peak = {cores} cores x {ceiling:.0f} x 2 x {f_h / 1e9:.3f} GHz = "
                  f"{cores * ceiling * 2 * f_h / 1e12:.2f} TOPS")
    print("\n  Wider shapes within a family are decomposed into several native VMACs")
    print("  (VIRTUAL) and buy nothing; narrower ones under-fill the VMAC.")

    if args.isa_only:
        return 0

    print("\nHardware check (MACs/s must agree with the ISA count):")
    print(f"  {'shape':<24} {'ns/iter':>9} {'GVMAC/s':>9} {'GMAC/s':>9} {'MACs/cyc':>9}")
    hw = {}
    for shape in shapes:
        if not isa[str(shape)].get("compiles"):
            continue
        pts = []
        for it in args.iters:
            built = build(shape, it, Path(args.cache), target)
            if not built:
                break
            t = run(*built, shape, args.reps)
            if t:
                pts.append((it, t))
        if len(pts) < 2:
            continue
        n = len(pts)
        sx = sum(x for x, _ in pts); sy = sum(y for _, y in pts)
        sxx = sum(x * x for x, _ in pts); sxy = sum(x * y for x, y in pts)
        slope = (n * sxy - sx * sy) / (n * sxx - sx * sx)
        call_s = CHAINS / slope  # mac() calls per second
        macs_s = call_s * sizes(shape)["macs"]
        vmac_s = call_s * isa[str(shape)]["vmac_per_call"]
        hw[str(shape)] = {"slope": slope, "macs_s": macs_s, "vmac_s": vmac_s,
                          "macs_per_cycle": macs_s / f_h if f_h else None}
        tag = f"<{shape[0]},{shape[1]},{shape[2]},{shape[3]},{shape[4]}>"
        print(f"  {tag:<24} {slope * 1e9:>9.3f} {vmac_s / 1e9:>9.3f} {macs_s / 1e9:>9.2f} "
              f"{(macs_s / f_h if f_h else 0):>9.1f}")

    if hw:
        best = max(hw, key=lambda k: hw[k]["macs_s"])
        print(f"\n  peak MACs/s at {best}: {hw[best]['macs_s'] / 1e9:.2f} G MACs/s/core")
        if f_h:
            print(f"  => {hw[best]['macs_per_cycle']:.0f} MACs/cycle/core")

    if args.json:
        Path(args.json).write_text(json.dumps({"isa": {k: {kk: vv for kk, vv in v.items() if kk != 'shape'} for k, v in isa.items()}, "hw": hw}, indent=2))
    if args.save and hw:
        key = calib.current_key()
        ev = [f"{k}: vmac/call={isa[k]['vmac_per_call']:.2f}, MACs/vmac={isa[k]['macs_per_vmac']:.0f}"
              for k in isa if isa[k].get("compiles")]
        ev += [f"{k}: {v['macs_s'] / 1e9:.2f} G MACs/s/core ({v['macs_per_cycle']:.0f} MACs/cycle)" for k, v in hw.items()]
        # The int8xint8 ceiling is measured, not assumed: it differs by generation
        # (AIE2 <4,8,8>=256 native, <8,8,8> virtual; AIE2P both native, ceiling 512).
        int8 = [isa[k] for k in isa if isa[k].get("compiles") and isa[k]["ta"] == "int8" and isa[k]["tb"] == "int8"]
        int8_ceiling = max((x["macs_per_vmac"] for x in int8), default=256.0)
        native_shapes = [f"<{x['shape'][0]},{x['shape'][1]},{x['shape'][2]}>" for x in int8 if x["native"]]
        virtual_shapes = [f"<{x['shape'][0]},{x['shape'][1]},{x['shape'][2]}>" for x in int8 if not x["native"]]
        native_note = f"native int8 shapes: {', '.join(native_shapes) or 'none'}"
        if virtual_shapes:
            native_note += f"; virtual (decompose, buy nothing): {', '.join(virtual_shapes)}"
        cs = {
            "cyc_per_vmac": calib.Constant(
                name="cyc_per_vmac", value=1.0, unit="cycles", bench="C4",
                method="ISA: 1 vmac per bundle in the unrolled chain loop; hardware: MACs/s tracks the ISA vmac count",
                admissible=True, evidence=ev,
                caveats=[f"{target.tile_isa}: int8 x int8 measured; other dtypes as probed"]),
            "macs_per_native_vmac_int8": calib.Constant(
                name="macs_per_native_vmac_int8", value=float(int8_ceiling), unit="MACs", bench="C4",
                method="MACs per source mac() divided by native vmacs emitted, across mmul shapes",
                admissible=True, evidence=ev,
                caveats=[
                    native_note,
                    f"peak int8 = {target.compute_cores} cores x {int8_ceiling:.0f} x 2 x f_H",
                ]),
        }
        print(f"\n  saved -> {calib.save(key, cs)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
