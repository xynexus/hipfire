#!/usr/bin/env python3
"""Hardware MAC issue-rate benchmark for AIE2P modes — the counterpart to macbench.py.

`macbench.py` counts VLIW bundles the compiler schedules. That bounds the *issue*
rate and is blind to memory stalls, which is exactly the caveat the FLM plan
attaches to its 1024-MACs/cycle figure for `mac_4x16_16x16`. This runs the same
MAC modes on real silicon.

One number cannot answer the question, because "is the wide mode real?" and "can
operands be fed fast enough to reach it?" are different questions with different
answers. Three variants separate them:

  resident  operands hoisted into registers before the loop; the loop is pure
            back-to-back MACs. This is the hardware issue rate, and it is what
            macbench.py predicts. If this misses, the reservation-table model
            is wrong. It found exactly that: mac_4x16_16x16 is 2 cyc/call, not
            1, so 512 MACs/cycle rather than the 1024 long assumed.
  seq       operands walked out of the L1 tile with pointer post-increment
            (`vldb x0, [p3], #0x40`), the addressing a real kernel uses and
            what FLM's layer kernel does (`paddb [p1], #-0x200`). This is the
            achievable rate under real operand movement.
  strided   same L1 reads but addressed by masked index arithmetic. Kept only
            as a control: the difference against `seq` is the cost of address
            computation, not of the loads themselves.

each swept over `reps` (MAC calls per delivered tile) = arithmetic intensity.

    python3 macbench_hw.py                        # default sweep
    python3 macbench_hw.py --modes w4a8           # one mode
    python3 macbench_hw.py --variants resident    # issue rate only
    python3 macbench_hw.py --reps 16384,65536     # explicit sweep

Two sanity properties any believable run must have, both encoded as checks:
rates must not exceed the mode's issue ceiling, and min_us must scale with reps
once past the overhead floor. Violating the first previously exposed a jit cache
collision that served one xclbin for every variant.

Requires PYTHONPATH=<mlir-aie>/build/python — the venv's mlir_aie wheel shadows
the build tree and will silently load a mismatched runtime.
"""

import argparse
import os
import sys

import numpy as np

# --- mode table -------------------------------------------------------------
# label -> (A type, B type, ACC type, intrinsic, MACs per call, cycles per call)
# Mirrors macbench.py so the static and hardware numbers are directly comparable.
#
# cycles-per-call is NOT 1 for every mode. mac_4x16_16x16 lowers to two vmac
# instructions (the v256int4 operand is unpacked into a y1/y2 pair and each half
# is MAC'd), so it issues once every 2 cycles. Confirmed from the scheduled loop
# body and independently on hardware at 2.073 cyc/call. Its ceiling is therefore
# 1024/2 = 512 MACs/cycle -- the same as int8, not double it.
MODES = {
    "w4a8":   ("v64int8",      "v256int4",     "v64acc32",    "mac_4x16_16x16", 4 * 16 * 16, 2),
    "w4a8u":  ("v64int8",      "v256uint4",    "v64acc32",    "mac_4x16_16x16", 4 * 16 * 16, 2),
    "w8a8":   ("v64int8",      "v64int8",      "v64acc32",    "mac_8x8_8x8",    8 * 8 * 8,   1),
    "bfp16":  ("v64bfp16ebs8", "v64bfp16ebs8", "v64accfloat", "mac_8x8_8x8T",   8 * 8 * 8,   1),
    "bf16_64": ("v64bfloat16", "v64bfloat16",  "v64accfloat", "mac_elem_64",    64,          1),
    "bf16_16": ("v32bfloat16", "v32bfloat16",  "v16accfloat", "mac_elem_16",    16,          1),
}

# Two accumulator chains cover the MAC latency without spilling. v256int4 is
# 1024 bits and v64acc32 is 2048, so a third chain risks stack traffic --
# macbench.py rejects any loop body containing a stack reference for this
# reason, and the same limit applies here.
CHAINS = 2

# The kernel reads operands out of the delivered int8 tile by reinterpret cast,
# so one tile must hold at least CHAINS distinct (a, b) operand pairs.
SRC = """
#include <aie2pintrin.h>
#include <stdint.h>

extern "C" {{

// in  : L1 tile delivered by DMA, reinterpreted as MAC operands
// out : L1 tile written back so the work cannot be dead-code eliminated
// n   : element count of the tile (supplied by the transform harness)
void {sym}(int8_t *in, int8_t *out, int32_t n) {{
  {A} *ap = ({A} *)in;
  {B} *bp = ({B} *)in;
  {ACC} *op = ({ACC} *)out;

{setup}
{loop}
{store}
}}

}}
"""

# RESIDENT: operands hoisted out of the loop -> pure issue rate.
RESIDENT_SETUP = "  {A} a0 = ap[0];\n  {B} b0 = bp[{boff}];"
RESIDENT_BODY = "    c{j} = {fn}(a0, b0, c{j});"

# STRIDED: operands re-read from L1 every iteration, addressed by masked index
# arithmetic. Honest about touching different lines, but the multiply/add/mask
# per operand is scalar work a real kernel would not do -- FLM addresses its
# operands with pointer post-increment (`paddb [p1], #-0x200`), one instruction
# folded into the load. So STRIDED is a lower bound on the streamed rate.
STRIDED_BODY = "    c{j} = {fn}(ap[(i * {chains} + {j}) & {amask}], bp[{boff} + (((i * {chains} + {j}) & {bmask}))], c{j});"

# SEQ: the realistic streamed case. Operands walk the tile with pointer
# post-increment inside an inner loop, resetting once per outer pass, which is
# the addressing mode a real kernel uses. Still one L1 load per operand per MAC
# -- only the address computation differs from STRIDED.
SEQ_TEMPLATE = """  for (int p = 0; p < {passes}; ++p) {{
    {A} *a = ap;
    {B} *b = bp + {boff};
    for (int i = 0; i < {inner}; ++i) {{
{body}
    }}
  }}"""
SEQ_BODY = "      c{j} = {fn}(*a++, *b++, c{j});"

# SPLIT: the bfp16 case. `struct v64bfp16ebs8 {v64int8 mantissa; v8int8 exponent;}`
# is 72 bytes and NOT 64-byte aligned, so a contiguous array of them misaligns
# every load after the first -- measured 43x slower before the harness learned to
# refuse it. A real KV store would keep mantissas and exponents as SEPARATE
# arrays, each naturally aligned, and assemble the pair in registers (the type is
# `return_in_regs` and lives in an EX register pair, so assembling it is free).
# This variant measures that layout.
SPLIT_TEMPLATE = """  for (int p = 0; p < {passes}; ++p) {{
    v64int8 *am = (v64int8 *)(in + {a_mant});
    v8int8  *ae = (v8int8  *)(in + {a_exp});
    v64int8 *bm = (v64int8 *)(in + {b_mant});
    v8int8  *be = (v8int8  *)(in + {b_exp});
    for (int i = 0; i < {inner}; ++i) {{
{body}
    }}
  }}"""
SPLIT_BODY = """      {{
        {A} av; av.mantissa = *am++; av.exponent = *ae++;
        {B} bv; bv.mantissa = *bm++; bv.exponent = *be++;
        c{j} = {fn}(av, bv, c{j});
      }}"""


VARIANTS = ("resident", "strided", "seq", "split")


def kernel_symbol(mode, reps, variant):
    return f"macbench_{mode}_{reps}_{variant}"


def make_source(mode, reps, variant, tile_bytes):
    """Return (source, effective_reps).

    effective_reps can differ from reps for the `seq` variant, which must land
    on a whole number of passes over the tile. The MAC accounting uses the
    effective value, so a rounded loop count cannot inflate the reported rate.
    """
    if variant not in VARIANTS:
        raise SystemExit(f"unknown variant {variant}")
    A, B, ACC, fn, _, _ = MODES[mode]
    a_bytes = type_bytes(A)
    b_bytes = type_bytes(B)

    # B operands live in the second half of the tile so A and B reads do not
    # alias, which would let the compiler collapse them into one load.
    half = tile_bytes // 2
    boff = half // b_bytes
    n_a = pow2_floor(half // a_bytes)
    n_b = pow2_floor(half // b_bytes)
    if n_a < CHAINS or n_b < CHAINS:
        raise SystemExit(f"{mode}: tile of {tile_bytes} B too small for {CHAINS} operand pairs")

    decl = "\n".join(f"  {ACC} c{j} = op[{j}];" for j in range(CHAINS))
    store = "\n".join(f"  op[{j}] = c{j};" for j in range(CHAINS))
    eff_reps = reps

    if variant == "resident":
        setup = RESIDENT_SETUP.format(A=A, B=B, boff=boff) + "\n" + decl
        body = "\n".join(RESIDENT_BODY.format(j=j, fn=fn) for j in range(CHAINS))
        loop = f"  for (int i = 0; i < {reps}; ++i) {{\n{body}\n  }}"
    elif variant == "strided":
        setup = decl
        body = "\n".join(
            STRIDED_BODY.format(j=j, fn=fn, chains=CHAINS, amask=n_a - 1,
                                bmask=n_b - 1, boff=boff)
            for j in range(CHAINS))
        loop = f"  for (int i = 0; i < {reps}; ++i) {{\n{body}\n  }}"
    elif variant == "split":
        # Mantissa and exponent in separate, individually aligned arrays. Only
        # meaningful for the block-float types, whose packed struct is what makes
        # a contiguous walk misalign in the first place.
        if "bfp16" not in A or "bfp16" not in B:
            raise SystemExit(f"{mode}: split is only for bfp16 operand types "
                             f"(got A={A}, B={B}); use seq")
        # Quarter the tile: A mantissas, A exponents, B mantissas, B exponents.
        q = tile_bytes // 4
        n = min(q // 64, q // 8)          # entries that fit in both sub-arrays
        n = pow2_floor(n)
        inner = n // CHAINS
        if inner < 1:
            raise SystemExit(f"{mode}: tile too small for a split pass")
        passes = max(1, reps // inner)
        eff_reps = passes * inner
        setup = decl
        body = "\n".join(SPLIT_BODY.format(j=j, fn=fn, A=A, B=B)
                         for j in range(CHAINS))
        loop = SPLIT_TEMPLATE.format(a_mant=0, a_exp=q, b_mant=2 * q,
                                     b_exp=3 * q, passes=passes, inner=inner,
                                     body=body)
    else:  # seq
        # A post-increment walk steps by sizeof(operand). AIE vector loads want
        # 64-byte alignment, so an operand type whose size is not a multiple of
        # 64 goes misaligned after the first step and the compiler falls back to
        # something far slower -- v64bfp16ebs8 is 72 bytes (9 bits x 64) and
        # measured 161 ms against resident's 1.8 ms, a 43x "result" that is
        # purely this artifact. Refuse rather than report it.
        for label, ty, sz in (("A", A, a_bytes), ("B", B, b_bytes)):
            if sz % 64:
                raise SystemExit(
                    f"{mode}: seq needs 64B-aligned operands; {label} type {ty} "
                    f"is {sz} B. Needs a layout that keeps the walk aligned "
                    f"(for bfp16, mantissa and exponent as separate arrays).")
        # Each inner iteration consumes CHAINS a-operands and CHAINS b-operands
        # via post-increment, so a pass covers min(n_a, n_b) // CHAINS of them
        # before the pointers must reset.
        inner = min(n_a, n_b) // CHAINS
        if inner < 1:
            raise SystemExit(f"{mode}: tile too small for a sequential pass")
        passes = max(1, reps // inner)
        eff_reps = passes * inner
        setup = decl
        body = "\n".join(SEQ_BODY.format(j=j, fn=fn) for j in range(CHAINS))
        loop = SEQ_TEMPLATE.format(A=A, B=B, boff=boff, passes=passes,
                                   inner=inner, body=body)

    src = SRC.format(A=A, B=B, ACC=ACC, setup=setup, loop=loop,
                     store=store, sym=kernel_symbol(mode, reps, variant))
    return src, eff_reps


def type_bytes(t):
    """Byte width of an AIE vector type name like v64int8 / v256int4 / v64acc32."""
    import re
    # Longest alternatives first: `acc` would otherwise match the "acc" of
    # "accfloat" and leave "float" for the width group, which then fails
    # int("") -- that was the FAIL on every bfp16/bf16 row.
    m = re.match(r"v(\d+)(accfloat|acc|bfp16ebs|bfloat|uint|int)(\d*)$", t)
    if not m:
        raise SystemExit(f"cannot size type {t}")
    lanes = int(m.group(1))
    kind, width = m.group(2), m.group(3)
    if kind == "bfp16ebs":
        # {v64int8 mantissa; v8int8 exponent} at ebs8 -> 9 bits/element
        return lanes * 9 // 8
    if kind == "accfloat":
        return lanes * 4
    if kind == "acc":
        return lanes * int(width) // 8
    if kind == "bfloat":
        return lanes * 2
    return lanes * int(width) // 8


def pow2_floor(x):
    p = 1
    while p * 2 <= x:
        p *= 2
    return p


def build_and_run(mode, reps, variant, tile_bytes, tensor_bytes, warmup, iters, clock_ghz):
    import aie.iron as iron
    from aie.iron import CompileTime, In, Out
    from aie.iron.kernel import ExternalFunction
    from aie.iron.algorithms import transform
    from aie.utils.benchmark import run_iters

    tile_elems = tile_bytes
    tensor_elems = tensor_bytes

    peano_inc = os.path.join(
        os.environ.get("PEANO_INSTALL_DIR", ""), "lib", "clang", "21", "include")
    flags = ["-I", peano_inc] if os.path.isdir(peano_inc) else []

    # mode / reps / variant MUST be CompileTime parameters of the jit'd design,
    # not closure captures. The jit cache key is built from the generator
    # function plus its runtime args and compile kwargs
    # (_create_function_cache_key in utils/callabledesign.py) and does NOT see
    # closure state. Capturing the generated source instead of declaring it made
    # every variant collide on one cached xclbin, which showed up as timings
    # flat in `reps` and an impossible 1416%-of-issue-rate result.
    @iron.jit(use_cache=True)
    def design(A: In, C: Out, *, n: CompileTime[int], t: CompileTime[int],
               mode: CompileTime[str], reps: CompileTime[int],
               variant: CompileTime[str]):
        tensor_ty = np.ndarray[(n,), np.dtype[np.int8]]
        tile_ty = np.ndarray[(t,), np.dtype[np.int8]]
        src, _ = make_source(mode, reps, variant, t)
        fn = ExternalFunction(
            kernel_symbol(mode, reps, variant),
            source_string=src,
            arg_types=[tile_ty, tile_ty, np.int32],
            compile_flags=flags,
        )
        return transform(fn, tensor_ty, tile_size=t)

    a = iron.tensor(np.zeros(tensor_elems, dtype=np.int8), dtype=np.int8, device="npu")
    c = iron.zeros(tensor_elems, dtype=np.int8, device="npu")

    bench = run_iters(design, a, c, n=tensor_elems, t=tile_elems,
                      mode=mode, reps=reps, variant=variant,
                      warmup=warmup, iters=iters)

    # Recompute effective reps the same way the kernel did, so a `seq` loop
    # rounded to whole passes is accounted at the count actually executed.
    _, eff_reps = make_source(mode, reps, variant, tile_elems)

    us, avg_us = bench_us(bench)
    n_tiles = tensor_elems // tile_elems
    macs_per_instr = MODES[mode][4]
    total_macs = n_tiles * eff_reps * CHAINS * macs_per_instr
    seconds = us * 1e-6
    macs_per_cycle = total_macs / (seconds * clock_ghz * 1e9)
    # Bytes the DMA must deliver per tile, and the resulting intensity.
    intensity = (eff_reps * CHAINS * macs_per_instr) / tile_bytes
    return us, avg_us, macs_per_cycle, intensity


def bench_us(bench):
    """NPU-side (min, avg) microseconds from run_iters' BenchmarkResult.

    min drives the reported rate: this is a ceiling measurement, so the
    least-perturbed run is the honest one. avg is printed alongside so a noisy
    measurement is visible rather than hidden.
    """
    npu = getattr(bench, "npu", None)
    if npu is None:
        raise SystemExit(f"cannot find NPU time in benchmark result: {bench!r}")
    return float(npu.min_us), float(npu.avg_us)


def main():
    p = argparse.ArgumentParser(description="AIE2P hardware MAC issue-rate benchmark")
    p.add_argument("--modes", default="w4a8,w8a8,bfp16,bf16_16",
                   help="comma-separated mode keys (default: the ones the FLM plan turns on)")
    p.add_argument("--reps", default="16,64,256,1024",
                   help="comma-separated MAC instructions per delivered tile")
    p.add_argument("--variants", default="resident,strided,seq",
                   help="comma-separated: resident (issue rate), strided "
                        "(masked-index L1 reads), seq (pointer post-increment)")
    p.add_argument("--tile-bytes", type=int, default=4096)
    p.add_argument("--tensor-bytes", type=int, default=4096 * 16)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--clock-ghz", type=float, default=1.8,
                   help="AIE compute clock; measured 1.8 GHz via npu_info h_mhz")
    opts = p.parse_args()

    modes = [m.strip() for m in opts.modes.split(",") if m.strip()]
    reps_list = [int(r) for r in opts.reps.split(",") if r.strip()]
    variants = [v.strip() for v in opts.variants.split(",") if v.strip()]
    for m in modes:
        if m not in MODES:
            raise SystemExit(f"unknown mode {m}; known: {', '.join(MODES)}")
    for v in variants:
        if v not in VARIANTS:
            raise SystemExit(f"unknown variant {v}; known: {', '.join(VARIANTS)}")

    bad = []
    print(f"{'mode':9s} {'variant':9s} {'reps':>6s} {'MAC/byte':>9s} "
          f"{'min_us':>8s} {'avg_us':>8s} {'MAC/cyc':>8s} {'vs static':>10s}")
    print("-" * 76)
    for mode in modes:
        # Issue-rate ceiling: MACs per intrinsic call divided by cycles per
        # call. mac_4x16_16x16 lowers to two vmacs, so it costs 2 cycles --
        # see docs/npu/flm-refe-log.md, 2026-07-31.
        static = MODES[mode][4] / MODES[mode][5]
        for variant in variants:
            for reps in reps_list:
                label = variant.upper()
                try:
                    us, avg_us, mpc, ai = build_and_run(
                        mode, reps, variant, opts.tile_bytes, opts.tensor_bytes,
                        opts.warmup, opts.iters, opts.clock_ghz)
                except Exception as e:
                    print(f"{mode:9s} {label:9s} {reps:6d} {'':>9s} "
                          f"{'FAIL':>8s}  {str(e).splitlines()[0][:44]}")
                    continue
                flag = ""
                # A rate above the issue ceiling is not a fast result, it is a
                # broken measurement -- the machine has one MAC slot per VLIW
                # bundle, so `static` MACs/cycle is a hard upper bound. This
                # fired on the first working version of this script (jit cache
                # collision serving one xclbin for every variant) and is the
                # reason the guard exists.
                if mpc > static * 1.02:
                    flag = "  <-- IMPOSSIBLE (>issue rate); measurement is wrong"
                    bad.append((mode, label, reps, mpc))
                print(f"{mode:9s} {label:9s} {reps:6d} {ai:9.1f} "
                      f"{us:8.1f} {avg_us:8.1f} {mpc:8.1f} "
                      f"{mpc / static * 100:9.1f}%{flag}")

    if bad:
        print("\nREJECTED: results above the hardware issue rate are a broken")
        print("measurement, not a fast one. Check that the jit actually rebuilt")
        print("per variant (mode/reps/variant must be CompileTime params, not")
        print("closure captures) and that min_us scales with reps.")
        sys.exit(1)


if __name__ == "__main__":
    main()
