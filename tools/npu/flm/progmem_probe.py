#!/usr/bin/env python3
"""Does one core hold ALL five phases' kernels? Measure, do not add ELF sizes.

Program memory is 16 KB per core and the fused layer wants P1 (qkv+RoPE), P2
(attention), P3 (o_proj+residual), P4 (gate/up/SwiGLU) and P5 (down) on the same
core. Summing the phase images says 138% and overflowing — but that sum
double-counts, because `flm_q4_1_tile` is `linkonce_odr` and is pulled in by
kernels in *every* phase. Combined, it is emitted once.

So the sum cannot answer the question and only a build can. This design declares
and calls every entry point the layer needs. It computes nothing meaningful —
the objects are the right *shapes*, not the right data — because the only output
that matters is the core ELF's `.text` size.

    python3 progmem_probe.py            # build and report
    python3 progmem_probe.py --phases 3 # P1..P3 only, to see the growth curve

Needs PYTHONPATH=<mlir-aie>/build/python plus the Peano/XRT env.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

import aie.iron as iron  # noqa: E402
from aie.iron import In, ObjectFifo, Out, Program, Runtime, TaskGroup, Worker  # noqa: E402
from aie.iron.device import AnyShimTile  # noqa: E402
from aie.iron.kernel import ExternalFunction  # noqa: E402
from ml_dtypes import bfloat16  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import q4nx  # noqa: E402

KDIR = Path(__file__).resolve().parents[3] / "kernels/npu"
K_DIM, NROWS, HEAD, TSEQ, GQA = 2048, 16, 64, 32, 4
OPERAND = 20544
WT = q4nx.tile_bytes(K_DIM, NROWS)

# every -D the phase kernels read, unioned; a wrong value only makes the kernel
# compute nonsense, which is fine here, but a MISSING one fails the build
FLAGS = [f"-DDIM_K={K_DIM}", f"-DDIM_NROWS={NROWS}", f"-DDIM_HEAD={HEAD}",
         f"-DDIM_ACT={K_DIM}", f"-DDIM_QHEADS=32", f"-DDIM_QKHEADS=40",
         f"-DDIM_GQA={GQA}", f"-DDIM_TSEQ={TSEQ}", f"-DDIM_KVPER=2",
         f"-DDIM_KVOBJ={OPERAND}", f"-DDIM_QSTRIDE={2 * HEAD}",
         f"-DDIM_NPADOFF={32 * 2 * HEAD}", "-DQOFF_FROM_KV=1",
         f"-DDIM_KVSTRIDE={OPERAND // 2}", f"-DDIM_NCHUNK=4"]

# (name, source, arity) per phase — arity is how many objects it takes
PHASES = {
    1: [("flm_norm_prepare", "flm_norm_prepare.cc", 1),
        ("flm_gemv_qkv", "flm_gemv_qkv.cc", 3),
        ("flm_p1_emit", "flm_p1_emit.cc", 1),
        ("flm_kv_emit", "flm_kv_emit.cc", 2)],
    2: [("flm_attn_begin", "flm_attn_begin.cc", 1),
        ("flm_attn_tile", "flm_attn_decode.cc", 2),
        ("flm_attn_finish", "flm_attn_finish.cc", 2)],
    3: [("flm_gemv_q4_1_residual", "flm_gemv_residual.cc", 3)],
    4: [("flm_gemv_gate", "flm_gemv_gate.cc", 2),
        ("flm_gemv_up_swiglu", "flm_gemv_up_swiglu.cc", 3)],
    6: [("flm_h_emit", "flm_h_emit.cc", 2)],
    7: [("flm_asum_prepare", "flm_asum_prepare.cc", 1)],
    5: [("flm_gemv_acc", "flm_gemv_acc.cc", 2),
        ("flm_gemv_flush", "flm_gemv_flush.cc", 3)],
}


def build(nphases, sel=None):
    # One generous object type. IRON only checks that the memref shape matches
    # the fifo the object came from, so a single big uint8 object satisfies
    # every kernel's declared argument.
    big_ty = np.ndarray[(OPERAND,), np.dtype[np.uint8]]
    tag = "".join(str(x) for x in (sel if sel else range(1, nphases + 1)))
    picked = sel if sel else list(range(1, nphases + 1))
    want = [k for p in picked for k in PHASES[p]]
    missing = [s for _, s, _ in want if not (KDIR / s).exists()]
    if missing:
        raise SystemExit(f"missing kernel sources: {missing}")

    decls = "\n    ".join(
        f'k{i} = ExternalFunction("{n}", source_file="{KDIR / s}", '
        f'arg_types=[big_ty] * {a}, compile_flags=FLAGS)'
        for i, (n, s, a) in enumerate(want))
    calls = "\n        ".join(f"k{i}(*[e] * {a})"
                              for i, (_, _, a) in enumerate(want))
    src = f'''
def _design(a: In, o: Out):
    {decls}
    f_a = ObjectFifo(big_ty, depth=1, name="pm_in_{tag}")
    f_o = ObjectFifo(big_ty, depth=1, name="pm_out_{tag}")

    def core(ac, op, {", ".join(f"k{i}" for i in range(len(want)))}):
        e = ac.acquire(1)
        eo = op.acquire(1)
        {calls}
        op.release(1)
        ac.release(1)

    w = Worker(core, fn_args=[f_a.cons(), f_o.prod(), {", ".join(f"k{i}" for i in range(len(want)))}],
               stack_size=8192)

    def seq(ab, ob, ah, oh):
        tg = TaskGroup()
        ah.fill(ab, group=tg)
        oh.drain(ob, wait=True, group=tg)
        tg.finish()

    rt = Runtime(seq, [big_ty, big_ty, f_a.prod(tile=AnyShimTile),
                       f_o.cons(tile=AnyShimTile)])
    return Program(iron.get_current_device(), rt, workers=[w]).resolve_program()
'''
    ns = dict(np=np, iron=iron, In=In, Out=Out, ObjectFifo=ObjectFifo,
              Program=Program, Runtime=Runtime, TaskGroup=TaskGroup,
              Worker=Worker, AnyShimTile=AnyShimTile,
              ExternalFunction=ExternalFunction, FLAGS=FLAGS,
              big_ty=big_ty, __name__=f"pm{tag}")
    exec(src, ns)
    srcs = [str(KDIR / s) for _, s, _ in want]
    return iron.jit(ns["_design"], source_files=srcs, full_elf=True), len(want)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--phases", type=int, default=5, choices=[1, 2, 3, 4, 5, 6, 7])
    p.add_argument("--only", help="comma-separated phase list, e.g. 1,3,4,5 for "
                                  "the cores that run no attention")
    o = p.parse_args()
    sel = [int(x) for x in o.only.split(",")] if o.only else None
    design, nk = build(o.phases, sel)
    a = iron.zeros(OPERAND, dtype=np.uint8, device="npu")
    b = iron.zeros(OPERAND, dtype=np.uint8, device="npu")
    try:
        design(a, b)
    except Exception as e:                      # running it is not the point
        print(f"(run failed, expected: {type(e).__name__}: {e})",
              file=sys.stderr)
    caches = sorted(Path.home().glob(".npu/cache/*"),
                    key=lambda d: d.stat().st_mtime, reverse=True)
    for d in caches[:4]:
        elf = next(d.glob("elfs_main_core_*/*.elf"), None)
        if elf and (d / "aie.mlir").exists() and \
                f"pm_in_{''.join(str(x) for x in (sel or range(1, o.phases + 1)))}" in (d / "aie.mlir").read_text():
            out = subprocess.run(["size", "-A", str(elf)], capture_output=True,
                                 text=True).stdout
            text = next((int(l.split()[1]) for l in out.splitlines()
                         if l.startswith(".text")), None)
            print(f"phases {sel or list(range(1, o.phases+1))}: {nk} kernels  "
                  f".text = {text} B = {100 * text / 16384:.0f}% of 16 KB  "
                  f"{'FITS' if text <= 16384 else 'OVERFLOWS'}")
            return 0 if text <= 16384 else 1
    print("could not locate this design's core ELF", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
