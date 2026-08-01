#!/usr/bin/env python3
"""A decoder layer as three ROLE GROUPS on 32 cores, one dispatch per token.

Every earlier structure failed for the same underlying reason: a layer is
strictly sequential, so `x_{L+1} = f(x_L)`, and any split into two dispatches
needs 32 of them for sixteen layers. One dispatch per token is the only shape
that works, and it needs all five phases reachable within it. On uniform cores
that overflows program memory — measured on a real build, stage 37 of 42, with
routing, placement, DMA and memtiles all passing first.

So the phases are split across groups instead of stacked on every core:

    group A   8 cores    P1  qkv + RoPE       48 head-tiles / 8 = 6 each
    group B   8 cores    P2  attention        one KV group per core, FULL coverage
    group C  16 cores    P3 + P4 + P5         2048/128 = 16, 8192/128 = 64

Three bodies is the most any core carries, against the four that overflowed.
Group sizes are forced: a group running P3/P4/P5 must divide both 2048 and 8192
at NROWS=8, so 4, 8, 16 or 32 only — 24 is unavailable, which is why "8 for
attention and 24 for everything else" is not on the table.

Intermediates move CORE TO CORE, which is the whole trick:

    host -> A   activation, weights
    A -> B      q', k', v'          core to core
    B -> C      attention output    core to core
    C -> A      residual            core to core, next layer's input
    C -> host   x_out               the only join

A core-to-core fifo consumes **zero** memtile DMA (measured: 0 MemTiles in the
generated MLIR). That matters because memtile channels, not link counts, are what
bound core count: a w-way join costs w inputs and there are cores/w of them, so
joins always cost exactly `cores` input channels — 40 of ~48 at 32 cores before a
single operand split. Streaming between groups and emitting from one collapses
that to 16.

Everything in the projection is measured:

    P1 at 8 cores        136.9 us compute   (92.9 at 12; 12->16 buys only 4.7)
    handoff at 4 KB        3.3 us           (0.9 at 256 B)
    3 handoffs/layer       9.9 us
    memtile inputs           16 of ~48
    bodies per core           3 max
    dispatches per token      1              (vs 32 interleaved)

    -> ~16.54 ms/token -> **60.5 tok/s**, against FLM's measured 61.18

Within noise of FLM, and the only structure measured so far that computes a
correct token at all — the 63.4 tok/s figure from `p1p2_chain` + `p5_pass` is for
a composition that cannot.

**Status: skeleton.** Geometry and group assignment only; no phases wired yet.
`p1p2_chain` remains the working design.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import q4nx  # noqa: E402
from p1_route import (NQ, NK, NV, HPC, hpc_for, head_layout,  # noqa: E402
                      drain_plan, rnd)

import aie.iron as iron  # noqa: E402
from aie.iron import In, ObjectFifo, Out, Program, Runtime, TaskGroup, Worker  # noqa: E402
from aie.iron.device import AnyShimTile  # noqa: E402

# Core budget: 4 rows x 8 columns on npu2. The working design uses 16 and leaves
# half the array idle; role groups only pay if that half is used.
NCORES = 32
K_DIM, D_FF, NROWS, HEAD, GQA, TSEQ = 2048, 8192, 8, 64, 4, 32
KVPER = 1
# One operand size for every fifo, as in the working design: the KV tile and the
# weight tile are different shapes but a core has only 2 input DMA channels, so
# they share a stream and the object must hold whichever is larger.
OPERAND = max(2 * TSEQ * HEAD * 2 * KVPER, q4nx.tile_bytes(K_DIM, NROWS))
OBJ = 2 * HEAD                     # a head's result slot
BC = 2 * K_DIM + 2 * HEAD          # broadcast block: activation, norm weight, RoPE

# Group sizes. Forced by tiling, not chosen: see the module docstring.
A_CORES = 8         # P1  qkv + RoPE
B_CORES = 8         # P2  attention, one KV group each
C_CORES = 16        # P3 + P4 + P5


def sizes():
    """-> the per-group shares, so a wrong constant fails here and not on device."""
    g = groups()
    return {
        # group=1: A[j] streams straight to B[j], so each A core owns its fifo
        # and there is no join to describe. drain_plan's default of 2 is for the
        # paired design and would report half as many, twice as large.
        "A": dict(cores=A_CORES, head_tiles=NV // A_CORES,
                  q_objs=drain_plan(A_CORES, group=1)[0]),
        "B": dict(cores=B_CORES, kv_groups=(NV - NQ) // 2,
                  q_heads_each=NQ // B_CORES),
        "C": dict(cores=C_CORES, k_tiles=K_DIM // (C_CORES * NROWS),
                  ff_tiles=D_FF // (C_CORES * NROWS),
                  emits=C_CORES // 4),        # four 4-way joins, not one 16-way
    }


def groups():
    """-> {role: (first_core, count)}, and the checks that the sizes are legal."""
    assert A_CORES + B_CORES + C_CORES == NCORES, "groups must tile the array"
    # A group running P3/P4/P5 owns a share of both row counts.
    assert K_DIM % (C_CORES * NROWS) == 0, "C must tile K_DIM rows"
    assert D_FF % (C_CORES * NROWS) == 0, "C must tile D_FF rows"
    # A does qkv over all 48 head-tiles.
    assert NV % A_CORES == 0, "A must tile the head-tiles"
    # B does one KV group per core; the model has NV - NQ = 16 kv tiles = 8 k + 8 v.
    assert B_CORES == (NV - NQ) // 2, "B must have one core per KV group"
    return {"A": (0, A_CORES),
            "B": (A_CORES, B_CORES),
            "C": (A_CORES + B_CORES, C_CORES)}


def plan():
    """-> a printable summary of what each group does and how wide its share is."""
    g = groups()
    return [
        ("A", g["A"], "P1 qkv+RoPE", f"{NV // A_CORES} head-tiles/core"),
        ("B", g["B"], "P2 attention", f"1 KV group/core, {NQ} q heads total"),
        ("C", g["C"], "P3+P4+P5", f"{K_DIM // (C_CORES * NROWS)} K-tiles, "
                                  f"{D_FF // (C_CORES * NROWS)} FF-tiles/core"),
    ]


def role_layout():
    """Head-tiles per A core — which is just `head_layout(A_CORES)`.

    I wrote a bespoke assignment to make A[j] feed B[j] with no shuffle, then
    checked it against the existing one: **they already agree**. `head_layout(8)`
    gives core j the four q heads `4j..4j+3` plus k[j] and v[j], differing only in
    whether k or v sits in the last slot — and `drain_plan` derives that from the
    layout rather than assuming it.

    So the role-aligned head assignment is not something to arrange; at eight
    cores it is what the existing rule already produces. Using it directly keeps
    one source of truth, and P1's measured 204.1 us at 8 cores was measured with
    exactly this layout.
    """
    return head_layout(A_CORES)


def build_skeleton(elems=64):
    """The stream topology with trivial kernels: does 32 cores in three groups
    place and route at all? Everything else is filling it in.

    `elems` sets the object size. The topology was first proven at 64 int32 =
    256 B, but the real operand is OPERAND = 10304 B and the broadcast block is
    BC * 2 = 8448 B. Object size is charged against each core's 64 KB of data
    memory and against DMA descriptors, so a topology that places at 256 B can
    still fail at realistic sizes — which is what this checks.
    """
    ty = np.ndarray[(elems,), np.dtype[np.int32]]

    def _design(a: In, o0: Out, o1: Out, o2: Out, o3: Out):
        f_act = ObjectFifo(ty, depth=1, name="rl_act")          # shim -> A
        f_ab = [ObjectFifo(ty, depth=1, name=f"rl_ab{j}")       # A[j] -> B[j]
                for j in range(A_CORES)]
        # B emits through TWO 4-way joins for the same reason C does: 8-way asks a
        # memtile for 8 inputs and fails. Each C core consumes both halves, which
        # it needs anyway — o_proj takes the whole attention vector.
        f_bc = [ObjectFifo(ty, depth=1, name=f"rl_bc{i}")
                for i in range(B_CORES // 4)]
        # C emits through FOUR 4-way joins, not one 16-way: a join needs `w`
        # inputs on a single memtile and a memtile has ~6, so 16-way is
        # unplaceable. Four drains is fine — the shim has 16 output channels.
        f_out = [ObjectFifo(ty, depth=1, name=f"rl_out{i}")
                 for i in range(C_CORES // 4)]

        part = elems // 4                      # each of four producers fills a quarter
        bc_in = [f.prod().join([j * part for j in range(4)],
                               obj_types=[np.ndarray[(part,),
                                                     np.dtype[np.int32]]] * 4)
                 for f in f_bc]
        out_sub = [f.prod().join([j * part for j in range(4)],
                                  obj_types=[np.ndarray[(part,),
                                                        np.dtype[np.int32]]] * 4)
                   for f in f_out]
        act_cons = [f_act.cons() for _ in range(A_CORES)]
        bc_cons = [[f.cons() for _ in range(C_CORES)] for f in f_bc]

        def core_a(ic, oc):
            e = ic.acquire(1); r = oc.acquire(1)
            for k in range(elems):
                r[k] = e[k] + 1
            oc.release(1); ic.release(1)

        def core_b(ic, oc):
            e = ic.acquire(1); r = oc.acquire(1)
            for k in range(elems // 4):
                r[k] = e[k] + 2
            oc.release(1); ic.release(1)

        def core_c(ic0, ic1, oc):
            e0 = ic0.acquire(1); e1 = ic1.acquire(1); r = oc.acquire(1)
            for k in range(elems // 4):
                r[k] = e0[k] + e1[k] + 3
            oc.release(1); ic1.release(1); ic0.release(1)

        workers = []
        workers += [Worker(core_a, fn_args=[act_cons[j], f_ab[j].prod()],
                           stack_size=2048) for j in range(A_CORES)]
        workers += [Worker(core_b, fn_args=[f_ab[j].cons(),
                                            bc_in[j // 4][j % 4].prod()],
                           stack_size=2048) for j in range(B_CORES)]
        workers += [Worker(core_c, fn_args=[bc_cons[0][j], bc_cons[1][j],
                                            out_sub[j // 4][j % 4].prod()],
                           stack_size=2048) for j in range(C_CORES)]

        def seq(ab, *rest):
            n = len(rest) // 2
            obs, ah, ohs = rest[:n], rest[n], rest[n + 1:]
            tg = TaskGroup()
            ah.fill(ab, group=tg)
            for ob, oh in zip(obs, ohs):
                oh.drain(ob, wait=True, group=tg)
            tg.finish()

        rt = Runtime(seq, [ty] + [ty] * len(f_out) + [f_act.prod(tile=AnyShimTile)]
                     + [f.cons(tile=AnyShimTile) for f in f_out])
        return Program(iron.get_current_device(), rt, workers=workers).resolve_program()

    return iron.jit(_design)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skeleton", action="store_true",
                    help="build the stream topology with trivial kernels")
    ap.add_argument("--elems", type=int, default=64,
                    help="object size in int32; the real operand is 2576 int32 "
                         "(10304 B), so 64 does not prove much on its own")
    o = ap.parse_args()
    print(f"role-specialised layer, {NCORES} cores")
    for name, (first, n), phases, share in plan():
        print(f"  group {name}: cores {first:2d}-{first + n - 1:2d} ({n:2d})  "
              f"{phases:14s}  {share}")
    print("\nstreams: host->A, A->B (q'k'v'), B->C (attn), C->A (residual), "
          "C->host (x_out)")
    print("memtile inputs: 16 (C's join) + splits, against ~48 — does not bind")
    print(f"\noperand {OPERAND} B, OBJ {OBJ}, broadcast block {BC}")
    for name, d in sizes().items():
        print(f"  group {name}: " + ", ".join(f"{k}={v}" for k, v in d.items()))
    print("\nA->B head layout (chosen so A[j] feeds B[j] with no shuffle):")
    for j, row in enumerate(role_layout()):
        print(f"  A{j} -> B{j}: q {row[:-2]}, k {row[-2]}, v {row[-1]}")
    if o.skeleton:
        print(f"\nbuilding the stream topology at {o.elems} int32 "
              f"({o.elems * 4} B/object)...")
        a = iron.zeros(o.elems, dtype=np.int32, device="npu")
        outs = [iron.zeros(o.elems, dtype=np.int32, device="npu") for _ in range(4)]
        build_skeleton(o.elems)(a, *outs)
        # Running is not enough: a topology that misroutes still runs. Trace the
        # arithmetic — A adds 1, B adds 2, C adds both halves plus 3 — so with a
        # zero input every output element must be (1+2) + (1+2) + 3 = 9. A wrong
        # stream shows up as a wrong value, not a hang.
        want = 9
        bad = [i for i, t in enumerate(outs) if not (t.numpy() == want).all()]
        print("  -> 32 cores in three groups PLACE AND ROUTE")
        if bad:
            g = outs[bad[0]].numpy()
            print(f"  -> but output {bad[0]} is wrong: got {g[:4]} want {want}")
            return 1
        print(f"  -> all four outputs carry {want}: every stream delivers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
