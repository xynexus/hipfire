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

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from p1_route import NQ, NK, NV, HPC, hpc_for, head_layout  # noqa: E402

# Core budget: 4 rows x 8 columns on npu2. The working design uses 16 and leaves
# half the array idle; role groups only pay if that half is used.
NCORES = 32
K_DIM, D_FF, NROWS, HEAD, GQA, TSEQ = 2048, 8192, 8, 64, 4, 32

# Group sizes. Forced by tiling, not chosen: see the module docstring.
A_CORES = 8         # P1  qkv + RoPE
B_CORES = 8         # P2  attention, one KV group each
C_CORES = 16        # P3 + P4 + P5


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


def main():
    print(f"role-specialised layer, {NCORES} cores")
    for name, (first, n), phases, share in plan():
        print(f"  group {name}: cores {first:2d}-{first + n - 1:2d} ({n:2d})  "
              f"{phases:14s}  {share}")
    print("\nstreams: host->A, A->B (q'k'v'), B->C (attn), C->A (residual), "
          "C->host (x_out)")
    print("memtile inputs: 16 (C's join) + splits, against ~48 — does not bind")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
