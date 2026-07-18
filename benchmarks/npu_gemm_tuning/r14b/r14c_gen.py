#!/usr/bin/env python3
# R14C — 8-weight-stream GEMM with core-RESIDENT activations (npu1/aie2).
#
# Why: R14 spends one of each column shim's two MM2S channels on the A-stripe, so only
# 4 DDR streams carry weights. At DFlash shapes the weight path is pinned at 9.3-9.8
# GB/s regardless of MACs, A bytes, or fifo depth. This variant frees all 8 MM2S
# channels for weights by taking the activation off the shim entirely.
#
# Two earlier formulations hit hard walls on npu1 and are recorded in the header of
# r14b_gen.py / the run log:
#   1. even/odd block split with a second (W,A) fifo pair per core ->
#      "'aie.tile' op number of input DMA channel exceeded!"  (AIE2 core tile has only
#      2 S2MM channels, and R14 already uses both).
#   2. fusing [W|A] into one shim object and splitting it in the memtile with a link
#      that is both a JOIN (2 shim channels -> 1 memtile buffer) and a DISTRIBUTE
#      (buffer -> W broadcast + A broadcast) ->
#      "ObjectFifoLinkOp does not support 'join' and 'distribute' at the same time".
#
# What is left, and what this file does: keep exactly ONE inbound objectfifo per core
# (the W broadcast), make it a JOIN of the column shim's TWO MM2S channels, and hold A
# in a core-resident aie.buffer with an initial value. Per-block bytes per channel drop
# from 16384 (W channel) / 4096 (A channel) to a balanced 8192 on all 8 channels.
#
# Channel budget: shim 2 MM2S / 1 S2MM; memtile 2 S2MM (wsh) + 4 S2MM (C) = 6 in,
# 2 MM2S out; core 1 S2MM in, 1 MM2S out.
#
# NOTE (scope): the resident A is a compile-time constant, so this measures the weight
# path with the activation channel removed -- it is a bandwidth probe, not a drop-in
# production dataflow. A production version needs A refreshed per dispatch, which on
# npu1 costs either a shim channel back (returning to 4-6 weight streams) or a
# core-side C relay to free memtile S2MM channels. AVAL is chosen != 1 so the
# correctness gate proves the resident buffer is really being read.
#
# Usage: r14c_gen.py LM LN KT N_BLK [WBYTES] [DEVICE] [CBASE] [DEPTH] [AVAL] > r14c.mlir
import sys
LM  = int(sys.argv[1]) if len(sys.argv) > 1 else 1
LN  = int(sys.argv[2]) if len(sys.argv) > 2 else 4
KT  = int(sys.argv[3]) if len(sys.argv) > 3 else 64
NBLK = int(sys.argv[4]) if len(sys.argv) > 4 else 128
WBYTES = int(sys.argv[5]) if len(sys.argv) > 5 else 64
DEVICE = sys.argv[6] if len(sys.argv) > 6 else "npu1"
CBASE = int(sys.argv[7]) if len(sys.argv) > 7 else 32
DEPTH = int(sys.argv[8]) if len(sys.argv) > 8 else 2
AVAL = int(sys.argv[9]) if len(sys.argv) > 9 else 3
NCH = int(sys.argv[10]) if len(sys.argv) > 10 else 2   # shim MM2S channels per column
ROWS = int(sys.argv[11]) if len(sys.argv) > 11 else 4  # active core rows per column (broadcast fanout probe)
AB = LM * KT * 64        # A-stripe bytes, now core-resident
WB = LN * KT * WBYTES    # W-stripe bytes (shared by a physical column)
CB = LM * LN * CBASE     # per-core C i32
CJ = (int(sys.argv[11]) if len(sys.argv) > 11 else 4) * CB              # joined C per column
INF = 9223372036854775807
G = range(4)
R = range(int(sys.argv[11]) if len(sys.argv) > 11 else 4)

S0 = ((WB // 2) // 64) * 64      # balanced cut of the W stripe across the 2 MM2S
S1 = WB - S0
if S0 <= 0 or S1 <= 0:
    raise ValueError("W stripe too small to split across two channels")

# npu1 (aie2) DMA BDs cap each dimension size at 1023 (10-bit) -- same _split as r14.
def _split(blk):
    for inner in range(min(blk, 1023), 0, -1):
        if blk % inner == 0 and blk // inner <= 1023:
            return blk // inner, inner
    raise ValueError(f"cannot split {blk} into two <=1023 dims")

def _bd_dims(nblk, blk, stride):
    o, inner = _split(blk)
    return (f"[<size = {nblk}, stride = {stride}>, "
            f"<size = {o}, stride = {inner}>, <size = {inner}, stride = 1>]")

out = ["module {", f"  aie.device({DEVICE}) {{"]
for c in G:
    out.append(f"    %shim{c} = aie.tile({c}, 0)")
    out.append(f"    %mt{c} = aie.tile({c}, 1)")
    for i in R:
        out.append(f"    %c{c}_{i} = aie.tile({c}, {2+i})")

# core-resident activation stripe (one per core, compile-time initialised)
for c in G:
    for i in R:
        out.append(f'    %abuf{c}_{i} = aie.buffer(%c{c}_{i}) {{sym_name = "abuf{c}_{i}"}} : memref<{AB}xi8> = dense<{AVAL}>')

# W: per column j, TWO shim MM2S channels joined in the memtile, then broadcast to the
# 4 cores of column j.
for j in G:
    colcores = ", ".join(f"%c{j}_{i}" for i in R)
    out.append(f"    aie.objectfifo @wbc{j}(%mt{j}, {{{colcores}}}, {DEPTH} : i32) : !aie.objectfifo<memref<{WB}xi8>>")
    if NCH == 2:
        out.append(f"    aie.objectfifo @wsh{j}_0(%shim{j}, {{%mt{j}}}, {DEPTH} : i32) : !aie.objectfifo<memref<{S0}xi8>>")
        out.append(f"    aie.objectfifo @wsh{j}_1(%shim{j}, {{%mt{j}}}, {DEPTH} : i32) : !aie.objectfifo<memref<{S1}xi8>>")
        out.append(f"    aie.objectfifo.link [@wsh{j}_0, @wsh{j}_1] -> [@wbc{j}] ([0, {S0}] [])")
    else:
        out.append(f"    aie.objectfifo @wsh{j}_0(%shim{j}, {{%mt{j}}}, {DEPTH} : i32) : !aie.objectfifo<memref<{WB}xi8>>")
        out.append(f"    aie.objectfifo.link [@wsh{j}_0] -> [@wbc{j}] ([] [0])")
# C: per column j, 4 cores -> memtile -> shim (join). Unchanged from r14.
for j in G:
    ins = ", ".join(f"@cc{j}_{i}" for i in R)
    offs = ", ".join(str(i * CB) for i in R)
    for i in R:
        out.append(f"    aie.objectfifo @cc{j}_{i}(%c{j}_{i}, {{%mt{j}}}, {DEPTH} : i32) : !aie.objectfifo<memref<{CB}xi32>>")
    out.append(f"    aie.objectfifo @csh{j}(%mt{j}, {{%shim{j}}}, {DEPTH} : i32) : !aie.objectfifo<memref<{CJ}xi32>>")
    out.append(f"    aie.objectfifo.link [{ins}] -> [@csh{j}] ([{offs}] [])")

out.append(f'    func.func private @r11_gemm(memref<{AB}xi8>, memref<{WB}xi8>, memref<{CB}xi32>) attributes {{link_with = "r11.o"}}')
for c in G:
    for i in R:
        out.append(f'''    %core{c}_{i} = aie.core(%c{c}_{i}) {{
      %z = arith.constant 0 : index
      %m = arith.constant {INF} : index
      %o = arith.constant 1 : index
      scf.for %k = %z to %m step %o {{
        %w = aie.objectfifo.acquire @wbc{c}(Consume, 1) : !aie.objectfifosubview<memref<{WB}xi8>>
        %wv = aie.objectfifo.subview.access %w[0] : !aie.objectfifosubview<memref<{WB}xi8>> -> memref<{WB}xi8>
        %cc = aie.objectfifo.acquire @cc{c}_{i}(Produce, 1) : !aie.objectfifosubview<memref<{CB}xi32>>
        %cv = aie.objectfifo.subview.access %cc[0] : !aie.objectfifosubview<memref<{CB}xi32>> -> memref<{CB}xi32>
        func.call @r11_gemm(%abuf{c}_{i}, %wv, %cv) : (memref<{AB}xi8>, memref<{WB}xi8>, memref<{CB}xi32>) -> ()
        aie.objectfifo.release @wbc{c}(Consume, 1)
        aie.objectfifo.release @cc{c}_{i}(Produce, 1)
      }}
      aie.end
    }}''')

# runtime sequence: 8 W half-streams (2 per column shim), 4 C joins out.
Wt = 4 * NBLK * WB               # total weight buffer bytes (identical to r14)
Ct = NBLK * CJ
args = ", ".join(["%A: memref<64xi8>", f"%W: memref<{Wt}xi8>", f"%C: memref<{4*Ct}xi32>"])
out.append(f"    aie.runtime_sequence({args}) {{")
for j in G:
    slices = ((j * WB, S0), (j * WB + S0, S1)) if NCH == 2 else ((j * WB, WB),)
    for p, (base_off, chunk) in enumerate(slices):
        out.append(f'''      %tw{j}_{p} = aiex.dma_configure_task_for @wsh{j}_{p} {{
        aie.dma_bd(%W : memref<{Wt}xi8>, {base_off}, {NBLK*chunk}, {_bd_dims(NBLK, chunk, 4*WB)}) {{burst_length = 0 : i32}}
        aie.end
      }}
      aiex.dma_start_task(%tw{j}_{p})''')
for j in G:
    out.append(f'''      %tc{j} = aiex.dma_configure_task_for @csh{j} {{
        aie.dma_bd(%C : memref<{4*Ct}xi32>, {j*Ct}, {Ct}, {_bd_dims(NBLK, CJ, CJ)}) {{burst_length = 0 : i32}}
        aie.end
      }} {{issue_token = true}}
      aiex.dma_start_task(%tc{j})''')
for j in G:
    out.append(f"      aiex.dma_await_task(%tc{j})")
for j in G:
    for p in range(NCH):
        out.append(f"      aiex.dma_free_task(%tw{j}_{p})")
out.append("    }")
out.append("  }")
out.append("}")
print("\n".join(out))
