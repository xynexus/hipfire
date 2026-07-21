#!/usr/bin/env python3
# R14B — dual-shim-channel, activation-folded variant of R14 (npu1/aie2).
#
# R14 uses BOTH MM2S channels of every column shim: one for the A-stripe, one for the
# W-stripe. That caps the weight path at 4 DDR streams. At DFlash shapes (M=16, K- and
# N-blocked) the ablation measured it pinned at 9.3-9.8 GB/s of *weight* bytes in every
# config -- not compute-bound, not aggregate-DDR-bound. The A channel carries only
# AB bytes/block while the W channel carries WB (4x more): the two shim channels of a
# column are badly unbalanced.
#
# R14B rebalances them. Per column j and block b, the fused stripe [W_j(b) | A_j(b)]
# (OB = WB + AB bytes) is cut at OB/2 and pulled by BOTH of column j's MM2S channels,
# then reassembled in the memtile by a single link that is simultaneously a JOIN (two
# shim fifos -> one memtile buffer) and a DISTRIBUTE (that buffer -> W broadcast down
# column j + A broadcast across row j):
#
#     link [@wash_j0, @wash_j1] -> [@wbc_j, @abc_j] ([0, S0] [0, WB])
#
# Every one of the 8 shim MM2S channels now moves exactly OB/2 bytes per block, versus
# R14's 16384/4096 split. Crucially the CORE side is unchanged (still exactly two
# inbound objectfifos, W and A) -- an AIE2 core tile has only 2 S2MM channels, so any
# design that gives a core more than two inbound fifos is rejected by aiecc with
# "'aie.tile' op number of input DMA channel exceeded!".
#
# Channel budget per column: shim 2 MM2S / 1 S2MM; memtile 6 S2MM in (2 wash + 4 cc),
# 3 MM2S out (wbc, abc, csh); core 2 S2MM in, 1 MM2S out. All within AIE2 limits.
#
# DDR layout of the single fused buffer: object-major, offset(b, j) = (b*4 + j) * OB,
# channel (j, p) reads base j*OB + p*S0, count NBLK, stride 4*OB, chunk S0/S1.
# Because A and W now come out of the SAME host buffer, the all-ones correctness gate
# fills it with 0x11: A bytes = int8 17, W nibbles = int4 1, so C[0] = 17 * KT * 16.
#
# Usage: r14b_gen.py LM LN KT N_BLK [WBYTES] [DEVICE] [CBASE] [DEPTH] > r14b.mlir
import sys
LM  = int(sys.argv[1]) if len(sys.argv) > 1 else 1
LN  = int(sys.argv[2]) if len(sys.argv) > 2 else 4
KT  = int(sys.argv[3]) if len(sys.argv) > 3 else 64
NBLK = int(sys.argv[4]) if len(sys.argv) > 4 else 128
WBYTES = int(sys.argv[5]) if len(sys.argv) > 5 else 64
DEVICE = sys.argv[6] if len(sys.argv) > 6 else "npu1"
CBASE = int(sys.argv[7]) if len(sys.argv) > 7 else 32
DEPTH = int(sys.argv[8]) if len(sys.argv) > 8 else 2
AB = LM * KT * 64        # A-stripe bytes (shared by a physical row)
WB = LN * KT * WBYTES    # W-stripe bytes (shared by a physical column)
OB = WB + AB             # fused per-column-block object: [W | A]
CB = LM * LN * CBASE     # per-core C i32
CJ = 4 * CB              # joined C per column
INF = 9223372036854775807
G = range(4)

# Balanced cut of the fused stripe across the column's two MM2S channels. Keep the cut
# on a 64-byte (one mmul W-tile / A-tile) boundary so both halves stay DMA-friendly.
S0 = ((OB // 2) // 64) * 64
S1 = OB - S0
if S0 <= 0 or S1 <= 0:
    raise ValueError("fused stripe too small to split")

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
    for i in G:
        out.append(f"    %c{c}_{i} = aie.tile({c}, {2+i})")

for j in G:
    colcores = ", ".join(f"%c{j}_{i}" for i in G)
    rowcores = ", ".join(f"%c{c}_{j}" for c in G)
    out.append(f"    aie.objectfifo @wash{j}_0(%shim{j}, {{%mt{j}}}, {DEPTH} : i32) : !aie.objectfifo<memref<{S0}xi8>>")
    out.append(f"    aie.objectfifo @wash{j}_1(%shim{j}, {{%mt{j}}}, {DEPTH} : i32) : !aie.objectfifo<memref<{S1}xi8>>")
    out.append(f"    aie.objectfifo @wbc{j}(%mt{j}, {{{colcores}}}, {DEPTH} : i32) : !aie.objectfifo<memref<{WB}xi8>>")
    out.append(f"    aie.objectfifo @abc{j}(%mt{j}, {{{rowcores}}}, {DEPTH} : i32) : !aie.objectfifo<memref<{AB}xi8>>")
    out.append(f"    aie.objectfifo.link [@wash{j}_0, @wash{j}_1] -> [@wbc{j}, @abc{j}] ([0, {S0}] [0, {WB}])")
# C: per column j, 4 cores -> memtile -> shim (join). Unchanged from r14.
for j in G:
    ins = ", ".join(f"@cc{j}_{i}" for i in G)
    offs = ", ".join(str(i * CB) for i in G)
    for i in G:
        out.append(f"    aie.objectfifo @cc{j}_{i}(%c{j}_{i}, {{%mt{j}}}, {DEPTH} : i32) : !aie.objectfifo<memref<{CB}xi32>>")
    out.append(f"    aie.objectfifo @csh{j}(%mt{j}, {{%shim{j}}}, {DEPTH} : i32) : !aie.objectfifo<memref<{CJ}xi32>>")
    out.append(f"    aie.objectfifo.link [{ins}] -> [@csh{j}] ([{offs}] [])")

out.append(f'    func.func private @r11_gemm(memref<{AB}xi8>, memref<{WB}xi8>, memref<{CB}xi32>) attributes {{link_with = "r11.o"}}')
for c in G:
    for i in G:
        out.append(f'''    %core{c}_{i} = aie.core(%c{c}_{i}) {{
      %z = arith.constant 0 : index
      %m = arith.constant {INF} : index
      %o = arith.constant 1 : index
      scf.for %k = %z to %m step %o {{
        %a = aie.objectfifo.acquire @abc{i}(Consume, 1) : !aie.objectfifosubview<memref<{AB}xi8>>
        %av = aie.objectfifo.subview.access %a[0] : !aie.objectfifosubview<memref<{AB}xi8>> -> memref<{AB}xi8>
        %w = aie.objectfifo.acquire @wbc{c}(Consume, 1) : !aie.objectfifosubview<memref<{WB}xi8>>
        %wv = aie.objectfifo.subview.access %w[0] : !aie.objectfifosubview<memref<{WB}xi8>> -> memref<{WB}xi8>
        %cc = aie.objectfifo.acquire @cc{c}_{i}(Produce, 1) : !aie.objectfifosubview<memref<{CB}xi32>>
        %cv = aie.objectfifo.subview.access %cc[0] : !aie.objectfifosubview<memref<{CB}xi32>> -> memref<{CB}xi32>
        func.call @r11_gemm(%av, %wv, %cv) : (memref<{AB}xi8>, memref<{WB}xi8>, memref<{CB}xi32>) -> ()
        aie.objectfifo.release @abc{i}(Consume, 1)
        aie.objectfifo.release @wbc{c}(Consume, 1)
        aie.objectfifo.release @cc{c}_{i}(Produce, 1)
      }}
      aie.end
    }}''')

# runtime sequence: 8 fused W|A half-streams (2 per column shim), 4 C joins out.
WT_ = 4 * NBLK * OB              # total fused buffer bytes
Ct = NBLK * CJ                   # per-column C i32 elements
args = ", ".join(["%A: memref<64xi8>", f"%W: memref<{WT_}xi8>", f"%C: memref<{4*Ct}xi32>"])
out.append(f"    aie.runtime_sequence({args}) {{")
for j in G:
    for p, (base_off, chunk) in enumerate(((j * OB, S0), (j * OB + S0, S1))):
        out.append(f'''      %tw{j}_{p} = aiex.dma_configure_task_for @wash{j}_{p} {{
        aie.dma_bd(%W : memref<{WT_}xi8>, {base_off}, {NBLK*chunk}, {_bd_dims(NBLK, chunk, 4*OB)}) {{burst_length = 0 : i32}}
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
    for p in (0, 1):
        out.append(f"      aiex.dma_free_task(%tw{j}_{p})")
out.append("    }")
out.append("  }")
out.append("}")
print("\n".join(out))
