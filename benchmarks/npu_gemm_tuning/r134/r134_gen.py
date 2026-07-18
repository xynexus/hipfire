#!/usr/bin/env python3
# R134 -- BO-SIZE SCALING of the multi-core W4A8 GEMM weight path (npu1/aie2).
#
# Question this settles: the R14C 8-stream GEMM measured 10.1 GB/s with an 8 MB W
# buffer, while R133 found a 1.77x bandwidth penalty from BO SIZE alone (8 MB ->
# 57 MB at constant read bytes). Real DFlash weights are ~100 MB, so the 2.45x
# "new vs old int_matmul" claim may be an 8 MB microbenchmark artifact.
#
# Differences from r14c_gen.py (which this is derived from -- r14c is NOT modified):
#
#  1. CONTIGUOUS-PER-CHANNEL W LAYOUT. r14c interleaves the four columns' stripes
#     and reads them with an outer BD dim of size N_BLK, which caps N_BLK at 1023
#     (npu1 10-bit BD dims) => max 64 MB. Here each of the 8 shim MM2S channels owns
#     one contiguous region of NBLK*S bytes, so the BD is a plain contiguous run that
#     factors into three <=1023 dims by search => no size cap. R133 measured layout /
#     region count FLAT, so this reorganisation is not expected to move the number;
#     R134_LAYOUT_CTL below is the control that proves it at 8 MB.
#
#  2. PAD -- inflate the %W memref (hence the allocated BO) without changing how many
#     bytes are read. This separates "big BO costs per-byte rate" from "big BO costs a
#     fixed per-dispatch overhead", which is the whole reconciliation R133 could not do.
#
# Everything else is r14c: 8 weight streams (2 shim MM2S per column joined in the
# memtile), core-resident activation buffer (AVAL, default 3), real r11_gemm compute,
# C joined per column and written out. C[0] == AVAL * KT * 16 is a real correctness gate.
#
# Usage: r134_gen.py LM LN KT N_BLK [WBYTES] [DEVICE] [CBASE] [DEPTH] [AVAL] [NCH] [ROWS] [PAD]
import sys

LM     = int(sys.argv[1])  if len(sys.argv) > 1  else 1
LN     = int(sys.argv[2])  if len(sys.argv) > 2  else 4
KT     = int(sys.argv[3])  if len(sys.argv) > 3  else 64
NBLK   = int(sys.argv[4])  if len(sys.argv) > 4  else 128
WBYTES = int(sys.argv[5])  if len(sys.argv) > 5  else 64
DEVICE = sys.argv[6]       if len(sys.argv) > 6  else "npu1"
CBASE  = int(sys.argv[7])  if len(sys.argv) > 7  else 32
DEPTH  = int(sys.argv[8])  if len(sys.argv) > 8  else 2
AVAL   = int(sys.argv[9])  if len(sys.argv) > 9  else 3
NCH    = int(sys.argv[10]) if len(sys.argv) > 10 else 2   # shim MM2S channels per column
ROWS   = int(sys.argv[11]) if len(sys.argv) > 11 else 4   # active core rows per column
PAD    = int(sys.argv[12]) if len(sys.argv) > 12 else 0   # pad %W to >= this many bytes
# Force the innermost (contiguous) BD dim. The free search below maximises it, which
# makes the innermost run length vary with size (512 B at 8-64 MB but 800 B at 100 MB)
# and silently confounds a size sweep with a DMA-burst-shape change. Pin it to compare.
INNER  = int(sys.argv[13]) if len(sys.argv) > 13 else 0

AB = LM * KT * 64        # A-stripe bytes (core-resident)
WB = LN * KT * WBYTES    # W-stripe bytes consumed per block by one column
CB = LM * LN * CBASE     # per-core C i32 elements
CJ = ROWS * CB           # joined C per column
INF = 9223372036854775807
G = range(4)
R = range(ROWS)

S0 = ((WB // 2) // 64) * 64
S1 = WB - S0
if NCH == 2:
    SEG = [S0, S1]
else:
    SEG = [WB]
assert all(s > 0 for s in SEG)

NSTREAM = 4 * NCH


# npu1 (aie2) shim BDs cap each dimension at 1023 (10-bit) and allow 3 dims.
# Factor a contiguous byte count into (a, b, c), each <= 1023, innermost c largest.
def _dims3(n):
    lo = INNER if INNER else 1
    for c in range(INNER if INNER else min(n, 1023), lo - 1, -1):
        if n % c:
            continue
        m = n // c
        for b in range(min(m, 1023), 0, -1):
            if m % b == 0 and m // b <= 1023:
                return m // b, b, c
    raise ValueError(f"cannot factor {n} into three dims <= 1023")


def _bd_contig(n):
    """A plain contiguous run of n bytes, expressed as three <=1023 BD dims."""
    a, b, c = _dims3(n)
    return (f"[<size = {a}, stride = {b*c}>, "
            f"<size = {b}, stride = {c}>, <size = {c}, stride = 1>]")


def _split(blk):
    for inner in range(min(blk, 1023), 0, -1):
        if blk % inner == 0 and blk // inner <= 1023:
            return blk // inner, inner
    raise ValueError(f"cannot split {blk}")


def _bd_dims(nblk, blk, stride):
    o, inner = _split(blk)
    return (f"[<size = {nblk}, stride = {stride}>, "
            f"<size = {o}, stride = {inner}>, <size = {inner}, stride = 1>]")


# Contiguous-per-channel DRAM layout: channel t owns [BASE[t], BASE[t] + NBLK*SEG[..]).
PERCH = [NBLK * SEG[p] for j in G for p in range(NCH)]
BASE = []
acc = 0
for n in PERCH:
    BASE.append(acc)
    acc += n
WREAD = acc                      # bytes actually streamed per dispatch
WBUF = max(WREAD, PAD)           # allocated %W memref (hence BO) size

out = ["module {", f"  aie.device({DEVICE}) {{"]
for c in G:
    out.append(f"    %shim{c} = aie.tile({c}, 0)")
    out.append(f"    %mt{c} = aie.tile({c}, 1)")
    for i in R:
        out.append(f"    %c{c}_{i} = aie.tile({c}, {2+i})")

for c in G:
    for i in R:
        out.append(f'    %abuf{c}_{i} = aie.buffer(%c{c}_{i}) {{sym_name = "abuf{c}_{i}"}} '
                   f': memref<{AB}xi8> = dense<{AVAL}>')

for j in G:
    colcores = ", ".join(f"%c{j}_{i}" for i in R)
    out.append(f"    aie.objectfifo @wbc{j}(%mt{j}, {{{colcores}}}, {DEPTH} : i32) "
               f": !aie.objectfifo<memref<{WB}xi8>>")
    if NCH == 2:
        out.append(f"    aie.objectfifo @wsh{j}_0(%shim{j}, {{%mt{j}}}, {DEPTH} : i32) "
                   f": !aie.objectfifo<memref<{S0}xi8>>")
        out.append(f"    aie.objectfifo @wsh{j}_1(%shim{j}, {{%mt{j}}}, {DEPTH} : i32) "
                   f": !aie.objectfifo<memref<{S1}xi8>>")
        out.append(f"    aie.objectfifo.link [@wsh{j}_0, @wsh{j}_1] -> [@wbc{j}] ([0, {S0}] [])")
    else:
        out.append(f"    aie.objectfifo @wsh{j}_0(%shim{j}, {{%mt{j}}}, {DEPTH} : i32) "
                   f": !aie.objectfifo<memref<{WB}xi8>>")
        out.append(f"    aie.objectfifo.link [@wsh{j}_0] -> [@wbc{j}] ([] [0])")

for j in G:
    ins = ", ".join(f"@cc{j}_{i}" for i in R)
    offs = ", ".join(str(i * CB) for i in R)
    for i in R:
        out.append(f"    aie.objectfifo @cc{j}_{i}(%c{j}_{i}, {{%mt{j}}}, {DEPTH} : i32) "
                   f": !aie.objectfifo<memref<{CB}xi32>>")
    out.append(f"    aie.objectfifo @csh{j}(%mt{j}, {{%shim{j}}}, {DEPTH} : i32) "
               f": !aie.objectfifo<memref<{CJ}xi32>>")
    out.append(f"    aie.objectfifo.link [{ins}] -> [@csh{j}] ([{offs}] [])")

out.append(f'    func.func private @r11_gemm(memref<{AB}xi8>, memref<{WB}xi8>, '
           f'memref<{CB}xi32>) attributes {{link_with = "r11.o"}}')
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

Ct = NBLK * CJ
out.append(f"    aie.runtime_sequence(%A: memref<64xi8>, %W: memref<{WBUF}xi8>, "
           f"%C: memref<{4*Ct}xi32>) {{")
for j in G:
    for p in range(NCH):
        t = j * NCH + p
        out.append(f'''      %tw{j}_{p} = aiex.dma_configure_task_for @wsh{j}_{p} {{
        aie.dma_bd(%W : memref<{WBUF}xi8>, {BASE[t]}, {PERCH[t]}, {_bd_contig(PERCH[t])}) {{burst_length = 0 : i32}}
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
print(f"// W_read={WREAD} W_bo={WBUF} nstream={NSTREAM} perch={PERCH[0]} "
      f"C_bytes={4*Ct*4} expect_c0={AVAL*KT*16}", file=sys.stderr)
