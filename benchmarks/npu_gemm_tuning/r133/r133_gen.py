#!/usr/bin/env python3
# R133 — dual-topology weight feed probe (npu1/aie2).
#
# R132 established the weight path caps at ~10.5 GB/s and that channel count, consumer
# count, depth, burst length and buffer-object split are ALL null. The real r14 GEMM
# nonetheless sustains ~14.0 GB/s of reads -- because W and A ride ORTHOGONAL routing
# topologies concurrently (W: shim_j->memtile_j->down column j; A: shim_i->memtile_i->
# across columns at row i).
#
# The open question: is ~10.5 GB/s a per-TOPOLOGY ceiling? If so, weights can use BOTH
# routes -- the horizontal path is not inherently a replication path, objectfifo.link
# can DISTRIBUTE distinct slices as easily as broadcast identical ones.
#
# TOPO=1: all 8 shim streams vertical (== R132, the 10.5 GB/s baseline).
# TOPO=2: 4 streams vertical + 4 horizontal. Core (j,i) takes @wbc{j} (from memtile j,
#         its own column) AND @wbh{i} (from memtile i, across columns at row i) = exactly
#         2 inbound DMA channels, the compute-tile limit.
# Identical total DDR bytes in both modes; only the on-chip route differs.
#
# Usage: r133_gen.py WB NBLK TOPO [DEVICE] [DEPTH] [BURST] > r133.mlir
import sys
WB     = int(sys.argv[1]) if len(sys.argv) > 1 else 16384
NBLK   = int(sys.argv[2]) if len(sys.argv) > 2 else 128
TOPO   = int(sys.argv[3]) if len(sys.argv) > 3 else 1
DEVICE = sys.argv[4] if len(sys.argv) > 4 else "npu1"
DEPTH  = int(sys.argv[5]) if len(sys.argv) > 5 else 2
BURST  = int(sys.argv[6]) if len(sys.argv) > 6 else 64
assert TOPO in (1, 2)
INF = 9223372036854775807
G = range(4)

def _split(blk):
    for inner in range(min(blk, 1023), 0, -1):
        if blk % inner == 0 and blk // inner <= 1023:
            return blk // inner, inner
    raise ValueError(f"cannot split {blk}")

def _bd_dims(nblk, stride, seg):
    o, inner = _split(seg)
    return (f"[<size = {nblk}, stride = {stride}>, "
            f"<size = {o}, stride = {inner}>, <size = {inner}, stride = 1>]")

out = ["module {", f"  aie.device({DEVICE}) {{"]
for c in G:
    out.append(f"    %shim{c} = aie.tile({c}, 0)")
    out.append(f"    %mt{c} = aie.tile({c}, 1)")
    for i in G:
        out.append(f"    %c{c}_{i} = aie.tile({c}, {2+i})")

# Vertical: shim_j -> memtile_j -> the 4 cores of column j.
NV = 2 if TOPO == 1 else 1          # vertical shim streams per column
SEGV = WB // NV
for j in G:
    cores = ", ".join(f"%c{j}_{i}" for i in G)
    for s in range(NV):
        out.append(f"    aie.objectfifo @wsh{j}_{s}(%shim{j}, {{%mt{j}}}, {DEPTH} : i32) : !aie.objectfifo<memref<{SEGV}xi8>>")
    out.append(f"    aie.objectfifo @wbc{j}(%mt{j}, {{{cores}}}, {DEPTH} : i32) : !aie.objectfifo<memref<{WB}xi8>>")
    ins = ", ".join(f"@wsh{j}_{s}" for s in range(NV))
    offs = ", ".join(str(s * SEGV) for s in range(NV))
    out.append(f"    aie.objectfifo.link [{ins}] -> [@wbc{j}] ([{offs}] [])")

# Horizontal (TOPO=2): shim_i -> memtile_i -> cores (c, 2+i) ACROSS all columns.
# Same physical route r14 uses for A, but carrying weight bytes.
if TOPO == 2:
    for i in G:
        cores = ", ".join(f"%c{c}_{i}" for c in G)
        out.append(f"    aie.objectfifo @wshh{i}(%shim{i}, {{%mt{i}}}, {DEPTH} : i32) : !aie.objectfifo<memref<{WB}xi8>>")
        out.append(f"    aie.objectfifo @wbh{i}(%mt{i}, {{{cores}}}, {DEPTH} : i32) : !aie.objectfifo<memref<{WB}xi8>>")
        out.append(f"    aie.objectfifo.link [@wshh{i}] -> [@wbh{i}] ([] [0])")

for c in G:
    for i in G:
        body = [f'        %w = aie.objectfifo.acquire @wbc{c}(Consume, 1) : !aie.objectfifosubview<memref<{WB}xi8>>',
                f'        aie.objectfifo.release @wbc{c}(Consume, 1)']
        if TOPO == 2:
            body += [f'        %h = aie.objectfifo.acquire @wbh{i}(Consume, 1) : !aie.objectfifosubview<memref<{WB}xi8>>',
                     f'        aie.objectfifo.release @wbh{i}(Consume, 1)']
        nl = chr(10)
        out.append(f'''    %core{c}_{i} = aie.core(%c{c}_{i}) {{
      %z = arith.constant 0 : index
      %m = arith.constant {INF} : index
      %o = arith.constant 1 : index
      scf.for %k = %z to %m step %o {{
{nl.join(body)}
      }}
      aie.end
    }}''')

# Total DDR bytes held constant across TOPO: TOPO=1 streams 4*NBLK*WB; TOPO=2 streams
# 4*NBLK*WB vertical/2 + horizontal/2 by halving NBLK on each route.
NB = NBLK if TOPO == 1 else NBLK // 2
Wt = NB * WB
tot = 4 * Wt * (1 if TOPO == 1 else 2)
out.append(f"    aie.runtime_sequence(%A: memref<64xi8>, %W: memref<{tot}xi8>, %C: memref<64xi32>) {{")
tasks = []
for j in G:
    for s in range(NV):
        base = j * Wt + s * SEGV
        tasks.append(f"tw{j}_{s}")
        out.append(f'''      %tw{j}_{s} = aiex.dma_configure_task_for @wsh{j}_{s} {{
        aie.dma_bd(%W : memref<{tot}xi8>, {base}, {NB*SEGV}, {_bd_dims(NB, WB, SEGV)}) {{burst_length = {BURST} : i32}}
        aie.end
      }} {{issue_token = true}}
      aiex.dma_start_task(%tw{j}_{s})''')
if TOPO == 2:
    for i in G:
        base = 4 * Wt + i * Wt
        tasks.append(f"th{i}")
        out.append(f'''      %th{i} = aiex.dma_configure_task_for @wshh{i} {{
        aie.dma_bd(%W : memref<{tot}xi8>, {base}, {NB*WB}, {_bd_dims(NB, WB, WB)}) {{burst_length = {BURST} : i32}}
        aie.end
      }} {{issue_token = true}}
      aiex.dma_start_task(%th{i})''')
for t in tasks:
    out.append(f"      aiex.dma_await_task(%{t})")
out.append("    }")
out.append("  }")
out.append("}")
print("\n".join(out))
