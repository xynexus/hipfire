#!/usr/bin/env python3
# R132 — weight-feed-only bandwidth probe (npu1/aie2). Isolates ONE question:
# does using both shim MM2S channels per column raise weight-fetch bandwidth?
#
# R14 measured the W path pinned at 9.3-9.8 GB/s across every shape, buffer depth,
# activation volume and MAC count -- while r10 measured a single core pulling ~6 GB/s
# from one shim channel. R14 spends 1 of each column's 2 MM2S channels on activations,
# so only 4 of 8 fetch units carry weights. This probe drops A and compute entirely and
# streams ONLY weights, at STREAMS=1 or 2 shim->memtile feeds per column.
#
# STREAMS=2 splits each column's per-block W stripe into two halves that JOIN in the
# memtile (objectfifo.link many->one, the C-join pattern reversed), so memtile->core
# stays a single channel and the 2-inbound-channel core limit is respected.
#
# Usage: r132_gen.py WB NBLK STREAMS [DEVICE] [DEPTH] > r132.mlir
import sys
WB      = int(sys.argv[1]) if len(sys.argv) > 1 else 16384   # W bytes per block per column
NBLK    = int(sys.argv[2]) if len(sys.argv) > 2 else 128
STREAMS = int(sys.argv[3]) if len(sys.argv) > 3 else 1       # shim MM2S feeds per column
DEVICE  = sys.argv[4] if len(sys.argv) > 4 else "npu1"
DEPTH   = int(sys.argv[5]) if len(sys.argv) > 5 else 2
assert STREAMS in (1, 2), "STREAMS must be 1 or 2"
assert WB % STREAMS == 0, "WB must divide evenly across streams"
SEG = WB // STREAMS
INF = 9223372036854775807
G = range(4)

# npu1 DMA BDs cap each dimension at 1023 (10-bit); split contiguous runs (r14_gen.py).
def _split(blk):
    for inner in range(min(blk, 1023), 0, -1):
        if blk % inner == 0 and blk // inner <= 1023:
            return blk // inner, inner
    raise ValueError(f"cannot split {blk}")

def _bd_dims(nblk, stride, seg):
    """nblk segments of `seg` contiguous bytes, `stride` bytes apart."""
    o, inner = _split(seg)
    return (f"[<size = {nblk}, stride = {stride}>, "
            f"<size = {o}, stride = {inner}>, <size = {inner}, stride = 1>]")

out = ["module {", f"  aie.device({DEVICE}) {{"]
for c in G:
    out.append(f"    %shim{c} = aie.tile({c}, 0)")
    out.append(f"    %mt{c} = aie.tile({c}, 1)")
    for i in G:
        out.append(f"    %c{c}_{i} = aie.tile({c}, {2+i})")

for j in G:
    cores = ", ".join(f"%c{j}_{i}" for i in G)
    # STREAMS shim->memtile feeds, joined into one memtile->cores broadcast.
    for s in range(STREAMS):
        out.append(f"    aie.objectfifo @wsh{j}_{s}(%shim{j}, {{%mt{j}}}, {DEPTH} : i32) : !aie.objectfifo<memref<{SEG}xi8>>")
    out.append(f"    aie.objectfifo @wbc{j}(%mt{j}, {{{cores}}}, {DEPTH} : i32) : !aie.objectfifo<memref<{WB}xi8>>")
    ins = ", ".join(f"@wsh{j}_{s}" for s in range(STREAMS))
    offs = ", ".join(str(s * SEG) for s in range(STREAMS))
    out.append(f"    aie.objectfifo.link [{ins}] -> [@wbc{j}] ([{offs}] [])")

# Cores: consume only. No compute -- this is a pure feed probe.
for c in G:
    for i in G:
        out.append(f'''    %core{c}_{i} = aie.core(%c{c}_{i}) {{
      %z = arith.constant 0 : index
      %m = arith.constant {INF} : index
      %o = arith.constant 1 : index
      scf.for %k = %z to %m step %o {{
        %w = aie.objectfifo.acquire @wbc{c}(Consume, 1) : !aie.objectfifosubview<memref<{WB}xi8>>
        aie.objectfifo.release @wbc{c}(Consume, 1)
      }}
      aie.end
    }}''')

Wt = NBLK * WB                      # bytes per column
# Signature mirrors r14 (A, W, C) so npu_gemm_bench drives it unchanged; A and C are
# unused here -- this probe streams weights only.
out.append(f"    aie.runtime_sequence(%A: memref<64xi8>, %W: memref<{4*Wt}xi8>, %C: memref<64xi32>) {{")
for j in G:
    for s in range(STREAMS):
        # Stream s of column j: stride WB between blocks, SEG-sized segment at offset s*SEG.
        base = j * Wt + s * SEG
        out.append(f'''      %tw{j}_{s} = aiex.dma_configure_task_for @wsh{j}_{s} {{
        aie.dma_bd(%W : memref<{4*Wt}xi8>, {base}, {NBLK*SEG}, {_bd_dims(NBLK, WB, SEG)}) {{burst_length = 0 : i32}}
        aie.end
      }} {{issue_token = true}}
      aiex.dma_start_task(%tw{j}_{s})''')
for j in G:
    for s in range(STREAMS):
        out.append(f"      aiex.dma_await_task(%tw{j}_{s})")
out.append("    }")
out.append("  }")
out.append("}")
print("\n".join(out))
