#!/usr/bin/env python3
# R133 — weight-feed-only bandwidth probe with DRAM REGION SPREAD (npu1/aie2).
#
# Settles the r132 locality anomaly: r132's 8 shim MM2S channels all draw from ONE
# contiguous 8 MB DRAM region and measure 10.37 GB/s, while the r14 GEMM pulls 50%
# MORE read bytes (A + W in separate buffers) at 13.99 GB/s. Question: is the
# ~10.4 GB/s "weight ceiling" a DRAM bank/page locality artifact of the probe?
#
# Same topology as r132 STREAMS=2 (2 shim->memtile feeds per column, joined in the
# memtile so each core keeps 1 inbound objectfifo; AIE2 cores have only 2 inbound
# DMA channels). ONE change: the 8 stream base addresses are spread across REGIONS
# distinct DRAM regions, SPREAD bytes apart. Total read bytes are held constant at
# 4 * NBLK * WB regardless of REGIONS, so every point is directly comparable.
#
#   REGIONS=1 -> all 8 streams in one dense contiguous region (r132-equivalent control)
#   REGIONS=2/4 -> intermediate points (threshold vs gradient)
#   REGIONS=8 -> every channel in its own widely-separated region
#
# Feed-only: cores consume and release, no compute. Output is meaningless (zeros);
# this probe has NO correctness claim.
#
# Usage: r133_gen.py WB NBLK REGIONS SPREAD_MB [DEVICE] [DEPTH] [NCORES] > r133.mlir
import sys
WB       = int(sys.argv[1]) if len(sys.argv) > 1 else 16384  # W bytes per block per column
NBLK     = int(sys.argv[2]) if len(sys.argv) > 2 else 128
REGIONS  = int(sys.argv[3]) if len(sys.argv) > 3 else 8      # distinct DRAM regions
SPREAD_MB= int(sys.argv[4]) if len(sys.argv) > 4 else 8      # MB between region bases
DEVICE   = sys.argv[5] if len(sys.argv) > 5 else "npu1"
DEPTH    = int(sys.argv[6]) if len(sys.argv) > 6 else 2
NCORES   = int(sys.argv[7]) if len(sys.argv) > 7 else 4      # consumers per column
PADBUF   = int(sys.argv[8]) if len(sys.argv) > 8 else 0      # pad W memref to >= this many bytes
SPLIT    = int(sys.argv[9]) if len(sys.argv) > 9 else 1      # 1 = all reads from %W; 2 = half from %A (separate BO)

STREAMS = 2                     # shim MM2S feeds per column (r132 STREAMS=2 topology)
NSTREAM = 4 * STREAMS           # 8 total
assert REGIONS in (1, 2, 4, 8), "REGIONS must divide 8"
assert WB % STREAMS == 0
SEG    = WB // STREAMS          # bytes this stream contributes per block
PERSTR = NBLK * SEG             # bytes read by one stream (1 MB at defaults)
PERREG = NSTREAM // REGIONS     # streams sharing a region
SPREAD = SPREAD_MB * 1024 * 1024
assert REGIONS == 1 or SPREAD >= PERREG * PERSTR, "regions would overlap"
WBUF   = max((REGIONS - 1) * SPREAD + PERREG * PERSTR, PADBUF)
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

def _base(t):
    """DRAM base of stream t: region t//PERREG, slot t%PERREG inside it."""
    return (t // PERREG) * SPREAD + (t % PERREG) * PERSTR

def _arg(t):
    if SPLIT == 2 and t < NSTREAM // 2:
        return f"%A : memref<{(NSTREAM//2)*PERSTR}xi8>"
    return f"%W : memref<{WBUF}xi8>"

def _off(t):
    if SPLIT == 2:
        return (t % (NSTREAM // 2)) * PERSTR
    return _base(t)

out = ["module {", f"  aie.device({DEVICE}) {{"]
for c in G:
    out.append(f"    %shim{c} = aie.tile({c}, 0)")
    out.append(f"    %mt{c} = aie.tile({c}, 1)")
    for i in G:
        out.append(f"    %c{c}_{i} = aie.tile({c}, {2+i})")

for j in G:
    cores = ", ".join(f"%c{j}_{i}" for i in range(NCORES))
    for s in range(STREAMS):
        out.append(f"    aie.objectfifo @wsh{j}_{s}(%shim{j}, {{%mt{j}}}, {DEPTH} : i32) : !aie.objectfifo<memref<{SEG}xi8>>")
    out.append(f"    aie.objectfifo @wbc{j}(%mt{j}, {{{cores}}}, {DEPTH} : i32) : !aie.objectfifo<memref<{WB}xi8>>")
    ins  = ", ".join(f"@wsh{j}_{s}" for s in range(STREAMS))
    offs = ", ".join(str(s * SEG) for s in range(STREAMS))
    out.append(f"    aie.objectfifo.link [{ins}] -> [@wbc{j}] ([{offs}] [])")

for c in G:
    for i in range(NCORES):
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

# Signature mirrors r14/r132 (A, W, C) so npu_gemm_bench drives it unchanged.
ABUF = (NSTREAM // 2) * PERSTR if SPLIT == 2 else 64
if SPLIT == 2:
    WBUF = max((NSTREAM // 2) * PERSTR, PADBUF)
out.append(f"    aie.runtime_sequence(%A: memref<{ABUF}xi8>, %W: memref<{WBUF}xi8>, %C: memref<64xi32>) {{")
for j in G:
    for s in range(STREAMS):
        t = j * STREAMS + s
        # Each stream reads PERSTR contiguous bytes at its own region base.
        out.append(f'''      %tw{j}_{s} = aiex.dma_configure_task_for @wsh{j}_{s} {{
        aie.dma_bd({_arg(t)}, {_off(t)}, {PERSTR}, {_bd_dims(NBLK, SEG, SEG)}) {{burst_length = 0 : i32}}
        aie.end
      }} {{issue_token = true}}
      aiex.dma_start_task(%tw{j}_{s})''')
for j in G:
    for s in range(STREAMS):
        out.append(f"      aiex.dma_await_task(%tw{j}_{s})")
out.append("    }")
out.append("  }")
out.append("}")
print("\n".join(out), file=sys.stderr if False else sys.stdout)
print(f"// WBUF={WBUF} read_bytes={NSTREAM*PERSTR} regions={REGIONS} spread={SPREAD}", file=sys.stderr)
