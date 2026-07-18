#!/usr/bin/env python3
# R135 -- BROADCAST vs DISTRIBUTE on both routing topologies (npu1/aie2).
#
# R133 measured TOPO=1 (all-vertical) 10.22 GB/s vs TOPO=2 (4 vertical + 4 horizontal)
# 13.22 GB/s at 8 MiB of DDR reads. Two things needed settling:
#
#  (a) BYTE ACCOUNTING. R133's horizontal link is `([] [0])` = BROADCAST: one WB read
#      from DDR, 4*WB delivered on-chip. If the 13.22 figure had been computed on
#      DELIVERED bytes the lead would be an artifact. It was not: r133's VERTICAL fifo
#      @wbc{j} is *also* a 1-producer/4-consumer broadcast, and r133 holds total DDR
#      bytes constant across TOPO by halving NBLK on each route. Both r133 rows are
#      8 MiB DDR / 32 MiB delivered. So the figures are already DDR-read GB/s and the
#      broadcast-inflation hypothesis is dead on inspection. This round measures the
#      real question instead: does a route that must actually PULL 4*WB from DDR
#      (distribute) sustain the same rate as one that pulls WB and replicates?
#
#  (b) THE JOIN CONFOUND r133 left in. TOPO=1 uses NV=2 shim streams per column joined
#      into one memtile buffer; TOPO=2 uses NV=1 and no join. So r133's delta is
#      "horizontal route" CONFOUNDED WITH "no memtile join". T4 below is vertical-only
#      with NV=1 (no join), which separates them.
#
# Route definitions (identical to r133's):
#   VERT  column j: shim j -> memtile j -> the 4 cores of column j
#   HORIZ row    i: shim i -> memtile i -> cores (c, 2+i) ACROSS all 4 columns
#
# Link modes:
#   b  broadcast  -- producer fifo memref<WB>,   one consumer fifo -> 4 cores.
#                    DDR per iteration per memtile = WB, delivered = 4*WB.
#   b2 broadcast, 2 shim streams joined in the memtile (r133 TOPO=1's vertical).
#   d  DISTRIBUTE -- producer fifo memref<4*WB>, FOUR consumer fifos memref<WB>, one
#                    per core, at output offsets [0, WB, 2WB, 3WB]. Each core gets a
#                    DISTINCT slice, which is what a real GEMM needs.
#                    DDR per iteration per memtile = 4*WB, delivered = 4*WB.
#
# Per-core inbound DMA channels is <= 2 in every config (AIE2 hard limit).
# Memtile MM2S usage peaks at 5 of 6 (VMODE=b + HMODE=d).
#
# Feed-only probe: cores acquire/release and compute nothing. C[0] is 0 and carries NO
# correctness meaning.
#
# Usage: r135_gen.py WB N VMODE HMODE [DEVICE] [DEPTH] [BURST] > r135.mlir
import sys

WB     = int(sys.argv[1]) if len(sys.argv) > 1 else 16384
N      = int(sys.argv[2]) if len(sys.argv) > 2 else 128
VMODE  = sys.argv[3]      if len(sys.argv) > 3 else "b"     # off | b | b2 | d
HMODE  = sys.argv[4]      if len(sys.argv) > 4 else "off"   # off | b | d
DEVICE = sys.argv[5]      if len(sys.argv) > 5 else "npu1"
DEPTH  = int(sys.argv[6]) if len(sys.argv) > 6 else 2
BURST  = int(sys.argv[7]) if len(sys.argv) > 7 else 64
# Acquires per core loop iteration on each route. A broadcast route pulls WB per
# memtile per acquire while a distribute route pulls 4*WB, so REP=4 on a broadcast
# route BALANCES its DDR load against a distribute route on the other axis. Without
# this the dual-route comparison is confounded by one route draining early.
VREP   = int(sys.argv[8]) if len(sys.argv) > 8 else 1
HREP   = int(sys.argv[9]) if len(sys.argv) > 9 else 1

assert VMODE in ("off", "b", "b2", "d")
assert HMODE in ("off", "b", "d")
assert not (VMODE == "off" and HMODE == "off")
assert N * max(VREP, HREP) <= 1023, "outer BD dim cap"

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

# (fifo_name, region_base, n_iter, stride, seg_bytes) for every shim MM2S stream.
streams = []
base = 0

# ---- VERTICAL ----------------------------------------------------------------
if VMODE != "off":
    velem = 4 * WB if VMODE == "d" else WB          # DDR bytes per column per iteration
    nv = 2 if VMODE == "b2" else 1                  # shim streams per column
    segv = velem // nv
    for j in G:
        for s in range(nv):
            out.append(f"    aie.objectfifo @wshv{j}_{s}(%shim{j}, {{%mt{j}}}, {DEPTH} : i32) "
                       f": !aie.objectfifo<memref<{segv}xi8>>")
        if VMODE == "d":
            for i in G:
                out.append(f"    aie.objectfifo @wbc{j}_{i}(%mt{j}, {{%c{j}_{i}}}, {DEPTH} : i32) "
                           f": !aie.objectfifo<memref<{WB}xi8>>")
            dst = ", ".join(f"@wbc{j}_{i}" for i in G)
            doff = ", ".join(str(i * WB) for i in G)
            out.append(f"    aie.objectfifo.link [@wshv{j}_0] -> [{dst}] ([] [{doff}])")
        else:
            cores = ", ".join(f"%c{j}_{i}" for i in G)
            out.append(f"    aie.objectfifo @wbc{j}(%mt{j}, {{{cores}}}, {DEPTH} : i32) "
                       f": !aie.objectfifo<memref<{WB}xi8>>")
            ins = ", ".join(f"@wshv{j}_{s}" for s in range(nv))
            offs = ", ".join(str(s * segv) for s in range(nv))
            out.append(f"    aie.objectfifo.link [{ins}] -> [@wbc{j}] ([{offs}] [])")
        for s in range(nv):
            streams.append((f"wshv{j}_{s}", base + s * segv, N * VREP, velem, segv))
        base += N * VREP * velem

# ---- HORIZONTAL --------------------------------------------------------------
if HMODE != "off":
    helem = 4 * WB if HMODE == "d" else WB
    for i in G:
        out.append(f"    aie.objectfifo @wshh{i}(%shim{i}, {{%mt{i}}}, {DEPTH} : i32) "
                   f": !aie.objectfifo<memref<{helem}xi8>>")
        if HMODE == "d":
            for c in G:
                out.append(f"    aie.objectfifo @wbh{i}_{c}(%mt{i}, {{%c{c}_{i}}}, {DEPTH} : i32) "
                           f": !aie.objectfifo<memref<{WB}xi8>>")
            dst = ", ".join(f"@wbh{i}_{c}" for c in G)
            doff = ", ".join(str(c * WB) for c in G)
            out.append(f"    aie.objectfifo.link [@wshh{i}] -> [{dst}] ([] [{doff}])")
        else:
            cores = ", ".join(f"%c{c}_{i}" for c in G)
            out.append(f"    aie.objectfifo @wbh{i}(%mt{i}, {{{cores}}}, {DEPTH} : i32) "
                       f": !aie.objectfifo<memref<{WB}xi8>>")
            out.append(f"    aie.objectfifo.link [@wshh{i}] -> [@wbh{i}] ([] [0])")
        streams.append((f"wshh{i}", base, N * HREP, helem, helem))
        base += N * HREP * helem

TOT = base

VCE = WB
HCE = WB

# ---- CORES -------------------------------------------------------------------
for c in G:
    for i in G:
        body = []
        if VMODE == "d":
            vf = f"wbc{c}_{i}"
        elif VMODE != "off":
            vf = f"wbc{c}"
        else:
            vf = None
        hf = (f"wbh{i}_{c}" if HMODE == "d" else f"wbh{i}") if HMODE != "off" else None
        for f, rep, esz in ((vf, VREP, VCE), (hf, HREP, HCE)):
            if f is None:
                continue
            for r in range(rep):
                body.append(f'        %{f}_v{r} = aie.objectfifo.acquire @{f}(Consume, 1) '
                            f': !aie.objectfifosubview<memref<{esz}xi8>>')
                body.append(f'        aie.objectfifo.release @{f}(Consume, 1)')
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

# ---- RUNTIME SEQUENCE --------------------------------------------------------
out.append(f"    aie.runtime_sequence(%A: memref<64xi8>, %W: memref<{TOT}xi8>, %C: memref<64xi32>) {{")
for name, off, nblk, stride, seg in streams:
    out.append(f'''      %t_{name} = aiex.dma_configure_task_for @{name} {{
        aie.dma_bd(%W : memref<{TOT}xi8>, {off}, {nblk*seg}, {_bd_dims(nblk, stride, seg)}) {{burst_length = {BURST} : i32}}
        aie.end
      }} {{issue_token = true}}
      aiex.dma_start_task(%t_{name})''')
for name, *_ in streams:
    out.append(f"      aiex.dma_await_task(%t_{name})")
out.append("    }")
out.append("  }")
out.append("}")
print("\n".join(out))

# Accounting to stderr so the runner can echo it.
vddr = 0 if VMODE == "off" else 4 * N * VREP * (4 * WB if VMODE == "d" else WB)
hddr = 0 if HMODE == "off" else 4 * N * HREP * (4 * WB if HMODE == "d" else WB)
vdel = 0 if VMODE == "off" else 16 * N * VREP * WB
hdel = 0 if HMODE == "off" else 16 * N * HREP * WB
assert vddr + hddr == TOT, (vddr, hddr, TOT)
print(f"ACCT vmode={VMODE}x{VREP} hmode={HMODE}x{HREP} WB={WB} N={N} streams={len(streams)} "
      f"ddr_read={vddr+hddr} (v={vddr} h={hddr}) delivered={vdel+hdel} (v={vdel} h={hdel})",
      file=sys.stderr)
