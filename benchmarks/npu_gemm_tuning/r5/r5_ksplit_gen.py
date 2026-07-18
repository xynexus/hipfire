#!/usr/bin/env python3
# R5 K-SPLIT cascade: the CORRECT GEMM (not r5_gen.py's replication probe).
#
# r5_gen.py broadcasts the SAME A/W tile to all ROWS cores in a column, so the
# cascade sums ROWS identical partials (a throughput probe, output = ROWS x truth).
# Here each of the ROWS cores gets a DIFFERENT K-slice of A and W (per-core fifos),
# so the cascade sum = the full-K dot product -> a numerically correct W4A8 GEMM.
#
# Geometry (aie2p, mmul<4,16,16>): each column produces ONE 4x16 int32 C tile.
#   K = ROWS * KSLICE * 16   (core r owns K-tiles [r*KSLICE, (r+1)*KSLICE))
#   A per core = KSLICE tiles * 64 int8   (4x16 tile, row-major m-fast per r5_cascade ldA)
#   W per core = KSLICE tiles * 128 bytes (16x16 int4 packed, per r5_cascade ldW)
# COLS independent columns -> COLS C tiles. The r5_cascade.cc kernel is UNCHANGED.
#
# Usage: r5_ksplit_gen.py COLS ROWS KSLICE > out.mlir   (build with r5_build.sh, same KSLICE)

import sys

ARCH = "aie2p"
DEV, MAXCOL = "npu2", 8
MMUL_N = 16
AWt = 64                 # int8 per A tile (4x16)
WWt = MMUL_N * 16 // 2   # 128 bytes per W tile (16x16 int4)
CW = 4 * MMUL_N          # 64 int32 per C tile (4x16)
INF = 9223372036854775807

COLS = int(sys.argv[1]) if len(sys.argv) > 1 else 1
ROWS = int(sys.argv[2]) if len(sys.argv) > 2 else 2
KSLICE = int(sys.argv[3]) if len(sys.argv) > 3 else 16
if COLS > MAXCOL:
    sys.exit(f"COLS={COLS} exceeds {MAXCOL}")
if not 2 <= ROWS <= 4:
    sys.exit("ROWS must be 2..4 (cascade depth = compute rows 2..5)")
AW = KSLICE * AWt        # int8 A bytes per core
WW = KSLICE * WWt        # W bytes per core
rows = list(range(2, 2 + ROWS))   # bottom(tail)..top(head)
top = rows[-1]


def core_body(cid, fifo, tile, kern, has_c):
    c_acq = (f'''
        %c = aie.objectfifo.acquire @fc{fifo}(Produce, 1) : !aie.objectfifosubview<memref<{CW}xi32>>
        %cv = aie.objectfifo.subview.access %c[0] : !aie.objectfifosubview<memref<{CW}xi32>> -> memref<{CW}xi32>''' if has_c else "")
    c_rel = "\n        aie.objectfifo.release @fc" + fifo + "(Produce, 1)" if has_c else ""
    call = (f"func.call @{kern}(%av, %wv, %cv) : (memref<{AW}xi8>, memref<{WW}xi8>, memref<{CW}xi32>) -> ()"
            if has_c else
            f"func.call @{kern}(%av, %wv) : (memref<{AW}xi8>, memref<{WW}xi8>) -> ()")
    return f'''    %core_{cid} = aie.core({tile}) {{
      %z = arith.constant 0 : index
      %m = arith.constant {INF} : index
      %o = arith.constant 1 : index
      scf.for %i = %z to %m step %o {{
        %a = aie.objectfifo.acquire @fa{fifo}(Consume, 1) : !aie.objectfifosubview<memref<{AW}xi8>>
        %av = aie.objectfifo.subview.access %a[0] : !aie.objectfifosubview<memref<{AW}xi8>> -> memref<{AW}xi8>
        %w = aie.objectfifo.acquire @fw{fifo}(Consume, 1) : !aie.objectfifosubview<memref<{WW}xi8>>
        %wv = aie.objectfifo.subview.access %w[0] : !aie.objectfifosubview<memref<{WW}xi8>> -> memref<{WW}xi8>{c_acq}
        {call}
        aie.objectfifo.release @fa{fifo}(Consume, 1)
        aie.objectfifo.release @fw{fifo}(Consume, 1){c_rel}
      }}
      aie.end
    }}'''


# One combined shim->memtile stream per column carries all ROWS cores' A|W
# (per core: AW int8 then WW bytes); the memtile link splits it to per-core fifos.
# This respects the shim's 2 out-channels (1 combined stream) and the memtile's 6
# (ROWS*2 <= 6 for ROWS<=3; ROWS=4 needs A|W combined per core — separate variant).
XE = AW + WW                 # combined bytes per core
XTOT_COL = ROWS * XE         # combined bytes per column
out = ["module {", f"  aie.device({DEV}) {{"]
for c in range(COLS):
    out.append(f"    %shim{c} = aie.tile({c}, 0)")
    out.append(f"    %mt{c} = aie.tile({c}, 1)")
    for r in rows:
        out.append(f"    %t{c}_{r} = aie.tile({c}, {r})")
# cascade: north(r+1) -> south(r); tail = row 2 stores C
for c in range(COLS):
    for r in rows[:-1]:
        out.append(f"    aie.cascade_flow(%t{c}_{r+1}, %t{c}_{r})")
# feed: shim -> memtile combined -> per-core A/W (link splits by offset)
for c in range(COLS):
    out.append(f"    aie.objectfifo @fx{c}(%shim{c}, {{%mt{c}}}, 1 : i32) : !aie.objectfifo<memref<{XTOT_COL}xi8>>")
    links = []
    offs = []
    for ri, r in enumerate(rows):
        out.append(f"    aie.objectfifo @fa{c}_{r}(%mt{c}, {{%t{c}_{r}}}, 2 : i32) : !aie.objectfifo<memref<{AW}xi8>>")
        out.append(f"    aie.objectfifo @fw{c}_{r}(%mt{c}, {{%t{c}_{r}}}, 2 : i32) : !aie.objectfifo<memref<{WW}xi8>>")
        links += [f"@fa{c}_{r}", f"@fw{c}_{r}"]
        offs += [str(ri * XE), str(ri * XE + AW)]
    out.append(f"    aie.objectfifo.link [@fx{c}] -> [{', '.join(links)}] ([] [{', '.join(offs)}])")
    out.append(f"    aie.objectfifo @fc{c}_2(%t{c}_2, {{%shim{c}}}, 1 : i32) : !aie.objectfifo<memref<{CW}xi32>>")
out.append(f'    func.func private @r5_cascade_head(memref<{AW}xi8>, memref<{WW}xi8>) attributes {{link_with = "r5_head.o"}}')
out.append(f'    func.func private @r5_cascade_mid(memref<{AW}xi8>, memref<{WW}xi8>) attributes {{link_with = "r5_mid.o"}}')
out.append(f'    func.func private @r5_cascade_tail(memref<{AW}xi8>, memref<{WW}xi8>, memref<{CW}xi32>) attributes {{link_with = "r5_tail.o"}}')
for c in range(COLS):
    for r in rows:
        kern = "r5_cascade_head" if r == top else ("r5_cascade_tail" if r == 2 else "r5_cascade_mid")
        out.append(core_body(f"{c}_{r}", f"{c}_{r}", f"%t{c}_{r}", kern, r == 2))
# runtime sequence: X [COLS*XTOT_COL] combined A|W per core, C [COLS*CW].
# X layout per column: [core0_A(AW), core0_W(WW), core1_A(AW), core1_W(WW), ...].
XTOT, CTOT = COLS * XTOT_COL, COLS * CW
out.append(f"    aie.runtime_sequence(%X: memref<{XTOT}xi8>, %C: memref<{CTOT}xi32>) {{")
for c in range(COLS):
    x_off = c * XTOT_COL
    out.append(f'''      %tx{c} = aiex.dma_configure_task_for @fx{c} {{
        aie.dma_bd(%X : memref<{XTOT}xi8>, {x_off}, {XTOT_COL}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = {XTOT_COL}, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }}
      aiex.dma_start_task(%tx{c})
      %tc{c} = aiex.dma_configure_task_for @fc{c}_2 {{
        aie.dma_bd(%C : memref<{CTOT}xi32>, {c*CW}, {CW}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = {CW}, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }} {{issue_token = true}}
      aiex.dma_start_task(%tc{c})''')
for c in range(COLS):
    out.append(f"      aiex.dma_await_task(%tc{c})")
for c in range(COLS):
    out.append(f"      aiex.dma_free_task(%tx{c})")
out += ["    }", "  }", "}"]
print("\n".join(out))
