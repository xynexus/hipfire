#!/usr/bin/env python3
"""Generate an N-PARALLEL exact W4 GEMM on row 2 with fused f32 scaling on row 3.

The sibling `r6_gen_mp_fullk_scaled.py` is M-parallel: `@fw` is a single
objectfifo broadcast to every compute core, every column iterates all `N/64`
slabs, and each column owns only `M/COLS` rows. That means every column streams
the ENTIRE weight matrix to compute its slice of rows — at decode (M=1) it is
COLS-fold wasted weight traffic, which is why wider arrays measure SLOWER there.

This generator inverts what is shared. Each column owns `NB/COLS` slabs of the N
dimension and ALL M rows, so a column reads only its own weights and no byte of
weight traffic is duplicated. Activations are duplicated per column instead —
they are `ROWS*K` bytes against the weights' `K*N/2`, so the trade is heavily
favourable for decode.

Per column: one combined input stream (activations + scale payload, split in the
memtile as in the M-parallel COMBINED path) plus one dedicated weight stream,
which keeps the shim at 2 inputs + 1 output. Column `col` owns global slabs
`col*NBC .. (col+1)*NBC`, i.e. output columns `col*NBC*64 ..`, so its C region is
contiguous.

Args: COLS NB AW WW CW KGROUPS GEMM_OBJECT SCALE_OBJECT
  NB is the TOTAL slab count (N/64); NBC = NB//COLS is per column.
  AW is the per-group activation bytes = ROWS*256 with ROWS = M (all rows).
"""

import sys

COLS, NB, AW, WW, CW, KGROUPS = map(int, sys.argv[1:7])
GEMM_OBJECT, SCALE_OBJECT = sys.argv[7:9]
if NB % COLS:
    raise SystemExit(f"NB={NB} must divide across COLS={COLS}")
NBC = NB // COLS
INF = 9223372036854775807
ROWS = AW // 256
# Weight scales are read with a 64-byte `aie::load_v<16>`, so the activation
# scales ahead of them must be padded to a multiple of 16 floats. Mirrors
# ROWS_PADDED in r6_scale_accum.cc and padded_scale_rows() in gemm_fullk.rs.
ROWS_PADDED = ((ROWS + 15) // 16) * 16
SE = (ROWS_PADDED + 64) * 4
XE = AW + SE
XTOT = COLS * NBC * KGROUPS * XE
WTOT = COLS * NBC * KGROUPS * WW
CTOT = COLS * NBC * CW

out = ["module {", "  aie.device(npu2) {"]
for col in range(COLS):
    out.append(f"    %shim{col} = aie.tile({col}, 0)")
    out.append(f"    %mt{col} = aie.tile({col}, 1)")
for col in range(COLS):
    out.append(f"    %g{col} = aie.tile({col}, 2)")
    out.append(f"    %s{col} = aie.tile({col}, 3)")
for col in range(COLS):
    # Combined activations+scales, split in this column's memtile.
    out.append(f"    aie.objectfifo @fx{col}(%shim{col}, {{%mt{col}}}, 1 : i32) : !aie.objectfifo<memref<{XE}xi8>>")
    out.append(f"    aie.objectfifo @fa{col}(%mt{col}, {{%g{col}}}, 1 : i32) : !aie.objectfifo<memref<{AW}xi8>>")
    out.append(f"    aie.objectfifo @fs{col}(%mt{col}, {{%s{col}}}, 1 : i32) : !aie.objectfifo<memref<{SE}xi8>>")
    out.append(f"    aie.objectfifo.link [@fx{col}] -> [@fa{col}, @fs{col}] ([] [0, {AW}])")
    # Per-column weights: the whole point of this schedule.
    out.append(f"    aie.objectfifo @fw{col}(%shim{col}, {{%g{col}}}, 1 : i32) : !aie.objectfifo<memref<{WW}xi8>>")
    out.append(f"    aie.objectfifo @fr{col}(%g{col}, {{%s{col}}}, 1 : i32) : !aie.objectfifo<memref<{CW}xi32>>")
    out.append(f"    aie.objectfifo @fc{col}(%s{col}, {{%shim{col}}}, 1 : i32) : !aie.objectfifo<memref<{CW}xi32>>")
out.append(f"    func.func private @r6_mac(memref<{AW}xi8>, memref<{WW}xi8>, memref<{CW}xi32>) attributes {{link_with = \"{GEMM_OBJECT}\"}}")
for name in ("r6_scale_init", "r6_scale_accum"):
    out.append(f"    func.func private @{name}(memref<{CW}xi32>, memref<{SE}xi8>, memref<{CW}xi32>) attributes {{link_with = \"{SCALE_OBJECT}\"}}")

for col in range(COLS):
    out.extend([
        f"    %gcore{col} = aie.core(%g{col}) {{",
        "      %z = arith.constant 0 : index",
        f"      %m = arith.constant {INF} : index",
        f"      %groups = arith.constant {KGROUPS} : index",
        f"      %slabs = arith.constant {NBC} : index",
        "      %o = arith.constant 1 : index",
        "      scf.for %i = %z to %m step %o {",
        "        scf.for %slab = %z to %slabs step %o {",
        "          scf.for %group = %z to %groups step %o {",
        f"            %a = aie.objectfifo.acquire @fa{col}(Consume, 1) : !aie.objectfifosubview<memref<{AW}xi8>>",
        f"            %av = aie.objectfifo.subview.access %a[0] : !aie.objectfifosubview<memref<{AW}xi8>> -> memref<{AW}xi8>",
        f"            %w = aie.objectfifo.acquire @fw{col}(Consume, 1) : !aie.objectfifosubview<memref<{WW}xi8>>",
        f"            %wv = aie.objectfifo.subview.access %w[0] : !aie.objectfifosubview<memref<{WW}xi8>> -> memref<{WW}xi8>",
        f"            %r = aie.objectfifo.acquire @fr{col}(Produce, 1) : !aie.objectfifosubview<memref<{CW}xi32>>",
        f"            %rv = aie.objectfifo.subview.access %r[0] : !aie.objectfifosubview<memref<{CW}xi32>> -> memref<{CW}xi32>",
        f"            func.call @r6_mac(%av, %wv, %rv) : (memref<{AW}xi8>, memref<{WW}xi8>, memref<{CW}xi32>) -> ()",
        f"            aie.objectfifo.release @fa{col}(Consume, 1)",
        f"            aie.objectfifo.release @fw{col}(Consume, 1)",
        f"            aie.objectfifo.release @fr{col}(Produce, 1)",
        "          }",
        "        }",
        "      }",
        "      aie.end",
        "    }",
        f"    %score{col} = aie.core(%s{col}) {{",
        "      %z = arith.constant 0 : index",
        f"      %m = arith.constant {INF} : index",
        f"      %groups = arith.constant {KGROUPS} : index",
        f"      %slabs = arith.constant {NBC} : index",
        "      %o = arith.constant 1 : index",
        "      scf.for %i = %z to %m step %o {",
        "        scf.for %slab = %z to %slabs step %o {",
        f"          %c = aie.objectfifo.acquire @fc{col}(Produce, 1) : !aie.objectfifosubview<memref<{CW}xi32>>",
        f"          %cv = aie.objectfifo.subview.access %c[0] : !aie.objectfifosubview<memref<{CW}xi32>> -> memref<{CW}xi32>",
        f"          %r0 = aie.objectfifo.acquire @fr{col}(Consume, 1) : !aie.objectfifosubview<memref<{CW}xi32>>",
        f"          %rv0 = aie.objectfifo.subview.access %r0[0] : !aie.objectfifosubview<memref<{CW}xi32>> -> memref<{CW}xi32>",
        f"          %sq0 = aie.objectfifo.acquire @fs{col}(Consume, 1) : !aie.objectfifosubview<memref<{SE}xi8>>",
        f"          %sqv0 = aie.objectfifo.subview.access %sq0[0] : !aie.objectfifosubview<memref<{SE}xi8>> -> memref<{SE}xi8>",
        f"          func.call @r6_scale_init(%rv0, %sqv0, %cv) : (memref<{CW}xi32>, memref<{SE}xi8>, memref<{CW}xi32>) -> ()",
        f"          aie.objectfifo.release @fr{col}(Consume, 1)",
        f"          aie.objectfifo.release @fs{col}(Consume, 1)",
        "          scf.for %group = %o to %groups step %o {",
        f"            %r = aie.objectfifo.acquire @fr{col}(Consume, 1) : !aie.objectfifosubview<memref<{CW}xi32>>",
        f"            %rv = aie.objectfifo.subview.access %r[0] : !aie.objectfifosubview<memref<{CW}xi32>> -> memref<{CW}xi32>",
        f"            %sqa = aie.objectfifo.acquire @fs{col}(Consume, 1) : !aie.objectfifosubview<memref<{SE}xi8>>",
        f"            %sqav = aie.objectfifo.subview.access %sqa[0] : !aie.objectfifosubview<memref<{SE}xi8>> -> memref<{SE}xi8>",
        f"            func.call @r6_scale_accum(%rv, %sqav, %cv) : (memref<{CW}xi32>, memref<{SE}xi8>, memref<{CW}xi32>) -> ()",
        f"            aie.objectfifo.release @fr{col}(Consume, 1)",
        f"            aie.objectfifo.release @fs{col}(Consume, 1)",
        "          }",
        f"          aie.objectfifo.release @fc{col}(Produce, 1)",
        "        }",
        "      }",
        "      aie.end",
        "    }",
    ])

N = COLS * NBC * 64
out.append(f"    aie.runtime_sequence(%X: memref<{XTOT}xi8>, %W: memref<{WTOT}xi8>, %C: memref<{CTOT}xi32>) {{")
for col in range(COLS):
    x_offset = col * NBC * KGROUPS * XE
    w_offset = col * NBC * KGROUPS * WW
    out.extend([
        f"      %tx{col} = aiex.dma_configure_task_for @fx{col} {{",
        f"        aie.dma_bd(%X : memref<{XTOT}xi8>, {x_offset}, {NBC * KGROUPS * XE}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = {NBC * KGROUPS * XE}, stride = 1>]) {{burst_length = 0 : i32}}",
        "        aie.end",
        "      }",
        f"      aiex.dma_start_task(%tx{col})",
        f"      %tw{col} = aiex.dma_configure_task_for @fw{col} {{",
        f"        aie.dma_bd(%W : memref<{WTOT}xi8>, {w_offset}, {NBC * KGROUPS * WW}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = {NBC * KGROUPS * WW}, stride = 1>]) {{burst_length = 0 : i32}}",
        "        aie.end",
        "      }",
        f"      aiex.dma_start_task(%tw{col})",
    ])
# One output task per (col, slab): C is [row][n] and this column's slab owns the
# 64 columns at (col*NBC + slab)*64, so each row contributes one 64-wide run.
for slab in range(NBC):
    for col in range(COLS):
        c_offset = (col * NBC + slab) * 64
        out.extend([
            f"      %tc{col}_{slab} = aiex.dma_configure_task_for @fc{col} {{",
            f"        aie.dma_bd(%C : memref<{CTOT}xi32>, {c_offset}, {CW}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = {ROWS}, stride = {N}>, <size = 64, stride = 1>]) {{burst_length = 0 : i32}}",
            "        aie.end",
            "      } {issue_token = true}",
            f"      aiex.dma_start_task(%tc{col}_{slab})",
        ])
    for col in range(COLS):
        out.append(f"      aiex.dma_await_task(%tc{col}_{slab})")
        out.append(f"      aiex.dma_free_task(%tc{col}_{slab})")
for col in range(COLS):
    out.append(f"      aiex.dma_free_task(%tx{col})")
    out.append(f"      aiex.dma_free_task(%tw{col})")
out.extend(["    }", "  }", "}"])
print("\n".join(out))
