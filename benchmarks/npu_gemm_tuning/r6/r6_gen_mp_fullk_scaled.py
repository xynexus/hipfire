#!/usr/bin/env python3
"""Generate exact W4 GEMM on row 2 and fused f32 scaling on row 3.

Four-column caches use independent activation and scale streams. Eight-column
caches combine those host payloads and split them in each memory tile, leaving
shim 0's second channel for resident broadcast weights. In both cases the row-2
GEMM sees its original exact A/W geometry.
"""

import sys

COLS, NB, AW, WW, CW, KGROUPS = map(int, sys.argv[1:7])
GEMM_OBJECT, SCALE_OBJECT = sys.argv[7:9]
if COLS not in (4, 8):
    raise SystemExit("scaled full-K schedule requires 4 or 8 columns")
COMBINED = COLS == 8
INF = 9223372036854775807
ROWS = AW // 256
# The scale kernel reads the weight scales with a 64-byte `aie::load_v<16>`, so
# the activation-scale region ahead of them must be padded to a multiple of 16
# floats or that load is misaligned and returns wrong lanes. See
# r6_scale_accum.cc. Host-side counterpart: `scale_bytes` in gemm_fullk.rs.
ROWS_PADDED = ((ROWS + 15) // 16) * 16
SE = (ROWS_PADDED + 64) * 4
ATOT = COLS * KGROUPS * AW
XE = AW + SE
XTOT = COLS * NB * KGROUPS * XE
WTOT = KGROUPS * NB * WW
STOT = COLS * NB * KGROUPS * SE
CTOT = COLS * NB * CW

out = ["module {", "  aie.device(npu2) {"]
for col in range(COLS):
    out.append(f"    %shim{col} = aie.tile({col}, 0)")
    if COMBINED:
        out.append(f"    %mt{col} = aie.tile({col}, 1)")
if not COMBINED:
    out.append(f"    %wshim = aie.tile({COLS}, 0)")
    out.append("    %mt = aie.tile(0, 1)")
for col in range(COLS):
    out.append(f"    %g{col} = aie.tile({col}, 2)")
    out.append(f"    %s{col} = aie.tile({col}, 3)")
for col in range(COLS):
    if COMBINED:
        out.append(f"    aie.objectfifo @fx{col}(%shim{col}, {{%mt{col}}}, 1 : i32) : !aie.objectfifo<memref<{XE}xi8>>")
        out.append(f"    aie.objectfifo @fa{col}(%mt{col}, {{%g{col}}}, 1 : i32) : !aie.objectfifo<memref<{AW}xi8>>")
        out.append(f"    aie.objectfifo @fs{col}(%mt{col}, {{%s{col}}}, 1 : i32) : !aie.objectfifo<memref<{SE}xi8>>")
        out.append(f"    aie.objectfifo.link [@fx{col}] -> [@fa{col}, @fs{col}] ([] [0, {AW}])")
    else:
        out.append(f"    aie.objectfifo @fa{col}(%shim{col}, {{%g{col}}}, 1 : i32) : !aie.objectfifo<memref<{AW}xi8>>")
        out.append(f"    aie.objectfifo @fs{col}(%shim{col}, {{%s{col}}}, 1 : i32) : !aie.objectfifo<memref<{SE}xi8>>")
    out.append(f"    aie.objectfifo @fr{col}(%g{col}, {{%s{col}}}, 1 : i32) : !aie.objectfifo<memref<{CW}xi32>>")
    out.append(f"    aie.objectfifo @fc{col}(%s{col}, {{%shim{col}}}, 1 : i32) : !aie.objectfifo<memref<{CW}xi32>>")
gcores = ", ".join(f"%g{col}" for col in range(COLS))
weight_shim = "%shim0" if COMBINED else "%wshim"
weight_mt = "%mt0" if COMBINED else "%mt"
out.append(f"    aie.objectfifo @fw_in({weight_shim}, {{{weight_mt}}}, 1 : i32) : !aie.objectfifo<memref<{WW}xi8>>")
out.append(f"    aie.objectfifo @fw({weight_mt}, {{{gcores}}}, 1 : i32) : !aie.objectfifo<memref<{WW}xi8>>")
out.append("    aie.objectfifo.link [@fw_in] -> [@fw]([] [])")
out.append(f"    func.func private @r6_mac(memref<{AW}xi8>, memref<{WW}xi8>, memref<{CW}xi32>) attributes {{link_with = \"{GEMM_OBJECT}\"}}")
for name in ("r6_scale_init", "r6_scale_accum"):
    out.append(f"    func.func private @{name}(memref<{CW}xi32>, memref<{SE}xi8>, memref<{CW}xi32>) attributes {{link_with = \"{SCALE_OBJECT}\"}}")

for col in range(COLS):
    out.extend([
        f"    %gcore{col} = aie.core(%g{col}) {{",
        "      %z = arith.constant 0 : index",
        f"      %m = arith.constant {INF} : index",
        f"      %groups = arith.constant {KGROUPS} : index",
        f"      %slabs = arith.constant {NB} : index",
        "      %o = arith.constant 1 : index",
        "      scf.for %i = %z to %m step %o {",
        "        scf.for %slab = %z to %slabs step %o {",
        "          scf.for %group = %z to %groups step %o {",
        f"            %a = aie.objectfifo.acquire @fa{col}(Consume, 1) : !aie.objectfifosubview<memref<{AW}xi8>>",
        f"            %av = aie.objectfifo.subview.access %a[0] : !aie.objectfifosubview<memref<{AW}xi8>> -> memref<{AW}xi8>",
        f"            %w = aie.objectfifo.acquire @fw(Consume, 1) : !aie.objectfifosubview<memref<{WW}xi8>>",
        f"            %wv = aie.objectfifo.subview.access %w[0] : !aie.objectfifosubview<memref<{WW}xi8>> -> memref<{WW}xi8>",
        f"            %r = aie.objectfifo.acquire @fr{col}(Produce, 1) : !aie.objectfifosubview<memref<{CW}xi32>>",
        f"            %rv = aie.objectfifo.subview.access %r[0] : !aie.objectfifosubview<memref<{CW}xi32>> -> memref<{CW}xi32>",
        f"            func.call @r6_mac(%av, %wv, %rv) : (memref<{AW}xi8>, memref<{WW}xi8>, memref<{CW}xi32>) -> ()",
        f"            aie.objectfifo.release @fa{col}(Consume, 1)",
        "            aie.objectfifo.release @fw(Consume, 1)",
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
        f"      %slabs = arith.constant {NB} : index",
        "      %o = arith.constant 1 : index",
        "      scf.for %i = %z to %m step %o {",
        "        scf.for %slab = %z to %slabs step %o {",
        f"          %c = aie.objectfifo.acquire @fc{col}(Produce, 1) : !aie.objectfifosubview<memref<{CW}xi32>>",
        f"          %cv = aie.objectfifo.subview.access %c[0] : !aie.objectfifosubview<memref<{CW}xi32>> -> memref<{CW}xi32>",
        f"          %r0 = aie.objectfifo.acquire @fr{col}(Consume, 1) : !aie.objectfifosubview<memref<{CW}xi32>>",
        f"          %rv0 = aie.objectfifo.subview.access %r0[0] : !aie.objectfifosubview<memref<{CW}xi32>> -> memref<{CW}xi32>",
        f"          %s0_{col} = aie.objectfifo.acquire @fs{col}(Consume, 1) : !aie.objectfifosubview<memref<{SE}xi8>>",
        f"          %sv0_{col} = aie.objectfifo.subview.access %s0_{col}[0] : !aie.objectfifosubview<memref<{SE}xi8>> -> memref<{SE}xi8>",
        f"          func.call @r6_scale_init(%rv0, %sv0_{col}, %cv) : (memref<{CW}xi32>, memref<{SE}xi8>, memref<{CW}xi32>) -> ()",
        f"          aie.objectfifo.release @fr{col}(Consume, 1)",
        f"          aie.objectfifo.release @fs{col}(Consume, 1)",
        "          scf.for %group = %o to %groups step %o {",
        f"            %r = aie.objectfifo.acquire @fr{col}(Consume, 1) : !aie.objectfifosubview<memref<{CW}xi32>>",
        f"            %rv = aie.objectfifo.subview.access %r[0] : !aie.objectfifosubview<memref<{CW}xi32>> -> memref<{CW}xi32>",
        f"            %ss_{col} = aie.objectfifo.acquire @fs{col}(Consume, 1) : !aie.objectfifosubview<memref<{SE}xi8>>",
        f"            %ssv_{col} = aie.objectfifo.subview.access %ss_{col}[0] : !aie.objectfifosubview<memref<{SE}xi8>> -> memref<{SE}xi8>",
        f"            func.call @r6_scale_accum(%rv, %ssv_{col}, %cv) : (memref<{CW}xi32>, memref<{SE}xi8>, memref<{CW}xi32>) -> ()",
        f"            aie.objectfifo.release @fr{col}(Consume, 1)",
        f"            aie.objectfifo.release @fs{col}(Consume, 1)",
        "          }",
        f"          aie.objectfifo.release @fc{col}(Produce, 1)",
        "        }",
        "      }",
        "      aie.end",
        "    }",
    ])

if COMBINED:
    out.append(f"    aie.runtime_sequence(%X: memref<{XTOT}xi8>, %W: memref<{WTOT}xi8>, %C: memref<{CTOT}xi32>) {{")
    for col in range(COLS):
        x_offset = col * NB * KGROUPS * XE
        out.extend([
            f"      %tx{col} = aiex.dma_configure_task_for @fx{col} {{",
            f"        aie.dma_bd(%X : memref<{XTOT}xi8>, {x_offset}, {NB * KGROUPS * XE}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = {NB * KGROUPS * XE}, stride = 1>]) {{burst_length = 0 : i32}}",
            "        aie.end",
            "      }",
            f"      aiex.dma_start_task(%tx{col})",
        ])
else:
    out.append(f"    aie.runtime_sequence(%A: memref<{ATOT}xi8>, %W: memref<{WTOT}xi8>, %S: memref<{STOT}xi8>, %C: memref<{CTOT}xi32>) {{")
    for col in range(COLS):
        s_offset = col * NB * KGROUPS * SE
        out.extend([
            f"      %ts{col} = aiex.dma_configure_task_for @fs{col} {{",
            f"        aie.dma_bd(%S : memref<{STOT}xi8>, {s_offset}, {NB * KGROUPS * SE}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = {NB * KGROUPS}, stride = {SE}>, <size = {SE}, stride = 1>]) {{burst_length = 0 : i32}}",
            "        aie.end",
            "      }",
            f"      aiex.dma_start_task(%ts{col})",
        ])
out.extend([
    "      %tw = aiex.dma_configure_task_for @fw_in {",
    f"        aie.dma_bd(%W : memref<{WTOT}xi8>, 0, {WTOT}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = {WTOT}, stride = 1>]) {{burst_length = 0 : i32}}",
    "        aie.end",
    "      }",
    "      aiex.dma_start_task(%tw)",
])
if COMBINED:
    for col in range(COLS):
        c_offset = col * ROWS * NB * 64
        out.extend([
            f"      %tc{col} = aiex.dma_configure_task_for @fc{col} {{",
            f"        aie.dma_bd(%C : memref<{CTOT}xi32>, {c_offset}, {CW}, [<size = {NB}, stride = 64>, <size = 1, stride = 0>, <size = {ROWS}, stride = {NB * 64}>, <size = 64, stride = 1>]) {{burst_length = 0 : i32}}",
            "        aie.end",
            f"      }} {{issue_token = true, repeat_count = {NB - 1} : i32}}",
            f"      aiex.dma_start_task(%tc{col})",
        ])
    for col in range(COLS):
        out.append(f"      aiex.dma_await_task(%tc{col})")
        out.append(f"      aiex.dma_free_task(%tc{col})")
else:
    for slab in range(NB):
        for col in range(COLS):
            a_offset = col * KGROUPS * AW
            c_offset = col * ROWS * NB * 64 + slab * 64
            out.extend([
                f"      %ta{col}_{slab} = aiex.dma_configure_task_for @fa{col} {{",
                f"        aie.dma_bd(%A : memref<{ATOT}xi8>, {a_offset}, {KGROUPS * AW}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = {KGROUPS * AW}, stride = 1>]) {{burst_length = 0 : i32}}",
                "        aie.end",
                f"      }}",
                f"      aiex.dma_start_task(%ta{col}_{slab})",
                f"      %tc{col}_{slab} = aiex.dma_configure_task_for @fc{col} {{",
                f"        aie.dma_bd(%C : memref<{CTOT}xi32>, {c_offset}, {CW}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = {ROWS}, stride = {NB * 64}>, <size = 64, stride = 1>]) {{burst_length = 0 : i32}}",
                "        aie.end",
                f"      }} {{issue_token = true}}",
                f"      aiex.dma_start_task(%tc{col}_{slab})",
            ])
        for col in range(COLS):
            out.append(f"      aiex.dma_await_task(%tc{col}_{slab})")
            out.append(f"      aiex.dma_free_task(%tc{col}_{slab})")
            out.append(f"      aiex.dma_free_task(%ta{col}_{slab})")
for col in range(COLS):
    out.append(f"      aiex.dma_free_task(%{'tx' if COMBINED else 'ts'}{col})")
out.extend(["      aiex.dma_free_task(%tw)", "    }", "  }", "}"])
print("\n".join(out))
