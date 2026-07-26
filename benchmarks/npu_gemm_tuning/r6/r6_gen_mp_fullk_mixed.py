#!/usr/bin/env python3
"""Generate one-dispatch full-K W4 plus dense W8-residual partials.

For every K group and N slab the broadcast stream supplies a padded W4 block
followed by one dense W8 residual block. The residual kernel accumulates into
the base int32 tile, so one exact group partial is emitted.
Arbitrary mixed-column Opus overlays are represented by zeros in the dense W8
residual. This spends MACs on zero residuals but avoids a scalar sparse loop.

Usage:
  r6_gen_mp_fullk_mixed.py COLS NB AW CW KGROUPS BASE_OBJECT RESIDUAL_OBJECT
"""

import sys

COLS = int(sys.argv[1])
NB = int(sys.argv[2])
AW = int(sys.argv[3])
CW = int(sys.argv[4])
KGROUPS = int(sys.argv[5])
BASE_OBJECT = sys.argv[6]
RESIDUAL_OBJECT = sys.argv[7]
WW = 16384
INF = 9223372036854775807

ATOT = COLS * KGROUPS * AW
WTOT = KGROUPS * NB * 2 * WW
CTOT = COLS * KGROUPS * NB * CW

out = ["module {", "  aie.device(npu2) {"]
for column in range(COLS):
    out.append(f"    %shim{column} = aie.tile({column}, 0)")
out.append("    %mt = aie.tile(0, 1)")
for column in range(COLS):
    out.append(f"    %t{column} = aie.tile({column}, 2)")

for column in range(COLS):
    out.append(
        f"    aie.objectfifo @fa{column}(%shim{column}, {{%t{column}}}, 1 : i32) : "
        f"!aie.objectfifo<memref<{AW}xi8>>"
    )
    out.append(
        f"    aie.objectfifo @fc{column}(%t{column}, {{%shim{column}}}, 1 : i32) : "
        f"!aie.objectfifo<memref<{CW}xi32>>"
    )
cores = ", ".join(f"%t{column}" for column in range(COLS))
out.append(
    f"    aie.objectfifo @fw_in(%shim0, {{%mt}}, 1 : i32) : "
    f"!aie.objectfifo<memref<{WW}xi8>>"
)
out.append(
    f"    aie.objectfifo @fw(%mt, {{{cores}}}, 1 : i32) : "
    f"!aie.objectfifo<memref<{WW}xi8>>"
)
out.append("    aie.objectfifo.link [@fw_in] -> [@fw]([] [])")
out.append(
    f"    func.func private @r6_mac(memref<{AW}xi8>, memref<{WW}xi8>, "
    f"memref<{CW}xi32>) attributes {{link_with = \"{BASE_OBJECT}\"}}"
)
out.append(
    f"    func.func private @r6_w8_residual(memref<{AW}xi8>, "
    f"memref<{WW}xi8>, memref<{CW}xi32>) "
    f"attributes {{link_with = \"{RESIDUAL_OBJECT}\"}}"
)

for column in range(COLS):
    out.extend([
        f"    %core{column} = aie.core(%t{column}) {{",
        "      %z = arith.constant 0 : index",
        f"      %m = arith.constant {INF} : index",
        f"      %groups = arith.constant {KGROUPS} : index",
        f"      %slabs = arith.constant {NB} : index",
        "      %o = arith.constant 1 : index",
        "      scf.for %i = %z to %m step %o {",
        "        scf.for %group = %z to %groups step %o {",
        f"          %a = aie.objectfifo.acquire @fa{column}(Consume, 1) : "
        f"!aie.objectfifosubview<memref<{AW}xi8>>",
        f"          %av = aie.objectfifo.subview.access %a[0] : "
        f"!aie.objectfifosubview<memref<{AW}xi8>> -> memref<{AW}xi8>",
        "          scf.for %slab = %z to %slabs step %o {",
        f"            %wb = aie.objectfifo.acquire @fw(Consume, 1) : "
        f"!aie.objectfifosubview<memref<{WW}xi8>>",
        f"            %wbv = aie.objectfifo.subview.access %wb[0] : "
        f"!aie.objectfifosubview<memref<{WW}xi8>> -> memref<{WW}xi8>",
        f"            %cb = aie.objectfifo.acquire @fc{column}(Produce, 1) : "
        f"!aie.objectfifosubview<memref<{CW}xi32>>",
        f"            %cbv = aie.objectfifo.subview.access %cb[0] : "
        f"!aie.objectfifosubview<memref<{CW}xi32>> -> memref<{CW}xi32>",
        f"            func.call @r6_mac(%av, %wbv, %cbv) : "
        f"(memref<{AW}xi8>, memref<{WW}xi8>, memref<{CW}xi32>) -> ()",
        "            aie.objectfifo.release @fw(Consume, 1)",
        f"            %wr = aie.objectfifo.acquire @fw(Consume, 1) : "
        f"!aie.objectfifosubview<memref<{WW}xi8>>",
        f"            %wrv = aie.objectfifo.subview.access %wr[0] : "
        f"!aie.objectfifosubview<memref<{WW}xi8>> -> memref<{WW}xi8>",
        f"            func.call @r6_w8_residual(%av, %wrv, %cbv) : "
        f"(memref<{AW}xi8>, memref<{WW}xi8>, memref<{CW}xi32>) -> ()",
        "            aie.objectfifo.release @fw(Consume, 1)",
        f"            aie.objectfifo.release @fc{column}(Produce, 1)",
        "          }",
        f"          aie.objectfifo.release @fa{column}(Consume, 1)",
        "        }",
        "      }",
        "      aie.end",
        "    }",
    ])

args = ", ".join(
    [
        f"%A: memref<{ATOT}xi8>",
        f"%W: memref<{WTOT}xi8>",
        f"%C: memref<{CTOT}xi32>",
    ]
)
out.append(f"    aie.runtime_sequence({args}) {{")
for column in range(COLS):
    rows_per_core = AW // 256
    a_offset = column * KGROUPS * AW
    out.extend(
        [
            f"      %ta{column} = aiex.dma_configure_task_for @fa{column} {{",
            f"        aie.dma_bd(%A : memref<{ATOT}xi8>, {a_offset}, {KGROUPS * AW}, "
            f"[<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, "
            f"<size = {KGROUPS * AW}, stride = 1>]) {{burst_length = 0 : i32}}",
            "        aie.end",
            "      }",
            f"      aiex.dma_start_task(%ta{column})",
        ]
    )
    if KGROUPS <= 5:
      for group in range(KGROUPS):
        c_offset = group * COLS * rows_per_core * NB * 64 + column * rows_per_core * NB * 64
        out.extend([
            f"      %tc{column}_{group} = aiex.dma_configure_task_for @fc{column} {{",
            f"        aie.dma_bd(%C : memref<{CTOT}xi32>, {c_offset}, {NB * CW}, "
            f"[<size = 1, stride = 0>, <size = {NB}, stride = 64>, "
            f"<size = {rows_per_core}, stride = {NB * 64}>, <size = 64, stride = 1>]) "
            "{burst_length = 0 : i32}",
            "        aie.end",
            f"      }} {{issue_token = true}}",
            f"      aiex.dma_start_task(%tc{column}_{group})",
        ])
    else:
        c_offset = column * KGROUPS * NB * CW
        out.extend([
            f"      %tc{column} = aiex.dma_configure_task_for @fc{column} {{",
            f"        aie.dma_bd(%C : memref<{CTOT}xi32>, {c_offset}, {KGROUPS * NB * CW}, "
            f"[<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, "
            f"<size = {KGROUPS * NB * CW}, stride = 1>]) {{burst_length = 0 : i32}}",
            "        aie.end",
            f"      }} {{issue_token = true}}",
            f"      aiex.dma_start_task(%tc{column})",
        ])
out.extend(
    [
        "      %tw = aiex.dma_configure_task_for @fw_in {",
        f"        aie.dma_bd(%W : memref<{WTOT}xi8>, 0, {WTOT}, "
        f"[<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, "
        f"<size = {WTOT}, stride = 1>]) {{burst_length = 0 : i32}}",
        "        aie.end",
        "      }",
        "      aiex.dma_start_task(%tw)",
    ]
)
for column in range(COLS):
    if KGROUPS <= 5:
        for group in range(KGROUPS):
            out.append(f"      aiex.dma_await_task(%tc{column}_{group})")
    else:
        out.append(f"      aiex.dma_await_task(%tc{column})")
for column in range(COLS):
    out.append(f"      aiex.dma_free_task(%ta{column})")
out.append("      aiex.dma_free_task(%tw)")
out.extend(["    }", "  }", "}"])

print("\n".join(out))
