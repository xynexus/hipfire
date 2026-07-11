#!/usr/bin/env python3
"""Parallel three-row packing plus direct-stream ring all-gather and W4 down."""

import sys

MODE = sys.argv[1] if len(sys.argv) == 2 else "w4"
if MODE == "w4":
    PREFIX, LINK = "r22", "r22.o"
    OUTBLOCKS, CB, LN, MR, COLS_STRIPE, NM = 3, 2304, 6, 4, 96, 1
elif MODE == "w8":
    PREFIX, LINK = "r23", "r23.o"
    OUTBLOCKS, CB, LN, MR, COLS_STRIPE, NM = 6, 1152, 3, 8, 48, 2
else:
    raise SystemExit("mode must be w4 or w8")

COLS = 8
CORE_ROWS = 4
GROUPS = 5
GROUP = 256
ROWS_PER_CORE = 3
X_CORE = ROWS_PER_CORE * GROUP
X_PAIR = 2 * X_CORE
X_JOIN = COLS * X_CORE
AB = 8192
WB = 16384
CJ = CORE_ROWS * CB
FRAGMENT = ROWS_PER_CORE * GROUP + 16
PAD_M = 288
PAD_N = 768
XBLOCKS = (OUTBLOCKS // NM) * GROUPS
WBLOCKS = OUTBLOCKS * GROUPS
INF = 9223372036854775807


def contiguous_dims(count, block):
    return (
        f"[<size = {count}, stride = {block}>, "
        "<size = 1, stride = 0>, "
        f"<size = {block // 16}, stride = 16>, "
        "<size = 16, stride = 1>]"
    )


def rowmajor_dims():
    return (
        f"[<size = {4 * (24 // MR)}, stride = {MR * PAD_N}>, "
        f"<size = {LN}, stride = 16>, "
        f"<size = {MR}, stride = {PAD_N}>, "
        "<size = 16, stride = 1>]"
    )


out = ["module {", "  aie.device(npu2) {"]
for col in range(COLS):
    out += [f"    %shim{col} = aie.tile({col}, 0)", f"    %mt{col} = aie.tile({col}, 1)"]
    for row in range(CORE_ROWS):
        out += [
            f"    %c{col}_{row} = aie.tile({col}, {row + 2})",
            f"    %apack{col}_{row} = aie.buffer(%c{col}_{row}) {{sym_name = \"apack{col}_{row}\"}} : memref<{AB}xi8>",
            f"    %scratch{col}_{row} = aie.buffer(%c{col}_{row}) {{sym_name = \"scratch{col}_{row}\"}} : memref<256xf32>",
            f"    %own{col}_{row} = aie.buffer(%c{col}_{row}) {{sym_name = \"own{col}_{row}\"}} : memref<{FRAGMENT}xi8>",
            f"    %transit{col}_{row} = aie.buffer(%c{col}_{row}) {{sym_name = \"transit{col}_{row}\"}} : memref<{FRAGMENT}xi8>",
        ]
for col in range(COLS):
    cores = ", ".join(f"%c{col}_{row}" for row in range(CORE_ROWS))
    out += [
        f"    aie.objectfifo @wsh{col}(%shim{col}, {{%mt{col}}}, 1 : i32) : !aie.objectfifo<memref<{WB}xi8>>",
        f"    aie.objectfifo @wbc{col}(%mt{col}, {{{cores}}}, 1 : i32) : !aie.objectfifo<memref<{WB}xi8>>",
        f"    aie.objectfifo.link [@wsh{col}] -> [@wbc{col}] ([] [0])",
    ]
for row in range(CORE_ROWS):
    xpairs = []
    for pair in range(COLS // 2):
        xpairs.append(f"@xpair{pair}_{row}")
        out.append(
            f"    aie.objectfifo @xpair{pair}_{row}(%mt{row}, {{%c{2 * pair}_{row}, %c{2 * pair + 1}_{row}}}, 1 : i32) : !aie.objectfifo<memref<{X_PAIR}xf32>>"
        )
    offsets = ", ".join(str(pair * X_PAIR) for pair in range(COLS // 2))
    out += [
        f"    aie.objectfifo @xsh{row}(%shim{row}, {{%mt{row}}}, 1 : i32) : !aie.objectfifo<memref<{X_JOIN}xf32>>",
        f"    aie.objectfifo.link [@xsh{row}] -> [{', '.join(xpairs)}] ([] [{offsets}])",
    ]
for row in range(CORE_ROWS):
    for col in range(COLS):
        out.append(
            f'    aie.flow(%c{col}_{row}, "Core" : 0, %c{(col + 1) % COLS}_{row}, "Core" : 0)'
        )
for col in range(COLS):
    inputs = ", ".join(f"@cc{col}_{row}" for row in range(CORE_ROWS))
    offsets = ", ".join(str(row * CB) for row in range(CORE_ROWS))
    for row in range(CORE_ROWS):
        out.append(
            f"    aie.objectfifo @cc{col}_{row}(%c{col}_{row}, {{%mt{col}}}, {2 if MODE == 'w8' else 1} : i32) : !aie.objectfifo<memref<{CB}xi32>>"
        )
    out += [
        f"    aie.objectfifo @csh{col}(%mt{col}, {{%shim{col}}}, 1 : i32) : !aie.objectfifo<memref<{CJ}xi32>>",
        f"    aie.objectfifo.link [{inputs}] -> [@csh{col}] ([{offsets}] [])",
    ]
out += [
    f'    func.func private @{PREFIX}_pack3(memref<{X_PAIR}xf32>, memref<{WB}xi8>, memref<{AB}xi8>, memref<256xf32>, memref<{FRAGMENT}xi8>, i32) attributes {{link_with = "{LINK}"}}',
    f'    func.func private @{PREFIX}_insert_fragment(memref<{FRAGMENT}xi8>, memref<{AB}xi8>, i32) attributes {{link_with = "{LINK}"}}',
    f'    func.func private @{PREFIX}_send_fragment(memref<{FRAGMENT}xi8>) attributes {{link_with = "{LINK}"}}',
    f'    func.func private @{PREFIX}_receive_fragment(memref<{FRAGMENT}xi8>) attributes {{link_with = "{LINK}"}}',
]
if MODE == "w4":
    out += [
        f'    func.func private @r15_w4_scaled_init(memref<{AB}xi8>, memref<{WB}xi8>, memref<{CB}xi32>) attributes {{link_with = "{LINK}"}}',
        f'    func.func private @r15_w4_scaled_accum(memref<{AB}xi8>, memref<{WB}xi8>, memref<{CB}xi32>) attributes {{link_with = "{LINK}"}}',
    ]
else:
    out.append(
        f'    func.func private @r23_w8_scaled(memref<{AB}xi8>, memref<{WB}xi8>, memref<{CB}xi32>, i32) attributes {{link_with = "{LINK}"}}'
    )
for col in range(COLS):
    for row in range(CORE_ROWS):
        if MODE == "w8":
            out += [
                f"    %core{col}_{row} = aie.core(%c{col}_{row}) {{",
                "      %z = arith.constant 0 : index",
                f"      %m = arith.constant {INF} : index",
                "      %one = arith.constant 1 : index",
                f"      %mblocks = arith.constant {OUTBLOCKS // NM} : index",
                f"      %groups = arith.constant {GROUPS} : index",
                f"      %owner = arith.constant {col} : i32",
                "      scf.for %outer = %z to %m step %one {",
                "        scf.for %mblock = %z to %mblocks step %one {",
                f"          %c = aie.objectfifo.acquire @cc{col}_{row}(Produce, 2) : !aie.objectfifosubview<memref<{CB}xi32>>",
                f"          %cv0 = aie.objectfifo.subview.access %c[0] : !aie.objectfifosubview<memref<{CB}xi32>> -> memref<{CB}xi32>",
                f"          %cv1 = aie.objectfifo.subview.access %c[1] : !aie.objectfifosubview<memref<{CB}xi32>> -> memref<{CB}xi32>",
                "          scf.for %group = %z to %groups step %one {",
                f"            %w0 = aie.objectfifo.acquire @wbc{col}(Consume, 1) : !aie.objectfifosubview<memref<{WB}xi8>>",
                f"            %wv0 = aie.objectfifo.subview.access %w0[0] : !aie.objectfifosubview<memref<{WB}xi8>> -> memref<{WB}xi8>",
                f"            %x = aie.objectfifo.acquire @xpair{col // 2}_{row}(Consume, 1) : !aie.objectfifosubview<memref<{X_PAIR}xf32>>",
                f"            %xv = aie.objectfifo.subview.access %x[0] : !aie.objectfifosubview<memref<{X_PAIR}xf32>> -> memref<{X_PAIR}xf32>",
                f"            func.call @r23_pack3(%xv, %wv0, %apack{col}_{row}, %scratch{col}_{row}, %own{col}_{row}, %owner) : (memref<{X_PAIR}xf32>, memref<{WB}xi8>, memref<{AB}xi8>, memref<256xf32>, memref<{FRAGMENT}xi8>, i32) -> ()",
                f"            func.call @r23_insert_fragment(%own{col}_{row}, %apack{col}_{row}, %owner) : (memref<{FRAGMENT}xi8>, memref<{AB}xi8>, i32) -> ()",
                f"            aie.objectfifo.release @xpair{col // 2}_{row}(Consume, 1)",
            ]
            for broadcast_owner in range(COLS):
                if col == broadcast_owner:
                    out.append(
                        f"            func.call @r23_send_fragment(%own{col}_{row}) : (memref<{FRAGMENT}xi8>) -> ()"
                    )
                else:
                    out += [
                        f"            func.call @r23_receive_fragment(%transit{col}_{row}) : (memref<{FRAGMENT}xi8>) -> ()",
                        f"            %broadcast_owner{broadcast_owner} = arith.constant {broadcast_owner} : i32",
                        f"            func.call @r23_insert_fragment(%transit{col}_{row}, %apack{col}_{row}, %broadcast_owner{broadcast_owner}) : (memref<{FRAGMENT}xi8>, memref<{AB}xi8>, i32) -> ()",
                    ]
                    if col != (broadcast_owner - 1) % COLS:
                        out.append(
                            f"            func.call @r23_send_fragment(%transit{col}_{row}) : (memref<{FRAGMENT}xi8>) -> ()"
                        )
            out += [
                "            %accumulate = arith.index_cast %group : index to i32",
                f"            func.call @r23_w8_scaled(%apack{col}_{row}, %wv0, %cv0, %accumulate) : (memref<{AB}xi8>, memref<{WB}xi8>, memref<{CB}xi32>, i32) -> ()",
                f"            aie.objectfifo.release @wbc{col}(Consume, 1)",
                f"            %w1 = aie.objectfifo.acquire @wbc{col}(Consume, 1) : !aie.objectfifosubview<memref<{WB}xi8>>",
                f"            %wv1 = aie.objectfifo.subview.access %w1[0] : !aie.objectfifosubview<memref<{WB}xi8>> -> memref<{WB}xi8>",
                f"            func.call @r23_w8_scaled(%apack{col}_{row}, %wv1, %cv1, %accumulate) : (memref<{AB}xi8>, memref<{WB}xi8>, memref<{CB}xi32>, i32) -> ()",
                f"            aie.objectfifo.release @wbc{col}(Consume, 1)",
                "          }",
                f"          aie.objectfifo.release @cc{col}_{row}(Produce, 2)",
                "        }",
                "      }",
                "      aie.end",
                "    } {stack_size = 4096 : i32}",
            ]
            continue
        out += [
            f"    %core{col}_{row} = aie.core(%c{col}_{row}) {{",
            "      %z = arith.constant 0 : index",
            f"      %m = arith.constant {INF} : index",
            "      %one = arith.constant 1 : index",
            f"      %outblocks = arith.constant {OUTBLOCKS} : index",
            f"      %groups = arith.constant {GROUPS} : index",
            f"      %owner = arith.constant {col} : i32",
            "      scf.for %outer = %z to %m step %one {",
            "        scf.for %outblock = %z to %outblocks step %one {",
            f"          %c = aie.objectfifo.acquire @cc{col}_{row}(Produce, 1) : !aie.objectfifosubview<memref<{CB}xi32>>",
            f"          %cv = aie.objectfifo.subview.access %c[0] : !aie.objectfifosubview<memref<{CB}xi32>> -> memref<{CB}xi32>",
            "          scf.for %group = %z to %groups step %one {",
            f"            %w = aie.objectfifo.acquire @wbc{col}(Consume, 1) : !aie.objectfifosubview<memref<{WB}xi8>>",
            f"            %wv = aie.objectfifo.subview.access %w[0] : !aie.objectfifosubview<memref<{WB}xi8>> -> memref<{WB}xi8>",
            f"            %x = aie.objectfifo.acquire @xpair{col // 2}_{row}(Consume, 1) : !aie.objectfifosubview<memref<{X_PAIR}xf32>>",
            f"            %xv = aie.objectfifo.subview.access %x[0] : !aie.objectfifosubview<memref<{X_PAIR}xf32>> -> memref<{X_PAIR}xf32>",
            f"            func.call @{PREFIX}_pack3(%xv, %wv, %apack{col}_{row}, %scratch{col}_{row}, %own{col}_{row}, %owner) : (memref<{X_PAIR}xf32>, memref<{WB}xi8>, memref<{AB}xi8>, memref<256xf32>, memref<{FRAGMENT}xi8>, i32) -> ()",
            f"            func.call @{PREFIX}_insert_fragment(%own{col}_{row}, %apack{col}_{row}, %owner) : (memref<{FRAGMENT}xi8>, memref<{AB}xi8>, i32) -> ()",
            f"            aie.objectfifo.release @xpair{col // 2}_{row}(Consume, 1)",
        ]
        for broadcast_owner in range(COLS):
            if col == broadcast_owner:
                out.append(
                    f"            func.call @{PREFIX}_send_fragment(%own{col}_{row}) : (memref<{FRAGMENT}xi8>) -> ()"
                )
            else:
                out += [
                    f"            func.call @{PREFIX}_receive_fragment(%transit{col}_{row}) : (memref<{FRAGMENT}xi8>) -> ()",
                    f"            %broadcast_owner{broadcast_owner} = arith.constant {broadcast_owner} : i32",
                    f"            func.call @{PREFIX}_insert_fragment(%transit{col}_{row}, %apack{col}_{row}, %broadcast_owner{broadcast_owner}) : (memref<{FRAGMENT}xi8>, memref<{AB}xi8>, i32) -> ()",
                ]
                if col != (broadcast_owner - 1) % COLS:
                    out.append(
                        f"            func.call @{PREFIX}_send_fragment(%transit{col}_{row}) : (memref<{FRAGMENT}xi8>) -> ()"
                    )
        if MODE == "w4":
            out += [
                "            %first = arith.cmpi eq, %group, %z : index",
                "            scf.if %first {",
                f"              func.call @r15_w4_scaled_init(%apack{col}_{row}, %wv, %cv) : (memref<{AB}xi8>, memref<{WB}xi8>, memref<{CB}xi32>) -> ()",
                "            } else {",
                f"              func.call @r15_w4_scaled_accum(%apack{col}_{row}, %wv, %cv) : (memref<{AB}xi8>, memref<{WB}xi8>, memref<{CB}xi32>) -> ()",
                "            }",
            ]
        else:
            out += [
                "            %accumulate = arith.index_cast %group : index to i32",
                f"            func.call @r23_w8_scaled(%apack{col}_{row}, %wv, %cv, %accumulate) : (memref<{AB}xi8>, memref<{WB}xi8>, memref<{CB}xi32>, i32) -> ()",
            ]
        out += [
            f"            aie.objectfifo.release @wbc{col}(Consume, 1)",
            "          }",
            f"          aie.objectfifo.release @cc{col}_{row}(Produce, 1)",
            "        }",
            "      }",
            "      aie.end",
            "    } {stack_size = 4096 : i32}",
        ]

XT = XBLOCKS * X_JOIN
WT = WBLOCKS * WB
out.append(
    f"    aie.runtime_sequence(%X: memref<{CORE_ROWS * XT}xf32>, %W: memref<{COLS * WT}xi8>, %C: memref<{PAD_M * PAD_N}xi32>) {{"
)
for row in range(CORE_ROWS):
    out += [
        f"      %tx{row} = aiex.dma_configure_task_for @xsh{row} {{",
        f"        aie.dma_bd(%X : memref<{CORE_ROWS * XT}xf32>, {row * XT}, {XT}, {contiguous_dims(XBLOCKS, X_JOIN)}) {{burst_length = 0 : i32}}",
        "        aie.end",
        "      }",
        f"      aiex.dma_start_task(%tx{row})",
    ]
for col in range(COLS):
    out += [
        f"      %tw{col} = aiex.dma_configure_task_for @wsh{col} {{",
        f"        aie.dma_bd(%W : memref<{COLS * WT}xi8>, {col * WT}, {WT}, {contiguous_dims(WBLOCKS, WB)}) {{burst_length = 0 : i32}}",
        "        aie.end",
        "      }",
        f"      aiex.dma_start_task(%tw{col})",
    ]
for outblock in range(OUTBLOCKS):
    for col in range(COLS):
        m_macro, n_macro = divmod(outblock, NM)
        offset = m_macro * 96 * PAD_N + n_macro * (COLS * COLS_STRIPE) + col * COLS_STRIPE
        name = f"tc{col}_{outblock}"
        out += [
            f"      %{name} = aiex.dma_configure_task_for @csh{col} {{",
            f"        aie.dma_bd(%C : memref<{PAD_M * PAD_N}xi32>, {offset}, {LN * MR * 16}, {rowmajor_dims()}) {{burst_length = 0 : i32}}",
            "        aie.end",
            f"      }} {{issue_token = true, repeat_count = {4 * (24 // MR) - 1} : i32}}",
            f"      aiex.dma_start_task(%{name})",
        ]
    for col in range(COLS):
        name = f"tc{col}_{outblock}"
        out += [f"      aiex.dma_await_task(%{name})", f"      aiex.dma_free_task(%{name})"]
for row in range(CORE_ROWS):
    out.append(f"      aiex.dma_free_task(%tx{row})")
for col in range(COLS):
    out.append(f"      aiex.dma_free_task(%tw{col})")
out += ["    }", "  }", "}"]
print("\n".join(out))
