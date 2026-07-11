#!/usr/bin/env python3
"""One-dispatch resident dense-W8 EmbeddingGemma FFN on AIE2P."""

COLS, CORE_ROWS = 8, 4
M_MACROS, GATE_N_MACROS = 3, 6
GATE_GROUPS, DOWN_GROUPS = 3, 5
GATE_OUTBLOCKS = M_MACROS * GATE_N_MACROS
DOWN_MBLOCKS = M_MACROS
DATA_PAIR = 9216
DATA_JOIN = 4 * DATA_PAIR
APACK = 6240
WB = 16384
OUTPUT_CO = 2304
FRAGMENT = 784
SCRATCH = 256
GATE_ACC = 1152
GATE_DATA_BLOCKS = GATE_OUTBLOCKS * GATE_GROUPS
WEIGHT_BLOCKS = GATE_DATA_BLOCKS + DOWN_MBLOCKS * DOWN_GROUPS * 2
T_ROWS, T_STRIDE, INTERMEDIATE, OUTPUT = 296, 5376, 1152, 768
PAD_M = 288
O_ELEMS = PAD_M * OUTPUT
INF = 9223372036854775807


def byte_blocks(count, block):
    return (
        f"[<size = {count}, stride = {block}>, "
        f"<size = {block // 512}, stride = 512>, "
        "<size = 512, stride = 1>]"
    )


def gate_output_dims():
    return (
        f"[<size = 32, stride = {3 * T_STRIDE}>, "
        f"<size = 3, stride = {T_STRIDE}>, "
        "<size = 3, stride = 32>, <size = 32, stride = 1>]"
    )


def down_input_dims():
    return (
        f"[<size = 4, stride = {6 * T_STRIDE}>, "
        f"<size = 8, stride = {T_STRIDE}>, "
        "<size = 12, stride = 96>, <size = 24, stride = 1>]"
    )


def down_output_dims():
    return (
        f"[<size = 3, stride = {32 * OUTPUT}>, "
        f"<size = 32, stride = {OUTPUT}>, "
        "<size = 2, stride = 384>, <size = 48, stride = 1>]"
    )


out = ["module {", "  aie.device(npu2) {"]
for col in range(COLS):
    out += [f"    %shim{col} = aie.tile({col}, 0)", f"    %mt{col} = aie.tile({col}, 1)"]
    for row in range(CORE_ROWS):
        out += [
            f"    %c{col}_{row} = aie.tile({col}, {row + 2})",
            f"    %gacc{col}_{row} = aie.buffer(%c{col}_{row}) {{sym_name = \"gacc{col}_{row}\"}} : memref<{GATE_ACC}xi32>",
            f"    %apack{col}_{row} = aie.buffer(%c{col}_{row}) {{sym_name = \"apack{col}_{row}\"}} : memref<{APACK}xi8>",
            f"    %scratch{col}_{row} = aie.buffer(%c{col}_{row}) {{sym_name = \"scratch{col}_{row}\"}} : memref<{SCRATCH}xf32>",
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
    pairs = []
    for pair in range(COLS // 2):
        pairs.append(f"@xpair{pair}_{row}")
        out.append(
            f"    aie.objectfifo @xpair{pair}_{row}(%mt{row}, "
            f"{{%c{2 * pair}_{row}, %c{2 * pair + 1}_{row}}}, 1 : i32) : "
            f"!aie.objectfifo<memref<{DATA_PAIR}xi8>>"
        )
    offsets = ", ".join(str(pair * DATA_PAIR) for pair in range(COLS // 2))
    out += [
        f"    aie.objectfifo @xsh{row}(%shim{row}, {{%mt{row}}}, 1 : i32) : !aie.objectfifo<memref<{DATA_JOIN}xi8>>",
        f"    aie.objectfifo.link [@xsh{row}] -> [{', '.join(pairs)}] ([] [{offsets}])",
    ]

for row in range(CORE_ROWS):
    for col in range(COLS):
        out.append(
            f'    aie.flow(%c{col}_{row}, "Core" : 0, '
            f'%c{(col + 1) % COLS}_{row}, "Core" : 0)'
        )

for col in range(COLS):
    inputs = ", ".join(f"@oc{col}_{row}" for row in range(CORE_ROWS))
    offsets = ", ".join(str(row * OUTPUT_CO) for row in range(CORE_ROWS))
    for row in range(CORE_ROWS):
        out.append(
            f"    aie.objectfifo @oc{col}_{row}(%c{col}_{row}, {{%mt{col}}}, 1 : i32) : !aie.objectfifo<memref<{OUTPUT_CO}xi32>>"
        )
    out += [
        f"    aie.objectfifo @osh{col}(%mt{col}, {{%shim{col}}}, 1 : i32) : !aie.objectfifo<memref<{CORE_ROWS * OUTPUT_CO}xi32>>",
        f"    aie.objectfifo.link [{inputs}] -> [@osh{col}] ([{offsets}] [])",
    ]

decls = [
    ("r26_gate_scaled", f"memref<{DATA_PAIR}xi8>, memref<{WB}xi8>, memref<{GATE_ACC}xi32>, i32"),
    ("r26_geglu_padded", f"memref<{GATE_ACC}xi32>, memref<{OUTPUT_CO}xi32>"),
    ("r26_pack3", f"memref<{DATA_PAIR}xi8>, memref<{WB}xi8>, memref<{APACK}xi8>, memref<{SCRATCH}xf32>, memref<{FRAGMENT}xi8>, i32, i32"),
    ("r26_insert_fragment", f"memref<{FRAGMENT}xi8>, memref<{APACK}xi8>, i32"),
    ("r26_send_fragment", f"memref<{FRAGMENT}xi8>"),
    ("r26_receive_fragment", f"memref<{FRAGMENT}xi8>"),
    ("r26_down0_scaled", f"memref<{APACK}xi8>, memref<{WB}xi8>, memref<{OUTPUT_CO}xi32>, i32"),
    ("r26_down1_scaled", f"memref<{APACK}xi8>, memref<{WB}xi8>, memref<{OUTPUT_CO}xi32>, i32"),
]
for name, args in decls:
    out.append(
        f'    func.func private @{name}({args}) attributes {{link_with = "r26.o"}}'
    )

for col in range(COLS):
    for row in range(CORE_ROWS):
        lines = [
            f"    %core{col}_{row} = aie.core(%c{col}_{row}) {{",
            "      %z = arith.constant 0 : index",
            f"      %inf = arith.constant {INF} : index",
            "      %one = arith.constant 1 : index",
            f"      %gate_outblocks = arith.constant {GATE_OUTBLOCKS} : index",
            f"      %gate_groups = arith.constant {GATE_GROUPS} : index",
            f"      %down_mblocks = arith.constant {DOWN_MBLOCKS} : index",
            f"      %down_groups = arith.constant {DOWN_GROUPS} : index",
            f"      %owner = arith.constant {col} : i32",
            "      scf.for %outer = %z to %inf step %one {",
            "        scf.for %outblock = %z to %gate_outblocks step %one {",
            f"          %go = aie.objectfifo.acquire @oc{col}_{row}(Produce, 1) : !aie.objectfifosubview<memref<{OUTPUT_CO}xi32>>",
            f"          %gov = aie.objectfifo.subview.access %go[0] : !aie.objectfifosubview<memref<{OUTPUT_CO}xi32>> -> memref<{OUTPUT_CO}xi32>",
            "          scf.for %group = %z to %gate_groups step %one {",
            f"            %x = aie.objectfifo.acquire @xpair{col // 2}_{row}(Consume, 1) : !aie.objectfifosubview<memref<{DATA_PAIR}xi8>>",
            f"            %xv = aie.objectfifo.subview.access %x[0] : !aie.objectfifosubview<memref<{DATA_PAIR}xi8>> -> memref<{DATA_PAIR}xi8>",
            f"            %w = aie.objectfifo.acquire @wbc{col}(Consume, 1) : !aie.objectfifosubview<memref<{WB}xi8>>",
            f"            %wv = aie.objectfifo.subview.access %w[0] : !aie.objectfifosubview<memref<{WB}xi8>> -> memref<{WB}xi8>",
            "            %accumulate = arith.index_cast %group : index to i32",
            f"            func.call @r26_gate_scaled(%xv, %wv, %gacc{col}_{row}, %accumulate) : (memref<{DATA_PAIR}xi8>, memref<{WB}xi8>, memref<{GATE_ACC}xi32>, i32) -> ()",
            f"            aie.objectfifo.release @xpair{col // 2}_{row}(Consume, 1)",
            f"            aie.objectfifo.release @wbc{col}(Consume, 1)",
            "          }",
            f"          func.call @r26_geglu_padded(%gacc{col}_{row}, %gov) : (memref<{GATE_ACC}xi32>, memref<{OUTPUT_CO}xi32>) -> ()",
            f"          aie.objectfifo.release @oc{col}_{row}(Produce, 1)",
            "        }",
            "        scf.for %mblock = %z to %down_mblocks step %one {",
            f"          %do = aie.objectfifo.acquire @oc{col}_{row}(Produce, 1) : !aie.objectfifosubview<memref<{OUTPUT_CO}xi32>>",
            f"          %dov = aie.objectfifo.subview.access %do[0] : !aie.objectfifosubview<memref<{OUTPUT_CO}xi32>> -> memref<{OUTPUT_CO}xi32>",
            "          scf.for %group = %z to %down_groups step %one {",
            f"            %w0 = aie.objectfifo.acquire @wbc{col}(Consume, 1) : !aie.objectfifosubview<memref<{WB}xi8>>",
            f"            %w0v = aie.objectfifo.subview.access %w0[0] : !aie.objectfifosubview<memref<{WB}xi8>> -> memref<{WB}xi8>",
            f"            %x = aie.objectfifo.acquire @xpair{col // 2}_{row}(Consume, 1) : !aie.objectfifosubview<memref<{DATA_PAIR}xi8>>",
            f"            %xv = aie.objectfifo.subview.access %x[0] : !aie.objectfifosubview<memref<{DATA_PAIR}xi8>> -> memref<{DATA_PAIR}xi8>",
            "            %group_i32 = arith.index_cast %group : index to i32",
            f"            func.call @r26_pack3(%xv, %w0v, %apack{col}_{row}, %scratch{col}_{row}, %own{col}_{row}, %owner, %group_i32) : (memref<{DATA_PAIR}xi8>, memref<{WB}xi8>, memref<{APACK}xi8>, memref<{SCRATCH}xf32>, memref<{FRAGMENT}xi8>, i32, i32) -> ()",
            f"            func.call @r26_insert_fragment(%own{col}_{row}, %apack{col}_{row}, %owner) : (memref<{FRAGMENT}xi8>, memref<{APACK}xi8>, i32) -> ()",
            f"            aie.objectfifo.release @xpair{col // 2}_{row}(Consume, 1)",
        ]
        for broadcast_owner in range(COLS):
            if col == broadcast_owner:
                lines.append(
                    f"            func.call @r26_send_fragment(%own{col}_{row}) : (memref<{FRAGMENT}xi8>) -> ()"
                )
            else:
                lines += [
                    f"            func.call @r26_receive_fragment(%transit{col}_{row}) : (memref<{FRAGMENT}xi8>) -> ()",
                    f"            %broadcast_owner{broadcast_owner} = arith.constant {broadcast_owner} : i32",
                    f"            func.call @r26_insert_fragment(%transit{col}_{row}, %apack{col}_{row}, %broadcast_owner{broadcast_owner}) : (memref<{FRAGMENT}xi8>, memref<{APACK}xi8>, i32) -> ()",
                ]
                if col != (broadcast_owner - 1) % COLS:
                    lines.append(
                        f"            func.call @r26_send_fragment(%transit{col}_{row}) : (memref<{FRAGMENT}xi8>) -> ()"
                    )
        lines += [
            "            %accumulate = arith.index_cast %group : index to i32",
            f"            func.call @r26_down0_scaled(%apack{col}_{row}, %w0v, %dov, %accumulate) : (memref<{APACK}xi8>, memref<{WB}xi8>, memref<{OUTPUT_CO}xi32>, i32) -> ()",
            f"            aie.objectfifo.release @wbc{col}(Consume, 1)",
            f"            %w1 = aie.objectfifo.acquire @wbc{col}(Consume, 1) : !aie.objectfifosubview<memref<{WB}xi8>>",
            f"            %w1v = aie.objectfifo.subview.access %w1[0] : !aie.objectfifosubview<memref<{WB}xi8>> -> memref<{WB}xi8>",
            f"            func.call @r26_down1_scaled(%apack{col}_{row}, %w1v, %dov, %accumulate) : (memref<{APACK}xi8>, memref<{WB}xi8>, memref<{OUTPUT_CO}xi32>, i32) -> ()",
            f"            aie.objectfifo.release @wbc{col}(Consume, 1)",
            "          }",
            f"          aie.objectfifo.release @oc{col}_{row}(Produce, 1)",
            "        }",
            "      }",
            "      aie.end",
            "    } {stack_size = 4096 : i32}",
        ]
        out += lines

DATA_ROW = GATE_DATA_BLOCKS * DATA_JOIN
WT = WEIGHT_BLOCKS * WB
out.append(
    f"    aie.runtime_sequence(%D: memref<{CORE_ROWS * DATA_ROW}xi8>, "
    f"%W: memref<{COLS * WT}xi8>, %T: memref<{T_ROWS * T_STRIDE}xf32>, "
    f"%O: memref<{O_ELEMS}xi32>) {{"
)
for row in range(CORE_ROWS):
    out += [
        f"      %tg{row} = aiex.dma_configure_task_for @xsh{row} {{",
        f"        aie.dma_bd(%D : memref<{CORE_ROWS * DATA_ROW}xi8>, {row * DATA_ROW}, {DATA_ROW}, {byte_blocks(GATE_DATA_BLOCKS, DATA_JOIN)}) {{burst_length = 0 : i32}}",
        "        aie.end",
        "      } {issue_token = true}",
        f"      aiex.dma_start_task(%tg{row})",
    ]
for col in range(COLS):
    out += [
        f"      %tw{col} = aiex.dma_configure_task_for @wsh{col} {{",
        f"        aie.dma_bd(%W : memref<{COLS * WT}xi8>, {col * WT}, {WT}, {byte_blocks(WEIGHT_BLOCKS, WB)}) {{burst_length = 0 : i32}}",
        "        aie.end",
        "      } {issue_token = true}",
        f"      aiex.dma_start_task(%tw{col})",
    ]

for outblock in range(GATE_OUTBLOCKS):
    mblock, nblock = divmod(outblock, GATE_N_MACROS)
    for col in range(COLS):
        offset = mblock * 96 * T_STRIDE + nblock * 8 * 96 + col * 96
        name = f"gt{col}_{outblock}"
        out += [
            f"      %{name} = aiex.dma_configure_task_for @osh{col} {{",
            f"        aie.dma_bd(%T : memref<{T_ROWS * T_STRIDE}xf32>, {offset}, 288, {gate_output_dims()}) {{burst_length = 0 : i32}}",
            "        aie.end",
            "      } {issue_token = true, repeat_count = 31 : i32}",
            f"      aiex.dma_start_task(%{name})",
        ]
    for col in range(COLS):
        name = f"gt{col}_{outblock}"
        out += [f"      aiex.dma_await_task(%{name})", f"      aiex.dma_free_task(%{name})"]

for row in range(CORE_ROWS):
    out += [f"      aiex.dma_await_task(%tg{row})", f"      aiex.dma_free_task(%tg{row})"]

for mblock in range(DOWN_MBLOCKS):
    for col in range(COLS):
        offset = mblock * 96 * OUTPUT + col * 48
        name = f"do{col}_{mblock}"
        out += [
            f"      %{name} = aiex.dma_configure_task_for @osh{col} {{",
            f"        aie.dma_bd(%O : memref<{O_ELEMS}xi32>, {offset}, 3072, {down_output_dims()}) {{burst_length = 0 : i32}}",
            "        aie.end",
            "      } {issue_token = true, repeat_count = 2 : i32}",
            f"      aiex.dma_start_task(%{name})",
        ]
    for group in range(DOWN_GROUPS):
        for row in range(CORE_ROWS):
            base_row = mblock * 96 + row * 24
            first_chunk = (group * 256) // 24
            offset = base_row * T_STRIDE + first_chunk * 96
            name = f"dx{row}_{mblock}_{group}"
            out += [
                f"      %{name} = aiex.dma_configure_task_for @xsh{row} {{",
                f"        aie.dma_bd(%T : memref<{T_ROWS * T_STRIDE}xf32>, {offset}, 2304, {down_input_dims()}) {{burst_length = 0 : i32}}",
                "        aie.end",
                "      } {issue_token = true, repeat_count = 3 : i32}",
                f"      aiex.dma_start_task(%{name})",
            ]
        for row in range(CORE_ROWS):
            name = f"dx{row}_{mblock}_{group}"
            out += [f"      aiex.dma_await_task(%{name})", f"      aiex.dma_free_task(%{name})"]
    for col in range(COLS):
        name = f"do{col}_{mblock}"
        out += [f"      aiex.dma_await_task(%{name})", f"      aiex.dma_free_task(%{name})"]

for col in range(COLS):
    out += [f"      aiex.dma_await_task(%tw{col})", f"      aiex.dma_free_task(%tw{col})"]
out += ["    }", "  }", "}"]
print("\n".join(out))
