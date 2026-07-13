#!/usr/bin/env python3
"""R93 canonical BF16 pre-FFN state to the resident R25 W4 activation ABI."""

COLS, CORE_ROWS = 8, 4
ROWS_PER_CORE, GROUPS, HIDDEN = 8, 3, 768
GROUP = 256
M = 256
PAD_M = 288
SOURCE_ROW_BYTES = HIDDEN * 2
X_ROW_BYTES = 2 * SOURCE_ROW_BYTES
X_BYTES = PAD_M * SOURCE_ROW_BYTES
X_JOIN_BYTES = CORE_ROWS * X_ROW_BYTES
PARAM_RECORD = X_ROW_BYTES
PARAM_TOTAL = GROUPS * CORE_ROWS * PARAM_RECORD
CHUNK_BYTES = ROWS_PER_CORE * GROUP + ROWS_PER_CORE * 4
BLOCK_PREFIX = 3 * ROWS_PER_CORE * GROUP + 3 * ROWS_PER_CORE * 4
R25_BLOCK = 6656
R25_BLOCKS = 4 * 27
R25_INPUT_BYTES = R25_BLOCKS * R25_BLOCK
INF = 9223372036854775807


def x_dims():
    return (
        f"[<size = {ROWS_PER_CORE}, stride = {SOURCE_ROW_BYTES}>, "
        f"<size = {CORE_ROWS}, stride = {ROWS_PER_CORE * SOURCE_ROW_BYTES}>, "
        f"<size = {X_ROW_BYTES // 32}, stride = 32>, "
        f"<size = 32, stride = 1>]"
    )


cores = [(col, row) for col in range(COLS) for row in range(CORE_ROWS)]
chains = [cores[index : index + 3] for index in range(0, len(cores), 3)]
roles = {}
packers = []
for block, chain in enumerate(chains):
    for owner, core in enumerate(chain):
        roles[core] = (block, owner, len(chain))
    packers.append(chain[-1])

out = ["module {", "  aie.device(npu2) {"]
for col in range(COLS):
    out += [f"    %shim{col} = aie.tile({col}, 0)", f"    %mt{col} = aie.tile({col}, 1)"]
    for row in range(CORE_ROWS):
        out += [
            f"    %c{col}_{row} = aie.tile({col}, {row + 2})",
            f'    %scratch{col}_{row} = aie.buffer(%c{col}_{row}) {{sym_name = "scratch{col}_{row}"}} : memref<256xf32>',
            f'    %chunk{col}_{row} = aie.buffer(%c{col}_{row}) {{sym_name = "chunk{col}_{row}"}} : memref<{CHUNK_BYTES}xi8>',
            *[
                f'    %param{group}_{col}_{row} = aie.buffer(%c{col}_{row}) {{sym_name = "param{group}_{col}_{row}"}} : memref<{PARAM_RECORD}xi8>'
                for group in range(GROUPS)
            ],
        ]
        if (col, row) in packers:
            out.append(
                f'    %block{col}_{row} = aie.buffer(%c{col}_{row}) {{sym_name = "block{col}_{row}"}} : memref<{BLOCK_PREFIX}xi8>'
            )

for col in range(COLS):
    xcores = []
    offsets = []
    for row in range(CORE_ROWS):
        xcores.append(f"@xc{col}_{row}")
        offsets.append(str(row * X_ROW_BYTES))
        out.append(
            f"    aie.objectfifo @xc{col}_{row}(%mt{col}, {{%c{col}_{row}}}, 1 : i32) : !aie.objectfifo<memref<{X_ROW_BYTES}xi8>>"
        )
    out += [
        f"    aie.objectfifo @xsh{col}(%shim{col}, {{%mt{col}}}, 1 : i32) : !aie.objectfifo<memref<{X_JOIN_BYTES}xi8>>",
        f"    aie.objectfifo.link [@xsh{col}] -> [{', '.join(xcores)}] ([] [{', '.join(offsets)}])",
    ]

for chain in chains:
    for source, target in zip(chain, chain[1:]):
        out.append(
            f'    aie.flow(%c{source[0]}_{source[1]}, "Core" : 0, %c{target[0]}_{target[1]}, "Core" : 0)'
        )

for col, row in packers:
    out += [
        f"    aie.objectfifo @oc{col}_{row}(%c{col}_{row}, {{%mt{col}}}, 1 : i32) : !aie.objectfifo<memref<{BLOCK_PREFIX}xi8>>",
        f"    aie.objectfifo @osh{col}_{row}(%mt{col}, {{%shim{col}}}, 1 : i32) : !aie.objectfifo<memref<{BLOCK_PREFIX}xi8>>",
        f"    aie.objectfifo.link [@oc{col}_{row}] -> [@osh{col}_{row}] ([] [])",
    ]

out += [
    f'    func.func private @r93_pack_rows(memref<{X_ROW_BYTES}xi8>, memref<{PARAM_RECORD}xi8>, memref<{CHUNK_BYTES}xi8>, memref<256xf32>, i32, i32) attributes {{link_with = "r93.o"}}',
    f'    func.func private @r93_copy_param(memref<{X_ROW_BYTES}xi8>, memref<{PARAM_RECORD}xi8>) attributes {{link_with = "r93.o"}}',
    f'    func.func private @r93_send_chunk(memref<{CHUNK_BYTES}xi8>) attributes {{link_with = "r93.o"}}',
    f'    func.func private @r93_relay_then_send(memref<{CHUNK_BYTES}xi8>) attributes {{link_with = "r93.o"}}',
    f'    func.func private @r93_assemble_block(memref<{BLOCK_PREFIX}xi8>, memref<{CHUNK_BYTES}xi8>, i32) attributes {{link_with = "r93.o"}}',
    f'    func.func private @r93_emit_block(memref<{BLOCK_PREFIX}xi8>, memref<{BLOCK_PREFIX}xi8>) attributes {{link_with = "r93.o"}}',
]

for col, row in cores:
    block, owner, chain_len = roles[(col, row)]
    out += [
        f"    %core{col}_{row} = aie.core(%c{col}_{row}) {{",
        "      %z = arith.constant 0 : index",
        f"      %inf = arith.constant {INF} : index",
        "      %one = arith.constant 1 : index",
        f"      %rows = arith.constant {ROWS_PER_CORE} : index",
        "      scf.for %outer = %z to %inf step %one {",
    ]
    for group in range(GROUPS):
        out += [
            f"        %pload{group} = aie.objectfifo.acquire @xc{col}_{row}(Consume, 1) : !aie.objectfifosubview<memref<{X_ROW_BYTES}xi8>>",
            f"        %pload{group}v = aie.objectfifo.subview.access %pload{group}[0] : !aie.objectfifosubview<memref<{X_ROW_BYTES}xi8>> -> memref<{X_ROW_BYTES}xi8>",
            f"        func.call @r93_copy_param(%pload{group}v, %param{group}_{col}_{row}) : (memref<{X_ROW_BYTES}xi8>, memref<{PARAM_RECORD}xi8>) -> ()",
            f"        aie.objectfifo.release @xc{col}_{row}(Consume, 1)",
        ]
    for group in range(GROUPS):
        out += [
            "        scf.for %row = %z to %rows step %one {",
            f"          %x{group} = aie.objectfifo.acquire @xc{col}_{row}(Consume, 1) : !aie.objectfifosubview<memref<{X_ROW_BYTES}xi8>>",
            f"          %x{group}v = aie.objectfifo.subview.access %x{group}[0] : !aie.objectfifosubview<memref<{X_ROW_BYTES}xi8>> -> memref<{X_ROW_BYTES}xi8>",
            "          %rowi = arith.index_cast %row : index to i32",
            f"          %group{group} = arith.constant {group} : i32",
            f"          func.call @r93_pack_rows(%x{group}v, %param{group}_{col}_{row}, %chunk{col}_{row}, %scratch{col}_{row}, %rowi, %group{group}) : (memref<{X_ROW_BYTES}xi8>, memref<{PARAM_RECORD}xi8>, memref<{CHUNK_BYTES}xi8>, memref<256xf32>, i32, i32) -> ()",
            f"          aie.objectfifo.release @xc{col}_{row}(Consume, 1)",
            "        }",
        ]
        if owner == 0 and chain_len > 1:
            out.append(
                f"        func.call @r93_send_chunk(%chunk{col}_{row}) : (memref<{CHUNK_BYTES}xi8>) -> ()"
            )
        elif owner < chain_len - 1:
            out.append(
                f"        func.call @r93_relay_then_send(%chunk{col}_{row}) : (memref<{CHUNK_BYTES}xi8>) -> ()"
            )
        else:
            out += [
                f"        %predecessors{group} = arith.constant {chain_len - 1} : i32",
                f"        func.call @r93_assemble_block(%block{col}_{row}, %chunk{col}_{row}, %predecessors{group}) : (memref<{BLOCK_PREFIX}xi8>, memref<{CHUNK_BYTES}xi8>, i32) -> ()",
                f"        %copies{group} = arith.constant 3 : index",
                f"        scf.for %copy{group} = %z to %copies{group} step %one {{",
                f"          %o{group} = aie.objectfifo.acquire @oc{col}_{row}(Produce, 1) : !aie.objectfifosubview<memref<{BLOCK_PREFIX}xi8>>",
                f"          %o{group}v = aie.objectfifo.subview.access %o{group}[0] : !aie.objectfifosubview<memref<{BLOCK_PREFIX}xi8>> -> memref<{BLOCK_PREFIX}xi8>",
                f"          func.call @r93_emit_block(%block{col}_{row}, %o{group}v) : (memref<{BLOCK_PREFIX}xi8>, memref<{BLOCK_PREFIX}xi8>) -> ()",
                f"          aie.objectfifo.release @oc{col}_{row}(Produce, 1)",
                "        }",
            ]
    out += [
        "      }",
        "      aie.end",
        f"    }} {{stack_size = {2048 if (col, row) in packers else 1024} : i32}}",
    ]

out.append(
    f"    aie.runtime_sequence(%X: memref<{X_BYTES}xi8>, %P: memref<{PARAM_TOTAL}xi8>, %O: memref<{R25_INPUT_BYTES}xi8>) {{"
)

for col, row in packers:
    block, _, _ = roles[(col, row)]
    token_base = block * 24
    m_macro = token_base // 96
    stripe = (token_base % 96) // 24
    offset = (stripe * 27 + m_macro * 9) * R25_BLOCK
    name = f"to{col}_{row}"
    out += [
        f"      %{name} = aiex.dma_configure_task_for @osh{col}_{row} {{",
        f"        aie.dma_bd(%O : memref<{R25_INPUT_BYTES}xi8>, {offset}, {3 * BLOCK_PREFIX}, [<size = 3, stride = {R25_BLOCK}>, <size = 3, stride = {3 * R25_BLOCK}>, <size = {BLOCK_PREFIX // 32}, stride = 32>, <size = 32, stride = 1>]) {{burst_length = 0 : i32}}",
        "        aie.end",
        "      } {issue_token = true, repeat_count = 2 : i32}",
        f"      aiex.dma_start_task(%{name})",
    ]

for col in range(COLS):
    name = f"tp{col}"
    out += [
        f"      %{name} = aiex.dma_configure_task_for @xsh{col} {{",
        f"        aie.dma_bd(%P : memref<{PARAM_TOTAL}xi8>, 0, {X_JOIN_BYTES}, [<size = {GROUPS}, stride = {CORE_ROWS * PARAM_RECORD}>, <size = {CORE_ROWS}, stride = {PARAM_RECORD}>, <size = {PARAM_RECORD // 32}, stride = 32>, <size = 32, stride = 1>]) {{burst_length = 0 : i32}}",
        "        aie.end",
        f"      }} {{issue_token = true, repeat_count = {GROUPS - 1} : i32}}",
        f"      aiex.dma_start_task(%{name})",
    ]
for col in range(COLS):
    out += [f"      aiex.dma_await_task(%tp{col})", f"      aiex.dma_free_task(%tp{col})"]

for group in range(GROUPS):
    for col in range(COLS):
        name = f"tx{group}_{col}"
        offset = col * CORE_ROWS * ROWS_PER_CORE * SOURCE_ROW_BYTES
        out += [
            f"      %{name} = aiex.dma_configure_task_for @xsh{col} {{",
            f"        aie.dma_bd(%X : memref<{X_BYTES}xi8>, {offset}, {X_JOIN_BYTES}, {x_dims()}) {{burst_length = 0 : i32}}",
            "        aie.end",
            f"      }} {{issue_token = true, repeat_count = {ROWS_PER_CORE - 1} : i32}}",
            f"      aiex.dma_start_task(%{name})",
        ]
    for col in range(COLS):
        out += [
            f"      aiex.dma_await_task(%tx{group}_{col})",
            f"      aiex.dma_free_task(%tx{group}_{col})",
        ]

for col, row in packers:
    name = f"to{col}_{row}"
    out += [f"      aiex.dma_await_task(%{name})", f"      aiex.dma_free_task(%{name})"]
out += ["    }", "  }", "}"]
print("\n".join(out))
