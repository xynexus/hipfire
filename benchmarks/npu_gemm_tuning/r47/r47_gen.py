#!/usr/bin/env python3
"""R47 compensated completed-state to resident R34 activation preparation."""

COLS, CORE_ROWS = 8, 4
ROWS_PER_CORE, GROUPS, HIDDEN = 8, 3, 768
GROUP = 256
PAD_M = 288
COMPLETED_ROW = 2 * HIDDEN * 2
COMPLETED_BYTES = PAD_M * COMPLETED_ROW
X_ROW_BYTES = COMPLETED_ROW
X_JOIN_BYTES = CORE_ROWS * X_ROW_BYTES
PARAM_BYTES = 2 * GROUP * 4 + 2 * GROUP * 2
PARAM_RECORD = X_ROW_BYTES
PARAM_TOTAL = GROUPS * CORE_ROWS * PARAM_RECORD
CHUNK_BYTES = ROWS_PER_CORE * GROUP + ROWS_PER_CORE * 4
BLOCK_PREFIX = 3 * ROWS_PER_CORE * GROUP + 3 * ROWS_PER_CORE * 4
R34_BLOCK = 16384
R34_BLOCKS = 4 * 45
R34_INPUT_BYTES = R34_BLOCKS * R34_BLOCK
INF = 9223372036854775807


def x_dims():
    return (
        f"[<size = {ROWS_PER_CORE}, stride = {COMPLETED_ROW}>, "
        f"<size = {CORE_ROWS}, stride = {ROWS_PER_CORE * COMPLETED_ROW}>, "
        f"<size = {COMPLETED_ROW // 32}, stride = 32>, "
        f"<size = 32, stride = 1>]"
    )


def linear_dims(size):
    return f"[<size = {size // 32}, stride = 32>, <size = 32, stride = 1>]"


cores = [(col, row) for col in range(COLS) for row in range(CORE_ROWS)]
chains = [cores[index : index + 3] for index in range(0, len(cores), 3)]
roles = {}
packers = []
for block, chain in enumerate(chains):
    for lm, core in enumerate(chain):
        roles[core] = (block, lm, len(chain))
    packers.append(chain[-1])

out = ["module {", "  aie.device(npu2) {"]
for col in range(COLS):
    out += [f"    %shim{col} = aie.tile({col}, 0)", f"    %mt{col} = aie.tile({col}, 1)"]
    for row in range(CORE_ROWS):
        out += [
            f"    %c{col}_{row} = aie.tile({col}, {row + 2})",
            f'    %scratch{col}_{row} = aie.buffer(%c{col}_{row}) {{sym_name = "scratch{col}_{row}"}} : memref<256xf32>',
            f'    %sum{col}_{row} = aie.buffer(%c{col}_{row}) {{sym_name = "sum{col}_{row}"}} : memref<8xf32>',
            f'    %chunk{col}_{row} = aie.buffer(%c{col}_{row}) {{sym_name = "chunk{col}_{row}"}} : memref<{CHUNK_BYTES}xi8>',
            *[
                f'    %param{group}_{col}_{row} = aie.buffer(%c{col}_{row}) {{sym_name = "param{group}_{col}_{row}"}} : memref<{PARAM_BYTES}xi8>'
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
    f'    func.func private @r47_accumulate_row(memref<{X_ROW_BYTES}xi8>, memref<8xf32>, i32) attributes {{link_with = "r47.o"}}',
    f'    func.func private @r47_pack_row(memref<{X_ROW_BYTES}xi8>, memref<{PARAM_BYTES}xi8>, memref<{CHUNK_BYTES}xi8>, memref<256xf32>, memref<8xf32>, i32, i32) attributes {{link_with = "r47.o"}}',
    f'    func.func private @r47_copy_param(memref<{X_ROW_BYTES}xi8>, memref<{PARAM_BYTES}xi8>, i32) attributes {{link_with = "r47.o"}}',
    f'    func.func private @r47_send_chunk(memref<{CHUNK_BYTES}xi8>) attributes {{link_with = "r47.o"}}',
    f'    func.func private @r47_relay_then_send(memref<{CHUNK_BYTES}xi8>) attributes {{link_with = "r47.o"}}',
    f'    func.func private @r47_assemble_block(memref<{BLOCK_PREFIX}xi8>, memref<{CHUNK_BYTES}xi8>, i32) attributes {{link_with = "r47.o"}}',
    f'    func.func private @r47_emit_block(memref<{BLOCK_PREFIX}xi8>, memref<{BLOCK_PREFIX}xi8>) attributes {{link_with = "r47.o"}}',
]

for col, row in cores:
    block, lm, chain_len = roles[(col, row)]
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
            f"        %group{group} = arith.constant {group} : i32",
            f"        func.call @r47_copy_param(%pload{group}v, %param{group}_{col}_{row}, %group{group}) : (memref<{X_ROW_BYTES}xi8>, memref<{PARAM_BYTES}xi8>, i32) -> ()",
            f"        aie.objectfifo.release @xc{col}_{row}(Consume, 1)",
        ]
    out += [
        "        scf.for %row = %z to %rows step %one {",
        f"          %xsum = aie.objectfifo.acquire @xc{col}_{row}(Consume, 1) : !aie.objectfifosubview<memref<{X_ROW_BYTES}xi8>>",
        f"          %xsumv = aie.objectfifo.subview.access %xsum[0] : !aie.objectfifosubview<memref<{X_ROW_BYTES}xi8>> -> memref<{X_ROW_BYTES}xi8>",
        "          %rowi = arith.index_cast %row : index to i32",
        f"          func.call @r47_accumulate_row(%xsumv, %sum{col}_{row}, %rowi) : (memref<{X_ROW_BYTES}xi8>, memref<8xf32>, i32) -> ()",
        f"          aie.objectfifo.release @xc{col}_{row}(Consume, 1)",
        "        }",
    ]
    for group in range(GROUPS):
        out += [
            "        scf.for %row = %z to %rows step %one {",
            f"          %x{group} = aie.objectfifo.acquire @xc{col}_{row}(Consume, 1) : !aie.objectfifosubview<memref<{X_ROW_BYTES}xi8>>",
            f"          %x{group}v = aie.objectfifo.subview.access %x{group}[0] : !aie.objectfifosubview<memref<{X_ROW_BYTES}xi8>> -> memref<{X_ROW_BYTES}xi8>",
            "          %rowi = arith.index_cast %row : index to i32",
            f"          func.call @r47_pack_row(%x{group}v, %param{group}_{col}_{row}, %chunk{col}_{row}, %scratch{col}_{row}, %sum{col}_{row}, %rowi, %group{group}) : (memref<{X_ROW_BYTES}xi8>, memref<{PARAM_BYTES}xi8>, memref<{CHUNK_BYTES}xi8>, memref<256xf32>, memref<8xf32>, i32, i32) -> ()",
            f"          aie.objectfifo.release @xc{col}_{row}(Consume, 1)",
            "        }",
        ]
        if lm == 0 and chain_len > 1:
            out.append(
                f"        func.call @r47_send_chunk(%chunk{col}_{row}) : (memref<{CHUNK_BYTES}xi8>) -> ()"
            )
        elif lm < chain_len - 1:
            out.append(
                f"        func.call @r47_relay_then_send(%chunk{col}_{row}) : (memref<{CHUNK_BYTES}xi8>) -> ()"
            )
        else:
            out += [
                f"        %predecessors{group} = arith.constant {chain_len - 1} : i32",
                f"        func.call @r47_assemble_block(%block{col}_{row}, %chunk{col}_{row}, %predecessors{group}) : (memref<{BLOCK_PREFIX}xi8>, memref<{CHUNK_BYTES}xi8>, i32) -> ()",
                f"        %copies{group} = arith.constant 5 : index",
                f"        scf.for %copy{group} = %z to %copies{group} step %one {{",
                f"          %o{group} = aie.objectfifo.acquire @oc{col}_{row}(Produce, 1) : !aie.objectfifosubview<memref<{BLOCK_PREFIX}xi8>>",
                f"          %o{group}v = aie.objectfifo.subview.access %o{group}[0] : !aie.objectfifosubview<memref<{BLOCK_PREFIX}xi8>> -> memref<{BLOCK_PREFIX}xi8>",
                f"          func.call @r47_emit_block(%block{col}_{row}, %o{group}v) : (memref<{BLOCK_PREFIX}xi8>, memref<{BLOCK_PREFIX}xi8>) -> ()",
                f"          aie.objectfifo.release @oc{col}_{row}(Produce, 1)",
                "        }",
            ]
    out += [
        "      }",
        "      aie.end",
        f"    }} {{stack_size = {2048 if (col, row) in packers else 1024} : i32}}",
    ]

out.append(
    f"    aie.runtime_sequence(%X: memref<{COMPLETED_BYTES}xi8>, %P: memref<{PARAM_TOTAL}xi8>, %O: memref<{R34_INPUT_BYTES}xi8>) {{"
)

for col, row in packers:
    block, _, _ = roles[(col, row)]
    token_base = block * 24
    m_macro = token_base // 96
    stripe = (token_base % 96) // 24
    offset = (stripe * 45 + m_macro * 15) * R34_BLOCK
    name = f"to{col}_{row}"
    out += [
        f"      %{name} = aiex.dma_configure_task_for @osh{col}_{row} {{",
        f"        aie.dma_bd(%O : memref<{R34_INPUT_BYTES}xi8>, {offset}, {5 * BLOCK_PREFIX}, [<size = 3, stride = {R34_BLOCK}>, <size = 5, stride = {3 * R34_BLOCK}>, <size = {BLOCK_PREFIX // 32}, stride = 32>, <size = 32, stride = 1>]) {{burst_length = 0 : i32}}",
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

for pass_index in range(1 + GROUPS):
    for col in range(COLS):
        name = f"tx{pass_index}_{col}"
        offset = col * CORE_ROWS * ROWS_PER_CORE * COMPLETED_ROW
        out += [
            f"      %{name} = aiex.dma_configure_task_for @xsh{col} {{",
            f"        aie.dma_bd(%X : memref<{COMPLETED_BYTES}xi8>, {offset}, {X_JOIN_BYTES}, {x_dims()}) {{burst_length = 0 : i32}}",
            "        aie.end",
            f"      }} {{issue_token = true, repeat_count = {ROWS_PER_CORE - 1} : i32}}",
            f"      aiex.dma_start_task(%{name})",
        ]
    for col in range(COLS):
        out += [
            f"      aiex.dma_await_task(%tx{pass_index}_{col})",
            f"      aiex.dma_free_task(%tx{pass_index}_{col})",
        ]

for col, row in packers:
    name = f"to{col}_{row}"
    out += [f"      aiex.dma_await_task(%{name})", f"      aiex.dma_free_task(%{name})"]
out += ["    }", "  }", "}"]
print("\n".join(out))
