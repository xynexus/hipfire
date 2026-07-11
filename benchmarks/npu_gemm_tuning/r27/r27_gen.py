#!/usr/bin/env python3
"""Generate the 32-core R27 M256 bidirectional attention graph."""

COLS, ROWS = 8, 4
QUERY_GROUPS, KEY_BLOCKS = 6, 16
Q_TILE = 4 * 256 * 2
Q_PAIR = 2 * Q_TILE
Q_JOIN = COLS * Q_TILE
KV_TILE = 2 * 16 * 256 * 2
OUT_TILE = Q_TILE
OUT_JOIN = ROWS * OUT_TILE
ACC = 4 * 256
STATS = 8
Q_BYTES = ROWS * QUERY_GROUPS * Q_JOIN
KV_BYTES = KEY_BLOCKS * KV_TILE
O_BYTES = COLS * QUERY_GROUPS * OUT_JOIN
INF = 9223372036854775807


def blocks(count, block):
    return (
        f"[<size = {count}, stride = {block}>, "
        f"<size = {block // 512}, stride = 512>, "
        "<size = 512, stride = 1>]"
    )


out = ["module {", "  aie.device(npu2) {"]
for col in range(COLS):
    out += [f"    %shim{col} = aie.tile({col}, 0)", f"    %mt{col} = aie.tile({col}, 1)"]
for row in range(ROWS):
    for col in range(COLS):
        out += [
            f"    %c{col}_{row} = aie.tile({col}, {row + 2})",
            f'    %acc{col}_{row} = aie.buffer(%c{col}_{row}) {{sym_name = "acc{col}_{row}"}} : memref<{ACC}xf32>',
            f'    %stats{col}_{row} = aie.buffer(%c{col}_{row}) {{sym_name = "stats{col}_{row}"}} : memref<{STATS}xf32>',
        ]

for row in range(ROWS):
    cores = ", ".join(f"%c{col}_{row}" for col in range(COLS))
    q_consumers = ", ".join(f"@qpair{pair}_{row}" for pair in range(COLS // 2))
    q_offsets = ", ".join(str(pair * Q_PAIR) for pair in range(COLS // 2))
    out += [
        f"    aie.objectfifo @qsh{row}(%shim{row}, {{%mt{row}}}, 1 : i32) : !aie.objectfifo<memref<{Q_JOIN}xi8>>",
    ]
    for pair in range(COLS // 2):
        out.append(
            f"    aie.objectfifo @qpair{pair}_{row}(%mt{row}, {{%c{2 * pair}_{row}, %c{2 * pair + 1}_{row}}}, 1 : i32) : !aie.objectfifo<memref<{Q_PAIR}xi8>>"
        )
    out += [
        f"    aie.objectfifo.link [@qsh{row}] -> [{q_consumers}] ([] [{q_offsets}])",
        f"    aie.objectfifo @kvsh{row}(%shim{row}, {{%mt{row}}}, 1 : i32) : !aie.objectfifo<memref<{KV_TILE}xi8>>",
        f"    aie.objectfifo @kv{row}(%mt{row}, {{{cores}}}, 1 : i32) : !aie.objectfifo<memref<{KV_TILE}xi8>>",
        f"    aie.objectfifo.link [@kvsh{row}] -> [@kv{row}] ([] [0])",
    ]

for col in range(COLS):
    o_producers = ", ".join(f"@o{col}_{row}" for row in range(ROWS))
    o_offsets = ", ".join(str(row * OUT_TILE) for row in range(ROWS))
    for row in range(ROWS):
        out.append(
            f"    aie.objectfifo @o{col}_{row}(%c{col}_{row}, {{%mt{col}}}, 1 : i32) : !aie.objectfifo<memref<{OUT_TILE}xi8>>"
        )
    out += [
        f"    aie.objectfifo @osh{col}(%mt{col}, {{%shim{col}}}, 1 : i32) : !aie.objectfifo<memref<{OUT_JOIN}xi8>>",
        f"    aie.objectfifo.link [{o_producers}] -> [@osh{col}] ([{o_offsets}] [])",
    ]

out += [
    f'    func.func private @r27_attention_init(memref<{ACC}xf32>, memref<{STATS}xf32>) attributes {{link_with = "r27.o"}}',
    f'    func.func private @r27_attention_block(memref<{Q_PAIR}xi8>, memref<{KV_TILE}xi8>, memref<{ACC}xf32>, memref<{STATS}xf32>, i32) attributes {{link_with = "r27.o"}}',
    f'    func.func private @r27_attention_finish(memref<{ACC}xf32>, memref<{STATS}xf32>, memref<{OUT_TILE}xi8>) attributes {{link_with = "r27.o"}}',
]

for row in range(ROWS):
    for col in range(COLS):
        out += [
            f"    %core{col}_{row} = aie.core(%c{col}_{row}) {{",
            "      %z = arith.constant 0 : index",
            f"      %inf = arith.constant {INF} : index",
            "      %one = arith.constant 1 : index",
            f"      %groups = arith.constant {QUERY_GROUPS} : index",
            f"      %blocks = arith.constant {KEY_BLOCKS} : index",
            "      scf.for %outer = %z to %inf step %one {",
            "        scf.for %group = %z to %groups step %one {",
            f"          %q = aie.objectfifo.acquire @qpair{col // 2}_{row}(Consume, 1) : !aie.objectfifosubview<memref<{Q_PAIR}xi8>>",
            f"          %qv = aie.objectfifo.subview.access %q[0] : !aie.objectfifosubview<memref<{Q_PAIR}xi8>> -> memref<{Q_PAIR}xi8>",
            f"          %pair_lane = arith.constant {col % 2} : i32",
            f"          %o = aie.objectfifo.acquire @o{col}_{row}(Produce, 1) : !aie.objectfifosubview<memref<{OUT_TILE}xi8>>",
            f"          %ov = aie.objectfifo.subview.access %o[0] : !aie.objectfifosubview<memref<{OUT_TILE}xi8>> -> memref<{OUT_TILE}xi8>",
            f"          func.call @r27_attention_init(%acc{col}_{row}, %stats{col}_{row}) : (memref<{ACC}xf32>, memref<{STATS}xf32>) -> ()",
            "          scf.for %block = %z to %blocks step %one {",
            f"            %kv = aie.objectfifo.acquire @kv{row}(Consume, 1) : !aie.objectfifosubview<memref<{KV_TILE}xi8>>",
            f"            %kvv = aie.objectfifo.subview.access %kv[0] : !aie.objectfifosubview<memref<{KV_TILE}xi8>> -> memref<{KV_TILE}xi8>",
            f"            func.call @r27_attention_block(%qv, %kvv, %acc{col}_{row}, %stats{col}_{row}, %pair_lane) : (memref<{Q_PAIR}xi8>, memref<{KV_TILE}xi8>, memref<{ACC}xf32>, memref<{STATS}xf32>, i32) -> ()",
            f"            aie.objectfifo.release @kv{row}(Consume, 1)",
            "          }",
            f"          func.call @r27_attention_finish(%acc{col}_{row}, %stats{col}_{row}, %ov) : (memref<{ACC}xf32>, memref<{STATS}xf32>, memref<{OUT_TILE}xi8>) -> ()",
            f"          aie.objectfifo.release @qpair{col // 2}_{row}(Consume, 1)",
            f"          aie.objectfifo.release @o{col}_{row}(Produce, 1)",
            "        }",
            "      }",
            "      aie.end",
            "    } {stack_size = 4096 : i32}",
        ]

out.append(
    f"    aie.runtime_sequence(%Q: memref<{Q_BYTES}xi8>, %KV: memref<{KV_BYTES}xi8>, %O: memref<{O_BYTES}xi8>) {{"
)
for row in range(ROWS):
    out += [
        f"      %tq{row} = aiex.dma_configure_task_for @qsh{row} {{",
        f"        aie.dma_bd(%Q : memref<{Q_BYTES}xi8>, {row * QUERY_GROUPS * Q_JOIN}, {QUERY_GROUPS * Q_JOIN}, {blocks(QUERY_GROUPS, Q_JOIN)}) {{burst_length = 0 : i32}}",
        "        aie.end",
        "      } {issue_token = true}",
        f"      aiex.dma_start_task(%tq{row})",
        f"      %tkv{row} = aiex.dma_configure_task_for @kvsh{row} {{",
        f"        aie.dma_bd(%KV : memref<{KV_BYTES}xi8>, 0, {KV_BYTES}, {blocks(KEY_BLOCKS, KV_TILE)}) {{burst_length = 0 : i32}}",
        "        aie.end",
        f"      }} {{issue_token = true, repeat_count = {QUERY_GROUPS - 1} : i32}}",
        f"      aiex.dma_start_task(%tkv{row})",
    ]
for col in range(COLS):
    out += [
        f"      %to{col} = aiex.dma_configure_task_for @osh{col} {{",
        f"        aie.dma_bd(%O : memref<{O_BYTES}xi8>, {col * QUERY_GROUPS * OUT_JOIN}, {QUERY_GROUPS * OUT_JOIN}, {blocks(QUERY_GROUPS, OUT_JOIN)}) {{burst_length = 0 : i32}}",
        "        aie.end",
        "      } {issue_token = true}",
        f"      aiex.dma_start_task(%to{col})",
    ]
for row in range(ROWS):
    for task in (f"tq{row}", f"tkv{row}"):
        out += [f"      aiex.dma_await_task(%{task})", f"      aiex.dma_free_task(%{task})"]
for col in range(COLS):
    out += [f"      aiex.dma_await_task(%to{col})", f"      aiex.dma_free_task(%to{col})"]
out += ["    }", "  }", "}"]
print("\n".join(out))
