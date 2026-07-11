#!/usr/bin/env python3
"""Generate the all-core R28 QKV headnorm/RoPE direct-pack graph."""

COLS, ROWS = 8, 4
QUERY_GROUPS = 6
RAW_TILE, RAW_PAIR, RAW_JOIN = 4096, 8192, 32768
PARAMS = 2048
OUT_TILE, OUT_JOIN = 2048, 8192
RAW_Q_BYTES = QUERY_GROUPS * ROWS * RAW_JOIN
RAW_K_BYTES = 2 * ROWS * RAW_JOIN
RAW_V_BYTES = 2 * ROWS * RAW_JOIN
RAW_BYTES = RAW_Q_BYTES + RAW_K_BYTES + RAW_V_BYTES
Q_BYTES = 393216
KV_BYTES = 262144
Q_JOIN = 16384
KV_TILE = 16384
K_HALF = 8192
INF = 9223372036854775807


def dims2(count, stride, block):
    return (
        f"[<size = {count}, stride = {stride}>, "
        f"<size = {block // 512}, stride = 512>, "
        "<size = 512, stride = 1>]"
    )


out = ["module {", "  aie.device(npu2) {"]
for col in range(COLS):
    out += [f"    %shim{col} = aie.tile({col}, 0)", f"    %mt{col} = aie.tile({col}, 1)"]
    for row in range(ROWS):
        out += [
            f"    %c{col}_{row} = aie.tile({col}, {row + 2})",
            f'    %kinv{col}_{row} = aie.buffer(%c{col}_{row}) {{sym_name = "kinv{col}_{row}"}} : memref<8xf32>',
        ]

for row in range(ROWS):
    pairs = []
    for pair in range(COLS // 2):
        pairs.append(f"@rawpair{pair}_{row}")
        out.append(
            f"    aie.objectfifo @rawpair{pair}_{row}(%mt{row}, "
            f"{{%c{2 * pair}_{row}, %c{2 * pair + 1}_{row}}}, 1 : i32) : "
            f"!aie.objectfifo<memref<{RAW_PAIR}xi8>>"
        )
    offsets = ", ".join(str(pair * RAW_PAIR) for pair in range(COLS // 2))
    cores = ", ".join(f"%c{col}_{row}" for col in range(COLS))
    out += [
        f"    aie.objectfifo @rawsh{row}(%shim{row}, {{%mt{row}}}, 1 : i32) : !aie.objectfifo<memref<{RAW_JOIN}xi8>>",
        f"    aie.objectfifo.link [@rawsh{row}] -> [{', '.join(pairs)}] ([] [{offsets}])",
        f"    aie.objectfifo @psh{row}(%shim{row}, {{%mt{row}}}, 1 : i32) : !aie.objectfifo<memref<{PARAMS}xi8>>",
        f"    aie.objectfifo @pbc{row}(%mt{row}, {{{cores}}}, 1 : i32) : !aie.objectfifo<memref<{PARAMS}xi8>>",
        f"    aie.objectfifo.link [@psh{row}] -> [@pbc{row}] ([] [0])",
    ]

for col in range(COLS):
    producers = ", ".join(f"@oc{col}_{row}" for row in range(ROWS))
    offsets = ", ".join(str(row * OUT_TILE) for row in range(ROWS))
    for row in range(ROWS):
        out.append(
            f"    aie.objectfifo @oc{col}_{row}(%c{col}_{row}, {{%mt{col}}}, 1 : i32) : !aie.objectfifo<memref<{OUT_TILE}xi8>>"
        )
    out += [
        f"    aie.objectfifo @osh{col}(%mt{col}, {{%shim{col}}}, 1 : i32) : !aie.objectfifo<memref<{OUT_JOIN}xi8>>",
        f"    aie.objectfifo.link [{producers}] -> [@osh{col}] ([{offsets}] [])",
    ]

out += [
    f'    func.func private @r28_pack_q(memref<{RAW_PAIR}xi8>, memref<{PARAMS}xi8>, memref<{OUT_TILE}xi8>, i32) attributes {{link_with = "r28.o"}}',
    f'    func.func private @r28_pack_k(memref<{RAW_PAIR}xi8>, memref<{PARAMS}xi8>, memref<{OUT_TILE}xi8>, memref<8xf32>, i32) attributes {{link_with = "r28.o"}}',
    f'    func.func private @r28_pack_v(memref<{RAW_PAIR}xi8>, memref<{OUT_TILE}xi8>, i32) attributes {{link_with = "r28.o"}}',
]


def acquire_raw(col, row, name, indent):
    return [
        f"{indent}%raw{name} = aie.objectfifo.acquire @rawpair{col // 2}_{row}(Consume, 1) : !aie.objectfifosubview<memref<{RAW_PAIR}xi8>>",
        f"{indent}%raw{name}v = aie.objectfifo.subview.access %raw{name}[0] : !aie.objectfifosubview<memref<{RAW_PAIR}xi8>> -> memref<{RAW_PAIR}xi8>",
    ]


def acquire_output(col, row, name, indent):
    return [
        f"{indent}%{name} = aie.objectfifo.acquire @oc{col}_{row}(Produce, 1) : !aie.objectfifosubview<memref<{OUT_TILE}xi8>>",
        f"{indent}%{name}v = aie.objectfifo.subview.access %{name}[0] : !aie.objectfifosubview<memref<{OUT_TILE}xi8>> -> memref<{OUT_TILE}xi8>",
    ]


for col in range(COLS):
    for row in range(ROWS):
        lines = [
            f"    %core{col}_{row} = aie.core(%c{col}_{row}) {{",
            "      %z = arith.constant 0 : index",
            f"      %inf = arith.constant {INF} : index",
            "      %one = arith.constant 1 : index",
            f"      %lane = arith.constant {col % 2} : i32",
            "      scf.for %outer = %z to %inf step %one {",
            f"        %params = aie.objectfifo.acquire @pbc{row}(Consume, 1) : !aie.objectfifosubview<memref<{PARAMS}xi8>>",
            f"        %paramsv = aie.objectfifo.subview.access %params[0] : !aie.objectfifosubview<memref<{PARAMS}xi8>> -> memref<{PARAMS}xi8>",
        ]
        for group in range(QUERY_GROUPS):
            lines += acquire_raw(col, row, f"q{group}", "        ")
            lines += acquire_output(col, row, f"qo{group}", "        ")
            lines += [
                f"        func.call @r28_pack_q(%rawq{group}v, %paramsv, %qo{group}v, %lane) : (memref<{RAW_PAIR}xi8>, memref<{PARAMS}xi8>, memref<{OUT_TILE}xi8>, i32) -> ()",
                f"        aie.objectfifo.release @rawpair{col // 2}_{row}(Consume, 1)",
                f"        aie.objectfifo.release @oc{col}_{row}(Produce, 1)",
        ]
        for wave in range(2):
            lines += acquire_raw(col, row, f"k{wave}", "        ")
            if col % 2 == 0:
                for half in range(2):
                    lines += acquire_output(col, row, f"ko{wave}_{half}", "        ")
                    lines += [
                        f"        %kh{wave}_{half} = arith.constant {half} : i32",
                        f"        func.call @r28_pack_k(%rawk{wave}v, %paramsv, %ko{wave}_{half}v, %kinv{col}_{row}, %kh{wave}_{half}) : (memref<{RAW_PAIR}xi8>, memref<{PARAMS}xi8>, memref<{OUT_TILE}xi8>, memref<8xf32>, i32) -> ()",
                        f"        aie.objectfifo.release @oc{col}_{row}(Produce, 1)",
                    ]
            lines += [f"        aie.objectfifo.release @rawpair{col // 2}_{row}(Consume, 1)"]
        for wave in range(2):
            lines += acquire_raw(col, row, f"v{wave}", "        ")
            if col % 2 == 0:
                for half in range(2):
                    lines += acquire_output(col, row, f"vo{wave}_{half}", "        ")
                    lines += [
                        f"        %vh{wave}_{half} = arith.constant {half} : i32",
                        f"        func.call @r28_pack_v(%rawv{wave}v, %vo{wave}_{half}v, %vh{wave}_{half}) : (memref<{RAW_PAIR}xi8>, memref<{OUT_TILE}xi8>, i32) -> ()",
                        f"        aie.objectfifo.release @oc{col}_{row}(Produce, 1)",
                    ]
            lines += [f"        aie.objectfifo.release @rawpair{col // 2}_{row}(Consume, 1)"]
        lines += [
            f"        aie.objectfifo.release @pbc{row}(Consume, 1)",
            "      }",
            "      aie.end",
            "    } {stack_size = 4096 : i32}",
        ]
        out += lines

out.append(
    f"    aie.runtime_sequence(%R: memref<{RAW_BYTES}xi8>, %P: memref<{PARAMS}xi8>, %Q: memref<{Q_BYTES}xi8>, %KV: memref<{KV_BYTES}xi8>) {{"
)
for row in range(ROWS):
    out += [
        f"      %tp{row} = aiex.dma_configure_task_for @psh{row} {{",
        f"        aie.dma_bd(%P : memref<{PARAMS}xi8>, 0, {PARAMS}, {dims2(1, 0, PARAMS)}) {{burst_length = 0 : i32}}",
        "        aie.end",
        "      } {issue_token = true}",
        f"      aiex.dma_start_task(%tp{row})",
    ]

for group in range(QUERY_GROUPS):
    for col in range(COLS):
        offset = group * Q_JOIN + col * OUT_TILE
        out += [
            f"      %tqo{group}_{col} = aiex.dma_configure_task_for @osh{col} {{",
            f"        aie.dma_bd(%Q : memref<{Q_BYTES}xi8>, {offset}, {ROWS * OUT_TILE}, {dims2(ROWS, QUERY_GROUPS * Q_JOIN, OUT_TILE)}) {{burst_length = 0 : i32}}",
            "        aie.end",
            "      } {issue_token = true}",
            f"      aiex.dma_start_task(%tqo{group}_{col})",
        ]
    for row in range(ROWS):
        offset = (group * ROWS + row) * RAW_JOIN
        out += [
            f"      %tqi{group}_{row} = aiex.dma_configure_task_for @rawsh{row} {{",
            f"        aie.dma_bd(%R : memref<{RAW_BYTES}xi8>, {offset}, {RAW_JOIN}, {dims2(1, 0, RAW_JOIN)}) {{burst_length = 0 : i32}}",
            "        aie.end",
            "      } {issue_token = true}",
            f"      aiex.dma_start_task(%tqi{group}_{row})",
        ]
    for row in range(ROWS):
        out += [f"      aiex.dma_await_task(%tqi{group}_{row})", f"      aiex.dma_free_task(%tqi{group}_{row})"]
    for col in range(COLS):
        out += [f"      aiex.dma_await_task(%tqo{group}_{col})", f"      aiex.dma_free_task(%tqo{group}_{col})"]

for phase, phase_offset in (("k", RAW_Q_BYTES), ("v", RAW_Q_BYTES + RAW_K_BYTES)):
    for wave in range(2):
        for col in range(0, COLS, 2):
            pair = col // 2
            group = wave * 16 + pair
            block, key_tile = divmod(group, 2)
            for half in range(2):
                if phase == "k":
                    offset = block * KV_TILE + key_tile * 4096 + half * OUT_TILE
                    dimensions = dims2(ROWS, 2 * KV_TILE, OUT_TILE)
                else:
                    offset = (
                        block * KV_TILE
                        + K_HALF
                        + (half * 16 * 2 + key_tile) * 128
                    )
                    dimensions = (
                        f"[<size = {ROWS}, stride = {2 * KV_TILE}>, "
                        "<size = 16, stride = 256>, <size = 128, stride = 1>]"
                    )
                name = f"t{phase}o{wave}_{col}_{half}"
                out += [
                    f"      %{name} = aiex.dma_configure_task_for @osh{col} {{",
                    f"        aie.dma_bd(%KV : memref<{KV_BYTES}xi8>, {offset}, {ROWS * OUT_TILE}, {dimensions}) {{burst_length = 0 : i32}}",
                    "        aie.end",
                    "      } {issue_token = true}",
                    f"      aiex.dma_start_task(%{name})",
                ]
        for row in range(ROWS):
            offset = phase_offset + (wave * ROWS + row) * RAW_JOIN
            name = f"t{phase}i{wave}_{row}"
            out += [
                f"      %{name} = aiex.dma_configure_task_for @rawsh{row} {{",
                f"        aie.dma_bd(%R : memref<{RAW_BYTES}xi8>, {offset}, {RAW_JOIN}, {dims2(1, 0, RAW_JOIN)}) {{burst_length = 0 : i32}}",
                "        aie.end",
                "      } {issue_token = true}",
                f"      aiex.dma_start_task(%{name})",
            ]
        for row in range(ROWS):
            name = f"t{phase}i{wave}_{row}"
            out += [f"      aiex.dma_await_task(%{name})", f"      aiex.dma_free_task(%{name})"]
        for col in range(0, COLS, 2):
            for half in range(2):
                name = f"t{phase}o{wave}_{col}_{half}"
                out += [f"      aiex.dma_await_task(%{name})", f"      aiex.dma_free_task(%{name})"]

for row in range(ROWS):
    out += [f"      aiex.dma_await_task(%tp{row})", f"      aiex.dma_free_task(%tp{row})"]
out += ["    }", "  }", "}"]
print("\n".join(out))
