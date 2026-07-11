#!/usr/bin/env python3
"""Generate one-command W8 QKV projection -> R28 pack for AIE2P."""

COLS, ROWS = 8, 4
GROUPS, M_MACROS, N_MACROS = 3, 3, 5
OUTBLOCKS = M_MACROS * N_MACROS
A_BLOCK, W_BLOCK = 10240, 16384
ACC_ELEMS = 768
INBLOCKS = GROUPS * OUTBLOCKS
A_BYTES = ROWS * INBLOCKS * A_BLOCK
W_BYTES = COLS * INBLOCKS * W_BLOCK

QUERY_GROUPS = 6
PAIR, PAIRS_PER_ROLE, ROLES = 10240, 48, 5
R_BYTES = ROLES * PAIRS_PER_ROLE * PAIR
OUT_TILE, OUT_JOIN = 2048, 8192
Q_BYTES, KV_BYTES = 393216, 262144
Q_JOIN, KV_TILE, K_HALF = 16384, 16384, 8192
INF = 9223372036854775807


def dims(count, block):
    return (
        f"[<size = {count}, stride = {block}>, "
        f"<size = {block // 512}, stride = 512>, "
        "<size = 512, stride = 1>]"
    )


def strided_dims(count, stride, block):
    return (
        f"[<size = {count}, stride = {stride}>, "
        f"<size = {block // 512}, stride = 512>, "
        "<size = 512, stride = 1>]"
    )


def projection_output_dims():
    # Each core emits a padded [32 tokens,32 columns] tile. Four joined cores
    # map to sixteen eight-token records; every fourth record is token padding.
    return (
        f"[<size = 4, stride = {4 * PAIR}>, "
        f"<size = 4, stride = {PAIR}>, "
        "<size = 8, stride = 512>, "
        "<size = 64, stride = 1>]"
    )


out = ["module {", "  aie.device(npu2) {"]
for col in range(COLS):
    out += [f"    %shim{col} = aie.tile({col}, 0)", f"    %mt{col} = aie.tile({col}, 1)"]
    for row in range(ROWS):
        out += [
            f"    %c{col}_{row} = aie.tile({col}, {row + 2})",
            f'    %acc{col}_{row} = aie.buffer(%c{col}_{row}) {{sym_name = "acc{col}_{row}"}} : memref<{ACC_ELEMS}xf32>',
            f'    %kinv{col}_{row} = aie.buffer(%c{col}_{row}) {{sym_name = "kinv{col}_{row}"}} : memref<8xf32>',
        ]

for col in range(COLS):
    cores = ", ".join(f"%c{col}_{row}" for row in range(ROWS))
    out += [
        f"    aie.objectfifo @wsh{col}(%shim{col}, {{%mt{col}}}, 1 : i32) : !aie.objectfifo<memref<{W_BLOCK}xi8>>",
        f"    aie.objectfifo @wbc{col}(%mt{col}, {{{cores}}}, 1 : i32) : !aie.objectfifo<memref<{W_BLOCK}xi8>>",
        f"    aie.objectfifo.link [@wsh{col}] -> [@wbc{col}] ([] [0])",
    ]

for row in range(ROWS):
    cores = ", ".join(f"%c{col}_{row}" for col in range(COLS))
    out += [
        f"    aie.objectfifo @ash{row}(%shim{row}, {{%mt{row}}}, 1 : i32) : !aie.objectfifo<memref<{A_BLOCK}xi8>>",
        f"    aie.objectfifo @abc{row}(%mt{row}, {{{cores}}}, 1 : i32) : !aie.objectfifo<memref<{A_BLOCK}xi8>>",
        f"    aie.objectfifo.link [@ash{row}] -> [@abc{row}] ([] [0])",
    ]

for col in range(COLS):
    attention_producers = ", ".join(f"@oc{col}_{row}" for row in range(ROWS))
    attention_offsets = ", ".join(str(row * OUT_TILE) for row in range(ROWS))
    for row in range(ROWS):
        out.append(
            f"    aie.objectfifo @oc{col}_{row}(%c{col}_{row}, {{%mt{col}}}, 1 : i32) : !aie.objectfifo<memref<{OUT_TILE}xi8>>"
        )
    out += [
        f"    aie.objectfifo @osh{col}(%mt{col}, {{%shim{col}}}, 1 : i32) : !aie.objectfifo<memref<{OUT_JOIN}xi8>>",
        f"    aie.objectfifo.link [{attention_producers}] -> [@osh{col}] ([{attention_offsets}] [])",
    ]

out += [
    f'    func.func private @r29_w8_projection_init(memref<{A_BLOCK}xi8>, memref<{W_BLOCK}xi8>, memref<{ACC_ELEMS}xf32>) attributes {{link_with = "r29.o"}}',
    f'    func.func private @r29_w8_projection_accum(memref<{A_BLOCK}xi8>, memref<{W_BLOCK}xi8>, memref<{ACC_ELEMS}xf32>) attributes {{link_with = "r29.o"}}',
    f'    func.func private @r29_w8_projection_finish(memref<{ACC_ELEMS}xf32>, memref<{OUT_TILE}xi8>) attributes {{link_with = "r29.o"}}',
    f'    func.func private @r29_pack_q(memref<{PAIR}xi8>, memref<{OUT_TILE}xi8>, i32) attributes {{link_with = "r29.o"}}',
    f'    func.func private @r29_pack_k(memref<{PAIR}xi8>, memref<{OUT_TILE}xi8>, memref<8xf32>, i32) attributes {{link_with = "r29.o"}}',
    f'    func.func private @r29_pack_v(memref<{PAIR}xi8>, memref<{OUT_TILE}xi8>, i32) attributes {{link_with = "r29.o"}}',
]


def acquire_a(row, name, indent="        "):
    return [
        f"{indent}%a{name} = aie.objectfifo.acquire @abc{row}(Consume, 1) : !aie.objectfifosubview<memref<{A_BLOCK}xi8>>",
        f"{indent}%a{name}v = aie.objectfifo.subview.access %a{name}[0] : !aie.objectfifosubview<memref<{A_BLOCK}xi8>> -> memref<{A_BLOCK}xi8>",
    ]


def acquire_w(col, name, indent="        "):
    return [
        f"{indent}%w{name} = aie.objectfifo.acquire @wbc{col}(Consume, 1) : !aie.objectfifosubview<memref<{W_BLOCK}xi8>>",
        f"{indent}%w{name}v = aie.objectfifo.subview.access %w{name}[0] : !aie.objectfifosubview<memref<{W_BLOCK}xi8>> -> memref<{W_BLOCK}xi8>",
    ]


def acquire_out(col, row, name, indent="        "):
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
            f"      %groups = arith.constant {GROUPS} : index",
            f"      %outblocks = arith.constant {OUTBLOCKS} : index",
            f"      %qgroups = arith.constant {QUERY_GROUPS} : index",
            "      %waves = arith.constant 2 : index",
            f"      %lane = arith.constant {col % 2} : i32",
            "      %h0 = arith.constant 0 : i32",
            "      %h1 = arith.constant 1 : i32",
            "      scf.for %outer = %z to %inf step %one {",
            "        scf.for %block = %z to %outblocks step %one {",
        ]
        lines += acquire_a(row, "p0", "          ")
        lines += acquire_w(col, "p0", "          ")
        lines += [
            f"          func.call @r29_w8_projection_init(%ap0v, %wp0v, %acc{col}_{row}) : (memref<{A_BLOCK}xi8>, memref<{W_BLOCK}xi8>, memref<{ACC_ELEMS}xf32>) -> ()",
            f"          aie.objectfifo.release @abc{row}(Consume, 1)",
            f"          aie.objectfifo.release @wbc{col}(Consume, 1)",
            "          scf.for %group = %one to %groups step %one {",
        ]
        lines += acquire_a(row, "pa", "            ")
        lines += acquire_w(col, "pa", "            ")
        lines += [
            f"            func.call @r29_w8_projection_accum(%apav, %wpav, %acc{col}_{row}) : (memref<{A_BLOCK}xi8>, memref<{W_BLOCK}xi8>, memref<{ACC_ELEMS}xf32>) -> ()",
            f"            aie.objectfifo.release @abc{row}(Consume, 1)",
            f"            aie.objectfifo.release @wbc{col}(Consume, 1)",
            "          }",
        ]
        lines += acquire_out(col, row, "po", "          ")
        lines += [
            f"          func.call @r29_w8_projection_finish(%acc{col}_{row}, %pov) : (memref<{ACC_ELEMS}xf32>, memref<{OUT_TILE}xi8>) -> ()",
            f"          aie.objectfifo.release @oc{col}_{row}(Produce, 1)",
            "        }",
            "        scf.for %qgroup = %z to %qgroups step %one {",
        ]
        lines += acquire_out(col, row, "qo", "          ")
        for pair in range(COLS // 2):
            name = f"q{pair}"
            lines += acquire_a(row, name, "          ")
            if col // 2 == pair:
                lines.append(
                    f"          func.call @r29_pack_q(%a{name}v, %qov, %lane) : (memref<{PAIR}xi8>, memref<{OUT_TILE}xi8>, i32) -> ()"
                )
            lines.append(f"          aie.objectfifo.release @abc{row}(Consume, 1)")
        lines += [f"          aie.objectfifo.release @oc{col}_{row}(Produce, 1)", "        }"]
        for phase in ("k", "v"):
            lines.append(f"        scf.for %{phase}wave = %z to %waves step %one {{")
            for pair in range(COLS // 2):
                name = f"{phase}{pair}"
                lines += acquire_a(row, name, "          ")
                if col % 2 == 0 and col // 2 == pair:
                    lines += acquire_out(col, row, f"{phase}o0", "          ")
                    if phase == "k":
                        lines.append(
                            f"          func.call @r29_pack_k(%a{name}v, %{phase}o0v, %kinv{col}_{row}, %h0) : (memref<{PAIR}xi8>, memref<{OUT_TILE}xi8>, memref<8xf32>, i32) -> ()"
                        )
                    else:
                        lines.append(
                            f"          func.call @r29_pack_v(%a{name}v, %{phase}o0v, %h0) : (memref<{PAIR}xi8>, memref<{OUT_TILE}xi8>, i32) -> ()"
                        )
                    lines.append(f"          aie.objectfifo.release @oc{col}_{row}(Produce, 1)")
                    lines += acquire_out(col, row, f"{phase}o1", "          ")
                    if phase == "k":
                        lines.append(
                            f"          func.call @r29_pack_k(%a{name}v, %{phase}o1v, %kinv{col}_{row}, %h1) : (memref<{PAIR}xi8>, memref<{OUT_TILE}xi8>, memref<8xf32>, i32) -> ()"
                        )
                    else:
                        lines.append(
                            f"          func.call @r29_pack_v(%a{name}v, %{phase}o1v, %h1) : (memref<{PAIR}xi8>, memref<{OUT_TILE}xi8>, i32) -> ()"
                        )
                    lines.append(f"          aie.objectfifo.release @oc{col}_{row}(Produce, 1)")
                lines.append(f"          aie.objectfifo.release @abc{row}(Consume, 1)")
            lines.append("        }")
        lines += [
            "      }",
            "      aie.end",
            "    } {stack_size = 4096 : i32}",
        ]
        out += lines

out.append(
    f"    aie.runtime_sequence(%A: memref<{A_BYTES}xi8>, %W: memref<{W_BYTES}xi8>, %R: memref<{R_BYTES}xi8>, %Q: memref<{Q_BYTES}xi8>, %KV: memref<{KV_BYTES}xi8>) {{"
)

for row in range(ROWS):
    out += [
        f"      %ta{row} = aiex.dma_configure_task_for @ash{row} {{",
        f"        aie.dma_bd(%A : memref<{A_BYTES}xi8>, {row * INBLOCKS * A_BLOCK}, {INBLOCKS * A_BLOCK}, {dims(INBLOCKS, A_BLOCK)}) {{burst_length = 0 : i32}}",
        "        aie.end",
        "      }",
        f"      aiex.dma_start_task(%ta{row})",
    ]
for col in range(COLS):
    out += [
        f"      %tw{col} = aiex.dma_configure_task_for @wsh{col} {{",
        f"        aie.dma_bd(%W : memref<{W_BYTES}xi8>, {col * INBLOCKS * W_BLOCK}, {INBLOCKS * W_BLOCK}, {dims(INBLOCKS, W_BLOCK)}) {{burst_length = 0 : i32}}",
        "        aie.end",
        "      }",
        f"      aiex.dma_start_task(%tw{col})",
    ]

for outblock in range(OUTBLOCKS):
    m_macro, n_macro = divmod(outblock, N_MACROS)
    for col in range(COLS):
        offset = (n_macro * PAIRS_PER_ROLE + m_macro * 16) * PAIR + col * 64
        name = f"tpo{outblock}_{col}"
        out += [
            f"      %{name} = aiex.dma_configure_task_for @osh{col} {{",
            f"        aie.dma_bd(%R : memref<{R_BYTES}xi8>, {offset}, {OUT_JOIN // 4}, {projection_output_dims()}) {{burst_length = 0 : i32}}",
            "        aie.end",
            "      } {issue_token = true, repeat_count = 3 : i32}",
            f"      aiex.dma_start_task(%{name})",
        ]
    for col in range(COLS):
        name = f"tpo{outblock}_{col}"
        out += [f"      aiex.dma_await_task(%{name})", f"      aiex.dma_free_task(%{name})"]

for row in range(ROWS):
    out.append(f"      aiex.dma_free_task(%ta{row})")
for col in range(COLS):
    out.append(f"      aiex.dma_free_task(%tw{col})")


def emit_raw_inputs(role, base_pair, stem):
    for row in range(ROWS):
        for pair in range(COLS // 2):
            logical_pair = base_pair + row * 4 + pair
            token = logical_pair * 8
            m_macro, within_macro = divmod(token, 96)
            core_row, within_core = divmod(within_macro, 24)
            pair_index = m_macro * 16 + core_row * 4 + within_core // 8
            name = f"t{stem}i{row}_{pair}"
            offset = (role * PAIRS_PER_ROLE + pair_index) * PAIR
            out.extend(
                [
                    f"      %{name} = aiex.dma_configure_task_for @ash{row} {{",
                    f"        aie.dma_bd(%R : memref<{R_BYTES}xi8>, {offset}, {PAIR}, {dims(1, PAIR)}) {{burst_length = 0 : i32}}",
                    "        aie.end",
                    "      } {issue_token = true}",
                    f"      aiex.dma_start_task(%{name})",
                ]
            )


def await_raw_inputs(stem):
    for row in range(ROWS):
        for pair in range(COLS // 2):
            name = f"t{stem}i{row}_{pair}"
            out.extend([f"      aiex.dma_await_task(%{name})", f"      aiex.dma_free_task(%{name})"])


for group in range(QUERY_GROUPS):
    for col in range(COLS):
        offset = group * Q_JOIN + col * OUT_TILE
        name = f"tqo{group}_{col}"
        out += [
            f"      %{name} = aiex.dma_configure_task_for @osh{col} {{",
            f"        aie.dma_bd(%Q : memref<{Q_BYTES}xi8>, {offset}, {ROWS * OUT_TILE}, {strided_dims(ROWS, QUERY_GROUPS * Q_JOIN, OUT_TILE)}) {{burst_length = 0 : i32}}",
            "        aie.end",
            "      } {issue_token = true}",
            f"      aiex.dma_start_task(%{name})",
        ]
    role, half = divmod(group, 2)
    emit_raw_inputs(role, half * 16, f"q{group}")
    await_raw_inputs(f"q{group}")
    for col in range(COLS):
        name = f"tqo{group}_{col}"
        out += [f"      aiex.dma_await_task(%{name})", f"      aiex.dma_free_task(%{name})"]

for phase, role in (("k", 3), ("v", 4)):
    for wave in range(2):
        for col in range(0, COLS, 2):
            pair = col // 2
            group = wave * 16 + pair
            block, key_tile = divmod(group, 2)
            for half in range(2):
                if phase == "k":
                    offset = block * KV_TILE + key_tile * 4096 + half * OUT_TILE
                    dimensions = strided_dims(ROWS, 2 * KV_TILE, OUT_TILE)
                else:
                    offset = block * KV_TILE + K_HALF + (half * 32 + key_tile) * 128
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
        emit_raw_inputs(role, wave * 16, f"{phase}{wave}")
        await_raw_inputs(f"{phase}{wave}")
        for col in range(0, COLS, 2):
            for half in range(2):
                name = f"t{phase}o{wave}_{col}_{half}"
                out += [f"      aiex.dma_await_task(%{name})", f"      aiex.dma_free_task(%{name})"]

out += ["    }", "  }", "}"]
print("\n".join(out))
