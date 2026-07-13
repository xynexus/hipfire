#!/usr/bin/env python3
"""Pack R65 inline 10-KiB QKV records into canonical R27 Q/KV layouts."""

import sys


SPLIT_KV_COLUMNS = "--split-kv-columns" in sys.argv[1:]
R71_PACK_FREE = "--r71-pack-free-4-7" in sys.argv[1:]
R72_DIRECT_Q = "--r72-direct-q" in sys.argv[1:]
R72_LOCAL_Q = "--r72-local-q" in sys.argv[1:]
R73_ADJACENT_Q = "--r73-adjacent-q" in sys.argv[1:]
R78_ODD_ATTENTION = "--r78-odd-attention" in sys.argv[1:]
if R71_PACK_FREE and not SPLIT_KV_COLUMNS:
    raise SystemExit("--r71-pack-free-4-7 requires --split-kv-columns")
if R72_DIRECT_Q and not R71_PACK_FREE:
    raise SystemExit("--r72-direct-q requires --r71-pack-free-4-7")
if R72_LOCAL_Q and not SPLIT_KV_COLUMNS:
    raise SystemExit("--r72-local-q requires --split-kv-columns")
if R72_DIRECT_Q and R72_LOCAL_Q:
    raise SystemExit("R72 direct-Q modes are mutually exclusive")
if R73_ADJACENT_Q and not SPLIT_KV_COLUMNS:
    raise SystemExit("--r73-adjacent-q requires --split-kv-columns")
if sum((R72_DIRECT_Q, R72_LOCAL_Q, R73_ADJACENT_Q)) > 1:
    raise SystemExit("direct-Q modes are mutually exclusive")
if R78_ODD_ATTENTION and not SPLIT_KV_COLUMNS:
    raise SystemExit("--r78-odd-attention requires --split-kv-columns")

COLS, ROWS = 8, 4
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


out = ["module {", "  aie.device(npu2) {"]
for col in range(COLS):
    out += [f"    %shim{col} = aie.tile({col}, 0)", f"    %mt{col} = aie.tile({col}, 1)"]
    for row in range(ROWS):
        out += [
            f"    %c{col}_{row} = aie.tile({col}, {row + 2})",
            f'    %kinv{col}_{row} = aie.buffer(%c{col}_{row}) {{sym_name = "kinv{col}_{row}"}} : memref<8xf32>',
        ]

for row in range(ROWS):
    cores = ", ".join(f"%c{col}_{row}" for col in range(COLS))
    out += [
        f"    aie.objectfifo @rsh{row}(%shim{row}, {{%mt{row}}}, 1 : i32) : !aie.objectfifo<memref<{PAIR}xi8>>",
        f"    aie.objectfifo @rbc{row}(%mt{row}, {{{cores}}}, 1 : i32) : !aie.objectfifo<memref<{PAIR}xi8>>",
        f"    aie.objectfifo.link [@rsh{row}] -> [@rbc{row}] ([] [0])",
    ]

if R72_DIRECT_Q:
    for pair in range(COLS // 2):
        for row in range(ROWS):
            target_col = 4 + row
            out.extend(
                [
                    f'    %qsend{pair}_{row} = aie.buffer(%c{pair}_{row}) {{sym_name = "qsend{pair}_{row}"}} : memref<{OUT_TILE}xi8>',
                    f'    %qcache{target_col}_{pair} = aie.buffer(%c{target_col}_{pair}) {{sym_name = "qcache{target_col}_{pair}"}} : memref<{QUERY_GROUPS * 2 * OUT_TILE}xi8>',
                    f'    aie.flow(%c{pair}_{row}, "Core" : 0, %c{target_col}_{pair}, "Core" : 0)',
                ]
            )

if R73_ADJACENT_Q:
    for pair in range(COLS // 2):
        producer_col = 2 * pair
        consumer_col = producer_col + 1
        for row in range(ROWS):
            out.append(
                f"    aie.objectfifo @qadj{pair}_{row}(%c{producer_col}_{row}, {{%c{consumer_col}_{row}}}, 1 : i32) : !aie.objectfifo<memref<{QUERY_GROUPS * 2 * OUT_TILE // 4}xi32>>"
            )

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
    f'    func.func private @r29_pack_q(memref<{PAIR}xi8>, memref<{OUT_TILE}xi8>, i32) attributes {{link_with = "r66.o"}}',
    f'    func.func private @r29_pack_k(memref<{PAIR}xi8>, memref<{OUT_TILE}xi8>, memref<8xf32>, i32) attributes {{link_with = "r66.o"}}',
    f'    func.func private @r29_pack_v(memref<{PAIR}xi8>, memref<{OUT_TILE}xi8>, i32) attributes {{link_with = "r66.o"}}',
]
if R72_DIRECT_Q:
    out += [
        f'    func.func private @r72_send_q(memref<{OUT_TILE}xi8>) attributes {{link_with = "r72stream.o"}}',
        f'    func.func private @r72_recv_q(memref<{QUERY_GROUPS * 2 * OUT_TILE}xi8>, i32, i32) attributes {{link_with = "r72stream.o"}}',
    ]
if R72_LOCAL_Q or R73_ADJACENT_Q:
    out.append(
        '    func.func private @r72_pack_q_cache(memref<10240xi8>, memref<6144xi32>, i32, i32) attributes {link_with = "r66.o"}'
    )


def acquire_record(row, name, indent="        "):
    return [
        f"{indent}%r{name} = aie.objectfifo.acquire @rbc{row}(Consume, 1) : !aie.objectfifosubview<memref<{PAIR}xi8>>",
        f"{indent}%r{name}v = aie.objectfifo.subview.access %r{name}[0] : !aie.objectfifosubview<memref<{PAIR}xi8>> -> memref<{PAIR}xi8>",
    ]


def acquire_output(col, row, name, indent="        "):
    return [
        f"{indent}%{name} = aie.objectfifo.acquire @oc{col}_{row}(Produce, 1) : !aie.objectfifosubview<memref<{OUT_TILE}xi8>>",
        f"{indent}%{name}v = aie.objectfifo.subview.access %{name}[0] : !aie.objectfifosubview<memref<{OUT_TILE}xi8>> -> memref<{OUT_TILE}xi8>",
    ]


def kv_owner(phase, pair):
    if R73_ADJACENT_Q or R78_ODD_ATTENTION:
        return 2 * pair
    if R72_LOCAL_Q:
        return COLS // 2 + pair
    if not SPLIT_KV_COLUMNS:
        return 2 * pair
    if R71_PACK_FREE and phase == "v":
        return pair
    return pair if phase == "k" else COLS // 2 + pair


def q_owner(logical_col):
    if R73_ADJACENT_Q or R78_ODD_ATTENTION:
        return 2 * (logical_col // 2)
    if R72_DIRECT_Q or R72_LOCAL_Q:
        return logical_col // 2
    if R71_PACK_FREE and logical_col >= 4:
        return logical_col - 4
    return logical_col


for col in range(COLS):
    for row in range(ROWS):
        lines = [
            f"    %core{col}_{row} = aie.core(%c{col}_{row}) {{",
            "      %z = arith.constant 0 : index",
            f"      %inf = arith.constant {INF} : index",
            "      %one = arith.constant 1 : index",
            f"      %lane = arith.constant {col % 2} : i32",
            "      %h0 = arith.constant 0 : i32",
            "      %h1 = arith.constant 1 : i32",
            "      scf.for %outer = %z to %inf step %one {",
        ]
        if R73_ADJACENT_Q and col % 2 == 0:
            pair = col // 2
            lines += [
                f"        %qadjp = aie.objectfifo.acquire @qadj{pair}_{row}(Produce, 1) : !aie.objectfifosubview<memref<{QUERY_GROUPS * 2 * OUT_TILE // 4}xi32>>",
                f"        %qadjpv = aie.objectfifo.subview.access %qadjp[0] : !aie.objectfifosubview<memref<{QUERY_GROUPS * 2 * OUT_TILE // 4}xi32>> -> memref<{QUERY_GROUPS * 2 * OUT_TILE // 4}xi32>",
            ]
        if R72_LOCAL_Q or R73_ADJACENT_Q:
            lines.insert(-1, f"      %qgroups = arith.constant {QUERY_GROUPS} : index")
            lines += [
                "        scf.for %qgroup = %z to %qgroups step %one {",
                "          %qgroupi = arith.index_cast %qgroup : index to i32",
            ]
            for pair in range(COLS // 2):
                name = f"q_{pair}"
                lines += acquire_record(row, name, indent="          ")
                for logical_col in (2 * pair, 2 * pair + 1):
                    if col == q_owner(logical_col):
                        lines.append(
                            f"          func.call @r72_pack_q_cache(%r{name}v, {'%qadjpv' if R73_ADJACENT_Q else f'%acc{col}_{row}'}, %qgroupi, %h{logical_col % 2}) : (memref<{PAIR}xi8>, memref<6144xi32>, i32, i32) -> ()"
                        )
                lines.append(f"          aie.objectfifo.release @rbc{row}(Consume, 1)")
            lines.append("        }")
            if R73_ADJACENT_Q and col % 2 == 0:
                lines.append(
                    f"        aie.objectfifo.release @qadj{col // 2}_{row}(Produce, 1)"
                )
        else:
            for group in range(QUERY_GROUPS):
                for pair in range(COLS // 2):
                    name = f"q{group}_{pair}"
                    lines += acquire_record(row, name)
                    for logical_col in (2 * pair, 2 * pair + 1):
                        if col == q_owner(logical_col):
                            output_name = f"qo{group}_{logical_col}"
                            if R72_DIRECT_Q:
                                lines += [
                                    f"        func.call @r29_pack_q(%r{name}v, %qsend{col}_{row}, %h{logical_col % 2}) : (memref<{PAIR}xi8>, memref<{OUT_TILE}xi8>, i32) -> ()",
                                    f"        func.call @r72_send_q(%qsend{col}_{row}) : (memref<{OUT_TILE}xi8>) -> ()",
                                ]
                            else:
                                lines += acquire_output(col, row, output_name)
                                lines += [
                                    f"        func.call @r29_pack_q(%r{name}v, %{output_name}v, %h{logical_col % 2}) : (memref<{PAIR}xi8>, memref<{OUT_TILE}xi8>, i32) -> ()",
                                    f"        aie.objectfifo.release @oc{col}_{row}(Produce, 1)",
                                ]
                    if R72_DIRECT_Q and col >= 4 and pair == row:
                        lines += [
                            f"        %qgroup{group} = arith.constant {group} : i32",
                            f"        func.call @r72_recv_q(%qcache{col}_{row}, %qgroup{group}, %h0) : (memref<{QUERY_GROUPS * 2 * OUT_TILE}xi8>, i32, i32) -> ()",
                            f"        func.call @r72_recv_q(%qcache{col}_{row}, %qgroup{group}, %h1) : (memref<{QUERY_GROUPS * 2 * OUT_TILE}xi8>, i32, i32) -> ()",
                        ]
                    lines.append(f"        aie.objectfifo.release @rbc{row}(Consume, 1)")
        for phase in ("k", "v"):
            for wave in range(2):
                for pair in range(COLS // 2):
                    name = f"{phase}{wave}_{pair}"
                    lines += acquire_record(row, name)
                    if col == kv_owner(phase, pair):
                        for half in range(2):
                            output_name = f"{phase}o{wave}_{pair}_{half}"
                            lines += acquire_output(col, row, output_name)
                            if phase == "k":
                                lines += [
                                    f"        func.call @r29_pack_k(%r{name}v, %{output_name}v, %kinv{col}_{row}, %h{half}) : (memref<{PAIR}xi8>, memref<{OUT_TILE}xi8>, memref<8xf32>, i32) -> ()",
                                ]
                            else:
                                lines += [
                                    f"        func.call @r29_pack_v(%r{name}v, %{output_name}v, %h{half}) : (memref<{PAIR}xi8>, memref<{OUT_TILE}xi8>, i32) -> ()",
                                ]
                            lines.append(f"        aie.objectfifo.release @oc{col}_{row}(Produce, 1)")
                    lines.append(f"        aie.objectfifo.release @rbc{row}(Consume, 1)")
        lines += ["      }", "      aie.end", "    } {stack_size = 4096 : i32}"]
        out += lines

out.append(
    f"    aie.runtime_sequence(%R: memref<{R_BYTES}xi8>, %Q: memref<{Q_BYTES}xi8>, %KV: memref<{KV_BYTES}xi8>) {{"
)


def emit_record_inputs(role, base_pair, stem):
    for row in range(ROWS):
        for pair in range(COLS // 2):
            logical_pair = base_pair + row * 4 + pair
            token = logical_pair * 8
            m_macro, within_macro = divmod(token, 96)
            core_row, within_core = divmod(within_macro, 24)
            pair_index = m_macro * 16 + core_row * 4 + within_core // 8
            offset = (role * PAIRS_PER_ROLE + pair_index) * PAIR
            name = f"ti{stem}_{row}_{pair}"
            out.extend(
                [
                    f"      %{name} = aiex.dma_configure_task_for @rsh{row} {{",
                    f"        aie.dma_bd(%R : memref<{R_BYTES}xi8>, {offset}, {PAIR}, {dims(1, PAIR)}) {{burst_length = 0 : i32}}",
                    "        aie.end",
                    "      } {issue_token = true}",
                    f"      aiex.dma_start_task(%{name})",
                ]
            )


def await_record_inputs(stem):
    for row in range(ROWS):
        for pair in range(COLS // 2):
            name = f"ti{stem}_{row}_{pair}"
            out.extend([f"      aiex.dma_await_task(%{name})", f"      aiex.dma_free_task(%{name})"])


for group in range(QUERY_GROUPS):
    if not (R72_DIRECT_Q or R72_LOCAL_Q or R73_ADJACENT_Q):
        for logical_col in range(COLS):
            owner = q_owner(logical_col)
            offset = group * Q_JOIN + logical_col * OUT_TILE
            name = f"tqo{group}_{logical_col}"
            out += [
                f"      %{name} = aiex.dma_configure_task_for @osh{owner} {{",
                f"        aie.dma_bd(%Q : memref<{Q_BYTES}xi8>, {offset}, {ROWS * OUT_TILE}, {strided_dims(ROWS, QUERY_GROUPS * Q_JOIN, OUT_TILE)}) {{burst_length = 0 : i32}}",
                "        aie.end",
                "      } {issue_token = true}",
                f"      aiex.dma_start_task(%{name})",
            ]
    role, half = divmod(group, 2)
    emit_record_inputs(role, half * 16, f"q{group}")
    await_record_inputs(f"q{group}")
    if not (R72_DIRECT_Q or R72_LOCAL_Q or R73_ADJACENT_Q):
        for logical_col in range(COLS):
            name = f"tqo{group}_{logical_col}"
            out += [f"      aiex.dma_await_task(%{name})", f"      aiex.dma_free_task(%{name})"]

for phase, role in (("k", 3), ("v", 4)):
    for wave in range(2):
        for pair in range(COLS // 2):
            col = kv_owner(phase, pair)
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
                name = f"t{phase}o{wave}_{pair}_{col}_{half}"
                out += [
                    f"      %{name} = aiex.dma_configure_task_for @osh{col} {{",
                    f"        aie.dma_bd(%KV : memref<{KV_BYTES}xi8>, {offset}, {ROWS * OUT_TILE}, {dimensions}) {{burst_length = 0 : i32}}",
                    "        aie.end",
                    "      } {issue_token = true}",
                    f"      aiex.dma_start_task(%{name})",
                ]
        emit_record_inputs(role, wave * 16, f"{phase}{wave}")
        await_record_inputs(f"{phase}{wave}")
        for pair in range(COLS // 2):
            col = kv_owner(phase, pair)
            for half in range(2):
                name = f"t{phase}o{wave}_{pair}_{col}_{half}"
                out += [f"      aiex.dma_await_task(%{name})", f"      aiex.dma_free_task(%{name})"]

out += ["    }", "  }", "}"]
print("\n".join(out))
