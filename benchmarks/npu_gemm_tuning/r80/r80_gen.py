#!/usr/bin/env python3
"""Paired compact-W4 projection on odd cores into the exact R65 stage ABI."""

import sys

COLS, ROWS = 8, 4
PAIRS = COLS // 2
GROUPS, OUTBLOCKS, SLICES = 3, 6, 3
AB, WB = 8192, 16384
OUT_TILE, OUT_JOIN = 2048, 8192
PAIR, PAIRS_PER_ROLE, ROLES = 10240, 48, 5
R_BYTES = ROLES * PAIRS_PER_ROLE * PAIR
INBLOCKS = GROUPS * OUTBLOCKS
WEIGHT_BLOCKS_PER_PAIR = 2 * INBLOCKS
INF = 9223372036854775807
SINGLE_GROUP_FUNCTION = "--single-group-function" in sys.argv[1:]
DYNAMIC_SLICE_LOOP = "--dynamic-slice-loop" in sys.argv[1:]


def dims(count, block):
    return (
        f"[<size = {count}, stride = {block}>, "
        f"<size = {block // 512}, stride = 512>, "
        "<size = 512, stride = 1>]"
    )


def projection_output_dims():
    return (
        f"[<size = 4, stride = {4 * PAIR}>, "
        f"<size = 4, stride = {PAIR}>, "
        "<size = 8, stride = 512>, <size = 64, stride = 1>]"
    )


out = ["module {", "  aie.device(npu2) {"]
for col in range(COLS):
    out += [f"    %shim{col} = aie.tile({col}, 0)", f"    %mt{col} = aie.tile({col}, 1)"]
    for row in range(ROWS):
        out.append(f"    %c{col}_{row} = aie.tile({col}, {row + 2})")
        if col % 2 == 1:
            out += [
                f'    %acc0{col}_{row} = aie.buffer(%c{col}_{row}) {{sym_name = "acc0{col}_{row}"}} : memref<2304xi32>',
                f'    %acc1{col}_{row} = aie.buffer(%c{col}_{row}) {{sym_name = "acc1{col}_{row}"}} : memref<2304xi32>',
            ]

for pair in range(PAIRS):
    col = 2 * pair + 1
    cores = ", ".join(f"%c{col}_{row}" for row in range(ROWS))
    out += [
        f"    aie.objectfifo @wsh{col}(%shim{col}, {{%mt{col}}}, 1 : i32) : !aie.objectfifo<memref<{WB}xi8>>",
        f"    aie.objectfifo @wbc{col}(%mt{col}, {{{cores}}}, 1 : i32) : !aie.objectfifo<memref<{WB}xi8>>",
        f"    aie.objectfifo.link [@wsh{col}] -> [@wbc{col}] ([] [0])",
    ]
for row in range(ROWS):
    cores = ", ".join(f"%c{col}_{row}" for col in range(1, COLS, 2))
    out += [
        f"    aie.objectfifo @ash{row}(%shim{row}, {{%mt{row}}}, 1 : i32) : !aie.objectfifo<memref<{AB}xi8>>",
        f"    aie.objectfifo @abc{row}(%mt{row}, {{{cores}}}, 1 : i32) : !aie.objectfifo<memref<{AB}xi8>>",
        f"    aie.objectfifo.link [@ash{row}] -> [@abc{row}] ([] [0])",
    ]
for pair in range(PAIRS):
    col = 2 * pair + 1
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

if SINGLE_GROUP_FUNCTION:
    out.append(
        f'    func.func private @r70_w4_scaled_group(memref<{AB}xi8>, memref<{WB}xi8>, memref<2304xi32>, i32) attributes {{link_with = "r70group.o"}}'
    )
else:
    for name in ("r15_w4_scaled_init", "r15_w4_scaled_accum"):
        out.append(
            f'    func.func private @{name}(memref<{AB}xi8>, memref<{WB}xi8>, memref<2304xi32>) attributes {{link_with = "r15.o"}}'
        )
if DYNAMIC_SLICE_LOOP:
    out.append(
        '    func.func private @r83_projection_slices() -> i32 attributes {link_with = "r83control.o"}'
    )
out.append(
    f'    func.func private @r65_w4_finish_bf16_slice(memref<2304xi32>, memref<{OUT_TILE}xi8>, i32) attributes {{link_with = "r65finish.o"}}'
)

for col in range(1, COLS, 2):
    for row in range(ROWS):
        lines = [
            f"    %core{col}_{row} = aie.core(%c{col}_{row}) {{",
            "      %z = arith.constant 0 : index",
            f"      %inf = arith.constant {INF} : index",
            "      %one = arith.constant 1 : index",
            *( ["      %init = arith.constant 0 : i32", "      %accumulate = arith.constant 1 : i32"] if SINGLE_GROUP_FUNCTION else [] ),
            *( ["      %projection_slicesi = func.call @r83_projection_slices() : () -> i32", "      %projection_slices = arith.index_cast %projection_slicesi : i32 to index"] if DYNAMIC_SLICE_LOOP else [] ),
            "      scf.for %outer = %z to %inf step %one {",
        ]
        pair = col // 2
        for outblock in range(OUTBLOCKS):
            _, n_macro = divmod(outblock, 2)
            for group in range(GROUPS):
                stem = f"{outblock}_{group}"
                symbol = "r15_w4_scaled_init" if group == 0 else "r15_w4_scaled_accum"
                lines += [
                    f"        %a{stem} = aie.objectfifo.acquire @abc{row}(Consume, 1) : !aie.objectfifosubview<memref<{AB}xi8>>",
                    f"        %a{stem}v = aie.objectfifo.subview.access %a{stem}[0] : !aie.objectfifosubview<memref<{AB}xi8>> -> memref<{AB}xi8>",
                ]
                for lane in range(2):
                    logical_col = 2 * pair + lane
                    valid = n_macro == 0 or logical_col < 5 or logical_col == 5
                    lines += [
                        f"        %w{stem}_{lane} = aie.objectfifo.acquire @wbc{col}(Consume, 1) : !aie.objectfifosubview<memref<{WB}xi8>>",
                        f"        %w{stem}_{lane}v = aie.objectfifo.subview.access %w{stem}_{lane}[0] : !aie.objectfifosubview<memref<{WB}xi8>> -> memref<{WB}xi8>",
                    ]
                    if valid:
                        if SINGLE_GROUP_FUNCTION:
                            accumulate = "%init" if group == 0 else "%accumulate"
                            lines.append(
                                f"        func.call @r70_w4_scaled_group(%a{stem}v, %w{stem}_{lane}v, %acc{lane}{col}_{row}, {accumulate}) : (memref<{AB}xi8>, memref<{WB}xi8>, memref<2304xi32>, i32) -> ()"
                            )
                        else:
                            lines.append(
                                f"        func.call @{symbol}(%a{stem}v, %w{stem}_{lane}v, %acc{lane}{col}_{row}) : (memref<{AB}xi8>, memref<{WB}xi8>, memref<2304xi32>) -> ()"
                            )
                    lines.append(f"        aie.objectfifo.release @wbc{col}(Consume, 1)")
                lines.append(f"        aie.objectfifo.release @abc{row}(Consume, 1)")
            for lane in range(2):
                logical_col = 2 * pair + lane
                valid_slices = SLICES if n_macro == 0 or logical_col < 5 else (1 if logical_col == 5 else 0)
                if DYNAMIC_SLICE_LOOP and valid_slices == SLICES:
                    name = f"o{outblock}_{lane}_loop"
                    lines += [
                        f"        scf.for %slice{outblock}_{lane} = %z to %projection_slices step %one {{",
                        f"          %{name} = aie.objectfifo.acquire @oc{col}_{row}(Produce, 1) : !aie.objectfifosubview<memref<{OUT_TILE}xi8>>",
                        f"          %{name}v = aie.objectfifo.subview.access %{name}[0] : !aie.objectfifosubview<memref<{OUT_TILE}xi8>> -> memref<{OUT_TILE}xi8>",
                        f"          %slicei{outblock}_{lane} = arith.index_cast %slice{outblock}_{lane} : index to i32",
                        f"          func.call @r65_w4_finish_bf16_slice(%acc{lane}{col}_{row}, %{name}v, %slicei{outblock}_{lane}) : (memref<2304xi32>, memref<{OUT_TILE}xi8>, i32) -> ()",
                        f"          aie.objectfifo.release @oc{col}_{row}(Produce, 1)",
                        "        }",
                    ]
                    continue
                for slice_index in range(valid_slices):
                    name = f"o{outblock}_{lane}_{slice_index}"
                    lines += [
                        f"        %{name} = aie.objectfifo.acquire @oc{col}_{row}(Produce, 1) : !aie.objectfifosubview<memref<{OUT_TILE}xi8>>",
                        f"        %{name}v = aie.objectfifo.subview.access %{name}[0] : !aie.objectfifosubview<memref<{OUT_TILE}xi8>> -> memref<{OUT_TILE}xi8>",
                        f"        %slice{outblock}_{lane}_{slice_index} = arith.constant {slice_index} : i32",
                        f"        func.call @r65_w4_finish_bf16_slice(%acc{lane}{col}_{row}, %{name}v, %slice{outblock}_{lane}_{slice_index}) : (memref<2304xi32>, memref<{OUT_TILE}xi8>, i32) -> ()",
                        f"        aie.objectfifo.release @oc{col}_{row}(Produce, 1)",
                    ]
        lines += ["      }", "      aie.end", "    } {stack_size = 2048 : i32}"]
        out += lines

A_BYTES = ROWS * INBLOCKS * AB
W_BYTES = PAIRS * WEIGHT_BLOCKS_PER_PAIR * WB
out.append(
    f"    aie.runtime_sequence(%A: memref<{A_BYTES}xi8>, %W: memref<{W_BYTES}xi8>, %R: memref<{R_BYTES}xi8>) {{"
)
for row in range(ROWS):
    out += [
        f"      %ta{row} = aiex.dma_configure_task_for @ash{row} {{",
        f"        aie.dma_bd(%A : memref<{A_BYTES}xi8>, {row * INBLOCKS * AB}, {INBLOCKS * AB}, {dims(INBLOCKS, AB)}) {{burst_length = 0 : i32}}",
        "        aie.end",
        "      }",
        f"      aiex.dma_start_task(%ta{row})",
    ]
for pair in range(PAIRS):
    col = 2 * pair + 1
    pair_bytes = WEIGHT_BLOCKS_PER_PAIR * WB
    out += [
        f"      %tw{col} = aiex.dma_configure_task_for @wsh{col} {{",
        f"        aie.dma_bd(%W : memref<{W_BYTES}xi8>, {pair * pair_bytes}, {pair_bytes}, {dims(WEIGHT_BLOCKS_PER_PAIR, WB)}) {{burst_length = 0 : i32}}",
        "        aie.end",
        "      }",
        f"      aiex.dma_start_task(%tw{col})",
    ]

for outblock in range(OUTBLOCKS):
    m_macro, n_macro = divmod(outblock, 2)
    for lane in range(2):
        for slice_index in range(SLICES):
            task_names = []
            for pair in range(PAIRS):
                col = 2 * pair + 1
                logical_col = 2 * pair + lane
                valid_slices = (
                    SLICES
                    if n_macro == 0 or logical_col < 5
                    else (1 if logical_col == 5 else 0)
                )
                if slice_index >= valid_slices:
                    continue
                stripe32 = n_macro * 24 + logical_col * 3 + slice_index
                if stripe32 >= ROLES * 8:
                    continue
                role, role_stripe = divmod(stripe32, 8)
                offset = (
                    (role * PAIRS_PER_ROLE + m_macro * 16) * PAIR
                    + role_stripe * 64
                )
                name = f"to{outblock}_{pair}_{lane}_{slice_index}"
                out += [
                    f"      %{name} = aiex.dma_configure_task_for @osh{col} {{",
                    f"        aie.dma_bd(%R : memref<{R_BYTES}xi8>, {offset}, {OUT_TILE}, {projection_output_dims()}) {{burst_length = 0 : i32}}",
                    "        aie.end",
                    "      } {issue_token = true, repeat_count = 3 : i32}",
                    f"      aiex.dma_start_task(%{name})",
                ]
                task_names.append(name)
            for name in task_names:
                out += [
                    f"      aiex.dma_await_task(%{name})",
                    f"      aiex.dma_free_task(%{name})",
                ]

for row in range(ROWS):
    out.append(f"      aiex.dma_free_task(%ta{row})")
for pair in range(PAIRS):
    out.append(f"      aiex.dma_free_task(%tw{2 * pair + 1})")
out += ["    }", "  }", "}"]
print("\n".join(out))
