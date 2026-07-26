#!/usr/bin/env python3
"""R63 W4 projection with direct BF16 output into the R29 raw-stage ABI."""

COLS, ROWS = 8, 4
GROUPS, OUTBLOCKS, SLICES = 3, 6, 3
AB, WB, ACC_BYTES = 8192, 16384, 2304 * 4
OUT_TILE, OUT_JOIN = 2048, 8192
# R29's raw attention record is 10 KiB: 4 KiB projected values, 4 KiB
# cos/sin, and 2 KiB norm/epsilon parameters. Projection DMA only overwrites
# the first region; the latter two are preseeded by the caller.
PAIR, PAIRS_PER_ROLE, ROLES = 10240, 48, 5
R_BYTES = ROLES * PAIRS_PER_ROLE * PAIR
INBLOCKS = GROUPS * OUTBLOCKS
INF = 9223372036854775807


def dims(count, block):
    return (
        f"[<size = {count}, stride = {block}>, "
        f"<size = {block // 512}, stride = 512>, "
        "<size = 512, stride = 1>]"
    )


def projection_output_dims():
    # One joined column carries four padded 32x32 BF16 core records. Scatter
    # the first 24 rows into sixteen eight-token R29 records; every fourth
    # record remains the established padding slot.
    return (
        f"[<size = 4, stride = {4 * PAIR}>, "
        f"<size = 4, stride = {PAIR}>, "
        "<size = 8, stride = 512>, <size = 64, stride = 1>]"
    )


out = ["module {", "  aie.device(npu2) {"]
for col in range(COLS):
    out += [f"    %shim{col} = aie.tile({col}, 0)", f"    %mt{col} = aie.tile({col}, 1)"]
    for row in range(ROWS):
        out += [
            f"    %c{col}_{row} = aie.tile({col}, {row + 2})",
            f'    %acc{col}_{row} = aie.buffer(%c{col}_{row}) {{sym_name = "acc{col}_{row}"}} : memref<2304xi32>',
        ]

for col in range(COLS):
    cores = ", ".join(f"%c{col}_{row}" for row in range(ROWS))
    out += [
        f"    aie.objectfifo @wsh{col}(%shim{col}, {{%mt{col}}}, 1 : i32) : !aie.objectfifo<memref<{WB}xi8>>",
        f"    aie.objectfifo @wbc{col}(%mt{col}, {{{cores}}}, 1 : i32) : !aie.objectfifo<memref<{WB}xi8>>",
        f"    aie.objectfifo.link [@wsh{col}] -> [@wbc{col}] ([] [0])",
    ]
for row in range(ROWS):
    cores = ", ".join(f"%c{col}_{row}" for col in range(COLS))
    out += [
        f"    aie.objectfifo @ash{row}(%shim{row}, {{%mt{row}}}, 1 : i32) : !aie.objectfifo<memref<{AB}xi8>>",
        f"    aie.objectfifo @abc{row}(%mt{row}, {{{cores}}}, 1 : i32) : !aie.objectfifo<memref<{AB}xi8>>",
        f"    aie.objectfifo.link [@ash{row}] -> [@abc{row}] ([] [0])",
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

for name in ("r15_w4_scaled_init", "r15_w4_scaled_accum"):
    out.append(
        f'    func.func private @{name}(memref<{AB}xi8>, memref<{WB}xi8>, memref<2304xi32>) attributes {{link_with = "r15.o"}}'
    )
out.append(
    f'    func.func private @r65_w4_finish_bf16_slice(memref<2304xi32>, memref<{OUT_TILE}xi8>, i32) attributes {{link_with = "r65finish.o"}}'
)

for col in range(COLS):
    for row in range(ROWS):
        lines = [
            f"    %core{col}_{row} = aie.core(%c{col}_{row}) {{",
            "      %z = arith.constant 0 : index",
            f"      %inf = arith.constant {INF} : index",
            "      %one = arith.constant 1 : index",
            f"      %groups = arith.constant {GROUPS} : index",
            "      scf.for %outer = %z to %inf step %one {",
        ]
        for outblock in range(OUTBLOCKS):
            _, n_macro = divmod(outblock, 2)
            valid_slices = SLICES if n_macro == 0 or col < 5 else (1 if col == 5 else 0)
            for group in range(GROUPS):
                stem = f"{outblock}_{group}"
                lines += [
                    f"        %a{stem} = aie.objectfifo.acquire @abc{row}(Consume, 1) : !aie.objectfifosubview<memref<{AB}xi8>>",
                    f"        %a{stem}v = aie.objectfifo.subview.access %a{stem}[0] : !aie.objectfifosubview<memref<{AB}xi8>> -> memref<{AB}xi8>",
                    f"        %w{stem} = aie.objectfifo.acquire @wbc{col}(Consume, 1) : !aie.objectfifosubview<memref<{WB}xi8>>",
                    f"        %w{stem}v = aie.objectfifo.subview.access %w{stem}[0] : !aie.objectfifosubview<memref<{WB}xi8>> -> memref<{WB}xi8>",
                ]
                if valid_slices:
                    symbol = "r15_w4_scaled_init" if group == 0 else "r15_w4_scaled_accum"
                    lines.append(
                        f"        func.call @{symbol}(%a{stem}v, %w{stem}v, %acc{col}_{row}) : (memref<{AB}xi8>, memref<{WB}xi8>, memref<2304xi32>) -> ()"
                    )
                lines += [
                    f"        aie.objectfifo.release @abc{row}(Consume, 1)",
                    f"        aie.objectfifo.release @wbc{col}(Consume, 1)",
                ]
            for slice_index in range(valid_slices):
                lines += [
                    f"        %o{outblock}_{slice_index} = aie.objectfifo.acquire @oc{col}_{row}(Produce, 1) : !aie.objectfifosubview<memref<{OUT_TILE}xi8>>",
                    f"        %o{outblock}_{slice_index}v = aie.objectfifo.subview.access %o{outblock}_{slice_index}[0] : !aie.objectfifosubview<memref<{OUT_TILE}xi8>> -> memref<{OUT_TILE}xi8>",
                    f"        %slice{outblock}_{slice_index} = arith.constant {slice_index} : i32",
                    f"        func.call @r65_w4_finish_bf16_slice(%acc{col}_{row}, %o{outblock}_{slice_index}v, %slice{outblock}_{slice_index}) : (memref<2304xi32>, memref<{OUT_TILE}xi8>, i32) -> ()",
                    f"        aie.objectfifo.release @oc{col}_{row}(Produce, 1)",
                ]
        lines += [
            "      }",
            "      aie.end",
            "    } {stack_size = 2048 : i32}",
        ]
        out += lines

A_BYTES = ROWS * INBLOCKS * AB
W_BYTES = COLS * INBLOCKS * WB
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
for col in range(COLS):
    out += [
        f"      %tw{col} = aiex.dma_configure_task_for @wsh{col} {{",
        f"        aie.dma_bd(%W : memref<{W_BYTES}xi8>, {col * INBLOCKS * WB}, {INBLOCKS * WB}, {dims(INBLOCKS, WB)}) {{burst_length = 0 : i32}}",
        "        aie.end",
        "      }",
        f"      aiex.dma_start_task(%tw{col})",
    ]

for outblock in range(OUTBLOCKS):
    m_macro, n_macro = divmod(outblock, 2)
    for slice_index in range(SLICES):
        for col in range(COLS):
            stripe32 = n_macro * 24 + col * 3 + slice_index
            if stripe32 >= ROLES * 8:
                continue
            role, role_stripe = divmod(stripe32, 8)
            offset = (role * PAIRS_PER_ROLE + m_macro * 16) * PAIR + role_stripe * 64
            name = f"to{outblock}_{slice_index}_{col}"
            out += [
                f"      %{name} = aiex.dma_configure_task_for @osh{col} {{",
                f"        aie.dma_bd(%R : memref<{R_BYTES}xi8>, {offset}, {OUT_TILE}, {projection_output_dims()}) {{burst_length = 0 : i32}}",
                "        aie.end",
                "      } {issue_token = true, repeat_count = 3 : i32}",
                f"      aiex.dma_start_task(%{name})",
            ]
        for col in range(COLS):
            stripe32 = n_macro * 24 + col * 3 + slice_index
            if stripe32 >= ROLES * 8:
                continue
            name = f"to{outblock}_{slice_index}_{col}"
            out += [f"      aiex.dma_await_task(%{name})", f"      aiex.dma_free_task(%{name})"]

for row in range(ROWS):
    out.append(f"      aiex.dma_free_task(%ta{row})")
for col in range(COLS):
    out.append(f"      aiex.dma_free_task(%tw{col})")
out += ["    }", "  }", "}"]
print("\n".join(out))
