#!/usr/bin/env python3
"""R63 W4 projection into an R28-concurrent padded joined BF16 stage."""

COLS, ROWS = 8, 4
GROUPS, OUTBLOCKS = 3, 6
SLICES, TOKEN_GROUPS = 3, 3
AB, WB, ACC_ELEMS = 8192, 16384, 2304
OUT_TILE, OUT_JOIN = 512, 2048
PAIR, PAIRS_PER_ROLE, ROLES = 8192, 36, 5
PARAMS = 2048
VALUE_STAGE_BYTES = ROLES * PAIRS_PER_ROLE * PAIR
STAGE_BYTES = VALUE_STAGE_BYTES + PARAMS
INBLOCKS = GROUPS * OUTBLOCKS
INF = 9223372036854775807


def dims(count, block):
    return (
        f"[<size = {count}, stride = {block}>, "
        f"<size = {block // 512}, stride = 512>, "
        "<size = 512, stride = 1>]"
    )


def stage_output_dims():
    # Four core rows contribute one 8x32 BF16 tile each. Their logical token
    # groups are three records apart in the compact consumption order.
    return (
        f"[<size = {ROWS}, stride = {3 * PAIR}>, "
        "<size = 8, stride = 512>, <size = 64, stride = 1>]"
    )


out = ["module {", "  aie.device(npu2) {"]
for col in range(COLS):
    out += [f"    %shim{col} = aie.tile({col}, 0)", f"    %mt{col} = aie.tile({col}, 1)"]
    for row in range(ROWS):
        out += [
            f"    %c{col}_{row} = aie.tile({col}, {row + 2})",
            f'    %acc{col}_{row} = aie.buffer(%c{col}_{row}) {{sym_name = "acc{col}_{row}"}} : memref<{ACC_ELEMS}xi32>',
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
        f'    func.func private @{name}(memref<{AB}xi8>, memref<{WB}xi8>, memref<{ACC_ELEMS}xi32>) attributes {{link_with = "r15.o"}}'
    )
out.append(
    f'    func.func private @r67_w4_finish_bf16_group(memref<{ACC_ELEMS}xi32>, memref<{OUT_TILE}xi8>, i32, i32) attributes {{link_with = "r67finish.o"}}'
)

for col in range(COLS):
    for row in range(ROWS):
        lines = [
            f"    %core{col}_{row} = aie.core(%c{col}_{row}) {{",
            "      %z = arith.constant 0 : index",
            f"      %inf = arith.constant {INF} : index",
            "      %one = arith.constant 1 : index",
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
                        f"        func.call @{symbol}(%a{stem}v, %w{stem}v, %acc{col}_{row}) : (memref<{AB}xi8>, memref<{WB}xi8>, memref<{ACC_ELEMS}xi32>) -> ()"
                    )
                lines += [
                    f"        aie.objectfifo.release @abc{row}(Consume, 1)",
                    f"        aie.objectfifo.release @wbc{col}(Consume, 1)",
                ]
            for slice_index in range(valid_slices):
                for token_group in range(TOKEN_GROUPS):
                    stem = f"{outblock}_{slice_index}_{token_group}"
                    lines += [
                        f"        %o{stem} = aie.objectfifo.acquire @oc{col}_{row}(Produce, 1) : !aie.objectfifosubview<memref<{OUT_TILE}xi8>>",
                        f"        %o{stem}v = aie.objectfifo.subview.access %o{stem}[0] : !aie.objectfifosubview<memref<{OUT_TILE}xi8>> -> memref<{OUT_TILE}xi8>",
                        f"        %slice{stem} = arith.constant {slice_index} : i32",
                        f"        %group{stem} = arith.constant {token_group} : i32",
                        f"        func.call @r67_w4_finish_bf16_group(%acc{col}_{row}, %o{stem}v, %slice{stem}, %group{stem}) : (memref<{ACC_ELEMS}xi32>, memref<{OUT_TILE}xi8>, i32, i32) -> ()",
                        f"        aie.objectfifo.release @oc{col}_{row}(Produce, 1)",
                    ]
        lines += ["      }", "      aie.end", "    } {stack_size = 2048 : i32}"]
        out += lines

A_BYTES = ROWS * INBLOCKS * AB
W_BYTES = COLS * INBLOCKS * WB
out.append(
    f"    aie.runtime_sequence(%A: memref<{A_BYTES}xi8>, %W: memref<{W_BYTES}xi8>, %R: memref<{STAGE_BYTES}xi8>) {{"
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
        for token_group in range(TOKEN_GROUPS):
            active = []
            for col in range(COLS):
                stripe32 = n_macro * 24 + col * 3 + slice_index
                if stripe32 >= ROLES * 8:
                    continue
                role, role_stripe = divmod(stripe32, 8)
                pair_index = m_macro * 12 + token_group
                offset = (role * PAIRS_PER_ROLE + pair_index) * PAIR + role_stripe * 64
                name = f"to{outblock}_{slice_index}_{token_group}_{col}"
                active.append(name)
                out += [
                    f"      %{name} = aiex.dma_configure_task_for @osh{col} {{",
                    f"        aie.dma_bd(%R : memref<{STAGE_BYTES}xi8>, {offset}, {OUT_JOIN}, {stage_output_dims()}) {{burst_length = 0 : i32}}",
                    "        aie.end",
                    "      } {issue_token = true}",
                    f"      aiex.dma_start_task(%{name})",
                ]
            for name in active:
                out += [f"      aiex.dma_await_task(%{name})", f"      aiex.dma_free_task(%{name})"]

for row in range(ROWS):
    out.append(f"      aiex.dma_free_task(%ta{row})")
for col in range(COLS):
    out.append(f"      aiex.dma_free_task(%tw{col})")
out += ["    }", "  }", "}"]
print("\n".join(out))
