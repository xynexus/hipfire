#!/usr/bin/env python3
"""Combined vector activation pack and resident W4 scaled down projection."""

COLS = 8
CORE_ROWS = 4
ROWS_PER_STRIPE = 24
OUTBLOCKS = 3
GROUPS = 5
PAD_K = 1280
X_WIDTH = 256
AB = 8192
WB = 16384
CB = 2304
CJ = CORE_ROWS * CB
PAD_M = 288
PAD_N = 768
XBLOCKS = OUTBLOCKS * GROUPS * ROWS_PER_STRIPE
WBLOCKS = OUTBLOCKS * GROUPS
INF = 9223372036854775807


def contiguous_dims(count, block):
    return (
        f"[<size = {count}, stride = {block}>, "
        "<size = 1, stride = 0>, "
        f"<size = {block // 16}, stride = 16>, "
        "<size = 16, stride = 1>]"
    )


def rowmajor_dims():
    return (
        f"[<size = 24, stride = {4 * PAD_N}>, "
        "<size = 6, stride = 16>, "
        f"<size = 4, stride = {PAD_N}>, "
        "<size = 16, stride = 1>]"
    )


out = ["module {", "  aie.device(npu2) {"]
for col in range(COLS):
    out += [f"    %shim{col} = aie.tile({col}, 0)", f"    %mt{col} = aie.tile({col}, 1)"]
    for row in range(CORE_ROWS):
        out.append(f"    %c{col}_{row} = aie.tile({col}, {row + 2})")
        if col == 0:
            out.append(
                f"    %scratch{row} = aie.buffer(%c0_{row}) {{sym_name = \"scratch{row}\"}} : memref<256xf32>"
            )
for col in range(COLS):
    cores = ", ".join(f"%c{col}_{row}" for row in range(CORE_ROWS))
    out += [
        f"    aie.objectfifo @wsh{col}(%shim{col}, {{%mt{col}}}, 1 : i32) : !aie.objectfifo<memref<{WB}xi8>>",
        f"    aie.objectfifo @wbc{col}(%mt{col}, {{{cores}}}, 1 : i32) : !aie.objectfifo<memref<{WB}xi8>>",
        f"    aie.objectfifo.link [@wsh{col}] -> [@wbc{col}] ([] [0])",
    ]
for row in range(CORE_ROWS):
    out += [
        f"    aie.objectfifo @xsh{row}(%shim{row}, {{%mt{row}}}, 1 : i32) : !aie.objectfifo<memref<{X_WIDTH}xf32>>",
        f"    aie.objectfifo @xbc{row}(%mt{row}, {{%c0_{row}}}, 1 : i32) : !aie.objectfifo<memref<{X_WIDTH}xf32>>",
        f"    aie.objectfifo.link [@xsh{row}] -> [@xbc{row}] ([] [0])",
        f"    aie.objectfifo @abc{row}(%c0_{row}, {{{', '.join(f'%c{col}_{row}' for col in range(1, COLS))}}}, 1 : i32) : !aie.objectfifo<memref<{AB}xi8>>",
    ]
for col in range(COLS):
    inputs = ", ".join(f"@cc{col}_{row}" for row in range(CORE_ROWS))
    offsets = ", ".join(str(row * CB) for row in range(CORE_ROWS))
    for row in range(CORE_ROWS):
        out.append(
            f"    aie.objectfifo @cc{col}_{row}(%c{col}_{row}, {{%mt{col}}}, 1 : i32) : !aie.objectfifo<memref<{CB}xi32>>"
        )
    out += [
        f"    aie.objectfifo @csh{col}(%mt{col}, {{%shim{col}}}, 1 : i32) : !aie.objectfifo<memref<{CJ}xi32>>",
        f"    aie.objectfifo.link [{inputs}] -> [@csh{col}] ([{offsets}] [])",
    ]
out += [
    f'    func.func private @r21_w4_pack_row(memref<{X_WIDTH}xf32>, memref<{WB}xi8>, memref<{AB}xi8>, memref<256xf32>, i32) attributes {{link_with = "r21.o"}}',
    f'    func.func private @r15_w4_scaled_init(memref<{AB}xi8>, memref<{WB}xi8>, memref<{CB}xi32>) attributes {{link_with = "r21.o"}}',
    f'    func.func private @r15_w4_scaled_accum(memref<{AB}xi8>, memref<{WB}xi8>, memref<{CB}xi32>) attributes {{link_with = "r21.o"}}',
]
for col in range(COLS):
    for row in range(CORE_ROWS):
        out += [
            f"    %core{col}_{row} = aie.core(%c{col}_{row}) {{",
            "      %z = arith.constant 0 : index",
            f"      %m = arith.constant {INF} : index",
            "      %one = arith.constant 1 : index",
            f"      %outblocks = arith.constant {OUTBLOCKS} : index",
            f"      %groups = arith.constant {GROUPS} : index",
            f"      %rows = arith.constant {ROWS_PER_STRIPE} : index",
            "      scf.for %outer = %z to %m step %one {",
            f"        scf.for %outblock = %z to %outblocks step %one {{",
            f"          %c = aie.objectfifo.acquire @cc{col}_{row}(Produce, 1) : !aie.objectfifosubview<memref<{CB}xi32>>",
            f"          %cv = aie.objectfifo.subview.access %c[0] : !aie.objectfifosubview<memref<{CB}xi32>> -> memref<{CB}xi32>",
            "          scf.for %group = %z to %groups step %one {",
            f"            %w = aie.objectfifo.acquire @wbc{col}(Consume, 1) : !aie.objectfifosubview<memref<{WB}xi8>>",
            f"            %wv = aie.objectfifo.subview.access %w[0] : !aie.objectfifosubview<memref<{WB}xi8>> -> memref<{WB}xi8>",
        ]
        if col == 0:
            out += [
                f"            %a = aie.objectfifo.acquire @abc{row}(Produce, 1) : !aie.objectfifosubview<memref<{AB}xi8>>",
                f"            %av = aie.objectfifo.subview.access %a[0] : !aie.objectfifosubview<memref<{AB}xi8>> -> memref<{AB}xi8>",
                "            scf.for %row = %z to %rows step %one {",
                f"              %x = aie.objectfifo.acquire @xbc{row}(Consume, 1) : !aie.objectfifosubview<memref<{X_WIDTH}xf32>>",
                f"              %xv = aie.objectfifo.subview.access %x[0] : !aie.objectfifosubview<memref<{X_WIDTH}xf32>> -> memref<{X_WIDTH}xf32>",
                "              %row_i32 = arith.index_cast %row : index to i32",
                f"              func.call @r21_w4_pack_row(%xv, %wv, %av, %scratch{row}, %row_i32) : (memref<{X_WIDTH}xf32>, memref<{WB}xi8>, memref<{AB}xi8>, memref<256xf32>, i32) -> ()",
                f"              aie.objectfifo.release @xbc{row}(Consume, 1)",
                "            }",
            ]
            activation = "%av"
        else:
            out += [
                f"            %a = aie.objectfifo.acquire @abc{row}(Consume, 1) : !aie.objectfifosubview<memref<{AB}xi8>>",
                f"            %av = aie.objectfifo.subview.access %a[0] : !aie.objectfifosubview<memref<{AB}xi8>> -> memref<{AB}xi8>",
            ]
            activation = "%av"
        out += [
            "            %first = arith.cmpi eq, %group, %z : index",
            "            scf.if %first {",
            f"              func.call @r15_w4_scaled_init({activation}, %wv, %cv) : (memref<{AB}xi8>, memref<{WB}xi8>, memref<{CB}xi32>) -> ()",
            "            } else {",
            f"              func.call @r15_w4_scaled_accum({activation}, %wv, %cv) : (memref<{AB}xi8>, memref<{WB}xi8>, memref<{CB}xi32>) -> ()",
            "            }",
        ]
        if col == 0:
            out.append(f"            aie.objectfifo.release @abc{row}(Produce, 1)")
        else:
            out.append(f"            aie.objectfifo.release @abc{row}(Consume, 1)")
        out += [
            f"            aie.objectfifo.release @wbc{col}(Consume, 1)",
            "          }",
            f"          aie.objectfifo.release @cc{col}_{row}(Produce, 1)",
            "        }",
            "      }",
            "      aie.end",
            "    } {stack_size = 4096 : i32}",
        ]

XT = XBLOCKS * X_WIDTH
WT = WBLOCKS * WB
out.append(
    f"    aie.runtime_sequence(%X: memref<{CORE_ROWS * XT}xf32>, %W: memref<{COLS * WT}xi8>, %C: memref<{PAD_M * PAD_N}xi32>) {{"
)
for row in range(CORE_ROWS):
    out += [
        f"      %tx{row} = aiex.dma_configure_task_for @xsh{row} {{",
        f"        aie.dma_bd(%X : memref<{CORE_ROWS * XT}xf32>, {row * XT}, {XT}, {contiguous_dims(XBLOCKS, X_WIDTH)}) {{burst_length = 0 : i32}}",
        "        aie.end",
        "      }",
        f"      aiex.dma_start_task(%tx{row})",
    ]
for col in range(COLS):
    out += [
        f"      %tw{col} = aiex.dma_configure_task_for @wsh{col} {{",
        f"        aie.dma_bd(%W : memref<{COLS * WT}xi8>, {col * WT}, {WT}, {contiguous_dims(WBLOCKS, WB)}) {{burst_length = 0 : i32}}",
        "        aie.end",
        "      }",
        f"      aiex.dma_start_task(%tw{col})",
    ]
for outblock in range(OUTBLOCKS):
    for col in range(COLS):
        offset = outblock * 96 * PAD_N + col * 96
        name = f"tc{col}_{outblock}"
        out += [
            f"      %{name} = aiex.dma_configure_task_for @csh{col} {{",
            f"        aie.dma_bd(%C : memref<{PAD_M * PAD_N}xi32>, {offset}, {6 * 4 * 16}, {rowmajor_dims()}) {{burst_length = 0 : i32}}",
            "        aie.end",
            f"      }} {{issue_token = true, repeat_count = {4 * 6 - 1} : i32}}",
            f"      aiex.dma_start_task(%{name})",
        ]
    for col in range(COLS):
        name = f"tc{col}_{outblock}"
        out += [f"      aiex.dma_await_task(%{name})", f"      aiex.dma_free_task(%{name})"]
for row in range(CORE_ROWS):
    out.append(f"      aiex.dma_free_task(%tx{row})")
for col in range(COLS):
    out.append(f"      aiex.dma_free_task(%tw{col})")
out += ["    }", "  }", "}"]
print("\n".join(out))
