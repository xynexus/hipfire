#!/usr/bin/env python3
"""All-32-core canonical AWQ/FWHT/int8 activation preprocessing at M=256."""

import sys

FUNCTION = sys.argv[1] if len(sys.argv) == 2 else "r19_fwht_quant"
if not FUNCTION.replace("_", "").isalnum():
    raise SystemExit("kernel function must contain only letters, digits, and underscores")

COLS = 8
CORE_ROWS = 4
ROWS_PER_CORE = 8
ROWS = COLS * CORE_ROWS * ROWS_PER_CORE
PAD_K = 1280
PARAM = PAD_K + 256 + 256
SCALE_WIDTH = 8
ROW_OUT = PAD_K + SCALE_WIDTH * 4
INPUT_JOIN = CORE_ROWS * PAD_K
OUTPUT_JOIN = CORE_ROWS * ROW_OUT
INF = 9223372036854775807


def row_dims(width, chunk):
    return (
        f"[<size = {ROWS_PER_CORE}, stride = {width}>, "
        f"<size = {CORE_ROWS}, stride = {ROWS_PER_CORE * width}>, "
        f"<size = {width // chunk}, stride = {chunk}>, "
        f"<size = {chunk}, stride = 1>]"
    )


def contiguous_dims(width):
    return f"[<size = 1, stride = 0>, <size = 1, stride = 0>, <size = {width // 16}, stride = 16>, <size = 16, stride = 1>]"


out = ["module {", "  aie.device(npu2) {"]
for col in range(COLS):
    out += [f"    %shim{col} = aie.tile({col}, 0)", f"    %mt{col} = aie.tile({col}, 1)"]
    for row in range(CORE_ROWS):
        out += [
            f"    %c{col}_{row} = aie.tile({col}, {row + 2})",
            f"    %scratch{col}_{row} = aie.buffer(%c{col}_{row}) {{sym_name = \"scratch{col}_{row}\"}} : memref<256xf32>",
        ]

for col in range(COLS):
    input_offsets = ", ".join(str(row * PAD_K) for row in range(CORE_ROWS))
    output_offsets = ", ".join(str(row * ROW_OUT) for row in range(CORE_ROWS))
    xcores, ocores = [], []
    for row in range(CORE_ROWS):
        xcores.append(f"@xcore{col}_{row}")
        ocores.append(f"@ocore{col}_{row}")
        out += [
            f"    aie.objectfifo @xcore{col}_{row}(%mt{col}, {{%c{col}_{row}}}, 1 : i32) : !aie.objectfifo<memref<{PAD_K}xf32>>",
            f"    aie.objectfifo @ocore{col}_{row}(%c{col}_{row}, {{%mt{col}}}, 1 : i32) : !aie.objectfifo<memref<{ROW_OUT}xi8>>",
        ]
    cores = ", ".join(f"%c{col}_{row}" for row in range(CORE_ROWS))
    out += [
        f"    aie.objectfifo @xsh{col}(%shim{col}, {{%mt{col}}}, 1 : i32) : !aie.objectfifo<memref<{INPUT_JOIN}xf32>>",
        f"    aie.objectfifo.link [@xsh{col}] -> [{', '.join(xcores)}] ([] [{input_offsets}])",
        f"    aie.objectfifo @psh{col}(%shim{col}, {{%mt{col}}}, 1 : i32) : !aie.objectfifo<memref<{PARAM}xf32>>",
        f"    aie.objectfifo @pcore{col}(%mt{col}, {{{cores}}}, 1 : i32) : !aie.objectfifo<memref<{PARAM}xf32>>",
        f"    aie.objectfifo.link [@psh{col}] -> [@pcore{col}] ([] [0])",
        f"    aie.objectfifo @osh{col}(%mt{col}, {{%shim{col}}}, 1 : i32) : !aie.objectfifo<memref<{OUTPUT_JOIN}xi8>>",
        f"    aie.objectfifo.link [{', '.join(ocores)}] -> [@osh{col}] ([{output_offsets}] [])",
    ]

out.append(
    f'    func.func private @{FUNCTION}(memref<{PAD_K}xf32>, memref<{PARAM}xf32>, memref<{ROW_OUT}xi8>, memref<256xf32>) attributes {{link_with = "r19.o"}}'
)
for col in range(COLS):
    for row in range(CORE_ROWS):
        out += [
            f"    %core{col}_{row} = aie.core(%c{col}_{row}) {{",
            "      %z = arith.constant 0 : index",
            f"      %m = arith.constant {INF} : index",
            f"      %rows = arith.constant {ROWS_PER_CORE} : index",
            "      %one = arith.constant 1 : index",
            "      scf.for %outer = %z to %m step %one {",
            f"        %p = aie.objectfifo.acquire @pcore{col}(Consume, 1) : !aie.objectfifosubview<memref<{PARAM}xf32>>",
            f"        %pv = aie.objectfifo.subview.access %p[0] : !aie.objectfifosubview<memref<{PARAM}xf32>> -> memref<{PARAM}xf32>",
            "        scf.for %row = %z to %rows step %one {",
            f"          %x = aie.objectfifo.acquire @xcore{col}_{row}(Consume, 1) : !aie.objectfifosubview<memref<{PAD_K}xf32>>",
            f"          %xv = aie.objectfifo.subview.access %x[0] : !aie.objectfifosubview<memref<{PAD_K}xf32>> -> memref<{PAD_K}xf32>",
            f"          %o = aie.objectfifo.acquire @ocore{col}_{row}(Produce, 1) : !aie.objectfifosubview<memref<{ROW_OUT}xi8>>",
            f"          %ov = aie.objectfifo.subview.access %o[0] : !aie.objectfifosubview<memref<{ROW_OUT}xi8>> -> memref<{ROW_OUT}xi8>",
            f"          func.call @{FUNCTION}(%xv, %pv, %ov, %scratch{col}_{row}) : (memref<{PAD_K}xf32>, memref<{PARAM}xf32>, memref<{ROW_OUT}xi8>, memref<256xf32>) -> ()",
            f"          aie.objectfifo.release @xcore{col}_{row}(Consume, 1)",
            f"          aie.objectfifo.release @ocore{col}_{row}(Produce, 1)",
            "        }",
            f"        aie.objectfifo.release @pcore{col}(Consume, 1)",
            "      }",
            "      aie.end",
            "    } {stack_size = 8192 : i32}",
        ]

out.append(
    f"    aie.runtime_sequence(%X: memref<{ROWS * PAD_K}xf32>, %P: memref<{PARAM}xf32>, %O: memref<{ROWS * ROW_OUT}xi8>) {{"
)
for col in range(COLS):
    row0 = col * CORE_ROWS * ROWS_PER_CORE
    out += [
        f"      %to{col} = aiex.dma_configure_task_for @osh{col} {{",
        f"        aie.dma_bd(%O : memref<{ROWS * ROW_OUT}xi8>, {row0 * ROW_OUT}, {CORE_ROWS * ROW_OUT}, {row_dims(ROW_OUT, 16)}) {{burst_length = 0 : i32}}",
        "        aie.end",
        f"      }} {{issue_token = true, repeat_count = {ROWS_PER_CORE - 1} : i32}}",
        f"      aiex.dma_start_task(%to{col})",
        f"      %tx{col} = aiex.dma_configure_task_for @xsh{col} {{",
        f"        aie.dma_bd(%X : memref<{ROWS * PAD_K}xf32>, {row0 * PAD_K}, {CORE_ROWS * PAD_K}, {row_dims(PAD_K, 16)}) {{burst_length = 0 : i32}}",
        "        aie.end",
        f"      }} {{repeat_count = {ROWS_PER_CORE - 1} : i32}}",
        f"      aiex.dma_start_task(%tx{col})",
        f"      %tp{col} = aiex.dma_configure_task_for @psh{col} {{",
        f"        aie.dma_bd(%P : memref<{PARAM}xf32>, 0, {PARAM}, {contiguous_dims(PARAM)}) {{burst_length = 0 : i32}}",
        "        aie.end",
        "      }",
        f"      aiex.dma_start_task(%tp{col})",
    ]
for col in range(COLS):
    out += [
        f"      aiex.dma_await_task(%to{col})",
        f"      aiex.dma_free_task(%to{col})",
        f"      aiex.dma_free_task(%tx{col})",
        f"      aiex.dma_free_task(%tp{col})",
    ]
out += ["    }", "  }", "}"]
print("\n".join(out))
