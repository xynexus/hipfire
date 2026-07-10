#!/usr/bin/env python3
"""All-32-tile resident GeGLU over [256,1152] EmbeddingGemma activations."""

COLS = 8
CORE_ROWS = 4
ROWS_PER_CORE = 8
ROWS = COLS * CORE_ROWS * ROWS_PER_CORE
INTER = 1152
COMBINED = 2 * INTER
INPUT_JOIN = CORE_ROWS * COMBINED
OUTPUT_JOIN = CORE_ROWS * INTER
INF = 9223372036854775807


def input_dims():
    return (
        f"[<size = {ROWS_PER_CORE}, stride = {COMBINED}>, "
        f"<size = {CORE_ROWS}, stride = {ROWS_PER_CORE * COMBINED}>, "
        f"<size = {COMBINED // 16}, stride = 16>, "
        "<size = 16, stride = 1>]"
    )


def output_dims():
    return (
        f"[<size = {ROWS_PER_CORE}, stride = {INTER}>, "
        f"<size = {CORE_ROWS}, stride = {ROWS_PER_CORE * INTER}>, "
        f"<size = {INTER // 16}, stride = 16>, "
        "<size = 16, stride = 1>]"
    )


out = ["module {", "  aie.device(npu2) {"]
for col in range(COLS):
    out += [f"    %shim{col} = aie.tile({col}, 0)", f"    %mt{col} = aie.tile({col}, 1)"]
    for row in range(CORE_ROWS):
        out.append(f"    %c{col}_{row} = aie.tile({col}, {row + 2})")

for col in range(COLS):
    input_offsets = ", ".join(str(row * COMBINED) for row in range(CORE_ROWS))
    output_offsets = ", ".join(str(row * INTER) for row in range(CORE_ROWS))
    input_cores = []
    for row in range(CORE_ROWS):
        input_cores.append(f"@xcore{col}_{row}")
        out.append(
            f"    aie.objectfifo @xcore{col}_{row}(%mt{col}, {{%c{col}_{row}}}, 1 : i32) : !aie.objectfifo<memref<{COMBINED}xf32>>"
        )
    out += [
        f"    aie.objectfifo @xsh{col}(%shim{col}, {{%mt{col}}}, 1 : i32) : !aie.objectfifo<memref<{INPUT_JOIN}xf32>>",
        f"    aie.objectfifo.link [@xsh{col}] -> [{', '.join(input_cores)}] ([] [{input_offsets}])",
    ]
    outputs = []
    for row in range(CORE_ROWS):
        name = f"ocore{col}_{row}"
        outputs.append(f"@{name}")
        out.append(
            f"    aie.objectfifo @{name}(%c{col}_{row}, {{%mt{col}}}, 1 : i32) : !aie.objectfifo<memref<{INTER}xf32>>"
        )
    out += [
        f"    aie.objectfifo @osh{col}(%mt{col}, {{%shim{col}}}, 1 : i32) : !aie.objectfifo<memref<{OUTPUT_JOIN}xf32>>",
        f"    aie.objectfifo.link [{', '.join(outputs)}] -> [@osh{col}] ([{output_offsets}] [])",
    ]

out.append(
    f'    func.func private @r17_geglu_f32(memref<{COMBINED}xf32>, memref<{INTER}xf32>) attributes {{link_with = "r17.o"}}'
)
for col in range(COLS):
    for row in range(CORE_ROWS):
        out += [
            f"    %core{col}_{row} = aie.core(%c{col}_{row}) {{",
            "      %z = arith.constant 0 : index",
            f"      %m = arith.constant {INF} : index",
            "      %o = arith.constant 1 : index",
            "      scf.for %outer = %z to %m step %o {",
            f"        %x = aie.objectfifo.acquire @xcore{col}_{row}(Consume, 1) : !aie.objectfifosubview<memref<{COMBINED}xf32>>",
            f"        %xv = aie.objectfifo.subview.access %x[0] : !aie.objectfifosubview<memref<{COMBINED}xf32>> -> memref<{COMBINED}xf32>",
            f"        %o0 = aie.objectfifo.acquire @ocore{col}_{row}(Produce, 1) : !aie.objectfifosubview<memref<{INTER}xf32>>",
            f"        %ov = aie.objectfifo.subview.access %o0[0] : !aie.objectfifosubview<memref<{INTER}xf32>> -> memref<{INTER}xf32>",
            f"        func.call @r17_geglu_f32(%xv, %ov) : (memref<{COMBINED}xf32>, memref<{INTER}xf32>) -> ()",
            f"        aie.objectfifo.release @xcore{col}_{row}(Consume, 1)",
            f"        aie.objectfifo.release @ocore{col}_{row}(Produce, 1)",
            "      }",
            "      aie.end",
            "    }",
        ]

out.append(
    f"    aie.runtime_sequence(%G: memref<{ROWS * COMBINED}xf32>, %O: memref<{ROWS * INTER}xf32>) {{"
)
for col in range(COLS):
    row0 = col * CORE_ROWS * ROWS_PER_CORE
    out += [
        f"      %to{col} = aiex.dma_configure_task_for @osh{col} {{",
        f"        aie.dma_bd(%O : memref<{ROWS * INTER}xf32>, {row0 * INTER}, {CORE_ROWS * INTER}, {output_dims()}) {{burst_length = 0 : i32}}",
        "        aie.end",
        f"      }} {{issue_token = true, repeat_count = {ROWS_PER_CORE - 1} : i32}}",
        f"      aiex.dma_start_task(%to{col})",
        f"      %tx{col} = aiex.dma_configure_task_for @xsh{col} {{",
        f"        aie.dma_bd(%G : memref<{ROWS * COMBINED}xf32>, {row0 * COMBINED}, {CORE_ROWS * COMBINED}, {input_dims()}) {{burst_length = 0 : i32}}",
        "        aie.end",
        f"      }} {{repeat_count = {ROWS_PER_CORE - 1} : i32}}",
        f"      aiex.dma_start_task(%tx{col})",
    ]
for col in range(COLS):
    out += [
        f"      aiex.dma_await_task(%to{col})",
        f"      aiex.dma_free_task(%to{col})",
        f"      aiex.dma_free_task(%tx{col})",
    ]
out += ["    }", "  }", "}"]
print("\n".join(out))
