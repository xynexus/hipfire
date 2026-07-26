#!/usr/bin/env python3
"""Canonical R44 direct-X to canonical unit-RMS BF16."""

COLS, CORE_ROWS = 8, 4
ROWS_PER_CORE, HIDDEN = 8, 768
ROW_BYTES = HIDDEN * 2
CORE_BYTES = ROWS_PER_CORE * ROW_BYTES
JOIN_BYTES = CORE_ROWS * CORE_BYTES
M, PAD_M = 256, 288
OUTPUT_BYTES = PAD_M * ROW_BYTES
R_STAGE_BYTES = 5 * 48 * 16384
ATTENTION_BYTES = M * ROW_BYTES
HIDDEN_BACKING_BYTES = R_STAGE_BYTES + 3 * ATTENTION_BYTES
INF = 9223372036854775807


def linear_dims(size):
    return (
        f"[<size = {size // 512}, stride = 512>, "
        "<size = 512, stride = 1>]"
    )


out = ["module {", "  aie.device(npu2) {"]
for col in range(COLS):
    out += [
        f"    %shim{col} = aie.tile({col}, 0)",
        f"    %mt{col} = aie.tile({col}, 1)",
    ]
    inputs = []
    outputs = []
    offsets = []
    for row in range(CORE_ROWS):
        out += [
            f"    %c{col}_{row} = aie.tile({col}, {row + 2})",
            f"    aie.objectfifo @xc{col}_{row}(%mt{col}, {{%c{col}_{row}}}, 1 : i32) : !aie.objectfifo<memref<{CORE_BYTES}xi8>>",
            f"    aie.objectfifo @oc{col}_{row}(%c{col}_{row}, {{%mt{col}}}, 1 : i32) : !aie.objectfifo<memref<{CORE_BYTES}xi8>>",
        ]
        inputs.append(f"@xc{col}_{row}")
        outputs.append(f"@oc{col}_{row}")
        offsets.append(str(row * CORE_BYTES))
    out += [
        f"    aie.objectfifo @xsh{col}(%shim{col}, {{%mt{col}}}, 1 : i32) : !aie.objectfifo<memref<{JOIN_BYTES}xi8>>",
        f"    aie.objectfifo.link [@xsh{col}] -> [{', '.join(inputs)}] ([] [{', '.join(offsets)}])",
        f"    aie.objectfifo @osh{col}(%mt{col}, {{%shim{col}}}, 1 : i32) : !aie.objectfifo<memref<{JOIN_BYTES}xi8>>",
        f"    aie.objectfifo.link [{', '.join(outputs)}] -> [@osh{col}] ([{', '.join(offsets)}] [])",
    ]

out.append(
    f'    func.func private @r105_direct_x_unit_rms(memref<{CORE_BYTES}xi8>, memref<{CORE_BYTES}xi8>) attributes {{link_with = "r105.o"}}'
)

for col in range(COLS):
    for row in range(CORE_ROWS):
        out += [
            f"    %core{col}_{row} = aie.core(%c{col}_{row}) {{",
            "      %z = arith.constant 0 : index",
            f"      %inf = arith.constant {INF} : index",
            "      %one = arith.constant 1 : index",
            "      scf.for %outer = %z to %inf step %one {",
            f"        %x = aie.objectfifo.acquire @xc{col}_{row}(Consume, 1) : !aie.objectfifosubview<memref<{CORE_BYTES}xi8>>",
            f"        %xv = aie.objectfifo.subview.access %x[0] : !aie.objectfifosubview<memref<{CORE_BYTES}xi8>> -> memref<{CORE_BYTES}xi8>",
            f"        %o = aie.objectfifo.acquire @oc{col}_{row}(Produce, 1) : !aie.objectfifosubview<memref<{CORE_BYTES}xi8>>",
            f"        %ov = aie.objectfifo.subview.access %o[0] : !aie.objectfifosubview<memref<{CORE_BYTES}xi8>> -> memref<{CORE_BYTES}xi8>",
            f"        func.call @r105_direct_x_unit_rms(%xv, %ov) : (memref<{CORE_BYTES}xi8>, memref<{CORE_BYTES}xi8>) -> ()",
            f"        aie.objectfifo.release @xc{col}_{row}(Consume, 1)",
            f"        aie.objectfifo.release @oc{col}_{row}(Produce, 1)",
            "      }",
            "      aie.end",
            "    } {stack_size = 1024 : i32}",
        ]

out.append(
    f"    aie.runtime_sequence(%X: memref<{HIDDEN_BACKING_BYTES}xi8>, %O: memref<{OUTPUT_BYTES}xi8>) {{"
)

for col in range(COLS):
    offset = col * 32 * ROW_BYTES
    out += [
        f"      %to{col} = aiex.dma_configure_task_for @osh{col} {{",
        f"        aie.dma_bd(%O : memref<{OUTPUT_BYTES}xi8>, {offset}, {JOIN_BYTES}, {linear_dims(JOIN_BYTES)}) {{burst_length = 0 : i32}}",
        "        aie.end",
        "      } {issue_token = true}",
        f"      aiex.dma_start_task(%to{col})",
        f"      %tx{col} = aiex.dma_configure_task_for @xsh{col} {{",
        f"        aie.dma_bd(%X : memref<{HIDDEN_BACKING_BYTES}xi8>, {offset}, {JOIN_BYTES}, {linear_dims(JOIN_BYTES)}) {{burst_length = 0 : i32}}",
        "        aie.end",
        "      } {issue_token = true}",
        f"      aiex.dma_start_task(%tx{col})",
    ]

for col in range(COLS):
    out += [
        f"      aiex.dma_await_task(%tx{col})",
        f"      aiex.dma_free_task(%tx{col})",
        f"      aiex.dma_await_task(%to{col})",
        f"      aiex.dma_free_task(%to{col})",
    ]

out += ["    }", "  }", "}"]
print("\n".join(out))
