#!/usr/bin/env python3
"""R46 BF16x2 completed state to padded R48 residual records."""

COLS, CORE_ROWS, ROWS_PER_CORE, HIDDEN = 8, 4, 8, 768
PAD_M = 288
COMPLETED_ROW = 2 * HIDDEN * 2
COMPLETED_BYTES = PAD_M * COMPLETED_ROW
X_ROW_BYTES = COMPLETED_ROW
X_JOIN_BYTES = CORE_ROWS * X_ROW_BYTES
R34_BLOCK = 16384
R34_INPUT_BYTES = 4 * 45 * R34_BLOCK
RESIDUAL_BYTES = COLS * CORE_ROWS * R34_BLOCK
OUTPUT_BYTES = R34_INPUT_BYTES + RESIDUAL_BYTES
INF = 9223372036854775807


def x_dims():
    return (
        f"[<size = {ROWS_PER_CORE}, stride = {COMPLETED_ROW}>, "
        f"<size = {CORE_ROWS}, stride = {ROWS_PER_CORE * COMPLETED_ROW}>, "
        f"<size = {COMPLETED_ROW // 32}, stride = 32>, "
        "<size = 32, stride = 1>]"
    )


def output_dims():
    return (
        f"[<size = {CORE_ROWS}, stride = {CORE_ROWS * R34_BLOCK}>, "
        "<size = 32, stride = 512>, <size = 512, stride = 1>]"
    )


out = ["module {", "  aie.device(npu2) {"]
for col in range(COLS):
    out += [f"    %shim{col} = aie.tile({col}, 0)", f"    %mt{col} = aie.tile({col}, 1)"]
    xcores, outputs, offsets = [], [], []
    for row in range(CORE_ROWS):
        out.append(f"    %c{col}_{row} = aie.tile({col}, {row + 2})")
        xcores.append(f"@xc{col}_{row}")
        outputs.append(f"@oc{col}_{row}")
        offsets.append(str(row * R34_BLOCK))
        out += [
            f"    aie.objectfifo @xc{col}_{row}(%mt{col}, {{%c{col}_{row}}}, 1 : i32) : !aie.objectfifo<memref<{X_ROW_BYTES}xi8>>",
            f"    aie.objectfifo @oc{col}_{row}(%c{col}_{row}, {{%mt{col}}}, 1 : i32) : !aie.objectfifo<memref<{R34_BLOCK}xi8>>",
        ]
    xoffsets = ", ".join(str(row * X_ROW_BYTES) for row in range(CORE_ROWS))
    out += [
        f"    aie.objectfifo @xsh{col}(%shim{col}, {{%mt{col}}}, 1 : i32) : !aie.objectfifo<memref<{X_JOIN_BYTES}xi8>>",
        f"    aie.objectfifo.link [@xsh{col}] -> [{', '.join(xcores)}] ([] [{xoffsets}])",
        f"    aie.objectfifo @osh{col}(%mt{col}, {{%shim{col}}}, 1 : i32) : !aie.objectfifo<memref<{CORE_ROWS * R34_BLOCK}xi8>>",
        f"    aie.objectfifo.link [{', '.join(outputs)}] -> [@osh{col}] ([{', '.join(offsets)}] [])",
    ]

out.append(
    f'    func.func private @r48_copy_residual_row(memref<{X_ROW_BYTES}xi8>, memref<{R34_BLOCK}xi8>, i32) attributes {{link_with = "r48prep.o"}}'
)

for col in range(COLS):
    for row in range(CORE_ROWS):
        out += [
            f"    %core{col}_{row} = aie.core(%c{col}_{row}) {{",
            "      %z = arith.constant 0 : index",
            f"      %inf = arith.constant {INF} : index",
            "      %one = arith.constant 1 : index",
            f"      %rows = arith.constant {ROWS_PER_CORE} : index",
            "      scf.for %outer = %z to %inf step %one {",
            f"        %o = aie.objectfifo.acquire @oc{col}_{row}(Produce, 1) : !aie.objectfifosubview<memref<{R34_BLOCK}xi8>>",
            f"        %ov = aie.objectfifo.subview.access %o[0] : !aie.objectfifosubview<memref<{R34_BLOCK}xi8>> -> memref<{R34_BLOCK}xi8>",
            "        scf.for %row = %z to %rows step %one {",
            f"          %x = aie.objectfifo.acquire @xc{col}_{row}(Consume, 1) : !aie.objectfifosubview<memref<{X_ROW_BYTES}xi8>>",
            f"          %xv = aie.objectfifo.subview.access %x[0] : !aie.objectfifosubview<memref<{X_ROW_BYTES}xi8>> -> memref<{X_ROW_BYTES}xi8>",
            "          %rowi = arith.index_cast %row : index to i32",
            f"          func.call @r48_copy_residual_row(%xv, %ov, %rowi) : (memref<{X_ROW_BYTES}xi8>, memref<{R34_BLOCK}xi8>, i32) -> ()",
            f"          aie.objectfifo.release @xc{col}_{row}(Consume, 1)",
            "        }",
            f"        aie.objectfifo.release @oc{col}_{row}(Produce, 1)",
            "      }",
            "      aie.end",
            "    } {stack_size = 1024 : i32}",
        ]

out.append(
    f"    aie.runtime_sequence(%X: memref<{COMPLETED_BYTES}xi8>, %O: memref<{OUTPUT_BYTES}xi8>) {{"
)
for col in range(COLS):
    name = f"to{col}"
    wave, core_row = divmod(col, CORE_ROWS)
    offset = R34_INPUT_BYTES + (wave * CORE_ROWS * CORE_ROWS + core_row) * R34_BLOCK
    out += [
        f"      %{name} = aiex.dma_configure_task_for @osh{col} {{",
        f"        aie.dma_bd(%O : memref<{OUTPUT_BYTES}xi8>, {offset}, {CORE_ROWS * R34_BLOCK}, {output_dims()}) {{burst_length = 0 : i32}}",
        "        aie.end",
        "      } {issue_token = true}",
        f"      aiex.dma_start_task(%{name})",
    ]
for col in range(COLS):
    name = f"tx{col}"
    offset = col * CORE_ROWS * ROWS_PER_CORE * COMPLETED_ROW
    out += [
        f"      %{name} = aiex.dma_configure_task_for @xsh{col} {{",
        f"        aie.dma_bd(%X : memref<{COMPLETED_BYTES}xi8>, {offset}, {X_JOIN_BYTES}, {x_dims()}) {{burst_length = 0 : i32}}",
        "        aie.end",
        f"      }} {{issue_token = true, repeat_count = {ROWS_PER_CORE - 1} : i32}}",
        f"      aiex.dma_start_task(%{name})",
    ]
for col in range(COLS):
    for name in (f"tx{col}", f"to{col}"):
        out += [f"      aiex.dma_await_task(%{name})", f"      aiex.dma_free_task(%{name})"]
out += ["    }", "  }", "}"]
print("\n".join(out))
