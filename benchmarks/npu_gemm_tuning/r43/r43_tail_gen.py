#!/usr/bin/env python3
"""Compensated-input/output EmbeddingGemma post-FFN tail on AIE2P."""

import sys

SPLIT_RESIDUAL = "--split-residual" in sys.argv[1:]

COLS, CORE_ROWS, HALVES = 8, 4, 2
PHASES, TOKENS_PER_CORE, HIDDEN = 4, 2, 768
BF16_ROW = HIDDEN * 2
COMPLETED_ROW = 2 * BF16_ROW  # completed high/low, token-major
COMBINED_ROW = HIDDEN * 3 * 2  # FFN high/low, residual
Y_ROW = HIDDEN * 2 * 2 if SPLIT_RESIDUAL else COMBINED_ROW
X_ROW = BF16_ROW
INPUT_TILE = TOKENS_PER_CORE * Y_ROW
X_TILE = TOKENS_PER_CORE * X_ROW
OUTPUT_TILE = TOKENS_PER_CORE * COMPLETED_ROW
INPUT_JOIN = (COLS // HALVES) * INPUT_TILE
X_JOIN = (COLS // HALVES) * X_TILE
OUTPUT_JOIN = (COLS // HALVES) * OUTPUT_TILE
INPUT_BYTES = 288 * COMBINED_ROW
X_BYTES = 288 * X_ROW
OUTPUT_BYTES = 288 * COMPLETED_ROW
PARAM_RECORD = INPUT_TILE
PARAM_BYTES_TOTAL = COLS * CORE_ROWS * PARAM_RECORD
PARAM_BYTES = HIDDEN * 2 + 4
INF = 9223372036854775807


def linear_dims(size):
    return f"[<size = {size // 512}, stride = 512>, <size = 512, stride = 1>]"


out = ["module {", "  aie.device(npu2) {"]
for col in range(COLS):
    out += [f"    %shim{col} = aie.tile({col}, 0)", f"    %mt{col} = aie.tile({col}, 1)"]
    for row in range(CORE_ROWS):
        out += [
            f"    %c{col}_{row} = aie.tile({col}, {row + 2})",
            f"    %params{col}_{row} = aie.buffer(%c{col}_{row}) {{sym_name = \"params{col}_{row}\"}} : memref<{PARAM_BYTES}xi8>",
            *(
                [
                    f"    %xlocal{col}_{row} = aie.buffer(%c{col}_{row}) {{sym_name = \"xlocal{col}_{row}\"}} : memref<{X_TILE}xi8>"
                ]
                if SPLIT_RESIDUAL
                else []
            ),
        ]

for row in range(CORE_ROWS):
    for half in range(HALVES):
        mt = row + half * CORE_ROWS
        first_col = half * (COLS // HALVES)
        consumers, producers = [], []
        input_offsets, output_offsets = [], []
        for local_col in range(COLS // HALVES):
            col = first_col + local_col
            consumers.append(f"@dc{col}_{row}")
            producers.append(f"@oc{col}_{row}")
            input_offsets.append(str(local_col * INPUT_TILE))
            output_offsets.append(str(local_col * OUTPUT_TILE))
            out += [
                f"    aie.objectfifo @dc{col}_{row}(%mt{mt}, {{%c{col}_{row}}}, 1 : i32) : !aie.objectfifo<memref<{INPUT_TILE}xi8>>",
                f"    aie.objectfifo @oc{col}_{row}(%c{col}_{row}, {{%mt{mt}}}, 1 : i32) : !aie.objectfifo<memref<{OUTPUT_TILE}xi8>>",
            ]
        out += [
            f"    aie.objectfifo @dsh{half}_{row}(%shim{mt}, {{%mt{mt}}}, 1 : i32) : !aie.objectfifo<memref<{INPUT_JOIN}xi8>>",
            f"    aie.objectfifo.link [@dsh{half}_{row}] -> [{', '.join(consumers)}] ([] [{', '.join(input_offsets)}])",
            f"    aie.objectfifo @osh{half}_{row}(%mt{mt}, {{%shim{mt}}}, 1 : i32) : !aie.objectfifo<memref<{OUTPUT_JOIN}xi8>>",
            f"    aie.objectfifo.link [{', '.join(producers)}] -> [@osh{half}_{row}] ([{', '.join(output_offsets)}] [])",
        ]
        if SPLIT_RESIDUAL:
            out.append(
                f"    aie.objectfifo @xsc{half}_{row}(%shim{mt}, {{%c{first_col}_{row}}}, 1 : i32) : !aie.objectfifo<memref<{X_JOIN}xi8>>"
            )
            for col in range(first_col, first_col + COLS // HALVES - 1):
                out.append(
                    f'    aie.flow(%c{col}_{row}, "Core" : 0, %c{col + 1}_{row}, "Core" : 0)'
                )

out += [
    f'    func.func private @r43_copy_params(memref<{INPUT_TILE}xi8>, memref<{PARAM_BYTES}xi8>) attributes {{link_with = "r43_tail.o"}}',
    f'    func.func private @r43_post_ffn_direct_tail_bf16x2(memref<{OUTPUT_TILE}xi8>, memref<{INPUT_TILE}xi8>, '
    + (f'memref<{X_TILE}xi8>, ' if SPLIT_RESIDUAL else '')
    + f'memref<{PARAM_BYTES}xi8>) attributes {{link_with = "r43_tail.o"}}',
]
if SPLIT_RESIDUAL:
    out += [
        f'    func.func private @r46_x_source(memref<{X_JOIN}xi8>, memref<{X_TILE}xi8>) attributes {{link_with = "r43_tail.o"}}',
        f'    func.func private @r46_x_relay(memref<{X_TILE}xi8>, i32) attributes {{link_with = "r43_tail.o"}}',
    ]

for col in range(COLS):
    for row in range(CORE_ROWS):
        half = col // (COLS // HALVES)
        local_col = col % (COLS // HALVES)
        out += [
            f"    %core{col}_{row} = aie.core(%c{col}_{row}) {{",
            "      %z = arith.constant 0 : index",
            f"      %inf = arith.constant {INF} : index",
            "      %one = arith.constant 1 : index",
            f"      %phases = arith.constant {PHASES} : index",
            "      scf.for %outer = %z to %inf step %one {",
            f"        %p = aie.objectfifo.acquire @dc{col}_{row}(Consume, 1) : !aie.objectfifosubview<memref<{INPUT_TILE}xi8>>",
            f"        %pv = aie.objectfifo.subview.access %p[0] : !aie.objectfifosubview<memref<{INPUT_TILE}xi8>> -> memref<{INPUT_TILE}xi8>",
            f"        func.call @r43_copy_params(%pv, %params{col}_{row}) : (memref<{INPUT_TILE}xi8>, memref<{PARAM_BYTES}xi8>) -> ()",
            f"        aie.objectfifo.release @dc{col}_{row}(Consume, 1)",
            "        scf.for %phase = %z to %phases step %one {",
            *(
                (
                    [
                        f"          %x = aie.objectfifo.acquire @xsc{half}_{row}(Consume, 1) : !aie.objectfifosubview<memref<{X_JOIN}xi8>>",
                        f"          %xv = aie.objectfifo.subview.access %x[0] : !aie.objectfifosubview<memref<{X_JOIN}xi8>> -> memref<{X_JOIN}xi8>",
                        f"          func.call @r46_x_source(%xv, %xlocal{col}_{row}) : (memref<{X_JOIN}xi8>, memref<{X_TILE}xi8>) -> ()",
                        f"          aie.objectfifo.release @xsc{half}_{row}(Consume, 1)",
                    ]
                    if local_col == 0
                    else [
                        f"          %forward = arith.constant {COLS // HALVES - local_col - 1} : i32",
                        f"          func.call @r46_x_relay(%xlocal{col}_{row}, %forward) : (memref<{X_TILE}xi8>, i32) -> ()",
                    ]
                )
                if SPLIT_RESIDUAL
                else []
            ),
            f"          %o = aie.objectfifo.acquire @oc{col}_{row}(Produce, 1) : !aie.objectfifosubview<memref<{OUTPUT_TILE}xi8>>",
            f"          %ov = aie.objectfifo.subview.access %o[0] : !aie.objectfifosubview<memref<{OUTPUT_TILE}xi8>> -> memref<{OUTPUT_TILE}xi8>",
            f"          %d = aie.objectfifo.acquire @dc{col}_{row}(Consume, 1) : !aie.objectfifosubview<memref<{INPUT_TILE}xi8>>",
            f"          %dv = aie.objectfifo.subview.access %d[0] : !aie.objectfifosubview<memref<{INPUT_TILE}xi8>> -> memref<{INPUT_TILE}xi8>",
            f"          func.call @r43_post_ffn_direct_tail_bf16x2(%ov, %dv, "
            + (f"%xlocal{col}_{row}, " if SPLIT_RESIDUAL else "")
            + f"%params{col}_{row}) : (memref<{OUTPUT_TILE}xi8>, memref<{INPUT_TILE}xi8>, "
            + (f"memref<{X_TILE}xi8>, " if SPLIT_RESIDUAL else "")
            + f"memref<{PARAM_BYTES}xi8>) -> ()",
            f"          aie.objectfifo.release @dc{col}_{row}(Consume, 1)",
            f"          aie.objectfifo.release @oc{col}_{row}(Produce, 1)",
            "        }",
            "      }",
            "      aie.end",
            "    } {stack_size = 2048 : i32}",
        ]

runtime_input = (
    f"%Y: memref<{INPUT_BYTES}xi8>, %X: memref<{X_BYTES}xi8>"
    if SPLIT_RESIDUAL
    else f"%D: memref<{INPUT_BYTES}xi8>"
)
out.append(
    f"    aie.runtime_sequence({runtime_input}, "
    f"%P: memref<{PARAM_BYTES_TOTAL}xi8>, %O: memref<{OUTPUT_BYTES}xi8>) {{"
)
for row in range(CORE_ROWS):
    for half in range(HALVES):
        first_col = half * (COLS // HALVES)
        record_base = row * COLS + first_col
        token_base = half * 128 + row * 32
        pname = f"tp{half}_{row}"
        out += [
            f"      %{pname} = aiex.dma_configure_task_for @dsh{half}_{row} {{",
            f"        aie.dma_bd(%P : memref<{PARAM_BYTES_TOTAL}xi8>, {record_base * PARAM_RECORD}, {INPUT_JOIN}, {linear_dims(INPUT_JOIN)}) {{burst_length = 0 : i32}}",
            "        aie.end",
            "      } {issue_token = true}",
            f"      aiex.dma_start_task(%{pname})",
        ]
        for phase in range(PHASES):
            phase_token = token_base + phase * (COLS // HALVES) * TOKENS_PER_CORE
            iname = f"td{half}_{row}_{phase}"
            oname = f"to{half}_{row}_{phase}"
            y_dims = (
                f"[<size = {COLS // HALVES * TOKENS_PER_CORE}, stride = {COMBINED_ROW}>, "
                f"<size = {Y_ROW}, stride = 1>]"
                if SPLIT_RESIDUAL
                else linear_dims(INPUT_JOIN)
            )
            out += [
                f"      %{iname} = aiex.dma_configure_task_for @dsh{half}_{row} {{",
                f"        aie.dma_bd(%{'Y' if SPLIT_RESIDUAL else 'D'} : memref<{INPUT_BYTES}xi8>, {phase_token * COMBINED_ROW}, {INPUT_JOIN}, {y_dims}) {{burst_length = 0 : i32}}",
                "        aie.end",
                "      } {issue_token = true}",
                f"      aiex.dma_start_task(%{iname})",
                f"      %{oname} = aiex.dma_configure_task_for @osh{half}_{row} {{",
                f"        aie.dma_bd(%O : memref<{OUTPUT_BYTES}xi8>, {phase_token * COMPLETED_ROW}, {OUTPUT_JOIN}, {linear_dims(OUTPUT_JOIN)}) {{burst_length = 0 : i32}}",
                "        aie.end",
                "      } {issue_token = true}",
                f"      aiex.dma_start_task(%{oname})",
            ]
            if SPLIT_RESIDUAL:
                xname = f"tx{half}_{row}_{phase}"
                out[-5:-5] = [
                    f"      %{xname} = aiex.dma_configure_task_for @xsc{half}_{row} {{",
                    f"        aie.dma_bd(%X : memref<{X_BYTES}xi8>, {phase_token * X_ROW}, {X_JOIN}, {linear_dims(X_JOIN)}) {{burst_length = 0 : i32}}",
                    "        aie.end",
                    "      } {issue_token = true}",
                    f"      aiex.dma_start_task(%{xname})",
                ]

for row in range(CORE_ROWS):
    for half in range(HALVES):
        names = [f"tp{half}_{row}"]
        for phase in range(PHASES):
            names += [f"td{half}_{row}_{phase}"]
            if SPLIT_RESIDUAL:
                names += [f"tx{half}_{row}_{phase}"]
            names += [f"to{half}_{row}_{phase}"]
        for name in names:
            out += [f"      aiex.dma_await_task(%{name})", f"      aiex.dma_free_task(%{name})"]
out += ["    }", "  }", "}"]
print("\n".join(out))
