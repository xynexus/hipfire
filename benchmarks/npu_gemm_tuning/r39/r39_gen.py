#!/usr/bin/env python3
"""Resident EmbeddingGemma post-FFN RMSNorm and residual tail on AIE2P."""

COLS, ROWS, HALVES = 8, 4, 2
TOKENS_PER_CORE, HIDDEN = 8, 768
TILE = TOKENS_PER_CORE * HIDDEN * 2
HALF_JOIN = (COLS // HALVES) * TILE
R_STAGE_BYTES = 5 * 48 * 16384
ATT_BYTES = 393216
H_BYTES = R_STAGE_BYTES + ATT_BYTES + 256 * HIDDEN * 4
Y_BYTES = 288 * HIDDEN * 2
P_BYTES = COLS * ROWS * TILE
PARAM_BYTES = HIDDEN * 4 + HIDDEN * 2 + 4
META_BASE = 256 * HIDDEN * 2
ROW_BYTES = HIDDEN * 2
INF = 9223372036854775807


def linear_dims(size):
    return (
        f"[<size = {size // 512}, stride = 512>, "
        "<size = 512, stride = 1>]"
    )


out = ["module {", "  aie.device(npu2) {"]
for col in range(COLS):
    out += [f"    %shim{col} = aie.tile({col}, 0)", f"    %mt{col} = aie.tile({col}, 1)"]
    for row in range(ROWS):
        out += [
            f"    %c{col}_{row} = aie.tile({col}, {row + 2})",
            f"    %inv{col}_{row} = aie.buffer(%c{col}_{row}) {{sym_name = \"inv{col}_{row}\"}} : memref<8xf32>",
            f"    %params{col}_{row} = aie.buffer(%c{col}_{row}) {{sym_name = \"params{col}_{row}\"}} : memref<{PARAM_BYTES}xi8>",
        ]

for row in range(ROWS):
    for half in range(HALVES):
        mt = row + half * ROWS
        first_col = half * (COLS // HALVES)
        consumers = []
        producers = []
        offsets = []
        for local_col in range(COLS // HALVES):
            col = first_col + local_col
            consumers.append(f"@dc{col}_{row}")
            producers.append(f"@oc{col}_{row}")
            offsets.append(str(local_col * TILE))
            out += [
                f"    aie.objectfifo @dc{col}_{row}(%mt{mt}, {{%c{col}_{row}}}, 1 : i32) : !aie.objectfifo<memref<{TILE}xi8>>",
                f"    aie.objectfifo @oc{col}_{row}(%c{col}_{row}, {{%mt{mt}}}, 1 : i32) : !aie.objectfifo<memref<{TILE}xi8>>",
            ]
        out += [
            f"    aie.objectfifo @dsh{half}_{row}(%shim{mt}, {{%mt{mt}}}, 1 : i32) : !aie.objectfifo<memref<{HALF_JOIN}xi8>>",
            f"    aie.objectfifo.link [@dsh{half}_{row}] -> [{', '.join(consumers)}] ([] [{', '.join(offsets)}])",
            f"    aie.objectfifo @osh{half}_{row}(%mt{mt}, {{%shim{mt}}}, 1 : i32) : !aie.objectfifo<memref<{HALF_JOIN}xi8>>",
            f"    aie.objectfifo.link [{', '.join(producers)}] -> [@osh{half}_{row}] ([{', '.join(offsets)}] [])",
        ]

out += [
    f'    func.func private @r39_copy_hidden(memref<{TILE}xi8>, memref<{TILE}xi8>) attributes {{link_with = "r39.o"}}',
    f'    func.func private @r39_copy_inverse(memref<{TILE}xi8>, memref<8xf32>) attributes {{link_with = "r39.o"}}',
    f'    func.func private @r39_copy_params(memref<{TILE}xi8>, memref<{PARAM_BYTES}xi8>) attributes {{link_with = "r39.o"}}',
    f'    func.func private @r39_post_ffn_tail(memref<{TILE}xi8>, memref<{TILE}xi8>, memref<{PARAM_BYTES}xi8>, memref<8xf32>) attributes {{link_with = "r39.o"}}',
]

for col in range(COLS):
    for row in range(ROWS):
        out += [
            f"    %core{col}_{row} = aie.core(%c{col}_{row}) {{",
            "      %z = arith.constant 0 : index",
            f"      %inf = arith.constant {INF} : index",
            "      %one = arith.constant 1 : index",
            "      scf.for %outer = %z to %inf step %one {",
            f"        %m = aie.objectfifo.acquire @dc{col}_{row}(Consume, 1) : !aie.objectfifosubview<memref<{TILE}xi8>>",
            f"        %mv = aie.objectfifo.subview.access %m[0] : !aie.objectfifosubview<memref<{TILE}xi8>> -> memref<{TILE}xi8>",
            f"        func.call @r39_copy_inverse(%mv, %inv{col}_{row}) : (memref<{TILE}xi8>, memref<8xf32>) -> ()",
            f"        aie.objectfifo.release @dc{col}_{row}(Consume, 1)",
            f"        %p = aie.objectfifo.acquire @dc{col}_{row}(Consume, 1) : !aie.objectfifosubview<memref<{TILE}xi8>>",
            f"        %pv = aie.objectfifo.subview.access %p[0] : !aie.objectfifosubview<memref<{TILE}xi8>> -> memref<{TILE}xi8>",
            f"        func.call @r39_copy_params(%pv, %params{col}_{row}) : (memref<{TILE}xi8>, memref<{PARAM_BYTES}xi8>) -> ()",
            f"        aie.objectfifo.release @dc{col}_{row}(Consume, 1)",
            f"        %o = aie.objectfifo.acquire @oc{col}_{row}(Produce, 1) : !aie.objectfifosubview<memref<{TILE}xi8>>",
            f"        %ov = aie.objectfifo.subview.access %o[0] : !aie.objectfifosubview<memref<{TILE}xi8>> -> memref<{TILE}xi8>",
            f"        %h = aie.objectfifo.acquire @dc{col}_{row}(Consume, 1) : !aie.objectfifosubview<memref<{TILE}xi8>>",
            f"        %hv = aie.objectfifo.subview.access %h[0] : !aie.objectfifosubview<memref<{TILE}xi8>> -> memref<{TILE}xi8>",
            f"        func.call @r39_copy_hidden(%hv, %ov) : (memref<{TILE}xi8>, memref<{TILE}xi8>) -> ()",
            f"        aie.objectfifo.release @dc{col}_{row}(Consume, 1)",
            f"        %y = aie.objectfifo.acquire @dc{col}_{row}(Consume, 1) : !aie.objectfifosubview<memref<{TILE}xi8>>",
            f"        %yv = aie.objectfifo.subview.access %y[0] : !aie.objectfifosubview<memref<{TILE}xi8>> -> memref<{TILE}xi8>",
            f"        func.call @r39_post_ffn_tail(%ov, %yv, %params{col}_{row}, %inv{col}_{row}) : (memref<{TILE}xi8>, memref<{TILE}xi8>, memref<{PARAM_BYTES}xi8>, memref<8xf32>) -> ()",
            f"        aie.objectfifo.release @dc{col}_{row}(Consume, 1)",
            f"        aie.objectfifo.release @oc{col}_{row}(Produce, 1)",
            "      }",
            "      aie.end",
            "    } {stack_size = 2048 : i32}",
        ]

out.append(
    f"    aie.runtime_sequence(%H: memref<{H_BYTES}xi8>, %Y: memref<{Y_BYTES}xi8>, "
    f"%P: memref<{P_BYTES}xi8>, %O: memref<{Y_BYTES}xi8>) {{"
)
for row in range(ROWS):
    for half in range(HALVES):
        first_col = half * (COLS // HALVES)
        token_base = half * 128 + row * 32
        record_base = row * COLS + first_col
        for stem, buffer, offset, bytes_ in [
            ("m", "H", META_BASE + record_base * TILE, H_BYTES),
            ("p", "P", record_base * TILE, P_BYTES),
            ("h", "H", token_base * ROW_BYTES, H_BYTES),
            ("y", "Y", token_base * ROW_BYTES, Y_BYTES),
        ]:
            name = f"t{stem}{half}_{row}"
            out += [
                f"      %{name} = aiex.dma_configure_task_for @dsh{half}_{row} {{",
                f"        aie.dma_bd(%{buffer} : memref<{bytes_}xi8>, {offset}, {HALF_JOIN}, {linear_dims(HALF_JOIN)}) {{burst_length = 0 : i32}}",
                "        aie.end",
                "      } {issue_token = true}",
                f"      aiex.dma_start_task(%{name})",
            ]
        out += [
            f"      %to{half}_{row} = aiex.dma_configure_task_for @osh{half}_{row} {{",
            f"        aie.dma_bd(%O : memref<{Y_BYTES}xi8>, {token_base * ROW_BYTES}, {HALF_JOIN}, {linear_dims(HALF_JOIN)}) {{burst_length = 0 : i32}}",
            "        aie.end",
            "      } {issue_token = true}",
            f"      aiex.dma_start_task(%to{half}_{row})",
        ]

for row in range(ROWS):
    for half in range(HALVES):
        for name in (
            f"tm{half}_{row}",
            f"tp{half}_{row}",
            f"th{half}_{row}",
            f"ty{half}_{row}",
            f"to{half}_{row}",
        ):
            out += [f"      aiex.dma_await_task(%{name})", f"      aiex.dma_free_task(%{name})"]
out += ["    }", "  }", "}"]
print("\n".join(out))
