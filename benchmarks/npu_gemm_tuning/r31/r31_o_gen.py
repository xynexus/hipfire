#!/usr/bin/env python3
"""Generate the full-array BF16 output projection consuming R30 layout."""

COLS, ROWS = 8, 4
A_BLOCK = W_BLOCK = 16384
OUT_TILE, OUT_JOIN = 4096, 16384
ACC_ELEMS = 1024
GROUPS, M_WAVES, N_SLICES = 3, 2, 3
A_BYTES = 393216
W_BYTES = COLS * GROUPS * N_SLICES * W_BLOCK
O_BYTES = 256 * 768 * 4
INF = 9223372036854775807


def attention_group_dims():
    return "[<size = 1, stride = 16384>, <size = 32, stride = 512>, <size = 512, stride = 1>]"


def weight_dims():
    return (
        f"[<size = {GROUPS * N_SLICES}, stride = {W_BLOCK}>, "
        f"<size = {W_BLOCK // 512}, stride = 512>, <size = 512, stride = 1>]"
    )


def output_dims():
    return (
        "[<size = 4, stride = 98304>, <size = 32, stride = 3072>, "
        "<size = 128, stride = 1>]"
    )


out = ["module {", "  aie.device(npu2) {"]
for col in range(COLS):
    out += [f"    %shim{col} = aie.tile({col}, 0)", f"    %mt{col} = aie.tile({col}, 1)"]
    for row in range(ROWS):
        out += [
            f"    %c{col}_{row} = aie.tile({col}, {row + 2})",
            f'    %acc{col}_{row} = aie.buffer(%c{col}_{row}) {{sym_name = "acc{col}_{row}"}} : memref<{ACC_ELEMS}xf32>',
        ]

for row in range(ROWS):
    cores = ", ".join(f"%c{col}_{row}" for col in range(COLS))
    out += [
        f"    aie.objectfifo @ash{row}(%shim{row}, {{%mt{row}}}, 1 : i32) : !aie.objectfifo<memref<{A_BLOCK}xi8>>",
        f"    aie.objectfifo @abc{row}(%mt{row}, {{{cores}}}, 1 : i32) : !aie.objectfifo<memref<{A_BLOCK}xi8>>",
        f"    aie.objectfifo.link [@ash{row}] -> [@abc{row}] ([] [0])",
    ]

for col in range(COLS):
    cores = ", ".join(f"%c{col}_{row}" for row in range(ROWS))
    out += [
        f"    aie.objectfifo @wsh{col}(%shim{col}, {{%mt{col}}}, 1 : i32) : !aie.objectfifo<memref<{W_BLOCK}xi8>>",
        f"    aie.objectfifo @wbc{col}(%mt{col}, {{{cores}}}, 1 : i32) : !aie.objectfifo<memref<{W_BLOCK}xi8>>",
        f"    aie.objectfifo.link [@wsh{col}] -> [@wbc{col}] ([] [0])",
    ]
    producers = ", ".join(f"@oc{col}_{row}" for row in range(ROWS))
    offsets = ", ".join(str(row * OUT_TILE) for row in range(ROWS))
    for row in range(ROWS):
        out += [
            f"    aie.objectfifo @oc{col}_{row}(%c{col}_{row}, {{%mt{col}}}, 1 : i32) : !aie.objectfifo<memref<{OUT_TILE}xi8>>"
        ]
    out += [
        f"    aie.objectfifo @osh{col}(%mt{col}, {{%shim{col}}}, 1 : i32) : !aie.objectfifo<memref<{OUT_JOIN}xi8>>",
        f"    aie.objectfifo.link [{producers}] -> [@osh{col}] ([{offsets}] [])",
    ]

out += [
    f'    func.func private @r31_output_projection_group(memref<{A_BLOCK}xi8>, memref<{W_BLOCK}xi8>, memref<{ACC_ELEMS}xf32>, i32) attributes {{link_with = "r31.o"}}',
    f'    func.func private @r31_output_projection_finish(memref<{ACC_ELEMS}xf32>, memref<{OUT_TILE}xi8>) attributes {{link_with = "r31.o"}}',
]

for col in range(COLS):
    for row in range(ROWS):
        out += [
            f"    %core{col}_{row} = aie.core(%c{col}_{row}) {{",
            "      %z = arith.constant 0 : index",
            f"      %inf = arith.constant {INF} : index",
            "      %one = arith.constant 1 : index",
            f"      %mwaves = arith.constant {M_WAVES} : index",
            f"      %slices = arith.constant {N_SLICES} : index",
            f"      %groups = arith.constant {GROUPS} : index",
            "      scf.for %outer = %z to %inf step %one {",
            "        scf.for %mwave = %z to %mwaves step %one {",
            "          scf.for %slice = %z to %slices step %one {",
            "            scf.for %group = %z to %groups step %one {",
            "              %groupi = arith.index_cast %group : index to i32",
            f"              %a = aie.objectfifo.acquire @abc{row}(Consume, 1) : !aie.objectfifosubview<memref<{A_BLOCK}xi8>>",
            f"              %av = aie.objectfifo.subview.access %a[0] : !aie.objectfifosubview<memref<{A_BLOCK}xi8>> -> memref<{A_BLOCK}xi8>",
            f"              %w = aie.objectfifo.acquire @wbc{col}(Consume, 1) : !aie.objectfifosubview<memref<{W_BLOCK}xi8>>",
            f"              %wv = aie.objectfifo.subview.access %w[0] : !aie.objectfifosubview<memref<{W_BLOCK}xi8>> -> memref<{W_BLOCK}xi8>",
            f"              func.call @r31_output_projection_group(%av, %wv, %acc{col}_{row}, %groupi) : (memref<{A_BLOCK}xi8>, memref<{W_BLOCK}xi8>, memref<{ACC_ELEMS}xf32>, i32) -> ()",
            f"              aie.objectfifo.release @abc{row}(Consume, 1)",
            f"              aie.objectfifo.release @wbc{col}(Consume, 1)",
            "            }",
            f"            %o = aie.objectfifo.acquire @oc{col}_{row}(Produce, 1) : !aie.objectfifosubview<memref<{OUT_TILE}xi8>>",
            f"            %ov = aie.objectfifo.subview.access %o[0] : !aie.objectfifosubview<memref<{OUT_TILE}xi8>> -> memref<{OUT_TILE}xi8>",
            f"            func.call @r31_output_projection_finish(%acc{col}_{row}, %ov) : (memref<{ACC_ELEMS}xf32>, memref<{OUT_TILE}xi8>) -> ()",
            f"            aie.objectfifo.release @oc{col}_{row}(Produce, 1)",
            "          }",
            "        }",
            "      }",
            "      aie.end",
            "    } {stack_size = 2048 : i32}",
        ]

out += [
    f"    aie.runtime_sequence(%A: memref<{A_BYTES}xi8>, %W: memref<{W_BYTES}xi8>, %O: memref<{O_BYTES}xi8>) {{"
]
for col in range(COLS):
    out += [
        f"      %tw{col} = aiex.dma_configure_task_for @wsh{col} {{",
        f"        aie.dma_bd(%W : memref<{W_BYTES}xi8>, {col * GROUPS * N_SLICES * W_BLOCK}, {GROUPS * N_SLICES * W_BLOCK}, {weight_dims()}) {{burst_length = 0 : i32}}",
        "        aie.end",
        f"      }} {{issue_token = true, repeat_count = {M_WAVES - 1} : i32}}",
        f"      aiex.dma_start_task(%tw{col})",
    ]

for mwave in range(M_WAVES):
    for nslice in range(N_SLICES):
        for col in range(COLS):
            name = f"to{mwave}_{nslice}_{col}"
            offset = mwave * 128 * 768 * 4 + (col * 96 + nslice * 32) * 4
            out += [
                f"      %{name} = aiex.dma_configure_task_for @osh{col} {{",
                f"        aie.dma_bd(%O : memref<{O_BYTES}xi8>, {offset}, {OUT_JOIN}, {output_dims()}) {{burst_length = 0 : i32}}",
                "        aie.end",
                "      } {issue_token = true}",
                f"      aiex.dma_start_task(%{name})",
            ]
        for group in range(GROUPS):
            attention_group = group * M_WAVES + mwave
            for row in range(ROWS):
                name = f"ta{mwave}_{nslice}_{group}_{row}"
                offset = (attention_group * ROWS + row) * A_BLOCK
                out += [
                    f"      %{name} = aiex.dma_configure_task_for @ash{row} {{",
                    f"        aie.dma_bd(%A : memref<{A_BYTES}xi8>, {offset}, {A_BLOCK}, {attention_group_dims()}) {{burst_length = 0 : i32}}",
                    "        aie.end",
                    "      } {issue_token = true}",
                    f"      aiex.dma_start_task(%{name})",
                ]
            for row in range(ROWS):
                name = f"ta{mwave}_{nslice}_{group}_{row}"
                out += [f"      aiex.dma_await_task(%{name})", f"      aiex.dma_free_task(%{name})"]
        for col in range(COLS):
            name = f"to{mwave}_{nslice}_{col}"
            out += [f"      aiex.dma_await_task(%{name})", f"      aiex.dma_free_task(%{name})"]

for col in range(COLS):
    out += [f"      aiex.dma_await_task(%tw{col})", f"      aiex.dma_free_task(%tw{col})"]
out += ["    }", "  }", "}"]
print("\n".join(out))
