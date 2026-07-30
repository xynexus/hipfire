#!/usr/bin/env python3
"""Stage R113 chunks once and stream full-K N32 output blocks."""

import sys

REPEAT_OUTPUT_TASK = "--repeat-output-task" in sys.argv[1:]
N_BLOCKS = 2
BATCH = 1
# K groups of 256 staged per full-K dispatch. 3 (= k768) is EmbeddingGemma's
# shape and stays the default; llama-3.2-1B needs 8 (k2048) and 32 (k8192).
GROUPS_ARG = 3
for arg in sys.argv[1:]:
    if arg.startswith("--n-blocks="):
        N_BLOCKS = int(arg.split("=", 1)[1])
    elif arg.startswith("--batch="):
        BATCH = int(arg.split("=", 1)[1])
    elif arg.startswith("--groups="):
        GROUPS_ARG = int(arg.split("=", 1)[1])
    elif arg != "--repeat-output-task":
        raise SystemExit(f"unknown argument: {arg}")
if GROUPS_ARG < 1:
    raise SystemExit("--groups must be positive")
if N_BLOCKS < 1:
    raise SystemExit("--n-blocks must be positive")
if BATCH < 1:
    raise SystemExit("--batch must be positive")
if N_BLOCKS != 2 and not REPEAT_OUTPUT_TASK:
    raise SystemExit("non-default --n-blocks requires --repeat-output-task")

COLS, ROWS = 8, 4
HALVES, GROUPS = 2, GROUPS_ARG
A_SLOT, A_JOIN = 6144, 4 * 6144
A_STAGE = 3 * 2112
W_RECORD = 8320
W_RECORDS_PER_COL = GROUPS * N_BLOCKS
OUT_SLOT = 8 * 32 * 4
OUT_JOIN = 4 * OUT_SLOT
RUNTIME_OUT_SLOT = OUT_SLOT if BATCH == 1 else BATCH * OUT_SLOT
RUNTIME_OUT_JOIN = OUT_JOIN if BATCH == 1 else BATCH * OUT_JOIN
A_DOCUMENT_BYTES = 4 * 2 * GROUPS * A_JOIN
A_BYTES = BATCH * A_DOCUMENT_BYTES
W_BYTES = COLS * W_RECORDS_PER_COL * W_RECORD
N = N_BLOCKS * 32
O_DOCUMENT_BYTES = 256 * N * 4
O_BYTES = BATCH * O_DOCUMENT_BYTES
INF = 9223372036854775807

out = ["module {", "  aie.device(npu2) {"]
for col in range(COLS):
    out += [f"    %shim{col} = aie.tile({col}, 0)", f"    %mt{col} = aie.tile({col}, 1)"]
    for row in range(ROWS):
        out.append(f"    %c{col}_{row} = aie.tile({col}, {row + 2})")
        if BATCH == 1:
            out.append(
                f'    %astage{col}_{row} = aie.buffer(%c{col}_{row}) {{sym_name = "astage{col}_{row}"}} : memref<{A_STAGE}xi8>'
            )
        else:
            for document in range(BATCH):
                out.append(
                    f'    %astage{col}_{row}_{document} = aie.buffer(%c{col}_{row}) {{sym_name = "astage{col}_{row}_{document}"}} : memref<{A_STAGE}xi8>'
                )

for row in range(ROWS):
    for half in range(HALVES):
        mt = row + half * ROWS
        first_col = half * 4
        a_consumers, a_offsets, o_producers = [], [], []
        for local_col in range(4):
            col = first_col + local_col
            a_consumers.append(f"@ac{col}_{row}")
            a_offsets.append(str(local_col * A_SLOT))
            o_producers.append(f"@oc{col}_{row}")
            out += [
                f"    aie.objectfifo @ac{col}_{row}(%mt{mt}, {{%c{col}_{row}}}, 1 : i32) : !aie.objectfifo<memref<{A_SLOT}xi8>>",
                f"    aie.objectfifo @oc{col}_{row}(%c{col}_{row}, {{%mt{mt}}}, 1 : i32) : !aie.objectfifo<memref<{RUNTIME_OUT_SLOT}xi8>>",
            ]
        out += [
            f"    aie.objectfifo @ash{half}_{row}(%shim{mt}, {{%mt{mt}}}, 1 : i32) : !aie.objectfifo<memref<{A_JOIN}xi8>>",
            f"    aie.objectfifo.link [@ash{half}_{row}] -> [{', '.join(a_consumers)}] ([] [{', '.join(a_offsets)}])",
            f"    aie.objectfifo @osh{half}_{row}(%mt{mt}, {{%shim{mt}}}, 1 : i32) : !aie.objectfifo<memref<{RUNTIME_OUT_JOIN}xi8>>",
            f"    aie.objectfifo.link [{', '.join(o_producers)}] -> [@osh{half}_{row}] ([0, {RUNTIME_OUT_SLOT}, {2 * RUNTIME_OUT_SLOT}, {3 * RUNTIME_OUT_SLOT}] [])",
        ]

for col in range(COLS):
    consumers = ", ".join(f"%c{col}_{row}" for row in range(ROWS))
    out += [
        f"    aie.objectfifo @wsh{col}(%shim{col}, {{%mt{col}}}, 1 : i32) : !aie.objectfifo<memref<{W_RECORD}xi8>>",
        f"    aie.objectfifo @wbc{col}(%mt{col}, {{{consumers}}}, 1 : i32) : !aie.objectfifo<memref<{W_RECORD}xi8>>",
        f"    aie.objectfifo.link [@wsh{col}] -> [@wbc{col}] ([] [0])",
    ]

out += [
    f'    func.func private @r118_stage_chunk(memref<{A_SLOT}xi8>, memref<{A_STAGE}xi8>, i32) attributes {{link_with = "r118.o"}}',
]
if BATCH == 1:
    out.append(
        f'    func.func private @r118_compact_group_n32(memref<{A_STAGE}xi8>, memref<{W_RECORD}xi8>, memref<{OUT_SLOT}xi8>, i32) attributes {{link_with = "r118.o"}}'
    )
else:
    if BATCH not in (2, 4):
        raise SystemExit("batched joined-output staged full-K supports BATCH=2 or BATCH=4")
    stage_arguments = ", ".join(f"memref<{A_STAGE}xi8>" for _ in range(BATCH))
    out.append(
        f'    func.func private @r129_compact_group_n32_b{BATCH}({stage_arguments}, memref<{W_RECORD}xi8>, memref<{RUNTIME_OUT_SLOT}xi8>, i32) attributes {{link_with = "r118.o"}}'
    )

for col in range(COLS):
    for row in range(ROWS):
        if BATCH == 1:
            out += [
                f"    %core{col}_{row} = aie.core(%c{col}_{row}) {{",
                f"      %inf = arith.constant {INF} : index",
                "      %z = arith.constant 0 : index",
                "      %one = arith.constant 1 : index",
                f"      %groups = arith.constant {GROUPS} : index",
                f"      %nblocks = arith.constant {N_BLOCKS} : index",
                "      scf.for %outer = %z to %inf step %one {",
                "        scf.for %group = %z to %groups step %one {",
                f"          %a = aie.objectfifo.acquire @ac{col}_{row}(Consume, 1) : !aie.objectfifosubview<memref<{A_SLOT}xi8>>",
                f"          %av = aie.objectfifo.subview.access %a[0] : !aie.objectfifosubview<memref<{A_SLOT}xi8>> -> memref<{A_SLOT}xi8>",
                "          %groupi = arith.index_cast %group : index to i32",
                f"          func.call @r118_stage_chunk(%av, %astage{col}_{row}, %groupi) : (memref<{A_SLOT}xi8>, memref<{A_STAGE}xi8>, i32) -> ()",
                f"          aie.objectfifo.release @ac{col}_{row}(Consume, 1)",
                "        }",
                "        scf.for %nblock = %z to %nblocks step %one {",
                f"          %o = aie.objectfifo.acquire @oc{col}_{row}(Produce, 1) : !aie.objectfifosubview<memref<{OUT_SLOT}xi8>>",
                f"          %ov = aie.objectfifo.subview.access %o[0] : !aie.objectfifosubview<memref<{OUT_SLOT}xi8>> -> memref<{OUT_SLOT}xi8>",
                "          scf.for %group = %z to %groups step %one {",
                f"            %w = aie.objectfifo.acquire @wbc{col}(Consume, 1) : !aie.objectfifosubview<memref<{W_RECORD}xi8>>",
                f"            %wv = aie.objectfifo.subview.access %w[0] : !aie.objectfifosubview<memref<{W_RECORD}xi8>> -> memref<{W_RECORD}xi8>",
                "            %groupi = arith.index_cast %group : index to i32",
                f"            func.call @r118_compact_group_n32(%astage{col}_{row}, %wv, %ov, %groupi) : (memref<{A_STAGE}xi8>, memref<{W_RECORD}xi8>, memref<{OUT_SLOT}xi8>, i32) -> ()",
                f"            aie.objectfifo.release @wbc{col}(Consume, 1)",
                "          }",
                f"          aie.objectfifo.release @oc{col}_{row}(Produce, 1)",
                "        }",
                "      }",
                "      aie.end",
                "    } {stack_size = 3072 : i32}",
            ]
            continue

        out += [
            f"    %core{col}_{row} = aie.core(%c{col}_{row}) {{",
            f"      %inf = arith.constant {INF} : index",
            "      %z = arith.constant 0 : index",
            "      %one = arith.constant 1 : index",
            f"      %groups = arith.constant {GROUPS} : index",
            f"      %nblocks = arith.constant {N_BLOCKS} : index",
            "      scf.for %outer = %z to %inf step %one {",
        ]
        for document in range(BATCH):
            out += [
                f"        scf.for %group{document} = %z to %groups step %one {{",
                f"          %a{document} = aie.objectfifo.acquire @ac{col}_{row}(Consume, 1) : !aie.objectfifosubview<memref<{A_SLOT}xi8>>",
                f"          %av{document} = aie.objectfifo.subview.access %a{document}[0] : !aie.objectfifosubview<memref<{A_SLOT}xi8>> -> memref<{A_SLOT}xi8>",
                f"          %groupi{document} = arith.index_cast %group{document} : index to i32",
                f"          func.call @r118_stage_chunk(%av{document}, %astage{col}_{row}_{document}, %groupi{document}) : (memref<{A_SLOT}xi8>, memref<{A_STAGE}xi8>, i32) -> ()",
                f"          aie.objectfifo.release @ac{col}_{row}(Consume, 1)",
                "        }",
            ]
        out += [
            "        scf.for %nblock = %z to %nblocks step %one {",
            f"          %o = aie.objectfifo.acquire @oc{col}_{row}(Produce, 1) : !aie.objectfifosubview<memref<{RUNTIME_OUT_SLOT}xi8>>",
            f"          %ov = aie.objectfifo.subview.access %o[0] : !aie.objectfifosubview<memref<{RUNTIME_OUT_SLOT}xi8>> -> memref<{RUNTIME_OUT_SLOT}xi8>",
            "          scf.for %group = %z to %groups step %one {",
            f"            %w = aie.objectfifo.acquire @wbc{col}(Consume, 1) : !aie.objectfifosubview<memref<{W_RECORD}xi8>>",
            f"            %wv = aie.objectfifo.subview.access %w[0] : !aie.objectfifosubview<memref<{W_RECORD}xi8>> -> memref<{W_RECORD}xi8>",
            "            %groupi = arith.index_cast %group : index to i32",
            f"            func.call @r129_compact_group_n32_b{BATCH}({', '.join(f'%astage{col}_{row}_{document}' for document in range(BATCH))}, %wv, %ov, %groupi) : ({', '.join(f'memref<{A_STAGE}xi8>' for _ in range(BATCH))}, memref<{W_RECORD}xi8>, memref<{RUNTIME_OUT_SLOT}xi8>, i32) -> ()",
        ]
        out += [
            f"            aie.objectfifo.release @wbc{col}(Consume, 1)",
            "          }",
            f"          aie.objectfifo.release @oc{col}_{row}(Produce, 1)",
            "        }",
            "      }",
            "      aie.end",
            "    } {stack_size = 3072 : i32}",
        ]

out.append(
    f"    aie.runtime_sequence(%A: memref<{A_BYTES}xi8>, %W: memref<{W_BYTES}xi8>, %O: memref<{O_BYTES}xi8>) {{"
)
activation_tasks, output_tasks = [], []
for row in range(ROWS):
    for half in range(HALVES):
        record = (row * HALVES + half) * GROUPS
        token_base = half * 128 + row * 32
        aname = f"ta{half}_{row}"
        activation_tasks.append(aname)
        activation_dimensions = (
            f"[<size = {48 * GROUPS}, stride = 512>, <size = 512, stride = 1>]"
            if BATCH == 1
            else f"[<size = {BATCH}, stride = {A_DOCUMENT_BYTES}>, <size = 1, stride = 4>, <size = {48 * GROUPS}, stride = 512>, <size = 512, stride = 1>]"
        )
        activation_attrs = (
            " {issue_token = true}"
            if BATCH == 1
            else f" {{issue_token = true, repeat_count = {BATCH - 1} : i32}}"
        )
        out += [
            f"      %{aname} = aiex.dma_configure_task_for @ash{half}_{row} {{",
            f"        aie.dma_bd(%A : memref<{A_BYTES}xi8>, {record * A_JOIN}, {GROUPS * A_JOIN}, {activation_dimensions}) {{burst_length = 0 : i32}}",
            "        aie.end",
            f"      }}{activation_attrs}",
            f"      aiex.dma_start_task(%{aname})",
        ]
        if BATCH > 1:
            stream = row * HALVES + half
            oname = f"to{half}_{row}_batch"
            output_tasks.append(oname)
            offset = stream * N_BLOCKS * RUNTIME_OUT_JOIN
            out += [
                f"      %{oname} = aiex.dma_configure_task_for @osh{half}_{row} {{",
                f"        aie.dma_bd(%O : memref<{O_BYTES}xi8>, {offset}, {RUNTIME_OUT_JOIN}, [<size = {N_BLOCKS}, stride = {RUNTIME_OUT_JOIN}>, <size = 1, stride = 4>, <size = {RUNTIME_OUT_JOIN // 512}, stride = 512>, <size = 512, stride = 1>]) {{burst_length = 0 : i32}}",
                "        aie.end",
                f"      }} {{issue_token = true, repeat_count = {N_BLOCKS - 1} : i32}}",
                f"      aiex.dma_start_task(%{oname})",
            ]
            continue
        output_variants = [None] if REPEAT_OUTPUT_TASK else range(N_BLOCKS)
        for n_block in output_variants:
            oname = (
                f"to{half}_{row}_repeat"
                if n_block is None
                else f"to{half}_{row}_{n_block}"
            )
            output_tasks.append(oname)
            offset = token_base * N * 4 + (0 if n_block is None else n_block * 32 * 4)
            dimensions = (
                f"[<size = {N_BLOCKS}, stride = 128>, <size = 4, stride = {8 * N * 4}>, <size = 8, stride = {N * 4}>, <size = 128, stride = 1>]"
                if n_block is None
                else f"[<size = 4, stride = {8 * N * 4}>, <size = 8, stride = {N * 4}>, <size = 128, stride = 1>]"
            )
            task_attrs = (
                f" {{issue_token = true, repeat_count = {N_BLOCKS - 1} : i32}}"
                if n_block is None
                else " {issue_token = true}"
            )
            out += [
                f"      %{oname} = aiex.dma_configure_task_for @osh{half}_{row} {{",
                f"        aie.dma_bd(%O : memref<{O_BYTES}xi8>, {offset}, {OUT_JOIN}, {dimensions}) {{burst_length = 0 : i32}}",
                "        aie.end",
                f"      }}{task_attrs}",
                f"      aiex.dma_start_task(%{oname})",
            ]

weight_tasks = []
for col in range(COLS):
    name = f"tw{col}"
    weight_tasks.append(name)
    out += [
        f"      %{name} = aiex.dma_configure_task_for @wsh{col} {{",
        f"        aie.dma_bd(%W : memref<{W_BYTES}xi8>, {col * W_RECORDS_PER_COL * W_RECORD}, {W_RECORDS_PER_COL * W_RECORD}, [<size = {W_RECORDS_PER_COL * W_RECORD // 32}, stride = 32>, <size = 32, stride = 1>]) {{burst_length = 0 : i32}}",
        "        aie.end",
        "      } {issue_token = true}",
        f"      aiex.dma_start_task(%{name})",
    ]

for name in output_tasks + activation_tasks + weight_tasks:
    out += [f"      aiex.dma_await_task(%{name})", f"      aiex.dma_free_task(%{name})"]
out += ["    }", "  }", "}"]
print("\n".join(out))
