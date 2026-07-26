#!/usr/bin/env python3
"""Generate direct consumers of R113 per-core compact chunks."""

import sys

active_arg = next(
    (arg for arg in sys.argv[1:] if arg.startswith("--active-groups=")), None
)
n_slices_arg = next(
    (arg for arg in sys.argv[1:] if arg.startswith("--n-slices=")), None
)
ACTIVE_GROUPS = (
    int(active_arg.split("=", 1)[1])
    if active_arg is not None
    else 3
    if "--all-groups" in sys.argv[1:]
    else 1
)
if ACTIVE_GROUPS not in (1, 2, 3):
    raise SystemExit("--active-groups must be 1, 2, or 3")
ALL_GROUPS = ACTIVE_GROUPS > 1
N_SLICES = int(n_slices_arg.split("=", 1)[1]) if n_slices_arg else 1
if N_SLICES not in (1, 2):
    raise SystemExit("--n-slices must be 1 or 2")
if N_SLICES != 1 and not ALL_GROUPS:
    raise SystemExit("wider N slices require the full-K consumer")

COLS, ROWS = 8, 4
HALVES, GROUPS = 2, 3
A_SLOT = 6144
A_JOIN = 4 * A_SLOT
W_RECORD = 4160 * N_SLICES
OUT_SLOT = 8 * 16 * N_SLICES * 4
OUT_JOIN = 4 * OUT_SLOT
A_BYTES = 4 * 2 * GROUPS * A_JOIN
W_BYTES = COLS * ACTIVE_GROUPS * W_RECORD
O_BYTES = 256 * 16 * N_SLICES * 4
INF = 9223372036854775807

out = ["module {", "  aie.device(npu2) {"]
for col in range(COLS):
    out += [f"    %shim{col} = aie.tile({col}, 0)", f"    %mt{col} = aie.tile({col}, 1)"]
    for row in range(ROWS):
        out.append(f"    %c{col}_{row} = aie.tile({col}, {row + 2})")

for row in range(ROWS):
    for half in range(HALVES):
        mt = row + half * ROWS
        first_col = half * 4
        consumers = []
        offsets = []
        producers = []
        for local_col in range(4):
            col = first_col + local_col
            consumers.append(f"@ac{col}_{row}")
            offsets.append(str(local_col * A_SLOT))
            producers.append(f"@oc{col}_{row}")
            out.append(
                f"    aie.objectfifo @ac{col}_{row}(%mt{mt}, {{%c{col}_{row}}}, 1 : i32) : !aie.objectfifo<memref<{A_SLOT}xi8>>"
            )
            out.append(
                f"    aie.objectfifo @oc{col}_{row}(%c{col}_{row}, {{%mt{mt}}}, 1 : i32) : !aie.objectfifo<memref<{OUT_SLOT}xi8>>"
            )
        out += [
            f"    aie.objectfifo @ash{half}_{row}(%shim{mt}, {{%mt{mt}}}, 1 : i32) : !aie.objectfifo<memref<{A_JOIN}xi8>>",
            f"    aie.objectfifo.link [@ash{half}_{row}] -> [{', '.join(consumers)}] ([] [{', '.join(offsets)}])",
            f"    aie.objectfifo @osh{half}_{row}(%mt{mt}, {{%shim{mt}}}, 1 : i32) : !aie.objectfifo<memref<{OUT_JOIN}xi8>>",
            f"    aie.objectfifo.link [{', '.join(producers)}] -> [@osh{half}_{row}] ([0, {OUT_SLOT}, {2 * OUT_SLOT}, {3 * OUT_SLOT}] [])",
        ]

for col in range(COLS):
    consumers = ", ".join(f"%c{col}_{row}" for row in range(ROWS))
    out += [
        f"    aie.objectfifo @wsh{col}(%shim{col}, {{%mt{col}}}, 1 : i32) : !aie.objectfifo<memref<{W_RECORD}xi8>>",
        f"    aie.objectfifo @wbc{col}(%mt{col}, {{{consumers}}}, 1 : i32) : !aie.objectfifo<memref<{W_RECORD}xi8>>",
        f"    aie.objectfifo.link [@wsh{col}] -> [@wbc{col}] ([] [0])",
    ]

if N_SLICES == 2:
    out.append(
        f'    func.func private @r117_compact_group_n32(memref<{A_SLOT}xi8>, memref<{W_RECORD}xi8>, memref<{OUT_SLOT}xi8>, i32) attributes {{link_with = "r117.o"}}'
    )
elif ALL_GROUPS:
    out.append(
        f'    func.func private @r116_compact_group_n16(memref<{A_SLOT}xi8>, memref<{W_RECORD}xi8>, memref<{OUT_SLOT}xi8>, i32) attributes {{link_with = "r116.o"}}'
    )
else:
    out.append(
        f'    func.func private @r115_compact_group_n16(memref<{A_SLOT}xi8>, memref<{W_RECORD}xi8>, memref<{OUT_SLOT}xi8>) attributes {{link_with = "r115.o"}}'
    )

for col in range(COLS):
    half = col // 4
    for row in range(ROWS):
        if not ALL_GROUPS:
            out += [
                f"    %core{col}_{row} = aie.core(%c{col}_{row}) {{",
                f"      %inf = arith.constant {INF} : index",
                "      %z = arith.constant 0 : index",
                "      %one = arith.constant 1 : index",
                "      scf.for %outer = %z to %inf step %one {",
                f"        %a = aie.objectfifo.acquire @ac{col}_{row}(Consume, 1) : !aie.objectfifosubview<memref<{A_SLOT}xi8>>",
                f"        %av = aie.objectfifo.subview.access %a[0] : !aie.objectfifosubview<memref<{A_SLOT}xi8>> -> memref<{A_SLOT}xi8>",
                f"        %w = aie.objectfifo.acquire @wbc{col}(Consume, 1) : !aie.objectfifosubview<memref<{W_RECORD}xi8>>",
                f"        %wv = aie.objectfifo.subview.access %w[0] : !aie.objectfifosubview<memref<{W_RECORD}xi8>> -> memref<{W_RECORD}xi8>",
                f"        %o = aie.objectfifo.acquire @oc{col}_{row}(Produce, 1) : !aie.objectfifosubview<memref<{OUT_SLOT}xi8>>",
                f"        %ov = aie.objectfifo.subview.access %o[0] : !aie.objectfifosubview<memref<{OUT_SLOT}xi8>> -> memref<{OUT_SLOT}xi8>",
                f"        func.call @r115_compact_group_n16(%av, %wv, %ov) : (memref<{A_SLOT}xi8>, memref<{W_RECORD}xi8>, memref<{OUT_SLOT}xi8>) -> ()",
                f"        aie.objectfifo.release @ac{col}_{row}(Consume, 1)",
                f"        aie.objectfifo.release @wbc{col}(Consume, 1)",
                f"        aie.objectfifo.release @oc{col}_{row}(Produce, 1)",
                "      }",
                "      aie.end",
                "    } {stack_size = 1024 : i32}",
            ]
            continue
        out += [
            f"    %core{col}_{row} = aie.core(%c{col}_{row}) {{",
            f"      %inf = arith.constant {INF} : index",
            "      %z = arith.constant 0 : index",
            "      %one = arith.constant 1 : index",
            f"      %groups = arith.constant {ACTIVE_GROUPS} : index",
            "      scf.for %outer = %z to %inf step %one {",
            f"        %o = aie.objectfifo.acquire @oc{col}_{row}(Produce, 1) : !aie.objectfifosubview<memref<{OUT_SLOT}xi8>>",
            f"        %ov = aie.objectfifo.subview.access %o[0] : !aie.objectfifosubview<memref<{OUT_SLOT}xi8>> -> memref<{OUT_SLOT}xi8>",
            "        scf.for %group = %z to %groups step %one {",
            f"          %a = aie.objectfifo.acquire @ac{col}_{row}(Consume, 1) : !aie.objectfifosubview<memref<{A_SLOT}xi8>>",
            f"          %av = aie.objectfifo.subview.access %a[0] : !aie.objectfifosubview<memref<{A_SLOT}xi8>> -> memref<{A_SLOT}xi8>",
            f"          %w = aie.objectfifo.acquire @wbc{col}(Consume, 1) : !aie.objectfifosubview<memref<{W_RECORD}xi8>>",
            f"          %wv = aie.objectfifo.subview.access %w[0] : !aie.objectfifosubview<memref<{W_RECORD}xi8>> -> memref<{W_RECORD}xi8>",
            "          %groupi = arith.index_cast %group : index to i32",
            f"          func.call @{'r117_compact_group_n32' if N_SLICES == 2 else 'r116_compact_group_n16'}(%av, %wv, %ov, %groupi) : (memref<{A_SLOT}xi8>, memref<{W_RECORD}xi8>, memref<{OUT_SLOT}xi8>, i32) -> ()",
            f"          aie.objectfifo.release @ac{col}_{row}(Consume, 1)",
            f"          aie.objectfifo.release @wbc{col}(Consume, 1)",
            "        }",
            f"        aie.objectfifo.release @oc{col}_{row}(Produce, 1)",
            "      }",
            "      aie.end",
            f"    }} {{stack_size = {3072 if N_SLICES == 2 else 2048} : i32}}",
        ]

out.append(
    f"    aie.runtime_sequence(%A: memref<{A_BYTES}xi8>, %W: memref<{W_BYTES}xi8>, %O: memref<{O_BYTES}xi8>) {{"
)

activation_tasks = []
output_tasks = []
for row in range(ROWS):
    for half in range(HALVES):
        mt = row + half * ROWS
        record = (row * HALVES + half) * GROUPS
        aname = f"ta{half}_{row}"
        oname = f"to{half}_{row}"
        activation_tasks.append(aname)
        output_tasks.append(oname)
        token_base = half * 128 + row * 32
        out += [
            f"      %{aname} = aiex.dma_configure_task_for @ash{half}_{row} {{",
            f"        aie.dma_bd(%A : memref<{A_BYTES}xi8>, {record * A_JOIN}, {ACTIVE_GROUPS * A_JOIN}, [<size = {48 * ACTIVE_GROUPS}, stride = 512>, <size = 512, stride = 1>]) {{burst_length = 0 : i32}}",
            "        aie.end",
            "      } {issue_token = true}",
            f"      aiex.dma_start_task(%{aname})",
            f"      %{oname} = aiex.dma_configure_task_for @osh{half}_{row} {{",
            f"        aie.dma_bd(%O : memref<{O_BYTES}xi8>, {token_base * 16 * N_SLICES * 4}, {OUT_JOIN}, [<size = 4, stride = {OUT_SLOT}>, <size = {OUT_SLOT}, stride = 1>]) {{burst_length = 0 : i32}}",
            "        aie.end",
            "      } {issue_token = true}",
            f"      aiex.dma_start_task(%{oname})",
        ]

weight_tasks = []
for col in range(COLS):
    name = f"tw{col}"
    weight_tasks.append(name)
    out += [
        f"      %{name} = aiex.dma_configure_task_for @wsh{col} {{",
        f"        aie.dma_bd(%W : memref<{W_BYTES}xi8>, {col * ACTIVE_GROUPS * W_RECORD}, {ACTIVE_GROUPS * W_RECORD}, [<size = {W_RECORD // 32 * ACTIVE_GROUPS}, stride = 32>, <size = 32, stride = 1>]) {{burst_length = 0 : i32}}",
        "        aie.end",
        "      } {issue_token = true}",
        f"      aiex.dma_start_task(%{name})",
    ]

for name in output_tasks + activation_tasks + weight_tasks:
    out += [f"      aiex.dma_await_task(%{name})", f"      aiex.dma_free_task(%{name})"]

out += ["    }", "  }", "}"]
print("\n".join(out))
