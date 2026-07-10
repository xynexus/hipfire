#!/usr/bin/env python3
"""Fused scaled gate/up projection and tile-local GeGLU on all 32 AIE2P cores."""

import sys

MODE = sys.argv[1]
GROUPS, OUTBLOCKS, COLS, ROWS, INTER = map(int, sys.argv[2:7])
if COLS != 8:
    raise SystemExit("R18 requires all 8 AIE2P columns")
if MODE == "w4":
    LM, LN, MR = 6, 6, 4
    AB, WB, CB, CO = 8192, 16384, 2304, 1152
    HALF_STRIPE, OUTPUT_STRIPE, VEC = 48, 48, 16
elif MODE == "w8":
    LM, LN, MR = 3, 3, 8
    AB, WB, CB, CO = 8192, 16384, 1152, 768
    HALF_STRIPE, OUTPUT_STRIPE, VEC = 24, 32, 16
else:
    raise SystemExit("MODE must be w4 or w8")

MACRO_M = 96
LOGICAL_MACRO_N = COLS * HALF_STRIPE
OUTPUT_MACRO_N = COLS * OUTPUT_STRIPE
MM = (ROWS + MACRO_M - 1) // MACRO_M
NM = (INTER + LOGICAL_MACRO_N - 1) // LOGICAL_MACRO_N
if OUTBLOCKS != MM * NM:
    raise SystemExit("OUTBLOCKS must equal ceil(M/96)*ceil(INTER/MACRO_N)")
PAD_M, PAD_N = MM * MACRO_M, NM * OUTPUT_MACRO_N
INBLOCKS = GROUPS * OUTBLOCKS
INF = 9223372036854775807


def contiguous_dims(count, block):
    return f"[<size = {count}, stride = {block}>, <size = {block // 512}, stride = 512>, <size = 512, stride = 1>]"


def rowmajor_dims():
    rows = 2 * LM * MR
    return (
        f"[<size = {rows}, stride = {PAD_N}>, "
        "<size = 1, stride = 0>, "
        "<size = 1, stride = 0>, "
        f"<size = {OUTPUT_STRIPE}, stride = 1>]"
    )


out = ["module {", "  aie.device(npu2) {"]
for col in range(COLS):
    out += [f"    %shim{col} = aie.tile({col}, 0)", f"    %mt{col} = aie.tile({col}, 1)"]
    for row in range(4):
        out += [
            f"    %c{col}_{row} = aie.tile({col}, {row + 2})",
            f"    %acc{col}_{row} = aie.buffer(%c{col}_{row}) {{sym_name = \"acc{col}_{row}\"}} : memref<{CB}xi32>",
        ]
for col in range(COLS):
    cores = ", ".join(f"%c{col}_{row}" for row in range(4))
    out += [
        f"    aie.objectfifo @wsh{col}(%shim{col}, {{%mt{col}}}, 1 : i32) : !aie.objectfifo<memref<{WB}xi8>>",
        f"    aie.objectfifo @wbc{col}(%mt{col}, {{{cores}}}, 1 : i32) : !aie.objectfifo<memref<{WB}xi8>>",
        f"    aie.objectfifo.link [@wsh{col}] -> [@wbc{col}] ([] [0])",
    ]
for row in range(4):
    cores = ", ".join(f"%c{col}_{row}" for col in range(COLS))
    out += [
        f"    aie.objectfifo @ash{row}(%shim{row}, {{%mt{row}}}, 1 : i32) : !aie.objectfifo<memref<{AB}xi8>>",
        f"    aie.objectfifo @abc{row}(%mt{row}, {{{cores}}}, 1 : i32) : !aie.objectfifo<memref<{AB}xi8>>",
        f"    aie.objectfifo.link [@ash{row}] -> [@abc{row}] ([] [0])",
    ]
for col in range(COLS):
    for row in range(4):
        out.append(
            f"    aie.objectfifo @oc{col}_{row}(%c{col}_{row}, {{%mt{col}}}, 1 : i32) : !aie.objectfifo<memref<{CO}xi32>>"
        )
    for pair in range(2):
        first = 2 * pair
        out += [
            f"    aie.objectfifo @osh{col}_{pair}(%mt{col}, {{%shim{col}}}, 1 : i32) : !aie.objectfifo<memref<{2 * CO}xi32>>",
            f"    aie.objectfifo.link [@oc{col}_{first}, @oc{col}_{first + 1}] -> [@osh{col}_{pair}] ([0, {CO}] [])",
        ]
for name in (f"r15_{MODE}_scaled_init", f"r15_{MODE}_scaled_accum"):
    out.append(
        f'    func.func private @{name}(memref<{AB}xi8>, memref<{WB}xi8>, memref<{CB}xi32>) attributes {{link_with = "r18.o"}}'
    )
out.append(
    f'    func.func private @r18_{MODE}_geglu(memref<{CB}xi32>, memref<{CO}xi32>) attributes {{link_with = "r18.o"}}'
)
for col in range(COLS):
    for row in range(4):
        out += [
            f"    %core{col}_{row} = aie.core(%c{col}_{row}) {{",
            "      %z = arith.constant 0 : index",
            f"      %m = arith.constant {INF} : index",
            f"      %groups = arith.constant {GROUPS} : index",
            "      %one = arith.constant 1 : index",
            "      scf.for %outer = %z to %m step %one {",
            f"        %a0 = aie.objectfifo.acquire @abc{row}(Consume, 1) : !aie.objectfifosubview<memref<{AB}xi8>>",
            f"        %av0 = aie.objectfifo.subview.access %a0[0] : !aie.objectfifosubview<memref<{AB}xi8>> -> memref<{AB}xi8>",
            f"        %w0 = aie.objectfifo.acquire @wbc{col}(Consume, 1) : !aie.objectfifosubview<memref<{WB}xi8>>",
            f"        %wv0 = aie.objectfifo.subview.access %w0[0] : !aie.objectfifosubview<memref<{WB}xi8>> -> memref<{WB}xi8>",
            f"        func.call @r15_{MODE}_scaled_init(%av0, %wv0, %acc{col}_{row}) : (memref<{AB}xi8>, memref<{WB}xi8>, memref<{CB}xi32>) -> ()",
            f"        aie.objectfifo.release @abc{row}(Consume, 1)",
            f"        aie.objectfifo.release @wbc{col}(Consume, 1)",
            "        scf.for %group = %one to %groups step %one {",
            f"          %a = aie.objectfifo.acquire @abc{row}(Consume, 1) : !aie.objectfifosubview<memref<{AB}xi8>>",
            f"          %av = aie.objectfifo.subview.access %a[0] : !aie.objectfifosubview<memref<{AB}xi8>> -> memref<{AB}xi8>",
            f"          %w = aie.objectfifo.acquire @wbc{col}(Consume, 1) : !aie.objectfifosubview<memref<{WB}xi8>>",
            f"          %wv = aie.objectfifo.subview.access %w[0] : !aie.objectfifosubview<memref<{WB}xi8>> -> memref<{WB}xi8>",
            f"          func.call @r15_{MODE}_scaled_accum(%av, %wv, %acc{col}_{row}) : (memref<{AB}xi8>, memref<{WB}xi8>, memref<{CB}xi32>) -> ()",
            f"          aie.objectfifo.release @abc{row}(Consume, 1)",
            f"          aie.objectfifo.release @wbc{col}(Consume, 1)",
            "        }",
            f"        %o = aie.objectfifo.acquire @oc{col}_{row}(Produce, 1) : !aie.objectfifosubview<memref<{CO}xi32>>",
            f"        %ov = aie.objectfifo.subview.access %o[0] : !aie.objectfifosubview<memref<{CO}xi32>> -> memref<{CO}xi32>",
            f"        func.call @r18_{MODE}_geglu(%acc{col}_{row}, %ov) : (memref<{CB}xi32>, memref<{CO}xi32>) -> ()",
            f"        aie.objectfifo.release @oc{col}_{row}(Produce, 1)",
            "      }",
            "      aie.end",
            "    } {stack_size = 4096 : i32}",
        ]

AT, WT = INBLOCKS * AB, INBLOCKS * WB
out.append(
    f"    aie.runtime_sequence(%A: memref<{4 * AT}xi8>, %W: memref<{COLS * WT}xi8>, %O: memref<{PAD_M * PAD_N}xi32>) {{"
)
for row in range(4):
    out += [
        f"      %ta{row} = aiex.dma_configure_task_for @ash{row} {{",
        f"        aie.dma_bd(%A : memref<{4 * AT}xi8>, {row * AT}, {AT}, {contiguous_dims(INBLOCKS, AB)}) {{burst_length = 0 : i32}}",
        "        aie.end",
        "      }",
        f"      aiex.dma_start_task(%ta{row})",
    ]
for col in range(COLS):
    out += [
        f"      %tw{col} = aiex.dma_configure_task_for @wsh{col} {{",
        f"        aie.dma_bd(%W : memref<{COLS * WT}xi8>, {col * WT}, {WT}, {contiguous_dims(INBLOCKS, WB)}) {{burst_length = 0 : i32}}",
        "        aie.end",
        "      }",
        f"      aiex.dma_start_task(%tw{col})",
    ]
for outblock in range(OUTBLOCKS):
    m_macro, n_macro = divmod(outblock, NM)
    for col in range(COLS):
        for pair in range(2):
            offset = (
                m_macro * MACRO_M * PAD_N
                + pair * 2 * LM * MR * PAD_N
                + n_macro * OUTPUT_MACRO_N
                + col * OUTPUT_STRIPE
            )
            name = f"to{col}_{pair}_{outblock}"
            out += [
                f"      %{name} = aiex.dma_configure_task_for @osh{col}_{pair} {{",
                f"        aie.dma_bd(%O : memref<{PAD_M * PAD_N}xi32>, {offset}, {OUTPUT_STRIPE}, {rowmajor_dims()}) {{burst_length = 0 : i32}}",
                "        aie.end",
                f"      }} {{issue_token = true, repeat_count = {2 * LM * MR - 1} : i32}}",
                f"      aiex.dma_start_task(%{name})",
            ]
    for col in range(COLS):
        for pair in range(2):
            name = f"to{col}_{pair}_{outblock}"
            out += [f"      aiex.dma_await_task(%{name})", f"      aiex.dma_free_task(%{name})"]
for row in range(4):
    out.append(f"      aiex.dma_free_task(%ta{row})")
for col in range(COLS):
    out.append(f"      aiex.dma_free_task(%tw{col})")
out += ["    }", "  }", "}"]
print("\n".join(out))
