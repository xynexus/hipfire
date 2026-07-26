#!/usr/bin/env python3
"""R15 scaled whole-array GEMM with direct padded-row-major C DMA.

The compute cores retain and scale full-K tiles exactly as R15.  Only the
output side changes: one queued shim DMA task per (column, output block)
scatters the joined core stream into [padded_M, padded_N] row-major memory.
"""

import sys

MODE = sys.argv[1]
GROUPS, OUTBLOCKS, COLS, ROWS, N = map(int, sys.argv[2:7])
if COLS not in (4, 8):
    raise SystemExit("COLS must be 4 or 8")
if MODE == "w4":
    LM, LN, MR = 6, 6, 4
    AB, WB, CB, CJ, COLS_STRIPE = 8192, 16384, 2304, 9216, 96
elif MODE == "w8":
    LM, LN, MR = 3, 3, 8
    AB, WB, CB, CJ, COLS_STRIPE = 8192, 16384, 1152, 4608, 48
else:
    raise SystemExit("MODE must be w4 or w8")

MACRO_M = 96
MACRO_N = COLS * COLS_STRIPE
MM = (ROWS + MACRO_M - 1) // MACRO_M
NM = (N + MACRO_N - 1) // MACRO_N
if OUTBLOCKS != MM * NM:
    raise SystemExit("OUTBLOCKS must equal ceil(M/96)*ceil(N/MACRO_N)")
PAD_M, PAD_N = MM * MACRO_M, NM * MACRO_N
INBLOCKS = GROUPS * OUTBLOCKS
INF = 9223372036854775807
GC, GR = range(COLS), range(4)


def contiguous_dims(count, block):
    return f"[<size = {count}, stride = {block}>, <size = {block // 512}, stride = 512>, <size = 512, stride = 1>]"


def rowmajor_dims():
    # Stream order from the joined four cores is:
    #   row_stripe, lm, ln, local_row, local_col.
    # row_stripe and lm are one contiguous row-block dimension.
    return (
        f"[<size = {4 * LM}, stride = {MR * PAD_N}>, "
        f"<size = {LN}, stride = 16>, "
        f"<size = {MR}, stride = {PAD_N}>, "
        f"<size = 16, stride = 1>]"
    )


out = ["module {", "  aie.device(npu2) {"]
for col in GC:
    out += [f"    %shim{col} = aie.tile({col}, 0)", f"    %mt{col} = aie.tile({col}, 1)"]
    for row in GR:
        out.append(f"    %c{col}_{row} = aie.tile({col}, {row + 2})")
for col in GC:
    cores = ", ".join(f"%c{col}_{row}" for row in GR)
    out += [
        f"    aie.objectfifo @wsh{col}(%shim{col}, {{%mt{col}}}, 1 : i32) : !aie.objectfifo<memref<{WB}xi8>>",
        f"    aie.objectfifo @wbc{col}(%mt{col}, {{{cores}}}, 1 : i32) : !aie.objectfifo<memref<{WB}xi8>>",
        f"    aie.objectfifo.link [@wsh{col}] -> [@wbc{col}] ([] [0])",
    ]
for row in GR:
    cores = ", ".join(f"%c{col}_{row}" for col in GC)
    out += [
        f"    aie.objectfifo @ash{row}(%shim{row}, {{%mt{row}}}, 1 : i32) : !aie.objectfifo<memref<{AB}xi8>>",
        f"    aie.objectfifo @abc{row}(%mt{row}, {{{cores}}}, 1 : i32) : !aie.objectfifo<memref<{AB}xi8>>",
        f"    aie.objectfifo.link [@ash{row}] -> [@abc{row}] ([] [0])",
    ]
for col in GC:
    inputs = ", ".join(f"@cc{col}_{row}" for row in GR)
    offsets = ", ".join(str(row * CB) for row in GR)
    for row in GR:
        out.append(
            f"    aie.objectfifo @cc{col}_{row}(%c{col}_{row}, {{%mt{col}}}, 1 : i32) : !aie.objectfifo<memref<{CB}xi32>>"
        )
    out += [
        f"    aie.objectfifo @csh{col}(%mt{col}, {{%shim{col}}}, 1 : i32) : !aie.objectfifo<memref<{CJ}xi32>>",
        f"    aie.objectfifo.link [{inputs}] -> [@csh{col}] ([{offsets}] [])",
    ]
for name in (f"r15_{MODE}_scaled_init", f"r15_{MODE}_scaled_accum"):
    out.append(
        f'    func.func private @{name}(memref<{AB}xi8>, memref<{WB}xi8>, memref<{CB}xi32>) attributes {{link_with = "r16.o"}}'
    )
for col in GC:
    for row in GR:
        out += [
            f"    %core{col}_{row} = aie.core(%c{col}_{row}) {{",
            "      %z = arith.constant 0 : index",
            f"      %m = arith.constant {INF} : index",
            f"      %groups = arith.constant {GROUPS} : index",
            "      %o = arith.constant 1 : index",
            "      scf.for %outer = %z to %m step %o {",
            f"        %c = aie.objectfifo.acquire @cc{col}_{row}(Produce, 1) : !aie.objectfifosubview<memref<{CB}xi32>>",
            f"        %cv = aie.objectfifo.subview.access %c[0] : !aie.objectfifosubview<memref<{CB}xi32>> -> memref<{CB}xi32>",
            f"        %a0 = aie.objectfifo.acquire @abc{row}(Consume, 1) : !aie.objectfifosubview<memref<{AB}xi8>>",
            f"        %av0 = aie.objectfifo.subview.access %a0[0] : !aie.objectfifosubview<memref<{AB}xi8>> -> memref<{AB}xi8>",
            f"        %w0 = aie.objectfifo.acquire @wbc{col}(Consume, 1) : !aie.objectfifosubview<memref<{WB}xi8>>",
            f"        %wv0 = aie.objectfifo.subview.access %w0[0] : !aie.objectfifosubview<memref<{WB}xi8>> -> memref<{WB}xi8>",
            f"        func.call @r15_{MODE}_scaled_init(%av0, %wv0, %cv) : (memref<{AB}xi8>, memref<{WB}xi8>, memref<{CB}xi32>) -> ()",
            f"        aie.objectfifo.release @abc{row}(Consume, 1)",
            f"        aie.objectfifo.release @wbc{col}(Consume, 1)",
            "        scf.for %group = %o to %groups step %o {",
            f"          %a = aie.objectfifo.acquire @abc{row}(Consume, 1) : !aie.objectfifosubview<memref<{AB}xi8>>",
            f"          %av = aie.objectfifo.subview.access %a[0] : !aie.objectfifosubview<memref<{AB}xi8>> -> memref<{AB}xi8>",
            f"          %w = aie.objectfifo.acquire @wbc{col}(Consume, 1) : !aie.objectfifosubview<memref<{WB}xi8>>",
            f"          %wv = aie.objectfifo.subview.access %w[0] : !aie.objectfifosubview<memref<{WB}xi8>> -> memref<{WB}xi8>",
            f"          func.call @r15_{MODE}_scaled_accum(%av, %wv, %cv) : (memref<{AB}xi8>, memref<{WB}xi8>, memref<{CB}xi32>) -> ()",
            f"          aie.objectfifo.release @abc{row}(Consume, 1)",
            f"          aie.objectfifo.release @wbc{col}(Consume, 1)",
            "        }",
            f"        aie.objectfifo.release @cc{col}_{row}(Produce, 1)",
            "      }",
            "      aie.end",
            "    }",
        ]

AT, WT = INBLOCKS * AB, INBLOCKS * WB
out.append(
    f"    aie.runtime_sequence(%A: memref<{4 * AT}xi8>, %W: memref<{COLS * WT}xi8>, %C: memref<{PAD_M * PAD_N}xi32>) {{"
)
for row in GR:
    out += [
        f"      %ta{row} = aiex.dma_configure_task_for @ash{row} {{",
        f"        aie.dma_bd(%A : memref<{4 * AT}xi8>, {row * AT}, {AT}, {contiguous_dims(INBLOCKS, AB)}) {{burst_length = 0 : i32}}",
        "        aie.end",
        "      }",
        f"      aiex.dma_start_task(%ta{row})",
    ]
for col in GC:
    out += [
        f"      %tw{col} = aiex.dma_configure_task_for @wsh{col} {{",
        f"        aie.dma_bd(%W : memref<{COLS * WT}xi8>, {col * WT}, {WT}, {contiguous_dims(INBLOCKS, WB)}) {{burst_length = 0 : i32}}",
        "        aie.end",
        "      }",
        f"      aiex.dma_start_task(%tw{col})",
    ]
for outblock in range(OUTBLOCKS):
    for col in GC:
        m_macro, n_macro = divmod(outblock, NM)
        offset = m_macro * MACRO_M * PAD_N + n_macro * MACRO_N + col * COLS_STRIPE
        name = f"tc{col}_{outblock}"
        out += [
            f"      %{name} = aiex.dma_configure_task_for @csh{col} {{",
            f"        aie.dma_bd(%C : memref<{PAD_M * PAD_N}xi32>, {offset}, {LN * MR * 16}, {rowmajor_dims()}) {{burst_length = 0 : i32}}",
            "        aie.end",
            f"      }} {{issue_token = true, repeat_count = {4 * LM - 1} : i32}}",
            f"      aiex.dma_start_task(%{name})",
        ]
    for col in GC:
        name = f"tc{col}_{outblock}"
        out += [f"      aiex.dma_await_task(%{name})", f"      aiex.dma_free_task(%{name})"]
for row in GR:
    out.append(f"      aiex.dma_free_task(%ta{row})")
for col in GC:
    out.append(f"      aiex.dma_free_task(%tw{col})")
out += ["    }", "  }", "}"]
print("\n".join(out))
