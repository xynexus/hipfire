#!/usr/bin/env python3
"""Generate the raw R16-style full-QKV graph with direct row-major output DMA."""

import sys

ACTIVATION_MODE = sys.argv[1] if len(sys.argv) > 1 else "r34-legacy"
if ACTIVATION_MODE not in ("r34-legacy", "w4-native"):
    raise SystemExit("activation mode must be r34-legacy or w4-native")
OUTPUT_MODE = sys.argv[2] if len(sys.argv) > 2 else "rowmajor"
if OUTPUT_MODE not in ("rowmajor", "physical"):
    raise SystemExit("output mode must be rowmajor or physical")
WEIGHT_MODE = sys.argv[3] if len(sys.argv) > 3 else "bundle"
if WEIGHT_MODE not in ("bundle", "qkv-only", "qkv-compact"):
    raise SystemExit("weight mode must be bundle, qkv-only, or qkv-compact")

COLUMNS, CORE_ROWS = 8, 4
GROUPS, OUTBLOCKS = 3, 6
AB = 16_384 if ACTIVATION_MODE == "r34-legacy" else 8192
WB, CB, CJ = 16_384, 2304, 9216
ACTIVATION_BLOCKS_PER_STRIPE = 45 if ACTIVATION_MODE == "r34-legacy" else 18
WEIGHT_BLOCKS_PER_COLUMN = 18 if WEIGHT_MODE == "qkv-compact" else 28
STREAMED_WEIGHT_BLOCKS = 28 if WEIGHT_MODE == "bundle" else OUTBLOCKS * GROUPS
PAD_M, PAD_N = 288, 1536
INF = 9223372036854775807


def contiguous_dims(count, block):
    return (
        f"[<size = {count}, stride = {block}>, "
        f"<size = {block // 512}, stride = 512>, "
        "<size = 512, stride = 1>]"
    )


def rowmajor_dims():
    return (
        f"[<size = 24, stride = {4 * PAD_N}>, "
        "<size = 6, stride = 16>, "
        f"<size = 4, stride = {PAD_N}>, "
        "<size = 16, stride = 1>]"
    )


out = ["module {", "  aie.device(npu2) {"]
for col in range(COLUMNS):
    out += [f"    %shim{col} = aie.tile({col}, 0)", f"    %mt{col} = aie.tile({col}, 1)"]
    for row in range(CORE_ROWS):
        out.append(f"    %c{col}_{row} = aie.tile({col}, {row + 2})")

for col in range(COLUMNS):
    cores = ", ".join(f"%c{col}_{row}" for row in range(CORE_ROWS))
    out += [
        f"    aie.objectfifo @wsh{col}(%shim{col}, {{%mt{col}}}, 1 : i32) : !aie.objectfifo<memref<{WB}xi8>>",
        f"    aie.objectfifo @wbc{col}(%mt{col}, {{{cores}}}, 1 : i32) : !aie.objectfifo<memref<{WB}xi8>>",
        f"    aie.objectfifo.link [@wsh{col}] -> [@wbc{col}] ([] [0])",
    ]
for row in range(CORE_ROWS):
    cores = ", ".join(f"%c{col}_{row}" for col in range(COLUMNS))
    out += [
        f"    aie.objectfifo @ash{row}(%shim{row}, {{%mt{row}}}, 1 : i32) : !aie.objectfifo<memref<{AB}xi8>>",
        f"    aie.objectfifo @abc{row}(%mt{row}, {{{cores}}}, 1 : i32) : !aie.objectfifo<memref<{AB}xi8>>",
        f"    aie.objectfifo.link [@ash{row}] -> [@abc{row}] ([] [0])",
    ]
for col in range(COLUMNS):
    inputs = ", ".join(f"@cc{col}_{row}" for row in range(CORE_ROWS))
    offsets = ", ".join(str(row * CB) for row in range(CORE_ROWS))
    for row in range(CORE_ROWS):
        out.append(
            f"    aie.objectfifo @cc{col}_{row}(%c{col}_{row}, {{%mt{col}}}, 1 : i32) : !aie.objectfifo<memref<{CB}xi32>>"
        )
    out += [
        f"    aie.objectfifo @csh{col}(%mt{col}, {{%shim{col}}}, 1 : i32) : !aie.objectfifo<memref<{CJ}xi32>>",
        f"    aie.objectfifo.link [{inputs}] -> [@csh{col}] ([{offsets}] [])",
    ]

INIT_SYMBOL = "r61_full_qkv_init" if ACTIVATION_MODE == "r34-legacy" else "r15_w4_scaled_init"
ACCUM_SYMBOL = "r61_full_qkv_accum" if ACTIVATION_MODE == "r34-legacy" else "r15_w4_scaled_accum"
for name in (INIT_SYMBOL, ACCUM_SYMBOL):
    out.append(
        f'    func.func private @{name}(memref<{AB}xi8>, memref<{WB}xi8>, memref<{CB}xi32>) attributes {{link_with = "compute.o"}}'
    )
out.append(
    f'    func.func private @r61_weight_sink(memref<{WB}xi8>) attributes {{link_with = "sink.o"}}'
)

for col in range(COLUMNS):
    for row in range(CORE_ROWS):
        if WEIGHT_MODE in ("qkv-only", "qkv-compact"):
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
                f"        func.call @{INIT_SYMBOL}(%av0, %wv0, %cv) : (memref<{AB}xi8>, memref<{WB}xi8>, memref<{CB}xi32>) -> ()",
                f"        aie.objectfifo.release @abc{row}(Consume, 1)",
                f"        aie.objectfifo.release @wbc{col}(Consume, 1)",
                "        scf.for %group = %o to %groups step %o {",
                f"          %a = aie.objectfifo.acquire @abc{row}(Consume, 1) : !aie.objectfifosubview<memref<{AB}xi8>>",
                f"          %av = aie.objectfifo.subview.access %a[0] : !aie.objectfifosubview<memref<{AB}xi8>> -> memref<{AB}xi8>",
                f"          %w = aie.objectfifo.acquire @wbc{col}(Consume, 1) : !aie.objectfifosubview<memref<{WB}xi8>>",
                f"          %wv = aie.objectfifo.subview.access %w[0] : !aie.objectfifosubview<memref<{WB}xi8>> -> memref<{WB}xi8>",
                f"          func.call @{ACCUM_SYMBOL}(%av, %wv, %cv) : (memref<{AB}xi8>, memref<{WB}xi8>, memref<{CB}xi32>) -> ()",
                f"          aie.objectfifo.release @abc{row}(Consume, 1)",
                f"          aie.objectfifo.release @wbc{col}(Consume, 1)",
                "        }",
                f"        aie.objectfifo.release @cc{col}_{row}(Produce, 1)",
                "      }",
                "      aie.end",
                "    }",
            ]
            continue
        out += [
            f"    %core{col}_{row} = aie.core(%c{col}_{row}) {{",
            "      %z = arith.constant 0 : index",
            f"      %m = arith.constant {INF} : index",
            "      %o = arith.constant 1 : index",
            "      scf.for %outer = %z to %m step %o {",
        ]
        core_outblocks = 1 if WEIGHT_MODE == "qkv-only" else OUTBLOCKS
        for outblock in range(core_outblocks):
            out += [
                f"        %co{outblock} = aie.objectfifo.acquire @cc{col}_{row}(Produce, 1) : !aie.objectfifosubview<memref<{CB}xi32>>",
                f"        %cov{outblock} = aie.objectfifo.subview.access %co{outblock}[0] : !aie.objectfifosubview<memref<{CB}xi32>> -> memref<{CB}xi32>",
            ]
            for group in range(GROUPS):
                suffix = f"{outblock}_{group}"
                out += [
                    f"        %a{suffix} = aie.objectfifo.acquire @abc{row}(Consume, 1) : !aie.objectfifosubview<memref<{AB}xi8>>",
                    f"        %av{suffix} = aie.objectfifo.subview.access %a{suffix}[0] : !aie.objectfifosubview<memref<{AB}xi8>> -> memref<{AB}xi8>",
                    f"        %w{suffix} = aie.objectfifo.acquire @wbc{col}(Consume, 1) : !aie.objectfifosubview<memref<{WB}xi8>>",
                    f"        %wv{suffix} = aie.objectfifo.subview.access %w{suffix}[0] : !aie.objectfifosubview<memref<{WB}xi8>> -> memref<{WB}xi8>",
                    f"        func.call @{INIT_SYMBOL if group == 0 else ACCUM_SYMBOL}(%av{suffix}, %wv{suffix}, %cov{outblock}) : (memref<{AB}xi8>, memref<{WB}xi8>, memref<{CB}xi32>) -> ()",
                    f"        aie.objectfifo.release @abc{row}(Consume, 1)",
                    f"        aie.objectfifo.release @wbc{col}(Consume, 1)",
                ]
            out.append(f"        aie.objectfifo.release @cc{col}_{row}(Produce, 1)")
        for extra in range(STREAMED_WEIGHT_BLOCKS - OUTBLOCKS * GROUPS):
            out += [
                f"        %ws{extra} = aie.objectfifo.acquire @wbc{col}(Consume, 1) : !aie.objectfifosubview<memref<{WB}xi8>>",
                f"        %wsv{extra} = aie.objectfifo.subview.access %ws{extra}[0] : !aie.objectfifosubview<memref<{WB}xi8>> -> memref<{WB}xi8>",
                f"        func.call @r61_weight_sink(%wsv{extra}) : (memref<{WB}xi8>) -> ()",
                f"        aie.objectfifo.release @wbc{col}(Consume, 1)",
            ]
        out += ["      }", "      aie.end", "    }"]

A_BYTES = CORE_ROWS * ACTIVATION_BLOCKS_PER_STRIPE * AB
W_BYTES = COLUMNS * WEIGHT_BLOCKS_PER_COLUMN * WB
C_ELEMS = PAD_M * PAD_N
out.append(
    f"    aie.runtime_sequence(%A: memref<{A_BYTES}xi8>, %W: memref<{W_BYTES}xi8>, %C: memref<{C_ELEMS}xi32>) {{"
)

activation_tasks = []
for row in range(CORE_ROWS):
    if ACTIVATION_MODE == "r34-legacy":
        for m_macro in range(3):
            for n_macro in range(2):
                name = f"ta{row}_{m_macro}_{n_macro}"
                offset = (
                    row * ACTIVATION_BLOCKS_PER_STRIPE
                    + (m_macro * 5 + n_macro) * GROUPS
                ) * AB
                activation_tasks.append(name)
                out += [
                    f"      %{name} = aiex.dma_configure_task_for @ash{row} {{",
                    f"        aie.dma_bd(%A : memref<{A_BYTES}xi8>, {offset}, {GROUPS * AB}, {contiguous_dims(GROUPS, AB)}) {{burst_length = 0 : i32}}",
                    "        aie.end",
                    "      }",
                    f"      aiex.dma_start_task(%{name})",
                ]
    else:
        name = f"ta{row}"
        offset = row * ACTIVATION_BLOCKS_PER_STRIPE * AB
        activation_tasks.append(name)
        out += [
            f"      %{name} = aiex.dma_configure_task_for @ash{row} {{",
            f"        aie.dma_bd(%A : memref<{A_BYTES}xi8>, {offset}, {ACTIVATION_BLOCKS_PER_STRIPE * AB}, {contiguous_dims(ACTIVATION_BLOCKS_PER_STRIPE, AB)}) {{burst_length = 0 : i32}}",
            "        aie.end",
            "      }",
            f"      aiex.dma_start_task(%{name})",
        ]

for col in range(COLUMNS):
    out += [
        f"      %tw{col} = aiex.dma_configure_task_for @wsh{col} {{",
        f"        aie.dma_bd(%W : memref<{W_BYTES}xi8>, {col * WEIGHT_BLOCKS_PER_COLUMN * WB}, {STREAMED_WEIGHT_BLOCKS * WB}, {contiguous_dims(STREAMED_WEIGHT_BLOCKS, WB)}) {{burst_length = 0 : i32}}",
        "        aie.end",
        "      }",
        f"      aiex.dma_start_task(%tw{col})",
    ]
    if OUTPUT_MODE == "physical" and WEIGHT_MODE == "qkv-compact":
        name = f"tc{col}"
        out += [
            f"      %{name} = aiex.dma_configure_task_for @csh{col} {{",
            f"        aie.dma_bd(%C : memref<{C_ELEMS}xi32>, {col * OUTBLOCKS * CJ}, {OUTBLOCKS * CJ}, {contiguous_dims(OUTBLOCKS, CJ)}) {{burst_length = 0 : i32}}",
            "        aie.end",
            "      } {issue_token = true}",
            f"      aiex.dma_start_task(%{name})",
        ]

if OUTPUT_MODE == "rowmajor":
    for outblock in range(OUTBLOCKS):
        for col in range(COLUMNS):
            m_macro, n_macro = divmod(outblock, 2)
            offset = m_macro * 96 * PAD_N + n_macro * 768 + col * 96
            name = f"tc{col}_{outblock}"
            out += [
                f"      %{name} = aiex.dma_configure_task_for @csh{col} {{",
                f"        aie.dma_bd(%C : memref<{C_ELEMS}xi32>, {offset}, {6 * 4 * 16}, {rowmajor_dims()}) {{burst_length = 0 : i32}}",
                "        aie.end",
                f"      }} {{issue_token = true, repeat_count = 23 : i32}}",
                f"      aiex.dma_start_task(%{name})",
            ]
        for col in range(COLUMNS):
            name = f"tc{col}_{outblock}"
            out += [f"      aiex.dma_await_task(%{name})", f"      aiex.dma_free_task(%{name})"]
else:
    if WEIGHT_MODE != "qkv-compact":
        for col in range(COLUMNS):
            name = f"tc{col}"
            out += [
                f"      %{name} = aiex.dma_configure_task_for @csh{col} {{",
                f"        aie.dma_bd(%C : memref<{C_ELEMS}xi32>, {col * OUTBLOCKS * CJ}, {OUTBLOCKS * CJ}, {contiguous_dims(OUTBLOCKS, CJ)}) {{burst_length = 0 : i32}}",
                "        aie.end",
                "      } {issue_token = true}",
                f"      aiex.dma_start_task(%{name})",
            ]
    for col in range(COLUMNS):
        name = f"tc{col}"
        out += [f"      aiex.dma_await_task(%{name})", f"      aiex.dma_free_task(%{name})"]

for name in activation_tasks:
    out.append(f"      aiex.dma_free_task(%{name})")
for col in range(COLUMNS):
    out.append(f"      aiex.dma_free_task(%tw{col})")
out += ["    }", "  }", "}"]
print("\n".join(out))
