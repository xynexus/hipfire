#!/usr/bin/env python3
"""One-dispatch resident dense-W8 EmbeddingGemma FFN on AIE2P."""

import sys

CANONICAL_BF16 = "--canonical-bf16-input" in sys.argv[1:]
CANONICAL_BF16X2_OUTPUT = "--canonical-bf16x2-output" in sys.argv[1:]
DIRECT_X_PRE_NORM = "--direct-x-pre-norm" in sys.argv[1:]
REUSE_GATE_ACTIVATION = "--reuse-gate-activation" in sys.argv[1:]
if CANONICAL_BF16X2_OUTPUT and not CANONICAL_BF16:
    raise SystemExit("--canonical-bf16x2-output requires --canonical-bf16-input")
if DIRECT_X_PRE_NORM and not (CANONICAL_BF16 and CANONICAL_BF16X2_OUTPUT):
    raise SystemExit(
        "--direct-x-pre-norm requires --canonical-bf16-input and "
        "--canonical-bf16x2-output"
    )
if REUSE_GATE_ACTIVATION and not DIRECT_X_PRE_NORM:
    raise SystemExit("--reuse-gate-activation requires --direct-x-pre-norm")


def _int_flag(flag, default):
    for a in sys.argv[1:]:
        if a.startswith(flag + "="):
            return int(a.split("=", 1)[1])
    return default


# Batch: number of 256-row documents packed into one dispatch. BATCH=1 is the
# original M256 kernel (value-preserving). Larger BATCH scales M by concatenating
# documents' rows; the FFN is per-row independent, so no cross-doc masking. Only
# the DMA/BD schedule length and the T/O/D buffer sizes grow linearly — the AIE
# core program (r26_w8_resident_ffn.cc) is unchanged (fixed 24-row tiles).
BATCH = _int_flag("--batch", 1)
REAL_M = 256 * BATCH

COLS, CORE_ROWS = 8, 4
M_MACROS, GATE_N_MACROS = 3 * BATCH, 6
GATE_GROUPS, DOWN_GROUPS = 3, 5
GATE_OUTBLOCKS = M_MACROS * GATE_N_MACROS
DOWN_MBLOCKS = M_MACROS
DATA_PAIR = 12288 if CANONICAL_BF16 else 9216
DATA_JOIN = DATA_PAIR if CANONICAL_BF16 else 4 * DATA_PAIR
APACK = 6240
WB = 16384
GATE_OUTPUT_BYTES = 1536
OUTPUT_CO = 2304
FRAGMENT = 784
OWN_FRAGMENT = GATE_GROUPS * FRAGMENT if REUSE_GATE_ACTIVATION else FRAGMENT
SCRATCH = 256
GATE_ACC = 1152
GATE_DATA_BLOCKS = GATE_OUTBLOCKS * GATE_GROUPS
GATE_PARAM_BLOCKS = M_MACROS * GATE_GROUPS if REUSE_GATE_ACTIVATION else 0
WEIGHT_BLOCKS = GATE_PARAM_BLOCKS + GATE_DATA_BLOCKS + DOWN_MBLOCKS * DOWN_GROUPS * 2
T_ROWS, T_STRIDE, INTERMEDIATE, PAD_INTERMEDIATE, OUTPUT = 96 * M_MACROS + 8, 5376, 1152, 1280, 768
PAD_M = 96 * M_MACROS
O_ELEMS = PAD_M * OUTPUT
CANONICAL_INPUT_BYTES = PAD_M * 768 * 2
DIRECT_X_INPUT_BYTES = 2 * REAL_M * 768 * 2
RUNTIME_INPUT_BYTES = DIRECT_X_INPUT_BYTES if DIRECT_X_PRE_NORM else CANONICAL_INPUT_BYTES
PRE_INVERSE_BASE = REAL_M * 768 * 2
PRE_INVERSE_RECORD_BYTES = 12288
INVERSE_TABLE = 32 * 8 * 4
CANONICAL_T_BYTES = PAD_M * PAD_INTERMEDIATE * 2
CANONICAL_OUTPUT_COMPONENTS = 3 if CANONICAL_BF16X2_OUTPUT else 1
CANONICAL_OUTPUT_BYTES = PAD_M * OUTPUT * 2 * CANONICAL_OUTPUT_COMPONENTS
INF = 9223372036854775807


def byte_blocks(count, block):
    return (
        f"[<size = {count}, stride = {block}>, "
        f"<size = {block // 512}, stride = 512>, "
        "<size = 512, stride = 1>]"
    )


def gate_output_dims():
    return (
        f"[<size = 32, stride = {3 * T_STRIDE}>, "
        f"<size = 3, stride = {T_STRIDE}>, "
        "<size = 3, stride = 32>, <size = 32, stride = 1>]"
    )


def down_input_dims():
    return (
        f"[<size = 4, stride = {6 * T_STRIDE}>, "
        f"<size = 8, stride = {T_STRIDE}>, "
        "<size = 12, stride = 96>, <size = 24, stride = 1>]"
    )


def down_output_dims():
    return (
        f"[<size = 3, stride = {32 * OUTPUT}>, "
        f"<size = 32, stride = {OUTPUT}>, "
        "<size = 2, stride = 384>, <size = 48, stride = 1>]"
    )


def canonical_input_dims():
    return "[<size = 24, stride = 1536>, <size = 512, stride = 1>]"


def canonical_gate_output_dims():
    return (
        f"[<size = {CORE_ROWS * 24}, stride = {PAD_INTERMEDIATE * 2}>, "
        "<size = 2, stride = 16>, <size = 32, stride = 1>]"
    )


def canonical_down_output_dims():
    return (
        f"[<size = {CORE_ROWS * 24}, stride = {OUTPUT * 2 * CANONICAL_OUTPUT_COMPONENTS}>, "
        "<size = 2, stride = 16>, <size = 32, stride = 1>]"
    )


def canonical_down_input_dims():
    return (
        f"[<size = 24, stride = {PAD_INTERMEDIATE * 2}>, "
        "<size = 512, stride = 1>]"
    )


out = ["module {", "  aie.device(npu2) {"]
for col in range(COLS):
    out += [f"    %shim{col} = aie.tile({col}, 0)", f"    %mt{col} = aie.tile({col}, 1)"]
    for row in range(CORE_ROWS):
        out += [
            f"    %c{col}_{row} = aie.tile({col}, {row + 2})",
            f"    %gacc{col}_{row} = aie.buffer(%c{col}_{row}) {{sym_name = \"gacc{col}_{row}\"}} : memref<{GATE_ACC}xi32>",
            f"    %apack{col}_{row} = aie.buffer(%c{col}_{row}) {{sym_name = \"apack{col}_{row}\"}} : memref<{APACK}xi8>",
            f"    %scratch{col}_{row} = aie.buffer(%c{col}_{row}) {{sym_name = \"scratch{col}_{row}\"}} : memref<{SCRATCH}xf32>",
            f"    %own{col}_{row} = aie.buffer(%c{col}_{row}) {{sym_name = \"own{col}_{row}\"}} : memref<{OWN_FRAGMENT}xi8>",
            f"    %transit{col}_{row} = aie.buffer(%c{col}_{row}) {{sym_name = \"transit{col}_{row}\"}} : memref<{OWN_FRAGMENT}xi8>",
            *(
                [
                    f"    %inv{col}_{row} = aie.buffer(%c{col}_{row}) {{sym_name = \"inv{col}_{row}\"}} : memref<9xf32>"
                ]
                if DIRECT_X_PRE_NORM
                else []
            ),
        ]

for col in range(COLS):
    cores = ", ".join(f"%c{col}_{row}" for row in range(CORE_ROWS))
    out += [
        f"    aie.objectfifo @wsh{col}(%shim{col}, {{%mt{col}}}, 1 : i32) : !aie.objectfifo<memref<{WB}xi8>>",
        f"    aie.objectfifo @wbc{col}(%mt{col}, {{{cores}}}, 1 : i32) : !aie.objectfifo<memref<{WB}xi8>>",
        f"    aie.objectfifo.link [@wsh{col}] -> [@wbc{col}] ([] [0])",
    ]

for row in range(CORE_ROWS):
    if CANONICAL_BF16:
        cores = ", ".join(f"%c{col}_{row}" for col in range(COLS))
        out += [
            f"    aie.objectfifo @xsh{row}(%shim{row}, {{%mt{row}}}, 1 : i32) : !aie.objectfifo<memref<{DATA_PAIR}xi8>>",
            f"    aie.objectfifo @xbc{row}(%mt{row}, {{{cores}}}, 1 : i32) : !aie.objectfifo<memref<{DATA_PAIR}xi8>>",
            f"    aie.objectfifo.link [@xsh{row}] -> [@xbc{row}] ([] [0])",
        ]
    else:
        pairs = []
        for pair in range(COLS // 2):
            pairs.append(f"@xpair{pair}_{row}")
            out.append(
                f"    aie.objectfifo @xpair{pair}_{row}(%mt{row}, "
                f"{{%c{2 * pair}_{row}, %c{2 * pair + 1}_{row}}}, 1 : i32) : "
                f"!aie.objectfifo<memref<{DATA_PAIR}xi8>>"
            )
        offsets = ", ".join(str(pair * DATA_PAIR) for pair in range(COLS // 2))
        out += [
            f"    aie.objectfifo @xsh{row}(%shim{row}, {{%mt{row}}}, 1 : i32) : !aie.objectfifo<memref<{DATA_JOIN}xi8>>",
            f"    aie.objectfifo.link [@xsh{row}] -> [{', '.join(pairs)}] ([] [{offsets}])",
        ]

for row in range(CORE_ROWS):
    for col in range(COLS):
        out.append(
            f'    aie.flow(%c{col}_{row}, "Core" : 0, '
            f'%c{(col + 1) % COLS}_{row}, "Core" : 0)'
        )

for col in range(COLS):
    if CANONICAL_BF16:
        inputs = ", ".join(f"@oc{col}_{row}" for row in range(CORE_ROWS))
        offsets = ", ".join(str(row * GATE_OUTPUT_BYTES) for row in range(CORE_ROWS))
        for row in range(CORE_ROWS):
            out.append(
                f"    aie.objectfifo @oc{col}_{row}(%c{col}_{row}, {{%mt{col}}}, 1 : i32) : !aie.objectfifo<memref<{GATE_OUTPUT_BYTES}xi8>>"
            )
        out += [
            f"    aie.objectfifo @osh{col}(%mt{col}, {{%shim{col}}}, 1 : i32) : !aie.objectfifo<memref<{CORE_ROWS * GATE_OUTPUT_BYTES}xi8>>",
            f"    aie.objectfifo.link [{inputs}] -> [@osh{col}] ([{offsets}] [])",
        ]
    else:
        inputs = ", ".join(f"@oc{col}_{row}" for row in range(CORE_ROWS))
        offsets = ", ".join(str(row * OUTPUT_CO) for row in range(CORE_ROWS))
        for row in range(CORE_ROWS):
            out.append(
                f"    aie.objectfifo @oc{col}_{row}(%c{col}_{row}, {{%mt{col}}}, 1 : i32) : !aie.objectfifo<memref<{OUTPUT_CO}xi32>>"
            )
        out += [
            f"    aie.objectfifo @osh{col}(%mt{col}, {{%shim{col}}}, 1 : i32) : !aie.objectfifo<memref<{CORE_ROWS * OUTPUT_CO}xi32>>",
            f"    aie.objectfifo.link [{inputs}] -> [@osh{col}] ([{offsets}] [])",
        ]

decls = [
    ("r26_gate_scaled", f"memref<{APACK if CANONICAL_BF16 else DATA_PAIR}xi8>, memref<{WB}xi8>, memref<{GATE_ACC}xi32>, i32"),
    ("r26_geglu_padded", f"memref<{GATE_ACC}xi32>, memref<{GATE_OUTPUT_BYTES}xi8>" if CANONICAL_BF16 else f"memref<{GATE_ACC}xi32>, memref<{OUTPUT_CO}xi32>"),
    (
        "r26_pack3",
        f"memref<{DATA_PAIR}xi8>, "
        + ("memref<9xf32>, " if DIRECT_X_PRE_NORM else "")
        + f"memref<{WB}xi8>, memref<{APACK}xi8>, memref<{SCRATCH}xf32>, memref<{OWN_FRAGMENT}xi8>, i32, i32",
    ),
    (
        "r26_insert_fragment",
        f"memref<{FRAGMENT if not REUSE_GATE_ACTIVATION else GATE_GROUPS * FRAGMENT}xi8>, memref<{APACK}xi8>, i32"
        + (", i32" if REUSE_GATE_ACTIVATION else ""),
    ),
    (
        "r26_send_fragment",
        f"memref<{FRAGMENT if not REUSE_GATE_ACTIVATION else GATE_GROUPS * FRAGMENT}xi8>"
        + (", i32" if REUSE_GATE_ACTIVATION else ""),
    ),
    ("r26_receive_fragment", f"memref<{OWN_FRAGMENT}xi8>"),
    ("r26_down0_scaled", f"memref<{APACK}xi8>, memref<{WB}xi8>, memref<{OUTPUT_CO}xi32>, i32"),
    ("r26_down1_scaled", f"memref<{APACK}xi8>, memref<{WB}xi8>, memref<{OUTPUT_CO}xi32>, i32"),
]
if DIRECT_X_PRE_NORM:
    decls.append(
        ("r45_select_inverses", f"memref<{WB}xi8>, memref<9xf32>, i32, i32")
    )
if REUSE_GATE_ACTIVATION:
    decls += [
        (
            "r55_pack3_cached",
            f"memref<{DATA_PAIR}xi8>, memref<9xf32>, memref<{WB}xi8>, memref<{APACK}xi8>, memref<{SCRATCH}xf32>, memref<{GATE_GROUPS * FRAGMENT}xi8>, i32, i32, i32",
        ),
    ]
if CANONICAL_BF16:
    decls.append(
        ("r35_finish_down48_bf16", f"memref<{GATE_ACC}xi32>, memref<{GATE_OUTPUT_BYTES}xi8>, i32" + (", i32" if CANONICAL_BF16X2_OUTPUT else ""))
    )
for name, args in decls:
    out.append(
        f'    func.func private @{name}({args}) attributes {{link_with = "r26.o"}}'
    )


def append_ring(
    lines,
    col,
    row,
    indent,
    insert_own=True,
    own=None,
    suffix="",
    cached=None,
    group=None,
):
    own = own or f"own{col}_{row}"
    fragment_memref = GATE_GROUPS * FRAGMENT if REUSE_GATE_ACTIVATION else FRAGMENT
    if insert_own:
        if cached:
            lines.append(
                f"{indent}func.call @r26_insert_fragment(%{cached}, %apack{col}_{row}, %owner, %{group}) : (memref<{fragment_memref}xi8>, memref<{APACK}xi8>, i32, i32) -> ()"
            )
        else:
            group_arg = ", %uncached_group" if REUSE_GATE_ACTIVATION else ""
            group_type = ", i32" if REUSE_GATE_ACTIVATION else ""
            lines.append(
                f"{indent}func.call @r26_insert_fragment(%{own}, %apack{col}_{row}, %owner{group_arg}) : (memref<{fragment_memref}xi8>, memref<{APACK}xi8>, i32{group_type}) -> ()"
            )
    for broadcast_owner in range(COLS):
        if col == broadcast_owner:
            if cached:
                lines.append(
                    f"{indent}func.call @r26_send_fragment(%{cached}, %{group}) : (memref<{fragment_memref}xi8>, i32) -> ()"
                )
            else:
                group_arg = ", %uncached_group" if REUSE_GATE_ACTIVATION else ""
                group_type = ", i32" if REUSE_GATE_ACTIVATION else ""
                lines.append(
                    f"{indent}func.call @r26_send_fragment(%{own}{group_arg}) : (memref<{fragment_memref}xi8>{group_type}) -> ()"
                )
        else:
            lines += [
                f"{indent}func.call @r26_receive_fragment(%transit{col}_{row}) : (memref<{OWN_FRAGMENT}xi8>) -> ()",
                f"{indent}%broadcast_owner{broadcast_owner}{suffix} = arith.constant {broadcast_owner} : i32",
                f"{indent}func.call @r26_insert_fragment(%transit{col}_{row}, %apack{col}_{row}, %broadcast_owner{broadcast_owner}{suffix}"
                + (", %uncached_group" if REUSE_GATE_ACTIVATION else "")
                + f") : (memref<{OWN_FRAGMENT}xi8>, memref<{APACK}xi8>, i32"
                + (", i32" if REUSE_GATE_ACTIVATION else "")
                + ") -> ()",
            ]
            if col != (broadcast_owner - 1) % COLS:
                lines.append(
                    f"{indent}func.call @r26_send_fragment(%transit{col}_{row}"
                    + (", %uncached_group" if REUSE_GATE_ACTIVATION else "")
                    + f") : (memref<{OWN_FRAGMENT}xi8>"
                    + (", i32" if REUSE_GATE_ACTIVATION else "")
                    + ") -> ()"
                )

for col in range(COLS):
    for row in range(CORE_ROWS):
        lines = [
            f"    %core{col}_{row} = aie.core(%c{col}_{row}) {{",
            "      %z = arith.constant 0 : index",
            f"      %inf = arith.constant {INF} : index",
            "      %one = arith.constant 1 : index",
            f"      %gate_outblocks = arith.constant {GATE_OUTBLOCKS} : index",
            f"      %gate_groups = arith.constant {GATE_GROUPS} : index",
            f"      %down_mblocks = arith.constant {DOWN_MBLOCKS} : index",
            f"      %down_groups = arith.constant {DOWN_GROUPS} : index",
            *(["      %down_nmacros = arith.constant 2 : index"] if CANONICAL_BF16 else []),
            f"      %owner = arith.constant {col} : i32",
            *(
                [
                    "      %uncached_group = arith.constant 0 : i32",
                ]
                if REUSE_GATE_ACTIVATION
                else []
            ),
            *(
                [f"      %core_row = arith.constant {row} : i32"]
                if DIRECT_X_PRE_NORM
                else []
            ),
            "      scf.for %outer = %z to %inf step %one {",
            *(
                [
                    f"        %iw = aie.objectfifo.acquire @wbc{col}(Consume, 1) : !aie.objectfifosubview<memref<{WB}xi8>>",
                    f"        %iwv = aie.objectfifo.subview.access %iw[0] : !aie.objectfifosubview<memref<{WB}xi8>> -> memref<{WB}xi8>",
                    f"        func.call @r45_select_inverses(%iwv, %inv{col}_{row}, %core_row, %owner) : (memref<{WB}xi8>, memref<9xf32>, i32, i32) -> ()",
                    f"        aie.objectfifo.release @wbc{col}(Consume, 1)",
                ]
                if DIRECT_X_PRE_NORM
                else []
            ),
            *(
                ["        scf.for %mblock = %z to %down_mblocks step %one {"]
                if REUSE_GATE_ACTIVATION
                else ["        scf.for %outblock = %z to %gate_outblocks step %one {"]
            ),
        ]
        if CANONICAL_BF16:
            if REUSE_GATE_ACTIVATION:
                lines += [
                    "          %mblock_i32 = arith.index_cast %mblock : index to i32",
                    "          %negative_one = arith.constant -1 : i32",
                    "          %negative_token = arith.subi %negative_one, %mblock_i32 : i32",
                ]
                lines += [
                    "          scf.for %pre_group = %z to %gate_groups step %one {",
                    f"            %px = aie.objectfifo.acquire @xbc{row}(Consume, 1) : !aie.objectfifosubview<memref<{DATA_PAIR}xi8>>",
                    f"            %pxv = aie.objectfifo.subview.access %px[0] : !aie.objectfifosubview<memref<{DATA_PAIR}xi8>> -> memref<{DATA_PAIR}xi8>",
                    f"            %pw = aie.objectfifo.acquire @wbc{col}(Consume, 1) : !aie.objectfifosubview<memref<{WB}xi8>>",
                    f"            %pwv = aie.objectfifo.subview.access %pw[0] : !aie.objectfifosubview<memref<{WB}xi8>> -> memref<{WB}xi8>",
                    "            %pre_group_i32 = arith.index_cast %pre_group : index to i32",
                    f"            func.call @r55_pack3_cached(%pxv, %inv{col}_{row}, %pwv, %apack{col}_{row}, %scratch{col}_{row}, %own{col}_{row}, %owner, %negative_token, %pre_group_i32) : (memref<{DATA_PAIR}xi8>, memref<9xf32>, memref<{WB}xi8>, memref<{APACK}xi8>, memref<{SCRATCH}xf32>, memref<{GATE_GROUPS * FRAGMENT}xi8>, i32, i32, i32) -> ()",
                    f"            aie.objectfifo.release @xbc{row}(Consume, 1)",
                    f"            aie.objectfifo.release @wbc{col}(Consume, 1)",
                    "          }",
                ]
                lines += ["          scf.for %nblock = %z to %gate_outblocks step %down_mblocks {"]
                lines += [
                    f"            %go = aie.objectfifo.acquire @oc{col}_{row}(Produce, 1) : !aie.objectfifosubview<memref<{GATE_OUTPUT_BYTES}xi8>>",
                    f"            %gov = aie.objectfifo.subview.access %go[0] : !aie.objectfifosubview<memref<{GATE_OUTPUT_BYTES}xi8>> -> memref<{GATE_OUTPUT_BYTES}xi8>",
                    "            scf.for %group = %z to %gate_groups step %one {",
                    f"              %w = aie.objectfifo.acquire @wbc{col}(Consume, 1) : !aie.objectfifosubview<memref<{WB}xi8>>",
                    f"              %wv = aie.objectfifo.subview.access %w[0] : !aie.objectfifosubview<memref<{WB}xi8>> -> memref<{WB}xi8>",
                    "              %accumulate = arith.index_cast %group : index to i32",
                ]
                append_ring(
                    lines,
                    col,
                    row,
                    "              ",
                    cached=f"own{col}_{row}",
                    group="accumulate",
                )
                lines += [
                    f"              func.call @r26_gate_scaled(%apack{col}_{row}, %wv, %gacc{col}_{row}, %accumulate) : (memref<{APACK}xi8>, memref<{WB}xi8>, memref<{GATE_ACC}xi32>, i32) -> ()",
                    f"              aie.objectfifo.release @wbc{col}(Consume, 1)",
                    "            }",
                    f"            func.call @r26_geglu_padded(%gacc{col}_{row}, %gov) : (memref<{GATE_ACC}xi32>, memref<{GATE_OUTPUT_BYTES}xi8>) -> ()",
                    f"            aie.objectfifo.release @oc{col}_{row}(Produce, 1)",
                    "          }",
                    "        }",
                ]
            elif DIRECT_X_PRE_NORM:
                lines += [
                    "          %outblock_i32 = arith.index_cast %outblock : index to i32",
                    "          %negative_one = arith.constant -1 : i32",
                    "          %negative_token = arith.subi %negative_one, %outblock_i32 : i32",
                ]
            if not REUSE_GATE_ACTIVATION:
                lines += [
                    f"          %go = aie.objectfifo.acquire @oc{col}_{row}(Produce, 1) : !aie.objectfifosubview<memref<{GATE_OUTPUT_BYTES}xi8>>",
                    f"          %gov = aie.objectfifo.subview.access %go[0] : !aie.objectfifosubview<memref<{GATE_OUTPUT_BYTES}xi8>> -> memref<{GATE_OUTPUT_BYTES}xi8>",
                    "          scf.for %group = %z to %gate_groups step %one {",
                    f"            %x = aie.objectfifo.acquire @xbc{row}(Consume, 1) : !aie.objectfifosubview<memref<{DATA_PAIR}xi8>>",
                    f"            %xv = aie.objectfifo.subview.access %x[0] : !aie.objectfifosubview<memref<{DATA_PAIR}xi8>> -> memref<{DATA_PAIR}xi8>",
                    f"            %w = aie.objectfifo.acquire @wbc{col}(Consume, 1) : !aie.objectfifosubview<memref<{WB}xi8>>",
                    f"            %wv = aie.objectfifo.subview.access %w[0] : !aie.objectfifosubview<memref<{WB}xi8>> -> memref<{WB}xi8>",
                    "            %accumulate = arith.index_cast %group : index to i32",
                    f"            func.call @r26_pack3(%xv, "
                    + (f"%inv{col}_{row}, " if DIRECT_X_PRE_NORM else "")
                    + f"%wv, %apack{col}_{row}, %scratch{col}_{row}, %own{col}_{row}, %owner, "
                    + ("%negative_token" if DIRECT_X_PRE_NORM else "%accumulate")
                    + f") : (memref<{DATA_PAIR}xi8>, "
                    + ("memref<9xf32>, " if DIRECT_X_PRE_NORM else "")
                    + f"memref<{WB}xi8>, memref<{APACK}xi8>, memref<{SCRATCH}xf32>, memref<{OWN_FRAGMENT}xi8>, i32, i32) -> ()",
                ]
                append_ring(lines, col, row, "            ")
                lines += [
                    f"            func.call @r26_gate_scaled(%apack{col}_{row}, %wv, %gacc{col}_{row}, %accumulate) : (memref<{APACK}xi8>, memref<{WB}xi8>, memref<{GATE_ACC}xi32>, i32) -> ()",
                    f"            aie.objectfifo.release @xbc{row}(Consume, 1)",
                    f"            aie.objectfifo.release @wbc{col}(Consume, 1)",
                    "          }",
                    f"          func.call @r26_geglu_padded(%gacc{col}_{row}, %gov) : (memref<{GATE_ACC}xi32>, memref<{GATE_OUTPUT_BYTES}xi8>) -> ()",
                    f"          aie.objectfifo.release @oc{col}_{row}(Produce, 1)",
                    "        }",
                ]
        else:
            lines += [
                f"          %go = aie.objectfifo.acquire @oc{col}_{row}(Produce, 1) : !aie.objectfifosubview<memref<{OUTPUT_CO}xi32>>",
                f"          %gov = aie.objectfifo.subview.access %go[0] : !aie.objectfifosubview<memref<{OUTPUT_CO}xi32>> -> memref<{OUTPUT_CO}xi32>",
                "          scf.for %group = %z to %gate_groups step %one {",
                f"            %x = aie.objectfifo.acquire @xpair{col // 2}_{row}(Consume, 1) : !aie.objectfifosubview<memref<{DATA_PAIR}xi8>>",
                f"            %xv = aie.objectfifo.subview.access %x[0] : !aie.objectfifosubview<memref<{DATA_PAIR}xi8>> -> memref<{DATA_PAIR}xi8>",
                f"            %w = aie.objectfifo.acquire @wbc{col}(Consume, 1) : !aie.objectfifosubview<memref<{WB}xi8>>",
                f"            %wv = aie.objectfifo.subview.access %w[0] : !aie.objectfifosubview<memref<{WB}xi8>> -> memref<{WB}xi8>",
                "            %accumulate = arith.index_cast %group : index to i32",
                f"            func.call @r26_gate_scaled(%xv, %wv, %gacc{col}_{row}, %accumulate) : (memref<{DATA_PAIR}xi8>, memref<{WB}xi8>, memref<{GATE_ACC}xi32>, i32) -> ()",
                f"            aie.objectfifo.release @xpair{col // 2}_{row}(Consume, 1)",
                f"            aie.objectfifo.release @wbc{col}(Consume, 1)",
                "          }",
                f"          func.call @r26_geglu_padded(%gacc{col}_{row}, %gov) : (memref<{GATE_ACC}xi32>, memref<{OUTPUT_CO}xi32>) -> ()",
                f"          aie.objectfifo.release @oc{col}_{row}(Produce, 1)",
                "        }",
                "        scf.for %mblock = %z to %down_mblocks step %one {",
                f"          %do = aie.objectfifo.acquire @oc{col}_{row}(Produce, 1) : !aie.objectfifosubview<memref<{OUTPUT_CO}xi32>>",
                f"          %dov = aie.objectfifo.subview.access %do[0] : !aie.objectfifosubview<memref<{OUTPUT_CO}xi32>> -> memref<{OUTPUT_CO}xi32>",
            ]
        if CANONICAL_BF16:
            lines += [
                "        scf.for %mblock = %z to %down_mblocks step %one {",
                "          scf.for %nmacro = %z to %down_nmacros step %one {",
                "            scf.for %group = %z to %down_groups step %one {",
                f"              %w0 = aie.objectfifo.acquire @wbc{col}(Consume, 1) : !aie.objectfifosubview<memref<{WB}xi8>>",
                f"              %w0v = aie.objectfifo.subview.access %w0[0] : !aie.objectfifosubview<memref<{WB}xi8>> -> memref<{WB}xi8>",
                f"              %x = aie.objectfifo.acquire @xbc{row}(Consume, 1) : !aie.objectfifosubview<memref<{DATA_PAIR}xi8>>",
                f"              %xv = aie.objectfifo.subview.access %x[0] : !aie.objectfifosubview<memref<{DATA_PAIR}xi8>> -> memref<{DATA_PAIR}xi8>",
                "              %group_i32 = arith.index_cast %group : index to i32",
                f"              func.call @r26_pack3(%xv, "
                + (f"%inv{col}_{row}, " if DIRECT_X_PRE_NORM else "")
                + f"%w0v, %apack{col}_{row}, %scratch{col}_{row}, %own{col}_{row}, %owner, %group_i32) : (memref<{DATA_PAIR}xi8>, "
                + ("memref<9xf32>, " if DIRECT_X_PRE_NORM else "")
                + f"memref<{WB}xi8>, memref<{APACK}xi8>, memref<{SCRATCH}xf32>, memref<{OWN_FRAGMENT}xi8>, i32, i32) -> ()",
                f"              aie.objectfifo.release @xbc{row}(Consume, 1)",
            ]
            append_ring(lines, col, row, "              ")
            lines += [
                "              %accumulate = arith.index_cast %group : index to i32",
                f"              func.call @r26_gate_scaled(%apack{col}_{row}, %w0v, %gacc{col}_{row}, %accumulate) : (memref<{APACK}xi8>, memref<{WB}xi8>, memref<{GATE_ACC}xi32>, i32) -> ()",
                f"              aie.objectfifo.release @wbc{col}(Consume, 1)",
                "            }",
            ]
            for component in range(2 if CANONICAL_BF16X2_OUTPUT else 1):
                for lane in range(2):
                    suffix = f"{component}_{lane}"
                    call_args = f"%gacc{col}_{row}, %do{suffix}v, %lane{suffix}"
                    call_types = f"memref<{GATE_ACC}xi32>, memref<{GATE_OUTPUT_BYTES}xi8>, i32"
                    lines += [
                        f"            %do{suffix} = aie.objectfifo.acquire @oc{col}_{row}(Produce, 1) : !aie.objectfifosubview<memref<{GATE_OUTPUT_BYTES}xi8>>",
                        f"            %do{suffix}v = aie.objectfifo.subview.access %do{suffix}[0] : !aie.objectfifosubview<memref<{GATE_OUTPUT_BYTES}xi8>> -> memref<{GATE_OUTPUT_BYTES}xi8>",
                        f"            %lane{suffix} = arith.constant {lane} : i32",
                    ]
                    if CANONICAL_BF16X2_OUTPUT:
                        lines += [
                            f"            %component{suffix} = arith.constant {component} : i32",
                            f"            func.call @r35_finish_down48_bf16({call_args}, %component{suffix}) : ({call_types}, i32) -> ()",
                        ]
                    else:
                        lines.append(
                            f"            func.call @r35_finish_down48_bf16({call_args}) : ({call_types}) -> ()"
                        )
                    lines.append(f"            aie.objectfifo.release @oc{col}_{row}(Produce, 1)")
            lines += ["          }", "        }"]
        else:
            input_fifo = f"xpair{col // 2}_{row}"
            lines += [
                "          scf.for %group = %z to %down_groups step %one {",
                f"            %w0 = aie.objectfifo.acquire @wbc{col}(Consume, 1) : !aie.objectfifosubview<memref<{WB}xi8>>",
                f"            %w0v = aie.objectfifo.subview.access %w0[0] : !aie.objectfifosubview<memref<{WB}xi8>> -> memref<{WB}xi8>",
                f"            %x = aie.objectfifo.acquire @{input_fifo}(Consume, 1) : !aie.objectfifosubview<memref<{DATA_PAIR}xi8>>",
                f"            %xv = aie.objectfifo.subview.access %x[0] : !aie.objectfifosubview<memref<{DATA_PAIR}xi8>> -> memref<{DATA_PAIR}xi8>",
                "            %group_i32 = arith.index_cast %group : index to i32",
                f"            func.call @r26_pack3(%xv, %w0v, %apack{col}_{row}, %scratch{col}_{row}, %own{col}_{row}, %owner, %group_i32) : (memref<{DATA_PAIR}xi8>, memref<{WB}xi8>, memref<{APACK}xi8>, memref<{SCRATCH}xf32>, memref<{OWN_FRAGMENT}xi8>, i32, i32) -> ()",
                f"            func.call @r26_insert_fragment(%own{col}_{row}, %apack{col}_{row}, %owner) : (memref<{FRAGMENT}xi8>, memref<{APACK}xi8>, i32) -> ()",
                f"            aie.objectfifo.release @{input_fifo}(Consume, 1)",
            ]
            append_ring(lines, col, row, "            ", insert_own=False)
            lines += [
                "            %accumulate = arith.index_cast %group : index to i32",
                f"            func.call @r26_down0_scaled(%apack{col}_{row}, %w0v, %dov, %accumulate) : (memref<{APACK}xi8>, memref<{WB}xi8>, memref<{OUTPUT_CO}xi32>, i32) -> ()",
                f"            aie.objectfifo.release @wbc{col}(Consume, 1)",
                f"            %w1 = aie.objectfifo.acquire @wbc{col}(Consume, 1) : !aie.objectfifosubview<memref<{WB}xi8>>",
                f"            %w1v = aie.objectfifo.subview.access %w1[0] : !aie.objectfifosubview<memref<{WB}xi8>> -> memref<{WB}xi8>",
                f"            func.call @r26_down1_scaled(%apack{col}_{row}, %w1v, %dov, %accumulate) : (memref<{APACK}xi8>, memref<{WB}xi8>, memref<{OUTPUT_CO}xi32>, i32) -> ()",
                f"            aie.objectfifo.release @wbc{col}(Consume, 1)",
                "          }",
                f"          aie.objectfifo.release @oc{col}_{row}(Produce, 1)",
                "        }",
            ]
        lines += [
            "      }",
            "      aie.end",
            "    } {stack_size = 4096 : i32}",
        ]
        out += lines

DATA_ROW = GATE_DATA_BLOCKS * DATA_JOIN
WT = WEIGHT_BLOCKS * WB
if CANONICAL_BF16:
    out.append(
        f"    aie.runtime_sequence(%D: memref<{RUNTIME_INPUT_BYTES}xi8>, "
        f"%W: memref<{COLS * WT}xi8>, %T: memref<{CANONICAL_T_BYTES}xi8>, "
        f"%O: memref<{CANONICAL_OUTPUT_BYTES}xi8>) {{"
    )
else:
    out.append(
        f"    aie.runtime_sequence(%D: memref<{CORE_ROWS * DATA_ROW}xi8>, "
        f"%W: memref<{COLS * WT}xi8>, %T: memref<{T_ROWS * T_STRIDE}xf32>, "
        f"%O: memref<{O_ELEMS}xi32>) {{"
    )
    for row in range(CORE_ROWS):
        out += [
            f"      %tg{row} = aiex.dma_configure_task_for @xsh{row} {{",
            f"        aie.dma_bd(%D : memref<{CORE_ROWS * DATA_ROW}xi8>, {row * DATA_ROW}, {DATA_ROW}, {byte_blocks(GATE_DATA_BLOCKS, DATA_JOIN)}) {{burst_length = 0 : i32}}",
            "        aie.end",
            "      } {issue_token = true}",
            f"      aiex.dma_start_task(%tg{row})",
        ]
if DIRECT_X_PRE_NORM:
    inverse_dims = (
        f"[<size = {PRE_INVERSE_BASE // PRE_INVERSE_RECORD_BYTES}, stride = {PRE_INVERSE_RECORD_BYTES}>, "
        "<size = 512, stride = 1>]"
    )
    for col in range(COLS):
        out += [
            f"      %ti{col} = aiex.dma_configure_task_for @wsh{col} {{",
            f"        aie.dma_bd(%D : memref<{RUNTIME_INPUT_BYTES}xi8>, {PRE_INVERSE_BASE}, {WB}, {inverse_dims}) {{burst_length = 0 : i32}}",
            "        aie.end",
            "      } {issue_token = true}",
            f"      aiex.dma_start_task(%ti{col})",
            f"      aiex.dma_await_task(%ti{col})",
            f"      aiex.dma_free_task(%ti{col})",
        ]

for col in range(COLS):
    out += [
        f"      %tw{col} = aiex.dma_configure_task_for @wsh{col} {{",
        f"        aie.dma_bd(%W : memref<{COLS * WT}xi8>, {col * WT}, {WT}, {byte_blocks(WEIGHT_BLOCKS, WB)}) {{burst_length = 0 : i32}}",
        "        aie.end",
        "      } {issue_token = true}",
        f"      aiex.dma_start_task(%tw{col})",
    ]

for outblock in range(GATE_OUTBLOCKS):
    mblock, nblock = divmod(outblock, GATE_N_MACROS)
    for col in range(COLS):
        name = f"gt{col}_{outblock}"
        if CANONICAL_BF16:
            offset = (mblock * 96 * PAD_INTERMEDIATE + nblock * 192 + col * 24) * 2
            out += [
                f"      %{name} = aiex.dma_configure_task_for @osh{col} {{",
                f"        aie.dma_bd(%T : memref<{CANONICAL_T_BYTES}xi8>, {offset}, {CORE_ROWS * GATE_OUTPUT_BYTES}, {canonical_gate_output_dims()}) {{burst_length = 0 : i32}}",
                "        aie.end",
                "      } {issue_token = true}",
                f"      aiex.dma_start_task(%{name})",
            ]
        else:
            offset = mblock * 96 * T_STRIDE + nblock * 8 * 96 + col * 96
            out += [
                f"      %{name} = aiex.dma_configure_task_for @osh{col} {{",
                f"        aie.dma_bd(%T : memref<{T_ROWS * T_STRIDE}xf32>, {offset}, 288, {gate_output_dims()}) {{burst_length = 0 : i32}}",
                "        aie.end",
                "      } {issue_token = true, repeat_count = 31 : i32}",
                f"      aiex.dma_start_task(%{name})",
            ]
    if CANONICAL_BF16 and (not REUSE_GATE_ACTIVATION or nblock == 0):
        for group in range(GATE_GROUPS):
            for row in range(CORE_ROWS):
                name = f"gx{row}_{outblock}_{group}"
                offset = ((mblock * 96 + row * 24) * 768 + group * 256) * 2
                out += [
                    f"      %{name} = aiex.dma_configure_task_for @xsh{row} {{",
                    f"        aie.dma_bd(%D : memref<{RUNTIME_INPUT_BYTES}xi8>, {offset}, {DATA_PAIR}, {canonical_input_dims()}) {{burst_length = 0 : i32}}",
                    "        aie.end",
                    "      } {issue_token = true}",
                    f"      aiex.dma_start_task(%{name})",
                ]
            for row in range(CORE_ROWS):
                name = f"gx{row}_{outblock}_{group}"
                out += [
                    f"      aiex.dma_await_task(%{name})",
                    f"      aiex.dma_free_task(%{name})",
                ]
    for col in range(COLS):
        name = f"gt{col}_{outblock}"
        out += [f"      aiex.dma_await_task(%{name})", f"      aiex.dma_free_task(%{name})"]

if not CANONICAL_BF16:
    for row in range(CORE_ROWS):
        out += [f"      aiex.dma_await_task(%tg{row})", f"      aiex.dma_free_task(%tg{row})"]

if CANONICAL_BF16:
    for mblock in range(DOWN_MBLOCKS):
        for nmacro in range(2):
            for component in range(2 if CANONICAL_BF16X2_OUTPUT else 1):
              for lane in range(2):
                for col in range(COLS):
                    name = f"do{col}_{mblock}_{nmacro}_{component}_{lane}"
                    offset = (
                        (mblock * 96) * OUTPUT * CANONICAL_OUTPUT_COMPONENTS
                        + component * OUTPUT
                        + nmacro * 384
                        + col * 48
                        + lane * 24
                    ) * 2
                    out += [
                        f"      %{name} = aiex.dma_configure_task_for @osh{col} {{",
                        f"        aie.dma_bd(%O : memref<{CANONICAL_OUTPUT_BYTES}xi8>, {offset}, {CORE_ROWS * GATE_OUTPUT_BYTES}, {canonical_down_output_dims()}) {{burst_length = 0 : i32}}",
                        "        aie.end",
                        "      } {issue_token = true}",
                        f"      aiex.dma_start_task(%{name})",
                    ]
            for group in range(DOWN_GROUPS):
                for row in range(CORE_ROWS):
                    name = f"dx{row}_{mblock}_{nmacro}_{group}"
                    offset = ((mblock * 96 + row * 24) * PAD_INTERMEDIATE + group * 256) * 2
                    out += [
                        f"      %{name} = aiex.dma_configure_task_for @xsh{row} {{",
                        f"        aie.dma_bd(%T : memref<{CANONICAL_T_BYTES}xi8>, {offset}, {DATA_PAIR}, {canonical_down_input_dims()}) {{burst_length = 0 : i32}}",
                        "        aie.end",
                        "      } {issue_token = true}",
                        f"      aiex.dma_start_task(%{name})",
                    ]
                for row in range(CORE_ROWS):
                    name = f"dx{row}_{mblock}_{nmacro}_{group}"
                    out += [
                        f"      aiex.dma_await_task(%{name})",
                        f"      aiex.dma_free_task(%{name})",
                    ]
            for component in range(2 if CANONICAL_BF16X2_OUTPUT else 1):
                for lane in range(2):
                    for col in range(COLS):
                        name = f"do{col}_{mblock}_{nmacro}_{component}_{lane}"
                        out += [
                            f"      aiex.dma_await_task(%{name})",
                            f"      aiex.dma_free_task(%{name})",
                        ]
else:
    for mblock in range(DOWN_MBLOCKS):
        for col in range(COLS):
            offset = mblock * 96 * OUTPUT + col * 48
            name = f"do{col}_{mblock}"
            out += [
                f"      %{name} = aiex.dma_configure_task_for @osh{col} {{",
                f"        aie.dma_bd(%O : memref<{O_ELEMS}xi32>, {offset}, 3072, {down_output_dims()}) {{burst_length = 0 : i32}}",
                "        aie.end",
                "      } {issue_token = true, repeat_count = 2 : i32}",
                f"      aiex.dma_start_task(%{name})",
            ]
        for group in range(DOWN_GROUPS):
            for row in range(CORE_ROWS):
                name = f"dx{row}_{mblock}_{group}"
                base_row = mblock * 96 + row * 24
                first_chunk = (group * 256) // 24
                offset = base_row * T_STRIDE + first_chunk * 96
                out += [
                    f"      %{name} = aiex.dma_configure_task_for @xsh{row} {{",
                    f"        aie.dma_bd(%T : memref<{T_ROWS * T_STRIDE}xf32>, {offset}, 2304, {down_input_dims()}) {{burst_length = 0 : i32}}",
                    "        aie.end",
                    "      } {issue_token = true, repeat_count = 3 : i32}",
                    f"      aiex.dma_start_task(%{name})",
                ]
            for row in range(CORE_ROWS):
                name = f"dx{row}_{mblock}_{group}"
                out += [f"      aiex.dma_await_task(%{name})", f"      aiex.dma_free_task(%{name})"]
        for col in range(COLS):
            name = f"do{col}_{mblock}"
            out += [f"      aiex.dma_await_task(%{name})", f"      aiex.dma_free_task(%{name})"]

for col in range(COLS):
    out += [f"      aiex.dma_await_task(%tw{col})", f"      aiex.dma_free_task(%tw{col})"]
out += ["    }", "  }", "}"]
print("\n".join(out))
