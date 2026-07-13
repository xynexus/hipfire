#!/usr/bin/env python3
"""Resident W4 EmbeddingGemma FFN with on-array GeGLU-to-down transpose."""

import sys
import os

PROBE = len(sys.argv) == 2 and sys.argv[1] == "probe"
PROBE_GATE = PROBE and os.environ.get("HIPFIRE_R25_GATE_PROBE") is not None
PROBE_GATE_ONLY = PROBE_GATE and os.environ.get("HIPFIRE_R25_GATE_ONLY_STREAM") is not None
PROBE_GATE_RAW_ONLY = PROBE_GATE and os.environ.get("HIPFIRE_R25_GATE_RAW_ONLY") is not None
PROBE_INPUTS = PROBE and os.environ.get("HIPFIRE_R25_INPUT_PROBE") is not None
PROBE_STREAM_ONLY = (
    PROBE
    and os.environ.get("HIPFIRE_R25_GATE_ONLY_STREAM") is not None
    and (PROBE_GATE or PROBE_INPUTS)
)
PROBE_RAW = PROBE and os.environ.get("HIPFIRE_R25_RAW_GATE_PROBE") is not None
PROBE_RAW_DIRECT = PROBE_RAW and os.environ.get("HIPFIRE_R25_RAW_DIRECT") is not None
PROBE_RAW_WARMUP = PROBE_RAW and os.environ.get("HIPFIRE_R25_RAW_WARMUP") is not None
PROBE_RAW_DOUBLE = PROBE_RAW and os.environ.get("HIPFIRE_R25_RAW_DOUBLE") is not None
PROBE_RAW_W_WARMUP = PROBE_RAW and os.environ.get("HIPFIRE_R25_RAW_W_WARMUP") is not None
PROBE_RAW_W_TAIL = PROBE_RAW and os.environ.get("HIPFIRE_R25_RAW_W_TAIL") is not None
WEIGHT_SCAN = os.environ.get("HIPFIRE_R25_WEIGHT_SCAN", "0") != "0"
WEIGHT_INITIAL_SCAN = os.environ.get("HIPFIRE_R25_WEIGHT_INITIAL_SCAN", "0") != "0"
WEIGHT_SHIM_DEPTH = int(os.environ.get("HIPFIRE_R25_WEIGHT_SHIM_DEPTH", "2"))
WEIGHT_CORE_DEPTH = int(os.environ.get("HIPFIRE_R25_WEIGHT_CORE_DEPTH", "2"))
WEIGHT_DIRECT = os.environ.get("HIPFIRE_R25_WEIGHT_DIRECT", "0") != "0"
OUTPUT_MEMTILE_DEPTH = int(os.environ.get("HIPFIRE_R25_OUTPUT_MEMTILE_DEPTH", "1"))
PROBE_FULL = PROBE and os.environ.get("HIPFIRE_R25_FULL_PROBE") is not None
PROBE_GROUP = int(os.environ.get("HIPFIRE_R25_PROBE_GROUP", "1"))
PROBE_RAW_GROUP = int(os.environ.get("HIPFIRE_R25_RAW_GROUP", "2"))
PROBE_RAW_NBLOCK = int(os.environ.get("HIPFIRE_R25_RAW_NBLOCK", "0"))
PROBE_MBLOCK = int(os.environ.get("HIPFIRE_R25_PROBE_MBLOCK", "1"))
PROBE_NBLOCK = int(os.environ.get("HIPFIRE_R25_PROBE_NBLOCK", "0"))
ZERO_ACCUM_GATE = os.environ.get("HIPFIRE_R25_ZERO_ACCUM_GATE") is not None
DYNAMIC_GEMM = os.environ.get("HIPFIRE_R25_DYNAMIC_GEMM") is not None
COMPACT_FRAGMENT_RING = os.environ.get("HIPFIRE_R25_COMPACT_FRAGMENT_RING") is not None
COMPACT_DOWN_BRANCHES = os.environ.get("HIPFIRE_R25_COMPACT_DOWN_BRANCHES") is not None
CANONICAL_GATE = os.environ.get("HIPFIRE_R25_CANONICAL_GATE") is not None
DIRECT_X_ROW_STATE = os.environ.get("HIPFIRE_R25_DIRECT_X_ROW_STATE") is not None
DIRECT_X_NORMALIZE = os.environ.get("HIPFIRE_R25_DIRECT_X_NORMALIZE") is not None
DIRECT_X_FULL_OBJECT = os.environ.get("HIPFIRE_R25_DIRECT_X_FULL_OBJECT") is not None
INTERLEAVED_BF16X2_OUTPUT = os.environ.get("HIPFIRE_R25_INTERLEAVED_BF16X2_OUTPUT") is not None
ACTIVATION_FENCE = os.environ.get("HIPFIRE_R25_ACTIVATION_FENCE") is not None
ACTIVATION_SCAN = os.environ.get("HIPFIRE_R25_ACTIVATION_SCAN") is not None
GATE_WRAPPER = os.environ.get("HIPFIRE_R25_GATE_WRAPPER") is not None
GATE_ACTIVATION_HASH = os.environ.get("HIPFIRE_R25_GATE_ACTIVATION_HASH") is not None
SEPARATE_GATE_A = os.environ.get("HIPFIRE_R25_SEPARATE_GATE_A") is not None
PRECALL_PACK = os.environ.get("HIPFIRE_R25_PRECALL_PACK") is not None
PRECALL_RING = os.environ.get("HIPFIRE_R25_PRECALL_RING") is not None
LOCAL_A_COPY = os.environ.get("HIPFIRE_R25_LOCAL_A_COPY") is not None
if SEPARATE_GATE_A and not PROBE_STREAM_ONLY:
    raise SystemExit("HIPFIRE_R25_SEPARATE_GATE_A is a stream-only diagnostic")
if DIRECT_X_ROW_STATE and not CANONICAL_GATE:
    raise SystemExit("HIPFIRE_R25_DIRECT_X_ROW_STATE requires canonical gate input")
if DIRECT_X_NORMALIZE and not CANONICAL_GATE:
    raise SystemExit("HIPFIRE_R25_DIRECT_X_NORMALIZE requires canonical gate input")
if DIRECT_X_FULL_OBJECT and not DIRECT_X_NORMALIZE:
    raise SystemExit("HIPFIRE_R25_DIRECT_X_FULL_OBJECT requires direct-X normalization")
PROBE_TARGETS = [(0, 0), (1, 0), (1, 1), (2, 0), (2, 1)]
PROBE_TARGET = PROBE_TARGETS[PROBE_GROUP]

COLS, CORE_ROWS = 8, 4
M_MACROS = 3
N_MACROS = int(os.environ.get("HIPFIRE_R25_N_MACROS", "3"))
GATE_GROUPS, DOWN_GROUPS = 3, 5
GROUP = 256
AB, WB, CB, TILE = 6656, 15872, 2304, 1152
FFN = AB if CANONICAL_GATE or LOCAL_A_COPY else 6240
PREP = 3 * GROUP
SCRATCH, FRAGMENT, PACK_SCALES = 384, 784, 4
CARRY = 0 if PROBE_STREAM_ONLY else 384
SAVED = (6 if GATE_ACTIVATION_HASH else 0) if PROBE_STREAM_ONLY else 768
PAD_M, PAD_N = 288, 768
CANONICAL_ROW = 1664 if DIRECT_X_ROW_STATE else 768 * 2
CANONICAL_BYTES = PAD_M * CANONICAL_ROW
CANONICAL_CORE = 3 * CANONICAL_ROW if DIRECT_X_ROW_STATE or DIRECT_X_FULL_OBJECT else 3 * GROUP * 2
CANONICAL_JOIN = CORE_ROWS * CANONICAL_CORE
OUTPUT_ROW_WORDS = int(os.environ.get("HIPFIRE_R25_OUTPUT_ROW_WORDS", str(PAD_N)))
INF = 9223372036854775807
A_BLOCKS = M_MACROS * N_MACROS * GATE_GROUPS
W_BLOCKS = M_MACROS * N_MACROS * GATE_GROUPS if PROBE_STREAM_ONLY else M_MACROS * (N_MACROS * GATE_GROUPS + DOWN_GROUPS)


def dims(count, block):
    return f"[<size = {count}, stride = {block}>, <size = {block // 512}, stride = 512>, <size = 512, stride = 1>]"


def output_dims():
    return f"[<size = 24, stride = {4 * OUTPUT_ROW_WORDS}>, <size = 6, stride = 16>, <size = 4, stride = {OUTPUT_ROW_WORDS}>, <size = 16, stride = 1>]"


out = ["module {", "  aie.device(npu2) {"]
for col in range(COLS):
    out += [f"    %shim{col} = aie.tile({col}, 0)", f"    %mt{col} = aie.tile({col}, 1)"]
    for row in range(CORE_ROWS):
        out += [
            f"    %c{col}_{row} = aie.tile({col}, {row + 2})",
            f"    %saved{col}_{row} = aie.buffer(%c{col}_{row}) {{sym_name = \"saved{col}_{row}\"}} : memref<{SAVED}xf32>",
            f"    %ffn{col}_{row} = aie.buffer(%c{col}_{row}) {{sym_name = \"ffn{col}_{row}\"}} : memref<{FFN}xi8>",
            *(
                [f"    %prep{col}_{row} = aie.buffer(%c{col}_{row}) {{sym_name = \"prep{col}_{row}\"}} : memref<{PREP}xf32>"]
                if SEPARATE_GATE_A
                else []
            ),
            f"    %scratch{col}_{row} = aie.buffer(%c{col}_{row}) {{sym_name = \"scratch{col}_{row}\"}} : memref<{SCRATCH}xf32>",
            f"    %packscales{col}_{row} = aie.buffer(%c{col}_{row}) {{sym_name = \"packscales{col}_{row}\"}} : memref<{PACK_SCALES}xf32>",
            *(
                [f"    %preinv{col}_{row} = aie.buffer(%c{col}_{row}) {{sym_name = \"preinv{col}_{row}\"}} : memref<4xf32>"]
                if DIRECT_X_NORMALIZE
                else []
            ),
            f"    %own{col}_{row} = aie.buffer(%c{col}_{row}) {{sym_name = \"own{col}_{row}\"}} : memref<{FRAGMENT}xi8>",
            f"    %transit{col}_{row} = aie.buffer(%c{col}_{row}) {{sym_name = \"transit{col}_{row}\"}} : memref<{FRAGMENT}xi8>",
            *(
                [
                    f"    %gateown{col}_{row} = aie.buffer(%c{col}_{row}) {{sym_name = \"gateown{col}_{row}\"}} : memref<{FRAGMENT}xi8>",
                    f"    %gatetransit{col}_{row} = aie.buffer(%c{col}_{row}) {{sym_name = \"gatetransit{col}_{row}\"}} : memref<{FRAGMENT}xi8>",
                ]
                if CANONICAL_GATE
                else []
            ),
            f"    %carry{col}_{row} = aie.buffer(%c{col}_{row}) {{sym_name = \"carry{col}_{row}\"}} : memref<{CARRY}xf32>",
        ]
for col in range(COLS):
    cores = ", ".join(f"%c{col}_{row}" for row in range(CORE_ROWS))
    if WEIGHT_DIRECT:
        out.append(
            f"    aie.objectfifo @wbc{col}(%shim{col}, {{{cores}}}, {WEIGHT_CORE_DEPTH} : i32) : !aie.objectfifo<memref<{WB}xi8>>"
        )
    else:
        out += [
            f"    aie.objectfifo @wsh{col}(%shim{col}, {{%mt{col}}}, {WEIGHT_SHIM_DEPTH} : i32) : !aie.objectfifo<memref<{WB}xi8>>",
            f"    aie.objectfifo @wbc{col}(%mt{col}, {{{cores}}}, {WEIGHT_CORE_DEPTH} : i32) : !aie.objectfifo<memref<{WB}xi8>>",
            f"    aie.objectfifo.link [@wsh{col}] -> [@wbc{col}] ([] [0])",
        ]
for row in range(CORE_ROWS):
    for col in range(COLS):
        out.append(f'    aie.flow(%c{col}_{row}, "Core" : 0, %c{(col + 1) % COLS}_{row}, "Core" : 0)')
for col in range(COLS):
    inputs = ", ".join(f"@cc{col}_{row}" for row in range(CORE_ROWS))
    offsets = ", ".join(str(row * CB) for row in range(CORE_ROWS))
    for row in range(CORE_ROWS):
        out.append(f"    aie.objectfifo @cc{col}_{row}(%c{col}_{row}, {{%mt{col}}}, 1 : i32) : !aie.objectfifo<memref<{CB}xi32>>")
    out += [
        f"    aie.objectfifo @csh{col}(%mt{col}, {{%shim{col}}}, {OUTPUT_MEMTILE_DEPTH} : i32) : !aie.objectfifo<memref<{CORE_ROWS * CB}xi32>>",
        f"    aie.objectfifo.link [{inputs}] -> [@csh{col}] ([{offsets}] [])",
    ]
if CANONICAL_GATE:
    for col in range(COLS):
        consumers = []
        offsets = []
        for row in range(CORE_ROWS):
            consumers.append(f"@xc{col}_{row}")
            offsets.append(str(row * CANONICAL_CORE))
            out.append(
                f"    aie.objectfifo @xc{col}_{row}(%mt{col}, {{%c{col}_{row}}}, 1 : i32) : !aie.objectfifo<memref<{CANONICAL_CORE}xi8>>"
            )
        out += [
            f"    aie.objectfifo @xsh{col}(%shim{col}, {{%mt{col}}}, 1 : i32) : !aie.objectfifo<memref<{CANONICAL_JOIN}xi8>>",
            f"    aie.objectfifo.link [@xsh{col}] -> [{', '.join(consumers)}] ([] [{', '.join(offsets)}])",
        ]
else:
    for row in range(CORE_ROWS):
        cores = ", ".join(f"%c{col}_{row}" for col in range(COLS))
        out += [
            f"    aie.objectfifo @ash{row}(%shim{row}, {{%mt{row}}}, 1 : i32) : !aie.objectfifo<memref<{AB}xi8>>",
            f"    aie.objectfifo @abc{row}(%mt{row}, {{{cores}}}, 1 : i32) : !aie.objectfifo<memref<{AB}xi8>>",
            f"    aie.objectfifo.link [@ash{row}] -> [@abc{row}] ([] [0])",
        ]

decls = [
    ("r25_zero", f"memref<{CB}xi32>"),
    ("r25_down", f"memref<{FFN}xi8>, memref<{WB}xi8>, memref<{CB}xi32>, i32"),
    *(
        [("r15_w4_scaled_dynamic", f"memref<{AB}xi8>, memref<{WB}xi8>, memref<{CB}xi32>, i32")]
        if DYNAMIC_GEMM
        else [
            ("r15_w4_scaled_accum", f"memref<{AB}xi8>, memref<{WB}xi8>, memref<{CB}xi32>"),
            ("r15_w4_scaled_init", f"memref<{AB}xi8>, memref<{WB}xi8>, memref<{CB}xi32>"),
        ]
    ),
    ("r25_touch_weight", f"memref<{WB}xi8>, memref<{SCRATCH}xf32>"),
    ("r25_touch_weight_tail", f"memref<{WB}xi8>, memref<{SCRATCH}xf32>"),
    ("r25_wait_weight", f"memref<{WB}xi8>, memref<{SCRATCH}xf32>"),
    ("r25_geglu_inplace", f"memref<{CB}xi32>, memref<{SCRATCH}xf32>"),
    ("r25_pack3", f"memref<{PREP}xf32>, memref<{CARRY}xf32>, memref<{WB}xi8>, memref<{SCRATCH}xf32>, memref<{PACK_SCALES}xf32>, memref<{FRAGMENT}xi8>, i32") if SEPARATE_GATE_A else ("r25_pack3", f"memref<{FFN}xi8>, memref<{CARRY}xf32>, memref<{WB}xi8>, memref<{SCRATCH}xf32>, memref<{PACK_SCALES}xf32>, memref<{FRAGMENT}xi8>, i32"),
    ("r25_save_carry_generic", f"memref<{FFN}xi8>, memref<{CARRY}xf32>, i32, i32"),
    ("r25_save_tile", f"memref<{FFN}xi8>, memref<{SAVED}xf32>, i32, i32"),
    ("r25_restore_tile", f"memref<{SAVED}xf32>, memref<{FFN}xi8>, i32"),
    ("r25_spill_down_bf16", f"memref<{CB}xi32>, memref<{SAVED}xf32>, memref<{FRAGMENT}xi8>, memref<{FRAGMENT}xi8>"),
    ("r25_restore_down_bf16", f"memref<{SAVED}xf32>, memref<{FRAGMENT}xi8>, memref<{FRAGMENT}xi8>, memref<{CB}xi32>"),
    ("r25_extract_local", f"memref<{CB}xi32>, memref<{FFN}xi8>, i32, i32"),
    ("r25_send_words", f"memref<{CB}xi32>, i32"),
    ("r25_receive_tile", f"memref<{FFN}xi8>, i32, i32, i32"),
    ("r25_insert_fragment", f"memref<{FRAGMENT}xi8>, memref<{FFN}xi8>, i32"),
    ("r25_send_fragment", f"memref<{FRAGMENT}xi8>"),
    ("r25_receive_fragment", f"memref<{FRAGMENT}xi8>"),
    *(([("r25_exchange_fragments", f"memref<{FFN}xi8>, memref<{FRAGMENT}xi8>, memref<{FRAGMENT}xi8>, i32")]) if COMPACT_FRAGMENT_RING else []),
    *(([(
        "r104_bf16_x_inverse_to_f32_3"
        if DIRECT_X_NORMALIZE
        else "r101_bf16_x_inverse_to_f32_3"
        if DIRECT_X_ROW_STATE
        else "r97_bf16_to_f32_3",
        (
            f"memref<{CANONICAL_CORE}xi8>, memref<4xf32>, "
            if DIRECT_X_NORMALIZE
            else f"memref<{CANONICAL_CORE}xi8>, "
            if DIRECT_X_ROW_STATE
            else f"memref<{CANONICAL_CORE}xi8>, "
        )
        + (f"memref<{PREP}xf32>" if SEPARATE_GATE_A else f"memref<{FFN}xi8>")
        + (", i32" if DIRECT_X_FULL_OBJECT else "")
        + (", i32" if DIRECT_X_ROW_STATE else ""),
    )]) if CANONICAL_GATE else []),
    *(([
        (
            "r104_rms_accumulate3",
            f"memref<{CANONICAL_CORE}xi8>, memref<4xf32>"
            + ("" if DIRECT_X_FULL_OBJECT else ", i32"),
        ),
    ]) if DIRECT_X_NORMALIZE else []),
    *(([("r98_finish_interleaved_bf16x2", f"memref<{CB}xi32>")]) if INTERLEAVED_BF16X2_OUTPUT else []),
    *(([("r97_activation_fence", f"memref<{FFN}xi8>")]) if CANONICAL_GATE and ACTIVATION_FENCE else []),
    *(([("r97_scan_activation", f"memref<{FFN}xi8>, memref<{SCRATCH}xf32>")]) if CANONICAL_GATE and ACTIVATION_SCAN else []),
    *(([("r97_snapshot_activation_hash", f"memref<{FFN}xi8>, memref<{SAVED}xf32>, i32"), ("r97_snapshot_weight_hash", f"memref<{WB}xi8>, memref<{SAVED}xf32>, i32"), ("r97_emit_activation_hashes", f"memref<{SAVED}xf32>, memref<{CB}xi32>")]) if CANONICAL_GATE and GATE_ACTIVATION_HASH else []),
    *(([("r97_copy_activation", f"memref<{AB}xi8>, memref<{FFN}xi8>")]) if LOCAL_A_COPY else []),
    *(([("r97_probe_source", f"memref<{CANONICAL_CORE}xi8>, memref<{CB}xi32>, i32")]) if CANONICAL_GATE and PROBE_INPUTS else []),
]
for name, args in decls:
    out.append(f'    func.func private @{name}({args}) attributes {{link_with = "r25.o"}}')


def gate_gemm(indent, activation, weight, output, accumulate, tag):
    if GATE_WRAPPER:
        flag = 1 if accumulate else 0
        return [
            f"{indent}%{tag}accumulate = arith.constant {flag} : i32",
            f"{indent}func.call @r25_down(%{activation}, %{weight}, %{output}, %{tag}accumulate) : (memref<{FFN}xi8>, memref<{WB}xi8>, memref<{CB}xi32>, i32) -> ()",
        ]
    if DYNAMIC_GEMM:
        flag = 1 if accumulate else 0
        return [
            f"{indent}%{tag}accumulate = arith.constant {flag} : i32",
            f"{indent}func.call @r15_w4_scaled_dynamic(%{activation}, %{weight}, %{output}, %{tag}accumulate) : (memref<{AB}xi8>, memref<{WB}xi8>, memref<{CB}xi32>, i32) -> ()",
        ]
    function = "r15_w4_scaled_accum" if accumulate else "r15_w4_scaled_init"
    return [
        f"{indent}func.call @{function}(%{activation}, %{weight}, %{output}) : (memref<{AB}xi8>, memref<{WB}xi8>, memref<{CB}xi32>) -> ()"
    ]


def gate_fragment_exchange(indent, col, row, tag, own="own", transit="transit"):
    lines = [
        f"{indent}func.call @r25_insert_fragment(%{own}{col}_{row}, %ffn{col}_{row}, %owner) : (memref<{FRAGMENT}xi8>, memref<{FFN}xi8>, i32) -> ()"
    ]
    for source in range(COLS):
        if col == source:
            lines.append(
                f"{indent}func.call @r25_send_fragment(%{own}{col}_{row}) : (memref<{FRAGMENT}xi8>) -> ()"
            )
        else:
            forward = col != (source - 1) % COLS
            lines += [
                f"{indent}func.call @r25_receive_fragment(%{transit}{col}_{row}) : (memref<{FRAGMENT}xi8>) -> ()",
                f"{indent}%{tag}source{source} = arith.constant {source} : i32",
                f"{indent}func.call @r25_insert_fragment(%{transit}{col}_{row}, %ffn{col}_{row}, %{tag}source{source}) : (memref<{FRAGMENT}xi8>, memref<{FFN}xi8>, i32) -> ()",
            ]
            if forward:
                lines.append(
                    f"{indent}func.call @r25_send_fragment(%{transit}{col}_{row}) : (memref<{FRAGMENT}xi8>) -> ()"
                )
    return lines
if PROBE:
    out.append(
        f'    func.func private @r25_probe_activation_rows(memref<{FFN}xi8>, memref<{CB}xi32>, i32) attributes {{link_with = "r25.o"}}'
    )
    out.append(
        f'    func.func private @r25_probe_gate_row(memref<{CB}xi32>, memref<{CB}xi32>) attributes {{link_with = "r25.o"}}'
    )
    out.append(
        f'    func.func private @r25_probe_gate_inputs(memref<{AB}xi8>, memref<{WB}xi8>, memref<{CB}xi32>, i32) attributes {{link_with = "r25.o"}}'
    )
    out.append(
        f'    func.func private @r25_snapshot_raw_gate(memref<{CB}xi32>, memref<{SCRATCH}xf32>) attributes {{link_with = "r25.o"}}'
    )
    out.append(
        f'    func.func private @r25_emit_raw_gate(memref<{SCRATCH}xf32>, memref<{CB}xi32>) attributes {{link_with = "r25.o"}}'
    )
    out.append(
        f'    func.func private @r25_probe_raw_samples(memref<{CB}xi32>, memref<{CB}xi32>) attributes {{link_with = "r25.o"}}'
    )


def acquire_weight(lines, indent, name):
    lines += [
        f"{indent}%{name} = aie.objectfifo.acquire @wbc{{col}}(Consume, 1) : !aie.objectfifosubview<memref<{WB}xi8>>",
        f"{indent}%{name}v = aie.objectfifo.subview.access %{name}[0] : !aie.objectfifosubview<memref<{WB}xi8>> -> memref<{WB}xi8>",
    ]


for col in range(COLS):
    for row in range(CORE_ROWS):
        gate_pack_input = f"prep{col}_{row}" if SEPARATE_GATE_A else f"ffn{col}_{row}"
        gate_pack_type = f"memref<{PREP}xf32>" if SEPARATE_GATE_A else f"memref<{FFN}xi8>"
        gate0_activation = f"ffn{col}_{row}" if CANONICAL_GATE or LOCAL_A_COPY else "a0v"
        gateg_activation = f"ffn{col}_{row}" if CANONICAL_GATE or LOCAL_A_COPY else "agv"
        canonical_group_source = "a0v" if DIRECT_X_ROW_STATE else "agv"
        gate0_canonical_source = f"directx{col}_{row}v" if DIRECT_X_FULL_OBJECT else "a0v"
        gateg_canonical_source = f"directx{col}_{row}v" if DIRECT_X_FULL_OBJECT else "agv"
        lines = [
            f"    %core{col}_{row} = aie.core(%c{col}_{row}) {{",
            "      %z = arith.constant 0 : index",
            f"      %inf = arith.constant {INF} : index",
            "      %one = arith.constant 1 : index",
            f"      %mm = arith.constant {M_MACROS} : index",
            f"      %nm = arith.constant {N_MACROS} : index",
            f"      %gg = arith.constant {GATE_GROUPS} : index",
            f"      %owner = arith.constant {col} : i32",
            f"      %destination = arith.constant {col} : i32",
            "      scf.for %outer = %z to %inf step %one {",
            "        scf.for %mblock = %z to %mm step %one {",
            f"          %c = aie.objectfifo.acquire @cc{col}_{row}(Produce, 1) : !aie.objectfifosubview<memref<{CB}xi32>>",
            f"          %cv = aie.objectfifo.subview.access %c[0] : !aie.objectfifosubview<memref<{CB}xi32>> -> memref<{CB}xi32>",
            *(
                [
                    f"          %directx{col}_{row} = aie.objectfifo.acquire @xc{col}_{row}(Consume, 1) : !aie.objectfifosubview<memref<{CANONICAL_CORE}xi8>>",
                    f"          %directx{col}_{row}v = aie.objectfifo.subview.access %directx{col}_{row}[0] : !aie.objectfifosubview<memref<{CANONICAL_CORE}xi8>> -> memref<{CANONICAL_CORE}xi8>",
                    f"          func.call @r104_rms_accumulate3(%directx{col}_{row}v, %preinv{col}_{row}) : (memref<{CANONICAL_CORE}xi8>, memref<4xf32>) -> ()",
                ]
                if DIRECT_X_FULL_OBJECT
                else []
            ),
            *(
                [
                    "          scf.for %pregrp = %z to %gg step %one {",
                    f"            %prea = aie.objectfifo.acquire @xc{col}_{row}(Consume, 1) : !aie.objectfifosubview<memref<{CANONICAL_CORE}xi8>>",
                    f"            %preav = aie.objectfifo.subview.access %prea[0] : !aie.objectfifosubview<memref<{CANONICAL_CORE}xi8>> -> memref<{CANONICAL_CORE}xi8>",
                    "            %pregrpi = arith.index_cast %pregrp : index to i32",
                    f"            func.call @r104_rms_accumulate3(%preav, %preinv{col}_{row}, %pregrpi) : (memref<{CANONICAL_CORE}xi8>, memref<4xf32>, i32) -> ()",
                    f"            aie.objectfifo.release @xc{col}_{row}(Consume, 1)",
                    "          }",
                ]
                if DIRECT_X_NORMALIZE and not DIRECT_X_FULL_OBJECT
                else []
            ),
            "          scf.for %nblock = %z to %nm step %one {",
            f"            %gw0 = aie.objectfifo.acquire @wbc{col}(Consume, 1) : !aie.objectfifosubview<memref<{WB}xi8>>",
            f"            %gw0v = aie.objectfifo.subview.access %gw0[0] : !aie.objectfifosubview<memref<{WB}xi8>> -> memref<{WB}xi8>",
            *(
                [
                    *(
                        []
                        if DIRECT_X_FULL_OBJECT
                        else [
                            f"            %a0 = aie.objectfifo.acquire @xc{col}_{row}(Consume, 1) : !aie.objectfifosubview<memref<{CANONICAL_CORE}xi8>>",
                            f"            %a0v = aie.objectfifo.subview.access %a0[0] : !aie.objectfifosubview<memref<{CANONICAL_CORE}xi8>> -> memref<{CANONICAL_CORE}xi8>",
                        ]
                    ),
                    *( [f"            %gate0sourceslot = arith.constant 0 : i32", f"            func.call @r97_probe_source(%{gate0_canonical_source}, %cv, %gate0sourceslot) : (memref<{CANONICAL_CORE}xi8>, memref<{CB}xi32>, i32) -> ()"] if PROBE_INPUTS else [] ),
                    (
                        f"            %gate0group = arith.constant 0 : i32\n            func.call @r104_bf16_x_inverse_to_f32_3(%{gate0_canonical_source}, %preinv{col}_{row}, %{gate_pack_input}, %gate0group) : (memref<{CANONICAL_CORE}xi8>, memref<4xf32>, {gate_pack_type}, i32) -> ()"
                        if DIRECT_X_FULL_OBJECT
                        else f"            func.call @r104_bf16_x_inverse_to_f32_3(%a0v, %preinv{col}_{row}, %{gate_pack_input}) : (memref<{CANONICAL_CORE}xi8>, memref<4xf32>, {gate_pack_type}) -> ()"
                        if DIRECT_X_NORMALIZE
                        else f"            %gate0group = arith.constant 0 : i32\n            func.call @r101_bf16_x_inverse_to_f32_3(%a0v, %{gate_pack_input}, %gate0group) : (memref<{CANONICAL_CORE}xi8>, {gate_pack_type}, i32) -> ()"
                        if DIRECT_X_ROW_STATE
                        else f"            func.call @r97_bf16_to_f32_3(%a0v, %{gate_pack_input}) : (memref<{CANONICAL_CORE}xi8>, {gate_pack_type}) -> ()"
                    ),
                    *(
                        []
                        if DIRECT_X_ROW_STATE or DIRECT_X_FULL_OBJECT
                        else [f"            aie.objectfifo.release @xc{col}_{row}(Consume, 1)"]
                    ),
                    f"            %gate0kind = arith.constant 2 : i32",
                    f"            func.call @r25_pack3(%{gate_pack_input}, %carry{col}_{row}, %gw0v, %scratch{col}_{row}, %packscales{col}_{row}, %gateown{col}_{row}, %gate0kind) : ({gate_pack_type}, memref<{CARRY}xf32>, memref<{WB}xi8>, memref<{SCRATCH}xf32>, memref<{PACK_SCALES}xf32>, memref<{FRAGMENT}xi8>, i32) -> ()",
                    *gate_fragment_exchange("            ", col, row, "gate0", "gateown", "gatetransit"),
                    *( [f"            func.call @r97_activation_fence(%ffn{col}_{row}) : (memref<{FFN}xi8>) -> ()"] if ACTIVATION_FENCE else [] ),
                    *( [f"            func.call @r97_scan_activation(%ffn{col}_{row}, %scratch{col}_{row}) : (memref<{FFN}xi8>, memref<{SCRATCH}xf32>) -> ()"] if ACTIVATION_SCAN else [] ),
                    *( [f"            %gate0hashslot = arith.constant 0 : i32", f"            func.call @r97_snapshot_activation_hash(%ffn{col}_{row}, %saved{col}_{row}, %gate0hashslot) : (memref<{FFN}xi8>, memref<{SAVED}xf32>, i32) -> ()"] if GATE_ACTIVATION_HASH else [] ),
                    *( [f"            %gate0weighthashslot = arith.constant 3 : i32", f"            func.call @r97_snapshot_weight_hash(%gw0v, %saved{col}_{row}, %gate0weighthashslot) : (memref<{WB}xi8>, memref<{SAVED}xf32>, i32) -> ()"] if GATE_ACTIVATION_HASH else [] ),
                    *( [f"            func.call @r25_probe_activation_rows(%ffn{col}_{row}, %cv, %owner) : (memref<{FFN}xi8>, memref<{CB}xi32>, i32) -> ()"] if PROBE_INPUTS and PROBE_GROUP == 0 else [] ),
                ]
                if CANONICAL_GATE
                else [
                    f"            %a0 = aie.objectfifo.acquire @abc{row}(Consume, 1) : !aie.objectfifosubview<memref<{AB}xi8>>",
                    f"            %a0v = aie.objectfifo.subview.access %a0[0] : !aie.objectfifosubview<memref<{AB}xi8>> -> memref<{AB}xi8>",
                ]
            ),
            *( [f"            func.call @r25_wait_weight(%gw0v, %scratch{col}_{row}) : (memref<{WB}xi8>, memref<{SCRATCH}xf32>) -> ()"] if WEIGHT_INITIAL_SCAN else [] ),
            *( [f"            func.call @r25_wait_weight(%gw0v, %scratch{col}_{row}) : (memref<{WB}xi8>, memref<{SCRATCH}xf32>) -> ()"] if WEIGHT_SCAN else [] ),
            *( [f"            func.call @r97_copy_activation(%a0v, %ffn{col}_{row}) : (memref<{AB}xi8>, memref<{FFN}xi8>) -> ()"] if LOCAL_A_COPY and not CANONICAL_GATE else [] ),
            *( [f"            %precall0kind = arith.constant 2 : i32", f"            func.call @r25_pack3(%ffn{col}_{row}, %carry{col}_{row}, %gw0v, %scratch{col}_{row}, %packscales{col}_{row}, %own{col}_{row}, %precall0kind) : (memref<{FFN}xi8>, memref<{CARRY}xf32>, memref<{WB}xi8>, memref<{SCRATCH}xf32>, memref<{PACK_SCALES}xf32>, memref<{FRAGMENT}xi8>, i32) -> ()"] if PRECALL_PACK and not CANONICAL_GATE else [] ),
            *( gate_fragment_exchange("            ", col, row, "precall0") if PRECALL_RING and not CANONICAL_GATE else [] ),
            *( [
                f"            %probe_mblock = arith.constant {PROBE_MBLOCK} : index",
                "            %probe_m1 = arith.cmpi eq, %mblock, %probe_mblock : index",
                f"            %probe_nblock = arith.constant {PROBE_NBLOCK} : index",
                "            %probe_n = arith.cmpi eq, %nblock, %probe_nblock : index",
                "            %probe_input = arith.andi %probe_m1, %probe_n : i1",
                "            scf.if %probe_input {",
                "              %probe_slot0 = arith.constant 0 : i32",
                f"              func.call @r25_probe_gate_inputs(%{'ffn' + str(col) + '_' + str(row) if CANONICAL_GATE else 'a0v'}, %gw0v, %cv, %probe_slot0) : (memref<{AB}xi8>, memref<{WB}xi8>, memref<{CB}xi32>, i32) -> ()",
                "            }",
            ] if PROBE_INPUTS else [] ),
            *( [f"            func.call @r25_zero(%cv) : (memref<{CB}xi32>) -> ()"] if ZERO_ACCUM_GATE else [] ),
            *( [
                "            %raw_warmup_slot = arith.constant 7 : i32",
                f"            func.call @r25_probe_gate_inputs(%a0v, %gw0v, %cv, %raw_warmup_slot) : (memref<{AB}xi8>, memref<{WB}xi8>, memref<{CB}xi32>, i32) -> ()",
            ] if PROBE_RAW_WARMUP else [] ),
            *( [
                f"            func.call @r25_touch_weight(%gw0v, %scratch{col}_{row}) : (memref<{WB}xi8>, memref<{SCRATCH}xf32>) -> ()",
            ] if PROBE_RAW_W_WARMUP else [] ),
            *( [
                f"            func.call @r25_touch_weight_tail(%gw0v, %scratch{col}_{row}) : (memref<{WB}xi8>, memref<{SCRATCH}xf32>) -> ()",
            ] if PROBE_RAW_W_TAIL else [] ),
            *( [] if PROBE_INPUTS else gate_gemm("            ", gate0_activation, "gw0v", "cv", ZERO_ACCUM_GATE, "gate0") ),
            *( [
                f"            func.call @r15_w4_scaled_init(%a0v, %gw0v, %cv) : (memref<{AB}xi8>, memref<{WB}xi8>, memref<{CB}xi32>) -> ()",
            ] if PROBE_RAW_DOUBLE else [] ),
            *( [
                f"            %raw_nblock = arith.constant {PROBE_RAW_NBLOCK} : index",
                "            %raw_is_nblock = arith.cmpi eq, %nblock, %raw_nblock : index",
                "            scf.if %raw_is_nblock {",
                (f"              func.call @r15_w4_scaled_init(%a0v, %gw0v, %cv) : (memref<{AB}xi8>, memref<{WB}xi8>, memref<{CB}xi32>) -> ()"
                 if PROBE_RAW_DIRECT else
                 f"              func.call @r25_probe_raw_samples(%cv, %cv) : (memref<{CB}xi32>, memref<{CB}xi32>) -> ()"),
                "            }",
            ] if PROBE_RAW and PROBE_RAW_GROUP == 0 else [] ),
            *([] if CANONICAL_GATE else [f"            aie.objectfifo.release @abc{row}(Consume, 1)"]),
            f"            aie.objectfifo.release @wbc{col}(Consume, 1)",
            "            scf.for %g = %one to %gg step %one {",
            f"              %gw = aie.objectfifo.acquire @wbc{col}(Consume, 1) : !aie.objectfifosubview<memref<{WB}xi8>>",
            f"              %gwv = aie.objectfifo.subview.access %gw[0] : !aie.objectfifosubview<memref<{WB}xi8>> -> memref<{WB}xi8>",
            *(
                [
                    *(
                        []
                        if DIRECT_X_ROW_STATE or DIRECT_X_FULL_OBJECT
                        else [
                            f"              %ag = aie.objectfifo.acquire @xc{col}_{row}(Consume, 1) : !aie.objectfifosubview<memref<{CANONICAL_CORE}xi8>>",
                            f"              %agv = aie.objectfifo.subview.access %ag[0] : !aie.objectfifosubview<memref<{CANONICAL_CORE}xi8>> -> memref<{CANONICAL_CORE}xi8>",
                        ]
                    ),
                    *( [f"              %gatesourceslot = arith.index_cast %g : index to i32", f"              func.call @r97_probe_source(%{gateg_canonical_source}, %cv, %gatesourceslot) : (memref<{CANONICAL_CORE}xi8>, memref<{CB}xi32>, i32) -> ()"] if PROBE_INPUTS else [] ),
                    (
                        f"              %directxgroup = arith.index_cast %g : index to i32\n              func.call @r104_bf16_x_inverse_to_f32_3(%{gateg_canonical_source}, %preinv{col}_{row}, %{gate_pack_input}, %directxgroup) : (memref<{CANONICAL_CORE}xi8>, memref<4xf32>, {gate_pack_type}, i32) -> ()"
                        if DIRECT_X_FULL_OBJECT
                        else f"              func.call @r104_bf16_x_inverse_to_f32_3(%agv, %preinv{col}_{row}, %{gate_pack_input}) : (memref<{CANONICAL_CORE}xi8>, memref<4xf32>, {gate_pack_type}) -> ()"
                        if DIRECT_X_NORMALIZE
                        else f"              %rowstategroup = arith.index_cast %g : index to i32\n              func.call @r101_bf16_x_inverse_to_f32_3(%{canonical_group_source}, %{gate_pack_input}, %rowstategroup) : (memref<{CANONICAL_CORE}xi8>, {gate_pack_type}, i32) -> ()"
                        if DIRECT_X_ROW_STATE
                        else f"              func.call @r97_bf16_to_f32_3(%agv, %{gate_pack_input}) : (memref<{CANONICAL_CORE}xi8>, {gate_pack_type}) -> ()"
                    ),
                    *(
                        []
                        if DIRECT_X_ROW_STATE or DIRECT_X_FULL_OBJECT
                        else [f"              aie.objectfifo.release @xc{col}_{row}(Consume, 1)"]
                    ),
                    f"              %gategkind = arith.constant 2 : i32",
                    f"              func.call @r25_pack3(%{gate_pack_input}, %carry{col}_{row}, %gwv, %scratch{col}_{row}, %packscales{col}_{row}, %gateown{col}_{row}, %gategkind) : ({gate_pack_type}, memref<{CARRY}xf32>, memref<{WB}xi8>, memref<{SCRATCH}xf32>, memref<{PACK_SCALES}xf32>, memref<{FRAGMENT}xi8>, i32) -> ()",
                    *gate_fragment_exchange("              ", col, row, "gateg", "gateown", "gatetransit"),
                    *( [f"              func.call @r97_activation_fence(%ffn{col}_{row}) : (memref<{FFN}xi8>) -> ()"] if ACTIVATION_FENCE else [] ),
                    *( [f"              func.call @r97_scan_activation(%ffn{col}_{row}, %scratch{col}_{row}) : (memref<{FFN}xi8>, memref<{SCRATCH}xf32>) -> ()"] if ACTIVATION_SCAN else [] ),
                    *( [f"              %gateghashslot = arith.index_cast %g : index to i32", f"              func.call @r97_snapshot_activation_hash(%ffn{col}_{row}, %saved{col}_{row}, %gateghashslot) : (memref<{FFN}xi8>, memref<{SAVED}xf32>, i32) -> ()"] if GATE_ACTIVATION_HASH else [] ),
                    *( [f"              %gategweighthashslot0 = arith.index_cast %g : index to i32", f"              %gategweighthashbase = arith.constant 3 : i32", f"              %gategweighthashslot = arith.addi %gategweighthashslot0, %gategweighthashbase : i32", f"              func.call @r97_snapshot_weight_hash(%gwv, %saved{col}_{row}, %gategweighthashslot) : (memref<{WB}xi8>, memref<{SAVED}xf32>, i32) -> ()"] if GATE_ACTIVATION_HASH else [] ),
                    *( [f"              %gateprobegroup = arith.constant {PROBE_GROUP} : index", "              %gateprobeactive = arith.cmpi eq, %g, %gateprobegroup : index", "              scf.if %gateprobeactive {", f"                func.call @r25_probe_activation_rows(%ffn{col}_{row}, %cv, %owner) : (memref<{FFN}xi8>, memref<{CB}xi32>, i32) -> ()", "              }"] if PROBE_INPUTS and PROBE_GROUP > 0 else [] ),
                ]
                if CANONICAL_GATE
                else [
                    f"              %ag = aie.objectfifo.acquire @abc{row}(Consume, 1) : !aie.objectfifosubview<memref<{AB}xi8>>",
                    f"              %agv = aie.objectfifo.subview.access %ag[0] : !aie.objectfifosubview<memref<{AB}xi8>> -> memref<{AB}xi8>",
                ]
            ),
            *( [f"              func.call @r25_wait_weight(%gwv, %scratch{col}_{row}) : (memref<{WB}xi8>, memref<{SCRATCH}xf32>) -> ()"] if WEIGHT_SCAN else [] ),
            *( [f"              func.call @r97_copy_activation(%agv, %ffn{col}_{row}) : (memref<{AB}xi8>, memref<{FFN}xi8>) -> ()"] if LOCAL_A_COPY and not CANONICAL_GATE else [] ),
            *( [f"              %precallgkind = arith.constant 2 : i32", f"              func.call @r25_pack3(%ffn{col}_{row}, %carry{col}_{row}, %gwv, %scratch{col}_{row}, %packscales{col}_{row}, %own{col}_{row}, %precallgkind) : (memref<{FFN}xi8>, memref<{CARRY}xf32>, memref<{WB}xi8>, memref<{SCRATCH}xf32>, memref<{PACK_SCALES}xf32>, memref<{FRAGMENT}xi8>, i32) -> ()"] if PRECALL_PACK and not CANONICAL_GATE else [] ),
            *( gate_fragment_exchange("              ", col, row, "precallg") if PRECALL_RING and not CANONICAL_GATE else [] ),
            *( [
                "              %probe_slot = arith.index_cast %g : index to i32",
                "              scf.if %probe_input {",
                f"                func.call @r25_probe_gate_inputs(%{'ffn' + str(col) + '_' + str(row) if CANONICAL_GATE else 'agv'}, %gwv, %cv, %probe_slot) : (memref<{AB}xi8>, memref<{WB}xi8>, memref<{CB}xi32>, i32) -> ()",
                "              }",
            ] if PROBE_INPUTS else [] ),
            *( [] if PROBE_INPUTS else gate_gemm("              ", gateg_activation, "gwv", "cv", True, "gateg") ),
            *( [
                f"              %raw_group = arith.constant {PROBE_RAW_GROUP} : index",
                "              %raw_is_group = arith.cmpi eq, %g, %raw_group : index",
                f"              %raw_nblock = arith.constant {PROBE_RAW_NBLOCK} : index",
                "              %raw_is_nblock = arith.cmpi eq, %nblock, %raw_nblock : index",
                "              %raw_take = arith.andi %raw_is_group, %raw_is_nblock : i1",
                "              scf.if %raw_take {",
                f"                func.call @r25_probe_raw_samples(%cv, %cv) : (memref<{CB}xi32>, memref<{CB}xi32>) -> ()",
                "              }",
            ] if PROBE_RAW and PROBE_RAW_GROUP > 0 else [] ),
            *([] if CANONICAL_GATE else [f"              aie.objectfifo.release @abc{row}(Consume, 1)"]),
            f"              aie.objectfifo.release @wbc{col}(Consume, 1)",
            "            }",
            *(
                [f"            aie.objectfifo.release @xc{col}_{row}(Consume, 1)"]
                if DIRECT_X_ROW_STATE
                else []
            ),
            *( [] if PROBE_RAW or PROBE_INPUTS or PROBE_GATE_RAW_ONLY else [
                f"            func.call @r25_geglu_inplace(%cv, %scratch{col}_{row}) : (memref<{CB}xi32>, memref<{SCRATCH}xf32>) -> ()",
            ] ),
        ]
        if PROBE_GATE and not PROBE_GATE_RAW_ONLY:
            lines += [
                "            %probe_is0 = arith.cmpi eq, %nblock, %z : index",
                "            scf.if %probe_is0 {",
                f"              func.call @r25_probe_gate_row(%cv, %cv) : (memref<{CB}xi32>, memref<{CB}xi32>) -> ()",
                "            }",
            ]
        if GATE_ACTIVATION_HASH:
            lines.append(
                f"            func.call @r97_emit_activation_hashes(%saved{col}_{row}, %cv) : (memref<{SAVED}xf32>, memref<{CB}xi32>) -> ()"
            )
        skip_gate_ring = (PROBE_GATE and not PROBE_FULL) or PROBE_INPUTS or PROBE_RAW
        for source in ([] if skip_gate_ring else range(COLS)):
            if col == source:
                lines += [
                    f"            %source{source} = arith.constant {source} : i32",
                    f"            func.call @r25_extract_local(%cv, %ffn{col}_{row}, %source{source}, %destination) : (memref<{CB}xi32>, memref<{FFN}xi8>, i32, i32) -> ()",
                    f"            %tilewords{source} = arith.constant {TILE} : i32",
                    f"            func.call @r25_send_words(%cv, %tilewords{source}) : (memref<{CB}xi32>, i32) -> ()",
                ]
            else:
                forward = 0 if col == (source - 1) % COLS else 1
                lines += [
                    f"            %source{source} = arith.constant {source} : i32",
                    f"            %forward{source} = arith.constant {forward} : i32",
                    f"            func.call @r25_receive_tile(%ffn{col}_{row}, %source{source}, %destination, %forward{source}) : (memref<{FFN}xi8>, i32, i32, i32) -> ()",
                ]
        if not PROBE:
            lines += [
                "            %has_down_partial = arith.cmpi ne, %nblock, %z : index",
                "            scf.if %has_down_partial {",
                f"              func.call @r25_restore_down_bf16(%saved{col}_{row}, %own{col}_{row}, %transit{col}_{row}, %cv) : (memref<{SAVED}xf32>, memref<{FRAGMENT}xi8>, memref<{FRAGMENT}xi8>, memref<{CB}xi32>) -> ()",
                "            }",
            ]
        if PROBE_STREAM_ONLY:
            lines += [
                "            }",
                f"          aie.objectfifo.release @cc{col}_{row}(Produce, 1)",
                "        }",
                "      }",
                "      aie.end",
                "    } {stack_size = 4096 : i32}",
            ]
            out += lines
            continue
        # nblock-specific down groups are expanded with scf.if branches.
        lines += ["            %is0 = arith.cmpi eq, %nblock, %z : index"]
        if COMPACT_DOWN_BRANCHES:
            lines.append("            %is1 = arith.cmpi eq, %nblock, %one : index")
        if not PROBE_RAW:
            lines += [
                "            scf.if %is0 {",
                f"              %carryoff = arith.constant 256 : i32",
                f"              %carrycount = arith.constant 128 : i32",
                f"              func.call @r25_save_carry_generic(%ffn{col}_{row}, %carry{col}_{row}, %carryoff, %carrycount) : (memref<{FFN}xi8>, memref<{CARRY}xf32>, i32, i32) -> ()",
                "            } else {",
                *( [] if COMPACT_DOWN_BRANCHES else ["              %is1 = arith.cmpi eq, %nblock, %one : index"] ),
                "              scf.if %is1 {",
                f"                %off = arith.constant 128 : i32",
                f"                %count = arith.constant 256 : i32",
                f"                func.call @r25_save_tile(%ffn{col}_{row}, %saved{col}_{row}, %off, %count) : (memref<{FFN}xi8>, memref<{SAVED}xf32>, i32, i32) -> ()",
                "              } else {",
                f"                %off = arith.constant 256 : i32",
                f"                %count = arith.constant 128 : i32",
                f"                func.call @r25_save_tile(%ffn{col}_{row}, %saved{col}_{row}, %off, %count) : (memref<{FFN}xi8>, memref<{SAVED}xf32>, i32, i32) -> ()",
                "              }",
                "            }",
            ]
        if COMPACT_DOWN_BRANCHES and not PROBE:
            lines += [
                "            %downkind0 = arith.constant 0 : i32",
                "            %downkind1 = arith.constant 1 : i32",
                "            %downfirstkind = arith.select %is1, %downkind1, %downkind0 : i32",
                "            %downacc0 = arith.constant 0 : i32",
                "            %downacc1 = arith.constant 1 : i32",
                "            %downfirstacc = arith.select %is0, %downacc0, %downacc1 : i32",
                f"            %downfirstw = aie.objectfifo.acquire @wbc{col}(Consume, 1) : !aie.objectfifosubview<memref<{WB}xi8>>",
                f"            %downfirstwv = aie.objectfifo.subview.access %downfirstw[0] : !aie.objectfifosubview<memref<{WB}xi8>> -> memref<{WB}xi8>",
                f"            func.call @r25_pack3(%ffn{col}_{row}, %carry{col}_{row}, %downfirstwv, %scratch{col}_{row}, %packscales{col}_{row}, %own{col}_{row}, %downfirstkind) : (memref<{FFN}xi8>, memref<{CARRY}xf32>, memref<{WB}xi8>, memref<{SCRATCH}xf32>, memref<{PACK_SCALES}xf32>, memref<{FRAGMENT}xi8>, i32) -> ()",
                f"            func.call @r25_exchange_fragments(%ffn{col}_{row}, %own{col}_{row}, %transit{col}_{row}, %owner) : (memref<{FFN}xi8>, memref<{FRAGMENT}xi8>, memref<{FRAGMENT}xi8>, i32) -> ()",
                f"            func.call @r25_down(%ffn{col}_{row}, %downfirstwv, %cv, %downfirstacc) : (memref<{FFN}xi8>, memref<{WB}xi8>, memref<{CB}xi32>, i32) -> ()",
                f"            aie.objectfifo.release @wbc{col}(Consume, 1)",
                "            %downhassecond = arith.cmpi ne, %nblock, %z : index",
                "            scf.if %downhassecond {",
                "              %downcount1 = arith.constant 256 : i32",
                "              %downcount2 = arith.constant 128 : i32",
                "              %downrestorecount = arith.select %is1, %downcount1, %downcount2 : i32",
                f"              func.call @r25_restore_tile(%saved{col}_{row}, %ffn{col}_{row}, %downrestorecount) : (memref<{SAVED}xf32>, memref<{FFN}xi8>, i32) -> ()",
                "              %downkind2 = arith.constant 2 : i32",
                "              %downkind4 = arith.constant 4 : i32",
                "              %downsecondkind = arith.select %is1, %downkind2, %downkind4 : i32",
                f"              %downsecondw = aie.objectfifo.acquire @wbc{col}(Consume, 1) : !aie.objectfifosubview<memref<{WB}xi8>>",
                f"              %downsecondwv = aie.objectfifo.subview.access %downsecondw[0] : !aie.objectfifosubview<memref<{WB}xi8>> -> memref<{WB}xi8>",
                f"              func.call @r25_pack3(%ffn{col}_{row}, %carry{col}_{row}, %downsecondwv, %scratch{col}_{row}, %packscales{col}_{row}, %own{col}_{row}, %downsecondkind) : (memref<{FFN}xi8>, memref<{CARRY}xf32>, memref<{WB}xi8>, memref<{SCRATCH}xf32>, memref<{PACK_SCALES}xf32>, memref<{FRAGMENT}xi8>, i32) -> ()",
                f"              func.call @r25_exchange_fragments(%ffn{col}_{row}, %own{col}_{row}, %transit{col}_{row}, %owner) : (memref<{FFN}xi8>, memref<{FRAGMENT}xi8>, memref<{FRAGMENT}xi8>, i32) -> ()",
                f"              func.call @r25_down(%ffn{col}_{row}, %downsecondwv, %cv, %downacc1) : (memref<{FFN}xi8>, memref<{WB}xi8>, memref<{CB}xi32>, i32) -> ()",
                f"              aie.objectfifo.release @wbc{col}(Consume, 1)",
                "            }",
            ]
        # Emit one or two down groups based on nblock. The generator uses branches
        # with fixed calls so every path consumes the matching weight count.
        for branch, kinds in ([] if COMPACT_DOWN_BRANCHES and not PROBE else [(0, [(0, "ffn")]), (1, [(1, "ffn"), (2, "tile")]), (2, [(0, "ffn"), (4, "tile")])]):
            prefix = "            " if branch == 0 else ("              " if branch == 1 else "                ")
            if branch == 0:
                lines.append("            scf.if %is0 {")
            elif branch == 1:
                lines += ["            } else {", "              %is1b = arith.cmpi eq, %nblock, %one : index", "              scf.if %is1b {"]
            else:
                lines += ["              } else {"]
            for local_group, (kind, source_buf) in enumerate(kinds):
                tag = f"d{branch}_{local_group}"
                if PROBE_RAW or (PROBE_GATE and not PROBE_FULL):
                    lines += [
                        f"{prefix}%{tag}w = aie.objectfifo.acquire @wbc{col}(Consume, 1) : !aie.objectfifosubview<memref<{WB}xi8>>",
                        f"{prefix}aie.objectfifo.release @wbc{col}(Consume, 1)",
                    ]
                    continue
                if PROBE and not PROBE_FULL and (PROBE_GATE or PROBE_INPUTS or (branch, local_group) != PROBE_TARGET):
                    lines += [
                        f"{prefix}%{tag}w = aie.objectfifo.acquire @wbc{col}(Consume, 1) : !aie.objectfifosubview<memref<{WB}xi8>>",
                        f"{prefix}%{tag}wv = aie.objectfifo.subview.access %{tag}w[0] : !aie.objectfifosubview<memref<{WB}xi8>> -> memref<{WB}xi8>",
                        f"{prefix}func.call @r25_touch_weight(%{tag}wv, %scratch{col}_{row}) : (memref<{WB}xi8>, memref<{SCRATCH}xf32>) -> ()",
                        f"{prefix}aie.objectfifo.release @wbc{col}(Consume, 1)",
                    ]
                    continue
                if source_buf == "tile":
                    count = 256 if kind == 2 else 128
                    lines += [
                        f"{prefix}%{tag}count = arith.constant {count} : i32",
                        f"{prefix}func.call @r25_restore_tile(%saved{col}_{row}, %ffn{col}_{row}, %{tag}count) : (memref<{SAVED}xf32>, memref<{FFN}xi8>, i32) -> ()",
                    ]
                lines += [
                    f"{prefix}%{tag}w = aie.objectfifo.acquire @wbc{col}(Consume, 1) : !aie.objectfifosubview<memref<{WB}xi8>>",
                    f"{prefix}%{tag}wv = aie.objectfifo.subview.access %{tag}w[0] : !aie.objectfifosubview<memref<{WB}xi8>> -> memref<{WB}xi8>",
                    *( [f"{prefix}func.call @r25_wait_weight(%{tag}wv, %scratch{col}_{row}) : (memref<{WB}xi8>, memref<{SCRATCH}xf32>) -> ()"] if WEIGHT_SCAN else [] ),
                    f"{prefix}%{tag}kind = arith.constant {kind} : i32",
                    f"{prefix}func.call @r25_pack3(%ffn{col}_{row}, %carry{col}_{row}, %{tag}wv, %scratch{col}_{row}, %packscales{col}_{row}, %own{col}_{row}, %{tag}kind) : (memref<{FFN}xi8>, memref<{CARRY}xf32>, memref<{WB}xi8>, memref<{SCRATCH}xf32>, memref<{PACK_SCALES}xf32>, memref<{FRAGMENT}xi8>, i32) -> ()",
                ]
                if PROBE_FULL and (branch, local_group) != PROBE_TARGET:
                    lines.append(f"{prefix}aie.objectfifo.release @wbc{col}(Consume, 1)")
                    continue
                if COMPACT_FRAGMENT_RING:
                    lines.append(
                        f"{prefix}func.call @r25_exchange_fragments(%ffn{col}_{row}, %own{col}_{row}, %transit{col}_{row}, %owner) : (memref<{FFN}xi8>, memref<{FRAGMENT}xi8>, memref<{FRAGMENT}xi8>, i32) -> ()"
                    )
                else:
                    lines.append(
                        f"{prefix}func.call @r25_insert_fragment(%own{col}_{row}, %ffn{col}_{row}, %owner) : (memref<{FRAGMENT}xi8>, memref<{FFN}xi8>, i32) -> ()"
                    )
                    for broadcast_owner in range(COLS):
                        if col == broadcast_owner:
                            lines.append(f"{prefix}func.call @r25_send_fragment(%own{col}_{row}) : (memref<{FRAGMENT}xi8>) -> ()")
                        else:
                            lines += [
                                f"{prefix}func.call @r25_receive_fragment(%transit{col}_{row}) : (memref<{FRAGMENT}xi8>) -> ()",
                                f"{prefix}%bo{branch}_{local_group}_{broadcast_owner} = arith.constant {broadcast_owner} : i32",
                                f"{prefix}func.call @r25_insert_fragment(%transit{col}_{row}, %ffn{col}_{row}, %bo{branch}_{local_group}_{broadcast_owner}) : (memref<{FRAGMENT}xi8>, memref<{FFN}xi8>, i32) -> ()",
                            ]
                            if col != (broadcast_owner - 1) % COLS:
                                lines.append(f"{prefix}func.call @r25_send_fragment(%transit{col}_{row}) : (memref<{FRAGMENT}xi8>) -> ()")
                if PROBE and not (PROBE_GATE or PROBE_INPUTS or PROBE_RAW) and (branch, local_group) == PROBE_TARGET:
                    lines.append(
                        f"{prefix}func.call @r25_probe_activation_rows(%ffn{col}_{row}, %cv, %owner) : (memref<{FFN}xi8>, memref<{CB}xi32>, i32) -> ()"
                    )
                if not PROBE:
                    accumulate = 0 if (branch, local_group) == (0, 0) else 1
                    lines += [
                        f"{prefix}%{tag}accumulate = arith.constant {accumulate} : i32",
                        f"{prefix}func.call @r25_down(%ffn{col}_{row}, %{tag}wv, %cv, %{tag}accumulate) : (memref<{FFN}xi8>, memref<{WB}xi8>, memref<{CB}xi32>, i32) -> ()",
                    ]
                lines.append(f"{prefix}aie.objectfifo.release @wbc{col}(Consume, 1)")
        if not (COMPACT_DOWN_BRANCHES and not PROBE):
            lines += [
                "                }",
                "              }",
            ]
        if not PROBE:
            lines += [
                "            %last_nblock = arith.constant 2 : index",
                "            %spill_down_partial = arith.cmpi slt, %nblock, %last_nblock : index",
                "            scf.if %spill_down_partial {",
                f"              func.call @r25_spill_down_bf16(%cv, %saved{col}_{row}, %own{col}_{row}, %transit{col}_{row}) : (memref<{CB}xi32>, memref<{SAVED}xf32>, memref<{FRAGMENT}xi8>, memref<{FRAGMENT}xi8>) -> ()",
                "            }",
            ]
        lines += [
            "            }",
            *(
                [f"          aie.objectfifo.release @xc{col}_{row}(Consume, 1)"]
                if DIRECT_X_FULL_OBJECT
                else []
            ),
            *( [f"          func.call @r98_finish_interleaved_bf16x2(%cv) : (memref<{CB}xi32>) -> ()"] if INTERLEAVED_BF16X2_OUTPUT and not PROBE else [] ),
            f"          aie.objectfifo.release @cc{col}_{row}(Produce, 1)",
            "        }",
            "      }",
            "      aie.end",
            "    } {stack_size = 2048 : i32}",
        ]
        out += lines

AT, WT = A_BLOCKS * AB, W_BLOCKS * WB
INPUT_BYTES = CANONICAL_BYTES if CANONICAL_GATE else 4 * AT
out.append(f"    aie.runtime_sequence(%A: memref<{INPUT_BYTES}xi8>, %W: memref<{COLS * WT}xi8>, %C: memref<{PAD_M * OUTPUT_ROW_WORDS}xi32>) {{")
if not CANONICAL_GATE:
    for row in range(CORE_ROWS):
        out += [
            f"      %ta{row} = aiex.dma_configure_task_for @ash{row} {{",
            f"        aie.dma_bd(%A : memref<{4 * AT}xi8>, {row * AT}, {AT}, {dims(A_BLOCKS, AB)}) {{burst_length = 0 : i32}}",
            "        aie.end",
            "      } {issue_token = true}",
            f"      aiex.dma_start_task(%ta{row})",
        ]
weight_runtime_fifo = "wbc" if WEIGHT_DIRECT else "wsh"
for col in range(COLS):
    out += [
        f"      %tw{col} = aiex.dma_configure_task_for @{weight_runtime_fifo}{col} {{",
        f"        aie.dma_bd(%W : memref<{COLS * WT}xi8>, {col * WT}, {WT}, {dims(W_BLOCKS, WB)}) {{burst_length = 0 : i32}}",
        "        aie.end",
        "      } {issue_token = true}",
        f"      aiex.dma_start_task(%tw{col})",
    ]
for mblock in range(M_MACROS):
    for col in range(COLS):
        offset = mblock * 96 * OUTPUT_ROW_WORDS + col * 96
        out += [
            f"      %tc{col}_{mblock} = aiex.dma_configure_task_for @csh{col} {{",
            f"        aie.dma_bd(%C : memref<{PAD_M * OUTPUT_ROW_WORDS}xi32>, {offset}, 384, {output_dims()}) {{burst_length = 0 : i32}}",
            "        aie.end",
            f"      }} {{issue_token = true, repeat_count = 23 : i32}}",
            f"      aiex.dma_start_task(%tc{col}_{mblock})",
        ]
    if CANONICAL_GATE:
        if DIRECT_X_FULL_OBJECT:
            for col in range(COLS):
                name = f"txfull{mblock}_{col}"
                source_row = mblock * 96 + col * 3
                source_offset = source_row * CANONICAL_ROW
                out += [
                    f"      %{name} = aiex.dma_configure_task_for @xsh{col} {{",
                    f"        aie.dma_bd(%A : memref<{CANONICAL_BYTES}xi8>, {source_offset}, {CANONICAL_JOIN}, [<size = {CORE_ROWS}, stride = {24 * CANONICAL_ROW}>, <size = 3, stride = {CANONICAL_ROW}>, <size = {CANONICAL_ROW}, stride = 1>]) {{burst_length = 0 : i32}}",
                    "        aie.end",
                    "      } {issue_token = true}",
                    f"      aiex.dma_start_task(%{name})",
                ]
            for col in range(COLS):
                name = f"txfull{mblock}_{col}"
                out += [
                    f"      aiex.dma_await_task(%{name})",
                    f"      aiex.dma_free_task(%{name})",
                ]
        if DIRECT_X_NORMALIZE and not DIRECT_X_FULL_OBJECT:
            for group in range(GATE_GROUPS):
                for col in range(COLS):
                    name = f"tp{mblock}_{group}_{col}"
                    source_row = mblock * 96 + col * 3
                    source_offset = source_row * CANONICAL_ROW + group * GROUP * 2
                    out += [
                        f"      %{name} = aiex.dma_configure_task_for @xsh{col} {{",
                        f"        aie.dma_bd(%A : memref<{CANONICAL_BYTES}xi8>, {source_offset}, {CANONICAL_JOIN}, [<size = {CORE_ROWS}, stride = {24 * CANONICAL_ROW}>, <size = 3, stride = {CANONICAL_ROW}>, <size = {GROUP * 2}, stride = 1>]) {{burst_length = 0 : i32}}",
                        "        aie.end",
                        "      } {issue_token = true}",
                        f"      aiex.dma_start_task(%{name})",
                    ]
                for col in range(COLS):
                    name = f"tp{mblock}_{group}_{col}"
                    out += [
                        f"      aiex.dma_await_task(%{name})",
                        f"      aiex.dma_free_task(%{name})",
                    ]
        for nblock in ([] if DIRECT_X_FULL_OBJECT else range(N_MACROS)):
            groups = [None] if DIRECT_X_ROW_STATE else range(GATE_GROUPS)
            for group in groups:
                for col in range(COLS):
                    tag = "row" if group is None else str(group)
                    name = f"tx{mblock}_{nblock}_{tag}_{col}"
                    source_row = mblock * 96 + col * 3
                    source_offset = source_row * CANONICAL_ROW + (
                        0 if group is None else group * GROUP * 2
                    )
                    transfer_dims = (
                        f"[<size = {CORE_ROWS}, stride = {24 * CANONICAL_ROW}>, <size = 3, stride = {CANONICAL_ROW}>, <size = {CANONICAL_ROW}, stride = 1>]"
                        if group is None
                        else f"[<size = {CORE_ROWS}, stride = {24 * CANONICAL_ROW}>, <size = 3, stride = {CANONICAL_ROW}>, <size = {GROUP * 2}, stride = 1>]"
                    )
                    out += [
                        f"      %{name} = aiex.dma_configure_task_for @xsh{col} {{",
                        f"        aie.dma_bd(%A : memref<{CANONICAL_BYTES}xi8>, {source_offset}, {CANONICAL_JOIN}, {transfer_dims}) {{burst_length = 0 : i32}}",
                        "        aie.end",
                        "      } {issue_token = true}",
                        f"      aiex.dma_start_task(%{name})",
                    ]
                for col in range(COLS):
                    tag = "row" if group is None else str(group)
                    name = f"tx{mblock}_{nblock}_{tag}_{col}"
                    out += [f"      aiex.dma_await_task(%{name})", f"      aiex.dma_free_task(%{name})"]
    for col in range(COLS):
        out += [f"      aiex.dma_await_task(%tc{col}_{mblock})", f"      aiex.dma_free_task(%tc{col}_{mblock})"]
if not CANONICAL_GATE:
    for row in range(CORE_ROWS):
        out += [f"      aiex.dma_await_task(%ta{row})", f"      aiex.dma_free_task(%ta{row})"]
for col in range(COLS):
    out += [f"      aiex.dma_await_task(%tw{col})", f"      aiex.dma_free_task(%tw{col})"]
out += ["    }", "  }", "}"]
print("\n".join(out))
