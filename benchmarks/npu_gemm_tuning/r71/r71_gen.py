#!/usr/bin/env python3
"""Attach R30 BF16 attention to R70 in the same AIE2P graph/context."""

from pathlib import Path
import re
import subprocess
import sys


HERE = Path(__file__).resolve().parent
R70 = HERE.parent / "r70" / "r70_gen.py"
R81 = HERE.parent / "r81" / "r81_gen.py"

STAGE_BYTES = 2_457_600
ATTENTION_BYTES = 393_216
Q_BYTES = 393_216
KV_BYTES = 262_144
QKV_WEIGHT_BYTES = 2_359_296
O_WEIGHT_BYTES = 4 * 72 * 16_384
Q_JOIN = 16_384
KV_TILE = 16_384
OUT_TILE = 2_048
OUT_JOIN = 8_192
QUERY_GROUPS = 6
DIRECT_Q = "--direct-q" in sys.argv[1:]
LOCAL_Q = "--local-q" in sys.argv[1:]
ADJACENT_Q = "--adjacent-q" in sys.argv[1:]
ODD_ATTENTION = "--odd-attention" in sys.argv[1:]
PAIRED_ATTENTION = "--paired-attention" in sys.argv[1:]
COMPACT_PAIRED_ATTENTION = "--compact-paired-attention" in sys.argv[1:]
DIRECT_OUTPUT_FFN_HANDOFF = "--direct-output-ffn-handoff" in sys.argv[1:]
DIRECT_OUTPUT_RESIDUAL_NORM = (
    "--direct-output-residual-norm" in sys.argv[1:] or DIRECT_OUTPUT_FFN_HANDOFF
)
DIRECT_OUTPUT_BF16_STAGE = (
    "--direct-output-bf16-stage" in sys.argv[1:] or DIRECT_OUTPUT_RESIDUAL_NORM
)
DIRECT_OUTPUT = (
    "--direct-output-projection" in sys.argv[1:] or DIRECT_OUTPUT_BF16_STAGE
)
DIRECT_ATTENTION_DRAIN = "--direct-attention-drain" in sys.argv[1:]
DIRECT_OUTPUT_WEIGHT_DRAIN = "--direct-output-weight-drain" in sys.argv[1:]
DIRECT_OUTPUT_LOCAL_FINISH = "--direct-output-local-finish" in sys.argv[1:]
OUTPUT_WEIGHT_SHIM_DEPTH2 = "--output-weight-shim-depth2" in sys.argv[1:]
OUTPUT_WEIGHT_SHIM_DEPTH3 = "--output-weight-shim-depth3" in sys.argv[1:]
if OUTPUT_WEIGHT_SHIM_DEPTH2 and OUTPUT_WEIGHT_SHIM_DEPTH3:
    raise SystemExit("output-weight shim depths are mutually exclusive")
OUTPUT_WEIGHT_SHIM_DEPTH = (
    3
    if OUTPUT_WEIGHT_SHIM_DEPTH3
    else 2
    if OUTPUT_WEIGHT_SHIM_DEPTH2 or DIRECT_OUTPUT_RESIDUAL_NORM
    else 1
)
O_BYTES = 256 * 768 * (2 if DIRECT_OUTPUT_BF16_STAGE else 4)
FFN_HANDOFF_BYTES = 256 * 768 * 2
RN_BLOCKS_PER_COL = 8
RN_WEIGHT_BYTES = 4 * RN_BLOCKS_PER_COL * 16_384
DIRECT_ATTENTION_HANDOFF = (
    DIRECT_OUTPUT
    or DIRECT_ATTENTION_DRAIN
    or DIRECT_OUTPUT_WEIGHT_DRAIN
    or DIRECT_OUTPUT_LOCAL_FINISH
)
DIRECT_OUTPUT_COMPUTE = DIRECT_OUTPUT or DIRECT_OUTPUT_LOCAL_FINISH
DIRECT_OUTPUT_WEIGHT_STREAM = DIRECT_OUTPUT_COMPUTE or DIRECT_OUTPUT_WEIGHT_DRAIN
BATCH2 = "--batch2" in sys.argv[1:]
ENQUEUE_WINDOW_FLAGS = [
    size for size in (2, 3) if f"--enqueue-window{size}" in sys.argv[1:]
]
if len(ENQUEUE_WINDOW_FLAGS) > 1:
    raise SystemExit("enqueue-window modes are mutually exclusive")
ENQUEUE_WINDOW = ENQUEUE_WINDOW_FLAGS[0] if ENQUEUE_WINDOW_FLAGS else 0
if COMPACT_PAIRED_ATTENTION:
    PAIRED_ATTENTION = True
if DIRECT_ATTENTION_HANDOFF:
    COMPACT_PAIRED_ATTENTION = True
    PAIRED_ATTENTION = True
if (
    sum(
        (
            DIRECT_OUTPUT,
            DIRECT_ATTENTION_DRAIN,
            DIRECT_OUTPUT_WEIGHT_DRAIN,
            DIRECT_OUTPUT_LOCAL_FINISH,
        )
    )
    > 1
):
    raise SystemExit("direct-output attribution modes are mutually exclusive")
if sum((DIRECT_Q, LOCAL_Q, ADJACENT_Q, PAIRED_ATTENTION)) > 1:
    raise SystemExit("direct-Q modes are mutually exclusive")
CACHED_Q = DIRECT_Q or LOCAL_Q or ADJACENT_Q
if BATCH2 and CACHED_Q:
    raise SystemExit("--batch2 currently requires the observable R71 Q path")
if ENQUEUE_WINDOW and (BATCH2 or CACHED_Q):
    raise SystemExit("enqueue-window modes currently require the base R71 schedule")
if DIRECT_ATTENTION_HANDOFF and ENQUEUE_WINDOW != 3:
    raise SystemExit("direct attention handoff requires the admitted three-group window")
if OUTPUT_WEIGHT_SHIM_DEPTH > 1 and not DIRECT_OUTPUT_WEIGHT_STREAM:
    raise SystemExit("output-weight shim depth requires an O-weight stream mode")
RESULT_BYTES = STAGE_BYTES + ATTENTION_BYTES + (O_BYTES if DIRECT_OUTPUT else 0)
WEIGHT_BYTES = QKV_WEIGHT_BYTES + (
    O_WEIGHT_BYTES if DIRECT_OUTPUT_WEIGHT_STREAM else 0
) + (RN_WEIGHT_BYTES if DIRECT_OUTPUT_RESIDUAL_NORM else 0)
ROWS = 4
ACTIVE_COLS = (
    range(1, 8, 2)
    if ADJACENT_Q or ODD_ATTENTION or PAIRED_ATTENTION
    else (range(0, 4) if LOCAL_Q else range(4, 8))
)


def generate(path, *args):
    return subprocess.run(
        [sys.executable, str(path), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def top_level_ops(text):
    lines = text.splitlines()
    if lines[:2] != ["module {", "  aie.device(npu2) {"] or lines[-2:] != ["  }", "}"]:
        raise SystemExit("R70 graph wrapper changed")
    body = lines[2:-2]
    ops = []
    index = 0
    while index < len(body):
        if not re.match(r"^    \S", body[index]):
            raise SystemExit(f"unexpected top-level continuation: {body[index]}")
        op = [body[index]]
        balance = body[index].count("{") - body[index].count("}")
        index += 1
        while balance > 0:
            op.append(body[index])
            balance += body[index].count("{") - body[index].count("}")
            index += 1
        ops.append(op)
    return ops


def core_key(op):
    match = re.search(r"%core(\d+)_(\d+) = aie\.core", op[0])
    return tuple(map(int, match.groups())) if match else None


def is_runtime(op):
    return "aie.runtime_sequence(" in op[0]


def outer_parts(op):
    outer = next(i for i, line in enumerate(op) if "scf.for %outer" in line)
    end = next(i for i in range(len(op) - 1, outer, -1) if op[i] == "      }")
    return op[:outer], op[outer], op[outer + 1 : end], op[end:]


def attention_top_ops():
    lines = []
    for col in ACTIVE_COLS:
        for row in range(ROWS):
            for slot in range(4 if BATCH2 else 2):
                lines.extend(
                    [
                        f'    %attacc{slot}{col}_{row} = aie.buffer(%c{col}_{row}) {{sym_name = "attacc{slot}{col}_{row}"}} : memref<1024xf32>',
                        f'    %attstats{slot}{col}_{row} = aie.buffer(%c{col}_{row}) {{sym_name = "attstats{slot}{col}_{row}"}} : memref<8xf32>',
                    ]
                )
            if not CACHED_Q:
                for slot in range(2 if BATCH2 else 1):
                    suffix = str(slot) if BATCH2 else ""
                    lines.append(
                        f'    %attq{suffix}{col}_{row} = aie.buffer(%c{col}_{row}) {{sym_name = "attq{suffix}{col}_{row}"}} : memref<4096xi8>'
                    )
    if DIRECT_OUTPUT_COMPUTE:
        for col in range(0, 8, 2):
            for row in range(ROWS):
                lines.extend(
                    [
                        f'    %oacc{col}_{row}_0 = aie.buffer(%c{col}_{row}) {{sym_name = "oacc{col}_{row}_0"}} : memref<256xf32>',
                        f'    %oacc{col}_{row}_1 = aie.buffer(%c{col}_{row}) {{sym_name = "oacc{col}_{row}_1"}} : memref<256xf32>',
                    ]
                )
                if DIRECT_OUTPUT_LOCAL_FINISH:
                    lines.append(
                        f'    %oscratch{col}_{row} = aie.buffer(%c{col}_{row}) {{sym_name = "oscratch{col}_{row}"}} : memref<2048xi8>'
                    )
                if DIRECT_OUTPUT_BF16_STAGE:
                    lines.append(
                        f'    %rntail{col}_{row} = aie.buffer(%c{col}_{row}) {{sym_name = "rntail{col}_{row}"}} : memref<4096xi8>'
                    )
    if DIRECT_OUTPUT_WEIGHT_STREAM:
        for col in range(0, 8, 2):
            cores = ", ".join(f"%c{col}_{row}" for row in range(ROWS))
            lines.extend(
                [
                    f"    aie.objectfifo @owsh{col}(%shim{col}, {{%mt{col}}}, {OUTPUT_WEIGHT_SHIM_DEPTH} : i32) : !aie.objectfifo<memref<16384xi8>>",
                    f"    aie.objectfifo @owbc{col}(%mt{col}, {{{cores}}}, 1 : i32) : !aie.objectfifo<memref<16384xi8>>",
                    f"    aie.objectfifo.link [@owsh{col}] -> [@owbc{col}] ([] [0])",
                ]
            )
    if DIRECT_ATTENTION_HANDOFF:
        for row in range(ROWS):
            for pair in range(4):
                lines.append(
                    f"    aie.objectfifo @ad{pair}_{row}(%c{2 * pair + 1}_{row}, {{%c{2 * pair}_{row}}}, 3 : i32) : !aie.objectfifo<memref<4096xi8>>"
                )
    lines.extend(
        [
            '    func.func private @r30_attention_init(memref<1024xf32>, memref<8xf32>) attributes {link_with = "r71att.o"}',
            *(
                [
                    '    func.func private @r32_attention_finish_pair_packed(memref<1024xf32>, memref<8xf32>, memref<1024xf32>, memref<8xf32>, memref<4096xi8>) attributes {link_with = "r32att.o"}',
                    '    func.func private @r32_output_projection_group_m8(memref<4096xi8>, memref<16384xi8>, memref<256xf32>, i32) attributes {link_with = "r32out.o"}',
                    '    func.func private @r32_output_projection_finish_pair_m8(memref<256xf32>, memref<256xf32>, memref<2048xi8>) attributes {link_with = "r32out.o"}',
                    '    func.func private @r84_output_pairs() -> i32 attributes {link_with = "r83control.o"}',
                    '    func.func private @r84_output_waves() -> i32 attributes {link_with = "r83control.o"}',
                ]
                if DIRECT_OUTPUT_COMPUTE
                else [
                    '    func.func private @r32_attention_finish_pair_packed(memref<1024xf32>, memref<8xf32>, memref<1024xf32>, memref<8xf32>, memref<4096xi8>) attributes {link_with = "r32att.o"}'
                ]
                if DIRECT_ATTENTION_DRAIN or DIRECT_OUTPUT_WEIGHT_DRAIN
                else [
                    '    func.func private @r30_attention_finish(memref<1024xf32>, memref<8xf32>, memref<2048xi8>) attributes {link_with = "r71att.o"}'
                ]
            ),
        ]
    )
    if COMPACT_PAIRED_ATTENTION:
        lines.append(
            '    func.func private @r83_attention_blocks() -> i32 attributes {link_with = "r83control.o"}'
        )
    if DIRECT_OUTPUT_BF16_STAGE:
        lines.extend(
            [
                '    func.func private @r89_output_projection_finish_pair_bf16_split(memref<256xf32>, memref<256xf32>, memref<10240xi8>, memref<4096xi8>, i32) attributes {link_with = "r32out.o"}',
                '    func.func private @r89_emit_bf16_chunk(memref<10240xi8>, memref<4096xi8>, memref<2048xi8>, index) attributes {link_with = "r32out.o"}',
                '    func.func private @r89_q_groups() -> i32 attributes {link_with = "r83control.o"}',
                '    func.func private @r89_output_chunks() -> i32 attributes {link_with = "r83control.o"}',
            ]
        )
    if DIRECT_OUTPUT_RESIDUAL_NORM:
        lines.extend(
            [
                '    func.func private @r90_projection_blocks() -> i32 attributes {link_with = "r83control.o"}',
                '    func.func private @r90_post_residual_pre_ffn_split(memref<10240xi8>, memref<4096xi8>, memref<16384xi8>) attributes {link_with = "r90norm.o"}',
            ]
        )
    if CACHED_Q:
        lines.append(
            '    func.func private @r72_attention_block_cached(memref<6144xi32>, memref<16384xi8>, memref<1024xf32>, memref<8xf32>, i32, i32) attributes {link_with = "r71att.o"}'
        )
        lines.extend(
            [
                '    func.func private @r72_w4_scaled_group_cache(memref<10240xi8>, memref<16384xi8>, memref<6144xi32>, i32) attributes {link_with = "r72group.o"}',
                '    func.func private @r72_w4_finish_bf16_slice_cache(memref<6144xi32>, memref<2048xi8>, i32) attributes {link_with = "r72finish.o"}',
            ]
        )
    else:
        lines.append(
            '    func.func private @r30_attention_block(memref<4096xi8>, memref<16384xi8>, memref<1024xf32>, memref<8xf32>, i32) attributes {link_with = "r71att.o"}'
        )
    if not CACHED_Q:
        lines.append(
            '    func.func private @r30_attention_load_q(memref<16384xi8>, memref<4096xi8>, i32) attributes {link_with = "r71att.o"}'
        )
    return lines


def extend_output_core(op, col, row, local_finish=False):
    prefix, outer, body, suffix = outer_parts(op)
    if DIRECT_OUTPUT_RESIDUAL_NORM:
        drain_bound = "      %pblocks = arith.constant 18 : index"
        try:
            drain_index = prefix.index(drain_bound)
        except ValueError as error:
            raise SystemExit(
                f"R90 could not find projection-drain bound on core {col},{row}"
            ) from error
        prefix[drain_index : drain_index + 1] = [
            "      %pblocksi = func.call @r90_projection_blocks() : () -> i32",
            "      %pblocks = arith.index_cast %pblocksi : i32 to index",
        ]
    if DIRECT_OUTPUT_BF16_STAGE:
        final_release = f"        aie.objectfifo.release @abc{row}(Consume, 1)"
        if not body or body[-1] != final_release:
            raise SystemExit(
                f"R89 expected final {final_release!r} on output core {col},{row}"
            )
        body.pop()
    prefix.extend(
        [
            "      %opairsi = func.call @r84_output_pairs() : () -> i32",
            "      %opairs = arith.index_cast %opairsi : i32 to index",
            "      %omwavesi = func.call @r84_output_waves() : () -> i32",
            "      %omwaves = arith.index_cast %omwavesi : i32 to index",
            "      %oh0 = arith.constant 0 : i32",
            "      %oh1 = arith.constant 1 : i32",
        ]
    )
    if DIRECT_OUTPUT_BF16_STAGE:
        prefix.extend(
            [
                "      %two = arith.constant 2 : index",
                "      %qgroupsi = func.call @r89_q_groups() : () -> i32",
                "      %qgroups = arith.index_cast %qgroupsi : i32 to index",
                "      %qdrains = arith.constant 3 : index",
                "      %ochunksi = func.call @r89_output_chunks() : () -> i32",
                "      %ochunks = arith.index_cast %ochunksi : i32 to index",
            ]
        )
    if DIRECT_OUTPUT_RESIDUAL_NORM:
        prefix.extend(
            [
                "      %rnrows = arith.constant 4 : index",
                f"      %rnrow = arith.constant {row} : index",
            ]
        )
    output = [
        "        scf.for %omwave = %z to %omwaves step %one {",
        f"          %oppair = aie.objectfifo.acquire @ad{col // 2}_{row}(Consume, 3) : !aie.objectfifosubview<memref<4096xi8>>",
        "          scf.for %opair = %z to %opairs step %one {",
    ]
    for local_slice in range(2):
        for group in range(3):
            output.extend(
                [
                    f"            %opa{local_slice}_{group} = aie.objectfifo.subview.access %oppair[{group}] : !aie.objectfifosubview<memref<4096xi8>> -> memref<4096xi8>",
                    f"            %wop{local_slice}_{group} = aie.objectfifo.acquire @owbc{col}(Consume, 1) : !aie.objectfifosubview<memref<16384xi8>>",
                    f"            %wop{local_slice}_{group}v = aie.objectfifo.subview.access %wop{local_slice}_{group}[0] : !aie.objectfifosubview<memref<16384xi8>> -> memref<16384xi8>",
                    f"            func.call @r32_output_projection_group_m8(%opa{local_slice}_{group}, %wop{local_slice}_{group}v, %oacc{col}_{row}_{local_slice}, {'%oh0' if group == 0 else '%oh1'}) : (memref<4096xi8>, memref<16384xi8>, memref<256xf32>, i32) -> ()",
                    f"            aie.objectfifo.release @owbc{col}(Consume, 1)",
                ]
            )
    if DIRECT_OUTPUT_BF16_STAGE:
        output.extend(
            [
                "            %opairi = arith.index_cast %opair : index to i32",
                f"            func.call @r89_output_projection_finish_pair_bf16_split(%oacc{col}_{row}_0, %oacc{col}_{row}_1, %rv1_3v, %rntail{col}_{row}, %opairi) : (memref<256xf32>, memref<256xf32>, memref<10240xi8>, memref<4096xi8>, i32) -> ()",
            ]
        )
    elif local_finish:
        output.append(
            f"            func.call @r32_output_projection_finish_pair_m8(%oacc{col}_{row}_0, %oacc{col}_{row}_1, %oscratch{col}_{row}) : (memref<256xf32>, memref<256xf32>, memref<2048xi8>) -> ()"
        )
    else:
        output.extend(
            [
                f"            %opo = aie.objectfifo.acquire @oc{col}_{row}(Produce, 1) : !aie.objectfifosubview<memref<2048xi8>>",
                "            %opov = aie.objectfifo.subview.access %opo[0] : !aie.objectfifosubview<memref<2048xi8>> -> memref<2048xi8>",
                f"            func.call @r32_output_projection_finish_pair_m8(%oacc{col}_{row}_0, %oacc{col}_{row}_1, %opov) : (memref<256xf32>, memref<256xf32>, memref<2048xi8>) -> ()",
                f"            aie.objectfifo.release @oc{col}_{row}(Produce, 1)",
            ]
        )
    output.extend(
        [
            "          }",
            f"          aie.objectfifo.release @ad{col // 2}_{row}(Consume, 3)",
        ]
    )
    if DIRECT_OUTPUT_RESIDUAL_NORM:
        output.extend(
            [
                "          scf.for %rnparam = %z to %rnrows step %one {",
                f"            %rnp = aie.objectfifo.acquire @owbc{col}(Consume, 1) : !aie.objectfifosubview<memref<16384xi8>>",
                "            %rnpv = aie.objectfifo.subview.access %rnp[0] : !aie.objectfifosubview<memref<16384xi8>> -> memref<16384xi8>",
                "            %rnactive = arith.cmpi eq, %rnparam, %rnrow : index",
                "            scf.if %rnactive {",
                f"              func.call @r90_post_residual_pre_ffn_split(%rv1_3v, %rntail{col}_{row}, %rnpv) : (memref<10240xi8>, memref<4096xi8>, memref<16384xi8>) -> ()",
                "            }",
                f"            aie.objectfifo.release @owbc{col}(Consume, 1)",
                "          }",
            ]
        )
    if DIRECT_OUTPUT_BF16_STAGE:
        output.extend(
            [
                "          scf.for %ochunk = %z to %ochunks step %one {",
                f"            %opo = aie.objectfifo.acquire @oc{col}_{row}(Produce, 1) : !aie.objectfifosubview<memref<2048xi8>>",
                "            %opov = aie.objectfifo.subview.access %opo[0] : !aie.objectfifosubview<memref<2048xi8>> -> memref<2048xi8>",
                f"            func.call @r89_emit_bf16_chunk(%rv1_3v, %rntail{col}_{row}, %opov, %ochunk) : (memref<10240xi8>, memref<4096xi8>, memref<2048xi8>, index) -> ()",
                f"            aie.objectfifo.release @oc{col}_{row}(Produce, 1)",
                "          }",
            ]
        )
    if local_finish:
        output.extend(
            [
                f"          %drainsignal = aie.objectfifo.acquire @oc{col}_{row}(Produce, 1) : !aie.objectfifosubview<memref<2048xi8>>",
                f"          aie.objectfifo.release @oc{col}_{row}(Produce, 1)",
            ]
        )
    output.append("        }")
    if DIRECT_OUTPUT_BF16_STAGE:
        output.append(f"        aie.objectfifo.release @abc{row}(Consume, 1)")
    body.extend(output)
    return prefix + [outer] + body + suffix


def extend_attention_drain_core(op, col, row):
    prefix, outer, body, suffix = outer_parts(op)
    prefix.append("      %drainwaves = arith.constant 2 : index")
    body.extend(
        [
            "        scf.for %drainwave = %z to %drainwaves step %one {",
            f"          %drain = aie.objectfifo.acquire @ad{col // 2}_{row}(Consume, 3) : !aie.objectfifosubview<memref<4096xi8>>",
            f"          aie.objectfifo.release @ad{col // 2}_{row}(Consume, 3)",
            f"          %drainsignal = aie.objectfifo.acquire @oc{col}_{row}(Produce, 1) : !aie.objectfifosubview<memref<2048xi8>>",
            f"          aie.objectfifo.release @oc{col}_{row}(Produce, 1)",
            "        }",
        ]
    )
    return prefix + [outer] + body + suffix


def extend_output_weight_drain_core(op, col, row):
    prefix, outer, body, suffix = outer_parts(op)
    prefix.extend(
        [
            "      %drainwaves = arith.constant 2 : index",
            "      %drainpairs = arith.constant 12 : index",
            "      %draingroups = arith.constant 6 : index",
        ]
    )
    body.extend(
        [
            "        scf.for %drainwave = %z to %drainwaves step %one {",
            f"          %drain = aie.objectfifo.acquire @ad{col // 2}_{row}(Consume, 3) : !aie.objectfifosubview<memref<4096xi8>>",
            "          scf.for %drainpair = %z to %drainpairs step %one {",
            "            scf.for %draingroup = %z to %draingroups step %one {",
            f"              %drainweight = aie.objectfifo.acquire @owbc{col}(Consume, 1) : !aie.objectfifosubview<memref<16384xi8>>",
            f"              aie.objectfifo.release @owbc{col}(Consume, 1)",
            "            }",
            "          }",
            f"          aie.objectfifo.release @ad{col // 2}_{row}(Consume, 3)",
            f"          %drainsignal = aie.objectfifo.acquire @oc{col}_{row}(Produce, 1) : !aie.objectfifosubview<memref<2048xi8>>",
            f"          aie.objectfifo.release @oc{col}_{row}(Produce, 1)",
            "        }",
        ]
    )
    return prefix + [outer] + body + suffix


def extend_core(op, col, row):
    if col not in ACTIVE_COLS:
        if DIRECT_OUTPUT and col % 2 == 0:
            return extend_output_core(op, col, row)
        if DIRECT_OUTPUT_LOCAL_FINISH and col % 2 == 0:
            return extend_output_core(op, col, row, local_finish=True)
        if DIRECT_ATTENTION_DRAIN and col % 2 == 0:
            return extend_attention_drain_core(op, col, row)
        if DIRECT_OUTPUT_WEIGHT_DRAIN and col % 2 == 0:
            return extend_output_weight_drain_core(op, col, row)
        return op
    prefix, outer, body, suffix = outer_parts(op)
    if CACHED_Q:
        suffix = [line.replace("stack_size = 4096", "stack_size = 2048") for line in suffix]
    prefix.append(f"      %attgroups = arith.constant {QUERY_GROUPS} : index")
    if COMPACT_PAIRED_ATTENTION:
        prefix.extend(
            [
                "      %attblocksi = func.call @r83_attention_blocks() : () -> i32",
                "      %attblocks = arith.index_cast %attblocksi : i32 to index",
            ]
        )
    else:
        prefix.append("      %attblocks = arith.constant 16 : index")
    if not CACHED_Q:
        prefix.append(f"      %attpair = arith.constant {row} : i32")
    attention = []
    if BATCH2:
        attention.extend(
            [
                "        scf.for %attbatch = %z to %attgroups step %two {",
                f"          %aattq0 = aie.objectfifo.acquire @wbc{col}(Consume, 1) : !aie.objectfifosubview<memref<16384xi8>>",
                "          %aattq0v = aie.objectfifo.subview.access %aattq0[0] : !aie.objectfifosubview<memref<16384xi8>> -> memref<16384xi8>",
                f"          func.call @r30_attention_load_q(%aattq0v, %attq0{col}_{row}, %attpair) : (memref<16384xi8>, memref<4096xi8>, i32) -> ()",
                f"          aie.objectfifo.release @wbc{col}(Consume, 1)",
                f"          %aattq1 = aie.objectfifo.acquire @wbc{col}(Consume, 1) : !aie.objectfifosubview<memref<16384xi8>>",
                "          %aattq1v = aie.objectfifo.subview.access %aattq1[0] : !aie.objectfifosubview<memref<16384xi8>> -> memref<16384xi8>",
                f"          func.call @r30_attention_load_q(%aattq1v, %attq1{col}_{row}, %attpair) : (memref<16384xi8>, memref<4096xi8>, i32) -> ()",
                f"          aie.objectfifo.release @wbc{col}(Consume, 1)",
            ]
        )
        for slot in range(4):
            attention.append(
                f"          func.call @r30_attention_init(%attacc{slot}{col}_{row}, %attstats{slot}{col}_{row}) : (memref<1024xf32>, memref<8xf32>) -> ()"
            )
        attention.extend(
            [
                "          scf.for %attblock = %z to %attblocks step %one {",
                f"            %aattkv = aie.objectfifo.acquire @wbc{col}(Consume, 1) : !aie.objectfifosubview<memref<16384xi8>>",
                "            %aattkvv = aie.objectfifo.subview.access %aattkv[0] : !aie.objectfifosubview<memref<16384xi8>> -> memref<16384xi8>",
            ]
        )
        for qslot in range(2):
            for lane in range(2):
                slot = 2 * qslot + lane
                attention.append(
                    f"            func.call @r30_attention_block(%attq{qslot}{col}_{row}, %aattkvv, %attacc{slot}{col}_{row}, %attstats{slot}{col}_{row}, %h{lane}) : (memref<4096xi8>, memref<16384xi8>, memref<1024xf32>, memref<8xf32>, i32) -> ()"
                )
        attention.extend(
            [
                f"            aie.objectfifo.release @wbc{col}(Consume, 1)",
                "          }",
            ]
        )
        for slot in range(4):
            attention.extend(
                [
                    f"          %atto{slot} = aie.objectfifo.acquire @oc{col}_{row}(Produce, 1) : !aie.objectfifosubview<memref<2048xi8>>",
                    f"          %atto{slot}v = aie.objectfifo.subview.access %atto{slot}[0] : !aie.objectfifosubview<memref<2048xi8>> -> memref<2048xi8>",
                    f"          func.call @r30_attention_finish(%attacc{slot}{col}_{row}, %attstats{slot}{col}_{row}, %atto{slot}v) : (memref<1024xf32>, memref<8xf32>, memref<2048xi8>) -> ()",
                    f"          aie.objectfifo.release @oc{col}_{row}(Produce, 1)",
                ]
            )
        attention.append("        }")
        prefix.append("      %two = arith.constant 2 : index")
        body.extend(attention)
        return prefix + [outer] + body + suffix
    if ADJACENT_Q:
        attention.extend(
            [
                f"        %qadjc = aie.objectfifo.acquire @qadj{col // 2}_{row}(Consume, 1) : !aie.objectfifosubview<memref<6144xi32>>",
                "        %qadjcv = aie.objectfifo.subview.access %qadjc[0] : !aie.objectfifosubview<memref<6144xi32>> -> memref<6144xi32>",
            ]
        )
    attention.append("        scf.for %attgroup = %z to %attgroups step %one {")
    if CACHED_Q:
        attention.append("          %attgroupi = arith.index_cast %attgroup : index to i32")
        qcache = "%qadjcv" if ADJACENT_Q else f"%acc{col}_{row}"
        call0 = (
            f"            func.call @r72_attention_block_cached({qcache}, %aattkvv, %attacc0{col}_{row}, %attstats0{col}_{row}, %attgroupi, %h0) : "
            f"(memref<6144xi32>, memref<16384xi8>, memref<1024xf32>, memref<8xf32>, i32, i32) -> ()"
        )
        call1 = (
            f"            func.call @r72_attention_block_cached({qcache}, %aattkvv, %attacc1{col}_{row}, %attstats1{col}_{row}, %attgroupi, %h1) : "
            f"(memref<6144xi32>, memref<16384xi8>, memref<1024xf32>, memref<8xf32>, i32, i32) -> ()"
        )
    else:
        attention.extend(
            [
                f"          %aattq = aie.objectfifo.acquire @wbc{col}(Consume, 1) : !aie.objectfifosubview<memref<16384xi8>>",
                "          %aattqv = aie.objectfifo.subview.access %aattq[0] : !aie.objectfifosubview<memref<16384xi8>> -> memref<16384xi8>",
                f"          func.call @r30_attention_load_q(%aattqv, %attq{col}_{row}, %attpair) : (memref<16384xi8>, memref<4096xi8>, i32) -> ()",
                f"          aie.objectfifo.release @wbc{col}(Consume, 1)",
            ]
        )
        q = f"%attq{col}_{row}"
        call0 = f"            func.call @r30_attention_block({q}, %aattkvv, %attacc0{col}_{row}, %attstats0{col}_{row}, %h0) : (memref<4096xi8>, memref<16384xi8>, memref<1024xf32>, memref<8xf32>, i32) -> ()"
        call1 = f"            func.call @r30_attention_block({q}, %aattkvv, %attacc1{col}_{row}, %attstats1{col}_{row}, %h1) : (memref<4096xi8>, memref<16384xi8>, memref<1024xf32>, memref<8xf32>, i32) -> ()"
    attention.extend(
        [
            f"          func.call @r30_attention_init(%attacc0{col}_{row}, %attstats0{col}_{row}) : (memref<1024xf32>, memref<8xf32>) -> ()",
            f"          func.call @r30_attention_init(%attacc1{col}_{row}, %attstats1{col}_{row}) : (memref<1024xf32>, memref<8xf32>) -> ()",
            "          scf.for %attblock = %z to %attblocks step %one {",
            f"            %aattkv = aie.objectfifo.acquire @wbc{col}(Consume, 1) : !aie.objectfifosubview<memref<16384xi8>>",
            "            %aattkvv = aie.objectfifo.subview.access %aattkv[0] : !aie.objectfifosubview<memref<16384xi8>> -> memref<16384xi8>",
            call0,
            call1,
            f"            aie.objectfifo.release @wbc{col}(Consume, 1)",
            "          }",
        ]
    )
    if DIRECT_ATTENTION_HANDOFF:
        attention.extend(
            [
                f"          %attout = aie.objectfifo.acquire @ad{col // 2}_{row}(Produce, 1) : !aie.objectfifosubview<memref<4096xi8>>",
                "          %attoutv = aie.objectfifo.subview.access %attout[0] : !aie.objectfifosubview<memref<4096xi8>> -> memref<4096xi8>",
                f"          func.call @r32_attention_finish_pair_packed(%attacc0{col}_{row}, %attstats0{col}_{row}, %attacc1{col}_{row}, %attstats1{col}_{row}, %attoutv) : (memref<1024xf32>, memref<8xf32>, memref<1024xf32>, memref<8xf32>, memref<4096xi8>) -> ()",
                f"          aie.objectfifo.release @ad{col // 2}_{row}(Produce, 1)",
                "        }",
            ]
        )
    else:
        attention.extend(
            [
                f"          %atto0 = aie.objectfifo.acquire @oc{col}_{row}(Produce, 1) : !aie.objectfifosubview<memref<2048xi8>>",
                "          %atto0v = aie.objectfifo.subview.access %atto0[0] : !aie.objectfifosubview<memref<2048xi8>> -> memref<2048xi8>",
                f"          func.call @r30_attention_finish(%attacc0{col}_{row}, %attstats0{col}_{row}, %atto0v) : (memref<1024xf32>, memref<8xf32>, memref<2048xi8>) -> ()",
                f"          aie.objectfifo.release @oc{col}_{row}(Produce, 1)",
                f"          %atto1 = aie.objectfifo.acquire @oc{col}_{row}(Produce, 1) : !aie.objectfifosubview<memref<2048xi8>>",
                "          %atto1v = aie.objectfifo.subview.access %atto1[0] : !aie.objectfifosubview<memref<2048xi8>> -> memref<2048xi8>",
                f"          func.call @r30_attention_finish(%attacc1{col}_{row}, %attstats1{col}_{row}, %atto1v) : (memref<1024xf32>, memref<8xf32>, memref<2048xi8>) -> ()",
                f"          aie.objectfifo.release @oc{col}_{row}(Produce, 1)",
                "        }",
            ]
        )
    if ADJACENT_Q:
        attention.append(
            f"        aie.objectfifo.release @qadj{col // 2}_{row}(Consume, 1)"
        )
    body.extend(attention)
    result = prefix + [outer] + body + suffix
    if DIRECT_Q or LOCAL_Q:
        converted = []
        for line in result:
            if "@r70_w4_scaled_group" in line:
                line = line.replace(
                    "@r70_w4_scaled_group", "@r72_w4_scaled_group_cache"
                ).replace("memref<2304xi32>", "memref<6144xi32>")
            if "@r65_w4_finish_bf16_slice" in line:
                line = line.replace(
                    "@r65_w4_finish_bf16_slice",
                    "@r72_w4_finish_bf16_slice_cache",
                ).replace("memref<2304xi32>", "memref<6144xi32>")
            converted.append(line)
        result = converted
    return result


def dims(count, block):
    return f"[<size = {count}, stride = {block}>, <size = {block // 512}, stride = 512>, <size = 512, stride = 1>]"


def direct_output_dims():
    return (
        "[<size = 4, stride = 24576>, <size = 8, stride = 3072>, "
        "<size = 256, stride = 1>]"
    )


def start_direct_output_tasks(lines, mwave, first_pair, pair_count):
    output_base = STAGE_BYTES + ATTENTION_BYTES
    names = []
    for output_pair in range(first_pair, first_pair + pair_count):
        for active_col, col in enumerate(range(0, 8, 2)):
            name = f"tdo{mwave}_{output_pair}_{col}"
            offset = (
                output_base
                + mwave * 128 * 768 * 4
                + active_col * 32 * 768 * 4
                + output_pair * 64 * 4
            )
            lines.extend(
                [
                    f"      %{name} = aiex.dma_configure_task_for @osh{col} {{",
                    f"        aie.dma_bd(%R : memref<{RESULT_BYTES}xi8>, {offset}, {OUT_JOIN}, {direct_output_dims()}) {{burst_length = 0 : i32}}",
                    "        aie.end",
                    "      } {issue_token = true}",
                    f"      aiex.dma_start_task(%{name})",
                ]
            )
            names.append(name)
    return names


def start_bf16_output_tasks(lines, mwave):
    output_base = 0 if DIRECT_OUTPUT_FFN_HANDOFF else STAGE_BYTES + ATTENTION_BYTES
    names = []
    for active_col, col in enumerate(range(0, 8, 2)):
        for block in range(3):
            for half in range(2):
                name = f"tbfo{mwave}_{col}_{block}_{half}"
                offset = (
                    output_base
                    + mwave * 128 * 768 * 2
                    + active_col * 32 * 768 * 2
                    + block * 256 * 2
                    + half * 4 * 768 * 2
                )
                lines.extend(
                    [
                        f"      %{name} = aiex.dma_configure_task_for @osh{col} {{",
                        f"        aie.dma_bd(%R : memref<{RESULT_BYTES}xi8>, {offset}, {OUT_JOIN}, [<size = {ROWS}, stride = {8 * 768 * 2}>, <size = 4, stride = {768 * 2}>, <size = 512, stride = 1>]) {{burst_length = 0 : i32}}",
                        "        aie.end",
                        "      } {issue_token = true}",
                        f"      aiex.dma_start_task(%{name})",
                    ]
                )
                names.append(name)
    return names


def await_tasks(lines, names):
    for name in names:
        lines.extend(
            [f"      aiex.dma_await_task(%{name})", f"      aiex.dma_free_task(%{name})"]
        )


def direct_attention_drain_runtime():
    lines = []
    weight_names = []
    if DIRECT_OUTPUT_WEIGHT_DRAIN or DIRECT_OUTPUT_LOCAL_FINISH:
        blocks_per_col = 72
        for active_col, col in enumerate(range(0, 8, 2)):
            name = f"tow{col}"
            offset = QKV_WEIGHT_BYTES + active_col * blocks_per_col * 16_384
            lines.extend(
                [
                    f"      %{name} = aiex.dma_configure_task_for @owsh{col} {{",
                    f"        aie.dma_bd(%W : memref<{WEIGHT_BYTES}xi8>, {offset}, {blocks_per_col * 16_384}, {dims(blocks_per_col, 16_384)}) {{burst_length = 0 : i32}}",
                    "        aie.end",
                    "      } {issue_token = true, repeat_count = 1 : i32}",
                    f"      aiex.dma_start_task(%{name})",
                ]
            )
            weight_names.append(name)
    order = [0, 2, 4, 1, 3, 5]
    for wave, groups in enumerate((order[:3], order[3:])):
        completion_names = []
        for active_col, col in enumerate(range(0, 8, 2)):
            name = f"tdrain{wave}_{col}"
            offset = STAGE_BYTES + (wave * 4 + active_col) * OUT_JOIN
            lines.extend(
                [
                    f"      %{name} = aiex.dma_configure_task_for @osh{col} {{",
                    f"        aie.dma_bd(%R : memref<{RESULT_BYTES}xi8>, {offset}, {OUT_JOIN}, {dims(1, OUT_JOIN)}) {{burst_length = 0 : i32}}",
                    "        aie.end",
                    "      } {issue_token = true}",
                    f"      aiex.dma_start_task(%{name})",
                ]
            )
            completion_names.append(name)
        input_names = []
        for execution_group, group in enumerate(groups, start=wave * 3):
            for token_row, col in enumerate(ACTIVE_COLS):
                qname = f"taqi{execution_group}_{col}"
                qoffset = (token_row * QUERY_GROUPS + group) * Q_JOIN
                lines.extend(
                    [
                        f"      %{qname} = aiex.dma_configure_task_for @wsh{col} {{",
                        f"        aie.dma_bd(%Q : memref<{Q_BYTES}xi8>, {qoffset}, {Q_JOIN}, {dims(1, Q_JOIN)}) {{burst_length = 0 : i32}}",
                        "        aie.end",
                        "      } {issue_token = true}",
                        f"      aiex.dma_start_task(%{qname})",
                    ]
                )
                input_names.append(qname)
                kvname = f"takvi{execution_group}_{col}"
                lines.extend(
                    [
                        f"      %{kvname} = aiex.dma_configure_task_for @wsh{col} {{",
                        f"        aie.dma_bd(%KV : memref<{KV_BYTES}xi8>, 0, {KV_BYTES}, {dims(16, KV_TILE)}) {{burst_length = 0 : i32}}",
                        "        aie.end",
                        "      } {issue_token = true}",
                        f"      aiex.dma_start_task(%{kvname})",
                    ]
                )
                input_names.append(kvname)
        await_tasks(lines, input_names)
        await_tasks(lines, completion_names)
    await_tasks(lines, weight_names)
    return lines


def direct_output_runtime():
    lines = []
    weight_names = []
    blocks_per_col = 72
    if not DIRECT_OUTPUT_RESIDUAL_NORM:
        for active_col, col in enumerate(range(0, 8, 2)):
            name = f"tow{col}"
            offset = QKV_WEIGHT_BYTES + active_col * blocks_per_col * 16_384
            lines.extend(
                [
                    f"      %{name} = aiex.dma_configure_task_for @owsh{col} {{",
                    f"        aie.dma_bd(%W : memref<{WEIGHT_BYTES}xi8>, {offset}, {blocks_per_col * 16_384}, {dims(blocks_per_col, 16_384)}) {{burst_length = 0 : i32}}",
                    "        aie.end",
                    "      } {issue_token = true, repeat_count = 1 : i32}",
                    f"      aiex.dma_start_task(%{name})",
                ]
            )
            weight_names.append(name)

    output_names = (
        start_bf16_output_tasks(lines, 0)
        if DIRECT_OUTPUT_BF16_STAGE
        else start_direct_output_tasks(lines, 0, 0, 6)
    )
    order = [0, 2, 4, 1, 3, 5]
    for window, groups in enumerate((order[:3], order[3:])):
        window_weight_names = []
        if DIRECT_OUTPUT_RESIDUAL_NORM:
            for active_col, col in enumerate(range(0, 8, 2)):
                weight_name = f"tow{window}_{col}"
                weight_offset = QKV_WEIGHT_BYTES + active_col * blocks_per_col * 16_384
                param_name = f"trnp{window}_{col}"
                param_offset = (
                    QKV_WEIGHT_BYTES
                    + O_WEIGHT_BYTES
                    + (active_col * RN_BLOCKS_PER_COL + window * ROWS) * 16_384
                )
                lines.extend(
                    [
                        f"      %{weight_name} = aiex.dma_configure_task_for @owsh{col} {{",
                        f"        aie.dma_bd(%W : memref<{WEIGHT_BYTES}xi8>, {weight_offset}, {blocks_per_col * 16_384}, {dims(blocks_per_col, 16_384)}) {{burst_length = 0 : i32}}",
                        "        aie.end",
                        "      } {issue_token = true}",
                        f"      aiex.dma_start_task(%{weight_name})",
                        f"      %{param_name} = aiex.dma_configure_task_for @owsh{col} {{",
                        f"        aie.dma_bd(%W : memref<{WEIGHT_BYTES}xi8>, {param_offset}, {ROWS * 16_384}, {dims(ROWS, 16_384)}) {{burst_length = 0 : i32}}",
                        "        aie.end",
                        "      } {issue_token = true}",
                        f"      aiex.dma_start_task(%{param_name})",
                    ]
                )
                window_weight_names.extend((weight_name, param_name))
        input_names = []
        for execution_group, group in enumerate(groups, start=window * 3):
            for token_row, col in enumerate(ACTIVE_COLS):
                qname = f"taqi{execution_group}_{col}"
                qoffset = (token_row * QUERY_GROUPS + group) * Q_JOIN
                lines.extend(
                    [
                        f"      %{qname} = aiex.dma_configure_task_for @wsh{col} {{",
                        f"        aie.dma_bd(%Q : memref<{Q_BYTES}xi8>, {qoffset}, {Q_JOIN}, {dims(1, Q_JOIN)}) {{burst_length = 0 : i32}}",
                        "        aie.end",
                        "      } {issue_token = true}",
                        f"      aiex.dma_start_task(%{qname})",
                    ]
                )
                input_names.append(qname)
                kvname = f"takvi{execution_group}_{col}"
                lines.extend(
                    [
                        f"      %{kvname} = aiex.dma_configure_task_for @wsh{col} {{",
                        f"        aie.dma_bd(%KV : memref<{KV_BYTES}xi8>, 0, {KV_BYTES}, {dims(16, KV_TILE)}) {{burst_length = 0 : i32}}",
                        "        aie.end",
                        "      } {issue_token = true}",
                        f"      aiex.dma_start_task(%{kvname})",
                    ]
                )
                input_names.append(kvname)
        await_tasks(lines, input_names)
        await_tasks(lines, output_names)
        await_tasks(lines, window_weight_names)
        if DIRECT_OUTPUT_BF16_STAGE and window == 0:
            output_names = start_bf16_output_tasks(lines, 1)
        elif DIRECT_OUTPUT_BF16_STAGE:
            pass
        elif window == 0:
            second_half = start_direct_output_tasks(lines, 0, 6, 6)
            await_tasks(lines, second_half)
            output_names = start_direct_output_tasks(lines, 1, 0, 6)
        else:
            second_half = start_direct_output_tasks(lines, 1, 6, 6)
            await_tasks(lines, second_half)

    await_tasks(lines, weight_names)
    return lines


def attention_runtime():
    if (
        DIRECT_ATTENTION_DRAIN
        or DIRECT_OUTPUT_WEIGHT_DRAIN
        or DIRECT_OUTPUT_LOCAL_FINISH
    ):
        return direct_attention_drain_runtime()
    if DIRECT_OUTPUT:
        return direct_output_runtime()
    lines = []
    if ENQUEUE_WINDOW:
        for base_group in range(0, QUERY_GROUPS, ENQUEUE_WINDOW):
            input_names = []
            output_names = []
            for group in range(
                base_group, min(base_group + ENQUEUE_WINDOW, QUERY_GROUPS)
            ):
                for token_row, col in enumerate(ACTIVE_COLS):
                    for lane in range(2):
                        name = f"tao{group}_{col}_{lane}"
                        offset = (
                            STAGE_BYTES
                            + (lane * QUERY_GROUPS + group) * OUT_JOIN
                            + token_row * OUT_TILE
                        )
                        lines.extend(
                            [
                                f"      %{name} = aiex.dma_configure_task_for @osh{col} {{",
                                f"        aie.dma_bd(%R : memref<{RESULT_BYTES}xi8>, {offset}, {OUT_JOIN}, [<size = 4, stride = {2 * QUERY_GROUPS * OUT_JOIN}>, <size = 4, stride = 512>, <size = 512, stride = 1>]) {{burst_length = 0 : i32}}",
                                "        aie.end",
                                "      } {issue_token = true}",
                                f"      aiex.dma_start_task(%{name})",
                            ]
                        )
                        output_names.append(name)
                for token_row, col in enumerate(ACTIVE_COLS):
                    qname = f"taqi{group}_{col}"
                    qoffset = (token_row * QUERY_GROUPS + group) * Q_JOIN
                    lines.extend(
                        [
                            f"      %{qname} = aiex.dma_configure_task_for @wsh{col} {{",
                            f"        aie.dma_bd(%Q : memref<{Q_BYTES}xi8>, {qoffset}, {Q_JOIN}, {dims(1, Q_JOIN)}) {{burst_length = 0 : i32}}",
                            "        aie.end",
                            "      } {issue_token = true}",
                            f"      aiex.dma_start_task(%{qname})",
                        ]
                    )
                    input_names.append(qname)
                    kvname = f"takvi{group}_{col}"
                    lines.extend(
                        [
                            f"      %{kvname} = aiex.dma_configure_task_for @wsh{col} {{",
                            f"        aie.dma_bd(%KV : memref<{KV_BYTES}xi8>, 0, {KV_BYTES}, {dims(16, KV_TILE)}) {{burst_length = 0 : i32}}",
                            "        aie.end",
                            "      } {issue_token = true}",
                            f"      aiex.dma_start_task(%{kvname})",
                        ]
                    )
                    input_names.append(kvname)
            for name in input_names + output_names:
                lines.extend(
                    [f"      aiex.dma_await_task(%{name})", f"      aiex.dma_free_task(%{name})"]
                )
        return lines
    if BATCH2:
        for base_group in range(0, QUERY_GROUPS, 2):
            groups = (base_group, base_group + 1)
            for group in groups:
                for token_row, col in enumerate(ACTIVE_COLS):
                    for lane in range(2):
                        name = f"tao{group}_{col}_{lane}"
                        offset = (
                            STAGE_BYTES
                            + (lane * QUERY_GROUPS + group) * OUT_JOIN
                            + token_row * OUT_TILE
                        )
                        lines.extend(
                            [
                                f"      %{name} = aiex.dma_configure_task_for @osh{col} {{",
                                f"        aie.dma_bd(%R : memref<{RESULT_BYTES}xi8>, {offset}, {OUT_JOIN}, [<size = 4, stride = {2 * QUERY_GROUPS * OUT_JOIN}>, <size = 4, stride = 512>, <size = 512, stride = 1>]) {{burst_length = 0 : i32}}",
                                "        aie.end",
                                "      } {issue_token = true}",
                                f"      aiex.dma_start_task(%{name})",
                            ]
                        )
            for group in groups:
                for token_row, col in enumerate(ACTIVE_COLS):
                    qname = f"taqi{group}_{col}"
                    qoffset = (token_row * QUERY_GROUPS + group) * Q_JOIN
                    lines.extend(
                        [
                            f"      %{qname} = aiex.dma_configure_task_for @wsh{col} {{",
                            f"        aie.dma_bd(%Q : memref<{Q_BYTES}xi8>, {qoffset}, {Q_JOIN}, {dims(1, Q_JOIN)}) {{burst_length = 0 : i32}}",
                            "        aie.end",
                            "      } {issue_token = true}",
                            f"      aiex.dma_start_task(%{qname})",
                        ]
                    )
            for col in ACTIVE_COLS:
                name = f"takvi{base_group}_{col}"
                lines.extend(
                    [
                        f"      %{name} = aiex.dma_configure_task_for @wsh{col} {{",
                        f"        aie.dma_bd(%KV : memref<{KV_BYTES}xi8>, 0, {KV_BYTES}, {dims(16, KV_TILE)}) {{burst_length = 0 : i32}}",
                        "        aie.end",
                        "      } {issue_token = true}",
                        f"      aiex.dma_start_task(%{name})",
                    ]
                )
            for group in groups:
                for col in ACTIVE_COLS:
                    name = f"taqi{group}_{col}"
                    lines.extend(
                        [f"      aiex.dma_await_task(%{name})", f"      aiex.dma_free_task(%{name})"]
                    )
            for col in ACTIVE_COLS:
                name = f"takvi{base_group}_{col}"
                lines.extend(
                    [f"      aiex.dma_await_task(%{name})", f"      aiex.dma_free_task(%{name})"]
                )
            for group in groups:
                for col in ACTIVE_COLS:
                    for lane in range(2):
                        name = f"tao{group}_{col}_{lane}"
                        lines.extend(
                            [f"      aiex.dma_await_task(%{name})", f"      aiex.dma_free_task(%{name})"]
                        )
        return lines
    for group in range(QUERY_GROUPS):
        for token_row, col in enumerate(ACTIVE_COLS):
            for lane in range(2):
                name = f"tao{group}_{col}_{lane}"
                if LOCAL_Q or ADJACENT_Q:
                    pair = col // 2 if ADJACENT_Q else col
                    offset = (
                        STAGE_BYTES
                        + pair * 2 * QUERY_GROUPS * OUT_JOIN
                        + (lane * QUERY_GROUPS + group) * OUT_JOIN
                    )
                    output_dims = dims(1, OUT_JOIN)
                else:
                    offset = (
                        STAGE_BYTES
                        + (lane * QUERY_GROUPS + group) * OUT_JOIN
                        + token_row * OUT_TILE
                    )
                    output_dims = (
                        f"[<size = 4, stride = {2 * QUERY_GROUPS * OUT_JOIN}>, "
                        "<size = 4, stride = 512>, <size = 512, stride = 1>]"
                    )
                lines.extend(
                    [
                        f"      %{name} = aiex.dma_configure_task_for @osh{col} {{",
                        f"        aie.dma_bd(%R : memref<{RESULT_BYTES}xi8>, {offset}, {OUT_JOIN}, {output_dims}) {{burst_length = 0 : i32}}",
                        "        aie.end",
                        "      } {issue_token = true}",
                        f"      aiex.dma_start_task(%{name})",
                    ]
                )
        for token_row, col in enumerate(ACTIVE_COLS):
            if not CACHED_Q:
                qname = f"taqi{group}_{col}"
                qoffset = (token_row * QUERY_GROUPS + group) * Q_JOIN
                lines.extend(
                    [
                        f"      %{qname} = aiex.dma_configure_task_for @wsh{col} {{",
                        f"        aie.dma_bd(%Q : memref<{Q_BYTES}xi8>, {qoffset}, {Q_JOIN}, {dims(1, Q_JOIN)}) {{burst_length = 0 : i32}}",
                        "        aie.end",
                        "      } {issue_token = true}",
                        f"      aiex.dma_start_task(%{qname})",
                    ]
                )
            lines.extend(
                [
                    f"      %takvi{group}_{col} = aiex.dma_configure_task_for @wsh{col} {{",
                    f"        aie.dma_bd(%KV : memref<{KV_BYTES}xi8>, 0, {KV_BYTES}, {dims(16, KV_TILE)}) {{burst_length = 0 : i32}}",
                    "        aie.end",
                    "      } {issue_token = true}",
                    f"      aiex.dma_start_task(%takvi{group}_{col})",
                ]
            )
        for col in ACTIVE_COLS:
            names = [f"takvi{group}_{col}"]
            if not CACHED_Q:
                names.insert(0, f"taqi{group}_{col}")
            for name in names:
                lines.extend([f"      aiex.dma_await_task(%{name})", f"      aiex.dma_free_task(%{name})"])
        for col in ACTIVE_COLS:
            for lane in range(2):
                name = f"tao{group}_{col}_{lane}"
                lines.extend([f"      aiex.dma_await_task(%{name})", f"      aiex.dma_free_task(%{name})"])
    return lines


def shift_stage_runtime_for_ffn_handoff(lines):
    pattern = re.compile(
        rf"(aie\.dma_bd\(%R : memref<{RESULT_BYTES}xi8>, )(\d+)(,)"
    )
    shifted = []
    count = 0
    for line in lines:
        def replace(match):
            nonlocal count
            count += 1
            return f"{match.group(1)}{int(match.group(2)) + FFN_HANDOFF_BYTES}{match.group(3)}"

        shifted.append(pattern.sub(replace, line))
    if count == 0:
        raise SystemExit("R91 found no source-stage DMA offsets to shift")
    return shifted


if PAIRED_ATTENTION:
    source_args = (
        ["--single-group-function", "--dynamic-slice-loop"]
        if COMPACT_PAIRED_ATTENTION
        else []
    )
elif DIRECT_Q:
    source_args = ["--r71-pack-free-4-7", "--r72-direct-q"]
elif LOCAL_Q:
    source_args = ["--r72-local-q"]
elif ADJACENT_Q:
    source_args = ["--r73-adjacent-q"]
elif ODD_ATTENTION:
    source_args = ["--r78-odd-attention"]
else:
    source_args = ["--r71-pack-free-4-7"]
source_generator = R81 if PAIRED_ATTENTION else R70
source = generate(source_generator, *source_args).replace(
    f"memref<{STAGE_BYTES}xi8>", f"memref<{RESULT_BYTES}xi8>"
)
if DIRECT_OUTPUT_WEIGHT_STREAM:
    source = source.replace(
        f"memref<{QKV_WEIGHT_BYTES}xi8>", f"memref<{WEIGHT_BYTES}xi8>"
    )
if ADJACENT_Q or BATCH2:
    source = source.replace("stack_size = 4096", "stack_size = 2048")
if DIRECT_Q or LOCAL_Q:
    for col in ACTIVE_COLS:
        for row in range(ROWS):
            source = source.replace(
                f'{{sym_name = "acc{col}_{row}"}} : memref<2304xi32>',
                f'{{sym_name = "acc{col}_{row}"}} : memref<6144xi32>',
            )
            if DIRECT_Q:
                source = source.replace(
                    f'    %qcache{col}_{row} = aie.buffer(%c{col}_{row}) {{sym_name = "qcache{col}_{row}"}} : memref<{QUERY_GROUPS * 2 * OUT_TILE}xi8>\n',
                    "",
                ).replace(f"%qcache{col}_{row}", f"%acc{col}_{row}")
    if DIRECT_Q:
        source = source.replace(
            f"memref<{QUERY_GROUPS * 2 * OUT_TILE}xi8>, i32, i32) attributes {{link_with = \"r72stream.o\"}}",
            'memref<6144xi32>, i32, i32) attributes {link_with = "r72stream.o"}',
        ).replace(
            f"memref<{QUERY_GROUPS * 2 * OUT_TILE}xi8>, i32, i32) -> ()",
            "memref<6144xi32>, i32, i32) -> ()",
        )
ops = top_level_ops(source)
combined = []
inserted = False
for op in ops:
    key = core_key(op)
    if key is not None and not inserted:
        combined.extend(attention_top_ops())
        inserted = True
    if key is not None:
        combined.extend(extend_core(op, *key))
    elif is_runtime(op):
        if op[-1] != "    }":
            raise SystemExit("R70 runtime sequence shape changed")
        source_runtime = op[:-1]
        if DIRECT_OUTPUT_FFN_HANDOFF:
            source_runtime = shift_stage_runtime_for_ffn_handoff(source_runtime)
        combined.extend(source_runtime)
        combined.extend(attention_runtime())
        combined.append(op[-1])
    else:
        combined.extend(op)

if not inserted:
    raise SystemExit("R70 cores not found")
print("\n".join(["module {", "  aie.device(npu2) {", *combined, "  }", "}"]))
