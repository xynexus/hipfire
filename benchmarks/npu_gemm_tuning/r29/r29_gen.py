#!/usr/bin/env python3
"""Generate resident W8 QKV packing, optionally followed by R27 attention."""

import sys

ATTENTION = "--attention" in sys.argv[1:]
OUTPUT_PROJECTION = "--output-projection" in sys.argv[1:]
OUTPUT_EXECUTION = OUTPUT_PROJECTION and "--no-output-execution" not in sys.argv[1:]
OUTPUT_FIRST = "--attention-output-first" in sys.argv[1:]
DIRECT_OUTPUT = "--direct-output-projection" in sys.argv[1:]
RESIDUAL_NORM = "--residual-norm" in sys.argv[1:]
EXTERNAL_RESIDUAL = "--external-residual" in sys.argv[1:]
if EXTERNAL_RESIDUAL and not RESIDUAL_NORM:
    raise SystemExit("--external-residual requires --residual-norm")
PAIRED_QKV = "--paired-qkv" in sys.argv[1:] or RESIDUAL_NORM
if PAIRED_QKV:
    DIRECT_OUTPUT = True
if DIRECT_OUTPUT:
    ATTENTION = True
    OUTPUT_PROJECTION = True
    OUTPUT_EXECUTION = True
if OUTPUT_PROJECTION and not ATTENTION:
    raise SystemExit("--output-projection requires --attention")
if OUTPUT_FIRST and (not ATTENTION or OUTPUT_PROJECTION):
    raise SystemExit("--attention-output-first requires standalone --attention")
if DIRECT_OUTPUT and OUTPUT_FIRST:
    raise SystemExit("--direct-output-projection cannot use --attention-output-first")
OBJECT = "r30.o" if ATTENTION else "r29.o"
OUTPUT_OBJECT = "r31.o"

COLS, ROWS = 8, 4
GROUPS, M_MACROS, N_MACROS = 3, 3, 5
OUTBLOCKS = M_MACROS * N_MACROS
A_BLOCK, W_BLOCK = (16384 if ATTENTION else 10240), 16384
PAIR_W_RECORD, PAIR_W_BLOCK = 8192, 16384
ACC_ELEMS = 768
O_ACC_ELEMS = 256 if DIRECT_OUTPUT else 1024
INBLOCKS = GROUPS * OUTBLOCKS
A_BASE_BYTES = ROWS * INBLOCKS * A_BLOCK
A_BYTES = A_BASE_BYTES + (2 * (COLS // 2) * ROWS * A_BLOCK if EXTERNAL_RESIDUAL else 0)
W_BYTES = (
    (COLS // 2) * INBLOCKS * PAIR_W_BLOCK
    if PAIRED_QKV
    else COLS * INBLOCKS * W_BLOCK
)

QUERY_GROUPS = 6
PAIR, PAIRS_PER_ROLE, ROLES = A_BLOCK, 48, 5
R_STAGE_BYTES = ROLES * PAIRS_PER_ROLE * PAIR
OUT_TILE, OUT_JOIN = 2048, 8192
DIRECT_ATT_TILE = 4096
Q_BYTES, KV_BYTES = 393216, 262144
Q_JOIN, KV_TILE, K_HALF = 16384, 16384, 8192
ATT_ACC, ATT_STATS, ATT_BYTES = 1024, 8, 393216
O_GROUPS, O_SLICES, O_M_WAVES = 3, (24 if DIRECT_OUTPUT else 6), 2
O_WEIGHTS_PER_COL = O_GROUPS * O_SLICES
O_ACTIVE_COLS = COLS // 2
O_W_BYTES = O_ACTIVE_COLS * O_WEIGHTS_PER_COL * W_BLOCK
RN_BLOCKS_PER_COL = O_M_WAVES * ROWS
RN_W_BYTES = O_ACTIVE_COLS * RN_BLOCKS_PER_COL * W_BLOCK
O_BYTES = 256 * 768 * (4 if DIRECT_OUTPUT else 2)
R_BYTES = R_STAGE_BYTES + (ATT_BYTES if ATTENTION else 0) + (
    O_BYTES if OUTPUT_PROJECTION else 0
)
TOTAL_W_BYTES = (
    W_BYTES
    + (O_W_BYTES if OUTPUT_PROJECTION else 0)
    + (RN_W_BYTES if RESIDUAL_NORM else 0)
)
RAW_BASE = ATT_BYTES if OUTPUT_FIRST else 0
ATTENTION_BASE = 0 if OUTPUT_FIRST else R_STAGE_BYTES
INF = 9223372036854775807


def dims(count, block):
    return (
        f"[<size = {count}, stride = {block}>, "
        f"<size = {block // 512}, stride = 512>, "
        "<size = 512, stride = 1>]"
    )


def strided_dims(count, stride, block):
    return (
        f"[<size = {count}, stride = {stride}>, "
        f"<size = {block // 512}, stride = 512>, "
        "<size = 512, stride = 1>]"
    )


def projection_output_dims():
    # Each core emits a padded [32 tokens,32 columns] tile. Four joined cores
    # map to sixteen eight-token records; every fourth record is token padding.
    return (
        f"[<size = 4, stride = {4 * PAIR}>, "
        f"<size = 4, stride = {PAIR}>, "
        "<size = 8, stride = 512>, "
        "<size = 64, stride = 1>]"
    )


def attention_group_dims():
    # Gather eight row-major 4x256 BF16 attention tiles and emit the
    # [M-tile,K-tile,row,dim] order consumed by mmul<4,8,8>. Consecutive
    # physical columns are 6x8 KiB apart.
    return (
        f"[<size = {COLS}, stride = {QUERY_GROUPS * OUT_JOIN}>, "
        "<size = 32, stride = 16>, <size = 4, stride = 512>, "
        "<size = 16, stride = 1>]"
    )


def output_projection_dims():
    # Four core rows contribute 32 tokens x 32 columns each. Scatter their
    # joined 8 KiB FIFO into canonical token-major [256,768] BF16 output.
    return (
        f"[<size = {ROWS}, stride = {32 * 768 * 2}>, "
        f"<size = 32, stride = {768 * 2}>, <size = 64, stride = 1>]"
    )


def direct_output_projection_dims():
    return (
        f"[<size = {ROWS}, stride = {32 * 768 * 4}>, "
        f"<size = 8, stride = {768 * 4}>, <size = 256, stride = 1>]"
    )


def packed_attention_output_dims():
    return (
        f"[<size = {ROWS}, stride = {A_BLOCK}>, "
        "<size = 32, stride = 64>, <size = 4, stride = 16>, "
        "<size = 16, stride = 1>]"
    )


out = ["module {", "  aie.device(npu2) {"]
for col in range(COLS):
    out += [f"    %shim{col} = aie.tile({col}, 0)", f"    %mt{col} = aie.tile({col}, 1)"]
    for row in range(ROWS):
        out += [
            f"    %c{col}_{row} = aie.tile({col}, {row + 2})",
        ]
        if not PAIRED_QKV or (col % 2 == 0 and not RESIDUAL_NORM):
            out += [
                f'    %kinv{col}_{row} = aie.buffer(%c{col}_{row}) {{sym_name = "kinv{col}_{row}"}} : memref<8xf32>',
            ]
        if not PAIRED_QKV or col % 2 == 1:
            out += [
                f'    %acc{col}_{row} = aie.buffer(%c{col}_{row}) {{sym_name = "acc{col}_{row}"}} : memref<{ACC_ELEMS}xf32>',
            ]
        if PAIRED_QKV and col % 2 == 1:
            out += [
                f'    %accpair{col}_{row} = aie.buffer(%c{col}_{row}) {{sym_name = "accpair{col}_{row}"}} : memref<{ACC_ELEMS}xf32>',
            ]
        if ATTENTION and col % 2 == 1:
            out += [
                f'    %attacc0{col}_{row} = aie.buffer(%c{col}_{row}) {{sym_name = "attacc0{col}_{row}"}} : memref<{ATT_ACC}xf32>',
                f'    %attacc1{col}_{row} = aie.buffer(%c{col}_{row}) {{sym_name = "attacc1{col}_{row}"}} : memref<{ATT_ACC}xf32>',
                f'    %attstats0{col}_{row} = aie.buffer(%c{col}_{row}) {{sym_name = "attstats0{col}_{row}"}} : memref<{ATT_STATS}xf32>',
                f'    %attstats1{col}_{row} = aie.buffer(%c{col}_{row}) {{sym_name = "attstats1{col}_{row}"}} : memref<{ATT_STATS}xf32>',
                f'    %attq{col}_{row} = aie.buffer(%c{col}_{row}) {{sym_name = "attq{col}_{row}"}} : memref<{2 * OUT_TILE}xi8>',
            ]
        if OUTPUT_EXECUTION and col % 2 == 0:
            slices = range(2) if DIRECT_OUTPUT else range(1)
            for output_slice in slices:
                suffix = f"_{output_slice}" if DIRECT_OUTPUT else ""
                out += [
                    f'    %oacc{col}_{row}{suffix} = aie.buffer(%c{col}_{row}) {{sym_name = "oacc{col}_{row}{suffix}"}} : memref<{O_ACC_ELEMS}xf32>',
                ]
        if RESIDUAL_NORM and col % 2 == 0:
            for scratch in range(3):
                out += [
                    f'    %rns{col}_{row}_{scratch} = aie.buffer(%c{col}_{row}) {{sym_name = "rns{col}_{row}_{scratch}"}} : memref<4096xi8>',
                ]

for col in range(COLS):
    cores = ", ".join(f"%c{col}_{row}" for row in range(ROWS))
    weight_block = PAIR_W_BLOCK if PAIRED_QKV and col % 2 == 1 else W_BLOCK
    out += [
        f"    aie.objectfifo @wsh{col}(%shim{col}, {{%mt{col}}}, 1 : i32) : !aie.objectfifo<memref<{weight_block}xi8>>",
        f"    aie.objectfifo @wbc{col}(%mt{col}, {{{cores}}}, 1 : i32) : !aie.objectfifo<memref<{weight_block}xi8>>",
        f"    aie.objectfifo.link [@wsh{col}] -> [@wbc{col}] ([] [0])",
    ]

if DIRECT_OUTPUT:
    for row in range(ROWS):
        for pair in range(COLS // 2):
            source_col = pair * 2 + 1
            target_col = pair * 2
            name = f"ad{pair}_{row}"
            out.append(
                f"    aie.objectfifo @{name}(%c{source_col}_{row}, {{%c{target_col}_{row}}}, 3 : i32) : !aie.objectfifo<memref<{DIRECT_ATT_TILE}xi8>>"
            )

for row in range(ROWS):
    cores = ", ".join(f"%c{col}_{row}" for col in range(COLS))
    out += [
        f"    aie.objectfifo @ash{row}(%shim{row}, {{%mt{row}}}, 1 : i32) : !aie.objectfifo<memref<{A_BLOCK}xi8>>",
        f"    aie.objectfifo @abc{row}(%mt{row}, {{{cores}}}, 1 : i32) : !aie.objectfifo<memref<{A_BLOCK}xi8>>",
        f"    aie.objectfifo.link [@ash{row}] -> [@abc{row}] ([] [0])",
    ]

for col in range(COLS):
    attention_producers = ", ".join(f"@oc{col}_{row}" for row in range(ROWS))
    attention_offsets = ", ".join(str(row * OUT_TILE) for row in range(ROWS))
    for row in range(ROWS):
        out.append(
            f"    aie.objectfifo @oc{col}_{row}(%c{col}_{row}, {{%mt{col}}}, 1 : i32) : !aie.objectfifo<memref<{OUT_TILE}xi8>>"
        )
    out += [
        f"    aie.objectfifo @osh{col}(%mt{col}, {{%shim{col}}}, 1 : i32) : !aie.objectfifo<memref<{OUT_JOIN}xi8>>",
        f"    aie.objectfifo.link [{attention_producers}] -> [@osh{col}] ([{attention_offsets}] [])",
    ]

if RESIDUAL_NORM:
    for col in range(0, COLS, 2):
        for row in range(ROWS):
            out.append(
                f"    aie.objectfifo @rmc{col}_{row}(%c{col}_{row}, {{%c{col + 1}_{row}}}, 1 : i32) : !aie.objectfifo<memref<64xi8>>"
            )

out += [
    f'    func.func private @r29_w8_projection_init(memref<{A_BLOCK}xi8>, memref<{W_BLOCK}xi8>, memref<{ACC_ELEMS}xf32>) attributes {{link_with = "{OBJECT}"}}',
    f'    func.func private @r29_w8_projection_accum(memref<{A_BLOCK}xi8>, memref<{W_BLOCK}xi8>, memref<{ACC_ELEMS}xf32>) attributes {{link_with = "{OBJECT}"}}',
    f'    func.func private @r29_w8_projection_finish(memref<{ACC_ELEMS}xf32>, memref<{OUT_TILE}xi8>) attributes {{link_with = "{OBJECT}"}}',
    f'    func.func private @r29_pack_q(memref<{PAIR}xi8>, memref<{OUT_TILE}xi8>, i32) attributes {{link_with = "{OBJECT}"}}',
    f'    func.func private @r29_pack_k(memref<{PAIR}xi8>, memref<{OUT_TILE}xi8>, memref<{O_ACC_ELEMS if RESIDUAL_NORM else 8}xf32>, i32) attributes {{link_with = "{OBJECT}"}}',
    f'    func.func private @r29_pack_v(memref<{PAIR}xi8>, memref<{OUT_TILE}xi8>, i32) attributes {{link_with = "{OBJECT}"}}',
]
if PAIRED_QKV:
    out += [
        f'    func.func private @r33_w8_projection_group_pair(memref<{A_BLOCK}xi8>, memref<{PAIR_W_BLOCK}xi8>, memref<{ACC_ELEMS}xf32>, memref<{ACC_ELEMS}xf32>, i32, i32) attributes {{link_with = "r33pair.o"}}',
    ]
if ATTENTION:
    attention_finish = (
        "r32_attention_finish_pair_packed"
        if DIRECT_OUTPUT
        else "r31_attention_finish_packed"
        if OUTPUT_FIRST
        else "r30_attention_finish"
    )
    attention_object = "r32att.o" if DIRECT_OUTPUT else OUTPUT_OBJECT if OUTPUT_FIRST else OBJECT
    out += [
        f'    func.func private @r29_w8_projection_group(memref<{A_BLOCK}xi8>, memref<{W_BLOCK}xi8>, memref<{ACC_ELEMS}xf32>, i32) attributes {{link_with = "{OBJECT}"}}',
        f'    func.func private @r30_attention_init(memref<{ATT_ACC}xf32>, memref<{ATT_STATS}xf32>) attributes {{link_with = "{OBJECT}"}}',
        f'    func.func private @r30_attention_load_q(memref<{A_BLOCK}xi8>, memref<{2 * OUT_TILE}xi8>, i32) attributes {{link_with = "{OBJECT}"}}',
        f'    func.func private @r30_attention_block(memref<{2 * OUT_TILE}xi8>, memref<{A_BLOCK}xi8>, memref<{ATT_ACC}xf32>, memref<{ATT_STATS}xf32>, i32) attributes {{link_with = "{OBJECT}"}}',
    ]
    if DIRECT_OUTPUT:
        out += [
            f'    func.func private @{attention_finish}(memref<{ATT_ACC}xf32>, memref<{ATT_STATS}xf32>, memref<{ATT_ACC}xf32>, memref<{ATT_STATS}xf32>, memref<{DIRECT_ATT_TILE}xi8>) attributes {{link_with = "{attention_object}"}}',
        ]
    else:
        out += [
            f'    func.func private @{attention_finish}(memref<{ATT_ACC}xf32>, memref<{ATT_STATS}xf32>, memref<{OUT_TILE}xi8>) attributes {{link_with = "{attention_object}"}}',
        ]
if OUTPUT_EXECUTION:
    output_group = (
        "r32_output_projection_group_m8"
        if DIRECT_OUTPUT
        else "r31_output_projection_group"
    )
    output_finish = (
        "r34_output_projection_finish_pair_bf16"
        if RESIDUAL_NORM
        else "r32_output_projection_finish_pair_m8"
        if DIRECT_OUTPUT
        else "r31_output_projection_finish"
    )
    output_object = "r32out.o" if DIRECT_OUTPUT else OUTPUT_OBJECT
    out += [
        f'    func.func private @{output_group}(memref<{DIRECT_ATT_TILE if DIRECT_OUTPUT else A_BLOCK}xi8>, memref<{W_BLOCK}xi8>, memref<{O_ACC_ELEMS}xf32>, i32) attributes {{link_with = "{output_object}"}}',
    ]
    if RESIDUAL_NORM:
        out += [
            f'    func.func private @{output_finish}(memref<{O_ACC_ELEMS}xf32>, memref<{O_ACC_ELEMS}xf32>, memref<4096xi8>, memref<4096xi8>, memref<4096xi8>, i32) attributes {{link_with = "r34norm.o"}}',
            *(
                [
                    f'    func.func private @r48_stage_post_norm(memref<16384xi8>, memref<{O_ACC_ELEMS}xf32>, memref<{O_ACC_ELEMS}xf32>) attributes {{link_with = "r34norm.o"}}',
                    f'    func.func private @r34_post_residual_pre_ffn(memref<4096xi8>, memref<4096xi8>, memref<4096xi8>, memref<16384xi8>, memref<{O_ACC_ELEMS}xf32>, memref<{O_ACC_ELEMS}xf32>, memref<64xi8>, i32) attributes {{link_with = "r34norm.o"}}',
                ]
                if EXTERNAL_RESIDUAL
                else [
                    '    func.func private @r34_post_residual_pre_ffn(memref<4096xi8>, memref<4096xi8>, memref<4096xi8>, memref<16384xi8>, memref<64xi8>, i32) attributes {link_with = "r34norm.o"}'
                ]
            ),
            f'    func.func private @r34_emit_norm_half(memref<4096xi8>, memref<{OUT_TILE}xi8>, index) attributes {{link_with = "r34norm.o"}}',
            f'    func.func private @r38_relay_pre_inverse(memref<64xi8>, memref<{OUT_TILE}xi8>) attributes {{link_with = "r34norm.o"}}',
        ]
    elif DIRECT_OUTPUT:
        out += [
            f'    func.func private @{output_finish}(memref<{O_ACC_ELEMS}xf32>, memref<{O_ACC_ELEMS}xf32>, memref<{OUT_TILE}xi8>) attributes {{link_with = "{output_object}"}}',
        ]
    else:
        out += [
            f'    func.func private @{output_finish}(memref<{O_ACC_ELEMS}xf32>, memref<{OUT_TILE}xi8>) attributes {{link_with = "{output_object}"}}',
        ]


def acquire_a(row, name, indent="        "):
    return [
        f"{indent}%a{name} = aie.objectfifo.acquire @abc{row}(Consume, 1) : !aie.objectfifosubview<memref<{A_BLOCK}xi8>>",
        f"{indent}%a{name}v = aie.objectfifo.subview.access %a{name}[0] : !aie.objectfifosubview<memref<{A_BLOCK}xi8>> -> memref<{A_BLOCK}xi8>",
    ]


def acquire_w(col, name, indent="        "):
    block = PAIR_W_BLOCK if PAIRED_QKV and col % 2 == 1 else W_BLOCK
    return [
        f"{indent}%w{name} = aie.objectfifo.acquire @wbc{col}(Consume, 1) : !aie.objectfifosubview<memref<{block}xi8>>",
        f"{indent}%w{name}v = aie.objectfifo.subview.access %w{name}[0] : !aie.objectfifosubview<memref<{block}xi8>> -> memref<{block}xi8>",
    ]


def acquire_out(col, row, name, indent="        "):
    return [
        f"{indent}%{name} = aie.objectfifo.acquire @oc{col}_{row}(Produce, 1) : !aie.objectfifosubview<memref<{OUT_TILE}xi8>>",
        f"{indent}%{name}v = aie.objectfifo.subview.access %{name}[0] : !aie.objectfifosubview<memref<{OUT_TILE}xi8>> -> memref<{OUT_TILE}xi8>",
    ]


def acquire_direct_attention(pair, row, name, indent="        "):
    return [
        f"{indent}%{name} = aie.objectfifo.acquire @ad{pair}_{row}(Produce, 1) : !aie.objectfifosubview<memref<{DIRECT_ATT_TILE}xi8>>",
        f"{indent}%{name}v = aie.objectfifo.subview.access %{name}[0] : !aie.objectfifosubview<memref<{DIRECT_ATT_TILE}xi8>> -> memref<{DIRECT_ATT_TILE}xi8>",
    ]


for col in range(COLS):
    for row in range(ROWS):
        lines = [
            f"    %core{col}_{row} = aie.core(%c{col}_{row}) {{",
            "      %z = arith.constant 0 : index",
            f"      %inf = arith.constant {INF} : index",
            "      %one = arith.constant 1 : index",
            f"      %groups = arith.constant {GROUPS} : index",
            f"      %outblocks = arith.constant {OUTBLOCKS} : index",
            f"      %qgroups = arith.constant {QUERY_GROUPS} : index",
            "      %attblocks = arith.constant 16 : index",
            "      %waves = arith.constant 2 : index",
            f"      %oslices = arith.constant {O_SLICES} : index",
            f"      %ogroups = arith.constant {O_GROUPS} : index",
            f"      %omwaves = arith.constant {O_M_WAVES} : index",
            *(
                [
                    f"      %rnrows = arith.constant {ROWS} : index",
                    f"      %rnrow = arith.constant {row} : index",
                ]
                if RESIDUAL_NORM
                else []
            ),
            f"      %odrops = arith.constant {O_M_WAVES * O_SLICES * O_GROUPS} : index",
            f"      %lane = arith.constant {col % 2} : i32",
            f"      %corecol = arith.constant {col // 2} : i32",
            "      %h0 = arith.constant 0 : i32",
            "      %h1 = arith.constant 1 : i32",
            "      scf.for %outer = %z to %inf step %one {",
        ]
        if PAIRED_QKV:
            lines += ["        scf.for %block = %z to %outblocks step %one {"]
            if col % 2 == 1:
                lines += acquire_a(row, "pp", "          ")
                lines += acquire_w(col, "pp", "          ")
                lines += [
                    f"          func.call @r33_w8_projection_group_pair(%appv, %wppv, %acc{col}_{row}, %accpair{col}_{row}, %corecol, %h0) : (memref<{A_BLOCK}xi8>, memref<{PAIR_W_BLOCK}xi8>, memref<{ACC_ELEMS}xf32>, memref<{ACC_ELEMS}xf32>, i32, i32) -> ()",
                    f"          aie.objectfifo.release @abc{row}(Consume, 1)",
                    f"          aie.objectfifo.release @wbc{col}(Consume, 1)",
                    "          scf.for %group = %one to %groups step %one {",
                ]
                lines += acquire_a(row, "ppa", "            ")
                lines += acquire_w(col, "ppa", "            ")
                lines += [
                    f"            func.call @r33_w8_projection_group_pair(%appav, %wppav, %acc{col}_{row}, %accpair{col}_{row}, %corecol, %h1) : (memref<{A_BLOCK}xi8>, memref<{PAIR_W_BLOCK}xi8>, memref<{ACC_ELEMS}xf32>, memref<{ACC_ELEMS}xf32>, i32, i32) -> ()",
                    f"            aie.objectfifo.release @abc{row}(Consume, 1)",
                    f"            aie.objectfifo.release @wbc{col}(Consume, 1)",
                    "          }",
                ]
                lines += acquire_out(col, row, "ppo0", "          ")
                lines += [
                    f"          func.call @r29_w8_projection_finish(%acc{col}_{row}, %ppo0v) : (memref<{ACC_ELEMS}xf32>, memref<{OUT_TILE}xi8>) -> ()",
                    f"          aie.objectfifo.release @oc{col}_{row}(Produce, 1)",
                ]
                lines += acquire_out(col, row, "ppo1", "          ")
                lines += [
                    f"          func.call @r29_w8_projection_finish(%accpair{col}_{row}, %ppo1v) : (memref<{ACC_ELEMS}xf32>, memref<{OUT_TILE}xi8>) -> ()",
                    f"          aie.objectfifo.release @oc{col}_{row}(Produce, 1)",
                ]
            else:
                lines += ["          scf.for %group = %z to %groups step %one {"]
                lines += acquire_a(row, "ppdrop", "            ")
                lines += [
                    f"            aie.objectfifo.release @abc{row}(Consume, 1)",
                    "          }",
                ]
            lines += ["        }", "        scf.for %qgroup = %z to %qgroups step %one {"]
            for pair in range(COLS // 2):
                name = f"q{pair}"
                lines += acquire_a(row, name, "          ")
                if col % 2 == 1 and col // 2 == pair:
                    lines += acquire_out(col, row, "qpo0", "          ")
                    lines += [
                        f"          func.call @r29_pack_q(%a{name}v, %qpo0v, %h0) : (memref<{PAIR}xi8>, memref<{OUT_TILE}xi8>, i32) -> ()",
                        f"          aie.objectfifo.release @oc{col}_{row}(Produce, 1)",
                    ]
                    lines += acquire_out(col, row, "qpo1", "          ")
                    lines += [
                        f"          func.call @r29_pack_q(%a{name}v, %qpo1v, %h1) : (memref<{PAIR}xi8>, memref<{OUT_TILE}xi8>, i32) -> ()",
                        f"          aie.objectfifo.release @oc{col}_{row}(Produce, 1)",
                    ]
                lines.append(f"          aie.objectfifo.release @abc{row}(Consume, 1)")
            lines += ["        }"]
        else:
            lines += ["        scf.for %block = %z to %outblocks step %one {"]
            lines += acquire_a(row, "p0", "          ")
            lines += acquire_w(col, "p0", "          ")
            first_projection = (
                f"          func.call @r29_w8_projection_group(%ap0v, %wp0v, %acc{col}_{row}, %h0) : (memref<{A_BLOCK}xi8>, memref<{W_BLOCK}xi8>, memref<{ACC_ELEMS}xf32>, i32) -> ()"
                if ATTENTION
                else f"          func.call @r29_w8_projection_init(%ap0v, %wp0v, %acc{col}_{row}) : (memref<{A_BLOCK}xi8>, memref<{W_BLOCK}xi8>, memref<{ACC_ELEMS}xf32>) -> ()"
            )
            lines += [first_projection, f"          aie.objectfifo.release @abc{row}(Consume, 1)", f"          aie.objectfifo.release @wbc{col}(Consume, 1)", "          scf.for %group = %one to %groups step %one {"]
            lines += acquire_a(row, "pa", "            ")
            lines += acquire_w(col, "pa", "            ")
            accumulated_projection = (
                f"            func.call @r29_w8_projection_group(%apav, %wpav, %acc{col}_{row}, %h1) : (memref<{A_BLOCK}xi8>, memref<{W_BLOCK}xi8>, memref<{ACC_ELEMS}xf32>, i32) -> ()"
                if ATTENTION
                else f"            func.call @r29_w8_projection_accum(%apav, %wpav, %acc{col}_{row}) : (memref<{A_BLOCK}xi8>, memref<{W_BLOCK}xi8>, memref<{ACC_ELEMS}xf32>) -> ()"
            )
            lines += [accumulated_projection, f"            aie.objectfifo.release @abc{row}(Consume, 1)", f"            aie.objectfifo.release @wbc{col}(Consume, 1)", "          }"]
            lines += acquire_out(col, row, "po", "          ")
            lines += [f"          func.call @r29_w8_projection_finish(%acc{col}_{row}, %pov) : (memref<{ACC_ELEMS}xf32>, memref<{OUT_TILE}xi8>) -> ()", f"          aie.objectfifo.release @oc{col}_{row}(Produce, 1)", "        }", "        scf.for %qgroup = %z to %qgroups step %one {"]
            lines += acquire_out(col, row, "qo", "          ")
            for pair in range(COLS // 2):
                name = f"q{pair}"
                lines += acquire_a(row, name, "          ")
                if col // 2 == pair:
                    lines.append(f"          func.call @r29_pack_q(%a{name}v, %qov, %lane) : (memref<{PAIR}xi8>, memref<{OUT_TILE}xi8>, i32) -> ()")
                lines.append(f"          aie.objectfifo.release @abc{row}(Consume, 1)")
            lines += [f"          aie.objectfifo.release @oc{col}_{row}(Produce, 1)", "        }"]
        for phase in ("k", "v"):
            lines.append(f"        scf.for %{phase}wave = %z to %waves step %one {{")
            for pair in range(COLS // 2):
                name = f"{phase}{pair}"
                lines += acquire_a(row, name, "          ")
                if col % 2 == 0 and col // 2 == pair:
                    lines += acquire_out(col, row, f"{phase}o0", "          ")
                    if phase == "k":
                        inverse_buffer = (
                            f"%oacc{col}_{row}_0"
                            if RESIDUAL_NORM
                            else f"%kinv{col}_{row}"
                        )
                        lines.append(
                            f"          func.call @r29_pack_k(%a{name}v, %{phase}o0v, {inverse_buffer}, %h0) : (memref<{PAIR}xi8>, memref<{OUT_TILE}xi8>, memref<{O_ACC_ELEMS if RESIDUAL_NORM else 8}xf32>, i32) -> ()"
                        )
                    else:
                        lines.append(
                            f"          func.call @r29_pack_v(%a{name}v, %{phase}o0v, %h0) : (memref<{PAIR}xi8>, memref<{OUT_TILE}xi8>, i32) -> ()"
                        )
                    lines.append(f"          aie.objectfifo.release @oc{col}_{row}(Produce, 1)")
                    lines += acquire_out(col, row, f"{phase}o1", "          ")
                    if phase == "k":
                        lines.append(
                            f"          func.call @r29_pack_k(%a{name}v, %{phase}o1v, {inverse_buffer}, %h1) : (memref<{PAIR}xi8>, memref<{OUT_TILE}xi8>, memref<{O_ACC_ELEMS if RESIDUAL_NORM else 8}xf32>, i32) -> ()"
                        )
                    else:
                        lines.append(
                            f"          func.call @r29_pack_v(%a{name}v, %{phase}o1v, %h1) : (memref<{PAIR}xi8>, memref<{OUT_TILE}xi8>, i32) -> ()"
                        )
                    lines.append(f"          aie.objectfifo.release @oc{col}_{row}(Produce, 1)")
                lines.append(f"          aie.objectfifo.release @abc{row}(Consume, 1)")
            lines.append("        }")
        if ATTENTION and col % 2 == 1:
            lines += [
                "        scf.for %agroup = %z to %qgroups step %one {",
            ]
            lines += acquire_a(row, "attq", "          ")
            lines += [
                f"          func.call @r30_attention_load_q(%aattqv, %attq{col}_{row}, %corecol) : (memref<{A_BLOCK}xi8>, memref<{2 * OUT_TILE}xi8>, i32) -> ()",
                f"          aie.objectfifo.release @abc{row}(Consume, 1)",
                f"          func.call @r30_attention_init(%attacc0{col}_{row}, %attstats0{col}_{row}) : (memref<{ATT_ACC}xf32>, memref<{ATT_STATS}xf32>) -> ()",
                f"          func.call @r30_attention_init(%attacc1{col}_{row}, %attstats1{col}_{row}) : (memref<{ATT_ACC}xf32>, memref<{ATT_STATS}xf32>) -> ()",
                "          scf.for %attblock = %z to %attblocks step %one {",
            ]
            lines += acquire_a(row, "attkv", "            ")
            lines += [
                f"            func.call @r30_attention_block(%attq{col}_{row}, %aattkvv, %attacc0{col}_{row}, %attstats0{col}_{row}, %h0) : (memref<{2 * OUT_TILE}xi8>, memref<{A_BLOCK}xi8>, memref<{ATT_ACC}xf32>, memref<{ATT_STATS}xf32>, i32) -> ()",
                f"            func.call @r30_attention_block(%attq{col}_{row}, %aattkvv, %attacc1{col}_{row}, %attstats1{col}_{row}, %h1) : (memref<{2 * OUT_TILE}xi8>, memref<{A_BLOCK}xi8>, memref<{ATT_ACC}xf32>, memref<{ATT_STATS}xf32>, i32) -> ()",
                f"            aie.objectfifo.release @abc{row}(Consume, 1)",
                "          }",
            ]
            if DIRECT_OUTPUT:
                lines += acquire_direct_attention(col // 2, row, "attpair", "          ")
                lines += [
                    f"          func.call @{attention_finish}(%attacc0{col}_{row}, %attstats0{col}_{row}, %attacc1{col}_{row}, %attstats1{col}_{row}, %attpairv) : (memref<{ATT_ACC}xf32>, memref<{ATT_STATS}xf32>, memref<{ATT_ACC}xf32>, memref<{ATT_STATS}xf32>, memref<{DIRECT_ATT_TILE}xi8>) -> ()",
                    f"          aie.objectfifo.release @ad{col // 2}_{row}(Produce, 1)",
                    "        }",
                ]
            else:
                lines += acquire_out(col, row, "atto0", "          ")
                lines += [
                    f"          func.call @{attention_finish}(%attacc0{col}_{row}, %attstats0{col}_{row}, %atto0v) : (memref<{ATT_ACC}xf32>, memref<{ATT_STATS}xf32>, memref<{OUT_TILE}xi8>) -> ()",
                    f"          aie.objectfifo.release @oc{col}_{row}(Produce, 1)",
                ]
                lines += acquire_out(col, row, "atto1", "          ")
                lines += [
                    f"          func.call @{attention_finish}(%attacc1{col}_{row}, %attstats1{col}_{row}, %atto1v) : (memref<{ATT_ACC}xf32>, memref<{ATT_STATS}xf32>, memref<{OUT_TILE}xi8>) -> ()",
                    f"          aie.objectfifo.release @oc{col}_{row}(Produce, 1)",
                    "        }",
                ]
        elif ATTENTION:
            if DIRECT_OUTPUT:
                if RESIDUAL_NORM:
                    lines += [
                        f"        %rnmeta = aie.objectfifo.acquire @rmc{col}_{row}(Produce, 1) : !aie.objectfifosubview<memref<64xi8>>",
                        f"        %rnmetav = aie.objectfifo.subview.access %rnmeta[0] : !aie.objectfifosubview<memref<64xi8>> -> memref<64xi8>",
                    ]
                lines += [
                    "        scf.for %omwave = %z to %omwaves step %one {",
                    "          scf.for %ogroup = %z to %ogroups step %one {",
                ]
                lines += acquire_a(row, "attqdrop", "            ")
                lines += [
                    f"            aie.objectfifo.release @abc{row}(Consume, 1)",
                    "            scf.for %attblock = %z to %attblocks step %one {",
                ]
                lines += acquire_a(row, "attkvdrop", "              ")
                lines += [
                    f"              aie.objectfifo.release @abc{row}(Consume, 1)",
                    "            }",
                    "          }",
                    f"          %oppair = aie.objectfifo.acquire @ad{col // 2}_{row}(Consume, 3) : !aie.objectfifosubview<memref<{DIRECT_ATT_TILE}xi8>>",
                    f"          %opairs = arith.constant {O_SLICES // 2} : index",
                    "          scf.for %opair = %z to %opairs step %one {",
                ]
                for local_slice in range(2):
                    for group in range(O_GROUPS):
                        lines += [
                            f"            %opa{local_slice}_{group} = aie.objectfifo.subview.access %oppair[{group}] : !aie.objectfifosubview<memref<{DIRECT_ATT_TILE}xi8>> -> memref<{DIRECT_ATT_TILE}xi8>",
                            f"            %wop{local_slice}_{group} = aie.objectfifo.acquire @wbc{col}(Consume, 1) : !aie.objectfifosubview<memref<{W_BLOCK}xi8>>",
                            f"            %wop{local_slice}_{group}v = aie.objectfifo.subview.access %wop{local_slice}_{group}[0] : !aie.objectfifosubview<memref<{W_BLOCK}xi8>> -> memref<{W_BLOCK}xi8>",
                            f"            func.call @{output_group}(%opa{local_slice}_{group}, %wop{local_slice}_{group}v, %oacc{col}_{row}_{local_slice}, {'%h0' if group == 0 else '%h1'}) : (memref<{DIRECT_ATT_TILE}xi8>, memref<{W_BLOCK}xi8>, memref<{O_ACC_ELEMS}xf32>, i32) -> ()",
                            f"            aie.objectfifo.release @wbc{col}(Consume, 1)",
                        ]
                if RESIDUAL_NORM:
                    lines += [
                        "            %opairi = arith.index_cast %opair : index to i32",
                        f"            func.call @{output_finish}(%oacc{col}_{row}_0, %oacc{col}_{row}_1, %rns{col}_{row}_0, %rns{col}_{row}_1, %rns{col}_{row}_2, %opairi) : (memref<{O_ACC_ELEMS}xf32>, memref<{O_ACC_ELEMS}xf32>, memref<4096xi8>, memref<4096xi8>, memref<4096xi8>, i32) -> ()",
                        "          }",
                        f"          aie.objectfifo.release @ad{col // 2}_{row}(Consume, 3)",
                    ]
                    lines += [
                        "          %omwavei = arith.index_cast %omwave : index to i32",
                    ]
                    if EXTERNAL_RESIDUAL:
                        lines += [
                            f"          %wrnp = aie.objectfifo.acquire @wbc{col}(Consume, 1) : !aie.objectfifosubview<memref<16384xi8>>",
                            "          %wrnpv = aie.objectfifo.subview.access %wrnp[0] : !aie.objectfifosubview<memref<16384xi8>> -> memref<16384xi8>",
                            f"          func.call @r48_stage_post_norm(%wrnpv, %oacc{col}_{row}_0, %oacc{col}_{row}_1) : (memref<16384xi8>, memref<{O_ACC_ELEMS}xf32>, memref<{O_ACC_ELEMS}xf32>) -> ()",
                            f"          aie.objectfifo.release @wbc{col}(Consume, 1)",
                            "          scf.for %rnparam = %z to %rnrows step %one {",
                        ]
                        lines += acquire_w(col, "rn", "            ")
                        lines += [
                            "            %rnactive = arith.cmpi eq, %rnparam, %rnrow : index",
                            "            scf.if %rnactive {",
                            f"              func.call @r34_post_residual_pre_ffn(%rns{col}_{row}_0, %rns{col}_{row}_1, %rns{col}_{row}_2, %wrnv, %oacc{col}_{row}_0, %oacc{col}_{row}_1, %rnmetav, %omwavei) : (memref<4096xi8>, memref<4096xi8>, memref<4096xi8>, memref<16384xi8>, memref<{O_ACC_ELEMS}xf32>, memref<{O_ACC_ELEMS}xf32>, memref<64xi8>, i32) -> ()",
                            "            }",
                            f"            aie.objectfifo.release @wbc{col}(Consume, 1)",
                            "          }",
                        ]
                    else:
                        lines += ["          scf.for %rnparam = %z to %rnrows step %one {"]
                        lines += acquire_w(col, "rn", "            ")
                        lines += [
                            "            %rnactive = arith.cmpi eq, %rnparam, %rnrow : index",
                            "            scf.if %rnactive {",
                            f"              func.call @r34_post_residual_pre_ffn(%rns{col}_{row}_0, %rns{col}_{row}_1, %rns{col}_{row}_2, %wrnv, %rnmetav, %omwavei) : (memref<4096xi8>, memref<4096xi8>, memref<4096xi8>, memref<16384xi8>, memref<64xi8>, i32) -> ()",
                            "            }",
                            f"            aie.objectfifo.release @wbc{col}(Consume, 1)",
                            "          }",
                        ]
                    for block in range(3):
                        name = f"rne{block}"
                        lines += ["          scf.for %rnhalf = %z to %waves step %one {"]
                        lines += acquire_out(col, row, name, "            ")
                        lines += [
                            f"            func.call @r34_emit_norm_half(%rns{col}_{row}_{block}, %{name}v, %rnhalf) : (memref<4096xi8>, memref<{OUT_TILE}xi8>, index) -> ()",
                            f"            aie.objectfifo.release @oc{col}_{row}(Produce, 1)",
                            "          }",
                        ]
                    lines += [
                        "        }",
                        f"        aie.objectfifo.release @rmc{col}_{row}(Produce, 1)",
                    ]
                else:
                    lines += acquire_out(col, row, "opo", "            ")
                    lines += [
                        f"            func.call @{output_finish}(%oacc{col}_{row}_0, %oacc{col}_{row}_1, %opov) : (memref<{O_ACC_ELEMS}xf32>, memref<{O_ACC_ELEMS}xf32>, memref<{OUT_TILE}xi8>) -> ()",
                        f"            aie.objectfifo.release @oc{col}_{row}(Produce, 1)",
                        "          }",
                        f"          aie.objectfifo.release @ad{col // 2}_{row}(Consume, 3)",
                        "        }",
                    ]
            else:
                lines += ["        scf.for %agroup = %z to %qgroups step %one {"]
                lines += acquire_a(row, "attqdrop", "          ")
                lines += [
                    f"          aie.objectfifo.release @abc{row}(Consume, 1)",
                    "          scf.for %attblock = %z to %attblocks step %one {",
                ]
                lines += acquire_a(row, "attkvdrop", "            ")
                lines += [
                    f"            aie.objectfifo.release @abc{row}(Consume, 1)",
                    "          }",
                    "        }",
                ]
        if RESIDUAL_NORM and col % 2 == 1:
            lines += [
                f"        %rnmetai = aie.objectfifo.acquire @rmc{col - 1}_{row}(Consume, 1) : !aie.objectfifosubview<memref<64xi8>>",
                f"        %rnmetaiv = aie.objectfifo.subview.access %rnmetai[0] : !aie.objectfifosubview<memref<64xi8>> -> memref<64xi8>",
            ]
            lines += acquire_out(col, row, "rnmetao", "        ")
            lines += [
                f"        func.call @r38_relay_pre_inverse(%rnmetaiv, %rnmetaov) : (memref<64xi8>, memref<{OUT_TILE}xi8>) -> ()",
                f"        aie.objectfifo.release @rmc{col - 1}_{row}(Consume, 1)",
                f"        aie.objectfifo.release @oc{col}_{row}(Produce, 1)",
            ]
        if OUTPUT_EXECUTION and not DIRECT_OUTPUT:
            if col % 2 == 0:
                lines += [
                    "        scf.for %omwave = %z to %omwaves step %one {",
                    "          scf.for %oslice = %z to %oslices step %one {",
                    "            scf.for %ogroup = %z to %ogroups step %one {",
                    "              %ogroupi = arith.index_cast %ogroup : index to i32",
                ]
                lines += acquire_a(row, "op", "              ")
                lines += acquire_w(col, "op", "              ")
                lines += [
                    f"              func.call @r31_output_projection_group(%aopv, %wopv, %oacc{col}_{row}, %ogroupi) : (memref<{A_BLOCK}xi8>, memref<{W_BLOCK}xi8>, memref<{O_ACC_ELEMS}xf32>, i32) -> ()",
                    f"              aie.objectfifo.release @wbc{col}(Consume, 1)",
                    f"              aie.objectfifo.release @abc{row}(Consume, 1)",
                    "            }",
                ]
                lines += acquire_out(col, row, "opo", "            ")
                lines += [
                    f"            func.call @r31_output_projection_finish(%oacc{col}_{row}, %opov) : (memref<{O_ACC_ELEMS}xf32>, memref<{OUT_TILE}xi8>) -> ()",
                    f"            aie.objectfifo.release @oc{col}_{row}(Produce, 1)",
                    "          }",
                    "        }",
                ]
            else:
                lines += ["        scf.for %odrop = %z to %odrops step %one {"]
                lines += acquire_a(row, "odrop", "          ")
                lines += [
                    f"          aie.objectfifo.release @abc{row}(Consume, 1)",
                    "        }",
                ]
        lines += [
            "      }",
            "      aie.end",
            f"    }} {{stack_size = {2048 if RESIDUAL_NORM else 4096} : i32}}",
        ]
        out += lines

runtime_args = f"%A: memref<{A_BYTES}xi8>, %W: memref<{TOTAL_W_BYTES}xi8>, %R: memref<{R_BYTES}xi8>, %Q: memref<{Q_BYTES}xi8>, %KV: memref<{KV_BYTES}xi8>"
out.append(f"    aie.runtime_sequence({runtime_args}) {{")

for row in range(ROWS):
    out += [
        f"      %ta{row} = aiex.dma_configure_task_for @ash{row} {{",
        f"        aie.dma_bd(%A : memref<{A_BYTES}xi8>, {row * INBLOCKS * A_BLOCK}, {INBLOCKS * A_BLOCK}, {dims(INBLOCKS, A_BLOCK)}) {{burst_length = 0 : i32}}",
        "        aie.end",
        "      }",
        f"      aiex.dma_start_task(%ta{row})",
    ]
weight_task_cols = range(1, COLS, 2) if PAIRED_QKV else range(COLS)
for col in weight_task_cols:
    pair = col // 2
    block_size = PAIR_W_BLOCK if PAIRED_QKV else W_BLOCK
    offset = pair * INBLOCKS * PAIR_W_BLOCK if PAIRED_QKV else col * INBLOCKS * W_BLOCK
    blocks = INBLOCKS
    out += [
        f"      %tw{col} = aiex.dma_configure_task_for @wsh{col} {{",
        f"        aie.dma_bd(%W : memref<{TOTAL_W_BYTES}xi8>, {offset}, {blocks * block_size}, {dims(blocks, block_size)}) {{burst_length = 0 : i32}}",
        "        aie.end",
        "      }",
        f"      aiex.dma_start_task(%tw{col})",
    ]

for outblock in range(OUTBLOCKS):
    m_macro, n_macro = divmod(outblock, N_MACROS)
    if PAIRED_QKV:
        for source_col in range(1, COLS, 2):
            for lane in range(2):
                target_col = source_col - 1 + lane
                offset = RAW_BASE + (n_macro * PAIRS_PER_ROLE + m_macro * 16) * PAIR + target_col * 64
                name = f"tpo{outblock}_{source_col}_{lane}"
                out += [
                    f"      %{name} = aiex.dma_configure_task_for @osh{source_col} {{",
                    f"        aie.dma_bd(%R : memref<{R_BYTES}xi8>, {offset}, {OUT_JOIN // 4}, {projection_output_dims()}) {{burst_length = 0 : i32}}",
                    "        aie.end",
                    "      } {issue_token = true, repeat_count = 3 : i32}",
                    f"      aiex.dma_start_task(%{name})",
                ]
        for source_col in range(1, COLS, 2):
            for lane in range(2):
                name = f"tpo{outblock}_{source_col}_{lane}"
                out += [f"      aiex.dma_await_task(%{name})", f"      aiex.dma_free_task(%{name})"]
    else:
        for col in range(COLS):
            offset = RAW_BASE + (n_macro * PAIRS_PER_ROLE + m_macro * 16) * PAIR + col * 64
            name = f"tpo{outblock}_{col}"
            out += [
                f"      %{name} = aiex.dma_configure_task_for @osh{col} {{",
                f"        aie.dma_bd(%R : memref<{R_BYTES}xi8>, {offset}, {OUT_JOIN // 4}, {projection_output_dims()}) {{burst_length = 0 : i32}}",
                "        aie.end",
                "      } {issue_token = true, repeat_count = 3 : i32}",
                f"      aiex.dma_start_task(%{name})",
            ]
        for col in range(COLS):
            name = f"tpo{outblock}_{col}"
            out += [f"      aiex.dma_await_task(%{name})", f"      aiex.dma_free_task(%{name})"]

for row in range(ROWS):
    out.append(f"      aiex.dma_free_task(%ta{row})")
for col in weight_task_cols:
    out.append(f"      aiex.dma_free_task(%tw{col})")


def emit_raw_inputs(role, base_pair, stem):
    for row in range(ROWS):
        for pair in range(COLS // 2):
            logical_pair = base_pair + row * 4 + pair
            token = logical_pair * 8
            m_macro, within_macro = divmod(token, 96)
            core_row, within_core = divmod(within_macro, 24)
            pair_index = m_macro * 16 + core_row * 4 + within_core // 8
            name = f"t{stem}i{row}_{pair}"
            offset = RAW_BASE + (role * PAIRS_PER_ROLE + pair_index) * PAIR
            out.extend(
                [
                    f"      %{name} = aiex.dma_configure_task_for @ash{row} {{",
                    f"        aie.dma_bd(%R : memref<{R_BYTES}xi8>, {offset}, {PAIR}, {dims(1, PAIR)}) {{burst_length = 0 : i32}}",
                    "        aie.end",
                    "      } {issue_token = true}",
                    f"      aiex.dma_start_task(%{name})",
                ]
            )


def await_raw_inputs(stem):
    for row in range(ROWS):
        for pair in range(COLS // 2):
            name = f"t{stem}i{row}_{pair}"
            out.extend([f"      aiex.dma_await_task(%{name})", f"      aiex.dma_free_task(%{name})"])


for group in range(QUERY_GROUPS):
    if PAIRED_QKV:
        for source_col in range(1, COLS, 2):
            for lane in range(2):
                target_col = source_col - 1 + lane
                offset = group * Q_JOIN + target_col * OUT_TILE
                name = f"tqo{group}_{source_col}_{lane}"
                out += [
                    f"      %{name} = aiex.dma_configure_task_for @osh{source_col} {{",
                    f"        aie.dma_bd(%Q : memref<{Q_BYTES}xi8>, {offset}, {ROWS * OUT_TILE}, {strided_dims(ROWS, QUERY_GROUPS * Q_JOIN, OUT_TILE)}) {{burst_length = 0 : i32}}",
                    "        aie.end",
                    "      } {issue_token = true}",
                    f"      aiex.dma_start_task(%{name})",
                ]
    else:
        for col in range(COLS):
            offset = group * Q_JOIN + col * OUT_TILE
            name = f"tqo{group}_{col}"
            out += [
                f"      %{name} = aiex.dma_configure_task_for @osh{col} {{",
                f"        aie.dma_bd(%Q : memref<{Q_BYTES}xi8>, {offset}, {ROWS * OUT_TILE}, {strided_dims(ROWS, QUERY_GROUPS * Q_JOIN, OUT_TILE)}) {{burst_length = 0 : i32}}",
                "        aie.end",
                "      } {issue_token = true}",
                f"      aiex.dma_start_task(%{name})",
            ]
    role, half = divmod(group, 2)
    emit_raw_inputs(role, half * 16, f"q{group}")
    await_raw_inputs(f"q{group}")
    if PAIRED_QKV:
        for source_col in range(1, COLS, 2):
            for lane in range(2):
                name = f"tqo{group}_{source_col}_{lane}"
                out += [f"      aiex.dma_await_task(%{name})", f"      aiex.dma_free_task(%{name})"]
    else:
        for col in range(COLS):
            name = f"tqo{group}_{col}"
            out += [f"      aiex.dma_await_task(%{name})", f"      aiex.dma_free_task(%{name})"]

for phase, role in (("k", 3), ("v", 4)):
    for wave in range(2):
        for col in range(0, COLS, 2):
            pair = col // 2
            group = wave * 16 + pair
            block, key_tile = divmod(group, 2)
            for half in range(2):
                if phase == "k":
                    offset = block * KV_TILE + key_tile * 4096 + half * OUT_TILE
                    dimensions = strided_dims(ROWS, 2 * KV_TILE, OUT_TILE)
                else:
                    offset = block * KV_TILE + K_HALF + (half * 32 + key_tile) * 128
                    dimensions = (
                        f"[<size = {ROWS}, stride = {2 * KV_TILE}>, "
                        "<size = 16, stride = 256>, <size = 128, stride = 1>]"
                    )
                name = f"t{phase}o{wave}_{col}_{half}"
                out += [
                    f"      %{name} = aiex.dma_configure_task_for @osh{col} {{",
                    f"        aie.dma_bd(%KV : memref<{KV_BYTES}xi8>, {offset}, {ROWS * OUT_TILE}, {dimensions}) {{burst_length = 0 : i32}}",
                    "        aie.end",
                    "      } {issue_token = true}",
                    f"      aiex.dma_start_task(%{name})",
                ]
        emit_raw_inputs(role, wave * 16, f"{phase}{wave}")
        await_raw_inputs(f"{phase}{wave}")
        for col in range(0, COLS, 2):
            for half in range(2):
                name = f"t{phase}o{wave}_{col}_{half}"
                out += [f"      aiex.dma_await_task(%{name})", f"      aiex.dma_free_task(%{name})"]


def start_direct_output_tasks(mwave, first_pair, pair_count):
    for output_pair in range(first_pair, first_pair + pair_count):
        for active_col, col in enumerate(range(0, COLS, 2)):
            name = f"tdo{mwave}_{output_pair}_{col}"
            offset = (
                mwave * 128 * 768 * 4
                + active_col * 8 * 768 * 4
                + output_pair * 64 * 4
            )
            out.extend(
                [
                    f"      %{name} = aiex.dma_configure_task_for @osh{col} {{",
                    f"        aie.dma_bd(%R : memref<{R_BYTES}xi8>, {R_STAGE_BYTES + ATT_BYTES + offset}, {OUT_JOIN}, {direct_output_projection_dims()}) {{burst_length = 0 : i32}}",
                    "        aie.end",
                    "      } {issue_token = true}",
                    f"      aiex.dma_start_task(%{name})",
                ]
            )


def await_direct_output_tasks(mwave, first_pair, pair_count):
    for output_pair in range(first_pair, first_pair + pair_count):
        for col in range(0, COLS, 2):
            name = f"tdo{mwave}_{output_pair}_{col}"
            out.extend(
                [
                    f"      aiex.dma_await_task(%{name})",
                    f"      aiex.dma_free_task(%{name})",
                ]
            )


def start_norm_output_tasks(mwave):
    for active_col, col in enumerate(range(0, COLS, 2)):
        if mwave == O_M_WAVES - 1:
            name = f"trnm{col}"
            offset = 256 * 768 * 2 + active_col * 12288
            out.extend(
                [
                    f"      %{name} = aiex.dma_configure_task_for @osh{col + 1} {{",
                    f"        aie.dma_bd(%R : memref<{R_BYTES}xi8>, {offset}, {OUT_JOIN}, [<size = {ROWS}, stride = {8 * 12288}>, <size = {O_M_WAVES}, stride = {4 * 12288}>, <size = 1024, stride = 1>]) {{burst_length = 0 : i32}}",
                    "        aie.end",
                    "      } {issue_token = true}",
                    f"      aiex.dma_start_task(%{name})",
                ]
            )
        for block in range(3):
            for half in range(2):
                name = f"trno{mwave}_{col}_{block}_{half}"
                offset = (
                    mwave * 128 * 768 * 2
                    + active_col * 8 * 768 * 2
                    + block * 256 * 2
                    + half * 4 * 768 * 2
                )
                out.extend(
                    [
                        f"      %{name} = aiex.dma_configure_task_for @osh{col} {{",
                        f"        aie.dma_bd(%R : memref<{R_BYTES}xi8>, {offset}, {OUT_JOIN}, [<size = {ROWS}, stride = {32 * 768 * 2}>, <size = 4, stride = {768 * 2}>, <size = 512, stride = 1>]) {{burst_length = 0 : i32}}",
                        "        aie.end",
                        "      } {issue_token = true}",
                        f"      aiex.dma_start_task(%{name})",
                    ]
                )


def await_norm_output_tasks(mwave):
    for col in range(0, COLS, 2):
        if mwave == O_M_WAVES - 1:
            name = f"trnm{col}"
            out.extend(
                [
                    f"      aiex.dma_await_task(%{name})",
                    f"      aiex.dma_free_task(%{name})",
                ]
            )
        for block in range(3):
            for half in range(2):
                name = f"trno{mwave}_{col}_{block}_{half}"
                out.extend(
                    [
                        f"      aiex.dma_await_task(%{name})",
                        f"      aiex.dma_free_task(%{name})",
                    ]
                )


def start_external_norm_inputs(mwave, include_weights):
    for active_col, col in enumerate(range(0, COLS, 2)):
        output_offset = W_BYTES + active_col * O_WEIGHTS_PER_COL * W_BLOCK
        if include_weights:
            weight_name = f"tow{mwave}_{col}"
            out.extend(
                [
                    f"      %{weight_name} = aiex.dma_configure_task_for @wsh{col} {{",
                    f"        aie.dma_bd(%W : memref<{TOTAL_W_BYTES}xi8>, {output_offset}, {O_WEIGHTS_PER_COL * W_BLOCK}, {dims(O_WEIGHTS_PER_COL, W_BLOCK)}) {{burst_length = 0 : i32}}",
                    "        aie.end",
                    "      } {issue_token = true}",
                    f"      aiex.dma_start_task(%{weight_name})",
                ]
            )
        params_name = f"trnp{mwave}_{col}"
        params_offset = (
            W_BYTES
            + O_W_BYTES
            + active_col * RN_BLOCKS_PER_COL * W_BLOCK
            + mwave * ROWS * W_BLOCK
        )
        out.extend(
            [
                f"      %{params_name} = aiex.dma_configure_task_for @wsh{col} {{",
                f"        aie.dma_bd(%W : memref<{TOTAL_W_BYTES}xi8>, {params_offset}, {W_BLOCK}, {dims(1, W_BLOCK)}) {{burst_length = 0 : i32}}",
                "        aie.end",
                "      } {issue_token = true}",
                f"      aiex.dma_start_task(%{params_name})",
            ]
        )
        for core_row in range(ROWS):
            residual_name = f"trnx{mwave}_{col}_{core_row}"
            residual_record = (mwave * O_ACTIVE_COLS + active_col) * ROWS + core_row
            residual_offset = A_BASE_BYTES + residual_record * A_BLOCK
            out.extend(
                [
                    f"      %{residual_name} = aiex.dma_configure_task_for @wsh{col} {{",
                    f"        aie.dma_bd(%A : memref<{A_BYTES}xi8>, {residual_offset}, {A_BLOCK}, {dims(1, A_BLOCK)}) {{burst_length = 0 : i32}}",
                    "        aie.end",
                    "      } {issue_token = true}",
                    f"      aiex.dma_start_task(%{residual_name})",
                ]
            )


def await_external_norm_inputs(mwave):
    for col in range(0, COLS, 2):
        names = [f"tow{mwave}_{col}", f"trnp{mwave}_{col}"]
        names += [f"trnx{mwave}_{col}_{row}" for row in range(ROWS)]
        for name in names:
            out.extend(
                [
                    f"      aiex.dma_await_task(%{name})",
                    f"      aiex.dma_free_task(%{name})",
                ]
            )


if OUTPUT_EXECUTION and DIRECT_OUTPUT:
    for active_col, col in enumerate(range(0, COLS, 2)):
        offset = W_BYTES + active_col * O_WEIGHTS_PER_COL * W_BLOCK
        if RESIDUAL_NORM:
            initial_waves = range(1) if EXTERNAL_RESIDUAL else range(O_M_WAVES)
            for mwave in initial_waves:
                weight_name = f"tow{mwave}_{col}"
                params_name = f"trnp{mwave}_{col}"
                params_offset = (
                    W_BYTES
                    + O_W_BYTES
                    + active_col * RN_BLOCKS_PER_COL * W_BLOCK
                    + mwave * ROWS * W_BLOCK
                )
                out += [
                    f"      %{weight_name} = aiex.dma_configure_task_for @wsh{col} {{",
                    f"        aie.dma_bd(%W : memref<{TOTAL_W_BYTES}xi8>, {offset}, {O_WEIGHTS_PER_COL * W_BLOCK}, {dims(O_WEIGHTS_PER_COL, W_BLOCK)}) {{burst_length = 0 : i32}}",
                    "        aie.end",
                    "      } {issue_token = true}",
                    f"      aiex.dma_start_task(%{weight_name})",
                ]
                if not EXTERNAL_RESIDUAL:
                    out += [
                            f"      %{params_name} = aiex.dma_configure_task_for @wsh{col} {{",
                            f"        aie.dma_bd(%W : memref<{TOTAL_W_BYTES}xi8>, {params_offset}, {ROWS * W_BLOCK}, {dims(ROWS, W_BLOCK)}) {{burst_length = 0 : i32}}",
                            "        aie.end",
                            "      } {issue_token = true}",
                            f"      aiex.dma_start_task(%{params_name})",
                    ]
        else:
            name = f"tow{col}"
            out += [
                f"      %{name} = aiex.dma_configure_task_for @wsh{col} {{",
                f"        aie.dma_bd(%W : memref<{TOTAL_W_BYTES}xi8>, {offset}, {O_WEIGHTS_PER_COL * W_BLOCK}, {dims(O_WEIGHTS_PER_COL, W_BLOCK)}) {{burst_length = 0 : i32}}",
                "        aie.end",
                f"      }} {{issue_token = true, repeat_count = {O_M_WAVES - 1} : i32}}",
                f"      aiex.dma_start_task(%{name})",
            ]
    if not RESIDUAL_NORM:
        start_direct_output_tasks(0, 0, O_SLICES // 4)
    else:
        start_norm_output_tasks(0)


if ATTENTION:
    attention_order = [0, 2, 4, 1, 3, 5] if DIRECT_OUTPUT else range(QUERY_GROUPS)
    for execution_group, group in enumerate(attention_order):
        if not DIRECT_OUTPUT:
            for pair in range(COLS // 2):
                source_col = pair * 2 + 1
                for lane in range(2):
                    target_col = pair * 2 + lane
                    name = f"tao{execution_group}_{source_col}_{lane}"
                    if OUTPUT_FIRST:
                        offset = group * ROWS * A_BLOCK + target_col * OUT_TILE
                        output_length = OUT_TILE
                        output_dimensions = packed_attention_output_dims()
                        output_task_attributes = (
                            f"issue_token = true, repeat_count = {ROWS - 1} : i32"
                        )
                    else:
                        offset = ATTENTION_BASE + (target_col * QUERY_GROUPS + group) * OUT_JOIN
                        output_length = OUT_JOIN
                        output_dimensions = dims(1, OUT_JOIN)
                        output_task_attributes = "issue_token = true"
                    out += [
                        f"      %{name} = aiex.dma_configure_task_for @osh{source_col} {{",
                        f"        aie.dma_bd(%R : memref<{R_BYTES}xi8>, {offset}, {output_length}, {output_dimensions}) {{burst_length = 0 : i32}}",
                        "        aie.end",
                        f"      }} {{{output_task_attributes}}}",
                        f"      aiex.dma_start_task(%{name})",
                    ]
        for row in range(ROWS):
            qname = f"taqi{execution_group}_{row}"
            qoffset = (row * QUERY_GROUPS + group) * Q_JOIN
            out += [
                f"      %{qname} = aiex.dma_configure_task_for @ash{row} {{",
                f"        aie.dma_bd(%Q : memref<{Q_BYTES}xi8>, {qoffset}, {Q_JOIN}, {dims(1, Q_JOIN)}) {{burst_length = 0 : i32}}",
                "        aie.end",
                "      } {issue_token = true}",
                f"      aiex.dma_start_task(%{qname})",
            ]
            kvname = f"takvi{execution_group}_{row}"
            out += [
                f"      %{kvname} = aiex.dma_configure_task_for @ash{row} {{",
                f"        aie.dma_bd(%KV : memref<{KV_BYTES}xi8>, 0, {KV_BYTES}, {dims(16, KV_TILE)}) {{burst_length = 0 : i32}}",
                "        aie.end",
                "      } {issue_token = true}",
                f"      aiex.dma_start_task(%{kvname})",
            ]
        for row in range(ROWS):
            for name in (
                f"taqi{execution_group}_{row}",
                f"takvi{execution_group}_{row}",
            ):
                out += [f"      aiex.dma_await_task(%{name})", f"      aiex.dma_free_task(%{name})"]
        if not DIRECT_OUTPUT:
            for pair in range(COLS // 2):
                source_col = pair * 2 + 1
                for lane in range(2):
                    name = f"tao{execution_group}_{source_col}_{lane}"
                    out += [
                        f"      aiex.dma_await_task(%{name})",
                        f"      aiex.dma_free_task(%{name})",
                    ]
        elif not RESIDUAL_NORM and execution_group == O_GROUPS - 1:
            await_direct_output_tasks(0, 0, O_SLICES // 4)
            start_direct_output_tasks(0, O_SLICES // 4, O_SLICES // 4)
            await_direct_output_tasks(0, O_SLICES // 4, O_SLICES // 4)
            start_direct_output_tasks(1, 0, O_SLICES // 4)
        elif not RESIDUAL_NORM and execution_group == QUERY_GROUPS - 1:
            await_direct_output_tasks(1, 0, O_SLICES // 4)
            start_direct_output_tasks(1, O_SLICES // 4, O_SLICES // 4)
            await_direct_output_tasks(1, O_SLICES // 4, O_SLICES // 4)
            for col in range(0, COLS, 2):
                name = f"tow{col}"
                out += [
                    f"      aiex.dma_await_task(%{name})",
                    f"      aiex.dma_free_task(%{name})",
                ]
        elif RESIDUAL_NORM and execution_group == O_GROUPS - 1:
            if EXTERNAL_RESIDUAL:
                start_external_norm_inputs(0, False)
            await_norm_output_tasks(0)
            if EXTERNAL_RESIDUAL:
                await_external_norm_inputs(0)
                start_external_norm_inputs(1, True)
            start_norm_output_tasks(1)
        elif RESIDUAL_NORM and execution_group == QUERY_GROUPS - 1:
            await_norm_output_tasks(1)
            if EXTERNAL_RESIDUAL:
                await_external_norm_inputs(1)
            else:
                for col in range(0, COLS, 2):
                    for mwave in range(O_M_WAVES):
                        for name in (f"tow{mwave}_{col}", f"trnp{mwave}_{col}"):
                            out += [
                                f"      aiex.dma_await_task(%{name})",
                                f"      aiex.dma_free_task(%{name})",
                            ]

if OUTPUT_EXECUTION and not DIRECT_OUTPUT:
    for active_col, col in enumerate(range(0, COLS, 2)):
        name = f"tow{col}"
        offset = W_BYTES + active_col * O_WEIGHTS_PER_COL * W_BLOCK
        out += [
            f"      %{name} = aiex.dma_configure_task_for @wsh{col} {{",
            f"        aie.dma_bd(%W : memref<{TOTAL_W_BYTES}xi8>, {offset}, {O_WEIGHTS_PER_COL * W_BLOCK}, {dims(O_WEIGHTS_PER_COL, W_BLOCK)}) {{burst_length = 0 : i32}}",
            "        aie.end",
            f"      }} {{issue_token = true, repeat_count = {O_M_WAVES - 1} : i32}}",
            f"      aiex.dma_start_task(%{name})",
        ]

    for mwave in range(O_M_WAVES):
        for oslice in range(O_SLICES):
            for active_col, col in enumerate(range(0, COLS, 2)):
                name = f"too{mwave}_{oslice}_{col}"
                offset = (
                    R_STAGE_BYTES
                    + ATT_BYTES
                    + mwave * 128 * 768 * 2
                    + (active_col * O_SLICES * 32 + oslice * 32) * 2
                )
                out += [
                    f"      %{name} = aiex.dma_configure_task_for @osh{col} {{",
                    f"        aie.dma_bd(%R : memref<{R_BYTES}xi8>, {offset}, {OUT_JOIN}, {output_projection_dims()}) {{burst_length = 0 : i32}}",
                    "        aie.end",
                    "      } {issue_token = true}",
                    f"      aiex.dma_start_task(%{name})",
                ]
            for group in range(O_GROUPS):
                attention_group = group * O_M_WAVES + mwave
                for row in range(ROWS):
                    name = f"toai{mwave}_{oslice}_{group}_{row}"
                    offset = (
                        R_STAGE_BYTES
                        + attention_group * OUT_JOIN
                        + row * OUT_TILE
                    )
                    out += [
                        f"      %{name} = aiex.dma_configure_task_for @ash{row} {{",
                        f"        aie.dma_bd(%R : memref<{R_BYTES}xi8>, {offset}, {A_BLOCK // COLS}, {attention_group_dims()}) {{burst_length = 0 : i32}}",
                        "        aie.end",
                        "      } {issue_token = true}",
                        f"      aiex.dma_start_task(%{name})",
                    ]
                for row in range(ROWS):
                    name = f"toai{mwave}_{oslice}_{group}_{row}"
                    out += [
                        f"      aiex.dma_await_task(%{name})",
                        f"      aiex.dma_free_task(%{name})",
                    ]
            for col in range(0, COLS, 2):
                name = f"too{mwave}_{oslice}_{col}"
                out += [
                    f"      aiex.dma_await_task(%{name})",
                    f"      aiex.dma_free_task(%{name})",
                ]

    for col in range(0, COLS, 2):
        name = f"tow{col}"
        out += [f"      aiex.dma_await_task(%{name})", f"      aiex.dma_free_task(%{name})"]

out += ["    }", "  }", "}"]
print("\n".join(out))
