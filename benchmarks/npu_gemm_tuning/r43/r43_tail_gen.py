#!/usr/bin/env python3
"""Compensated-input/output EmbeddingGemma post-FFN tail on AIE2P."""

import sys


def _int_flag(flag, default):
    for argument in sys.argv[1:]:
        if argument.startswith(flag + "="):
            return int(argument.split("=", 1)[1])
    return default

SPLIT_RESIDUAL = "--split-residual" in sys.argv[1:]
FUSION_READY = "--fusion-ready" in sys.argv[1:]
FUSED_R34_PACK = "--fused-r34-pack" in sys.argv[1:]
FUSED_NEXT_PACK = "--fused-next-pack" in sys.argv[1:] or FUSED_R34_PACK
if FUSION_READY and SPLIT_RESIDUAL:
    raise SystemExit("--fusion-ready uses the joined FFN/X row-state input")
if FUSED_NEXT_PACK and not FUSION_READY:
    raise SystemExit("--fused-next-pack requires --fusion-ready")
BATCH = _int_flag("--batch", 1)
if BATCH < 1:
    raise SystemExit("--batch must be positive")
if BATCH > 1 and (FUSION_READY or FUSED_NEXT_PACK or FUSED_R34_PACK):
    raise SystemExit("batched tail currently supports the non-fused split/joined ABI")
X_ROW = next(
    (
        int(arg.split("=", 1)[1])
        for arg in sys.argv[1:]
        if arg.startswith("--x-row-bytes=")
    ),
    768 * 2,
)

COLS, CORE_ROWS, HALVES = 8, 4, 2
PHASES_PER_DOCUMENT, TOKENS_PER_CORE, HIDDEN = 4, 2, 768
PHASES = PHASES_PER_DOCUMENT * BATCH
RUNTIME_PHASES = PHASES_PER_DOCUMENT
BF16_ROW = HIDDEN * 2
COMPLETED_ROW = 2 * BF16_ROW  # completed high/low, token-major
COMBINED_ROW = HIDDEN * 3 * 2  # FFN high/low, residual
Y_ROW = HIDDEN * 2 * 2 if SPLIT_RESIDUAL else COMBINED_ROW
INPUT_TILE = TOKENS_PER_CORE * Y_ROW
X_TILE = TOKENS_PER_CORE * BF16_ROW
OUTPUT_TILE = TOKENS_PER_CORE * COMPLETED_ROW
INPUT_JOIN = (COLS // HALVES) * INPUT_TILE
X_JOIN = (COLS // HALVES) * X_TILE
OUTPUT_JOIN = (COLS // HALVES) * OUTPUT_TILE
DOCUMENT_ROWS = 288
INPUT_BYTES = BATCH * DOCUMENT_ROWS * COMBINED_ROW
X_BYTES = BATCH * DOCUMENT_ROWS * X_ROW
OUTPUT_BYTES = BATCH * DOCUMENT_ROWS * COMPLETED_ROW
PARAM_RECORD = INPUT_TILE
PARAM_BYTES_TOTAL = COLS * CORE_ROWS * PARAM_RECORD
PARAM_BYTES = HIDDEN * 2 + 4
NEXT_PARAM_BYTES = 3 * (2 * 256 * 4 + 2 * 256 * 2)
PACK_BYTES = 3 * (8 * 256 + 8 * 4)
CHUNK_BYTES = PACK_BYTES // 3
PACK_DIAGNOSTIC_BYTES = CORE_ROWS * HALVES * 3 * OUTPUT_JOIN
R34_BLOCK = 16_384
R34_PREFIX = PACK_BYTES
R34_INPUT_BYTES = 4 * 45 * R34_BLOCK
R34_COMPACT_MTS = 4
R34_COMPACT_BYTES = R34_COMPACT_MTS * 3 * 2 * OUTPUT_JOIN
INF = 9223372036854775807

cores = [(col, row) for col in range(COLS) for row in range(CORE_ROWS)]


def token_base(core):
    col, row = core
    return (col // (COLS // HALVES)) * 128 + row * 32 + (col % 4) * 8


owner_order = sorted(cores, key=token_base)
# R114 assigns each canonical 24-token block to an adjacent triomino (and the
# final 16-token block to a domino). Neighboring-core ObjectFIFOs use shared
# data memory and locks, avoiding the stream-switch routes that the fused tail
# cannot legally accommodate.
r34_chains = [
    *[[(col, 0), (col, 1), (col, 2)] for col in range(COLS)],
    [(0, 3), (1, 3), (2, 3)],
    [(3, 3), (4, 3), (5, 3)],
    [(6, 3), (7, 3)],
]
r34_roles = {}
r34_packers = []
for block, chain in enumerate(r34_chains):
    for lm, core in enumerate(chain):
        r34_roles[core] = (block, lm, len(chain))
    r34_packers.append(chain[-1])

r34_chunk_edges = [
    (source, target)
    for chain in r34_chains
    for source, target in zip(chain, chain[1:])
]
r34_chunk_previous = {target: source for source, target in r34_chunk_edges}
r34_chunk_next = {source: target for source, target in r34_chunk_edges}

r34_packers_by_mt = {}
for core in r34_packers:
    col, row = core
    mt = row + (col // (COLS // HALVES)) * CORE_ROWS
    r34_packers_by_mt.setdefault(mt, []).append(core)
r34_compact_mts = sorted(r34_packers_by_mt)
assert len(r34_compact_mts) == R34_COMPACT_MTS


def linear_dims(size):
    return f"[<size = {size // 512}, stride = 512>, <size = 512, stride = 1>]"


def document_dims(dimensions, document_stride):
    if BATCH == 1:
        return dimensions
    return (
        f"[<size = {BATCH}, stride = {document_stride}>, "
        "<size = 1, stride = 4>, "
        + dimensions.removeprefix("[")
    )


def repeated_task_attrs():
    if BATCH == 1:
        return " {issue_token = true}"
    return f" {{issue_token = true, repeat_count = {BATCH - 1} : i32}}"


def r34_core_lines(col, row):
    if not FUSED_R34_PACK:
        return []
    core = (col, row)
    block, lm, chain_len = r34_roles[core]
    packer = r34_packers[block]
    lines = []

    def fifo_name(prefix, source, target):
        return f"{prefix}{source[0]}_{source[1]}_{target[0]}_{target[1]}"

    def emit_real(tag, group, source):
        result = []
        for plane, function in [("q", "r114_emit_q"), ("s", "r114_emit_scales")]:
            result += [
                f"        %{tag}{plane}o{group} = aie.objectfifo.acquire @oc{col}_{row}(Produce, 1) : !aie.objectfifosubview<memref<{OUTPUT_TILE}xi8>>",
                f"        %{tag}{plane}ov{group} = aie.objectfifo.subview.access %{tag}{plane}o{group}[0] : !aie.objectfifosubview<memref<{OUTPUT_TILE}xi8>> -> memref<{OUTPUT_TILE}xi8>",
                f"        func.call @{function}({source}, %{tag}{plane}ov{group}) : (memref<{R34_PREFIX}xi8>, memref<{OUTPUT_TILE}xi8>) -> ()",
                f"        aie.objectfifo.release @oc{col}_{row}(Produce, 1)",
            ]
        return result

    def emit_placeholder(tag, group):
        return [
            f"        %{tag}copies{group} = arith.constant 2 : index",
            f"        scf.for %{tag}copy{group} = %z to %{tag}copies{group} step %one {{",
            f"          %{tag}o{group} = aie.objectfifo.acquire @oc{col}_{row}(Produce, 1) : !aie.objectfifosubview<memref<{OUTPUT_TILE}xi8>>",
            f"          %{tag}ov{group} = aie.objectfifo.subview.access %{tag}o{group}[0] : !aie.objectfifosubview<memref<{OUTPUT_TILE}xi8>> -> memref<{OUTPUT_TILE}xi8>",
            f"          func.call @r114_zero_output(%{tag}ov{group}) : (memref<{OUTPUT_TILE}xi8>) -> ()",
            f"          aie.objectfifo.release @oc{col}_{row}(Produce, 1)",
            "        }",
        ]

    for group in range(3):
        lines.append(f"        %r34group{group} = arith.constant {group} : i32")
        if lm == 0 and chain_len > 1:
            target = r34_chunk_next[core]
            name = fifo_name("r34k", core, target)
            lines += [
                f"        %kout{group} = aie.objectfifo.acquire @{name}(Produce, 1) : !aie.objectfifosubview<memref<{CHUNK_BYTES}xi8>>",
                f"        %koutv{group} = aie.objectfifo.subview.access %kout{group}[0] : !aie.objectfifosubview<memref<{CHUNK_BYTES}xi8>> -> memref<{CHUNK_BYTES}xi8>",
                f"        func.call @r114_copy_group(%packblob{col}_{row}, %koutv{group}, %r34group{group}) : (memref<{PACK_BYTES}xi8>, memref<{CHUNK_BYTES}xi8>, i32) -> ()",
                f"        aie.objectfifo.release @{name}(Produce, 1)",
            ]
        elif lm < chain_len - 1:
            source = r34_chunk_previous[core]
            target = r34_chunk_next[core]
            incoming = fifo_name("r34k", source, core)
            outgoing = fifo_name("r34k", core, target)
            lines += [
                f"        %kin{group} = aie.objectfifo.acquire @{incoming}(Consume, 1) : !aie.objectfifosubview<memref<{CHUNK_BYTES}xi8>>",
                f"        %kinv{group} = aie.objectfifo.subview.access %kin{group}[0] : !aie.objectfifosubview<memref<{CHUNK_BYTES}xi8>> -> memref<{CHUNK_BYTES}xi8>",
                f"        %krelay{group} = aie.objectfifo.acquire @{outgoing}(Produce, 1) : !aie.objectfifosubview<memref<{CHUNK_BYTES}xi8>>",
                f"        %krelayv{group} = aie.objectfifo.subview.access %krelay{group}[0] : !aie.objectfifosubview<memref<{CHUNK_BYTES}xi8>> -> memref<{CHUNK_BYTES}xi8>",
                f"        func.call @r114_copy_chunk(%kinv{group}, %krelayv{group}) : (memref<{CHUNK_BYTES}xi8>, memref<{CHUNK_BYTES}xi8>) -> ()",
                f"        aie.objectfifo.release @{incoming}(Consume, 1)",
                f"        aie.objectfifo.release @{outgoing}(Produce, 1)",
                f"        %kown{group} = aie.objectfifo.acquire @{outgoing}(Produce, 1) : !aie.objectfifosubview<memref<{CHUNK_BYTES}xi8>>",
                f"        %kownv{group} = aie.objectfifo.subview.access %kown{group}[0] : !aie.objectfifosubview<memref<{CHUNK_BYTES}xi8>> -> memref<{CHUNK_BYTES}xi8>",
                f"        func.call @r114_copy_group(%packblob{col}_{row}, %kownv{group}, %r34group{group}) : (memref<{PACK_BYTES}xi8>, memref<{CHUNK_BYTES}xi8>, i32) -> ()",
                f"        aie.objectfifo.release @{outgoing}(Produce, 1)",
            ]
        else:
            source = r34_chunk_previous.get(core)
            if source is not None:
                incoming = fifo_name("r34k", source, core)
                for predecessor in range(chain_len - 1):
                    lines += [
                        f"        %kpack{group}_{predecessor} = aie.objectfifo.acquire @{incoming}(Consume, 1) : !aie.objectfifosubview<memref<{CHUNK_BYTES}xi8>>",
                        f"        %kpackv{group}_{predecessor} = aie.objectfifo.subview.access %kpack{group}_{predecessor}[0] : !aie.objectfifosubview<memref<{CHUNK_BYTES}xi8>> -> memref<{CHUNK_BYTES}xi8>",
                        f"        %klm{group}_{predecessor} = arith.constant {predecessor} : i32",
                        f"        func.call @r114_place_chunk(%r34block{col}_{row}, %kpackv{group}_{predecessor}, %klm{group}_{predecessor}) : (memref<{R34_PREFIX}xi8>, memref<{CHUNK_BYTES}xi8>, i32) -> ()",
                        f"        aie.objectfifo.release @{incoming}(Consume, 1)",
                    ]
            lines += [
                f"        %kownlm{group} = arith.constant {lm} : i32",
                f"        func.call @r114_place_group(%r34block{col}_{row}, %packblob{col}_{row}, %kownlm{group}, %r34group{group}) : (memref<{R34_PREFIX}xi8>, memref<{PACK_BYTES}xi8>, i32, i32) -> ()",
            ]

        mt = row + (col // (COLS // HALVES)) * CORE_ROWS
        if mt in r34_packers_by_mt:
            if core == packer:
                lines += emit_real("r34compact_", group, f"%r34block{col}_{row}")
            else:
                lines += emit_placeholder("r34compactzero_", group)
    return lines


out = ["module {", "  aie.device(npu2) {"]
for col in range(COLS):
    out += [f"    %shim{col} = aie.tile({col}, 0)", f"    %mt{col} = aie.tile({col}, 1)"]
    for row in range(CORE_ROWS):
        out += [
            f"    %c{col}_{row} = aie.tile({col}, {row + 2})",
            f"    %params{col}_{row} = aie.buffer(%c{col}_{row}) {{sym_name = \"params{col}_{row}\"}} : memref<{PARAM_BYTES}xi8>",
            *(
                [
                    f"    %xlocal{col}_{row} = aie.buffer(%c{col}_{row}) {{sym_name = \"xlocal{col}_{row}\"}} : memref<{X_TILE}xi8>"
                ]
                if SPLIT_RESIDUAL and not FUSION_READY
                else []
            ),
            *(
                [
                    f'    %nextparams{col}_{row} = aie.buffer(%c{col}_{row}) {{sym_name = "nextparams{col}_{row}"}} : memref<{NEXT_PARAM_BYTES}xi8>',
                    f'    %packblob{col}_{row} = aie.buffer(%c{col}_{row}) {{sym_name = "packblob{col}_{row}"}} : memref<{PACK_BYTES}xi8>',
                    f'    %packscratch{col}_{row} = aie.buffer(%c{col}_{row}) {{sym_name = "packscratch{col}_{row}"}} : memref<256xf32>',
                    f'    %packsum{col}_{row} = aie.buffer(%c{col}_{row}) {{sym_name = "packsum{col}_{row}"}} : memref<8xf32>',
                    *(
                        [
                            f'    %r34block{col}_{row} = aie.buffer(%c{col}_{row}) {{sym_name = "r34block{col}_{row}"}} : memref<{R34_PREFIX}xi8>'
                        ]
                        if FUSED_R34_PACK and (col, row) in r34_packers
                        else []
                    ),
                ]
                if FUSED_NEXT_PACK
                else []
            ),
        ]

for row in range(CORE_ROWS):
    for half in range(HALVES):
        mt = row + half * CORE_ROWS
        first_col = half * (COLS // HALVES)
        consumers, producers = [], []
        input_offsets, output_offsets = [], []
        for local_col in range(COLS // HALVES):
            col = first_col + local_col
            consumers.append(f"@dc{col}_{row}")
            producers.append(f"@oc{col}_{row}")
            input_offsets.append(str(local_col * INPUT_TILE))
            output_offsets.append(str(local_col * OUTPUT_TILE))
            out += [
                f"    aie.objectfifo @dc{col}_{row}(%mt{mt}, {{%c{col}_{row}}}, 1 : i32) : !aie.objectfifo<memref<{INPUT_TILE}xi8>>",
                f"    aie.objectfifo @oc{col}_{row}(%c{col}_{row}, {{%mt{mt}}}, 1 : i32) : !aie.objectfifo<memref<{OUTPUT_TILE}xi8>>",
            ]
        out += [
            f"    aie.objectfifo @dsh{half}_{row}(%shim{mt}, {{%mt{mt}}}, 1 : i32) : !aie.objectfifo<memref<{INPUT_JOIN}xi8>>",
            f"    aie.objectfifo.link [@dsh{half}_{row}] -> [{', '.join(consumers)}] ([] [{', '.join(input_offsets)}])",
            f"    aie.objectfifo @osh{half}_{row}(%mt{mt}, {{%shim{mt}}}, 1 : i32) : !aie.objectfifo<memref<{OUTPUT_JOIN}xi8>>",
            f"    aie.objectfifo.link [{', '.join(producers)}] -> [@osh{half}_{row}] ([{', '.join(output_offsets)}] [])",
        ]
        if SPLIT_RESIDUAL:
            if FUSION_READY:
                x_consumers, x_offsets = [], []
                for local_col in range(COLS // HALVES):
                    col = first_col + local_col
                    x_consumers.append(f"@xc{col}_{row}")
                    x_offsets.append(str(local_col * X_TILE))
                    out.append(
                        f"    aie.objectfifo @xc{col}_{row}(%mt{mt}, {{%c{col}_{row}}}, 1 : i32) : !aie.objectfifo<memref<{X_TILE}xi8>>"
                    )
                out += [
                    f"    aie.objectfifo @xsc{half}_{row}(%shim{mt}, {{%mt{mt}}}, 1 : i32) : !aie.objectfifo<memref<{X_JOIN}xi8>>",
                    f"    aie.objectfifo.link [@xsc{half}_{row}] -> [{', '.join(x_consumers)}] ([] [{', '.join(x_offsets)}])",
                ]
            else:
                out.append(
                    f"    aie.objectfifo @xsc{half}_{row}(%shim{mt}, {{%c{first_col}_{row}}}, 1 : i32) : !aie.objectfifo<memref<{X_JOIN}xi8>>"
                )
                for col in range(first_col, first_col + COLS // HALVES - 1):
                    out.append(
                        f'    aie.flow(%c{col}_{row}, "Core" : 0, %c{col + 1}_{row}, "Core" : 0)'
                    )

if FUSED_R34_PACK:
    for source, target in r34_chunk_edges:
        out.append(
            f"    aie.objectfifo @r34k{source[0]}_{source[1]}_{target[0]}_{target[1]}(%c{source[0]}_{source[1]}, {{%c{target[0]}_{target[1]}}}, 1 : i32) : !aie.objectfifo<memref<{CHUNK_BYTES}xi8>>"
        )
out += [
    f'    func.func private @r43_copy_params(memref<{INPUT_TILE}xi8>, memref<{PARAM_BYTES}xi8>) attributes {{link_with = "r43_tail.o"}}',
    f'    func.func private @r43_post_ffn_direct_tail_bf16x2(memref<{OUTPUT_TILE}xi8>, memref<{INPUT_TILE}xi8>, '
    + (f'memref<{X_TILE}xi8>, ' if SPLIT_RESIDUAL else '')
    + f'memref<{PARAM_BYTES}xi8>) attributes {{link_with = "r43_tail.o"}}',
]
if SPLIT_RESIDUAL and not FUSION_READY:
    out += [
        f'    func.func private @r46_x_source(memref<{X_JOIN}xi8>, memref<{X_TILE}xi8>) attributes {{link_with = "r43_tail.o"}}',
        f'    func.func private @r46_x_relay(memref<{X_TILE}xi8>, i32) attributes {{link_with = "r43_tail.o"}}',
    ]
if FUSED_NEXT_PACK:
    pack_object = "r114.o" if FUSED_R34_PACK else "r113.o"
    out += [
        f'    func.func private @r113_copy_next_params(memref<{INPUT_TILE}xi8>, memref<{NEXT_PARAM_BYTES}xi8>) attributes {{link_with = "{pack_object}"}}',
        f'    func.func private @r113_pack_phase(memref<{OUTPUT_TILE}xi8>, memref<{NEXT_PARAM_BYTES}xi8>, memref<{PACK_BYTES}xi8>, memref<256xf32>, memref<8xf32>, i32) attributes {{link_with = "{pack_object}"}}',
        *(
            [f'    func.func private @r113_emit_pack_group(memref<{PACK_BYTES}xi8>, memref<{OUTPUT_TILE}xi8>, i32) attributes {{link_with = "{pack_object}"}}']
            if not FUSED_R34_PACK
            else []
        ),
    ]
if FUSED_R34_PACK:
    out += [
        f'    func.func private @r114_copy_group(memref<{PACK_BYTES}xi8>, memref<{CHUNK_BYTES}xi8>, i32) attributes {{link_with = "r114.o"}}',
        f'    func.func private @r114_copy_chunk(memref<{CHUNK_BYTES}xi8>, memref<{CHUNK_BYTES}xi8>) attributes {{link_with = "r114.o"}}',
        f'    func.func private @r114_place_chunk(memref<{R34_PREFIX}xi8>, memref<{CHUNK_BYTES}xi8>, i32) attributes {{link_with = "r114.o"}}',
        f'    func.func private @r114_place_group(memref<{R34_PREFIX}xi8>, memref<{PACK_BYTES}xi8>, i32, i32) attributes {{link_with = "r114.o"}}',
        f'    func.func private @r114_emit_q(memref<{R34_PREFIX}xi8>, memref<{OUTPUT_TILE}xi8>) attributes {{link_with = "r114.o"}}',
        f'    func.func private @r114_emit_scales(memref<{R34_PREFIX}xi8>, memref<{OUTPUT_TILE}xi8>) attributes {{link_with = "r114.o"}}',
        f'    func.func private @r114_zero_output(memref<{OUTPUT_TILE}xi8>) attributes {{link_with = "r114.o"}}',
    ]

for col in range(COLS):
    for row in range(CORE_ROWS):
        half = col // (COLS // HALVES)
        local_col = col % (COLS // HALVES)
        out += [
            f"    %core{col}_{row} = aie.core(%c{col}_{row}) {{",
            "      %z = arith.constant 0 : index",
            f"      %inf = arith.constant {INF} : index",
            "      %one = arith.constant 1 : index",
            f"      %phases = arith.constant {PHASES} : index",
            "      scf.for %outer = %z to %inf step %one {",
            f"        %p = aie.objectfifo.acquire @dc{col}_{row}(Consume, 1) : !aie.objectfifosubview<memref<{INPUT_TILE}xi8>>",
            f"        %pv = aie.objectfifo.subview.access %p[0] : !aie.objectfifosubview<memref<{INPUT_TILE}xi8>> -> memref<{INPUT_TILE}xi8>",
            f"        func.call @r43_copy_params(%pv, %params{col}_{row}) : (memref<{INPUT_TILE}xi8>, memref<{PARAM_BYTES}xi8>) -> ()",
            f"        aie.objectfifo.release @dc{col}_{row}(Consume, 1)",
            *(
                [
                    f"        %next = aie.objectfifo.acquire @dc{col}_{row}(Consume, 1) : !aie.objectfifosubview<memref<{INPUT_TILE}xi8>>",
                    f"        %nextv = aie.objectfifo.subview.access %next[0] : !aie.objectfifosubview<memref<{INPUT_TILE}xi8>> -> memref<{INPUT_TILE}xi8>",
                    f"        func.call @r113_copy_next_params(%nextv, %nextparams{col}_{row}) : (memref<{INPUT_TILE}xi8>, memref<{NEXT_PARAM_BYTES}xi8>) -> ()",
                    f"        aie.objectfifo.release @dc{col}_{row}(Consume, 1)",
                ]
                if FUSED_NEXT_PACK
                else []
            ),
            "        scf.for %phase = %z to %phases step %one {",
            *(
                (
                    [
                        f"          %x = aie.objectfifo.acquire @xc{col}_{row}(Consume, 1) : !aie.objectfifosubview<memref<{X_TILE}xi8>>",
                        f"          %xv = aie.objectfifo.subview.access %x[0] : !aie.objectfifosubview<memref<{X_TILE}xi8>> -> memref<{X_TILE}xi8>",
                    ]
                    if FUSION_READY
                    else
                    [
                        f"          %x = aie.objectfifo.acquire @xsc{half}_{row}(Consume, 1) : !aie.objectfifosubview<memref<{X_JOIN}xi8>>",
                        f"          %xv = aie.objectfifo.subview.access %x[0] : !aie.objectfifosubview<memref<{X_JOIN}xi8>> -> memref<{X_JOIN}xi8>",
                        f"          func.call @r46_x_source(%xv, %xlocal{col}_{row}) : (memref<{X_JOIN}xi8>, memref<{X_TILE}xi8>) -> ()",
                        f"          aie.objectfifo.release @xsc{half}_{row}(Consume, 1)",
                    ]
                    if local_col == 0
                    else [
                        f"          %forward = arith.constant {COLS // HALVES - local_col - 1} : i32",
                        f"          func.call @r46_x_relay(%xlocal{col}_{row}, %forward) : (memref<{X_TILE}xi8>, i32) -> ()",
                    ]
                )
                if SPLIT_RESIDUAL
                else []
            ),
            f"          %o = aie.objectfifo.acquire @oc{col}_{row}(Produce, 1) : !aie.objectfifosubview<memref<{OUTPUT_TILE}xi8>>",
            f"          %ov = aie.objectfifo.subview.access %o[0] : !aie.objectfifosubview<memref<{OUTPUT_TILE}xi8>> -> memref<{OUTPUT_TILE}xi8>",
            f"          %d = aie.objectfifo.acquire @dc{col}_{row}(Consume, 1) : !aie.objectfifosubview<memref<{INPUT_TILE}xi8>>",
            f"          %dv = aie.objectfifo.subview.access %d[0] : !aie.objectfifosubview<memref<{INPUT_TILE}xi8>> -> memref<{INPUT_TILE}xi8>",
            f"          func.call @r43_post_ffn_direct_tail_bf16x2(%ov, %dv, "
            + (
                ("%xv, " if FUSION_READY else f"%xlocal{col}_{row}, ")
                if SPLIT_RESIDUAL
                else ""
            )
            + f"%params{col}_{row}) : (memref<{OUTPUT_TILE}xi8>, memref<{INPUT_TILE}xi8>, "
            + (f"memref<{X_TILE}xi8>, " if SPLIT_RESIDUAL else "")
            + f"memref<{PARAM_BYTES}xi8>) -> ()",
            *(
                [
                    "          %phasei = arith.index_cast %phase : index to i32",
                    f"          func.call @r113_pack_phase(%ov, %nextparams{col}_{row}, %packblob{col}_{row}, %packscratch{col}_{row}, %packsum{col}_{row}, %phasei) : (memref<{OUTPUT_TILE}xi8>, memref<{NEXT_PARAM_BYTES}xi8>, memref<{PACK_BYTES}xi8>, memref<256xf32>, memref<8xf32>, i32) -> ()",
                ]
                if FUSED_NEXT_PACK
                else []
            ),
            f"          aie.objectfifo.release @dc{col}_{row}(Consume, 1)",
            f"          aie.objectfifo.release @oc{col}_{row}(Produce, 1)",
            *(
                [f"          aie.objectfifo.release @xc{col}_{row}(Consume, 1)"]
                if SPLIT_RESIDUAL and FUSION_READY
                else []
            ),
            "        }",
            *(
                [
                    *[
                        line
                        for group in range(3)
                        for line in [
                            f"        %packgroup{group} = arith.constant {group} : i32",
                            f"        %packout{group} = aie.objectfifo.acquire @oc{col}_{row}(Produce, 1) : !aie.objectfifosubview<memref<{OUTPUT_TILE}xi8>>",
                            f"        %packout{group}v = aie.objectfifo.subview.access %packout{group}[0] : !aie.objectfifosubview<memref<{OUTPUT_TILE}xi8>> -> memref<{OUTPUT_TILE}xi8>",
                            f"        func.call @r113_emit_pack_group(%packblob{col}_{row}, %packout{group}v, %packgroup{group}) : (memref<{PACK_BYTES}xi8>, memref<{OUTPUT_TILE}xi8>, i32) -> ()",
                            f"        aie.objectfifo.release @oc{col}_{row}(Produce, 1)",
                        ]
                    ],
                ]
                if FUSED_NEXT_PACK and not FUSED_R34_PACK
                else []
            ),
            *r34_core_lines(col, row),
            "      }",
            "      aie.end",
            "    } {stack_size = 2048 : i32}",
        ]

runtime_input = (
    f"%Y: memref<{INPUT_BYTES}xi8>, %X: memref<{X_BYTES}xi8>"
    if SPLIT_RESIDUAL
    else f"%D: memref<{INPUT_BYTES}xi8>"
)
runtime_tail = f"%P: memref<{PARAM_BYTES_TOTAL}xi8>, "
if FUSED_NEXT_PACK:
    runtime_tail += f"%N: memref<{PARAM_BYTES_TOTAL}xi8>, "
runtime_tail += f"%O: memref<{OUTPUT_BYTES}xi8>"
if FUSED_NEXT_PACK:
    runtime_tail += (
        f", %Q: memref<{R34_COMPACT_BYTES}xi8>"
        if FUSED_R34_PACK
        else f", %Q: memref<{PACK_DIAGNOSTIC_BYTES}xi8>"
    )
out.append(f"    aie.runtime_sequence({runtime_input}, {runtime_tail}) {{")
pack_retire_lines = []
pack_q_lines = []
for row in range(CORE_ROWS):
    for half in range(HALVES):
        first_col = half * (COLS // HALVES)
        record_base = row * COLS + first_col
        if FUSED_R34_PACK and row < CORE_ROWS - 1:
            token_base = half * 96 + row * PHASES * TOKENS_PER_CORE
            core_token_stride = 3 * PHASES * TOKENS_PER_CORE
        elif FUSED_R34_PACK:
            token_base = 192 + half * (COLS // HALVES) * PHASES * TOKENS_PER_CORE
            core_token_stride = PHASES * TOKENS_PER_CORE
        else:
            token_base = half * 128 + row * 32
            core_token_stride = (COLS // HALVES) * TOKENS_PER_CORE
        pname = f"tp{half}_{row}"
        out += [
            f"      %{pname} = aiex.dma_configure_task_for @dsh{half}_{row} {{",
            f"        aie.dma_bd(%P : memref<{PARAM_BYTES_TOTAL}xi8>, {record_base * PARAM_RECORD}, {INPUT_JOIN}, {linear_dims(INPUT_JOIN)}) {{burst_length = 0 : i32}}",
            "        aie.end",
            "      } {issue_token = true}",
            f"      aiex.dma_start_task(%{pname})",
        ]
        if FUSED_NEXT_PACK:
            nname = f"tn{half}_{row}"
            out += [
                f"      %{nname} = aiex.dma_configure_task_for @dsh{half}_{row} {{",
                f"        aie.dma_bd(%N : memref<{PARAM_BYTES_TOTAL}xi8>, {record_base * INPUT_TILE}, {INPUT_JOIN}, {linear_dims(INPUT_JOIN)}) {{burst_length = 0 : i32}}",
                "        aie.end",
                "      } {issue_token = true}",
                f"      aiex.dma_start_task(%{nname})",
            ]
        for phase in range(RUNTIME_PHASES):
            phase_token = token_base + phase * (
                TOKENS_PER_CORE
                if FUSION_READY
                else (COLS // HALVES) * TOKENS_PER_CORE
            )
            iname = f"td{half}_{row}_{phase}"
            oname = f"to{half}_{row}_{phase}"
            y_dims = (
                f"[<size = {COLS // HALVES}, stride = {core_token_stride * COMBINED_ROW}>, "
                f"<size = {INPUT_TILE // 512}, stride = 512>, <size = 512, stride = 1>]"
                if FUSION_READY
                else f"[<size = {COLS // HALVES * TOKENS_PER_CORE}, stride = {COMBINED_ROW}>, "
                f"<size = {Y_ROW}, stride = 1>]"
                if SPLIT_RESIDUAL
                else linear_dims(INPUT_JOIN)
            )
            output_dims = (
                f"[<size = {COLS // HALVES}, stride = {core_token_stride * COMPLETED_ROW}>, "
                f"<size = {OUTPUT_TILE // 512}, stride = 512>, <size = 512, stride = 1>]"
                if FUSION_READY
                else linear_dims(OUTPUT_JOIN)
            )
            y_dims = document_dims(y_dims, DOCUMENT_ROWS * COMBINED_ROW)
            output_dims = document_dims(
                output_dims, DOCUMENT_ROWS * COMPLETED_ROW
            )
            task_attrs = repeated_task_attrs()
            out += [
                f"      %{iname} = aiex.dma_configure_task_for @dsh{half}_{row} {{",
                f"        aie.dma_bd(%{'Y' if SPLIT_RESIDUAL else 'D'} : memref<{INPUT_BYTES}xi8>, {phase_token * COMBINED_ROW}, {INPUT_JOIN}, {y_dims}) {{burst_length = 0 : i32}}",
                "        aie.end",
                f"      }}{task_attrs}",
                f"      aiex.dma_start_task(%{iname})",
                f"      %{oname} = aiex.dma_configure_task_for @osh{half}_{row} {{",
                f"        aie.dma_bd(%O : memref<{OUTPUT_BYTES}xi8>, {phase_token * COMPLETED_ROW}, {OUTPUT_JOIN}, {output_dims}) {{burst_length = 0 : i32}}",
                "        aie.end",
                f"      }}{task_attrs}",
                f"      aiex.dma_start_task(%{oname})",
            ]
            if SPLIT_RESIDUAL:
                xname = f"tx{half}_{row}_{phase}"
                x_dims = (
                    f"[<size = {COLS // HALVES}, stride = {(COLS // HALVES) * TOKENS_PER_CORE * X_ROW}>, <size = {X_TILE}, stride = 1>]"
                    if FUSION_READY and X_ROW == BF16_ROW
                    else f"[<size = {COLS // HALVES}, stride = {(COLS // HALVES) * TOKENS_PER_CORE * X_ROW}>, <size = {TOKENS_PER_CORE}, stride = {X_ROW}>, <size = {BF16_ROW}, stride = 1>]"
                    if FUSION_READY
                    else linear_dims(X_JOIN)
                    if X_ROW == BF16_ROW
                    else f"[<size = {COLS // HALVES * TOKENS_PER_CORE}, stride = {X_ROW}>, <size = {BF16_ROW}, stride = 1>]"
                )
                x_dims = document_dims(x_dims, DOCUMENT_ROWS * X_ROW)
                out[-5:-5] = [
                    f"      %{xname} = aiex.dma_configure_task_for @xsc{half}_{row} {{",
                    f"        aie.dma_bd(%X : memref<{X_BYTES}xi8>, {phase_token * X_ROW}, {X_JOIN}, {x_dims}) {{burst_length = 0 : i32}}",
                    "        aie.end",
                    f"      }}{task_attrs}",
                    f"      aiex.dma_start_task(%{xname})",
                ]
        if FUSED_NEXT_PACK and not FUSED_R34_PACK:
            # One shim S2MM queue cannot retain all four completed-state tasks
            # plus three diagnostic pack tasks. Retire the oldest completed
            # output before publishing the three pack objects. Defer this
            # until every row/half has launched so the wait does not serialize
            # otherwise independent stripes.
            first_output = f"to{half}_{row}_0"
            pack_retire_lines += [
                f"      aiex.dma_await_task(%{first_output})",
                f"      aiex.dma_free_task(%{first_output})",
            ]
            for group in range(3):
                qname = f"tq{half}_{row}_{group}"
                qoffset = ((row * HALVES + half) * 3 + group) * OUTPUT_JOIN
                pack_q_lines += [
                    f"      %{qname} = aiex.dma_configure_task_for @osh{half}_{row} {{",
                    f"        aie.dma_bd(%Q : memref<{PACK_DIAGNOSTIC_BYTES}xi8>, {qoffset}, {OUTPUT_JOIN}, {linear_dims(OUTPUT_JOIN)}) {{burst_length = 0 : i32}}",
                    "        aie.end",
                    "      } {issue_token = true}",
                    f"      aiex.dma_start_task(%{qname})",
                ]

if FUSED_NEXT_PACK and not FUSED_R34_PACK:
    out += pack_retire_lines
    out += pack_q_lines

if FUSED_R34_PACK:
    for compact_index, mt in enumerate(r34_compact_mts):
        row = mt % CORE_ROWS
        half = mt // CORE_ROWS
        # Reuse the admitted completed-output route and retain one compact Q
        # plane plus one compact scale plane. The next resident GEMM reuses
        # these bytes across its five N-macros instead of materializing five
        # 16 KiB activation replicas.
        for phase in range(RUNTIME_PHASES):
            completed = f"to{half}_{row}_{phase}"
            out += [
                f"      aiex.dma_await_task(%{completed})",
                f"      aiex.dma_free_task(%{completed})",
            ]
        for group in range(3):
            for plane_index, plane in enumerate(["q", "s"]):
                offset = (compact_index * 3 * 2 + group * 2 + plane_index) * OUTPUT_JOIN
                name = f"tr34{plane}{mt}_{group}"
                out += [
                    f"      %{name} = aiex.dma_configure_task_for @osh{half}_{row} {{",
                    f"        aie.dma_bd(%Q : memref<{R34_COMPACT_BYTES}xi8>, {offset}, {OUTPUT_JOIN}, {linear_dims(OUTPUT_JOIN)}) {{burst_length = 0 : i32}}",
                    "        aie.end",
                    "      } {issue_token = true}",
                    f"      aiex.dma_start_task(%{name})",
                    f"      aiex.dma_await_task(%{name})",
                    f"      aiex.dma_free_task(%{name})",
                ]

for row in range(CORE_ROWS):
    for half in range(HALVES):
        names = [f"tp{half}_{row}"]
        if FUSED_NEXT_PACK:
            names += [f"tn{half}_{row}"]
        for phase in range(RUNTIME_PHASES):
            names += [f"td{half}_{row}_{phase}"]
            if SPLIT_RESIDUAL:
                names += [f"tx{half}_{row}_{phase}"]
            used_r34_mt = row + half * CORE_ROWS in r34_packers_by_mt
            if not (
                (FUSED_NEXT_PACK and not FUSED_R34_PACK and phase == 0)
                or (FUSED_R34_PACK and used_r34_mt)
            ):
                names += [f"to{half}_{row}_{phase}"]
        if FUSED_NEXT_PACK and not FUSED_R34_PACK:
            names += [f"tq{half}_{row}_{group}" for group in range(3)]
        for name in names:
            out += [f"      aiex.dma_await_task(%{name})", f"      aiex.dma_free_task(%{name})"]
out += ["    }", "  }", "}"]
print("\n".join(out))
