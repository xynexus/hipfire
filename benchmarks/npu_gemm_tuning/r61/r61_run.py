#!/usr/bin/env python3
"""R61: complete M256/K768/N1280 scaled QKV from the resident R34 ABI."""

from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import struct
import subprocess
import sys
import time

import numpy as np
from aie.dialects.aie import WireBundle
from aie.helpers.taplib import TensorAccessPattern, TensorTiler2D
from aie.iron import ObjectFifo, Program, Runtime, Worker, tensor, zeros
from aie.iron.device import Tile
from aie.iron.kernel import ExternalFunction
from aie.utils.jit import jit
import aie.utils as aie_utils
from aie.utils.trace import TraceConfig
from aie.utils.trace.events import PortEvent, get_events_for_device


DEV = os.environ.get("NPU_DEV", "npu2")
HFP = Path(os.environ["R61_R34_BUNDLE_HFP"]).expanduser()
DEPTH = int(os.environ.get("DEPTH", "1"))
TRACE_COLS = int(os.environ.get("TRACE_COLS", "4"))
HCLK_MHZ = float(os.environ.get("HCLK_MHZ", "1800"))
TRACE_SIZE = int(os.environ.get("TRACE_SIZE", "16777216"))
DDR_ID = int(os.environ.get("DDR_ID", "4"))
INC = os.environ["MLIR_AIE_INC"]

M, K, N = 256, 768, 1280
PAD_M, PAD_N = 288, 1536
GROUPS, COLUMNS, CORE_ROWS = 3, 8, 4
OUTBLOCKS, R34_BLOCKS_PER_STRIPE = 6, 45
BLOCK_BYTES, C_TILE = 16_384, 2304
R34_INPUT_BYTES = CORE_ROWS * R34_BLOCKS_PER_STRIPE * BLOCK_BYTES
OUTPUT_ELEMS = COLUMNS * OUTBLOCKS * CORE_ROWS * C_TILE


def parse_bundle(path):
    raw = path.read_bytes()
    if len(raw) < 192 or raw[:8] != b"HFOPHFP2":
        raise SystemExit(f"not a version-2 Opus HFP artifact: {path}")
    fields = struct.unpack_from("<18I", raw, 8)
    payload_bytes = struct.unpack_from("<Q", raw, 80)[0]
    segments = struct.unpack_from("<4Q", raw, 160)
    payload = raw[192:]
    if (
        fields[0] != 2
        or fields[1] != 192
        or fields[2] != 1
        or fields[3] != 3
        or fields[5] != 3
        or fields[6] != M
        or fields[9] != COLUMNS
        or fields[14] != BLOCK_BYTES
        or payload_bytes != len(payload)
        or segments[0] != COLUMNS * OUTBLOCKS * GROUPS * BLOCK_BYTES
        or segments[1] != COLUMNS * 9 * BLOCK_BYTES
        or not 0 < segments[2] <= BLOCK_BYTES
        or segments[3] != 0
    ):
        raise SystemExit(f"unsupported R34 bundle geometry: {fields} {segments}")
    if hashlib.sha256(payload).digest() != raw[128:160]:
        raise SystemExit("R34 bundle payload SHA-256 mismatch")
    return payload, segments


BUNDLE, SEGMENTS = parse_bundle(HFP)
BLOCKS_PER_COLUMN = len(BUNDLE) // (COLUMNS * BLOCK_BYTES)
BYTES_PER_COLUMN = BLOCKS_PER_COLUMN * BLOCK_BYTES
if BLOCKS_PER_COLUMN != 28:
    raise SystemExit(f"R34 bundle has {BLOCKS_PER_COLUMN} blocks/column, expected 28")


def canonical_activations():
    rows = np.arange(M, dtype=np.int32)[:, None]
    inner = np.arange(K, dtype=np.int32)[None, :]
    return (((rows * 29 + inner * 17 + 11) % 255) - 127).astype(np.int8)


def pack_r34_activations(values):
    packed = np.zeros(R34_INPUT_BYTES, dtype=np.int8)
    for stripe in range(CORE_ROWS):
        for m_macro in range(3):
            for n_macro in range(5):
                outblock = m_macro * 5 + n_macro
                for group in range(GROUPS):
                    block = outblock * GROUPS + group
                    base = (stripe * R34_BLOCKS_PER_STRIPE + block) * BLOCK_BYTES
                    for lm in range(3):
                        for kt in range(32):
                            for local_row in range(8):
                                row = m_macro * 96 + stripe * 24 + lm * 8 + local_row
                                if row < M:
                                    source = group * 256 + kt * 8
                                    target = base + (lm * 32 + kt) * 64 + local_row * 8
                                    packed[target : target + 8] = values[
                                        row, source : source + 8
                                    ]
                    for local_row in range(24):
                        row = m_macro * 96 + stripe * 24 + local_row
                        scale = np.float32(1.0 if row < M else 0.0)
                        packed[
                            base + 6144 + local_row * 4 : base + 6148 + local_row * 4
                        ] = np.frombuffer(scale.tobytes(), dtype=np.int8)
    return packed


ACTIVATIONS = canonical_activations()
SHARED_INPUT = pack_r34_activations(ACTIVATIONS)
tile_ty = np.ndarray[(BLOCK_BYTES,), np.dtype[np.int8]]
output_tile_ty = np.ndarray[(C_TILE,), np.dtype[np.float32]]
joined_output_ty = np.ndarray[(CORE_ROWS * C_TILE,), np.dtype[np.float32]]
bundle_ty = np.ndarray[(len(BUNDLE),), np.dtype[np.int8]]
input_ty = np.ndarray[(R34_INPUT_BYTES,), np.dtype[np.int8]]
output_ty = np.ndarray[(OUTPUT_ELEMS,), np.dtype[np.float32]]

compute = ExternalFunction(
    "r61_full_qkv_block",
    source_file="r61_full_qkv.cc",
    arg_types=[tile_ty, tile_ty, output_tile_ty, np.int32],
    include_dirs=[INC],
    compile_flags=["-std=c++20", "-O2"],
)
weight_sink = ExternalFunction(
    "r61_weight_sink",
    source_file="r61_weight_sink.cc",
    arg_types=[tile_ty],
    include_dirs=[INC],
    compile_flags=["-std=c++20", "-O2"],
)

CoreEvent = get_events_for_device(DEV).CoreEvent
CORE_EVENTS = [
    PortEvent(CoreEvent.PORT_RUNNING_0, port=WireBundle.DMA, channel=0, master=True),
    PortEvent(CoreEvent.PORT_STALLED_0, port=WireBundle.DMA, channel=0, master=True),
    PortEvent(CoreEvent.PORT_IDLE_0, port=WireBundle.DMA, channel=0, master=True),
    PortEvent(CoreEvent.PORT_RUNNING_1, port=WireBundle.DMA, channel=1, master=True),
    PortEvent(CoreEvent.PORT_STALLED_1, port=WireBundle.DMA, channel=1, master=True),
    PortEvent(CoreEvent.PORT_IDLE_1, port=WireBundle.DMA, channel=1, master=True),
]


@jit(use_cache=True)
def r61(bundle_source, shared_input, output, compute_fn, sink_fn, **_kwargs):
    device = aie_utils.get_current_device()
    host_weights = [
        ObjectFifo(tile_ty, name=f"r61_w_l3_{column}", depth=DEPTH)
        for column in range(COLUMNS)
    ]
    core_weights = [
        host_weights[column]
        .cons()
        .forward(
            tile=Tile(col=column, row=1),
            obj_type=tile_ty,
            name=f"r61_w_l2_{column}",
            depth=DEPTH,
        )
        for column in range(COLUMNS)
    ]
    host_activations = [
        ObjectFifo(tile_ty, name=f"r61_a_l3_{row}", depth=1)
        for row in range(CORE_ROWS)
    ]
    core_activations = [
        host_activations[row]
        .cons()
        .forward(
            tile=Tile(col=row, row=1),
            obj_type=tile_ty,
            name=f"r61_a_l2_{row}",
            depth=1,
        )
        for row in range(CORE_ROWS)
    ]
    host_outputs = [
        ObjectFifo(joined_output_ty, name=f"r61_c_join_{column}", depth=1)
        for column in range(COLUMNS)
    ]
    core_outputs = [
        host_outputs[column]
        .prod()
        .join(
            [row * C_TILE for row in range(CORE_ROWS)],
            tile=Tile(col=column, row=1),
            depths=[1] * CORE_ROWS,
            obj_types=[output_tile_ty] * CORE_ROWS,
            names=[f"r61_c_{column}_{row}" for row in range(CORE_ROWS)],
        )
        for column in range(COLUMNS)
    ]

    def compute_body(a_fifo, w_fifo, out_fifo, block_fn, sink):
        for _outblock in range(OUTBLOCKS):
            result = out_fifo.acquire(1)
            for group in range(GROUPS):
                activation = a_fifo.acquire(1)
                weight = w_fifo.acquire(1)
                block_fn(activation, weight, result, group)
                w_fifo.release(1)
                a_fifo.release(1)
            out_fifo.release(1)
        for _ in range(BLOCKS_PER_COLUMN - OUTBLOCKS * GROUPS):
            weight = w_fifo.acquire(1)
            sink(weight)
            w_fifo.release(1)

    workers = []
    traced = []
    for column in range(COLUMNS):
        for row in range(CORE_ROWS):
            worker = Worker(
                compute_body,
                [
                    core_activations[row].cons(),
                    core_weights[column].cons(),
                    core_outputs[column][row].prod(),
                    compute_fn,
                    sink_fn,
                ],
                tile=Tile(col=column, row=2 + row),
            )
            workers.append(worker)
            if row == 0:
                traced.append(worker)

    runtime = Runtime()
    runtime.enable_trace(
        trace_size=TRACE_SIZE,
        workers=traced[:TRACE_COLS],
        ddr_id=DDR_ID,
        coretile_events=CORE_EVENTS,
    )
    with runtime.sequence(bundle_ty, input_ty, output_ty) as (
        bundle_arg,
        input_arg,
        output_arg,
    ):
        for worker in workers:
            runtime.start(worker)
        bundle_regions = TensorTiler2D.simple_tiler(
            [len(BUNDLE)], [BYTES_PER_COLUMN]
        )
        for column in range(COLUMNS):
            runtime.fill(
                host_weights[column].prod(), bundle_arg, tap=bundle_regions[column]
            )
        for row in range(CORE_ROWS):
            activation_tap = TensorAccessPattern(
                [R34_INPUT_BYTES],
                row * R34_BLOCKS_PER_STRIPE * BLOCK_BYTES,
                [3, 2, GROUPS, BLOCK_BYTES],
                [15 * BLOCK_BYTES, 3 * BLOCK_BYTES, BLOCK_BYTES, 1],
            )
            runtime.fill(host_activations[row].prod(), input_arg, tap=activation_tap)
        output_regions = TensorTiler2D.simple_tiler(
            [OUTPUT_ELEMS], [OUTBLOCKS * CORE_ROWS * C_TILE]
        )
        for column in range(COLUMNS):
            runtime.drain(
                host_outputs[column].cons(),
                output_arg,
                tap=output_regions[column],
                wait=True,
            )
    return Program(device, runtime).resolve_program()


def sext4(value):
    value &= 0xF
    return value - 16 if value & 8 else value


def reference_output():
    weights = np.zeros((GROUPS, 256, PAD_N), dtype=np.int8)
    scales = np.zeros((GROUPS, PAD_N), dtype=np.float32)
    for column in range(COLUMNS):
        stream = BUNDLE[column * BYTES_PER_COLUMN : (column + 1) * BYTES_PER_COLUMN]
        for n_macro in range(2):
            for group in range(GROUPS):
                block_index = n_macro * GROUPS + group
                block = stream[
                    block_index * BLOCK_BYTES : (block_index + 1) * BLOCK_BYTES
                ]
                for jn in range(6):
                    col = n_macro * 768 + column * 96 + jn * 16
                    for kt in range(16):
                        packed = block[
                            (jn * 16 + kt) * 128 : (jn * 16 + kt + 1) * 128
                        ]
                        decoded = np.array(
                            [
                                value
                                for byte in packed
                                for value in (sext4(byte), sext4(byte >> 4))
                            ],
                            dtype=np.int8,
                        ).reshape(16, 16)
                        weights[group, kt * 16 : (kt + 1) * 16, col : col + 16] = decoded
                    scales[group, col : col + 16] = np.frombuffer(
                        block[12288 + jn * 64 : 12288 + (jn + 1) * 64],
                        dtype=np.float32,
                    )
    expected = np.zeros((PAD_M, PAD_N), dtype=np.float32)
    for group in range(GROUPS):
        dots = (
            ACTIVATIONS[:, group * 256 : (group + 1) * 256].astype(np.int32)
            @ weights[group].astype(np.int32)
        )
        expected[:M] = (
            expected[:M] + dots.astype(np.float32) * scales[group]
        ).astype(np.float32)
    return expected


def unpack_physical(values):
    physical = values.reshape(COLUMNS, OUTBLOCKS, CORE_ROWS, C_TILE)
    output = np.zeros((PAD_M, PAD_N), dtype=np.float32)
    for column in range(COLUMNS):
        for outblock in range(OUTBLOCKS):
            m_macro, n_macro = divmod(outblock, 2)
            for row_stripe in range(CORE_ROWS):
                tile = physical[column, outblock, row_stripe]
                for im in range(6):
                    for jn in range(6):
                        for local_row in range(4):
                            row = m_macro * 96 + row_stripe * 24 + im * 4 + local_row
                            col = n_macro * 768 + column * 96 + jn * 16
                            source = (im * 6 + jn) * 64 + local_row * 16
                            output[row, col : col + 16] = tile[source : source + 16]
    return output


def paired_intervals(events, name):
    by_pid = defaultdict(list)
    for event in events:
        if event.get("name") == name and "ts" in event:
            by_pid[event.get("pid", 0)].append(event)
    intervals = defaultdict(list)
    for pid, records in by_pid.items():
        records.sort(key=lambda event: event["ts"])
        opened = None
        for event in records:
            if event.get("ph") == "B":
                opened = event["ts"]
            elif event.get("ph") == "E" and opened is not None:
                intervals[pid].append((opened, event["ts"]))
                opened = None
    return intervals


def interval_sum(intervals):
    return sum(end - begin for begin, end in intervals)


bundle_tensor = tensor(np.frombuffer(BUNDLE, dtype=np.int8).copy(), dtype=np.int8)
input_tensor = tensor(SHARED_INPUT.copy(), dtype=np.int8)
output = zeros(OUTPUT_ELEMS, dtype=np.float32)
trace_dir = Path(os.environ.get("R61_TRACE_DIR", "~/.hipfire/r61-traces")).expanduser()
trace_dir.mkdir(parents=True, exist_ok=True)
trace_txt = str(trace_dir / "full-qkv.txt")
trace_json = str(trace_dir / "full-qkv.json")
trace = TraceConfig(trace_size=TRACE_SIZE, trace_file=trace_txt, ddr_id=DDR_ID)

started = time.perf_counter()
r61(
    bundle_tensor,
    input_tensor,
    output,
    compute,
    weight_sink,
    trace_config=trace,
)
host_seconds = time.perf_counter() - started
got = unpack_physical(output.numpy())
expected = reference_output()
absolute = np.abs(got - expected)
tolerance = 1e-4 + np.abs(expected) * 1e-5
mismatches = int(np.count_nonzero(absolute > tolerance))
if mismatches:
    index = np.argwhere(absolute > tolerance)[0]
    row, col = map(int, index)
    raise SystemExit(
        f"full QKV mismatch count={mismatches} first=({row},{col}) "
        f"got={got[row,col]} expected={expected[row,col]} abs={absolute[row,col]}"
    )

parse = subprocess.run(
    [
        sys.executable,
        "-m",
        "aie.utils.trace.parse",
        "--input",
        trace_txt,
        "--mlir",
        trace.physical_mlir_path,
        "--output",
        trace_json,
    ],
    capture_output=True,
    text=True,
    check=False,
)
if parse.returncode != 0 or not Path(trace_json).is_file():
    raise SystemExit(f"trace parse failed rc={parse.returncode}\n{parse.stderr[-800:]}")
events = json.loads(Path(trace_json).read_text(encoding="utf-8"))
running_by_channel = {
    channel: paired_intervals(events, f"PORT_RUNNING_{channel}")
    for channel in (0, 1)
}


def channel_span(intervals):
    stamps = [stamp for records in intervals.values() for pair in records for stamp in pair]
    return max(stamps) - min(stamps) if stamps else 0


weight_dma_channel = max(
    running_by_channel,
    key=lambda channel: channel_span(running_by_channel[channel]),
)
running = running_by_channel[weight_dma_channel]
stalled = paired_intervals(events, f"PORT_STALLED_{weight_dma_channel}")
idle = paired_intervals(events, f"PORT_IDLE_{weight_dma_channel}")
global_span = channel_span(running)
if not global_span:
    raise SystemExit("trace contains no receive-port running intervals")
span_seconds = global_span / (HCLK_MHZ * 1e6)


def fractions(samples):
    values = []
    for pid, intervals in running.items():
        local_span = max(end for _, end in intervals) - min(begin for begin, _ in intervals)
        if local_span:
            values.append(interval_sum(samples.get(pid, ())) / local_span)
    return sum(values) / len(values)


xrt = subprocess.run(
    ["xrt-smi", "--version"], capture_output=True, text=True, check=False
).stdout


def version_field(label):
    match = re.search(rf"{re.escape(label)}\s*:\s*([^\n]+)", xrt)
    return match.group(1).strip() if match else "unknown"


repo_root = Path(__file__).resolve().parents[3]
git_commit = subprocess.run(
    ["git", "rev-parse", "HEAD"],
    cwd=repo_root,
    capture_output=True,
    text=True,
    check=True,
).stdout.strip()
benchmark_sha = hashlib.sha256()
for path in (
    Path(__file__),
    Path(__file__).with_name("r61_full_qkv.cc"),
    Path(__file__).with_name("r61_weight_sink.cc"),
):
    benchmark_sha.update(path.name.encode())
    benchmark_sha.update(path.read_bytes())

useful_macs = M * K * N
padded_macs = PAD_M * K * PAD_N
row = {
    "mode": "FULL_R34_SHARED_INPUT_OQ4_QKV",
    "profile": "embeddinggemma-r34-resident-bundle-v1",
    "git_commit": git_commit,
    "git_dirty": int(bool(subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip())),
    "benchmark_sha256": benchmark_sha.hexdigest(),
    "host": socket.gethostname(),
    "npu_device": DEV,
    "xrt_version": version_field("Version"),
    "amdxdna_version": version_field("amdxdna Version"),
    "firmware_version": version_field("NPU Firmware Version"),
    "bundle_file": str(HFP),
    "bundle_payload_sha256": hashlib.sha256(BUNDLE).hexdigest(),
    "m": M,
    "k": K,
    "n": N,
    "padded_m": PAD_M,
    "padded_n": PAD_N,
    "columns": COLUMNS,
    "core_rows": CORE_ROWS,
    "fifo_depth": DEPTH,
    "bundle_blocks_per_column": BLOCKS_PER_COLUMN,
    "wire_bytes": len(BUNDLE),
    "output_bytes": OUTPUT_ELEMS * 4,
    "weight_dma_channel": weight_dma_channel,
    "global_span_cycles": global_span,
    "hclk_mhz": HCLK_MHZ,
    "wire_gbs": len(BUNDLE) / span_seconds / 1e9,
    "useful_macs": useful_macs,
    "padded_macs": padded_macs,
    "useful_tops": 2 * useful_macs / span_seconds / 1e12,
    "padded_tops": 2 * padded_macs / span_seconds / 1e12,
    "mean_receive_busy": fractions(running),
    "mean_receive_stalled": fractions(stalled),
    "mean_receive_idle": fractions(idle),
    "host_call_ms": host_seconds * 1e3,
    "checked_outputs": M * N,
    "padded_outputs": PAD_M * PAD_N,
    "mismatches": mismatches,
    "max_abs": float(absolute.max()),
    "full_qkv_oracle": "pass",
    "global_weight_reorder_in_kernel": False,
}
print("R61_JSON " + json.dumps(row, sort_keys=True))
