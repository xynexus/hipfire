#!/usr/bin/env python3
"""Trace-timed NIBBLE_DECODE over a real packed Opus `.rdna2.hfp` stream."""

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
from aie.helpers.taplib import TensorTiler2D
from aie.iron import ObjectFifo, Program, Runtime, Worker, tensor, zeros
from aie.iron.controlflow import range_
from aie.iron.device import Tile
from aie.iron.kernel import ExternalFunction
from aie.utils.jit import jit
import aie.utils as aie_utils
from aie.utils.trace import TraceConfig
from aie.utils.trace.events import PortEvent, get_events_for_device

from test_decode_oracle import decode_low_high, lane_sums


DEV = os.environ.get("NPU_DEV", "npu2")
MODE = os.environ.get("R58_MODE", "NIBBLE_DECODE").upper()
if MODE not in (
    "PACKED_FEED_ONLY",
    "NIBBLE_DECODE",
    "COMPUTE_STAGE1",
    "COMPUTE_STAGE2",
):
    raise SystemExit(
        "R58_MODE must be PACKED_FEED_ONLY, NIBBLE_DECODE, COMPUTE_STAGE1, "
        "or COMPUTE_STAGE2"
    )
HFP = Path(os.environ["R58_LAYOUT_FILE"]).expanduser()
DEPTH = int(os.environ.get("DEPTH", "4"))
FANOUT = int(os.environ.get("FANOUT", "4"))
TRACE_COLS = int(os.environ.get("TRACE_COLS", "4"))
HCLK_MHZ = float(os.environ.get("HCLK_MHZ", "1800"))
TRACE_SIZE = int(os.environ.get("TRACE_SIZE", "4194304"))
DDR_ID = int(os.environ.get("DDR_ID", "4"))
INC = os.environ["MLIR_AIE_INC"]

raw = HFP.read_bytes()
if len(raw) < 192 or raw[:8] != b"HFOPHFP2":
    raise SystemExit("R58_LAYOUT_FILE is not a version-2 Opus HFP artifact")
fields = struct.unpack_from("<18I", raw, 8)
(
    version,
    header_bytes,
    encoding,
    layout,
    quant_type,
    flags,
    m,
    k,
    n,
    columns,
    groups,
    m_macros,
    n_macros,
    outblocks,
    tile_bytes,
    data_bytes,
    scale_offset,
    scale_values,
) = fields
payload_bytes = struct.unpack_from("<Q", raw, 80)[0]
source_sha256 = raw[96:128].hex()
payload_sha256 = raw[128:160]
payload = raw[header_bytes:]
if (
    version != 2
    or header_bytes != 192
    or encoding != 1
    or layout != 1
    or quant_type not in (33, 34)
    or flags != 0
    or columns != 8
    or tile_bytes != 16_384
    or data_bytes != 12_288
    or scale_offset != data_bytes
    or scale_values != 96
    or payload_bytes != len(payload)
    or len(payload) != columns * groups * outblocks * tile_bytes
):
    raise SystemExit(f"unsupported Opus HFP geometry: fields={fields} payload={len(payload)}")
if hashlib.sha256(payload).digest() != payload_sha256:
    raise SystemExit("Opus HFP payload SHA-256 mismatch")
if not 1 <= FANOUT <= 4:
    raise SystemExit("FANOUT must be in 1..=4")
if not 1 <= TRACE_COLS <= columns:
    raise SystemExit("TRACE_COLS must be in 1..=columns")

blocks_per_column = groups * outblocks
bytes_per_column = blocks_per_column * tile_bytes
guard_words = 80
input_ty = np.ndarray[(len(payload),), np.dtype[np.int8]]
tile_ty = np.ndarray[(tile_bytes,), np.dtype[np.int8]]
guard_ty = np.ndarray[(guard_words,), np.dtype[np.int32]]
output_ty = np.ndarray[(columns * guard_words,), np.dtype[np.int32]]

guard_name = {
    "PACKED_FEED_ONLY": "r58_feed_guard",
    "NIBBLE_DECODE": "r58_decode_guard",
    "COMPUTE_STAGE1": "r58_compute_stage1",
    "COMPUTE_STAGE2": "r58_compute_stage2",
}[MODE]
guard_source = {
    "PACKED_FEED_ONLY": "r58_feed_guard.cc",
    "NIBBLE_DECODE": "r58_nibble_decode.cc",
    "COMPUTE_STAGE1": "r58_compute_stage1.cc",
    "COMPUTE_STAGE2": "r58_compute_stage2.cc",
}[MODE]
sink_name = {
    "PACKED_FEED_ONLY": "r58_feed_sink",
    "NIBBLE_DECODE": "r58_decode_sink",
    "COMPUTE_STAGE1": "r58_compute_stage1_sink",
    "COMPUTE_STAGE2": "r58_compute_stage2_sink",
}[MODE]
sink_source = {
    "PACKED_FEED_ONLY": "r58_feed_sink.cc",
    "NIBBLE_DECODE": "r58_sink.cc",
    "COMPUTE_STAGE1": "r58_compute_stage1_sink.cc",
    "COMPUTE_STAGE2": "r58_compute_stage2_sink.cc",
}[MODE]
decode = ExternalFunction(
    guard_name,
    source_file=guard_source,
    arg_types=[tile_ty, guard_ty],
    include_dirs=[INC],
    compile_flags=[
        "-std=c++20",
        "-O2",
        f"-DTILE_BYTES={tile_bytes}",
        f"-DDATA_BYTES={data_bytes}",
    ],
)
zero_guard = ExternalFunction(
    "r58_zero_guard",
    source_file="r58_zero.cc",
    arg_types=[guard_ty],
    include_dirs=[INC],
    compile_flags=["-std=c++20", "-O2"],
)
decode_sink = ExternalFunction(
    sink_name,
    source_file=sink_source,
    arg_types=[tile_ty],
    include_dirs=[INC],
    compile_flags=[
        "-std=c++20",
        "-O2",
        f"-DTILE_BYTES={tile_bytes}",
        f"-DDATA_BYTES={data_bytes}",
    ],
)

CoreEvent = get_events_for_device(DEV).CoreEvent
CORE_EVENTS = [
    PortEvent(CoreEvent.PORT_RUNNING_0, port=WireBundle.DMA, channel=0, master=True),
    PortEvent(CoreEvent.PORT_STALLED_0, port=WireBundle.DMA, channel=0, master=True),
    PortEvent(CoreEvent.PORT_IDLE_0, port=WireBundle.DMA, channel=0, master=True),
]


@jit(use_cache=True)
def nibble_decode(source, output, decode_fn, zero_fn, sink_fn, **_kwargs):
    device = aie_utils.get_current_device()
    host_inputs = [
        ObjectFifo(tile_ty, name=f"w4_l3_{column}", depth=DEPTH)
        for column in range(columns)
    ]
    core_inputs = [
        host_inputs[column]
        .cons()
        .forward(
            tile=Tile(col=column, row=1),
            obj_type=tile_ty,
            name=f"w4_l2_{column}",
            depth=DEPTH,
        )
        for column in range(columns)
    ]
    host_outputs = [
        ObjectFifo(guard_ty, name=f"decode_guard_{column}", depth=1)
        for column in range(columns)
    ]

    def guard_body(in_fifo, out_fifo, decoder, zeroer):
        guard = out_fifo.acquire(1)
        zeroer(guard)
        for _ in range_(blocks_per_column):
            tile = in_fifo.acquire(1)
            decoder(tile, guard)
            in_fifo.release(1)
        out_fifo.release(1)

    def sink_body(in_fifo, decoder):
        for _ in range_(blocks_per_column):
            tile = in_fifo.acquire(1)
            decoder(tile)
            in_fifo.release(1)

    workers = []
    traced = []
    for column in range(columns):
        worker = Worker(
            guard_body,
            [core_inputs[column].cons(), host_outputs[column].prod(), decode_fn, zero_fn],
            tile=Tile(col=column, row=2),
        )
        workers.append(worker)
        traced.append(worker)
        for row in range(1, FANOUT):
            workers.append(
                Worker(
                    sink_body,
                    [core_inputs[column].cons(), sink_fn],
                    tile=Tile(col=column, row=2 + row),
                )
            )

    runtime = Runtime()
    runtime.enable_trace(
        trace_size=TRACE_SIZE,
        workers=traced[:TRACE_COLS],
        ddr_id=DDR_ID,
        coretile_events=CORE_EVENTS,
    )
    with runtime.sequence(input_ty, output_ty) as (source_arg, output_arg):
        for worker in workers:
            runtime.start(worker)
        input_regions = TensorTiler2D.simple_tiler([len(payload)], [bytes_per_column])
        output_regions = TensorTiler2D.simple_tiler(
            [columns * guard_words], [guard_words]
        )
        for column in range(columns):
            runtime.fill(host_inputs[column].prod(), source_arg, tap=input_regions[column])
        for column in range(columns):
            runtime.drain(
                host_outputs[column].cons(),
                output_arg,
                tap=output_regions[column],
                wait=True,
            )
    return Program(device, runtime).resolve_program()


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


expected_lanes = []
expected_first = []
expected_compute = []
for column in range(columns):
    stream = payload[column * bytes_per_column : (column + 1) * bytes_per_column]
    expected_lanes.append(lane_sums(stream, tile_bytes=tile_bytes, data_bytes=data_bytes))
    last_tile = stream[-tile_bytes:]
    expected_first.append(decode_low_high(last_tile[:32]))
    column_sums = [0] * 16
    for tile_offset in range(0, len(stream), tile_bytes):
        data = stream[tile_offset : tile_offset + data_bytes]
        decoded = decode_low_high(data)
        for weight_tile in range(0, len(decoded), 256):
            values = decoded[weight_tile : weight_tile + 256]
            for output_column in range(16):
                column_sums[output_column] += sum(
                    values[row * 16 + output_column] for row in range(16)
                )
    expected_compute.append(column_sums * 4)

source = tensor(np.frombuffer(payload, dtype=np.int8).copy(), dtype=np.int8)
output = zeros(columns * guard_words, dtype=np.int32)
trace_dir = Path(os.environ.get("R58_TRACE_DIR", "~/.hipfire/r58-traces")).expanduser()
trace_dir.mkdir(parents=True, exist_ok=True)
trace_txt = str(trace_dir / "nibble-decode.txt")
trace_json = str(trace_dir / "nibble-decode.json")
trace = TraceConfig(trace_size=TRACE_SIZE, trace_file=trace_txt, ddr_id=DDR_ID)

started = time.perf_counter()
nibble_decode(source, output, decode, zero_guard, decode_sink, trace_config=trace)
host_seconds = time.perf_counter() - started
actual = output.numpy().reshape(columns, guard_words)
for column in range(columns):
    if MODE == "NIBBLE_DECODE":
        if actual[column, :64].tolist() != expected_lanes[column]:
            raise SystemExit(f"column {column} lane-sum mismatch")
        decoded_bytes = actual[column, 64:].view(np.int8)[:64].tolist()
        if decoded_bytes != expected_first[column]:
            raise SystemExit(f"column {column} byte-exact first-vector mismatch")
    elif MODE == "PACKED_FEED_ONLY":
        stream = payload[column * bytes_per_column : (column + 1) * bytes_per_column]
        total = int(
            np.sum(
                np.frombuffer(stream[-tile_bytes : -tile_bytes + 64], dtype=np.int8),
                dtype=np.int64,
            )
        )
        expected = (total + 128) % 256 - 128
        if actual[column, 0] != expected:
            raise SystemExit(f"column {column} feed guard mismatch")
    else:
        coefficient = 1 if MODE == "COMPUTE_STAGE1" else sum(range(1, 7))
        expected = [value * coefficient for value in expected_compute[column]]
        if actual[column, :64].tolist() != expected:
            raise SystemExit(f"column {column} {MODE.lower()} MMUL mismatch")

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
running = paired_intervals(events, "PORT_RUNNING_0")
stalled = paired_intervals(events, "PORT_STALLED_0")
idle = paired_intervals(events, "PORT_IDLE_0")
timestamps = [stamp for intervals in running.values() for pair in intervals for stamp in pair]
if not timestamps:
    raise SystemExit("trace contains no receive-port running intervals")
global_span = max(timestamps) - min(timestamps)
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
    ["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True, check=True
).stdout.strip()
git_dirty = int(bool(subprocess.run(
    ["git", "status", "--porcelain", "--untracked-files=no"],
    cwd=repo_root, capture_output=True, text=True, check=True
).stdout.strip()))
benchmark_sha = hashlib.sha256()
for path in (
    Path(__file__),
    Path(__file__).with_name("r58_nibble_decode.cc"),
    Path(__file__).with_name("r58_zero.cc"),
    Path(__file__).with_name("r58_sink.cc"),
    Path(__file__).with_name("r58_feed_guard.cc"),
    Path(__file__).with_name("r58_feed_sink.cc"),
    Path(__file__).with_name("r58_compute_stage1.cc"),
    Path(__file__).with_name("r58_compute_stage1_sink.cc"),
    Path(__file__).with_name("r58_compute_stage2.cc"),
    Path(__file__).with_name("r58_compute_stage2_sink.cc"),
    Path(__file__).with_name("test_decode_oracle.py"),
):
    benchmark_sha.update(path.name.encode())
    benchmark_sha.update(path.read_bytes())

wire_bytes = len(payload)
packed_data_bytes = columns * blocks_per_column * data_bytes
decoded_bytes = packed_data_bytes * 2 * FANOUT
mmul_reuse = {
    "COMPUTE_STAGE1": 1,
    "COMPUTE_STAGE2": 6,
}.get(MODE, 0)
mmul_count = packed_data_bytes // 128 * FANOUT * mmul_reuse
macs = mmul_count * 1024
row = {
    "mode": MODE,
    "profile": "embeddinggemma-opus-qkv-whole-scaled-v1",
    "git_commit": git_commit,
    "git_dirty": git_dirty,
    "benchmark_sha256": benchmark_sha.hexdigest(),
    "host": socket.gethostname(),
    "npu_device": DEV,
    "xrt_version": version_field("Version"),
    "amdxdna_version": version_field("amdxdna Version"),
    "firmware_version": version_field("NPU Firmware Version"),
    "power_mode": os.environ.get("R58_POWER_MODE", "default-unspecified"),
    "concurrent_load": os.environ.get("R58_CONCURRENT_LOAD", "idle"),
    "layout_file": str(HFP),
    "layout_source_sha256": source_sha256,
    "layout_payload_sha256": payload_sha256.hex(),
    "m": m,
    "k": k,
    "n": n,
    "columns": columns,
    "groups": groups,
    "m_macros": m_macros,
    "n_macros": n_macros,
    "outblocks": outblocks,
    "blocks_per_column": blocks_per_column,
    "tile_bytes": tile_bytes,
    "data_bytes_per_tile": data_bytes,
    "scale_values_per_tile": scale_values,
    "fanout": FANOUT,
    "trace_columns": len(running),
    "fifo_depth": DEPTH,
    "wire_bytes": wire_bytes,
    "packed_data_bytes": packed_data_bytes,
    "decoded_logical_bytes": decoded_bytes,
    "mmul_count": mmul_count,
    "macs": macs,
    "global_span_cycles": global_span,
    "hclk_mhz": HCLK_MHZ,
    "wire_gbs": wire_bytes / span_seconds / 1e9,
    "packed_data_gbs": packed_data_bytes / span_seconds / 1e9,
    "decoded_logical_gbs": decoded_bytes / span_seconds / 1e9,
    "tmac_s": macs / span_seconds / 1e12,
    "tops": 2 * macs / span_seconds / 1e12,
    "mean_receive_busy": fractions(running),
    "mean_receive_stalled": fractions(stalled),
    "mean_receive_idle": fractions(idle),
    "host_call_ms": host_seconds * 1e3,
    "lane_sum_oracle": "pass" if MODE == "NIBBLE_DECODE" else "not-run",
    "byte_exact_vector_oracle": "pass" if MODE == "NIBBLE_DECODE" else "not-run",
    "mmul_oracle": "pass" if MODE.startswith("COMPUTE_STAGE") else "not-run",
}
print("R58_JSON " + json.dumps(row, sort_keys=True))
