#!/usr/bin/env python3
"""R60: first exact OQ4 MMUL from the R34 bundle and shared input layout."""

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


DEV = os.environ.get("NPU_DEV", "npu2")
STAGE = os.environ.get("R60_STAGE", "FIRST_MMUL").upper()
if STAGE not in ("FIRST_MMUL", "FULL_K_GROUP_STAGE1", "THREE_GROUP_SCALED_STAGE2"):
    raise SystemExit(
        "R60_STAGE must be FIRST_MMUL, FULL_K_GROUP_STAGE1, or "
        "THREE_GROUP_SCALED_STAGE2"
    )
HFP = Path(os.environ["R60_R34_BUNDLE_HFP"]).expanduser()
DEPTH = int(os.environ.get("DEPTH", "4"))
FANOUT = int(os.environ.get("FANOUT", "4"))
TRACE_COLS = int(os.environ.get("TRACE_COLS", "4"))
HCLK_MHZ = float(os.environ.get("HCLK_MHZ", "1800"))
TRACE_SIZE = int(os.environ.get("TRACE_SIZE", "8388608"))
DDR_ID = int(os.environ.get("DDR_ID", "4"))
INC = os.environ["MLIR_AIE_INC"]

M, K, GROUPS = 256, 768, 3
COLUMNS, BLOCK_BYTES = 8, 16_384
R34_BLOCKS_PER_STRIPE = 45
R34_INPUT_BYTES = 4 * R34_BLOCKS_PER_STRIPE * BLOCK_BYTES
OUTPUT_WORDS = 64


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
        or len(payload) % (COLUMNS * BLOCK_BYTES) != 0
        or segments[0] != 8 * 18 * BLOCK_BYTES
        or segments[1] != 8 * 9 * BLOCK_BYTES
        or not 0 < segments[2] <= BLOCK_BYTES
        or segments[3] != 0
    ):
        raise SystemExit(f"unsupported R34 resident bundle geometry: {fields} {segments}")
    if hashlib.sha256(payload).digest() != raw[128:160]:
        raise SystemExit("R34 bundle payload SHA-256 mismatch")
    return raw, payload, fields, segments


RAW_HFP, BUNDLE, FIELDS, SEGMENTS = parse_bundle(HFP)
BLOCKS_PER_COLUMN = len(BUNDLE) // (COLUMNS * BLOCK_BYTES)
BYTES_PER_COLUMN = BLOCKS_PER_COLUMN * BLOCK_BYTES
if BLOCKS_PER_COLUMN != 28:
    raise SystemExit(f"R34 bundle has {BLOCKS_PER_COLUMN} blocks/column, expected 28")
if not 1 <= FANOUT <= 4 or not 1 <= TRACE_COLS <= COLUMNS:
    raise SystemExit("FANOUT must be 1..4 and TRACE_COLS must be 1..8")


def canonical_activations():
    rows = np.arange(M, dtype=np.int32)[:, None]
    inner = np.arange(K, dtype=np.int32)[None, :]
    return (((rows * 29 + inner * 17 + 11) % 255) - 127).astype(np.int8)


def pack_r34_activations(values):
    packed = np.zeros(R34_INPUT_BYTES, dtype=np.int8)
    for stripe in range(4):
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
                                    packed[target : target + 8] = values[row, source : source + 8]
                    for local_row in range(24):
                        row = m_macro * 96 + stripe * 24 + local_row
                        scale = np.float32(1.0 if row < M else 0.0)
                        packed[base + 6144 + local_row * 4 : base + 6148 + local_row * 4] = (
                            np.frombuffer(scale.tobytes(), dtype=np.int8)
                        )
    return packed


ACTIVATIONS = canonical_activations()
SHARED_INPUT = pack_r34_activations(ACTIVATIONS)
tile_ty = np.ndarray[(BLOCK_BYTES,), np.dtype[np.int8]]
bundle_ty = np.ndarray[(len(BUNDLE),), np.dtype[np.int8]]
input_ty = np.ndarray[(R34_INPUT_BYTES,), np.dtype[np.int8]]
OUTPUT_DTYPE = np.float32 if STAGE == "THREE_GROUP_SCALED_STAGE2" else np.int32
result_ty = np.ndarray[(OUTPUT_WORDS,), np.dtype[OUTPUT_DTYPE]]
output_ty = np.ndarray[(COLUMNS * OUTPUT_WORDS,), np.dtype[OUTPUT_DTYPE]]

compute = ExternalFunction(
    {
        "FIRST_MMUL": "r60_first_mmul",
        "FULL_K_GROUP_STAGE1": "r60_fullk_group",
        "THREE_GROUP_SCALED_STAGE2": "r60_scaled_group",
    }[STAGE],
    source_file={
        "FIRST_MMUL": "r60_first_mmul.cc",
        "FULL_K_GROUP_STAGE1": "r60_fullk_group.cc",
        "THREE_GROUP_SCALED_STAGE2": "r60_scaled_groups.cc",
    }[STAGE],
    arg_types=[tile_ty, tile_ty, result_ty]
    + ([np.int32] if STAGE == "THREE_GROUP_SCALED_STAGE2" else []),
    include_dirs=[INC],
    compile_flags=["-std=c++20", "-O2"],
)
weight_sink = ExternalFunction(
    "r60_weight_sink",
    source_file="r60_weight_sink.cc",
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
def r60(bundle_source, shared_input, output, compute_fn, sink_fn, **_kwargs):
    device = aie_utils.get_current_device()
    host_weights = [
        ObjectFifo(tile_ty, name=f"r60_w_l3_{column}", depth=DEPTH)
        for column in range(COLUMNS)
    ]
    core_weights = [
        host_weights[column]
        .cons()
        .forward(
            tile=Tile(col=column, row=1),
            obj_type=tile_ty,
            name=f"r60_w_l2_{column}",
            depth=DEPTH,
        )
        for column in range(COLUMNS)
    ]
    host_activations = [
        ObjectFifo(tile_ty, name=f"r60_a_l3_{column}", depth=1)
        for column in range(COLUMNS)
    ]
    core_activations = [
        host_activations[column]
        .cons()
        .forward(
            tile=Tile(col=column, row=1),
            obj_type=tile_ty,
            name=f"r60_a_l2_{column}",
            depth=1,
        )
        for column in range(COLUMNS)
    ]
    host_outputs = [
        ObjectFifo(result_ty, name=f"r60_c_{column}", depth=1)
        for column in range(COLUMNS)
    ]

    def compute_body(a_fifo, w_fifo, out_fifo, compute, sink):
        result = out_fifo.acquire(1)
        if STAGE == "THREE_GROUP_SCALED_STAGE2":
            for group in range(GROUPS):
                activation = a_fifo.acquire(1)
                weight = w_fifo.acquire(1)
                compute(activation, weight, result, group)
                w_fifo.release(1)
                a_fifo.release(1)
            consumed = GROUPS
        else:
            activation = a_fifo.acquire(1)
            weight = w_fifo.acquire(1)
            compute(activation, weight, result)
            w_fifo.release(1)
            a_fifo.release(1)
            consumed = 1
        for _ in range_(BLOCKS_PER_COLUMN - consumed):
            weight = w_fifo.acquire(1)
            sink(weight)
            w_fifo.release(1)
        out_fifo.release(1)

    def sink_body(w_fifo, sink):
        for _ in range_(BLOCKS_PER_COLUMN):
            weight = w_fifo.acquire(1)
            sink(weight)
            w_fifo.release(1)

    workers = []
    traced = []
    for column in range(COLUMNS):
        worker = Worker(
            compute_body,
            [
                core_activations[column].cons(),
                core_weights[column].cons(),
                host_outputs[column].prod(),
                compute_fn,
                sink_fn,
            ],
            tile=Tile(col=column, row=2),
        )
        workers.append(worker)
        traced.append(worker)
        for row in range(1, FANOUT):
            workers.append(
                Worker(
                    sink_body,
                    [core_weights[column].cons(), sink_fn],
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
        activation_bytes = (
            GROUPS * BLOCK_BYTES
            if STAGE == "THREE_GROUP_SCALED_STAGE2"
            else BLOCK_BYTES
        )
        activation_region = TensorTiler2D.simple_tiler(
            [R34_INPUT_BYTES], [activation_bytes]
        )[0]
        output_regions = TensorTiler2D.simple_tiler(
            [COLUMNS * OUTPUT_WORDS], [OUTPUT_WORDS]
        )
        for column in range(COLUMNS):
            runtime.fill(
                host_weights[column].prod(), bundle_arg, tap=bundle_regions[column]
            )
            runtime.fill(
                host_activations[column].prod(), input_arg, tap=activation_region
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


def expected_for_column(column):
    stream = BUNDLE[column * BYTES_PER_COLUMN : (column + 1) * BYTES_PER_COLUMN]
    if STAGE == "THREE_GROUP_SCALED_STAGE2":
        result = np.zeros((4, 16), dtype=np.float32)
        for group in range(GROUPS):
            weight_block = stream[
                group * BLOCK_BYTES : (group + 1) * BLOCK_BYTES
            ]
            dots = np.zeros((4, 16), dtype=np.int32)
            for kt in range(16):
                packed = weight_block[kt * 128 : (kt + 1) * 128]
                decoded = np.array(
                    [
                        value
                        for byte in packed
                        for value in (sext4(byte), sext4(byte >> 4))
                    ],
                    dtype=np.int32,
                ).reshape(16, 16)
                start = group * 256 + kt * 16
                a = ACTIVATIONS[:4, start : start + 16].astype(np.int32)
                dots += a @ decoded
            weight_scales = np.frombuffer(
                weight_block[12288 : 12288 + 64], dtype=np.float32
            )
            result = (
                result + dots.astype(np.float32) * weight_scales
            ).astype(np.float32)
        return result
    result = np.zeros((4, 16), dtype=np.int32)
    tiles = 1 if STAGE == "FIRST_MMUL" else 16
    for kt in range(tiles):
        packed = stream[kt * 128 : (kt + 1) * 128]
        decoded = np.array(
            [value for byte in packed for value in (sext4(byte), sext4(byte >> 4))],
            dtype=np.int32,
        ).reshape(16, 16)
        a = ACTIVATIONS[:4, kt * 16 : (kt + 1) * 16].astype(np.int32)
        result += a @ decoded
    return result


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
output = zeros(COLUMNS * OUTPUT_WORDS, dtype=OUTPUT_DTYPE)
trace_dir = Path(os.environ.get("R60_TRACE_DIR", "~/.hipfire/r60-traces")).expanduser()
trace_dir.mkdir(parents=True, exist_ok=True)
trace_stem = STAGE.lower().replace("_", "-")
trace_txt = str(trace_dir / f"{trace_stem}.txt")
trace_json = str(trace_dir / f"{trace_stem}.json")
trace = TraceConfig(trace_size=TRACE_SIZE, trace_file=trace_txt, ddr_id=DDR_ID)

started = time.perf_counter()
r60(
    bundle_tensor,
    input_tensor,
    output,
    compute,
    weight_sink,
    trace_config=trace,
)
host_seconds = time.perf_counter() - started
actual = output.numpy().reshape(COLUMNS, OUTPUT_WORDS)
for column in range(COLUMNS):
    expected = expected_for_column(column).reshape(-1)
    matches = (
        np.allclose(actual[column], expected, rtol=1e-5, atol=1e-4)
        if STAGE == "THREE_GROUP_SCALED_STAGE2"
        else np.array_equal(actual[column], expected)
    )
    if not matches:
        mismatch = int(
            np.flatnonzero(
                ~np.isclose(actual[column], expected, rtol=1e-5, atol=1e-4)
            )[0]
        )
        raise SystemExit(
            f"column {column} {STAGE} mismatch at {mismatch}: "
            f"{actual[column, mismatch]} != {expected[mismatch]}"
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


weight_dma_channel = max(running_by_channel, key=lambda channel: channel_span(running_by_channel[channel]))
running = running_by_channel[weight_dma_channel]
stalled = paired_intervals(events, f"PORT_STALLED_{weight_dma_channel}")
idle = paired_intervals(events, f"PORT_IDLE_{weight_dma_channel}")
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
benchmark_sha = hashlib.sha256()
for path in (
    Path(__file__),
    Path(__file__).with_name("r60_first_mmul.cc"),
    Path(__file__).with_name("r60_fullk_group.cc"),
    Path(__file__).with_name("r60_scaled_groups.cc"),
    Path(__file__).with_name("r60_weight_sink.cc"),
):
    benchmark_sha.update(path.name.encode())
    benchmark_sha.update(path.read_bytes())

wire_bytes = len(BUNDLE)
row = {
    "mode": STAGE,
    "profile": "embeddinggemma-r34-resident-bundle-v1",
    "git_commit": git_commit,
    "git_dirty": int(bool(subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=repo_root, capture_output=True, text=True, check=True
    ).stdout.strip())),
    "benchmark_sha256": benchmark_sha.hexdigest(),
    "host": socket.gethostname(),
    "npu_device": DEV,
    "xrt_version": version_field("Version"),
    "amdxdna_version": version_field("amdxdna Version"),
    "firmware_version": version_field("NPU Firmware Version"),
    "bundle_file": str(HFP),
    "bundle_payload_sha256": hashlib.sha256(BUNDLE).hexdigest(),
    "shared_input_bytes": R34_INPUT_BYTES,
    "shared_input_tile_offset": 0,
    "columns": COLUMNS,
    "fanout": FANOUT,
    "trace_columns": len(running),
    "fifo_depth": DEPTH,
    "bundle_blocks_per_column": BLOCKS_PER_COLUMN,
    "wire_bytes": wire_bytes,
    "weight_dma_channel": weight_dma_channel,
    "global_span_cycles": global_span,
    "hclk_mhz": HCLK_MHZ,
    "wire_gbs": wire_bytes / span_seconds / 1e9,
    "mean_receive_busy": fractions(running),
    "mean_receive_stalled": fractions(stalled),
    "mean_receive_idle": fractions(idle),
    "host_call_ms": host_seconds * 1e3,
    "first_mmul_oracle": "pass" if STAGE == "FIRST_MMUL" else "not-run",
    "full_k_group_oracle": "pass" if STAGE == "FULL_K_GROUP_STAGE1" else "not-run",
    "three_group_scaled_oracle": (
        "pass" if STAGE == "THREE_GROUP_SCALED_STAGE2" else "not-run"
    ),
    "global_weight_reorder_in_kernel": False,
}
print("R60_JSON " + json.dumps(row, sort_keys=True))
