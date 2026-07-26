#!/usr/bin/env python3
"""R57 steps 0/1: trace-timed production-shaped R34 weight DMA."""

import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time
from collections import defaultdict

import numpy as np
from aie.dialects.aie import WireBundle
from aie.helpers.taplib import TensorTiler2D
from aie.iron import ObjectFifo, Program, Runtime, Worker, randint, tensor, zeros
from aie.iron.controlflow import range_
from aie.iron.device import Tile
from aie.iron.kernel import ExternalFunction
from aie.utils.jit import jit
import aie.utils as aie_utils
from aie.utils.trace import TraceConfig
from aie.utils.trace.events import PortEvent, get_events_for_device

from r57_profile import R34_ATTENTION, accounting_for_columns


DEV = os.environ.get("NPU_DEV", "npu2")
MODE = os.environ.get("R57_MODE", "PRODUCTION_DMA").upper()
if MODE not in (
    "PRODUCTION_DMA",
    "MEMTILE_STAGE",
    "MEMTILE_BROADCAST",
    "PRECONVERTED_LAYOUT",
):
    raise SystemExit(
        "R57_MODE must be PRODUCTION_DMA, MEMTILE_STAGE, MEMTILE_BROADCAST, "
        "or PRECONVERTED_LAYOUT"
    )
FANOUT = 4 if MODE in ("MEMTILE_BROADCAST", "PRECONVERTED_LAYOUT") else 1
COLS = int(os.environ.get("COLS", str(R34_ATTENTION.production_columns)))
if COLS not in (1, 2, 4, 8):
    raise SystemExit("COLS must be one of 1,2,4,8")
if MODE == "PRECONVERTED_LAYOUT" and COLS != R34_ATTENTION.production_columns:
    raise SystemExit("PRECONVERTED_LAYOUT requires the exact four-column R34 profile")
TRACE_COLS = int(os.environ.get("TRACE_COLS", str(min(COLS, 4))))
if not 1 <= TRACE_COLS <= COLS:
    raise SystemExit("TRACE_COLS must be between 1 and COLS")
DEPTH = int(os.environ.get("DEPTH", "4"))
HCLK_MHZ = float(os.environ.get("HCLK_MHZ", "1800"))
TRACE_SIZE = int(os.environ.get("TRACE_SIZE", "4194304"))
DDR_ID = int(os.environ.get("DDR_ID", "4"))
INC = os.environ["MLIR_AIE_INC"]

TILE_N = R34_ATTENTION.tile_bytes
N_TILES = R34_ATTENTION.blocks_per_column
PER_COLUMN = R34_ATTENTION.bytes_per_column
TOTAL_INPUT = COLS * PER_COLUMN
TRACE_DIR = Path(os.environ.get("R57_TRACE_DIR", "~/.hipfire/r57-traces")).expanduser()
TRACE_DIR.mkdir(parents=True, exist_ok=True)
TRACE_TXT = os.environ.get("TRACE_TXT", str(TRACE_DIR / f"trace-r57-c{COLS}.txt"))
TRACE_JSON = os.environ.get("TRACE_JSON", str(TRACE_DIR / f"trace-r57-c{COLS}.json"))

CoreEvent = get_events_for_device(DEV).CoreEvent
CORE_EVENTS = [
    PortEvent(
        CoreEvent.PORT_RUNNING_0,
        port=WireBundle.DMA,
        channel=0,
        master=True,
    ),
    PortEvent(
        CoreEvent.PORT_STALLED_0,
        port=WireBundle.DMA,
        channel=0,
        master=True,
    ),
    PortEvent(
        CoreEvent.PORT_IDLE_0,
        port=WireBundle.DMA,
        channel=0,
        master=True,
    ),
]

input_ty = np.ndarray[(TOTAL_INPUT,), np.dtype[np.int8]]
tile_ty = np.ndarray[(TILE_N,), np.dtype[np.int8]]
acc_ty = np.ndarray[(64,), np.dtype[np.int32]]

feed = ExternalFunction(
    "feed_sum",
    source_file="r57_feed.cc",
    arg_types=[tile_ty, acc_ty],
    include_dirs=[INC],
    compile_flags=["-std=c++20", "-O2", f"-DTILE_N={TILE_N}"],
)
sink = ExternalFunction(
    "feed_sink",
    source_file="r57_feed.cc",
    arg_types=[tile_ty],
    include_dirs=[INC],
    compile_flags=["-std=c++20", "-O2", f"-DTILE_N={TILE_N}"],
)


@jit(use_cache=True)
def production_dma(source, output, kernel_fn, sink_fn, **_kwargs):
    device = aie_utils.get_current_device()
    host_inputs = [
        ObjectFifo(tile_ty, name=f"weight_l3_{i}", depth=DEPTH) for i in range(COLS)
    ]
    if MODE in ("MEMTILE_STAGE", "MEMTILE_BROADCAST", "PRECONVERTED_LAYOUT"):
        core_inputs = [
            host_inputs[column]
            .cons()
            .forward(
                tile=Tile(col=column, row=1),
                obj_type=tile_ty,
                name=f"weight_l2_{column}",
                depth=DEPTH,
            )
            for column in range(COLS)
        ]
    else:
        core_inputs = host_inputs
    host_outputs = [
        ObjectFifo(acc_ty, name=f"guard_out_{column}", depth=1)
        for column in range(COLS)
    ]

    def core_body(in_fifo, out_fifo, fn):
        acc = out_fifo.acquire(1)
        for _ in range_(N_TILES):
            tile = in_fifo.acquire(1)
            fn(tile, acc)
            in_fifo.release(1)
        out_fifo.release(1)

    def sink_body(in_fifo, fn):
        for _ in range_(N_TILES):
            tile = in_fifo.acquire(1)
            fn(tile)
            in_fifo.release(1)

    workers = []
    trace_workers = []
    for column in range(COLS):
        guarded = Worker(
            core_body,
            [core_inputs[column].cons(), host_outputs[column].prod(), kernel_fn],
            tile=Tile(col=column, row=2),
        )
        workers.append(guarded)
        trace_workers.append(guarded)
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
        workers=trace_workers[:TRACE_COLS],
        ddr_id=DDR_ID,
        coretile_events=CORE_EVENTS,
    )
    with runtime.sequence(input_ty, acc_ty) as (source_arg, output_arg):
        for worker in workers:
            runtime.start(worker)
        regions = TensorTiler2D.simple_tiler([TOTAL_INPUT], [PER_COLUMN])
        for column in range(COLS):
            runtime.fill(host_inputs[column].prod(), source_arg, tap=regions[column])
        # These guards intentionally share a sink to stay within XRT's inout BO
        # budget. Every value is identical, so the final racing write is stable.
        for column in range(COLS):
            runtime.drain(host_outputs[column].cons(), output_arg, wait=True)
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


layout_file = ""
layout_source_sha256 = ""
layout_payload_sha256 = ""
if MODE == "PRECONVERTED_LAYOUT":
    layout_path = Path(os.environ.get("R57_LAYOUT_FILE", "")).expanduser()
    if not str(layout_path).endswith(".rdna2.hfp") or not layout_path.is_file():
        raise SystemExit("R57_LAYOUT_FILE must name an existing .rdna2.hfp file")
    layout_bytes = layout_path.read_bytes()
    if len(layout_bytes) != 128 + TOTAL_INPUT or layout_bytes[:8] != b"HFR34P01":
        raise SystemExit("invalid R34 .rdna2.hfp header or payload length")
    payload = layout_bytes[128:]
    expected_payload_sha256 = layout_bytes[80:112]
    actual_payload_sha256 = hashlib.sha256(payload).digest()
    if actual_payload_sha256 != expected_payload_sha256:
        raise SystemExit("R34 .rdna2.hfp payload SHA-256 mismatch")
    layout_file = str(layout_path)
    layout_source_sha256 = layout_bytes[48:80].hex()
    layout_payload_sha256 = actual_payload_sha256.hex()
    source = tensor(np.frombuffer(payload, dtype=np.int8).copy(), dtype=np.int8)
    expected_checksums = {
        int(
            np.sum(
                np.frombuffer(
                    payload[
                        column * PER_COLUMN
                        + (N_TILES - 1) * TILE_N : column * PER_COLUMN
                        + (N_TILES - 1) * TILE_N
                        + 64
                    ],
                    dtype=np.int8,
                ),
                dtype=np.int64,
            )
        )
        for column in range(COLS)
    }
else:
    source = randint(1, 2, (TOTAL_INPUT,), dtype=np.int8)
    expected_checksums = {64}
output = zeros(64, dtype=np.int32)
trace = TraceConfig(trace_size=TRACE_SIZE, trace_file=TRACE_TXT, ddr_id=DDR_ID)

started = time.perf_counter()
production_dma(source, output, feed, sink, trace_config=trace)
host_seconds = time.perf_counter() - started
checksum = int(output.numpy().reshape(-1)[0])
if checksum not in expected_checksums:
    raise SystemExit(
        f"guard checksum mismatch: expected one of {sorted(expected_checksums)}, got {checksum}"
    )

with open(TRACE_TXT, encoding="utf-8") as trace_file:
    packet_count = sum(
        1 for line in trace_file if line.strip() and line.strip() != "00000000"
    )
parse = subprocess.run(
    [
        sys.executable,
        "-m",
        "aie.utils.trace.parse",
        "--input",
        TRACE_TXT,
        "--mlir",
        trace.physical_mlir_path,
        "--output",
        TRACE_JSON,
    ],
    capture_output=True,
    text=True,
    check=False,
)
if parse.returncode != 0 or not os.path.exists(TRACE_JSON):
    raise SystemExit(f"trace parse failed rc={parse.returncode}\n{parse.stderr[-800:]}")

with open(TRACE_JSON, encoding="utf-8") as event_file:
    events = json.load(event_file)
running = paired_intervals(events, "PORT_RUNNING_0")
stalled = paired_intervals(events, "PORT_STALLED_0")
idle = paired_intervals(events, "PORT_IDLE_0")
timestamps = [stamp for intervals in running.values() for pair in intervals for stamp in pair]
if not timestamps:
    raise SystemExit("trace contains no PORT_RUNNING_0 intervals")

global_span = max(timestamps) - min(timestamps)
span_seconds = global_span / (HCLK_MHZ * 1e6)
busy_fractions = []
stall_fractions = []
idle_fractions = []
for pid, intervals in running.items():
    local_span = max(end for _, end in intervals) - min(begin for begin, _ in intervals)
    if local_span:
        busy_fractions.append(interval_sum(intervals) / local_span)
        stall_fractions.append(interval_sum(stalled.get(pid, ())) / local_span)
        idle_fractions.append(interval_sum(idle.get(pid, ())) / local_span)

accounting = accounting_for_columns(R34_ATTENTION, COLS)
xrt_version_output = subprocess.run(
    ["xrt-smi", "--version"], capture_output=True, text=True, check=False
).stdout


def version_field(label):
    match = re.search(rf"{re.escape(label)}\s*:\s*([^\n]+)", xrt_version_output)
    return match.group(1).strip() if match else "unknown"


try:
    repo_root = Path(__file__).resolve().parents[3]
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    git_dirty = int(
        bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
    )
except (OSError, subprocess.CalledProcessError):
    git_commit = "unknown"
    git_dirty = -1

benchmark_digest = hashlib.sha256()
for source_path in (
    Path(__file__),
    Path(__file__).with_name("r57_feed.cc"),
    Path(__file__).with_name("r57_profile.py"),
):
    benchmark_digest.update(source_path.name.encode())
    benchmark_digest.update(source_path.read_bytes())

row = {
    "mode": MODE,
    "profile": R34_ATTENTION.name,
    "git_commit": git_commit,
    "git_dirty": git_dirty,
    "benchmark_sha256": benchmark_digest.hexdigest(),
    "host": socket.gethostname(),
    "npu_device": DEV,
    "xrt_version": version_field("Version"),
    "amdxdna_version": version_field("amdxdna Version"),
    "firmware_version": version_field("NPU Firmware Version"),
    "power_mode": os.environ.get("R57_POWER_MODE", "default-unspecified"),
    "concurrent_load": os.environ.get("R57_CONCURRENT_LOAD", "idle"),
    "layout_file": layout_file,
    "layout_source_sha256": layout_source_sha256,
    "layout_payload_sha256": layout_payload_sha256,
    "production_exact": int(accounting.production_exact),
    "m": R34_ATTENTION.m,
    "k": R34_ATTENTION.k,
    "n": R34_ATTENTION.n,
    "columns": COLS,
    "fanout": FANOUT,
    "trace_columns": len(running),
    "tile_bytes": TILE_N,
    "blocks_per_column": N_TILES,
    "fifo_depth": DEPTH,
    "wire_bytes": accounting.wire_bytes,
    "nonpadding_bytes": accounting.nonpadding_bytes,
    "semantic_unique_bytes": accounting.semantic_unique_bytes,
    "logical_semantic_bytes": accounting.semantic_unique_bytes * FANOUT,
    "wire_over_unique": accounting.wire_over_unique,
    "nonpadding_fraction": accounting.nonpadding_fraction,
    "global_span_cycles": global_span,
    "hclk_mhz": HCLK_MHZ,
    "wire_gbs": accounting.wire_bytes / span_seconds / 1e9,
    "nonpadding_gbs": accounting.nonpadding_bytes / span_seconds / 1e9,
    "semantic_unique_gbs": accounting.semantic_unique_bytes / span_seconds / 1e9,
    "logical_semantic_gbs": accounting.semantic_unique_bytes * FANOUT / span_seconds / 1e9,
    "mean_receive_busy": sum(busy_fractions) / len(busy_fractions),
    "mean_receive_stalled": sum(stall_fractions) / len(stall_fractions),
    "mean_receive_idle": sum(idle_fractions) / len(idle_fractions),
    "host_call_ms": host_seconds * 1e3,
    "trace_packets": packet_count,
    "trace_overflow": int(packet_count >= TRACE_SIZE // 4 - 2),
    "checksum": checksum,
}
print("R57_JSON " + json.dumps(row, sort_keys=True))
