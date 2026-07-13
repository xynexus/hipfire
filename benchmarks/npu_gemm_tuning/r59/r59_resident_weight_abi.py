#!/usr/bin/env python3
"""Trace the first packed-resident ABI: four HFP BOs plus a parameter BO."""

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
MODE = os.environ.get("R59_ABI_MODE", "R34_SEPARATE_BOS").upper()
if MODE not in (
    "R34_SEPARATE_BOS",
    "R35_SEPARATE_BOS",
    "R34_BUNDLED_BO",
    "R35_BUNDLED_BO",
):
    raise SystemExit(
        "R59_ABI_MODE must select the R34/R35 separate or bundled BO mode"
    )
DEPTH = int(os.environ.get("DEPTH", "4"))
FANOUT = int(os.environ.get("FANOUT", "4"))
TRACE_COLS = int(os.environ.get("TRACE_COLS", "4"))
HCLK_MHZ = float(os.environ.get("HCLK_MHZ", "1800"))
TRACE_SIZE = int(os.environ.get("TRACE_SIZE", "8388608"))
DDR_ID = int(os.environ.get("DDR_ID", "4"))
INC = os.environ["MLIR_AIE_INC"]
PARAMETER_LOGICAL_BYTES = 4096
PARAMETER_PHYSICAL_BYTES = 16_384
GUARD_WORDS = 8


def read_hfp(environment, role):
    path = Path(os.environ[environment]).expanduser()
    raw = path.read_bytes()
    if len(raw) < 192 or raw[:8] != b"HFOPHFP2":
        raise SystemExit(f"{role}: not a version-2 Opus HFP artifact: {path}")
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
    payload = raw[header_bytes:]
    if (
        version != 2
        or header_bytes != 192
        or encoding not in (1, 2)
        or layout != 1
        or quant_type not in (33, 34, 35)
        or flags != 0
        or m != 256
        or columns != 8
        or tile_bytes != 16_384
        or data_bytes not in (12_288, 16_384)
        or scale_offset != data_bytes
        or payload_bytes != len(payload)
        or len(payload) != columns * groups * outblocks * tile_bytes
    ):
        raise SystemExit(f"{role}: unsupported HFP geometry: {fields}")
    if hashlib.sha256(payload).digest() != raw[128:160]:
        raise SystemExit(f"{role}: payload SHA-256 mismatch")
    return {
        "role": role,
        "path": path,
        "payload": payload,
        "raw": raw,
        "source_sha256": raw[96:128].hex(),
        "payload_sha256": raw[128:160].hex(),
        "encoding": encoding,
        "quant_type": quant_type,
        "m": m,
        "k": k,
        "n": n,
        "columns": columns,
        "groups": groups,
        "m_macros": m_macros,
        "n_macros": n_macros,
        "outblocks": outblocks,
        "tile_bytes": tile_bytes,
        "data_bytes": data_bytes,
        "scale_values": scale_values,
        "blocks_per_column": groups * outblocks,
    }


ROLES = [
    read_hfp("R59_QKV_HFP", "qkv"),
    read_hfp("R59_O_HFP", "o"),
    read_hfp("R59_GATE_UP_HFP", "gate_up"),
    read_hfp("R59_DOWN_HFP", "down"),
]
ACTIVE_ROLES = (
    ROLES[:2]
    if MODE.startswith("R34_")
    else ROLES[2:]
)
SEPARATE_BOS = MODE.endswith("SEPARATE_BOS")
COLUMNS = ROLES[0]["columns"]
TILE_BYTES = ROLES[0]["tile_bytes"]
if any(role["columns"] != COLUMNS or role["tile_bytes"] != TILE_BYTES for role in ROLES):
    raise SystemExit("resident HFP roles must share columns and tile bytes")
if not 1 <= FANOUT <= 4 or not 1 <= TRACE_COLS <= COLUMNS:
    raise SystemExit("FANOUT must be 1..4 and TRACE_COLS must be 1..8")


def framed_sha256(parts):
    digest = hashlib.sha256()
    for part in parts:
        digest.update(struct.pack("<Q", len(part)))
        digest.update(part)
    return digest.digest()


def build_resident_bundle():
    parameter_tile = np.zeros(PARAMETER_PHYSICAL_BYTES, dtype=np.int8)
    parameter_tile[:PARAMETER_LOGICAL_BYTES] = np.arange(
        PARAMETER_LOGICAL_BYTES, dtype=np.uint8
    ).view(np.int8)
    parameter_bytes = parameter_tile.tobytes()
    payload_parts = []
    for column in range(COLUMNS):
        for role in ACTIVE_ROLES:
            column_bytes = len(role["payload"]) // COLUMNS
            payload_parts.append(
                role["payload"][column * column_bytes : (column + 1) * column_bytes]
            )
        payload_parts.append(parameter_bytes)
    payload = b"".join(payload_parts)
    source_sha = framed_sha256(
        [role["raw"] for role in ACTIVE_ROLES] + [parameter_bytes[:PARAMETER_LOGICAL_BYTES]]
    )
    payload_sha = hashlib.sha256(payload).digest()
    header = bytearray(192)
    header[:8] = b"HFOPHFP2"
    fields = (
        2,  # version
        192,
        ACTIVE_ROLES[0]["encoding"],
        3,  # ResidentLayerBundleV1
        ACTIVE_ROLES[0]["quant_type"],
        len(ACTIVE_ROLES) + 1,
        256,
        0,
        0,
        COLUMNS,
        0,
        0,
        0,
        sum(role["blocks_per_column"] for role in ACTIVE_ROLES) + 1,
        TILE_BYTES,
        0,
        0,
        0,
    )
    struct.pack_into("<18I", header, 8, *fields)
    struct.pack_into("<Q", header, 80, len(payload))
    header[96:128] = source_sha
    header[128:160] = payload_sha
    for index, role in enumerate(ACTIVE_ROLES):
        struct.pack_into("<Q", header, 160 + index * 8, len(role["payload"]))
    struct.pack_into(
        "<Q", header, 160 + len(ACTIVE_ROLES) * 8, PARAMETER_LOGICAL_BYTES
    )
    contents = bytes(header) + payload
    path = Path(
        os.environ.get(
            "R59_BUNDLE_HFP",
            f"~/.hipfire/npu/prepacked/EmbeddingGemma-300M.npu.oq4.layer-0.{MODE.lower().replace('_', '-')}.rdna2.hfp",
        )
    ).expanduser()
    if not path.is_file() or path.read_bytes() != contents:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_bytes(contents)
        os.replace(temporary, path)
    return path, payload, source_sha.hex(), payload_sha.hex()


BUNDLE_PATH, BUNDLE_PAYLOAD, BUNDLE_SOURCE_SHA, BUNDLE_PAYLOAD_SHA = (
    build_resident_bundle()
)

tile_ty = np.ndarray[(TILE_BYTES,), np.dtype[np.int8]]
parameter_ty = np.ndarray[(PARAMETER_LOGICAL_BYTES,), np.dtype[np.int8]]
guard_ty = np.ndarray[(GUARD_WORDS,), np.dtype[np.int32]]
output_ty = np.ndarray[(COLUMNS * GUARD_WORDS,), np.dtype[np.int32]]
bundle_ty = np.ndarray[(len(BUNDLE_PAYLOAD),), np.dtype[np.int8]]
source_types = [
    np.ndarray[(len(role["payload"]),), np.dtype[np.int8]] for role in ACTIVE_ROLES
]

weight_guard = ExternalFunction(
    "r59_weight_guard",
    source_file="r59_resident_weight_abi.cc",
    arg_types=[tile_ty, guard_ty, np.int32],
    include_dirs=[INC],
    compile_flags=["-std=c++20", "-O2"],
)
zero_guard = ExternalFunction(
    "r59_zero_guard",
    source_file="r59_zero_guard.cc",
    arg_types=[guard_ty],
    include_dirs=[INC],
    compile_flags=["-std=c++20", "-O2"],
)
parameter_guard = ExternalFunction(
    "r59_parameter_guard",
    source_file="r59_parameter_guard.cc",
    arg_types=[parameter_ty if SEPARATE_BOS else tile_ty, guard_ty],
    include_dirs=[INC],
    compile_flags=["-std=c++20", "-O2"],
)
weight_sink = ExternalFunction(
    "r59_weight_sink",
    source_file="r59_resident_weight_sink.cc",
    arg_types=[tile_ty],
    include_dirs=[INC],
    compile_flags=["-std=c++20", "-O2"],
)

CoreEvent = get_events_for_device(DEV).CoreEvent
CORE_EVENTS = [
    PortEvent(CoreEvent.PORT_RUNNING_0, port=WireBundle.DMA, channel=0, master=True),
    PortEvent(CoreEvent.PORT_STALLED_0, port=WireBundle.DMA, channel=0, master=True),
    PortEvent(CoreEvent.PORT_IDLE_0, port=WireBundle.DMA, channel=0, master=True),
]


@jit(use_cache=True)
def resident_weight_abi(*arguments, **_kwargs):
    bundle_source, output, guard_fn, zero_fn, parameter_fn, sink_fn = arguments
    device = aie_utils.get_current_device()
    host_weights = [
        ObjectFifo(tile_ty, name=f"resident_w_l3_{column}", depth=DEPTH)
        for column in range(COLUMNS)
    ]
    core_weights = [
        host_weights[column]
        .cons()
        .forward(
            tile=Tile(col=column, row=1),
            obj_type=tile_ty,
            name=f"resident_w_l2_{column}",
            depth=DEPTH,
        )
        for column in range(COLUMNS)
    ]
    host_outputs = [
        ObjectFifo(guard_ty, name=f"resident_guard_{column}", depth=1)
        for column in range(COLUMNS)
    ]

    def guard_body(in_fifo, out_fifo, checker, zeroer, param_checker):
        guard = out_fifo.acquire(1)
        zeroer(guard)
        for role_index in range(len(ACTIVE_ROLES)):
            for _ in range_(ACTIVE_ROLES[role_index]["blocks_per_column"]):
                block = in_fifo.acquire(1)
                checker(block, guard, role_index)
                in_fifo.release(1)
        params = in_fifo.acquire(1)
        param_checker(params, guard)
        in_fifo.release(1)
        out_fifo.release(1)

    total_blocks = sum(role["blocks_per_column"] for role in ACTIVE_ROLES) + 1

    def sink_body(in_fifo, sink, block_count):
        for _ in range_(block_count):
            block = in_fifo.acquire(1)
            sink(block)
            in_fifo.release(1)

    workers = []
    traced = []
    for column in range(COLUMNS):
        worker = Worker(
            guard_body,
            [
                core_weights[column].cons(),
                host_outputs[column].prod(),
                guard_fn,
                zero_fn,
                parameter_fn,
            ],
            tile=Tile(col=column, row=2),
        )
        workers.append(worker)
        traced.append(worker)
        for row in range(1, FANOUT):
            workers.append(
                Worker(
                    sink_body,
                    [
                        core_weights[column].cons(),
                        sink_fn,
                        total_blocks,
                    ],
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
    with runtime.sequence(bundle_ty, output_ty) as sequence_args:
        bundle_arg, output_arg = sequence_args
        for worker in workers:
            runtime.start(worker)
        bundle_regions = TensorTiler2D.simple_tiler(
            [len(BUNDLE_PAYLOAD)], [len(BUNDLE_PAYLOAD) // COLUMNS]
        )
        for column in range(COLUMNS):
            runtime.fill(
                host_weights[column].prod(), bundle_arg, tap=bundle_regions[column]
            )
        output_regions = TensorTiler2D.simple_tiler(
            [COLUMNS * GUARD_WORDS], [GUARD_WORDS]
        )
        for column in range(COLUMNS):
            runtime.drain(
                host_outputs[column].cons(),
                output_arg,
                tap=output_regions[column],
                wait=True,
            )
    return Program(device, runtime).resolve_program()


@jit(use_cache=True)
def resident_weight_abi_separate(*arguments, **_kwargs):
    *sources, parameters, output, guard_fn, zero_fn, parameter_fn, sink_fn = arguments
    device = aie_utils.get_current_device()
    host_weights = [
        ObjectFifo(tile_ty, name=f"resident_sep_w_l3_{column}", depth=DEPTH)
        for column in range(COLUMNS)
    ]
    core_weights = [
        host_weights[column]
        .cons()
        .forward(
            tile=Tile(col=column, row=1),
            obj_type=tile_ty,
            name=f"resident_sep_w_l2_{column}",
            depth=DEPTH,
        )
        for column in range(COLUMNS)
    ]
    host_parameters = ObjectFifo(
        parameter_ty, name="resident_sep_params_l3", depth=1
    )
    core_parameters = host_parameters.cons().forward(
        tile=Tile(col=0, row=1),
        obj_type=parameter_ty,
        name="resident_sep_params_l2",
        depth=1,
    )
    host_outputs = [
        ObjectFifo(guard_ty, name=f"resident_sep_guard_{column}", depth=1)
        for column in range(COLUMNS)
    ]

    def checked_body(in_fifo, params_fifo, out_fifo, checker, zeroer, param_checker):
        guard = out_fifo.acquire(1)
        zeroer(guard)
        for role_index in range(len(ACTIVE_ROLES)):
            for _ in range_(ACTIVE_ROLES[role_index]["blocks_per_column"]):
                block = in_fifo.acquire(1)
                checker(block, guard, role_index)
                in_fifo.release(1)
        params = params_fifo.acquire(1)
        param_checker(params, guard)
        params_fifo.release(1)
        out_fifo.release(1)

    def guard_body(in_fifo, out_fifo, checker, zeroer):
        guard = out_fifo.acquire(1)
        zeroer(guard)
        for role_index in range(len(ACTIVE_ROLES)):
            for _ in range_(ACTIVE_ROLES[role_index]["blocks_per_column"]):
                block = in_fifo.acquire(1)
                checker(block, guard, role_index)
                in_fifo.release(1)
        out_fifo.release(1)

    total_blocks = sum(role["blocks_per_column"] for role in ACTIVE_ROLES)

    def sink_body(in_fifo, sink):
        for _ in range_(total_blocks):
            block = in_fifo.acquire(1)
            sink(block)
            in_fifo.release(1)

    workers = []
    traced = []
    for column in range(COLUMNS):
        if column == 0:
            worker = Worker(
                checked_body,
                [
                    core_weights[column].cons(),
                    core_parameters.cons(),
                    host_outputs[column].prod(),
                    guard_fn,
                    zero_fn,
                    parameter_fn,
                ],
                tile=Tile(col=column, row=2),
            )
        else:
            worker = Worker(
                guard_body,
                [
                    core_weights[column].cons(),
                    host_outputs[column].prod(),
                    guard_fn,
                    zero_fn,
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
    with runtime.sequence(*source_types, parameter_ty, output_ty) as sequence_args:
        *source_args, parameter_arg, output_arg = sequence_args
        for worker in workers:
            runtime.start(worker)
        for role, source_arg in zip(ACTIVE_ROLES, source_args):
            regions = TensorTiler2D.simple_tiler(
                [len(role["payload"])], [len(role["payload"]) // COLUMNS]
            )
            for column in range(COLUMNS):
                runtime.fill(host_weights[column].prod(), source_arg, tap=regions[column])
        runtime.fill(host_parameters.prod(), parameter_arg)
        output_regions = TensorTiler2D.simple_tiler(
            [COLUMNS * GUARD_WORDS], [GUARD_WORDS]
        )
        for column in range(COLUMNS):
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


bundle_tensor = tensor(
    np.frombuffer(BUNDLE_PAYLOAD, dtype=np.int8).copy(), dtype=np.int8
)
source_tensors = [
    tensor(np.frombuffer(role["payload"], dtype=np.int8).copy(), dtype=np.int8)
    for role in ACTIVE_ROLES
]
parameter_values = np.arange(
    PARAMETER_LOGICAL_BYTES, dtype=np.uint8
).view(np.int8)
parameter_tensor = tensor(parameter_values.copy(), dtype=np.int8)
output = zeros(COLUMNS * GUARD_WORDS, dtype=np.int32)
trace_dir = Path(os.environ.get("R59_TRACE_DIR", "~/.hipfire/r59-traces")).expanduser()
trace_dir.mkdir(parents=True, exist_ok=True)
trace_stem = MODE.lower().replace("_", "-")
trace_txt = str(trace_dir / f"resident-weight-abi-{trace_stem}.txt")
trace_json = str(trace_dir / f"resident-weight-abi-{trace_stem}.json")
trace = TraceConfig(trace_size=TRACE_SIZE, trace_file=trace_txt, ddr_id=DDR_ID)

started = time.perf_counter()
if SEPARATE_BOS:
    resident_weight_abi_separate(
        *source_tensors,
        parameter_tensor,
        output,
        weight_guard,
        zero_guard,
        parameter_guard,
        weight_sink,
        trace_config=trace,
    )
else:
    resident_weight_abi(
        bundle_tensor,
        output,
        weight_guard,
        zero_guard,
        parameter_guard,
        weight_sink,
        trace_config=trace,
    )
host_seconds = time.perf_counter() - started
actual = output.numpy().reshape(COLUMNS, GUARD_WORDS)
for column in range(COLUMNS):
    for role_index, role in enumerate(ACTIVE_ROLES):
        column_bytes = len(role["payload"]) // COLUMNS
        stream = role["payload"][column * column_bytes : (column + 1) * column_bytes]
        expected = int(
            np.sum(
                np.frombuffer(stream[-TILE_BYTES : -TILE_BYTES + 64], dtype=np.int8),
                dtype=np.int64,
            )
        )
        expected = (expected + 128) % 256 - 128
        if actual[column, role_index] != expected:
            raise SystemExit(
                f"column {column} role {role['role']} guard mismatch: "
                f"{actual[column, role_index]} != {expected}"
            )
parameter_expected = sum(
    (
        int(np.sum(parameter_values[offset : offset + 64], dtype=np.int64)) + 128
    )
    % 256
    - 128
    for offset in range(0, PARAMETER_LOGICAL_BYTES, 64)
)
if SEPARATE_BOS:
    parameter_ok = actual[0, 4] == parameter_expected and np.all(actual[1:, 4] == 0)
else:
    parameter_ok = np.all(actual[:, 4] == parameter_expected)
if not parameter_ok:
    raise SystemExit("parameter BO guard mismatch")

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
    Path(__file__).with_name("r59_resident_weight_abi.cc"),
    Path(__file__).with_name("r59_zero_guard.cc"),
    Path(__file__).with_name("r59_parameter_guard.cc"),
    Path(__file__).with_name("r59_resident_weight_sink.cc"),
):
    benchmark_sha.update(path.name.encode())
    benchmark_sha.update(path.read_bytes())

wire_bytes = (
    sum(len(role["payload"]) for role in ACTIVE_ROLES) + PARAMETER_LOGICAL_BYTES
    if SEPARATE_BOS
    else len(BUNDLE_PAYLOAD)
)
packed_data_bytes = sum(
    role["columns"]
    * role["blocks_per_column"]
    * role["data_bytes"]
    for role in ACTIVE_ROLES
)
row = {
    "mode": f"RESIDENT_WEIGHT_ABI_{MODE}",
    "profile": "embeddinggemma-r34-r35-hfp-arguments-v1",
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
    "columns": COLUMNS,
    "fanout": FANOUT,
    "trace_columns": len(running),
    "fifo_depth": DEPTH,
    "weight_arguments": len(ACTIVE_ROLES) if SEPARATE_BOS else 1,
    "weight_source_artifacts": len(ACTIVE_ROLES),
    "parameter_arguments": 1 if SEPARATE_BOS else 0,
    "embedded_parameter_segments": 0 if SEPARATE_BOS else COLUMNS,
    "parameter_logical_bytes": PARAMETER_LOGICAL_BYTES,
    "parameter_physical_bytes": PARAMETER_PHYSICAL_BYTES,
    "bundle_path": str(BUNDLE_PATH),
    "bundle_source_sha256": BUNDLE_SOURCE_SHA,
    "bundle_payload_sha256": BUNDLE_PAYLOAD_SHA,
    "wire_bytes": wire_bytes,
    "packed_data_bytes": packed_data_bytes,
    "global_span_cycles": global_span,
    "hclk_mhz": HCLK_MHZ,
    "wire_gbs": wire_bytes / span_seconds / 1e9,
    "packed_data_gbs": packed_data_bytes / span_seconds / 1e9,
    "mean_receive_stalled": fractions(stalled),
    "host_call_ms": host_seconds * 1e3,
    "weight_guard_oracle": "pass",
    "parameter_guard_oracle": "pass",
    "roles": [
        {
            key: role[key]
            for key in (
                "role",
                "path",
                "source_sha256",
                "payload_sha256",
                "encoding",
                "quant_type",
                "k",
                "n",
                "groups",
                "outblocks",
                "blocks_per_column",
            )
        }
        for role in ACTIVE_ROLES
    ],
}
for role in row["roles"]:
    role["path"] = str(role["path"])
print("R59_JSON " + json.dumps(row, sort_keys=True))
