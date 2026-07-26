#!/usr/bin/env python3
"""Run the raw-MLIR R61 cache and verify direct row-major QKV output."""

from collections import defaultdict
import ctypes
import hashlib
import json
import os
from pathlib import Path
import socket
import statistics
import struct
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, "/opt/xilinx/xrt/python")
ctypes.CDLL(
    "/opt/xilinx/xrt/lib/libxrt_coreutil.so.2", mode=ctypes.RTLD_GLOBAL
)
from aie.utils.hostruntime.xrtruntime.hostruntime import XRTHostRuntime
from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor
from aie.utils.npukernel import NPUKernel


ACTIVATION_MODE = os.environ.get("R61_ACTIVATION_MODE", "r34-legacy")
if ACTIVATION_MODE not in ("r34-legacy", "w4-native"):
    raise SystemExit("R61_ACTIVATION_MODE must be r34-legacy or w4-native")
OUTPUT_MODE = os.environ.get("R61_OUTPUT_MODE", "rowmajor")
if OUTPUT_MODE not in ("rowmajor", "physical"):
    raise SystemExit("R61_OUTPUT_MODE must be rowmajor or physical")
hfp_value = (
    os.environ.get("R63_QKV_HFP")
    or os.environ.get("R62_R34_BUNDLE_HFP")
    or os.environ.get("R61_R34_BUNDLE_HFP")
)
if not hfp_value:
    raise SystemExit(
        "R63_QKV_HFP, R62_R34_BUNDLE_HFP, or R61_R34_BUNDLE_HFP is required"
    )
HFP = Path(hfp_value).expanduser()
CACHE = Path(
    os.environ.get(
        "R61_CACHE_DIR",
        (
            "~/.hipfire/npu/embgemma_r61_full_qkv_m256_k768_n1280"
            if ACTIVATION_MODE == "r34-legacy"
            else (
                "~/.hipfire/npu/embgemma_r62_w4_native_qkv_m256_k768_n1280"
                if OUTPUT_MODE == "rowmajor"
                else "~/.hipfire/npu/embgemma_r62_w4_native_physical_qkv_m256_k768_n1280"
            )
        ),
    )
).expanduser()
M, K, N = 256, 768, 1280
PAD_M, PAD_N = 288, 1536
GROUPS, COLUMNS, CORE_ROWS = 3, 8, 4
OUTBLOCKS, BLOCK_BYTES = 6, 16_384
R34_BLOCKS_PER_STRIPE = 45
R34_INPUT_BYTES = CORE_ROWS * R34_BLOCKS_PER_STRIPE * BLOCK_BYTES


def parse_weight_payload(path):
    raw = path.read_bytes()
    fields = struct.unpack_from("<18I", raw, 8)
    payload_bytes = struct.unpack_from("<Q", raw, 80)[0]
    segments = struct.unpack_from("<4Q", raw, 160)
    payload = raw[192:]
    common_invalid = (
        len(raw) < 192
        or raw[:8] != b"HFOPHFP2"
        or fields[0] != 2
        or fields[2] != 1
        or fields[9] != COLUMNS
        or payload_bytes != len(payload)
        or hashlib.sha256(payload).digest() != raw[128:160]
    )
    resident_bundle = (
        fields[3] == 3
        and segments[0] == COLUMNS * OUTBLOCKS * GROUPS * BLOCK_BYTES
    )
    compact_qkv = (
        fields[3] == 1
        and fields[5] == 0
        and fields[6:9] == (256, K, N)
        and fields[10:15] == (GROUPS, 3, 2, OUTBLOCKS, BLOCK_BYTES)
        and payload_bytes == COLUMNS * OUTBLOCKS * GROUPS * BLOCK_BYTES
    )
    if common_invalid or not (resident_bundle or compact_qkv):
        raise SystemExit("invalid R61/R62/R63 Opus weight HFP")
    return payload, "resident-bundle-v1" if resident_bundle else "whole-scaled-v1"


BUNDLE, HFP_PROFILE = parse_weight_payload(HFP)
BLOCKS_PER_COLUMN = len(BUNDLE) // (COLUMNS * BLOCK_BYTES)
BYTES_PER_COLUMN = BLOCKS_PER_COLUMN * BLOCK_BYTES


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


def pack_w4_activations(values):
    blocks_per_stripe = OUTBLOCKS * GROUPS
    packed = np.zeros(CORE_ROWS * blocks_per_stripe * 8192, dtype=np.int8)
    for stripe in range(CORE_ROWS):
        for m_macro in range(3):
            for n_macro in range(2):
                outblock = m_macro * 2 + n_macro
                for group in range(GROUPS):
                    block = outblock * GROUPS + group
                    base = (stripe * blocks_per_stripe + block) * 8192
                    for lm in range(6):
                        for kt in range(16):
                            for local_row in range(4):
                                row = m_macro * 96 + stripe * 24 + lm * 4 + local_row
                                if row < M:
                                    source = group * 256 + kt * 16
                                    target = base + (lm * 16 + kt) * 64 + local_row * 16
                                    packed[target : target + 16] = values[
                                        row, source : source + 16
                                    ]
                    for local_row in range(24):
                        row = m_macro * 96 + stripe * 24 + local_row
                        scale = np.float32(1.0 if row < M else 0.0)
                        packed[
                            base + 6144 + local_row * 4 : base + 6148 + local_row * 4
                        ] = np.frombuffer(scale.tobytes(), dtype=np.int8)
    return packed


def sext4(value):
    value &= 0xF
    return value - 16 if value & 8 else value


def reference_output(activations):
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
            activations[:, group * 256 : (group + 1) * 256].astype(np.int32)
            @ weights[group].astype(np.int32)
        )
        expected[:M] = (
            expected[:M] + dots.astype(np.float32) * scales[group]
        ).astype(np.float32)
    return expected


def unpack_physical(values):
    physical = values.reshape(COLUMNS, OUTBLOCKS, CORE_ROWS, 2304)
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


activations = canonical_activations()
shared_input = (
    pack_r34_activations(activations)
    if ACTIVATION_MODE == "r34-legacy"
    else pack_w4_activations(activations)
)
expected = reference_output(activations)
t_input = XRTTensor(shared_input.copy(), dtype=np.int8, device="cpu")
t_bundle = XRTTensor(
    np.frombuffer(BUNDLE, dtype=np.int8).copy(), dtype=np.int8, device="cpu"
)
t_output = XRTTensor((PAD_M * PAD_N,), dtype=np.int32, device="cpu")
trace_size = int(os.environ.get("R64_TRACE_SIZE", "0"))
if trace_size < 0 or trace_size % 4 != 0:
    raise SystemExit("R64_TRACE_SIZE must be zero or a positive multiple of four")
t_trace = (
    XRTTensor((trace_size // 4,), dtype=np.uint32, device="cpu")
    if trace_size
    else None
)
# Declarative tracing reserves DPU data argument index 4 (group_id 7). The
# production graph has three data arguments, so index 3 must remain occupied by
# a harmless padding BO for the trace address patch to target index 4.
t_trace_padding = XRTTensor((1,), dtype=np.uint32, device="cpu") if trace_size else None
npu_kernel = NPUKernel(
    xclbin_path=CACHE / "final.xclbin",
    insts_path=CACHE / "insts.bin",
    kernel_name="MLIR_AIE",
)
runtime = XRTHostRuntime()
handle = runtime.load(npu_kernel)
warmup = int(os.environ.get("R61_WARMUP", "0"))
iterations = int(os.environ.get("R61_ITERS", "1"))
if warmup < 0 or iterations <= 0:
    raise SystemExit("R61_WARMUP must be nonnegative and R61_ITERS must be positive")
npu_times = []
host_times = []
result = None
for iteration in range(warmup + iterations):
    started = time.perf_counter()
    arguments = [t_input, t_bundle, t_output]
    if t_trace is not None:
        assert t_trace_padding is not None
        arguments.extend((t_trace_padding, t_trace))
    result = runtime.run(handle, arguments)
    elapsed = time.perf_counter() - started
    if not result.is_success():
        raise SystemExit(f"R61 NPU command failed: {result.ret}")
    if iteration >= warmup:
        npu_times.append(result.npu_time)
        host_times.append(elapsed)
assert result is not None
t_output.to("cpu")
raw_output = t_output.numpy().view(np.float32)
got = (
    raw_output.reshape(PAD_M, PAD_N)
    if OUTPUT_MODE == "rowmajor"
    else unpack_physical(raw_output)
)
absolute = np.abs(got - expected)
tolerance = 1e-4 + np.abs(expected) * 1e-5
mismatches = int(np.count_nonzero(absolute > tolerance))
if mismatches:
    row, col = map(int, np.argwhere(absolute > tolerance)[0])
    raise SystemExit(
        f"R61 mismatch count={mismatches} first=({row},{col}) "
        f"got={got[row,col]} expected={expected[row,col]} abs={absolute[row,col]}"
    )

npu_time_ns = statistics.median(npu_times)
npu_seconds = npu_time_ns / 1e9
host_seconds = statistics.median(host_times)
useful_macs = M * K * N
padded_macs = PAD_M * K * PAD_N
repo_root = Path(__file__).resolve().parents[3]
git_commit = subprocess.run(
    ["git", "rev-parse", "HEAD"],
    cwd=repo_root,
    capture_output=True,
    text=True,
    check=True,
).stdout.strip()
row = {
    "mode": (
        "FULL_R34_SHARED_INPUT_OQ4_QKV_ROWMAJOR"
        if ACTIVATION_MODE == "r34-legacy"
        else (
            "FULL_W4_NATIVE_INPUT_OQ4_QKV_ROWMAJOR"
            if OUTPUT_MODE == "rowmajor"
            else "FULL_W4_NATIVE_INPUT_OQ4_QKV_PHYSICAL"
        )
    ),
    "profile": f"embeddinggemma-{HFP_PROFILE}",
    "kernel_variant": os.environ.get("R61_KERNEL_VARIANT", "current"),
    "git_commit": git_commit,
    "git_dirty": int(bool(subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip())),
    "host": socket.gethostname(),
    "bundle_file": str(HFP),
    "bundle_payload_sha256": hashlib.sha256(BUNDLE).hexdigest(),
    "cache_dir": str(CACHE),
    "m": M,
    "k": K,
    "n": N,
    "padded_m": PAD_M,
    "padded_n": PAD_N,
    "bundle_bytes": len(BUNDLE),
    "shared_input_bytes": len(shared_input),
    "output_bytes": got.nbytes,
    "warmup": warmup,
    "iterations": iterations,
    "npu_time_ns": npu_time_ns,
    "npu_time_min_ns": min(npu_times),
    "npu_time_max_ns": max(npu_times),
    "npu_ms": npu_seconds * 1e3,
    "host_call_ms": host_seconds * 1e3,
    "useful_tops": 2 * useful_macs / npu_seconds / 1e12,
    "padded_tops": 2 * padded_macs / npu_seconds / 1e12,
    "checked_outputs": M * N,
    "padded_outputs": PAD_M * PAD_N,
    "mismatches": mismatches,
    "max_abs": float(absolute.max()),
    "full_qkv_oracle": "pass",
    "output_layout": "padded-row-major" if OUTPUT_MODE == "rowmajor" else "whole-scaled-physical",
    "activation_swizzle": (
        "tile-local-6k-vector-cache"
        if ACTIVATION_MODE == "r34-legacy"
        else "producer-native-w4-none-in-core"
    ),
    "global_weight_reorder_in_kernel": False,
}

if t_trace is not None:
    from aie.utils.trace import TraceConfig

    trace_dir = Path(os.environ.get("R64_TRACE_DIR", "~/.hipfire/r64-traces")).expanduser()
    trace_dir.mkdir(parents=True, exist_ok=True)
    trace_txt = trace_dir / "full-qkv.txt"
    trace_json = trace_dir / "full-qkv.json"
    trace_config = TraceConfig(trace_size=trace_size, trace_file=str(trace_txt), ddr_id=4)
    t_trace.to("cpu")
    trace_config.write_trace(t_trace.numpy().astype(np.uint32, copy=False))
    physical_pointer = CACHE / "trace_physical_mlir.txt"
    if not physical_pointer.is_file():
        raise SystemExit("R64 cache is missing trace_physical_mlir.txt")
    physical_mlir = Path(physical_pointer.read_text(encoding="utf-8").strip())
    parse = subprocess.run(
        [
            sys.executable,
            "-m",
            "aie.utils.trace.parse",
            "--input",
            str(trace_txt),
            "--mlir",
            str(physical_mlir),
            "--output",
            str(trace_json),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if parse.returncode or not trace_json.is_file():
        raise SystemExit(f"R64 trace parse failed rc={parse.returncode}\n{parse.stderr[-800:]}")
    events = json.loads(trace_json.read_text(encoding="utf-8"))

    def paired_intervals(name):
        records_by_pid = defaultdict(list)
        for event in events:
            if event.get("name") == name and "ts" in event:
                records_by_pid[event.get("pid", 0)].append(event)
        intervals = defaultdict(list)
        unmatched = 0
        for pid, records in records_by_pid.items():
            opened = None
            for event in sorted(records, key=lambda value: value["ts"]):
                if event.get("ph") == "B":
                    if opened is not None:
                        unmatched += 1
                    opened = event["ts"]
                elif event.get("ph") == "E":
                    if opened is None:
                        unmatched += 1
                    else:
                        intervals[pid].append((opened, event["ts"]))
                        opened = None
            unmatched += int(opened is not None)
        return intervals, unmatched

    trace_packets = sum(
        1 for line in trace_txt.read_text(encoding="utf-8").splitlines() if line.strip()
    )
    overflow = trace_packets >= trace_size // 4 - 2
    expected_trace_columns = int(os.environ.get("R64_TRACE_COLS", str(COLUMNS)))
    trace_start_column = int(os.environ.get("R64_TRACE_START", "0"))
    hclk_mhz = float(os.environ.get("R64_HCLK_MHZ", "1800"))
    trace_tile = os.environ.get("R64_TRACE_TILE", "core")

    def all_pairs(groups):
        return [pair for group in groups for pairs in group.values() for pair in pairs]

    def interval_sum(groups):
        return sum(end - begin for begin, end in all_pairs(groups))

    if trace_tile == "shim":
        def starts(name):
            return [
                (event.get("pid", 0), event["ts"])
                for event in events
                if event.get("name") == name and event.get("ph") == "B" and "ts" in event
            ]

        input0_start = starts("DMA_MM2S_0_START_TASK")
        input0_finish = starts("DMA_MM2S_0_FINISHED_TASK")
        input1_start = starts("DMA_MM2S_1_START_TASK")
        input1_finish = starts("DMA_MM2S_1_FINISHED_TASK")
        output_start_events = starts("DMA_S2MM_0_START_TASK")
        output_finish = starts("DMA_S2MM_0_FINISHED_TASK")
        required = [input0_start, input0_finish, output_start_events, output_finish]
        # Activation stripes enter through shim columns 0..3. Columns 4..7
        # carry only the weight MM2S stream plus output S2MM.
        has_activation_stream = trace_start_column < 4
        if has_activation_stream:
            required.extend((input1_start, input1_finish))
        if overflow or any(not values for values in required):
            raise SystemExit(
                f"R64 invalid shim trace overflow={int(overflow)} "
                f"event_counts={[len(values) for values in required]}"
            )
        traced_pids = {pid for values in required for pid, _ in values}
        if len(traced_pids) != expected_trace_columns:
            raise SystemExit(
                f"R64 traced {len(traced_pids)} shim columns, expected {expected_trace_columns}"
            )
        input_start_groups = (input0_start, input1_start) if has_activation_stream else (input0_start,)
        input_end_groups = (input0_finish, input1_finish) if has_activation_stream else (input0_finish,)
        input_start = min(stamp for values in input_start_groups for _, stamp in values)
        input_end = max(stamp for values in input_end_groups for _, stamp in values)
        output_start = min(stamp for _, stamp in output_start_events)
        output_end = max(stamp for _, stamp in output_finish)
        starvation0, starvation_unmatched = paired_intervals("DMA_S2MM_0_STREAM_STARVATION")
        if starvation_unmatched:
            raise SystemExit(f"R64 output starvation trace has {starvation_unmatched} unmatched events")
        trace_unmatched = 0
        trace_details = {
            "input0_start_cycle": min(stamp for _, stamp in input0_start),
            "input0_end_cycle": max(stamp for _, stamp in input0_finish),
            "input1_start_cycle": min((stamp for _, stamp in input1_start), default=0),
            "input1_end_cycle": max((stamp for _, stamp in input1_finish), default=0),
            "output_starved_cycles": interval_sum((starvation0,)),
        }
    else:
        input0, unmatched0 = paired_intervals("PORT_RUNNING_0")
        input1, unmatched1 = paired_intervals("PORT_RUNNING_1")
        output0, unmatched2 = paired_intervals("PORT_RUNNING_2")
        stalled0, unmatched3 = paired_intervals("PORT_STALLED_0")
        stalled1, unmatched4 = paired_intervals("PORT_STALLED_1")
        stalled2, unmatched5 = paired_intervals("PORT_STALLED_2")
        trace_unmatched = sum(
            (unmatched0, unmatched1, unmatched2, unmatched3, unmatched4, unmatched5)
        )
        if overflow or trace_unmatched or not input0 or not input1 or not output0:
            raise SystemExit(
                f"R64 invalid core trace overflow={int(overflow)} unmatched={trace_unmatched} "
                f"ports={len(input0)}/{len(input1)}/{len(output0)}"
            )
        traced_pids = set(input0) | set(input1) | set(output0)
        if len(traced_pids) != expected_trace_columns:
            raise SystemExit(
                f"R64 traced {len(traced_pids)} core columns, expected {expected_trace_columns}"
            )
        input_pairs = all_pairs((input0, input1))
        output_pairs = all_pairs((output0,))
        input_start = min(begin for begin, _ in input_pairs)
        input_end = max(end for _, end in input_pairs)
        output_start = min(begin for begin, _ in output_pairs)
        output_end = max(end for _, end in output_pairs)
        trace_details = {
            "input0_running_cycles": interval_sum((input0,)),
            "input1_running_cycles": interval_sum((input1,)),
            "output_running_cycles": interval_sum((output0,)),
            "input0_stalled_cycles": interval_sum((stalled0,)),
            "input1_stalled_cycles": interval_sum((stalled1,)),
            "output_stalled_cycles": interval_sum((stalled2,)),
        }

    device_span_cycles = output_end - input_start
    if device_span_cycles <= 0:
        raise SystemExit("R64 trace has non-positive input-to-output span")
    device_seconds = device_span_cycles / (hclk_mhz * 1e6)
    row.update(
        {
            "trace_file": str(trace_txt),
            "trace_json": str(trace_json),
            "trace_physical_mlir": str(physical_mlir),
            "trace_tile": trace_tile,
            "trace_packets": trace_packets,
            "trace_overflow": int(overflow),
            "trace_unmatched": trace_unmatched,
            "trace_columns": len(traced_pids),
            "trace_start_column": trace_start_column,
            "hclk_mhz": hclk_mhz,
            "device_span_cycles": device_span_cycles,
            "device_span_us": device_seconds * 1e6,
            "device_traffic_gbs": (len(shared_input) + len(BUNDLE) + got.nbytes)
            / device_seconds
            / 1e9,
            "input_start_cycle": input_start,
            "input_end_cycle": input_end,
            "output_start_cycle": output_start,
            "output_end_cycle": output_end,
            "compute_drain_tail_cycles": output_end - input_end,
            **trace_details,
        }
    )

prefix = "R64_JSON " if t_trace is not None else (
    "R61_JSON " if ACTIVATION_MODE == "r34-legacy" else "R62_JSON "
)
print(prefix + json.dumps(row, sort_keys=True))
