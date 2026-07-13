#!/usr/bin/env python3
"""Run R65 and verify its BF16 R29 raw-attention staging ABI."""

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

from ml_dtypes import bfloat16
import numpy as np

sys.path.insert(0, "/opt/xilinx/xrt/python")
ctypes.CDLL("/opt/xilinx/xrt/lib/libxrt_coreutil.so.2", mode=ctypes.RTLD_GLOBAL)
from aie.utils.hostruntime.xrtruntime.hostruntime import XRTHostRuntime
from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor
from aie.utils.npukernel import NPUKernel


M, K, N = 256, 768, 1280
PAD_M, PAD_N = 288, 1536
GROUPS, COLS, CORE_ROWS = 3, 8, 4
OUTBLOCKS, BLOCK_BYTES = 6, 16_384
STAGE_LAYOUT = os.environ.get("R65_STAGE_LAYOUT", "inline")
if STAGE_LAYOUT not in ("inline", "joined", "joined37"):
    raise SystemExit("R65_STAGE_LAYOUT must be inline, joined, or joined37")
if STAGE_LAYOUT == "inline":
    PAIR, PAIRS_PER_ROLE, ROLES = 10_240, 48, 5
else:
    PAIR, PAIRS_PER_ROLE, ROLES = 8192, 37 if STAGE_LAYOUT == "joined37" else 36, 5
VALUE_BYTES = 4096
PARAM_BYTES = 2048 if STAGE_LAYOUT != "inline" else 0
R_BYTES = ROLES * PAIRS_PER_ROLE * PAIR + PARAM_BYTES
TAIL_SENTINEL = np.int8(0x5A)
CACHE = Path(
    os.environ.get(
        "R65_CACHE_DIR",
        (
            "~/.hipfire/npu/embgemma_r65_w4_bf16_raw_attention_m256_k768_n1280"
            if STAGE_LAYOUT == "inline"
            else (
                "~/.hipfire/npu/embgemma_r68_w4_overlap_joined_stage_m256_k768_n1280"
                if STAGE_LAYOUT == "joined37"
                else "~/.hipfire/npu/embgemma_r67_w4_joined_stage_m256_k768_n1280"
            )
        ),
    )
).expanduser()
hfp_value = os.environ.get("R63_QKV_HFP")
if not hfp_value:
    raise SystemExit("R63_QKV_HFP is required")
HFP = Path(hfp_value).expanduser()
KERNEL_HFP = Path(os.environ.get("R65_KERNEL_HFP", hfp_value)).expanduser()


def parse_weight_payload(path):
    raw = path.read_bytes()
    if len(raw) < 192:
        raise SystemExit("truncated R65 Opus HFP")
    fields = struct.unpack_from("<18I", raw, 8)
    payload_bytes = struct.unpack_from("<Q", raw, 80)[0]
    payload = raw[192:]
    valid = (
        raw[:8] == b"HFOPHFP2"
        and fields[0] == 2
        and fields[2] == 1
        and fields[3] == 1
        and fields[5] == 0
        and fields[6:9] == (256, K, N)
        and fields[9] == COLS
        and fields[10:15] == (GROUPS, 3, 2, OUTBLOCKS, BLOCK_BYTES)
        and payload_bytes == COLS * OUTBLOCKS * GROUPS * BLOCK_BYTES
        and payload_bytes == len(payload)
        and hashlib.sha256(payload).digest() == raw[128:160]
    )
    if not valid:
        raise SystemExit("invalid compact whole-scaled R65 Opus HFP")
    return payload


def parse_kernel_weight_payload(path):
    raw = path.read_bytes()
    if len(raw) < 192 or raw[:8] != b"HFOPHFP2":
        raise SystemExit("invalid R65 kernel HFP header")
    payload_bytes = struct.unpack_from("<Q", raw, 80)[0]
    payload = raw[192:]
    if (
        payload_bytes != len(payload)
        or payload_bytes != COLS * OUTBLOCKS * GROUPS * BLOCK_BYTES
        or hashlib.sha256(payload).digest() != raw[128:160]
    ):
        raise SystemExit("invalid R65 kernel HFP payload")
    return payload


def canonical_activations():
    rows = np.arange(M, dtype=np.int32)[:, None]
    inner = np.arange(K, dtype=np.int32)[None, :]
    return (((rows * 29 + inner * 17 + 11) % 255) - 127).astype(np.int8)


def pack_activations(values):
    packed = np.zeros(CORE_ROWS * OUTBLOCKS * GROUPS * 8192, dtype=np.int8)
    for stripe in range(CORE_ROWS):
        for m_macro in range(3):
            for n_macro in range(2):
                outblock = m_macro * 2 + n_macro
                for group in range(GROUPS):
                    block = outblock * GROUPS + group
                    base = (stripe * OUTBLOCKS * GROUPS + block) * 8192
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


def reference_output(activations, bundle):
    weights = np.zeros((GROUPS, 256, PAD_N), dtype=np.int8)
    scales = np.zeros((GROUPS, PAD_N), dtype=np.float32)
    bytes_per_col = len(bundle) // COLS
    for column in range(COLS):
        stream = bundle[column * bytes_per_col : (column + 1) * bytes_per_col]
        for n_macro in range(2):
            for group in range(GROUPS):
                block_index = n_macro * GROUPS + group
                block = stream[block_index * BLOCK_BYTES : (block_index + 1) * BLOCK_BYTES]
                for jn in range(6):
                    col = n_macro * 768 + column * 96 + jn * 16
                    for kt in range(16):
                        encoded = block[(jn * 16 + kt) * 128 : (jn * 16 + kt + 1) * 128]
                        decoded = np.array(
                            [v for byte in encoded for v in (sext4(byte), sext4(byte >> 4))],
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


def unpack_stage(stage):
    records = stage[: ROLES * PAIRS_PER_ROLE * PAIR].reshape(
        ROLES, PAIRS_PER_ROLE, PAIR
    )
    output = np.zeros((PAD_M, N), dtype=np.float32)
    if STAGE_LAYOUT != "inline":
        for role in range(ROLES):
            for pair_index in range(PAIRS_PER_ROLE):
                row = pair_index * 8
                if row >= PAD_M:
                    continue
                values = np.frombuffer(
                    records[role, pair_index, :VALUE_BYTES].tobytes(),
                    dtype=bfloat16,
                ).astype(np.float32).reshape(8, 256)
                output[row : row + 8, role * 256 : (role + 1) * 256] = values
        return output
    for role in range(ROLES):
        for m_macro in range(3):
            for core_row in range(CORE_ROWS):
                for token_group in range(3):
                    pair_index = m_macro * 16 + core_row * 4 + token_group
                    row = m_macro * 96 + core_row * 24 + token_group * 8
                    values = np.frombuffer(
                        records[role, pair_index, :VALUE_BYTES].tobytes(),
                        dtype=bfloat16,
                    ).astype(np.float32).reshape(8, 256)
                    output[row : row + 8, role * 256 : (role + 1) * 256] = values
    return output


def floor_to_bf16(values):
    """Match the AIE floor rounding mode left active by the R15 kernel."""
    bits = values.astype(np.float32, copy=False).view(np.uint32)
    rounded = bits & np.uint32(0xFFFF0000)
    discarded = (bits & np.uint32(0x0000FFFF)) != 0
    negative = (bits & np.uint32(0x80000000)) != 0
    rounded = rounded + ((discarded & negative).astype(np.uint32) << np.uint32(16))
    return rounded.view(np.float32)


bundle = parse_weight_payload(HFP)
kernel_bundle = parse_kernel_weight_payload(KERNEL_HFP)
activations = canonical_activations()
shared_input = pack_activations(activations)
expected_f32 = reference_output(activations, bundle)[:, :N]
expected = floor_to_bf16(expected_f32)
stage_seed = np.full(R_BYTES, TAIL_SENTINEL, dtype=np.int8)
if dump_value := os.environ.get("R65_DUMP_DIR"):
    dump_dir = Path(dump_value).expanduser()
    dump_dir.mkdir(parents=True, exist_ok=True)
    (dump_dir / "activations.bin").write_bytes(shared_input.tobytes())
    (dump_dir / "weights.bin").write_bytes(bundle)
    (dump_dir / "stage-seed.bin").write_bytes(stage_seed.tobytes())
t_input = XRTTensor(shared_input.copy(), dtype=np.int8, device="cpu")
t_weights = XRTTensor(
    np.frombuffer(kernel_bundle, dtype=np.int8).copy(), dtype=np.int8, device="cpu"
)
t_stage = XRTTensor(stage_seed.copy(), dtype=np.int8, device="cpu")
kernel = NPUKernel(
    xclbin_path=CACHE / "final.xclbin",
    insts_path=CACHE / "insts.bin",
    kernel_name="MLIR_AIE",
)
runtime = XRTHostRuntime()
handle = runtime.load(kernel)
warmup = int(os.environ.get("R65_WARMUP", "0"))
iterations = int(os.environ.get("R65_ITERS", "1"))
if warmup < 0 or iterations <= 0:
    raise SystemExit("R65_WARMUP must be nonnegative and R65_ITERS positive")
npu_times, host_times = [], []
for iteration in range(warmup + iterations):
    started = time.perf_counter()
    result = runtime.run(handle, [t_input, t_weights, t_stage])
    elapsed = time.perf_counter() - started
    if not result.is_success():
        raise SystemExit(f"R65 NPU command failed: {result.ret}")
    if iteration >= warmup:
        npu_times.append(result.npu_time)
        host_times.append(elapsed)
t_stage.to("cpu")
stage = t_stage.numpy().astype(np.int8, copy=False)
got = unpack_stage(stage)
absolute = np.abs(got - expected)
mismatch_mask = got.view(np.uint32) != expected.view(np.uint32)
mismatches = int(np.count_nonzero(mismatch_mask))
if mismatches:
    row, col = map(int, np.argwhere(mismatch_mask)[0])
    raise SystemExit(
        f"R65 BF16 mismatch count={mismatches} first=({row},{col}) "
        f"got={got[row,col]} expected={expected[row,col]} abs={absolute[row,col]}"
    )

records = stage[: ROLES * PAIRS_PER_ROLE * PAIR].reshape(
    ROLES, PAIRS_PER_ROLE, PAIR
)
tail_mismatches = int(np.count_nonzero(records[:, :, VALUE_BYTES:] != TAIL_SENTINEL))
if PARAM_BYTES:
    tail_mismatches += int(
        np.count_nonzero(stage[ROLES * PAIRS_PER_ROLE * PAIR :] != TAIL_SENTINEL)
    )
if tail_mismatches:
    raise SystemExit(f"R65 overwrote {tail_mismatches} preseeded attention-tail bytes")
padding_mismatches = 0
for role in range(ROLES):
    if STAGE_LAYOUT != "inline":
        padding_pairs = range(32, PAIRS_PER_ROLE)
    else:
        padding_pairs = (
            m_macro * 16 + core_row * 4 + 3
            for m_macro in range(3)
            for core_row in range(CORE_ROWS)
        )
    for pair_index in padding_pairs:
        padding_mismatches += int(np.count_nonzero(records[role, pair_index, :VALUE_BYTES]))
if padding_mismatches:
    raise SystemExit(f"R65 wrote {padding_mismatches} nonzero padding bytes")

npu_time_ns = statistics.median(npu_times)
npu_seconds = npu_time_ns / 1e9
repo_root = Path(__file__).resolve().parents[3]
git_commit = subprocess.run(
    ["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True, check=True
).stdout.strip()
row = {
    "mode": (
        "FULL_W4_QKV_BF16_R29_RAW_STAGE"
        if STAGE_LAYOUT == "inline"
        else "FULL_W4_QKV_BF16_R28_JOINED_STAGE"
    ),
    "profile": "embeddinggemma-whole-scaled-v1",
    "git_commit": git_commit,
    "git_dirty": int(bool(subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=repo_root, capture_output=True, text=True, check=True,
    ).stdout.strip())),
    "host": socket.gethostname(),
    "bundle_file": str(HFP),
    "bundle_payload_sha256": hashlib.sha256(bundle).hexdigest(),
    "cache_dir": str(CACHE),
    "m": M,
    "k": K,
    "n": N,
    "bundle_bytes": len(bundle),
    "shared_input_bytes": len(shared_input),
    "stage_bytes": R_BYTES,
    "projected_value_bytes": M * N * 2,
    "warmup": warmup,
    "iterations": iterations,
    "npu_time_ns": npu_time_ns,
    "npu_time_min_ns": min(npu_times),
    "npu_time_max_ns": max(npu_times),
    "npu_ms": npu_seconds * 1e3,
    "host_call_ms": statistics.median(host_times) * 1e3,
    "useful_tops": 2 * M * K * N / npu_seconds / 1e12,
    "checked_outputs": M * N,
    "mismatches": mismatches,
    "max_abs_vs_bf16_oracle": float(absolute.max()),
    "attention_tail_mismatches": tail_mismatches,
    "padding_mismatches": padding_mismatches,
    "full_qkv_bf16_oracle": "pass",
    "output_layout": (
        "r29-raw-attention-bf16"
        if STAGE_LAYOUT == "inline"
        else "r28-joined-padded-bf16"
    ),
    "global_weight_reorder_in_kernel": False,
}
print(json.dumps(row, sort_keys=True))
