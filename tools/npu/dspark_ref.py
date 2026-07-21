#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire - see LICENSE and NOTICE in the project root.

"""Numpy reference for the DSpark head epilogue (DFlash Phase E, step 1).

Mirrors `crates/hipfire-specdecode-dspark/src/dspark_core.rs::run_heads` plus the
confidence-threshold truncation in `DsparkDrafter::mtp_step`, at f32 precision.
This is the golden the NPU head kernels are validated against.

The `lm_head` GEMV belongs to the TARGET model, not the drafter, so it is NOT
ported: `logits[block, vocab]` is supplied (synthetic, fixed-seed) and the heads
are validated on top of it.

Sidecar: /srv/hipfire/models/qwen3-0.6b.bf16.dspark.hfq (HFQ arch_id 22).
NOTE: this is a qwen3-0.6b drafter (hidden 1024), NOT the Qwen3.5-9B DFlash body
validated in Phases A-D. Gate E validates the head kernels at these dims.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import mmap
from pathlib import Path
import struct

import numpy as np

DEFAULT_SIDECAR = "/srv/hipfire/models/qwen3-0.6b.bf16.dspark.hfq"
# `DsparkDrafter::conf_threshold` default (survival probability, post-sigmoid).
DEFAULT_CONF_THRESHOLD = 0.5


@dataclass(frozen=True)
class TensorInfo:
    quant_type: int
    shape: tuple[int, ...]
    group_size: int
    offset: int
    size: int


class Hfq:
    """HFQM container reader.

    Header (32 B): magic "HFQM", u32 version, u32 arch_id, u32 tensor_count,
    u64 metadata_offset, u64 data_offset. Metadata is a JSON object at
    `metadata_offset`; the tensor index follows it (u32 count, then per tensor:
    u16 name_len, name, u8 quant_type, u8 rank, rank*u32 shape, u32 group_size,
    u64 byte_size). Tensor payloads are laid out sequentially from `data_offset`
    in index order.
    """

    def __init__(self, path: Path) -> None:
        self._file = path.open("rb")
        self.data = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        magic, self.version, self.arch_id, self.tensor_count = struct.unpack_from(
            "<4sIII", self.data, 0
        )
        if magic != b"HFQM":
            raise ValueError(f"{path} is not an HFQM container")
        metadata_offset, self.data_offset = struct.unpack_from("<QQ", self.data, 16)
        json_end = self._json_end(metadata_offset, self.data_offset)
        self.metadata = json.loads(
            self.data[metadata_offset : metadata_offset + json_end].decode()
        )
        position = metadata_offset + json_end
        count = struct.unpack_from("<I", self.data, position)[0]
        position += 4
        self.tensors: dict[str, TensorInfo] = {}
        running = self.data_offset
        for _ in range(count):
            name_len = struct.unpack_from("<H", self.data, position)[0]
            position += 2
            name = self.data[position : position + name_len].decode()
            position += name_len
            quant_type, rank = struct.unpack_from("<BB", self.data, position)
            position += 2
            shape = struct.unpack_from(f"<{rank}I", self.data, position)
            position += rank * 4
            group_size = struct.unpack_from("<I", self.data, position)[0]
            position += 4
            size = struct.unpack_from("<Q", self.data, position)[0]
            position += 8
            self.tensors[name] = TensorInfo(
                quant_type=quant_type,
                shape=tuple(shape),
                group_size=group_size,
                offset=running,
                size=size,
            )
            running += size
        if running != self.data.size():
            raise ValueError(
                f"HFQ index walk ended at {running}, file is {self.data.size()} bytes"
            )

    def _json_end(self, start: int, end: int) -> int:
        depth = 0
        in_string = False
        escaped = False
        for relative, byte in enumerate(self.data[start:end]):
            if escaped:
                escaped = False
            elif byte == ord("\\") and in_string:
                escaped = True
            elif byte == ord('"'):
                in_string = not in_string
            elif not in_string and byte == ord("{"):
                depth += 1
            elif not in_string and byte == ord("}"):
                depth -= 1
                if depth == 0:
                    return relative + 1
        raise ValueError("HFQM metadata JSON did not end")

    def close(self) -> None:
        self.data.close()
        self._file.close()

    def tensor(self, name: str) -> np.ndarray:
        """Dequantize a tensor to f32. Only the dtypes the heads use are handled."""
        info = self.tensors[name]
        payload = memoryview(self.data)[info.offset : info.offset + info.size]
        elements = int(np.prod(info.shape))
        if info.quant_type == 1:  # F16
            values = np.frombuffer(payload, dtype="<f2", count=elements).astype(np.float32)
        elif info.quant_type == 2:  # F32
            values = np.frombuffer(payload, dtype="<f4", count=elements).copy()
        elif info.quant_type == 16:  # BF16
            bits = np.frombuffer(payload, dtype="<u2", count=elements).astype(np.uint32)
            values = (bits << 16).view(np.float32)
        else:
            raise ValueError(
                f"{name}: quant_type={info.quant_type} not needed by the DSpark heads"
            )
        return values.reshape(info.shape)


@dataclass(frozen=True)
class DsparkConfig:
    block_size: int
    target_layer_ids: tuple[int, ...]
    markov_rank: int
    noise_token_id: int
    enable_confidence: bool
    confidence_uses_normed: bool
    rms_norm_eps: float

    @staticmethod
    def from_metadata(metadata: dict) -> "DsparkConfig":
        cfg = metadata["config"]
        return DsparkConfig(
            block_size=int(cfg["dspark_block_size"]),
            target_layer_ids=tuple(int(v) for v in cfg["dspark_target_layer_ids"]),
            markov_rank=int(cfg["dspark_markov_rank"]),
            noise_token_id=int(cfg["dspark_noise_token_id"]),
            enable_confidence=bool(cfg.get("dspark_enable_confidence", True)),
            confidence_uses_normed=bool(cfg.get("dspark_confidence_uses_normed", False)),
            rms_norm_eps=float(cfg.get("norm_eps", 1e-6)),
        )


class DsparkHeads:
    """The DSpark head weights, dequantized to f32 and held in host memory."""

    def __init__(self, path: Path = Path(DEFAULT_SIDECAR)) -> None:
        hfq = Hfq(path)
        try:
            self.cfg = DsparkConfig.from_metadata(hfq.metadata)
            self.stage_norm = hfq.tensor("norm.weight")  # qwen3 drafter final norm
            self.markov_w1 = hfq.tensor("markov_head.markov_w1.weight")
            self.markov_w2 = hfq.tensor("markov_head.markov_w2.weight")
            self.confidence_proj = hfq.tensor("confidence_head.proj.weight")
            self.confidence_bias = hfq.tensor("confidence_head.proj.bias")
        finally:
            hfq.close()
        self.hidden = int(self.stage_norm.shape[0])
        self.vocab = int(self.markov_w1.shape[0])
        assert self.markov_w1.shape == (self.vocab, self.cfg.markov_rank)
        assert self.markov_w2.shape == (self.vocab, self.cfg.markov_rank)
        assert self.confidence_proj.shape == (1, self.hidden + self.cfg.markov_rank)


def rmsnorm(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    """Row-wise RMSNorm, matching `gpu.rmsnorm_batched`."""
    x32 = x.astype(np.float32)
    rms = np.sqrt(np.mean(x32**2, axis=-1, keepdims=True) + eps)
    return (x32 / rms) * weight.astype(np.float32)


def argmax_f32(values: np.ndarray) -> int:
    """Strict `>` reduction with lowest-index tie-break, matching `argmax_f32`."""
    return int(np.argmax(values))


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-np.float32(x)))


@dataclass
class HeadResult:
    out_ids: np.ndarray  # [block + 1]; out_ids[0] == prev_token
    tokens: np.ndarray  # [block]; out_ids[1:]
    confidence: np.ndarray  # [block] RAW logits (pre-sigmoid), as run_heads returns
    survival: np.ndarray  # [block] sigmoid(confidence)
    normed: np.ndarray  # [block, hidden]
    confident_len: int  # truncation point from mtp_step


def run_heads(
    heads: DsparkHeads,
    x_head: np.ndarray,  # [block, hidden]
    logits: np.ndarray,  # [block, vocab] supplied (target lm_head output)
    prev_token: int,
    conf_threshold: float = DEFAULT_CONF_THRESHOLD,
) -> HeadResult:
    """f32 reference for `dspark_core::run_heads` + `mtp_step` truncation."""
    cfg = heads.cfg
    block, hidden = x_head.shape
    assert hidden == heads.hidden
    assert logits.shape == (block, heads.vocab)

    normed = rmsnorm(x_head, heads.stage_norm, cfg.rms_norm_eps)

    out_ids = np.full(block + 1, prev_token, dtype=np.int64)
    confidence = np.zeros(block, dtype=np.float32)
    proj = heads.confidence_proj[0].astype(np.float32)
    bias = float(heads.confidence_bias[0])

    for i in range(block):
        emb = heads.markov_w1[out_ids[i]].astype(np.float32)  # [rank]

        if cfg.enable_confidence:
            hidden_i = normed[i] if cfg.confidence_uses_normed else x_head[i]
            concat = np.concatenate([hidden_i.astype(np.float32), emb])
            confidence[i] = float(np.dot(proj, concat)) + bias

        bias_v = heads.markov_w2.astype(np.float32) @ emb  # [vocab]
        out_ids[i + 1] = argmax_f32(logits[i].astype(np.float32) + bias_v)

    if cfg.enable_confidence:
        survival = 1.0 / (1.0 + np.exp(-confidence.astype(np.float32)))
    else:
        confidence = np.full(block, np.inf, dtype=np.float32)
        survival = np.ones(block, dtype=np.float32)

    # `mtp_step`: truncate at the FIRST slot whose survival < threshold; keep >= 1.
    confident_len = block
    for i in range(block):
        if survival[i] < conf_threshold:
            confident_len = i
            break
    confident_len = max(confident_len, 1)

    return HeadResult(
        out_ids=out_ids,
        tokens=out_ids[1:],
        confidence=confidence,
        survival=survival,
        normed=normed,
        confident_len=confident_len,
    )


def synthetic_inputs(heads: DsparkHeads, seed: int, block: int | None = None):
    """Fixed-seed synthetic `x_head` and `logits`, so runs are reproducible.

    `logits` stands in for the TARGET model's lm_head output, which the drafter
    does not own. Scaled to the same order as the markov bias so the argmax is a
    genuine contest between the two terms rather than decided by one alone.
    """
    block = block or heads.cfg.block_size
    rng = np.random.default_rng(seed)
    x_head = rng.standard_normal((block, heads.hidden), dtype=np.float32)
    logits = rng.standard_normal((block, heads.vocab), dtype=np.float32).astype(np.float32)
    return x_head, logits


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sidecar", default=DEFAULT_SIDECAR)
    parser.add_argument("--seed", type=int, default=0, help="synthetic-input seed")
    parser.add_argument(
        "--prev-token", type=int, default=None, help="defaults to dspark_noise_token_id"
    )
    parser.add_argument("--conf-threshold", type=float, default=DEFAULT_CONF_THRESHOLD)
    parser.add_argument("--dump", type=Path, default=None, help="write results to .npz")
    args = parser.parse_args()

    heads = DsparkHeads(Path(args.sidecar))
    cfg = heads.cfg
    prev_token = args.prev_token if args.prev_token is not None else cfg.noise_token_id

    print(f"sidecar        {args.sidecar}")
    print(f"hidden={heads.hidden} vocab={heads.vocab} rank={cfg.markov_rank}")
    print(f"block_size={cfg.block_size} noise_token_id={cfg.noise_token_id}")
    print(
        f"enable_confidence={cfg.enable_confidence} "
        f"confidence_uses_normed={cfg.confidence_uses_normed} eps={cfg.rms_norm_eps}"
    )
    print(f"prev_token={prev_token} seed={args.seed} conf_threshold={args.conf_threshold}")

    x_head, logits = synthetic_inputs(heads, args.seed)
    result = run_heads(heads, x_head, logits, prev_token, args.conf_threshold)

    print("\nslot  out_id      conf(raw)     survival")
    for i in range(cfg.block_size):
        print(
            f"{i:4d}  {result.tokens[i]:9d}  {result.confidence[i]:+12.6f}  "
            f"{result.survival[i]:10.6f}"
        )
    print(f"\nconfident_len = {result.confident_len} / {cfg.block_size}")

    if args.dump:
        np.savez(
            args.dump,
            x_head=x_head,
            logits=logits,
            normed=result.normed,
            out_ids=result.out_ids,
            confidence=result.confidence,
            survival=result.survival,
            confident_len=result.confident_len,
            prev_token=prev_token,
        )
        print(f"wrote {args.dump}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
