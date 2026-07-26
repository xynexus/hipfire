#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
"""Build the DFlash-drafter NON-CAUSAL cross-attention image.

The DFlash draft block attends bidirectionally (non-causal) over KV =
concat(projected target_hidden context, current block K/V): Q length =
block_size, K/V length = ctx_len + block_size. The compute kernel
(`segmented_attention_bf16.cc`) already takes `causal` as a RUNTIME flag and
`real_length` for window masking, so the ONLY differences from the causal
prefill image are:

  1. causal = 0  (bidirectional within the block).
  2. the caller stages KV = [ctx | block] and Q as the full (ctx+block) token
     sequence padded to the bucket, with real_length = ctx_len + block_size;
     the block's outputs are the last `block_size` rows.

This reuses `build_qwen3_segmented_attention.generate_mlir` verbatim and patches
only the causal constant (and the device, so it also builds for npu1/aie2 on
nix1 — the drafter's target — not just npu2/aie2p).

The `.cc` compute is UNCHANGED (arch-neutral: aie::mmul<4,8,8,bf16,bf16> exists
on both aie2 and aie2p). Validate against the Phase-A golden l0 attention
tensors (rust_l0_q_roped / k_roped / v / attn_out).

Usage:
  python tools/npu/build_qwen3_dflash_attention.py --bucket 128 --batch 1 \
      --query-heads 32 --kv-heads 8 --head-dim 128 --npu npu1 \
      --output target/npu/dflash-attn
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import build_qwen3_segmented_attention as seg  # noqa: E402  (reuse generate_mlir + toolchain)

_DEVICE = {"npu1": ("npu1", "aie2"), "npu2": ("npu2", "aie2p")}


def generate_noncausal_mlir(bucket, batch, query_heads, kv_heads, head_dim, npu):
    """Reuse the causal generator, then patch causal=0 and the device line."""
    mlir = seg.generate_mlir(bucket, batch, query_heads, kv_heads, head_dim)
    dev = _DEVICE[npu][0]
    # bidirectional within the block: drop the causal mask.
    mlir = mlir.replace("%causal = arith.constant 1 : i32",
                        "%causal = arith.constant 0 : i32")
    # retarget the device (npu1/aie2 for nix1, npu2/aie2p for halo).
    mlir = mlir.replace("aie.device(npu2)", f"aie.device({dev})")
    assert "%causal = arith.constant 0 : i32" in mlir, "causal patch failed"
    return mlir


def build(output: Path, bucket, batch, query_heads, kv_heads, head_dim, npu, emit_mlir_only):
    output.mkdir(parents=True, exist_ok=True)
    target_arch = _DEVICE[npu][1]
    mlir = output / "aie.mlir"
    mlir.write_text(generate_noncausal_mlir(bucket, batch, query_heads, kv_heads, head_dim, npu),
                    encoding="utf-8")
    if not emit_mlir_only:
        mlir_aie, peano = seg._toolchain()
        env = os.environ.copy()
        env["PATH"] = os.pathsep.join(
            ["/opt/xilinx/xrt/bin", str(peano / "bin"), str(mlir_aie / "bin"), env.get("PATH", "")])
        source = SCRIPT_DIR / "segmented_attention_bf16.cc"
        subprocess.run(
            [str(peano / "bin/clang++"), str(source), "-c", "-o",
             str(output / "segmented_attention.o"), f"-I{mlir_aie / 'include'}",
             "-std=c++20", "-O2", "-DNDEBUG", "-Wno-parentheses", "-Wno-attributes",
             "-Wno-macro-redefined", "-Wno-empty-body", "-Wno-deprecated-declarations",
             f"--target={target_arch}-none-unknown-elf"],
            check=True, env=env)
        aiecc = shutil.which("aiecc", path=env["PATH"])
        if aiecc is None:
            raise RuntimeError("aiecc not found")
        subprocess.run(
            [aiecc, str(mlir), "--no-compile-host", "--no-xchesscc", "--no-xbridge",
             f"--peano={peano}", "--aie-generate-npu-insts",
             f"--npu-insts-name={output / 'insts.bin'}", "--aie-generate-xclbin",
             f"--xclbin-name={output / 'final.xclbin'}", f"--tmpdir={output}"],
            check=True, env=env)
    manifest = {
        "schema": "hipfire.npu_dflash_attention.v1",
        "npu_architecture": target_arch,
        "attention": "non_causal_cross",
        "bucket": bucket, "batch": batch, "query_heads": query_heads,
        "kv_heads": kv_heads, "head_dim": head_dim,
        "note": "causal=0; KV=[ctx|block], real_length=ctx+block, block outputs are last block_size rows",
        "xclbin": "final.xclbin", "instructions": "insts.bin",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bucket", type=int, default=128, choices=seg.BUCKETS)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--query-heads", type=int, choices=(16, 32), default=32)
    p.add_argument("--kv-heads", type=int, choices=(8,), default=8)
    p.add_argument("--head-dim", type=int, choices=(128,), default=128)
    p.add_argument("--npu", choices=("npu1", "npu2"), default="npu1")
    p.add_argument("--output", type=Path, default=SCRIPT_DIR.parent.parent / "target/npu/dflash-attn")
    p.add_argument("--emit-mlir-only", action="store_true")
    args = p.parse_args()
    try:
        build(args.output, args.bucket, args.batch, args.query_heads, args.kv_heads,
              args.head_dim, args.npu, args.emit_mlir_only)
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
