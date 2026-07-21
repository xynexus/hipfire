#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
"""Phase B: FULL-rotation RoPE parity for the DFlash drafter on the NPU.

The DFlash draft (Qwen3) rotates the ENTIRE head_dim (neox / half-split) at
rope_theta=1e7 — unlike Qwen3.5's n_rot=head_dim/4 partial rotary. Validates
`dflash_rope_bf16.cc` (built by build_qwen3_dflash_rope.py) by reusing the host
runtime + float reference from test_rope_npu.py with n_rot = head_dim and the
sidecar's freq_base.

Build first:
  python tools/npu/build_qwen3_dflash_rope.py --n-heads 32 --n-kv-heads 8 --head-dim 128
Run:
  python tools/npu/test_dflash_rope_npu.py --n-heads 32 --n-kv-heads 8 --head-dim 128
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_rope_npu as tr  # noqa: E402  (reuses run_one / reference_rope host machinery)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-heads", type=int, default=32)
    p.add_argument("--n-kv-heads", type=int, default=8)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--rope-theta", type=float, default=1e7)  # z-lab 9B drafter
    p.add_argument("--xclbin-dir", type=Path, default=REPO_ROOT / "target/npu")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--timed", type=int, default=100)
    args = p.parse_args()

    hd = args.head_dim
    n_rot = hd  # FULL rotation
    d = args.xclbin_dir
    specs = [
        ("Q", args.n_heads, d / f"dflash-rope-q-{args.n_heads}h{hd}d.xclbin",
         d / f"dflash-rope-q-{args.n_heads}h{hd}d-instr.bin"),
        ("K", args.n_kv_heads, d / f"dflash-rope-k-{args.n_kv_heads}h{hd}d.xclbin",
         d / f"dflash-rope-k-{args.n_kv_heads}h{hd}d-instr.bin"),
    ]
    for _, _, xcl, instr in specs:
        for f in (xcl, instr):
            if not f.exists():
                raise FileNotFoundError(
                    f"{f} not found. Build: python tools/npu/build_qwen3_dflash_rope.py "
                    f"--n-heads {args.n_heads} --n-kv-heads {args.n_kv_heads} --head-dim {hd}")

    print(f"=== DFlash RoPE (FULL rotation) NPU test: head_dim={hd} n_rot={n_rot} "
          f"theta={args.rope_theta:.0e} ===")
    ok = True
    for label, nh, xcl, instr in specs:
        r = tr.run_one(label, nh, hd, n_rot, xcl, instr, args.warmup, args.timed,
                       freq_base=args.rope_theta)
        ok &= r["max_abs"] < 0.1  # bf16 rope tolerance
    print(f"=== {'PASS' if ok else 'FAIL'} ===")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
