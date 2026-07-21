#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
"""Add the flash-style attention kernel to a dflash body manifest.

Unlike `swap_attn_manifest.py`, this is NOT a pure xclbin swap. The flash
kernel's host ABI differs from the sc kernel's on every axis:

  * iterations are q-heads, not kv-heads (so `n_iters` = NH / (q_len // B));
  * Q, Kᵀ and V are pre-tiled into the aie::mmul<4,8,4> block layout;
  * KV arrives as `n_tiles` fixed-size tiles, each carrying an additive f32
    score mask so `tot` need not be a multiple of `kv_tile`;
  * O comes back in the tiled C-layout.

So this writes the flash entry ALONGSIDE the sc one under a distinct
`dflash_attn_flash:*` key. `dflash_body_native.rs` picks it up only with
`--attn flash`, leaving the sc path as the default fallback.

  source tools/npu/npuenv.sh
  npupy tools/npu/swap_attn_flash_manifest.py \
        --in /tmp/dflash_manifest_mc4.json --out /tmp/dflash_manifest_flash.json \
        --tot 48 --block 16 --n-q 32 --q-len 16 --kv-tile 48
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", type=Path, required=True)
    ap.add_argument("--out", dest="dst", type=Path, required=True)
    ap.add_argument("--tot", type=int, required=True, help="ctx + block")
    ap.add_argument("--block", type=int, default=16)
    ap.add_argument("--n-q", type=int, default=32, help="query heads")
    ap.add_argument("--q-len", type=int, default=16)
    ap.add_argument("--kv-tile", type=int, default=48)
    ap.add_argument("--kv-depth", type=int, default=1)
    ap.add_argument("--cores", type=int, default=4)
    args = ap.parse_args()

    from build_dflash_attention_flash import dflash_attn_flash_mc

    if args.q_len % args.block:
        raise SystemExit(f"q_len={args.q_len} must be a multiple of block={args.block}")
    if args.kv_tile % 16:
        raise SystemExit(f"kv_tile={args.kv_tile} must be a multiple of 16")
    heads_per_iter = args.q_len // args.block
    n_iters = args.n_q // heads_per_iter
    if n_iters % args.cores:
        raise SystemExit(f"n_iters={n_iters} not divisible by cores={args.cores}")
    # tail masking: pad tot up to a whole number of tiles
    n_tiles = (args.tot + args.kv_tile - 1) // args.kv_tile

    ca = dict(q_len=args.q_len, kv_tile=args.kv_tile, n_tiles=n_tiles,
              n_iters=n_iters, n_cores=args.cores, kv_depth=args.kv_depth)
    xcl, ins = dflash_attn_flash_mc.specialize(**ca).compile()

    m = json.loads(args.src.read_text())
    key = f"dflash_attn_flash:q{args.q_len}_kvt{args.kv_tile}_nt{n_tiles}_it{n_iters}"
    m["kernels"][key] = {
        "xclbin": str(xcl), "insts": str(ins),
        "xclbin_bytes": os.path.getsize(xcl), "insts_bytes": os.path.getsize(ins),
        "compile_args": ca,
    }
    args.dst.write_text(json.dumps(m, indent=1))
    print(f"[flash] + {key}  (tot={args.tot} -> {n_tiles}x{args.kv_tile}, "
          f"pad {n_tiles * args.kv_tile - args.tot})")
    print(f"[flash] xclbin {xcl}\n[flash] wrote {args.dst}")


if __name__ == "__main__":
    main()
