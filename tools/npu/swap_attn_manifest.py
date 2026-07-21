#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
"""Swap the whole-layer attention kernel in a dflash body manifest.

Rewrites the `dflash_attn_all:*` entry to the N-core `dflash_attn_mc` build (key
`dflash_attn_all_mc<N>:*`, which still matches dflash_body_native.rs's
"dflash_attn_all" prefix). The dispatch ABI and buffer sizes are identical, so
this is a pure xclbin swap — no Rust change, no re-run of the Python body.

  source tools/npu/npuenv.sh
  npupy tools/npu/swap_attn_manifest.py --in /tmp/dflash_manifest.json \
        --out /tmp/dflash_manifest_mc4.json --cores 4
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
    ap.add_argument("--cores", type=int, default=4)
    args = ap.parse_args()

    from build_dflash_attention_sc import dflash_attn_mc

    m = json.loads(args.src.read_text())
    kernels = m["kernels"]
    old = [k for k in kernels if k.startswith("dflash_attn_all")]
    if len(old) != 1:
        raise SystemExit(f"expected exactly one attention kernel, found {old}")
    old_key = old[0]
    ca = dict(kernels[old_key]["compile_args"])
    ca["n_cores"] = args.cores

    xcl, ins = dflash_attn_mc.specialize(**ca).compile()
    new_key = (f"dflash_attn_all_mc{args.cores}:"
               f"q{ca['q_len']}_kv{ca['kv_len']}_it{ca['n_iters']}")
    del kernels[old_key]
    kernels[new_key] = {
        "xclbin": str(xcl), "insts": str(ins),
        "xclbin_bytes": os.path.getsize(xcl), "insts_bytes": os.path.getsize(ins),
        "compile_args": ca,
    }
    for d in m.get("dispatches", []):
        if d.get("kernel") == old_key:
            d["kernel"] = new_key

    args.dst.write_text(json.dumps(m, indent=1))
    print(f"[swap] {old_key} -> {new_key}\n[swap] xclbin {xcl}\n[swap] wrote {args.dst}")


if __name__ == "__main__":
    main()
