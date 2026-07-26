#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# hipfire — W4A8 quality-gate helper (offline tooling, NOT hot path).
#
# Simulates the r14 Phoenix NPU GEMM's weight precision — plain symmetric int4,
# per-group G=256 along the contraction dim K — by quantize→dequantizing each
# backbone projection weight of an embeddinggemma prep dir, keeping the sensitive
# Dense heads, norms, and embedding table at full precision. The result is imported
# through the normal bf16 path; comparing its embeddings to the true bf16 model
# (cosine) is the go/no-go for a W4A8 NPU forward.
#
# Usage: python3 scripts/embeddinggemma_w4_sim.py <prep_dir> <out_dir> [--dense]
#   --dense also quantizes the Dense heads (the hardest case).

import json
import os
import shutil
import struct
import sys

import numpy as np

GROUP = 256
PROJ = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def read_safetensors(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        data = f.read()
    return header, data


def write_safetensors(path, tensors):
    header, blob = {}, bytearray()
    for name, dtype, shape, raw in tensors:
        s = len(blob)
        blob.extend(raw)
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [s, len(blob)]}
    hb = json.dumps(header, separators=(",", ":")).encode()
    hb += b" " * ((-len(hb)) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        f.write(blob)


def quant_dequant_int(w, group=GROUP, bits=4):
    """Symmetric per-group int`bits` round-trip along the last (K) axis. Groups of
    `group`; a final short group (e.g. K=1152 → 4×256 + 128) gets its own scale,
    matching a K-padded NPU kernel. Returns the dequantized f32."""
    qmax = float((1 << (bits - 1)) - 1)  # int4→7, int8→127
    rows, k = w.shape
    out = np.empty_like(w, dtype=np.float32)
    for g0 in range(0, k, group):
        g1 = min(g0 + group, k)
        seg = w[:, g0:g1]
        amax = np.abs(seg).max(axis=1, keepdims=True)
        scale = np.where(amax > 0, amax / qmax, 1.0)
        q = np.clip(np.round(seg / scale), -qmax, qmax)
        out[:, g0:g1] = q * scale
    return out


def main():
    src, out = sys.argv[1], sys.argv[2]
    do_dense = "--dense" in sys.argv[3:]
    bits = 4
    group = GROUP
    for i, a in enumerate(sys.argv):
        if a == "--group":
            group = int(sys.argv[i + 1])
        if a == "--bits":
            bits = int(sys.argv[i + 1])
    os.makedirs(out, exist_ok=True)

    h, data = read_safetensors(os.path.join(src, "model.safetensors"))
    tensors, n_q = [], 0
    for name in h:
        if name == "__metadata__":
            continue
        info = h[name]
        s, e = info["data_offsets"]
        raw = data[s:e]
        is_proj = any(p in name for p in PROJ) and name.endswith(".weight")
        is_dense = name.startswith("dense.") and do_dense
        if info["dtype"] == "F32" and (is_proj or is_dense) and len(info["shape"]) == 2:
            w = np.frombuffer(raw, dtype=np.float32).reshape(info["shape"]).copy()
            wq = quant_dequant_int(w, group, bits)
            raw = wq.tobytes()
            n_q += 1
        tensors.append((name, info["dtype"], info["shape"], raw))

    write_safetensors(os.path.join(out, "model.safetensors"), tensors)
    with open(os.path.join(src, "config.json")) as f:
        cfg = json.load(f)
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    for fn in ("tokenizer.json", "tokenizer.model", "tokenizer_config.json",
               "special_tokens_map.json", "added_tokens.json", "generation_config.json"):
        p = os.path.join(src, fn)
        if os.path.exists(p):
            shutil.copy(p, os.path.join(out, fn))
    print(f"w4-sim: int4-roundtripped {n_q} weight tensors "
          f"({'backbone+dense' if do_dense else 'backbone projections only'}) → {out}")


if __name__ == "__main__":
    main()
