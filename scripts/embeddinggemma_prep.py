#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# hipfire — embeddinggemma → hipfire-quantize prep (offline tooling, NOT hot path).
#
# google/embeddinggemma-300m ships as a sentence-transformers bundle:
#   - model.safetensors        : a Gemma3TextModel (tensors WITHOUT the `model.`
#                                prefix, no lm_head), all F32
#   - 2_Dense/model.safetensors: `linear.weight` [3072, 768]  (768 -> 3072)
#   - 3_Dense/model.safetensors: `linear.weight` [768, 3072]  (3072 -> 768)
#
# hipfire's Gemma3 loader (reused by hipfire-arch-embeddinggemma) expects the
# `model.*` naming and folds the two Dense heads in as `dense.{0,1}.weight`. This
# script rewrites the backbone safetensors with the `model.` prefix, appends the two
# Dense heads, and writes a merged config.json carrying the sentence-transformers
# post-processing block (pooling mode, Dense dims, Matryoshka dims, task prompts) so
# the quantizer embeds it in the .hfq metadata. Pure stdlib — no torch/safetensors.
#
# Usage:
#   python3 scripts/embeddinggemma_prep.py <hf_snapshot_dir> <out_dir>
# Then:
#   hipfire-quantize --input <out_dir> --output embeddinggemma-300m.bf16.hfq \
#                    --format bf16   # arch auto-detects to 17 via model_type

import json
import os
import shutil
import struct
import sys


def read_safetensors(path):
    """Return (header_dict, raw_data_bytes). Keys map name -> {dtype,shape,data_offsets}."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        data = f.read()
    return header, data


def write_safetensors(path, tensors):
    """tensors: list of (name, dtype, shape, raw_bytes). Writes a valid safetensors file."""
    header = {}
    blob = bytearray()
    for name, dtype, shape, raw in tensors:
        start = len(blob)
        blob.extend(raw)
        end = len(blob)
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [start, end]}
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    # safetensors requires 8-byte alignment of the header length is not mandated,
    # but the data must directly follow the header; pad header to 8 for cleanliness.
    pad = (-len(header_bytes)) % 8
    header_bytes += b" " * pad
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        f.write(blob)


def slice_tensor(header, data, name):
    info = header[name]
    s, e = info["data_offsets"]
    return info["dtype"], info["shape"], data[s:e]


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    src, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)

    # ── 1. backbone: rename every tensor to the `model.` convention ──
    bb_header, bb_data = read_safetensors(os.path.join(src, "model.safetensors"))
    out_tensors = []
    n_renamed = 0
    for name in bb_header:
        if name == "__metadata__":
            continue
        dtype, shape, raw = slice_tensor(bb_header, bb_data, name)
        # Gemma3TextModel: embed_tokens/layers/norm at the top level → prefix `model.`
        new_name = name if name.startswith("model.") else f"model.{name}"
        out_tensors.append((new_name, dtype, shape, raw))
        n_renamed += 1

    # ── 2. Dense heads → dense.0.weight / dense.1.weight ──
    dense_dims = []
    for i, sub in enumerate(("2_Dense", "3_Dense")):
        dh, dd = read_safetensors(os.path.join(src, sub, "model.safetensors"))
        dtype, shape, raw = slice_tensor(dh, dd, "linear.weight")
        out_tensors.append((f"dense.{i}.weight", dtype, shape, raw))
        # shape is [out_features, in_features]
        dense_dims.append({"in_features": shape[1], "out_features": shape[0],
                           "bias": False, "activation": "identity"})

    write_safetensors(os.path.join(out, "model.safetensors"), out_tensors)
    print(f"wrote model.safetensors: {n_renamed} backbone tensors + {len(dense_dims)} Dense heads")

    # ── 3. merged config.json with the sentence-transformers block ──
    with open(os.path.join(src, "config.json")) as f:
        cfg = json.load(f)
    # Tag for unambiguous hipfire arch detection (arch_id 17).
    cfg["model_type"] = "embeddinggemma"
    # Pooling config (1_Pooling/config.json) → mode string.
    with open(os.path.join(src, "1_Pooling", "config.json")) as f:
        pool = json.load(f)
    mode = "mean"
    if pool.get("pooling_mode_lasttoken"):
        mode = "lasttoken"
    elif pool.get("pooling_mode_cls_token"):
        mode = "cls"
    # Task prompts (config_sentence_transformers.json).
    prompts = {}
    st_path = os.path.join(src, "config_sentence_transformers.json")
    if os.path.exists(st_path):
        with open(st_path) as f:
            prompts = json.load(f).get("prompts", {})
    cfg["sentence_transformers"] = {
        "pooling_mode": mode,
        "dense": dense_dims,
        # embeddinggemma-300m officially supports these Matryoshka lengths.
        "matryoshka_dims": [768, 512, 256, 128],
        "query_prompt": prompts.get("query", "task: search result | query: "),
        "document_prompt": prompts.get("document", "title: none | text: "),
    }
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"wrote config.json (model_type=embeddinggemma, pooling={mode}, dense={dense_dims})")

    # ── 4. copy the tokenizer so the quantizer embeds it ──
    for fn in ("tokenizer.json", "tokenizer.model", "tokenizer_config.json",
               "special_tokens_map.json", "added_tokens.json", "generation_config.json"):
        p = os.path.join(src, fn)
        if os.path.exists(p):
            shutil.copy(p, os.path.join(out, fn))
    print("copied tokenizer + aux files")
    print(f"done → {out}")


if __name__ == "__main__":
    main()
