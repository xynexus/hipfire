"""
Reconstruction-error sweep for down_proj on Qwen3.6-A3B.

For a sampled set of expert rows from the worst-tailed layers, compare:
  1. MQ4G256-FWHT (current): per-256-group, FWHT-rotated, 4-bit asymm
  2. MQ4G64-FWHT: smaller group (256→64) — outliers contained to fewer neighbors
  3. MQ4G256-FWHT + outlier sidecar: top-N abs values per row stored FP16,
     residual quantized MQ4G256-FWHT
  4. MQ6G256-FWHT (current upper bound)

Outputs per-scheme:
  - mean MSE per row
  - p99 MSE per row
  - mean cosine-similarity vs original

If outlier-sidecar at small N (say 8 per row) beats MQ4G256 by a margin
larger than MQ6's beat — and at lower bytes — it's worth prototyping in
the quantizer.

This is a *simulation* — it does NOT exercise the actual quant kernel; it
runs the equivalent math in NumPy. Confirms whether outlier-aware schemes
actually help BEFORE we touch the engine.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from safetensors import safe_open

try:
    import ml_dtypes
    HAS_ML_DTYPES = True
except ImportError:
    HAS_ML_DTYPES = False


def _read_tensor(f, key) -> np.ndarray:
    s = f.get_slice(key)
    shape = tuple(s.get_shape())
    dtype = s.get_dtype()
    if dtype == "BF16":
        raw = s[:].tobytes() if hasattr(s[:], "tobytes") else bytes(s[:])
        if HAS_ML_DTYPES:
            arr = np.frombuffer(raw, dtype=ml_dtypes.bfloat16).reshape(shape)
            return arr.astype(np.float32)
        u16 = np.frombuffer(raw, dtype="<u2").astype(np.uint32)
        return ((u16 << 16).view(np.float32)).reshape(shape)
    return np.asarray(f.get_tensor(key), dtype=np.float32)


def fwht256(x: np.ndarray, signs1: np.ndarray, signs2: np.ndarray) -> np.ndarray:
    """In-place FWHT of length 256 with two sign sequences (one pre, one post).
    Matches the engine's `cpu_fwht_256(group, signs1, signs2)` semantics:
      group = signs2 * fwht256_raw(signs1 * group)
    """
    out = x * signs1
    h = 1
    while h < 256:
        for i in range(0, 256, h * 2):
            for j in range(h):
                a = out[i + j]
                b = out[i + j + h]
                out[i + j] = a + b
                out[i + j + h] = a - b
        h *= 2
    out = out / np.sqrt(256.0)
    return out * signs2


def inv_fwht256(x: np.ndarray, signs1: np.ndarray, signs2: np.ndarray) -> np.ndarray:
    """Inverse of fwht256: signs1 * fwht256_raw(signs2 * x). Self-inverse up to
    sign placement."""
    out = x * signs2
    h = 1
    while h < 256:
        for i in range(0, 256, h * 2):
            for j in range(h):
                a = out[i + j]
                b = out[i + j + h]
                out[i + j] = a + b
                out[i + j + h] = a - b
        h *= 2
    out = out / np.sqrt(256.0)
    return out * signs1


def make_signs(seed: int = 0xCAFEBABE) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    s1 = rng.choice([-1.0, 1.0], size=256).astype(np.float32)
    s2 = rng.choice([-1.0, 1.0], size=256).astype(np.float32)
    return s1, s2


def quantize_mq4g256_then_recon(group: np.ndarray, s1, s2) -> np.ndarray:
    """Simulate MQ4G256: FWHT, asymm 4-bit per-group, dequant, inv-FWHT."""
    rotated = fwht256(group, s1, s2)
    mn, mx = rotated.min(), rotated.max()
    rng = mx - mn
    if rng <= 0:
        return group.copy()
    scale = rng / 15.0
    q = np.clip(np.round((rotated - mn) / scale), 0, 15)
    deq = q * scale + mn
    return inv_fwht256(deq, s1, s2)


def quantize_mq6g256_then_recon(group: np.ndarray, s1, s2) -> np.ndarray:
    rotated = fwht256(group, s1, s2)
    mn, mx = rotated.min(), rotated.max()
    rng = mx - mn
    if rng <= 0:
        return group.copy()
    scale = rng / 63.0
    q = np.clip(np.round((rotated - mn) / scale), 0, 63)
    deq = q * scale + mn
    return inv_fwht256(deq, s1, s2)


def quantize_mq4g64_then_recon(row: np.ndarray) -> np.ndarray:
    """No FWHT (G64 < 256). Per-group 4-bit asymm, group_size=64."""
    out = np.zeros_like(row)
    for g in range(0, len(row), 64):
        grp = row[g:g + 64]
        mn, mx = grp.min(), grp.max()
        rng = mx - mn
        if rng <= 0:
            out[g:g + 64] = grp
            continue
        scale = rng / 15.0
        q = np.clip(np.round((grp - mn) / scale), 0, 15)
        out[g:g + 64] = q * scale + mn
    return out


def quantize_mq4g256_outlier_sidecar(row: np.ndarray, n_outliers: int, s1, s2) -> np.ndarray:
    """Pull top-N |values| out of the row, store FP16, quant residual MQ4G256."""
    abs_row = np.abs(row)
    top_idx = np.argpartition(abs_row, -n_outliers)[-n_outliers:]
    out_vals = row[top_idx].astype(np.float16).astype(np.float32)  # FP16 round-trip
    residual = row.copy()
    residual[top_idx] = 0.0
    recon = np.zeros_like(row)
    for g in range(0, len(row), 256):
        grp = residual[g:g + 256]
        if len(grp) < 256:
            grp = np.concatenate([grp, np.zeros(256 - len(grp), dtype=np.float32)])
        recon[g:g + 256] = quantize_mq4g256_then_recon(grp, s1, s2)[:len(residual[g:g + 256])]
    recon[top_idx] = out_vals
    return recon


def row_metrics(orig: np.ndarray, recon: np.ndarray) -> dict:
    err = orig - recon
    mse = float(np.mean(err * err))
    cos = float(np.dot(orig, recon) / (np.linalg.norm(orig) * np.linalg.norm(recon) + 1e-12))
    return {"mse": mse, "cos": cos}


def quantize_row_g256(row: np.ndarray, fn, s1, s2) -> np.ndarray:
    out = np.zeros_like(row)
    for g in range(0, len(row), 256):
        grp = row[g:g + 256]
        if len(grp) < 256:
            grp = np.concatenate([grp, np.zeros(256 - len(grp), dtype=np.float32)])
        out[g:g + 256] = fn(grp, s1, s2)[:len(row[g:g + 256])]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-rows", type=int, default=512,
                   help="rows to sample per (layer, expert); cheap-test default 512")
    p.add_argument("--layers", default=None,
                   help="comma-separated layer indices; default = pick worst 2")
    args = p.parse_args()

    snap = Path(args.snapshot)
    shards = sorted(snap.glob("model-*.safetensors"))

    # Layer-tail diagnostic first so we sample from worst layers.
    if args.layers:
        layers_to_test = [int(s) for s in args.layers.split(",")]
    else:
        # Sample 2 layers — the tail-diagnostic showed worst layers are extremes
        # of the 40-layer span. We pick mid (20) and high (35) as a reasonable
        # blend.
        layers_to_test = [10, 20, 35]

    s1, s2 = make_signs()

    schemes = ["MQ4G256", "MQ4G64", "MQ6G256",
               "MQ4G256+sidecar4", "MQ4G256+sidecar16", "MQ4G256+sidecar64"]
    accum: dict[str, dict[str, list[float]]] = {
        s: {"mse": [], "cos": []} for s in schemes
    }
    layer_summaries: dict[int, dict] = {}

    for layer_idx in layers_to_test:
        print(f"[layer {layer_idx}] scanning shards", file=sys.stderr)
        target_key = f"model.language_model.layers.{layer_idx}.mlp.experts.down_proj"
        found = None
        for shard in shards:
            with safe_open(shard, framework="numpy") as f:
                if target_key in f.keys():
                    found = shard
                    print(f"[layer {layer_idx}] in {shard.name}", file=sys.stderr)
                    arr = _read_tensor(f, target_key)  # [256 experts, 2048, 512]
                    break
        if found is None:
            print(f"[layer {layer_idx}] NOT FOUND — skipping", file=sys.stderr)
            continue

        # arr: experts × M(=output rows) × K(=input cols)
        # down_proj is [M=2048, K=512]; we quantize per row of K=512.
        # Wait — reshape says [256 experts, 2048, 512]. So output dim is 2048,
        # input dim is 512. Quant runs along K=512 (input dim).
        n_expert, m, k = arr.shape
        print(f"[layer {layer_idx}] shape ({n_expert} experts, {m} out, {k} in)", file=sys.stderr)

        # Sample rows: pick experts evenly + worst-tailed rows within them
        rng = np.random.default_rng(layer_idx)
        sample_experts = rng.choice(n_expert, size=4, replace=False)
        rows_per_expert = max(1, args.max_rows // len(sample_experts))
        layer_summaries[layer_idx] = {"sampled_rows": 0, "per_expert": {}}

        for e in sample_experts:
            mat = arr[e]  # [m, k]
            abs_row = np.abs(mat).max(axis=1)
            med_row = np.median(np.abs(mat), axis=1)
            ratio = abs_row / np.maximum(med_row, 1e-12)
            # Pick the rows_per_expert WORST rows by tail ratio
            worst_idx = np.argpartition(ratio, -rows_per_expert)[-rows_per_expert:]
            for r in worst_idx:
                row = mat[r].astype(np.float32)  # [k=512]
                # k=512 isn't a multiple of 256, but is exactly 2 groups
                # MQ4G256
                rec = quantize_row_g256(row, quantize_mq4g256_then_recon, s1, s2)
                m1 = row_metrics(row, rec); accum["MQ4G256"]["mse"].append(m1["mse"]); accum["MQ4G256"]["cos"].append(m1["cos"])
                # MQ4G64
                rec = quantize_mq4g64_then_recon(row)
                m1 = row_metrics(row, rec); accum["MQ4G64"]["mse"].append(m1["mse"]); accum["MQ4G64"]["cos"].append(m1["cos"])
                # MQ6G256
                rec = quantize_row_g256(row, quantize_mq6g256_then_recon, s1, s2)
                m1 = row_metrics(row, rec); accum["MQ6G256"]["mse"].append(m1["mse"]); accum["MQ6G256"]["cos"].append(m1["cos"])
                # Sidecars
                for n_o, name in [(4, "sidecar4"), (16, "sidecar16"), (64, "sidecar64")]:
                    rec = quantize_mq4g256_outlier_sidecar(row, n_o, s1, s2)
                    m1 = row_metrics(row, rec); accum[f"MQ4G256+{name}"]["mse"].append(m1["mse"]); accum[f"MQ4G256+{name}"]["cos"].append(m1["cos"])
            layer_summaries[layer_idx]["per_expert"][int(e)] = int(rows_per_expert)
            layer_summaries[layer_idx]["sampled_rows"] += rows_per_expert

    summary = {}
    for s in schemes:
        if not accum[s]["mse"]:
            continue
        mse_arr = np.asarray(accum[s]["mse"])
        cos_arr = np.asarray(accum[s]["cos"])
        summary[s] = {
            "mean_mse": float(mse_arr.mean()),
            "p50_mse": float(np.median(mse_arr)),
            "p99_mse": float(np.percentile(mse_arr, 99)),
            "mean_cos": float(cos_arr.mean()),
            "p01_cos": float(np.percentile(cos_arr, 1)),
            "n_rows": len(mse_arr),
        }

    out = {
        "snapshot": str(snap),
        "layers_tested": layers_to_test,
        "layer_summaries": layer_summaries,
        "schemes": summary,
        "interpretation": (
            "Compare 'mean_mse' and 'p99_mse'. MQ6G256 is the precision upper bound. "
            "If MQ4G256+sidecarN approaches or beats MQ6G256 at fewer bytes per row "
            "(sidecar = 4 bytes/outlier vs MQ6's +50%/group), it's worth productizing. "
            "If MQ4G64 alone closes most of the gap, smaller-group quant is the simpler win."
        ),
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(summary, indent=2), file=sys.stderr)


if __name__ == "__main__":
    main()
