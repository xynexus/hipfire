"""
Per-expert weight magnitude diagnostics for Qwen3.5/3.6-A3B safetensors.

For each MoE-expert weight tensor (gate_up_proj, down_proj, by layer × expert):
  - Compute per-output-row absmax / per-row median ratio (scale tail-heaviness).
  - Aggregate across experts and across layers per model.
  - Compare 3.5 vs 3.6 distributional summary.

Hypothesis (axis 1 from the post-compaction dialog): if 3.6's per-row
absmax/median tail ratios run >1.5× 3.5's, MQ4 quantization noise per expert
compounds across 8 active × 40 layers = 320 expert touchpoints and explains
why 3.6-A3B emits low-quality content even after Path B + sampler structural
fixes.

This script is *purely* CPU NumPy + safetensors. No GPU. No model load.
Runs in O(weights-on-disk) read + per-tensor compute.

Models compared are hard-coded inside the script: the HF repos
`Qwen/Qwen3.5-35B-A3B` and `Qwen/Qwen3.6-35B-A3B` (see `repos` list in
`main()`). Both must already be present in the HF cache (no download).
Requires `safetensors`, `numpy`, and `ml_dtypes` (the latter for bf16
support; falls back to a manual u16→f32 cast if missing).

Usage:
    python3 expert_absmax_stats.py \
        --hf-cache ~/.cache/huggingface/hub \
        --out absmax_results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from safetensors import safe_open

try:
    import ml_dtypes  # noqa: F401  enables np.dtype("bfloat16") cast paths
    HAS_ML_DTYPES = True
except ImportError:
    HAS_ML_DTYPES = False


def _read_bf16_raw(slice_handle, shape) -> np.ndarray:
    raw = slice_handle[:].tobytes() if hasattr(slice_handle[:], "tobytes") else bytes(slice_handle[:])
    if HAS_ML_DTYPES:
        arr = np.frombuffer(raw, dtype=ml_dtypes.bfloat16).reshape(shape)
        return arr.astype(np.float32)
    u16 = np.frombuffer(raw, dtype="<u2").astype(np.uint32)
    f32_bits = u16 << 16
    return f32_bits.view(np.float32).reshape(shape)


def _read_tensor(f, key) -> np.ndarray:
    slice_handle = f.get_slice(key)
    shape = tuple(slice_handle.get_shape())
    dtype = slice_handle.get_dtype()
    if dtype in ("BF16", "bfloat16"):
        return _read_bf16_raw(slice_handle, shape)
    arr = f.get_tensor(key)
    return np.asarray(arr, dtype=np.float32)


def find_snapshot(hf_cache: Path, repo: str) -> Path:
    repo_dir = hf_cache / f"models--{repo.replace('/', '--')}"
    snaps = list((repo_dir / "snapshots").iterdir())
    snaps = [s for s in snaps if any(s.iterdir())]
    if not snaps:
        sys.exit(f"no snapshots under {repo_dir}/snapshots — did the download finish?")
    snap = max(snaps, key=lambda p: sum((p / f).stat().st_size for f in p.iterdir() if (p / f).is_file()))
    return snap


def per_row_absmax_median(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    abs_arr = np.abs(arr)
    if abs_arr.ndim == 2:
        absmax = abs_arr.max(axis=1)
        median = np.median(abs_arr, axis=1)
    elif abs_arr.ndim == 3:
        absmax = abs_arr.max(axis=2).reshape(-1)
        median = np.median(abs_arr, axis=2).reshape(-1)
    else:
        raise ValueError(f"unexpected dim {abs_arr.ndim}")
    return absmax, median


def collect_expert_stats(snapshot: Path) -> dict:
    # Qwen3.5-A3B uses "model.safetensors-NNNNN-of-NNNNN.safetensors" while
    # Qwen3.6 uses "model-NNNNN-of-NNNNN.safetensors". Cover both.
    shards = sorted(snapshot.glob("model-*.safetensors")) + \
             sorted(snapshot.glob("model.safetensors-*.safetensors"))
    if not shards:
        sys.exit(f"no safetensors shards under {snapshot}")

    per_layer_per_expert: dict[tuple[int, str], dict] = {}

    for shard in shards:
        with safe_open(shard, framework="numpy") as f:
            for key in f.keys():
                # Match e.g.   model.layers.5.mlp.experts.42.gate_up_proj.weight
                #         OR   model.layers.5.mlp.experts.42.down_proj.weight
                #         OR   3D-packed:  model.layers.5.mlp.experts.gate_up_proj.weight
                if ".mlp.experts" not in key:
                    continue
                if not (key.endswith("gate_up_proj") or key.endswith("down_proj")
                        or key.endswith("gate_up_proj.weight") or key.endswith("down_proj.weight")):
                    continue
                parts = key.split(".")
                try:
                    layer_idx = int(parts[parts.index("layers") + 1])
                except (ValueError, IndexError):
                    continue
                if "gate_up_proj" in key:
                    proj = "gate_up_proj"
                elif "down_proj" in key:
                    proj = "down_proj"
                else:
                    continue
                arr = _read_tensor(f, key)
                absmax, median = per_row_absmax_median(arr)
                ratios = absmax / np.maximum(median, 1e-9)
                bucket = per_layer_per_expert.setdefault(
                    (layer_idx, proj), {"ratios": [], "absmax": [], "median": []}
                )
                bucket["ratios"].extend(ratios.tolist())
                bucket["absmax"].extend(absmax.tolist())
                bucket["median"].extend(median.tolist())

    out: dict = {}
    for (layer_idx, proj), bucket in sorted(per_layer_per_expert.items()):
        ratios = np.asarray(bucket["ratios"], dtype=np.float32)
        absmax = np.asarray(bucket["absmax"], dtype=np.float32)
        median = np.asarray(bucket["median"], dtype=np.float32)
        out[f"layer{layer_idx}.{proj}"] = {
            "n_rows": int(ratios.size),
            "ratio_mean": float(ratios.mean()),
            "ratio_median": float(np.median(ratios)),
            "ratio_p90": float(np.percentile(ratios, 90)),
            "ratio_p99": float(np.percentile(ratios, 99)),
            "ratio_max": float(ratios.max()),
            "absmax_mean": float(absmax.mean()),
            "absmax_p99": float(np.percentile(absmax, 99)),
            "median_mean": float(median.mean()),
        }
    return out


def summarize(per_layer: dict, label: str) -> dict:
    by_proj: dict[str, list[float]] = {"gate_up_proj": [], "down_proj": []}
    for k, stats in per_layer.items():
        for proj in by_proj:
            if k.endswith(f".{proj}"):
                by_proj[proj].append(stats["ratio_p99"])
    summary = {"label": label}
    for proj, p99s in by_proj.items():
        if not p99s:
            continue
        arr = np.asarray(p99s, dtype=np.float32)
        summary[proj] = {
            "n_layers": len(p99s),
            "p99_ratio_mean_across_layers": float(arr.mean()),
            "p99_ratio_max_across_layers": float(arr.max()),
            "p99_ratio_median_across_layers": float(np.median(arr)),
        }
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hf-cache", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    hf_cache = Path(args.hf_cache).expanduser()

    repos = [
        ("Qwen3.5-35B-A3B", "Qwen/Qwen3.5-35B-A3B"),
        ("Qwen3.6-35B-A3B", "Qwen/Qwen3.6-35B-A3B"),
    ]

    results = {"per_model": {}, "summary": {}}

    for label, repo in repos:
        snapshot = find_snapshot(hf_cache, repo)
        print(f"[{label}] reading {snapshot}", file=sys.stderr)
        per_layer = collect_expert_stats(snapshot)
        results["per_model"][label] = per_layer
        results["summary"][label] = summarize(per_layer, label)
        print(f"[{label}] summary {results['summary'][label]}", file=sys.stderr)

    if "Qwen3.5-35B-A3B" in results["summary"] and "Qwen3.6-35B-A3B" in results["summary"]:
        ratios = {}
        for proj in ("gate_up_proj", "down_proj"):
            s35 = results["summary"]["Qwen3.5-35B-A3B"].get(proj)
            s36 = results["summary"]["Qwen3.6-35B-A3B"].get(proj)
            if s35 and s36:
                ratios[proj] = {
                    "p99_mean_36_over_35": s36["p99_ratio_mean_across_layers"]
                                            / max(s35["p99_ratio_mean_across_layers"], 1e-9),
                    "p99_max_36_over_35": s36["p99_ratio_max_across_layers"]
                                            / max(s35["p99_ratio_max_across_layers"], 1e-9),
                }
        results["axis1_verdict"] = {
            "ratios": ratios,
            "criterion": "if any p99_*_36_over_35 > 1.5 -> axis 1 (per-expert quant noise) confirmed",
        }
        decided = any(v["p99_mean_36_over_35"] > 1.5 or v["p99_max_36_over_35"] > 1.5
                      for v in ratios.values())
        results["axis1_verdict"]["confirmed"] = decided

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
