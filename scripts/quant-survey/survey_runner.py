"""
survey_runner.py — Phase 1A Phase 1 main entry.

Wires safetensors_reader (per-layer streaming) + quant_ops (production-matched
MQ4G256+FWHT) + diagnostics (D1 NRMSE / D2 down_proj absmax+ratio /
D4 FWHT pre-post) into a single CLI runnable per-model.

D3 (per-channel activation absmax via forward-pass hooks) is a separate
module deferred to a later iteration; it requires transformers + torch
ROCm wheel installed (task #35 prerequisite).

Output layout per --output-dir:
  per_tensor.jsonl    one JSON record per surveyed tensor (D1/D2/D4 stats)
  summary.json        per-model summary: model, n_layers, projection
                      counts, top-N outliers by ratio_p99 z-score, etc.

Resident memory: bounded by stream_layer_tensors's two-pass design;
peak is one layer's worth of fp32 tensors. For 9B that's ~1 GB.

Usage:

  python3 scripts/quant-survey/survey_runner.py \
      --model qwen3.5-9b \
      --hf-cache ~/.cache/huggingface/hub \
      --output-dir /tmp/hiptrx-survey/runs/qwen3.5-9b/ \
      --diagnostics d1,d2,d4 \
      --projections q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,gate_up_proj,router,shared_expert.gate_proj,shared_expert.up_proj,shared_expert.down_proj,shared_expert_gate

The --gpu flag is reserved for the future D3 module; weight-side D1/D2/D4
are CPU-only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS))
sys.path.insert(0, str(_THIS / "diagnostics"))

from quant_ops import (  # noqa: E402
    PRODUCTION_SIGNS1_SEED,
    PRODUCTION_SIGNS2_SEED,
    gen_fwht_signs,
)
from safetensors_reader import (  # noqa: E402
    find_hf_snapshot,
    list_safetensors_shards,
    read_config,
    stream_layer_tensors,
    TensorBatch,
)
from diagnostics.d1_nrmse import run_d1  # noqa: E402
from diagnostics.d2_down_proj_max import run_d2  # noqa: E402
from diagnostics.d4_fwht import run_d4  # noqa: E402


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY: dict[str, str] = {
    "qwen3.5-9b": "Qwen/Qwen3.5-9B",
    "qwen3.5-27b": "Qwen/Qwen3.5-27B",
    "qwen3.5-a3b": "Qwen/Qwen3.5-35B-A3B",
    "qwen3.6-a3b": "Qwen/Qwen3.6-35B-A3B",
}

# Default projections to survey. The synthesis stage filters further;
# this is just the runner's whitelist for "interesting weight tensors."
DEFAULT_PROJECTIONS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj", "gate_up_proj",
    "router",
    "shared_expert.gate_proj", "shared_expert.up_proj", "shared_expert.down_proj",
    "shared_expert_gate",
]


# ---------------------------------------------------------------------------
# Per-tensor diagnostic invocation
# ---------------------------------------------------------------------------

def _run_diagnostics_on_tensor(
    batch: TensorBatch,
    diagnostics: set[str],
    signs1: np.ndarray,
    signs2: np.ndarray,
    seen_keys: set[tuple[str, int]] | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """Apply requested diagnostics to one TensorBatch. Handles both 2D
    dense projections and 3D-stacked MoE expert tensors (which are
    expanded into per-expert 2D records before D2 runs).

    If `seen_keys` is provided, skips experts whose (name, expert_idx)
    is already in the set — the expensive MQ4 round-trip is NOT executed
    for those. This is the resume path for kill-restart cycles.

    Returns (records_emitted, n_skipped).
    """
    records = []
    n_skipped = 0
    name = batch.ref.name
    layer_idx = batch.ref.layer_idx
    projection = batch.ref.projection
    is_stacked = batch.ref.is_stacked_3d
    data = batch.data

    if is_stacked:
        # Shape [n_experts, M, K]; emit one record per expert.
        n_experts = data.shape[0]
        for e in range(n_experts):
            if seen_keys is not None and (name, e) in seen_keys:
                n_skipped += 1
                continue
            sub = data[e]  # 2D [M, K]
            rec = _run_one(name, layer_idx, e, projection, True, sub, diagnostics, signs1, signs2)
            records.append(rec)
    else:
        # Dense or per-expert 2D, OR 1D (norm/bias). Most diagnostics
        # require 2D for D2; the runner handles dimension-aware fallbacks.
        if seen_keys is not None and (name, batch.ref.expert_idx) in seen_keys:
            return records, 1
        rec = _run_one(name, layer_idx, batch.ref.expert_idx, projection, False, data, diagnostics, signs1, signs2)
        records.append(rec)
    return records, n_skipped


def _run_one(
    name: str,
    layer_idx: int,
    expert_idx: int,
    projection: str,
    is_stacked_origin: bool,
    arr: np.ndarray,
    diagnostics: set[str],
    signs1: np.ndarray,
    signs2: np.ndarray,
) -> dict[str, Any]:
    """Run all requested diagnostics on a single tensor (1D or 2D)."""
    out: dict[str, Any] = {
        "name": name,
        "layer_idx": layer_idx,
        "expert_idx": expert_idx,
        "projection": projection,
        "is_stacked_origin": is_stacked_origin,
        "shape": list(arr.shape),
        "n_elements": int(arr.size),
    }
    t0 = time.time()

    if "d1" in diagnostics:
        try:
            r = run_d1(arr, signs1, signs2)
            out["d1"] = r.to_json()
        except Exception as e:
            out["d1"] = {"error": f"{type(e).__name__}: {e}"}

    if "d2" in diagnostics:
        if arr.ndim == 2:
            try:
                r = run_d2(arr)
                out["d2"] = r.to_json()
            except Exception as e:
                out["d2"] = {"error": f"{type(e).__name__}: {e}"}
        else:
            out["d2"] = {"skipped": f"D2 expects 2D; tensor is {arr.ndim}D"}

    if "d4" in diagnostics:
        try:
            r = run_d4(arr, signs1, signs2)
            out["d4"] = r.to_json()
        except Exception as e:
            out["d4"] = {"error": f"{type(e).__name__}: {e}"}

    out["wall_time_s"] = round(time.time() - t0, 3)
    return out


# ---------------------------------------------------------------------------
# Outlier classification (model-level)
# ---------------------------------------------------------------------------

def _classify_outliers(
    records: list[dict[str, Any]],
    z_threshold: float = 3.0,
) -> list[dict[str, Any]]:
    """Identify outlier tensors by z-score on D2 ratio_p99 (per
    projection). Outliers in synthesis are characterized by either
    weight-side ratio anomaly or NRMSE anomaly; this runner only
    classifies on D2 ratio_p99 (the AMENDED CONTRACT criterion).
    NRMSE-based classification is left to the synthesis (02-doc).

    Returns list of {layer, expert, projection, criterion, ratio_p99,
    z_score} dicts, sorted by z_score desc.
    """
    by_proj: dict[str, list[tuple[dict, float]]] = {}
    for r in records:
        d2 = r.get("d2")
        if not d2 or "skipped" in d2 or "error" in d2:
            continue
        rp99 = d2["ratio"]["p99"]
        if not np.isfinite(rp99):
            continue
        by_proj.setdefault(r["projection"], []).append((r, rp99))

    outliers = []
    for proj, items in by_proj.items():
        if len(items) < 8:
            continue  # too few to compute meaningful z-score
        vals = np.array([p99 for _, p99 in items], dtype=np.float64)
        median = float(np.median(vals))
        mad = float(np.median(np.abs(vals - median)))
        if mad <= 0:
            continue
        # Robust z-score using MAD (resilient to extreme outliers
        # that would inflate stddev).
        scale = 1.4826 * mad
        for r, rp99 in items:
            z = (rp99 - median) / scale
            if z >= z_threshold:
                outliers.append({
                    "name": r["name"],
                    "layer_idx": r["layer_idx"],
                    "expert_idx": r["expert_idx"],
                    "projection": proj,
                    "criterion": f"d2.ratio.p99 z>={z_threshold:.1f} (MAD-based)",
                    "ratio_p99": float(rp99),
                    "absmax_max": float(r["d2"]["row_max"]["max"]),
                    "z_score": float(z),
                })
    outliers.sort(key=lambda d: -d["z_score"])
    return outliers


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODEL_REGISTRY.keys()))
    ap.add_argument("--hf-cache", default=str(Path.home() / ".cache/huggingface/hub"))
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--diagnostics", default="d1,d2,d4",
                    help="comma-separated subset of d1,d2,d4 (D3 deferred)")
    ap.add_argument("--projections", default=",".join(DEFAULT_PROJECTIONS))
    ap.add_argument("--gpu", type=int, default=-1, help="reserved for D3; ignored for weight-side")
    ap.add_argument("--max-layers", type=int, default=-1,
                    help="cap on number of layers to survey; -1 = all")
    args = ap.parse_args()

    diagnostics = {d.strip() for d in args.diagnostics.split(",") if d.strip()}
    valid = {"d1", "d2", "d4"}
    if not diagnostics.issubset(valid):
        print(f"ERROR: unknown diagnostic in {diagnostics}; valid: {valid}", file=sys.stderr)
        return 1

    projections = [p.strip() for p in args.projections.split(",") if p.strip()]
    repo = MODEL_REGISTRY[args.model]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"survey_runner: model={args.model} repo={repo}", file=sys.stderr)
    print(f"survey_runner: hf_cache={args.hf_cache}", file=sys.stderr)
    print(f"survey_runner: output_dir={output_dir}", file=sys.stderr)
    print(f"survey_runner: diagnostics={sorted(diagnostics)}", file=sys.stderr)

    snapshot = find_hf_snapshot(Path(args.hf_cache).expanduser(), repo)
    print(f"survey_runner: snapshot={snapshot}", file=sys.stderr)

    config = read_config(snapshot)
    print(f"survey_runner: arch={config.get('architectures', ['?'])} "
          f"hidden_size={config.get('hidden_size')} "
          f"num_hidden_layers={config.get('num_hidden_layers')} "
          f"num_experts={config.get('num_experts', 'N/A')}",
          file=sys.stderr)

    signs1 = gen_fwht_signs(PRODUCTION_SIGNS1_SEED)
    signs2 = gen_fwht_signs(PRODUCTION_SIGNS2_SEED)

    per_tensor_path = output_dir / "per_tensor.jsonl"
    summary_path = output_dir / "summary.json"

    # Resume support: read existing JSONL records (if any) into all_records
    # and a "seen" set keyed by tensor (name, expert_idx). Process skips
    # any tensor already in the set. Append-only file open for new records.
    all_records: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, int]] = set()
    if per_tensor_path.exists():
        with open(per_tensor_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue  # skip malformed last line if previous run was killed mid-write
                all_records.append(rec)
                seen_keys.add((rec["name"], rec.get("expert_idx", -1)))
        if seen_keys:
            print(f"survey_runner: resuming with {len(seen_keys)} existing records",
                  file=sys.stderr)

    n_layers_seen = 0
    t_start = time.time()

    with open(per_tensor_path, "a") as out_jsonl:
        for layer_idx, batches in stream_layer_tensors(snapshot, projections=projections):
            if args.max_layers >= 0 and n_layers_seen >= args.max_layers:
                break
            n_layers_seen += 1
            t_layer = time.time()
            n_records_layer = 0
            n_skipped_layer = 0
            for batch in batches:
                records, n_skip = _run_diagnostics_on_tensor(
                    batch, diagnostics, signs1, signs2, seen_keys=seen_keys)
                n_skipped_layer += n_skip
                for r in records:
                    key = (r["name"], r.get("expert_idx", -1))
                    out_jsonl.write(json.dumps(r) + "\n")
                    out_jsonl.flush()
                    all_records.append(r)
                    seen_keys.add(key)
                    n_records_layer += 1
                # Drop data ref so GC can reclaim before next batch.
                batch.data = None
            skip_msg = f" skipped={n_skipped_layer}" if n_skipped_layer else ""
            print(f"  layer={layer_idx:3d} batches={len(batches)} "
                  f"records={n_records_layer}{skip_msg} dt={time.time() - t_layer:.1f}s",
                  file=sys.stderr)

    elapsed = time.time() - t_start
    print(f"survey_runner: surveyed {n_layers_seen} layers, "
          f"{len(all_records)} tensor records in {elapsed:.1f}s",
          file=sys.stderr)

    # Build summary
    proj_counts: dict[str, int] = {}
    layers_seen: set[int] = set()
    for r in all_records:
        proj_counts[r["projection"]] = proj_counts.get(r["projection"], 0) + 1
        layers_seen.add(r["layer_idx"])
    outliers = _classify_outliers(all_records, z_threshold=3.0)

    summary = {
        "model": args.model,
        "repo": repo,
        "snapshot": str(snapshot),
        "config": {
            "architectures": config.get("architectures"),
            "hidden_size": config.get("hidden_size"),
            "num_hidden_layers": config.get("num_hidden_layers"),
            "num_experts": config.get("num_experts"),
            "num_experts_per_tok": config.get("num_experts_per_tok"),
            "moe_intermediate_size": config.get("moe_intermediate_size"),
        },
        "diagnostics_run": sorted(diagnostics),
        "fwht_seeds": {"signs1": PRODUCTION_SIGNS1_SEED, "signs2": PRODUCTION_SIGNS2_SEED},
        "n_layers_surveyed": n_layers_seen,
        "n_tensor_records": len(all_records),
        "n_layers_observed": len(layers_seen),
        "projection_counts": proj_counts,
        "wall_time_seconds": round(elapsed, 1),
        "outlier_count_z3": len(outliers),
        "outliers_top20_by_ratio_p99_z": outliers[:20],
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"survey_runner: wrote {per_tensor_path}", file=sys.stderr)
    print(f"survey_runner: wrote {summary_path}", file=sys.stderr)
    print(f"survey_runner: outliers (z>=3, MAD-based): {len(outliers)} found", file=sys.stderr)
    if outliers[:5]:
        for o in outliers[:5]:
            print(f"  outlier: layer={o['layer_idx']} expert={o['expert_idx']} "
                  f"proj={o['projection']} ratio_p99={o['ratio_p99']:.2e} "
                  f"z={o['z_score']:.2f}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
