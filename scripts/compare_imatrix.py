#!/usr/bin/env python3
"""Per-tensor divergence reporter for imatrix files.

Compares two imatrix outputs (currently `.imatrix.gguf`; `.imq` support
to be added in Phase 2C) and reports per-tensor NRMSE, max/p99/p95/mean
relative error, cosine distance, and counts ratio.

Names normalize to llama.cpp GGUF style (`blk.{N}.<slot>.weight`) via
the mapping mirrored from hipfire-quantize's `safetensors_to_ggml_name`
(see crates/hipfire-quantize/src/main.rs:2503).

Statistics use `in_sum2 / counts` (mean-square per channel) so token-
count mismatches don't dominate.

Exit codes:
  0 — PASS  (max NRMSE ≤ --max-nrmse for all shared tensors, coverage ≥ 75%)
  1 — SOFT  (some tensors over threshold, median within band)
  2 — HARD  (median NRMSE over threshold OR coverage gap > 25%)

Usage:
    python3 scripts/compare_imatrix.py \
        --oracle benchmarks/quality-baselines/refs/qwen3.5-0.8b-bf16.imatrix.gguf \
        --target /tmp/qwen3.5-0.8b.tier1-v1.imatrix.gguf \
        --max-nrmse 0.05 \
        [--json out.json] [--markdown out.md]
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import numpy as np

try:
    import gguf
except ImportError:
    sys.stderr.write("error: requires `gguf` python package\n")
    sys.exit(3)


_PER_LAYER_MAP = {
    "mlp.gate_proj": "ffn_gate",
    "mlp.up_proj": "ffn_up",
    "mlp.down_proj": "ffn_down",
    "self_attn.q_proj": "attn_q",
    "self_attn.k_proj": "attn_k",
    "self_attn.v_proj": "attn_v",
    "self_attn.o_proj": "attn_output",
    "linear_attn.in_proj_qkv": "attn_qkv",
    "linear_attn.in_proj_z": "attn_gate",
    "linear_attn.in_proj_a": "ssm_alpha",
    "linear_attn.in_proj_b": "ssm_beta",
    "linear_attn.out_proj": "ssm_out",
}

_TOP_LEVEL_MAP = {
    "embed_tokens.weight": "token_embd.weight",
    "lm_head.weight": "output.weight",
    "norm.weight": "output_norm.weight",
}


def safetensors_to_ggml_name(name: str) -> Optional[str]:
    normalized = name
    for prefix in ("model.language_model.", "model."):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix):]
            break

    if normalized in _TOP_LEVEL_MAP:
        return _TOP_LEVEL_MAP[normalized]

    m = re.match(r"layers\.(\d+)\.(.+)\.weight$", normalized)
    if not m:
        return None
    layer_idx, slot = m.group(1), m.group(2)
    translated = _PER_LAYER_MAP.get(slot)
    if translated is None:
        return None
    return f"blk.{layer_idx}.{translated}.weight"


def canonicalize(raw_name: str) -> Optional[str]:
    for sfx in (".in_sum2", ".counts"):
        if raw_name.endswith(sfx):
            stem = raw_name[: -len(sfx)]
            break
    else:
        return None

    # GGUF style passes through (already canonical)
    if stem.startswith("blk.") or stem in _TOP_LEVEL_MAP.values():
        return stem if stem.endswith(".weight") else f"{stem}.weight"

    return safetensors_to_ggml_name(stem if stem.endswith(".weight") else stem + ".weight")


@dataclass
class ImatrixTensor:
    name: str
    in_sum2: np.ndarray
    counts: float


def load_imatrix(path: Path) -> Dict[str, ImatrixTensor]:
    if not path.exists():
        sys.stderr.write(f"error: file not found: {path}\n")
        sys.exit(3)
    reader = gguf.GGUFReader(str(path))
    pairs: Dict[str, Dict[str, np.ndarray]] = {}
    for t in reader.tensors:
        canon = canonicalize(t.name)
        if canon is None:
            continue
        suffix = "in_sum2" if t.name.endswith(".in_sum2") else "counts"
        pairs.setdefault(canon, {})[suffix] = np.asarray(t.data).astype(np.float32, copy=False)

    out: Dict[str, ImatrixTensor] = {}
    for canon, fields in pairs.items():
        in_sum2 = fields.get("in_sum2")
        counts = fields.get("counts")
        if in_sum2 is None:
            continue
        counts_scalar = float(counts.flat[0]) if counts is not None else float("nan")
        in_sum2_flat = in_sum2.reshape(-1).astype(np.float32, copy=False)
        out[canon] = ImatrixTensor(canon, in_sum2_flat, counts_scalar)
    return out


@dataclass
class TensorDivergence:
    name: str
    k: int
    counts_oracle: float
    counts_target: float
    counts_ratio: float
    nrmse: float
    rel_err_max: float
    rel_err_p99: float
    rel_err_p95: float
    rel_err_mean: float
    cos_dist: float
    notes: list = field(default_factory=list)


def compute_divergence(target: ImatrixTensor, oracle: ImatrixTensor) -> TensorDivergence:
    eps = 1e-12
    if oracle.counts > 0 and not math.isnan(oracle.counts):
        o = oracle.in_sum2 / oracle.counts
    else:
        o = oracle.in_sum2
    if target.counts > 0 and not math.isnan(target.counts):
        t = target.in_sum2 / target.counts
    else:
        t = target.in_sum2

    # Shape mismatch — fall back to safe metrics
    if t.shape != o.shape:
        return TensorDivergence(
            name=target.name, k=int(target.in_sum2.size),
            counts_oracle=oracle.counts, counts_target=target.counts,
            counts_ratio=float("nan"),
            nrmse=float("inf"), rel_err_max=float("inf"),
            rel_err_p99=float("inf"), rel_err_p95=float("inf"),
            rel_err_mean=float("inf"), cos_dist=float("inf"),
            notes=[f"shape mismatch: target={t.shape} oracle={o.shape}"],
        )

    diff = t - o
    nrmse = float(np.linalg.norm(diff) / (np.linalg.norm(o) + eps))
    rel = np.abs(diff) / (np.abs(o) + eps)
    cos = float(np.dot(t, o) / (np.linalg.norm(t) * np.linalg.norm(o) + eps))
    counts_ratio = float(target.counts / oracle.counts) if oracle.counts > 0 else float("nan")

    notes = []
    if counts_ratio < 0.1 or counts_ratio > 10:
        notes.append(f"counts ratio extreme ({counts_ratio:.3f})")

    return TensorDivergence(
        name=target.name, k=int(t.size),
        counts_oracle=oracle.counts, counts_target=target.counts,
        counts_ratio=counts_ratio,
        nrmse=nrmse,
        rel_err_max=float(np.max(rel)),
        rel_err_p99=float(np.percentile(rel, 99)),
        rel_err_p95=float(np.percentile(rel, 95)),
        rel_err_mean=float(np.mean(rel)),
        cos_dist=1.0 - cos,
        notes=notes,
    )


def classify(canon: str) -> str:
    suffix_map = {
        ".attn_q.weight": "attn_q",
        ".attn_k.weight": "attn_k",
        ".attn_v.weight": "attn_v",
        ".attn_output.weight": "attn_output (FullAttn o_proj)",
        ".attn_qkv.weight": "attn_qkv (DeltaNet in_proj_qkv)",
        ".attn_gate.weight": "attn_gate (DeltaNet in_proj_z)",
        ".ssm_alpha.weight": "ssm_alpha (DeltaNet in_proj_a)",
        ".ssm_beta.weight": "ssm_beta (DeltaNet in_proj_b)",
        ".ssm_out.weight": "ssm_out (DeltaNet out_proj)",
        ".ffn_gate.weight": "ffn_gate (MLP)",
        ".ffn_up.weight": "ffn_up (MLP)",
        ".ffn_down.weight": "ffn_down (MLP)",
    }
    for sfx, role in suffix_map.items():
        if canon.endswith(sfx):
            return role
    if canon in ("token_embd.weight", "output.weight", "output_norm.weight"):
        return canon
    return "other"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", type=Path, required=True)
    ap.add_argument("--target", type=Path, required=True)
    ap.add_argument("--max-nrmse", type=float, default=0.05)
    ap.add_argument("--json", type=Path)
    ap.add_argument("--markdown", type=Path)
    args = ap.parse_args()

    print(f"oracle: {args.oracle}")
    print(f"target: {args.target}")
    print()

    oracle = load_imatrix(args.oracle)
    target = load_imatrix(args.target)

    shared = sorted(set(oracle.keys()) & set(target.keys()))
    only_o = sorted(set(oracle.keys()) - set(target.keys()))
    only_t = sorted(set(target.keys()) - set(oracle.keys()))

    print(f"oracle tensors: {len(oracle)}  target: {len(target)}  shared: {len(shared)}")
    if only_o:
        print(f"  only-oracle ({len(only_o)}): {only_o[:3]}")
    if only_t:
        print(f"  only-target ({len(only_t)}): {only_t[:3]}")
    print()

    divs = [compute_divergence(target[n], oracle[n]) for n in shared]
    divs.sort(key=lambda d: d.nrmse, reverse=True)

    by_role: Dict[str, list] = {}
    for d in divs:
        by_role.setdefault(classify(d.name), []).append(d.nrmse)

    print("=" * 100)
    print("Top divergences (worst-first NRMSE)")
    print("=" * 100)
    print(f"{'tensor':<54} {'k':>6} {'NRMSE':>8} {'cos_d':>8} {'rel_max':>8} {'rel_p99':>8}")
    print("-" * 100)
    for d in divs[:30]:
        print(f"{d.name:<54} {d.k:>6} {d.nrmse:>8.4f} {d.cos_dist:>8.4f} "
              f"{d.rel_err_max:>8.4f} {d.rel_err_p99:>8.4f}")
    print()

    print("=" * 60)
    print("Per-role NRMSE summary")
    print("=" * 60)
    print(f"{'role':<40} {'n':>4} {'median':>9} {'max':>9}")
    print("-" * 60)
    for role in sorted(by_role.keys()):
        arr = np.array(by_role[role])
        print(f"{role:<40} {len(arr):>4} {np.median(arr):>9.4f} {np.max(arr):>9.4f}")
    print()

    all_n = np.array([d.nrmse for d in divs]) if divs else np.array([0.0])
    print(f"Overall: n={len(all_n)} median={np.median(all_n):.4f} "
          f"max={np.max(all_n):.4f} p95={np.percentile(all_n, 95):.4f}")

    if args.json:
        args.json.write_text(json.dumps({
            "oracle": str(args.oracle), "target": str(args.target),
            "max_nrmse_threshold": args.max_nrmse,
            "shared_count": len(shared),
            "only_oracle": only_o, "only_target": only_t,
            "overall": {
                "median_nrmse": float(np.median(all_n)),
                "max_nrmse": float(np.max(all_n)),
                "p95_nrmse": float(np.percentile(all_n, 95)),
            },
            "by_role": {
                role: {"n": len(np.array(v)),
                       "median_nrmse": float(np.median(v)),
                       "max_nrmse": float(np.max(v))}
                for role, v in by_role.items()
            },
            "per_tensor": [{
                "name": d.name, "k": d.k,
                "nrmse": d.nrmse, "cos_dist": d.cos_dist,
                "rel_err_max": d.rel_err_max, "rel_err_p99": d.rel_err_p99,
                "rel_err_p95": d.rel_err_p95, "rel_err_mean": d.rel_err_mean,
                "counts_oracle": d.counts_oracle, "counts_target": d.counts_target,
                "counts_ratio": d.counts_ratio, "notes": d.notes,
            } for d in divs],
        }, indent=2))
        print(f"JSON: {args.json}")

    if args.markdown:
        lines = [
            "# imatrix divergence report",
            "",
            f"- oracle: `{args.oracle}`",
            f"- target: `{args.target}`",
            f"- threshold: NRMSE ≤ {args.max_nrmse}",
            f"- shared: {len(shared)} / oracle={len(oracle)} target={len(target)}",
            "",
            "## Per-role NRMSE",
            "",
            "| role | n | median | max |",
            "|---|---:|---:|---:|",
        ]
        for role in sorted(by_role.keys()):
            arr = np.array(by_role[role])
            lines.append(f"| {role} | {len(arr)} | {np.median(arr):.4f} | {np.max(arr):.4f} |")
        lines.extend(["", "## Top 30 divergences", "",
                      "| tensor | k | NRMSE | cos_dist | rel_max | rel_p99 |",
                      "|---|---:|---:|---:|---:|---:|"])
        for d in divs[:30]:
            lines.append(f"| {d.name} | {d.k} | {d.nrmse:.4f} | {d.cos_dist:.4f} | "
                         f"{d.rel_err_max:.4f} | {d.rel_err_p99:.4f} |")
        args.markdown.write_text("\n".join(lines) + "\n")
        print(f"markdown: {args.markdown}")

    over = [d for d in divs if d.nrmse > args.max_nrmse]
    coverage = len(shared) / max(len(oracle), len(target), 1)
    median_n = float(np.median(all_n))

    print()
    if not over and coverage >= 0.75:
        print(f"PASS — {len(shared)} tensors, all NRMSE ≤ {args.max_nrmse}, coverage {coverage:.1%}")
        return 0
    elif median_n > args.max_nrmse or coverage < 0.75:
        print(f"HARD FAIL — median NRMSE {median_n:.4f} vs threshold {args.max_nrmse} "
              f"OR coverage {coverage:.1%} < 75%")
        return 2
    else:
        print(f"SOFT FAIL — {len(over)}/{len(shared)} tensors over threshold "
              f"(median {median_n:.4f} within band)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
