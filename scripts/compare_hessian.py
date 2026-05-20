#!/usr/bin/env python3
"""Per-tensor divergence reporter for HFHS Hessian sidecar files.

Compares two HFHS v1 files (one produced by the Rust Tier 1
`collect_hessian` binary, one produced by `scripts/collect_hessian.py`
on the PyTorch oracle path) and reports per-tensor NRMSE plus a
per-role summary, mirroring the format/exit-code conventions of
`scripts/compare_imatrix.py`.

The HFHS format is documented in `crates/hipfire-quantize/src/hfhs_writer.rs`
and `crates/hipfire-quantize/src/hessian_io.rs`. Both writers store the
per-tensor average `H = (1/N) · Σ x · xᵀ`, so the loaded Hessians are
directly comparable byte-for-byte at the format level — any divergence
comes from the BF16 forward path (Tier 1) vs the FP32-on-CPU eager
forward (oracle), not from format mismatches.

Exit codes:
  0 — PASS  (max NRMSE ≤ --max-nrmse for all shared tensors, coverage ≥ 75%)
  1 — SOFT  (some tensors over threshold, median within band)
  2 — HARD  (median NRMSE over threshold OR coverage gap > 25%)

Usage:
    python3 scripts/compare_hessian.py \\
        --oracle /workspace/qwen3.5-0.8b.pytorch.hessian.bin \\
        --target /workspace/qwen3.5-0.8b.tier1.hessian.bin \\
        --max-nrmse 0.01 \\
        [--json out.json] [--markdown out.md]
"""

from __future__ import annotations

import argparse
import json
import re
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


HFHS_MAGIC = b"HFHS"
HFHS_VERSION = 1
HEADER_SIZE = 24
DTYPE_F32 = 1
DTYPE_F64 = 2


@dataclass
class HessianTensor:
    """One per-tensor Hessian loaded from an HFHS file.

    `h` is the row-major K×K Hessian as a `(K*K,)` flat F32 NumPy array.
    """

    name: str
    expert_idx: int
    k: int
    h: np.ndarray


def load_hfhs(path: Path) -> Dict[str, HessianTensor]:
    """Load an HFHS v1 file. Returns a `name → HessianTensor` map.

    `expert_idx` is dropped from the key under the assumption that the
    Tier 1 / Tier 2 pipelines both emit dense (`expert_idx = 0`) entries
    for the current model set. If MoE Hessians ever land both sides,
    extend the key to `(name, expert_idx)` here.
    """
    if not path.exists():
        sys.stderr.write(f"error: file not found: {path}\n")
        sys.exit(3)
    raw = path.read_bytes()
    if len(raw) < HEADER_SIZE:
        sys.stderr.write(
            f"error: {path} is {len(raw)} bytes (need ≥ {HEADER_SIZE} for header)\n"
        )
        sys.exit(3)
    magic = raw[0:4]
    if magic != HFHS_MAGIC:
        sys.stderr.write(f"error: {path} has bad magic {magic!r}, expected {HFHS_MAGIC!r}\n")
        sys.exit(3)
    version = struct.unpack_from("<I", raw, 4)[0]
    if version != HFHS_VERSION:
        sys.stderr.write(
            f"error: {path} has version {version}, this script handles v{HFHS_VERSION}\n"
        )
        sys.exit(3)
    n_tensors = struct.unpack_from("<Q", raw, 8)[0]
    # reserved at [16..24)

    out: Dict[str, HessianTensor] = {}
    pos = HEADER_SIZE
    for _ in range(n_tensors):
        if pos + 4 > len(raw):
            sys.stderr.write(f"error: {path} truncated at name_len header\n")
            sys.exit(3)
        name_len = struct.unpack_from("<I", raw, pos)[0]
        pos += 4
        if pos + name_len + 12 > len(raw):
            sys.stderr.write(f"error: {path} truncated at tensor record name+meta\n")
            sys.exit(3)
        name = raw[pos:pos + name_len].decode("utf-8")
        pos += name_len
        expert_idx = struct.unpack_from("<I", raw, pos)[0]
        pos += 4
        k = struct.unpack_from("<I", raw, pos)[0]
        pos += 4
        dtype_flag = struct.unpack_from("<I", raw, pos)[0]
        pos += 4
        if dtype_flag == DTYPE_F32:
            elem_bytes = 4
            np_dtype = np.float32
        elif dtype_flag == DTYPE_F64:
            elem_bytes = 8
            np_dtype = np.float64
        else:
            sys.stderr.write(f"error: {path} tensor {name!r} has unknown dtype flag {dtype_flag}\n")
            sys.exit(3)
        payload_bytes = k * k * elem_bytes
        if pos + payload_bytes > len(raw):
            sys.stderr.write(
                f"error: {path} truncated at tensor {name!r} payload (need {payload_bytes}, "
                f"have {len(raw) - pos})\n"
            )
            sys.exit(3)
        h = np.frombuffer(raw, dtype=np_dtype, count=k * k, offset=pos).astype(
            np.float32, copy=False
        )
        pos += payload_bytes
        # Key by name only (assume dense, expert_idx=0); see docstring.
        # Tolerate either with-`.weight` or without-`.weight` keys by
        # stripping the suffix at index time. The format spec says the
        # canonical form is without `.weight` (hessian_io.rs:242-243),
        # but the comparator must keep working even if one side carries
        # the legacy suffix.
        key = name[:-len(".weight")] if name.endswith(".weight") else name
        out[key] = HessianTensor(name=key, expert_idx=expert_idx, k=k, h=h)
    return out


@dataclass
class TensorDivergence:
    name: str
    k: int
    nrmse: float
    rel_err_max: float
    rel_err_p99: float
    rel_err_p95: float
    rel_err_mean: float
    cos_dist: float
    diag_nrmse: float
    notes: list = field(default_factory=list)


def compute_divergence(target: HessianTensor, oracle: HessianTensor) -> TensorDivergence:
    eps = 1e-12
    t = target.h
    o = oracle.h
    if t.shape != o.shape:
        return TensorDivergence(
            name=target.name,
            k=int(target.k),
            nrmse=float("inf"),
            rel_err_max=float("inf"),
            rel_err_p99=float("inf"),
            rel_err_p95=float("inf"),
            rel_err_mean=float("inf"),
            cos_dist=float("inf"),
            diag_nrmse=float("inf"),
            notes=[f"shape mismatch: target K={target.k} oracle K={oracle.k}"],
        )
    diff = t - o
    nrmse = float(np.linalg.norm(diff) / (np.linalg.norm(o) + eps))
    rel = np.abs(diff) / (np.abs(o) + eps)
    cos = float(np.dot(t, o) / (np.linalg.norm(t) * np.linalg.norm(o) + eps))
    # Diagonal-only NRMSE — useful sanity check: even a noisy off-diagonal
    # pattern usually preserves the (1/N)·Σ x²  diagonal, which is what
    # the imatrix pipeline measured.
    k = target.k
    diag_idx = np.arange(k, dtype=np.int64) * (k + 1)
    t_diag = t[diag_idx]
    o_diag = o[diag_idx]
    diag_nrmse = float(np.linalg.norm(t_diag - o_diag) / (np.linalg.norm(o_diag) + eps))
    return TensorDivergence(
        name=target.name,
        k=int(k),
        nrmse=nrmse,
        rel_err_max=float(np.max(rel)),
        rel_err_p99=float(np.percentile(rel, 99)),
        rel_err_p95=float(np.percentile(rel, 95)),
        rel_err_mean=float(np.mean(rel)),
        cos_dist=1.0 - cos,
        diag_nrmse=diag_nrmse,
    )


# Mirror compare_imatrix.py's role classification.  HFHS names are the
# safetensors keys minus the trailing `.weight` (per the HFHS format
# spec — see hessian_io.rs:242-243), so the classifier matches stems
# like "model.layers.0.self_attn.q_proj".
_ROLE_PATTERNS: List[tuple] = [
    (re.compile(r"\.self_attn\.q_proj$"), "attn_q (FullAttn q_proj)"),
    (re.compile(r"\.self_attn\.k_proj$"), "attn_k (FullAttn k_proj)"),
    (re.compile(r"\.self_attn\.v_proj$"), "attn_v (FullAttn v_proj)"),
    (re.compile(r"\.self_attn\.o_proj$"), "attn_output (FullAttn o_proj)"),
    (re.compile(r"\.linear_attn\.in_proj_qkv$"), "attn_qkv (DeltaNet in_proj_qkv)"),
    (re.compile(r"\.linear_attn\.in_proj_z$"), "attn_gate (DeltaNet in_proj_z)"),
    (re.compile(r"\.linear_attn\.in_proj_a$"), "ssm_alpha (DeltaNet in_proj_a)"),
    (re.compile(r"\.linear_attn\.in_proj_b$"), "ssm_beta (DeltaNet in_proj_b)"),
    (re.compile(r"\.linear_attn\.out_proj$"), "ssm_out (DeltaNet out_proj)"),
    (re.compile(r"\.mlp\.gate_proj$"), "ffn_gate (MLP)"),
    (re.compile(r"\.mlp\.up_proj$"), "ffn_up (MLP)"),
    (re.compile(r"\.mlp\.down_proj$"), "ffn_down (MLP)"),
    (re.compile(r"\.mlp\.gate$"), "moe_gate (MoE router)"),
]


def _strip_weight(name: str) -> str:
    """HFHS canonical names drop the trailing `.weight`; tolerate either form
    in the comparator so an upgrade that crosses the rename boundary still
    finds shared tensors."""
    return name[:-len(".weight")] if name.endswith(".weight") else name


def classify(name: str) -> str:
    stem = _strip_weight(name)
    for pat, role in _ROLE_PATTERNS:
        if pat.search(stem):
            return role
    return "other"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", type=Path, required=True,
                    help="HFHS file from scripts/collect_hessian.py (PyTorch reference).")
    ap.add_argument("--target", type=Path, required=True,
                    help="HFHS file from the Rust collect_hessian binary (Tier 1).")
    ap.add_argument("--max-nrmse", type=float, default=0.01,
                    help="Per-tensor NRMSE threshold for PASS (default: 0.01).")
    ap.add_argument("--json", type=Path)
    ap.add_argument("--markdown", type=Path)
    args = ap.parse_args()

    print(f"oracle: {args.oracle}")
    print(f"target: {args.target}")
    print()

    oracle = load_hfhs(args.oracle)
    target = load_hfhs(args.target)

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
        by_role.setdefault(classify(d.name), []).append(d)

    print("=" * 110)
    print("Top divergences (worst-first NRMSE)")
    print("=" * 110)
    print(f"{'tensor':<60} {'k':>6} {'NRMSE':>8} {'diag_NRMSE':>11} {'cos_d':>8} {'rel_p99':>8}")
    print("-" * 110)
    for d in divs[:30]:
        print(
            f"{d.name:<60} {d.k:>6} {d.nrmse:>8.4f} {d.diag_nrmse:>11.4f} "
            f"{d.cos_dist:>8.4f} {d.rel_err_p99:>8.4f}"
        )
    print()

    print("=" * 75)
    print("Per-role NRMSE summary")
    print("=" * 75)
    print(f"{'role':<46} {'n':>4} {'median':>9} {'max':>9} {'diag_med':>9}")
    print("-" * 75)
    for role in sorted(by_role.keys()):
        arr_n = np.array([d.nrmse for d in by_role[role]])
        arr_diag = np.array([d.diag_nrmse for d in by_role[role]])
        print(
            f"{role:<46} {len(arr_n):>4} {np.median(arr_n):>9.4f} {np.max(arr_n):>9.4f} "
            f"{np.median(arr_diag):>9.4f}"
        )
    print()

    all_n = np.array([d.nrmse for d in divs]) if divs else np.array([0.0])
    print(
        f"Overall: n={len(all_n)} median={np.median(all_n):.4f} "
        f"max={np.max(all_n):.4f} p95={np.percentile(all_n, 95):.4f}"
    )

    if args.json:
        args.json.write_text(json.dumps({
            "oracle": str(args.oracle),
            "target": str(args.target),
            "max_nrmse_threshold": args.max_nrmse,
            "shared_count": len(shared),
            "only_oracle": only_o,
            "only_target": only_t,
            "overall": {
                "median_nrmse": float(np.median(all_n)),
                "max_nrmse": float(np.max(all_n)),
                "p95_nrmse": float(np.percentile(all_n, 95)),
            },
            "by_role": {
                role: {
                    "n": len(by_role[role]),
                    "median_nrmse": float(np.median([d.nrmse for d in by_role[role]])),
                    "max_nrmse": float(np.max([d.nrmse for d in by_role[role]])),
                    "median_diag_nrmse": float(
                        np.median([d.diag_nrmse for d in by_role[role]])
                    ),
                }
                for role in by_role
            },
            "per_tensor": [{
                "name": d.name, "k": d.k,
                "nrmse": d.nrmse, "diag_nrmse": d.diag_nrmse,
                "cos_dist": d.cos_dist,
                "rel_err_max": d.rel_err_max, "rel_err_p99": d.rel_err_p99,
                "rel_err_p95": d.rel_err_p95, "rel_err_mean": d.rel_err_mean,
                "notes": d.notes,
            } for d in divs],
        }, indent=2))
        print(f"JSON: {args.json}")

    if args.markdown:
        lines = [
            "# Hessian divergence report",
            "",
            f"- oracle: `{args.oracle}`",
            f"- target: `{args.target}`",
            f"- threshold: NRMSE ≤ {args.max_nrmse}",
            f"- shared: {len(shared)} / oracle={len(oracle)} target={len(target)}",
            "",
            "## Per-role NRMSE",
            "",
            "| role | n | median | max | diag_median |",
            "|---|---:|---:|---:|---:|",
        ]
        for role in sorted(by_role.keys()):
            arr_n = np.array([d.nrmse for d in by_role[role]])
            arr_diag = np.array([d.diag_nrmse for d in by_role[role]])
            lines.append(
                f"| {role} | {len(arr_n)} | {np.median(arr_n):.4f} | "
                f"{np.max(arr_n):.4f} | {np.median(arr_diag):.4f} |"
            )
        lines.extend([
            "",
            "## Top 30 divergences",
            "",
            "| tensor | k | NRMSE | diag_NRMSE | cos_dist | rel_p99 |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for d in divs[:30]:
            lines.append(
                f"| {d.name} | {d.k} | {d.nrmse:.4f} | {d.diag_nrmse:.4f} | "
                f"{d.cos_dist:.4f} | {d.rel_err_p99:.4f} |"
            )
        args.markdown.write_text("\n".join(lines) + "\n")
        print(f"markdown: {args.markdown}")

    over = [d for d in divs if d.nrmse > args.max_nrmse]
    coverage = len(shared) / max(len(oracle), len(target), 1)
    median_n = float(np.median(all_n))

    print()
    if not over and coverage >= 0.75:
        print(
            f"PASS — {len(shared)} tensors, all NRMSE ≤ {args.max_nrmse}, coverage {coverage:.1%}"
        )
        return 0
    elif median_n > args.max_nrmse or coverage < 0.75:
        print(
            f"HARD FAIL — median NRMSE {median_n:.4f} vs threshold {args.max_nrmse} "
            f"OR coverage {coverage:.1%} < 75%"
        )
        return 2
    else:
        print(
            f"SOFT FAIL — {len(over)}/{len(shared)} tensors over threshold "
            f"(median {median_n:.4f} within band)"
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
