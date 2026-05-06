"""Analyze D3 per-expert activation absmax results.

Reports for each A3B model:
  - Layer-by-layer max(absmax_max) (where's the activation cliff?)
  - Top-N (layer, expert) by absmax_max output side
  - Cross-check vs the D2-derived 17 SE candidates at layer 0
  - Cross-model concordance between 3.5-A3B and 3.6-A3B top D3 sets

Run from repo root:
  python3 scripts/quant-survey/analyze_d3.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

D3_BASE = Path("/tmp/hiptrx-survey-pull/runs")
A3B_MODELS = ["qwen3.5-a3b", "qwen3.6-a3b"]

# D2-derived SE candidates from 02-survey-results.md (layer 0 down_proj
# weight ratio_p99 top-20 union).
D2_SE = {3, 8, 42, 49, 70, 115, 119, 132, 164, 167,
         190, 195, 203, 225, 237, 239, 253}


def load_per_expert(model: str) -> list[dict]:
    path = D3_BASE / f"{model}-d3-full" / "per_expert.jsonl"
    records: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            # Drop the absmax array to keep memory low; we only need
            # absmax_max for ranking.
            r.pop("absmax", None)
            records.append(r)
    return records


def layer_max_absmax(records: list[dict], side: str) -> dict[int, float]:
    by_layer: dict[int, float] = defaultdict(float)
    for r in records:
        if r["side"] != side:
            continue
        v = r["absmax_max"]
        if v > by_layer[r["layer_idx"]]:
            by_layer[r["layer_idx"]] = v
    return dict(by_layer)


def top_per_expert(records: list[dict], side: str, top_n: int = 20) -> list[dict]:
    rs = [r for r in records if r["side"] == side]
    rs.sort(key=lambda d: -d["absmax_max"])
    return rs[:top_n]


def main() -> int:
    by_model: dict[str, list[dict]] = {}
    for m in A3B_MODELS:
        by_model[m] = load_per_expert(m)
        print(f"loaded {len(by_model[m])} records for {m}")

    print()
    print("=" * 78)
    print("Layer-by-layer max(absmax_max) on output side")
    print("=" * 78)
    print(f"{'layer':>5} | {'3.5-A3B':>10} | {'3.6-A3B':>10}")
    by_layer_a = layer_max_absmax(by_model["qwen3.5-a3b"], "output")
    by_layer_b = layer_max_absmax(by_model["qwen3.6-a3b"], "output")
    for L in sorted(set(by_layer_a) | set(by_layer_b)):
        a = by_layer_a.get(L, 0)
        b = by_layer_b.get(L, 0)
        print(f"{L:>5} | {a:>10.3f} | {b:>10.3f}")

    print()
    print("=" * 78)
    print("Top-20 per-expert by absmax_max (output side)")
    print("=" * 78)
    for m in A3B_MODELS:
        print(f"\n--- {m} ---")
        print(f"{'rank':>4} | {'layer':>5} | {'expert':>6} | {'absmax_max':>10} | {'n_tokens_routed':>15}")
        for i, r in enumerate(top_per_expert(by_model[m], "output", 20), 1):
            print(f"{i:>4} | {r['layer_idx']:>5} | {r['expert_idx']:>6} | {r['absmax_max']:>10.3f} | {r['n_tokens_routed']:>15}")

    print()
    print("=" * 78)
    print("D2 layer-0 SE candidates: their D3 absmax_max (output side)")
    print("=" * 78)
    print(f"{'expert':>6} | {'3.5-A3B abs':>12} | {'3.5-A3B tokens':>14} | {'3.6-A3B abs':>12} | {'3.6-A3B tokens':>14}")
    by_l0_a = {r["expert_idx"]: r for r in by_model["qwen3.5-a3b"] if r["layer_idx"] == 0 and r["side"] == "output"}
    by_l0_b = {r["expert_idx"]: r for r in by_model["qwen3.6-a3b"] if r["layer_idx"] == 0 and r["side"] == "output"}
    for ex in sorted(D2_SE):
        a = by_l0_a.get(ex, None)
        b = by_l0_b.get(ex, None)
        a_abs = f"{a['absmax_max']:.3f}" if a else "n/a"
        a_tok = f"{a['n_tokens_routed']}" if a else "n/a"
        b_abs = f"{b['absmax_max']:.3f}" if b else "n/a"
        b_tok = f"{b['n_tokens_routed']}" if b else "n/a"
        print(f"{ex:>6} | {a_abs:>12} | {a_tok:>14} | {b_abs:>12} | {b_tok:>14}")

    print()
    print("=" * 78)
    print("Cross-model D3 SE concordance (top-20 union)")
    print("=" * 78)
    a_top = top_per_expert(by_model["qwen3.5-a3b"], "output", 20)
    b_top = top_per_expert(by_model["qwen3.6-a3b"], "output", 20)
    a_keys = {(r["layer_idx"], r["expert_idx"]) for r in a_top}
    b_keys = {(r["layer_idx"], r["expert_idx"]) for r in b_top}
    common = a_keys & b_keys
    only_a = a_keys - b_keys
    only_b = b_keys - a_keys
    print(f"3.5-A3B top-20 keys: {sorted(a_keys)}")
    print(f"3.6-A3B top-20 keys: {sorted(b_keys)}")
    print(f"common: {len(common)} / 20 - {sorted(common)}")
    print(f"only in 3.5: {sorted(only_a)}")
    print(f"only in 3.6: {sorted(only_b)}")

    # Layer 0 max output absmax across ALL experts (including non-D2 candidates).
    print()
    print("=" * 78)
    print("Layer 0 max absmax_max (output side) — any expert")
    print("=" * 78)
    for m in A3B_MODELS:
        l0 = [r for r in by_model[m] if r["layer_idx"] == 0 and r["side"] == "output"]
        l0.sort(key=lambda d: -d["absmax_max"])
        print(f"\n{m} top-10 layer 0:")
        print(f"{'rank':>4} | {'expert':>6} | {'absmax_max':>10} | {'tokens':>7} | {'in_D2_SE':>9}")
        for i, r in enumerate(l0[:10], 1):
            mark = "*" if r["expert_idx"] in D2_SE else ""
            print(f"{i:>4} | {r['expert_idx']:>6} | {r['absmax_max']:>10.3f} | {r['n_tokens_routed']:>7} | {mark:>9}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
