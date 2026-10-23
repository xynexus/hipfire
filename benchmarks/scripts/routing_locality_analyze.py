#!/usr/bin/env python3
"""Routing locality from a `routing_trace_qwen4exp` trace.

A batch is B documents decoded in lockstep (step t = position t in every
stream). Per layer per step, the batch needs the UNION of its streams' routed
experts; microbatching reads each module once for all tokens routed to it.

    routing_locality_analyze.py <trace_dir> [--warmup 64] [--draws 24]

Reports, per group and for mixed-group batches:
  union   distinct experts per layer per step, and its ratio to the
          uniform-independent-routing expectation E*(1-(1-k/E)^B)
  LRU     per-layer LRU cache holding a fraction of E experts; steady-state
          misses per layer per step (after --warmup steps) and the per-routing
          miss rate misses/(B*k), i.e. what reaches the slow tier.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from routing_trace import load

ap = argparse.ArgumentParser()
ap.add_argument("trace")
ap.add_argument("--warmup", type=int, default=64)
ap.add_argument("--draws", type=int, default=24)
ap.add_argument("--batches", default="1,2,4,8,16")
ap.add_argument("--cache", default="0.10,0.22,0.33,0.50")
ap.add_argument("--belady", action="store_true", help="offline-optimal eviction instead of LRU")
a = ap.parse_args()

d = Path(a.trace)
tr = load(d)
L, K, E, T, groups = tr.layers, tr.k, tr.experts, tr.n_tok, tr.groups
r = tr.routed.astype(np.int64)
# One-hot membership [doc, t, layer, E] — a union is an OR over docs.
hot = np.zeros((len(groups), T, L, E), bool)
np.put_along_axis(hot, r.astype(np.int64), True, axis=3)
rng = np.random.default_rng(0)
Bs = [int(b) for b in a.batches.split(",")]
fracs = [float(f) for f in a.cache.split(",")]


def lru_misses(need, cap):
    """need: [T, L, E] tokens routed to each expert per step (0 = unused).
    Per-layer LRU of `cap` experts; returns misses per step per layer, [T, L].

    Modules stream through one at a time, so a step's union may exceed `cap`;
    after the step the cache is trimmed back to `cap`, keeping the most recent
    and, within this step, the most-used experts."""
    last = np.full((L, E), -1, np.int64)
    inc = np.zeros((L, E), bool)
    miss = np.zeros((T, L), np.int64)
    for t in range(T):
        n = need[t] > 0
        miss[t] = (n & ~inc).sum(1)
        inc |= n
        last[n] = t
        over = inc.sum(1) - cap
        for l in np.nonzero(over > 0)[0]:
            cand = np.nonzero(inc[l])[0]
            key = last[l, cand] * (1 << 20) + need[t, l, cand]
            inc[l, cand[np.argsort(key, kind="stable")[: over[l]]]] = False
    return miss


def belady_misses(need, cap):
    """As `lru_misses` but evicting the expert whose next use is farthest away
    (offline optimal) — bounds what any smarter eviction policy can buy."""
    T_ = need.shape[0]
    nxt = np.full((T_ + 1, L, E), T_ + 1, np.int64)  # next step >= t that uses e
    for t in range(T_ - 1, -1, -1):
        nxt[t] = np.where(need[t] > 0, t, nxt[t + 1])
    inc = np.zeros((L, E), bool)
    miss = np.zeros((T_, L), np.int64)
    for t in range(T_):
        n = need[t] > 0
        miss[t] = (n & ~inc).sum(1)
        inc |= n
        over = inc.sum(1) - cap
        for l in np.nonzero(over > 0)[0]:
            cand = np.nonzero(inc[l])[0]
            inc[l, cand[np.argsort(-nxt[t + 1, l, cand], kind="stable")[: over[l]]]] = False
    return miss


POLICY = belady_misses if a.belady else lru_misses


def evaluate(pool, B):
    """Mean union / miss stats over random B-subsets of `pool` doc indices."""
    if B > len(pool):
        return None
    draws = 1 if B == len(pool) else a.draws
    us, ms = [], {f: [] for f in fracs}
    for _ in range(draws):
        sel = rng.choice(pool, B, replace=False)
        need = hot[sel].sum(0, dtype=np.int32)  # [T, L, E] tokens per expert
        us.append((need > 0).sum(2)[a.warmup :].mean())
        for f in fracs:
            ms[f].append(POLICY(need, int(round(f * E)))[a.warmup :].mean())
    return np.mean(us), {f: np.mean(v) for f, v in ms.items()}


def uniform(B):
    return E * (1 - (1 - K / E) ** B)


names = list(dict.fromkeys(groups))
pools = {g: [i for i, x in enumerate(groups) if x == g] for g in names}
pools["MIXED"] = list(range(len(groups)))
print(f"trace: {len(groups)} docs x {T} tok, {L} layers, top-{K}/{E}; steady state t>={a.warmup}")
hdr = "group     B  union  /unif " + " ".join(f"miss@{f:.0%}  p@{f:.0%}" for f in fracs)
print(hdr)
out = []
for g, pool in pools.items():
    for B in Bs:
        res = evaluate(pool, B)
        if res is None:
            continue
        u, m = res
        row = {"group": g, "B": B, "union": u, "union_vs_uniform": u / uniform(B)}
        cells = []
        for f in fracs:
            row[f"miss_{f}"] = m[f]
            row[f"p_{f}"] = m[f] / (B * K)
            cells.append(f"{m[f]:8.2f} {m[f] / (B * K):6.3f}")
        out.append(row)
        print(f"{g:8s} {B:2d} {u:6.1f} {u / uniform(B):5.2f}  " + " ".join(cells))
# Static hot set (in-sample, so an optimistic bound): share of a group's
# routings that land in its own most-used `f*E` experts, per layer.
print("\nstatic hot-set coverage (in-sample upper bound)")
print("group    " + " ".join(f"top{f:.0%}" for f in fracs))
for g, pool in pools.items():
    cnt = hot[pool].sum((0, 1))  # [L, E]
    srt = -np.sort(-cnt, 1)
    cov = [(srt[:, : int(round(f * E))].sum() / cnt.sum()) for f in fracs]
    print(f"{g:8s} " + " ".join(f"{c:6.3f}" for c in cov))
    out.append({"group": g, "static_coverage": dict(zip(map(str, fracs), map(float, cov)))})
(d / "locality.json").write_text(json.dumps(out, indent=1))
