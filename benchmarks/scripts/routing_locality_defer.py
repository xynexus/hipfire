#!/usr/bin/env python3
"""Deferred (pause-until-queued) expert loading over a routing trace.

Rounds sweep layers 0..L-1. A stream at layer l whose routed experts are all
resident advances to l+1 within the round; a stream needing a non-resident
module waits there. A missing module is loaded only once `--min-waiters`
streams are queued on it, or its oldest waiter has waited `--max-wait` rounds.
`--min-waiters 1` is plain lockstep. Per-layer LRU cache of a fraction of E,
trimmed back to capacity after each layer visit.

    routing_locality_defer.py <trace_dir> --group MIXED --B 112 --cache 0.22

Reports loads per routed pick (p_eff, what reaches the slow tier) and rounds
per completed token (latency multiplier vs lockstep = 1).
"""
import argparse

import numpy as np
from routing_trace import load

ap = argparse.ArgumentParser()
ap.add_argument("trace")
ap.add_argument("--group", default="MIXED")
ap.add_argument("--B", type=int, default=16)
ap.add_argument("--cache", type=float, default=0.22)
ap.add_argument("--min-waiters", type=int, default=1)
ap.add_argument("--max-wait", type=int, default=8)
ap.add_argument("--rounds", type=int, default=400)
ap.add_argument("--warmup", type=int, default=64)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

tr = load(a.trace)
L, K, E, T, groups = tr.layers, tr.k, tr.experts, tr.n_tok, tr.groups
r = tr.routed.astype(np.int64)
pool = [i for i, g in enumerate(groups) if a.group in ("MIXED", g)]
sel = np.random.default_rng(a.seed).choice(pool, a.B, replace=False)
R = r[sel]  # [B, T, L, K]
cap = int(round(a.cache * E))

pos_t = np.zeros(a.B, np.int64)
pos_l = np.zeros(a.B, np.int64)
waited = np.zeros(a.B, np.int64)  # rounds spent at the current layer
# Expert outputs accumulate (moe_accum_scaled), so a pick served once stays
# served; a token waits only for its still-unserved picks.
served = np.zeros((a.B, K), bool)
inc = np.zeros((L, E), bool)
last = np.full((L, E), -1, np.int64)
clock = 0
loads = picks = tokens = 0

for rnd in range(a.rounds):
    count = rnd >= a.warmup
    for l in range(L):
        clock += 1
        at = np.nonzero((pos_l == l) & (pos_t < T))[0]
        if at.size == 0:
            continue
        need = R[at, pos_t[at], l]  # [n, K]
        missing = ~served[at] & ~inc[l][need]
        # Queue depth and oldest wait per missing module.
        mods, inv = np.unique(need[missing], return_inverse=True)
        depth = np.bincount(inv, minlength=mods.size)
        rows = np.nonzero(missing)[0]
        oldest = np.zeros(mods.size, np.int64)
        np.maximum.at(oldest, inv, waited[at[rows]])
        go = mods[(depth >= a.min_waiters) | (oldest >= a.max_wait)]
        inc[l, go] = True
        if count:
            loads += go.size
        hit = ~served[at] & inc[l][need]  # served now, by a resident module
        last[l, np.unique(need[hit])] = clock
        served[at] |= hit
        ok = served[at].all(1)
        adv = at[ok]
        if count:
            picks += adv.size * K
        pos_l[adv] += 1
        served[adv] = False
        waited[adv] = 0
        waited[at[~ok]] += 1
        # Trim to capacity, oldest first.
        over = inc[l].sum() - cap
        if over > 0:
            c = np.nonzero(inc[l])[0]
            inc[l, c[np.argsort(last[l, c], kind="stable")[:over]]] = False
    done = pos_l == L
    if count:
        tokens += done.sum()
    pos_t[done] += 1
    pos_l[done] = 0

# Throughput in token-equivalents (layer passes / L): deferred streams may
# finish few whole tokens inside the window while still making progress.
span = a.rounds - a.warmup
tok_eq = picks / (K * L)
print(
    f"{a.group} B={a.B} cache={a.cache:.0%} min_waiters={a.min_waiters} max_wait={a.max_wait}: "
    f"p_eff={loads / max(picks, 1):.3f} loads/round={loads / span:.0f} "
    f"tok/round={tok_eq / span:.2f} rounds/token/stream={span * a.B / max(tok_eq, 1e-9):.1f}"
)
