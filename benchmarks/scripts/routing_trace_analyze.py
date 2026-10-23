#!/usr/bin/env python3
"""Structure of a routing trace (format: routing_trace.py): what an expert
scheduler could exploit beyond LRU.

    routing_trace_analyze.py <trace_dir> [--cache 0.22]

Even-indexed streams fit any statistic (popularity, co-occurrence); odd ones
are scored. Values are averaged over layer bands.
  1. weight spread inside the routed top-k
  2. temporal reuse: overlap of S(t) with S(t-d), same stream and layer
  3. token-id determinism: overlap between occurrences of the same token
  4. prefetch: recall@m of S(t, l) predicted before layer l's router runs
  5. substitution: static hot set; can a missed pick be replaced by a
     resident expert from the top-N alternatives, and at what probability cost
  6. subject specificity of hot sets
  7. per-stream working set over a window of W tokens
  8. one-layer-ahead prefetch into a per-stream LRU cache: which misses it
     takes off the critical path, and the extra loads it costs
"""
import argparse

import numpy as np
from routing_trace import load

ap = argparse.ArgumentParser()
ap.add_argument("trace")
ap.add_argument("--cache", type=float, default=0.22)
a = ap.parse_args()

tr = load(a.trace)
R = tr.routed.astype(np.int64)  # [S, T, L, K]
S, T, L, K = R.shape
E = tr.experts
G = np.array(tr.groups)
train, test = np.arange(S) % 2 == 0, np.arange(S) % 2 == 1
cap = int(round(a.cache * E))
bands = [range(i, min(i + L // 4, L)) for i in range(0, L, L // 4)]
bname = [f"L{b[0]}-{b[-1]}" for b in bands]


def onehot(x):  # [..., K] -> [..., E] bool
    h = np.zeros(x.shape[:-1] + (E,), bool)
    np.put_along_axis(h, x, True, -1)
    return h


H = [onehot(R[:, :, l]) for l in range(L)]  # per layer [S, T, E]


def ov(x, y):  # mean |x & y| / K
    return (x & y).sum(-1).mean() / K


def by_band(f):  # f(layer) -> value; mean per band
    v = np.array([f(l) for l in range(L)])
    return [v[list(b)].mean() for b in bands]


def row(name, vals, fmt="{:7.3f}"):
    print(f"  {name:<30}" + "".join(fmt.format(v) for v in vals))


def head(title):
    print(f"\n{title}\n  {'':<30}" + "".join(f"{n:>7}" for n in bname))


print(f"trace: S={S} T={T} L={L} K={K} E={E}; uniform overlap K/E={K / E:.3f}")

# 1. Weight spread.
if tr.alt_p is not None:
    w = tr.alt_p[..., :K].astype(np.float32)
    w /= w.sum(-1, keepdims=True)
    mw = w.reshape(-1, K).mean(0)
    print("\n1. routed weight by rank (renormalised, descending prob)")
    print("  " + " ".join(f"{x:.3f}" for x in mw))
    print("  cumulative " + " ".join(f"{x:.2f}" for x in np.cumsum(mw)))

# 2. Temporal reuse.
ug = list(dict.fromkeys(G))
members = [np.nonzero(G == g)[0] for g in ug]
partner = np.empty(S, np.int64)  # next stream of the same group, cyclically
other = np.empty(S, np.int64)  # same slot in the next group
for gi, m in enumerate(members):
    partner[m] = np.roll(m, -1)
    nxt = members[(gi + 1) % len(ug)]
    other[m] = nxt[np.arange(m.size) % nxt.size]
head("2. reuse: overlap of S(t) with ...")
for d in (1, 2, 4, 16, 64, 256):
    row(f"same stream, t-{d}", by_band(lambda l: ov(H[l][:, d:], H[l][:, :-d])))
row("other stream, same group, same t", by_band(lambda l: ov(H[l], H[l][partner])))
row("other stream, other group, same t", by_band(lambda l: ov(H[l], H[l][other])))

# 3. Token-id determinism: each occurrence paired with the next one of the same id.
if tr.tokens is not None:
    tok = tr.tokens.reshape(-1)
    o = np.argsort(tok, kind="stable")
    same = tok[o[1:]] == tok[o[:-1]]
    i, j = o[:-1][same], o[1:][same]
    cross = i // T != j // T
    rng = np.random.default_rng(0)
    ri, rj = rng.integers(0, S * T, i.size), rng.integers(0, S * T, i.size)
    head(f"3. same token id ({same.sum()} pairs, {cross.sum()} cross-doc)")
    flat = [h.reshape(S * T, E) for h in H]
    row("same id, same doc", by_band(lambda l: ov(flat[l][i[~cross]], flat[l][j[~cross]])))
    row("same id, other doc", by_band(lambda l: ov(flat[l][i[cross]], flat[l][j[cross]])))
    row("random pair", by_band(lambda l: ov(flat[l][ri], flat[l][rj])))

# 4. Prefetch: predict S(t, l) for test tokens before layer l's router.
pop = [H[l][train].reshape(-1, E).sum(0).astype(np.float32) for l in range(L)]


def recall(score, truth, m):
    top = np.argpartition(-score, m - 1, -1)[..., :m]
    return np.take_along_axis(truth, top, -1).sum(-1).mean() / K


ms = (10, 20, 40, 80)
rec = {}
Cs = [None]
for l in range(1, L):
    truth = H[l][test][:, 8:].reshape(-1, E)
    prev = sum(0.5**d * H[l][test][:, 8 - d : T - d] for d in range(1, 9)).reshape(-1, E)
    X = H[l - 1][train].reshape(-1, E).astype(np.float32)
    C = X.T @ H[l][train].reshape(-1, E).astype(np.float32)
    C /= np.maximum(X.sum(0), 1)[:, None]  # P(e at l | e' at l-1)
    Cs.append(C)
    cooc = H[l - 1][test][:, 8:].reshape(-1, E).astype(np.float32) @ C
    tie = pop[l] / pop[l].sum() * 1e-3
    preds = {
        "popularity (static)": np.broadcast_to(pop[l], truth.shape),
        "previous tokens, same layer": prev + tie,
        "previous layer, co-occurrence": cooc + tie,
        "prev tokens + prev layer": prev + cooc / K + tie,
    }
    for name, sc in preds.items():
        for m in ms:
            rec.setdefault((name, m), [np.nan] * L)[l] = recall(sc, truth, m)
for m in ms:
    head(f"4. prefetch recall@{m} (fraction of S(t,l) in the {m} predicted)")
    for name in preds:
        v = np.array(rec[(name, m)])
        row(name, [np.nanmean(v[list(b)]) for b in bands])

# 5. Substitution against a static per-layer hot set of `cap` experts.
if tr.alt_ids is not None:
    head(f"5. static hot set {a.cache:.0%} ({cap}/{E}) from train popularity; test picks")
    hot = np.zeros((L, E), bool)
    for l in range(L):
        hot[l, np.argsort(-pop[l])[:cap]] = True
    alt = tr.alt_ids[test].astype(np.int64)  # [St, T, L, N]
    p = tr.alt_p[test].astype(np.float32)
    res = hot[np.arange(L)[None, None, :, None], alt]  # resident?
    miss = ~res[..., :K]
    nmiss = miss.sum(-1)
    w = p[..., :K] / p[..., :K].sum(-1, keepdims=True)
    # Substitutes: resident non-routed alternatives, best first (alt is sorted).
    sub = res[..., K:]
    nsub = sub.sum(-1)
    cov = np.minimum(nmiss, nsub)
    rank = np.cumsum(sub, -1)
    psub = (p[..., K:] * (sub & (rank <= cov[..., None]))).sum(-1)
    pmiss_cov = np.where(nmiss > 0, (p[..., :K] * miss).sum(-1) * cov / np.maximum(nmiss, 1), 0)
    lay = lambda x: [x[:, :, list(b)].mean() for b in bands]
    row("miss rate p", lay(miss.mean(-1)))
    row("routed weight on misses", lay((w * miss).sum(-1)))
    row("misses with resident substitute", [cov[:, :, list(b)].sum() / max(nmiss[:, :, list(b)].sum(), 1) for b in bands])
    row("p(substitute) / p(missed)", [psub[:, :, list(b)].sum() / max(pmiss_cov[:, :, list(b)].sum(), 1e-9) for b in bands])
    row("tokens with no miss", lay(nmiss == 0))

# 6. Subject-specific hot sets: hit rate of group g's test picks.
head(f"6. hot set {a.cache:.0%} hit rate on a group's test streams, built from ...")
gpop = {g: [H[l][train & (G == g)].reshape(-1, E).sum(0) for l in range(L)] for g in ug}


def hit(src_pop, g):
    return by_band(lambda l: H[l][test & (G == g)][..., np.argsort(-src_pop[l])[:cap]].sum(-1).mean() / K)


same_g = np.mean([hit(gpop[g], g) for g in ug], 0)
all_g = np.mean([hit(pop, g) for g in ug], 0)
other = np.mean([hit(gpop[h], g) for g in ug for h in ug if h != g], 0)
row("its own group's train streams", same_g)
row("all train streams", all_g)
row("a different group", other)
for g in ug:
    row(f"  own group: {g}", hit(gpop[g], g))

# 7. Working set per stream.
head("7. distinct experts used by one stream in W tokens, / E")
for W in (16, 64, 256, T):
    uni = 1 - (1 - K / E) ** W
    row(f"W={W} (uniform {uni:.2f})", by_band(lambda l: np.mean([H[l][:, s : s + W].any(1).sum(-1).mean() for s in range(0, T - W + 1, W)]) / E))

# 8. Prefetch into a per-stream LRU cache (B=1, test streams). Before layer l's
# router, load the predicted top-m ("prev tokens + prev layer") that are not
# resident; then demand-load what is still missing. Layer 0 has no prefetch.
head(f"8. per-stream LRU {a.cache:.0%} + one-layer-ahead prefetch; per routed pick")
St = test.sum()
Ht = [h[test] for h in H]
for m in (0, 10, 20, 40):
    inc = np.zeros((St, L, E), bool)
    last = np.zeros((St, L, E), np.int64)
    hist = np.zeros((St, L, E), np.float32)  # decayed previous-token picks
    dem = np.zeros(L)
    pre = np.zeros(L)
    useful = np.zeros(L)
    clock = 0
    for t in range(T):
        for l in range(L):
            clock += 1
            need = Ht[l][:, t]
            if m and l:
                sc = hist[:, l] + Ht[l - 1][:, t].astype(np.float32) @ Cs[l] / K
                top = np.argpartition(-sc, m - 1, -1)[:, :m]
                fetch = np.zeros_like(need)
                np.put_along_axis(fetch, top, True, -1)
                fetch &= ~inc[:, l]
                inc[:, l] |= fetch
                last[:, l][fetch] = clock
                if t >= 64:
                    pre[l] += fetch.sum()
                    useful[l] += (fetch & need).sum()
            miss = need & ~inc[:, l]
            if t >= 64:
                dem[l] += miss.sum()
            inc[:, l] |= need
            last[:, l][need] = clock
            over = inc[:, l].sum(-1) - cap
            for s in np.nonzero(over > 0)[0]:
                c = np.nonzero(inc[s, l])[0]
                inc[s, l, c[np.argsort(last[s, l, c], kind="stable")[: over[s]]]] = False
            hist[:, l] = 0.5 * hist[:, l] + 0.5 * need
    picks = St * (T - 64) * K  # per layer
    bb = lambda x: [x[list(b)].sum() / (picks * len(b)) for b in bands]
    if m == 0:
        row("no prefetch: demand misses", bb(dem))
    else:
        row(f"top-{m}: demand misses", bb(dem))
        row(f"top-{m}: prefetch loads", bb(pre))
        row(f"top-{m}: prefetch loads used", bb(useful))
