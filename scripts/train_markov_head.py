#!/usr/bin/env python3
"""Train a low-rank Markov (bigram) head and emit an MKV1 file.

Mirrors DSpark's `markov_w1`/`markov_w2` `[vocab, rank]`: the head scores
next-token logits as `w1[prev] . w2[next]`, i.e. a rank-r factorization of the
bigram transition matrix.

Only tokens OBSERVED in training get a row. That is not a shortcut — an unseen
`prev` has no evidence, and restricting the argmax to observed `next` keeps
scoring O(n_obs * r) instead of O(vocab * r), which is what makes a host-side
draft cheap enough to be worth doing at all.

    python3 scripts/train_markov_head.py --rank 32 --out head.mkv tok_*.json
"""
import argparse, json, collections
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--out", required=True)
    ap.add_argument("streams", nargs="+")
    a = ap.parse_args()

    streams = [json.load(open(p)) for p in a.streams]
    counts = collections.Counter()
    for s in streams:
        for x, y in zip(s, s[1:]):
            counts[(x, y)] += 1
    obs = sorted({t for pair in counts for t in pair})
    idx = {t: i for i, t in enumerate(obs)}
    n = len(obs)
    print(f"streams={len(streams)} bigrams={sum(counts.values())} distinct_tokens={n}")

    # log1p on counts: the head only has to RANK successors, and log damps the
    # few very frequent pairs that would otherwise dominate every singular vector.
    m = np.zeros((n, n), dtype=np.float32)
    for (x, y), c in counts.items():
        m[idx[x], idx[y]] = np.log1p(c)

    r = min(a.rank, n)
    u, s, vt = np.linalg.svd(m, full_matrices=False)
    w1 = (u[:, :r] * s[:r]).astype(np.float32)
    w2 = vt[:r, :].T.astype(np.float32)

    approx = w1 @ w2.T
    exact_hits = int((approx.argmax(1) == m.argmax(1)).sum())
    print(f"rank={r}  argmax agreement with the FULL bigram table: {exact_hits}/{n}"
          f" ({100.0*exact_hits/n:.1f}%)")

    with open(a.out, "wb") as f:
        f.write(b"MKV1")
        f.write(np.uint32(r).tobytes())
        f.write(np.uint32(n).tobytes())
        f.write(np.asarray(obs, dtype=np.uint32).tobytes())
        f.write(w1.tobytes())
        f.write(w2.tobytes())
    print(f"wrote {a.out}: rank={r} n_obs={n} "
          f"({4 + 8 + 4*n + 4*n*r*2} bytes)")


if __name__ == "__main__":
    main()
