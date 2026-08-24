#!/usr/bin/env python3
"""Upper-bound tau for a Markov-head drafter, measured on a target's own output.

WHY: the DSpark drafter carries a Markov head (`markov_w1`/`markov_w2`
[vocab, rank] — a rank-r factorization of the BIGRAM transition matrix). The
case for using it as a cheap drafter is that drafting becomes near-free. That
case only holds if tau survives, and tau is cheap to bound before anyone builds
a training pipeline.

METHOD, deliberately OPTIMISTIC: fit the n-gram table on the SAME stream it is
evaluated on. That is train-on-test and invalid for estimating real accuracy —
it is used here only as a CEILING. Every eval context is therefore present in
the table, so a miss is never sparsity; it is the genuine limit that one context
recurs with different successors and argmax can only serve the majority.

A low-rank factorization can only be WORSE than the full table measured here, so
this bounds the head from above.

Feed it token streams the TARGET produced (that is what a drafter must predict):
`dflash_spec_demo` prints `DFlash tokens: [...]`.

    python3 scripts/markov_head_tau_ceiling.py tok_*.json
"""
import json, sys, collections


def tau_for(streams, order, block=8):
    tbl = collections.defaultdict(collections.Counter)
    for s in streams:
        for i in range(order, len(s)):
            tbl[tuple(s[i - order:i])][s[i]] += 1
    best = {k: c.most_common(1)[0][0] for k, c in tbl.items()}
    acc_tot = cycles = 0
    for s in streams:
        i = order
        while i + block < len(s):
            n_acc = 0
            ctx = list(s[max(0, i - order):i])
            for b in range(block - 1):        # B-1 drafted positions, as DFlash does
                pred = best.get(tuple(ctx[-order:]))
                if pred is not None and pred == s[i + b]:
                    n_acc += 1
                    ctx.append(pred)
                else:
                    break
            acc_tot += n_acc
            cycles += 1
            i += n_acc + 1                    # commit accepted + 1 bonus
    return ((acc_tot / cycles) + 1 if cycles else 0.0), cycles


def main(paths):
    streams = []
    for p in paths:
        with open(p) as f:
            streams.append(json.load(f))
    print(f"streams={len(streams)} total_tokens={sum(len(s) for s in streams)}")
    for order in (1, 2, 3):
        tau, cy = tau_for(streams, order)
        note = "  <- the Markov head's shape" if order == 1 else ""
        print(f"  order-{order}: tau={tau:.3f} (ceiling, train==test) cycles={cy}{note}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
