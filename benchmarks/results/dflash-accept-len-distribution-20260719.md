# DFlash accept_len distribution — 2026-07-19

Decides whether Phase 2 (CPU DDTree) is worth building. The plan's own criterion:
**a tree earns its keep when divergence is EARLY and VARIABLE, and buys little
when acceptance is consistently deep.**

`stats.acceptance_hist` was already instrumented — no code change was needed.

## Method

Full Phase F corpus, 8 prompts, f16 drafter, one process per prompt:

```
--no-adaptive-b --block-size 16 --temp 0.0 --seed 1234 --max 256
```

**318 cycles, 1598 accepted, τ = 5.025** — consistent with Phase F's 5.739
(different corpus weighting; same regime).

## Distribution

| accept_len | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| cycles | 51 | 44 | 25 | 30 | 32 | 28 | 9 | 18 |

| accept_len | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
|---|---|---|---|---|---|---|---|---|
| cycles | 9 | 14 | 9 | 9 | 5 | 1 | 4 | 30 |

## Reading

**The distribution is strongly bimodal and the mean is not representative — τ=5.0
sits in a trough.** Do not reason about this drafter from τ alone.

- 29.9% of cycles diverge at accept_len ≤ 1
- 66.0% at ≤ 5
- 9.4% accept the full block (the second mode, at 15)

Per-prompt τ ranges 4.06 (`coherence_lloyd_long`) to 11.31
(`merge_sort_thinking_off`, where 9 of 13 cycles were full-accept).

## Verdict: Phase 2 is justified

Divergence is both **early** and **variable** — exactly the condition the plan
names. Nearly a third of cycles die within the first two positions, so a tree
hedging early positions addresses the dominant failure mode rather than a tail.

The bimodality also suggests the *adaptive* block-size path is worth revisiting:
the two modes look like two different prompt regimes (structured/predictable vs.
open-ended), not one distribution with spread.
