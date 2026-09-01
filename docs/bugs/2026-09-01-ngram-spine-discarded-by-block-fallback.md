# Drafter-free n-gram spec decode never speculated — the spine was discarded every step

Status: **FIXED 2026-09-01**. Introduced 2026-08-29 by `375b09642` ("spec
decode: make the drafter optional in spec_step_dflash"), which enabled the
drafter-free path but left nothing to supply its verify width. The feature was
inert for its entire life, and measurably slower than not using it.

## The defect

`spec_step_dflash` sized the verify block as:

    let requested_b = requested_block_size(block_size_override, draft_cfg.map(|c| c.block_size));
    let b = match pld_spine {
        Some(pld) => (1 + pld.len()).min(requested_b).max(1),
        None => requested_b,
    };

and `requested_block_size` is
`block_size_override.or(draft_block_size).unwrap_or(1)`.

With **no drafter** there is no `draft_cfg.block_size`, and with no
`HIPFIRE_DFLASH_BLOCK` pin there is no override — so `requested_b` is the bare
fallback `1`. A 16-token n-gram spine then became `b = 17.min(1) = 1`, which
this same function documents as *"an empty spine means b=1 ... which is exactly
an AR step"*. So every step of drafter-free n-gram speculation was a plain
autoregressive step.

The adaptive `BlockController` does not cover for this: it is constructed only
when `df.block_size > 2`, which no drafter satisfies. Nothing else in the
drafter-free path proposes a width.

## Why it was invisible

The n-gram tier reported healthy numbers, because it *was* healthy — it drafted
correctly and its work was thrown away downstream:

    "coverage":0.992  "hot_entries":9876  "drafted":9321  "accepted":0

Only the outer verify window showed the failure — `"mean_draft_len":1.0` and an
`acceptance_hist` with everything in bucket 0. Two adjacent counters describing
the same step disagreed, and the tier-local one looked fine.

There was also a passing unit test asserting `requested_block_size(None, None)
== 1`. That assertion is correct — with no drafter and no spine, `b = 1` is the
right "do not speculate" step. The bug was applying that fallback to a case
where a spine *was* supplied, which no test covered.

## Measured (gfx1103 / nix2, `qwen3.5-0.8b--oq4++.hfq`, 2×400 greedy tokens, code prompt)

| arm | decode tok/s | mean_draft_len | accepted / drafted |
|---|---|---|---|
| n-gram off (plain AR) | 72.0 / 72.2 | — | — |
| n-gram on, before fix | 66.0 / 66.1 | 1.0 | 0 / 18897 |
| `HIPFIRE_DFLASH_BLOCK=16` (proves causation) | 397 / 557 | 16.0 | 740 / 1206 |
| n-gram on, after fix (no env var) | 328 / 397 | 17.0 | 761 / 1182 |

Note the second row: enabling the feature was an **8% net loss** against not
enabling it. The spine still cost something to build; it just never bought
anything. That is the shape to look for — a mechanism whose own stats look good
while the number it exists to move goes the wrong way.

This corpus is a favourable one (the prompt is Rust source, which the n-gram
predicts well), so treat the ratio as a best case, not a headline. The direction
and the zero are the findings.

## Fix

`crates/hipfire-arch-qwen35/src/speculative.rs`: the width decision moved into
`effective_block_size`, which takes `untrained_cap: Option<usize>` — `Some(
verify_scratch.max_n)` exactly when no drafter and no override supply a trained
width. There the spine *is* the width, bounded by the verify scratch the caller
sized (the same `max_n` the verify already asserts on). An empty spine still
yields `b = 1`, and a drafter or an explicit override is still authoritative, so
the DFlash path is unchanged.

Regression test:
`speculative::tests::a_supplied_spine_is_not_bounded_by_the_no_drafter_fallback`,
placed next to the `requested_block_size(None, None) == 1` test it does not
contradict.

## Left open

The drafter-free path has no adaptive width control — `b` is now whatever the
n-gram's `max_spine` produces, and `BlockController`'s `df.block_size > 2` gate
keeps it out. Tuning that width is a separate, smaller piece of work; the
measured 328/397 vs the pinned 397/557 suggests there is something to win.
