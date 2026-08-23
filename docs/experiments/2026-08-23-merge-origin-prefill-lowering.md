# Merging origin/master: the prefill-lowering refactor is DEFERRED, not dropped

2026-08-23. `perf/dram-read-bandwidth` was 175 ahead / 52 behind origin/master.
Merged to pick up the v2 daemon (M3b1 "the daemon owns the decode loop", M6
priority admission). One conflict, and it is worth recording rather than burying.

## The conflict

origin's M2a refactor lowered the qwen35 batched prefill through
`superop::run_layer_program`, which **deleted 4272 lines** from
`prefill_chunk.rs` (151 insertions, 4272 deletions) and moved the body into a new
`prefill_lowered.rs`.

This branch had rewritten the SAME body for compact Opus: `OqCompactG256` GEMM
routing, the interleaved activation quantize, the hoisted k-major transpose, and
the opt-in W4A4 path.

Decisive fact: **origin's lowered version has ZERO `OqCompactG256` references**;
ours has 16. Taking theirs would have removed batched compact prefill outright —
i.e. the 313 tok/s serving path this branch exists to produce.

## Resolution

`prefill_chunk.rs` = ours. `prefill_lowered.rs` is kept **compiled but unused**
behind `#[allow(dead_code)]` rather than deleted, so the refactor stays in the
tree and stays mergeable.

Integrating it later = re-applying the compact / W4A4 arms on top of
`run_layer_program`. That is real work, not a rebase, because the dtype branch
structure is exactly what the refactor removed.

## A silent loss the automerge produced

`ep.rs` auto-merged "successfully" and **dropped both of this branch's
`kv_layer_offset` initializers** (lines 319 and 2450) in favour of origin's
hunks. It surfaced only as `error[E0063]: missing field kv_layer_offset` at
build time.

Worth remembering: a clean automerge on a file both sides touched is not
evidence the result is correct. Build, then diff your own changes back out of
the merged tree — `grep -c` for each thing you added — before trusting it.

## Verified after the merge

- workspace builds clean, `no-gpu-ci` PASS
- daemon prefill still 310.3 tok/s, decode 14.5 (unchanged)
- every change from this session survives: W4A4 route, row-coalesced overlay,
  transpose hoist, GEMM grid swizzle, zero-seeded accumulators, kvarn flip
