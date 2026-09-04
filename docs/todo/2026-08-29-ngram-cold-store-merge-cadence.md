# TODO — n-gram cold store: merge cadence and rebalancing

Status: **partly done**. The unload question below is DECIDED AND DONE
(2026-09-04); the cadence work — triggers (1)/(2)/(3) — and crash safety are
still open. The store works and persists correctly; this is about what happens
to its *layout* over a long-lived load.

## Why

`ColdStore` has two write paths, and only one of them runs during serving.

- `insert_in_place` — one block read-modify-write, one dirty page. This is the
  per-token trickle, and it now bootstraps a fresh store on its own via the
  on-demand block allocator (`cold.rs`, `alloc_block`). A key that owns no
  blocks gets exactly one on first write.
- `merge` — rewrites the whole file and recomputes the directory from scratch.
  This is the *only* thing that reassigns blocks between keys, i.e. the only
  rebalance.

`NgramSpec::merge` is currently called from exactly one place in serving:
`generate_dflash`, when a request arrives whose scope key differs from the live
state's (`crates/hipfire-serving-core/src/generate.rs`, the `carried` match).
A single-user, single-topic daemon therefore **never merges**, however long it
runs.

## What goes wrong without it

Block assignment is decided at the moment a key is first written and never
revisited, so it reflects first-touch order rather than the eventual key
distribution:

- A key that turns out hot keeps the one block it was handed. Once that block
  is full, every further gram for it must win an eviction against a resident
  (`Record::score` = count minus age), so a genuinely busy context is capped at
  170 records while cold keys sit on near-empty blocks.
- `next_free` only moves forward. Blocks belonging to keys that have gone cold
  are never reclaimed, so the store reports itself full while holding mostly
  dead weight.
- Measured skew makes this concrete: on 1M tokens of Rust the hottest key holds
  76,305 grams against a median of 28 (see
  `benchmarks/results/devlog_20260829_ngram_specdecode_replay_sweep.md`). One
  block for the hot key is off by ~450×.

None of this is a correctness problem — a miss just falls through to the drafter
— but it silently caps the value of a large store.

## Shape of the fix

A cadence, plus a signal that says a rebalance would pay.

Candidate triggers, cheapest first:

1. **Idle.** The daemon knows when it has no in-flight request. A merge of a
   256 MiB store is well under a second (measured ~7 s/GiB), so an idle-time
   merge is invisible.
2. **Promotion volume.** Merge after N grams have been promoted since the last
   one — N proportional to store size, so a small store merges more often.
3. **Pressure.** Track `insert_in_place` rejections (the return-`false` path:
   full block, newcomer does not outrank the weakest resident). A rising
   rejection rate is the direct signal that block assignment no longer fits the
   key distribution — better than either proxy above, and nearly free to count.

(3) is the one worth measuring first; it is the actual symptom. It needs a
counter on `ColdStore` and a threshold, and it composes with (1) — merge when
pressure is high *and* the daemon is idle.

Also decide:

- ~~**Unload.**~~ **DONE 2026-09-04** — `impl Drop for NgramState` merges the live
  tables when the load goes away, which covers `unload_model`'s seven call sites,
  a worker swap, and daemon exit without any of them having to remember.

  The gap was bigger than "the file is left untidy". `take_live_for` merges on a
  scope SWAP, so the single-user, single-topic daemon described at the top of
  this doc — the one that never merges — also never flushed its backlog. Every
  gram that lost an in-place insert was dropped with the `NgramSpec`.
- **Crash safety.** Still open, and slightly more reachable now that unload
  merges: unload is also what shutdown does. `merge` zeroes the data region of a
  live mmap before rewriting it, so a kill mid-merge leaves a store with a valid
  superblock and a partly zeroed body. Reads return misses, which is survivable.
  A tmp-file + rename would make it atomic and is the pattern
  `hipfire-vision-cache` already uses for its manifest — but on an mmap'd store
  that means building the replacement and re-mapping it, not just renaming a
  path, so it is real work rather than a one-line change.

## Not this

Do not move rebalancing into the per-token path. The whole reason the layout is
a directory plus a second-level hash rather than a tree is that a fixed budget
with a periodic rewrite has no need for online rebalancing — that trade is the
design, and reversing it costs a dirty page per write for a property we get for
free at merge time.
