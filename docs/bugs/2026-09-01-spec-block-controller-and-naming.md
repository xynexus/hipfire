# The verify block: a controller that could not reach its range, and a knob named after the wrong thing

Status: **DONE 2026-09-01.** Follow-on to
`2026-09-01-ngram-spine-discarded-by-block-fallback.md`, which fixed the spine
being discarded. This is the next question that finding raised: once the spine
IS verified, what sets its width, and who can change it?

## 1. `BlockController` could never return a block above 8

Three independent depth caps, written as bare literals in three places:

    observe():        let depth = n_proposed.min(8);        // survival only to k=8
    argmax_block():   for n in 1..=self.max_tried.min(8)    // can never RETURN >8
    observe_timing(): (2..10).contains(&n_verify)           // timing only to n=9

The second is decisive: `max_block` could be any value the caller liked and the
argmax still could not name a block above 8. Passing `max_block = 16` was
accepted, ramped to, and then silently discarded on the first argmax.

Invisible while only a DFlash drafter drove this — trained blocks are ≤ 8, so
the cap never bound. The drafter-free n-gram path routinely wants 16 and hit all
three at once.

Now one `MAX_DEPTH = 32` constant, with every array and range derived from it.

## 2. ~~The controller does not pay on the n-gram path~~ — WITHDRAWN 2026-09-01

**This section's conclusion was wrong and is retracted.** It reported the
controller as 10-30% slower than a fixed block of 16 and reasoned that n-gram
survival is flat, so "the optimum sits at the boundary and an argmax has nothing
to find".

Both halves were wrong.

**The reasoning was wrong.** A search whose range contains the fixed value
cannot legitimately lose to it — a correct argmax evaluating a boundary optimum
returns the boundary. "It lost" was evidence the search was broken, not evidence
there was nothing to find. It was broken, in four ways, all now fixed:

- `argmax_block` iterated the BLOCK but indexed survival by drafted DEPTH.
  `observe` receives `n_proposed = drafted.len() = block - 1`, so at
  `block == max_block` the top index never got a sample: every trace line read
  `16:0.00`, tau could not grow there, and **the largest block could never win**.
  The fixed configuration was not in the controller's own reachable set.
- Depths the block stopped reaching kept a stale estimate measured during the
  ramp, when the n-gram store was still cold. That is a downward RATCHET, not
  the "cap-trap fix" the comment claimed: a traced run walked 15 -> 14 -> 13
  while true acceptance at those depths was ~0.9.
- The survival estimator is not monotone — a real trace read
  `1:0.89 2:0.85 3:0.94`, impossible for `P(accept_len >= k)` — because each
  depth conditions on a different subset, which the argmax then sums as one
  curve.
- The controller was rebuilt per request, discarding the fitted cost curve that
  `reset()` documents as "calibrate once, reuse across requests". Two generates
  fitted two different curves for one GPU, one of them off a 6-sample slice with
  a negative slope clamped to `dt=0`.

**The measurement was also invalid**, independently. Block width changes the
emitted token sequence on this stack, so the two arms generated different text —
one 1943 tokens, the other 1999 — and their tok/s were not measuring the same
work. See `2026-09-01-spec-decode-not-output-equivalent-to-ar.md`.

After the four fixes, on the one comparison that was valid (a warm request where
both arms emitted a byte-identical sequence), the controller **matched** the
fixed block: 338.7 vs 343.9 tok/s, at better verify efficiency (0.78 vs 0.58),
and the argmax selected 16 on all 503 post-ramp calls.

`spec_adaptive_block` still ships OFF, but as the conservative default pending a
benchmark that can hold output constant — not as a measured verdict against it.

### Still open on the controller

- The ramp sweeps upward from `min_block`, spending its most expensive windows
  at the worst width before climbing. For a workload whose optimum is at the top
  that is the worst exploration order.
- `max_block` is `max_spine`, so the search cannot look ABOVE the spine supply.
  If 16 wins, 16 is where the fence is, not a demonstrated optimum: proving it
  needs the range and the spine raised together.

## 3. `dflash_adaptive_b` was a setting that applied to nothing

    schema.rs        "dflash_adaptive_b"  default Some("true")
    lib.rs           default_dflash_adaptive_b() -> bool { true }
    model.rs doc     "default OFF (opt-in)"
    load.rs (both)   adaptive_b: false     <- hardcoded, param never read

Three sources of truth disagreeing: the schema said on, the doc said off, the
code said false unconditionally. The config field was fully plumbed into
`ModelLoadParams` and both load sites ignored it. A per-load param path did
reach `df.adaptive_b`, but with `.unwrap_or(false)` — so config could not reach
it by any route.

Now `spec_adaptive_block`, default `false` (what the code actually did), applied
once by the daemon to whichever speculative state exists. `load.rs` sets
fallbacks only; the layer that reads config is the layer that decides.

## 4. The knob was named after the wrong thing

`HIPFIRE_DFLASH_BLOCK` was a raw `std::env` read — the pattern AGENTS.md names
explicitly: env is a resolution LAYER, not a bypass; a direct read silently
outranks config and nothing announces it.

It is also misnamed. `spec_step_dflash` is the shared VERIFY engine, not "the
DFlash path": it verifies a DFlash drafter's block and a drafter-free n-gram
spine alike, and the block size applies with no drafter loaded at all. The name
caused real confusion — a reader seeing `HIPFIRE_DFLASH_BLOCK=16` change
throughput on a model with no drafter reasonably concludes it enabled a drafter.
It does not; it supplies one integer, the verify width.

Now `spec_block` (0 = auto), a schema field. `spec_step_dflash` keeps its name —
renaming the engine is a much larger diff — but the settings no longer claim the
drafter owns something it does not.

The output JSON still reports `"dflash": true` on drafter-free runs, meaning
only "the shared verify engine ran". That is the same misnomer one layer out and
is **not** fixed here; it is a wire-format field with consumers.

## 5. A renamed key's ENV spelling used to evaporate

Found while adding the rename. `apply_renamed_keys` is applied to config LAYERS,
but the environment layer is built by iterating schema FIELDS — and a retired
name is not a field. So `HIPFIRE_NGRAM_MAX_SPINE` and every other renamed key's
env spelling silently stopped working, with no diagnostic. This affected all 11
pre-existing renames, not just the two added here.

The env layer now probes old spellings too and inserts under the OLD key, so
`apply_renamed_keys` moves the value and emits the usual warning — the
deprecation path is now identical to the config-file one.

## 6. The daemon threw those diagnostics away

`load_config_bundle().config` discards `.diagnostics`, and the admin console was
the only place that ever rendered them. So even a correctly-generated rename
warning never reached a daemon operator. The daemon now logs them.

Verified end to end — `HIPFIRE_DFLASH_BLOCK=8` on a live load:

    WARN config: config key `dflash_block` was renamed to `spec_block`;
         the value was applied, but rename it — the old name will stop working.

and `HIPFIRE_DFLASH_BLOCK=8` vs `HIPFIRE_SPEC_BLOCK=8` produce identical
throughput to three digits (200.7/275.9 vs 200.8/275.9), both distinct from auto
(231.9/312.7). The old spelling applies AND says so.

## Verification

- `tests/no-gpu-ci.sh` exit 0; `tests/tiny-state-gate.sh` PASS (18/18 output
  hashes match baseline — token output is unchanged)
- `cargo clippy --workspace --all-targets` clean (two pre-existing
  `hipfire-coexistence` warnings untouched)
- new tests: `renamed_env_tests` (old spelling resolves; new wins over old;
  every rename target is a live field) and `renamed_env_e2e::legacy_env_applies_and_warns`
  (the value applies AND a diagnostic is produced — asserting only the value
  would have passed while the operator was told nothing)
- default decode path unchanged: auto measures 231.9/312.7 with mean_draft_len
  10.357/14.0, matching the pre-change baseline of 231.6/313.0 at the same draft
  lengths

## Left open

- `"dflash": true` in the generate JSON on drafter-free runs (§4).
- `spec_step_dflash` is still named for one of its two callers.
- `lifecycle.rs` calls `load_config_bundle()` deep in the stack, which AGENTS.md
  warns drops CLI overrides and every `model_overrides` entry. The existing
  ngram settings already had this shape and the new two follow it; fixing it
  means routing them through `ModelLoadParams` like the rest, which is a
  separate change. The daemon already warns when it happens.
