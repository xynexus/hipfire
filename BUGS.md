# Bugs To Investigate

This is a lightweight reminder list. Add a short description, or record
revision + file + line number with a one-line explanation. Do not turn entries
into full investigations here.

## [Critical] Example Bug
- Category: Architecture / Maintainability
- Location: crates/hipfire-arch-qwen35/src/qwen35.rs, crates/hipfire-rdna/src/dispatch/mod.rs
- Summary: This is an example of a brief bug report
- Suggested fix: Do nothing 
- Scope: Architectural
- Confidence: High

## [Low] Opportunistic .unwrap() → error-handling cleanup (convention, not a tracked bug)
- Category: Reliability / Maintainability
- Location: Project-wide (~6.8k non-test `.unwrap()` sites; most guard true
  invariants, not user input)
- Summary: Prefer `?`/descriptive `expect()` over bare `.unwrap()` on paths
  that can fail on user input or external files. This is a fix-as-you-touch
  convention, not a specific reproducible crash — a blanket sweep is neither
  feasible (6.8k sites) nor desirable (many unwraps encode real invariants).
- Named exemplars — both resolved (2026-07-21/22):
  - `hipfire-runtime/src/weights.rs`: 14 raw
    `unsafe { …as_ref().unwrap().buf.alias() }` rotated-scratch sites → one
    documented `Gpu::mq_x_rot_f32()` accessor (SAFETY comment + actionable
    `expect()`).
  - `hipfire-quantize/src/main.rs` `SafetensorsFile::open`: the model-load
    header parse (`from_utf8`/`from_str`/`from_value`/8-byte length) now returns
    clean `io::Error(InvalidData)` messages instead of panicking on a
    truncated/malformed `.safetensors` file.
- Confidence: Low (convention; no open crash tracked)

## [Closed] "Excessive" global state via OnceLock — intentional, not a defect
- Category: Architecture / Maintainability
- Location: crates/hipfire-arch-deepseek4/src/forward.rs (`mod env_cache`),
  crates/hipfire-rdna/src/dispatch/mod.rs, crates/hip-bridge/src/ffi.rs
- Resolution (2026-07-22): Investigated. The flagged `OnceLock`/`thread_local!`
  statics are a deliberate, documented hot-path optimization: they cache
  `HIPFIRE_*` env-derived debug/tuning knobs read once, because an uncached
  `std::env::var` per lookup cost ~200μs/token (43 layers × ~5 lookups × ~1μs
  syscall). They are set-once, read-only, and idiomatic. Converting them to
  injected config context would re-add that per-token cost (or require threading
  a config struct through the entire hot path) for near-zero benefit — these are
  debug/tuning knobs, not core mutable state. Not a bug.
- Residual guidance (minor): do not introduce globals for *core mutable state*
  or *user-facing config*; those belong in explicit context objects. Env
  debug/tuning knobs behind `OnceLock` remain the accepted pattern.

## [High] Stale SWA ring-buffer cache slots after speculative reject
- Category: Reliability / Correctness
- Location: crates/hipfire-arch-deepseek4/src/spec_decode.rs:412-427
- Summary: After a speculative verify, MTP/main SWA caches hold entries at
  positions beyond `n_accept` computed from rejected draft tokens. They are
  normally invalidated when the caller's next forward overwrites the ring
  slots, but a forward that READS a stale slot before overwriting it (narrow
  ring-index-alias window) would consume rejected-token state. Documented in
  code as a production-hardening follow-up; tied to the still-open spec-decode
  hybrid-state rewind gap.
- Suggested fix: Speculative verify must write into scratch cache state and
  commit only the accepted prefix, or explicitly invalidate/rewind every
  per-layer SWA slot beyond `n_accept` before returning. Moving `n_tokens`
  alone is insufficient.
- Scope: Architectural
- Confidence: Medium (latent; no confirmed reproduction yet)
- Note: The sibling `forward.rs` chunk/ring path is NOT affected — its
  non-aligned-with-compress-events case returns an explicit `Err` rather than
  silently corrupting the ring, so that edge is guarded.
