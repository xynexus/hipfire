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

## [High] Excessive use of .unwrap() leading to potential panics
- Category: Reliability / Maintainability
- Location: Project-wide (e.g., crates/hipfire-quantize/src/main.rs, crates/hipfire-arch-deepseek4/src/forward.rs)
- Summary: The codebase heavily relies on `.unwrap()` on Results and Options, which can cause the daemon or CLI to crash abruptly on unexpected inputs.
- Suggested fix: Replace `.unwrap()` with proper error handling using `Result` and `?`, or provide descriptive `expect()` messages.
- Scope: Cross-cutting
- Confidence: High
- Note (2026-07-21): The flagship instance previously tracked here — the
  rotated `mq_x_rot` scratch aliasing in `hipfire-runtime/src/weights.rs`
  (14 raw `unsafe { …as_ref().unwrap().buf.alias() }` sites) — is resolved:
  routed through a single documented `Gpu::mq_x_rot_f32()` accessor with a
  SAFETY comment and an actionable `expect()` message. This entry remains as a
  broad, lower-priority cleanup for the remaining sites; no single reproducible
  crash is currently tracked.

## [Medium] Excessive global state via OnceLock and thread_local!
- Category: Architecture / Maintainability
- Location: Project-wide (e.g., crates/hipfire-arch-deepseek4/src/forward.rs, crates/hipfire-arch-qwen35/src/qwen35.rs, crates/hipfire-rdna/src/dispatch/mod.rs, crates/hip-bridge/src/ffi.rs)
- Summary: `OnceLock` / `thread_local!` are used for environment-derived behavior in hot and shared code paths, hiding explicit configuration inputs and increasing hidden coupling. Makes testing difficult and dependencies implicit.
- Suggested fix: Inject configuration and state through structs/context objects instead of relying on global statics. When touching a module boundary, list the env-backed globals it reads and move them behind an explicit config context.
- Scope: Architectural
- Confidence: High

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
