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

## [High] Stale SWA ring-buffer slots after speculative reject (post-wrap corruption)
- Category: Reliability / Correctness
- Location: crates/hipfire-arch-deepseek4/src/spec_decode.rs:224-233,401-428;
  read side kernels/src/deepseek4_attn_swa.hip; config `sliding_window=128`.
- Mechanism (code-confirmed 2026-07-22, no empirical run — see blocker):
  1. The draft/verify loop increments `state.n_tokens` per step so SWA K/V
     writes land IN THE REAL per-layer ring at draft positions N+1..N+K
     (spec_decode.rs:224-230). Slot index = `n_tokens % sliding_window`.
  2. On partial accept only `state.n_tokens` is restored (line 428); the ring
     DATA at the K−n_accept uncommitted slots is never invalidated.
  3. The decode SWA kernel reads slots `[0, n_valid)` LINEARLY with no
     per-slot position mask (deepseek4_attn_swa.hip) — it trusts n_valid.
  Result: PRE-wrap (total seq < 128) the stale slots sit at indices ≥ n_valid
  and are excluded → safe. POST-wrap (seq ≥ sliding_window=128) the ring is
  full; uncommitted draft writes evict positions still inside the next
  forward's 128-wide window, so the linear read consumes rejected-token K/V →
  silent attention corruption.
  Refined boundary (2026-07-22, from the verify/accept indexing): verify feeds
  `[last_token, draft[0..k-2]]` at base `last_position+1`, and the NEXT decode
  overwrites exactly ONE stale slot (the corrected token's, verify column
  `accepted_len`). So the still-stuck stale columns are `[accepted_len+1, k)`,
  nonempty only when **k ≥ n_accept+3** (never k=2; k=3 only at n_accept=0) AND
  post-wrap. Real but narrower than "any partial accept". Only the modular SWA
  ring aliases; `full_k_cache` is absolute-indexed + causally safe, and the MTP
  ring only affects draft acceptance (verify still guarantees correct output).
- Fix: IMPLEMENTED, gated OFF pending GPU validation. `spec_decode::swa_rewind`
  (behind `HIPFIRE_DEEPSEEK4_SPEC_KV_REWIND=1`) snapshots the K soon-to-be-
  evicted main-layer SWA slots before the verify (strided per-slot copy into
  per-layer `swa_k_snap`/`swa_v_snap`) and restores the uncommitted columns
  `[accepted_len+1, k)` after the accept, wrap-aware. Pure slot arithmetic is
  unit-tested (`cargo test -p hipfire-arch-deepseek4 swa_rewind`, 4/4). Enable-
  by-default is blocked on an AR-vs-spec losslessness A/B on a runnable model:
  a compressor-F16 `deepseek4-q8-mtp` re-quant is in progress on halo (the mq4
  artifact is unloadable — see below). Validation: pre-fix expect divergence
  post-128 with k=3; post-fix expect token-identical.
- Empirical status (halo, gfx1151): BLOCKED. The only deepseek4 artifact on
  halo (`deepseek-v4-flash--mq4.hfq`) will not run on the current daemon build:
  its MQ4 `compressor.wkv` is rejected by the F16-native compressor path
  (`HIPFIRE_DEEPSEEK4_COMP_F16_WMMA=1` default), and `=0` routes it to an
  unsupported `gemv.unknown`. Black-box AR-vs-spec-decode A/B needs a
  re-quantized compressor-F16 model first.
- Scope: Architectural
- Confidence: High on mechanism (code-confirmed); reproduction pending a
  runnable model.
- Note: The sibling `forward.rs` chunk/ring path is NOT affected — its
  non-aligned-with-compress-events case returns an explicit `Err`.
