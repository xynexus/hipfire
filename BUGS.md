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

## [RESOLVED] Quantized-from-HFQ artifacts lose config/tokenizer (dangling v2 tail pointer)
- Category: Correctness / Tooling (hipfire-quantize)
- Location: crates/hipfire-quantize/src/main.rs `HfqInputFile::open`
- Root cause: an HFQ v2 source keeps `config` / `tokenizer` / `tokenizer_config`
  / `generation_config` / `gguf_meta` in a TAIL blob addressed by a
  `tail_metadata` = `{offset, size, hash}` pointer in the FRONT metadata, where
  `offset` is a byte offset into that source file. `HfqInputFile::open` read only
  the front JSON (stopping at brace depth 0) and never dereferenced the tail, so
  the quantizer forwarded the front metadata verbatim to the derived artifact
  (main.rs ~L4705 builds the output metadata from `hfq.metadata_json`). The
  forwarded `tail_metadata.offset` then points PAST the (smaller) derived file,
  into the original source — dangling. Result: every bf16→oq/mq artifact loaded
  with NO config and NO tokenizer (`Tokenizer::from_hfq_metadata` → "tokenizer |
  gguf_meta" missing; and `config_from_hfq` → "failed to parse config"). Hit on
  all four MiniCPM5-1B.oq* variants produced 2026-07-27.
- Fix: `merge_source_tail_metadata` resolves and inlines the source tail into the
  front metadata at open time (mirrors the runtime's `merge_tail_metadata`:
  read+hash-verify the tail blob, merge its `metadata` object with front-wins
  semantics), and strips the container-level `tail_metadata` / `hfq_format` keys
  so `hfq_out::write_hfq` regenerates a correct tail for the OUTPUT. No-op for v1
  or already-inlined sources. Unit tests: `merge_source_tail_*` (inline,
  front-wins, no-op, hash-mismatch). Weights are untouched — the bug was
  metadata-only.
- Note: existing broken artifacts were repaired out-of-band (tokenizer/config
  re-injected, weights byte-identical); with this fix, re-quantizing from a v2
  bf16 source now embeds them correctly at emit time.
- Confidence: High (root-caused, unit-tested; re-quant end-to-end recommended).

## [RESOLVED] Batched prefill garbage for bf16/f16 llama models (was: "attention_q8_0_kv_batched masked prefill garbage for decoupled head_dim")
- Category: Correctness / Dispatch
- Location: crates/hipfire-runtime/src/llama.rs `forward_prefill_chunk`
  (QKV / wo / gate+up / down projection dispatch)
- Root cause (CONFIRMED, not the attention kernel): `is_batchable_la`
  (crates/hipfire-runtime/src/dispatch.rs L100 `bf16_f16_wmma` arm) marks BF16/F16
  weights batchable on every WMMA arch incl. gfx1103, so a native-bf16 llama model
  (e.g. MiniCPM5-1B.bf16) routes into `forward_prefill_chunk`. But the chunk's four
  per-linear projection blocks only had arms for the quantized formats
  (6bit / q8 / mq3 / fp4 / else=HFQ4). BF16/F16 matched NONE, so every projection
  fell through the `else` to `gemm_qkv_hfq4g256` / `gemm_*_hfq4g256_residual`, which
  reinterpret the raw bf16 weight bytes as 4-bit HFQ4 blocks — garbage from layer 0.
  Only q_dim≠hidden was incidental (MiniCPM happens to be both bf16 AND decoupled);
  the true trigger is the bf16/f16 weight dtype. The attention kernels
  (attention_q8_0_kv_batched generic + gfx1103), the Q8 KV write, and the KV read
  stride were all verified CORRECT. The original bisection couldn't isolate it
  because BOTH the mis-projection and the attention live inside path C and it only
  compared final logits; path B (`prefill_forward`) uses the dtype-dispatched
  `weight_gemm`, which handles bf16 — that's why B was clean.
- Fix: add BF16/F16 arms to all four projection blocks in `forward_prefill_chunk`,
  routing through `crate::weights::weight_gemm` (identical to the correct
  `prefill_forward` path). Verified with `debug_batched_prefill_divergence` on
  MiniCPM5-1B.bf16 / gfx1103: path C (flash/masked) cosine vs per-token reference
  went from **−0.194 → 0.99996**, argmax now matches at every prefix length
  (n=4/6/8). Regression guard: `chunk_projection_handles_dtype` helper + a
  `debug_assert` in the chunk + the no-GPU unit test
  `llama::tests::chunk_projection_covers_all_batchable_dtypes`, which asserts every
  dtype `is_batchable_la` accepts has an explicit projection arm (so batchability
  and projection coverage can't drift apart again).
- Serving route (measured, decided): with the projection fix BOTH batched
  prefill paths are correct, so the earlier `prefill_forward` route in
  LlamaBackend::prefill (crates/hipfire-arch-llama/src/arch.rs) is no longer a
  correctness workaround. A clean same-build A/B on gfx1103 / MiniCPM5-1B.bf16
  (`hipfire bench`, 5 reps) shows `prefill_forward` (attention_causal_batched) at
  pp512 **602.3 ± 2.4 t/s** vs the fixed chunked path (attention_q8_0_kv_batched)
  at **580.8 ± 2.3 t/s** — the `prefill_forward` route is ~3.6% faster (tg128
  identical, ~11.9 t/s). So it is RETAINED on perf grounds, and the chunk-path
  projection fix stands as correctness + the coverage guard. (The one-time "1227
  t/s garbage" figure was the mis-dispatched HFQ4 kernel reading half the bytes,
  never a real correct speed.)
- Confidence: High (root-caused + numerically verified end-to-end + benchmarked).

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
