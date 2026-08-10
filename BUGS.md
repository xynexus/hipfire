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

## [High] Indexed OQ MoE decode kernels produce non-finite KLD
- Category: Correctness / Kernels (Opus routed experts)
- Location: `gemv_oq4g256_moe_*`/`gemv_oq8g256_moe_*` indexed kernels, gated by
  `qwen35_moe_oq_indexed_decode_enabled` (`hipfire-arch-qwen35/src/qwen35/mod.rs:1824`)
- Summary: with `HIPFIRE_QWEN35_MOE_OQ_INDEXED=1`, ALL SEVEN Opus cells of the
  `qwen3_5_moe` tiny fixture fail with **non-finite KLD** (oq4, oq8, oq4+, oq4++,
  oq4.25++, oq8+, oq8++). With the flag unset the same seven **pass** with drift
  `-0` against baseline. One variable, opposite verdicts.
- Reproduce in minutes:
  `HIPFIRE_TINYQUANT_FAMILIES=qwen3_5_moe HIPFIRE_QWEN35_MOE_OQ_INDEXED=1 ./tests/tiny-quant-gate.sh`
- Why it matters beyond the flag: **paged routed-expert residency REQUIRES this
  path.** Under paging `routed_experts_resident` is false by design, so
  `check_moe_decode_supported` admits only the GPU-top-K indexed route. Paged
  Opus MoE is therefore blocked on this defect — on a 35B it presents as ~160 s
  per MoE layer with one core pinned, which is how it was found.
- Non-indexed Opus MoE decode is healthy; this is specific to the indexed routed
  kernels.
- Suggested fix: debug on the tiny fixture, not a 35B. The gate comment already
  describes this as a known "finite-KLD failure" being debugged, so this entry is
  a reproduction and a scope statement rather than a new discovery.
- Scope: Correctness (premier quant family, paged path)
- Confidence: High (clean A/B, one variable)

## [High] tiny-quant is RED on master for Opus across four MoE families
- Category: Correctness / Quant (Opus)
- Location: `tests/tiny-quant-gate.sh` cells; baselines in `tests/tiny-quant-baselines.txt`
- Summary: `./tests/tiny-affected-gate.sh --require-coverage` fails 14 cells, all
  Opus, on deepseek4, deepseek4_compressed, lfm2_moe and minimax. Worst is
  deepseek4 `kld:oq8` at KLD **0.038652 vs baseline 0.000193** — ~200x, against a
  RELATIVE budget of 25% of baseline (±0.000048). deepseek4_compressed oq8 is
  0.038414 vs 0.000107. minimax oq4/oq4+/oq4++/oq4.25++ drift ~2-3x. Two cells do
  not drift but hard-fail the quantizer: `lfm2_moe` and `minimax`
  `quantize:oq8++(calib)` exit 1 with "calibrated plus format requested, but no
  LDLQ-eligible tensors were attempted".
- Pre-existing, NOT from any change on the current branch: verified by reverting
  the branch's quant-format/pager/quant.rs edits to their parent and re-running
  the identical gate with the identical `--files-from` list — the failure sets are
  BYTE-IDENTICAL (same 14 cells, same drift values).
- Why it was not caught sooner: `tiny-affected-gate` selects a family allowlist
  from the touched paths, so a green run means "the SELECTED families passed", not
  "the suite passed". Runs that touched only qwen35 paths never selected these
  four families. Comparing two gate runs with different `--files-from` inputs
  compares different tests.
- **RESOLVED as to cause, and a fix already exists off-master (found 2026-08-10).**
  The open question below was answered before this entry was written, on the
  unmerged branch `fix/oq8-from-flag-and-rotation-guards`
  (`docs/plans/moe-expert-residency-unification.md`, Phase 0, commit `691e7730e`
  — that file exists ONLY on that branch). Do not re-bisect this.
  - **9 of the 14 are one real regression**: deepseek4 ×3, deepseek4_compressed ×3,
    lfm2_moe ×3. Green at `0060481ee`, red at `8b9ee5392`.
  - **Root cause**: `8b9ee5392` inserted an unguarded `_` wildcard arm *above* the
    `oq8` literals in `HfqInputFormat::from_flag`
    (`hipfire-quantize/src/main.rs:3919`), replacing a guarded
    `_ if parse_opus_mixed_format(flag).is_some()`. That made `"oq8"`/`"oq8+"`/
    `"oq8++"` unreachable, so `from_flag` returned `None` for every OQ8 flag and
    each call site degraded differently — silently skipped tensors (lfm2_moe
    scoring KLD exactly 0.000000), a wrong-format fallback (deepseek4 at 0.0387),
    or the "no LDLQ-eligible tensors were attempted" hard error. `oq4` was
    unaffected because its arm sits above the wildcard. **rustc reported this as
    `unreachable_patterns` — the warning was the diagnosis.**
  - **The remaining 5 are minimax and are genuinely pre-existing**, failing
    identically at the baseline-record commit `5dc01e4b0`. Fixing the wildcard
    *unmasks* minimax rather than fixing it: its oq8 cells were passing only
    because the quantizer was not producing OQ8 at all. Cross-referenced to
    `2026-08-05-opus-across-model-families.md:82-93` (oq4 0.003531, oq8 0.000259).
    **minimax cannot serve as a parity oracle**; scope it out with
    `HIPFIRE_TINYQUANT_FAMILIES` until that separate cause is found.
- Action for master: this is a merge/cherry-pick decision, not a debugging task.
  The 14-cell figure above remains accurate *for master as it stands today*.
- Scope: Correctness (premier quant family)
- Confidence: High (byte-identical reproduction on pristine code)

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
## [RESOLVED] hipfire-daemon inference worker killed on client disconnect (was theorized as "GPU fault under model-swap churn")
- TRUE ROOT CAUSE (confirmed): a client closing the socket mid-generation, NOT
  model-swap churn. On cancel, chat.rs `execute_blocking_chat_cancellable` hits
  `Ok(None)` and `drop(engine)`s the `DaemonEngine`; `StdioTransport` is spawned
  `kill_on_drop(true)`, so the drop SIGKILLs the whole worker (destroying the
  loaded model; recovery was a lazy reload / "Broken pipe"). Intermittent because
  the disconnect must land while generating; no dmesg trace because SIGKILL(9)
  leaves none. The "model-swap churn / GPU fault" below was a red herring from an
  aggressive repro hammer — real trigger is disconnect (e.g. a coding agent
  `pkill`ing a request, or a client timeout).
- FIX: cooperative cancellation. The daemon installs a SIGUSR1 handler that sets
  a process-global `GENERATION_CANCEL: AtomicBool` (async-signal-safe: atomic
  store only); the shared decode loop (`arch.rs decode_loop_with_timing_terminators`
  + the qwen35 / multi-GPU loops) checks it at loop TOP (KV-safe: identical to a
  natural max_tokens stop, drops only the un-written pending sample) and stops,
  emitting a normal terminal `done`. The frontend, on disconnect, sends SIGUSR1 to
  the worker (`DaemonEngine::abort_and_drain` → `libc::kill`) and drains to the
  (now-fast) terminal event, then RESTORES the engine instead of dropping it —
  worker + model stay resident. Verified on gfx1103: worker PID stable across
  10+ disconnects (was killed in 1–2), post-disconnect request in ~0.16–0.78 s
  (proves the gen was cancelled, not run to max on the serial worker), normal gen
  unaffected. NOT covered (fall back to the old drop): spec-decode (mtp/dflash)
  and VL loops (multi-token-per-iter, not provably KV-safe to break) — a
  follow-up. Related: the worker still does not auto-respawn; #204 added durable
  `~/.hipfire/daemon.log` + honest `degraded` status.
- --- earlier (INCORRECT) hypothesis, kept for the record ---
- Category: Reliability / Correctness (worker process)
- Location: hipfire-daemon worker (spawned by `hipfire serve` via
  crates/hipfire-daemon-adapter/src/lib.rs); GPU load/unload + decode path.
- Summary: The `hipfire-daemon` inference worker (a child of the `hipfire serve`
  front-end) dies intermittently. Reproduced under sustained model-swap churn on
  gfx1103: crash at req 308 (`MiniCPM5-1B.oq4.25++.coarse`, a 48-token decode)
  after 307 clean reloads; also observed under a single coding agent's light
  normal use. It is cumulative, not a leak (worker RSS flat), not a concurrency
  race (a `text_concurrency` limiter serializes loads), and not a plain OOM
  (45 GB GTT budget). Signature: GPU-state accumulation across many model
  load/unload cycles, tripping a fault on a subsequent decode. Exact fault line
  still UNCAPTURED at the time of filing — now capturable, see below.
- Two contributing DEFECTS, both FIXED (observability, PR
  `fix/daemon-crash-logging-and-worker-health`):
  1. The worker died SILENTLY — its stderr was only re-emitted to the front-end's
     (variably-routed) stderr and its death was a silent EOF `break`, so the
     backtrace/signal evaporated. FIX: set `RUST_BACKTRACE=1` on the worker, tee
     its stderr to a durable `~/.hipfire/daemon.log`, log an EOF death marker, and
     log the exit status/signal via `DaemonEngine::worker_alive`
     (`try_wait`). Verified: a SIGKILL now logs `signal: 9 (SIGKILL)` + a
     daemon.log marker.
  2. `hipfire status` reported `healthy` while the worker was dead — it only pinged
     the HTTP front-end. FIX: `/health` now probes the worker and reports
     `status:degraded` + `worker_alive:false`; `hipfire status` renders
     `degraded (inference worker down)` with a pointer to daemon.log.
- Also found (NOT fixed — follow-up): the worker does NOT auto-respawn. After it
  dies, requests fail with raw `Broken pipe (os error 32)` until `hipfire restart`.
  Needs: respawn-on-death + a clean "worker down" error (crash-handling, #3).
- Next: re-run under the durable log to capture the real signal/backtrace, then
  fix the GPU fault. Mitigation: the `*.coarse` variants load heavier FP32 KV
  (no `q8` override) which raises swap-churn stress.
- Confidence: High on reproduction + the two observability fixes; root cause of
  the GPU fault itself pending a captured trace.

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

## [High] bf16 KLD reference artifacts contain chunk 0 replicated 1175×
- Category: Correctness / Evidence tooling
- Location: `/srv/hipfire/kldrefs/qwen3.5-{0.8b,2b,4b}-bf16.kldref.hfq`
  (and the `.arch0.bak` copy under `/srv/Public`); produced 2026-06-05 by
  `build_kld_ref_hipfire` (hipfire 0.2.0). That producer is no longer in the
  tree — only the artifacts remain.
- Summary: `kldref.tokens` is correct (1175 contiguous 2048-token windows of the
  wikitext2 slice), but the `kldref.top_indices` / `top_log_probs` /
  `residual_mass` blocks for EVERY chunk are byte-identical to chunk 0's. The
  block cursor never advanced. Verified with
  `cargo run --release -p hipfire-runtime --example kldref_selftest -- <ref>`:
  chunk 0's argmax agrees with the corpus's next token 44–50% of the time (a
  healthy bf16 reference), chunks 1..N agree ~1% (chance), and a slide of chunk
  1's blocks over the token stream best-matches token position 1025 — chunk 0's
  scoring window. All three model sizes show it identically, so it is a producer
  bug, not file corruption.
- Impact: any absolute KLD-vs-bf16 computed from these files past chunk 0 is
  meaningless (a candidate is scored against a different passage's predictions;
  observed ~11.5 nats/tok vs ~0.3 for the valid chunk). Chunk 0 alone (1023
  positions) IS usable, which is how the defect stayed invisible to spot checks.
  The daemon's own loader independently refuses these files — their metadata
  `arch_id` is 0 (`read_hfqm_kld_ref_archive`, `hipfire-daemon/src/main.rs`) — so
  the in-tree evidence path was never exposed; the risk is ad-hoc harnesses that
  bypass that check.
- Suggested fix: regenerate against a bf16 `.hfq` (none currently on disk for
  qwen3.5-0.8b; `/srv/hipfire/archives/models--Qwen--Qwen3.5-0.8B.hfa` holds the
  HF source) with a per-chunk block-cursor assertion, and have any new reader
  run the `kldref_selftest` agreement check before trusting a reference. Until
  then treat these three artifacts as single-chunk.
- Scope: Tooling / evidence integrity
- Confidence: High (self-test is deterministic and reproduces on all 3 files)
