# DFlash NPU-draft ‖ GPU-verify overlap seam — scope

Scopes the load-bearing integration for the DFlash-on-NPU drafter: draft block
N+1 on the NPU while the GPU verifies block N. Prereqs are done and measured —
see `docs/plans/2026-07-19-dflash-phase0-brief.md`. The reality check (task #31)
showed the NPU drafter beats GPU-only AR by 1.6–3.4× **only under overlap**; the
serial path is marginal. This doc is the data-flow-grounded design for that
overlap. All citations are file:line on branch `chaingun`.

## 1. The dependency verdict — it is NUMERICAL, and it sets the whole shape

**Draft N+1 depends on verify N's output, and not merely for control flow — for
the actual floating-point context vectors.** Traced through one
`spec_step_dflash` (`crates/hipfire-arch-qwen35/src/speculative.rs:6756`):

1. **Verify** (`:7292`, `verify_dflash_block`) runs the target over block N's B
   tokens and writes B hidden rows into `hidden_rb` (`HiddenStateRingBuffer`,
   `:5482`). These are the **target model's** hidden states for block N.
2. **Scatter** (`:7565`, default `ctx_slice=None`): the first
   `rows_to_keep = accept_len + 1` of those rows are pushed into the draft's
   context `draft_scratch.target_hidden` via a GPU D2D scatter. (On the
   `ctx_slice=Some` path they are `download_hidden_block`'d into
   `target_hidden_host`, `:7555`.)
3. **Next cycle draft** (`:6979`, `draft_forward_opts`) consumes exactly that
   buffer as `target_hidden` — `thp = rmsnorm(fc @ target_hidden)`
   (`dflash.rs:1434`), and every layer's context K/V derives from `thp`.

So `draft_scratch.target_hidden` **is** the verify forward's output. Two things
draft N+1 needs from verify N, neither available until verify N runs:

- **The hidden vectors** for block N's committed positions
  (`rows_to_keep × ne × h` f32) — produced by the target body; the drafter
  *consumes* them, it cannot produce them (`draft_forward_opts` takes
  `target_hidden` as an input, `dflash.rs:1331`). **Not predictable.**
- **The control outcome** — `accept_len` (`:7322`) sets `rows_to_keep`
  (`:7552`) and next-cycle RoPE positions (`:7581`); `bonus_token` (`:7431`)
  becomes next cycle's `seed_token` (`:6835`). **Predictable (cheaply).**

Because the hidden vectors are a numerical dependency on the target body, the
literal "draft N+1 concurrent with the whole of verify N, with correct context"
is **impossible** — the NPU would consume precisely what the GPU is still
computing.

## 2. The overlap model the data flow permits — Grain B (stale context)

Three grains were considered; only one hides the full verify wall:

- **Grain A — post-body tail overlap.** The hidden rows exist as soon as verify
  N's *body forward* finishes, before lm_head + accept + rollback. The NPU could
  overlap only that tail, which is a small fraction of the ~100 ms (9B) /
  ~271 ms (27B) verify wall. Correct context, but hides almost nothing. Rejected.
- **Grain B — stale-context "tokens after next" (RECOMMENDED).** The NPU drafts
  N+1 concurrent with the **entire** verify N, using context that ends at block
  **N-1** (omits block N's newest target-hidden rows) and a **predicted seed**
  (predicted `bonus_token`). This is the only model that realizes the
  `step = max(draft, verify)` throughput the reality check assumes. It
  approximates the drafter's *input*, never its output — **losslessness holds
  because the target verifies every token regardless** — but τ (acceptance rate)
  drops by an unknown amount. **That τ cost is the load-bearing unknown of the
  whole phase.**
- **Grain C — speculative-all-accepted.** Needs verify N's body forward done, so
  it collapses toward Grain A's small window. Rejected.

**Grain B pipeline.** While the GPU verifies block N (body + lm_head + accept +
rollback), the NPU drafts block N+1 against `draft_scratch.target_hidden` as it
stood after block **N-1**, seeded by the predicted bonus. When verify N
finishes, compare predicted vs actual:

- **match** (predicted seed == real `bonus_token`, predicted append ==
  `rows_to_keep`) → the speculative N+1 draft is valid; feed it straight to
  verify N+1. Full overlap achieved.
- **mispredict** → discard the speculative N+1 draft, append verify N's real
  rows, re-draft N+1. That step falls back to serial.

**The seed-prediction oracle already exists and is instrumented.** Step 7b
(`:7469`–`:7495`) computes `TAIL_MATCH` (`drafted[b-1]`), `ANYPOS_MATCH`,
`REJ_MATCH` proxies for next cycle's `seed_token` and records them via
`record_seed_oracle`. Whoever built that was scoping this exact speculation. Its
measured hit-rate **bounds the achievable overlap fraction** and is cheap to
read — so it gates the phase before any concurrency is built (phase 2 below).

## 3. Rewind requirements — two domains, do not conflate

**(a) Target-side DeltaNet/KV rollback — already built, unchanged by the seam.**
`target_snap: DeltaNetSnapshot` (`:1037`, `save_from` `:7268`, `restore_to`
`:7602`) and `GdnTape` (`:2034`, replay `:7656`) restore target recurrent + KV
state to `accept_len+1` forwards after each verify. Orthogonal to the overlap —
the target runs its normal verify+rollback for block N regardless of NPU
speculation.

**(b) Draft-side speculative rewind — does NOT exist. This is the main new
correctness surface.** For Grain B, a mispredicted block-N outcome means
discarding speculative state that lives entirely in `DflashScratch`
(`dflash.rs:657`): `target_hidden` (+ `uploaded_target_hidden_rows` `:7576`),
`draft_ctx_cached_rows` (the incremental-projection watermark, `dflash.rs:1423`),
`target_hidden_abs_positions` (`:7583`). Rewind = watermark rollback + discard of
speculatively-appended/projected rows. Cheap, but **no snapshot API exists on
`DflashScratch`** — only `invalidate_draft_ctx_cache` / `reset_upload_tracking`
(`dflash.rs:1414`), which are full resets, not a one-block rewind. This is
net-new.

Crucially: the drafter carries **no recurrent state of its own** (it's a
cross-attention block over `target_hidden`), so there is no GDN-style recurrence
to snapshot on the draft side — only the append-only context buffers. That makes
draft-side rewind materially simpler than the target side.

**⚠ A draft-side rewind bug degrades τ, not correctness** (the target still
verifies every token), so it will **NOT** be caught by the md5 losslessness
gate. It needs a **τ-regression check**, not just the digest. Flagged.

## 4. Relationship to the ddtree tree variants — orthogonal

`spec_step_ddtree_batched` (`:10856`), `_path_c` (`:11508`), `_ddtree`
(`:10547`) widen a *single* block N into a tree of candidate continuations
(top-K per row, `:11542`). They raise τ *within* one draft→verify step; they do
not draft N+1 during verify N. The overlap seam operates on the
step-N→step-N+1 axis; the trees operate on the width of step N. **Build the seam
on the plain `spec_step_dflash` path first; compose with trees later** (a tree
draft can itself become the speculative N+1). `HiddenStateRingBuffer` is shared
infrastructure both reuse.

## 5. Concurrency mechanism

- **GPU** rides HIP streams (`gpu.active_stream`, `:6801`); verify/draft launches
  are stream-ordered async, with explicit `device_synchronize` only under
  `HIPFIRE_SPEC_PHASES=1` diagnostics.
- **NPU** is XRT-based in `hipfire-xdna` and today **strictly serial,
  single-threaded on the host** (`dispatch_synced` + `sync_output` block). The
  body lives only in the harness `dflash_body_native.rs`.

**What serializes them:** a single host thread would call the blocking NPU draft
and the async GPU verify sequentially; the host blocks inside NPU dispatch, so
they never run at once.

**Mechanism:** a **dedicated host thread drives the NPU** while the main thread
enqueues GPU verify on `gpu.active_stream` (non-blocking) and synchronizes only
at the acceptance point. The NPU worker drafts N+1 on the N-1 snapshot + predicted
seed and returns block N+1's tokens; a match/mispredict check gates use vs
discard. **Race safety:** the speculative draft reads context rows `[0..N-1]`
while the GPU scatter (`:7565`) writes the N rows *after* acceptance, into a
region the speculative draft deliberately did not touch — so no NPU-reads-GPU-
writes hazard during the overlap window, **provided the regions are proven
disjoint** under eviction/`compact_offset` shifts (`:6938`, `:7581`). Verify
disjointness or add a fence (see Risks). Async XRT submit+fence is an
alternative to the worker thread but a larger `hipfire-xdna` change — worker
thread is the smaller first step.

## 6. Hot-path invariant — consistent, prior art exists

Calling `hipfire-xdna` (HIP/NPU-direct XRT dispatch — not Python, not the
`dflash_convert` conversion tooling) from the spec loop is consistent with
AGENTS.md ("keep the inference path lean and HIP-direct"). **`hipfire-runtime`
already declares `hipfire-xdna` as an unconditional path dep
(`Cargo.toml:119`)** — no new crate edge. Keep the NPU path flag-gated so the
all-GPU spec path stays reproducible.

## 7. Integration points

- **Draft output seam (one substitution):** `spec_step_dflash` leaves the draft
  block hidden in `draft_scratch.x`, and `:7029`+ applies `target.lm_head` to
  rows `1..B` of it. Replace the GPU `draft_forward_opts` call (`:6979`) with the
  NPU body's forward, landing its output into `draft_scratch.x` at the same
  layout. Everything downstream (lm_head, argmax, accept, scatter, rollback) is
  unchanged — this is why losslessness is structurally safe.
- **NPU input feed:** the NPU needs `target_hidden` **host-side**. The default
  runtime path keeps it GPU-resident (`ctx_slice=None`, passes `None` to skip
  H2D, `:6969`). The seam needs a **new "GPU-resident context, host-mirror the
  new tail rows for the NPU" mode** — not the existing `ctx_slice=Some`
  diagnostic path, which forces full re-upload + contiguous-positions semantics
  (`:6941`). On Phoenix UMA a D2H is a cache flush (cheap) but non-zero; cost it
  and keep it flag-gated so the GPU-resident fast path never silently regresses.

## 8. Phased implementation — smallest-first, gate held throughout

Losslessness gate every phase:
`dflash_spec_demo --target $T --draft <D> --prompt "Explain how a four-stroke engine works." --max 96 2>/dev/null | md5sum`
→ AR-baseline == flag-off == flag-on, ≥3 repeats, assert digest ≠ `d41d8cd98f00`
first (the empty-string digest from a mis-invoked `--ar-baseline`). **The gate is
drafter-INDEPENDENCE, not a fixed digest** — the original `02e621bd56b5` target
is gone; the current rebuilt mq4 target commits `a099a2729d04…` (phase 1, task
#34). Anchor to whatever target you have; see the brief's Losslessness §.

**Phase 1 is DONE** (`0fa4a2972`): `HIPFIRE_DFLASH_NPU_DRAFT=1`,
`hipfire_xdna::DflashNpuBody` (`crates/hipfire-xdna/src/dflash_body.rs`),
tail-mirror feed on `DflashScratch.npu_target_hidden_host`. AR == flag-off ==
flag-on = `a099a2729d04…`, 3/3 each. Run with `--no-adaptive-b` so the
B=16/l_ctx=32-baked body engages every eligible cycle (else it falls back to the
GPU draft, losslessly).

1. **Reachability (serial, no overlap).** Wire the NPU body into
   `spec_step_dflash` behind a flag as a *serial* substitute for
   `draft_forward_opts`, feeding `target_hidden` via a new tail-mirror mode.
   Prove byte-identical `02e621bd56b5`. Validates plumbing + the `draft_scratch.x`
   seam + the host feed, zero concurrency. (`--ctx-slice 32` can smoke-test
   plumbing but do not quote its τ.)
2. ~~**Seed-oracle measurement — the cheap go/no-go.**~~ **DONE → NO-GO on
   phases 3–4** (task #35, `b55f6c638`,
   `benchmarks/results/dflash-seed-oracle-hitrate-20260720.md`). **TAIL_MATCH =
   2.0% (GPU drafter) / 1.0% (NPU drafter)** over the Phase F corpus — the
   Grain-B predicted seed (`drafted[b-1] == bonus_token`) is almost never right.
   Blended throughput at that rate: serial (phase 1, already built) 33.2 tok/s →
   overlap at TAIL 2% = **33.5 tok/s (+0.9%)**. The full `max(draft,verify)`
   promise (62.8) is unreachable because the mispredict rate is ~98%.

   **It is STRUCTURAL, not a drafter-quality issue** (GPU and NPU rates are
   statistically identical): `bonus_token` is by construction either the target's
   *correction at the rejection boundary* (the token the draft got wrong, ~70% of
   cycles) or one position beyond a full-accept block — neither sits at a
   predictable draft position. This **refutes the earlier "high-τ → high TAIL"
   guess**: the highest-τ prompt (merge_sort, τ=11.3, 9/13 full-accept) has
   TAIL=**0.000**, because high τ means *more* full-accept cycles = *worse* tail
   prediction. ANYPOS ceiling is 22–23% (only +12% even if a predictor could hit
   it).

3. ~~**Draft-side context snapshot/restore.**~~ **NOT PROCEEDING** — gated NO-GO
   by phase 2. Kept for the record: a one-block snapshot/rewind of `DflashScratch`
   (`target_hidden` watermark, `uploaded_target_hidden_rows`,
   `draft_ctx_cached_rows`, `target_hidden_abs_positions`) with a τ-regression
   check (the md5 gate cannot see a rewind bug). Only revisit if the seed
   predictor is redesigned (see below).
4. ~~**Concurrency via NPU worker thread.**~~ **NOT PROCEEDING** — gated NO-GO by
   phase 2. Would buy +0.9% over the already-shipped serial path.
5. **Compose** with ddtree width variants or async XRT dispatch — moot while 3–4
   are shelved.

**What could rescue an overlap (out of current scope):** a seed predictor NOT
built from the draft's own tokens — a dedicated bonus head, or top-K seed
hedging. Hedging blows the `max(draft,verify)` budget on Phoenix; a bonus head is
a drafter redesign. Neither is in scope; both are the only paths that move
TAIL_MATCH off ~2%.

**Landed instead: the serial NPU draft (phase 1) is the deliverable** — ~1.9×
GPU-only AR at τ≈5, structurally lossless, already shipped. The overlap was the
hoped-for multiplier on top; the go/no-go says it is not there on this axis.

## 9. Risks & open questions

- **τ under stale context is unmeasured and load-bearing.** If it collapses, the
  `max(draft, verify)` throughput never materializes and the 1.6–3.4× evaporates.
  Not determinable from code — phase 2 gives the seed-hit ceiling, phase 4 the
  real τ.
- **Draft-side rewind is net-new and invisible to the md5 gate** (bug → τ loss,
  not wrong tokens). Needs a τ-regression check. Main new correctness surface.
- **New per-cycle D2H for the host `target_hidden` feed.** The GPU-resident fast
  path avoids it (`:6960`). Cheap on UMA, non-zero — keep flag-gated.
- **6-hwctx / thermal.** The body pins ~3 of 6 hw contexts with
  `--cpu-primitives`; running it *continuously during* GPU verify (not
  interleaved) changes the thermal duty cycle and could shift GEMM bandwidth. All
  prior measurements were serial — re-validate under sustained overlap.
- **Cross-device race on `draft_scratch.target_hidden`.** Speculative draft reads
  `[0..N-1]`, GPU scatter writes the N rows post-accept. Disjointness under
  eviction/`compact_offset` (`:6938`, `:7581`) is **not confirmed from code** —
  verify or add a fence before relying on lock-free overlap.
- ~~**Async NPU dispatch depth unresolved.**~~ **RESOLVED (task #30,
  `990ef9fa0`): async XRT submit+fence DOES exist on npu1** —
  `NpuKernel::submit_synced` returns a timeline `seq`, `wait(seq)` fences on the
  syncobj, `sync_output` clears the pipelined read-back hazard. So a clean
  non-blocking submit is available and a worker thread is not required for
  NPU/CPU overlap. (Phase 4 is shelved regardless per the phase-2 no-go, but the
  mechanism question it raised is now answered.)

## Critical files
- `crates/hipfire-arch-qwen35/src/speculative.rs` — `spec_step_dflash` (:6756),
  verify/accept/scatter/rollback (:7292–:7666), seed oracle (:7469)
- `crates/hipfire-runtime/src/dflash.rs` — `draft_forward_opts` (:1326),
  `DflashScratch` (:657), `target_hidden` upload/cache (:1362–:1444)
- `crates/hipfire-xdna/examples/dflash_body_native.rs` — the NPU body to lift out
  of the harness into the runtime path
- `crates/hipfire-runtime/Cargo.toml:119` — existing `hipfire-xdna` dep
- `docs/plans/2026-07-19-dflash-phase0-brief.md` — measured state, gates, dead ends
