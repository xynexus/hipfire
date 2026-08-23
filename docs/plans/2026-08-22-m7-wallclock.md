# §M7 wall-clock — measured, and the crossover cannot exist yet

Status: measured 2026-08-22, nix1 / gfx1103, `Qwen3.6-35B-A3B--oq4`, kvarn KV.
Companion to `2026-08-22-m7-amortization-measured.md`, which settled the
weight-byte half.

Unblocked by the executor fix in `m3-drain-before-teardown`: before it, the v2
executor lost every admitted stream when a teardown frame shared the batch, so
no width measurement was possible at all.

## The scaling curve

N concurrent sessions, 16 tokens each, greedy. Decode span is measured
externally from first token to last, so model load is excluded.

| N | v2 aggregate | inline aggregate | v2 advantage | v2 per-stream |
|---|---|---|---|---|
| 1 | 24.01 tok/s | 24.10 tok/s | 1.00× | 24.01 |
| 4 | 22.58 | — | — | 5.64 |
| 16 | 22.43 | 10.86 | **2.07×** | 1.40 |
| 32 | 18.42 | 7.89 | **2.33×** | 0.58 |

Two facts, and the second is the one that matters.

**1. The v2 executor is worth having.** At width it is 2.07–2.33× the inline
path. The inline path *collapses* under concurrency (24.1 → 10.9 → 7.9); v2
holds roughly flat (24.0 → 22.4 → 18.4). That is a real result and it is what
the march loop buys.

**2. Aggregate throughput does not GROW with N under either path.** Sixteen
concurrent streams produce no more tokens per second than one. Concurrency is
being time-sliced, not exploited.

## CORRECTION — the crossover exists, and here it is

The section below concluded that no crossover was measurable because nothing
coalesces across streams at decode. **That was wrong, and the error was mine:**
I checked the callers of `select_qwen35_decode_batch_backend` and found only
tests, then generalised. I did not check
`run_generate_batch_decode_step_qwen35`, which **is** called — by
`batch_executor.rs:160`, reachable from the daemon as the
`generate_batch_decode_step` message.

Driving that path directly (prefill the batch, then step it N-wide):

| N | batched | v2 round-robin | inline | batched ÷ v2 |
|---|---|---|---|---|
| 1 | — | 24.01 | 24.10 | — |
| 4 | **33.96** | 22.58 | — | 1.50× |
| 16 | **50.67** | 22.43 | 10.86 | **2.26×** |
| 32 | **55.02** | 18.42 | 7.89 | **2.99×** |

**Batched decode scales with N; round-robin does not.** 33.96 → 50.67 → 55.02
against 22.58 → 22.43 → 18.42. Against solo decode (24.0 tok/s) the batch is
1.41× at N=4, 2.11× at N=16, 2.29× at N=32.

So §M7's capacity thesis holds at wall-clock, not just in weight bytes, and the
crossover against single-stream is **between N=1 and N=4**. The falsification
condition — no crossover below the N whose KV fits VRAM (~129) — does not fire.

**The real gap is that the v2 executor does not use this path.** Both mechanisms
exist; the march loop round-robins one stream per quantum while the batched
entry point sits behind a different message type. Wiring `march_streams` to
dispatch a batched step is worth 2.26× at N=16 and 2.99× at N=32 on these
numbers — and it is scheduling work, not kernel work.

The section below is kept for the reasoning it records, but its conclusion is
superseded by this measurement.

## Why there is no crossover to find

§M7's thesis is that module-major execution beats layer-major past some N,
because distinct experts touched grows sublinearly — measured at 1.74× sharing
at N=16 and 6.59× at N=128 (`2026-08-22-m7-amortization-measured.md`).

**Realising that requires one forward pass to serve N streams' tokens at once.**
Neither path does. The march loop steps one stream per quantum through the
single resident session slot, park/resume between quanta: N sequential forwards,
each amortising nothing across streams. The available sharing is never
collected.

So the crossover is not "not yet reached" — it is **not measurable**, because no
execution mode on the decode path coalesces across streams.

## Integrating it: what the remaining step actually is

Wiring `march_streams` to the batched entry is **not** a plumbing job, and the
reason is worth stating so nobody starts it expecting one.

The session-residency half IS small. Park installs into `m.active`
(`session.rs:341`), batched decode requires every session in
`m.q35_registry.sessions` (`qwen35_decode.rs:489`), and `activate_session`
already moves between them — so the executor need only evict the active slot
before a batched step.

**The generation-state half is the real work.** `Qwen35Generation` exposes only
`step` / `fail` / `park` / `resume` / `should_continue` / `finish`. It privately
owns the sampler RNG, the emitted-token count, the stop-sequence buffers and the
per-stream `max_tokens`. `GenerateBatchDecodeEnvelope` is a *second, independent*
generation state machine over the same session, carrying its own
`max_tokens_remaining` and `logical_position` and doing its own sampling and
stop detection.

Driving a session through the batch path behind its handle therefore desyncs
every one of those. Two ways out:

- **(a) Make the handle authoritative.** Add a serving-core entry that steps N
  `Qwen35Generation`s through one batched forward — `qwen35_step_batch(&mut
  [Qwen35Generation], ..)` — so per-stream RNG, stop sequences and counters stay
  where they already live. This is the correct shape and it is a new API, not a
  call.
- **(b) Abandon the handle for batched streams** and re-implement stop and
  sampling bookkeeping against the envelope. Cheaper to write, and it duplicates
  the exact state machine that #287's greedy-nondeterminism work went through
  once already.

(a) is right. Note also that the two paths sample independently, so they agree
under greedy and will diverge at `temperature > 0` — any parity check between
them must be greedy, or compare distributions rather than tokens.

### The seam that makes (a) concrete

Reading `qwen35_decode_step_fused_grouped_moe_native_chunks`
(`qwen35_decode.rs:1040`), the batched step is already **sample-then-forward**:

1. sample an outcome per session from that session's *existing* `state.logits`
   (`qwen35_decode_token_outcome`),
2. push each chosen token into its conversation,
3. build one `DensePrefillSessionBatchRow` per session carrying that single
   token,
4. call `forward_prefill_grouped_moe_session_batch` — the same fused batched
   forward prefill uses — which writes fresh logits back into each
   `state.logits`.

Sampling and the forward are therefore **already separate**, and the residency
eviction is already handled (`qwen35_save_active_session` at the top of the
step).

So `qwen35_step_batch` is not new machinery. It is a split of
`Qwen35Generation::step` into its sample half and its forward half — the same
shape as the stage split in `run_moe_decode` — with a driver that:

- lets each handle sample its own next token from its own session's logits,
  keeping per-stream RNG, stop sequences and counters exactly where they are;
- collects the N chosen tokens into rows and issues **one** batched forward;
- lets each handle update its own counters from its own token.

That is what makes the handle authoritative while still getting the single fused
forward, and it is why (a) costs a refactor of `step` rather than a
reimplementation of the generation state machine.

### The two sampling sites, resolved — the common path is uniform

The complication recorded below turns out to be smaller than it reads, and this
is the analysis it asked for.

The two `sampler::sample` sites are **not** two alternative branches of one
step. `qwen35_decode_one` is:

```
forward -> logits -> sample                       (the normal path, :2312)
```

and separately, an env-gated feature:

```
if budget alert fires (:2170, needs HIPFIRE_EXPERIMENTAL_BUDGET_ALERT
                        + budget_alert_at_tok + non-empty alert text):
    sample                                        (:2218)
    encode nudge text, push its tokens            (:2230)
    forward_scratch EACH nudge token              (:2260, in a loop)
```

So a single `decode_one` is one forward and one sample **except** when the
budget-alert nudge fires, in which case it is sample → N extra single-token
forwards → sample. The alert fires at most once per generation (`st.alert_fired`
latches) and only while inside an open `<think>` block.

**What that means for the batched step.** The common path — every stream, every
token, unless the feature is enabled and its threshold is crossed — is exactly
`forward -> sample`, which batches in lockstep with no special handling. A
stream whose nudge fires needs N extra forwards that the other streams do not,
so it cannot stay in the round.

That is a clean, testable condition, not a structural obstacle: the batched
driver excludes a stream that is about to nudge and steps it solo that round,
exactly as it would for any stream whose shape diverges. `st.alert_fired` and
the threshold are already the predicate.

So the `step` split is closer to the `run_moe_decode` extraction than the
earlier note feared. What it is not is unconditional, and pretending the second
site does not exist would silently drop the nudge forwards from any stream that
fires one.

### One complication, found by reading rather than assuming

The split is clean on the *batched* side but not on the handle side.
`qwen35_decode_one` (`generate.rs:1981`, ~365 lines) has **two** `sampler::sample`
call sites, in different branches — around `:2219` and `:2330` — not one sampling
point with a forward either side.

So "split `step` into a sample half and a forward half" is not a single cut. It
needs the branch structure resolved first: which arms sample before their
forward, which after, and whether both can be expressed as
`forward -> logits -> sample` without changing either arm's behaviour. That is
the actual first task, and it is a reading task before it is an editing one.

This does not change the design — the handle must stay authoritative, and the
batched forward is still `forward_prefill_grouped_moe_session_batch` over N
single-token rows. It changes the estimate: the `step` split is not the
mechanical extraction the `run_moe_decode` split was.

## Available today, without that work

The batch protocol is reachable now: `generate_batch_prefill` →
`generate_batch_decode_step` → `release_sessions`. A client that wants
multi-stream throughput can have the 2.26× at N=16 today by using it instead of
N independent `generate` calls. The executor integration is what makes plain
`generate` requests benefit automatically.

## What is missing is wired-but-uncalled

The batched decode machinery exists. `Qwen35DecodeBatchBackend` carries
`FusedDenseLayerChunked` and `FusedGroupedMoeLayerChunked`, and
`select_qwen35_decode_batch_backend` chooses between them by arch and session
count (`hipfire-generate/src/lib.rs:1476-1523`).

**Nothing in production calls it.** Outside its own unit tests and the daemon's
`generate_batch_prefill_tests.rs`, there is no consumer — the selector is tested
and unreachable. Prefill has its fused multi-session path; decode has the
backends and no caller.

## The next step, precisely

Make `march_streams` dispatch **one batched step across all runnable streams**
rather than one stream per quantum, routing through
`select_qwen35_decode_batch_backend`. That is where §M7's 1.74×–6.59× would
finally be realised, and only then does the crossover become a measurable
quantity.

Note the shape of that change: it does **not** need the per-slot MoE split. Per
`2026-08-22-m4-premise-falsified.md`, per-slot dispatch costs +39% launch count
in a workload measured at >99% launch-bound. Cross-stream batching is the
opposite move — it *reduces* launches per token by serving N streams in one
pass. Those two directions were conflated in §M4's framing; only the second one
serves M7.
