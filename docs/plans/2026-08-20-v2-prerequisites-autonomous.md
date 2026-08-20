# Plan: the v2 prerequisite queue, worked autonomously

Companion to `2026-08-09-v2-daemon-module-major-multistream.md`. That doc scopes
the destination; this one is the working plan for the prerequisite queue, written
to be executed with limited supervision.

Status of the queue is tracked in the parent plan's "Status, 2026-08-20" section.
Done: **M0** (instrument existed; exit measured, 2 of 3 criteria pass and the
third is restated — see its exit record) and **`hipEventQuery`/`hipStreamQuery`**
(PR #258).

---

## Ground rules

These are not style preferences. Each one is a mistake made during the session
that produced this plan, and each cost real time.

1. **Check whether it already exists before building it.** M0 was already
   implemented (499 lines, wired end to end). The shared calibration library
   already existed (10,720 lines, larger than all five adapters). The pager fix
   already existed on another branch. All three were nearly rebuilt. Grep for the
   symbol, the concept, and the adjacent noun before writing a line.
2. **State what an instrument can see before concluding from it.**
   `tests/tiny-quant-baselines.txt` stores 8 decimals; `results.jsonl` rounds to
   ~6 significant figures. An "identical" reading may be neither identical nor a
   no-op. Two wrong conclusions in one session came from this.
3. **Rebuild explicitly before any A/B that depends on which commit built the
   binary.** `git bisect run` leaves `target/release/*` at an arbitrary commit. A
   whole real-model measurement was invalid because of this and looked *clean*.
4. **An impossibly clean agreement is a symptom.** Two different weight sets do
   not produce a bit-identical mean KLD over 1023 tokens. Investigate before
   reporting.
5. **Read gate output before committing.** `no-gpu-ci` returning 1 is not a
   formality; it was ignored once and caught a real violation.
6. **One PR per concern**, each stating its verification AND its coverage gaps.
   `tiny-affected-gate` answering "no tiny coverage selected" is a result to
   report, not a green tick.
7. **Do not widen scope to fix what the change does not require.** File it.

## Stop and ask

Do not proceed autonomously past any of these:

- a change that would alter **serving numerics** (logits, sampled tokens, KLD);
- **deleting a fallback** or an escape hatch;
- a decision needing a measurement **this box cannot run** (65 GB free; the
  smallest calibration-capable real MoE needs ~170 GB);
- anything that would **rewrite published history** or push to `master` directly.

---

## The queue

### P1 — `load_progress::SINK` (do this first)

`static SINK: Mutex<Option<Box<ProgressFn>>>` (`hipfire-runtime/src/load_progress.rs:30`),
three public functions, and `report()` called from **six arch loaders**
(deepseek4, qwen2, qwen35, lfm2moe, minimax, plus `hfq.rs`).

**Why first: it is the tractable one, and it is genuinely per-request rather than
per-token.** Model load is a whole-frame operation — it does not interleave with
decode the way steering does — so the fix is to carry a sink handle on the load
path rather than to thread state through the forward. Mechanical, wide, and
independently revertible.

*Exit:* two concurrent loads report progress to their own callers with no
crosstalk, asserted in a test rather than by inspection. Existing single-load
progress output is unchanged.

*Scope note:* six call sites is wide but shallow. If it grows a seventh kind of
caller, stop and re-scope — that would mean `report()` is being used for
something other than load progress.

### P2 — `hipfire_steer` — **and it is NOT a sibling of P1**

Four pieces of state: `SESSION: OnceLock<RwLock<Session>>`, `ACTIVE: AtomicBool`
(hot-path gate), `EPOCH: AtomicU64` (invalidates the cache below), and
`APPLY_CACHE`, a `thread_local RefCell<Option<ApplyCache>>` that exists precisely
because `GpuTensor` is `!Sync` and cannot live in the `Sync` static.

Eighteen public functions, but only **two real forward-path call sites**:
`qwen35/decode_layers.rs` (`maybe_steer_block`) and `qwen35/prefill_chunk.rs`
(`maybe_steer_block_batched`). The rest of the greps are documentation.

**Why it is harder than it looks, and why it is not P1:** the parent plan calls
steer "the hard one", and the tell is that **`is_active()` currently forces the
hand path** — it is one of the four escapes M2b has to retire. So making steer
per-stream does not, by itself, make steer survive the substrate: a correct
per-stream steer that still forces the hand path has moved the state without
removing the escape.

That means P2 has a **sequencing question that must be answered before coding**:
either (a) make it per-stream now and leave the hand-path escape for M2b, which
is honest but delivers half the value; or (b) treat it as part of M2b and do both
together, which is larger but coherent. **This plan does not pick.** Decide it
with the parent plan's M2b in view, and record the choice.

*Exit (for (a)):* two streams with different steer specs decode in one batched
step and each gets its own steering, asserted on output. `APPLY_CACHE`'s epoch
invalidation still holds per stream.

*Do not start P2 until P1 has landed* — they touch different subsystems, and
bundling them makes the steer sequencing question harder to see.

### P3 — `upload_raw` fixed-frame slabs

The API asymmetry survives even though the paged call sites are fixed:
`Gpu::upload_raw` calls `hip.malloc` directly (`dispatch/mod.rs:2169`) while
`upload_raw_pooled` (`:2212`) uses the pool, and both free through
`free_tensor` into `GpuPool`. Any **new** caller reintroduces the leak.

The parent plan's structural fix is fixed-frame slabs — `ExpertShape` already
establishes that every routed expert in a layer is the same size, so admission
becomes `free.pop()` and eviction `free.push()`.

*Exit:* the existing M1a example (`pool_churn_upload_raw`) still passes, and a
new caller using the plain `upload_raw` cannot strand memory — demonstrated, not
argued. Current baseline for comparison: 200 unpooled cycles strand 400 MiB;
4000 pooled cycles strand nothing.

### Not in this queue

**Prefill lowering.** ~14.2k lines across `prefill_chunk.rs` and
`prefill_batch.rs`, zero `SuperOp` references. It is the real blocker for the
latency claim and it is a project with its own plan, not autonomous work.

**The M0 sampler-vs-emitter question.** `TokenEmitted` counts emitted text
frames, so a step producing no printable text folds its duration into the next
gap. Fixing it changes what the timestamp means and wants two events rather than
a move — a design choice, deferred to whoever needs per-step attribution (M3 is
the first plausible consumer).

---

## Working rhythm

One branch and one PR per queue item, off current `master`, verified with
`./tests/no-gpu-ci.sh` and — for runtime changes —
`./tests/tiny-affected-gate.sh --require-coverage`, reporting honestly when it
answers "no tiny coverage selected for changed paths" rather than letting a green
tick imply more than it did.

Between items, re-read the parent plan's status section: three Tier-1 blockers
cleared themselves through other people's work during a single session, and the
cheapest way to do unnecessary work here is to not notice that again.
