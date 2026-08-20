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

### P3 — `upload_raw` fixed-frame slabs — **re-scoped 2026-08-20, it is bigger than written**

The original entry called this a structural fix and implied it was localized. It
is not, and the measurement that says so is cheap: **`upload_raw` has ~757 call
sites**, and the asymmetry is a BORROW CONSTRAINT rather than an oversight.

    Gpu::upload_raw(&self, ...)          -> hip.malloc directly
    Gpu::upload_raw_pooled(&mut self,..) -> pool.alloc
    GpuPool::alloc(&mut self, ...)
    Gpu { pool: GpuPool }                 // held by value

A pooled upload cannot be offered behind `&self` because `GpuPool::alloc` needs
`&mut`. So unifying them means one of:

* change `upload_raw` to `&mut self` — touches ~757 call sites, many in arch
  crates that hold `&Gpu` deliberately;
* give `GpuPool` interior mutability — a `RefCell` is arguably redundant (the
  daemon threads `Gpu` as `&mut` anyway) and a `Mutex` puts a lock on an
  allocation hot path. Either changes the allocator's aliasing story.

Neither is a small change, and **rushing an allocator refactor is exactly how the
leak class this exists to prevent gets reintroduced.** Treat P3 as its own
project with its own plan, not as a queue item to be worked between others.

Also worth carrying into that plan: `GpuPool` buckets its free lists by
**power-of-two rounded size** (`free_lists: HashMap<usize, Vec<DeviceBuffer>>`),
so a 1.59 MiB routed expert occupies a 2 MiB bucket — ~26% waste. Fixed-frame
slabs would address that as well as the asymmetry, which strengthens the case for
doing it properly rather than by patching `upload_raw`.

**Landed in the meantime (zero risk, no behaviour change):** a warning on
`upload_raw` naming the hazard, the pairing that causes it, the two incidents it
has already caused, and the `pool_churn_upload_raw` bound. The danger the parent
plan identifies is *"any new caller reintroduces the leak"*, and a new caller
reads the doc comment — so this addresses the live risk without touching
allocation.

*Exit (unchanged, for whenever it is done properly):* `pool_churn_upload_raw`
still passes, and a caller using the plain `upload_raw` cannot strand memory —
demonstrated, not argued. Baseline: 200 unpooled cycles strand 400 MiB; 4000
pooled cycles strand nothing.

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

---

## P2 sequencing decision, 2026-08-20: **(b) — do it with M2b, not before**

The plan above posed the choice and declined to pick. Decided here, with the
evidence that settles it.

### Why (a) is worth less than "half the value" — it is worth ~nothing observable

Option (a) was "make steer per-stream now, leave the hand-path escape for M2b".
The escape reads (`qwen35/decode_layers.rs`):

> An active steer/capture session needs the per-layer block-boundary hook, which
> only the hand arms below carry — force the hand path.

So a steering stream **leaves the lowered march entirely**. Per-stream steer
state is therefore unobservable while the escape stands: two streams cannot
steer differently within one march, because a steering stream is not in the march
at all. (a) buys state hygiene with no behavioural change anyone can test —
which fails this plan's own bar that a stage exits on a measurement.

### Why (b) is cheaper than assumed — the hook already exists

The plan assumed wiring steer into the lowered path was the expensive half. It
is not, because the precedent is already live: `qwen35/lowered.rs:651` takes
`hidden_rb: Option<&HiddenStateRingBuffer>` and acts on it at the layer boundary
(`:690-696`), for spec-decode per-position hidden extraction. **The lowered
executor already accepts an optional per-layer boundary hook and fires it.**
`hidden_rb` used to force the hand path for exactly the reason steer does today,
and it was retired by giving the lowered path the hook rather than by keeping the
escape.

Steer needs the same shape, and the coupling is the argument for doing both at
once: the lowered hook must know *which stream's* spec to apply, which is
precisely the per-stream state (a) would have added blind. Doing them together
means the state has a consumer that proves it works.

### Consequences to accept before starting

* **This crosses two of this plan's stop lines** — it retires a fallback (the
  escape) and it touches the forward path for steering users. It is authorised
  as a decision, not as a licence to skip verification: the exit is
  **lowered-path steering matching hand-path steering**, asserted on output,
  before the escape is removed. Retire the escape only after parity holds.
* Non-steering numerics must not move at all. `HIPFIRE_FORWARD_LOWERED=0` stays
  as the general opt-out.
* `APPLY_CACHE` is a `thread_local` holding `GpuTensor` because it is `!Sync`.
  Per-stream state does not change that — it changes what the epoch invalidates
  against. Do not try to move it into the shared state.

### Found while deciding: a stale comment that says the opposite

`qwen35/lowered.rs:28` claims "hidden_rb engages only the hand path". Lines
651 and 690-696 of the same file are the lowered path doing exactly that
extraction, and `decode_layers.rs` says so too ("the lowered executor extracts
per-layer hidden itself now").

Worth fixing on the way past, and worth noting as a pattern: a stale comment in
this tree already caused one correct finding to be retracted wrongly during the
session that produced this plan. The parent plan opens by fixing two comments
that claim the lowered path is off by default. This is a third.
