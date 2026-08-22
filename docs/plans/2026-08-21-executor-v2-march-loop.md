# Plan: executor v2 — admission, the march loop, and suspension

Implementation plan for the parent plan's §M3
(`2026-08-09-v2-daemon-module-major-multistream.md`). That section scopes the
destination in one paragraph and three measurements; this is how.

Written 2026-08-21 against `master` at `b63d14a75`.

## Why now

Two things are finished and waiting on this, which is the argument for doing it
next rather than anything else in the queue:

* **`RunningStream` / `StreamTable` are scaffolding with no driver.** They carry a
  `#![allow(dead_code)]` whose removal condition is literally "when the executor
  lands".
* **P2/M3 (per-stream steering) is blocked on it.** The steer subsystem is
  finished and keyed end to end, and it cannot be reached: nothing admits a
  stream, so `session_id` resolves against an empty table.

## Two corrections to the parent plan, found before writing code

**`batch_runner.rs` is not in the daemon.** §M3 says "`batch_runner.rs` is
deleted, not moved" alongside `transport.rs` and `queue.rs`, which reads as a
daemon file. It is not: `crates/hipfire-daemon/src/batch_runner.rs` does not
exist. The file is `crates/hipfire-server/src/batch_runner.rs`, in a different
crate, with its own callers (`routes/health.rs`, `routes/train.rs`). **Deleting it
is not part of this work** and must not be folded in on the strength of that
sentence.

**A stream identity already exists in the trace.** `exec_trace.rs` has
`pub fn stream_id_of(id: &str) -> u32`, which derives a trace stream from a
request-id string. So the M0 instrument already partitions by stream, and its
notion must be reconciled with `SessionKey` — the exit measurements are read off
that trace, so if the two disagree the numbers partition by the wrong thing.
Reconcile explicitly; do not assume they agree.

## Scope

**In:** admission, the march loop, the suspension boundary, and the
`RunningStream` lifecycle wiring. **Out:** the module split (§M4), residency
(§M5), realtime classes (§M6), and prefill lowering. The executor may march
whatever quanta exist today; making the quanta finer is a later milestone.

## Staging

Each stage lands behind `HIPFIRE_DAEMON_EXECUTOR=v2`, default off. That is what
keeps every stage revertible, and the parent plan already names the flag.

### M3a — admit without executing

`PendingQueue` keeps its per-connection FIFO invariant and narrows its job: a
`Generate` frame **admits** a `RunningStream` into a `StreamTable` on
`DaemonState` and returns, instead of running the forward inline. With the flag
off, admission is followed immediately by running that stream to completion — so
behaviour is unchanged and only the *shape* moves.

*Exit:* with the flag off, a greedy generation is byte-identical to today, and
`StreamTable::len()` returns to zero after each request (no leaked streams).
`transport.rs` is untouched — if this stage edits it, the seam is wrong.

*Falsified by:* any token difference with the flag off.

### M3b — the march loop

**Rewritten 2026-08-21 after M3a landed.** The original text is preserved at the
end of this section, because what it got wrong is instructive: it costed the
*loop* and never costed the *quantum*, and the quantum is the whole job.

A serial loop that picks a runnable stream, advances it one quantum, records the
cursor, and repeats. Serial is deliberate: the parallelism is intra-kernel, and
per-stream state exists because a serial executor interleaves streams *within* a
march.

#### M3b0 — a quantum has to exist first

**There is no quantum on the `Generate` path today.** `handlers::generate::text`
ends in a single `generate(...)` call, and the qwen35 per-token loop lives
*inside that function body* — `crates/hipfire-serving-core/src/generate.rs:3124`,
`while generated < max_tokens {`, ~365 lines — with every cross-token variable a
stack local (`rng_state`, `filter`, `think_count`, `loop_guard`, and ~10 more).
`crates/hipfire-runtime/src/arch.rs:1192` is a second such loop with ~16 more,
two of them RAII GPU scratch that returns to the pool when the frame drops.

Neither is re-enterable. A march loop built today would march **whole requests**,
which is indistinguishable from not having one — the trap `stream.rs` already
warns about ("a green run under the flag is not evidence the executor ran").

So M3b0 is: extract the loop body into a state struct plus a one-token function,
and leave the existing loop calling it. One struct, one function, exactly one
caller. No flag, no march loop, no `StreamTable` change.

*Exit:* a pure extraction, byte-identical by construction, with
`./tests/tiny-affected-gate.sh --require-coverage` green and a real generation
compared byte-for-byte. **Inert on both paths**, not merely flag-off — which is a
stronger safety argument than the flag, and is needed here because this is
flag-off code.

*Scope:* qwen35 (pp==1) only. That is where the exit can be demonstrated at all
(see the arch scope below), and it is graph-free by a hard literal —
`crates/hipfire-arch-qwen35/src/qwen35/mod.rs:1000` is `let use_graph = false;`.
Doing `arch.rs:1192` as well doubles the diff and buys nothing M3b can test.

#### M3b0.5 — the quantum has to ESCAPE `generate()` first

**Added 2026-08-21, after M3b0 landed and this section's estimate was checked
against the tree.** M3b1 below said "small once M3b0 exists". That is true of the
*loop*, and it was the wrong thing to cost.

M3b0 made advancing a token a call. It did **not** make that call reachable from
the daemon:

* `qwen35_decode_one`, `Qwen35DecodeState`, `Qwen35DecodeCfg` and `Qwen35Step`
  are all **private** to `serving-core/src/generate.rs` — no `pub`.
* The daemon makes **one** call, to `generate(...)`
  (`handlers/generate.rs`). Everything else is inside it.
* The qwen35 arm begins at `generate.rs:3304` and the decode loop at `:3590`, so
  roughly **286 lines of per-request setup** run before the first quantum is even
  reachable: the prefill (`forward_prefill_batch`, and the multi/PFlash-compressed
  variants), the ngram scope, and the `tok0` sample. Teardown follows the loop:
  the `\n` trailer after `<|im_end|>`, the `done` frame with its timing
  arithmetic, and `qwen35_restore_or_error` / `restore_into_loaded` — which
  **consumes** the session.

So a march loop cannot drive anything until `serving-core` exposes a resumable
generation: `start()` (that setup, returning a handle) → `step()` (M3b0's
`qwen35_decode_one`) → `finish()` (that teardown). That is a hot-path refactor on
the order of M3b0 itself, not a by-product of the loop.

**M3b0 did get the load-bearing half right:** `Qwen35DecodeState` owns no borrow
of the GPU or the model, which is exactly what lets it live across a suspension
while the executor hands the device to another stream. The remaining work is
moving the setup and teardown to the same footing, and making the four items
`pub` (or wrapping them in one `pub` handle).

*Exit:* the daemon can start a generation, advance it one token, and finish it,
with output byte-identical to today — verified the same way M3b0 was, on a real
qwen35 artifact with at least one sampled run.

**Concrete boundaries, measured** (`serving-core/src/generate.rs`, post-M3b0):

| phase | lines | what it owns |
|---|---|---|
| `start()` | 3304 → 3590 | prefill (`forward_prefill_batch` + the multi / PFlash-compressed variants), the multi-turn auto-reset, ngram scope, `tok0` sample, `t_prefill` |
| `step()` | — | **done**: `qwen35_decode_one` |
| `finish()` | 3623 → 3716 | the `\n` trailer after `<|im_end|>`, the timing arithmetic, evidence + MoE-router histogram writes, the `done` frame, `qwen35_restore_or_error` |

**Do `finish()` first, and take `session` BY VALUE.** That is not a style
preference — it dissolves the constraint that shaped M3b0. M3b0's
`Qwen35Step::Failed(String)` exists only because `qwen35_restore_or_error`
consumes the session while `kv`/`dn` are `&mut` borrows *out of* it, so the
unwind could not live inside the extracted function. A `finish` that **owns**
`session` can derive `kv`/`dn` from it internally and call the restore directly —
no `Failed` hand-back, no borrow gymnastics.

What `finish()` must close over, all verified present: `prefill_tokens`
(`= new_tokens.len()`, :3301), `t0`/`t_prefill`, `nl`, `im_end_token`,
`evidence_dir: Option<&str>`, `DaemonMoeRouterHistogramGuard` (:3343), and the
PFlash triple — `pflash_summary: Option<CompressedPrompt>` (:2978),
`pflash_bypass_reason: Option<String>` (:2982), `pflash_alpha: Option<f32>`
(:2985), which the `done` frame needs via `pflash_done_fragment`.

**M3b0.5 does not decompose the way M3b0 did.** The handle's field set is
determined by what `finish()` closes over, so "extract the phases" and "define
the handle" are one change, not two. Budget it as one ~300-line pass with the
M3b0 verification method attached, rather than expecting to land it in slices.

#### M3b0.75 — make the quantum REACHABLE from the daemon

**Added 2026-08-22, measured after M3b0.5 landed.** M3b0.5 gave the qwen35 arm a
`step()` and a `finish()` and bundled them into `Qwen35Generation`. The daemon
still cannot drive any of it, for two reasons — and the second is the real work.

**1. Everything is private.** `Qwen35Generation`, `Qwen35DecodeState`,
`Qwen35DecodeCfg`, `Qwen35Step`, `qwen35_decode_one` and
`qwen35_finish_generation` are all module-private in
`serving-core/src/generate.rs`. Trivial to change; do it last, so the visibility
matches what the daemon actually needs rather than exporting the whole set.

**2. `start()` does not exist.** Current boundaries in `generate.rs`:

| | line | what |
|---|---|---|
| qwen35 arm opens | 3580 | `if is_qwen35_family_arch_id(m.arch_id) {` |
| **setup** | 3580→3827 | **~248 lines: this is the work** |
| handle constructed | 3828 | `let mut generation = Qwen35Generation {` |
| loop | 3872 | `while generation.st.generated < generation.cfg.max_tokens` |
| teardown | 3885 | `generation.finish(...)` — done |

That setup is not a cosmetic slice. It contains the prefill
(`forward_prefill_batch` plus the multi and PFlash-compressed variants), the
multi-turn auto-reset, the ngram scope, the `tok0` sample, and **several early
`return`s** on error. It also reads a large fraction of `generate()`'s 29
parameters, so a free-function `start()` inherits a parameter list on the order
of `qwen35_finish_generation`'s sixteen.

*Exit:* the daemon can call `start()` → `step()` → `finish()` and get output
byte-identical to calling `generate()`. Verify the M3b0.5 way — real qwen35
artifact, at least one sampled run, plus the 3-turn trailer and budget-alert
harnesses, all compared against master **as it stands at the time**, not against
older recorded hashes (two of those harnesses contain session-less generates,
whose values moved when #288 landed).

**Do not slice this one either.** Same reason as M3b0.5: the parameter list is
determined by what the setup reads, so "extract it" and "decide its signature"
are one change. Budget it as a single pass with the verification attached.

##### The boundary above is WRONG. Measured 2026-08-22 by attempting it.

Extracting `3580→3869` into a `Qwen35Generation::start(m, gpu, stdout, id)` was
tried and reverted. The compiler enumerated **26 further identifiers** the block
reads from `generate()`'s scope, making `start()` a **30-parameter function** —
larger than the 29-parameter `generate()` it came out of. That alone would only
be ugly. What makes the boundary wrong is *which* identifiers:

`new_tokens` (:1344), `nl` (:1250), `im_end_token` (:162), `think_pair` (:1443)
and `tool_call_pair` (:1436) are products of `generate()`'s **shared framing
prologue** — chat-template rendering, tokenisation, the PFlash decision — which
runs long before the qwen35 arm and is shared with every other arch.

**The daemon has none of them.** So a `start()` cut at the arm is not merely
awkward, it is *uncallable by the daemon*, which is the only reason to build it.

So M3b0.75 is not "extract the arm's setup". It is **split `generate()` itself**:
everything up to the loop becomes `start()`, the loop stays at the call site, and
`finish()` already exists. The framing prologue goes with `start()` because that
is what makes the handle constructible from a request rather than from
`generate()`'s locals.

Two consequences for whoever does it:

* The natural signature is close to `generate()`'s own — which argues for taking
  the request (or a small bundle) rather than 30 positional parameters. The file
  already has the pattern: `Qwen35DecodeCfg` bundles 17 for the loop.
* The other arches share that prologue. Splitting it must not fork their path —
  the llama arm and the early-return spec-decode routes all run through it.

The full-parameter list, for costing: `budget_alert_at_tok`, `budget_alert_text`,
`evidence_dir`, `frequency_penalty`, `im_end_token`, `max_think_tokens`,
`max_tokens`, `new_tokens`, `nl`, `pflash_alpha`, `pflash_bypass_reason`,
`pflash_summary`, `prefill_already_done`, `prefilled_prompt_tokens`,
`prefill_tokens`, `presence_penalty`, `q35_session`, `repeat_penalty`,
`repeat_window`, `request_stop_sequences`, `t0`, `temp`, `think_pair`,
`tool_call_pair`, `top_k`, `top_p`.

#### M3b1 — the loop itself

Small **once M3b0.75 exists**: ~150–250 lines across `main.rs`, `stream.rs`,
`state.rs`. Four things must move with it, none of them optional and none in the
`Generate` arm:

1. **Reply routing.** `Responder` is one per process, and `out.sink` is swapped
   per *frame*. `RunningStream` gains a `ReplySink` clone plus its `request_id`,
   installed before each quantum. `ReplySink` is already `Clone` over an
   `Arc<Mutex<..>>` and `pub(crate)`, so **this needs no `transport.rs` edit** —
   but note `RunningStream` and `StreamTable` both `#[derive(Debug)]` and
   `ReplySink` does not, which tempts a derive *in* `transport.rs`. Hand-write or
   drop the stream-side `Debug` instead. That is the stop-line, reached by
   accident.
2. **The blocking `recv`.** With a live marching stream and an empty queue, the
   executor parks forever instead of marching. It must block only when the queue
   is empty **and** nothing is runnable.
3. **The per-frame cancel resets.** Under a march loop one frame is not one
   stream, so dispatching any frame wipes a live stream's unpolled abort.
4. **`activate_session` per quantum**, not per frame — otherwise a stream switch
   drives stream B's tokens into stream A's KV.
5. **Re-check M3b0's per-step snapshots.** `qwen35_decode_one` takes
   `m.eviction`, `m.physical_cap` and `nl.len()` as values captured once per
   step, where the original re-read them at four separate points deep in the
   body. That is provably equivalent *today* — and specifically because the
   borrow checker forbids it from being otherwise: `weights`/`config`/`scratch`
   are shared reborrows out of `*m` held across the whole loop, so nothing can
   take `&mut LoadedModel` mid-step.

   **M3b1 removes exactly that guarantee.** The moment the march loop can hand
   `&mut LoadedModel` to another stream between quanta, a per-step snapshot and
   a per-use re-read stop agreeing — and they diverge *silently*: a stale
   `need_kv <= physical_cap` suppresses or emits a budget-alert nudge, changing
   the token stream with no error frame. Either re-read them per use, or assert
   the model cannot be swapped mid-quantum. Do not let this be rediscovered as a
   token-stream mystery.

*Exit:* two concurrent `Generate` requests interleave at quantum granularity
under the flag, each producing output byte-identical to running it alone.

*Falsified by:* either stream's output differing from its solo run. That is the
accept-and-miscompute class; only exact comparison catches it.

#### The exit needs an arch scope and a harness decision

**Arch scope: qwen35 (pp==1) and lfm2-moe only.** Interleaving requires
`activate_session` per switch, and `handlers/generate.rs:249-262` gates it to
`(is_qwen35_family_arch_id(m.arch_id) && m.pp == 1) || is_lfm2_generate_session`;
every other arch errors on `session_id` and holds one unkeyed KV. Naming the
scope is the difference between a scoped milestone and a fixture that silently
corrupts KV.

**Harness: DECIDED 2026-08-22 — one connection, two pipelined `generate` frames
with distinct `session_id`s.**

The transport already supports it and needs no edit: `spawn_stdin_reader` puts
the read loop on its own thread feeding a 256-deep `sync_channel`, and the
executor drains with `try_recv`, so two frames written back to back are both
pending. `queue.rs`'s dependency invariant is about **mixed** frame types — a
pipelined `generate` + `unload` running out of order — which a two-`generate`
harness simply does not create.

Distinct `session_id`s are mandatory, not hygiene: with the same id the second
stream is refused by `AdmitError::SessionAlreadyLive` and the test silently
measures one stream twice. Since #288 they are also what makes each stream a
real conversation rather than a one-shot, and since the typed request gained a
`session_id` field a client can actually send them.

The two-client route via `--listen` stays unbuilt and unneeded for this exit.

The original wording is kept below because the constraint it names is real; only
the conclusion changed.

> Two concurrent `Generate`s on ONE connection break the invariant `queue.rs`
> exists to protect —
> its own words are "a connection's frames are a dependency chain… the order *is*
> the declaration", which holds today only because dispatch order equals execution
> order. Two connections need a listening daemon, and **nothing in the tree starts
> one**: `--listen` appears only in `main.rs` (help text, parser, dispatch); grep
> over `scripts/` and `tests/` returns zero, and `daemon-adapter`'s
> `attach_or_spawn` always falls through to a stdio spawn.

The harness must also use **distinct `session_id`s**, or
`AdmitError::SessionAlreadyLive` refuses the second stream and the test silently
measures one stream twice. And once the march loop owns execution, that refusal
stops being swallowable: M3a's `admit_generate` returns `None` and the caller
runs the frame anyway, which is right for a shape-only stage but leaves a refused
stream with no runner. Decide the policy — error frame, or queue behind the live
stream — in M3b1.

#### Cancellation does NOT consolidate here

The original text said "cancellation moves here — from three per-token hook sites
into the pick step, one site instead of three-and-counting." **There are six
sites across two different primitives**, and the consolidation is not reachable
in M3b:

* id-keyed `cancel::is_cancelled`: `arch.rs:1206`,
  `serving-core/generate_arch.rs:1073`, `serving-core/events.rs:174`.
* unkeyed `take_generation_cancel()`: `arch.rs:1215`,
  `serving-core/generate.rs:1557`, `serving-core/generate.rs:3132`.

`GENERATION_CANCEL` is a process-global `AtomicBool` and its take is a consuming
`swap(false)`, so with two streams marching, whichever polls first eats the
other's cancel. Its own doc justifies that with "a cancel never leaks into the
next request" — premised on one request at a time, the exact premise M3b removes.

And **the pick step does not hold the key.** `cancel::PENDING` is keyed on the
*request* id; `RunningStream` carries a `SessionKey`, which resolves to the
client's `session_id` whenever one is sent. A pick-step check on the session key
would silently never match for exactly the clients that set it.

Two served decode loops also have **zero** cancellation coverage today —
`serving-core/generate_vl.rs` (qwen35-VL and dots_ocr) contains no occurrence of
"cancel" at all. Consolidating would either newly cancel paths that never could
be — a behaviour change under the flag, which this plan forbids — or leave them
uncovered.

**M3b adds a pick-step check and leaves the per-token ones.** Removing them needs
every decode path routed through the march loop, which is M3c at the earliest.

<details><summary>Original M3b text, superseded</summary>

> A serial loop that picks a runnable stream, advances it one quantum, records
> the cursor, and repeats. […] Cancellation moves here — from three per-token
> hook sites into the pick step, one site instead of three-and-counting.
>
> *Exit:* two concurrent `Generate` requests interleave at quantum granularity
> under the flag, each producing output byte-identical to running it alone.

</details>

### M3c — the suspension boundary

Park and resume across a real forward. `RunningStream`'s cursor contract already
exists and is tested at the type level; this makes it true of an actual forward.

*Exit:* a stream parked mid-generation and resumed produces output
byte-identical to an uninterrupted run. This is the contract §M6 later depends on.

**`hipGraph` capture must be TURNED off on the v2 path** until its WCET is
declared — a graph is one indivisible quantum by construction, and a declared
WCET that ignores an enabled graph is exactly the failure the contract exists to
prevent.

This sentence previously read "is off", as a statement of fact. **It is not off.**
It is default-ON on a served path: `hipfire-arch-deepseek4/src/forward.rs:1612`
is `env_override.unwrap_or_else(|| arch.starts_with("gfx11") || arch.starts_with("gfx12"))`,
it captures the entire decode body in one graph, and it is reached
unconditionally from `serving-core/generate_arch.rs`. qwen35's DFlash verify
graph is default-on too, via a different API. Turning it off is an action item
and not one switch — `hipfire-rdna/src/dispatch/mod.rs` holds three independent
capture families, enumerated together by `graph_state_live()`.

qwen35 **AR** decode is the exception and that is why M3b0 scopes to it:
`hipfire-arch-qwen35/src/qwen35/mod.rs:1000` is a literal `let use_graph = false;`
with the parsed `HIPFIRE_GRAPH` deliberately discarded on the next line. Do not
widen the v2 scope to deepseek4 without dealing with the graph first — the
smallest possible quantum there is one whole captured forward.

### M3d — the exit measurements

The parent plan's exit is three numbers read from the M0 trace, one
realtime-class stream against one bulk stream. They are measurements, not a build
milestone:

1. **p99 and max module duration**, and which `SuperOpKind` owns the max — the
   achievable suspension floor, and therefore the tightest drain budget the design
   can hold.

   **Not obtainable from the M0 trace today.** `TraceRecord.module`'s own doc says
   "Which module (M4 onward). 0 until the module graph exists", every `record()`
   call site passes 0, `TraceRecord` has no field that could hold a `SuperOpKind`,
   and `exec_trace.rs` contains zero references to that type. `TraceEvent` has
   five coarse variants and the file records that a `Yielded` variant was left out
   deliberately because "today nothing can construct it". So either scope M3d to
   measurements 2 and 3, or budget the instrumentation as part of it — but do not
   plan to read (1) off the trace as it stands.
2. **Time from realtime admission to first dispatch** under saturating bulk load.
3. **Bulk throughput, loaded vs solo — ≥ 0.6×.** Without (3), (2) is trivially
   satisfiable by refusing to run the bulk job. Report all three or none.

Expect (1) to name `Moe` as the max. **If it does not, say so** — that means the
workload never routed widely and the measurement was too easy.

## What this unblocks, and what still will not be

Landing M3a alone unblocks half of P2/M3: `session_id` resolves against a table
with real streams in it.

**It does not unblock P2/M3's exit.** That needs "two streams with different
specs steered differently in one batched step", and
`crates/hipfire-serving-core` contains **zero** references to `hipfire_steer` —
the fused batched decode path never calls the hook. That is a separate piece of
work in the serving crate, and it should be planned separately rather than
discovered again.

## Verification

Every stage: `./tests/no-gpu-ci.sh` (which now covers `hipfire-daemon`'s tests)
and `./tests/tiny-affected-gate.sh --require-coverage`, reporting honestly when
it selects no coverage.

Every stage must also show a **flag-off greedy generation byte-identical to
master**. The flag is the whole safety argument; if off is not inert, the stage
is wrong regardless of what the on-path measurements say.

Remove `#![allow(dead_code)]` from `stream.rs` when M3b lands. From that point a
dead item there is a real signal, and the attribute hides it.

**Except that it will fire immediately, on `StreamCursor::advance_module`.**
Nothing in the tree produces a module index: `run_layer_program` is `for op in
program` with no resumption index, and qwen35 rebuilds a fresh per-layer op `Vec`
every token. Its only callers are this file's own tests. So either module
marching lands with M3b — it cannot, see M3b0 — or the attribute stays and this
plan says why. It stays, and the reason is that the cursor's `module` half is
scaffolding for §M4, which is honest as long as it is written down.

## Stop and ask

* the seam requires editing `transport.rs` — the parent plan says it survives
  verbatim, so this means the boundary is wrong;
* a stage cannot be made inert with the flag off;
* the march loop appears to need a second thread — the executor is serial by
  design, and wanting a thread means the quantum is wrong, not the threading;
* `StreamTable` needs to become `Sync` — it holds streams the serial loop owns.

## Note for whoever picks this up

The habit that has repeatedly paid on this subsystem: **grep before building.**
Four things assumed missing already existed (P1's primitive, M2's oracle, the
`session_id` wire field, the steer hook's arch coverage), and two plan sentences
pointed at things that do not exist (`StreamState`, and `batch_runner.rs` in the
daemon — see above).

And **verify the artifact, not the exit code.** Three checks silently no-op'd in
one session — a doc edit whose literal did not match, a falsification broken by a
bad escape, and a gate that selected zero targets — each passing while proving
nothing.
