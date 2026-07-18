# Continuous Cross-Modality Scheduler — Implementation Plan

**Date:** 2026-07-18
**Branch:** chaingun
**Status:** proposed

## Goal

Make the headline feature — *"batch coalescing + continuous management of
multiple prioritized sessions across modalities and tasks"* — actually true in
the serving path, incrementally, without a big-bang rewrite.

## Where we are (measured 2026-07-18)

- `ContinuousWorkScheduler` (`crates/hipfire-scheduler/src/lib.rs:419`) is the
  object the pitch describes: class-agnostic buckets, `WorkloadResources`
  admission (`fits_within`), lease tracking, `next_batch()` coalescing by
  `microbatch_key`. It is **constructed only in `#[cfg(test)]`
  (`lib.rs:2792+`) — zero production callers.**
- The live path is `PriorityPrefillScheduler` (`lib.rs:1458`), wired at
  `state.rs:82` / `chat.rs:1598`. It is **text-prefill only**, **off by default**
  (`HIPFIRE_SERVER_PREFILL_BATCH=1`), and acts as a **turn-gate**: it selects
  request IDs into a `HashSet` (`state.rs:84`) and each request then runs its
  **own** prefill. Health JSON is hardcoded zeros
  (`server_prefill_batch_health_json`, `lib.rs:1021`) tagged
  `rust_server_scheduler_metadata_only`.
- The daemon is a clean single GPU owner but **single-threaded, one model
  resident, serial** (`state.rs:69` "one request at a time";
  `daemon/main.rs:3275`).
- The fusion primitive already exists: selection builds
  `PrefillBatchSelection { sessions, fused_state_batch }` (`lib.rs:858`) and the
  daemon exposes `generate_batch_prefill` / `generate_batch_decode_step`
  (adapter `lib.rs:408/454`). **Nothing calls the fused path from the general
  serving route.**
- Two unrelated `WorkloadClass` enums exist: scheduler
  `{TokenPrefill,TokenDecode,ImageGeneration,Training,Maintenance}`
  (`lib.rs:239`, test-only) and auth `{Text,Image,Training,Other}`
  (`rate_limit.rs:470`, the one routes touch).
- Per-modality backends are all real (text, image-gen `hipfire-diffusion`,
  vision `generate_vl.rs`, embeddings, standalone `hipfire-train`) but each runs
  on an independent path; only text is scheduled. Image-gen bypasses the
  scheduler via `spawn_blocking` (`sdapi.rs`), training is a separate process.
- **MoE router histogram EXISTS but is offline-only.** `MoeRouterHistogram`
  (`qwen35/telemetry.rs:11`) records per-layer top1/topk expert-selection counts;
  it works, but is written to an evidence dir on drop
  (`DaemonMoeRouterHistogramGuard`, `evidence.rs:70`) and gated on `evidence_dir`
  being set. **Neither the scheduler nor the weight pager consumes it** — the
  only "histogram" in the scheduler is an empty `batch_size_histogram: {}`
  placeholder (`lib.rs:1039`), and the pager's predictive prefetch is explicitly
  a "later commit" (`cpu_router.rs:11/24`). It is thread-local and qwen35-only.
- No runtime cost/duration model exists. The scheduler admits by static
  `WorkloadResources` (bytes/slots) only; it has no notion of how *long* a job
  will run, so it cannot do deadline-aware or long-horizon planning.
- **DFlash/DSpark spec-decode is fully disconnected from BOTH the pager and the
  scheduler, and there is no guard against co-enabling paged experts + spec.**
  `weight_pager.rs`/`cpu_router.rs` have zero spec awareness; `speculative.rs`/
  `mtp_compose.rs`/dspark have zero pager awareness; the scheduler has no
  draft/verify/acceptance notion. Paged residency is sized for **single-token
  decode** (`ensure_paged_experts_resident`, `moe_decode.rs:811`), but spec
  verify evaluates a *K-token tree* in one pass — a much larger, burstier expert
  union. This is a live latent risk today (untested combo), not just a missing
  optimization. See the DFlash/DSpark cross-cut below.

## Non-goals (explicitly out of scope for this plan)

- True concurrent multi-model execution / co-residency. The single-HIP-thread,
  one-resident-model daemon is a hard constraint here; we deliver *coordinated*
  (arbitrated + preemptible) cross-modality, not *simultaneous*. Concurrency is
  Phase 7, its own track.
- Cross-node / multi-host. Unchanged (RCCL single-node only).
- De-welding every arch from qwen35. We de-weld only the session seam we must
  cross (Phase 6); full arch generalization is tracked separately
  (MODEL-SUPPORT gap #5).
- Interrupt-style / hardware GPU preemption. Pausing is **cooperative** — a
  running job yields at a loop boundary, it is not force-killed mid-kernel.
  Coarse latency (a few iterations) is acceptable; a whole generation/image
  task holding the GPU is not.

## Sequencing principle

Each phase must ship something demonstrable and be independently revertible.
Order is smallest-diff-that-delivers-value first, welds cut only when a phase
needs them.

## Genericity principle (applies to EVERY phase)

qwen35 is the first full implementation, **not** the abstraction boundary. Each
phase's mechanism lands as an **arch-agnostic seam**, with qwen35 as the first
`impl` behind it — never as new concrete-typed code the generic layer calls by
name. Rules:

- **Scheduler-side logic stays in `hipfire-scheduler`, which is already
  arch-agnostic** (`WorkloadClass`, `WorkloadResources`, `ContinuousWorkScheduler`,
  priority/fairness all operate on opaque specs). No phase may push arch-specific
  types into it. This is most of P2/P3/P4-policy/P7-estimator — it's *born*
  generic; just don't contaminate it.
- **The daemon JSON-lines protocol is already model-agnostic** (`DaemonEngine`
  RPCs). Batch dispatch, park/resume, and histogram export should be **protocol
  messages**, so any arch the daemon can load participates for free. Add the
  capability at the protocol boundary, not inside a qwen35 handler.
- **Executor-side hooks land as traits in `hipfire-serving-core`**, not in
  `hipfire-arch-qwen35`. The three hooks this plan introduces —
  `BatchableSession` (P1), `PreemptibleJob` / `SessionCheckpoint` (P4),
  `ExpertRoutingSignal` (P7) — are **defined generically when the phase needs
  them**, and qwen35 implements them. Do not defer all trait definitions to the
  de-weld phase (P6); P6's job shrinks to *completing* the `SessionRegistry<S>`
  hoist and **proving** the seams by porting a second arch, not inventing them.
- **First non-qwen35 target is deepseek4** (arch 9, closest MoE, own batched
  prefill + native MTP), then a new arch (GLM-5.2). If a seam can't be
  implemented for deepseek4 without touching the generic layer, the seam is
  wrong — fix the abstraction, not deepseek4.
- **Test the seam, not the impl:** every trait gets one arch-agnostic unit test
  (a dummy/mock impl, like the existing `dummy.rs` in serving-core) so the
  generic path is exercised without a GPU or a real model.

---

## Phase 1 — Make text batch coalescing REAL (dispatch the fused selection)

**The single highest-value, smallest diff.** The selection and the daemon fused
kernels already exist; the route just doesn't connect them.

- Change `wait_for_prefill_scheduler_turn` (`chat.rs:1598`) so the request that
  wins selection dispatches the **whole `PrefillBatchSelection`** through
  `generate_batch_prefill` (adapter `lib.rs:408`) as one runtime call, instead
  of clearing IDs and having each request prefill independently.
- One request in the selected batch is the "driver" that holds the engine lock
  and runs the fused call; the others await its completion and read their slice
  of the result. (Selection already carries `sessions: Vec<..>` +
  `fused_state_batch`.)
- Same treatment for decode via `PriorityDecodeScheduler` +
  `generate_batch_decode_step` (adapter `lib.rs:454`) — currently that scheduler
  has **no production caller at all**, so this is where it gets wired.

**Generic seam:** the fused dispatch is a **daemon protocol RPC**
(`generate_batch_prefill`/`_decode_step`) — already model-agnostic, so keep the
route-side logic keyed off the generic `PrefillBatchSelection`, never off a
qwen35 type. Where the route needs to ask a session "can you join this batch?",
define a `BatchableSession` trait in serving-core (`compatibility_key`,
`fused_slice`); qwen35 is the first impl. Any arch the daemon can batch-prefill
then coalesces for free.
**Delivers:** genuine same-model text batch coalescing (N sessions → 1 GPU
invocation), decode included.
**Files:** `chat.rs`, `responses.rs`, `state.rs`, adapter `lib.rs`;
`serving-core` (`BatchableSession` trait).
**Risk:** medium — driver/follower result routing + cancellation. Contained to
text path.
**Exit check:** `coherence-gate-dflash.sh` clean; a 4-concurrent-request bench
shows one `generate_batch_prefill` daemon call, not four, with byte-identical
outputs vs serial; `BatchableSession` has a dummy-impl unit test.

### Phase 1 — CONFIRMED SCOPE (verified 2026-07-18, supersedes "smallest diff")

Investigation corrected the framing: this is **not** a small diff, it is the
**server-side continuous-batching orchestrator**. What actually exists vs missing:

- ✅ **Daemon executes fused batched prefill + decode over N co-resident sessions
  on real GPU today.** `main.rs:4989` → `run_generate_batch_prefill_serial_qwen35`
  → fused kernels; `main.rs:5130` → `run_generate_batch_decode_step_qwen35` →
  fused backend when `session_count>=2` (`qwen35_decode.rs:347`); one batched GPU
  forward over N sessions each carrying its own KV (`prefill_batch.rs:2226/2961`).
  `SessionRegistry` holds N sessions co-resident (`session.rs:66`).
- ✅ **Full lifecycle RPC surface at the adapter:** `reserve_session_state` →
  `generate_batch_prefill` → loop `generate_batch_decode_step` → `release_sessions`
  (`daemon-adapter/src/lib.rs:408/454/+`). Only mock-tested; **no serving caller.**
- ✅ **Scheduler config plumbing** reads `HIPFIRE_SERVER_PREFILL_BATCH_MAX/_WAIT_MS`
  (`scheduler.rs:1143/1166`) and produces `PrefillBatchSelection`.
- ✅ **Backend selection** via `HIPFIRE_QWEN35_DECODE_BATCH` /
  `HIPFIRE_QWEN35_PREFILL_SESSION_BATCH` (auto|serial|fused|fused_grouped_moe|off),
  read daemon/runtime-side (`generate/lib.rs:1428/1460/1484`, `qwen35_decode.rs:360`).
- ✅ **Acceptance test already written (aspirational):**
  `tests/smoke-server-decode-batch.sh` drives concurrent `/v1/chat/completions`
  behind a barrier and asserts `health.decode_batch.{total_batches, serial_batches,
  selected_batch_size, last_backend, last_chunk_count/size, last_decode_ms}` and
  `health.prefill_batch.{resident_runtime_sessions, resident_decode_sessions,
  pending_requests, selected_batch_size}`. Fails today at `decode_total_batches<1`.
- ❌ **Missing = the orchestrator (the whole task):** consume the scheduler
  selection → drive the daemon lifecycle → fan tokens back per-request → populate
  the telemetry (currently hardcoded zeros, `scheduler/lib.rs:1036-1091`).

**Two hard constraints:** (a) only single-step (`max_tokens=1`) has ever run on
GPU — the many-token decode loop is untested, budget real validation; (b) fused
path requires qwen35/qwen35-moe, `pp==1`, **no DFlash**, no CASK/TriAttn eviction,
and `HIPFIRE_KV_HIERARCHICAL=1` forces serial swap (`qwen35_decode.rs:353,406`).
Outside that envelope → serial fallback or error. NOTE: "no DFlash" collides with
the DFlash/DSpark cross-cut — batched decode + spec-decode are mutually exclusive
in the current fused envelope; unifying them is future work, not Phase 1.

### Phase 1 — BUILD BLUEPRINT (turnkey)

New module `crates/hipfire-server/src/batch_runner.rs`, gated by the existing
default-off `HIPFIRE_SERVER_PREFILL_BATCH`; old monolithic path
(`chat.rs:1798` engine.lock → `generate_streaming_events_controlled`) stays as
the flag-off fallback.

1. **Registry** on `AppState`: `batch_inbox: Mutex<HashMap<String, PendingRequest>>`
   where `PendingRequest { gen_req, worker_key_id, sampling, tx: mpsc::Sender<BatchEvent> }`
   (`BatchEvent = Token(String) | Done(DoneEvent) | Error(String)`).
2. **Route change** (`chat.rs`/`responses.rs`, flag-on branch): build `gen_req`
   as today, insert `PendingRequest` into `batch_inbox`, enqueue the scheduler
   draft (as now), then `await` the `mpsc::Receiver` — accumulate tokens, return
   on `Done`. Requests no longer touch `state.engine` directly.
3. **Runner task** (spawned once at serve start when flag on, owns the engine
   via `state.engine`): loop —
   - `next_prefill_batch` → `PrefillBatchSelection`; pull matching `PendingRequest`s.
   - `reserve_session_state` for the batch; `generate_batch_prefill`
     ({batch_id, worker_key_id, sessions:[{id, prompt}]}) once.
   - loop `generate_batch_decode_step` over resident sessions; route each
     `..._session_done`/token payload to that req's `tx`; drop sessions on stop /
     max_tokens; admit newly-selected requests as slots free (continuous batching).
   - `release_sessions` when a session completes; update telemetry counters.
4. **Telemetry**: replace hardcoded-zero `server_decode_batch_health_json` /
   `server_prefill_batch_health_json` (`scheduler/lib.rs:1021/1080`) with a shared
   `BatchTelemetry` the runner writes (satisfies the smoke's field contract).
5. **Generic seam**: `BatchableSession` trait in serving-core keys compatibility
   (`compatibility_key` = worker_key_id + cache-mode + exclusions) so the runner
   groups by trait, not qwen35 type; qwen35 first impl + dummy-impl unit test.

**Remaining reads before coding** (next session): `AppState` construction site in
`state.rs` (add field + spawn runner), `GenerateRequest`/`DoneEvent` shape in
`hipfire-generate`, and the exact per-session token/done payload fields emitted by
`run_generate_batch_decode_step_qwen35`. **Validate on nix2** (gfx1103, CWSR off)
via `smoke-server-decode-batch.sh` with `HIPFIRE_QWEN35_DECODE_BATCH=fused_grouped_moe`.

## Phase 2 — Turn it on and tell the truth

- Flip `server_prefill_batch_enabled` (`lib.rs:1010`) default to on once Phase 1
  is stable (keep the env var as a kill switch).
- Replace the hardcoded-zero health JSON (`lib.rs:1021/1080`) with real
  counters from the live scheduler (queued/selected/batch-size/aging hits).

**Delivers:** the feature is on by default and observable.
**Risk:** low. **Exit check:** `/health` reports non-zero live batch metrics
under load; kill switch returns to serial.

## Phase 3 — Adopt `ContinuousWorkScheduler` as the admission spine (text-only)

Now make the headline object load-bearing, still one modality.

- Instantiate `ContinuousWorkScheduler` in `AppState` alongside (then in front
  of) `PriorityPrefillScheduler`. Text prefill/decode enqueue as
  `WorkloadClass::TokenPrefill`/`TokenDecode` and admit through
  `WorkloadResources::fits_within` against real daemon accelerator inventory
  (`scheduler.rs:16`).
- Unify the two `WorkloadClass` enums: make auth `{Text,Image,Training,Other}`
  (`rate_limit.rs:470`) a `From`/derive of the scheduler enum, or collapse to
  one. Removes the two-taxonomy footgun before more modalities arrive.

**Delivers:** the real resource-accounting spine runs in prod; priority/fairness
(already real, `lib.rs:814`) now feeds admission, not just a gate.
**Risk:** medium — behavior parity with Phase 1's priority ordering.
**Exit check:** text throughput/fairness unchanged vs Phase 2; leases visible in
`snapshot()`.

## Phase 4 — Cooperative preemption (yield checkpoints)

The precondition for admitting a long-running modality without starving
interactive work. A queued high-priority job must be able to interrupt a job
**already running**, at a granularity coarse enough to be cheap but far finer
than a whole task — and **without discarding completed work.**

Mechanism (fits the single-thread cooperative daemon — no GPU interrupt needed):

- **Yield checkpoint = the iterative backend's loop boundary.** Every
  long-running executor is a loop; the boundary between iterations is the
  natural, cheap suspend point:
  - text decode — between decode steps (per-token, sub-100ms)
  - text prefill — between prefill chunks (`prefill_chunk.rs`; the chunk *is*
    the quantum, already bounded)
  - image gen — between diffusion **sampler steps** (one UNet forward each, tens
    of ms) — NOT between whole txt2img calls
  - training — between micro-steps / grad-accum boundaries
- **Scheduler-side API:** the running job holds its lease and, at each
  checkpoint, calls `scheduler.should_yield(lease, now) -> Option<PendingLease>`.
  If a higher-priority waiter exists and the lease has run its **min-quantum**
  (≥1 iteration, ≤ a small cap — the anti-thrash floor), the job parks and
  releases the engine lock; the scheduler hands the device to the waiter and
  re-enqueues the parked job at its current priority (with aging so it resumes
  promptly).
- **No work thrown away = state stays resident/parked at the checkpoint.** Text
  reuses the existing `SessionRegistry` swap-out (`session.rs:106`): KV +
  recurrent state is already parkable. Image gen needs the diffusion loop made
  *resumable* — persist `{latent, step_idx, sampler_state}` so resume continues
  at `step_idx+1` instead of restarting. That resumability is the only real new
  work here.
- **Preemptibility is per-class policy.** Interactive text decode: preemptible
  every step. Image: preemptible every sampler step but with a slightly larger
  min-quantum. Training: preemptible at micro-step, lowest priority, always
  yields. Maintenance: yields to everything.

**Generic seam:** the yield API (`should_yield` + min-quantum) lives in the
arch-agnostic scheduler. The checkpoint contract is a `PreemptibleJob` trait
(`yield_point()` at the loop boundary) + a `SessionCheckpoint` trait
(`park()`/`resume()` over opaque state) in serving-core — each iterative executor
(text decode/prefill, diffusion, training) implements them; the daemon loop
speaks a generic park/resume protocol message. qwen35 decode/prefill is the
first `PreemptibleJob` impl. Nothing in the yield/park machinery names a qwen35
type — the state it parks is `impl SessionCheckpoint`, opaque to the scheduler.
**Delivers:** a high-priority request interrupts a running low-priority job
within one iteration and resumes it later with zero recomputation — first
demonstrable on text-only (high-pri prompt preempts a low-pri long generation
between tokens), before any second modality exists.
**Files:** `hipfire-scheduler/src/lib.rs` (yield API + min-quantum policy),
`serving-core` (`PreemptibleJob`/`SessionCheckpoint` traits), `qwen35_decode.rs`/
`qwen35_prefill.rs` (first impl), daemon loop (`main.rs`) for generic park/resume.
**Risk:** medium — cancellation vs park must not corrupt session state; the
min-quantum floor is the tuning knob (a real one — set too low it thrashes, too
high it starves; leave it env-tunable).
**Exit check:** under a low-pri long generation, a high-pri request starts within
one min-quantum (measured), and the preempted job's final output is
byte-identical to running it uninterrupted; `coherence-gate-dflash.sh` clean.

## Phase 5 — Merge the arbiters: one universal GPU arbiter across every class

**Revised target (2026-07-18, per direction):** not "bolt image/embed admission
onto the scheduler," but *merge the arbiters*. Today there are two-and-a-half
independent GPU arbiters: (1) `work_scheduler` (`ContinuousWorkScheduler`) —
admission/priority, text-only so far; (2) the `state.engine` mutex — the de-facto
*execution* gate every daemon op (`engine.lock()`: chat legacy, embeddings,
`responses`, admin, steer) contends on; (3) the diffusion path
(`spawn_blocking` on `diffusion_pipelines`) which bypasses both and contends on
the physical GPU. The end state: **`work_scheduler` is the one arbiter that
orders work, and the batch runner is the one thing that touches the GPU.** Every
class enqueues a lease and the runner dispatches it inside its exclusive GPU
turn; the P4 preemption machinery (park/resume stack, yield checkpoints) applies
uniformly at each class's step boundary. The engine mutex stops being a separate
arbiter; diffusion stops bypassing.

Scope spans **all five workload classes** (`WorkloadClass` already carries
`TokenPrefill`/`TokenDecode`/`ImageGeneration`/`Training`/`Maintenance`), across
**two coordination regimes**:

- **In-process (runner-dispatched)** — the runner owns the engine + diffusion
  pipelines and runs these on its GPU turn:
  - **Text** (`TokenPrefill`/`TokenDecode`) — fused decode cycle. DONE (P1–P4).
  - **Image** (`ImageGeneration`) — diffusion pipeline moved off the route's
    `spawn_blocking` into the runner's turn. Preempts at *sampler-step*
    boundaries via the existing progress callback: return `Err`/`Interrupted`
    when a higher-priority waiter appears, then **restart from the same seed**
    (diffusion is deterministic → the restarted image is byte-identical to the
    uninterrupted one; cost is redone steps, not correctness). This is the meaty
    preemption case and drives the design.
  - **Embed** (`Maintenance`/dedicated) — `engine.embed`, short; whole-op
    quantum (non-preemptible, min-quantum covers it).
  - **Steering** (`hipfire_steer`, in-daemon, process-global) — daemon steer
    ops (capture/apply/load_lora) routed like a short daemon op; its own class
    or `Maintenance`.
- **Cross-process (flock bridge)** — **Training** (`hipfire-train`, a separate
  non-daemon binary) coordinates via the `hipfire lock` GPU flock, not the
  runner. The arbiter bridges: when the scheduler grants a training lease the
  runner *releases* the GPU flock and quiesces; training takes a scheduler-aware
  lease + flock; interactive work preempts training back by priority. This is
  the hardest, novel piece and can trail the in-process merge.

**Incremental order (each GPU-validated + committed):**
1. **Runner executor seam** — generalize the runner loop from "text batch only"
   to `dispatch(lease)` by `WorkloadClass`; a unified job inbox
   (`ScheduledJob { Text | Embed | Image | Steer }`) parallel to `batch_inbox`.
   Text is the existing impl; prove the seam end-to-end with **Embed** first
   (simplest new class) — embed route enqueues + awaits instead of locking the
   engine.
2. **Image under the runner** — move diffusion into the runner turn; sampler-step
   preemption via callback + restart-from-seed; the byte-identical exit check.
3. **Steering under the runner** — steer ops as a runner-dispatched class.
4. **Training flock bridge** — cross-process yield/resume against the GPU flock.

**Delivers:** one priority/resource arbiter and one GPU executor for text,
image, embed, steering (and, via the bridge, training) — a long txt2img or a
training step no longer blocks interactive text; everything time-slices by
priority. Still coordinated, not simultaneous (serial GPU).
**Risk:** medium→high — execution-path surgery + the cross-process flock bridge.
**Exit check (per step):** interleaved load shows priority-ordered, preemptive
GPU handoff (trace); preempted image restart is byte-identical; p99 interactive-
text latency bounded under concurrent image/train load.

## Phase 6 — Complete the hoist + PROVE genericity with a second arch

By now the three seams (`BatchableSession`, `PreemptibleJob`/`SessionCheckpoint`,
`ExpertRoutingSignal`) already exist from P1/P4/P7 with qwen35 impls. P6 is not
"invent the abstraction" — it's finish the fat one and validate the whole set on
real second/third arches.

- Finish the `SessionRegistry<S>` "trait hoist" (`session.rs:64/106`): the
  remaining generic `RequestSession` trait so the front end isn't tied to
  `Qwen35RequestSessionState`; fold the feature-flagged lfm2 copy into it.
- **Port deepseek4 onto all four seams** (arch 9, MoE, own batched prefill +
  native MTP). This is the genericity acceptance test: if any seam needs a
  generic-layer change to fit deepseek4, the seam was wrong — fix it there, not
  in deepseek4. Then a fresh arch (GLM-5.2) should need only trait impls.
- Give the standalone trainer (`hipfire-train`) a **cooperative VRAM lease**:
  it submits a `WorkloadClass::Training` (singleton, `lib.rs:348`) claim to the
  daemon's resource accounting so serving evicts/pauses cleanly instead of
  OOM-contending. Trainer stays a separate process; it just respects the lease.

**Delivers:** proof the cross-modality arbiter is arch-generic — a second MoE
arch runs through the same scheduler/preemption/estimator path with no
generic-layer edits; training coexists without VRAM collisions.
**Risk:** high — touches the hottest serving struct. Gate behind coherence +
per-arch parity.
**Exit check:** deepseek4 admits, batches, preempts, and emits routing signal
through the same generic path as qwen35 with zero arch-specific branches in
scheduler/serving-core; trainer + serve on one GPU without OOM.

## Phase 7 — Learning job-timing estimator (long-horizon planning)

The scheduler currently plans on *space* (`WorkloadResources`) but not *time*.
Long-horizon planning — deadline-aware admission, "preempt vs let-finish"
decisions, ordering to hit SLAs — needs a predicted duration per job. Build an
online empirical estimator, not an ML model.

- **Model = observed completions, keyed by cost class.** Every lease already has
  `enqueued_at_ms` and flows through `complete(lease_id)` (`lib.rs:593`); record
  actual run duration there. Key by `(WorkloadClass, model/worker, size bucket)`
  where the size bucket is the natural cost driver per class:
  - text prefill — prompt-token count
  - text decode — requested max_tokens (updated to actual as it streams)
  - image — sampler steps × resolution
  - training — micro-steps
  Maintain a rolling estimate per key (EWMA of mean + a high quantile for the
  tail). **ponytail:** an EWMA/quantile table over real completions is the lazy
  correct thing — no regression model until it measurably falls short; the key
  schema is the upgrade seam.
- **Feed the arbiter three ways:** (1) admission — reject/queue if predicted
  finish blows a deadline; (2) preemption (Phase 4) — "let a low-pri job with
  <ε remaining finish instead of parking it"; (3) ordering — shortest-predicted
  -job-first within a priority band when fairness allows.
- **Wire the MoE histogram in as the per-expert cost signal — generically.**
  The existing `MoeRouterHistogram` (`telemetry.rs:11`) is the hot/cold expert
  distribution, but it's qwen35-only and thread-local. Hoist it to a generic
  `ExpertRoutingSignal` trait in `hipfire-runtime` (`record_selection(layer,
  experts)`, `hotness_snapshot()`) that **any** MoE arch emits — deepseek4 and
  GLM-5.2 are MoE and need the same signal for paging-cost/prefetch. qwen35's
  histogram becomes the first impl. For paged-expert models job duration depends
  on how many *cold* experts must be paged in — a static resource count can't see
  that. Promote it from evidence-dir-only to a live rolling residency/hotness
  signal the estimator reads (predict paging cost) and the pager reads
  (predictive prefetch — the "later commit" `cpu_router.rs:24` already names).
  One generic signal, two consumers, all MoE archs. Cross-links the weight-pager
  work.

**Delivers:** the scheduler can predict job cost and plan over a horizon, not
just admit by bytes. First win is deadline-aware text admission; the paging-cost
term makes it real for big MoE (DeepSeek-V4 / GLM-5.2).
**Generic seam:** the estimator lives entirely in `hipfire-scheduler` (already
arch-agnostic) and keys on opaque `(class, model, size-bucket)` tuples — no arch
types. The only per-arch piece is the `ExpertRoutingSignal` impl, and it's a
trait; a non-MoE or non-paged arch simply reports a null hotness signal and the
estimator degrades to the size-bucket term.
**Files:** `hipfire-scheduler/src/lib.rs` (estimator + `complete` hook),
`hipfire-runtime` (`ExpertRoutingSignal` trait), `qwen35/telemetry.rs` (first
impl), `weight_pager.rs` / `cpu_router.rs` (prefetch consumer).
**Risk:** low-medium — a wrong estimate degrades scheduling quality, never
correctness. Bound its authority: estimates influence ordering/admission, never
skip work.
**Exit check:** predicted vs actual duration MAPE tracked in health JSON and
converging under steady load; a synthetic mix shows deadline-aware ordering
beating FIFO on p99 for the tagged-urgent class.

**Data-collection note:** the estimator's telemetry (lease durations + the live
histogram) can start logging as soon as P3 lands — collect early so the model is
warm by the time P4/P5 want to consult it.

## Cross-cut: DFlash/DSpark spec-decode ↔ pager + scheduler

Spec-decode reshapes both the residency working set and the timing model, and
today it is **disconnected from both** (no cross-references either way, no
co-enable guard). This is not its own phase — it is a set of requirements
attached to P4 (safety), P7 (estimator + `ExpertRoutingSignal`), and the
weight-pager work. Address in this order:

1. **Guard first (do this regardless of the rest).** Paged experts + spec-decode
   is an untested combo that can be enabled together today. Before optimizing,
   add the interaction test (`coherence-gate-dflash.sh` with
   `HIPFIRE_QWEN35_PAGED_EXPERTS=1`) and, if it misbehaves, a temporary refusal
   at load. Cheap; closes a live latent risk.
2. **Size residency for the verify union, not one token.** `ensure_paged_experts
   _resident` is fed a single token's `indices` (`moe_decode.rs:811`). Under spec
   verify, feed it the **union of experts across the K-token draft tree** (the
   batched-prefill path at `prefill_chunk.rs:766` already takes a set — reuse
   that shape). Without this the pager thrashes precisely when spec is trying to
   win latency.
3. **Count the drafter in the VRAM budget.** The DSpark drafter is a second
   resident model; the MTP head adds weights too. The pager's
   `vram_budget_bytes` must subtract the draft engine's footprint (and the
   scheduler's `WorkloadResources` for a spec-enabled worker must include it).
4. **Use the drafter AS the prefetch oracle** — the payoff. DSpark/MTP predicts
   the next tokens before the target runs, i.e. it predicts *which experts the
   verify will touch*. Feed that prediction into `ExpertRoutingSignal` /
   `cpu_router` prefetch (the "later commit" `cpu_router.rs:24` names) so cold
   experts are paged in *during* drafting, overlapped, before verify needs them.
   This is the "massively influence" upside: spec-decode turns the pager from
   reactive to predictive for free.
5. **Make the estimator spec-aware.** A decode step emits 1..K accepted tokens by
   acceptance rate; the P7 cost model must key decode on the existing
   `SpecMetrics` acceptance signal (unified SpecMetrics / `drain_extra_metrics`),
   not a fixed one-token-per-step assumption. Treat draft+verify as one coupled
   lease, not two independent jobs.

**Generic seam:** none of this is qwen35-specific in principle — the verify-union
set, the drafter-footprint accounting, and the acceptance-rate signal are all
arch-neutral. DFlash/MTP already exists on lfm2moe (`arch-lfm2moe/src/dflash.rs`)
as well, so the prefetch-oracle hook belongs on the generic `ExpertRoutingSignal`
/ spec-metrics seam, qwen35 first.

## Phase 8 (separate track) — True concurrent execution

Breaking one-model-resident-serial (`state.rs:69`) into overlapped execution
(multi-stream, co-residency, or interleaved decode across models) is a large
architectural item with its own design. **Do not fold into the above.** It
becomes tractable only after Phases 3–7 give a real lease/resource/preemption/
timing model to schedule against. List as follow-up; scope separately.

---

## Dependency graph

```
P1 (fuse text)  ──▶ P2 (default on + telemetry) ──▶ P3 (ContinuousWorkScheduler spine)
                                                          │
                              ┌───────────────────────────┼───────────────────────────┐
                              ▼                            ▼                            │
                    P4 (cooperative preemption)   P7 (timing estimator —               │
                              │                    data collection starts here) ◀──────┘
                              ▼                            │
                    P5 (image/embed arbiter) ◀────────────┘  (estimator sharpens P4/P5)
                              │
                              ▼
                    P6 (de-weld session + training lease)
                              │
                              ▼
                    P8 (true concurrency — separate track)
```

Notes:
- P4 (preemption) can begin on the text-only path as soon as P3 lands — it does
  not need P5. P5 depends on P4 (image must be preemptible before it takes a
  shared lease).
- P7 (estimator) branches off P3: its data collection starts with the spine, and
  its predictions *sharpen* P4's preempt-vs-finish and P5's deadline-aware
  admission — useful before P6, hence numbered after P5 but wired earlier.

## The honest changelog line at each stage

- After P2: *"Same-model text batch coalescing, on by default, in front of a
  serial single-model daemon."*
- After P4: *"Priority scheduling with cooperative preemption — a high-priority
  request interrupts a running low-priority job within one iteration and resumes
  it with no recomputation."*
- After P5: *"Priority + resource-arbitrated, preemptible scheduling across text,
  image, and embeddings on a shared GPU (time-sliced)."*
- After P7: *"Deadline-aware, cost-predicting scheduling — plans over a horizon
  using learned job durations and MoE expert-hotness."*
- After P8: the full headline.

## Verification gates (every phase)

- `./tests/no-gpu-ci.sh` for workflow/logic changes.
- `./tests/coherence-gate-dflash.sh` after any change touching the runtime hot
  path (P1, P3, P4, P6).
- Concurrency/latency benches under `benchmarks/` for P1/P4/P5 (batch-size,
  preemption-latency, and p99 claims must be measured, not asserted).
- P7 estimator: predicted-vs-actual MAPE tracked in health JSON; estimates may
  influence ordering/admission but must **never** skip or truncate work — a bad
  prediction is a scheduling-quality regression, never a correctness bug.
- **Preemption correctness invariant:** a preempted-and-resumed job's output
  must be byte-identical to the same job run uninterrupted. This is a hard gate
  on P4/P5 — a park/resume that perturbs KV or latent state is a correctness
  bug, not a perf regression.
