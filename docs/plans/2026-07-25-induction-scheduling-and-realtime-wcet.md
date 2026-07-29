# Induction through the scheduler, and the realtime WCET contract

Status: design, 2026-07-25. Companion to
`docs/plans/2026-07-25-zaya-streamed-calib-position-batching.md`, which this
constrains: it fixes what the position-slice batch size `S` must be capped by,
and why the yield point is a dispatch rather than a block.

## Why

Three complaints about induction (`hipfire-coexistence calibrate`) as it runs today:

1. Inducting a model **stops** inference rather than dropping the induction job's
   priority.
2. Memory limits are not shared between the daemon config and induction, which
   matters when the box has other bursty memory consumers.
3. Multiple code paths are maintained where one could be reused.

(3) turns out to be the weakest and is addressed last. (1) and (2) are real and
are what this document is mostly about.

## Findings

### Why `hipfire-daemon` has no scheduler yet still does work

The daemon is a strictly serial JSONL executor: `for line in stdin.lock().lines()`
(`hipfire-daemon/src/main.rs:3319`), one request at a time, in arrival order, no
queue and no concurrency. It never *chooses* anything, so it needs no scheduler.

`hipfire-server` is its only client and does all the deciding first:
`ContinuousWorkScheduler` lives in `AppState.work_scheduler`
(`hipfire-server/src/state.rs:90`) as an in-process `Mutex` with no persistence
and no IPC surface. Policy in the server, mechanism in the daemon.

Three processes each own one piece of the resource picture and none of them talk:

| process | owns | knows about the others |
|---|---|---|
| `hipfire-server` | `ContinuousWorkScheduler` — priority buckets, leases, aging | — |
| `hipfire-daemon` | `ResourceReservationManager` — real `hipMalloc` + touched-page placeholders (`main.rs:581`, `:654`) | no `hipfire-scheduler` dependency at all |
| `hipfire-coexistence` | `calibrate` | `flock` only |

### The daemon is the only place that can preempt within a request

Once a request crosses the JSONL boundary the daemon runs it to completion — the
loop has no interrupt point. So a server-side scheduler can only preempt *between*
requests. A scheduler co-located with execution can preempt *within* one, at
dispatch granularity.

For a realtime workload that difference is the jitter bound. This is the
correctness argument for the stated global target (everything through the daemon,
daemon does scheduling and batching), and it is stronger than the
architectural-tidiness argument.

### Worst-case blocking today

| running job | preempt check | yield granularity | worst-case block |
|---|---|---|---|
| Text decode | after each decode step past `min_quantum()=4` (`batch_runner.rs:1114`) | 4 decode steps | ~40 ms @100 tok/s, ~130 ms @30 tok/s |
| Image | watcher polls every 5 ms (`:812`), honored at a sampler step | 1 sampler step | ~50–200 ms |
| **Train** | **none — holds the runner turn to completion** (`:124-126`) | whole quantum, default 25 steps (`routes/train.rs:86`) | **seconds to tens of seconds** |

Text and image are genuinely preemptible. `Train` is not. **Induction must not
copy the `Train` shape**, which was the obvious move and is the trap.

`Train`'s figure is a dial, not an intrinsic property: `quantum` is a request
field defaulting to 25 (`routes/train.rs:86`), and `LoraTrainSession` is already
described as micro-step-preemptible, driving one quantum per `TrainLora` request
(`daemon/main.rs:3142`). Set `quantum: 1` and its WCET becomes a single training
step. Training is a blocker today because nobody turned the dial down, not
because it cannot yield.

### The scheduler's resource dimension is currently decorative

`routes/train.rs:105` enqueues with `WorkloadResources::default()` — all zeros.
`state.rs:274` declares `npu_slots: u32::MAX`. Nothing has ever exercised
`fits_within` against real numbers. The first workload to declare true byte counts
will also be the first to discover whether the capacity value is tuned.

## The realtime constraint is WCET, not bandwidth

The first framing attempted here was bandwidth contention: two bandwidth-bound
jobs splitting one ~256 GB/s bus. That framing is **wrong** for the workload that
matters. A small streaming speech model (Kyutai-100M class) needs a trivial share
of bandwidth. What breaks audio is a single indivisible unit of work landing
inside the interpacket gap and overrunning it. That is a worst-case-execution-time
constraint, and it is indifferent to how much headroom exists on average.

Two relaxations make it tractable:

- **Startup slack is generous, mid-stream slack is not.** ~300 ms of delay before
  TTS *starts* is fine. The same delay once it is generating corrupts audio. The
  deadline only binds while a stream is live.
- **Cascade beats omni.** With separate STT → LLM → TTS stages, buffering between
  stages absorbs LLM jitter, and only the speech models sit in the frame loop.
  True omni realtime models put the large model *in* the loop and are much harder.

### The number that decides it

WCET is set by the largest indivisible dispatch, which is a per-model property and
computable ahead of time. For ZAYA1-8B, from the companion plan's measurements,
one expert is `expert_gate_up` 16.8 MB + `expert_down` 8.4 MB ≈ 25 MB, i.e. **~100 µs**
of weight read at 256 GB/s. The 0.25 s/block figure at S=128 is *thousands* of such
dispatches, not one long kernel.

The consequence is important and initially counterintuitive: **a large `S` and a
fine yield granularity are compatible.** Induction does not need to shrink its
batch to become preemptible, because the natural yield point is between dispatches,
not between microbatches or blocks. Shrinking `S` would be the worst of both worlds
— the companion plan measures 32.7 s/block at S=1 against 0.25 s/block at S=128, so
a small batch is ~100x slower *and* buys no WCET improvement that yielding between
dispatches doesn't already give.

The constraint binds where the expert is large enough that one dispatch alone
overruns the gap. Then the only fixes are to tile the dispatch or to move the model.

## Dispatch graphs set the indivisible unit

A launched `hipGraph` has no host control point: its nodes cannot be preempted
between. **The graph becomes the indivisible unit**, so WCET is the duration of the
whole graph rather than of the largest kernel inside it. Dispatch-level preemption
and graph capture are therefore mutually exclusive at the same granularity.

Graph *scope* is a free design variable, and both extremes already ship here:

| site | captured scope | indivisible unit |
|---|---|---|
| `hipfire-arch-lfm2moe/src/forward.rs:2093` | "the layer loop + head" | an **entire decode step** |
| `hipfire-arch-deepseek4`, `hipfire-arch-minimax` | `decode_step_with_graph` | an entire decode step |
| `hipfire-runtime/src/dflash.rs:1414` | `draft_ffn_graphs[layer][batch]` | **one FFN** |

So the rule is **graph scope ≤ WCET budget**, and dropping from whole-step to
per-layer capture (the dflash pattern) is the escape hatch when a step overruns.

The conflict lands on the workload that gains least from graphs, which is what
makes it tractable:

- Graphs only pay when **launch-bound**. Bulk induction after position-slice
  batching is bandwidth-bound, so graphs buy approximately nothing and leaving it
  ungraphed and dispatch-preemptible costs approximately nothing. This matches the
  measured ZAYA decode result: graph capture correct, tok/s flat, because decode is
  GPU-exec-bound rather than launch-bound.
- A small streaming speech model is maximally launch-bound — many tiny kernels — so
  graphs are a large win there, and it is top priority so it is never the workload
  being preempted. Graph it aggressively.
- The case that needs measuring is **large-model interactive decode under
  whole-step capture**, where WCET is one full decode step. A ~15 ms body against an
  ~80 ms frame gap fits, but that is a measurement to take, not to assume.

## The declared bound belongs in the artifact

Store the schedulability contract in the `.hfq` so a model carries its own
admission terms, rather than having the scheduler guess. Two caveats govern the
design.

**It is not a scalar.** WCET is conditional on graph scope, batch size, and quant
mode — each changes both the number of dispatches and the size of the largest one.
Store the *primitives* (largest single dispatch bytes and FLOPs, expert dimensions,
graph scope) plus a measured calibration point, and let the runtime derive the bound
for its actual configuration. Derived-from-primitives goes stale loudly when a
kernel changes; a baked scalar goes stale silently, and this is a safety number.

**Measure it on the serving path, not the induction path.** Induction runs the
calibration forward, which is not the serving forward: the streamed path uses
per-token flash-decode attention where the resident path uses full-sequence
prefill, and quantized serving uses different gemv kernels than the bf16
calibration pass. A WCET captured during induction would be wrong for serving, and
wrong in the unsafe direction. Quantization is the better capture point because it
knows the final weight layout; anything measured earlier is a lower bound at best.

Alongside the bound, record the **maximum yield granularity the model is compatible
with** — the largest quantum it can be scheduled at without breaking its own
latency assumptions. That is what lets admission reject the pairing directly: a
realtime speech model declares a small tolerance, a large-quantum model or an
untuned training job declares a large one, and the scheduler refuses to co-admit
them without having to model either workload's internals.

## The design principle: a declared WCET contract

Every workload declares the duration of its largest indivisible unit. Realtime
co-admits with bulk only when the bulk workload's declared WCET fits inside the
realtime stream's slack.

- Induction can honestly declare sub-millisecond, because its dispatches are small.
- `Train` at its default `quantum: 25` declares a bound too large to co-run with a
  live stream. Lowering the dial toward 1 shrinks it to a single training step; how
  small that actually is remains model-dependent and must be measured, not assumed.
- A model whose single expert dispatch exceeds the gap cannot co-run at all, and
  is refused or evicted rather than admitted and allowed to corrupt audio.

This replaces binary admission control ("suspend induction when voice is live")
with a negotiation that keeps bulk work running whenever it provably cannot hurt.
Binary suspend remains the fallback for workloads that cannot declare a bound.

`SchedulerPriorityClass::Realtime` already exists as the top class
(`hipfire-scheduler/src/lib.rs:706`) and is currently unused. It is the natural
carrier for the slack figure.

## Shape of the change

### Phase 0 — unify the memory budget (fixes issue 2)

`ResourceReservationManager::status_json()` already publishes `vram_target_bytes`
and `held_vram_placeholder_bytes` (`daemon/main.rs:606`), and `hipfire-coexistence`
already links `hipfire-daemon-adapter` (`main.rs:244`).

1. Expose that status where a non-daemon process can read it.
2. `tune_geometry_for_gpu` (`calibrate.rs:3005`) sizes from `budget − daemon_target`
   instead of raw `get_vram_info()`.

This is worth doing independently of everything below. The current live-VRAM probe
is accurate at the instant it runs and meaningless thereafter, because induction
then holds that memory for hours while the daemon independently releases and
reacquires placeholders on every worker load (`main.rs:3806-3835`). The two
allocators are coupled by nothing but timing. A budget is less precise instantly
and is a contract that persists, which is what a long job needs.

Estimated: ~half a day.

### Phase 1 — induction as a resumable daemon session

Model it on `LoraTrainSession` / `DrafterTrainSession` (`daemon/main.rs:3145`,
`:3167`) — resident, micro-step-preemptible, held alive between quanta and advanced
one quantum per request — **not** on `Dispatch::Train`, which holds the turn.

The quantum mechanism already exists on both sides and does not need inventing:

- Calibrate side: `--pause-after-layers` (`calibrate.rs:459`), `--resume` as the
  default, and a checkpoint guarded by an engine-build fingerprint
  (`calibration_checkpoint_execution_fingerprint`, `:1535`) that refuses a resume
  from a mismatched binary.
- Server side: `batch_runner.rs:715-750` re-enqueues an unfinished job under a fresh
  id carrying the same `run_id`, answering `tx` only on the terminal quantum.

The session-in-daemon plus quantum-per-request shape is **invariant under the
scheduler migration**: it works with today's server-side scheduler and continues to
work unchanged when scheduling moves into the daemon. Only the caller changes. This
is why process-per-quantum (spawning the coexistence binary per slice) was rejected
— it is cheaper now and is throwaway under the target architecture.

Estimated: ~4–6 days, dominated by sessionizing `calibrate::run_cli` into
`new()` / `step()` / `finish()`.

### Phase 2 — the WCET contract

1. Add a declared-WCET field and a maximum-yield-granularity field to
   `WorkloadSpec`, both defaulting to "unbounded" so nothing silently claims to be
   schedulable alongside realtime work.
2. Emit the WCET primitives and graph scope into `.hfq` metadata at quantization
   time, measured on the serving path.
3. Induction declares its measured per-dispatch bound; the scheduler checks it
   against any live `Realtime` workload's slack before co-admitting.
4. Give induction a preempt check between dispatches, mirroring the text decode
   path's cooperative yield rather than the `Train` hold.
5. Lower the `Train` default quantum, or make it derive from the same budget, so
   training stops being an unbounded blocker by default.

Phase 2 is what actually delivers "drop the priority instead of stopping
inference." Phases 0 and 1 are prerequisites.

## Deliberately not in this change

- **Moving `ContinuousWorkScheduler` into the daemon.** It is the right target, but
  it has no persistence and no IPC surface, and the migration touches serving
  admission. It should not ride along with an offline job's needs. Phase 1's shape
  is chosen so that migration is a caller change, not a redesign.
- **Omni realtime models.** These put the large model in the frame loop and need
  large streaming experts evicted to CPU or NPU with the omni model's bandwidth
  guarded. That is a residency and placement problem for `plan_model_residency` and
  `WorkloadResources.npu_slots`, not a preemption problem, and it deserves its own
  document.
- **GPU CU masking.** No support exists anywhere in the tree; the only `cu_mask` is
  in the XDNA packet (`hipfire-xdna/src/submit.rs:250`), NPU-side.
- **Collapsing the two collectors.** `CalibratableBackend::collect_calibration_job`
  (`calibration.rs:1266`) already takes the same `CalibrationJob` contract as the
  streamed path, so `CalibrationJob`, `SampleSet`, `CalibrationOptions`,
  `CaptureRegistry` and `CalibSummary` are shared today. What remains duplicated is
  the two forward strategies, and that difference is intrinsic (full-sequence
  prefill against layer-at-a-time). It is also load-bearing: the resident path is
  the streamed path's parity oracle, and the companion plan expects drift to
  *increase* under position-slice batching. Do not merge them while that
  measurement is still needed.

## Resolved: the AGENTS.md invariant

**Decided 2026-07-25 — amended.** Calibration is a forward pass, not a conversion.
The coexistence invariant now draws its line at format conversion rather than at
GPU work: a workload that runs kernels over model weights (calibration/induction,
Hessian and imatrix capture, KLD evaluation, training) is inference-shaped and may
live in the daemon where it can be scheduled, batched and preempted. Container and
format translation stays in `hipfire-coexistence`.

Phase 1 is therefore unblocked. The `calibrate` CLI stays where it is as the
daemon-free path, the same way `collect_artifacts` does.

## The daemon refactor this implies

The daemon is serial for a specific, nameable reason, and it is not the loop.
**Its entire state is `main()`'s stack frame** — 14 loop-carried mutable locals
(`daemon/main.rs:3150-3319`):

```
gpu, model, active_worker_id, resident_models, generic_state_arena,
pflash_state, pflash_cfg, cett_colnorms, pflash_drafter_gpu, dummy_model,
lora_train_session, drafter_train_session, resource_reservations, stdout
```

Every one of the 41 request handlers borrows these mutably in turn. Two handlers
cannot run concurrently because both need `&mut gpu`. The
`for line in stdin.lock().lines()` loop is a *consequence* of that ownership
structure, not the cause. So "make the daemon schedule and preempt" means:

1. Hoist the 14 locals into a state struct.
2. Choose a concurrency model. A single GPU-owning executor fed by a work queue is
   likely right over shared locks, since `gpu` is the contended resource and HIP
   contexts are not freely shareable.
3. Give long-running handlers interior preemption points. The two train sessions
   already have the right shape and are the template.
4. Move `ContinuousWorkScheduler` in.

Most of the 41 handlers change only from `&mut local` to `&mut state.field` —
mechanical, low-risk per site, high total churn across 8196 lines. This is a
multi-week structural change that wants to be staged with the serial path
preserved, not a rewrite.

**It does not block the near-term work.** Phases 0–2 land on today's serial daemon,
because a quantum-per-request session *is* preemption at the request boundary: the
server simply does not send the next quantum while something higher-priority is
waiting. The refactor buys **sub-request** preemption, which is what realtime
co-residency needs and what request-boundary yielding cannot provide. Staging:

| stage | preemption granularity | needs daemon refactor |
|---|---|---|
| Phases 0–2 | one quantum (request boundary) | no |
| Phase 3 | one dispatch / graph launch | yes |

Sequencing this way also means the refactor is informed by a real second scheduled
workload rather than designed against one.

## Validation

- Phase 0: assert the geometry chosen with a daemon resident differs from the
  geometry chosen without one, and that their sum stays inside the configured
  budget.
- Phase 1: byte-identical artifact from a run interrupted and resumed at a quantum
  boundary against an uninterrupted run. The checkpoint fingerprint already
  enforces the binary-identity half of this.
- Phase 2: measure worst-case blocking directly — run induction, inject a
  higher-priority workload, record the delay until it is dispatched. That number is
  the deliverable; the priority plumbing is only the means.
- Phase 2, graph scope: measure observed blocking with whole-step capture enabled
  and confirm it matches the declared bound. A declared WCET that does not account
  for an enabled graph is the specific failure this contract exists to prevent, so
  the test must run with capture *on*.
- Phase 2, metadata: assert the runtime-derived bound tracks a deliberate change to
  graph scope or batch size. A bound that stays constant across those is not being
  derived, and has silently become a baked scalar.
- The companion plan's block-0 numerical-identity gate stays in force. Nothing here
  changes the block math, and if it moves, something in the batching change is
  wrong rather than something here.
