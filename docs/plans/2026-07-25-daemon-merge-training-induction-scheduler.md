# Merge training, induction, and the scheduler into hipfire-daemon

Status: scope, 2026-07-25. Supersedes the deferred "The daemon refactor this
implies" section of
`docs/plans/2026-07-25-induction-scheduling-and-realtime-wcet.md`, and closes
P5.3 of `docs/plans/2026-07-18-continuous-scheduler-headline.md`.

Four scoping decisions were taken by the user on 2026-07-25 and are treated as
settled premises below, not open options:

1. **Transport:** the daemon becomes a socket service; `StdioTransport` is kept
   for tests and one-shot runs. Clients attach instead of spawning.
2. **Diffusion is in scope** — image generation moves into the daemon as a
   fourth workload class rather than keeping an external-turn lease.
3. **All three training paths collapse** onto the daemon session; no daemon-free
   training path remains.
4. **Target architecture first**, migration path second — hence the two-part
   shape of this document.

## Context

Three complaints drive this, all traceable to one cause:

1. Inducting a model **stops** inference instead of dropping the induction job's priority.
2. Memory limits are not shared between the daemon config and induction.
3. Several code paths are maintained where one would do.

The cause is that **policy and mechanism sit on opposite sides of a process boundary.**
`ContinuousWorkScheduler` lives in `hipfire-server` (`state.rs:90`) and can only decide
*which whole JSONL request to send next*. The daemon owns the GPU and is the only place
that can preempt *within* a request. Meanwhile induction (`hipfire-coexistence calibrate`)
and 74 `hipfire-train` examples each take their own GPU lock and are invisible to the
scheduler entirely.

Prior work this builds on — read both before starting:
- `docs/plans/2026-07-25-induction-scheduling-and-realtime-wcet.md` — the WCET contract,
  the memory-budget unification, and why induction must not copy `Dispatch::Train`.
- `docs/plans/2026-07-18-continuous-scheduler-headline.md` — the phased scheduler buildout
  this completes.

Intended outcome: **one process owns the GPU and arbitrates every workload on it** — text,
embed/steer, image, training, induction — with preemption at dispatch granularity and a
single memory budget.

### What already exists (do not rebuild)

Scoping found far more in place than the framing suggests:

- **The daemon already calibrates.** `DaemonRequest::Collect`
  (`hipfire-daemon/src/main.rs:5767`) runs the resident collector in place. The
  AGENTS.md line has already been crossed; only the *layer-streamed* engine is outside.
- **Training is already a daemon op.** `TrainLora` (`main.rs:7512`) and `TrainDrafter`
  (`main.rs:7207`) are resident, `run_id`-keyed, quantum-per-request sessions.
  `hipfire-train` is already a daemon dependency.
- **The transport seam exists.** `trait DaemonTransport: Send`
  (`hipfire-daemon-adapter/src/lib.rs:103`) sits under all 45 `DaemonEngine` request
  methods; a socket transport swaps in without touching any of them.
- **The scheduler is trivially relocatable.** `hipfire-scheduler` is one 3018-line file,
  pure sync, with zero GPU/lock/async/env/IO coupling. Moving it is `mv` + a Cargo edit.
- **Induction's quantum machinery exists on both sides.** `--pause-after-layers`
  (`calibrate.rs:389`), `CalibrationRunOutcome::Complete | Paused` (`:1286`), `--resume`
  as the default (`:317`), and a fingerprint-guarded checkpoint that refuses a resume from
  a mismatched binary (`:1535`, `calibration/boundary.rs:171`).
- Every `CalibrationFamilyAdapter` already takes `gpu: &mut Gpu` as a parameter
  (`runtime/calibration/stream.rs:408`), so **moving the driver requires zero adapter changes.**

The real work is not new machinery. It is the daemon's ownership structure.

---

# Part 1 — Target architecture

## 1.1 Why the daemon is serial

Not the `for line in stdin.lock().lines()` loop (`main.rs:3319`). **Its entire state is
`main()`'s stack frame** — 14 loop-carried `let mut` locals at `main.rs:3270-3308`:

```
gpu, model, active_worker_id, resident_models, generic_state_arena, pflash_state,
pflash_cfg, cett_colnorms, pflash_drafter_gpu, dummy_model, lora_train_session,
drafter_train_session, resource_reservations, stdout
```

All 48 request arms borrow these mutably in turn, so two can never run concurrently. The
loop is a *consequence* of that ownership, not the cause.

## 1.2 The executor stays serial — deliberately

The obvious target ("make the daemon concurrent") is wrong. Process-global mutable state
forbids interleaving two GPU ops today:

| global | site |
|---|---|
| `hipfire_steer` capture/apply session | process-global; defensively cleared at `main.rs:3380`, `:5706` |
| `RAW_OVERRIDE` | `serving-core/src/model.rs:204`, thread-local `Cell`, set per request at `main.rs:4272` |
| sampler RNG reseed | `serving-core/src/generate.rs:1907`, reseeded at the top of every `generate()` |
| `load_progress::set_sink` | global sink installed for the duration of a load (`main.rs:3799`) |
| `latent_kv` projectors | `hipfire-train/src/latent_kv.rs:40` |

So the target is a **single GPU-owning executor** plus a **multi-client front end**:

```
 socket listener ──┬── conn task ──┐
                   ├── conn task ──┼──► inbound queue ──► ContinuousWorkScheduler
                   └── conn task ──┘         │                      │
                                             │                      ▼
   control channel (ping/abort/status) ◄──────┘        ONE executor thread
        answered without touching the executor          owns DaemonState (incl. Gpu)
                                                        exclusively — no locks
```

What changes is not parallelism. It is that **requests now queue inside the daemon**, where
the scheduler can reorder them and the executor can yield between dispatches — and that a
**control plane exists that does not queue behind GPU work.**

That last point makes `Abort`/`ForceAnswer` real for the first time. Today they are dead
wire variants that reply *"handled on the control channel, not the request channel"*
(`main.rs:8185`) — and there is no control channel, so an abort can only be read after the
generation it wants to cancel has already finished.

## 1.3 Crate structure

`main.rs` is 8196 lines in one flat file; `main()` alone is 5019 lines (`:3178-8196`), of
which ~4830 is the request `match`. Target split:

| module | contents |
|---|---|
| `main.rs` | arg parse, locks, transport bind, executor spawn. Thin. |
| `state.rs` | `DaemonState` — the 14 hoisted locals |
| `executor.rs` | the single GPU-owning loop (`batch_runner_loop` lands here) |
| `transport/{stdio,socket}.rs` | listener, framing, id correlation |
| `handlers/{generate,load,batch,steer,lora,calibrate,kld,train,image,diag}.rs` | the 48 arms by family |
| `sessions/{train,induct}.rs` | resumable session types |

**Blocker to clear first:** a 2141-line `#[cfg(test)] mod generate_batch_prefill_tests`
sits at `main.rs:974-3115`, *between* the helpers and `main()`. It straddles every cut and
must be relocated before any split.

## 1.4 Workload classes and the WCET contract

Six classes on one executor: `TokenPrefill`, `TokenDecode`, `Embed`/`Steer`, `Image`,
`Training`, `Induction`. `WorkloadClass` (`scheduler/lib.rs:239`) already has `Training`;
`Maintenance` is declared and never constructed — reuse or replace it for `Induction`.

Preemption has two tiers:

| tier | granularity | needs the executor? |
|---|---|---|
| quantum / request boundary | one training quantum, one calibration layer, one sampler step | no — works on today's serial daemon |
| dispatch / graph launch | one kernel or one captured `hipGraph` | **yes** |

Per the prior doc, every workload declares the duration of its **largest indivisible unit**
plus the **maximum yield granularity it tolerates**, both defaulting to unbounded so
nothing silently claims to be co-schedulable with realtime work. `SchedulerPriorityClass::Realtime`
(`scheduler/lib.rs:706`) already exists, unused, as the carrier.

## 1.5 Transport

- Server side gains a `Listener` abstraction and a `SocketTransport` over
  `~/.hipfire/daemon.sock`, mode 0600 — same-uid only, matching the existing
  `admin.secret` trust model. No new auth surface.
- `StdioTransport` stays for tests, one-shot runs, and `--precompile`.
- **The hard part is frame correlation.** The daemon does not serialize `DaemonResponse` —
  it hand-writes `writeln!(stdout, r#"{{"type":"pong"}}"#)`-style literals, and many frames
  carry no request id. The contract is enforced only on the read side. Multiplexing N
  clients requires an id on every frame, which means finally serializing `DaemonResponse`
  across all 48 arms' write sites. **This is the largest mechanical risk in the plan.**

## 1.6 Lock and memory model

- The daemon keeps its two process-lifetime guards (`main.rs:3256-3257`): the
  `daemon.pid` singleton (`main.rs:850`) and the resource lease
  (`adapter/lib.rs:1455`). No per-request locking.
- **`calibrate`'s self-lock is deleted on the daemon path.** `calibrate.rs:693` does
  `lock_blocking(2s, None)` — an unbounded wait — which is why
  `scripts/two_pass_quantize.py:433` documents that wrapping it deadlocks. The daemon-free
  CLI keeps it.
- Both `calibrate` and the qtip encoder currently violate AGENTS.md's *"non-daemon GPU
  binaries do not self-lock"* clause. Folding induction in fixes the first for free.
- **Latent deadlock to fix independently:** `hipfire-quantize` self-locks on the qtip path
  (`main.rs:4356`, called `:4442`, `#[cfg(feature = "gpu")]`) while
  `two_pass_quantize.py:437` wraps the quantizer in `hipfire lock run` — a *parent* process
  holding the flock. `--format qtip3`/`qtip4` under two-pass would hang. It does not fire
  today only because the induction default is `oq4.25++`.
- One memory budget: `ResourceReservationManager` (`main.rs:424`) becomes the single
  authority, and `tune_geometry_for_gpu` (`calibrate.rs:3006`) sizes from
  `budget − daemon_target` instead of a live `get_vram_info()` probe that is accurate only
  at the instant it runs.
- `plan_model_residency` (`scheduler/lib.rs:154`) is the real residency planner and the
  daemon never calls it. Wire it; drop the ballast allocator's estimate-VRAM-as-file-size
  heuristic (`main.rs:494`).

## 1.7 What stays out

- **`hipfire-quantize` stays standalone.** It is CPU/rayon (80% of cores,
  `main.rs:5944`); its only GPU use is an optional offline Viterbi encoder with a complete
  CPU fallback. It rewrites bytes between formats — the coexistence side of the line.
- **`hipfire-coexistence` keeps everything index/bytes:** `import_safetensors.rs`,
  `artifact.rs`, `calibration_audit.rs`, `calibration_compare.rs`, `residual_compare.rs`,
  `router_profile.rs`, and every `arch-*/ingest.rs`. Zero GPU in any of them.
- The `calibrate` CLI stays as the daemon-free path, the way `collect_artifacts` does.

---

# Part 2 — Migration path

Ordering is chosen so the serial path keeps working throughout and each stage is
independently revertible.

### M0 — Preparation (no behavior change)

- Relocate the 2141-line test module out of `main.rs:974-3115`.
- Split `main.rs` into the §1.3 modules with `main()` still holding the locals — purely
  mechanical `&mut` threading.
- **Delete the dead scheduler API before moving it:** `PriorityDecodeScheduler` entirely
  (never constructed), `preview_next_prefill_batch`, all `SchedulerPriorityPolicy` fields
  (no caller reads one), ~60 hardcoded-zero health-JSON keys. That is ~700–800 of 1934
  production lines — a third less to migrate.
- Fix two bugs found in passing: `?`-on-`Option` head-of-line blocking at
  `scheduler/lib.rs:1653`, `:1690`, `:1904`; and O(bucket²) `RequestSessionDraft` cloning
  in `fair_ordered_prefill_bucket` (`:837`).

### M1 — `DaemonState` hoist

14 locals → one struct; all 48 arms go `&mut local` → `&mut state.field`. Loop unchanged,
still serial. Collapse the 25-line `activate_model_worker` block duplicated at
`main.rs:4173, 4226, 4283, 4951, 5074, 5141, 5189, 5270, 5427, 5485` into one helper.

*Exit:* byte-identical generation output; no behavior change.

### M2 — Protocol: id correlation + typed responses

Serialize `DaemonResponse` instead of hand-writing frames; stamp the request id on every
frame. Prerequisite for M3.

*Exit:* existing single-client callers unaffected.

### M3 — Socket transport + control channel

`Listener` trait, `SocketTransport`, front-end connection tasks feeding an inbound queue,
executor thread owning `DaemonState`. Implement `Abort`/`ForceAnswer` for real — which
requires threading a cancellation token into `generate()` (`serving-core/generate.rs:1875`,
28 positional params, no cancellation today). Retire the six independent
`DaemonEngine::spawn` sites (`server/lib.rs:339`, `server/routes/chat.rs:632`,
`cli/chat.rs:212`, `cli/bench.rs:68`, `eval/executor_daemon.rs:174` and `:417`,
`coherence/lib.rs:528`, `steer-harness/lib.rs:92`) in favour of attach.

*Exit:* two concurrent clients on one daemon; an abort mid-generation actually stops it.

### M4 — Scheduler moves in

`hipfire-scheduler` becomes a daemon dep. `batch_runner_loop`
(`hipfire-server/src/batch_runner.rs:464`) moves to `daemon/src/executor.rs` with its LIFO
`parked` stack and `Dispatch` classification. `batch_inbox` becomes the daemon's inbound
queue — and the id↔payload split across the process boundary disappears, because the
payload *is* the job now.

`AppState` splits cleanly. Moves out: `engine`, `loaded_model_*`, `loaded_models`,
`prefill_scheduler`, `work_scheduler`, `selected_prefill_requests`, `prefill_dispatch`,
`prefill_notify`, `batch_inbox`, `batch_runner_active`, `batch_telemetry`. Stays: everything
HTTP-shaped — `responses_*`, `files`, `batches`, `sdapi_*`, `models_dir`, `admin_*`,
`access`, `rate_limiter`, `usage_writer`.

Add a lease reaper while you are here: `complete(lease_id)` must be called exactly once and
has no timeout, so a dropped exclusive `Training` lease wedges `next_batch` forever
(`scheduler/lib.rs:542`). Today that is held together by discipline across 10 call sites.

*Exit:* `/health` scheduler telemetry served from real daemon counters, not the current
hardcoded zeros and `"fallback_reason": "rust_server_scheduler_metadata_only"`.

### M5 — Training collapses to one path

- Split `train_dspark_loop` (`hipfire-train/src/dspark_train.rs:776`) into
  `init`/`step`/`finish`, matching `train_loop.rs:165/209/300`.
- Drafter quantum is **epochs**, not steps (`main.rs:7418`) — hoist the inner `i` into
  `DrafterLoopState` alongside `ep` to get a step-sized quantum.
- Add an `SsmDrafter` on-disk checkpoint. Today `PFDC`/`DSCK` are keyed to the *legacy*
  `Drafter` type and the daemon drafter op has no checkpoint at all, so **no run survives a
  daemon restart.**
- Give `train_lora` a real data loader — it hard-errors on anything but `data=overfit`
  (`main.rs:7583`).
- **Close the observability disconnect.** The daemon emits `train_start`/`train_epoch`/
  `train_progress`/`train_done` to stdout; the operator API and TUI read `status.json` +
  `events.jsonl` under `~/.hipfire/training/runs/<id>/` (`hipfire-operator/src/training.rs:15-18`).
  The only writer today is `deferred_jobs.rs:588`/`:631`, which shells out to a subprocess
  and records start/end only. Have the daemon write the operator schema directly from the
  existing `on_epoch` callback; then delete the coarse writer.
- Route `training_command` to a daemon session; make the `hipfire-train` CLI a socket client
  (M3 is what makes this possible); delete direct crate use.
- Delete the 44 non-`gradcheck` examples (~14.5k LOC of shadow production surface, each
  `Gpu::init()`-ing with **no lock**, racing the daemon today). Keep the 30 `gradcheck_*`
  as correctness tests. Drop the unused `ratatui`/`crossterm` deps.

*Note:* merging does **not** share the forward pass. `hipfire-train` owns an un-fused fp32
autograd and deliberately does not differentiate the fused inference kernels
(`hipfire-train/src/lib.rs:8-11`). Two numeric paths become resident in one process.

### M6 — Induction as a daemon session

- Sessionize `calibrate::run_cli` (`calibrate.rs:534`) into `new`/`step`/`finish`, modelled
  on `LoraTrainSession` — **not** on `Dispatch::Train`, which holds the runner turn to
  completion. `--pause-after-layers` becomes the quantum; the layer loop at `:2348` is the
  yield point.
- Delete the self-lock on the daemon path (`calibrate.rs:693`).
- Unify the memory budget (§1.6) — worth landing on its own, ahead of everything else.
- Port the load-bearing Python: ~700–900 semantic lines of the 2200 in
  `scripts/induct_model.py` + `scripts/two_pass_quantize.py`. The centrepiece is
  `run_calibration_pass` (`two_pass_quantize.py:696`), which is already a process-level
  quantum scheduler — respawn the binary, assert `completed_layers` advanced, sleep, repeat.
  The daemon session replaces it directly. Also port `recipe_manifest`/`update_manifest`,
  the recipe fingerprint gating, and `pass_two_storage_preflight` (which reimplements the
  quant format's byte math in Python). Build glue and `subprocess` shells over
  `coexistence artifact` can stay Python.
- **Depends on** `docs/plans/2026-07-25-zaya-streamed-calib-position-batching.md` landing
  first. ZAYA's streamed inner loop is currently one token at a time
  (`arch-zaya/src/calibration_stream.rs:1475`) at 0.07% of bf16 peak, and the WCET argument
  is derived from the batched dispatch shape.

### M7 — Diffusion as the fourth class

`hipfire-diffusion` becomes a daemon dep; image generation becomes a daemon op. The
`ImageTurn` grant and preempt `AtomicBool` (`batch_runner.rs:773-796`) become daemon
-internal instead of a cross-process handshake, and `state.diffusion_pipelines` moves.
`routes/sdapi.rs` is 8866 lines — expect the RPC surface (txt2img/img2img + progress) to be
the bulk of this stage.

### M8 — Dispatch-level preemption and the WCET contract

Yield checks between dispatches in the induction and text paths; declared-WCET and
max-yield-granularity fields on `WorkloadSpec`; WCET *primitives* (largest dispatch bytes
and FLOPs, expert dims, graph scope) emitted into `.hfq` at quantization time — **measured
on the serving path, not the induction path**, since the calibration forward is not the
serving forward. Lower the `Train` default quantum from 25 (`routes/train.rs:86`).

---

## Top risks

1. **Memory, not concurrency, is the leading risk.** `GpuTensor` has no `Drop` — every free
   is manual, and `model.rs:266` documents ~2 GB/forward accumulation if one is missed.
   `LoraTrainSession` holds the entire base model in fp32 on device (4 B/param — a 7B model
   is ~28 GB) and would now be resident alongside a served model in the same process.
2. **Response-contract churn (M2).** Hand-written frames across 48 arms, with the contract
   enforced only on the read side.
3. **Lease bookkeeping has no reaper.** After M4 a panicking handler wedges every workload
   rather than one request.
4. **Process globals cap the design.** They are why the executor must stay serial; if any
   future stage wants true intra-daemon parallelism, these must be eliminated first.

## Verification

- **M1:** byte-identical generation output before/after the hoist; `./tests/no-gpu-ci.sh`.
- **M3:** two clients concurrently attached; a `generate` aborted mid-stream stops within
  one decode step. Assert the daemon still refuses a second instance
  (`daemon.pid` singleton).
- **M4:** `tests/smoke-server-decode-batch.sh` — already written and currently failing at
  `decode_total_batches < 1`. It asserts real `health.decode_batch` / `health.prefill_batch`
  counters, so it is the natural exit gate.
- **M5:** a training run interrupted by a daemon restart resumes from checkpoint; the
  operator API and TUI show a live in-daemon run (impossible today).
- **M6:** byte-identical calibration artifact from a run interrupted and resumed at a
  quantum boundary vs. an uninterrupted run — the checkpoint fingerprint
  (`calibrate.rs:1535`) already enforces the binary-identity half. Assert the geometry
  chosen with a daemon resident differs from the geometry chosen without one, and that the
  sum stays inside the configured budget.
- **M8:** the deliverable *is* a number — run induction, inject a higher-priority workload,
  record the delay until it is dispatched. Run it with `hipGraph` capture **on**, since a
  declared WCET that ignores an enabled graph is the exact failure the contract exists to
  prevent.
- Throughout: `./tests/tiny-affected-gate.sh --require-coverage` for runtime changes; the
  companion plan's block-0 numerical-identity gate stays in force.

## Docs to update as part of the work

- `AGENTS.md` — the coexistence invariant was amended 2026-07-25 (line at format
  conversion, not GPU work); the branch/`chaingun` model is stale.
- `crates/hipfire-daemon/AGENTS.md` — claims lock paths `/tmp/hipfire-gpu.lock` and
  `/tmp/hipfire-resource-locks/<resource>.lock`; the code resolves `~/.hipfire/locks/hip-gpu-0.lock`
  (`hipfire-lock/src/lib.rs:201-222`).
- `docs/continuous-orchestrator.md` — its "Still to connect" list is what M4 closes.
- `docs/DEFERRED-JOBS.md` — `training_command` semantics change in M5.
- `docs/MODEL-INDUCTION.md` — the Python orchestration surface shrinks in M6.
