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
- **Delete the dead scheduler API before moving it.** Done: 3018 → 2638 lines (−380).
  Removed `PriorityDecodeScheduler` and its whole cluster (`ActiveDecodeSession`,
  `DecodeBatchSelection`, `decode_sessions_compatible_for_batch`, `decode_state_kinds`,
  `inferred_decode_state_kinds`, `state_kinds_have_mamba`) plus
  `preview_next_prefill_batch`/`PreviewPrefillBatchInput`,
  `clamp_scheduler_priority_f64`, and `server_batch_health_json`.

  The original ~700–800 estimate was wrong, in two directions worth recording:
  - **`should_dispatch_opportunistic` and `parse_default_scheduler_priority` are live**,
    called from `select_from_bucket` and a health-JSON builder respectively. Having no
    caller *outside* the crate is not the same as being dead. They stay (candidates for
    demotion to private, not deletion).
  - **The ~60 hardcoded-zero health-JSON keys must not be deleted here.**
    `tests/smoke-server-decode-batch.sh` asserts `health.decode_batch.total_batches` and
    `selected_batch_size` — deleting them now would break the exact smoke test M4 uses as
    its exit gate. They get *replaced by real counters* in M4, not removed in M0.
- **The `?`-on-`Option` "head-of-line blocking bug" is not a bug — retracted.** It is
  deliberate priority protection, and it is test-locked: a test whose variable is literally
  named `blocked` asserts `next_prefill_batch(..).is_none()` while an interactive request is
  still inside its coalesce window with an opportunistic request queued behind it. Aborting
  the bucket scan is what stops the GPU being handed to opportunistic work microseconds
  before a higher-priority batch forms; cooperative preemption would not get it back
  cheaply. Two of the three reported sites were in code deleted above, and the third is
  correct as written. Do not "fix" this.
- **Fixed for real: `fair_ordered_prefill_bucket` was quadratic** (`:830`). It re-scanned
  the bucket with `filter(..).nth(i)` once per emitted element, and deep-cloned every
  queued request's two token vectors on every `next_prefill_batch` call — including via the
  `owners.len() <= 1` fast path, which did a whole-bucket `to_vec()`. Now a single grouping
  pass returning borrowed entries, so only the entries actually selected (at most
  `selection_limit`) are cloned. Ordering is unchanged and now pinned by a direct test
  covering the uneven-drain and rotation cases the scheduler-level tests never reached.

### M1 — `DaemonState` hoist

14 locals → one struct; all 48 arms go `&mut local` → `&mut state.field`. Loop unchanged,
still serial. Collapse the 25-line `activate_model_worker` block duplicated at
`main.rs:4173, 4226, 4283, 4951, 5074, 5141, 5189, 5270, 5427, 5485` into one helper.

*Exit:* byte-identical generation output; no behavior change.

### M2 — Protocol: id correlation + typed responses

Splits in two, and only the first half is a genuine M3 prerequisite.

**M2a — id correlation. Done.** Every frame now goes out through
`Responder::emit`, which stamps the current request id (an explicit `id` in the
frame wins, since batch and session ops answer per-envelope ids that are
deliberately not the request id). `Responder` is its own struct rather than two
fields on `DaemonState` because `emit(&mut self)` on the whole state would
conflict with handlers that legitimately hold a mutable borrow of another field
across an emit — the drafter loop emits per-epoch progress while holding
`&mut drafter_train_session`. It is also the seam M3 replaces: one `Responder`
per connection instead of one per process.

Two things this fixed that were not in the original scope:
- **85 error frames were emitted with an empty id**, i.e. uncorrelatable. (An
  early count said 13; `grep -o` undercounts because rustfmt wraps those calls
  across lines.) They now use `Responder::error`, which stamps.
- **13 frames were hand-written string literals** (`pong`, `reset`, `steer_ok`,
  `lora_ok`, `unloaded`) that bypassed `serde_json` entirely. `serving-core::events`
  already documented why that is unsafe — a user-controlled id containing `"` or a
  newline desyncs every following line — but the daemon's own literals did not
  follow it. All frames now serialise.

Backward compatible, which is what made it safe to do first: the adapter parses
with a plain `serde_json::from_str` and there is no `deny_unknown_fields` anywhere
in the protocol crate, so an added `id` is ignored by existing readers.

**M2b — typed `DaemonResponse` serialization. Deferred, and it is larger than it
looks.** The enum is `Deserialize`-only, no variant carries an `id`, and it ends
in `#[serde(other)] Unknown`. Serialising it means adding `Serialize` to a dozen
foreign payload types, deciding what `Unknown` does on the write side, and
inventing variants for the frames that currently fall through to it
(`train_start`, `train_epoch`, `pflash_labels_*`, `diag`, …). None of that is
needed for multiplexing — routing needs the id, not the type — so it should not
block M3.

*Exit (M2a, met):* ids observed on the wire for `pong`, `lora_listed`,
`resource_status`, `reset` and both protocol-level and handler-level error frames,
including the escaping case; existing callers unaffected.

### M3 — Socket transport + control channel

Two measurements set the shape here, and neither was obvious from the original
sketch:

- **The daemon has no async runtime at all** (zero tokio deps). So the front end
  is threads plus `std::os::unix::net::UnixListener`, not tokio tasks. That suits
  it: execution must stay on one thread anyway, and adding an async runtime to a
  fully synchronous binary would be a large change bought for nothing.
- **`stdout: &mut std::io::Stdout` appears in 42 signatures across 10
  `serving-core` files**, including `generate`. That concrete type — not the
  listener — is what actually blocks multiple clients, because per-connection
  writers cannot exist until it is abstracted. It is the real cost of M3.

Hence five steps, and execution stays on the **main** thread throughout:
`hipfire_rdna::Gpu` is initialised there and HIP contexts are thread-affine, so
moving the executor would mean moving GPU init with it. Moving the *reader*
achieves the same decoupling with none of that risk.

**M3a — split reading from executing. Done.** A named reader thread parses frames
and forwards them over a rendezvous channel (`transport.rs`); the main loop is now
an executor consuming it. Behaviour-preserving, and it creates the two things the
rest needs: somewhere for a scheduler to sit (M4), and a producer a socket
listener can join. Capacity is deliberately 0 so backpressure is exactly today's
— the OS pipe stays the queue; M4 raises it, because a scheduler needs several
pending requests before it has anything to choose between. Malformed input is
forwarded as a frame rather than reported by the reader, so responses keep a
single writer and stay ordered against the requests around them.

This is also where an end-to-end run caught a bug unit tests could not: parse
errors were inheriting the *previous* request's id, blaming a request that had
succeeded. The id is now assigned in exactly one place, from an exhaustive match
on the frame, so a new inbound variant cannot reintroduce it.

**M3b — abstract the response sink.** The 42 `&mut std::io::Stdout` signatures
become `&mut dyn Write` (or generic). Wide but mechanical; unblocks M3c and also
makes `Responder` unit-testable, which it currently is not.

**M3c — `UnixListener` + per-connection writers. Done.** `--listen [PATH]`
(default `~/.hipfire/daemon.sock`, mode 0600) serves many clients; stdio stays the
default so the six existing spawners are untouched. Each accepted connection gets
its own reader thread and its own `ReplySink`, which every frame carries; the
executor swaps `Responder::sink` per frame, so a reply always goes back to the
connection that asked.

`ReplySink` is an `Arc<Mutex<Box<dyn Write + Send>>>`. The mutex is effectively
uncontended — one request in flight, one writer — but it is what allows a handle to
be shared across the many frames of a connection when `Write` needs `&mut`. The
read and write directions use separate fds via `UnixStream::try_clone`, so a reader
can block on input while the executor answers.

A stale socket file is removed before bind, and the justification is the flock: we
already hold the exclusive lock on `daemon.pid`, so no live daemon can own that
socket, and bind would otherwise fail `EADDRINUSE` against a file nobody is
listening on. The lock is the authority on singleton-ness, not the socket file.

This is also where socket mode exposed a latent bug: `emit_load_progress` took a
*fresh process-stdout lock*, which was correct only while stdout was the sole
destination. On a socket connection every `load_progress` frame went to the
daemon's own stdout and the client loading the model saw nothing. It now writes to
the requesting connection's sink and carries the request id, which it never did.

*Exit (M3c, met):* two concurrent clients on one daemon, verified with interleaved
sends (B asks first, A second, each reply routed correctly), sinks surviving across
multiple frames per connection, one client disconnecting without taking the daemon
down, and socket mode 0600. stdio behaviour unchanged.

**M3d — control channel + real `Abort`. Done.** Reader threads consume `abort` /
`force_answer` themselves and record the ask in `hipfire_runtime::cancel`; the
frames never enter the queue. That is the point — with a rendezvous channel and a
busy executor, *enqueuing* an abort would block until the generation it wanted to
stop had already finished, which is exactly why the old reply pointed at a control
channel that did not exist. A control frame naming no request is still forwarded,
so the caller gets told rather than ignored.

Cancellation is a process global with an id guard rather than a token parameter.
`generate` takes 28 positional parameters and immediately delegates to one of
several decode paths, so a token would have to be plumbed through all of them and
every future one, and any loop that forgot to check it would silently ignore
aborts. The id guard is what makes the global safe: an abort names its request, so
a stale one cannot stop an unrelated successor, and the executor clears the slot as
it takes up new work.

**The plan's assumption here was wrong, and only a real model exposed it.** The
first implementation hooked `emit_filter_action` alone, on the stated belief that
every decode loop passes through it once per token. An end-to-end abort against
gemma3-vl then ran all 400 tokens to completion: the arch-generic
`arch::decode_loop_*` — which the gemma3 family and every factory/registered
backend use — never calls it. Cancellation is now checked at three places, and the
count is the finding:

1. `arch::decode_loop_*` (the generic loop) — the one that mattered here.
2. `events::emit_filter_action` — the older inline loops in `generate.rs`.
3. `generate_deepseek4`'s own loop, which streams via `emit_stream_event` and
   reaches neither of the above.

*Exit (M3d, partially met):* verified end-to-end on gemma3-vl — abort sent from a
*second connection* against a generation on the first, 12 tokens emitted of a
400-token request, stopped **0.16 s** later (one decode step at 6.3 tok/s, i.e.
one-token granularity), terminal frame `finish_reason: "stop"` with `aborted: true`,
and the daemon still serving afterwards. Paths (2) and (3) are hooked but **not**
yet verified against a live model — that needs a qwen35 and a deepseek4 run
respectively.

**M3e — attach instead of spawn. Capability done; only one caller could migrate.**

`SocketTransport` + `DaemonEngine::connect(path)` + `attach_or_spawn(bin)` land in
the adapter, and `default_socket_path()` now lives there so the listening end and
the connecting end cannot disagree about where the door is.

`hipfire chat` migrates, and it is the caller that was actually broken: it spawned
a private daemon, which tried to take the same exclusive `daemon.pid` flock a
running `hipfire serve` already held, so **chatting while serving failed outright**.
Verified end to end — spawning a second daemon still returns
`hipfire daemon already running (PID …)`, while `hipfire chat` against the same
live daemon now attaches and generates.

**The other seven cannot migrate yet, and the reason is a finding rather than an
oversight.** They split into two kinds:

- **Blocked on configuration channel** — `server/lib.rs` and `server/routes/chat.rs`.
  Both configure the daemon through the environment the *spawned child* inherits:
  `apply_daemon_startup_env` sets `HIPFIRE_RESOURCE_LOCK` and the scheduler memory
  budgets, and the chat route builds a per-model `daemon_spawn_env` (the DFlash
  n-gram override). Attaching to an already-running daemon would silently drop all
  of it and leave the daemon on whatever config it started with. **Daemon
  configuration has to move from spawn-time env onto the wire before the server can
  attach** — and that is the same constraint M4 hits, since a scheduler in the
  daemon needs config it can change at runtime, not only at exec.
- **Correctly private by design** — `cli/bench.rs`, `eval/executor_daemon.rs` (×2),
  `coherence/lib.rs`, `steer-harness` (×2). These build a daemon and must measure
  *that* build; attaching would silently exercise whatever was already running, and
  a shared daemon would have another model resident. The coherence gate literally
  rebuilds the daemon before running it. Each site now carries a comment saying so,
  so the next person does not "finish the migration" and quietly break the gates.

*Exit (M3e, partially met):* attach capability verified against a live daemon,
including that the pre-existing spawn collision still reproduces. Six of eight
sites stay on `spawn` deliberately; the two server sites are blocked on wire
configuration.

*Exit (M3a, met):* identical observable behaviour on stdio, including blank-line
skipping, malformed-frame ordering and EOF shutdown; reader unit-tested for
forwarding, blank skipping, deferred parse errors and executor hangup.

*Exit (M3 overall):* two concurrent clients on one daemon; an abort mid-generation
actually stops it.

### M4 — Scheduler moves in

**M4a — daemon configuration on the wire. Done.** M3e found that the server
configures the daemon through the spawned child's environment, which blocks both
attaching *and* a daemon-resident scheduler that needs runtime-changeable config.
Examining what is actually passed splits it three ways, and only one third needed
moving:

| kind | settings | movable? |
|---|---|---|
| exec-time | `HIPFIRE_DEVICES`, `HIPFIRE_RESOURCE_LOCK*` | **no** — consumed before `Gpu::init()` and before the process takes its flocks, so they describe locks already held |
| runtime | the four `HIPFIRE_SCHEDULER_*_BYTES` budgets | **yes** — they only size the ballast, which `release_placeholders`/`reacquire_placeholders` already re-apply |
| per-request | `HIPFIRE_DFLASH_NGRAM_BLOCK`, `HIPFIRE_NORMALIZE_PROMPT` | belongs in the request, not in config |

So `set_resource_budget` carries the middle row: optional fields (an omitted field
leaves that budget alone), applied by release → update → reacquire, answering with
the same payload as `resource_status`. The release/reacquire pair is the point —
changing the numbers without re-applying would leave the daemon holding the old
reservation while reporting the new budget.

An attaching caller therefore pushes its budgets and accepts the daemon's locks as
given, which is the correct division: the daemon holds those locks.

Verified against a live daemon: 4 GiB budget / 1 GiB headroom → 3 GiB actually
held; a headroom-only update left the budget intact and moved held to 2 GiB;
restoring zero released it. Partial-update semantics are unit-tested, because a
caller adjusting one field must not silently zero the others and drop the whole
reservation.

*Still blocking a server attach:* the per-request row. `routes/chat.rs` sets
per-model DFlash/normalize behaviour by spawning a daemon with that environment, so
it needs those as request fields before it can attach.

**M4b — the daemon chooses. Done.** `hipfire-scheduler` is now a daemon dependency
(for its priority scale, so both ends mean the same thing by a number), requests
carry an optional `priority`, and a `PendingQueue` sits between the readers and the
executor. The inbound channel goes from rendezvous to bounded-256: a scheduler needs
several frames *in hand* to reorder, and at capacity 0 a long generation blocked
every reader so nothing could queue behind it.

**The invariant that makes reordering safe is per-connection ordering.** A
connection's frames are a dependency chain — `reserve_session_state` →
`generate_batch_prefill` → `generate_batch_decode_step` → `release_sessions`, or
`load` → `generate` → `unload` — and nothing in the protocol declares those
dependencies: the order *is* the declaration. So the queue holds one FIFO per
connection and chooses only among their heads. Priority decides which *client* goes
next, never which of a client's own requests goes next. Equal priority falls back to
arrival order, so connections cannot starve each other by id.

A consequence worth stating: with one connection — all of stdio, and any single
socket client — selection is a no-op and behaviour is exactly FIFO as before.
Reordering only becomes observable with concurrent clients, which is also the only
situation where it is safe.

*Exit (M4b, met):* five ordering invariants unit-tested, and demonstrated live —
five bulk (priority 200) frames queued on one connection *before* five realtime
(priority 0) frames on another, both behind a 25 s model load; the daemon chose
`[0,0,0,0,0,200,200,200,200,200]`.

**A measurement trap worth recording, since it produced two false failures.**
Client-side reply order does NOT report service order. Replies written microseconds
apart race through separate sockets, so with one watcher thread per connection the
recording reflects thread wake order, and with a single `select()` loop it reflects
iteration order of the fd list. Both said "arrival order, no reordering" while the
daemon was in fact reordering correctly. Scheduling decisions are only observable
from inside, which is why the executor now has an env-gated `[sched]` trace
(`HIPFIRE_DAEMON_SCHED_DEBUG=1`) and why the test asserts on that.


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

**M4c — real scheduler telemetry. Done.** The daemon counts what it schedules —
`scheduled_total`, `queue_depth`, `queue_depth_max`, per-priority-class totals, and
`overtaken_total` — served over a new `scheduler_status` request that `/health`
surfaces as `daemon_scheduler`.

`overtaken_total` is the counter that earns its place: it counts frames chosen ahead
of an older waiting head, so it is the only figure distinguishing a working scheduler
from an idle one — totals and depth look identical either way. It deliberately does
NOT increment when equal-priority work is served in arrival order, since that is
plain FIFO and counting it would make the scheduler look busy while doing nothing.

**The hardcoded zeros in `prefill_batch` / `decode_batch` / `state_cache` are left
alone on purpose.** They describe SERVER-side batching, which has not moved yet;
back-filling them with daemon numbers would report one subsystem's activity under
another's name, and `tests/smoke-server-decode-batch.sh` asserts on those exact keys.
They become real in M4d when the batch runner crosses over. An honest new block beats
making a misleading old one look populated.

`/health` degrades rather than fails: an absent or unreachable daemon reports
`{"available": false, "reason": …}`, because that is a fact worth showing and the
endpoint has to answer regardless.

*Exit (M4c, met):* verified live — four priority-200 pings and two default-priority
status queries classified `bulk: 4` / `interactive: 2`, `overtaken_total: 0` on a
single connection (no phantom overtakes), complementing the M4b run where
cross-connection priority produced real ones.

*Remaining for M4:* the server-side batching figures above, which need M4d.

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
