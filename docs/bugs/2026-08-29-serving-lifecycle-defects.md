# Four serving-path lifecycle defects

Status: found 2026-08-29, master `0c9e3d252`, nix1. **All four confirmed 3/3 by
independent verification. Three FIXED the same day; #2 (`aging_ms`) remains open
and needs a design decision.** Grouped because they share a shape: state entered on
one path and never unwound on an alternate exit.

In each case the verification pass **corrected the finder's scenario**; the
corrected version is what is recorded here.

---

## 1. [FIXED] `run_batch_cycle`'s early exits skip `release_sessions` (high)

`crates/hipfire-server/src/batch_runner.rs:1165` (park) and `:979` (prefill error)
both return before the only `release_sessions` call at `:1171-1173`.

On the park exit, sessions that **finished earlier in the same loop** are dropped
from `active` (`:1014`), so they appear in no later `specs` and are never released
for the life of the loaded worker. On the prefill-error exit, **all N sessions of
the batch** leak — not a subset: the server always sends fresh prompt sessions
(`routes/chat.rs:1854`, `resume_position: None` at `:1861`) and resumed batches
skip prefill entirely (`batch_runner.rs:969`), so the reachable triggers are
errors *after* the activation loop completes — `qwen35_prefill_suffix_batch`
(`qwen35_prefill.rs:2014`, e.g. the KV-budget error at `:788-795`) and the
checkpoint `?` at `:2029-2035`.

`qwen35_release_sessions` (`session.rs:1627`) is the only **per-session** reclaim;
the only other is a whole-model reset (`handlers/lifecycle.rs:982`
`sessions.clear()`). So a leak persists until the model is unloaded.

This contradicts the invariant stated by commit `5250ac25a` and by the comment at
`batch_runner.rs:1026` (*"KV is freed with the batch at cycle end"*), and the
per-session-release design in `docs/plans/2026-07-18-continuous-scheduler-headline.md:211`.

Parking still-**active** sessions resident is deliberate (`:933-937`) and is not
the bug.

**FIXED** exactly that way: the park arm releases `specs` minus the ids actually
parked (using `parked`, not `active`, so the filter_map's dropped entries are freed
too), and the prefill-error arm releases all of `specs`. `qwen35_release_sessions`
ignores unknown ids, so releasing a never-activated session is a no-op. The stale
comment at `batch_runner.rs:1026` — which asserted KV "is freed with the batch at
cycle end" — was corrected to say what the park exit actually does now.

---

## 2. [OPEN — needs a decision] Image workloads starve: `aging_ms = 0` kills the override (high)

`crates/hipfire-server/src/state.rs:294` builds `work_scheduler` with
`aging_ms = 0`, so `next_seed`'s aged-oldest override
(`crates/hipfire-scheduler/src/lib.rs:836-840`) never fires: with `aging_ms` zero
nothing is ever "old enough", `oldest` stays `None`, and seeding falls through to
the plain lowest-bucket scan. A priority-128 image is therefore never seeded while
any sub-128 workload is queued.

On top of that, `denoise.rs:614` aborts the run the first time the preempt flag is
seen past step 4, and `sdapi.rs:658-712` restarts **from the seed**. So the image
needs one entire uninterrupted sampling window (28 steps) with zero lower-priority
arrivals; any single chat request inside that window discards all progress.

Only **txt2img** is affected — `execute_hfq_diffusion_img2img` (`sdapi.rs:739+`)
never reads `batch_runner_active` and still uses the legacy direct `spawn_blocking`.

**The finder's primary fix is wrong and must not be applied.** Setting
`aging_ms > 0` only reorders the queue: a freshly-arrived priority-64 chat still
makes `peek_next_priority` return 64, the watcher still preempts at step 4, and
the image still restarts. Worse, a frozen `enqueued_at` is permanently the oldest,
so once aged the image always wins the override and starves the text instead.

**Fix:** cap consecutive restarts per image job and let it run to completion once
the cap is hit, and/or make the sampler resume from `completed_steps` rather than
discarding to the seed.

---

## 3. [FIXED] Executor v2 swaps two clients' answers (medium)

`crates/hipfire-daemon/src/handlers/generate.rs:901` stashes the parked generation
into the session's live stream **without checking whether admission was refused**.
`request_id` is written only at admission (`stream.rs:202`), so a refused second
frame (`stream.rs:199/335`) overwrites stream S's generation while S keeps
request A's id.

Result is an answer **swap**, not a hang for A: the march loop emits every token
and the terminal `done` tagged `id = A` (`lib.rs:1392-1395`, `:1466`, `:1505-1510`)
carrying **B's** output. A's own generation is dropped without `finish`/`fail`, so
A's real answer is lost. **B** is the request that hangs — it is never tagged or
terminated anywhere.

Reachability is narrow: gated on `HIPFIRE_DAEMON_EXECUTOR=v2`, which is default-off
(`stream.rs:44-56`) and set by nothing in the tree except
`tests/harness/listen-preempt-probe.py:11`. It also needs an explicit non-empty
`session_id` — `session_of` (`stream.rs:135-143`) falls back to `msg["id"]`, so
session-less generates get distinct keys. `hipfire-server` cannot trigger it
(`routes/chat.rs:1776` uses a per-request UUID and never sends `session_id`). The
reachable caller is the daemon stdin / `--listen` JSON-lines protocol directly.

The trigger is wider than pipelined frames: any second `generate` naming a session
whose stream is still marching hits it, since the stream stays in `by_session`
until `Outcome::Done`/`Retired` (`lib.rs:1500-1516`) — a client retry, or the next
turn after a client-side timeout, is enough.

**FIXED** exactly that way: `handlers::generate::text` now takes the
`Option<StreamId>` that `admit_generate` returned and stashes only into that id via
`streams.get_mut(id)`. A refused frame falls through to the inline drive, which is
what the flag-off path does anyway. It also no longer parks when admission was
refused, since parking is a state change the inline path never wanted.

---

## 4. [FIXED] `selected_prefill_requests` grows without bound (low)

`crates/hipfire-server/src/routes/chat.rs:1701` inserts every selected request id;
removal happens only on the request's own path (`:1679`, `:1688`). There is no
other mutator and no eviction, so an id whose future is dropped mid-wait is
inserted by a *different* concurrent request's `next_prefill_batch` and never
removed.

**The finder named the wrong route.** Non-streaming `/v1/chat/completions` cannot
leak: `blocking_chat` (`chat.rs:2101`) `tokio::spawn`s the executor at `:2107` with
cooperative `tx.is_closed()` cancellation, so a disconnect leaves the task running
and it removes its own id. All three callers of `wait_for_prefill_scheduler_turn`
sit inside spawned tasks.

The genuinely drop-cancelled routes are the handlers that await **inline**:
non-streaming `/v1/responses` (`lib.rs:181` → `responses.rs:173` → `:282` →
`chat.rs:1945`) and the sdapi text fallback (`lib.rs:182` → `sdapi.rs:235` → `:283`).
`chat.rs:1992-1996` already documents that axum cancels these on disconnect, and
the existing `CancelWorkerOnDrop` guard covers the engine but **not** the prefill
enqueue or `selected_prefill_requests`.

Live by default (`HIPFIRE_SERVER_PREFILL_BATCH` unset/on).

**FIXED** with a `PrefillWaitGuard` armed once the session is enqueued and
disarmed on each success return. Its `Drop` spawns the cleanup, because `Drop` is
sync while both maps sit behind async mutexes.
