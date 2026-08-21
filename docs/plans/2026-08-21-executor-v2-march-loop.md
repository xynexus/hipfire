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

A serial loop that picks a runnable stream, advances it one quantum, records the
cursor, and repeats. Serial is deliberate: the parallelism is intra-kernel, and
per-stream state exists because a serial executor interleaves streams *within* a
march.

Cancellation moves here — from three per-token hook sites into the pick step, one
site instead of three-and-counting.

*Exit:* two concurrent `Generate` requests interleave at quantum granularity
under the flag, each producing output byte-identical to running it alone.

*Falsified by:* either stream's output differing from its solo run. That is the
accept-and-miscompute class; only exact comparison catches it.

### M3c — the suspension boundary

Park and resume across a real forward. `RunningStream`'s cursor contract already
exists and is tested at the type level; this makes it true of an actual forward.

*Exit:* a stream parked mid-generation and resumed produces output
byte-identical to an uninterrupted run. This is the contract §M6 later depends on.

**`hipGraph` capture is off on the v2 path** until its WCET is declared — a graph
is one indivisible quantum by construction, and a declared WCET that ignores an
enabled graph is exactly the failure the contract exists to prevent.

### M3d — the exit measurements

The parent plan's exit is three numbers read from the M0 trace, one
realtime-class stream against one bulk stream. They are measurements, not a build
milestone:

1. **p99 and max module duration**, and which `SuperOpKind` owns the max — the
   achievable suspension floor, and therefore the tightest drain budget the design
   can hold.
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
