// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `RunningStream` — the admitted, resumable unit of forward work for executor v2.
//!
//! Today the daemon executes a `Generate` frame to completion inside the request
//! `match`. Executor v2 (`docs/plans/2026-08-09-v2-daemon-module-major-multistream.md`
//! §M3) replaces that with cursors: a `Generate` frame **admits a stream** and
//! returns, and a serial march loop advances admitted streams a module at a time.
//! `PendingQueue` keeps its per-connection FIFO invariant; it just stops running
//! the work itself.
//!
//! This module is **additive scaffolding**. Nothing executes against it yet — the
//! march loop, admission, and the suspension boundary are still to come, and no
//! existing path changes behaviour by this file existing. It is here because the
//! per-stream state migration (§M1d) has no home without it: two of the three
//! globals in that milestone were retired by threading a value the caller already
//! owned (`RAW_OVERRIDE` → a request parameter, `SAMPLER_STATE` → a `SamplerRng`),
//! and the third, `hipfire_steer`'s session, has no such owner. See
//! `docs/plans/2026-08-20-p2-steer-per-stream-and-lowered.md` §M3.
//!
//! **The executor is serial on purpose.** Nothing in shared-weight multi-stream
//! decoding needs two threads issuing GPU work; the parallelism is intra-kernel.
//! Per-stream state is required anyway, because a serial executor interleaves
//! streams *within* a march — which is exactly when a process global becomes
//! silently wrong rather than merely inelegant.

// Deliberately unused for now: this type is admitted into the tree ahead of the
// march loop that will drive it, so that §M1d's remaining per-stream migration
// has somewhere to move state TO. The unit tests below exercise every item, so
// this suppresses "never used by non-test code", not "never verified".
//
// Remove this attribute when the executor lands — at that point a dead item here
// is a real signal that something was designed and then not wired.
#![allow(dead_code)]

use hipfire_runtime::sampler::SamplerRng;
use hipfire_scheduler::{WorkloadClass, WorkloadSpec};
use hipfire_steer::SteerSpec;

/// Monotonic per-daemon stream identity.
///
/// Deliberately NOT the request id, the worker key, or the connection: a worker
/// holds a model and many streams decode against one worker, so reusing either
/// would collapse the distinction the executor exists to make. Opaque and
/// process-local — it never reaches the wire.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) struct StreamId(u64);

impl StreamId {
    pub(crate) fn get(self) -> u64 {
        self.0
    }
}

impl std::fmt::Display for StreamId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "stream:{}", self.0)
    }
}

/// Hands out `StreamId`s. Owned by the executor, so a plain counter is enough —
/// no atomics, because there is exactly one march loop and it is single-threaded.
#[derive(Debug, Default)]
pub(crate) struct StreamIds(u64);

impl StreamIds {
    pub(crate) fn next(&mut self) -> StreamId {
        self.0 += 1;
        StreamId(self.0)
    }
}

/// Where a stream resumes.
///
/// Module-major: `module` indexes the op sequence *within the current token's
/// forward*, `token` counts completed tokens. The pair is the whole of what
/// "lossless suspension" means — a parked stream resumes from here and never
/// restarts a token it had partially run.
///
/// The cursor is deliberately dumb: it does not know how many modules a token
/// has, because that is a property of the lowered `LayerProgram` for the model,
/// not of the stream. The executor advances it; the cursor only guarantees the
/// arithmetic is monotonic and that parking does not perturb it.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct StreamCursor {
    pub(crate) token: u64,
    pub(crate) module: u32,
}

impl StreamCursor {
    /// One module of the current token completed.
    pub(crate) fn advance_module(&mut self) {
        self.module += 1;
    }

    /// The current token's forward completed; the next token starts at module 0.
    pub(crate) fn complete_token(&mut self) {
        self.token += 1;
        self.module = 0;
    }
}

/// Lifecycle of an admitted stream.
///
/// `Parked` is the interesting one: it is not "cancelled" and not "queued". A
/// parked stream holds its cursor and its per-stream state and is expected to
/// resume — realtime admission parks bulk streams for a session's duration, and
/// their output must be byte-identical to an uninterrupted run.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum StreamStatus {
    /// Admitted, not yet given a quantum.
    Admitted,
    /// Advancing in the march loop.
    Running,
    /// Suspended, resumable, holding its cursor.
    Parked,
    /// Terminal. Cursor is frozen for inspection.
    Finished,
}

/// One admitted forward-executing stream.
///
/// Holds the per-stream state that used to be process globals. Those fields are
/// **slots, not wiring**: nothing reads them through this struct yet, and this
/// file does not change the behaviour of any existing path. The migration is
/// §M1d's remaining third, and it needs a place to move the state *to* before it
/// can move it.
#[derive(Debug)]
pub(crate) struct RunningStream {
    pub(crate) id: StreamId,
    /// Scheduler attribution and admission inputs. Reused rather than
    /// re-modelled — `WorkloadSpec` already carries class, priority, owner, and
    /// the microbatch compatibility key.
    pub(crate) spec: WorkloadSpec,
    pub(crate) cursor: StreamCursor,
    status: StreamStatus,

    /// Per-stream sampler RNG (§M1b, landed as a value — this is its owner once
    /// streams interleave). Greedy decode does not consult it, which is why
    /// greedy baselines must not move when the executor lands.
    pub(crate) rng: SamplerRng,

    /// Per-stream raw-prompt override (§M1d, landed as a request parameter).
    /// `None` means "fall back to the model's framing default".
    pub(crate) raw_override: Option<bool>,

    /// Per-stream steering spec (§M1d's remaining third). `None` is the common
    /// case and must stay free: an unsteered stream may not pay for this.
    ///
    /// The *spec* lives here; `hipfire_steer`'s `APPLY_CACHE` deliberately stays
    /// a `thread_local` holding `GpuTensor`, which is `!Sync` and cannot move
    /// into shared state. What per-stream ownership changes is what the apply
    /// epoch invalidates against, so a cache entry uploaded for one stream is
    /// not reused for another.
    pub(crate) steer: Option<SteerSpec>,
}

impl RunningStream {
    /// Admit a stream. Starts at cursor zero in `Admitted`.
    pub(crate) fn admit(id: StreamId, spec: WorkloadSpec, rng: SamplerRng) -> Self {
        Self {
            id,
            spec,
            cursor: StreamCursor::default(),
            status: StreamStatus::Admitted,
            rng,
            raw_override: None,
            steer: None,
        }
    }

    pub(crate) fn status(&self) -> StreamStatus {
        self.status
    }

    pub(crate) fn class(&self) -> WorkloadClass {
        self.spec.class
    }

    /// True once terminal. The march loop uses this to drop the stream.
    pub(crate) fn is_finished(&self) -> bool {
        self.status == StreamStatus::Finished
    }

    /// Eligible to be given a quantum. `Parked` is NOT runnable — resuming is an
    /// explicit act by whatever parked it, so a suspended bulk stream cannot be
    /// picked up again merely because the march loop came round.
    pub(crate) fn is_runnable(&self) -> bool {
        matches!(self.status, StreamStatus::Admitted | StreamStatus::Running)
    }

    /// Give this stream the GPU. No-op if already running.
    pub(crate) fn run(&mut self) {
        if self.status != StreamStatus::Finished {
            self.status = StreamStatus::Running;
        }
    }

    /// Suspend losslessly. The cursor is untouched — that is the entire contract.
    pub(crate) fn park(&mut self) {
        if self.status != StreamStatus::Finished {
            self.status = StreamStatus::Parked;
        }
    }

    /// Resume a parked stream at its cursor. Returns false if it was not parked,
    /// so a double-resume is visible rather than silent.
    pub(crate) fn resume(&mut self) -> bool {
        if self.status == StreamStatus::Parked {
            self.status = StreamStatus::Running;
            true
        } else {
            false
        }
    }

    /// Terminal. Idempotent, because both normal completion and cancellation
    /// reach it and the executor should not have to track which came first.
    pub(crate) fn finish(&mut self) {
        self.status = StreamStatus::Finished;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hipfire_scheduler::{WorkloadClass, WorkloadResources};

    fn spec(id: &str) -> WorkloadSpec {
        WorkloadSpec::singleton(
            id,
            WorkloadClass::TokenDecode,
            0,
            0,
            WorkloadResources::default(),
        )
    }

    fn stream(ids: &mut StreamIds, name: &str) -> RunningStream {
        RunningStream::admit(ids.next(), spec(name), SamplerRng::from_seed(7))
    }

    #[test]
    fn ids_are_unique_and_monotonic() {
        let mut ids = StreamIds::default();
        let a = ids.next();
        let b = ids.next();
        let c = ids.next();
        assert!(a < b && b < c, "ids must increase");
        assert_eq!((a.get(), b.get(), c.get()), (1, 2, 3), "ids start at 1");
    }

    #[test]
    fn cursor_advances_module_then_resets_on_token_completion() {
        let mut cur = StreamCursor::default();
        cur.advance_module();
        cur.advance_module();
        assert_eq!(
            cur,
            StreamCursor {
                token: 0,
                module: 2
            }
        );
        cur.complete_token();
        assert_eq!(
            cur,
            StreamCursor {
                token: 1,
                module: 0
            },
            "a completed token restarts module indexing"
        );
    }

    /// The headline invariant: suspension is lossless. A parked stream resumes
    /// from its cursor and never restarts the token it had partially run.
    #[test]
    fn park_and_resume_preserve_the_cursor() {
        let mut ids = StreamIds::default();
        let mut s = stream(&mut ids, "bulk");
        s.run();
        for _ in 0..5 {
            s.cursor.advance_module();
        }
        s.cursor.complete_token();
        s.cursor.advance_module();
        let at_park = s.cursor;

        s.park();
        assert_eq!(s.cursor, at_park, "parking must not perturb the cursor");
        assert!(
            !s.is_runnable(),
            "a parked stream is not picked up by the march"
        );

        assert!(s.resume(), "resume must report that it acted");
        assert_eq!(s.cursor, at_park, "resume must continue, not restart");
        assert!(s.is_runnable());

        s.cursor.advance_module();
        assert_eq!(
            s.cursor,
            StreamCursor {
                token: 1,
                module: 2
            },
            "advancing after resume continues from the parked position"
        );
    }

    #[test]
    fn resume_reports_false_when_not_parked() {
        let mut ids = StreamIds::default();
        let mut s = stream(&mut ids, "a");
        s.run();
        assert!(
            !s.resume(),
            "resuming a running stream is not a silent no-op"
        );
    }

    #[test]
    fn finish_is_terminal_and_idempotent() {
        let mut ids = StreamIds::default();
        let mut s = stream(&mut ids, "a");
        s.run();
        s.finish();
        assert!(s.is_finished());
        assert!(!s.is_runnable());

        // Cancellation and normal completion both land here; neither may revive.
        s.finish();
        s.run();
        s.park();
        assert!(s.is_finished(), "a finished stream must stay finished");
        assert!(!s.resume(), "a finished stream cannot be resumed");
    }

    /// Per-stream state is per stream — the property the M1d globals could not
    /// hold once a serial executor interleaves streams within one march.
    #[test]
    fn per_stream_state_does_not_alias_between_streams() {
        let mut ids = StreamIds::default();
        let mut a = RunningStream::admit(ids.next(), spec("a"), SamplerRng::from_seed(1));
        let mut b = RunningStream::admit(ids.next(), spec("b"), SamplerRng::from_seed(2));

        a.raw_override = Some(true);
        a.cursor.advance_module();

        assert_ne!(a.id, b.id);
        assert_ne!(a.rng, b.rng, "distinct seeds must not collapse");
        assert_eq!(b.raw_override, None, "stream A's override must not reach B");
        assert_eq!(b.cursor, StreamCursor::default(), "cursors are independent");

        b.park();
        assert!(a.is_runnable() || a.status() == StreamStatus::Admitted);
        assert!(!b.is_runnable(), "parking B must not depend on or affect A");
    }
}
