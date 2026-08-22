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
use hipfire_scheduler::{WorkloadClass, WorkloadResources, WorkloadSpec};
use hipfire_steer::SteerSpec;

use crate::state::DaemonState;

/// Executor v2 master switch (`HIPFIRE_DAEMON_EXECUTOR=v2`), default off.
///
/// M3a admits **unconditionally**, and that is deliberate: gating admission on
/// the flag would leave the shape untested by every default run, which is the
/// one thing this stage exists to move. What the flag selects is who *runs* an
/// admitted stream, and until the march loop lands (M3b) there is exactly one
/// runner. So today it changes nothing — and [`admit_generate`] says so out
/// loud, because a green run under the flag is not evidence the executor ran.
pub(crate) fn executor_v2_enabled() -> bool {
    std::env::var("HIPFIRE_DAEMON_EXECUTOR").is_ok_and(|v| v == "v2")
}

/// §M7. Step all runnable streams through ONE batched forward instead of one
/// stream per quantum.
///
/// Off by default, and requires the v2 executor. The round-robin march is
/// correct but does not scale: measured aggregate throughput is flat-to-falling
/// in N (24.0 -> 22.4 -> 18.4 tok/s at N=1/16/32) because every stream pays its
/// own full forward. The batched decode path reaches 50.7 tok/s at N=16 on the
/// same model, which is what this is for.
pub(crate) fn executor_batched_enabled() -> bool {
    executor_v2_enabled()
        && matches!(
            std::env::var("HIPFIRE_DAEMON_EXECUTOR_BATCHED").ok().as_deref(),
            Some("1" | "true" | "on" | "yes")
        )
}

/// The session a `Generate` frame is admitted under.
///
/// Deliberately the SAME fallback `handlers::generate::text` applies — an
/// explicit non-empty `session_id`, else the request id — so admission and the
/// handler name the same conversation. Keying the table on anything else would
/// invent a second identity for something the protocol already names.
///
/// The request id is read straight off `msg["id"]` here where the handler
/// prefers the parsed `GenerateTextRequest::id`. Those agree: that field is a
/// bare `String` with no serde default, so the parse can only succeed when
/// `msg["id"]` is a string, and both derivations fall back to `"0"` otherwise.
pub(crate) fn session_of(msg: &serde_json::Value) -> SessionKey {
    let id = msg.get("id").and_then(|v| v.as_str()).unwrap_or("0");
    SessionKey::new(
        msg.get("session_id")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .unwrap_or(id),
    )
}

/// Admit the stream a `Generate` frame belongs to, and give it the GPU.
///
/// `None` means admission was refused — that session already has a live stream.
/// The caller **must still run the frame**: M3a moves the shape, not the
/// behaviour, and a refusal here would be user-visible. `None` doubles as
/// "nothing to retire".
pub(crate) fn admit_generate(state: &mut DaemonState, msg: &serde_json::Value) -> Option<StreamId> {
    if executor_v2_enabled() {
        static ANNOUNCED: std::sync::Once = std::sync::Once::new();
        ANNOUNCED.call_once(|| {
            tracing::warn!(
                "HIPFIRE_DAEMON_EXECUTOR=v2: admission is live, but the march loop lands in \
                 M3b — generate still runs inline"
            );
        });
    }
    let session = session_of(msg);
    // priority and enqueued_at_ms are 0 because nothing schedules on them yet;
    // the march loop is what gives them meaning.
    // Priority comes off the wire. `WorkloadSpec` has carried a `priority` field
    // all along and admission hardcoded 0, so latency-sensitive work had no way
    // to say so. Reusing it beats adding realtime `WorkloadClass` variants ahead
    // of anything that consumes them — §M6 can still add those when the classes
    // carry contracts, but the ordering lever already exists.
    let priority = msg
        .get("priority")
        .and_then(|v| v.as_u64())
        .unwrap_or(0)
        .min(u8::MAX as u64) as u8;
    let spec = WorkloadSpec::singleton(
        session.as_str(),
        WorkloadClass::TokenDecode,
        priority,
        0,
        WorkloadResources::default(),
    );
    // The rng is a slot: the daemon's decode path does not sample through
    // `SamplerRng` yet, and greedy decode never consults it — which is why
    // greedy baselines must not move when this lands.
    let id = match state
        .streams
        .admit(session, spec, SamplerRng::from_entropy())
    {
        Ok(id) => id,
        Err(AdmitError::SessionAlreadyLive(_)) => return None,
    };
    if let Some(stream) = state.streams.get_mut(id) {
        stream.request_id = msg
            .get("id")
            .and_then(|v| v.as_str())
            .unwrap_or("0")
            .to_string();
        stream.run();
    }
    Some(id)
}

/// Monotonic per-daemon stream identity — the executor's INTERNAL key.
///
/// Deliberately not the worker key: a worker holds a model and many streams
/// decode against one worker, so keying on it would collapse the distinction the
/// executor exists to make.
///
/// This is `Copy + Ord + Hash` and cheap, which is what the march loop wants for
/// bookkeeping. It is not what a *client* names — see [`SessionKey`] for the
/// wire-facing identity and [`StreamTable`] for the mapping between them.
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

/// The WIRE-facing stream identity: the `session_id` a client already sends.
///
/// **Not invented here.** `handlers/generate.rs` has accepted a client-supplied
/// `session_id` since before this module existed, falling back to the request id
/// when absent:
///
/// ```ignore
/// let session_id = msg.get("session_id").and_then(|v| v.as_str())
///     .filter(|s| !s.is_empty())
///     .unwrap_or(id);
/// ```
///
/// So making streams addressable by clients costs no new naming scheme — which
/// is what lets a control-plane op (a steer spec, say) name the stream it applies
/// to without the protocol growing a second, competing notion of identity.
///
/// Kept distinct from [`StreamId`] rather than collapsed into it: this one is a
/// `String` chosen by someone else and may be absent, malformed, or reused across
/// time. The executor should not key its hot-path bookkeeping on that.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) struct SessionKey(String);

impl SessionKey {
    pub(crate) fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    pub(crate) fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for SessionKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Why an admission was refused.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum AdmitError {
    /// That `session_id` already has a live stream. Carries it, because the
    /// caller usually wants to address the existing one rather than fail.
    ///
    /// This must be refused rather than silently allowed: two streams sharing a
    /// session id would drive the same KV and conversation state from two
    /// cursors, which is the multi-turn corruption this key exists to prevent.
    SessionAlreadyLive(StreamId),
}

/// The executor's set of admitted streams, and the only place the wire identity
/// is resolved to an internal one.
///
/// Two maps on purpose. The march loop iterates and mutates by [`StreamId`]
/// (cheap, `Copy`, dense); control-plane frames arrive naming a [`SessionKey`]
/// and need a lookup. Collapsing them would put string hashing on the march.
#[derive(Debug, Default)]
pub(crate) struct StreamTable {
    ids: StreamIds,
    streams: std::collections::BTreeMap<StreamId, RunningStream>,
    by_session: std::collections::HashMap<SessionKey, StreamId>,
}

impl StreamTable {
    /// Admit a stream for `session`. Refuses if that session already has a live
    /// stream — see [`AdmitError::SessionAlreadyLive`].
    pub(crate) fn admit(
        &mut self,
        session: SessionKey,
        spec: WorkloadSpec,
        rng: SamplerRng,
    ) -> Result<StreamId, AdmitError> {
        // Liveness, not mere presence. `finish()` does not touch `by_session` —
        // only `retire()` does — so a terminal-but-unretired stream would
        // otherwise refuse its own conversation's next turn, naming a dead
        // stream. A finished holder is reclaimed here instead.
        if let Some(&held) = self.by_session.get(&session) {
            let finished = self.streams.get(&held).is_none_or(|s| s.is_finished());
            if !finished {
                return Err(AdmitError::SessionAlreadyLive(held));
            }
            self.streams.remove(&held);
            self.by_session.remove(&session);
        }
        let id = self.ids.next();
        self.by_session.insert(session.clone(), id);
        self.streams
            .insert(id, RunningStream::admit(id, session, spec, rng));
        Ok(id)
    }

    pub(crate) fn get(&self, id: StreamId) -> Option<&RunningStream> {
        self.streams.get(&id)
    }

    pub(crate) fn get_mut(&mut self, id: StreamId) -> Option<&mut RunningStream> {
        self.streams.get_mut(&id)
    }

    /// Resolve a wire `session_id` to its live stream.
    pub(crate) fn id_for_session(&self, session: &SessionKey) -> Option<StreamId> {
        self.by_session.get(session).copied()
    }

    pub(crate) fn by_session_mut(&mut self, session: &SessionKey) -> Option<&mut RunningStream> {
        let id = self.id_for_session(session)?;
        self.streams.get_mut(&id)
    }

    /// Drop a stream and FREE ITS SESSION KEY.
    ///
    /// Freeing the key is the point: a session is a conversation and decodes
    /// again on the next turn. If retiring left the key mapped, the second turn
    /// of every conversation would be refused as a duplicate.
    pub(crate) fn retire(&mut self, id: StreamId) -> Option<RunningStream> {
        let stream = self.streams.remove(&id)?;
        self.by_session.remove(&stream.session);
        Some(stream)
    }

    /// Ids the march loop may give a quantum to, in admission order.
    /// Ids the march loop may give a quantum to, **highest priority first**,
    /// admission order within a priority.
    ///
    /// The march loop steps this list in order, so a higher-priority stream is
    /// dispatched before any lower-priority one in the same pass — which is what
    /// makes admission→first-dispatch bounded for latency-sensitive work while
    /// bulk streams keep running. `BTreeMap` iteration is already admission
    /// order, and `sort_by_key` is stable, so equal priorities keep it.
    pub(crate) fn runnable(&self) -> Vec<StreamId> {
        let mut ids: Vec<(u8, StreamId)> = self
            .streams
            .iter()
            .filter(|(_, s)| s.is_runnable())
            .map(|(id, s)| (s.spec.priority, *id))
            .collect();
        ids.sort_by_key(|(p, _)| std::cmp::Reverse(*p));
        ids.into_iter().map(|(_, id)| id).collect()
    }

    pub(crate) fn len(&self) -> usize {
        self.streams.len()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.streams.is_empty()
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
pub(crate) struct RunningStream {
    pub(crate) id: StreamId,
    /// The in-flight generation this stream advances, once admitted under the
    /// executor flag. `None` on the inline path, where `generate` is driven to
    /// completion inside the request instead.
    pub(crate) generation: Option<hipfire_serving_core::generate::Qwen35Generation>,
    /// The request id this stream's frames are tagged with. Stored because the
    /// march loop emits `token`/`done` long after the frame that created it, and
    /// `session` is NOT a substitute — they differ whenever a client sends an
    /// explicit `session_id`.
    pub(crate) request_id: String,
    /// The wire identity this stream was admitted for. Immutable: rebinding a
    /// live stream to a different session would silently redirect its KV and
    /// conversation state, so a session change is a new stream, not a mutation.
    session: SessionKey,
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

impl std::fmt::Debug for RunningStream {
    /// Hand-written because `Qwen35Generation` is not `Debug` and should not be
    /// made so just to satisfy a derive here — the same reasoning the plan gives
    /// for not adding a derive in `transport.rs`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RunningStream")
            .field("id", &self.id)
            .field("session", &self.session)
            .field("status", &self.status)
            .field("cursor", &self.cursor)
            .field("generation", &self.generation.is_some())
            .finish_non_exhaustive()
    }
}

impl RunningStream {
    /// Admit a stream. Starts at cursor zero in `Admitted`.
    ///
    /// Prefer [`StreamTable::admit`], which allocates the id and enforces that a
    /// session has at most one live stream.
    pub(crate) fn admit(
        id: StreamId,
        session: SessionKey,
        spec: WorkloadSpec,
        rng: SamplerRng,
    ) -> Self {
        Self {
            id,
            session,
            spec,
            generation: None,
            request_id: String::new(),
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

    /// The wire identity a client uses to address this stream.
    pub(crate) fn session(&self) -> &SessionKey {
        &self.session
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

    /// Give this stream the GPU. No-op unless it is runnable.
    ///
    /// Guards `Parked` as well as `Finished`. It previously guarded only
    /// `Finished`, so `run()` silently un-parked a suspended stream — the exact
    /// transition this module documents as impossible without an explicit act,
    /// and the one `resume()` returns a bool to make visible. A march loop that
    /// called `run()` on everything would have quietly resurrected every parked
    /// bulk stream.
    pub(crate) fn run(&mut self) {
        if self.is_runnable() {
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
        RunningStream::admit(
            ids.next(),
            SessionKey::new(name),
            spec(name),
            SamplerRng::from_seed(7),
        )
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

    /// `run()` must not be a back door around `resume()`.
    #[test]
    fn run_does_not_unpark() {
        let mut ids = StreamIds::default();
        let mut s = stream(&mut ids, "bulk");
        s.run();
        s.cursor.advance_module();
        s.park();
        s.run();
        assert_eq!(s.status(), StreamStatus::Parked, "run() silently un-parked");
        assert!(!s.is_runnable());
        assert!(s.resume(), "only an explicit resume may un-park");
    }

    /// A finished-but-unretired stream must not lock its conversation out.
    #[test]
    fn a_finished_stream_does_not_block_its_session() {
        let mut t = StreamTable::default();
        let first = table_admit(&mut t, "conv-9");
        t.get_mut(first).unwrap().finish();
        // NOTE: no retire() — this is the leak path.
        let second = table_admit(&mut t, "conv-9");
        assert_ne!(second, first);
        assert_eq!(t.id_for_session(&SessionKey::new("conv-9")), Some(second));
        assert_eq!(
            t.len(),
            1,
            "the dead stream must be reclaimed, not accumulate"
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

    fn table_admit(t: &mut StreamTable, name: &str) -> StreamId {
        t.admit(SessionKey::new(name), spec(name), SamplerRng::from_seed(3))
            .expect("first admission of a session must succeed")
    }

    #[test]
    fn a_wire_session_id_resolves_to_its_stream_and_back() {
        let mut t = StreamTable::default();
        let id = table_admit(&mut t, "conv-42");

        let key = SessionKey::new("conv-42");
        assert_eq!(
            t.id_for_session(&key),
            Some(id),
            "session must resolve to its stream"
        );
        assert_eq!(
            t.get(id).map(|s| s.session().as_str()),
            Some("conv-42"),
            "and the stream must report the session it was admitted for"
        );
        assert_eq!(
            t.id_for_session(&SessionKey::new("nope")),
            None,
            "an unknown session resolves to nothing, not to some stream"
        );
    }

    /// Two streams on one `session_id` would drive the same KV and conversation
    /// state from two cursors. Refused, and the refusal names the live stream so
    /// the caller can address it instead.
    #[test]
    fn a_session_may_not_have_two_live_streams() {
        let mut t = StreamTable::default();
        let first = table_admit(&mut t, "conv-1");

        let again = t.admit(
            SessionKey::new("conv-1"),
            spec("conv-1"),
            SamplerRng::from_seed(9),
        );
        assert_eq!(again, Err(AdmitError::SessionAlreadyLive(first)));
        assert_eq!(
            t.len(),
            1,
            "the refused admission must not leave a stream behind"
        );
    }

    /// The multi-turn property: retiring frees the key, so the next turn of the
    /// same conversation can be admitted. If `retire` left the key mapped, every
    /// conversation would work exactly once.
    #[test]
    fn retiring_frees_the_session_for_the_next_turn() {
        let mut t = StreamTable::default();
        let turn1 = table_admit(&mut t, "conv-7");
        t.get_mut(turn1).unwrap().finish();

        let retired = t.retire(turn1).expect("retire returns the stream");
        assert_eq!(retired.session().as_str(), "conv-7");
        assert!(t.is_empty());
        assert_eq!(t.id_for_session(&SessionKey::new("conv-7")), None);

        let turn2 = table_admit(&mut t, "conv-7");
        assert_ne!(
            turn2, turn1,
            "the next turn is a NEW stream, not the old one"
        );
        assert_eq!(t.id_for_session(&SessionKey::new("conv-7")), Some(turn2));
    }

    #[test]
    fn runnable_excludes_parked_and_finished_streams() {
        let mut t = StreamTable::default();
        let a = table_admit(&mut t, "a");
        let b = table_admit(&mut t, "b");
        let c = table_admit(&mut t, "c");
        assert_eq!(
            t.runnable(),
            vec![a, b, c],
            "all admitted streams are runnable"
        );

        t.get_mut(b).unwrap().park();
        t.get_mut(c).unwrap().finish();
        assert_eq!(
            t.runnable(),
            vec![a],
            "a parked stream is not picked up, and a finished one never again"
        );

        assert!(t.by_session_mut(&SessionKey::new("b")).unwrap().resume());
        assert_eq!(t.runnable(), vec![a, b], "an explicit resume puts it back");
    }

    /// Admission must name the conversation the HANDLER names. `session_of` is
    /// a second copy of `handlers::generate::text`'s fallback (that one derives
    /// the request id from the parsed request, which a unit test cannot reach
    /// without a `DaemonState`), so the copy is what can drift — pin all four
    /// branches of it.
    ///
    /// Drift is not cosmetic: a table keyed on a different string than the
    /// handler uses would admit a second stream for a conversation that already
    /// has one, which is the multi-turn KV corruption `SessionKey` exists to
    /// prevent.
    #[test]
    fn session_of_matches_the_generate_handlers_fallback() {
        use serde_json::json;

        assert_eq!(
            session_of(&json!({"id": "req-1", "session_id": "conv-9"})),
            SessionKey::new("conv-9"),
            "an explicit session_id wins over the request id"
        );
        assert_eq!(
            session_of(&json!({"id": "req-1", "session_id": ""})),
            SessionKey::new("req-1"),
            "an EMPTY session_id falls back to the request id, as the handler's \
             `.filter(|s| !s.is_empty())` does — not to an empty key that every \
             unnamed request would collide on"
        );
        assert_eq!(
            session_of(&json!({"id": "req-1"})),
            SessionKey::new("req-1"),
            "absent session_id falls back to the request id"
        );
        assert_eq!(
            session_of(&json!({"prompt": "hi"})),
            SessionKey::new("0"),
            "no id at all lands on the handler's own \"0\" default"
        );
    }

    /// Per-stream state is per stream — the property the M1d globals could not
    /// hold once a serial executor interleaves streams within one march.
    #[test]
    fn per_stream_state_does_not_alias_between_streams() {
        let mut ids = StreamIds::default();
        let mut a = RunningStream::admit(
            ids.next(),
            SessionKey::new("a"),
            spec("a"),
            SamplerRng::from_seed(1),
        );
        let mut b = RunningStream::admit(
            ids.next(),
            SessionKey::new("b"),
            spec("b"),
            SamplerRng::from_seed(2),
        );

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
