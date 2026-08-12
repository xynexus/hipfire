//! The daemon's resident state.
//!
//! Until this existed, every one of these fields was a `let mut` local in
//! `main()`'s stack frame, and all 48 request handlers borrowed them mutably in
//! turn. That ownership — not the `for line in stdin.lock().lines()` loop — is
//! why the daemon is strictly serial: two handlers can never run concurrently
//! because both need `&mut gpu`. The loop is a consequence of it.
//!
//! Collecting them here is the prerequisite for the rest of the daemon merge:
//! handlers can be extracted taking `&mut DaemonState`, and a single
//! GPU-owning executor can own one instance while a multi-client front end
//! queues work in front of it.
//!
//! Field access stays direct and public-in-crate rather than going through
//! accessors, because handlers routinely need two fields at once (`&mut
//! state.gpu` alongside `&mut state.model`); disjoint field borrows are allowed
//! where `&mut self` methods would conflict.

use std::io::Write;

use hipfire_serving_core::dummy::DummyModelState;
use hipfire_serving_core::model::LoadedModel;
use hipfire_serving_core::session::DEFAULT_MODEL_WORKER_ID;
use hipfire_state::GenericSequenceStateArena;

use crate::queue::SchedulerStats;
use crate::transport::ReplySink;
use crate::{
    CalibrateDaemonSession, DrafterTrainSession, LoraTrainSession, ResourceReservationManager,
};

pub(crate) struct DaemonState {
    /// The single GPU handle. Threaded as `&mut` into every GPU-touching
    /// handler; the contended resource that makes execution serial.
    pub gpu: hipfire_rdna::Gpu,
    /// The *active* model slot — the only one whose GPU state is live.
    pub model: Option<LoadedModel>,
    /// Which key `model` currently corresponds to.
    pub active_worker_id: String,
    /// Parked workers, keyed by `worker_key_id`. Swapped into `model` by
    /// `activate_model_worker`; there is no eviction policy.
    pub resident_models: std::collections::HashMap<String, LoadedModel>,
    pub generic_state_arena: GenericSequenceStateArena,
    /// PFlash speculative-prefill state. None unless the load message
    /// includes a `prefill_drafter` path AND `prefill_compression` != "off".
    /// Lives alongside `model` so unload_model + this state are paired
    /// teardowns.
    pub pflash_state: Option<hipfire_arch_qwen35::pflash::PflashState>,
    /// The PflashConfig captured at load time. Per-request `prefill_*`
    /// params override individual fields; the rest fall back to these
    /// load-time defaults. Cleared alongside `pflash_state`.
    pub pflash_cfg: Option<hipfire_arch_qwen35::pflash::PflashConfig>,
    /// H-Neurons CETT capture: per-layer down_proj column norms
    /// (`[n_layers][intermediate]`), loaded once via `cett_load_colnorms` and
    /// reused for every `cett_capture` prefill.
    pub cett_colnorms: Option<Vec<Vec<f32>>>,
    /// Hetero PFlash: when prefill_drafter_device differs from the target,
    /// the drafter weights/KV/scratch live on a sibling device. The compress
    /// output is a host-side Vec<u32>, so no peer-copy is needed — generate
    /// routes maybe_compress_prompt to this handle, decode stays on target.
    /// None means the drafter shares the target gpu (single-card, unchanged).
    pub pflash_drafter_gpu: Option<hipfire_rdna::Gpu>,
    pub dummy_model: Option<DummyModelState>,
    /// Resident micro-step-preemptible LoRA training session (see
    /// LoraTrainSession). Some between quanta of a run; runner drives one
    /// quantum per TrainLora request.
    pub lora_train_session: Option<LoraTrainSession>,
    /// Resident micro-step-preemptible SSM-drafter training session (see
    /// DrafterTrainSession). Some between quanta of a run; runner drives one
    /// quantum of EPOCHS per TrainDrafter request.
    pub drafter_train_session: Option<DrafterTrainSession>,
    /// Resident layer-preemptible calibration/induction session (see
    /// CalibrateDaemonSession). Some between layers of a run, keyed by `run_id`;
    /// the runner drives exactly one layer per Calibrate request. One layer is
    /// the calibration quantum; the parked session carries the boxed adapter,
    /// source, and job the engine borrows each turn.
    pub calibrate_session: Option<CalibrateDaemonSession>,
    pub resource_reservations: ResourceReservationManager,
    /// Where replies go, and what they are tagged with.
    pub out: Responder,
    /// Snapshot of the scheduler's counters, refreshed by the executor as it takes
    /// up each frame.
    ///
    /// A snapshot rather than a live borrow because the queue lives in the
    /// executor loop, not in the state: a handler answering `scheduler_status`
    /// cannot hold the queue while the executor is mid-dispatch. Taken after the
    /// current frame is popped, so `queue_depth` is what remains behind it.
    pub scheduler_stats: SchedulerStats,
}

/// The response sink plus the id every frame written through it is stamped with.
///
/// This is deliberately a separate struct rather than two fields on
/// [`DaemonState`]. `emit` needs `&mut` on both the writer and the id, and a
/// method on `DaemonState` would therefore borrow the *whole* state — which
/// breaks the handlers that legitimately hold a mutable borrow of another field
/// across an emit (the drafter training loop emits per-epoch progress while
/// holding `&mut drafter_train_session`, and the KLD scorer emits per-chunk while
/// holding the backend). Field-level disjointness keeps those working.
///
/// It is also the seam a multi-client transport replaces: one `Responder` per
/// connection instead of one per process.
pub(crate) struct Responder {
    /// Where frames go. Handlers write whole lines, which is why interleaved
    /// progress frames are safe on a single thread.
    ///
    /// Abstract rather than a concrete `Stdout` so each connection can supply its
    /// own writer, and so [`Responder::emit`] is testable against a buffer at all.
    /// The executor swaps this per frame to the sink the frame arrived on.
    pub sink: ReplySink,
    /// The `id` of the request being handled right now, refreshed once per
    /// read-loop iteration; empty when the request carried no id.
    ///
    /// A single executor handles one request at a time, so "current" stays
    /// well-defined once a multi-client transport lands; what changes then is
    /// which writer a frame goes to, not how it is tagged.
    pub request_id: String,
}

impl Responder {
    /// A responder writing to process stdout — the stdio transport's sink, and the
    /// placeholder the executor holds before the first frame arrives.
    pub fn to_stdout() -> Self {
        Self {
            sink: ReplySink::new(std::io::stdout()),
            request_id: String::new(),
        }
    }

    /// Write one JSONL response frame, stamped with the current request id, and
    /// flush.
    ///
    /// Use this rather than `writeln!(..)` for anything a caller is meant to
    /// read. Two reasons, both previously handled per-site or not at all:
    ///
    /// - **Correlation.** The id is added here, so a frame cannot reach the wire
    ///   untagged. An explicit `id` already in `frame` wins — batch ops answer
    ///   per-envelope or per-session ids that are deliberately not the request id.
    /// - **Framing safety.** The frame goes out through `serde_json`, so a
    ///   user-controlled id or message containing `"`, `\` or a newline is
    ///   escaped rather than desyncing every following line of the stream. The
    ///   generate path in `serving-core::events` already worked this way; the
    ///   hand-written literals in the daemon did not.
    pub fn emit(&mut self, mut frame: serde_json::Value) {
        stamp_request_id(&mut frame, &self.request_id);
        // `Display for Value` writes incrementally into the writer, so large
        // payloads (embeddings, the model registry) are not materialised first.
        let _ = writeln!(self.sink, "{frame}");
        let _ = self.sink.flush();
    }

    /// Emit an error frame tagged with the current request id.
    ///
    /// Prefer this over `emit_error_with_id(.., "", ..)`: an error carrying an
    /// empty id cannot be routed back to whoever asked, which is precisely what
    /// a multiplexing transport needs it to be.
    pub fn error(&mut self, message: impl std::fmt::Display) {
        self.emit(serde_json::json!({
            "type": "error",
            "message": format!("{message}"),
        }));
    }
}

impl DaemonState {
    /// Build the resident state around an initialized GPU handle. Startup
    /// policy — GPU init failure reporting and reservation claiming — stays in
    /// `main()`, so this constructor cannot fail.
    pub fn new(gpu: hipfire_rdna::Gpu) -> Self {
        Self {
            gpu,
            model: None,
            active_worker_id: DEFAULT_MODEL_WORKER_ID.to_string(),
            resident_models: std::collections::HashMap::new(),
            generic_state_arena: GenericSequenceStateArena::new(),
            pflash_state: None,
            pflash_cfg: None,
            cett_colnorms: None,
            pflash_drafter_gpu: None,
            dummy_model: None,
            lora_train_session: None,
            drafter_train_session: None,
            calibrate_session: None,
            resource_reservations: ResourceReservationManager::from_env(),
            out: Responder::to_stdout(),
            scheduler_stats: SchedulerStats::default(),
        }
    }

    /// Claim the configured resource reservations against the resident GPU.
    /// Wrapped as a method so callers do not have to spell out the disjoint
    /// borrow of two fields at once.
    pub fn reacquire_reservations(&mut self) -> Result<(), String> {
        self.resource_reservations
            .reacquire_placeholders(&mut self.gpu)
    }
}

/// Tag `frame` with `request_id` unless it already names an `id`.
///
/// Split out of [`Responder::emit`] so the rule is testable without a writer:
/// `Responder` holds a concrete `Stdout`, which a unit test cannot substitute.
/// (Genericising that writer is M3's job, when per-connection writers land.)
fn stamp_request_id(frame: &mut serde_json::Value, request_id: &str) {
    if let Some(object) = frame.as_object_mut() {
        object
            .entry("id")
            .or_insert_with(|| serde_json::Value::String(request_id.to_string()));
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::{stamp_request_id, Responder};
    use crate::transport::ReplySink;

    /// A sink that keeps what was written so a test can read it back. This is the
    /// reason `Responder::sink` is boxed rather than a concrete `Stdout`.
    #[derive(Clone, Default)]
    struct Captured(Arc<Mutex<Vec<u8>>>);

    impl std::io::Write for Captured {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    impl Captured {
        fn lines(&self) -> Vec<serde_json::Value> {
            let bytes = self.0.lock().unwrap().clone();
            String::from_utf8(bytes)
                .expect("frames are utf8")
                .lines()
                .map(|l| serde_json::from_str(l).expect("each frame is one JSON line"))
                .collect()
        }
    }

    #[test]
    fn emitted_frames_are_one_json_line_each_and_carry_the_request_id() {
        let captured = Captured::default();
        let mut out = Responder {
            sink: ReplySink::new(captured.clone()),
            request_id: "req-1".to_string(),
        };

        out.emit(serde_json::json!({ "type": "lora_ok" }));
        out.error("something broke");
        // An id already in the frame wins over the current request id.
        out.emit(serde_json::json!({ "type": "token", "id": "session-9" }));

        let frames = captured.lines();
        assert_eq!(
            frames.len(),
            3,
            "one line per frame, nothing merged or split"
        );
        assert_eq!(
            frames[0],
            serde_json::json!({ "type": "lora_ok", "id": "req-1" })
        );
        assert_eq!(
            frames[1],
            serde_json::json!({ "type": "error", "message": "something broke", "id": "req-1" })
        );
        assert_eq!(frames[2]["id"], "session-9");
    }

    #[test]
    fn a_hostile_request_id_cannot_desync_the_stream() {
        // The whole reason frames go through serde_json: an id carrying a quote
        // and a newline must be escaped, not break the line protocol. Before this,
        // hand-written literals in the daemon interpolated raw.
        let captured = Captured::default();
        let mut out = Responder {
            sink: ReplySink::new(captured.clone()),
            request_id: "evil\"}\n{\"type\":\"injected".to_string(),
        };

        out.emit(serde_json::json!({ "type": "pong" }));

        let frames = captured.lines();
        assert_eq!(
            frames.len(),
            1,
            "the id must not have injected a second frame"
        );
        assert_eq!(frames[0]["type"], "pong");
        assert_eq!(frames[0]["id"], "evil\"}\n{\"type\":\"injected");
    }

    #[test]
    fn stamping_adds_the_request_id_and_never_overwrites_an_explicit_one() {
        // The common case: a handler emits a bare frame and correlation is added
        // for it, so an untagged frame cannot reach the wire.
        let mut frame = serde_json::json!({ "type": "lora_ok" });
        stamp_request_id(&mut frame, "req-1");
        assert_eq!(
            frame,
            serde_json::json!({ "type": "lora_ok", "id": "req-1" })
        );

        // Batch and session ops answer per-envelope ids that are deliberately
        // NOT the request id, so an explicit id must win.
        let mut explicit = serde_json::json!({ "type": "token", "id": "session-7" });
        stamp_request_id(&mut explicit, "req-1");
        assert_eq!(
            explicit,
            serde_json::json!({ "type": "token", "id": "session-7" })
        );

        // An empty current id still yields the field, so the shape is uniform
        // for a reader that expects it.
        let mut no_id = serde_json::json!({ "type": "pong" });
        stamp_request_id(&mut no_id, "");
        assert_eq!(no_id, serde_json::json!({ "type": "pong", "id": "" }));

        // Non-objects are left alone rather than panicking.
        let mut scalar = serde_json::json!("not-a-frame");
        stamp_request_id(&mut scalar, "req-1");
        assert_eq!(scalar, serde_json::json!("not-a-frame"));
    }
}
