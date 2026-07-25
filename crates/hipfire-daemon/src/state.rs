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

use hipfire_serving_core::dummy::DummyModelState;
use hipfire_serving_core::model::LoadedModel;
use hipfire_serving_core::session::DEFAULT_MODEL_WORKER_ID;
use hipfire_state::GenericSequenceStateArena;

use crate::{DrafterTrainSession, LoraTrainSession, ResourceReservationManager};

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
    pub resource_reservations: ResourceReservationManager,
    /// The response sink. Handlers write whole locked lines to it, which is why
    /// interleaved progress frames are safe on a single thread. A multi-client
    /// transport replaces this with a per-connection writer.
    pub stdout: std::io::Stdout,
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
            resource_reservations: ResourceReservationManager::from_env(),
            stdout: std::io::stdout(),
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
