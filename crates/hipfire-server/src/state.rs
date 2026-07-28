use std::collections::{HashMap, HashSet, VecDeque};
use std::path::PathBuf;
use std::sync::{Arc, Mutex as StdMutex};
use tokio::sync::{Mutex, Notify};

use hipfire_auth::{AccessStore, CredentialSnapshot, RateLimiter, UsageWriter};
use hipfire_config::{HipfireConfig, LoadedConfig};
use hipfire_daemon_adapter::DaemonEngine;
use hipfire_diffusion::{DiffusionGenerationRuntimeOptions, DiffusionPipeline};
use hipfire_prompt::Message;
use hipfire_scheduler::{
    ContinuousWorkScheduler, PriorityPrefillScheduler, SchedulerPolicyEnv, WorkloadResources,
};

#[derive(Clone, Debug)]
pub struct StoredResponsesContext {
    pub messages: Vec<Message>,
}

#[derive(Clone, Debug)]
pub struct StoredFile {
    pub id: String,
    pub filename: String,
    pub bytes: usize,
    pub purpose: String,
    pub created_at: u64,
    pub content: String,
}

#[derive(Clone, Debug)]
pub struct StoredBatch {
    pub id: String,
    pub status: String,
    pub endpoint: String,
    pub completion_window: String,
    pub input_file_id: String,
    pub output_file_id: Option<String>,
    pub error_file_id: Option<String>,
    pub request_count: usize,
    pub completed_requests: usize,
    pub created_at: u64,
    pub in_progress_at: Option<u64>,
    pub completed_at: Option<u64>,
    pub failed_reason: Option<String>,
}

#[derive(Clone, Debug, Default)]
pub struct SdapiProgressState {
    pub active: bool,
    pub skipped: bool,
    pub interrupted: bool,
    pub task_id: Option<String>,
    pub mode: Option<String>,
    pub prompt: Option<String>,
    pub sampling_step: usize,
    pub sampling_steps: usize,
    pub current_image: Option<String>,
    pub textinfo: Option<String>,
    pub started_at_unix_secs: Option<u64>,
    pub completed_at_unix_secs: Option<u64>,
}

#[derive(Clone, Debug)]
pub struct LoadedModelState {
    pub worker_key_id: Option<String>,
    pub cache_capable: bool,
    pub max_seq: u32,
    /// Arch tag reported by the daemon load (a registered model_type), cached
    /// for batch-eligibility routing.
    pub arch: Option<String>,
}

pub struct AppState {
    /// Serializes all daemon I/O. Phase A: one request at a time.
    pub engine: Mutex<Option<DaemonEngine>>,
    pub loaded_config: Mutex<LoadedConfig>,
    pub config: Mutex<HipfireConfig>,
    /// Worker key ID of the currently loaded model, if any.
    pub loaded_model_path: Mutex<Option<String>>,
    /// Whether the current daemon worker can safely keep prefix-cache state across requests.
    pub loaded_model_cache_capable: Mutex<Option<bool>>,
    /// Effective max_seq used when the current model was loaded.
    pub loaded_model_max_seq: Mutex<Option<u32>>,
    /// Loaded daemon workers keyed by resolved model path.
    pub loaded_models: Mutex<HashMap<String, LoadedModelState>>,
    /// Shared prefill scheduler used by Rust request paths when enabled.
    pub prefill_scheduler: Mutex<PriorityPrefillScheduler>,
    /// Continuous work scheduler: the admission spine the batch runner pulls
    /// batches from (priority + microbatch grouping + resource leases). Phase 3.
    pub work_scheduler: Mutex<ContinuousWorkScheduler>,
    /// Request IDs selected by the scheduler and ready to enter daemon I/O.
    pub selected_prefill_requests: Mutex<HashSet<String>>,
    /// Serializes scheduler selection so one request path chooses batches at a time.
    pub prefill_dispatch: Mutex<()>,
    pub prefill_notify: Notify,
    /// Continuous-batching request registry (Phase 1). Requests park here and
    /// await the runner instead of taking `engine` directly. Only populated
    /// when `HIPFIRE_SERVER_PREFILL_BATCH` is enabled and the runner is spawned.
    pub batch_inbox: crate::batch_runner::BatchInbox,
    /// True once [`crate::batch_runner::spawn_batch_runner`] has started the
    /// runner loop. Routes that enqueue + await a runner result (image gen) must
    /// only do so when this is set — otherwise (e.g. unit tests that build an
    /// `AppState` without a serve loop) the request would park forever. Text/embed
    /// are additionally gated by arch-eligibility, but image has no such gate.
    pub batch_runner_active: std::sync::atomic::AtomicBool,
    /// Live batch-runner telemetry surfaced by `/health`.
    pub batch_telemetry: Mutex<crate::batch_runner::BatchTelemetry>,
    /// What the listener actually bound to, recorded by `serve_loaded` once the
    /// socket is up. `/health` reports this so clients read the real bind
    /// instead of re-deriving it from their own (possibly newer) config. `None`
    /// in unit tests and any context without a serve loop.
    pub bind: StdMutex<Option<crate::bind::BindInfo>>,
    pub responses_contexts: Mutex<HashMap<String, StoredResponsesContext>>,
    pub responses_order: Mutex<VecDeque<String>>,
    pub files: Mutex<HashMap<String, StoredFile>>,
    pub file_order: Mutex<VecDeque<String>>,
    pub batches: Mutex<HashMap<String, StoredBatch>>,
    pub batch_order: Mutex<VecDeque<String>>,
    /// Opened diffusion HFQ pipelines keyed by resolved model path.
    pub diffusion_pipelines: Mutex<HashMap<PathBuf, Arc<DiffusionPipeline>>>,
    /// Daemon-resolved default diffusion runtime backend. Resolved once at
    /// `serve` launch (HIP/ROCm-first: the detected GPU, or the CPU reference
    /// oracle when `HIPFIRE_DIFFUSION_CPU_REFERENCE` is set). The bare
    /// constructor leaves this as the CPU reference so unit/route tests run
    /// without a GPU; a per-request `rocm_device_id` still overrides it.
    pub diffusion_runtime_default: StdMutex<DiffusionGenerationRuntimeOptions>,
    /// Stable-Diffusion-WebUI option compatibility values posted to
    /// `/sdapi/v1/options` that do not map to native Hipfire config fields.
    pub sdapi_options: Mutex<HashMap<String, serde_json::Value>>,
    pub sdapi_progress: Arc<StdMutex<SdapiProgressState>>,
    pub training_runs_dir: PathBuf,
    /// Server-owned root for images saved by the SD API routes. Derived from
    /// config at construction; request `outdir_*` overrides never reach it.
    pub sdapi_output_root: PathBuf,
    /// Admin-configured DoS ceiling on SD API request geometry. Derived from
    /// config at construction; clients may request smaller, never larger.
    pub(crate) sdapi_geometry_limits: crate::routes::sdapi::SdapiGeometryLimits,
    /// Primary local model root. Derived from config at construction; network
    /// model resolution is confined to this root plus `models_network_dir`.
    pub models_dir: PathBuf,
    /// Optional admin-configured extra read-only model root (e.g. an NFS
    /// share). Network model resolution is confined to `models_dir`
    /// plus this root; unset by default. Derived from config at construction.
    pub models_network_dir: Option<PathBuf>,
    /// Local admin bearer secret (`~/.hipfire/admin.secret`); same-box
    /// CLI/TUI present this to skip the `/admin` login flow.
    pub admin_secret: String,
    /// Active `/admin` browser sessions: token -> expiry (unix secs).
    pub admin_sessions: Mutex<HashMap<String, u64>>,
    /// Durable API identities plus the immutable verification snapshot used by
    /// request middleware. Database errors remain visible and fail closed.
    pub access: AccessRuntime,
    pub rate_limiter: RateLimiter,
    pub usage_writer: Option<UsageWriter>,
}

pub struct AccessRuntime {
    store: std::sync::RwLock<Option<AccessStore>>,
    credentials: std::sync::RwLock<Option<Arc<CredentialSnapshot>>>,
    init_error: std::sync::RwLock<Option<String>>,
}

impl AccessRuntime {
    fn open(directory: &std::path::Path) -> Self {
        match AccessStore::open_in(directory) {
            Ok(store) => match CredentialSnapshot::load(&store) {
                Ok(snapshot) => Self {
                    store: std::sync::RwLock::new(Some(store)),
                    credentials: std::sync::RwLock::new(Some(Arc::new(snapshot))),
                    init_error: std::sync::RwLock::new(None),
                },
                Err(error) => Self::failed(error.to_string()),
            },
            Err(error) => Self::failed(error.to_string()),
        }
    }

    fn failed(error: String) -> Self {
        Self {
            store: std::sync::RwLock::new(None),
            credentials: std::sync::RwLock::new(None),
            init_error: std::sync::RwLock::new(Some(error)),
        }
    }

    pub fn ensure_ready(&self) -> Result<(), String> {
        if let Some(error) = self.init_error.read().unwrap().clone() {
            return Err(format!("API access store unavailable: {error}"));
        }
        if self.store.read().unwrap().is_none() || self.credentials.read().unwrap().is_none() {
            return Err("API access store unavailable".to_string());
        }
        Ok(())
    }

    pub fn store(&self) -> Result<AccessStore, String> {
        self.ensure_ready()?;
        self.store
            .read()
            .unwrap()
            .clone()
            .ok_or_else(|| "API access store unavailable".to_string())
    }

    pub fn credentials(&self) -> Result<Arc<CredentialSnapshot>, String> {
        self.ensure_ready()?;
        self.credentials
            .read()
            .unwrap()
            .clone()
            .ok_or_else(|| "API credential cache unavailable".to_string())
    }

    pub fn refresh_credentials(&self) -> Result<(), String> {
        let store = self.store()?;
        let snapshot = CredentialSnapshot::load(&store).map_err(|error| error.to_string())?;
        *self.credentials.write().unwrap() = Some(Arc::new(snapshot));
        Ok(())
    }
}

impl AppState {
    pub fn new(config: HipfireConfig) -> Arc<Self> {
        Self::new_loaded(LoadedConfig::from_config(config))
    }

    pub fn new_loaded(loaded_config: LoadedConfig) -> Arc<Self> {
        let training_runs_dir =
            hipfire_operator::training::training_runs_dir(hipfire_config::hipfire_dir());
        Self::new_loaded_with_training_runs_dir(loaded_config, training_runs_dir)
    }

    pub fn new_loaded_with_training_runs_dir(
        loaded_config: LoadedConfig,
        training_runs_dir: PathBuf,
    ) -> Arc<Self> {
        Self::new_loaded_with_directories(
            loaded_config,
            training_runs_dir,
            hipfire_config::hipfire_dir(),
        )
    }

    pub fn new_loaded_with_directories(
        loaded_config: LoadedConfig,
        training_runs_dir: PathBuf,
        access_dir: PathBuf,
    ) -> Arc<Self> {
        let scheduler_env = SchedulerPolicyEnv::from_pairs(std::env::vars());
        let config = loaded_config.config.clone();
        let sdapi_output_root = PathBuf::from(&config.sdapi_output_root);
        let sdapi_geometry_limits = crate::routes::sdapi::SdapiGeometryLimits::from_config(&config);
        let models_dir = hipfire_config::configured_models_dir(&config);
        let models_network_dir = config
            .models_network_dir
            .as_deref()
            .filter(|dir| !dir.is_empty())
            .map(PathBuf::from);
        let access = AccessRuntime::open(&access_dir);
        let usage_writer = access.store().ok().map(UsageWriter::spawn);
        Arc::new(Self {
            engine: Mutex::new(None),
            loaded_config: Mutex::new(loaded_config),
            config: Mutex::new(config),
            loaded_model_path: Mutex::new(None),
            loaded_model_cache_capable: Mutex::new(None),
            loaded_model_max_seq: Mutex::new(None),
            loaded_models: Mutex::new(HashMap::new()),
            prefill_scheduler: Mutex::new(PriorityPrefillScheduler::new(scheduler_env)),
            // Generous capacity: admission is behavior-neutral for now (the
            // scheduler provides priority ordering, microbatch grouping, and
            // resource leases; real inventory-based capacity is a follow-up).
            // Unbounded queue (max_queued=0) mirrors the prior inbox behaviour.
            work_scheduler: Mutex::new(ContinuousWorkScheduler::new(
                WorkloadResources {
                    system_memory_bytes: u64::MAX,
                    vram_bytes: u64::MAX,
                    gpu_slots: u32::MAX,
                    npu_slots: u32::MAX,
                    cpu_threads: u32::MAX,
                },
                0,
                0,
            )),
            selected_prefill_requests: Mutex::new(HashSet::new()),
            prefill_dispatch: Mutex::new(()),
            prefill_notify: Notify::new(),
            batch_inbox: Mutex::new(HashMap::new()),
            batch_runner_active: std::sync::atomic::AtomicBool::new(false),
            bind: StdMutex::new(None),
            batch_telemetry: Mutex::new(crate::batch_runner::BatchTelemetry::default()),
            responses_contexts: Mutex::new(HashMap::new()),
            responses_order: Mutex::new(VecDeque::new()),
            files: Mutex::new(HashMap::new()),
            file_order: Mutex::new(VecDeque::new()),
            batches: Mutex::new(HashMap::new()),
            batch_order: Mutex::new(VecDeque::new()),
            diffusion_pipelines: Mutex::new(HashMap::new()),
            diffusion_runtime_default: StdMutex::new(
                DiffusionGenerationRuntimeOptions::cpu_reference(),
            ),
            sdapi_options: Mutex::new(HashMap::new()),
            sdapi_progress: Arc::new(StdMutex::new(SdapiProgressState::default())),
            training_runs_dir,
            sdapi_output_root,
            sdapi_geometry_limits,
            models_dir,
            models_network_dir,
            admin_secret: hipfire_config::ensure_admin_secret().unwrap_or_default(),
            admin_sessions: Mutex::new(HashMap::new()),
            access,
            rate_limiter: RateLimiter::default(),
            usage_writer,
        })
    }

    /// Resolve the daemon's default diffusion backend once at `serve` launch.
    /// HIP/ROCm-first: detect the primary GPU and use it; honor
    /// `HIPFIRE_DIFFUSION_CPU_REFERENCE` for the CPU oracle; if no GPU is
    /// available and CPU was not requested, warn and keep the CPU reference so
    /// the daemon still serves (slowly) rather than failing every request.
    pub fn resolve_diffusion_runtime_default(&self) {
        let resolved = if DiffusionGenerationRuntimeOptions::cpu_reference_requested() {
            eprintln!(
                "[hipfire] diffusion: HIPFIRE_DIFFUSION_CPU_REFERENCE set; using the CPU reference oracle"
            );
            DiffusionGenerationRuntimeOptions::cpu_reference()
        } else {
            match hipfire_runtime::multi_gpu::resolve_primary_device(None) {
                Ok(device) => DiffusionGenerationRuntimeOptions::rocm_hybrid(device),
                Err(error) => {
                    eprintln!(
                        "[hipfire] diffusion: no ROCm device resolved ({error}); falling back to \
                         the slow CPU reference oracle. Set HIPFIRE_DIFFUSION_CPU_REFERENCE=1 to \
                         silence this warning."
                    );
                    DiffusionGenerationRuntimeOptions::cpu_reference()
                }
            }
        };
        *self.diffusion_runtime_default.lock().unwrap() = resolved;
    }

    /// The daemon's resolved default diffusion backend (see
    /// [`resolve_diffusion_runtime_default`]).
    pub fn diffusion_runtime_default(&self) -> DiffusionGenerationRuntimeOptions {
        *self.diffusion_runtime_default.lock().unwrap()
    }
}

pub type SharedState = Arc<AppState>;
