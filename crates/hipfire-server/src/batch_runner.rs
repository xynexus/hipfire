//! Server-side continuous-batching orchestrator (Phase 1 of
//! docs/plans/2026-07-18-continuous-scheduler-headline.md).
//!
//! The daemon already executes fused batched prefill + decode over N
//! co-resident sessions (`generate_batch_prefill` / `generate_batch_decode_step`);
//! nothing in serving drives that lifecycle. This module is the missing middle:
//! a request registry the HTTP routes park into, plus the telemetry surface the
//! `/health` route and `smoke-server-decode-batch.sh` read.
//!
//! It holds the request registry + telemetry, the runner task that owns the
//! engine and drives reserve -> batch_prefill -> decode-step loop -> release,
//! and the batch-eligibility gate ([`batch_eligible`], a declared arch
//! capability via the arch-specs registry). Routing to the runner is on by
//! default but only for eligible requests; `HIPFIRE_SERVER_PREFILL_BATCH=0` is
//! the kill switch, and any ineligible request falls back to the legacy
//! per-request path (chat.rs engine.lock).

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use hipfire_arch_api::ArchRegistry;
use hipfire_daemon_adapter::{DaemonEngine, EmbedRequest, EmbeddingVector};
use tokio::sync::{mpsc, oneshot, Mutex};

use crate::state::SharedState;

/// The arch registry, built once from the force-linked `-spec` crates
/// (`hipfire-arch-specs` is a dep so their capability registrations are linked).
fn arch_registry() -> &'static ArchRegistry {
    static REG: OnceLock<ArchRegistry> = OnceLock::new();
    REG.get_or_init(ArchRegistry::build)
}

/// Whether a request whose model reports arch tag `arch` may be routed through the
/// continuous-batching
/// runner. Eligibility is a **declared arch capability** (`ContinuousBatching`)
/// resolved via the arch-specs registry, AND the runtime envelope the daemon's
/// fused batch path requires. Anything ineligible falls back to the legacy
/// per-request path — the safe default that always works.
pub fn batch_eligible(arch: Option<&str>, batch_prefill_capable: Option<bool>) -> bool {
    // `batch_prefill_capable` is the daemon's own probe of the LOADED model,
    // so a state the batched prefill refuses (a speculative-decode drafter,
    // CASK eviction) routes to the legacy path instead of being dispatched and
    // failing the whole cycle. `None` is an older daemon that does not report
    // it; keep the previous behaviour rather than silently disabling batching.
    if batch_prefill_capable == Some(false) {
        return false;
    }
    arch_supports_continuous_batching(arch) && batch_envelope_ok()
}

fn arch_supports_continuous_batching(arch: Option<&str>) -> bool {
    let Some(arch) = arch else {
        return false;
    };
    // `resolve`, NOT `find_by_model_type`: the string's provenance is exactly the
    // ambiguity `resolve` exists for. The daemon reports `loaded.family` whenever the
    // model has a registered backend — "qwen3.5", with a dot — while `model_types` holds
    // "qwen3_5"/"qwen3_5_text". An exact match on model_types therefore missed EVERY
    // qwen3.5 model, so the whole family fell to the legacy per-request path and
    // concurrent requests never fused. Measured before this fix: three prefix-sharing
    // requests sent 0.4 ms apart took 72.3 s against 14.1 s for one, with
    // `prefill_batch.selected_batch_size` stuck at 0 across 191 requests while
    // `/health` cheerfully reported batching "enabled" and "supported".
    // `resolve` is separator- and case-insensitive and tries model_types before family,
    // so it accepts either form on purpose.
    arch_registry()
        .resolve(arch)
        .and_then(|a| a.caps.continuous_batching)
        .is_some()
}

/// Runtime toggles the fused batch path can't (or shouldn't yet) run under.
/// Conservative: DFlash, pipeline-parallel > 1, and hierarchical KV route to the
/// proven legacy path. (Hierarchical KV would fall to the serial-swap decode
/// backend inside the batch op; excluded here until validated.)
///
/// These read ENV, so they only see a drafter passed as HIPFIRE_DFLASH_DRAFT —
/// not one found by sibling discovery or carried in an embedded manifest. The
/// loaded-model answer is `batch_prefill_capable` above; this stays as a
/// pre-load backstop.
fn batch_envelope_ok() -> bool {
    let hierarchical = std::env::var("HIPFIRE_KV_HIERARCHICAL").ok().as_deref() == Some("1");
    let dflash = std::env::var("HIPFIRE_DFLASH_DRAFT")
        .ok()
        .map(|v| !v.trim().is_empty())
        .unwrap_or(false);
    let pp_multi = std::env::var("HIPFIRE_PP_LAYERS")
        .ok()
        .map(|v| v.split(',').filter(|s| !s.trim().is_empty()).count() > 1)
        .unwrap_or(false);
    !hierarchical && !dflash && !pp_multi
}

/// One event delivered from the batch runner back to a waiting request task.
#[derive(Clone, Debug)]
pub enum BatchEvent {
    /// Visible text of one decoded token for this request's session.
    Token(String),
    /// Terminal completion. Carries the raw daemon `done` payload so the route
    /// can build its response without this module depending on the generate
    /// crate's event types.
    Done(serde_json::Value),
    /// Terminal error for this request.
    Error(String),
}

/// A request admitted to the batch path but not yet complete. The route builds
/// one, inserts it under its `req_id`, then awaits the paired receiver.
pub struct PendingRequest {
    /// Per-session prefill/decode inputs (`spec.id` is the request id).
    pub spec: SessionSpec,
    /// Resolved daemon worker key this request must run on. Requests only batch
    /// together when their `worker_key_id` (and cache mode) match.
    pub worker_key_id: String,
    /// Delivers tokens / completion back to the awaiting HTTP task.
    pub tx: mpsc::UnboundedSender<BatchEvent>,
    /// Set when this request was preempted mid-generation and re-queued: the
    /// daemon session is already prefilled and resident at this `logical_position`,
    /// so the resume cycle skips prefill and continues decoding from here. `None`
    /// for a fresh request. `spec.max_tokens` carries the remaining token budget.
    pub resume_position: Option<usize>,
}

/// Outcome of one `run_batch_cycle`. `Parked` carries the still-active requests
/// (with resume cursors) when the batch yielded to higher-priority work before
/// finishing; the daemon sessions stay resident so no decoded work is discarded.
enum CycleOutcome {
    Completed,
    Parked(Vec<PendingRequest>),
    /// The batch's prefill ran out of device memory before anything was decoded.
    /// Its sessions are released and every request is still waiting, so the
    /// runner can retry them in smaller batches instead of failing them all.
    OutOfMemory(Vec<PendingRequest>),
}

/// hipErrorOutOfMemory, as the daemon's allocator reports it.
fn is_device_oom(err: &str) -> bool {
    err.contains("hipError=2)")
}

/// A unit of GPU work admitted to the runner (P5). The runner is the single GPU
/// executor: it leases a workload from the scheduler and dispatches by variant.
/// A lease is single-class, so a lease's jobs are all the same variant.
pub enum ScheduledJob {
    /// Fused text prefill+decode (the park/resume, preemptible path).
    Text(PendingRequest),
    /// A single embedding request. Short; runs to completion (no parking yet).
    Embed(EmbedJob),
    /// A txt2img/img2img diffusion job (the restart-from-seed preemptible path).
    Image(ImageJob),
    /// A steering/abliteration control op (capture session / apply / clear),
    /// routed through the runner so it shares the one GPU arbiter.
    Steer(SteerJob),
    /// A drafter-training run, executed in-daemon on the runner-owned engine.
    /// Long (minutes) and HOLDS the runner turn to completion — see `Dispatch::Train`.
    Train(TrainJob),
}

/// A drafter-training job run on the runner-owned engine. `req` is the raw
/// `train_drafter` request Value (the daemon re-parses raw JSON for this op); the
/// terminal `train_done` payload (or an error) goes back over `tx`.
pub struct TrainJob {
    pub req: serde_json::Value,
    pub tx: oneshot::Sender<Result<serde_json::Value, String>>,
}

/// A steer control op run on the runner-owned engine. Capture is a WHOLE session
/// (begin → prefill each prompt → finish) executed atomically in one runner turn:
/// the capture hook is process-global, so an interleaved text generate would fold
/// its residuals into the means — the atomic session is what prevents that. Apply
/// and Clear are instantaneous daemon state ops (an active apply steers ordinary
/// generation, which already rides the runner, so it needs no exclusivity).
pub struct SteerJob {
    pub op: SteerOp,
    /// `Some(means)` for a finished capture session; `None` for apply/clear.
    pub tx: oneshot::Sender<Result<Option<Vec<Vec<f32>>>, String>>,
}

pub enum SteerOp {
    /// Atomic capture: begin(num_layers, hidden) → steer_capture(system, user) for
    /// each prompt → finish → per-block means. One runner turn, no interleaving.
    CaptureSession {
        num_layers: usize,
        hidden: usize,
        prompts: Vec<(String, String)>,
    },
    /// Install an apply session (directions/mode/strength/layer range).
    BeginApply(hipfire_daemon_adapter::SteerApplyRequest),
    /// Tear down any active steer session.
    Clear,
}

/// An embedding request routed through the runner instead of locking the engine
/// directly, so it shares the one GPU arbiter (and never races the runner's
/// engine `take`). The result goes back over a oneshot.
pub struct EmbedJob {
    pub req: EmbedRequest,
    pub tx: oneshot::Sender<Result<Vec<EmbeddingVector>, String>>,
}

/// A request for an exclusive GPU turn to run diffusion. The image route
/// registers one and awaits `grant`; the runner leases it, grants the turn, and
/// **holds** it (parks its loop) until the route releases — that hold is what
/// serializes image generation with text/embed on the single GPU.
///
/// The runner does **not** run the diffusion itself: the route runs it in its own
/// `spawn_blocking` (the proven diffusion execution path — running the blocking
/// pipeline from the runner task instead wedges before the first sampler step).
/// The runner only arbitrates: grant a turn, watch for a higher-priority waiter
/// (setting the preempt flag at a sampler-step boundary), and hold until done.
pub struct ImageJob {
    pub grant: oneshot::Sender<ImageTurn>,
    pub priority: u8,
}

/// The granted GPU turn handed to the image route: a preempt flag the diffusion
/// progress callback checks each sampler step (set by the runner's watcher when a
/// strictly-higher-priority workload appears), and a `release` channel the route
/// signals when the diffusion finishes or is interrupted so the runner frees the
/// turn and dispatches the next workload.
pub struct ImageTurn {
    pub preempt: Arc<AtomicBool>,
    pub release: oneshot::Sender<()>,
}

/// Min decode/sampler steps a job runs before it may be preempted (anti-thrash
/// floor). Shared by text decode and image sampler-step preemption.
pub fn min_quantum() -> u32 {
    std::env::var("HIPFIRE_SERVER_PREEMPT_MIN_QUANTUM")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or(4)
}

/// Max simultaneously-parked batches. Each parked batch pins its sessions
/// resident in the daemon, so nested preemption is bounded to cap VRAM.
fn preempt_max_depth() -> usize {
    std::env::var("HIPFIRE_SERVER_PREEMPT_MAX_DEPTH")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&d| d >= 1)
        .unwrap_or(4)
}

/// Registry of in-flight runner jobs (any class), keyed by workload id.
pub type BatchInbox = Mutex<HashMap<String, ScheduledJob>>;

/// Live counters the runner writes and `/health` reads. Field names mirror the
/// `smoke-server-decode-batch.sh` telemetry contract so the acceptance test can
/// assert against them once the runner populates them.
#[derive(Clone, Debug, Default)]
pub struct BatchTelemetry {
    pub total_batches: u64,
    pub serial_batches: u64,
    pub selected_batch_size: u64,
    pub last_backend: Option<String>,
    pub last_chunk_count: u64,
    pub last_chunk_size: u64,
    pub last_decode_ms: Option<f64>,
    pub compatible_state_kinds: Vec<String>,
    pub cached_prefix_tokens: Option<u64>,
    pub fallback_reason: Option<String>,
    pub active_sessions: u64,
    pub resident_runtime_sessions: u64,
    pub resident_decode_sessions: u64,
    pub pending_requests: u64,
    /// Times a running batch yielded to a higher-priority workload (Phase 4).
    pub preemptions: u64,
    /// Times a running image yielded at a sampler-step boundary and was
    /// restarted from seed (Phase 5.2).
    pub image_preemptions: u64,
}

/// The generic seam the runner groups by: two requests may share one fused
/// prefill+decode batch only when their [`batch_key`](BatchableSession::batch_key)
/// values match. The key must fold in every property that changes the fused GPU
/// invocation — worker/model, cache mode, and state kinds. qwen35 chat requests
/// are the first real impl; the runner never inspects arch-specific types, only
/// this key, so any arch the daemon can batch-prefill coalesces for free.
pub trait BatchableSession {
    fn batch_key(&self) -> String;
}

/// Minimal per-session inputs the runner needs to build the daemon batch
/// requests. The route fills one of these per admitted request; how the prompt
/// is rendered (chat template, raw text) is the route's concern, not the
/// protocol layer's.
#[derive(Clone, Debug)]
pub struct SessionSpec {
    pub id: String,
    /// Current user-turn text. Required by the daemon validator (a session must
    /// carry exactly one of `prompt` / `suffix_tokens`); the daemon renders it
    /// with `messages_history` + `system_prompt` via the chat template.
    pub prompt: String,
    /// Prior conversation turns as prompt messages (JSON `Vec<PromptMessage>`),
    /// so the batched prefill templates identically to the serial path.
    pub messages_history: Option<serde_json::Value>,
    /// System prompt, if any.
    pub system_prompt: Option<String>,
    /// Sequence-state kinds this session needs resident. qwen35 (DeltaNet)
    /// needs both `attention_kv` and `deltanet_recurrent`; a plain-attention
    /// arch needs only `attention_kv`.
    pub state_kinds: Vec<String>,
    /// Assistant-turn prefix mode passed through to the daemon template.
    pub assistant_prefix: String,
    /// Thinking budget marker. The daemon derives `enable_thinking =
    /// max_think_tokens != 1`, so `1` disables thinking and `0` enables it.
    pub max_think_tokens: u32,
    /// Total tokens this request may still generate.
    pub max_tokens: usize,
    /// The request's tool declarations, rendered into the chat template by the
    /// daemon. Without them the model never learns a tool exists and answers in
    /// prose, or invents a call nothing will parse.
    pub tools: Option<serde_json::Value>,
}

/// Post-prefill decode cursor for one resident session, sourced from that
/// session's `generate_batch_prefill_session_done` (`logical_position`) and the
/// request's remaining token budget.
#[derive(Clone, Debug)]
pub struct DecodeCursor {
    pub id: String,
    pub logical_position: usize,
    pub max_tokens_remaining: usize,
}

/// How one session of a prefill uses the prefix cache.
#[derive(Clone, Debug)]
pub enum PrefixReuse {
    /// Fork from a cached checkpoint and prefill only the rest of the prompt; the
    /// boundaries after the fork point are checkpointed in turn.
    Attach(PrefixEntry),
    /// Nothing cached: have the daemon checkpoint this prompt's boundaries.
    Mint,
}

/// A chat-template boundary checkpoint the daemon holds for reuse.
#[derive(Clone, Debug, PartialEq)]
pub struct PrefixEntry {
    pub worker: String,
    /// `prefix_hash` as the daemon reports it: `{algorithm, value, prefix_len}`.
    pub prefix_hash: serde_json::Value,
    pub prefix_len: usize,
    pub checkpoint_id: String,
    /// Requests that attached to it. Decides eviction: see `PrefixIndex::insert`.
    pub hits: u32,
    /// Which prefill minted it (the index's batch counter). Decides eviction too.
    pub batch: u64,
    /// Set only on a session's attach PLAN, never in the index: an extra
    /// boundary (absolute prompt position) for the daemon to checkpoint — the
    /// deepest one this session shares with a sibling that waits to attach it.
    pub mint_at: Option<usize>,
}

impl PrefixEntry {
    fn hash_value(&self) -> &str {
        self.prefix_hash["value"].as_str().unwrap_or_default()
    }
}

/// Prefix checkpoints the server has asked the daemon to keep, least recently
/// used first. Qwen3.5's DeltaNet state cannot be rewound to a shorter prefix the
/// way a KV cache can be truncated, so reuse means forking a state snapshotted at
/// exactly that boundary — which is what these are.
#[derive(Default)]
pub struct PrefixIndex {
    entries: Vec<PrefixEntry>,
    batches: u64,
}

/// Checkpoints held for reuse, per server. Each is a resident daemon session — on
/// the 27B ~72 MB of DeltaNet state plus ~13 KB per prefix token of KV — and the
/// daemon evicts checkpoints over `HIPFIRE_SCHED_RESIDENT_STATE_MAX` (96, sized
/// for this plus a full batch of fresh mints), so keep this well under that.
/// ponytail: a count, not bytes; a byte budget once prefixes get long (a 30K-token
/// prefix is ~0.4 GB of KV per checkpoint).
fn prefix_cache_max() -> usize {
    std::env::var("HIPFIRE_SERVER_PREFIX_CACHE_MAX")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(16)
}

impl PrefixIndex {
    /// The longest of `candidates` (a preflight's `prefixes`) that is cached for
    /// `worker`, marked most recently used.
    pub fn lookup(
        &mut self,
        worker: &str,
        candidates: &[serde_json::Value],
    ) -> Option<PrefixEntry> {
        let best = candidates
            .iter()
            .filter_map(|c| {
                let value = c["value"].as_str()?;
                self.entries
                    .iter()
                    .position(|e| e.worker == worker && e.hash_value() == value)
            })
            .max_by_key(|&i| self.entries[i].prefix_len)?;
        let mut entry = self.entries.remove(best);
        entry.hits += 1;
        self.entries.push(entry.clone());
        Some(entry)
    }

    /// Record a checkpoint. Returns checkpoint ids the caller must release: a
    /// duplicate of a hash already held, or the evictions that bring the index
    /// back under its cap.
    ///
    /// Least recently used goes first; a lookup, an insert, and a duplicate mint
    /// of a held hash all count as use. `record_prefix_checkpoints` inserts one
    /// prefill's checkpoints longest first, so its shortest boundary — the shared
    /// system turn — is the most recent of them, and every later prompt that shares
    /// it refreshes it again.
    ///
    /// History: this used to evict never-attached entries first (oldest batch,
    /// then longest), to protect the shared system turn. Under concurrent tool
    /// loops that is backwards: each chain's newest checkpoint is never-attached
    /// until that chain's next step, so one chain's insert evicted another's fresh
    /// tail and the victim re-prefilled thousands of tokens per step (measured: 3
    /// research chains on CAE pinned to checkpoints 3-6 steps old).
    pub fn insert(&mut self, entry: PrefixEntry) -> Vec<String> {
        if let Some(i) = self
            .entries
            .iter()
            .position(|e| e.worker == entry.worker && e.hash_value() == entry.hash_value())
        {
            let held = self.entries.remove(i);
            self.entries.push(held);
            return vec![entry.checkpoint_id];
        }
        // The cap is PER WORKER, and only this worker's entries are evicted: the
        // ids returned are released to this worker's daemon session registry.
        // Evicting across workers sent a 35B checkpoint's release to the 27B (a
        // no-op there) and the 35B's checkpoints piled up -- 40 resident against a
        // cap of 16, GTT 98 GiB, over one two-model CAE swarm turn.
        let worker = entry.worker.clone();
        self.entries.push(entry);
        let mut evicted = Vec::new();
        while self.entries.iter().filter(|e| e.worker == worker).count() > prefix_cache_max() {
            // Front = least recently used; the entry just inserted is at the back.
            let victim = self
                .entries
                .iter()
                .position(|e| e.worker == worker)
                .expect("this worker has entries");
            evicted.push(self.entries.remove(victim).checkpoint_id);
        }
        evicted
    }

    /// Forget a checkpoint (the daemon no longer has it, or it failed to attach).
    pub fn forget(&mut self, checkpoint_id: &str) {
        self.entries.retain(|e| e.checkpoint_id != checkpoint_id);
    }
}

/// One session of a `generate_batch_prefill` or `prefix_hash_preflight`. Both
/// must render the prompt identically, or a preflight hash would never match the
/// checkpoint a prefill minted.
fn session_json(s: &SessionSpec, reuse: Option<&PrefixReuse>) -> serde_json::Value {
    // The daemon reads assistant_prefix / max_think_tokens /
    // semantic_boundary_checkpoints from a nested `params` object, the
    // conversation from `messages`, and the system prompt from `system`
    // (see hipfire_generate::validate_generate_batch_prefill).
    let mut session = serde_json::json!({
        "id": s.id,
        "prompt": s.prompt,
        "params": {
            "assistant_prefix": s.assistant_prefix,
            "max_think_tokens": s.max_think_tokens,
            // An attached session checkpoints the boundaries past its prefix, so a
            // conversation's next step can attach at this one's end.
            "semantic_boundary_checkpoints": reuse.is_some(),
            // Sizes the session's KV to prompt + this, not the model's max_seq.
            "max_tokens": s.max_tokens,
        },
        "state_handle": {
            "state_kinds": s.state_kinds,
            "logical_position": 0,
            "cached_prefix_tokens": 0,
        },
    });
    if let Some(PrefixReuse::Attach(entry)) = reuse {
        let handle = &mut session["state_handle"];
        handle["logical_position"] = serde_json::json!(entry.prefix_len);
        handle["cached_prefix_tokens"] = serde_json::json!(entry.prefix_len);
        handle["runtime_state_handle"] = serde_json::json!(entry.checkpoint_id);
        handle["prefix_hash"] = entry.prefix_hash.clone();
        if let Some(at) = entry.mint_at {
            session["params"]["checkpoint_at"] = serde_json::json!([at]);
        }
    }
    if let Some(obj) = session.as_object_mut() {
        if let Some(history) = &s.messages_history {
            obj.insert("messages".to_string(), history.clone());
        }
        if let Some(system) = &s.system_prompt {
            obj.insert("system".to_string(), serde_json::json!(system));
        }
        if let Some(tools) = &s.tools {
            obj.insert("tools".to_string(), tools.clone());
        }
    }
    session
}

/// Build a `generate_batch_prefill` request for `specs` on `worker_key_id`.
/// Fused-prefill of all sessions in one daemon call; each emits a
/// `generate_batch_prefill_session_done` carrying its `logical_position`.
pub fn build_batch_prefill_request(
    batch_id: &str,
    worker_key_id: &str,
    specs: &[SessionSpec],
    reuse: &HashMap<String, PrefixReuse>,
) -> serde_json::Value {
    let sessions: Vec<serde_json::Value> = specs
        .iter()
        .map(|s| session_json(s, reuse.get(&s.id)))
        .collect();
    serde_json::json!({
        "type": "generate_batch_prefill",
        "id": batch_id,
        "batch_id": batch_id,
        "worker_key_id": worker_key_id,
        "sessions": sessions,
    })
}

/// Build a `prefix_hash_preflight` request: hash `spec`'s chat-template
/// boundaries without prefilling anything.
pub fn build_prefix_preflight_request(
    worker_key_id: &str,
    spec: &SessionSpec,
) -> serde_json::Value {
    serde_json::json!({
        "type": "prefix_hash_preflight",
        "id": format!("preflight-{}", spec.id),
        "worker_key_id": worker_key_id,
        "boundary_policy": "semantic_chat_template",
        "session": session_json(spec, None),
    })
}

/// Decide, per session, whether to attach a cached prefix or mint one, and which
/// sessions should wait for a checkpoint another session is about to mint.
///
/// Only one session per not-yet-cached boundary mints: sessions arriving together
/// with the same prefix would otherwise each snapshot identical state. The rest
/// are returned as deferred, to attach once it exists. A failed preflight leaves
/// the session alone — the cache is an optimisation, never a reason to fail a
/// request.
///
/// Also returns each session's full prompt length from the preflight, which the row
/// budget works from — so it runs even with the cache off (`use_cache` false:
/// lengths only).
async fn plan_prefix_reuse(
    engine: &mut DaemonEngine,
    worker: &str,
    specs: &[SessionSpec],
    index: &mut PrefixIndex,
    use_cache: bool,
) -> (
    HashMap<String, PrefixReuse>,
    Vec<String>,
    HashMap<String, usize>,
) {
    let mut plan = HashMap::new();
    let mut deferred = Vec::new();
    let mut full = HashMap::new();
    // Boundary hashes some earlier session in this batch will checkpoint.
    let mut minting: Vec<String> = Vec::new();
    // Preflight every session first, so planning can see which boundaries the
    // batch's sessions share.
    let mut preflighted: Vec<(&SessionSpec, Vec<serde_json::Value>)> = Vec::new();
    for spec in specs {
        let reply = match engine
            .prefix_hash_preflight(build_prefix_preflight_request(worker, spec))
            .await
        {
            Ok(reply) => reply,
            Err(e) => {
                tracing::debug!("prefix preflight for {}: {e}", spec.id);
                continue;
            }
        };
        if let Some(n) = reply["full"]["prefix_len"].as_u64() {
            full.insert(spec.id.clone(), n as usize);
        }
        if !use_cache {
            continue;
        }
        // `full` is the whole prompt; checkpoints are only ever taken at the
        // boundaries inside it.
        let boundaries: Vec<serde_json::Value> = reply["prefixes"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter(|p| p["boundary"] != "full")
                    .cloned()
                    .collect()
            })
            .unwrap_or_default();
        preflighted.push((spec, boundaries));
    }
    let mut shared: HashMap<String, usize> = HashMap::new();
    for (_, boundaries) in &preflighted {
        for b in boundaries {
            if let Some(h) = b["value"].as_str() {
                *shared.entry(h.to_string()).or_default() += 1;
            }
        }
    }
    for (spec, boundaries) in preflighted {
        if let Some(entry) = index.lookup(worker, &boundaries) {
            // Siblings that hit the same cached prefix and then share more of
            // the prompt (`p1 + p3 + pD/pE/pF`: all hit `p1`) used to each
            // prefill the shared part. An attached session checkpoints only its
            // last boundary, which is its own (the final message differs), so the
            // first session that shares a deeper boundary with another one in
            // this batch asks the daemon to checkpoint that boundary too
            // (`mint_at`), and the rest wait a round to attach it.
            let mut entry = entry;
            let deeper_shared = boundaries
                .iter()
                .filter(|b| b["prefix_len"].as_u64().unwrap_or(0) as usize > entry.prefix_len)
                .filter(|b| {
                    b["value"]
                        .as_str()
                        .is_some_and(|h| shared.get(h) > Some(&1))
                })
                .max_by_key(|b| b["prefix_len"].as_u64().unwrap_or(0));
            if let Some(b) = deeper_shared {
                let h = b["value"].as_str().unwrap_or_default().to_string();
                if minting.contains(&h) {
                    tracing::debug!(
                        "session {}: deferred; a sibling checkpoints its deeper prefix",
                        spec.id
                    );
                    deferred.push(spec.id.clone());
                    continue;
                }
                minting.push(h);
                entry.mint_at = b["prefix_len"].as_u64().map(|n| n as usize);
            }
            tracing::debug!(
                "session {}: attaching cached prefix {} ({} tokens)",
                spec.id,
                entry.checkpoint_id,
                entry.prefix_len
            );
            plan.insert(spec.id.clone(), PrefixReuse::Attach(entry));
            continue;
        }
        tracing::debug!(
            "session {}: no cached prefix among {:?}; index holds {:?}",
            spec.id,
            boundaries
                .iter()
                .map(|b| (
                    b["prefix_len"].as_u64().unwrap_or(0),
                    b["value"].as_str().unwrap_or("")
                ))
                .collect::<Vec<_>>(),
            index
                .entries
                .iter()
                .map(|e| (e.prefix_len, e.hash_value()))
                .collect::<Vec<_>>()
        );
        // Mint unless an earlier session already checkpoints one of these
        // boundaries: sessions sharing a system turn differ in their tails, so
        // keying on the longest boundary had every one of them mint.
        let hashes: Vec<String> = boundaries
            .iter()
            .filter_map(|p| p["value"].as_str().map(str::to_string))
            .collect();
        if hashes.iter().any(|h| minting.contains(h)) {
            deferred.push(spec.id.clone());
        } else if !hashes.is_empty() {
            minting.extend(hashes);
            plan.insert(spec.id.clone(), PrefixReuse::Mint);
        }
    }
    (plan, deferred, full)
}

/// Rows each session still has to prefill under `reuse`: its full length, less the
/// prefix it attaches. Recomputed whenever an attach is dropped, since a session
/// that loses its prefix prefills in full.
fn rows_to_prefill(
    full: &HashMap<String, usize>,
    reuse: &HashMap<String, PrefixReuse>,
) -> HashMap<String, usize> {
    full.iter()
        .map(|(id, &n)| {
            let attached = match reuse.get(id) {
                Some(PrefixReuse::Attach(entry)) => entry.prefix_len,
                _ => 0,
            };
            (id.clone(), n.saturating_sub(attached))
        })
        .collect()
}

/// Prompt rows one daemon prefill call may carry, `HIPFIRE_SERVER_PREFILL_ROW_BUDGET`
/// (default 8192; 0 = no limit).
///
/// Batched prefill sizes its scratch to every row of the call, and the model keeps
/// that scratch at the largest size it has seen: 16 prompts of 16.7K tokens left ~90
/// GB pinned for the life of the process, and every later large request failed or
/// crawled against it. At 8192 rows the scratch stays ~3 GB on the 27B.
fn prefill_row_budget() -> usize {
    std::env::var("HIPFIRE_SERVER_PREFILL_ROW_BUDGET")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(8192)
}

/// Split `specs` into prefill calls of at most `budget` rows each, in order. A
/// session larger than the budget goes alone — the single-session path prefills in
/// bounded chunks of its own. A session whose rows are unknown (its preflight
/// failed) also goes alone, counted as a full budget. Decode is unaffected: every
/// session still decodes in one batch.
fn prefill_groups(
    specs: &[SessionSpec],
    rows: &HashMap<String, usize>,
    budget: usize,
) -> Vec<Vec<SessionSpec>> {
    if budget == 0 {
        return vec![specs.to_vec()];
    }
    let mut groups: Vec<Vec<SessionSpec>> = Vec::new();
    let mut current: Vec<SessionSpec> = Vec::new();
    let mut used = 0usize;
    for spec in specs {
        let n = rows.get(&spec.id).copied().unwrap_or(budget);
        if !current.is_empty() && used + n > budget {
            groups.push(std::mem::take(&mut current));
            used = 0;
        }
        current.push(spec.clone());
        used += n;
    }
    if !current.is_empty() {
        groups.push(current);
    }
    groups
}

/// Prefill `specs` under `reuse` in calls of at most `budget` rows, indexing what
/// each call mints.
///
/// An attach can fail for reasons the index cannot see (the daemon evicted the
/// checkpoint). Drop just the checkpoint the error names — or every attach in the
/// call, if it names none — and retry, re-splitting by the budget: a session that
/// lost its prefix now prefills in full, and retrying it inside the group formed
/// for its attached size put 15 full 16.7K prompts in one call and ran the device
/// out of memory.
#[allow(clippy::too_many_arguments)]
async fn prefill_budgeted(
    engine: &mut DaemonEngine,
    batch_id: &str,
    worker: &str,
    specs: &[SessionSpec],
    reuse: &mut HashMap<String, PrefixReuse>,
    full: &HashMap<String, usize>,
    budget: usize,
    index: &mut PrefixIndex,
) -> anyhow::Result<Vec<serde_json::Value>> {
    let mut events = Vec::new();
    let mut queue: std::collections::VecDeque<Vec<SessionSpec>> =
        prefill_groups(specs, &rows_to_prefill(full, reuse), budget).into();
    let mut calls = 0usize;
    while let Some(group) = queue.pop_front() {
        let id = if calls == 0 {
            batch_id.to_string()
        } else {
            format!("{batch_id}-g{calls}")
        };
        calls += 1;
        let result = engine
            .generate_batch_prefill(build_batch_prefill_request(&id, worker, &group, reuse))
            .await;
        let e = match result {
            Ok(call_events) => {
                record_prefix_checkpoints(engine, worker, &call_events, index).await;
                events.extend(call_events);
                continue;
            }
            Err(e) => e,
        };
        let message = e.to_string();
        let attached: Vec<String> = group
            .iter()
            .filter_map(|s| match reuse.get(&s.id) {
                Some(PrefixReuse::Attach(entry)) => Some(entry.checkpoint_id.clone()),
                _ => None,
            })
            .collect();
        if attached.is_empty() {
            return Err(e);
        }
        let named: Vec<String> = attached
            .iter()
            .filter(|id| message.contains(id.as_str()))
            .cloned()
            .collect();
        let dropping = if named.is_empty() { attached } else { named };
        tracing::warn!(
            "prefill with cached prefixes failed ({message}); retrying without {}",
            dropping.join(", ")
        );
        for s in &group {
            if let Some(PrefixReuse::Attach(entry)) = reuse.get(&s.id) {
                if dropping.contains(&entry.checkpoint_id) {
                    index.forget(&entry.checkpoint_id);
                    reuse.remove(&s.id);
                }
            }
        }
        let handles: Vec<String> = group.iter().map(|s| s.id.clone()).collect();
        let _ = engine
            .release_sessions(build_release_request(worker, &handles))
            .await;
        let regrouped = prefill_groups(&group, &rows_to_prefill(full, reuse), budget);
        for g in regrouped.into_iter().rev() {
            queue.push_front(g);
        }
    }
    Ok(events)
}

/// Prefill `specs`, reusing cached prefixes. Sessions sharing a boundary with one
/// being minted in this same batch wait a round: round 1 prefills everything else,
/// round 2 attaches them to what round 1 checkpointed — so a cold batch of N
/// requests with a common system turn prefills it once, not N times. They still
/// decode together; only prefill is split, by rounds and by the row budget.
async fn prefill_with_prefix_reuse(
    engine: &mut DaemonEngine,
    batch_id: &str,
    worker: &str,
    specs: &[SessionSpec],
    index: &mut PrefixIndex,
) -> anyhow::Result<Vec<serde_json::Value>> {
    let use_cache = prefix_cache_enabled();
    let budget = prefill_row_budget();
    if !use_cache && budget == 0 {
        let request = build_batch_prefill_request(batch_id, worker, specs, &HashMap::new());
        return engine.generate_batch_prefill(request).await;
    }
    const MAX_ROUNDS: usize = 3;
    let mut events = Vec::new();
    let mut pending: Vec<SessionSpec> = specs.to_vec();
    for round in 0..MAX_ROUNDS {
        let (mut reuse, mut deferred, full) =
            plan_prefix_reuse(engine, worker, &pending, index, use_cache).await;
        if round + 1 == MAX_ROUNDS {
            deferred.clear();
        }
        let (now, later): (Vec<SessionSpec>, Vec<SessionSpec>) =
            pending.into_iter().partition(|s| !deferred.contains(&s.id));
        let id = if round == 0 {
            batch_id.to_string()
        } else {
            format!("{batch_id}-r{round}")
        };
        events.extend(
            prefill_budgeted(engine, &id, worker, &now, &mut reuse, &full, budget, index).await?,
        );
        if later.is_empty() {
            break;
        }
        pending = later;
    }
    Ok(events)
}

/// Index the checkpoints a prefill minted, releasing duplicates and evictions.
async fn record_prefix_checkpoints(
    engine: &mut DaemonEngine,
    worker: &str,
    events: &[serde_json::Value],
    index: &mut PrefixIndex,
) {
    let mut release = Vec::new();
    index.batches += 1;
    let batch = index.batches;
    for ev in events {
        let Some(checkpoints) = ev["state_handle"]["prefix_checkpoints"].as_array() else {
            continue;
        };
        // Longest first: the shortest (most shared) boundary ends up most recent.
        let mut checkpoints: Vec<&serde_json::Value> = checkpoints.iter().collect();
        checkpoints.sort_by_key(|c| std::cmp::Reverse(c["prefix_len"].as_u64().unwrap_or(0)));
        for c in checkpoints {
            let (Some(id), Some(len)) = (c["checkpoint_id"].as_str(), c["prefix_len"].as_u64())
            else {
                continue;
            };
            release.extend(index.insert(PrefixEntry {
                worker: worker.to_string(),
                prefix_hash: c["prefix_hash"].clone(),
                prefix_len: len as usize,
                checkpoint_id: id.to_string(),
                hits: 0,
                batch,
                mint_at: None,
            }));
        }
    }
    if !release.is_empty() {
        let _ = engine
            .release_sessions(build_release_request(worker, &release))
            .await;
    }
}

/// Prefix reuse on by default; `HIPFIRE_SERVER_PREFIX_CACHE=0` turns it off.
fn prefix_cache_enabled() -> bool {
    !matches!(
        std::env::var("HIPFIRE_SERVER_PREFIX_CACHE").as_deref(),
        Ok("0" | "off" | "false" | "no")
    )
}

/// A request's terminal payload: why it stopped, and its usage in tokens.
fn done_payload(
    finish_reason: &str,
    prompt_tokens: Option<&usize>,
    completion_tokens: usize,
) -> serde_json::Value {
    let mut done = serde_json::json!({
        "finish_reason": finish_reason,
        "completion_tokens": completion_tokens,
    });
    if let Some(n) = prompt_tokens {
        done["prompt_tokens"] = serde_json::json!(n);
    }
    done
}

/// Build one `generate_batch_decode_step` request: advance every resident
/// session by one token in a single fused GPU forward.
pub fn build_batch_decode_request(
    batch_id: &str,
    worker_key_id: &str,
    cursors: &[DecodeCursor],
) -> serde_json::Value {
    let cached_prefix_tokens = cursors
        .iter()
        .map(|cursor| cursor.logical_position)
        .min()
        .unwrap_or_default();
    let sessions: Vec<serde_json::Value> = cursors
        .iter()
        .map(|c| {
            serde_json::json!({
                "id": c.id,
                "session_id": c.id,
                "max_tokens_remaining": c.max_tokens_remaining,
                "logical_position": c.logical_position,
            })
        })
        .collect();
    serde_json::json!({
        "type": "generate_batch_decode_step",
        "id": batch_id,
        "batch_id": batch_id,
        "worker_key_id": worker_key_id,
        "cached_prefix_tokens": cached_prefix_tokens,
        "session_count": cursors.len(),
        "sessions": sessions,
    })
}

/// Build a `release_sessions` request that frees the given resident session
/// handles once their requests complete.
pub fn build_release_request(worker_key_id: &str, handles: &[String]) -> serde_json::Value {
    serde_json::json!({
        "type": "release_sessions",
        "id": "batch-runner-release",
        "worker_key_id": worker_key_id,
        "sessions": handles,
    })
}

impl BatchTelemetry {
    /// `health.decode_batch` view. Mirrors the field names the smoke asserts.
    pub fn decode_health_json(&self) -> serde_json::Value {
        serde_json::json!({
            "enabled": true,
            "total_batches": self.total_batches,
            "serial_batches": self.serial_batches,
            "selected_batch_size": self.selected_batch_size,
            "last_backend": self.last_backend,
            "last_chunk_count": self.last_chunk_count,
            "last_chunk_size": self.last_chunk_size,
            "last_decode_ms": self.last_decode_ms,
            "compatible_state_kinds": self.compatible_state_kinds,
            "cached_prefix_tokens": self.cached_prefix_tokens,
            "fallback_reason": self.fallback_reason,
            "active_sessions": self.active_sessions,
            "preemptions": self.preemptions,
            "image_preemptions": self.image_preemptions,
        })
    }

    /// `health.prefill_batch` residency/pending view.
    pub fn prefill_health_json(&self) -> serde_json::Value {
        serde_json::json!({
            "resident_runtime_sessions": self.resident_runtime_sessions,
            "resident_decode_sessions": self.resident_decode_sessions,
            "pending_requests": self.pending_requests,
            "selected_batch_size": self.selected_batch_size,
        })
    }
}

/// Default max sessions fused into one batch when `HIPFIRE_SERVER_PREFILL_BATCH_MAX`
/// is unset.
///
/// **16 since 2026-08-11, measured** (was 8). This is a hard cap on envelope
/// width, so at 8 a deployment with 16 concurrent sessions got two half-width
/// batches and left throughput on the table. Aggregate tok/s on
/// `Qwen3.6-35B-A3B--oq4`, gfx1103, one daemon lifetime per row, KVarN KV:
///
/// | concurrent | aggregate | achieved width |
/// |---|---|---|
/// | 1 | 7.88 | 1 |
/// | 8 | 9.80 | 8 |
/// | **16** | **10.25** | **16** |
/// | 32 | 9.96 | ~18 (capped by session residency, not by this) |
/// | 64 | 2.22 | collapses — 20/64 sessions survive |
///
/// 16 was where achieved width stopped tracking demand: past it the limit was
/// session residency, and at 64 the batch collapsed outright.
///
/// **64 since 2026-09-30.** The collapse was the daemon's resident-session
/// eviction counting live sessions of the batch's other prefill groups (and the
/// shared prefix checkpoint) against a budget of 32; it now budgets checkpoints
/// only. Measured, `Qwen3.8-27B--oq4.25++`, gfx1151, KVarN, decode tok/s:
///
/// | concurrent | cap 16 | cap 64 |
/// |---|---|---|
/// | 16, short prompts | 101.3 | 101.5 |
/// | 32, short prompts | 83.5 (3 batches) | 93.3 |
/// | 64, short prompts | 93.8 (5 batches) | 121.0 |
/// | 64, 8K shared prefix | — | 82.1 (all 64 correct; 42.7 at 16) |
///
/// Raising the cap does not itself allocate: the sessions are already resident,
/// this only governs how many of them fuse into one step.
const BATCH_MAX_DEFAULT: usize = 64;

/// Spawn the continuous-batching runner. Call once at serve startup when
/// `HIPFIRE_SERVER_PREFILL_BATCH` is enabled. The runner owns `state.engine`
/// for the duration of each batch cycle and returns it between cycles so other
/// engine users (embeddings, sdapi) still interleave.
pub fn spawn_batch_runner(state: SharedState) {
    // Mark the runner live so enqueue-and-await routes (image gen) know a loop is
    // actually draining the inbox; without this an `AppState` built outside a
    // serve loop (unit tests) would park image requests forever.
    state
        .batch_runner_active
        .store(true, std::sync::atomic::Ordering::Relaxed);
    // Supervised: the runner is the only thing draining the batch inbox, so a
    // panic in it used to leave every later request queued forever. Restart it,
    // and if the panic took the checked-out engine with it, forget the loaded
    // models so the next request respawns a worker instead of waiting on an
    // empty slot.
    tokio::spawn(async move {
        loop {
            match tokio::spawn(batch_runner_loop(state.clone())).await {
                Ok(()) => break,
                Err(e) => {
                    tracing::error!("batch runner died ({e}); restarting it");
                    if state.engine.lock().await.is_none() {
                        crate::routes::chat::clear_loaded_model_state_for_failed_daemon(&state)
                            .await;
                    }
                }
            }
        }
    });
}

pub fn batch_max() -> usize {
    std::env::var("HIPFIRE_SERVER_PREFILL_BATCH_MAX")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|n| *n >= 1)
        .unwrap_or(BATCH_MAX_DEFAULT)
}

/// Gather window: after the first request appears, wait this long for more to
/// arrive before forming the batch. This is what makes concurrent requests
/// actually coalesce into one fused call instead of running one-at-a-time.
fn batch_wait_ms() -> u64 {
    std::env::var("HIPFIRE_SERVER_PREFILL_BATCH_WAIT_MS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(10)
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

async fn batch_runner_loop(state: SharedState) {
    // Parked batches form a stack (LIFO): each entry was preempted by a
    // strictly-higher-priority workload, so priorities decrease toward the top.
    // The daemon sessions for parked requests stay resident, so resume skips
    // prefill. Depth is bounded (each level pins resident VRAM) — see
    // `preempt_max_depth`. `.1` is the priority the batch was running at.
    let mut parked: Vec<(Vec<PendingRequest>, u8)> = Vec::new();
    let max_depth = preempt_max_depth();
    // Largest fresh batch known to fit in device memory, learned from prefill OOMs.
    // ponytail: only ratchets down until restart; probe upward if memory is freed
    // (a model unloaded) and batches stay needlessly small.
    let mut fit_cap = usize::MAX;
    // Prefix checkpoints the daemon holds for this runner; see `PrefixIndex`.
    let mut prefix_index = PrefixIndex::default();
    loop {
        // What would the scheduler grant next (honouring aging)? Lower = sooner.
        let waiter = {
            let sched = state.work_scheduler.lock().await;
            sched.peek_next_priority(now_ms())
        };
        // Resume the top parked batch unless a strictly-higher-priority workload
        // is queued (waiter < top parked priority). Nothing queued resumes it.
        let top_parked = parked.last().map(|(_, pri)| *pri);
        let resume_parked = match top_parked {
            Some(pri) => waiter.map_or(true, |top| top >= pri),
            None => false,
        };

        if !resume_parked && waiter.is_none() {
            // Nothing queued and nothing to resume: wait for work.
            tokio::select! {
                _ = state.prefill_notify.notified() => {}
                _ = tokio::time::sleep(Duration::from_millis(5)) => {}
            }
            continue;
        }

        // Select what to run this iteration: a resumed parked text batch (no
        // lease), or a fresh lease — which the runner dispatches by class.
        let dispatch: Dispatch = if resume_parked {
            let (b, pri) = parked.pop().unwrap();
            Dispatch::Text {
                batch: b,
                running_priority: pri,
                lease_id: None,
            }
        } else {
            // Gather window: let concurrent requests accumulate before the
            // scheduler forms the batch (so a burst coalesces).
            let wait_ms = batch_wait_ms();
            if wait_ms > 0 {
                tokio::time::sleep(Duration::from_millis(wait_ms)).await;
            }
            let lease = {
                let mut sched = state.work_scheduler.lock().await;
                sched.next_batch(now_ms())
            };
            let Some(lease) = lease else {
                continue;
            };
            tracing::debug!(
                "lease {} class={:?} workloads={}",
                lease.lease_id,
                lease.class,
                lease.workloads.len()
            );
            let jobs: Vec<ScheduledJob> = {
                let mut inbox = state.batch_inbox.lock().await;
                lease
                    .workloads
                    .iter()
                    .filter_map(|w| inbox.remove(&w.id))
                    .collect()
            };
            if jobs.is_empty() {
                state.work_scheduler.lock().await.complete(lease.lease_id);
                continue;
            }
            let pri = lease
                .workloads
                .first()
                .map(|w| w.priority)
                .unwrap_or(u8::MAX);
            // A lease is single-class; classify by the first job's variant and
            // dispatch the whole (single-variant) set accordingly.
            match jobs.first() {
                Some(ScheduledJob::Embed(_)) => {
                    let embeds = jobs
                        .into_iter()
                        .filter_map(|j| match j {
                            ScheduledJob::Embed(e) => Some(e),
                            _ => None,
                        })
                        .collect();
                    Dispatch::Embed {
                        jobs: embeds,
                        lease_id: lease.lease_id,
                    }
                }
                Some(ScheduledJob::Image(_)) => {
                    // Image leases are singletons (max_microbatch_size 1).
                    let job = jobs
                        .into_iter()
                        .find_map(|j| match j {
                            ScheduledJob::Image(i) => Some(i),
                            _ => None,
                        })
                        .expect("image lease carries one image job");
                    Dispatch::Image {
                        job,
                        running_priority: pri,
                        lease_id: lease.lease_id,
                    }
                }
                Some(ScheduledJob::Steer(_)) => {
                    let jobs = jobs
                        .into_iter()
                        .filter_map(|j| match j {
                            ScheduledJob::Steer(s) => Some(s),
                            _ => None,
                        })
                        .collect();
                    Dispatch::Steer {
                        jobs,
                        lease_id: lease.lease_id,
                    }
                }
                Some(ScheduledJob::Train(_)) => {
                    // Training leases are singletons (max_microbatch_size 1).
                    let job = jobs
                        .into_iter()
                        .find_map(|j| match j {
                            ScheduledJob::Train(t) => Some(t),
                            _ => None,
                        })
                        .expect("train lease carries one train job");
                    Dispatch::Train {
                        job,
                        lease_id: lease.lease_id,
                    }
                }
                _ => {
                    let batch = jobs
                        .into_iter()
                        .filter_map(|j| match j {
                            ScheduledJob::Text(p) => Some(p),
                            _ => None,
                        })
                        .collect();
                    Dispatch::Text {
                        batch,
                        running_priority: pri,
                        lease_id: Some(lease.lease_id),
                    }
                }
            }
        };

        match dispatch {
            Dispatch::Text {
                mut batch,
                running_priority,
                lease_id,
            } => {
                // Split before attempting a batch already known not to fit. The
                // remainder waits on the parked stack as a fresh batch: it holds no
                // resident state, so it pins nothing while it waits.
                if batch.len() > fit_cap && batch[0].resume_position.is_none() {
                    parked.push((batch.split_off(fit_cap), running_priority));
                }
                // Let a caller blocked on the engine have it first (it polls every
                // 20 ms; taking it straight back would starve it again).
                while state.engine_wanted() {
                    tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                }
                let mut engine = match state.engine.lock().await.take() {
                    Some(e) => e,
                    None => {
                        for p in &batch {
                            let _ =
                                p.tx.send(BatchEvent::Error("daemon not running".to_string()));
                        }
                        if let Some(id) = lease_id {
                            state.work_scheduler.lock().await.complete(id);
                        }
                        continue;
                    }
                };
                // Park only while the stack has room (bounds resident VRAM from
                // nested preemption); at the cap the batch runs to completion.
                let can_park = parked.len() < max_depth;
                let outcome = run_batch_cycle(
                    &mut engine,
                    &state,
                    batch,
                    running_priority,
                    can_park,
                    &mut prefix_index,
                )
                .await;
                // A worker that died during the cycle (panic, OOM, killed) must not go
                // back in the slot: requests take `ensure_model_loaded`'s fast path,
                // which trusts `loaded_models` and never pings, so every later request
                // -- to either resident model -- failed against the dead engine for the
                // rest of the run. Drop it and forget the models; the next request
                // respawns the daemon.
                if engine.worker_alive() {
                    *state.engine.lock().await = Some(engine);
                } else {
                    tracing::error!(
                        "inference daemon died during a batch cycle; next request respawns it"
                    );
                    drop(engine);
                    crate::routes::chat::clear_loaded_model_state_for_failed_daemon(&state).await;
                }
                if let Some(id) = lease_id {
                    state.work_scheduler.lock().await.complete(id);
                }
                match outcome {
                    CycleOutcome::Completed => {}
                    CycleOutcome::Parked(remaining) => parked.push((remaining, running_priority)),
                    CycleOutcome::OutOfMemory(mut batch) => {
                        let half = batch.len() / 2;
                        fit_cap = half;
                        tracing::warn!(
                            "batch of {} ran out of device memory; retrying as {} + {}",
                            batch.len(),
                            half,
                            batch.len() - half
                        );
                        // LIFO: push the second half first so the first runs next.
                        parked.push((batch.split_off(half), running_priority));
                        parked.push((batch, running_priority));
                    }
                }
            }
            Dispatch::Embed { jobs, lease_id } => {
                let mut engine = match state.engine.lock().await.take() {
                    Some(e) => e,
                    None => {
                        for j in jobs {
                            let _ = j.tx.send(Err("daemon not running".to_string()));
                        }
                        state.work_scheduler.lock().await.complete(lease_id);
                        continue;
                    }
                };
                run_embed_jobs(&mut engine, jobs).await;
                *state.engine.lock().await = Some(engine);
                state.work_scheduler.lock().await.complete(lease_id);
            }
            Dispatch::Steer { jobs, lease_id } => {
                let mut engine = match state.engine.lock().await.take() {
                    Some(e) => e,
                    None => {
                        for j in jobs {
                            let _ = j.tx.send(Err("daemon not running".to_string()));
                        }
                        state.work_scheduler.lock().await.complete(lease_id);
                        continue;
                    }
                };
                run_steer_jobs(&mut engine, jobs).await;
                *state.engine.lock().await = Some(engine);
                state.work_scheduler.lock().await.complete(lease_id);
            }
            Dispatch::Train { job, lease_id } => {
                // One Train lease covers every training op; dispatch by the raw
                // wire `type` the route stamped (drafter vs LoRA adapter).
                //
                // Both train ops are MICRO-STEP PREEMPTIBLE: the daemon runs ONE
                // quantum (train_lora = steps, train_drafter = epochs) and returns;
                // this lease completes, and if the run is unfinished the job is
                // RE-ENQUEUED (carrying its run_id + tx) as a fresh low-priority
                // Training workload. Because training sits below interactive
                // text/steer, the scheduler serves any pending interactive request
                // between quanta, then resumes training — cooperative yield via
                // re-enqueue, no explicit park/resume. The daemon keeps the training
                // session resident (keyed by run_id) across quanta, so we pass the
                // same req each time and never reload the model/labels.
                let mut engine = match state.engine.lock().await.take() {
                    Some(e) => e,
                    None => {
                        let _ = job.tx.send(Err("daemon not running".to_string()));
                        state.work_scheduler.lock().await.complete(lease_id);
                        continue;
                    }
                };
                let step = match job.req.get("type").and_then(|v| v.as_str()) {
                    Some("train_drafter") => engine.train_drafter_step(job.req.clone()).await,
                    // train_lora (default): the other stepwise training op.
                    _ => engine.train_lora_step(job.req.clone()).await,
                };
                *state.engine.lock().await = Some(engine);
                state.work_scheduler.lock().await.complete(lease_id);
                match step {
                    Ok((true, payload)) => {
                        // Final quantum: the run is done — answer the route.
                        let _ = job.tx.send(Ok(payload));
                    }
                    Ok((false, _progress)) => {
                        // Unfinished: re-enqueue the SAME job (req carries run_id)
                        // under a fresh id, moving tx into it. tx is answered only
                        // on the terminal (done) quantum — never here.
                        let run_id = job
                            .req
                            .get("run_id")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        let new_id = uuid::Uuid::new_v4().to_string();
                        state.batch_inbox.lock().await.insert(
                            new_id.clone(),
                            ScheduledJob::Train(TrainJob {
                                req: job.req,
                                tx: job.tx,
                            }),
                        );
                        let workload = hipfire_scheduler::WorkloadSpec::microbatchable(
                            new_id.clone(),
                            hipfire_scheduler::WorkloadClass::Training,
                            160,
                            now_ms(),
                            hipfire_scheduler::WorkloadResources::default(),
                            format!("train:{run_id}"),
                            1,
                        );
                        if let Err(e) = state.work_scheduler.lock().await.enqueue(workload) {
                            // Admission failed: reclaim the job so its tx is
                            // answered instead of leaving the route hung.
                            if let Some(ScheduledJob::Train(j)) =
                                state.batch_inbox.lock().await.remove(&new_id)
                            {
                                let _ = j.tx.send(Err(format!("train re-enqueue admission: {e}")));
                            }
                        } else {
                            state.prefill_notify.notify_waiters();
                        }
                    }
                    Err(e) => {
                        let _ = job.tx.send(Err(e.to_string()));
                    }
                }
            }
            Dispatch::Image {
                job,
                running_priority,
                lease_id,
            } => {
                // Own the GPU turn: take the daemon engine out (if attached) so no
                // daemon op runs while the route drives diffusion on the physical
                // GPU. Diffusion is in-process and does not use the daemon, so a
                // missing engine is fine (a diffusion-only server has no resident
                // LLM). Grant the turn to the route and HOLD it (park this loop on
                // `release`) until the route reports done — that hold is what
                // serializes image with text/embed. A watcher sets the preempt
                // flag when a strictly-higher-priority workload appears; it polls
                // the scheduler, not the engine, so holding the engine cannot
                // deadlock the yield path.
                let engine = state.engine.lock().await.take();
                let preempt = Arc::new(AtomicBool::new(false));
                let (release_tx, release_rx) = oneshot::channel();
                let watcher = spawn_preempt_watcher(&state, preempt.clone(), running_priority);
                tracing::debug!("image turn granted (pri {running_priority}), holding GPU");
                if job
                    .grant
                    .send(ImageTurn {
                        preempt,
                        release: release_tx,
                    })
                    .is_ok()
                {
                    // Route disconnected? `release_rx` errors and we free the turn.
                    let _ = release_rx.await;
                }
                watcher.abort();
                *state.engine.lock().await = engine;
                state.work_scheduler.lock().await.complete(lease_id);
            }
        }
    }
}

/// Spawn the sampler-step preempt watcher: set `preempt` once a strictly-higher-
/// priority workload is queued. The min-quantum step floor is enforced by the
/// route's diffusion callback, so the watcher only signals intent.
fn spawn_preempt_watcher(
    state: &SharedState,
    preempt: Arc<AtomicBool>,
    running_priority: u8,
) -> tokio::task::JoinHandle<()> {
    let state = state.clone();
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(Duration::from_millis(5)).await;
            let waiter = state
                .work_scheduler
                .lock()
                .await
                .peek_next_priority(now_ms());
            if waiter.is_some_and(|top| top < running_priority) {
                preempt.store(true, Ordering::SeqCst);
                break;
            }
        }
    })
}

/// What one runner iteration executes. A text batch is park/resume-capable; an
/// embed batch runs to completion; an image is restart-from-seed preemptible.
enum Dispatch {
    Text {
        batch: Vec<PendingRequest>,
        running_priority: u8,
        lease_id: Option<u64>,
    },
    Embed {
        jobs: Vec<EmbedJob>,
        lease_id: u64,
    },
    Steer {
        jobs: Vec<SteerJob>,
        lease_id: u64,
    },
    Train {
        job: TrainJob,
        lease_id: u64,
    },
    Image {
        job: ImageJob,
        running_priority: u8,
        lease_id: u64,
    },
}

/// Run each steer control op on the runner-owned engine. A CaptureSession runs
/// its whole begin→capture*→finish atomically here (one runner turn), so no other
/// GPU work interleaves while the capture hook is folding residuals into the means.
async fn run_steer_jobs(engine: &mut DaemonEngine, jobs: Vec<SteerJob>) {
    for job in jobs {
        let result = match job.op {
            SteerOp::CaptureSession {
                num_layers,
                hidden,
                prompts,
            } => run_capture_session(engine, num_layers, hidden, prompts).await,
            SteerOp::BeginApply(req) => engine
                .steer_begin_apply(req)
                .await
                .map(|_| None)
                .map_err(|e| e.to_string()),
            SteerOp::Clear => engine
                .steer_clear()
                .await
                .map(|_| None)
                .map_err(|e| e.to_string()),
        };
        let _ = job.tx.send(result);
    }
}

/// begin → capture each prompt → finish, returning the per-block means. On any
/// error mid-session, clear the daemon session so a partial capture can't leak.
async fn run_capture_session(
    engine: &mut DaemonEngine,
    num_layers: usize,
    hidden: usize,
    prompts: Vec<(String, String)>,
) -> Result<Option<Vec<Vec<f32>>>, String> {
    engine
        .steer_begin_capture(num_layers, hidden)
        .await
        .map_err(|e| e.to_string())?;
    for (system, user) in prompts {
        if let Err(e) = engine.steer_capture(system, user).await {
            let _ = engine.steer_clear().await;
            return Err(e.to_string());
        }
    }
    engine
        .steer_finish_capture()
        .await
        .map(Some)
        .map_err(|e| e.to_string())
}

/// Run each embedding job on the runner-owned engine and return its result.
async fn run_embed_jobs(engine: &mut DaemonEngine, jobs: Vec<EmbedJob>) {
    for job in jobs {
        let result = engine.embed(job.req).await.map_err(|e| e.to_string());
        let _ = job.tx.send(result);
    }
}

fn fail_all(txs: &HashMap<String, mpsc::UnboundedSender<BatchEvent>>, msg: &str) {
    for tx in txs.values() {
        let _ = tx.send(BatchEvent::Error(msg.to_string()));
    }
}

/// Fold a batch prefill's per-session results into the cycle's cursors: each
/// session's logical position, and its remaining budget clamped to the rows its
/// KV cache has left (a decode step writes one KV row, and the request's
/// max_tokens is not that bound — the default far exceeds max_seq).
fn fold_prefill_events(
    events: &[serde_json::Value],
    positions: &mut HashMap<String, usize>,
    remaining: &mut HashMap<String, usize>,
) {
    for ev in events {
        if ev.get("type").and_then(|t| t.as_str()) == Some("generate_batch_prefill_session_done") {
            if let (Some(sid), Some(pos)) = (
                ev.get("session_id").and_then(|v| v.as_str()),
                ev.get("logical_position").and_then(|v| v.as_u64()),
            ) {
                positions.insert(sid.to_string(), pos as usize);
                // `/health` has no prefix-cache counters; this is where reuse shows.
                tracing::debug!(
                    "session {sid}: prefilled {} token(s), {} reused from a cached prefix",
                    ev.get("prefill_tokens")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0),
                    ev.get("cached_prefix_tokens")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0),
                );
                // Each decode step writes one KV row, so a session can generate
                // only as many tokens as its cache has rows left. The request's
                // max_tokens is not that bound (the default far exceeds max_seq).
                if let (Some(cap), Some(rem)) = (
                    ev.get("kv_capacity").and_then(|v| v.as_u64()),
                    remaining.get_mut(sid),
                ) {
                    *rem = (*rem).min(cap.saturating_sub(pos) as usize);
                }
            }
        }
    }
}

/// Whether a running text cycle admits queued requests between decode steps.
/// `HIPFIRE_SERVER_MIDCYCLE_ADMIT=0` restores cycle-granular batching.
fn midcycle_admit_enabled() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        !matches!(
            std::env::var("HIPFIRE_SERVER_MIDCYCLE_ADMIT").as_deref(),
            Ok("0") | Ok("false") | Ok("off")
        )
    })
}

/// Take up to `room` queued text requests the running cycle can absorb (same
/// worker, priority at least as urgent as `running_priority`) and prefill them,
/// reusing cached prefixes. Returns each with its logical position and remaining
/// budget. A failed prefill fails and releases only the newcomers.
async fn admit_into_cycle(
    engine: &mut DaemonEngine,
    state: &SharedState,
    worker: &str,
    batch_id: &str,
    running_priority: u8,
    room: usize,
    prefix_index: &mut PrefixIndex,
) -> Vec<(PendingRequest, usize, usize)> {
    if room == 0 {
        return Vec::new();
    }
    let taken = state
        .work_scheduler
        .lock()
        .await
        .take_microbatch_compatible(
            hipfire_scheduler::WorkloadClass::TokenPrefill,
            worker,
            running_priority,
            room,
        );
    if taken.is_empty() {
        return Vec::new();
    }
    let newcomers: Vec<PendingRequest> = {
        let mut inbox = state.batch_inbox.lock().await;
        taken
            .iter()
            .filter_map(|w| match inbox.remove(&w.id) {
                Some(ScheduledJob::Text(p)) if p.worker_key_id == worker => Some(p),
                // TokenPrefill workloads are text requests on this worker; anything
                // else goes back where it came from.
                Some(other) => {
                    inbox.insert(w.id.clone(), other);
                    None
                }
                None => None,
            })
            .collect()
    };
    if newcomers.is_empty() {
        return Vec::new();
    }
    tracing::debug!(
        "admitted {} request(s) into the running batch",
        newcomers.len()
    );
    let specs: Vec<SessionSpec> = newcomers.iter().map(|p| p.spec.clone()).collect();
    let mut positions = HashMap::new();
    let mut remaining: HashMap<String, usize> = specs
        .iter()
        .map(|s| (s.id.clone(), s.max_tokens.max(1)))
        .collect();
    match prefill_with_prefix_reuse(engine, batch_id, worker, &specs, prefix_index).await {
        Ok(events) => fold_prefill_events(&events, &mut positions, &mut remaining),
        Err(e) => {
            let handles: Vec<String> = specs.iter().map(|s| s.id.clone()).collect();
            let _ = engine
                .release_sessions(build_release_request(worker, &handles))
                .await;
            for p in &newcomers {
                let _ = p.tx.send(BatchEvent::Error(format!("batch prefill: {e}")));
            }
            return Vec::new();
        }
    }
    newcomers
        .into_iter()
        .filter_map(|p| {
            let Some(&pos) = positions.get(&p.spec.id) else {
                let _ = p.tx.send(BatchEvent::Error(
                    "batch prefill produced no session state".to_string(),
                ));
                return None;
            };
            let rem = remaining[&p.spec.id];
            Some((p, pos, rem))
        })
        .collect()
}

/// One fused prefill + decode cycle over `batch`. Runs to completion unless
/// `can_park` and a higher-priority (lower number than `running_priority`)
/// workload appears after the min-quantum floor, in which case the still-active
/// requests are returned as `Parked` with their resume cursors and the daemon
/// sessions are left resident.
async fn run_batch_cycle(
    engine: &mut DaemonEngine,
    state: &SharedState,
    batch: Vec<PendingRequest>,
    running_priority: u8,
    can_park: bool,
    prefix_index: &mut PrefixIndex,
) -> CycleOutcome {
    let worker = batch[0].worker_key_id.clone();
    let batch_id = format!("batch-{}", batch[0].spec.id);
    let resuming = batch[0].resume_position.is_some();
    let mut specs: Vec<SessionSpec> = batch.iter().map(|p| p.spec.clone()).collect();
    tracing::debug!(
        "{} {} request(s) into one batch",
        if resuming { "resumed" } else { "coalesced" },
        specs.len()
    );

    let mut txs: HashMap<String, mpsc::UnboundedSender<BatchEvent>> = HashMap::new();
    let mut remaining: HashMap<String, usize> = HashMap::new();
    let mut specs_by_id: HashMap<String, SessionSpec> = HashMap::new();
    let mut resume_pos: HashMap<String, usize> = HashMap::new();
    for p in &batch {
        txs.insert(p.spec.id.clone(), p.tx.clone());
        remaining.insert(p.spec.id.clone(), p.spec.max_tokens.max(1));
        specs_by_id.insert(p.spec.id.clone(), p.spec.clone());
        if let Some(pos) = p.resume_position {
            resume_pos.insert(p.spec.id.clone(), pos);
        }
    }

    // Fresh batches prefill every session's KV in one daemon call. Resumed
    // batches skip prefill: the sessions are already resident at their cursor.
    let mut positions: HashMap<String, usize> = HashMap::new();
    if resuming {
        positions = resume_pos.clone();
    } else {
        let result =
            prefill_with_prefix_reuse(engine, &batch_id, &worker, &specs, prefix_index).await;
        let events = match result {
            Ok(events) => events,
            Err(e) => {
                // The daemon's activation loop can run to completion for every
                // session and THEN fail (suffix prefill, or the checkpoint step),
                // so sessions may already be resident. This exit bypasses the
                // cycle-end release below, so release them here or they stay
                // pinned until the model is unloaded. Unknown ids are a no-op.
                let handles: Vec<String> = specs.iter().map(|s| s.id.clone()).collect();
                let _ = engine
                    .release_sessions(build_release_request(&worker, &handles))
                    .await;
                // A lone request that does not fit cannot be split any further.
                if !resuming && batch.len() > 1 && is_device_oom(&e.to_string()) {
                    return CycleOutcome::OutOfMemory(batch);
                }
                fail_all(&txs, &format!("batch prefill: {e}"));
                return CycleOutcome::Completed;
            }
        };
        fold_prefill_events(&events, &mut positions, &mut remaining);
    }
    // Usage for each request's Done: its prompt length (the position prefill left
    // it at) and the tokens it committed. A resumed (parked) session's prompt
    // length is not known here; its usage reports this cycle's tokens only.
    // ponytail: per-cycle counts; carry them on PendingRequest if parking is common.
    let mut prompt_len: HashMap<String, usize> = if resuming {
        HashMap::new()
    } else {
        positions.clone()
    };
    let mut generated: HashMap<String, usize> = HashMap::new();

    let mut active: Vec<String> = specs
        .iter()
        .map(|s| s.id.clone())
        .filter(|id| positions.contains_key(id))
        .collect();
    // A prompt that filled its KV has nothing left to decode into: finish it now,
    // since one decode step for it would fail the whole batch at the daemon guard.
    active.retain(|id| {
        if remaining[id] > 0 {
            return true;
        }
        if let Some(tx) = txs.get(id) {
            let _ = tx.send(BatchEvent::Done(done_payload(
                "length",
                prompt_len.get(id),
                0,
            )));
        }
        false
    });
    // Any session with no prefill checkpoint can't decode — fail it, don't hang.
    for s in &specs {
        if !positions.contains_key(&s.id) {
            if let Some(tx) = txs.get(&s.id) {
                let _ = tx.send(BatchEvent::Error(
                    "batch prefill produced no session state".to_string(),
                ));
            }
        }
    }

    let quantum = min_quantum();
    let mut last_backend: Option<String> = None;
    let mut last_chunk_count = 0u64;
    let mut last_chunk_size = 0u64;
    let mut decode_ms = 0.0f64;
    let mut compatible_state_kinds = Vec::new();
    let mut cached_prefix_tokens = None;
    let mut fallback_reason = None;
    let mut steps: u32 = 0;
    // Batched speculation (daemon `qwen35_batch_spec`): drafts proposed and
    // accepted, and tokens committed, over the cycle.
    let (mut spec_drafted, mut spec_accepted, mut committed_total) = (0u64, 0u64, 0u64);
    while !active.is_empty() {
        // Drop sessions whose client disconnected (response receiver closed):
        // stop decoding, and don't park/resume them. A session that was parked
        // and then abandoned is retired here on its resume cycle. Their KV is
        // freed with the rest of the batch at cycle end — including on the park
        // exit, which used to return without releasing them at all, pinning every
        // session that finished or disconnected before the preemption.
        // ponytail: eager per-session release could reclaim KV sooner; the
        // batch-end release is enough until nested-preemption VRAM bites.
        active.retain(|id| txs.get(id).is_some_and(|tx| !tx.is_closed()));
        if active.is_empty() {
            break;
        }
        let cursors: Vec<DecodeCursor> = active
            .iter()
            .map(|id| DecodeCursor {
                id: id.clone(),
                logical_position: positions[id],
                max_tokens_remaining: remaining[id],
            })
            .collect();
        let decode_req = build_batch_decode_request(&batch_id, &worker, &cursors);
        let events = match engine.generate_batch_decode_step(decode_req).await {
            Ok(events) => events,
            Err(e) => {
                fail_all(&txs, &format!("batch decode: {e}"));
                break;
            }
        };
        steps += 1;

        let mut still_active = Vec::new();
        for id in &active {
            let done_ev = events.iter().find(|e| {
                e.get("type").and_then(|t| t.as_str())
                    == Some("generate_batch_decode_step_session_done")
                    && e.get("session_id").and_then(|v| v.as_str()) == Some(id.as_str())
            });
            let Some(ev) = done_ev else {
                // No per-session event this step: end the request cleanly.
                if let Some(tx) = txs.get(id) {
                    let _ = tx.send(BatchEvent::Done(done_payload(
                        "stop",
                        prompt_len.get(id),
                        generated.get(id).copied().unwrap_or(0),
                    )));
                }
                continue;
            };
            let text = ev.get("text").and_then(|v| v.as_str()).unwrap_or("");
            let stop = ev.get("stop").and_then(|v| v.as_bool()).unwrap_or(false);
            if let Some(pos) = ev.get("logical_position").and_then(|v| v.as_u64()) {
                positions.insert(id.clone(), pos as usize);
            }
            if !text.is_empty() {
                if let Some(tx) = txs.get(id) {
                    let _ = tx.send(BatchEvent::Token(text.to_string()));
                }
            }
            // A speculative step commits `tokens` (0..=n: a step that only
            // feeds a pending run emits none); the plain step, one `token`.
            let committed = ev
                .get("tokens")
                .and_then(|v| v.as_array())
                .map_or(1, Vec::len);
            committed_total += committed as u64;
            *generated.entry(id.clone()).or_default() += committed;
            if let Some(spec) = ev.get("spec") {
                spec_drafted += spec.get("drafted").and_then(|v| v.as_u64()).unwrap_or(0);
                spec_accepted += spec.get("accepted").and_then(|v| v.as_u64()).unwrap_or(0);
            }
            let rem = remaining.get_mut(id).map(|r| {
                *r = r.saturating_sub(committed);
                *r
            });
            if stop || rem == Some(0) {
                // The daemon says why it stopped; `stop` alone is also raised when
                // the budget runs out, which made every truncated reply look done.
                let finish_reason = ev
                    .get("finish_reason")
                    .and_then(|v| v.as_str())
                    .unwrap_or(if stop { "stop" } else { "length" });
                if let Some(tx) = txs.get(id) {
                    let _ = tx.send(BatchEvent::Done(done_payload(
                        finish_reason,
                        prompt_len.get(id),
                        generated.get(id).copied().unwrap_or(0),
                    )));
                }
            } else {
                still_active.push(id.clone());
            }
        }
        if let Some(done) = events.iter().find(|e| {
            e.get("type").and_then(|t| t.as_str()) == Some("generate_batch_decode_step_done")
        }) {
            last_backend = done
                .get("backend")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());
            last_chunk_count = done
                .get("chunk_count")
                .and_then(|v| v.as_u64())
                .unwrap_or_default();
            last_chunk_size = done
                .get("chunk_size")
                .and_then(|v| v.as_u64())
                .unwrap_or_default();
            decode_ms += done
                .get("elapsed_ms")
                .and_then(|v| v.as_f64())
                .unwrap_or_default();
            compatible_state_kinds = done
                .get("compatible_state_kinds")
                .and_then(|v| v.as_array())
                .map(|values| {
                    values
                        .iter()
                        .filter_map(|value| value.as_str().map(str::to_string))
                        .collect()
                })
                .unwrap_or_default();
            cached_prefix_tokens = done.get("cached_prefix_tokens").and_then(|v| v.as_u64());
            fallback_reason = done
                .get("fallback_reason")
                .and_then(|v| v.as_str())
                .map(str::to_string);
        }
        // Release what finished this step now, not at cycle end. With mid-cycle
        // admission a cycle runs as long as requests keep arriving -- for a whole
        // swarm turn -- and every finished request's KV stayed resident until it
        // ended: 18+ dead sessions and ~30 GiB of GTT over one CAE turn.
        let finished: Vec<String> = active
            .iter()
            .filter(|id| !still_active.contains(id))
            .cloned()
            .collect();
        if !finished.is_empty() {
            let _ = engine
                .release_sessions(build_release_request(&worker, &finished))
                .await;
        }
        active = still_active;

        // Mid-cycle admission: take queued requests this batch can run (same
        // worker, at least as urgent) and prefill them in, so they decode from
        // the next step instead of waiting for the whole batch to finish. Runs
        // before the preemption check: a more urgent request on this worker joins
        // rather than parking the batch.
        if midcycle_admit_enabled() && !active.is_empty() && !state.engine_wanted() {
            let admitted = admit_into_cycle(
                engine,
                state,
                &worker,
                &format!("{batch_id}-a{steps}"),
                running_priority,
                batch_max().saturating_sub(active.len()),
                prefix_index,
            )
            .await;
            for (p, pos, rem) in admitted {
                let id = p.spec.id.clone();
                txs.insert(id.clone(), p.tx.clone());
                positions.insert(id.clone(), pos);
                prompt_len.insert(id.clone(), pos);
                remaining.insert(id.clone(), rem);
                specs_by_id.insert(id.clone(), p.spec.clone());
                specs.push(p.spec);
                if rem > 0 {
                    active.push(id);
                } else {
                    let _ =
                        p.tx.send(BatchEvent::Done(done_payload("length", Some(&pos), 0)));
                }
            }
        }

        // Cooperative preemption: past the min-quantum floor, yield to a
        // strictly-higher-priority waiter. Park the still-active sessions
        // (left resident in the daemon) so no decoded work is discarded; the
        // runner resumes them once the higher-priority work drains.
        if can_park && !active.is_empty() && steps >= quantum {
            let waiter = state
                .work_scheduler
                .lock()
                .await
                .peek_next_priority(now_ms());
            // A caller blocked on the engine (another model's load) is waited for
            // like a more urgent request: nothing else ends a cycle that keeps
            // admitting work.
            let engine_wanted = state.engine_wanted();
            if engine_wanted || waiter.is_some_and(|top| top < running_priority) {
                tracing::debug!(
                    "parked {} session(s) at step {steps}: pri {running_priority} yields to {}",
                    active.len(),
                    if engine_wanted {
                        "an engine waiter".to_string()
                    } else {
                        format!("pri {}", waiter.unwrap_or_default())
                    }
                );
                let parked: Vec<PendingRequest> = active
                    .iter()
                    .filter_map(|id| {
                        let mut spec = specs_by_id.get(id)?.clone();
                        spec.max_tokens = remaining[id];
                        Some(PendingRequest {
                            spec,
                            worker_key_id: worker.clone(),
                            tx: txs.get(id)?.clone(),
                            resume_position: Some(positions[id]),
                        })
                    })
                    .collect();
                // Parked sessions stay resident on purpose. Everything else this
                // cycle allocated must be released HERE: sessions that finished
                // earlier, or whose client disconnected (dropped from `active` at
                // the retain above), are in no later batch, so the cycle-end
                // release below never sees them and their KV would be pinned for
                // the life of the loaded worker.
                let kept: std::collections::HashSet<&str> =
                    parked.iter().map(|p| p.spec.id.as_str()).collect();
                let to_release: Vec<String> = specs
                    .iter()
                    .map(|s| s.id.clone())
                    .filter(|id| !kept.contains(id.as_str()))
                    .collect();
                if !to_release.is_empty() {
                    let _ = engine
                        .release_sessions(build_release_request(&worker, &to_release))
                        .await;
                }

                let mut tel = state.batch_telemetry.lock().await;
                tel.preemptions += 1;
                tel.last_chunk_count = last_chunk_count;
                tel.last_chunk_size = last_chunk_size;
                tel.last_decode_ms = Some(decode_ms);
                tel.compatible_state_kinds = compatible_state_kinds;
                tel.cached_prefix_tokens = cached_prefix_tokens;
                tel.fallback_reason = fallback_reason;
                tel.last_backend = last_backend;
                return CycleOutcome::Parked(parked);
            }
        }
    }

    tracing::debug!(
        steps,
        committed_total,
        spec_drafted,
        spec_accepted,
        decode_ms,
        "decode cycle done ({:.2} tokens/session-step)",
        committed_total as f64 / f64::from(steps.max(1)) / specs.len().max(1) as f64
    );
    // All requests finished: release the sessions and record the batch.
    let handles: Vec<String> = specs.iter().map(|s| s.id.clone()).collect();
    let _ = engine
        .release_sessions(build_release_request(&worker, &handles))
        .await;

    let mut tel = state.batch_telemetry.lock().await;
    tel.total_batches += 1;
    if last_backend.as_deref() == Some("serial_reference") {
        tel.serial_batches += 1;
    }
    tel.selected_batch_size = specs.len() as u64;
    tel.last_chunk_count = last_chunk_count;
    tel.last_chunk_size = last_chunk_size;
    tel.last_decode_ms = Some(decode_ms);
    tel.compatible_state_kinds = compatible_state_kinds;
    tel.cached_prefix_tokens = cached_prefix_tokens;
    tel.fallback_reason = fallback_reason;
    tel.last_backend = last_backend;
    CycleOutcome::Completed
}

#[cfg(test)]
mod tests {
    use super::*;

    // The form the daemon ACTUALLY sends. `lifecycle.rs` reports `loaded.family` for any
    // model with a registered backend — "qwen3.5", with a dot — and only falls back to a
    // model_type ("qwen3_5") when there is none. Every test here used the underscore
    // form, so they all passed while no real qwen3.5 request was ever batch-eligible:
    // the family fell to the legacy path and concurrent requests ran one at a time.
    #[test]
    fn the_family_tag_the_daemon_reports_is_batch_eligible() {
        for tag in ["qwen3.5", "qwen3_5", "qwen3_5_text", "Qwen3.5"] {
            assert!(
                batch_eligible(Some(tag), None),
                "{tag}: the daemon reports this form; it must resolve to the arch"
            );
        }
        // Still honestly false for an arch nothing linked in declares.
        assert!(!batch_eligible(Some("not-an-arch"), None));
        assert!(!batch_eligible(None, None));
    }

    #[test]
    fn a_model_the_prefill_refuses_is_not_batch_eligible() {
        // The daemon probes the LOADED model and reports the answer. Before
        // this, eligibility came from `batch_envelope_ok`, which reads
        // HIPFIRE_DFLASH_DRAFT — unset for a drafter found by sibling
        // discovery, so such a model was dispatched and then refused
        // mid-cycle, and `fail_all` took every session down with it.
        assert!(
            !batch_eligible(Some("qwen3_5"), Some(false)),
            "a model the batched prefill refuses must route to the legacy path"
        );
        // `None` is a daemon that does not report it: keep the old behaviour
        // rather than silently disabling batching for everyone.
        assert_eq!(
            batch_eligible(Some("qwen3_5"), None),
            batch_eligible(Some("qwen3_5"), Some(true)),
            "an unreported capability must not change routing on its own"
        );
    }

    #[test]
    fn decode_health_exposes_daemon_scheduler_metadata() {
        let health = BatchTelemetry::default().decode_health_json();
        assert_eq!(health["compatible_state_kinds"], serde_json::json!([]));
        assert!(health.get("cached_prefix_tokens").is_some());
        assert!(health.get("fallback_reason").is_some());
    }

    fn spec(id: &str) -> SessionSpec {
        SessionSpec {
            id: id.to_string(),
            prompt: format!("hello from {id}"),
            messages_history: None,
            system_prompt: None,
            state_kinds: vec!["attention_kv".to_string(), "deltanet_recurrent".to_string()],
            assistant_prefix: "plain".to_string(),
            max_think_tokens: 0,
            max_tokens: 16,
            tools: None,
        }
    }

    // The built requests must be accepted by the daemon's own validators —
    // that is the real protocol contract, stronger than shape assertions.
    #[test]
    fn prefill_request_passes_daemon_validator() {
        let mut a = spec("req-a");
        a.assistant_prefix = "closed_think".to_string();
        a.max_think_tokens = 1; // thinking disabled (daemon: enable_thinking = != 1)
        let specs = [a, spec("req-b")];
        let req = build_batch_prefill_request("batch-1", "worker-xyz", &specs, &HashMap::new());
        let env = hipfire_generate::validate_generate_batch_prefill(&req)
            .expect("built prefill request must validate");
        assert_eq!(env.batch_id, "batch-1");
        assert_eq!(env.sessions.len(), 2);
        assert_eq!(env.sessions[0].id, "req-a");
        // The daemon reads these from a nested `params` object — assert they
        // survive the round-trip so a top-level regression can't recur.
        assert_eq!(
            env.sessions[0].assistant_prefix, "closed_think",
            "assistant_prefix must reach the daemon via params"
        );
        assert_eq!(
            env.sessions[0].max_think_tokens, 1,
            "max_think_tokens must reach the daemon via params"
        );
    }

    #[test]
    fn the_shared_prefix_outlives_the_request_specific_tails() {
        let entry = |len: usize, hash: &str| PrefixEntry {
            worker: "w".into(),
            prefix_hash: serde_json::json!({"algorithm": "xxh128", "value": hash, "prefix_len": len}),
            prefix_len: len,
            checkpoint_id: format!("ck-{hash}"),
            hits: 0,
            batch: 1,
            mint_at: None,
        };
        let mut index = PrefixIndex::default();
        // Each cold prompt mints its own tail and the shared system turn, recorded
        // longest first; from the second prompt on the system turn is a duplicate,
        // released, and refreshes the held one. Fill the index past its cap so.
        let mut released = Vec::new();
        released.extend(index.insert(entry(603, "system")));
        for i in 0..prefix_cache_max() {
            released.extend(index.insert(entry(622, &format!("tail-{i}"))));
            let mut dup = entry(603, "system");
            dup.checkpoint_id = format!("ck-system-dup{i}");
            assert_eq!(index.insert(dup), vec![format!("ck-system-dup{i}")]);
        }
        assert!(
            !released.contains(&"ck-system".to_string()),
            "shared prefix evicted: {released:?}"
        );
        // A hit on it is found, and it is the longest cached match.
        let candidates = [
            serde_json::json!({"value": "system"}),
            serde_json::json!({"value": "nope"}),
        ];
        assert_eq!(
            index.lookup("w", &candidates).unwrap().checkpoint_id,
            "ck-system"
        );
        // A second checkpoint of an already-held hash (minted by another batch) is
        // handed back for release; the held one stays.
        let mut dup = entry(603, "system");
        dup.checkpoint_id = "ck-system-2".into();
        assert_eq!(index.insert(dup), vec!["ck-system-2".to_string()]);
        assert_eq!(
            index.lookup("w", &candidates).unwrap().checkpoint_id,
            "ck-system"
        );

        // A later batch's fresh tail outlives the earlier batch's unused tails —
        // what a tool loop's next step attaches to.
        let mut fresh = entry(900, "next-step");
        fresh.batch = 2;
        let released = index.insert(fresh);
        assert!(
            !released.contains(&"ck-next-step".to_string()),
            "fresh tail evicted: {released:?}"
        );
        let next = [serde_json::json!({"value": "next-step"})];
        assert_eq!(
            index.lookup("w", &next).unwrap().checkpoint_id,
            "ck-next-step"
        );
    }

    #[test]
    fn a_new_checkpoint_survives_an_index_full_of_attached_ones() {
        // A tool loop: every step attaches the previous step's checkpoint (hits > 0)
        // and mints its own. Once the index is full, the new mint must displace the
        // least recently used attached entry, not itself.
        let mut index = PrefixIndex::default();
        for i in 0..=prefix_cache_max() {
            let hash = format!("step-{i}");
            let released = index.insert(PrefixEntry {
                worker: "w".into(),
                prefix_hash: serde_json::json!({"value": hash, "prefix_len": 1000 + i}),
                prefix_len: 1000 + i,
                checkpoint_id: format!("ck-{hash}"),
                hits: 0,
                batch: i as u64 + 1,
                mint_at: None,
            });
            assert!(
                !released.contains(&format!("ck-{hash}")),
                "step {i}'s fresh checkpoint evicted itself: {released:?}"
            );
            let next = [serde_json::json!({"value": hash})];
            assert_eq!(
                index.lookup("w", &next).unwrap().checkpoint_id,
                format!("ck-{hash}")
            );
        }
        // The oldest step went, as least recently used.
        assert!(index
            .lookup("w", &[serde_json::json!({"value": "step-0"})])
            .is_none());
    }

    #[test]
    fn concurrent_tool_loops_each_keep_their_newest_checkpoint() {
        // Three conversations step in turn: each attaches its previous checkpoint
        // and mints the next. Never-attached-first eviction let one chain's mint
        // evict another's not-yet-attached newest checkpoint.
        let mut index = PrefixIndex::default();
        let mk = |hash: String, len: usize| PrefixEntry {
            worker: "w".into(),
            prefix_hash: serde_json::json!({"value": hash, "prefix_len": len}),
            prefix_len: len,
            checkpoint_id: format!("ck-{hash}"),
            hits: 0,
            batch: 0,
            mint_at: None,
        };
        // Stale attached leftovers fill the index.
        for i in 0..prefix_cache_max() {
            index.insert(mk(format!("old-{i}"), 100 + i));
            index.lookup("w", &[serde_json::json!({"value": format!("old-{i}")})]);
        }
        for step in 0..10 {
            for chain in 0..3 {
                if step > 0 {
                    let prev = format!("c{chain}-s{}", step - 1);
                    let hit = index.lookup("w", &[serde_json::json!({"value": prev})]);
                    assert!(
                        hit.is_some(),
                        "chain {chain} lost its step-{} checkpoint",
                        step - 1
                    );
                }
                index.insert(mk(format!("c{chain}-s{step}"), 1000 * (step + 1) + chain));
            }
        }
    }

    #[test]
    fn eviction_stays_within_the_inserting_worker() {
        // Two models share the index; one's mints must never evict (and hand back
        // for release to the wrong daemon session registry) the other's.
        let mut index = PrefixIndex::default();
        let mk = |worker: &str, hash: String| PrefixEntry {
            worker: worker.into(),
            prefix_hash: serde_json::json!({"value": hash}),
            prefix_len: 1,
            checkpoint_id: format!("ck-{worker}-{hash}"),
            hits: 0,
            batch: 0,
            mint_at: None,
        };
        for i in 0..prefix_cache_max() {
            assert!(index.insert(mk("b", format!("{i}"))).is_empty());
        }
        for i in 0..3 * prefix_cache_max() {
            for id in index.insert(mk("a", format!("{i}"))) {
                assert!(id.starts_with("ck-a-"), "worker a's insert evicted {id}");
            }
        }
        let b_left = index.entries.iter().filter(|e| e.worker == "b").count();
        assert_eq!(
            b_left,
            prefix_cache_max(),
            "worker b lost entries to a's inserts"
        );
    }

    #[test]
    fn prefill_groups_respect_the_row_budget() {
        let specs: Vec<SessionSpec> = (0..5).map(|i| spec(&format!("s{i}"))).collect();
        let rows: HashMap<String, usize> = [
            ("s0", 3000),
            ("s1", 3000),
            ("s2", 3000),
            ("s3", 20000),
            ("s4", 100),
        ]
        .iter()
        .map(|(k, v)| (k.to_string(), *v))
        .collect();
        let ids = |g: &Vec<Vec<SessionSpec>>| -> Vec<Vec<String>> {
            g.iter()
                .map(|v| v.iter().map(|s| s.id.clone()).collect())
                .collect()
        };
        // 3000+3000 fits 8192, a third does not; the 20000-row session goes alone.
        assert_eq!(
            ids(&prefill_groups(&specs, &rows, 8192)),
            vec![vec!["s0", "s1"], vec!["s2"], vec!["s3"], vec!["s4"]]
        );
        // Unknown rows count as a full budget, so that session goes alone.
        let mut partial = rows.clone();
        partial.remove("s1");
        assert_eq!(
            ids(&prefill_groups(&specs[..3], &partial, 8192)),
            vec![vec!["s0"], vec!["s1"], vec!["s2"]]
        );
        // 0 disables the budget.
        assert_eq!(prefill_groups(&specs, &rows, 0).len(), 1);
    }

    #[test]
    fn device_oom_is_recognised_in_the_daemon_error() {
        // Verbatim from a 16-session prefill on gfx1151. The split-and-retry keys
        // on this; a rewording of the daemon error must not silently disable it.
        let err = "daemon generate_batch_prefill error: qwen35 fused dense prefill-session \
                   batch backend failed: HipError { code: 2, message: \"hipMalloc(703840256 \
                   bytes = 671.23 MiB), free=299.6 MiB of total=110592.0 MiB (hipError=2)\" }";
        assert!(is_device_oom(err));
        assert!(!is_device_oom("hipModuleLaunchKernel failed (hipError=98)"));
    }

    #[test]
    fn prefill_request_carries_tools_to_the_daemon() {
        // Dropped once already: the batch path sent no tools, so a model whose
        // request declared them answered in prose or invented a call.
        let tool = serde_json::json!({"type": "function", "function": {
            "name": "read_file",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
        }});
        let mut a = spec("req-a");
        a.tools = Some(serde_json::json!([tool.clone()]));
        let specs = [a, spec("req-b")];
        let req = build_batch_prefill_request("batch-1", "worker-xyz", &specs, &HashMap::new());
        let env = hipfire_generate::validate_generate_batch_prefill(&req)
            .expect("a prefill request with tools must validate");
        assert_eq!(env.sessions[0].tools, Some(vec![tool]));
        assert_eq!(env.sessions[1].tools, None);
    }

    #[test]
    fn decode_request_passes_daemon_validator() {
        let cursors = [
            DecodeCursor {
                id: "req-a".to_string(),
                logical_position: 7,
                max_tokens_remaining: 15,
            },
            DecodeCursor {
                id: "req-b".to_string(),
                logical_position: 9,
                max_tokens_remaining: 15,
            },
        ];
        let req = build_batch_decode_request("batch-1", "worker-xyz", &cursors);
        let env = hipfire_generate::validate_generate_batch_decode(&req)
            .expect("built decode request must validate");
        assert_eq!(env.batch_id, "batch-1");
        assert_eq!(env.sessions.len(), 2);
        assert_eq!(env.sessions[1].session_id, "req-b");
        assert_eq!(env.sessions[0].logical_position, 7);
        assert_eq!(env.cached_prefix_tokens, 7);
    }

    // Dummy impl exercises the generic seam without a GPU or a real arch:
    // same key coalesces, differing worker/cache/state does not.
    struct DummySession {
        worker: &'static str,
        cache: &'static str,
        state_kinds: &'static str,
    }
    impl BatchableSession for DummySession {
        fn batch_key(&self) -> String {
            format!("{}|{}|{}", self.worker, self.cache, self.state_kinds)
        }
    }

    #[test]
    fn batchable_session_groups_by_key() {
        let a = DummySession {
            worker: "w1",
            cache: "fp32",
            state_kinds: "kv,dn",
        };
        let b = DummySession {
            worker: "w1",
            cache: "fp32",
            state_kinds: "kv,dn",
        };
        let diff_worker = DummySession {
            worker: "w2",
            cache: "fp32",
            state_kinds: "kv,dn",
        };
        let diff_cache = DummySession {
            worker: "w1",
            cache: "q8",
            state_kinds: "kv,dn",
        };
        assert_eq!(
            a.batch_key(),
            b.batch_key(),
            "identical sessions must coalesce"
        );
        assert_ne!(
            a.batch_key(),
            diff_worker.batch_key(),
            "different worker must not"
        );
        assert_ne!(
            a.batch_key(),
            diff_cache.batch_key(),
            "different cache mode must not"
        );
    }

    #[test]
    fn batch_eligibility_reads_continuous_batching_capability() {
        // qwen3.5 dense + MoE declare ContinuousBatching in their -spec crates,
        // which hipfire-arch-specs force-links into this binary.
        assert!(
            arch_supports_continuous_batching(Some("qwen3_5")),
            "qwen3.5 dense declares ContinuousBatching"
        );
        assert!(
            arch_supports_continuous_batching(Some("qwen3_5_moe")),
            "qwen3.5 MoE declares ContinuousBatching"
        );
        // An arch tag with no ContinuousBatching capability (or unknown) is not
        // eligible — it routes to the legacy path.
        assert!(!arch_supports_continuous_batching(Some(
            "no_such_model_type"
        )));
        assert!(!arch_supports_continuous_batching(None));
    }

    #[test]
    fn release_request_shape() {
        let req = build_release_request("worker-xyz", &["h1".to_string(), "h2".to_string()]);
        assert_eq!(req["type"], "release_sessions");
        assert_eq!(req["worker_key_id"], "worker-xyz");
        assert_eq!(req["sessions"], serde_json::json!(["h1", "h2"]));
    }
}

#[cfg(test)]
mod arch_registry_link_tests {
    use super::*;

    /// The registry is only populated if the arch `-spec` crates survive rlib
    /// pruning. A Cargo.toml dependency alone is not enough — something must
    /// reference them. If this fails, `batch_eligible` silently reports
    /// `false` for every model and continuous batching never engages.
    #[test]
    fn arch_specs_are_linked_into_the_server_binary() {
        assert!(
            arch_registry().len() > 0,
            "arch registry is empty: the -spec crates were pruned from the link",
        );
    }
}
