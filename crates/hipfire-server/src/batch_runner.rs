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

/// Whether a request whose model reports arch tag `arch` (the daemon's
/// `model_type`, e.g. `qwen3_5`) may be routed through the continuous-batching
/// runner. Eligibility is a **declared arch capability** (`ContinuousBatching`)
/// resolved via the arch-specs registry, AND the runtime envelope the daemon's
/// fused batch path requires. Anything ineligible falls back to the legacy
/// per-request path — the safe default that always works.
pub fn batch_eligible(arch: Option<&str>) -> bool {
    arch_supports_continuous_batching(arch) && batch_envelope_ok()
}

fn arch_supports_continuous_batching(arch: Option<&str>) -> bool {
    let Some(arch) = arch else {
        return false;
    };
    arch_registry()
        .find_by_model_type(arch)
        .and_then(|a| a.caps.continuous_batching)
        .is_some()
}

/// Runtime toggles the fused batch path can't (or shouldn't yet) run under.
/// Conservative: DFlash, pipeline-parallel > 1, and hierarchical KV route to the
/// proven legacy path. (Hierarchical KV would fall to the serial-swap decode
/// backend inside the batch op; excluded here until validated.)
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
    /// Times a running batch admitted new requests at a decode-step boundary.
    pub joins: u64,
    /// Sessions admitted mid-cycle by those joins.
    pub joined_sessions: u64,
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

/// Build a `generate_batch_prefill` request for `specs` on `worker_key_id`.
/// Fused-prefill of all sessions in one daemon call; each emits a
/// `generate_batch_prefill_session_done` carrying its `logical_position`.
pub fn build_batch_prefill_request(
    batch_id: &str,
    worker_key_id: &str,
    specs: &[SessionSpec],
) -> serde_json::Value {
    let sessions: Vec<serde_json::Value> = specs
        .iter()
        .map(|s| {
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
                    "semantic_boundary_checkpoints": false,
                },
                "state_handle": {
                    "state_kinds": s.state_kinds,
                    "logical_position": 0,
                    "cached_prefix_tokens": 0,
                },
            });
            if let Some(obj) = session.as_object_mut() {
                if let Some(history) = &s.messages_history {
                    obj.insert("messages".to_string(), history.clone());
                }
                if let Some(system) = &s.system_prompt {
                    obj.insert("system".to_string(), serde_json::json!(system));
                }
            }
            session
        })
        .collect();
    serde_json::json!({
        "type": "generate_batch_prefill",
        "id": batch_id,
        "batch_id": batch_id,
        "worker_key_id": worker_key_id,
        "sessions": sessions,
    })
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
            "joins": self.joins,
            "joined_sessions": self.joined_sessions,
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
const BATCH_MAX_DEFAULT: usize = 8;

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
    tokio::spawn(async move {
        batch_runner_loop(state).await;
    });
}

pub fn batch_max() -> usize {
    std::env::var("HIPFIRE_SERVER_PREFILL_BATCH_MAX")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|n| *n >= 1)
        .unwrap_or(BATCH_MAX_DEFAULT)
}

/// Whether a request that arrives mid-cycle may join the running batch at the
/// next decode-step boundary instead of waiting for it to drain. Kill switch
/// `HIPFIRE_SERVER_BATCH_JOIN=0` restores the drain-first behaviour.
fn join_enabled() -> bool {
    std::env::var("HIPFIRE_SERVER_BATCH_JOIN").ok().as_deref() != Some("0")
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
            if std::env::var("HIPFIRE_BATCH_RUNNER_DEBUG").as_deref() == Ok("1") {
                eprintln!(
                    "[batch-cycle] lease {} class={:?} workloads={}",
                    lease.lease_id,
                    lease.class,
                    lease.workloads.len()
                );
            }
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
                batch,
                running_priority,
                lease_id,
            } => {
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
                let outcome =
                    run_batch_cycle(&mut engine, &state, batch, running_priority, can_park).await;
                *state.engine.lock().await = Some(engine);
                if let Some(id) = lease_id {
                    state.work_scheduler.lock().await.complete(id);
                }
                if let CycleOutcome::Parked(remaining) = outcome {
                    parked.push((remaining, running_priority));
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
                if std::env::var("HIPFIRE_BATCH_RUNNER_DEBUG").as_deref() == Ok("1") {
                    eprintln!(
                        "[batch-cycle] image turn granted (pri {running_priority}), holding GPU"
                    );
                }
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

/// Record each session's post-prefill cursor from a `generate_batch_prefill`
/// event stream. Shared by the initial prefill and by mid-cycle joins.
fn collect_prefill_positions(events: &[serde_json::Value], positions: &mut HashMap<String, usize>) {
    for ev in events {
        if ev.get("type").and_then(|t| t.as_str()) != Some("generate_batch_prefill_session_done") {
            continue;
        }
        if let (Some(sid), Some(pos)) = (
            ev.get("session_id").and_then(|v| v.as_str()),
            ev.get("logical_position").and_then(|v| v.as_u64()),
        ) {
            positions.insert(sid.to_string(), pos as usize);
        }
    }
}

/// Sessions admitted into a running batch at a decode-step boundary.
struct Joined {
    specs: Vec<SessionSpec>,
    lease_ids: Vec<u64>,
}

/// Try to admit queued requests into the running batch.
///
/// The daemon keys resident sessions by `session_id`, not by batch — a prefill
/// adds to `q35_registry.sessions` without disturbing anyone else, and a decode
/// step takes an arbitrary cursor list. So a newcomer only has to be prefilled
/// and then named in the next step's cursors; nothing about the running
/// sessions changes. That is the whole join.
///
/// Only same-`microbatch_key` (same worker/model) waiters are eligible, checked
/// by peeking BEFORE leasing so an incompatible waiter is never pulled out of
/// the queue. Anything leased beyond the free-slot count is handed straight
/// back — inbox entry and workload both — so the scheduler re-offers it.
#[allow(clippy::too_many_arguments)]
async fn try_join(
    engine: &mut DaemonEngine,
    state: &SharedState,
    batch_id: &str,
    worker: &str,
    active: &mut Vec<String>,
    positions: &mut HashMap<String, usize>,
    remaining: &mut HashMap<String, usize>,
    txs: &mut HashMap<String, mpsc::UnboundedSender<BatchEvent>>,
    specs_by_id: &mut HashMap<String, SessionSpec>,
) -> Option<Joined> {
    let free = batch_max().saturating_sub(active.len());
    if free == 0 {
        return None;
    }
    // Peek first: never lease a waiter this batch cannot absorb.
    let compatible = {
        let sched = state.work_scheduler.lock().await;
        sched
            .peek_next_microbatch_key(now_ms())
            .is_some_and(|(_, key)| key.as_deref() == Some(worker))
    };
    if !compatible {
        return None;
    }
    let lease = {
        let mut sched = state.work_scheduler.lock().await;
        sched.next_batch(now_ms())
    }?;
    let lease_id = lease.lease_id;

    // Pull the leased jobs, keep the Text ones we have room for, hand the rest
    // back to the scheduler untouched.
    let mut admitted: Vec<PendingRequest> = Vec::new();
    let mut handed_back: Vec<hipfire_scheduler::WorkloadSpec> = Vec::new();
    {
        // Inbox lock only — never nested with the scheduler lock (the outer
        // loop takes them sequentially too, so no order to invert).
        let mut inbox = state.batch_inbox.lock().await;
        for workload in &lease.workloads {
            let Some(job) = inbox.remove(&workload.id) else {
                continue;
            };
            match job {
                ScheduledJob::Text(p) if p.worker_key_id == worker && admitted.len() < free => {
                    admitted.push(p);
                }
                other => {
                    inbox.insert(workload.id.clone(), other);
                    handed_back.push(workload.clone());
                }
            }
        }
    }
    if !handed_back.is_empty() {
        let mut sched = state.work_scheduler.lock().await;
        for workload in handed_back {
            let _ = sched.enqueue(workload);
        }
    }
    if admitted.is_empty() {
        state.work_scheduler.lock().await.complete(lease_id);
        return None;
    }

    let specs: Vec<SessionSpec> = admitted.iter().map(|p| p.spec.clone()).collect();
    let events = match engine
        .generate_batch_prefill(build_batch_prefill_request(batch_id, worker, &specs))
        .await
    {
        Ok(events) => events,
        Err(e) => {
            // The running batch is untouched — fail only the joiners.
            for p in &admitted {
                let _ = p.tx.send(BatchEvent::Error(format!("join prefill: {e}")));
            }
            state.work_scheduler.lock().await.complete(lease_id);
            return None;
        }
    };
    let mut joined_positions: HashMap<String, usize> = HashMap::new();
    collect_prefill_positions(&events, &mut joined_positions);

    let mut joined_specs = Vec::new();
    for p in admitted {
        let id = p.spec.id.clone();
        let Some(pos) = joined_positions.get(&id).copied() else {
            let _ = p.tx.send(BatchEvent::Error(
                "join prefill produced no session state".to_string(),
            ));
            continue;
        };
        positions.insert(id.clone(), pos);
        remaining.insert(id.clone(), p.spec.max_tokens.max(1));
        txs.insert(id.clone(), p.tx.clone());
        specs_by_id.insert(id.clone(), p.spec.clone());
        joined_specs.push(p.spec);
        active.push(id);
    }
    if joined_specs.is_empty() {
        state.work_scheduler.lock().await.complete(lease_id);
        return None;
    }
    if std::env::var("HIPFIRE_BATCH_RUNNER_DEBUG").as_deref() == Ok("1") {
        eprintln!(
            "[batch-cycle] joined {} session(s) mid-cycle; batch now {}",
            joined_specs.len(),
            active.len()
        );
    }
    {
        let mut tel = state.batch_telemetry.lock().await;
        tel.joins += 1;
        tel.joined_sessions += joined_specs.len() as u64;
    }
    Some(Joined {
        specs: joined_specs,
        lease_ids: vec![lease_id],
    })
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
) -> CycleOutcome {
    let worker = batch[0].worker_key_id.clone();
    let batch_id = format!("batch-{}", batch[0].spec.id);
    let resuming = batch[0].resume_position.is_some();
    // `mut`: mid-cycle joiners are appended so the end-of-cycle release frees
    // their sessions too.
    let mut specs: Vec<SessionSpec> = batch.iter().map(|p| p.spec.clone()).collect();
    let mut join_leases: Vec<u64> = Vec::new();
    if std::env::var("HIPFIRE_BATCH_RUNNER_DEBUG").as_deref() == Ok("1") {
        eprintln!(
            "[batch-cycle] {} {} request(s) into one batch",
            if resuming { "resumed" } else { "coalesced" },
            specs.len()
        );
    }

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
        let prefill_req = build_batch_prefill_request(&batch_id, &worker, &specs);
        let events = match engine.generate_batch_prefill(prefill_req).await {
            Ok(events) => events,
            Err(e) => {
                fail_all(&txs, &format!("batch prefill: {e}"));
                return CycleOutcome::Completed;
            }
        };
        collect_prefill_positions(&events, &mut positions);
    }

    let mut active: Vec<String> = specs
        .iter()
        .map(|s| s.id.clone())
        .filter(|id| positions.contains_key(id))
        .collect();
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
    while !active.is_empty() {
        // Drop sessions whose client disconnected (response receiver closed):
        // stop decoding, and don't park/resume them. A session that was parked
        // and then abandoned is retired here on its resume cycle. Their KV is
        // freed with the rest of the batch at cycle end.
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
                    let _ = tx.send(BatchEvent::Done(
                        serde_json::json!({ "finish_reason": "stop" }),
                    ));
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
            let rem = remaining.get_mut(id).map(|r| {
                *r = r.saturating_sub(1);
                *r
            });
            if stop || rem == Some(0) {
                if let Some(tx) = txs.get(id) {
                    let _ = tx.send(BatchEvent::Done(serde_json::json!({
                        "finish_reason": if stop { "stop" } else { "length" }
                    })));
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
        active = still_active;

        // Join at the step boundary: admit compatible waiters into THIS batch
        // rather than making them wait for it to drain. Tried before the park
        // check below, so a same-model waiter joins instead of forcing a park —
        // one fused batch beats two batches with a parked one pinning VRAM.
        if join_enabled() && !active.is_empty() {
            if let Some(joined) = try_join(
                engine,
                state,
                &batch_id,
                &worker,
                &mut active,
                &mut positions,
                &mut remaining,
                &mut txs,
                &mut specs_by_id,
            )
            .await
            {
                specs.extend(joined.specs);
                join_leases.extend(joined.lease_ids);
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
            if waiter.is_some_and(|top| top < running_priority) {
                if std::env::var("HIPFIRE_BATCH_RUNNER_DEBUG").as_deref() == Ok("1") {
                    eprintln!(
                        "[batch-cycle] parked {} session(s) at step {steps}: pri {running_priority} yields to {}",
                        active.len(),
                        waiter.unwrap()
                    );
                }
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
                {
                    let mut sched = state.work_scheduler.lock().await;
                    for id in &join_leases {
                        sched.complete(*id);
                    }
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

    // Mid-cycle joins hold their own leases; the outer loop only knows about
    // the one it took, so release ours here.
    {
        let mut sched = state.work_scheduler.lock().await;
        for id in &join_leases {
            sched.complete(*id);
        }
    }

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
        let req = build_batch_prefill_request("batch-1", "worker-xyz", &specs);
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
