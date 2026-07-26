//! Steering / abliteration control routes.
//!
//! Thin pass-throughs of the daemon's steer ops, routed through the batch runner
//! (the one GPU arbiter) so a capture or apply shares priority admission with
//! text/image/embed on the single resident daemon — no second GPU-flock-holding
//! process. The client (e.g. `hipfire-steer-harness`) still owns the driver
//! logic (derive directions, sweep strengths); these routes only move the
//! GPU-touching ops onto the shared daemon.
//!
//! Capture is a WHOLE session (begin → prefill each prompt → finish → means) in
//! one request, executed atomically in one runner turn: the capture hook is
//! process-global, so an interleaved generate would fold its residuals into the
//! means. Apply/clear are instantaneous daemon state ops.

use axum::{extract::State, response::Response, Json};
use hipfire_daemon_adapter::SteerApplyRequest;
use hipfire_scheduler::{
    server_prefill_batch_enabled, SchedulerPolicyEnv, WorkloadClass, WorkloadResources,
    WorkloadSpec,
};
use serde::Deserialize;
use serde_json::json;
use uuid::Uuid;

use crate::batch_runner::{ScheduledJob, SteerJob, SteerOp};
use crate::state::SharedState;

fn now_ms() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn ok_json(value: serde_json::Value) -> Response {
    use axum::response::IntoResponse;
    Json(value).into_response()
}

fn err_response(code: axum::http::StatusCode, msg: impl Into<String>) -> Response {
    use axum::response::IntoResponse;
    (code, Json(json!({ "error": msg.into() }))).into_response()
}

/// Enqueue one steer op onto the runner and await its result. Returns the capture
/// means (`Some` only for a capture session) or a ready-to-return error response.
async fn run_on_runner(
    state: &SharedState,
    op: SteerOp,
    key: String,
) -> Result<Option<Vec<Vec<f32>>>, Response> {
    if !server_prefill_batch_enabled(&SchedulerPolicyEnv::from_pairs(std::env::vars())) {
        return Err(err_response(
            axum::http::StatusCode::SERVICE_UNAVAILABLE,
            "steer routes require the batch runner (HIPFIRE_SERVER_PREFILL_BATCH != 0)",
        ));
    }
    let req_id = Uuid::new_v4().to_string();
    let (tx, rx) = tokio::sync::oneshot::channel();
    state
        .batch_inbox
        .lock()
        .await
        .insert(req_id.clone(), ScheduledJob::Steer(SteerJob { op, tx }));
    // Its own lease (never coalesces): unique microbatch key, size 1, Maintenance
    // class. Priority 96 — below interactive text (64) so steering yields to live
    // serving, but still ahead of the u8::MAX idle floor.
    let workload = WorkloadSpec::microbatchable(
        req_id.clone(),
        WorkloadClass::Maintenance,
        96,
        now_ms(),
        WorkloadResources::default(),
        format!("steer:{key}:{req_id}"),
        1,
    );
    if let Err(e) = state.work_scheduler.lock().await.enqueue(workload) {
        state.batch_inbox.lock().await.remove(&req_id);
        return Err(err_response(
            axum::http::StatusCode::SERVICE_UNAVAILABLE,
            format!("scheduler admission: {e}"),
        ));
    }
    state.prefill_notify.notify_waiters();
    match rx.await {
        Ok(Ok(means)) => Ok(means),
        Ok(Err(e)) => Err(err_response(
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            format!("steer: {e}"),
        )),
        Err(_) => Err(err_response(
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            "steer job dropped before completion",
        )),
    }
}

#[derive(Debug, Deserialize)]
pub struct SteerPrompt {
    #[serde(default)]
    pub system: String,
    pub user: String,
}

#[derive(Debug, Deserialize)]
pub struct CaptureRequest {
    pub num_layers: usize,
    pub hidden: usize,
    pub prompts: Vec<SteerPrompt>,
}

/// `POST /steer/capture` — run a whole capture session and return per-block means.
pub async fn post_steer_capture(
    State(state): State<SharedState>,
    Json(body): Json<CaptureRequest>,
) -> Response {
    if body.prompts.is_empty() {
        return err_response(
            axum::http::StatusCode::BAD_REQUEST,
            "prompts must be non-empty",
        );
    }
    let prompts = body
        .prompts
        .into_iter()
        .map(|p| (p.system, p.user))
        .collect();
    let op = SteerOp::CaptureSession {
        num_layers: body.num_layers,
        hidden: body.hidden,
        prompts,
    };
    match run_on_runner(&state, op, "capture".to_string()).await {
        Ok(means) => ok_json(json!({ "means": means.unwrap_or_default() })),
        Err(resp) => resp,
    }
}

#[derive(Debug, Deserialize)]
pub struct ApplyRequest {
    pub directions: Vec<Vec<f32>>,
    /// "steer" or "ablate".
    pub mode: String,
    pub strength: f32,
    pub layer_start: usize,
    pub layer_end: usize,
}

/// `POST /steer/apply` — install an apply session; ordinary generation is then
/// steered (that generation already rides the runner, so apply needs no turn).
pub async fn post_steer_apply(
    State(state): State<SharedState>,
    Json(body): Json<ApplyRequest>,
) -> Response {
    let op = SteerOp::BeginApply(SteerApplyRequest {
        directions: body.directions,
        mode: body.mode,
        strength: body.strength,
        layer_start: body.layer_start,
        layer_end: body.layer_end,
    });
    match run_on_runner(&state, op, "apply".to_string()).await {
        Ok(_) => ok_json(json!({ "ok": true })),
        Err(resp) => resp,
    }
}

/// `POST /steer/clear` — tear down any active steer session.
pub async fn post_steer_clear(State(state): State<SharedState>) -> Response {
    match run_on_runner(&state, SteerOp::Clear, "clear".to_string()).await {
        Ok(_) => ok_json(json!({ "ok": true })),
        Err(resp) => resp,
    }
}
