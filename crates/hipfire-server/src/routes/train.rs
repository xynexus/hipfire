//! Drafter-training launch route.
//!
//! `POST /train/drafter` enqueues one in-daemon SSM-drafter training run onto the
//! batch runner (the one GPU arbiter), so training shares priority admission with
//! text/steer/embed/image on the single resident daemon — no second
//! GPU-flock-holding process. The run HOLDS the runner turn to completion
//! (minutes); it is queued at a low priority so interactive text/steer yield ahead
//! of it. Micro-step preemption is a documented follow-on (see batch_runner).
//!
//! This mirrors the steer route. Label source is FILE-based (`labels.path` = a
//! JSONL + `.embed.bin` QEMB sidecar); the in-process teacher-tap capture source
//! is a separate follow-on.

use axum::{extract::State, response::Response, Json};
use hipfire_scheduler::{
    server_prefill_batch_enabled, SchedulerPolicyEnv, WorkloadClass, WorkloadResources,
    WorkloadSpec,
};
use serde_json::json;
use uuid::Uuid;

use crate::batch_runner::{ScheduledJob, TrainJob};
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

/// `POST /train/drafter` — run one in-daemon drafter training job to completion.
/// Body is the `train_drafter` request minus `type` (`{arch, output, labels, train, config}`).
pub async fn post_train_drafter(
    state: State<SharedState>,
    body: Json<serde_json::Value>,
) -> Response {
    enqueue_train(state, body, "train_drafter").await
}

/// `POST /train/lora` — run one in-daemon LoRA-adapter training job to completion.
/// Body is the `train_lora` request minus `type` (`{output, data/labels, train}`).
/// Same runner/lease path as `/train/drafter`. NOTE: this trains hipfire-train's
/// own frozen-base LlamaModel, NOT the served qwen35 arch's adapters — and it is
/// currently a scaffold (the daemon returns a not-implemented error until the
/// assembled LoRA loop lands). See docs/plans/2026-07-19-daemon-training-steering.md.
pub async fn post_train_lora(state: State<SharedState>, body: Json<serde_json::Value>) -> Response {
    enqueue_train(state, body, "train_lora").await
}

/// Shared body for the training routes: stamp the raw-JSON wire `type`, enqueue a
/// singleton `Training` lease on the batch runner, and await the terminal payload.
async fn enqueue_train(
    State(state): State<SharedState>,
    Json(mut body): Json<serde_json::Value>,
    wire_type: &str,
) -> Response {
    if !server_prefill_batch_enabled(&SchedulerPolicyEnv::from_pairs(std::env::vars())) {
        return err_response(
            axum::http::StatusCode::SERVICE_UNAVAILABLE,
            "train routes require the batch runner (HIPFIRE_SERVER_PREFILL_BATCH != 0)",
        );
    }
    // Stamp the wire type the daemon dispatches on (raw-JSON op).
    if let Some(obj) = body.as_object_mut() {
        obj.insert("type".to_string(), json!(wire_type));
        // Both train ops are micro-step preemptible: the daemon keys the resident
        // training session on `run_id`, and the runner re-enqueues one `quantum`
        // at a time (train_lora = steps, train_drafter = epochs). Inject defaults
        // when the caller omits them.
        if wire_type == "train_lora" || wire_type == "train_drafter" {
            obj.entry("run_id")
                .or_insert_with(|| json!(Uuid::new_v4().to_string()));
            obj.entry("quantum").or_insert_with(|| json!(25));
        }
    } else {
        return err_response(
            axum::http::StatusCode::BAD_REQUEST,
            "body must be a JSON object",
        );
    }

    let req_id = Uuid::new_v4().to_string();
    let (tx, rx) = tokio::sync::oneshot::channel();
    state.batch_inbox.lock().await.insert(
        req_id.clone(),
        ScheduledJob::Train(TrainJob { req: body, tx }),
    );
    // Singleton Training lease (never coalesces, exclusive). Priority 160 — well
    // below interactive text (64) and steer (96) so training yields to them, but
    // ahead of the u8::MAX idle floor.
    let workload = WorkloadSpec::singleton(
        req_id.clone(),
        WorkloadClass::Training,
        160,
        now_ms(),
        WorkloadResources::default(),
    );
    if let Err(e) = state.work_scheduler.lock().await.enqueue(workload) {
        state.batch_inbox.lock().await.remove(&req_id);
        return err_response(
            axum::http::StatusCode::SERVICE_UNAVAILABLE,
            format!("scheduler admission: {e}"),
        );
    }
    state.prefill_notify.notify_waiters();
    match rx.await {
        Ok(Ok(payload)) => ok_json(payload),
        Ok(Err(e)) => err_response(
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            format!("train: {e}"),
        ),
        Err(_) => err_response(
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            "train job dropped before completion",
        ),
    }
}
