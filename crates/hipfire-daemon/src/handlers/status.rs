//! Status, inventory and liveness requests: cheap, non-GPU control-plane
//! replies plus the two dead wire variants.
//!
//! `Abort` / `ForceAnswer` are answered here only to say they are not handled on
//! this channel. There is no control channel yet, so an abort can only be read
//! after the generation it wanted to cancel has already finished; making them
//! real is what the socket transport and a separate control plane unlock.

// Handler bodies were lifted verbatim out of `main()`, so they depend on the same
// root-level imports and arch aliases (`qwen35`, `deepseek4`, `minimax`, `lfm2moe`,
// `qwen2`, `prompt_frame`) that the crate root sets up. Glob-importing the root
// keeps that dependency in one place instead of re-deriving it per module.
use crate::*;

pub(crate) fn registry(
    daemon_state: &mut DaemonState,
    llm_registry: &hipfire_model::LlmModelRegistry,
) {
    daemon_state.out.emit(serde_json::json!({
        "type": "model_registry",
        "registry": llm_registry
    }));
}

pub(crate) fn worker_status(daemon_state: &mut DaemonState) {
    let status = resident_worker_status_json(
        &daemon_state.active_worker_id,
        daemon_state.model.as_ref(),
        &daemon_state.resident_models,
    );
    daemon_state.out.emit(status);
}

pub(crate) fn resource_status(daemon_state: &mut DaemonState) {
    let status = daemon_state.resource_reservations.status_json();
    daemon_state.out.emit(status);
}

pub(crate) fn inventory(daemon_state: &mut DaemonState) {
    let inventory = daemon_accelerator_inventory(&mut daemon_state.gpu);
    let mut payload = serde_json::to_value(inventory)
        .unwrap_or_else(|_| serde_json::json!({"source": "daemon", "devices": []}));
    payload["type"] = serde_json::json!("inventory");
    daemon_state.out.emit(payload);
}

pub(crate) fn ping(daemon_state: &mut DaemonState) {
    daemon_state.out.emit(serde_json::json!({ "type": "pong" }));
}

pub(crate) fn unsupported_on_request_channel(
    daemon_state: &mut DaemonState,
    msg: &serde_json::Value,
    msg_type: &str,
) {
    emit_error_with_id(
        &mut daemon_state.out.stdout,
        msg.get("id").and_then(|v| v.as_str()).unwrap_or(""),
        format!("{msg_type} is handled on the control channel, not the request channel"),
    );
}
