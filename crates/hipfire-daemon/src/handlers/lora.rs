//! LoRA adapter stack (shares the steer APPLY session). Load a `.lora`
//! container onto the live model, adjust per-adapter intensity, stack or remove
//! adapters, and list — all without reload. The abliteration directions
//! materialized by `lora_export`/the harness become a portable adapter served
//! here. See docs/plans/2026-06-30-abliteration-lora.md.

// Handler bodies were lifted verbatim out of `main()`, so they depend on the same
// root-level imports and arch aliases (`qwen35`, `deepseek4`, `minimax`, `lfm2moe`,
// `qwen2`, `prompt_frame`) that the crate root sets up. Glob-importing the root
// keeps that dependency in one place instead of re-deriving it per module.
use crate::*;

pub(crate) fn load(daemon_state: &mut DaemonState, msg: &serde_json::Value) {
    let Some(path) = msg.get("path").and_then(|v| v.as_str()).map(String::from) else {
        daemon_state
            .out
            .error("lora_load: missing 'path'".to_string());
        return;
    };
    let scale_override = msg.get("scale").and_then(|v| v.as_f64()).map(|v| v as f32);
    let id_override = msg.get("id").and_then(|v| v.as_str()).map(String::from);
    let mut adapter = match hipfire_lora_hfq::read_lora_any(std::path::Path::new(&path)) {
        Ok(a) => a,
        Err(e) => {
            daemon_state.out.error(format!("lora_load: {e}"));
            return;
        }
    };
    if let Some(new_id) = id_override {
        adapter.id = new_id;
    }
    // The adapter is base-specific (directions sized to the model's
    // hidden width); reject a mismatched load before it faults at apply.
    let model_hidden = daemon_state.model.as_ref().and_then(|m| {
        m.gemma3_text
            .as_ref()
            .map(|b| b.config.hidden_size)
            .or_else(|| m.gemma3_vl.as_ref().map(|b| b.text_cfg.hidden_size))
    });
    if let Some(h) = model_hidden {
        if adapter.meta.hidden != h {
            daemon_state.out.error(format!(
                "lora_load: adapter hidden {} != model hidden {h}",
                adapter.meta.hidden
            ));
            return;
        }
    }
    let id = adapter.id.clone();
    if let Err(e) = hipfire_steer::load_lora_adapter(&adapter) {
        daemon_state.out.error(format!("lora_load: {e}"));
        return;
    }
    if let Some(s) = scale_override {
        hipfire_steer::set_adapter_scale(&id, s);
    }
    daemon_state
        .out
        .emit(serde_json::json!({ "type": "lora_ok" }));
}

pub(crate) fn set_scale(daemon_state: &mut DaemonState, msg: &serde_json::Value) {
    let id = msg.get("id").and_then(|v| v.as_str()).map(String::from);
    let scale = msg.get("scale").and_then(|v| v.as_f64()).map(|v| v as f32);
    let (Some(id), Some(scale)) = (id, scale) else {
        daemon_state
            .out
            .error("lora_set_scale: missing 'id'/'scale'".to_string());
        return;
    };
    if hipfire_steer::set_adapter_scale(&id, scale) {
        daemon_state
            .out
            .emit(serde_json::json!({ "type": "lora_ok" }));
    } else {
        daemon_state
            .out
            .error(format!("lora_set_scale: no adapter {id:?} loaded"));
    }
}

pub(crate) fn unload(daemon_state: &mut DaemonState, msg: &serde_json::Value) {
    let Some(id) = msg.get("id").and_then(|v| v.as_str()).map(String::from) else {
        daemon_state
            .out
            .error("lora_unload: missing 'id'".to_string());
        return;
    };
    if hipfire_steer::unload_adapter(&id) {
        daemon_state
            .out
            .emit(serde_json::json!({ "type": "lora_ok" }));
    } else {
        daemon_state
            .out
            .error(format!("lora_unload: no adapter {id:?} loaded"));
    }
}

pub(crate) fn clear(daemon_state: &mut DaemonState) {
    hipfire_steer::clear();
    daemon_state
        .out
        .emit(serde_json::json!({ "type": "lora_ok" }));
}

pub(crate) fn list(daemon_state: &mut DaemonState) {
    let adapters: Vec<_> = hipfire_steer::loaded_adapters()
        .into_iter()
        .map(|(id, scale)| serde_json::json!({ "id": id, "scale": scale }))
        .collect();
    let resp = serde_json::json!({ "type": "lora_listed", "adapters": adapters });
    daemon_state.out.emit(resp);
}
