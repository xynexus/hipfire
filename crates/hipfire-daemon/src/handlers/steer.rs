//! Steering / abliteration: residual-stream capture and in-forward apply.
//!
//! The capture and apply branches are the same in-forward call site
//! (`maybe_steer_block`), branching on the active session. Note the session is
//! process-global and outlives the model it was captured against, which is why
//! load and unload clear it defensively — and why two steer ops must never
//! interleave.

// Handler bodies were lifted verbatim out of `main()`, so they depend on the same
// root-level imports and arch aliases (`qwen35`, `deepseek4`, `minimax`, `lfm2moe`,
// `qwen2`, `prompt_frame`) that the crate root sets up. Glob-importing the root
// keeps that dependency in one place instead of re-deriving it per module.
use crate::*;

pub(crate) fn begin_capture(daemon_state: &mut DaemonState, msg: &serde_json::Value) {
    let num_layers = msg
        .get("num_layers")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize);
    let hidden = msg
        .get("hidden")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize);
    let (Some(num_layers), Some(hidden)) = (num_layers, hidden) else {
        emit_error_with_id(
            &mut daemon_state.stdout,
            "",
            "steer_begin_capture: missing 'num_layers'/'hidden'".to_string(),
        );
        return;
    };
    hipfire_steer::begin_capture(num_layers, hidden);
    let _ = writeln!(daemon_state.stdout, r#"{{"type":"steer_ok"}}"#);
    let _ = daemon_state.stdout.flush();
}

pub(crate) fn capture(daemon_state: &mut DaemonState, msg: &serde_json::Value) {
    let system = msg
        .get("system")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let Some(user) = msg.get("user").and_then(|v| v.as_str()).map(String::from) else {
        emit_error_with_id(
            &mut daemon_state.stdout,
            "",
            "steer_capture: missing 'user'".to_string(),
        );
        return;
    };
    let Some(m) = daemon_state.model.as_mut() else {
        emit_error_with_id(
            &mut daemon_state.stdout,
            "",
            "steer_capture: no model loaded".to_string(),
        );
        return;
    };
    if m.pp != 1 {
        emit_error_with_id(
            &mut daemon_state.stdout,
            "",
            "steer_capture: requires a single-GPU resident model (pp == 1)".to_string(),
        );
        return;
    }
    let Some(tokenizer) = m.tokenizer.as_ref() else {
        emit_error_with_id(
            &mut daemon_state.stdout,
            "",
            "steer_capture: resident model has no tokenizer".to_string(),
        );
        return;
    };
    // Frame the turn byte-identically to the `generate` path so capture
    // sees the exact residuals serving would. gemma3 uses its literal
    // turn frame; qwen35 (loose-slot) uses its jinja `chat_template`
    // single-turn render.
    let system_opt = (!system.is_empty()).then_some(system.as_str());
    let framed = if is_qwen35_family_arch_id(m.arch_id) {
        match hipfire_serving_core::generate_arch::framed_qwen35_prompt(m, &user, system_opt) {
            Ok(f) => f,
            Err(e) => {
                emit_error_with_id(&mut daemon_state.stdout, "", format!("steer_capture: {e}"));
                return;
            }
        }
    } else {
        hipfire_serving_core::generate_arch::framed_gemma3_prompt(&user, system_opt)
    };
    let tokens = tokenizer.encode(&framed);
    if tokens.is_empty() {
        emit_error_with_id(
            &mut daemon_state.stdout,
            "",
            "steer_capture: empty prompt after framing".to_string(),
        );
        return;
    }
    // Prefill-only through whichever resident arch fires the
    // block-boundary hook so it observes the last-prompt-token residual
    // per block. No decode loop. gemma3 (12/13) folds via its backend
    // prefill; qwen35 (loose-slot) folds via a fresh single-sequence
    // capture prefill. Both hit `maybe_steer_block[_batched]`.
    use hipfire_runtime::arch::SimpleAr;
    let result: Result<(), String> = if is_qwen35_family_arch_id(m.arch_id) {
        run_steer_capture_prefill_qwen35(m, &mut daemon_state.gpu, &tokens)
    } else if let Some(b) = m.gemma3_text.as_mut() {
        b.state.reset();
        SimpleAr::prefill(b, &mut daemon_state.gpu, &tokens)
    } else if let Some(b) = m.gemma3_vl.as_mut() {
        b.state.reset();
        SimpleAr::prefill(b, &mut daemon_state.gpu, &tokens)
    } else {
        Err(format!(
            "steer_capture: arch_id {} is unsupported (need gemma3 or qwen35)",
            m.arch_id
        ))
    };
    match result {
        Ok(()) => {
            hipfire_steer::commit_capture();
            let _ = writeln!(daemon_state.stdout, r#"{{"type":"steer_ok"}}"#);
            let _ = daemon_state.stdout.flush();
        }
        Err(e) => emit_error_with_id(&mut daemon_state.stdout, "", format!("steer_capture: {e}")),
    }
}

pub(crate) fn begin_apply(daemon_state: &mut DaemonState, msg: &serde_json::Value) {
    let directions: Option<Vec<Vec<f32>>> =
        msg.get("directions")
            .and_then(|v| v.as_array())
            .map(|rows| {
                rows.iter()
                    .map(|row| {
                        row.as_array()
                            .map(|cols| {
                                cols.iter()
                                    .filter_map(|x| x.as_f64().map(|f| f as f32))
                                    .collect()
                            })
                            .unwrap_or_default()
                    })
                    .collect()
            });
    let Some(directions) = directions else {
        emit_error_with_id(
            &mut daemon_state.stdout,
            "",
            "steer_begin_apply: missing 'directions'".to_string(),
        );
        return;
    };
    let mode = match msg.get("mode").and_then(|v| v.as_str()).unwrap_or("ablate") {
        "steer" => hipfire_steer::SteerMode::Steer,
        "ablate" => hipfire_steer::SteerMode::Ablate,
        other => {
            emit_error_with_id(
                &mut daemon_state.stdout,
                "",
                format!("steer_begin_apply: unknown mode {other:?} (steer|ablate)"),
            );
            return;
        }
    };
    let strength = msg.get("strength").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32;
    let layer_start = msg.get("layer_start").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
    let layer_end = msg
        .get("layer_end")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(directions.len());
    hipfire_steer::begin_apply(hipfire_steer::SteerSpec {
        directions,
        mode,
        strength,
        layer_range: layer_start..layer_end,
    });
    let _ = writeln!(daemon_state.stdout, r#"{{"type":"steer_ok"}}"#);
    let _ = daemon_state.stdout.flush();
}

pub(crate) fn clear(daemon_state: &mut DaemonState) {
    hipfire_steer::clear();
    let _ = writeln!(daemon_state.stdout, r#"{{"type":"steer_ok"}}"#);
    let _ = daemon_state.stdout.flush();
}

pub(crate) fn finish_capture(daemon_state: &mut DaemonState) {
    match hipfire_steer::finish_capture() {
        Some(means) => {
            let resp = serde_json::json!({
                "type": "steer_captured",
                "means": means.0,
            });
            let _ = writeln!(daemon_state.stdout, "{resp}");
            let _ = daemon_state.stdout.flush();
        }
        None => emit_error_with_id(
            &mut daemon_state.stdout,
            "",
            "steer_finish_capture: no capture session active".to_string(),
        ),
    }
}
