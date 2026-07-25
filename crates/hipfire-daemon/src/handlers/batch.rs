//! Batched prefill / decode over N co-resident sessions, and the prefix-hash
//! preflight that lets a caller skip re-prefilling a shared prefix.
//!
//! These are the "extended control plane" ops: driven by the server's batch
//! runner rather than by the `DaemonEngine` convenience methods. Each one
//! re-parses the raw message with an authoritative validator instead of trusting
//! the routing-only enum variant.

// Handler bodies were lifted verbatim out of `main()`, so they depend on the same
// root-level imports and arch aliases that the crate root sets up.
use crate::*;

pub(crate) fn prefill(daemon_state: &mut DaemonState, msg: &serde_json::Value) {
    match validate_generate_batch_prefill(&msg) {
        Ok(envelope) => {
            let target_worker_id = message_worker_id(&msg);
            if daemon_state.dummy_model.is_none() {
                match activate_model_worker(
                    &target_worker_id,
                    &mut daemon_state.active_worker_id,
                    &mut daemon_state.model,
                    &mut daemon_state.gpu,
                    &mut daemon_state.resident_models,
                ) {
                    Ok(true) => {}
                    Ok(false) => {
                        emit_error_with_id(
                            &mut daemon_state.out.stdout,
                            &envelope.id,
                            format!("unknown model worker {target_worker_id}"),
                        );
                        return;
                    }
                    Err(e) => {
                        emit_error_with_id(
                            &mut daemon_state.out.stdout,
                            &envelope.id,
                            format!("worker switch failed: {e}"),
                        );
                        return;
                    }
                }
            }
            if envelope.is_probe() {
                if daemon_state.dummy_model.is_some() {
                    emit_dummy_generate_batch_prefill_ready(
                        &mut daemon_state.out.stdout,
                        &envelope,
                    );
                    return;
                }
                match daemon_state.model.as_ref() {
                    Some(m) if is_qwen35_family_arch_id(m.arch_id) && m.pp == 1 => {
                        emit_generate_batch_prefill_ready(&mut daemon_state.out.stdout, &envelope);
                    }
                    #[cfg(feature = "arch-lfm2moe")]
                    Some(m) if m.arch_id == ARCH_ID_LFM2_MOE && m.pp == 1 => {
                        emit_lfm2_generate_batch_prefill_ready(
                            &mut daemon_state.out.stdout,
                            &envelope,
                        );
                    }
                    Some(m) => {
                        let reason = format!(
                            "generate_batch_prefill currently supports qwen35/qwen35-moe and lfm2-moe only (arch_id={})",
                            m.arch_id
                        );
                        emit_generate_batch_prefill_unsupported(
                            &mut daemon_state.out.stdout,
                            &envelope,
                            &reason,
                        );
                    }
                    None => {
                        emit_generate_batch_prefill_unsupported(
                            &mut daemon_state.out.stdout,
                            &envelope,
                            "no model loaded",
                        );
                    }
                }
                return;
            }
            if let Some(dummy) = daemon_state.dummy_model.as_mut() {
                tracing::info!(
                    request_id = envelope.id,
                    batch_id = envelope.batch_id,
                    sessions = envelope.session_count,
                    "dummy generate_batch_prefill"
                );
                if let Err(e) =
                    run_generate_batch_prefill_dummy(dummy, &mut daemon_state.out.stdout, &envelope)
                {
                    emit_error_with_id(&mut daemon_state.out.stdout, &envelope.id, e);
                }
                return;
            }
            let m = match daemon_state.model.as_mut() {
                Some(m) => m,
                None => {
                    emit_error_with_id(
                        &mut daemon_state.out.stdout,
                        &envelope.id,
                        "no model loaded",
                    );
                    return;
                }
            };
            if is_qwen35_family_arch_id(m.arch_id) {
                if let Err(e) = run_generate_batch_prefill_serial_qwen35(
                    m,
                    &mut daemon_state.gpu,
                    &mut daemon_state.out.stdout,
                    &envelope,
                    daemon_state.pflash_state.is_some(),
                ) {
                    emit_error_with_id(&mut daemon_state.out.stdout, &envelope.id, e);
                }
            } else {
                #[cfg(feature = "arch-lfm2moe")]
                if m.arch_id == ARCH_ID_LFM2_MOE {
                    if let Err(e) = run_generate_batch_prefill_serial_lfm2(
                        m,
                        &mut daemon_state.gpu,
                        &mut daemon_state.out.stdout,
                        &envelope,
                    ) {
                        emit_error_with_id(&mut daemon_state.out.stdout, &envelope.id, e);
                    }
                    return;
                }
                emit_error_with_id(
                    &mut daemon_state.out.stdout,
                    &envelope.id,
                    format!(
                        "generate_batch_prefill currently supports qwen35/qwen35-moe and lfm2-moe only (arch_id={})",
                        m.arch_id
                    ),
                );
            }
        }
        Err(e) => {
            let id = msg.get("id").and_then(|v| v.as_str()).unwrap_or("");
            emit_error_with_id(&mut daemon_state.out.stdout, id, e);
        }
    }
}

pub(crate) fn prefix_hash_preflight(daemon_state: &mut DaemonState, msg: &serde_json::Value) {
    match validate_prefix_hash_preflight(&msg) {
        Ok(envelope) => {
            let target_worker_id = message_worker_id(&msg);
            match activate_model_worker(
                &target_worker_id,
                &mut daemon_state.active_worker_id,
                &mut daemon_state.model,
                &mut daemon_state.gpu,
                &mut daemon_state.resident_models,
            ) {
                Ok(true) => {}
                Ok(false) => {
                    emit_error_with_id(
                        &mut daemon_state.out.stdout,
                        &envelope.id,
                        format!("unknown model worker {target_worker_id}"),
                    );
                    return;
                }
                Err(e) => {
                    emit_error_with_id(
                        &mut daemon_state.out.stdout,
                        &envelope.id,
                        format!("worker switch failed: {e}"),
                    );
                    return;
                }
            }
            let m = match daemon_state.model.as_ref() {
                Some(m) => m,
                None => {
                    emit_error_with_id(
                        &mut daemon_state.out.stdout,
                        &envelope.id,
                        "no model loaded",
                    );
                    return;
                }
            };
            let preflight_result = if is_qwen35_family_arch_id(m.arch_id) {
                run_prefix_hash_preflight_qwen35(m, &mut daemon_state.out.stdout, &envelope)
            } else {
                #[cfg(feature = "arch-lfm2moe")]
                {
                    if m.arch_id == ARCH_ID_LFM2_MOE {
                        run_prefix_hash_preflight_lfm2(m, &mut daemon_state.out.stdout, &envelope)
                    } else {
                        Err(format!(
                            "prefix_hash_preflight currently supports qwen35/qwen35-moe and lfm2-moe only (arch_id={})",
                            m.arch_id
                        ))
                    }
                }
                #[cfg(not(feature = "arch-lfm2moe"))]
                {
                    Err(format!(
                        "prefix_hash_preflight currently supports qwen35/qwen35-moe only (arch_id={})",
                        m.arch_id
                    ))
                }
            };
            if let Err(e) = preflight_result {
                emit_error_with_id(&mut daemon_state.out.stdout, &envelope.id, e);
            }
        }
        Err(e) => {
            let id = msg.get("id").and_then(|v| v.as_str()).unwrap_or("");
            emit_error_with_id(&mut daemon_state.out.stdout, id, e);
        }
    }
}

pub(crate) fn decode_step(daemon_state: &mut DaemonState, msg: &serde_json::Value) {
    match validate_generate_batch_decode(&msg) {
        Ok(envelope) => {
            let target_worker_id = message_worker_id(&msg);
            match activate_model_worker(
                &target_worker_id,
                &mut daemon_state.active_worker_id,
                &mut daemon_state.model,
                &mut daemon_state.gpu,
                &mut daemon_state.resident_models,
            ) {
                Ok(true) => {}
                Ok(false) => {
                    emit_error_with_id(
                        &mut daemon_state.out.stdout,
                        &envelope.id,
                        format!("unknown model worker {target_worker_id}"),
                    );
                    return;
                }
                Err(e) => {
                    emit_error_with_id(
                        &mut daemon_state.out.stdout,
                        &envelope.id,
                        format!("worker switch failed: {e}"),
                    );
                    return;
                }
            }
            let m = match daemon_state.model.as_mut() {
                Some(m) => m,
                None => {
                    emit_error_with_id(
                        &mut daemon_state.out.stdout,
                        &envelope.id,
                        "no model loaded",
                    );
                    return;
                }
            };
            if let Err(e) = run_generate_batch_decode_step_qwen35(
                m,
                &mut daemon_state.gpu,
                &mut daemon_state.out.stdout,
                &envelope,
            ) {
                emit_error_with_id(&mut daemon_state.out.stdout, &envelope.id, e);
            }
        }
        Err(e) => {
            let id = msg.get("id").and_then(|v| v.as_str()).unwrap_or("");
            emit_error_with_id(&mut daemon_state.out.stdout, id, e);
        }
    }
}
