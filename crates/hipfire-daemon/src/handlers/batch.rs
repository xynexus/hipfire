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
                            &mut daemon_state.out.sink,
                            &envelope.id,
                            format!("unknown model worker {target_worker_id}"),
                        );
                        return;
                    }
                    Err(e) => {
                        emit_error_with_id(
                            &mut daemon_state.out.sink,
                            &envelope.id,
                            format!("worker switch failed: {e}"),
                        );
                        return;
                    }
                }
            }
            if envelope.is_probe() {
                if daemon_state.dummy_model.is_some() {
                    emit_dummy_generate_batch_prefill_ready(&mut daemon_state.out.sink, &envelope);
                    return;
                }
                match daemon_state.model.as_ref() {
                    // Arch identity and runtime envelope both come from the
                    // executor seam; this handler holds no arch knowledge.
                    Some(m) => match batch_executor_for(m.arch_id) {
                        Some(ex) => match ex.probe(m) {
                            Ok(()) => ex.emit_ready(&mut daemon_state.out.sink, &envelope),
                            Err(reason) => emit_generate_batch_prefill_unsupported(
                                &mut daemon_state.out.sink,
                                &envelope,
                                &reason,
                            ),
                        },
                        None => {
                            let reason =
                                batch_unsupported_reason("generate_batch_prefill", m.arch_id);
                            emit_generate_batch_prefill_unsupported(
                                &mut daemon_state.out.sink,
                                &envelope,
                                &reason,
                            );
                        }
                    },
                    None => {
                        emit_generate_batch_prefill_unsupported(
                            &mut daemon_state.out.sink,
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
                    run_generate_batch_prefill_dummy(dummy, &mut daemon_state.out.sink, &envelope)
                {
                    emit_error_with_id(&mut daemon_state.out.sink, &envelope.id, e);
                }
                return;
            }
            let m = match daemon_state.model.as_mut() {
                Some(m) => m,
                None => {
                    emit_error_with_id(&mut daemon_state.out.sink, &envelope.id, "no model loaded");
                    return;
                }
            };
            let executor = match batch_executor_for(m.arch_id) {
                Some(ex) => ex,
                None => {
                    emit_error_with_id(
                        &mut daemon_state.out.sink,
                        &envelope.id,
                        batch_unsupported_reason("generate_batch_prefill", m.arch_id),
                    );
                    return;
                }
            };
            if let Err(e) = executor.prefill(
                m,
                &mut daemon_state.gpu,
                &mut daemon_state.out.sink,
                &envelope,
                daemon_state.pflash_state.is_some(),
            ) {
                emit_error_with_id(&mut daemon_state.out.sink, &envelope.id, e);
            }
        }
        Err(e) => {
            let id = msg.get("id").and_then(|v| v.as_str()).unwrap_or("");
            emit_error_with_id(&mut daemon_state.out.sink, id, e);
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
                        &mut daemon_state.out.sink,
                        &envelope.id,
                        format!("unknown model worker {target_worker_id}"),
                    );
                    return;
                }
                Err(e) => {
                    emit_error_with_id(
                        &mut daemon_state.out.sink,
                        &envelope.id,
                        format!("worker switch failed: {e}"),
                    );
                    return;
                }
            }
            let m = match daemon_state.model.as_ref() {
                Some(m) => m,
                None => {
                    emit_error_with_id(&mut daemon_state.out.sink, &envelope.id, "no model loaded");
                    return;
                }
            };
            let preflight_result = match batch_executor_for(m.arch_id) {
                Some(ex) => ex.prefix_hash_preflight(m, &mut daemon_state.out.sink, &envelope),
                None => Err(batch_unsupported_reason("prefix_hash_preflight", m.arch_id)),
            };
            if let Err(e) = preflight_result {
                emit_error_with_id(&mut daemon_state.out.sink, &envelope.id, e);
            }
        }
        Err(e) => {
            let id = msg.get("id").and_then(|v| v.as_str()).unwrap_or("");
            emit_error_with_id(&mut daemon_state.out.sink, id, e);
        }
    }
}

pub(crate) fn decode_step(daemon_state: &mut DaemonState, msg: &serde_json::Value) {
    match validate_generate_batch_decode(&msg) {
        Ok(envelope) => {
            // How wide each fused decode step actually is. Without this, a flat
            // aggregate-throughput curve is unattributable: N sessions arriving
            // as N one-row steps (never coalesced) looks identical from outside
            // to N sessions in one step that fails to amortise the weight pass.
            // Those are opposite defects.
            if std::env::var("HIPFIRE_BATCH_WIDTH_TRACE").ok().as_deref() == Some("1") {
                eprintln!("  [batch] decode_step rows={}", envelope.sessions.len());
            }
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
                        &mut daemon_state.out.sink,
                        &envelope.id,
                        format!("unknown model worker {target_worker_id}"),
                    );
                    return;
                }
                Err(e) => {
                    emit_error_with_id(
                        &mut daemon_state.out.sink,
                        &envelope.id,
                        format!("worker switch failed: {e}"),
                    );
                    return;
                }
            }
            let m = match daemon_state.model.as_mut() {
                Some(m) => m,
                None => {
                    emit_error_with_id(&mut daemon_state.out.sink, &envelope.id, "no model loaded");
                    return;
                }
            };
            // This path previously called the qwen35 decode unconditionally,
            // with no arch check — safe only because the server routes eligible
            // arches. The seam makes the guard structural.
            let result = match batch_executor_for(m.arch_id) {
                Some(ex) => ex.decode_step(
                    m,
                    &mut daemon_state.gpu,
                    &mut daemon_state.out.sink,
                    &envelope,
                ),
                None => Err(batch_unsupported_reason(
                    "generate_batch_decode_step",
                    m.arch_id,
                )),
            };
            if let Err(e) = result {
                emit_error_with_id(&mut daemon_state.out.sink, &envelope.id, e);
            }
        }
        Err(e) => {
            let id = msg.get("id").and_then(|v| v.as_str()).unwrap_or("");
            emit_error_with_id(&mut daemon_state.out.sink, id, e);
        }
    }
}
