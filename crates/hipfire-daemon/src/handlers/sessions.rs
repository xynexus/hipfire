//! Session-state lifecycle: reserve, describe, release.
//!
//! These are the control-plane half of the batched serving path — the server
//! reserves state, drives batch prefill/decode against it, then releases. They
//! are cheap and non-GPU apart from the arena bookkeeping.

// Handler bodies were lifted verbatim out of `main()`, so they depend on the same
// root-level imports and arch aliases (`qwen35`, `deepseek4`, `minimax`, `lfm2moe`,
// `qwen2`, `prompt_frame`) that the crate root sets up. Glob-importing the root
// keeps that dependency in one place instead of re-deriving it per module.
use crate::*;

pub(crate) fn release_sessions(daemon_state: &mut DaemonState, msg: &serde_json::Value) {
    let id = msg.get("id").and_then(|v| v.as_str()).unwrap_or("release");
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
                    id,
                    format!("unknown model worker {target_worker_id}"),
                );
                return;
            }
            Err(e) => {
                emit_error_with_id(
                    &mut daemon_state.out.stdout,
                    id,
                    format!("worker switch failed: {e}"),
                );
                return;
            }
        }
    }
    let request = match parse_release_sessions_request(&msg, &target_worker_id) {
        Ok(request) => request,
        Err(e) => {
            emit_error_with_id(&mut daemon_state.out.stdout, id, e);
            return;
        }
    };
    if let Some(dummy) = daemon_state.dummy_model.as_mut() {
        let released = dummy.release_sessions(&request.sessions);
        let done = release_sessions_done_json(
            id,
            request.sessions.len(),
            released,
            dummy.session_count(),
            None,
        );
        daemon_state.out.emit(done);
        return;
    }
    let m = match daemon_state.model.as_mut() {
        Some(m) => m,
        None => {
            emit_error_with_id(&mut daemon_state.out.stdout, id, "no model loaded");
            return;
        }
    };
    let arena_backend = loaded_model_state_arena_backend(m);
    match sequence_state_arena_release_sessions(
        arena_backend,
        m,
        &mut daemon_state.gpu,
        &request.sessions,
    ) {
        Ok(released) => {
            let worker = loaded_model_worker_runtime_view(m);
            let done = release_sessions_done_json(
                id,
                request.sessions.len(),
                released,
                sequence_state_arena_resident_session_count(arena_backend, m),
                Some(&worker),
            );
            daemon_state.out.emit(done);
        }
        Err(e) => emit_error_with_id(&mut daemon_state.out.stdout, id, e),
    }
}

pub(crate) fn reserve_session_state(daemon_state: &mut DaemonState, msg: &serde_json::Value) {
    let id = msg.get("id").and_then(|v| v.as_str()).unwrap_or("reserve");
    let target_worker_id = message_worker_id(&msg);
    daemon_state.generic_state_arena.purge_expired();
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
                    id,
                    format!("unknown model worker {target_worker_id}"),
                );
                return;
            }
            Err(e) => {
                emit_error_with_id(
                    &mut daemon_state.out.stdout,
                    id,
                    format!("worker switch failed: {e}"),
                );
                return;
            }
        }
    }
    let request = match parse_reserve_session_state_request(&msg, &target_worker_id) {
        Ok(request) => request,
        Err(e) => {
            emit_error_with_id(&mut daemon_state.out.stdout, id, e);
            return;
        }
    };
    let reservation_id = request.reservation_id.clone().unwrap_or_else(|| {
        format!(
            "reserve:{}:{}",
            request.worker_id,
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        )
    });
    let reservation_plan = if let Some(m) = daemon_state.model.as_ref() {
        let budget = request
            .budget_bytes
            .unwrap_or_else(resident_state_reservation_budget_bytes);
        let arena_backend = loaded_model_state_arena_backend(m);
        let descriptors = sequence_state_arena_page_descriptors(arena_backend, m);
        sequence_state_reservation_plan(
            &descriptors,
            request.physical_cap,
            daemon_state
                .generic_state_arena
                .outstanding_bytes_for_worker(&request.worker_id),
            budget,
        )
    } else if daemon_state.dummy_model.is_some() {
        let budget = request
            .budget_bytes
            .unwrap_or_else(resident_state_reservation_budget_bytes);
        sequence_state_reservation_plan_for_reserved_bytes(1024, 0, 0, budget)
    } else {
        emit_error_with_id(&mut daemon_state.out.stdout, id, "no model loaded");
        return;
    };
    if reservation_plan.rejected_for_memory_pressure {
        let rejected = reserve_session_state_rejected_json(
            id,
            &request.worker_id,
            reservation_plan.reserved_bytes,
            reservation_plan.current_session_bytes,
            reservation_plan.outstanding_reserved_bytes,
            reservation_plan.projected_reserved_bytes,
            reservation_plan.budget_bytes,
        );
        daemon_state.out.emit(rejected);
        return;
    }
    let reservation = daemon_state.generic_state_arena.reserve(
        &request.worker_id,
        reservation_id.clone(),
        &request.state_kinds,
        request.physical_cap,
        reservation_plan.reserved_bytes,
        request.ttl_ms,
    );
    let done = reserve_session_state_done_json(
        id,
        &reservation,
        reservation_plan.current_session_bytes,
        reservation_plan.outstanding_reserved_bytes,
        reservation_plan.projected_reserved_bytes,
        reservation_plan.budget_bytes,
    );
    daemon_state.out.emit(done);
}

pub(crate) fn describe_state(daemon_state: &mut DaemonState, msg: &serde_json::Value) {
    let id = msg
        .get("id")
        .and_then(|v| v.as_str())
        .unwrap_or("describe-state");
    daemon_state.generic_state_arena.purge_expired();
    let request = match parse_describe_sequence_state_request(&msg) {
        Ok(request) => request,
        Err(e) => {
            emit_error_with_id(&mut daemon_state.out.stdout, id, e);
            return;
        }
    };
    if parsed_handle_may_target_generic(&request.handle) {
        if let Some(reservation) = daemon_state
            .generic_state_arena
            .describe(&request.handle.id, request.handle.generation)
        {
            let done = session_state_reservation_describe_json(id, reservation);
            daemon_state.out.emit(done);
            return;
        }
    }
    let Some(described) = describe_loaded_sequence_state(
        &daemon_state.active_worker_id,
        daemon_state.model.as_ref(),
        &daemon_state.resident_models,
        &request.handle,
    ) else {
        emit_error_with_id(
            &mut daemon_state.out.stdout,
            id,
            format!(
                "describe_state unknown runtime_state_handle {}",
                request.handle.id
            ),
        );
        return;
    };
    let done = described_sequence_state_json(id, &described);
    daemon_state.out.emit(done);
}

pub(crate) fn release_state(daemon_state: &mut DaemonState, msg: &serde_json::Value) {
    let id = msg
        .get("id")
        .and_then(|v| v.as_str())
        .unwrap_or("release-reservation");
    let request = parse_release_sequence_state_request(&msg);
    let generic_handles = request
        .handles
        .iter()
        .filter(|handle| parsed_handle_may_target_generic(handle))
        .map(|handle| (handle.id.clone(), handle.generation))
        .collect::<Vec<_>>();
    let (generic_released, generic_released_bytes) =
        daemon_state.generic_state_arena.release(generic_handles);
    let (loaded_released, loaded_released_bytes) = match release_loaded_sequence_state_handles(
        &mut daemon_state.model,
        &mut daemon_state.resident_models,
        &mut daemon_state.gpu,
        &request.handles,
    ) {
        Ok(released) => released,
        Err(e) => {
            emit_error_with_id(&mut daemon_state.out.stdout, id, e);
            return;
        }
    };
    let done = release_state_done_json(
        request.response_kind,
        id,
        generic_released,
        generic_released_bytes,
        loaded_released,
        loaded_released_bytes,
    );
    daemon_state.out.emit(done);
}
