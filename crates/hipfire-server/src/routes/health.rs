use axum::{extract::State, response::Json};
use hipfire_model::AcceleratorInventory;
use hipfire_scheduler::{
    server_decode_batch_health_json, server_prefill_batch_health_json,
    server_state_cache_health_json, SchedulerPolicyEnv,
};
use hipfire_state::runtime_workers_health_json_with_inventory;
use serde_json::{json, Value};
use std::env;

use crate::scheduler::server_accelerator_inventory;
use crate::state::SharedState;

/// Model-load progress for the chat UI's loading bar. Reflects the daemon's
/// per-layer weight-load stream (parsed from its stderr by the daemon adapter).
/// `loading` is true only while actively loading (`current < total`); once
/// weights are in (`current == total`) prefill follows, for which no per-step
/// progress exists (the UI falls back to an indeterminate bar).
pub async fn get_load_progress() -> Json<Value> {
    let (current, total, phase) = hipfire_daemon_adapter::model_load_progress();
    let loading = total > 0 && current < total;
    let fraction = if total > 0 {
        current as f64 / total as f64
    } else {
        0.0
    };
    Json(json!({
        "current": current,
        "total": total,
        "phase": phase,
        "loading": loading,
        "fraction": fraction,
    }))
}

pub async fn get_health(state: State<SharedState>) -> Json<Value> {
    let loaded = {
        let loaded = state.loaded_model_path.lock().await;
        loaded.clone()
    };
    let diffusion = diffusion_health_payload(&state).await;
    let active_model = loaded
        .clone()
        .or_else(|| diffusion_active_model(&diffusion));
    let scheduler_resources = scheduler_resource_health_payload(&state).await;
    let prefill_queue_size = state.prefill_scheduler.lock().await.size();
    let selected_prefill_requests = state.selected_prefill_requests.lock().await.len();
    let accelerator_inventory = server_accelerator_inventory(&state).await;
    let runtime_workers = runtime_workers_health_payload(&state, &accelerator_inventory).await;
    let scheduler_env = scheduler_env_from_process();
    let mut prefill_batch = server_prefill_batch_health_json(&scheduler_env);
    if let Some(obj) = prefill_batch.as_object_mut() {
        obj.insert("queue_size".to_string(), json!(prefill_queue_size));
        obj.insert("queued".to_string(), json!(prefill_queue_size));
        obj.insert(
            "selected_pending_dispatch".to_string(),
            json!(selected_prefill_requests),
        );
        if prefill_queue_size > 0 || selected_prefill_requests > 0 {
            obj.insert(
                "runtime_dispatch_skipped_reason".to_string(),
                json!("rust_server_requests_waiting_for_serial_daemon_dispatch"),
            );
        }
    }
    Json(json!({
        "status": "ok",
        "version": hipfire_build_info::VERSION,
        "model": loaded,
        "active_model": active_model,
        "diffusion": diffusion,
        "pid": std::process::id(),
        "scheduler_resources": scheduler_resources,
        "prefill_batch": prefill_batch,
        "decode_batch": server_decode_batch_health_json(&scheduler_env),
        "state_cache": server_state_cache_health_json(&scheduler_env),
        "deferred_jobs": crate::deferred_jobs::deferred_jobs_health_json(),
        "runtime_workers": runtime_workers,
        "batches": batch_health_payload(&state).await,
    }))
}

async fn diffusion_health_payload(state: &SharedState) -> Value {
    let cache = state.diffusion_pipelines.lock().await;
    let mut models = cache
        .iter()
        .map(|(path, pipeline)| {
            let summary = pipeline.summary();
            json!({
                "path": path,
                "title": summary.title,
                "model_name": summary.model_name,
                "pipeline": summary.pipeline_class,
                "weight_format": summary.weight_format,
                "max_batch": summary.max_batch,
            })
        })
        .collect::<Vec<_>>();
    models.sort_by(|left, right| {
        left["path"]
            .as_str()
            .unwrap_or_default()
            .cmp(right["path"].as_str().unwrap_or_default())
    });
    json!({
        "loaded": !models.is_empty(),
        "pipeline_count": models.len(),
        "models": models,
    })
}

fn diffusion_active_model(diffusion: &Value) -> Option<String> {
    diffusion
        .get("models")
        .and_then(Value::as_array)
        .and_then(|models| models.first())
        .and_then(|model| model.get("path"))
        .and_then(Value::as_str)
        .map(str::to_string)
}

fn scheduler_env_from_process() -> SchedulerPolicyEnv {
    SchedulerPolicyEnv::from_pairs(env::vars())
}

async fn scheduler_resource_health_payload(state: &SharedState) -> serde_json::Value {
    let cfg = state.config.lock().await.clone();
    let locks = hipfire_daemon_adapter::resource_lock_report(&hipfire_lock::resource_lock_root())
        .into_iter()
        .map(|(resource, path, state)| {
            let (locked, holder) = match state {
                hipfire_daemon_adapter::LockState::Free => (false, String::new()),
                hipfire_daemon_adapter::LockState::Busy(holder) => (true, holder),
            };
            json!({
                "resource": resource,
                "path": path,
                "locked": locked,
                "holder": holder,
            })
        })
        .collect::<Vec<_>>();
    let daemon_resource_status = {
        let mut engine = state.engine.lock().await;
        if let Some(engine) = engine.as_mut() {
            match engine.resource_status().await {
                Ok(status) => Some(status),
                Err(err) => {
                    tracing::warn!("daemon resource_status failed for health route: {err}");
                    None
                }
            }
        } else {
            None
        }
    };
    let mut payload = json!({
        "resource_lock_enabled": cfg.resource_lock_enabled,
        "resource_lock_gpus": cfg.resource_lock_gpus,
        "resource_lock_npus": cfg.resource_lock_npus,
        "resource_lock_wait_ms": cfg.resource_lock_wait_ms,
        "system_memory_budget_bytes": cfg.scheduler_system_memory_budget_bytes,
        "system_memory_headroom_bytes": cfg.scheduler_system_memory_headroom_bytes,
        "vram_budget_bytes": cfg.scheduler_vram_budget_bytes,
        "vram_headroom_bytes": cfg.scheduler_vram_headroom_bytes,
        "model_residency_mode": cfg.model_residency_mode,
        "locks": locks,
    });
    if let Some(status) = daemon_resource_status {
        for key in [
            "system_memory_target_bytes",
            "held_system_memory_placeholder_bytes",
            "resident_system_memory_bytes",
            "vram_target_bytes",
            "held_vram_placeholder_bytes",
            "resident_vram_bytes",
            "resident_workers",
        ] {
            if let Some(value) = status.get(key).cloned() {
                payload[key] = value;
            }
        }
        payload["daemon_resource_status"] = status;
    }
    payload
}

async fn runtime_workers_health_payload(
    state: &SharedState,
    inventory: &AcceleratorInventory,
) -> serde_json::Value {
    let mut engine = state.engine.lock().await;
    if let Some(engine) = engine.as_mut() {
        match engine.list_workers().await {
            Ok(status) => return status,
            Err(err) => {
                tracing::warn!("daemon worker_status failed for health route: {err}");
            }
        }
    }
    runtime_workers_health_json_with_inventory(&[], 0, None, 0, 0, "none", inventory)
}

async fn batch_health_payload(state: &SharedState) -> serde_json::Value {
    let batches = state.batches.lock().await;
    let total = batches.len();
    let completed = batches
        .values()
        .filter(|batch| batch.status == "completed")
        .count();
    let failed = batches
        .values()
        .filter(|batch| batch.status == "failed")
        .count();
    let cancelled = batches
        .values()
        .filter(|batch| batch.status == "cancelled")
        .count();
    let queued = batches
        .values()
        .filter(|batch| {
            matches!(
                batch.status.as_str(),
                "validating" | "in_progress" | "finalizing"
            )
        })
        .count();
    json!({
        "enabled": true,
        "queued": queued,
        "selected": queued,
        "total": total,
        "failed": failed,
        "cancelled": cancelled,
        "completed": completed,
        "completion_window_supported": true,
        "supported_endpoints": [
            "/v1/chat/completions",
            "/v1/embeddings",
            "/v1/rerank",
            "/v1/responses"
        ],
        "execution_mode": "serial_fallback",
        "last_fallback_reason": "daemon_serialized_request_path",
        "batch_capability": "supported",
        "batch_capability_reason": "rust_axum_batch_control_plane",
        "selected_batch_execution_mode": "serial_fallback",
        "fallback_reason": "generate_batch_prefill_not_used_for_file_batches",
        "runtime_dispatch_skipped_reason": "batch_jobs_execute_via_blocking_routes",
        "unsupported_mode_hits_total": 0,
        "validation_errors_total": failed,
        "streaming_rejections_total": 0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::extract::State;
    use hipfire_diffusion::{
        DiffusionBatchMetadata, DiffusionHfqMetadata, DiffusionPipeline, DiffusionPipelineMetadata,
        DiffusionQuantizationMetadata, DiffusionTokenizerMetadata, DIFFUSION_ARTIFACT_KIND,
        DIFFUSION_SCHEMA_VERSION, HFQ_ARCH_DIFFUSION,
    };
    use hipfire_runtime::hfq::{write_hfqm_package_mem, HfqMemTensor};
    use std::collections::BTreeMap;
    use std::sync::Arc;

    #[test]
    fn health_route_uses_disabled_shared_scheduler_payloads() {
        let runtime_workers = runtime_workers_health_json_with_inventory(
            &[],
            0,
            None,
            0,
            0,
            "none",
            &AcceleratorInventory::not_probed(),
        );
        let payload = json!({
            "prefill_batch": server_prefill_batch_health_json(&SchedulerPolicyEnv::empty()),
            "decode_batch": server_decode_batch_health_json(&SchedulerPolicyEnv::empty()),
            "state_cache": server_state_cache_health_json(&SchedulerPolicyEnv::empty()),
            "deferred_jobs": crate::deferred_jobs::deferred_jobs_health_json(),
            "runtime_workers": runtime_workers,
            "batches": json!({ "enabled": true }),
        });

        assert_eq!(payload["prefill_batch"], json!({ "enabled": false }));
        assert_eq!(payload["decode_batch"], json!({ "enabled": false }));
        assert_eq!(payload["state_cache"], json!({ "enabled": false }));
        assert_eq!(
            payload["deferred_jobs"]["execution_mode"],
            "startup_sequential"
        );
        assert_eq!(payload["runtime_workers"]["resident_workers"], 0);
        assert_eq!(payload["runtime_workers"]["state_arena_backend"], "none");
        assert_eq!(
            payload["runtime_workers"]["accelerator_inventory"]["source"],
            "not_probed"
        );
        assert_eq!(
            payload["runtime_workers"]["accelerator_inventory"]["device_count"],
            0
        );
        assert_eq!(payload["runtime_workers"]["workers"], json!([]));
        assert_eq!(payload["batches"], json!({ "enabled": true }));
    }

    #[test]
    fn health_runtime_workers_can_embed_daemon_inventory() {
        let inventory = AcceleratorInventory::from_devices(
            "daemon",
            vec![hipfire_model::AcceleratorDeviceInfo::hip(
                "0",
                0,
                Some("gfx1201".to_string()),
                Some(24_000_000_000),
                Some(false),
                Some("HIP 6.4".to_string()),
            )],
        );
        let payload =
            runtime_workers_health_json_with_inventory(&[], 0, None, 0, 0, "none", &inventory);

        assert_eq!(payload["accelerator_inventory"]["source"], "daemon");
        assert_eq!(payload["accelerator_inventory"]["device_count"], 1);
        assert_eq!(
            payload["accelerator_inventory"]["devices"][0]["device_class"],
            "discrete"
        );
    }

    #[tokio::test]
    async fn health_reports_cached_diffusion_pipeline_as_active_model() {
        let dir = std::env::temp_dir().join(format!(
            "hipfire-health-diffusion-test-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let hfq_path = dir.join("metadata-only-diffusion.hfq");
        write_metadata_only_diffusion_hfq(&hfq_path);

        let state = crate::AppState::new(hipfire_config::HipfireConfig::default());
        let pipeline = Arc::new(DiffusionPipeline::open_hfq(&hfq_path).unwrap());
        state
            .diffusion_pipelines
            .lock()
            .await
            .insert(hfq_path.clone(), pipeline);

        let Json(payload) = get_health(State(state)).await;

        assert_eq!(payload["model"], Value::Null);
        assert_eq!(
            payload["active_model"].as_str().unwrap(),
            hfq_path.to_string_lossy()
        );
        assert_eq!(payload["diffusion"]["loaded"], true);
        assert_eq!(payload["diffusion"]["pipeline_count"], 1);
        assert_eq!(
            payload["diffusion"]["models"][0]["model_name"],
            "metadata-only-diffusion"
        );
        assert_eq!(
            payload["diffusion"]["models"][0]["pipeline"],
            "StableDiffusionPipeline"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn write_metadata_only_diffusion_hfq(path: &std::path::Path) {
        let metadata = DiffusionHfqMetadata {
            artifact_kind: DIFFUSION_ARTIFACT_KIND.to_string(),
            schema_version: DIFFUSION_SCHEMA_VERSION,
            pipeline: DiffusionPipelineMetadata {
                class_name: "StableDiffusionPipeline".to_string(),
                source: "/tmp/metadata-only-diffusion".to_string(),
                model_name: "metadata-only-diffusion".to_string(),
                latent_channels: Some(4),
                latent_height: Some(64),
                latent_width: Some(64),
                supported_widths: vec![512],
                supported_heights: vec![512],
            },
            tokenizer: DiffusionTokenizerMetadata::default(),
            tokenizer_2: None,
            batch: DiffusionBatchMetadata {
                max_batch: 1,
                batched_runtime: true,
            },
            quantization: DiffusionQuantizationMetadata {
                weight_format: "metadata-only".to_string(),
                activation_format: "fp16".to_string(),
                tensor_roles_version: 1,
            },
            components: BTreeMap::new(),
        };
        let tensors: Vec<HfqMemTensor> = Vec::new();
        write_hfqm_package_mem(
            path,
            HFQ_ARCH_DIFFUSION,
            &serde_json::to_string(&metadata).unwrap(),
            &tensors,
        )
        .unwrap();
    }
}
