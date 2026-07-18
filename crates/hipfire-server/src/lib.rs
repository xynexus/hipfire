#![allow(
    clippy::explicit_counter_loop,
    clippy::field_reassign_with_default,
    clippy::too_many_arguments
)]

pub mod access_admin;
pub mod accounting;
pub mod admin_ui;
pub mod api_auth;
pub mod auth;
pub mod batch_runner;
pub mod deferred_jobs;
pub mod model;
pub mod routes;
pub mod scheduler;
pub mod state;
pub mod telemetry;

pub use state::{AppState, SharedState};

use std::collections::BTreeMap;

use axum::{
    http::HeaderValue,
    middleware,
    routing::{get, post},
    Router,
};
use hipfire_config::{HipfireConfig, LoadedConfig};
use hipfire_generate::{GenerateTextRequest, GenerationSamplingPolicy};
use tower_http::cors::{AllowOrigin, Any, CorsLayer};

/// Build a CORS layer from the configured allowlist.
///
/// - empty list -> `None` (no CORS layer; same-origin requests only)
/// - `["*"]`    -> allow any origin
/// - otherwise  -> explicit origin allowlist
fn cors_layer(allowed_origins: &[String]) -> Option<CorsLayer> {
    if allowed_origins.is_empty() {
        return None;
    }
    let base = CorsLayer::new().allow_methods(Any).allow_headers(Any);
    if allowed_origins.iter().any(|origin| origin == "*") {
        return Some(base.allow_origin(Any));
    }
    let origins: Vec<HeaderValue> = allowed_origins
        .iter()
        .filter_map(|origin| origin.parse::<HeaderValue>().ok())
        .collect();
    Some(base.allow_origin(AllowOrigin::list(origins)))
}

pub fn build_router(state: SharedState, cors_allowed_origins: &[String]) -> Router {
    // Gated admin data endpoints: require a valid session cookie or the local
    // bearer secret (see `auth::admin_gate`). The `/admin` shell and the
    // login/logout endpoints below stay ungated so the page can load and the
    // user can authenticate.
    let admin_data = Router::new()
        .route(
            "/admin/config/schema",
            get(routes::admin::get_config_schema),
        )
        .route(
            "/admin/config/resolved",
            get(routes::admin::get_resolved_config),
        )
        .route(
            "/admin/config/editor",
            get(routes::admin::get_config_editor).patch(routes::admin::patch_config_editor),
        )
        .route(
            "/admin/diagnostics",
            get(routes::admin::get_admin_diagnostics),
        )
        .route("/admin/logs", get(routes::admin::get_admin_logs))
        .route("/admin/stats", get(routes::admin::get_admin_stats))
        .route(
            "/admin/access/users",
            get(access_admin::list_users).post(access_admin::create_user),
        )
        .route(
            "/admin/access/users/{id}",
            get(access_admin::get_user).patch(access_admin::patch_user),
        )
        .route(
            "/admin/access/users/{id}/tokens",
            get(access_admin::list_user_tokens).post(access_admin::create_token),
        )
        .route(
            "/admin/access/tokens/{id}",
            axum::routing::delete(access_admin::revoke_token),
        )
        .route("/admin/access/usage", get(access_admin::get_usage))
        .route(
            "/admin/access/rate-limits",
            get(access_admin::get_rate_limits),
        )
        .route("/admin/access/audit", get(access_admin::get_audit))
        .route(
            "/admin/runtime/reset",
            post(routes::admin::post_runtime_reset),
        )
        .route(
            "/admin/runtime/unload-worker",
            post(routes::admin::post_runtime_unload_worker),
        )
        .route(
            "/admin/models/registry",
            get(routes::models::get_model_registry),
        )
        .route(
            "/admin/training/runs",
            get(routes::training::list_training_runs_route),
        )
        .route(
            "/admin/training/runs/{id}",
            get(routes::training::get_training_run),
        )
        .route(
            "/admin/training/runs/{id}/events",
            get(routes::training::get_training_run_events),
        )
        .route_layer(middleware::from_fn_with_state(
            state.clone(),
            auth::admin_gate,
        ))
        .route_layer(middleware::from_fn(auth::admin_mutation_same_origin));

    let router = Router::new()
        .route("/", get(routes::chat_ui::get_chat_index))
        .route("/chat", get(routes::chat_ui::get_chat_index))
        .route("/chat/", get(routes::chat_ui::get_chat_index))
        .route("/chat/{*path}", get(routes::chat_ui::get_chat_asset))
        .route("/health", get(routes::health::get_health))
        .route("/load-progress", get(routes::health::get_load_progress))
        .route("/admin", get(routes::admin::get_admin_index))
        .route("/admin/", get(routes::admin::get_admin_index))
        .route("/admin/login", post(auth::login))
        .route("/admin/logout", post(auth::logout))
        // New Leptos console (transitional path; shell + assets are not secret,
        // the data endpoints it calls stay gated).
        .route("/admin/ui", get(admin_ui::index))
        .route("/admin/ui/", get(admin_ui::index))
        .route("/admin/ui/{*path}", get(admin_ui::asset))
        .merge(admin_data)
        .route("/v1/models", get(routes::models::get_models))
        .route(
            "/v1/files",
            get(routes::files::list_files).post(routes::files::create_file),
        )
        .route(
            "/v1/files/{id}",
            get(routes::files::get_file).delete(routes::files::delete_file),
        )
        .route(
            "/v1/files/{id}/content",
            get(routes::files::get_file_content),
        )
        .route(
            "/v1/batches",
            get(routes::batches::list_batches).post(routes::batches::create_batch),
        )
        .route("/v1/batches/{id}", get(routes::batches::get_batch))
        .route(
            "/v1/batches/{id}/cancel",
            post(routes::batches::cancel_batch),
        )
        .route(
            "/v1/chat/completions",
            post(routes::chat::post_chat_completions),
        )
        .route("/v1/embeddings", post(routes::embeddings::post_embeddings))
        .route("/v1/rerank", post(routes::embeddings::post_rerank))
        .route("/v1/responses", post(routes::responses::post_responses))
        .route("/sdapi/v1/txt2img", post(routes::sdapi::post_txt2img))
        .route("/sdapi/v1/img2img", post(routes::sdapi::post_img2img))
        .route(
            "/sdapi/v1/extra-single-image",
            post(routes::sdapi::post_extra_single_image),
        )
        .route(
            "/sdapi/v1/extra-batch-images",
            post(routes::sdapi::post_extra_batch_images),
        )
        .route("/sdapi/v1/png-info", post(routes::sdapi::post_png_info))
        .route("/sdapi/v1/progress", get(routes::sdapi::get_progress))
        .route(
            "/sdapi/v1/interrogate",
            post(routes::sdapi::post_interrogate),
        )
        .route("/sdapi/v1/interrupt", post(routes::sdapi::post_interrupt))
        .route("/sdapi/v1/skip", post(routes::sdapi::post_skip))
        .route("/sdapi/v1/options", get(routes::sdapi::get_options))
        .route("/sdapi/v1/options", post(routes::sdapi::post_options))
        .route("/sdapi/v1/memory", get(routes::sdapi::get_memory))
        .route("/sdapi/v1/cmd-flags", get(routes::sdapi::get_cmd_flags))
        .route("/sdapi/v1/samplers", get(routes::sdapi::get_samplers))
        .route("/sdapi/v1/schedulers", get(routes::sdapi::get_schedulers))
        .route("/sdapi/v1/upscalers", get(routes::sdapi::get_upscalers))
        .route(
            "/sdapi/v1/latent-upscale-modes",
            get(routes::sdapi::get_latent_upscale_modes),
        )
        .route("/sdapi/v1/sd-models", get(routes::sdapi::get_sd_models))
        .route("/sdapi/v1/sd-vae", get(routes::sdapi::get_sd_vae))
        .route(
            "/sdapi/v1/hypernetworks",
            get(routes::sdapi::get_hypernetworks),
        )
        .route(
            "/sdapi/v1/face-restorers",
            get(routes::sdapi::get_face_restorers),
        )
        .route(
            "/sdapi/v1/realesrgan-models",
            get(routes::sdapi::get_realesrgan_models),
        )
        .route(
            "/sdapi/v1/prompt-styles",
            get(routes::sdapi::get_prompt_styles),
        )
        .route("/sdapi/v1/loras", get(routes::sdapi::get_loras))
        .route(
            "/sdapi/v1/refresh-loras",
            post(routes::sdapi::post_control_noop),
        )
        .route("/sdapi/v1/embeddings", get(routes::sdapi::get_embeddings))
        .route(
            "/sdapi/v1/refresh-embeddings",
            post(routes::sdapi::post_control_noop),
        )
        .route(
            "/sdapi/v1/refresh-checkpoints",
            post(routes::sdapi::post_control_noop),
        )
        .route(
            "/sdapi/v1/reload-checkpoint",
            post(routes::sdapi::post_reload_checkpoint),
        )
        .route(
            "/sdapi/v1/unload-checkpoint",
            post(routes::sdapi::post_unload_checkpoint),
        )
        .route(
            "/sdapi/v1/refresh-vae",
            post(routes::sdapi::post_control_noop),
        )
        .route(
            "/sdapi/v1/create/embedding",
            post(routes::sdapi::post_unsupported_training_endpoint),
        )
        .route(
            "/sdapi/v1/create/hypernetwork",
            post(routes::sdapi::post_unsupported_training_endpoint),
        )
        .route(
            "/sdapi/v1/train/embedding",
            post(routes::sdapi::post_unsupported_training_endpoint),
        )
        .route(
            "/sdapi/v1/train/hypernetwork",
            post(routes::sdapi::post_unsupported_training_endpoint),
        )
        .route("/sdapi/v1/scripts", get(routes::sdapi::get_scripts))
        .route("/sdapi/v1/script-info", get(routes::sdapi::get_script_info))
        .route("/sdapi/v1/extensions", get(routes::sdapi::get_extensions))
        .route(
            "/sdapi/v1/server-kill",
            post(routes::sdapi::post_server_kill_noop),
        )
        .route(
            "/sdapi/v1/server-restart",
            post(routes::sdapi::post_server_restart_noop),
        )
        .route(
            "/sdapi/v1/server-stop",
            post(routes::sdapi::post_server_stop_noop),
        );
    let router = router.route_layer(middleware::from_fn_with_state(
        state.clone(),
        api_auth::api_gate,
    ));
    let router = match cors_layer(cors_allowed_origins) {
        Some(cors) => router.layer(cors),
        None => router,
    };
    router.with_state(state)
}

pub async fn serve(config: HipfireConfig) -> anyhow::Result<()> {
    serve_loaded(LoadedConfig::from_config(config)).await
}

pub async fn serve_loaded(config: LoadedConfig) -> anyhow::Result<()> {
    api_auth::validate_api_auth_config(&config.config).map_err(anyhow::Error::msg)?;
    let addr = format!("{}:{}", config.config.host, config.config.port);
    let cors_allowed_origins = config.config.cors_allowed_origins.clone();
    let state = AppState::new_loaded(config);
    state.access.ensure_ready().map_err(anyhow::Error::msg)?;

    spawn_daemon_for_serving(&state).await?;

    // HIP/ROCm-first: detect the GPU once at server launch so diffusion requests
    // target the same resolved device (CPU reference only via env opt-in). This
    // runs after daemon startup so the daemon owns resource leases first.
    state.resolve_diffusion_runtime_default();
    deferred_jobs::spawn_deferred_job_runner(state.clone());

    // Continuous-batching runner: owns the engine and fuses concurrent same-model
    // requests into batched prefill + decode. Opt-in; when the flag is off the
    // legacy per-request path (engine.lock in chat.rs) is used unchanged.
    if hipfire_scheduler::server_prefill_batch_enabled(
        &hipfire_scheduler::SchedulerPolicyEnv::from_pairs(std::env::vars()),
    ) {
        batch_runner::spawn_batch_runner(state.clone());
        tracing::info!("continuous-batching runner enabled (HIPFIRE_SERVER_PREFILL_BATCH)");
    }

    let app = build_router(state.clone(), &cors_allowed_origins);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    tracing::info!("hipfire listening on http://{addr}");
    spawn_deferred_prewarm(state.clone());
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal(state))
        .await?;
    Ok(())
}

async fn spawn_daemon_for_serving(state: &SharedState) -> anyhow::Result<()> {
    let cfg = state.config.lock().await.clone();
    apply_daemon_startup_env(&cfg);
    let bin = hipfire_daemon_adapter::find_daemon_bin_or_error()?;
    let mut engine = hipfire_daemon_adapter::DaemonEngine::spawn(&bin).await?;
    engine.ping().await?;
    *state.engine.lock().await = Some(engine);
    Ok(())
}

fn apply_daemon_startup_env(cfg: &HipfireConfig) {
    std::env::set_var(
        "HIPFIRE_RESOURCE_LOCK",
        if cfg.resource_lock_enabled { "1" } else { "0" },
    );
    std::env::set_var(
        "HIPFIRE_RESOURCE_LOCK_WAIT_MS",
        cfg.resource_lock_wait_ms.to_string(),
    );
    std::env::set_var(
        "HIPFIRE_SCHEDULER_SYSTEM_MEMORY_BUDGET_BYTES",
        cfg.scheduler_system_memory_budget_bytes.to_string(),
    );
    std::env::set_var(
        "HIPFIRE_SCHEDULER_SYSTEM_MEMORY_HEADROOM_BYTES",
        cfg.scheduler_system_memory_headroom_bytes.to_string(),
    );
    std::env::set_var(
        "HIPFIRE_SCHEDULER_VRAM_BUDGET_BYTES",
        cfg.scheduler_vram_budget_bytes.to_string(),
    );
    std::env::set_var(
        "HIPFIRE_SCHEDULER_VRAM_HEADROOM_BYTES",
        cfg.scheduler_vram_headroom_bytes.to_string(),
    );
    apply_resource_list_env("HIPFIRE_DEVICES", &cfg.resource_lock_gpus, true);
    apply_resource_list_env("HIPFIRE_RESOURCE_LOCK_NPUS", &cfg.resource_lock_npus, false);
}

fn apply_resource_list_env(key: &str, values: &[String], auto_removes: bool) {
    let values = values
        .iter()
        .map(|value| value.trim())
        .filter(|value| !value.is_empty())
        .collect::<Vec<_>>();
    if values.is_empty() {
        std::env::remove_var(key);
    } else if values.len() == 1 && values[0].eq_ignore_ascii_case("auto") {
        if auto_removes {
            std::env::remove_var(key);
        } else {
            std::env::set_var(key, "1");
        }
    } else {
        std::env::set_var(key, values.join(","));
    }
}

async fn clear_loaded_model_state(state: &SharedState) {
    state.loaded_models.lock().await.clear();
    *state.loaded_model_path.lock().await = None;
    *state.loaded_model_cache_capable.lock().await = None;
    *state.loaded_model_max_seq.lock().await = None;
}

async fn clear_diffusion_pipeline_cache(state: &SharedState) -> usize {
    let mut cache = state.diffusion_pipelines.lock().await;
    let count = cache.len();
    cache.clear();
    count
}

async fn shutdown_signal(state: SharedState) {
    #[cfg(unix)]
    let terminate = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut signal) => {
                signal.recv().await;
            }
            Err(e) => {
                tracing::warn!("failed to install SIGTERM handler: {e}");
                std::future::pending::<()>().await;
            }
        }
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = tokio::signal::ctrl_c() => {}
        _ = terminate => {}
    }

    tracing::info!("shutdown signal received; unloading daemon and diffusion pipelines");
    let diffusion_count = clear_diffusion_pipeline_cache(&state).await;
    if diffusion_count > 0 {
        tracing::info!("unloaded {diffusion_count} diffusion pipeline(s) during shutdown");
    }
    let mut engine = state.engine.lock().await;
    if let Some(mut engine) = engine.take() {
        if let Err(e) = engine.unload().await {
            tracing::warn!("daemon unload during shutdown failed: {e}");
        }
    }
    clear_loaded_model_state(&state).await;
}

fn spawn_deferred_prewarm(state: SharedState) {
    tokio::spawn(async move {
        prewarm_configured_models(&state).await;
    });
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct PrewarmTarget {
    model: String,
    priority: u32,
}

fn prewarm_targets_from_config(cfg: &HipfireConfig) -> Vec<PrewarmTarget> {
    let mut targets = BTreeMap::<String, u32>::new();
    if cfg.prewarm_priority > 0 {
        if let Some(model) = cfg
            .default_model
            .as_deref()
            .filter(|model| !model.is_empty())
        {
            targets.insert(model.to_string(), cfg.prewarm_priority);
        }
    }
    for model in cfg.model_overrides.keys() {
        let resolved = cfg.resolve_for_model(model);
        if resolved.prewarm_priority > 0 {
            targets
                .entry(model.clone())
                .and_modify(|priority| *priority = (*priority).max(resolved.prewarm_priority))
                .or_insert(resolved.prewarm_priority);
        }
    }
    let mut targets = targets
        .into_iter()
        .map(|(model, priority)| PrewarmTarget { model, priority })
        .collect::<Vec<_>>();
    targets.sort_by(|a, b| {
        b.priority
            .cmp(&a.priority)
            .then_with(|| a.model.cmp(&b.model))
    });
    targets
}

async fn prewarm_configured_models(state: &SharedState) {
    let targets = {
        let cfg = state.config.lock().await;
        prewarm_targets_from_config(&cfg)
    };
    for target in targets {
        prewarm_model(state, &target).await;
    }
}

async fn prewarm_model(state: &SharedState, target: &PrewarmTarget) {
    let model = &target.model;
    tracing::info!(model = %model, priority = target.priority, "pre-warming model");
    if prewarm_diffusion_model(state, model).await {
        return;
    }

    let required_max_seq = {
        let cfg = state.config.lock().await;
        cfg.resolve_for_model(model).max_seq
    };
    match routes::chat::ensure_model_loaded(state, model, required_max_seq).await {
        Ok(loaded) => {
            let mut engine_guard = state.engine.lock().await;
            let Some(engine) = engine_guard.as_mut() else {
                tracing::warn!("pre-warm loaded model but daemon engine is unavailable");
                return;
            };
            let req = GenerateTextRequest::from_prompt(
                "warmup".to_string(),
                "Hi",
                GenerationSamplingPolicy::greedy(1),
            )
            .with_worker_key_id(loaded.worker_key_id);
            if let Err(e) = engine.generate(req).await {
                tracing::warn!(
                    "pre-warm generate failed: {e}; first request will continue normally"
                );
                return;
            }
            if let Err(e) = engine.reset().await {
                tracing::warn!(
                    "pre-warm reset failed: {e}; first request will reset before generate"
                );
                return;
            }
            tracing::info!(model = %model, "warm-up complete");
        }
        Err(e) => {
            tracing::warn!(model = %model, "pre-warm load failed: {e}; will load on first request");
        }
    }
}

async fn prewarm_diffusion_model(state: &SharedState, model: &str) -> bool {
    let Some(path) = routes::sdapi::resolve_diffusion_hfq_candidate(
        model,
        &state.models_dir,
        state.models_network_dir.as_deref(),
    ) else {
        return false;
    };

    match routes::sdapi::cached_diffusion_pipeline(state, path.clone()).await {
        Ok(pipeline) => {
            let summary = pipeline.summary();
            tracing::info!(
                model = %summary.model_name,
                pipeline = %summary.pipeline_class,
                path = %path.display(),
                "diffusion warm-up complete"
            );
        }
        Err(e) => {
            tracing::warn!(
                "diffusion pre-warm failed for {}: {e}; first SDAPI request will retry",
                path.display()
            );
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        body::Body,
        http::{Method, Request},
    };
    use hipfire_diffusion::{
        DiffusionBatchMetadata, DiffusionHfqMetadata, DiffusionPipelineMetadata,
        DiffusionQuantizationMetadata, DiffusionTokenizerMetadata, DIFFUSION_ARTIFACT_KIND,
        DIFFUSION_SCHEMA_VERSION, HFQ_ARCH_DIFFUSION,
    };
    use hipfire_runtime::hfq::{write_hfqm_package_mem, HfqMemTensor};
    use serde_json::{json, Value};
    use std::collections::BTreeMap;
    use std::path::Path;
    use std::sync::{Mutex, OnceLock};
    use tower::ServiceExt;

    fn daemon_startup_env_test_guard() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(())).lock().unwrap()
    }

    #[test]
    fn cors_layer_disabled_when_no_origins() {
        assert!(cors_layer(&[]).is_none());
    }

    #[test]
    fn cors_layer_present_for_wildcard_and_allowlist() {
        assert!(cors_layer(&["*".to_string()]).is_some());
        assert!(cors_layer(&["http://localhost:8080".to_string()]).is_some());
    }

    #[tokio::test]
    async fn sdapi_server_command_routes_are_registered_as_safe_noops() {
        let app = build_router(AppState::new(HipfireConfig::default()), &[]);

        for route in [
            "/sdapi/v1/server-kill",
            "/sdapi/v1/server-restart",
            "/sdapi/v1/server-stop",
        ] {
            let response = app
                .clone()
                .oneshot(
                    Request::builder()
                        .method(Method::POST)
                        .uri(route)
                        .header("content-type", "application/json")
                        .body(Body::from("{}"))
                        .unwrap(),
                )
                .await
                .unwrap();

            assert_eq!(response.status(), axum::http::StatusCode::OK);
            let body = axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap();
            let body: Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(body["success"], false);
            assert_eq!(
                body["command"].as_str().unwrap(),
                route.trim_start_matches("/sdapi/v1/")
            );
            assert_eq!(body["server_command_supported"], false);
        }
    }

    #[tokio::test]
    async fn prewarm_priority_routes_diffusion_hfq_to_diffusion_cache() {
        let dir = std::env::temp_dir().join(format!(
            "hipfire-diffusion-prewarm-test-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let hfq_path = dir.join("metadata-only-diffusion.hfq");
        write_metadata_only_diffusion_hfq(&hfq_path);

        let mut config = HipfireConfig::default();
        config.models_network_dir = Some(dir.to_string_lossy().into_owned());
        let model = hfq_path.file_name().unwrap().to_string_lossy().into_owned();
        config
            .model_overrides
            .insert(model.clone(), json!({"prewarm_priority": 10}));
        let state = AppState::new(config);

        prewarm_configured_models(&state).await;

        assert!(state.engine.lock().await.is_none());
        assert!(state.loaded_model_path.lock().await.is_none());
        assert_eq!(state.diffusion_pipelines.lock().await.len(), 1);
    }

    #[test]
    fn default_model_does_not_prewarm_without_priority() {
        let config = HipfireConfig {
            default_model: Some("qwen".to_string()),
            ..HipfireConfig::default()
        };

        assert!(prewarm_targets_from_config(&config).is_empty());
    }

    #[test]
    fn prewarm_targets_sort_by_priority_and_include_multiple_models() {
        let mut config = HipfireConfig {
            default_model: Some("default-model".to_string()),
            prewarm_priority: 5,
            ..HipfireConfig::default()
        };
        config
            .model_overrides
            .insert("low".to_string(), json!({"prewarm_priority": 1}));
        config
            .model_overrides
            .insert("high".to_string(), json!({"prewarm_priority": 20}));

        let targets = prewarm_targets_from_config(&config);

        assert_eq!(
            targets,
            vec![
                PrewarmTarget {
                    model: "high".to_string(),
                    priority: 20,
                },
                PrewarmTarget {
                    model: "default-model".to_string(),
                    priority: 5,
                },
                PrewarmTarget {
                    model: "low".to_string(),
                    priority: 1,
                },
            ]
        );
    }

    #[test]
    fn daemon_startup_env_maps_resource_lock_config() {
        let _guard = daemon_startup_env_test_guard();
        let mut config = HipfireConfig::default();
        config.resource_lock_gpus = vec!["0".to_string(), "2".to_string()];
        config.resource_lock_npus = vec!["auto".to_string()];
        config.resource_lock_wait_ms = 250;
        config.scheduler_system_memory_budget_bytes = 1024;
        config.scheduler_system_memory_headroom_bytes = 128;
        config.scheduler_vram_budget_bytes = 2048;
        config.scheduler_vram_headroom_bytes = 256;

        apply_daemon_startup_env(&config);

        assert_eq!(std::env::var("HIPFIRE_RESOURCE_LOCK").unwrap(), "1");
        assert_eq!(
            std::env::var("HIPFIRE_RESOURCE_LOCK_WAIT_MS").unwrap(),
            "250"
        );
        assert_eq!(std::env::var("HIPFIRE_DEVICES").unwrap(), "0,2");
        assert_eq!(std::env::var("HIPFIRE_RESOURCE_LOCK_NPUS").unwrap(), "1");
        assert_eq!(
            std::env::var("HIPFIRE_SCHEDULER_SYSTEM_MEMORY_BUDGET_BYTES").unwrap(),
            "1024"
        );
        assert_eq!(
            std::env::var("HIPFIRE_SCHEDULER_SYSTEM_MEMORY_HEADROOM_BYTES").unwrap(),
            "128"
        );
        assert_eq!(
            std::env::var("HIPFIRE_SCHEDULER_VRAM_BUDGET_BYTES").unwrap(),
            "2048"
        );
        assert_eq!(
            std::env::var("HIPFIRE_SCHEDULER_VRAM_HEADROOM_BYTES").unwrap(),
            "256"
        );
    }

    #[test]
    fn daemon_startup_env_auto_gpu_uses_daemon_default_resolution() {
        let _guard = daemon_startup_env_test_guard();
        let mut config = HipfireConfig::default();
        config.resource_lock_enabled = false;
        config.resource_lock_gpus = vec!["auto".to_string()];
        config.resource_lock_npus = Vec::new();

        apply_daemon_startup_env(&config);

        assert_eq!(std::env::var("HIPFIRE_RESOURCE_LOCK").unwrap(), "0");
        assert!(std::env::var("HIPFIRE_DEVICES").is_err());
        assert!(std::env::var("HIPFIRE_RESOURCE_LOCK_NPUS").is_err());
    }

    fn write_metadata_only_diffusion_hfq(path: &Path) {
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
                ..DiffusionPipelineMetadata::default()
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
