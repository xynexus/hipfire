use axum::{
    extract::{Query, State},
    http::StatusCode,
    response::{Html, IntoResponse, Json},
};
use hipfire_config::{
    apply_config_edit, build_config_editor_snapshot_from_paths, config_schema,
    configured_models_dir, ConfigEditRequest, ConfigEditorPaths, LoadedConfig,
};
use serde::Deserialize;
use serde_json::{json, Value};
use std::{fs, path::Path};

use crate::SharedState;

pub async fn get_admin_index() -> Html<&'static str> {
    Html(ADMIN_INDEX_HTML)
}

pub async fn get_admin_diagnostics(State(state): State<SharedState>) -> Json<Value> {
    let loaded = state.loaded_config.lock().await;
    let root = hipfire_config::hipfire_dir();
    Json(json!({
        "hipfire_dir": root.display().to_string(),
        "config_path": loaded.config_path.display().to_string(),
        "host_config_path": loaded.host_config_path.display().to_string(),
        "config_read_error": loaded.read_error,
        "host_config_read_error": loaded.host_read_error,
        "paths": path_statuses(&root, &configured_models_dir(&loaded.config)),
        "binaries": binary_statuses(&root.join("bin")),
        "kernel_caches": kernel_cache_statuses(&root.join("kernels")),
        "resource_locks": resource_lock_statuses(&hipfire_lock::resource_lock_root()),
        "logs": log_file_statuses(&root),
    }))
}

#[derive(Debug, Deserialize)]
pub struct AdminLogsQuery {
    pub lines: Option<usize>,
}

pub async fn get_admin_logs(Query(query): Query<AdminLogsQuery>) -> Json<Value> {
    let root = hipfire_config::hipfire_dir();
    let lines = query.lines.unwrap_or(120).clamp(1, 1000);
    Json(json!({
        "lines": lines,
        "logs": log_tails(&root, lines),
    }))
}

/// Live host/GPU telemetry snapshot for the dashboard (sysfs + `/proc`-backed).
pub async fn get_admin_stats() -> Json<hipfire_admin_types::AdminStats> {
    Json(hipfire_sysinfo::snapshot(now_unix_secs()))
}

pub async fn post_runtime_reset(State(state): State<SharedState>) -> impl IntoResponse {
    let mut engine = state.engine.lock().await;
    match engine.as_mut() {
        Some(engine) => match engine.reset().await {
            Ok(()) => (
                StatusCode::OK,
                Json(json!({
                    "status": "ok",
                    "reset": true,
                })),
            ),
            Err(error) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({
                    "error": {
                        "message": format!("daemon reset failed: {error}"),
                        "type": "runtime_reset_failed",
                    }
                })),
            ),
        },
        None => (
            StatusCode::OK,
            Json(json!({
                "status": "ok",
                "reset": false,
                "reason": "no daemon engine is loaded",
            })),
        ),
    }
}

#[derive(Debug, Deserialize)]
pub struct RuntimeUnloadWorkerRequest {
    pub worker_key_id: String,
}

pub async fn post_runtime_unload_worker(
    State(state): State<SharedState>,
    Json(request): Json<RuntimeUnloadWorkerRequest>,
) -> impl IntoResponse {
    let worker_key_id = request.worker_key_id.trim().to_string();
    if worker_key_id.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({
                "error": {
                    "message": "worker_key_id is required",
                    "type": "invalid_request_error",
                }
            })),
        );
    }

    let mut engine = state.engine.lock().await;
    let Some(engine) = engine.as_mut() else {
        return (
            StatusCode::NOT_FOUND,
            Json(json!({
                "error": {
                    "message": "daemon engine is not running",
                    "type": "daemon_unavailable",
                }
            })),
        );
    };

    match engine.unload_worker(&worker_key_id).await {
        Ok(done) => {
            let active_path_still_loaded = {
                let mut loaded_models = state.loaded_models.lock().await;
                loaded_models.retain(|_, loaded| {
                    loaded.worker_key_id.as_deref() != Some(worker_key_id.as_str())
                });
                state
                    .loaded_model_path
                    .lock()
                    .await
                    .as_ref()
                    .is_some_and(|path| loaded_models.contains_key(path))
            };
            if !active_path_still_loaded {
                *state.loaded_model_path.lock().await = None;
                *state.loaded_model_cache_capable.lock().await = None;
                *state.loaded_model_max_seq.lock().await = None;
            }
            (StatusCode::OK, Json(done))
        }
        Err(error) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({
                "error": {
                    "message": format!("daemon unload_worker failed: {error}"),
                    "type": "runtime_unload_failed",
                }
            })),
        ),
    }
}

fn now_unix_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

pub async fn get_config_schema() -> Json<Value> {
    Json(config_schema_json())
}

#[derive(Debug, Deserialize)]
pub struct ResolvedConfigQuery {
    pub model: Option<String>,
}

pub async fn get_resolved_config(
    State(state): State<SharedState>,
    Query(query): Query<ResolvedConfigQuery>,
) -> Json<Value> {
    let loaded = state.loaded_config.lock().await;
    Json(resolved_config_json(&loaded, query.model.as_deref()))
}

pub async fn get_config_editor(
    State(state): State<SharedState>,
    Query(query): Query<ResolvedConfigQuery>,
) -> Json<Value> {
    let loaded = state.loaded_config.lock().await;
    Json(config_editor_json(&loaded, query.model.as_deref()))
}

pub async fn patch_config_editor(
    State(state): State<SharedState>,
    Json(request): Json<ConfigEditRequest>,
) -> impl IntoResponse {
    let loaded = state.loaded_config.lock().await;
    let paths = ConfigEditorPaths {
        global: loaded.config_path.clone(),
        host: loaded.host_config_path.clone(),
    };
    match apply_config_edit(&paths, &request, &loaded) {
        Ok(snapshot) => (
            StatusCode::OK,
            Json(serde_json::to_value(snapshot).unwrap_or_else(json_error)),
        ),
        Err(error) => (
            StatusCode::BAD_REQUEST,
            Json(json!({
                "error": {
                    "message": error,
                    "type": "invalid_config_edit",
                }
            })),
        ),
    }
}

fn config_schema_json() -> Value {
    serde_json::to_value(config_schema()).unwrap_or_else(|err| {
        json!({
            "error": {
                "message": format!("failed to serialize config schema: {err}"),
                "type": "internal_error"
            }
        })
    })
}

fn config_editor_json(loaded: &LoadedConfig, model: Option<&str>) -> Value {
    let paths = ConfigEditorPaths {
        global: loaded.config_path.clone(),
        host: loaded.host_config_path.clone(),
    };
    serde_json::to_value(build_config_editor_snapshot_from_paths(
        &paths, loaded, model,
    ))
    .unwrap_or_else(json_error)
}

fn json_error(err: serde_json::Error) -> Value {
    json!({
        "error": {
            "message": format!("failed to serialize config editor snapshot: {err}"),
            "type": "internal_error"
        }
    })
}

fn resolved_config_json(loaded: &LoadedConfig, model: Option<&str>) -> Value {
    let (config, layers, resolution, diagnostics) = match model {
        Some(model) => {
            let resolved = loaded.resolve_for_model(model);
            (
                resolved.config,
                resolved.layers,
                resolved.resolution,
                resolved.diagnostics,
            )
        }
        None => (
            loaded.config.clone(),
            loaded.layers.clone(),
            loaded.resolution.clone(),
            loaded.diagnostics.clone(),
        ),
    };
    json!({
        "source": "active_runtime",
        "config_path": loaded.config_path.display().to_string(),
        "host_config_path": loaded.host_config_path.display().to_string(),
        "model": model,
        "read_error": loaded.read_error.clone(),
        "host_read_error": loaded.host_read_error.clone(),
        "diagnostics": diagnostics,
        "layers": layers,
        "resolution": resolution,
        "config": config,
    })
}

fn path_statuses(root: &Path, models_dir: &Path) -> Vec<Value> {
    [
        ("hipfire_dir", root.to_path_buf()),
        ("models", models_dir.to_path_buf()),
        ("config", root.join("config.json")),
        ("host_config", root.join("config.local.json")),
        ("per_model_config", root.join("per_model_config.json")),
        ("training_runs", root.join("training").join("runs")),
        ("logs", root.join("logs")),
        ("kernels", root.join("kernels")),
    ]
    .into_iter()
    .map(|(name, path)| {
        json!({
            "name": name,
            "path": path.display().to_string(),
            "exists": path.exists(),
            "is_dir": path.is_dir(),
            "is_file": path.is_file(),
        })
    })
    .collect()
}

fn binary_statuses(bin_dir: &Path) -> Vec<Value> {
    [
        "hipfire",
        "hipfire-daemon",
        "hipfire-tui",
        "hipfire-eval",
        "hipfire-host-profile",
    ]
    .into_iter()
    .map(|name| {
        let path = bin_dir.join(name);
        json!({
            "name": name,
            "path": path.display().to_string(),
            "exists": path.exists(),
        })
    })
    .collect()
}

fn kernel_cache_statuses(kernel_root: &Path) -> Vec<Value> {
    let Ok(entries) = fs::read_dir(kernel_root) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let arch = entry.file_name().to_string_lossy().to_string();
        let (hsaco, hash) = count_kernel_files(&path);
        out.push(json!({
            "arch": arch,
            "path": path.display().to_string(),
            "hsaco": hsaco,
            "hash": hash,
            "balanced": hsaco == hash,
        }));
    }
    out.sort_by(|a, b| a["arch"].as_str().cmp(&b["arch"].as_str()));
    out
}

fn count_kernel_files(path: &Path) -> (usize, usize) {
    let Ok(entries) = fs::read_dir(path) else {
        return (0, 0);
    };
    let mut hsaco = 0;
    let mut hash = 0;
    for entry in entries.flatten() {
        match entry.path().extension().and_then(|ext| ext.to_str()) {
            Some("hsaco") => hsaco += 1,
            Some("hash") => hash += 1,
            _ => {}
        }
    }
    (hsaco, hash)
}

fn resource_lock_statuses(lock_dir: &Path) -> Vec<Value> {
    // flock(2)-based leases now: report the live kernel lock state (Free/Busy) +
    // holder line for the shared GPU lock and any per-resource files under lock_dir.
    use hipfire_daemon_adapter::LockState;
    hipfire_daemon_adapter::resource_lock_report(lock_dir)
        .into_iter()
        .map(|(name, path, state)| {
            let (held, holder) = match state {
                LockState::Free => (false, String::new()),
                LockState::Busy(h) => (true, h),
            };
            json!({
                "name": name,
                "path": path.display().to_string(),
                "held": held,
                "holder": holder,
            })
        })
        .collect()
}

fn log_file_statuses(root: &Path) -> Vec<Value> {
    candidate_log_files(root)
        .into_iter()
        .map(|path| {
            let metadata = fs::metadata(&path).ok();
            json!({
                "path": path.display().to_string(),
                "exists": metadata.is_some(),
                "bytes": metadata.map(|m| m.len()).unwrap_or(0),
            })
        })
        .collect()
}

fn log_tails(root: &Path, lines: usize) -> Vec<Value> {
    candidate_log_files(root)
        .into_iter()
        .filter(|path| path.is_file())
        .map(|path| {
            json!({
                "path": path.display().to_string(),
                "text": tail_file(&path, lines),
            })
        })
        .collect()
}

fn candidate_log_files(root: &Path) -> Vec<std::path::PathBuf> {
    let mut paths = vec![root.join("serve.log")];
    let logs_dir = root.join("logs");
    if let Ok(entries) = fs::read_dir(&logs_dir) {
        let mut extra = entries
            .flatten()
            .map(|entry| entry.path())
            .filter(|path| path.extension().and_then(|ext| ext.to_str()) == Some("log"))
            .collect::<Vec<_>>();
        extra.sort();
        paths.extend(extra);
    }
    paths
}

fn tail_file(path: &Path, lines: usize) -> String {
    let Ok(raw) = fs::read_to_string(path) else {
        return String::new();
    };
    let mut selected = raw.lines().rev().take(lines).collect::<Vec<_>>();
    selected.reverse();
    selected.join("\n")
}

const ADMIN_INDEX_HTML: &str = r#"<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>hipfire admin console</title>
  <style>
    :root {
      color-scheme: light dark;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --text: #172026;
      --muted: #66717c;
      --line: #d7dde3;
      --accent: #0f766e;
      --accent-2: #7c3aed;
      --warn: #b45309;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    @media (prefers-color-scheme: dark) {
      :root {
        --bg: #111417;
        --panel: #171b20;
        --text: #e7ecef;
        --muted: #9aa5af;
        --line: #29313a;
        --accent: #2dd4bf;
        --accent-2: #a78bfa;
        --warn: #f59e0b;
      }
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-size: 14px;
    }
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 18px 24px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
    }
    h1 {
      margin: 0;
      font-size: 18px;
      font-weight: 650;
      letter-spacing: 0;
    }
    main {
      width: min(1280px, 100%);
      margin: 0 auto;
      padding: 18px 24px 32px;
    }
    .toolbar {
      display: flex;
      flex-wrap: wrap;
      align-items: end;
      gap: 12px;
      padding-bottom: 16px;
    }
    label {
      display: grid;
      gap: 5px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 600;
    }
    input, textarea, select, button {
      height: 34px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--panel);
      color: var(--text);
      font: inherit;
    }
    input, textarea, select {
      width: min(360px, 76vw);
      padding: 0 10px;
    }
    textarea {
      width: 100%;
      min-height: 112px;
      height: auto;
      padding: 10px;
      resize: vertical;
      line-height: 1.4;
    }
    button {
      padding: 0 12px;
      cursor: pointer;
    }
    button:hover { border-color: var(--accent); }
    .inline-actions {
      display: flex;
      gap: 6px;
      align-items: center;
    }
    .inline-actions input, .inline-actions select {
      width: min(180px, 34vw);
    }
    .inline-actions button {
      padding: 0 8px;
    }
    .status {
      margin-left: auto;
      color: var(--muted);
      min-width: 180px;
      text-align: right;
    }
    .summary {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
      margin-bottom: 16px;
    }
    .metric {
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: 8px;
      padding: 10px 12px;
      min-width: 0;
    }
    .metric span {
      display: block;
      color: var(--muted);
      font-size: 12px;
      font-weight: 600;
    }
    .metric strong {
      display: block;
      margin-top: 5px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      font-size: 14px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      background: var(--panel);
      border: 1px solid var(--line);
    }
    th, td {
      padding: 9px 10px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
    }
    th {
      position: sticky;
      top: 0;
      background: var(--panel);
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      z-index: 1;
    }
    td.key { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; color: var(--accent); }
    td.value { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; max-width: 280px; overflow-wrap: anywhere; }
    .source { color: var(--accent-2); }
    .muted { color: var(--muted); }
    .warn { color: var(--warn); }
    .tabs {
      display: flex;
      gap: 8px;
      padding-bottom: 16px;
    }
    .tab {
      min-width: 96px;
    }
    .tab.active {
      border-color: var(--accent);
      color: var(--accent);
      font-weight: 700;
    }
    .panel[hidden] { display: none; }
    .grid {
      display: grid;
      grid-template-columns: minmax(260px, 0.9fr) minmax(340px, 1.1fr);
      gap: 16px;
      align-items: start;
    }
    .section-title {
      margin: 0 0 10px;
      font-size: 14px;
      color: var(--muted);
      font-weight: 700;
    }
    .event-list {
      display: grid;
      gap: 8px;
    }
    .stack {
      display: grid;
      gap: 12px;
    }
    .event {
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: 8px;
      padding: 9px 10px;
      overflow-wrap: anywhere;
    }
    .event strong {
      color: var(--accent);
      font-size: 12px;
    }
    .event code {
      display: block;
      margin-top: 5px;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      color: var(--muted);
      white-space: pre-wrap;
    }
    .chat-transcript {
      min-height: 260px;
      max-height: 560px;
      overflow: auto;
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: 8px;
      padding: 10px;
    }
    pre {
      margin: 0;
      overflow: auto;
      max-height: 520px;
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: 8px;
      padding: 10px;
      color: var(--text);
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
      line-height: 1.45;
    }
    tr.selectable { cursor: pointer; }
    tr.selected { background: color-mix(in srgb, var(--accent) 14%, transparent); }
    @media (max-width: 820px) {
      header, main { padding-left: 14px; padding-right: 14px; }
      .summary { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .grid { grid-template-columns: 1fr; }
      .status { width: 100%; text-align: left; margin-left: 0; }
      table { font-size: 13px; }
      th:nth-child(4), td:nth-child(4), th:nth-child(5), td:nth-child(5) { display: none; }
      .tab { min-width: auto; }
    }
    .overlay {
      position: fixed;
      inset: 0;
      display: flex;
      align-items: center;
      justify-content: center;
      background: color-mix(in srgb, var(--bg) 80%, black);
      z-index: 10;
    }
    .overlay[hidden] { display: none; }
    .login-card {
      width: min(360px, 92vw);
      display: grid;
      gap: 12px;
      padding: 22px;
      border: 1px solid var(--line);
      border-radius: 10px;
      background: var(--panel);
    }
    .login-card h2 { margin: 0; font-size: 16px; }
    .login-card .login-error { color: var(--warn); min-height: 16px; font-size: 12px; }
    #logout { margin-left: 12px; }
  </style>
</head>
<body>
  <header>
    <h1>hipfire admin console</h1>
    <div>
      <span id="status" class="status">connecting</span>
      <button id="logout" type="button" hidden>Log out</button>
    </div>
  </header>
  <div id="login" class="overlay" hidden>
    <form id="login-form" class="login-card">
      <h2>admin sign in</h2>
      <label>User
        <input id="login-user" name="login-user" autocomplete="username" value="admin">
      </label>
      <label>Password
        <input id="login-password" name="login-password" type="password" autocomplete="current-password">
      </label>
      <div id="login-error" class="login-error"></div>
      <button type="submit">Sign in</button>
    </form>
  </div>
  <main>
    <nav class="tabs" aria-label="Admin sections">
      <button class="tab active" data-tab="overview" type="button">Overview</button>
      <button class="tab" data-tab="chat" type="button">Chat</button>
      <button class="tab" data-tab="models" type="button">Models</button>
      <button class="tab" data-tab="runtime" type="button">Runtime</button>
      <button class="tab" data-tab="diagnostics" type="button">Diagnostics</button>
      <button class="tab" data-tab="logs" type="button">Logs</button>
      <button class="tab" data-tab="config" id="tab-config" type="button">Config</button>
      <button class="tab" data-tab="training" id="tab-training" type="button">Training</button>
    </nav>
    <section id="overview-panel" class="panel">
      <div class="toolbar">
        <button id="overview-refresh" type="button">Refresh</button>
      </div>
      <section class="summary" aria-label="Overview summary">
        <div class="metric"><span>Health</span><strong id="overview-health">-</strong></div>
        <div class="metric"><span>PID</span><strong id="overview-pid">-</strong></div>
        <div class="metric"><span>Model</span><strong id="overview-model">-</strong></div>
        <div class="metric"><span>Training</span><strong id="overview-training">-</strong></div>
      </section>
      <div class="grid">
        <section>
          <h2 class="section-title">Runtime</h2>
          <table><tbody id="overview-runtime"></tbody></table>
        </section>
        <section>
          <h2 class="section-title">Recent Issues</h2>
          <div id="overview-issues" class="event-list"></div>
        </section>
      </div>
    </section>
    <section id="chat-panel" class="panel" hidden>
      <div class="toolbar">
        <label>Model
          <input id="chat-model" name="chat-model" autocomplete="off" placeholder="default runtime model">
        </label>
        <label>Max tokens
          <input id="chat-max-tokens" type="number" min="1" max="32768" value="256">
        </label>
        <label>Temperature
          <input id="chat-temperature" type="number" min="0" max="2" step="0.05" value="0.7">
        </label>
        <button id="chat-send" type="button">Send</button>
        <button id="chat-clear" type="button">Clear</button>
      </div>
      <div class="grid">
        <section>
          <h2 class="section-title">Messages</h2>
          <div id="chat-transcript" class="chat-transcript event-list"></div>
        </section>
        <section>
          <h2 class="section-title">Prompt</h2>
          <textarea id="chat-prompt" autocomplete="off" spellcheck="true" placeholder="Ask the loaded/default model"></textarea>
        </section>
      </div>
    </section>
    <section id="models-panel" class="panel" hidden>
      <div class="toolbar">
        <button id="models-refresh" type="button">Refresh</button>
      </div>
      <section class="summary" aria-label="Model summary">
        <div class="metric"><span>Registry Models</span><strong id="models-count">-</strong></div>
        <div class="metric"><span>Aliases</span><strong id="models-aliases">-</strong></div>
        <div class="metric"><span>Local Files</span><strong id="models-local">-</strong></div>
        <div class="metric"><span>Loaded</span><strong id="models-loaded">-</strong></div>
      </section>
      <table>
        <thead><tr><th>ID</th><th>File</th><th>Size</th><th>VRAM</th><th>Sidecars</th></tr></thead>
        <tbody id="models-rows"></tbody>
      </table>
    </section>
    <section id="runtime-panel" class="panel" hidden>
      <div class="toolbar">
        <button id="runtime-refresh" type="button">Refresh</button>
      </div>
      <section class="summary" aria-label="Runtime summary">
        <div class="metric"><span>Status</span><strong id="runtime-status">-</strong></div>
        <div class="metric"><span>Residency</span><strong id="runtime-residency">-</strong></div>
        <div class="metric"><span>Prefill Queue</span><strong id="runtime-prefill">-</strong></div>
        <div class="metric"><span>Batches</span><strong id="runtime-batches">-</strong></div>
      </section>
      <pre id="runtime-json"></pre>
    </section>
    <section id="diagnostics-panel" class="panel" hidden>
      <div class="toolbar">
        <button id="diagnostics-refresh" type="button">Refresh</button>
      </div>
      <div class="grid">
        <section>
          <h2 class="section-title">Paths and Binaries</h2>
          <table><tbody id="diagnostics-paths"></tbody></table>
        </section>
        <section>
          <h2 class="section-title">Kernel Cache and Locks</h2>
          <table><tbody id="diagnostics-kernels"></tbody></table>
        </section>
      </div>
    </section>
    <section id="logs-panel" class="panel" hidden>
      <div class="toolbar">
        <label>Lines
          <input id="logs-lines" type="number" min="1" max="1000" value="160">
        </label>
        <button id="logs-refresh" type="button">Refresh</button>
      </div>
      <div id="logs-list" class="stack"></div>
    </section>
    <section id="config-panel" class="panel" hidden>
      <div class="toolbar">
        <label>Model
          <input id="model" name="model" autocomplete="off" placeholder="optional model tag">
        </label>
        <label>Write target
          <select id="config-target" name="config-target">
            <option value="global">Global</option>
            <option value="host">Host</option>
            <option value="model">Model</option>
          </select>
        </label>
        <button id="refresh" type="button">Refresh</button>
      </div>
      <section class="summary" aria-label="Config summary">
        <div class="metric"><span>Source</span><strong id="source">-</strong></div>
        <div class="metric"><span>Path</span><strong id="path">-</strong></div>
        <div class="metric"><span>Fields</span><strong id="fields">-</strong></div>
        <div class="metric"><span>Diagnostics</span><strong id="diagnostics">-</strong></div>
      </section>
      <table>
        <thead>
          <tr>
            <th>Key</th>
            <th>Local</th>
            <th>Active</th>
            <th>Source</th>
            <th>Applies</th>
            <th>Edit</th>
            <th>Description</th>
          </tr>
        </thead>
        <tbody id="rows"></tbody>
      </table>
    </section>
    <section id="training-panel" class="panel" hidden>
      <div class="toolbar">
        <button id="training-refresh" type="button">Refresh</button>
      </div>
      <section class="summary" aria-label="Training summary">
        <div class="metric"><span>Runs</span><strong id="training-count">-</strong></div>
        <div class="metric"><span>Active</span><strong id="training-active">-</strong></div>
        <div class="metric"><span>Stale</span><strong id="training-stale">-</strong></div>
        <div class="metric"><span>Directory</span><strong id="training-dir">-</strong></div>
      </section>
      <div class="grid">
        <section>
          <h2 class="section-title">Runs</h2>
          <table>
            <thead>
              <tr>
                <th>ID</th>
                <th>Status</th>
                <th>Phase</th>
                <th>Progress</th>
                <th>Best</th>
              </tr>
            </thead>
            <tbody id="training-rows"></tbody>
          </table>
        </section>
        <section>
          <h2 class="section-title">Selected Run</h2>
          <section class="summary" aria-label="Selected training run">
            <div class="metric"><span>Target</span><strong id="training-target">-</strong></div>
            <div class="metric"><span>Artifact</span><strong id="training-artifact">-</strong></div>
            <div class="metric"><span>Checkpoint</span><strong id="training-checkpoint">-</strong></div>
            <div class="metric"><span>Admission</span><strong id="training-admission">-</strong></div>
          </section>
          <div id="training-events" class="event-list"></div>
        </section>
      </div>
    </section>
  </main>
  <script>
    const statusEl = document.getElementById("status");
    const loginOverlayEl = document.getElementById("login");
    const loginFormEl = document.getElementById("login-form");
    const loginUserEl = document.getElementById("login-user");
    const loginPasswordEl = document.getElementById("login-password");
    const loginErrorEl = document.getElementById("login-error");
    const logoutEl = document.getElementById("logout");
    let activeTab = "overview";
    const tabEls = [...document.querySelectorAll(".tab")];
    const panelEls = [...document.querySelectorAll(".panel")];
    const tabConfigEl = document.getElementById("tab-config");
    const tabTrainingEl = document.getElementById("tab-training");
    const configPanelEl = document.getElementById("config-panel");
    const trainingPanelEl = document.getElementById("training-panel");
    const overviewHealthEl = document.getElementById("overview-health");
    const overviewPidEl = document.getElementById("overview-pid");
    const overviewModelEl = document.getElementById("overview-model");
    const overviewTrainingEl = document.getElementById("overview-training");
    const overviewRuntimeEl = document.getElementById("overview-runtime");
    const overviewIssuesEl = document.getElementById("overview-issues");
    const chatModelEl = document.getElementById("chat-model");
    const chatMaxTokensEl = document.getElementById("chat-max-tokens");
    const chatTemperatureEl = document.getElementById("chat-temperature");
    const chatPromptEl = document.getElementById("chat-prompt");
    const chatSendEl = document.getElementById("chat-send");
    const chatClearEl = document.getElementById("chat-clear");
    const chatTranscriptEl = document.getElementById("chat-transcript");
    const modelsCountEl = document.getElementById("models-count");
    const modelsAliasesEl = document.getElementById("models-aliases");
    const modelsLocalEl = document.getElementById("models-local");
    const modelsLoadedEl = document.getElementById("models-loaded");
    const modelsRowsEl = document.getElementById("models-rows");
    const runtimeStatusEl = document.getElementById("runtime-status");
    const runtimeResidencyEl = document.getElementById("runtime-residency");
    const runtimePrefillEl = document.getElementById("runtime-prefill");
    const runtimeBatchesEl = document.getElementById("runtime-batches");
    const runtimeJsonEl = document.getElementById("runtime-json");
    const diagnosticsPathsEl = document.getElementById("diagnostics-paths");
    const diagnosticsKernelsEl = document.getElementById("diagnostics-kernels");
    const logsLinesEl = document.getElementById("logs-lines");
    const logsListEl = document.getElementById("logs-list");
    const modelEl = document.getElementById("model");
    const configTargetEl = document.getElementById("config-target");
    const refreshEl = document.getElementById("refresh");
    const rowsEl = document.getElementById("rows");
    const sourceEl = document.getElementById("source");
    const pathEl = document.getElementById("path");
    const fieldsEl = document.getElementById("fields");
    const diagnosticsEl = document.getElementById("diagnostics");
    const trainingRefreshEl = document.getElementById("training-refresh");
    const trainingCountEl = document.getElementById("training-count");
    const trainingActiveEl = document.getElementById("training-active");
    const trainingStaleEl = document.getElementById("training-stale");
    const trainingDirEl = document.getElementById("training-dir");
    const trainingRowsEl = document.getElementById("training-rows");
    const trainingTargetEl = document.getElementById("training-target");
    const trainingArtifactEl = document.getElementById("training-artifact");
    const trainingCheckpointEl = document.getElementById("training-checkpoint");
    const trainingAdmissionEl = document.getElementById("training-admission");
    const trainingEventsEl = document.getElementById("training-events");
    let selectedTrainingRun = null;
    let chatMessages = [];
    let currentConfigSnapshot = null;

    function text(value) {
      if (value === null || value === undefined) return "";
      if (typeof value === "string") return value;
      return JSON.stringify(value);
    }

    function sourceLabel(source) {
      if (!source) return "";
      return source.id ? `${source.kind}:${source.id}` : source.kind;
    }

    function impactLabel(row) {
      if (row.pending) return "pending reload";
      if (row.restart_impact && row.restart_impact !== "none") return row.restart_impact.replaceAll("_", " ");
      if (row.mutability === "request_only") return "new requests";
      if (row.mutability === "runtime_reloadable") return "runtime reload";
      if (row.mutability === "load_time") return "reload model";
      return "restart daemon";
    }

    function requireAuthorized(resp) {
      if (resp.status === 401) {
        showLogin();
        throw new Error("admin authentication required");
      }
      return resp;
    }

    async function fetchJson(path) {
      const resp = requireAuthorized(await fetch(path));
      if (!resp.ok) throw new Error(`${path} ${resp.status}`);
      return await resp.json();
    }

    function showLogin() {
      loginOverlayEl.hidden = false;
      logoutEl.hidden = true;
      loginPasswordEl.focus();
    }

    function hideLogin() {
      loginOverlayEl.hidden = true;
      logoutEl.hidden = false;
      loginErrorEl.textContent = "";
    }

    async function submitLogin(event) {
      event.preventDefault();
      loginErrorEl.textContent = "";
      const resp = await fetch("/admin/login", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({user: loginUserEl.value, password: loginPasswordEl.value}),
      });
      if (!resp.ok) {
        loginErrorEl.textContent = resp.status === 401 ? "invalid credentials" : `login failed (${resp.status})`;
        return;
      }
      loginPasswordEl.value = "";
      hideLogin();
      showTab(activeTab);
    }

    async function logout() {
      await fetch("/admin/logout", {method: "POST"}).catch(() => {});
      showLogin();
      statusEl.textContent = "signed out";
    }

    function td(value, cls = "") {
      const cell = document.createElement("td");
      if (cls) cell.className = cls;
      cell.textContent = text(value) || "-";
      return cell;
    }

    function keyValueRows(entries) {
      return entries.map(([key, value, cls]) => {
        const tr = document.createElement("tr");
        tr.append(td(key, "key"), td(value, cls || "value"));
        return tr;
      });
    }

    function issueCard(title, detail, warn = false) {
      const div = document.createElement("div");
      div.className = warn ? "event warn" : "event";
      const strong = document.createElement("strong");
      strong.textContent = title;
      const code = document.createElement("code");
      code.textContent = detail || "";
      div.append(strong, code);
      return div;
    }

    function chatCard(message) {
      const div = document.createElement("div");
      div.className = "event";
      const role = document.createElement("strong");
      role.textContent = message.role || "message";
      const code = document.createElement("code");
      code.textContent = typeof message.content === "string" ? message.content : text(message.content);
      div.append(role, code);
      return div;
    }

    function renderChat() {
      if (!chatMessages.length) {
        chatTranscriptEl.replaceChildren(issueCard("No messages", "Send a prompt to /v1/chat/completions."));
        return;
      }
      chatTranscriptEl.replaceChildren(...chatMessages.map(chatCard));
      chatTranscriptEl.scrollTop = chatTranscriptEl.scrollHeight;
    }

    async function loadChat() {
      if (!chatModelEl.value.trim()) {
        const health = await fetchJson("/health");
        chatModelEl.value = health.active_model || health.model || "";
      }
      renderChat();
      statusEl.textContent = "chat";
    }

    async function sendChat() {
      const prompt = chatPromptEl.value.trim();
      if (!prompt) {
        statusEl.textContent = "empty prompt";
        return;
      }
      const nextMessages = [...chatMessages, {role: "user", content: prompt}];
      const body = {
        stream: false,
        messages: nextMessages,
        max_tokens: Math.max(1, Number(chatMaxTokensEl.value || 256)),
        temperature: Number(chatTemperatureEl.value || 0.7),
      };
      const model = chatModelEl.value.trim();
      if (model) body.model = model;
      statusEl.textContent = "sending chat";
      chatSendEl.disabled = true;
      try {
        const resp = await fetch("/v1/chat/completions", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify(body),
        });
        const payload = await resp.json();
        if (!resp.ok || payload.error) {
          throw new Error((payload.error && payload.error.message) || `chat ${resp.status}`);
        }
        const choice = payload.choices && payload.choices[0];
        const message = choice && choice.message || {};
        const content = message.content || (message.tool_calls ? JSON.stringify(message.tool_calls, null, 2) : "");
        chatMessages = [...nextMessages, {role: "assistant", content}];
        chatPromptEl.value = "";
        renderChat();
        statusEl.textContent = "chat complete";
      } finally {
        chatSendEl.disabled = false;
      }
    }

    async function loadOverview() {
      statusEl.textContent = "loading overview";
      const [health, diagnostics, training] = await Promise.all([
        fetchJson("/health"),
        fetchJson("/admin/diagnostics"),
        fetchJson("/admin/training/runs"),
      ]);
      overviewHealthEl.textContent = health.status || "-";
      overviewPidEl.textContent = health.pid || "-";
      overviewModelEl.textContent = health.active_model || health.model || "none";
      const runs = training.runs || [];
      overviewTrainingEl.textContent = `${runs.filter(isActiveRun).length} active / ${runs.length} runs`;
      overviewRuntimeEl.replaceChildren(...keyValueRows([
        ["Bind", location.origin],
        ["Residency", health.scheduler_resources && health.scheduler_resources.model_residency_mode],
        ["Prefill queue", health.prefill_batch && (health.prefill_batch.queue_size ?? health.prefill_batch.queued)],
        ["Batches", health.batches && `${health.batches.queued || 0} queued / ${health.batches.total || 0} total`],
        ["Kernel caches", (diagnostics.kernel_caches || []).map((k) => `${k.arch}:${k.hsaco}/${k.hash}`).join(", ") || "none"],
        ["Resource locks", (diagnostics.resource_locks || []).length],
      ]));
      const issues = [];
      for (const path of diagnostics.paths || []) {
        if (!path.exists && ["config", "models", "kernels"].includes(path.name)) {
          issues.push(issueCard(`missing ${path.name}`, path.path, true));
        }
      }
      for (const cache of diagnostics.kernel_caches || []) {
        if (!cache.balanced) issues.push(issueCard(`kernel cache mismatch ${cache.arch}`, `${cache.hsaco} hsaco / ${cache.hash} hash`, true));
      }
      if (!issues.length) issues.push(issueCard("No current admin issues", "Health, diagnostics, and kernel cache checks did not report a warning."));
      overviewIssuesEl.replaceChildren(...issues);
      statusEl.textContent = "overview";
    }

    async function loadModels() {
      statusEl.textContent = "loading models";
      const [health, registry] = await Promise.all([
        fetchJson("/health"),
        fetchJson("/admin/models/registry"),
      ]);
      const models = registry.models || [];
      const aliases = registry.aliases || {};
      modelsCountEl.textContent = String(models.length);
      modelsAliasesEl.textContent = String(Object.keys(aliases).length);
      modelsLocalEl.textContent = String(models.filter((m) => m.path || m.file).length);
      modelsLoadedEl.textContent = health.active_model || health.model || "none";
      modelsRowsEl.replaceChildren(...models.map((model) => {
        const tr = document.createElement("tr");
        const sidecarText = [
          ...(model.triattn || []),
          ...(model.drafts || []),
          ...(model.chat_templates || []),
        ].map((s) => typeof s === "string" ? s : (s.file || s.path || s.id || "")).filter(Boolean).join(", ");
        tr.append(
          td(model.id || model.tag || "-", "key"),
          td(model.file || model.path || "-", "value"),
          td(model.bytes ? `${(model.bytes / 1_000_000_000).toFixed(2)} GB` : (model.size_gb ? `${model.size_gb} GB` : "-")),
          td(model.min_vram_gb ? `${model.min_vram_gb} GB` : "-"),
          td(sidecarText || "-")
        );
        return tr;
      }));
      if (!models.length) {
        const tr = document.createElement("tr");
        const cell = td("No models found", "muted");
        cell.colSpan = 5;
        tr.append(cell);
        modelsRowsEl.replaceChildren(tr);
      }
      statusEl.textContent = "models";
    }

    async function loadRuntime() {
      statusEl.textContent = "loading runtime";
      const health = await fetchJson("/health");
      runtimeStatusEl.textContent = health.status || "-";
      runtimeResidencyEl.textContent = health.scheduler_resources ? (health.scheduler_resources.model_residency_mode || "-") : "-";
      runtimePrefillEl.textContent = health.prefill_batch ? `${health.prefill_batch.queue_size || health.prefill_batch.queued || 0}` : "-";
      runtimeBatchesEl.textContent = health.batches ? `${health.batches.queued || 0} queued / ${health.batches.total || 0} total` : "-";
      runtimeJsonEl.textContent = JSON.stringify(health, null, 2);
      statusEl.textContent = "runtime";
    }

    async function loadDiagnostics() {
      statusEl.textContent = "loading diagnostics";
      const diagnostics = await fetchJson("/admin/diagnostics");
      const paths = [...(diagnostics.paths || []), ...(diagnostics.binaries || [])];
      diagnosticsPathsEl.replaceChildren(...paths.map((item) => {
        const tr = document.createElement("tr");
        tr.append(td(item.name, "key"), td(item.exists ? "present" : "missing", item.exists ? "" : "warn"), td(item.path, "value"));
        return tr;
      }));
      const kernels = diagnostics.kernel_caches || [];
      const locks = diagnostics.resource_locks || [];
      const rows = kernels.map((kernel) => {
        const tr = document.createElement("tr");
        tr.append(td(kernel.arch, "key"), td(`${kernel.hsaco} hsaco / ${kernel.hash} hash`, kernel.balanced ? "" : "warn"), td(kernel.path, "value"));
        return tr;
      });
      rows.push(...locks.map((lock) => {
        const tr = document.createElement("tr");
        tr.append(td(lock.name, "key"), td("lock"), td(lock.content || lock.path, "value"));
        return tr;
      }));
      if (!rows.length) {
        const tr = document.createElement("tr");
        const cell = td("No kernel caches or locks found", "muted");
        cell.colSpan = 3;
        tr.append(cell);
        rows.push(tr);
      }
      diagnosticsKernelsEl.replaceChildren(...rows);
      statusEl.textContent = "diagnostics";
    }

    async function loadLogs() {
      statusEl.textContent = "loading logs";
      const lines = Number(logsLinesEl.value || 160);
      const payload = await fetchJson(`/admin/logs?lines=${Math.max(1, Math.min(1000, lines))}`);
      const logs = payload.logs || [];
      logsListEl.replaceChildren(...logs.map((log) => {
        const section = document.createElement("section");
        const title = document.createElement("h2");
        title.className = "section-title";
        title.textContent = log.path;
        const pre = document.createElement("pre");
        pre.textContent = log.text || "";
        section.append(title, pre);
        return section;
      }));
      if (!logs.length) {
        logsListEl.replaceChildren(issueCard("No logs found", "No known hipfire log files were present."));
      }
      statusEl.textContent = "logs";
    }

    async function loadConfig() {
      const model = modelEl.value.trim();
      const suffix = model ? `?model=${encodeURIComponent(model)}` : "";
      statusEl.textContent = "loading";
      const snapshot = await fetchJson(`/admin/config/editor${suffix}`);
      render(snapshot);
      statusEl.textContent = model ? `model ${model}` : "config editor";
    }

    async function loadTraining() {
      statusEl.textContent = "loading training";
      const resp = await fetch("/admin/training/runs");
      if (!resp.ok) throw new Error(`training ${resp.status}`);
      const payload = await resp.json();
      renderTraining(payload);
      const ids = (payload.runs || []).map((run) => run.id);
      const first = ids[0] || null;
      if (!selectedTrainingRun || !ids.includes(selectedTrainingRun)) selectedTrainingRun = first;
      if (selectedTrainingRun) {
        await loadTrainingDetail(selectedTrainingRun);
      } else {
        clearTrainingDetail();
      }
      statusEl.textContent = "training";
    }

    async function loadTrainingDetail(id) {
      selectedTrainingRun = id;
      const resp = await fetch(`/admin/training/runs/${encodeURIComponent(id)}`);
      if (!resp.ok) throw new Error(`training run ${resp.status}`);
      const detail = await resp.json();
      renderTrainingDetail(detail);
    }

    function selectedConfigTarget() {
      const target = configTargetEl.value || "global";
      if (target === "model") {
        const id = modelEl.value.trim();
        if (!id) throw new Error("model target requires a model id");
        return {kind: "model", id};
      }
      return {kind: target};
    }

    function rowEditableForTarget(row) {
      const target = configTargetEl.value || "global";
      if (target === "model") return !!row.editable_model;
      if (target === "host") return !!row.editable_host;
      return !!row.editable_global;
    }

    function controlValue(control, row) {
      const kind = row.type && row.type.kind;
      if (kind === "bool") return control.checked;
      if (kind === "u8" || kind === "u16" || kind === "u32" || kind === "i32" || kind === "f64") return control.value;
      if (kind === "json") {
        try { return JSON.parse(control.value); } catch (_) { return control.value; }
      }
      return control.value;
    }

    async function saveConfigRow(row, control) {
      const model = modelEl.value.trim();
      const body = {
        target: selectedConfigTarget(),
        key: row.key,
        operation: "set",
        value: controlValue(control, row),
      };
      if (model) body.model = model;
      const resp = requireAuthorized(await fetch("/admin/config/editor", {
        method: "PATCH",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(body),
      }));
      const payload = await resp.json();
      if (!resp.ok || payload.error) throw new Error((payload.error && payload.error.message) || `config ${resp.status}`);
      render(payload);
      statusEl.textContent = `saved ${row.key}`;
    }

    async function unsetConfigRow(row) {
      const model = modelEl.value.trim();
      const body = {
        target: selectedConfigTarget(),
        key: row.key,
        operation: "unset",
      };
      if (model) body.model = model;
      const resp = requireAuthorized(await fetch("/admin/config/editor", {
        method: "PATCH",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(body),
      }));
      const payload = await resp.json();
      if (!resp.ok || payload.error) throw new Error((payload.error && payload.error.message) || `config ${resp.status}`);
      render(payload);
      statusEl.textContent = `unset ${row.key}`;
    }

    function configControl(row) {
      const wrap = document.createElement("div");
      wrap.className = "inline-actions";
      const kind = row.type && row.type.kind;
      const value = row.local_value === undefined || row.local_value === null ? "" : row.local_value;
      let control;
      if (kind === "bool") {
        control = document.createElement("input");
        control.type = "checkbox";
        control.checked = value === true;
      } else if (row.enum_choices && row.enum_choices.length) {
        control = document.createElement("select");
        for (const choice of row.enum_choices) {
          const option = document.createElement("option");
          option.value = choice;
          option.textContent = choice;
          option.selected = choice === value;
          control.append(option);
        }
      } else {
        control = document.createElement("input");
        control.type = (kind === "u8" || kind === "u16" || kind === "u32" || kind === "i32" || kind === "f64") ? "number" : "text";
        control.value = kind === "json" ? JSON.stringify(value) : text(value);
      }
      control.disabled = !rowEditableForTarget(row);
      const save = document.createElement("button");
      save.type = "button";
      save.textContent = "Save";
      save.disabled = control.disabled;
      save.addEventListener("click", () => saveConfigRow(row, control).catch(showError));
      const unset = document.createElement("button");
      unset.type = "button";
      unset.textContent = "Unset";
      unset.disabled = control.disabled;
      unset.addEventListener("click", () => unsetConfigRow(row).catch(showError));
      wrap.append(control, save, unset);
      return wrap;
    }

    function render(snapshot) {
      currentConfigSnapshot = snapshot;
      const rows = snapshot.rows || [];
      sourceEl.textContent = snapshot.source || "-";
      pathEl.textContent = snapshot.config_path || "-";
      fieldsEl.textContent = String(rows.length);
      const diagnostics = snapshot.diagnostics || [];
      const readError = [snapshot.read_error, snapshot.host_read_error].filter(Boolean);
      const unknown = (snapshot.unknown_keys || []).map((key) => `unknown ${key.key} in ${sourceLabel(key.source)}`);
      diagnosticsEl.textContent = [...readError, ...diagnostics.map((d) => d.message), ...unknown].join("; ") || "none";
      diagnosticsEl.className = diagnostics.length || readError.length || unknown.length ? "warn" : "";
      rowsEl.replaceChildren(...rows.map((entry) => {
        const tr = document.createElement("tr");
        const key = document.createElement("td");
        key.className = "key";
        key.textContent = entry.key;
        const local = document.createElement("td");
        local.className = entry.pending ? "value warn" : "value";
        local.textContent = text(entry.local_value);
        const active = document.createElement("td");
        active.className = "value";
        active.textContent = text(entry.active_value);
        const source = document.createElement("td");
        source.className = "source";
        source.textContent = `${sourceLabel(entry.local_source) || "-"} / ${sourceLabel(entry.active_source) || "-"}`;
        const applies = document.createElement("td");
        applies.className = entry.pending ? "warn" : "muted";
        applies.textContent = impactLabel(entry);
        const edit = document.createElement("td");
        edit.append(configControl(entry));
        const desc = document.createElement("td");
        desc.textContent = entry.description || "";
        tr.append(key, local, active, source, applies, edit, desc);
        return tr;
      }));
    }

    function isActiveRun(run) {
      return ["queued", "capturing", "training", "evaluating", "checkpointing", "exporting"].includes(run.status || "");
    }

    function progressLabel(run) {
      const progress = run.progress || {};
      if (progress.percent !== undefined && progress.percent !== null) return `${Number(progress.percent).toFixed(1)}%`;
      if (progress.current_step !== undefined && progress.total_steps !== undefined) return `${progress.current_step}/${progress.total_steps}`;
      if (progress.current_step !== undefined) return String(progress.current_step);
      return "-";
    }

    function metricLabel(run) {
      const metrics = run.metrics || {};
      const value = metrics.best_eval_metric ?? metrics.eval_metric;
      return value === undefined || value === null ? "-" : Number(value).toFixed(4);
    }

    function renderTraining(payload) {
      const runs = payload.runs || [];
      trainingCountEl.textContent = String(runs.length);
      trainingActiveEl.textContent = String(runs.filter(isActiveRun).length);
      trainingStaleEl.textContent = String(runs.filter((run) => run.stale).length);
      trainingDirEl.textContent = payload.runs_dir || "-";
      trainingRowsEl.replaceChildren(...runs.map((run) => {
        const tr = document.createElement("tr");
        tr.className = `selectable${run.id === selectedTrainingRun ? " selected" : ""}`;
        tr.addEventListener("click", () => loadTrainingDetail(run.id).catch(showError));
        const id = document.createElement("td");
        id.className = "key";
        id.textContent = run.id || "-";
        const status = document.createElement("td");
        status.className = run.last_error || run.read_error ? "warn" : "";
        status.textContent = run.stale ? `${run.status || "unknown"} stale` : run.status || "unknown";
        const phase = document.createElement("td");
        phase.textContent = (run.progress && run.progress.phase) || run.status || "unknown";
        const progress = document.createElement("td");
        progress.textContent = progressLabel(run);
        const best = document.createElement("td");
        best.textContent = metricLabel(run);
        tr.append(id, status, phase, progress, best);
        return tr;
      }));
      if (!runs.length) {
        const row = document.createElement("tr");
        const cell = document.createElement("td");
        cell.colSpan = 5;
        cell.className = "muted";
        cell.textContent = "No training runs found.";
        row.append(cell);
        trainingRowsEl.replaceChildren(row);
      }
    }

    function renderTrainingDetail(detail) {
      const run = detail.summary || {};
      trainingTargetEl.textContent = run.target_model || "-";
      trainingArtifactEl.textContent = run.artifact || (run.handoff && run.handoff.artifact) || "-";
      trainingCheckpointEl.textContent = run.checkpoint && (run.checkpoint.path || run.checkpoint.state) || "-";
      trainingAdmissionEl.textContent = run.handoff && (run.handoff.admission_verdict || run.handoff.admission_status) || "-";
      const events = detail.recent_events || [];
      const errors = detail.event_errors || [];
      const cards = [];
      if (run.last_error || run.read_error) {
        const div = document.createElement("div");
        div.className = "event warn";
        div.innerHTML = `<strong>latest issue</strong><code></code>`;
        div.querySelector("code").textContent = (run.last_error && run.last_error.message) || run.read_error || "";
        cards.push(div);
      }
      for (const record of events.slice(-12).reverse()) {
        const div = document.createElement("div");
        div.className = "event";
        const event = record.event || {};
        const title = document.createElement("strong");
        title.textContent = `${record.line}: ${event.type || "unknown"}`;
        const code = document.createElement("code");
        code.textContent = JSON.stringify(event, null, 2);
        div.append(title, code);
        cards.push(div);
      }
      for (const err of errors.slice(-4)) {
        const div = document.createElement("div");
        div.className = "event warn";
        const title = document.createElement("strong");
        title.textContent = `line ${err.line}: malformed event`;
        const code = document.createElement("code");
        code.textContent = err.message;
        div.append(title, code);
        cards.push(div);
      }
      if (!cards.length) {
        const div = document.createElement("div");
        div.className = "event muted";
        div.textContent = "No events recorded for this run.";
        cards.push(div);
      }
      trainingEventsEl.replaceChildren(...cards);
      for (const row of trainingRowsEl.querySelectorAll("tr")) {
        const idCell = row.querySelector("td");
        row.classList.toggle("selected", idCell && idCell.textContent === selectedTrainingRun);
      }
    }

    function clearTrainingDetail() {
      trainingTargetEl.textContent = "-";
      trainingArtifactEl.textContent = "-";
      trainingCheckpointEl.textContent = "-";
      trainingAdmissionEl.textContent = "-";
      const div = document.createElement("div");
      div.className = "event muted";
      div.textContent = "No selected training run.";
      trainingEventsEl.replaceChildren(div);
    }

    function showTab(name) {
      activeTab = name;
      for (const panel of panelEls) panel.hidden = panel.id !== `${name}-panel`;
      for (const tab of tabEls) tab.classList.toggle("active", tab.dataset.tab === name);
      const loaders = {
        overview: loadOverview,
        chat: loadChat,
        models: loadModels,
        runtime: loadRuntime,
        diagnostics: loadDiagnostics,
        logs: loadLogs,
        config: loadConfig,
        training: loadTraining,
      };
      (loaders[name] || loadOverview)().catch(showError);
    }

    document.getElementById("overview-refresh").addEventListener("click", () => loadOverview().catch(showError));
    chatSendEl.addEventListener("click", () => sendChat().catch(showError));
    chatClearEl.addEventListener("click", () => {
      chatMessages = [];
      renderChat();
      statusEl.textContent = "chat cleared";
    });
    chatPromptEl.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
        sendChat().catch(showError);
      }
    });
    document.getElementById("models-refresh").addEventListener("click", () => loadModels().catch(showError));
    document.getElementById("runtime-refresh").addEventListener("click", () => loadRuntime().catch(showError));
    document.getElementById("diagnostics-refresh").addEventListener("click", () => loadDiagnostics().catch(showError));
    document.getElementById("logs-refresh").addEventListener("click", () => loadLogs().catch(showError));
    refreshEl.addEventListener("click", () => loadConfig().catch(showError));
    configTargetEl.addEventListener("change", () => {
      if (currentConfigSnapshot) render(currentConfigSnapshot);
    });
    trainingRefreshEl.addEventListener("click", () => loadTraining().catch(showError));
    for (const tab of tabEls) tab.addEventListener("click", () => showTab(tab.dataset.tab));
    modelEl.addEventListener("keydown", (event) => {
      if (event.key === "Enter") loadConfig().catch(showError);
    });
    loginFormEl.addEventListener("submit", (event) => submitLogin(event).catch(showError));
    logoutEl.addEventListener("click", () => logout().catch(showError));
    function showError(error) {
      statusEl.textContent = error.message;
      statusEl.className = "status warn";
    }
    loadOverview().catch(showError);
  </script>
</body>
</html>"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_schema_route_exposes_schema_fields() {
        let payload = config_schema_json();
        let fields = payload.as_array().expect("schema array");

        assert!(fields
            .iter()
            .any(|field| field.get("key").and_then(Value::as_str) == Some("max_tokens")));
        assert!(fields.iter().any(|field| {
            field
                .get("requirement")
                .and_then(|req| req.get("kind"))
                .and_then(Value::as_str)
                == Some("required_when")
        }));
    }

    #[test]
    fn admin_index_fetches_config_endpoints() {
        assert!(ADMIN_INDEX_HTML.contains("/admin/config/editor"));
        assert!(ADMIN_INDEX_HTML.contains("method: \"PATCH\""));
    }

    #[test]
    fn admin_index_exposes_login_surface() {
        assert!(ADMIN_INDEX_HTML.contains("/admin/login"));
        assert!(ADMIN_INDEX_HTML.contains("/admin/logout"));
        assert!(ADMIN_INDEX_HTML.contains("id=\"login-form\""));
        assert!(ADMIN_INDEX_HTML.contains("function showLogin"));
        assert!(ADMIN_INDEX_HTML.contains("requireAuthorized"));
    }

    #[test]
    fn admin_index_html_tags_are_balanced() {
        // An unclosed <style> swallows the whole <body> as CSS text and the
        // page renders blank below the head — guard the structural tags.
        for (open, close) in [
            ("<style>", "</style>"),
            ("<head>", "</head>"),
            ("<body>", "</body>"),
            ("<main>", "</main>"),
            ("<script>", "</script>"),
        ] {
            assert_eq!(
                ADMIN_INDEX_HTML.matches(open).count(),
                ADMIN_INDEX_HTML.matches(close).count(),
                "unbalanced {open}/{close} in admin console HTML"
            );
            assert_eq!(
                ADMIN_INDEX_HTML.matches(open).count(),
                1,
                "expected exactly one {open}"
            );
        }
    }

    #[test]
    fn admin_index_exposes_training_surface() {
        assert!(ADMIN_INDEX_HTML.contains("Training"));
        assert!(ADMIN_INDEX_HTML.contains("/admin/training/runs"));
        assert!(ADMIN_INDEX_HTML.contains("training-events"));
    }

    #[test]
    fn admin_index_exposes_runtime_diagnostics_and_logs_surfaces() {
        assert!(ADMIN_INDEX_HTML.contains("Overview"));
        assert!(ADMIN_INDEX_HTML.contains("Chat"));
        assert!(ADMIN_INDEX_HTML.contains("Models"));
        assert!(ADMIN_INDEX_HTML.contains("Runtime"));
        assert!(ADMIN_INDEX_HTML.contains("Diagnostics"));
        assert!(ADMIN_INDEX_HTML.contains("Logs"));
        assert!(ADMIN_INDEX_HTML.contains("/v1/chat/completions"));
        assert!(ADMIN_INDEX_HTML.contains("/admin/diagnostics"));
        assert!(ADMIN_INDEX_HTML.contains("/admin/logs"));
        assert!(ADMIN_INDEX_HTML.contains("/admin/models/registry"));
    }

    #[test]
    fn tail_file_limits_lines() {
        let path =
            std::env::temp_dir().join(format!("hipfire-admin-tail-{}.log", std::process::id()));
        std::fs::write(&path, "one\ntwo\nthree\nfour\n").expect("write log");

        let tail = tail_file(&path, 2);

        let _ = std::fs::remove_file(path);
        assert_eq!(tail, "three\nfour");
    }

    #[test]
    fn resolved_config_route_explains_model_override_source() {
        let document = json!({
            "max_tokens": 256,
            "model_overrides": {
                "qwen3.5:9b": {
                    "max_tokens": 64
                }
            }
        });

        let loaded = hipfire_config::loaded_config_from_document(
            std::path::PathBuf::from("/tmp/config.json"),
            document,
            None,
            Vec::new(),
        );

        let payload = resolved_config_json(&loaded, Some("qwen3.5:9b"));
        let values = payload["resolution"]["values"]
            .as_array()
            .expect("resolved values");
        let max_tokens = values
            .iter()
            .find(|value| value["key"] == "max_tokens")
            .expect("max_tokens");

        assert_eq!(max_tokens["value"], json!(64));
        assert_eq!(max_tokens["source"]["kind"], "model");
        assert_eq!(max_tokens["source"]["id"], "qwen3.5:9b");
        assert!(max_tokens["overrode"]
            .as_array()
            .expect("overrode")
            .iter()
            .any(|source| source["kind"] == "global"));
    }

    #[test]
    fn config_editor_snapshot_explains_pending_cli_override() {
        let path = std::env::temp_dir().join(format!(
            "hipfire-admin-config-editor-{}.json",
            std::process::id()
        ));
        let document = json!({
            "max_tokens": 256,
        });
        std::fs::write(&path, serde_json::to_string(&document).unwrap()).expect("write config");
        let cli_layer = hipfire_config::ConfigLayer::new(hipfire_config::ConfigLayerKind::Cli)
            .with_value("max_tokens", 64);
        let loaded = hipfire_config::loaded_config_from_document(
            path.clone(),
            document,
            None,
            vec![cli_layer],
        );

        let payload = config_editor_json(&loaded, None);
        let _ = std::fs::remove_file(path);
        let rows = payload["rows"].as_array().expect("rows");
        let max_tokens = rows
            .iter()
            .find(|row| row["key"] == "max_tokens")
            .expect("max_tokens row");

        assert_eq!(max_tokens["local_value"], json!(256));
        assert_eq!(max_tokens["active_value"], json!(64));
        assert_eq!(max_tokens["active_source"]["kind"], "cli");
        assert_eq!(max_tokens["pending"], json!(true));
    }

    #[test]
    fn resolved_config_route_reports_active_cli_layer() {
        let document = json!({
            "host": "127.0.0.1",
            "port": 11435
        });
        let cli_layer = hipfire_config::ConfigLayer::new(hipfire_config::ConfigLayerKind::Cli)
            .with_value("port", 12000);
        let loaded = hipfire_config::loaded_config_from_document(
            std::path::PathBuf::from("/tmp/config.json"),
            document,
            None,
            vec![cli_layer],
        );

        let payload = resolved_config_json(&loaded, None);
        assert_eq!(payload["config"]["port"], json!(12000));
        assert!(payload["layers"]
            .as_array()
            .expect("layers")
            .iter()
            .any(|layer| layer["kind"] == "cli"));
    }
}
