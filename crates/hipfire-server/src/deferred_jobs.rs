use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    process::Stdio,
    time::{SystemTime, UNIX_EPOCH},
};

use axum::{body::to_bytes, extract::State, response::Response, Json};
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::process::Command;
use uuid::Uuid;

use crate::{
    routes::{
        chat::{
            execute_blocking_chat, openai_chat_completion_response_with_tool_calls_json,
            ChatRequest,
        },
        responses::{execute_responses, ResponsesRequest},
        sdapi::{post_img2img, post_txt2img, SdGenerationRequest},
    },
    SharedState,
};

const QUEUED_DIR: &str = "queued";
const RUNNING_DIR: &str = "running";
const DONE_DIR: &str = "done";
const FAILED_DIR: &str = "failed";
const LOGS_DIR: &str = "logs";

#[derive(Debug, Deserialize)]
pub struct DeferredJobSpec {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(flatten)]
    pub kind: DeferredJobKind,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum DeferredJobKind {
    /// Execute the same body accepted by POST /sdapi/v1/txt2img.
    SdapiTxt2img { body: SdGenerationRequest },
    /// Execute the same body accepted by POST /sdapi/v1/img2img.
    SdapiImg2img { body: SdGenerationRequest },
    /// Execute a supported in-process POST endpoint.
    HttpPost { endpoint: String, body: Value },
    /// Launch a no-shell training command from a local deferred job file.
    ///
    /// This is intentionally file-backed only; no HTTP route creates these jobs.
    TrainingCommand {
        argv: Vec<String>,
        #[serde(default)]
        cwd: Option<PathBuf>,
        #[serde(default)]
        env: BTreeMap<String, String>,
        #[serde(default)]
        run_id: Option<String>,
        #[serde(default)]
        target_model: Option<String>,
        #[serde(default)]
        artifact: Option<String>,
    },
}

#[derive(Debug, Default)]
pub struct DeferredStartupSummary {
    pub queued: usize,
    pub completed: usize,
    pub failed: usize,
}

pub fn deferred_jobs_root() -> PathBuf {
    hipfire_config::hipfire_dir().join("jobs").join("deferred")
}

pub fn spawn_deferred_job_runner(state: SharedState) {
    if !deferred_jobs_enabled() {
        tracing::info!("deferred job startup runner disabled");
        return;
    }
    let root = deferred_jobs_root();
    tokio::spawn(async move {
        match run_startup_deferred_jobs(state, root.clone()).await {
            Ok(summary) => {
                if summary.queued > 0 {
                    tracing::info!(
                        root = %root.display(),
                        queued = summary.queued,
                        completed = summary.completed,
                        failed = summary.failed,
                        "deferred startup jobs finished"
                    );
                }
            }
            Err(error) => {
                tracing::error!(
                    root = %root.display(),
                    error = %error,
                    "deferred startup job scan failed"
                );
            }
        }
    });
}

pub fn deferred_jobs_health_json() -> Value {
    deferred_jobs_health_json_at(&deferred_jobs_root())
}

fn deferred_jobs_health_json_at(root: &Path) -> Value {
    json!({
        "enabled": deferred_jobs_enabled(),
        "root": root,
        "queued": count_json_files(&root.join(QUEUED_DIR)),
        "running": count_json_files(&root.join(RUNNING_DIR)),
        "done": count_terminal_job_files(&root.join(DONE_DIR)),
        "failed": count_terminal_job_files(&root.join(FAILED_DIR)),
        "execution_mode": "startup_sequential",
        "supported_kinds": [
            "sdapi_txt2img",
            "sdapi_img2img",
            "http_post",
            "training_command"
        ],
    })
}

pub async fn run_startup_deferred_jobs(
    state: SharedState,
    root: PathBuf,
) -> Result<DeferredStartupSummary, String> {
    ensure_deferred_dirs(&root)?;
    recover_running_jobs(&root)?;

    let mut summary = DeferredStartupSummary::default();
    loop {
        let Some(path) = next_queued_job(&root)? else {
            break;
        };
        summary.queued += 1;
        let running_path = claim_job(&root, &path)?;
        match run_claimed_job(state.clone(), &root, running_path).await {
            Ok(()) => summary.completed += 1,
            Err(error) => {
                summary.failed += 1;
                tracing::error!(error = %error, "deferred job failed");
            }
        }
    }
    Ok(summary)
}

fn ensure_deferred_dirs(root: &Path) -> Result<(), String> {
    for name in [QUEUED_DIR, RUNNING_DIR, DONE_DIR, FAILED_DIR, LOGS_DIR] {
        fs::create_dir_all(root.join(name))
            .map_err(|e| format!("create deferred job dir {}: {e}", root.join(name).display()))?;
    }
    Ok(())
}

fn recover_running_jobs(root: &Path) -> Result<(), String> {
    for path in list_json_files(&root.join(RUNNING_DIR))? {
        let file_name = path
            .file_name()
            .ok_or_else(|| format!("invalid running job path {}", path.display()))?;
        let queued = unique_path(&root.join(QUEUED_DIR).join(file_name));
        fs::rename(&path, &queued).map_err(|e| {
            format!(
                "recover deferred job {} -> {}: {e}",
                path.display(),
                queued.display()
            )
        })?;
    }
    Ok(())
}

fn next_queued_job(root: &Path) -> Result<Option<PathBuf>, String> {
    Ok(list_json_files(&root.join(QUEUED_DIR))?.into_iter().next())
}

fn list_json_files(dir: &Path) -> Result<Vec<PathBuf>, String> {
    let entries = match fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(err) => return Err(format!("read deferred job dir {}: {err}", dir.display())),
    };
    let mut paths = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|e| format!("read deferred job entry: {e}"))?;
        let path = entry.path();
        if path.extension().and_then(|ext| ext.to_str()) == Some("json") {
            paths.push(path);
        }
    }
    paths.sort();
    Ok(paths)
}

fn claim_job(root: &Path, queued_path: &Path) -> Result<PathBuf, String> {
    let file_name = queued_path
        .file_name()
        .ok_or_else(|| format!("invalid queued job path {}", queued_path.display()))?;
    let running = unique_path(&root.join(RUNNING_DIR).join(file_name));
    fs::rename(queued_path, &running).map_err(|e| {
        format!(
            "claim deferred job {} -> {}: {e}",
            queued_path.display(),
            running.display()
        )
    })?;
    Ok(running)
}

async fn run_claimed_job(
    state: SharedState,
    root: &Path,
    running_path: PathBuf,
) -> Result<(), String> {
    let started_at = now_secs();
    let fallback_id = running_path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .map(sanitize_job_id)
        .unwrap_or_else(|| format!("job_{}", Uuid::new_v4().simple()));
    let raw = match fs::read_to_string(&running_path) {
        Ok(raw) => raw,
        Err(e) => {
            let error = format!("read deferred job {}: {e}", running_path.display());
            fail_unparsed_job(root, &fallback_id, &running_path, started_at, &error)?;
            return Err(error);
        }
    };
    let spec: DeferredJobSpec = match serde_json::from_str(&raw) {
        Ok(spec) => spec,
        Err(e) => {
            let error = format!("parse deferred job {}: {e}", running_path.display());
            fail_unparsed_job(root, &fallback_id, &running_path, started_at, &error)?;
            return Err(error);
        }
    };
    let id = job_id(&spec, &running_path);
    let kind = kind_name(&spec.kind);

    tracing::info!(
        id = %id,
        kind,
        source = %running_path.display(),
        "starting deferred job"
    );

    let outcome = execute_deferred_job(state, root, &id, &spec.kind).await;
    let completed_at = now_secs();
    match outcome {
        Ok(output) => {
            write_result(
                root,
                DONE_DIR,
                &id,
                json!({
                    "id": id,
                    "kind": kind,
                    "status": "completed",
                    "started_at": started_at,
                    "completed_at": completed_at,
                    "source": running_path,
                    "output": output,
                }),
            )?;
            finish_job_file(root, DONE_DIR, &id, &running_path)?;
            Ok(())
        }
        Err(error) => {
            write_result(
                root,
                FAILED_DIR,
                &id,
                json!({
                    "id": id,
                    "kind": kind,
                    "status": "failed",
                    "started_at": started_at,
                    "completed_at": completed_at,
                    "source": running_path,
                    "error": error,
                }),
            )?;
            finish_job_file(root, FAILED_DIR, &id, &running_path)?;
            Err(error)
        }
    }
}

fn fail_unparsed_job(
    root: &Path,
    id: &str,
    running_path: &Path,
    started_at: u64,
    error: &str,
) -> Result<(), String> {
    write_result(
        root,
        FAILED_DIR,
        id,
        json!({
            "id": id,
            "kind": "unknown",
            "status": "failed",
            "started_at": started_at,
            "completed_at": now_secs(),
            "source": running_path,
            "error": error,
        }),
    )?;
    finish_job_file(root, FAILED_DIR, id, running_path)
}

async fn execute_deferred_job(
    state: SharedState,
    root: &Path,
    id: &str,
    kind: &DeferredJobKind,
) -> Result<Value, String> {
    match kind {
        DeferredJobKind::SdapiTxt2img { body } => {
            response_to_value(post_txt2img(State(state), Json(body.clone())).await).await
        }
        DeferredJobKind::SdapiImg2img { body } => {
            response_to_value(post_img2img(State(state), Json(body.clone())).await).await
        }
        DeferredJobKind::HttpPost { endpoint, body } => {
            execute_http_post(state, endpoint, body.clone()).await
        }
        DeferredJobKind::TrainingCommand {
            argv,
            cwd,
            env,
            run_id,
            target_model,
            artifact,
        } => {
            execute_training_command(
                state,
                root,
                id,
                argv,
                cwd.as_deref(),
                env,
                run_id.as_deref(),
                target_model.as_deref(),
                artifact.as_deref(),
            )
            .await
        }
    }
}

async fn execute_http_post(
    state: SharedState,
    endpoint: &str,
    mut body: Value,
) -> Result<Value, String> {
    if let Some(obj) = body.as_object_mut() {
        obj.insert("stream".to_string(), Value::Bool(false));
    }
    match endpoint {
        "/sdapi/v1/txt2img" => {
            let request = serde_json::from_value::<SdGenerationRequest>(body)
                .map_err(|e| format!("invalid txt2img body: {e}"))?;
            response_to_value(post_txt2img(State(state), Json(request)).await).await
        }
        "/sdapi/v1/img2img" => {
            let request = serde_json::from_value::<SdGenerationRequest>(body)
                .map_err(|e| format!("invalid img2img body: {e}"))?;
            response_to_value(post_img2img(State(state), Json(request)).await).await
        }
        "/v1/chat/completions" => {
            let mut request = serde_json::from_value::<ChatRequest>(body)
                .map_err(|e| format!("invalid chat completion body: {e}"))?;
            request.stream = false;
            let generated = execute_blocking_chat(state, request)
                .await
                .map_err(|error| error.to_string())?;
            Ok(openai_chat_completion_response_with_tool_calls_json(
                &generated.req_id,
                generated.created,
                &generated.model,
                &generated.text,
                &generated.tool_calls,
                &generated.done,
                generated.request_max_tokens,
            ))
        }
        "/v1/responses" => {
            let mut request = serde_json::from_value::<ResponsesRequest>(body)
                .map_err(|e| format!("invalid responses body: {e}"))?;
            request.stream = false;
            execute_responses(state, request)
                .await
                .map_err(|error| error.to_string())
        }
        other => Err(format!(
            "unsupported deferred http_post endpoint {other:?}; supported endpoints are /sdapi/v1/txt2img, /sdapi/v1/img2img, /v1/chat/completions, /v1/responses"
        )),
    }
}

async fn response_to_value(response: Response) -> Result<Value, String> {
    let status = response.status();
    let max = response_max_bytes();
    let bytes = to_bytes(response.into_body(), max)
        .await
        .map_err(|e| format!("read deferred route response body: {e}"))?;
    let body = serde_json::from_slice::<Value>(&bytes).unwrap_or_else(|_| {
        json!({
            "raw": String::from_utf8_lossy(&bytes),
        })
    });
    if status.is_success() {
        Ok(body)
    } else {
        Err(json!({
            "status": status.as_u16(),
            "body": body,
        })
        .to_string())
    }
}

async fn execute_training_command(
    state: SharedState,
    root: &Path,
    job_id: &str,
    argv: &[String],
    cwd: Option<&Path>,
    extra_env: &BTreeMap<String, String>,
    requested_run_id: Option<&str>,
    target_model: Option<&str>,
    artifact: Option<&str>,
) -> Result<Value, String> {
    if !deferred_command_jobs_enabled() {
        return Err(
            "training_command deferred jobs are disabled by HIPFIRE_DEFERRED_ALLOW_COMMANDS=0"
                .to_string(),
        );
    }
    if argv.is_empty() || argv[0].trim().is_empty() {
        return Err("training_command requires a non-empty argv".to_string());
    }

    let run_id = requested_run_id
        .filter(|id| !id.trim().is_empty())
        .map(sanitize_job_id)
        .unwrap_or_else(|| sanitize_job_id(job_id));
    let run_dir = state.training_runs_dir.join(&run_id);
    fs::create_dir_all(&run_dir)
        .map_err(|e| format!("create training run dir {}: {e}", run_dir.display()))?;
    write_training_status(
        &run_dir,
        &run_id,
        "training",
        target_model,
        artifact,
        None,
        None,
    )?;
    append_training_event(
        &run_dir,
        json!({
            "type": "command_started",
            "timestamp": now_secs().to_string(),
            "argv": argv,
            "cwd": cwd.map(|path| path.display().to_string()),
        }),
    )?;

    let logs_dir = root.join(LOGS_DIR);
    fs::create_dir_all(&logs_dir)
        .map_err(|e| format!("create deferred logs dir {}: {e}", logs_dir.display()))?;
    let stdout_path = logs_dir.join(format!("{job_id}.stdout.log"));
    let stderr_path = logs_dir.join(format!("{job_id}.stderr.log"));
    let stdout = fs::File::create(&stdout_path)
        .map_err(|e| format!("create stdout log {}: {e}", stdout_path.display()))?;
    let stderr = fs::File::create(&stderr_path)
        .map_err(|e| format!("create stderr log {}: {e}", stderr_path.display()))?;

    let mut command = Command::new(&argv[0]);
    command.args(&argv[1..]);
    if let Some(cwd) = cwd {
        command.current_dir(cwd);
    }
    command.envs(extra_env);
    command.stdout(Stdio::from(stdout));
    command.stderr(Stdio::from(stderr));

    let status = command.status().await.map_err(|e| {
        let message = format!("spawn training command {:?}: {e}", argv[0]);
        let _ = write_training_status(
            &run_dir,
            &run_id,
            "failed",
            target_model,
            artifact,
            Some(&message),
            Some(now_secs()),
        );
        message
    })?;
    let exit_code = status.code();
    let output = json!({
        "run_id": run_id,
        "exit_code": exit_code,
        "success": status.success(),
        "stdout": stdout_path,
        "stderr": stderr_path,
    });

    if status.success() {
        write_training_status(
            &run_dir,
            &run_id,
            "completed",
            target_model,
            artifact,
            None,
            Some(now_secs()),
        )?;
        append_training_event(
            &run_dir,
            json!({
                "type": "command_completed",
                "timestamp": now_secs().to_string(),
                "exit_code": exit_code,
                "stdout": stdout_path,
                "stderr": stderr_path,
            }),
        )?;
        Ok(output)
    } else {
        let message = format!("training command exited with status {status}");
        write_training_status(
            &run_dir,
            &run_id,
            "failed",
            target_model,
            artifact,
            Some(&message),
            Some(now_secs()),
        )?;
        append_training_event(
            &run_dir,
            json!({
                "type": "command_failed",
                "timestamp": now_secs().to_string(),
                "exit_code": exit_code,
                "stdout": stdout_path,
                "stderr": stderr_path,
                "message": message,
            }),
        )?;
        Err(format!("{message}; output={output}"))
    }
}

fn write_training_status(
    run_dir: &Path,
    run_id: &str,
    status: &str,
    target_model: Option<&str>,
    artifact: Option<&str>,
    error: Option<&str>,
    completed_at: Option<u64>,
) -> Result<(), String> {
    let now = now_secs();
    let mut value = json!({
        "id": run_id,
        "kind": "deferred_training_command",
        "status": status,
        "target_model": target_model,
        "artifact": artifact,
        "updated_at": now.to_string(),
        "run_dir": run_dir,
        "progress": {
            "phase": status,
        },
    });
    if status == "training" {
        value["started_at"] = json!(now.to_string());
    }
    if let Some(completed_at) = completed_at {
        value["completed_at"] = json!(completed_at.to_string());
    }
    if let Some(error) = error {
        value["last_error"] = json!({
            "level": "error",
            "message": error,
            "phase": status,
            "event_type": "deferred_training_command",
        });
    }
    fs::write(
        run_dir.join(hipfire_operator::training::STATUS_FILE),
        serde_json::to_vec_pretty(&value).map_err(|e| format!("serialize training status: {e}"))?,
    )
    .map_err(|e| format!("write training status {}: {e}", run_dir.display()))
}

fn append_training_event(run_dir: &Path, event: Value) -> Result<(), String> {
    use std::io::Write as _;
    let path = run_dir.join(hipfire_operator::training::EVENTS_FILE);
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .map_err(|e| format!("open training events {}: {e}", path.display()))?;
    writeln!(
        file,
        "{}",
        serde_json::to_string(&event).map_err(|e| format!("serialize training event: {e}"))?
    )
    .map_err(|e| format!("write training event {}: {e}", path.display()))
}

fn write_result(root: &Path, dir: &str, id: &str, value: Value) -> Result<(), String> {
    let path = unique_path(&root.join(dir).join(format!("{id}.result.json")));
    fs::write(
        &path,
        serde_json::to_vec_pretty(&value).map_err(|e| format!("serialize job result: {e}"))?,
    )
    .map_err(|e| format!("write deferred job result {}: {e}", path.display()))
}

fn finish_job_file(root: &Path, dir: &str, id: &str, running_path: &Path) -> Result<(), String> {
    let dest = unique_path(&root.join(dir).join(format!("{id}.job.json")));
    fs::rename(running_path, &dest).map_err(|e| {
        format!(
            "finish deferred job {} -> {}: {e}",
            running_path.display(),
            dest.display()
        )
    })
}

fn job_id(spec: &DeferredJobSpec, path: &Path) -> String {
    spec.id
        .as_deref()
        .filter(|id| !id.trim().is_empty())
        .map(sanitize_job_id)
        .unwrap_or_else(|| {
            path.file_stem()
                .and_then(|stem| stem.to_str())
                .map(sanitize_job_id)
                .unwrap_or_else(|| format!("job_{}", Uuid::new_v4().simple()))
        })
}

fn sanitize_job_id(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    for ch in raw.chars() {
        if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.') {
            out.push(ch);
        } else {
            out.push('_');
        }
    }
    let out = out.trim_matches(|ch| matches!(ch, '_' | '.')).to_string();
    if out.is_empty() {
        format!("job_{}", Uuid::new_v4().simple())
    } else {
        out
    }
}

fn kind_name(kind: &DeferredJobKind) -> &'static str {
    match kind {
        DeferredJobKind::SdapiTxt2img { .. } => "sdapi_txt2img",
        DeferredJobKind::SdapiImg2img { .. } => "sdapi_img2img",
        DeferredJobKind::HttpPost { .. } => "http_post",
        DeferredJobKind::TrainingCommand { .. } => "training_command",
    }
}

fn unique_path(path: &Path) -> PathBuf {
    if !path.exists() {
        return path.to_path_buf();
    }
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let stem = path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("job");
    let ext = path.extension().and_then(|ext| ext.to_str());
    for attempt in 1.. {
        let name = match ext {
            Some(ext) => format!("{stem}-{attempt}.{ext}"),
            None => format!("{stem}-{attempt}"),
        };
        let candidate = parent.join(name);
        if !candidate.exists() {
            return candidate;
        }
    }
    unreachable!()
}

fn count_json_files(dir: &Path) -> usize {
    list_json_files(dir).map(|paths| paths.len()).unwrap_or(0)
}

fn count_terminal_job_files(dir: &Path) -> usize {
    list_json_files(dir)
        .map(|paths| {
            paths
                .into_iter()
                .filter(|path| {
                    path.file_name()
                        .and_then(|name| name.to_str())
                        .is_some_and(|name| name.ends_with(".job.json"))
                })
                .count()
        })
        .unwrap_or(0)
}

fn response_max_bytes() -> usize {
    std::env::var("HIPFIRE_DEFERRED_RESPONSE_MAX_BYTES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(512 * 1024 * 1024)
}

fn deferred_jobs_enabled() -> bool {
    !matches!(
        std::env::var("HIPFIRE_DEFERRED_JOBS")
            .unwrap_or_else(|_| "1".to_string())
            .trim()
            .to_ascii_lowercase()
            .as_str(),
        "0" | "false" | "off" | "no"
    )
}

fn deferred_command_jobs_enabled() -> bool {
    !matches!(
        std::env::var("HIPFIRE_DEFERRED_ALLOW_COMMANDS")
            .unwrap_or_else(|_| "1".to_string())
            .trim()
            .to_ascii_lowercase()
            .as_str(),
        "0" | "false" | "off" | "no"
    )
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_sdapi_deferred_job() {
        let spec: DeferredJobSpec = serde_json::from_value(json!({
            "id": "image-1",
            "kind": "sdapi_txt2img",
            "body": {
                "prompt": "a test image",
                "steps": 1,
                "width": 64,
                "height": 64
            }
        }))
        .unwrap();
        assert_eq!(spec.id.as_deref(), Some("image-1"));
        assert!(matches!(spec.kind, DeferredJobKind::SdapiTxt2img { .. }));
    }

    #[test]
    fn sanitizes_job_ids_for_paths() {
        assert_eq!(sanitize_job_id("../run id"), "run_id");
        assert!(sanitize_job_id("///").starts_with("job_"));
    }

    #[test]
    fn health_counts_deferred_job_dirs() {
        let root = temp_root("deferred-health");
        fs::create_dir_all(root.join(QUEUED_DIR)).unwrap();
        fs::create_dir_all(root.join(DONE_DIR)).unwrap();
        fs::write(root.join(QUEUED_DIR).join("a.json"), "{}").unwrap();
        fs::write(root.join(DONE_DIR).join("b.job.json"), "{}").unwrap();
        fs::write(root.join(DONE_DIR).join("b.result.json"), "{}").unwrap();

        let health = deferred_jobs_health_json_at(&root);
        assert_eq!(health["queued"], 1);
        assert_eq!(health["done"], 1);

        let _ = fs::remove_dir_all(root);
    }

    #[tokio::test]
    async fn malformed_startup_job_moves_to_failed() {
        let root = temp_root("deferred-malformed");
        fs::create_dir_all(root.join(QUEUED_DIR)).unwrap();
        fs::write(root.join(QUEUED_DIR).join("bad.json"), "{not-json").unwrap();
        let state = crate::AppState::new(hipfire_config::HipfireConfig::default());

        let summary = run_startup_deferred_jobs(state, root.clone())
            .await
            .unwrap();

        assert_eq!(summary.queued, 1);
        assert_eq!(summary.completed, 0);
        assert_eq!(summary.failed, 1);
        assert_eq!(count_json_files(&root.join(QUEUED_DIR)), 0);
        assert_eq!(count_json_files(&root.join(RUNNING_DIR)), 0);
        assert_eq!(count_terminal_job_files(&root.join(FAILED_DIR)), 1);
        assert_eq!(count_json_files(&root.join(FAILED_DIR)), 2);

        let _ = fs::remove_dir_all(root);
    }

    fn temp_root(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "hipfire-{name}-{}-{}",
            std::process::id(),
            Uuid::new_v4().simple()
        ))
    }
}
