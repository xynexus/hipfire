use std::{
    fmt::Write as _,
    fs::{self, OpenOptions},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    time::{Duration, Instant},
};

use anyhow::{anyhow, Context, Result};
use clap::Args;
use hipfire_config::{hipfire_dir, ConfigLayer, ConfigLayerKind, LoadedConfig};
use serde_json::json;

#[derive(Debug, Args)]
#[command(
    after_help = "Examples:\n  hipfire start\n  hipfire start --model Qwen3.5-30B-A3B --port 11435\n  hipfire start --host 0.0.0.0\n"
)]
pub struct StartArgs {
    /// Override bind host for the background server.
    #[arg(long)]
    pub host: Option<String>,
    /// Override bind port for the background server.
    #[arg(long, short)]
    pub port: Option<u16>,
    /// Pre-load a model on startup by name, shorthand, alias, or path.
    #[arg(long, short)]
    pub model: Option<String>,
    /// Log full raw chat requests and raw model replies.
    #[arg(long)]
    pub debug_chat: bool,
    /// Seconds to wait for /health before returning. Default 0 returns immediately.
    #[arg(long, default_value_t = 0)]
    pub wait_secs: u64,
}

#[derive(Debug, Args)]
#[command(after_help = "Examples:\n  hipfire stop\n  hipfire stop --force\n")]
pub struct StopArgs {
    /// Skip the graceful wait and send SIGKILL immediately.
    #[arg(long, short)]
    pub force: bool,
}

#[derive(Debug, Args)]
#[command(after_help = "Examples:\n  hipfire restart\n  hipfire restart --model Qwen3.5-30B-A3B\n")]
pub struct RestartArgs {
    /// Override bind host for the restarted background server.
    #[arg(long)]
    pub host: Option<String>,
    /// Override bind port for the restarted background server.
    #[arg(long, short)]
    pub port: Option<u16>,
    /// Pre-load a model on startup by name, shorthand, alias, or path.
    #[arg(long, short)]
    pub model: Option<String>,
    /// Log full raw chat requests and raw model replies.
    #[arg(long)]
    pub debug_chat: bool,
    /// Seconds to wait for /health before returning. Default 0 returns immediately.
    #[arg(long, default_value_t = 0)]
    pub wait_secs: u64,
}

#[derive(Debug, Args)]
#[command(after_help = "Examples:\n  hipfire status\n")]
pub struct StatusArgs {}

pub async fn start(args: StartArgs, loaded: LoadedConfig) -> Result<()> {
    let paths = DaemonPaths::new()?;
    let effective = effective_config(
        &loaded,
        args.host.as_deref(),
        args.port,
        args.model.as_deref(),
    );
    let status = current_status(&paths, &effective).await;
    if status.pid_alive || status.health_ok {
        println!(
            "{}",
            status.render("Hipfire server is already running.", &paths, &effective)
        );
        return Ok(());
    }

    fs::create_dir_all(&paths.root)?;
    let log = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&paths.serve_log)
        .with_context(|| format!("open {}", paths.serve_log.display()))?;
    let log_err = log.try_clone()?;

    let exe = std::env::current_exe().context("resolve current hipfire executable")?;
    let mut command = Command::new(exe);
    command.arg("serve");
    if let Some(host) = args.host {
        command.arg("--host").arg(host);
    }
    if let Some(port) = args.port {
        command.arg("--port").arg(port.to_string());
    }
    if let Some(model) = args.model {
        command.arg("--model").arg(model);
    }
    if args.debug_chat {
        command.arg("--debug-chat");
    }
    command
        .stdin(Stdio::null())
        .stdout(Stdio::from(log))
        .stderr(Stdio::from(log_err));
    detach_process_group(&mut command);

    let child = command.spawn().context("spawn `hipfire serve`")?;
    fs::write(&paths.serve_pid, child.id().to_string())
        .with_context(|| format!("write {}", paths.serve_pid.display()))?;

    if args.wait_secs == 0 {
        println!("{}", render_start_summary(child.id(), &paths, &effective));
        return Ok(());
    }

    match wait_for_health(&effective, Duration::from_secs(args.wait_secs)).await {
        Ok(()) => {
            let status = current_status(&paths, &effective).await;
            println!(
                "{}",
                status.render("Hipfire server is ready.", &paths, &effective)
            );
            Ok(())
        }
        Err(err) => {
            let status = current_status(&paths, &effective).await;
            let mut output = status.render(
                "Hipfire server started, but is not ready yet.",
                &paths,
                &effective,
            );
            let _ = write!(
                output,
                "\n\n  Wait error  {err}\n  Next step   tail -f {}",
                human_path(&paths.serve_log)
            );
            println!("{output}");
            Ok(())
        }
    }
}

pub async fn stop(args: StopArgs, loaded: LoadedConfig) -> Result<()> {
    let paths = DaemonPaths::new()?;
    let status = current_status(&paths, &loaded).await;
    let Some(pid) = status.pid else {
        println!("hipfire is not running: no {}", paths.serve_pid.display());
        return Ok(());
    };
    if !pid_alive(pid) {
        println!("hipfire serve pid {pid} is not alive; removing stale pid file");
        let _ = fs::remove_file(&paths.serve_pid);
        return Ok(());
    }

    if args.force {
        signal(pid, libc::SIGKILL)?;
        let _ = fs::remove_file(&paths.serve_pid);
        println!("sent SIGKILL to hipfire serve pid {pid}");
        return Ok(());
    }

    signal(pid, libc::SIGTERM)?;
    let deadline = Instant::now() + Duration::from_secs(5);
    while Instant::now() < deadline {
        if !pid_alive(pid) {
            let _ = fs::remove_file(&paths.serve_pid);
            println!("stopped hipfire serve pid {pid}");
            return Ok(());
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    signal(pid, libc::SIGKILL)?;
    let _ = fs::remove_file(&paths.serve_pid);
    println!("sent SIGKILL to hipfire serve pid {pid} after graceful stop timed out");
    Ok(())
}

pub async fn restart(args: RestartArgs, loaded: LoadedConfig) -> Result<()> {
    stop(StopArgs { force: false }, loaded.clone()).await?;
    start(
        StartArgs {
            host: args.host,
            port: args.port,
            model: args.model,
            debug_chat: args.debug_chat,
            wait_secs: args.wait_secs,
        },
        loaded,
    )
    .await
}

pub async fn status(_args: StatusArgs, loaded: LoadedConfig) -> Result<()> {
    let paths = DaemonPaths::new()?;
    let status = current_status(&paths, &loaded).await;
    println!("{}", status.render("Hipfire server", &paths, &loaded));
    Ok(())
}

#[derive(Clone)]
struct DaemonPaths {
    root: PathBuf,
    serve_pid: PathBuf,
    serve_log: PathBuf,
}

impl DaemonPaths {
    fn new() -> Result<Self> {
        let root = hipfire_dir();
        Ok(Self {
            serve_pid: root.join("serve.pid"),
            serve_log: root.join("serve.log"),
            root,
        })
    }
}

struct ServeStatus {
    pid: Option<u32>,
    pid_alive: bool,
    health_ok: bool,
    health_text: Option<String>,
}

impl ServeStatus {
    fn render(&self, title: &str, paths: &DaemonPaths, loaded: &LoadedConfig) -> String {
        let health = self
            .health_text
            .as_deref()
            .and_then(|text| serde_json::from_str::<serde_json::Value>(text).ok());
        // The front-end answers /health even when its inference worker has
        // crashed; surface that as `degraded` rather than a blanket `healthy`.
        let worker_down = health.as_ref().is_some_and(|h| {
            h.get("worker_alive") == Some(&serde_json::Value::Bool(false))
                || h.get("status").and_then(serde_json::Value::as_str) == Some("degraded")
        });
        let status_label = if worker_down {
            "degraded".to_string()
        } else {
            self.state_label().to_string()
        };
        let health_label = if worker_down {
            "degraded (inference worker down)".to_string()
        } else {
            self.health_label().to_string()
        };
        let mut fields = vec![
            ("Status", status_label),
            ("Address", server_url(loaded)),
            ("Process", self.process_label()),
            ("Health", health_label),
        ];
        if let Some(version) = health
            .as_ref()
            .and_then(|value| value.get("version"))
            .and_then(serde_json::Value::as_str)
        {
            fields.push(("Version", version.to_string()));
        }
        if let Some(model) = health.as_ref().and_then(health_model) {
            fields.push(("Model", model.to_string()));
        }
        fields.push(("PID file", human_path(&paths.serve_pid)));
        fields.push(("Log", human_path(&paths.serve_log)));

        let note = if worker_down {
            Some(format!(
                "The HTTP front-end is up but the inference worker has exited; \
                 requests fail until it respawns. See {}.",
                human_path(&paths.root.join("daemon.log"))
            ))
        } else {
            self.note(paths, loaded)
        };
        render_block(title, &fields, note.as_deref())
    }

    fn state_label(&self) -> &'static str {
        match (self.pid, self.pid_alive, self.health_ok) {
            (Some(_), true, true) | (None, _, true) => "online",
            (_, true, false) => "starting",
            (Some(_), false, true) => "degraded",
            (_, false, false) => "offline",
        }
    }

    fn process_label(&self) -> String {
        match (self.pid, self.pid_alive) {
            (Some(pid), true) => format!("{pid} (running)"),
            (Some(pid), false) => format!("{pid} (not running)"),
            (None, _) => "not found".to_string(),
        }
    }

    fn health_label(&self) -> &'static str {
        if self.health_ok {
            "healthy"
        } else if self.pid_alive {
            "not ready"
        } else {
            "unreachable"
        }
    }

    fn note(&self, paths: &DaemonPaths, loaded: &LoadedConfig) -> Option<String> {
        match (self.pid, self.pid_alive, self.health_ok) {
            (Some(_), true, true) => None,
            (Some(_), true, false) => Some(format!(
                "The process is running, but {} is not responding yet.",
                health_url(loaded)
            )),
            (Some(pid), false, true) => Some(format!(
                "The server is healthy, but {} still refers to process {pid}.",
                human_path(&paths.serve_pid)
            )),
            (Some(pid), false, false) => Some(format!(
                "The PID file refers to process {pid}, which is no longer running."
            )),
            (None, _, true) => Some(format!(
                "The server is healthy, but {} is missing.",
                human_path(&paths.serve_pid)
            )),
            (None, _, false) => Some(format!(
                "Could not reach the server. Start it with `hipfire start` or inspect {}.",
                human_path(&paths.serve_log)
            )),
        }
    }
}

fn health_model(value: &serde_json::Value) -> Option<&str> {
    value
        .get("active_model")
        .and_then(serde_json::Value::as_str)
        .or_else(|| value.get("model").and_then(serde_json::Value::as_str))
}

fn render_start_summary(pid: u32, paths: &DaemonPaths, loaded: &LoadedConfig) -> String {
    let mut fields = vec![
        ("Status", "starting".to_string()),
        ("Address", server_url(loaded)),
        ("Process", format!("{pid} (running)")),
        ("Health", "not checked".to_string()),
    ];
    if let Some(model) = loaded.config.default_model.as_deref() {
        fields.push(("Model", model.to_string()));
    }
    fields.push(("Log", human_path(&paths.serve_log)));
    render_block(
        "Hipfire server started.",
        &fields,
        Some("Run `hipfire status` to check readiness."),
    )
}

fn render_block(title: &str, fields: &[(&str, String)], note: Option<&str>) -> String {
    let width = fields
        .iter()
        .map(|(label, _)| label.len())
        .max()
        .unwrap_or_default();
    let mut output = String::from(title);
    if !fields.is_empty() {
        output.push('\n');
        for (label, value) in fields {
            let _ = write!(output, "\n  {label:<width$}  {value}");
        }
    }
    if let Some(note) = note {
        let _ = write!(output, "\n\n  {note}");
    }
    output
}

fn human_path(path: &Path) -> String {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .and_then(|home| {
            path.strip_prefix(home)
                .ok()
                .map(|relative| format!("~/{}", relative.display()))
        })
        .unwrap_or_else(|| path.display().to_string())
}

async fn current_status(paths: &DaemonPaths, loaded: &LoadedConfig) -> ServeStatus {
    let pid = fs::read_to_string(&paths.serve_pid)
        .ok()
        .and_then(|raw| raw.trim().parse::<u32>().ok());
    let pid_alive = pid.is_some_and(pid_alive);
    let (health_ok, health_text) = probe_health(loaded).await;
    ServeStatus {
        pid,
        pid_alive,
        health_ok,
        health_text,
    }
}

async fn wait_for_health(loaded: &LoadedConfig, timeout: Duration) -> Result<()> {
    let deadline = Instant::now() + timeout;
    loop {
        let (ok, text) = probe_health(loaded).await;
        if ok {
            return Ok(());
        }
        if Instant::now() >= deadline {
            return Err(anyhow!(
                "{} did not become healthy within {}s{}",
                health_url(loaded),
                timeout.as_secs(),
                text.as_deref()
                    .map(|text| format!("; last error: {text}"))
                    .unwrap_or_default()
            ));
        }
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
}

async fn probe_health(loaded: &LoadedConfig) -> (bool, Option<String>) {
    let client = match reqwest::Client::builder()
        .timeout(Duration::from_millis(700))
        .build()
    {
        Ok(client) => client,
        Err(err) => return (false, Some(err.to_string())),
    };
    match client.get(health_url(loaded)).send().await {
        Ok(response) if response.status().is_success() => {
            let text = response.text().await.ok();
            (true, text)
        }
        Ok(response) => {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            (false, Some(format!("HTTP {status}: {text}")))
        }
        Err(err) => (false, Some(err.to_string())),
    }
}

fn effective_config(
    loaded: &LoadedConfig,
    host: Option<&str>,
    port: Option<u16>,
    model: Option<&str>,
) -> LoadedConfig {
    let mut layer = ConfigLayer::new(ConfigLayerKind::Cli);
    if let Some(host) = host {
        layer.values.insert("host".to_string(), json!(host));
    }
    if let Some(port) = port {
        layer.values.insert("port".to_string(), json!(port));
    }
    if let Some(model) = model {
        layer
            .values
            .insert("default_model".to_string(), json!(model));
    }
    loaded.clone().with_additional_layer(layer)
}

fn health_url(loaded: &LoadedConfig) -> String {
    format!("{}/health", server_url(loaded))
}

fn server_url(loaded: &LoadedConfig) -> String {
    let host = match loaded.config.host.as_str() {
        "" | "0.0.0.0" => "127.0.0.1",
        "::" => "::1",
        other => other,
    };
    if host.contains(':') {
        format!("http://[{host}]:{}", loaded.config.port)
    } else {
        format!("http://{host}:{}", loaded.config.port)
    }
}

fn pid_alive(pid: u32) -> bool {
    if pid <= 1 {
        return false;
    }
    unsafe { libc::kill(pid as i32, 0) == 0 || *libc::__errno_location() == libc::EPERM }
}

fn signal(pid: u32, sig: i32) -> Result<()> {
    let rc = unsafe { libc::kill(pid as i32, sig) };
    if rc == 0 {
        Ok(())
    } else {
        Err(std::io::Error::last_os_error()).with_context(|| format!("signal pid {pid}"))
    }
}

fn detach_process_group(command: &mut Command) {
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        command.process_group(0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn paths() -> DaemonPaths {
        DaemonPaths {
            root: PathBuf::from("/tmp/hipfire-test"),
            serve_pid: PathBuf::from("/tmp/hipfire-test/serve.pid"),
            serve_log: PathBuf::from("/tmp/hipfire-test/serve.log"),
        }
    }

    fn loaded(host: &str, port: u16) -> LoadedConfig {
        let mut layer = ConfigLayer::new(ConfigLayerKind::Cli);
        layer.values.insert("host".to_string(), json!(host));
        layer.values.insert("port".to_string(), json!(port));
        hipfire_config::load_config_bundle().with_additional_layer(layer)
    }

    #[test]
    fn status_render_is_scannable_and_summarizes_health_json() {
        let status = ServeStatus {
            pid: Some(4242),
            pid_alive: true,
            health_ok: true,
            health_text: Some(
                json!({
                    "status": "ok",
                    "version": "v0.3.0-1-gdeadbeef",
                    "active_model": "Qwen3.5-9B"
                })
                .to_string(),
            ),
        };
        let output = status.render("Hipfire server", &paths(), &loaded("0.0.0.0", 11435));

        assert!(output.starts_with("Hipfire server\n\n  Status"));
        assert!(output.contains("online"));
        assert!(output.contains("http://127.0.0.1:11435"));
        assert!(output.contains("4242 (running)"));
        assert!(output.contains("v0.3.0-1-gdeadbeef"));
        assert!(output.contains("Qwen3.5-9B"));
        assert!(!output.contains("active_model"));
    }

    #[test]
    fn offline_status_recommends_the_next_action_without_raw_http_noise() {
        let status = ServeStatus {
            pid: None,
            pid_alive: false,
            health_ok: false,
            health_text: Some("error sending request for url".to_string()),
        };
        let output = status.render("Hipfire server", &paths(), &loaded("127.0.0.1", 11435));

        assert!(output.contains("offline"));
        assert!(output.contains("unreachable"));
        assert!(output.contains("`hipfire start`"));
        assert!(!output.contains("error sending request"));
    }

    #[test]
    fn start_summary_sets_expectations_and_includes_the_configured_model() {
        let base = loaded("127.0.0.1", 11435);
        let mut layer = ConfigLayer::new(ConfigLayerKind::Cli);
        layer
            .values
            .insert("default_model".to_string(), json!("Qwen3.5-9B"));
        let loaded = base.with_additional_layer(layer);
        let output = render_start_summary(4242, &paths(), &loaded);

        assert!(output.starts_with("Hipfire server started."));
        assert!(output.contains("starting"));
        assert!(output.contains("not checked"));
        assert!(output.contains("4242 (running)"));
        assert!(output.contains("Qwen3.5-9B"));
        assert!(output.contains("`hipfire status`"));
    }

    #[test]
    fn ipv6_health_url_uses_brackets() {
        let loaded = loaded("::", 11435);
        assert_eq!(server_url(&loaded), "http://[::1]:11435");
        assert_eq!(health_url(&loaded), "http://[::1]:11435/health");
    }
}
