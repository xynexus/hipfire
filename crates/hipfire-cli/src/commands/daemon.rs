use std::{
    fs::{self, OpenOptions},
    path::PathBuf,
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
            "hipfire already running: {}",
            status.summary_line(&paths, &effective)
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

    print!("started hipfire serve pid {}", child.id());
    if let Some(model) = effective.config.default_model.as_deref() {
        print!(" model {model}");
    }
    println!(" log {}", paths.serve_log.display());

    if args.wait_secs == 0 {
        println!("health: not waited; check with `hipfire status`");
        return Ok(());
    }

    match wait_for_health(&effective, Duration::from_secs(args.wait_secs)).await {
        Ok(()) => {
            println!("health ok at {}", health_url(&effective));
            Ok(())
        }
        Err(err) => {
            println!("health not ready yet: {err}");
            println!("tail the log with: tail -f {}", paths.serve_log.display());
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
    println!("{}", status.summary_line(&paths, &loaded));
    if let Some(text) = status.health_text {
        println!("health: {text}");
    }
    println!("pid file: {}", paths.serve_pid.display());
    println!("log: {}", paths.serve_log.display());
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
    fn summary_line(&self, paths: &DaemonPaths, loaded: &LoadedConfig) -> String {
        let url = health_url(loaded);
        match (self.pid, self.pid_alive, self.health_ok) {
            (Some(pid), true, true) => format!("online pid {pid} at {url}"),
            (Some(pid), true, false) => format!("pid {pid} alive, health not ready at {url}"),
            (Some(_pid), false, true) => {
                format!(
                    "health ok at {url}, but {} is stale",
                    paths.serve_pid.display()
                )
            }
            (Some(pid), false, false) => format!("offline, stale pid {pid}"),
            (None, _, true) => format!("online at {url}, pid file missing"),
            (None, _, false) => format!("offline at {url}"),
        }
    }
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
    let host = match loaded.config.host.as_str() {
        "" | "0.0.0.0" => "127.0.0.1",
        "::" => "::1",
        other => other,
    };
    format!("http://{}:{}/health", host, loaded.config.port)
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
