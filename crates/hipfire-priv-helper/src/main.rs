// SPDX-License-Identifier: Apache-2.0
// hipfire — narrow privileged helper for `hipfire doctor --fix`.

use anyhow::Context;
use serde::Serialize;
use serde_json::{json, Value};
use std::ffi::CString;
use std::fs;
use std::path::Path;
use std::process::Command;

const RESCTRL_PATH: &str = "/sys/fs/resctrl";
const PERF_EVENT_PARANOID: &str = "/proc/sys/kernel/perf_event_paranoid";
const ALLOWED_MODULES: &[&str] = &["amd_uncore"];

#[derive(Debug, Serialize)]
struct HelperResponse {
    ok: bool,
    action: String,
    message: String,
    details: Value,
}

fn main() {
    let response = match run() {
        Ok(response) => response,
        Err(err) => HelperResponse {
            ok: false,
            action: "error".to_string(),
            message: err.to_string(),
            details: json!({}),
        },
    };
    println!(
        "{}",
        serde_json::to_string_pretty(&response).unwrap_or_else(|_| {
            "{\"ok\":false,\"action\":\"error\",\"message\":\"json encode failed\",\"details\":{}}"
                .to_string()
        })
    );
    if !response.ok {
        std::process::exit(1);
    }
}

fn run() -> anyhow::Result<HelperResponse> {
    let mut args = std::env::args().skip(1);
    match args.next().as_deref() {
        Some("probe") => Ok(probe()),
        Some("load-module") => {
            let module = args.next().context("load-module requires a module name")?;
            ensure_no_extra_args(args)?;
            load_module(&module)
        }
        Some("mount-resctrl") => {
            ensure_no_extra_args(args)?;
            mount_resctrl()
        }
        Some("set-perf-event-paranoid") => {
            let value = args
                .next()
                .context("set-perf-event-paranoid requires a value")?;
            ensure_no_extra_args(args)?;
            set_perf_event_paranoid(&value)
        }
        Some("-h" | "--help") | None => Ok(help_response()),
        Some(other) => anyhow::bail!("unknown helper command: {other}"),
    }
}

fn ensure_no_extra_args(mut args: impl Iterator<Item = String>) -> anyhow::Result<()> {
    if let Some(extra) = args.next() {
        anyhow::bail!("unexpected extra argument: {extra}");
    }
    Ok(())
}

fn help_response() -> HelperResponse {
    HelperResponse {
        ok: true,
        action: "help".to_string(),
        message: "usage: hipfire-priv-helper probe | load-module amd_uncore | mount-resctrl | set-perf-event-paranoid <value>".to_string(),
        details: json!({
            "allowed_modules": ALLOWED_MODULES,
            "allowed_perf_event_paranoid": [-1, 0, 1, 2, 3, 4],
        }),
    }
}

fn probe() -> HelperResponse {
    let uid = unsafe { libc::geteuid() };
    let amd_uncore_loaded = Path::new("/sys/module/amd_uncore").exists();
    let resctrl_mounted = is_mountpoint(RESCTRL_PATH);
    let perf_event_paranoid = fs::read_to_string(PERF_EVENT_PARANOID)
        .ok()
        .and_then(|s| s.trim().parse::<i32>().ok());
    HelperResponse {
        ok: true,
        action: "probe".to_string(),
        message: "privileged helper probe complete".to_string(),
        details: json!({
            "euid": uid,
            "is_root": uid == 0,
            "amd_uncore_loaded": amd_uncore_loaded,
            "resctrl_mounted": resctrl_mounted,
            "perf_event_paranoid": perf_event_paranoid,
        }),
    }
}

fn load_module(module: &str) -> anyhow::Result<HelperResponse> {
    if !ALLOWED_MODULES.contains(&module) {
        anyhow::bail!("module not allowed: {module}");
    }
    if Path::new(&format!("/sys/module/{module}")).exists() {
        return Ok(HelperResponse {
            ok: true,
            action: "load-module".to_string(),
            message: format!("kernel module already loaded: {module}"),
            details: json!({ "module": module, "already_loaded": true }),
        });
    }
    let modprobe = find_absolute_command(&["/usr/sbin/modprobe", "/sbin/modprobe"])
        .context("modprobe not found at /usr/sbin/modprobe or /sbin/modprobe")?;
    let output = Command::new(&modprobe)
        .arg(module)
        .output()
        .with_context(|| format!("spawn {}", modprobe.display()))?;
    if !output.status.success() {
        anyhow::bail!(
            "modprobe {module} failed with {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(HelperResponse {
        ok: true,
        action: "load-module".to_string(),
        message: format!("loaded kernel module: {module}"),
        details: json!({ "module": module }),
    })
}

fn mount_resctrl() -> anyhow::Result<HelperResponse> {
    if is_mountpoint(RESCTRL_PATH) {
        return Ok(HelperResponse {
            ok: true,
            action: "mount-resctrl".to_string(),
            message: "resctrl is already mounted".to_string(),
            details: json!({ "mountpoint": RESCTRL_PATH, "already_mounted": true }),
        });
    }
    fs::create_dir_all(RESCTRL_PATH).context("create /sys/fs/resctrl")?;
    let source = CString::new("resctrl")?;
    let target = CString::new(RESCTRL_PATH)?;
    let fstype = CString::new("resctrl")?;
    let rc = unsafe {
        libc::mount(
            source.as_ptr(),
            target.as_ptr(),
            fstype.as_ptr(),
            0,
            std::ptr::null(),
        )
    };
    if rc != 0 {
        return Err(std::io::Error::last_os_error()).context("mount resctrl");
    }
    Ok(HelperResponse {
        ok: true,
        action: "mount-resctrl".to_string(),
        message: "mounted resctrl".to_string(),
        details: json!({ "mountpoint": RESCTRL_PATH }),
    })
}

fn set_perf_event_paranoid(value: &str) -> anyhow::Result<HelperResponse> {
    let parsed: i32 = value
        .parse()
        .with_context(|| format!("invalid perf_event_paranoid value: {value}"))?;
    if !(-1..=4).contains(&parsed) {
        anyhow::bail!("perf_event_paranoid must be between -1 and 4, got {parsed}");
    }
    fs::write(PERF_EVENT_PARANOID, format!("{parsed}\n"))
        .with_context(|| format!("write {PERF_EVENT_PARANOID}"))?;
    Ok(HelperResponse {
        ok: true,
        action: "set-perf-event-paranoid".to_string(),
        message: format!("set perf_event_paranoid to {parsed}"),
        details: json!({ "path": PERF_EVENT_PARANOID, "value": parsed }),
    })
}

fn find_absolute_command(candidates: &[&str]) -> Option<std::path::PathBuf> {
    candidates
        .iter()
        .map(std::path::PathBuf::from)
        .find(|path| path.is_file())
}

fn is_mountpoint(path: &str) -> bool {
    fs::read_to_string("/proc/self/mountinfo")
        .map(|mountinfo| {
            mountinfo.lines().any(|line| {
                line.split_ascii_whitespace()
                    .nth(4)
                    .is_some_and(|mount| mount == path)
            })
        })
        .unwrap_or(false)
}
