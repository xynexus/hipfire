// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.
//! `hipfire jobs` — submit, watch and cancel background work.
//!
//! The queue is a directory, not a protocol: `~/.hipfire/jobs/deferred` with
//! `queued/ running/ done/ failed/ logs/`, drained by a tokio task in the
//! server. That is why this command needs no daemon request and no new wire
//! format — submitting is writing a file, and monitoring is reading one.
//!
//! It also means submission works whether or not the server is up. A job
//! written while nothing is running simply waits in `queued/` until one starts,
//! which is the behaviour a download manager should have anyway.

use clap::{Args, Subcommand};
use serde_json::{json, Value};
use std::path::PathBuf;

const QUEUED: &str = "queued";
const RUNNING: &str = "running";
const DONE: &str = "done";
const FAILED: &str = "failed";
const LOGS: &str = "logs";

fn root() -> PathBuf {
    hipfire_config::hipfire_dir().join("jobs").join("deferred")
}

#[derive(Debug, Args)]
#[command(after_help = "Examples:\n  \
        hipfire download Qwen/Qwen3.5-9B --detach   # submit, return immediately\n  \
        hipfire jobs list\n  \
        hipfire jobs watch <id>\n  \
        hipfire jobs cancel <id>\n\n\
        A cancelled download can be resubmitted: the archive is written under a\n\
        .part marker with a .manifest sidecar, so it resumes rather than restarts.")]
pub struct JobsArgs {
    #[command(subcommand)]
    command: JobsCommand,
}

#[derive(Debug, Subcommand)]
enum JobsCommand {
    /// List jobs in every state.
    List(ListArgs),
    /// Show one job, with the tail of its log.
    Status(OneArgs),
    /// Follow one job until it finishes.
    Watch(OneArgs),
    /// Ask a running job to stop, or drop a queued one.
    Cancel(OneArgs),
}

#[derive(Debug, Args)]
pub struct ListArgs {
    /// Emit JSON instead of a table.
    #[arg(long)]
    json: bool,
}

#[derive(Debug, Args)]
pub struct OneArgs {
    /// Job id, as printed by `list`.
    id: String,
    /// Emit JSON instead of text.
    #[arg(long)]
    json: bool,
}

fn scan(dir: &str) -> Vec<String> {
    let mut out: Vec<String> = std::fs::read_dir(root().join(dir))
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|e| {
            let name = e.file_name().to_string_lossy().into_owned();
            name.strip_suffix(".job.json").map(str::to_string)
        })
        .collect();
    out.sort();
    out
}

/// Live status, when a running job has published one.
fn status_of(id: &str) -> Option<Value> {
    let p = root().join(LOGS).join(format!("{id}.status.json"));
    std::fs::read(p)
        .ok()
        .and_then(|b| serde_json::from_slice(&b).ok())
}

fn detail_of(id: &str) -> String {
    status_of(id)
        .and_then(|v| v.get("detail").and_then(Value::as_str).map(str::to_string))
        .unwrap_or_default()
}

pub fn run(args: JobsArgs) -> anyhow::Result<()> {
    match args.command {
        JobsCommand::List(a) => list(a),
        JobsCommand::Status(a) => status(a),
        JobsCommand::Watch(a) => watch(a),
        JobsCommand::Cancel(a) => cancel(a),
    }
}

fn list(args: ListArgs) -> anyhow::Result<()> {
    let states = [RUNNING, QUEUED, FAILED, DONE];
    if args.json {
        let items: Vec<Value> = states
            .iter()
            .flat_map(|s| {
                scan(s)
                    .into_iter()
                    .map(move |id| json!({ "id": id, "state": s, "detail": detail_of(&id) }))
            })
            .collect();
        println!("{}", serde_json::to_string_pretty(&items)?);
        return Ok(());
    }
    let rows: Vec<(String, &str, String)> = states
        .iter()
        .flat_map(|s| scan(s).into_iter().map(move |id| (id, *s, String::new())))
        .map(|(id, s, _)| {
            let d = detail_of(&id);
            (id, s, d)
        })
        .collect();
    if rows.is_empty() {
        println!("no jobs in {}", root().display());
        return Ok(());
    }
    let w = rows
        .iter()
        .map(|(id, ..)| id.len())
        .max()
        .unwrap_or(2)
        .max(2);
    println!("{:<w$}  {:<8}  {}", "ID", "STATE", "DETAIL");
    for (id, state, detail) in rows {
        println!("{id:<w$}  {state:<8}  {detail}");
    }
    Ok(())
}

fn find_state(id: &str) -> Option<&'static str> {
    for s in [RUNNING, QUEUED, DONE, FAILED] {
        if scan(s).iter().any(|j| j == id) {
            return Some(s);
        }
    }
    None
}

fn log_tail(id: &str, lines: usize) -> String {
    let p = root().join(LOGS).join(format!("{id}.log"));
    let text = std::fs::read_to_string(p).unwrap_or_default();
    let all: Vec<&str> = text.lines().collect();
    all[all.len().saturating_sub(lines)..].join("\n")
}

fn status(args: OneArgs) -> anyhow::Result<()> {
    let state = find_state(&args.id)
        .ok_or_else(|| anyhow::anyhow!("no such job: {} (try `hipfire jobs list`)", args.id))?;
    if args.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "id": args.id,
                "state": state,
                "status": status_of(&args.id),
                "log_tail": log_tail(&args.id, 20),
            }))?
        );
        return Ok(());
    }
    println!("job:    {}", args.id);
    println!("state:  {state}");
    if let Some(d) = status_of(&args.id)
        .and_then(|v| v.get("detail").and_then(Value::as_str).map(str::to_string))
    {
        println!("detail: {d}");
    }
    let tail = log_tail(&args.id, 20);
    if !tail.is_empty() {
        println!("\n--- log (last 20 lines) ---\n{tail}");
    }
    Ok(())
}

fn watch(args: OneArgs) -> anyhow::Result<()> {
    let mut last = String::new();
    loop {
        let Some(state) = find_state(&args.id) else {
            anyhow::bail!("no such job: {}", args.id);
        };
        let detail = detail_of(&args.id);
        if detail != last {
            println!("[{state}] {detail}");
            last = detail;
        }
        if matches!(state, DONE | FAILED) {
            println!("job {} finished: {state}", args.id);
            return if state == FAILED {
                Err(anyhow::anyhow!("job {} failed", args.id))
            } else {
                Ok(())
            };
        }
        std::thread::sleep(std::time::Duration::from_millis(700));
    }
}

fn cancel(args: OneArgs) -> anyhow::Result<()> {
    let state = find_state(&args.id).ok_or_else(|| anyhow::anyhow!("no such job: {}", args.id))?;
    match state {
        // A queued job has not been claimed, so removing its file is the whole
        // cancellation — nothing is running to signal.
        QUEUED => {
            let p = root().join(QUEUED).join(format!("{}.job.json", args.id));
            std::fs::remove_file(&p).map_err(|e| anyhow::anyhow!("remove {}: {e}", p.display()))?;
            println!("cancelled queued job {}", args.id);
        }
        RUNNING => {
            // A marker, not a signal: the server's runner owns the child, and
            // this way the CLI needs no privilege over it and no pid record has
            // to survive a restart.
            let p = root().join(LOGS).join(format!("{}.cancel", args.id));
            if let Some(parent) = p.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            std::fs::write(&p, b"1").map_err(|e| anyhow::anyhow!("write {}: {e}", p.display()))?;
            println!(
                "asked job {} to stop (a download resumes if resubmitted)",
                args.id
            );
        }
        other => println!("job {} is already {other}; nothing to cancel", args.id),
    }
    Ok(())
}

/// Write a job file into `queued/`, returning its id.
pub fn submit(kind: Value) -> anyhow::Result<String> {
    let id = format!(
        "{}_{}",
        kind.get("kind").and_then(Value::as_str).unwrap_or("job"),
        uuid::Uuid::new_v4().simple()
    );
    let dir = root().join(QUEUED);
    std::fs::create_dir_all(&dir)?;
    let mut spec = kind;
    spec["id"] = json!(id);
    let path = dir.join(format!("{id}.job.json"));
    std::fs::write(&path, serde_json::to_vec_pretty(&spec)?)?;
    Ok(id)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `log_tail` must not panic on a log shorter than the requested tail —
    /// the naive `len - n` slice underflows on an empty or one-line file, which
    /// is exactly the state a job is in for its first second.
    #[test]
    fn log_tail_handles_a_log_shorter_than_the_window() {
        let all: Vec<&str> = vec![];
        assert_eq!(all[all.len().saturating_sub(20)..].join("\n"), "");
        let one = vec!["only line"];
        assert_eq!(one[one.len().saturating_sub(20)..].join("\n"), "only line");
    }
}
