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
use hipfire_operator::jobs::{
    cancel_job, job_log_tail, job_status, list_jobs, CancelOutcome, DONE, FAILED, QUEUED,
};
use serde_json::{json, Value};
use std::path::PathBuf;

fn root() -> PathBuf {
    hipfire_operator::jobs::jobs_dir(&hipfire_config::hipfire_dir())
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

fn status_of(id: &str) -> Option<Value> {
    job_status(&root(), id)
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
    let jobs = list_jobs(&root());
    if args.json {
        println!("{}", serde_json::to_string_pretty(&jobs)?);
        return Ok(());
    }
    if jobs.is_empty() {
        println!("no jobs in {}", root().display());
        return Ok(());
    }
    let w = jobs.iter().map(|j| j.id.len()).max().unwrap_or(2).max(2);
    let k = jobs.iter().map(|j| j.kind.len()).max().unwrap_or(4).max(4);
    println!("{:<w$}  {:<k$}  {:<8}  {}", "ID", "KIND", "STATE", "DETAIL");
    for j in jobs {
        println!(
            "{:<w$}  {:<k$}  {:<8}  {}",
            j.id,
            j.kind,
            j.state,
            if j.detail.is_empty() {
                j.label.clone()
            } else {
                j.detail.clone()
            }
        );
    }
    Ok(())
}

fn find_state(id: &str) -> Option<&'static str> {
    hipfire_operator::jobs::find_state(&root(), id)
}

fn log_tail(id: &str, lines: usize) -> String {
    job_log_tail(&root(), id, lines)
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
    match cancel_job(&root(), &args.id).map_err(|e| anyhow::anyhow!(e))? {
        CancelOutcome::DroppedQueued => println!("cancelled queued job {}", args.id),
        CancelOutcome::AskedToStop => println!(
            "asked job {} to stop (a download resumes if resubmitted)",
            args.id
        ),
        CancelOutcome::AlreadyFinished(state) => {
            println!("job {} is already {state}; nothing to cancel", args.id)
        }
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
