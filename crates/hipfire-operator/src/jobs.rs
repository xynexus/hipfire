// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

//! Read side of the deferred job queue, shared by `hipfire jobs` and the TUI.
//!
//! The queue is a directory, not a protocol: `<root>/queued|running|done|failed`
//! hold `<id>.job.json`, and `<root>/logs` holds the child's `<id>.log`, the
//! `<id>.status.json` the server publishes as it runs, and the `<id>.cancel`
//! marker a client drops to ask it to stop. Everything here is therefore plain
//! filesystem reads — no daemon request, no wire format, and it works whether
//! or not the server is up.
//!
//! Submitting lives in the CLI instead: it needs a uuid, and this crate stays
//! serde-plus-filesystem by charter.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    fs,
    path::{Path, PathBuf},
};

pub const QUEUED: &str = "queued";
pub const RUNNING: &str = "running";
pub const DONE: &str = "done";
pub const FAILED: &str = "failed";
pub const LOGS: &str = "logs";

/// States in the order an operator wants to see them: what is happening now,
/// what is about to, then what went wrong, then the archive.
pub const STATES: [&str; 4] = [RUNNING, QUEUED, FAILED, DONE];

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct JobSummary {
    pub id: String,
    pub state: String,
    /// Job kind (`download`, `induct`, …), from the queued spec.
    #[serde(default)]
    pub kind: String,
    /// What the job is acting on — a repo id, a model source.
    #[serde(default)]
    pub label: String,
    /// Last non-empty line the child printed, as published by the runner.
    #[serde(default)]
    pub detail: String,
    #[serde(default)]
    pub updated_at: Option<u64>,
}

impl JobSummary {
    pub fn is_active(&self) -> bool {
        self.state == RUNNING
    }

    /// Percent complete, read back out of the progress line the child prints
    /// (`hub: 0.30/17.72 GB (2%) — …`).
    ///
    /// Parsed rather than published as a field because the fetcher already
    /// emits it for its own terminal output, and a job whose child reports
    /// nothing simply has no percentage to show.
    pub fn progress_percent(&self) -> Option<u8> {
        let open = self.detail.rfind('(')?;
        let close = self.detail[open..].find("%)")? + open;
        self.detail[open + 1..close].trim().parse().ok()
    }

    pub fn is_finished(&self) -> bool {
        self.state == DONE || self.state == FAILED
    }
}

/// The queue directory under a hipfire root (`~/.hipfire`).
pub fn jobs_dir(hipfire_root: &Path) -> PathBuf {
    hipfire_root.join("jobs").join("deferred")
}

fn ids_in(root: &Path, state: &str) -> Vec<String> {
    let mut out: Vec<String> = fs::read_dir(root.join(state))
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

/// Live status a running job has published, if any.
pub fn job_status(root: &Path, id: &str) -> Option<Value> {
    let path = root.join(LOGS).join(format!("{id}.status.json"));
    fs::read(path)
        .ok()
        .and_then(|bytes| serde_json::from_slice(&bytes).ok())
}

/// The queued spec, which is where a job's kind and target live before the
/// runner has published anything.
fn job_spec(root: &Path, id: &str, state: &str) -> Option<Value> {
    let path = root.join(state).join(format!("{id}.job.json"));
    fs::read(path)
        .ok()
        .and_then(|bytes| serde_json::from_slice(&bytes).ok())
}

fn str_field(value: &Option<Value>, key: &str) -> Option<String> {
    value
        .as_ref()?
        .get(key)
        .and_then(Value::as_str)
        .map(str::to_string)
}

fn summarize(root: &Path, id: &str, state: &str) -> JobSummary {
    let status = job_status(root, id);
    let spec = job_spec(root, id, state);
    // The kind prefixes the id (`download_<uuid>`), so a job whose spec has
    // been consumed still reports something better than blank.
    let kind = str_field(&status, "kind")
        .or_else(|| str_field(&spec, "kind"))
        .unwrap_or_else(|| id.split('_').next().unwrap_or_default().to_string());
    let label = str_field(&status, "label")
        .or_else(|| str_field(&spec, "repo"))
        .or_else(|| str_field(&spec, "source"))
        .unwrap_or_default();
    JobSummary {
        id: id.to_string(),
        state: state.to_string(),
        kind,
        label,
        detail: str_field(&status, "detail").unwrap_or_default(),
        updated_at: status
            .as_ref()
            .and_then(|v| v.get("updated_at"))
            .and_then(Value::as_u64),
    }
}

/// Every job in the queue, running first and archived last.
pub fn list_jobs(root: &Path) -> Vec<JobSummary> {
    STATES
        .iter()
        .flat_map(|state| {
            ids_in(root, state)
                .into_iter()
                .map(move |id| summarize(root, &id, state))
        })
        .collect()
}

pub fn find_state(root: &Path, id: &str) -> Option<&'static str> {
    STATES
        .into_iter()
        .find(|state| ids_in(root, state).iter().any(|found| found == id))
}

/// Last `lines` lines of a job's log, empty when it has not written one.
pub fn job_log_tail(root: &Path, id: &str, lines: usize) -> String {
    let text = fs::read_to_string(root.join(LOGS).join(format!("{id}.log"))).unwrap_or_default();
    let all: Vec<&str> = text.lines().collect();
    all[all.len().saturating_sub(lines)..].join("\n")
}

/// Outcome of asking a job to stop, so callers can word their own message.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CancelOutcome {
    /// Never claimed by a runner; its job file was deleted.
    DroppedQueued,
    /// Running: a cancel marker was written for the runner to notice.
    AskedToStop,
    /// Already `done` or `failed`.
    AlreadyFinished(&'static str),
}

/// Cancel by file, not by signal: the server's runner owns the child process,
/// so a client needs no privilege over it and no pid record has to survive a
/// server restart.
pub fn cancel_job(root: &Path, id: &str) -> Result<CancelOutcome, String> {
    let state = find_state(root, id).ok_or_else(|| format!("no such job: {id}"))?;
    match state {
        QUEUED => {
            let path = root.join(QUEUED).join(format!("{id}.job.json"));
            fs::remove_file(&path).map_err(|e| format!("remove {}: {e}", path.display()))?;
            Ok(CancelOutcome::DroppedQueued)
        }
        RUNNING => {
            let path = root.join(LOGS).join(format!("{id}.cancel"));
            if let Some(parent) = path.parent() {
                let _ = fs::create_dir_all(parent);
            }
            fs::write(&path, b"1").map_err(|e| format!("write {}: {e}", path.display()))?;
            Ok(CancelOutcome::AskedToStop)
        }
        other => Ok(CancelOutcome::AlreadyFinished(other)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("hipfire-jobs-test-{name}"));
        let _ = fs::remove_dir_all(&dir);
        for state in STATES.iter().chain([&LOGS]) {
            fs::create_dir_all(dir.join(state)).unwrap();
        }
        dir
    }

    fn write_job(root: &Path, state: &str, id: &str, spec: Value) {
        fs::write(
            root.join(state).join(format!("{id}.job.json")),
            serde_json::to_vec(&spec).unwrap(),
        )
        .unwrap();
    }

    #[test]
    fn a_queued_job_reports_its_kind_and_target_before_the_runner_publishes() {
        let root = scratch("queued");
        write_job(
            &root,
            QUEUED,
            "download_abc",
            serde_json::json!({ "kind": "download", "repo": "org/model" }),
        );
        let jobs = list_jobs(&root);
        assert_eq!(jobs.len(), 1);
        assert_eq!(jobs[0].kind, "download");
        assert_eq!(jobs[0].label, "org/model");
        assert_eq!(jobs[0].state, QUEUED);
        assert!(!jobs[0].is_active());
    }

    #[test]
    fn cancelling_a_queued_job_drops_it_and_a_running_one_leaves_a_marker() {
        let root = scratch("cancel");
        write_job(&root, QUEUED, "download_q", serde_json::json!({}));
        write_job(&root, RUNNING, "induct_r", serde_json::json!({}));

        assert_eq!(
            cancel_job(&root, "download_q"),
            Ok(CancelOutcome::DroppedQueued)
        );
        assert!(list_jobs(&root).iter().all(|j| j.id != "download_q"));

        assert_eq!(
            cancel_job(&root, "induct_r"),
            Ok(CancelOutcome::AskedToStop)
        );
        assert!(root.join(LOGS).join("induct_r.cancel").exists());
        // The marker is the whole cancellation; the job stays listed until the
        // runner moves it, which is what the TUI shows in the meantime.
        assert_eq!(find_state(&root, "induct_r"), Some(RUNNING));

        assert!(cancel_job(&root, "nope").is_err());
    }

    #[test]
    fn a_percentage_is_read_back_out_of_the_progress_line() {
        let percent = |detail: &str| {
            JobSummary {
                detail: detail.to_string(),
                ..Default::default()
            }
            .progress_percent()
        };

        assert_eq!(
            percent("hub: 0.30/17.72 GB (2%) — model-00001-of-00004.safetensors — 3.4 MB/s"),
            Some(2)
        );
        assert_eq!(percent("hub: 17.72/17.72 GB (100%) — done"), Some(100));
        // Nothing to show rather than a wrong number: no line yet, a child that
        // reports no percentage, and a stray bracket that is not one.
        assert_eq!(percent(""), None);
        assert_eq!(percent("starting"), None);
        assert_eq!(percent("wrote archive (final)"), None);
    }

    #[test]
    fn a_log_shorter_than_the_window_comes_back_whole() {
        let root = scratch("log");
        fs::write(root.join(LOGS).join("x.log"), "one\ntwo\n").unwrap();
        assert_eq!(job_log_tail(&root, "x", 20), "one\ntwo");
        assert_eq!(job_log_tail(&root, "missing", 20), "");
    }
}
