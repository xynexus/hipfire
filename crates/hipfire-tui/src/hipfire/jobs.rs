// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

//! Jobs tab state — background downloads and inductions.
//!
//! Unlike Training, this reads local files only: the queue lives beside the
//! config this TUI is already editing, so there is nothing an operator API
//! would add for a same-host view.
//! ponytail: local files only, add an operator route when the TUI has to watch
//! a queue on another host.

use hipfire_operator::jobs::{cancel_job, job_log_tail, list_jobs, CancelOutcome, JobSummary};

use super::HipfirePaths;

/// Lines of the selected job's log kept for the detail pane.
const LOG_LINES: usize = 60;

#[derive(Clone, Debug, Default)]
pub struct JobsState {
    pub dir: std::path::PathBuf,
    pub jobs: Vec<JobSummary>,
    pub selected: usize,
    pub log: String,
}

impl JobsState {
    pub fn load(paths: &HipfirePaths) -> Self {
        let mut state = Self {
            dir: paths.jobs.clone(),
            jobs: list_jobs(&paths.jobs),
            selected: 0,
            log: String::new(),
        };
        state.reload_log(paths);
        state
    }

    /// Refresh the list without losing the operator's place: a job that moves
    /// from `running` to `done` reorders the list, so follow the id rather than
    /// the index.
    pub fn refresh(&mut self, paths: &HipfirePaths) {
        let selected_id = self.selected_id().map(str::to_string);
        self.jobs = list_jobs(&paths.jobs);
        self.selected = selected_id
            .and_then(|id| self.jobs.iter().position(|j| j.id == id))
            .unwrap_or(0);
        self.reload_log(paths);
    }

    pub fn selected_job(&self) -> Option<&JobSummary> {
        self.jobs.get(self.selected)
    }

    pub fn selected_id(&self) -> Option<&str> {
        self.selected_job().map(|j| j.id.as_str())
    }

    pub fn active_count(&self) -> usize {
        self.jobs.iter().filter(|j| j.is_active()).count()
    }

    pub fn select_delta(&mut self, delta: isize, paths: &HipfirePaths) {
        if self.jobs.is_empty() {
            self.selected = 0;
            self.log.clear();
            return;
        }
        let max = self.jobs.len() as isize - 1;
        self.selected = (self.selected as isize + delta).clamp(0, max) as usize;
        self.reload_log(paths);
    }

    fn reload_log(&mut self, paths: &HipfirePaths) {
        self.log = match self.selected_id() {
            Some(id) => job_log_tail(&paths.jobs, id, LOG_LINES),
            None => String::new(),
        };
    }

    /// Cancel the selected job, returning the line to show in the status bar.
    pub fn cancel_selected(&mut self, paths: &HipfirePaths) -> String {
        let Some(id) = self.selected_id().map(str::to_string) else {
            return "no job selected".into();
        };
        let message = match cancel_job(&paths.jobs, &id) {
            Ok(CancelOutcome::DroppedQueued) => format!("dropped queued job {id}"),
            Ok(CancelOutcome::AskedToStop) => format!("asked {id} to stop"),
            Ok(CancelOutcome::AlreadyFinished(state)) => {
                format!("job {id} is already {state}")
            }
            Err(err) => format!("cancel {id}: {err}"),
        };
        self.refresh(paths);
        message
    }
}
