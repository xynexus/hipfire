// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! `/admin/jobs` — the background job queue, for the admin UI.
//!
//! The queue is a directory the CLI and TUI read straight off disk; a browser
//! cannot, so these routes are the same `hipfire_operator::jobs` reads served
//! over HTTP. They stay behind the admin gate with the rest of `/admin/*`.

use axum::{
    extract::Path,
    http::StatusCode,
    response::{IntoResponse, Json, Response},
};
use hipfire_operator::jobs::{cancel_job, find_state, job_log_tail, list_jobs, CancelOutcome};
use serde_json::{json, Value};

use crate::deferred_jobs::deferred_jobs_root;

/// Lines of a job's log returned with its detail — enough to see what a fetch
/// has been doing without shipping a multi-megabyte log to the browser.
const LOG_LINES: usize = 200;

pub async fn list_jobs_route() -> Json<Value> {
    let root = deferred_jobs_root();
    Json(json!({
        "jobs_dir": root.display().to_string(),
        "jobs": list_jobs(&root),
    }))
}

pub async fn get_job(Path(id): Path<String>) -> Response {
    let root = deferred_jobs_root();
    // Resolve the id against the queue listing before touching any path built
    // from it: `id` is request input, and this is what keeps a traversal
    // attempt from reaching a file outside the queue.
    let Some(summary) = list_jobs(&root).into_iter().find(|job| job.id == id) else {
        return not_found(&id);
    };
    Json(json!({
        "summary": summary,
        "log": job_log_tail(&root, &id, LOG_LINES),
    }))
    .into_response()
}

pub async fn post_job_cancel(Path(id): Path<String>) -> Response {
    let root = deferred_jobs_root();
    if find_state(&root, &id).is_none() {
        return not_found(&id);
    }
    match cancel_job(&root, &id) {
        Ok(CancelOutcome::DroppedQueued) => Json(json!({
            "status": "ok",
            "id": id,
            "outcome": "dropped_queued",
            "detail": "job was never claimed; its job file was removed",
        }))
        .into_response(),
        Ok(CancelOutcome::AskedToStop) => Json(json!({
            "status": "ok",
            "id": id,
            "outcome": "asked_to_stop",
            "detail": "cancel marker written; a download resumes if resubmitted",
        }))
        .into_response(),
        Ok(CancelOutcome::AlreadyFinished(state)) => Json(json!({
            "status": "ok",
            "id": id,
            "outcome": "already_finished",
            "detail": format!("job is already {state}"),
        }))
        .into_response(),
        Err(error) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({
                "error": { "message": error, "type": "job_cancel_failed" }
            })),
        )
            .into_response(),
    }
}

fn not_found(id: &str) -> Response {
    (
        StatusCode::NOT_FOUND,
        Json(json!({
            "error": {
                "message": format!("job '{id}' not found"),
                "type": "not_found"
            }
        })),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::to_bytes;

    async fn body_json(response: Response) -> Value {
        let bytes = to_bytes(response.into_body(), 1 << 20).await.unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    /// `id` reaches these handlers straight from the URL, and both of them use
    /// it to build a path. Resolving it against the queue listing first is what
    /// stops `../` walking out of the queue directory — assert that rather than
    /// trusting it stays that way.
    #[tokio::test]
    async fn an_id_that_is_not_in_the_queue_is_rejected_before_any_path_is_built() {
        for id in ["../../etc/passwd", "..", "nope", ""] {
            let response = get_job(Path(id.to_string())).await;
            assert_eq!(
                response.status(),
                StatusCode::NOT_FOUND,
                "get_job accepted {id:?}"
            );

            let response = post_job_cancel(Path(id.to_string())).await;
            assert_eq!(
                response.status(),
                StatusCode::NOT_FOUND,
                "post_job_cancel accepted {id:?}"
            );
        }
    }

    #[tokio::test]
    async fn the_listing_names_the_queue_directory_it_read() {
        let listed = list_jobs_route().await;
        let dir = listed.0["jobs_dir"].as_str().unwrap_or_default();
        assert!(dir.ends_with("jobs/deferred"), "unexpected dir {dir}");
        assert!(listed.0["jobs"].is_array());
    }

    #[tokio::test]
    async fn a_missing_job_reports_not_found_as_json() {
        let payload = body_json(get_job(Path("missing_job".into())).await).await;
        assert_eq!(payload["error"]["type"], "not_found");
    }
}
