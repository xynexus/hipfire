// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Cooperative cancellation for the in-flight request.
//!
//! `abort` used to be a dead wire variant: the daemon replied that it "is handled
//! on the control channel, not the request channel", and there was no control
//! channel — so an abort could only be *read* after the generation it wanted to
//! cancel had already finished. Two things were missing, and this is the second.
//! The first is a reader thread that can accept a frame while the executor is
//! busy; this is somewhere for that reader to leave the message.
//!
//! **Why a process global rather than a token parameter.** The obvious shape is a
//! cancellation token threaded into `generate`. That function takes 28 positional
//! parameters and immediately delegates to one of several decode paths
//! (`generate_mtp`, `generate_dflash`, `generate_multi`, the registered-backend
//! fast path, plus the per-arch decode loops), so a token would have to be
//! plumbed through all of them and every future one — and a loop that forgot to
//! check it would silently ignore aborts.
//!
//! Instead cancellation is observed at the two places a decode loop already
//! decides whether to keep going:
//!
//! - `arch::decode_loop_*`, the generic loop the gemma3 family and every
//!   factory/registered backend run through. This is the one that matters, and it
//!   was found the hard way: an end-to-end abort against gemma3-vl ran all 400
//!   tokens to completion because hooking only the emitter below was not enough —
//!   this loop never calls it.
//! - `serving-core::events::emit_filter_action`, which the older inline decode
//!   loops in `generate.rs` call once per token and already treat as "stop
//!   generating" when it returns true.
//!
//! Living in `hipfire-runtime` rather than `serving-core` is what lets the generic
//! loop see it at all: `serving-core` depends on this crate, not the reverse.
//!
//! A global is honest here for the same reason the daemon is serial: one request
//! is in flight at a time. The id guard is what keeps that safe — an abort names
//! the request it is for, so a stale one cannot stop an unrelated successor.

use std::sync::Mutex;

/// What a caller asked for.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StopKind {
    /// Stop and discard: the caller no longer wants the answer.
    Abort,
    /// Stop but keep what was produced: the caller wants it wrapped up now.
    ForceAnswer,
}

struct Pending {
    id: String,
    kind: StopKind,
}

static PENDING: Mutex<Option<Pending>> = Mutex::new(None);

/// Ask the in-flight request `id` to stop.
///
/// Called from a reader thread while the executor may be mid-generation, which is
/// the whole point — it must not need the executor to service a queue first. An
/// empty id is ignored: a request with no id cannot be named, so an abort for one
/// would be indistinguishable from an abort for every other unnamed request.
pub fn request(id: &str, kind: StopKind) -> bool {
    if id.is_empty() {
        return false;
    }
    if let Ok(mut pending) = PENDING.lock() {
        *pending = Some(Pending {
            id: id.to_string(),
            kind,
        });
        return true;
    }
    false
}

/// True when `id` has been asked to stop. Checked once per generated token.
pub fn is_cancelled(id: &str) -> bool {
    stop_kind(id).is_some()
}

/// How `id` was asked to stop, if it was.
pub fn stop_kind(id: &str) -> Option<StopKind> {
    if id.is_empty() {
        return None;
    }
    PENDING
        .lock()
        .ok()?
        .as_ref()
        .filter(|pending| pending.id == id)
        .map(|pending| pending.kind)
}

/// Drop any pending stop. The executor calls this as it takes up a new request, so
/// an abort that arrived too late for its target cannot carry over and kill an
/// unrelated successor.
pub fn clear() {
    if let Ok(mut pending) = PENDING.lock() {
        *pending = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // These share one process global, so they run as a single test rather than
    // racing each other under the default parallel harness.
    #[test]
    fn cancellation_is_scoped_to_the_request_it_names() {
        clear();
        assert!(!is_cancelled("r1"), "nothing pending to start with");

        assert!(request("r1", StopKind::Abort));
        assert!(is_cancelled("r1"));
        assert_eq!(stop_kind("r1"), Some(StopKind::Abort));

        // The crux: an abort for one request must not stop another. Without the id
        // guard, a late abort would kill whatever happened to be running next.
        assert!(
            !is_cancelled("r2"),
            "an abort names one request, not all of them"
        );

        // Taking up a new request drops the stale ask.
        clear();
        assert!(!is_cancelled("r1"));

        // force_answer is carried distinctly so the terminal frame can say which
        // happened, even though both stop the loop.
        assert!(request("r3", StopKind::ForceAnswer));
        assert_eq!(stop_kind("r3"), Some(StopKind::ForceAnswer));

        // A request with no id cannot be named, so it cannot be aborted.
        clear();
        assert!(!request("", StopKind::Abort));
        assert!(!is_cancelled(""));

        clear();
    }
}
