// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

//! Structured model-load progress sink.
//!
//! Model loaders (`hfq.rs` here, plus the per-arch loaders in the
//! `hipfire-arch-*` crates) walk their layers synchronously and historically
//! only `eprintln!`'d "loading layer N/M" for humans. That human line was being
//! scraped off the daemon's stderr to drive the chat UI's load bar — fragile
//! (couples to log wording; six different format strings) and, on a piped
//! stderr, deadlock-prone on non-UTF-8 bytes.
//!
//! Instead, loaders call [`report`] at the same points. The daemon installs a
//! THREAD-scoped sink (via [`ThreadSinkGuard`]) for the duration of a load, which
//! serializes each report into a `load_progress` frame on its framed stdout channel — the same structured,
//! UTF-8-safe path used for tokens and other events. When no sink is installed
//! (CLI loads, tests, eval batteries), [`report`] is a cheap no-op.
//!
//! This lives in `hipfire-runtime` because it is the lowest crate reachable by
//! every loader: `hfq.rs` is in this crate, and the arch crates all depend on
//! it. The daemon (which also depends on this crate) is the only installer.
//!
//! **There is no process-wide sink.** There was one, kept as a fallback for a
//! hypothetical cross-thread reporter, with no production caller. It is gone
//! (v2 plan, M1d): while it existed, two overlapping loads could still
//! cross-talk through it, so the per-thread scoping was a convention rather
//! than a guarantee. Deleting it makes the property structural — there is now
//! no shared location for a second load to redirect the first's frames.
//!
//! Verified before removal, because it is the load-bearing assumption: all 14
//! `report` call sites sit directly in a synchronous loader on the calling
//! thread. `hipfire-arch-zaya/src/gpu.rs` does use rayon, but ~2,000 lines away
//! in CPU GEMM helpers, not in its `load()`. If a loader ever reports from a
//! spawned thread it will fall through to the human stderr line — visibly
//! degraded, not silently misrouted.

/// Sink signature: `(current, total, phase)`. Kept simple (no error return) so
/// loader call sites stay one-liners next to the existing `eprintln!`.
pub type ProgressFn = dyn Fn(u32, u32, &str) + Send + Sync;

thread_local! {
    /// The sink. Per-thread by construction — see the module doc for why there
    /// is no process-wide fallback beside it.
    ///
    /// The process-wide sink is correct only while one load runs at a time. The
    /// v2 daemon moves loading off the executor thread precisely so a
    /// multi-second `LoadModel` stops being a non-preemptible frame — at which
    /// point two loads can overlap, and a single global sink means the second
    /// installer silently redirects the first load's progress to the second
    /// caller. Nothing errors; the frames just go to the wrong client.
    ///
    /// A thread-local is the right shape rather than a plumbed handle because
    /// loaders walk their layers SYNCHRONOUSLY on the calling thread — checked
    /// across all ten `report` call sites, none sits in a `rayon`/`spawn`
    /// context — so "the thread doing the load" and "the load" are the same
    /// thing. That also means `report`'s signature does not change, which is
    /// what keeps this off six arch loaders.
    static THREAD_SINK: std::cell::RefCell<Option<Box<ProgressFn>>> =
        const { std::cell::RefCell::new(None) };
}

/// Install (or clear) a sink for THIS THREAD only. Returns the previous thread sink so a caller can restore
/// it, which is what makes nesting safe.
///
pub fn set_thread_sink(sink: Option<Box<ProgressFn>>) -> Option<Box<ProgressFn>> {
    THREAD_SINK.with(|cell| std::mem::replace(&mut *cell.borrow_mut(), sink))
}

/// Scope guard: installs a thread sink and restores the previous one on drop,
/// including on an early return or a panic. Prefer this to bare
/// [`set_thread_sink`] — a load that returns `Err` partway through must not
/// leave its sink installed for whatever the thread does next.
pub struct ThreadSinkGuard(Option<Option<Box<ProgressFn>>>);

impl ThreadSinkGuard {
    pub fn install(sink: Box<ProgressFn>) -> Self {
        Self(Some(set_thread_sink(Some(sink))))
    }
}

impl Drop for ThreadSinkGuard {
    fn drop(&mut self) {
        if let Some(prev) = self.0.take() {
            let _ = set_thread_sink(prev);
        }
    }
}

/// Report load progress. `current`/`total` are phase-relative unit counts
/// (e.g. layer `i+1` of `n`); `phase` is a coarse label such as `"weights"`.
/// No-op when no sink is installed. Touches no shared state, so concurrent
/// loads on different threads never contend.
pub fn report(current: u32, total: u32, phase: &str) {
    let handled = THREAD_SINK.with(|cell| match cell.try_borrow() {
        Ok(guard) => match guard.as_ref() {
            Some(sink) => {
                sink(current, total, phase);
                true
            }
            None => false,
        },
        // Already borrowed = a sink that itself calls `report`. Decline rather
        // than panic; observability must not take down a load.
        Err(_) => false,
    });
    if handled {
        return;
    }
    // No sink (CLI-direct load, eval, tests), or a report from a thread that
    // installed none: emit the human progress line loaders used to `eprintln!`
    // themselves, so text callers still see it.
    tracing::info!("loading {phase} {current}/{total}");
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{
        atomic::{AtomicU32, Ordering},
        Arc, Mutex,
    };

    /// Serializes every test in this module, because one of them installs the
    /// PROCESS-WIDE sink and Rust runs a crate's tests in parallel threads inside
    /// a single process. With correct precedence the others are insulated (they
    /// each install a thread sink before reporting), but that is a property of how
    /// they happen to be written, not a guarantee — a future test that reports
    /// without a thread sink would flake depending on scheduling. Cheap insurance
    /// against a heisentest.
    ///
    /// Poison is ignored deliberately: a panicking test must not cascade into
    /// failures in the others.
    static SERIAL: Mutex<()> = Mutex::new(());

    fn serial() -> std::sync::MutexGuard<'static, ()> {
        SERIAL.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// The property P1 exists for: two concurrent loads must not report into
    /// each other's caller. Asserted rather than inspected — with the global
    /// sink alone, whichever thread installed second captures both.
    #[test]
    fn concurrent_thread_sinks_do_not_cross_talk() {
        let _serial = serial();
        let a = Arc::new(AtomicU32::new(0));
        let b = Arc::new(AtomicU32::new(0));
        let (a2, b2) = (Arc::clone(&a), Arc::clone(&b));

        let ta = std::thread::spawn(move || {
            let _g = ThreadSinkGuard::install(Box::new(move |c, _, _| {
                a2.fetch_add(c, Ordering::Relaxed);
            }));
            for i in 1..=10 {
                report(i, 10, "weights");
            }
        });
        let tb = std::thread::spawn(move || {
            let _g = ThreadSinkGuard::install(Box::new(move |c, _, _| {
                b2.fetch_add(c * 1000, Ordering::Relaxed);
            }));
            for i in 1..=10 {
                report(i, 10, "weights");
            }
        });
        ta.join().unwrap();
        tb.join().unwrap();

        // 1..=10 sums to 55. Each thread must see exactly its own reports.
        assert_eq!(
            a.load(Ordering::Relaxed),
            55,
            "thread A saw the wrong reports"
        );
        assert_eq!(
            b.load(Ordering::Relaxed),
            55_000,
            "thread B saw the wrong reports"
        );
    }

    /// The guard must restore on drop, so a load that fails partway does not
    /// leave its sink installed for whatever the thread does next.
    #[test]
    fn guard_restores_previous_thread_sink() {
        let _serial = serial();
        let outer = Arc::new(AtomicU32::new(0));
        let o2 = Arc::clone(&outer);
        let _g = ThreadSinkGuard::install(Box::new(move |c, _, _| {
            o2.fetch_add(c, Ordering::Relaxed);
        }));
        {
            let inner = Arc::new(AtomicU32::new(0));
            let i2 = Arc::clone(&inner);
            let _inner_guard = ThreadSinkGuard::install(Box::new(move |c, _, _| {
                i2.fetch_add(c, Ordering::Relaxed);
            }));
            report(7, 10, "weights");
            assert_eq!(inner.load(Ordering::Relaxed), 7);
        }
        report(3, 10, "weights");
        assert_eq!(
            outer.load(Ordering::Relaxed),
            3,
            "outer sink was not restored"
        );
        let _ = set_thread_sink(None);
    }

    // NOTE: the test that used to sit here,
    // `thread_sink_takes_precedence_over_installed_global`, was removed with the
    // process-wide sink it tested. It is deliberately NOT replaced by a
    // "sinkless thread cannot reach another thread's sink" test: with no shared
    // location, `thread_local!` enforces that at compile time, so such a test
    // cannot fail and would be coverage theatre. The property is now structural
    // — the way to break it is to re-add a global, which is a code review
    // question, not a runtime one.
}
