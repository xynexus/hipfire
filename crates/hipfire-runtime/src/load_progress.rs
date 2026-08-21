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
//! it. The daemon (which also depends on this crate) is the only installer, and
//! it installs the thread-scoped sink — [`set_sink`] has no production caller.
//! The process-wide sink remains only as the fallback a CROSS-THREAD reporter
//! would reach; no loader currently reports off the loading thread, so in
//! practice it is unused and a report reaching it would print the human line.

use std::sync::Mutex;

/// Sink signature: `(current, total, phase)`. Kept simple (no error return) so
/// loader call sites stay one-liners next to the existing `eprintln!`.
pub type ProgressFn = dyn Fn(u32, u32, &str) + Send + Sync;

static SINK: Mutex<Option<Box<ProgressFn>>> = Mutex::new(None);

thread_local! {
    /// Per-thread sink, consulted before [`SINK`].
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

/// Install (or clear, with `None`) the process-wide load-progress sink.
///
/// No production caller: the daemon moved to [`ThreadSinkGuard`]. Kept as the
/// fallback a cross-thread reporter would reach, and for single-load callers.
///
/// Prefer [`set_thread_sink`] for anything that can run concurrently with
/// another load; this one stays for single-load callers and as the fallback a
/// cross-thread reporter still reaches.
pub fn set_sink(sink: Option<Box<ProgressFn>>) {
    if let Ok(mut guard) = SINK.lock() {
        *guard = sink;
    }
}

/// Install (or clear) a sink for THIS THREAD only, taking precedence over the
/// process-wide one. Returns the previous thread sink so a caller can restore
/// it, which is what makes nesting safe.
///
/// Additive on purpose: the global path is untouched, so a caller that has not
/// migrated behaves exactly as before.
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
/// No-op when no sink is installed. The `SINK` lock is held across the sink
/// call, which serializes concurrent reports into whole, non-interleaved frames.
pub fn report(current: u32, total: u32, phase: &str) {
    // Thread sink first: it is the narrower scope, and when two loads overlap it
    // is the only one that can name the right caller. Returns without touching
    // the global mutex, so a migrated load path does not contend with another
    // thread's load at all.
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
    if let Ok(guard) = SINK.lock() {
        if let Some(sink) = guard.as_ref() {
            sink(current, total, phase);
            return;
        }
        // No sink (CLI-direct load, eval, tests): emit the human progress line
        // loaders used to `eprintln!` themselves, so text callers still see it.
        eprintln!("  loading {phase} {current}/{total}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{
        atomic::{AtomicU32, Ordering},
        Arc,
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
        assert_eq!(a.load(Ordering::Relaxed), 55, "thread A saw the wrong reports");
        assert_eq!(b.load(Ordering::Relaxed), 55_000, "thread B saw the wrong reports");
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
        assert_eq!(outer.load(Ordering::Relaxed), 3, "outer sink was not restored");
        let _ = set_thread_sink(None);
    }

    /// Precedence over an INSTALLED global, not merely over an absent one — and
    /// the global still reached by a thread that has not migrated.
    ///
    /// The two tests above only ever install thread sinks, so they pass even if
    /// `report` consulted the global first. That is the daemon's actual
    /// configuration during the v2 transition (thread sink on the load path, the
    /// global left as the fallback a cross-thread reporter still reaches), so the
    /// precedence is the property the migration rests on.
    #[test]
    fn thread_sink_takes_precedence_over_installed_global() {
        let _serial = serial();
        let global = Arc::new(AtomicU32::new(0));
        let g2 = Arc::clone(&global);
        set_sink(Some(Box::new(move |c, _, _| {
            g2.fetch_add(c, Ordering::Relaxed);
        })));

        let mine = Arc::new(AtomicU32::new(0));
        let m2 = Arc::clone(&mine);
        std::thread::spawn(move || {
            let _g = ThreadSinkGuard::install(Box::new(move |c, _, _| {
                m2.fetch_add(c, Ordering::Relaxed);
            }));
            report(5, 10, "weights");
        })
        .join()
        .unwrap();

        assert_eq!(
            mine.load(Ordering::Relaxed),
            5,
            "thread sink did not receive its own report"
        );
        assert_eq!(
            global.load(Ordering::Relaxed),
            0,
            "report reached the global sink too — a migrated load would double-report"
        );

        // A thread that has NOT installed one still falls through to the global.
        std::thread::spawn(|| report(4, 10, "weights"))
            .join()
            .unwrap();
        assert_eq!(
            global.load(Ordering::Relaxed),
            4,
            "global fallback missed an unmigrated report"
        );

        set_sink(None);
    }
}
