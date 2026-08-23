// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! What the forward ACTUALLY chose, printed once per load.
//!
//! Every routing decision in this crate is a conjunction of conditions spread
//! across several files, and when one flips the only outward sign is a kernel
//! histogram. Recovering "did this run batched?", "did MoE take the grouped
//! path?", "is A4 live?" has repeatedly meant a `rocprofv3` trace — for
//! decisions the code already knows at dispatch time.
//!
//! So each decision site calls `note()` with its resolved value and the REASON,
//! and the first completed prefill flushes them as one `[features]` block. On by
//! default; `HIPFIRE_FEATURE_REPORT=0` silences it.
//!
//! Rules that keep this honest:
//!  * report the DECISION, not one of its inputs. decode_layers.rs records an
//!    earlier version that chose its message from one predicate and announced
//!    "lowered path" while the hand arms ran — in exactly the configuration the
//!    flag existed to compare.
//!  * always carry the reason. "moe=indexed" invites a trace; "moe=indexed
//!    (resident experts never enter the bucketed path-2 block)" ends the question.

use std::collections::BTreeMap;
use std::sync::{Mutex, OnceLock};

fn slots() -> &'static Mutex<BTreeMap<&'static str, String>> {
    static S: OnceLock<Mutex<BTreeMap<&'static str, String>>> = OnceLock::new();
    S.get_or_init(|| Mutex::new(BTreeMap::new()))
}

fn enabled() -> bool {
    static E: OnceLock<bool> = OnceLock::new();
    *E.get_or_init(|| std::env::var("HIPFIRE_FEATURE_REPORT").ok().as_deref() != Some("0"))
}

/// True until the report has been printed. Per-layer call sites must check this
/// BEFORE building their message.
pub fn wanted() -> bool {
    enabled() && !FLUSHED.load(std::sync::atomic::Ordering::Relaxed)
}

static FLUSHED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Record one resolved decision.
///
/// ⚠️ Call sites on the per-layer path MUST guard with `wanted()` and build the
/// string inside that guard. The first version of this took an
/// `impl Into<String>`, so a `format!` ran on every layer of every chunk and
/// the mutex was taken with it: MoE prefill measured 215.1 -> 175.7 tok/s, an
/// 18% regression caused purely by the reporting. After the flush `wanted()` is
/// one relaxed atomic load and the message is never built.
pub fn note(key: &'static str, value: String) {
    if !wanted() {
        return;
    }
    if let Ok(mut m) = slots().lock() {
        m.insert(key, value);
    }
}

/// Print the accumulated decisions once. Safe to call on every forward.
pub fn flush_once() {
    if !enabled() {
        return;
    }
    if FLUSHED.swap(true, std::sync::atomic::Ordering::Relaxed) {
        return;
    }
    let Ok(m) = slots().lock() else { return };
    if m.is_empty() {
        return;
    }
    eprintln!("[features] resolved forward routing (HIPFIRE_FEATURE_REPORT=0 to silence)");
    for (k, v) in m.iter() {
        eprintln!("[features]   {k:<16} {v}");
    }
}

/// Re-arm the report. A new model load should describe itself.
pub fn reset() {
    FLUSHED.store(false, std::sync::atomic::Ordering::Relaxed);
    if let Ok(mut m) = slots().lock() {
        m.clear();
    }
}
