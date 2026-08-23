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

/// Record one resolved decision. Last writer wins, so a per-layer site may call
/// this every layer without growing the report.
pub fn note(key: &'static str, value: impl Into<String>) {
    if !enabled() {
        return;
    }
    if let Ok(mut m) = slots().lock() {
        m.insert(key, value.into());
    }
}

/// Print the accumulated decisions once. Safe to call on every forward.
pub fn flush_once() {
    if !enabled() {
        return;
    }
    static DONE: OnceLock<()> = OnceLock::new();
    if DONE.set(()).is_err() {
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
    if let Ok(mut m) = slots().lock() {
        m.clear();
    }
}
