// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! Which kernels actually ran — a dispatch histogram, gated by
//! `HIPFIRE_KERNEL_TRACE`.
//!
//! This exists because a measurement can name a path it never took. `hipfire
//! bench --pp-tokens` reported `pp512 t/s` from a handler whose arch-0/1 arm is
//! a per-token `decode_step` loop, and the mislabel survived a full
//! investigation — four causes "ruled out", two documents written — because
//! nothing in the output said which kernels ran. A histogram showing
//! `gemv_oq4_grouped` and no `gemm_*` during a "prefill" bench ends that in one
//! line. See `~/measurement-integrity-goal.md`.
//!
//! `Gpu::ensure_kernel` is the universal chokepoint: every dispatch calls it to
//! resolve the function, cache-hit or not, so counting there catches everything
//! without touching call sites. Off by default; when off the cost is one relaxed
//! atomic load.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::{Mutex, OnceLock};

static STATE: AtomicU8 = AtomicU8::new(0); // 0 = unknown, 1 = off, 2 = on
static WARNED: AtomicBool = AtomicBool::new(false);

fn counts() -> &'static Mutex<HashMap<String, u64>> {
    static C: OnceLock<Mutex<HashMap<String, u64>>> = OnceLock::new();
    C.get_or_init(|| Mutex::new(HashMap::new()))
}

/// `HIPFIRE_KERNEL_TRACE` set and not `0`.
pub fn enabled() -> bool {
    match STATE.load(Ordering::Relaxed) {
        1 => false,
        2 => true,
        _ => {
            let on = std::env::var_os("HIPFIRE_KERNEL_TRACE")
                .map(|v| v != "0")
                .unwrap_or(false);
            STATE.store(if on { 2 } else { 1 }, Ordering::Relaxed);
            on
        }
    }
}

/// Count one dispatch of `func_name`. Called from `ensure_kernel`.
pub fn record(func_name: &str) {
    if !enabled() {
        return;
    }
    let mut c = match counts().lock() {
        Ok(c) => c,
        // A poisoned trace lock must never take the process with it — this is
        // diagnostics, not correctness.
        Err(_) => {
            if !WARNED.swap(true, Ordering::Relaxed) {
                eprintln!("[kernel-trace] lock poisoned; trace disabled");
            }
            STATE.store(1, Ordering::Relaxed);
            return;
        }
    };
    *c.entry(func_name.to_string()).or_insert(0) += 1;
}

/// Dispatch counts, descending. Empty when tracing is off.
pub fn snapshot() -> Vec<(String, u64)> {
    let Ok(c) = counts().lock() else {
        return Vec::new();
    };
    let mut v: Vec<(String, u64)> = c.iter().map(|(k, n)| (k.clone(), *n)).collect();
    v.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    v
}

/// Drop all counts, so a caller can scope a trace to one request.
pub fn reset() {
    if let Ok(mut c) = counts().lock() {
        c.clear();
    }
}

/// Format the histogram for a log line. `None` when tracing is off or nothing
/// was dispatched — callers print nothing rather than an empty table.
pub fn report(label: &str) -> Option<String> {
    if !enabled() {
        return None;
    }
    let snap = snapshot();
    if snap.is_empty() {
        return None;
    }
    let total: u64 = snap.iter().map(|(_, n)| n).sum();
    let mut s = format!(
        "[kernel-trace] {label}: {total} dispatches across {} kernels\n",
        snap.len()
    );
    for (name, n) in snap.iter().take(20) {
        s.push_str(&format!(
            "  {:>9}  {:5.1}%  {name}\n",
            n,
            100.0 * *n as f64 / total as f64
        ));
    }
    if snap.len() > 20 {
        s.push_str(&format!("  ... {} more kernels\n", snap.len() - 20));
    }
    Some(s)
}
