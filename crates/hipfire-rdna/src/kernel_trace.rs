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

/// (call count, weight bytes) per `kernel  m x k x n` shape.
type ShapeMap = HashMap<String, (u64, u128)>;
fn shapes() -> &'static Mutex<ShapeMap> {
    static S: OnceLock<Mutex<ShapeMap>> = OnceLock::new();
    S.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Slow-path / fallback events: `site` -> (count, one example detail).
fn fallbacks() -> &'static Mutex<HashMap<String, (u64, String)>> {
    static F: OnceLock<Mutex<HashMap<String, (u64, String)>>> = OnceLock::new();
    F.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Record that a dispatch took a SLOW/fallback branch — a per-row loop, a
/// per-token forward, a dtype ladder's default arm.
///
/// WHY: three separate wins this session were the same defect — a dtype ladder
/// that stopped at the MQ/HFQ families while an Opus artifact silently took the
/// slow branch. None of them showed up in a kernel histogram, a phase timer, or
/// a correctness gate; each needed hand-instrumentation to find. A fallback that
/// ANNOUNCES itself turns that from an afternoon into one line of output.
///
/// Cheap when tracing is off (one relaxed atomic load), so call it freely at
/// every `_ =>` arm that means "the fast path did not apply here".
pub fn record_fallback(site: &str, detail: &str) {
    if !enabled() {
        return;
    }
    if let Ok(mut f) = fallbacks().lock() {
        let e = f
            .entry(site.to_string())
            .or_insert_with(|| (0, detail.to_string()));
        e.0 += 1;
    }
}

/// Record one shaped dispatch. `bytes` is the WEIGHT traffic for this call —
/// the term that actually decides whether a kernel is the bottleneck. Call
/// counts alone mislead: a [248320, 5120] lm_head at 8 calls/cycle outweighs
/// thousands of small projections.
pub fn record_shape(kernel: &str, m: usize, k: usize, n: usize, bytes: u128) {
    if !enabled() {
        return;
    }
    if let Ok(mut sh) = shapes().lock() {
        let key = format!("{kernel}  m={m} k={k} n={n}");
        let e = sh.entry(key).or_insert((0, 0));
        e.0 += 1;
        e.1 = e.1.saturating_add(bytes);
    }
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
    if let Ok(mut s) = shapes().lock() {
        s.clear();
    }
    if let Ok(mut f) = fallbacks().lock() {
        f.clear();
    }
}

/// Print `report(label)` to stderr. Binaries call this at end-of-run so
/// `HIPFIRE_KERNEL_TRACE=1` produces output withoutevery call site opting in.
pub fn dump(label: &str) {
    if let Some(r) = report(label) {
        eprint!("{r}");
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

    // Shapes, ranked by WEIGHT BYTES rather than call count — the ordering that
    // actually identifies the bottleneck.
    if let Ok(sh) = shapes().lock() {
        if !sh.is_empty() {
            let mut v: Vec<(&String, &(u64, u128))> = sh.iter().collect();
            v.sort_by(|a, b| b.1 .1.cmp(&a.1 .1));
            let total: u128 = v.iter().map(|(_, (_, b))| *b).sum();
            s.push_str(&format!(
                "[kernel-trace] {label}: shaped traffic, {:.2} GiB total\n",
                total as f64 / (1024.0 * 1024.0 * 1024.0)
            ));
            for (key, (n, bytes)) in v.iter().take(15) {
                s.push_str(&format!(
                    "  {:>8} calls  {:>8.2} GiB  {:5.1}%  {key}\n",
                    n,
                    *bytes as f64 / (1024.0 * 1024.0 * 1024.0),
                    if total > 0 {
                        100.0 * *bytes as f64 / total as f64
                    } else {
                        0.0
                    }
                ));
            }
        }
    }

    // Slow paths last, because this is the section that explains a bad number.
    if let Ok(f) = fallbacks().lock() {
        if !f.is_empty() {
            let mut v: Vec<(&String, &(u64, String))> = f.iter().collect();
            v.sort_by(|a, b| b.1 .0.cmp(&a.1 .0));
            s.push_str(&format!(
                "[kernel-trace] {label}: SLOW PATHS TAKEN ({} site(s)) \
                 — each is a fast path that did not apply\n",
                v.len()
            ));
            for (site, (n, detail)) in v {
                s.push_str(&format!("  {n:>9}x  {site}  [{detail}]\n"));
            }
        }
    }
    Some(s)
}
