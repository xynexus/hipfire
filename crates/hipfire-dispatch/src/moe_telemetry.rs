// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! MoE router selection histogram: thread-local telemetry for expert routing.
//!
//! Lives in `hipfire-dispatch` rather than an arch crate because there are TWO
//! MoE decode implementations that must feed one histogram:
//!
//!   - `hipfire_arch_qwen35::qwen35::moe_decode::moe_ffn_decode_impl`
//!   - `hipfire_dispatch::pipeline`'s executor, which ports that body verbatim
//!
//! Arch crates depend on `hipfire-dispatch`, not the reverse, so this is the
//! lowest common crate both can reach. `hipfire_arch_qwen35::qwen35` re-exports
//! the whole API, so existing `qwen35::reset_moe_router_histogram(..)` callers
//! are unaffected.
//!
//! The state is `thread_local!`, so both implementations recording on the same
//! worker thread accumulate into the same histogram.

use std::cell::RefCell;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct MoeRouterHistogram {
    pub num_experts: usize,
    pub k_top: usize,
    pub routed_tokens: u64,
    pub routed_slots: u64,
    pub top1_histogram: Vec<u64>,
    pub topk_histogram: Vec<u64>,
    pub weight_sums: Vec<f64>,
    pub dropped_indices: u64,
    pub per_layer: Vec<MoeRouterLayerHistogram>,
}

#[derive(Debug, Clone)]
pub struct MoeRouterLayerHistogram {
    pub layer_idx: usize,
    pub routed_tokens: u64,
    pub routed_slots: u64,
    pub top1_histogram: Vec<u64>,
    pub topk_histogram: Vec<u64>,
    pub weight_sums: Vec<f64>,
    pub dropped_indices: u64,
    pub cooccurrence: HashMap<u64, u64>,
}

impl MoeRouterHistogram {
    fn new(num_experts: usize, k_top: usize) -> Self {
        Self {
            num_experts,
            k_top,
            routed_tokens: 0,
            routed_slots: 0,
            top1_histogram: vec![0; num_experts],
            topk_histogram: vec![0; num_experts],
            weight_sums: vec![0.0; num_experts],
            dropped_indices: 0,
            per_layer: Vec::new(),
        }
    }

    fn ensure_layer(&mut self, layer_idx: usize) -> &mut MoeRouterLayerHistogram {
        while self.per_layer.len() <= layer_idx {
            let next = self.per_layer.len();
            self.per_layer.push(MoeRouterLayerHistogram {
                layer_idx: next,
                routed_tokens: 0,
                routed_slots: 0,
                top1_histogram: vec![0; self.num_experts],
                topk_histogram: vec![0; self.num_experts],
                weight_sums: vec![0.0; self.num_experts],
                dropped_indices: 0,
                cooccurrence: HashMap::new(),
            });
        }
        &mut self.per_layer[layer_idx]
    }
}

thread_local! {
    static MOE_ROUTER_HISTOGRAM: RefCell<Option<MoeRouterHistogram>> = const { RefCell::new(None) };
}

pub fn reset_moe_router_histogram(num_experts: usize, k_top: usize) {
    MOE_ROUTER_HISTOGRAM.with(|hist| {
        *hist.borrow_mut() = Some(MoeRouterHistogram::new(num_experts, k_top));
    });
}

pub fn take_moe_router_histogram() -> Option<MoeRouterHistogram> {
    MOE_ROUTER_HISTOGRAM.with(|hist| hist.borrow_mut().take())
}

// ── Per-request routing trace (`moe_trace` setting) ──────────────────────────
//
// One trace per generate request, keyed by request id. The daemon opens it when
// the request starts and closes it when the request retires; serving code
// `select`s the request whose forward is about to run (the march loop
// interleaves requests); the forward itself names the absolute position of each
// row it routes (`trace_rows`). A row with no named position is not written —
// that is the multi-session batched executor, whose rows belong to several
// requests at once.
//
// `<stem>.bin` is a stream of records, all little-endian:
//   u32 position, u16 layer, u16 k, k x u16 expert, k x u16 weight (x / 65535)
// Spec-decode verify rows are written too, so a position can repeat; the LAST
// record for a (position, layer) is the one the sequence kept.

/// What one closed trace saw.
#[derive(Debug, Clone, Default)]
pub struct MoeTraceSummary {
    pub path: std::path::PathBuf,
    pub records: u64,
    /// Rows routed with no position named (not written).
    pub unplaced: u64,
    pub k: usize,
    pub positions: Option<(u32, u32)>,
    /// Per layer, how often each expert was picked.
    pub layer_counts: Vec<Vec<u64>>,
    /// Sum of routing weight at each rank, over all records.
    pub weight_by_rank: Vec<f64>,
}

struct TraceFile {
    out: std::io::BufWriter<std::fs::File>,
    sum: MoeTraceSummary,
}

#[derive(Default)]
struct TraceState {
    open: HashMap<String, TraceFile>,
    current: Option<String>,
    rows: Vec<u32>,
}

fn trace_state() -> &'static std::sync::Mutex<TraceState> {
    static S: std::sync::OnceLock<std::sync::Mutex<TraceState>> = std::sync::OnceLock::new();
    S.get_or_init(Default::default)
}

/// Any trace open. Refreshed on open/close; read on every routed row, so the
/// off path is one relaxed load.
static TRACE_ANY: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Start tracing request `id` into `<stem>.bin` and make it current.
pub fn trace_open(id: &str, stem: &std::path::Path) -> std::io::Result<()> {
    if let Some(dir) = stem.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let path = stem.with_extension("bin");
    let out = std::io::BufWriter::new(std::fs::File::create(&path)?);
    let mut st = trace_state().lock().unwrap_or_else(|e| e.into_inner());
    st.open.insert(
        id.to_owned(),
        TraceFile {
            out,
            sum: MoeTraceSummary {
                path,
                ..Default::default()
            },
        },
    );
    st.current = Some(id.to_owned());
    TRACE_ANY.store(true, std::sync::atomic::Ordering::Relaxed);
    Ok(())
}

/// The request whose forward runs next. A no-op unless `id` is being traced.
pub fn trace_select(id: &str) {
    if !TRACE_ANY.load(std::sync::atomic::Ordering::Relaxed) {
        return;
    }
    let mut st = trace_state().lock().unwrap_or_else(|e| e.into_inner());
    st.current = st.open.contains_key(id).then(|| id.to_owned());
}

/// Flush and close request `id`'s trace. `None` if it was not traced, so
/// calling it from every exit path is safe.
pub fn trace_close(id: &str) -> Option<MoeTraceSummary> {
    let mut st = trace_state().lock().unwrap_or_else(|e| e.into_inner());
    let mut f = st.open.remove(id)?;
    if st.current.as_deref() == Some(id) {
        st.current = None;
    }
    TRACE_ANY.store(!st.open.is_empty(), std::sync::atomic::Ordering::Relaxed);
    use std::io::Write;
    let _ = f.out.flush();
    Some(f.sum)
}

/// Clears the row positions named by [`trace_rows`] when the forward returns.
pub struct TraceRows(());

impl Drop for TraceRows {
    fn drop(&mut self) {
        if TRACE_ANY.load(std::sync::atomic::Ordering::Relaxed) {
            trace_state()
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .rows
                .clear();
        }
    }
}

/// Name the absolute sequence position of each row of the forward about to
/// run, for as long as the returned guard lives.
pub fn trace_rows(positions: impl IntoIterator<Item = usize>) -> TraceRows {
    if TRACE_ANY.load(std::sync::atomic::Ordering::Relaxed) {
        let mut st = trace_state().lock().unwrap_or_else(|e| e.into_inner());
        st.rows.clear();
        st.rows.extend(positions.into_iter().map(|p| p as u32));
    }
    TraceRows(())
}

fn trace_record(layer_idx: usize, row: usize, indices: &[usize], weights: &[f32]) {
    let mut st = trace_state().lock().unwrap_or_else(|e| e.into_inner());
    let pos = st.rows.get(row).copied();
    let TraceState { open, current, .. } = &mut *st;
    let Some(f) = current.as_ref().and_then(|id| open.get_mut(id)) else {
        return;
    };
    let Some(pos) = pos else {
        f.sum.unplaced += 1;
        return;
    };
    let k = indices.len().min(weights.len());
    let mut rec = Vec::with_capacity(8 + 4 * k);
    rec.extend_from_slice(&pos.to_le_bytes());
    rec.extend_from_slice(&(layer_idx as u16).to_le_bytes());
    rec.extend_from_slice(&(k as u16).to_le_bytes());
    for &e in &indices[..k] {
        rec.extend_from_slice(&(e.min(u16::MAX as usize) as u16).to_le_bytes());
    }
    for &w in &weights[..k] {
        rec.extend_from_slice(&((w.clamp(0.0, 1.0) * 65535.0).round() as u16).to_le_bytes());
    }
    use std::io::Write;
    if f.out.write_all(&rec).is_err() {
        return;
    }
    let s = &mut f.sum;
    s.records += 1;
    s.k = s.k.max(k);
    s.positions = Some(
        s.positions
            .map_or((pos, pos), |(a, b)| (a.min(pos), b.max(pos))),
    );
    if s.layer_counts.len() <= layer_idx {
        s.layer_counts.resize(layer_idx + 1, Vec::new());
    }
    let counts = &mut s.layer_counts[layer_idx];
    for &e in &indices[..k] {
        if counts.len() <= e {
            counts.resize(e + 1, 0);
        }
        counts[e] += 1;
    }
    if s.weight_by_rank.len() < k {
        s.weight_by_rank.resize(k, 0.0);
    }
    for (acc, &w) in s.weight_by_rank.iter_mut().zip(&weights[..k]) {
        *acc += w as f64;
    }
}

/// Record one routed row. `row` is its index within the forward in flight
/// (0 for single-token decode); the trace maps it to a position.
pub fn record_moe_router_selection(
    layer_idx: usize,
    row: usize,
    indices: &[usize],
    weights: &[f32],
) {
    if TRACE_ANY.load(std::sync::atomic::Ordering::Relaxed) {
        trace_record(layer_idx, row, indices, weights);
    }
    MOE_ROUTER_HISTOGRAM.with(|hist| {
        let mut hist = hist.borrow_mut();
        let Some(hist) = hist.as_mut() else {
            return;
        };
        hist.routed_tokens += 1;
        let num_experts = hist.num_experts;
        let k_top = hist.k_top;
        let mut valid_experts = Vec::with_capacity(k_top);
        let mut layer_updates = Vec::with_capacity(k_top);
        let mut dropped_indices = 0u64;
        for (rank, &expert_idx) in indices.iter().take(k_top).enumerate() {
            if expert_idx >= num_experts {
                hist.dropped_indices += 1;
                dropped_indices += 1;
                continue;
            }
            let weight = weights.get(rank).copied().unwrap_or(0.0);
            if rank == 0 {
                hist.top1_histogram[expert_idx] += 1;
            }
            hist.topk_histogram[expert_idx] += 1;
            hist.routed_slots += 1;
            hist.weight_sums[expert_idx] += weight as f64;
            valid_experts.push(expert_idx);
            layer_updates.push((rank, expert_idx, weight));
        }
        let layer = hist.ensure_layer(layer_idx);
        layer.routed_tokens += 1;
        layer.dropped_indices += dropped_indices;
        for (rank, expert_idx, weight) in layer_updates {
            if rank == 0 {
                layer.top1_histogram[expert_idx] += 1;
            }
            layer.topk_histogram[expert_idx] += 1;
            layer.routed_slots += 1;
            layer.weight_sums[expert_idx] += weight as f64;
        }
        for i in 0..valid_experts.len() {
            for j in (i + 1)..valid_experts.len() {
                let a = valid_experts[i].min(valid_experts[j]);
                let b = valid_experts[i].max(valid_experts[j]);
                let key = (a as u64) * (num_experts as u64) + b as u64;
                *layer.cooccurrence.entry(key).or_insert(0) += 1;
            }
        }
    });
}

pub fn moe_router_histogram_active() -> bool {
    TRACE_ANY.load(std::sync::atomic::Ordering::Relaxed)
        || MOE_ROUTER_HISTOGRAM.with(|hist| hist.borrow().is_some())
}

pub fn router_index_i32_to_usize(idx: i32) -> usize {
    usize::try_from(idx).unwrap_or(usize::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trace_writes_placed_rows_of_the_selected_request_only() {
        let stem = std::env::temp_dir().join(format!("moe-trace-test-{}", std::process::id()));
        trace_open("a", &stem).unwrap();
        {
            let _rows = trace_rows([7, 8]);
            record_moe_router_selection(2, 1, &[5, 300], &[0.75, 0.25]); // pos 8
            record_moe_router_selection(2, 2, &[1, 2], &[0.5, 0.5]); // no row 2
            trace_select("other"); // not traced: nothing is current
            record_moe_router_selection(3, 0, &[1, 2], &[0.5, 0.5]);
            trace_select("a");
        }
        record_moe_router_selection(4, 0, &[1, 2], &[0.5, 0.5]); // guard dropped
        let sum = trace_close("a").unwrap();
        assert!(trace_close("a").is_none(), "close is idempotent");
        assert!(!moe_router_histogram_active());

        let bytes = std::fs::read(&sum.path).unwrap();
        std::fs::remove_file(&sum.path).unwrap();
        let mut want = Vec::new();
        for v in [
            8u32.to_le_bytes().to_vec(),
            2u16.to_le_bytes().to_vec(),
            2u16.to_le_bytes().to_vec(),
        ] {
            want.extend(v);
        }
        for v in [5u16, 300, 49151, 16384] {
            want.extend(v.to_le_bytes());
        }
        assert_eq!(bytes, want);
        assert_eq!((sum.records, sum.unplaced, sum.k), (1, 2, 2));
        assert_eq!(sum.positions, Some((8, 8)));
        assert_eq!(sum.layer_counts[2][300], 1);
    }
}
