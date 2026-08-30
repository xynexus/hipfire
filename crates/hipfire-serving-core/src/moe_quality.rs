// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! Persistent per-model MoE expert-utilisation counters.
//!
//! The router histogram was already collected during serving, but only when a
//! request asked for an evidence directory (`DaemonMoeRouterHistogramGuard::
//! start`: `evidence_dir.is_some() && num_experts > 0`), so ordinary serving
//! neither collected nor persisted it. This accumulates it across runs.
//!
//! ## Why sampled, not always-on
//!
//! Collection is NOT free. With the histogram active,
//! `qwen35/moe_decode.rs` downloads the top-K indices AND weights from the
//! device on **every MoE layer of every token** — two blocking D2H syncs per
//! layer. On a 48-layer MoE that is 96 syncs per token against a ~17.5 ms token,
//! i.e. a double-digit percentage throughput cost. So capture runs on a sampled
//! subset of *generations*: a sampled generation pays the full cost and yields a
//! complete routing profile, and the amortised cost is that divided by the
//! sampling period.
//!
//! Sampling whole generations rather than individual tokens keeps each captured
//! profile internally coherent — a prompt's routing is not independent across
//! its own tokens, so a token-sampled profile would misrepresent the
//! co-occurrence structure it is most useful for.
//!
//! ## What it is for
//!
//! Expert *utilisation skew*. A corpus that never exercises some experts leaves
//! them starved, and that has already bitten this project once — the ZAYA
//! strict-coverage failure was root-caused to an English-only calibration
//! corpus. This is the serving-side counterpart: what the live traffic actually
//! routes to, as opposed to what calibration assumed.

use std::io::Write;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

/// `b"MOE\0"`.
pub const MOE_MAGIC: [u8; 4] = *b"MOE\0";
pub const MOE_VERSION: u16 = 2;

/// Capture one generation in every `MOE_SAMPLE_PERIOD`. Chosen so the amortised
/// cost of the per-layer D2H syncs stays well under 1%, while still filling a
/// 128-expert histogram quickly — one 200-token generation on a 48-layer,
/// k_top=8 model contributes ~76 800 routed slots.
const MOE_SAMPLE_PERIOD: u64 = 16;

static GENERATION_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Fixed header; the two `num_experts`-long `u64` arrays follow it in the file.
///
/// Variable-length payload is why this is a header plus arrays rather than one
/// flat struct: expert count is a per-model property, so it cannot be baked in.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct MoeQualityHeader {
    pub magic: [u8; 4],
    pub version: u16,
    pub k_top: u16,
    pub num_experts: u32,
    pub _reserved: u32,
    /// Digest of the artifact's producer-stamped content hash — see
    /// `hipfire_runtime::lmhead_quality::model_key`. Keys the counters to a
    /// specific BUILD so a re-quant never merges into the previous one's
    /// expert distribution.
    pub model_key: u64,
    /// Sampled generations merged into these counters.
    pub generations: u64,
    pub routed_tokens: u64,
    pub routed_slots: u64,
    pub dropped_indices: u64,
    pub first_ts: u64,
    pub last_ts: u64,
}

/// Decoded record: header plus the per-expert counts.
#[derive(Clone, Debug)]
pub struct MoeQuality {
    pub header: MoeQualityHeader,
    /// Times each expert appeared anywhere in a token's top-K.
    pub topk: Vec<u64>,
    /// Times each expert was the top-1 choice.
    pub top1: Vec<u64>,
}

impl MoeQuality {
    fn new(num_experts: usize, k_top: usize, model_key: u64) -> Self {
        Self {
            header: MoeQualityHeader {
                magic: MOE_MAGIC,
                version: MOE_VERSION,
                k_top: k_top.min(u16::MAX as usize) as u16,
                num_experts: num_experts.min(u32::MAX as usize) as u32,
                _reserved: 0,
                model_key,
                generations: 0,
                routed_tokens: 0,
                routed_slots: 0,
                dropped_indices: 0,
                first_ts: 0,
                last_ts: 0,
            },
            topk: vec![0; num_experts],
            top1: vec![0; num_experts],
        }
    }

    /// Share of routed slots taken by the busiest expert. 1/num_experts is
    /// perfectly uniform; a large value means the traffic is concentrating.
    pub fn max_share(&self) -> Option<f64> {
        let total: u64 = self.topk.iter().sum();
        (total > 0).then(|| *self.topk.iter().max().unwrap_or(&0) as f64 / total as f64)
    }

    /// Experts that never appeared in any top-K. These are the ones a
    /// calibration corpus would have starved.
    pub fn unused_experts(&self) -> usize {
        self.topk.iter().filter(|&&c| c == 0).count()
    }

    fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(
            std::mem::size_of::<MoeQualityHeader>() + (self.topk.len() + self.top1.len()) * 8,
        );
        // SAFETY: `#[repr(C)]` over fixed-width integers; no pointers, no padding
        // that was not zero-initialised by construction.
        out.extend_from_slice(unsafe {
            std::slice::from_raw_parts(
                (&self.header as *const MoeQualityHeader) as *const u8,
                std::mem::size_of::<MoeQualityHeader>(),
            )
        });
        for v in self.topk.iter().chain(self.top1.iter()) {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }
}

struct Store {
    rec: Option<MoeQuality>,
    stem: String,
    model_key: u64,
    dirty: bool,
}

static STORE: Mutex<Option<Store>> = Mutex::new(None);

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// True when this generation should pay for router capture.
///
/// Call exactly once per generation — it advances the counter.
pub fn should_sample() -> bool {
    GENERATION_COUNTER.fetch_add(1, Ordering::Relaxed) % MOE_SAMPLE_PERIOD == 0
}

/// Bind to a model; resets when the model changes.
pub fn set_model(stem: &str, model_hash: Option<&str>) {
    let key = model_hash
        .map(hipfire_runtime::lmhead_quality::model_key)
        .unwrap_or(0);
    if let Ok(mut g) = STORE.lock() {
        let s = g.get_or_insert_with(|| Store {
            rec: None,
            stem: String::new(),
            model_key: 0,
            dirty: false,
        });
        // Key on CONTENT: the stem is only how a human finds the file.
        if s.model_key != key {
            s.rec = None;
            s.stem = stem.to_string();
            s.model_key = key;
            s.dirty = false;
        }
    }
}

/// Merge one sampled generation's histogram, then flush.
///
/// Flushing here rather than on a timer is deliberate: a sampled generation is
/// rare and already expensive, so the write is free by comparison, and it means
/// the file is current whenever anyone looks.
pub fn record(hist: &hipfire_arch_qwen35::qwen35::MoeRouterHistogram) {
    if hist.num_experts == 0 || hist.routed_slots == 0 {
        return;
    }
    let Ok(mut g) = STORE.lock() else { return };
    let Some(s) = g.as_mut() else { return };
    let key = s.model_key;
    let rec = s
        .rec
        .get_or_insert_with(|| MoeQuality::new(hist.num_experts, hist.k_top, key));
    // A model swap that kept the same stem, or a config change, would make the
    // arrays disagree. Restart rather than merge mismatched shapes.
    if rec.topk.len() != hist.num_experts {
        *rec = MoeQuality::new(hist.num_experts, hist.k_top, key);
    }
    let now = unix_now();
    if rec.header.generations == 0 {
        rec.header.first_ts = now;
    }
    rec.header.last_ts = now;
    rec.header.generations += 1;
    rec.header.routed_tokens = rec.header.routed_tokens.saturating_add(hist.routed_tokens);
    rec.header.routed_slots = rec.header.routed_slots.saturating_add(hist.routed_slots);
    rec.header.dropped_indices = rec
        .header
        .dropped_indices
        .saturating_add(hist.dropped_indices);
    for (dst, src) in rec.topk.iter_mut().zip(hist.topk_histogram.iter()) {
        *dst = dst.saturating_add(*src);
    }
    for (dst, src) in rec.top1.iter_mut().zip(hist.top1_histogram.iter()) {
        *dst = dst.saturating_add(*src);
    }
    s.dirty = true;
    flush_locked(s);
}

fn flush_locked(s: &mut Store) {
    if !s.dirty || s.stem.is_empty() {
        return;
    }
    let (Some(rec), Some(home)) = (s.rec.as_ref(), std::env::var_os("HOME")) else {
        return;
    };
    let dir = std::path::Path::new(&home).join(".hipfire").join("quality");
    if std::fs::create_dir_all(&dir).is_err() {
        return;
    }
    let stem = sanitize(&s.stem);
    let key = s.model_key;
    let final_path = dir.join(format!("{stem}.{key:016x}.moe.lmq"));
    let tmp_path = dir.join(format!(".{stem}.{key:016x}.moe.lmq.tmp"));
    let bytes = rec.to_bytes();
    // Write-then-rename so a concurrent reader never sees a half-written array.
    let wrote = std::fs::File::create(&tmp_path)
        .and_then(|mut f| f.write_all(&bytes).and_then(|()| f.sync_all()));
    if wrote.is_ok() && std::fs::rename(&tmp_path, &final_path).is_ok() {
        s.dirty = false;
    } else {
        let _ = std::fs::remove_file(&tmp_path);
    }
}

fn sanitize(stem: &str) -> String {
    stem.chars()
        .map(|c| match c {
            'a'..='z' | 'A'..='Z' | '0'..='9' | '.' | '+' | '-' | '_' => c,
            _ => '_',
        })
        .collect()
}

/// Read a record back. `None` on any inconsistency — wrong magic, unknown
/// version, or a length that disagrees with the declared expert count.
pub fn read(path: &std::path::Path) -> Option<MoeQuality> {
    let bytes = std::fs::read(path).ok()?;
    let hsize = std::mem::size_of::<MoeQualityHeader>();
    if bytes.len() < hsize {
        return None;
    }
    // SAFETY: length checked; `#[repr(C)]` over integers, so any bit pattern is
    // a valid value. Read unaligned because `Vec<u8>` gives no alignment.
    let header: MoeQualityHeader = unsafe { std::ptr::read_unaligned(bytes.as_ptr() as *const _) };
    if header.magic != MOE_MAGIC || header.version != MOE_VERSION {
        return None;
    }
    let n = header.num_experts as usize;
    if bytes.len() != hsize + n * 16 {
        return None;
    }
    let rd = |off: usize| {
        (0..n)
            .map(|i| {
                let p = off + i * 8;
                u64::from_le_bytes(bytes[p..p + 8].try_into().unwrap())
            })
            .collect::<Vec<u64>>()
    };
    Some(MoeQuality {
        header,
        topk: rd(hsize),
        top1: rd(hsize + n * 8),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrips_and_rejects_malformed() {
        let mut r = MoeQuality::new(4, 2, 0xdead_beef);
        r.header.generations = 3;
        r.topk = vec![10, 0, 5, 5];
        r.top1 = vec![8, 0, 1, 1];
        assert_eq!(r.unused_experts(), 1);
        assert_eq!(r.max_share(), Some(0.5));

        let dir = std::env::temp_dir().join("hipfire-moe-lmq-test");
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("rt.moe.lmq");
        std::fs::write(&p, r.to_bytes()).unwrap();
        let back = read(&p).expect("valid record");
        assert_eq!(back.header.num_experts, 4);
        assert_eq!(back.header.model_key, 0xdead_beef);
        assert_eq!(back.topk, vec![10, 0, 5, 5]);
        assert_eq!(back.top1, vec![8, 0, 1, 1]);

        // A file whose length disagrees with the declared expert count is
        // rejected rather than read as a short array.
        let mut truncated = r.to_bytes();
        truncated.truncate(truncated.len() - 8);
        std::fs::write(&p, &truncated).unwrap();
        assert!(read(&p).is_none());
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn sampling_period_hits_one_in_n() {
        // The counter is global, so assert the RATE over a window rather than
        // which specific calls fire.
        let hits = (0..MOE_SAMPLE_PERIOD * 4)
            .filter(|_| should_sample())
            .count();
        assert_eq!(hits, 4);
    }
}
