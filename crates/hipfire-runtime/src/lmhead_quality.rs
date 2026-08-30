// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! Per-model two-stage lm_head quality counters.
//!
//! The two-stage path selects a coarse shortlist and rescores only those rows,
//! so its correctness rests entirely on "did the shortlist contain the row the
//! full fine pass would have picked?" — and nothing in serving has ever looked.
//! This accumulates the signals that are **free** at decode time and flushes a
//! fixed-layout binary record per model.
//!
//! ## What this can and cannot prove
//!
//! It records the SHORTLIST SIZE distribution, not recall. True recall would
//! need the full fine scoring of every vocab row, which is exactly the work the
//! two-stage path exists to skip — so it is not obtainable for free, and this
//! module deliberately does not claim it. What the shortlist distribution *does*
//! catch is a gross regression: a new head whose coarse ranking behaves
//! differently, a K default carried over from another model, or a query
//! distribution the offline sweep never saw. Treat it as a **regression
//! detector, not a certification**, and do not surface these numbers to users as
//! "recall".
//!
//! ## Why a binary struct rather than a log line
//!
//! The record is fixed-size and versioned, so a reader can `mmap` it, check the
//! magic and version, and reject anything it does not understand. It lives in
//! memory during serving and is flushed on a cadence — writing per token would
//! dominate the very cost being measured.

use std::io::Write;
use std::sync::Mutex;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// `b"LMQ\0"`. Present so a truncated or unrelated file is rejected rather than
/// read as garbage counters.
pub const LMQ_MAGIC: [u8; 4] = *b"LMQ\0";
/// Bump on ANY layout change. Readers must refuse a version they do not know
/// rather than reinterpret the bytes.
pub const LMQ_VERSION: u16 = 2;
/// Flush cadence. Also flushed on generation completion, whichever comes first.
const FLUSH_EVERY: Duration = Duration::from_secs(10);
/// Shortlist-size histogram buckets, by power of two: `[0,1) [1,2) [2,4) …`.
/// 20 buckets covers up to 2^19 = 524288, past any real vocab shortlist.
pub const LMQ_BUCKETS: usize = 20;

/// Fixed-layout counters for one model's two-stage lm_head.
///
/// `#[repr(C)]` with only fixed-width integers, so the on-disk bytes are the
/// in-memory bytes on every target this project builds for. No padding is
/// implied: fields are ordered widest-first.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct LmheadQuality {
    pub magic: [u8; 4],
    pub version: u16,
    /// Coarse tier bit depth (2 or 4). 0 = two-stage never engaged.
    pub coarse_bits: u16,

    /// Digest of the artifact's producer-stamped content hash
    /// (`hipfire_model::model_hash` → HFQ `quantization_hash`). This is what
    /// makes the counters belong to a specific BUILD rather than a filename:
    /// a re-quant under the same name, or two artifacts whose stems collide
    /// after tag-stripping (`--oq4.25` vs `--oq4.25++`), get different keys and
    /// therefore different files, instead of silently merging.
    pub model_key: u64,

    /// Unix seconds, first and most recent record into this struct.
    pub first_ts: u64,
    pub last_ts: u64,
    /// Decode steps that actually took the two-stage path.
    pub tokens: u64,
    /// Sum of shortlist sizes, for the mean. Saturating.
    pub shortlist_sum: u64,
    /// Requests bypassed because `top_p < 1.0` while sampling.
    pub bypass_top_p: u64,
    /// Requests bypassed because the caller needed full-vocab logits.
    pub bypass_full_logits: u64,
    /// Times K was raised to cover the sampler's `top_k`.
    pub k_raised: u64,

    pub shortlist_min: u32,
    pub shortlist_max: u32,
    /// Configured K, and the effective K after the sampling policy.
    pub k_cfg: u32,
    pub k_eff: u32,
    pub vocab: u32,
    pub hidden: u32,
    /// Power-of-two histogram of shortlist size.
    pub buckets: [u32; LMQ_BUCKETS],
}

impl Default for LmheadQuality {
    fn default() -> Self {
        Self {
            magic: LMQ_MAGIC,
            version: LMQ_VERSION,
            coarse_bits: 0,
            model_key: 0,
            first_ts: 0,
            last_ts: 0,
            tokens: 0,
            shortlist_sum: 0,
            bypass_top_p: 0,
            bypass_full_logits: 0,
            k_raised: 0,
            shortlist_min: u32::MAX,
            shortlist_max: 0,
            k_cfg: 0,
            k_eff: 0,
            vocab: 0,
            hidden: 0,
            buckets: [0; LMQ_BUCKETS],
        }
    }
}

impl LmheadQuality {
    /// Mean shortlist size, or `None` before any token.
    pub fn mean_shortlist(&self) -> Option<f64> {
        (self.tokens > 0).then(|| self.shortlist_sum as f64 / self.tokens as f64)
    }

    fn as_bytes(&self) -> &[u8] {
        // SAFETY: `#[repr(C)]` over fixed-width integer fields and arrays of
        // them — no padding bytes are read that were not written by `Default`,
        // and there are no pointers or references to invalidate.
        unsafe {
            std::slice::from_raw_parts(
                (self as *const Self) as *const u8,
                std::mem::size_of::<Self>(),
            )
        }
    }

    fn record_shortlist(&mut self, size: usize) {
        let now = unix_now();
        if self.tokens == 0 {
            self.first_ts = now;
        }
        self.last_ts = now;
        self.tokens = self.tokens.saturating_add(1);
        let s = size.min(u32::MAX as usize) as u32;
        self.shortlist_sum = self.shortlist_sum.saturating_add(s as u64);
        self.shortlist_min = self.shortlist_min.min(s);
        self.shortlist_max = self.shortlist_max.max(s);
        let b = (usize::BITS - size.leading_zeros()) as usize;
        self.buckets[b.min(LMQ_BUCKETS - 1)] += 1;
    }
}

/// FNV-1a over the content-hash string. Only needs to be stable and
/// well-distributed for keying — it is not a security boundary, and the
/// authoritative hash is the one it digests.
pub fn model_key(model_hash: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in model_hash.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

struct Store {
    rec: LmheadQuality,
    /// Artifact stem, e.g. `Qwen3.5-9B--oq4.25++`. Empty = nowhere to flush yet.
    stem: String,
    last_flush: Option<Instant>,
    dirty: bool,
}

static STORE: Mutex<Option<Store>> = Mutex::new(None);

fn with_store<R>(f: impl FnOnce(&mut Store) -> R) -> Option<R> {
    let mut g = STORE.lock().ok()?;
    let s = g.get_or_insert_with(|| Store {
        rec: LmheadQuality::default(),
        stem: String::new(),
        last_flush: None,
        dirty: false,
    });
    Some(f(s))
}

/// Bind the counters to a model. Called on load; resets the record, because
/// merging two models' shortlist distributions would be meaningless.
pub fn set_model(stem: &str, model_hash: Option<&str>, vocab: usize, hidden: usize) {
    let key = model_hash.map(model_key).unwrap_or(0);
    with_store(|s| {
        // Key on CONTENT, not the display stem: the stem is only how a human
        // finds the file.
        if s.rec.model_key != key {
            s.rec = LmheadQuality::default();
            s.rec.model_key = key;
            s.stem = stem.to_string();
            s.last_flush = None;
            s.dirty = false;
        }
        s.rec.vocab = vocab.min(u32::MAX as usize) as u32;
        s.rec.hidden = hidden.min(u32::MAX as usize) as u32;
    });
}

/// One decode step took the two-stage path with `shortlist` candidates.
pub fn record_token(shortlist: usize, coarse_bits: usize, k_cfg: usize, k_eff: usize) {
    with_store(|s| {
        s.rec.coarse_bits = coarse_bits as u16;
        s.rec.k_cfg = k_cfg.min(u32::MAX as usize) as u32;
        s.rec.k_eff = k_eff.min(u32::MAX as usize) as u32;
        if k_eff > k_cfg {
            s.rec.k_raised = s.rec.k_raised.saturating_add(1);
        }
        s.rec.record_shortlist(shortlist);
        s.dirty = true;
        // Time-based flush is checked here rather than on a timer thread: the
        // executor is serial, so this is the only place that reliably runs.
        let due = s
            .last_flush
            .map(|t| t.elapsed() >= FLUSH_EVERY)
            .unwrap_or(true);
        if due {
            flush_locked(s);
        }
    });
}

/// A request bypassed the two-stage path. Recorded so the counters explain a
/// low token count instead of just showing one.
pub fn record_bypass(nucleus: bool) {
    with_store(|s| {
        if nucleus {
            s.rec.bypass_top_p = s.rec.bypass_top_p.saturating_add(1);
        } else {
            s.rec.bypass_full_logits = s.rec.bypass_full_logits.saturating_add(1);
        }
        s.dirty = true;
    });
}

/// Flush now — call at generation completion.
pub fn flush() {
    with_store(flush_locked);
}

/// Flushes when it goes out of scope.
///
/// A request handler has many exit paths — early validation returns, error
/// arms, the normal end — and a flush call at each is one `return` away from
/// being wrong. Bind this once at the top instead.
pub struct FlushOnDrop;

impl Drop for FlushOnDrop {
    fn drop(&mut self) {
        flush();
    }
}

/// Bind at the top of a request handler: `let _lmq = flush_on_drop();`
pub fn flush_on_drop() -> FlushOnDrop {
    FlushOnDrop
}

fn flush_locked(s: &mut Store) {
    s.last_flush = Some(Instant::now());
    if !s.dirty || s.stem.is_empty() {
        return;
    }
    let Some(home) = std::env::var_os("HOME") else {
        return;
    };
    let dir = std::path::Path::new(&home).join(".hipfire").join("quality");
    if std::fs::create_dir_all(&dir).is_err() {
        return;
    }
    // Write-then-rename: a reader mmapping this must never see a torn record.
    let stem = sanitize(&s.stem);
    let key = s.rec.model_key;
    let final_path = dir.join(format!("{stem}.{key:016x}.lmq"));
    let tmp_path = dir.join(format!(".{stem}.{key:016x}.lmq.tmp"));
    let wrote = std::fs::File::create(&tmp_path)
        .and_then(|mut f| f.write_all(s.rec.as_bytes()).and_then(|()| f.sync_all()));
    if wrote.is_ok() && std::fs::rename(&tmp_path, &final_path).is_ok() {
        s.dirty = false;
    } else {
        let _ = std::fs::remove_file(&tmp_path);
    }
}

/// Artifact stems carry `+` and `.`; keep those, drop anything that would
/// escape the directory or confuse a shell.
fn sanitize(stem: &str) -> String {
    stem.chars()
        .map(|c| match c {
            'a'..='z' | 'A'..='Z' | '0'..='9' | '.' | '+' | '-' | '_' => c,
            _ => '_',
        })
        .collect()
}

/// Read a record back. `None` if absent, truncated, or a version this build
/// does not know — never a partial reinterpretation.
pub fn read(path: &std::path::Path) -> Option<LmheadQuality> {
    let bytes = std::fs::read(path).ok()?;
    if bytes.len() != std::mem::size_of::<LmheadQuality>() {
        return None;
    }
    // SAFETY: length checked against the exact struct size; the type is
    // `#[repr(C)]` over integers, so every bit pattern is a valid value.
    let rec: LmheadQuality = unsafe { std::ptr::read_unaligned(bytes.as_ptr() as *const _) };
    (rec.magic == LMQ_MAGIC && rec.version == LMQ_VERSION).then_some(rec)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_roundtrips_through_bytes() {
        let mut r = LmheadQuality::default();
        r.record_shortlist(520);
        r.record_shortlist(600);
        assert_eq!(r.tokens, 2);
        assert_eq!(r.shortlist_min, 520);
        assert_eq!(r.shortlist_max, 600);
        assert_eq!(r.mean_shortlist(), Some(560.0));

        let dir = std::env::temp_dir().join("hipfire-lmq-test");
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("roundtrip.lmq");
        std::fs::write(&p, r.as_bytes()).unwrap();
        let back = read(&p).expect("valid record");
        assert_eq!(back.tokens, 2);
        assert_eq!(back.shortlist_max, 600);
        assert_eq!(back.magic, LMQ_MAGIC);

        // A wrong-length file is rejected rather than read as a partial struct.
        std::fs::write(&p, [0u8; 8]).unwrap();
        assert!(read(&p).is_none());
        // So is a good-length file with a bad magic — this is the guard that
        // stops an unrelated file being read as counters.
        let mut bad = r;
        bad.magic = *b"XXXX";
        std::fs::write(&p, bad.as_bytes()).unwrap();
        assert!(read(&p).is_none());
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn buckets_are_power_of_two() {
        let mut r = LmheadQuality::default();
        r.record_shortlist(0); // bucket 0
        r.record_shortlist(1); // bucket 1
        r.record_shortlist(3); // bucket 2
        r.record_shortlist(4); // bucket 3
        assert_eq!(r.buckets[0], 1);
        assert_eq!(r.buckets[1], 1);
        assert_eq!(r.buckets[2], 1);
        assert_eq!(r.buckets[3], 1);
        // Anything past the top bucket saturates into it rather than panicking.
        r.record_shortlist(usize::MAX);
        assert_eq!(r.buckets[LMQ_BUCKETS - 1], 1);
    }
}
