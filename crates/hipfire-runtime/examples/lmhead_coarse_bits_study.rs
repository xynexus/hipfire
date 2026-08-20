// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! How many bits does the two-stage lm_head COARSE tier actually need?
//!
//! The coarse tier only has to *rank* — it selects a top-K shortlist that a
//! full-precision fine pass then rescores exactly. So its quality metric is not
//! reconstruction error, it is: **where does the true argmax land in the coarse
//! ordering?** If the true argmax never falls below rank K, the two-stage path
//! is greedy-exact and the coarse codes can be as crude as that bound allows.
//!
//! The shipped tier is 4-bit, chosen so it could double as a standalone 4-bit
//! lm_head — a dual purpose that was never measured. This measures the ranking
//! half directly, on host, so 1- and 3-bit can be evaluated without first
//! writing their packers.
//!
//! Queries must be REAL post-final-norm hidden states, not random vectors: the
//! whole question is how the coarse ordering behaves on the actual query
//! distribution. Capture them with
//!   `HIPFIRE_DUMP_HIDDEN=<prefix> HIPFIRE_DUMP_HIDDEN_ALL=1 \
//!    HIPFIRE_DUMP_HIDDEN_LAYER=<n_layers>` → `<prefix>.fnorm`, raw f32 [dim].
//!
//! usage: lmhead_coarse_bits_study <model.hfq> <fnorm.bin> [n_queries]

use hipfire_runtime::hfq::HfqFile;
use rayon::prelude::*;
use std::path::Path;

/// Symmetric code range for `bits`. 1-bit is sign-only (±1) rather than the
/// two's-complement [-1, 0], which would carry no direction information at all.
fn code_range(bits: usize) -> (f32, f32, f32) {
    if bits == 1 {
        return (-1.0, 1.0, 1.0);
    }
    let half = (1i32 << (bits - 1)) as f32;
    (-half, half - 1.0, half)
}

/// Quantize each row's UNIT direction to `bits`, keeping the exact row norm in
/// the scale — the encoding `build_lmhead_coarse_bf16` uses, with the clip
/// constant lifted out so it can be swept (3.0 is what ships).
fn encode_coarse(
    head: &[u16],
    vocab: usize,
    hidden: usize,
    bits: usize,
    clip: f32,
) -> (Vec<i8>, Vec<f32>) {
    let (lo, hi, max_mag) = code_range(bits);
    let unit_scale = clip / (max_mag * (hidden as f32).sqrt());
    let inv = 1.0 / unit_scale;
    let mut codes = vec![0i8; vocab * hidden];
    let mut scales = vec![0f32; vocab];
    codes
        .par_chunks_mut(hidden)
        .zip(scales.par_iter_mut())
        .enumerate()
        .for_each(|(v, (crow, scale))| {
            let row = &head[v * hidden..(v + 1) * hidden];
            let mut norm = 0f32;
            for &u in row {
                let x = f32::from_bits((u as u32) << 16);
                norm += x * x;
            }
            let norm = norm.sqrt();
            if norm > 0.0 {
                let ni = inv / norm;
                for (i, &u) in row.iter().enumerate() {
                    let x = f32::from_bits((u as u32) << 16);
                    crow[i] = if bits == 1 {
                        if x >= 0.0 {
                            1
                        } else {
                            -1
                        }
                    } else {
                        (x * ni).round().clamp(lo, hi) as i8
                    };
                }
            }
            *scale = norm * unit_scale;
        });
    (codes, scales)
}

/// Rank of `target` in `scores` — how many rows outscore it. Rank 0 means the
/// coarse tier already puts the true argmax first, so K=1 would suffice.
fn rank_of(scores: &[f32], target: usize) -> usize {
    let t = scores[target];
    scores.par_iter().filter(|&&s| s > t).count()
}

fn pct(sorted: &[usize], p: f64) -> usize {
    if sorted.is_empty() {
        return 0;
    }
    sorted[(((sorted.len() - 1) as f64) * p).round() as usize]
}

fn main() {
    let argv: Vec<String> = std::env::args().collect();
    let model = argv
        .get(1)
        .expect("usage: <model.hfq> <fnorm.bin> [n_queries]");
    let fnorm_path = argv
        .get(2)
        .expect("usage: <model.hfq> <fnorm.bin> [n_queries]");
    let want_q: usize = argv.get(3).and_then(|s| s.parse().ok()).unwrap_or(64);

    let hfq = HfqFile::open(Path::new(model)).expect("open model");
    let (info, bytes) = hfq
        .tensor_data_cow("lm_head.weight")
        .or_else(|| hfq.tensor_data_cow("model.language_model.embed_tokens.weight"))
        .expect("no lm_head.weight or tied embed_tokens.weight");
    let (vocab, hidden) = (info.shape[0] as usize, info.shape[1] as usize);
    // `expand_bf16_index` deliberately leaves HEAD tensors LUT3-packed (it is
    // GPU-decodable), so `tensor_data_cow` hands them back still coded.
    let bytes: std::borrow::Cow<[u8]> = if info.quant_type == 49 {
        std::borrow::Cow::Owned(
            hipfire_primitives::bf16_lut3::decode(&bytes, vocab * hidden)
                .expect("Bf16Lut3 head payload is corrupt or truncated"),
        )
    } else {
        bytes
    };
    assert_eq!(
        bytes.len(),
        vocab * hidden * 2,
        "expected a bf16 head ({vocab}x{hidden}); got {} bytes (qt {})",
        bytes.len(),
        info.quant_type
    );
    let head: Vec<u16> = bytes
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect();
    println!("head [{vocab}, {hidden}] from qt {}", info.quant_type);

    // Real queries, strided across the capture so decode steps from every
    // prompt are represented rather than just the first prompt's.
    let raw = std::fs::read(fnorm_path).expect("read fnorm");
    let total = raw.len() / 4 / hidden;
    assert!(total > 0, "fnorm file holds no complete [{hidden}] vectors");
    let stride = (total / want_q).max(1);
    let queries: Vec<Vec<f32>> = (0..total)
        .step_by(stride)
        .take(want_q)
        .map(|i| {
            raw[i * hidden * 4..(i + 1) * hidden * 4]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        })
        .collect();
    println!(
        "queries: {} of {total} captured (stride {stride})\n",
        queries.len()
    );

    // Ground truth: exact bf16 argmax per query.
    let exact_argmax: Vec<usize> = queries
        .iter()
        .map(|q| {
            let scores: Vec<f32> = (0..vocab)
                .into_par_iter()
                .map(|v| {
                    let row = &head[v * hidden..(v + 1) * hidden];
                    row.iter()
                        .zip(q)
                        .map(|(&u, &x)| f32::from_bits((u as u32) << 16) * x)
                        .sum::<f32>()
                })
                .collect();
            let mut best = 0usize;
            for v in 1..vocab {
                if scores[v] > scores[best] {
                    best = v;
                }
            }
            best
        })
        .collect();

    println!(
        "{:<5} {:<6} {:>9} {:>9} {:>9} {:>9}  {}",
        "bits", "clip", "p50", "p99", "max", "minK", "recall@1 at K=32 / 256 / 2048"
    );
    for bits in [1usize, 2, 3, 4] {
        for clip in [1.5f32, 2.0, 2.5, 3.0, 4.0] {
            let (codes, scales) = encode_coarse(&head, vocab, hidden, bits, clip);
            let mut ranks: Vec<usize> = queries
                .iter()
                .zip(&exact_argmax)
                .map(|(q, &tgt)| {
                    let scores: Vec<f32> = (0..vocab)
                        .into_par_iter()
                        .map(|v| {
                            let row = &codes[v * hidden..(v + 1) * hidden];
                            let d: f32 = row.iter().zip(q).map(|(&c, &x)| c as f32 * x).sum();
                            scales[v] * d
                        })
                        .collect();
                    rank_of(&scores, tgt)
                })
                .collect();
            ranks.sort_unstable();
            let at = |k: usize| {
                let hit = ranks.iter().filter(|&&r| r < k).count();
                100.0 * hit as f64 / ranks.len() as f64
            };
            println!(
                "{bits:<5} {clip:<6.1} {:>9} {:>9} {:>9} {:>9}  {:>6.1}% {:>6.1}% {:>6.1}%",
                pct(&ranks, 0.50),
                pct(&ranks, 0.99),
                ranks[ranks.len() - 1],
                ranks[ranks.len() - 1] + 1,
                at(32),
                at(256),
                at(2048)
            );
        }
    }
    println!(
        "\nminK = smallest top-K that contained the true argmax for EVERY query here.\n\
         Bits/weight of the coarse tier: 1→0.125, 2→0.25, 3→0.375, 4→0.5 bytes per weight."
    );
}
