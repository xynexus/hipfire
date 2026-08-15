// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! What does a STANDALONE Opus lm_head cost in quality?
//!
//! Companion to `lmhead_coarse_bits_study`, which asks how few bits the
//! two-stage COARSE tier needs to *rank*. This asks the other question: how few
//! bits an Opus head needs to be the sole storage for the logits — the
//! VRAM-vs-quality trade rather than the VRAM-vs-performance one.
//!
//! Measured head-only, against the exact bf16 head on the same real queries, so
//! nothing from body quantization contaminates the number. `OqPlusCompact`
//! spends `4.0625 + n_out/16` b/w (`n_out` int8 overlays per 256-group), so the
//! 4 → 4.25 range is `n_out` 0..3 — the sweep that decides whether the shipped
//! 4.25 is buying anything.
//!
//! Capture queries exactly as the coarse study does (`<prefix>.fnorm`).
//!
//! usage: lmhead_opus_kld_study <model.hfq> <fnorm.bin> [n_queries]

use hipfire_primitives::fwht::{gen_fwht_signs, signed_fwht};
use hipfire_quantize::codecs;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::oq8_arch::oq8_arch_load;
use rayon::prelude::*;
use std::path::Path;

const GROUP: usize = 256;

/// Softmax in f64 with the max subtracted — the logit spreads here are wide
/// enough that a naive f32 exp underflows the tail we are trying to measure.
fn softmax(logits: &[f32]) -> Vec<f64> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
    let mut p: Vec<f64> = logits.iter().map(|&l| (l as f64 - max).exp()).collect();
    let sum: f64 = p.iter().sum();
    for v in &mut p {
        *v /= sum;
    }
    p
}

/// KL(reference || candidate) in nats.
fn kld(reference: &[f64], candidate: &[f64]) -> f64 {
    reference
        .iter()
        .zip(candidate)
        .filter(|(&r, _)| r > 0.0)
        .map(|(&r, &c)| r * (r / c.max(1e-300)).ln())
        .sum()
}

fn argmax(v: &[f32]) -> usize {
    let mut best = 0;
    for i in 1..v.len() {
        if v[i] > v[best] {
            best = i;
        }
    }
    best
}

/// Logits from the combined Opus layout: `[int8 codes m*k][f32 scales m*ng]`,
/// dotted against the PRE-ROTATED query. The quantizer rotates each weight
/// group by the signed FWHT, so the activation carries the matching rotation —
/// that is the same split the runtime GEMV uses.
fn opus_logits(combined: &[u8], vocab: usize, hidden: usize, x_rot: &[f32]) -> Vec<f32> {
    let ng = hidden / GROUP;
    let codes = &combined[..vocab * hidden];
    let scale_bytes = &combined[vocab * hidden..];
    (0..vocab)
        .into_par_iter()
        .map(|v| {
            let row = &codes[v * hidden..(v + 1) * hidden];
            let mut acc = 0f32;
            for g in 0..ng {
                let s = f32::from_le_bytes([
                    scale_bytes[(v * ng + g) * 4],
                    scale_bytes[(v * ng + g) * 4 + 1],
                    scale_bytes[(v * ng + g) * 4 + 2],
                    scale_bytes[(v * ng + g) * 4 + 3],
                ]);
                let mut d = 0f32;
                for i in 0..GROUP {
                    d += (row[g * GROUP + i] as i8) as f32 * x_rot[g * GROUP + i];
                }
                acc += s * d;
            }
            acc
        })
        .collect()
}

fn main() {
    let argv: Vec<String> = std::env::args().collect();
    let model = argv
        .get(1)
        .expect("usage: <model.hfq> <fnorm.bin> [n_queries]");
    let fnorm_path = argv
        .get(2)
        .expect("usage: <model.hfq> <fnorm.bin> [n_queries]");
    let want_q: usize = argv.get(3).and_then(|s| s.parse().ok()).unwrap_or(128);

    let hfq = HfqFile::open(Path::new(model)).expect("open model");
    let (info, raw_bytes) = hfq
        .tensor_data_cow("lm_head.weight")
        .or_else(|| hfq.tensor_data_cow("model.language_model.embed_tokens.weight"))
        .expect("no lm_head.weight or tied embed_tokens.weight");
    let (vocab, hidden) = (info.shape[0] as usize, info.shape[1] as usize);
    assert_eq!(
        hidden % GROUP,
        0,
        "hidden {hidden} must be a multiple of 256"
    );
    let bytes: std::borrow::Cow<[u8]> = if info.quant_type == 49 {
        std::borrow::Cow::Owned(
            hipfire_primitives::bf16_lut3::decode(&raw_bytes, vocab * hidden)
                .expect("Bf16Lut3 head payload is corrupt or truncated"),
        )
    } else {
        raw_bytes
    };
    assert_eq!(bytes.len(), vocab * hidden * 2, "expected a bf16 head");
    let head: Vec<f32> = bytes
        .chunks_exact(2)
        .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
        .collect();
    println!("head [{vocab}, {hidden}] from qt {}", info.quant_type);

    let raw = std::fs::read(fnorm_path).expect("read fnorm");
    let total = raw.len() / 4 / hidden;
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
        "queries: {} of {total} captured (stride {stride})",
        queries.len()
    );

    // Reference: exact bf16 logits, and the rotated form of each query that the
    // Opus arms consume.
    let signs1 = gen_fwht_signs(42, GROUP);
    let signs2 = gen_fwht_signs(1042, GROUP);
    let reference: Vec<(Vec<f32>, Vec<f64>, usize, Vec<f32>)> = queries
        .iter()
        .map(|q| {
            let logits: Vec<f32> = (0..vocab)
                .into_par_iter()
                .map(|v| {
                    let row = &head[v * hidden..(v + 1) * hidden];
                    row.iter().zip(q).map(|(&w, &x)| w * x).sum::<f32>()
                })
                .collect();
            let p = softmax(&logits);
            let top = argmax(&logits);
            let mut xr = q.clone();
            for g in 0..hidden / GROUP {
                signed_fwht(&mut xr[g * GROUP..(g + 1) * GROUP], &signs1, &signs2);
            }
            (logits, p, top, xr)
        })
        .collect();

    // n_out=0 (pure int4, 4.0625 b/w) is not reachable through the compact
    // encoder — it clamps n_out to >= 1 — so the bottom of the range is the
    // plain Oq4G256 codec, which is the same thing without an overlay table.
    let arms: Vec<(String, f32, u8, Box<dyn Fn() -> Vec<u8> + Sync>)> = vec![
        (
            "oq8      ".into(),
            8.0625,
            35,
            Box::new(|| codecs::quantize_oq8g256(&head, &signs1, &signs2)),
        ),
        (
            // `Oq4G256` bytes (130 B/group) expand through qt 33 (`OqPlusG256`),
            // not qt 34 — 33 is the code `oq8_arch_load` registers that decode
            // under. Same on-disk layout, so the encoder pairs with it directly.
            "oq4      ".into(),
            4.0625,
            33,
            Box::new(|| codecs::quantize_oq4g256(&head, &signs1, &signs2)),
        ),
        (
            "oq+c n1  ".into(),
            4.125,
            36,
            Box::new(|| {
                codecs::quantize_oqplus_compact_g(&head, &signs1, &signs2, 1.0 / 256.0, GROUP)
            }),
        ),
        (
            "oq+c n2  ".into(),
            4.1875,
            36,
            Box::new(|| {
                codecs::quantize_oqplus_compact_g(&head, &signs1, &signs2, 2.0 / 256.0, GROUP)
            }),
        ),
        (
            "oq+c n3  ".into(),
            4.25,
            36,
            Box::new(|| {
                codecs::quantize_oqplus_compact_g(&head, &signs1, &signs2, 3.0 / 256.0, GROUP)
            }),
        ),
        (
            "oq3      ".into(),
            3.0625,
            38,
            Box::new(|| codecs::quantize_oq3g256(&head, &signs1, &signs2)),
        ),
    ];

    println!(
        "\n{:<9} {:>6} {:>9} {:>12} {:>12} {:>9}",
        "format", "b/w", "head MB", "mean KLD", "max KLD", "top1"
    );
    for (name, bw, qt, encode) in &arms {
        let packed = encode();
        let head_mb = packed.len() as f64 / 1e6;
        let (combined, _) = oq8_arch_load(*qt, &packed, vocab, hidden)
            .unwrap_or_else(|| panic!("{name}: oq8_arch_load rejected qt {qt}"));
        let mut sum = 0f64;
        let mut max = 0f64;
        let mut agree = 0usize;
        for (_, p_ref, top_ref, x_rot) in &reference {
            let logits = opus_logits(&combined, vocab, hidden, x_rot);
            let d = kld(p_ref, &softmax(&logits));
            sum += d;
            max = max.max(d);
            if argmax(&logits) == *top_ref {
                agree += 1;
            }
        }
        let n = reference.len() as f64;
        println!(
            "{name} {bw:>6.4} {head_mb:>9.1} {:>12.6} {:>12.6} {:>8.1}%",
            sum / n,
            max,
            100.0 * agree as f64 / n
        );
    }
    println!("\nKLD is KL(exact bf16 head || quantized head) in nats, head-only.");
}
