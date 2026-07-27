// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
#![allow(
    clippy::needless_range_loop,
    clippy::manual_div_ceil,
    clippy::too_many_arguments,
    clippy::unnecessary_cast
)]

//! Verify the generic two-stage lm_head decode against a full bf16 GEMV.
//!
//! Usage: `verify_lmhead_twostage <bf16_hfq_path> [topk] [n_probes]`
//!   topk    default 32
//!   n_probes default 64
//!
//! For `n_probes` deterministic (xorshift-seeded) f32 hidden vectors it computes
//! (a) full logits via `gemv_bf16_f32` over ALL vocab rows → `argmax_full`, and
//! (b) the two-stage argmax via `lmhead_twostage_serve_bf16` → `argmax_two`, and
//! reports `recall@1 = matches / n_probes`. `recall@1 == 1.0` means the coarse
//! shortlist is greedy-exact (lossless) at this `topk`.
//!
//! TODO(real-weight): this v1 SYNTHESIZES a deterministic random bf16
//! `[vocab, hidden]` lm_head weight instead of loading `lm_head.weight` (or the
//! tied `model.embed_tokens.weight`) from the `.hfq` at `<bf16_hfq_path>`. The
//! real-tensor path needs an HFQ single-tensor reader; `HfqFile` lives in
//! `hipfire-runtime`, which depends on `hipfire-rdna`, so wiring it in as a
//! dev-dependency here risks a build cycle. The synthetic path still exercises
//! the full kernel chain (coarse Q4 GEMV → device top-K → bf16 gather) end to
//! end; on-GPU numerical correctness against a real model is verified separately
//! by the caller. The path argument is accepted but currently only reported.

use hipfire_rdna::lmhead_twostage::{build_lmhead_coarse_bf16, lmhead_twostage_serve_bf16};
use hipfire_rdna::{DType, Gpu};

/// SplitMix64 step — deterministic, no external RNG crate.
fn next_u64(s: &mut u64) -> u64 {
    *s = s.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *s;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// Uniform f32 in [-1, 1).
fn next_f32(s: &mut u64) -> f32 {
    let u = (next_u64(s) >> 40) as f32 / (1u64 << 24) as f32; // [0,1)
    u * 2.0 - 1.0
}

/// f32 → bf16 bits (truncate the low 16 mantissa bits).
fn f32_to_bf16_bits(f: f32) -> u16 {
    (f.to_bits() >> 16) as u16
}

fn argmax(v: &[f32]) -> (usize, f32) {
    let mut bi = 0usize;
    let mut bv = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > bv {
            bv = x;
            bi = i;
        }
    }
    (bi, bv)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).cloned().unwrap_or_else(|| "<none>".to_string());
    let topk: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(32);
    let n_probes: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(64);
    let bits = 4usize;

    // Synthetic geometry: big enough vocab to make the shortlist meaningful,
    // small enough to run instantly. See the TODO above for the real path.
    let vocab = 8192usize;
    let hidden = 512usize;

    println!(
        "verify_lmhead_twostage: path={path} (SYNTHETIC weight), vocab={vocab} hidden={hidden} \
         bits={bits} topk={topk} n_probes={n_probes}"
    );

    let mut gpu = Gpu::init().expect("GPU init failed");

    // Deterministic random bf16 lm_head weight [vocab, hidden], packed as raw
    // little-endian bf16 bytes (the kernels read the buffer as bf16 regardless
    // of the tensor's declared dtype).
    let mut seed: u64 = 0xD1B54A32D192ED03;
    let mut wbytes = vec![0u8; vocab * hidden * 2];
    for i in 0..vocab * hidden {
        let b = f32_to_bf16_bits(next_f32(&mut seed)).to_le_bytes();
        wbytes[2 * i] = b[0];
        wbytes[2 * i + 1] = b[1];
    }
    let lmhead_bf16 = gpu
        .upload_raw(&wbytes, &[vocab, hidden])
        .expect("upload lm_head bf16");

    // Build the coarse tier once.
    let coarse = build_lmhead_coarse_bf16(&mut gpu, &lmhead_bf16, vocab, hidden, bits)
        .expect("build coarse tier");
    println!("built coarse tier: kdim={} bits={}", coarse.kdim, coarse.bits);

    let logits_full = gpu.zeros(&[vocab], DType::F32).expect("alloc logits_full");
    let logits_two = gpu.zeros(&[vocab], DType::F32).expect("alloc logits_two");

    let mut matches = 0usize;
    let mut pseed: u64 = 0x243F6A8885A308D3;
    for p in 0..n_probes {
        // Random f32 hidden vector, uploaded fresh each probe.
        let h: Vec<f32> = (0..hidden).map(|_| next_f32(&mut pseed)).collect();
        let fnorm = gpu.upload_f32(&h, &[hidden]).expect("upload hidden");

        // (a) full bf16 logits over ALL rows → argmax_full.
        gpu.gemv_bf16_f32(&lmhead_bf16, &fnorm, &logits_full, vocab, hidden)
            .expect("full gemv");
        let full = gpu.download_f32(&logits_full).expect("download full");
        let (amax_full, lf) = argmax(&full);

        // (b) two-stage → argmax_two.
        lmhead_twostage_serve_bf16(
            &mut gpu,
            &lmhead_bf16,
            &coarse,
            &fnorm,
            &logits_two,
            vocab,
            hidden,
            topk,
        )
        .expect("two-stage serve");
        let two = gpu.download_f32(&logits_two).expect("download two");
        let (amax_two, lt) = argmax(&two);

        if amax_full == amax_two {
            matches += 1;
        } else {
            println!(
                "  MISS probe {p}: argmax_full={amax_full} (logit {lf:.5}) vs \
                 argmax_two={amax_two} (logit {lt:.5}), full-logit@two={:.5} gap={:.5}",
                full[amax_two],
                lf - full[amax_two]
            );
        }
        let _ = gpu.free_tensor(fnorm);
    }

    let recall = matches as f64 / n_probes as f64;
    println!("recall@1 = {matches}/{n_probes} = {recall:.4}");
    if (recall - 1.0).abs() < f64::EPSILON {
        println!("PASS: two-stage is greedy-exact (lossless) at topk={topk}");
    } else {
        println!("NOTE: recall@1 < 1.0 — coarse shortlist missed the true argmax on some probes (raise topk).");
    }
}
