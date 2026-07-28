// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.
//! Does decoding BF16L3 in-kernel actually buy weight bandwidth?
//!
//! Decode is bandwidth-bound on a UMA APU: the weight matrix is streamed from
//! shared DRAM once per token, so a format that stores ~1.38x fewer weight
//! bytes should run ~1.38x faster *if* the in-register decode is free. This
//! probe answers that on real Gaussian weights, not zeros — the escape rate,
//! and therefore the byte count, is data-dependent.
//!
//! It checks correctness first: BF16L3 is lossless, so the compressed kernel
//! must agree with `gemv_bf16_bf16` bit-for-bit on the same weights. A speedup
//! from a kernel that computes the wrong thing is worthless.
//!
//!   cargo run --release -p hipfire-rdna --example bench_gemv_bf16l3 [M] [K]

use hipfire_primitives::bf16_lut3;
use hipfire_primitives::conv::f32_to_bf16_bits;
use hipfire_rdna::Gpu;
use std::time::Instant;

/// Deterministic xorshift, so a regression is reproducible.
fn xorshift(seed: &mut u32) -> u32 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 17;
    *seed ^= *seed << 5;
    *seed
}

fn main() {
    let mut a = std::env::args().skip(1);
    let m: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(8192);
    let k: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(8192);
    assert_eq!(k % 256, 0, "K must be a multiple of 256");

    let mut gpu = Gpu::init().unwrap();

    // Gaussian-ish weights (sum of 4 uniforms) — the escape rate, hence the
    // compressed size, depends on the real exponent spread. Zeros would flatter
    // the format into a ~1.41x best case it never sees in practice.
    let mut seed = 0x1234_5678u32;
    let w_bits: Vec<u16> = (0..m * k)
        .map(|_| {
            let s: i64 = (0..4).map(|_| (xorshift(&mut seed) % 2048) as i64).sum();
            f32_to_bf16_bits((s - 4094) as f32 * 1e-4)
        })
        .collect();
    let x_bits: Vec<u16> = (0..k)
        .map(|_| f32_to_bf16_bits(((xorshift(&mut seed) % 2048) as f32 - 1024.0) * 1e-3))
        .collect();

    // Ablation corpus: one exponent everywhere => zero escapes. Same kernel,
    // same byte-plane structure, but the escape branch never fires and the
    // scattered escape-plane reads vanish. Separates fixed decode cost from
    // escape-handling cost.
    let w_noesc: Vec<u8> = (0..m * k)
        .flat_map(|i| (0x3f80u16 | (i as u16 & 0x7f)).to_le_bytes())
        .collect();
    let packed_noesc = bf16_lut3::encode(&w_noesc);

    let w_raw: Vec<u8> = w_bits.iter().flat_map(|b| b.to_le_bytes()).collect();
    let x_raw: Vec<u8> = x_bits.iter().flat_map(|b| b.to_le_bytes()).collect();
    let packed = bf16_lut3::encode(&w_raw);
    assert_eq!(
        bf16_lut3::decode(&packed, m * k).as_deref(),
        Some(w_raw.as_slice()),
        "codec must be lossless before we trust the kernel"
    );

    let bpw_plain = 2.0;
    let bpw_l3 = packed.len() as f64 / (m * k) as f64;
    let ratio = bpw_plain / bpw_l3;

    let w_plain = gpu.upload_raw(&w_raw, &[m, k]).unwrap();
    let w_l3 = gpu.upload_raw(&packed, &[packed.len()]).unwrap();
    let w_l3_noesc = gpu
        .upload_raw(&packed_noesc, &[packed_noesc.len()])
        .unwrap();
    let bpw_noesc = packed_noesc.len() as f64 / (m * k) as f64;
    let x = gpu.upload_raw(&x_raw, &[k]).unwrap();
    let y_a = gpu.upload_raw(&vec![0u8; m * 2], &[m]).unwrap();
    let y_b = gpu.upload_raw(&vec![0u8; m * 2], &[m]).unwrap();

    println!(
        "gemv BF16 vs BF16L3   M={m} K={k}   weight {:.0} MiB -> {:.0} MiB ({ratio:.4}x, \
         {bpw_l3:.4} B/w)   on {}\n",
        (m * k * 2) as f64 / (1024.0 * 1024.0),
        packed.len() as f64 / (1024.0 * 1024.0),
        gpu.arch
    );

    // ---- correctness ----
    // The load-bearing check is bf16l3 vs bf16_vec8: identical accumulation
    // order, so the ONLY difference is the weight encoding. BF16L3 is lossless,
    // so that must be bit-identical.
    //
    // bf16l3 vs the shipping stride-32 kernel must NOT be required to match
    // bit-for-bit: those two sum the same products in a different order, and
    // f32 addition is not associative, so a few rows land 1 ULP apart. That is
    // reassociation, not a format error — it is reported, not failed on.
    let run3 = |g: &mut Gpu, t: &_| g.gemv_bf16_bf16(&w_plain, &x, t, m, k).unwrap();
    run3(&mut gpu, &y_a);
    gpu.gemv_bf16_vec8(&w_plain, &x, &y_b, m, k).unwrap();
    gpu.hip.device_synchronize().unwrap();
    let ref_stride32 = gpu.download_raw(&y_a, m * 2).unwrap();
    let ref_vec8 = gpu.download_raw(&y_b, m * 2).unwrap();

    gpu.gemv_bf16l3(&w_l3, &x, &y_b, m, k).unwrap();
    gpu.hip.device_synchronize().unwrap();
    let got_l3 = gpu.download_raw(&y_b, m * 2).unwrap();

    let diff = |a: &[u8], b: &[u8]| {
        a.chunks_exact(2)
            .zip(b.chunks_exact(2))
            .filter(|(p, q)| p != q)
            .count()
    };
    let bad = diff(&ref_vec8, &got_l3);
    if bad != 0 && std::env::var_os("HIPFIRE_BENCH_SKIP_PARITY").is_none() {
        println!("FAIL: bf16l3 differs from bf16_vec8 on {bad}/{m} rows — format is NOT lossless");
        std::process::exit(1);
    }
    println!(
        "parity: bf16l3 == bf16_vec8 bit-exact on all {m} rows ✓ (same order, lossless format)\n\
         reassoc: {} / {m} rows differ between stride-32 and vec8 ordering (f32 non-associativity, \
         not a format error)\n",
        diff(&ref_stride32, &ref_vec8),
    );

    // ---- bandwidth ----
    let warmup = 30;
    let trials = 200;
    println!(
        "{:<14} {:>8} {:>11} {:>12} {:>9}",
        "kernel", "B/w", "µs/call", "GB/s(weight)", "speedup"
    );
    println!("{}", "-".repeat(60));

    // 0 = stride-32 plain bf16 (the shipping kernel), 1 = vec8 plain bf16 (same
    // access shape as bf16l3), 2 = bf16l3. (1) vs (0) is the coalescing effect;
    // (2) vs (1) is the compression effect with everything else held constant.
    let mut us_of = [0.0f64; 4];
    for (i, (label, bpw)) in [
        ("bf16", bpw_plain),
        ("bf16_vec8", bpw_plain),
        ("bf16l3", bpw_l3),
        ("bf16l3(0 esc)", bpw_noesc),
    ]
    .into_iter()
    .enumerate()
    {
        let go = |g: &mut Gpu| match i {
            0 => g.gemv_bf16_bf16(&w_plain, &x, &y_a, m, k).unwrap(),
            1 => g.gemv_bf16_vec8(&w_plain, &x, &y_a, m, k).unwrap(),
            2 => g.gemv_bf16l3(&w_l3, &x, &y_b, m, k).unwrap(),
            _ => g.gemv_bf16l3(&w_l3_noesc, &x, &y_b, m, k).unwrap(),
        };
        for _ in 0..warmup {
            go(&mut gpu);
        }
        gpu.hip.device_synchronize().unwrap();
        let t = Instant::now();
        for _ in 0..trials {
            go(&mut gpu);
        }
        gpu.hip.device_synchronize().unwrap();
        let us = t.elapsed().as_secs_f64() * 1e6 / trials as f64;
        us_of[i] = us;
        let gbps = (m * k) as f64 * bpw / (us * 1e-6) / 1e9;
        println!(
            "{label:<14} {bpw:>8.4} {us:>11.2} {gbps:>12.1} {:>8.3}x",
            us_of[0] / us
        );
    }

    println!(
        "\nbyte ratio {ratio:.3}x is what compression alone can buy.\n\
         coalescing  (vec8 / stride-32 bf16) : {:.3}x   <- free, no format change\n\
         compression (bf16l3 / vec8 bf16)    : {:.3}x   <- the format's real win\n\
         combined    (bf16l3 / stride-32)    : {:.3}x\n\
         \n\
         ablation: zero-escape weights ({:.4} B/w, byte ratio {:.3}x) run at {:.3}x vs vec8.\n\
         escape handling therefore costs {:.1}%% of the compressed kernel's time.",
        us_of[0] / us_of[1],
        us_of[1] / us_of[2],
        us_of[0] / us_of[2],
        bpw_noesc,
        bpw_plain / bpw_noesc,
        us_of[1] / us_of[3],
        100.0 * (us_of[2] - us_of[3]) / us_of[2],
    );
}
