// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! Which lm_head form decodes faster: F32 via GEMV, or BF16 via batch-1 WMMA?
//!
//! `weight_gemv` special-cases BF16 to `gemm_bf16_x_bf16_wmma(.., 1)` with the
//! note "dispatch family has no BF16 GEMV entry", while F32/F16 heads reach
//! dedicated GEMV paths. The tied-embedding loader therefore expands a BF16
//! head to F32 (`hfq.rs:2689`), doubling per-token head traffic to buy the
//! better kernel — see `docs/tied-lmhead-f32-expansion.md`.
//!
//! Whether that trade pays is a hardware question, not an opinion, and this
//! measures it at the real shape: llama3.2:1b's 128256 x 2048 head.
//!
//! Run: cargo run --release -p hipfire-runtime --example bench_lmhead_dtype

use hipfire_rdna::{DType, Gpu};
use std::time::Instant;

fn f32_to_bf16_bytes(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 2);
    for &x in v {
        let bits = x.to_bits();
        let r = ((bits >> 16) & 1).wrapping_add(0x7fff).wrapping_add(bits);
        out.extend_from_slice(&((r >> 16) as u16).to_le_bytes());
    }
    out
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<usize> = std::env::args()
        .skip(1)
        .filter_map(|x| x.parse().ok())
        .collect();
    let vocab = *a.first().unwrap_or(&128256);
    let hidden = *a.get(1).unwrap_or(&2048);
    let iters = *a.get(2).unwrap_or(&30);

    let mut gpu = Gpu::init()?;
    println!("lm_head decode: {vocab} x {hidden}, {iters} iters");
    println!(
        "  F32 weight {:.1} MB   BF16 weight {:.1} MB",
        (vocab * hidden * 4) as f64 / 1e6,
        (vocab * hidden * 2) as f64 / 1e6
    );

    // Deterministic weights; values are irrelevant to timing but NaN/Inf are not.
    let mut s = 0x51ead_u64;
    let w: Vec<f32> = (0..vocab * hidden)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as f32 / (1u64 << 31) as f32 - 1.0) * 0.02
        })
        .collect();
    let x: Vec<f32> = (0..hidden)
        .map(|i| ((i % 17) as f32 - 8.0) * 0.05)
        .collect();

    let w_f32 = gpu.upload_f32(&w, &[vocab, hidden])?;
    // upload_raw types the tensor as DType::Raw; the WMMA path asserts BF16.
    let mut w_bf16 = gpu.upload_raw(&f32_to_bf16_bytes(&w), &[vocab, hidden])?;
    w_bf16.dtype = DType::BF16;
    let x_f32 = gpu.upload_f32(&x, &[hidden])?;
    // Activations stay F32: gemm_bf16_x_bf16_wmma stages them to BF16 itself,
    // and asserts F32 on the way in — same as the real weight_gemv call site.
    let y = gpu.zeros(&[vocab], DType::F32)?;

    // Straight-line timing: no closures, so no GpuTensor clones.
    for _ in 0..3 {
        gpu.gemv_f32(&w_f32, &x_f32, &y)?;
    }
    gpu.hip.device_synchronize()?;
    let t = Instant::now();
    for _ in 0..iters {
        gpu.gemv_f32(&w_f32, &x_f32, &y)?;
    }
    gpu.hip.device_synchronize()?;
    let t_f32 = t.elapsed().as_secs_f64() / iters as f64;

    for _ in 0..3 {
        gpu.gemm_bf16_x_bf16_wmma(&w_bf16, &x_f32, &y, vocab, hidden, 1)?;
    }
    gpu.hip.device_synchronize()?;
    let t = Instant::now();
    for _ in 0..iters {
        gpu.gemm_bf16_x_bf16_wmma(&w_bf16, &x_f32, &y, vocab, hidden, 1)?;
    }
    gpu.hip.device_synchronize()?;
    let t_bf16 = t.elapsed().as_secs_f64() / iters as f64;

    // Third option: BF16 weights through a REAL gemv. The kernel and its
    // binding both exist (gemv_bf16_f32.hip, gemv.rs:4843); it is simply not in
    // the dispatch family run_auto consults, which is what forces the WMMA
    // special case above.
    for _ in 0..3 {
        gpu.gemv_bf16_f32(&w_bf16, &x_f32, &y, vocab, hidden)?;
    }
    gpu.hip.device_synchronize()?;
    let t = Instant::now();
    for _ in 0..iters {
        gpu.gemv_bf16_f32(&w_bf16, &x_f32, &y, vocab, hidden)?;
    }
    gpu.hip.device_synchronize()?;
    let t_bf16_gemv = t.elapsed().as_secs_f64() / iters as f64;

    for (label, per, bytes) in [
        ("BF16 gemv_bf16_f32", t_bf16_gemv, vocab * hidden * 2),
        ("F32 gemv_f32", t_f32, vocab * hidden * 4),
        ("BF16 gemm_wmma(batch=1)", t_bf16, vocab * hidden * 2),
    ] {
        println!(
            "  {:<26} {:8.3} ms   {:7.1} GB/s   {:6.1} tok/s if head-bound",
            label,
            per * 1e3,
            bytes as f64 / per / 1e9,
            1.0 / per
        );
    }

    println!();
    println!(
        "BF16 gemv vs F32 gemv: {:.2}x   BF16 gemv vs BF16 wmma: {:.2}x",
        t_f32 / t_bf16_gemv,
        t_bf16 / t_bf16_gemv
    );
    // Report the three-way winner, not a pairwise one: "F32 beats BF16" is true
    // only against the WMMA fallback the dispatch family currently forces, and
    // stating it alone would recommend keeping an expansion that a wired-in
    // BF16 gemv beats on both bytes and time.
    let best = [
        ("BF16 gemv_bf16_f32", t_bf16_gemv),
        ("F32 gemv_f32", t_f32),
        ("BF16 gemm_wmma", t_bf16),
    ]
    .into_iter()
    .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
    .unwrap();
    println!("fastest: {} at {:.3} ms", best.0, best.1 * 1e3);
    Ok(())
}
