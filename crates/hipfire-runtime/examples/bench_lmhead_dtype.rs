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

    // The runtime's F32 head is a BF16 weight UPCONVERTED, so the honest
    // reference rounds w to bf16 first and then widens. Comparing against the
    // original f32 w would charge the bf16 path for weight rounding that is not
    // an error — the stored weight IS bf16, so that rounding is ground truth.
    let w_rt: Vec<f32> = f32_to_bf16_bytes(&w)
        .chunks_exact(2)
        .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
        .collect();
    let w_f32 = gpu.upload_f32(&w_rt, &[vocab, hidden])?;
    // upload_raw types the tensor as DType::Raw; the WMMA path asserts BF16.
    let mut w_bf16 = gpu.upload_raw(&f32_to_bf16_bytes(&w), &[vocab, hidden])?;
    w_bf16.dtype = DType::BF16;
    let x_f32 = gpu.upload_f32(&x, &[hidden])?;
    // Two different activation contracts, and mixing them up yields garbage
    // rather than an error: gemm_bf16_x_bf16_wmma asserts F32 in and stages to
    // BF16 itself, while gemv_bf16_f32 wants BF16 already (its x is
    // `const unsigned short*`). Feeding it the F32 buffer reads float bytes as
    // bf16 pairs — worst |diff| 5.8e8 against a magnitude of 0.284.
    let mut x_bf16 = gpu.upload_raw(&f32_to_bf16_bytes(&x), &[hidden])?;
    x_bf16.dtype = DType::BF16;
    let y = gpu.zeros(&[vocab], DType::F32)?;

    // BF16L3: lossless 3-bit-LUT exponent coding, decoded in-kernel so the
    // ratio applies to bandwidth. The question this answers is only "is the
    // packed format faster at this shape" — `gemv_bf16l3` takes a BF16
    // activation and emits BF16 logits, so it is NOT comparable on quality with
    // the xf32 paths above, and an lm_head use would need a `gemv_bf16l3_xf32`
    // that does not exist yet. Measure before writing it.
    let bf16_bytes = f32_to_bf16_bytes(&w);
    let packed = hipfire_primitives::bf16_lut3::encode(&bf16_bytes);
    let mut w_l3 = gpu.upload_raw(&packed, &[packed.len()])?;
    w_l3.dtype = DType::Raw;
    let y_bf16 = gpu.alloc_tensor(&[vocab], DType::BF16)?;
    println!(
        "  BF16L3 packed {:.1} MB ({:.3}x smaller than bf16)",
        packed.len() as f64 / 1e6,
        bf16_bytes.len() as f64 / packed.len() as f64
    );

    // Correctness BEFORE speed: a faster kernel that disagrees is worthless,
    // and the whole recommendation here is "switch the lm_head to this path".
    // bf16 has ~8 mantissa bits, so exact equality is not the bar; what matters
    // is that the ARGMAX (the decoded token) and the top of the distribution
    // agree with the f32 reference.
    gpu.gemv_f32(&w_f32, &x_f32, &y)?;
    let ref_y = gpu.download_f32(&y)?;
    gpu.gemv_bf16_f32(&w_bf16, &x_bf16, &y, vocab, hidden)?;
    let bf_y = gpu.download_f32(&y)?;
    gpu.gemv_bf16_xf32(&w_bf16, &x_f32, &y, vocab, hidden)?;
    let mx_y = gpu.download_f32(&y)?;
    gpu.gemv_bf16l3_xf32(&w_l3, &x_f32, &y, vocab, hidden)?;
    let l3_y = gpu.download_f32(&y)?;
    gpu.gemm_bf16_x_bf16_wmma(&w_bf16, &x_f32, &y, vocab, hidden, 1)?;
    let wm_y = gpu.download_f32(&y)?;

    let argmax = |v: &[f32]| {
        v.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap()
    };
    let mag = ref_y.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let a_ref = argmax(&ref_y);
    for (label, v) in [
        ("gemv_bf16l3_xf32 (x=f32)", &l3_y),
        ("gemv_bf16_xf32 (x=f32)", &mx_y),
        ("gemv_bf16_f32 (x=bf16)", &bf_y),
        ("gemm_wmma (x staged)", &wm_y),
    ] {
        let worst = ref_y
            .iter()
            .zip(v.iter())
            .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
        let am = argmax(v);
        println!(
            "  {:<22} argmax {} vs f32 {} {}  worst |diff| {:.3e} (rel {:.2e}, mag {:.3})",
            label,
            am,
            a_ref,
            if am == a_ref { "OK" } else { "MISMATCH" },
            worst,
            worst / mag.max(1e-9),
            mag
        );
    }
    // Isolate ACTIVATION rounding: same bf16 weight both sides, f32 x vs bf16 x.
    // This is the quality question — does staging x to bf16 cost anything the
    // F32 path keeps?
    let n_diff = ref_y
        .iter()
        .zip(bf_y.iter())
        .filter(|(a, b)| (*a - *b).abs() > 0.0)
        .count();
    let rms = (ref_y
        .iter()
        .zip(bf_y.iter())
        .map(|(a, b)| ((a - b) as f32).powi(2) as f64)
        .sum::<f64>()
        / ref_y.len() as f64)
        .sqrt();
    println!(
        "  activation rounding (bf16 W both sides): {} / {} logits differ, rms {:.3e}, mag {:.3}",
        n_diff,
        ref_y.len(),
        rms,
        mag
    );
    println!();

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
        gpu.gemv_bf16_xf32(&w_bf16, &x_f32, &y, vocab, hidden)?;
    }
    gpu.hip.device_synchronize()?;
    let t = Instant::now();
    for _ in 0..iters {
        gpu.gemv_bf16_xf32(&w_bf16, &x_f32, &y, vocab, hidden)?;
    }
    gpu.hip.device_synchronize()?;
    let t_mixed = t.elapsed().as_secs_f64() / iters as f64;

    for _ in 0..3 {
        gpu.gemv_bf16_f32(&w_bf16, &x_bf16, &y, vocab, hidden)?;
    }
    gpu.hip.device_synchronize()?;
    let t = Instant::now();
    for _ in 0..iters {
        gpu.gemv_bf16_f32(&w_bf16, &x_bf16, &y, vocab, hidden)?;
    }
    gpu.hip.device_synchronize()?;
    let t_bf16_gemv = t.elapsed().as_secs_f64() / iters as f64;

    for _ in 0..3 {
        gpu.gemv_bf16l3_xf32(&w_l3, &x_f32, &y, vocab, hidden)?;
    }
    gpu.hip.device_synchronize()?;
    let t = Instant::now();
    for _ in 0..iters {
        gpu.gemv_bf16l3_xf32(&w_l3, &x_f32, &y, vocab, hidden)?;
    }
    gpu.hip.device_synchronize()?;
    let t_l3x = t.elapsed().as_secs_f64() / iters as f64;

    for _ in 0..3 {
        gpu.gemv_bf16l3(&w_l3, &x_bf16, &y_bf16, vocab, hidden)?;
    }
    gpu.hip.device_synchronize()?;
    let t = Instant::now();
    for _ in 0..iters {
        gpu.gemv_bf16l3(&w_l3, &x_bf16, &y_bf16, vocab, hidden)?;
    }
    gpu.hip.device_synchronize()?;
    let t_l3 = t.elapsed().as_secs_f64() / iters as f64;

    for (label, per, bytes) in [
        ("BF16L3 gemv_bf16l3_xf32", t_l3x, packed.len()),
        ("BF16L3 gemv_bf16l3", t_l3, packed.len()),
        ("BF16 gemv_bf16_xf32", t_mixed, vocab * hidden * 2),
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
        ("BF16L3 gemv_bf16l3_xf32", t_l3x),
        ("BF16 gemv_bf16_xf32", t_mixed),
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
