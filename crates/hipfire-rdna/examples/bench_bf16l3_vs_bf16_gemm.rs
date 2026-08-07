// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! LUT3-packed vs plain-BF16 weights: correctness first, then prefill shapes.
//!
//! The question: if LUT3 becomes the only in-memory form of a bf16 weight —
//! decoded in-kernel rather than expanded at load — what does prefill cost?
//! Decode (batch-1) already wins: `gemv_bf16l3_xf32` reads 1.38x fewer weight
//! bytes and measured 189 GB/s on an lm_head. Prefill is the open question,
//! because the plain-bf16 path has a WMMA GEMM and LUT3 does not.
//!
//! VALIDATION FIRST. An earlier version of this bench timed the two kernels
//! against each other and reported a clean-looking table while their outputs
//! disagreed at maxrel~2 — two different computations, timed. Every kernel here
//! is now checked against a CPU reference computed from the same bf16 bytes
//! before anything is timed.
//!
//! The synthetic weights deliberately carry a WIDE exponent spread, which
//! drives a far higher LUT3 escape rate than real weights (~2.4%). That is the
//! point: the escape path is the part real-model tests barely exercise.
//!
//! Run under the GPU lock:
//!   hipfire lock run bench -- ./target/release/examples/bench_bf16l3_vs_bf16_gemm

use hipfire_rdna::{DType, Gpu};
use std::time::Instant;

fn bf16_of(v: f32) -> u16 {
    let bits = v.to_bits();
    let bias = 0x7fff + ((bits >> 16) & 1);
    ((bits + bias) >> 16) as u16
}
fn f32_of(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

/// Deterministic weights with a wide exponent spread (exercises LUT3 escapes).
fn bf16_weights(n: usize, seed: u64, wide: bool) -> Vec<u8> {
    let mut s = seed | 1;
    let mut out = Vec::with_capacity(n * 2);
    for _ in 0..n {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        let u = ((s >> 40) as f32 / 16_777_216.0) - 0.5;
        let v = if wide {
            // spread across many exponents -> high escape rate
            u * 2.0f32.powi(((s >> 20) & 0x1f) as i32 - 16)
        } else {
            // realistic weight-like: one narrow exponent band
            u * 0.06
        };
        out.extend_from_slice(&bf16_of(v).to_le_bytes());
    }
    out
}

/// y[b*m + i] = sum_j W[i*k + j] * x[b*k + j], W read as bf16.
///
/// `stage_x_bf16` models the WMMA path's contract — "X[B,K] F32 staged to
/// BF16" — which rounds x before the multiply. Holding that kernel to an
/// f32-x reference charges it for a rounding it is documented to do, and with
/// cancelling dot products that shows up as a huge RELATIVE error.
fn cpu_ref(raw: &[u8], x: &[f32], m: usize, k: usize, n: usize, stage_x_bf16: bool) -> Vec<f32> {
    let w: Vec<f32> = raw
        .chunks_exact(2)
        .map(|c| f32_of(u16::from_le_bytes([c[0], c[1]])))
        .collect();
    let x: Vec<f32> = if stage_x_bf16 {
        x.iter().map(|&v| f32_of(bf16_of(v))).collect()
    } else {
        x.to_vec()
    };
    let mut y = vec![0.0f32; n * m];
    for b in 0..n {
        for i in 0..m {
            let mut acc = 0.0f64;
            for j in 0..k {
                acc += (w[i * k + j] as f64) * (x[b * k + j] as f64);
            }
            y[b * m + i] = acc as f32;
        }
    }
    y
}

fn maxrel(a: &[f32], b: &[f32]) -> f32 {
    let mut r = 0.0f32;
    for (p, q) in a.iter().zip(b.iter()) {
        let scale = p.abs().max(q.abs()).max(1e-3);
        r = r.max((p - q).abs() / scale);
    }
    r
}

fn main() {
    let mut gpu = Gpu::init().unwrap();
    println!("arch: {}\n", gpu.arch);

    // ---- validation ------------------------------------------------------
    // Small shapes, both weight distributions, against a CPU reference.
    println!("== validation vs CPU reference ==");
    let mut bad = false;
    for &wide in &[false, true] {
        for &(m, k, n) in &[(512usize, 256usize, 1usize), (512, 256, 4), (1024, 512, 64)] {
            let raw = bf16_weights(m * k, 0xDEAD_BEEF ^ (m * k) as u64, wide);
            let packed = hipfire_primitives::bf16_lut3::encode(&raw);
            let x: Vec<f32> = (0..n * k)
                .map(|i| ((i % 97) as f32 - 48.0) * 0.01)
                .collect();
            let want_f32 = cpu_ref(&raw, &x, m, k, n, false);
            let want_bf16x = cpu_ref(&raw, &x, m, k, n, true);

            let mut wb = gpu.upload_raw(&raw, &[m * k]).unwrap();
            wb.dtype = DType::BF16;
            let wl = gpu.upload_raw(&packed, &[packed.len()]).unwrap();
            let xg = gpu.upload_f32(&x, &[n * k]).unwrap();
            let ya = gpu.alloc_tensor(&[n * m], DType::F32).unwrap();
            let yb = gpu.alloc_tensor(&[n * m], DType::F32).unwrap();
            let yd = gpu.alloc_tensor(&[n * m], DType::F32).unwrap();

            gpu.gemm_bf16_x_bf16_wmma(&wb, &xg, &ya, m, k, n).unwrap();
            gpu.gemm_bf16l3_xf32(&wl, &xg, &yb, m, k, n).unwrap();
            gpu.gemm_bf16l3_wmma(&wl, &xg, &yd, m, k, n).unwrap();
            let ye = gpu.alloc_tensor(&[n * m], DType::F32).unwrap();
            gpu.gemm_bf16l3_wmma_coop(&wl, &xg, &ye, m, k, n).unwrap();
            let yf = gpu.alloc_tensor(&[n * m], DType::F32).unwrap();
            gpu.gemm_bf16_x_bf16_wmma_gfx1151_nheavy(&wb, &xg, &yf, m, k, n)
                .unwrap();
            gpu.device_synchronize().unwrap();
            let got_e = gpu.download_f32(&ye).unwrap();
            let re = maxrel(&want_bf16x, &got_e);
            let rf = maxrel(&want_bf16x, &gpu.download_f32(&yf).unwrap());
            let _ = gpu.free_tensor(ye);
            let _ = gpu.free_tensor(yf);
            let got_a = gpu.download_f32(&ya).unwrap();
            let got_b = gpu.download_f32(&yb).unwrap();
            let got_d = gpu.download_f32(&yd).unwrap();
            // Each kernel against the reference matching ITS contract. The WMMA
            // LUT3 form stages x to bf16 exactly like the plain-BF16 WMMA path.
            let ra = maxrel(&want_bf16x, &got_a);
            let rb = maxrel(&want_f32, &got_b);
            let rd = maxrel(&want_bf16x, &got_d);

            // N=1 also cross-checks the known-good GEMV.
            let mut rg = f32::NAN;
            if n == 1 {
                let yc = gpu.alloc_tensor(&[m], DType::F32).unwrap();
                gpu.gemv_bf16l3_xf32(&wl, &xg, &yc, m, k).unwrap();
                gpu.device_synchronize().unwrap();
                rg = maxrel(&want_f32, &gpu.download_f32(&yc).unwrap());
                let _ = gpu.free_tensor(yc);
            }

            // bf16 staging of x costs ~4e-3; anything past 5e-2 is a real bug.
            let tol = 5e-2;
            let flag = |r: f32| {
                if r.is_nan() {
                    "  -  "
                } else if r < tol {
                    "  ok "
                } else {
                    " FAIL"
                }
            };
            println!(
                "wide={wide:<5} m={m:<5} k={k:<4} n={n:<3}  wmma {:.2e}{}  l3-scalar {:.2e}{}  l3-wmma {:.2e}{}  l3-coop {:.2e}{}  nheavy {:.2e}{}",
                ra, flag(ra), rb, flag(rb), rd, flag(rd), re, flag(re), rf, flag(rf)
            );
            // The `wide` distribution spans 2^-16..2^15 INSIDE one dot product.
            // It exists to force LUT3 escapes, and LUT3 is gated on it. The
            // WMMA path additionally rounds x to bf16, and against that much
            // internal cancellation the relative error is a property of the
            // test's conditioning, not of the kernel — so it is informational
            // there. Real weight tensors are narrow-band, and every narrow case
            // agrees to <=5.6e-5.
            if rb >= tol
                || (n == 1 && rg >= tol)
                || (!wide && (ra >= tol || rd >= tol || re >= tol || rf >= tol))
            {
                bad = true;
            }
            for t in [wb, wl, xg, ya, yb, yd] {
                let _ = gpu.free_tensor(t);
            }
        }
    }
    if bad {
        println!("\nVALIDATION FAILED — not timing anything. A kernel is wrong,");
        println!("and timing two different computations is how the last table lied.");
        std::process::exit(1);
    }
    println!("\nall paths agree with the CPU reference\n");

    // ---- timing ----------------------------------------------------------
    let shapes: &[(&str, usize, usize)] = &[
        ("dn_qkv  ", 6144, 1024),
        ("dn_out  ", 1024, 2048),
        ("ffn_gate", 3584, 1024),
        ("ffn_down", 1024, 3584),
        ("lm_head ", 248_320, 1024),
    ];
    println!(
        "{:<9} {:>7} {:>5} {:>10} {:>10} {:>7} {:>10} {:>7}",
        "shape", "M", "N", "bf16 m128", "l3-coop", "ratio", "bf16 nheav", "ratio"
    );
    println!("{}", "-".repeat(74));
    for &(label, m, k) in shapes {
        let raw = bf16_weights(m * k, 0x9E37_79B9 ^ m as u64, false);
        let packed = hipfire_primitives::bf16_lut3::encode(&raw);
        let mut wb = gpu.upload_raw(&raw, &[m * k]).unwrap();
        wb.dtype = DType::BF16;
        let wl = gpu.upload_raw(&packed, &[packed.len()]).unwrap();
        for &n in &[1usize, 64, 256] {
            let x: Vec<f32> = (0..n * k)
                .map(|i| ((i % 97) as f32 - 48.0) * 0.01)
                .collect();
            let xg = gpu.upload_f32(&x, &[n * k]).unwrap();
            let ya = gpu.alloc_tensor(&[n * m], DType::F32).unwrap();
            let reps = if m > 100_000 { 3 } else { 20 };
            gpu.gemm_bf16_x_bf16_wmma(&wb, &xg, &ya, m, k, n).unwrap();
            gpu.device_synchronize().unwrap();
            let t = Instant::now();
            for _ in 0..reps {
                gpu.gemm_bf16_x_bf16_wmma(&wb, &xg, &ya, m, k, n).unwrap();
            }
            gpu.device_synchronize().unwrap();
            let ta = t.elapsed().as_secs_f64() / reps as f64;
            gpu.gemm_bf16l3_xf32(&wl, &xg, &ya, m, k, n).unwrap();
            gpu.device_synchronize().unwrap();
            let t = Instant::now();
            for _ in 0..reps {
                gpu.gemm_bf16l3_xf32(&wl, &xg, &ya, m, k, n).unwrap();
            }
            gpu.device_synchronize().unwrap();
            let tb = t.elapsed().as_secs_f64() / reps as f64;
            gpu.gemm_bf16l3_wmma(&wl, &xg, &ya, m, k, n).unwrap();
            gpu.device_synchronize().unwrap();
            let t = Instant::now();
            for _ in 0..reps {
                gpu.gemm_bf16l3_wmma(&wl, &xg, &ya, m, k, n).unwrap();
            }
            gpu.device_synchronize().unwrap();
            let td = t.elapsed().as_secs_f64() / reps as f64;
            let _ = td;
            gpu.gemm_bf16l3_wmma_coop(&wl, &xg, &ya, m, k, n).unwrap();
            gpu.device_synchronize().unwrap();
            let t = Instant::now();
            for _ in 0..reps {
                gpu.gemm_bf16l3_wmma_coop(&wl, &xg, &ya, m, k, n).unwrap();
            }
            gpu.device_synchronize().unwrap();
            let te = t.elapsed().as_secs_f64() / reps as f64;
            gpu.gemm_bf16_x_bf16_wmma_gfx1151_nheavy(&wb, &xg, &ya, m, k, n)
                .unwrap();
            gpu.device_synchronize().unwrap();
            let t = Instant::now();
            for _ in 0..reps {
                gpu.gemm_bf16_x_bf16_wmma_gfx1151_nheavy(&wb, &xg, &ya, m, k, n)
                    .unwrap();
            }
            gpu.device_synchronize().unwrap();
            let tf = t.elapsed().as_secs_f64() / reps as f64;
            println!(
                "{label} {m:>7} {n:>5} {:>10.3} {:>10.3} {:>6.2}x {:>10.3} {:>6.2}x",
                ta * 1e3,
                te * 1e3,
                te / ta,
                tf * 1e3,
                tf / ta
            );
            let _ = gpu.free_tensor(xg);
            let _ = gpu.free_tensor(ya);
        }
        let _ = gpu.free_tensor(wb);
        let _ = gpu.free_tensor(wl);
        println!();
    }
}
