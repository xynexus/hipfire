// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.
//! `gemm_oq_compact_iu4x2_w64_fold` (overlay added at the group fold from LDS)
//! against `gemm_oq_compact_iu4x2_w64` + the separate n_ov=3 pass, from ONE raw
//! int8 activation: the fold reads it de-interleaved from the GEMM's own staging,
//! the pass reads its K-major transpose. The fold sums the overlay in i32 before
//! the scale where the pass rounds its own f32 sum, so the gate is error relative
//! to the output scale, not bits. Shapes cover both prefill tiles (default below
//! 384 rows, m128 from 384), partial M and B blocks and the 27B's K; then the
//! 27B projection shapes at B=512 are timed both ways.
//!
//!   cargo run --release -p hipfire-rdna --example parity_oq_overlay_fold
use hipfire_rdna::{DType, Gpu, OQ_OVERLAY_SLACK};
use std::time::Instant;

struct Case {
    m: usize,
    k: usize,
    b: usize,
    w: hipfire_rdna::GpuTensor,
    x: hipfire_rdna::GpuTensor,
    xs: hipfire_rdna::GpuTensor,
    xt: hipfire_rdna::GpuTensor,
    xst: hipfire_rdna::GpuTensor,
}

fn build(gpu: &mut Gpu, rnd: &mut impl FnMut() -> u32, m: usize, k: usize, b: usize) -> Case {
    let ng = k / 256;
    let mut w = vec![0u8; m * ng * 136];
    for v in w[..m * ng * 128].iter_mut() {
        *v = rnd() as u8;
    }
    for i in 0..m * ng {
        let o = m * ng * 128 + i * 8;
        let bits = (((12 + rnd() % 3) as u16) << 10) | (rnd() % 1024) as u16;
        w[o..o + 2].copy_from_slice(&bits.to_le_bytes());
        for e in 0..3 {
            w[o + 2 + 2 * e] = rnd() as u8;
            w[o + 3 + 2 * e] = rnd() as u8;
        }
    }
    let x8: Vec<u8> = (0..b * k).map(|_| rnd() as u8).collect();
    // Fragment-interleaved, as the w64 GEMM consumes it: per 16-K step, 8 bytes
    // of hi digits then 8 of lo (see parity_gemm_oq_compact_iu4x2_w64).
    let mut xi = vec![0u8; b * k];
    for c in 0..b {
        for st in 0..k / 16 {
            for j in 0..8 {
                let (e, o) = (x8[c * k + st * 16 + 2 * j], x8[c * k + st * 16 + 2 * j + 1]);
                xi[c * k + st * 16 + j] = (e >> 4) | (o & 0xf0);
                xi[c * k + st * 16 + 8 + j] = (e & 0xf) | ((o & 0xf) << 4);
            }
        }
    }
    let xs: Vec<f32> = (0..b * ng)
        .map(|_| 1e-4 + (rnd() % 1000) as f32 * 1e-5)
        .collect();
    let mut xt = vec![0u8; k * b + OQ_OVERLAY_SLACK];
    let mut xst = vec![0f32; ng * b + OQ_OVERLAY_SLACK];
    for c in 0..b {
        for kk in 0..k {
            xt[kk * b + c] = x8[c * k + kk];
        }
        for g in 0..ng {
            xst[g * b + c] = xs[c * ng + g];
        }
    }
    Case {
        m,
        k,
        b,
        w: gpu.upload_raw(&w, &[w.len()]).unwrap(),
        x: gpu.upload_raw(&xi, &[xi.len()]).unwrap(),
        xs: gpu.upload_f32(&xs, &[xs.len()]).unwrap(),
        xt: gpu.upload_raw(&xt, &[xt.len()]).unwrap(),
        xst: gpu.upload_f32(&xst, &[xst.len()]).unwrap(),
    }
}

fn separate(gpu: &mut Gpu, c: &Case, y: &hipfire_rdna::GpuTensor) {
    gpu.gemm_oq_compact_iu4x2_w64(&c.w, &c.x, &c.xs, y, c.m, c.k, c.b, 136)
        .unwrap();
    gpu.oq_compact_overlay_correct_t(&c.w, &c.xt, &c.xst, y, c.m, c.k, c.b, 256, 136)
        .unwrap();
}

fn fold(gpu: &mut Gpu, c: &Case, y: &hipfire_rdna::GpuTensor) {
    gpu.gemm_oq_compact_iu4x2_w64_fold(&c.w, &c.x, &c.xs, y, c.m, c.k, c.b, 136)
        .unwrap();
}

fn main() {
    let mut gpu = Gpu::init().unwrap();
    let mut s = 0x5bd1_e995u32;
    let mut rnd = move || {
        s = s.wrapping_mul(1_103_515_245).wrapping_add(12345);
        s >> 8
    };
    let mut ok = true;
    for &(m, k, b) in &[
        (300usize, 5120usize, 65usize),
        (1000, 6144, 200),
        (5120, 5120, 384),
        (1000, 5120, 513),
        (5120, 17408, 512),
    ] {
        let c = build(&mut gpu, &mut rnd, m, k, b);
        let ya = gpu.alloc_tensor(&[b * m], DType::F32).unwrap();
        let yb = gpu.alloc_tensor(&[b * m], DType::F32).unwrap();
        separate(&mut gpu, &c, &ya);
        fold(&mut gpu, &c, &yb);
        gpu.device_synchronize().unwrap();
        let (a, f) = (
            gpu.download_f32(&ya).unwrap(),
            gpu.download_f32(&yb).unwrap(),
        );
        let scale = a.iter().fold(0f32, |acc, v| acc.max(v.abs()));
        let err = a
            .iter()
            .zip(&f)
            .fold(0f32, |acc, (x, y)| acc.max((x - y).abs()));
        let pass = scale > 0.0 && err <= scale * 1e-5;
        ok &= pass;
        println!(
            "M={m} K={k} B={b}: max_err/scale={:.2e} -> {}",
            err / scale,
            if pass { "PASS" } else { "FAIL" }
        );
    }
    println!("\n  proj        M      K    B  gemm+pass_ms  fold_ms  speedup");
    for &(name, m, k) in &[
        ("gate/up", 34816usize, 5120usize),
        ("down", 5120, 17408),
        ("qkv", 12288, 5120),
        ("o", 5120, 6144),
    ] {
        let b = 512;
        let c = build(&mut gpu, &mut rnd, m, k, b);
        let y = gpu.alloc_tensor(&[b * m], DType::F32).unwrap();
        let time = |gpu: &mut Gpu, f: fn(&mut Gpu, &Case, &hipfire_rdna::GpuTensor)| {
            for _ in 0..3 {
                f(gpu, &c, &y);
            }
            gpu.device_synchronize().unwrap();
            let t = Instant::now();
            for _ in 0..10 {
                f(gpu, &c, &y);
            }
            gpu.device_synchronize().unwrap();
            t.elapsed().as_secs_f64() * 100.0
        };
        let (sep, fol) = (time(&mut gpu, separate), time(&mut gpu, fold));
        println!(
            "  {name:8} {m:6} {k:6} {b:4} {sep:12.3} {fol:8.3}  {:5.2}x",
            sep / fol
        );
    }
    if !ok {
        std::process::exit(1);
    }
    println!("ALL PASS");
}
