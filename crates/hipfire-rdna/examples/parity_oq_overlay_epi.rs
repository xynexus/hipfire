// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.
//! PARKED (measured slower, not merged): `gemm_oq_compact_iu4x2_w64_epi` against
//! `gemm_oq_compact_iu4x2_w64` followed by the separate n_ov=3 pass, on the same
//! device bytes: the two must agree BIT FOR BIT. Shapes cover both prefill tiles
//! (default below 384 rows, m128 from 384), partial M and B blocks, and the
//! 27B's K. Each arm's correctness against an oracle is its own parity example
//! (parity_gemm_oq_compact_iu4x2_w64, parity_oq_overlay_tr); this checks that
//! fusing changed nothing.
//!
//!   cargo run --release -p hipfire-rdna --example parity_oq_overlay_epi
use hipfire_rdna::{DType, Gpu, OQ_OVERLAY_SLACK};

fn main() {
    let mut gpu = Gpu::init().unwrap();
    let mut s = 0x9e37_79b9u32;
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
        let ng = k / 256;
        let stride = 136usize;
        // Device format: nibble plane [M][ng][128], then side [M][ng][8] =
        // f16 scale + 3 (idx, val) pairs.
        let mut w = vec![0u8; m * ng * stride];
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
        let xilv: Vec<u8> = (0..b * k).map(|_| rnd() as u8).collect();
        let xs: Vec<f32> = (0..b * ng)
            .map(|_| 1e-4 + (rnd() % 1000) as f32 * 1e-5)
            .collect();
        let xt: Vec<u8> = (0..k * b + OQ_OVERLAY_SLACK).map(|_| rnd() as u8).collect();
        let xst: Vec<f32> = (0..ng * b + OQ_OVERLAY_SLACK)
            .map(|_| 1e-4 + (rnd() % 1000) as f32 * 1e-5)
            .collect();
        let wd = gpu.upload_raw(&w, &[w.len()]).unwrap();
        let xd = gpu.upload_raw(&xilv, &[xilv.len()]).unwrap();
        let xsd = gpu.upload_f32(&xs, &[xs.len()]).unwrap();
        let xtd = gpu.upload_raw(&xt, &[xt.len()]).unwrap();
        let xstd = gpu.upload_f32(&xst, &[xst.len()]).unwrap();
        let ya = gpu.alloc_tensor(&[b * m], DType::F32).unwrap();
        let yb = gpu.alloc_tensor(&[b * m], DType::F32).unwrap();

        gpu.gemm_oq_compact_iu4x2_w64(&wd, &xd, &xsd, &ya, m, k, b, stride)
            .unwrap();
        gpu.oq_compact_overlay_correct_t(&wd, &xtd, &xstd, &ya, m, k, b, 256, stride)
            .unwrap();
        gpu.gemm_oq_compact_iu4x2_w64_epi(&wd, &xd, &xsd, &yb, m, k, b, stride, &xtd, &xstd)
            .unwrap();
        gpu.device_synchronize().unwrap();
        let (a, f) = (
            gpu.download_f32(&ya).unwrap(),
            gpu.download_f32(&yb).unwrap(),
        );
        let diff = a
            .iter()
            .zip(&f)
            .filter(|(x, y)| x.to_bits() != y.to_bits())
            .count();
        let pass = diff == 0 && a.iter().any(|v| *v != 0.0);
        ok &= pass;
        println!(
            "M={m} K={k} B={b}: {diff} of {} outputs differ -> {}",
            a.len(),
            if pass { "PASS" } else { "FAIL" }
        );
    }
    if !ok {
        std::process::exit(1);
    }
    println!("ALL PASS");
}
