// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.
//! Parity for `gemm_oq4_residual_mmq` (OQ4+ int8-WMMA MMQ) vs the W4A16 f16-WMMA
//! kernel `gemm_oq4_grouped_f16_wmma` (the validated reference). MMQ uses q8_1
//! int8 activation quant, so it carries ~int8 activation error vs the f16 path;
//! the tolerance allows that while catching layout/sign bugs (which blow up huge).
//!
//! Also covers `add=true` (`Y += W·x`). That arm shipped unexercised — every
//! caller passed `add=false` — and the o_proj W4A8 lever is its first real user,
//! so it is checked here against `R + set` and must match **bit-exactly**: the
//! GPU's `*yp += v` and a host add of the same `v` are the same f32 operation.
//! Both dispatch arms are covered: the `_full_add` fast path (M%128==0 &&
//! N%128==0) and the generic bounds-checked kernel.
//!
//!   cargo run --release -p hipfire-rdna --example parity_gemm_oq4_mmq [M K N]

use hipfire_rdna::{DType, Gpu};

fn lcg(seed: u32, n: usize) -> Vec<u8> {
    let mut s = seed.max(1);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12345) & 0x7fff_ffff;
            (s >> 13) as u8
        })
        .collect()
}
fn lcgf_vals(seed: u32, n: usize) -> Vec<f32> {
    let mut s = seed.max(1);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12345) & 0x7fff_ffff;
            -1.0 + (s as f32 / 2_147_483_648.0) * 2.0
        })
        .collect()
}

fn run_shape(gpu: &mut Gpu, m: usize, k: usize, n: usize) -> bool {
    let group = 256usize;
    let ng = k / group;
    let full = m % 128 == 0 && n % 128 == 0;

    let wnib = lcg(1, m * (k / 2));
    let wsc: Vec<f32> = lcgf_vals(0x11, m * ng)
        .iter()
        .map(|v| 0.01 + v.abs() * 0.1)
        .collect();
    let mut wbuf = wnib.clone();
    for s in &wsc {
        wbuf.extend_from_slice(&s.to_le_bytes());
    }
    let x: Vec<f32> = lcgf_vals(3, n * k);

    let wd = gpu.upload_raw(&wbuf, &[wbuf.len()]).unwrap();
    let xd = gpu.upload_f32(&x, &[n, k]).unwrap();

    // Reference: f16 W4A16 (per-batch-row grouped GEMM).
    let yref = gpu.alloc_tensor(&[n * m], DType::F32).unwrap();
    gpu.gemm_oq4_grouped_f16_wmma(&wd, &xd, &yref, m, k, n, group)
        .unwrap();
    gpu.device_synchronize().unwrap();
    let y_ref = gpu.download_f32(&yref).unwrap();

    // MMQ (int8 q8_1 activation), add=false (SET).
    let ymmq = gpu.alloc_tensor(&[n * m], DType::F32).unwrap();
    gpu.gemm_oq4_residual_mmq(&wd, &xd, &ymmq, m, k, n, false)
        .unwrap();
    gpu.device_synchronize().unwrap();
    let y_set = gpu.download_f32(&ymmq).unwrap();

    let mut max_abs = 0.0f32;
    let mut max_mag = 0.0f32;
    for i in 0..n * m {
        max_abs = max_abs.max((y_set[i] - y_ref[i]).abs());
        max_mag = max_mag.max(y_ref[i].abs());
    }
    let rel = max_abs / max_mag.max(1e-6);
    // int8-act error budget: ~1/127 per element, partially averaging over K.
    let set_pass = rel <= 0.05;

    // MMQ add=true onto a preloaded residual: must equal residual + SET exactly.
    let resid: Vec<f32> = lcgf_vals(0x5eed, n * m).iter().map(|v| v * 3.0).collect();
    let yadd = gpu.upload_f32(&resid, &[n * m]).unwrap();
    gpu.gemm_oq4_residual_mmq(&wd, &xd, &yadd, m, k, n, true)
        .unwrap();
    gpu.device_synchronize().unwrap();
    let y_add = gpu.download_f32(&yadd).unwrap();

    let mut add_max = 0.0f32;
    let mut touched = 0usize;
    for i in 0..n * m {
        let want = resid[i] + y_set[i];
        add_max = add_max.max((y_add[i] - want).abs());
        if y_add[i] != resid[i] {
            touched += 1;
        }
    }
    // Every output element must have been written; an add arm that skips tiles
    // would still pass a tolerance check on the elements it did touch.
    let add_pass = add_max == 0.0 && touched > (n * m) * 9 / 10;

    println!(
        "  M={m:<5} K={k} N={n:<5} [{}]: set-vs-f16 rel={rel:.4}[{}]  add-vs-(resid+set) max_abs={add_max:.6}[{}] touched={touched}/{} -> {}",
        if full { "full_add" } else { "generic" },
        if set_pass { "ok" } else { "FAIL" },
        if add_pass { "BIT-EXACT" } else { "FAIL" },
        n * m,
        if set_pass && add_pass { "PASS" } else { "FAIL" }
    );
    set_pass && add_pass
}

fn main() {
    let mut a = std::env::args().skip(1);
    let argm: Option<usize> = a.next().and_then(|s| s.parse().ok());
    let argk: Option<usize> = a.next().and_then(|s| s.parse().ok());
    let argn: Option<usize> = a.next().and_then(|s| s.parse().ok());

    let mut gpu = Gpu::init().unwrap();
    if !gpu.arch_caps.has_wmma_w32() {
        println!("SKIP parity_gemm_oq4_mmq: {} lacks wave32 WMMA", gpu.arch);
        return;
    }
    println!("parity_gemm_oq4_mmq on {}", gpu.arch);

    let shapes: Vec<(usize, usize, usize)> = match (argm, argk, argn) {
        (Some(m), Some(k), Some(n)) => vec![(m, k, n)],
        _ => vec![
            (256, 1024, 256),  // full_add fast path
            (1024, 1024, 512), // full_add, o_proj-shaped
            (1000, 1024, 100), // generic bounds-checked add
            (17, 512, 5),      // generic, tiny/ragged
        ],
    };
    let mut all = true;
    for (m, k, n) in shapes {
        all &= run_shape(&mut gpu, m, k, n);
    }
    println!("{}", if all { "ALL PASS" } else { "SOME FAILED" });
    if !all {
        std::process::exit(1);
    }
}
