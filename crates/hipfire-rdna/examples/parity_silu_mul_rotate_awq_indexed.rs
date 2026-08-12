// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.
//! Parity for `fused_silu_mul_mq_rotate_awq_indexed` — the per-EXPERT AWQ
//! silu_mul+rotate that the indexed MoE down_proj stage needs. The down-side
//! mirror of `parity_rotate_x_mq_awq_indexed`.
//!
//! Reference is the EXISTING, trusted `fused_silu_mul_rotate_mq_awq_batched`,
//! driven one slot at a time with that slot's expert scale. So this asks
//! exactly one question: does the indexed kernel select the right awq_scale per
//! (token, krank)? The silu*up and the FWHT are shared code between the two, so
//! any disagreement is selection or addressing — the bug class this kernel
//! exists to fix.
//!
//! Also checks two degenerate arms that production relies on:
//!   - null per-expert pointer (mixed artifact: some experts AWQ-scaled, some
//!     not) vs `fused_silu_mul_rotate_mq_batched`;
//!   - null TABLE (no expert at this layer has a sidecar) vs the same, which
//!     must be EXACTLY bit-identical since it is the same arithmetic.
//!
//!   cargo run --release -p hipfire-rdna --example parity_silu_mul_rotate_awq_indexed [MI n_exp N]

use hipfire_rdna::{DType, Gpu};

const K_TOP: usize = 8;

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

fn lcg_unit(seed: u32, n: usize) -> Vec<f32> {
    let mut s = seed.max(1);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12345) & 0x7fff_ffff;
            -1.0 + (s as f32 / 2_147_483_648.0) * 2.0
        })
        .collect()
}

fn main() {
    let mut a = std::env::args().skip(1);
    let mi: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(768);
    let n_exp: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(6);
    let n: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(3);
    assert_eq!(mi % 256, 0, "MI must be a multiple of 256");
    let slots = n * K_TOP;

    let mut gpu = Gpu::init().unwrap();
    let f32b = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };

    // gate/up are already expanded per slot — that is the down-side layout.
    let gate = lcg_unit(11, slots * mi);
    let up = lcg_unit(29, slots * mi);
    let gate_d = gpu.upload_raw(&f32b(&gate), &[slots * mi]).unwrap();
    let up_d = gpu.upload_raw(&f32b(&up), &[slots * mi]).unwrap();

    // Per-expert scales, deliberately DIFFERENT — that difference is the whole
    // reason this kernel exists. Kept away from zero so the divide is stable.
    let scales: Vec<Vec<f32>> = (0..n_exp)
        .map(|e| {
            lcg_unit(0x700 + e as u32, mi)
                .into_iter()
                .map(|v| 0.25 + v.abs() * 1.5)
                .collect()
        })
        .collect();
    // Expert n_exp-1 deliberately has NO sidecar (null pointer) to exercise the
    // mixed-artifact arm.
    let null_expert = n_exp - 1;
    let scale_tensors: Vec<_> = scales
        .iter()
        .map(|s| gpu.upload_raw(&f32b(s), &[mi]).unwrap())
        .collect();
    let ptr_bytes: Vec<u8> = (0..n_exp)
        .flat_map(|e| {
            let p: u64 = if e == null_expert {
                0
            } else {
                scale_tensors[e].buf.as_ptr() as u64
            };
            p.to_ne_bytes()
        })
        .collect();
    let ptr_tbl = gpu.alloc_tensor(&[2 * n_exp], DType::F32).unwrap();
    gpu.hip.memcpy_htod(&ptr_tbl.buf, &ptr_bytes).unwrap();

    // The stride MUST be coprime with n_exp or the sequence only visits a
    // subgroup and silently never exercises the null-sidecar arm — the kind of
    // hole that makes a test pass while covering nothing.
    let stride = (1..n_exp).find(|s| gcd(*s, n_exp) == 1).unwrap_or(1);
    let topk: Vec<i32> = (0..slots)
        .map(|i| ((i * stride + 1) % n_exp) as i32)
        .collect();
    let covered: std::collections::BTreeSet<i32> = topk.iter().copied().collect();
    assert_eq!(
        covered.len(),
        n_exp,
        "topk must select every expert (got {covered:?}); otherwise arms go untested"
    );
    let topk_d = gpu
        .upload_raw(
            &topk
                .iter()
                .flat_map(|v| v.to_ne_bytes())
                .collect::<Vec<u8>>(),
            &[slots],
        )
        .unwrap();

    let out = gpu
        .upload_raw(&vec![0u8; slots * mi * 4], &[slots * mi])
        .unwrap();
    gpu.fused_silu_mul_rotate_mq_awq_indexed(
        &gate_d,
        &up_d,
        Some(&ptr_tbl),
        &topk_d,
        &out,
        mi,
        slots,
    )
    .unwrap();
    gpu.device_synchronize().unwrap();
    let got = gpu.download_f32(&out).unwrap();

    // Reference: one slot at a time through the trusted kernels.
    let mut worst = 0.0f32;
    let mut mag = 0.0f32;
    let mut worst_slot = 0usize;
    let mut null_checked = 0usize;
    for slot in 0..slots {
        let e = topk[slot] as usize;
        let g = &gate[slot * mi..slot * mi + mi];
        let u = &up[slot * mi..slot * mi + mi];
        let gd = gpu.upload_raw(&f32b(g), &[mi]).unwrap();
        let ud = gpu.upload_raw(&f32b(u), &[mi]).unwrap();
        let refd = gpu.upload_raw(&vec![0u8; mi * 4], &[mi]).unwrap();
        if e == null_expert {
            gpu.fused_silu_mul_rotate_mq_batched(&gd, &ud, &refd, mi, 1)
                .unwrap();
            null_checked += 1;
        } else {
            gpu.fused_silu_mul_rotate_mq_awq_batched(&gd, &ud, &scale_tensors[e], &refd, mi, 1)
                .unwrap();
        }
        gpu.device_synchronize().unwrap();
        let expect = gpu.download_f32(&refd).unwrap();
        for i in 0..mi {
            mag = mag.max(expect[i].abs());
            let d = (got[slot * mi + i] - expect[i]).abs();
            if d > worst {
                worst = d;
                worst_slot = slot;
            }
        }
    }
    assert!(
        null_checked > 0,
        "no null-sidecar slot was exercised — the mixed-artifact arm is untested"
    );

    // Null TABLE arm: no expert has a sidecar, so every slot must be the plain
    // silu_mul+rotate. Same arithmetic, so demand exact equality.
    let out_null = gpu
        .upload_raw(&vec![0u8; slots * mi * 4], &[slots * mi])
        .unwrap();
    gpu.fused_silu_mul_rotate_mq_awq_indexed(&gate_d, &up_d, None, &topk_d, &out_null, mi, slots)
        .unwrap();
    let plain = gpu
        .upload_raw(&vec![0u8; slots * mi * 4], &[slots * mi])
        .unwrap();
    gpu.fused_silu_mul_rotate_mq_batched(&gate_d, &up_d, &plain, mi, slots)
        .unwrap();
    gpu.device_synchronize().unwrap();
    let got_null = gpu.download_f32(&out_null).unwrap();
    let want_null = gpu.download_f32(&plain).unwrap();
    let worst_null = got_null
        .iter()
        .zip(&want_null)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    let tol = 1e-4 * mag.max(1.0);
    let pass = worst <= tol && worst_null == 0.0;
    println!(
        "parity_silu_mul_rotate_awq_indexed MI={mi} n_exp={n_exp} N={n} K_TOP={K_TOP} on {}",
        gpu.arch
    );
    println!("  null-sidecar slots exercised: {null_checked}");
    println!(
        "  null-TABLE arm vs fused_silu_mul_rotate_mq_batched: worst |diff| = {worst_null:.8} (must be exactly 0)"
    );
    println!(
        "  worst |diff| = {worst:.8} (tol {tol:.8}, mag {mag:.4}, worst slot {worst_slot} expert {}) -> {}",
        topk[worst_slot],
        if pass { "PASS" } else { "FAIL" }
    );
    if !pass {
        std::process::exit(1);
    }
}
