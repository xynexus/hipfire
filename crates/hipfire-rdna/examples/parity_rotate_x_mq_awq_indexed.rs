// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.
//! Parity for `rotate_x_mq_awq_indexed_batched` — the per-EXPERT AWQ rotate that
//! the indexed MoE path needs.
//!
//! Reference is the EXISTING, trusted `rotate_x_mq_awq_batched`, driven one slot
//! at a time with that slot's expert scale. So this asks exactly one question:
//! does the indexed kernel select the right awq_scale per (token, krank) and
//! write it to the right slot? The FWHT itself is shared code between the two,
//! so any disagreement is selection or addressing — which is the bug class this
//! kernel exists to fix.
//!
//! Also checks the null-pointer arm (expert with no sidecar) against
//! `rotate_x_mq_batched`, since mixed artifacts — some experts AWQ-scaled, some
//! not — must stay correct.
//!
//!   cargo run --release -p hipfire-rdna --example parity_rotate_x_mq_awq_indexed [K n_exp N]

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
    let k: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(2048);
    let n_exp: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(6);
    let n: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(3);
    assert_eq!(k % 256, 0, "K must be a multiple of 256");

    let mut gpu = Gpu::init().unwrap();
    let f32b = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };

    let x = lcg_unit(5, n * k);
    let xd = gpu.upload_raw(&f32b(&x), &[n * k]).unwrap();

    // Per-expert scales, deliberately DIFFERENT — that difference is the whole
    // reason this kernel exists. Kept away from zero so the divide is stable.
    let scales: Vec<Vec<f32>> = (0..n_exp)
        .map(|e| {
            lcg_unit(0x300 + e as u32, k)
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
        .map(|s| gpu.upload_raw(&f32b(s), &[k]).unwrap())
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

    // Include the null expert in the selection, and repeat experts across slots.
    // The stride MUST be coprime with n_exp or the sequence only visits a
    // subgroup — `(i*3+1) % 6` visits just {1,4} and silently never exercises
    // the null-sidecar arm, which is exactly the kind of hole that makes a test
    // pass while covering nothing.
    let stride = (1..n_exp).find(|s| gcd(*s, n_exp) == 1).unwrap_or(1);
    let topk: Vec<i32> = (0..n * K_TOP)
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
            &[n * K_TOP],
        )
        .unwrap();

    let xrot = gpu
        .upload_raw(&vec![0u8; n * K_TOP * k * 4], &[n * K_TOP * k])
        .unwrap();
    gpu.rotate_x_mq_awq_indexed_batched(&xd, Some(&ptr_tbl), &topk_d, &xrot, k, K_TOP, n)
        .unwrap();
    gpu.device_synchronize().unwrap();
    let got = gpu.download_f32(&xrot).unwrap();

    // Reference: one slot at a time through the trusted kernels.
    let mut worst = 0.0f32;
    let mut mag = 0.0f32;
    let mut worst_slot = 0usize;
    let mut null_checked = 0usize;
    for slot in 0..n * K_TOP {
        let token = slot / K_TOP;
        let e = topk[slot] as usize;
        let row = &x[token * k..token * k + k];
        let rowd = gpu.upload_raw(&f32b(row), &[k]).unwrap();
        let refd = gpu.upload_raw(&vec![0u8; k * 4], &[k]).unwrap();
        if e == null_expert {
            gpu.rotate_x_mq_batched(&rowd, &refd, k, 1).unwrap();
            null_checked += 1;
        } else {
            gpu.rotate_x_mq_awq_batched(&rowd, &scale_tensors[e], &refd, k, 1)
                .unwrap();
        }
        gpu.device_synchronize().unwrap();
        let expect = gpu.download_f32(&refd).unwrap();
        for i in 0..k {
            mag = mag.max(expect[i].abs());
            let d = (got[slot * k + i] - expect[i]).abs();
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
    // Null TABLE arm: no expert at this layer has a sidecar. Every slot must
    // then be the PLAIN rotation of its token — this is the path non-AWQ
    // artifacts take, and it must still expand per slot because the indexed
    // gate_up GEMVs read x per slot regardless of AWQ. Compared bit-for-bit
    // against `rotate_x_mq_batched`, the kernel it has to degenerate to.
    let xrot_null = gpu
        .upload_raw(&vec![0u8; n * K_TOP * k * 4], &[n * K_TOP * k])
        .unwrap();
    gpu.rotate_x_mq_awq_indexed_batched(&xd, None, &topk_d, &xrot_null, k, K_TOP, n)
        .unwrap();
    let plain = gpu.upload_raw(&vec![0u8; n * k * 4], &[n * k]).unwrap();
    gpu.rotate_x_mq_batched(&xd, &plain, k, n).unwrap();
    gpu.device_synchronize().unwrap();
    let got_null = gpu.download_f32(&xrot_null).unwrap();
    let want_null = gpu.download_f32(&plain).unwrap();
    let mut worst_null = 0.0f32;
    for slot in 0..n * K_TOP {
        let token = slot / K_TOP;
        for i in 0..k {
            worst_null = worst_null.max((got_null[slot * k + i] - want_null[token * k + i]).abs());
        }
    }

    let tol = 1e-4 * mag.max(1.0);
    let pass = worst <= tol && worst_null == 0.0;
    println!(
        "parity_rotate_x_mq_awq_indexed K={k} n_exp={n_exp} N={n} K_TOP={K_TOP} on {}",
        gpu.arch
    );
    println!("  null-sidecar slots exercised: {null_checked}");
    println!(
        "  null-TABLE arm vs rotate_x_mq_batched: worst |diff| = {worst_null:.8} (must be exactly 0)"
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
