// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.
//! Differential test for the INDEXED OQ8 MoE **down + combine** pair — the other
//! half of the path gated by `HIPFIRE_QWEN35_MOE_OQ_INDEXED` (disabled in
//! 753df2b27 for an undiagnosed finite-KLD failure, with no test coverage).
//! Companion to `parity_gemv_oq8_moe_indexed`, which covers gate_up.
//!
//! This pair is the more interesting suspect: NEITHER GEMV applies the top-K
//! routing weight, so it must all happen in `moe_down_combine_k8_batched`. A
//! double-applied, missing, or mis-indexed routing weight there is finite and
//! plausible-looking — exactly a finite-KLD signature rather than a crash.
//!
//! Covered:
//!   1. `gemv_oq8g256_moe_down_k8_indexed_batched_expanded` per (batch, krank),
//!      against `gemv_oq8_grouped` on the same numbers — isolates the down GEMV.
//!   2. `moe_down_combine_k8_batched` folding [B×K_TOP×M] into [B×M], against a
//!      CPU oracle — isolates the routing-weight application.
//!
//! The residual is seeded NON-ZERO on purpose: the combine documents itself as
//! an in-place `+=`, and an overwrite would be invisible against a zeroed buffer.
//! Routing weights are deliberately NOT uniform and do NOT sum to 1, so a
//! dropped or doubled weight cannot hide behind a normalisation coincidence.
//!
//!   cargo run --release -p hipfire-rdna --example parity_moe_down_combine_oq8_indexed [M K n_exp B]

use hipfire_rdna::{DType, Gpu};

const K_TOP: usize = 8;
const OQ8_BLK: usize = 260;
const GROUP: usize = 256;

fn lcg(seed: u32, n: usize) -> Vec<u32> {
    let mut s = seed.max(1);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12345) & 0x7fff_ffff;
            s
        })
        .collect()
}

fn lcg_i8(seed: u32, n: usize) -> Vec<i8> {
    lcg(seed, n)
        .into_iter()
        .map(|s| {
            let v = (((s >> 11) & 0x7f) as i8).min(127);
            if s & 1 == 0 {
                v
            } else {
                -v
            }
        })
        .collect()
}

fn lcg_unit(seed: u32, n: usize) -> Vec<f32> {
    lcg(seed, n)
        .into_iter()
        .map(|s| -1.0 + (s as f32 / 2_147_483_648.0) * 2.0)
        .collect()
}

fn main() {
    let mut a = std::env::args().skip(1);
    let m: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(1024);
    let k: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(768);
    let n_exp: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(4);
    let batch: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(2);
    assert_eq!(k % GROUP, 0, "K must be a multiple of {GROUP}");
    assert!(n_exp > 0 && batch > 0);
    let ng = k / GROUP;

    let mut gpu = Gpu::init().unwrap();

    let codes: Vec<Vec<i8>> = (0..n_exp).map(|e| lcg_i8(7 + e as u32, m * k)).collect();
    let scales: Vec<Vec<f32>> = (0..n_exp)
        .map(|e| {
            lcg_unit(0x90 + e as u32, m * ng)
                .into_iter()
                .map(|v| 0.005 + v.abs() * 0.05)
                .collect()
        })
        .collect();

    // Each (batch, krank) slot gets its OWN activation — in the real forward each
    // expert consumes its own silu(gate)*up hidden, so a shared x would mask any
    // slot-indexing error in rot_batch.
    let rot: Vec<f32> = lcg_unit(11, batch * K_TOP * k);
    let topk: Vec<i32> = (0..batch * K_TOP)
        .map(|i| ((i * 5 + 2) % n_exp) as i32)
        .collect();
    // Non-uniform, not summing to 1.
    let weights: Vec<f32> = lcg_unit(23, batch * K_TOP)
        .into_iter()
        .map(|v| 0.05 + v.abs() * 0.9)
        .collect();
    let residual0: Vec<f32> = lcg_unit(31, batch * m)
        .into_iter()
        .map(|v| v * 2.0)
        .collect();

    // ── expert blobs + pointer table (indexed layout) ────────────────────────
    let mut blobs = Vec::with_capacity(n_exp);
    for e in 0..n_exp {
        let mut blob = vec![0u8; m * ng * OQ8_BLK];
        for row in 0..m {
            for g in 0..ng {
                let dst = (row * ng + g) * OQ8_BLK;
                blob[dst..dst + 4].copy_from_slice(&scales[e][row * ng + g].to_le_bytes());
                for j in 0..GROUP {
                    blob[dst + 4 + j] = codes[e][row * k + g * GROUP + j] as u8;
                }
            }
        }
        blobs.push(gpu.upload_raw(&blob, &[blob.len()]).unwrap());
    }
    let ptr_bytes: Vec<u8> = blobs
        .iter()
        .flat_map(|t| (t.buf.as_ptr() as u64).to_ne_bytes())
        .collect();
    let ptr_tbl = gpu.alloc_tensor(&[2 * n_exp], DType::F32).unwrap();
    gpu.hip.memcpy_htod(&ptr_tbl.buf, &ptr_bytes).unwrap();

    let to_bytes = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
    let topk_d = gpu
        .upload_raw(
            &topk
                .iter()
                .flat_map(|v| v.to_ne_bytes())
                .collect::<Vec<u8>>(),
            &[batch * K_TOP],
        )
        .unwrap();
    let rot_d = gpu
        .upload_raw(&to_bytes(&rot), &[batch * K_TOP * k])
        .unwrap();
    let eout_d = gpu
        .upload_raw(&vec![0u8; batch * K_TOP * m * 4], &[batch * K_TOP * m])
        .unwrap();

    // ── 1. indexed down GEMV ────────────────────────────────────────────────
    gpu.gemv_oq8g256_moe_down_k8_indexed_batched_expanded(
        &ptr_tbl, &topk_d, &rot_d, &eout_d, m, k, K_TOP, batch,
    )
    .unwrap();
    gpu.device_synchronize().unwrap();
    let eout = gpu.download_f32(&eout_d).unwrap();

    // Reference: the production non-indexed GEMV, one launch per (batch, krank).
    let mut worst_down = 0.0f32;
    let mut mag = 0.0f32;
    for b in 0..batch {
        for krank in 0..K_TOP {
            let slot = b * K_TOP + krank;
            let e = topk[slot] as usize;
            let xs = &rot[slot * k..slot * k + k];
            let wb: Vec<u8> = codes[e].iter().map(|&c| c as u8).collect();
            let wd = gpu.upload_raw(&wb, &[m * k]).unwrap();
            let sd = gpu.upload_raw(&to_bytes(&scales[e]), &[m * ng]).unwrap();
            let xd = gpu.upload_raw(&to_bytes(xs), &[1, k]).unwrap();
            let yd = gpu.upload_raw(&vec![0u8; m * 4], &[m]).unwrap();
            gpu.gemv_oq8_grouped(&wd, &sd, &xd, &yd, m, k, GROUP)
                .unwrap();
            gpu.device_synchronize().unwrap();
            let yref = gpu.download_f32(&yd).unwrap();
            for row in 0..m {
                mag = mag.max(yref[row].abs());
                worst_down = worst_down.max((eout[slot * m + row] - yref[row]).abs());
            }
        }
    }

    // ── 2. combine: x_residual += Σ_krank w · expert_outputs ────────────────
    let resid_d = gpu.upload_raw(&to_bytes(&residual0), &[batch * m]).unwrap();
    let w_d = gpu
        .upload_raw(&to_bytes(&weights), &[batch * K_TOP])
        .unwrap();
    gpu.moe_down_combine_k8_batched(&eout_d, &w_d, &resid_d, m, K_TOP, batch)
        .unwrap();
    gpu.device_synchronize().unwrap();
    let resid = gpu.download_f32(&resid_d).unwrap();

    // Oracle built from the GPU's own expert_outputs, so this isolates the
    // combine's weighting/accumulation from any down-GEMV error already measured.
    let mut worst_comb = 0.0f32;
    let mut resid_mag = 0.0f32;
    for b in 0..batch {
        for row in 0..m {
            let mut acc = residual0[b * m + row];
            for krank in 0..K_TOP {
                let slot = b * K_TOP + krank;
                acc += weights[slot] * eout[slot * m + row];
            }
            resid_mag = resid_mag.max(acc.abs());
            worst_comb = worst_comb.max((resid[b * m + row] - acc).abs());
        }
    }

    let tol_down = 1e-3 * mag.max(1.0);
    let tol_comb = 1e-3 * resid_mag.max(1.0);
    let down_ok = worst_down <= tol_down;
    let comb_ok = worst_comb <= tol_comb;

    println!(
        "parity_moe_down_combine_oq8_indexed M={m} K={k} n_exp={n_exp} B={batch} K_TOP={K_TOP} on {}",
        gpu.arch
    );
    println!(
        "  down    vs grouped = {worst_down:.6} (tol {tol_down:.6}) -> {}",
        if down_ok { "PASS" } else { "FAIL" }
    );
    println!(
        "  combine vs oracle  = {worst_comb:.6} (tol {tol_comb:.6}) -> {}",
        if comb_ok { "PASS" } else { "FAIL" }
    );

    if !down_ok {
        println!(
            "  => INDEXED down GEMV diverges from the production grouped path: this is the bug"
        );
        std::process::exit(1);
    }
    if !comb_ok {
        println!(
            "  => COMBINE misapplies the top-K routing weights or the residual +=: this is the bug"
        );
        std::process::exit(1);
    }
    println!("  => down + combine match. The finite-KLD failure is NOT in this pair.");
}
