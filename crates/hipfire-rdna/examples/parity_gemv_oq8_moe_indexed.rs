// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.
//! Differential test for the INDEXED OQ8 MoE gate_up GEMV — the kernel gated by
//! `HIPFIRE_QWEN35_MOE_OQ_INDEXED`, disabled in 753df2b27 (2026-07-22) for an
//! undiagnosed "finite-KLD failure" and shipped with NO test coverage.
//! See `docs/todo/2026-08-11-122b-paged-serving.md`.
//!
//! Contract note: `x` is PER-EXPERT — `[K_TOP x K]` non-batched,
//! `[N x K_TOP x K]` batched — produced by `rotate_x_mq_awq_indexed_batched`.
//! It used to be one shared row per token, which silently computed (W·s)·x for
//! every routed expert whose AWQ scale differed from the representative's.
//! Feeding a DIFFERENT x per slot here is deliberate: a shared-x kernel cannot
//! pass this test, so the old contract cannot regress back in unnoticed.
//!
//! Three-way comparison over one logical weight in BOTH layouts:
//!
//!   oracle  — CPU f32 dot. Ground truth.
//!   grouped — `gemv_oq8_grouped`, the production NON-indexed path. Known good:
//!             a 35B oq4.25++ generates coherent text through it.
//!   indexed — the kernel under test.
//!
//! Three ways rather than two because indexed-vs-oracle alone cannot separate
//! "kernel is wrong" from "oracle is wrong"; `grouped` anchors the oracle.
//!
//! K defaults cover both `gemv_oq8_grouped` variants: it switches to
//! `gemv_oq8_grouped_v2` when `K % 512 == 0`, so 768 exercises the base kernel
//! and 1024 the v2 one.
//!
//!   cargo run --release -p hipfire-rdna --example parity_gemv_oq8_moe_indexed [M K n_exp]

use hipfire_rdna::{DType, Gpu};

const K_TOP: usize = 8;
const OQ8_BLK: usize = 260;
const GROUP: usize = 256;

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

fn lcg(seed: u32, n: usize) -> Vec<u32> {
    let mut s = seed.max(1);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12345) & 0x7fff_ffff;
            s
        })
        .collect()
}

/// Full int8 range — real OQ+ blocks are mostly int4-range with a few int8
/// outliers, so this is a strictly harder input for the dequant path.
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
    let m: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(512);
    let k: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(768);
    let n_exp: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(4);
    assert_eq!(
        m % 2,
        0,
        "M must be even — the kernel splits rows into gate|up"
    );
    assert_eq!(k % GROUP, 0, "K must be a multiple of {GROUP}");
    assert!(n_exp > 0);
    let ng = k / GROUP;
    let mi = m / 2;

    let mut gpu = Gpu::init().unwrap();
    let f32b = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };

    let codes: Vec<Vec<i8>> = (0..n_exp).map(|e| lcg_i8(1 + e as u32, m * k)).collect();
    let scales: Vec<Vec<f32>> = (0..n_exp)
        .map(|e| {
            lcg_unit(0x50 + e as u32, m * ng)
                .into_iter()
                .map(|v| 0.005 + v.abs() * 0.05)
                .collect()
        })
        .collect();

    // Stride coprime with n_exp so every expert is actually selected; a shared
    // factor silently visits only a subgroup and leaves arms untested.
    let stride = (1..n_exp.max(2)).find(|s| gcd(*s, n_exp) == 1).unwrap_or(1);
    let topk: Vec<i32> = (0..K_TOP)
        .map(|j| ((j * stride + 1) % n_exp) as i32)
        .collect();

    // ── indexed layout: interleaved 260 B blocks, one blob per expert ────────
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
    // 2 f32 slots == 8 bytes == one u64 device address per expert, matching
    // `expert_gate_up_ptrs` in the qwen35 loader.
    let ptr_bytes: Vec<u8> = blobs
        .iter()
        .flat_map(|t| (t.buf.as_ptr() as u64).to_ne_bytes())
        .collect();
    let ptr_tbl = gpu.alloc_tensor(&[2 * n_exp], DType::F32).unwrap();
    gpu.hip.memcpy_htod(&ptr_tbl.buf, &ptr_bytes).unwrap();
    let topk_d = gpu
        .upload_raw(
            &topk
                .iter()
                .flat_map(|v| v.to_ne_bytes())
                .collect::<Vec<u8>>(),
            &[K_TOP],
        )
        .unwrap();

    // ── non-batched: x is [K_TOP x K], a DIFFERENT basis per krank ──────────
    let x: Vec<f32> = lcg_unit(3, K_TOP * k);
    let xd = gpu.upload_raw(&f32b(&x), &[K_TOP * k]).unwrap();
    let ygate = gpu
        .upload_raw(&vec![0u8; K_TOP * mi * 4], &[K_TOP * mi])
        .unwrap();
    let yup = gpu
        .upload_raw(&vec![0u8; K_TOP * mi * 4], &[K_TOP * mi])
        .unwrap();
    gpu.gemv_oq8g256_moe_gate_up_k8_indexed(&ptr_tbl, &topk_d, &xd, &ygate, &yup, m, k, true)
        .unwrap();
    gpu.device_synchronize().unwrap();
    let idx_gate = gpu.download_f32(&ygate).unwrap();
    let idx_up = gpu.download_f32(&yup).unwrap();

    // CPU oracle + grouped reference, per SLOT (each has its own x now).
    let oracle_slot = |krank: usize| -> Vec<f32> {
        let e = topk[krank] as usize;
        let xs = &x[krank * k..krank * k + k];
        (0..m)
            .map(|row| {
                let mut acc = 0.0f32;
                for g in 0..ng {
                    let mut gs = 0.0f32;
                    for j in 0..GROUP {
                        gs += codes[e][row * k + g * GROUP + j] as f32 * xs[g * GROUP + j];
                    }
                    acc += gs * scales[e][row * ng + g];
                }
                acc
            })
            .collect()
    };

    let mut mag = 0.0f32;
    let mut worst_grouped = 0.0f32;
    let mut worst_indexed = 0.0f32;
    let mut worst_krank = 0usize;
    for krank in 0..K_TOP {
        let e = topk[krank] as usize;
        let oracle = oracle_slot(krank);
        let wb: Vec<u8> = codes[e].iter().map(|&c| c as u8).collect();
        let wd = gpu.upload_raw(&wb, &[m * k]).unwrap();
        let sd = gpu.upload_raw(&f32b(&scales[e]), &[m * ng]).unwrap();
        let xslot = gpu
            .upload_raw(&f32b(&x[krank * k..krank * k + k]), &[1, k])
            .unwrap();
        let yd = gpu.upload_raw(&vec![0u8; m * 4], &[m]).unwrap();
        gpu.gemv_oq8_grouped(&wd, &sd, &xslot, &yd, m, k, GROUP)
            .unwrap();
        gpu.device_synchronize().unwrap();
        let grouped = gpu.download_f32(&yd).unwrap();

        let mut worst_here = 0.0f32;
        for row in 0..m {
            mag = mag.max(oracle[row].abs());
            worst_grouped = worst_grouped.max((grouped[row] - oracle[row]).abs());
            let got = if row < mi {
                idx_gate[krank * mi + row]
            } else {
                idx_up[krank * mi + (row - mi)]
            };
            worst_here = worst_here.max((got - oracle[row]).abs());
        }
        if worst_here > worst_indexed {
            worst_indexed = worst_here;
            worst_krank = krank;
        }
    }

    let tol = 1e-3 * mag.max(1.0);
    let grouped_ok = worst_grouped <= tol;
    let indexed_ok = worst_indexed <= tol;
    println!(
        "parity_gemv_oq8_moe_indexed M={m} K={k} n_exp={n_exp} K_TOP={K_TOP} on {}",
        gpu.arch
    );
    println!("  topk_indices      = {topk:?}");
    println!("  oracle max |y|    = {mag:.4}   tol = {tol:.6}");
    println!(
        "  grouped vs oracle = {worst_grouped:.6}  -> {}",
        if grouped_ok { "PASS" } else { "FAIL" }
    );
    println!(
        "  indexed vs oracle = {worst_indexed:.6}  -> {}  (worst krank={worst_krank}, expert={})",
        if indexed_ok { "PASS" } else { "FAIL" },
        topk[worst_krank]
    );
    if !grouped_ok {
        println!("  => reference path disagrees with the CPU oracle; suspect the ORACLE");
        std::process::exit(2);
    }
    if !indexed_ok {
        println!("  => INDEXED kernel diverges while grouped matches: this is the bug");
        std::process::exit(1);
    }

    // ── batched variant: x is [N x K_TOP x K]; this is what PREFILL uses ─────
    let batch = 3usize;
    let xb: Vec<f32> = lcg_unit(97, batch * K_TOP * k);
    let topk_b: Vec<i32> = (0..batch * K_TOP)
        .map(|i| ((i * stride + 1) % n_exp) as i32)
        .collect();
    let xbd = gpu.upload_raw(&f32b(&xb), &[batch * K_TOP * k]).unwrap();
    let tkbd = gpu
        .upload_raw(
            &topk_b
                .iter()
                .flat_map(|v| v.to_ne_bytes())
                .collect::<Vec<u8>>(),
            &[batch * K_TOP],
        )
        .unwrap();
    let ygb = gpu
        .upload_raw(&vec![0u8; batch * K_TOP * mi * 4], &[batch * K_TOP * mi])
        .unwrap();
    let yub = gpu
        .upload_raw(&vec![0u8; batch * K_TOP * mi * 4], &[batch * K_TOP * mi])
        .unwrap();
    gpu.gemv_oq8g256_moe_gate_up_k8_indexed_batched(
        &ptr_tbl, &tkbd, &xbd, &ygb, &yub, m, k, K_TOP, batch, true,
    )
    .unwrap();
    gpu.device_synchronize().unwrap();
    let bg = gpu.download_f32(&ygb).unwrap();
    let bu = gpu.download_f32(&yub).unwrap();

    let mut worst_batched = 0.0f32;
    let mut mag_b = 0.0f32;
    for slot in 0..batch * K_TOP {
        let e = topk_b[slot] as usize;
        let xs = &xb[slot * k..slot * k + k];
        let (b, krank) = (slot / K_TOP, slot % K_TOP);
        let base_out = b * K_TOP * mi + krank * mi;
        for row in 0..m {
            let mut acc = 0.0f32;
            for g in 0..ng {
                let mut gs = 0.0f32;
                for j in 0..GROUP {
                    gs += codes[e][row * k + g * GROUP + j] as f32 * xs[g * GROUP + j];
                }
                acc += gs * scales[e][row * ng + g];
            }
            mag_b = mag_b.max(acc.abs());
            let got = if row < mi {
                bg[base_out + row]
            } else {
                bu[base_out + (row - mi)]
            };
            worst_batched = worst_batched.max((got - acc).abs());
        }
    }
    let tol_b = 1e-3 * mag_b.max(1.0);
    let batched_ok = worst_batched <= tol_b;
    println!(
        "  batched vs oracle = {worst_batched:.6} (tol {tol_b:.6}, N={batch}) -> {}",
        if batched_ok { "PASS" } else { "FAIL" }
    );
    if !batched_ok {
        println!("  => BATCHED indexed gate_up diverges — this is the prefill-path bug");
        std::process::exit(1);
    }
    println!("  => indexed OQ8 gate_up matches under the per-expert x contract.");
}
