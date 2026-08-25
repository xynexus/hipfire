// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Reference check for `gemm_oq_compact_moe_grouped_wmma` — the grouped MoE
//! GEMM that `moe_grouped_gemm_supported_for_dtype` keeps behind
//! `HIPFIRE_MOE_COMPACT_GROUPED=1` because, in its own words, "the kernel has
//! never been checked against a reference".
//!
//! This is that reference. Two independent ones, actually:
//!
//!   oracle  — f64 CPU dot over `decode_logical` weights. Says whether the
//!             kernel computes the right VALUE.
//!   gemv    — `gemv_oq_compact_moe_gate_up_k8_indexed_batched`, the shipping
//!             decode path, verified by `parity_gemv_oq_compact_moe`. Says
//!             whether the kernel agrees with what production actually runs.
//!
//! The second is the one that matters for spec-decode: DFlash verify commits
//! tokens that must match AR decode, and AR decode runs the GEMV.
//!
//! ⚠️ EXPECT A GAP, and read it carefully. The grouped kernel converts
//! activations to f16 (`ensure_fp16_x`, because WMMA needs f16 inputs) while the
//! GEMV consumes f32. So it CANNOT be bit-exact with decode by construction, and
//! a small cross error is the floor, not a bug. What this harness is for is
//! telling a ~1e-3 f16-rounding gap apart from a real addressing/layout defect,
//! which is what the Oq8 sibling turned out to have ("FAST and WRONG", 1.8x and
//! garbage output).
//!
//! Usage:
//!   cargo run --release -p hipfire-runtime --example parity_gemm_oq_compact_moe_grouped [M K n_exp N]

use hipfire_rdna::{DType, Gpu};
use hipfire_runtime::oq8_arch::{normalize_compact_overlays, split_compact_planes};

const GROUP: usize = 256;
const K_TOP: usize = 8;
const N_OUT: usize = 3;
const BLOCK_STRIDE: usize = 130 + 2 * N_OUT;
const TILE: usize = 16; // slots per tile — the kernel's blockIdx.y granularity

fn lcg(seed: u32) -> impl FnMut() -> u32 {
    let mut s = seed.max(1);
    move || {
        s = s.wrapping_mul(1_103_515_245).wrapping_add(12345) & 0x7fff_ffff;
        s
    }
}

fn f32_to_f16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mant = ((bits >> 13) & 0x3ff) as u16;
    if exp <= 0 {
        return sign;
    }
    sign | ((exp as u16) << 10) | mant
}
fn f16_bits_to_f32(bits: u16) -> f32 {
    hipfire_primitives::conv::f16_to_f32(bits)
}
fn sext4(nib: u8) -> i32 {
    ((nib as i32) << 28) >> 28
}

fn build_expert(seed: u32, m: usize, k: usize) -> Vec<u8> {
    let ng = k / GROUP;
    let mut rng = lcg(seed);
    let mut blob = vec![0u8; m * ng * BLOCK_STRIDE];
    for b in 0..(m * ng) {
        let base = b * BLOCK_STRIDE;
        let scale = f16_bits_to_f32(f32_to_f16_bits(0.004 + (rng() % 64) as f32 * 0.001));
        blob[base..base + 2].copy_from_slice(&f32_to_f16_bits(scale).to_le_bytes());
        for i in 0..128 {
            blob[base + 2 + i] = (rng() & 0xff) as u8;
        }
        for e in 0..N_OUT {
            blob[base + 130 + 2 * e] = (rng() % GROUP as u32) as u8;
            blob[base + 130 + 2 * e + 1] = (rng() & 0xff) as u8;
        }
    }
    blob
}

fn decode_logical(blob: &[u8], m: usize, k: usize) -> Vec<f32> {
    let ng = k / GROUP;
    let mut w = vec![0f32; m * k];
    for row in 0..m {
        for g in 0..ng {
            let base = (row * ng + g) * BLOCK_STRIDE;
            let scale = f16_bits_to_f32(u16::from_le_bytes([blob[base], blob[base + 1]]));
            let mut code = [0i32; GROUP];
            for i in 0..128 {
                let byte = blob[base + 2 + i];
                code[2 * i] = sext4(byte & 0x0f);
                code[2 * i + 1] = sext4(byte >> 4);
            }
            for e in 0..N_OUT {
                code[blob[base + 130 + 2 * e] as usize] = blob[base + 130 + 2 * e + 1] as i8 as i32;
            }
            for i in 0..GROUP {
                w[row * k + g * GROUP + i] = scale * code[i] as f32;
            }
        }
    }
    w
}

fn max_rel(a: &[f32], b: &[f32]) -> f64 {
    let scale = a.iter().fold(0f64, |m, v| m.max(v.abs() as f64)).max(1e-30);
    a.iter()
        .zip(b)
        .fold(0f64, |m, (p, q)| m.max((*p as f64 - *q as f64).abs()))
        / scale
}

fn f32b(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn i32b(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// Sort routed slots by expert into TILE-sized runs, exactly the layout the
/// kernel reads: `expert_tile_ids[tile]` names one expert for all 16 of its
/// slots (-1 = skip), and `sorted_slot_index[slot]` is the flat `(token*K_TOP +
/// krank)` id, or -1 for padding. Padding to a tile boundary per expert is what
/// lets one tile carry one expert.
fn build_sort(topk: &[i32], n_exp: usize) -> (Vec<i32>, Vec<i32>, Vec<usize>) {
    let mut by_expert: Vec<Vec<i32>> = vec![Vec::new(); n_exp];
    for (flat, &e) in topk.iter().enumerate() {
        by_expert[e as usize].push(flat as i32);
    }
    let mut sorted = Vec::new();
    let mut tile_ids = Vec::new();
    let mut slot_of_flat = vec![usize::MAX; topk.len()];
    for (e, slots) in by_expert.iter().enumerate() {
        if slots.is_empty() {
            continue;
        }
        for chunk in slots.chunks(TILE) {
            tile_ids.push(e as i32);
            for &flat in chunk {
                slot_of_flat[flat as usize] = sorted.len();
                sorted.push(flat);
            }
            for _ in chunk.len()..TILE {
                sorted.push(-1);
            }
        }
    }
    (sorted, tile_ids, slot_of_flat)
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let p = |i: usize, d: usize| a.get(i).and_then(|s| s.parse().ok()).unwrap_or(d);
    let (m, k, n_exp, batch) = (p(1, 512), p(2, 1024), p(3, 8), p(4, 16));
    assert!(m % 2 == 0 && k % GROUP == 0, "M even, K a multiple of 256");

    let mut gpu = Gpu::init().expect("gpu");
    println!("gpu: {}", gpu.arch);
    println!("M={m} K={k} n_exp={n_exp} batch={batch} K_TOP={K_TOP}");

    let mi = m / 2;
    let raw: Vec<Vec<u8>> = (0..n_exp)
        .map(|e| build_expert(11 + e as u32, m, k))
        .collect();
    let logical: Vec<Vec<f32>> = raw.iter().map(|b| decode_logical(b, m, k)).collect();
    let planes: Vec<Vec<u8>> = raw
        .iter()
        .map(|b| {
            let mut owned = b.clone();
            normalize_compact_overlays(&mut owned, m, k, GROUP);
            split_compact_planes(&owned, m, k, GROUP)
        })
        .collect();

    // Every expert selected at least once — an unvisited slot hides addressing bugs.
    let topk: Vec<i32> = (0..batch * K_TOP).map(|j| (j % n_exp) as i32).collect();
    let mut rng = lcg(0xC0FFEE);
    let x: Vec<f32> = (0..batch * K_TOP * k)
        .map(|_| ((rng() % 2001) as f32 - 1000.0) / 1000.0)
        .collect();

    // ── f64 CPU oracle ──────────────────────────────────────────────────────
    let mut oracle = vec![0f32; batch * K_TOP * m];
    for s in 0..batch * K_TOP {
        let e = topk[s] as usize;
        let xr = &x[s * k..s * k + k];
        for row in 0..m {
            let mut acc = 0f64;
            for j in 0..k {
                acc += logical[e][row * k + j] as f64 * xr[j] as f64;
            }
            oracle[s * m + row] = acc as f32;
        }
    }

    // ── upload ──────────────────────────────────────────────────────────────
    let wt: Vec<_> = planes
        .iter()
        .map(|b| gpu.upload_raw(b, &[b.len()]).unwrap())
        .collect();
    let ptrs: Vec<u64> = wt.iter().map(|t| t.buf.as_ptr() as u64).collect();
    let ptr_t = gpu
        .upload_raw(
            &ptrs
                .iter()
                .flat_map(|q| q.to_le_bytes())
                .collect::<Vec<u8>>(),
            &[2 * n_exp],
        )
        .unwrap();
    let strides = gpu
        .upload_raw(&i32b(&vec![BLOCK_STRIDE as i32; n_exp]), &[n_exp])
        .unwrap();
    let idx_t = gpu.upload_raw(&i32b(&topk), &[topk.len()]).unwrap();
    let x_t = gpu.upload_raw(&f32b(&x), &[x.len()]).unwrap();

    // ── arm 1: shipping decode GEMV ─────────────────────────────────────────
    let gt = gpu.alloc_tensor(&[batch * K_TOP * mi], DType::F32).unwrap();
    let ut = gpu.alloc_tensor(&[batch * K_TOP * mi], DType::F32).unwrap();
    gpu.gemv_oq_compact_moe_gate_up_k8_indexed_batched(
        &ptr_t, &idx_t, &strides, &x_t, &gt, &ut, m, k, K_TOP, batch, true,
    )
    .unwrap();
    gpu.device_synchronize().unwrap();
    let (gv, uv) = (
        gpu.download_f32(&gt).unwrap(),
        gpu.download_f32(&ut).unwrap(),
    );
    let mut y_gemv = vec![0f32; batch * K_TOP * m];
    for s in 0..batch * K_TOP {
        y_gemv[s * m..s * m + mi].copy_from_slice(&gv[s * mi..s * mi + mi]);
        y_gemv[s * m + mi..s * m + m].copy_from_slice(&uv[s * mi..s * mi + mi]);
    }

    // ── arm 2: grouped WMMA GEMM ────────────────────────────────────────────
    let (sorted, tile_ids, slot_of_flat) = build_sort(&topk, n_exp);
    let m_total = sorted.len();
    let sorted_t = gpu.upload_raw(&i32b(&sorted), &[m_total]).unwrap();
    let tiles_t = gpu.upload_raw(&i32b(&tile_ids), &[tile_ids.len()]).unwrap();
    let y_g = gpu.alloc_tensor(&[m_total * m], DType::F32).unwrap();
    gpu.gemm_oq_compact_moe_grouped_wmma(
        &ptr_t,
        &tiles_t,
        &sorted_t,
        &x_t,
        &y_g,
        m,
        k,
        1, // x rows ARE flat slots here (x is per-slot), so no division
        m_total,
        batch * K_TOP,
        BLOCK_STRIDE,
    )
    .unwrap();
    gpu.device_synchronize().unwrap();
    let yg_raw = gpu.download_f32(&y_g).unwrap();
    // un-permute back to flat slot order
    let mut y_grp = vec![0f32; batch * K_TOP * m];
    for flat in 0..batch * K_TOP {
        let s = slot_of_flat[flat];
        assert!(s != usize::MAX, "slot {flat} never placed");
        y_grp[flat * m..flat * m + m].copy_from_slice(&yg_raw[s * m..s * m + m]);
    }

    // ── arm 3: f32-activation grouped GEMM (must be BIT-EXACT vs the GEMV) ──
    let y_f = gpu.alloc_tensor(&[m_total * m], DType::F32).unwrap();
    gpu.gemm_oq_compact_moe_grouped_f32(
        &ptr_t,
        &tiles_t,
        &sorted_t,
        &x_t,
        &y_f,
        m,
        k,
        1,
        m_total,
        BLOCK_STRIDE,
    )
    .unwrap();
    gpu.device_synchronize().unwrap();
    let yf_raw = gpu.download_f32(&y_f).unwrap();
    let mut y_f32 = vec![0f32; batch * K_TOP * m];
    for flat in 0..batch * K_TOP {
        let s = slot_of_flat[flat];
        y_f32[flat * m..flat * m + m].copy_from_slice(&yf_raw[s * m..s * m + m]);
    }
    let exact = y_f32
        .iter()
        .zip(&y_gemv)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();

    // ── timing: does hoisting the weight read out of the slot loop pay? ─────
    let iters = 50;
    let t_gemv = {
        gpu.device_synchronize().unwrap();
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            gpu.gemv_oq_compact_moe_gate_up_k8_indexed_batched(
                &ptr_t, &idx_t, &strides, &x_t, &gt, &ut, m, k, K_TOP, batch, true,
            )
            .unwrap();
        }
        gpu.device_synchronize().unwrap();
        t0.elapsed().as_secs_f64() * 1e3 / iters as f64
    };
    let t_f32 = {
        gpu.device_synchronize().unwrap();
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            gpu.gemm_oq_compact_moe_grouped_f32(
                &ptr_t,
                &tiles_t,
                &sorted_t,
                &x_t,
                &y_f,
                m,
                k,
                1,
                m_total,
                BLOCK_STRIDE,
            )
            .unwrap();
        }
        gpu.device_synchronize().unwrap();
        t0.elapsed().as_secs_f64() * 1e3 / iters as f64
    };
    println!(
        "  timing: gemv {t_gemv:.3} ms   f32-grouped {t_f32:.3} ms   speedup {:.2}x",
        t_gemv / t_f32.max(1e-9)
    );

    // ── verdict ─────────────────────────────────────────────────────────────
    let e_gemv = max_rel(&oracle, &y_gemv);
    let e_grp = max_rel(&oracle, &y_grp);
    let cross = max_rel(&y_gemv, &y_grp);
    println!("  gemv   vs oracle : {e_gemv:.3e}   (shipping decode path)");
    println!("  grouped vs oracle: {e_grp:.3e}   (kernel under test)");
    println!("  grouped vs gemv  : {cross:.3e}   (what spec-decode verify needs small)");
    let e_f32 = max_rel(&oracle, &y_f32);
    let cross_f32 = max_rel(&y_gemv, &y_f32);
    println!("  f32grp vs oracle : {e_f32:.3e}   (f32-activation kernel)");
    println!(
        "  f32grp vs gemv   : {cross_f32:.3e}   bit-mismatches {exact}/{}  {}",
        y_gemv.len(),
        if exact == 0 {
            "BIT-EXACT"
        } else {
            "NOT bit-exact"
        }
    );
    // f16 activations put the floor near 1e-3; a layout/addressing defect is
    // orders above that, which is how the Oq8 sibling failed.
    // The f32 arm's bar is strict equality; the WMMA arm's is the f16 floor.
    let ok = e_grp < 5e-3 && cross < 5e-3 && exact == 0;
    println!(
        "{}",
        if ok {
            "PASS — wmma arm at the f16 floor, f32 arm BIT-EXACT vs the decode GEMV"
        } else {
            "FAIL — see which arm: wmma above the f16 floor, or f32 arm not bit-exact"
        }
    );
    if !ok {
        std::process::exit(1);
    }
}
