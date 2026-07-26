// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.
//
//! Cross-codec op4++ parity: **GPU serving path vs the shared NPU op4++ codec**.
//!
//! Milestone M1b (`docs/plans/2026-07-17-npu-w4a8-op4pp-moe-qwen35.md`). One set
//! of *stored rotated int4* weights + per-(col,group) scales is the single source
//! of truth. It is encoded into BOTH device layouts:
//!   - NPU: `OpusPackedMatrix::from_payload` (qt=33, 130-byte blocks) — the exact
//!     matrix the resident W4A8 FFN consumes.
//!   - GPU: `WeightTensor{Oq4G256}` combined buffer (`[N,K/2]` nibbles ++ f32 scales).
//! Because both come from the same nibbles+scales, weight decode is **bit-exact by
//! construction (parity leg L1)**.
//!
//! Then the GPU `weight_gemv` Oq4G256 decode arm (W4A16: full-precision activation
//! × dequant 4-bit weight, `weights.rs:562`) is checked against
//! `OpusPackedMatrix::reference_dequantized_bf16_f32` — the *same math* on the CPU
//! (parity leg L3). This ties the GPU serving path to the shared codec the NPU
//! runs, so a heterogeneous NPU/GPU expert split stays numerically coherent.
//! `reference_f32` (W4A8, the NPU's precision) is reported alongside to show the
//! int8-activation gap.
//!
//!   cargo run --release -p hipfire-runtime --example oq4_shared_codec_gpu_parity [K N]

use hipfire_primitives::conv::{f16_to_f32, f32_to_f16};
use hipfire_rdna::{DType, Gpu};
use hipfire_runtime::weights::{weight_gemv, WeightTensor};
use hipfire_xdna::OpusPackedMatrix;

const GROUP: usize = 256;

fn lcg(seed: u32, n: usize) -> Vec<f32> {
    let mut s = seed.max(1);
    (0..n)
        .map(|i| {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12345) & 0x7fff_ffff;
            let v = (s as f32 / 2_147_483_648.0) - 0.5;
            if i % 89 == 0 {
                v * 8.0
            } else {
                v
            }
        })
        .collect()
}

/// Stored rotated int4 weight q[k][n] in [-7, 7] and fp16-rounded per-(n,group)
/// scale (fp16 so the NPU fp16 block scale and GPU f32 scale hold the same value).
fn source_weight(k: usize, n: usize, seed: u64) -> (Vec<i8>, Vec<f32>) {
    let ng = k / GROUP;
    let mut q = vec![0i8; k * n];
    let mut scale = vec![0.0f32; n * ng];
    for col in 0..n {
        for g in 0..ng {
            let raw = 0.012 * (1.0 + ((col + 3 * g + seed as usize) % 7) as f32 * 0.03);
            scale[col * ng + g] = f16_to_f32(f32_to_f16(raw));
            for i in 0..GROUP {
                let kk = g * GROUP + i;
                let mixed = (i as u64).wrapping_mul(0x9e37_79b1)
                    ^ (col as u64).wrapping_mul(0x85eb_ca77)
                    ^ (g as u64).wrapping_mul(0xc2b2_ae3d)
                    ^ seed.wrapping_mul(0x27d4_eb2f);
                q[kk * n + col] = (mixed % 15) as i8 - 7;
            }
        }
    }
    (q, scale)
}

/// NPU op4++ layout: per (col, group) a 130-byte block = fp16 scale + 128 nibbles
/// (low = inner 2j, high = inner 2j+1).
fn npu_payload(q: &[i8], scale: &[f32], k: usize, n: usize) -> Vec<u8> {
    let ng = k / GROUP;
    let mut payload = vec![0u8; n * ng * 130];
    for col in 0..n {
        for g in 0..ng {
            let block = &mut payload[(col * ng + g) * 130..(col * ng + g + 1) * 130];
            block[..2].copy_from_slice(&f32_to_f16(scale[col * ng + g]).to_le_bytes());
            for j in 0..128 {
                let lo = (q[(g * GROUP + 2 * j) * n + col] as u8) & 0x0f;
                let hi = (q[(g * GROUP + 2 * j + 1) * n + col] as u8) & 0x0f;
                block[2 + j] = lo | (hi << 4);
            }
        }
    }
    payload
}

/// GPU Oq4G256 combined buffer: [N rows × K/2 nibble bytes] ++ [f32 scales N*ng].
fn gpu_combined(q: &[i8], scale: &[f32], k: usize, n: usize) -> Vec<u8> {
    let ng = k / GROUP;
    let packed = n * (k / 2);
    let mut combined = vec![0u8; packed + n * ng * 4];
    for row in 0..n {
        for g in 0..ng {
            let dst = row * (k / 2) + g * (GROUP / 2);
            for j in 0..128 {
                let lo = (q[(g * GROUP + 2 * j) * n + row] as u8) & 0x0f;
                let hi = (q[(g * GROUP + 2 * j + 1) * n + row] as u8) & 0x0f;
                combined[dst + j] = lo | (hi << 4);
            }
            let so = packed + (row * ng + g) * 4;
            combined[so..so + 4].copy_from_slice(&scale[row * ng + g].to_le_bytes());
        }
    }
    combined
}

fn sqnr_db(reference: &[f32], got: &[f32]) -> f64 {
    let mut sig = 0.0f64;
    let mut noise = 0.0f64;
    for (&r, &g) in reference.iter().zip(got) {
        sig += (r as f64).powi(2);
        noise += ((r - g) as f64).powi(2);
    }
    10.0 * (sig / noise.max(1e-30)).log10()
}

fn main() {
    let mut a = std::env::args().skip(1);
    let k: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(768);
    let n: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(1152);
    assert_eq!(k % GROUP, 0, "K must be % 256");

    let (q, scale) = source_weight(k, n, 3);

    // Shared NPU codec (bit-exact same nibbles+scales as the GPU buffer → L1).
    let matrix =
        OpusPackedMatrix::from_payload(33, k, n, &npu_payload(&q, &scale, k, n), None).unwrap();

    let x = lcg(2, k);
    let y_a16 = matrix.reference_dequantized_bf16_f32(1, &x).unwrap(); // W4A16 oracle
    let y_a8 = matrix.reference_f32(1, &x).unwrap(); // W4A8 (NPU precision)

    let mut gpu = Gpu::init().unwrap();
    if !gpu.arch_caps.has_wmma_w32() {
        println!(
            "SKIP oq4_shared_codec_gpu_parity: {} lacks wave32 WMMA",
            gpu.arch
        );
        return;
    }
    let buf = gpu
        .upload_raw(
            &gpu_combined(&q, &scale, k, n),
            &[n * (k / 2) + n * (k / GROUP) * 4],
        )
        .unwrap();
    let wt = WeightTensor {
        buf,
        gpu_dtype: DType::Oq4G256,
        m: n,
        k,
        row_stride: 0,
        paro: None,
        awq_scale: None,
    };
    let xd = gpu
        .upload_raw(
            &x.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>(),
            &[k],
        )
        .unwrap();
    let yd = gpu.upload_raw(&vec![0u8; n * 4], &[n]).unwrap();
    weight_gemv(&mut gpu, &wt, &xd, &yd).unwrap();
    gpu.device_synchronize().unwrap();
    let y_gpu = gpu.download_f32(&yd).unwrap();

    // L3: GPU decode (W4A16) vs the same-math CPU oracle.
    let l3 = sqnr_db(&y_a16, &y_gpu);
    // Context: how far the GPU@A16 sits from the NPU's W4A8 precision.
    let a8_gap = sqnr_db(&y_a16, &y_a8);
    let pass = l3 > 30.0;
    println!(
        "oq4_shared_codec_gpu_parity K={k} N={n} on {}: L1=bit-exact(by-construction) L3(GPU@A16 vs oracle)={l3:.2} dB  A8-vs-A16-gap={a8_gap:.2} dB -> {}",
        gpu.arch,
        if pass { "PASS" } else { "FAIL" }
    );
    if !pass {
        std::process::exit(1);
    }
}
