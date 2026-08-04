// SPDX-License-Identifier: Apache-2.0
//! Phase 0 root-cause: which rmsnorm+FWHT rotation path produces the W4A4
//! batched-prefill divergence? The int4-act GEMM is already proven batch-exact
//! (parity_oq4_batched_vs_pertoken), so the e2e drift must be UPSTREAM in the
//! rotated activation. This probe feeds the SAME f32 x [N×K] through three
//! rmsnorm+rotate paths, int4-quantizes each (quantize_act_oq4), and compares
//! the resulting nibbles + dequant values:
//!
//!   A. fused_rmsnorm_rotate_mq_batched(x)           — the batched-prefill path
//!   B. fused_rmsnorm_rotate_mq(x[r]) per row        — same kernel, grid=1
//!   C. rmsnorm_f32(x[r]) then rotate_x_mq per row   — the per-token/decode path
//!
//! A-vs-B should be BIT-EXACT (identical kernel, only grid.x differs).
//! A-vs-C is the hypothesis under test: on gfx1103 the standalone rmsnorm is
//! wave-reduced (rmsnorm_f32_gfx1103, 8 slots) while the fused kernel uses a
//! 256-slot block reduction — a different f32 sum-of-squares rounding that the
//! int4 activation step (a nonlinearity) amplifies into flipped nibbles.

use hipfire_rdna::{DType, Gpu};

fn lcgf(seed: u32, n: usize) -> Vec<f32> {
    let mut s = seed.max(1);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12345) & 0x7fff_ffff;
            ((s as f32 / 2_147_483_648.0) - 0.5) * 2.0
        })
        .collect()
}

fn main() {
    let k = 1024usize;
    let group = 256usize;
    let ng = k / group;
    let n = 8usize;
    let eps = 1e-6f32;

    let mut gpu = Gpu::init().unwrap();
    if !gpu.arch_caps.has_wmma_w32() {
        println!("SKIP: no wmma");
        return;
    }

    // rmsnorm gamma weight [K] (non-trivial, like a real norm scale) and x [N×K].
    let gamma: Vec<f32> = lcgf(3, k).iter().map(|v| 1.0 + 0.25 * v).collect();
    let wg = gpu.upload_f32(&gamma, &[k]).unwrap();
    let xv = lcgf(1234, n * k);
    let xb = gpu.upload_f32(&xv, &[n, k]).unwrap();

    // Helper: int4-quantize a [rows×K] f32 rotated activation, return (nibbles, scales).
    let quantize = |gpu: &mut Gpu, xr: &hipfire_rdna::GpuTensor, rows: usize| {
        let q = gpu.alloc_tensor(&[rows * (k / 2)], DType::Raw).unwrap();
        let s = gpu.alloc_tensor(&[rows * ng], DType::F32).unwrap();
        gpu.quantize_act_oq4(xr, &q, &s, rows, k, group).unwrap();
        gpu.device_synchronize().unwrap();
        (
            gpu.download_raw(&q, rows * (k / 2)).unwrap(),
            gpu.download_f32(&s).unwrap(),
        )
    };

    // --- Path A: batched fused rmsnorm+rotate ---
    let xra = gpu.alloc_tensor(&[n * k], DType::F32).unwrap();
    gpu.fused_rmsnorm_rotate_mq_batched(&xb, &wg, &xra, k, eps, n)
        .unwrap();
    gpu.device_synchronize().unwrap();
    let xra_f = gpu.download_f32(&xra).unwrap();
    let (qa, sa) = quantize(&mut gpu, &xra, n);

    // --- Path B: per-row fused rmsnorm+rotate (same kernel, grid=1) ---
    let mut xrb_f = vec![0f32; n * k];
    for r in 0..n {
        let xr = xb.sub_offset(r * k, k);
        let out = gpu.alloc_tensor(&[k], DType::F32).unwrap();
        gpu.fused_rmsnorm_rotate_mq(&xr, &wg, &out, k, eps).unwrap();
        gpu.device_synchronize().unwrap();
        xrb_f[r * k..(r + 1) * k].copy_from_slice(&gpu.download_f32(&out).unwrap());
    }
    let xrb = gpu.upload_f32(&xrb_f, &[n, k]).unwrap();
    let (qb, sb) = quantize(&mut gpu, &xrb, n);

    // --- Path C: standalone rmsnorm then rotate_x_mq (the per-token/decode path) ---
    let mut xrc_f = vec![0f32; n * k];
    for r in 0..n {
        let xr = xb.sub_offset(r * k, k);
        let normed = gpu.alloc_tensor(&[k], DType::F32).unwrap();
        gpu.rmsnorm_f32(&xr, &wg, &normed, eps).unwrap();
        let out = gpu.alloc_tensor(&[k], DType::F32).unwrap();
        gpu.rotate_x_mq(&normed, &out, k).unwrap();
        gpu.device_synchronize().unwrap();
        xrc_f[r * k..(r + 1) * k].copy_from_slice(&gpu.download_f32(&out).unwrap());
    }
    let xrc = gpu.upload_f32(&xrc_f, &[n, k]).unwrap();
    let (qc, sc) = quantize(&mut gpu, &xrc, n);

    // Compare two paths: pre-quant rotated-activation max_abs, nibble mismatch %,
    // and max scale rel diff.
    let cmp = |label: &str, f1: &[f32], f2: &[f32], q1: &[u8], q2: &[u8], s1: &[f32], s2: &[f32]| {
        let mut mx = 0f32;
        for i in 0..f1.len() {
            mx = mx.max((f1[i] - f2[i]).abs());
        }
        let mut nib_mm = 0usize;
        for i in 0..q1.len() {
            if q1[i] != q2[i] {
                // count both nibbles that differ in this byte
                if (q1[i] & 0x0f) != (q2[i] & 0x0f) {
                    nib_mm += 1;
                }
                if (q1[i] >> 4) != (q2[i] >> 4) {
                    nib_mm += 1;
                }
            }
        }
        let total_nib = q1.len() * 2;
        let mut smx = 0f32;
        for i in 0..s1.len() {
            let d = (s1[i] - s2[i]).abs();
            let r = if s2[i].abs() > 0.0 { d / s2[i].abs() } else { d };
            smx = smx.max(r);
        }
        println!(
            "{label:<28} xrot max_abs={mx:.3e}  nib_mismatch={nib_mm}/{total_nib} ({:.3e})  scale_rel={smx:.3e}  -> {}",
            nib_mm as f32 / total_nib as f32,
            if nib_mm == 0 && mx == 0.0 { "BIT-EXACT" } else if nib_mm == 0 { "quant-exact" } else { "DIFFERS" }
        );
    };

    println!("oq4 rotation-path parity  N={n} K={k} group={group} arch={}", gpu.arch);
    cmp("A(fused-batched) vs B(fused-1)", &xra_f, &xrb_f, &qa, &qb, &sa, &sb);
    cmp("A(fused-batched) vs C(standalone)", &xra_f, &xrc_f, &qa, &qc, &sa, &sc);
    cmp("B(fused-1) vs C(standalone)", &xrb_f, &xrc_f, &qb, &qc, &sb, &sc);
}
