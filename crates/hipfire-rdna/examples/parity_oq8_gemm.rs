// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.
//! Parity for `gemm_oq8_grouped_wmma` (Opus Quant W8A8 core) against a CPU
//! reference of the exact grouped int8×int8 + per-group-scale math:
//!   Y[b,m] = Σ_g sw[m,g]·sx[b,g] · Σ_{k∈g} qw[m,k]·qx[b,k]
//! The inner int8 dot is exact i32 and the epilogue order matches the kernel, so
//! GPU and CPU agree to f32 rounding. This is rung 2 of the W8A8 test ladder —
//! the gate that, had its iu4 sibling existed, would have caught the Oq4
//! batched-prefill divergence.
//!
//!   cargo run --release -p hipfire-rdna --example parity_oq8_gemm [M K B]

use hipfire_rdna::Gpu;

fn lcg(seed: u32, n: usize) -> Vec<u8> {
    let mut s = seed.max(1);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12345) & 0x7fff_ffff;
            (s >> 13) as u8 // interpreted as signed int8 by both GPU and CPU
        })
        .collect()
}
fn lcg_scales(seed: u32, n: usize) -> Vec<f32> {
    let mut s = seed.max(1);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12345) & 0x7fff_ffff;
            0.001 + (s as f32 / 2_147_483_648.0) * 0.05
        })
        .collect()
}
fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// The M-slab split must be BIT-identical: it changes which launch computes a
/// row, never the row's arithmetic. Runs a tall-thin shape (the one the split
/// exists for) both ways and compares words. `HIPFIRE_OQ8_GEMM_SLAB=0` forces
/// the single launch, so this can only be checked from a fresh process per mode
/// — the flag is read once. Here we compare the CURRENT mode against a CPU
/// recomputation of a few rows instead, which catches an off-by-one in `m_base`
/// (the failure the split can actually introduce).
fn check_slab_row_placement(gpu: &mut hipfire_rdna::Gpu) {
    use hipfire_rdna::DType;
    const M: usize = 17408; // > SLAB_TARGET, so the split is live
    const K: usize = 512;
    const G: usize = 256;
    const B: usize = 3;
    let ng = K / G;
    // Weight row m is all zeros except a single +1 at column (m % K); activation
    // row b is all zeros except +1 at column (b * 97 % K). Then Y[b, m] is
    // nonzero exactly when those columns coincide — a placement test that a
    // wrong m_base cannot pass by luck.
    let mut w = vec![0i8; M * K];
    for m in 0..M {
        w[m * K + (m % K)] = 1;
    }
    let mut x = vec![0i8; B * K];
    for b in 0..B {
        x[b * K + (b * 97 % K)] = 1;
    }
    let wb: Vec<u8> = w.iter().map(|&v| v as u8).collect();
    let xb: Vec<u8> = x.iter().map(|&v| v as u8).collect();
    let wt = gpu.upload_raw(&wb, &[M * K]).unwrap();
    let xt = gpu.upload_raw(&xb, &[B * K]).unwrap();
    let ws = gpu.upload_f32(&vec![1.0f32; M * ng], &[M * ng]).unwrap();
    let xs = gpu.upload_f32(&vec![1.0f32; B * ng], &[B * ng]).unwrap();
    let y = gpu.alloc_tensor(&[B * M], DType::F32).unwrap();
    gpu.gemm_oq8_grouped_wmma(&wt, &ws, &xt, &xs, &y, M, K, B, G)
        .unwrap();
    let got = gpu.download_f32(&y).unwrap();
    let mut bad = 0usize;
    for b in 0..B {
        for m in 0..M {
            let want = if m % K == b * 97 % K { 1.0f32 } else { 0.0 };
            if (got[b * M + m] - want).abs() > 1e-6 {
                bad += 1;
            }
        }
    }
    if bad == 0 {
        println!(
            "M-slab row placement M={M} (split into slabs): {} words exact -> PASS",
            B * M
        );
    } else {
        println!(
            "M-slab row placement: {bad} of {} words WRONG -> FAIL",
            B * M
        );
        std::process::exit(1);
    }
}

fn main() {
    let mut a = std::env::args().skip(1);
    let m: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(256);
    let k: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(1024);
    let b: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(48);
    let group = 256usize;
    assert_eq!(k % group, 0, "K must be a multiple of {group}");
    let ng = k / group;

    let mut gpu = Gpu::init().unwrap();
    if !gpu.arch_caps.has_wmma_w32() {
        println!("SKIP parity_oq8_gemm: {} lacks wave32 WMMA", gpu.arch);
        return;
    }

    // Inputs: int8 weights/acts (bytes reinterpreted as i8), f32 per-group scales.
    let qw = lcg(1, m * k);
    let sw = lcg_scales(2, m * ng);
    let qx = lcg(3, b * k);
    let sx = lcg_scales(4, b * ng);

    // CPU reference — exact grouped int8 dot, f32 epilogue in the kernel's order.
    let mut y_ref = vec![0.0f32; b * m];
    for bb in 0..b {
        for mm in 0..m {
            let mut acc = 0.0f32;
            for g in 0..ng {
                let mut idot: i32 = 0;
                for kk in 0..group {
                    let idx = g * group + kk;
                    idot += (qw[mm * k + idx] as i8 as i32) * (qx[bb * k + idx] as i8 as i32);
                }
                acc += (idot as f32) * sw[mm * ng + g] * sx[bb * ng + g];
            }
            y_ref[bb * m + mm] = acc;
        }
    }

    // GPU: combined weight buffer [int8 M*K | f32 scales M*ng]; scale view via sub_offset.
    let mut w_combined = qw.clone();
    w_combined.extend_from_slice(&f32_bytes(&sw));
    let wd = gpu.upload_raw(&w_combined, &[w_combined.len()]).unwrap();
    let wsd = wd.sub_offset(m * k, m * ng * 4);
    let xqd = gpu.upload_raw(&qx, &[b, k]).unwrap();
    let xsd = gpu.upload_raw(&f32_bytes(&sx), &[b, ng]).unwrap();
    let yd = gpu.upload_raw(&vec![0u8; b * m * 4], &[b, m]).unwrap();

    gpu.gemm_oq8_grouped_wmma(&wd, &wsd, &xqd, &xsd, &yd, m, k, b, group)
        .unwrap();
    gpu.device_synchronize().unwrap();
    let y_gpu = gpu.download_f32(&yd).unwrap();

    let (mut max_abs, mut max_rel, mut ref_mag) = (0.0f32, 0.0f32, 0.0f32);
    for (&g, &r) in y_gpu.iter().zip(&y_ref) {
        let d = (g - r).abs();
        max_abs = max_abs.max(d);
        ref_mag = ref_mag.max(r.abs());
        if r.abs() > 1e-6 {
            max_rel = max_rel.max(d / r.abs());
        }
    }
    // int8 dot is exact; only the f32 rescale/accumulate can differ by rounding.
    let pass = max_rel < 1e-4 || max_abs < ref_mag * 1e-4;
    println!(
        "parity_oq8_gemm M={m} K={k} B={b} on {}: max|Δ|={max_abs:.3e} (ref|max|={ref_mag:.3e}) \
         max_rel={max_rel:.3e} -> {}",
        gpu.arch,
        if pass { "PASS" } else { "FAIL" }
    );

    // quantize_act_oq8 round-trip: quantize a random f32 activation, dequant
    // (scale·int8 per group), and check SQNR — int8 acts must be near-lossless.
    let mut s = 7u32;
    let xf: Vec<f32> = (0..b * k)
        .map(|_| {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12345) & 0x7fff_ffff;
            ((s as f32 / 2_147_483_648.0) - 0.5) * 6.0 // ~[-3, 3]
        })
        .collect();
    let xfd = gpu.upload_f32(&xf, &[b, k]).unwrap();
    let xq8 = gpu.upload_raw(&vec![0u8; b * k], &[b, k]).unwrap();
    let xs8 = gpu.upload_raw(&vec![0u8; b * ng * 4], &[b, ng]).unwrap();
    gpu.quantize_act_oq8(&xfd, &xq8, &xs8, b, k, group).unwrap();
    gpu.device_synchronize().unwrap();
    let q = gpu.download_raw(&xq8, b * k).unwrap();
    let qs = gpu.download_f32(&xs8).unwrap();
    let (mut sig, mut noise) = (0.0f64, 0.0f64);
    for bb in 0..b {
        for g in 0..ng {
            let sc = qs[bb * ng + g];
            for kk in 0..group {
                let idx = bb * k + g * group + kk;
                let rec = (q[idx] as i8 as f32) * sc;
                sig += (xf[idx] as f64).powi(2);
                noise += ((xf[idx] - rec) as f64).powi(2);
            }
        }
    }
    let act_sqnr = 10.0 * (sig / noise.max(1e-30)).log10();
    let act_pass = act_sqnr > 40.0; // int8 dynamic-range ceiling ~48 dB
    println!(
        "quantize_act_oq8 round-trip: SQNR={act_sqnr:.2} dB -> {}",
        if act_pass { "PASS" } else { "FAIL" }
    );

    // QUALITY vs full precision: run the REAL W8A8 GPU path (int8-quantize both
    // operands, grouped-iu8 GEMM) and compare to the TRUE f32 matmul. This is the
    // near-lossless question — measured against f32, not a lower-bit quant. (No
    // FWHT here, so this is a conservative floor; the rotation only tightens int8
    // quant. The definitive model-level number is rung 3: KLD vs bf16.)
    let mut s2 = 99u32;
    let mut rndf = |scale: f32| {
        s2 = s2.wrapping_mul(1_103_515_245).wrapping_add(12345) & 0x7fff_ffff;
        ((s2 as f32 / 2_147_483_648.0) - 0.5) * scale
    };
    let wf: Vec<f32> = (0..m * k).map(|_| rndf(2.0)).collect();
    let xf2: Vec<f32> = (0..b * k).map(|_| rndf(2.0)).collect();
    // True f32 matmul reference (the bf16/f32 baseline).
    let mut y_true = vec![0.0f32; b * m];
    for bb in 0..b {
        for mm in 0..m {
            let mut acc = 0.0f32;
            for kk in 0..k {
                acc += xf2[bb * k + kk] * wf[mm * k + kk];
            }
            y_true[bb * m + mm] = acc;
        }
    }
    // Weight quant: per-group symmetric int8 (absmax/127), the codec's RTN path.
    let mut qwf = vec![0u8; m * k];
    let mut swf = vec![0.0f32; m * ng];
    for mm in 0..m {
        for g in 0..ng {
            let mut amax = 0.0f32;
            for kk in 0..group {
                amax = amax.max(wf[mm * k + g * group + kk].abs());
            }
            let sc = if amax > 0.0 { amax / 127.0 } else { 1.0 };
            swf[mm * ng + g] = sc;
            let inv = if amax > 0.0 { 127.0 / amax } else { 0.0 };
            for kk in 0..group {
                let idx = mm * k + g * group + kk;
                qwf[idx] = ((wf[idx] * inv).round().clamp(-127.0, 127.0) as i8) as u8;
            }
        }
    }
    let mut wq_combined = qwf.clone();
    wq_combined.extend_from_slice(&f32_bytes(&swf));
    let wqd = gpu.upload_raw(&wq_combined, &[wq_combined.len()]).unwrap();
    let wqs = wqd.sub_offset(m * k, m * ng * 4);
    // Activation quant via the real GPU kernel.
    let xf2d = gpu.upload_f32(&xf2, &[b, k]).unwrap();
    let xq2 = gpu.upload_raw(&vec![0u8; b * k], &[b, k]).unwrap();
    let xs2 = gpu.upload_raw(&vec![0u8; b * ng * 4], &[b, ng]).unwrap();
    gpu.quantize_act_oq8(&xf2d, &xq2, &xs2, b, k, group)
        .unwrap();
    let yq = gpu.upload_raw(&vec![0u8; b * m * 4], &[b, m]).unwrap();
    gpu.gemm_oq8_grouped_wmma(&wqd, &wqs, &xq2, &xs2, &yq, m, k, b, group)
        .unwrap();
    gpu.device_synchronize().unwrap();
    let y_q = gpu.download_f32(&yq).unwrap();
    let (mut qsig, mut qnoise) = (0.0f64, 0.0f64);
    for (&t, &q) in y_true.iter().zip(&y_q) {
        qsig += (t as f64).powi(2);
        qnoise += ((t - q) as f64).powi(2);
    }
    let gemm_sqnr = 10.0 * (qsig / qnoise.max(1e-30)).log10();
    let gemm_pass = gemm_sqnr > 35.0;
    println!(
        "W8A8 GEMM vs f32 matmul (real kernels, no FWHT floor): SQNR={gemm_sqnr:.2} dB -> {}",
        if gemm_pass { "PASS" } else { "FAIL" }
    );

    if !pass || !act_pass || !gemm_pass {
        std::process::exit(1);
    }
    check_slab_row_placement(&mut gpu);
}
