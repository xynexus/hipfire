// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.
//! Parity for `gemm_oq4_grouped_wmma_lds` (LDS-staged optimization of the Opus
//! Quant W4A4 core) against the original zero-LDS `gemm_oq4_grouped_wmma`. The
//! LDS kernel keeps the SAME per-group f32 accumulation order (ascending g,
//! `(iacc*sw)*sx`), so it must be **BIT-EXACT** vs the original — max_abs == 0.
//! Also checks vs a CPU reference (loose tol) and the bf16-out variant.
//! Runs several shapes incl. M/B not multiples of the 64/128 block tile (bounds).
//!
//!   cargo run --release -p hipfire-rdna --example parity_gemm_oq4_grouped_wmma_lds

use hipfire_rdna::Gpu;

fn lcg(seed: u32, n: usize) -> Vec<f32> {
    let mut s = seed.max(1);
    (0..n)
        .map(|i| {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12345) & 0x7fff_ffff;
            let base = (s as f32 / 2_147_483_648.0) - 0.5;
            if i % 97 == 0 {
                base * 9.0
            } else {
                base
            }
        })
        .collect()
}

fn quant_i4(src: &[f32], rows: usize, k: usize, group: usize) -> (Vec<u8>, Vec<f32>, Vec<i8>) {
    let ng = k / group;
    let mut packed = vec![0u8; rows * (k / 2)];
    let mut scales = vec![0f32; rows * ng];
    let mut qvals = vec![0i8; rows * k];
    for r in 0..rows {
        for g in 0..ng {
            let g0 = g * group;
            let mut amax = 1e-12f32;
            for c in g0..g0 + group {
                amax = amax.max(src[r * k + c].abs());
            }
            let scale = amax / 7.0;
            scales[r * ng + g] = scale;
            for c in g0..g0 + group {
                let q = (src[r * k + c] / scale).round().clamp(-7.0, 7.0) as i8;
                qvals[r * k + c] = q;
            }
        }
        for j in (0..k).step_by(2) {
            let lo = (qvals[r * k + j] as u8) & 0xf;
            let hi = (qvals[r * k + j + 1] as u8) & 0xf;
            packed[r * (k / 2) + j / 2] = lo | (hi << 4);
        }
    }
    (packed, scales, qvals)
}

fn run_shape(gpu: &mut Gpu, m: usize, k: usize, b: usize, group: usize) -> bool {
    let ng = k / group;
    let w = lcg(1, m * k);
    let x = lcg(2, b * k);
    let (wp, ws, wq) = quant_i4(&w, m, k, group);
    let (xp, xs, xq) = quant_i4(&x, b, k, group);

    // CPU reference
    let mut yref = vec![0.0f32; b * m];
    for bi in 0..b {
        for mi in 0..m {
            let mut acc = 0.0f32;
            for g in 0..ng {
                let g0 = g * group;
                let mut isum = 0i32;
                for c in g0..g0 + group {
                    isum += wq[mi * k + c] as i32 * xq[bi * k + c] as i32;
                }
                acc += isum as f32 * ws[mi * ng + g] * xs[bi * ng + g];
            }
            yref[bi * m + mi] = acc;
        }
    }

    let wsb: Vec<u8> = ws.iter().flat_map(|v| v.to_le_bytes()).collect();
    let xsb: Vec<u8> = xs.iter().flat_map(|v| v.to_le_bytes()).collect();
    let wd = gpu.upload_raw(&wp, &[m, k / 2]).unwrap();
    let wsd = gpu.upload_raw(&wsb, &[m, ng]).unwrap();
    let xd = gpu.upload_raw(&xp, &[b, k / 2]).unwrap();
    let xsd = gpu.upload_raw(&xsb, &[b, ng]).unwrap();

    // reference kernel (zero-LDS original)
    let y_orig = gpu.upload_raw(&vec![0u8; b * m * 4], &[b, m]).unwrap();
    gpu.gemm_oq4_grouped_wmma(&wd, &wsd, &xd, &xsd, &y_orig, m, k, b, group)
        .unwrap();
    // new LDS kernel (f32 out)
    let y_lds = gpu.upload_raw(&vec![0u8; b * m * 4], &[b, m]).unwrap();
    gpu.gemm_oq4_grouped_wmma_lds(&wd, &wsd, &xd, &xsd, &y_lds, m, k, b, group)
        .unwrap();
    // new LDS kernel (bf16 out)
    let y_bf16 = gpu.upload_raw(&vec![0u8; b * m * 2], &[b * m * 2]).unwrap();
    gpu.gemm_oq4_grouped_wmma_lds_bf16out(&wd, &wsd, &xd, &xsd, &y_bf16, m, k, b, group)
        .unwrap();
    gpu.device_synchronize().unwrap();

    let yo = gpu.download_f32(&y_orig).unwrap();
    let yl = gpu.download_f32(&y_lds).unwrap();
    let ybf_raw = gpu.download_raw(&y_bf16, b * m * 2).unwrap();

    // LDS-f32 vs original: must be BIT-EXACT.
    let mut exact_max = 0.0f32;
    let mut max_mag = 0.0f32;
    for i in 0..b * m {
        exact_max = exact_max.max((yl[i] - yo[i]).abs());
        max_mag = max_mag.max(yref[i].abs());
    }
    // LDS-f32 vs CPU ref (loose — different summation on CPU).
    let mut ref_max = 0.0f32;
    for i in 0..b * m {
        ref_max = ref_max.max((yl[i] - yref[i]).abs());
    }
    // bf16 vs LDS-f32 (bf16 truncation ~ 2^-8 relative).
    let mut bf_rel = 0.0f32;
    for i in 0..b * m {
        let bits = u16::from_le_bytes([ybf_raw[i * 2], ybf_raw[i * 2 + 1]]);
        let v = f32::from_bits((bits as u32) << 16);
        let d = (v - yl[i]).abs();
        let denom = yl[i].abs().max(1e-6);
        bf_rel = bf_rel.max(d / denom);
    }

    let ref_tol = 1e-3 * max_mag.max(1.0);
    let exact_pass = exact_max == 0.0;
    let ref_pass = ref_max <= ref_tol;
    let bf_pass = bf_rel <= 0.02; // bf16 has ~8 mantissa bits
    let pass = exact_pass && ref_pass && bf_pass;
    println!(
        "  M={m:<5} K={k} B={b:<5} g={group}: LDS-vs-orig max_abs={exact_max:.6} [{}]  \
         LDS-vs-cpu={ref_max:.5}(tol {ref_tol:.5})[{}]  bf16-rel={bf_rel:.5}[{}]  -> {}",
        if exact_pass { "BIT-EXACT" } else { "DIFF!" },
        if ref_pass { "ok" } else { "FAIL" },
        if bf_pass { "ok" } else { "FAIL" },
        if pass { "PASS" } else { "FAIL" }
    );
    pass
}

fn main() {
    let mut gpu = Gpu::init().unwrap();
    if !gpu.arch_caps.has_wmma_w32() {
        println!("SKIP gemm_oq4_grouped_wmma_lds parity: {} lacks wave32 WMMA", gpu.arch);
        return;
    }
    let group = 256usize;
    println!("gemm_oq4_grouped_wmma_lds parity on {}", gpu.arch);
    let shapes = [
        // aligned to BM=64 / BN=128
        (128usize, 1024usize, 128usize),
        (1536, 1024, 512),
        (1024, 1024, 512),
        (3072, 1024, 512),
        // unaligned M (not %64) and B (not %128) — exercises bounds clamps
        (1000, 1024, 100),
        (17, 512, 5),
        (64, 768, 129),
    ];
    let mut all = true;
    for (m, k, b) in shapes {
        all &= run_shape(&mut gpu, m, k, b, group);
    }
    println!("{}", if all { "ALL PASS" } else { "SOME FAILED" });
    if !all {
        std::process::exit(1);
    }
}
