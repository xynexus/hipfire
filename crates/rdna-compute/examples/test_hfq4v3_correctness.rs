//! gfx11 (RDNA3) HFQ4v3 K=64 iu8 MMQ residual GEMM correctness test.
//!
//! Compares the new HFQ4v3 GEMM kernel
//! (`gemm_hfq4v3_residual_iu8_mmq.gfx11.hip`) against the existing FP16
//! dequant→WMMA reference (`gemm_hfq4g256_residual_wmma`). Both kernels
//! consume the SAME conceptual weight values — we synthesize one HFQ4-G256
//! weight buffer, then re-quantize it as HFQ4v3 with K=64 grouping +
//! FP16 (d, m). The HFQ4v3 path also pre-quantizes activations via the
//! shared Q8_1 MMQ quantizer.
//!
//! Expected error sources:
//!   - Q8_1 activation quant: signed INT8 in [-127, 127] = ~0.8% precision
//!     per element. Already validated against MMQ at ~0.05 max-abs / ~0.005
//!     mean-abs on similar shapes.
//!   - K=256 → K=64 regrouping: HFQ4v3 has 4× sharper per-group quant
//!     resolution than HFQ4-G256, so dequantization noise REDUCES vs the
//!     reference. The error here is dominated by the activation Q8_1
//!     quant, not weights.
//!   - FP16 (d, m) round-trip: ~0.05% per group. Negligible.
//!
//! PASS thresholds:
//!   max abs err        < 0.05   (matches the v1 MMQ correctness gate)
//!   mean abs err       < 0.005
//!   mean rel err†      < 0.02
//!   max rel err†       < 0.5
//!   pct rel-err > 5%†  < 10%
//!
//! † Rel-err only on elements where |fp16_output| > REL_FLOOR (0.1).
//!
//! Run on gfx1100/gfx1101/gfx1102/gfx1150/gfx1151:
//!   cargo run --release -p rdna-compute --example test_hfq4v3_correctness

use rdna_compute::{DType, Gpu};

fn main() {
    let m: usize = std::env::var("M").ok().and_then(|s| s.parse().ok()).unwrap_or(256);
    let k: usize = std::env::var("K").ok().and_then(|s| s.parse().ok()).unwrap_or(512);
    let n: usize = std::env::var("N").ok().and_then(|s| s.parse().ok()).unwrap_or(128);

    assert!(k % 256 == 0);
    assert!(m % 16 == 0);
    assert!(n % 16 == 0);

    let mut gpu = Gpu::init().expect("gpu init");
    let arch = gpu.arch.clone();
    eprintln!("GPU: {arch}");

    let supported = matches!(
        arch.as_str(),
        "gfx1100" | "gfx1101" | "gfx1102" | "gfx1103" | "gfx1150" | "gfx1151"
    );
    if !supported {
        eprintln!("SKIP: requires gfx11xx (RDNA3/3.5). Current: {arch}");
        std::process::exit(0);
    }

    eprintln!("=== gfx11 HFQ4v3 K=64 iu8 MMQ vs FP16-WMMA correctness test ===");
    eprintln!("M={m}, K={k}, N={n}");

    let groups256 = k / 256;
    let row_bytes_v1 = groups256 * 136;

    // Synth HFQ4-G256 weights (the same generator as test_gemm_hfq4g256_residual).
    let weight_v1: Vec<u8> = synth_hfq4g256_weights(m, groups256, 0xC0DE_FACEu64);
    let a_v1 = gpu.upload_raw(&weight_v1, &[m * row_bytes_v1]).expect("upload v1 weights");

    // CPU regroup: HFQ4-G256 → HFQ4v3 K=64 + FP16 (d, m).
    let weight_v3: Vec<u8> = regroup_hfq4g256_to_hfq4v3(&weight_v1, m, k);
    let row_bytes_v3 = (k / 64) * 36;
    assert_eq!(weight_v3.len(), m * row_bytes_v3);
    // Tag dtype so the dispatch-layer short-circuit can be exercised below.
    let a_v3 = gpu.upload_raw_with_dtype(&weight_v3, &[m * row_bytes_v3], DType::HFQ4V3G64)
        .expect("upload v3 weights");

    let x_host: Vec<f32> = (0..n * k)
        .map(|i| {
            let v = ((i as i64).wrapping_mul(1103515245).wrapping_add(12345)) as f32;
            (v * 1e-9) % 2.0 - 1.0
        })
        .collect();
    let y_init_host: Vec<f32> = (0..n * m)
        .map(|i| {
            let v = ((i as i64).wrapping_mul(2147483647).wrapping_add(7)) as f32;
            (v * 1e-7) % 1.0
        })
        .collect();

    let x_gpu = gpu.alloc_tensor(&[n * k], DType::F32).expect("alloc x");
    let y_fp16 = gpu.alloc_tensor(&[n * m], DType::F32).expect("alloc y_fp16");
    let y_v3 = gpu.alloc_tensor(&[n * m], DType::F32).expect("alloc y_v3");

    gpu.hip.memcpy_htod(&x_gpu.buf, bytes_of(&x_host)).unwrap();

    // Path A: FP16 WMMA reference (HFQ4-G256, dequant → fp16 wmma).
    gpu.hip.memcpy_htod(&y_fp16.buf, bytes_of(&y_init_host)).unwrap();
    gpu.hip.device_synchronize().unwrap();
    gpu.gemm_hfq4g256_residual_wmma(&a_v1, &x_gpu, &y_fp16, m, k, n)
        .expect("fp16 wmma reference");
    gpu.hip.device_synchronize().unwrap();
    let y_fp16_host: Vec<f32> = gpu.download_f32(&y_fp16).expect("download y_fp16");

    // Path B: HFQ4v3 K=64 iu8 MMQ via the high-level dispatch entry. The
    // dispatcher inspects `a_v3.dtype == HFQ4V3G64` and short-circuits to
    // `gemm_hfq4v3_residual_iu8_mmq_gfx11`. Calling through the public
    // entrypoint instead of the v3-specific fn directly verifies the
    // routing wiring along with the kernel correctness.
    gpu.hip.memcpy_htod(&y_v3.buf, bytes_of(&y_init_host)).unwrap();
    gpu.hip.device_synchronize().unwrap();
    gpu.gemm_hfq4g256_residual(&a_v3, &x_gpu, &y_v3, m, k, n)
        .expect("hfq4v3 via dispatch routing");
    gpu.hip.device_synchronize().unwrap();
    let y_v3_host: Vec<f32> = gpu.download_f32(&y_v3).expect("download y_v3");

    assert_eq!(y_fp16_host.len(), n * m);
    assert_eq!(y_v3_host.len(), n * m);

    const REL_FLOOR: f32 = 0.1;
    let mut max_abs_err: f32 = 0.0;
    let mut max_rel_err: f32 = 0.0;
    let mut sum_abs_err: f64 = 0.0;
    let mut sum_rel_err: f64 = 0.0;
    let mut max_loc: (usize, usize) = (0, 0);
    let mut samples_above_5pct: usize = 0;
    let mut rel_eligible: usize = 0;

    for col in 0..n {
        for row in 0..m {
            let idx = col * m + row;
            let a = y_fp16_host[idx];
            let b = y_v3_host[idx];
            let err = (a - b).abs();
            if err > max_abs_err {
                max_abs_err = err;
                max_loc = (col, row);
            }
            sum_abs_err += err as f64;
            if a.abs() > REL_FLOOR {
                let rel = err / a.abs();
                if rel > max_rel_err { max_rel_err = rel; }
                sum_rel_err += rel as f64;
                rel_eligible += 1;
                if rel > 0.05 { samples_above_5pct += 1; }
            }
        }
    }

    let total = (n * m) as f64;
    let mean_abs_err = sum_abs_err / total;
    let mean_rel_err = if rel_eligible > 0 { sum_rel_err / (rel_eligible as f64) } else { 0.0 };
    let pct_above = 100.0 * samples_above_5pct as f32 / rel_eligible.max(1) as f32;

    eprintln!("\n--- per-element error (n*m = {} elements) ---", n * m);
    eprintln!("  max abs err:             {:.6}  at (col={}, row={})",
              max_abs_err, max_loc.0, max_loc.1);
    eprintln!("  mean abs err:            {:.6}", mean_abs_err);
    eprintln!("  rel-err eligible (|out| > {:.2}): {} / {} ({:.1}%)",
              REL_FLOOR, rel_eligible, n * m,
              100.0 * rel_eligible as f32 / (n * m) as f32);
    eprintln!("  max rel err†:            {:.4}", max_rel_err);
    eprintln!("  mean rel err†:           {:.4}", mean_rel_err);
    eprintln!("  samples > 5% rel†:       {} / {} ({:.3}%)",
              samples_above_5pct, rel_eligible.max(1), pct_above);
    eprintln!("  † counted only on non-near-zero outputs (|out| > {REL_FLOOR})");

    eprintln!("\n--- sample triples (col=0..2, row=0..4) ---");
    for col in 0..2.min(n) {
        for row in 0..4.min(m) {
            let idx = col * m + row;
            let a = y_fp16_host[idx];
            let b = y_v3_host[idx];
            eprintln!("  col={col} row={row}: fp16={a:>10.4}  v3={b:>10.4}  err={:.4}",
                      (a - b).abs());
        }
    }

    let max_abs_thresh = 0.05;
    let mean_abs_thresh = 0.005;
    let max_rel_thresh = 0.5;
    let mean_rel_thresh = 0.02;
    let pct_thresh = 10.0;

    let max_abs_ok = max_abs_err < max_abs_thresh;
    let mean_abs_ok = (mean_abs_err as f32) < mean_abs_thresh;
    let max_rel_ok = max_rel_err < max_rel_thresh;
    let mean_rel_ok = (mean_rel_err as f32) < mean_rel_thresh;
    let pct_ok = pct_above < pct_thresh;

    eprintln!("\n--- PASS criteria (Q8_1 activation quant + FP16 d/m) ---");
    eprintln!("  max abs err   < {max_abs_thresh}:   {}", if max_abs_ok { "OK" } else { "FAIL" });
    eprintln!("  mean abs err  < {mean_abs_thresh}: {}", if mean_abs_ok { "OK" } else { "FAIL" });
    eprintln!("  max rel err†  < {max_rel_thresh}:   {}", if max_rel_ok { "OK" } else { "FAIL" });
    eprintln!("  mean rel err† < {mean_rel_thresh}: {}", if mean_rel_ok { "OK" } else { "FAIL" });
    eprintln!("  pct >5% rel†  < {pct_thresh}%:    {}", if pct_ok { "OK" } else { "FAIL" });

    if max_abs_ok && mean_abs_ok && max_rel_ok && mean_rel_ok && pct_ok {
        eprintln!("\nPASS: HFQ4v3 K=64 iu8 MMQ is numerically equivalent to FP16 WMMA \
                   reference within Q8_1 tolerance.");
        std::process::exit(0);
    } else {
        eprintln!("\nFAIL: HFQ4v3 K=64 iu8 MMQ diverges beyond Q8_1 tolerance.");
        std::process::exit(1);
    }
}

fn synth_hfq4g256_weights(m: usize, groups_per_row: usize, seed: u64) -> Vec<u8> {
    let total = m * groups_per_row * 136;
    let mut out = vec![0u8; total];
    let mut state = seed;
    let mut next = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (state >> 33) as u32
    };
    for row in 0..m {
        for g in 0..groups_per_row {
            let gp = (row * groups_per_row + g) * 136;
            let scale_exp: u32 = 0x3a + (next() & 0x7);
            let scale_bits = (scale_exp << 23) | (next() & 0x007F_FFFF);
            let zp_bits = ((next() & 0x80) << 24) | 0x39000000u32 | (next() & 0x007F_FFFF);
            let scale = f32::from_bits(scale_bits);
            let zp = f32::from_bits(zp_bits);
            let scale_ok = if scale.is_finite() && scale.abs() < 1e-2 && scale > 0.0 { scale } else { 1e-3 };
            let zp_ok = if zp.is_finite() && zp.abs() < 1.0 { zp } else { -0.5 };
            out[gp..gp + 4].copy_from_slice(&scale_ok.to_le_bytes());
            out[gp + 4..gp + 8].copy_from_slice(&zp_ok.to_le_bytes());
            for i in 0..128 {
                out[gp + 8 + i] = (next() & 0xFF) as u8;
            }
        }
    }
    out
}

/// Dequantize HFQ4-G256 row to f32, then re-quantize as HFQ4v3 (K=64
/// groups, FP16 d + FP16 m + 32 B nibbles). Same flow as the offline
/// converter but operating in-process so the test is self-contained.
fn regroup_hfq4g256_to_hfq4v3(weight_v1: &[u8], m: usize, k: usize) -> Vec<u8> {
    assert!(k % 256 == 0);
    let groups256 = k / 256;
    let row_bytes_v1 = groups256 * 136;
    let row_bytes_v3 = (k / 64) * 36;
    let mut out = vec![0u8; m * row_bytes_v3];

    for row in 0..m {
        let row_in = &weight_v1[row * row_bytes_v1..(row + 1) * row_bytes_v1];

        // Dequant the full row to f32.
        let mut f32_row = vec![0.0f32; k];
        for g in 0..groups256 {
            let off = g * 136;
            let scale = f32::from_le_bytes(row_in[off..off + 4].try_into().unwrap());
            let zp = f32::from_le_bytes(row_in[off + 4..off + 8].try_into().unwrap());
            for i in 0..128 {
                let byte = row_in[off + 8 + i];
                let lo = (byte & 0xF) as f32;
                let hi = (byte >> 4) as f32;
                f32_row[g * 256 + 2 * i + 0] = lo * scale + zp;
                f32_row[g * 256 + 2 * i + 1] = hi * scale + zp;
            }
        }

        // Re-quantize as HFQ4v3 K=64 + FP16 (d, m).
        let row_out = &mut out[row * row_bytes_v3..(row + 1) * row_bytes_v3];
        let groups64 = k / 64;
        for g in 0..groups64 {
            let mut min_v = f32::INFINITY;
            let mut max_v = f32::NEG_INFINITY;
            for i in 0..64 {
                let v = f32_row[g * 64 + i];
                if v < min_v { min_v = v; }
                if v > max_v { max_v = v; }
            }
            let range = max_v - min_v;
            let d_f32 = if range > 0.0 { range / 15.0 } else { 1.0 };
            let m_f32 = min_v;
            let d_h = f32_to_f16(d_f32);
            let m_h = f32_to_f16(m_f32);
            let d_round = f16_to_f32(d_h);
            let m_round = f16_to_f32(m_h);
            let inv = if d_round > 0.0 { 1.0 / d_round } else { 0.0 };

            let off = g * 36;
            row_out[off + 0] = (d_h & 0xFF) as u8;
            row_out[off + 1] = (d_h >> 8) as u8;
            row_out[off + 2] = (m_h & 0xFF) as u8;
            row_out[off + 3] = (m_h >> 8) as u8;

            for i in 0..32 {
                let v_lo = f32_row[g * 64 + 2 * i + 0];
                let v_hi = f32_row[g * 64 + 2 * i + 1];
                let q_lo = ((v_lo - m_round) * inv + 0.5).clamp(0.0, 15.0) as u8;
                let q_hi = ((v_hi - m_round) * inv + 0.5).clamp(0.0, 15.0) as u8;
                row_out[off + 4 + i] = q_lo | (q_hi << 4);
            }
        }
    }

    out
}

fn f32_to_f16(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = (bits >> 31) & 1;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let frac = bits & 0x7FFFFF;
    if exp == 0xFF {
        let f16_frac = if frac == 0 { 0 } else { (frac >> 13) | 1 };
        return ((sign << 15) | (0x1F << 10) | f16_frac) as u16;
    }
    let exp_unbiased = exp - 127;
    if exp_unbiased < -24 {
        return (sign << 15) as u16;
    }
    if exp_unbiased < -14 {
        let shift = -14 - exp_unbiased;
        let mantissa = (frac | 0x800000) >> (13 + shift);
        let round_bit = ((frac | 0x800000) >> (12 + shift)) & 1;
        let m16 = mantissa + round_bit;
        return ((sign << 15) | m16) as u16;
    }
    if exp_unbiased > 15 {
        return ((sign << 15) | (0x1F << 10)) as u16;
    }
    let exp16 = (exp_unbiased + 15) as u32;
    let frac16 = frac >> 13;
    let round_bit = (frac >> 12) & 1;
    let f16_low = (sign << 15) | (exp16 << 10) | frac16;
    (f16_low + round_bit) as u16
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let frac = (bits & 0x3FF) as u32;
    if exp == 0 {
        if frac == 0 {
            return f32::from_bits(sign << 31);
        }
        let mut e = 0i32;
        let mut f = frac;
        while f & 0x400 == 0 { f <<= 1; e -= 1; }
        f &= 0x3FF;
        let exp32 = (127 - 15 + 1 + e) as u32;
        return f32::from_bits((sign << 31) | (exp32 << 23) | (f << 13));
    }
    if exp == 31 {
        let frac32 = if frac == 0 { 0 } else { (frac << 13) | 1 };
        return f32::from_bits((sign << 31) | (0xFF << 23) | frac32);
    }
    let exp32 = exp + 127 - 15;
    f32::from_bits((sign << 31) | (exp32 << 23) | (frac << 13))
}

fn bytes_of(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}
