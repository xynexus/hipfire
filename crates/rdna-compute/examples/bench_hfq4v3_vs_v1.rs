//! Microbench: HFQ4v3 K=64 iu8 MMQ vs HFQ4-G256 v1 paths.
//!
//! Measures `gemm_hfq4g256_residual` (auto-routes to FP16 WMMA / MMQ /
//! v3 based on weight dtype) against the new HFQ4v3 path at the same
//! GEMM shapes as Qwen3.5-9B prefill. Reports tok/s-equivalent
//! throughput so the result is directly comparable to bench_qwen35_mq4
//! prefill numbers.
//!
//! The key bench shapes for residual / FFN-down on Qwen3.5 9B:
//!   - wo_residual: M=4096, K=4096
//!   - w_down:      M=4096, K=12288
//! For 4B and 27B, swap M/K to the corresponding hidden dims.
//!
//! Prefill batch sizes covered: 32 (small), 128 (mid), 256 (MMQ-eligible),
//! 512 (MMQ-eligible-dense). The MMQ auto-router activates at batch_size
//! >= 256 by default, so the v1 reference is FP16 WMMA below 256 and
//! MMQ at 256+.
//!
//! Run on gfx11xx:
//!   cargo run --release -p rdna-compute --example bench_hfq4v3_vs_v1

use rdna_compute::{DType, Gpu};
use std::time::Instant;

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    let arch = gpu.arch.clone();
    eprintln!("GPU: {arch}");
    if !matches!(arch.as_str(),
        "gfx1100" | "gfx1101" | "gfx1102" | "gfx1103" | "gfx1150" | "gfx1151")
    {
        eprintln!("SKIP: requires gfx11xx (RDNA3/3.5). Current: {arch}");
        std::process::exit(0);
    }

    // Shapes chosen from Qwen3.5-9B / 4B / 27B production.
    let shapes: &[(&str, usize, usize)] = &[
        ("9b_wo",   4096, 4096),
        ("9b_down", 4096, 12288),
        ("4b_wo",   2560, 2560),
        ("4b_down", 2560, 9728),
        ("27b_wo", 5120, 5120),
        ("27b_down", 5120, 17920),
    ];
    let batches = &[32usize, 128, 256, 512];

    println!("\n=== HFQ4v3 K=64 iu8 MMQ vs HFQ4-G256 v1 (auto-route) ===\n");
    println!("{:<10} {:>5} {:>6} {:>5}  {:>9} {:>9} {:>7}",
             "shape", "M", "K", "B", "v1_us", "v3_us", "speedup");
    println!("{}", "─".repeat(64));

    for &(name, m, k) in shapes {
        let groups256 = k / 256;
        let row_bytes_v1 = groups256 * 136;
        let row_bytes_v3 = (k / 64) * 36;

        // Synth weights (deterministic).
        let weight_v1 = synth_hfq4g256_weights(m, groups256, 0xC0DE_FACE);
        let weight_v3 = regroup_v1_to_v3(&weight_v1, m, k);

        let a_v1 = gpu.upload_raw_with_dtype(&weight_v1, &[m * row_bytes_v1], DType::HFQ4G256)
            .expect("upload v1");
        let a_v3 = gpu.upload_raw_with_dtype(&weight_v3, &[m * row_bytes_v3], DType::HFQ4V3G64)
            .expect("upload v3");

        for &b in batches {
            let x_host: Vec<f32> = (0..b * k).map(|i| {
                let v = ((i as i64).wrapping_mul(1103515245).wrapping_add(12345)) as f32;
                (v * 1e-9) % 2.0 - 1.0
            }).collect();
            let y_init: Vec<f32> = (0..b * m).map(|i| {
                let v = ((i as i64).wrapping_mul(2147483647).wrapping_add(7)) as f32;
                (v * 1e-7) % 1.0
            }).collect();

            let x = gpu.upload_f32(&x_host, &[b * k]).unwrap();
            let y = gpu.alloc_tensor(&[b * m], DType::F32).unwrap();
            gpu.hip.memcpy_htod(&y.buf, bytes_of(&y_init)).unwrap();

            // Warm up both paths so JIT compile / cache is amortized.
            for _ in 0..3 {
                gpu.hip.memcpy_htod(&y.buf, bytes_of(&y_init)).unwrap();
                gpu.gemm_hfq4g256_residual(&a_v1, &x, &y, m, k, b).unwrap();
                gpu.hip.memcpy_htod(&y.buf, bytes_of(&y_init)).unwrap();
                gpu.gemm_hfq4g256_residual(&a_v3, &x, &y, m, k, b).unwrap();
            }
            gpu.hip.device_synchronize().unwrap();

            // Time best of 5.
            let mut best_v1 = f64::INFINITY;
            let mut best_v3 = f64::INFINITY;
            for _ in 0..5 {
                gpu.hip.memcpy_htod(&y.buf, bytes_of(&y_init)).unwrap();
                gpu.hip.device_synchronize().unwrap();
                let t = Instant::now();
                gpu.gemm_hfq4g256_residual(&a_v1, &x, &y, m, k, b).unwrap();
                gpu.hip.device_synchronize().unwrap();
                best_v1 = best_v1.min(t.elapsed().as_secs_f64() * 1e6);

                gpu.hip.memcpy_htod(&y.buf, bytes_of(&y_init)).unwrap();
                gpu.hip.device_synchronize().unwrap();
                let t = Instant::now();
                gpu.gemm_hfq4g256_residual(&a_v3, &x, &y, m, k, b).unwrap();
                gpu.hip.device_synchronize().unwrap();
                best_v3 = best_v3.min(t.elapsed().as_secs_f64() * 1e6);
            }

            let speedup = best_v1 / best_v3;
            println!("{:<10} {:>5} {:>6} {:>5}  {:>9.1} {:>9.1} {:>6.2}x",
                     name, m, k, b, best_v1, best_v3, speedup);
        }
    }
}

fn synth_hfq4g256_weights(m: usize, groups: usize, seed: u64) -> Vec<u8> {
    let total = m * groups * 136;
    let mut out = vec![0u8; total];
    let mut state = seed;
    let mut next = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (state >> 33) as u32
    };
    for row in 0..m {
        for g in 0..groups {
            let gp = (row * groups + g) * 136;
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

fn regroup_v1_to_v3(weight_v1: &[u8], m: usize, k: usize) -> Vec<u8> {
    let groups256 = k / 256;
    let row_bytes_v1 = groups256 * 136;
    let row_bytes_v3 = (k / 64) * 36;
    let mut out = vec![0u8; m * row_bytes_v3];

    for row in 0..m {
        let row_in = &weight_v1[row * row_bytes_v1..(row + 1) * row_bytes_v1];

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
    if exp_unbiased < -24 { return (sign << 15) as u16; }
    if exp_unbiased < -14 {
        let shift = -14 - exp_unbiased;
        let mantissa = (frac | 0x800000) >> (13 + shift);
        let round_bit = ((frac | 0x800000) >> (12 + shift)) & 1;
        let m16 = mantissa + round_bit;
        return ((sign << 15) | m16) as u16;
    }
    if exp_unbiased > 15 { return ((sign << 15) | (0x1F << 10)) as u16; }
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
        if frac == 0 { return f32::from_bits(sign << 31); }
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
