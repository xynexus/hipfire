//! Channel test for `sumsq_reduce_bf16` — per-channel sum-of-squares
//! accumulator over a BF16 activation tensor.
//!
//! Reference: CPU FP64 `Σ_t x[t,k]²` over a small `[batch, K]` random
//! BF16 input. Tolerance: relative error < 1e-3 + abs error < 1e-4 (BF16
//! has 7 mantissa bits — wider tolerance than F16 reductions, but the
//! `Σx²` itself is FP32 on the GPU and FP64 on the CPU, so the only
//! divergence source is the BF16 → f32 upconvert of each input element).
//!
//! Host gpu may be gfx1100 (no MFMA needed for this kernel — sumsq_reduce
//! is cross-arch). Foundational test for the Tier 1 hipfire-native AWQ
//! calibration path.

use rdna_compute::{DType, Gpu};

const BATCH: usize = 8;
const K: usize = 64;

/// Truncate FP32 → BF16 lane bits (top 16 bits, round-to-nearest-even via
/// the standard IEEE float-to-bf16 conversion). Returns the BF16 as a u16
/// (the host-side raw lane bits — what gets written to HBM and re-read by
/// the kernel as `__bf16`).
fn f32_to_bf16_bits(f: f32) -> u16 {
    let bits = f.to_bits();
    // Round-to-nearest-even: bias is 0x7FFF + (lower-16-bits-LSB).
    let rounded = bits.wrapping_add(0x7FFF + ((bits >> 16) & 1));
    (rounded >> 16) as u16
}

/// LCG random source — deterministic, range [-1.0, 1.0).
fn rand_f32(seed: &mut u32) -> f32 {
    *seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
    let u = (*seed >> 16) & 0x7fff;
    (u as f32 / 32_768.0 - 0.5) * 2.0
}

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");

    // ─── Generate BF16 input ───
    let mut seed: u32 = 0xDEADBEEF;
    let mut x_f32: Vec<f32> = Vec::with_capacity(BATCH * K);
    for _ in 0..BATCH * K {
        x_f32.push(rand_f32(&mut seed));
    }

    // Round-trip through BF16 so the CPU reference exactly mirrors what
    // the GPU sees (BF16 lossy conversion happens here, on the host).
    let x_bf16_bits: Vec<u16> = x_f32.iter().map(|&f| f32_to_bf16_bits(f)).collect();
    let x_round: Vec<f32> = x_bf16_bits
        .iter()
        .map(|&b| f32::from_bits((b as u32) << 16))
        .collect();

    // ─── Upload BF16 to GPU as a DType::F16-tagged tensor (lane bits identical) ───
    let x_bytes_u8: Vec<u8> = x_bf16_bits
        .iter()
        .flat_map(|&b| b.to_le_bytes())
        .collect();

    let x = gpu
        .alloc_tensor(&[BATCH, K], DType::F16)
        .expect("alloc x");
    gpu.hip
        .memcpy_htod(&x.buf, &x_bytes_u8)
        .expect("memcpy x → device");

    // ─── Allocate zeroed output buffers ───
    let mut acc = gpu.zeros(&[K], DType::F32).expect("alloc acc");
    let mut n_tokens = gpu.zeros(&[1], DType::F32).expect("alloc n_tokens");

    // ─── Run kernel ───
    gpu.sumsq_reduce_bf16(&x, &mut acc, &mut n_tokens)
        .expect("sumsq_reduce_bf16 launch");

    let acc_host = gpu.download_f32(&acc).expect("download acc");
    let n_tokens_host = gpu.download_f32(&n_tokens).expect("download n_tokens");

    // ─── CPU reference (FP64 accumulation of the BF16-roundtripped inputs) ───
    let mut acc_ref = vec![0.0f64; K];
    for t in 0..BATCH {
        for k in 0..K {
            let v = x_round[t * K + k] as f64;
            acc_ref[k] += v * v;
        }
    }

    // ─── Compare ───
    let mut max_abs_err = 0.0f64;
    let mut max_rel_err = 0.0f64;
    let mut argmax_k = 0usize;
    for k in 0..K {
        let got = acc_host[k] as f64;
        let want = acc_ref[k];
        let abs_err = (got - want).abs();
        let denom = want.abs().max(1e-12);
        let rel_err = abs_err / denom;
        if abs_err > max_abs_err {
            max_abs_err = abs_err;
            argmax_k = k;
        }
        if rel_err > max_rel_err {
            max_rel_err = rel_err;
        }
    }

    let n_tok_expected = BATCH as f32;
    let n_tok_ok = (n_tokens_host[0] - n_tok_expected).abs() < 1e-6;

    println!("sumsq_reduce_bf16 channel test");
    println!("  BATCH={BATCH} K={K}");
    println!("  max_abs_err = {:.3e}  (at k={argmax_k})", max_abs_err);
    println!("  max_rel_err = {:.3e}", max_rel_err);
    println!("  n_tokens    = {} (expected {})", n_tokens_host[0], n_tok_expected);

    let abs_ok = max_abs_err < 1e-4;
    let rel_ok = max_rel_err < 1e-3;

    if abs_ok && rel_ok && n_tok_ok {
        println!("PASS");
        std::process::exit(0);
    }

    // FAIL diagnostics
    eprintln!("FAIL");
    eprintln!("  abs_ok = {abs_ok} (need < 1e-4)");
    eprintln!("  rel_ok = {rel_ok} (need < 1e-3)");
    eprintln!("  n_tok_ok = {n_tok_ok}");
    // Print first few entries
    for k in 0..K.min(8) {
        eprintln!(
            "  k={k}: got={:.6e}  want={:.6e}  diff={:.3e}",
            acc_host[k] as f64,
            acc_ref[k],
            (acc_host[k] as f64 - acc_ref[k]).abs(),
        );
    }
    std::process::exit(1);
}
