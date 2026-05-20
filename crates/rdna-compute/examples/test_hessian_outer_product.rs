//! Channel test for `hessian_outer_product_bf16` — `H += xᵀ x` over a
//! BF16 activation tensor on gfx942.
//!
//! Reference: CPU FP64 `H[i,j] = Σ_t x[t,i] * x[t,j]` over a small
//! `[batch, K]` random BF16 input. Tolerance: relative error < 1e-3 +
//! abs error < 1e-4. BF16 has 7 mantissa bits — each product is FP32
//! exact, but the accumulator carries `batch`-element sums of products
//! of 8-bit-mantissa numbers; for `batch=16` the worst-case relative
//! error stays well inside 1e-3.
//!
//! Host gpu may be gfx1100 — kernel compiles via `--offload-arch=gfx942`
//! at runtime, so the test binary builds but does NOT exercise the MFMA
//! path on a non-CDNA3 host. Use this as a compile gate; runtime PASS
//! requires MI300x.

use rdna_compute::{DType, Gpu};

const BATCH: usize = 16;
const K: usize = 128;

fn f32_to_bf16_bits(f: f32) -> u16 {
    let bits = f.to_bits();
    let rounded = bits.wrapping_add(0x7FFF + ((bits >> 16) & 1));
    (rounded >> 16) as u16
}

fn rand_f32(seed: &mut u32) -> f32 {
    *seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
    let u = (*seed >> 16) & 0x7fff;
    (u as f32 / 32_768.0 - 0.5) * 2.0
}

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    let arch = gpu.arch.clone();
    eprintln!("=== test_hessian_outer_product ===");
    eprintln!("  arch = {arch}");
    // MFMA `f32_16x16x16_bf16` is gfx940/941/942 only (CDNA3). On any
    // other arch the JIT will reject the intrinsic with "needs target
    // feature mai-insts" — skip cleanly instead of panicking.
    if !matches!(arch.as_str(), "gfx940" | "gfx941" | "gfx942") {
        eprintln!("  SKIPPED: MFMA bf16 requires CDNA3 (gfx940/941/942), got {arch}");
        std::process::exit(0);
    }

    // ─── Generate BF16 input ───
    let mut seed: u32 = 0xC0FFEE42;
    let mut x_f32: Vec<f32> = Vec::with_capacity(BATCH * K);
    for _ in 0..BATCH * K {
        x_f32.push(rand_f32(&mut seed));
    }

    let x_bf16_bits: Vec<u16> = x_f32.iter().map(|&f| f32_to_bf16_bits(f)).collect();
    let x_round: Vec<f32> = x_bf16_bits
        .iter()
        .map(|&b| f32::from_bits((b as u32) << 16))
        .collect();

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

    // ─── Allocate zeroed H ───
    let mut h = gpu.zeros(&[K, K], DType::F32).expect("alloc H");

    // ─── Run kernel ───
    gpu.hessian_outer_product_bf16(&x, &mut h)
        .expect("hessian_outer_product_bf16 launch");

    let h_host = gpu.download_f32(&h).expect("download H");

    // ─── CPU reference (FP64 accumulation of BF16-roundtripped inputs) ───
    let mut h_ref = vec![0.0f64; K * K];
    for t in 0..BATCH {
        for i in 0..K {
            let xi = x_round[t * K + i] as f64;
            for j in 0..K {
                let xj = x_round[t * K + j] as f64;
                h_ref[i * K + j] += xi * xj;
            }
        }
    }

    // ─── Compare ───
    let mut max_abs_err = 0.0f64;
    let mut max_rel_err = 0.0f64;
    let mut argmax_i = 0usize;
    let mut argmax_j = 0usize;
    let mut max_ref_mag = 0.0f64;
    for i in 0..K {
        for j in 0..K {
            let got = h_host[i * K + j] as f64;
            let want = h_ref[i * K + j];
            max_ref_mag = max_ref_mag.max(want.abs());
            let abs_err = (got - want).abs();
            let denom = want.abs().max(1e-12);
            let rel_err = abs_err / denom;
            if abs_err > max_abs_err {
                max_abs_err = abs_err;
                argmax_i = i;
                argmax_j = j;
            }
            if rel_err > max_rel_err {
                max_rel_err = rel_err;
            }
        }
    }

    println!("hessian_outer_product_bf16 channel test");
    println!("  BATCH={BATCH} K={K}");
    println!("  max_ref_mag = {:.3e}", max_ref_mag);
    println!("  max_abs_err = {:.3e}  (at i={argmax_i}, j={argmax_j})", max_abs_err);
    println!("  max_rel_err = {:.3e}", max_rel_err);

    let abs_ok = max_abs_err < 1e-4;
    let rel_ok = max_rel_err < 1e-3;

    if abs_ok && rel_ok {
        println!("PASS");
        std::process::exit(0);
    }

    eprintln!("FAIL");
    eprintln!("  abs_ok = {abs_ok} (need < 1e-4)");
    eprintln!("  rel_ok = {rel_ok} (need < 1e-3)");
    // Print diagonal as a sanity sample (should equal Σx_i²)
    for k in 0..K.min(8) {
        let got = h_host[k * K + k] as f64;
        let want = h_ref[k * K + k];
        eprintln!(
            "  H[{k},{k}] (diagonal): got={:.6e}  want={:.6e}  diff={:.3e}",
            got,
            want,
            (got - want).abs(),
        );
    }
    std::process::exit(1);
}
