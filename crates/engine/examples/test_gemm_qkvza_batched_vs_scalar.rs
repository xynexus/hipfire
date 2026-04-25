//! Isolated correctness check for `gemm_qkvza_hfq4g256` (scalar batched 4-way)
//! vs `fused_qkvza_hfq4g256` (single-row 4-way).
//!
//! The batched scalar kernel claims byte-exact parity with running the
//! single-row kernel N times on the same x[b]. This harness verifies that
//! claim on synthetic weights + activations at N=1 and N>1.
//!
//! The scalar path is forced by setting HIPFIRE_FP16=0 so that the gfx1100
//! WMMA fast path in `gemm_qkvza_hfq4g256()` is bypassed.
//!
//! A mismatch at row 0 (where the single-row kernel and the scalar batched
//! kernel should see identical inputs and produce identical outputs)
//! localises the LA-preamble divergence purely to this batched kernel.

fn main() {
    // Force the scalar batched path (skip WMMA).
    std::env::set_var("HIPFIRE_FP16", "0");

    let mut gpu = rdna_compute::Gpu::init().unwrap();

    // Qwen3.5-9B layer-0 LA preamble dimensions.
    let k: usize = 4096;
    let qkv_m: usize = 4096; // k_dim (4×256) + k_dim (4×256) + v_dim (16×128) = 4096 (representative)
    let z_m: usize = 2048;
    let beta_m: usize = 16;
    let alpha_m: usize = 16;

    // We fabricate a single HFQ4-G256 weight matrix per projection. Each row
    // has groups_per_row groups of 136 bytes: 4 bytes scale, 4 bytes zero,
    // 128 bytes (32 × uint32) packed nibbles = 256 nibbles.
    let groups_per_row = k / 256;
    let row_bytes = groups_per_row * 136;

    fn make_weights(m: usize, k: usize, seed: u32) -> Vec<u8> {
        let groups_per_row = k / 256;
        let row_bytes = groups_per_row * 136;
        let mut out = vec![0u8; m * row_bytes];
        for r in 0..m {
            for g in 0..groups_per_row {
                let off = r * row_bytes + g * 136;
                // Deterministic per-(row, group, seed) scale/zero.
                let sc = ((((r as u32).wrapping_mul(7) ^ (g as u32).wrapping_mul(131) ^ seed) % 97) as f32) / 4096.0 + 1e-4;
                let zp = (((((r as u32).wrapping_mul(11) ^ (g as u32).wrapping_mul(59) ^ seed) % 63) as f32) / 2048.0) - 0.015;
                out[off..off + 4].copy_from_slice(&sc.to_le_bytes());
                out[off + 4..off + 8].copy_from_slice(&zp.to_le_bytes());
                for i in 0..32 {
                    let u: u32 = (r as u32)
                        .wrapping_mul(0x9E3779B1)
                        .wrapping_add((g as u32).wrapping_mul(0x1B873593))
                        .wrapping_add(i as u32 * 0x85EBCA6B)
                        .wrapping_add(seed);
                    out[off + 8 + i * 4..off + 8 + i * 4 + 4].copy_from_slice(&u.to_le_bytes());
                }
            }
        }
        out
    }

    let w_qkv = make_weights(qkv_m, k, 0x1111);
    let w_z = make_weights(z_m, k, 0x2222);
    let w_beta = make_weights(beta_m, k, 0x3333);
    let w_alpha = make_weights(alpha_m, k, 0x4444);

    let d_w_qkv = gpu.upload_raw(&w_qkv, &[qkv_m * row_bytes]).unwrap();
    let d_w_z = gpu.upload_raw(&w_z, &[z_m * row_bytes]).unwrap();
    let d_w_beta = gpu.upload_raw(&w_beta, &[beta_m * row_bytes]).unwrap();
    let d_w_alpha = gpu.upload_raw(&w_alpha, &[alpha_m * row_bytes]).unwrap();

    for &n in &[1usize, 2, 4, 5, 8, 9][..] {
        eprintln!("\n=== N = {n} ===");

        // Deterministic per-row x.
        let mut x: Vec<f32> = Vec::with_capacity(n * k);
        for r in 0..n {
            for i in 0..k {
                let v = (((i * 7 + 13 + r * 131) % 97) as f32) / 47.0 - 0.8 + (r as f32) * 0.01;
                x.push(v);
            }
        }
        let d_x = gpu.upload_f32(&x, &[n, k]).unwrap();

        // Batched run: one gemm_qkvza_hfq4g256 launch with N rows.
        let d_y_qkv_b = gpu.zeros(&[n, qkv_m], rdna_compute::DType::F32).unwrap();
        let d_y_z_b = gpu.zeros(&[n, z_m], rdna_compute::DType::F32).unwrap();
        let d_y_beta_b = gpu.zeros(&[n, beta_m], rdna_compute::DType::F32).unwrap();
        let d_y_alpha_b = gpu.zeros(&[n, alpha_m], rdna_compute::DType::F32).unwrap();
        gpu.gemm_qkvza_hfq4g256(
            &d_w_qkv, &d_w_z, &d_w_beta, &d_w_alpha,
            &d_x,
            &d_y_qkv_b, &d_y_z_b, &d_y_beta_b, &d_y_alpha_b,
            qkv_m, z_m, beta_m, alpha_m,
            k, n,
        ).unwrap();

        let y_qkv_b = gpu.download_f32(&d_y_qkv_b).unwrap();
        let y_z_b = gpu.download_f32(&d_y_z_b).unwrap();
        let y_beta_b = gpu.download_f32(&d_y_beta_b).unwrap();
        let y_alpha_b = gpu.download_f32(&d_y_alpha_b).unwrap();

        // Per-row reference: N launches of fused_qkvza_hfq4g256 on x[r,:].
        let mut y_qkv_ref = vec![0f32; n * qkv_m];
        let mut y_z_ref = vec![0f32; n * z_m];
        let mut y_beta_ref = vec![0f32; n * beta_m];
        let mut y_alpha_ref = vec![0f32; n * alpha_m];

        for r in 0..n {
            let d_x_row = gpu.upload_f32(&x[r * k..(r + 1) * k], &[k]).unwrap();
            let d_yq = gpu.zeros(&[qkv_m], rdna_compute::DType::F32).unwrap();
            let d_yz = gpu.zeros(&[z_m], rdna_compute::DType::F32).unwrap();
            let d_yb = gpu.zeros(&[beta_m], rdna_compute::DType::F32).unwrap();
            let d_ya = gpu.zeros(&[alpha_m], rdna_compute::DType::F32).unwrap();
            gpu.fused_qkvza_hfq4g256(
                &d_w_qkv, &d_w_z, &d_w_beta, &d_w_alpha,
                &d_x_row,
                &d_yq, &d_yz, &d_yb, &d_ya,
                qkv_m, z_m, beta_m, alpha_m,
                k,
            ).unwrap();
            let yq = gpu.download_f32(&d_yq).unwrap();
            let yz = gpu.download_f32(&d_yz).unwrap();
            let yb = gpu.download_f32(&d_yb).unwrap();
            let ya = gpu.download_f32(&d_ya).unwrap();
            y_qkv_ref[r * qkv_m..(r + 1) * qkv_m].copy_from_slice(&yq);
            y_z_ref[r * z_m..(r + 1) * z_m].copy_from_slice(&yz);
            y_beta_ref[r * beta_m..(r + 1) * beta_m].copy_from_slice(&yb);
            y_alpha_ref[r * alpha_m..(r + 1) * alpha_m].copy_from_slice(&ya);
            gpu.free_tensor(d_x_row).unwrap();
            gpu.free_tensor(d_yq).unwrap();
            gpu.free_tensor(d_yz).unwrap();
            gpu.free_tensor(d_yb).unwrap();
            gpu.free_tensor(d_ya).unwrap();
        }

        let mut any_fail = false;
        for r in 0..n {
            let diag = |name: &str, got: &[f32], refv: &[f32], m: usize| -> (f32, usize, f32, f32) {
                let mut max_abs = 0f32;
                let mut first_bad = usize::MAX;
                let mut first_got = 0f32;
                let mut first_ref = 0f32;
                for i in 0..m {
                    let d = (got[r * m + i] - refv[r * m + i]).abs();
                    if d > max_abs {
                        max_abs = d;
                    }
                    if first_bad == usize::MAX && got[r * m + i].to_bits() != refv[r * m + i].to_bits() {
                        first_bad = i;
                        first_got = got[r * m + i];
                        first_ref = refv[r * m + i];
                    }
                }
                (max_abs, first_bad, first_got, first_ref)
            };
            let (d_qkv, fb_qkv, g_qkv, r_qkv) = diag("qkv", &y_qkv_b, &y_qkv_ref, qkv_m);
            let (d_z, fb_z, g_z, r_z) = diag("z", &y_z_b, &y_z_ref, z_m);
            let (d_beta, fb_beta, g_beta, r_beta) = diag("beta", &y_beta_b, &y_beta_ref, beta_m);
            let (d_alpha, fb_alpha, g_alpha, r_alpha) = diag("alpha", &y_alpha_b, &y_alpha_ref, alpha_m);

            let bad = d_qkv > 0.0 || d_z > 0.0 || d_beta > 0.0 || d_alpha > 0.0;
            if bad {
                any_fail = true;
            }
            eprintln!(
                "  row {r}: qkv(max={:.3e} first_bad={} got={:.6} ref={:.6})  z(max={:.3e} fb={})  beta(max={:.3e} fb={})  alpha(max={:.3e} fb={})",
                d_qkv, fb_qkv as isize, g_qkv, r_qkv, d_z, fb_z as isize, d_beta, fb_beta as isize, d_alpha, fb_alpha as isize,
            );
            let _ = (g_z, r_z, g_beta, r_beta, g_alpha, r_alpha);
        }

        gpu.free_tensor(d_x).unwrap();
        gpu.free_tensor(d_y_qkv_b).unwrap();
        gpu.free_tensor(d_y_z_b).unwrap();
        gpu.free_tensor(d_y_beta_b).unwrap();
        gpu.free_tensor(d_y_alpha_b).unwrap();

        if any_fail {
            eprintln!("  => N={n} DIVERGES from per-row reference");
        } else {
            eprintln!("  => N={n} byte-exact with per-row reference");
        }
    }
}
