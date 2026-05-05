//! WMMA vs scalar GEMM correctness test — find the exact error pattern
fn main() {
    let mut gpu = rdna_compute::Gpu::init().unwrap();
    eprintln!("GPU: {}", gpu.arch);

    // Minimal case: 16 rows × 256 cols × 1 batch element
    // This should produce exactly one 16×16 WMMA tile with 1 valid batch column
    let m = 16usize;
    let k = 256usize;

    for batch in [1, 2, 4, 16] {
        // Identity-like weights: row r, col c → r == c ? 1.0 : 0.0 (within group)
        let mut f32_w = vec![0.0f32; m * k];
        for r in 0..m {
            for c in 0..k {
                f32_w[r * k + c] = if c % m == r { 1.0 } else { 0.0 };
            }
        }

        let quantized = quantize_hfq4g256(&f32_w);

        // X: batch b, col c → (b + 1) * (c + 1) scaled small
        let mut x_data = vec![0.0f32; batch * k];
        for b in 0..batch {
            for c in 0..k {
                x_data[b * k + c] = (b as f32 + 1.0) * 0.1;
            }
        }

        let y_init = vec![0.0f32; batch * m]; // no residual

        let d_a = gpu.upload_raw(&quantized, &[quantized.len()]).unwrap();
        let d_x = gpu.upload_f32(&x_data, &[batch * k]).unwrap();

        // Scalar
        let d_y_s = gpu.upload_f32(&y_init, &[batch * m]).unwrap();
        std::env::set_var("HIPFIRE_FP16", "0");
        gpu.gemm_hfq4g256_residual(&d_a, &d_x, &d_y_s, m, k, batch)
            .unwrap();
        let ys = gpu.download_f32(&d_y_s).unwrap();

        // WMMA
        let d_y_w = gpu.upload_f32(&y_init, &[batch * m]).unwrap();
        std::env::remove_var("HIPFIRE_FP16");
        gpu.gemm_hfq4g256_residual_wmma(&d_a, &d_x, &d_y_w, m, k, batch)
            .unwrap();
        let yw = gpu.download_f32(&d_y_w).unwrap();

        let mut max_err = 0.0f32;
        let mut bad = 0;
        for b in 0..batch {
            for r in 0..m {
                let idx = b * m + r;
                let err = (ys[idx] - yw[idx]).abs();
                max_err = max_err.max(err);
                if err > 0.1 {
                    bad += 1;
                }
            }
        }
        eprintln!(
            "batch={batch:2}: max_err={max_err:.4} bad={bad}/{}",
            batch * m
        );
        if bad > 0 && batch <= 2 {
            for b in 0..batch {
                for r in 0..m {
                    let idx = b * m + r;
                    let err = (ys[idx] - yw[idx]).abs();
                    if err > 0.01 {
                        eprintln!(
                            "  b={b} r={r}: scalar={:.4} wmma={:.4} err={:.4}",
                            ys[idx], yw[idx], err
                        );
                    }
                }
            }
        }
    }
}

fn quantize_hfq4g256(f32_data: &[f32]) -> Vec<u8> {
    let group_size = 256;
    let block_bytes = 136;
    let n = f32_data.len();
    let n_blocks = (n + group_size - 1) / group_size;
    let mut output = vec![0u8; n_blocks * block_bytes];
    for b in 0..n_blocks {
        let start = b * group_size;
        let end = (start + group_size).min(n);
        let group = &f32_data[start..end];
        let min_val = group.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = group.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let range = max_val - min_val;
        let scale = if range > 0.0 { range / 15.0 } else { 1.0 };
        let inv_scale = if range > 0.0 { 1.0 / scale } else { 0.0 };
        let out_off = b * block_bytes;
        output[out_off..out_off + 4].copy_from_slice(&scale.to_le_bytes());
        output[out_off + 4..out_off + 8].copy_from_slice(&min_val.to_le_bytes());
        for i in 0..128 {
            let lo = if 2 * i < (end - start) {
                ((group[2 * i] - min_val) * inv_scale + 0.5) as u8
            } else {
                0
            };
            let hi = if 2 * i + 1 < (end - start) {
                ((group[2 * i + 1] - min_val) * inv_scale + 0.5) as u8
            } else {
                0
            };
            output[out_off + 8 + i] = lo.min(15) | (hi.min(15) << 4);
        }
    }
    output
}
