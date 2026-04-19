//! Isolated correctness check for `fused_rmsnorm_rotate_mq_batched`.
//!
//! Builds an [N × K] x, calls the batched kernel once, and compares each row
//! against the single-row kernel applied N times to the same x. A mismatch
//! (especially zeros on rows ≥ 1) localises the batched-prefill bug purely
//! to this kernel.

fn main() {
    use rdna_compute::DType;

    let mut gpu = rdna_compute::Gpu::init().unwrap();
    gpu.ensure_mq_signs().unwrap();

    let k: usize = 4096;
    let n: usize = 5;
    let eps = 1e-6f32;

    // Deterministic per-row content so every row is distinct.
    let mut x: Vec<f32> = Vec::with_capacity(n * k);
    for r in 0..n {
        for i in 0..k {
            let v = (((i * 7 + 13 + r * 131) % 97) as f32) / 47.0 - 0.8 + (r as f32) * 0.01;
            x.push(v);
        }
    }
    let w: Vec<f32> = (0..k).map(|i| 1.0 + ((i * 11 + 3) % 51) as f32 / 233.0).collect();

    let d_x = gpu.upload_f32(&x, &[n, k]).unwrap();
    let d_w = gpu.upload_f32(&w, &[k]).unwrap();
    let d_out_batched = gpu.zeros(&[n, k], DType::F32).unwrap();

    // Batched call — single launch, grid.x = N.
    gpu.fused_rmsnorm_rotate_mq_batched(&d_x, &d_w, &d_out_batched, k, eps, n)
        .unwrap();

    // Reference: N single-row launches.
    let d_out_ref = gpu.zeros(&[n, k], DType::F32).unwrap();
    for r in 0..n {
        let d_x_row = gpu.upload_f32(&x[r * k..(r + 1) * k], &[k]).unwrap();
        let d_row_out = gpu.zeros(&[k], DType::F32).unwrap();
        gpu.fused_rmsnorm_rotate_mq(&d_x_row, &d_w, &d_row_out, k, eps).unwrap();
        let row = gpu.download_f32(&d_row_out).unwrap();
        // Splice into ref buffer by uploading into the right slot.
        let mut cur = gpu.download_f32(&d_out_ref).unwrap();
        cur[r * k..(r + 1) * k].copy_from_slice(&row);
        let fresh = gpu.upload_f32(&cur, &[n, k]).unwrap();
        // Swap d_out_ref's buffer contents.
        // Simpler: just hold fresh values in host memory.
        gpu.free_tensor(d_x_row).unwrap();
        gpu.free_tensor(d_row_out).unwrap();
        gpu.free_tensor(fresh).unwrap();
        // We'll use the host buffer below.
        let _ = cur;
    }

    // Rebuild reference purely on host by re-running per-row and stitching.
    let mut ref_host = vec![0f32; n * k];
    for r in 0..n {
        let d_x_row = gpu.upload_f32(&x[r * k..(r + 1) * k], &[k]).unwrap();
        let d_row_out = gpu.zeros(&[k], DType::F32).unwrap();
        gpu.fused_rmsnorm_rotate_mq(&d_x_row, &d_w, &d_row_out, k, eps).unwrap();
        let row = gpu.download_f32(&d_row_out).unwrap();
        ref_host[r * k..(r + 1) * k].copy_from_slice(&row);
        gpu.free_tensor(d_x_row).unwrap();
        gpu.free_tensor(d_row_out).unwrap();
    }

    let got = gpu.download_f32(&d_out_batched).unwrap();

    let mut any_diff = false;
    for r in 0..n {
        let row_got = &got[r * k..(r + 1) * k];
        let row_ref = &ref_host[r * k..(r + 1) * k];
        let all_zero = row_got.iter().all(|&v| v == 0.0);
        let mut max_abs = 0f32;
        for i in 0..k {
            let d = (row_got[i] - row_ref[i]).abs();
            if d > max_abs { max_abs = d; }
        }
        eprintln!(
            "row {r:>2}: all_zero_got={all_zero}  max_abs_vs_ref={max_abs:.3e}  first4_got={:?}  first4_ref={:?}",
            &row_got[..4],
            &row_ref[..4]
        );
        if max_abs > 1e-4 { any_diff = true; }
    }

    if any_diff {
        eprintln!("\nRESULT: BATCHED != per-row reference");
        std::process::exit(1);
    } else {
        eprintln!("\nRESULT: batched matches per-row reference for all {n} rows");
    }
}
