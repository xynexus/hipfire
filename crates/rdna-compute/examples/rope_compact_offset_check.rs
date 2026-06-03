// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

// Verifies the batched RoPE pos_offset parameter without a model:
// T1: batched positions [K..K+B), offset 0 equals positions [0..B), offset K.
// T2: batched B=1, position 0, offset K equals per-token RoPE at position K.

use rdna_compute::{DType, Gpu};

fn lcg(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as f32 / (1u64 << 31) as f32) - 1.0
        })
        .collect()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn pos_tensor(gpu: &mut Gpu, vals: &[i32]) -> rdna_compute::GpuTensor {
    let t = gpu.alloc_tensor(&[vals.len()], DType::F32).unwrap();
    let bytes: Vec<u8> = vals.iter().flat_map(|p| p.to_ne_bytes()).collect();
    gpu.hip.memcpy_htod(&t.buf, &bytes).unwrap();
    t
}

fn main() {
    let b = 4usize;
    let n_heads_q = 4usize;
    let n_heads_k = 2usize;
    let head_dim = 64usize;
    let n_rot = 64usize;
    let freq_base = 1.0e6f32;
    let k_off = 137i32;

    let mut gpu = Gpu::init().expect("GPU init");
    eprintln!(
        "GPU: {} (b={b} nhq={n_heads_q} nhk={n_heads_k} hd={head_dim} n_rot={n_rot} K={k_off})",
        gpu.arch
    );

    let q_src = lcg(0xA5A5, b * n_heads_q * head_dim);
    let k_src = lcg(0xC3C3, b * n_heads_k * head_dim);

    let qa = gpu.upload_f32(&q_src, &[q_src.len()]).unwrap();
    let ka = gpu.upload_f32(&k_src, &[k_src.len()]).unwrap();
    let pos_a = pos_tensor(
        &mut gpu,
        &(0..b as i32).map(|i| k_off + i).collect::<Vec<_>>(),
    );
    gpu.rope_partial_interleaved_f32_batched(
        &qa, &ka, &pos_a, n_heads_q, n_heads_k, head_dim, n_rot, freq_base, b, 0,
    )
    .unwrap();
    let qa_out = gpu.download_f32(&qa).unwrap();
    let ka_out = gpu.download_f32(&ka).unwrap();

    let qb = gpu.upload_f32(&q_src, &[q_src.len()]).unwrap();
    let kb = gpu.upload_f32(&k_src, &[k_src.len()]).unwrap();
    let pos_b = pos_tensor(&mut gpu, &(0..b as i32).collect::<Vec<_>>());
    gpu.rope_partial_interleaved_f32_batched(
        &qb, &kb, &pos_b, n_heads_q, n_heads_k, head_dim, n_rot, freq_base, b, k_off,
    )
    .unwrap();
    let qb_out = gpu.download_f32(&qb).unwrap();
    let kb_out = gpu.download_f32(&kb).unwrap();

    let t1q = max_abs_diff(&qa_out, &qb_out);
    let t1k = max_abs_diff(&ka_out, &kb_out);
    eprintln!("T1 offset equivalence max|dq|={t1q:.3e} max|dk|={t1k:.3e}");

    let q1 = q_src[..n_heads_q * head_dim].to_vec();
    let k1 = k_src[..n_heads_k * head_dim].to_vec();

    let qt = gpu.upload_f32(&q1, &[q1.len()]).unwrap();
    let kt = gpu.upload_f32(&k1, &[k1.len()]).unwrap();
    let pos_buf = gpu.hip.malloc(4).unwrap();
    gpu.hip.memcpy_htod(&pos_buf, &k_off.to_ne_bytes()).unwrap();
    gpu.rope_partial_interleaved_f32(
        &qt, &kt, &pos_buf, n_heads_q, n_heads_k, head_dim, n_rot, freq_base,
    )
    .unwrap();
    let qt_out = gpu.download_f32(&qt).unwrap();
    let kt_out = gpu.download_f32(&kt).unwrap();

    let qc = gpu.upload_f32(&q1, &[q1.len()]).unwrap();
    let kc = gpu.upload_f32(&k1, &[k1.len()]).unwrap();
    let pos_c = pos_tensor(&mut gpu, &[0]);
    gpu.rope_partial_interleaved_f32_batched(
        &qc, &kc, &pos_c, n_heads_q, n_heads_k, head_dim, n_rot, freq_base, 1, k_off,
    )
    .unwrap();
    let qc_out = gpu.download_f32(&qc).unwrap();
    let kc_out = gpu.download_f32(&kc).unwrap();

    let t2q = max_abs_diff(&qt_out, &qc_out);
    let t2k = max_abs_diff(&kt_out, &kc_out);
    eprintln!("T2 batched offset equals per-token max|dq|={t2q:.3e} max|dk|={t2k:.3e}");

    let eps = 1e-4f32;
    let pass = t1q < eps && t1k < eps && t2q < eps && t2k < eps;
    println!("RESULT: {}", if pass { "PASS" } else { "FAIL" });
    if !pass {
        std::process::exit(1);
    }
}
