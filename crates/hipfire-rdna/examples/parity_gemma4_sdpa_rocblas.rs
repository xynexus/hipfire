// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! Reproduce one Gemma 4 SDPA output row with the same decomposition used by
//! PyTorch's ROCm math backend: BF16 QK with F32 accumulation, F32 softmax,
//! F32 PV over exactly materialized BF16 values, then BF16 output. Inputs come
//! from `capture_layer0_sdpa.py`.

use hipfire_rdna::{DType, Gpu};
use std::fs;
use std::path::Path;

fn read_f32(path: &Path) -> Result<Vec<f32>, String> {
    let bytes = fs::read(path).map_err(|error| format!("{}: {error}", path.display()))?;
    if bytes.len() % 4 != 0 {
        return Err(format!("{} has a partial F32 value", path.display()));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect())
}

fn main() -> Result<(), String> {
    let args = std::env::args().collect::<Vec<_>>();
    if args.len() != 6 {
        return Err(
            "usage: parity_gemma4_sdpa_rocblas DIR N_HEADS N_KV_HEADS HEAD_DIM QUERY_ROW"
                .to_string(),
        );
    }
    let dir = Path::new(&args[1]);
    let n_heads = args[2]
        .parse::<usize>()
        .map_err(|error| error.to_string())?;
    let n_kv_heads = args[3]
        .parse::<usize>()
        .map_err(|error| error.to_string())?;
    let head_dim = args[4]
        .parse::<usize>()
        .map_err(|error| error.to_string())?;
    let query_row = args[5]
        .parse::<usize>()
        .map_err(|error| error.to_string())?;
    let q = read_f32(&dir.join("operator_q_rope.f32"))?;
    let k = read_f32(&dir.join("operator_k_rope.f32"))?;
    let v = read_f32(&dir.join("operator_v_attention.f32"))?;
    let oracle = read_f32(&dir.join("operator_attention_raw.f32"))?;
    let q_width = n_heads * head_dim;
    let kv_width = n_kv_heads * head_dim;
    let seq = q.len() / q_width;
    if query_row >= seq
        || k.len() != seq * kv_width
        || v.len() != seq * kv_width
        || oracle.len() != q.len()
    {
        return Err("captured tensor geometry mismatch".to_string());
    }

    let mut gpu = Gpu::init().map_err(|error| format!("GPU init: {error:?}"))?;
    if gpu.rocblas.is_none() {
        return Err("rocBLAS did not load; set HIPFIRE_ROCBLAS_ALL_ARCHS=1".to_string());
    }
    let mut candidate = vec![0.0f32; q_width];
    let group = n_heads / n_kv_heads;
    for head in 0..n_heads {
        let kv_head = head / group;
        let q_head = q
            [query_row * q_width + head * head_dim..query_row * q_width + (head + 1) * head_dim]
            .to_vec();
        let mut k_head = Vec::with_capacity(seq * head_dim);
        let mut v_head = Vec::with_capacity(seq * head_dim);
        for position in 0..seq {
            let start = position * kv_width + kv_head * head_dim;
            k_head.extend_from_slice(&k[start..start + head_dim]);
            v_head.extend_from_slice(&v[start..start + head_dim]);
        }

        let q_f32 = gpu
            .upload_f32(&q_head, &[head_dim])
            .map_err(|e| format!("Q: {e:?}"))?;
        let k_f32 = gpu
            .upload_f32(&k_head, &[seq * head_dim])
            .map_err(|e| format!("K: {e:?}"))?;
        let v_f32 = gpu
            .upload_f32(&v_head, &[seq * head_dim])
            .map_err(|e| format!("V: {e:?}"))?;
        let q_bf16 = gpu
            .alloc_tensor(&[head_dim], DType::BF16)
            .map_err(|e| format!("Q16: {e:?}"))?;
        let k_bf16 = gpu
            .alloc_tensor(&[seq * head_dim], DType::BF16)
            .map_err(|e| format!("K16: {e:?}"))?;
        gpu.cast_f32_to_bf16(&q_f32, &q_bf16)
            .map_err(|e| format!("cast Q: {e:?}"))?;
        gpu.cast_f32_to_bf16(&k_f32, &k_bf16)
            .map_err(|e| format!("cast K: {e:?}"))?;

        let scores = gpu
            .zeros(&[1, seq], DType::F32)
            .map_err(|e| format!("scores: {e:?}"))?;
        gpu.rocblas_sdpa_qk_strided_bf16_f32(&k_bf16, &q_bf16, &scores, seq, head_dim, head_dim)
            .map_err(|e| format!("QK GEMM: {e:?}"))?;
        let probabilities = gpu
            .zeros(&[1, seq], DType::F32)
            .map_err(|e| format!("probabilities: {e:?}"))?;
        gpu.softmax_train_fwd(&scores, &probabilities, 1, seq)
            .map_err(|e| format!("softmax: {e:?}"))?;
        let output = gpu
            .zeros(&[head_dim], DType::F32)
            .map_err(|e| format!("output: {e:?}"))?;
        gpu.rocblas_sdpa_pv_strided_f32(&v_f32, &probabilities, &output, seq, head_dim, head_dim)
            .map_err(|e| format!("PV GEMM: {e:?}"))?;
        gpu.bf16_round_trip_f32(&output)
            .map_err(|e| format!("round output: {e:?}"))?;
        let values = gpu
            .download_f32(&output)
            .map_err(|e| format!("download: {e:?}"))?;
        candidate[head * head_dim..(head + 1) * head_dim].copy_from_slice(&values);
    }

    let reference = &oracle[query_row * q_width..(query_row + 1) * q_width];
    let mut max_abs = 0.0f32;
    let mut error_sq = 0.0f64;
    let mut reference_sq = 0.0f64;
    let mut exact = 0usize;
    for (&expected, &actual) in reference.iter().zip(&candidate) {
        let error = (expected - actual).abs();
        max_abs = max_abs.max(error);
        error_sq += f64::from(error).powi(2);
        reference_sq += f64::from(expected).powi(2);
        exact += usize::from(expected.to_bits() == actual.to_bits());
    }
    println!(
        "gemma4_sdpa_rocblas: row={query_row} max_abs={max_abs:.9} nrmse={:.9} exact={exact}/{}",
        (error_sq / reference_sq).sqrt(),
        reference.len()
    );
    Ok(())
}
