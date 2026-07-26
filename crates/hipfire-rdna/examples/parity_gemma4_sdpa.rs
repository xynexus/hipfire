// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! Compare the BF16 WMMA causal parity kernel against captured Gemma 4 ROCm
//! SDPA tensors. The oracle capture writes position-major raw F32 inputs.
//!
//! Usage:
//! `parity_gemma4_sdpa DIR N_HEADS N_KV_HEADS HEAD_DIM [SCALE [QUERY_ROW]]`

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
    if !(5..=7).contains(&args.len()) {
        return Err(
            "usage: parity_gemma4_sdpa DIR N_HEADS N_KV_HEADS HEAD_DIM [SCALE [QUERY_ROW]]"
                .to_string(),
        );
    }
    let dir = Path::new(&args[1]);
    let n_heads = args[2]
        .parse::<usize>()
        .map_err(|error| format!("invalid N_HEADS: {error}"))?;
    let n_kv_heads = args[3]
        .parse::<usize>()
        .map_err(|error| format!("invalid N_KV_HEADS: {error}"))?;
    let head_dim = args[4]
        .parse::<usize>()
        .map_err(|error| format!("invalid HEAD_DIM: {error}"))?;
    let scale = args
        .get(5)
        .map(|value| value.parse::<f32>())
        .transpose()
        .map_err(|error| format!("invalid SCALE: {error}"))?
        .unwrap_or(1.0);
    let query_row = args
        .get(6)
        .map(|value| value.parse::<usize>())
        .transpose()
        .map_err(|error| format!("invalid QUERY_ROW: {error}"))?;

    let mut q = read_f32(&dir.join("operator_q_rope.f32"))?;
    let k = read_f32(&dir.join("operator_k_rope.f32"))?;
    let v = read_f32(&dir.join("operator_v_attention.f32"))?;
    let mut oracle = read_f32(&dir.join("operator_attention_raw.f32"))?;
    let q_width = n_heads * head_dim;
    let kv_width = n_kv_heads * head_dim;
    if k.len() % kv_width != 0 {
        return Err(format!(
            "K length {} is not divisible by {kv_width}",
            k.len()
        ));
    }
    let seq = k.len() / kv_width;
    if k.len() != seq * kv_width || v.len() != seq * kv_width {
        return Err(format!(
            "K/V sizes do not match seq={seq}, kv_width={kv_width}: k={} v={}",
            k.len(),
            v.len()
        ));
    }
    if oracle.len() != q.len() || q.len() != seq * q_width {
        return Err(format!(
            "Q/oracle sizes do not match seq={seq}, q_width={q_width}: q={} oracle={}",
            q.len(),
            oracle.len()
        ));
    }
    let (batch, query_position_base) = if let Some(row) = query_row {
        if row >= seq {
            return Err(format!("QUERY_ROW {row} exceeds sequence length {seq}"));
        }
        q = q[row * q_width..(row + 1) * q_width].to_vec();
        oracle = oracle[row * q_width..(row + 1) * q_width].to_vec();
        (1, row)
    } else {
        (seq, 0)
    };

    let mut gpu = Gpu::init().map_err(|error| format!("GPU init: {error:?}"))?;
    let q_gpu = gpu
        .upload_f32(&q, &[q.len()])
        .map_err(|error| format!("upload Q: {error:?}"))?;
    let k_gpu = gpu
        .upload_f32(&k, &[k.len()])
        .map_err(|error| format!("upload K: {error:?}"))?;
    let v_gpu = gpu
        .upload_f32(&v, &[v.len()])
        .map_err(|error| format!("upload V: {error:?}"))?;
    let output = gpu
        .zeros(&[q.len()], DType::F32)
        .map_err(|error| format!("allocate output: {error:?}"))?;
    gpu.attention_dflash_wmma_bf16_causal_f32(
        &q_gpu,
        &k_gpu,
        &v_gpu,
        &output,
        batch,
        seq,
        n_heads,
        n_kv_heads,
        head_dim,
        scale,
        query_position_base,
    )
    .map_err(|error| format!("BF16 attention: {error:?}"))?;
    let candidate = gpu
        .download_f32(&output)
        .map_err(|error| format!("download output: {error:?}"))?;

    let mut max_abs = 0.0f32;
    let mut squared_error = 0.0f64;
    let mut squared_reference = 0.0f64;
    let mut exact = 0usize;
    for (&reference, &actual) in oracle.iter().zip(&candidate) {
        let error = (reference - actual).abs();
        max_abs = max_abs.max(error);
        squared_error += f64::from(error) * f64::from(error);
        squared_reference += f64::from(reference) * f64::from(reference);
        exact += usize::from(reference.to_bits() == actual.to_bits());
    }
    let nrmse = (squared_error / squared_reference.max(f64::MIN_POSITIVE)).sqrt();
    println!(
        "gemma4_sdpa_parity: seq={seq} batch={batch} query_base={query_position_base} heads={n_heads}/{n_kv_heads} head_dim={head_dim} scale={scale} max_abs={max_abs:.9} nrmse={nrmse:.9} exact={exact}/{}",
        oracle.len()
    );
    Ok(())
}
