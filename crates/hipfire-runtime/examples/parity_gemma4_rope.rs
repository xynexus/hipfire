// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors

//! Compare Hipfire's half-split RoPE with captured Transformers Gemma 4 values.
//!
//! The capture directory must contain `operator_{q,k}_norm.f32` and
//! `operator_{q,k}_rope.f32` in position-major layout, as emitted by
//! `benchmarks/gemma4/capture_layer0_sdpa.py`.
//!
//! Usage:
//! `parity_gemma4_rope DIR Q_HEADS KV_HEADS HEAD_DIM ROTARY_DIM BASIS_DIM THETA`

use hipfire_rdna::Gpu;
use std::fs;
use std::path::Path;

fn read_f32(path: &Path) -> Result<Vec<f32>, String> {
    let bytes = fs::read(path).map_err(|error| format!("{}: {error}", path.display()))?;
    if !bytes.len().is_multiple_of(4) {
        return Err(format!("{} has a partial F32 value", path.display()));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect())
}

fn parse_usize(value: &str, name: &str) -> Result<usize, String> {
    value
        .parse::<usize>()
        .map_err(|error| format!("invalid {name} {value:?}: {error}"))
}

fn round_bf16(value: f32) -> f32 {
    if !value.is_finite() {
        return value;
    }
    let bits = value.to_bits();
    let lsb = (bits >> 16) & 1;
    f32::from_bits(bits.wrapping_add(0x7fff + lsb) & 0xffff_0000)
}

fn report(label: &str, expected: &[f32], actual: &[f32]) {
    let mut raw_max = 0.0f32;
    let mut raw_error_sq = 0.0f64;
    let mut rounded_max = 0.0f32;
    let mut rounded_error_sq = 0.0f64;
    let mut reference_sq = 0.0f64;
    let mut rounded_exact = 0usize;
    for (&reference, &candidate) in expected.iter().zip(actual) {
        let raw_error = (reference - candidate).abs();
        raw_max = raw_max.max(raw_error);
        raw_error_sq += f64::from(raw_error).powi(2);

        let rounded = round_bf16(candidate);
        let rounded_error = (reference - rounded).abs();
        rounded_max = rounded_max.max(rounded_error);
        rounded_error_sq += f64::from(rounded_error).powi(2);
        rounded_exact += usize::from(reference.to_bits() == rounded.to_bits());
        reference_sq += f64::from(reference).powi(2);
    }
    println!(
        "{label}: raw_max_abs={raw_max:.9} raw_nrmse={:.9} \
         bf16_max_abs={rounded_max:.9} bf16_nrmse={:.9} bf16_exact={rounded_exact}/{}",
        (raw_error_sq / reference_sq).sqrt(),
        (rounded_error_sq / reference_sq).sqrt(),
        expected.len(),
    );
}

fn main() -> Result<(), String> {
    let args = std::env::args().collect::<Vec<_>>();
    if args.len() != 8 {
        return Err(
            "usage: parity_gemma4_rope DIR Q_HEADS KV_HEADS HEAD_DIM ROTARY_DIM BASIS_DIM THETA"
                .to_string(),
        );
    }
    let capture = Path::new(&args[1]);
    let q_heads = parse_usize(&args[2], "Q_HEADS")?;
    let kv_heads = parse_usize(&args[3], "KV_HEADS")?;
    let head_dim = parse_usize(&args[4], "HEAD_DIM")?;
    let rotary_dim = parse_usize(&args[5], "ROTARY_DIM")?;
    let basis_dim = parse_usize(&args[6], "BASIS_DIM")?;
    let theta = args[7]
        .parse::<f32>()
        .map_err(|error| format!("invalid THETA {:?}: {error}", args[7]))?;
    if rotary_dim == 0
        || rotary_dim > basis_dim
        || basis_dim > head_dim
        || !rotary_dim.is_multiple_of(2)
        || !basis_dim.is_multiple_of(2)
    {
        return Err(
            "RoPE dimensions must be nonzero even values with rotary <= basis <= head".into(),
        );
    }

    let q_input = read_f32(&capture.join("operator_q_norm.f32"))?;
    let k_input = read_f32(&capture.join("operator_k_norm.f32"))?;
    let q_expected = read_f32(&capture.join("operator_q_rope.f32"))?;
    let k_expected = read_f32(&capture.join("operator_k_rope.f32"))?;
    let q_width = q_heads * head_dim;
    let k_width = kv_heads * head_dim;
    if q_input.len() != q_expected.len()
        || k_input.len() != k_expected.len()
        || !q_input.len().is_multiple_of(q_width)
        || !k_input.len().is_multiple_of(k_width)
    {
        return Err("capture tensor lengths do not match the requested geometry".into());
    }
    let positions = q_input.len() / q_width;
    if positions == 0 || k_input.len() / k_width != positions {
        return Err("Q/K capture position counts differ or are empty".into());
    }

    let mut gpu = Gpu::init().map_err(|error| format!("GPU init: {error:?}"))?;
    let pos_buf = gpu
        .hip
        .malloc(std::mem::size_of::<i32>())
        .map_err(|error| format!("position allocation: {error:?}"))?;
    let mut q_actual = Vec::with_capacity(q_input.len());
    let mut k_actual = Vec::with_capacity(k_input.len());
    let mut q_staged = Vec::with_capacity(q_input.len());
    let mut k_staged = Vec::with_capacity(k_input.len());
    let mut cos_actual = Vec::with_capacity(positions * rotary_dim / 2);
    let mut sin_actual = Vec::with_capacity(positions * rotary_dim / 2);
    for position in 0..positions {
        let q_start = position * q_width;
        let k_start = position * k_width;
        let q = gpu
            .upload_owned_f32(&q_input[q_start..q_start + q_width], &[q_heads, head_dim])
            .map_err(|error| format!("Q upload at position {position}: {error:?}"))?;
        let k = gpu
            .upload_owned_f32(&k_input[k_start..k_start + k_width], &[kv_heads, head_dim])
            .map_err(|error| format!("K upload at position {position}: {error:?}"))?;
        let position_i32 = i32::try_from(position).map_err(|_| "position exceeds i32")?;
        gpu.hip
            .memcpy_htod(&pos_buf, &position_i32.to_ne_bytes())
            .map_err(|error| format!("position upload: {error:?}"))?;
        gpu.rope_partial_interleaved_f32(
            &q, &k, &pos_buf, q_heads, kv_heads, head_dim, rotary_dim, basis_dim, theta,
        )
        .map_err(|error| format!("RoPE at position {position}: {error:?}"))?;
        q_actual.extend(
            gpu.download_f32(&q)
                .map_err(|error| format!("Q download: {error:?}"))?,
        );
        k_actual.extend(
            gpu.download_f32(&k)
                .map_err(|error| format!("K download: {error:?}"))?,
        );

        let staged_q = gpu
            .upload_owned_f32(&q_input[q_start..q_start + q_width], &[q_heads, head_dim])
            .map_err(|error| format!("staged Q upload at position {position}: {error:?}"))?;
        let staged_k = gpu
            .upload_owned_f32(&k_input[k_start..k_start + k_width], &[kv_heads, head_dim])
            .map_err(|error| format!("staged K upload at position {position}: {error:?}"))?;
        gpu.rope_partial_halfsplit_bf16_staged_f32(
            &staged_q, &staged_k, &pos_buf, q_heads, kv_heads, head_dim, rotary_dim, basis_dim,
            theta,
        )
        .map_err(|error| format!("BF16-staged RoPE at position {position}: {error:?}"))?;
        q_staged.extend(
            gpu.download_f32(&staged_q)
                .map_err(|error| format!("staged Q download: {error:?}"))?,
        );
        k_staged.extend(
            gpu.download_f32(&staged_k)
                .map_err(|error| format!("staged K download: {error:?}"))?,
        );

        // A synthetic head with (first-half=1, partner-half=0) exposes the
        // kernel's generated cosine and sine values directly.
        let mut basis = vec![0.0f32; head_dim];
        basis[..rotary_dim / 2].fill(1.0);
        let table_q = gpu
            .upload_owned_f32(&basis, &[1, head_dim])
            .map_err(|error| format!("table Q upload: {error:?}"))?;
        let table_k = gpu
            .upload_owned_f32(&basis, &[1, head_dim])
            .map_err(|error| format!("table K upload: {error:?}"))?;
        gpu.rope_partial_interleaved_f32(
            &table_q, &table_k, &pos_buf, 1, 1, head_dim, rotary_dim, basis_dim, theta,
        )
        .map_err(|error| format!("RoPE table at position {position}: {error:?}"))?;
        let table = gpu
            .download_f32(&table_q)
            .map_err(|error| format!("table download: {error:?}"))?;
        cos_actual.extend_from_slice(&table[..rotary_dim / 2]);
        sin_actual.extend_from_slice(&table[basis_dim / 2..basis_dim / 2 + rotary_dim / 2]);
    }
    gpu.hip
        .free(pos_buf)
        .map_err(|error| format!("position free: {error:?}"))?;
    gpu.reclaim_pending();
    gpu.drain_pool();

    println!(
        "gemma4_rope_parity: positions={positions} heads={q_heads}/{kv_heads} \
         head_dim={head_dim} rotary_dim={rotary_dim} basis_dim={basis_dim} theta={theta}"
    );
    report("Q ordinary", &q_expected, &q_actual);
    report("K ordinary", &k_expected, &k_actual);
    report("Q BF16-staged", &q_expected, &q_staged);
    report("K BF16-staged", &k_expected, &k_staged);
    let cos_path = capture.join("operator_rope_cos.f32");
    let sin_path = capture.join("operator_rope_sin.f32");
    if cos_path.exists() && sin_path.exists() {
        let cos_full = read_f32(&cos_path)?;
        let sin_full = read_f32(&sin_path)?;
        if cos_full.len() != positions * head_dim || sin_full.len() != positions * head_dim {
            return Err("captured RoPE table lengths do not match position/head geometry".into());
        }
        let mut cos_expected = Vec::with_capacity(cos_actual.len());
        let mut sin_expected = Vec::with_capacity(sin_actual.len());
        for position in 0..positions {
            let start = position * head_dim;
            cos_expected.extend_from_slice(&cos_full[start..start + rotary_dim / 2]);
            sin_expected.extend_from_slice(&sin_full[start..start + rotary_dim / 2]);
        }
        report("cos table", &cos_expected, &cos_actual);
        report("sin table", &sin_expected, &sin_actual);
    }
    Ok(())
}
