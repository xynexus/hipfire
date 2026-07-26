// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! Compare layer-0 Q projection with the pinned ROCm BF16 GEMM contract.
//!
//! Usage: `parity_bf16_linear_rocblas MODEL.hfq LAYER0_CAPTURE_DIR`

use hipfire_arch_gemma4::{load_dense_weights, Gemma4};
use hipfire_rdna::{DType, Gpu};
use hipfire_runtime::arch::Architecture;
use hipfire_runtime::hfq::HfqFile;
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

fn main() -> Result<(), String> {
    let args = std::env::args().collect::<Vec<_>>();
    if args.len() != 3 {
        return Err("usage: parity_bf16_linear_rocblas MODEL.hfq LAYER0_CAPTURE_DIR".to_string());
    }
    let model = Path::new(&args[1]);
    let capture = Path::new(&args[2]);
    let mut hfq = HfqFile::open(model).map_err(|error| error.to_string())?;
    if hfq.arch_id != Gemma4::arch_id() {
        return Err(format!(
            "expected Gemma 4 arch {}, got {}",
            Gemma4::arch_id(),
            hfq.arch_id
        ));
    }
    let config = Gemma4::config_from_hfq(&hfq)?;
    let mut gpu = Gpu::init_with_device(0).map_err(|error| format!("GPU init: {error:?}"))?;
    if gpu.rocblas.is_none() {
        return Err("rocBLAS did not load; set HIPFIRE_ROCBLAS_ALL_ARCHS=1".to_string());
    }
    let weights = load_dense_weights(&mut hfq, &mut gpu, &config)
        .map_err(|error| format!("Gemma 4 weights: {error:?}"))?;
    let input = read_f32(&capture.join("operator_input_norm.f32"))?;
    let oracle = read_f32(&capture.join("operator_q_proj.f32"))?;
    let wq = &weights.layers[0].wq;
    if wq.gpu_dtype != DType::BF16 || !input.len().is_multiple_of(wq.k) {
        return Err("layer-0 Q projection is not a compatible BF16 matrix".to_string());
    }
    let batch = input.len() / wq.k;
    if oracle.len() != batch * wq.m {
        return Err(format!(
            "oracle Q projection has {} values; expected {}",
            oracle.len(),
            batch * wq.m
        ));
    }

    let input_f32 = gpu
        .upload_f32(&input, &[batch * wq.k])
        .map_err(|error| format!("input upload: {error:?}"))?;
    let input_bf16 = gpu
        .alloc_tensor(&[batch * wq.k], DType::BF16)
        .map_err(|error| format!("input BF16 allocation: {error:?}"))?;
    let output = gpu
        .zeros(&[batch * wq.m], DType::BF16)
        .map_err(|error| format!("output allocation: {error:?}"))?;
    gpu.cast_f32_to_bf16(&input_f32, &input_bf16)
        .map_err(|error| format!("input BF16 cast: {error:?}"))?;
    gpu.rocblas_gemm_bf16_nt_bf16(&wq.buf, &input_bf16, &output, wq.m, batch, wq.k)
        .map_err(|error| format!("rocBLAS Q projection: {error:?}"))?;
    let output_bytes = gpu
        .download_raw(&output, batch * wq.m * 2)
        .map_err(|error| format!("output download: {error:?}"))?;
    let candidate = output_bytes
        .chunks_exact(2)
        .map(|bytes| f32::from_bits(u32::from(u16::from_ne_bytes(bytes.try_into().unwrap())) << 16))
        .collect::<Vec<_>>();

    let mut max_abs = 0.0f32;
    let mut error_sq = 0.0f64;
    let mut reference_sq = 0.0f64;
    let mut exact = 0usize;
    for (&expected, &actual) in oracle.iter().zip(&candidate) {
        let error = (expected - actual).abs();
        max_abs = max_abs.max(error);
        error_sq += f64::from(error).powi(2);
        reference_sq += f64::from(expected).powi(2);
        exact += usize::from(expected.to_bits() == actual.to_bits());
    }
    println!(
        "gemma4_bf16_qproj_rocblas: batch={batch} max_abs={max_abs:.9} nrmse={:.9} exact={exact}/{}",
        (error_sq / reference_sq).sqrt(),
        oracle.len()
    );
    Ok(())
}
