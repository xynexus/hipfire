//! Verify one real compact mixed Opus HFQ tensor through paired NPU kernels.
//!
//! Usage:
//! `embeddinggemma_npu_opus_mixed MODEL.hfq TENSOR_NAME W4_CACHE SPARSE3_CACHE`

#[cfg(target_os = "linux")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use hipfire_primitives::conv::f16_to_f32;
    use hipfire_runtime::hfq::HfqFile;
    use hipfire_xdna::NpuOpusMixedGemmMp;
    use std::path::Path;

    let args: Vec<String> = std::env::args().skip(1).collect();
    if !(4..=5).contains(&args.len()) {
        return Err(
            "usage: embeddinggemma_npu_opus_mixed MODEL.hfq TENSOR_NAME W4_CACHE SPARSE3_CACHE [ROWS]"
                .into(),
        );
    }
    let hfq = HfqFile::open(Path::new(&args[0]))?;
    let tensor_name = &args[1];
    let (info, compact) = hfq
        .tensor_data_vec(tensor_name)
        .ok_or_else(|| format!("missing tensor {tensor_name}"))?;
    if info.quant_type != 36 || info.shape.len() != 2 {
        return Err(format!(
            "{tensor_name} must use the supported compact mixed Opus qt=36 rank-2 layout, got qt={} shape={:?}",
            info.quant_type, info.shape
        )
        .into());
    }
    let n = info.shape[0] as usize;
    let k = info.shape[1] as usize;
    let sidecar_name = tensor_name.strip_suffix(".weight").map_or_else(
        || format!("{tensor_name}.awq_scale.weight"),
        |stem| format!("{stem}.awq_scale.weight"),
    );
    let awq_scale = hfq.tensor_data_vec(&sidecar_name).map(|(sidecar, data)| {
        assert_eq!(sidecar.quant_type, 1, "AWQ sidecar must be F16");
        assert_eq!(data.len(), k * 2, "AWQ sidecar length");
        data.chunks_exact(2)
            .map(|bytes| f16_to_f32(u16::from_le_bytes([bytes[0], bytes[1]])))
            .collect::<Vec<_>>()
    });
    let mut gemm =
        NpuOpusMixedGemmMp::load_cached(&args[2], &args[3], k, n, &compact, awq_scale.clone())?;
    let m = args
        .get(4)
        .map(|rows| rows.parse::<usize>())
        .transpose()?
        .unwrap_or_else(|| gemm.rows_per_dispatch());
    if m % gemm.rows_per_dispatch() != 0 {
        return Err(format!("ROWS must be divisible by {}", gemm.rows_per_dispatch()).into());
    }
    let x: Vec<f32> = (0..m * k)
        .map(|index| ((index as f32 * 0.0097).sin() * 1.5) + (index % 13) as f32 * 0.01)
        .collect();
    let reference = gemm.reference_f32(m, &x)?;
    let mut output = vec![0.0f32; m * n];
    let started = std::time::Instant::now();
    gemm.run_f32(m, &x, &mut output)?;
    let elapsed_ms = started.elapsed().as_secs_f64() * 1e3;
    let mut mismatches = 0usize;
    let mut max_abs = 0.0f32;
    for (got, expected) in output.iter().zip(&reference) {
        let error = (got - expected).abs();
        max_abs = max_abs.max(error);
        if error > 1e-4 + expected.abs() * 1e-5 {
            mismatches += 1;
        }
    }
    println!(
        "model={} tensor={} variant={} M={} K={} N={} mismatches={} max_abs={:.6} elapsed_ms={:.3}",
        args[0],
        tensor_name,
        if awq_scale.is_some() {
            "mixed-opus-awq"
        } else {
            "mixed-opus"
        },
        m,
        k,
        n,
        mismatches,
        max_abs,
        elapsed_ms
    );
    if mismatches != 0 {
        return Err("real HFQ mixed Opus NPU parity failed".into());
    }
    Ok(())
}

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("amdxdna is Linux-only");
}
