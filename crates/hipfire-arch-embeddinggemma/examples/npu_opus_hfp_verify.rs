//! Verify a real Opus tensor (or concatenated roles) through a packed HFP file.
//!
//! Usage:
//! `npu_opus_hfp_verify MODEL.hfq WHOLE_SCALED_CACHE WEIGHTS.rdna2.hfp \
//!     TENSOR [TENSOR ...] [--iters N] [--fullk-cols N]`

#[cfg(target_os = "linux")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::path::Path;
    use std::time::Instant;

    use hipfire_runtime::hfq::HfqFile;
    use hipfire_xdna::{NpuOpusExecutor, OpusMatrixEncoding};

    let arguments = std::env::args().skip(1).collect::<Vec<_>>();
    if arguments.len() < 4 {
        return Err(
            "usage: npu_opus_hfp_verify MODEL.hfq CACHE WEIGHTS.rdna2.hfp TENSOR [TENSOR ...] [--iters N] [--fullk-cols N]"
                .into(),
        );
    }
    let option = |name: &str| {
        arguments
            .iter()
            .position(|argument| argument == name)
            .and_then(|index| arguments.get(index + 1))
    };
    let iterations = option("--iters")
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(0);
    let option_start = arguments
        .iter()
        .position(|argument| argument.starts_with("--"))
        .unwrap_or(arguments.len());
    let tensor_names = &arguments[3..option_start];
    if tensor_names.is_empty() {
        return Err("at least one tensor role is required".into());
    }

    let hfq = HfqFile::open(Path::new(&arguments[0]))?;
    let mut quant_type = None;
    let mut k = None;
    let mut n = 0usize;
    let mut payload = Vec::new();
    let mut shared_awq: Option<Option<Vec<f32>>> = None;
    for name in tensor_names {
        let (info, bytes) = hfq
            .tensor_data_vec(name)
            .ok_or_else(|| format!("missing tensor {name}"))?;
        if info.shape.len() != 2 {
            return Err(format!("{name} must be rank two, got {:?}", info.shape).into());
        }
        let matrix_n = info.shape[0] as usize;
        let matrix_k = info.shape[1] as usize;
        let matrix_quant_type = info.quant_type;
        if quant_type.is_some_and(|value| value != matrix_quant_type)
            || k.is_some_and(|value| value != matrix_k)
        {
            return Err("concatenated tensors must share quant type and K".into());
        }
        quant_type = Some(matrix_quant_type);
        k = Some(matrix_k);
        n += matrix_n;
        let awq = load_awq_scale(&hfq, name, matrix_k)?;
        if shared_awq.as_ref().is_some_and(|value| value != &awq) {
            return Err("concatenated tensors must share one AWQ sidecar".into());
        }
        shared_awq.get_or_insert_with(|| awq.clone());
        payload.extend_from_slice(&bytes);
    }
    let quant_type = quant_type.expect("non-empty role list");
    let k = k.expect("non-empty role list");
    let encoding = OpusMatrixEncoding::classify(quant_type, payload.len(), k, n)?;

    let fullk_cols = option("--fullk-cols")
        .map(|value| value.parse::<usize>())
        .transpose()?;
    let mut executor = if let Some(cols) = fullk_cols {
        NpuOpusExecutor::load_fullk_cached(&[(&arguments[1], cols)], n)?
    } else {
        NpuOpusExecutor::load_whole_scaled_cached(&[&arguments[1]], n)?
    };
    let matrix = executor.pack_matrix_prepacked(
        quant_type,
        k,
        n,
        &payload,
        shared_awq.unwrap_or(None),
        Path::new(&arguments[2]),
    )?;
    let m = 256usize;
    let input = (0..m * k)
        .map(|index| ((index as f32 * 0.013).sin() * 2.0) + ((index % 7) as f32 - 3.0) * 0.1)
        .collect::<Vec<_>>();
    let reference = executor.reference_f32(&matrix, m, &input)?;
    let mut output = vec![0.0f32; m * n];
    executor.run_f32(&matrix, m, &input, &mut output)?;

    let mut mismatches = 0usize;
    let mut max_abs = 0.0f32;
    let mut first = None;
    for (index, (&actual, &expected)) in output.iter().zip(&reference).enumerate() {
        let error = (actual - expected).abs();
        max_abs = max_abs.max(error);
        if error > 1e-4 + expected.abs() * 1e-5 {
            mismatches += 1;
            first.get_or_insert((index, actual, expected, error));
        }
    }
    println!(
        "opus-hfp qt={quant_type} encoding={encoding:?} roles={} M={m} K={k} N={n}: mismatches={mismatches} max_abs={max_abs:.7}",
        tensor_names.join(",")
    );
    if let Some((index, actual, expected, error)) = first {
        println!(
            "first_mismatch index={index} actual={actual:.7} expected={expected:.7} abs={error:.7}"
        );
    }
    if mismatches != 0 {
        return Err("real Opus HFP parity failed".into());
    }

    if iterations > 0 {
        for _ in 0..2 {
            executor.run_f32(&matrix, m, &input, &mut output)?;
        }
        let started = Instant::now();
        for _ in 0..iterations {
            executor.run_f32(&matrix, m, &input, &mut output)?;
        }
        let seconds = started.elapsed().as_secs_f64() / iterations as f64;
        println!(
            "iters={iterations} wrapper_ms={:.4} logical_tops={:.4}",
            seconds * 1e3,
            2.0 * m as f64 * k as f64 * n as f64 / seconds / 1e12
        );
    }
    Ok(())
}

#[cfg(target_os = "linux")]
fn load_awq_scale(
    hfq: &hipfire_runtime::hfq::HfqFile,
    name: &str,
    k: usize,
) -> Result<Option<Vec<f32>>, Box<dyn std::error::Error>> {
    use hipfire_runtime::quant::f16_to_f32;

    let sidecar = name.strip_suffix(".weight").map_or_else(
        || format!("{name}.awq_scale.weight"),
        |stem| format!("{stem}.awq_scale.weight"),
    );
    let Some((info, bytes)) = hfq.tensor_data_vec(&sidecar) else {
        return Ok(None);
    };
    if info.quant_type != 1 || bytes.len() != k * 2 {
        return Err(format!("{sidecar} must be f16[{k}]").into());
    }
    Ok(Some(
        bytes
            .chunks_exact(2)
            .map(|pair| f16_to_f32(u16::from_le_bytes([pair[0], pair[1]])))
            .collect(),
    ))
}

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("amdxdna is Linux-only");
}
