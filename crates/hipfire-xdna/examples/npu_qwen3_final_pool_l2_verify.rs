//! Live AIE2P parity for Qwen3 final norm, last-token pool, and L2.

#[cfg(target_os = "linux")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use hipfire_primitives::conv::{bf16_bits_to_f32, f32_to_bf16_bits};
    use hipfire_xdna::NpuQwen3FinalPoolL2;

    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.len() != 4 {
        return Err("usage: npu_qwen3_final_pool_l2_verify CACHE BUCKET BATCH HIDDEN".into());
    }
    let bucket = args[1].parse::<usize>()?;
    let batch = args[2].parse::<usize>()?;
    let hidden_size = args[3].parse::<usize>()?;
    let hidden = (0..batch * bucket * hidden_size)
        .map(|index| f32_to_bf16_bits((index as f32 * 0.001_731).sin() * 0.31))
        .collect::<Vec<_>>();
    let lengths = (0..batch)
        .map(|document| bucket.saturating_sub(document * 19 + 1).max(1) as u32)
        .collect::<Vec<_>>();
    let weight = (0..hidden_size)
        .map(|index| 0.83 + (index % 31) as f32 * 0.008)
        .collect::<Vec<_>>();
    let epsilon = 1e-6;
    let reference = (0..batch)
        .map(|document| {
            let row = document * bucket + lengths[document] as usize - 1;
            let values = hidden[row * hidden_size..(row + 1) * hidden_size]
                .iter()
                .copied()
                .map(bf16_bits_to_f32)
                .collect::<Vec<_>>();
            let inv_rms = 1.0
                / (values.iter().map(|value| value * value).sum::<f32>() / hidden_size as f32
                    + epsilon)
                    .sqrt();
            let mut normalized = values
                .iter()
                .zip(&weight)
                .map(|(&value, &weight)| value * weight * inv_rms)
                .collect::<Vec<_>>();
            let inv_l2 = 1.0
                / normalized
                    .iter()
                    .map(|value| value * value)
                    .sum::<f32>()
                    .sqrt();
            for value in &mut normalized {
                *value *= inv_l2;
            }
            normalized
        })
        .collect::<Vec<_>>();
    let xclbin = std::fs::read(format!("{}/final.xclbin", args[0]))?;
    let instructions = std::fs::read(format!("{}/insts.bin", args[0]))?;
    let mut op = NpuQwen3FinalPoolL2::load(
        &xclbin,
        &instructions,
        bucket,
        batch,
        hidden_size,
        &weight,
        epsilon,
    )?;
    let actual = op.run(&hidden, &lengths)?;
    let repeated = op.run(&hidden, &lengths)?;
    if actual != repeated {
        return Err("Qwen3 final pool/L2 changed across repeated dispatches".into());
    }
    let mut minimum_cosine = 1.0f32;
    let mut max_abs = 0.0f32;
    for (actual, reference) in actual.iter().zip(&reference) {
        let metrics = metrics(actual, reference);
        minimum_cosine = minimum_cosine.min(metrics.0);
        max_abs = max_abs.max(metrics.1);
    }
    if minimum_cosine < 0.99999 || max_abs > 0.002 {
        return Err(format!(
            "Qwen3 final pool/L2 parity failed: min_cosine={minimum_cosine:.8} max_abs={max_abs:.7}"
        )
        .into());
    }
    println!(
        "qwen3-final-pool-l2 S={bucket} B={batch} K={hidden_size}: min_cosine={minimum_cosine:.8} max_abs={max_abs:.7} lengths={lengths:?} repeat_stable=true"
    );
    Ok(())
}

#[cfg(target_os = "linux")]
fn metrics(actual: &[f32], reference: &[f32]) -> (f32, f32) {
    let mut dot = 0.0f64;
    let mut actual_norm = 0.0f64;
    let mut reference_norm = 0.0f64;
    let mut max_abs = 0.0f32;
    for (&actual, &reference) in actual.iter().zip(reference) {
        dot += actual as f64 * reference as f64;
        actual_norm += (actual as f64).powi(2);
        reference_norm += (reference as f64).powi(2);
        max_abs = max_abs.max((actual - reference).abs());
    }
    (
        (dot / (actual_norm.sqrt() * reference_norm.sqrt())) as f32,
        max_abs,
    )
}

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("npu_qwen3_final_pool_l2_verify is Linux-only");
}
