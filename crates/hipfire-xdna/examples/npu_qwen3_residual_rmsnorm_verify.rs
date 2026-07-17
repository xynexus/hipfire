//! Live AIE2P parity for Qwen3 residual-add plus weighted RMSNorm.

#[cfg(target_os = "linux")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use hipfire_primitives::conv::bf16_bits_to_f32;
    use hipfire_xdna::NpuQwen3ResidualRmsNorm;

    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.len() != 3 {
        return Err("usage: npu_qwen3_residual_rmsnorm_verify CACHE M K".into());
    }
    let rows = args[1].parse::<usize>()?;
    let hidden = args[2].parse::<usize>()?;
    let residual = values(rows * hidden, 0.001_731, 0.31);
    let delta = values(rows * hidden, 0.002_117, 0.07);
    let weight = (0..hidden)
        .map(|index| 0.81 + (index % 37) as f32 * 0.007)
        .collect::<Vec<_>>();
    let epsilon = 1.0e-6;
    let (reference_completed, reference_normalized) =
        reference(&residual, &delta, &weight, epsilon, rows, hidden);

    let xclbin = std::fs::read(format!("{}/final.xclbin", args[0]))?;
    let instructions = std::fs::read(format!("{}/insts.bin", args[0]))?;
    let mut op =
        NpuQwen3ResidualRmsNorm::load(&xclbin, &instructions, rows, hidden, &weight, epsilon)?;
    let (completed, normalized) = op.run(&residual, &delta)?;
    let repeated = op.run(&residual, &delta)?;
    if repeated != (completed.clone(), normalized.clone()) {
        return Err("Qwen3 residual RMSNorm changed across repeated dispatches".into());
    }
    let completed_actual = completed
        .iter()
        .copied()
        .map(bf16_bits_to_f32)
        .collect::<Vec<_>>();
    let completed_expected = reference_completed
        .iter()
        .copied()
        .map(bf16_bits_to_f32)
        .collect::<Vec<_>>();
    let (completed_cosine, completed_max_abs) = metrics(&completed_actual, &completed_expected);
    let actual = normalized
        .iter()
        .copied()
        .map(bf16_bits_to_f32)
        .collect::<Vec<_>>();
    let expected = reference_normalized
        .iter()
        .copied()
        .map(bf16_bits_to_f32)
        .collect::<Vec<_>>();
    let (cosine, max_abs) = metrics(&actual, &expected);
    if !completed_cosine.is_finite()
        || completed_cosine < 0.99999
        || completed_max_abs > 0.004
        || !cosine.is_finite()
        || cosine < 0.99998
        || max_abs > 0.03
    {
        return Err(format!(
            "Qwen3 residual RMSNorm parity failed: completed={completed_cosine:.8}/{completed_max_abs:.7} normalized={cosine:.8}/{max_abs:.7}"
        )
        .into());
    }
    println!(
        "qwen3-residual-rmsnorm M={rows} K={hidden}: completed_cosine={completed_cosine:.8} completed_max_abs={completed_max_abs:.7} normalized_cosine={cosine:.8} normalized_max_abs={max_abs:.7} repeat_stable=true"
    );
    Ok(())
}

#[cfg(target_os = "linux")]
fn values(length: usize, frequency: f32, scale: f32) -> Vec<u16> {
    use hipfire_primitives::conv::f32_to_bf16_bits;
    (0..length)
        .map(|index| f32_to_bf16_bits((index as f32 * frequency).sin() * scale))
        .collect()
}

#[cfg(target_os = "linux")]
fn reference(
    residual: &[u16],
    delta: &[u16],
    weight: &[f32],
    epsilon: f32,
    rows: usize,
    hidden: usize,
) -> (Vec<u16>, Vec<u16>) {
    use hipfire_primitives::conv::{bf16_bits_to_f32, f32_to_bf16_bits};
    let mut completed = vec![0u16; rows * hidden];
    let mut normalized = vec![0u16; rows * hidden];
    for row in 0..rows {
        let base = row * hidden;
        let mut sum_sq = 0.0f32;
        for inner in 0..hidden {
            let value =
                bf16_bits_to_f32(residual[base + inner]) + bf16_bits_to_f32(delta[base + inner]);
            completed[base + inner] = f32_to_bf16_bits(value);
            let rounded = bf16_bits_to_f32(completed[base + inner]);
            sum_sq += rounded * rounded;
        }
        let inv_rms = 1.0 / (sum_sq / hidden as f32 + epsilon).sqrt();
        for inner in 0..hidden {
            normalized[base + inner] = f32_to_bf16_bits(
                bf16_bits_to_f32(completed[base + inner]) * weight[inner] * inv_rms,
            );
        }
    }
    (completed, normalized)
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
    eprintln!("npu_qwen3_residual_rmsnorm_verify is Linux-only");
}
