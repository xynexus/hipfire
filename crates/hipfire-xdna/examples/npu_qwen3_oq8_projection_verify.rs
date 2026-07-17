//! Live AIE2P parity for the resident Qwen3 OQ8+ projection primitive.

#[cfg(target_os = "linux")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use hipfire_primitives::conv::{bf16_bits_to_f32, f32_to_bf16_bits};
    use hipfire_xdna::{NpuQwen3Oq8Projection, OpusPackedMatrix};

    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.len() != 1 && args.len() != 4 {
        return Err("usage: npu_qwen3_oq8_projection_verify CACHE [M K N]".into());
    }
    let (m, k, n) = if args.len() == 4 {
        (
            args[1].parse::<usize>()?,
            args[2].parse::<usize>()?,
            args[3].parse::<usize>()?,
        )
    } else {
        (256, 256, 16)
    };
    let groups = k / 256;
    let mut payload = Vec::with_capacity(n * groups * 258);
    for output in 0..n {
        for group in 0..groups {
            payload.extend_from_slice(&0x3800u16.to_le_bytes()); // f16(0.5)
            for inner in 0..256 {
                payload.push((((inner * 7 + output * 11 + group * 13) % 23) as i8 - 11) as u8);
            }
        }
    }
    let awq = (0..k)
        .map(|inner| 0.75 + (inner % 31) as f32 * 0.013)
        .collect::<Vec<_>>();
    let matrix = OpusPackedMatrix::from_payload(35, k, n, &payload, Some(awq))?;
    let input = (0..m * k)
        .map(|index| {
            f32_to_bf16_bits(
                (index as f32 * 0.001_731).sin() * 0.31 + ((index / k) % 17) as f32 * 0.002,
            )
        })
        .collect::<Vec<_>>();
    let input_f32 = input
        .iter()
        .copied()
        .map(bf16_bits_to_f32)
        .collect::<Vec<_>>();
    let reference = matrix.reference_dequantized_bf16_f32(m, &input_f32)?;

    let xclbin = std::fs::read(format!("{}/final.xclbin", args[0]))?;
    let instructions = std::fs::read(format!("{}/insts.bin", args[0]))?;
    let mut projection = NpuQwen3Oq8Projection::load(&xclbin, &instructions, m, k, n, &matrix)?;
    let actual_bits = projection.run(&input)?;
    let repeated = projection.run(&input)?;
    let actual = actual_bits
        .iter()
        .copied()
        .map(bf16_bits_to_f32)
        .collect::<Vec<_>>();
    let (cosine, max_abs, mean_abs) = metrics(&actual, &reference);
    let repeated_f32 = repeated
        .iter()
        .copied()
        .map(bf16_bits_to_f32)
        .collect::<Vec<_>>();
    let repeated_metrics = metrics(&repeated_f32, &reference);
    if repeated != actual_bits {
        let first_difference = repeated
            .iter()
            .zip(&actual_bits)
            .position(|(repeated, first)| repeated != first)
            .unwrap_or(repeated.len().min(actual_bits.len()));
        return Err(format!(
            "OQ8 projection changed at value {first_difference}: first={cosine:.8}/{max_abs:.7}/{mean_abs:.8} repeated={:.8}/{:.7}/{:.8} values={:?}->{:?}",
            repeated_metrics.0,
            repeated_metrics.1,
            repeated_metrics.2,
            &actual_bits[first_difference..actual_bits.len().min(first_difference + 8)],
            &repeated[first_difference..repeated.len().min(first_difference + 8)]
        )
        .into());
    }
    let max_abs_limit = 0.25 + (groups.saturating_sub(1)) as f32 * 0.025;
    if !cosine.is_finite() || cosine < 0.99999 || max_abs > max_abs_limit {
        return Err(format!(
            "Qwen3 OQ8 projection parity failed: cosine={cosine:.8} max_abs={max_abs:.7}/{max_abs_limit:.7} mean_abs={mean_abs:.8}"
        )
        .into());
    }
    println!(
        "qwen3-oq8-projection M={m} K={k} N={n}: cosine={cosine:.8} max_abs={max_abs:.7} mean_abs={mean_abs:.8} repeat_stable=true"
    );
    Ok(())
}

#[cfg(target_os = "linux")]
fn metrics(actual: &[f32], reference: &[f32]) -> (f32, f32, f32) {
    let mut dot = 0.0f64;
    let mut actual_norm = 0.0f64;
    let mut reference_norm = 0.0f64;
    let mut max_abs = 0.0f32;
    let mut sum_abs = 0.0f64;
    for (&actual, &reference) in actual.iter().zip(reference) {
        dot += actual as f64 * reference as f64;
        actual_norm += (actual as f64).powi(2);
        reference_norm += (reference as f64).powi(2);
        let error = (actual - reference).abs();
        max_abs = max_abs.max(error);
        sum_abs += error as f64;
    }
    (
        (dot / (actual_norm.sqrt() * reference_norm.sqrt())) as f32,
        max_abs,
        (sum_abs / actual.len() as f64) as f32,
    )
}

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("npu_qwen3_oq8_projection_verify is Linux-only");
}
