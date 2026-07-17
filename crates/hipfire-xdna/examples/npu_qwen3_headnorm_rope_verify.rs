//! Live AIE2P parity for segmented Qwen3 Q/K head RMSNorm plus RoPE.

#[cfg(target_os = "linux")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use hipfire_primitives::conv::bf16_bits_to_f32;
    use hipfire_xdna::{NpuQwen3HeadNormRope, Qwen3HeadNormRopeGeometry};

    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.len() != 3 {
        return Err("usage: npu_qwen3_headnorm_rope_verify CACHE BUCKET BATCH".into());
    }
    let geometry = Qwen3HeadNormRopeGeometry {
        sequence_bucket: args[1].parse()?,
        dispatch_batch: args[2].parse()?,
        query_heads: 16,
        kv_heads: 8,
        head_dim: 128,
    }
    .validate()?;
    let rows = geometry.sequence_bucket * geometry.dispatch_batch;
    let query = values(
        rows * geometry.query_heads * geometry.head_dim,
        0.001_731,
        0.31,
    );
    let key = values(
        rows * geometry.kv_heads * geometry.head_dim,
        0.002_117,
        0.27,
    );
    let query_weight = (0..geometry.head_dim)
        .map(|index| 0.83 + (index % 29) as f32 * 0.009)
        .collect::<Vec<_>>();
    let key_weight = (0..geometry.head_dim)
        .map(|index| 0.79 + (index % 31) as f32 * 0.008)
        .collect::<Vec<_>>();
    let theta = 1_000_000.0;
    let epsilon = 1.0e-6;
    let reference_query = reference(
        &query,
        geometry.query_heads,
        geometry,
        &query_weight,
        theta,
        epsilon,
    );
    let reference_key = reference(
        &key,
        geometry.kv_heads,
        geometry,
        &key_weight,
        theta,
        epsilon,
    );
    let xclbin = std::fs::read(format!("{}/final.xclbin", args[0]))?;
    let instructions = std::fs::read(format!("{}/insts.bin", args[0]))?;
    let mut op = NpuQwen3HeadNormRope::load(
        &xclbin,
        &instructions,
        geometry,
        &query_weight,
        &key_weight,
        theta,
        epsilon,
    )?;
    let (actual_query, actual_key) = op.run(&query, &key)?;
    let repeated = op.run(&query, &key)?;
    if repeated != (actual_query.clone(), actual_key.clone()) {
        return Err("Qwen3 headnorm/RoPE changed across repeated dispatches".into());
    }
    let q_metrics = metrics(&actual_query, &reference_query);
    let k_metrics = metrics(&actual_key, &reference_key);
    if q_metrics.0 < 0.9999 || q_metrics.1 > 0.04 || k_metrics.0 < 0.9999 || k_metrics.1 > 0.04 {
        return Err(format!(
            "Qwen3 headnorm/RoPE parity failed: Q={:.8}/{:.7} K={:.8}/{:.7}",
            q_metrics.0, q_metrics.1, k_metrics.0, k_metrics.1
        )
        .into());
    }
    let q_actual = actual_query
        .iter()
        .copied()
        .map(bf16_bits_to_f32)
        .collect::<Vec<_>>();
    let k_actual = actual_key
        .iter()
        .copied()
        .map(bf16_bits_to_f32)
        .collect::<Vec<_>>();
    if q_actual.iter().any(|value| !value.is_finite())
        || k_actual.iter().any(|value| !value.is_finite())
    {
        return Err("Qwen3 headnorm/RoPE produced non-finite values".into());
    }
    println!(
        "qwen3-headnorm-rope S={} B={}: q_cosine={:.8} q_max_abs={:.7} k_cosine={:.8} k_max_abs={:.7} repeat_stable=true",
        geometry.sequence_bucket, geometry.dispatch_batch, q_metrics.0, q_metrics.1, k_metrics.0, k_metrics.1
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
    input: &[u16],
    heads: usize,
    geometry: hipfire_xdna::Qwen3HeadNormRopeGeometry,
    weight: &[f32],
    theta: f32,
    epsilon: f32,
) -> Vec<u16> {
    use hipfire_primitives::conv::{bf16_bits_to_f32, f32_to_bf16_bits};
    let rows = geometry.sequence_bucket * geometry.dispatch_batch;
    let dim = geometry.head_dim;
    let half = dim / 2;
    let weight = weight
        .iter()
        .map(|&value| bf16_bits_to_f32(f32_to_bf16_bits(value)))
        .collect::<Vec<_>>();
    let mut output = vec![0u16; input.len()];
    for row in 0..rows {
        let position = row % geometry.sequence_bucket;
        for head in 0..heads {
            let base = (row * heads + head) * dim;
            let sum_sq = input[base..base + dim]
                .iter()
                .map(|&value| bf16_bits_to_f32(value).powi(2))
                .sum::<f32>();
            let inv = 1.0 / (sum_sq / dim as f32 + epsilon).sqrt();
            for inner in 0..half {
                let frequency = theta.powf(-((2 * inner) as f32) / dim as f32);
                let angle = position as f32 * frequency;
                let cosine = bf16_bits_to_f32(f32_to_bf16_bits(angle.cos()));
                let sine = bf16_bits_to_f32(f32_to_bf16_bits(angle.sin()));
                let x = bf16_bits_to_f32(f32_to_bf16_bits(
                    bf16_bits_to_f32(input[base + inner]) * inv,
                ));
                let y = bf16_bits_to_f32(f32_to_bf16_bits(
                    bf16_bits_to_f32(input[base + half + inner]) * inv,
                ));
                let x = bf16_bits_to_f32(f32_to_bf16_bits(x * weight[inner]));
                let y = bf16_bits_to_f32(f32_to_bf16_bits(y * weight[half + inner]));
                output[base + inner] = f32_to_bf16_bits(x * cosine - y * sine);
                output[base + half + inner] = f32_to_bf16_bits(y * cosine + x * sine);
            }
        }
    }
    output
}

#[cfg(target_os = "linux")]
fn metrics(actual: &[u16], reference: &[u16]) -> (f32, f32) {
    use hipfire_primitives::conv::bf16_bits_to_f32;
    let mut dot = 0.0f64;
    let mut actual_norm = 0.0f64;
    let mut reference_norm = 0.0f64;
    let mut max_abs = 0.0f32;
    for (&actual, &reference) in actual.iter().zip(reference) {
        let actual = bf16_bits_to_f32(actual);
        let reference = bf16_bits_to_f32(reference);
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
    eprintln!("npu_qwen3_headnorm_rope_verify is Linux-only");
}
