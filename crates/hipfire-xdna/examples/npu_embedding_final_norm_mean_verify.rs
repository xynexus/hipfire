//! Hardware parity and latency gate for resident final norm plus mean pooling.

#[cfg(target_os = "linux")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::time::Instant;

    use hipfire_primitives::conv::{bf16_bits_to_f32, f32_to_bf16_bits};
    use hipfire_xdna::NpuEmbeddingFinalNormMean;

    const ROWS: usize = 256;
    const K: usize = 768;
    const ROW_BYTES: usize = 2 * K * size_of::<u16>();
    const EPSILON: f32 = 1.0e-6;

    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if !(1..=2).contains(&args.len()) {
        return Err("usage: npu_embedding_final_norm_mean_verify CACHE [ITERS]".into());
    }
    let iterations = args
        .get(1)
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(20);
    if iterations == 0 {
        return Err("final norm/mean verifier needs at least one iteration".into());
    }

    let mut completed = vec![0u8; NpuEmbeddingFinalNormMean::completed_bytes()];
    let mut reconstructed = vec![0.0f32; ROWS * K];
    for row in 0..ROWS {
        for hidden in 0..K {
            let source = ((row * 131 + hidden * 17) % 509) as f32 / 64.0 - 4.0;
            let high = f32_to_bf16_bits(source);
            let high_f = bf16_bits_to_f32(high);
            let low = f32_to_bf16_bits(source - high_f);
            reconstructed[row * K + hidden] = high_f + bf16_bits_to_f32(low);
            let offset = row * ROW_BYTES + hidden * size_of::<u16>();
            completed[offset..offset + 2].copy_from_slice(&high.to_le_bytes());
            let low_offset = offset + K * size_of::<u16>();
            completed[low_offset..low_offset + 2].copy_from_slice(&low.to_le_bytes());
        }
    }
    let norm = (0..K)
        .map(|hidden| 0.75 + hidden as f32 / (2.0 * K as f32))
        .collect::<Vec<_>>();
    let mut expected = vec![0.0f32; K];
    for row in 0..ROWS {
        let values = &reconstructed[row * K..(row + 1) * K];
        let inverse = (values.iter().map(|value| value * value).sum::<f32>() / K as f32 + EPSILON)
            .sqrt()
            .recip();
        for hidden in 0..K {
            expected[hidden] += values[hidden] * norm[hidden] * inverse / ROWS as f32;
        }
    }

    let mut kernel = NpuEmbeddingFinalNormMean::load_cached(&args[0])?;
    let params = kernel.upload_params(&norm, EPSILON)?;
    kernel.write_completed_bf16x2(&completed)?;
    kernel.run_shared(&params)?;
    let got = kernel.read_pooled_f32();
    let max_abs = got
        .iter()
        .zip(&expected)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0f32, f32::max);
    let cosine = got
        .iter()
        .zip(&expected)
        .map(|(left, right)| left * right)
        .sum::<f32>()
        / (got.iter().map(|value| value * value).sum::<f32>().sqrt()
            * expected
                .iter()
                .map(|value| value * value)
                .sum::<f32>()
                .sqrt());
    if cosine < 0.999_999 || max_abs > 2.0e-5 {
        let nonzero = got.iter().filter(|value| **value != 0.0).count();
        let finite = got.iter().filter(|value| value.is_finite()).count();
        let min = got.iter().copied().fold(f32::INFINITY, f32::min);
        let max = got.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        return Err(format!(
            "final norm/mean parity failed: cosine={cosine:.8} max_abs={max_abs:.8} nonzero={nonzero} finite={finite} min={min:.8} max={max:.8}"
        )
        .into());
    }
    let started = Instant::now();
    for _ in 0..iterations {
        kernel.run_shared(&params)?;
    }
    let dispatch_ms = started.elapsed().as_secs_f64() * 1e3 / iterations as f64;
    println!(
        "embedding-final-norm-mean M=256 K=768: cosine={cosine:.8} max_abs={max_abs:.8} dispatch_ms={dispatch_ms:.4}"
    );
    Ok(())
}

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("npu_embedding_final_norm_mean_verify is Linux-only");
}
