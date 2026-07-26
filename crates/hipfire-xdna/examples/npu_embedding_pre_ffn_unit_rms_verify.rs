//! R105 hardware numerical oracle and sustained timing probe.
//! Usage: `npu_embedding_pre_ffn_unit_rms_verify CACHE [ITERS]`

#[cfg(target_os = "linux")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::time::Instant;

    use hipfire_primitives::conv::f32_to_bf16_bits;
    use hipfire_xdna::NpuEmbeddingPreFfnUnitRms;

    const M: usize = 256;
    const K: usize = 768;
    const EPSILON: f32 = 1.0e-6;

    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if !(1..=2).contains(&args.len()) {
        return Err("usage: npu_embedding_pre_ffn_unit_rms_verify CACHE [ITERS]".into());
    }
    let iterations = args
        .get(1)
        .map(|value| value.parse())
        .transpose()?
        .unwrap_or(100usize);

    let input = (0..M * K)
        .map(|index| {
            let row = index / K;
            let col = index % K;
            let value = ((row * 29 + col * 13) as f32 * 0.0047).sin() * 3.25
                + ((row * 7 + col) % 17) as f32 * 0.013;
            f32_to_bf16_bits(value)
        })
        .collect::<Vec<_>>();
    let rounded = input
        .iter()
        .map(|bits| f32::from_bits((*bits as u32) << 16))
        .collect::<Vec<_>>();
    let mut expected = vec![0u16; M * K];
    for row in 0..M {
        let values = &rounded[row * K..(row + 1) * K];
        let sum = values.iter().map(|value| value * value).sum::<f32>();
        let inverse = (sum / K as f32 + EPSILON).sqrt().recip();
        for col in 0..K {
            expected[row * K + col] = f32_to_bf16_bits(values[col] * inverse);
        }
    }

    let mut kernel = NpuEmbeddingPreFfnUnitRms::load_cached(&args[0])?;
    kernel.write_direct_x_bf16(&input)?;
    kernel.run_shared()?;
    let output = kernel.read_output_bf16()?;

    let mut mismatches = 0usize;
    let mut max_abs = 0.0f32;
    let mut dot = 0.0f64;
    let mut norm_got = 0.0f64;
    let mut norm_expected = 0.0f64;
    let mut first = None;
    for (index, (&got_bits, &expected_bits)) in output.iter().zip(&expected).enumerate() {
        let got = f32::from_bits((got_bits as u32) << 16);
        let want = f32::from_bits((expected_bits as u32) << 16);
        if got_bits != expected_bits {
            mismatches += 1;
            first.get_or_insert((index / K, index % K, got, want));
        }
        max_abs = max_abs.max((got - want).abs());
        dot += got as f64 * want as f64;
        norm_got += got as f64 * got as f64;
        norm_expected += want as f64 * want as f64;
    }
    let cosine = dot / (norm_got.sqrt() * norm_expected.sqrt()).max(f64::MIN_POSITIVE);
    if cosine < 0.99999 || max_abs > 0.03125 {
        return Err(format!(
            "R105 parity failed: mismatches={mismatches} cosine={cosine:.8} max_abs={max_abs:.7} first={first:?}"
        )
        .into());
    }

    for _ in 0..2 {
        kernel.run_shared()?;
    }
    let started = Instant::now();
    for _ in 0..iterations {
        kernel.run_shared()?;
    }
    let dispatch_ms = started.elapsed().as_secs_f64() * 1e3 / iterations as f64;
    let physical_gib_s = (2 * NpuEmbeddingPreFfnUnitRms::direct_x_bytes()) as f64
        / (1u64 << 30) as f64
        / (dispatch_ms * 1e-3);
    let rows_per_s = M as f64 / (dispatch_ms * 1e-3);
    println!(
        "embedding-pre-ffn-unit-rms M={M} K={K}: mismatches={mismatches} cosine={cosine:.8} max_abs={max_abs:.7} dispatch_ms={dispatch_ms:.4} rows_per_s={rows_per_s:.1} physical_gib_s={physical_gib_s:.3}"
    );
    Ok(())
}

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("AIE2P pre-FFN unit-RMS verification is Linux-only");
}
