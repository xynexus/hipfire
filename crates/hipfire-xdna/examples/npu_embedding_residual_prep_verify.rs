//! Hardware gate for R46 BF16x2 to padded R48 residual records.

#[cfg(target_os = "linux")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::time::Instant;

    use hipfire_primitives::conv::{bf16_bits_to_f32, f32_to_bf16_bits};
    use hipfire_xdna::NpuEmbeddingResidualPrep;

    const K: usize = 768;
    const ROW_BYTES: usize = 2 * K * size_of::<u16>();
    const RECORD_BYTES: usize = 16_384;

    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if !(1..=2).contains(&args.len()) {
        return Err("usage: npu_embedding_residual_prep_verify CACHE [ITERS]".into());
    }
    let iterations = args
        .get(1)
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(20);
    if iterations == 0 {
        return Err("residual prep verifier needs at least one iteration".into());
    }

    let mut completed = vec![0u8; NpuEmbeddingResidualPrep::completed_bytes()];
    for token in 0..256 {
        for hidden in 0..K {
            let high = f32_to_bf16_bits((token as f32 - 128.0) * 0.125 + hidden as f32 * 0.003);
            let low = f32_to_bf16_bits((hidden as f32 % 7.0 - 3.0) * 0.0005);
            let base = token * ROW_BYTES;
            completed[base + hidden * 2..base + hidden * 2 + 2]
                .copy_from_slice(&high.to_le_bytes());
            let low_offset = base + K * 2 + hidden * 2;
            completed[low_offset..low_offset + 2].copy_from_slice(&low.to_le_bytes());
        }
    }

    let mut prep = NpuEmbeddingResidualPrep::load_cached(&args[0])?;
    prep.write_bootstrap_bf16x2(&completed)?;
    prep.fill_output(0x5a)?;
    prep.run_bootstrap()?;

    let output = prep.output();
    if output[..NpuEmbeddingResidualPrep::activation_bytes()]
        .iter()
        .any(|&byte| byte != 0x5a)
    {
        return Err("residual prep overwrote the resident activation prefix".into());
    }
    let records = &output[NpuEmbeddingResidualPrep::activation_bytes()..];
    let mut mismatches = 0usize;
    let mut padding_nonzero = 0usize;
    for col in 0..8 {
        for core_row in 0..4 {
            let wave = col / 4;
            let active_col = core_row;
            let source_core_row = col % 4;
            let record = ((wave * 4 + active_col) * 4 + source_core_row) * RECORD_BYTES;
            let token_base = col * 32 + core_row * 8;
            for row in 0..8 {
                let source = (token_base + row) * ROW_BYTES;
                let target = record + row * K * 2;
                for hidden in 0..K {
                    let high = u16::from_le_bytes(
                        completed[source + hidden * 2..source + hidden * 2 + 2]
                            .try_into()
                            .unwrap(),
                    );
                    let low_offset = source + K * 2 + hidden * 2;
                    let low = u16::from_le_bytes(
                        completed[low_offset..low_offset + 2].try_into().unwrap(),
                    );
                    let expected = f32_to_bf16_bits(bf16_bits_to_f32(high) + bf16_bits_to_f32(low));
                    let got_offset = target + hidden * 2;
                    let got =
                        u16::from_le_bytes(records[got_offset..got_offset + 2].try_into().unwrap());
                    mismatches += usize::from(got != expected);
                }
            }
            padding_nonzero += records[record + 8 * K * 2..record + RECORD_BYTES]
                .iter()
                .filter(|&&byte| byte != 0)
                .count();
        }
    }
    if mismatches != 0 || padding_nonzero != 0 {
        return Err(format!(
            "residual prep parity failed: mismatches={mismatches} padding_nonzero={padding_nonzero}"
        )
        .into());
    }

    let started = Instant::now();
    for _ in 0..iterations {
        prep.run_bootstrap()?;
    }
    let dispatch_ms = started.elapsed().as_secs_f64() * 1e3 / iterations as f64;
    println!(
        "embedding-residual-prep M=256 K=768 records=32: mismatches=0 padding_nonzero=0 dispatch_ms={dispatch_ms:.4}"
    );
    Ok(())
}

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("npu_embedding_residual_prep_verify is Linux-only");
}
