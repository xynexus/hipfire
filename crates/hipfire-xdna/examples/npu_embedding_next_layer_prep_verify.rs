//! Exact hardware gate for R47 completed-state to R34 activation preparation.
//! Usage: `npu_embedding_next_layer_prep_verify CACHE [ITERS]`

#[cfg(target_os = "linux")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::time::Instant;

    use hipfire_primitives::conv::f32_to_bf16_bits;
    use hipfire_primitives::fwht::{cpu_fwht_256, gen_fwht_signs};
    use hipfire_xdna::NpuEmbeddingNextLayerPrepW8;

    const M: usize = 256;
    const PAD_M: usize = 288;
    const K: usize = 768;
    const GROUP: usize = 256;
    const GROUPS: usize = 3;
    const BLOCK: usize = 16_384;

    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if !(1..=2).contains(&args.len()) {
        return Err("usage: npu_embedding_next_layer_prep_verify CACHE [ITERS]".into());
    }
    let iterations = args
        .get(1)
        .map(|value| value.parse())
        .transpose()?
        .unwrap_or(10usize);

    let mut prep = NpuEmbeddingNextLayerPrepW8::load_cached(&args[0])?;
    let batch = prep.batch();

    let mut source = vec![0.0f32; batch * PAD_M * K];
    for document in 0..batch {
        for row in 0..M {
            for col in 0..K {
                source[(document * PAD_M + row) * K + col] =
                    ((row * 31 + col * 17) as f32 * 0.0031).sin() * 2.5
                        + ((row + col * 3) % 11) as f32 * 0.019;
            }
        }
    }
    let mut completed = vec![0u8; batch * PAD_M * K * 2 * size_of::<u16>()];
    let mut rounded = vec![0.0f32; batch * PAD_M * K];
    for row in 0..batch * PAD_M {
        let word_base = row * 2 * K;
        for col in 0..K {
            let value = source[row * K + col];
            let high = f32_to_bf16_bits(value);
            let high_f32 = f32::from_bits((high as u32) << 16);
            let low = f32_to_bf16_bits(value - high_f32);
            rounded[row * K + col] = high_f32 + f32::from_bits((low as u32) << 16);
            let high_offset = (word_base + col) * 2;
            let low_offset = (word_base + K + col) * 2;
            completed[high_offset..high_offset + 2].copy_from_slice(&high.to_le_bytes());
            completed[low_offset..low_offset + 2].copy_from_slice(&low.to_le_bytes());
        }
    }
    let input_norm = (0..K)
        .map(|col| 0.8 + (col % 29) as f32 * 0.011)
        .collect::<Vec<_>>();
    let awq = (0..K)
        .map(|col| 0.7 + (col % 23) as f32 * 0.017)
        .collect::<Vec<_>>();
    let signs1 = gen_fwht_signs(42, GROUP);
    let signs2 = gen_fwht_signs(1042, GROUP);

    let params = prep.upload_params(&input_norm, Some(&awq))?;
    prep.write_completed_bf16x2(&completed)?;
    prep.run_shared(&params)?;

    let output = prep.output_prefixes();
    let mut mismatches = 0usize;
    let mut mismatches_by_group = [0usize; GROUPS];
    let mut mismatches_by_owner = [0usize; 32];
    let mut negated_mismatches = 0usize;
    let mut first = None;
    let mut max_q_delta = 0i16;
    let mut max_scale_abs = 0.0f32;
    for document in 0..batch {
        for row in 0..M {
            let physical_row = document * PAD_M + row;
            let inverse = (rounded[physical_row * K..(physical_row + 1) * K]
                .iter()
                .map(|value| value * value)
                .sum::<f32>()
                / K as f32
                + 1.0e-6)
                .sqrt()
                .recip();
            for group in 0..GROUPS {
                let mut rotated = [0.0f32; GROUP];
                for inner in 0..GROUP {
                    let col = group * GROUP + inner;
                    rotated[inner] =
                        rounded[physical_row * K + col] * inverse * input_norm[col] / awq[col];
                }
                cpu_fwht_256(&mut rotated, &signs1, &signs2);
                let max_abs = rotated
                    .iter()
                    .fold(0.0f32, |value, item| value.max(item.abs()));
                let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 0.0 };

                let m_macro = row / 96;
                let within_macro = row % 96;
                let stripe = within_macro / 24;
                let within_stripe = within_macro % 24;
                let lm = within_stripe / 8;
                let local_row = within_stripe % 8;
                for n_macro in 0..5 {
                    let block = document * NpuEmbeddingNextLayerPrepW8::canonical_output_bytes()
                        + (stripe * 45 + (m_macro * 5 + n_macro) * 3 + group) * BLOCK;
                    let got_scale = f32::from_le_bytes(
                        output[block + 6144 + within_stripe * 4
                            ..block + 6144 + within_stripe * 4 + 4]
                            .try_into()?,
                    );
                    max_scale_abs = max_scale_abs.max((got_scale - scale).abs());
                    for inner in 0..GROUP {
                        let kt = inner / 8;
                        let kk = inner % 8;
                        let target = block + (lm * 32 + kt) * 64 + local_row * 8 + kk;
                        let got = output[target] as i8;
                        let expected = if scale > 0.0 {
                            (rotated[inner] / scale).round().clamp(-127.0, 127.0) as i8
                        } else {
                            0
                        };
                        if got != expected {
                            mismatches += 1;
                            mismatches_by_group[group] += 1;
                            mismatches_by_owner[row / 8] += 1;
                            negated_mismatches += usize::from(got == expected.saturating_neg());
                            max_q_delta = max_q_delta.max((got as i16 - expected as i16).abs());
                            first.get_or_insert((row, group, n_macro, inner, got, expected));
                        }
                    }
                }
            }
        }
    }
    if mismatches > 16 || max_q_delta > 1 || max_scale_abs > 1.0e-6 {
        let nonzero = output.iter().filter(|&&value| value != 0).count();
        eprintln!(
            "R47 debug nonzero={nonzero}/{} block0_q={:?} block0_scales={:?}",
            output.len(),
            &output[..32],
            &output[6144..6240],
        );
        eprintln!(
            "R47 debug mismatches_by_group={mismatches_by_group:?} mismatches_by_owner={mismatches_by_owner:?} negated_mismatches={negated_mismatches}"
        );
        return Err(format!(
            "R47 parity failed: mismatches={mismatches} max_q_delta={max_q_delta} max_scale_abs={max_scale_abs:.9} first={first:?}"
        )
        .into());
    }
    if batch > 1 {
        let document_bytes = NpuEmbeddingNextLayerPrepW8::canonical_output_bytes();
        if output[..document_bytes] != output[document_bytes..2 * document_bytes] {
            return Err("duplicated next-prep documents differ".into());
        }
    }

    for _ in 0..2 {
        prep.run_shared(&params)?;
    }
    let started = Instant::now();
    for _ in 0..iterations {
        prep.run_shared(&params)?;
    }
    let dispatch_ms = started.elapsed().as_secs_f64() * 1e3 / iterations as f64;
    println!(
        "embedding-next-layer-prep M={} K={K}: q_mismatches={mismatches} max_q_delta={max_q_delta} max_scale_abs={max_scale_abs:.9} dispatch_ms={dispatch_ms:.4}",
        M * batch
    );
    Ok(())
}

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("AIE2P next-layer prep verification is Linux-only");
}
