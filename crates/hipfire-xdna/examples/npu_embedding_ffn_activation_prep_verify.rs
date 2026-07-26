//! Hardware byte-oracle and sustained timing gate for the R93 BF16-to-R25 producer.
//! Usage: `npu_embedding_ffn_activation_prep_verify CACHE [ITERS]`

#[cfg(target_os = "linux")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::time::Instant;

    use hipfire_primitives::conv::f32_to_bf16_bits;
    use hipfire_primitives::fwht::{cpu_fwht_256, gen_fwht_signs};
    use hipfire_xdna::NpuEmbeddingFfnActivationPrepW4;

    const M: usize = 256;
    const K: usize = 768;
    const GROUP: usize = 256;
    const GROUPS: usize = 3;
    const BLOCK: usize = 6_656;
    const PREFIX: usize = 6_240;

    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if !(1..=2).contains(&args.len()) {
        return Err("usage: npu_embedding_ffn_activation_prep_verify CACHE [ITERS]".into());
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
            let value = ((row * 31 + col * 17) as f32 * 0.0031).sin() * 2.5
                + ((row + col * 3) % 11) as f32 * 0.019;
            f32_to_bf16_bits(value)
        })
        .collect::<Vec<_>>();
    let rounded = input
        .iter()
        .map(|bits| f32::from_bits((*bits as u32) << 16))
        .collect::<Vec<_>>();
    let awq = (0..K)
        .map(|col| 0.7 + (col % 23) as f32 * 0.017)
        .collect::<Vec<_>>();
    let signs1 = gen_fwht_signs(42, GROUP);
    let signs2 = gen_fwht_signs(1042, GROUP);

    let mut prep = NpuEmbeddingFfnActivationPrepW4::load_cached(&args[0])?;
    let params = prep.upload_params(Some(&awq))?;
    prep.write_input_bf16(&input)?;
    prep.run_shared(&params)?;

    let output = prep.output();
    if let Some(path) = std::env::var_os("HIPFIRE_R93_DUMP_OUTPUT") {
        std::fs::write(path, output)?;
    }
    if std::env::var_os("HIPFIRE_R93_HASHES").is_some() {
        let fnv = |bytes: &[u8]| {
            bytes.iter().fold(2166136261u32, |hash, &byte| {
                (hash ^ byte as u32).wrapping_mul(16777619)
            })
        };
        eprintln!(
            "R93 prefix hashes={:?}",
            output
                .chunks_exact(BLOCK)
                .map(|block| fnv(&block[..PREFIX]))
                .collect::<Vec<_>>()
        );
    }
    let mut mismatches = 0usize;
    let mut first = None;
    let mut max_q_delta = 0i16;
    let mut max_scale_abs = 0.0f32;
    for row in 0..M {
        for group in 0..GROUPS {
            let mut rotated = [0.0f32; GROUP];
            for inner in 0..GROUP {
                let col = group * GROUP + inner;
                rotated[inner] = rounded[row * K + col] / awq[col];
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
            let local_m = within_stripe / 4;
            let local_row = within_stripe % 4;
            for n_macro in 0..3 {
                let block = (stripe * 27 + (m_macro * 3 + n_macro) * 3 + group) * BLOCK;
                let got_scale = f32::from_le_bytes(
                    output[block + 6144 + within_stripe * 4..block + 6144 + within_stripe * 4 + 4]
                        .try_into()?,
                );
                max_scale_abs = max_scale_abs.max((got_scale - scale).abs());
                for inner in 0..GROUP {
                    let kt = inner / 16;
                    let kk = inner % 16;
                    let target = block + (local_m * 16 + kt) * 64 + local_row * 16 + kk;
                    let got = output[target] as i8;
                    let expected = if scale > 0.0 {
                        (rotated[inner] / scale).round().clamp(-127.0, 127.0) as i8
                    } else {
                        0
                    };
                    if got != expected {
                        mismatches += 1;
                        max_q_delta = max_q_delta.max((got as i16 - expected as i16).abs());
                        first.get_or_insert((row, group, n_macro, inner, got, expected));
                    }
                }
            }
        }
    }

    let mut padding_nonzero = 0usize;
    for block in output.chunks_exact(BLOCK) {
        padding_nonzero += block[PREFIX..].iter().filter(|&&value| value != 0).count();
    }
    // Rows 256..287 occupy m_macro=2, stripes 2 and 3 and must remain untouched.
    let mut padded_prefix_nonzero = 0usize;
    for stripe in 3..4 {
        for block in 18..27 {
            let base = (stripe * 27 + block) * BLOCK;
            padded_prefix_nonzero += output[base..base + PREFIX]
                .iter()
                .filter(|&&value| value != 0)
                .count();
        }
    }
    for n_macro in 0..3 {
        for group in 0..GROUPS {
            let base = (2 * 27 + (2 * 3 + n_macro) * 3 + group) * BLOCK;
            padded_prefix_nonzero += output[base + 4096..base + 6144]
                .iter()
                .filter(|&&value| value != 0)
                .count();
            padded_prefix_nonzero += output[base + 6144 + 64..base + PREFIX]
                .iter()
                .filter(|&&value| value != 0)
                .count();
        }
    }
    if mismatches > 16
        || max_q_delta > 1
        || max_scale_abs > 1.0e-6
        || padding_nonzero != 0
        || padded_prefix_nonzero != 0
    {
        let nonzero = output.iter().filter(|&&value| value != 0).count();
        let first_nonzero = output
            .iter()
            .enumerate()
            .filter(|(_, value)| **value != 0)
            .take(24)
            .map(|(offset, value)| (offset, *value))
            .collect::<Vec<_>>();
        eprintln!(
            "R93 debug nonzero={nonzero}/{} first_nonzero={first_nonzero:?} block0_q={:?} block0_scales={:?}",
            output.len(),
            &output[..32],
            &output[6144..6240],
        );
        return Err(format!(
            "R93 parity failed: mismatches={mismatches} max_q_delta={max_q_delta} max_scale_abs={max_scale_abs:.9} padding_nonzero={padding_nonzero} padded_prefix_nonzero={padded_prefix_nonzero} first={first:?}"
        )
        .into());
    }

    for _ in 0..2 {
        prep.run_shared(&params)?;
    }
    let started = Instant::now();
    for _ in 0..iterations {
        prep.run_shared(&params)?;
    }
    let dispatch_ms = started.elapsed().as_secs_f64() * 1e3 / iterations as f64;
    let source_gib_s =
        (M * K * size_of::<u16>()) as f64 / (1u64 << 30) as f64 / (dispatch_ms * 1e-3);
    let physical_gib_s = (NpuEmbeddingFfnActivationPrepW4::input_bytes()
        + NpuEmbeddingFfnActivationPrepW4::output_bytes()) as f64
        / (1u64 << 30) as f64
        / (dispatch_ms * 1e-3);
    println!(
        "embedding-ffn-activation-prep M={M} K={K}: q_mismatches={mismatches} max_q_delta={max_q_delta} max_scale_abs={max_scale_abs:.9} padding_nonzero={padding_nonzero} dispatch_ms={dispatch_ms:.4} source_gib_s={source_gib_s:.3} physical_gib_s={physical_gib_s:.3}"
    );
    Ok(())
}

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("AIE2P FFN activation-prep verification is Linux-only");
}
