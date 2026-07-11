//! Hardware parity check for the R35 canonical-BF16 resident dense-W8 FFN.
//!
//! Usage: `npu_resident_ffn_w8_canonical_verify [CACHE] [--iters N]`

#[cfg(target_os = "linux")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::time::Instant;

    use hipfire_primitives::conv::{bf16_bits_to_f32, f32_to_bf16_bits};
    use hipfire_xdna::{NpuResidentFfnDenseW8, NpuResidentFfnDenseW8IoMode, OpusPackedMatrix};

    const M: usize = 256;
    const K: usize = 768;
    const INTERMEDIATE: usize = 1152;
    const OUTPUT: usize = 768;

    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let cache = args
        .first()
        .filter(|value| !value.starts_with("--"))
        .cloned()
        .unwrap_or_else(|| {
            format!(
                "{}/.hipfire/npu/embgemma_aie2p_resident_ffn_dense_w8_canonical_bf16_m256_k768_i1152_o768",
                std::env::var("HOME").expect("HOME")
            )
        });
    let iterations = args
        .iter()
        .position(|value| value == "--iters")
        .and_then(|index| args.get(index + 1))
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(10);

    let gate = OpusPackedMatrix::from_payload(
        35,
        K,
        INTERMEDIATE,
        &w8_payload(K, INTERMEDIATE, 3, 0.0060),
        None,
    )?;
    let up = OpusPackedMatrix::from_payload(
        35,
        K,
        INTERMEDIATE,
        &w8_payload(K, INTERMEDIATE, 11, 0.0055),
        None,
    )?;
    let down = OpusPackedMatrix::from_payload(
        35,
        INTERMEDIATE,
        OUTPUT,
        &w8_payload(INTERMEDIATE, OUTPUT, 23, 0.0040),
        None,
    )?;

    let mut executor = NpuResidentFfnDenseW8::load_cached(&cache)?;
    if !matches!(
        executor.io_mode(),
        NpuResidentFfnDenseW8IoMode::CanonicalBf16
            | NpuResidentFfnDenseW8IoMode::CanonicalBf16Bf16x2Output
    ) {
        return Err("cache did not select the canonical-BF16 ABI".into());
    }
    let weights = executor.upload_weights(&gate, &up, &down)?;

    let input_bf16 = (0..M * K)
        .map(|index| {
            f32_to_bf16_bits(
                (index as f32 * 0.0037).sin() * 0.7 + (index % 19) as f32 * 0.009 - 0.08,
            )
        })
        .collect::<Vec<_>>();
    let input = input_bf16
        .iter()
        .copied()
        .map(bf16_bits_to_f32)
        .collect::<Vec<_>>();

    let reference_started = Instant::now();
    let gate_reference = gate.reference_f32(M, &input)?;
    let up_reference = up.reference_f32(M, &input)?;
    let intermediate = gate_reference
        .iter()
        .zip(&up_reference)
        .map(|(&gate, &up)| {
            let gelu =
                0.5 * gate * (1.0 + (0.797_884_6 * (gate + 0.044_715 * gate.powi(3))).tanh());
            bf16_bits_to_f32(f32_to_bf16_bits(gelu * up))
        })
        .collect::<Vec<_>>();
    let mut reference = down.reference_f32(M, &intermediate)?;
    if executor.io_mode() == NpuResidentFfnDenseW8IoMode::CanonicalBf16 {
        reference.iter_mut().for_each(|value| {
            *value = bf16_bits_to_f32(f32_to_bf16_bits(*value));
        });
    }
    let reference_ms = reference_started.elapsed().as_secs_f64() * 1e3;

    let output = executor.run_canonical_bf16(&weights, &input_bf16)?;
    let (cosine, max_abs, mean_abs) = metrics(&output, &reference);
    let max_reference = reference
        .iter()
        .fold(0.0f32, |maximum, value| maximum.max(value.abs()));
    let max_allowed = 0.02 + 0.03 * max_reference;
    if !cosine.is_finite() || cosine < 0.999 || max_abs > max_allowed {
        let got_intermediate = executor.read_canonical_intermediate_f32()?;
        let (intermediate_cosine, intermediate_max_abs, _) =
            metrics(&got_intermediate, &intermediate);
        eprintln!(
            "R35 intermediate: cosine={intermediate_cosine:.8} max_abs={intermediate_max_abs:.7}"
        );
        for index in [0, 1, OUTPUT - 1, OUTPUT, M * OUTPUT - 1] {
            eprintln!(
                "output[{index}] got={:.7} reference={:.7}",
                output[index], reference[index]
            );
        }
        return Err(format!(
            "R35 parity failed: cosine={cosine:.8} max_abs={max_abs:.7} allowed={max_allowed:.7}"
        )
        .into());
    }

    for _ in 0..2 {
        executor.run_canonical_bf16(&weights, &input_bf16)?;
    }
    let started = Instant::now();
    for _ in 0..iterations {
        executor.run_canonical_bf16(&weights, &input_bf16)?;
    }
    let dispatch_ms = started.elapsed().as_secs_f64() * 1e3 / iterations as f64;
    println!(
        "r35-canonical-bf16 M={M} K={K} I={INTERMEDIATE} O={OUTPUT}: cosine={cosine:.8} max_abs={max_abs:.7} mean_abs={mean_abs:.8} allowed={max_allowed:.7} reference_ms={reference_ms:.1} dispatch_ms={dispatch_ms:.4} iters={iterations}"
    );
    Ok(())
}

#[cfg(target_os = "linux")]
fn w8_payload(k: usize, n: usize, seed: usize, base_scale: f32) -> Vec<u8> {
    use hipfire_primitives::conv::f32_to_f16;

    const GROUP: usize = 256;
    const BLOCK: usize = 258;
    let groups = k.div_ceil(GROUP);
    let mut payload = vec![0u8; n * groups * BLOCK];
    for col in 0..n {
        for group in 0..groups {
            let block =
                &mut payload[(col * groups + group) * BLOCK..(col * groups + group + 1) * BLOCK];
            let scale = base_scale * (1.0 + ((col + 3 * group + seed) % 7) as f32 * 0.025);
            block[..2].copy_from_slice(&f32_to_f16(scale).to_le_bytes());
            for inner in 0..GROUP {
                let mixed = (inner as u64).wrapping_mul(0x9e37_79b1)
                    ^ (col as u64).wrapping_mul(0x85eb_ca77)
                    ^ (group as u64).wrapping_mul(0xc2b2_ae3d)
                    ^ (seed as u64).wrapping_mul(0x27d4_eb2f);
                let value = (mixed % 15) as i8 - 7;
                block[2 + inner] = value as u8;
            }
        }
    }
    payload
}

#[cfg(target_os = "linux")]
fn metrics(got: &[f32], expected: &[f32]) -> (f64, f32, f64) {
    let mut dot = 0.0;
    let mut got_norm = 0.0;
    let mut expected_norm = 0.0;
    let mut max_abs = 0.0f32;
    let mut sum_abs = 0.0;
    for (&got, &expected) in got.iter().zip(expected) {
        let error = (got - expected).abs();
        max_abs = max_abs.max(error);
        sum_abs += error as f64;
        dot += got as f64 * expected as f64;
        got_norm += (got as f64).powi(2);
        expected_norm += (expected as f64).powi(2);
    }
    (
        dot / (got_norm.sqrt() * expected_norm.sqrt()),
        max_abs,
        sum_abs / got.len() as f64,
    )
}

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("resident AIE2P FFN verification is Linux-only");
}
