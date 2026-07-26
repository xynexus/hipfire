//! Verify the document-padded direct-X resident FFN batch contract.
//!
//! A canonical M256 run provides an absolute reference. The direct-X M256
//! cache must match it after applying the same BF16 RMSNorm, and an M512 run
//! over `[X; X]` must reproduce the direct-X M256 result for both documents.
//!
//! Usage: `npu_resident_ffn_w8_direct_batched_verify [CANONICAL_M256] [DIRECT_M256] [DIRECT_M512]`

#[cfg(target_os = "linux")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use hipfire_primitives::conv::{bf16_bits_to_f32, f32_to_bf16_bits};
    use hipfire_xdna::{NpuResidentFfnDenseW8, NpuResidentFfnDenseW8IoMode, OpusPackedMatrix};

    const M: usize = 256;
    const K: usize = 768;
    const INTERMEDIATE: usize = 1152;
    const OUTPUT: usize = 768;
    const EPSILON: f32 = 1.0e-6;

    let home = std::env::var("HOME").expect("HOME");
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let canonical_cache = args.first().cloned().unwrap_or_else(|| {
        format!("{home}/.hipfire/npu/embgemma_aie2p_resident_ffn_dense_w8_canonical_bf16x2_m256_k768_i1152_o768")
    });
    let direct256_cache = args.get(1).cloned().unwrap_or_else(|| {
        format!("{home}/.hipfire/npu/embgemma_aie2p_resident_ffn_dense_w8_direct_x_gate_reuse_bf16x2_m256_k768_i1152_o768")
    });
    let direct512_cache = args.get(2).cloned().unwrap_or_else(|| {
        format!("{home}/.hipfire/npu/embgemma_aie2p_resident_ffn_dense_w8_direct_x_gate_reuse_bf16x2_m512_k768_i1152_o768")
    });

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

    let x = (0..M * K)
        .map(|index| {
            f32_to_bf16_bits(
                (index as f32 * 0.0037).sin() * 0.7 + (index % 19) as f32 * 0.009 - 0.08,
            )
        })
        .collect::<Vec<_>>();
    let pre_norm = (0..K)
        .map(|hidden| f32_to_bf16_bits(0.91 + (hidden % 29) as f32 * 0.0015))
        .collect::<Vec<_>>();
    let inverse = (0..M)
        .map(|token| {
            let row = &x[token * K..(token + 1) * K];
            let sum = row
                .iter()
                .map(|&bits| {
                    let value = bf16_bits_to_f32(bits);
                    value * value
                })
                .sum::<f32>();
            (sum / K as f32 + EPSILON).sqrt().recip()
        })
        .collect::<Vec<_>>();
    let normalized = x
        .chunks_exact(K)
        .zip(inverse.iter().copied())
        .flat_map(|(row, inverse)| {
            row.iter().zip(&pre_norm).map(move |(&x, &norm)| {
                f32_to_bf16_bits(bf16_bits_to_f32(x) * bf16_bits_to_f32(norm) * inverse)
            })
        })
        .collect::<Vec<_>>();

    let mut canonical = NpuResidentFfnDenseW8::load_cached(&canonical_cache)?;
    if !matches!(
        canonical.io_mode(),
        NpuResidentFfnDenseW8IoMode::CanonicalBf16
            | NpuResidentFfnDenseW8IoMode::CanonicalBf16Bf16x2Output
    ) {
        return Err("canonical M256 cache does not accept canonical BF16 input".into());
    }
    let canonical_weights = canonical.upload_weights(&gate, &up, &down)?;
    let expected = canonical.run_canonical_bf16(&canonical_weights, &normalized)?;

    let mut direct256 = NpuResidentFfnDenseW8::load_cached(&direct256_cache)?;
    let direct256_weights =
        direct256.upload_weights_with_pre_ffn_norm(&gate, &up, &down, &pre_norm)?;
    direct256.write_direct_x_bf16_with_inverse(&x, &inverse)?;
    direct256.run_shared(&direct256_weights)?;
    let y256 = direct256.read_canonical_output_f32()?;

    let mut direct512 = NpuResidentFfnDenseW8::load_cached(&direct512_cache)?;
    let direct512_weights =
        direct512.upload_weights_with_pre_ffn_norm(&gate, &up, &down, &pre_norm)?;
    let mut xx = x.clone();
    xx.extend_from_slice(&x);
    let mut inverse2 = inverse.clone();
    inverse2.extend_from_slice(&inverse);
    direct512.write_direct_x_bf16_with_inverse(&xx, &inverse2)?;
    direct512.run_shared(&direct512_weights)?;
    let y512 = direct512.read_canonical_output_f32()?;
    if y512.len() != 2 * y256.len() {
        return Err(format!(
            "direct-X M512 output {} != 2x M256 output {}",
            y512.len(),
            y256.len()
        )
        .into());
    }
    let (doc0, doc1) = y512.split_at(y256.len());

    let (absolute_cosine, absolute_max, absolute_mean) = metrics(&y256, &expected);
    let (doc0_cosine, doc0_max, doc0_mean) = metrics(doc0, &y256);
    let (doc1_cosine, doc1_max, doc1_mean) = metrics(doc1, &y256);
    let (self_cosine, self_max, self_mean) = metrics(doc0, doc1);
    println!(
        "direct M256 vs canonical: cosine={absolute_cosine:.8} max_abs={absolute_max:.7} mean_abs={absolute_mean:.8}"
    );
    println!(
        "direct M512 doc0 vs M256: cosine={doc0_cosine:.8} max_abs={doc0_max:.7} mean_abs={doc0_mean:.8}"
    );
    println!(
        "direct M512 doc1 vs M256: cosine={doc1_cosine:.8} max_abs={doc1_max:.7} mean_abs={doc1_mean:.8}"
    );
    println!(
        "direct M512 doc0 vs doc1: cosine={self_cosine:.8} max_abs={self_max:.7} mean_abs={self_mean:.8}"
    );

    // Direct-X gate-fragment reuse has an existing extra quantization boundary
    // relative to the canonical path. Keep that absolute guard strict enough
    // to catch layout corruption while requiring bit-exact batch replication.
    let correct = absolute_cosine > 0.9998
        && absolute_max < 0.02
        && doc0_cosine > 0.99999
        && doc1_cosine > 0.99999
        && self_cosine > 0.99999
        && doc0_max < 1.0e-4
        && doc1_max < 1.0e-4
        && self_max < 1.0e-4;
    if !correct {
        return Err("DIRECT-X BATCHED FFN MISMATCH".into());
    }

    let iterations = 20;
    for _ in 0..3 {
        direct256.run_shared(&direct256_weights)?;
        direct512.run_shared(&direct512_weights)?;
    }
    let start = std::time::Instant::now();
    for _ in 0..iterations {
        direct256.run_shared(&direct256_weights)?;
    }
    let ms256 = start.elapsed().as_secs_f64() * 1e3 / iterations as f64;
    let start = std::time::Instant::now();
    for _ in 0..iterations {
        direct512.run_shared(&direct512_weights)?;
    }
    let ms512 = start.elapsed().as_secs_f64() * 1e3 / iterations as f64;
    println!(
        "timing: M256={ms256:.3} ms M512={ms512:.3} ms row_throughput_gain={:.2}x",
        2.0 * ms256 / ms512
    );
    println!("DIRECT-X BATCHED FFN OK");
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
                block[2 + inner] = ((mixed % 15) as i8 - 7) as u8;
            }
        }
    }
    payload
}

#[cfg(target_os = "linux")]
fn metrics(got: &[f32], expected: &[f32]) -> (f64, f32, f64) {
    let (mut dot, mut got_norm, mut expected_norm, mut max_abs, mut sum_abs) =
        (0.0, 0.0, 0.0, 0.0f32, 0.0);
    for (&got, &expected) in got.iter().zip(expected) {
        let difference = (got - expected).abs();
        max_abs = max_abs.max(difference);
        sum_abs += difference as f64;
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
