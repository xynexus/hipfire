//! Hardware parity check for the R97 canonical-BF16 resident **W4** (op4++) FFN.
//!
//! This is the W4A8 sibling of `npu_resident_ffn_w8_canonical_verify`: it drives
//! the productionised `NpuResidentFfnW4::{upload_weights, run_canonical_bf16}`
//! path on silicon and compares against the shared `OpusPackedMatrix::reference_f32`
//! op4++ oracle (AWQ → FWHT-256 → int8 activation quant → integer dot → scale).
//! It is the first milestone of the NPU W4A8 op4++ MoE effort
//! (`docs/plans/2026-07-17-npu-w4a8-op4pp-moe-qwen35.md`, M1): lock the clean-API
//! hardware parity gate at EmbeddingGemma shape; the same harness retargets to the
//! Qwen3.5 expert shape once that xclbin exists.
//!
//! Serialize with the NPU: hold `hipfire lock` while running (single hw queue).
//!
//! Usage: `npu_resident_ffn_w4_canonical_verify [CACHE] [--iters N]`

#[cfg(target_os = "linux")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::time::Instant;

    use hipfire_primitives::conv::{bf16_bits_to_f32, f32_to_bf16_bits};
    use hipfire_xdna::{NpuResidentFfnW4, NpuResidentFfnW4IoMode, OpusPackedMatrix};

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
                "{}/.hipfire/npu/embgemma_r99_canonical_bf16_w4_resident_ffn_combined_bf16x2_m256_k768_i1152_o768",
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

    // op4++ weights in the exact on-disk W4 layout the qt=33 loader produces.
    let gate =
        OpusPackedMatrix::from_payload(33, K, INTERMEDIATE, &w4_payload(K, INTERMEDIATE, 3), None)?;
    let up = OpusPackedMatrix::from_payload(
        33,
        K,
        INTERMEDIATE,
        &w4_payload(K, INTERMEDIATE, 11),
        None,
    )?;
    let down = OpusPackedMatrix::from_payload(
        33,
        INTERMEDIATE,
        OUTPUT,
        &w4_payload(INTERMEDIATE, OUTPUT, 23),
        None,
    )?;

    let mut executor = NpuResidentFfnW4::load_cached(&cache)?;
    if executor.io_mode() != NpuResidentFfnW4IoMode::CanonicalBf16InterleavedBf16x2 {
        return Err(format!(
            "cache did not select the plain canonical-BF16 W4 ABI (got {:?})",
            executor.io_mode()
        )
        .into());
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

    // GeGLU reference (the R97/EmbeddingGemma schedule fuses gelu, not silu).
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
    // BF16x2 output keeps ~f32 precision, so no final rounding of the reference.
    let reference = down.reference_f32(M, &intermediate)?;
    let reference_ms = reference_started.elapsed().as_secs_f64() * 1e3;

    let output = executor.run_canonical_bf16(&weights, &input_bf16)?;
    let (cosine, max_abs, mean_abs) = metrics(&output, &reference);
    let max_reference = reference.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let max_allowed = 0.02 + 0.03 * max_reference;
    if !cosine.is_finite() || cosine < 0.999 || max_abs > max_allowed {
        for index in [0, 1, OUTPUT - 1, OUTPUT, M * OUTPUT - 1] {
            eprintln!(
                "output[{index}] got={:.7} reference={:.7}",
                output[index], reference[index]
            );
        }
        return Err(format!(
            "R97 W4 parity failed: cosine={cosine:.8} max_abs={max_abs:.7} allowed={max_allowed:.7}"
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
        "r99-canonical-bf16-w4 M={M} K={K} I={INTERMEDIATE} O={OUTPUT}: cosine={cosine:.8} max_abs={max_abs:.7} mean_abs={mean_abs:.8} allowed={max_allowed:.7} reference_ms={reference_ms:.1} dispatch_ms={dispatch_ms:.4} iters={iterations}"
    );
    Ok(())
}

/// Synthetic op4++ weights in the qt=33 W4 on-disk layout `decode_opus_groups`
/// consumes: per (column, K-group) a 130-byte block = fp16 scale (2B) + 128
/// packed nibbles, low nibble = inner `2i`, high nibble = inner `2i+1`, each a
/// signed 4-bit value. Block order is column-major over K-groups.
#[cfg(target_os = "linux")]
fn w4_payload(k: usize, n: usize, seed: usize) -> Vec<u8> {
    use hipfire_primitives::conv::f32_to_f16;

    const GROUP: usize = 256;
    const BLOCK: usize = 130;
    let groups = k.div_ceil(GROUP);
    let mut payload = vec![0u8; n * groups * BLOCK];
    for col in 0..n {
        for group in 0..groups {
            let block =
                &mut payload[(col * groups + group) * BLOCK..(col * groups + group + 1) * BLOCK];
            let scale = 0.012 * (1.0 + ((col + 3 * group + seed) % 7) as f32 * 0.03);
            block[..2].copy_from_slice(&f32_to_f16(scale).to_le_bytes());
            for packed_idx in 0..128 {
                let mut byte = 0u8;
                for lane in 0..2 {
                    let inner = 2 * packed_idx + lane;
                    let mixed = (inner as u64).wrapping_mul(0x9e37_79b1)
                        ^ (col as u64).wrapping_mul(0x85eb_ca77)
                        ^ (group as u64).wrapping_mul(0xc2b2_ae3d)
                        ^ (seed as u64).wrapping_mul(0x27d4_eb2f);
                    // signed 4-bit in [-7, 7]
                    let value = ((mixed % 15) as i8 - 7) & 0x0f;
                    byte |= (value as u8) << (4 * lane);
                }
                block[2 + packed_idx] = byte;
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
