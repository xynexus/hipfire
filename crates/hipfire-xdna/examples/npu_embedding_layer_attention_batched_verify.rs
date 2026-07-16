//! Compare one fused M512 segmented-attention command with two M256 commands.
//!
//! Distinct documents are required: each M512 half must reproduce its own M256
//! hardware oracle, which catches accidental cross-document Q/K/V attention.

#[cfg(target_os = "linux")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use hipfire_primitives::conv::f32_to_bf16_bits;
    use hipfire_xdna::{NpuEmbeddingLayerAttentionDenseW8, OpusPackedMatrix};

    const M: usize = 256;
    const K: usize = 768;
    const QKV_N: usize = 1280;
    const GROUPS: usize = 3;

    let home = std::env::var("HOME").expect("HOME");
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let cache256 = args.first().cloned().unwrap_or_else(|| {
        format!("{home}/.hipfire/npu/embgemma_r108_resident_w8_qkv_attention_direct_completed_residual_m256_k768_n1280")
    });
    let cache512 = args.get(1).cloned().unwrap_or_else(|| {
        format!("{home}/.hipfire/npu/embgemma_r108_resident_w8_qkv_attention_direct_completed_residual_m512_k768_n1280")
    });
    let iterations = args
        .get(2)
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(10);
    if iterations == 0 {
        return Err("batched attention verifier needs at least one iteration".into());
    }

    let groups = (0..GROUPS)
        .map(|group| {
            (0..256 * QKV_N)
                .map(|index| (((index * 13 + index / 31 + group * 7) % 11) as i8) - 5)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let scales = (0..GROUPS)
        .map(|group| {
            (0..QKV_N)
                .map(|column| 0.0032 + ((column * 5 + group * 11) % 23) as f32 * 0.000_017)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let group_refs = groups.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let scale_refs = scales.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let output = OpusPackedMatrix::from_payload(35, K, K, &w8_payload(K, K, 41, 0.0045), None)?;
    let qnorm = (0..256)
        .map(|index| 0.83 + (index % 29) as f32 * 0.004)
        .collect::<Vec<_>>();
    let knorm = (0..256)
        .map(|index| 0.91 + (index % 23) as f32 * 0.003)
        .collect::<Vec<_>>();
    let post_norm = bf16_values(K, |index| 0.87 + (index % 31) as f32 * 0.002);
    let pre_norm = bf16_values(K, |index| 0.91 + (index % 29) as f32 * 0.0015);
    let upload_residual = vec![0u16; M * K];

    let documents = (0..2)
        .map(|document| {
            let activations = (0..M * K)
                .map(|index| {
                    (((index * (17 + document * 2) + index / (29 + document) + document * 5) % 15)
                        as i8)
                        - 7
                })
                .collect::<Vec<_>>();
            let activation_scales = (0..GROUPS * M)
                .map(|index| 0.0045 + ((index + document * 7) % (19 + document)) as f32 * 0.000_031)
                .collect::<Vec<_>>();
            let residual = (0..M * K)
                .map(|index| {
                    f32_to_bf16_bits(
                        ((index + document * 97) as f32 * (0.0037 + document as f32 * 0.0004))
                            .sin()
                            * (0.42 + document as f32 * 0.07)
                            + ((index % (17 + document * 2)) as f32) * 0.006
                            - 0.05,
                    )
                })
                .collect::<Vec<_>>();
            let packed = NpuEmbeddingLayerAttentionDenseW8::prepack_activations(
                &activations,
                &activation_scales,
            )?;
            Ok::<_, hipfire_xdna::XdnaError>((packed, completed_bf16x2(&residual)))
        })
        .collect::<Result<Vec<_>, _>>()?;

    let mut single = NpuEmbeddingLayerAttentionDenseW8::load_cached(&cache256)?;
    let single_weights = single.upload_dense_groups(
        &group_refs,
        &scale_refs,
        None,
        &output,
        &upload_residual,
        &qnorm,
        &knorm,
        &post_norm,
        &pre_norm,
        1.0e-6,
        10_000.0,
    )?;
    let mut references = Vec::with_capacity(2);
    for (packed, completed) in &documents {
        single.prepare_layer(&single_weights)?;
        single.set_prepacked_input(&single_weights, packed)?;
        single.set_completed_bf16x2(completed)?;
        single.run_shared(&single_weights)?;
        references.push((single.read_hidden_f32()?, single.read_pre_inverse_f32()?));
    }

    let mut batched = NpuEmbeddingLayerAttentionDenseW8::load_cached(&cache512)?;
    if batched.loaded_rows() != 2 * M {
        return Err(format!(
            "batched cache exposes {} rows, expected {}",
            batched.loaded_rows(),
            2 * M
        )
        .into());
    }
    let batched_weights = batched.upload_dense_groups(
        &group_refs,
        &scale_refs,
        None,
        &output,
        &upload_residual,
        &qnorm,
        &knorm,
        &post_norm,
        &pre_norm,
        1.0e-6,
        10_000.0,
    )?;
    let packed = documents
        .iter()
        .flat_map(|(packed, _)| packed.iter().copied())
        .collect::<Vec<_>>();
    let completed = documents
        .iter()
        .flat_map(|(_, completed)| completed.iter().copied())
        .collect::<Vec<_>>();
    batched.prepare_layer(&batched_weights)?;
    batched.set_prepacked_input(&batched_weights, &packed)?;
    batched.set_completed_bf16x2(&completed)?;
    batched.run_shared(&batched_weights)?;
    let got = batched.read_hidden_f32()?;
    let got_inverse = batched.read_pre_inverse_f32()?;

    for document in 0..2 {
        let output_range = document * M * K..(document + 1) * M * K;
        let inverse_range = document * M..(document + 1) * M;
        let output_metrics = metrics(&got[output_range], &references[document].0);
        let inverse_metrics = metrics(&got_inverse[inverse_range], &references[document].1);
        println!(
            "document={document} x_cosine={:.8} x_max={:.7} x_mean={:.8} inverse_cosine={:.8} inverse_max={:.7}",
            output_metrics.0,
            output_metrics.1,
            output_metrics.2,
            inverse_metrics.0,
            inverse_metrics.1,
        );
        if !output_metrics.0.is_finite()
            || output_metrics.0 < 0.99999
            || output_metrics.1 > 1.0e-4
            || !inverse_metrics.0.is_finite()
            || inverse_metrics.0 < 0.99999
            || inverse_metrics.1 > 1.0e-6
        {
            return Err(format!("fused batched attention document {document} mismatch").into());
        }
    }

    for _ in 0..2 {
        single.run_shared(&single_weights)?;
        batched.run_shared(&batched_weights)?;
    }
    let started = std::time::Instant::now();
    for _ in 0..iterations {
        single.run_shared(&single_weights)?;
    }
    let ms256 = started.elapsed().as_secs_f64() * 1e3 / iterations as f64;
    let started = std::time::Instant::now();
    for _ in 0..iterations {
        batched.run_shared(&batched_weights)?;
    }
    let ms512 = started.elapsed().as_secs_f64() * 1e3 / iterations as f64;
    println!(
        "timing: M256={ms256:.3} ms M512={ms512:.3} ms row_throughput_gain={:.2}x",
        2.0 * ms256 / ms512
    );
    println!("FUSED SEGMENTED ATTENTION BATCH OK");
    Ok(())
}

#[cfg(target_os = "linux")]
fn completed_bf16x2(residual: &[u16]) -> Vec<u8> {
    const M: usize = 256;
    const K: usize = 768;
    const PAD_M: usize = 288;
    debug_assert_eq!(residual.len(), M * K);
    let mut output = vec![0u8; PAD_M * 2 * K * size_of::<u16>()];
    for row in 0..M {
        let target = row * 2 * K * size_of::<u16>();
        for (word, &bits) in output[target..target + K * 2]
            .chunks_exact_mut(2)
            .zip(&residual[row * K..(row + 1) * K])
        {
            word.copy_from_slice(&bits.to_le_bytes());
        }
    }
    output
}

#[cfg(target_os = "linux")]
fn bf16_values(length: usize, mut value: impl FnMut(usize) -> f32) -> Vec<u16> {
    (0..length)
        .map(|index| hipfire_primitives::conv::f32_to_bf16_bits(value(index)))
        .collect()
}

#[cfg(target_os = "linux")]
fn w8_payload(k: usize, n: usize, seed: usize, base_scale: f32) -> Vec<u8> {
    use hipfire_primitives::conv::f32_to_f16;
    const GROUP: usize = 256;
    const BLOCK: usize = 258;
    let groups = k.div_ceil(GROUP);
    let mut payload = vec![0u8; n * groups * BLOCK];
    for column in 0..n {
        for group in 0..groups {
            let block = &mut payload
                [(column * groups + group) * BLOCK..(column * groups + group + 1) * BLOCK];
            let scale = base_scale * (1.0 + ((column + 3 * group + seed) % 7) as f32 * 0.025);
            block[..2].copy_from_slice(&f32_to_f16(scale).to_le_bytes());
            for inner in 0..GROUP {
                let mixed = (inner as u64).wrapping_mul(0x9e37_79b1)
                    ^ (column as u64).wrapping_mul(0x85eb_ca77)
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
    eprintln!("npu_embedding_layer_attention_batched_verify is Linux-only");
}
