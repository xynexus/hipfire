//! Hardware oracle for the resident EmbeddingGemma post-FFN tail.

#[cfg(target_os = "linux")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use hipfire_primitives::conv::{bf16_bits_to_f32, f32_to_bf16_bits};
    use hipfire_xdna::NpuEmbeddingPostFfnTail;

    const M: usize = 256;
    const HIDDEN: usize = 768;
    const TILE: usize = 8 * HIDDEN * 2;
    const META_BASE: usize = M * HIDDEN * 2;
    const EPSILON: f32 = 1.0e-6;

    let cache = std::env::args().nth(1).unwrap_or_else(|| {
        format!(
            "{}/.hipfire/npu/embgemma_aie2p_post_ffn_tail_m256_k768",
            std::env::var("HOME").expect("HOME")
        )
    });
    let pre_norm = (0..HIDDEN)
        .map(|hidden| f32_to_bf16_bits(0.91 + (hidden % 29) as f32 * 0.0015))
        .collect::<Vec<_>>();
    let post_norm = (0..HIDDEN)
        .map(|hidden| f32_to_bf16_bits(0.87 + (hidden % 31) as f32 * 0.0018))
        .collect::<Vec<_>>();
    let mut hidden_bits = vec![0u16; M * HIDDEN];
    let mut ffn_bits = vec![0u16; M * HIDDEN];
    let mut inverse = vec![0.0f32; M];
    for token in 0..M {
        let mut sum = 0.0f32;
        for hidden in 0..HIDDEN {
            let value = (((token * 37 + hidden * 13) % 257) as f32 - 128.0) * 0.0017;
            sum += value * value;
        }
        inverse[token] = (sum / HIDDEN as f32 + EPSILON).sqrt().recip();
        for hidden in 0..HIDDEN {
            let value = (((token * 37 + hidden * 13) % 257) as f32 - 128.0) * 0.0017;
            hidden_bits[token * HIDDEN + hidden] =
                f32_to_bf16_bits(value * bf16_bits_to_f32(pre_norm[hidden]) * inverse[token]);
            let ffn = (((token * 19 + hidden * 23 + 7) % 193) as f32 - 96.0) * 0.0011;
            ffn_bits[token * HIDDEN + hidden] = f32_to_bf16_bits(ffn);
        }
    }

    let gpu = hipfire_rdna::Gpu::init()?;
    let mut hidden_shared = gpu.alloc_shared_gtt(NpuEmbeddingPostFfnTail::hidden_backing_bytes())?;
    let mut ffn_shared = gpu.alloc_shared_gtt(NpuEmbeddingPostFfnTail::shared_state_bytes())?;
    hidden_shared.as_mut_slice().fill(0);
    ffn_shared.as_mut_slice().fill(0);
    write_bf16(hidden_shared.as_mut_slice(), &hidden_bits);
    write_bf16(ffn_shared.as_mut_slice(), &ffn_bits);
    for core_row in 0..4 {
        for core_col in 0..8 {
            let token_base = (core_col / 4) * 128 + core_row * 32 + (core_col % 4) * 8;
            let record = META_BASE + (core_row * 8 + core_col) * TILE;
            for row in 0..8 {
                let offset = record + row * size_of::<f32>();
                hidden_shared.as_mut_slice()[offset..offset + size_of::<f32>()]
                    .copy_from_slice(&inverse[token_base + row].to_le_bytes());
            }
        }
    }
    let mut tail = NpuEmbeddingPostFfnTail::load_cached(&cache)?;
    tail.attach_shared_state(
        hidden_shared.dmabuf_fd(),
        hidden_shared.len(),
        ffn_shared.dmabuf_fd(),
        ffn_shared.len(),
    )?;
    let params = tail.upload_params(&pre_norm, &post_norm, EPSILON)?;
    tail.run_shared(&params)?;
    let got = tail.read_output_f32()?;
    let reference = reference(
        &hidden_bits,
        &ffn_bits,
        &inverse,
        &pre_norm,
        &post_norm,
        EPSILON,
    );
    let measured = metrics(&got, &reference);
    if !measured.0.is_finite() || measured.0 < 0.9999 || measured.1 > 0.025 {
        return Err(format!(
            "R39 post-FFN tail parity failed: cosine={:.8} max_abs={:.7} got={:?} ref={:?}",
            measured.0,
            measured.1,
            &got[..8],
            &reference[..8]
        )
        .into());
    }
    println!(
        "embeddinggemma-post-ffn-tail M=256 K=768: cosine={:.8} max_abs={:.7}",
        measured.0, measured.1
    );
    Ok(())
}

#[cfg(target_os = "linux")]
fn write_bf16(destination: &mut [u8], values: &[u16]) {
    for (bytes, value) in destination.chunks_exact_mut(2).zip(values.iter().copied()) {
        bytes.copy_from_slice(&value.to_le_bytes());
    }
}

#[cfg(target_os = "linux")]
fn reference(
    hidden: &[u16],
    ffn: &[u16],
    inverse: &[f32],
    pre_norm: &[u16],
    post_norm: &[u16],
    epsilon: f32,
) -> Vec<f32> {
    use hipfire_primitives::conv::{bf16_bits_to_f32, f32_to_bf16_bits};

    const M: usize = 256;
    const HIDDEN: usize = 768;
    let mut output = vec![0.0f32; M * HIDDEN];
    for token in 0..M {
        let ffn_row = &ffn[token * HIDDEN..(token + 1) * HIDDEN];
        let sum = ffn_row
            .iter()
            .map(|&bits| {
                let value = bf16_bits_to_f32(bits);
                value * value
            })
            .sum::<f32>();
        let post_inverse = (sum / HIDDEN as f32 + epsilon).sqrt().recip();
        for hidden_index in 0..HIDDEN {
            let index = token * HIDDEN + hidden_index;
            let residual = bf16_bits_to_f32(hidden[index])
                * bf16_bits_to_f32(pre_norm[hidden_index]).recip()
                * inverse[token].recip();
            let normalized = bf16_bits_to_f32(ffn[index])
                * bf16_bits_to_f32(post_norm[hidden_index])
                * post_inverse;
            output[index] = bf16_bits_to_f32(f32_to_bf16_bits(residual + normalized));
        }
    }
    output
}

#[cfg(target_os = "linux")]
fn metrics(got: &[f32], expected: &[f32]) -> (f64, f32) {
    let mut dot = 0.0f64;
    let mut got_norm = 0.0f64;
    let mut expected_norm = 0.0f64;
    let mut max_abs = 0.0f32;
    for (&got, &expected) in got.iter().zip(expected) {
        dot += got as f64 * expected as f64;
        got_norm += got as f64 * got as f64;
        expected_norm += expected as f64 * expected as f64;
        max_abs = max_abs.max((got - expected).abs());
    }
    (dot / (got_norm.sqrt() * expected_norm.sqrt()), max_abs)
}

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("npu_embedding_post_ffn_tail_verify is Linux-only");
}
