//! Hardware oracle for the R40 direct-residual post-FFN tail.

#[cfg(target_os = "linux")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use hipfire_primitives::conv::{bf16_bits_to_f32, f32_to_bf16_bits};
    use hipfire_xdna::NpuEmbeddingPostFfnDirectTail;

    const M: usize = 256;
    const HIDDEN: usize = 768;
    const EPSILON: f32 = 1.0e-6;
    let cache = std::env::args().nth(1).unwrap_or_else(|| {
        format!(
            "{}/.hipfire/npu/embgemma_aie2p_post_ffn_direct_tail_m256_k768",
            std::env::var("HOME").expect("HOME")
        )
    });
    let post_norm = (0..HIDDEN)
        .map(|hidden| f32_to_bf16_bits(0.87 + (hidden % 31) as f32 * 0.0018))
        .collect::<Vec<_>>();
    let residual = (0..M * HIDDEN)
        .map(|index| f32_to_bf16_bits(((index * 37 % 257) as f32 - 128.0) * 0.0017))
        .collect::<Vec<_>>();
    let ffn = (0..M * HIDDEN)
        .map(|index| f32_to_bf16_bits(((index * 23 % 193) as f32 - 96.0) * 0.0011))
        .collect::<Vec<_>>();
    let mut expected = vec![0.0f32; M * HIDDEN];
    for token in 0..M {
        let base = token * HIDDEN;
        let sum = ffn[base..base + HIDDEN]
            .iter()
            .map(|&bits| bf16_bits_to_f32(bits).powi(2))
            .sum::<f32>();
        let inverse = (sum / HIDDEN as f32 + EPSILON).sqrt().recip();
        for hidden in 0..HIDDEN {
            let index = base + hidden;
            expected[index] = bf16_bits_to_f32(f32_to_bf16_bits(
                bf16_bits_to_f32(residual[index])
                    + bf16_bits_to_f32(ffn[index]) * bf16_bits_to_f32(post_norm[hidden]) * inverse,
            ));
        }
    }

    let gpu = hipfire_rdna::Gpu::init()?;
    let mut residual_shared =
        gpu.alloc_shared_gtt(NpuEmbeddingPostFfnDirectTail::shared_state_bytes())?;
    let mut ffn_shared =
        gpu.alloc_shared_gtt(NpuEmbeddingPostFfnDirectTail::shared_state_bytes())?;
    residual_shared.as_mut_slice().fill(0);
    ffn_shared.as_mut_slice().fill(0);
    write_bf16(residual_shared.as_mut_slice(), &residual);
    write_bf16(ffn_shared.as_mut_slice(), &ffn);
    let mut tail = NpuEmbeddingPostFfnDirectTail::load_cached(&cache)?;
    tail.attach_shared_state(
        residual_shared.dmabuf_fd(),
        residual_shared.len(),
        ffn_shared.dmabuf_fd(),
        ffn_shared.len(),
    )?;
    let params = tail.upload_params(&post_norm, EPSILON)?;
    tail.run_shared(&params)?;
    let got = tail.read_output_f32()?;
    let (cosine, max_abs) = metrics(&got, &expected);
    if cosine < 0.9999 || max_abs > 0.025 {
        return Err(format!(
            "R40 direct tail parity failed: cosine={cosine:.8} max_abs={max_abs:.7}"
        )
        .into());
    }
    println!(
        "embeddinggemma-post-ffn-direct-tail M=256 K=768: cosine={cosine:.8} max_abs={max_abs:.7}"
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
fn metrics(got: &[f32], expected: &[f32]) -> (f64, f32) {
    let mut dot = 0.0f64;
    let mut got_norm = 0.0f64;
    let mut expected_norm = 0.0f64;
    let mut max_abs = 0.0f32;
    for (&got, &expected) in got.iter().zip(expected) {
        dot += got as f64 * expected as f64;
        got_norm += (got as f64).powi(2);
        expected_norm += (expected as f64).powi(2);
        max_abs = max_abs.max((got - expected).abs());
    }
    (dot / (got_norm.sqrt() * expected_norm.sqrt()), max_abs)
}

#[cfg(not(target_os = "linux"))]
fn main() {}
