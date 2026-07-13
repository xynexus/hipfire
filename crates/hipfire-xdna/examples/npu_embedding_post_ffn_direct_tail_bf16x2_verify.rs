//! Hardware oracle for the compensated-BF16x2 post-FFN tail.

#[cfg(target_os = "linux")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use hipfire_primitives::conv::{bf16_bits_to_f32, f32_to_bf16_bits};
    use hipfire_xdna::NpuEmbeddingPostFfnDirectTailBf16x2;

    const M: usize = 256;
    const HIDDEN: usize = 768;
    const EPSILON: f32 = 1.0e-6;
    let cache = std::env::args().nth(1).unwrap_or_else(|| {
        format!(
            "{}/.hipfire/npu/embgemma_aie2p_post_ffn_direct_tail_bf16x2_m256_k768",
            std::env::var("HOME").expect("HOME")
        )
    });
    let iterations = std::env::args()
        .nth(2)
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(0);
    let manifest = std::fs::read_to_string(format!("{cache}/shape.txt"))?;
    let split_residual = manifest.lines().any(|line| line.contains("split-x-bf16"));
    let interleaved_ffn = manifest
        .lines()
        .any(|line| line.contains("interleaved-bf16x2"));
    let post_norm = (0..HIDDEN)
        .map(|hidden| f32_to_bf16_bits(0.87 + (hidden % 31) as f32 * 0.0018))
        .collect::<Vec<_>>();
    let residual = (0..M * HIDDEN)
        .map(|index| f32_to_bf16_bits(((index * 37 % 257) as f32 - 128.0) * 0.0017))
        .collect::<Vec<_>>();
    let exact_ffn = (0..M * HIDDEN)
        .map(|index| ((index * 23 % 193) as f32 - 96.0) * 0.0911 + 0.00317)
        .collect::<Vec<_>>();
    let (high, low): (Vec<_>, Vec<_>) = exact_ffn
        .iter()
        .map(|&value| {
            let high = f32_to_bf16_bits(value);
            let low = f32_to_bf16_bits(value - bf16_bits_to_f32(high));
            (high, low)
        })
        .unzip();
    let compensated = high
        .iter()
        .zip(&low)
        .map(|(&high, &low)| bf16_bits_to_f32(high) + bf16_bits_to_f32(low))
        .collect::<Vec<_>>();
    let mut expected = vec![0.0f32; M * HIDDEN];
    for token in 0..M {
        let base = token * HIDDEN;
        let sum = compensated[base..base + HIDDEN]
            .iter()
            .map(|value| value.powi(2))
            .sum::<f32>();
        let inverse = (sum / HIDDEN as f32 + EPSILON).sqrt().recip();
        for hidden in 0..HIDDEN {
            let index = base + hidden;
            expected[index] = bf16_bits_to_f32(f32_to_bf16_bits(
                bf16_bits_to_f32(residual[index])
                    + compensated[index] * bf16_bits_to_f32(post_norm[hidden]) * inverse,
            ));
        }
    }

    let gpu = hipfire_rdna::Gpu::init()?;
    let mut tail = NpuEmbeddingPostFfnDirectTailBf16x2::load_cached(&cache)?;
    let mut output_shared = gpu.alloc_shared_gtt(tail.output_bytes())?;
    let mut combined_shared =
        gpu.alloc_shared_gtt(NpuEmbeddingPostFfnDirectTailBf16x2::combined_bytes())?;
    output_shared.as_mut_slice().fill(0);
    combined_shared.as_mut_slice().fill(0);
    if interleaved_ffn {
        if split_residual {
            write_interleaved_rows(combined_shared.as_mut_slice(), &high, &low, M, HIDDEN);
        } else {
            write_interleaved_combined_rows(
                combined_shared.as_mut_slice(),
                &high,
                &low,
                &residual,
                M,
                HIDDEN,
            );
        }
    } else {
        write_combined_rows(
            combined_shared.as_mut_slice(),
            &high,
            &low,
            &residual,
            M,
            HIDDEN,
        );
    }
    let residual_shared = if split_residual {
        let mut buffer =
            gpu.alloc_shared_gtt(NpuEmbeddingPostFfnDirectTailBf16x2::residual_bytes())?;
        write_bf16(buffer.as_mut_slice(), &residual);
        tail.attach_shared_split_state(
            combined_shared.dmabuf_fd(),
            combined_shared.len(),
            buffer.dmabuf_fd(),
            buffer.len(),
            output_shared.dmabuf_fd(),
            output_shared.len(),
        )?;
        Some(buffer)
    } else {
        tail.attach_shared_state(
            combined_shared.dmabuf_fd(),
            combined_shared.len(),
            output_shared.dmabuf_fd(),
            output_shared.len(),
        )?;
        None
    };
    let _residual_shared = residual_shared;
    tail.sync_shared_inputs()?;
    let params = tail.upload_params(&post_norm, EPSILON)?;
    tail.run_shared(&params)?;
    let got = tail.read_output_f32()?;
    let (cosine, max_abs) = metrics(&got, &expected);
    if cosine < 0.99999 || max_abs > 0.025 {
        let first_bad = got
            .iter()
            .zip(&expected)
            .enumerate()
            .find(|(_, (&got, &expected))| !got.is_finite() || (got - expected).abs() > 0.025);
        let stripe_heads = (0..8)
            .map(|stripe| {
                let token = (stripe % 4) * 32 + (stripe / 4) * 128;
                (token, got[token * HIDDEN], expected[token * HIDDEN])
            })
            .collect::<Vec<_>>();
        return Err(format!(
            "compensated direct-tail parity failed: cosine={cosine:.8} max_abs={max_abs:.7} first_bad={first_bad:?} stripes={stripe_heads:?} got_head={:?} expected_head={:?}",
            &got[..16],
            &expected[..16],
        )
        .into());
    }
    println!(
        "embeddinggemma-post-ffn-direct-tail-bf16x2 M=256 K=768: cosine={cosine:.8} max_abs={max_abs:.7}"
    );
    if iterations > 0 {
        let started = std::time::Instant::now();
        for _ in 0..iterations {
            tail.run_shared(&params)?;
        }
        let dispatch_ms = started.elapsed().as_secs_f64() * 1e3 / iterations as f64;
        println!("iterations={iterations} dispatch_ms={dispatch_ms:.6}");
    }
    Ok(())
}

#[cfg(target_os = "linux")]
fn write_bf16(destination: &mut [u8], values: &[u16]) {
    for (bytes, value) in destination.chunks_exact_mut(2).zip(values.iter().copied()) {
        bytes.copy_from_slice(&value.to_le_bytes());
    }
}

#[cfg(target_os = "linux")]
fn write_interleaved_combined_rows(
    destination: &mut [u8],
    high: &[u16],
    low: &[u16],
    residual: &[u16],
    rows: usize,
    hidden: usize,
) {
    let row_bytes = 3 * hidden * size_of::<u16>();
    for row in 0..rows {
        let destination = &mut destination[row * row_bytes..(row + 1) * row_bytes];
        let source = row * hidden;
        for column in 0..hidden {
            let offset = 2 * column * size_of::<u16>();
            destination[offset..offset + 2].copy_from_slice(&high[source + column].to_le_bytes());
            destination[offset + 2..offset + 4]
                .copy_from_slice(&low[source + column].to_le_bytes());
        }
        write_bf16(
            &mut destination[2 * hidden * 2..],
            &residual[source..source + hidden],
        );
    }
}

#[cfg(target_os = "linux")]
fn write_combined_rows(
    destination: &mut [u8],
    high: &[u16],
    low: &[u16],
    residual: &[u16],
    rows: usize,
    columns: usize,
) {
    for row in 0..rows {
        let source = row * columns;
        let target = row * columns * 6;
        write_bf16(
            &mut destination[target..target + columns * 2],
            &high[source..source + columns],
        );
        write_bf16(
            &mut destination[target + columns * 2..target + columns * 4],
            &low[source..source + columns],
        );
        write_bf16(
            &mut destination[target + columns * 4..target + columns * 6],
            &residual[source..source + columns],
        );
    }
}

#[cfg(target_os = "linux")]
fn write_interleaved_rows(
    destination: &mut [u8],
    high: &[u16],
    low: &[u16],
    rows: usize,
    columns: usize,
) {
    for row in 0..rows {
        for column in 0..columns {
            let source = row * columns + column;
            let target = row * columns * 6 + column * 4;
            destination[target..target + 2].copy_from_slice(&high[source].to_le_bytes());
            destination[target + 2..target + 4].copy_from_slice(&low[source].to_le_bytes());
        }
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
