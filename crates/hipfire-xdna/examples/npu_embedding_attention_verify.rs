//! CPU-oracle parity and timing for the R27 M256 bidirectional attention graph.

#[cfg(target_os = "linux")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use hipfire_primitives::conv::{bf16_bits_to_f32, f32_to_bf16_bits};
    use hipfire_xdna::{EmbeddingGemmaAttentionLayout as Layout, NpuKernel};

    const M: usize = Layout::TOKENS;
    const HEADS: usize = Layout::QUERY_HEADS;
    const D: usize = Layout::HEAD_DIM;

    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if !(1..=2).contains(&args.len()) {
        return Err("usage: npu_embedding_attention_verify CACHE [ITERS]".into());
    }
    let iterations = args
        .get(1)
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(20);
    if iterations == 0 {
        return Err("attention verifier needs at least one iteration".into());
    }

    let manifest = std::fs::read_to_string(format!("{}/shape.txt", args[0]))?;
    for field in [
        "op=attention",
        "mode=bf16",
        "m=256",
        "heads=3",
        "kv_heads=1",
        "head_dim=256",
        "q_layout=mmul-packed",
        "kv_layout=mmul-packed-single-replay",
    ] {
        if !manifest.lines().any(|line| line == field) {
            return Err(format!("attention cache missing {field}").into());
        }
    }

    let q = (0..HEADS * M * D)
        .map(|index| {
            let value = (index as f32 * 0.001_731).sin() * 0.23 + ((index / D) % 17) as f32 * 0.002;
            bf16_bits_to_f32(f32_to_bf16_bits(value))
        })
        .collect::<Vec<_>>();
    let k = (0..M * D)
        .map(|index| {
            let value =
                (index as f32 * 0.002_117).cos() * 0.19 - ((index / D) % 13) as f32 * 0.0015;
            bf16_bits_to_f32(f32_to_bf16_bits(value))
        })
        .collect::<Vec<_>>();
    let v = (0..M * D)
        .map(|index| {
            let value = (index as f32 * 0.001_337).sin() * 0.31 + (index % 11) as f32 * 0.003;
            bf16_bits_to_f32(f32_to_bf16_bits(value))
        })
        .collect::<Vec<_>>();
    let reference = attention_reference(&q, &k, &v, HEADS, M, D);

    let q_bits = q.iter().copied().map(f32_to_bf16_bits).collect::<Vec<_>>();
    let k_bits = k.iter().copied().map(f32_to_bf16_bits).collect::<Vec<_>>();
    let v_bits = v.iter().copied().map(f32_to_bf16_bits).collect::<Vec<_>>();
    let packed_q = Layout::pack_q_bf16(&q_bits).ok_or("invalid Q layout input")?;
    let packed_kv = Layout::pack_kv_bf16(&k_bits, &v_bits).ok_or("invalid K/V layout input")?;

    let xclbin = std::fs::read(format!("{}/final.xclbin", args[0]))?;
    let insts = std::fs::read(format!("{}/insts.bin", args[0]))?;
    let kernel = NpuKernel::load(&xclbin, &insts)?;
    let mut q_buffer = kernel.alloc_arg(Layout::Q_BYTES)?;
    let mut kv_buffer = kernel.alloc_arg(Layout::KV_BYTES)?;
    let mut output_buffer = kernel.alloc_arg(Layout::OUTPUT_BYTES)?;
    q_buffer.as_mut_slice().copy_from_slice(&packed_q);
    kv_buffer.as_mut_slice().copy_from_slice(&packed_kv);
    output_buffer.as_mut_slice().fill(0);

    // Prime a fresh array context once, then evaluate the next command.
    kernel.dispatch_synced(
        &[&q_buffer, &kv_buffer, &output_buffer],
        &[true, true, true],
    )?;
    output_buffer.as_mut_slice().fill(0);
    kernel.sync_to_device(&output_buffer)?;
    kernel.dispatch_synced(
        &[&q_buffer, &kv_buffer, &output_buffer],
        &[false, false, false],
    )?;
    kernel.sync_output(&output_buffer)?;
    let output = unpack_output(output_buffer.as_slice())?;
    let (cosine, max_abs, mean_abs) = metrics(&output, &reference);
    if !cosine.is_finite() || cosine < 0.998 || max_abs > 0.04 {
        return Err(format!(
            "R27 attention parity failed: cosine={cosine:.8} max_abs={max_abs:.7} mean_abs={mean_abs:.8}"
        )
        .into());
    }

    let started = std::time::Instant::now();
    for _ in 0..iterations {
        kernel.dispatch_synced(
            &[&q_buffer, &kv_buffer, &output_buffer],
            &[false, false, false],
        )?;
    }
    kernel.sync_output(&output_buffer)?;
    let final_output = unpack_output(output_buffer.as_slice())?;
    let (final_cosine, final_max_abs, _) = metrics(&final_output, &reference);
    if final_cosine < 0.998 || final_max_abs > 0.04 {
        return Err(format!(
            "R27 sustained parity failed: cosine={final_cosine:.8} max_abs={final_max_abs:.7}"
        )
        .into());
    }
    let dispatch_ms = started.elapsed().as_secs_f64() * 1e3 / iterations as f64;
    println!(
        "embedding-attention-bf16 M={M} H={HEADS} D={D}: cosine={cosine:.8} max_abs={max_abs:.7} mean_abs={mean_abs:.8} dispatch_ms={dispatch_ms:.4}"
    );
    Ok(())
}

#[cfg(target_os = "linux")]
fn unpack_output(bytes: &[u8]) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let bits = hipfire_xdna::EmbeddingGemmaAttentionLayout::unpack_output_bf16(bytes)
        .ok_or("invalid physical attention output")?;
    Ok(bits
        .into_iter()
        .map(hipfire_primitives::conv::bf16_bits_to_f32)
        .collect())
}

#[cfg(target_os = "linux")]
fn attention_reference(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    heads: usize,
    rows: usize,
    dim: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; q.len()];
    let mut scores = vec![0.0f32; rows];
    for head in 0..heads {
        for query in 0..rows {
            let qrow = &q[(head * rows + query) * dim..(head * rows + query + 1) * dim];
            for key in 0..rows {
                scores[key] = qrow
                    .iter()
                    .zip(&k[key * dim..(key + 1) * dim])
                    .map(|(&left, &right)| left * right)
                    .sum::<f32>()
                    * 0.0625;
            }
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let sum = scores
                .iter_mut()
                .map(|score| {
                    *score = (*score - max).exp();
                    *score
                })
                .sum::<f32>();
            let destination =
                &mut output[(head * rows + query) * dim..(head * rows + query + 1) * dim];
            for key in 0..rows {
                let probability = scores[key] / sum;
                for index in 0..dim {
                    destination[index] += probability * v[key * dim + index];
                }
            }
        }
    }
    output
}

#[cfg(target_os = "linux")]
fn metrics(got: &[f32], expected: &[f32]) -> (f64, f32, f64) {
    let mut dot = 0.0f64;
    let mut got_norm = 0.0f64;
    let mut expected_norm = 0.0f64;
    let mut max_abs = 0.0f32;
    let mut sum_abs = 0.0f64;
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
    eprintln!("R27 attention verification is Linux-only");
}
