//! CPU-oracle parity and timing for the R27 M256 bidirectional attention graph.

#[cfg(target_os = "linux")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use hipfire_primitives::conv::{bf16_bits_to_f32, f32_to_bf16_bits};
    use hipfire_xdna::NpuKernel;

    const M: usize = 256;
    const HEADS: usize = 3;
    const D: usize = 256;
    const CORES: usize = 32;
    const ROWS: usize = 4;
    const COLS: usize = 8;
    const QUERIES: usize = 4;
    const GROUPS: usize = HEADS * M / (CORES * QUERIES);
    const BLOCK_KEYS: usize = 16;
    const BLOCKS: usize = M / BLOCK_KEYS;
    const MMUL_K: usize = 8;
    const MMUL_N: usize = 8;
    const DIM_TILES: usize = D / MMUL_K;
    const KEY_TILES: usize = BLOCK_KEYS / MMUL_N;
    const Q_TILE: usize = QUERIES * D * 2;
    const Q_JOIN: usize = COLS * Q_TILE;
    const KV_TILE: usize = 2 * BLOCK_KEYS * D * 2;
    const O_JOIN: usize = ROWS * Q_TILE;
    const Q_BYTES: usize = ROWS * GROUPS * Q_JOIN;
    const KV_BYTES: usize = GROUPS * BLOCKS * KV_TILE;
    const O_BYTES: usize = COLS * GROUPS * O_JOIN;

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

    let mut packed_q = vec![0u8; Q_BYTES];
    for linear in 0..HEADS * M {
        let group = linear / (CORES * QUERIES);
        let remainder = linear % (CORES * QUERIES);
        let core = remainder / QUERIES;
        let lane = remainder % QUERIES;
        let row = core / COLS;
        let col = core % COLS;
        let tile = (row * GROUPS + group) * Q_JOIN + col * Q_TILE;
        for dim_tile in 0..DIM_TILES {
            for dim_lane in 0..MMUL_K {
                let destination =
                    tile + (dim_tile * QUERIES * MMUL_K + lane * MMUL_K + dim_lane) * 2;
                write_bf16_value(
                    &mut packed_q,
                    destination,
                    q[linear * D + dim_tile * MMUL_K + dim_lane],
                );
            }
        }
    }
    let mut packed_kv = vec![0u8; KV_BYTES];
    for group in 0..GROUPS {
        for block in 0..BLOCKS {
            let destination = (group * BLOCKS + block) * KV_TILE;
            for key_tile in 0..KEY_TILES {
                for dim_tile in 0..DIM_TILES {
                    for dim_lane in 0..MMUL_K {
                        for key_lane in 0..MMUL_N {
                            let key = block * BLOCK_KEYS + key_tile * MMUL_N + key_lane;
                            let dim = dim_tile * MMUL_K + dim_lane;
                            let packed = ((key_tile * DIM_TILES + dim_tile) * MMUL_K * MMUL_N)
                                + dim_lane * MMUL_N
                                + key_lane;
                            write_bf16_value(
                                &mut packed_kv,
                                destination + packed * 2,
                                k[key * D + dim],
                            );
                        }
                    }
                }
            }
            let values = destination + BLOCK_KEYS * D * 2;
            for dim_tile in 0..DIM_TILES {
                for key_tile in 0..KEY_TILES {
                    for key_lane in 0..MMUL_K {
                        for dim_lane in 0..MMUL_N {
                            let key = block * BLOCK_KEYS + key_tile * MMUL_K + key_lane;
                            let dim = dim_tile * MMUL_N + dim_lane;
                            let packed = ((dim_tile * KEY_TILES + key_tile) * MMUL_K * MMUL_N)
                                + key_lane * MMUL_N
                                + dim_lane;
                            write_bf16_value(&mut packed_kv, values + packed * 2, v[key * D + dim]);
                        }
                    }
                }
            }
        }
    }

    let xclbin = std::fs::read(format!("{}/final.xclbin", args[0]))?;
    let insts = std::fs::read(format!("{}/insts.bin", args[0]))?;
    let kernel = NpuKernel::load(&xclbin, &insts)?;
    let mut q_buffer = kernel.alloc_arg(Q_BYTES)?;
    let mut kv_buffer = kernel.alloc_arg(KV_BYTES)?;
    let mut output_buffer = kernel.alloc_arg(O_BYTES)?;
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
    let output = unpack_output(output_buffer.as_slice(), HEADS, M, D);
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
    let final_output = unpack_output(output_buffer.as_slice(), HEADS, M, D);
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
fn write_bf16_value(destination: &mut [u8], byte_offset: usize, value: f32) {
    destination[byte_offset..byte_offset + 2]
        .copy_from_slice(&hipfire_primitives::conv::f32_to_bf16_bits(value).to_le_bytes());
}

#[cfg(target_os = "linux")]
fn unpack_output(bytes: &[u8], heads: usize, rows: usize, dim: usize) -> Vec<f32> {
    const CORE_ROWS: usize = 4;
    const COLS: usize = 8;
    const QUERIES: usize = 4;
    const GROUPS: usize = 6;
    let tile_bytes = QUERIES * dim * 2;
    let join_bytes = CORE_ROWS * tile_bytes;
    let mut output = vec![0.0f32; heads * rows * dim];
    for linear in 0..heads * rows {
        let group = linear / (CORE_ROWS * COLS * QUERIES);
        let remainder = linear % (CORE_ROWS * COLS * QUERIES);
        let core = remainder / QUERIES;
        let lane = remainder % QUERIES;
        let core_row = core / COLS;
        let col = core % COLS;
        let source = (col * GROUPS + group) * join_bytes + core_row * tile_bytes + lane * dim * 2;
        for index in 0..dim {
            let offset = source + index * 2;
            let bits = u16::from_le_bytes([bytes[offset], bytes[offset + 1]]);
            output[linear * dim + index] = hipfire_primitives::conv::bf16_bits_to_f32(bits);
        }
    }
    output
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
