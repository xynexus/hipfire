//! CPU-oracle parity and timing for the R28 QKV headnorm/RoPE pack graph.

#[cfg(target_os = "linux")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use hipfire_primitives::conv::f32_to_bf16_bits;
    use hipfire_xdna::{EmbeddingGemmaAttentionLayout as Layout, NpuKernel};

    const ROWS: usize = 4;
    const RAW_JOIN: usize = 32768;
    const PARAMS: usize = 2048;
    const RAW_Q_BYTES: usize = Layout::QUERY_GROUPS * ROWS * RAW_JOIN;
    const RAW_K_BYTES: usize = 2 * ROWS * RAW_JOIN;
    const RAW_V_BYTES: usize = 2 * ROWS * RAW_JOIN;
    const RAW_BYTES: usize = RAW_Q_BYTES + RAW_K_BYTES + RAW_V_BYTES;
    const EPSILON: f32 = 1.0e-6;
    const ROPE_BASE: f32 = 10_000.0;

    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if !(1..=2).contains(&args.len()) {
        return Err("usage: npu_embedding_qkv_pack_verify CACHE [ITERS]".into());
    }
    let iterations = args
        .get(1)
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(100);
    if iterations == 0 {
        return Err("QKV pack verifier needs at least one iteration".into());
    }

    let manifest = std::fs::read_to_string(format!("{}/shape.txt", args[0]))?;
    for field in [
        "op=qkv-headnorm-rope-pack",
        "mode=bf16",
        "m=256",
        "heads=3",
        "kv_heads=1",
        "head_dim=256",
        "q_layout=mmul-packed",
        "kv_layout=mmul-packed-single-replay",
    ] {
        if !manifest.lines().any(|line| line == field) {
            return Err(format!("QKV pack cache missing {field}").into());
        }
    }

    let q = bf16_values(
        Layout::QUERY_HEADS * Layout::TOKENS * Layout::HEAD_DIM,
        |index| (index as f32 * 0.001_731).sin() * 0.43 + ((index / 256) % 17) as f32 * 0.003,
    );
    let k = bf16_values(Layout::TOKENS * Layout::HEAD_DIM, |index| {
        (index as f32 * 0.002_117).cos() * 0.37 - ((index / 256) % 13) as f32 * 0.002
    });
    let v = bf16_values(Layout::TOKENS * Layout::HEAD_DIM, |index| {
        (index as f32 * 0.001_337).sin() * 0.31 + (index % 11) as f32 * 0.003
    });
    let qnorm = bf16_values(Layout::HEAD_DIM, |index| 0.83 + (index % 29) as f32 * 0.004);
    let knorm = bf16_values(Layout::HEAD_DIM, |index| 0.91 + (index % 23) as f32 * 0.003);
    let cs = if std::env::var_os("HIPFIRE_R28_IDENTITY_ROPE").is_some() {
        identity_rope_cs()
    } else {
        rope_cs(ROPE_BASE)
    };

    let q_reference = headnorm_rope(&q, &qnorm, &cs, Layout::QUERY_HEADS, EPSILON);
    let k_reference = headnorm_rope(&k, &knorm, &cs, Layout::KV_HEADS, EPSILON);
    let q_reference_bits = to_bf16_bits(&q_reference);
    let k_reference_bits = to_bf16_bits(&k_reference);
    let v_bits = to_bf16_bits(&v);

    let raw = pack_raw_inputs(&q, &k, &v, &cs);
    let params = pack_params(&qnorm, &knorm, EPSILON);
    assert_eq!(raw.len(), RAW_BYTES);
    assert_eq!(params.len(), PARAMS);

    let xclbin = std::fs::read(format!("{}/final.xclbin", args[0]))?;
    let insts = std::fs::read(format!("{}/insts.bin", args[0]))?;
    let kernel = NpuKernel::load(&xclbin, &insts)?;
    let mut raw_buffer = kernel.alloc_arg(RAW_BYTES)?;
    let mut params_buffer = kernel.alloc_arg(PARAMS)?;
    let mut q_buffer = kernel.alloc_arg(Layout::Q_BYTES)?;
    let mut kv_buffer = kernel.alloc_arg(Layout::KV_BYTES)?;
    raw_buffer.as_mut_slice().copy_from_slice(&raw);
    params_buffer.as_mut_slice().copy_from_slice(&params);
    q_buffer.as_mut_slice().fill(0);
    kv_buffer.as_mut_slice().fill(0);

    kernel.dispatch_synced(
        &[&raw_buffer, &params_buffer, &q_buffer, &kv_buffer],
        &[true, true, true, true],
    )?;
    q_buffer.as_mut_slice().fill(0);
    kv_buffer.as_mut_slice().fill(0);
    kernel.sync_to_device(&q_buffer)?;
    kernel.sync_to_device(&kv_buffer)?;
    kernel.dispatch_synced(
        &[&raw_buffer, &params_buffer, &q_buffer, &kv_buffer],
        &[false, false, false, false],
    )?;
    kernel.sync_output(&q_buffer)?;
    kernel.sync_output(&kv_buffer)?;

    let q_got = read_q(q_buffer.as_slice());
    let k_got = read_k(kv_buffer.as_slice());
    let v_got = read_v(kv_buffer.as_slice());
    let (q_cosine, q_max, q_mean) = metrics(&q_got, &q_reference);
    let (k_cosine, k_max, k_mean) = metrics(&k_got, &k_reference);
    let q_bit_mismatches = q_got
        .iter()
        .map(|&value| f32_to_bf16_bits(value))
        .zip(&q_reference_bits)
        .filter(|(got, expected)| got != *expected)
        .count();
    let k_bit_mismatches = k_got
        .iter()
        .map(|&value| f32_to_bf16_bits(value))
        .zip(&k_reference_bits)
        .filter(|(got, expected)| got != *expected)
        .count();
    let v_bit_mismatches = v_got
        .iter()
        .map(|&value| f32_to_bf16_bits(value))
        .zip(&v_bits)
        .filter(|(got, expected)| got != *expected)
        .count();
    if q_cosine < 0.999 || k_cosine < 0.999 || q_max > 0.03 || k_max > 0.03 {
        let q_worst = worst_error(&q_got, &q_reference);
        let k_worst = worst_error(&k_got, &k_reference);
        let q_halves = half_metrics(&q_got, &q_reference);
        let k_halves = half_metrics(&k_got, &k_reference);
        return Err(format!(
            "R28 parity failed: q_cos={q_cosine:.8} q_max={q_max:.7} q_worst={q_worst:?} q_halves={q_halves:?} k_cos={k_cosine:.8} k_max={k_max:.7} k_worst={k_worst:?} k_halves={k_halves:?} v_bit_mismatches={v_bit_mismatches}; q[0..8]={:?} q_ref[0..8]={:?}; k[0..8]={:?} k_ref[0..8]={:?}",
            &q_got[..8],
            &q_reference[..8],
            &k_got[..8],
            &k_reference[..8],
        )
        .into());
    }
    if v_bit_mismatches != 0 {
        return Err(format!("R28 V pack has {v_bit_mismatches} bit mismatches").into());
    }

    let started = std::time::Instant::now();
    for _ in 0..iterations {
        kernel.dispatch_synced(
            &[&raw_buffer, &params_buffer, &q_buffer, &kv_buffer],
            &[false, false, false, false],
        )?;
    }
    let dispatch_ms = started.elapsed().as_secs_f64() * 1e3 / iterations as f64;
    println!(
        "qkv-headnorm-rope-pack M={} H={} D={}: q_cosine={q_cosine:.8} q_max={q_max:.7} q_mean={q_mean:.8} q_bit_mismatches={q_bit_mismatches} k_cosine={k_cosine:.8} k_max={k_max:.7} k_mean={k_mean:.8} k_bit_mismatches={k_bit_mismatches} v_bit_mismatches={v_bit_mismatches} dispatch_ms={dispatch_ms:.4}",
        Layout::TOKENS,
        Layout::QUERY_HEADS,
        Layout::HEAD_DIM,
    );
    Ok(())
}

#[cfg(target_os = "linux")]
fn bf16_values(length: usize, value: impl Fn(usize) -> f32) -> Vec<f32> {
    (0..length)
        .map(|index| {
            hipfire_primitives::conv::bf16_bits_to_f32(hipfire_primitives::conv::f32_to_bf16_bits(
                value(index),
            ))
        })
        .collect()
}

#[cfg(target_os = "linux")]
fn to_bf16_bits(values: &[f32]) -> Vec<u16> {
    values
        .iter()
        .copied()
        .map(hipfire_primitives::conv::f32_to_bf16_bits)
        .collect()
}

#[cfg(target_os = "linux")]
fn rope_cs(base: f32) -> Vec<u16> {
    use hipfire_primitives::conv::f32_to_bf16_bits;
    use hipfire_xdna::EmbeddingGemmaAttentionLayout as Layout;
    let half = Layout::HEAD_DIM / 2;
    let mut cs = vec![0u16; Layout::TOKENS * Layout::HEAD_DIM];
    for token in 0..Layout::TOKENS {
        for dim in 0..half {
            let frequency = 1.0 / base.powf((2 * dim) as f32 / Layout::HEAD_DIM as f32);
            let angle = token as f32 * frequency;
            cs[token * Layout::HEAD_DIM + dim] = f32_to_bf16_bits(angle.cos());
            cs[token * Layout::HEAD_DIM + half + dim] = f32_to_bf16_bits(angle.sin());
        }
    }
    cs
}

#[cfg(target_os = "linux")]
fn identity_rope_cs() -> Vec<u16> {
    use hipfire_primitives::conv::f32_to_bf16_bits;
    use hipfire_xdna::EmbeddingGemmaAttentionLayout as Layout;
    let mut cs = vec![0u16; Layout::TOKENS * Layout::HEAD_DIM];
    for token in 0..Layout::TOKENS {
        for dim in 0..Layout::HEAD_DIM / 2 {
            cs[token * Layout::HEAD_DIM + dim] = f32_to_bf16_bits(1.0);
        }
    }
    cs
}

#[cfg(target_os = "linux")]
fn headnorm_rope(
    input: &[f32],
    weight: &[f32],
    cs: &[u16],
    heads: usize,
    epsilon: f32,
) -> Vec<f32> {
    use hipfire_primitives::conv::{bf16_bits_to_f32, f32_to_bf16_bits};
    use hipfire_xdna::EmbeddingGemmaAttentionLayout as Layout;
    let half = Layout::HEAD_DIM / 2;
    let mut output = vec![0.0f32; input.len()];
    for head in 0..heads {
        for token in 0..Layout::TOKENS {
            let base = (head * Layout::TOKENS + token) * Layout::HEAD_DIM;
            let row = &input[base..base + Layout::HEAD_DIM];
            let inv_rms = 1.0
                / (row.iter().map(|value| value * value).sum::<f32>() / Layout::HEAD_DIM as f32
                    + epsilon)
                    .sqrt();
            for dim in 0..half {
                let x = row[dim] * weight[dim] * inv_rms;
                let y = row[half + dim] * weight[half + dim] * inv_rms;
                let cosine = bf16_bits_to_f32(cs[token * Layout::HEAD_DIM + dim]);
                let sine = bf16_bits_to_f32(cs[token * Layout::HEAD_DIM + half + dim]);
                output[base + dim] = bf16_bits_to_f32(f32_to_bf16_bits(x * cosine - y * sine));
                output[base + half + dim] =
                    bf16_bits_to_f32(f32_to_bf16_bits(y * cosine + x * sine));
            }
        }
    }
    output
}

#[cfg(target_os = "linux")]
fn pack_raw_inputs(q: &[f32], k: &[f32], v: &[f32], cs: &[u16]) -> Vec<u8> {
    use hipfire_primitives::conv::f32_to_bf16_bits;
    use hipfire_xdna::EmbeddingGemmaAttentionLayout as Layout;
    const ROWS: usize = 4;
    const COLS: usize = 8;
    const RAW_PAIR: usize = 8192;
    const RAW_JOIN: usize = 32768;
    const RAW_Q_BYTES: usize = Layout::QUERY_GROUPS * ROWS * RAW_JOIN;
    const RAW_K_BYTES: usize = 2 * ROWS * RAW_JOIN;
    let mut raw = vec![0u8; RAW_Q_BYTES + 2 * RAW_K_BYTES];

    for group in 0..Layout::QUERY_GROUPS {
        for row in 0..ROWS {
            for pair in 0..COLS / 2 {
                let pair_base = (group * ROWS + row) * RAW_JOIN + pair * RAW_PAIR;
                for pair_lane in 0..2 {
                    let core = row * COLS + 2 * pair + pair_lane;
                    for query in 0..Layout::QUERIES_PER_CORE {
                        let linear = group * Layout::CORES * Layout::QUERIES_PER_CORE
                            + core * Layout::QUERIES_PER_CORE
                            + query;
                        let head = linear / Layout::TOKENS;
                        let token = linear % Layout::TOKENS;
                        let raw_row = pair_lane * Layout::QUERIES_PER_CORE + query;
                        for dim in 0..Layout::HEAD_DIM {
                            write_u16(
                                &mut raw,
                                pair_base + (raw_row * Layout::HEAD_DIM + dim) * 2,
                                f32_to_bf16_bits(
                                    q[(head * Layout::TOKENS + token) * Layout::HEAD_DIM + dim],
                                ),
                            );
                            write_u16(
                                &mut raw,
                                pair_base + 4096 + (raw_row * Layout::HEAD_DIM + dim) * 2,
                                cs[token * Layout::HEAD_DIM + dim],
                            );
                        }
                    }
                }
            }
        }
    }
    for (role, values) in [(0usize, k), (1usize, v)] {
        let phase = RAW_Q_BYTES + role * RAW_K_BYTES;
        for wave in 0..2 {
            for row in 0..ROWS {
                for pair in 0..COLS / 2 {
                    let token0 = wave * 128 + (row * 4 + pair) * 8;
                    let pair_base = phase + (wave * ROWS + row) * RAW_JOIN + pair * RAW_PAIR;
                    for key in 0..8 {
                        for dim in 0..Layout::HEAD_DIM {
                            write_u16(
                                &mut raw,
                                pair_base + (key * Layout::HEAD_DIM + dim) * 2,
                                f32_to_bf16_bits(values[(token0 + key) * Layout::HEAD_DIM + dim]),
                            );
                            if role == 0 {
                                write_u16(
                                    &mut raw,
                                    pair_base + 4096 + (key * Layout::HEAD_DIM + dim) * 2,
                                    cs[(token0 + key) * Layout::HEAD_DIM + dim],
                                );
                            }
                        }
                    }
                }
            }
        }
    }
    raw
}

#[cfg(target_os = "linux")]
fn pack_params(qnorm: &[f32], knorm: &[f32], epsilon: f32) -> Vec<u8> {
    use hipfire_primitives::conv::f32_to_bf16_bits;
    let mut params = vec![0u8; 2048];
    for (index, &value) in qnorm.iter().enumerate() {
        write_u16(&mut params, index * 2, f32_to_bf16_bits(value));
    }
    for (index, &value) in knorm.iter().enumerate() {
        write_u16(&mut params, 512 + index * 2, f32_to_bf16_bits(value));
    }
    params[1024..1028].copy_from_slice(&epsilon.to_le_bytes());
    params
}

#[cfg(target_os = "linux")]
fn write_u16(destination: &mut [u8], offset: usize, value: u16) {
    destination[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
}

#[cfg(target_os = "linux")]
fn read_q(bytes: &[u8]) -> Vec<f32> {
    use hipfire_primitives::conv::bf16_bits_to_f32;
    use hipfire_xdna::EmbeddingGemmaAttentionLayout as Layout;
    let mut output = vec![0.0f32; Layout::QUERY_HEADS * Layout::TOKENS * Layout::HEAD_DIM];
    for head in 0..Layout::QUERY_HEADS {
        for token in 0..Layout::TOKENS {
            for dim in 0..Layout::HEAD_DIM {
                let offset = Layout::q_offset(head, token, dim).unwrap();
                output[(head * Layout::TOKENS + token) * Layout::HEAD_DIM + dim] =
                    bf16_bits_to_f32(u16::from_le_bytes([bytes[offset], bytes[offset + 1]]));
            }
        }
    }
    output
}

#[cfg(target_os = "linux")]
fn read_k(bytes: &[u8]) -> Vec<f32> {
    read_kv(bytes, true)
}

#[cfg(target_os = "linux")]
fn read_v(bytes: &[u8]) -> Vec<f32> {
    read_kv(bytes, false)
}

#[cfg(target_os = "linux")]
fn read_kv(bytes: &[u8], key: bool) -> Vec<f32> {
    use hipfire_primitives::conv::bf16_bits_to_f32;
    use hipfire_xdna::EmbeddingGemmaAttentionLayout as Layout;
    let mut output = vec![0.0f32; Layout::TOKENS * Layout::HEAD_DIM];
    for token in 0..Layout::TOKENS {
        for dim in 0..Layout::HEAD_DIM {
            let offset = if key {
                Layout::k_offset(token, dim)
            } else {
                Layout::v_offset(token, dim)
            }
            .unwrap();
            output[token * Layout::HEAD_DIM + dim] =
                bf16_bits_to_f32(u16::from_le_bytes([bytes[offset], bytes[offset + 1]]));
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
    let mut mean_abs = 0.0f64;
    for (&got, &expected) in got.iter().zip(expected) {
        dot += got as f64 * expected as f64;
        got_norm += (got as f64).powi(2);
        expected_norm += (expected as f64).powi(2);
        let error = (got - expected).abs();
        max_abs = max_abs.max(error);
        mean_abs += error as f64;
    }
    (
        dot / (got_norm * expected_norm).sqrt(),
        max_abs,
        mean_abs / got.len() as f64,
    )
}

#[cfg(target_os = "linux")]
fn worst_error(got: &[f32], expected: &[f32]) -> (usize, f32, f32) {
    got.iter()
        .copied()
        .zip(expected.iter().copied())
        .enumerate()
        .max_by(|(_, (got_a, expected_a)), (_, (got_b, expected_b))| {
            (got_a - expected_a)
                .abs()
                .total_cmp(&(got_b - expected_b).abs())
        })
        .map(|(index, (got, expected))| (index, got, expected))
        .unwrap()
}

#[cfg(target_os = "linux")]
fn half_metrics(got: &[f32], expected: &[f32]) -> [(f64, f32, f64); 2] {
    use hipfire_xdna::EmbeddingGemmaAttentionLayout as Layout;
    let rows = got.len() / Layout::HEAD_DIM;
    std::array::from_fn(|half| {
        let mut got_half = Vec::with_capacity(rows * Layout::HEAD_DIM / 2);
        let mut expected_half = Vec::with_capacity(got_half.capacity());
        for row in 0..rows {
            let start = row * Layout::HEAD_DIM + half * Layout::HEAD_DIM / 2;
            let end = start + Layout::HEAD_DIM / 2;
            got_half.extend_from_slice(&got[start..end]);
            expected_half.extend_from_slice(&expected[start..end]);
        }
        metrics(&got_half, &expected_half)
    })
}

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("npu_embedding_qkv_pack_verify is Linux-only");
}
