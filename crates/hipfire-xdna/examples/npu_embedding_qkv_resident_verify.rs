//! CPU-oracle gate for the resident R29 W8 QKV projection-to-attention pack.

#[cfg(target_os = "linux")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use hipfire_primitives::conv::f32_to_bf16_bits;
    use hipfire_xdna::{EmbeddingGemmaAttentionLayout as Layout, NpuKernel};

    const K: usize = 768;
    const N: usize = 1280;
    const GROUPS: usize = 3;
    const W_BYTES: usize = 8 * 45 * 16384;
    const EPSILON: f32 = 1.0e-6;

    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if !(1..=2).contains(&args.len()) {
        return Err("usage: npu_embedding_qkv_resident_verify CACHE [ITERS]".into());
    }
    let iterations = args
        .get(1)
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(20);
    if iterations == 0 {
        return Err("R29 verifier needs at least one iteration".into());
    }
    let manifest = std::fs::read_to_string(format!("{}/shape.txt", args[0]))?;
    let attention = manifest
        .lines()
        .any(|line| line == "op=resident-qkv-attention");
    let operation = if attention {
        "op=resident-qkv-attention"
    } else {
        "op=resident-qkv-headnorm-rope-pack"
    };
    for field in [
        operation,
        "mode=w8-scaled",
        "m=256",
        "k=768",
        "n=1280",
        "roles=q0,q1,q2,k,v",
    ] {
        if !manifest.lines().any(|line| line == field) {
            return Err(format!("R29 cache missing {field}").into());
        }
    }
    let pair_bytes = if attention { 16384 } else { 10240 };
    let a_bytes = 4 * 45 * pair_bytes;
    let r_stage_bytes = 5 * 48 * pair_bytes;
    let r_bytes = r_stage_bytes + if attention { Layout::OUTPUT_BYTES } else { 0 };

    let activations = (0..Layout::TOKENS * K)
        .map(|index| (((index * 17 + index / 29) % 15) as i8) - 7)
        .collect::<Vec<_>>();
    let activation_scales = (0..GROUPS * Layout::TOKENS)
        .map(|index| 0.0045 + (index % 19) as f32 * 0.000_031)
        .collect::<Vec<_>>();
    let weights = (0..GROUPS)
        .map(|group| {
            (0..256 * N)
                .map(|index| (((index * 13 + index / 31 + group * 7) % 11) as i8) - 5)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let weight_scales = (0..GROUPS)
        .map(|group| {
            (0..N)
                .map(|col| 0.0032 + ((col * 5 + group * 11) % 23) as f32 * 0.000_017)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let qnorm = bf16_values(Layout::HEAD_DIM, |index| 0.83 + (index % 29) as f32 * 0.004);
    let knorm = bf16_values(Layout::HEAD_DIM, |index| 0.91 + (index % 23) as f32 * 0.003);
    let cs = rope_cs(10_000.0);

    let projected = cpu_projection(&activations, &activation_scales, &weights, &weight_scales);

    let packed_a = pack_activations(&activations, &activation_scales, pair_bytes);
    let packed_w = pack_weights(&weights, &weight_scales);
    let mut staged_r = stage_positions_and_params(&cs, &qnorm, &knorm, EPSILON, pair_bytes);
    staged_r.resize(r_bytes, 0);
    assert_eq!(packed_a.len(), a_bytes);
    assert_eq!(packed_w.len(), W_BYTES);
    assert_eq!(staged_r.len(), r_bytes);

    let kernel = NpuKernel::load(
        &std::fs::read(format!("{}/final.xclbin", args[0]))?,
        &std::fs::read(format!("{}/insts.bin", args[0]))?,
    )?;
    let mut a_buffer = kernel.alloc_arg(a_bytes)?;
    let mut w_buffer = kernel.alloc_arg(W_BYTES)?;
    let mut r_buffer = kernel.alloc_arg(r_bytes)?;
    let mut q_buffer = kernel.alloc_arg(Layout::Q_BYTES)?;
    let mut kv_buffer = kernel.alloc_arg(Layout::KV_BYTES)?;
    a_buffer.as_mut_slice().copy_from_slice(&packed_a);
    w_buffer.as_mut_slice().copy_from_slice(&packed_w);
    r_buffer.as_mut_slice().copy_from_slice(&staged_r);
    q_buffer.as_mut_slice().fill(0);
    kv_buffer.as_mut_slice().fill(0);

    dispatch_resident(
        &kernel, &a_buffer, &w_buffer, &r_buffer, &q_buffer, &kv_buffer, true,
    )?;
    r_buffer.as_mut_slice().copy_from_slice(&staged_r);
    q_buffer.as_mut_slice().fill(0);
    kv_buffer.as_mut_slice().fill(0);
    kernel.sync_to_device(&r_buffer)?;
    kernel.sync_to_device(&q_buffer)?;
    kernel.sync_to_device(&kv_buffer)?;
    dispatch_resident(
        &kernel, &a_buffer, &w_buffer, &r_buffer, &q_buffer, &kv_buffer, false,
    )?;
    kernel.sync_output(&r_buffer)?;
    kernel.sync_output(&q_buffer)?;
    kernel.sync_output(&kv_buffer)?;

    let projected_got = read_projected(r_buffer.as_slice(), pair_bytes);
    let q_got = read_q(q_buffer.as_slice());
    let k_got = read_kv(kv_buffer.as_slice(), true);
    let v_got = read_kv(kv_buffer.as_slice(), false);
    let projection = metrics(&projected_got, &projected);
    let q_handoff = role_q(&projected_got);
    let k_handoff = role_kv(&projected_got, 3);
    let v_handoff = role_kv(&projected_got, 4);
    let q_handoff_reference = headnorm_rope(&q_handoff, &qnorm, &cs, Layout::QUERY_HEADS, EPSILON);
    let k_handoff_reference = headnorm_rope(&k_handoff, &knorm, &cs, Layout::KV_HEADS, EPSILON);
    let q = metrics(&q_got, &q_handoff_reference);
    let k = metrics(&k_got, &k_handoff_reference);
    let v_mismatches = v_got
        .iter()
        .map(|&value| f32_to_bf16_bits(value))
        .zip(v_handoff.iter().map(|&value| f32_to_bf16_bits(value)))
        .filter(|(got, expected)| got != expected)
        .count();
    if projection.0 < 0.9999 || projection.1 > 0.01 {
        return Err(format!("R29 projection parity failed: {projection:?}").into());
    }
    if q.0 < 0.999 || k.0 < 0.999 || q.1 > 0.04 || k.1 > 0.04 || v_mismatches != 0 {
        let first_v_mismatch = v_got
            .iter()
            .zip(&v_handoff)
            .position(|(&got, &expected)| f32_to_bf16_bits(got) != f32_to_bf16_bits(expected));
        let k_non_finite = k_got.iter().filter(|value| !value.is_finite()).count();
        return Err(format!(
            "R29 pack parity failed: projection={projection:?} q={q:?} k={k:?} k_non_finite={k_non_finite} v_bit_mismatches={v_mismatches} first_v_mismatch={first_v_mismatch:?}; k[0..8]={:?} k_ref[0..8]={:?}; v[0..8]={:?} v_ref[0..8]={:?}",
            &k_got[..8],
            &k_handoff_reference[..8],
            &v_got[..8],
            &v_handoff[..8],
        )
        .into());
    }

    let attention_metrics = if attention {
        let got = unpack_attention(&r_buffer.as_slice()[r_stage_bytes..])?;
        let reference = attention_reference(&q_got, &k_got, &v_got);
        let measured = metrics(&got, &reference);
        if !measured.0.is_finite() || measured.0 < 0.998 || measured.1 > 0.04 {
            return Err(format!("R30 attention parity failed: {measured:?}").into());
        }
        Some(measured)
    } else {
        None
    };
    let runtime_metrics = if attention {
        let mut resident = hipfire_xdna::NpuResidentAttentionDenseW8::load_cached(&args[0])?;
        let group_refs = weights.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let scale_refs = weight_scales.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let awq = vec![1.0f32; K];
        let resident_weights = resident.upload_dense_groups(
            &group_refs,
            &scale_refs,
            Some(&awq),
            &qnorm,
            &knorm,
            EPSILON,
            10_000.0,
        )?;
        if resident_weights.awq_scale() != Some(awq.as_slice()) {
            return Err("resident attention did not preserve generic AWQ metadata".into());
        }
        let resident_input = hipfire_xdna::NpuResidentAttentionDenseW8::prepack_activations(
            &activations,
            &activation_scales,
        )?;
        resident.set_prepacked_input(&resident_input)?;
        resident.run_shared_to_device(&resident_weights)?;
        let got = resident.read_output_f32(&resident_weights)?;
        let reference = attention_reference(&q_got, &k_got, &v_got);
        let measured = metrics(&got, &reference);
        if !measured.0.is_finite() || measured.0 < 0.998 || measured.1 > 0.04 {
            return Err(format!("resident runtime attention parity failed: {measured:?}").into());
        }
        Some(measured)
    } else {
        None
    };

    let started = std::time::Instant::now();
    for _ in 0..iterations {
        dispatch_resident(
            &kernel, &a_buffer, &w_buffer, &r_buffer, &q_buffer, &kv_buffer, false,
        )?;
    }
    let dispatch_ms = started.elapsed().as_secs_f64() * 1e3 / iterations as f64;
    if let (Some(attention), Some(runtime)) = (attention_metrics, runtime_metrics) {
        println!(
            "resident-w8-qkv-attention M=256 K=768 N=1280: projection_cosine={:.8} projection_max={:.7} q_cosine={:.8} q_max={:.7} k_cosine={:.8} k_max={:.7} v_bit_mismatches={v_mismatches} attention_cosine={:.8} attention_max={:.7} runtime_cosine={:.8} runtime_max={:.7} dispatch_ms={dispatch_ms:.4}",
            projection.0,
            projection.1,
            q.0,
            q.1,
            k.0,
            k.1,
            attention.0,
            attention.1,
            runtime.0,
            runtime.1,
        );
    } else {
        println!(
            "resident-w8-qkv-pack M=256 K=768 N=1280: projection_cosine={:.8} projection_max={:.7} q_cosine={:.8} q_max={:.7} k_cosine={:.8} k_max={:.7} v_bit_mismatches={v_mismatches} dispatch_ms={dispatch_ms:.4}",
            projection.0, projection.1, q.0, q.1, k.0, k.1
        );
    }
    Ok(())
}

#[cfg(target_os = "linux")]
#[allow(clippy::too_many_arguments)]
fn dispatch_resident(
    kernel: &hipfire_xdna::NpuKernel,
    activations: &hipfire_xdna::DeviceBuffer,
    weights: &hipfire_xdna::DeviceBuffer,
    staging: &hipfire_xdna::DeviceBuffer,
    queries: &hipfire_xdna::DeviceBuffer,
    key_values: &hipfire_xdna::DeviceBuffer,
    sync: bool,
) -> Result<(), hipfire_xdna::XdnaError> {
    let args = [activations, weights, staging, queries, key_values];
    let sync_flags = vec![sync; args.len()];
    kernel.dispatch_synced(&args, &sync_flags)
}

#[cfg(target_os = "linux")]
fn unpack_attention(bytes: &[u8]) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let bits = hipfire_xdna::EmbeddingGemmaAttentionLayout::unpack_output_bf16(bytes)
        .ok_or("invalid R30 physical attention output")?;
    Ok(bits
        .into_iter()
        .map(hipfire_primitives::conv::bf16_bits_to_f32)
        .collect())
}

#[cfg(target_os = "linux")]
fn attention_reference(q: &[f32], k: &[f32], v: &[f32]) -> Vec<f32> {
    let mut output = vec![0.0f32; q.len()];
    let mut scores = vec![0.0f32; 256];
    for head in 0..3 {
        for query in 0..256 {
            let qrow = &q[(head * 256 + query) * 256..(head * 256 + query + 1) * 256];
            for key in 0..256 {
                scores[key] = qrow
                    .iter()
                    .zip(&k[key * 256..(key + 1) * 256])
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
                &mut output[(head * 256 + query) * 256..(head * 256 + query + 1) * 256];
            for key in 0..256 {
                let probability = scores[key] / sum;
                for dim in 0..256 {
                    destination[dim] += probability * v[key * 256 + dim];
                }
            }
        }
    }
    output
}

#[cfg(target_os = "linux")]
fn cpu_projection(
    activations: &[i8],
    activation_scales: &[f32],
    weights: &[Vec<i8>],
    weight_scales: &[Vec<f32>],
) -> Vec<f32> {
    use hipfire_primitives::conv::{bf16_bits_to_f32, f32_to_bf16_bits};
    const M: usize = 256;
    const K: usize = 768;
    const N: usize = 1280;
    let mut output = vec![0.0; M * N];
    for row in 0..M {
        for col in 0..N {
            let mut sum = 0.0f32;
            for group in 0..3 {
                let mut dot = 0i32;
                for inner in 0..256 {
                    dot += activations[row * K + group * 256 + inner] as i32
                        * weights[group][inner * N + col] as i32;
                }
                let scaled = dot as f32 * weight_scales[group][col];
                sum += scaled * activation_scales[group * M + row];
            }
            output[row * N + col] = bf16_bits_to_f32(f32_to_bf16_bits(sum));
        }
    }
    output
}

#[cfg(target_os = "linux")]
fn pack_activations(values: &[i8], scales: &[f32], block_bytes: usize) -> Vec<u8> {
    const M: usize = 256;
    const K: usize = 768;
    const OUTBLOCKS: usize = 15;
    let mut packed = vec![0u8; 4 * 45 * block_bytes];
    for stripe in 0..4 {
        for m_macro in 0..3 {
            for n_macro in 0..5 {
                let outblock = m_macro * 5 + n_macro;
                for group in 0..3 {
                    let block = outblock * 3 + group;
                    let base = (stripe * OUTBLOCKS * 3 + block) * block_bytes;
                    for lm in 0..3 {
                        for kt in 0..32 {
                            for local_row in 0..8 {
                                let row = m_macro * 96 + stripe * 24 + lm * 8 + local_row;
                                if row < M {
                                    let source = row * K + group * 256 + kt * 8;
                                    let target = base + (lm * 32 + kt) * 64 + local_row * 8;
                                    for lane in 0..8 {
                                        packed[target + lane] = values[source + lane] as u8;
                                    }
                                }
                            }
                        }
                    }
                    for local_row in 0..24 {
                        let row = m_macro * 96 + stripe * 24 + local_row;
                        let scale = if row < M {
                            scales[group * M + row]
                        } else {
                            0.0
                        };
                        packed[base + 6144 + local_row * 4..base + 6148 + local_row * 4]
                            .copy_from_slice(&scale.to_ne_bytes());
                    }
                }
            }
        }
    }
    packed
}

#[cfg(target_os = "linux")]
fn pack_weights(weights: &[Vec<i8>], scales: &[Vec<f32>]) -> Vec<u8> {
    const N: usize = 1280;
    const BLOCK: usize = 16384;
    const OUTBLOCKS: usize = 15;
    let mut packed = vec![0u8; 8 * 45 * BLOCK];
    for stripe in 0..8 {
        for m_macro in 0..3 {
            for n_macro in 0..5 {
                let outblock = m_macro * 5 + n_macro;
                for group in 0..3 {
                    let block = outblock * 3 + group;
                    let base = (stripe * OUTBLOCKS * 3 + block) * BLOCK;
                    for ln in 0..2 {
                        for kt in 0..32 {
                            for kk in 0..8 {
                                for nn in 0..16 {
                                    let col = n_macro * 256 + stripe * 32 + ln * 16 + nn;
                                    let index =
                                        (ln * 32 + kt) * 128 + (nn / 8) * 64 + kk * 8 + nn % 8;
                                    packed[base + index] =
                                        weights[group][(kt * 8 + kk) * N + col] as u8;
                                }
                            }
                        }
                    }
                    for local_col in 0..32 {
                        let col = n_macro * 256 + stripe * 32 + local_col;
                        let offset = base + 8192 + local_col * 4;
                        packed[offset..offset + 4]
                            .copy_from_slice(&scales[group][col].to_ne_bytes());
                    }
                }
            }
        }
    }
    packed
}

#[cfg(target_os = "linux")]
fn stage_positions_and_params(
    cs: &[u16],
    qnorm: &[f32],
    knorm: &[f32],
    eps: f32,
    pair_bytes: usize,
) -> Vec<u8> {
    use hipfire_primitives::conv::f32_to_bf16_bits;
    let mut staged = vec![0u8; 5 * 48 * pair_bytes];
    for role in 0..5 {
        for physical_pair in 0..48 {
            let base = (role * 48 + physical_pair) * pair_bytes;
            let m_macro = physical_pair / 16;
            let within = physical_pair % 16;
            let core_row = within / 4;
            let subpair = within % 4;
            if subpair < 3 {
                let token0 = m_macro * 96 + core_row * 24 + subpair * 8;
                for row in 0..8 {
                    if token0 + row < 256 {
                        for dim in 0..256 {
                            write_u16(
                                &mut staged,
                                base + 4096 + (row * 256 + dim) * 2,
                                cs[(token0 + row) * 256 + dim],
                            );
                        }
                    }
                }
            }
            for (index, &value) in qnorm.iter().enumerate() {
                write_u16(
                    &mut staged,
                    base + 8192 + index * 2,
                    f32_to_bf16_bits(value),
                );
            }
            for (index, &value) in knorm.iter().enumerate() {
                write_u16(
                    &mut staged,
                    base + 8192 + 512 + index * 2,
                    f32_to_bf16_bits(value),
                );
            }
            staged[base + 8192 + 1024..base + 8192 + 1028].copy_from_slice(&eps.to_le_bytes());
        }
    }
    staged
}

#[cfg(target_os = "linux")]
fn physical_pair(logical_pair: usize) -> usize {
    let token = logical_pair * 8;
    let m_macro = token / 96;
    let within = token % 96;
    m_macro * 16 + (within / 24) * 4 + (within % 24) / 8
}

#[cfg(target_os = "linux")]
fn read_projected(bytes: &[u8], pair_bytes: usize) -> Vec<f32> {
    use hipfire_primitives::conv::bf16_bits_to_f32;
    let mut output = vec![0.0; 256 * 1280];
    for role in 0..5 {
        for token in 0..256 {
            let pair = physical_pair(token / 8);
            let row = token % 8;
            for dim in 0..256 {
                let offset = (role * 48 + pair) * pair_bytes + (row * 256 + dim) * 2;
                output[token * 1280 + role * 256 + dim] =
                    bf16_bits_to_f32(u16::from_le_bytes([bytes[offset], bytes[offset + 1]]));
            }
        }
    }
    output
}

#[cfg(target_os = "linux")]
fn role_q(projected: &[f32]) -> Vec<f32> {
    let mut output = vec![0.0; 3 * 256 * 256];
    for head in 0..3 {
        for token in 0..256 {
            output[(head * 256 + token) * 256..(head * 256 + token + 1) * 256].copy_from_slice(
                &projected[token * 1280 + head * 256..token * 1280 + (head + 1) * 256],
            );
        }
    }
    output
}

#[cfg(target_os = "linux")]
fn role_kv(projected: &[f32], role: usize) -> Vec<f32> {
    let mut output = vec![0.0; 256 * 256];
    for token in 0..256 {
        output[token * 256..(token + 1) * 256].copy_from_slice(
            &projected[token * 1280 + role * 256..token * 1280 + (role + 1) * 256],
        );
    }
    output
}

#[cfg(target_os = "linux")]
fn bf16_values(length: usize, value: impl Fn(usize) -> f32) -> Vec<f32> {
    use hipfire_primitives::conv::{bf16_bits_to_f32, f32_to_bf16_bits};
    (0..length)
        .map(|index| bf16_bits_to_f32(f32_to_bf16_bits(value(index))))
        .collect()
}

#[cfg(target_os = "linux")]
fn rope_cs(base: f32) -> Vec<u16> {
    use hipfire_primitives::conv::f32_to_bf16_bits;
    let mut cs = vec![0; 256 * 256];
    for token in 0..256 {
        for dim in 0..128 {
            let frequency = 1.0 / base.powf((2 * dim) as f32 / 256.0);
            let angle = token as f32 * frequency;
            cs[token * 256 + dim] = f32_to_bf16_bits(angle.cos());
            cs[token * 256 + 128 + dim] = f32_to_bf16_bits(angle.sin());
        }
    }
    cs
}

#[cfg(target_os = "linux")]
fn headnorm_rope(input: &[f32], weight: &[f32], cs: &[u16], heads: usize, eps: f32) -> Vec<f32> {
    use hipfire_primitives::conv::{bf16_bits_to_f32, f32_to_bf16_bits};
    let mut output = vec![0.0; input.len()];
    for head in 0..heads {
        for token in 0..256 {
            let base = (head * 256 + token) * 256;
            let row = &input[base..base + 256];
            let inv = 1.0 / (row.iter().map(|x| x * x).sum::<f32>() / 256.0 + eps).sqrt();
            for dim in 0..128 {
                let x = row[dim] * weight[dim] * inv;
                let y = row[128 + dim] * weight[128 + dim] * inv;
                let cosine = bf16_bits_to_f32(cs[token * 256 + dim]);
                let sine = bf16_bits_to_f32(cs[token * 256 + 128 + dim]);
                output[base + dim] = bf16_bits_to_f32(f32_to_bf16_bits(x * cosine - y * sine));
                output[base + 128 + dim] =
                    bf16_bits_to_f32(f32_to_bf16_bits(y * cosine + x * sine));
            }
        }
    }
    output
}

#[cfg(target_os = "linux")]
fn read_q(bytes: &[u8]) -> Vec<f32> {
    use hipfire_primitives::conv::bf16_bits_to_f32;
    use hipfire_xdna::EmbeddingGemmaAttentionLayout as Layout;
    let mut output = vec![0.0; 3 * 256 * 256];
    for head in 0..3 {
        for token in 0..256 {
            for dim in 0..256 {
                let offset = Layout::q_offset(head, token, dim).unwrap();
                output[(head * 256 + token) * 256 + dim] =
                    bf16_bits_to_f32(u16::from_le_bytes([bytes[offset], bytes[offset + 1]]));
            }
        }
    }
    output
}

#[cfg(target_os = "linux")]
fn read_kv(bytes: &[u8], key: bool) -> Vec<f32> {
    use hipfire_primitives::conv::bf16_bits_to_f32;
    use hipfire_xdna::EmbeddingGemmaAttentionLayout as Layout;
    let mut output = vec![0.0; 256 * 256];
    for token in 0..256 {
        for dim in 0..256 {
            let offset = if key {
                Layout::k_offset(token, dim)
            } else {
                Layout::v_offset(token, dim)
            }
            .unwrap();
            output[token * 256 + dim] =
                bf16_bits_to_f32(u16::from_le_bytes([bytes[offset], bytes[offset + 1]]));
        }
    }
    output
}

#[cfg(target_os = "linux")]
fn write_u16(destination: &mut [u8], offset: usize, value: u16) {
    destination[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
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

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("npu_embedding_qkv_resident_verify is Linux-only");
}
