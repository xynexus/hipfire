//! CPU-oracle gate for the resident R29 W8 QKV projection-to-attention pack.

#[cfg(target_os = "linux")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use hipfire_primitives::conv::f32_to_bf16_bits;
    use hipfire_xdna::{EmbeddingGemmaAttentionLayout as Layout, NpuKernel};

    const K: usize = 768;
    const N: usize = 1280;
    const GROUPS: usize = 3;
    const EPSILON: f32 = 1.0e-6;

    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if !(1..=3).contains(&args.len()) {
        return Err("usage: npu_embedding_qkv_resident_verify CACHE [ITERS] [FFN_CACHE]".into());
    }
    let iterations = args
        .get(1)
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(20);
    let ffn_cache = args.get(2).cloned().unwrap_or_else(|| {
        format!(
            "{}/.hipfire/npu/embgemma_aie2p_resident_ffn_dense_w8_canonical_bf16_m256_k768_i1152_o768",
            std::env::var("HOME").expect("HOME")
        )
    });
    if iterations == 0 {
        return Err("R29 verifier needs at least one iteration".into());
    }
    let manifest = std::fs::read_to_string(format!("{}/shape.txt", args[0]))?;
    let residual_norm = manifest
        .lines()
        .any(|line| line == "op=resident-qkv-paired-attention-output-norm");
    let paired_qkv = residual_norm
        || manifest
            .lines()
            .any(|line| line == "op=resident-qkv-paired-attention-output-direct");
    let direct_output = paired_qkv
        || manifest
            .lines()
            .any(|line| line == "op=resident-qkv-attention-output-direct");
    let output_projection = direct_output
        || manifest
            .lines()
            .any(|line| line == "op=resident-qkv-attention-o");
    let packed_attention = manifest
        .lines()
        .any(|line| line == "op=resident-qkv-attention-packed");
    let attention = output_projection
        || packed_attention
        || manifest
            .lines()
            .any(|line| line == "op=resident-qkv-attention");
    let operation = if residual_norm {
        "op=resident-qkv-paired-attention-output-norm"
    } else if paired_qkv {
        "op=resident-qkv-paired-attention-output-direct"
    } else if direct_output {
        "op=resident-qkv-attention-output-direct"
    } else if output_projection {
        "op=resident-qkv-attention-o"
    } else if packed_attention {
        "op=resident-qkv-attention-packed"
    } else if attention {
        "op=resident-qkv-attention"
    } else {
        "op=resident-qkv-headnorm-rope-pack"
    };
    let mode = if direct_output {
        "mode=w8-scaled"
    } else if output_projection {
        "mode=w8-qkv-bf16-o"
    } else {
        "mode=w8-scaled"
    };
    let roles = if output_projection {
        "roles=q0,q1,q2,k,v,o"
    } else {
        "roles=q0,q1,q2,k,v"
    };
    for field in [operation, mode, "m=256", "k=768", "n=1280", roles] {
        if !manifest.lines().any(|line| line == field) {
            return Err(format!("R29 cache missing {field}").into());
        }
    }
    if residual_norm
        && !manifest
            .lines()
            .any(|line| line == "handoff=staging-prefix-dmabuf")
    {
        return Err("R34 cache missing shared staging-prefix handoff contract".into());
    }
    if residual_norm
        && ![
            "state=pre-ffn-inverse-f32",
            "state-layout=active-column,core-row,wave,row",
        ]
        .iter()
        .all(|field| manifest.lines().any(|line| line == *field))
    {
        return Err("R38 cache missing pre-FFN inverse state contract".into());
    }
    let pair_bytes = if attention { 16384 } else { 10240 };
    let a_bytes = 4 * 45 * pair_bytes;
    let r_stage_bytes = 5 * 48 * pair_bytes;
    let r_bytes = r_stage_bytes
        + if attention { Layout::OUTPUT_BYTES } else { 0 }
        + if direct_output {
            2 * Layout::OUTPUT_BYTES
        } else if output_projection {
            Layout::OUTPUT_BYTES
        } else {
            0
        };
    let qkv_w_bytes = (if paired_qkv { 4 } else { 8 }) * 45 * 16384;
    let w_bytes = qkv_w_bytes
        + if direct_output {
            4 * 72 * 16384
        } else if output_projection {
            4 * 18 * 16384
        } else {
            0
        }
        + if residual_norm { 4 * 8 * 16384 } else { 0 };

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

    let mut packed_a = pack_activations(&activations, &activation_scales, pair_bytes);
    let output_weights = output_projection_weights();
    let unpacked_w = pack_weights(&weights, &weight_scales);
    let mut packed_w = if paired_qkv {
        inject_paired_weight_scales(&mut packed_a, &unpacked_w, pair_bytes);
        pack_weights_paired(&unpacked_w)
    } else {
        unpacked_w
    };
    if direct_output {
        packed_w.extend_from_slice(&pack_output_projection_direct(&output_weights));
    } else if output_projection {
        packed_w.extend_from_slice(&pack_output_projection(&output_weights));
    }
    if residual_norm {
        packed_w.extend_from_slice(&pack_residual_norm_params(EPSILON));
    }
    let raw_staging = stage_positions_and_params(&cs, &qnorm, &knorm, EPSILON, pair_bytes);
    let raw_base = if packed_attention {
        Layout::OUTPUT_BYTES
    } else {
        0
    };
    let mut staged_r = vec![0u8; raw_base];
    staged_r.extend_from_slice(&raw_staging);
    staged_r.resize(r_bytes, 0);
    assert_eq!(packed_a.len(), a_bytes);
    assert_eq!(packed_w.len(), w_bytes);
    assert_eq!(staged_r.len(), r_bytes);

    let kernel = NpuKernel::load(
        &std::fs::read(format!("{}/final.xclbin", args[0]))?,
        &std::fs::read(format!("{}/insts.bin", args[0]))?,
    )?;
    let mut a_buffer = kernel.alloc_arg(a_bytes)?;
    let mut w_buffer = kernel.alloc_arg(w_bytes)?;
    let shared_handoff = if residual_norm {
        let gpu = hipfire_rdna::Gpu::init()?;
        let mut shared = gpu.alloc_shared_gtt(r_bytes)?;
        shared.as_mut_slice().fill(0);
        Some(shared)
    } else {
        None
    };
    let mut r_buffer = if let Some(shared) = shared_handoff.as_ref() {
        kernel.import_dmabuf(shared.dmabuf_fd(), shared.len(), true)?
    } else {
        kernel.alloc_arg(r_bytes)?
    };
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
    if output_projection || packed_attention {
        kernel.sync_output(&r_buffer)?;
        let attention_region = if packed_attention {
            &r_buffer.as_slice()[..Layout::OUTPUT_BYTES]
        } else {
            &r_buffer.as_slice()[r_stage_bytes..r_stage_bytes + Layout::OUTPUT_BYTES]
        };
        let prime_output_nonzero = if residual_norm {
            r_buffer.as_slice()[..Layout::TOKENS * 768 * 2]
                .iter()
                .filter(|&&byte| byte != 0)
                .count()
        } else if direct_output {
            r_buffer.as_slice()[r_stage_bytes + Layout::OUTPUT_BYTES..]
                .iter()
                .filter(|&&byte| byte != 0)
                .count()
        } else if output_projection {
            r_buffer.as_slice()
                [r_stage_bytes + Layout::OUTPUT_BYTES..r_stage_bytes + 2 * Layout::OUTPUT_BYTES]
                .iter()
                .filter(|&&byte| byte != 0)
                .count()
        } else {
            0
        };
        if direct_output {
            eprintln!("R32 prime: output_nonzero={prime_output_nonzero}");
        } else {
            eprintln!(
                "R31 prime tails: attention_nonzero={} output_nonzero={}",
                attention_region.iter().filter(|&&byte| byte != 0).count(),
                prime_output_nonzero,
            );
        }
    }
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
    let projected_got = if residual_norm {
        projected.clone()
    } else {
        read_projected(&r_buffer.as_slice()[raw_base..], pair_bytes)
    };
    let k_got = read_kv(kv_buffer.as_slice(), true);
    let v_got = read_kv(kv_buffer.as_slice(), false);
    let projection = metrics(&projected_got, &projected);
    let q_handoff = role_q(&projected_got);
    let k_handoff = role_kv(&projected_got, 3);
    let v_handoff = role_kv(&projected_got, 4);
    let q_handoff_reference = headnorm_rope(&q_handoff, &qnorm, &cs, Layout::QUERY_HEADS, EPSILON);
    let k_handoff_reference = headnorm_rope(&k_handoff, &knorm, &cs, Layout::KV_HEADS, EPSILON);
    let q_got = read_q(q_buffer.as_slice());
    let q = metrics(&q_got, &q_handoff_reference);
    let k = metrics(&k_got, &k_handoff_reference);
    let v = metrics(&v_got, &v_handoff);
    let v_mismatches = v_got
        .iter()
        .map(|&value| f32_to_bf16_bits(value))
        .zip(v_handoff.iter().map(|&value| f32_to_bf16_bits(value)))
        .filter(|(got, expected)| got != expected)
        .count();
    if projection.0 < 0.9999 || projection.1 > 0.01 {
        return Err(format!("R29 projection parity failed: {projection:?}").into());
    }
    if q.0 < 0.999
        || q.1 > 0.04
        || k.0 < 0.999
        || k.1 > 0.04
        || if residual_norm {
            v.0 < 0.9999 || v.1 > 0.001
        } else {
            v_mismatches != 0
        }
    {
        let first_v_mismatch = v_got
            .iter()
            .zip(&v_handoff)
            .position(|(&got, &expected)| f32_to_bf16_bits(got) != f32_to_bf16_bits(expected));
        let k_non_finite = k_got.iter().filter(|value| !value.is_finite()).count();
        return Err(format!(
            "R29 pack parity failed: projection={projection:?} q={q:?} k={k:?} k_non_finite={k_non_finite} v={v:?} v_bit_mismatches={v_mismatches} first_v_mismatch={first_v_mismatch:?}; k[0..8]={:?} k_ref[0..8]={:?}; v[0..8]={:?} v_ref[0..8]={:?}",
            &k_got[..8],
            &k_handoff_reference[..8],
            &v_got[..8],
            &v_handoff[..8],
        )
        .into());
    }

    let attention_reference_values = attention_reference(&q_got, &k_got, &v_got);
    let attention_metrics = if attention && !direct_output {
        let got = if packed_attention {
            unpack_projection_attention(&r_buffer.as_slice()[..Layout::OUTPUT_BYTES])
        } else {
            unpack_attention(
                &r_buffer.as_slice()[r_stage_bytes..r_stage_bytes + Layout::OUTPUT_BYTES],
            )?
        };
        let measured = metrics(&got, &attention_reference_values);
        if !measured.0.is_finite() || measured.0 < 0.998 || measured.1 > 0.04 {
            let physical = if packed_attention {
                &r_buffer.as_slice()[..Layout::OUTPUT_BYTES]
            } else {
                &r_buffer.as_slice()[r_stage_bytes..r_stage_bytes + Layout::OUTPUT_BYTES]
            };
            let output_nonzero = if output_projection {
                r_buffer.as_slice()
                    [r_stage_bytes + Layout::OUTPUT_BYTES..r_stage_bytes + 2 * Layout::OUTPUT_BYTES]
                    .iter()
                    .filter(|&&byte| byte != 0)
                    .count()
            } else {
                0
            };
            return Err(format!(
                "R30 attention parity failed: {measured:?}; attention_nonzero={} output_nonzero={output_nonzero}",
                physical.iter().filter(|&&byte| byte != 0).count()
            )
            .into());
        }
        Some(measured)
    } else {
        None
    };
    let output_reference =
        output_projection_reference(&attention_reference_values, &output_weights);
    let mut norm_bf16 = None;
    let norm_metrics = if residual_norm {
        let output_bytes = Layout::TOKENS * 768 * 2;
        let output = &r_buffer.as_slice()[..output_bytes];
        let got_bits = output
            .chunks_exact(2)
            .map(|bytes| u16::from_le_bytes([bytes[0], bytes[1]]))
            .collect::<Vec<_>>();
        let got = got_bits
            .iter()
            .copied()
            .map(hipfire_primitives::conv::bf16_bits_to_f32)
            .collect::<Vec<_>>();
        let (reference, inverse_reference, residual_reference) =
            residual_norm_reference(&output_reference, EPSILON);
        let measured = metrics(&got, &reference);
        if !measured.0.is_finite() || measured.0 < 0.9998 || measured.1 > 0.065 {
            return Err(format!(
                "R34 full residual/norm parity failed: {measured:?}; nonfinite={} nonzero={} got={:?} ref={:?}",
                got.iter().filter(|value| !value.is_finite()).count(),
                got.iter().filter(|&&value| value != 0.0).count(),
                &got[..16],
                &reference[..16],
            )
            .into());
        }
        let inverse =
            read_pre_inverse_metadata(r_buffer.as_slice(), r_stage_bytes + Layout::OUTPUT_BYTES);
        let inverse_measured = metrics(&inverse, &inverse_reference);
        if !inverse_measured.0.is_finite()
            || inverse_measured.0 < 0.9999
            || inverse_measured.1 > 0.02
        {
            return Err(format!(
                "R38 pre-FFN inverse metadata parity failed: {inverse_measured:?}"
            )
            .into());
        }
        let reconstructed = got
            .iter()
            .enumerate()
            .map(|(index, &value)| {
                let token = index / 768;
                let hidden = index % 768;
                let pre_norm = hipfire_primitives::conv::bf16_bits_to_f32(
                    hipfire_primitives::conv::f32_to_bf16_bits(
                        0.91 + (hidden % 29) as f32 * 0.0015,
                    ),
                );
                value / (pre_norm * inverse[token])
            })
            .collect::<Vec<_>>();
        let reconstructed_measured = metrics(&reconstructed, &residual_reference);
        if !reconstructed_measured.0.is_finite()
            || reconstructed_measured.0 < 0.9998
            || reconstructed_measured.1 > 0.08
        {
            return Err(format!(
                "R38 attention residual reconstruction failed: {reconstructed_measured:?}"
            )
            .into());
        }
        norm_bf16 = Some(got_bits);
        Some((measured, inverse_measured, reconstructed_measured))
    } else {
        None
    };
    let ffn_chain_metrics = if let Some(input) = norm_bf16.as_deref() {
        let shared = shared_handoff
            .as_ref()
            .expect("R34 shared canonical output backing");
        Some(verify_canonical_ffn_handoff(
            &ffn_cache,
            input,
            shared.dmabuf_fd(),
            shared.len(),
            || {
                dispatch_resident(
                    &kernel, &a_buffer, &w_buffer, &r_buffer, &q_buffer, &kv_buffer, false,
                )
            },
        )?)
    } else {
        None
    };
    let output_metrics = if output_projection && !residual_norm {
        let output = &r_buffer.as_slice()[r_stage_bytes + Layout::OUTPUT_BYTES..];
        let got = if direct_output {
            output
                .chunks_exact(4)
                .map(|word| f32::from_le_bytes([word[0], word[1], word[2], word[3]]))
                .collect::<Vec<_>>()
        } else {
            output[..Layout::OUTPUT_BYTES]
                .chunks_exact(2)
                .map(|pair| {
                    hipfire_primitives::conv::bf16_bits_to_f32(u16::from_le_bytes([
                        pair[0], pair[1],
                    ]))
                })
                .collect::<Vec<_>>()
        };
        let measured = metrics(&got, &output_reference);
        if !measured.0.is_finite() || measured.0 < 0.998 || measured.1 > 0.04 {
            return Err(format!(
                "R31/R32 output projection parity failed: {measured:?}; nonfinite={} nonzero={} got={:?} ref={:?}",
                got.iter().filter(|value| !value.is_finite()).count(),
                got.iter().filter(|&&value| value != 0.0).count(),
                &got[..16],
                &output_reference[..16],
            )
            .into());
        }
        Some(measured)
    } else {
        None
    };
    let mut chained_output_dispatch_ms = None;
    let chained_output_metrics = if attention && !output_projection {
        if let Ok(cache) = std::env::var("HIPFIRE_R31_O_CACHE") {
            let mut output_executor = hipfire_xdna::NpuAttentionOutputBf16::load_cached(&cache)?;
            let output_weights = output_executor.upload_bf16(&output_weights)?;
            let packed = if packed_attention {
                r_buffer.as_slice()[..Layout::OUTPUT_BYTES].to_vec()
            } else {
                pack_attention_for_output_projection(
                    &r_buffer.as_slice()[r_stage_bytes..r_stage_bytes + Layout::OUTPUT_BYTES],
                )
            };
            output_executor.set_input(&packed)?;
            output_executor.run(&output_weights)?;
            let got = output_executor.read_output_f32()?;
            let measured = metrics(&got, &output_reference);
            if !measured.0.is_finite() || measured.0 < 0.998 || measured.1 > 0.04 {
                return Err(format!(
                    "R31 chained output parity failed: {measured:?}; nonzero={} got={:?} ref={:?}",
                    got.iter().filter(|&&value| value != 0.0).count(),
                    &got[..8],
                    &output_reference[..8]
                )
                .into());
            }
            let output_iterations = std::env::var("HIPFIRE_R31_O_ITERS")
                .ok()
                .map(|value| value.parse::<usize>())
                .transpose()?
                .unwrap_or(20);
            if output_iterations == 0 {
                return Err("HIPFIRE_R31_O_ITERS must be non-zero".into());
            }
            let started = std::time::Instant::now();
            for _ in 0..output_iterations {
                output_executor.run(&output_weights)?;
            }
            chained_output_dispatch_ms =
                Some(started.elapsed().as_secs_f64() * 1e3 / output_iterations as f64);
            Some(measured)
        } else {
            None
        }
    } else {
        None
    };
    let runtime_metrics = if attention && !packed_attention && !output_projection {
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
        let measured = metrics(&got, &attention_reference_values);
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
    if residual_norm {
        let (norm, inverse, reconstructed) = norm_metrics.expect("residual norm metrics");
        let ffn = ffn_chain_metrics.expect("R34 to R35 FFN metrics");
        println!(
            "resident-w8-qkv-attention-output-norm M=256 K=768 N=1280: staging_prefix_reused_for_norm=true q_cosine={:.8} q_max={:.7} k_cosine={:.8} k_max={:.7} v_cosine={:.8} v_max={:.7} norm_full_cosine={:.8} norm_full_max={:.7} pre_inverse_cosine={:.8} pre_inverse_max={:.7} residual_reconstruct_cosine={:.8} residual_reconstruct_max={:.7} ffn_zero_copy_cosine={:.8} ffn_zero_copy_max={:.7} ffn_zero_copy_ms={:.4} zero_copy_chain_ms={:.4} zero_copy_chain_rows_s={:.1} dispatch_ms={dispatch_ms:.4}",
            q.0,
            q.1,
            k.0,
            k.1,
            v.0,
            v.1,
            norm.0,
            norm.1,
            inverse.0,
            inverse.1,
            reconstructed.0,
            reconstructed.1,
            ffn.0,
            ffn.1,
            ffn.2,
            ffn.3,
            256_000.0 / ffn.3,
        );
    } else if direct_output {
        let output = output_metrics.expect("direct output projection metrics");
        println!(
            "resident-w8-qkv-attention-output-direct M=256 K=768 N=1280: projection_cosine={:.8} projection_max={:.7} q_cosine={:.8} q_max={:.7} k_cosine={:.8} k_max={:.7} v_bit_mismatches={v_mismatches} output_cosine={:.8} output_max={:.7} dispatch_ms={dispatch_ms:.4}",
            projection.0,
            projection.1,
            q.0,
            q.1,
            k.0,
            k.1,
            output.0,
            output.1,
        );
    } else if let (Some(attention), Some(output), Some(runtime)) =
        (attention_metrics, output_metrics, runtime_metrics)
    {
        println!(
            "resident-w8-qkv-attention-o M=256 K=768 N=1280: projection_cosine={:.8} projection_max={:.7} q_cosine={:.8} q_max={:.7} k_cosine={:.8} k_max={:.7} v_bit_mismatches={v_mismatches} attention_cosine={:.8} attention_max={:.7} output_cosine={:.8} output_max={:.7} runtime_cosine={:.8} runtime_max={:.7} dispatch_ms={dispatch_ms:.4}",
            projection.0,
            projection.1,
            q.0,
            q.1,
            k.0,
            k.1,
            attention.0,
            attention.1,
            output.0,
            output.1,
            runtime.0,
            runtime.1,
        );
    } else if let (Some(attention), Some(runtime)) = (attention_metrics, runtime_metrics) {
        println!(
            "resident-w8-qkv-attention M=256 K=768 N=1280: projection_cosine={:.8} projection_max={:.7} q_cosine={:.8} q_max={:.7} k_cosine={:.8} k_max={:.7} v_bit_mismatches={v_mismatches} attention_cosine={:.8} attention_max={:.7} chained_output={:?} chained_output_ms={:?} runtime_cosine={:.8} runtime_max={:.7} dispatch_ms={dispatch_ms:.4}",
            projection.0,
            projection.1,
            q.0,
            q.1,
            k.0,
            k.1,
            attention.0,
            attention.1,
            chained_output_metrics,
            chained_output_dispatch_ms,
            runtime.0,
            runtime.1,
        );
    } else if let (Some(attention), Some(chained)) = (attention_metrics, chained_output_metrics) {
        println!(
            "resident-w8-qkv-attention-packed M=256 K=768 N=1280: projection_cosine={:.8} projection_max={:.7} q_cosine={:.8} q_max={:.7} k_cosine={:.8} k_max={:.7} v_bit_mismatches={v_mismatches} attention_cosine={:.8} attention_max={:.7} chained_output_cosine={:.8} chained_output_max={:.7} chained_output_ms={:?} dispatch_ms={dispatch_ms:.4}",
            projection.0,
            projection.1,
            q.0,
            q.1,
            k.0,
            k.1,
            attention.0,
            attention.1,
            chained.0,
            chained.1,
            chained_output_dispatch_ms,
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
fn unpack_projection_attention(bytes: &[u8]) -> Vec<f32> {
    let mut output = vec![0.0f32; 3 * 256 * 256];
    for head in 0..3 {
        for token in 0..256 {
            let linear = head * 256 + token;
            let group = linear / 128;
            let remainder = linear % 128;
            let core = remainder / 4;
            let core_row = core / 8;
            let col = core % 8;
            let query = remainder % 4;
            let block = (group * 4 + core_row) * 16384;
            for dim in 0..256 {
                let offset = block + col * 2048 + (dim / 8 * 4 + query) * 16 + dim % 8 * 2;
                output[(head * 256 + token) * 256 + dim] =
                    hipfire_primitives::conv::bf16_bits_to_f32(u16::from_le_bytes([
                        bytes[offset],
                        bytes[offset + 1],
                    ]));
            }
        }
    }
    output
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
fn output_projection_weights() -> Vec<u16> {
    let mut weights = vec![0u16; 768 * 768];
    for index in 0..768 {
        weights[index * 768 + index] =
            hipfire_primitives::conv::f32_to_bf16_bits(0.5 + (index % 17) as f32 * 0.01);
    }
    weights
}

#[cfg(target_os = "linux")]
fn pack_output_projection(weights: &[u16]) -> Vec<u8> {
    const BLOCK: usize = 16384;
    let mut packed = vec![0u8; 4 * 18 * BLOCK];
    for active_col in 0..4 {
        for slice in 0..6 {
            let column_base = active_col * 192 + slice * 32;
            for group in 0..3 {
                let block = (active_col * 18 + slice * 3 + group) * BLOCK;
                for nt in 0..4 {
                    for kt in 0..32 {
                        for kk in 0..8 {
                            for nn in 0..8 {
                                let k = group * 256 + kt * 8 + kk;
                                let n = column_base + nt * 8 + nn;
                                let target = block + ((nt * 32 + kt) * 64 + kk * 8 + nn) * 2;
                                packed[target..target + 2]
                                    .copy_from_slice(&weights[k * 768 + n].to_le_bytes());
                            }
                        }
                    }
                }
            }
        }
    }
    packed
}

#[cfg(target_os = "linux")]
fn pack_output_projection_direct(weights: &[u16]) -> Vec<u8> {
    const BLOCK: usize = 16384;
    let mut packed = vec![0u8; 4 * 72 * BLOCK];
    for active_col in 0..4 {
        for slice in 0..24 {
            let column_base = slice * 32;
            for group in 0..3 {
                let block = (active_col * 72 + slice * 3 + group) * BLOCK;
                for nt in 0..4 {
                    for kt in 0..32 {
                        for kk in 0..8 {
                            for nn in 0..8 {
                                let k = group * 256 + kt * 8 + kk;
                                let n = column_base + nt * 8 + nn;
                                let target = block + ((nt * 32 + kt) * 64 + kk * 8 + nn) * 2;
                                packed[target..target + 2]
                                    .copy_from_slice(&weights[k * 768 + n].to_le_bytes());
                            }
                        }
                    }
                }
            }
        }
    }
    packed
}

#[cfg(target_os = "linux")]
fn pack_residual_norm_params(epsilon: f32) -> Vec<u8> {
    use hipfire_primitives::conv::f32_to_bf16_bits;

    const BLOCK: usize = 16384;
    const HIDDEN: usize = 768;
    const ROWS_PER_CORE: usize = 8;
    const POST_NORM: usize = ROWS_PER_CORE * HIDDEN * 2;
    const PRE_NORM: usize = POST_NORM + HIDDEN * 2;
    const EPSILON: usize = PRE_NORM + HIDDEN * 2;
    let mut packed = vec![0u8; 4 * 2 * 4 * BLOCK];
    for active_col in 0..4 {
        for mwave in 0..2 {
            for core_row in 0..4 {
                let block = ((active_col * 2 + mwave) * 4 + core_row) * BLOCK;
                let token_base = mwave * 128 + core_row * 32 + active_col * 8;
                for row in 0..ROWS_PER_CORE {
                    for hidden in 0..HIDDEN {
                        let token = token_base + row;
                        let value = ((token * 17 + hidden * 7) % 97) as f32 * 0.0005 - 0.024;
                        write_u16(
                            &mut packed,
                            block + (row * HIDDEN + hidden) * 2,
                            f32_to_bf16_bits(value),
                        );
                    }
                }
                for hidden in 0..HIDDEN {
                    let post = 0.86 + (hidden % 31) as f32 * 0.002;
                    let pre = 0.91 + (hidden % 29) as f32 * 0.0015;
                    write_u16(
                        &mut packed,
                        block + POST_NORM + hidden * 2,
                        f32_to_bf16_bits(post),
                    );
                    write_u16(
                        &mut packed,
                        block + PRE_NORM + hidden * 2,
                        f32_to_bf16_bits(pre),
                    );
                }
                packed[block + EPSILON..block + EPSILON + 4]
                    .copy_from_slice(&epsilon.to_le_bytes());
            }
        }
    }
    packed
}

#[cfg(target_os = "linux")]
fn read_pre_inverse_metadata(bytes: &[u8], base: usize) -> Vec<f32> {
    const OUT_JOIN: usize = 8192;
    const OUT_TILE: usize = 2048;
    let mut output = vec![0.0f32; 256];
    for mwave in 0..2 {
        for active_col in 0..4 {
            for core_row in 0..4 {
                for row in 0..8 {
                    let token = mwave * 128 + core_row * 32 + active_col * 8 + row;
                    let offset = base
                        + active_col * OUT_JOIN
                        + core_row * OUT_TILE
                        + mwave * 8 * size_of::<f32>()
                        + row * size_of::<f32>();
                    output[token] = f32::from_le_bytes(
                        bytes[offset..offset + size_of::<f32>()]
                            .try_into()
                            .expect("pre-inverse metadata word"),
                    );
                }
            }
        }
    }
    output
}

#[cfg(target_os = "linux")]
fn residual_norm_reference(output: &[f32], epsilon: f32) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    use hipfire_primitives::conv::{bf16_bits_to_f32, f32_to_bf16_bits};

    const HIDDEN: usize = 768;
    let bf16 = |value: f32| bf16_bits_to_f32(f32_to_bf16_bits(value));
    let mut normalized = vec![0.0f32; output.len()];
    let mut inverse = vec![0.0f32; 256];
    let mut residual_output = vec![0.0f32; output.len()];
    let mut residual = vec![0.0f32; HIDDEN];
    for token in 0..256 {
        let source = &output[token * HIDDEN..(token + 1) * HIDDEN];
        let output_sum = source
            .iter()
            .map(|&value| {
                let value = bf16(value);
                value * value
            })
            .sum::<f32>();
        let post_inverse = (output_sum / HIDDEN as f32 + epsilon).sqrt().recip();
        for hidden in 0..HIDDEN {
            let output = bf16(source[hidden]);
            let post = bf16(0.86 + (hidden % 31) as f32 * 0.002);
            let input = bf16(((token * 17 + hidden * 7) % 97) as f32 * 0.0005 - 0.024);
            residual[hidden] = bf16(output * post * post_inverse + input);
        }
        let residual_sum = residual.iter().map(|value| value * value).sum::<f32>();
        let pre_inverse = (residual_sum / HIDDEN as f32 + epsilon).sqrt().recip();
        inverse[token] = pre_inverse;
        residual_output[token * HIDDEN..(token + 1) * HIDDEN].copy_from_slice(&residual);
        for hidden in 0..HIDDEN {
            let pre = bf16(0.91 + (hidden % 29) as f32 * 0.0015);
            normalized[token * HIDDEN + hidden] = bf16(residual[hidden] * pre * pre_inverse);
        }
    }
    (normalized, inverse, residual_output)
}

#[cfg(target_os = "linux")]
fn pack_attention_for_output_projection(physical: &[u8]) -> Vec<u8> {
    let mut packed = vec![0u8; 6 * 4 * 16384];
    for group in 0..6 {
        for core_row in 0..4 {
            let block = (group * 4 + core_row) * 16384;
            for col in 0..8 {
                for query in 0..4 {
                    for dim in 0..256 {
                        let source =
                            (col * 6 + group) * 8192 + core_row * 2048 + query * 512 + dim * 2;
                        let target = block + col * 2048 + (dim / 8 * 4 + query) * 16 + dim % 8 * 2;
                        packed[target..target + 2].copy_from_slice(&physical[source..source + 2]);
                    }
                }
            }
        }
    }
    packed
}

#[cfg(target_os = "linux")]
fn output_projection_reference(attention_head_major: &[f32], weights: &[u16]) -> Vec<f32> {
    use hipfire_primitives::conv::{bf16_bits_to_f32, f32_to_bf16_bits};
    let mut output = vec![0.0f32; 256 * 768];
    for token in 0..256 {
        for dim in 0..768 {
            let head = dim / 256;
            let head_dim = dim % 256;
            let input = attention_head_major[(head * 256 + token) * 256 + head_dim];
            let weight = bf16_bits_to_f32(weights[dim * 768 + dim]);
            output[token * 768 + dim] = bf16_bits_to_f32(f32_to_bf16_bits(input * weight));
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
fn pack_weights_paired(unpaired: &[u8]) -> Vec<u8> {
    const BLOCK: usize = 16384;
    const DATA: usize = 8192;
    const BLOCKS_PER_STRIPE: usize = 45;
    let mut paired = vec![0u8; 4 * BLOCKS_PER_STRIPE * BLOCK];
    for pair in 0..4 {
        for block in 0..BLOCKS_PER_STRIPE {
            let target = (pair * BLOCKS_PER_STRIPE + block) * BLOCK;
            for lane in 0..2 {
                let source = ((pair * 2 + lane) * BLOCKS_PER_STRIPE + block) * BLOCK;
                paired[target + lane * DATA..target + (lane + 1) * DATA]
                    .copy_from_slice(&unpaired[source..source + DATA]);
            }
        }
    }
    paired
}

#[cfg(target_os = "linux")]
fn inject_paired_weight_scales(activations: &mut [u8], unpaired: &[u8], block_bytes: usize) {
    const BLOCK: usize = 16384;
    const SCALE_OFFSET: usize = 8192;
    const SCALE_BYTES: usize = 128;
    const PAIRED_SCALE_BASE: usize = 6272;
    const BLOCKS_PER_STRIPE: usize = 45;
    for row_stripe in 0..4 {
        for block in 0..BLOCKS_PER_STRIPE {
            let activation = (row_stripe * BLOCKS_PER_STRIPE + block) * block_bytes;
            for pair in 0..4 {
                for lane in 0..2 {
                    let source =
                        ((pair * 2 + lane) * BLOCKS_PER_STRIPE + block) * BLOCK + SCALE_OFFSET;
                    let target = activation
                        + PAIRED_SCALE_BASE
                        + pair * 2 * SCALE_BYTES
                        + lane * SCALE_BYTES;
                    activations[target..target + SCALE_BYTES]
                        .copy_from_slice(&unpaired[source..source + SCALE_BYTES]);
                }
            }
        }
    }
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
fn verify_canonical_ffn_handoff<F>(
    cache: &str,
    input_bf16: &[u16],
    input_fd: i32,
    input_bytes: usize,
    mut produce_input: F,
) -> Result<(f64, f32, f64, f64), Box<dyn std::error::Error>>
where
    F: FnMut() -> Result<(), hipfire_xdna::XdnaError>,
{
    use std::time::Instant;

    use hipfire_primitives::conv::{bf16_bits_to_f32, f32_to_bf16_bits};
    use hipfire_xdna::{NpuResidentFfnDenseW8, NpuResidentFfnDenseW8IoMode, OpusPackedMatrix};

    const M: usize = 256;
    const K: usize = 768;
    const INTERMEDIATE: usize = 1152;
    const OUTPUT: usize = 768;

    let gate = OpusPackedMatrix::from_payload(
        35,
        K,
        INTERMEDIATE,
        &ffn_w8_payload(K, INTERMEDIATE, 3, 0.0060),
        None,
    )?;
    let up = OpusPackedMatrix::from_payload(
        35,
        K,
        INTERMEDIATE,
        &ffn_w8_payload(K, INTERMEDIATE, 11, 0.0055),
        None,
    )?;
    let down = OpusPackedMatrix::from_payload(
        35,
        INTERMEDIATE,
        OUTPUT,
        &ffn_w8_payload(INTERMEDIATE, OUTPUT, 23, 0.0040),
        None,
    )?;
    let input = input_bf16
        .iter()
        .copied()
        .map(bf16_bits_to_f32)
        .collect::<Vec<_>>();
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
    let reference = down
        .reference_f32(M, &intermediate)?
        .into_iter()
        .map(|value| bf16_bits_to_f32(f32_to_bf16_bits(value)))
        .collect::<Vec<_>>();

    let mut executor = NpuResidentFfnDenseW8::load_cached(cache)?;
    if executor.io_mode() != NpuResidentFfnDenseW8IoMode::CanonicalBf16 {
        return Err("R35 cache did not select the canonical-BF16 ABI".into());
    }
    executor.attach_shared_input(input_fd, input_bytes)?;
    let weights = executor.upload_weights(&gate, &up, &down)?;
    executor.run_shared(&weights)?;
    let output = executor.read_canonical_output_f32()?;
    let measured = metrics(&output, &reference);
    let max_reference = reference
        .iter()
        .fold(0.0f32, |maximum, value| maximum.max(value.abs()));
    let max_allowed = 0.02 + 0.03 * max_reference;
    if !measured.0.is_finite() || measured.0 < 0.999 || measured.1 > max_allowed {
        return Err(format!(
            "R34 to R35 canonical handoff failed: cosine={:.8} max_abs={:.7} allowed={max_allowed:.7}",
            measured.0, measured.1
        )
        .into());
    }

    const TIMED_RUNS: usize = 3;
    let started = Instant::now();
    for _ in 0..TIMED_RUNS {
        executor.run_shared(&weights)?;
    }
    let dispatch_ms = started.elapsed().as_secs_f64() * 1e3 / TIMED_RUNS as f64;
    let started = Instant::now();
    for _ in 0..TIMED_RUNS {
        produce_input()?;
        executor.run_shared(&weights)?;
    }
    let chain_ms = started.elapsed().as_secs_f64() * 1e3 / TIMED_RUNS as f64;
    Ok((measured.0, measured.1, dispatch_ms, chain_ms))
}

#[cfg(target_os = "linux")]
fn ffn_w8_payload(k: usize, n: usize, seed: usize, base_scale: f32) -> Vec<u8> {
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
