//! R71/R72 single-context W4 projection + QKV pack + attention verifier.

#[cfg(target_os = "linux")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::path::{Path, PathBuf};
    use std::time::Instant;

    use hipfire_xdna::EmbeddingGemmaAttentionLayout as Layout;

    const ACTIVATION_BYTES: usize = 737_280;
    const QKV_WEIGHT_BYTES: usize = 2_359_296;
    const OUTPUT_WEIGHT_BYTES: usize = 4 * 72 * 16_384;
    const NORM_WEIGHT_BYTES: usize = 4 * 2 * 4 * 16_384;
    const STAGE_BYTES: usize = 2_457_600;
    const OUTPUT_OFFSET: usize = STAGE_BYTES + Layout::OUTPUT_BYTES;
    const FFN_HANDOFF_BYTES: usize = 256 * 768 * 2;

    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if !(6..=13).contains(&args.len()) {
        return Err("usage: npu_embedding_qkv_attention_fused_verify FUSED_CACHE R70_CACHE R27_CACHE PADDED_ACTIVATIONS.bin REFERENCE_WEIGHTS.bin|WEIGHTS.rdna2.hfp STAGE_SEED.bin [ITERS] [--direct-q] [--direct-o] [--direct-o-bf16] [--direct-o-residual-norm] [--direct-o-ffn-prefix] [--direct-o-ffn-handoff] [--direct-o-ffn-peer] [--attention-drain] [--output-weight-drain] [--output-local-finish] [--fused-weights=PATH] [--ffn-cache=PATH]".into());
    }
    let ffn_handoff = args.iter().any(|arg| arg == "--direct-o-ffn-handoff");
    let ffn_peer = args.iter().any(|arg| arg == "--direct-o-ffn-peer");
    let ffn_prefix =
        ffn_handoff || ffn_peer || args.iter().any(|arg| arg == "--direct-o-ffn-prefix");
    let residual_norm = ffn_prefix || args.iter().any(|arg| arg == "--direct-o-residual-norm");
    let direct_o_bf16 = residual_norm || args.iter().any(|arg| arg == "--direct-o-bf16");
    let direct_o = direct_o_bf16 || args.iter().any(|arg| arg == "--direct-o");
    let output_weight_drain = args.iter().any(|arg| arg == "--output-weight-drain");
    let output_local_finish = args.iter().any(|arg| arg == "--output-local-finish");
    let attention_drain = output_weight_drain
        || output_local_finish
        || args.iter().any(|arg| arg == "--attention-drain");
    if direct_o && attention_drain {
        return Err("--direct-o and attention-drain modes are mutually exclusive".into());
    }
    let direct_q = direct_o || args.iter().any(|arg| arg == "--direct-q");
    let fused_weight_path = args
        .iter()
        .find_map(|arg| arg.strip_prefix("--fused-weights="));
    let ffn_cache = args.iter().find_map(|arg| arg.strip_prefix("--ffn-cache="));
    if (ffn_handoff || ffn_peer) && ffn_cache.is_none() {
        return Err("zero-copy FFN modes require --ffn-cache=PATH".into());
    }
    let iterations = args
        .get(6)
        .filter(|value| !value.starts_with("--"))
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(100);
    if iterations == 0 {
        return Err("R71 verifier needs at least one iteration".into());
    }
    let read = |path: &str, expected: usize| -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let bytes = std::fs::read(Path::new(path))?;
        if bytes.len() != expected {
            return Err(format!("{path} has {} bytes, expected {expected}", bytes.len()).into());
        }
        Ok(bytes)
    };
    let activations = read(&args[3], ACTIVATION_BYTES)?;
    let hfp_path = args[4]
        .ends_with(".rdna2.hfp")
        .then(|| PathBuf::from(&args[4]));
    let weights = if hfp_path.is_some() {
        let raw = std::fs::read(Path::new(&args[4]))?;
        if raw.len() != 192 + QKV_WEIGHT_BYTES || &raw[..8] != b"HFOPHFP2" {
            return Err(format!("{} is not an R76-sized Opus HFP", args[4]).into());
        }
        raw[192..].to_vec()
    } else {
        read(&args[4], QKV_WEIGHT_BYTES)?
    };
    let fused_weights = if let Some(path) = fused_weight_path {
        let raw = std::fs::read(Path::new(path))?;
        if path.ends_with(".rdna2.hfp") {
            if raw.len() != 192 + QKV_WEIGHT_BYTES || &raw[..8] != b"HFOPHFP2" {
                return Err(format!("{path} is not an R82-sized Opus HFP").into());
            }
            raw[192..].to_vec()
        } else if raw.len() == QKV_WEIGHT_BYTES {
            raw
        } else {
            return Err(format!("{path} has invalid fused weight size").into());
        }
    } else {
        weights.clone()
    };
    let mut fused_weights = fused_weights;
    let output_weights = output_projection_weights();
    if direct_o || output_weight_drain || output_local_finish {
        if hfp_path.is_some() {
            return Err(
                "direct O-weight modes currently require the raw R70 reference-weight path".into(),
            );
        }
        fused_weights.extend_from_slice(&pack_output_projection_direct(&output_weights));
        if residual_norm {
            fused_weights.extend_from_slice(&pack_residual_norm_parameters());
        }
        debug_assert_eq!(
            fused_weights.len(),
            QKV_WEIGHT_BYTES
                + OUTPUT_WEIGHT_BYTES
                + if residual_norm { NORM_WEIGHT_BYTES } else { 0 }
        );
    }
    let stage_seed = read(&args[5], STAGE_BYTES)?;

    let (reference_stage, reference_q, reference_kv) = {
        let kernel = load_kernel(&args[1])?;
        let mut activation = kernel.alloc_arg(ACTIVATION_BYTES)?;
        let mut weight = kernel.alloc_arg(QKV_WEIGHT_BYTES)?;
        let mut stage = kernel.alloc_arg(STAGE_BYTES)?;
        let mut q = kernel.alloc_arg(Layout::Q_BYTES)?;
        let mut kv = kernel.alloc_arg(Layout::KV_BYTES)?;
        activation.as_mut_slice().copy_from_slice(&activations);
        weight.as_mut_slice().copy_from_slice(&weights);
        stage.as_mut_slice().copy_from_slice(&stage_seed);
        kernel.dispatch_synced(
            &[&activation, &weight, &stage, &q, &kv],
            &[true, true, true, false, false],
        )?;
        q.as_mut_slice().fill(0);
        kv.as_mut_slice().fill(0);
        kernel.sync_to_device(&q)?;
        kernel.sync_to_device(&kv)?;
        kernel.dispatch_synced(
            &[&activation, &weight, &stage, &q, &kv],
            &[false, false, false, false, false],
        )?;
        kernel.sync_output(&stage)?;
        kernel.sync_output(&q)?;
        kernel.sync_output(&kv)?;
        (
            stage.as_slice().to_vec(),
            q.as_slice().to_vec(),
            kv.as_slice().to_vec(),
        )
    };

    let reference_attention = {
        let kernel = load_kernel(&args[2])?;
        let mut q = kernel.alloc_arg(Layout::Q_BYTES)?;
        let mut kv = kernel.alloc_arg(Layout::KV_BYTES)?;
        let mut output = kernel.alloc_arg(Layout::OUTPUT_BYTES)?;
        q.as_mut_slice().copy_from_slice(&reference_q);
        kv.as_mut_slice().copy_from_slice(&reference_kv);
        kernel.dispatch_synced(&[&q, &kv, &output], &[true, true, false])?;
        output.as_mut_slice().fill(0);
        kernel.sync_to_device(&output)?;
        kernel.dispatch_synced(&[&q, &kv, &output], &[false, false, false])?;
        kernel.sync_output(&output)?;
        output.as_slice().to_vec()
    };
    let reference_output = if direct_o {
        let attention_bits = Layout::unpack_output_bf16(&reference_attention)
            .ok_or("invalid R27 physical attention output")?;
        let attention = attention_bits
            .into_iter()
            .map(hipfire_primitives::conv::bf16_bits_to_f32)
            .collect::<Vec<_>>();
        let output = output_projection_reference(&attention, &output_weights);
        Some(if residual_norm {
            residual_norm_reference(&output)
        } else {
            output
        })
    } else {
        None
    };

    if let Some(hfp_path) = hfp_path {
        use hipfire_xdna::NpuEmbeddingQkvAttentionOpus;

        if direct_q {
            return Err("resident HFP verifier requires the observable R76 Q path".into());
        }
        let mut fused = NpuEmbeddingQkvAttentionOpus::load_cached(&args[0])?;
        let resident_weights = fused.upload_weights_prepacked(&hfp_path)?;
        fused.set_input(&activations, &stage_seed)?;
        fused.run(&resident_weights)?;
        let output = fused.read_output()?;
        verify(
            "resident stage",
            &reference_stage,
            &output.result[..STAGE_BYTES],
        )?;
        verify("resident Q", &reference_q, &output.queries)?;
        verify("resident KV", &reference_kv, &output.key_values)?;
        verify(
            "resident attention",
            &reference_attention,
            &output.result[STAGE_BYTES..],
        )?;

        for _ in 0..2 {
            fused.run(&resident_weights)?;
        }
        let started = Instant::now();
        for _ in 0..iterations {
            fused.run(&resident_weights)?;
        }
        let fused_ms = started.elapsed().as_secs_f64() * 1e3 / iterations as f64;
        let output = fused.read_output()?;
        verify(
            "timed resident stage",
            &reference_stage,
            &output.result[..STAGE_BYTES],
        )?;
        verify("timed resident Q", &reference_q, &output.queries)?;
        verify("timed resident KV", &reference_kv, &output.key_values)?;
        verify(
            "timed resident attention",
            &reference_attention,
            &output.result[STAGE_BYTES..],
        )?;
        println!(
            "r76-resident-hfp stage_mismatches=0 q_mismatches=0 kv_mismatches=0 attention_mismatches=0 iterations={iterations} fused_ms={fused_ms:.4}"
        );
        return Ok(());
    }

    let fused = load_kernel(&args[0])?;
    let mut activation = fused.alloc_arg(ACTIVATION_BYTES)?;
    let mut weight = fused.alloc_arg(fused_weights.len())?;
    let result_bytes = if direct_o {
        OUTPUT_OFFSET + 256 * 768 * if direct_o_bf16 { 2 } else { 4 }
    } else {
        STAGE_BYTES + Layout::OUTPUT_BYTES
    };
    let mut shared_result = if ffn_handoff || ffn_peer {
        Some(hipfire_rdna::Gpu::init()?.alloc_shared_gtt(result_bytes)?)
    } else {
        None
    };
    let mut result = if let Some(shared) = shared_result.as_mut() {
        shared.as_mut_slice().fill(0);
        fused.import_dmabuf(shared.dmabuf_fd(), shared.len(), true)?
    } else {
        fused.alloc_arg(result_bytes)?
    };
    let mut q = fused.alloc_arg(Layout::Q_BYTES)?;
    let mut kv = fused.alloc_arg(Layout::KV_BYTES)?;
    activation.as_mut_slice().copy_from_slice(&activations);
    weight.as_mut_slice().copy_from_slice(&fused_weights);
    result.as_mut_slice().fill(0);
    let stage_base = if ffn_prefix { FFN_HANDOFF_BYTES } else { 0 };
    result.as_mut_slice()[stage_base..stage_base + STAGE_BYTES].copy_from_slice(&stage_seed);

    fused.dispatch_synced(
        &[&activation, &weight, &result, &q, &kv],
        &[true, true, true, false, false],
    )?;
    q.as_mut_slice().fill(0);
    kv.as_mut_slice().fill(0);
    fused.sync_to_device(&q)?;
    fused.sync_to_device(&kv)?;
    if !ffn_prefix {
        result.as_mut_slice()[STAGE_BYTES..].fill(0);
        fused.sync_to_device(&result)?;
    }
    fused.dispatch_synced(
        &[&activation, &weight, &result, &q, &kv],
        &[false, false, false, false, false],
    )?;
    fused.sync_output(&result)?;
    if !direct_q {
        fused.sync_output(&q)?;
    }
    fused.sync_output(&kv)?;
    if !ffn_prefix {
        verify("stage", &reference_stage, &result.as_slice()[..STAGE_BYTES])?;
    }
    if !direct_q {
        verify("Q", &reference_q, q.as_slice())?;
    }
    verify("KV", &reference_kv, kv.as_slice())?;
    if direct_o {
        if direct_o_bf16 {
            verify_output_bf16(
                "BF16 output projection",
                reference_output.as_deref().unwrap(),
                &result.as_slice()[if ffn_prefix { 0 } else { OUTPUT_OFFSET }..][..256 * 768 * 2],
                residual_norm,
            )?;
        } else {
            verify_output(
                "output projection",
                reference_output.as_deref().unwrap(),
                &result.as_slice()[OUTPUT_OFFSET..],
            )?;
        }
    } else if !attention_drain {
        verify(
            "attention",
            &reference_attention,
            &result.as_slice()[STAGE_BYTES..],
        )?;
    }

    let mut ffn_chain = if ffn_handoff || ffn_peer {
        let shared = shared_result.as_ref().expect("R91 shared result backing");
        Some(prepare_ffn_chain(
            ffn_cache.unwrap(),
            ffn_peer.then_some(&fused),
            shared.dmabuf_fd(),
            shared.len(),
            &result.as_slice()[..FFN_HANDOFF_BYTES],
        )?)
    } else {
        None
    };
    if let Some((executor, weights, reference)) = ffn_chain.as_mut() {
        executor.run_shared(weights)?;
        verify_ffn_output(&executor.read_canonical_output_f32()?, reference)?;
    }

    for _ in 0..2 {
        fused.dispatch_synced(
            &[&activation, &weight, &result, &q, &kv],
            &[false, false, false, false, false],
        )?;
    }
    let started = Instant::now();
    for _ in 0..iterations {
        fused.dispatch_synced(
            &[&activation, &weight, &result, &q, &kv],
            &[false, false, false, false, false],
        )?;
    }
    let fused_ms = started.elapsed().as_secs_f64() * 1e3 / iterations as f64;
    fused.sync_output(&result)?;
    if !direct_q {
        fused.sync_output(&q)?;
    }
    fused.sync_output(&kv)?;
    if !ffn_prefix {
        verify(
            "timed stage",
            &reference_stage,
            &result.as_slice()[..STAGE_BYTES],
        )?;
    }
    if !direct_q {
        verify("timed Q", &reference_q, q.as_slice())?;
    }
    verify("timed KV", &reference_kv, kv.as_slice())?;
    if direct_o {
        if direct_o_bf16 {
            verify_output_bf16(
                "timed BF16 output projection",
                reference_output.as_deref().unwrap(),
                &result.as_slice()[if ffn_prefix { 0 } else { OUTPUT_OFFSET }..][..256 * 768 * 2],
                residual_norm,
            )?;
        } else {
            verify_output(
                "timed output projection",
                reference_output.as_deref().unwrap(),
                &result.as_slice()[OUTPUT_OFFSET..],
            )?;
        }
    } else if !attention_drain {
        verify(
            "timed attention",
            &reference_attention,
            &result.as_slice()[STAGE_BYTES..],
        )?;
    }

    let chain_status = if let Some((executor, weights, reference)) = ffn_chain.as_mut() {
        for _ in 0..2 {
            executor.run_shared(weights)?;
        }
        let ffn_started = Instant::now();
        for _ in 0..iterations {
            executor.run_shared(weights)?;
        }
        let ffn_ms = ffn_started.elapsed().as_secs_f64() * 1e3 / iterations as f64;
        for _ in 0..2 {
            fused.dispatch_synced(
                &[&activation, &weight, &result, &q, &kv],
                &[false, false, false, false, false],
            )?;
            executor.run_shared(weights)?;
        }
        let started = Instant::now();
        for _ in 0..iterations {
            fused.dispatch_synced(
                &[&activation, &weight, &result, &q, &kv],
                &[false, false, false, false, false],
            )?;
            executor.run_shared(weights)?;
        }
        let chain_ms = started.elapsed().as_secs_f64() * 1e3 / iterations as f64;
        let output = executor.read_canonical_output_f32()?;
        let (cosine, max_abs) = verify_ffn_output(&output, reference)?;
        format!(
            " ffn_zero_copy=pass ffn_cosine={cosine:.8} ffn_max_abs={max_abs:.7} ffn_ms={ffn_ms:.4} chain_ms={chain_ms:.4}"
        )
    } else {
        String::new()
    };

    let mode = if ffn_peer {
        "r92-peer-context-residual-norm-ffn-handoff"
    } else if ffn_handoff {
        "r91-direct-residual-norm-ffn-handoff"
    } else if ffn_prefix {
        "r91-direct-residual-norm-ffn-prefix"
    } else if residual_norm {
        "r90-direct-attention-output-residual-norm"
    } else if direct_o_bf16 {
        "r89-direct-attention-output-bf16-stage"
    } else if direct_o {
        "r84-direct-attention-output"
    } else if output_local_finish {
        "r84-output-local-finish"
    } else if output_weight_drain {
        "r84-output-weight-drain"
    } else if attention_drain {
        "r84-attention-handoff-drain"
    } else if direct_q {
        "r72-direct-q"
    } else {
        "r71-fused-qkv-attention"
    };
    let q_status = if direct_q {
        "q_external=unused"
    } else {
        "q_mismatches=0"
    };
    let stage_status = if ffn_prefix {
        "stage_prefix=reused"
    } else {
        "stage_mismatches=0"
    };
    println!(
        "{mode} {stage_status} {q_status} kv_mismatches=0 {} iterations={iterations} fused_ms={fused_ms:.4}{chain_status}",
        if direct_o {
            "output_projection=pass"
        } else if output_local_finish {
            "attention_handoff=drained output_local_finish=pass"
        } else if output_weight_drain {
            "attention_handoff=drained output_weights=drained"
        } else if attention_drain {
            "attention_handoff=drained"
        } else {
            "attention_mismatches=0"
        }
    );
    Ok(())
}

#[cfg(target_os = "linux")]
fn load_kernel(cache: &str) -> Result<hipfire_xdna::NpuKernel, Box<dyn std::error::Error>> {
    Ok(hipfire_xdna::NpuKernel::load(
        &std::fs::read(format!("{cache}/final.xclbin"))?,
        &std::fs::read(format!("{cache}/insts.bin"))?,
    )?)
}

#[cfg(target_os = "linux")]
fn verify(label: &str, expected: &[u8], actual: &[u8]) -> Result<(), Box<dyn std::error::Error>> {
    let mismatches = expected
        .iter()
        .zip(actual)
        .filter(|(left, right)| left != right)
        .count();
    if mismatches == 0 {
        return Ok(());
    }
    let first = expected
        .iter()
        .zip(actual)
        .position(|(left, right)| left != right)
        .unwrap();
    Err(format!(
        "R71 {label} has {mismatches} byte mismatches; first offset {first}: expected={} actual={}",
        expected[first], actual[first]
    )
    .into())
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
fn pack_output_projection_direct(weights: &[u16]) -> Vec<u8> {
    const BLOCK: usize = 16_384;
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
fn residual_value(token: usize, hidden: usize) -> f32 {
    use hipfire_primitives::conv::{bf16_bits_to_f32, f32_to_bf16_bits};
    bf16_bits_to_f32(f32_to_bf16_bits(
        (token as f32 - 128.0) * 0.001 + (hidden % 23) as f32 * 0.002,
    ))
}

#[cfg(target_os = "linux")]
fn post_norm_value(hidden: usize) -> f32 {
    use hipfire_primitives::conv::{bf16_bits_to_f32, f32_to_bf16_bits};
    bf16_bits_to_f32(f32_to_bf16_bits(0.91 + (hidden % 29) as f32 * 0.0015))
}

#[cfg(target_os = "linux")]
fn pre_norm_value(hidden: usize) -> f32 {
    use hipfire_primitives::conv::{bf16_bits_to_f32, f32_to_bf16_bits};
    bf16_bits_to_f32(f32_to_bf16_bits(0.87 + (hidden % 31) as f32 * 0.0018))
}

#[cfg(target_os = "linux")]
fn pack_residual_norm_parameters() -> Vec<u8> {
    use hipfire_primitives::conv::f32_to_bf16_bits;
    const RECORD: usize = 16_384;
    let mut packed = vec![0u8; 4 * 2 * 4 * RECORD];
    for active_col in 0..4 {
        for wave in 0..2 {
            for core_row in 0..4 {
                let record = ((active_col * 2 + wave) * 4 + core_row) * RECORD;
                for row in 0..8 {
                    let token = wave * 128 + active_col * 32 + core_row * 8 + row;
                    for hidden in 0..768 {
                        let offset = record + (row * 768 + hidden) * 2;
                        packed[offset..offset + 2].copy_from_slice(
                            &f32_to_bf16_bits(residual_value(token, hidden)).to_le_bytes(),
                        );
                    }
                }
                for hidden in 0..768 {
                    let post_offset = record + (8 * 768 + hidden) * 2;
                    packed[post_offset..post_offset + 2]
                        .copy_from_slice(&f32_to_bf16_bits(post_norm_value(hidden)).to_le_bytes());
                    let pre_offset = record + (8 * 768 + 768 + hidden) * 2;
                    packed[pre_offset..pre_offset + 2]
                        .copy_from_slice(&f32_to_bf16_bits(pre_norm_value(hidden)).to_le_bytes());
                }
                packed[record + 15_360..record + 15_364].copy_from_slice(&1.0e-6f32.to_le_bytes());
            }
        }
    }
    packed
}

#[cfg(target_os = "linux")]
fn residual_norm_reference(output: &[f32]) -> Vec<f32> {
    use hipfire_primitives::conv::{bf16_bits_to_f32, f32_to_bf16_bits};
    let bf16 = |value: f32| bf16_bits_to_f32(f32_to_bf16_bits(value));
    let mut result = output.to_vec();
    for token in 0..256 {
        let row = &mut result[token * 768..(token + 1) * 768];
        let output_sum = row.iter().map(|value| value * value).sum::<f32>();
        let post_inverse = (output_sum / 768.0 + 1.0e-6).sqrt().recip();
        let mut residual_sum = 0.0f32;
        for (hidden, value) in row.iter_mut().enumerate() {
            let x = *value * post_norm_value(hidden) * post_inverse + residual_value(token, hidden);
            residual_sum += x * x;
            *value = bf16(x);
        }
        let pre_inverse = (residual_sum / 768.0 + 1.0e-6).sqrt().recip();
        for (hidden, value) in row.iter_mut().enumerate() {
            *value = bf16(*value * pre_norm_value(hidden) * pre_inverse);
        }
    }
    result
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
fn verify_output(
    label: &str,
    expected: &[f32],
    actual: &[u8],
) -> Result<(), Box<dyn std::error::Error>> {
    verify_output_with_limits(label, expected, actual, 0.04, 0.998, 0.998)
}

#[cfg(target_os = "linux")]
fn verify_output_with_limits(
    label: &str,
    expected: &[f32],
    actual: &[u8],
    max_abs_limit: f32,
    cosine_limit: f64,
    min_row_cosine_limit: f64,
) -> Result<(), Box<dyn std::error::Error>> {
    let got = actual
        .chunks_exact(4)
        .map(|word| f32::from_le_bytes([word[0], word[1], word[2], word[3]]))
        .collect::<Vec<_>>();
    if got.len() != expected.len() {
        return Err(format!(
            "R84 {label} has {} values, expected {}",
            got.len(),
            expected.len()
        )
        .into());
    }
    let dot = got
        .iter()
        .zip(expected)
        .map(|(&left, &right)| left as f64 * right as f64)
        .sum::<f64>();
    let got_norm = got.iter().map(|&value| (value as f64).powi(2)).sum::<f64>();
    let expected_norm = expected
        .iter()
        .map(|&value| (value as f64).powi(2))
        .sum::<f64>();
    let cosine = dot / (got_norm * expected_norm).sqrt();
    let max_abs = got
        .iter()
        .zip(expected)
        .map(|(&left, &right)| (left - right).abs())
        .fold(0.0f32, f32::max);
    let min_row_cosine = got
        .chunks_exact(768)
        .zip(expected.chunks_exact(768))
        .map(|(got_row, expected_row)| {
            let dot = got_row
                .iter()
                .zip(expected_row)
                .map(|(&left, &right)| left as f64 * right as f64)
                .sum::<f64>();
            let got_norm = got_row
                .iter()
                .map(|&value| (value as f64).powi(2))
                .sum::<f64>();
            let expected_norm = expected_row
                .iter()
                .map(|&value| (value as f64).powi(2))
                .sum::<f64>();
            dot / (got_norm * expected_norm).sqrt()
        })
        .fold(1.0f64, f64::min);
    if !cosine.is_finite()
        || cosine < cosine_limit
        || min_row_cosine < min_row_cosine_limit
        || max_abs > max_abs_limit
    {
        let worst = got
            .iter()
            .zip(expected)
            .enumerate()
            .max_by(|(_, (left_a, right_a)), (_, (left_b, right_b))| {
                (*left_a - *right_a)
                    .abs()
                    .total_cmp(&(*left_b - *right_b).abs())
            })
            .map(|(index, (&left, &right))| (index, index / 768, index % 768, left, right));
        return Err(format!(
            "R84 {label} parity failed: cosine={cosine:.8} min_row_cosine={min_row_cosine:.8} max_abs={max_abs:.8}; worst={worst:?}; nonfinite={} nonzero={} got={:?} ref={:?}",
            got.iter().filter(|value| !value.is_finite()).count(),
            got.iter().filter(|&&value| value != 0.0).count(),
            &got[..16],
            &expected[..16],
        )
        .into());
    }
    Ok(())
}

#[cfg(target_os = "linux")]
fn verify_output_bf16(
    label: &str,
    expected: &[f32],
    actual: &[u8],
    residual_norm: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let f32_bytes = actual
        .chunks_exact(2)
        .flat_map(|word| {
            hipfire_primitives::conv::bf16_bits_to_f32(u16::from_le_bytes([word[0], word[1]]))
                .to_le_bytes()
        })
        .collect::<Vec<_>>();
    if residual_norm {
        // Two AIE2P reciprocal-square-root approximations mainly introduce a
        // row-wise scale error. Keep a strict directional gate while allowing
        // the observed three-BF16-step envelope at the largest magnitudes.
        verify_output_with_limits(label, expected, &f32_bytes, 0.1, 0.9998, 0.9998)
    } else {
        verify_output_with_limits(label, expected, &f32_bytes, 0.0625, 0.998, 0.998)
    }
}

#[cfg(target_os = "linux")]
fn prepare_ffn_chain(
    cache: &str,
    peer: Option<&hipfire_xdna::NpuKernel>,
    input_fd: i32,
    input_bytes: usize,
    input_bf16: &[u8],
) -> Result<
    (
        hipfire_xdna::NpuResidentFfnDenseW8,
        hipfire_xdna::NpuResidentFfnDenseW8Weights,
        Vec<f32>,
    ),
    Box<dyn std::error::Error>,
> {
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
        .chunks_exact(2)
        .map(|word| bf16_bits_to_f32(u16::from_le_bytes([word[0], word[1]])))
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

    let mut executor = if let Some(peer) = peer {
        NpuResidentFfnDenseW8::load_cached_peer(cache, peer)
            .map_err(|error| format!("R92 load peer FFN context failed: {error}"))?
    } else {
        NpuResidentFfnDenseW8::load_cached(cache)?
    };
    if executor.io_mode() != NpuResidentFfnDenseW8IoMode::CanonicalBf16 {
        return Err("R91 FFN cache did not select the canonical-BF16 ABI".into());
    }
    executor
        .attach_shared_input(input_fd, input_bytes)
        .map_err(|error| format!("R92 attach shared FFN input failed: {error}"))?;
    let weights = executor.upload_weights(&gate, &up, &down)?;
    Ok((executor, weights, reference))
}

#[cfg(target_os = "linux")]
fn verify_ffn_output(
    got: &[f32],
    expected: &[f32],
) -> Result<(f64, f32), Box<dyn std::error::Error>> {
    let dot = got
        .iter()
        .zip(expected)
        .map(|(&left, &right)| left as f64 * right as f64)
        .sum::<f64>();
    let got_norm = got.iter().map(|&value| (value as f64).powi(2)).sum::<f64>();
    let expected_norm = expected
        .iter()
        .map(|&value| (value as f64).powi(2))
        .sum::<f64>();
    let cosine = dot / (got_norm * expected_norm).sqrt();
    let max_abs = got
        .iter()
        .zip(expected)
        .map(|(&left, &right)| (left - right).abs())
        .fold(0.0f32, f32::max);
    let max_reference = expected
        .iter()
        .fold(0.0f32, |maximum, value| maximum.max(value.abs()));
    let max_allowed = 0.02 + 0.03 * max_reference;
    if !cosine.is_finite() || cosine < 0.999 || max_abs > max_allowed {
        return Err(format!(
            "R91 zero-copy FFN parity failed: cosine={cosine:.8} max_abs={max_abs:.7} allowed={max_allowed:.7}"
        )
        .into());
    }
    Ok((cosine, max_abs))
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

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("npu_embedding_qkv_attention_fused_verify is Linux-only");
}
