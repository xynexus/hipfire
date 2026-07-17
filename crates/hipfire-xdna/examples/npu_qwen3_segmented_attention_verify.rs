//! Live AIE2P parity check for real-length-masked Qwen3 causal GQA.

#[cfg(target_os = "linux")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use hipfire_primitives::conv::bf16_bits_to_f32;
    use hipfire_xdna::{NpuSegmentedAttention, SegmentedAttentionGeometry};

    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if !(3..=5).contains(&args.len()) {
        return Err(
            "usage: npu_qwen3_segmented_attention_verify CACHE BUCKET BATCH [ITERS] [QUERY_HEADS]"
                .into(),
        );
    }
    let bucket = args[1].parse::<usize>()?;
    let batch = args[2].parse::<usize>()?;
    let iterations = args
        .get(3)
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(2);
    if iterations == 0 {
        return Err("segmented-attention verifier needs at least one iteration".into());
    }
    let query_heads = args
        .get(4)
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(16);
    let geometry = SegmentedAttentionGeometry {
        sequence_bucket: bucket,
        dispatch_batch: batch,
        query_heads,
        kv_heads: 8,
        head_dim: 128,
    }
    .validate()?;
    validate_manifest(&args[0], geometry)?;
    let uniform_oracle = std::env::var_os("HIPFIRE_SEGMENTED_ATTENTION_UNIFORM_ORACLE").is_some();

    let q_elements = batch * geometry.query_heads * bucket * geometry.head_dim;
    let kv_elements = batch * geometry.kv_heads * bucket * geometry.head_dim;
    let queries = if uniform_oracle {
        vec![0; q_elements]
    } else {
        bf16_values(q_elements, |index| {
            ((index as f32 * 0.001_731).sin() * 0.21) + ((index / 128) % 17) as f32 * 0.0017
        })
    };
    let keys = if uniform_oracle {
        vec![0; kv_elements]
    } else {
        bf16_values(kv_elements, |index| {
            ((index as f32 * 0.002_117).cos() * 0.18) - ((index / 128) % 13) as f32 * 0.0013
        })
    };
    let values = bf16_values(kv_elements, |index| {
        ((index as f32 * 0.001_337).sin() * 0.29) + (index % 11) as f32 * 0.0021
    });
    let lengths = match std::env::var("HIPFIRE_SEGMENTED_ATTENTION_LENGTHS") {
        Ok(value) => {
            let lengths = value
                .split(',')
                .map(str::parse::<u32>)
                .collect::<Result<Vec<_>, _>>()?;
            if lengths.len() != batch
                || lengths
                    .iter()
                    .any(|&length| length == 0 || length as usize > bucket)
            {
                return Err(format!(
                    "HIPFIRE_SEGMENTED_ATTENTION_LENGTHS must contain {batch} values in 1..={bucket}"
                )
                .into());
            }
            lengths
        }
        Err(std::env::VarError::NotPresent) => (0..batch)
            .map(|document| bucket.saturating_sub(1 + document * 17).max(1) as u32)
            .collect::<Vec<_>>(),
        Err(error) => return Err(error.into()),
    };
    let reference = if uniform_oracle {
        uniform_causal_attention_reference(&values, geometry, &lengths)
    } else {
        attention_reference(&queries, &keys, &values, geometry, &lengths)
    };

    let xclbin = std::fs::read(format!("{}/final.xclbin", args[0]))?;
    let instructions = std::fs::read(format!("{}/insts.bin", args[0]))?;
    let mut attention = NpuSegmentedAttention::load(&xclbin, &instructions, geometry)?;

    let output_bits = attention.run(&queries, &keys, &values, &lengths)?;
    let independently_loaded = NpuSegmentedAttention::load(&xclbin, &instructions, geometry)?
        .run(&queries, &keys, &values, &lengths)?;
    if independently_loaded != output_bits {
        let first_difference = output_bits
            .iter()
            .zip(&independently_loaded)
            .position(|(first, second)| first != second)
            .unwrap();
        let elements_per_head = bucket * geometry.head_dim;
        let token = first_difference % elements_per_head / geometry.head_dim;
        return Err(format!(
            "segmented-attention output changed across independent image loads at element {first_difference} (token={token}, visible={}): 0x{:04x} != 0x{:04x}",
            token < lengths[first_difference / (geometry.query_heads * elements_per_head)] as usize,
            output_bits[first_difference],
            independently_loaded[first_difference]
        )
        .into());
    }
    let output = output_bits
        .iter()
        .copied()
        .map(bf16_bits_to_f32)
        .collect::<Vec<_>>();
    let (cosine, max_abs, mean_abs) = metrics(&output, &reference);
    if !cosine.is_finite() || cosine < 0.997 || max_abs > 0.05 {
        return Err(format!(
            "Qwen3 segmented attention parity failed: cosine={cosine:.8} max_abs={max_abs:.7} mean_abs={mean_abs:.8}"
        )
        .into());
    }
    verify_padded_rows_are_zero(&output_bits, geometry, &lengths)?;
    if batch > 1 {
        verify_document_isolation(
            &mut attention,
            &queries,
            &keys,
            &values,
            &lengths,
            &output_bits,
            geometry,
        )?;
    }

    let started = std::time::Instant::now();
    for _ in 0..iterations {
        let sustained = attention.run(&queries, &keys, &values, &lengths)?;
        if sustained != output_bits {
            return Err("segmented-attention output changed across identical dispatches".into());
        }
    }
    let e2e_ms = started.elapsed().as_secs_f64() * 1e3 / iterations as f64;
    println!(
        "qwen3-segmented-attention-bf16 S={bucket} B={batch} Hq={} Hkv={} D={} oracle={}: cosine={cosine:.8} max_abs={max_abs:.7} mean_abs={mean_abs:.8} e2e_ms={e2e_ms:.4} lengths={lengths:?}",
        geometry.query_heads,
        geometry.kv_heads,
        geometry.head_dim,
        if uniform_oracle { "uniform" } else { "dense" }
    );
    Ok(())
}

#[cfg(target_os = "linux")]
fn validate_manifest(
    cache: &str,
    geometry: hipfire_xdna::SegmentedAttentionGeometry,
) -> Result<(), Box<dyn std::error::Error>> {
    let manifest = std::fs::read_to_string(format!("{cache}/manifest.json"))?;
    for field in [
        "\"schema\": \"hipfire.npu_segmented_attention_image.v1\"".to_string(),
        "\"npu_architecture\": \"aie2p\"".to_string(),
        "\"architecture\": \"qwen3\"".to_string(),
        "\"attention\": \"causal\"".to_string(),
        format!("\"sequence_bucket\": {}", geometry.sequence_bucket),
        format!("\"dispatch_batch\": {}", geometry.dispatch_batch),
        format!("\"query_heads\": {}", geometry.query_heads),
        format!("\"kv_heads\": {}", geometry.kv_heads),
        format!("\"head_dim\": {}", geometry.head_dim),
    ] {
        if !manifest.contains(&field) {
            return Err(format!("segmented-attention manifest missing {field}").into());
        }
    }
    Ok(())
}

#[cfg(target_os = "linux")]
fn bf16_values(length: usize, mut value: impl FnMut(usize) -> f32) -> Vec<u16> {
    (0..length)
        .map(|index| hipfire_primitives::conv::f32_to_bf16_bits(value(index)))
        .collect()
}

#[cfg(target_os = "linux")]
fn attention_reference(
    queries: &[u16],
    keys: &[u16],
    values: &[u16],
    geometry: hipfire_xdna::SegmentedAttentionGeometry,
    lengths: &[u32],
) -> Vec<f32> {
    use hipfire_primitives::conv::bf16_bits_to_f32;

    let mut output = vec![0.0f32; queries.len()];
    let mut scores = vec![0.0f32; geometry.sequence_bucket];
    let scale = 1.0 / (geometry.head_dim as f32).sqrt();
    for (document, &real_length) in lengths.iter().enumerate() {
        let real_length = real_length as usize;
        for query_head in 0..geometry.query_heads {
            let kv_head = query_head / (geometry.query_heads / geometry.kv_heads);
            for query_token in 0..real_length {
                let q_base = (((document * geometry.query_heads + query_head)
                    * geometry.sequence_bucket
                    + query_token)
                    * geometry.head_dim) as usize;
                for key_token in 0..=query_token {
                    let k_base = ((document * geometry.kv_heads + kv_head)
                        * geometry.sequence_bucket
                        + key_token)
                        * geometry.head_dim;
                    scores[key_token] = (0..geometry.head_dim)
                        .map(|dim| {
                            bf16_bits_to_f32(queries[q_base + dim])
                                * bf16_bits_to_f32(keys[k_base + dim])
                        })
                        .sum::<f32>()
                        * scale;
                }
                let active = &mut scores[..=query_token];
                let max = active.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let sum = active
                    .iter_mut()
                    .map(|score| {
                        *score = (*score - max).exp();
                        *score
                    })
                    .sum::<f32>();
                let out_base = q_base;
                for key_token in 0..=query_token {
                    let probability = active[key_token] / sum;
                    let v_base = ((document * geometry.kv_heads + kv_head)
                        * geometry.sequence_bucket
                        + key_token)
                        * geometry.head_dim;
                    for dim in 0..geometry.head_dim {
                        output[out_base + dim] +=
                            probability * bf16_bits_to_f32(values[v_base + dim]);
                    }
                }
            }
        }
    }
    output
}

#[cfg(target_os = "linux")]
fn uniform_causal_attention_reference(
    values: &[u16],
    geometry: hipfire_xdna::SegmentedAttentionGeometry,
    lengths: &[u32],
) -> Vec<f32> {
    use hipfire_primitives::conv::bf16_bits_to_f32;

    let output_elements = geometry.dispatch_batch
        * geometry.query_heads
        * geometry.sequence_bucket
        * geometry.head_dim;
    let mut output = vec![0.0f32; output_elements];
    let q_heads_per_kv = geometry.query_heads / geometry.kv_heads;
    for (document, &real_length) in lengths.iter().enumerate() {
        for kv_head in 0..geometry.kv_heads {
            let mut prefix = vec![0.0f32; geometry.head_dim];
            for token in 0..real_length as usize {
                let value_base =
                    ((document * geometry.kv_heads + kv_head) * geometry.sequence_bucket + token)
                        * geometry.head_dim;
                for dim in 0..geometry.head_dim {
                    prefix[dim] += bf16_bits_to_f32(values[value_base + dim]);
                }
                let inverse = 1.0 / (token + 1) as f32;
                for q_local in 0..q_heads_per_kv {
                    let query_head = kv_head * q_heads_per_kv + q_local;
                    let output_base = ((document * geometry.query_heads + query_head)
                        * geometry.sequence_bucket
                        + token)
                        * geometry.head_dim;
                    for dim in 0..geometry.head_dim {
                        output[output_base + dim] = prefix[dim] * inverse;
                    }
                }
            }
        }
    }
    output
}

#[cfg(target_os = "linux")]
fn verify_padded_rows_are_zero(
    output: &[u16],
    geometry: hipfire_xdna::SegmentedAttentionGeometry,
    lengths: &[u32],
) -> Result<(), Box<dyn std::error::Error>> {
    for (document, &length) in lengths.iter().enumerate() {
        for head in 0..geometry.query_heads {
            let start = ((document * geometry.query_heads + head) * geometry.sequence_bucket
                + length as usize)
                * geometry.head_dim;
            let end = ((document * geometry.query_heads + head + 1) * geometry.sequence_bucket)
                * geometry.head_dim;
            if output[start..end].iter().any(|&value| value != 0) {
                return Err(format!(
                    "document {document} head {head} produced nonzero padded query rows"
                )
                .into());
            }
        }
    }
    Ok(())
}

#[cfg(target_os = "linux")]
fn verify_document_isolation(
    attention: &mut hipfire_xdna::NpuSegmentedAttention,
    queries: &[u16],
    keys: &[u16],
    values: &[u16],
    lengths: &[u32],
    baseline: &[u16],
    geometry: hipfire_xdna::SegmentedAttentionGeometry,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut changed_q = queries.to_vec();
    let mut changed_k = keys.to_vec();
    let mut changed_v = values.to_vec();
    let q_document = geometry.query_heads * geometry.sequence_bucket * geometry.head_dim;
    let kv_document = geometry.kv_heads * geometry.sequence_bucket * geometry.head_dim;
    for value in &mut changed_q[q_document..2 * q_document] {
        *value ^= 0x0123;
    }
    for value in &mut changed_k[kv_document..2 * kv_document] {
        *value ^= 0x0211;
    }
    for value in &mut changed_v[kv_document..2 * kv_document] {
        *value ^= 0x0107;
    }
    let changed = attention.run(&changed_q, &changed_k, &changed_v, lengths)?;
    if changed[..q_document] != baseline[..q_document] {
        return Err("changing document 1 changed document 0 output".into());
    }
    if changed[q_document..2 * q_document] == baseline[q_document..2 * q_document] {
        return Err("changing document 1 did not change document 1 output".into());
    }
    Ok(())
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
    eprintln!("Qwen3 segmented-attention verification is Linux-only");
}
