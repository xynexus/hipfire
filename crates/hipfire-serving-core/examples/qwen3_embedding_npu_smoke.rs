//! Full-artifact Qwen3 embedding NPU smoke.

fn main() -> Result<(), Box<dyn std::error::Error>> {
    use hipfire_model::embedding::EmbeddingMetadata;
    use hipfire_runtime::hfq::{config_from_hfq, HfqFile};
    use hipfire_serving_core::qwen3_embedding::Qwen3EmbeddingState;

    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let artifact = args
        .first()
        .ok_or("usage: qwen3_embedding_npu_smoke ARTIFACT [OUTPUT] [TOKEN_IDS_JSON|lengths:CSV]")?;
    let hfq = HfqFile::open(std::path::Path::new(artifact))?;
    let config = config_from_hfq(&hfq).ok_or("artifact has no supported model config")?;
    let metadata = EmbeddingMetadata::from_hfq_metadata_json(&hfq.metadata_json)?
        .ok_or("artifact has no embedding metadata")?;
    let state = Qwen3EmbeddingState::load(&hfq, config, metadata)?;
    let tokenized = if let Some(input) = args.get(2) {
        if let Some(lengths) = input.strip_prefix("lengths:") {
            lengths
                .split(',')
                .map(|length| {
                    let length = length.parse::<usize>()?;
                    Ok((0..length)
                        .map(|index| 10 + (index * 7919 % (state.config.vocab_size - 10)) as u32)
                        .collect::<Vec<_>>())
                })
                .collect::<Result<Vec<_>, Box<dyn std::error::Error>>>()?
        } else {
            serde_json::from_slice::<Vec<Vec<u32>>>(&std::fs::read(input)?)?
        }
    } else {
        vec![
            (0..127).map(|index| 10 + index as u32).collect::<Vec<_>>(),
            (0..108)
                .map(|index| 500 + (index * 7) as u32)
                .collect::<Vec<_>>(),
        ]
    };
    let lengths = tokenized.iter().map(Vec::len).collect::<Vec<_>>();
    let dispatches = hipfire_serving_core::embedding_batch::plan_embedding_dispatches(
        &lengths,
        &state.metadata.sequence,
    )?;
    let padded_rows = dispatches
        .iter()
        .map(|dispatch| dispatch.padded_rows)
        .sum::<usize>();
    if std::env::var("HIPFIRE_QWEN3_WARMUP").is_ok_and(|value| value != "0") {
        state.encode_token_batches(&tokenized)?;
    }
    let started = std::time::Instant::now();
    let capture_layer_trace = std::env::var_os("HIPFIRE_QWEN3_LAYER_TRACE").is_some();
    let (embeddings, layer_trace) = if capture_layer_trace {
        let (embeddings, layers) = state.encode_token_batches_with_layer_trace(&tokenized)?;
        (embeddings, Some(layers))
    } else {
        (state.encode_token_batches(&tokenized)?, None)
    };
    let elapsed = started.elapsed();
    if embeddings.len() != tokenized.len()
        || embeddings.iter().any(|embedding| embedding.len() != 1024)
    {
        return Err("full Qwen3 NPU smoke returned the wrong embedding shape".into());
    }
    let norms = embeddings
        .iter()
        .map(|embedding| {
            embedding
                .iter()
                .map(|value| value * value)
                .sum::<f32>()
                .sqrt()
        })
        .collect::<Vec<_>>();
    if embeddings.iter().flatten().any(|value| !value.is_finite())
        || norms.iter().any(|norm| (norm - 1.0).abs() > 1e-4)
    {
        return Err(format!("invalid full Qwen3 NPU embeddings; norms={norms:?}").into());
    }
    let cross_cosine = embeddings.get(1).map(|right| {
        embeddings[0]
            .iter()
            .zip(right)
            .map(|(left, right)| left * right)
            .sum::<f32>()
    });
    let elapsed_seconds = elapsed.as_secs_f64();
    let actual_tokens = lengths.iter().sum::<usize>();
    println!(
        "qwen3-embedding-full-npu backend=xdna documents={} dimensions=1024 lengths={lengths:?} norms={norms:?} cross_cosine={cross_cosine:?} elapsed_ms={:.3} documents_per_second={:.6} tokens_per_second={:.6} padding_ratio={:.6}",
        embeddings.len(),
        elapsed.as_secs_f64() * 1000.0,
        embeddings.len() as f64 / elapsed_seconds,
        actual_tokens as f64 / elapsed_seconds,
        1.0 - actual_tokens as f64 / padded_rows as f64,
    );
    if let Some(output) = args.get(1) {
        std::fs::write(
            output,
            serde_json::to_vec_pretty(&serde_json::json!({
                "token_ids": tokenized,
                "embeddings": embeddings,
                "layer_last_token_residuals": layer_trace
                    .as_ref()
                    .map(|trace| &trace.layer_last_token_residuals),
                "last_layer_stages": layer_trace
                    .as_ref()
                    .map(|trace| &trace.last_layer_stages),
                "stage_layer": layer_trace.as_ref().map(|trace| trace.stage_layer),
                "stage_sequence_bucket": layer_trace
                    .as_ref()
                    .map(|trace| trace.sequence_bucket),
                "stage_dispatch_batch": layer_trace
                    .as_ref()
                    .map(|trace| trace.dispatch_batch),
                "stage_token_major": layer_trace
                    .as_ref()
                    .map(|trace| &trace.stage_token_major),
            }))?,
        )?;
    }
    Ok(())
}
