//! Hybrid GPU/XDNA parity and latency probe for generic Opus artifacts.
//!
//! `hipfire lock acquire embeddinggemma-npu && cargo run --release -p \
//! hipfire-arch-embeddinggemma --example embed_e2e_npu_opus -- MODEL.hfq \
//! [CACHE_ROOT] [ITERS]; hipfire lock release`

#[cfg(target_os = "linux")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::path::Path;
    use std::time::Instant;

    use hipfire_arch_embeddinggemma as embeddinggemma;
    use hipfire_rdna::Gpu;
    use hipfire_runtime::hfq::HfqFile;
    use hipfire_runtime::tokenizer::Tokenizer;

    let args: Vec<String> = std::env::args().skip(1).collect();
    if !(1..=3).contains(&args.len()) {
        return Err("usage: embed_e2e_npu_opus MODEL.hfq [CACHE_ROOT] [ITERS]".into());
    }
    let cache_root = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| format!("{}/.hipfire/npu", std::env::var("HOME").unwrap()));
    let iterations = args
        .get(2)
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(1);

    let mut hfq = HfqFile::open(Path::new(&args[0]))?;
    let config = embeddinggemma::config_from_metadata_json(&hfq.metadata_json)
        .ok_or("embeddinggemma config missing")?;
    let tokenizer = Tokenizer::from_hfq_metadata(&hfq.metadata_json)?;
    let documents = [
        "The cat sat on the warm windowsill in the afternoon sun.",
        "A feline rested by the sunny window during the day.",
        "Quarterly revenue grew twelve percent driven by cloud services.",
    ];
    let token_batches: Vec<Vec<u32>> = documents
        .iter()
        .map(|document| tokenizer.encode(&format!("{}{document}", config.document_prompt)))
        .collect();

    let mut gpu = Gpu::init()?;
    let weights = embeddinggemma::EmbeddingGemmaWeights::load(&mut hfq, &config, &mut gpu)?;
    let mut projector =
        embeddinggemma::NpuOpusProjector::load_cached(&hfq, &config, Path::new(&cache_root))?;
    eprintln!(
        "loaded {} layers across {} resident NPU executor widths",
        projector.layer_count(),
        projector.executor_count()
    );

    let gpu_started = Instant::now();
    let mut gpu_embeddings = Vec::new();
    for _ in 0..iterations {
        gpu_embeddings.clear();
        for tokens in &token_batches {
            gpu_embeddings.push(embeddinggemma::embed_forward(
                &mut gpu, &weights, &config, tokens,
            )?);
        }
    }
    let encodes = iterations * token_batches.len();
    let gpu_ms = gpu_started.elapsed().as_secs_f64() * 1e3 / encodes as f64;

    let npu_started = Instant::now();
    let mut npu_embeddings = Vec::new();
    for _ in 0..iterations {
        npu_embeddings.clear();
        for tokens in &token_batches {
            npu_embeddings.push(embeddinggemma::embed_forward_with_projector(
                &mut gpu,
                &weights,
                &config,
                tokens,
                &mut projector,
            )?);
        }
    }
    let npu_ms = npu_started.elapsed().as_secs_f64() * 1e3 / encodes as f64;

    let cosines: Vec<f32> = gpu_embeddings
        .iter()
        .zip(&npu_embeddings)
        .map(|(gpu_embedding, npu_embedding)| {
            gpu_embedding
                .iter()
                .zip(npu_embedding)
                .map(|(left, right)| left * right)
                .sum()
        })
        .collect();
    let mean_cosine = cosines.iter().sum::<f32>() / cosines.len() as f32;
    let min_cosine = cosines.iter().copied().fold(f32::INFINITY, f32::min);
    let max_abs = gpu_embeddings
        .iter()
        .zip(&npu_embeddings)
        .flat_map(|(gpu_embedding, npu_embedding)| gpu_embedding.iter().zip(npu_embedding))
        .map(|(left, right)| (left - right).abs())
        .fold(0.0f32, f32::max);
    println!(
        "model={} docs={} tokens={:?} dims={} mean_cosine={mean_cosine:.8} min_cosine={min_cosine:.8} max_abs={max_abs:.8} gpu_ms={gpu_ms:.3} hybrid_ms={npu_ms:.3} slowdown={:.2}x",
        args[0],
        token_batches.len(),
        token_batches.iter().map(Vec::len).collect::<Vec<_>>(),
        gpu_embeddings[0].len(),
        npu_ms / gpu_ms
    );
    if min_cosine < 0.999 {
        return Err(format!("hybrid parity minimum cosine {min_cosine:.8} is below 0.999").into());
    }
    weights.free_gpu(&mut gpu);
    Ok(())
}

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("EmbeddingGemma XDNA Opus execution is Linux-only");
}
