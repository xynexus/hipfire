//! Hybrid GPU/XDNA parity and latency probe for generic Opus artifacts.
//!
//! `hipfire lock acquire embeddinggemma-npu && cargo run --release -p \
//! hipfire-arch-embeddinggemma --example embed_e2e_npu_opus -- MODEL.hfq \
//! [CACHE_ROOT] [ITERS]; hipfire lock release`

#[cfg(target_os = "linux")]
fn package_power_path() -> Option<std::path::PathBuf> {
    std::fs::read_dir("/sys/class/hwmon")
        .ok()?
        .flatten()
        .map(|entry| entry.path())
        .find(|path| {
            std::fs::read_to_string(path.join("name")).is_ok_and(|name| name.trim() == "amdgpu")
                && path.join("power1_average").is_file()
        })
        .map(|path| path.join("power1_average"))
}

#[cfg(target_os = "linux")]
fn package_watts(path: Option<&std::path::Path>) -> Option<f64> {
    let microwatts: f64 = std::fs::read_to_string(path?).ok()?.trim().parse().ok()?;
    Some(microwatts / 1e6)
}

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
    let compare_resident_ffn =
        std::env::var("HIPFIRE_EMBED_COMPARE_RESIDENT_FFN").is_ok_and(|value| value != "0");

    let mut hfq = HfqFile::open(Path::new(&args[0]))?;
    let config = embeddinggemma::config_from_metadata_json(&hfq.metadata_json)
        .ok_or("embeddinggemma config missing")?;
    let tokenizer = Tokenizer::from_hfq_metadata(&hfq.metadata_json)?;
    let reference_model = std::env::var("HIPFIRE_EMBED_REFERENCE_MODEL").ok();
    let documents = [
        "The cat sat on the warm windowsill in the afternoon sun.",
        "A feline rested by the sunny window during the day.",
        "Quarterly revenue grew twelve percent driven by cloud services.",
    ];
    let mut token_batches: Vec<Vec<u32>> = documents
        .iter()
        .map(|document| tokenizer.encode(&format!("{}{document}", config.document_prompt)))
        .collect();
    if let Some(target_tokens) = std::env::var("HIPFIRE_EMBED_E2E_TOKENS")
        .ok()
        .map(|value| value.parse::<usize>())
        .transpose()?
    {
        if target_tokens == 0 {
            return Err("HIPFIRE_EMBED_E2E_TOKENS must be non-zero".into());
        }
        let mut document = String::new();
        let mut tokens = Vec::new();
        while tokens.len() < target_tokens {
            document.push_str(" The cat rested by the sunny window while cloud revenue grew.");
            tokens = tokenizer.encode(&format!("{}{document}", config.document_prompt));
        }
        tokens.truncate(target_tokens);
        token_batches = vec![tokens];
    }

    let mut gpu = Gpu::init()?;
    let weights = if let Some(reference_model) = &reference_model {
        let mut reference_hfq = HfqFile::open(Path::new(reference_model))?;
        let reference_config =
            embeddinggemma::config_from_metadata_json(&reference_hfq.metadata_json)
                .ok_or("embeddinggemma reference config missing")?;
        if reference_config.hidden_size != config.hidden_size
            || reference_config.num_hidden_layers != config.num_hidden_layers
            || reference_config.embedding_dim != config.embedding_dim
        {
            return Err("embeddinggemma reference model shape does not match candidate".into());
        }
        embeddinggemma::EmbeddingGemmaWeights::load(&mut reference_hfq, &config, &mut gpu)?
    } else {
        embeddinggemma::EmbeddingGemmaWeights::load(&mut hfq, &config, &mut gpu)?
    };
    let mut projector =
        embeddinggemma::NpuOpusProjector::load_cached(&hfq, &config, Path::new(&cache_root))?;
    let power_path = package_power_path();
    eprintln!(
        "loaded {} layers across {} resident NPU executor widths; complete_resident_ffn={}",
        projector.layer_count(),
        projector.executor_count(),
        projector.resident_ffn_enabled(),
    );

    let gpu_started = Instant::now();
    let mut gpu_embeddings = Vec::new();
    let mut gpu_power = Vec::new();
    for _ in 0..iterations {
        gpu_embeddings.clear();
        for tokens in &token_batches {
            gpu_embeddings.push(embeddinggemma::embed_forward(
                &mut gpu, &weights, &config, tokens,
            )?);
            gpu_power.extend(package_watts(power_path.as_deref()));
        }
    }
    let encodes = iterations * token_batches.len();
    let gpu_ms = gpu_started.elapsed().as_secs_f64() * 1e3 / encodes as f64;

    let (fallback_embeddings, fallback_ms) = if compare_resident_ffn {
        projector.select_resident_ffn(false)?;
        let started = Instant::now();
        let mut embeddings = Vec::new();
        for _ in 0..iterations {
            embeddings.clear();
            for tokens in &token_batches {
                embeddings.push(embeddinggemma::embed_forward_with_projector(
                    &mut gpu,
                    &weights,
                    &config,
                    tokens,
                    &mut projector,
                )?);
            }
        }
        let elapsed_ms = started.elapsed().as_secs_f64() * 1e3 / encodes as f64;
        projector.select_resident_ffn(true)?;
        (Some(embeddings), Some(elapsed_ms))
    } else {
        (None, None)
    };

    let npu_started = Instant::now();
    let mut npu_embeddings = Vec::new();
    let mut npu_power = Vec::new();
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
            npu_power.extend(package_watts(power_path.as_deref()));
        }
    }
    let npu_ms = npu_started.elapsed().as_secs_f64() * 1e3 / encodes as f64;

    let (mean_cosine, min_cosine, max_abs) = embedding_metrics(&gpu_embeddings, &npu_embeddings);
    if let (Some(fallback), Some(fallback_ms)) = (&fallback_embeddings, fallback_ms) {
        let (mean, min, max_abs) = embedding_metrics(fallback, &npu_embeddings);
        println!(
            "resident_ffn_vs_projection_fallback mean_cosine={mean:.8} min_cosine={min:.8} max_abs={max_abs:.8} fallback_ms={fallback_ms:.3} resident_ms={npu_ms:.3}"
        );
    }
    let token_count = iterations * token_batches.iter().map(Vec::len).sum::<usize>();
    let gpu_tok_s = token_count as f64 / (gpu_ms * encodes as f64 / 1e3);
    let hybrid_tok_s = token_count as f64 / (npu_ms * encodes as f64 / 1e3);
    let mean_power = |samples: &[f64]| {
        (!samples.is_empty()).then(|| samples.iter().sum::<f64>() / samples.len() as f64)
    };
    let gpu_w = mean_power(&gpu_power);
    let hybrid_w = mean_power(&npu_power);
    println!(
        "model={} reference_model={} docs={} tokens={:?} dims={} mean_cosine={mean_cosine:.8} min_cosine={min_cosine:.8} max_abs={max_abs:.8} gpu_ms={gpu_ms:.3} hybrid_ms={npu_ms:.3} slowdown={:.2}x gpu_tok_s={gpu_tok_s:.1} hybrid_tok_s={hybrid_tok_s:.1} gpu_pkg_w={} hybrid_pkg_w={} gpu_pkg_tok_j={} hybrid_pkg_tok_j={}",
        args[0],
        reference_model.as_deref().unwrap_or(&args[0]),
        token_batches.len(),
        token_batches.iter().map(Vec::len).collect::<Vec<_>>(),
        gpu_embeddings[0].len(),
        npu_ms / gpu_ms,
        gpu_w.map_or_else(|| "n/a".into(), |watts| format!("{watts:.2}")),
        hybrid_w.map_or_else(|| "n/a".into(), |watts| format!("{watts:.2}")),
        gpu_w.map_or_else(|| "n/a".into(), |watts| format!("{:.1}", gpu_tok_s / watts)),
        hybrid_w.map_or_else(
            || "n/a".into(),
            |watts| format!("{:.1}", hybrid_tok_s / watts)
        ),
    );
    if reference_model.is_none() && min_cosine < 0.999 {
        return Err(format!("hybrid parity minimum cosine {min_cosine:.8} is below 0.999").into());
    }
    weights.free_gpu(&mut gpu);
    Ok(())
}

#[cfg(target_os = "linux")]
fn embedding_metrics(reference: &[Vec<f32>], candidate: &[Vec<f32>]) -> (f32, f32, f32) {
    let cosines: Vec<f32> = reference
        .iter()
        .zip(candidate)
        .map(|(reference, candidate)| {
            reference
                .iter()
                .zip(candidate)
                .map(|(left, right)| left * right)
                .sum()
        })
        .collect();
    let mean = cosines.iter().sum::<f32>() / cosines.len() as f32;
    let minimum = cosines.iter().copied().fold(f32::INFINITY, f32::min);
    let max_abs = reference
        .iter()
        .zip(candidate)
        .flat_map(|(reference, candidate)| reference.iter().zip(candidate))
        .map(|(left, right)| (left - right).abs())
        .fold(0.0f32, f32::max);
    (mean, minimum, max_abs)
}

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("EmbeddingGemma XDNA Opus execution is Linux-only");
}
