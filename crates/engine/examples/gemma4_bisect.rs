//! Bisect batched vs single-token prefill divergence.
//! Usage:
//!   HIPFIRE_DUMP_LAYERS=1 cargo run --release --features deltanet \
//!     --example gemma4_bisect -- ~/.hipfire/models/gemma-4-31b/gemma-4-31b.mg4.hfq
#[cfg(not(feature = "deltanet"))]
fn main() { eprintln!("build with --features deltanet"); }

#[cfg(feature = "deltanet")]
fn main() {
    use engine::hfq::HfqFile;
    use engine::gemma4::{self, Gemma4Scratch};
    use engine::llama::{self, KvCache};
    use std::path::Path;

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: gemma4_bisect <model.hfq>");
        std::process::exit(1);
    }
    let model_path = &args[1];

    eprintln!("Opening: {model_path}");
    let hfq = HfqFile::open(Path::new(model_path)).expect("open model");
    let config = gemma4::config_from_hfq(&hfq).expect("read config");
    let plan = config.kv_share_plan();

    eprintln!("Loading weights...");
    let mut gpu = rdna_compute::Gpu::init().expect("gpu init");
    let weights = gemma4::load_weights(&hfq, &config, &mut gpu).expect("load weights");

    let kv_seq = 256usize;
    let tokenizer = engine::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
        .expect("tokenizer");

    // Short prompt for fast bisect.
    let raw = "What is the capital of France?";
    let mut tokens: Vec<u32> = vec![config.bos_token];
    tokens.extend(tokenizer.encode(raw));
    let n_tok = tokens.len();
    eprintln!("Prompt: {n_tok} tokens: {:?}", &tokens[..n_tok.min(12)]);

    // === Single-token (ground truth) ===
    eprintln!("\n=== SINGLE-TOKEN path (ground truth) ===");
    let mut kv_s1 = KvCache::new_gpu_asym3(&mut gpu, plan.n_sliding_own, config.sliding_n_kv_heads, config.sliding_head_dim, kv_seq).expect("kv sliding");
    let mut kv_f1 = KvCache::new_gpu_asym3(&mut gpu, plan.n_full_own, config.full_n_kv_heads, config.full_head_dim, kv_seq).expect("kv full");
    let sc1 = Gemma4Scratch::new(&mut gpu, &config, kv_seq).expect("scratch");
    gemma4::init_scratch_constants(&mut gpu, &sc1, config.full_head_dim).expect("init constants");

    for (pos, &tok) in tokens.iter().enumerate() {
        gemma4::forward_scratch(&mut gpu, &weights, &config, tok, pos, &mut kv_s1, &mut kv_f1, &sc1).expect("forward_scratch");
    }
    let logits_single = gpu.download_f32(&sc1.logits).expect("download logits single");
    let argmax_single = llama::argmax(&logits_single);
    eprintln!("Single-token argmax: {argmax_single} = '{}'", tokenizer.decode(&[argmax_single]));

    // === Batched path ===
    eprintln!("\n=== BATCHED path ===");
    let mut kv_s2 = KvCache::new_gpu_asym3(&mut gpu, plan.n_sliding_own, config.sliding_n_kv_heads, config.sliding_head_dim, kv_seq).expect("kv sliding 2");
    let mut kv_f2 = KvCache::new_gpu_asym3(&mut gpu, plan.n_full_own, config.full_n_kv_heads, config.full_head_dim, kv_seq).expect("kv full 2");
    let sc2 = Gemma4Scratch::new(&mut gpu, &config, kv_seq).expect("scratch 2");
    gemma4::init_scratch_constants(&mut gpu, &sc2, config.full_head_dim).expect("init constants 2");

    std::env::set_var("HIPFIRE_GEMMA4_BATCHED_KEQV", "1");
    gemma4::forward_prefill_batch(&mut gpu, &weights, &config, &tokens, 0, &mut kv_s2, &mut kv_f2, &sc2).expect("forward_prefill_batch");
    let logits_batch = gpu.download_f32(&sc2.logits).expect("download logits batch");
    let argmax_batch = llama::argmax(&logits_batch);
    eprintln!("Batched argmax: {argmax_batch} = '{}'", tokenizer.decode(&[argmax_batch]));

    // === Compare logits ===
    let n = logits_single.len().min(logits_batch.len());
    let max_diff = (0..n).map(|i| (logits_single[i] - logits_batch[i]).abs()).fold(0.0f32, f32::max);
    let mean_diff = (0..n).map(|i| (logits_single[i] - logits_batch[i]).abs()).sum::<f32>() / n as f32;
    eprintln!("\nLogit comparison: max_abs_diff={max_diff:.6}  mean_abs_diff={mean_diff:.6}");
    if argmax_single == argmax_batch {
        eprintln!("Argmax MATCHES — paths agree on next token");
    } else {
        eprintln!("Argmax MISMATCH — single={argmax_single} batch={argmax_batch}");
    }
}
