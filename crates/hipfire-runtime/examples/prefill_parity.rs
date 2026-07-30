//! Logit-space parity: batched prefill vs the per-token reference.
//!
//! The repo's standard for accepting a batched-prefill path is a cosine comparison of the
//! final-position logits against the per-token reference ("bisection cosine 0.9998+", see
//! `hipfire-arch-llama/src/arch.rs`), NOT token equality — a near-tie in argmax can flip a
//! generated token even when the implementation is correct, so diffing generated text
//! proves nothing either way.
//!
//! This runs the SAME prompt twice on a freshly-zeroed KV cache each time:
//!   1. `forward_prefill_batch` (batched; for Oq4G256 this is the
//!      `forward_prefill_chunk` path with the grouped-Oq4 projection arms)
//!   2. `forward_scratch` in a per-token loop (the reference)
//! then reports cosine similarity, max abs/rel deviation, and whether argmax agrees.
//!
//! Run: cargo run --release -p hipfire-runtime --example prefill_parity -- <model.hfq> [prompt...]
fn main() {
    use hipfire_arch_llama::Llama;
    use hipfire_rdna::Gpu;
    use hipfire_runtime::arch::Architecture;
    use hipfire_runtime::hfq::HfqFile;
    use hipfire_runtime::kv::KvCache;
    use hipfire_runtime::llama;
    use std::path::Path;

    let args: Vec<String> = std::env::args().collect();
    let model_path = args.get(1).unwrap_or_else(|| {
        eprintln!("usage: prefill_parity <model.hfq> [prompt...]");
        std::process::exit(2);
    });
    let prompt_text = if args.len() > 2 {
        args[2..].join(" ")
    } else {
        "the quick brown fox jumps over the lazy dog and keeps running ".repeat(45)
    };

    let mut gpu = Gpu::init().expect("GPU init failed");
    let mut hfq = HfqFile::open(Path::new(model_path)).expect("open hfq");
    let config = <Llama as Architecture>::config_from_hfq(&hfq).expect("config");
    let tokenizer = hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
        .expect("tokenizer from hfq metadata");
    let weights =
        <Llama as Architecture>::load_weights(&mut hfq, &config, &mut gpu).expect("weights");
    let scratch = <Llama as Architecture>::new_state(&mut gpu, &config).expect("scratch");

    let tokens: Vec<u32> = tokenizer.encode(&prompt_text);
    let n = tokens.len();
    let kv_seq_len = (n + 64).next_power_of_two().max(256);
    println!("model={model_path}\nprompt tokens={n} kv_seq_len={kv_seq_len}");

    let mk_kv = |gpu: &mut Gpu| {
        KvCache::new_gpu_q8(
            gpu,
            config.n_layers,
            config.n_kv_heads,
            config.head_dim,
            kv_seq_len,
        )
        .expect("kv")
    };

    // --- 1. batched -------------------------------------------------------------
    // PARITY_PATH=chunk (default) -> forward_prefill_batch / forward_prefill_chunk (W4A4).
    // PARITY_PATH=fwd            -> prefill_forward / weights::weight_gemm, which honours
    //                               HIPFIRE_OQ4_PREFILL_ACT_BITS (4 | 8 | 16) — use this to
    //                               separate activation precision from wiring.
    let path = std::env::var("PARITY_PATH").unwrap_or_else(|_| "chunk".into());
    let mut kv = mk_kv(&mut gpu);
    let batched = if path == "fwd" {
        let logits = llama::prefill_forward(&mut gpu, &weights, &config, &tokens, &mut kv)
            .expect("prefill_forward");
        logits
    } else {
        llama::forward_prefill_batch(
            &mut gpu, &weights, &config, &tokens, 0, &mut kv, &scratch, None,
        )
        .expect("forward_prefill_batch");
        gpu.download_f32(&scratch.logits)
            .expect("download batched logits")
    };
    println!("batched path        : {path}");

    // --- 2. per-token reference -------------------------------------------------
    let mut kv_ref = mk_kv(&mut gpu);
    for (pos, &tok) in tokens.iter().enumerate() {
        llama::forward_scratch_embed(&mut gpu, &weights, &config, tok, pos, &scratch)
            .expect("forward_scratch_embed");
        llama::forward_scratch_compute(&mut gpu, &weights, &config, pos, &mut kv_ref, &scratch)
            .expect("forward_scratch_compute");
    }
    let reference = gpu
        .download_f32(&scratch.logits)
        .expect("download reference logits");

    // --- compare ----------------------------------------------------------------
    let v = config.vocab_size.min(batched.len()).min(reference.len());
    let (a, b) = (&batched[..v], &reference[..v]);
    let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
    let na: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    let cos = if na > 0.0 && nb > 0.0 {
        dot / (na * nb)
    } else {
        f64::NAN
    };

    let mut max_abs = 0.0f64;
    let mut max_rel = 0.0f64;
    for (x, y) in a.iter().zip(b) {
        let d = (*x as f64 - *y as f64).abs();
        if d > max_abs {
            max_abs = d;
        }
        let scale = (*y as f64).abs().max(1e-3);
        if d / scale > max_rel {
            max_rel = d / scale;
        }
    }
    let am = |s: &[f32]| {
        s.iter()
            .enumerate()
            .max_by(|p, q| p.1.partial_cmp(q.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    };
    let (ab, ar) = (am(a), am(b));

    println!("vocab compared      : {v}");
    println!("cosine              : {cos:.8}");
    println!("max abs deviation   : {max_abs:.6}");
    println!("max rel deviation   : {max_rel:.6}");
    println!(
        "argmax batched/ref  : {ab} / {ar} ({})",
        if ab == ar { "agree" } else { "DIFFER" }
    );
    // The repo's acceptance bar for a batched-prefill path.
    let pass = cos >= 0.9998;
    println!(
        "\n{} (bar: cosine >= 0.9998)",
        if pass { "PASS" } else { "FAIL" }
    );
    if !pass {
        std::process::exit(1);
    }
}
