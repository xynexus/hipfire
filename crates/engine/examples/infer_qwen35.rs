//! Qwen3.5 (DeltaNet) inference — matches ollama quality settings.
//! Usage: infer_qwen35 <model.hfq> [prompt text...]

use engine::hfq::{self, HfqFile};
use engine::llama;
use engine::qwen35;
use engine::qwen35::DeltaNetState;
use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

static RUNNING: AtomicBool = AtomicBool::new(true);
extern "C" fn handle_sigint(_: libc::c_int) {
    RUNNING.store(false, Ordering::SeqCst);
}

fn main() {
    unsafe {
        libc::signal(libc::SIGINT, handle_sigint as libc::sighandler_t);
    }
    let args: Vec<String> = std::env::args().collect();
    let model_path = args.get(1).unwrap_or_else(|| {
        eprintln!("Usage: infer_qwen35 <model.hfq> [prompt...]");
        std::process::exit(1);
    });

    let prompt_text = if args.len() > 2 {
        args[2..].join(" ")
    } else {
        "Hello".to_string()
    };

    eprintln!("=== hipfire Qwen3.5 inference ===");
    eprintln!("Model: {model_path}");

    let hfq = HfqFile::open(Path::new(model_path)).expect("failed to parse HFQ");
    let config = qwen35::config_from_hfq(&hfq).expect("failed to read Qwen3.5 config");
    eprintln!(
        "Config: dim={}, layers={}, heads={}, vocab={}",
        config.dim, config.n_layers, config.n_heads, config.vocab_size
    );

    let tokenizer = engine::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
        .unwrap_or_else(|| {
            let gguf = engine::gguf::GgufFile::open(Path::new(
                "/home/kaden/llama.cpp/models/Qwen3-0.6B-Q8_0.gguf",
            ))
            .expect("need GGUF for tokenizer");
            engine::tokenizer::Tokenizer::from_gguf(&gguf).expect("tokenizer failed")
        });

    // ChatML with <think> for thinking mode
    let mut prompt_tokens = tokenizer.encode(&prompt_text);
    let has_chatml = tokenizer.encode("<|im_start|>").len() == 1;
    if has_chatml {
        let im_start = tokenizer.encode("<|im_start|>");
        let im_end = tokenizer.encode("<|im_end|>");
        let user = tokenizer.encode("user");
        let asst = tokenizer.encode("assistant");
        let nl = tokenizer.encode("\n");
        let think = tokenizer.encode("<think>");

        let mut chat = Vec::new();
        // No system message (matches ollama defaults)
        // User message
        chat.extend_from_slice(&im_start);
        chat.extend_from_slice(&user);
        chat.extend_from_slice(&nl);
        chat.extend_from_slice(&prompt_tokens);
        chat.extend_from_slice(&im_end);
        chat.extend_from_slice(&nl);
        // Assistant start with <think>
        chat.extend_from_slice(&im_start);
        chat.extend_from_slice(&asst);
        chat.extend_from_slice(&nl);
        chat.extend_from_slice(&think);
        chat.extend_from_slice(&nl);
        prompt_tokens = chat;
    }
    eprintln!(
        "Prompt: \"{}\" ({} tokens)",
        prompt_text,
        prompt_tokens.len()
    );

    let mut gpu = rdna_compute::Gpu::init().expect("GPU init failed");
    eprintln!("Loading weights...");
    let weights = qwen35::load_weights(&hfq, &config, &mut gpu).expect("failed to load weights");

    let kv_seq = 2048usize;
    let kv_mode = std::env::var("HIPFIRE_KV_MODE").unwrap_or_else(|_| "q8".to_string());
    let mut kv_cache = match kv_mode.as_str() {
        "givens4" => {
            eprintln!("KV cache: givens4");
            llama::KvCache::new_gpu_asym3(
                &mut gpu,
                config.n_layers,
                config.n_kv_heads,
                config.head_dim,
                kv_seq,
            )
            .unwrap()
        }
        "givens2" => {
            eprintln!("KV cache: givens2");
            llama::KvCache::new_gpu_asym2(
                &mut gpu,
                config.n_layers,
                config.n_kv_heads,
                config.head_dim,
                kv_seq,
            )
            .unwrap()
        }
        _ => llama::KvCache::new_gpu_q8(
            &mut gpu,
            config.n_layers,
            config.n_kv_heads,
            config.head_dim,
            kv_seq,
        )
        .unwrap(),
    };
    let mut dn_state = if std::env::var("FP32_STATE").is_ok() {
        DeltaNetState::new_with_quant(&mut gpu, &config, engine::qwen35::StateQuant::FP32).unwrap()
    } else {
        DeltaNetState::new(&mut gpu, &config).unwrap()
    };

    // Phase 3a-A: use forward_scratch path (avoids per-call alloc/free + uses
    // the fused repeat_interleave kernel). Allocate scratch once, reuse forever.
    let scratch = qwen35::Qwen35Scratch::new(&mut gpu, &config, 128).unwrap();

    // ── GPU-assisted top-K sampler (opt-in, experimental) ──
    //
    // HIPFIRE_GPU_TOPK=1 enables the GPU topk_logits_f32 kernel + CPU
    // sample_top_p_from_candidates path, which avoids the 600 KB logits
    // DtoH per token. It is BIT-EXACT with the full-CPU path for
    // sampling-from-full-logits because apply_repeat_penalty can only
    // decrease logits, so the pre-penalty top-128 ⊇ post-penalty top-20.
    //
    // HIPFIRE_SAMPLE_COMPARE=1 additionally runs the old full-CPU path
    // every step and panics on token divergence — long-generation safety
    // check for the opt-in flag before enabling it by default.
    let use_gpu_topk = std::env::var("HIPFIRE_GPU_TOPK").ok().as_deref() == Some("1");
    let sample_compare = std::env::var("HIPFIRE_SAMPLE_COMPARE").ok().as_deref() == Some("1");
    if use_gpu_topk || sample_compare {
        eprintln!(
            "sampler: gpu_topk={} compare={}",
            use_gpu_topk, sample_compare
        );
    }
    const TOPK: usize = 1024; // 256 threads × top-4 each
                              // Single 8 KB device buffer laid out as [1024 × u32 indices | 1024 × f32 values]
                              // so the whole top-K candidate set downloads in one memcpy_dtoh call.
    let topk_buf = gpu
        .alloc_tensor(&[2 * TOPK], rdna_compute::DType::F32)
        .unwrap();
    let mut topk_host = vec![0u8; 2 * TOPK * 4]; // reused across steps

    // Sequential prefill
    let t1 = Instant::now();
    let mut logits = vec![0.0f32; config.vocab_size];
    for (pos, &token) in prompt_tokens.iter().enumerate() {
        qwen35::forward_scratch(
            &mut gpu,
            &weights,
            &config,
            token,
            pos,
            &mut kv_cache,
            &mut dn_state,
            &scratch,
        )
        .expect("forward failed");
        logits = gpu.download_f32(&scratch.logits).unwrap();
    }
    let prefill_ms = t1.elapsed().as_millis();
    eprintln!(
        "Prefill: {}ms ({} tokens, {:.0} tok/s)",
        prefill_ms,
        prompt_tokens.len(),
        prompt_tokens.len() as f64 / (prefill_ms as f64 / 1000.0)
    );

    // Detect special tokens
    let think_end_id = tokenizer.encode("</think>");
    let think_end_token = if think_end_id.len() == 1 {
        Some(think_end_id[0])
    } else {
        None
    };
    let im_end_id = tokenizer.encode("<|im_end|>");
    let im_end_token = if im_end_id.len() == 1 {
        Some(im_end_id[0])
    } else {
        None
    };

    let sc = llama::SamplingConfig::text_thinking();
    let max_gen = 2048;

    let t2 = Instant::now();
    let mut token_history: Vec<u32> = prompt_tokens.clone();
    let mut in_thinking = true;
    let mut generated = Vec::new();

    eprint!("<think>");
    let mut next_token = llama::sample_top_p(&logits, sc.think_temp, sc.top_p);

    for _gi in 0..max_gen {
        generated.push(next_token);
        token_history.push(next_token);

        if in_thinking && think_end_token == Some(next_token) {
            in_thinking = false;
            eprint!("</think>\n");
        } else {
            let text = tokenizer.decode(&[next_token]);
            if in_thinking {
                eprint!("{text}");
            } else {
                print!("{text}");
                std::io::stdout().flush().ok();
            }
        }

        if next_token == config.eos_token {
            break;
        }
        if im_end_token == Some(next_token) {
            break;
        }
        if !RUNNING.load(Ordering::Relaxed) {
            break;
        }

        let pos = prompt_tokens.len() + generated.len() - 1;
        qwen35::forward_scratch(
            &mut gpu,
            &weights,
            &config,
            next_token,
            pos,
            &mut kv_cache,
            &mut dn_state,
            &scratch,
        )
        .expect("forward failed");

        let temp = if in_thinking {
            sc.think_temp
        } else {
            sc.answer_temp
        };

        next_token = if use_gpu_topk || sample_compare {
            // GPU-assisted path: run topk_logits_f32 kernel, ONE DtoH,
            // run the same CPU sampler on the top-1024 subset.
            gpu.topk_logits_f32(&scratch.logits, &topk_buf, config.vocab_size)
                .expect("topk_logits");
            gpu.hip.memcpy_dtoh(&mut topk_host, &topk_buf.buf).unwrap();

            // (memcpy_dtoh already done above for timing)
            let mut cand_ids = vec![0u32; TOPK];
            let mut cand_vals = vec![0.0f32; TOPK];
            for i in 0..TOPK {
                cand_ids[i] = u32::from_ne_bytes([
                    topk_host[i * 4],
                    topk_host[i * 4 + 1],
                    topk_host[i * 4 + 2],
                    topk_host[i * 4 + 3],
                ]);
                let v_off = TOPK * 4 + i * 4;
                cand_vals[i] = f32::from_ne_bytes([
                    topk_host[v_off],
                    topk_host[v_off + 1],
                    topk_host[v_off + 2],
                    topk_host[v_off + 3],
                ]);
            }

            if sample_compare {
                // Snapshot RNG so both samplers see the same state.
                let rng_before = llama::sampler_rng_snapshot();

                // GPU-assisted path (advances RNG)
                let mut cand_vals_gpu = cand_vals.clone();
                let gpu_tok = llama::sample_top_p_from_candidates(
                    &cand_ids,
                    &mut cand_vals_gpu,
                    &token_history,
                    sc.repeat_window,
                    sc.repeat_penalty,
                    temp,
                    sc.top_p,
                );
                let rng_after_gpu = llama::sampler_rng_snapshot();

                // Restore and run full-CPU path
                llama::sampler_rng_restore(rng_before);
                logits = gpu.download_f32(&scratch.logits).unwrap();
                llama::apply_repeat_penalty(
                    &mut logits,
                    &token_history,
                    sc.repeat_window,
                    sc.repeat_penalty,
                );
                let cpu_tok = llama::sample_top_p(&logits, temp, sc.top_p);
                let rng_after_cpu = llama::sampler_rng_snapshot();

                if cpu_tok != gpu_tok || rng_after_cpu != rng_after_gpu {
                    eprintln!(
                        "\n!! SAMPLE_COMPARE divergence at step {}: cpu={} gpu={} (pos={})",
                        generated.len(),
                        cpu_tok,
                        gpu_tok,
                        pos
                    );
                    eprintln!("   rng state: before={rng_before:#x} after_gpu={rng_after_gpu:#x} after_cpu={rng_after_cpu:#x}");
                    // Show the top-128 candidate set + which of them the CPU's top-20 came from
                    let mut sorted: Vec<(u32, f32)> = cand_ids
                        .iter()
                        .copied()
                        .zip(cand_vals.iter().copied())
                        .collect();
                    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                    eprintln!("   gpu top-8 candidates (by raw logit):");
                    for (i, (tok, val)) in sorted.iter().take(8).enumerate() {
                        eprintln!("     [{i}] tok={tok} raw_logit={val}");
                    }
                    panic!("sample comparison failed");
                }
                // Both matched and advanced RNG to the same state. Leave it at rng_after_gpu.
                gpu_tok
            } else {
                llama::sample_top_p_from_candidates(
                    &cand_ids,
                    &mut cand_vals,
                    &token_history,
                    sc.repeat_window,
                    sc.repeat_penalty,
                    temp,
                    sc.top_p,
                )
            }
        } else {
            logits = gpu.download_f32(&scratch.logits).unwrap();
            llama::apply_repeat_penalty(
                &mut logits,
                &token_history,
                sc.repeat_window,
                sc.repeat_penalty,
            );
            llama::sample_top_p(&logits, temp, sc.top_p)
        };
    }

    let gen_ms = t2.elapsed().as_millis();
    let tok_s = if gen_ms > 0 {
        generated.len() as f64 / (gen_ms as f64 / 1000.0)
    } else {
        0.0
    };
    eprintln!(
        "\n\n=== Done: {} tokens in {}ms ({:.1} tok/s) ===",
        generated.len(),
        gen_ms,
        tok_s
    );
}
