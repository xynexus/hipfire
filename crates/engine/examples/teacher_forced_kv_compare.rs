//! Teacher-forced same-model KV-mode comparator.
//!
//! Loads one Qwen3.5 HFQ model, creates two KV caches with different modes,
//! and compares per-step logits while both paths consume the reference argmax.
//! This isolates local KV quantization drift from free-running autoregressive
//! history divergence.
//!
//! Usage:
//!   teacher_forced_kv_compare <model.hfq> <out.csv> [--ref-mode q8] [--cand-mode asym4_tqv4]
//!       [--max-gen N] [--ctx N] [--prompt-mode raw|chat|thinking] [--system TEXT] [prompt...]

#[cfg(not(feature = "deltanet"))]
fn main() {
    eprintln!("build with --features deltanet");
}

#[cfg(feature = "deltanet")]
fn main() {
    use engine::hfq::HfqFile;
    use engine::llama::{self, KvCache};
    use engine::qwen35::{self, DeltaNetState, Qwen35Scratch};
    use std::collections::HashSet;
    use std::io::Write;
    use std::path::Path;

    fn usage(program: &str) -> ! {
        eprintln!(
            "Usage: {program} <model.hfq> <out.csv> [--ref-mode MODE] [--cand-mode MODE] \
             [--max-gen N] [--ctx N] [--prompt-mode raw|chat|thinking] [--system TEXT] [prompt...]"
        );
        std::process::exit(2);
    }

    fn make_kv(
        gpu: &mut rdna_compute::Gpu,
        config: &qwen35::Qwen35Config,
        mode: &str,
        kv_seq: usize,
    ) -> KvCache {
        match mode {
            "fp32" | "f32" => KvCache::new_gpu(
                gpu,
                config.n_layers,
                config.n_kv_heads,
                config.head_dim,
                kv_seq,
            )
            .unwrap(),
            "q8" => KvCache::new_gpu_q8(
                gpu,
                config.n_layers,
                config.n_kv_heads,
                config.head_dim,
                kv_seq,
            )
            .unwrap(),
            "asym4_tqv1" | "tqv1" | "tq1" => KvCache::new_gpu_asym4_tqv1_capped(
                gpu,
                config.n_layers,
                config.n_kv_heads,
                config.head_dim,
                kv_seq,
                kv_seq,
            )
            .unwrap(),
            "asym4_tqv2" | "tqv2" => KvCache::new_gpu_asym4_tqv2_capped(
                gpu,
                config.n_layers,
                config.n_kv_heads,
                config.head_dim,
                kv_seq,
                kv_seq,
            )
            .unwrap(),
            "asym4_tqv3" | "tqv3" => KvCache::new_gpu_asym4_tqv3_capped(
                gpu,
                config.n_layers,
                config.n_kv_heads,
                config.head_dim,
                kv_seq,
                kv_seq,
            )
            .unwrap(),
            "asym4_tqv4" | "tqv4" => KvCache::new_gpu_asym4_tqv4_capped(
                gpu,
                config.n_layers,
                config.n_kv_heads,
                config.head_dim,
                kv_seq,
                kv_seq,
            )
            .unwrap(),
            "asym4" | "turbo4" => KvCache::new_gpu_asym4(
                gpu,
                config.n_layers,
                config.n_kv_heads,
                config.head_dim,
                kv_seq,
            )
            .unwrap(),
            "asym3" | "turbo3" | "turbo" => KvCache::new_gpu_asym3(
                gpu,
                config.n_layers,
                config.n_kv_heads,
                config.head_dim,
                kv_seq,
            )
            .unwrap(),
            "asym2" | "turbo2" => KvCache::new_gpu_asym2(
                gpu,
                config.n_layers,
                config.n_kv_heads,
                config.head_dim,
                kv_seq,
            )
            .unwrap(),
            other => panic!(
                "unknown KV mode: {other} (use fp32|q8|asym4_tqv1|asym4_tqv2|asym4_tqv3|asym4_tqv4|asym4|asym3|asym2)"
            ),
        }
    }

    fn top5(logits: &[f32]) -> [(u32, f32); 5] {
        let mut best: [(u32, f32); 5] = [(0, f32::NEG_INFINITY); 5];
        for (i, &v) in logits.iter().enumerate() {
            if v <= best[4].1 {
                continue;
            }
            best[4] = (i as u32, v);
            for j in (1..5).rev() {
                if best[j].1 > best[j - 1].1 {
                    best.swap(j, j - 1);
                } else {
                    break;
                }
            }
        }
        best
    }

    fn rank_of(logits: &[f32], token: u32) -> u32 {
        let v = logits[token as usize];
        1 + logits.iter().filter(|&&x| x > v).count() as u32
    }

    fn top5_overlap(a: &[(u32, f32); 5], b: &[(u32, f32); 5]) -> u32 {
        let ids: HashSet<u32> = a.iter().map(|x| x.0).collect();
        b.iter().filter(|x| ids.contains(&x.0)).count() as u32
    }

    fn logit_metrics(a: &[f32], b: &[f32]) -> (f32, f32, f32) {
        let mut max_abs = 0.0f32;
        let mut sum_abs = 0.0f32;
        let mut dot = 0.0f32;
        let mut aa = 0.0f32;
        let mut bb = 0.0f32;
        for (&x, &y) in a.iter().zip(b) {
            let d = (x - y).abs();
            max_abs = max_abs.max(d);
            sum_abs += d;
            dot += x * y;
            aa += x * x;
            bb += y * y;
        }
        let mean_abs = sum_abs / a.len() as f32;
        let cosine = dot / ((aa.sqrt() * bb.sqrt()).max(1e-20));
        (max_abs, mean_abs, cosine)
    }

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        usage(&args[0]);
    }
    let model_path = &args[1];
    let out_csv = &args[2];
    let mut ref_mode = "q8".to_string();
    let mut cand_mode = "asym4_tqv4".to_string();
    let mut max_gen = 64usize;
    let mut kv_seq = 2048usize;
    let mut prompt_mode = std::env::var("PROMPT_MODE").unwrap_or_else(|_| "thinking".to_string());
    let mut system_prompt: Option<String> = None;
    let mut prompt_args = Vec::new();
    let mut i = 3;
    while i < args.len() {
        match args[i].as_str() {
            "--ref-mode" => {
                ref_mode = args.get(i + 1).cloned().unwrap_or_else(|| usage(&args[0]));
                i += 2;
            }
            "--cand-mode" => {
                cand_mode = args.get(i + 1).cloned().unwrap_or_else(|| usage(&args[0]));
                i += 2;
            }
            "--max-gen" => {
                max_gen = args
                    .get(i + 1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or_else(|| usage(&args[0]));
                i += 2;
            }
            "--ctx" => {
                kv_seq = args
                    .get(i + 1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or_else(|| usage(&args[0]));
                i += 2;
            }
            "--prompt-mode" => {
                prompt_mode = args.get(i + 1).cloned().unwrap_or_else(|| usage(&args[0]));
                i += 2;
            }
            "--system" => {
                system_prompt = Some(args.get(i + 1).cloned().unwrap_or_else(|| usage(&args[0])));
                i += 2;
            }
            "-h" | "--help" => usage(&args[0]),
            other => {
                prompt_args.push(other.to_string());
                i += 1;
            }
        }
    }

    let prompt_text = if !prompt_args.is_empty() {
        prompt_args.join(" ")
    } else {
        "Write a Python function named square that returns x*x.".to_string()
    };

    eprintln!(
        "teacher_forced_kv_compare: model={model_path} ref={ref_mode} cand={cand_mode} mode={prompt_mode}"
    );

    let hfq = HfqFile::open(Path::new(model_path)).expect("open model");
    let config = qwen35::config_from_hfq(&hfq).expect("read config");
    let tokenizer =
        engine::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json).expect("tok");

    let prompt_tokens: Vec<u32> = match prompt_mode.as_str() {
        "raw" => tokenizer.encode(&prompt_text),
        _ => {
            let im_start = tokenizer.encode("<|im_start|>");
            let im_end = tokenizer.encode("<|im_end|>");
            let user = tokenizer.encode("user");
            let system = tokenizer.encode("system");
            let asst = tokenizer.encode("assistant");
            let nl = tokenizer.encode("\n");
            let user_body = tokenizer.encode(&prompt_text);
            let mut chat = Vec::new();
            if let Some(system_prompt) = system_prompt.as_deref() {
                chat.extend_from_slice(&im_start);
                chat.extend_from_slice(&system);
                chat.extend_from_slice(&nl);
                chat.extend_from_slice(&tokenizer.encode(system_prompt));
                chat.extend_from_slice(&im_end);
                chat.extend_from_slice(&nl);
            }
            chat.extend_from_slice(&im_start);
            chat.extend_from_slice(&user);
            chat.extend_from_slice(&nl);
            chat.extend_from_slice(&user_body);
            chat.extend_from_slice(&im_end);
            chat.extend_from_slice(&nl);
            chat.extend_from_slice(&im_start);
            chat.extend_from_slice(&asst);
            chat.extend_from_slice(&nl);
            if prompt_mode == "thinking" {
                chat.extend_from_slice(&tokenizer.encode("<think>"));
                chat.extend_from_slice(&nl);
            }
            chat
        }
    };
    if prompt_tokens.len() + max_gen + 8 > kv_seq {
        kv_seq = prompt_tokens.len() + max_gen + 8;
    }
    eprintln!("prompt: {} tokens, ctx={kv_seq}", prompt_tokens.len());

    let mut gpu = rdna_compute::Gpu::init().expect("gpu init");
    let weights = qwen35::load_weights(&hfq, &config, &mut gpu).expect("load weights");

    let mut ref_kv = make_kv(&mut gpu, &config, &ref_mode, kv_seq);
    let mut cand_kv = make_kv(&mut gpu, &config, &cand_mode, kv_seq);
    let mut ref_dn = DeltaNetState::new(&mut gpu, &config).unwrap();
    let mut cand_dn = DeltaNetState::new(&mut gpu, &config).unwrap();
    let ref_scratch = Qwen35Scratch::new(&mut gpu, &config, 128).unwrap();
    let cand_scratch = Qwen35Scratch::new(&mut gpu, &config, 128).unwrap();

    for (pos, &token) in prompt_tokens.iter().enumerate() {
        qwen35::forward_scratch(
            &mut gpu,
            &weights,
            &config,
            token,
            pos,
            &mut ref_kv,
            &mut ref_dn,
            &ref_scratch,
        )
        .expect("reference prefill forward failed");
        qwen35::forward_scratch(
            &mut gpu,
            &weights,
            &config,
            token,
            pos,
            &mut cand_kv,
            &mut cand_dn,
            &cand_scratch,
        )
        .expect("candidate prefill forward failed");
    }

    let mut out = std::fs::File::create(out_csv).expect("create out csv");
    writeln!(
        out,
        "step,pos,ref_token,cand_token,match,ref_margin,cand_margin,top5_overlap,cand_rank_of_ref,ref_logit_ref,cand_logit_ref,cand_logit_cand,logit_max_abs,logit_mean_abs,logit_cosine,ref_text,cand_text"
    )
    .ok();

    let mut first_divergence: Option<usize> = None;
    let mut matches = 0usize;
    let mut committed = Vec::with_capacity(max_gen);

    for step in 0..max_gen {
        let ref_logits = gpu.download_f32(&ref_scratch.logits).unwrap();
        let cand_logits = gpu.download_f32(&cand_scratch.logits).unwrap();
        let ref_token = llama::argmax(&ref_logits);
        let cand_token = llama::argmax(&cand_logits);
        let matched = ref_token == cand_token;
        if matched {
            matches += 1;
        } else if first_divergence.is_none() {
            first_divergence = Some(step);
        }
        let ref_top5 = top5(&ref_logits);
        let cand_top5 = top5(&cand_logits);
        let ref_margin = ref_top5[0].1 - ref_top5[1].1;
        let cand_margin = cand_top5[0].1 - cand_top5[1].1;
        let overlap = top5_overlap(&ref_top5, &cand_top5);
        let cand_rank = rank_of(&cand_logits, ref_token);
        let (max_abs, mean_abs, cosine) = logit_metrics(&ref_logits, &cand_logits);
        let ref_text = tokenizer
            .decode(&[ref_token])
            .replace('"', "'")
            .replace('\n', "\\n");
        let cand_text = tokenizer
            .decode(&[cand_token])
            .replace('"', "'")
            .replace('\n', "\\n");
        writeln!(
            out,
            "{step},{},{ref_token},{cand_token},{matched},{ref_margin:.8},{cand_margin:.8},{overlap},{cand_rank},{:.8},{:.8},{:.8},{max_abs:.8},{mean_abs:.8},{cosine:.8},\"{}\",\"{}\"",
            prompt_tokens.len() + step,
            ref_logits[ref_token as usize],
            cand_logits[ref_token as usize],
            cand_logits[cand_token as usize],
            ref_text,
            cand_text
        )
        .ok();

        committed.push(ref_token);
        if ref_token == config.eos_token {
            break;
        }

        let pos = prompt_tokens.len() + step;
        qwen35::forward_scratch(
            &mut gpu,
            &weights,
            &config,
            ref_token,
            pos,
            &mut ref_kv,
            &mut ref_dn,
            &ref_scratch,
        )
        .expect("reference forward failed");
        qwen35::forward_scratch(
            &mut gpu,
            &weights,
            &config,
            ref_token,
            pos,
            &mut cand_kv,
            &mut cand_dn,
            &cand_scratch,
        )
        .expect("candidate forward failed");
    }

    out.flush().ok();
    eprintln!(
        "steps={} matches={} agreement={:.4} first_divergence={:?}",
        committed.len(),
        matches,
        matches as f32 / committed.len().max(1) as f32,
        first_divergence
    );
    eprintln!("reference text:");
    println!("{}", tokenizer.decode(&committed));
}
