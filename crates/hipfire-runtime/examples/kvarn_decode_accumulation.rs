//! Decode-time KV-cache error ACCUMULATION on a REAL qwen35 forward.
//!
//! The synthetic pseudo-decode proxy (hipfire-kvquant example
//! `kvarn_pseudo_decode`) showed the shape but its driving process was invented.
//! This runs the same experiment where it actually counts: two identical decode
//! runs over one model, differing ONLY in the KV tier, measuring how far the
//! quantised run's logits drift from an fp32-KV reference as steps accumulate.
//!
//! TEACHER FORCING is the load-bearing detail: both runs are fed the SAME token
//! at every step. Left to sample their own, they diverge in token space and the
//! drift stops being attributable to KV error at all.
//!
//! Reports per step, quantised vs fp32-KV reference:
//!   KLD     — KL(ref || test) over the full vocabulary, the metric that matters
//!   top1    — whether the argmax still agrees
//!   relerr  — relative L2 of the logit vector
//!
//!   kvarn_decode_accumulation <model.hfq> [--steps 64] [--prompt-len 32]
//!                             [--kv kvarn|q8|asym3] [--bits 2|4|8]
use hipfire_arch_qwen35::qwen35::{self, DeltaNetState, Qwen35Scratch};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::kv::KvCache;

fn softmax(x: &[f32]) -> Vec<f64> {
    let m = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f64> = x.iter().map(|v| ((v - m) as f64).exp()).collect();
    let z: f64 = e.iter().sum::<f64>().max(1e-300);
    e.into_iter().map(|v| v / z).collect()
}

fn main() {
    let argv: Vec<String> = std::env::args().collect();
    if argv.len() < 2 {
        eprintln!("usage: kvarn_decode_accumulation <model.hfq> [--steps N] [--prompt-len N] [--kv MODE] [--bits N]");
        std::process::exit(2);
    }
    let model = argv[1].clone();
    let mut steps = 64usize;
    let mut prompt_len = 32usize;
    let mut kv_mode = "kvarn".to_string();
    let mut i = 2;
    while i < argv.len() {
        match argv[i].as_str() {
            "--steps" => { steps = argv[i + 1].parse().unwrap(); i += 2; }
            "--prompt-len" => { prompt_len = argv[i + 1].parse().unwrap(); i += 2; }
            "--kv" => { kv_mode = argv[i + 1].clone(); i += 2; }
            "--bits" => { std::env::set_var("HIPFIRE_KVARN_BITS", &argv[i + 1]); i += 2; }
            _ => i += 1,
        }
    }

    let mut hfq = HfqFile::open(std::path::Path::new(&model)).expect("open model");
    let config = qwen35::config_from_hfq(&hfq).expect("config");
    let mut gpu = hipfire_rdna::Gpu::init().expect("gpu init");
    let weights = qwen35::load_weights(&mut hfq, &config, &mut gpu).expect("weights");
    let kv_max = prompt_len + steps + 16;
    eprintln!(
        "arch={} dim={} layers={} prompt={prompt_len} steps={steps} kv={kv_mode} bits={}",
        gpu.arch,
        config.dim,
        config.n_layers,
        KvCache::kvarn_bits_from_env()
    );

    // KVarN quantises in blocks of GROUP tokens and keeps the trailing partial
    // block in an f32 WINDOW. Below GROUP total positions nothing has been
    // flushed, so every bit width returns bit-identical numbers and the tool
    // silently reports "the KV tier does not matter". This has now bitten three
    // separate diagnostics in this repo (compare_prefill_hidden_paths at n=48,
    // the prefill chunk boundary at 256, and this at GROUP).
    const KVARN_GROUP: usize = 128;
    let total_pos = prompt_len + steps;
    if kv_mode == "kvarn" && total_pos <= KVARN_GROUP {
        eprintln!(
            "WARNING: prompt+steps = {total_pos} <= KVarN GROUP ({KVARN_GROUP}), so NOTHING is \
             flushed out of the f32 window and every --bits will read identically. \
             Use more steps."
        );
    }

    // Deterministic prompt; the content is irrelevant since both runs see it.
    let prompt: Vec<u32> = (0..prompt_len as u32).map(|i| 1000 + i * 7).collect();

    let mut run = |mode: &str| -> Vec<Vec<f32>> {
        let mut kv = match mode {
            "fp32" => KvCache::new_gpu(&mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_max),
            "q8" => KvCache::new_gpu_q8(&mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_max),
            "asym3" => KvCache::new_gpu_asym3(&mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_max),
            "kvarn" => KvCache::new_gpu_kvarn(
                &mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_max,
                KvCache::kvarn_bits_from_env(),
            ),
            other => panic!("unknown --kv {other} (have fp32, q8, asym3, kvarn)"),
        }
        .expect("kv alloc");
        let mut dn = DeltaNetState::new(&mut gpu, &config).expect("dn state");
        let scratch = Qwen35Scratch::new_with_kv_max(&mut gpu, &config, 128, kv_max).expect("scratch");

        // prompt
        for (p, t) in prompt.iter().enumerate() {
            qwen35::forward_scratch(&mut gpu, &weights, &config, *t, p, &mut kv, &mut dn, &scratch)
                .expect("prompt forward");
        }
        // decode, TEACHER FORCED on a fixed token stream so both runs walk the
        // same path and any divergence is attributable to the KV tier alone.
        let mut out = Vec::with_capacity(steps);
        for s in 0..steps {
            let tok = 2000 + (s as u32 * 13) % 5000;
            let pos = prompt_len + s;
            qwen35::forward_scratch(&mut gpu, &weights, &config, tok, pos, &mut kv, &mut dn, &scratch)
                .expect("decode forward");
            out.push(gpu.download_f32(&scratch.logits).expect("logits"));
        }
        out
    };

    let reference = run("fp32");
    let test = run(&kv_mode);

    println!("\n{:>6} {:>12} {:>10} {:>12}", "step", "KLD", "top1", "relerr");
    let mut first_top1_break = None;
    for (s, (r, t)) in reference.iter().zip(&test).enumerate() {
        let (pr, pt) = (softmax(r), softmax(t));
        let kld: f64 = pr.iter().zip(&pt).map(|(a, b)| a * ((a + 1e-30).ln() - (b + 1e-30).ln())).sum();
        let am = |v: &[f32]| v.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
        let same = am(r) == am(t);
        if !same && first_top1_break.is_none() {
            first_top1_break = Some(s + 1);
        }
        let num: f64 = r.iter().zip(t).map(|(a, b)| ((a - b) as f64).powi(2)).sum();
        let den: f64 = r.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().max(1e-30);
        if s < 4 || (s + 1) % 8 == 0 || s + 1 == reference.len() {
            println!("{:>6} {:>12.6} {:>10} {:>12.6}", s + 1, kld, if same { "ok" } else { "DIFF" }, (num / den).sqrt());
        }
    }
    match first_top1_break {
        Some(s) => println!("\nfirst top-1 disagreement at step {s}"),
        None => println!("\ntop-1 agreed at every step"),
    }
}
