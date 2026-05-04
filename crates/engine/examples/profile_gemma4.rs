//! Per-kernel profiler for Gemma 4 forward_scratch (single-token path).
//!
//! Mirrors profile_qwen35_mq4 but adapted to Gemma 4's two-cache layout
//! (sliding + full) and the lack of a batched prefill — warmup is done
//! token-serial through forward_scratch, which is the same hot path
//! decode uses, so the profile is representative of where per-token
//! time goes.
//!
//! Usage: profile_gemma4 <model.hfq> [--prefill N] [--warmup N] [--profile-steps N]

#[cfg(not(feature = "deltanet"))]
fn main() { eprintln!("build with --features deltanet"); }

#[cfg(feature = "deltanet")]
fn main() {
    use engine::hfq::HfqFile;
    use engine::gemma4::{self, Gemma4Scratch};
    use engine::llama::{self, KvCache};
    use rdna_compute::profile;
    use std::collections::BTreeMap;
    use std::path::Path;
    use std::time::Instant;

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: profile_gemma4 <model.hfq> [--prefill N] [--warmup N] [--profile-steps N]");
        std::process::exit(1);
    }
    let model_path = &args[1];

    let mut prefill_len: usize = 16;
    let mut warmup_len: usize = 4;
    let mut profile_steps: usize = 10;
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--prefill"       => { prefill_len    = args[i + 1].parse().unwrap(); i += 2; }
            "--warmup"        => { warmup_len     = args[i + 1].parse().unwrap(); i += 2; }
            "--profile-steps" => { profile_steps  = args[i + 1].parse().unwrap(); i += 2; }
            other => { eprintln!("unknown arg: {other}"); std::process::exit(1); }
        }
    }

    eprintln!("=== profile_gemma4 ===");
    eprintln!("Model: {model_path}");
    eprintln!("Prefill: {prefill_len}  Warmup: {warmup_len}  Profile: {profile_steps}");

    let hfq = HfqFile::open(Path::new(model_path)).expect("open model");
    let config = gemma4::config_from_hfq(&hfq).expect("read gemma4 config");
    eprintln!("Config: dim={} layers={} sliding_n_kv_heads={} full_n_kv_heads={} sliding_hd={} full_hd={} mi={} num_experts={}",
        config.dim, config.n_layers, config.sliding_n_kv_heads, config.full_n_kv_heads,
        config.sliding_head_dim, config.full_head_dim,
        config.moe_intermediate_size, config.num_experts);

    let mut gpu = rdna_compute::Gpu::init().expect("gpu init");
    eprintln!("GPU: {}", gpu.arch);

    let t_load = Instant::now();
    let weights = gemma4::load_weights(&hfq, &config, &mut gpu).expect("load weights");
    eprintln!("Weights loaded in {:.2}s", t_load.elapsed().as_secs_f64());

    let kv_seq = (prefill_len + warmup_len + profile_steps + 16).max(512);
    let plan = config.kv_share_plan();
    let mut kv_sliding = KvCache::new_gpu_asym3(
        &mut gpu, plan.n_sliding_own, config.sliding_n_kv_heads, config.sliding_head_dim, kv_seq
    ).expect("kv sliding");
    let mut kv_full = KvCache::new_gpu_asym3(
        &mut gpu, plan.n_full_own, config.full_n_kv_heads, config.full_head_dim, kv_seq
    ).expect("kv full");
    let scratch = Gemma4Scratch::new(&mut gpu, &config, kv_seq).expect("scratch");
    gemma4::init_scratch_constants(&mut gpu, &scratch, config.full_head_dim).expect("init");

    // Deterministic fake prompt: tokens 1..prefill_len (token 0 often a special).
    let prompt_tokens: Vec<u32> = (1..=prefill_len as u32).collect();
    eprintln!("\nPrefill {prefill_len} tokens (token-serial — Gemma 4 has no batched prefill)...");
    let t_prefill = Instant::now();
    for (i, &tok) in prompt_tokens.iter().enumerate() {
        gemma4::forward_scratch(
            &mut gpu, &weights, &config, tok, i,
            &mut kv_sliding, &mut kv_full, &scratch,
        ).expect("prefill forward failed");
    }
    let prefill_ms = t_prefill.elapsed().as_secs_f64() * 1000.0;
    eprintln!("  prefill: {:.1}ms ({:.1} tok/s)", prefill_ms, prefill_len as f64 / (prefill_ms / 1000.0));
    let logits = gpu.download_f32(&scratch.logits).unwrap();
    let mut next_token = llama::argmax(&logits);

    eprintln!("Warmup {warmup_len} steps (untimed)...");
    for step in 0..warmup_len {
        let pos = prompt_tokens.len() + step;
        gemma4::forward_scratch(
            &mut gpu, &weights, &config, next_token, pos,
            &mut kv_sliding, &mut kv_full, &scratch,
        ).expect("warmup forward failed");
        let logits = gpu.download_f32(&scratch.logits).unwrap();
        next_token = llama::argmax(&logits);
    }

    // === PROFILED PHASE ===
    eprintln!("\n=== profiled run: {profile_steps} gen steps at ctx ~{} ===",
        prompt_tokens.len() + warmup_len);
    profile::start();
    let t_profile = Instant::now();
    for step in 0..profile_steps {
        let pos = prompt_tokens.len() + warmup_len + step;
        gemma4::forward_scratch(
            &mut gpu, &weights, &config, next_token, pos,
            &mut kv_sliding, &mut kv_full, &scratch,
        ).expect("profile forward failed");
        let logits = gpu.download_f32(&scratch.logits).unwrap();
        next_token = llama::argmax(&logits);
    }
    let profile_wall_ms = t_profile.elapsed().as_secs_f64() * 1000.0;
    let entries = profile::stop().unwrap_or_default();
    eprintln!("Captured {} profile entries over {} steps", entries.len(), profile_steps);
    eprintln!("Wall time under profiling: {profile_wall_ms:.1}ms ({:.2}ms/step)",
        profile_wall_ms / profile_steps as f64);

    #[derive(Default)]
    struct Agg { calls: usize, total_us: f64, total_bytes: usize }

    // Aggregate by (category, kernel)
    let mut by_kernel: BTreeMap<(&'static str, &'static str), Agg> = BTreeMap::new();
    let mut total_us = 0.0f64;
    let mut total_bytes = 0usize;
    for e in &entries {
        let a = by_kernel.entry((e.category, e.kernel)).or_default();
        a.calls += 1;
        a.total_us += e.time_us;
        a.total_bytes += e.bytes;
        total_us += e.time_us;
        total_bytes += e.bytes;
    }

    let mut sorted: Vec<_> = by_kernel.into_iter().collect();
    sorted.sort_by(|a, b| b.1.total_us.partial_cmp(&a.1.total_us).unwrap());

    println!();
    println!("PER-KERNEL (top 30 by total time):");
    println!("{:<4} {:<10} {:<42} {:>8} {:>11} {:>10} {:>12} {:>9} {:>7}",
        "rnk", "category", "kernel", "calls", "total_us", "avg_us", "total_MiB", "GiB/s", "%");
    println!("{:-<118}", "");
    for (rank, ((cat, name), a)) in sorted.iter().take(30).enumerate() {
        let avg_us = a.total_us / a.calls as f64;
        let mib = a.total_bytes as f64 / (1024.0 * 1024.0);
        let gbps = if a.total_us > 0.0 {
            (a.total_bytes as f64 / (1024.0 * 1024.0 * 1024.0)) / (a.total_us / 1_000_000.0)
        } else { 0.0 };
        let pct = a.total_us * 100.0 / total_us;
        println!("{:<4} {:<10} {:<42} {:>8} {:>10.1}us {:>9.2}us {:>10.1} MiB {:>8.1} {:>6.2}%",
            rank + 1, cat, name, a.calls, a.total_us, avg_us, mib, gbps, pct);
    }
    println!("{:-<118}", "");

    // Aggregate by category
    let mut by_cat: BTreeMap<&'static str, Agg> = BTreeMap::new();
    for e in &entries {
        let a = by_cat.entry(e.category).or_default();
        a.calls += 1;
        a.total_us += e.time_us;
        a.total_bytes += e.bytes;
    }
    let mut sorted_cat: Vec<_> = by_cat.into_iter().collect();
    sorted_cat.sort_by(|a, b| b.1.total_us.partial_cmp(&a.1.total_us).unwrap());

    println!();
    println!("BY CATEGORY:");
    println!("{:<14} {:>8} {:>11} {:>9} {:>7}", "category", "calls", "total_us", "%", "");
    println!("{:-<60}", "");
    for (cat, a) in &sorted_cat {
        let pct = a.total_us * 100.0 / total_us;
        println!("{:<14} {:>8} {:>10.1}us {:>6.2}%", cat, a.calls, a.total_us, pct);
    }

    println!();
    println!("Per-step (averaged over {profile_steps} profiled steps):");
    println!("  kernel time: {:.2}ms", total_us / 1000.0 / profile_steps as f64);
    println!("  wall time:   {:.2}ms (profiling serializes launches)",
        profile_wall_ms / profile_steps as f64);
    println!("  num launches/step: {}", entries.len() / profile_steps);
}
