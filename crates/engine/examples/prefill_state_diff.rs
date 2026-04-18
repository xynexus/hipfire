//! Post-prefill state diff: batched vs per-token.
//!
//! Runs the same prompt through BOTH prefill paths (with completely separate
//! KV caches + DeltaNet states), then FNV-1a hashes every per-layer buffer
//! and prints a side-by-side diff. The first layer that diverges is the
//! subsystem containing the regression.
//!
//! Usage:
//!   cargo run --release --features deltanet --example prefill_state_diff -- \
//!       ~/.hipfire/models/qwen3.5-9b.mq4
//!
//! Env:
//!   HIPFIRE_PSD_PROMPT   — prompt text (default "The capital of France is")
//!   HIPFIRE_PSD_KV_SEQ   — KV cache max_seq (default 128)

#[cfg(not(feature = "deltanet"))]
fn main() { eprintln!("build with --features deltanet"); }

#[cfg(feature = "deltanet")]
fn main() {
    use engine::hfq::HfqFile;
    use engine::qwen35::{self, DeltaNetState, PrefillBatchScratch, Qwen35Scratch};
    use engine::llama::KvCache;
    use std::path::Path;

    fn fnv1a(bytes: &[u8]) -> u64 {
        let mut h: u64 = 0xcbf29ce484222325;
        for &b in bytes {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        h
    }

    fn f32_bytes(v: &[f32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(v.len() * 4);
        for &x in v { out.extend_from_slice(&x.to_ne_bytes()); }
        out
    }

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: prefill_state_diff <model.mq4>");
        std::process::exit(1);
    }
    let model_path = &args[1];

    let prompt = std::env::var("HIPFIRE_PSD_PROMPT")
        .unwrap_or_else(|_| "The capital of France is".to_string());
    let kv_seq: usize = std::env::var("HIPFIRE_PSD_KV_SEQ")
        .ok().and_then(|v| v.parse().ok()).unwrap_or(128);

    eprintln!("Opening: {model_path}");
    let hfq = HfqFile::open(Path::new(model_path)).expect("open model");
    let config = qwen35::config_from_hfq(&hfq).expect("read config");
    eprintln!("Config: dim={} layers={} n_heads={} n_kv_heads={}",
        config.dim, config.n_layers, config.n_heads, config.n_kv_heads);

    eprintln!("Loading weights ...");
    let mut gpu = rdna_compute::Gpu::init().expect("gpu init");
    let weights = qwen35::load_weights(&hfq, &config, &mut gpu).expect("load weights");

    let tokenizer = engine::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
        .expect("tokenizer");
    let tokens: Vec<u32> = tokenizer.encode(&prompt);
    eprintln!("Prompt: {:?}  ({} tokens)", prompt, tokens.len());
    if tokens.len() < 2 {
        eprintln!("Need >= 2 tokens to exercise the batched path; got {}", tokens.len());
        std::process::exit(1);
    }

    // ── Side A: batched prefill ───────────────────────────────────────────
    let mut kv_a = KvCache::new_gpu_q8(
        &mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_seq,
    ).expect("kv_a alloc");
    let mut dn_a = DeltaNetState::new(&mut gpu, &config).expect("dn_a alloc");
    let scratch_a = Qwen35Scratch::new(&mut gpu, &config, 64).expect("scratch_a alloc");
    let pbs = PrefillBatchScratch::new(&mut gpu, &config, tokens.len().max(8))
        .expect("pbs alloc");

    eprintln!("\n[A] Running BATCHED prefill ...");
    // IMPORTANT: clear env override if user set it for per-token fallback.
    std::env::remove_var("HIPFIRE_PREFILL_BATCHED");
    qwen35::forward_prefill_batch_with_pbs(
        &mut gpu, &weights, &config, &tokens, 0,
        &mut kv_a, &mut dn_a, &scratch_a,
        None, None, None, None, Some(&pbs),
    ).expect("batched prefill");
    gpu.hip.device_synchronize().expect("sync A");

    // ── Side B: per-token prefill ─────────────────────────────────────────
    let mut kv_b = KvCache::new_gpu_q8(
        &mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_seq,
    ).expect("kv_b alloc");
    let mut dn_b = DeltaNetState::new(&mut gpu, &config).expect("dn_b alloc");
    let scratch_b = Qwen35Scratch::new(&mut gpu, &config, 64).expect("scratch_b alloc");

    eprintln!("[B] Running PER-TOKEN prefill ...");
    std::env::set_var("HIPFIRE_PREFILL_BATCHED", "0");
    for (pos, &tok) in tokens.iter().enumerate() {
        qwen35::forward_scratch(
            &mut gpu, &weights, &config, tok, pos,
            &mut kv_b, &mut dn_b, &scratch_b,
        ).expect("per-token prefill");
    }
    gpu.hip.device_synchronize().expect("sync B");

    // ── Hash per-layer state ──────────────────────────────────────────────
    //
    // The written region of the KV cache is `tokens.len()` positions of size
    // `n_kv_heads * blocks_per_head * 34` bytes. Everything past that is
    // untouched zero in both caches, so a full-buffer hash is still a valid
    // comparator (zeros bytes agree trivially).
    eprintln!("\n=== Per-layer hashes (A=batched, B=per-token) ===");
    eprintln!("{:>6}  {:<16}  {:<16}  {:<16}  {:<16}  match?",
        "layer", "K[A]", "K[B]", "V[A]", "V[B]");

    let mut k_mismatch: Vec<usize> = Vec::new();
    let mut v_mismatch: Vec<usize> = Vec::new();
    for l in 0..config.n_layers {
        let ka = gpu.download_f32(&kv_a.k_gpu[l]).expect("dl ka");
        let kb = gpu.download_f32(&kv_b.k_gpu[l]).expect("dl kb");
        let va = gpu.download_f32(&kv_a.v_gpu[l]).expect("dl va");
        let vb = gpu.download_f32(&kv_b.v_gpu[l]).expect("dl vb");
        let hka = fnv1a(&f32_bytes(&ka));
        let hkb = fnv1a(&f32_bytes(&kb));
        let hva = fnv1a(&f32_bytes(&va));
        let hvb = fnv1a(&f32_bytes(&vb));
        let k_ok = hka == hkb;
        let v_ok = hva == hvb;
        if !k_ok { k_mismatch.push(l); }
        if !v_ok { v_mismatch.push(l); }
        let tag = match (k_ok, v_ok) {
            (true, true) => "OK",
            (false, true) => "K_DIFF",
            (true, false) => "V_DIFF",
            (false, false) => "KV_DIFF",
        };
        eprintln!("{:>6}  {:016x}  {:016x}  {:016x}  {:016x}  {}",
            l, hka, hkb, hva, hvb, tag);
    }

    // ── Hash DeltaNet state (only the LA layers have it) ──────────────────
    eprintln!("\n=== DeltaNet state hashes ===");
    eprintln!("{:>6}  {:<16}  {:<16}  {:<16}  {:<16}  {:<16}  {:<16}  match?",
        "dn_l", "S[A]", "S[B]", "Sscl[A]", "Sscl[B]", "Conv[A]", "Conv[B]");

    let mut s_mismatch: Vec<usize> = Vec::new();
    let mut sc_mismatch: Vec<usize> = Vec::new();
    let mut cv_mismatch: Vec<usize> = Vec::new();
    let n_dn = dn_a.s_matrices.len();
    for dl in 0..n_dn {
        let sa = gpu.download_f32(&dn_a.s_matrices[dl]).expect("dl sa");
        let sb = gpu.download_f32(&dn_b.s_matrices[dl]).expect("dl sb");
        let ssa = gpu.download_f32(&dn_a.s_scales[dl]).expect("dl ssa");
        let ssb = gpu.download_f32(&dn_b.s_scales[dl]).expect("dl ssb");
        let ca = gpu.download_f32(&dn_a.conv_states[dl]).expect("dl ca");
        let cb = gpu.download_f32(&dn_b.conv_states[dl]).expect("dl cb");
        let hsa = fnv1a(&f32_bytes(&sa));
        let hsb = fnv1a(&f32_bytes(&sb));
        let hssa = fnv1a(&f32_bytes(&ssa));
        let hssb = fnv1a(&f32_bytes(&ssb));
        let hca = fnv1a(&f32_bytes(&ca));
        let hcb = fnv1a(&f32_bytes(&cb));
        let s_ok = hsa == hsb;
        let sc_ok = hssa == hssb;
        let cv_ok = hca == hcb;
        if !s_ok { s_mismatch.push(dl); }
        if !sc_ok { sc_mismatch.push(dl); }
        if !cv_ok { cv_mismatch.push(dl); }
        let tag = if s_ok && sc_ok && cv_ok { "OK" } else { "DIFF" };
        eprintln!("{:>6}  {:016x}  {:016x}  {:016x}  {:016x}  {:016x}  {:016x}  {}",
            dl, hsa, hsb, hssa, hssb, hca, hcb, tag);
    }

    // ── Summary ───────────────────────────────────────────────────────────
    eprintln!("\n=== Summary ===");
    eprintln!("K mismatch layers: {:?}", k_mismatch);
    eprintln!("V mismatch layers: {:?}", v_mismatch);
    eprintln!("DN S mismatch:    {:?}", s_mismatch);
    eprintln!("DN Sscale mismatch:{:?}", sc_mismatch);
    eprintln!("DN Conv mismatch: {:?}", cv_mismatch);

    let first_diverge = [
        k_mismatch.first().copied().map(|i| ("K cache", i)),
        v_mismatch.first().copied().map(|i| ("V cache", i)),
    ].into_iter().flatten().min_by_key(|&(_, i)| i);
    if let Some((what, l)) = first_diverge {
        eprintln!("\nFIRST DIVERGING KV LAYER: {} @ layer {} (type {:?})",
            what, l, config.layer_types[l]);
    } else if !s_mismatch.is_empty() || !cv_mismatch.is_empty() {
        eprintln!("\nKV matches — divergence is in DeltaNet state");
    } else {
        eprintln!("\nAll state hashes MATCH — batched and per-token are byte-exact");
    }
}
