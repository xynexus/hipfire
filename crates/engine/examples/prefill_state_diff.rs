//! Post-prefill state diff: two batched runs + one per-token reference.
//!
//! Runs the SAME prompt through THREE prefill invocations, each with its own
//! KV cache + DeltaNet state, then FNV-1a hashes every per-layer buffer and
//! prints a side-by-side diff:
//!
//!   A1 — batched prefill, run 1   (snapshots x_batch after each layer)
//!   A2 — batched prefill, run 2   (snapshots x_batch after each layer)
//!   B  — per-token prefill        (known-good reference; no snapshot)
//!
//! What the output tells you:
//!   - A1 x_batch vs A2 x_batch: first mismatching layer is where
//!     NONDETERMINISM enters. The layer's kernels produced different bytes
//!     for byte-identical inputs.
//!   - A1 KV/DN vs B KV/DN: first mismatching layer is where the BATCHED
//!     path diverges from the known-good per-token reference — root cause of
//!     the correctness bug.
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
    use rdna_compute::{DType, GpuTensor};
    use std::path::Path;

    fn fnv1a(bytes: &[u8]) -> u64 {
        let mut h: u64 = 0xcbf29ce484222325;
        for &b in bytes {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        h
    }

    /// Download the RAW device-buffer bytes (respecting the actual allocation
    /// size, not `numel * dtype.size()` — the latter over-estimates for Q8
    /// DeltaNet S-matrices where shape is kept in f32 units but the malloc
    /// is `numel` bytes).
    fn download_bytes(gpu: &rdna_compute::Gpu, t: &GpuTensor) -> Vec<u8> {
        let n = t.buf.size();
        let mut out = vec![0u8; n];
        gpu.hip.memcpy_dtoh(&mut out, &t.buf).expect("dtoh");
        out
    }

    fn hash_gpu(gpu: &rdna_compute::Gpu, t: &GpuTensor) -> u64 {
        fnv1a(&download_bytes(gpu, t))
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
    let n = tokens.len();

    // Allocate per-layer x_batch snapshot buffers for both batched runs.
    // Each snapshot is [n × dim] f32 = layer output seen by the NEXT layer.
    let alloc_snaps = |gpu: &mut rdna_compute::Gpu| -> Vec<GpuTensor> {
        (0..config.n_layers)
            .map(|_| gpu.zeros(&[n * config.dim], DType::F32).expect("snap alloc"))
            .collect()
    };
    let snaps_a1 = alloc_snaps(&mut gpu);
    let snaps_a2 = alloc_snaps(&mut gpu);

    // ── Side A1: batched prefill, first run ───────────────────────────────
    let mut kv_a1 = KvCache::new_gpu_q8(
        &mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_seq,
    ).expect("kv_a1 alloc");
    let mut dn_a1 = DeltaNetState::new(&mut gpu, &config).expect("dn_a1 alloc");
    let scratch_a1 = Qwen35Scratch::new(&mut gpu, &config, 64).expect("scratch_a1 alloc");
    let pbs_a1 = PrefillBatchScratch::new(&mut gpu, &config, n.max(8))
        .expect("pbs_a1 alloc");

    eprintln!("\n[A1] Running BATCHED prefill (run 1) ...");
    std::env::remove_var("HIPFIRE_PREFILL_BATCHED");
    qwen35::forward_prefill_batch_with_pbs_and_snapshots(
        &mut gpu, &weights, &config, &tokens, 0,
        &mut kv_a1, &mut dn_a1, &scratch_a1,
        None, None, None, None, Some(&pbs_a1),
        Some(&snaps_a1),
    ).expect("batched prefill A1");
    gpu.hip.device_synchronize().expect("sync A1");

    // ── Side A2: batched prefill, second run (fresh state) ────────────────
    let mut kv_a2 = KvCache::new_gpu_q8(
        &mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_seq,
    ).expect("kv_a2 alloc");
    let mut dn_a2 = DeltaNetState::new(&mut gpu, &config).expect("dn_a2 alloc");
    let scratch_a2 = Qwen35Scratch::new(&mut gpu, &config, 64).expect("scratch_a2 alloc");
    let pbs_a2 = PrefillBatchScratch::new(&mut gpu, &config, n.max(8))
        .expect("pbs_a2 alloc");

    eprintln!("[A2] Running BATCHED prefill (run 2) ...");
    qwen35::forward_prefill_batch_with_pbs_and_snapshots(
        &mut gpu, &weights, &config, &tokens, 0,
        &mut kv_a2, &mut dn_a2, &scratch_a2,
        None, None, None, None, Some(&pbs_a2),
        Some(&snaps_a2),
    ).expect("batched prefill A2");
    gpu.hip.device_synchronize().expect("sync A2");

    // ── Side B: per-token reference ───────────────────────────────────────
    let mut kv_b = KvCache::new_gpu_q8(
        &mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_seq,
    ).expect("kv_b alloc");
    let mut dn_b = DeltaNetState::new(&mut gpu, &config).expect("dn_b alloc");
    let scratch_b = Qwen35Scratch::new(&mut gpu, &config, 64).expect("scratch_b alloc");

    eprintln!("[B]  Running PER-TOKEN prefill ...");
    std::env::set_var("HIPFIRE_PREFILL_BATCHED", "0");
    for (pos, &tok) in tokens.iter().enumerate() {
        qwen35::forward_scratch(
            &mut gpu, &weights, &config, tok, pos,
            &mut kv_b, &mut dn_b, &scratch_b,
        ).expect("per-token prefill");
    }
    gpu.hip.device_synchronize().expect("sync B");

    // ── Per-layer x_batch diff (batched run 1 vs run 2) ───────────────────
    eprintln!("\n=== Per-layer x_batch hashes (A1 vs A2) — locates NONDETERMINISM ===");
    eprintln!("{:>6}  {:<8}  {:<16}  {:<16}  match?",
        "layer", "type", "x_batch[A1]", "x_batch[A2]");
    let mut first_nondet: Option<usize> = None;
    for l in 0..config.n_layers {
        let ha1 = hash_gpu(&gpu, &snaps_a1[l]);
        let ha2 = hash_gpu(&gpu, &snaps_a2[l]);
        let ok = ha1 == ha2;
        if !ok && first_nondet.is_none() { first_nondet = Some(l); }
        let lt = match config.layer_types[l] {
            engine::qwen35::LayerType::LinearAttention => "LA",
            engine::qwen35::LayerType::FullAttention => "FA",
        };
        eprintln!("{:>6}  {:<8}  {:016x}  {:016x}  {}",
            l, lt, ha1, ha2, if ok { "OK" } else { "DIFF" });
    }

    // ── Per-layer KV nondeterminism (A1 vs A2) + correctness (A1 vs B) ────
    //
    // Only FA layers write KV cache; only LA layers have DN state. Both hash
    // the full device buffer — unwritten tail is zeros in both caches so the
    // hash is still a valid comparator.
    eprintln!("\n=== Per-layer KV hashes (A1 vs A2 = determinism, A1 vs B = correctness) ===");
    eprintln!("{:>6}  {:<16}  {:<16}  {:<16}  det?  cor?",
        "layer", "K[A1]", "K[A2]", "K[B]");
    let mut first_kv_nondet: Option<usize> = None;
    let mut first_kv_wrong: Option<usize> = None;
    for l in 0..config.n_layers {
        let hka1 = hash_gpu(&gpu, &kv_a1.k_gpu[l]);
        let hka2 = hash_gpu(&gpu, &kv_a2.k_gpu[l]);
        let hkb  = hash_gpu(&gpu, &kv_b.k_gpu[l]);
        let hva1 = hash_gpu(&gpu, &kv_a1.v_gpu[l]);
        let hva2 = hash_gpu(&gpu, &kv_a2.v_gpu[l]);
        let hvb  = hash_gpu(&gpu, &kv_b.v_gpu[l]);
        let det = hka1 == hka2 && hva1 == hva2;
        let cor = hka1 == hkb  && hva1 == hvb;
        if !det && first_kv_nondet.is_none() { first_kv_nondet = Some(l); }
        if !cor && first_kv_wrong.is_none()  { first_kv_wrong  = Some(l); }
        eprintln!("{:>6}  {:016x}  {:016x}  {:016x}  {:<4}  {:<4}",
            l, hka1, hka2, hkb,
            if det { "OK" } else { "DIFF" },
            if cor { "OK" } else { "DIFF" });
    }

    // ── DN state nondeterminism (A1 vs A2) + correctness (A1 vs B) ────────
    eprintln!("\n=== DeltaNet state hashes (A1/A2/B) ===");
    eprintln!("{:>6}  {:<16}  {:<16}  {:<16}  det?  cor?",
        "dn_l", "S[A1]", "S[A2]", "S[B]");
    let mut first_dn_nondet: Option<usize> = None;
    let mut first_dn_wrong: Option<usize> = None;
    let n_dn = dn_a1.s_matrices.len();
    for dl in 0..n_dn {
        let hsa1 = hash_gpu(&gpu, &dn_a1.s_matrices[dl]);
        let hsa2 = hash_gpu(&gpu, &dn_a2.s_matrices[dl]);
        let hsb  = hash_gpu(&gpu, &dn_b.s_matrices[dl]);
        let hssa1 = hash_gpu(&gpu, &dn_a1.s_scales[dl]);
        let hssa2 = hash_gpu(&gpu, &dn_a2.s_scales[dl]);
        let hssb  = hash_gpu(&gpu, &dn_b.s_scales[dl]);
        let hca1 = hash_gpu(&gpu, &dn_a1.conv_states[dl]);
        let hca2 = hash_gpu(&gpu, &dn_a2.conv_states[dl]);
        let hcb  = hash_gpu(&gpu, &dn_b.conv_states[dl]);
        let det = hsa1 == hsa2 && hssa1 == hssa2 && hca1 == hca2;
        let cor = hsa1 == hsb  && hssa1 == hssb  && hca1 == hcb;
        if !det && first_dn_nondet.is_none() { first_dn_nondet = Some(dl); }
        if !cor && first_dn_wrong.is_none()  { first_dn_wrong  = Some(dl); }
        eprintln!("{:>6}  {:016x}  {:016x}  {:016x}  {:<4}  {:<4}",
            dl, hsa1, hsa2, hsb,
            if det { "OK" } else { "DIFF" },
            if cor { "OK" } else { "DIFF" });
    }

    // ── Summary ───────────────────────────────────────────────────────────
    eprintln!("\n=== Summary ===");
    match first_nondet {
        Some(l) => eprintln!("FIRST NONDETERMINISTIC LAYER (A1 vs A2 x_batch): layer {} ({:?})",
            l, config.layer_types[l]),
        None => eprintln!("Batched is DETERMINISTIC across runs — A1 x_batch == A2 x_batch for all layers"),
    }
    match first_kv_wrong {
        Some(l) => eprintln!("FIRST WRONG KV LAYER (A1 vs B): layer {} ({:?})",
            l, config.layer_types[l]),
        None => eprintln!("KV cache matches per-token reference across all layers"),
    }
    match first_dn_wrong {
        Some(dl) => eprintln!("FIRST WRONG DN LAYER (A1 vs B): dn_idx {}", dl),
        None => eprintln!("DN state matches per-token reference across all layers"),
    }
}
