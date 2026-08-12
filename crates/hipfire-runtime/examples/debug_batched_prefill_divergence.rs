#![allow(
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::type_complexity
)]
// hipfire example clippy sweep: examples are GPU probes/benches, not reusable APIs.

// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Numerical bisection for the MiniCPM5-1B batched-prefill garbage bug.
//!
//! For a FIXED short token sequence, compute the last-position logits three
//! ways — each on a FRESH Q8 KvCache (the mode that forces the batched path in
//! the daemon) — and compare the two batched paths against the per-token
//! reference:
//!
//!   A. per-token   : `forward_scratch_embed` + `forward_scratch_compute` loop
//!                    (reference; not garbage).
//!   B. prefill_forward : batched, uses `attention_causal_batched`.
//!   C. forward_prefill_batch → forward_prefill_chunk : batched FLASH attention
//!                    (`attention_flash_*_batched_masked`) — the daemon serving
//!                    prefill path, the suspect.
//!
//! Reports argmax_match / max_abs_diff / cosine and top-5 argmax for B-vs-A and
//! C-vs-A so we can see WHICH batched path diverges from the reference.
//!
//! Usage: cargo run --release -p hipfire-runtime \
//!            --example debug_batched_prefill_divergence [model.hfq]

use hipfire_arch_llama::Llama;
use hipfire_rdna::Gpu;
use hipfire_runtime::arch::Architecture;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::kv::KvCache;
use hipfire_runtime::llama::{self, ForwardScratch, LlamaConfig, LlamaWeights};
use std::path::Path;

/// The fixed probe sequence. Eight ids ≥ MIN_BATCH(4) so path C takes the
/// batched flash kernel rather than the per-token fallback.
const TOKENS: &[u32] = &[1, 100, 200, 300, 400, 500, 600, 700];

fn fresh_q8_kv(gpu: &mut Gpu, config: &LlamaConfig) -> KvCache {
    // Mirror the daemon's llama Q8 build (serving-core/src/load.rs ~L2735):
    //   KvCache::new_gpu_q8(gpu, n_layers, n_kv_heads, head_dim, max_seq)
    let kv_seq = config.max_seq_len.min(2048);
    KvCache::new_gpu_q8(
        gpu,
        config.n_layers,
        config.n_kv_heads,
        config.head_dim,
        kv_seq,
    )
    .expect("KvCache::new_gpu_q8 failed")
}

/// Path A — per-token reference. Loops embed+compute over `tokens`, then returns
/// the last-position logits from `scratch.logits`.
fn run_pertoken(
    gpu: &mut Gpu,
    weights: &LlamaWeights,
    config: &LlamaConfig,
    scratch: &ForwardScratch,
    tokens: &[u32],
) -> Vec<f32> {
    let kv = fresh_q8_kv(gpu, config);
    let mut kv = kv;
    for (pos, &tok) in tokens.iter().enumerate() {
        llama::forward_scratch_embed(gpu, weights, config, tok, pos, scratch)
            .expect("forward_scratch_embed failed");
        llama::forward_scratch_compute(gpu, weights, config, pos, &mut kv, scratch)
            .expect("forward_scratch_compute failed");
    }
    let logits = gpu
        .download_f32(&scratch.logits)
        .expect("download logits (A)");
    kv.free_gpu(gpu);
    logits
}

/// Path B — `prefill_forward` (attention_causal_batched). Returns the
/// last-position logits directly as a Vec<f32>.
fn run_prefill_forward(
    gpu: &mut Gpu,
    weights: &LlamaWeights,
    config: &LlamaConfig,
    tokens: &[u32],
) -> Vec<f32> {
    let mut kv = fresh_q8_kv(gpu, config);
    let logits = llama::prefill_forward(gpu, weights, config, tokens, &mut kv)
        .expect("prefill_forward failed");
    kv.free_gpu(gpu);
    logits
}

/// Path C — `forward_prefill_batch` → `forward_prefill_chunk` (flash). Leaves
/// last-position logits in `scratch.logits`.
fn run_flash(
    gpu: &mut Gpu,
    weights: &LlamaWeights,
    config: &LlamaConfig,
    scratch: &ForwardScratch,
    tokens: &[u32],
) -> Vec<f32> {
    let mut kv = fresh_q8_kv(gpu, config);
    llama::forward_prefill_batch(gpu, weights, config, tokens, 0, &mut kv, scratch, None)
        .expect("forward_prefill_batch failed");
    let logits = gpu
        .download_f32(&scratch.logits)
        .expect("download logits (C)");
    kv.free_gpu(gpu);
    logits
}

fn argmax(v: &[f32]) -> usize {
    let mut bi = 0usize;
    let mut bv = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > bv {
            bv = x;
            bi = i;
        }
    }
    bi
}

fn top5(v: &[f32]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&a, &b| v[b].partial_cmp(&v[a]).unwrap_or(std::cmp::Ordering::Equal));
    idx.into_iter().take(5).collect()
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let mut dot = 0f64;
    let mut na = 0f64;
    let mut nb = 0f64;
    for i in 0..a.len().min(b.len()) {
        let x = a[i] as f64;
        let y = b[i] as f64;
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    let mut m = 0f32;
    for i in 0..a.len().min(b.len()) {
        let d = (a[i] - b[i]).abs();
        if d > m {
            m = d;
        }
    }
    m
}

fn report(label: &str, reference: &[f32], other: &[f32]) {
    let am_ref = argmax(reference);
    let am_oth = argmax(other);
    println!("--- {label} ---");
    println!("  argmax_match : {}", am_ref == am_oth);
    println!("  max_abs_diff : {:.6}", max_abs_diff(reference, other));
    println!("  cosine       : {:.8}", cosine(reference, other));
    println!("  ref  top5    : {:?}", top5(reference));
    println!("  this top5    : {:?}", top5(other));
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let default_model = format!(
        "{}/.hipfire/models/MiniCPM5-1B.bf16.hfq",
        std::env::var("HOME").unwrap_or_else(|_| "/root".to_string())
    );
    let model_path = args.get(1).cloned().unwrap_or(default_model);

    println!("=== batched-prefill divergence bisection ===");
    println!("model  : {model_path}");
    println!("tokens : {TOKENS:?}");

    let mut hfq = HfqFile::open(Path::new(&model_path)).expect("failed to parse HFQ");
    let config =
        <Llama as Architecture>::config_from_hfq(&hfq).expect("failed to read model config");
    println!(
        "config : dim={} layers={} heads={} kv_heads={} head_dim={} q_dim={} vocab={}",
        config.dim,
        config.n_layers,
        config.n_heads,
        config.n_kv_heads,
        config.head_dim,
        config.n_heads * config.head_dim,
        config.vocab_size,
    );

    let mut gpu = Gpu::init().expect("GPU init failed");
    println!("gpu    : arch={}", gpu.arch.as_str());
    println!(
        "batched: prefill_batched={} (MIN_BATCH gate: n={} needs >=4 for flash)",
        hipfire_runtime::config::get().prefill_batched,
        TOKENS.len(),
    );

    let weights = <Llama as Architecture>::load_weights(&mut hfq, &config, &mut gpu)
        .expect("failed to load weights");
    let scratch = <Llama as Architecture>::new_state(&mut gpu, &config).expect("new_state failed");

    // Primary A/B/C comparison over the full fixed sequence.
    let a = run_pertoken(&mut gpu, &weights, &config, &scratch, TOKENS);
    let b = run_prefill_forward(&mut gpu, &weights, &config, TOKENS);
    let c = run_flash(&mut gpu, &weights, &config, &scratch, TOKENS);

    println!("\n#### PRIMARY RESULT (n={}) ####", TOKENS.len());
    println!("A per-token argmax        = {}", argmax(&a));
    println!("B prefill_forward argmax  = {}", argmax(&b));
    println!("C flash argmax            = {}", argmax(&c));
    println!();
    report(
        "B (prefill_forward / attention_causal_batched) vs A (per-token)",
        &a,
        &b,
    );
    println!();
    report("C (forward_prefill_batch / FLASH) vs A (per-token)", &a, &c);

    // Optional narrowing: sweep prefix length for the flash path. n<MIN_BATCH(4)
    // falls back to per-token inside forward_prefill_batch, so those lengths
    // exercise the reference, not the flash kernel — reported for contrast.
    println!("\n#### FLASH-vs-per-token BY PREFIX LENGTH ####");
    println!("(n<4 => forward_prefill_batch internally falls back to per-token)");
    for n in [1usize, 2, 4, 6, 8] {
        if n > TOKENS.len() {
            continue;
        }
        let toks = &TOKENS[..n];
        let ref_a = run_pertoken(&mut gpu, &weights, &config, &scratch, toks);
        let flash = run_flash(&mut gpu, &weights, &config, &scratch, toks);
        println!(
            "  n={n}: argmax_match={} max_abs_diff={:.5} cosine={:.6} (A={} C={})",
            argmax(&ref_a) == argmax(&flash),
            max_abs_diff(&ref_a, &flash),
            cosine(&ref_a, &flash),
            argmax(&ref_a),
            argmax(&flash),
        );
    }

    println!("\n=== done ===");
}
