#![allow(
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::type_complexity
)]
// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! BATCHED-prefill perplexity / KLD eval for qwen3.5 — the A4 int4-act harness.
//!
//! Unlike `perplexity.rs` (which forwards token-by-token through the DECODE path
//! = W4A16 always), this runs the whole window through ONE
//! `forward_prefill_batch_with_pbs_opts` call so the oq4 PREFILL kernels are
//! exercised, then fans out the lm-head over every scored position. That makes
//! `HIPFIRE_OQ4_PREFILL_ACT_BITS` (4 = W4A4 int4-act, 16 = W4A16 baseline) the
//! selectable activation precision — so KLD(act16 ‖ act4) is the int4-activation
//! prefill quality penalty A4 needs.
//!
//! Scoring goes through the SAME `hipfire_kld` math as `perplexity.rs` and the
//! daemon `kld_eval`, so the .pkld files are cross-compatible: build a reference
//! with per-token `perplexity.rs --dump-ref` to validate that act16 here matches
//! the W4A16 decode path (KLD ≈ batched-vs-per-token noise), then
//! act16 `--dump-ref` + act4 `--kld-ref` for the penalty.
//!
//! CRITICAL: default `--kv-mode q8`. With f32 KV the batched path silently falls
//! back to per-token decode (act4 == act16 == W4A16 → KLD ≈ 0 and you learn
//! nothing). Set `HIPFIRE_DEBUG_PREFILL_ELIGIBLE=1` and confirm `final=true`.
//!
//! Usage:
//!   HIPFIRE_OQ4_PREFILL_ACT_BITS=16 perplexity_batched <model> <corpus> --ctx 512 --dump-ref a16.pkld
//!   HIPFIRE_OQ4_PREFILL_ACT_BITS=4  perplexity_batched <model> <corpus> --ctx 512 --kld-ref a16.pkld

use hipfire_arch_qwen35::qwen35::{self, DeltaNetState, Qwen35Scratch};
use hipfire_kld::math::{log_z, score_position, top_k_log_softmax};
use hipfire_kld::refblock::RefBlock;
use hipfire_rdna::{DType, Gpu};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::kv::KvCache;
use hipfire_runtime::weights::weight_gemv;
use std::path::Path;
use std::time::Instant;

fn main() {
    let mut args = std::env::args().skip(1);
    let model_path = args
        .next()
        .expect("usage: perplexity_batched <model> <corpus> [--ctx N] [--warmup N] [--offset N] [--kv-mode q8|f32] [--dump-ref f] [--kld-ref f] [--top-k N]");
    let corpus_path = args.next().expect("usage: perplexity_batched <model> <corpus> ...");

    let mut ctx_len: usize = 512;
    let mut warmup: usize = 8;
    let mut offset: usize = 0;
    let mut kv_mode: String = "q8".to_string();
    let mut dump_ref: Option<String> = None;
    let mut kld_ref: Option<String> = None;
    let mut top_k: usize = 128;
    while let Some(flag) = args.next() {
        let val = args.next().expect("flag missing value");
        match flag.as_str() {
            "--ctx" => ctx_len = val.parse().unwrap(),
            "--warmup" => warmup = val.parse().unwrap(),
            "--offset" => offset = val.parse().unwrap(),
            "--kv-mode" => kv_mode = val,
            "--dump-ref" => dump_ref = Some(val),
            "--kld-ref" => kld_ref = Some(val),
            "--top-k" => top_k = val.parse().unwrap(),
            _ => panic!("unknown flag: {flag}"),
        }
    }
    assert!(ctx_len > warmup + 4, "ctx must exceed warmup by enough to score");
    let act_bits = std::env::var("HIPFIRE_OQ4_PREFILL_ACT_BITS").unwrap_or_else(|_| "(default)".into());
    eprintln!("BATCHED prefill KLD — HIPFIRE_OQ4_PREFILL_ACT_BITS={act_bits}  kv-mode={kv_mode}");
    if kv_mode == "f32" {
        eprintln!("WARNING: f32 KV → batched path falls back to per-token decode (W4A16); use q8 for a real int4-act measurement.");
    }

    let want_bytes = (offset + ctx_len) * 8;
    let raw = std::fs::read(&corpus_path).expect("read corpus");
    let take = want_bytes.min(raw.len());
    let corpus = String::from_utf8_lossy(&raw[..take]).to_string();

    let mut hfq = HfqFile::open(Path::new(&model_path)).expect("open model");
    let config = qwen35::config_from_hfq(&hfq).expect("config");
    let tokenizer = hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
        .expect("tokenizer");
    eprintln!("Tokenizing...");
    let all_tokens: Vec<u32> = tokenizer.encode(&corpus);
    let end = (offset + ctx_len).min(all_tokens.len());
    if end <= offset + warmup + 4 {
        panic!("not enough tokens past offset={offset} for warmup={warmup} + scoring");
    }
    let window: Vec<u32> = all_tokens[offset..end].to_vec();
    let n = window.len();
    eprintln!("Window: offset={offset} ctx={n} (warmup {warmup}, scoring {})", n - warmup - 1);

    let mut gpu = Gpu::init().expect("GPU init");
    eprintln!("GPU: {}", gpu.arch);
    eprintln!("Loading weights from {model_path}...");
    let weights = qwen35::load_weights(&mut hfq, &config, &mut gpu).expect("load_weights");

    let kv_max = n + 16;
    let mut kv_cache = match kv_mode.as_str() {
        "f32" | "fp16" => {
            KvCache::new_gpu(&mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_max)
                .unwrap()
        }
        "q8" => {
            KvCache::new_gpu_q8(&mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, kv_max)
                .unwrap()
        }
        other => panic!("unknown --kv-mode: {other} (use q8 or f32)"),
    };
    let mut dn_state = DeltaNetState::new(&mut gpu, &config).unwrap();
    let scratch = Qwen35Scratch::new(&mut gpu, &config, 64).unwrap();

    // Per-position POST-output-norm hidden [n × dim] (per_token_hidden_out). The
    // prefill runs the final RMSNorm over ALL rows into this buffer, so the
    // lm-head below must NOT re-norm.
    let dim = config.dim;
    let hidden = gpu.alloc_tensor(&[n * dim], DType::F32).unwrap();

    let t0 = Instant::now();
    // ONE batched forward over the whole window (teacher-forced, start_pos=0).
    qwen35::forward_prefill_batch_with_pbs_opts(
        &mut gpu,
        &weights,
        &config,
        &window,
        0,
        &mut kv_cache,
        &mut dn_state,
        &scratch,
        None,           // hidden_rb
        Some(&hidden),  // per_token_hidden_out — fills all n rows post-norm
        None,           // gdn_tape
        None,           // tree_verify
        None,           // pbs_in (self-allocated)
        None,           // mask_override
        None,           // max_layer
        false,          // needs_last_token_logits (we fan out the lm-head ourselves)
        false,          // force_q8_gdn_per_token
    )
    .expect("forward_prefill_batch");
    gpu.device_synchronize().unwrap();
    eprintln!("Batched prefill: {n} tok in {:.2}s", t0.elapsed().as_secs_f64());

    // Fan out the lm-head over scored positions [warmup, n-1), scoring next token.
    let mut total_nll: f64 = 0.0;
    let mut scored: usize = 0;
    let mut ref_records: Vec<(f32, Vec<(u32, f32)>)> = Vec::new();
    let kld_records: Option<Vec<(f32, Vec<(u32, f32)>)>> = kld_ref.as_ref().map(|p| read_kldref(p));
    let mut total_kld: f64 = 0.0;
    let mut kld_scored: usize = 0;
    let ts = Instant::now();

    for pos in warmup..n - 1 {
        let hrow = hidden.sub_offset(pos * dim, dim);
        // decode-precision lm-head (env-independent) so the KLD isolates the BODY,
        // not lm-head act-quant error.
        weight_gemv(&mut gpu, &weights.output, &hrow, &scratch.logits).expect("lm-head");
        let logits = gpu.download_f32(&scratch.logits).unwrap();
        let target = window[pos + 1] as usize;
        let lz = log_z(&logits);
        if target >= logits.len() {
            continue;
        }
        let nll = (lz - logits[target] as f64) as f32;
        if !nll.is_finite() {
            continue;
        }
        total_nll += nll as f64;

        if dump_ref.is_some() {
            let red = top_k_log_softmax(&logits, top_k);
            let topk: Vec<(u32, f32)> = red.indices.iter().map(|&i| (i, logits[i as usize])).collect();
            ref_records.push((lz as f32, topk));
        }
        if let Some(ref recs) = kld_records {
            if scored < recs.len() {
                let (ref_logz, ref_topk) = &recs[scored];
                let idxs: Vec<u32> = ref_topk.iter().map(|&(i, _)| i).collect();
                let lps: Vec<f32> = ref_topk.iter().map(|&(_, lg)| lg - ref_logz).collect();
                let mut p_sum = 0.0f64;
                for &lp in &lps {
                    p_sum += (lp as f64).exp();
                }
                let residual = (1.0 - p_sum).max(0.0) as f32;
                let rb = RefBlock {
                    top_indices: &idxs,
                    top_log_probs: &lps,
                    residual_mass: residual,
                };
                let ps = score_position(&rb, &logits, target);
                if ps.kld.is_finite() {
                    total_kld += ps.kld as f64;
                    kld_scored += 1;
                }
            }
        }
        scored += 1;
    }

    let avg_nll = if scored > 0 { total_nll / scored as f64 } else { 0.0 };
    println!();
    println!("Model:    {model_path}");
    println!("Act bits: {act_bits}   kv-mode: {kv_mode}");
    println!("Scored:   {scored}  (lm-head fan-out in {:.1}s)", ts.elapsed().as_secs_f64());
    println!("NLL/tok:  {:.10}", avg_nll);
    println!("PPL:      {:.4}", avg_nll.exp());
    if kld_records.is_some() && kld_scored > 0 {
        let mean_kld = total_kld / kld_scored as f64;
        println!("KLD/tok:  {:.6} (top-{top_k}, {kld_scored} pos)  <-- act-precision penalty", mean_kld);
    }
    if let Some(path) = dump_ref {
        write_kldref(&path, &ref_records, top_k);
        println!("Wrote KLD reference: {path} ({} positions)", ref_records.len());
    }
}

/// KLD-reference file: magic "PKLD", u32 top_k, u64 n_pos, then per position:
/// f32 logZ, u32 n, n×(u32 idx, f32 logit). Byte-compatible with perplexity.rs.
fn write_kldref(path: &str, records: &[(f32, Vec<(u32, f32)>)], top_k: usize) {
    let mut buf: Vec<u8> = Vec::new();
    buf.extend_from_slice(b"PKLD");
    buf.extend_from_slice(&(top_k as u32).to_le_bytes());
    buf.extend_from_slice(&(records.len() as u64).to_le_bytes());
    for (logz, topk) in records {
        buf.extend_from_slice(&logz.to_le_bytes());
        buf.extend_from_slice(&(topk.len() as u32).to_le_bytes());
        for &(idx, logit) in topk {
            buf.extend_from_slice(&idx.to_le_bytes());
            buf.extend_from_slice(&logit.to_le_bytes());
        }
    }
    std::fs::write(path, buf).expect("write kldref");
}

fn read_kldref(path: &str) -> Vec<(f32, Vec<(u32, f32)>)> {
    let b = std::fs::read(path).expect("read kldref");
    assert_eq!(&b[0..4], b"PKLD", "bad kldref magic");
    let n_pos = u64::from_le_bytes(b[8..16].try_into().unwrap()) as usize;
    let mut off = 16;
    let rd_f32 = |b: &[u8], o: usize| f32::from_le_bytes(b[o..o + 4].try_into().unwrap());
    let rd_u32 = |b: &[u8], o: usize| u32::from_le_bytes(b[o..o + 4].try_into().unwrap());
    let mut out = Vec::with_capacity(n_pos);
    for _ in 0..n_pos {
        let logz = rd_f32(&b, off);
        off += 4;
        let nn = rd_u32(&b, off) as usize;
        off += 4;
        let mut topk = Vec::with_capacity(nn);
        for _ in 0..nn {
            let idx = rd_u32(&b, off);
            off += 4;
            let logit = rd_f32(&b, off);
            off += 4;
            topk.push((idx, logit));
        }
        out.push((logz, topk));
    }
    out
}
