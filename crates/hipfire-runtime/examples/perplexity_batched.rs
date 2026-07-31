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
//! = W4A16 always), this runs each chunk through ONE
//! `forward_prefill_batch_with_pbs_opts` call so the oq4 PREFILL kernels are
//! exercised, then fans out the lm-head over every scored position. That makes
//! `HIPFIRE_OQ4_PREFILL_ACT_BITS` (4 = W4A4 int4-act, 16 = W4A16 baseline) +
//! `HIPFIRE_OQ4_ACT_CLIP` selectable, so KLD(act16 ‖ act4) is the int4-activation
//! prefill quality penalty A4 needs.
//!
//! `--chunks N` scores N independent ctx-windows (fresh KV+DeltaNet state per
//! chunk, matching the daemon `kld_eval`), for a house-rule ≥16-chunk KLD with a
//! per-chunk spread. Scoring goes through the SAME `hipfire_kld` math as
//! `perplexity.rs`; .pkld files are cross-compatible (dump/score in the same
//! chunk order).
//!
//! CRITICAL: default `--kv-mode q8`. With f32 KV the batched path silently falls
//! back to per-token decode (act4 == act16 == W4A16 → KLD ≈ 0). Set
//! `HIPFIRE_DEBUG_PREFILL_ELIGIBLE=1` and confirm `final=true`.

use hipfire_arch_qwen35::qwen35::{self, DeltaNetState, Qwen35Scratch};
use hipfire_kld::math::{log_z, score_position, top_k_log_softmax};
use hipfire_kld::refblock::RefBlock;
use hipfire_rdna::{DType, Gpu};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::kv::KvCache;
use hipfire_runtime::weights::weight_gemv;
use std::path::Path;
use std::time::Instant;

fn make_kv(gpu: &mut Gpu, mode: &str, cfg: &qwen35::Qwen35Config, kv_max: usize) -> KvCache {
    match mode {
        "f32" | "fp16" => {
            KvCache::new_gpu(gpu, cfg.n_layers, cfg.n_kv_heads, cfg.head_dim, kv_max).unwrap()
        }
        "q8" => {
            KvCache::new_gpu_q8(gpu, cfg.n_layers, cfg.n_kv_heads, cfg.head_dim, kv_max).unwrap()
        }
        other => panic!("unknown --kv-mode: {other} (use q8 or f32)"),
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let model_path = args
        .next()
        .expect("usage: perplexity_batched <model> <corpus> [--ctx N] [--warmup N] [--offset N] [--chunks N] [--kv-mode q8|f32] [--dump-ref f] [--kld-ref f] [--top-k N]");
    let corpus_path = args.next().expect("usage: perplexity_batched <model> <corpus> ...");

    let mut ctx_len: usize = 512;
    let mut warmup: usize = 8;
    let mut offset: usize = 0;
    let mut chunks: usize = 1;
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
            "--chunks" => chunks = val.parse().unwrap(),
            "--kv-mode" => kv_mode = val,
            "--dump-ref" => dump_ref = Some(val),
            "--kld-ref" => kld_ref = Some(val),
            "--top-k" => top_k = val.parse().unwrap(),
            _ => panic!("unknown flag: {flag}"),
        }
    }
    assert!(ctx_len > warmup + 4, "ctx must exceed warmup by enough to score");
    let act_bits =
        std::env::var("HIPFIRE_OQ4_PREFILL_ACT_BITS").unwrap_or_else(|_| "(default)".into());
    let clip = std::env::var("HIPFIRE_OQ4_ACT_CLIP").as_deref() == Ok("1");
    eprintln!("BATCHED prefill KLD — ACT_BITS={act_bits} clip={clip} kv={kv_mode} chunks={chunks} ctx={ctx_len}");
    if kv_mode == "f32" {
        eprintln!("WARNING: f32 KV → per-token fallback (W4A16); use q8 for a real int4-act measurement.");
    }

    let want_bytes = (offset + ctx_len * chunks) * 8;
    let raw = std::fs::read(&corpus_path).expect("read corpus");
    let take = want_bytes.min(raw.len());
    let corpus = String::from_utf8_lossy(&raw[..take]).to_string();

    let mut hfq = HfqFile::open(Path::new(&model_path)).expect("open model");
    let config = qwen35::config_from_hfq(&hfq).expect("config");
    let tokenizer = hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
        .expect("tokenizer");
    eprintln!("Tokenizing...");
    let all_tokens: Vec<u32> = tokenizer.encode(&corpus);

    let mut gpu = Gpu::init().expect("GPU init");
    eprintln!("GPU: {}  Loading {model_path}...", gpu.arch);
    let weights = qwen35::load_weights(&mut hfq, &config, &mut gpu).expect("load_weights");

    let dim = config.dim;
    let scratch = Qwen35Scratch::new(&mut gpu, &config, 64).unwrap();
    // Reused across chunks; sized for the max window. Prefill writes the first
    // n rows (post-output-norm), so the lm-head below must NOT re-norm.
    let hidden = gpu.alloc_tensor(&[ctx_len * dim], DType::F32).unwrap();

    let kld_records: Option<Vec<(f32, Vec<(u32, f32)>)>> =
        kld_ref.as_ref().map(|p| read_kldref(p));
    let mut ref_records: Vec<(f32, Vec<(u32, f32)>)> = Vec::new();
    let mut total_nll: f64 = 0.0;
    let mut scored: usize = 0;
    let mut total_kld: f64 = 0.0;
    let mut kld_scored: usize = 0;
    let mut chunk_klds: Vec<f64> = Vec::new();
    let t0 = Instant::now();

    let mut done_chunks = 0usize;
    for c in 0..chunks {
        let wstart = offset + c * ctx_len;
        let wend = (wstart + ctx_len).min(all_tokens.len());
        if wend <= wstart + warmup + 4 {
            eprintln!("(ran out of corpus at chunk {c}/{chunks})");
            break;
        }
        let window: Vec<u32> = all_tokens[wstart..wend].to_vec();
        let n = window.len();

        // Fresh state per chunk = independent teacher-forced sequence (start_pos=0),
        // matching the daemon kld_eval. new/drop each chunk (the daemon does too).
        let mut kv = make_kv(&mut gpu, &kv_mode, &config, n + 16);
        let mut dn = DeltaNetState::new(&mut gpu, &config).unwrap();
        qwen35::forward_prefill_batch_with_pbs_opts(
            &mut gpu, &weights, &config, &window, 0, &mut kv, &mut dn, &scratch,
            None, Some(&hidden), None, None, None, None, None, false, false,
        )
        .expect("forward_prefill_batch");
        gpu.device_synchronize().unwrap();

        let mut chunk_kld_sum = 0.0f64;
        let mut chunk_kld_n = 0usize;
        for pos in warmup..n - 1 {
            let hrow = hidden.sub_offset(pos * dim, dim);
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
                let topk: Vec<(u32, f32)> =
                    red.indices.iter().map(|&i| (i, logits[i as usize])).collect();
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
                        chunk_kld_sum += ps.kld as f64;
                        chunk_kld_n += 1;
                    }
                }
            }
            scored += 1;
        }
        if chunk_kld_n > 0 {
            chunk_klds.push(chunk_kld_sum / chunk_kld_n as f64);
        }
        done_chunks += 1;
    }

    let avg_nll = if scored > 0 { total_nll / scored as f64 } else { 0.0 };
    println!();
    println!("Model:    {model_path}");
    println!("Act bits: {act_bits}  clip: {clip}  kv: {kv_mode}");
    println!(
        "Chunks:   {done_chunks}  Scored:   {scored}  ({:.1}s total)",
        t0.elapsed().as_secs_f64()
    );
    println!("NLL/tok:  {:.10}", avg_nll);
    println!("PPL:      {:.4}", avg_nll.exp());
    if kld_records.is_some() && kld_scored > 0 {
        let mean_kld = total_kld / kld_scored as f64;
        println!(
            "KLD/tok:  {:.6} (top-{top_k}, {kld_scored} pos)  <-- act-precision penalty",
            mean_kld
        );
        if chunk_klds.len() > 1 {
            let mut cs = chunk_klds.clone();
            cs.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let p99 = cs[((cs.len() as f64 * 0.99).ceil() as usize).min(cs.len()) - 1];
            println!(
                "  per-chunk KLD: min {:.6} / max {:.6} / p99 {:.6} over {} chunks",
                cs[0],
                cs[cs.len() - 1],
                p99,
                cs.len()
            );
        }
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
