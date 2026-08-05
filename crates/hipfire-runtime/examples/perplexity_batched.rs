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
    let corpus_path = args
        .next()
        .expect("usage: perplexity_batched <model> <corpus> ...");

    let mut ctx_len: usize = 512;
    let mut warmup: usize = 8;
    let mut offset: usize = 0;
    let mut chunks: usize = 1;
    let mut kv_mode: String = "q8".to_string();
    let mut dump_ref: Option<String> = None;
    let mut kld_ref: Option<String> = None;
    let mut hfqm_ref: Option<String> = None;
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
            "--hfqm-ref" => hfqm_ref = Some(val),
            "--top-k" => top_k = val.parse().unwrap(),
            _ => panic!("unknown flag: {flag}"),
        }
    }
    // An HFQM kldref carries its OWN tokens, ctx, and scoring window — it drives
    // the run so the candidate is scored on exactly the reference's positions.
    let hfqm = hfqm_ref.as_ref().map(|p| read_hfqm_ref(p));
    if let Some(h) = hfqm.as_ref() {
        ctx_len = h.n_ctx;
        warmup = h.scoring_start;
        top_k = h.top_k;
        eprintln!(
            "HFQM ref: {} ({}), n_ctx={} scoring_start={} scored/chunk={} top_k={} n_chunk={} kv_mode(ref)={}",
            h.base_model_id, h.reference_precision, h.n_ctx, h.scoring_start,
            h.scored_per_chunk, h.top_k, h.n_chunk, h.ref_kv_mode
        );
    }
    assert!(
        ctx_len > warmup + 4,
        "ctx must exceed warmup by enough to score"
    );
    let act_bits =
        std::env::var("HIPFIRE_OQ4_PREFILL_ACT_BITS").unwrap_or_else(|_| "(default)".into());
    let clip = std::env::var("HIPFIRE_OQ4_ACT_CLIP").as_deref() == Ok("1");
    eprintln!("BATCHED prefill KLD — ACT_BITS={act_bits} clip={clip} kv={kv_mode} chunks={chunks} ctx={ctx_len}");
    if kv_mode == "f32" {
        eprintln!(
            "WARNING: f32 KV → per-token fallback (W4A16); use q8 for a real int4-act measurement."
        );
    }

    let mut hfq = HfqFile::open(Path::new(&model_path)).expect("open model");
    let config = qwen35::config_from_hfq(&hfq).expect("config");
    let all_tokens: Vec<u32> = if hfqm.is_some() {
        Vec::new() // the ref supplies the tokens, per chunk
    } else {
        let want_bytes = (offset + ctx_len * chunks) * 8;
        let raw = std::fs::read(&corpus_path).expect("read corpus");
        let take = want_bytes.min(raw.len());
        let corpus = String::from_utf8_lossy(&raw[..take]).to_string();
        let tokenizer =
            hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
                .expect("tokenizer");
        eprintln!("Tokenizing...");
        tokenizer.encode(&corpus)
    };

    let mut gpu = Gpu::init().expect("GPU init");
    eprintln!("GPU: {}  Loading {model_path}...", gpu.arch);
    let weights = qwen35::load_weights(&mut hfq, &config, &mut gpu).expect("load_weights");

    let dim = config.dim;
    let scratch = Qwen35Scratch::new(&mut gpu, &config, 64).unwrap();
    // Reused across chunks; sized for the max window. Prefill writes the first
    // n rows (post-output-norm), so the lm-head below must NOT re-norm.
    let hidden = gpu.alloc_tensor(&[ctx_len * dim], DType::F32).unwrap();

    let kld_records: Option<Vec<(f32, Vec<(u32, f32)>)>> = kld_ref.as_ref().map(|p| read_kldref(p));
    let mut ref_records: Vec<(f32, Vec<(u32, f32)>)> = Vec::new();
    let mut total_nll: f64 = 0.0;
    let mut scored: usize = 0;
    let mut total_kld: f64 = 0.0;
    let mut kld_scored: usize = 0;
    let mut chunk_klds: Vec<f64> = Vec::new();
    // Real-model PREFILL wall time (excludes the lm-head fan-out, which is a
    // harness artifact, not part of serving): the honest tok/s comparison
    // between activation precisions.
    let mut prefill_secs = 0.0f64;
    let mut prefill_toks = 0usize;
    let t0 = Instant::now();

    let mut done_chunks = 0usize;
    for c in 0..chunks {
        let window: Vec<u32> = if let Some(h) = hfqm.as_ref() {
            let ci = offset + c;
            if ci >= h.n_chunk {
                eprintln!("(ran out of reference chunks at {c}/{chunks})");
                break;
            }
            h.tokens[ci * h.n_ctx..(ci + 1) * h.n_ctx].to_vec()
        } else {
            let wstart = offset + c * ctx_len;
            let wend = (wstart + ctx_len).min(all_tokens.len());
            if wend <= wstart + warmup + 4 {
                eprintln!("(ran out of corpus at chunk {c}/{chunks})");
                break;
            }
            all_tokens[wstart..wend].to_vec()
        };
        let n = window.len();

        // Fresh state per chunk = independent teacher-forced sequence (start_pos=0),
        // matching the daemon kld_eval. new/drop each chunk (the daemon does too).
        let mut kv = make_kv(&mut gpu, &kv_mode, &config, n + 16);
        let mut dn = DeltaNetState::new(&mut gpu, &config).unwrap();
        let tp = Instant::now();
        qwen35::forward_prefill_batch_with_pbs_opts(
            &mut gpu,
            &weights,
            &config,
            &window,
            0,
            &mut kv,
            &mut dn,
            &scratch,
            None,
            Some(&hidden),
            None,
            None,
            None,
            None,
            None,
            false,
            false,
        )
        .expect("forward_prefill_batch");
        gpu.device_synchronize().unwrap();
        prefill_secs += tp.elapsed().as_secs_f64();
        prefill_toks += n;

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
                let topk: Vec<(u32, f32)> = red
                    .indices
                    .iter()
                    .map(|&i| (i, logits[i as usize]))
                    .collect();
                ref_records.push((lz as f32, topk));
            }
            if let Some(h) = hfqm.as_ref() {
                // Reference block index: chunk-major, scored positions only.
                let j = pos - h.scoring_start;
                if j < h.scored_per_chunk {
                    let ci = offset + c;
                    let b = (ci * h.scored_per_chunk + j) * h.top_k;
                    if std::env::var("HIPFIRE_REF_DIAG").is_ok() && j < 3 {
                        // Alignment check: the reference's argmax and the target
                        // token, next to ours. A block/position misalignment shows
                        // up here as a ref top-1 unrelated to what we predict.
                        let rtop = h.top_indices[b];
                        let rlp = h.top_log_probs[b];
                        let mut mine = 0usize;
                        for (i, v) in logits.iter().enumerate() {
                            if *v > logits[mine] {
                                mine = i;
                            }
                        }
                        eprintln!(
                            "[refdiag] chunk={ci} j={j} pos={pos} target={target} ref_top1={rtop} \
                             ref_lp={rlp:.4} my_top1={mine} resid={:.4}",
                            h.residual_mass[ci * h.scored_per_chunk + j]
                        );
                    }
                    let rb = RefBlock {
                        top_indices: &h.top_indices[b..b + h.top_k],
                        top_log_probs: &h.top_log_probs[b..b + h.top_k],
                        residual_mass: h.residual_mass[ci * h.scored_per_chunk + j],
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

    let avg_nll = if scored > 0 {
        total_nll / scored as f64
    } else {
        0.0
    };
    println!();
    println!("Model:    {model_path}");
    println!("Act bits: {act_bits}  clip: {clip}  kv: {kv_mode}");
    println!(
        "Chunks:   {done_chunks}  Scored:   {scored}  ({:.1}s total)",
        t0.elapsed().as_secs_f64()
    );
    println!(
        "PREFILL:  {prefill_toks} tok in {prefill_secs:.3}s = {:.1} tok/s  <-- real-model prefill throughput",
        prefill_toks as f64 / prefill_secs.max(1e-9)
    );
    println!("NLL/tok:  {:.10}", avg_nll);
    println!("PPL:      {:.4}", avg_nll.exp());
    if (kld_records.is_some() || hfqm.is_some()) && kld_scored > 0 {
        let mean_kld = total_kld / kld_scored as f64;
        let what = match hfqm.as_ref() {
            Some(h) => format!(
                "ABSOLUTE vs {} ({})",
                h.base_model_id, h.reference_precision
            ),
            None => "act-precision penalty".to_string(),
        };
        println!(
            "KLD/tok:  {:.6} (top-{top_k}, {kld_scored} pos)  <-- {what}",
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
            // In chunk order (not sorted) so two runs can be compared PAIRWISE.
            // min/max alone only show the distribution shifted; a per-chunk win
            // needs the same window on both sides.
            println!(
                "  per-chunk KLD (in order): {}",
                chunk_klds
                    .iter()
                    .map(|v| format!("{v:.6}"))
                    .collect::<Vec<_>>()
                    .join(" ")
            );
        }
    }
    if let Some(path) = dump_ref {
        write_kldref(&path, &ref_records, top_k);
        println!(
            "Wrote KLD reference: {path} ({} positions)",
            ref_records.len()
        );
    }
}

/// A decoded `*.kldref.hfq` (HFQM package, `hipfire.kldref.v1`) — the house
/// bf16 reference produced by `build_kld_ref_hipfire`. It carries its own token
/// stream, so scoring against it needs no corpus and no tokenizer: the candidate
/// runs on exactly the reference's windows and positions. That makes the KLD an
/// ABSOLUTE vs-bf16 number, not an act-precision delta.
struct HfqmRef {
    base_model_id: String,
    reference_precision: String,
    ref_kv_mode: String,
    n_ctx: usize,
    n_chunk: usize,
    scored_per_chunk: usize,
    scoring_start: usize,
    top_k: usize,
    tokens: Vec<u32>,
    top_indices: Vec<u32>,
    top_log_probs: Vec<f32>,
    residual_mass: Vec<f32>,
}

fn read_hfqm_ref(path: &str) -> HfqmRef {
    let package =
        hipfire_runtime::hfq::HfqPackage::open(Path::new(path)).expect("open HFQM kldref");
    let meta: serde_json::Value =
        serde_json::from_str(&package.metadata_json).expect("kldref metadata json");
    assert_eq!(
        meta.get("artifact_kind").and_then(|v| v.as_str()),
        Some("hipfire.kldref"),
        "not a hipfire.kldref package"
    );
    let usize_of = |k: &str| meta.get(k).and_then(|v| v.as_u64()).unwrap() as usize;
    let str_of = |k: &str| {
        meta.get(k)
            .and_then(|v| v.as_str())
            .unwrap_or("?")
            .to_string()
    };
    let n_ctx = usize_of("n_ctx");
    let n_chunk = usize_of("n_chunk");
    let scored_per_chunk = usize_of("scored_per_chunk");
    let top_k = usize_of("top_k");
    let blob = |name: &str| {
        package
            .blob_data(name)
            .unwrap_or_else(|| panic!("missing {name}"))
    };
    let u32s = |b: &[u8]| -> Vec<u32> {
        b.chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    };
    let f32s = |b: &[u8]| -> Vec<f32> {
        b.chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    };
    let tokens = u32s(blob("kldref.tokens"));
    let top_indices = u32s(blob("kldref.top_indices"));
    let top_log_probs = f32s(blob("kldref.top_log_probs"));
    let residual_mass = f32s(blob("kldref.residual_mass"));
    assert_eq!(tokens.len(), n_chunk * n_ctx, "kldref.tokens length");
    assert_eq!(
        top_indices.len(),
        n_chunk * scored_per_chunk * top_k,
        "kldref.top_indices length"
    );
    HfqmRef {
        base_model_id: str_of("base_model_id"),
        reference_precision: str_of("reference_precision"),
        ref_kv_mode: str_of("kv_mode"),
        n_ctx,
        n_chunk,
        scored_per_chunk,
        scoring_start: usize_of("scoring_start"),
        top_k,
        tokens,
        top_indices,
        top_log_probs,
        residual_mass,
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
