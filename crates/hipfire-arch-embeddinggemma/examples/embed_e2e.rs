// hipfire example clippy sweep: examples are GPU probes, not reusable APIs.
#![allow(clippy::needless_range_loop)]

//! Standalone embedding driver for `hipfire-arch-embeddinggemma` (bring-up).
//!
//! Bypasses the daemon / `LoadedModel` seam so the bidirectional encoder forward
//! can be validated before the serving wiring lands. Loads an embeddinggemma
//! `.hfq`, encodes a handful of sentences, and prints per-embedding stats plus a
//! cosine-similarity matrix for a relative-quality sanity check (a semantically
//! close pair must outscore an unrelated pair).
//!
//! ```text
//! hipfire lock acquire embgemma && \
//! cargo run --release --example embed_e2e -p hipfire-arch-embeddinggemma -- \
//!     --hfq embeddinggemma-300m.bf16.hfq ; \
//! hipfire lock release
//! ```
//!
//! Exact-parity vs the HF `sentence_transformers` reference (cosine ≥ 0.99) is the
//! stronger gate; it needs `sentence_transformers` installed and is driven
//! separately (see docs/plans NPU/embeddings validation).

use std::path::Path;

use hipfire_arch_embeddinggemma as eg;
use hipfire_rdna::Gpu;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::tokenizer::Tokenizer;

/// amdgpu hwmon `power1_average` (SoC package power, µW) — the same rail the NPU
/// bench reads, for a fair GPU-vs-NPU tok/joule comparison.
fn amdgpu_power_path() -> Option<std::path::PathBuf> {
    for e in std::fs::read_dir("/sys/class/hwmon").ok()?.flatten() {
        let p = e.path();
        if std::fs::read_to_string(p.join("name"))
            .map(|s| s.trim() == "amdgpu")
            .unwrap_or(false)
        {
            let pw = p.join("power1_average");
            if pw.exists() {
                return Some(pw);
            }
        }
    }
    None
}
fn read_watts(path: &std::path::Path) -> Option<f64> {
    std::fs::read_to_string(path)
        .ok()?
        .trim()
        .parse::<f64>()
        .ok()
        .map(|uw: f64| uw / 1e6)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut hfq_path: Option<String> = None;
    let mut ref_hfq: Option<String> = None;
    let mut dims: Option<usize> = None;
    let mut bench_m: usize = 0;
    let mut bench_iters: usize = 50;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--hfq" => hfq_path = it.next(),
            "--dims" => dims = it.next().and_then(|s| s.parse().ok()),
            // --bench-m <M>: skip the sanity docs and instead time embed_forward on a
            // synthetic M-token sequence (the GPU tok/s baseline for the NPU comparison).
            "--bench-m" => bench_m = it.next().and_then(|s| s.parse().ok()).unwrap_or(0),
            "--bench-iters" => bench_iters = it.next().and_then(|s| s.parse().ok()).unwrap_or(50),
            // --ref-hfq <path>: quality gate. Encode the sanity docs with both the
            // primary and the reference model, print per-doc cosine(primary, ref) and
            // the mean; PASS if mean >= 0.99. Used to check int4/W4A8 vs bf16.
            "--ref-hfq" => ref_hfq = it.next(),
            "-h" | "--help" => {
                eprintln!("usage: embed_e2e --hfq <path.hfq> [--dims N] [--bench-m M --bench-iters N] [--ref-hfq <bf16.hfq>]");
                return Ok(());
            }
            other => return Err(format!("unknown arg: {other}").into()),
        }
    }
    let hfq_path = hfq_path.ok_or("--hfq is required")?;

    eprintln!("[1/4] opening HFQ: {hfq_path}");
    let mut hfq = HfqFile::open(Path::new(&hfq_path))?;
    eprintln!("      arch_id (header) = {}", hfq.arch_id);
    if hfq.arch_id != 19 {
        eprintln!(
            "      warning: arch_id={} (embeddinggemma expects 19)",
            hfq.arch_id
        );
    }

    eprintln!("[2/4] parsing EmbeddingGemmaConfig");
    let cfg = eg::config_from_metadata_json(&hfq.metadata_json)
        .ok_or("embeddinggemma: failed to parse config")?;
    eprintln!(
        "      hidden={} layers={} n_heads={} n_kv={} head_dim={} vocab={}\n\
               bidirectional={} pooling={:?} dense_heads={} embed_dim={} \
         matryoshka={:?} norm_offset={}",
        cfg.hidden_size,
        cfg.num_hidden_layers,
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        cfg.head_dim,
        cfg.vocab_size,
        cfg.bidirectional,
        cfg.pooling_mode,
        cfg.dense_heads.len(),
        cfg.embedding_dim,
        cfg.matryoshka_dims,
        cfg.gemma_norm_offset,
    );
    if cfg.gemma_norm_offset == 0.0 {
        eprintln!(
            "      WARNING: gemma_norm_offset=0 — the (1+w) RMSNorm bake is missing; \
             re-ingest with the embeddinggemma importer or output will be wrong."
        );
    }
    let out_dims = cfg.resolve_dims(dims);

    eprintln!("[3/4] building tokenizer");
    let tok = Tokenizer::from_hfq_metadata(&hfq.metadata_json)
        .map_err(|e| format!("embeddinggemma: tokenizer not found: {e}"))?;
    eprintln!("      vocab_size={}", tok.vocab_size());

    eprintln!("[4/4] loading weights + Gpu");
    let mut gpu = Gpu::init()?;
    let weights = eg::EmbeddingGemmaWeights::load(&mut hfq, &cfg, &mut gpu)?;

    // Quality gate: encode the sanity docs with the primary model and a reference
    // model, and report per-doc cosine(primary, ref). Used to check W4A8 (int4) vs
    // the bf16 baseline — embeddings are unit-norm, so cosine == dot product.
    if let Some(ref_path) = &ref_hfq {
        let docs = [
            "The cat sat on the warm windowsill in the afternoon sun.",
            "A feline rested by the sunny window during the day.",
            "Quarterly revenue grew twelve percent driven by cloud services.",
        ];
        let mut prim: Vec<Vec<f32>> = Vec::new();
        for text in &docs {
            let ids = tok.encode(&format!("{}{}", cfg.document_prompt, text));
            prim.push(eg::embed_forward(&mut gpu, &weights, &cfg, &ids)?);
        }
        weights.free_gpu(&mut gpu);

        let mut rhfq = HfqFile::open(Path::new(ref_path))?;
        let rcfg = eg::config_from_metadata_json(&rhfq.metadata_json)
            .ok_or("ref: failed to parse config")?;
        let rtok = Tokenizer::from_hfq_metadata(&rhfq.metadata_json)
            .map_err(|e| format!("ref tokenizer: {e}"))?;
        let rweights = eg::EmbeddingGemmaWeights::load(&mut rhfq, &rcfg, &mut gpu)?;

        let cos = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
        let mut sum = 0.0f32;
        println!("=== W4A8 quality gate: cosine(primary, ref) per doc ===");
        for (i, text) in docs.iter().enumerate() {
            let ids = rtok.encode(&format!("{}{}", rcfg.document_prompt, text));
            let re = eg::embed_forward(&mut gpu, &rweights, &rcfg, &ids)?;
            let c = cos(&prim[i], &re);
            sum += c;
            println!("  doc[{i}] cosine = {c:.5}");
        }
        let mean = sum / docs.len() as f32;
        println!(
            "mean cosine = {mean:.5}  =>  {}",
            if mean >= 0.99 {
                "PASS (>=0.99)"
            } else {
                "FAIL (<0.99)"
            }
        );
        return Ok(());
    }

    // GPU throughput baseline: time embed_forward on a synthetic M-token sequence,
    // reporting steady-state tok/s (sample SoC package power externally for tok/J).
    if bench_m > 0 {
        // Real token ids (avoid pad/eos); cycle a small ascii-ish range.
        let toks: Vec<u32> = (0..bench_m).map(|i| 100 + (i as u32 % 2000)).collect();
        let pw_path = amdgpu_power_path();
        // idle baseline
        let mut idle_w = f64::NAN;
        if let Some(p) = &pw_path {
            let (mut s, mut n) = (0.0, 0.0);
            for _ in 0..15 {
                if let Some(w) = read_watts(p) {
                    s += w;
                    n += 1.0;
                }
                std::thread::sleep(std::time::Duration::from_millis(40));
            }
            if n > 0.0 {
                idle_w = s / n;
            }
        }
        for _ in 0..3 {
            let _ = eg::embed_forward(&mut gpu, &weights, &cfg, &toks)?; // warm
        }
        let (mut pw_sum, mut pw_n, mut pw_peak) = (0.0f64, 0.0f64, 0.0f64);
        let t0 = std::time::Instant::now();
        for _ in 0..bench_iters {
            let _ = eg::embed_forward(&mut gpu, &weights, &cfg, &toks)?;
            if let Some(p) = &pw_path {
                if let Some(w) = read_watts(p) {
                    pw_sum += w;
                    pw_n += 1.0;
                    pw_peak = pw_peak.max(w);
                }
            }
        }
        let dt = t0.elapsed().as_secs_f64();
        let tok_s = (bench_m * bench_iters) as f64 / dt;
        let pkg_w = if pw_n > 0.0 { pw_sum / pw_n } else { f64::NAN };
        let dyn_w = pkg_w - idle_w;
        eprintln!(
            "[gpu-bench] m={bench_m} iters={bench_iters}  {:.1} ms/encode  => {tok_s:.0} tok/s",
            dt / bench_iters as f64 * 1e3
        );
        eprintln!(
            "  SoC package power: idle={idle_w:.2} W  active={pkg_w:.2} W  peak={pw_peak:.2} W  (GPU-dynamic ≈ {dyn_w:.2} W)"
        );
        eprintln!(
            "  efficiency (pkg)= {:.0} tok/joule   (dyn)= {:.0} tok/joule",
            tok_s / pkg_w,
            tok_s / dyn_w
        );
        println!(
            "gpu_tok_s={tok_s:.6} idle_w={idle_w:.6} pkg_w={pkg_w:.6} dyn_w={dyn_w:.6} pkg_tok_j={:.6} dyn_tok_j={:.6}",
            tok_s / pkg_w,
            tok_s / dyn_w
        );
        return Ok(());
    }

    // Two semantically-close sentences + one unrelated; encoded as documents.
    let docs = [
        "The cat sat on the warm windowsill in the afternoon sun.",
        "A feline rested by the sunny window during the day.",
        "Quarterly revenue grew twelve percent driven by cloud services.",
    ];
    let mut embs: Vec<Vec<f32>> = Vec::new();
    for (i, text) in docs.iter().enumerate() {
        let prompt = format!("{}{}", cfg.document_prompt, text);
        let ids = tok.encode(&prompt);
        let t0 = std::time::Instant::now();
        let mut e = eg::embed_forward(&mut gpu, &weights, &cfg, &ids)?;
        eg::forward::l2_normalize(&mut e); // already normed; harmless
                                           // Matryoshka truncation for display.
        if out_dims < e.len() {
            e.truncate(out_dims);
            let n = e.iter().map(|x| x * x).sum::<f32>().sqrt();
            if n > 0.0 {
                for x in &mut e {
                    *x /= n;
                }
            }
        }
        let norm = e.iter().map(|x| x * x).sum::<f32>().sqrt();
        eprintln!(
            "  doc[{i}] tokens={} dim={} L2={:.4} in {} ms | first4={:?}",
            ids.len(),
            e.len(),
            norm,
            t0.elapsed().as_millis(),
            &e[..e.len().min(4)],
        );
        embs.push(e);
    }

    let cos = |a: &[f32], b: &[f32]| -> f32 { a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>() };
    println!("\n=== cosine similarity matrix (dim={out_dims}) ===");
    for i in 0..embs.len() {
        let row: Vec<String> = (0..embs.len())
            .map(|j| format!("{:.4}", cos(&embs[i], &embs[j])))
            .collect();
        println!("  doc[{i}] {}", row.join("  "));
    }
    let close = cos(&embs[0], &embs[1]);
    let far = cos(&embs[0], &embs[2]);
    println!("\nclose(0,1)={close:.4}  far(0,2)={far:.4}");
    if close > far {
        println!("SANITY PASS: semantically-close pair outscores the unrelated pair.");
    } else {
        println!("SANITY FAIL: close pair did not outscore the unrelated pair — investigate.");
    }
    Ok(())
}
