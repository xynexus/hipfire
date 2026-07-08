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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut hfq_path: Option<String> = None;
    let mut dims: Option<usize> = None;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--hfq" => hfq_path = it.next(),
            "--dims" => dims = it.next().and_then(|s| s.parse().ok()),
            "-h" | "--help" => {
                eprintln!("usage: embed_e2e --hfq <path.hfq> [--dims N]");
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
        eprintln!("      warning: arch_id={} (embeddinggemma expects 19)", hfq.arch_id);
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

    let cos = |a: &[f32], b: &[f32]| -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>()
    };
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
