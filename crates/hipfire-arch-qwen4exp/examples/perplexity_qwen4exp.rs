// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! Teacher-forced perplexity for qwen4_exp, because the shared `perplexity`
//! example cannot load this family.
//!
//! That example goes through the qwen35 loader and dies on `tensor not found:
//! norm.weight` for an arch-26 artifact, so `hipfire eval --battery perplexity`
//! reports `exit status: 101` for every Qwen3.8-Flash-Next model. Perplexity is
//! the only ABSOLUTE quality number available here -- a KLD needs a
//! higher-precision reference, and no bf16 Flash-Next fits on a 124 GB box --
//! so without this there is no way to rank two quantisations of this model.
//!
//!     perplexity_qwen4exp <model.hfq> <corpus.txt> [n_tokens]
//!
//! USE A HELD-OUT CORPUS. Scoring the calibration corpus measures how well the
//! quantiser memorised its own calibration set (see the retraction in
//! `reference_moe_experts_imatrix_only`: a "-13.6% budget" result was a corpus
//! and a KLD reference that were the same file).

use hipfire_arch_qwen4exp::serving::Qwen4ExpBackend;
use hipfire_model::tokenizer::Tokenizer;
use hipfire_runtime::arch::SimpleAr;
use hipfire_runtime::hfq::HfqFile;
use std::path::Path;
use std::time::Instant;

fn main() {
    let mut args = std::env::args().skip(1);
    let model = args
        .next()
        .expect("usage: perplexity_qwen4exp <model.hfq> <corpus.txt> [n_tokens]");
    let corpus = args.next().expect("corpus path");
    let n_tok: usize = args.next().and_then(|v| v.parse().ok()).unwrap_or(1024);

    let mut hfq = HfqFile::open(Path::new(&model)).expect("open model");
    let mut gpu = match hipfire_rdna::Gpu::init() {
        Ok(g) => g,
        Err(e) => {
            println!("perplexity_qwen4exp: no GPU ({e:?}) — skipped");
            return;
        }
    };
    let max_seq = (n_tok + 8).next_power_of_two().max(256);
    let t0 = Instant::now();
    let mut m = Qwen4ExpBackend::load(&mut gpu, &mut hfq, max_seq).expect("load");
    println!("loaded in {:.1}s", t0.elapsed().as_secs_f32());

    let text = std::fs::read_to_string(&corpus).expect("read corpus");
    let tok = Tokenizer::from_hfq_metadata(&hfq.metadata_json).expect("tokenizer from hfq");
    let ids = tok.encode(&text);
    let n_use = n_tok.min(ids.len());
    assert!(n_use >= 2, "corpus tokenized to {n_use} tokens");
    let ids = &ids[..n_use];

    // Teacher-forced: score the TRUE next token at each position, never the
    // model's own argmax. Scoring a self-generated continuation measures the
    // model against itself and is not a perplexity at all.
    let t1 = Instant::now();
    m.prefill(&mut gpu, &ids[..1]).expect("prefill");
    let mut nll = 0.0f64;
    let mut scored = 0usize;
    for i in 1..ids.len() {
        if i > 1 {
            m.decode_step(&mut gpu, ids[i - 1], i - 1).expect("decode");
        }
        let lg = gpu.download_f32(m.trunk_logits()).expect("logits");
        let target = ids[i] as usize;
        assert!(
            target < lg.len(),
            "token {target} outside vocab {}",
            lg.len()
        );
        // log_softmax in f64 off a max-shifted f32 vector: the sum runs over
        // 248320 terms, and an f32 accumulator loses the tail that distinguishes
        // two close quantisations -- which is the entire point of this measurement.
        let max = lg.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let sum: f64 = lg.iter().map(|&v| ((v - max) as f64).exp()).sum();
        nll -= (lg[target] - max) as f64 - sum.ln();
        scored += 1;
    }
    let ppl = (nll / scored as f64).exp();
    println!(
        "scored {scored} tokens in {:.1}s",
        t1.elapsed().as_secs_f32()
    );
    println!("mean NLL {:.6}", nll / scored as f64);
    println!("PPL {ppl:.4}");
}
