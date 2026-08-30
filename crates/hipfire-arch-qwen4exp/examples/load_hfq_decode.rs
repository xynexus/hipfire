// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! Load a qwen4_exp `.hfq` through the `Architecture` impl and decode from it.
//!
//! This is the daemon-facing path end to end: config out of the artifact's
//! metadata, weights off the HFQ one tensor at a time, per-layer state, and real
//! logits. Every earlier parity in this crate uses synthetic weights uploaded from
//! host arrays; nothing else exercises the LOADER.
//!
//!     cargo run --release -p hipfire-arch-qwen4exp --example load_hfq_decode -- model.hfq

use hipfire_arch_qwen4exp::arch::{load_ngram_table, HfqTensorReader, Qwen4Exp};
use hipfire_arch_qwen4exp::trunk_gpu::{decode_step, TensorReader, TrunkScratch, TrunkState};
use hipfire_rdna::Gpu;
use hipfire_runtime::arch::Architecture;
use hipfire_runtime::hfq::HfqFile;
use std::path::Path;

fn main() {
    let path = match std::env::args().nth(1) {
        Some(p) => p,
        None => {
            eprintln!("usage: load_hfq_decode <model.hfq>");
            std::process::exit(2);
        }
    };
    let mut hfq = match HfqFile::open(Path::new(&path)) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("load_hfq_decode: cannot open {path}: {e}");
            std::process::exit(1);
        }
    };

    let cfg = match Qwen4Exp::config_from_hfq(&hfq) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("load_hfq_decode: config: {e}");
            std::process::exit(1);
        }
    };
    let sparse = cfg.sparse_attention_layers().count();
    println!(
        "config: {} layers ({} sparse-attn / {} linear), hidden {}, vocab {}, \
         {} experts top-{}, hc {}, ngram {}",
        cfg.layers,
        sparse,
        cfg.layers - sparse,
        cfg.hidden,
        cfg.vocab,
        cfg.moe.num_experts,
        cfg.moe.experts_per_tok,
        cfg.gated_residual.count,
        if cfg.ngram.is_some() { "yes" } else { "no" },
    );

    let mut gpu = match Gpu::init() {
        Ok(g) => g,
        Err(e) => {
            println!("load_hfq_decode: no GPU ({e}) — skipped");
            return;
        }
    };
    let w = match Qwen4Exp::load_weights(&mut hfq, &cfg, &mut gpu) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("load_hfq_decode: load_weights: {e}");
            std::process::exit(1);
        }
    };
    let mut st = TrunkState::new(&mut gpu, &cfg, 64).unwrap();
    let mut sc = TrunkScratch::new(&mut gpu, &cfg, 64).unwrap();

    // The n-gram table and the embedding stay host-side; see `trunk_gpu`.
    let r = HfqTensorReader { hfq: &hfq };
    let embed = r.read("model.language_model.embed_tokens.weight").unwrap();
    let ngram = cfg
        .ngram
        .as_ref()
        .map(|_| load_ngram_table(&r, &cfg).unwrap());

    let eos = 2u32;
    let prompt: Vec<u32> = vec![3, 17, 42, 5, 9, 7, 61, 23];
    let mut history = Vec::new();
    let mut argmaxes = Vec::new();
    for (t, &tok) in prompt.iter().enumerate() {
        history.push(tok.min(cfg.vocab as u32 - 1));
        let logits = decode_step(
            &mut gpu,
            &cfg,
            &w,
            &mut st,
            &mut sc,
            &embed,
            ngram.as_deref(),
            &history,
            t,
            eos,
        )
        .unwrap();
        let (am, mx) = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, v)| (i, *v))
            .unwrap();
        assert!(
            logits.iter().all(|v| v.is_finite()),
            "non-finite logit at position {t} — the load produced garbage"
        );
        argmaxes.push(am);
        if t == prompt.len() - 1 {
            println!("last-position logits: argmax {am} (value {mx:.4}), all finite");
        }
    }
    // A model whose argmax never moves is usually a dead forward (zeroed weights,
    // a state that never advances), not a confident one.
    let distinct = argmaxes
        .iter()
        .collect::<std::collections::HashSet<_>>()
        .len();
    println!(
        "argmax over {} positions: {argmaxes:?} ({distinct} distinct)",
        prompt.len()
    );
    if distinct == 1 {
        println!("load_hfq_decode: WARNING — the argmax never moved; check the load");
    }
    println!("load_hfq_decode: OK");
}
