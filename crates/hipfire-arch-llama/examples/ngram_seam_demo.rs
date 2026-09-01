// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! End-to-end proof of the `Speculator` seam: `NgramSpeculator` driving a real
//! `LlamaBackend` target, checked against plain AR over the same weights.
//!
//! Spec decode is LOSSLESS by construction — the target verifies every token —
//! so the bar is **token-for-token identity with AR**, not a quality score. A
//! speculative loop that is merely "close" is broken.
//!
//! The two runs share one loaded model and one KV cache; the spec run starts by
//! calling `spec_advance(reset = true)`, which zeroes recurrent + KV state, so
//! it begins from the same place AR did.
//!
//! Usage: ngram_seam_demo <model.hfq> [n_tokens] [prompt_tokens...]

use hipfire_arch_llama::{Llama, LlamaBackend};
use hipfire_runtime::arch::{Architecture, SimpleAr};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::kv::KvCache;
use hipfire_specdecode_dspark::ngram_speculator::NgramSpeculator;
use hipfire_specdecode_dspark::spec::{PrefillOutcome, SpecTarget, Speculator};
use hipfire_specdecode_ngram::{NgramConfig, NgramSpec, WriteTarget};

fn argmax(v: &[f32]) -> u32 {
    let mut best = 0usize;
    for (i, x) in v.iter().enumerate() {
        if *x > v[best] {
            best = i;
        }
    }
    best as u32
}

fn main() {
    let mut args = std::env::args().skip(1);
    let model_path = args
        .next()
        .expect("usage: ngram_seam_demo <model.hfq> [n] [toks...]");
    let n_tokens: usize = args.next().and_then(|a| a.parse().ok()).unwrap_or(48);
    let mut prompt: Vec<u32> = args.filter_map(|a| a.parse().ok()).collect();
    if prompt.is_empty() {
        // A deliberately REPETITIVE prompt. The n-gram tables are built from
        // this run's own history, so a prompt with no repeated structure gives
        // the drafter nothing to hit and the run degenerates to AR — which
        // would still pass the identity check while proving nothing about
        // acceptance. The `proposed` counter below is what catches that.
        prompt = vec![
            1, 450, 4996, 17354, 1701, 29916, 432, 17204, 975, 278, 17366, 11203, 29889, 450, 4996,
            17354, 1701, 29916, 432, 17204, 975, 278, 17366, 11203, 29889, 450, 4996, 17354, 1701,
            29916, 432, 17204, 975, 278,
        ];
    }

    let mut hfq = HfqFile::open(std::path::Path::new(&model_path)).expect("open model");
    let config = Llama::config_from_hfq(&hfq).expect("config");
    let mut gpu = hipfire_rdna::Gpu::init().expect("GPU init");
    eprintln!("gpu:    {}", gpu.arch);
    eprintln!("model:  {model_path}");
    let weights = Llama::load_weights(&mut hfq, &config, &mut gpu).expect("load_weights");
    let kv_max = prompt.len() + n_tokens + 32;
    let kv_cache = KvCache::new_gpu_q8(
        &mut gpu,
        config.n_layers,
        config.n_kv_heads,
        config.head_dim,
        kv_max,
    )
    .expect("kv_cache");
    let scratch = Llama::new_state(&mut gpu, &config).expect("state");
    let mut backend = LlamaBackend::new(hfq.arch_id, config, weights, scratch, kv_cache);

    // ── Arm 1: plain AR, greedy ─────────────────────────────────────────────
    let mut ar: Vec<u32> = Vec::with_capacity(n_tokens);
    {
        let mut last = 0u32;
        for (pos, &tok) in prompt.iter().enumerate() {
            backend.decode_step(&mut gpu, tok, pos).expect("prefill");
            last = argmax(&gpu.download_f32(backend.logits()).unwrap());
        }
        for i in 0..n_tokens {
            ar.push(last);
            let pos = prompt.len() + i;
            backend.decode_step(&mut gpu, last, pos).expect("decode");
            last = argmax(&gpu.download_f32(backend.logits()).unwrap());
        }
    }
    eprintln!("AR:     {} tokens", ar.len());

    // ── Arm 2: the seam ─────────────────────────────────────────────────────
    let cfg = NgramConfig {
        write_target: WriteTarget::None, // hot tier only — no store attached in this demo
        ..Default::default()
    };
    let max_spine = cfg.max_spine;
    let mut spec = NgramSpeculator::new(NgramSpec::new(cfg), max_spine, backend.ctx_capacity());

    let target: &mut dyn SpecTarget = &mut backend;
    let abort = || false;
    let first = match spec
        .prefill(&mut gpu, target, &prompt, &prompt, 0, false, None, &abort)
        .expect("prefill")
    {
        PrefillOutcome::Ready { first_token } => first_token,
        PrefillOutcome::Aborted => panic!("prefill aborted"),
    };

    let mut got: Vec<u32> = vec![first];
    let mut seed = first;
    let mut position = prompt.len();
    let mut proposed = 0usize;
    let mut accepted = 0usize;
    let mut windows = 0usize;
    while got.len() < n_tokens {
        let step = spec
            .step(&mut gpu, target, position, seed, &got, None, 0.0)
            .expect("step");
        assert!(
            !step.emit.is_empty(),
            "a window emitted nothing — the loop would stall"
        );
        proposed += step.proposed;
        accepted += step.accepted;
        windows += 1;
        position += step.emit.len();
        seed = step.next_seed;
        got.extend(step.emit.iter().copied());
    }
    got.truncate(n_tokens);

    let tau = if windows > 0 {
        got.len() as f64 / windows as f64
    } else {
        0.0
    };
    eprintln!(
        "seam:   {} tokens in {windows} windows  proposed={proposed} accepted={accepted}  tau={tau:.3}",
        got.len()
    );

    // ── The bar ─────────────────────────────────────────────────────────────
    if proposed == 0 {
        eprintln!(
            "FAIL: the drafter never proposed anything, so the identity below is \
             vacuous — it only proves AR equals AR. Use a prompt with repeated \
             structure, or check the n-gram config."
        );
        std::process::exit(1);
    }
    if got != ar {
        let at = got.iter().zip(&ar).position(|(a, b)| a != b);
        eprintln!("FAIL: spec stream != AR stream (first difference at {at:?})");
        eprintln!("  ar:   {:?}", &ar[..ar.len().min(24)]);
        eprintln!("  seam: {:?}", &got[..got.len().min(24)]);
        std::process::exit(1);
    }
    println!("ngram_seam_demo: OK — {} tokens identical to AR, {proposed} drafts proposed, {accepted} accepted", got.len());
}
