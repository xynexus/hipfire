// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! Load and decode from a REAL Qwen3.8-Flash-Next artifact, once.
//!
//! `serve_fixture` loads the model three times to difference the resident and
//! streamed n-gram paths against each other. That is right for a few-MB fixture
//! and wrong for a 170 GB one, so this does a single load and reports what
//! actually became resident.
//!
//!     cargo run --release -p hipfire-arch-qwen4exp --example serve_real -- model.hfq

use hipfire_arch_qwen4exp::serving::Qwen4ExpBackend;
use hipfire_runtime::arch::SimpleAr;
use hipfire_runtime::hfq::HfqFile;
use std::path::Path;
use std::time::Instant;

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: serve_real <model.hfq>");
    let mut hfq = HfqFile::open(Path::new(&path)).expect("open artifact");
    println!(
        "artifact: arch {}, {} tensors",
        hfq.arch_id,
        hfq.tensors().len()
    );

    let mut gpu = match hipfire_rdna::Gpu::init() {
        Ok(g) => g,
        Err(e) => {
            println!("serve_real: no GPU ({e:?}) — skipped");
            return;
        }
    };

    let t0 = Instant::now();
    let mut m = match Qwen4ExpBackend::load(&mut gpu, &mut hfq, 256) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("serve_real: load FAILED: {e}");
            std::process::exit(1);
        }
    };
    println!("loaded in {:.1}s", t0.elapsed().as_secs_f32());
    println!("  routed experts resident as {:?}", m.routed_expert_dtype());
    println!(
        "  vocab {}, eos {}",
        m.vocab_size(),
        m.config().eos_token_id
    );

    // A short prompt: prefill is per-token, and this model has 48 layers.
    // Length overridable (argv[3]) so prefill SCALING can be measured: this
    // trunk prefills one token at a time, so cost should be strictly linear and
    // the slope is what a batched forward would attack.
    let prompt_len: usize = std::env::args()
        .nth(3)
        .and_then(|v| v.parse().ok())
        .unwrap_or(4);
    let prompt: Vec<u32> = (0..prompt_len)
        .map(|i| 9707u32 + (i as u32 % 977))
        .collect();
    let t1 = Instant::now();
    m.prefill(&mut gpu, &prompt).expect("prefill");
    let logits = gpu.download_f32(m.logits()).expect("download");
    println!(
        "prefill {} tokens in {:.2}s",
        prompt.len(),
        t1.elapsed().as_secs_f32()
    );
    assert_eq!(logits.len(), m.vocab_size());
    let finite = logits.iter().all(|v| v.is_finite());
    let (am, mx) =
        logits
            .iter()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &v)| {
                if v > bv {
                    (i, v)
                } else {
                    (bi, bv)
                }
            });
    println!("  argmax {am} ({mx:.4}), all finite: {finite}");
    assert!(finite, "logits must be finite");

    let mut tok = am as u32;
    let mut moved = false;
    let t2 = Instant::now();
    // 4 steps is a liveness check, not a throughput measurement: with paged
    // experts the first tokens are dominated by cold page-ins, so a short run
    // reports the warm-up rather than the steady state. Override to measure.
    let steps: usize = std::env::args()
        .nth(2)
        .and_then(|v| v.parse().ok())
        .unwrap_or(4);
    for i in 0..steps {
        m.decode_step(&mut gpu, tok, prompt.len() + i)
            .expect("decode");
        let l = gpu.download_f32(m.logits()).expect("download");
        assert!(l.iter().all(|v| v.is_finite()), "decode logits finite");
        let (a, _) = l
            .iter()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &v)| {
                if v > bv {
                    (i, v)
                } else {
                    (bi, bv)
                }
            });
        moved |= a as u32 != tok;
        tok = a as u32;
    }
    println!(
        "decode {steps} steps in {:.2}s ({:.2} s/tok), argmax moved: {moved}",
        t2.elapsed().as_secs_f32(),
        t2.elapsed().as_secs_f32() / steps as f32
    );
    assert!(moved, "a frozen argmax means a dead forward");
    // Paging is the whole reason this model loads at all: routed experts are
    // 97.3% of the trunk. Printed after decoding, when the counters mean
    // something — at load time they are all zero.
    match m.expert_pager_stats() {
        Some(st) => println!(
            "  paging: {} modules registered, {} resident ({:.1} GiB), {} cold loads, {} hits, {} evictions",
            st.registered_modules,
            st.resident_modules,
            st.resident_module_bytes as f64 / (1u64 << 30) as f64,
            st.module_cold_loads,
            st.module_cache_hits,
            st.module_evictions
        ),
        None => println!("  paging: routed experts loaded resident"),
    }
    println!("serve_real: OK");
}
