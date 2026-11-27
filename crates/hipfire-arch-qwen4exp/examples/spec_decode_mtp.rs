// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! MTP SPECULATIVE DECODE on the real 180B, through the SERVING path.
//!
//! Greedy-decodes the same real-text prompt twice — plain, then with the MTP
//! head attached to the backend (`attach_mtp`, which is what the factory does
//! for `mtp_mode=auto|on`) — and reports tok/s, the hit rate, and whether the
//! two runs produced the same tokens.
//!
//! Speculation is exact (every logit row comes from the trunk), so a token
//! difference can only be a near-tie flipped by the 2-row verify's numerics
//! (its GEMMs are W8A8/W4A16 where decode's GEMVs are not); the first such
//! position is reported, not treated as a failure.
//!
//! The prompt must be REAL text: a synthetic id stream drives the trunk into a
//! repetition cycle that inflates acceptance (the 2026-09-24 probe learned that
//! the hard way).
//!
//!     spec_decode_mtp <base.hfq> <mtp-sidecar.hfq> <prompt.txt> [n_tokens] [prompt_tokens]

use hipfire_arch_qwen4exp::serving::Qwen4ExpBackend;
use hipfire_model::tokenizer::Tokenizer;
use hipfire_rdna::Gpu;
use hipfire_runtime::arch::SimpleAr;
use hipfire_runtime::hfq::HfqFile;
use std::path::Path;
use std::time::Instant;

fn argmax(v: &[f32]) -> u32 {
    v.iter()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &x)| {
            if x > bv {
                (i, x)
            } else {
                (bi, bv)
            }
        })
        .0 as u32
}

/// Greedy decode `n` tokens after `prompt`; returns tokens and decode seconds.
fn generate(m: &mut Qwen4ExpBackend, gpu: &mut Gpu, prompt: &[u32], n: usize) -> (Vec<u32>, f32) {
    m.prefill(gpu, prompt).expect("prefill");
    let mut toks = Vec::with_capacity(n);
    let t0 = Instant::now();
    for i in 0..n {
        let t = argmax(&gpu.download_f32(m.logits()).expect("logits"));
        toks.push(t);
        m.decode_step(gpu, t, prompt.len() + i).expect("decode");
    }
    (toks, t0.elapsed().as_secs_f32())
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let usage = "usage: spec_decode_mtp <base> <mtp> <prompt.txt> [n_tokens] [prompt_tokens]";
    let (base, mtp, prompt_path) = match a.as_slice() {
        [b, m, p, ..] => (b, m, p),
        _ => panic!("{usage}"),
    };
    let n: usize = a.get(3).and_then(|v| v.parse().ok()).unwrap_or(128);
    let p_len: usize = a.get(4).and_then(|v| v.parse().ok()).unwrap_or(256);

    let mut hfq = HfqFile::open(Path::new(base)).expect("open base");
    let mut gpu = match Gpu::init() {
        Ok(g) => g,
        Err(e) => {
            println!("spec_decode_mtp: no GPU ({e:?}) — skipped");
            return;
        }
    };
    let tok = Tokenizer::from_hfq_metadata(&hfq.metadata_json).expect("tokenizer");
    let text = std::fs::read_to_string(prompt_path).expect("read prompt");
    let ids = tok.encode(&text);
    let prompt = &ids[..p_len.min(ids.len())];
    let mut m = Qwen4ExpBackend::load(&mut gpu, &mut hfq, prompt.len() + n + 16).expect("load");

    // Warm both paths once so neither timed run pays the kernel JIT.
    let _ = generate(&mut m, &mut gpu, prompt, 4);
    let colds = |m: &Qwen4ExpBackend| m.expert_pager_stats().map_or(0, |s| s.module_cold_loads);
    let c0 = colds(&m);
    let (plain, t_plain) = generate(&mut m, &mut gpu, prompt, n);
    let c_plain = colds(&m) - c0;

    m.attach_mtp(&mut gpu, Path::new(mtp)).expect("attach mtp");
    let _ = generate(&mut m, &mut gpu, prompt, 4);
    let before = m.mtp_stats().expect("attached");
    let c1 = colds(&m);
    let (spec, t_spec) = generate(&mut m, &mut gpu, prompt, n);
    let c_spec = colds(&m) - c1;
    let st = m.mtp_stats().expect("attached");
    let (v, h, x) = (
        st.verifies - before.verifies,
        st.hits - before.hits,
        st.misses - before.misses,
    );

    let first_diff = plain.iter().zip(&spec).position(|(a, b)| a != b);
    println!("prompt {} tokens, {n} generated", prompt.len());
    println!(
        "  plain  {:.2} tok/s ({:.3} s/tok)",
        n as f32 / t_plain,
        t_plain / n as f32
    );
    println!(
        "  +MTP   {:.2} tok/s ({:.3} s/tok)  {:+.1}%",
        n as f32 / t_spec,
        t_spec / n as f32,
        100.0 * (t_plain / t_spec - 1.0)
    );
    println!(
        "  verifies {v}, hits {h}, misses {x}  -> acceptance {:.1}%, {:.2} tokens per verify",
        100.0 * h as f32 / (h + x).max(1) as f32,
        n as f32 / v.max(1) as f32
    );
    // Each run re-prefills the same prompt, so the cold loads include that
    // prefill's; the DIFFERENCE between the runs is the decode's.
    println!("  expert cold loads (prefill + decode): plain {c_plain}, +MTP {c_spec}");
    match first_diff {
        None => println!("  tokens: identical over {n}"),
        Some(i) => {
            println!("  tokens: first differ at {i} (a near-tie under the verify's numerics)")
        }
    }
}
