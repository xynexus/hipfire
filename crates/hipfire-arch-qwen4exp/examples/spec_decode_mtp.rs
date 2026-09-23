// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! MTP SPECULATIVE DECODE on the real 180B: acceptance, and what it is worth.
//!
//! Measures the drafter end to end — the GPU head (`mtp_gpu`) against the live
//! trunk, on self-generated text — and reports the two numbers that decide
//! whether speculation pays here.
//!
//! # ⚠️ Why this cannot be faster TODAY, and is measured anyway
//!
//! Speculation pays only when a verify checks the draft AND produces the next
//! token in ONE forward. `serving.rs` says it plainly: "Prefill is per-token.
//! The trunk has no batched prefill." So a draft must be verified by a full
//! trunk step — the same step plain autoregression would have run — and nothing
//! is saved:
//!
//!     AR                          2.000 trunk-equivalents / 2 tokens
//!     MTP, per-token verify       2.045   (the head is 4.5% of a trunk step)
//!     MTP, 2-token batched verify 1.280   -> 1.56x, at 56% acceptance
//!
//! The head is therefore pure overhead until a batched forward exists, and this
//! example exists to measure ACCEPTANCE on the real model rather than to claim a
//! speedup. Acceptance is the number that says whether the 1.56x is there to be
//! collected once batching lands; the tok/s columns are reported so the overhead
//! is visible rather than implied.
//!
//!     spec_decode_mtp <base.hfq> <mtp-sidecar.hfq> [n_tokens]

use hipfire_arch_qwen4exp::arch::HfqTensorReader;
use hipfire_arch_qwen4exp::mtp::Fusion;
use hipfire_arch_qwen4exp::mtp_gpu::{mtp_decode_step, MtpScratchGpu, MtpWeightsGpu};
use hipfire_arch_qwen4exp::serving::Qwen4ExpBackend;
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

fn main() {
    let mut args = std::env::args().skip(1);
    let base = args
        .next()
        .expect("usage: spec_decode_mtp <base> <mtp> [n]");
    let mtp_path = args
        .next()
        .expect("usage: spec_decode_mtp <base> <mtp> [n]");
    let n_tok: usize = args.next().and_then(|v| v.parse().ok()).unwrap_or(32);

    let mut hfq = HfqFile::open(Path::new(&base)).expect("open base");
    let mut gpu = match hipfire_rdna::Gpu::init() {
        Ok(g) => g,
        Err(e) => {
            println!("spec_decode_mtp: no GPU ({e:?}) — skipped");
            return;
        }
    };
    let t0 = Instant::now();
    let mut m = Qwen4ExpBackend::load(&mut gpu, &mut hfq, 256).expect("load base");
    let cfg = m.config().clone();
    println!("trunk loaded in {:.1}s", t0.elapsed().as_secs_f32());

    // The head is ~2.6 B params — it goes on the GPU whole, no host copy.
    let mut mtp_hfq = HfqFile::open(Path::new(&mtp_path)).expect("open mtp sidecar");
    let t1 = Instant::now();
    let head = {
        let reader = HfqTensorReader { hfq: &mut mtp_hfq };
        MtpWeightsGpu::upload(&mut gpu, &cfg, &reader).expect("upload mtp head")
    };
    let mut hs = MtpScratchGpu::new(&mut gpu, &cfg, 256).expect("mtp scratch");
    let mut hcache =
        hipfire_arch_qwen4exp::attn_gpu::QsaCache::new(&mut gpu, &cfg, 256).expect("head cache");
    println!("mtp head loaded in {:.1}s", t1.elapsed().as_secs_f32());

    // ── arm 1: plain autoregression, the baseline ─────────────────────────
    let seed = 9707u32;
    m.prefill(&mut gpu, &[seed]).expect("prefill");
    let mut ar_tokens = vec![seed];
    let t2 = Instant::now();
    for i in 0..n_tok {
        if i > 0 {
            let last = *ar_tokens.last().unwrap();
            m.decode_step(&mut gpu, last, i).expect("decode");
        }
        let lg = gpu.download_f32(m.trunk_logits()).expect("logits");
        ar_tokens.push(argmax(&lg));
    }
    let ar_s = t2.elapsed().as_secs_f32();
    let ar_tps = n_tok as f32 / ar_s;

    // ── arm 2: the same sequence, drafting each next token ────────────────
    //
    // The head drafts x_{t+2} from the trunk's wide residual at t plus
    // emb(x_{t+1}); the NEXT trunk step is the verify. Acceptance is how often
    // the draft equals what the trunk then produced.
    m.prefill(&mut gpu, &[seed]).expect("prefill");
    hcache.reset();
    let mut toks = vec![seed];
    let (mut drafted, mut accepted) = (0usize, 0usize);
    let mut pending: Option<u32> = None;
    let t3 = Instant::now();
    for i in 0..n_tok {
        if i > 0 {
            let last = *toks.last().unwrap();
            m.decode_step(&mut gpu, last, i).expect("decode");
        }
        let lg = gpu.download_f32(m.trunk_logits()).expect("logits");
        let real = argmax(&lg);

        // Verify whatever the head proposed for THIS position.
        if let Some(d) = pending.take() {
            drafted += 1;
            if d == real {
                accepted += 1;
            }
        }
        toks.push(real);

        // Draft the next one: h_t (wide) + emb(real).
        let (wide_t, _) = m.trunk_states();
        let g_emb = gpu
            .upload_f32(m.embed_row(real), &[cfg.hidden])
            .expect("upload emb");
        mtp_decode_step(
            &mut gpu,
            &cfg,
            &head,
            &mut hcache,
            &mut hs,
            wide_t,
            &g_emb,
            i,
            Fusion::BroadcastAllStreams,
        )
        .expect("mtp step");
        // The head shares the trunk's lm_head.
        let dl = m.logits_of(&mut gpu, &hs.collapsed).expect("head logits");
        pending = Some(argmax(&dl));
    }
    let sp_s = t3.elapsed().as_secs_f32();

    let acc = if drafted > 0 {
        accepted as f32 / drafted as f32
    } else {
        0.0
    };
    println!("\n  AR          {n_tok} tok in {ar_s:.2}s  ({ar_tps:.2} tok/s)");
    println!(
        "  +MTP draft  {n_tok} tok in {sp_s:.2}s  ({:.2} tok/s)   overhead {:+.1}%",
        n_tok as f32 / sp_s,
        100.0 * (sp_s / ar_s - 1.0)
    );
    println!(
        "\n  drafts {accepted}/{drafted} accepted  ({:.1}%)",
        100.0 * acc
    );
    println!(
        "\n  With a 2-token BATCHED verify this acceptance is worth {:.2}x;\n  \
         without one the draft cannot be cashed in and the head is overhead.\n  \
         Batched prefill is the prerequisite (see serving.rs's own header).",
        (1.0 + acc) / (1.0 + 0.045)
    );
}
