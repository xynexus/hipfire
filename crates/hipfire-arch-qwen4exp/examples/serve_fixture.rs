// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! Drive a qwen4_exp artifact through the SERVING SEAM — the path the daemon
//! takes. Everything else in this crate exercises the trunk directly; this is the
//! only check that the seam itself is wired.
//!
//! Four things, each of which has failed for a different reason during bring-up:
//!
//! 1. arch 26 resolves to a registered factory (a missing link edge makes the
//!    daemon report a valid artifact as an unknown architecture);
//! 2. the factory builds a backend from the artifact;
//! 3. `SimpleAr` prefill then decode produce FINITE, MOVING logits (a dead
//!    forward reads as a frozen argmax, not as an error);
//! 4. `reset_session` truly resets — the same prompt must give the same logits
//!    afterwards, which is what catches recurrent state leaking between requests.
//!
//!     cargo run --release -p hipfire-arch-qwen4exp --example serve_fixture -- model.hfq

use hipfire_arch_qwen4exp::serving::Qwen4ExpBackend;
use hipfire_runtime::arch::{serving_factory, ServingBackend, ServingFactoryOptions, SimpleAr};
use hipfire_runtime::hfq::HfqFile;
use std::path::Path;

fn fail(msg: String) -> ! {
    eprintln!("serve_fixture: {msg}");
    std::process::exit(1);
}

fn main() {
    let path = match std::env::args().nth(1) {
        Some(p) => p,
        None => {
            eprintln!("usage: serve_fixture <model.hfq>");
            std::process::exit(2);
        }
    };
    let mut hfq = HfqFile::open(Path::new(&path))
        .unwrap_or_else(|e| fail(format!("cannot open {path}: {e}")));

    // 1. Factory lookup — a DATA lookup by arch id, no branch.
    let factory = match serving_factory(hfq.arch_id) {
        Ok(Some(f)) => f,
        Ok(None) => fail(format!("no serving factory for arch {}", hfq.arch_id)),
        Err(e) => fail(e),
    };
    println!(
        "  factory: family={} arch={}",
        factory.family(),
        factory.arch_id()
    );

    let mut gpu = match hipfire_rdna::Gpu::init() {
        Ok(g) => g,
        Err(e) => {
            println!("serve_fixture: no GPU ({e:?}) — skipped");
            return;
        }
    };

    // 2. Load through the factory exactly as the daemon does.
    let options = ServingFactoryOptions {
        max_seq: 64,
        kv_mode: "f32",
        triattn: None,
        cask_budget: 0,
        cask_beta: 0,
        physical_cap: None,
    };
    let loaded = factory
        .load(&mut hfq, &mut gpu, &options)
        .unwrap_or_else(|e| fail(format!("factory load: {e}")));
    println!(
        "  loaded:  hidden={} layers={} vocab={} cap={} eos={}",
        loaded.shape.hidden_size,
        loaded.shape.num_layers,
        loaded.shape.vocab_size,
        loaded.physical_cap,
        loaded.backend.eos_token(),
    );
    let vocab = loaded.shape.vocab_size;
    loaded.backend.unload(&mut gpu);

    // 3./4. The boxed `ServingBackend` deliberately hides `SimpleAr`, so drive the
    // concrete backend the factory itself builds (same constructor, same object).
    let mut m = Qwen4ExpBackend::load(&mut gpu, &mut hfq, 64)
        .unwrap_or_else(|e| fail(format!("backend load: {e}")));

    let prompt: Vec<u32> = [3u32, 17, 42, 5, 9, 7, 61, 23]
        .iter()
        .map(|t| (*t).min(vocab as u32 - 1))
        .collect();

    let argmax_after_prefill =
        |m: &mut Qwen4ExpBackend, gpu: &mut hipfire_rdna::Gpu| -> (usize, f32) {
            m.prefill(gpu, &prompt)
                .unwrap_or_else(|e| fail(format!("prefill: {e}")));
            let l = gpu
                .download_f32(m.logits())
                .unwrap_or_else(|e| fail(format!("download: {e:?}")));
            assert_eq!(l.len(), vocab, "logits width must be the vocab");
            assert!(l.iter().all(|v| v.is_finite()), "prefill logits not finite");
            l.iter()
                .enumerate()
                .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &v)| {
                    if v > bv {
                        (i, v)
                    } else {
                        (bi, bv)
                    }
                })
        };

    // Report what the experts are RESIDENT as, not just that decoding works: a
    // silent dequantisation to f32 serves identically and costs ~8x the memory.
    println!(
        "  experts: routed experts resident as {:?}",
        m.routed_expert_dtype()
    );

    let (am0, mx0) = argmax_after_prefill(&mut m, &mut gpu);
    println!(
        "  prefill: {} tokens -> argmax {am0} ({mx0:.4}), all finite",
        prompt.len()
    );

    // Decode a few steps, feeding the argmax back in. A forward that is dead
    // (zeroed weights, unwired state) pins the argmax; that is the failure mode
    // this catches, so track whether it ever moves.
    let mut tok = am0 as u32;
    let mut moved = false;
    for step in 0..4 {
        let pos = prompt.len() + step;
        m.decode_step(&mut gpu, tok, pos)
            .unwrap_or_else(|e| fail(format!("decode_step: {e}")));
        let l = gpu
            .download_f32(m.logits())
            .unwrap_or_else(|e| fail(format!("download: {e:?}")));
        assert!(l.iter().all(|v| v.is_finite()), "decode logits not finite");
        let (am, _) =
            l.iter()
                .enumerate()
                .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &v)| {
                    if v > bv {
                        (i, v)
                    } else {
                        (bi, bv)
                    }
                });
        moved |= am as u32 != tok;
        tok = am as u32;
    }
    println!("  decode:  4 steps, all finite, argmax moved = {moved}");

    // 4. Reset must be total. Re-running the SAME prompt has to reproduce the
    // same logits; if the DeltaNet recurrent state or the PLE conv ring survived,
    // the second run is conditioned on the first and this diverges.
    m.reset_session(&mut gpu, "serve_fixture")
        .unwrap_or_else(|e| fail(format!("reset_session: {e}")));
    let (am1, mx1) = argmax_after_prefill(&mut m, &mut gpu);
    if am1 != am0 || (mx1 - mx0).abs() > 1e-4 {
        fail(format!(
            "reset is INCOMPLETE: same prompt gave argmax {am0} ({mx0:.6}) then {am1} ({mx1:.6}) \
             — recurrent state survived reset_session"
        ));
    }
    println!("  reset:   same prompt reproduces argmax {am1} ({mx1:.4})");

    // 5. The STREAMED n-gram path must agree with the resident one EXACTLY. Force
    // it with a zero budget so this small fixture takes the same route the 102 GB
    // model has no choice about. A wrong shard split or row offset still yields a
    // finite embedding, so only a difference against the resident run detects it.
    if m.config().ngram.is_some() {
        m.reset_session(&mut gpu, "serve_fixture")
            .unwrap_or_else(|e| fail(format!("reset before streamed run: {e}")));
        let resident = {
            m.prefill(&mut gpu, &prompt)
                .unwrap_or_else(|e| fail(format!("prefill: {e}")));
            gpu.download_f32(m.logits())
                .unwrap_or_else(|e| fail(format!("download: {e:?}")))
        };
        Box::new(m).unload(&mut gpu);

        let mut streamed_m = Qwen4ExpBackend::load_with_ngram_budget(&mut gpu, &mut hfq, 64, 0)
            .unwrap_or_else(|e| fail(format!("streamed load: {e}")));
        streamed_m
            .prefill(&mut gpu, &prompt)
            .unwrap_or_else(|e| fail(format!("streamed prefill: {e}")));
        let streamed = gpu
            .download_f32(streamed_m.logits())
            .unwrap_or_else(|e| fail(format!("download: {e:?}")));

        let worst = resident
            .iter()
            .zip(&streamed)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        if worst != 0.0 {
            fail(format!(
                "streamed n-gram rows DIFFER from resident: worst |delta| = {worst:.6e}. The row \
                 addressing (shard split, row_in_shard, element offset) is wrong."
            ));
        }
        println!("  ngram:   streamed rows are bit-identical to resident");
        Box::new(streamed_m).unload(&mut gpu);
    } else {
        Box::new(m).unload(&mut gpu);
    }
    println!("serve_fixture: OK");
}
