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
        mtp_mode: "off",
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

    // 3b. BATCHED prefill vs the per-token decode step. `prefill` above ran the
    // layer-major batched forward; here the same prompt goes one token at a time
    // through the decode step (the pre-batching path), and again in chunks of 3
    // so a chunk boundary — the n-gram window and every recurrent state crossing
    // it — is exercised on an 8-token prompt. Not a hash compare: Opus GEMMs
    // quantise activations where the GEMV does not. One decode step after each
    // arm then checks the STATE the prefill left, not just its last logits.
    {
        let max_abs = |l: &[f32]| l.iter().fold(0.0f32, |a, v| a.max(v.abs()));
        let argmax = |l: &[f32]| {
            l.iter()
                .enumerate()
                .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &v)| {
                    if v > bv {
                        (i, v)
                    } else {
                        (bi, bv)
                    }
                })
                .0
        };
        let run = |m: &mut Qwen4ExpBackend, gpu: &mut hipfire_rdna::Gpu, mode: &str| {
            m.reset_session(gpu, "serve_fixture")
                .unwrap_or_else(|e| fail(format!("reset: {e}")));
            match mode {
                "per-token" => {
                    for (pos, &t) in prompt.iter().enumerate() {
                        if pos == 0 {
                            m.prefill(gpu, &prompt[..1])
                        } else {
                            m.decode_step(gpu, t, pos)
                        }
                        .unwrap_or_else(|e| fail(format!("per-token step {pos}: {e}")));
                    }
                }
                _ => m
                    .prefill(gpu, &prompt)
                    .unwrap_or_else(|e| fail(format!("{mode} prefill: {e}"))),
            }
            let after_prefill = gpu.download_f32(m.logits()).unwrap();
            m.decode_step(gpu, prompt[0], prompt.len())
                .unwrap_or_else(|e| fail(format!("{mode} decode: {e}")));
            (after_prefill, gpu.download_f32(m.logits()).unwrap())
        };
        let (ref_p, ref_d) = run(&mut m, &mut gpu, "per-token");
        let one = run(&mut m, &mut gpu, "batched");
        m.set_prefill_chunk(&mut gpu, 3);
        let chunked = run(&mut m, &mut gpu, "chunked");
        m.set_prefill_chunk(&mut gpu, 64);
        let scale = max_abs(&ref_p).max(max_abs(&ref_d)).max(1e-6);
        for (name, (p, d)) in [("batched", &one), ("chunk=3", &chunked)] {
            let dp = p
                .iter()
                .zip(&ref_p)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f32::max);
            let dd = d
                .iter()
                .zip(&ref_d)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f32::max);
            println!(
                "  {name:8} vs per-token: prefill argmax {} / {}, max|d| {dp:.3e}; \
                 next-step argmax {} / {}, max|d| {dd:.3e} (scale {scale:.3})",
                argmax(p),
                argmax(&ref_p),
                argmax(d),
                argmax(&ref_d)
            );
            if argmax(p) != argmax(&ref_p) || argmax(d) != argmax(&ref_d) {
                fail(format!("{name} prefill changed the argmax vs per-token"));
            }
            // 2% of the logit scale: int8 activation quantisation is ~0.4% per
            // GEMM; a wrong row, state or chunk boundary is O(1) — bf16 agrees to
            // ~1e-7, so the layer-major forward itself contributes nothing.
            //
            // qwen4_exp's prefill asks for f16 activations on Oq4 (the shared
            // default is int4 at small batch, 13-15% here); the gate also runs
            // this at HIPFIRE_OQ4_PREFILL_ACT_BITS=8, which must hold the bound too.
            let bound = 0.02;
            if dp.max(dd) > bound * scale {
                fail(format!(
                    "{name} prefill diverges from per-token: max|d| {:.3e} > {}% of {scale:.3}",
                    dp.max(dd),
                    bound * 100.0
                ));
            }
        }
    }

    // 3c. The KLD/PPL scorer the daemon's `kld_eval` drives: the batched scorer
    // (`kld_forward`) against the blanket per-position `SimpleAr` walk, over the
    // same chunk with a chunk boundary inside it. Mean NLL must agree; the row
    // count must cover exactly `[scoring_start, len-1)`.
    {
        use hipfire_runtime::kld_eval::ChunkScoredForward;
        let chunk: Vec<u32> = (0..12u32).map(|i| (i * 37 + 5) % vocab as u32).collect();
        let start = 2usize;
        let nll = |w: &hipfire_runtime::kld_eval::ScoredWindow<'_>, out: &mut Vec<f64>| {
            for i in 0..w.rows() {
                let r = w.row(i);
                let mx = r.iter().fold(f32::NEG_INFINITY, |a, &v| a.max(v)) as f64;
                let lse = mx + r.iter().map(|&v| (v as f64 - mx).exp()).sum::<f64>().ln();
                out.push(lse - r[w.nexts[i] as usize] as f64);
            }
        };
        let mut walk = Vec::new();
        ChunkScoredForward::forward_chunk_scored(&mut m, &mut gpu, &chunk, start, &mut |w| {
            nll(w, &mut walk)
        })
        .unwrap_or_else(|e| fail(format!("walk scorer: {e}")));
        m.set_prefill_chunk(&mut gpu, 4);
        let mut batched = Vec::new();
        m.kld_forward()
            .unwrap_or_else(|| fail("no kld_forward".into()))
            .forward_chunk_scored(&mut gpu, &chunk, start, &mut |w| nll(w, &mut batched))
            .unwrap_or_else(|e| fail(format!("batched scorer: {e}")));
        m.set_prefill_chunk(&mut gpu, 64);
        let want = chunk.len() - 1 - start;
        let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len().max(1) as f64;
        println!(
            "  scorer:  batched {} rows mean NLL {:.5}, walk {} rows {:.5}",
            batched.len(),
            mean(&batched),
            walk.len(),
            mean(&walk)
        );
        if batched.len() != want || walk.len() != want {
            fail(format!(
                "scorer row count: batched {} walk {} want {want}",
                batched.len(),
                walk.len()
            ));
        }
        if (mean(&batched) - mean(&walk)).abs() > 0.02 * mean(&walk).abs().max(1.0) {
            fail("batched scorer NLL disagrees with the per-position walk".into());
        }
    }

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

    // Whether the experts came from the pager, and if so how hard it worked.
    // Printed HERE, after decoding — read at load time every counter is zero,
    // and "paged agrees with resident" is evidence only if the paged run paged.
    // A `0 cold loads` line means the paged arm silently served resident weights
    // and the comparison proved nothing.
    match m.expert_pager_stats() {
        Some(st) => {
            println!(
                "  paging: {} modules registered, {} resident, {} cold loads, {} hits, {} evictions",
                st.registered_modules,
                st.resident_modules,
                st.module_cold_loads,
                st.module_cache_hits,
                st.module_evictions
            );
            if st.module_cold_loads == 0 {
                fail("a module table was registered but nothing was ever paged in".into());
            }
        }
        None => println!("  paging: experts loaded resident (no module table)"),
    }

    // 4b. `serve()` END TO END. Everything above drives `SimpleAr` directly; this
    // is the entry point the DAEMON actually calls, and it exercises the pieces
    // between — the tokenizer, the sampler, the EOS filter and the streaming sink.
    // A backend can pass every check above and still fail here (a wrong eos, a
    // vocab the sampler sizes differently), so it is worth its own run.
    {
        let tok = hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
            .unwrap_or_else(|e| fail(format!("fixture has no usable tokenizer: {e:?}")));
        let mut out: Vec<u8> = Vec::new();
        let mut ctx = hipfire_runtime::arch::GenerateCtx {
            id: "serve_fixture",
            prompt: "hello",
            temperature: 0.0,
            top_p: 1.0,
            top_k: 1,
            max_tokens: 8,
            repeat_penalty: 1.0,
            repeat_window: 0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            max_think_tokens: 0,
            stop_sequences: &[],
            output_holdback_prefixes: &[],
            strip_think: false,
            output_protocol: hipfire_runtime::arch::OutputProtocol::Plain,
            images: &[],
            sink: &mut out,
        };
        match m.serve(&mut gpu, &tok, &mut ctx) {
            Ok(outcome) => println!(
                "  serve():  {} token(s) from {} prompt token(s), stop = {:?}, {} visible byte(s)",
                outcome.tokens_generated,
                outcome.prompt_tokens,
                outcome.stop_reason,
                out.len()
            ),
            Err(e) => fail(format!("serve(): {e}")),
        }
    }

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
