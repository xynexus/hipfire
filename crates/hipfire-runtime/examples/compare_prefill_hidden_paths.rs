// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! Do the BATCHED and PER-TOKEN prefill paths export the same hidden states?
//!
//! This is the comparison the DFlash drafter's correctness actually rests on.
//! Enabling compact-resident Opus for a hidden-capturing forward reproducibly
//! took DFlash2's accept_rate 0.468 -> 0.000, and the obvious diagnostics do NOT
//! answer why:
//!
//!   * `HIPFIRE_DUMP_HIDDEN_ALLLAYERS` writes `.batched.L{i}` from one path and
//!     `.pertoken.L{i}` from the other — DIFFERENT call sites capturing
//!     different quantities, so diffing them is apples-to-oranges (it reads as
//!     500% divergence at layer 0, which the character-identical generated text
//!     disproves).
//!   * `.fnorm` fires once per forward and, in the batched path, reads a
//!     scratch buffer that path never fills — 5120/5120 NaN, a dump artifact.
//!
//! So this runs BOTH paths in ONE process against the SAME
//! `HiddenStateRingBuffer` — the buffer the drafter consumes — and diffs it
//! layer by layer, position by position.
//!
//!   compare_prefill_hidden_paths --model <m.hfq> [--n 48] [--kv-mode kvarn]
//!
//! Needs `HIPFIRE_PROBE_COMPACT_HIDDEN=1` to be meaningful on a compact model:
//! without it compact declines the hidden-exporting forward by design and both
//! arms run per-token.

use hipfire_arch_qwen35::qwen35::{self, DeltaNetState, Qwen35Scratch};
use hipfire_arch_qwen35::speculative::HiddenStateRingBuffer;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::kv::KvCache;

fn main() {
    let argv: Vec<String> = std::env::args().collect();
    let mut model = None;
    let mut n: usize = 48;
    let mut kv_mode = "kvarn".to_string();
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--model" => {
                model = Some(argv[i + 1].clone());
                i += 2;
            }
            "--n" => {
                n = argv[i + 1].parse().expect("--n");
                i += 2;
            }
            "--kv-mode" => {
                kv_mode = argv[i + 1].clone();
                i += 2;
            }
            _ => i += 1,
        }
    }
    let model = model.expect("--model required");

    let mut hfq = HfqFile::open(std::path::Path::new(&model)).expect("open model");
    let config = qwen35::config_from_hfq(&hfq).expect("config");
    let mut gpu = hipfire_rdna::Gpu::init().expect("gpu init");
    let weights = qwen35::load_weights(&mut hfq, &config, &mut gpu).expect("weights");
    eprintln!(
        "arch={} dim={} n_layers={} n={n} kv={kv_mode}",
        gpu.arch, config.dim, config.n_layers
    );

    // Arbitrary but fixed token ids — this compares two execution paths on the
    // same input, so the text does not matter, only that both see it.
    let tokens: Vec<u32> = (0..n).map(|i| (1000 + i * 7) as u32).collect();
    let kv_max = n + 16;

    let run = |gpu: &mut hipfire_rdna::Gpu, batched: bool| -> Vec<Vec<f32>> {
        if batched {
            std::env::remove_var("HIPFIRE_PREFILL_BATCHED");
        } else {
            std::env::set_var("HIPFIRE_PREFILL_BATCHED", "0");
        }
        let mut kv_cache = match kv_mode.as_str() {
            "kvarn" => KvCache::new_gpu_kvarn(
                gpu,
                config.n_layers,
                config.n_kv_heads,
                config.head_dim,
                kv_max,
                8,
            )
            .expect("kv kvarn"),
            "q8" => KvCache::new_gpu_q8(
                gpu,
                config.n_layers,
                config.n_kv_heads,
                config.head_dim,
                kv_max,
            )
            .expect("kv q8"),
            "fp32" => KvCache::new_gpu(
                gpu,
                config.n_layers,
                config.n_kv_heads,
                config.head_dim,
                kv_max,
            )
            .expect("kv fp32"),
            other => panic!("unsupported --kv-mode {other}"),
        };
        let scratch = Qwen35Scratch::new(gpu, &config, 64).expect("scratch");
        let mut dn_state = DeltaNetState::new(gpu, &config).expect("dn state");
        let mut rb = HiddenStateRingBuffer::new_for_layers(
            gpu,
            config.n_layers,
            (0..config.n_layers).collect(),
            config.dim,
            n,
            // max_batch MUST cover the batched path's chunk commit — it stages
            // n rows at once, where the per-token path stages 1.
            n,
        )
        .expect("hidden rb");
        qwen35::forward_prefill_batch(
            gpu,
            &weights,
            &config,
            &tokens,
            0,
            &mut kv_cache,
            &mut dn_state,
            &scratch,
            Some(&mut rb),
            None,
            None,
            None,
        )
        .expect("forward_prefill_batch");
        // forward_prefill_batch commits each chunk itself; a second commit here
        // would re-advance the head and overwrite the capture with empty staging.
        rb.layer_bufs
            .iter()
            .map(|b| gpu.download_f32(b).expect("download layer"))
            .collect()
    };

    eprintln!("-- batched --");
    let a = run(&mut gpu, true);
    eprintln!("-- per-token --");
    let b = run(&mut gpu, false);

    let dim = config.dim;
    println!("\n  layer   worst|rel|   at row   nonfinite(batched)");
    let mut first_bad = None;
    let mut worst_all = 0f32;
    for (l, (x, y)) in a.iter().zip(&b).enumerate() {
        let rows = (x.len() / dim).min(y.len() / dim).min(n);
        let mut worst = 0f32;
        let mut wr = 0usize;
        let mut nonfinite = 0usize;
        for r in 0..rows {
            let (xs, ys) = (&x[r * dim..(r + 1) * dim], &y[r * dim..(r + 1) * dim]);
            if xs.iter().any(|v| !v.is_finite()) {
                nonfinite += 1;
                continue;
            }
            let scale = ys.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-30);
            let d = xs
                .iter()
                .zip(ys)
                .fold(0f32, |m, (p, q)| m.max((p - q).abs()))
                / scale;
            if d > worst {
                worst = d;
                wr = r;
            }
        }
        worst_all = worst_all.max(worst);
        let bad = worst > 1e-3 || nonfinite > 0;
        if bad && first_bad.is_none() {
            first_bad = Some(l);
        }
        if l < 4 || bad {
            println!("  {l:>5}   {worst:>9.2e}   {wr:>6}   {nonfinite}");
        }
    }
    match first_bad {
        Some(l) => println!("\nFIRST DIVERGING LAYER: {l}   (worst overall {worst_all:.2e})"),
        None => println!("\nIDENTICAL across all layers (worst {worst_all:.2e})"),
    }
}
