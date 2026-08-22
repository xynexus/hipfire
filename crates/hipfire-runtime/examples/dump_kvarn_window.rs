// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! Does batched prefill WRITE the same KVarN cache as per-token, or only READ it
//! differently?
//!
//! `compare_prefill_hidden_paths` shows the two paths diverge under kvarn but not
//! under fp32. That is a symptom at the far end of the pipe. This dumps the cache
//! itself — the f32 recent window and the Q8 V plane — right after each path
//! prefills the SAME tokens, and diffs them byte for byte.
//!
//! Read it as a two-way split:
//!   * windows/V differ  → the WRITE side is wrong (layout, ordering, rotation).
//!   * windows/V match   → the write side is fine and the divergence is in the
//!                         flash kernel's multi-row READ (per-token issues n
//!                         calls of 1 row; batched issues 1 call of n rows).
//!
//! Run at n < 128 so no 128-token block ever flushes: then every K row is still
//! f32 in the window and NO quantization has happened, which removes the codec
//! from the question entirely.
//!
//!   dump_kvarn_window --model <m.hfq> [--n 64]

use hipfire_arch_qwen35::qwen35::{self, DeltaNetState, Qwen35Scratch};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::kv::KvCache;

fn main() {
    let argv: Vec<String> = std::env::args().collect();
    let mut model = None;
    let mut n: usize = 64;
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
            other => panic!("unknown arg {other}"),
        }
    }
    let model = model.expect("--model required");

    let mut hfq = HfqFile::open(std::path::Path::new(&model)).expect("open model");
    let config = qwen35::config_from_hfq(&hfq).expect("config");
    let mut gpu = hipfire_rdna::Gpu::init().expect("gpu init");
    let weights = qwen35::load_weights(&mut hfq, &config, &mut gpu).expect("weights");
    let tokens: Vec<u32> = (0..n).map(|i| (1000 + i * 7) as u32).collect();
    let kv_max = n + 16;
    eprintln!(
        "arch={} dim={} n_layers={} head_dim={} n_kv_heads={} n={n}",
        gpu.arch, config.dim, config.n_layers, config.head_dim, config.n_kv_heads
    );
    assert!(n < 128, "run with n < 128 so no block flushes");

    // (window per layer, v per layer)
    let run = |gpu: &mut hipfire_rdna::Gpu, batched: bool| -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        if batched {
            std::env::remove_var("HIPFIRE_PREFILL_BATCHED");
        } else {
            std::env::set_var("HIPFIRE_PREFILL_BATCHED", "0");
        }
        let mut kv_cache = KvCache::new_gpu_kvarn(
            gpu,
            config.n_layers,
            config.n_kv_heads,
            config.head_dim,
            kv_max,
            8,
        )
        .expect("kv kvarn");
        let scratch = Qwen35Scratch::new(gpu, &config, 64).expect("scratch");
        let mut dn_state = DeltaNetState::new(gpu, &config).expect("dn state");
        qwen35::forward_prefill_batch(
            gpu,
            &weights,
            &config,
            &tokens,
            0,
            &mut kv_cache,
            &mut dn_state,
            &scratch,
            None,
            None,
            None,
            None,
        )
        .expect("forward_prefill_batch");
        let win = kv_cache
            .k_window
            .iter()
            .map(|b| gpu.download_f32(b).unwrap_or_default())
            .collect();
        let v = kv_cache
            .v_gpu
            .iter()
            .map(|b| gpu.download_f32(b).unwrap_or_default())
            .collect();
        (win, v)
    };

    eprintln!("-- batched --");
    let (wb, vb) = run(&mut gpu, true);
    eprintln!("-- per-token --");
    let (wp, vp) = run(&mut gpu, false);

    let kv_dim = config.n_kv_heads * config.head_dim;
    // Only the first n rows of the window are live; the rest is untouched staging.
    let live = n * kv_dim;
    let cmp = |a: &Vec<f32>, b: &Vec<f32>, limit: usize| -> (f32, usize) {
        let m = a.len().min(b.len()).min(limit);
        let mut worst = 0f32;
        let mut ndiff = 0usize;
        for i in 0..m {
            let d = (a[i] - b[i]).abs();
            if d > 0.0 {
                ndiff += 1;
            }
            worst = worst.max(d);
        }
        (worst, ndiff)
    };

    println!("\nK WINDOW (f32, first {n} rows = {live} floats/layer)");
    println!("  layer   worst|abs|   #differing   verdict");
    let mut any_win = false;
    for l in 0..wb.len().min(wp.len()) {
        if wb[l].is_empty() || wp[l].is_empty() {
            continue;
        }
        let (w, d) = cmp(&wb[l], &wp[l], live);
        if d > 0 {
            any_win = true;
        }
        if d > 0 || l < 3 {
            let mag = |x: &Vec<f32>| -> (f32, usize) {
                let m = x.len().min(live);
                let mut mx = 0f32;
                let mut nz = 0usize;
                for i in 0..m {
                    mx = mx.max(x[i].abs());
                    if x[i] != 0.0 {
                        nz += 1;
                    }
                }
                (mx, nz)
            };
            let (mb, nb) = mag(&wb[l]);
            let (mp, np) = mag(&wp[l]);
            println!(
                "  {l:>5}   {w:>9.3e}   {d:>10}   {}   batched[max {mb:.3e} nonzero {nb}] per-token[max {mp:.3e} nonzero {np}]",
                if d == 0 { "match" } else { "DIFFER" }
            );
        }
    }
    if !any_win {
        println!("  all layers MATCH exactly");
    }

    println!("\nV PLANE (Q8_0 bytes viewed as f32 words)");
    let mut any_v = false;
    for l in 0..vb.len().min(vp.len()) {
        if vb[l].is_empty() || vp[l].is_empty() {
            continue;
        }
        // Q8_0: head_dim/32 blocks per head, 34 bytes each, per row.
        let v_row_words = config.n_kv_heads * (config.head_dim / 32) * 34 / 4;
        let (w, d) = cmp(&vb[l], &vp[l], n * v_row_words);
        if d > 0 {
            any_v = true;
        }
        if d > 0 || l < 3 {
            println!(
                "  {l:>5}   {w:>9.3e}   {d:>10}   {}",
                if d == 0 { "match" } else { "DIFFER" }
            );
        }
    }
    if !any_v {
        println!("  all layers MATCH exactly");
    }

    println!(
        "\nVERDICT: write side {}",
        if any_win || any_v {
            "DIFFERS → the bug is in the KVarN WRITE path"
        } else {
            "IDENTICAL → the bug is in the flash kernel's multi-row READ"
        }
    );
}
