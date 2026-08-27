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
//! `HIPFIRE_COMPACT_BATCHED_CAPTURE=0` forces the per-token arm on a compact model:
//! without it compact declines the hidden-exporting forward by design and both
//! arms run per-token.

use hipfire_arch_qwen35::qwen35::{self, DeltaNetState, Qwen35Scratch};
use hipfire_arch_qwen35::speculative::HiddenStateRingBuffer;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::kv::KvCache;

fn main() {
    let argv: Vec<String> = std::env::args().collect();
    let mut model = None;
    // Default must CROSS the prefill chunk boundary (PREFILL_MAX_BATCH = 256).
    // Below it there is exactly one chunk, attention reads the in-flight f32 K/V
    // and NEVER reads the quantised cache — so every KV tier returns bit-identical
    // numbers and the tool silently reports "the KV tier does not matter". It was
    // 48, and that made every KV comparison ever run with it vacuous: kvarn2,
    // kvarn8 and q8 all read 2.766e-2 / 1.203e-2. At n=512 they separate properly
    // (kvarn2 2.842e-1 vs kvarn8 3.443e-2 batched).
    let mut n: usize = 512;
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
    // PREFILL_MAX_BATCH, duplicated rather than imported: this example does not
    // depend on hipfire-arch-qwen35's internals and the number is stable.
    const PREFILL_CHUNK: usize = 256;
    if n <= PREFILL_CHUNK {
        eprintln!(
            "WARNING: --n {n} <= the prefill chunk size ({PREFILL_CHUNK}), so prefill \
             runs as ONE chunk, attention never reads the quantised KV cache, and \
             every --kv-mode / HIPFIRE_KVARN_BITS will return IDENTICAL numbers. \
             Use --n > {PREFILL_CHUNK} to compare KV tiers."
        );
    }

    // Accept an HF safetensors snapshot directory as well as a .hfq, so this
    // can be pointed at the same weights a transformers oracle loads.
    let model_path = std::path::Path::new(&model);
    let mut hfq = if model_path.is_dir() {
        HfqFile::from_safetensors(model_path).expect("open safetensors dir")
    } else {
        HfqFile::open(model_path).expect("open model")
    };
    let config = qwen35::config_from_hfq(&hfq).expect("config");
    let mut gpu = hipfire_rdna::Gpu::init().expect("gpu init");
    let weights = qwen35::load_weights(&mut hfq, &config, &mut gpu).expect("weights");
    eprintln!(
        "arch={} dim={} n_layers={} n={n} kv={kv_mode}{}",
        gpu.arch,
        config.dim,
        config.n_layers,
        if kv_mode == "kvarn" {
            format!(" bits={}", KvCache::kvarn_bits_from_env())
        } else {
            String::new()
        }
    );

    // Arbitrary but fixed token ids — this compares two execution paths on the
    // same input, so the text does not matter, only that both see it.
    let tokens: Vec<u32> = (0..n).map(|i| (1000 + i * 7) as u32).collect();
    let kv_max = n + 16;

    // `grouped` toggles the MoE path-2 grouped GEMM. It is read per call by
    // `moe_grouped_gemm_supported_for_dtype`, so both arms can run in ONE
    // process against ONE loaded model -- which is the only way to compare them
    // honestly. (An env var set outside the process does NOT reach the daemon,
    // and `hipfire-eval --battery perplexity` does not route through
    // `forward_prefill_batch` at all, so neither can gate this path.)
    let run = |gpu: &mut hipfire_rdna::Gpu,
               batched: bool,
               kv_mode: &str,
               grouped: bool|
     -> Vec<Vec<f32>> {
        if grouped {
            std::env::set_var("HIPFIRE_MOE_COMPACT_GROUPED", "1");
        } else {
            std::env::set_var("HIPFIRE_MOE_COMPACT_GROUPED", "0");
        }
        if batched {
            std::env::remove_var("HIPFIRE_PREFILL_BATCHED");
        } else {
            std::env::set_var("HIPFIRE_PREFILL_BATCHED", "0");
        }
        let mut kv_cache = match kv_mode {
            // Width from HIPFIRE_KVARN_BITS (2/4/8, default 4) — the SAME source
            // serving uses via KvMode::Kvarn. This was hardcoded to 8, so the
            // fidelity tool measured KVarN-8 while the shipping default is 4;
            // its kvarn and q8 rows then came out byte-identical because both
            // were 8-bit, which reads as "the KV tier does not matter" when what
            // it actually meant was "you compared 8-bit against 8-bit".
            "kvarn" => KvCache::new_gpu_kvarn(
                gpu,
                config.n_layers,
                config.n_kv_heads,
                config.head_dim,
                kv_max,
                KvCache::kvarn_bits_from_env(),
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
    let a = run(&mut gpu, true, &kv_mode, false);
    eprintln!("-- per-token --");
    let b = run(&mut gpu, false, &kv_mode, false);
    // WHICH PATH IS RIGHT? Unquantized KV is the reference: both paths agree
    // EXACTLY there (measured 0.00e0), so it is the only fixed point available.
    // Comparing each quantized arm against it says which one the fix should make
    // canonical, instead of leaving that a judgement call.
    let reference = if kv_mode == "fp32" {
        None
    } else {
        eprintln!("-- reference: per-token, fp32 KV --");
        Some(run(&mut gpu, false, "fp32", false))
    };

    let dim = config.dim;

    // Oracle export: HIPFIRE_ORACLE_DUMP=<dir> writes each arm's per-layer
    // hidden states as raw f32 so an external reference (HF transformers on the
    // same weights and the same token ids) can say WHICH arm is closer. The
    // internal fp32-KV reference above only fixes the KV axis; it cannot
    // arbitrate the batched-vs-per-token forward, because both arms are ours.
    if let Ok(dir) = std::env::var("HIPFIRE_ORACLE_DUMP") {
        let d = std::path::Path::new(&dir);
        std::fs::create_dir_all(d).expect("oracle dump dir");
        for (tag, arm) in [("batched", &a), ("pertoken", &b)] {
            for (l, x) in arm.iter().enumerate() {
                let bytes: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
                std::fs::write(d.join(format!("{tag}.L{l}.f32")), &bytes).expect("write dump");
            }
        }
        std::fs::write(
            d.join("meta.json"),
            format!(
                "{{\"dim\":{},\"n\":{},\"layers\":{},\"token_rule\":\"1000+i*7\"}}",
                dim,
                n,
                a.len()
            ),
        )
        .expect("write meta");
        eprintln!("oracle dump: {} layers x2 arms -> {}", a.len(), dir);
    }

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

    // PER-ROW profile of the last layer. If the batched export were merely
    // noisier this would be flat; if it is positionally wrong — the signature of
    // spec-decode accepting EXACTLY 2 tokens every cycle — early rows match and
    // later ones do not.
    if let (Some(x), Some(y)) = (a.last(), b.last()) {
        println!("\nper-row divergence, last layer (batched vs per-token):");
        let rows = (x.len() / dim).min(y.len() / dim).min(n);
        for r0 in 0..rows.min(12) {
            let (xs, ys) = (&x[r0 * dim..(r0 + 1) * dim], &y[r0 * dim..(r0 + 1) * dim]);
            let sc = ys.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-30);
            let d = xs
                .iter()
                .zip(ys)
                .fold(0f32, |m, (p, q)| m.max((p - q).abs()))
                / sc;
            println!("  row {r0:>3}  {d:.3e}");
        }
    }

    // ── MoE PATH-2 GATE ─────────────────────────────────────────────────────
    //
    // The grouped MoE GEMM is worth +37.6% prefill and takes DFlash decode past
    // AR, but it changes what every prefill computes, and until now NOTHING
    // measured it: perplexity does not route through `forward_prefill_batch`,
    // and a token-stream diff is not evidence on this model (four separate
    // false alarms on this branch alone). This is the missing measurement --
    // same tokens, same model, same process, batched path either side, MoE
    // grouped GEMM the only thing that changes.
    {
        eprintln!("-- batched, MoE grouped (path 2) --");
        let g = run(&mut gpu, true, &kv_mode, true);
        let mut worst = 0f32;
        let mut worst_layer = 0usize;
        let mut nonfinite = 0usize;
        let mut first_bad: Option<(usize, f32)> = None;
        let mut per_layer: Vec<f32> = Vec::new();
        for (l, (x, y)) in g.iter().zip(&a).enumerate() {
            let mut lworst = 0f32;
            let rows = (x.len() / dim).min(y.len() / dim).min(n);
            for row in 0..rows {
                let (xs, ys) = (
                    &x[row * dim..(row + 1) * dim],
                    &y[row * dim..(row + 1) * dim],
                );
                if xs.iter().any(|v| !v.is_finite()) {
                    nonfinite += 1;
                    continue;
                }
                let sc = ys.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-30);
                let d = xs
                    .iter()
                    .zip(ys)
                    .fold(0f32, |m, (p, q)| m.max((p - q).abs()))
                    / sc;
                if d > lworst {
                    lworst = d;
                }
                if d > worst {
                    worst = d;
                    worst_layer = l;
                }
            }
            per_layer.push(lworst);
            if lworst > 1e-4 && first_bad.is_none() {
                first_bad = Some((l, lworst));
            }
        }
        println!("\nMoE PATH-2 GATE (grouped vs indexed, both batched, same KV):");
        println!("  worst |rel| {worst:.3e} at layer {worst_layer}   nonfinite rows {nonfinite}");
        match first_bad {
            Some((l, v)) => println!("  first layer over 1e-4: {l} ({v:.3e})"),
            None => println!("  no layer over 1e-4"),
        }
        let shown = per_layer.len().min(10);
        let head: Vec<String> = per_layer[..shown]
            .iter()
            .enumerate()
            .map(|(i, v)| format!("L{i}={v:.1e}"))
            .collect();
        println!("  per-layer head: {}", head.join(" "));
        // The GEMM itself is bit-exact (parity_gemm_oq_compact_moe_grouped), so
        // anything here is the surrounding pipeline: scatter/unscatter and the
        // down combine's summation order. Order-of-summation over k_top experts
        // is legitimate and lands near f32 epsilon; a layout or indexing fault
        // does not.
        let ok = nonfinite == 0 && worst < 1e-4;
        println!(
            "  {}",
            if ok {
                "PASS — consistent with summation order in the down combine"
            } else {
                "FAIL — too large for reordering: suspect scatter/unscatter or indexing"
            }
        );
    }

    if let Some(r) = reference {
        let dist = |x: &Vec<Vec<f32>>| -> f32 {
            let mut w = 0f32;
            for (l, (p, q)) in x.iter().zip(&r).enumerate() {
                let _ = l;
                let rows = (p.len() / dim).min(q.len() / dim).min(n);
                for row in 0..rows {
                    let (ps, qs) = (
                        &p[row * dim..(row + 1) * dim],
                        &q[row * dim..(row + 1) * dim],
                    );
                    if ps.iter().any(|v| !v.is_finite()) {
                        continue;
                    }
                    let sc = qs.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-30);
                    w = w.max(
                        ps.iter()
                            .zip(qs)
                            .fold(0f32, |m, (u, v)| m.max((u - v).abs()))
                            / sc,
                    );
                }
            }
            w
        };
        println!("\nagainst the fp32-KV reference (lower is more faithful):");
        println!("  batched   {:.3e}", dist(&a));
        println!("  per-token {:.3e}", dist(&b));
    }
}
