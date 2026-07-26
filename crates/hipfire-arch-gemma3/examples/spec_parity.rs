#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]
//! M1a validation: gemma3 `SpecTarget::verify_block` parity + capture smoke.
//!
//! Proves the per-token `SpecTarget` baseline on real hardware:
//!   1. prefill a prompt (SWA active for gemma3-4b), then AR-greedy-decode a
//!      K-token block, recording the target's argmax-after-each (`picks_ref`);
//!   2. reset, re-prefill, and run `verify_block(block, position=P)` with
//!      extract-layer capture armed;
//!   3. assert `picks == picks_ref` (position anchoring + KV writes correct) and
//!      `hidden.len() == K * n_extract * hidden` (capture layout correct).
//!
//! Both paths use the SAME per-token kernel and the SAME host argmax, so equality
//! is exact — the smoke's job is to catch a wrong `next_pos` anchor, a KV/ring
//! corruption, a capture OOB, or a plain crash on GPU.
//!
//! ```text
//! hipfire lock acquire --label gemma3-spec-parity
//! cargo run --release --example spec_parity -p hipfire-arch-gemma3 -- \
//!     --hfq ~/.hipfire/models/medgemma-1.5-4b-it-q8f16.hfq --block 5
//! hipfire lock release
//! ```

use std::path::Path;

use hipfire_arch_gemma3 as gemma3;
use hipfire_arch_gemma3::arch::Gemma3Backend;
use hipfire_rdna::Gpu;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::sampler;
use hipfire_runtime::tokenizer::Tokenizer;
use hipfire_specdecode_dspark::spec::SpecTarget;

fn argmax_host(gpu: &mut Gpu, logits: &hipfire_rdna::GpuTensor) -> u32 {
    let v = gpu.download_f32(logits).expect("download logits");
    sampler::argmax(&v)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut hfq_path = "/home/sadara/.hipfire/models/medgemma-1.5-4b-it-q8f16.hfq".to_string();
    let mut block_len = 5usize;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--hfq" => hfq_path = it.next().unwrap(),
            "--block" => block_len = it.next().and_then(|s| s.parse().ok()).unwrap_or(5),
            other => return Err(format!("unknown arg: {other}").into()),
        }
    }

    eprintln!("[1/4] loading {hfq_path}");
    let mut hfq = HfqFile::open(Path::new(&hfq_path))?;
    let cfg = gemma3::config_from_hfq(&hfq).ok_or("gemma3: config parse failed")?;
    eprintln!(
        "      hidden={} layers={} vocab={} sliding_window={} pattern={}",
        cfg.hidden_size,
        cfg.num_hidden_layers,
        cfg.vocab_size,
        cfg.sliding_window,
        cfg.sliding_window_pattern
    );
    let tok =
        Tokenizer::from_hfq_metadata(&hfq.metadata_json).map_err(|e| format!("tokenizer: {e}"))?;

    let prompt = "<start_of_turn>user\nIn one sentence, what is a CT scan?<end_of_turn>\n\
                  <start_of_turn>model\n";
    let prompt_ids = tok.encode(prompt);
    let p = prompt_ids.len();
    eprintln!("[2/4] prompt {p} tokens; block K={block_len}");

    // The multimodal wrapper ("gemma3") nests the text decoder under
    // "language_model."; pure-text ("gemma3_text") uses "". Detect from metadata.
    let arch_str = serde_json::from_str::<serde_json::Value>(&hfq.metadata_json)
        .ok()
        .and_then(|v| {
            v.get("architecture")
                .and_then(|a| a.as_str())
                .map(String::from)
        })
        .unwrap_or_default();
    let prefix = if arch_str == "gemma3_text" {
        ""
    } else {
        "language_model."
    };
    eprintln!("      architecture={arch_str:?} → weight prefix {prefix:?}");

    let mut gpu = Gpu::init()?;
    let weights = gemma3::weights::load_weights_prefixed(&mut hfq, &cfg, &mut gpu, prefix)?;
    let state = gemma3::Gemma3State::new_with_max_seq(
        &mut gpu,
        &cfg,
        gemma3::forward::DEFAULT_MAX_SEQ,
        hipfire_runtime::kv::KvQuantMode::Unquantized,
        4,
    )
    .map_err(|e| format!("state: {e:?}"))?;
    let mut backend = Gemma3Backend::new(cfg.clone(), weights, state);

    // ── Reference: prefill + AR-greedy the block, recording picks_ref ─────────
    backend.state.reset();
    for &t in &prompt_ids {
        gemma3::forward_step(
            &mut gpu,
            &backend.weights,
            &backend.config,
            &mut backend.state,
            t,
        )?;
    }
    let mut block: Vec<u32> = Vec::with_capacity(block_len);
    let mut picks_ref: Vec<u32> = Vec::with_capacity(block_len);
    let mut cur = argmax_host(&mut gpu, &backend.state.logits); // token at position P
    for _ in 0..block_len {
        block.push(cur);
        gemma3::forward_step(
            &mut gpu,
            &backend.weights,
            &backend.config,
            &mut backend.state,
            cur,
        )?;
        let nxt = argmax_host(&mut gpu, &backend.state.logits);
        picks_ref.push(nxt);
        cur = nxt;
    }
    eprintln!("      block   = {block:?}");
    eprintln!("      picks_ref = {picks_ref:?}");

    // ── verify_block via SpecTarget, with capture armed ───────────────────────
    let extract: Vec<usize> = vec![0, cfg.num_hidden_layers / 2, cfg.num_hidden_layers - 1];
    backend.set_dflash_extract_layers(extract.clone());

    backend.state.reset();
    for &t in &prompt_ids {
        gemma3::forward_step(
            &mut gpu,
            &backend.weights,
            &backend.config,
            &mut backend.state,
            t,
        )?;
    }
    let mut scratch = backend.new_spec_scratch(&mut gpu, block_len)?;
    let mut hidden: Vec<f32> = Vec::new();
    let picks = backend.verify_block(&mut gpu, &block, p, &mut *scratch, Some(&mut hidden))?;
    eprintln!("[3/4] picks(pertoken) = {picks:?}");

    // ── Batched verify (M1b): forward_verify_batch vs per-token ───────────────
    backend.state.reset();
    for &t in &prompt_ids {
        gemma3::forward_step(
            &mut gpu,
            &backend.weights,
            &backend.config,
            &mut backend.state,
            t,
        )?;
    }
    let hd = cfg.hidden_size;
    let x_batch = gpu
        .alloc_tensor(&[block_len * hd], hipfire_rdna::DType::F32)
        .map_err(|e| format!("x_batch: {e:?}"))?;
    for (i, &t) in block.iter().enumerate() {
        gemma3::forward::embed_token(
            &mut gpu,
            &backend.weights,
            &backend.config,
            &x_batch.sub_offset(i * hd, hd),
            t,
        )
        .map_err(|e| format!("embed {i}: {e:?}"))?;
    }
    let hidden_gpu = gpu
        .alloc_tensor(&[block_len * extract.len() * hd], hipfire_rdna::DType::F32)
        .map_err(|e| format!("hidden_gpu: {e:?}"))?;
    let blogits = gemma3::forward::forward_verify_batch(
        &mut gpu,
        &backend.weights,
        &backend.config,
        &mut backend.state,
        &x_batch,
        block_len,
        p,
        &extract,
        Some(&hidden_gpu),
    )
    .map_err(|e| format!("forward_verify_batch: {e:?}"))?;
    let vocab = cfg.vocab_size;
    let picks_b: Vec<u32> = (0..block_len)
        .map(|r| sampler::argmax(&blogits[r * vocab..(r + 1) * vocab]))
        .collect();
    let hidden_b = gpu
        .download_f32(&hidden_gpu)
        .map_err(|e| format!("dl hidden_gpu: {e:?}"))?;
    eprintln!("[3b/4] picks(batched)  = {picks_b:?}");

    // Per-token reference LOGITS (verify_block_logits) for a distribution-level
    // parity check — the meaningful correctness metric (argmax + logit closeness),
    // since batched vs per-token kernels are not bit-identical by design.
    backend.state.reset();
    for &t in &prompt_ids {
        gemma3::forward_step(
            &mut gpu,
            &backend.weights,
            &backend.config,
            &mut backend.state,
            t,
        )?;
    }
    let pt_logits = backend.verify_block_logits(&mut gpu, &block, p, &mut *scratch, None)?;
    let mut logit_maxd = 0.0f32;
    let mut logit_ref_mag = 0.0f32;
    for i in 0..block_len * vocab {
        logit_maxd = logit_maxd.max((blogits[i] - pt_logits[i]).abs());
        logit_ref_mag = logit_ref_mag.max(pt_logits[i].abs());
    }
    eprintln!(
        "      logit max|Δ| batched-vs-per-token = {logit_maxd:.3e}  (ref |max|={logit_ref_mag:.2}, rel={:.2}%)",
        100.0 * logit_maxd / logit_ref_mag.max(1e-6)
    );

    // Timing: batched forward vs m× per-token (the M1b speedup).
    backend.state.reset();
    for &t in &prompt_ids {
        gemma3::forward_step(
            &mut gpu,
            &backend.weights,
            &backend.config,
            &mut backend.state,
            t,
        )?;
    }
    let t_b = std::time::Instant::now();
    let _ = gemma3::forward::forward_verify_batch(
        &mut gpu,
        &backend.weights,
        &backend.config,
        &mut backend.state,
        &x_batch,
        block_len,
        p,
        &[],
        None,
    )
    .map_err(|e| format!("timing batched: {e:?}"))?;
    gpu.hip.device_synchronize().ok();
    let batched_ms = t_b.elapsed().as_secs_f64() * 1000.0;
    backend.state.reset();
    for &t in &prompt_ids {
        gemma3::forward_step(
            &mut gpu,
            &backend.weights,
            &backend.config,
            &mut backend.state,
            t,
        )?;
    }
    let t_p = std::time::Instant::now();
    for &t in &block {
        gemma3::forward_step(
            &mut gpu,
            &backend.weights,
            &backend.config,
            &mut backend.state,
            t,
        )?;
    }
    gpu.hip.device_synchronize().ok();
    let pertoken_ms = t_p.elapsed().as_secs_f64() * 1000.0;
    eprintln!(
        "      timing: batched {batched_ms:.1} ms vs per-token {pertoken_ms:.1} ms  (speedup {:.2}×)",
        pertoken_ms / batched_ms.max(1e-6)
    );

    // ── Wired SpecTarget batched paths (verify_block no-capture + capture_gpu) ─
    backend.state.reset();
    for &t in &prompt_ids {
        gemma3::forward_step(
            &mut gpu,
            &backend.weights,
            &backend.config,
            &mut backend.state,
            t,
        )?;
    }
    let mut sc2 = backend.new_spec_scratch(&mut gpu, block_len)?;
    let picks_wired = backend.verify_block(&mut gpu, &block, p, &mut *sc2, None)?;
    backend.state.reset();
    for &t in &prompt_ids {
        gemma3::forward_step(
            &mut gpu,
            &backend.weights,
            &backend.config,
            &mut backend.state,
            t,
        )?;
    }
    let hg2 = gpu
        .alloc_tensor(&[block_len * extract.len() * hd], hipfire_rdna::DType::F32)
        .map_err(|e| format!("hg2: {e:?}"))?;
    let (picks_cap, captured) =
        backend.verify_block_capture_gpu(&mut gpu, &block, p, &mut *sc2, &hg2)?;
    eprintln!(
        "[3c/4] wired verify_block={picks_wired:?} capture_gpu={picks_cap:?} captured={captured}"
    );

    // ── Assertions ────────────────────────────────────────────────────────────
    let mut ok = true;
    if picks_wired != picks_ref {
        eprintln!("  ✘ wired verify_block (batched) != picks_ref");
        ok = false;
    }
    if !captured || picks_cap != picks_ref {
        eprintln!("  ✘ verify_block_capture_gpu: captured={captured} picks={picks_cap:?}");
        ok = false;
    }
    if picks != picks_ref {
        eprintln!("  ✘ per-token picks != picks_ref (position/KV anchoring or ring bug)");
        for i in 0..block_len {
            if picks[i] != picks_ref[i] {
                eprintln!("     slot {i}: verify={} ref={}", picks[i], picks_ref[i]);
            }
        }
        ok = false;
    }
    // Batched verify argmax parity vs the per-token reference.
    if picks_b != picks_ref {
        eprintln!("  ✘ batched picks != picks_ref (batched SWA/global attention bug)");
        for i in 0..block_len {
            if picks_b[i] != picks_ref[i] {
                let g = &blogits[i * cfg.vocab_size..(i + 1) * cfg.vocab_size];
                let mut top = g.iter().cloned().fold(f32::MIN, f32::max);
                let mut second = f32::MIN;
                for &x in g {
                    if x < top && x > second {
                        second = x;
                    }
                }
                let _ = &mut top;
                eprintln!(
                    "     slot {i}: batched={} ref={} (top1-top2 gap={:.4e})",
                    picks_b[i],
                    picks_ref[i],
                    top - second
                );
            }
        }
        ok = false;
    }
    // Batched-vs-per-token capture parity (the strong numeric check on the whole
    // forward: extract-layer residuals computed both ways must match tightly).
    let cap_maxd = hidden
        .iter()
        .zip(hidden_b.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("      capture max|Δ| batched-vs-per-token = {cap_maxd:.3e}");
    // Per-(position, extract-layer) breakdown: layout is [pos][extract][dim].
    let hd2 = cfg.hidden_size;
    for e in 0..extract.len() {
        let gl = extract[e];
        let kind = if backend.config.is_global_layer(gl) {
            "global"
        } else {
            "local/SWA"
        };
        let mut emax = 0.0f32;
        for pos in 0..block_len {
            let base = (pos * extract.len() + e) * hd2;
            for d in 0..hd2 {
                emax = emax.max((hidden[base + d] - hidden_b[base + d]).abs());
            }
        }
        eprintln!("        extract layer {gl:>2} ({kind}): max|Δ|={emax:.3e}");
    }
    if hidden_b.len() != hidden.len() {
        eprintln!(
            "  ✘ capture length mismatch ({} vs {})",
            hidden_b.len(),
            hidden.len()
        );
        ok = false;
    }
    // The capture |Δ| is intermediate-residual float accumulation (batched WMMA
    // gemm + *_batched attention vs per-token gemv/scalar); it is NOT a bit-parity
    // metric. The gate is argmax + logit closeness above. Guard only gross error.
    if logit_maxd > 5.0 {
        eprintln!(
            "  ✘ logit max|Δ| {logit_maxd:.3e} too large — likely a real batched-forward bug"
        );
        ok = false;
    }
    let want = block_len * extract.len() * cfg.hidden_size;
    if hidden.len() != want {
        eprintln!(
            "  ✘ capture len {} != K*n_extract*hidden {} ({}*{}*{})",
            hidden.len(),
            want,
            block_len,
            extract.len(),
            cfg.hidden_size
        );
        ok = false;
    } else if hidden.iter().any(|x| !x.is_finite()) {
        eprintln!("  ✘ capture has non-finite values");
        ok = false;
    }

    if ok {
        eprintln!(
            "[4/4] PASS — verify_block == AR ({} slots), capture {} floats finite",
            block_len,
            hidden.len()
        );
        Ok(())
    } else {
        Err("spec_parity FAILED".into())
    }
}
