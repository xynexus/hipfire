//! Diagnostics, benchmarks and kernel precompilation. Long GPU ops with no
//! progress frames and no yield point.

// Handler bodies were lifted verbatim out of `main()`, so they depend on the same
// root-level imports and arch aliases (`qwen35`, `deepseek4`, `minimax`, `lfm2moe`,
// `qwen2`, `prompt_frame`) that the crate root sets up. Glob-importing the root
// keeps that dependency in one place instead of re-deriving it per module.
use crate::*;

pub(crate) fn diag(daemon_state: &mut DaemonState) {
    let (vram_free, vram_total) = daemon_state.gpu.hip.get_vram_info().unwrap_or((0, 0));
    let hip_ver = daemon_state.gpu.hip.runtime_version().unwrap_or((0, 0));
    let has_model = daemon_state.model.is_some();
    let model_arch = daemon_state
        .model
        .as_ref()
        .map(|m| match m.arch_id {
            5 => "qwen3_5",
            6 => "qwen3_5_moe",
            7 => "qwen2",
            9 => "deepseek4",
            10 => "minimax_m2",
            11 => "lfm2moe",
            14 => "nemotron_h",
            16 => "zaya",
            _ => "qwen3",
        })
        .unwrap_or("none");
    // Count pre-compiled kernels
    let kernel_dir = std::env::current_exe()
        .ok()
        .and_then(|e| {
            e.parent().map(|p| {
                p.join("kernels")
                    .join("compiled")
                    .join(&daemon_state.gpu.arch)
            })
        })
        .filter(|p| p.is_dir());
    let (hsaco_count, hash_count) = kernel_dir
        .map(|d| {
            let hsaco = std::fs::read_dir(&d)
                .map(|r| {
                    r.filter(|e| {
                        e.as_ref()
                            .ok()
                            .map(|e| e.path().extension().map(|x| x == "hsaco").unwrap_or(false))
                            .unwrap_or(false)
                    })
                    .count()
                })
                .unwrap_or(0);
            let hash = std::fs::read_dir(&d)
                .map(|r| {
                    r.filter(|e| {
                        e.as_ref()
                            .ok()
                            .map(|e| e.path().extension().map(|x| x == "hash").unwrap_or(false))
                            .unwrap_or(false)
                    })
                    .count()
                })
                .unwrap_or(0);
            (hsaco, hash)
        })
        .unwrap_or((0, 0));
    let _ = writeln!(
        daemon_state.out.sink,
        r#"{{"type":"diag","arch":"{}","hip_version":"{}.{}","vram_free_mb":{},"vram_total_mb":{},"model_loaded":{},"model_arch":"{}","kernels":{},"kernel_hashes":{}}}"#,
        daemon_state.gpu.arch,
        hip_ver.0,
        hip_ver.1,
        vram_free / (1024 * 1024),
        vram_total / (1024 * 1024),
        has_model,
        model_arch,
        hsaco_count,
        hash_count
    );
    let _ = daemon_state.out.sink.flush();
}

pub(crate) fn bench_prefill(daemon_state: &mut DaemonState, msg: &serde_json::Value) {
    // Synthetic prefill benchmark — measures forward_prefill_batch on N
    // deterministic tokens from a zeroed state. Used by `hipfire bench`
    // to produce canonical pp128/pp512/pp1024 numbers that don't depend
    // on the user's prompt tokenizing to a round number.
    let m = match daemon_state.model.as_mut() {
        Some(m) => m,
        None => {
            let _ = writeln!(
                daemon_state.out.sink,
                r#"{{"type":"error","message":"no model loaded"}}"#
            );
            let _ = daemon_state.out.sink.flush();
            return;
        }
    };
    // bench_prefill drives forward_prefill_batch / forward_scratch
    // with the single-GPU `gpu` handle — those entry points panic
    // when pp>1 because q35_scratch is None and the multi-GPU
    // tensors live on Gpus instead. Refuse cleanly per snapshot
    // review patch f253472. A pp>1 prefill bench is out of scope
    // for v1.
    if m.pp > 1 {
        let _ = writeln!(
            daemon_state.out.sink,
            r#"{{"type":"error","message":"bench_prefill requires pp=1 (multi-GPU bench not implemented)"}}"#
        );
        let _ = daemon_state.out.sink.flush();
        return;
    }
    let n = msg.get("tokens").and_then(|v| v.as_u64()).unwrap_or(128) as usize;
    // Guard physical_cap — reserve 32 slots of headroom so a subsequent
    // generate request against the loaded model still has room. We guard
    // on the *physical* buffer (not the advertised max_seq) because this
    // bench intentionally bypasses eviction to measure raw prefill.
    if n + 32 > m.physical_cap {
        let _ = writeln!(
            daemon_state.out.sink,
            r#"{{"type":"error","message":"bench_prefill tokens={} exceeds loaded physical_cap={}"}}"#,
            n, m.physical_cap
        );
        let _ = daemon_state.out.sink.flush();
        return;
    }
    // Deterministic synthetic token IDs. Skip 0 (often <pad>) and the
    // low specials by offsetting, and wrap in a 1000-wide window so the
    // embedding lookup cost stays realistic rather than hitting one
    // cache-hot row repeatedly.
    let synthetic: Vec<u32> = (0..n as u32).map(|i| 10 + (i % 1000)).collect();

    // Reset state BEFORE timing so we're measuring cold prefill, not
    // prefill-on-top-of-prior-state.
    m.active.cursor.seq_pos = 0;
    m.active.cursor.conversation_tokens.clear();
    if let Some(dn) = m
        .active
        .sequence_state
        .as_ref()
        .and_then(|s| s.recurrent_as::<qwen35::DeltaNetState>())
    {
        for s in &dn.s_matrices {
            let _ = daemon_state.gpu.hip.memset(&s.buf, 0, s.buf.size());
        }
        for s in &dn.s_scales {
            let _ = daemon_state.gpu.hip.memset(&s.buf, 0, s.buf.size());
        }
        for s in &dn.conv_states {
            let _ = daemon_state.gpu.hip.memset(&s.buf, 0, s.buf.size());
        }
    }
    // Qwen2 (arch_id=7) doesn't have a separate KV buffer — the cache
    // and the per-step scratch share `Qwen2State`. Reset its position
    // cursor here so bench_prefill measures cold prefill.
    if let Some(ref mut s) = m.qwen2_state {
        s.reset();
    }
    // MiniMax-M2 (arch_id=10): same — KV cache + scratch share
    // MiniMaxState; reset its cursor for a cold prefill bench.
    if let Some(ref mut s) = m.minimax_state {
        s.reset();
    }
    // LFM2.5-MoE (arch_id=11): same — KV + conv-state cache share
    // Lfm2MoeState; reset cursors (takes gpu) for a cold bench.
    #[cfg(feature = "arch-lfm2moe")]
    if let Some(ref mut s) = m.active.lfm2moe_state {
        let _ = s.reset(&mut daemon_state.gpu);
    }

    // Flush any residual GPU work so it doesn't bleed into the
    // measured interval, then time forward_prefill_batch + a
    // trailing device_synchronize so we capture actual GPU
    // completion (kernel launches are async by default).
    let _ = daemon_state.gpu.hip.device_synchronize();
    let t0 = Instant::now();
    let run_ok = if is_qwen35_family_arch_id(m.arch_id) {
        let config = m.q35_config.as_ref().unwrap();
        let weights = m.q35_weights.as_ref().unwrap();
        let scratch = m.q35_scratch.as_ref().unwrap();
        let ss = m
            .active
            .sequence_state
            .as_mut()
            .expect("qwen35 active state present");
        let kv = ss.kv.as_mut().expect("qwen35 active state has KV");
        let dn = ss
            .recurrent
            .as_mut()
            .expect("qwen35 active state has DeltaNet")
            .as_any_mut()
            .downcast_mut::<qwen35::DeltaNetState>()
            .expect("qwen35 active recurrent state is DeltaNetState");
        qwen35::forward_prefill_batch(
            &mut daemon_state.gpu,
            weights,
            config,
            &synthetic,
            0,
            kv,
            dn,
            scratch,
            None,
            None,
            None,
            None,
        )
        .is_ok()
    } else if m.arch_id == ARCH_ID_QWEN2 {
        // Qwen2 has no batched prefill kernel yet — per-token loop
        // mirroring the LLaMA fallback path. The loop seeds
        // position via `state.next_pos` (already reset above to 0).
        let config = m.qwen2_config.as_ref().unwrap();
        let weights = m.qwen2_weights.as_ref().unwrap();
        let state = m.qwen2_state.as_mut().unwrap();
        let mut ok = true;
        for &tok in &synthetic {
            if qwen2::forward_step(&mut daemon_state.gpu, weights, config, state, tok).is_err() {
                ok = false;
                break;
            }
        }
        ok
    } else if m.arch_id == ARCH_ID_DEEPSEEK4_FLASH {
        // DeepSeek V4 warm-pass: per-token decode_step. Saturates
        // the kernel cache (HC, indexer, compressor,
        // attention, MoE) on a short synthetic prompt
        // before any user-facing generate. Not the
        // production prefill path (that's
        // forward_prefill_batch_chunked in `generate`).
        let config = m.deepseek4_config.as_ref().unwrap();
        let weights = m.deepseek4_weights.as_ref().unwrap();
        let state = m.deepseek4_state.as_mut().unwrap();
        let mut ok = true;
        for (i, &tok) in synthetic.iter().enumerate() {
            if deepseek4::forward::decode_step(
                config,
                weights,
                state,
                &mut daemon_state.gpu,
                tok,
                i as u32,
            )
            .is_err()
            {
                ok = false;
                break;
            }
        }
        ok
    } else if m.arch_id == ARCH_ID_MINIMAX_M2 {
        // MiniMax-M2 warm-pass: per-token decode_step over the
        // synthetic prompt. Saturates the GQA + QK-norm + RoPE +
        // MoE kernel set before any user-facing generate. This
        // IS the production prefill shape (no batched kernel).
        let config = m.minimax_config.as_ref().unwrap();
        let weights = m.minimax_weights.as_ref().unwrap();
        let state = m.minimax_state.as_mut().unwrap();
        let mut ok = true;
        for (i, &tok) in synthetic.iter().enumerate() {
            if minimax::forward::decode_step(
                config,
                weights,
                state,
                &mut daemon_state.gpu,
                tok,
                i as u32,
            )
            .is_err()
            {
                ok = false;
                break;
            }
        }
        ok
    } else if cfg!(feature = "arch-lfm2moe") && m.arch_id == ARCH_ID_LFM2_MOE {
        // LFM2.5-MoE warm-pass: per-token decode_step over the
        // synthetic prompt. Saturates the conv + GQA + QK-norm +
        // RoPE + top-4 MoE kernel set before any user-facing
        // generate. This IS the production prefill shape (no
        // batched kernel).
        #[cfg(feature = "arch-lfm2moe")]
        {
            let config = m.lfm2moe_config.as_ref().unwrap();
            let weights = m.lfm2moe_weights.as_ref().unwrap();
            let state = m.active.lfm2moe_state.as_mut().unwrap();
            let mut ok = true;
            for (i, &tok) in synthetic.iter().enumerate() {
                if lfm2moe::forward::decode_step(
                    config,
                    weights,
                    state,
                    &mut daemon_state.gpu,
                    tok,
                    i as u32,
                )
                .is_err()
                {
                    ok = false;
                    break;
                }
            }
            ok
        }
        #[cfg(not(feature = "arch-lfm2moe"))]
        {
            false
        }
    } else if let Some(backend) = m.llama_backend.as_mut() {
        // LLaMA/Qwen3 (arch 0/1) warm-pass via the ServingBackend
        // (P3.2): per-token decode_step saturates the dense
        // attention/GEMV/RoPE kernel set before the first real
        // request. Logits-only (decode_loop samples in production).
        use hipfire_runtime::arch::SimpleAr;
        let mut ok = true;
        for (i, &tok) in synthetic.iter().enumerate() {
            if backend.decode_step(&mut daemon_state.gpu, tok, i).is_err() {
                ok = false;
                break;
            }
        }
        ok
    } else {
        // Unhandled arch for this prefill bench (e.g. gemma3 text/VL
        // arch 12/13, dots.ocr arch 8): no warm-pass is wired, so skip
        // rather than assume the llama path (which would unwrap None
        // and panic). Kernels JIT on the first real request.
        true
    };
    let _ = daemon_state.gpu.hip.device_synchronize();
    let elapsed = t0.elapsed().as_secs_f64();

    // Reset state AFTER measurement — we've written N KV slots and a
    // DeltaNet state that the next real request must not inherit.
    m.active.cursor.seq_pos = 0;
    m.active.cursor.conversation_tokens.clear();
    if let Some(dn) = m
        .active
        .sequence_state
        .as_ref()
        .and_then(|s| s.recurrent_as::<qwen35::DeltaNetState>())
    {
        for s in &dn.s_matrices {
            let _ = daemon_state.gpu.hip.memset(&s.buf, 0, s.buf.size());
        }
        for s in &dn.s_scales {
            let _ = daemon_state.gpu.hip.memset(&s.buf, 0, s.buf.size());
        }
        for s in &dn.conv_states {
            let _ = daemon_state.gpu.hip.memset(&s.buf, 0, s.buf.size());
        }
    }

    if run_ok {
        let tok_s = if elapsed > 0.0 {
            n as f64 / elapsed
        } else {
            0.0
        };
        let _ = writeln!(
            daemon_state.out.sink,
            r#"{{"type":"prefill_result","tokens":{},"ms":{:.2},"tok_s":{:.1}}}"#,
            n,
            elapsed * 1000.0,
            tok_s
        );
    } else {
        let _ = writeln!(
            daemon_state.out.sink,
            r#"{{"type":"error","message":"bench_prefill forward failed"}}"#
        );
    }
    let _ = daemon_state.out.sink.flush();
}

pub(crate) fn profile(daemon_state: &mut DaemonState) {
    // Precompile kernels for common configurations so we have something to profile.
    // If a model is loaded its kernels are already compiled; this fills in the rest.
    // Cover all KV modes × weight formats × head_dims to catch all kernel variants.
    #[cfg(feature = "deltanet")]
    for kv in &["q8"] {
        for wq in &["hfq4", "hfq6", "q8"] {
            for hd in &[128usize, 256] {
                let _ = daemon_state.gpu.precompile_qwen35(wq, kv, *hd);
            }
        }
    }
    let (cap, kernels) = daemon_state.gpu.profile();
    let kernels_json: Vec<String> = kernels.iter().map(|k| k.to_json()).collect();
    let _ = writeln!(
        daemon_state.out.sink,
        r#"{{"type":"profile","gpu":{},"kernels":[{}]}}"#,
        cap.to_json(),
        kernels_json.join(",")
    );
    let _ = daemon_state.out.sink.flush();
}
