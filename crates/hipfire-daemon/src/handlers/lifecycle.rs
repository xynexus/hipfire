//! Model load / reset / unload, and the worker slot swap.
//!
//! `load` is by far the largest handler and carries the teardown ordering that
//! matters: the PFlash drafter must be released before `unload_model`, because
//! `free_tensor` only queues into the pool and it is `unload_model -> drain_pool`
//! that actually `hipFree`s.
//!
//! `unload_worker` promotes an arbitrary remaining resident worker into the
//! active slot. There is no eviction policy anywhere here — the only capacity
//! mechanism is the reservation ballast in `DaemonState`.

// Handler bodies were lifted verbatim out of `main()`, so they depend on the same
// root-level imports and arch aliases that the crate root sets up.
use crate::*;

pub(crate) fn load(
    daemon_state: &mut DaemonState,
    msg: &serde_json::Value,
    protocol_load: &Option<hipfire_model::ModelLoadRequest>,
) {
    // A steer session is process-global and outlives the model it was
    // captured/applied against; drop it before swapping models so a
    // stale apply can't perturb the freshly-loaded one.
    hipfire_steer::clear();
    let requested_worker_id = message_worker_id(&msg);
    // Unload previous if any. PFlash drafter goes first so
    // its tensors join the pool before unload_model drains
    // it -- otherwise free_tensor would queue them into the
    // pool just-emptied by drain_pool with no follow-up
    // drain, leaving drafter VRAM resident across the next
    // load (the explicit "unload" handler has the same
    // ordering for the same reason).
    if requested_worker_id == daemon_state.active_worker_id {
        daemon_state
            .generic_state_arena
            .release_worker(&requested_worker_id);
        if let Some(mut pf) = daemon_state.pflash_state.take() {
            if let Some(mut dg) = daemon_state.pflash_drafter_gpu.take() {
                dg.bind_thread_or_warn();
                pf.unload_drafter(&mut dg); // sibling-device drafter: free on its own handle, then drop
                daemon_state.gpu.bind_thread_or_warn();
            } else {
                pf.unload_drafter(&mut daemon_state.gpu);
            }
        }
        daemon_state.pflash_cfg = None;
        if let Some(m) = daemon_state.model.take() {
            unload_model(m, &mut daemon_state.gpu);
        }
        daemon_state
            .resource_reservations
            .remove_worker(&requested_worker_id);
    } else {
        if let Err(e) = park_active_model(
            &mut daemon_state.model,
            &mut daemon_state.gpu,
            &daemon_state.active_worker_id,
            &mut daemon_state.resident_models,
        ) {
            write_error(
                &mut daemon_state.out.sink,
                "",
                &format!("worker switch failed: {e}"),
            );
            let _ = daemon_state.out.sink.flush();
            return;
        }
        daemon_state.active_worker_id = requested_worker_id.clone();
    }
    if let Some(m) = daemon_state.resident_models.remove(&requested_worker_id) {
        daemon_state
            .generic_state_arena
            .release_worker(&requested_worker_id);
        unload_model(m, &mut daemon_state.gpu);
        daemon_state
            .resource_reservations
            .remove_worker(&requested_worker_id);
    }
    daemon_state.dummy_model = None;

    let path = protocol_load
        .as_ref()
        .map(|req| req.model.as_str())
        .or_else(|| msg.get("model").and_then(|v| v.as_str()))
        .unwrap_or("");
    let dummy_requested = msg
        .get("params")
        .and_then(|p| p.get("dummy_model"))
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    if dummy_requested {
        daemon_state.dummy_model = Some(DummyModelState::default());
        if let Err(err) = daemon_state
            .resource_reservations
            .reacquire_placeholders(&mut daemon_state.gpu)
        {
            write_error(
                &mut daemon_state.out.sink,
                "",
                &format!("dummy load resource reservation failed: {err}"),
            );
            let _ = daemon_state.out.sink.flush();
            return;
        }
        tracing::info!(
            daemon_state.model = "hipfire:dummy",
            arch = "qwen35_dummy",
            "dummy model loaded"
        );
        let line = serde_json::json!({
            "type": "loaded",
            "worker_key_id": requested_worker_id,
            "arch": "qwen35_dummy",
            "cache_capable": false,
            "dim": 16,
            "layers": 1,
            "vocab": 1024,
            "vl": false,
        });
        daemon_state.out.emit(line);
        return;
    }

    let max_seq = protocol_load
        .as_ref()
        .map(|req| req.params.max_seq as usize)
        .or_else(|| {
            msg.get("params")
                .and_then(|p| p.get("max_seq"))
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
        })
        .unwrap_or(8192);
    let requested_physical_cap = protocol_load
        .as_ref()
        .and_then(|req| req.params.physical_cap.map(|v| v as usize))
        .or_else(|| {
            msg.get("params")
                .and_then(|p| p.get("physical_cap"))
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
        })
        .filter(|v| *v > 0);
    let raw_dflash_mode = msg
        .get("params")
        .and_then(|p| p.get("dflash_mode"))
        .and_then(|v| v.as_str());
    let raw_draft_param = msg
        .get("params")
        .and_then(|p| p.get("draft"))
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty());
    // Optional DFlash draft model path. When supplied AND the target
    // is a Qwen3.5 arch (5 or 6), we load draft weights + scratch
    // alongside the target and the temp=0 generate fast path routes
    // through `spec_step_dflash` for the 1.7-2.5× speedup on the
    // 27B target. Non-matching archs / missing draft file are
    // logged but don't fail the load.
    //
    // `dflash_mode=off` is a hard daemon-side override: even if a
    // draft path was passed, skip the load. CLI-side gating is the
    // primary path (saves the wire round-trip for the draft path
    // string), but this guard makes the flag durable when the
    // daemon is driven by a non-hipfire-CLI client.
    let dflash_mode = protocol_load
        .as_ref()
        .and_then(|req| req.params.dflash_mode.as_deref())
        .or(raw_dflash_mode)
        .unwrap_or("auto");
    let raw_draft = protocol_load
        .as_ref()
        .and_then(|req| req.params.draft.as_deref())
        .or(raw_draft_param)
        .filter(|s| !s.is_empty());
    let draft_path = if dflash_mode == "off" {
        if raw_draft.is_some() {
            eprintln!(
                "[hipfire-daemon] dflash_mode=off — skipping draft load ({})",
                raw_draft.unwrap()
            );
        }
        None
    } else {
        raw_draft.map(|s| s.to_string())
    };
    let kv_mode_override = protocol_load
        .as_ref()
        .and_then(|req| req.params.kv_cache.as_deref())
        .or_else(|| {
            msg.get("params")
                .and_then(|p| p.get("kv_mode"))
                .and_then(|v| v.as_str())
        })
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string());

    // MTP speculative decode config. `mtp_mode` gates weight
    // discovery at load time (off=skip, on=error-if-missing,
    // auto=scan+log). `mtp_k` sets the draft window size.
    let mtp_mode = msg
        .get("params")
        .and_then(|p| p.get("mtp_mode"))
        .and_then(|v| v.as_str())
        .unwrap_or("auto")
        .to_string();
    // Default K=2: empirically the sweet spot for Qwen3.5 MTP
    // (0.8B: τ=1.66 @ K=2 vs 1.62 @ K=3/4, and best tok/s — higher
    // K just wastes draft forwards that acceptance tapering rejects;
    // see NEXT-STEPS Phase B4). Overridable per-load via mtp_k.
    let mtp_k: usize = msg
        .get("params")
        .and_then(|p| p.get("mtp_k"))
        .and_then(|v| v.as_u64())
        .unwrap_or(2) as usize;

    // 0.1.7-alpha: DFlash tuning knobs forwarded from the CLI.
    // `adaptive_b` matches dflash_spec_demo's --adaptive-b default.
    // Accepted here; the generate loop will honor it in the
    // 0.1.7-stable release where we port the demo's outer τ-window
    // trip-wire (below 2.5 → shrink block to 8).
    let _adaptive_b = msg
        .get("params")
        .and_then(|p| p.get("dflash_adaptive_b"))
        .and_then(|v| v.as_bool())
        .unwrap_or(true);

    // 0.1.7: TriAttention / CASK eviction protocol fields. When
    // `cask_sidecar` is set, `load_model` sizes the KV cache to a
    // *physical_cap* (budget+beta+safety, clamped to max_seq) instead
    // of the full max_seq, and wires an `Eviction` policy that the
    // generate loop calls after every prefill-chunk / decode-forward.
    // That decouples advertised context length from VRAM footprint —
    // a 128K max_seq can run in ~1K-slot physical buffer when the
    // operator opts in.
    let cask_sidecar = protocol_load
        .as_ref()
        .and_then(|req| req.params.cask_sidecar.as_deref())
        .or_else(|| {
            msg.get("params")
                .and_then(|p| p.get("cask_sidecar"))
                .and_then(|v| v.as_str())
        })
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string());
    let cask_enabled = msg
        .get("params")
        .and_then(|p| p.get("cask"))
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let cask_budget = msg
        .get("params")
        .and_then(|p| p.get("cask_budget"))
        .and_then(|v| v.as_u64())
        .unwrap_or(512) as usize;
    let cask_beta = msg
        .get("params")
        .and_then(|p| p.get("cask_beta"))
        .and_then(|v| v.as_u64())
        .unwrap_or(128) as usize;
    let cask_core_frac = msg
        .get("params")
        .and_then(|p| p.get("cask_core_frac"))
        .and_then(|v| v.as_f64())
        .unwrap_or(0.5) as f32;
    let cask_fold_m = msg
        .get("params")
        .and_then(|p| p.get("cask_fold_m"))
        .and_then(|v| v.as_u64())
        .unwrap_or(2) as usize;
    // Known-broken combo guard: CASK m-folding + DFlash spec decode
    // degenerates into single-token loops after the first eviction
    // (the m-folded synthetic K/V rows are off the draft's trained
    // hidden-state distribution). Until that's fixed at the library
    // level, downgrade m-folding to plain TriAttention drop-eviction
    // when a draft is attached. User's context window + eviction
    // cadence still work; just the fold step is skipped.
    let cask_m_folding_effective = if cask_enabled && draft_path.is_some() {
        eprintln!(
            "[hipfire-daemon] cask:true + draft: both set — downgrading to plain TriAttention drop-eviction (CASK m-fold + DFlash is a known-broken combo; see feedback_cask_mfold_dflash_broken.md)",
        );
        false
    } else {
        cask_enabled
    };
    let cask = CaskConfig {
        sidecar: cask_sidecar,
        cask_m_folding: cask_m_folding_effective,
        budget: cask_budget,
        beta: cask_beta,
        core_frac: cask_core_frac,
        fold_m: cask_fold_m,
    };

    // MMQ per-weight screening (#87): detect outlier rows that
    // cause Q8_1 precision loss and fall back to WMMA for those
    // weights. Disabled by default; enable with mmq_screen=true
    // (or HIPFIRE_MMQ_SCREEN=1) when adding new quant formats.
    if let Some(v) = msg
        .get("params")
        .and_then(|p| p.get("mmq_screen"))
        .and_then(|v| v.as_bool())
    {
        daemon_state.gpu.mmq_screen = v;
    }
    if let Some(v) = msg
        .get("params")
        .and_then(|p| p.get("mmq_screen_threshold"))
        .and_then(|v| v.as_f64())
    {
        daemon_state.gpu.mmq_screen_threshold = v as f32;
    }

    // ── PFlash load-time params (Phase 4.0 #93) ──────────────
    //
    // Parse compression knobs per PRD §5.3.2. None of these
    // affect the target load itself; they only configure the
    // optional drafter that PFlash uses for prompt scoring.
    // Drafter loading happens AFTER target load succeeds so
    // we can use the target's tokenizer for the compat check.
    let pflash_mode_str = msg
        .get("params")
        .and_then(|p| p.get("prefill_compression"))
        .and_then(|v| v.as_str())
        .unwrap_or("off")
        .to_string();
    let pflash_threshold = msg
        .get("params")
        .and_then(|p| p.get("prefill_threshold"))
        .and_then(|v| v.as_u64())
        .unwrap_or(32768) as usize;
    let pflash_keep_ratio = msg
        .get("params")
        .and_then(|p| p.get("prefill_keep_ratio"))
        .and_then(|v| v.as_f64())
        .unwrap_or(0.05) as f32;
    let pflash_alpha = msg
        .get("params")
        .and_then(|p| p.get("prefill_alpha"))
        .and_then(|v| v.as_f64())
        .unwrap_or(0.85) as f32;
    let pflash_min_keep = msg
        .get("params")
        .and_then(|p| p.get("prefill_min_keep"))
        .and_then(|v| v.as_u64())
        .unwrap_or(2048) as usize;
    let pflash_sink = msg
        .get("params")
        .and_then(|p| p.get("prefill_sink"))
        .and_then(|v| v.as_u64())
        .unwrap_or(256) as usize;
    let pflash_recent = msg
        .get("params")
        .and_then(|p| p.get("prefill_recent"))
        .and_then(|v| v.as_u64())
        .unwrap_or(1024) as usize;
    let pflash_block = msg
        .get("params")
        .and_then(|p| p.get("prefill_block"))
        .and_then(|v| v.as_u64())
        .unwrap_or(128) as usize;
    let pflash_drafter = msg
        .get("params")
        .and_then(|p| p.get("prefill_drafter"))
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string());
    // -1 = drafter shares the target gpu (default). >=0 routes
    // the drafter to that HIP device for hetero compress.
    let pflash_drafter_device: i32 = msg
        .get("params")
        .and_then(|p| p.get("prefill_drafter_device"))
        .and_then(|v| v.as_i64())
        .unwrap_or(-1) as i32;
    let pflash_profile = msg
        .get("params")
        .and_then(|p| p.get("prefill_profile"))
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let pflash_sparse_threshold = msg
        .get("params")
        .and_then(|p| p.get("prefill_sparse_threshold"))
        .and_then(|v| v.as_u64())
        .unwrap_or(32768) as usize;

    // Validate load-time PFlash params before they reach
    // PflashConfig + load_drafter. Same range rules the
    // per-request override path uses; without these, a
    // bad load-time value would silently be accepted and
    // panic the daemon at the first generate request.
    let pflash_load_err: Option<String> = if !(pflash_keep_ratio > 0.0 && pflash_keep_ratio <= 1.0)
    {
        Some(format!(
            "prefill_keep_ratio={pflash_keep_ratio} not in (0, 1]"
        ))
    } else if pflash_block == 0 {
        Some("prefill_block must be > 0".to_string())
    } else {
        None
    };

    // Pipeline-parallel degree (Stage 7 of #58). Default 1 =
    // single-GPU (no behavior change). pp > 1 routes through
    // Gpus + *_multi paths and refuses VL / DFlash / CASK /
    // PFlash at load time. v1 supports Qwen3.5 dense + MoE
    // only — see load_model_pp for the arch_id check.
    let pp = msg
        .get("params")
        .and_then(|p| p.get("pp"))
        .and_then(|v| v.as_u64())
        .unwrap_or(1) as usize;
    if pp > 1 {
        if draft_path.is_some() && std::env::var("HIPFIRE_PP_DFLASH").ok().as_deref() != Some("1") {
            let _ = writeln!(
                daemon_state.out.sink,
                r#"{{"type":"error","message":"DFlash speculative decode requires pp=1 in v1 (set HIPFIRE_PP_DFLASH=1 to opt into the experimental pp>1 PRD path; note PR2-4 of docs/plans/hetero-pflash-dflash.prd are not yet implemented — the load message will accept but generate will not run cross-card spec-decode). See issue #58 v1.1 roadmap."}}"#
            );
            let _ = daemon_state.out.sink.flush();
            return;
        }
        if cask.sidecar.is_some() {
            let _ = writeln!(
                daemon_state.out.sink,
                r#"{{"type":"error","message":"CASK / TriAttention eviction requires pp=1 in v1; see issue #58 v1.1 roadmap"}}"#
            );
            let _ = daemon_state.out.sink.flush();
            return;
        }
        if (pflash_drafter.is_some() || pflash_mode_str != "off")
            && std::env::var("HIPFIRE_PP_PFLASH").ok().as_deref() != Some("1")
        {
            let _ = writeln!(
                daemon_state.out.sink,
                r#"{{"type":"error","message":"PFlash prefill compression requires pp=1 in v1 (set HIPFIRE_PP_PFLASH=1 to opt into the experimental pp>1 PoC); see issue #58 v1.1 roadmap"}}"#
            );
            let _ = daemon_state.out.sink.flush();
            return;
        }
    }

    let state_quant_override = msg
        .get("params")
        .and_then(|p| p.get("state_quant"))
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string());

    // Stream per-layer load progress to the client on the framed
    // stdout channel (see `emit_load_progress`). Loaders call
    // `load_progress::report`, which this sink turns into a
    // `load_progress` frame. Installed only for the duration of this
    // load and cleared right after the match, so no stray frames
    // leak into later ops. `load_model` runs synchronously on this
    // thread, so the sink writes interleave safely with our own
    // stdout writes (each is a whole locked line).
    hipfire_runtime::load_progress::set_sink(Some(Box::new(|current, total, phase| {
        emit_load_progress(current, total, phase)
    })));
    let _qwen_residency_env =
        qwen_residency_load_env(protocol_load.as_ref().map(|req| &req.params));
    let planned_resource_usage = daemon_state
        .resource_reservations
        .planned_usage_for_load(path, protocol_load.as_ref().map(|req| &req.params));
    if let Err(err) = daemon_state
        .resource_reservations
        .release_placeholders(&mut daemon_state.gpu)
    {
        hipfire_runtime::load_progress::set_sink(None);
        write_error(
            &mut daemon_state.out.sink,
            "",
            &format!("resource reservation release failed before load: {err}"),
        );
        let _ = daemon_state.out.sink.flush();
        return;
    }
    let load_result = load_model(
        path,
        max_seq,
        requested_physical_cap,
        draft_path.as_deref(),
        kv_mode_override.as_deref(),
        state_quant_override.as_deref(),
        &cask,
        pp,
        &mut daemon_state.gpu,
    );
    hipfire_runtime::load_progress::set_sink(None);
    match load_result {
        Ok(mut m) => {
            daemon_state
                .resource_reservations
                .set_worker_usage(requested_worker_id.clone(), planned_resource_usage);
            if let Err(err) = daemon_state
                .resource_reservations
                .reacquire_placeholders(&mut daemon_state.gpu)
            {
                daemon_state
                    .resource_reservations
                    .remove_worker(&requested_worker_id);
                unload_model(m, &mut daemon_state.gpu);
                let _ = daemon_state
                    .resource_reservations
                    .reacquire_placeholders(&mut daemon_state.gpu);
                write_error(
                    &mut daemon_state.out.sink,
                    "",
                    &format!("resource reservation reacquire failed after load: {err}"),
                );
                let _ = daemon_state.out.sink.flush();
                return;
            }
            let arch = m.registered_backend.as_ref().map_or_else(
                || match m.arch_id {
                    5 => "qwen3_5",
                    6 => "qwen3_5_moe",
                    7 => "qwen2",
                    8 => "dots-ocr",
                    9 => "deepseek4",
                    10 => "minimax_m2",
                    11 => "lfm2moe",
                    12 => "gemma3",
                    13 => "gemma3_vl",
                    14 => "nemotron_h",
                    15 => "mamba2",
                    16 => "zaya",
                    ARCH_ID_EMBEDDINGGEMMA => "embeddinggemma",
                    _ => "qwen3",
                },
                |loaded| loaded.family,
            );
            let vl =
                m.vision_config.is_some() || m.dots_ocr_config.is_some() || m.gemma3_vl.is_some();
            let (dim, layers, vocab) = if let Some(ref loaded) = m.registered_backend {
                (
                    loaded.shape.hidden_size,
                    loaded.shape.num_layers,
                    loaded.shape.vocab_size,
                )
            } else if let Some(ref b) = m.gemma3_vl {
                (
                    b.text_cfg.hidden_size,
                    b.text_cfg.num_hidden_layers,
                    b.text_cfg.vocab_size,
                )
            } else if let Some(ref e) = m.embeddinggemma {
                (
                    e.config.max_output_dim(),
                    e.config.num_hidden_layers,
                    e.config.vocab_size,
                )
            } else if let Some(ref b) = m.gemma3_text {
                (
                    b.config.hidden_size,
                    b.config.num_hidden_layers,
                    b.config.vocab_size,
                )
            } else if let Some(ref c) = m.q35_config {
                (c.dim, c.n_layers, c.vocab_size)
            } else if let Some(ref c) = m.llama_config {
                (c.dim, c.n_layers, c.vocab_size)
            } else if let Some(ref b) = m.nemotron_backend {
                let c = b.config();
                (c.hidden_size, c.num_layers, c.vocab_size)
            } else if let Some(ref c) = m.qwen2_config {
                (c.hidden_size, c.num_hidden_layers, c.vocab_size)
            } else if let Some(ref c) = m.dots_ocr_config {
                (
                    c.text.hidden_size,
                    c.text.num_hidden_layers,
                    c.text.vocab_size,
                )
            } else if let Some(ref c) = m.minimax_config {
                (c.hidden_size, c.num_hidden_layers, c.vocab_size)
            } else if let Some((d, l, v)) = {
                #[cfg(feature = "arch-lfm2moe")]
                {
                    m.lfm2moe_config
                        .as_ref()
                        .map(|c| (c.hidden_size, c.num_hidden_layers, c.vocab_size))
                }
                #[cfg(not(feature = "arch-lfm2moe"))]
                {
                    None::<(usize, usize, usize)>
                }
            } {
                (d, l, v)
            } else {
                (0, 0, 0)
            };

            // Apply MTP config from load-message params.
            m.mtp_mode = mtp_mode;
            m.mtp_k = mtp_k;
            // Detect whether MTP weights are present in the loaded
            // model. DeepSeek V4: mtp_layer in weights. Qwen3.5/3.6
            // (arch 5/6): a bundled `-mq4+mtp.hfq` trailer or a
            // sibling `.mtp.hfq` sidecar. Used by mtp_mode to decide
            // whether to drive the MTP spec-decode path at generate.
            let qwen35_mtp_present = is_qwen35_family_arch_id(m.arch_id) && {
                let bundled = hipfire_arch_qwen35::mtp_head::detect_bundled_mtp_offset(
                    std::path::Path::new(&m.model_path),
                )
                .ok()
                .flatten()
                .is_some();
                let sidecar =
                    std::path::Path::new(&m.model_path.replace(".hfq", ".mtp.hfq")).exists();
                bundled || sidecar
            };
            m.mtp_weights_present = qwen35_mtp_present
                || m.deepseek4_weights
                    .as_ref()
                    .and_then(|w| w.mtp_layer.as_ref())
                    .is_some();

            // Auto-apply a bundled abliteration/LoRA adapter if this
            // model carries one (a `--merge-lora` artifact: the adapter
            // HFQM section + a trailer appended to the `.hfq`). Additive
            // and best-effort — a plain model has no trailer, so this is
            // a 16-byte read + magic miss. The load arm already cleared
            // the steer session up top, so this seeds a fresh apply
            // stack that lives for the model's lifetime.
            match hipfire_lora_hfq::read_bundled_lora(std::path::Path::new(&m.model_path)) {
                Ok(Some(adapter)) => {
                    let (id, n) = (adapter.id.clone(), adapter.deltas.len());
                    match hipfire_steer::load_lora_adapter(&adapter) {
                        Ok(()) => eprintln!(
                            "[hipfire-daemon] auto-applied bundled LoRA '{id}' ({n} deltas, scale {:.2})",
                            adapter.scale
                        ),
                        Err(e) => eprintln!(
                            "[hipfire-daemon] bundled LoRA '{id}' load failed: {e}"
                        ),
                    }
                }
                Ok(None) => {}
                Err(e) => {
                    eprintln!("[hipfire-daemon] bundled LoRA probe failed: {e}")
                }
            }

            // ── Optional DPM stabilization (perf instrumentation) ──
            //
            // Pins the GPU at high sclk/mclk so the first `generate`
            // request doesn't pay the 1-10s DPM ramp from idle. Same
            // `HIPFIRE_DPM_WARMUP_SECS` env the in-process bench tools
            // honor (`bench_qwen35_speed`, `dflash_spec_demo`,
            // `bench_stream_overlap`); see
            // `crates/hipfire-rdna/src/dispatch.rs::dpm_warmup` and
            // `docs/methodology/perf-benchmarking.md`.
            //
            // Runs AFTER weight upload but BEFORE the `loaded` ack so
            // the contract becomes "loaded means daemon is fully ready
            // including DPM-pinned." Critical for probe-side timing:
            // if warmup ran AFTER the ack, the probe would receive
            // `loaded`, immediately send `generate`, and the daemon
            // (still warming up in this handler) wouldn't process the
            // generate until warmup finished — folding the warmup
            // into the probe-measured TTFT and breaking
            // `tok_s = total_tokens / wall_ms`. With warmup before the
            // ack, the probe sees `loaded` only when the daemon is
            // truly ready, and TTFT measures real prefill alone.
            //
            // Default OFF (production daemon load latency unchanged).
            if let Ok(secs_str) = std::env::var("HIPFIRE_DPM_WARMUP_SECS") {
                if let Ok(secs) = secs_str.parse::<f32>() {
                    if secs > 0.0 {
                        if let Err(e) = daemon_state.gpu.dpm_warmup(secs) {
                            eprintln!("[daemon] dpm_warmup failed (non-fatal): {e:?}");
                        }
                    }
                }
            }

            let model_worker =
                model_worker_runtime_view_json(&loaded_model_worker_runtime_view(&m));
            let cache_capable =
                m.arch_id == ARCH_ID_DEEPSEEK4_FLASH || is_qwen35_family_arch_id(m.arch_id);
            let _ = writeln!(
                daemon_state.out.sink,
                "{}",
                serde_json::json!({
                    "type": "loaded",
                    "worker_key_id": requested_worker_id,
                    "arch": arch,
                    "cache_capable": cache_capable,
                    "dim": dim,
                    "layers": layers,
                    "vocab": vocab,
                    "vl": vl,
                    "model_worker": model_worker,
                })
            );

            // ── PFlash drafter load (Phase 4.0) ──────────────
            //
            // Only attempt when mode != off AND a drafter path
            // was provided. Failures here are NON-FATAL: log
            // the reason and return with PFlash disabled so
            // the operator gets a clear "model is up, but
            // compression isn't" signal rather than losing
            // the entire session.
            if let Some(ref pf_drafter_path) = pflash_drafter {
                if pflash_mode_str != "off" {
                    if let Some(ref reason) = pflash_load_err {
                        let _ = writeln!(
                            daemon_state.out.sink,
                            r#"{{"type":"pflash_load_failed","reason":"invalid load param: {}"}}"#,
                            reason.replace('"', "'")
                        );
                        let _ = daemon_state.out.sink.flush();
                        daemon_state.model = Some(m);
                        return;
                    }
                    let pf_cfg = hipfire_arch_qwen35::pflash::PflashConfig {
                        mode: hipfire_arch_qwen35::pflash::PflashMode::parse(&pflash_mode_str)
                            .unwrap_or(hipfire_arch_qwen35::pflash::PflashMode::Off),
                        threshold_tokens: pflash_threshold,
                        keep_ratio: pflash_keep_ratio,
                        alpha: pflash_alpha,
                        min_keep_tokens: pflash_min_keep,
                        sink_tokens: pflash_sink,
                        recent_tokens: pflash_recent,
                        block_size: pflash_block,
                        profile: pflash_profile,
                        drafter_path: Some(pf_drafter_path.clone()),
                        sparse_threshold: pflash_sparse_threshold,
                    };
                    let mut pf_state = hipfire_arch_qwen35::pflash::PflashState::new(&pf_cfg);
                    // Pull the target tokenizer out of the loaded model
                    // for the compat check. Both Qwen3.5 and plain
                    // Qwen3 paths expose `tokenizer` on LoadedModel.
                    let tgt_tok_ref = m.tokenizer.as_ref();
                    if let Some(tok) = tgt_tok_ref {
                        let pf_max_kv = max_seq.max(2048);
                        // Hetero: when prefill_drafter_device >= 0 and isn't
                        // device 0 (target), allocate a sibling Gpu handle so
                        // drafter weights/KV/scratch live on the secondary
                        // card. Compress output is host-side, so decode stays
                        // on target. -1 / 0 => share target gpu (unchanged).
                        let mut sibling: Option<hipfire_rdna::Gpu> = None;
                        if pflash_drafter_device > 0 {
                            match hipfire_rdna::Gpu::init_with_device(pflash_drafter_device) {
                                Ok(g) => sibling = Some(g),
                                Err(e) => {
                                    let _ = writeln!(
                                        daemon_state.out.sink,
                                        r#"{{"type":"pflash_load_failed","reason":"drafter device {} init: {}"}}"#,
                                        pflash_drafter_device,
                                        e.to_string().replace('"', "'")
                                    );
                                }
                            }
                        }
                        let dg: &mut hipfire_rdna::Gpu =
                            sibling.as_mut().unwrap_or(&mut daemon_state.gpu);
                        dg.bind_thread_or_warn();
                        match hipfire_arch_qwen35::pflash::load_drafter(
                            &mut pf_state,
                            dg,
                            std::path::Path::new(pf_drafter_path),
                            tok,
                            pf_max_kv,
                        ) {
                            Ok(()) => {
                                eprintln!("[pflash] LOADED drafter={} dev={} mode={} compat={} keep={} thr={}",
                                    pf_drafter_path, pflash_drafter_device, pflash_mode_str,
                                    pf_state.tokenizer_compat, pflash_keep_ratio, pflash_threshold);
                                let _ = writeln!(
                                    daemon_state.out.sink,
                                    r#"{{"type":"pflash","mode":"{}","drafter":"{}","drafter_device":{},"tokenizer_compat":{},"keep_ratio":{},"threshold":{}}}"#,
                                    pflash_mode_str,
                                    pf_drafter_path,
                                    pflash_drafter_device,
                                    pf_state.tokenizer_compat,
                                    pflash_keep_ratio,
                                    pflash_threshold
                                );
                                daemon_state.pflash_state = Some(pf_state);
                                daemon_state.pflash_cfg = Some(pf_cfg);
                                daemon_state.pflash_drafter_gpu = sibling;
                                // persist sibling across requests (None if shared)
                            }
                            Err(e) => {
                                eprintln!("[pflash] LOAD FAILED: {}", e);
                                let _ = writeln!(
                                    daemon_state.out.sink,
                                    r#"{{"type":"pflash_load_failed","reason":"{}"}}"#,
                                    e.to_string().replace('"', "'")
                                );
                            }
                        }
                    } else {
                        let _ = writeln!(
                            daemon_state.out.sink,
                            r#"{{"type":"pflash_load_failed","reason":"target tokenizer unavailable"}}"#
                        );
                    }
                }
            }

            daemon_state.model = Some(m);
        }
        Err(e) => {
            if let Err(err) = daemon_state
                .resource_reservations
                .reacquire_placeholders(&mut daemon_state.gpu)
            {
                eprintln!(
                    "[hipfire-daemon] failed to restore resource reservations after load failure: {err}"
                );
            }
            let (vram_free, vram_total) = daemon_state.gpu.hip.get_vram_info().unwrap_or((0, 0));
            let free_mb = vram_free / (1024 * 1024);
            let total_mb = vram_total / (1024 * 1024);
            // serde-escape: raw HipError debug contains { } and "
            // which corrupt the JSONL protocol if interpolated raw.
            write_error(
                &mut daemon_state.out.sink,
                "",
                &format!(
                    "load failed: {e}. GPU: {} ({free_mb} MB free / {total_mb} MB total)",
                    daemon_state.gpu.arch
                ),
            );
        }
    }
    let _ = daemon_state.out.sink.flush();
}

pub(crate) fn reset(daemon_state: &mut DaemonState, msg: &serde_json::Value) {
    let target_worker_id = reset_target_worker_id(&msg, &daemon_state.active_worker_id);
    if reset_has_no_resident_model(
        &daemon_state.dummy_model,
        &daemon_state.model,
        &daemon_state.resident_models,
    ) {
        daemon_state
            .generic_state_arena
            .release_worker(&target_worker_id);
        daemon_state
            .out
            .emit(serde_json::json!({ "type": "reset", "seq_pos": 0 }));
        return;
    }
    if daemon_state.dummy_model.is_none() {
        match activate_model_worker(
            &target_worker_id,
            &mut daemon_state.active_worker_id,
            &mut daemon_state.model,
            &mut daemon_state.gpu,
            &mut daemon_state.resident_models,
        ) {
            Ok(true) => {}
            Ok(false) => {
                daemon_state
                    .out
                    .error(format!("unknown model worker {target_worker_id}"));
                return;
            }
            Err(e) => {
                daemon_state.out.error(format!("worker switch failed: {e}"));
                return;
            }
        }
    }
    // Reset conversation state without unloading the model.
    if let Some(dummy) = daemon_state.dummy_model.as_mut() {
        daemon_state
            .generic_state_arena
            .release_worker(&target_worker_id);
        dummy.reset();
        daemon_state
            .out
            .emit(serde_json::json!({ "type": "reset" }));
        return;
    }
    // Under eviction, also zero the compact_offset so absolute
    // RoPE phase restarts from zero for the fresh conversation.
    if let Some(ref mut m) = daemon_state.model {
        daemon_state
            .generic_state_arena
            .release_worker(&target_worker_id);
        m.active.cursor.seq_pos = 0;
        m.active.cursor.conversation_tokens.clear();
        m.q35_registry.sessions.clear();
        m.q35_registry.active_session_id = if is_qwen35_family_arch_id(m.arch_id)
            && m.pp == 1
            && m.active.sequence_state.is_some()
        {
            m.q35_registry.allocation_epoch = next_qwen35_state_allocation_epoch();
            Some(QWEN35_LEGACY_SESSION_ID.to_string())
        } else {
            m.q35_registry.allocation_epoch = 0;
            None
        };
        // Multi-GPU branch: route per-LA-layer memsets through
        // pp_dn_la_to_device so each buffer is zeroed on its
        // owning device. The single-GPU `gpu` parameter is left
        // alone — its scratch state isn't aliased to per-device
        // tensors when pp > 1.
        if m.pp > 1 {
            if let (Some(dn), Some(ref mut gpus), Some(ref la)) = (
                m.active
                    .sequence_state
                    .as_ref()
                    .and_then(|s| s.recurrent_as::<qwen35::DeltaNetState>()),
                m.pp_gpus.as_mut(),
                m.pp_dn_la_to_device.as_ref(),
            ) {
                for (i, s) in dn.s_matrices.iter().enumerate() {
                    let g = &mut gpus.devices[la[i] as usize];
                    let _ = g.bind_thread();
                    let _ = g.hip.memset(&s.buf, 0, s.buf.size());
                }
                for (i, s) in dn.s_scales.iter().enumerate() {
                    let g = &mut gpus.devices[la[i] as usize];
                    let _ = g.bind_thread();
                    let _ = g.hip.memset(&s.buf, 0, s.buf.size());
                }
                for (i, s) in dn.conv_states.iter().enumerate() {
                    let g = &mut gpus.devices[la[i] as usize];
                    let _ = g.bind_thread();
                    let _ = g.hip.memset(&s.buf, 0, s.buf.size());
                }
            }
        } else if let Some(dn) = m
            .active
            .sequence_state
            .as_ref()
            .and_then(|s| s.recurrent_as::<qwen35::DeltaNetState>())
        {
            // Zero DeltaNet recurrent state (Qwen3.5)
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
        if let Some(kv) = m.active.sequence_state.as_mut().and_then(|s| s.kv_mut()) {
            kv.compact_offset = 0;
        }
        if let Some(kv) = m.llama_kv.as_mut() {
            kv.compact_offset = 0;
        }
        // arch_id=7: rewind the Qwen2State position cursor so
        // the next prefill writes from KV[0]. Without this, a
        // reset between turns would leak the prior turn's KV
        // entries into attention for the new turn — fluent
        // garbage, no panic. See `Qwen2State::reset` doc.
        if let Some(ref mut s) = m.qwen2_state {
            s.reset();
        }
        // arch_id=9: same rationale for DeepSeek V4. Prior to
        // 2026-05-24 the V4F state was NEVER reset, so
        // `state.n_tokens` accumulated across requests and
        // every new prefill wrote AFTER the previous turn's
        // KV residue — fitting symptom for the multi-turn
        // pi-coding-agent corruption (`CLion` for
        // `CLionProjects`, `/home/n/` for `/home/nick/`).
        // See `DeepseekV4State::reset` doc.
        if let Some(ref mut s) = m.deepseek4_state {
            s.reset();
            // Drop the captured V4F decode hipGraph alongside
            // the state. The captured kernarg blobs hold
            // session-1's device-buffer pointers; a fresh
            // capture on session-2 binds against session-2's
            // pointers and host scalars. Without this the
            // replay path crashes with "illegal memory access"
            // on the post-launch logits D2H — the captured
            // graph dispatched against a stale slot/n_valid
            // computation that mis-ordered against this
            // session's prefill state. The matching
            // `ar_forward_warmed_up = false` in `reset()`
            // ensures we retrace warmup → capture → replay
            // rather than jumping straight back to replay.
            daemon_state.gpu.invalidate_graph_state();
        }
        // arch_id=10 (MiniMax-M2): clear KV cursor between turns.
        // No captured hipGraph on this path, so no graph
        // invalidation needed.
        if let Some(ref mut s) = m.minimax_state {
            s.reset();
        }
        // arch_id=11 (LFM2.5-MoE): clear KV + conv-state cursors
        // between turns. reset() also zeroes the rolling conv
        // states on-GPU, so it takes `gpu` and returns Result.
        #[cfg(feature = "arch-lfm2moe")]
        {
            if let Some(ref mut s) = m.active.lfm2moe_state {
                let _ = s.reset(&mut daemon_state.gpu);
            }
            m.lfm2_registry.sessions.clear();
            if m.arch_id == ARCH_ID_LFM2_MOE && m.pp == 1 && m.active.lfm2moe_state.is_some() {
                m.lfm2_registry.active_session_id = Some(LFM2_LEGACY_SESSION_ID.to_string());
                m.lfm2_registry.allocation_epoch = next_qwen35_state_allocation_epoch();
            } else {
                m.lfm2_registry.active_session_id = None;
                m.lfm2_registry.allocation_epoch = 0;
            }
        }
        // arch_id=12/13 (Gemma3 text / Gemma3-VL text): rewind the
        // backend-owned Gemma decode state. Without this, a reset
        // after a distractor turn leaves the internal KV cursor at
        // the prior turn and the same prompt produces different
        // greedy output.
        if let Some(ref mut b) = m.gemma3_text {
            b.state.reset();
        }
        if let Some(ref mut b) = m.gemma3_vl {
            b.state.reset();
        }
        if let Some(ref mut loaded) = m.registered_backend {
            let _ = loaded
                .backend
                .reset_session(&mut daemon_state.gpu, "default");
        }
        daemon_state
            .out
            .emit(serde_json::json!({ "type": "reset", "seq_pos": 0 }));
    } else {
        daemon_state
            .out
            .emit(serde_json::json!({ "type": "error", "message": "no model loaded" }));
    }
    let _ = daemon_state.out.sink.flush();
}

pub(crate) fn unload(daemon_state: &mut DaemonState) {
    // PFlash drafter goes FIRST: its weights/scratch/KV
    // tensors are released via Gpu::free_tensor, which only
    // queues into the GPU pool. The actual hipFree happens
    // inside unload_model -> drain_pool. Calling
    // unload_drafter AFTER unload_model would leave the
    // drafter buffers cached in the just-emptied pool with
    // no drain to follow, so the VRAM stays resident until
    // the next load message arrives. Order matters here.
    if let Some(mut pf) = daemon_state.pflash_state.take() {
        if let Some(mut dg) = daemon_state.pflash_drafter_gpu.take() {
            dg.bind_thread_or_warn();
            pf.unload_drafter(&mut dg); // sibling-device drafter: free on its own handle, then drop
            daemon_state.gpu.bind_thread_or_warn();
        } else {
            pf.unload_drafter(&mut daemon_state.gpu);
        }
    }
    daemon_state.pflash_cfg = None;
    if let Some(m) = daemon_state.model.take() {
        unload_model(m, &mut daemon_state.gpu);
    }
    for (_, m) in daemon_state.resident_models.drain() {
        unload_model(m, &mut daemon_state.gpu);
    }
    daemon_state.resource_reservations.clear_workers();
    if let Err(err) = daemon_state
        .resource_reservations
        .reacquire_placeholders(&mut daemon_state.gpu)
    {
        eprintln!("[hipfire-daemon] failed to restore resource reservations after unload: {err}");
    }
    daemon_state.generic_state_arena.clear();
    daemon_state.dummy_model = None;
    daemon_state.active_worker_id = DEFAULT_MODEL_WORKER_ID.to_string();
    // Drop any steer session so a stale capture/apply can't leak its
    // process-global state across model loads.
    hipfire_steer::clear();
    daemon_state
        .out
        .emit(serde_json::json!({ "type": "unloaded" }));
}

pub(crate) fn unload_worker(daemon_state: &mut DaemonState, msg: &serde_json::Value) {
    let id = msg
        .get("id")
        .and_then(|v| v.as_str())
        .unwrap_or("unload_worker");
    let request = parse_unload_worker_request(&msg, DEFAULT_MODEL_WORKER_ID);
    let worker_id = request.worker_id;
    let mut unloaded = false;
    daemon_state.generic_state_arena.release_worker(&worker_id);
    if worker_id == daemon_state.active_worker_id {
        if let Some(m) = daemon_state.model.take() {
            unload_model(m, &mut daemon_state.gpu);
            unloaded = true;
        }
        daemon_state.active_worker_id = DEFAULT_MODEL_WORKER_ID.to_string();
        if let Some((next_worker_id, next_model)) = daemon_state
            .resident_models
            .iter()
            .next()
            .map(|(k, _)| k.clone())
            .and_then(|k| daemon_state.resident_models.remove(&k).map(|m| (k, m)))
        {
            daemon_state.active_worker_id = next_worker_id;
            daemon_state.model = Some(next_model);
        }
    } else if let Some(m) = daemon_state.resident_models.remove(&worker_id) {
        unload_model(m, &mut daemon_state.gpu);
        unloaded = true;
    }
    if unloaded {
        daemon_state.resource_reservations.remove_worker(&worker_id);
        if let Err(err) = daemon_state
            .resource_reservations
            .reacquire_placeholders(&mut daemon_state.gpu)
        {
            eprintln!(
                "[hipfire-daemon] failed to restore resource reservations after worker unload: {err}"
            );
        }
    }
    let done = unload_worker_done_json(
        id,
        &worker_id,
        unloaded,
        daemon_state.resident_models.len() + usize::from(daemon_state.model.is_some()),
    );
    daemon_state.out.emit(done);
}
