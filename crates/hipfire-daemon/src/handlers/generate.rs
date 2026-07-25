//! Token generation, embeddings and reranking — the serving hot path.
//!
//! `generate` writes tokens straight to `daemon_state.stdout` as it produces
//! them and takes no cancellation token, which is why `Abort` cannot currently
//! interrupt it: the abort would arrive on the same channel this handler is
//! still occupying.

// Handler bodies were lifted verbatim out of `main()`, so they depend on the same
// root-level imports and arch aliases that the crate root sets up.
use crate::*;

pub(crate) fn embed(
    daemon_state: &mut DaemonState,
    msg: &serde_json::Value,
    req: hipfire_daemon_protocol::EmbedRequest,
) {
    let id = msg.get("id").and_then(|v| v.as_str()).unwrap_or("");
    let target_worker_id = message_worker_id(&msg);
    if daemon_state.dummy_model.is_some() {
        emit_error_with_id(
            &mut daemon_state.stdout,
            id,
            "embed is not supported for the dummy model",
        );
        return;
    }
    match activate_model_worker(
        &target_worker_id,
        &mut daemon_state.active_worker_id,
        &mut daemon_state.model,
        &mut daemon_state.gpu,
        &mut daemon_state.resident_models,
    ) {
        Ok(true) => {}
        Ok(false) => {
            emit_error_with_id(
                &mut daemon_state.stdout,
                id,
                format!("unknown model worker {target_worker_id}"),
            );
            return;
        }
        Err(e) => {
            emit_error_with_id(
                &mut daemon_state.stdout,
                id,
                format!("worker switch failed: {e}"),
            );
            return;
        }
    }
    let Some(m) = daemon_state.model.as_ref() else {
        emit_error_with_id(&mut daemon_state.stdout, id, "no model loaded");
        return;
    };
    match embeddinggemma_embed(
        &mut daemon_state.gpu,
        m,
        &req.texts,
        req.input_type,
        req.dims,
    ) {
        Ok(embeddings) => {
            let _ = serde_json::to_writer(
                &mut daemon_state.stdout,
                &serde_json::json!({
                    "type": "embeddings",
                    "id": id,
                    "embeddings": embeddings,
                }),
            );
            let _ = writeln!(daemon_state.stdout);
            let _ = daemon_state.stdout.flush();
        }
        Err(e) => emit_error_with_id(&mut daemon_state.stdout, id, e),
    }
}

pub(crate) fn rerank(
    daemon_state: &mut DaemonState,
    msg: &serde_json::Value,
    req: hipfire_daemon_protocol::RerankRequest,
) {
    let id = msg.get("id").and_then(|v| v.as_str()).unwrap_or("");
    let target_worker_id = message_worker_id(&msg);
    if daemon_state.dummy_model.is_some() {
        emit_error_with_id(
            &mut daemon_state.stdout,
            id,
            "rerank is not supported for the dummy model",
        );
        return;
    }
    match activate_model_worker(
        &target_worker_id,
        &mut daemon_state.active_worker_id,
        &mut daemon_state.model,
        &mut daemon_state.gpu,
        &mut daemon_state.resident_models,
    ) {
        Ok(true) => {}
        Ok(false) => {
            emit_error_with_id(
                &mut daemon_state.stdout,
                id,
                format!("unknown model worker {target_worker_id}"),
            );
            return;
        }
        Err(e) => {
            emit_error_with_id(
                &mut daemon_state.stdout,
                id,
                format!("worker switch failed: {e}"),
            );
            return;
        }
    }
    let Some(m) = daemon_state.model.as_ref() else {
        emit_error_with_id(&mut daemon_state.stdout, id, "no model loaded");
        return;
    };
    match embeddinggemma_rerank(&mut daemon_state.gpu, m, &req.query, &req.documents) {
        Ok(results) => {
            let _ = serde_json::to_writer(
                &mut daemon_state.stdout,
                &serde_json::json!({
                    "type": "rerank_scores",
                    "id": id,
                    "results": results,
                }),
            );
            let _ = writeln!(daemon_state.stdout);
            let _ = daemon_state.stdout.flush();
        }
        Err(e) => emit_error_with_id(&mut daemon_state.stdout, id, e),
    }
}

pub(crate) fn text(daemon_state: &mut DaemonState, msg: &serde_json::Value) {
    // Explicit per-request raw-prompt override (optional `"raw"`
    // bool). Absent → None → auto default (raw iff no chat_template).
    // Always set, so it resets every request (no cross-request leak).
    RAW_OVERRIDE.with(|c| c.set(msg.get("raw").and_then(|v| v.as_bool())));
    let protocol_generate =
        serde_json::from_value::<hipfire_generate::GenerateTextRequest>(msg.clone()).ok();
    let id = protocol_generate
        .as_ref()
        .map(|req| req.id.as_str())
        .or_else(|| msg.get("id").and_then(|v| v.as_str()))
        .unwrap_or("0");
    let target_worker_id = message_worker_id(&msg);
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
                emit_error_with_id(
                    &mut daemon_state.stdout,
                    id,
                    format!("unknown model worker {target_worker_id}"),
                );
                return;
            }
            Err(e) => {
                emit_error_with_id(
                    &mut daemon_state.stdout,
                    id,
                    format!("worker switch failed: {e}"),
                );
                return;
            }
        }
    }
    let session_id = msg
        .get("session_id")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .unwrap_or(id);
    let prefill_already_done = msg
        .get("prefill_already_done")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    if let Some(dummy) = daemon_state.dummy_model.as_mut() {
        let prompt = protocol_generate
            .as_ref()
            .map(|req| req.prompt.as_str())
            .or_else(|| msg.get("prompt").and_then(|v| v.as_str()))
            .unwrap_or("Hello");
        let max_tokens = protocol_generate
            .as_ref()
            .map(|req| req.sampling.max_tokens as usize)
            .or_else(|| {
                msg.get("max_tokens")
                    .and_then(|v| v.as_u64())
                    .map(|v| v as usize)
            })
            .unwrap_or(512);
        tracing::debug!(
            request_id = id,
            session_id,
            max_tokens,
            prefill_already_done,
            "dummy generate"
        );
        dummy.generate(
            &mut daemon_state.stdout,
            id,
            session_id,
            prompt,
            prefill_already_done,
            max_tokens,
        );
        return;
    }
    let m = match daemon_state.model.as_mut() {
        Some(m) => m,
        None => {
            let _ = writeln!(
                daemon_state.stdout,
                r#"{{"type":"error","message":"no model loaded"}}"#
            );
            let _ = daemon_state.stdout.flush();
            return;
        }
    };
    let session_id = msg
        .get("session_id")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty());
    let prefill_already_done = msg
        .get("prefill_already_done")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let prefilled_prompt_tokens = msg
        .get("prefilled_prompt_tokens")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize);
    #[cfg(feature = "arch-lfm2moe")]
    let is_lfm2_generate_session = m.arch_id == ARCH_ID_LFM2_MOE && m.pp == 1;
    #[cfg(not(feature = "arch-lfm2moe"))]
    let is_lfm2_generate_session = false;
    // S4: one `SessionServingBackend::activate_session` dispatch for the
    // rich-session arches (qwen35 5/6, lfm2 11) instead of the per-arch
    // `qwen35_*`/`lfm2_*` activate ladder. The arch-specific default
    // ("legacy") session id is resolved by `loaded_model_default_session_id`.
    let supports_generate_session =
        (is_qwen35_family_arch_id(m.arch_id) && m.pp == 1) || is_lfm2_generate_session;
    if supports_generate_session {
        let target_session_id = session_id.unwrap_or_else(|| loaded_model_default_session_id(m));
        if let Err(e) = m.activate_session(&mut daemon_state.gpu, target_session_id) {
            emit_error_with_id(&mut daemon_state.stdout, id, e);
            return;
        }
    } else if session_id.is_some() || prefill_already_done {
        emit_error_with_id(
            &mut daemon_state.stdout,
            id,
            "session_id/prefill_already_done are only supported for single-GPU qwen35/qwen35-moe/lfm2-moe",
        );
        return;
    }
    let prompt = protocol_generate
        .as_ref()
        .map(|req| req.prompt.as_str())
        .or_else(|| msg.get("prompt").and_then(|v| v.as_str()))
        .unwrap_or("Hello");
    let prompt_norm = normalize_daemon_prompt(prompt);
    let prompt: &str = &prompt_norm;
    if std::env::var("HIPFIRE_PROMPT_TOKEN_HEAT").ok().as_deref() == Some("1") {
        if let Some(tok) = m.tokenizer.as_ref() {
            tok.dump_prompt_heat(prompt);
        }
    }
    let system = protocol_generate
        .as_ref()
        .and_then(|req| req.system.as_deref())
        .or_else(|| msg.get("system").and_then(|v| v.as_str()));
    let image = msg.get("image").and_then(|v| v.as_str());
    let image_base64 = protocol_generate
        .as_ref()
        .and_then(|req| req.image_base64.as_deref())
        .or_else(|| msg.get("image_base64").and_then(|v| v.as_str()));

    // Structured-tools + structured-messages support (Phase 1 of
    // Jinja-everywhere migration). When present, both fields are
    // routed through `JinjaChatFrame::render_messages` so the
    // model sees the upstream template's `{% if tools %}` and
    // multi-turn branches (XML/JSON tool-call format per arch,
    // tool-response role mapping, etc.).
    //
    // Backward compat: when neither is present, legacy
    // `prompt`+`system` continues to drive a synthesized
    // [system?, user] slice — byte-identical to today's
    // `JinjaChatFrame::render()` single-turn path.
    //
    // Parse errors emit a structured error event and skip the
    // request (rather than silently dropping the fields).
    let tools_json: Option<Vec<serde_json::Value>> =
        if let Some(tools) = protocol_generate.as_ref().and_then(|req| req.tools.clone()) {
            match serde_json::from_value::<Vec<serde_json::Value>>(tools) {
                Ok(t) => Some(t),
                Err(e) => {
                    let _ = writeln!(
                        daemon_state.stdout,
                        r#"{{"type":"error","id":"{}","message":"invalid tools field: {}"}}"#,
                        id,
                        e.to_string().replace('"', "'"),
                    );
                    let _ = daemon_state.stdout.flush();
                    return;
                }
            }
        } else {
            match msg.get("tools") {
                Some(v) => match serde_json::from_value::<Vec<serde_json::Value>>(v.clone()) {
                    Ok(t) => Some(t),
                    Err(e) => {
                        let _ = writeln!(
                            daemon_state.stdout,
                            r#"{{"type":"error","id":"{}","message":"invalid tools field: {}"}}"#,
                            id,
                            e.to_string().replace('"', "'"),
                        );
                        let _ = daemon_state.stdout.flush();
                        return;
                    }
                },
                None => None,
            }
        };
    let messages_history: Option<Vec<prompt_frame::Message>> = if let Some(messages) =
        protocol_generate
            .as_ref()
            .and_then(|req| req.messages.clone())
    {
        Some(messages)
    } else {
        match msg.get("messages") {
            Some(v) => match serde_json::from_value::<Vec<prompt_frame::Message>>(v.clone()) {
                Ok(m) => Some(m),
                Err(e) => {
                    let _ = writeln!(
                        daemon_state.stdout,
                        r#"{{"type":"error","id":"{}","message":"invalid messages field: {}"}}"#,
                        id,
                        e.to_string().replace('"', "'"),
                    );
                    let _ = daemon_state.stdout.flush();
                    return;
                }
            },
            None => None,
        }
    };
    let request_stop_sequences = protocol_generate
        .as_ref()
        .and_then(|req| req.stop.clone())
        .unwrap_or_else(|| normalize_request_stop_sequences(msg.get("stop")));
    // Sampling defaults differ by arch: qwen35 family was tuned
    // at `temp=0.3, top_p=0.8` (DFlash-friendly, instruct-stable);
    // DeepSeek V4 Flash's HF card recommends `temp=1.0, top_p=1.0`
    // for local deployment, and lower values consistently fall
    // into block-level attractors on this quantized instruct
    // model. Pick arch-shaped defaults so a vanilla
    // `/v1/chat/completions` POST (no sampling fields) works on
    // both. Explicit per-request values still override either.
    let (mut default_temp, mut default_top_p) = if m.arch_id == ARCH_ID_LFM2_MOE {
        // LFM2.5-MoE (11): Liquid's model card recommends specific
        // sampling — temperature=0.2, top_p=0.80 (+ repetition_penalty
        // 1.05, set below). Use those exact values, not the generic
        // MoE-instruct (temp=1.0) default — they're tuned for this
        // model and keep it on-distribution.
        (0.2_f64, 0.80_f64)
    } else if m.arch_id == ARCH_ID_DEEPSEEK4_FLASH || m.arch_id == ARCH_ID_MINIMAX_M2 {
        // DeepSeek V4 (9) + MiniMax-M2 (10): quantized instruct
        // MoE models that fall into block-level attractors under
        // pure greedy. Default to the HF-recommended sampling
        // (temp=1.0, top_p=1.0); explicit per-request values
        // still override.
        (1.0_f64, 1.0_f64)
    } else {
        (0.3_f64, 0.8_f64)
    };
    let mut default_top_k = 20_usize;
    if let Some(sampler) = m
        .registered_backend
        .as_ref()
        .map(|loaded| &loaded.profile.sampler)
    {
        default_temp = sampler.temperature.map(f64::from).unwrap_or(default_temp);
        default_top_p = sampler.top_p.map(f64::from).unwrap_or(default_top_p);
        default_top_k = sampler.top_k.unwrap_or(default_top_k);
    }
    let temp_override = match protocol_generate.as_ref() {
        Some(req) if !req.sampling.temperature_is_default => Some(req.sampling.temperature),
        Some(_) => None,
        None => msg.get("temperature").and_then(|v| v.as_f64()),
    };
    let temp = temp_override.unwrap_or(default_temp) as f32;
    let max_tokens = protocol_generate
        .as_ref()
        .map(|req| req.sampling.max_tokens as usize)
        .or_else(|| {
            msg.get("max_tokens")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
        })
        .unwrap_or(512);
    let top_p_override = match protocol_generate.as_ref() {
        Some(req) if !req.sampling.top_p_is_default => req.sampling.top_p,
        Some(_) => None,
        None => msg.get("top_p").and_then(|v| v.as_f64()),
    };
    let top_p = top_p_override.unwrap_or(default_top_p) as f32;
    let top_k = protocol_generate
        .as_ref()
        .and_then(|req| req.sampling.top_k)
        .or_else(|| {
            msg.get("top_k")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
        })
        .unwrap_or(default_top_k);
    // Default 1.0 (off). Matches llama.cpp `--repeat-penalty 1.0`
    // and HF transformers `generate(repetition_penalty=1.0)`
    // defaults. The prior 1.3 default suppressed legitimately
    // repeated formatting tokens (e.g. `' **'` for bullets,
    // indentation patterns) on multi-step reasoning prompts,
    // pushing structured chain-of-thought trajectories off the
    // model's well-trained path into a self-doubt / number-
    // hallucination attractor on 9B Qwen3.5 at greedy decode.
    // Root cause writeup: issue #258 comment "Bug B root cause"
    // and docs/investigations/2026-05-15-9b-reasoning-loop/.
    // Clients can still opt in to a non-1.0 value per request.
    // LFM2.5-MoE (arch_id 11): Liquid's card recommends
    // repetition_penalty=1.05; default to it (others stay 1.0/off).
    // gemma3-vl (arch 13) decodes greedily through `decode_loop`;
    // near-identical video slices push bare greedy into a token
    // attractor, so default to a 1.3 repeat penalty (matches the
    // bring-up example) unless the client overrides it.
    let default_repeat_penalty = if m.arch_id == ARCH_ID_LFM2_MOE {
        1.05_f64
    } else if m.arch_id == ARCH_ID_GEMMA3_VL {
        1.3_f64
    } else {
        1.0_f64
    };
    let repeat_penalty = protocol_generate
        .as_ref()
        .and_then(|req| req.sampling.repeat_penalty)
        .or_else(|| msg.get("repeat_penalty").and_then(|v| v.as_f64()))
        .unwrap_or(default_repeat_penalty) as f32;
    // OpenAI-compatible `reasoning_effort` (also accept our custom
    // `thinking_mode` alias) — only consumed by arch_id=9 today.
    // Default = NonThink, matching the safe HF chat frame.
    let think_mode = protocol_generate
        .as_ref()
        .and_then(|req| {
            req.reasoning_effort
                .as_deref()
                .or(req.thinking_mode.as_deref())
                .or(req.thinking.as_deref())
        })
        .or_else(|| {
            msg.get("reasoning_effort")
                .or_else(|| msg.get("thinking_mode"))
                .and_then(|v| v.as_str())
        })
        .map(ThinkMode::from_str)
        .unwrap_or(ThinkMode::NonThink);
    let repeat_window = msg
        .get("repeat_window")
        .and_then(|v| v.as_u64())
        .unwrap_or(128) as usize;
    let presence_penalty = protocol_generate
        .as_ref()
        .and_then(|req| req.presence_penalty)
        .or_else(|| msg.get("presence_penalty").and_then(|v| v.as_f64()))
        .unwrap_or(0.0)
        .max(0.0) as f32;
    let frequency_penalty = protocol_generate
        .as_ref()
        .and_then(|req| req.frequency_penalty)
        .or_else(|| msg.get("frequency_penalty").and_then(|v| v.as_f64()))
        .unwrap_or(0.0)
        .max(0.0) as f32;
    // Experimental: inject a nudge string at a specific generated-
    // token count. The nudge tokens get forward-fed through the KV
    // cache so the model "sees" them as part of its own trajectory,
    // and are emitted to stdout so the client stream includes them.
    // Used to test whether telling a thinking model "time's up"
    // gets it to close </think> and commit to an answer.
    //
    // GATED: off by default. The feature has a real UX hazard — if
    // the alert fires after </think> has already closed, the nudge
    // leaks into the visible answer. Only honor the params when the
    // operator has explicitly opted in via config
    // (`experimental_budget_alert: true` → HIPFIRE_EXPERIMENTAL_
    // BUDGET_ALERT=1 set by the CLI). Research use only; not a
    // stable contract.
    let experimental_ok = std::env::var("HIPFIRE_EXPERIMENTAL_BUDGET_ALERT")
        .ok()
        .as_deref()
        == Some("1");
    let budget_alert_at_tok = if experimental_ok {
        msg.get("budget_alert_at_tok")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize
    } else {
        0
    };
    let budget_alert_text = if experimental_ok {
        msg.get("budget_alert_text")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string()
    } else {
        String::new()
    };
    // Budget for tokens emitted INSIDE the model's <think>...</think>
    // block. 0 = uncapped (model thinks until it naturally closes).
    // Triggered from the CLI by per-model `max_think_tokens` config,
    // OpenAI `chat_template_kwargs.enable_thinking=false` (cap=1),
    // and `reasoning.effort` (none=1, minimal=64, low=256, medium=
    // 1024, high=4096, xhigh=0).
    //
    // When the cap is reached the daemon force-emits "</think>\n"
    // through the same KV-write + sample path as a normal token,
    // closing the thinking block so the model commits to an
    // answer with the remaining max_tokens budget. Caught by
    // Codex stop-time review on 2026-04-28: the field had been
    // shipping in genParams from the HTTP layer but the daemon
    // was silently ignoring it, making the new reasoning.effort
    // / enable_thinking knobs no-ops on the wire.
    let max_think_tokens = protocol_generate
        .as_ref()
        .and_then(|req| req.max_think_tokens.map(u64::from))
        .or_else(|| msg.get("max_think_tokens").and_then(|v| v.as_u64()))
        .unwrap_or(0) as usize;

    // assistant_prefix: "plain", "open_think", or "closed_think"
    // Controls the ChatML framing after the assistant role header.
    // Consumed by the text path; VL path does not yet propagate
    // it (tracked as a follow-up to the post-#169 rebase).
    let assistant_prefix = prompt_frame::AssistantPrefix::from_label(
        protocol_generate
            .as_ref()
            .and_then(|req| req.assistant_prefix.as_deref())
            .or_else(|| msg.get("assistant_prefix").and_then(|v| v.as_str())),
    );

    let has_image = image_base64.is_some() || image.is_some();
    // Cache-warm: encode + cache the image embeddings, skip LM decode
    // (gemma3-vl only). Lets a dataset be pre-encoded into the vision
    // cache cheaply without the per-token prefill cost.
    let vision_cache_only = msg
        .get("vision_cache_only")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let video = msg.get("video").and_then(|v| v.as_str());
    let max_frames = msg.get("max_frames").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
    // `images`: a JSON array of paths for true multi-image (distinct
    // images) on the gemma3-vl path. Non-string entries are skipped.
    let images: Vec<&str> = msg
        .get("images")
        .and_then(|v| v.as_array())
        .map(|a| a.iter().filter_map(|v| v.as_str()).collect())
        .unwrap_or_default();
    // Optional per-image text labels (gemma3-vl), emitted before each
    // image so the model can order/reference distinct slices.
    let image_labels: Vec<String> = msg
        .get("image_labels")
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .map(|v| v.as_str().unwrap_or("").to_string())
                .collect()
        })
        .unwrap_or_default();
    let is_dots_ocr = m.arch_id == ARCH_ID_DOTS_OCR;
    let is_gemma3_vl = m.gemma3_vl.is_some(); // arch 13 (medgemma)
    let has_media = has_image || video.is_some() || !images.is_empty();
    let has_vl = m.vision_config.is_some() || is_dots_ocr || is_gemma3_vl;

    if video.is_some() && !is_gemma3_vl {
        write_error(
            &mut daemon_state.stdout,
            id,
            "video input is only supported on gemma3-vl (arch 13)",
        );
    } else if has_media && !has_vl {
        write_error(&mut daemon_state.stdout, id, "model has no vision encoder");
    } else if is_gemma3_vl && has_media {
        // arch-13 gemma3-vl: decode image / image_base64 / video into raw
        // frames daemon-side, then serve through Gemma3VlBackend (SigLIP
        // encode → projector splice → shared greedy decode_loop). A video
        // (or an image path that is_video) expands to up to max_frames.
        let vl_max_think_tokens = if max_think_tokens == 0 {
            256
        } else {
            max_think_tokens
        };
        match decode_vl_frames(image, &images, image_base64, video, max_frames) {
            Ok(frames) => {
                let params = GenerateVLParams {
                    id,
                    prompt,
                    system_prompt: system,
                    // Unused on the gemma3-vl path: bytes arrive via `frames`.
                    image_source: ImageSource::Path(""),
                    temp,
                    top_p,
                    max_tokens,
                    repeat_penalty,
                    repeat_window,
                    max_think_tokens: vl_max_think_tokens,
                    encode_only: vision_cache_only,
                };
                generate_vl_gemma3(
                    m,
                    &mut daemon_state.gpu,
                    &mut daemon_state.stdout,
                    &params,
                    &frames,
                    &image_labels,
                );
            }
            Err(e) => write_error(&mut daemon_state.stdout, id, &e),
        }
    } else if has_image && has_vl {
        if image_base64.is_some() && image.is_some() {
            eprintln!("[daemon/vl] both image and image_base64 provided — using image_base64");
        }
        let source = if let Some(b64) = image_base64 {
            if b64.len() > MAX_BASE64_ENCODED_LEN {
                write_error(
                    &mut daemon_state.stdout,
                    id,
                    &format!(
                        "image payload exceeds maximum encoded size ({} bytes)",
                        MAX_BASE64_ENCODED_LEN,
                    ),
                );
                return;
            }
            ImageSource::Base64(b64)
        } else {
            ImageSource::Path(image.unwrap())
        };
        // Plan-mandated Phase-1 stopgap (docs/plans/completions_vision.md §2.1):
        // VL dispatch defaults `max_think_tokens` to 256 when the
        // client doesn't specify one. Caps runaway thinking
        // without needing the full `ThinkState` extraction. Text
        // path keeps unwrap_or(0) — it has different defaults
        // controlled per-model on the CLI side.
        let vl_max_think_tokens = if max_think_tokens == 0 {
            256
        } else {
            max_think_tokens
        };
        let params = GenerateVLParams {
            id,
            prompt,
            system_prompt: system,
            image_source: source,
            temp,
            top_p,
            max_tokens,
            repeat_penalty,
            repeat_window,
            max_think_tokens: vl_max_think_tokens,
            encode_only: false, // qwen35-vl / dots-ocr always decode
        };
        if is_dots_ocr {
            generate_vl_dots_ocr(m, &mut daemon_state.gpu, &mut daemon_state.stdout, &params);
        } else {
            generate_vl(m, &mut daemon_state.gpu, &mut daemon_state.stdout, &params);
        }
    } else {
        // Per-request PflashConfig: clone the load-time cfg
        // and apply any per-request overrides from `params`.
        // None when no drafter was configured at load --
        // generate() then takes the identity path.
        //
        // Out-of-range overrides (keep_ratio outside (0, 1],
        // block_size == 0) would otherwise reach asserts inside
        // select_spans / scoring and panic the entire daemon.
        // Reject the request with an explicit error event so
        // the client gets a clean signal and the daemon stays up.
        let mut pf_override_err: Option<String> = None;
        let pf_cfg_owned = daemon_state.pflash_cfg.as_ref().map(|base| {
            let mut c = base.clone();
            if let Some(s) = msg
                .get("params")
                .and_then(|p| p.get("prefill_compression"))
                .and_then(|v| v.as_str())
            {
                if let Some(m) = hipfire_arch_qwen35::pflash::PflashMode::parse(s) {
                    c.mode = m;
                }
            }
            if let Some(v) = msg
                .get("params")
                .and_then(|p| p.get("prefill_threshold"))
                .and_then(|v| v.as_u64())
            {
                c.threshold_tokens = v as usize;
            }
            if let Some(v) = msg
                .get("params")
                .and_then(|p| p.get("prefill_keep_ratio"))
                .and_then(|v| v.as_f64())
            {
                let r = v as f32;
                if !(r > 0.0 && r <= 1.0) {
                    pf_override_err = Some(format!("prefill_keep_ratio={r} not in (0, 1]"));
                } else {
                    c.keep_ratio = r;
                }
            }
            if let Some(v) = msg
                .get("params")
                .and_then(|p| p.get("prefill_min_keep"))
                .and_then(|v| v.as_u64())
            {
                c.min_keep_tokens = v as usize;
            }
            if let Some(v) = msg
                .get("params")
                .and_then(|p| p.get("prefill_sink"))
                .and_then(|v| v.as_u64())
            {
                c.sink_tokens = v as usize;
            }
            if let Some(v) = msg
                .get("params")
                .and_then(|p| p.get("prefill_recent"))
                .and_then(|v| v.as_u64())
            {
                c.recent_tokens = v as usize;
            }
            if let Some(v) = msg
                .get("params")
                .and_then(|p| p.get("prefill_block"))
                .and_then(|v| v.as_u64())
            {
                let b = v as usize;
                if b == 0 {
                    pf_override_err = Some("prefill_block must be > 0".to_string());
                } else {
                    c.block_size = b;
                }
            }
            c
        });
        if let Some(reason) = pf_override_err {
            let _ = writeln!(
                daemon_state.stdout,
                r#"{{"type":"error","id":"{}","message":"invalid pflash override: {}"}}"#,
                id,
                reason.replace('"', "'"),
            );
            let _ = daemon_state.stdout.flush();
            return;
        }
        generate(
            m,
            &mut daemon_state.gpu,
            daemon_state.pflash_drafter_gpu.as_mut(),
            &mut daemon_state.stdout,
            id,
            prompt,
            system,
            temp,
            top_p,
            top_k,
            max_tokens,
            repeat_penalty,
            repeat_window,
            presence_penalty,
            frequency_penalty,
            budget_alert_at_tok,
            &budget_alert_text,
            max_think_tokens,
            assistant_prefix,
            daemon_state.pflash_state.as_mut(),
            pf_cfg_owned.as_ref(),
            tools_json.as_deref(),
            messages_history.as_deref(),
            think_mode,
            prefill_already_done,
            prefilled_prompt_tokens,
            &request_stop_sequences,
            protocol_generate
                .as_ref()
                .and_then(|req| req.evidence_dir.as_deref())
                .or_else(|| msg.get("evidence_dir").and_then(|v| v.as_str())),
        );
    }
}
