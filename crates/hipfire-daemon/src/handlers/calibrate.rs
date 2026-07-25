//! Calibration and KLD evaluation — GPU forward passes over the resident model.
//!
//! These are the precedent for the AGENTS.md line sitting at format conversion
//! rather than at GPU work: `collect` calibrates the resident model in place and
//! only the request and the resulting artifact path cross the JSONL boundary.
//! `kld_eval` is also the one op that already streams incremental progress
//! frames per chunk, though still without yielding mid-op.

// Handler bodies were lifted verbatim out of `main()`, so they depend on the same
// root-level imports and arch aliases (`qwen35`, `deepseek4`, `minimax`, `lfm2moe`,
// `qwen2`, `prompt_frame`) that the crate root sets up. Glob-importing the root
// keeps that dependency in one place instead of re-deriving it per module.
use crate::*;

pub(crate) fn collect(daemon_state: &mut DaemonState, msg: &serde_json::Value) {
    // Parse fields directly from the JSON message (the daemon is the
    // server side; the typed CollectRequest contract lives in
    // hipfire-daemon-protocol for clients). Field names must match.
    let Some(corpus) = msg.get("corpus").and_then(|v| v.as_str()).map(String::from) else {
        emit_error_with_id(
            &mut daemon_state.stdout,
            "",
            "collect: missing 'corpus'".to_string(),
        );
        return;
    };
    let Some(output) = msg.get("output").and_then(|v| v.as_str()).map(String::from) else {
        emit_error_with_id(
            &mut daemon_state.stdout,
            "",
            "collect: missing 'output'".to_string(),
        );
        return;
    };
    let max_tokens = msg
        .get("max_tokens")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(512);
    let kldref = msg.get("kldref").and_then(|v| v.as_bool()).unwrap_or(false);
    let Some(m) = daemon_state.model.as_ref() else {
        emit_error_with_id(
            &mut daemon_state.stdout,
            "",
            "collect: no model loaded".to_string(),
        );
        return;
    };
    if m.pp != 1 {
        emit_error_with_id(
            &mut daemon_state.stdout,
            "",
            "collect: requires a single-GPU resident model (pp == 1)".to_string(),
        );
        return;
    }
    // Only the tokenizer is needed up front (to encode the corpus);
    // the per-arch calibration backend is resolved below. Every arch
    // with a collector reaches it through the one CalibratableBackend
    // seam — no qwen3.5-only gate.
    let Some(tokenizer) = m.tokenizer.as_ref() else {
        emit_error_with_id(
            &mut daemon_state.stdout,
            "",
            "collect: resident model has no tokenizer".to_string(),
        );
        return;
    };
    let text = match std::fs::read_to_string(&corpus) {
        Ok(t) => t,
        Err(e) => {
            emit_error_with_id(
                &mut daemon_state.stdout,
                "",
                format!("collect: read corpus {corpus}: {e}"),
            );
            return;
        }
    };
    // Bound tokenization to `max_tokens`: the tokenizer is superlinear
    // in input length, so encoding a whole multi-MB corpus would grind
    // for hours (the same stall fixed for kld_eval in 8571b79b). Only
    // the first `max_tokens` are ever calibrated on; tokenize just that
    // prefix (+ headroom).
    let take_chars = max_tokens.saturating_mul(8);
    let bounded: String = text.chars().take(take_chars).collect();
    let all = tokenizer.encode(&bounded);
    let n_tok = all.len().min(max_tokens);
    let tokens = all[..n_tok].to_vec();
    let provenance = [
        ("source_model", serde_json::json!(m.model_path)),
        ("corpus", serde_json::json!(corpus)),
        ("n_calib_tokens", serde_json::json!(n_tok)),
    ];
    let out_path = std::path::Path::new(&output);
    // Arch-agnostic calibration seam: resolve the resident backend's
    // collector and delegate. Each impl streams the .calib.hfq directly
    // to `output` one tensor at a time (no full-RAM materialization),
    // returning a summary. Probe order matches the resident slot layout.
    use hipfire_runtime::calibration::CalibratableBackend;
    let result: Result<hipfire_runtime::calibration::CalibSummary, String> = 'pick: {
        if let Some(b) = m.zaya_backend.as_ref() {
            break 'pick b.collect_calibration(
                &mut daemon_state.gpu,
                tokenizer,
                &tokens,
                kldref,
                out_path,
                &provenance,
            );
        }
        if let Some(b) = m.gemma3_text.as_ref() {
            break 'pick b.collect_calibration(
                &mut daemon_state.gpu,
                tokenizer,
                &tokens,
                kldref,
                out_path,
                &provenance,
            );
        }
        #[cfg(feature = "arch-lfm2moe")]
        if let (Some(w), Some(c)) = (m.lfm2moe_weights.as_ref(), m.lfm2moe_config.as_ref()) {
            let be = lfm2moe::calibration::Lfm2MoeCalibBackend {
                weights: w,
                config: c,
            };
            break 'pick be.collect_calibration(
                &mut daemon_state.gpu,
                tokenizer,
                &tokens,
                kldref,
                out_path,
                &provenance,
            );
        }
        if let (Some(w), Some(c)) = (m.q35_weights.as_ref(), m.q35_config.as_ref()) {
            let be = qwen35::Qwen35CalibBackend {
                weights: w,
                config: c,
            };
            break 'pick be.collect_calibration(
                &mut daemon_state.gpu,
                tokenizer,
                &tokens,
                kldref,
                out_path,
                &provenance,
            );
        }
        Err(format!(
            "collect: arch_id {} has no calibration-capable backend",
            m.arch_id
        ))
    };
    match result {
        Ok(summary) => {
            let resp = serde_json::json!({
                "type": "collected",
                "output": output,
                "n_hessian": summary.n_hessian,
                "n_calib_tokens": n_tok,
                "max_consistency": summary.max_consistency,
            });
            let _ = writeln!(daemon_state.stdout, "{resp}");
            let _ = daemon_state.stdout.flush();
        }
        Err(e) => emit_error_with_id(&mut daemon_state.stdout, "", format!("collect: {e}")),
    }
}

pub(crate) fn kld_eval(daemon_state: &mut DaemonState, msg: &serde_json::Value) {
    let mode = msg.get("mode").and_then(|v| v.as_str()).unwrap_or("");
    let corpus = msg.get("corpus").and_then(|v| v.as_str()).map(String::from);
    let ref_path = msg
        .get("ref_path")
        .and_then(|v| v.as_str())
        .map(String::from);
    let n_ctx = msg
        .get("n_ctx")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(2048);
    let max_chunks = msg
        .get("max_chunks")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize);
    let top_k = msg
        .get("config")
        .and_then(|c| c.get("top_k"))
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(256);
    let output = msg.get("output").and_then(|v| v.as_str()).map(String::from);
    let Some(m) = daemon_state.model.as_mut() else {
        emit_error_with_id(
            &mut daemon_state.stdout,
            "",
            "kld_eval: no model loaded".to_string(),
        );
        return;
    };
    if m.pp != 1 {
        emit_error_with_id(
            &mut daemon_state.stdout,
            "",
            "kld_eval: requires a single-GPU resident model (pp == 1)".to_string(),
        );
        return;
    }
    let arch_id = m.arch_id;
    let base_model = m.model_path.clone();
    let cfg: hipfire_kld::KldConfig = msg
        .get("config")
        .and_then(|c| serde_json::from_value(c.clone()).ok())
        .unwrap_or_default();
    let version = hipfire_build_info::VERSION.to_string();
    // Encode the corpus up front — needs the tokenizer, which must be
    // borrowed BEFORE the mutable backend borrow below. `score` mode
    // reads its tokens from the reference archive, so it needs none.
    let tokens: Vec<u32> = if mode == "self_score" || mode == "build_ref" {
        let Some(corpus_path) = corpus.clone() else {
            emit_error_with_id(
                &mut daemon_state.stdout,
                "",
                format!("kld_eval: mode={mode} requires 'corpus'"),
            );
            return;
        };
        let text = match std::fs::read_to_string(&corpus_path) {
            Ok(t) => t,
            Err(e) => {
                emit_error_with_id(
                    &mut daemon_state.stdout,
                    "",
                    format!("kld_eval: read {corpus_path}: {e}"),
                );
                return;
            }
        };
        let Some(tk) = m.tokenizer.as_ref() else {
            emit_error_with_id(
                &mut daemon_state.stdout,
                "",
                "kld_eval: resident model has no tokenizer".to_string(),
            );
            return;
        };
        // Only the first `n_ctx × max_chunks` tokens are ever scored, so
        // tokenize just that prefix (+ a chunk of headroom). The tokenizer
        // is superlinear in input length, so encoding a whole multi-MB
        // corpus slice would grind for hours — this is the reference-load
        // stall. With no chunk cap we still encode the full slice.
        match max_chunks {
            Some(mc) => {
                let want = n_ctx.saturating_mul(mc.saturating_add(1)).max(n_ctx);
                let take_chars = want.saturating_mul(8);
                let bounded: String = text.chars().take(take_chars).collect();
                tk.encode(&bounded)
            }
            None => tk.encode(&text),
        }
    } else {
        Vec::new()
    };
    // Respect the model's trained context. The load already clamped
    // `max_seq` to `max_position_embeddings` (see
    // `clamp_max_seq_to_model_context`, gated by
    // `HIPFIRE_MAX_SEQ_ALLOW_OVERRIDE`), so `m.max_seq` is the true
    // usable window. A KLD chunk longer than that decodes past the
    // model's positions and overruns position-indexed GPU buffers
    // (RoPE cos/sin table, KV) → a hard VMFault, not a graceful error.
    // Small-context models (e.g. Supra-50M, max_position_embeddings
    // =1024) hit this with the default n_ctx=2048. The override gate
    // flows through naturally: forcing a larger max_seq at load raises
    // `m.max_seq`, which raises this ceiling.
    let model_ctx = m.max_seq.max(2);
    if n_ctx > model_ctx {
        eprintln!(
            "kld_eval: clamping n_ctx {n_ctx} → {model_ctx} (model trained context; \
             load with HIPFIRE_MAX_SEQ_ALLOW_OVERRIDE=1 + a larger --max-seq to raise it)"
        );
    }
    let n_ctx = n_ctx.min(model_ctx);
    // Clamp the KLD window to the corpus: chunks are non-overlapping
    // `n_ctx` windows counted by floor (`tokens.len() / n_ctx`) with the
    // partial tail discarded, so a corpus shorter than n_ctx would yield
    // ZERO chunks and silently score nothing. Clamping makes any corpus
    // with ≥2 tokens form exactly one chunk; no effect once the corpus is
    // ≥ n_ctx. `score` reads its window from the archive, and `tokens` is
    // empty there, so this only adjusts build_ref / self_score. The
    // clamped value flows into KldRefPayloads.n_ctx → RefMeta, keeping
    // scoring_start (= n_ctx/2) consistent for the later score pass.
    let n_ctx = if tokens.is_empty() {
        n_ctx
    } else {
        n_ctx.min(tokens.len())
    };
    // Arch-agnostic forward seam: owned AR backends ride the blanket
    // SimpleAr impl; loose-slot arches (qwen3.5, lfm2moe, deepseek4,
    // minimax) go through their `*KldForward` adapter. All arches
    // equal. Probe order matches the resident slot layout; the
    // labeled block keeps the lfm2moe `#[cfg]` arm clean.
    use hipfire_runtime::kld_eval::ChunkScoredForward;
    let fwd_opt: Option<Box<dyn ChunkScoredForward + '_>> = 'pick: {
        if let Some(b) = m.zaya_backend.as_mut() {
            break 'pick Some(Box::new(b as &mut dyn ChunkScoredForward));
        }
        if let Some(b) = m.gemma3_text.as_mut() {
            break 'pick Some(Box::new(b as &mut dyn ChunkScoredForward));
        }
        if let Some(b) = m.gemma3_vl.as_mut() {
            break 'pick Some(Box::new(b as &mut dyn ChunkScoredForward));
        }
        if let Some(loaded) = m.registered_backend.as_mut() {
            if let Some(forward) = loaded.backend.kld_forward() {
                break 'pick Some(Box::new(forward));
            }
        }
        if let Some(b) = m.nemotron_backend.as_mut() {
            break 'pick Some(Box::new(b as &mut dyn ChunkScoredForward));
        }
        if let Some(b) = m.llama_backend.as_mut() {
            break 'pick Some(Box::new(b as &mut dyn ChunkScoredForward));
        }
        if let (Some(w), Some(c)) = (m.deepseek4_weights.as_ref(), m.deepseek4_config.as_ref()) {
            break 'pick Some(Box::new(deepseek4::kld::DeepseekV4KldForward {
                weights: w,
                config: c,
            }));
        }
        if let (Some(w), Some(c)) = (m.minimax_weights.as_ref(), m.minimax_config.as_ref()) {
            break 'pick Some(Box::new(minimax::kld::MiniMaxKldForward {
                weights: w,
                config: c,
            }));
        }
        #[cfg(feature = "arch-lfm2moe")]
        if let (Some(w), Some(c)) = (m.lfm2moe_weights.as_ref(), m.lfm2moe_config.as_ref()) {
            break 'pick Some(Box::new(lfm2moe::kld::Lfm2MoeKldForward {
                weights: w,
                config: c,
            }));
        }
        if let (Some(w), Some(c)) = (m.q35_weights.as_ref(), m.q35_config.as_ref()) {
            break 'pick Some(Box::new(qwen35::Qwen35KldForward {
                weights: w,
                config: c,
            }));
        }
        None
    };
    let mut fwd = match fwd_opt {
        Some(f) => f,
        None => {
            emit_error_with_id(
                &mut daemon_state.stdout,
                "",
                format!("kld_eval: arch_id {arch_id} has no KLD-scorable backend"),
            );
            return;
        }
    };
    let n_vocab = fwd.kld_vocab_size();

    macro_rules! kld_chunk_cb {
        () => {
            |c, n, s, k| {
                let _ = writeln!(
                    daemon_state.stdout,
                    "{}",
                    serde_json::json!({"type":"kld_chunk","chunk":c,"n_chunk":n,"scored":s,"mean_kld":k})
                );
                let _ = daemon_state.stdout.flush();
            }
        };
    }
    macro_rules! emit_kld_evaled {
        ($mode:expr, $out:expr, $seq:expr, $findings:expr) => {{
            let resp = serde_json::json!({
                "type": "kld_evaled", "mode": $mode,
                "n_chunk": $out.n_chunk, "total_scored": $out.total_scored,
                "mean_kld": $out.mean_kld, "p99_kld": $out.p99_kld,
                "mean_nll": $out.mean_nll, "ppl": ($out.mean_nll as f64).exp(),
                "seq_output": $seq, "compat_findings": $findings,
            });
            let _ = writeln!(daemon_state.stdout, "{resp}");
            let _ = daemon_state.stdout.flush();
        }};
    }

    match mode {
        "self_score" | "build_ref" => {
            if mode == "self_score" {
                match hipfire_runtime::kld_eval::kld_self_score(
                    &mut *fwd,
                    &mut daemon_state.gpu,
                    &tokens,
                    n_ctx,
                    top_k,
                    max_chunks,
                    kld_chunk_cb!(),
                ) {
                    Ok(out) => {
                        let mut seq = serde_json::Value::Null;
                        if let Some(p) = output.as_deref() {
                            match hipfire_kld::hfkseq::write_file(
                                std::path::Path::new(p),
                                &out.per_chunk,
                            ) {
                                Ok(()) => seq = serde_json::json!(p),
                                Err(e) => emit_error_with_id(
                                    &mut daemon_state.stdout,
                                    "",
                                    format!("kld_eval: write {p}: {e}"),
                                ),
                            }
                        }
                        emit_kld_evaled!("self_score", out, seq, serde_json::json!([]));
                    }
                    Err(e) => {
                        emit_error_with_id(&mut daemon_state.stdout, "", format!("kld_eval: {e}"))
                    }
                }
            } else {
                let Some(ref_out) = ref_path.clone() else {
                    emit_error_with_id(
                        &mut daemon_state.stdout,
                        "",
                        "kld_eval: build_ref requires 'ref_path'".to_string(),
                    );
                    return;
                };
                match hipfire_runtime::kld_eval::kld_build_ref(
                    &mut *fwd,
                    &mut daemon_state.gpu,
                    &tokens,
                    n_ctx,
                    top_k,
                    max_chunks,
                    |c, n, s| {
                        let _ = writeln!(
                            daemon_state.stdout,
                            "{}",
                            serde_json::json!({"type":"kld_chunk","chunk":c,"n_chunk":n,"scored":s,"mean_kld":0.0})
                        );
                        let _ = daemon_state.stdout.flush();
                    },
                ) {
                    Ok(p) => {
                        let meta = hipfire_kld::RefMeta {
                            schema: 2,
                            base_model_id: base_model.clone(),
                            source_model_sha256: String::new(),
                            tokenizer_sha256: None,
                            arch_id,
                            n_vocab: p.n_vocab,
                            n_ctx: p.n_ctx,
                            n_chunk: p.n_chunk,
                            scored_per_chunk: p.scored_per_chunk,
                            scoring_start: p.n_ctx / 2,
                            top_k: p.top_k,
                            total_scored: p.n_chunk * p.scored_per_chunk,
                            slice_path: corpus.clone().unwrap_or_default(),
                            slice_md5: String::new(),
                            config: cfg.clone(),
                            producer: hipfire_kld::ProducerInfo {
                                hipfire_version: version.clone(),
                                git_commit: Some(version.clone()),
                                git_describe: Some(version.clone()),
                                git_dirty: Some(version.contains("dirty")),
                                gpu_arch: daemon_state.gpu.arch.clone(),
                                producer_cmd: None,
                            },
                            payload_codecs: Default::default(),
                            content_sha256: None,
                        };
                        let archive = hipfire_kld::RefArchive {
                            meta,
                            tokens: p.tokens,
                            top_indices: p.top_indices,
                            top_log_probs: p.top_log_probs,
                            residual_mass: p.residual_mass,
                        };
                        let mut ref_output = serde_json::Value::Null;
                        match archive.write_file(std::path::Path::new(&ref_out)) {
                            Ok(()) => ref_output = serde_json::json!(ref_out),
                            Err(e) => emit_error_with_id(
                                &mut daemon_state.stdout,
                                "",
                                format!("kld_eval: write ref {ref_out}: {e}"),
                            ),
                        }
                        let resp = serde_json::json!({
                            "type": "kld_evaled", "mode": "build_ref",
                            "n_chunk": p.n_chunk,
                            "total_scored": p.n_chunk * p.scored_per_chunk,
                            "ref_output": ref_output, "compat_findings": [],
                        });
                        let _ = writeln!(daemon_state.stdout, "{resp}");
                        let _ = daemon_state.stdout.flush();
                    }
                    Err(e) => {
                        emit_error_with_id(&mut daemon_state.stdout, "", format!("kld_eval: {e}"))
                    }
                }
            }
        }
        "score" => {
            let Some(ref_in) = ref_path.clone() else {
                emit_error_with_id(
                    &mut daemon_state.stdout,
                    "",
                    "kld_eval: score requires 'ref_path'".to_string(),
                );
                return;
            };
            let archive = match read_kld_ref_archive(std::path::Path::new(&ref_in)) {
                Ok(a) => a,
                Err(e) => {
                    emit_error_with_id(
                        &mut daemon_state.stdout,
                        "",
                        format!("kld_eval: read ref {ref_in}: {e}"),
                    );
                    return;
                }
            };
            let run = hipfire_kld::RunEnv {
                git_commit: Some(version.clone()),
                gpu_arch: daemon_state.gpu.arch.clone(),
                arch_id,
                n_vocab,
                tokenizer_sha256: None,
                config: cfg.clone(),
            };
            let report = hipfire_kld::compat(&archive.meta, &run);
            if report.has_errors() {
                let errs: Vec<String> = report
                    .errors()
                    .map(|m| format!("{}: {}", m.field, m.detail))
                    .collect();
                emit_error_with_id(
                    &mut daemon_state.stdout,
                    "",
                    format!(
                        "kld_eval: refusing score — ref incompatible: {}",
                        errs.join("; ")
                    ),
                );
                return;
            }
            let findings: Vec<String> = report
                .mismatches
                .iter()
                .map(|m| format!("{:?} {}: {}", m.severity, m.field, m.detail))
                .collect();
            match hipfire_runtime::kld_eval::kld_score(
                &mut *fwd,
                &mut daemon_state.gpu,
                &archive,
                max_chunks,
                kld_chunk_cb!(),
            ) {
                Ok(out) => {
                    let mut seq = serde_json::Value::Null;
                    if let Some(p) = output.as_deref() {
                        match hipfire_kld::hfkseq::write_file(
                            std::path::Path::new(p),
                            &out.per_chunk,
                        ) {
                            Ok(()) => seq = serde_json::json!(p),
                            Err(e) => emit_error_with_id(
                                &mut daemon_state.stdout,
                                "",
                                format!("kld_eval: write {p}: {e}"),
                            ),
                        }
                    }
                    emit_kld_evaled!("score", out, seq, serde_json::json!(findings));
                }
                Err(e) => {
                    emit_error_with_id(&mut daemon_state.stdout, "", format!("kld_eval: {e}"))
                }
            }
        }
        other => emit_error_with_id(
            &mut daemon_state.stdout,
            "",
            format!("kld_eval: unknown mode {other:?}"),
        ),
    }
}
