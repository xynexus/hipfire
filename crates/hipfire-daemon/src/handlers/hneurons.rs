//! H-Neurons interventions and CETT capture.
//!
//! `cett_load_colnorms` populates the per-layer down_proj column norms that
//! `cett_capture` then reuses for every prefill, so the two are ordered: capture
//! without a prior load has nothing to score against.

// Handler bodies were lifted verbatim out of `main()`, so they depend on the same
// root-level imports and arch aliases (`qwen35`, `deepseek4`, `minimax`, `lfm2moe`,
// `qwen2`, `prompt_frame`) that the crate root sets up. Glob-importing the root
// keeps that dependency in one place instead of re-deriving it per module.
use crate::*;

pub(crate) fn intervene(daemon_state: &mut DaemonState, msg: &serde_json::Value) {
    let gain = msg.get("gain").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32;
    let indices: Vec<usize> = msg
        .get("indices")
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|x| x.as_u64().map(|u| u as usize))
                .collect()
        })
        .unwrap_or_default();
    // Mask geometry from the resident model config (immutable borrow,
    // dropped before the mutable `gpu` use below).
    let dims = match daemon_state.model.as_ref() {
        Some(m) => {
            if let Some(b) = m.gemma3_text.as_ref() {
                Some((b.config.num_hidden_layers, b.config.intermediate_size))
            } else if let Some(b) = m.gemma3_vl.as_ref() {
                Some((b.text_cfg.num_hidden_layers, b.text_cfg.intermediate_size))
            } else if let Some(b) = m.llama_backend.as_ref() {
                Some((b.config.n_layers, b.config.hidden_dim))
            } else {
                None
            }
        }
        None => {
            daemon_state
                .out
                .error("hneuron_intervene: no model loaded".to_string());
            return;
        }
    };
    let Some((n_layers, inter)) = dims else {
        daemon_state
            .out
            .error("hneuron_intervene: no resident dense backend (llama|gemma3)".to_string());
        return;
    };
    let n_intervened = indices.len();
    let result = if indices.is_empty() || (gain - 1.0).abs() < f32::EPSILON {
        hipfire_hneurons::intervene::clear();
        Ok(())
    } else {
        hipfire_hneurons::intervene::begin_intervention(
            &mut daemon_state.gpu,
            n_layers,
            inter,
            &indices,
            gain,
        )
    };
    match result {
        Ok(()) => {
            let resp = serde_json::json!({
                "type": "hneuron_ok",
                "n_intervened": n_intervened,
                "gain": gain,
            });
            daemon_state.out.emit(resp);
        }
        Err(e) => daemon_state.out.error(format!("hneuron_intervene: {e:?}")),
    }
}

pub(crate) fn cett_load_colnorms(daemon_state: &mut DaemonState, msg: &serde_json::Value) {
    let Some(path) = msg.get("path").and_then(|v| v.as_str()).map(String::from) else {
        daemon_state
            .out
            .error("cett_load_colnorms: missing 'path'".to_string());
        return;
    };
    let bytes = match std::fs::read(&path) {
        Ok(b) => b,
        Err(e) => {
            daemon_state.out.error(format!("cett_load_colnorms: {e}"));
            return;
        }
    };
    if bytes.len() < 8 {
        daemon_state
            .out
            .error("cett_load_colnorms: file too short".to_string());
        return;
    }
    let n_layers = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
    let inter = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;
    let want = 8 + n_layers * inter * 4;
    if bytes.len() != want {
        daemon_state.out.error(format!(
            "cett_load_colnorms: size mismatch (got {} want {want})",
            bytes.len()
        ));
        return;
    }
    let mut cn = Vec::with_capacity(n_layers);
    let mut off = 8usize;
    for _ in 0..n_layers {
        let mut row = Vec::with_capacity(inter);
        for _ in 0..inter {
            row.push(f32::from_le_bytes([
                bytes[off],
                bytes[off + 1],
                bytes[off + 2],
                bytes[off + 3],
            ]));
            off += 4;
        }
        cn.push(row);
    }
    daemon_state.cett_colnorms = Some(cn);
    let resp = serde_json::json!({
        "type": "cett_ok",
        "n_layers": n_layers,
        "intermediate": inter,
    });
    daemon_state.out.emit(resp);
}

pub(crate) fn cett_capture(daemon_state: &mut DaemonState, msg: &serde_json::Value) {
    let system = msg
        .get("system")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let Some(user) = msg.get("user").and_then(|v| v.as_str()).map(String::from) else {
        daemon_state
            .out
            .error("cett_capture: missing 'user'".to_string());
        return;
    };
    let response = msg
        .get("response")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let Some(colnorms) = daemon_state.cett_colnorms.clone() else {
        daemon_state
            .out
            .error("cett_capture: no colnorms (call cett_load_colnorms first)".to_string());
        return;
    };
    let Some(m) = daemon_state.model.as_mut() else {
        daemon_state
            .out
            .error("cett_capture: no model loaded".to_string());
        return;
    };
    let arch_id = m.arch_id;
    // Frame the prompt via the model's jinja chat_template, then build
    // the full [prompt ++ response] token sequence. These are all
    // immutable borrows of `m`, released before the mutable backend
    // borrow below (mirrors the steer_capture ordering).
    let framed = {
        let Some(tokenizer) = m.tokenizer.as_ref() else {
            daemon_state
                .out
                .error("cett_capture: resident model has no tokenizer".to_string());
            return;
        };
        let Some(tmpl) = m.chat_template.as_ref() else {
            daemon_state
                .out
                .error("cett_capture: model has no chat_template".to_string());
            return;
        };
        let frame = prompt_frame::JinjaChatFrame {
            tokenizer,
            template: tmpl,
            system: (!system.is_empty()).then_some(system.as_str()),
            user: &user,
            enable_thinking: false,
            bos_token: None,
        };
        match frame.render() {
            Ok(t) => t,
            Err(e) => {
                daemon_state
                    .out
                    .error(format!("cett_capture: jinja render: {e}"));
                return;
            }
        }
    };
    let (full, response_start) = {
        let tokenizer = m.tokenizer.as_ref().unwrap();
        let prompt_ids = tokenizer.encode(&framed);
        let response_ids = tokenizer.encode(&response);
        let rs = prompt_ids.len();
        let mut full = prompt_ids;
        full.extend(response_ids);
        (full, rs)
    };
    if full.len() <= response_start {
        daemon_state
            .out
            .error("cett_capture: empty response after tokenization".to_string());
        return;
    }
    // Optional answer-token span (paper's answer-token CETT). The probe
    // passes the token offset+len of the factual answer WITHIN the
    // response (computed from the dataset's tokenized_response +
    // answer_tokens); we capture only that span. Absent → whole response.
    let (cap_start, cap_end) = match (
        msg.get("answer_offset").and_then(|v| v.as_u64()),
        msg.get("answer_len").and_then(|v| v.as_u64()),
    ) {
        (Some(off), Some(len)) if len > 0 => {
            let s = (response_start + off as usize).min(full.len().saturating_sub(1));
            let e = (s + len as usize).min(full.len());
            (s, e.max(s + 1))
        }
        _ => (response_start, usize::MAX),
    };
    // Run the tapped prefill on whichever dense backend is resident.
    // llama uses the generic prefill_forward (materializes down_proj
    // in+out); gemma3 (text + vl) route through SimpleAr::prefill →
    // the shared, tapped forward_prefill_batch. Both feed the same
    // capture session. Helper to finalize identically per backend.
    use hipfire_runtime::arch::SimpleAr;
    fn finish(gpu: &mut hipfire_rdna::Gpu) -> Result<(Vec<Vec<f32>>, usize), String> {
        hipfire_hneurons::capture::finish_capture(gpu)
            .map_err(|e| format!("finish: {e:?}"))?
            .ok_or_else(|| "capture produced no feature".to_string())
    }
    let outcome: Result<(Vec<Vec<f32>>, usize), String> = if let Some(b) = m.llama_backend.as_mut()
    {
        if colnorms.len() != b.config.n_layers {
            Err(format!(
                "colnorms n_layers {} != model n_layers {}",
                colnorms.len(),
                b.config.n_layers
            ))
        } else if let Err(e) = hipfire_hneurons::capture::begin_capture(
            &mut daemon_state.gpu,
            colnorms,
            cap_start,
            cap_end,
            b.config.dim,
        ) {
            Err(format!("begin_capture: {e:?}"))
        } else {
            // Fast path: the WMMA forward_prefill_batch (tapped via
            // the residual snapshot), not the ~40× slower generic
            // prefill_forward. Requires a q8 KV cache for batch
            // eligibility (the probe loads with kv_cache=q8).
            match SimpleAr::prefill(b, &mut daemon_state.gpu, &full) {
                Ok(()) => finish(&mut daemon_state.gpu),
                Err(e) => {
                    hipfire_hneurons::capture::clear();
                    Err(format!("prefill: {e}"))
                }
            }
        }
    } else if let Some(b) = m.gemma3_text.as_mut() {
        if colnorms.len() != b.config.num_hidden_layers {
            Err(format!(
                "colnorms n_layers {} != model n_layers {}",
                colnorms.len(),
                b.config.num_hidden_layers
            ))
        } else if let Err(e) = hipfire_hneurons::capture::begin_capture(
            &mut daemon_state.gpu,
            colnorms,
            cap_start,
            cap_end,
            b.config.hidden_size,
        ) {
            Err(format!("begin_capture: {e:?}"))
        } else {
            b.state.reset();
            match SimpleAr::prefill(b, &mut daemon_state.gpu, &full) {
                Ok(()) => finish(&mut daemon_state.gpu),
                Err(e) => {
                    hipfire_hneurons::capture::clear();
                    Err(format!("prefill: {e}"))
                }
            }
        }
    } else if let Some(b) = m.gemma3_vl.as_mut() {
        if colnorms.len() != b.text_cfg.num_hidden_layers {
            Err(format!(
                "colnorms n_layers {} != model n_layers {}",
                colnorms.len(),
                b.text_cfg.num_hidden_layers
            ))
        } else if let Err(e) = hipfire_hneurons::capture::begin_capture(
            &mut daemon_state.gpu,
            colnorms,
            cap_start,
            cap_end,
            b.text_cfg.hidden_size,
        ) {
            Err(format!("begin_capture: {e:?}"))
        } else {
            b.state.reset();
            match SimpleAr::prefill(b, &mut daemon_state.gpu, &full) {
                Ok(()) => finish(&mut daemon_state.gpu),
                Err(e) => {
                    hipfire_hneurons::capture::clear();
                    Err(format!("prefill: {e}"))
                }
            }
        }
    } else {
        Err(format!(
            "arch_id {arch_id} has no supported backend (llama|gemma3)"
        ))
    };
    match outcome {
        Ok((feature, count)) => {
            let resp = serde_json::json!({
                "type": "cett_feature",
                "feature": feature,
                "count": count,
            });
            daemon_state.out.emit(resp);
        }
        Err(e) => daemon_state.out.error(format!("cett_capture: {e}")),
    }
}
