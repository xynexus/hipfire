// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The Qwen3.5 / Qwen3 (llama) text generate paths — the daemon's default
//! autoregressive decode.
//!
//! `generate` is the central AR text path the request loop calls for the
//! qwen35/llama families: prompt framing, prefill, the per-token
//! sample/EOS-filter/loop-guard/stream loop, multi-turn KV + eviction, and
//! optional PFlash prefill compression. The qwen35 spec-decode fast paths layer
//! on top: `generate_mtp` (MTP head), `generate_dflash` (DFlash diffusion
//! drafter + DDTree), and `generate_multi` (multi-GPU pipeline-parallel).
//! Extracted verbatim from the former `main.rs` monolith (no behavior change);
//! items called from `main.rs` are `pub`.

use std::path::Path;
use std::time::Instant;

use hipfire_arch_qwen35::qwen35;
use hipfire_arch_qwen35::speculative;
use hipfire_generate::eos_filter::EosFilter;
use hipfire_generate::loop_guard::StopReason;
use hipfire_generate::sampler::{collect_unclosed_attractor_blocks, SamplerConfig};
#[cfg(feature = "arch-lfm2moe")]
use hipfire_model::ARCH_ID_LFM2_MOE;
use hipfire_model::{
    is_qwen35_family_arch_id, ARCH_ID_DEEPSEEK4_FLASH, ARCH_ID_GEMMA3_TEXT, ARCH_ID_GEMMA3_VL,
    ARCH_ID_LLAMA_MISTRAL, ARCH_ID_MAMBA2, ARCH_ID_MINIMAX_M2, ARCH_ID_NEMOTRON_H,
    ARCH_ID_QWEN3_QWEN2_LEGACY, ARCH_ID_ZAYA,
};
use hipfire_prompt as prompt_frame;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::llama;
use hipfire_runtime::sampler;

use crate::events::{emit_committed_event, emit_filter_action, write_error};
use crate::evidence::{
    write_daemon_moe_router_evidence, write_daemon_runtime_oneshot_evidence,
    DaemonMoeRouterHistogramGuard,
};
#[cfg(feature = "arch-lfm2moe")]
use crate::generate_arch::generate_lfm2moe;
use crate::generate_arch::{
    generate_deepseek4, generate_gemma3, generate_gemma3_vl_text, generate_llama, generate_minimax,
    generate_nemotron, generate_registered_backend, generate_zaya,
};
use crate::model::{effective_raw, LoadedModel};
use crate::output_filter::chat_output_filter;
use crate::output_filter::{chat_output_filter_from_profile, loop_guard_from_runtime_config};
use crate::request::ThinkMode;
use crate::session::{
    put_qwen35_state_into_model, qwen35_restore_or_error, take_qwen35_state_from_model,
    Qwen35RequestSessionState,
};
use crate::spec_metrics::emit_spec_done;
use hipfire_specdecode::SpecMetrics;

/// MTP (Multi-Token Prediction) spec-decode generate path for Qwen3.5/3.6.
///
/// Selected when a qwen35 model carries a co-trained MTP head (bundled
/// `-mq4+mtp.hfq` or sibling `.mtp.hfq`), `mtp_mode != "off"`, and no DFlash
/// drafter is loaded. Uses the **non-tree** `mtp_spec::spec_step_mtp` (linear
/// autoregressive K-token draft) because the FP32 DeltaNet state the small
/// models run on does not support tree-mode replay (see TODO.md /
/// `qwen35::default_state_quant`). The MTP head is lazy-loaded from
/// `m.model_path` per request and freed at function end (cache-on-slot is a
/// future optimization).
#[allow(clippy::too_many_arguments)]
pub fn generate_mtp(
    m: &mut LoadedModel,
    gpu: &mut hipfire_rdna::Gpu,
    stdout: &mut dyn std::io::Write,
    id: &str,
    prompt: &str,
    system_prompt: Option<&str>,
    max_tokens: usize,
    max_think_tokens: usize,
    assistant_prefix: prompt_frame::AssistantPrefix,
    tools: Option<&[serde_json::Value]>,
    messages_history: Option<&[prompt_frame::Message]>,
    request_stop_sequences: &[String],
    // Explicit per-request `"raw"` override; `None` = auto (raw iff the model
    // has no chat_template). Threaded rather than read from a global — see
    // `effective_raw` for the cross-request leak that motivated it.
    raw_override: Option<bool>,
) {
    use hipfire_arch_qwen35::mtp_head::{self, MtpKvMode};
    use hipfire_arch_qwen35::mtp_spec::{self, MtpSpecState};
    use hipfire_arch_qwen35::qwen35;
    use hipfire_arch_qwen35::speculative::{ModelSlot, ModelSlotConfig};

    let tokenizer = m.tokenizer.as_ref().unwrap();

    // Prompt build mirrors generate_dflash: Jinja chat template when enabled,
    // else the hand-rolled ChatFrame::Plain scaffold.
    let jinja_enabled = std::env::var("HIPFIRE_JINJA_CHAT").ok().as_deref() == Some("1");
    let try_jinja = jinja_enabled && m.chat_template.is_some();
    let prompt_tokens: Vec<u32> = if try_jinja {
        let template = m.chat_template.as_ref().unwrap();
        let frame = prompt_frame::JinjaChatFrame {
            tokenizer,
            template,
            system: system_prompt,
            user: prompt,
            enable_thinking: max_think_tokens != 1,
            bos_token: None,
        };
        let render_result = if tools.is_some() || messages_history.is_some() {
            let synthesized: Vec<prompt_frame::Message>;
            let messages_slice: &[prompt_frame::Message] = match messages_history {
                Some(mh) => mh,
                None => {
                    let mut v = Vec::new();
                    if let Some(sys) = system_prompt {
                        v.push(prompt_frame::Message {
                            role: prompt_frame::Role::System,
                            content: sys.to_string(),
                            tool_calls: Vec::new(),
                            tool_call_id: None,
                        });
                    }
                    v.push(prompt_frame::Message {
                        role: prompt_frame::Role::User,
                        content: prompt.to_string(),
                        tool_calls: Vec::new(),
                        tool_call_id: None,
                    });
                    synthesized = v;
                    &synthesized
                }
            };
            frame.render_messages(messages_slice, tools, None)
        } else {
            frame.render()
        };
        match render_result {
            Ok(rendered) => tokenizer.encode(&rendered),
            Err(e) => {
                tracing::warn!("jinja render failed in mtp path ({e}) — falling back to Plain");
                prompt_frame::ChatFrame {
                    tokenizer,
                    system: system_prompt,
                    user: prompt,
                    assistant_prefix,
                    raw: effective_raw(m, raw_override),
                }
                .build()
            }
        }
    } else {
        prompt_frame::ChatFrame {
            tokenizer,
            system: system_prompt,
            user: prompt,
            assistant_prefix,
            raw: effective_raw(m, raw_override),
        }
        .build()
    };

    let im_end = tokenizer.encode("<|im_end|>");
    let im_end_token = if im_end.len() == 1 {
        Some(im_end[0])
    } else {
        None
    };

    // Fresh target state — MTP runs its own prefill from position 0.
    m.active.cursor.seq_pos = 0;
    m.active.cursor.conversation_tokens.clear();
    {
        let dn = m.dn_state().unwrap();
        for s in &dn.s_matrices {
            let _ = gpu.hip.memset(&s.buf, 0, s.buf.size());
        }
        for s in &dn.s_scales {
            let _ = gpu.hip.memset(&s.buf, 0, s.buf.size());
        }
        for s in &dn.conv_states {
            let _ = gpu.hip.memset(&s.buf, 0, s.buf.size());
        }
    }

    // Assemble a transient ModelSlot, mirroring generate_dflash's take/putback.
    let target_config = m.q35_config.as_ref().unwrap().clone();
    let weights = m.q35_weights.take().expect("q35 weights");
    let (kv_cache, dn_state) = take_qwen35_state_from_model(&mut m.active.sequence_state);
    let scratch = m.q35_scratch.take().expect("q35 scratch");
    macro_rules! putback {
        ($t:expr) => {{
            m.q35_weights = Some($t.weights);
            put_qwen35_state_into_model(m, $t.kv_cache, $t.dn_state);
            m.q35_scratch = Some($t.scratch);
        }};
    }
    let hfq = match HfqFile::open(Path::new(&m.model_path)) {
        Ok(h) => h,
        Err(e) => {
            write_error(stdout, id, &format!("reopen model: {e}"));
            m.q35_weights = Some(weights);
            put_qwen35_state_into_model(m, kv_cache, dn_state);
            m.q35_scratch = Some(scratch);
            return;
        }
    };
    let mut target = ModelSlot {
        name: String::from("target"),
        hfq,
        config: target_config,
        weights,
        kv_cache,
        dn_state,
        scratch,
        slot_config: ModelSlotConfig::default(),
    };

    let max_seq_total = m.physical_cap.max(m.max_seq);
    let max_n = m.mtp_k.clamp(1, 8);
    let eos_token = target.config.eos_token;
    let dim = target.config.dim;

    if prompt_tokens.len() + max_tokens * (max_n + 1) + 16 > max_seq_total {
        write_error(
            stdout,
            id,
            &format!(
                "prompt ({}) + max_tokens ({}) × (max_n+1) won't fit in max_seq {}",
                prompt_tokens.len(),
                max_tokens,
                max_seq_total
            ),
        );
        putback!(target);
        return;
    }

    // Load the MTP head from the bundled trailer (or a sibling .mtp.hfq).
    let head = match mtp_head::load_mtp_head_bundled(Path::new(&m.model_path), gpu, max_seq_total) {
        Ok(Some(h)) => h,
        Ok(None) => {
            // No bundled trailer — try a sibling "<stem>.mtp.hfq".
            let sidecar = m.model_path.replace(".hfq", ".mtp.hfq");
            match mtp_head::load_mtp_head(Path::new(&sidecar), gpu, max_seq_total) {
                Ok(h) => h,
                Err(e) => {
                    write_error(stdout, id, &format!("load mtp head: {e}"));
                    putback!(target);
                    return;
                }
            }
        }
        Err(e) => {
            write_error(stdout, id, &format!("load bundled mtp head: {e}"));
            putback!(target);
            return;
        }
    };

    let kv_mode = m
        .q35_kv_mode
        .as_deref()
        .and_then(|s| MtpKvMode::parse(s).ok())
        .unwrap_or(MtpKvMode::Q8);
    let mut state =
        match MtpSpecState::new_for_slot_with_kv_mode(gpu, &target, &head, max_n, kv_mode) {
            Ok(s) => s,
            Err(e) => {
                write_error(stdout, id, &format!("alloc MtpSpecState: {e}"));
                head.free_gpu(gpu);
                putback!(target);
                return;
            }
        };

    let t0 = Instant::now();

    // Run prefill + spec-decode loop; centralize cleanup after.
    // Returns (hit_eos, generated, cycles, accepted_total, decode_secs).
    let run: Result<(bool, usize, SpecMetrics, f64), String> = (|| {
        // Prefill the prompt through the batched WMMA path.
        qwen35::forward_prefill_batch(
            gpu,
            &target.weights,
            &target.config,
            &prompt_tokens,
            0,
            &mut target.kv_cache,
            &mut target.dn_state,
            &target.scratch,
            None,
            None,
            None,
            None,
        )
        .map_err(|e| format!("prefill forward_prefill_batch: {e:?}"))?;

        // Snapshot trunk's post-output-norm hidden at the last prefill position.
        state
            .capture_prev_hidden_from_scratch_tmp(gpu, &target.scratch.tmp, dim)
            .map_err(|e| format!("capture prev_hidden: {e:?}"))?;

        // Seed token = argmax of trunk logits at the last prefill position.
        let logits0 = gpu
            .download_f32(&target.scratch.logits)
            .map_err(|e| format!("download seed logits: {e:?}"))?;
        let mut seed_token = 0u32;
        let mut best = f32::NEG_INFINITY;
        for (i, &v) in logits0.iter().enumerate() {
            if v > best {
                best = v;
                seed_token = i as u32;
            }
        }

        // Streaming state (mirrors generate_dflash).
        let mut streamed_tokens: Vec<u32> = Vec::new();
        let mut bytes_fed_to_filter = 0usize;
        let mut filter = chat_output_filter(m, request_stop_sequences);
        let mut generated = 0usize;
        let mut think_count: usize = 0;
        let mut prev_in_think: bool = false;

        // Helper closure semantics inlined: stream one committed token, return
        // (hit_eos, think_cap_hit).
        let emit_token = |stdout: &mut dyn std::io::Write,
                          tok: u32,
                          streamed_tokens: &mut Vec<u32>,
                          bytes_fed_to_filter: &mut usize,
                          filter: &mut EosFilter,
                          think_count: &mut usize,
                          prev_in_think: &mut bool|
         -> (bool, bool) {
            streamed_tokens.push(tok);
            emit_committed_event(
                stdout,
                id,
                tok,
                streamed_tokens.len() - 1,
                t0.elapsed().as_millis() as u64,
            );
            let all_bytes = tokenizer.decode_bytes(streamed_tokens);
            let new_bytes = &all_bytes[*bytes_fed_to_filter..];
            *bytes_fed_to_filter = all_bytes.len();
            if emit_filter_action(stdout, id, filter.observe(new_bytes)) {
                return (true, false);
            }
            let hit_eos =
                tok == eos_token || im_end_token == Some(tok) || tokenizer.is_terminator(tok);
            let mut think_cap_hit = false;
            if !hit_eos && max_think_tokens > 0 {
                let raw = tokenizer.decode_bytes(streamed_tokens);
                let raw_str = std::str::from_utf8(&raw).unwrap_or("");
                let open_idx = raw_str.rfind("<think>");
                let close_idx = raw_str.rfind("</think>");
                let in_think = match (open_idx, close_idx) {
                    (Some(o), Some(c)) => o > c,
                    (Some(_), None) => true,
                    _ => false,
                };
                if in_think && !*prev_in_think {
                    *think_count = 0;
                }
                if in_think {
                    *think_count += 1;
                }
                *prev_in_think = in_think;
                if in_think && *think_count >= max_think_tokens {
                    let _ = writeln!(
                        stdout,
                        r#"{{"type":"token","id":"{}","text":"</think>\n"}}"#,
                        id
                    );
                    let _ = stdout.flush();
                    think_cap_hit = true;
                }
            }
            (hit_eos, think_cap_hit)
        };

        // Emit the seed token first.
        let (mut hit_eos, mut think_cap_hit) = emit_token(
            stdout,
            seed_token,
            &mut streamed_tokens,
            &mut bytes_fed_to_filter,
            &mut filter,
            &mut think_count,
            &mut prev_in_think,
        );
        generated += 1;

        let mut last_committed = seed_token;
        let mut cur_pos = prompt_tokens.len();
        let mut spec_metrics = SpecMetrics::new(max_n);
        let t_decode = Instant::now();

        while !hit_eos && !think_cap_hit && generated < max_tokens {
            if cur_pos + max_n + 1 >= max_seq_total {
                break;
            }
            let result = mtp_spec::spec_step_mtp(
                gpu,
                &mut target,
                &head,
                &mut state,
                cur_pos,
                last_committed,
                eos_token,
            )
            .map_err(|e| format!("spec_step_mtp: {e:?}"))?;
            // `committed` EXCLUDES the seed (unlike DFlash); `drafts_generated`
            // is the proposed count this window.
            spec_metrics.record_window(
                result.drafts_generated,
                result.accept_count,
                result.committed.len(),
            );

            // result.committed already EXCLUDES the seed (unlike DFlash).
            for &tok in &result.committed {
                if generated >= max_tokens {
                    break;
                }
                let (eos, cap) = emit_token(
                    stdout,
                    tok,
                    &mut streamed_tokens,
                    &mut bytes_fed_to_filter,
                    &mut filter,
                    &mut think_count,
                    &mut prev_in_think,
                );
                generated += 1;
                if eos {
                    hit_eos = true;
                    break;
                }
                if cap {
                    think_cap_hit = true;
                    break;
                }
            }
            if let Some(&last) = result.committed.last() {
                last_committed = last;
            }
            cur_pos += result.advance;
            if result.hit_eos {
                hit_eos = true;
            }
        }
        Ok((
            hit_eos,
            generated,
            spec_metrics,
            t_decode.elapsed().as_secs_f64(),
        ))
    })();

    // Cleanup: free MTP state + head, put trunk pieces back on the model.
    state.free_gpu(gpu);
    head.free_gpu(gpu);
    putback!(target);

    match run {
        Ok((_hit_eos, generated, spec_metrics, decode_secs)) => {
            let tok_s = generated as f64 / decode_secs.max(1e-9);
            // Non-metric + legacy fields: decode_tok_s (== tok_s here), the
            // `cycles` alias (== windows), and `max_n` (block size). Canonical
            // `tau`/`accepted`/`windows` come from `spec_metrics`.
            let ext = serde_json::json!({
                "decode_tok_s": (tok_s * 10.0).round() / 10.0,
                "cycles": spec_metrics.windows,
                "max_n": max_n,
            });
            emit_spec_done(
                stdout,
                id,
                generated,
                tok_s,
                "mtp",
                &spec_metrics,
                Some(ext),
            );
        }
        Err(e) => {
            write_error(stdout, id, &e);
        }
    }
}

/// DFlash-powered greedy decode. Mirrors `generate`'s ChatML shape and
/// token-streaming output but replaces the AR sample loop with
/// `spec_step_dflash` cycles — each cycle drafts B tokens via the diffusion
/// drafter and verifies them in one target forward, committing accept_len+1 at
/// a time. With DDTree enabled it uses the tree-verify path instead of the
/// linear chain. Single-turn: resets target state at entry (stateless
/// chat-completions contract).
pub fn generate_dflash(
    m: &mut LoadedModel,
    gpu: &mut hipfire_rdna::Gpu,
    stdout: &mut dyn std::io::Write,
    id: &str,
    prompt: &str,
    system_prompt: Option<&str>,
    max_tokens: usize,
    max_think_tokens: usize,
    assistant_prefix: prompt_frame::AssistantPrefix,
    pflash_bypass_reason: Option<&str>,
    pflash_alpha: Option<f32>,
    tools: Option<&[serde_json::Value]>,
    messages_history: Option<&[prompt_frame::Message]>,
    request_stop_sequences: &[String],
    // Explicit per-request `"raw"` override; `None` = auto (raw iff the model
    // has no chat_template). Threaded rather than read from a global — see
    // `effective_raw` for the cross-request leak that motivated it.
    raw_override: Option<bool>,
    // Per-request identity for n-gram table scoping; see NgramRequestScope.
    ngram_scope: Option<crate::model::NgramRequestScope<'_>>,
) {
    use hipfire_arch_qwen35::speculative::{
        spec_step_ddtree_batched, spec_step_ddtree_path_c, spec_step_dflash, ModelSlot,
        ModelSlotConfig, Phase2Snapshots,
    };

    // Prompt build: same two-path branch as the AR-path generate() — when
    // `HIPFIRE_JINJA_CHAT=1` AND the model carries a chat_template, render
    // via `JinjaChatFrame` so structured `tools` / `messages` can reach
    // the upstream template's `{% if tools %}` / multi-turn branches.
    // Otherwise fall back to the hand-rolled `ChatFrame::Plain` scaffold
    // (byte-identical to the prior DFlash-path build).
    //
    // DFlash is single-turn by construction — `seq_pos` is reset to 0
    // below before seed_target_hidden_from_prompt runs — so we never
    // need to guard on `seq_pos == 0` here.
    let tokenizer = m.tokenizer.as_ref().unwrap();
    let jinja_enabled = std::env::var("HIPFIRE_JINJA_CHAT").ok().as_deref() == Some("1");
    let try_jinja = jinja_enabled && m.chat_template.is_some();
    let prompt_tokens: Vec<u32> = if try_jinja {
        let template = m.chat_template.as_ref().unwrap();
        let frame = prompt_frame::JinjaChatFrame {
            tokenizer,
            template,
            system: system_prompt,
            user: prompt,
            enable_thinking: max_think_tokens != 1,
            bos_token: None,
        };
        let render_result = if tools.is_some() || messages_history.is_some() {
            let synthesized: Vec<prompt_frame::Message>;
            let messages_slice: &[prompt_frame::Message] = match messages_history {
                Some(m) => m,
                None => {
                    let mut v = Vec::new();
                    if let Some(sys) = system_prompt {
                        v.push(prompt_frame::Message {
                            role: prompt_frame::Role::System,
                            content: sys.to_string(),
                            tool_calls: Vec::new(),
                            tool_call_id: None,
                        });
                    }
                    v.push(prompt_frame::Message {
                        role: prompt_frame::Role::User,
                        content: prompt.to_string(),
                        tool_calls: Vec::new(),
                        tool_call_id: None,
                    });
                    synthesized = v;
                    &synthesized
                }
            };
            frame.render_messages(messages_slice, tools, None)
        } else {
            frame.render()
        };
        match render_result {
            Ok(rendered) => tokenizer.encode(&rendered),
            Err(e) => {
                tracing::warn!("jinja render failed in dflash path ({e}) — falling back to Plain");
                prompt_frame::ChatFrame {
                    tokenizer,
                    system: system_prompt,
                    user: prompt,
                    assistant_prefix,
                    raw: effective_raw(m, raw_override),
                }
                .build()
            }
        }
    } else {
        prompt_frame::ChatFrame {
            tokenizer,
            system: system_prompt,
            user: prompt,
            assistant_prefix,
            raw: effective_raw(m, raw_override),
        }
        .build()
    };

    // `im_end_token` is still needed downstream for the EOS check.
    let im_end = tokenizer.encode("<|im_end|>");
    let im_end_token = if im_end.len() == 1 {
        Some(im_end[0])
    } else {
        None
    };

    // Fresh target state — DFlash seed_target_hidden_from_prompt does its own
    // full prefill, so we reset first to avoid double-accounting.
    m.active.cursor.seq_pos = 0;
    m.active.cursor.conversation_tokens.clear();
    {
        let dn = m.dn_state().unwrap();
        for s in &dn.s_matrices {
            let _ = gpu.hip.memset(&s.buf, 0, s.buf.size());
        }
        for s in &dn.s_scales {
            let _ = gpu.hip.memset(&s.buf, 0, s.buf.size());
        }
        for s in &dn.conv_states {
            let _ = gpu.hip.memset(&s.buf, 0, s.buf.size());
        }
    }
    let chat_template_profile = m.chat_template_profile.clone();
    let df = m.dflash.as_mut().unwrap();
    df.target_hidden_host.clear();
    df.draft_scratch.reset_upload_tracking();

    // Assemble a transient ModelSlot for the spec helpers — they both take
    // `&mut ModelSlot`. We own the pieces on LoadedModel individually, so
    // take them, build the ModelSlot, run, then put them back.
    //
    // ModelSlot needs its own HfqFile field but spec_step_dflash doesn't
    // actually touch it. Reopening via mmap is essentially free (few µs).
    let target_config = m.q35_config.as_ref().unwrap().clone();
    let weights = m.q35_weights.take().expect("q35 weights");
    let (kv_cache, dn_state) = take_qwen35_state_from_model(&mut m.active.sequence_state);
    let scratch = m.q35_scratch.take().expect("q35 scratch");
    let hfq = match HfqFile::open(Path::new(&m.model_path)) {
        Ok(h) => h,
        Err(e) => {
            let _ = writeln!(
                stdout,
                r#"{{"type":"error","id":"{}","message":"reopen model: {}"}}"#,
                id, e
            );
            let _ = stdout.flush();
            m.q35_weights = Some(weights);
            put_qwen35_state_into_model(m, kv_cache, dn_state);
            m.q35_scratch = Some(scratch);
            return;
        }
    };
    let slot_config = ModelSlotConfig::default();
    let mut target = ModelSlot {
        name: String::from("target"),
        hfq,
        config: target_config,
        weights,
        kv_cache,
        dn_state,
        scratch,
        slot_config,
    };

    let t0 = Instant::now();
    let ctx_capacity = df.ctx_capacity;
    // Capacity checks. With eviction enabled the advertised context window is
    // effectively unbounded (eviction fires between spec cycles), but the
    // *prompt* must still fit in one physical_cap span because
    // seed_target_hidden_from_prompt writes it per-token without chunking.
    let eff_prompt_cap = if m.eviction.is_some() {
        m.physical_cap
    } else {
        ctx_capacity
    };
    if prompt_tokens.len() + df.block_size > eff_prompt_cap {
        let _ = writeln!(
            stdout,
            r#"{{"type":"error","id":"{}","message":"prompt+block_size exceeds {} {} (eviction {})"}}"#,
            id,
            if m.eviction.is_some() {
                "physical_cap"
            } else {
                "ctx_capacity"
            },
            eff_prompt_cap,
            if m.eviction.is_some() { "on" } else { "off" },
        );
        let _ = stdout.flush();
        m.q35_weights = Some(target.weights);
        put_qwen35_state_into_model(m, target.kv_cache, target.dn_state);
        m.q35_scratch = Some(target.scratch);
        return;
    }
    if m.eviction.is_none() && prompt_tokens.len() + max_tokens + df.block_size > ctx_capacity {
        let _ = writeln!(
            stdout,
            r#"{{"type":"error","id":"{}","message":"prompt+max_tokens exceeds ctx_capacity {} (enable cask_sidecar for long decode)"}}"#,
            id, ctx_capacity,
        );
        let _ = stdout.flush();
        m.q35_weights = Some(target.weights);
        put_qwen35_state_into_model(m, target.kv_cache, target.dn_state);
        m.q35_scratch = Some(target.scratch);
        return;
    }

    // Seed target_hidden via the demo's helper — runs a per-token prefill
    // with hidden extraction into hidden_rb, then downloads prompt-length
    // worth of rows into target_hidden_host. The draft's first forward
    // uses these as context.
    if let Err(e) = speculative::seed_target_hidden_from_prompt(
        gpu,
        &mut target,
        &mut df.hidden_rb,
        &mut df.target_hidden_host,
        &prompt_tokens,
    ) {
        let _ = writeln!(
            stdout,
            r#"{{"type":"error","id":"{}","message":"prefill: {}"}}"#,
            id, e
        );
        let _ = stdout.flush();
        m.q35_weights = Some(target.weights);
        put_qwen35_state_into_model(m, target.kv_cache, target.dn_state);
        m.q35_scratch = Some(target.scratch);
        return;
    }
    // Prime the draft's GPU target_hidden buffer from the prompt rows so the
    // first spec step can skip the CPU→GPU upload of the whole context.
    if let Err(e) = speculative::scatter_hidden_block_to_interleaved(
        gpu,
        &df.hidden_rb,
        &df.draft_scratch.target_hidden,
        0,
        prompt_tokens.len(),
        prompt_tokens.len(),
    ) {
        tracing::warn!("scatter failed: {e} — falling back to per-cycle upload");
    }
    df.draft_scratch.uploaded_target_hidden_rows = prompt_tokens.len();
    df.draft_scratch.target_hidden_abs_positions = (0..prompt_tokens.len() as i32).collect();

    // First emit = target's argmax at the final prompt position. seed_target_hidden
    // already ran the per-token forward for every prompt token; its scratch.logits
    // holds the post-prompt logits.
    let first_logits = match gpu.download_f32(&target.scratch.logits) {
        Ok(v) => v,
        Err(e) => {
            let _ = writeln!(
                stdout,
                r#"{{"type":"error","id":"{}","message":"download logits: {}"}}"#,
                id, e
            );
            let _ = stdout.flush();
            m.q35_weights = Some(target.weights);
            put_qwen35_state_into_model(m, target.kv_cache, target.dn_state);
            m.q35_scratch = Some(target.scratch);
            return;
        }
    };
    let first_token = first_logits
        .iter()
        .enumerate()
        .fold((0u32, f32::NEG_INFINITY), |(best, bv), (i, &v)| {
            if v > bv {
                (i as u32, v)
            } else {
                (best, bv)
            }
        })
        .0;

    let t_prefill = Instant::now();

    // Opt-in drafter-free n-gram spec decode. Strictly additive: it supplies a
    // spine through the existing `pld_spine` seam, and on a miss that seam gets
    // `None` and `spec_step_dflash` runs its own drafter exactly as before.
    // Scope key: a different user or session type must not inherit the previous
    // request's tables or hot state.
    let ngram_key = {
        let sc = ngram_scope.unwrap_or_default();
        format!(
            "{}\u{1}{}",
            sc.user_id.unwrap_or(""),
            sc.session_type.unwrap_or("")
        )
    };
    // Reuse the live state when this request has the same scope; otherwise drop
    // the old one (persisting what it staged) and build fresh.
    let carried = match df.ngram_live.take() {
        Some((k, prev)) if k == ngram_key => Some(prev),
        Some((_, mut prev)) => {
            let _ = prev.merge();
            None
        }
        None => None,
    };
    let mut ngram = carried.or_else(|| {
        df.ngram.clone().map(|setup| {
            use crate::model::NgramWriteTarget as Wt;
            use hipfire_specdecode_ngram::WriteTarget;
            let cfg = hipfire_specdecode_ngram::NgramConfig {
                orders: setup.orders.clone(),
                chain_floor: setup.chain_floor,
                max_spine: setup.max_spine,
                promote_count: setup.promote_count,
                write_target: match setup.write_target {
                    Wt::User => WriteTarget::User,
                    Wt::Topic => WriteTarget::Topic,
                    Wt::None => WriteTarget::None,
                },
                ..Default::default()
            };
            let mut ng = hipfire_specdecode_ngram::NgramSpec::new(cfg);
            if setup.persists() {
                let layout = hipfire_specdecode_ngram::ScopeLayout::new(&setup.store_root);
                let vocab = target.config.vocab_size;
                // The writable table belongs to one user. With no request identity
                // the daemon is single-tenant, so everything shares `local` — which
                // is correct there and unsafe the moment it is not, because
                // `next`/`next2` are plaintext and a shared writable table is a
                // continuation oracle.
                let scope = ngram_scope.unwrap_or_default();
                let user = scope.user_id.unwrap_or("local");
                if let Some(p) = layout.user(&setup.scope, user) {
                    if let Err(e) = ng.attach_user(&p, vocab, setup.blocks) {
                        eprintln!("[ngram] user store unavailable ({}): {e}", p.display());
                    }
                }
                // A topic table lives *under the user*, so it is private and may be
                // written. A topic table shared across users would need to be
                // read-only for the same reason the base one is.
                if let Some(t) = scope.session_type {
                    if let Some(p) = layout.topic(&setup.scope, t, Some(user)) {
                        // Writable because it lives under the user, so it is
                        // private. A topic table shared across users would have to
                        // be read-only, like the base tier.
                        if let Err(e) = ng.attach_topic(&p, vocab, setup.blocks, true) {
                            eprintln!("[ngram] topic store unavailable ({}): {e}", p.display());
                        }
                    }
                }
                if let Some(p) = layout.base(&setup.scope) {
                    if p.exists() {
                        if let Err(e) = ng.attach_base(&p) {
                            eprintln!("[ngram] base store unavailable ({}): {e}", p.display());
                        }
                    }
                }
            }
            ng
        })
    });
    // Seed from the prompt: prompt-echo is where a training-free drafter earns
    // most of its acceptance. `reset()` drops the previous request's rolling
    // history without touching the learned tables.
    if let Some(ng) = ngram.as_mut() {
        ng.reset_sequence();
        ng.observe(&prompt_tokens);
        ng.observe(&[first_token]);
    }

    // Decode loop — spec_step_dflash returns a committed batch per cycle.
    let mut emitted: Vec<u32> = vec![first_token];
    let mut streamed_tokens: Vec<u32> = Vec::new();
    // `bytes_fed_to_filter` is the index into the freshly-decoded byte
    // stream past which we have not yet handed bytes to the filter.
    // The filter owns UTF-8 boundary buffering and any future arch
    // quirks (Gemma 4 marker holdback, strip-think, byte-level stop_at);
    // see crates/engine/src/eos_filter.rs.
    let mut bytes_fed_to_filter = 0usize;
    let mut filter =
        chat_output_filter_from_profile(chat_template_profile.as_ref(), request_stop_sequences);
    let mut position = prompt_tokens.len();
    let mut seed_token = first_token;
    let mut spec_metrics = SpecMetrics::new(df.block_size);
    // Reset the per-request thread-local specialized accumulators (ddtree meta
    // tree-size + seed-oracle) so this request's `done` ext reflects only its
    // own cycles. The DFlash step records into these on this same thread.
    hipfire_arch_qwen35::speculative::reset_ddtree_meta_stats();
    hipfire_arch_qwen35::speculative::reset_seed_oracle_stats();
    // max_think_tokens enforcement state (mirrors the AR path).
    let mut think_count: usize = 0;
    let mut prev_in_think = false;
    let mut generated = 0usize;

    // Post-prefill compaction (FlashCASK pattern from dflash_spec_demo).
    // If the prompt already filled past budget+beta, compact once before
    // entering the spec loop so the first spec_step writes at physical slot
    // `budget`. compact_offset is maintained on target.kv_cache; subsequent
    // forwards inside spec_step_dflash read it for RoPE phase automatically.
    if let Some(ref ev) = m.eviction {
        if let Some(res) = ev.maybe_evict(gpu, &mut target.kv_cache, position).unwrap() {
            let pre_phys = position;
            tracing::debug!(
                "post-prefill evict: {} -> {} (compact_offset={})",
                pre_phys,
                res.new_physical,
                target.kv_cache.compact_offset,
            );
            position = res.new_physical;
            if !res.retain_mask.is_empty() {
                let _ = speculative::apply_eviction_retain_to_draft(
                    gpu,
                    &mut df.draft_scratch,
                    &res.retain_mask,
                    df.draft_config.num_extract(),
                    df.draft_config.hidden,
                    pre_phys,
                );
                speculative::compact_target_hidden_host(
                    &mut df.target_hidden_host,
                    &res.retain_mask,
                    df.draft_config.num_extract(),
                    df.draft_config.hidden,
                );
            }
        }
    }

    // Emit the first token immediately so TTFT is the prefill time.
    streamed_tokens.push(first_token);
    emit_committed_event(
        stdout,
        id,
        first_token,
        streamed_tokens.len() - 1,
        t0.elapsed().as_millis() as u64,
    );
    let all_bytes = tokenizer.decode_bytes(&streamed_tokens);
    let new_bytes = &all_bytes[bytes_fed_to_filter..];
    bytes_fed_to_filter = all_bytes.len();
    let first_filter_stop = emit_filter_action(stdout, id, filter.observe(new_bytes));
    generated += 1;

    // First-token EOS guard. The first token is already emitted above; if
    // it is itself a terminator, do not seed another drafted/verified block.
    // The committed-tail check inside the loop applies the same terminator
    // test to every subsequent token.
    let first_token_is_eos = first_token == target.config.eos_token
        || im_end_token == Some(first_token)
        || tokenizer.is_terminator(first_token);

    // NOT routed through `sampler::initial_rng_state()`, deliberately: this is the
    // u64 spec-decode state, and its only consumer below passes a hardcoded
    // `0.0_f32` temperature, so it is never consulted. Left constant rather than
    // randomized so the DFlash/DDTree path keeps a fixed, reproducible state; if
    // that call site ever takes a real temperature, route it then.
    let mut rng_state: u64 = 0x13579BDFu64;

    // Resolve `HIPFIRE_DDTREE_PATH_C` ONCE before the decode loop. The
    // previous version re-read the env-var on every spec cycle which
    // is microseconds of waste on a hot path. Validate eagerly: invalid
    // values fall back to spec_step_ddtree_batched (the documented
    // behavior) but warn so misconfigurations don't fail silently.
    //
    // Only meaningful when DDTree itself is enabled (HIPFIRE_DDTREE_BUDGET).
    // `phase1` runs Step 1 only (linear main-path verify); `phase2` adds
    // the lazy branch FA-only re-verify (Steps 2+3). See
    // `docs/plans/ddtree-path-c-main-path-first-from-lucebox.prd`.
    let path_c_mode_owned: Option<&'static str> = match std::env::var("HIPFIRE_DDTREE_PATH_C").ok()
    {
        None => None,
        Some(s) if s.is_empty() => None,
        Some(s) if s == "phase1" => Some("phase1"),
        Some(s) if s == "phase2" => Some("phase2"),
        Some(s) => {
            if df.ddtree.is_some() {
                tracing::warn!(
                    "HIPFIRE_DDTREE_PATH_C={:?} is not 'phase1' or 'phase2'. \
                     Falling back to spec_step_ddtree_batched.",
                    s
                );
            }
            None
        }
    };

    // Adaptive block sizing (scope doc item 3): the dspark cost-model
    // controller, driven with per-window accept depth + wall time. Chain mode
    // only (the DDTree steps take no block override). max_block is clamped to
    // the trained block so every scratch stays validly sized (item 4 — sizing
    // for B above trained — is not landed); an explicit HIPFIRE_DFLASH_BLOCK
    // pin wins over the controller.
    let mut block_controller = if df.adaptive_b
        && df.ddtree.is_none()
        && dflash_block_override().is_none()
        && df.block_size > 2
    {
        Some(
            hipfire_specdecode_dspark::dspark_block_controller::BlockController::new(
                df.block_size,
                2,
                df.block_size,
                0.18, // dormant cost prior; live window timing replaces it
            ),
        )
    } else {
        None
    };

    // Fast path exit conditions (mirrors the dflash_spec_demo outer loop).
    while !first_filter_stop && !first_token_is_eos && generated < max_tokens {
        if position + df.block_size >= ctx_capacity {
            break;
        }
        let window_started = Instant::now();

        // Dispatch: when DDTree is configured (HIPFIRE_DDTREE_BUDGET set
        // at startup), route through `spec_step_ddtree_batched`. Otherwise
        // keep the existing chain-mode `spec_step_dflash` path. The two
        // produce the same `SpecStepResult` shape so the rest of the loop
        // is unchanged. Note: `spec_step_ddtree_batched` is greedy-only
        // (temp=0); the daemon currently runs at 0.0_f32 so this matches.
        let path_c_mode = path_c_mode_owned;
        // Spine the n-gram tier proposed this cycle, if any. Kept alive across
        // the step call so `pld_spine` can borrow it.
        let mut ngram_spine: Option<Vec<u32>> = None;
        let step_result = if let Some(dd) = df.ddtree.as_mut() {
            if path_c_mode == Some("phase1") || path_c_mode == Some("phase2") {
                let phase2_snaps = if path_c_mode == Some("phase2") {
                    Some(Phase2Snapshots {
                        parent_pre_snap: &mut dd.path_c_parent_pre_snap,
                        main_end_snap: &mut dd.path_c_main_end_snap,
                    })
                } else {
                    None
                };
                spec_step_ddtree_path_c(
                    gpu,
                    &mut target,
                    &df.draft_weights,
                    &df.draft_config,
                    &mut df.draft_scratch,
                    &mut df.hidden_rb,
                    &mut df.target_hidden_host,
                    &mut df.target_snap,
                    &mut df.gdn_tape,
                    &df.verify_scratch,
                    position,
                    seed_token,
                    None, // ctx_slice = full history
                    dd.budget,
                    dd.topk,
                    phase2_snaps,
                )
            } else {
                spec_step_ddtree_batched(
                    gpu,
                    &mut target,
                    &df.draft_weights,
                    &df.draft_config,
                    &mut df.draft_scratch,
                    &mut df.hidden_rb,
                    &mut df.target_hidden_host,
                    &mut df.target_snap,
                    &mut dd.post_seed_snap,
                    &mut df.gdn_tape,
                    &dd.scratch,
                    &df.verify_scratch,
                    position,
                    seed_token,
                    None, // ctx_slice = full history
                    dd.budget,
                    dd.topk,
                )
            }
        } else {
            // n-gram first: a hit shrinks the verify batch to the spine and
            // skips the drafter forward entirely; a miss leaves this `None`
            // and the drafter runs.
            ngram_spine = ngram.as_mut().and_then(|n| n.draft().map(|sp| sp.to_vec()));
            spec_step_dflash(
                gpu,
                &mut target,
                Some(&df.draft_weights),
                Some(&df.draft_config),
                Some(&mut df.draft_scratch),
                Some(&mut df.hidden_rb),
                &mut df.target_hidden_host,
                &mut df.target_snap,
                &df.verify_scratch,
                position,
                seed_token,
                None, // ctx_slice = full history
                Some(&mut df.gdn_tape),
                0.0_f32, // temperature
                &mut rng_state,
                // Block override: env pin, else the adaptive controller.
                dflash_block_override().or_else(|| block_controller.as_ref().map(|c| c.block())),
                None, // ngram_cache
                &emitted,
                0.0_f32, // cactus_delta
                ngram_spine.as_deref(),
                1.0_f32, // repeat_penalty (off)
                0,       // repeat_window
            )
        };
        let step = match step_result {
            Ok(s) => s,
            Err(e) => {
                let _ = writeln!(
                    stdout,
                    r#"{{"type":"error","id":"{}","message":"spec_step: {}"}}"#,
                    id, e
                );
                let _ = stdout.flush();
                break;
            }
        };
        // Per the metrics-unification plan: proposed = drafted.len(), accepted,
        // committed = committed.len(). (`committed` includes the seed, length
        // accept+2.)
        spec_metrics.record_window(step.drafted.len(), step.accepted, step.committed.len());
        // Attribute only when the n-gram tier actually supplied the draft;
        // otherwise `step.accepted` belongs to the DFlash drafter.
        if ngram_spine.is_some() {
            if let Some(n) = ngram.as_mut() {
                n.record_acceptance(step.accepted);
            }
        }
        if let Some(c) = block_controller.as_mut() {
            // Full-window wall time (draft+verify), the controller's calibration
            // signal; n_verify = 1 + drafted block, matching its indexing.
            let t_window_ms = window_started.elapsed().as_secs_f32() * 1000.0;
            c.observe_timing(t_window_ms, step.drafted.len() + 1);
            c.observe(step.accepted, step.drafted.len());
        }
        let committed_tail: Vec<u32> = step.committed.iter().skip(1).copied().collect();
        if let Some(n) = ngram.as_mut() {
            n.observe(&committed_tail);
        }

        let mut hit_eos = false;
        let mut think_cap_hit = false;
        for &tok in &committed_tail {
            if generated >= max_tokens {
                break;
            }
            emitted.push(tok);
            streamed_tokens.push(tok);
            emit_committed_event(
                stdout,
                id,
                tok,
                streamed_tokens.len() - 1,
                t0.elapsed().as_millis() as u64,
            );
            let all_bytes = tokenizer.decode_bytes(&streamed_tokens);
            let new_bytes = &all_bytes[bytes_fed_to_filter..];
            bytes_fed_to_filter = all_bytes.len();
            if emit_filter_action(stdout, id, filter.observe(new_bytes)) {
                hit_eos = true;
                break;
            }
            generated += 1;
            if tok == target.config.eos_token
                || im_end_token == Some(tok)
                || tokenizer.is_terminator(tok)
            {
                hit_eos = true;
                break;
            }

            // max_think_tokens enforcement (mirrors the AR path). Track
            // <think>/<⁄think> in decoded text and count tokens inside.
            if max_think_tokens > 0 {
                let raw_so_far = tokenizer.decode_bytes(&streamed_tokens);
                let raw_str = std::str::from_utf8(&raw_so_far).unwrap_or("");
                let open_idx = raw_str.rfind("<think>");
                let close_idx = raw_str.rfind("</think>");
                let in_think = match (open_idx, close_idx) {
                    (Some(o), Some(c)) => o > c,
                    (Some(_), None) => true,
                    _ => false,
                };
                if in_think && !prev_in_think {
                    think_count = 0;
                }
                if in_think {
                    think_count += 1;
                }
                prev_in_think = in_think;

                if in_think && think_count >= max_think_tokens {
                    // Force-close: emit </think>\n and break out of this batch.
                    // Unlike the AR path we can't splice into the KV cache mid-
                    // spec-cycle, so we just stream the close text and break.
                    // The next request will start fresh.
                    let _ = writeln!(
                        stdout,
                        r#"{{"type":"token","id":"{}","text":"</think>\n"}}"#,
                        id
                    );
                    let _ = stdout.flush();
                    think_cap_hit = true;
                    break;
                }
            }
        }
        let rollback = speculative::spec_rollback_parity_decision_for_step(position, &step);
        if !rollback.allow_single_session {
            let _ = writeln!(
                stdout,
                r#"{{"type":"error","id":"{}","message":"DFlash rollback parity guard failed: {}"}}"#,
                id, rollback.reason
            );
            let _ = stdout.flush();
            break;
        }
        debug_assert!(!rollback.allow_multi_request_verify_batch);
        position = rollback.next_position;
        seed_token = step.bonus_token;
        // Per-cycle eviction (FlashCASK). Fires whenever current physical
        // has grown to budget+β since the last compaction. No-op when
        // physical < budget+β, so non-firing cycles pay only the check cost.
        if let Some(ref ev) = m.eviction {
            if let Some(res) = ev.maybe_evict(gpu, &mut target.kv_cache, position).unwrap() {
                let pre_phys = position;
                position = res.new_physical;
                if !res.retain_mask.is_empty() {
                    let _ = speculative::apply_eviction_retain_to_draft(
                        gpu,
                        &mut df.draft_scratch,
                        &res.retain_mask,
                        df.draft_config.num_extract(),
                        df.draft_config.hidden,
                        pre_phys,
                    );
                    speculative::compact_target_hidden_host(
                        &mut df.target_hidden_host,
                        &res.retain_mask,
                        df.draft_config.num_extract(),
                        df.draft_config.hidden,
                    );
                }
            }
        }
        if hit_eos || think_cap_hit {
            break;
        }
    }

    // Snapshot the n-gram telemetry while we still own the state, then hand the
    // state back to the load so the next request inherits its hot tier and
    // promotion counters. Both have to happen here: `df` is borrowed from `m`,
    // and the `m.q35_*` writes below end that borrow.
    let ngram_ext: Option<serde_json::Value> = ngram.as_ref().map(|n| {
        use hipfire_specdecode_ngram::Tier;
        let st = n.stats();
        let mut by_tier = serde_json::Map::new();
        for t in Tier::ALL {
            if st.lookups_by_tier[t.idx()] == 0 && st.drafted_in(t) == 0 && n.store(t).is_none() {
                continue;
            }
            let mut row = serde_json::json!({
                "lookups": st.lookups_by_tier[t.idx()],
                "hits": st.hits_by_tier[t.idx()],
                "drafted": st.drafted_in(t),
                "accepted": st.accepted_in(t),
                "marginal_share": st.marginal_share(t),
            });
            if let Some(store) = n.store(t) {
                let (recs, blks) = store.occupancy();
                row["records"] = serde_json::json!(recs);
                row["blocks_used"] = serde_json::json!(blks);
                row["blocks_free"] = serde_json::json!(store.free_blocks());
                row["read_only"] = serde_json::json!(store.is_read_only());
            }
            by_tier.insert(t.name().to_string(), row);
        }
        serde_json::json!({
            "steps": st.steps,
            "steps_proposed": st.steps_proposed,
            "coverage": st.coverage(),
            "drafted": st.drafted,
            "accepted": st.accepted,
            "accepted_per_step": st.accepted_per_step(),
            "verify_efficiency": st.verify_efficiency(),
            "hot_entries": n.hot_len(),
            "merge_backlog": n.merge_backlog_len(),
            "by_tier": by_tier,
        })
    });
    if let Some(n) = ngram {
        df.ngram_live = Some((ngram_key, n));
    }

    // Put target state back on LoadedModel so the next request sees fresh
    // (reset) state. We zero DN/kv on entry anyway, but we still need the
    // ownership back.
    m.q35_weights = Some(target.weights);
    put_qwen35_state_into_model(m, target.kv_cache, target.dn_state);
    m.q35_scratch = Some(target.scratch);
    m.active.cursor.seq_pos = position;
    m.active.cursor.conversation_tokens = emitted.clone();

    let t_end = Instant::now();
    let total_s = t_end.duration_since(t0).as_secs_f64();
    let prefill_s = t_prefill.duration_since(t0).as_secs_f64();
    let decode_s = t_end.duration_since(t_prefill).as_secs_f64();
    let tok_s = if total_s > 0.0 {
        generated as f64 / total_s
    } else {
        0.0
    };
    let decode_tok_s = if decode_s > 0.0 {
        generated as f64 / decode_s
    } else {
        0.0
    };
    let prefill_tok_s = if prefill_s > 0.0 {
        prompt_tokens.len() as f64 / prefill_s
    } else {
        0.0
    };
    // Non-metric done fields (prefill/decode timings) + legacy `cycles` alias
    // (== windows). Per PRD §3.1, when PFlash is bypassed (e.g.
    // dflash_decode_active for this branch) the `done` object must also surface
    // the bypass reason and alpha alongside the dflash perf metrics.
    let mut ext = serde_json::json!({
        "prefill_tokens": prompt_tokens.len(),
        "prefill_ms": prefill_s * 1000.0,
        "prefill_tok_s": prefill_tok_s,
        "decode_tok_s": decode_tok_s,
        "ttft_ms": prefill_s * 1000.0,
        "cycles": spec_metrics.windows,
    });
    if let (Some(r), Some(a)) = (pflash_bypass_reason, pflash_alpha) {
        ext["pflash"] = serde_json::json!({ "bypass_reason": r, "alpha": a });
    }
    // Specialized per-request spec metrics, drained from the qwen35 DFlash
    // container (migrated off process-global atomics in P6). Each is `None`
    // when this request drove no such cycle, so absent means "not exercised".
    if let Some(ddm) = hipfire_arch_qwen35::speculative::read_ddtree_meta_stats().to_json() {
        ext["ddtree_meta"] = ddm;
    }
    if let Some(so) = hipfire_arch_qwen35::speculative::read_seed_oracle_stats().to_json() {
        ext["seed_oracle"] = so;
    }
    // n-gram spec-decode, when this request ran it. `by_tier` is the whole
    // point: tiers are probed most-specific first and the first hit wins, so a
    // tier's accepted count is the marginal value it added over every tier
    // below it — which answers "does the cold table earn its bytes?" from live
    // traffic rather than another experiment.
    if let Some(ng) = ngram_ext {
        ext["ngram"] = ng;
    }
    emit_spec_done(
        stdout,
        id,
        generated,
        tok_s,
        "dflash",
        &spec_metrics,
        Some(ext),
    );
}

/// Multi-GPU pipeline-parallel AR decode (Stage 7 of #58). Mirrors the pp=1
/// `generate` Qwen3.5 branch feature-for-feature: ChatFrame ChatML wrap,
/// EosFilter UTF-8 streaming + strip-think + stop_at, LoopGuard n-gram
/// detection, repeat penalty, attractor block on unclosed tool/think
/// openers, max_think_tokens force-close, budget-alert nudge, ChatML \n
/// trailer. Forward calls fan out to per-device tensors via
/// `gpus.devices[dev]` and `scratch_set.per_device[dev]`; the final
/// sample lives on `gpus.output_device`. DFlash, CASK, PFlash, VL and
/// arch_id < 5 are refused upstream at load.
#[allow(clippy::too_many_arguments)]
/// Multi-GPU pipeline-parallel generate (`pp > 1`): drives the per-stage
/// `Gpus` orchestrator + `Qwen35ScratchSet` through prefill and the per-token
/// decode loop, streaming tokens. Single-session; DFlash/MTP/CASK/VL are refused
/// into this path at load time.
pub fn generate_multi(
    m: &mut LoadedModel,
    gpu: &mut hipfire_rdna::Gpu,
    pflash_state: Option<&mut hipfire_arch_qwen35::pflash::PflashState>,
    pflash_cfg: Option<&hipfire_arch_qwen35::pflash::PflashConfig>,
    stdout: &mut dyn std::io::Write,
    id: &str,
    prompt: &str,
    system_prompt: Option<&str>,
    temp: f32,
    top_p: f32,
    top_k: usize,
    max_tokens: usize,
    repeat_penalty: f32,
    repeat_window: usize,
    presence_penalty: f32,
    frequency_penalty: f32,
    budget_alert_at_tok: usize,
    budget_alert_text: &str,
    max_think_tokens: usize,
    assistant_prefix: prompt_frame::AssistantPrefix,
    tools: Option<&[serde_json::Value]>,
    messages_history: Option<&[prompt_frame::Message]>,
    request_stop_sequences: &[String],
    // Explicit per-request `"raw"` override; `None` = auto (raw iff the model
    // has no chat_template). Threaded rather than read from a global — see
    // `effective_raw` for the cross-request leak that motivated it.
    raw_override: Option<bool>,
) {
    let tokenizer = m.tokenizer.as_ref().unwrap();
    let prompt_est = tokenizer.encode(prompt).len() + 20;
    if m.active.cursor.seq_pos + prompt_est + max_tokens > m.max_seq {
        tracing::warn!(
            "context full ({}/{}) — resetting conversation",
            m.active.cursor.seq_pos,
            m.max_seq
        );
        m.active.cursor.seq_pos = 0;
        m.active.cursor.conversation_tokens.clear();
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
        if let Some(kv) = m.active.sequence_state.as_mut().and_then(|s| s.kv_mut()) {
            kv.compact_offset = 0;
        }
    }

    let im_end = tokenizer.encode("<|im_end|>");
    let nl = tokenizer.encode("\n");
    let raw_q_tokens = tokenizer.encode(prompt);

    // PFlash compression on first turn (seq_pos == 0). Drafter runs on the
    // daemon's single-GPU `gpu` handle, which binds to the same physical
    // device as `pp_gpus.devices[0]` (HIP enumerates within ROCR_VISIBLE).
    // VRAM is shared between the two Gpu handles via the HIP heap, so
    // drafter weights coexist with the target's dev 0 portion. Output is
    // a Vec<u32> of kept token IDs which feeds forward_prefill_batch_multi
    // unchanged. Mode=Off / drafter unloaded falls through to raw tokens.
    let request_kind = match tokenizer.special_token_id("<tool_call>") {
        Some(tid) => {
            let in_user = raw_q_tokens.iter().any(|&t| t == tid);
            let in_system = system_prompt
                .map(|s| tokenizer.encode(s).iter().any(|&t| t == tid))
                .unwrap_or(false);
            if in_user || in_system {
                hipfire_arch_qwen35::pflash::RequestKind::ToolCall
            } else {
                hipfire_arch_qwen35::pflash::RequestKind::Text
            }
        }
        None => hipfire_arch_qwen35::pflash::RequestKind::Text,
    };
    let q_tokens = if let (Some(state), Some(cfg)) = (pflash_state, pflash_cfg) {
        if m.active.cursor.seq_pos == 0 {
            match hipfire_arch_qwen35::pflash::maybe_compress_prompt(
                gpu,
                state,
                cfg,
                &raw_q_tokens,
                request_kind,
                &[],
            ) {
                Ok(hipfire_arch_qwen35::pflash::PflashDecision::Compressed(cp)) => {
                    let _ = writeln!(
                        stdout,
                        r#"{{"type":"pflash_compressed","id":"{}","source_tokens":{},"kept_tokens":{},"keep_ratio":{:.6},"source_md5":"{}","compressed_md5":"{}","score_ms":{},"total_ms":{}}}"#,
                        id,
                        cp.source_tokens,
                        cp.kept_tokens,
                        cp.kept_tokens as f32 / cp.source_tokens.max(1) as f32,
                        cp.source_md5,
                        cp.compressed_md5,
                        cp.timings.score_ms,
                        cp.timings.total_ms,
                    );
                    let _ = stdout.flush();
                    cp.token_ids
                }
                Ok(hipfire_arch_qwen35::pflash::PflashDecision::Bypass { reason }) => {
                    if !matches!(reason, hipfire_arch_qwen35::pflash::BypassReason::ModeOff) {
                        let _ = writeln!(
                            stdout,
                            r#"{{"type":"pflash_bypass","id":"{}","reason":"{}"}}"#,
                            id,
                            reason.as_str().replace('"', "'"),
                        );
                        let _ = stdout.flush();
                    }
                    raw_q_tokens
                }
                Err(e) => {
                    let _ = writeln!(
                        stdout,
                        r#"{{"type":"pflash_error","id":"{}","reason":"{}"}}"#,
                        id,
                        e.to_string().replace('"', "'"),
                    );
                    let _ = stdout.flush();
                    raw_q_tokens
                }
            }
        } else {
            raw_q_tokens
        }
    } else {
        raw_q_tokens
    };

    // ChatML framing — two paths, same shape as the single-GPU AR
    // generate() (line 3147+):
    //
    //   1) HIPFIRE_JINJA_CHAT=1 + model has chat_template + seq_pos==0
    //      → render via JinjaChatFrame so structured tools/messages
    //      reach the upstream template. PFlash compression is bypassed
    //      under Jinja (q_tokens is unused; the rendered prompt string
    //      is re-tokenized straight through).
    //
    //   2) Default: hand-rolled ChatFrame::Plain scaffold, byte-
    //      identical to the pp=1 default path so multi-turn behavior
    //      matches between pp=1 and pp>1 when both run the same prompt.
    let jinja_enabled = std::env::var("HIPFIRE_JINJA_CHAT").ok().as_deref() == Some("1");
    let try_jinja = jinja_enabled && m.active.cursor.seq_pos == 0 && m.chat_template.is_some();
    let new_tokens = if try_jinja {
        let template = m.chat_template.as_ref().unwrap();
        let frame = prompt_frame::JinjaChatFrame {
            tokenizer,
            template,
            system: system_prompt,
            user: prompt,
            enable_thinking: max_think_tokens != 1,
            bos_token: None,
        };
        let render_result = if tools.is_some() || messages_history.is_some() {
            let synthesized: Vec<prompt_frame::Message>;
            let messages_slice: &[prompt_frame::Message] = match messages_history {
                Some(m) => m,
                None => {
                    let mut v = Vec::new();
                    if let Some(sys) = system_prompt {
                        v.push(prompt_frame::Message {
                            role: prompt_frame::Role::System,
                            content: sys.to_string(),
                            tool_calls: Vec::new(),
                            tool_call_id: None,
                        });
                    }
                    v.push(prompt_frame::Message {
                        role: prompt_frame::Role::User,
                        content: prompt.to_string(),
                        tool_calls: Vec::new(),
                        tool_call_id: None,
                    });
                    synthesized = v;
                    &synthesized
                }
            };
            frame.render_messages(messages_slice, tools, None)
        } else {
            frame.render()
        };
        match render_result {
            Ok(rendered) => tokenizer.encode(&rendered),
            Err(e) => {
                tracing::warn!("jinja render failed in pp path ({e}) — falling back to Plain");
                prompt_frame::ChatFrame {
                    tokenizer,
                    system: if m.active.cursor.seq_pos == 0 {
                        system_prompt
                    } else {
                        None
                    },
                    user: "",
                    assistant_prefix,
                    raw: effective_raw(m, raw_override),
                }
                .build_with_user_tokens(&q_tokens)
            }
        }
    } else {
        prompt_frame::ChatFrame {
            tokenizer,
            system: if m.active.cursor.seq_pos == 0 {
                system_prompt
            } else {
                None
            },
            user: "",
            assistant_prefix,
            raw: effective_raw(m, raw_override),
        }
        .build_with_user_tokens(&q_tokens)
    };

    let trailer = nl.len();
    if m.active.cursor.seq_pos + new_tokens.len() + max_tokens + trailer > m.physical_cap {
        let _ = writeln!(
            stdout,
            r#"{{"type":"error","id":"{}","message":"request exceeds loaded KV budget: seq_pos={} + prefill={} + max_tokens={} + trailer={} > physical_cap={} — reload model with a larger max_seq"}}"#,
            id,
            m.active.cursor.seq_pos,
            new_tokens.len(),
            max_tokens,
            trailer,
            m.physical_cap
        );
        let _ = stdout.flush();
        return;
    }

    let im_end_token = if im_end.len() == 1 {
        Some(im_end[0])
    } else {
        None
    };
    let tool_call_pair = match (
        tokenizer.special_token_id("<tool_call>"),
        tokenizer.special_token_id("</tool_call>"),
    ) {
        (Some(o), Some(c)) => Some((o, c)),
        _ => None,
    };
    let think_pair = match (
        tokenizer.special_token_id("<think>"),
        tokenizer.special_token_id("</think>"),
    ) {
        (Some(o), Some(c)) => Some((o, c)),
        _ => None,
    };

    let prefill_tokens = new_tokens.len();
    let t0 = Instant::now();
    let chat_template_profile = m.chat_template_profile.clone();

    let config = m.q35_config.as_ref().unwrap();
    let weights = m.q35_weights.as_ref().unwrap();
    let scratch_set = m.pp_scratch_set.as_ref().unwrap();
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
    let gpus = m.pp_gpus.as_mut().unwrap();

    let dev_last = gpus.output_device;
    let vocab_size = config.vocab_size;
    let repeat_buf_cap =
        (scratch_set.per_device[dev_last].repeat_buf.buf.size() / 4).min(repeat_window);

    if let Err(e) = qwen35::forward_prefill_batch_multi(
        gpus,
        weights,
        config,
        &new_tokens,
        m.active.cursor.seq_pos,
        kv,
        dn,
        scratch_set,
    ) {
        let _ = writeln!(
            stdout,
            r#"{{"type":"error","id":"{}","message":"forward_prefill_batch_multi: {}"}}"#,
            id, e
        );
        let _ = stdout.flush();
        return;
    }
    m.active.cursor.seq_pos += new_tokens.len();
    m.active
        .cursor
        .conversation_tokens
        .extend_from_slice(&new_tokens);

    // ngram scope: generated tokens only (matches pp=1).
    let ngram_scope_start = m.active.cursor.conversation_tokens.len();

    let mut rng_state: u32 = hipfire_runtime::sampler::initial_rng_state();

    let attractor_pairs: Vec<(u32, u32)> = tool_call_pair
        .into_iter()
        .chain(think_pair.into_iter())
        .collect();

    // First sample on the output device.
    let ngram_scope = &m.active.cursor.conversation_tokens[ngram_scope_start..];
    let mut blocked0: Vec<u32> = Vec::new();
    collect_unclosed_attractor_blocks(ngram_scope, &attractor_pairs, 20, 2, &mut blocked0);
    let cfg0 = SamplerConfig {
        temperature: temp,
        top_p,
        top_k,
        repeat_penalty,
        repeat_window: repeat_buf_cap,
        presence_penalty,
        frequency_penalty,
        blocked_tokens: blocked0,
    };
    let tok0 = {
        let s_last = &scratch_set.per_device[dev_last];
        let g_last = &mut gpus.devices[dev_last];
        sampler::sample(
            g_last,
            &s_last.logits,
            &s_last.sample_buf,
            &s_last.repeat_buf,
            vocab_size,
            ngram_scope,
            &cfg0,
            &mut rng_state,
        )
    };
    let t_prefill = Instant::now();
    let mut next_token = tok0;

    let mut generated = 0usize;
    let mut streamed_tokens: Vec<u32> = Vec::new();
    let mut bytes_fed_to_filter = 0usize;
    let mut filter =
        chat_output_filter_from_profile(chat_template_profile.as_ref(), request_stop_sequences);
    let mut alert_fired = false;
    let mut think_count: usize = 0;
    let mut prev_in_think: bool = false;
    let loop_guard = loop_guard_from_runtime_config();

    while generated < max_tokens {
        // Cooperative cancellation (SIGUSR1). Same KV-safe top-of-loop chokepoint
        // as the single-GPU path: all committed tokens are already written across
        // the pipeline, the pending `next_token` is not — dropping it leaves the
        // multi-GPU KV consistent, identical to a natural `max_tokens` stop.
        if hipfire_runtime::take_generation_cancel() {
            break;
        }
        generated += 1;
        m.active.cursor.conversation_tokens.push(next_token);
        streamed_tokens.push(next_token);
        emit_committed_event(
            stdout,
            id,
            next_token,
            streamed_tokens.len() - 1,
            t0.elapsed().as_millis() as u64,
        );
        let all_bytes = tokenizer.decode_bytes(&streamed_tokens);
        let new_bytes = &all_bytes[bytes_fed_to_filter..];
        bytes_fed_to_filter = all_bytes.len();
        let filter_stop = emit_filter_action(stdout, id, filter.observe(new_bytes));

        if let Err(e) = qwen35::forward_scratch_multi(
            gpus,
            weights,
            config,
            next_token,
            m.active.cursor.seq_pos,
            kv,
            dn,
            scratch_set,
        ) {
            let _ = writeln!(
                stdout,
                r#"{{"type":"error","id":"{}","message":"forward_scratch_multi decode: {}"}}"#,
                id, e
            );
            let _ = stdout.flush();
            return;
        }
        m.active.cursor.seq_pos += 1;

        if next_token == config.eos_token {
            break;
        }
        if filter_stop {
            break;
        }
        if im_end_token == Some(next_token) {
            break;
        }
        if tokenizer.is_terminator(next_token) {
            break;
        }

        // max_think_tokens enforcement: same decoded-text scan as pp=1.
        if max_think_tokens > 0 {
            let raw_so_far = tokenizer.decode_bytes(&streamed_tokens);
            let raw_str = std::str::from_utf8(&raw_so_far).unwrap_or("");
            let open_idx = raw_str.rfind("<think>");
            let close_idx = raw_str.rfind("</think>");
            let in_think = match (open_idx, close_idx) {
                (Some(o), Some(c)) => o > c,
                (Some(_), None) => true,
                _ => false,
            };
            if in_think {
                if !prev_in_think {
                    think_count = 1;
                } else {
                    think_count += 1;
                }
            } else {
                think_count = 0;
            }
            prev_in_think = in_think;

            if in_think && think_count >= max_think_tokens {
                let close_tokens = tokenizer.encode("</think>\n");
                let budget_left = max_tokens.saturating_sub(generated);
                let take = close_tokens.len().min(budget_left);
                for &t in &close_tokens[..take] {
                    if let Err(e) = qwen35::forward_scratch_multi(
                        gpus,
                        weights,
                        config,
                        t,
                        m.active.cursor.seq_pos,
                        kv,
                        dn,
                        scratch_set,
                    ) {
                        tracing::error!("max_think close forward_scratch_multi: {}", e);
                        break;
                    }
                    m.active.cursor.seq_pos += 1;
                    m.active.cursor.conversation_tokens.push(t);
                    streamed_tokens.push(t);
                    emit_committed_event(
                        stdout,
                        id,
                        t,
                        streamed_tokens.len() - 1,
                        t0.elapsed().as_millis() as u64,
                    );
                    let all_bytes = tokenizer.decode_bytes(&streamed_tokens);
                    let new_bytes = &all_bytes[bytes_fed_to_filter..];
                    bytes_fed_to_filter = all_bytes.len();
                    let _ = emit_filter_action(stdout, id, filter.observe(new_bytes));
                    generated += 1;
                }
                think_count = 0;
                prev_in_think = false;
                if generated >= max_tokens {
                    break;
                }
            }
        }

        // N-gram loop detector (token-side, no GPU work).
        if let Some(StopReason::NgramRepeat { count, .. }) = loop_guard.check(&streamed_tokens) {
            let window_len = loop_guard.window_len(streamed_tokens.len());
            let _ = writeln!(
                stdout,
                r#"{{"type":"info","id":"{}","message":"ngram loop detected (4gram repeated {}× in last {} tokens) — forcing EOS"}}"#,
                id, count, window_len
            );
            let _ = stdout.flush();
            break;
        }

        // Budget-alert injection: gated to inside an open <think> block.
        if !alert_fired
            && budget_alert_at_tok > 0
            && generated >= budget_alert_at_tok
            && !budget_alert_text.is_empty()
        {
            alert_fired = true;
            let raw_so_far = tokenizer.decode_bytes(&streamed_tokens);
            let raw_str = std::str::from_utf8(&raw_so_far).unwrap_or("");
            let in_think = match (raw_str.rfind("<think>"), raw_str.rfind("</think>")) {
                (Some(o), Some(c)) => o > c,
                (Some(_), None) => true,
                _ => false,
            };
            if !in_think {
                let _ = writeln!(
                    stdout,
                    r#"{{"type":"info","id":"{}","message":"budget_alert skipped: not inside an open <think> block"}}"#,
                    id
                );
                let _ = stdout.flush();
                let ngram_scope = &m.active.cursor.conversation_tokens[ngram_scope_start..];
                let mut blocked: Vec<u32> = Vec::new();
                collect_unclosed_attractor_blocks(
                    ngram_scope,
                    &attractor_pairs,
                    20,
                    2,
                    &mut blocked,
                );
                let cfg = SamplerConfig {
                    temperature: temp,
                    top_p,
                    top_k,
                    repeat_penalty,
                    repeat_window: repeat_buf_cap,
                    presence_penalty,
                    frequency_penalty,
                    blocked_tokens: blocked,
                };
                next_token = {
                    let s_last = &scratch_set.per_device[dev_last];
                    let g_last = &mut gpus.devices[dev_last];
                    sampler::sample(
                        g_last,
                        &s_last.logits,
                        &s_last.sample_buf,
                        &s_last.repeat_buf,
                        vocab_size,
                        ngram_scope,
                        &cfg,
                        &mut rng_state,
                    )
                };
                continue;
            }
            let nudge_tokens = tokenizer.encode(budget_alert_text);
            let budget_left = max_tokens.saturating_sub(generated);
            let nudge_len = nudge_tokens.len().min(budget_left);
            let need_kv = m.active.cursor.seq_pos
                + nudge_len
                + (max_tokens - generated - nudge_len)
                + nl.len();
            if nudge_len > 0 && need_kv <= m.physical_cap {
                for &tok in &nudge_tokens[..nudge_len] {
                    m.active.cursor.conversation_tokens.push(tok);
                    streamed_tokens.push(tok);
                    emit_committed_event(
                        stdout,
                        id,
                        tok,
                        streamed_tokens.len() - 1,
                        t0.elapsed().as_millis() as u64,
                    );
                    let all_bytes2 = tokenizer.decode_bytes(&streamed_tokens);
                    let new_bytes2 = &all_bytes2[bytes_fed_to_filter..];
                    bytes_fed_to_filter = all_bytes2.len();
                    let _ = emit_filter_action(stdout, id, filter.observe(new_bytes2));
                    if let Err(e) = qwen35::forward_scratch_multi(
                        gpus,
                        weights,
                        config,
                        tok,
                        m.active.cursor.seq_pos,
                        kv,
                        dn,
                        scratch_set,
                    ) {
                        tracing::error!("budget_alert forward_scratch_multi: {}", e);
                        break;
                    }
                    m.active.cursor.seq_pos += 1;
                    generated += 1;
                }
            } else if nudge_len < nudge_tokens.len() {
                let _ = writeln!(
                    stdout,
                    r#"{{"type":"info","id":"{}","message":"budget_alert clipped or skipped: nudge_len={} budget_left={}"}}"#,
                    id, nudge_len, budget_left
                );
                let _ = stdout.flush();
            } else {
                let _ = writeln!(
                    stdout,
                    r#"{{"type":"info","id":"{}","message":"budget_alert skipped: not enough KV headroom"}}"#,
                    id
                );
                let _ = stdout.flush();
            }
            if generated >= max_tokens {
                break;
            }
        }

        // Steady-state sample.
        let ngram_scope = &m.active.cursor.conversation_tokens[ngram_scope_start..];
        let mut blocked: Vec<u32> = Vec::new();
        collect_unclosed_attractor_blocks(ngram_scope, &attractor_pairs, 20, 2, &mut blocked);
        let cfg = SamplerConfig {
            temperature: temp,
            top_p,
            top_k,
            repeat_penalty,
            repeat_window: repeat_buf_cap,
            presence_penalty,
            frequency_penalty,
            blocked_tokens: blocked,
        };
        next_token = {
            let s_last = &scratch_set.per_device[dev_last];
            let g_last = &mut gpus.devices[dev_last];
            sampler::sample(
                g_last,
                &s_last.logits,
                &s_last.sample_buf,
                &s_last.repeat_buf,
                vocab_size,
                ngram_scope,
                &cfg,
                &mut rng_state,
            )
        };
    }

    // ChatML \n trailer so the next turn opens cleanly.
    if im_end_token == Some(*m.active.cursor.conversation_tokens.last().unwrap_or(&0))
        && !nl.is_empty()
    {
        for &t in &nl {
            if let Err(e) = qwen35::forward_scratch_multi(
                gpus,
                weights,
                config,
                t,
                m.active.cursor.seq_pos,
                kv,
                dn,
                scratch_set,
            ) {
                tracing::error!("trailer forward_scratch_multi: {}", e);
                break;
            }
            m.active.cursor.seq_pos += 1;
            m.active.cursor.conversation_tokens.push(t);
        }
    }

    let t_end = Instant::now();
    let total_s = t_end.duration_since(t0).as_secs_f64();
    let prefill_s = t_prefill.duration_since(t0).as_secs_f64();
    let decode_s = t_end.duration_since(t_prefill).as_secs_f64();
    let tok_s = if total_s > 0.0 {
        generated as f64 / total_s
    } else {
        0.0
    };
    let prefill_tok_s = if prefill_s > 0.0 {
        prefill_tokens as f64 / prefill_s
    } else {
        0.0
    };
    let decode_tok_s = if decode_s > 0.0 {
        generated as f64 / decode_s
    } else {
        0.0
    };
    let _ = writeln!(
        stdout,
        r#"{{"type":"done","id":"{}","tokens":{},"tok_s":{:.1},"prefill_tokens":{},"prefill_ms":{:.1},"prefill_tok_s":{:.1},"decode_tok_s":{:.1},"ttft_ms":{:.1}}}"#,
        id,
        generated,
        tok_s,
        prefill_tokens,
        prefill_s * 1000.0,
        prefill_tok_s,
        decode_tok_s,
        prefill_s * 1000.0
    );
    let _ = stdout.flush();
}

/// Cross-token state of one qwen35 decode loop.
///
/// Hoisted out of [`generate`]'s stack frame so that advancing one token is a
/// **function call** rather than a loop iteration. That is the entire point:
/// executor v2's march loop needs a quantum it can step, and a `while` body
/// holding nine live locals is not one. See
/// `docs/plans/2026-08-21-executor-v2-march-loop.md` §M3b0.
///
/// It deliberately owns **no borrow of the GPU or the model**. A serial executor
/// hands the GPU to another stream between quanta, so anything a stream keeps
/// across a suspension must not borrow it; the resources arrive per step
/// instead. A struct that held `&mut Gpu` would pin the device to one stream for
/// its whole life and defeat the interleaving it exists to enable.
struct Qwen35DecodeState {
    /// Sampler RNG. Advanced by `sampler::sample`; greedy decode never consults
    /// it, which is why greedy baselines do not move when this is threaded.
    rng_state: u32,
    /// Sampled but NOT yet committed: written to the KV cache at the top of the
    /// next step. The cancellation check is placed before that write on purpose
    /// — see the KV-safe chokepoint comment in [`qwen35_decode_one`].
    next_token: u32,
    generated: usize,
    streamed_tokens: Vec<u32>,
    /// Index into the freshly-decoded byte stream past which bytes have not yet
    /// been handed to `filter`; the filter owns UTF-8 boundary buffering.
    bytes_fed_to_filter: usize,
    filter: EosFilter,
    alert_fired: bool,
    /// `max_think_tokens` enforcement. Counts only while inside an open
    /// `<think>` block, and re-arms if the model opens another one.
    think_count: usize,
    prev_in_think: bool,
}

/// Per-request decode settings that do not change between tokens.
///
/// Split from [`Qwen35DecodeState`] along the mutable/immutable line: the state
/// is what a suspended stream must carry, this is what it can be handed again on
/// resume. Keeping them apart is also what stops the step function from taking
/// thirty arguments.
struct Qwen35DecodeCfg {
    max_tokens: usize,
    max_think_tokens: usize,
    budget_alert_at_tok: usize,
    budget_alert_text: String,
    im_end_token: Option<u32>,
    /// `tokenizer.encode("\n").len()` — only the length is read, by the
    /// budget-alert KV headroom check.
    nl_len: usize,
    vocab_size: usize,
    /// Candidates retained before nucleus sampling. Threaded from the request
    /// (`0` = the backend's full candidate set); the daemon defaults it to 20,
    /// which is the literal this replaced.
    top_k: usize,
    /// Repeat window bounded by the GPU `repeat_buf` capacity.
    repeat_buf_cap: usize,
    /// Start of the repeat-penalty ngram scope: generated tokens only, never the
    /// prompt.
    ngram_scope_start: usize,
    attractor_pairs: Vec<(u32, u32)>,
    loop_guard: hipfire_generate::loop_guard::LoopGuard,
    temperature: f32,
    top_p: f32,
    repeat_penalty: f32,
    presence_penalty: f32,
    frequency_penalty: f32,
}

/// What one step of the qwen35 decode loop decided.
pub enum Qwen35Step {
    /// The next token was sampled; the loop continues.
    Continue,
    /// A stop condition fired (EOS, terminator, filter, loop guard, cancel, or a
    /// spent budget). The caller breaks and emits the done frame as before.
    Stop,
    /// A forward failed. The message is formatted but NOT written here, and the
    /// caller unwinds.
    ///
    /// This is not stylistic. `qwen35_restore_or_error` takes the session **by
    /// value**, while `kv` and `dn` are disjoint `&mut` borrows *out of* that
    /// same session. Inline in a loop body, NLL accepts the move because the
    /// borrows are dead on a path that returns immediately. Across a function
    /// signature they must all coexist, so the consuming call cannot live in
    /// here — it has to happen where the borrows have been released.
    Failed(String),
}

/// Advance one qwen35 decode step: commit the pending token, write its K/V,
/// apply the stop/think/loop-guard/budget-alert rules, and sample the next one.
///
/// Extracted verbatim from [`generate`]'s `while generated < max_tokens` body.
/// It is a **pure move**: every branch, every emit, and every arithmetic
/// operation is the code that was there before, so output is byte-identical by
/// construction rather than by argument.
#[allow(clippy::too_many_arguments)]
/// §M7's sample half, lifted verbatim out of `qwen35_decode_one`.
///
/// This is the second of the two halves a batched step needs. The forward that
/// produced `scratch.logits` has already run; everything here is per-stream and
/// touches no GPU weights — the attractor block list, the per-stream
/// `SamplerConfig`, the RNG. That is precisely the state a batched driver must
/// keep with each `Qwen35Generation` rather than reimplement against a shared
/// envelope, and making it a function is what lets N handles each sample from
/// their own logits after one shared forward.
///
/// Note this is the NORMAL path's sample. The budget-alert nudge samples
/// separately at its own site and then issues extra single-token forwards, so a
/// batched round must exclude a stream that is about to fire one.
/// Sample from an explicit logits tensor.
///
/// Split out from `qwen35_sample_next` because a batched forward writes
/// **per-session** logits — one tensor per row — while the single-stream
/// `forward_scratch` writes the one shared `scratch.logits`. Everything else is
/// per-stream and unchanged: the attractor blocks, the `SamplerConfig`, the RNG.
/// `scratch` is still passed for `sample_buf` / `repeat_buf`, which are scratch
/// proper and reused sequentially across rows.
fn qwen35_sample_next_from(
    gpu: &mut hipfire_rdna::Gpu,
    logits: &hipfire_rdna::GpuTensor,
    scratch: &qwen35::Qwen35Scratch,
    cursor: &crate::session::SessionCursor,
    cfg: &Qwen35DecodeCfg,
    st: &mut Qwen35DecodeState,
) {
    // Decide which paired-opener tokens (if any) trip the depth
    // threshold over a 20-token window. #111 attractor block —
    // cheap when not tripped, ~5 µs per blocked token when
    // tripped (single 4-byte H2D into the logits buffer
    // performed inside sampler::sample).
    let ngram_scope = &cursor.conversation_tokens[cfg.ngram_scope_start..];
    let mut blocked: Vec<u32> = Vec::new();
    collect_unclosed_attractor_blocks(ngram_scope, &cfg.attractor_pairs, 20, 2, &mut blocked);
    let sampler_cfg = SamplerConfig {
        temperature: cfg.temperature,
        top_p: cfg.top_p,
        top_k: cfg.top_k,
        repeat_penalty: cfg.repeat_penalty,
        repeat_window: cfg.repeat_buf_cap,
        presence_penalty: cfg.presence_penalty,
        frequency_penalty: cfg.frequency_penalty,
        blocked_tokens: blocked,
    };
    // GPU sample: reads scratch.logits (already on GPU), writes
    // token+rng to scratch.sample_buf. Blocks only on the 8-byte
    // D2H readback inside sampler::sample.
    st.next_token = sampler::sample(
        gpu,
        logits,
        &scratch.sample_buf,
        &scratch.repeat_buf,
        cfg.vocab_size,
        ngram_scope,
        &sampler_cfg,
        &mut st.rng_state,
    );
}

/// §M7. Everything `qwen35_decode_one` does AFTER its forward: advance the
/// cursor, evict, emit and account for the token the forward just committed,
/// run the stop machinery, and sample the next one.
///
/// Lifted verbatim so a batched driver can reuse it. The batched shape is one
/// forward over N rows, then this per stream in a loop — every line here is
/// per-stream and touches no model weights, so running it sequentially after a
/// shared forward changes no result.
///
/// `logits` is what the forward wrote for THIS stream: `scratch.logits` on the
/// single-stream path, the row's own session logits on the batched one.
#[allow(clippy::too_many_arguments)]
/// Prefill `tokens` into the session, honouring the eviction window and the
/// latency band. Extracted from `generate_start` so the SAME code can run either
/// as one call (today) or as a sequence of bands driven by the executor — the
/// band boundaries are already committed KV states, so a caller may stop between
/// them.
///
/// Returns the number of prompt tokens consumed. That is `tokens.len()` unless a
/// caller-supplied band budget cut it short, which is how a low-priority prefill
/// becomes preemptible.
#[allow(clippy::too_many_arguments)]
/// KVarN prefill was per-token until the 2026-08-23 coherence battery; it is
/// batched by default now and only rolls back on an explicit `=0`.
///
/// The spec-decode gate (`kvarn_specdecode_ok`) is now default-on too. It
/// originally required `=1` on a 2026-08-22 measurement of 5.77 vs 14.4 tok/s
/// plain, but two later measurements contradict that: dflash_spec_demo records
/// near-parity drafter engagement under kvarn (57.39 vs 57.54 tok/s), and
/// Qwen3.8-27B+dflash2 on gfx1103 measured 8.03 vs 5.2 tok/s plain -- a 1.54x
/// WIN (docs/plans/2026-08-27-dflash-blocksize-gfx1103.md). A model that loses
/// with its drafter under kvarn should not carry that drafter; per-model
/// opt-out is `dflash_mode=off` (or simply no draft), not a KV-mode side gate.
///
///   unset -> batched prefill, drafter ON    (default)
///   =1    -> same as unset (back-compat with the old opt-in)
///   =0    -> per-token prefill, drafter OFF (full rollback)
fn kvarn_forced_per_token() -> bool {
    std::env::var("HIPFIRE_KVARN_BATCHED_PREFILL")
        .ok()
        .as_deref()
        == Some("0")
}

fn qwen35_prefill_tokens(
    gpu: &mut hipfire_rdna::Gpu,
    weights: &qwen35::Qwen35Weights,
    config: &qwen35::Qwen35Config,
    scratch: &qwen35::Qwen35Scratch,
    kv: &mut hipfire_runtime::kv::KvCache,
    dn: &mut qwen35::DeltaNetState,
    cursor: &mut crate::session::SessionCursor,
    eviction: Option<&crate::model::Eviction>,
    tokens: &[u32],
    id: &str,
) -> Result<usize, String> {
    if kv.quant_kvarn && kvarn_forced_per_token() {
        // ROLLBACK ONLY. This used to be unconditional, on the reasoning that
        // KVarN required the per-token attention dispatch -- the batched forward
        // ran its own batched attention and never populated the KVarN
        // window/records, so the prompt KV came out wrong. That stopped being
        // true when prefill_chunk.rs took over the batched KVarN write, and the
        // remaining divergence closed with 8ea5a303e (attend each segment BEFORE
        // flushing it). Battery 2026-08-23, 150 greedy tokens, batched vs
        // per-token: BYTE-IDENTICAL on every model where the batched path
        // engages (qwen3.5-2b/4b bf16, Qwen3.5/3.6/3.8-27B oq4.25++), prefill
        // 15-21 -> 308-1573 tok/s.
        //
        // ⚠️ THAT BATTERY MEASURES TOKENS, NOT HIDDEN STATES — do not read it as
        // blanket path equivalence. `compare_prefill_hidden_paths` on the SAME
        // buffer a DFlash drafter consumes (2026-08-26):
        //
        //     model         kv      first diverging layer   worst
        //     27B dense     kvarn   2                       2.28e-2
        //     27B dense     q8      2                       1.32e-2
        //     35B-A3B MoE   kvarn   0                       1.57e-1
        //     35B-A3B MoE   q8      0                       1.36e-1
        //     35B-A3B MoE   fp32    NONE                    0.000e0
        //
        // Both results hold: hidden states differ ~1e-2 without flipping a
        // greedy argmax, so token output stays byte-identical. NOT a regression
        // in 8ea5a303e.
        //
        // But a DFlash drafter consumes the HiddenStateRingBuffer, not the token
        // stream, so this path can pass the battery and still starve a drafter —
        // a candidate mechanism for serving tau 0.05 vs 0.764 in
        // dflash_spec_demo on the same pair. Quantised KV is the trigger (fp32
        // is exactly 0); the MoE expert path amplifies it ~7x and moves first
        // divergence from layer 2 to layer 0. If drafter fidelity under
        // quantised KV matters, that battery needs a hidden-state arm.
        //
        // Only the LAST prompt token's logits are ever read; the rest are
        // discarded. On Qwen3.8-27B the lm_head is 675 MB at oq4.25 (vocab
        // 248320 x 5120) = ~2.9 ms per call, so a logits-producing forward per
        // prompt token spent ~2.9 ms each producing nothing. Measured on a
        // 3000-token prompt: 2122 lm_head dispatches against 7 on the path that
        // already skips it.
        let last = tokens.len().saturating_sub(1);
        for (i, &tok) in tokens.iter().enumerate() {
            if i == last {
                qwen35::forward_scratch(gpu, weights, config, tok, cursor.seq_pos, kv, dn, scratch)
                    .unwrap();
            } else {
                qwen35::forward_scratch_no_logits(
                    gpu,
                    weights,
                    config,
                    tok,
                    cursor.seq_pos,
                    kv,
                    dn,
                    scratch,
                )
                .unwrap();
            }
            cursor.seq_pos += 1;
        }
    } else if eviction.is_some() || prefill_band_tokens().is_some() {
        // Chunked prefill. Two independent reasons to cut, and they
        // compose by taking the smaller cut:
        //
        //   * eviction — chunk to the (budget+beta) window so physical
        //     never exceeds `physical_cap`. A MEMORY bound.
        //   * `HIPFIRE_PREFILL_BAND_TOKENS` — chunk so no single dispatch
        //     runs longer than the operator will tolerate. A LATENCY
        //     bound, and the one that makes a low-priority prefill
        //     preemptible: each band ends at a committed KV boundary with
        //     `seq_pos` advanced, which is exactly the state a second
        //     conversation turn resumes from. Nothing extra is pinned.
        //
        // Previously this loop existed only when eviction was configured,
        // so an operator who wanted a latency cap had to enable an unrelated
        // memory feature to get one.
        let band = prefill_band_tokens();
        let window = eviction.as_ref().map(|ev| ev.budget() + ev.beta());
        let mut remaining: &[u32] = tokens;
        while !remaining.is_empty() {
            let space = window
                .map(|w| w.saturating_sub(cursor.seq_pos).max(1))
                .unwrap_or(usize::MAX);
            let chunk_len = remaining.len().min(space).min(band.unwrap_or(usize::MAX));
            let (chunk, rest) = remaining.split_at(chunk_len);
            if prefill_path_trace() {
                eprintln!(
                    "[prefill-path] id={id} band chunk_len={chunk_len} at seq_pos={} remaining={}",
                    cursor.seq_pos,
                    remaining.len(),
                );
            }
            qwen35::forward_prefill_batch(
                gpu,
                weights,
                config,
                chunk,
                cursor.seq_pos,
                kv,
                dn,
                scratch,
                None,
                None,
                None,
                None,
            )
            .unwrap();
            cursor.seq_pos += chunk_len;
            if let Some(ref ev) = eviction {
                if let Some(hipfire_runtime::triattn::EvictionResult {
                    new_physical: new_phys,
                    ..
                }) = ev.maybe_evict(gpu, kv, cursor.seq_pos).unwrap()
                {
                    cursor.seq_pos = new_phys;
                }
            }
            remaining = rest;
        }
    } else {
        // A prefill failure here used to `.unwrap()`, which killed the
        // whole daemon process — the client saw the socket close with no
        // explanation. That is not hypothetical: the paged-MoE capability
        // refusals (`unsupported moe.decode-*`) surface exactly here, so a
        // configuration the runtime deliberately declines took down every
        // other session on the worker. Report it and unwind like the other
        // fallible steps in this function do.
        if let Err(e) = qwen35::forward_prefill_batch(
            gpu,
            weights,
            config,
            tokens,
            cursor.seq_pos,
            kv,
            dn,
            scratch,
            None,
            None,
            None,
            None,
        ) {
            // The caller reports and unwinds; this function does not own the
            // session, so it cannot restore it.
            return Err(format!("prefill failed: {e}"));
        }
        cursor.seq_pos += tokens.len();
    }
    Ok(tokens.len())
}

fn qwen35_decode_after_forward(
    gpu: &mut hipfire_rdna::Gpu,
    weights: &qwen35::Qwen35Weights,
    config: &qwen35::Qwen35Config,
    scratch: &qwen35::Qwen35Scratch,
    logits: &hipfire_rdna::GpuTensor,
    // Whether the output filter asked to stop, decided BEFORE the forward from
    // the bytes the previous token produced. It crosses the seam because the
    // filter observes a token and the stop it implies is acted on only after
    // that token's K/V is committed.
    filter_stop: bool,
    kv: &mut hipfire_runtime::kv::KvCache,
    dn: &mut qwen35::DeltaNetState,
    cursor: &mut crate::session::SessionCursor,
    eviction: Option<&crate::model::Eviction>,
    physical_cap: usize,
    tokenizer: &hipfire_model::tokenizer::Tokenizer,
    stdout: &mut dyn std::io::Write,
    id: &str,
    t0: Instant,
    cfg: &Qwen35DecodeCfg,
    st: &mut Qwen35DecodeState,
) -> Qwen35Step {
    cursor.seq_pos += 1;
    if let Some(ev) = eviction {
        if let Some(hipfire_runtime::triattn::EvictionResult {
            new_physical: new_phys,
            ..
        }) = ev.maybe_evict(gpu, kv, cursor.seq_pos).unwrap()
        {
            cursor.seq_pos = new_phys;
        }
    }
    if filter_stop {
        return Qwen35Step::Stop;
    }

    if st.next_token == config.eos_token {
        return Qwen35Step::Stop;
    }
    if cfg.im_end_token == Some(st.next_token) {
        return Qwen35Step::Stop;
    }
    if tokenizer.is_terminator(st.next_token) {
        return Qwen35Step::Stop;
    }

    // max_think_tokens enforcement. Track whether we're inside an
    // open <think>...</think> block and how many tokens we've
    // emitted there. When the cap is hit, splice "</think>\n" into
    // the stream (KV write + stdout emit + advance generated) so
    // the model commits to an answer with the remaining budget.
    // Same decoded-text scan budget_alert uses; counter is
    // incremented per-iteration only when we're still inside.
    if cfg.max_think_tokens > 0 {
        let raw_so_far = tokenizer.decode_bytes(&st.streamed_tokens);
        let raw_str = std::str::from_utf8(&raw_so_far).unwrap_or("");
        let open_idx = raw_str.rfind("<think>");
        let close_idx = raw_str.rfind("</think>");
        let in_think = match (open_idx, close_idx) {
            (Some(o), Some(c)) => o > c,
            (Some(_), None) => true,
            _ => false,
        };
        if in_think {
            if !st.prev_in_think {
                st.think_count = 1;
            } else {
                st.think_count += 1;
            }
        } else {
            st.think_count = 0;
        }
        st.prev_in_think = in_think;

        if in_think && st.think_count >= cfg.max_think_tokens {
            // Force-close. Encode the close sequence and run each
            // token through the KV write + emit path the same way
            // a normally-sampled token does. This ensures the
            // model's next sample is conditioned on having "said"
            // </think>\n itself, instead of seeing a hidden-state
            // discontinuity. Respect max_tokens — clip the close
            // sequence if not enough room remains and bail.
            let close_tokens = tokenizer.encode("</think>\n");
            let budget_left = cfg.max_tokens.saturating_sub(st.generated);
            let take = close_tokens.len().min(budget_left);
            for &t in &close_tokens[..take] {
                qwen35::forward_scratch(gpu, weights, config, t, cursor.seq_pos, kv, dn, scratch)
                    .unwrap();
                cursor.seq_pos += 1;
                if let Some(ev) = eviction {
                    if let Some(hipfire_runtime::triattn::EvictionResult {
                        new_physical: new_phys,
                        ..
                    }) = ev.maybe_evict(gpu, kv, cursor.seq_pos).unwrap()
                    {
                        cursor.seq_pos = new_phys;
                    }
                }
                cursor.conversation_tokens.push(t);
                st.streamed_tokens.push(t);
                emit_committed_event(
                    stdout,
                    id,
                    t,
                    st.streamed_tokens.len() - 1,
                    t0.elapsed().as_millis() as u64,
                );
                let all_bytes = tokenizer.decode_bytes(&st.streamed_tokens);
                let new_bytes = &all_bytes[st.bytes_fed_to_filter..];
                st.bytes_fed_to_filter = all_bytes.len();
                let _ = emit_filter_action(stdout, id, st.filter.observe(new_bytes));
                st.generated += 1;
            }
            st.think_count = 0;
            st.prev_in_think = false;
            if st.generated >= cfg.max_tokens {
                return Qwen35Step::Stop;
            }
        }
    }

    // N-gram loop detector: check if any 4-gram in the recent window
    // repeats excessively. When detected, emit an info message and
    // force EOS to prevent wasting the remaining token budget on
    // repetitive output. Logic lives in `hipfire-generate` loop_guard.
    if let Some(StopReason::NgramRepeat { count, .. }) = cfg.loop_guard.check(&st.streamed_tokens) {
        let window_len = cfg.loop_guard.window_len(st.streamed_tokens.len());
        let _ = writeln!(
            stdout,
            r#"{{"type":"info","id":"{}","message":"ngram loop detected (4gram repeated {}× in last {} tokens) — forcing EOS"}}"#,
            id, count, window_len
        );
        let _ = stdout.flush();
        return Qwen35Step::Stop;
    }

    // Budget-alert injection: once we hit the configured token count,
    // splice the nudge text into the stream. Tokens are emitted to
    // stdout (so the client sees them) AND forward-fed through the KV
    // cache (so the model's next sample is conditioned on having
    // "said" them itself). Injected tokens count against `max_tokens`
    // — we never exceed the caller's requested budget — so we clip
    // the nudge if not enough room remains, and stop if the budget is
    // fully spent after injection.
    if !st.alert_fired
        && cfg.budget_alert_at_tok > 0
        && st.generated >= cfg.budget_alert_at_tok
        && !cfg.budget_alert_text.is_empty()
    {
        st.alert_fired = true;
        // Only inject while the model is inside an open <think> block.
        // The whole point of the feature is to nudge the model's
        // reasoning; firing past </think> just graffities the visible
        // answer with a system-alert string. Check the raw decoded
        // text rather than token IDs since <think> tokenizes as a
        // multi-token sequence in Qwen3.5's vocab.
        let raw_so_far = tokenizer.decode_bytes(&st.streamed_tokens);
        let raw_str = std::str::from_utf8(&raw_so_far).unwrap_or("");
        let think_open_idx = raw_str.rfind("<think>");
        let think_close_idx = raw_str.rfind("</think>");
        let in_think = match (think_open_idx, think_close_idx) {
            (Some(o), Some(c)) => o > c,
            (Some(_), None) => true,
            _ => false,
        };
        if !in_think {
            let _ = writeln!(
                stdout,
                r#"{{"type":"info","id":"{}","message":"budget_alert skipped: not inside an open <think> block"}}"#,
                id
            );
            let _ = stdout.flush();
            // Fall through — resample next token as normal
            let ngram_scope = &cursor.conversation_tokens[cfg.ngram_scope_start..];
            let mut blocked: Vec<u32> = Vec::new();
            collect_unclosed_attractor_blocks(
                ngram_scope,
                &cfg.attractor_pairs,
                20,
                2,
                &mut blocked,
            );
            let sampler_cfg = SamplerConfig {
                temperature: cfg.temperature,
                top_p: cfg.top_p,
                top_k: cfg.top_k,
                repeat_penalty: cfg.repeat_penalty,
                repeat_window: cfg.repeat_buf_cap,
                presence_penalty: cfg.presence_penalty,
                frequency_penalty: cfg.frequency_penalty,
                blocked_tokens: blocked,
            };
            st.next_token = sampler::sample(
                gpu,
                &scratch.logits,
                &scratch.sample_buf,
                &scratch.repeat_buf,
                cfg.vocab_size,
                ngram_scope,
                &sampler_cfg,
                &mut st.rng_state,
            );
            return Qwen35Step::Continue;
        }
        let nudge_tokens = tokenizer.encode(&cfg.budget_alert_text);
        let budget_left = cfg.max_tokens.saturating_sub(st.generated);
        let nudge_len = nudge_tokens.len().min(budget_left);
        // KV headroom check — don't run past physical_cap. If we don't
        // have room for the clipped nudge, skip entirely rather than
        // emit a partial nudge that poisons the trajectory. Under
        // eviction the physical check is trivially satisfied (budget
        // always holds post-evict), but we still respect the check for
        // the non-eviction path.
        let need_kv =
            cursor.seq_pos + nudge_len + (cfg.max_tokens - st.generated - nudge_len) + cfg.nl_len;
        if nudge_len > 0 && (eviction.is_some() || need_kv <= physical_cap) {
            for &tok in &nudge_tokens[..nudge_len] {
                cursor.conversation_tokens.push(tok);
                st.streamed_tokens.push(tok);
                emit_committed_event(
                    stdout,
                    id,
                    tok,
                    st.streamed_tokens.len() - 1,
                    t0.elapsed().as_millis() as u64,
                );
                // Emit the injected token's text to stdout so the client
                // sees it as part of the stream (will be inside <think>
                // if that's the current state, and get stripped client-
                // side just like any other think token).
                let all_bytes2 = tokenizer.decode_bytes(&st.streamed_tokens);
                let new_bytes2 = &all_bytes2[st.bytes_fed_to_filter..];
                st.bytes_fed_to_filter = all_bytes2.len();
                let _ = emit_filter_action(stdout, id, st.filter.observe(new_bytes2));
                if let Err(e) = qwen35::forward_scratch(
                    gpu,
                    weights,
                    config,
                    tok,
                    cursor.seq_pos,
                    kv,
                    dn,
                    scratch,
                ) {
                    return Qwen35Step::Failed(format!(
                        "qwen35 budget-alert forward_scratch failed: {e:?}"
                    ));
                }
                cursor.seq_pos += 1;
                if let Some(ev) = eviction {
                    if let Some(hipfire_runtime::triattn::EvictionResult {
                        new_physical: new_phys,
                        ..
                    }) = ev.maybe_evict(gpu, kv, cursor.seq_pos).unwrap()
                    {
                        cursor.seq_pos = new_phys;
                    }
                }
                st.generated += 1;
            }
        } else if nudge_len < nudge_tokens.len() {
            let _ = writeln!(
                stdout,
                r#"{{"type":"info","id":"{}","message":"budget_alert clipped or skipped: nudge_len={} budget_left={}"}}"#,
                id, nudge_len, budget_left
            );
            let _ = stdout.flush();
        } else {
            let _ = writeln!(
                stdout,
                r#"{{"type":"info","id":"{}","message":"budget_alert skipped: not enough KV headroom"}}"#,
                id
            );
            let _ = stdout.flush();
        }
        // Respect max_tokens: if injection used the remainder, bail
        // before sampling another model token.
        if st.generated >= cfg.max_tokens {
            return Qwen35Step::Stop;
        }
    }

    qwen35_sample_next_from(gpu, logits, scratch, cursor, cfg, st);
    Qwen35Step::Continue
}

/// §M7. Everything `qwen35_decode_one` does BEFORE its forward: cancellation,
/// the stop machinery for the token already sampled, emission, and the output
/// filter — ending with the `filter_stop` the post-forward half consumes.
///
/// The third piece of the decode step. With this, `qwen35_decode_one` is
/// literally pre -> forward -> post, and a batched driver runs pre per stream,
/// ONE forward over the survivors' tokens, then post per stream.
///
/// Returns `Stop` when the step ends before committing anything — the caller
/// must not forward that stream this round.
enum PreForward {
    Stop,
    Ready { filter_stop: bool },
}

#[allow(clippy::too_many_arguments)]
fn qwen35_decode_before_forward(
    tokenizer: &hipfire_model::tokenizer::Tokenizer,
    stdout: &mut dyn std::io::Write,
    id: &str,
    t0: Instant,
    st: &mut Qwen35DecodeState,
    cursor: &mut crate::session::SessionCursor,
) -> PreForward {
    // Cooperative cancellation (SIGUSR1 → GENERATION_CANCEL). KV-safe
    // chokepoint: on entry every prior token's K/V has been written
    // via forward_scratch and seq_pos advanced; the pending `next_token`
    // is sampled but not yet written. Stopping here drops only that
    // unwritten sample, so the cache/session stay consistent — identical
    // to a natural `max_tokens` stop — and the done frame below is
    // emitted normally.
    if hipfire_runtime::take_generation_cancel() {
        return PreForward::Stop;
    }
    st.generated += 1;
    cursor.conversation_tokens.push(st.next_token);
    st.streamed_tokens.push(st.next_token);
    emit_committed_event(
        stdout,
        id,
        st.next_token,
        st.streamed_tokens.len() - 1,
        t0.elapsed().as_millis() as u64,
    );
    // Incremental UTF-8 + filter routing: feed only the new
    // bytes since last call, let the filter buffer any partial
    // codepoint or marker prefix until disambiguated.
    let all_bytes = tokenizer.decode_bytes(&st.streamed_tokens);
    let new_bytes = &all_bytes[st.bytes_fed_to_filter..];
    st.bytes_fed_to_filter = all_bytes.len();
    let filter_stop = emit_filter_action(stdout, id, st.filter.observe(new_bytes));
    PreForward::Ready { filter_stop }
}

fn qwen35_decode_one(
    gpu: &mut hipfire_rdna::Gpu,
    weights: &qwen35::Qwen35Weights,
    config: &qwen35::Qwen35Config,
    scratch: &qwen35::Qwen35Scratch,
    kv: &mut hipfire_runtime::kv::KvCache,
    dn: &mut qwen35::DeltaNetState,
    cursor: &mut crate::session::SessionCursor,
    eviction: Option<&crate::model::Eviction>,
    physical_cap: usize,
    tokenizer: &hipfire_model::tokenizer::Tokenizer,
    stdout: &mut dyn std::io::Write,
    id: &str,
    t0: Instant,
    cfg: &Qwen35DecodeCfg,
    st: &mut Qwen35DecodeState,
) -> Qwen35Step {
    let filter_stop = match qwen35_decode_before_forward(tokenizer, stdout, id, t0, st, cursor) {
        PreForward::Stop => return Qwen35Step::Stop,
        PreForward::Ready { filter_stop } => filter_stop,
    };

    // Write this token's K/V to the cache FIRST so the next turn
    // always starts from a fully-written context. Stopping before
    // forward_scratch used to leave a hole at the im_end/eos
    // position — the next turn then attended over zero-init K/V
    // at that slot.
    //
    // Under eviction, cursor.seq_pos is the *physical* write slot; we
    // advance and call maybe_evict immediately so the next write
    // never overruns physical_cap. compact_offset bookkeeping on
    // the cache itself keeps RoPE phase correct across evictions.
    if let Err(e) = qwen35::forward_scratch(
        gpu,
        weights,
        config,
        st.next_token,
        cursor.seq_pos,
        kv,
        dn,
        scratch,
    ) {
        return Qwen35Step::Failed(format!("qwen35 decode forward_scratch failed: {e:?}"));
    }
    qwen35_decode_after_forward(
        gpu,
        weights,
        config,
        scratch,
        &scratch.logits,
        filter_stop,
        kv,
        dn,
        cursor,
        eviction,
        physical_cap,
        tokenizer,
        stdout,
        id,
        t0,
        cfg,
        st,
    )
}

/// Helper: render the JSON field fragment for `done` per PRD §3.1.
/// Three states:
///   - compressed: full metadata + alpha
///   - bypass (non-Off, drafter loaded): alpha + bypass_reason
///   - nothing: empty string so backwards-compatible clients see the
///     original done shape
/// Formats the `"pflash"` fragment of the `done` frame. Lifted out of
/// `generate()` so `qwen35_finish_generation` can reach it; it is a plain
/// capture-free `fn`, so moving it changes nothing.
fn pflash_done_fragment(
    s: &Option<hipfire_arch_qwen35::pflash::CompressedPrompt>,
    bypass_reason: &Option<String>,
    alpha: Option<f32>,
) -> String {
    match (s, bypass_reason) {
        (Some(cp), _) => format!(
            r#","pflash":{{"source_tokens":{},"kept_tokens":{},"keep_ratio":{:.6},"alpha":{:.6},"score_ms":{},"total_ms":{},"source_md5":"{}","compressed_md5":"{}"}}"#,
            cp.source_tokens,
            cp.kept_tokens,
            cp.kept_tokens as f32 / cp.source_tokens.max(1) as f32,
            alpha.unwrap_or(0.0),
            cp.timings.score_ms,
            cp.timings.total_ms,
            cp.source_md5,
            cp.compressed_md5,
        ),
        (None, Some(reason)) => format!(
            r#","pflash":{{"bypass_reason":"{}","alpha":{:.6}}}"#,
            reason.replace('"', "'"),
            alpha.unwrap_or(0.0),
        ),
        (None, None) => String::new(),
    }
}

#[allow(clippy::too_many_arguments)]
/// Finish a qwen35 generation: the ChatML `\n` trailer, the timing arithmetic,
/// the evidence writes, the `done` frame, and the session restore.
///
/// Executor v2 §M3b0.5 — the `finish()` half. Moved verbatim out of
/// [`generate`]'s qwen35 arm, so behaviour is byte-identical by construction.
///
/// **Takes `session` BY VALUE, and that is the point.** `qwen35_restore_or_error`
/// consumes the session while `kv`/`dn` are disjoint `&mut` borrows *out of* it.
/// That is exactly why M3b0's [`Qwen35Step::Failed`] has to hand a message back
/// to its caller instead of unwinding in place. Owning the session here dissolves
/// the constraint: `kv`/`dn` are derived internally and the restore is a direct
/// call, so the whole teardown — including its own error path — lives in one
/// place.
fn qwen35_finish_generation(
    m: &mut LoadedModel,
    gpu: &mut hipfire_rdna::Gpu,
    stdout: &mut dyn std::io::Write,
    id: &str,
    mut session: Qwen35RequestSessionState,
    generated: usize,
    nl: &[u32],
    im_end_token: Option<u32>,
    t0: Instant,
    t_prefill: Instant,
    prefill_tokens: usize,
    evidence_dir: Option<&str>,
    moe_router_histogram: DaemonMoeRouterHistogramGuard,
    pflash_summary: Option<hipfire_arch_qwen35::pflash::CompressedPrompt>,
    pflash_bypass_reason: Option<String>,
    pflash_alpha: Option<f32>,
) {
    let config = m.q35_config.as_ref().unwrap();
    let weights = m.q35_weights.as_ref().unwrap();
    let scratch = m.q35_scratch.as_ref().unwrap();
    // Same disjoint field-path borrow the decode phase uses: kv and dn are
    // distinct fields of session.sequence_state, so both &mut live at once.
    let kv = session
        .sequence_state
        .kv
        .as_mut()
        .expect("qwen35 session always has KV");
    let dn = session
        .sequence_state
        .recurrent
        .as_mut()
        .expect("qwen35 session has DeltaNet state")
        .as_any_mut()
        .downcast_mut::<qwen35::DeltaNetState>()
        .expect("qwen35 session recurrent state is DeltaNetState");
    // session.cursor.seq_pos is already the "next physical write slot" — advanced
    // per-token in the decode loop above, and evicted back down to
    // `budget` whenever maybe_evict fired. No post-loop fix-up needed.

    // ChatML requires \n after <|im_end|>. Run it through forward so KV cache
    // and DeltaNet state stay in sync with seq_pos.
    if im_end_token == Some(*session.cursor.conversation_tokens.last().unwrap_or(&0))
        && !nl.is_empty()
    {
        for &t in nl {
            if let Err(e) = qwen35::forward_scratch(
                gpu,
                weights,
                config,
                t,
                session.cursor.seq_pos,
                kv,
                dn,
                scratch,
            ) {
                write_error(
                    stdout,
                    id,
                    &format!("qwen35 ChatML newline forward_scratch failed: {e:?}"),
                );
                qwen35_restore_or_error(stdout, id, m, gpu, session);
                return;
            }
            session.cursor.seq_pos += 1;
            if let Some(ref ev) = m.eviction {
                if let Some(hipfire_runtime::triattn::EvictionResult {
                    new_physical: new_phys,
                    ..
                }) = ev.maybe_evict(gpu, kv, session.cursor.seq_pos).unwrap()
                {
                    session.cursor.seq_pos = new_phys;
                }
            }
            session.cursor.conversation_tokens.push(t);
        }
    }

    let t_end = Instant::now();
    let total_s = t_end.duration_since(t0).as_secs_f64();
    let prefill_s = t_prefill.duration_since(t0).as_secs_f64();
    let decode_s = t_end.duration_since(t_prefill).as_secs_f64();
    let tok_s = if total_s > 0.0 {
        generated as f64 / total_s
    } else {
        0.0
    };
    let prefill_tok_s = if prefill_s > 0.0 {
        prefill_tokens as f64 / prefill_s
    } else {
        0.0
    };
    let decode_tok_s = if decode_s > 0.0 {
        generated as f64 / decode_s
    } else {
        0.0
    };
    if let Some(dir) = evidence_dir {
        write_daemon_runtime_oneshot_evidence(
            dir,
            m,
            gpu,
            id,
            prefill_tokens,
            generated,
            prefill_s,
            decode_s,
            prefill_s * 1000.0,
        );
        if let Some(hist) = moe_router_histogram.take() {
            write_daemon_moe_router_evidence(dir, m, id, hist);
        }
    }
    let _ = writeln!(
        stdout,
        r#"{{"type":"done","id":"{}","tokens":{},"tok_s":{:.1},"prefill_tokens":{},"prefill_ms":{:.1},"prefill_tok_s":{:.1},"decode_tok_s":{:.1},"ttft_ms":{:.1}{}}}"#,
        id,
        generated,
        tok_s,
        prefill_tokens,
        prefill_s * 1000.0,
        prefill_tok_s,
        decode_tok_s,
        prefill_s * 1000.0,
        pflash_done_fragment(&pflash_summary, &pflash_bypass_reason, pflash_alpha),
    );
    let _ = stdout.flush();
    qwen35_restore_or_error(stdout, id, m, gpu, session);
}

/// A qwen35 generation in flight — everything a suspended stream must carry
/// between quanta, and nothing it must not.
///
/// Executor v2 §M3b0.5. [`Qwen35Generation::step`] advances one token,
/// [`Qwen35Generation::finish`] runs the teardown.
///
/// **Owns no borrow of the GPU or the model.** Those arrive per call, because a
/// serial executor hands the device to another stream between quanta; a handle
/// holding `&mut Gpu` would pin the device to one stream for its whole life and
/// defeat the interleaving it exists to enable. That property is why
/// `Qwen35DecodeState` was built borrow-free in M3b0 and why `Qwen35DecodeCfg`
/// had its lifetime removed — this type is what both were for.
pub struct Qwen35Generation {
    /// `Some` while this stream owns the session; `None` while it is parked
    /// back in the model's resident slot. Only one place holds it at a time.
    session: Option<Qwen35RequestSessionState>,
    /// Prompt tokens not yet prefilled. NON-EMPTY means this handle is still in
    /// its prefill phase and `st.next_token` is NOT yet meaningful — nothing may
    /// read it until the last band drains and `tok0` is sampled.
    ///
    /// Prefill used to run to completion inside `generate_start`, which made it
    /// one indivisible unit ahead of the executor and put a whole prompt's
    /// latency in front of any higher-priority stream. Carrying the remainder
    /// here lets `step` advance it one band at a time, so the march's existing
    /// priority ordering interleaves it — no suspend/resume machinery needed,
    /// because a band boundary is already a committed KV state.
    prefill_pending: Vec<u32>,
    st: Qwen35DecodeState,
    cfg: Qwen35DecodeCfg,
    /// The ChatML `\n` tokens; `finish` forwards them when the turn ended on
    /// `<|im_end|>`. `cfg.nl_len` is the same sequence's length, read by the
    /// budget-alert headroom check.
    nl: Vec<u32>,
    t0: Instant,
    t_prefill: Instant,
    prefill_tokens: usize,
    evidence_dir: Option<String>,
    moe_router_histogram: DaemonMoeRouterHistogramGuard,
    pflash_summary: Option<hipfire_arch_qwen35::pflash::CompressedPrompt>,
    pflash_bypass_reason: Option<String>,
    pflash_alpha: Option<f32>,
}

impl Qwen35Generation {
    /// Advance one token. The model and GPU are borrowed for the call only.
    ///
    /// `weights`/`config`/`scratch` and `eviction`/`physical_cap` are derived
    /// per step rather than captured once, which is what §M3b1 asks for: the
    /// moment the march loop can hand `&mut LoadedModel` to another stream
    /// between quanta, a captured snapshot and a per-use read stop agreeing.
    /// Advance the prefill by one band. Returns `Continue` until the prompt is
    /// drained, then samples `tok0` and hands the handle over to decoding.
    ///
    /// The band size is `HIPFIRE_PREFILL_BAND_TOKENS`; unset means the whole
    /// remaining prompt in one call, which is the pre-existing behaviour moved
    /// under the march rather than a new cost.
    fn prefill_band(
        &mut self,
        m: &LoadedModel,
        gpu: &mut hipfire_rdna::Gpu,
        id: &str,
    ) -> Qwen35Step {
        let config = m.q35_config.as_ref().unwrap();
        let weights = m.q35_weights.as_ref().unwrap();
        let scratch = m.q35_scratch.as_ref().unwrap();
        let band = prefill_band_tokens().unwrap_or(usize::MAX);
        let take = self.prefill_pending.len().min(band);
        let chunk: Vec<u32> = self.prefill_pending[..take].to_vec();
        let session = self
            .session
            .as_mut()
            .expect("stream must be resumed before it is stepped");
        let kv = session
            .sequence_state
            .kv
            .as_mut()
            .expect("qwen35 session always has KV");
        let dn = session
            .sequence_state
            .recurrent
            .as_mut()
            .expect("qwen35 session has DeltaNet state")
            .as_any_mut()
            .downcast_mut::<qwen35::DeltaNetState>()
            .expect("qwen35 session recurrent state is DeltaNetState");
        if let Err(e) = qwen35_prefill_tokens(
            gpu,
            weights,
            config,
            scratch,
            kv,
            dn,
            &mut session.cursor,
            m.eviction.as_ref(),
            &chunk,
            id,
        ) {
            return Qwen35Step::Failed(e);
        }
        // Record the band's tokens as they commit, not the whole prompt at the
        // end. `cfg.ngram_scope_start` was set to index PAST the prompt, so by
        // the time the last band drains, `conversation_tokens.len()` has reached
        // exactly that index and the first sample sees an empty scope — the same
        // scope the inline path gave it.
        session.cursor.conversation_tokens.extend_from_slice(&chunk);
        self.prefill_pending.drain(..take);
        if !self.prefill_pending.is_empty() {
            return Qwen35Step::Continue;
        }
        qwen35_sample_next_from(
            gpu,
            &scratch.logits,
            scratch,
            &mut session.cursor,
            &self.cfg,
            &mut self.st,
        );
        self.t_prefill = Instant::now();
        Qwen35Step::Continue
    }

    pub fn step(
        &mut self,
        m: &LoadedModel,
        gpu: &mut hipfire_rdna::Gpu,
        tokenizer: &hipfire_model::tokenizer::Tokenizer,
        stdout: &mut dyn std::io::Write,
        id: &str,
    ) -> Qwen35Step {
        let config = m.q35_config.as_ref().unwrap();
        let weights = m.q35_weights.as_ref().unwrap();
        let scratch = m.q35_scratch.as_ref().unwrap();
        // Prefill phase: advance ONE band and return. The march gives the next
        // quantum to whichever stream its priority order picks, so a
        // higher-priority arrival overtakes here instead of waiting out the whole
        // prompt. The band boundary is a committed KV state — `seq_pos` advanced,
        // K/V and recurrent written — which is what a continued conversation turn
        // resumes from, so stopping between bands preserves nothing extra and
        // loses nothing.
        if !self.prefill_pending.is_empty() {
            return self.prefill_band(m, gpu, id);
        }
        // Disjoint field paths: kv and recurrent are distinct fields of
        // session.sequence_state, and cursor is a third field of session.
        let session = self
            .session
            .as_mut()
            .expect("stream must be resumed before it is stepped");
        let kv = session
            .sequence_state
            .kv
            .as_mut()
            .expect("qwen35 session always has KV");
        let dn = session
            .sequence_state
            .recurrent
            .as_mut()
            .expect("qwen35 session has DeltaNet state")
            .as_any_mut()
            .downcast_mut::<qwen35::DeltaNetState>()
            .expect("qwen35 session recurrent state is DeltaNetState");
        qwen35_decode_one(
            gpu,
            weights,
            config,
            scratch,
            kv,
            dn,
            &mut session.cursor,
            m.eviction.as_ref(),
            m.physical_cap,
            tokenizer,
            stdout,
            id,
            self.t0,
            &self.cfg,
            &mut self.st,
        )
    }

    /// Take this stream's session straight out of the registry, for batched
    /// stepping.
    ///
    /// `resume` cannot serve a batch: it routes through the single resident
    /// slot via `qwen35_activate_session`, and that slot holds exactly one
    /// session — the second stream to resume without an intervening park dies
    /// with "qwen35 session missing decode state". A batched round needs all N
    /// sessions held at once, so it bypasses the slot entirely, exactly as the
    /// batched prefill and decode paths already do.
    ///
    /// Call `qwen35_save_active_session` first so whatever occupies the slot is
    /// back in the registry and therefore findable here.
    pub fn acquire_from_registry(
        &mut self,
        m: &mut LoadedModel,
        session_id: &str,
    ) -> Result<(), String> {
        if self.session.is_some() {
            return Ok(());
        }
        self.session = Some(
            m.q35_registry
                .sessions
                .remove(session_id)
                .ok_or_else(|| format!("session {session_id} is not resident in the registry"))?,
        );
        Ok(())
    }

    /// Put the session back where `acquire_from_registry` found it.
    ///
    /// The counterpart to that call and the reason a batched round leaves no
    /// trace: the slot is untouched throughout, so a later `resume` on the
    /// single-stream path still finds what it expects.
    pub fn release_to_registry(&mut self, m: &mut LoadedModel, session_id: &str) {
        if let Some(session) = self.session.take() {
            m.q35_registry
                .sessions
                .insert(session_id.to_string(), session);
        }
    }

    /// §M7. Step N streams through ONE batched forward.
    ///
    /// The decode step is `pre -> forward -> post`, and only the forward touches
    /// model weights. So this runs `before_forward` for every stream, issues a
    /// single `forward_prefill_grouped_moe_session_batch` over the survivors'
    /// pending tokens, then runs `after_forward` for each — every handle
    /// sampling from its OWN session's logits with its own RNG, penalties and
    /// stop state. That is what keeps the existing generation state machine
    /// authoritative instead of reimplementing it against a shared envelope.
    ///
    /// Each handle must already hold its session; how N handles come to hold
    /// them simultaneously is the caller's problem (the single resident slot's
    /// park/resume dance cannot do it, and the batched decode path takes them
    /// from `q35_registry.sessions` instead).
    ///
    /// A stream whose `before_forward` returns `Stop` is excluded from the
    /// round — it must not be forwarded. The same exclusion covers a stream
    /// about to fire the budget-alert nudge, which needs extra single-token
    /// forwards the others do not.
    ///
    /// Returns one outcome per entry, in order.
    pub fn step_batch(
        entries: &mut [(&str, &mut Qwen35Generation)],
        m: &LoadedModel,
        gpu: &mut hipfire_rdna::Gpu,
        tokenizer: &hipfire_model::tokenizer::Tokenizer,
        stdout: &mut dyn std::io::Write,
    ) -> Result<Vec<Qwen35Step>, String> {
        let config = m
            .q35_config
            .as_ref()
            .ok_or_else(|| "qwen35 config missing".to_string())?;
        let weights = m
            .q35_weights
            .as_ref()
            .ok_or_else(|| "qwen35 weights missing".to_string())?;
        let scratch = m
            .q35_scratch
            .as_ref()
            .ok_or_else(|| "qwen35 scratch missing".to_string())?;
        let pbs = scratch
            .prefill_batch
            .as_ref()
            .ok_or_else(|| "qwen35 batched decode needs prefill-batch scratch".to_string())?;

        // Ids copied up front: the loops below hold one long-lived mutable
        // borrow of `entries`, so it cannot also be indexed for the id.
        let ids: Vec<String> = entries.iter().map(|(id, _)| (*id).to_string()).collect();
        let n = entries.len();
        let mut outcome: Vec<Option<Qwen35Step>> = (0..n).map(|_| None).collect();
        let mut filter_stop: Vec<bool> = vec![false; n];

        // ── pre: per stream, no weights touched ──
        for (i, (id, g)) in entries.iter_mut().enumerate() {
            let t0 = g.t0;
            // Disjoint field paths: `st` and `session` are separate fields.
            let st = &mut g.st;
            let cursor = &mut g
                .session
                .as_mut()
                .ok_or_else(|| "step_batch: stream does not hold its session".to_string())?
                .cursor;
            match qwen35_decode_before_forward(tokenizer, stdout, id, t0, st, cursor) {
                PreForward::Stop => outcome[i] = Some(Qwen35Step::Stop),
                PreForward::Ready { filter_stop: fs } => filter_stop[i] = fs,
            }
        }

        // One mutable borrow of the survivors, reused by the forward and the
        // post pass. Indexing `entries` per row cannot prove disjointness.
        let mut live: Vec<(usize, &mut Qwen35Generation)> = entries
            .iter_mut()
            .enumerate()
            .filter(|(i, _)| outcome[*i].is_none())
            .map(|(i, (_, g))| (i, &mut **g))
            .collect();

        // ── forward: ONE dispatch over the survivors ──
        if live.len() >= 2 {
            let tokens: Vec<[u32; 1]> = live.iter().map(|(_, g)| [g.st.next_token]).collect();
            let mut rows: Vec<qwen35::DensePrefillSessionBatchRow<'_>> =
                Vec::with_capacity(live.len());
            for (slot, (_, g)) in live.iter_mut().enumerate() {
                let session = g.session.as_mut().expect("checked in the pre pass");
                rows.push(qwen35::DensePrefillSessionBatchRow {
                    tokens: &tokens[slot],
                    start_pos: session.cursor.seq_pos,
                    kv_cache: session
                        .sequence_state
                        .kv
                        .as_mut()
                        .expect("qwen35 session always has KV"),
                    dn_state: session
                        .sequence_state
                        .recurrent
                        .as_mut()
                        .expect("qwen35 session has DeltaNet state")
                        .as_any_mut()
                        .downcast_mut::<qwen35::DeltaNetState>()
                        .expect("qwen35 session recurrent state is DeltaNetState"),
                    logits: &session.logits,
                });
            }
            qwen35::forward_prefill_grouped_moe_session_batch(
                gpu, weights, config, &mut rows, scratch, pbs,
            )
            .map_err(|e| format!("qwen35 batched decode forward failed: {e:?}"))?;
            // Probe, not decoration: "the batched run produced identical output"
            // is also what a silent fallback to round-robin produces. Only a
            // positive count from INSIDE the fused arm distinguishes them.
            if std::env::var("HIPFIRE_BATCH_PROBE").is_ok() {
                eprintln!("[batch-probe] fused decode rows={}", rows.len());
            }
            drop(rows);

            // ── post: per stream, each sampling from its OWN logits ──
            for (i, g) in live.iter_mut() {
                let t0 = g.t0;
                let cfg = &g.cfg;
                let st = &mut g.st;
                let session = g.session.as_mut().expect("checked in the pre pass");
                let kv = session
                    .sequence_state
                    .kv
                    .as_mut()
                    .expect("qwen35 session always has KV");
                let dn = session
                    .sequence_state
                    .recurrent
                    .as_mut()
                    .expect("qwen35 session has DeltaNet state")
                    .as_any_mut()
                    .downcast_mut::<qwen35::DeltaNetState>()
                    .expect("qwen35 session recurrent state is DeltaNetState");
                outcome[*i] = Some(qwen35_decode_after_forward(
                    gpu,
                    weights,
                    config,
                    scratch,
                    &session.logits,
                    filter_stop[*i],
                    kv,
                    dn,
                    &mut session.cursor,
                    m.eviction.as_ref(),
                    m.physical_cap,
                    tokenizer,
                    stdout,
                    &ids[*i],
                    t0,
                    cfg,
                    st,
                ));
            }
        } else if let Some((i, g)) = live.first_mut() {
            // The fused entry refuses fewer than two rows, so a lone survivor
            // steps solo rather than being dropped.
            let idx = *i;
            outcome[idx] = Some(g.step(m, gpu, tokenizer, stdout, &ids[idx]));
        }

        Ok(outcome
            .into_iter()
            .map(|o| o.unwrap_or(Qwen35Step::Continue))
            .collect())
    }

    /// Run the teardown and emit the `done` frame. Consumes the handle, because
    /// the session restore consumes the session.
    /// Unwind after [`Qwen35Step::Failed`]: report and restore the session.
    ///
    /// Exists so a driver outside this crate can handle `Failed` without
    /// touching `session`, which stays private because `qwen35_restore_or_error`
    /// consumes it.
    pub fn fail(
        self,
        m: &mut LoadedModel,
        gpu: &mut hipfire_rdna::Gpu,
        stdout: &mut dyn std::io::Write,
        id: &str,
        message: &str,
    ) {
        write_error(stdout, id, message);
        if let Some(session) = self.session {
            qwen35_restore_or_error(stdout, id, m, gpu, session);
        }
    }

    /// True while the decode loop should keep stepping. Encapsulates the cap so
    /// a driver does not need the private `st`/`cfg` fields.
    /// Hand the session back to the model's resident slot.
    ///
    /// Called after every quantum. The slot must be POPULATED between steps or
    /// the next stream's `activate_session` has nothing to save and dies with
    /// "qwen35 session missing decode state" — which is exactly how two
    /// concurrent streams failed before this existed.
    pub fn park(&mut self, m: &mut LoadedModel, gpu: &mut hipfire_rdna::Gpu) -> Result<(), String> {
        match self.session.take() {
            Some(session) => session.restore_into_loaded(m, gpu),
            None => Ok(()),
        }
    }

    /// Make this stream's session resident again and take ownership for a step.
    ///
    /// `activate_session` saves whichever stream was parked and swaps ours in;
    /// the take then gives this handle exclusive use for the quantum.
    pub fn resume(
        &mut self,
        m: &mut LoadedModel,
        gpu: &mut hipfire_rdna::Gpu,
        session_id: &str,
    ) -> Result<(), String> {
        if self.session.is_some() {
            return Ok(());
        }
        crate::session::qwen35_activate_session(m, gpu, session_id)?;
        self.session = Some(Qwen35RequestSessionState::take_from_loaded(m, gpu)?);
        Ok(())
    }

    /// Still in the prefill phase, so `st.next_token` is not yet meaningful and
    /// this stream cannot contribute a row to a decode batch. The batched march
    /// must leave it to the round-robin pass — marking it stepped in a batch it
    /// did not join would starve its prefill, because the caller skips whatever
    /// the batch claims.
    pub fn is_prefilling(&self) -> bool {
        !self.prefill_pending.is_empty()
    }

    pub fn should_continue(&self) -> bool {
        self.st.generated < self.cfg.max_tokens
    }

    pub fn finish(
        self,
        m: &mut LoadedModel,
        gpu: &mut hipfire_rdna::Gpu,
        stdout: &mut dyn std::io::Write,
        id: &str,
    ) {
        let Qwen35Generation {
            session,
            prefill_pending: _,
            st,
            cfg,
            nl,
            t0,
            t_prefill,
            prefill_tokens,
            evidence_dir,
            moe_router_histogram,
            pflash_summary,
            pflash_bypass_reason,
            pflash_alpha,
        } = self;
        qwen35_finish_generation(
            m,
            gpu,
            stdout,
            id,
            session.expect("finish requires a resumed stream"),
            st.generated,
            &nl,
            cfg.im_end_token,
            t0,
            t_prefill,
            prefill_tokens,
            evidence_dir.as_deref(),
            moe_router_histogram,
            pflash_summary,
            pflash_bypass_reason,
            pflash_alpha,
        );
    }
}

/// What [`generate_start`] decided.
pub enum Qwen35Start {
    /// A qwen35 AR generation is prefilled and ready to be marched.
    Ready(Qwen35Generation),
    /// Nothing to march: the request was served by another route (spec-decode,
    /// VL, the llama path) or already failed and reported it.
    Handled,
}

#[allow(clippy::too_many_arguments)]
/// [`generate`] minus the decode loop: frames the prompt, prefills, and returns
/// the handle the caller marches. Executor v2 §M3b0.75 — the entry the executor
/// uses instead of running a whole request inline.
///
/// The split is HERE, above the framing prologue, not at the qwen35 arm. The
/// prologue is what produces `new_tokens` / `nl` / `im_end_token` / the attractor
/// pairs; a cut below it yields a function the daemon cannot call, because it
/// holds none of those. Measured and recorded in the M3b0.75 plan note.
#[allow(clippy::too_many_arguments)]
/// The default autoregressive text generate path for the qwen35 / qwen3 (llama)
/// families, and the central text dispatcher: frames the prompt (chat template
/// or raw), prefills (optionally PFlash-compressed), and runs the per-token
/// sample → EOS-filter → loop-guard → stream loop with multi-turn KV + eviction.
/// Delegates to the spec-decode fast paths ([`generate_mtp`], [`generate_dflash`],
/// [`generate_multi`]) and to the non-qwen35 arch paths ([`generate_deepseek4`]
/// etc.) when the loaded model calls for them.
/// Latency band for inline prefill, in tokens (`HIPFIRE_PREFILL_BAND_TOKENS`).
///
/// A prefill dispatch is indivisible, so it sets the floor on how long a
/// higher-priority stream waits for the GPU. Banding cuts it into pieces that
/// each end at a committed KV boundary — `seq_pos` advanced, KV and recurrent
/// state written — which is precisely the state a continued conversation turn
/// resumes from, so nothing extra has to be preserved to stop between bands.
///
/// Unset keeps the single-dispatch behaviour. Values below 2 are rejected for
/// the same reason the fused-batch knob rejects them: a 1-token band drives the
/// batched path per token, which is the serial path wearing a different name.
///
/// Measured free: total prefill was flat across unbanded / 128 / 32 / 8-token
/// bands (14952 / 14917 / 14903 / 14918 ms on a two-session prompt), so the band
/// size can be chosen from the latency target alone.
/// Log which prefill arm each request takes (`HIPFIRE_PREFILL_PATH_TRACE`).
/// Draft block size B for spec decode, overriding what the drafter was trained
/// at (`HIPFIRE_DFLASH_BLOCK`).
///
/// Worth exposing because B and tau interact and the trained B is not
/// necessarily the throughput optimum. At B=8 with tau 2.42 on
/// Qwen3.8-27B/DFlash2, five of eight drafted positions are discarded every
/// cycle, and the draft phase is charged for all of them (52 ms) while a
/// rejection also pays the replay (67-336 ms). Shrinking B cuts both terms
/// without touching acceptance per position; a batched verify costs about one
/// weight sweep whether B is 3 or 8.
///
/// `spec_step_dflash` has taken a `block_size_override` all along -- its own
/// comment anticipates "a caller doing adaptive-B based on rolling tau" -- but
/// no serving caller ever passed one.
fn dflash_block_override() -> Option<usize> {
    std::env::var("HIPFIRE_DFLASH_BLOCK")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|b| *b >= 2)
}

fn prefill_path_trace() -> bool {
    std::env::var("HIPFIRE_PREFILL_PATH_TRACE").is_ok()
}

/// Run prefill inside the march loop instead of the frame handler.
/// ON by default; `HIPFIRE_MARCH_PREFILL=0` restores the frame-handler prefill.
///
/// Defaulted on once the case it exists for was finally measurable. Over the
/// stdin protocol it looked worth only ~13%, because stdin drains every frame
/// before the march and so a request can never arrive mid-prefill. Over
/// `--listen`, injecting a priority-9 request 2 s into a bulk prefill:
///
///   frame-handler prefill   8516.7 / 8505.5 ms client time to first token
///   march-driven + strict    575.2 /  571.0 ms
///
/// 14.8x. Parity is byte-identical on both drivers — the executor-v2 march and
/// the inline `generate` loop — banded and unbanded.
fn march_driven_prefill() -> bool {
    !matches!(
        std::env::var("HIPFIRE_MARCH_PREFILL").ok().as_deref(),
        Some("0" | "false" | "off" | "no")
    )
}

fn prefill_band_tokens() -> Option<usize> {
    std::env::var("HIPFIRE_PREFILL_BAND_TOKENS")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|&v| v >= 2)
}

pub fn generate_start(
    m: &mut LoadedModel,
    gpu: &mut hipfire_rdna::Gpu,
    drafter_gpu: Option<&mut hipfire_rdna::Gpu>,
    stdout: &mut dyn std::io::Write,
    id: &str,
    prompt: &str,
    system_prompt: Option<&str>,
    temp: f32,
    top_p: f32,
    top_k: usize,
    max_tokens: usize,
    repeat_penalty: f32,
    repeat_window: usize,
    presence_penalty: f32,
    frequency_penalty: f32,
    budget_alert_at_tok: usize,
    budget_alert_text: &str,
    max_think_tokens: usize,
    assistant_prefix: prompt_frame::AssistantPrefix,
    pflash_state: Option<&mut hipfire_arch_qwen35::pflash::PflashState>,
    pflash_cfg: Option<&hipfire_arch_qwen35::pflash::PflashConfig>,
    tools: Option<&[serde_json::Value]>,
    messages_history: Option<&[prompt_frame::Message]>,
    think_mode: ThinkMode,
    prefill_already_done: bool,
    prefilled_prompt_tokens: Option<usize>,
    request_stop_sequences: &[String],
    evidence_dir: Option<&str>,
    // Explicit per-request `"raw"` override; `None` = auto (raw iff the model
    // has no chat_template). Threaded rather than read from a global — see
    // `effective_raw` for the cross-request leak that motivated it.
    raw_override: Option<bool>,
    // Per-stream sampler seed, derived at admission
    // (`sampler::derive_stream_seed`). `None` = no stream identity to derive
    // from (CLI, eval, tests), which falls back to `initial_rng_state()`.
    // Threaded rather than read from a global, for the same reason
    // `raw_override` is: a stream's sampling must not depend on what else is
    // running beside it.
    sampler_seed: Option<u32>,
    // Per-request identity for n-gram table scoping. Threaded for the same
    // reason as `sampler_seed`, and more urgently: picking the wrong table
    // would cross-contaminate users' stored text.
    ngram_scope: Option<crate::model::NgramRequestScope<'_>>,
) -> Qwen35Start {
    // No RNG reset here any more. This used to seed a process-global CPU sampler
    // state so a request would not inherit RNG from its predecessor; the global
    // is gone (v2 plan, M1b), because with streams interleaved at module
    // granularity a shared stream makes each request's tokens depend on whatever
    // else was sampling beside it. The GPU path already carries a function-local
    // `rng_state`, and `sampler::sample`'s CPU fallback now continues that same
    // stream instead of a global one.

    if m.registered_backend.is_some() {
        // Factory-loaded text families share one prompt/render/serve path. Fast
        // speculative and prefill-compression state is deliberately absent from
        // this capability tier.
        let _ = (
            budget_alert_at_tok,
            budget_alert_text,
            pflash_state,
            pflash_cfg,
            prefill_already_done,
            prefilled_prompt_tokens,
            think_mode,
            evidence_dir,
        );
        generate_registered_backend(
            m,
            gpu,
            stdout,
            id,
            prompt,
            system_prompt,
            temp,
            top_p,
            top_k,
            max_tokens,
            repeat_penalty,
            repeat_window,
            presence_penalty,
            frequency_penalty,
            max_think_tokens,
            assistant_prefix,
            tools,
            messages_history,
            request_stop_sequences,
            raw_override,
        );
        return Qwen35Start::Handled;
    }

    // Compress runs on the PFlash drafter handle when one is set (hetero
    // sibling device), else on the target gpu. The handle is consumed at
    // the seq_pos==0 compress site; decode always uses `gpu`.
    let mut drafter_gpu = drafter_gpu;
    // arch_id=7 (hipfire-arch-qwen2) short-circuit. The standard
    // generate() body is qwen35/llama-shaped and would panic on
    // None unwraps for q35_*/llama_* fields when applied to a
    // Qwen2 model. Route here BEFORE PFlash / DFlash / multi-GPU
    // / ChatML scaffolding since none of those are wired for
    // arch_id=7 yet (R3 bring-up scope).
    if m.arch_id == ARCH_ID_LLAMA_MISTRAL || m.arch_id == ARCH_ID_QWEN3_QWEN2_LEGACY {
        // LLaMA / Mistral / plain-Qwen3 — routed through the ServingBackend seam
        // (P3.2). generate_llama applies the chat-framing then prefill +
        // decode_loop. Fast paths (DFlash/MTP/tools-execution) not on this path.
        let _ = (
            budget_alert_at_tok,
            budget_alert_text,
            pflash_state,
            pflash_cfg,
            prefill_already_done,
        );
        generate_llama(
            m,
            gpu,
            stdout,
            id,
            prompt,
            system_prompt,
            temp,
            top_p,
            max_tokens,
            repeat_penalty,
            repeat_window,
            max_think_tokens,
            assistant_prefix,
            tools,
            messages_history,
            evidence_dir,
            raw_override,
        );
        return Qwen35Start::Handled;
    }
    if m.arch_id == ARCH_ID_ZAYA {
        // ZAYA1 — CCA attention + EDA/MoD MoE, routed through the shared
        // ServingBackend seam (same dense-AR path as nemotron). No fast paths.
        let _ = (
            budget_alert_at_tok,
            budget_alert_text,
            pflash_state,
            pflash_cfg,
            prefill_already_done,
        );
        generate_zaya(
            m,
            gpu,
            stdout,
            id,
            prompt,
            system_prompt,
            temp,
            top_p,
            max_tokens,
            repeat_penalty,
            repeat_window,
            max_think_tokens,
            assistant_prefix,
            tools,
            messages_history,
            evidence_dir,
            raw_override,
        );
        return Qwen35Start::Handled;
    }
    if m.arch_id == ARCH_ID_NEMOTRON_H || m.arch_id == ARCH_ID_MAMBA2 {
        // nemotron_h / mamba2 — routed through the Mamba-capable ServingBackend
        // seam, same dense-AR path as llama. Fast paths are not on this path.
        let _ = (
            budget_alert_at_tok,
            budget_alert_text,
            pflash_state,
            pflash_cfg,
            prefill_already_done,
        );
        generate_nemotron(
            m,
            gpu,
            stdout,
            id,
            prompt,
            system_prompt,
            temp,
            top_p,
            max_tokens,
            repeat_penalty,
            repeat_window,
            max_think_tokens,
            assistant_prefix,
            tools,
            messages_history,
            evidence_dir,
            raw_override,
        );
        return Qwen35Start::Handled;
    }
    if m.arch_id == ARCH_ID_GEMMA3_TEXT {
        // arch_id=12 (gemma3 text, e.g. medgemma-*-text). Plain dense-AR via the
        // `ServingBackend::serve` seam — same short-circuit shape as qwen2 above.
        // PFlash / DFlash / VL / multi-GPU / tools / think-budget all bypass.
        let _ = (
            budget_alert_at_tok,
            budget_alert_text,
            max_think_tokens,
            assistant_prefix,
            pflash_state,
            pflash_cfg,
            tools,
            messages_history,
            prefill_already_done,
        );
        generate_gemma3(
            m,
            gpu,
            stdout,
            id,
            prompt,
            system_prompt,
            temp,
            top_p,
            max_tokens,
            repeat_penalty,
            repeat_window,
        );
        return Qwen35Start::Handled;
    }
    if m.arch_id == ARCH_ID_GEMMA3_VL {
        // arch_id=13 (gemma3-vl / full MedGemma) with no media payload. Image and
        // video requests are routed in the daemon VL branch before calling this
        // text generate path; plain prompts reuse the VL backend's text-only
        // `ServingBackend::serve` path with an empty image slice.
        let _ = (
            budget_alert_at_tok,
            budget_alert_text,
            max_think_tokens,
            assistant_prefix,
            pflash_state,
            pflash_cfg,
            tools,
            messages_history,
            prefill_already_done,
        );
        generate_gemma3_vl_text(
            m,
            gpu,
            stdout,
            id,
            prompt,
            system_prompt,
            temp,
            top_p,
            max_tokens,
            repeat_penalty,
            repeat_window,
        );
        return Qwen35Start::Handled;
    }
    if m.arch_id == ARCH_ID_DEEPSEEK4_FLASH {
        // arch_id=9 (DeepSeek V4 Flash). Standalone bring-up — same
        // shape as the qwen2 short-circuit above. PFlash / DFlash / VL
        // / multi-GPU / sampler-budget / ChatML scaffolding all bypass.
        // We honour `system_prompt`, `temp`, `top_p`, `tools`, and
        // `messages_history` per HF V4 chat template + sampling
        // recommendations; everything else routes through future
        // follow-ups.
        let _ = (
            budget_alert_at_tok,
            budget_alert_text,
            max_think_tokens,
            assistant_prefix,
            pflash_state,
            pflash_cfg,
        );
        let _ = (repeat_penalty, repeat_window);
        generate_deepseek4(
            m,
            gpu,
            stdout,
            id,
            prompt,
            system_prompt,
            temp,
            top_p,
            max_tokens,
            think_mode,
            tools,
            messages_history,
        );
        return Qwen35Start::Handled;
    }
    if m.arch_id == ARCH_ID_MINIMAX_M2 {
        // arch_id=10 (MiniMax-M2). Minimal AR bring-up — same shape as the
        // qwen2 / deepseek4 short-circuits above. PFlash / DFlash / VL /
        // multi-GPU / sampler-budget / grammar / tools-execution all bypass.
        // We honour `system_prompt`, `temp`, `top_p`, and (via JinjaChatFrame)
        // `messages_history` + `tools` rendering; spec-decode / MTP / grammar
        // are out of scope for the scaffold.
        let _ = (
            budget_alert_at_tok,
            budget_alert_text,
            assistant_prefix,
            pflash_state,
            pflash_cfg,
            repeat_penalty,
            repeat_window,
            think_mode,
        );
        generate_minimax(
            m,
            gpu,
            stdout,
            id,
            prompt,
            system_prompt,
            temp,
            top_p,
            max_tokens,
            max_think_tokens,
            tools,
            messages_history,
            raw_override,
        );
        return Qwen35Start::Handled;
    }
    #[cfg(feature = "arch-lfm2moe")]
    if m.arch_id == ARCH_ID_LFM2_MOE {
        // arch_id=11 (LFM2.5-MoE). LFM2's arch-local path owns AR, deterministic
        // DFlash, and resident-session decode; VL / multi-GPU / sampler-budget /
        // grammar / tools-execution still bypass here. We honour `system_prompt`,
        // `temp`, `top_p`, and (via JinjaChatFrame) `messages_history` + `tools`
        // rendering; MTP / grammar are out of scope for this scaffold.
        let _ = (
            budget_alert_at_tok,
            budget_alert_text,
            assistant_prefix,
            pflash_state,
            pflash_cfg,
            repeat_penalty,
            repeat_window,
            think_mode,
        );
        generate_lfm2moe(
            m,
            gpu,
            stdout,
            id,
            prompt,
            system_prompt,
            temp,
            top_p,
            max_tokens,
            max_think_tokens,
            tools,
            messages_history,
            prefill_already_done,
            prefilled_prompt_tokens,
            raw_override,
        );
        return Qwen35Start::Handled;
    }
    // Multi-GPU pipeline-parallel dispatch (Stage 7 of #58). pp>1 is refused
    // at load when DFlash / CASK / PFlash / VL is requested, so this branch
    // doesn't need to thread any of those args through.
    if m.pp > 1 {
        generate_multi(
            m,
            gpu,
            pflash_state,
            pflash_cfg,
            stdout,
            id,
            prompt,
            system_prompt,
            temp,
            top_p,
            top_k,
            max_tokens,
            repeat_penalty,
            repeat_window,
            presence_penalty,
            frequency_penalty,
            budget_alert_at_tok,
            budget_alert_text,
            max_think_tokens,
            assistant_prefix,
            tools,
            messages_history,
            request_stop_sequences,
            raw_override,
        );
        return Qwen35Start::Handled;
    }
    // DFlash fast path -- only when a draft model is loaded AND temperature is
    // effectively 0 (DFlash is greedy-only in this integration). Skip the
    // normal AR sampling setup entirely.
    //
    // Exception: thinking-on + max_think_tokens currently needs the AR path.
    // DFlash's budget cap can close/strip the think span but does not yet
    // continue into visible answer text after the forced close. AR already
    // splices </think> through KV and continues generation, so route budgeted
    // thinking requests there until DFlash continuation is implemented.
    let budgeted_thinking_needs_ar = max_think_tokens > 0
        && !matches!(assistant_prefix, prompt_frame::AssistantPrefix::ClosedThink);
    if prefill_already_done && m.dflash.is_some() {
        write_error(
            stdout,
            id,
            "prefill_already_done is not supported on DFlash-loaded models",
        );
        return Qwen35Start::Handled;
    }
    if prefill_already_done && pflash_state.is_some() {
        write_error(
            stdout,
            id,
            "prefill_already_done is not supported when PFlash state is loaded",
        );
        return Qwen35Start::Handled;
    }

    // KVarN (and the deferred-hierarchical two-tier cache built on it) require the
    // per-token attention dispatch (kv_cache_attention_dispatch). The spec-decode
    // paths (DFlash, MTP) batch-prefill the prompt and would bypass it, leaving the
    // KVarN window/records (and hier hot ring) unpopulated → garbage. Route kvarn
    // models to the plain AR path below (per-token forward_scratch prefill+decode).
    let kvarn_active = m.kv_cache().map(|c| c.quant_kvarn).unwrap_or(false);
    // Default-on since 2026-08-27 (see kvarn_forced_per_token's doc comment for
    // the measurement history); `=0` rolls back to per-token prefill with the
    // drafter off.
    let kvarn_specdecode_ok = std::env::var("HIPFIRE_KVARN_BATCHED_PREFILL")
        .ok()
        .as_deref()
        != Some("0");
    // Under the `=0` rollback a loaded drafter never runs. Nothing else says
    // so, so an operator benchmarking "DFlash on kvarn8" would measure plain AR
    // and read it as a DFlash number. Say it once.
    if m.dflash.is_some() && kvarn_active && !kvarn_specdecode_ok {
        static WARNED: std::sync::Once = std::sync::Once::new();
        WARNED.call_once(|| {
            tracing::warn!(
                "a DFlash drafter is loaded but HIPFIRE_KVARN_BATCHED_PREFILL=0 \
                 forces KVarN per-token, so decode is running PLAIN AR. Any tok/s \
                 measured here is NOT a DFlash number. Unset the variable to engage \
                 the drafter."
            );
        });
    }
    if m.dflash.is_some()
        && (!kvarn_active || kvarn_specdecode_ok)
        && temp <= 1e-6
        && is_qwen35_family_arch_id(m.arch_id)
        && !budgeted_thinking_needs_ar
    {
        // PFlash + DFlash decode path is not yet wired -- the DFlash spec
        // loop builds its own prompt token stream internally, so the
        // generate() PFlash block below never runs. Surface this loud so
        // an operator who set prefill_compression != off sees a clear
        // bypass event instead of silently getting full-prefill behavior
        // they didn't ask for. Compression-on-DFlash lands in a future
        // phase that threads PflashState through generate_dflash().
        let mut dflash_bypass_reason: Option<&'static str> = None;
        let dflash_alpha = pflash_cfg.as_ref().map(|c| c.alpha);
        if let Some(cfg) = pflash_cfg.as_ref() {
            if cfg.mode != hipfire_arch_qwen35::pflash::PflashMode::Off {
                let _ = writeln!(
                    stdout,
                    r#"{{"type":"pflash_bypass","id":"{}","reason":"dflash_decode_active (pflash compression on the DFlash path is a follow-up; set dflash_mode=off to compress with AR decode)"}}"#,
                    id,
                );
                let _ = stdout.flush();
                dflash_bypass_reason = Some("dflash_decode_active");
            }
        }
        // max_think_tokens is now enforced inside generate_dflash (it
        // mirrors the AR path's <think>/</think> counter). The "ignored
        // on DFlash" warning that used to live here is gone -- the cap
        // is real on both paths now.
        generate_dflash(
            m,
            gpu,
            stdout,
            id,
            prompt,
            system_prompt,
            max_tokens,
            max_think_tokens,
            assistant_prefix,
            dflash_bypass_reason,
            dflash_alpha,
            tools,
            messages_history,
            request_stop_sequences,
            raw_override,
            ngram_scope,
        );
        // Silence unused-variable warnings for the params DFlash doesn't
        // consume (top_p / repeat penalties are AR-only sampling knobs;
        // pflash_state is bypassed on the DFlash decode path).
        let _ = (
            top_p,
            repeat_penalty,
            repeat_window,
            budget_alert_at_tok,
            budget_alert_text,
            pflash_state,
        );
        return Qwen35Start::Handled;
    }

    // MTP spec-decode: qwen35 model with a co-trained MTP head, no DFlash
    // drafter, greedy, mtp_mode enabled. Uses the non-tree spec_step_mtp
    // (FP32 DeltaNet state is tree-incompatible — see generate_mtp / TODO.md).
    if m.dflash.is_none()
        && m.mtp_weights_present
        && m.mtp_mode != "off"
        && !kvarn_active
        && temp <= 1e-6
        && is_qwen35_family_arch_id(m.arch_id)
        && !prefill_already_done
        && !budgeted_thinking_needs_ar
    {
        generate_mtp(
            m,
            gpu,
            stdout,
            id,
            prompt,
            system_prompt,
            max_tokens,
            max_think_tokens,
            assistant_prefix,
            tools,
            messages_history,
            request_stop_sequences,
            raw_override,
        );
        let _ = (
            top_p,
            repeat_penalty,
            repeat_window,
            budget_alert_at_tok,
            budget_alert_text,
            pflash_state,
        );
        return Qwen35Start::Handled;
    }

    let is_qwen35_ar = is_qwen35_family_arch_id(m.arch_id);
    let mut q35_session = if is_qwen35_ar {
        match Qwen35RequestSessionState::take_from_loaded(m, gpu) {
            Ok(session) => Some(session),
            Err(e) => {
                write_error(stdout, id, &format!("qwen35 request session state: {e}"));
                return Qwen35Start::Handled;
            }
        }
    } else {
        None
    };

    // Auto-reset on multi-turn rollover. When eviction is active (operator
    // enabled cask_sidecar at load), the physical buffer is bounded by
    // budget+beta+safety regardless of conversation length, so reset never
    // needs to fire — eviction reclaims slots after each token. When eviction
    // is OFF, physical grows unbounded up to max_seq; reset when we'd overrun.
    let tokenizer = m.tokenizer.as_ref().unwrap();
    let prompt_est = tokenizer.encode(prompt).len() + 20;
    let current_seq_pos = q35_session
        .as_ref()
        .map(|s| s.cursor.seq_pos)
        .unwrap_or(m.active.cursor.seq_pos);
    if !prefill_already_done
        && m.eviction.is_none()
        && current_seq_pos + prompt_est + max_tokens > m.max_seq
    {
        tracing::warn!(
            "context full ({}/{}) — resetting conversation",
            current_seq_pos,
            m.max_seq
        );
        if let Some(session) = q35_session.as_mut() {
            session.reset(gpu);
        } else {
            m.active.cursor.seq_pos = 0;
            m.active.cursor.conversation_tokens.clear();
            if let Some(kv) = m.llama_kv.as_mut() {
                kv.compact_offset = 0;
            }
        }
    }

    // `nl` is needed for the trailer write after natural <|im_end|>
    // termination; `im_end` derives the EOS-check token id. Other
    // ChatML scaffolding tokens are now built inside hipfire-prompt.
    let im_end = tokenizer.encode("<|im_end|>");
    let nl = tokenizer.encode("\n");
    let raw_q_tokens = tokenizer.encode(prompt);

    // ── PFlash compression (Phase 4.1 #93) ──────────────────────────────
    //
    // Only on first turn (seq_pos == 0). Multi-turn compression of newly-
    // added user content has knock-on effects on prior KV state that we
    // haven't validated yet, so subsequent turns always bypass.
    //
    // Compression operates on the user's actual content tokens
    // (`raw_q_tokens`); chat-template scaffolding (im_start / role / nl /
    // im_end) wraps the result AFTER and is never compressed away.
    // Empty must_keep_spans is correct: there are no chat boundaries
    // INSIDE q_tokens (they live in the scaffolding the daemon adds).
    //
    // Bypass / compressed status is reported as a `pflash_compressed` or
    // `pflash_bypass` event so operators can see what the request actually
    // ran through.
    //
    // Tool-call detection: the prompt may contain a `<tool_call>` token
    // that the parser uses for structure. Compressing those tokens away
    // would corrupt the response shape, so we surface a ToolCall request
    // kind to the gate and let `decide_bypass` reject the request loudly.
    //
    // Two scan locations:
    //   1. raw_q_tokens (the user message itself).
    //   2. system_prompt -- the OpenAI serve path puts tool definitions
    //      and the `<tool_call>` format example in the system prompt
    //      when `body.tools` is present. A
    //      first-turn user message with tools therefore needs a system-
    //      prompt scan or it would slip through as Text and get its
    //      schema text mangled by compression.
    //
    // Detection is best-effort -- the special-token id is missing on
    // older vocabs, in which case the gate just routes through Text.
    let request_kind = match tokenizer.special_token_id("<tool_call>") {
        Some(tid) => {
            let in_user = raw_q_tokens.iter().any(|&t| t == tid);
            let in_system = system_prompt
                .map(|s| tokenizer.encode(s).iter().any(|&t| t == tid))
                .unwrap_or(false);
            if in_user || in_system {
                hipfire_arch_qwen35::pflash::RequestKind::ToolCall
            } else {
                hipfire_arch_qwen35::pflash::RequestKind::Text
            }
        }
        None => hipfire_arch_qwen35::pflash::RequestKind::Text,
    };

    // Stashed CompressedPrompt summary (when compression actually fired);
    // appended to the `done` event later so a streaming client gets one
    // consolidated line. None means no compression happened on this request.
    let mut pflash_summary: Option<hipfire_arch_qwen35::pflash::CompressedPrompt> = None;
    // Bypass reason when compression was attempted but skipped (mode != Off
    // and a drafter was loaded). PRD §3.1 requires "bypass reason if
    // skipped" in the done object.
    let mut pflash_bypass_reason: Option<String> = None;
    // Effective alpha for this request (from cfg if pflash_state is loaded).
    // PRD §3.1 lists alpha as a required done-object field.
    let pflash_alpha: Option<f32> = pflash_cfg.map(|c| c.alpha);
    if std::env::var("HIPFIRE_PFLASH_DEBUG").is_ok() {
        tracing::debug!(
            "gen: state={} cfg-present seq_pos={} q={} drafter_gpu={}",
            pflash_state.is_some(),
            q35_session
                .as_ref()
                .map(|s| s.cursor.seq_pos)
                .unwrap_or(m.active.cursor.seq_pos),
            raw_q_tokens.len(),
            drafter_gpu.is_some()
        );
    }
    let q_tokens = if let (Some(state), Some(cfg)) = (pflash_state, pflash_cfg) {
        let seq_pos = q35_session
            .as_ref()
            .map(|s| s.cursor.seq_pos)
            .unwrap_or(m.active.cursor.seq_pos);
        if seq_pos == 0 {
            let compress_gpu: &mut hipfire_rdna::Gpu = drafter_gpu.as_deref_mut().unwrap_or(gpu);
            // Sibling-device drafter: bind its device before compress, then
            // restore the target binding for decode. No-op when shared.
            compress_gpu.bind_thread_or_warn();
            let decision = hipfire_arch_qwen35::pflash::maybe_compress_prompt(
                compress_gpu,
                state,
                cfg,
                &raw_q_tokens,
                request_kind,
                &[],
            );
            gpu.bind_thread_or_warn();
            match decision {
                Ok(hipfire_arch_qwen35::pflash::PflashDecision::Compressed(cp)) => {
                    tracing::debug!(
                        "COMPRESSED {} -> {} tok dev1 ({}ms)",
                        cp.source_tokens,
                        cp.kept_tokens,
                        cp.timings.total_ms
                    );
                    let _ = writeln!(
                        stdout,
                        r#"{{"type":"pflash_compressed","id":"{}","source_tokens":{},"kept_tokens":{},"keep_ratio":{:.6},"source_md5":"{}","compressed_md5":"{}","score_ms":{},"select_ms":{},"gather_ms":{},"total_ms":{}}}"#,
                        id,
                        cp.source_tokens,
                        cp.kept_tokens,
                        cp.kept_tokens as f32 / cp.source_tokens.max(1) as f32,
                        cp.source_md5,
                        cp.compressed_md5,
                        cp.timings.score_ms,
                        cp.timings.select_ms,
                        cp.timings.gather_ms,
                        cp.timings.total_ms,
                    );
                    let _ = stdout.flush();
                    let token_ids = cp.token_ids.clone();
                    pflash_summary = Some(cp);
                    token_ids
                }
                Ok(hipfire_arch_qwen35::pflash::PflashDecision::Bypass { reason }) => {
                    tracing::debug!("BYPASS reason={} q={}", reason.as_str(), raw_q_tokens.len());
                    // Only emit bypass events for non-trivial reasons.
                    // ModeOff is the silent default; nothing to report.
                    if !matches!(reason, hipfire_arch_qwen35::pflash::BypassReason::ModeOff) {
                        let r = reason.as_str();
                        let _ = writeln!(
                            stdout,
                            r#"{{"type":"pflash_bypass","id":"{}","reason":"{}"}}"#,
                            id,
                            r.replace('"', "'"),
                        );
                        let _ = stdout.flush();
                        // Stash for the `done` object too so a single-line
                        // log scrape sees both the bypass reason and the
                        // request's prefill timings.
                        pflash_bypass_reason = Some(r);
                    }
                    raw_q_tokens
                }
                Err(e) => {
                    tracing::warn!("pflash compress failed: {e}");
                    let _ = writeln!(
                        stdout,
                        r#"{{"type":"pflash_error","id":"{}","reason":"{}"}}"#,
                        id,
                        e.to_string().replace('"', "'"),
                    );
                    let _ = stdout.flush();
                    raw_q_tokens
                }
            }
        } else {
            raw_q_tokens
        }
    } else {
        raw_q_tokens
    };

    // ChatML framing — two paths:
    //
    //   1) `HIPFIRE_JINJA_CHAT=1` AND model carries an embedded chat_template
    //      AND first turn (seq_pos == 0): render through `JinjaChatFrame`
    //      against the upstream HF Jinja template, producing the byte
    //      sequence the model was actually trained on (fixes the "hand-roll
    //      drifted from upstream template" class — XML tool calls on
    //      Qwen3.5/3.6 instead of JSON, `<|im_start|>user` for tool
    //      responses instead of `<|im_start|>tool`, etc.). PFlash
    //      compression is bypassed under Jinja for now (q_tokens not
    //      reusable when the template renders to a String).
    //
    //   2) Default: hand-rolled `prompt_frame::ChatFrame::Plain`
    //      scaffold, byte-identical to today's behavior.
    //
    // Multi-turn (seq_pos > 0) currently always uses path 2 — Jinja
    // single-turn parity is Stage 2; multi-turn message-history state on
    // the daemon side is Stage 2 follow-up.
    //
    // Thinking-off interop with `assistant_prefix`: the CLI sets BOTH
    // `max_think_tokens = 1` AND `assistant_prefix = ClosedThink` when
    // the request asks for non-thinking. The Jinja path keys off
    // `max_think_tokens != 1` for `enable_thinking`; the Plain path
    // honors `assistant_prefix` directly (ClosedThink emits a closed
    // `<think></think>` block after the assistant prefix). Each path
    // picks up the signal it needs.
    let jinja_enabled = std::env::var("HIPFIRE_JINJA_CHAT").ok().as_deref() == Some("1");
    let seq_pos_for_prompt = if prefill_already_done {
        0
    } else {
        q35_session
            .as_ref()
            .map(|s| s.cursor.seq_pos)
            .unwrap_or(m.active.cursor.seq_pos)
    };
    let try_jinja = jinja_enabled && seq_pos_for_prompt == 0 && m.chat_template.is_some();
    let new_tokens = if try_jinja {
        let template = m.chat_template.as_ref().unwrap();
        let frame = prompt_frame::JinjaChatFrame {
            tokenizer,
            template,
            system: system_prompt,
            user: prompt,
            enable_thinking: max_think_tokens != 1,
            bos_token: None,
        };
        // Phase 1 of Jinja-everywhere migration: when the caller supplies
        // either a `tools` array or a `messages` history (or both), route
        // through `render_messages` so the upstream template's
        // `{% if tools %}` / multi-turn branches fire. With neither
        // supplied, fall through to the single-turn `render()` convenience,
        // which is byte-identical to the synthesized [system?, user]
        // path that shipped under HIPFIRE_JINJA_CHAT=1 before this change.
        let render_result = if tools.is_some() || messages_history.is_some() {
            // Synthesize [system?, user] when no explicit history was
            // provided. Tools-with-legacy-prompt is the natural OpenAI
            // function-calling shape (one turn + tool definitions).
            let synthesized: Vec<prompt_frame::Message>;
            let messages_slice: &[prompt_frame::Message] = match messages_history {
                Some(m) => m,
                None => {
                    let mut v = Vec::new();
                    if let Some(sys) = system_prompt {
                        v.push(prompt_frame::Message {
                            role: prompt_frame::Role::System,
                            content: sys.to_string(),
                            tool_calls: Vec::new(),
                            tool_call_id: None,
                        });
                    }
                    v.push(prompt_frame::Message {
                        role: prompt_frame::Role::User,
                        content: prompt.to_string(),
                        tool_calls: Vec::new(),
                        tool_call_id: None,
                    });
                    synthesized = v;
                    &synthesized
                }
            };
            frame.render_messages(messages_slice, tools, None)
        } else {
            frame.render()
        };
        match render_result {
            Ok(rendered) => tokenizer.encode(&rendered),
            Err(e) => {
                tracing::warn!("jinja render failed ({e}) — falling back to Plain");
                prompt_frame::ChatFrame {
                    tokenizer,
                    system: system_prompt,
                    user: "",
                    assistant_prefix,
                    raw: effective_raw(m, raw_override),
                }
                .build_with_user_tokens(&q_tokens)
            }
        }
    } else {
        prompt_frame::ChatFrame {
            tokenizer,
            system: if seq_pos_for_prompt == 0 {
                system_prompt
            } else {
                None
            },
            user: "", // unused: we pass tokens directly via build_with_user_tokens
            assistant_prefix,
            raw: effective_raw(m, raw_override),
        }
        .build_with_user_tokens(&q_tokens)
    };

    // KV-budget guard. Without eviction the physical buffer is the hard cap;
    // we must fit prefill + generation + trailer in one allocation. With
    // eviction, physical is bounded by physical_cap regardless of total tokens
    // — the chunked prefill below calls maybe_evict between chunks, and the
    // decode loop evicts after every token. The only ceiling under eviction is
    // the advertised context window (max_seq) — refuse requests that would
    // overflow it in absolute position terms (current absolute + new).
    let trailer = nl.len();
    let current_seq_pos = q35_session
        .as_ref()
        .map(|s| s.cursor.seq_pos)
        .unwrap_or(m.active.cursor.seq_pos);
    let budget_prefill_tokens = if prefill_already_done {
        0
    } else {
        new_tokens.len()
    };
    let absolute_pos = if let Some(session) = q35_session.as_ref() {
        session.cursor.seq_pos + session.kv_cache().compact_offset
    } else {
        m.active.cursor.seq_pos + m.llama_kv.as_ref().map(|kv| kv.compact_offset).unwrap_or(0)
    };
    if m.eviction.is_none() {
        if current_seq_pos + budget_prefill_tokens + max_tokens + trailer > m.physical_cap {
            let _ = writeln!(
                stdout,
                r#"{{"type":"error","id":"{}","message":"request exceeds loaded KV budget: seq_pos={} + prefill={} + max_tokens={} + trailer={} > physical_cap={} — reload model with a larger max_seq"}}"#,
                id, current_seq_pos, budget_prefill_tokens, max_tokens, trailer, m.physical_cap
            );
            let _ = stdout.flush();
            if let Some(session) = q35_session.take() {
                qwen35_restore_or_error(stdout, id, m, gpu, session);
            }
            return Qwen35Start::Handled;
        }
    } else if absolute_pos + budget_prefill_tokens + max_tokens + trailer > m.max_seq {
        let _ = writeln!(
            stdout,
            r#"{{"type":"error","id":"{}","message":"request exceeds advertised context window: absolute={} + prefill={} + max_tokens={} + trailer={} > max_seq={}"}}"#,
            id, absolute_pos, budget_prefill_tokens, max_tokens, trailer, m.max_seq
        );
        let _ = stdout.flush();
        if let Some(session) = q35_session.take() {
            qwen35_restore_or_error(stdout, id, m, gpu, session);
        }
        return Qwen35Start::Handled;
    }

    let im_end_token = if im_end.len() == 1 {
        Some(im_end[0])
    } else {
        None
    };
    // Special-token attractor blocking (#111). Resolve the token IDs once;
    // each pair is `Some` only when the tokenizer registers both opener
    // and closer as single special tokens (Qwen3+ vocabs). Older vocabs
    // return `None` and the block is silently skipped — no behavior
    // change.
    let tool_call_pair = match (
        tokenizer.special_token_id("<tool_call>"),
        tokenizer.special_token_id("</tool_call>"),
    ) {
        (Some(o), Some(c)) => Some((o, c)),
        _ => None,
    };
    let think_pair = match (
        tokenizer.special_token_id("<think>"),
        tokenizer.special_token_id("</think>"),
    ) {
        (Some(o), Some(c)) => Some((o, c)),
        _ => None,
    };
    let prefill_tokens = new_tokens.len();
    let t0 = Instant::now();

    if is_qwen35_family_arch_id(m.arch_id) {
        // Qwen3.5 / Qwen3.5-MoE — multi-turn: prefill only the NEW turn tokens,
        // continuing from session.cursor.seq_pos (KV cache + DeltaNet state are cumulative)
        let mut session = q35_session.take().expect("qwen35 request session state");
        if prefill_already_done {
            let current_position = session.cursor.seq_pos + session.kv_cache().compact_offset;
            let expected_position = prefilled_prompt_tokens.unwrap_or(new_tokens.len());
            if current_position != expected_position {
                write_error(
                    stdout,
                    id,
                    &format!(
                        "prefill_already_done requested but active session position {} does not match expected prefilled prompt token count {}",
                        current_position,
                        expected_position
                    ),
                );
                qwen35_restore_or_error(stdout, id, m, gpu, session);
                return Qwen35Start::Handled;
            }
        }
        let config = m.q35_config.as_ref().unwrap();
        let weights = m.q35_weights.as_ref().unwrap();
        let scratch = m.q35_scratch.as_ref().unwrap();
        // Disjoint field-path borrow: kv and dn are distinct fields of
        // session.sequence_state, so both &mut live simultaneously.
        let kv = session
            .sequence_state
            .kv
            .as_mut()
            .expect("qwen35 session always has KV");
        let dn = session
            .sequence_state
            .recurrent
            .as_mut()
            .expect("qwen35 session has DeltaNet state")
            .as_any_mut()
            .downcast_mut::<qwen35::DeltaNetState>()
            .expect("qwen35 session recurrent state is DeltaNetState");
        let moe_router_histogram = DaemonMoeRouterHistogramGuard::start(evidence_dir, config);

        // Prefill this turn's tokens via the batched prefill entry point.
        // On gfx11+ for MQ4/HFQ4/MQ6/HFQ6 weights this hits the WMMA GEMM
        // fast path; other archs fall back to dp2 / FP16-packed / scalar
        // variants. The one sequential hotspot inside is the gated_delta_net
        // Q8 state update (N sequential per-token calls per LA layer, byte-
        // exact with decode to keep the quality gate green).
        //
        // Note: forward_prefill_batch launches HIP kernels asynchronously.
        // The t_prefill mark below lives AFTER the first sample_top_p, whose
        // D2H readback of tok0 forces a device sync — that's the point at
        // which the first token is actually ready to stream. Placing the
        // mark earlier captures CPU-dispatch time, which under-reports
        // prefill by a large factor (prefill_tok_s ~5–10× too optimistic).
        //
        // Under eviction: chunk prefill to the (budget+beta) eviction window
        // and call `maybe_evict` between chunks so physical never exceeds
        // physical_cap. Chunk size caps out at physical capacity available —
        // when physical is at post-evict `budget`, a full `beta`-sized chunk
        // can run before the next eviction fires.
        if !prefill_already_done {
            // Deferred-hierarchical KV: on a continued turn (seq_pos > 0), drain the
            // hot ring into cold here at the prefill entry — the between-turns idle
            // gap, off the decode critical path — before the new prompt tokens append.
            // (The batch-protocol path does this in qwen35_prefill_active_session;
            // single `generate` prefills inline, so mirror it here.)
            if session.cursor.seq_pos > 0 {
                if let Some(h) = kv.hier.as_mut() {
                    if h.enabled {
                        let keep = std::env::var("HIPFIRE_KV_IDLE_KEEP")
                            .ok()
                            .and_then(|s| s.parse().ok())
                            .unwrap_or(0usize);
                        h.idle_compact(gpu, keep).unwrap();
                    }
                }
            }
            // Which prefill arm ran, and why. Three arms serve `generate` and
            // they differ by orders of magnitude in granularity, so "which one
            // did it take" is the first question of any prefill investigation —
            // and until now the only way to answer it was to read the source and
            // guess. Two separate wrong conclusions were reached that way.
            if prefill_path_trace() {
                let arm = if kv.quant_kvarn && kvarn_forced_per_token() {
                    "per_token(kvarn rollback)"
                } else if m.eviction.is_some() || prefill_band_tokens().is_some() {
                    "chunked"
                } else {
                    "single_dispatch"
                };
                eprintln!(
                    "[prefill-path] id={id} arm={arm} tokens={} seq_pos={} band={:?} eviction={} march={}",
                    new_tokens.len(),
                    session.cursor.seq_pos,
                    prefill_band_tokens(),
                    m.eviction.is_some(),
                    march_driven_prefill(),
                );
            }
            if !march_driven_prefill() {
                match qwen35_prefill_tokens(
                    gpu,
                    weights,
                    config,
                    scratch,
                    kv,
                    dn,
                    &mut session.cursor,
                    m.eviction.as_ref(),
                    &new_tokens,
                    id,
                ) {
                    Ok(_) => {}
                    Err(e) => {
                        write_error(stdout, id, &e);
                        qwen35_restore_or_error(stdout, id, m, gpu, session);
                        return Qwen35Start::Handled;
                    }
                }
                session
                    .cursor
                    .conversation_tokens
                    .extend_from_slice(&new_tokens);
            }
        }
        // Deferred prefill carries the prompt on the handle instead. The extend
        // above and the sample below both move to the transition in `step`.
        let deferred_prefill: Vec<u32> = if prefill_already_done || !march_driven_prefill() {
            Vec::new()
        } else {
            new_tokens.clone()
        };

        // ngram scope for the repeat penalty: ONLY generated tokens (never the
        // prompt). Prior design included the user's prompt as an anti-loop
        // anchor, but that penalizes the very tokens we're asked to recall
        // (names, numbers, facts) under MQ4/MQ6 quantizations that are more
        // RP-sensitive than llama.cpp's Q4_K. First sample: empty scope (no
        // generated tokens yet); subsequent samples: generated-so-far only.
        let ngram_scope_start = if prefill_already_done {
            session
                .cursor
                .conversation_tokens
                .len()
                .saturating_sub(session.prefilled_generated_suffix_len)
        } else {
            // Deferred prefill has NOT extended `conversation_tokens` yet, and
            // this index must describe the state after it does — the first
            // sample's n-gram scope is empty by construction (`#111`), and it
            // stops being empty if the index is taken before the extend. The
            // extend is deterministic, so add its length here.
            session.cursor.conversation_tokens.len() + deferred_prefill.len()
        };
        session.prefilled_generated_suffix_len = 0;

        // Generate. GPU-side sampling eliminates per-token logits download +
        // CPU softmax + CPU repeat penalty. Closes the 2× gap between raw
        // bench throughput and daemon throughput.
        //
        // Kernel signature reads `repeat_tokens[0..repeat_window]`, so we
        // only need to upload the tokens that will actually be read — no
        // need to clear the buffer between calls. The upload is on the same
        // stream as the sample kernel launch, so the copy and compute pipeline
        // naturally.
        let vocab_size = config.vocab_size;
        let mut rng_state: u32 =
            sampler_seed.unwrap_or_else(hipfire_runtime::sampler::initial_rng_state);
        let repeat_buf_cap = (scratch.repeat_buf.buf.size() / 4).min(repeat_window);

        // Build the list of paired (open, close) attractor pairs once;
        // collect_unclosed_attractor_blocks decides per-call
        // which openers (if any) trip the depth threshold.
        let attractor_pairs: Vec<(u32, u32)> = tool_call_pair
            .into_iter()
            .chain(think_pair.into_iter())
            .collect();

        // First sample: use conversation so far as scope.
        // Clamp: under march-driven prefill `ngram_scope_start` deliberately
        // indexes PAST a `conversation_tokens` the deferred prefill has not
        // written yet, so slicing it raw panics. The clamped slice is empty,
        // which is correct — `cfg0` below is only consumed when prefill already
        // ran, and `step` re-derives the scope from `cfg.ngram_scope_start` once
        // the prompt is recorded.
        let ngram_scope = &session.cursor.conversation_tokens
            [ngram_scope_start.min(session.cursor.conversation_tokens.len())..];
        // #111 attractor block: empty `ngram_scope` on first sample (no
        // generated tokens yet), so the unclosed-depth is always 0 and
        // `blocked` is empty. Still call collect_* for symmetry with
        // the loop body, in case a future change moves this block into
        // a multi-step warmup.
        let mut blocked0: Vec<u32> = Vec::new();
        collect_unclosed_attractor_blocks(ngram_scope, &attractor_pairs, 20, 2, &mut blocked0);
        let cfg0 = SamplerConfig {
            temperature: temp,
            top_p,
            top_k,
            repeat_penalty,
            // Window is bounded by the GPU repeat_buf capacity. Pre-PR3 code did this
            // bound by setting `scope_start = len - repeat_buf_cap`
            // and passing `scope.len()` to the kernel; we let
            // sampler::sample do the same `min(window, buf_cap)`
            // internally.
            repeat_window: repeat_buf_cap,
            presence_penalty,
            frequency_penalty,
            blocked_tokens: blocked0,
        };
        // Under march-driven prefill nothing has run yet, so there are no logits
        // to sample from. `st.next_token` stays a placeholder that NOTHING may
        // read while `prefill_pending` is non-empty; `step` fills it in on the
        // band that drains the prompt, using the same `cfg` fields this block
        // would have used.
        let tok0 = if !deferred_prefill.is_empty() {
            0
        } else {
            sampler::sample(
                gpu,
                &scratch.logits,
                &scratch.sample_buf,
                &scratch.repeat_buf,
                vocab_size,
                ngram_scope,
                &cfg0,
                &mut rng_state,
            )
        };
        // First token is ready (sample_top_p's D2H forces GPU sync). This is
        // the user-observable "time to first token" boundary — prefill above,
        // decode loop below.
        let t_prefill = Instant::now();

        // Cross-token decode state, hoisted into one value so that advancing a
        // token is a CALL rather than a loop iteration — the quantum executor
        // v2's march loop needs (§M3b0). The `while` stays exactly as it was:
        // budget-alert injection can push `generated` past the iteration count,
        // so the cap is rechecked at each loop start.
        return Qwen35Start::Ready(Qwen35Generation {
            session: Some(session),
            prefill_pending: deferred_prefill,
            st: Qwen35DecodeState {
                rng_state,
                next_token: tok0,
                generated: 0,
                streamed_tokens: Vec::new(),
                bytes_fed_to_filter: 0,
                filter: chat_output_filter(m, request_stop_sequences),
                alert_fired: false,
                think_count: 0,
                prev_in_think: false,
            },
            cfg: Qwen35DecodeCfg {
                max_tokens,
                max_think_tokens,
                budget_alert_at_tok,
                budget_alert_text: budget_alert_text.to_string(),
                im_end_token,
                nl_len: nl.len(),
                vocab_size,
                top_k,
                repeat_buf_cap,
                ngram_scope_start,
                attractor_pairs,
                loop_guard: loop_guard_from_runtime_config(),
                temperature: temp,
                top_p,
                repeat_penalty,
                presence_penalty,
                frequency_penalty,
            },
            nl,
            t0,
            t_prefill,
            prefill_tokens,
            evidence_dir: evidence_dir.map(str::to_string),
            moe_router_histogram,
            pflash_summary,
            pflash_bypass_reason,
            pflash_alpha,
        });
    } else {
        // Qwen3 / LLaMA path -- multi-turn aware. This is the `not qwen35`
        // fallback, but it's specifically the llama (arch 0/1) state — every
        // other non-qwen35 arch has a dedicated route above (7/9/10/11/12
        // short-circuit; 8/13 VL route in the daemon). Guard the llama unwraps
        // so a future/misrouted arch reaching here gets a clean error instead of
        // a None-unwrap panic (it must NOT silently run on the llama path).
        if m.llama_config.is_none() {
            write_error(
                stdout,
                id,
                &format!(
                    "generate(): arch_id {} reached the llama fallback with no \
                     llama state loaded — this arch needs a dedicated generate route",
                    m.arch_id
                ),
            );
            return Qwen35Start::Handled;
        }
        let chat_template_profile = m.chat_template_profile.clone();
        let config = m.llama_config.as_ref().unwrap();
        let weights = m.llama_weights.as_ref().unwrap();
        let scratch = m.llama_scratch.as_ref().unwrap();
        let kv = m.llama_kv.as_mut().unwrap();

        let mut rng_state = 42u32;
        for (i, &tok) in new_tokens.iter().enumerate() {
            let pos = m.active.cursor.seq_pos + i;
            let (_, rng) = llama::forward_scratch(
                gpu, weights, config, tok, pos, kv, scratch, temp, top_p, rng_state, 0, 1.0,
            )
            .unwrap();
            rng_state = rng;
        }
        let this_turn_prompt_len_llama = new_tokens.len();
        m.active.cursor.seq_pos += new_tokens.len();
        m.active
            .cursor
            .conversation_tokens
            .extend_from_slice(&new_tokens);
        let ngram_scope_start_llama =
            m.active.cursor.conversation_tokens.len() - this_turn_prompt_len_llama;

        let mut out_bytes = [0u8; 8];
        gpu.hip
            .memcpy_dtoh(&mut out_bytes, &scratch.sample_buf.buf)
            .unwrap();
        let mut next_token =
            u32::from_ne_bytes([out_bytes[0], out_bytes[1], out_bytes[2], out_bytes[3]]);
        rng_state = u32::from_ne_bytes([out_bytes[4], out_bytes[5], out_bytes[6], out_bytes[7]]);
        // Prefill ends here: prompt is processed AND first token is ready (D2H
        // sync is the user-observable "time to first token" boundary). Decode
        // below measures the pure forward+sample steady-state.
        let t_prefill = Instant::now();

        let mut generated = 0;
        let mut streamed_tokens: Vec<u32> = Vec::new();
        // `bytes_fed_to_filter` is the index into the freshly-decoded
        // byte stream past which we have not yet handed bytes to the
        // filter. The filter owns UTF-8 boundary buffering and any
        // future arch quirks (Gemma 4 marker holdback, strip-think,
        // byte-level stop_at); see crates/engine/src/eos_filter.rs.
        let mut bytes_fed_to_filter = 0usize;
        let mut filter =
            chat_output_filter_from_profile(chat_template_profile.as_ref(), request_stop_sequences);

        for _ in 0..max_tokens {
            generated += 1;
            m.active.cursor.conversation_tokens.push(next_token);
            streamed_tokens.push(next_token);
            emit_committed_event(
                stdout,
                id,
                next_token,
                streamed_tokens.len() - 1,
                t0.elapsed().as_millis() as u64,
            );
            let all_bytes = tokenizer.decode_bytes(&streamed_tokens);
            let new_bytes = &all_bytes[bytes_fed_to_filter..];
            bytes_fed_to_filter = all_bytes.len();
            let filter_stop = emit_filter_action(stdout, id, filter.observe(new_bytes));

            // Scope repeat_buf to this turn's prompt + generated tokens
            // (same logic as the Qwen3.5 path: prompt anchor + current turn).
            let rw = repeat_window.min(64);
            let scope_start = ngram_scope_start_llama
                .max(m.active.cursor.conversation_tokens.len().saturating_sub(rw));
            let hist_slice = &m.active.cursor.conversation_tokens[scope_start..];
            let hist_bytes: Vec<u8> = hist_slice.iter().flat_map(|t| t.to_ne_bytes()).collect();
            gpu.hip
                .memcpy_htod(&scratch.repeat_buf.buf, &hist_bytes)
                .unwrap();

            // Write K/V for this token FIRST so the next turn's context is
            // always fully populated. The sampled next_token from this call
            // is discarded when we break on im_end/eos — wasteful by one
            // launch but avoids a KV cache gap at the terminator.
            let pos = m.active.cursor.seq_pos + generated - 1;
            let (tok, rng) = llama::forward_scratch(
                gpu,
                weights,
                config,
                next_token,
                pos,
                kv,
                scratch,
                temp,
                top_p,
                rng_state,
                hist_slice.len(),
                repeat_penalty,
            )
            .unwrap();

            if next_token == config.eos_token {
                break;
            }
            if filter_stop {
                break;
            }
            if im_end_token == Some(next_token) {
                break;
            }
            if tokenizer.is_terminator(next_token) {
                break;
            }

            next_token = tok;
            rng_state = rng;
        }
        m.active.cursor.seq_pos += generated;

        // ChatML \n boundary — run through forward to keep KV cache in sync
        if im_end_token == Some(*m.active.cursor.conversation_tokens.last().unwrap_or(&0))
            && !nl.is_empty()
        {
            for &t in &nl {
                let (_, rng2) = llama::forward_scratch(
                    gpu,
                    weights,
                    config,
                    t,
                    m.active.cursor.seq_pos,
                    kv,
                    scratch,
                    temp,
                    top_p,
                    rng_state,
                    0,
                    1.0,
                )
                .unwrap();
                rng_state = rng2;
                m.active.cursor.seq_pos += 1;
                m.active.cursor.conversation_tokens.push(t);
            }
        }

        let t_end = Instant::now();
        let total_s = t_end.duration_since(t0).as_secs_f64();
        let prefill_s = t_prefill.duration_since(t0).as_secs_f64();
        let decode_s = t_end.duration_since(t_prefill).as_secs_f64();
        let tok_s = if total_s > 0.0 {
            generated as f64 / total_s
        } else {
            0.0
        };
        let prefill_tok_s = if prefill_s > 0.0 {
            prefill_tokens as f64 / prefill_s
        } else {
            0.0
        };
        let decode_tok_s = if decode_s > 0.0 {
            generated as f64 / decode_s
        } else {
            0.0
        };
        if let Some(dir) = evidence_dir {
            write_daemon_runtime_oneshot_evidence(
                dir,
                m,
                gpu,
                id,
                prefill_tokens,
                generated,
                prefill_s,
                decode_s,
                prefill_s * 1000.0,
            );
        }
        let _ = writeln!(
            stdout,
            r#"{{"type":"done","id":"{}","tokens":{},"tok_s":{:.1},"prefill_tokens":{},"prefill_ms":{:.1},"prefill_tok_s":{:.1},"decode_tok_s":{:.1},"ttft_ms":{:.1}{}}}"#,
            id,
            generated,
            tok_s,
            prefill_tokens,
            prefill_s * 1000.0,
            prefill_tok_s,
            decode_tok_s,
            prefill_s * 1000.0,
            pflash_done_fragment(&pflash_summary, &pflash_bypass_reason, pflash_alpha),
        );
        let _ = stdout.flush();
    }
    Qwen35Start::Handled
}

#[allow(clippy::too_many_arguments)]
/// The original entry point, now a thin driver over [`generate_start`]: start,
/// march to completion, finish. Behaviour is unchanged for every caller.
pub fn generate(
    m: &mut LoadedModel,
    gpu: &mut hipfire_rdna::Gpu,
    drafter_gpu: Option<&mut hipfire_rdna::Gpu>,
    stdout: &mut dyn std::io::Write,
    id: &str,
    prompt: &str,
    system_prompt: Option<&str>,
    temp: f32,
    top_p: f32,
    top_k: usize,
    max_tokens: usize,
    repeat_penalty: f32,
    repeat_window: usize,
    presence_penalty: f32,
    frequency_penalty: f32,
    budget_alert_at_tok: usize,
    budget_alert_text: &str,
    max_think_tokens: usize,
    assistant_prefix: prompt_frame::AssistantPrefix,
    pflash_state: Option<&mut hipfire_arch_qwen35::pflash::PflashState>,
    pflash_cfg: Option<&hipfire_arch_qwen35::pflash::PflashConfig>,
    tools: Option<&[serde_json::Value]>,
    messages_history: Option<&[prompt_frame::Message]>,
    think_mode: ThinkMode,
    prefill_already_done: bool,
    prefilled_prompt_tokens: Option<usize>,
    request_stop_sequences: &[String],
    evidence_dir: Option<&str>,
    // Explicit per-request `"raw"` override; `None` = auto (raw iff the model
    // has no chat_template). Threaded rather than read from a global — see
    // `effective_raw` for the cross-request leak that motivated it.
    raw_override: Option<bool>,
) {
    match generate_start(
        m,
        gpu,
        drafter_gpu,
        stdout,
        id,
        prompt,
        system_prompt,
        temp,
        top_p,
        top_k,
        max_tokens,
        repeat_penalty,
        repeat_window,
        presence_penalty,
        frequency_penalty,
        budget_alert_at_tok,
        budget_alert_text,
        max_think_tokens,
        assistant_prefix,
        pflash_state,
        pflash_cfg,
        tools,
        messages_history,
        think_mode,
        prefill_already_done,
        prefilled_prompt_tokens,
        request_stop_sequences,
        evidence_dir,
        raw_override,
        // Legacy in-crate driver: no stream identity to derive a seed from, so
        // it keeps the process-level behaviour `initial_rng_state()` selects.
        None,
        // Same — no request identity here, so n-gram tables stay daemon-local.
        None,
    ) {
        Qwen35Start::Handled => {}
        Qwen35Start::Ready(mut generation) => {
            while generation.st.generated < generation.cfg.max_tokens {
                let tokenizer = m
                    .tokenizer
                    .as_ref()
                    .expect("qwen35 model always has a tokenizer");
                match generation.step(m, gpu, tokenizer, stdout, id) {
                    Qwen35Step::Continue => {}
                    Qwen35Step::Stop => break,
                    // The unwind stays out here: qwen35_restore_or_error consumes
                    // the session, so it cannot run while `step` holds borrows.
                    Qwen35Step::Failed(message) => {
                        write_error(stdout, id, &message);
                        if let Some(session) = generation.session.take() {
                            qwen35_restore_or_error(stdout, id, m, gpu, session);
                        }
                        return;
                    }
                }
            }
            generation.finish(m, gpu, stdout, id);
        }
    }
}
