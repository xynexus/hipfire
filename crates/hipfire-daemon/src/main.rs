// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! hipfire engine daemon — JSON lines over stdin/stdout.
//! The Rust server/CLI spawns this process and communicates via IPC.
//! Usage: daemon (reads JSON from stdin, writes JSON to stdout)
//!
//! Exactly one daemon runs at a time per machine — enforced by an exclusive
//! flock(2) on ~/.hipfire/daemon.pid. A second daemon invocation exits with
//! `FATAL: hipfire daemon already running (PID N)` before touching the GPU,
//! preventing orphan doubles from silently double-consuming VRAM. Startup also
//! takes per-resource leases in /tmp before HIP init so multi-worktree agents
//! contend on GPU/NPU/CPU resources through the runtime, not a shell helper.
//!
//! Protocol:
//!   → {"type":"load","model":"path.hfq","params":{"max_seq":4096}}
//!   ← {"type":"loaded","arch":"qwen3_5","dim":4096,"layers":32,"vocab":248320,"vl":true}
//!   → {"type":"generate","id":"r1","prompt":"Hello","temperature":0.3,"max_tokens":512}
//!   → {"type":"generate","id":"r1","prompt":"Describe this","image":"/path/to/img.png","temperature":0.3,"max_tokens":512}
//!   → {"type":"generate","id":"r1","prompt":"Describe this MRI series","video":"/path/scan.webm","max_frames":8}   (gemma3-vl / arch 13)
//!   ← {"type":"token","id":"r1","text":"The"}
//!   ← {"type":"done","id":"r1","tokens":42,"tok_s":44.5}
//!   → {"type":"unload"}
//!   ← {"type":"unloaded"}

use hipfire_arch_deepseek4 as deepseek4;
#[cfg(feature = "arch-lfm2moe")]
use hipfire_arch_lfm2moe as lfm2moe;
use hipfire_arch_minimax as minimax;
use hipfire_arch_qwen2::qwen2;
use hipfire_arch_qwen35::qwen35;
#[cfg(test)]
use hipfire_generate::validate_qwen35_fused_dense_prefill_batch_preflight;
use hipfire_generate::{
    validate_generate_batch_decode, validate_generate_batch_prefill,
    validate_prefix_hash_preflight, GenerateVLParams, ImageSource,
};
use hipfire_model::{
    build_local_llm_registry, is_qwen35_family_arch_id, ARCH_ID_DEEPSEEK4_FLASH, ARCH_ID_DOTS_OCR,
    ARCH_ID_EMBEDDINGGEMMA, ARCH_ID_GEMMA3_VL, ARCH_ID_LFM2_MOE, ARCH_ID_MINIMAX_M2, ARCH_ID_QWEN2,
};
use hipfire_prompt as prompt_frame;
use hipfire_state::{
    described_sequence_state_json, model_worker_runtime_view_json,
    parse_describe_sequence_state_request, parse_release_sequence_state_request,
    parse_release_sessions_request, parse_reserve_session_state_request,
    parse_unload_worker_request, parsed_handle_may_target_generic, release_sessions_done_json,
    release_state_done_json, reserve_session_state_done_json, reserve_session_state_rejected_json,
    sequence_state_reservation_plan, sequence_state_reservation_plan_for_reserved_bytes,
    session_state_reservation_describe_json, unload_worker_done_json, GenericSequenceStateArena,
};
#[cfg(test)]
use hipfire_state::{
    generic_state_reservation_descriptors, parse_reserve_session_state_kinds,
    parse_sequence_state_handle, sequence_state_handle_id, sequence_state_handle_parts,
    sequence_state_page_descriptor_json, SequenceStateHandle,
};
use std::io::{BufRead, Write};
use std::time::Instant;

// These modules now live in `hipfire-serving-core` (workstream A0). Re-import
// them at the crate root so existing `crate::events` / `crate::model` /
// `crate::session` / … paths in the daemon's other modules keep resolving
// unchanged.
use dummy::{
    emit_dummy_generate_batch_prefill_ready, run_generate_batch_prefill_dummy, DummyModelState,
};
use events::{emit_error_with_id, write_error, MAX_BASE64_ENCODED_LEN};
use generate::*;
use generate_vl::{decode_vl_frames, generate_vl, generate_vl_dots_ocr, generate_vl_gemma3};
use hipfire_daemon_protocol::{DaemonRequest, EmbeddingInputType, EmbeddingVector, RerankResult};
#[cfg(feature = "arch-lfm2moe")]
use hipfire_serving_core::lfm2_prefill;
use hipfire_serving_core::{
    dummy, events, generate, generate_vl, load, model, output_filter, qwen35_decode,
    qwen35_prefill, request, session,
};
#[cfg(feature = "arch-lfm2moe")]
use lfm2_prefill::*;
use load::*;
use model::{CaskConfig, EmbeddingGemmaState, LoadedModel, RAW_OVERRIDE};
use output_filter::{normalize_daemon_prompt, normalize_request_stop_sequences};
use qwen35_decode::*;
use qwen35_prefill::*;
use request::ThinkMode;
use session::*;
// Rich session protocol (qwen35/lfm2) is dispatched through this trait
// (impl'd on `LoadedModel`) instead of a per-arch `if is_qwen35 {} else …` ladder.
use hipfire_runtime::arch::SessionServingBackend;

fn invalid_kld_ref(msg: impl Into<String>) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, msg.into())
}

/// Emit a `load_progress` frame on the framed stdout channel. Called by the
/// load-progress sink the `load` handler installs around `load_model`. Takes a
/// fresh `std::io::stdout()` lock (rather than the handler's local `stdout`) so
/// it can be a plain free fn invoked from the sink closure; loads run on this
/// thread, so this never races the handler's own writes. `phase` is a controlled
/// identifier (e.g. `"weights"`) — no JSON escaping needed.
fn emit_load_progress(current: u32, total: u32, phase: &str) {
    use std::io::Write;
    let mut out = std::io::stdout().lock();
    let _ = writeln!(
        out,
        r#"{{"type":"load_progress","current":{current},"total":{total},"phase":"{phase}"}}"#
    );
    let _ = out.flush();
}

fn embeddinggemma_parts<'a>(
    m: &'a LoadedModel,
    op: &str,
) -> Result<
    (
        &'a EmbeddingGemmaState,
        &'a hipfire_model::tokenizer::Tokenizer,
    ),
    String,
> {
    let state = m.embeddinggemma.as_ref().ok_or_else(|| {
        format!(
            "{op}: loaded model is arch_id={}, expected embeddinggemma arch_id=19",
            m.arch_id
        )
    })?;
    let tokenizer = m
        .tokenizer
        .as_ref()
        .ok_or_else(|| format!("{op}: loaded embeddinggemma model has no tokenizer"))?;
    Ok((state, tokenizer))
}

fn embeddinggemma_encode_prefixed(
    gpu: &mut hipfire_rdna::Gpu,
    state: &EmbeddingGemmaState,
    tokenizer: &hipfire_model::tokenizer::Tokenizer,
    texts: &[String],
    prefix: &str,
    dims: Option<usize>,
) -> Result<Vec<Vec<f32>>, String> {
    let tokenized = texts
        .iter()
        .map(|text| {
            if prefix.is_empty() {
                tokenizer.encode(text)
            } else {
                tokenizer.encode(&format!("{prefix}{text}"))
            }
        })
        .collect::<Vec<_>>();
    if let Some(metadata) = state.embedding_metadata.as_ref() {
        for (index, tokens) in tokenized.iter().enumerate() {
            metadata
                .sequence
                .bucket_for_len(tokens.len())
                .map_err(|error| format!("embedding input {index}: {error}"))?;
        }
    } else {
        for (index, tokens) in tokenized.iter().enumerate() {
            if tokens.len() > 2048 {
                return Err(format!(
                    "embedding input {index} has {} tokens; maximum supported length is 2048",
                    tokens.len()
                ));
            }
        }
    }
    let dims = state.config.resolve_dims(dims);
    hipfire_serving_core::pooling::embed_batch_embeddinggemma(gpu, state, &tokenized, dims)
}

fn qwen3_embedding_encode_prefixed(
    state: &hipfire_serving_core::qwen3_embedding::Qwen3EmbeddingState,
    tokenizer: &hipfire_model::tokenizer::Tokenizer,
    texts: &[String],
    prefix: &str,
    dims: Option<usize>,
) -> Result<Vec<Vec<f32>>, String> {
    let tokenized = texts
        .iter()
        .map(|text| {
            if prefix.is_empty() {
                tokenizer.encode(text)
            } else {
                tokenizer.encode(&format!("{prefix}{text}"))
            }
        })
        .collect::<Vec<_>>();
    let dimensions = dims.unwrap_or(state.metadata.output.native_dimensions);
    if dimensions != state.metadata.output.native_dimensions
        && !state
            .metadata
            .output
            .matryoshka_dimensions
            .contains(&dimensions)
    {
        return Err(format!(
            "unsupported embedding dimensions {dimensions}; native={} supported_matryoshka={:?}",
            state.metadata.output.native_dimensions, state.metadata.output.matryoshka_dimensions
        ));
    }
    let mut embeddings = state.encode_token_batches(&tokenized)?;
    if dimensions != state.metadata.output.native_dimensions {
        for embedding in &mut embeddings {
            embedding.truncate(dimensions);
            let norm = embedding
                .iter()
                .map(|value| value * value)
                .sum::<f32>()
                .sqrt();
            if norm > 0.0 {
                for value in embedding {
                    *value /= norm;
                }
            }
        }
    }
    Ok(embeddings)
}

fn embeddinggemma_embed(
    gpu: &mut hipfire_rdna::Gpu,
    m: &LoadedModel,
    texts: &[String],
    input_type: EmbeddingInputType,
    dims: Option<usize>,
) -> Result<Vec<EmbeddingVector>, String> {
    if let Some(state) = m.qwen3_embedding.as_ref() {
        let tokenizer = m
            .tokenizer
            .as_ref()
            .ok_or_else(|| "embed: loaded Qwen3 embedding model has no tokenizer".to_string())?;
        let prefix = state.metadata.prompt(input_type);
        let embeddings = qwen3_embedding_encode_prefixed(state, tokenizer, texts, prefix, dims)?;
        return Ok(embeddings
            .into_iter()
            .enumerate()
            .map(|(index, embedding)| EmbeddingVector { index, embedding })
            .collect());
    }
    let (state, tokenizer) = embeddinggemma_parts(m, "embed")?;
    let prefix = state
        .embedding_metadata
        .as_ref()
        .map(|metadata| metadata.prompt(input_type))
        .unwrap_or_else(|| match input_type {
            EmbeddingInputType::Query => &state.config.query_prompt,
            EmbeddingInputType::Document => &state.config.document_prompt,
        });
    let embeddings = embeddinggemma_encode_prefixed(gpu, state, tokenizer, texts, prefix, dims)?;
    Ok(embeddings
        .into_iter()
        .enumerate()
        .map(|(index, embedding)| EmbeddingVector { index, embedding })
        .collect())
}

fn embeddinggemma_rerank(
    gpu: &mut hipfire_rdna::Gpu,
    m: &LoadedModel,
    query: &str,
    documents: &[String],
) -> Result<Vec<RerankResult>, String> {
    if let Some(state) = m.qwen3_embedding.as_ref() {
        let tokenizer = m
            .tokenizer
            .as_ref()
            .ok_or_else(|| "rerank: loaded Qwen3 embedding model has no tokenizer".to_string())?;
        let query_texts = vec![query.to_string()];
        let query_embedding = qwen3_embedding_encode_prefixed(
            state,
            tokenizer,
            &query_texts,
            state
                .metadata
                .prompt(hipfire_model::embedding::EmbeddingInputType::Query),
            None,
        )?
        .into_iter()
        .next()
        .ok_or_else(|| "rerank: query produced no embedding".to_string())?;
        let document_embeddings = qwen3_embedding_encode_prefixed(
            state,
            tokenizer,
            documents,
            state
                .metadata
                .prompt(hipfire_model::embedding::EmbeddingInputType::Document),
            None,
        )?;
        return Ok(hipfire_serving_core::pooling::rank_by_cosine(
            &query_embedding,
            &document_embeddings,
        )
        .into_iter()
        .map(|(index, relevance_score)| RerankResult {
            index,
            relevance_score,
        })
        .collect());
    }
    let (state, tokenizer) = embeddinggemma_parts(m, "rerank")?;
    let query_texts = vec![query.to_string()];
    let query_embedding = embeddinggemma_encode_prefixed(
        gpu,
        state,
        tokenizer,
        &query_texts,
        &state.config.query_prompt,
        None,
    )?
    .into_iter()
    .next()
    .ok_or_else(|| "rerank: query produced no embedding".to_string())?;
    let doc_embeddings = embeddinggemma_encode_prefixed(
        gpu,
        state,
        tokenizer,
        documents,
        &state.config.document_prompt,
        None,
    )?;
    Ok(
        hipfire_serving_core::pooling::rank_by_cosine(&query_embedding, &doc_embeddings)
            .into_iter()
            .map(|(index, relevance_score)| RerankResult {
                index,
                relevance_score,
            })
            .collect(),
    )
}

fn json_u64(meta: &serde_json::Value, key: &str) -> std::io::Result<u64> {
    meta.get(key)
        .and_then(|v| v.as_u64())
        .ok_or_else(|| invalid_kld_ref(format!("HFQM kldref metadata missing integer {key}")))
}

fn json_usize(meta: &serde_json::Value, key: &str) -> std::io::Result<usize> {
    json_u64(meta, key).map(|v| v as usize)
}

fn json_string(meta: &serde_json::Value, key: &str) -> std::io::Result<String> {
    meta.get(key)
        .and_then(|v| v.as_str())
        .map(str::to_string)
        .ok_or_else(|| invalid_kld_ref(format!("HFQM kldref metadata missing string {key}")))
}

fn json_opt_string(meta: &serde_json::Value, key: &str) -> Option<String> {
    meta.get(key).and_then(|v| v.as_str()).map(str::to_string)
}

fn json_opt_bool(meta: &serde_json::Value, key: &str) -> Option<bool> {
    meta.get(key).and_then(|v| v.as_bool())
}

fn json_opt_usize(meta: &serde_json::Value, key: &str) -> Option<usize> {
    meta.get(key).and_then(|v| {
        v.as_u64()
            .or_else(|| v.as_str().and_then(|s| s.parse::<u64>().ok()))
            .map(|n| n as usize)
    })
}

struct ScopedEnvVar {
    key: &'static str,
    previous: Option<std::ffi::OsString>,
}

impl ScopedEnvVar {
    fn set(key: &'static str, value: impl AsRef<std::ffi::OsStr>) -> Self {
        let previous = std::env::var_os(key);
        std::env::set_var(key, value);
        Self { key, previous }
    }
}

impl Drop for ScopedEnvVar {
    fn drop(&mut self) {
        if let Some(previous) = &self.previous {
            std::env::set_var(self.key, previous);
        } else {
            std::env::remove_var(self.key);
        }
    }
}

fn qwen_residency_load_env(params: Option<&hipfire_model::ModelLoadParams>) -> Vec<ScopedEnvVar> {
    let mut guards = Vec::new();
    let Some(params) = params else {
        return guards;
    };
    if let Some(mode) = params
        .residency_mode
        .as_deref()
        .filter(|mode| !mode.is_empty())
    {
        guards.push(ScopedEnvVar::set("HIPFIRE_QWEN35_RESIDENCY_MODE", mode));
    }
    if let Some(bytes) = params.module_vram_budget_bytes.filter(|bytes| *bytes > 0) {
        guards.push(ScopedEnvVar::set(
            "HIPFIRE_QWEN35_EXPERT_CACHE_BYTES",
            bytes.to_string(),
        ));
    }
    guards
}

const SYSTEM_RESERVATION_CHUNK_BYTES: usize = 64 * 1024 * 1024;
const VRAM_RESERVATION_CHUNK_BYTES: usize = 512 * 1024 * 1024;

#[derive(Clone, Debug, Default)]
struct ResidentResourceUsage {
    model_path: String,
    system_memory_bytes: u64,
    vram_bytes: u64,
    residency_mode: String,
}

struct ResourceReservationManager {
    system_memory_budget_bytes: u64,
    system_memory_headroom_bytes: u64,
    vram_budget_bytes: u64,
    vram_headroom_bytes: u64,
    system_chunks: Vec<Vec<u8>>,
    vram_chunks: Vec<hip_bridge::DeviceBuffer>,
    resident_usage: std::collections::HashMap<String, ResidentResourceUsage>,
}

impl ResourceReservationManager {
    fn from_env() -> Self {
        Self::from_env_reader(|key| std::env::var(key).ok())
    }

    fn from_env_reader(mut get: impl FnMut(&str) -> Option<String>) -> Self {
        Self {
            system_memory_budget_bytes: parse_env_u64(get(
                "HIPFIRE_SCHEDULER_SYSTEM_MEMORY_BUDGET_BYTES",
            )),
            system_memory_headroom_bytes: parse_env_u64(get(
                "HIPFIRE_SCHEDULER_SYSTEM_MEMORY_HEADROOM_BYTES",
            )),
            vram_budget_bytes: parse_env_u64(get("HIPFIRE_SCHEDULER_VRAM_BUDGET_BYTES")),
            vram_headroom_bytes: parse_env_u64(get("HIPFIRE_SCHEDULER_VRAM_HEADROOM_BYTES")),
            system_chunks: Vec::new(),
            vram_chunks: Vec::new(),
            resident_usage: std::collections::HashMap::new(),
        }
    }

    fn system_target_bytes(&self) -> u64 {
        reservation_target_bytes(
            self.system_memory_budget_bytes,
            self.system_memory_headroom_bytes,
        )
    }

    fn vram_target_bytes(&self) -> u64 {
        reservation_target_bytes(self.vram_budget_bytes, self.vram_headroom_bytes)
    }

    fn resident_system_memory_bytes(&self) -> u64 {
        self.resident_usage
            .values()
            .map(|usage| usage.system_memory_bytes)
            .sum()
    }

    fn resident_vram_bytes(&self) -> u64 {
        self.resident_usage
            .values()
            .map(|usage| usage.vram_bytes)
            .sum()
    }

    fn held_system_memory_placeholder_bytes(&self) -> u64 {
        self.system_chunks
            .iter()
            .map(|chunk| chunk.len() as u64)
            .sum()
    }

    fn held_vram_placeholder_bytes(&self) -> u64 {
        self.vram_chunks
            .iter()
            .map(|chunk| chunk.size() as u64)
            .sum()
    }

    fn planned_usage_for_load(
        &self,
        path: &str,
        params: Option<&hipfire_model::ModelLoadParams>,
    ) -> ResidentResourceUsage {
        let residency_mode = params
            .and_then(|params| params.residency_mode.as_deref())
            .filter(|mode| !mode.is_empty())
            .unwrap_or("full")
            .to_string();
        let file_bytes = std::fs::metadata(path)
            .map(|metadata| metadata.len())
            .unwrap_or(0);
        let module_budget = params
            .and_then(|params| params.module_vram_budget_bytes)
            .filter(|bytes| *bytes > 0);
        let vram_bytes = if residency_mode == "qwen_moe_modules" {
            module_budget.unwrap_or(file_bytes)
        } else {
            file_bytes
        };
        ResidentResourceUsage {
            model_path: path.to_string(),
            system_memory_bytes: 0,
            vram_bytes,
            residency_mode,
        }
    }

    fn set_worker_usage(&mut self, worker_id: impl Into<String>, usage: ResidentResourceUsage) {
        self.resident_usage.insert(worker_id.into(), usage);
    }

    fn remove_worker(&mut self, worker_id: &str) {
        self.resident_usage.remove(worker_id);
    }

    fn clear_workers(&mut self) {
        self.resident_usage.clear();
    }

    fn release_placeholders(&mut self, gpu: &mut hipfire_rdna::Gpu) -> Result<(), String> {
        self.system_chunks.clear();
        let mut errors = Vec::new();
        for chunk in self.vram_chunks.drain(..) {
            if let Err(err) = gpu.hip.free(chunk) {
                errors.push(err.to_string());
            }
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(format!(
                "failed to release {} VRAM reservation chunk(s): {}",
                errors.len(),
                errors.join("; ")
            ))
        }
    }

    fn reacquire_placeholders(&mut self, gpu: &mut hipfire_rdna::Gpu) -> Result<(), String> {
        self.release_placeholders(gpu)?;
        let system_target = self
            .system_target_bytes()
            .saturating_sub(self.resident_system_memory_bytes());
        let vram_target = self
            .vram_target_bytes()
            .saturating_sub(self.resident_vram_bytes());
        self.allocate_system_placeholders(system_target)?;
        self.allocate_vram_placeholders(gpu, vram_target)?;
        Ok(())
    }

    fn allocate_system_placeholders(&mut self, mut bytes: u64) -> Result<(), String> {
        while bytes > 0 {
            let chunk_len = bytes.min(SYSTEM_RESERVATION_CHUNK_BYTES as u64);
            let chunk_len = usize::try_from(chunk_len).map_err(|_| {
                format!("system reservation chunk {chunk_len} exceeds addressable usize")
            })?;
            let mut chunk = vec![0u8; chunk_len];
            touch_system_memory(&mut chunk);
            self.system_chunks.push(chunk);
            bytes = bytes.saturating_sub(chunk_len as u64);
        }
        Ok(())
    }

    fn allocate_vram_placeholders(
        &mut self,
        gpu: &mut hipfire_rdna::Gpu,
        mut bytes: u64,
    ) -> Result<(), String> {
        while bytes > 0 {
            let chunk_len = bytes.min(VRAM_RESERVATION_CHUNK_BYTES as u64);
            let chunk_len = usize::try_from(chunk_len)
                .map_err(|_| format!("VRAM reservation chunk {chunk_len} exceeds usize"))?;
            let chunk = gpu
                .hip
                .malloc(chunk_len)
                .map_err(|err| format!("hipMalloc reservation {chunk_len} bytes: {err}"))?;
            if let Err(err) = gpu.hip.memset(&chunk, 0, chunk_len) {
                let _ = gpu.hip.free(chunk);
                return Err(format!(
                    "hipMemset reservation {chunk_len} bytes failed: {err}"
                ));
            }
            self.vram_chunks.push(chunk);
            bytes = bytes.saturating_sub(chunk_len as u64);
        }
        Ok(())
    }

    fn status_json(&self) -> serde_json::Value {
        let mut workers = self
            .resident_usage
            .iter()
            .map(|(worker_id, usage)| {
                serde_json::json!({
                    "worker_key_id": worker_id,
                    "model_path": usage.model_path,
                    "residency_mode": usage.residency_mode,
                    "system_memory_bytes": usage.system_memory_bytes,
                    "vram_bytes": usage.vram_bytes,
                })
            })
            .collect::<Vec<_>>();
        workers.sort_by(|a, b| {
            a.get("worker_key_id")
                .and_then(|value| value.as_str())
                .cmp(&b.get("worker_key_id").and_then(|value| value.as_str()))
        });
        serde_json::json!({
            "type": "resource_status",
            "system_memory_budget_bytes": self.system_memory_budget_bytes,
            "system_memory_headroom_bytes": self.system_memory_headroom_bytes,
            "system_memory_target_bytes": self.system_target_bytes(),
            "held_system_memory_placeholder_bytes": self.held_system_memory_placeholder_bytes(),
            "resident_system_memory_bytes": self.resident_system_memory_bytes(),
            "vram_budget_bytes": self.vram_budget_bytes,
            "vram_headroom_bytes": self.vram_headroom_bytes,
            "vram_target_bytes": self.vram_target_bytes(),
            "held_vram_placeholder_bytes": self.held_vram_placeholder_bytes(),
            "resident_vram_bytes": self.resident_vram_bytes(),
            "resident_workers": workers.len(),
            "workers": workers,
        })
    }
}

fn parse_env_u64(value: Option<String>) -> u64 {
    value
        .as_deref()
        .and_then(|value| value.trim().parse::<u64>().ok())
        .unwrap_or(0)
}

fn reservation_target_bytes(budget_bytes: u64, headroom_bytes: u64) -> u64 {
    budget_bytes.saturating_sub(headroom_bytes)
}

fn touch_system_memory(bytes: &mut [u8]) {
    if bytes.is_empty() {
        return;
    }
    let mut offset = 0usize;
    while offset < bytes.len() {
        bytes[offset] = bytes[offset].wrapping_add(1);
        offset = offset.saturating_add(4096);
    }
    let last = bytes.len() - 1;
    bytes[last] = bytes[last].wrapping_add(1);
}

fn reset_has_no_resident_model(
    dummy_model: &Option<DummyModelState>,
    model: &Option<LoadedModel>,
    resident_models: &std::collections::HashMap<String, LoadedModel>,
) -> bool {
    dummy_model.is_none() && model.is_none() && resident_models.is_empty()
}

fn reset_target_worker_id(msg: &serde_json::Value, active_worker_id: &str) -> String {
    if hipfire_model::has_worker_or_model_identity(msg) {
        message_worker_id(msg)
    } else {
        active_worker_id.to_string()
    }
}

fn le_u32_vec(bytes: &[u8], name: &str, expected: usize) -> std::io::Result<Vec<u32>> {
    if bytes.len() != expected * 4 {
        return Err(invalid_kld_ref(format!(
            "HFQM kldref {name} byte length {} != expected {}",
            bytes.len(),
            expected * 4
        )));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect())
}

fn le_f32_vec(bytes: &[u8], name: &str, expected: usize) -> std::io::Result<Vec<f32>> {
    if bytes.len() != expected * 4 {
        return Err(invalid_kld_ref(format!(
            "HFQM kldref {name} byte length {} != expected {}",
            bytes.len(),
            expected * 4
        )));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect())
}

fn hfqm_blob<'a>(
    package: &'a hipfire_runtime::hfq::HfqPackage,
    name: &str,
) -> std::io::Result<&'a [u8]> {
    package
        .blob_data(name)
        .ok_or_else(|| invalid_kld_ref(format!("HFQM kldref missing payload {name}")))
}

fn read_kld_ref_archive(path: &std::path::Path) -> std::io::Result<hipfire_kld::RefArchive> {
    let mut magic = [0u8; 4];
    let mut f = std::fs::File::open(path)?;
    use std::io::Read;
    let n = f.read(&mut magic)?;
    if n == 4 && &magic == hipfire_runtime::hfq::HFQM_MAGIC {
        return read_hfqm_kld_ref_archive(path);
    }
    hipfire_kld::RefArchive::read_file(path)
}

fn read_hfqm_kld_ref_archive(path: &std::path::Path) -> std::io::Result<hipfire_kld::RefArchive> {
    let package = hipfire_runtime::hfq::HfqPackage::open(path)?;
    if package.arch_id == hipfire_runtime::hfq::HFQM_ARCH_NON_WEIGHT_PACKAGE {
        return Err(invalid_kld_ref(
            "HFQM kldref arch_id is 0; regenerate the ref with the parent model arch_id",
        ));
    }
    let meta_json: serde_json::Value = serde_json::from_str(&package.metadata_json)
        .map_err(|e| invalid_kld_ref(format!("HFQM kldref metadata json: {e}")))?;
    if meta_json.get("artifact_kind").and_then(|v| v.as_str()) != Some("hipfire.kldref") {
        return Err(invalid_kld_ref(
            "HFQM package is not artifact_kind=hipfire.kldref",
        ));
    }
    if meta_json.get("package_schema").and_then(|v| v.as_str()) != Some("hipfire.kldref.v1") {
        return Err(invalid_kld_ref(
            "HFQM kldref package_schema is not hipfire.kldref.v1",
        ));
    }
    if let Some(meta_arch) = meta_json.get("arch_id").and_then(|v| v.as_u64()) {
        if meta_arch as u32 != package.arch_id {
            return Err(invalid_kld_ref(format!(
                "HFQM kldref metadata arch_id {} != header arch_id {}",
                meta_arch, package.arch_id
            )));
        }
    }

    let n_ctx = json_usize(&meta_json, "n_ctx")?;
    let n_vocab = json_usize(&meta_json, "n_vocab")?;
    let n_chunk = json_usize(&meta_json, "n_chunk")?;
    let scored_per_chunk = json_usize(&meta_json, "scored_per_chunk")?;
    let scoring_start = json_usize(&meta_json, "scoring_start")?;
    let top_k = json_usize(&meta_json, "top_k")?;
    let total_scored = json_usize(&meta_json, "total_scored")?;

    let tokens = le_u32_vec(
        hfqm_blob(&package, "kldref.tokens")?,
        "kldref.tokens",
        n_chunk * n_ctx,
    )?;
    let top_count = n_chunk * scored_per_chunk * top_k;
    let top_indices = le_u32_vec(
        hfqm_blob(&package, "kldref.top_indices")?,
        "kldref.top_indices",
        top_count,
    )?;
    let top_log_probs = le_f32_vec(
        hfqm_blob(&package, "kldref.top_log_probs")?,
        "kldref.top_log_probs",
        top_count,
    )?;
    let residual_mass = le_f32_vec(
        hfqm_blob(&package, "kldref.residual_mass")?,
        "kldref.residual_mass",
        n_chunk * scored_per_chunk,
    )?;

    let mut cfg = hipfire_kld::KldConfig::default();
    cfg.top_k = top_k;
    if let Some(kv_mode) = json_opt_string(&meta_json, "kv_mode") {
        cfg.kv_mode = kv_mode.to_ascii_lowercase();
    }
    if let Some(graph) = json_opt_bool(&meta_json, "kld_graph_prefill") {
        cfg.graph = graph;
    }
    cfg.prefill_max_batch = json_opt_usize(&meta_json, "kld_graph_prefill_max_batch")
        .or_else(|| json_opt_usize(&meta_json, "prefill_max_batch"))
        .filter(|&v| v >= 2);

    Ok(hipfire_kld::RefArchive {
        meta: hipfire_kld::RefMeta {
            schema: json_opt_usize(&meta_json, "schema").unwrap_or(1) as u32,
            base_model_id: json_string(&meta_json, "base_model_id")?,
            source_model_sha256: json_string(&meta_json, "source_model_sha256")?,
            tokenizer_sha256: json_opt_string(&meta_json, "tokenizer_sha256"),
            arch_id: package.arch_id,
            n_vocab,
            n_ctx,
            n_chunk,
            scored_per_chunk,
            scoring_start,
            top_k,
            total_scored,
            slice_path: json_string(&meta_json, "slice")?,
            slice_md5: json_string(&meta_json, "slice_md5")?,
            config: cfg,
            producer: hipfire_kld::ProducerInfo {
                hipfire_version: json_opt_string(&meta_json, "hipfire_version").unwrap_or_default(),
                git_commit: json_opt_string(&meta_json, "git_commit"),
                git_describe: json_opt_string(&meta_json, "git_describe"),
                git_dirty: json_opt_bool(&meta_json, "git_dirty"),
                gpu_arch: json_opt_string(&meta_json, "gpu_arch").unwrap_or_default(),
                producer_cmd: json_opt_string(&meta_json, "producer_cmd"),
            },
            payload_codecs: Default::default(),
            content_sha256: None,
        },
        tokens,
        top_indices,
        top_log_probs,
        residual_mass,
    })
}

/// Acquire a machine-wide exclusive lock on ~/.hipfire/daemon.pid.
///
/// On Unix: flock(2) is the kernel-level lock. The kernel releases it
/// automatically on process death (including SIGKILL), so no manual
/// cleanup is required — stale PID file contents are fine, the fd is
/// what holds the lock.
///
/// On Windows: no kernel-level lock; we write the PID file but don't
/// guarantee single-instance semantics. A second daemon launch may
/// silently overwrite the PID. This matches the v0.1.0-alpha Windows
/// behavior; tightening it is tracked in a follow-up.
///
/// Returns the [`FlockGuard`]; caller MUST keep it alive for the process
/// lifetime (dropping it closes the fd and releases the lock).
fn acquire_daemon_lock() -> hipfire_lock::FlockGuard {
    #[cfg(unix)]
    let home = std::env::var("HOME").expect("HOME environment variable not set");
    #[cfg(windows)]
    let home = std::env::var("USERPROFILE").expect("USERPROFILE environment variable not set");

    let pid_path = std::path::PathBuf::from(home)
        .join(".hipfire")
        .join("daemon.pid");

    let mut guard =
        hipfire_lock::FlockGuard::open(&pid_path).expect("failed to open ~/.hipfire/daemon.pid");
    match guard.try_lock() {
        Ok(true) => {}
        Ok(false) => {
            // Already held: surface the holder PID (written below by the live
            // daemon) in the fatal message.
            let holder = guard.holder().unwrap_or_default();
            let pid = holder.trim();
            let pid_display = if pid.is_empty() { "<unknown>" } else { pid };
            let kill_arg = if pid.is_empty() { "<pid>" } else { pid };
            hipfire_daemon_adapter::fatal_startup_error(
                &format!(
                    "hipfire daemon already running (PID {pid_display}). Run `kill {kill_arg}` and retry."
                ),
                None,
            );
        }
        Err(e) => panic!("failed to flock ~/.hipfire/daemon.pid: {e}"),
    }

    // Got the lock. Record our PID so the fatal message above (in a second
    // daemon) and external tooling can show a useful number. `flock` is on the
    // open fd, so rewriting the contents doesn't drop the lock.
    let _ = guard.write_holder(&std::process::id().to_string());
    guard
}

#[cfg(test)]
mod resource_reservation_tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn resource_reservation_env_applies_budget_headroom_targets() {
        let values = HashMap::from([
            (
                "HIPFIRE_SCHEDULER_SYSTEM_MEMORY_BUDGET_BYTES",
                "4096".to_string(),
            ),
            (
                "HIPFIRE_SCHEDULER_SYSTEM_MEMORY_HEADROOM_BYTES",
                "512".to_string(),
            ),
            ("HIPFIRE_SCHEDULER_VRAM_BUDGET_BYTES", "8192".to_string()),
            ("HIPFIRE_SCHEDULER_VRAM_HEADROOM_BYTES", "1024".to_string()),
        ]);
        let manager = ResourceReservationManager::from_env_reader(|key| values.get(key).cloned());

        assert_eq!(manager.system_target_bytes(), 3584);
        assert_eq!(manager.vram_target_bytes(), 7168);
        assert_eq!(
            manager.status_json()["held_system_memory_placeholder_bytes"],
            0
        );
    }

    #[test]
    fn resource_reservation_usage_prefers_module_budget_for_qwen_moe_modules() {
        let dir = std::env::temp_dir().join(format!(
            "hipfire-resource-reservation-test-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let model_path = dir.join("Qwen3.5-122B-A10B.mq4.hfq");
        std::fs::write(&model_path, vec![0u8; 1234]).unwrap();

        let manager = ResourceReservationManager::from_env_reader(|_| None);
        let params = hipfire_model::ModelLoadParams {
            residency_mode: Some("qwen_moe_modules".to_string()),
            module_vram_budget_bytes: Some(256),
            ..Default::default()
        };
        let usage = manager.planned_usage_for_load(model_path.to_str().unwrap(), Some(&params));

        assert_eq!(usage.residency_mode, "qwen_moe_modules");
        assert_eq!(usage.vram_bytes, 256);

        let _ = std::fs::remove_file(&model_path);
        let _ = std::fs::remove_dir(&dir);
    }

    #[test]
    fn reset_without_resident_model_is_idempotent() {
        let dummy_model = None;
        let model = None;
        let resident_models = HashMap::new();

        assert!(reset_has_no_resident_model(
            &dummy_model,
            &model,
            &resident_models
        ));
    }

    #[test]
    fn bare_reset_targets_active_worker() {
        let msg = serde_json::json!({"type": "reset"});
        assert_eq!(
            reset_target_worker_id(&msg, "server-model:active"),
            "server-model:active"
        );
    }

    #[test]
    fn explicit_reset_worker_overrides_active_worker() {
        let msg = serde_json::json!({"type": "reset", "worker_key_id": "worker-a"});
        assert_eq!(
            reset_target_worker_id(&msg, "server-model:active"),
            "worker-a"
        );
    }
}

#[cfg(test)]
mod generate_batch_prefill_tests;

/// Print a friendly, user-actionable message when Gpu::init fails. Matches
/// the panic shape we used to emit (which dumped a Rust backtrace and the
/// raw HipError debug-format) but turns it into a concrete next-step list.
/// The most common cause on Windows (#112) is HIP SDK present but no
/// AMD GPU driver visible to the runtime; on Linux it is usually missing
/// `libamdhip64.so` or kernel-side amdgpu / kfd not loaded.
fn report_gpu_init_failure(err: &hip_bridge::HipError) {
    eprintln!();
    eprintln!("hipfire: failed to initialize GPU runtime.");
    eprintln!("  HIP error: {} (code {})", err.message, err.code);
    eprintln!();
    if cfg!(target_os = "windows") {
        eprintln!("  Most common Windows cause: HIP SDK is loaded but no");
        eprintln!("  AMD GPU is visible to the runtime. Verify:");
        eprintln!("    1. AMD Adrenalin driver is installed and current.");
        eprintln!("    2. AMD HIP SDK 6.2 or newer is installed:");
        eprintln!("       https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html");
        eprintln!("    3. `amdhip64.dll` is reachable (HIP_PATH set or DLL on PATH).");
        eprintln!("    4. Reboot after driver / SDK install if you have not yet.");
    } else {
        eprintln!("  Most common Linux causes:");
        eprintln!("    1. amdgpu kernel module not loaded (check `lsmod | grep amdgpu`).");
        eprintln!("    2. /dev/kfd missing or not readable by the current user");
        eprintln!("       (add to the `render` group; reboot).");
        eprintln!("    3. ROCm not installed or libamdhip64.so missing");
        eprintln!("       (check `ldconfig -p | grep amdhip64`).");
    }
    eprintln!();
    eprintln!("  Run `hipfire diag` for a full environment report.");
}

/// Resident state of a micro-step-preemptible `train_lora` run. The daemon runs
/// one `quantum` of steps per `TrainLora` request and keeps this alive between
/// requests (keyed by `run_id`); the runner re-enqueues the training lease each
/// quantum so lower-priority training time-slices with interactive serving.
struct LoraTrainSession {
    run_id: String,
    model: hipfire_train::model::LlamaModel,
    opt: hipfire_train::optim::AdamW,
    batch: Vec<(Vec<u32>, Vec<f32>)>,
    pos: Vec<f32>,
    target_tokens: f32,
    step: usize,
    total: usize,
    initial_ce: f32,
    last_ce: f32,
    output: String,
    vocab: usize,
}

/// Resident state of a micro-step-preemptible `train_drafter` run. The daemon
/// runs one `quantum` of EPOCHS per `TrainDrafter` request and keeps this alive
/// between requests (keyed by `run_id`); the runner re-enqueues the training
/// lease each quantum so drafter training time-slices with interactive serving.
/// `embed` is moved into `drafter`, so we hold the label tensors still needed
/// across quanta (chunks + mid labels; base_shallow was consumed by init for the
/// `bar` baseline) rather than the whole LabelSet.
struct DrafterTrainSession {
    run_id: String,
    drafter: hipfire_train::ssm_drafter::SsmDrafter,
    chunks: Vec<Vec<u32>>,
    label_mid: Vec<Vec<f32>>,
    cfg: hipfire_train::train_loop::TrainCfg,
    st: hipfire_train::train_loop::DrafterLoopState,
    output: String,
    quantum: usize,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.iter().any(|a| a == "--version" || a == "-V") {
        println!("hipfire-daemon {}", hipfire_build_info::VERSION);
        return;
    }

    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!(
            "Usage: daemon [options]\n\
             \n\
             Reads JSON requests from stdin and writes JSON events to stdout.\n\
             \n\
             Options:\n\
               --precompile        compile/cache kernels for the current GPU and exit\n\
               --version, -V       print the build version and exit\n\
               --help, -h          print this help"
        );
        return;
    }

    // --precompile: compile all kernels for this GPU, write hash files, exit.
    // Used by install.sh and `hipfire update` so first `hipfire chat`
    // isn't a 2-minute hipcc wait.
    //
    // Covers the current default path (mq4 weights + asym3 KV) plus the legacy
    // compat paths (hfq4, hfq6, q8 weights × asym3, q8 KV) so models from any
    // era of the registry start instantly.
    if args.iter().any(|a| a == "--precompile") {
        // Pre-create the expected precompiled-dir next to this binary so the
        // compiler's writeback path fires. Without this, Gpu::init probes for
        // an existing dir and silently disables writeback if it's missing —
        // meaning fresh installs would compile but never cache cross-invocation.
        if let Some(exe_dir) = std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(|d| d.to_path_buf()))
        {
            // Arch is unknown until Gpu::init; use a broad mkdir for the common arches
            // we support so the probe picks one up. The real arch check after init
            // will log the active dir.
            for arch in [
                "gfx906", "gfx1010", "gfx1013", "gfx1030", "gfx1031", "gfx1100", "gfx1101",
                "gfx1102", "gfx1103", "gfx1151", "gfx1152", "gfx1200", "gfx1201",
            ] {
                let _ =
                    std::fs::create_dir_all(exe_dir.join("kernels").join("compiled").join(arch));
            }
        }
        let mut gpu = match hipfire_rdna::Gpu::init() {
            Ok(g) => g,
            Err(e) => {
                report_gpu_init_failure(&e);
                std::process::exit(1);
            }
        };
        eprintln!("Pre-compiling kernels for {}...", gpu.arch);
        let mut errors = 0usize;
        for kv in &["asym3", "q8"] {
            for wq in &["mq4", "mq6", "hfq4", "hfq6", "q8"] {
                if let Err(e) = gpu.precompile_qwen35(wq, kv, 256) {
                    eprintln!("  {wq}/{kv}: {e}");
                    errors += 1;
                }
            }
        }
        if errors > 0 {
            eprintln!("Kernel precompilation finished with {errors} failure(s) — the missing kernels will JIT on first use.");
        } else {
            eprintln!("Kernel precompilation done.");
        }
        return;
    }

    // Machine-wide mutex — prevents orphan daemons from silently coexisting
    // (observed 2026-04-13: two daemons at 100% CPU survived pkill -f rounds
    // because they'd been reparented to PID 1 after their serve parent died).
    // Kept in a binding so the fd lives for the full process lifetime.
    let _daemon_lock = acquire_daemon_lock();
    let _resource_lease = hipfire_daemon_adapter::acquire_resource_lease_or_exit();
    hipfire_runtime::logging::init_stderr_logging("daemon");
    let llm_registry = build_local_llm_registry();
    eprintln!(
        "[hipfire-daemon] model registry: {} model(s), {} sidecar/template artifact(s) (models={}, triattn={}, drafts={}, templates={})",
        llm_registry.model_count(),
        llm_registry.sidecar_count(),
        llm_registry.models_dir,
        llm_registry.triattn_dir,
        llm_registry.drafts_dir,
        llm_registry.templates_dir,
    );

    let mut gpu = match hipfire_rdna::Gpu::init() {
        Ok(g) => g,
        Err(e) => {
            report_gpu_init_failure(&e);
            std::process::exit(1);
        }
    };
    let mut model: Option<LoadedModel> = None;
    let mut active_worker_id = DEFAULT_MODEL_WORKER_ID.to_string();
    let mut resident_models: std::collections::HashMap<String, LoadedModel> =
        std::collections::HashMap::new();
    let mut generic_state_arena = GenericSequenceStateArena::new();
    // PFlash speculative-prefill state. None unless the load message
    // includes a `prefill_drafter` path AND `prefill_compression` != "off".
    // Lives alongside `model` so unload_model + this state are paired
    // teardowns.
    let mut pflash_state: Option<hipfire_arch_qwen35::pflash::PflashState> = None;
    // The PflashConfig captured at load time. Per-request `prefill_*`
    // params override individual fields; the rest fall back to these
    // load-time defaults. Cleared alongside `pflash_state`.
    let mut pflash_cfg: Option<hipfire_arch_qwen35::pflash::PflashConfig> = None;
    // H-Neurons CETT capture: per-layer down_proj column norms (`[n_layers][intermediate]`),
    // loaded once via `cett_load_colnorms` and reused for every `cett_capture` prefill.
    let mut cett_colnorms: Option<Vec<Vec<f32>>> = None;
    // Hetero PFlash: when prefill_drafter_device differs from the target,
    // the drafter weights/KV/scratch live on a sibling device. The compress
    // output is a host-side Vec<u32>, so no peer-copy is needed — generate
    // routes maybe_compress_prompt to this handle, decode stays on target.
    // None means the drafter shares the target gpu (single-card, unchanged).
    let mut pflash_drafter_gpu: Option<hipfire_rdna::Gpu> = None;
    let mut dummy_model: Option<DummyModelState> = None;
    // Resident micro-step-preemptible LoRA training session (see LoraTrainSession).
    // Some between quanta of a run; runner drives one quantum per TrainLora request.
    let mut lora_train_session: Option<LoraTrainSession> = None;
    // Resident micro-step-preemptible SSM-drafter training session (see
    // DrafterTrainSession). Some between quanta of a run; runner drives one
    // quantum of EPOCHS per TrainDrafter request.
    let mut drafter_train_session: Option<DrafterTrainSession> = None;
    let mut resource_reservations = ResourceReservationManager::from_env();
    if let Err(err) = resource_reservations.reacquire_placeholders(&mut gpu) {
        hipfire_daemon_adapter::fatal_startup_error(
            &format!("failed to claim configured resource reservations: {err}"),
            None,
        );
    }

    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout();

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        if line.trim().is_empty() {
            continue;
        }

        let msg: serde_json::Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(e) => {
                // Build the envelope through serde_json: the parse-error text is
                // not JSON-safe (serde messages can carry quotes/newlines and
                // echo offending input), so raw interpolation would emit a
                // malformed line and corrupt the JSONL stream.
                emit_error_with_id(&mut stdout, "", format!("invalid JSON: {e}"));
                continue;
            }
        };

        let msg_type = msg.get("type").and_then(|v| v.as_str()).unwrap_or("");
        let protocol_load = if msg_type == "load" {
            serde_json::from_value::<hipfire_model::ModelLoadRequest>(msg.clone()).ok()
        } else {
            None
        };

        let request: DaemonRequest = match serde_json::from_value(msg.clone()) {
            Ok(request) => request,
            Err(e) => {
                // Unknown "type" tag, or a known tag whose payload the typed
                // contract rejects. Build the envelope through serde_json
                // (emit_error_with_id) so the error text can't corrupt the
                // JSONL stream.
                let id = msg.get("id").and_then(|v| v.as_str()).unwrap_or("");
                emit_error_with_id(
                    &mut stdout,
                    id,
                    format!("unsupported or malformed request '{msg_type}': {e}"),
                );
                continue;
            }
        };

        match request {
            DaemonRequest::ModelRegistry => {
                let _ = serde_json::to_writer(
                    &mut stdout,
                    &serde_json::json!({
                        "type": "model_registry",
                        "registry": llm_registry
                    }),
                );
                let _ = writeln!(stdout);
                let _ = stdout.flush();
            }
            DaemonRequest::Load(_) => {
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
                if requested_worker_id == active_worker_id {
                    generic_state_arena.release_worker(&requested_worker_id);
                    if let Some(mut pf) = pflash_state.take() {
                        if let Some(mut dg) = pflash_drafter_gpu.take() {
                            dg.bind_thread_or_warn();
                            pf.unload_drafter(&mut dg); // sibling-device drafter: free on its own handle, then drop
                            gpu.bind_thread_or_warn();
                        } else {
                            pf.unload_drafter(&mut gpu);
                        }
                    }
                    pflash_cfg = None;
                    if let Some(m) = model.take() {
                        unload_model(m, &mut gpu);
                    }
                    resource_reservations.remove_worker(&requested_worker_id);
                } else {
                    if let Err(e) = park_active_model(
                        &mut model,
                        &mut gpu,
                        &active_worker_id,
                        &mut resident_models,
                    ) {
                        write_error(&mut stdout, "", &format!("worker switch failed: {e}"));
                        let _ = stdout.flush();
                        continue;
                    }
                    active_worker_id = requested_worker_id.clone();
                }
                if let Some(m) = resident_models.remove(&requested_worker_id) {
                    generic_state_arena.release_worker(&requested_worker_id);
                    unload_model(m, &mut gpu);
                    resource_reservations.remove_worker(&requested_worker_id);
                }
                dummy_model = None;

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
                    dummy_model = Some(DummyModelState::default());
                    if let Err(err) = resource_reservations.reacquire_placeholders(&mut gpu) {
                        write_error(
                            &mut stdout,
                            "",
                            &format!("dummy load resource reservation failed: {err}"),
                        );
                        let _ = stdout.flush();
                        continue;
                    }
                    tracing::info!(
                        model = "hipfire:dummy",
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
                    let _ = writeln!(stdout, "{line}");
                    let _ = stdout.flush();
                    continue;
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
                    gpu.mmq_screen = v;
                }
                if let Some(v) = msg
                    .get("params")
                    .and_then(|p| p.get("mmq_screen_threshold"))
                    .and_then(|v| v.as_f64())
                {
                    gpu.mmq_screen_threshold = v as f32;
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
                let pflash_load_err: Option<String> =
                    if !(pflash_keep_ratio > 0.0 && pflash_keep_ratio <= 1.0) {
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
                    if draft_path.is_some()
                        && std::env::var("HIPFIRE_PP_DFLASH").ok().as_deref() != Some("1")
                    {
                        let _ = writeln!(
                            stdout,
                            r#"{{"type":"error","message":"DFlash speculative decode requires pp=1 in v1 (set HIPFIRE_PP_DFLASH=1 to opt into the experimental pp>1 PRD path; note PR2-4 of docs/plans/hetero-pflash-dflash.prd are not yet implemented — the load message will accept but generate will not run cross-card spec-decode). See issue #58 v1.1 roadmap."}}"#
                        );
                        let _ = stdout.flush();
                        continue;
                    }
                    if cask.sidecar.is_some() {
                        let _ = writeln!(
                            stdout,
                            r#"{{"type":"error","message":"CASK / TriAttention eviction requires pp=1 in v1; see issue #58 v1.1 roadmap"}}"#
                        );
                        let _ = stdout.flush();
                        continue;
                    }
                    if (pflash_drafter.is_some() || pflash_mode_str != "off")
                        && std::env::var("HIPFIRE_PP_PFLASH").ok().as_deref() != Some("1")
                    {
                        let _ = writeln!(
                            stdout,
                            r#"{{"type":"error","message":"PFlash prefill compression requires pp=1 in v1 (set HIPFIRE_PP_PFLASH=1 to opt into the experimental pp>1 PoC); see issue #58 v1.1 roadmap"}}"#
                        );
                        let _ = stdout.flush();
                        continue;
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
                hipfire_runtime::load_progress::set_sink(Some(Box::new(
                    |current, total, phase| emit_load_progress(current, total, phase),
                )));
                let _qwen_residency_env =
                    qwen_residency_load_env(protocol_load.as_ref().map(|req| &req.params));
                let planned_resource_usage = resource_reservations
                    .planned_usage_for_load(path, protocol_load.as_ref().map(|req| &req.params));
                if let Err(err) = resource_reservations.release_placeholders(&mut gpu) {
                    hipfire_runtime::load_progress::set_sink(None);
                    write_error(
                        &mut stdout,
                        "",
                        &format!("resource reservation release failed before load: {err}"),
                    );
                    let _ = stdout.flush();
                    continue;
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
                    &mut gpu,
                );
                hipfire_runtime::load_progress::set_sink(None);
                match load_result {
                    Ok(mut m) => {
                        resource_reservations
                            .set_worker_usage(requested_worker_id.clone(), planned_resource_usage);
                        if let Err(err) = resource_reservations.reacquire_placeholders(&mut gpu) {
                            resource_reservations.remove_worker(&requested_worker_id);
                            unload_model(m, &mut gpu);
                            let _ = resource_reservations.reacquire_placeholders(&mut gpu);
                            write_error(
                                &mut stdout,
                                "",
                                &format!("resource reservation reacquire failed after load: {err}"),
                            );
                            let _ = stdout.flush();
                            continue;
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
                        let vl = m.vision_config.is_some()
                            || m.dots_ocr_config.is_some()
                            || m.gemma3_vl.is_some();
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
                                std::path::Path::new(&m.model_path.replace(".hfq", ".mtp.hfq"))
                                    .exists();
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
                        match hipfire_lora_hfq::read_bundled_lora(std::path::Path::new(
                            &m.model_path,
                        )) {
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
                                    if let Err(e) = gpu.dpm_warmup(secs) {
                                        eprintln!("[daemon] dpm_warmup failed (non-fatal): {e:?}");
                                    }
                                }
                            }
                        }

                        let model_worker =
                            model_worker_runtime_view_json(&loaded_model_worker_runtime_view(&m));
                        let cache_capable = m.arch_id == ARCH_ID_DEEPSEEK4_FLASH
                            || is_qwen35_family_arch_id(m.arch_id);
                        let _ = writeln!(
                            stdout,
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
                        // the reason and continue with PFlash disabled so
                        // the operator gets a clear "model is up, but
                        // compression isn't" signal rather than losing
                        // the entire session.
                        if let Some(ref pf_drafter_path) = pflash_drafter {
                            if pflash_mode_str != "off" {
                                if let Some(ref reason) = pflash_load_err {
                                    let _ = writeln!(
                                        stdout,
                                        r#"{{"type":"pflash_load_failed","reason":"invalid load param: {}"}}"#,
                                        reason.replace('"', "'")
                                    );
                                    let _ = stdout.flush();
                                    model = Some(m);
                                    continue;
                                }
                                let pf_cfg = hipfire_arch_qwen35::pflash::PflashConfig {
                                    mode: hipfire_arch_qwen35::pflash::PflashMode::parse(
                                        &pflash_mode_str,
                                    )
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
                                let mut pf_state =
                                    hipfire_arch_qwen35::pflash::PflashState::new(&pf_cfg);
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
                                        match hipfire_rdna::Gpu::init_with_device(
                                            pflash_drafter_device,
                                        ) {
                                            Ok(g) => sibling = Some(g),
                                            Err(e) => {
                                                let _ = writeln!(
                                                    stdout,
                                                    r#"{{"type":"pflash_load_failed","reason":"drafter device {} init: {}"}}"#,
                                                    pflash_drafter_device,
                                                    e.to_string().replace('"', "'")
                                                );
                                            }
                                        }
                                    }
                                    let dg: &mut hipfire_rdna::Gpu =
                                        sibling.as_mut().unwrap_or(&mut gpu);
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
                                                stdout,
                                                r#"{{"type":"pflash","mode":"{}","drafter":"{}","drafter_device":{},"tokenizer_compat":{},"keep_ratio":{},"threshold":{}}}"#,
                                                pflash_mode_str,
                                                pf_drafter_path,
                                                pflash_drafter_device,
                                                pf_state.tokenizer_compat,
                                                pflash_keep_ratio,
                                                pflash_threshold
                                            );
                                            pflash_state = Some(pf_state);
                                            pflash_cfg = Some(pf_cfg);
                                            pflash_drafter_gpu = sibling; // persist sibling across requests (None if shared)
                                        }
                                        Err(e) => {
                                            eprintln!("[pflash] LOAD FAILED: {}", e);
                                            let _ = writeln!(
                                                stdout,
                                                r#"{{"type":"pflash_load_failed","reason":"{}"}}"#,
                                                e.to_string().replace('"', "'")
                                            );
                                        }
                                    }
                                } else {
                                    let _ = writeln!(
                                        stdout,
                                        r#"{{"type":"pflash_load_failed","reason":"target tokenizer unavailable"}}"#
                                    );
                                }
                            }
                        }

                        model = Some(m);
                    }
                    Err(e) => {
                        if let Err(err) = resource_reservations.reacquire_placeholders(&mut gpu) {
                            eprintln!(
                                "[hipfire-daemon] failed to restore resource reservations after load failure: {err}"
                            );
                        }
                        let (vram_free, vram_total) = gpu.hip.get_vram_info().unwrap_or((0, 0));
                        let free_mb = vram_free / (1024 * 1024);
                        let total_mb = vram_total / (1024 * 1024);
                        // serde-escape: raw HipError debug contains { } and "
                        // which corrupt the JSONL protocol if interpolated raw.
                        write_error(&mut stdout, "", &format!(
                            "load failed: {e}. GPU: {} ({free_mb} MB free / {total_mb} MB total)", gpu.arch));
                    }
                }
                let _ = stdout.flush();
            }

            DaemonRequest::Embed(req) => {
                let id = msg.get("id").and_then(|v| v.as_str()).unwrap_or("");
                let target_worker_id = message_worker_id(&msg);
                if dummy_model.is_some() {
                    emit_error_with_id(
                        &mut stdout,
                        id,
                        "embed is not supported for the dummy model",
                    );
                    continue;
                }
                match activate_model_worker(
                    &target_worker_id,
                    &mut active_worker_id,
                    &mut model,
                    &mut gpu,
                    &mut resident_models,
                ) {
                    Ok(true) => {}
                    Ok(false) => {
                        emit_error_with_id(
                            &mut stdout,
                            id,
                            format!("unknown model worker {target_worker_id}"),
                        );
                        continue;
                    }
                    Err(e) => {
                        emit_error_with_id(&mut stdout, id, format!("worker switch failed: {e}"));
                        continue;
                    }
                }
                let Some(m) = model.as_ref() else {
                    emit_error_with_id(&mut stdout, id, "no model loaded");
                    continue;
                };
                match embeddinggemma_embed(&mut gpu, m, &req.texts, req.input_type, req.dims) {
                    Ok(embeddings) => {
                        let _ = serde_json::to_writer(
                            &mut stdout,
                            &serde_json::json!({
                                "type": "embeddings",
                                "id": id,
                                "embeddings": embeddings,
                            }),
                        );
                        let _ = writeln!(stdout);
                        let _ = stdout.flush();
                    }
                    Err(e) => emit_error_with_id(&mut stdout, id, e),
                }
            }

            DaemonRequest::Rerank(req) => {
                let id = msg.get("id").and_then(|v| v.as_str()).unwrap_or("");
                let target_worker_id = message_worker_id(&msg);
                if dummy_model.is_some() {
                    emit_error_with_id(
                        &mut stdout,
                        id,
                        "rerank is not supported for the dummy model",
                    );
                    continue;
                }
                match activate_model_worker(
                    &target_worker_id,
                    &mut active_worker_id,
                    &mut model,
                    &mut gpu,
                    &mut resident_models,
                ) {
                    Ok(true) => {}
                    Ok(false) => {
                        emit_error_with_id(
                            &mut stdout,
                            id,
                            format!("unknown model worker {target_worker_id}"),
                        );
                        continue;
                    }
                    Err(e) => {
                        emit_error_with_id(&mut stdout, id, format!("worker switch failed: {e}"));
                        continue;
                    }
                }
                let Some(m) = model.as_ref() else {
                    emit_error_with_id(&mut stdout, id, "no model loaded");
                    continue;
                };
                match embeddinggemma_rerank(&mut gpu, m, &req.query, &req.documents) {
                    Ok(results) => {
                        let _ = serde_json::to_writer(
                            &mut stdout,
                            &serde_json::json!({
                                "type": "rerank_scores",
                                "id": id,
                                "results": results,
                            }),
                        );
                        let _ = writeln!(stdout);
                        let _ = stdout.flush();
                    }
                    Err(e) => emit_error_with_id(&mut stdout, id, e),
                }
            }

            DaemonRequest::Generate(_) => {
                // Explicit per-request raw-prompt override (optional `"raw"`
                // bool). Absent → None → auto default (raw iff no chat_template).
                // Always set, so it resets every request (no cross-request leak).
                RAW_OVERRIDE.with(|c| c.set(msg.get("raw").and_then(|v| v.as_bool())));
                let protocol_generate =
                    serde_json::from_value::<hipfire_generate::GenerateTextRequest>(msg.clone())
                        .ok();
                let id = protocol_generate
                    .as_ref()
                    .map(|req| req.id.as_str())
                    .or_else(|| msg.get("id").and_then(|v| v.as_str()))
                    .unwrap_or("0");
                let target_worker_id = message_worker_id(&msg);
                if dummy_model.is_none() {
                    match activate_model_worker(
                        &target_worker_id,
                        &mut active_worker_id,
                        &mut model,
                        &mut gpu,
                        &mut resident_models,
                    ) {
                        Ok(true) => {}
                        Ok(false) => {
                            emit_error_with_id(
                                &mut stdout,
                                id,
                                format!("unknown model worker {target_worker_id}"),
                            );
                            continue;
                        }
                        Err(e) => {
                            emit_error_with_id(
                                &mut stdout,
                                id,
                                format!("worker switch failed: {e}"),
                            );
                            continue;
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
                if let Some(dummy) = dummy_model.as_mut() {
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
                        &mut stdout,
                        id,
                        session_id,
                        prompt,
                        prefill_already_done,
                        max_tokens,
                    );
                    continue;
                }
                let m = match model.as_mut() {
                    Some(m) => m,
                    None => {
                        let _ =
                            writeln!(stdout, r#"{{"type":"error","message":"no model loaded"}}"#);
                        let _ = stdout.flush();
                        continue;
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
                    let target_session_id =
                        session_id.unwrap_or_else(|| loaded_model_default_session_id(m));
                    if let Err(e) = m.activate_session(&mut gpu, target_session_id) {
                        emit_error_with_id(&mut stdout, id, e);
                        continue;
                    }
                } else if session_id.is_some() || prefill_already_done {
                    emit_error_with_id(
                        &mut stdout,
                        id,
                        "session_id/prefill_already_done are only supported for single-GPU qwen35/qwen35-moe/lfm2-moe",
                    );
                    continue;
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
                let tools_json: Option<Vec<serde_json::Value>> = if let Some(tools) =
                    protocol_generate.as_ref().and_then(|req| req.tools.clone())
                {
                    match serde_json::from_value::<Vec<serde_json::Value>>(tools) {
                        Ok(t) => Some(t),
                        Err(e) => {
                            let _ = writeln!(
                                stdout,
                                r#"{{"type":"error","id":"{}","message":"invalid tools field: {}"}}"#,
                                id,
                                e.to_string().replace('"', "'"),
                            );
                            let _ = stdout.flush();
                            continue;
                        }
                    }
                } else {
                    match msg.get("tools") {
                        Some(v) => {
                            match serde_json::from_value::<Vec<serde_json::Value>>(v.clone()) {
                                Ok(t) => Some(t),
                                Err(e) => {
                                    let _ = writeln!(
                                        stdout,
                                        r#"{{"type":"error","id":"{}","message":"invalid tools field: {}"}}"#,
                                        id,
                                        e.to_string().replace('"', "'"),
                                    );
                                    let _ = stdout.flush();
                                    continue;
                                }
                            }
                        }
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
                        Some(v) => {
                            match serde_json::from_value::<Vec<prompt_frame::Message>>(v.clone()) {
                                Ok(m) => Some(m),
                                Err(e) => {
                                    let _ = writeln!(
                                        stdout,
                                        r#"{{"type":"error","id":"{}","message":"invalid messages field: {}"}}"#,
                                        id,
                                        e.to_string().replace('"', "'"),
                                    );
                                    let _ = stdout.flush();
                                    continue;
                                }
                            }
                        }
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
                    Some(req) if !req.sampling.temperature_is_default => {
                        Some(req.sampling.temperature)
                    }
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
                let max_frames =
                    msg.get("max_frames").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
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
                        &mut stdout,
                        id,
                        "video input is only supported on gemma3-vl (arch 13)",
                    );
                } else if has_media && !has_vl {
                    write_error(&mut stdout, id, "model has no vision encoder");
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
                                &mut gpu,
                                &mut stdout,
                                &params,
                                &frames,
                                &image_labels,
                            );
                        }
                        Err(e) => write_error(&mut stdout, id, &e),
                    }
                } else if has_image && has_vl {
                    if image_base64.is_some() && image.is_some() {
                        eprintln!(
                            "[daemon/vl] both image and image_base64 provided — using image_base64"
                        );
                    }
                    let source = if let Some(b64) = image_base64 {
                        if b64.len() > MAX_BASE64_ENCODED_LEN {
                            write_error(
                                &mut stdout,
                                id,
                                &format!(
                                    "image payload exceeds maximum encoded size ({} bytes)",
                                    MAX_BASE64_ENCODED_LEN,
                                ),
                            );
                            continue;
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
                        generate_vl_dots_ocr(m, &mut gpu, &mut stdout, &params);
                    } else {
                        generate_vl(m, &mut gpu, &mut stdout, &params);
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
                    let pf_cfg_owned = pflash_cfg.as_ref().map(|base| {
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
                                pf_override_err =
                                    Some(format!("prefill_keep_ratio={r} not in (0, 1]"));
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
                            stdout,
                            r#"{{"type":"error","id":"{}","message":"invalid pflash override: {}"}}"#,
                            id,
                            reason.replace('"', "'"),
                        );
                        let _ = stdout.flush();
                        continue;
                    }
                    generate(
                        m,
                        &mut gpu,
                        pflash_drafter_gpu.as_mut(),
                        &mut stdout,
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
                        pflash_state.as_mut(),
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

            DaemonRequest::GenerateBatchPrefill => match validate_generate_batch_prefill(&msg) {
                Ok(envelope) => {
                    let target_worker_id = message_worker_id(&msg);
                    if dummy_model.is_none() {
                        match activate_model_worker(
                            &target_worker_id,
                            &mut active_worker_id,
                            &mut model,
                            &mut gpu,
                            &mut resident_models,
                        ) {
                            Ok(true) => {}
                            Ok(false) => {
                                emit_error_with_id(
                                    &mut stdout,
                                    &envelope.id,
                                    format!("unknown model worker {target_worker_id}"),
                                );
                                continue;
                            }
                            Err(e) => {
                                emit_error_with_id(
                                    &mut stdout,
                                    &envelope.id,
                                    format!("worker switch failed: {e}"),
                                );
                                continue;
                            }
                        }
                    }
                    if envelope.is_probe() {
                        if dummy_model.is_some() {
                            emit_dummy_generate_batch_prefill_ready(&mut stdout, &envelope);
                            continue;
                        }
                        match model.as_ref() {
                            Some(m) if is_qwen35_family_arch_id(m.arch_id) && m.pp == 1 => {
                                emit_generate_batch_prefill_ready(&mut stdout, &envelope);
                            }
                            #[cfg(feature = "arch-lfm2moe")]
                            Some(m) if m.arch_id == ARCH_ID_LFM2_MOE && m.pp == 1 => {
                                emit_lfm2_generate_batch_prefill_ready(&mut stdout, &envelope);
                            }
                            Some(m) => {
                                let reason = format!(
                                    "generate_batch_prefill currently supports qwen35/qwen35-moe and lfm2-moe only (arch_id={})",
                                    m.arch_id
                                );
                                emit_generate_batch_prefill_unsupported(
                                    &mut stdout,
                                    &envelope,
                                    &reason,
                                );
                            }
                            None => {
                                emit_generate_batch_prefill_unsupported(
                                    &mut stdout,
                                    &envelope,
                                    "no model loaded",
                                );
                            }
                        }
                        continue;
                    }
                    if let Some(dummy) = dummy_model.as_mut() {
                        tracing::info!(
                            request_id = envelope.id,
                            batch_id = envelope.batch_id,
                            sessions = envelope.session_count,
                            "dummy generate_batch_prefill"
                        );
                        if let Err(e) =
                            run_generate_batch_prefill_dummy(dummy, &mut stdout, &envelope)
                        {
                            emit_error_with_id(&mut stdout, &envelope.id, e);
                        }
                        continue;
                    }
                    let m = match model.as_mut() {
                        Some(m) => m,
                        None => {
                            emit_error_with_id(&mut stdout, &envelope.id, "no model loaded");
                            continue;
                        }
                    };
                    if is_qwen35_family_arch_id(m.arch_id) {
                        if let Err(e) = run_generate_batch_prefill_serial_qwen35(
                            m,
                            &mut gpu,
                            &mut stdout,
                            &envelope,
                            pflash_state.is_some(),
                        ) {
                            emit_error_with_id(&mut stdout, &envelope.id, e);
                        }
                    } else {
                        #[cfg(feature = "arch-lfm2moe")]
                        if m.arch_id == ARCH_ID_LFM2_MOE {
                            if let Err(e) = run_generate_batch_prefill_serial_lfm2(
                                m,
                                &mut gpu,
                                &mut stdout,
                                &envelope,
                            ) {
                                emit_error_with_id(&mut stdout, &envelope.id, e);
                            }
                            continue;
                        }
                        emit_error_with_id(
                            &mut stdout,
                            &envelope.id,
                            format!(
                                "generate_batch_prefill currently supports qwen35/qwen35-moe and lfm2-moe only (arch_id={})",
                                m.arch_id
                            ),
                        );
                    }
                }
                Err(e) => {
                    let id = msg.get("id").and_then(|v| v.as_str()).unwrap_or("");
                    emit_error_with_id(&mut stdout, id, e);
                }
            },

            DaemonRequest::PrefixHashPreflight => match validate_prefix_hash_preflight(&msg) {
                Ok(envelope) => {
                    let target_worker_id = message_worker_id(&msg);
                    match activate_model_worker(
                        &target_worker_id,
                        &mut active_worker_id,
                        &mut model,
                        &mut gpu,
                        &mut resident_models,
                    ) {
                        Ok(true) => {}
                        Ok(false) => {
                            emit_error_with_id(
                                &mut stdout,
                                &envelope.id,
                                format!("unknown model worker {target_worker_id}"),
                            );
                            continue;
                        }
                        Err(e) => {
                            emit_error_with_id(
                                &mut stdout,
                                &envelope.id,
                                format!("worker switch failed: {e}"),
                            );
                            continue;
                        }
                    }
                    let m = match model.as_ref() {
                        Some(m) => m,
                        None => {
                            emit_error_with_id(&mut stdout, &envelope.id, "no model loaded");
                            continue;
                        }
                    };
                    let preflight_result = if is_qwen35_family_arch_id(m.arch_id) {
                        run_prefix_hash_preflight_qwen35(m, &mut stdout, &envelope)
                    } else {
                        #[cfg(feature = "arch-lfm2moe")]
                        {
                            if m.arch_id == ARCH_ID_LFM2_MOE {
                                run_prefix_hash_preflight_lfm2(m, &mut stdout, &envelope)
                            } else {
                                Err(format!(
                                    "prefix_hash_preflight currently supports qwen35/qwen35-moe and lfm2-moe only (arch_id={})",
                                    m.arch_id
                                ))
                            }
                        }
                        #[cfg(not(feature = "arch-lfm2moe"))]
                        {
                            Err(format!(
                                "prefix_hash_preflight currently supports qwen35/qwen35-moe only (arch_id={})",
                                m.arch_id
                            ))
                        }
                    };
                    if let Err(e) = preflight_result {
                        emit_error_with_id(&mut stdout, &envelope.id, e);
                    }
                }
                Err(e) => {
                    let id = msg.get("id").and_then(|v| v.as_str()).unwrap_or("");
                    emit_error_with_id(&mut stdout, id, e);
                }
            },

            DaemonRequest::GenerateBatchDecodeStep => match validate_generate_batch_decode(&msg) {
                Ok(envelope) => {
                    let target_worker_id = message_worker_id(&msg);
                    match activate_model_worker(
                        &target_worker_id,
                        &mut active_worker_id,
                        &mut model,
                        &mut gpu,
                        &mut resident_models,
                    ) {
                        Ok(true) => {}
                        Ok(false) => {
                            emit_error_with_id(
                                &mut stdout,
                                &envelope.id,
                                format!("unknown model worker {target_worker_id}"),
                            );
                            continue;
                        }
                        Err(e) => {
                            emit_error_with_id(
                                &mut stdout,
                                &envelope.id,
                                format!("worker switch failed: {e}"),
                            );
                            continue;
                        }
                    }
                    let m = match model.as_mut() {
                        Some(m) => m,
                        None => {
                            emit_error_with_id(&mut stdout, &envelope.id, "no model loaded");
                            continue;
                        }
                    };
                    if let Err(e) =
                        run_generate_batch_decode_step_qwen35(m, &mut gpu, &mut stdout, &envelope)
                    {
                        emit_error_with_id(&mut stdout, &envelope.id, e);
                    }
                }
                Err(e) => {
                    let id = msg.get("id").and_then(|v| v.as_str()).unwrap_or("");
                    emit_error_with_id(&mut stdout, id, e);
                }
            },

            DaemonRequest::ReleaseSessions => {
                let id = msg.get("id").and_then(|v| v.as_str()).unwrap_or("release");
                let target_worker_id = message_worker_id(&msg);
                if dummy_model.is_none() {
                    match activate_model_worker(
                        &target_worker_id,
                        &mut active_worker_id,
                        &mut model,
                        &mut gpu,
                        &mut resident_models,
                    ) {
                        Ok(true) => {}
                        Ok(false) => {
                            emit_error_with_id(
                                &mut stdout,
                                id,
                                format!("unknown model worker {target_worker_id}"),
                            );
                            continue;
                        }
                        Err(e) => {
                            emit_error_with_id(
                                &mut stdout,
                                id,
                                format!("worker switch failed: {e}"),
                            );
                            continue;
                        }
                    }
                }
                let request = match parse_release_sessions_request(&msg, &target_worker_id) {
                    Ok(request) => request,
                    Err(e) => {
                        emit_error_with_id(&mut stdout, id, e);
                        continue;
                    }
                };
                if let Some(dummy) = dummy_model.as_mut() {
                    let released = dummy.release_sessions(&request.sessions);
                    let done = release_sessions_done_json(
                        id,
                        request.sessions.len(),
                        released,
                        dummy.session_count(),
                        None,
                    );
                    let _ = writeln!(stdout, "{done}");
                    let _ = stdout.flush();
                    continue;
                }
                let m = match model.as_mut() {
                    Some(m) => m,
                    None => {
                        emit_error_with_id(&mut stdout, id, "no model loaded");
                        continue;
                    }
                };
                let arena_backend = loaded_model_state_arena_backend(m);
                match sequence_state_arena_release_sessions(
                    arena_backend,
                    m,
                    &mut gpu,
                    &request.sessions,
                ) {
                    Ok(released) => {
                        let worker = loaded_model_worker_runtime_view(m);
                        let done = release_sessions_done_json(
                            id,
                            request.sessions.len(),
                            released,
                            sequence_state_arena_resident_session_count(arena_backend, m),
                            Some(&worker),
                        );
                        let _ = writeln!(stdout, "{done}");
                        let _ = stdout.flush();
                    }
                    Err(e) => emit_error_with_id(&mut stdout, id, e),
                }
            }

            DaemonRequest::ReserveSessionState => {
                let id = msg.get("id").and_then(|v| v.as_str()).unwrap_or("reserve");
                let target_worker_id = message_worker_id(&msg);
                generic_state_arena.purge_expired();
                if dummy_model.is_none() {
                    match activate_model_worker(
                        &target_worker_id,
                        &mut active_worker_id,
                        &mut model,
                        &mut gpu,
                        &mut resident_models,
                    ) {
                        Ok(true) => {}
                        Ok(false) => {
                            emit_error_with_id(
                                &mut stdout,
                                id,
                                format!("unknown model worker {target_worker_id}"),
                            );
                            continue;
                        }
                        Err(e) => {
                            emit_error_with_id(
                                &mut stdout,
                                id,
                                format!("worker switch failed: {e}"),
                            );
                            continue;
                        }
                    }
                }
                let request = match parse_reserve_session_state_request(&msg, &target_worker_id) {
                    Ok(request) => request,
                    Err(e) => {
                        emit_error_with_id(&mut stdout, id, e);
                        continue;
                    }
                };
                let reservation_id = request.reservation_id.clone().unwrap_or_else(|| {
                    format!(
                        "reserve:{}:{}",
                        request.worker_id,
                        std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .map(|d| d.as_nanos())
                            .unwrap_or(0)
                    )
                });
                let reservation_plan = if let Some(m) = model.as_ref() {
                    let budget = request
                        .budget_bytes
                        .unwrap_or_else(resident_state_reservation_budget_bytes);
                    let arena_backend = loaded_model_state_arena_backend(m);
                    let descriptors = sequence_state_arena_page_descriptors(arena_backend, m);
                    sequence_state_reservation_plan(
                        &descriptors,
                        request.physical_cap,
                        generic_state_arena.outstanding_bytes_for_worker(&request.worker_id),
                        budget,
                    )
                } else if dummy_model.is_some() {
                    let budget = request
                        .budget_bytes
                        .unwrap_or_else(resident_state_reservation_budget_bytes);
                    sequence_state_reservation_plan_for_reserved_bytes(1024, 0, 0, budget)
                } else {
                    emit_error_with_id(&mut stdout, id, "no model loaded");
                    continue;
                };
                if reservation_plan.rejected_for_memory_pressure {
                    let rejected = reserve_session_state_rejected_json(
                        id,
                        &request.worker_id,
                        reservation_plan.reserved_bytes,
                        reservation_plan.current_session_bytes,
                        reservation_plan.outstanding_reserved_bytes,
                        reservation_plan.projected_reserved_bytes,
                        reservation_plan.budget_bytes,
                    );
                    let _ = writeln!(stdout, "{rejected}");
                    let _ = stdout.flush();
                    continue;
                }
                let reservation = generic_state_arena.reserve(
                    &request.worker_id,
                    reservation_id.clone(),
                    &request.state_kinds,
                    request.physical_cap,
                    reservation_plan.reserved_bytes,
                    request.ttl_ms,
                );
                let done = reserve_session_state_done_json(
                    id,
                    &reservation,
                    reservation_plan.current_session_bytes,
                    reservation_plan.outstanding_reserved_bytes,
                    reservation_plan.projected_reserved_bytes,
                    reservation_plan.budget_bytes,
                );
                let _ = writeln!(stdout, "{done}");
                let _ = stdout.flush();
            }

            DaemonRequest::DescribeState => {
                let id = msg
                    .get("id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("describe-state");
                generic_state_arena.purge_expired();
                let request = match parse_describe_sequence_state_request(&msg) {
                    Ok(request) => request,
                    Err(e) => {
                        emit_error_with_id(&mut stdout, id, e);
                        continue;
                    }
                };
                if parsed_handle_may_target_generic(&request.handle) {
                    if let Some(reservation) =
                        generic_state_arena.describe(&request.handle.id, request.handle.generation)
                    {
                        let done = session_state_reservation_describe_json(id, reservation);
                        let _ = writeln!(stdout, "{done}");
                        let _ = stdout.flush();
                        continue;
                    }
                }
                let Some(described) = describe_loaded_sequence_state(
                    &active_worker_id,
                    model.as_ref(),
                    &resident_models,
                    &request.handle,
                ) else {
                    emit_error_with_id(
                        &mut stdout,
                        id,
                        format!(
                            "describe_state unknown runtime_state_handle {}",
                            request.handle.id
                        ),
                    );
                    continue;
                };
                let done = described_sequence_state_json(id, &described);
                let _ = writeln!(stdout, "{done}");
                let _ = stdout.flush();
            }

            DaemonRequest::ReleaseState => {
                let id = msg
                    .get("id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("release-reservation");
                let request = parse_release_sequence_state_request(&msg);
                let generic_handles = request
                    .handles
                    .iter()
                    .filter(|handle| parsed_handle_may_target_generic(handle))
                    .map(|handle| (handle.id.clone(), handle.generation))
                    .collect::<Vec<_>>();
                let (generic_released, generic_released_bytes) =
                    generic_state_arena.release(generic_handles);
                let (loaded_released, loaded_released_bytes) =
                    match release_loaded_sequence_state_handles(
                        &mut model,
                        &mut resident_models,
                        &mut gpu,
                        &request.handles,
                    ) {
                        Ok(released) => released,
                        Err(e) => {
                            emit_error_with_id(&mut stdout, id, e);
                            continue;
                        }
                    };
                let done = release_state_done_json(
                    request.response_kind,
                    id,
                    generic_released,
                    generic_released_bytes,
                    loaded_released,
                    loaded_released_bytes,
                );
                let _ = writeln!(stdout, "{done}");
                let _ = stdout.flush();
            }

            DaemonRequest::WorkerStatus => {
                let status = resident_worker_status_json(
                    &active_worker_id,
                    model.as_ref(),
                    &resident_models,
                );
                let _ = writeln!(stdout, "{status}");
                let _ = stdout.flush();
            }

            DaemonRequest::ResourceStatus => {
                let status = resource_reservations.status_json();
                let _ = writeln!(stdout, "{status}");
                let _ = stdout.flush();
            }

            DaemonRequest::Inventory => {
                let inventory = daemon_accelerator_inventory(&mut gpu);
                let mut payload = serde_json::to_value(inventory)
                    .unwrap_or_else(|_| serde_json::json!({"source": "daemon", "devices": []}));
                payload["type"] = serde_json::json!("inventory");
                let _ = writeln!(stdout, "{payload}");
                let _ = stdout.flush();
            }

            DaemonRequest::Reset => {
                let target_worker_id = reset_target_worker_id(&msg, &active_worker_id);
                if reset_has_no_resident_model(&dummy_model, &model, &resident_models) {
                    generic_state_arena.release_worker(&target_worker_id);
                    let _ = writeln!(stdout, r#"{{"type":"reset","seq_pos":0}}"#);
                    let _ = stdout.flush();
                    continue;
                }
                if dummy_model.is_none() {
                    match activate_model_worker(
                        &target_worker_id,
                        &mut active_worker_id,
                        &mut model,
                        &mut gpu,
                        &mut resident_models,
                    ) {
                        Ok(true) => {}
                        Ok(false) => {
                            emit_error_with_id(
                                &mut stdout,
                                "",
                                format!("unknown model worker {target_worker_id}"),
                            );
                            continue;
                        }
                        Err(e) => {
                            emit_error_with_id(
                                &mut stdout,
                                "",
                                format!("worker switch failed: {e}"),
                            );
                            continue;
                        }
                    }
                }
                // Reset conversation state without unloading the model.
                if let Some(dummy) = dummy_model.as_mut() {
                    generic_state_arena.release_worker(&target_worker_id);
                    dummy.reset();
                    let _ = writeln!(stdout, r#"{{"type":"reset"}}"#);
                    let _ = stdout.flush();
                    continue;
                }
                // Under eviction, also zero the compact_offset so absolute
                // RoPE phase restarts from zero for the fresh conversation.
                if let Some(ref mut m) = model {
                    generic_state_arena.release_worker(&target_worker_id);
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
                            let _ = gpu.hip.memset(&s.buf, 0, s.buf.size());
                        }
                        for s in &dn.s_scales {
                            let _ = gpu.hip.memset(&s.buf, 0, s.buf.size());
                        }
                        for s in &dn.conv_states {
                            let _ = gpu.hip.memset(&s.buf, 0, s.buf.size());
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
                        gpu.invalidate_graph_state();
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
                            let _ = s.reset(&mut gpu);
                        }
                        m.lfm2_registry.sessions.clear();
                        if m.arch_id == ARCH_ID_LFM2_MOE
                            && m.pp == 1
                            && m.active.lfm2moe_state.is_some()
                        {
                            m.lfm2_registry.active_session_id =
                                Some(LFM2_LEGACY_SESSION_ID.to_string());
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
                        let _ = loaded.backend.reset_session(&mut gpu, "default");
                    }
                    let _ = writeln!(stdout, r#"{{"type":"reset","seq_pos":0}}"#);
                } else {
                    let _ = writeln!(stdout, r#"{{"type":"error","message":"no model loaded"}}"#);
                }
                let _ = stdout.flush();
            }

            DaemonRequest::Unload => {
                // PFlash drafter goes FIRST: its weights/scratch/KV
                // tensors are released via Gpu::free_tensor, which only
                // queues into the GPU pool. The actual hipFree happens
                // inside unload_model -> drain_pool. Calling
                // unload_drafter AFTER unload_model would leave the
                // drafter buffers cached in the just-emptied pool with
                // no drain to follow, so the VRAM stays resident until
                // the next load message arrives. Order matters here.
                if let Some(mut pf) = pflash_state.take() {
                    if let Some(mut dg) = pflash_drafter_gpu.take() {
                        dg.bind_thread_or_warn();
                        pf.unload_drafter(&mut dg); // sibling-device drafter: free on its own handle, then drop
                        gpu.bind_thread_or_warn();
                    } else {
                        pf.unload_drafter(&mut gpu);
                    }
                }
                pflash_cfg = None;
                if let Some(m) = model.take() {
                    unload_model(m, &mut gpu);
                }
                for (_, m) in resident_models.drain() {
                    unload_model(m, &mut gpu);
                }
                resource_reservations.clear_workers();
                if let Err(err) = resource_reservations.reacquire_placeholders(&mut gpu) {
                    eprintln!(
                        "[hipfire-daemon] failed to restore resource reservations after unload: {err}"
                    );
                }
                generic_state_arena.clear();
                dummy_model = None;
                active_worker_id = DEFAULT_MODEL_WORKER_ID.to_string();
                // Drop any steer session so a stale capture/apply can't leak its
                // process-global state across model loads.
                hipfire_steer::clear();
                let _ = writeln!(stdout, r#"{{"type":"unloaded"}}"#);
                let _ = stdout.flush();
            }

            DaemonRequest::UnloadWorker => {
                let id = msg
                    .get("id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unload_worker");
                let request = parse_unload_worker_request(&msg, DEFAULT_MODEL_WORKER_ID);
                let worker_id = request.worker_id;
                let mut unloaded = false;
                generic_state_arena.release_worker(&worker_id);
                if worker_id == active_worker_id {
                    if let Some(m) = model.take() {
                        unload_model(m, &mut gpu);
                        unloaded = true;
                    }
                    active_worker_id = DEFAULT_MODEL_WORKER_ID.to_string();
                    if let Some((next_worker_id, next_model)) = resident_models
                        .iter()
                        .next()
                        .map(|(k, _)| k.clone())
                        .and_then(|k| resident_models.remove(&k).map(|m| (k, m)))
                    {
                        active_worker_id = next_worker_id;
                        model = Some(next_model);
                    }
                } else if let Some(m) = resident_models.remove(&worker_id) {
                    unload_model(m, &mut gpu);
                    unloaded = true;
                }
                if unloaded {
                    resource_reservations.remove_worker(&worker_id);
                    if let Err(err) = resource_reservations.reacquire_placeholders(&mut gpu) {
                        eprintln!(
                            "[hipfire-daemon] failed to restore resource reservations after worker unload: {err}"
                        );
                    }
                }
                let done = unload_worker_done_json(
                    id,
                    &worker_id,
                    unloaded,
                    resident_models.len() + usize::from(model.is_some()),
                );
                let _ = writeln!(stdout, "{done}");
                let _ = stdout.flush();
            }

            DaemonRequest::Ping => {
                let _ = writeln!(stdout, r#"{{"type":"pong"}}"#);
                let _ = stdout.flush();
            }

            // Calibrate the resident model in place (no reload): run the Tier-1
            // collector over a corpus and write a .calib.hfq. The data plane stays
            // daemon-internal — only the request + the resulting path/summary cross
            // JSONL. Single-GPU qwen3.5-family bf16 only (capture fires at the
            // bf16 chokepoints); additive and gated, never on the decode hot path.
            DaemonRequest::Collect(_) => {
                // Parse fields directly from the JSON message (the daemon is the
                // server side; the typed CollectRequest contract lives in
                // hipfire-daemon-protocol for clients). Field names must match.
                let Some(corpus) = msg.get("corpus").and_then(|v| v.as_str()).map(String::from)
                else {
                    emit_error_with_id(&mut stdout, "", "collect: missing 'corpus'".to_string());
                    continue;
                };
                let Some(output) = msg.get("output").and_then(|v| v.as_str()).map(String::from)
                else {
                    emit_error_with_id(&mut stdout, "", "collect: missing 'output'".to_string());
                    continue;
                };
                let max_tokens = msg
                    .get("max_tokens")
                    .and_then(|v| v.as_u64())
                    .map(|v| v as usize)
                    .unwrap_or(512);
                let kldref = msg.get("kldref").and_then(|v| v.as_bool()).unwrap_or(false);
                let Some(m) = model.as_ref() else {
                    emit_error_with_id(&mut stdout, "", "collect: no model loaded".to_string());
                    continue;
                };
                if m.pp != 1 {
                    emit_error_with_id(
                        &mut stdout,
                        "",
                        "collect: requires a single-GPU resident model (pp == 1)".to_string(),
                    );
                    continue;
                }
                // Only the tokenizer is needed up front (to encode the corpus);
                // the per-arch calibration backend is resolved below. Every arch
                // with a collector reaches it through the one CalibratableBackend
                // seam — no qwen3.5-only gate.
                let Some(tokenizer) = m.tokenizer.as_ref() else {
                    emit_error_with_id(
                        &mut stdout,
                        "",
                        "collect: resident model has no tokenizer".to_string(),
                    );
                    continue;
                };
                let text = match std::fs::read_to_string(&corpus) {
                    Ok(t) => t,
                    Err(e) => {
                        emit_error_with_id(
                            &mut stdout,
                            "",
                            format!("collect: read corpus {corpus}: {e}"),
                        );
                        continue;
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
                            &mut gpu,
                            tokenizer,
                            &tokens,
                            kldref,
                            out_path,
                            &provenance,
                        );
                    }
                    if let Some(b) = m.gemma3_text.as_ref() {
                        break 'pick b.collect_calibration(
                            &mut gpu,
                            tokenizer,
                            &tokens,
                            kldref,
                            out_path,
                            &provenance,
                        );
                    }
                    #[cfg(feature = "arch-lfm2moe")]
                    if let (Some(w), Some(c)) =
                        (m.lfm2moe_weights.as_ref(), m.lfm2moe_config.as_ref())
                    {
                        let be = lfm2moe::calibration::Lfm2MoeCalibBackend {
                            weights: w,
                            config: c,
                        };
                        break 'pick be.collect_calibration(
                            &mut gpu,
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
                            &mut gpu,
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
                        let _ = writeln!(stdout, "{resp}");
                        let _ = stdout.flush();
                    }
                    Err(e) => emit_error_with_id(&mut stdout, "", format!("collect: {e}")),
                }
            }

            // Daemon-resident KLD evaluation (no reload). `self_score` builds a
            // reference from the loaded model and scores the SAME model against
            // it through one forward path → ≈0 on a healthy run; the guard that
            // catches the historical two-binary drift. build_ref/score (with the
            // .kldref container) land next.
            DaemonRequest::KldEval(_) => {
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
                let Some(m) = model.as_mut() else {
                    emit_error_with_id(&mut stdout, "", "kld_eval: no model loaded".to_string());
                    continue;
                };
                if m.pp != 1 {
                    emit_error_with_id(
                        &mut stdout,
                        "",
                        "kld_eval: requires a single-GPU resident model (pp == 1)".to_string(),
                    );
                    continue;
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
                            &mut stdout,
                            "",
                            format!("kld_eval: mode={mode} requires 'corpus'"),
                        );
                        continue;
                    };
                    let text = match std::fs::read_to_string(&corpus_path) {
                        Ok(t) => t,
                        Err(e) => {
                            emit_error_with_id(
                                &mut stdout,
                                "",
                                format!("kld_eval: read {corpus_path}: {e}"),
                            );
                            continue;
                        }
                    };
                    let Some(tk) = m.tokenizer.as_ref() else {
                        emit_error_with_id(
                            &mut stdout,
                            "",
                            "kld_eval: resident model has no tokenizer".to_string(),
                        );
                        continue;
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
                    if let (Some(w), Some(c)) =
                        (m.deepseek4_weights.as_ref(), m.deepseek4_config.as_ref())
                    {
                        break 'pick Some(Box::new(deepseek4::kld::DeepseekV4KldForward {
                            weights: w,
                            config: c,
                        }));
                    }
                    if let (Some(w), Some(c)) =
                        (m.minimax_weights.as_ref(), m.minimax_config.as_ref())
                    {
                        break 'pick Some(Box::new(minimax::kld::MiniMaxKldForward {
                            weights: w,
                            config: c,
                        }));
                    }
                    #[cfg(feature = "arch-lfm2moe")]
                    if let (Some(w), Some(c)) =
                        (m.lfm2moe_weights.as_ref(), m.lfm2moe_config.as_ref())
                    {
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
                            &mut stdout,
                            "",
                            format!("kld_eval: arch_id {arch_id} has no KLD-scorable backend"),
                        );
                        continue;
                    }
                };
                let n_vocab = fwd.kld_vocab_size();

                macro_rules! kld_chunk_cb {
                    () => {
                        |c, n, s, k| {
                            let _ = writeln!(
                                stdout,
                                "{}",
                                serde_json::json!({"type":"kld_chunk","chunk":c,"n_chunk":n,"scored":s,"mean_kld":k})
                            );
                            let _ = stdout.flush();
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
                        let _ = writeln!(stdout, "{resp}");
                        let _ = stdout.flush();
                    }};
                }

                match mode {
                    "self_score" | "build_ref" => {
                        if mode == "self_score" {
                            match hipfire_runtime::kld_eval::kld_self_score(
                                &mut *fwd,
                                &mut gpu,
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
                                                &mut stdout,
                                                "",
                                                format!("kld_eval: write {p}: {e}"),
                                            ),
                                        }
                                    }
                                    emit_kld_evaled!("self_score", out, seq, serde_json::json!([]));
                                }
                                Err(e) => {
                                    emit_error_with_id(&mut stdout, "", format!("kld_eval: {e}"))
                                }
                            }
                        } else {
                            let Some(ref_out) = ref_path.clone() else {
                                emit_error_with_id(
                                    &mut stdout,
                                    "",
                                    "kld_eval: build_ref requires 'ref_path'".to_string(),
                                );
                                continue;
                            };
                            match hipfire_runtime::kld_eval::kld_build_ref(
                                &mut *fwd,
                                &mut gpu,
                                &tokens,
                                n_ctx,
                                top_k,
                                max_chunks,
                                |c, n, s| {
                                    let _ = writeln!(
                                        stdout,
                                        "{}",
                                        serde_json::json!({"type":"kld_chunk","chunk":c,"n_chunk":n,"scored":s,"mean_kld":0.0})
                                    );
                                    let _ = stdout.flush();
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
                                            gpu_arch: gpu.arch.clone(),
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
                                            &mut stdout,
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
                                    let _ = writeln!(stdout, "{resp}");
                                    let _ = stdout.flush();
                                }
                                Err(e) => {
                                    emit_error_with_id(&mut stdout, "", format!("kld_eval: {e}"))
                                }
                            }
                        }
                    }
                    "score" => {
                        let Some(ref_in) = ref_path.clone() else {
                            emit_error_with_id(
                                &mut stdout,
                                "",
                                "kld_eval: score requires 'ref_path'".to_string(),
                            );
                            continue;
                        };
                        let archive = match read_kld_ref_archive(std::path::Path::new(&ref_in)) {
                            Ok(a) => a,
                            Err(e) => {
                                emit_error_with_id(
                                    &mut stdout,
                                    "",
                                    format!("kld_eval: read ref {ref_in}: {e}"),
                                );
                                continue;
                            }
                        };
                        let run = hipfire_kld::RunEnv {
                            git_commit: Some(version.clone()),
                            gpu_arch: gpu.arch.clone(),
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
                                &mut stdout,
                                "",
                                format!(
                                    "kld_eval: refusing score — ref incompatible: {}",
                                    errs.join("; ")
                                ),
                            );
                            continue;
                        }
                        let findings: Vec<String> = report
                            .mismatches
                            .iter()
                            .map(|m| format!("{:?} {}: {}", m.severity, m.field, m.detail))
                            .collect();
                        match hipfire_runtime::kld_eval::kld_score(
                            &mut *fwd,
                            &mut gpu,
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
                                            &mut stdout,
                                            "",
                                            format!("kld_eval: write {p}: {e}"),
                                        ),
                                    }
                                }
                                emit_kld_evaled!("score", out, seq, serde_json::json!(findings));
                            }
                            Err(e) => emit_error_with_id(&mut stdout, "", format!("kld_eval: {e}")),
                        }
                    }
                    other => emit_error_with_id(
                        &mut stdout,
                        "",
                        format!("kld_eval: unknown mode {other:?}"),
                    ),
                }
            }

            // Refusal-direction steering / abliteration session control. The
            // in-forward `maybe_steer_block` hook (compiled into the gemma3
            // forward) keeps a process-global capture/apply session; these arms
            // expose control over it so a client (hipfire-steer-harness) can drive
            // capture→derive→apply→score through the daemon's correct inference +
            // templating instead of a reimplemented harness. See
            // docs/plans/2026-06-30-steer-daemon-pivot.md.
            DaemonRequest::SteerBeginCapture(_) => {
                let num_layers = msg
                    .get("num_layers")
                    .and_then(|v| v.as_u64())
                    .map(|v| v as usize);
                let hidden = msg
                    .get("hidden")
                    .and_then(|v| v.as_u64())
                    .map(|v| v as usize);
                let (Some(num_layers), Some(hidden)) = (num_layers, hidden) else {
                    emit_error_with_id(
                        &mut stdout,
                        "",
                        "steer_begin_capture: missing 'num_layers'/'hidden'".to_string(),
                    );
                    continue;
                };
                hipfire_steer::begin_capture(num_layers, hidden);
                let _ = writeln!(stdout, r#"{{"type":"steer_ok"}}"#);
                let _ = stdout.flush();
            }

            // Prefill ONE chat turn through the hooked forward (no decode) and fold
            // its last-prompt-token residuals into the capture means. Prefill-only:
            // a decoded token's forward would overwrite the residual the hook just
            // recorded (the `collect` arm is prefill-only for the same reason).
            DaemonRequest::SteerCapture(_) => {
                let system = msg
                    .get("system")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let Some(user) = msg.get("user").and_then(|v| v.as_str()).map(String::from) else {
                    emit_error_with_id(
                        &mut stdout,
                        "",
                        "steer_capture: missing 'user'".to_string(),
                    );
                    continue;
                };
                let Some(m) = model.as_mut() else {
                    emit_error_with_id(
                        &mut stdout,
                        "",
                        "steer_capture: no model loaded".to_string(),
                    );
                    continue;
                };
                if m.pp != 1 {
                    emit_error_with_id(
                        &mut stdout,
                        "",
                        "steer_capture: requires a single-GPU resident model (pp == 1)".to_string(),
                    );
                    continue;
                }
                let Some(tokenizer) = m.tokenizer.as_ref() else {
                    emit_error_with_id(
                        &mut stdout,
                        "",
                        "steer_capture: resident model has no tokenizer".to_string(),
                    );
                    continue;
                };
                // Frame the turn byte-identically to the `generate` path so capture
                // sees the exact residuals serving would. gemma3 uses its literal
                // turn frame; qwen35 (loose-slot) uses its jinja `chat_template`
                // single-turn render.
                let system_opt = (!system.is_empty()).then_some(system.as_str());
                let framed = if is_qwen35_family_arch_id(m.arch_id) {
                    match hipfire_serving_core::generate_arch::framed_qwen35_prompt(
                        m, &user, system_opt,
                    ) {
                        Ok(f) => f,
                        Err(e) => {
                            emit_error_with_id(&mut stdout, "", format!("steer_capture: {e}"));
                            continue;
                        }
                    }
                } else {
                    hipfire_serving_core::generate_arch::framed_gemma3_prompt(&user, system_opt)
                };
                let tokens = tokenizer.encode(&framed);
                if tokens.is_empty() {
                    emit_error_with_id(
                        &mut stdout,
                        "",
                        "steer_capture: empty prompt after framing".to_string(),
                    );
                    continue;
                }
                // Prefill-only through whichever resident arch fires the
                // block-boundary hook so it observes the last-prompt-token residual
                // per block. No decode loop. gemma3 (12/13) folds via its backend
                // prefill; qwen35 (loose-slot) folds via a fresh single-sequence
                // capture prefill. Both hit `maybe_steer_block[_batched]`.
                use hipfire_runtime::arch::SimpleAr;
                let result: Result<(), String> = if is_qwen35_family_arch_id(m.arch_id) {
                    run_steer_capture_prefill_qwen35(m, &mut gpu, &tokens)
                } else if let Some(b) = m.gemma3_text.as_mut() {
                    b.state.reset();
                    SimpleAr::prefill(b, &mut gpu, &tokens)
                } else if let Some(b) = m.gemma3_vl.as_mut() {
                    b.state.reset();
                    SimpleAr::prefill(b, &mut gpu, &tokens)
                } else {
                    Err(format!(
                        "steer_capture: arch_id {} is unsupported (need gemma3 or qwen35)",
                        m.arch_id
                    ))
                };
                match result {
                    Ok(()) => {
                        hipfire_steer::commit_capture();
                        let _ = writeln!(stdout, r#"{{"type":"steer_ok"}}"#);
                        let _ = stdout.flush();
                    }
                    Err(e) => emit_error_with_id(&mut stdout, "", format!("steer_capture: {e}")),
                }
            }

            // End the capture session and return the per-block means as a
            // num_layers × hidden f32 matrix (the client derives directions from
            // the +/- means it collected).
            DaemonRequest::SteerFinishCapture => match hipfire_steer::finish_capture() {
                Some(means) => {
                    let resp = serde_json::json!({
                        "type": "steer_captured",
                        "means": means.0,
                    });
                    let _ = writeln!(stdout, "{resp}");
                    let _ = stdout.flush();
                }
                None => emit_error_with_id(
                    &mut stdout,
                    "",
                    "steer_finish_capture: no capture session active".to_string(),
                ),
            },

            // Begin an apply session: steer (additive) or ablate (projective) each
            // block in [layer_start, layer_end) along the per-block `directions`.
            DaemonRequest::SteerBeginApply(_) => {
                let directions: Option<Vec<Vec<f32>>> = msg
                    .get("directions")
                    .and_then(|v| v.as_array())
                    .map(|rows| {
                        rows.iter()
                            .map(|row| {
                                row.as_array()
                                    .map(|cols| {
                                        cols.iter()
                                            .filter_map(|x| x.as_f64().map(|f| f as f32))
                                            .collect()
                                    })
                                    .unwrap_or_default()
                            })
                            .collect()
                    });
                let Some(directions) = directions else {
                    emit_error_with_id(
                        &mut stdout,
                        "",
                        "steer_begin_apply: missing 'directions'".to_string(),
                    );
                    continue;
                };
                let mode = match msg.get("mode").and_then(|v| v.as_str()).unwrap_or("ablate") {
                    "steer" => hipfire_steer::SteerMode::Steer,
                    "ablate" => hipfire_steer::SteerMode::Ablate,
                    other => {
                        emit_error_with_id(
                            &mut stdout,
                            "",
                            format!("steer_begin_apply: unknown mode {other:?} (steer|ablate)"),
                        );
                        continue;
                    }
                };
                let strength = msg.get("strength").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32;
                let layer_start =
                    msg.get("layer_start").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                let layer_end = msg
                    .get("layer_end")
                    .and_then(|v| v.as_u64())
                    .map(|v| v as usize)
                    .unwrap_or(directions.len());
                hipfire_steer::begin_apply(hipfire_steer::SteerSpec {
                    directions,
                    mode,
                    strength,
                    layer_range: layer_start..layer_end,
                });
                let _ = writeln!(stdout, r#"{{"type":"steer_ok"}}"#);
                let _ = stdout.flush();
            }

            // Tear down any active steer session (back to the base model).
            DaemonRequest::SteerClear => {
                hipfire_steer::clear();
                let _ = writeln!(stdout, r#"{{"type":"steer_ok"}}"#);
                let _ = stdout.flush();
            }

            // ── H-Neurons intervention gain (arXiv 2512.01797) ──────────────
            // Set a process-global per-neuron activation gain on the resident
            // dense model: each FLAT feature index (`layer*intermediate+neuron`)
            // is scaled by `gain` in the FFN forward (prefill + decode); every
            // other neuron by 1.0. `gain == 1.0` or an empty set clears the
            // session — the identity control point of the dose-response sweep.
            DaemonRequest::HneuronIntervene(_) => {
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
                let dims = match model.as_ref() {
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
                        emit_error_with_id(
                            &mut stdout,
                            "",
                            "hneuron_intervene: no model loaded".to_string(),
                        );
                        continue;
                    }
                };
                let Some((n_layers, inter)) = dims else {
                    emit_error_with_id(
                        &mut stdout,
                        "",
                        "hneuron_intervene: no resident dense backend (llama|gemma3)".to_string(),
                    );
                    continue;
                };
                let n_intervened = indices.len();
                let result = if indices.is_empty() || (gain - 1.0).abs() < f32::EPSILON {
                    hipfire_hneurons::intervene::clear();
                    Ok(())
                } else {
                    hipfire_hneurons::intervene::begin_intervention(
                        &mut gpu, n_layers, inter, &indices, gain,
                    )
                };
                match result {
                    Ok(()) => {
                        let resp = serde_json::json!({
                            "type": "hneuron_ok",
                            "n_intervened": n_intervened,
                            "gain": gain,
                        });
                        let _ = writeln!(stdout, "{resp}");
                        let _ = stdout.flush();
                    }
                    Err(e) => {
                        emit_error_with_id(&mut stdout, "", format!("hneuron_intervene: {e:?}"))
                    }
                }
            }

            // ── H-Neurons CETT capture (arXiv 2512.01797) ───────────────────
            // Load the per-layer down_proj column norms (`‖W_down[:,j]‖`) once
            // from a compact little-endian binary produced host-side from the
            // source fp16 weights:
            //   [u32 n_layers][u32 intermediate][f32 × n_layers*intermediate].
            // Cached in `cett_colnorms` and reused for every `cett_capture`.
            DaemonRequest::CettLoadColnorms(_) => {
                let Some(path) = msg.get("path").and_then(|v| v.as_str()).map(String::from) else {
                    emit_error_with_id(
                        &mut stdout,
                        "",
                        "cett_load_colnorms: missing 'path'".to_string(),
                    );
                    continue;
                };
                let bytes = match std::fs::read(&path) {
                    Ok(b) => b,
                    Err(e) => {
                        emit_error_with_id(&mut stdout, "", format!("cett_load_colnorms: {e}"));
                        continue;
                    }
                };
                if bytes.len() < 8 {
                    emit_error_with_id(
                        &mut stdout,
                        "",
                        "cett_load_colnorms: file too short".to_string(),
                    );
                    continue;
                }
                let n_layers =
                    u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
                let inter = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;
                let want = 8 + n_layers * inter * 4;
                if bytes.len() != want {
                    emit_error_with_id(
                        &mut stdout,
                        "",
                        format!(
                            "cett_load_colnorms: size mismatch (got {} want {want})",
                            bytes.len()
                        ),
                    );
                    continue;
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
                cett_colnorms = Some(cn);
                let resp = serde_json::json!({
                    "type": "cett_ok",
                    "n_layers": n_layers,
                    "intermediate": inter,
                });
                let _ = writeln!(stdout, "{resp}");
                let _ = stdout.flush();
            }

            // Prefill (jinja-framed prompt + response) through the CETT-tapped
            // llama forward and return the per-layer mean-over-response-tokens
            // CETT feature (`[n_layers][intermediate]`). Requires a resident
            // llama backend (arch 10) and a prior `cett_load_colnorms`.
            DaemonRequest::CettCapture(_) => {
                let system = msg
                    .get("system")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let Some(user) = msg.get("user").and_then(|v| v.as_str()).map(String::from) else {
                    emit_error_with_id(&mut stdout, "", "cett_capture: missing 'user'".to_string());
                    continue;
                };
                let response = msg
                    .get("response")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let Some(colnorms) = cett_colnorms.clone() else {
                    emit_error_with_id(
                        &mut stdout,
                        "",
                        "cett_capture: no colnorms (call cett_load_colnorms first)".to_string(),
                    );
                    continue;
                };
                let Some(m) = model.as_mut() else {
                    emit_error_with_id(
                        &mut stdout,
                        "",
                        "cett_capture: no model loaded".to_string(),
                    );
                    continue;
                };
                let arch_id = m.arch_id;
                // Frame the prompt via the model's jinja chat_template, then build
                // the full [prompt ++ response] token sequence. These are all
                // immutable borrows of `m`, released before the mutable backend
                // borrow below (mirrors the steer_capture ordering).
                let framed = {
                    let Some(tokenizer) = m.tokenizer.as_ref() else {
                        emit_error_with_id(
                            &mut stdout,
                            "",
                            "cett_capture: resident model has no tokenizer".to_string(),
                        );
                        continue;
                    };
                    let Some(tmpl) = m.chat_template.as_ref() else {
                        emit_error_with_id(
                            &mut stdout,
                            "",
                            "cett_capture: model has no chat_template".to_string(),
                        );
                        continue;
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
                            emit_error_with_id(
                                &mut stdout,
                                "",
                                format!("cett_capture: jinja render: {e}"),
                            );
                            continue;
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
                    emit_error_with_id(
                        &mut stdout,
                        "",
                        "cett_capture: empty response after tokenization".to_string(),
                    );
                    continue;
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
                let outcome: Result<(Vec<Vec<f32>>, usize), String> =
                    if let Some(b) = m.llama_backend.as_mut() {
                        if colnorms.len() != b.config.n_layers {
                            Err(format!(
                                "colnorms n_layers {} != model n_layers {}",
                                colnorms.len(),
                                b.config.n_layers
                            ))
                        } else if let Err(e) = hipfire_hneurons::capture::begin_capture(
                            &mut gpu,
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
                            match SimpleAr::prefill(b, &mut gpu, &full) {
                                Ok(()) => finish(&mut gpu),
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
                            &mut gpu,
                            colnorms,
                            cap_start,
                            cap_end,
                            b.config.hidden_size,
                        ) {
                            Err(format!("begin_capture: {e:?}"))
                        } else {
                            b.state.reset();
                            match SimpleAr::prefill(b, &mut gpu, &full) {
                                Ok(()) => finish(&mut gpu),
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
                            &mut gpu,
                            colnorms,
                            cap_start,
                            cap_end,
                            b.text_cfg.hidden_size,
                        ) {
                            Err(format!("begin_capture: {e:?}"))
                        } else {
                            b.state.reset();
                            match SimpleAr::prefill(b, &mut gpu, &full) {
                                Ok(()) => finish(&mut gpu),
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
                        let _ = writeln!(stdout, "{resp}");
                        let _ = stdout.flush();
                    }
                    Err(e) => emit_error_with_id(&mut stdout, "", format!("cett_capture: {e}")),
                }
            }

            // LoRA adapter stack (shares the steer APPLY session). Load a `.lora`
            // container onto the live model, adjust per-adapter intensity, stack or
            // remove adapters, and list — all without reload. The abliteration
            // directions materialized by `lora_export`/the harness become a portable
            // adapter served here. See docs/plans/2026-06-30-abliteration-lora.md.
            DaemonRequest::LoraLoad(_) => {
                let Some(path) = msg.get("path").and_then(|v| v.as_str()).map(String::from) else {
                    emit_error_with_id(&mut stdout, "", "lora_load: missing 'path'".to_string());
                    continue;
                };
                let scale_override = msg.get("scale").and_then(|v| v.as_f64()).map(|v| v as f32);
                let id_override = msg.get("id").and_then(|v| v.as_str()).map(String::from);
                let mut adapter = match hipfire_lora_hfq::read_lora_any(std::path::Path::new(&path))
                {
                    Ok(a) => a,
                    Err(e) => {
                        emit_error_with_id(&mut stdout, "", format!("lora_load: {e}"));
                        continue;
                    }
                };
                if let Some(new_id) = id_override {
                    adapter.id = new_id;
                }
                // The adapter is base-specific (directions sized to the model's
                // hidden width); reject a mismatched load before it faults at apply.
                let model_hidden = model.as_ref().and_then(|m| {
                    m.gemma3_text
                        .as_ref()
                        .map(|b| b.config.hidden_size)
                        .or_else(|| m.gemma3_vl.as_ref().map(|b| b.text_cfg.hidden_size))
                });
                if let Some(h) = model_hidden {
                    if adapter.meta.hidden != h {
                        emit_error_with_id(
                            &mut stdout,
                            "",
                            format!(
                                "lora_load: adapter hidden {} != model hidden {h}",
                                adapter.meta.hidden
                            ),
                        );
                        continue;
                    }
                }
                let id = adapter.id.clone();
                if let Err(e) = hipfire_steer::load_lora_adapter(&adapter) {
                    emit_error_with_id(&mut stdout, "", format!("lora_load: {e}"));
                    continue;
                }
                if let Some(s) = scale_override {
                    hipfire_steer::set_adapter_scale(&id, s);
                }
                let _ = writeln!(stdout, r#"{{"type":"lora_ok"}}"#);
                let _ = stdout.flush();
            }

            DaemonRequest::LoraSetScale(_) => {
                let id = msg.get("id").and_then(|v| v.as_str()).map(String::from);
                let scale = msg.get("scale").and_then(|v| v.as_f64()).map(|v| v as f32);
                let (Some(id), Some(scale)) = (id, scale) else {
                    emit_error_with_id(
                        &mut stdout,
                        "",
                        "lora_set_scale: missing 'id'/'scale'".to_string(),
                    );
                    continue;
                };
                if hipfire_steer::set_adapter_scale(&id, scale) {
                    let _ = writeln!(stdout, r#"{{"type":"lora_ok"}}"#);
                    let _ = stdout.flush();
                } else {
                    emit_error_with_id(
                        &mut stdout,
                        "",
                        format!("lora_set_scale: no adapter {id:?} loaded"),
                    );
                }
            }

            DaemonRequest::LoraUnload(_) => {
                let Some(id) = msg.get("id").and_then(|v| v.as_str()).map(String::from) else {
                    emit_error_with_id(&mut stdout, "", "lora_unload: missing 'id'".to_string());
                    continue;
                };
                if hipfire_steer::unload_adapter(&id) {
                    let _ = writeln!(stdout, r#"{{"type":"lora_ok"}}"#);
                    let _ = stdout.flush();
                } else {
                    emit_error_with_id(
                        &mut stdout,
                        "",
                        format!("lora_unload: no adapter {id:?} loaded"),
                    );
                }
            }

            DaemonRequest::LoraClear => {
                hipfire_steer::clear();
                let _ = writeln!(stdout, r#"{{"type":"lora_ok"}}"#);
                let _ = stdout.flush();
            }

            DaemonRequest::LoraList => {
                let adapters: Vec<_> = hipfire_steer::loaded_adapters()
                    .into_iter()
                    .map(|(id, scale)| serde_json::json!({ "id": id, "scale": scale }))
                    .collect();
                let resp = serde_json::json!({ "type": "lora_listed", "adapters": adapters });
                let _ = writeln!(stdout, "{resp}");
                let _ = stdout.flush();
            }

            // PFlash drafter TEACHER: forward the resident qwen3.5 target over a
            // corpus and emit per-chunk per-block cosine-K scores at the shallow +
            // mid FullAttention layers — the labels `pflash_drafter_train` distils
            // (teacher/student split, docs/plans/2026-06-19-training-via-daemon-forward.md).
            // Output is JSONL, one line per chunk; the trainer's daemon-label
            // loader converts it to the v2 label cache.
            DaemonRequest::PflashLabels(_) => {
                let Some(corpus) = msg.get("corpus").and_then(|v| v.as_str()).map(String::from)
                else {
                    emit_error_with_id(
                        &mut stdout,
                        "",
                        "pflash_labels: missing 'corpus'".to_string(),
                    );
                    continue;
                };
                let Some(output) = msg.get("output").and_then(|v| v.as_str()).map(String::from)
                else {
                    emit_error_with_id(
                        &mut stdout,
                        "",
                        "pflash_labels: missing 'output'".to_string(),
                    );
                    continue;
                };
                let seq = msg
                    .get("seq")
                    .and_then(|v| v.as_u64())
                    .map(|v| v as usize)
                    .unwrap_or(512);
                let block = msg
                    .get("block")
                    .and_then(|v| v.as_u64())
                    .map(|v| v as usize)
                    .unwrap_or(64);
                let n_chunks = msg
                    .get("n_chunks")
                    .and_then(|v| v.as_u64())
                    .map(|v| v as usize)
                    .unwrap_or(40);
                let Some(m) = model.as_ref() else {
                    emit_error_with_id(
                        &mut stdout,
                        "",
                        "pflash_labels: no model loaded".to_string(),
                    );
                    continue;
                };
                let (Some(weights), Some(config), Some(tokenizer)) = (
                    m.q35_weights.as_ref(),
                    m.q35_config.as_ref(),
                    m.tokenizer.as_ref(),
                ) else {
                    emit_error_with_id(
                        &mut stdout,
                        "",
                        "pflash_labels: resident model is not a qwen3.5-family model".to_string(),
                    );
                    continue;
                };
                let fa = qwen35::full_attention_layers(config);
                if fa.is_empty() {
                    emit_error_with_id(
                        &mut stdout,
                        "",
                        "pflash_labels: no FullAttention layers".to_string(),
                    );
                    continue;
                }
                let shallow = fa[0];
                let mid = fa[fa.len() / 2];
                let text = match std::fs::read_to_string(&corpus) {
                    Ok(t) => t,
                    Err(e) => {
                        emit_error_with_id(
                            &mut stdout,
                            "",
                            format!("pflash_labels: read {corpus}: {e}"),
                        );
                        continue;
                    }
                };
                let all = tokenizer.encode(&text);
                if all.len() < n_chunks * seq {
                    emit_error_with_id(
                        &mut stdout,
                        "",
                        format!(
                            "pflash_labels: corpus too small: {} toks < {}",
                            all.len(),
                            n_chunks * seq
                        ),
                    );
                    continue;
                }
                let mut out_file = match std::fs::File::create(&output) {
                    Ok(f) => std::io::BufWriter::new(f),
                    Err(e) => {
                        emit_error_with_id(
                            &mut stdout,
                            "",
                            format!("pflash_labels: create {output}: {e}"),
                        );
                        continue;
                    }
                };
                let mut failed = false;
                for ci in 0..n_chunks {
                    let toks = all[ci * seq..(ci + 1) * seq].to_vec();
                    match qwen35::capture_pflash_block_scores(
                        &mut gpu,
                        weights,
                        config,
                        &toks,
                        block,
                        &[shallow, mid],
                    ) {
                        Ok(scores) => {
                            let line = serde_json::json!({
                                "chunk": ci,
                                "tokens": toks,
                                "shallow_scores": scores[0],
                                "mid_scores": scores[1],
                            });
                            if writeln!(out_file, "{line}").is_err() {
                                failed = true;
                                break;
                            }
                        }
                        Err(e) => {
                            emit_error_with_id(
                                &mut stdout,
                                "",
                                format!("pflash_labels: chunk {ci}: {e}"),
                            );
                            failed = true;
                            break;
                        }
                    }
                }
                use std::io::Write as _;
                let _ = out_file.flush();
                if failed {
                    continue;
                }
                // Dump the shared fp32 embedding once (the drafter shares it RO).
                let embed_path = format!("{output}.embed.bin");
                let embed_dims = match qwen35::dump_embed_fp32(
                    &mut gpu,
                    weights,
                    config,
                    std::path::Path::new(&embed_path),
                ) {
                    Ok(d) => Some(d),
                    Err(e) => {
                        emit_error_with_id(&mut stdout, "", format!("pflash_labels: embed: {e}"));
                        None
                    }
                };
                let resp = serde_json::json!({
                    "type": "pflash_labels",
                    "output": output,
                    "embed": embed_dims.map(|_| embed_path.clone()),
                    "embed_vocab": embed_dims.map(|(v, _)| v),
                    "embed_dim": embed_dims.map(|(_, d)| d),
                    "n_chunks": n_chunks,
                    "seq": seq,
                    "block": block,
                    "shallow_layer": shallow,
                    "mid_layer": mid,
                });
                let _ = writeln!(stdout, "{resp}");
                let _ = stdout.flush();
            }

            // Train a PFlash importance-scorer drafter in-process against the
            // resident target (teacher/student split). STEP 1: plumbing only —
            // validates args + the hipfire-train link; the loop wiring lands in
            // step 3. See docs/plans/2026-06-19-train-as-daemon-op.md.
            DaemonRequest::TrainDrafter => {
                // Micro-step-PREEMPTIBLE SSM-drafter training as a daemon op. Runs
                // up to `quantum` EPOCHS per request and keeps a resident
                // DrafterTrainSession alive between requests (keyed by `run_id`);
                // the runner re-enqueues the low-priority training lease each
                // quantum so it time-slices with interactive serving. Numerics are
                // verbatim from the whole-run loop (drafter_loop_init/run_epochs/
                // finish reproduce train_ssm_drafter_loop). Per-eval-epoch stream
                // uses type `train_epoch` (not `train_progress`) so the runner's
                // adapter only sees ONE quantum-boundary `train_progress`/`train_done`.
                let run_id = msg
                    .get("run_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let t = msg
                    .get("train")
                    .cloned()
                    .unwrap_or_else(|| serde_json::json!({}));
                let quantum = msg
                    .get("quantum")
                    .or_else(|| t.get("quantum"))
                    .and_then(|v| v.as_u64())
                    .map(|v| v as usize)
                    .unwrap_or(25)
                    .max(1);

                // CONTINUE the resident session iff its run_id matches; else START
                // fresh (loading labels + building drafter/optimizer once).
                let continue_run = !run_id.is_empty()
                    && drafter_train_session
                        .as_ref()
                        .map(|s| s.run_id == run_id)
                        .unwrap_or(false);
                if !continue_run {
                    drafter_train_session = None; // drop any stale session, free VRAM
                    let arch = msg
                        .get("arch")
                        .and_then(|v| v.as_str())
                        .unwrap_or("ssm")
                        .to_string();
                    if arch != "ssm" {
                        emit_error_with_id(
                            &mut stdout,
                            "",
                            format!(
                                "train_drafter: arch '{arch}' not implemented (only ssm; step 3)"
                            ),
                        );
                        continue;
                    }
                    // Parse the train/labels blocks into the SHARED TrainCfg.
                    let labels = msg
                        .get("labels")
                        .cloned()
                        .unwrap_or_else(|| serde_json::json!({}));
                    let getu = |o: &serde_json::Value, k: &str, d: usize| -> usize {
                        o.get(k)
                            .and_then(|v| v.as_u64())
                            .map(|v| v as usize)
                            .unwrap_or(d)
                    };
                    let getf = |o: &serde_json::Value, k: &str, d: f32| -> f32 {
                        o.get(k)
                            .and_then(|v| v.as_f64())
                            .map(|v| v as f32)
                            .unwrap_or(d)
                    };
                    let cfg = hipfire_train::train_loop::TrainCfg {
                        seq: getu(&labels, "seq", 512),
                        block: getu(&labels, "block", 64),
                        n_eval: getu(&labels, "n_eval", 20),
                        epochs: getu(&t, "epochs", 300),
                        lr: getf(&t, "lr", 1e-3),
                        wd: getf(&t, "wd", 0.0),
                        tau: getf(&t, "tau", 0.1),
                        eval_every: getu(&t, "eval_every", 15),
                        report_train: t
                            .get("report_train")
                            .and_then(|v| v.as_bool())
                            .unwrap_or(false),
                    };
                    let source = labels
                        .get("source")
                        .and_then(|v| v.as_str())
                        .unwrap_or("file");
                    if source != "file" {
                        emit_error_with_id(
                            &mut stdout,
                            "",
                            format!("train_drafter: label source '{source}' not implemented (only file; capture is step 4)"),
                        );
                        continue;
                    }
                    let Some(path) = labels
                        .get("path")
                        .and_then(|v| v.as_str())
                        .map(String::from)
                    else {
                        emit_error_with_id(
                            &mut stdout,
                            "",
                            "train_drafter: labels.path required for source=file".to_string(),
                        );
                        continue;
                    };
                    let Some(output) = msg.get("output").and_then(|v| v.as_str()).map(String::from)
                    else {
                        emit_error_with_id(
                            &mut stdout,
                            "",
                            "train_drafter: 'output' (checkpoint path) required".to_string(),
                        );
                        continue;
                    };

                    // ── load cached labels + frozen target embedding (file source) ──
                    let mut ls =
                        match hipfire_train::labels::load_daemon_labels(&mut gpu, &path, cfg.seq) {
                            Ok(ls) => ls,
                            Err(e) => {
                                emit_error_with_id(
                                    &mut stdout,
                                    "",
                                    format!("train_drafter: load labels {path}: {e}"),
                                );
                                continue;
                            }
                        };
                    let shuffle_seed = getu(&labels, "shuffle_seed", 0x5EED) as u64;
                    hipfire_train::labels::shuffle_in_place(
                        &mut ls.chunks,
                        &mut ls.label_mid,
                        &mut ls.base_shallow,
                        shuffle_seed,
                    );

                    // ── build the SSM drafter from the request config ──
                    let dc = msg
                        .get("config")
                        .cloned()
                        .unwrap_or_else(|| serde_json::json!({}));
                    let mut dcfg =
                        hipfire_train::ssm_drafter::SsmDrafterConfig::tiny(10000.0, 1e-5);
                    dcfg.h_draft = getu(&dc, "h_draft", 512);
                    dcfg.n_layers = getu(&dc, "n_layers", 3);
                    dcfg.inter = getu(&dc, "inter", 1024);
                    dcfg.n_kv = getu(&dc, "n_kv", 4);
                    dcfg.head_dim = getu(&dc, "head_dim", 64);
                    let (h_t, vocab) = (ls.h_t, ls.vocab);
                    let drafter = match hipfire_train::ssm_drafter::SsmDrafter::new(
                        &mut gpu, ls.embed, h_t, vocab, dcfg, cfg.seq,
                    ) {
                        Ok(d) => d,
                        Err(e) => {
                            emit_error_with_id(
                                &mut stdout,
                                "",
                                format!("train_drafter: build drafter: {e}"),
                            );
                            continue;
                        }
                    };
                    // Set up the resumable loop state (bar, optimizer, scores scratch).
                    let st = match hipfire_train::train_loop::drafter_loop_init(
                        &mut gpu,
                        &drafter,
                        &ls.chunks,
                        &ls.label_mid,
                        &ls.base_shallow,
                        &cfg,
                    ) {
                        Ok(s) => s,
                        Err(e) => {
                            emit_error_with_id(
                                &mut stdout,
                                "",
                                format!("train_drafter: loop init: {e}"),
                            );
                            continue;
                        }
                    };
                    let nparams: usize = drafter.param_sizes().iter().sum();
                    let _ = writeln!(
                        stdout,
                        "{}",
                        serde_json::json!({
                            "type": "train_start", "arch": arch, "params": nparams,
                            "chunks": ls.chunks.len(), "n_train": ls.chunks.len().saturating_sub(cfg.n_eval),
                            "n_eval": cfg.n_eval, "epochs": cfg.epochs,
                            "run_id": run_id, "quantum": quantum,
                        })
                    );
                    let _ = stdout.flush();
                    drafter_train_session = Some(DrafterTrainSession {
                        run_id: run_id.clone(),
                        drafter,
                        chunks: ls.chunks,
                        label_mid: ls.label_mid,
                        cfg,
                        st,
                        output,
                        quantum,
                    });
                }

                // ── run ONE quantum of epochs, streaming per-epoch `train_epoch` ──
                let quantum_result: Result<(), String> = {
                    let sess = drafter_train_session
                        .as_mut()
                        .expect("session present after start/continue");
                    let ep_end = (sess.st.ep + sess.quantum).min(sess.cfg.epochs);
                    hipfire_train::train_loop::drafter_loop_run_epochs(
                        &mut gpu,
                        &sess.drafter,
                        sess.chunks.as_slice(),
                        sess.label_mid.as_slice(),
                        &sess.cfg,
                        &mut sess.st,
                        ep_end,
                        |ep, train_loss, corr, best, best_ep, train_corr| {
                            let mut ev = serde_json::json!({
                                "type": "train_epoch", "epoch": ep, "train_loss": train_loss,
                                "eval": corr, "best": best, "best_epoch": best_ep,
                            });
                            if let Some(tc) = train_corr {
                                ev["train_rho"] = serde_json::json!(tc);
                            }
                            let _ = writeln!(stdout, "{ev}");
                            let _ = stdout.flush();
                        },
                    )
                    .map_err(|e| e.to_string())
                };
                if let Err(e) = quantum_result {
                    drafter_train_session = None;
                    emit_error_with_id(&mut stdout, "", format!("train_drafter: train loop: {e}"));
                    continue;
                }

                let done = drafter_train_session
                    .as_ref()
                    .map(|s| s.st.ep >= s.cfg.epochs)
                    .unwrap_or(false);
                if done {
                    // Final quantum: finish (free scratch) → checkpoint best-eval
                    // weights → terminal event. `take()` drops the resident session.
                    let sess = drafter_train_session.take().expect("done implies present");
                    let output = sess.output.clone();
                    let run_id = sess.run_id.clone();
                    let report = hipfire_train::train_loop::drafter_loop_finish(&mut gpu, sess.st);
                    let saved = hipfire_train::labels::save_ssm_drafter_weights(
                        &output,
                        &report.best_weights,
                        report.best_epoch as u32,
                    );
                    let _ = writeln!(
                        stdout,
                        "{}",
                        serde_json::json!({
                            "type": "train_done",
                            "best_eval": report.best_eval, "best_epoch": report.best_epoch,
                            "bar": report.bar, "final_eval": report.final_eval,
                            "beat_bar": report.best_eval > report.bar,
                            "checkpoint": if saved.is_ok() { Some(output.clone()) } else { None },
                            "checkpoint_error": saved.err().map(|e| e.to_string()),
                            "run_id": run_id,
                        })
                    );
                    let _ = stdout.flush();
                } else {
                    // Quantum done but run unfinished: report progress and keep the
                    // session resident. The runner re-enqueues; training yields to
                    // any pending interactive request before the next quantum.
                    let sess = drafter_train_session
                        .as_ref()
                        .expect("unfinished implies present");
                    let _ = writeln!(
                        stdout,
                        "{}",
                        serde_json::json!({
                            "type": "train_progress", "run_id": sess.run_id,
                            "epoch": sess.st.ep, "total": sess.cfg.epochs,
                            "eval": sess.st.final_eval, "best": sess.st.best_eval,
                            "done": false,
                        })
                    );
                    let _ = stdout.flush();
                }
            }

            // Train a LoRA adapter on a frozen bf16 base, in-process on the
            // resident engine. SCAFFOLD: this validates args + the hipfire-train
            // link and reserves the wire/runner/route path (mirrors TrainDrafter),
            // but the ASSEMBLED, data-driven, adapter-SAVING LoRA loop is not yet
            // wired — hipfire-train has the proven primitives (model::from_f32_weights
            // → model_forward → model_loss_backward → optim::AdamW, demonstrated in
            // examples/overfit_supra50m.rs) but no reusable loop that loads real
            // data/labels and serializes an adapter checkpoint. Emits a clear
            // not-implemented error until that lands.
            //
            // NOTE: even when assembled, this trains hipfire-train's OWN un-fused
            // LlamaModel — NOT the served qwen35 arch's adapters. Training the
            // served forward via activation-save at the in-forward HOOK sites is
            // the large P3 follow-on. See docs/plans/2026-07-19-daemon-training-steering.md.
            DaemonRequest::TrainLora => {
                // Micro-step-PREEMPTIBLE LoRA-on-frozen training as a daemon op.
                // Runs up to `quantum` steps per request and keeps a resident
                // LoraTrainSession alive between requests (keyed by `run_id`); the
                // runner re-enqueues the low-priority training lease each quantum so
                // it time-slices with interactive serving. Compute is verbatim from
                // the validated whole-run loop (hipfire_train, overfit_supra50m.rs):
                // forward → loss → backward-THROUGH-ADAPTERS → AdamW, then a final
                // HFLORA01 adapter dump. NOTE: trains hipfire-train's own un-fused
                // LlamaModel, NOT the served qwen35 adapters (a follow-on).
                // `data=overfit` is a deterministic synthetic batch.
                const IGNORE: i32 = -100;
                let run_id = msg
                    .get("run_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let train = msg
                    .get("train")
                    .cloned()
                    .unwrap_or_else(|| serde_json::json!({}));
                let quantum = msg
                    .get("quantum")
                    .or_else(|| train.get("quantum"))
                    .and_then(|v| v.as_u64())
                    .map(|v| v as usize)
                    .unwrap_or(25)
                    .max(1);

                // CONTINUE the resident session iff its run_id matches; else START
                // fresh (loading the model + building the batch/optimizer once).
                let continue_run = !run_id.is_empty()
                    && lora_train_session
                        .as_ref()
                        .map(|s| s.run_id == run_id)
                        .unwrap_or(false);
                if !continue_run {
                    lora_train_session = None; // drop any stale session, free VRAM
                    let Some(output) = msg.get("output").and_then(|v| v.as_str()).map(String::from)
                    else {
                        emit_error_with_id(
                            &mut stdout,
                            "",
                            "train_lora: 'output' (adapter checkpoint path) required".to_string(),
                        );
                        continue;
                    };
                    let Some(base_dir) = msg
                        .get("model")
                        .or_else(|| msg.get("base"))
                        .and_then(|v| v.as_str())
                        .map(String::from)
                    else {
                        emit_error_with_id(
                            &mut stdout,
                            "",
                            "train_lora: 'model' (fp32 base model dir) required".to_string(),
                        );
                        continue;
                    };
                    let getu = |k: &str, d: usize| -> usize {
                        train
                            .get(k)
                            .and_then(|v| v.as_u64())
                            .map(|v| v as usize)
                            .unwrap_or(d)
                    };
                    let getf = |k: &str, d: f32| -> f32 {
                        train
                            .get(k)
                            .and_then(|v| v.as_f64())
                            .map(|v| v as f32)
                            .unwrap_or(d)
                    };
                    let steps = getu("steps", 200);
                    let rank = getu("rank", 16);
                    let seq = getu("seq", 8);
                    let n_seqs = getu("n_seqs", 3);
                    let alpha = getf("alpha", 32.0);
                    let lr = getf("lr", 5e-3);
                    let data_mode = msg
                        .get("data")
                        .and_then(|v| v.as_str())
                        .unwrap_or("overfit");
                    if data_mode != "overfit" {
                        emit_error_with_id(
                            &mut stdout,
                            "",
                            format!("train_lora: data source '{data_mode}' not implemented (only 'overfit' synthetic batch is wired; real-corpus loading is a follow-on)"),
                        );
                        continue;
                    }
                    let _ = writeln!(
                        stdout,
                        "{}",
                        serde_json::json!({
                            "type": "train_start", "op": "train_lora", "base": base_dir,
                            "steps": steps, "rank": rank, "alpha": alpha, "lr": lr,
                            "run_id": run_id, "quantum": quantum,
                        })
                    );
                    let _ = stdout.flush();
                    let built: Result<LoraTrainSession, String> = (|| {
                        let dir = std::path::Path::new(&base_dir);
                        if !dir.exists() {
                            return Err(format!("base model dir not found: {base_dir}"));
                        }
                        let (cfg, weights) = hipfire_train::loader::load_llama_fp32(&mut gpu, dir)
                            .map_err(|e| e.to_string())?;
                        let vocab = cfg.vocab_size;
                        let model = hipfire_train::model::LlamaModel::from_f32_weights(
                            &mut gpu, &cfg, weights, seq, rank, alpha,
                        )
                        .map_err(|e| e.to_string())?;
                        let pos: Vec<f32> = (0..seq).map(|t| t as f32).collect();
                        let batch: Vec<(Vec<u32>, Vec<f32>)> = (0..n_seqs)
                            .map(|s| {
                                let toks: Vec<u32> = (0..seq)
                                    .map(|t| (((t + 1) * 2654435761 + s * 40503) % vocab) as u32)
                                    .collect();
                                let mut tgts: Vec<f32> =
                                    (0..seq).map(|t| toks[(t + 1) % seq] as f32).collect();
                                tgts[seq - 1] = IGNORE as f32;
                                (toks, tgts)
                            })
                            .collect();
                        let target_tokens = (n_seqs * (seq - 1)).max(1) as f32;
                        let sizes = model.lora_param_sizes();
                        let opt = hipfire_train::optim::AdamW::new(
                            &mut gpu, &sizes, lr, 0.9, 0.999, 1e-8, 0.0,
                        )
                        .map_err(|e| e.to_string())?;
                        // total = steps + 1: the final pass is eval-only (no update),
                        // matching the validated whole-run `for step in 0..=steps`.
                        Ok(LoraTrainSession {
                            run_id: run_id.clone(),
                            model,
                            opt,
                            batch,
                            pos,
                            target_tokens,
                            step: 0,
                            total: steps + 1,
                            initial_ce: 0.0,
                            last_ce: 0.0,
                            output,
                            vocab,
                        })
                    })();
                    match built {
                        Ok(sess) => lora_train_session = Some(sess),
                        Err(e) => {
                            emit_error_with_id(&mut stdout, "", format!("train_lora: {e}"));
                            continue;
                        }
                    }
                }

                // Run ONE quantum of steps on the resident session. Destructure the
                // &mut session into disjoint field bindings so the per-step
                // forward/backward (reads `model`) and `opt.step` (mut `opt`) don't
                // trip the borrow checker through a single `sess`.
                let quantum_result: Result<(), String> = {
                    let sess = lora_train_session
                        .as_mut()
                        .expect("session present after start/continue");
                    let LoraTrainSession {
                        model,
                        opt,
                        batch,
                        pos,
                        target_tokens,
                        step,
                        total,
                        initial_ce,
                        last_ce,
                        ..
                    } = sess;
                    (|| {
                        let end = (*step + quantum).min(*total);
                        while *step < end {
                            let s = *step;
                            let mut total_loss = 0.0f32;
                            for (toks, tgts) in batch.iter() {
                                let acts = hipfire_train::model::model_forward(
                                    &mut gpu,
                                    &*model,
                                    toks,
                                    pos.as_slice(),
                                )
                                .map_err(|e| e.to_string())?;
                                let (loss, grads) = hipfire_train::model::model_loss_backward(
                                    &mut gpu, &*model, &acts, tgts, IGNORE,
                                )
                                .map_err(|e| e.to_string())?;
                                total_loss += loss;
                                // Last pass (step == total-1) is eval-only.
                                if s < *total - 1 {
                                    let params = model.lora_params();
                                    let gflat = hipfire_train::model::flatten_lora_grads(&grads);
                                    opt.step(&mut gpu, &params, &gflat)
                                        .map_err(|e| e.to_string())?;
                                }
                                // Free per-step activations + grads. model_forward /
                                // model_loss_backward allocate fresh GPU scratch each
                                // step and neither frees it; without this the resident
                                // session leaks VRAM across steps → OOM after a few
                                // hundred steps (the overfit example only "works"
                                // because it runs alone on a big-VRAM box).
                                hipfire_train::model::free_model_acts(&mut gpu, acts)
                                    .map_err(|e| e.to_string())?;
                                for g in grads {
                                    gpu.free_tensor(g.daq).map_err(|e| e.to_string())?;
                                    gpu.free_tensor(g.dbq).map_err(|e| e.to_string())?;
                                    gpu.free_tensor(g.dav).map_err(|e| e.to_string())?;
                                    gpu.free_tensor(g.dbv).map_err(|e| e.to_string())?;
                                    gpu.free_tensor(g.dnorm1).map_err(|e| e.to_string())?;
                                    gpu.free_tensor(g.dnorm2).map_err(|e| e.to_string())?;
                                }
                            }
                            *last_ce = total_loss / *target_tokens;
                            if s == 0 {
                                *initial_ce = *last_ce;
                            }
                            *step += 1;
                        }
                        Ok(())
                    })()
                };
                if let Err(e) = quantum_result {
                    lora_train_session = None;
                    emit_error_with_id(&mut stdout, "", format!("train_lora: {e}"));
                    continue;
                }

                let done = lora_train_session
                    .as_ref()
                    .map(|s| s.step >= s.total)
                    .unwrap_or(false);
                if done {
                    // Final quantum: dump the adapter and finish. `take()` drops the
                    // resident session (frees VRAM) before we emit the terminal event.
                    let sess = lora_train_session.take().expect("done implies present");
                    // Persist the trained adapter: layer-major [aq,bq,av,bv] f32
                    // tensors. Minimal container (magic + count + per-tensor
                    // shape/data) — a serving-loadable format is a follow-on.
                    let dump: Result<usize, String> = (|| {
                        let params = sess.model.lora_params();
                        let mut buf: Vec<u8> = Vec::new();
                        buf.extend_from_slice(b"HFLORA01");
                        buf.extend_from_slice(&(params.len() as u32).to_le_bytes());
                        for t in &params {
                            let data = gpu.download_f32(t).map_err(|e| e.to_string())?;
                            buf.extend_from_slice(&(t.shape.len() as u32).to_le_bytes());
                            for &d in &t.shape {
                                buf.extend_from_slice(&(d as u32).to_le_bytes());
                            }
                            buf.extend_from_slice(&(data.len() as u32).to_le_bytes());
                            for &f in &data {
                                buf.extend_from_slice(&f.to_le_bytes());
                            }
                        }
                        std::fs::write(&sess.output, &buf)
                            .map_err(|e| format!("write adapter {}: {e}", sess.output))?;
                        Ok(params.len())
                    })();
                    match dump {
                        Ok(n_trainable) => {
                            let _ = writeln!(
                                stdout,
                                "{}",
                                serde_json::json!({
                                    "type": "train_done", "op": "train_lora",
                                    "initial_per_tok_ce": sess.initial_ce,
                                    "final_per_tok_ce": sess.last_ce,
                                    "steps": sess.total - 1, "trainable_tensors": n_trainable,
                                    "baseline_ce_ln_vocab": (sess.vocab as f32).ln(),
                                    "output": sess.output, "run_id": sess.run_id,
                                    "note": "trained hipfire-train LlamaModel LoRA (overfit synthetic batch); served-qwen35 adapters + real-corpus loading are follow-ons",
                                })
                            );
                            let _ = stdout.flush();
                        }
                        Err(e) => emit_error_with_id(&mut stdout, "", format!("train_lora: {e}")),
                    }
                } else {
                    // Quantum done but run unfinished: report progress and keep the
                    // session resident. The runner re-enqueues; training yields to
                    // any pending interactive request before the next quantum.
                    let sess = lora_train_session
                        .as_ref()
                        .expect("unfinished implies present");
                    let _ = writeln!(
                        stdout,
                        "{}",
                        serde_json::json!({
                            "type": "train_progress", "run_id": sess.run_id,
                            "step": sess.step, "total": sess.total,
                            "per_tok_ce": sess.last_ce, "done": false,
                        })
                    );
                    let _ = stdout.flush();
                }
            }

            DaemonRequest::Diag => {
                let (vram_free, vram_total) = gpu.hip.get_vram_info().unwrap_or((0, 0));
                let hip_ver = gpu.hip.runtime_version().unwrap_or((0, 0));
                let has_model = model.is_some();
                let model_arch = model
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
                        e.parent()
                            .map(|p| p.join("kernels").join("compiled").join(&gpu.arch))
                    })
                    .filter(|p| p.is_dir());
                let (hsaco_count, hash_count) = kernel_dir
                    .map(|d| {
                        let hsaco = std::fs::read_dir(&d)
                            .map(|r| {
                                r.filter(|e| {
                                    e.as_ref()
                                        .ok()
                                        .map(|e| {
                                            e.path()
                                                .extension()
                                                .map(|x| x == "hsaco")
                                                .unwrap_or(false)
                                        })
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
                                        .map(|e| {
                                            e.path()
                                                .extension()
                                                .map(|x| x == "hash")
                                                .unwrap_or(false)
                                        })
                                        .unwrap_or(false)
                                })
                                .count()
                            })
                            .unwrap_or(0);
                        (hsaco, hash)
                    })
                    .unwrap_or((0, 0));
                let _ = writeln!(
                    stdout,
                    r#"{{"type":"diag","arch":"{}","hip_version":"{}.{}","vram_free_mb":{},"vram_total_mb":{},"model_loaded":{},"model_arch":"{}","kernels":{},"kernel_hashes":{}}}"#,
                    gpu.arch,
                    hip_ver.0,
                    hip_ver.1,
                    vram_free / (1024 * 1024),
                    vram_total / (1024 * 1024),
                    has_model,
                    model_arch,
                    hsaco_count,
                    hash_count
                );
                let _ = stdout.flush();
            }

            DaemonRequest::BenchPrefill(_) => {
                // Synthetic prefill benchmark — measures forward_prefill_batch on N
                // deterministic tokens from a zeroed state. Used by `hipfire bench`
                // to produce canonical pp128/pp512/pp1024 numbers that don't depend
                // on the user's prompt tokenizing to a round number.
                let m = match model.as_mut() {
                    Some(m) => m,
                    None => {
                        let _ =
                            writeln!(stdout, r#"{{"type":"error","message":"no model loaded"}}"#);
                        let _ = stdout.flush();
                        continue;
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
                        stdout,
                        r#"{{"type":"error","message":"bench_prefill requires pp=1 (multi-GPU bench not implemented)"}}"#
                    );
                    let _ = stdout.flush();
                    continue;
                }
                let n = msg.get("tokens").and_then(|v| v.as_u64()).unwrap_or(128) as usize;
                // Guard physical_cap — reserve 32 slots of headroom so a subsequent
                // generate request against the loaded model still has room. We guard
                // on the *physical* buffer (not the advertised max_seq) because this
                // bench intentionally bypasses eviction to measure raw prefill.
                if n + 32 > m.physical_cap {
                    let _ = writeln!(
                        stdout,
                        r#"{{"type":"error","message":"bench_prefill tokens={} exceeds loaded physical_cap={}"}}"#,
                        n, m.physical_cap
                    );
                    let _ = stdout.flush();
                    continue;
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
                        let _ = gpu.hip.memset(&s.buf, 0, s.buf.size());
                    }
                    for s in &dn.s_scales {
                        let _ = gpu.hip.memset(&s.buf, 0, s.buf.size());
                    }
                    for s in &dn.conv_states {
                        let _ = gpu.hip.memset(&s.buf, 0, s.buf.size());
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
                    let _ = s.reset(&mut gpu);
                }

                // Flush any residual GPU work so it doesn't bleed into the
                // measured interval, then time forward_prefill_batch + a
                // trailing device_synchronize so we capture actual GPU
                // completion (kernel launches are async by default).
                let _ = gpu.hip.device_synchronize();
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
                        &mut gpu, weights, config, &synthetic, 0, kv, dn, scratch, None, None,
                        None, None,
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
                        if qwen2::forward_step(&mut gpu, weights, config, state, tok).is_err() {
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
                            config, weights, state, &mut gpu, tok, i as u32,
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
                            config, weights, state, &mut gpu, tok, i as u32,
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
                                config, weights, state, &mut gpu, tok, i as u32,
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
                        if backend.decode_step(&mut gpu, tok, i).is_err() {
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
                let _ = gpu.hip.device_synchronize();
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
                        let _ = gpu.hip.memset(&s.buf, 0, s.buf.size());
                    }
                    for s in &dn.s_scales {
                        let _ = gpu.hip.memset(&s.buf, 0, s.buf.size());
                    }
                    for s in &dn.conv_states {
                        let _ = gpu.hip.memset(&s.buf, 0, s.buf.size());
                    }
                }

                if run_ok {
                    let tok_s = if elapsed > 0.0 {
                        n as f64 / elapsed
                    } else {
                        0.0
                    };
                    let _ = writeln!(
                        stdout,
                        r#"{{"type":"prefill_result","tokens":{},"ms":{:.2},"tok_s":{:.1}}}"#,
                        n,
                        elapsed * 1000.0,
                        tok_s
                    );
                } else {
                    let _ = writeln!(
                        stdout,
                        r#"{{"type":"error","message":"bench_prefill forward failed"}}"#
                    );
                }
                let _ = stdout.flush();
            }

            DaemonRequest::Profile => {
                // Precompile kernels for common configurations so we have something to profile.
                // If a model is loaded its kernels are already compiled; this fills in the rest.
                // Cover all KV modes × weight formats × head_dims to catch all kernel variants.
                #[cfg(feature = "deltanet")]
                for kv in &["q8"] {
                    for wq in &["hfq4", "hfq6", "q8"] {
                        for hd in &[128usize, 256] {
                            let _ = gpu.precompile_qwen35(wq, kv, *hd);
                        }
                    }
                }
                let (cap, kernels) = gpu.profile();
                let kernels_json: Vec<String> = kernels.iter().map(|k| k.to_json()).collect();
                let _ = writeln!(
                    stdout,
                    r#"{{"type":"profile","gpu":{},"kernels":[{}]}}"#,
                    cap.to_json(),
                    kernels_json.join(",")
                );
                let _ = stdout.flush();
            }

            DaemonRequest::Abort(_) | DaemonRequest::ForceAnswer(_) => {
                emit_error_with_id(
                    &mut stdout,
                    msg.get("id").and_then(|v| v.as_str()).unwrap_or(""),
                    format!(
                        "{msg_type} is handled on the control channel, not the request channel"
                    ),
                );
            }
        }
    }
}
