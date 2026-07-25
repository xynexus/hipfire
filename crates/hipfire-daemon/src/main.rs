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

// Several of these imports are consumed by the `handlers` submodules via
// `use crate::*` rather than by this file directly, so they are `pub(crate)`
// re-exports: the crate root is deliberately the single place the arch aliases
// and shared helper sets are named.
pub(crate) use hipfire_arch_deepseek4 as deepseek4;
#[cfg(feature = "arch-lfm2moe")]
pub(crate) use hipfire_arch_lfm2moe as lfm2moe;
pub(crate) use hipfire_arch_minimax as minimax;
pub(crate) use hipfire_arch_qwen2::qwen2;
pub(crate) use hipfire_arch_qwen35::qwen35;
#[cfg(test)]
use hipfire_generate::validate_qwen35_fused_dense_prefill_batch_preflight;
pub(crate) use hipfire_generate::{
    validate_generate_batch_decode, validate_generate_batch_prefill,
    validate_prefix_hash_preflight, GenerateVLParams, ImageSource,
};
pub(crate) use hipfire_model::{
    build_local_llm_registry, is_qwen35_family_arch_id, ARCH_ID_DEEPSEEK4_FLASH, ARCH_ID_DOTS_OCR,
    ARCH_ID_EMBEDDINGGEMMA, ARCH_ID_GEMMA3_VL, ARCH_ID_LFM2_MOE, ARCH_ID_MINIMAX_M2, ARCH_ID_QWEN2,
};
pub(crate) use hipfire_prompt as prompt_frame;
pub(crate) use hipfire_state::{
    described_sequence_state_json, model_worker_runtime_view_json,
    parse_describe_sequence_state_request, parse_release_sequence_state_request,
    parse_release_sessions_request, parse_reserve_session_state_request,
    parse_unload_worker_request, parsed_handle_may_target_generic, release_sessions_done_json,
    release_state_done_json, reserve_session_state_done_json, reserve_session_state_rejected_json,
    sequence_state_reservation_plan, sequence_state_reservation_plan_for_reserved_bytes,
    session_state_reservation_describe_json, unload_worker_done_json,
};
#[cfg(test)]
use hipfire_state::{
    generic_state_reservation_descriptors, parse_reserve_session_state_kinds,
    parse_sequence_state_handle, sequence_state_handle_id, sequence_state_handle_parts,
    sequence_state_page_descriptor_json, GenericSequenceStateArena, SequenceStateHandle,
};
use std::io::Write;
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

/// Emit a `load_progress` frame to the caller that asked for the load.
///
/// This used to take a fresh `std::io::stdout()` lock so it could be a plain free
/// fn callable from the progress-sink closure. That was fine while stdout was the
/// only place a reply could go — and silently wrong the moment it was not: on a
/// socket connection the frames went to the daemon's own stdout and the client
/// loading the model saw no progress at all. It now writes to the requesting
/// connection's sink, and carries the request id like every other frame.
fn emit_load_progress(
    sink: &mut dyn std::io::Write,
    id: &str,
    current: u32,
    total: u32,
    phase: &str,
) {
    let frame = serde_json::json!({
        "type": "load_progress",
        "id": id,
        "current": current,
        "total": total,
        "phase": phase,
    });
    let _ = writeln!(sink, "{frame}");
    let _ = sink.flush();
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

mod handlers;
mod state;
mod transport;
use state::DaemonState;

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
               --listen [PATH]     serve on a unix socket instead of stdin/stdout\n\
                                   (default ~/.hipfire/daemon.sock, mode 0600)\n\
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
    // `--listen` optionally takes a path; anything starting with `-` after it is
    // the next flag, not a socket path.
    let listen_path: Option<std::path::PathBuf> =
        args.iter().position(|a| a == "--listen").map(|index| {
            args.get(index + 1)
                .filter(|next| !next.starts_with('-'))
                .map(std::path::PathBuf::from)
                .unwrap_or_else(transport::default_socket_path)
        });

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

    let gpu = match hipfire_rdna::Gpu::init() {
        Ok(g) => g,
        Err(e) => {
            report_gpu_init_failure(&e);
            std::process::exit(1);
        }
    };
    // Every field below used to be a separate `let mut` local here. They are one
    // struct so handlers can be extracted taking `&mut DaemonState`; see
    // `state.rs` for why that ownership, not the read loop, is what serialises
    // the daemon.
    let mut daemon_state = DaemonState::new(gpu);
    if let Err(err) = daemon_state.reacquire_reservations() {
        hipfire_daemon_adapter::fatal_startup_error(
            &format!("failed to claim configured resource reservations: {err}"),
            None,
        );
    }

    // Reading happens on its own thread(s); this loop is the executor and owns the
    // GPU. See `transport` for why the split matters and why execution stays on
    // the main thread.
    //
    // stdio stays the default so every existing caller that spawns this binary and
    // talks over pipes is unaffected. `--listen` is the shared-service mode: the
    // listener outlives any one client, so the daemon keeps serving across
    // disconnects rather than exiting with its only pipe.
    let inbound = match listen_path {
        None => transport::spawn_stdin_reader(),
        Some(path) => match transport::spawn_socket_listener(&path) {
            Ok(inbound) => {
                eprintln!("hipfire daemon listening on {}", path.display());
                inbound
            }
            // `fatal_startup_error` diverges — it emits a fatal frame and exits.
            Err(error) => hipfire_daemon_adapter::fatal_startup_error(
                &format!("failed to listen on {}: {error}", path.display()),
                None,
            ),
        },
    };

    for frame in inbound {
        let transport::Inbound { payload, reply } = frame;

        // Answer on the connection this frame arrived on, rather than wherever the
        // previous one came from. With a single stdio client this is a no-op; with
        // several socket clients it is the whole point, and getting it wrong would
        // deliver one client's tokens to another.
        daemon_state.out.sink = reply;

        // Set the stamp for this frame in exactly one place, before anything can
        // emit. Every frame the request produces is tagged with it (see
        // `Responder::emit`) so a caller can correlate replies.
        //
        // The match is what makes this safe: a frame that failed to parse has no
        // id to recover, and clearing it here means a parse error cannot inherit
        // the *previous* request's id and blame a request that in fact succeeded.
        // Doing it per-branch instead invites exactly that bug back the next time
        // a variant is added.
        daemon_state.out.request_id = match &payload {
            transport::Payload::Request(msg) => msg
                .get("id")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string(),
            transport::Payload::Malformed(_) => String::new(),
        };

        // Drop any stop that arrived too late for its target. An abort names the
        // request it is for, so a stale one could not stop this request anyway —
        // but clearing here means it also cannot linger and stop some later
        // request that happens to reuse the id.
        hipfire_runtime::cancel::clear();

        let msg = match payload {
            transport::Payload::Request(msg) => msg,
            transport::Payload::Malformed(error) => {
                // Reported here rather than in the reader so every write stays on
                // this thread and keeps its place relative to real responses. The
                // envelope goes through serde_json because the parse-error text is
                // not JSON-safe: serde messages carry quotes/newlines and echo the
                // offending input, so raw interpolation would corrupt the stream.
                daemon_state.out.error(format!("invalid JSON: {error}"));
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
                    &mut daemon_state.out.sink,
                    id,
                    format!("unsupported or malformed request '{msg_type}': {e}"),
                );
                continue;
            }
        };

        match request {
            DaemonRequest::ModelRegistry => {
                handlers::status::registry(&mut daemon_state, &llm_registry)
            }
            DaemonRequest::Load(_) => {
                handlers::lifecycle::load(&mut daemon_state, &msg, &protocol_load)
            }

            DaemonRequest::Embed(req) => handlers::generate::embed(&mut daemon_state, &msg, req),

            DaemonRequest::Rerank(req) => handlers::generate::rerank(&mut daemon_state, &msg, req),

            DaemonRequest::Generate(_) => handlers::generate::text(&mut daemon_state, &msg),

            DaemonRequest::GenerateBatchPrefill => {
                handlers::batch::prefill(&mut daemon_state, &msg)
            }

            DaemonRequest::PrefixHashPreflight => {
                handlers::batch::prefix_hash_preflight(&mut daemon_state, &msg)
            }

            DaemonRequest::GenerateBatchDecodeStep => {
                handlers::batch::decode_step(&mut daemon_state, &msg)
            }

            DaemonRequest::ReleaseSessions => {
                handlers::sessions::release_sessions(&mut daemon_state, &msg)
            }

            DaemonRequest::ReserveSessionState => {
                handlers::sessions::reserve_session_state(&mut daemon_state, &msg)
            }

            DaemonRequest::DescribeState => {
                handlers::sessions::describe_state(&mut daemon_state, &msg)
            }

            DaemonRequest::ReleaseState => {
                handlers::sessions::release_state(&mut daemon_state, &msg)
            }

            DaemonRequest::WorkerStatus => handlers::status::worker_status(&mut daemon_state),

            DaemonRequest::ResourceStatus => handlers::status::resource_status(&mut daemon_state),

            DaemonRequest::Inventory => handlers::status::inventory(&mut daemon_state),

            DaemonRequest::Reset => handlers::lifecycle::reset(&mut daemon_state, &msg),

            DaemonRequest::Unload => handlers::lifecycle::unload(&mut daemon_state),

            DaemonRequest::UnloadWorker => {
                handlers::lifecycle::unload_worker(&mut daemon_state, &msg)
            }

            DaemonRequest::Ping => handlers::status::ping(&mut daemon_state),

            // Calibrate the resident model in place (no reload): run the Tier-1
            // collector over a corpus and write a .calib.hfq. The data plane stays
            // daemon-internal — only the request + the resulting path/summary cross
            // JSONL. Single-GPU qwen3.5-family bf16 only (capture fires at the
            // bf16 chokepoints); additive and gated, never on the decode hot path.
            DaemonRequest::Collect(_) => handlers::calibrate::collect(&mut daemon_state, &msg),

            // Daemon-resident KLD evaluation (no reload). `self_score` builds a
            // reference from the loaded model and scores the SAME model against
            // it through one forward path → ≈0 on a healthy run; the guard that
            // catches the historical two-binary drift. build_ref/score (with the
            // .kldref container) land next.
            DaemonRequest::KldEval(_) => handlers::calibrate::kld_eval(&mut daemon_state, &msg),

            // Refusal-direction steering / abliteration session control. The
            // in-forward `maybe_steer_block` hook (compiled into the gemma3
            // forward) keeps a process-global capture/apply session; these arms
            // expose control over it so a client (hipfire-steer-harness) can drive
            // capture→derive→apply→score through the daemon's correct inference +
            // templating instead of a reimplemented harness. See
            // docs/plans/2026-06-30-steer-daemon-pivot.md.
            DaemonRequest::SteerBeginCapture(_) => {
                handlers::steer::begin_capture(&mut daemon_state, &msg)
            }

            // Prefill ONE chat turn through the hooked forward (no decode) and fold
            // its last-prompt-token residuals into the capture means. Prefill-only:
            // a decoded token's forward would overwrite the residual the hook just
            // recorded (the `collect` arm is prefill-only for the same reason).
            DaemonRequest::SteerCapture(_) => handlers::steer::capture(&mut daemon_state, &msg),

            // End the capture session and return the per-block means as a
            // num_layers × hidden f32 matrix (the client derives directions from
            // the +/- means it collected).
            DaemonRequest::SteerFinishCapture => handlers::steer::finish_capture(&mut daemon_state),

            // Begin an apply session: steer (additive) or ablate (projective) each
            // block in [layer_start, layer_end) along the per-block `directions`.
            DaemonRequest::SteerBeginApply(_) => {
                handlers::steer::begin_apply(&mut daemon_state, &msg)
            }

            // Tear down any active steer session (back to the base model).
            DaemonRequest::SteerClear => handlers::steer::clear(&mut daemon_state),

            // ── H-Neurons intervention gain (arXiv 2512.01797) ──────────────
            // Set a process-global per-neuron activation gain on the resident
            // dense model: each FLAT feature index (`layer*intermediate+neuron`)
            // is scaled by `gain` in the FFN forward (prefill + decode); every
            // other neuron by 1.0. `gain == 1.0` or an empty set clears the
            // session — the identity control point of the dose-response sweep.
            DaemonRequest::HneuronIntervene(_) => {
                handlers::hneurons::intervene(&mut daemon_state, &msg)
            }

            // ── H-Neurons CETT capture (arXiv 2512.01797) ───────────────────
            // Load the per-layer down_proj column norms (`‖W_down[:,j]‖`) once
            // from a compact little-endian binary produced host-side from the
            // source fp16 weights:
            //   [u32 n_layers][u32 intermediate][f32 × n_layers*intermediate].
            // Cached in `cett_colnorms` and reused for every `cett_capture`.
            DaemonRequest::CettLoadColnorms(_) => {
                handlers::hneurons::cett_load_colnorms(&mut daemon_state, &msg)
            }

            // Prefill (jinja-framed prompt + response) through the CETT-tapped
            // llama forward and return the per-layer mean-over-response-tokens
            // CETT feature (`[n_layers][intermediate]`). Requires a resident
            // llama backend (arch 10) and a prior `cett_load_colnorms`.
            DaemonRequest::CettCapture(_) => {
                handlers::hneurons::cett_capture(&mut daemon_state, &msg)
            }

            DaemonRequest::LoraLoad(_) => handlers::lora::load(&mut daemon_state, &msg),
            DaemonRequest::LoraSetScale(_) => handlers::lora::set_scale(&mut daemon_state, &msg),
            DaemonRequest::LoraUnload(_) => handlers::lora::unload(&mut daemon_state, &msg),
            DaemonRequest::LoraClear => handlers::lora::clear(&mut daemon_state),
            DaemonRequest::LoraList => handlers::lora::list(&mut daemon_state),

            // PFlash drafter TEACHER: forward the resident qwen3.5 target over a
            // corpus and emit per-chunk per-block cosine-K scores at the shallow +
            // mid FullAttention layers — the labels `pflash_drafter_train` distils
            // (teacher/student split, docs/plans/2026-06-19-training-via-daemon-forward.md).
            // Output is JSONL, one line per chunk; the trainer's daemon-label
            // loader converts it to the v2 label cache.
            DaemonRequest::PflashLabels(_) => {
                handlers::train::pflash_labels(&mut daemon_state, &msg)
            }

            // Train a PFlash importance-scorer drafter in-process against the
            // resident target (teacher/student split). STEP 1: plumbing only —
            // validates args + the hipfire-train link; the loop wiring lands in
            // step 3. See docs/plans/2026-06-19-train-as-daemon-op.md.
            DaemonRequest::TrainDrafter => handlers::train::train_drafter(&mut daemon_state, &msg),

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
            DaemonRequest::TrainLora => handlers::train::train_lora(&mut daemon_state, &msg),

            DaemonRequest::Diag => handlers::diag::diag(&mut daemon_state),

            DaemonRequest::BenchPrefill(_) => {
                handlers::diag::bench_prefill(&mut daemon_state, &msg)
            }

            DaemonRequest::Profile => handlers::diag::profile(&mut daemon_state),

            DaemonRequest::Abort(_) | DaemonRequest::ForceAnswer(_) => {
                handlers::status::control_frame_names_no_request(&mut daemon_state, msg_type)
            }
        }
    }
}
