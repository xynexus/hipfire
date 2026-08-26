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
use hipfire_serving_core::batch_executor::{batch_executor_for, batch_unsupported_reason};
use hipfire_serving_core::{
    dummy, events, generate, generate_vl, load, model, output_filter, qwen35_decode,
    qwen35_prefill, request, session,
};
use load::*;
use model::{CaskConfig, EmbeddingGemmaState, LoadedModel};
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

    /// Replace the memory budgets at runtime. Unset fields keep their value.
    ///
    /// These four are the only part of the daemon's startup environment that is
    /// *not* fixed at exec: `HIPFIRE_DEVICES` and the resource-lock settings are
    /// consumed before `Gpu::init()` and before the process takes its flocks, so
    /// they describe locks already held and cannot be revised. The budgets only
    /// size the ballast allocation, which `release_placeholders` /
    /// `reacquire_placeholders` already re-apply — so they can travel over the
    /// wire instead of only through a spawned child's environment.
    ///
    /// That distinction is what a caller attaching to a running daemon needs:
    /// it can push its budgets, and must accept the daemon's locks as given.
    fn set_budgets(
        &mut self,
        system_memory_budget_bytes: Option<u64>,
        system_memory_headroom_bytes: Option<u64>,
        vram_budget_bytes: Option<u64>,
        vram_headroom_bytes: Option<u64>,
    ) {
        if let Some(v) = system_memory_budget_bytes {
            self.system_memory_budget_bytes = v;
        }
        if let Some(v) = system_memory_headroom_bytes {
            self.system_memory_headroom_bytes = v;
        }
        if let Some(v) = vram_budget_bytes {
            self.vram_budget_bytes = v;
        }
        if let Some(v) = vram_headroom_bytes {
            self.vram_headroom_bytes = v;
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

fn hfqm_blob(package: &hipfire_runtime::hfq::HfqPackage, name: &str) -> std::io::Result<Vec<u8>> {
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
        &hfqm_blob(&package, "kldref.tokens")?,
        "kldref.tokens",
        n_chunk * n_ctx,
    )?;
    let top_count = n_chunk * scored_per_chunk * top_k;
    let top_indices = le_u32_vec(
        &hfqm_blob(&package, "kldref.top_indices")?,
        "kldref.top_indices",
        top_count,
    )?;
    let top_log_probs = le_f32_vec(
        &hfqm_blob(&package, "kldref.top_log_probs")?,
        "kldref.top_log_probs",
        top_count,
    )?;
    let residual_mass = le_f32_vec(
        &hfqm_blob(&package, "kldref.residual_mass")?,
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

    /// A budget update names only what it changes. Without this, a caller adjusting
    /// headroom alone would silently zero the budget it did not mention — and the
    /// daemon would release its whole reservation in response.
    #[test]
    fn set_budgets_leaves_unnamed_fields_alone() {
        let mut mgr = ResourceReservationManager::from_env_reader(|_| None);
        mgr.set_budgets(Some(8), Some(2), Some(64), Some(16));
        assert_eq!(mgr.system_memory_budget_bytes, 8);
        assert_eq!(mgr.vram_budget_bytes, 64);
        assert_eq!(mgr.vram_headroom_bytes, 16);

        // Change one field; the other three must survive.
        mgr.set_budgets(None, None, None, Some(32));
        assert_eq!(mgr.vram_headroom_bytes, 32);
        assert_eq!(mgr.vram_budget_bytes, 64, "budget must not be zeroed");
        assert_eq!(mgr.system_memory_budget_bytes, 8);
        assert_eq!(mgr.system_memory_headroom_bytes, 2);

        // Zero is a value, not "unset" — clearing a budget has to be expressible.
        mgr.set_budgets(None, None, Some(0), Some(0));
        assert_eq!(mgr.vram_budget_bytes, 0);
        assert_eq!(mgr.vram_headroom_bytes, 0);
    }

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
        let model_path = dir.join("Qwen3.5-122B-A10B--mq4.hfq");
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
mod queue;
mod state;
mod stream;
mod transport;
use state::DaemonState;

/// Print a friendly, user-actionable message when Gpu::init fails. Matches
/// the panic shape we used to emit (which dumped a Rust backtrace and the
/// raw HipError debug-format) but turns it into a concrete next-step list.
/// The most common cause on Windows (#112) is HIP SDK present but no
/// AMD GPU driver visible to the runtime; on Linux it is usually missing
/// `libamdhip64.so` or kernel-side amdgpu / kfd not loaded.
fn report_gpu_init_failure(err: &hip_bridge::HipError) {
    let hints = if cfg!(target_os = "windows") {
        "  Most common Windows cause: HIP SDK is loaded but no\n  \
         AMD GPU is visible to the runtime. Verify:\n    \
         1. AMD Adrenalin driver is installed and current.\n    \
         2. AMD HIP SDK 6.2 or newer is installed:\n       \
         https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html\n    \
         3. `amdhip64.dll` is reachable (HIP_PATH set or DLL on PATH).\n    \
         4. Reboot after driver / SDK install if you have not yet."
    } else {
        "  Most common Linux causes:\n    \
         1. amdgpu kernel module not loaded (check `lsmod | grep amdgpu`).\n    \
         2. /dev/kfd missing or not readable by the current user\n       \
         (add to the `render` group; reboot).\n    \
         3. ROCm not installed or libamdhip64.so missing\n       \
         (check `ldconfig -p | grep amdhip64`)."
    };
    tracing::error!(
        "hipfire: failed to initialize GPU runtime.\n  HIP error: {} (code {})\n\n{hints}\n\n  Run `hipfire diag` for a full environment report.",
        err.message,
        err.code
    );
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

/// Resident state of a layer-preemptible `calibrate` (induction) run. The daemon
/// runs exactly one calibration layer per `Calibrate` request and keeps this
/// alive between requests (keyed by `run_id`); the caller re-enqueues per layer,
/// so induction time-slices with interactive serving. The engine lives in
/// `hipfire_runtime::calibration::layer_stream`; `DaemonCalibration` owns the
/// boxed native adapter, the safetensors source, and the calibration job the
/// engine borrows each turn, so parking it here is a move. No GPU self-lock is
/// taken (unlike the daemon-free `hipfire-coexistence calibrate` CLI): the daemon
/// already holds the process-lifetime GPU lease.
struct CalibrateDaemonSession {
    run_id: String,
    session: hipfire_runtime::calibration::layer_stream::DaemonCalibration,
}

/// Advance every runnable stream by ONE token, then return.
///
/// Executor v2 §M3b1. This is the march loop: serial by design, because the
/// parallelism in shared-weight decoding is intra-kernel, not across threads.
/// One quantum per stream per pass is what lets two requests interleave instead
/// of one draining to completion while the other waits.
///
/// Structured to hold at most one borrow of `daemon_state` at a time: the
/// generation is TAKEN OUT of the table before the model is borrowed, and put
/// back (or retired) only after the model borrow ends. Holding
/// `&mut streams` and `&mut model` together does not compile, and working
/// around that by cloning would copy a live KV handle.
/// §M7. One batched forward for every runnable stream.
///
/// The round-robin march steps one stream per quantum through the single
/// resident slot, so N streams cost N full forwards and aggregate throughput is
/// flat in N. This hands all of them to `step_batch`, which issues ONE forward
/// and lets each handle sample from its own session's logits.
///
/// Sessions come from the registry, not the resident slot: the slot holds one,
/// and `resume`-ing a second without an intervening park is the documented way
/// to get "qwen35 session missing decode state". `qwen35_save_active_session`
/// puts the occupant back first so every stream is findable.
/// Returns the streams it actually stepped. The caller MUST skip those in the
/// round-robin pass: without that, every stream stepped twice per march and
/// exactly half the tokens bypassed the fused arm (measured: 8 fused rows=4
/// steps for 64 tokens).
fn march_streams_batched(daemon_state: &mut state::DaemonState) -> Vec<stream::StreamId> {
    let ids = daemon_state.streams.runnable();
    if ids.len() < 2 {
        return Vec::new();
    }
    // Take the handles out of the table for the round; each is put back or
    // retired below, mirroring the round-robin path's ownership.
    let mut taken: Vec<(
        stream::StreamId,
        String,
        String,
        hipfire_serving_core::generate::Qwen35Generation,
    )> = Vec::with_capacity(ids.len());
    for id in ids {
        if let Some(s) = daemon_state.streams.get_mut(id) {
            // A prefilling stream has no decodable token yet. Leaving it in the
            // table hands it to the round-robin pass, which advances it one
            // prefill band per quantum. Taking it here would be worse than
            // useless: the caller skips whatever this batch claims, so the
            // stream's prefill would never advance at all.
            if s.generation.as_ref().is_some_and(|g| g.is_prefilling()) {
                continue;
            }
            if let Some(g) = s.generation.take() {
                taken.push((
                    id,
                    s.request_id.clone(),
                    s.session().as_str().to_string(),
                    g,
                ));
            }
        }
    }
    if taken.len() < 2 {
        // Put back whatever we took; the round-robin path will handle it.
        for (id, _, _, g) in taken {
            if let Some(s) = daemon_state.streams.get_mut(id) {
                s.generation = Some(g);
            }
        }
        return Vec::new();
    }

    let Some(m) = daemon_state.model.as_mut() else {
        for (id, _, _, _) in taken {
            daemon_state.streams.retire(id);
        }
        return Vec::new();
    };
    if let Err(e) =
        hipfire_serving_core::session::qwen35_save_active_session(m, &mut daemon_state.gpu)
    {
        eprintln!("[executor] batched march: cannot free the resident slot: {e}");
        for (id, _, _, g) in taken {
            if let Some(s) = daemon_state.streams.get_mut(id) {
                s.generation = Some(g);
            }
        }
        return Vec::new();
    }
    // The fused batch scratch is allocated lazily by the batched PREFILL path,
    // which never runs for plain `generate` requests — so on this path it is
    // simply absent and `step_batch` refuses. Measured, not predicted: the first
    // run of this code fell back to round-robin with "needs prefill-batch
    // scratch" and produced byte-identical output, which looks like success and
    // is not.
    {
        let rows_needed = taken.len();
        let need_alloc = m
            .q35_scratch
            .as_ref()
            .map(|sc| {
                sc.prefill_batch
                    .as_ref()
                    .is_none_or(|pbs| pbs.max_batch < rows_needed)
            })
            .unwrap_or(false);
        if need_alloc {
            let cfg = m.q35_config.as_ref().expect("qwen35 config").clone();
            if let Some(sc) = m.q35_scratch.as_mut() {
                if let Some(existing) = sc.prefill_batch.take() {
                    existing.free_gpu(&mut daemon_state.gpu);
                }
                match hipfire_arch_qwen35::qwen35::PrefillBatchScratch::new(
                    &mut daemon_state.gpu,
                    &cfg,
                    rows_needed.max(2),
                ) {
                    Ok(pbs) => sc.prefill_batch = Some(pbs),
                    Err(e) => {
                        eprintln!("[executor] batched march: cannot allocate batch scratch: {e:?}")
                    }
                }
            }
        }
    }
    // Same quantum event as the round-robin path: a batched round hands every
    // row in it a slice, so each row gets one record.
    for (_, req_id, _, _) in taken.iter() {
        hipfire_runtime::exec_trace::record(
            hipfire_runtime::exec_trace::TraceEvent::QuantumBegin,
            hipfire_runtime::exec_trace::stream_id_of(req_id),
            0,
            0,
        );
    }
    for (_, _, sid, g) in taken.iter_mut() {
        if let Err(e) = g.acquire_from_registry(m, sid) {
            eprintln!("[executor] batched march: {e}");
        }
    }

    let outcomes = {
        let mut entries: Vec<(&str, &mut hipfire_serving_core::generate::Qwen35Generation)> = taken
            .iter_mut()
            .map(|(_, rid, _, g)| (rid.as_str(), g))
            .collect();
        let model = daemon_state.model.as_ref().expect("checked above");
        hipfire_serving_core::generate::Qwen35Generation::step_batch(
            &mut entries,
            model,
            &mut daemon_state.gpu,
            model
                .tokenizer
                .as_ref()
                .expect("qwen35 model has a tokenizer"),
            &mut daemon_state.out.sink,
        )
    };
    let outcomes = match outcomes {
        Ok(o) => o,
        Err(e) => {
            eprintln!("[executor] batched march failed: {e}");
            let m = daemon_state.model.as_mut().expect("checked above");
            for (id, _, sid, mut g) in taken {
                g.release_to_registry(m, &sid);
                if let Some(s) = daemon_state.streams.get_mut(id) {
                    s.generation = Some(g);
                }
            }
            return Vec::new();
        }
    };

    let stepped: Vec<stream::StreamId> = taken.iter().map(|(id, ..)| *id).collect();
    for ((id, req_id, sid, mut g), step) in taken.into_iter().zip(outcomes) {
        let m = daemon_state.model.as_mut().expect("checked above");
        // Release ONLY on the continue path. `finish` and `fail` both consume
        // the session themselves — releasing first leaves the handle empty and
        // `finish` panics with "finish requires a resumed stream". Measured: the
        // daemon died at the first stream to hit EOS, ~32 tokens in, and the
        // truncated-but-plausible output read as a 32-token cap.
        match step {
            hipfire_serving_core::generate::Qwen35Step::Continue if g.should_continue() => {
                g.release_to_registry(m, &sid);
                if let Some(s) = daemon_state.streams.get_mut(id) {
                    s.generation = Some(g);
                }
            }
            hipfire_serving_core::generate::Qwen35Step::Failed(message) => {
                g.fail(
                    m,
                    &mut daemon_state.gpu,
                    &mut daemon_state.out.sink,
                    &req_id,
                    &message,
                );
                daemon_state.streams.retire(id);
            }
            _ => {
                g.finish(
                    m,
                    &mut daemon_state.gpu,
                    &mut daemon_state.out.sink,
                    &req_id,
                );
                daemon_state.streams.retire(id);
            }
        }
    }
    stepped
}

/// Run the executor until nothing is runnable.
///
/// Required before anything that destroys the model the live streams are
/// mid-generation on. The march loop only runs when the pending queue is EMPTY
/// (`pop_next() -> None`), so a batch that arrives together — the stdin
/// protocol delivers exactly that — dispatches `unload` BEFORE the executor has
/// stepped anything. `daemon_state.model` is then `None` when the march finally
/// runs, every admitted stream takes the `None => Outcome::Retired` arm, and the
/// whole batch is lost **without a single error frame**.
///
/// Measured on a 4-stream batch: with `unload` present, 0 tokens and 0 errors;
/// with it removed, 4 x 16 tokens. Same binary, same flag.
fn drain_streams(daemon_state: &mut state::DaemonState) {
    if !stream::executor_v2_enabled() {
        return;
    }
    // Bounded: `march_streams` retires a stream on every terminal outcome, so
    // the runnable set strictly shrinks unless a stream is making progress. The
    // cap is a backstop against a stream that neither progresses nor retires —
    // it would otherwise hang the daemon on teardown.
    let mut guard = 0usize;
    while !daemon_state.streams.runnable().is_empty() {
        march_streams(daemon_state);
        guard += 1;
        if guard > 1_000_000 {
            eprintln!(
                "[executor] drain gave up with {} stream(s) still runnable",
                daemon_state.streams.runnable().len()
            );
            break;
        }
    }
}

fn march_streams(daemon_state: &mut state::DaemonState) {
    if !stream::executor_v2_enabled() {
        return;
    }
    stream::warn_if_banding_without_priority();
    let batched: Vec<stream::StreamId> = if stream::executor_batched_enabled() {
        march_streams_batched(daemon_state)
    } else {
        Vec::new()
    };
    // Whatever the batched round could not take (fewer than two runnable, or a
    // stream it excluded) falls through to the round-robin march. What it DID
    // step must not step again this march.
    for id in daemon_state.streams.runnable() {
        if batched.contains(&id) {
            continue;
        }
        let Some((generation, req_id, session)) = daemon_state.streams.get_mut(id).and_then(|s| {
            s.generation
                .take()
                .map(|g| (g, s.request_id.clone(), s.session().as_str().to_string()))
        }) else {
            continue;
        };
        // The march is handing this stream a slice. First one per stream is
        // "first dispatch" for §M3d measurement 2.
        hipfire_runtime::exec_trace::record(
            hipfire_runtime::exec_trace::TraceEvent::QuantumBegin,
            hipfire_runtime::exec_trace::stream_id_of(&req_id),
            0,
            0,
        );
        // `fail` and `finish` both consume the handle, and only one of them runs.
        // An Option makes that move trackable instead of fighting the borrow
        // checker across match arms.
        let mut slot = Some(generation);

        enum Outcome {
            Stepped,
            Done,
            Retired,
        }

        // §M1d: scope this quantum to the stream's own steer session.
        //
        // One executor thread serves every stream, so without this the forward's
        // `maybe_steer_block` hooks resolve against an unset `CURRENT_KEY` and
        // every stream shares one process-wide session — which is what
        // `handlers/steer.rs` means by "two steer ops must never interleave", a
        // rule the executor removes the ability to honour.
        //
        // Per QUANTUM, and dropped at the end of it: the guard is `!Send` and
        // restores the previous key on drop, so it cannot leak into whichever
        // stream this thread marches next.
        //
        // A stream with no session of its own falls back to the unscoped one
        // (`hipfire_steer::effective_key`), so a process-wide steer op keeps
        // applying exactly as before.
        let _steer_scope = hipfire_steer::SteerKeyGuard::install(hipfire_steer::SteerKey::session(
            session.clone(),
        ));

        let outcome = match daemon_state.model.as_mut() {
            None => Outcome::Retired,
            Some(m) => {
                // Per QUANTUM, not per frame: the resident session slot holds one
                // live KV/DeltaNet, so marching stream B without re-activating
                // would drive B's tokens into A's cache.
                // Resume: swap this stream's session into the resident slot and
                // take it for the quantum. `activate_session` inside saves
                // whichever stream parked last, which is why park-after-step is
                // not optional.
                if slot
                    .as_mut()
                    .expect("present")
                    .resume(m, &mut daemon_state.gpu, &session)
                    .is_err()
                {
                    Outcome::Retired
                } else {
                    match m.tokenizer.as_ref() {
                        None => Outcome::Retired,
                        Some(_) => {
                            let tok = m.tokenizer.as_ref().expect("checked above");
                            let g = slot.as_mut().expect("present until consumed");
                            match g.step(
                                m,
                                &mut daemon_state.gpu,
                                tok,
                                &mut daemon_state.out.sink,
                                &req_id,
                            ) {
                                hipfire_serving_core::generate::Qwen35Step::Continue => {
                                    if slot.as_ref().is_some_and(|g| g.should_continue()) {
                                        Outcome::Stepped
                                    } else {
                                        Outcome::Done
                                    }
                                }
                                hipfire_serving_core::generate::Qwen35Step::Stop => Outcome::Done,
                                hipfire_serving_core::generate::Qwen35Step::Failed(message) => {
                                    slot.take().expect("present until consumed").fail(
                                        m,
                                        &mut daemon_state.gpu,
                                        &mut daemon_state.out.sink,
                                        &req_id,
                                        &message,
                                    );
                                    Outcome::Retired
                                }
                            }
                        }
                    }
                }
            }
        };

        match outcome {
            Outcome::Stepped => {
                // Park before yielding to the next stream, so the slot is
                // populated and the next `activate_session` has something to
                // save. Without this the second stream dies with "qwen35
                // session missing decode state".
                if let (Some(m), Some(g)) = (daemon_state.model.as_mut(), slot.as_mut()) {
                    let _ = g.park(m, &mut daemon_state.gpu);
                }
                if let Some(s) = daemon_state.streams.get_mut(id) {
                    s.generation = slot.take();
                }
            }
            Outcome::Done => {
                if let (Some(m), Some(g)) = (daemon_state.model.as_mut(), slot.take()) {
                    g.finish(
                        m,
                        &mut daemon_state.gpu,
                        &mut daemon_state.out.sink,
                        &req_id,
                    );
                }
                daemon_state.streams.retire(id);
            }
            Outcome::Retired => {
                daemon_state.streams.retire(id);
            }
        }
    }
}

fn main() {
    // Cooperative generation cancellation: install the SIGUSR1 handler so the
    // HTTP server can abort an in-flight generation (on client disconnect)
    // without SIGKILL-ing this worker and destroying the loaded model. The
    // handler only sets a process-global atomic (async-signal-safe); the
    // per-token decode loops poll it and stop cleanly. See
    // `hipfire_runtime::GENERATION_CANCEL`.
    hipfire_runtime::install_generation_cancel_handler();
    // Init logging first so every later tracing event (including --precompile
    // and lock-acquisition paths) is captured, not dropped.
    hipfire_runtime::logging::init_stderr_logging("daemon", "info");

    // Before ANY Gpu::init: FeatureFlags snapshots once at init and hot paths
    // read the cached struct, so config values installed later would be ignored.
    hipfire_runtime::config::install_rdna_overrides();

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
        tracing::info!("Pre-compiling kernels for {}...", gpu.arch);
        let mut errors = 0usize;
        for kv in &["asym3", "q8"] {
            for wq in &["mq4", "mq6", "hfq4", "hfq6", "q8"] {
                if let Err(e) = gpu.precompile_qwen35(wq, kv, 256) {
                    tracing::warn!("{wq}/{kv}: {e}");
                    errors += 1;
                }
            }
        }
        if errors > 0 {
            tracing::warn!("Kernel precompilation finished with {errors} failure(s) — the missing kernels will JIT on first use.");
        } else {
            tracing::info!("Kernel precompilation done.");
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
    // Per-module durations (§M3d measurement 1) land in the executor trace only
    // if something wires the two crates together; dispatch cannot reach the
    // trace on its own.
    hipfire_runtime::exec_trace::install_dispatch_module_observer();
    let llm_registry = build_local_llm_registry();
    tracing::info!(
        "model registry: {} model(s), {} sidecar/template artifact(s) (models={}, triattn={}, drafts={}, templates={})",
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
                tracing::info!("hipfire daemon listening on {}", path.display());
                inbound
            }
            // `fatal_startup_error` diverges — it emits a fatal frame and exits.
            Err(error) => hipfire_daemon_adapter::fatal_startup_error(
                &format!("failed to listen on {}: {error}", path.display()),
                None,
            ),
        },
    };

    // Drain whatever has arrived, then run whichever pending frame the queue picks.
    // Draining first is what gives the scheduler a choice: blocking straight on
    // `recv` would take frames in arrival order and there would be nothing to
    // choose between. See `queue` for why reordering is safe only across
    // connections.
    let mut pending = queue::PendingQueue::default();
    loop {
        // Block only when there is nothing to run AND nothing to march. With a
        // live stream mid-generation an unconditional `recv` would park the
        // executor forever waiting on a client that is waiting on us.
        if pending.is_empty() && daemon_state.streams.runnable().is_empty() {
            // Nothing in hand: block until something arrives, and stop when every
            // reader has hung up (stdin EOF, or the listener shutting down).
            match inbound.recv() {
                Ok(frame) => pending.push(frame),
                Err(_) => break,
            }
        }
        // Take everything else already queued so the choice is over the whole
        // backlog rather than just the first arrival.
        while let Ok(frame) = inbound.try_recv() {
            pending.push(frame);
        }
        let Some(frame) = pending.pop_next() else {
            march_streams(&mut daemon_state);
            continue;
        };
        // Scheduling decisions are otherwise unobservable from outside: replies
        // race each other through the client's sockets, so client-side arrival
        // order does NOT report what the daemon actually chose. This trace is the
        // ground truth, and is what the reordering test asserts on.
        if std::env::var("HIPFIRE_DAEMON_SCHED_DEBUG").as_deref() == Ok("1") {
            tracing::debug!(
                "[sched] chose conn={} seq={} pri={} (queue depth after: {})",
                frame.conn,
                frame.seq,
                frame.priority,
                pending.len()
            );
        }

        daemon_state.scheduler_stats = pending.stats().clone();

        // One dispatch boundary pair per frame, closed by the guard's `Drop` on
        // every exit path including the `continue`s below. `aux` carries the
        // queue depth left behind this frame, which is what distinguishes a
        // daemon that had a choice from one that was simply handed work in
        // arrival order — the difference `overtaken_total` reports as a count
        // and this reports in time.
        let _dispatch = hipfire_runtime::exec_trace::DispatchGuard::begin(pending.len() as u64);
        // VRAM is sampled per frame, not per record: it is a driver call, and
        // the leak it exists to catch (v2 plan, risk 1) develops over minutes.
        // Sampling it into the same ring as the latency series is the point —
        // a paging executor's failure reads as "the model got slower" long
        // before it reads as OOM, and only a shared timeline separates the two.
        if hipfire_runtime::exec_trace::enabled() {
            if let Ok((free, total)) = daemon_state.gpu.hip.get_vram_info() {
                hipfire_runtime::exec_trace::record(
                    hipfire_runtime::exec_trace::TraceEvent::VramSample,
                    hipfire_runtime::exec_trace::NO_STREAM,
                    0,
                    total.saturating_sub(free) as u64,
                );
            }
        }

        let transport::Inbound {
            payload,
            reply,
            conn: _,
            seq: _,
            priority: _,
        } = frame;

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
        // Same reasoning for the out-of-band #205 wire: a SIGUSR1 (cooperative
        // cancel on client disconnect) that landed after the previous generation
        // already finished must not immediately cancel this fresh request. The
        // daemon is serial, so clearing the GENERATION_CANCEL flag here — before
        // any request is dispatched — is race-free.
        hipfire_runtime::reset_generation_cancel();

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
                // A load replaces the resident model, so it destroys live
                // streams exactly as an unload does.
                drain_streams(&mut daemon_state);
                handlers::lifecycle::load(&mut daemon_state, &msg, &protocol_load)
            }

            DaemonRequest::Embed(req) => handlers::generate::embed(&mut daemon_state, &msg, req),

            DaemonRequest::Rerank(req) => handlers::generate::rerank(&mut daemon_state, &msg, req),

            DaemonRequest::GenerateBatchPrefill => {
                handlers::batch::prefill(&mut daemon_state, &msg)
            }

            DaemonRequest::PrefixHashPreflight => {
                handlers::batch::prefix_hash_preflight(&mut daemon_state, &msg)
            }

            DaemonRequest::GenerateBatchDecodeStep => {
                handlers::batch::decode_step(&mut daemon_state, &msg)
            }

            DaemonRequest::Generate(_) => {
                // Clear any stale cancel request before starting a fresh
                // generation: a SIGUSR1 delivered after the previous request
                // already finished (a disconnect racing the terminal `done`)
                // must not immediately cancel this one. The single serial worker
                // guarantees no other generation is in flight, so this is
                // race-free. Ported from origin/master, whose ~700-line inline
                // arm this branch had already refactored into
                // handlers::generate::text — only the reset is new behaviour.
                hipfire_runtime::reset_generation_cancel();
                // §M3a: the frame ADMITS a stream, then runs it. Only the shape
                // moves — the run is still inline and unchanged.
                //
                // The seam is HERE rather than inside `text` because that
                // handler has many early returns, and a retire per exit path is
                // exactly the leak shape `ThreadSinkGuard` exists to prevent.
                //
                // `None` means the session already had a live stream. It still
                // runs: refusing would be user-visible, which M3a must not be.
                let admitted = stream::admit_generate(&mut daemon_state, &msg);
                handlers::generate::text(&mut daemon_state, &msg);
                if let Some(id) = admitted {
                    // Retire here only on the INLINE path. Under the executor
                    // flag the handler stashed a generation on this stream and
                    // the march loop below owns its lifetime; retiring now would
                    // drop a half-run request on the floor.
                    let marching = daemon_state
                        .streams
                        .get(id)
                        .is_some_and(|s| s.generation.is_some());
                    if !marching {
                        daemon_state.streams.retire(id);
                    }
                }
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
            DaemonRequest::SchedulerStatus => handlers::status::scheduler_status(&mut daemon_state),
            DaemonRequest::ExecutorTrace => handlers::status::executor_trace(&mut daemon_state),
            DaemonRequest::SetResourceBudget(req) => {
                handlers::status::set_resource_budget(&mut daemon_state, req)
            }

            DaemonRequest::Inventory => handlers::status::inventory(&mut daemon_state),

            DaemonRequest::Reset => handlers::lifecycle::reset(&mut daemon_state, &msg),

            DaemonRequest::Unload => {
                // Finish live streams before their model disappears.
                drain_streams(&mut daemon_state);
                handlers::lifecycle::unload(&mut daemon_state)
            }

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
            DaemonRequest::SteerFinishCapture => {
                handlers::steer::finish_capture(&mut daemon_state, &msg)
            }

            // Begin an apply session: steer (additive) or ablate (projective) each
            // block in [layer_start, layer_end) along the per-block `directions`.
            DaemonRequest::SteerBeginApply(_) => {
                handlers::steer::begin_apply(&mut daemon_state, &msg)
            }

            // Tear down any active steer session (back to the base model).
            DaemonRequest::SteerClear => handlers::steer::clear(&mut daemon_state, &msg),

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

            // Layer-preemptible calibration/induction as a daemon op. Runs one
            // calibration layer per request against a resident DaemonCalibration
            // session (keyed by `run_id`); the caller re-enqueues per layer so
            // induction time-slices with interactive serving. The artifact is
            // byte-identical to the daemon-free `hipfire-coexistence calibrate`
            // CLI path (both feed the same `build_calibration_run_inputs`).
            DaemonRequest::Calibrate => handlers::calibrate::calibrate(&mut daemon_state, &msg),

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
