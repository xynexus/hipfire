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

mod handlers;
mod state;
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

    let stdin = std::io::stdin();

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
                emit_error_with_id(&mut daemon_state.stdout, "", format!("invalid JSON: {e}"));
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
                    &mut daemon_state.stdout,
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
                            &mut daemon_state.stdout,
                            "",
                            &format!("worker switch failed: {e}"),
                        );
                        let _ = daemon_state.stdout.flush();
                        continue;
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
                            &mut daemon_state.stdout,
                            "",
                            &format!("dummy load resource reservation failed: {err}"),
                        );
                        let _ = daemon_state.stdout.flush();
                        continue;
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
                    let _ = writeln!(daemon_state.stdout, "{line}");
                    let _ = daemon_state.stdout.flush();
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
                            daemon_state.stdout,
                            r#"{{"type":"error","message":"DFlash speculative decode requires pp=1 in v1 (set HIPFIRE_PP_DFLASH=1 to opt into the experimental pp>1 PRD path; note PR2-4 of docs/plans/hetero-pflash-dflash.prd are not yet implemented — the load message will accept but generate will not run cross-card spec-decode). See issue #58 v1.1 roadmap."}}"#
                        );
                        let _ = daemon_state.stdout.flush();
                        continue;
                    }
                    if cask.sidecar.is_some() {
                        let _ = writeln!(
                            daemon_state.stdout,
                            r#"{{"type":"error","message":"CASK / TriAttention eviction requires pp=1 in v1; see issue #58 v1.1 roadmap"}}"#
                        );
                        let _ = daemon_state.stdout.flush();
                        continue;
                    }
                    if (pflash_drafter.is_some() || pflash_mode_str != "off")
                        && std::env::var("HIPFIRE_PP_PFLASH").ok().as_deref() != Some("1")
                    {
                        let _ = writeln!(
                            daemon_state.stdout,
                            r#"{{"type":"error","message":"PFlash prefill compression requires pp=1 in v1 (set HIPFIRE_PP_PFLASH=1 to opt into the experimental pp>1 PoC); see issue #58 v1.1 roadmap"}}"#
                        );
                        let _ = daemon_state.stdout.flush();
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
                let planned_resource_usage = daemon_state
                    .resource_reservations
                    .planned_usage_for_load(path, protocol_load.as_ref().map(|req| &req.params));
                if let Err(err) = daemon_state
                    .resource_reservations
                    .release_placeholders(&mut daemon_state.gpu)
                {
                    hipfire_runtime::load_progress::set_sink(None);
                    write_error(
                        &mut daemon_state.stdout,
                        "",
                        &format!("resource reservation release failed before load: {err}"),
                    );
                    let _ = daemon_state.stdout.flush();
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
                                &mut daemon_state.stdout,
                                "",
                                &format!("resource reservation reacquire failed after load: {err}"),
                            );
                            let _ = daemon_state.stdout.flush();
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
                                    if let Err(e) = daemon_state.gpu.dpm_warmup(secs) {
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
                            daemon_state.stdout,
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
                                        daemon_state.stdout,
                                        r#"{{"type":"pflash_load_failed","reason":"invalid load param: {}"}}"#,
                                        reason.replace('"', "'")
                                    );
                                    let _ = daemon_state.stdout.flush();
                                    daemon_state.model = Some(m);
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
                                                    daemon_state.stdout,
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
                                                daemon_state.stdout,
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
                                                daemon_state.stdout,
                                                r#"{{"type":"pflash_load_failed","reason":"{}"}}"#,
                                                e.to_string().replace('"', "'")
                                            );
                                        }
                                    }
                                } else {
                                    let _ = writeln!(
                                        daemon_state.stdout,
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
                        let (vram_free, vram_total) =
                            daemon_state.gpu.hip.get_vram_info().unwrap_or((0, 0));
                        let free_mb = vram_free / (1024 * 1024);
                        let total_mb = vram_total / (1024 * 1024);
                        // serde-escape: raw HipError debug contains { } and "
                        // which corrupt the JSONL protocol if interpolated raw.
                        write_error(&mut daemon_state.stdout, "", &format!(
                            "load failed: {e}. GPU: {} ({free_mb} MB free / {total_mb} MB total)", daemon_state.gpu.arch));
                    }
                }
                let _ = daemon_state.stdout.flush();
            }

            DaemonRequest::Embed(req) => {
                let id = msg.get("id").and_then(|v| v.as_str()).unwrap_or("");
                let target_worker_id = message_worker_id(&msg);
                if daemon_state.dummy_model.is_some() {
                    emit_error_with_id(
                        &mut daemon_state.stdout,
                        id,
                        "embed is not supported for the dummy model",
                    );
                    continue;
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
                        continue;
                    }
                    Err(e) => {
                        emit_error_with_id(
                            &mut daemon_state.stdout,
                            id,
                            format!("worker switch failed: {e}"),
                        );
                        continue;
                    }
                }
                let Some(m) = daemon_state.model.as_ref() else {
                    emit_error_with_id(&mut daemon_state.stdout, id, "no model loaded");
                    continue;
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

            DaemonRequest::Rerank(req) => {
                let id = msg.get("id").and_then(|v| v.as_str()).unwrap_or("");
                let target_worker_id = message_worker_id(&msg);
                if daemon_state.dummy_model.is_some() {
                    emit_error_with_id(
                        &mut daemon_state.stdout,
                        id,
                        "rerank is not supported for the dummy model",
                    );
                    continue;
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
                        continue;
                    }
                    Err(e) => {
                        emit_error_with_id(
                            &mut daemon_state.stdout,
                            id,
                            format!("worker switch failed: {e}"),
                        );
                        continue;
                    }
                }
                let Some(m) = daemon_state.model.as_ref() else {
                    emit_error_with_id(&mut daemon_state.stdout, id, "no model loaded");
                    continue;
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
                            continue;
                        }
                        Err(e) => {
                            emit_error_with_id(
                                &mut daemon_state.stdout,
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
                    continue;
                }
                let m = match daemon_state.model.as_mut() {
                    Some(m) => m,
                    None => {
                        let _ = writeln!(
                            daemon_state.stdout,
                            r#"{{"type":"error","message":"no model loaded"}}"#
                        );
                        let _ = daemon_state.stdout.flush();
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
                    if let Err(e) = m.activate_session(&mut daemon_state.gpu, target_session_id) {
                        emit_error_with_id(&mut daemon_state.stdout, id, e);
                        continue;
                    }
                } else if session_id.is_some() || prefill_already_done {
                    emit_error_with_id(
                        &mut daemon_state.stdout,
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
                                daemon_state.stdout,
                                r#"{{"type":"error","id":"{}","message":"invalid tools field: {}"}}"#,
                                id,
                                e.to_string().replace('"', "'"),
                            );
                            let _ = daemon_state.stdout.flush();
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
                                        daemon_state.stdout,
                                        r#"{{"type":"error","id":"{}","message":"invalid tools field: {}"}}"#,
                                        id,
                                        e.to_string().replace('"', "'"),
                                    );
                                    let _ = daemon_state.stdout.flush();
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
                                        daemon_state.stdout,
                                        r#"{{"type":"error","id":"{}","message":"invalid messages field: {}"}}"#,
                                        id,
                                        e.to_string().replace('"', "'"),
                                    );
                                    let _ = daemon_state.stdout.flush();
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
                        eprintln!(
                            "[daemon/vl] both image and image_base64 provided — using image_base64"
                        );
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
                        generate_vl_dots_ocr(
                            m,
                            &mut daemon_state.gpu,
                            &mut daemon_state.stdout,
                            &params,
                        );
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
                            daemon_state.stdout,
                            r#"{{"type":"error","id":"{}","message":"invalid pflash override: {}"}}"#,
                            id,
                            reason.replace('"', "'"),
                        );
                        let _ = daemon_state.stdout.flush();
                        continue;
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

            DaemonRequest::GenerateBatchPrefill => match validate_generate_batch_prefill(&msg) {
                Ok(envelope) => {
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
                                    &envelope.id,
                                    format!("unknown model worker {target_worker_id}"),
                                );
                                continue;
                            }
                            Err(e) => {
                                emit_error_with_id(
                                    &mut daemon_state.stdout,
                                    &envelope.id,
                                    format!("worker switch failed: {e}"),
                                );
                                continue;
                            }
                        }
                    }
                    if envelope.is_probe() {
                        if daemon_state.dummy_model.is_some() {
                            emit_dummy_generate_batch_prefill_ready(
                                &mut daemon_state.stdout,
                                &envelope,
                            );
                            continue;
                        }
                        match daemon_state.model.as_ref() {
                            Some(m) if is_qwen35_family_arch_id(m.arch_id) && m.pp == 1 => {
                                emit_generate_batch_prefill_ready(
                                    &mut daemon_state.stdout,
                                    &envelope,
                                );
                            }
                            #[cfg(feature = "arch-lfm2moe")]
                            Some(m) if m.arch_id == ARCH_ID_LFM2_MOE && m.pp == 1 => {
                                emit_lfm2_generate_batch_prefill_ready(
                                    &mut daemon_state.stdout,
                                    &envelope,
                                );
                            }
                            Some(m) => {
                                let reason = format!(
                                    "generate_batch_prefill currently supports qwen35/qwen35-moe and lfm2-moe only (arch_id={})",
                                    m.arch_id
                                );
                                emit_generate_batch_prefill_unsupported(
                                    &mut daemon_state.stdout,
                                    &envelope,
                                    &reason,
                                );
                            }
                            None => {
                                emit_generate_batch_prefill_unsupported(
                                    &mut daemon_state.stdout,
                                    &envelope,
                                    "no model loaded",
                                );
                            }
                        }
                        continue;
                    }
                    if let Some(dummy) = daemon_state.dummy_model.as_mut() {
                        tracing::info!(
                            request_id = envelope.id,
                            batch_id = envelope.batch_id,
                            sessions = envelope.session_count,
                            "dummy generate_batch_prefill"
                        );
                        if let Err(e) = run_generate_batch_prefill_dummy(
                            dummy,
                            &mut daemon_state.stdout,
                            &envelope,
                        ) {
                            emit_error_with_id(&mut daemon_state.stdout, &envelope.id, e);
                        }
                        continue;
                    }
                    let m = match daemon_state.model.as_mut() {
                        Some(m) => m,
                        None => {
                            emit_error_with_id(
                                &mut daemon_state.stdout,
                                &envelope.id,
                                "no model loaded",
                            );
                            continue;
                        }
                    };
                    if is_qwen35_family_arch_id(m.arch_id) {
                        if let Err(e) = run_generate_batch_prefill_serial_qwen35(
                            m,
                            &mut daemon_state.gpu,
                            &mut daemon_state.stdout,
                            &envelope,
                            daemon_state.pflash_state.is_some(),
                        ) {
                            emit_error_with_id(&mut daemon_state.stdout, &envelope.id, e);
                        }
                    } else {
                        #[cfg(feature = "arch-lfm2moe")]
                        if m.arch_id == ARCH_ID_LFM2_MOE {
                            if let Err(e) = run_generate_batch_prefill_serial_lfm2(
                                m,
                                &mut daemon_state.gpu,
                                &mut daemon_state.stdout,
                                &envelope,
                            ) {
                                emit_error_with_id(&mut daemon_state.stdout, &envelope.id, e);
                            }
                            continue;
                        }
                        emit_error_with_id(
                            &mut daemon_state.stdout,
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
                    emit_error_with_id(&mut daemon_state.stdout, id, e);
                }
            },

            DaemonRequest::PrefixHashPreflight => match validate_prefix_hash_preflight(&msg) {
                Ok(envelope) => {
                    let target_worker_id = message_worker_id(&msg);
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
                                &envelope.id,
                                format!("unknown model worker {target_worker_id}"),
                            );
                            continue;
                        }
                        Err(e) => {
                            emit_error_with_id(
                                &mut daemon_state.stdout,
                                &envelope.id,
                                format!("worker switch failed: {e}"),
                            );
                            continue;
                        }
                    }
                    let m = match daemon_state.model.as_ref() {
                        Some(m) => m,
                        None => {
                            emit_error_with_id(
                                &mut daemon_state.stdout,
                                &envelope.id,
                                "no model loaded",
                            );
                            continue;
                        }
                    };
                    let preflight_result = if is_qwen35_family_arch_id(m.arch_id) {
                        run_prefix_hash_preflight_qwen35(m, &mut daemon_state.stdout, &envelope)
                    } else {
                        #[cfg(feature = "arch-lfm2moe")]
                        {
                            if m.arch_id == ARCH_ID_LFM2_MOE {
                                run_prefix_hash_preflight_lfm2(
                                    m,
                                    &mut daemon_state.stdout,
                                    &envelope,
                                )
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
                        emit_error_with_id(&mut daemon_state.stdout, &envelope.id, e);
                    }
                }
                Err(e) => {
                    let id = msg.get("id").and_then(|v| v.as_str()).unwrap_or("");
                    emit_error_with_id(&mut daemon_state.stdout, id, e);
                }
            },

            DaemonRequest::GenerateBatchDecodeStep => match validate_generate_batch_decode(&msg) {
                Ok(envelope) => {
                    let target_worker_id = message_worker_id(&msg);
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
                                &envelope.id,
                                format!("unknown model worker {target_worker_id}"),
                            );
                            continue;
                        }
                        Err(e) => {
                            emit_error_with_id(
                                &mut daemon_state.stdout,
                                &envelope.id,
                                format!("worker switch failed: {e}"),
                            );
                            continue;
                        }
                    }
                    let m = match daemon_state.model.as_mut() {
                        Some(m) => m,
                        None => {
                            emit_error_with_id(
                                &mut daemon_state.stdout,
                                &envelope.id,
                                "no model loaded",
                            );
                            continue;
                        }
                    };
                    if let Err(e) = run_generate_batch_decode_step_qwen35(
                        m,
                        &mut daemon_state.gpu,
                        &mut daemon_state.stdout,
                        &envelope,
                    ) {
                        emit_error_with_id(&mut daemon_state.stdout, &envelope.id, e);
                    }
                }
                Err(e) => {
                    let id = msg.get("id").and_then(|v| v.as_str()).unwrap_or("");
                    emit_error_with_id(&mut daemon_state.stdout, id, e);
                }
            },

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

            DaemonRequest::Reset => {
                let target_worker_id = reset_target_worker_id(&msg, &daemon_state.active_worker_id);
                if reset_has_no_resident_model(
                    &daemon_state.dummy_model,
                    &daemon_state.model,
                    &daemon_state.resident_models,
                ) {
                    daemon_state
                        .generic_state_arena
                        .release_worker(&target_worker_id);
                    let _ = writeln!(daemon_state.stdout, r#"{{"type":"reset","seq_pos":0}}"#);
                    let _ = daemon_state.stdout.flush();
                    continue;
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
                            emit_error_with_id(
                                &mut daemon_state.stdout,
                                "",
                                format!("unknown model worker {target_worker_id}"),
                            );
                            continue;
                        }
                        Err(e) => {
                            emit_error_with_id(
                                &mut daemon_state.stdout,
                                "",
                                format!("worker switch failed: {e}"),
                            );
                            continue;
                        }
                    }
                }
                // Reset conversation state without unloading the model.
                if let Some(dummy) = daemon_state.dummy_model.as_mut() {
                    daemon_state
                        .generic_state_arena
                        .release_worker(&target_worker_id);
                    dummy.reset();
                    let _ = writeln!(daemon_state.stdout, r#"{{"type":"reset"}}"#);
                    let _ = daemon_state.stdout.flush();
                    continue;
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
                        let _ = loaded
                            .backend
                            .reset_session(&mut daemon_state.gpu, "default");
                    }
                    let _ = writeln!(daemon_state.stdout, r#"{{"type":"reset","seq_pos":0}}"#);
                } else {
                    let _ = writeln!(
                        daemon_state.stdout,
                        r#"{{"type":"error","message":"no model loaded"}}"#
                    );
                }
                let _ = daemon_state.stdout.flush();
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
                    eprintln!(
                        "[hipfire-daemon] failed to restore resource reservations after unload: {err}"
                    );
                }
                daemon_state.generic_state_arena.clear();
                daemon_state.dummy_model = None;
                daemon_state.active_worker_id = DEFAULT_MODEL_WORKER_ID.to_string();
                // Drop any steer session so a stale capture/apply can't leak its
                // process-global state across model loads.
                hipfire_steer::clear();
                let _ = writeln!(daemon_state.stdout, r#"{{"type":"unloaded"}}"#);
                let _ = daemon_state.stdout.flush();
            }

            DaemonRequest::UnloadWorker => {
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
                let _ = writeln!(daemon_state.stdout, "{done}");
                let _ = daemon_state.stdout.flush();
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
            DaemonRequest::SteerFinishCapture => match hipfire_steer::finish_capture() {
                Some(means) => {
                    let resp = serde_json::json!({
                        "type": "steer_captured",
                        "means": means.0,
                    });
                    let _ = writeln!(daemon_state.stdout, "{resp}");
                    let _ = daemon_state.stdout.flush();
                }
                None => emit_error_with_id(
                    &mut daemon_state.stdout,
                    "",
                    "steer_finish_capture: no capture session active".to_string(),
                ),
            },

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
                handlers::status::unsupported_on_request_channel(&mut daemon_state, &msg, &msg_type)
            }
        }
    }
}
