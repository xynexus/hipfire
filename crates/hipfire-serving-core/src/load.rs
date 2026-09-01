// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Model load / unload: turning an HFQ path + load-message
//! params into a [`LoadedModel`], and tearing it down.
//!
//! Covers the per-arch single-GPU `load_model`, the multi-GPU
//! pipeline-parallel `load_model_pp`,
//! `unload_model`, the optional DFlash drafter (`load_dflash_state`), and the
//! small config/metadata helpers (chat-template resolution, state-quant parsing,
//! parameter counting, tiny-model bring-up). Extracted verbatim from the former
//! `main.rs` monolith (no behavior change); items called from `main.rs` are
//! `pub`.

use std::path::{Path, PathBuf};

use hipfire_arch_cohere2 as _;
use hipfire_arch_deepseek4 as deepseek4;
use hipfire_arch_gemma3::{Gemma3Backend, Gemma3State};
use hipfire_arch_gemma3_vl::{load_vl, Gemma3VlBackend, LoadedVl};
#[cfg(feature = "arch-lfm2moe")]
use hipfire_arch_lfm2moe as lfm2moe;
use hipfire_arch_llama::Llama;
use hipfire_arch_minimax as minimax;
use hipfire_arch_qwen2::qwen2;
use hipfire_arch_qwen35::qwen35;
use hipfire_arch_qwen35::qwen35::{DeltaNetState, LayerType, Qwen35ScratchSet};
use hipfire_arch_qwen35::speculative::{
    DdtreeScratch, DeltaNetSnapshot, GdnTape, HiddenStateRingBuffer, VerifyScratch,
};
use hipfire_arch_qwen35_vl::qwen35_vl;
use hipfire_model::{
    arch_features, is_qwen35_dense_arch_id, is_qwen35_family_arch_id, FeatureSupport,
    ARCH_ID_DEEPSEEK4_FLASH, ARCH_ID_DOTS_OCR, ARCH_ID_EMBEDDINGGEMMA, ARCH_ID_GEMMA3_TEXT,
    ARCH_ID_GEMMA3_VL, ARCH_ID_LFM2_MOE, ARCH_ID_MAMBA2, ARCH_ID_MINIMAX_M2, ARCH_ID_NEMOTRON_H,
    ARCH_ID_ZAYA,
};
use hipfire_prompt as prompt_frame;
use hipfire_runtime::cask::CaskCtx;
use hipfire_runtime::dflash::{DflashConfig, DflashScratch, DflashSource, DflashWeights};
use hipfire_runtime::hfq::{HfqFile, HfqTensorInfo};
use hipfire_runtime::hfq_compose::{
    compose_manifest_from_metadata, file_component_view, ComposeManifest,
};
use hipfire_runtime::kv;
use hipfire_runtime::llama;
use hipfire_runtime::multi_gpu::Gpus;
use hipfire_runtime::quant::QuantType;
use hipfire_runtime::triattn::{EvictionCtx, TriAttnArtifact, TriAttnCenters};

use crate::embedding_runtime::{classify_embedding_workload, EmbeddingRuntimeKind};
use crate::memory::hfq_model_memory;
use crate::model::CaskConfig;
#[cfg(feature = "arch-lfm2moe")]
use crate::model::Lfm2DflashState;
use crate::model::{
    DdtreeState, DflashState, DsparkState, EmbeddingGemmaState, Eviction, LoadedModel,
    ResidentSession,
};
use crate::qwen3_embedding::Qwen3EmbeddingState;
use crate::session::{
    next_qwen35_state_allocation_epoch, SessionRegistry, QWEN35_LEGACY_SESSION_ID,
};
use hipfire_runtime::sequence_state::SequenceState;

/// Matrix-backed admission gate: refuse a request for `feature` on a model whose
/// arch capability matrix (the generated `arch_features`, source
/// `docs/model-support.toml`) does not mark it fully supported. Returns a clean,
/// operator-facing reason. This is the single authority — adding an arch to the
/// matrix as feature-capable enables it here without touching this code.
fn require_arch_feature(
    arch_id: u32,
    feature: &str,
    support: FeatureSupport,
) -> Result<(), String> {
    if support.is_full() {
        return Ok(());
    }
    let f = arch_features(arch_id);
    Err(format!(
        "{feature} requested but arch {} (arch_id={arch_id}) does not support it \
         (capability matrix: {}={}). Reload without it. See MODEL-SUPPORT.md / \
         docs/model-support.toml.",
        f.label,
        feature,
        support.mark()
    ))
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ResolvedComponentSource<'a> {
    Explicit(&'a str),
    Embedded,
    Sibling(PathBuf),
}

impl ResolvedComponentSource<'_> {
    fn path(&self) -> Option<&Path> {
        match self {
            Self::Explicit(path) => Some(Path::new(path)),
            Self::Sibling(path) => Some(path),
            Self::Embedded => None,
        }
    }

    fn label(&self) -> &'static str {
        match self {
            Self::Explicit(_) => "explicit",
            Self::Embedded => "embedded",
            Self::Sibling(_) => "sibling",
        }
    }
}

fn resolve_component_source<'a>(
    explicit: Option<&'a str>,
    embedded: bool,
    sibling: Option<PathBuf>,
    off: bool,
) -> Option<ResolvedComponentSource<'a>> {
    if off {
        None
    } else if let Some(explicit) = explicit {
        Some(ResolvedComponentSource::Explicit(explicit))
    } else if embedded {
        Some(ResolvedComponentSource::Embedded)
    } else {
        sibling.map(ResolvedComponentSource::Sibling)
    }
}

fn manifest_has_role(manifest: Option<&ComposeManifest>, role: &str) -> bool {
    manifest.is_some_and(|manifest| manifest.components.iter().any(|item| item.tag == role))
}

enum LoadedTriAttn {
    Uniform(TriAttnCenters),
    Layered(TriAttnArtifact),
}

impl LoadedTriAttn {
    fn uniform_centers(&self) -> Result<TriAttnCenters, String> {
        match self {
            Self::Uniform(centers) => Ok(centers.clone()),
            Self::Layered(artifact) => artifact
                .to_uniform_centers()
                .map_err(|error| format!("CASK package is not uniform: {error}")),
        }
    }

    fn layered(&self) -> Option<&TriAttnArtifact> {
        match self {
            Self::Layered(artifact) => Some(artifact),
            Self::Uniform(_) => None,
        }
    }
}

fn load_resolved_triattn(
    source: &ResolvedComponentSource<'_>,
    bundle: &HfqFile,
    manifest: Option<&ComposeManifest>,
) -> Result<LoadedTriAttn, String> {
    fn load_path(path: &Path) -> Result<LoadedTriAttn, String> {
        use std::io::Read as _;
        let mut file = std::fs::File::open(path)
            .map_err(|error| format!("open {}: {error}", path.display()))?;
        let mut magic = [0u8; 4];
        file.read_exact(&mut magic)
            .map_err(|error| format!("read {} magic: {error}", path.display()))?;
        match &magic {
            b"TRIA" => TriAttnCenters::load(path)
                .map(LoadedTriAttn::Uniform)
                .map_err(|error| format!("parse TRIA v1 {}: {error}", path.display())),
            b"HFQM" => TriAttnArtifact::load_hfqm(path)
                .map(LoadedTriAttn::Layered)
                .map_err(|error| format!("parse TriAttention HFQM {}: {error}", path.display())),
            _ => Err(format!(
                "{} is neither a TRIA v1 nor TriAttention HFQM artifact",
                path.display()
            )),
        }
    }
    match source {
        ResolvedComponentSource::Explicit(path) => load_path(Path::new(path))
            .map_err(|error| format!("explicit CASK/TriAttention load failed ({path}): {error}")),
        ResolvedComponentSource::Sibling(path) => load_path(path).map_err(|error| {
            format!(
                "sibling CASK/TriAttention load failed ({}): {error}",
                path.display()
            )
        }),
        ResolvedComponentSource::Embedded => {
            let manifest = manifest.ok_or("embedded triattn selected without a manifest")?;
            let view = file_component_view(bundle, manifest, "triattn")
                .map_err(|error| format!("embedded triattn view: {error}"))?
                .ok_or("embedded triattn role disappeared from manifest")?;
            view.verify_digest()
                .map_err(|error| format!("embedded triattn digest: {error}"))?;
            match view.source_format() {
                "tria-v1" => {
                    let bytes = view
                        .opaque_bytes_vec()
                        .map_err(|error| format!("embedded triattn payload: {error}"))?
                        .ok_or("embedded triattn has no opaque payload")?;
                    TriAttnCenters::from_bytes(&bytes)
                        .map(LoadedTriAttn::Uniform)
                        .map_err(|error| format!("embedded triattn parse: {error}"))
                }
                "hfqm" => TriAttnArtifact::from_source(&view)
                    .map(LoadedTriAttn::Layered)
                    .map_err(|error| format!("embedded TriAttention HFQM parse: {error}")),
                other => Err(format!(
                    "embedded triattn uses unsupported source format {other:?}"
                )),
            }
        }
    }
}

fn load_resolved_dflash_config(
    source: &ResolvedComponentSource<'_>,
    bundle: &HfqFile,
    manifest: Option<&ComposeManifest>,
) -> Result<DflashConfig, String> {
    match source {
        ResolvedComponentSource::Explicit(path) => {
            let file = HfqFile::open(Path::new(path))
                .map_err(|error| format!("open explicit DFLASH {path}: {error}"))?;
            DflashConfig::from_hfq(&file).ok_or_else(|| {
                format!("explicit DFLASH {path} has invalid arch or metadata/config")
            })
        }
        ResolvedComponentSource::Sibling(path) => {
            let file = HfqFile::open(path)
                .map_err(|error| format!("open sibling DFLASH {}: {error}", path.display()))?;
            DflashConfig::from_hfq(&file).ok_or_else(|| {
                format!(
                    "sibling DFLASH {} has invalid arch or metadata/config",
                    path.display()
                )
            })
        }
        ResolvedComponentSource::Embedded => {
            let manifest = manifest.ok_or("embedded DFLASH selected without a manifest")?;
            let view = file_component_view(bundle, manifest, "dflash")
                .map_err(|error| format!("embedded DFLASH view: {error}"))?
                .ok_or("embedded DFLASH role disappeared from manifest")?;
            DflashConfig::from_source(&view)
                .ok_or_else(|| "embedded DFLASH has invalid arch or metadata/config".to_string())
        }
    }
}

#[cfg(feature = "arch-lfm2moe")]
fn lfm2_triattn_kv_layer_ids(config: &lfm2moe::config::Lfm2MoeConfig) -> Vec<usize> {
    (0..config.num_attention_layers()).collect()
}

/// Resolve the effective chat template for a model: the HFQ-embedded
/// `tokenizer_config.chat_template`, with sidecar/path fallbacks. `None` when
/// the model ships no template (base/completion model).
pub fn resolve_chat_template(
    hfq: &hipfire_runtime::hfq::HfqFile,
    model_path: &str,
) -> Option<String> {
    match prompt_frame::resolve_chat_template(model_path, hfq.chat_template()) {
        Some(resolved) => {
            prompt_frame::log_resolved_chat_template_source(&resolved.source);
            Some(resolved.template)
        }
        None => {
            // Defensive tripwire. A `None` here means `effective_raw()` will
            // default to true (see model::effective_raw) and the daemon serves
            // the prompt RAW/unframed — correct for a base/completion model, but
            // a disaster for a chat model: a bare prompt has no
            // `<|im_start|>assistant` decode anchor and collapses heavily-quantized
            // chat models into a token attractor. The normal quantize path always
            // embeds the template (folding `chat_template.jinja`, and requant
            // carries metadata forward), so a chat arch arriving template-less
            // means a hand-built / corrupted HFQ. Warn loudly rather than
            // silently mis-serving. Scoped to the qwen3.5/3.6 family (arch_id
            // 5/6), the known chat arch this guards; base models of other arches
            // legitimately resolve no template and must not be warned on.
            if hipfire_model::is_qwen35_family_arch_id(hfq.arch_id) {
                tracing::warn!(
                    "chat arch (arch_id={}) resolved NO chat template \
                     for {model_path} — the daemon will serve RAW/unframed prompts, which can \
                     collapse a chat model into a token attractor. Provide one via \
                     HIPFIRE_CHAT_TEMPLATE_FILE, ~/.hipfire/templates/<model-basename>.j2, or \
                     re-quantize from a source that embeds tokenizer_config.chat_template.",
                    hfq.arch_id
                );
            }
            None
        }
    }
}

/// Parse a resolved chat-template string into a `ChatTemplateProfile` (the
/// stop-token / holdback / framing metadata the output filter and prompt
/// framing consume).
pub fn profile_chat_template(
    chat_template: Option<String>,
    tokenizer: Option<&hipfire_model::tokenizer::Tokenizer>,
) -> (Option<String>, Option<prompt_frame::ChatTemplateProfile>) {
    let profile = match (chat_template.as_deref(), tokenizer) {
        (Some(template), Some(tokenizer)) => {
            match prompt_frame::ChatTemplateProfile::from_template(tokenizer, template) {
                Ok(profile) => Some(profile),
                Err(e) => {
                    tracing::warn!("failed to profile template ({e}); using fallback stop policy");
                    None
                }
            }
        }
        _ => None,
    };
    (chat_template, profile)
}

/// Parse the DeltaNet state-quant mode from a load-message param string.
///
/// Absent or `auto` resolves through the redundancy gate
/// (`qwen35::default_state_quant`), which yields **FP32** for all current models.
///
/// **Q8 is REMOVED (2026-08-09)** — the variants, the kernels and their dispatch
/// entry points are all deleted. An explicit `q8`/`q4` is REJECTED, not mapped
/// to FP32: an operator who set it deliberately should learn it is gone rather
/// than get different numerics without being told.
///
/// This previously mapped absent/`""`/`auto` straight to Q8 — bypassing the gate
/// entirely — while its own doc comment claimed it fell "back to the arch
/// default". It did not, and that is why production ran Q8 state despite the gate
/// being configured (`fp32_below = usize::MAX`) to select FP32 everywhere.
pub fn parse_state_quant(
    mode: Option<&str>,
    config: &hipfire_arch_qwen35::qwen35::Qwen35Config,
) -> Result<hipfire_arch_qwen35::qwen35::StateQuant, String> {
    use hipfire_arch_qwen35::qwen35::{default_state_quant, StateQuant};
    match mode.unwrap_or("auto").to_ascii_lowercase().as_str() {
        "" | "auto" => Ok(default_state_quant(config)),
        "fp32" | "f32" => Ok(StateQuant::FP32),
        "fp16" | "f16" => Ok(StateQuant::FP16),
        // Removed 2026-08-09. Rejected rather than silently mapped to FP32: an
        // operator who set q8 deliberately should learn it is gone, not get
        // different numerics without being told.
        "q8" | "int8" | "q4" | "int4" => Err(format!(
            "DeltaNet state_quant '{}' was removed — quantized recurrent state \
             caused silent corruption (long-decode attractors, a rounding seed \
             leaking execution history, and an error-feedback buffer that broke \
             spec-decode rollback). Use auto|fp32|fp16.",
            mode.unwrap_or("")
        )),
        other => Err(format!(
            "unsupported DeltaNet state_quant '{other}' (expected auto|fp32|fp16)"
        )),
    }
}

/// Human-readable label for a `StateQuant` (for the `loaded` event / status).
pub fn state_quant_label(q: hipfire_arch_qwen35::qwen35::StateQuant) -> &'static str {
    use hipfire_arch_qwen35::qwen35::StateQuant;
    match q {
        StateQuant::FP32 => "FP32",
        StateQuant::FP16 => "FP16",
    }
}

/// Total parameter count across an HFQ's tensors (summed element counts), for
/// the reported model size.
pub fn hfq_parameter_count(hfq: &HfqFile) -> u128 {
    hfq.tensors()
        .iter()
        .map(|t| {
            t.shape
                .iter()
                .fold(1u128, |acc, &dim| acc.saturating_mul(dim as u128))
        })
        .sum()
}

/// True if any tensor in the HFQ is stored as bf16 — gates the bf16-capable
/// load path / dtype handling.
pub fn hfq_has_bf16_weights(hfq: &HfqFile) -> bool {
    hfq.tensors().iter().any(|t| t.quant_type == 16)
}

/// Log a one-line notice when the model carries canonical OQ4 weights
/// (quant_type 34) that the loader repacks into the arch-combined device layout
/// on EVERY load. `hipfire optimize` prebakes that layout (34 -> 37) so later
/// loads upload verbatim with no per-load repack. Index-level scan only — no GPU
/// work; called once per load from both the single-GPU and pipeline-parallel
/// load paths.
fn warn_if_unoptimized(path: &str, hfq: &HfqFile) {
    let canonical_oq4 = hfq
        .tensors()
        .iter()
        .filter(|t| t.quant_type == hipfire_runtime::hfq::OQ4_CANONICAL_QT)
        .count();
    if canonical_oq4 > 0 {
        tracing::warn!(
            "'{path}' has {canonical_oq4} canonical OQ4 tensor(s) repacked per load; \
             run `hipfire optimize {path}` to prebake the arch-optimal layout and skip the per-load repack"
        );
    }
}

/// True only when the model is *predominantly* BF16 (a full-precision artifact),
/// not merely a quantized model that keeps a few small tensors (norms) at BF16.
/// Decided on the 2-D weight tensors (the matmul projections): a full BF16 model
/// has them all BF16 (qt==16); an MQ4/Q8 model has them quantized with only 1-D
/// norms left at BF16. Used to decide whether to FORCE fp32 KV — quantized
/// models must NOT be forced (it locks them out of batched prefill).
pub fn hfq_is_bf16_dominant(hfq: &HfqFile) -> bool {
    let (mut bf16_2d, mut total_2d) = (0usize, 0usize);
    for t in hfq.tensors() {
        if t.shape.len() == 2 {
            total_2d += 1;
            if t.quant_type == 16 {
                bf16_2d += 1;
            }
        }
    }
    total_2d > 0 && bf16_2d * 2 > total_2d
}

/// Read the model's trained context window (`max_position_embeddings`) from the
/// HFQ metadata JSON. Handles the multimodal wrapper shape (gemma3-vl etc.)
/// where the decoder config is nested under `text_config`.
fn model_max_position_embeddings(metadata_json: &str) -> Option<usize> {
    let v: serde_json::Value = serde_json::from_str(metadata_json).ok()?;
    let read = |obj: &serde_json::Value| {
        obj.get("max_position_embeddings")
            .and_then(|x| x.as_u64())
            .map(|x| x as usize)
    };
    // HFQ metadata wraps the HF config under `config`; multimodal wrappers
    // (gemma3-vl etc.) nest the decoder shape under `config.text_config`. Also
    // accept the field at the top level for formats that hoist it. Mirrors
    // `hipfire_arch_gemma3::config_from_metadata_json`.
    let config = v.get("config");
    None.or_else(|| config.and_then(|c| c.get("text_config")).and_then(read))
        .or_else(|| config.and_then(read))
        .or_else(|| v.get("text_config").and_then(read))
        .or_else(|| read(&v))
}

/// Clamp a requested `max_seq` to the model's trained context window. Allocating
/// KV for more than `max_position_embeddings` wastes memory (and on RDNA APUs
/// can OOM the shared GTT pool) for context the model was never trained to use.
/// An operator can opt out with `HIPFIRE_MAX_SEQ_ALLOW_OVERRIDE=1` (a warning is
/// printed either way).
fn clamp_max_seq_to_model_context(max_seq: usize, metadata_json: &str) -> usize {
    match model_max_position_embeddings(metadata_json) {
        Some(model_max) if max_seq > model_max => {
            if std::env::var("HIPFIRE_MAX_SEQ_ALLOW_OVERRIDE")
                .ok()
                .as_deref()
                == Some("1")
            {
                tracing::warn!(
                    "max_seq={max_seq} exceeds model max_position_embeddings={model_max}; \
                     HIPFIRE_MAX_SEQ_ALLOW_OVERRIDE=1 set — proceeding with {max_seq} \
                     (may exceed trained context and/or OOM the KV allocation)"
                );
                max_seq
            } else {
                tracing::warn!(
                    "max_seq={max_seq} exceeds model max_position_embeddings={model_max}; \
                     clamping to {model_max}. Set HIPFIRE_MAX_SEQ_ALLOW_OVERRIDE=1 to force the larger value."
                );
                model_max
            }
        }
        _ => max_seq,
    }
}

/// Budget cap for gemma3 context on shared-GTT RDNA APUs.
///
/// gemma3 now uses sliding-window attention: the 5-of-6 LOCAL layers keep only a
/// `sliding_window` (1024) ring, so their KV is fixed and tiny. The remaining
/// term is the ~10 GLOBAL layers, which still carry a full-context cache. At
/// medgemma's default F32 KV that is ~1.6 MB/token across the global layers, so
/// full 128K would be ~21 GB of global KV + ~15 GB weights — still over the
/// ~32 GB effective budget (the 43 GB GTT is shared with the host OS). q8 global
/// KV (`kv_mode=q8`) drops that ~4× and reaches full context; until that is the
/// default, cap F32 gemma3 at a budget-safe context. Operators can force the
/// full value with `HIPFIRE_MAX_SEQ_ALLOW_OVERRIDE=1`.
///
/// 65536: ~10.7 GB global F32 KV + ~0.9 GB local rings + ~15 GB weights ≈ 27 GB.
const GEMMA3_STOPGAP_MAX_SEQ: usize = 65_536;

/// Map an operator `kv_mode` string to the gemma3 KV cache mode + KVarN bit
/// width. gemma3's state wires F32, Q8 (`q8`/`int8`), and KVarN (variance-
/// normalized K + Q8 V) at `kvarn2`/`kvarn`(=4)/`kvarn8`; the rotated asym/fwht
/// tiers have no gemma3 kernel yet and fall back to F32. The `usize` is the
/// KVarN K bit width (meaningful only for the Kvarn mode; 4 otherwise).
fn gemma3_kv_mode(kv_mode: &str) -> (hipfire_runtime::kv::KvQuantMode, usize) {
    use hipfire_runtime::kv::KvQuantMode;
    match kv_mode {
        "q8" | "int8" => (KvQuantMode::Q8, 4),
        "kvarn2" => (KvQuantMode::Kvarn, 2),
        "kvarn" | "kvarn4" => (KvQuantMode::Kvarn, 4),
        "kvarn8" => (KvQuantMode::Kvarn, 8),
        _ => (KvQuantMode::Unquantized, 4),
    }
}

/// KVarN K-code bit width for a `kvarn*` kv_mode token — the shared-`KvCache`
/// analog of the kvarn arm in [`gemma3_kv_mode`], so the qwen3.5/llama serving
/// paths expose the full KVarN menu (`kvarn8` near-lossless, `kvarn`/`kvarn4`
/// the default, `kvarn2` the aggressive cold/CASK tier) instead of only the
/// hard-coded 4-bit tier. Non-kvarn tokens never reach this (the match arm
/// guards on the `kvarn*` patterns), so the fallback is a harmless 4.
/// KV modes being retired down to two families: **KVarN** (quantized, fused
/// write + causal flash) and the **unquantized** family (fp32 today; bf16/fp16
/// do not exist in `KvQuantMode` yet and are the gap to fill).
///
/// Everything else — q4/q8, int8/int8c, hfq4kv/hfq8, asym2/3/4, fwht2/3/4 — is a
/// separate storage layout with its own constructors and its own prefill arm, and
/// that combinatorial spread is what makes the KV surface 51 constructors across
/// 3000 lines.
///
/// Gated rather than deleted first, on purpose: refusing at load names every
/// model and config that still depends on a retired mode, which is the point —
/// the breakage IS the inventory. Deleting the constructors first would surface
/// the same information as a wall of compile errors with no runtime context.
///
/// `HIPFIRE_KV_ALLOW_DEPRECATED=1` re-admits them for a transition period.
/// The list itself lives in `hipfire-config` as `DEPRECATED_KV_MODES`, so config
/// resolution warns about the same modes this refuses (issue #386) rather than
/// holding a second opinion that only disagrees at load time.
fn reject_deprecated_kv_mode(kv_mode: &str) -> Result<(), String> {
    if !hipfire_config::DEPRECATED_KV_MODES.contains(&kv_mode) {
        return Ok(());
    }
    if std::env::var("HIPFIRE_KV_ALLOW_DEPRECATED").ok().as_deref() == Some("1") {
        tracing::warn!(
            "kv_mode={kv_mode} is deprecated and will be removed; running only \
             because HIPFIRE_KV_ALLOW_DEPRECATED=1. Migrate to kvarn (kvarn2/kvarn/kvarn8) \
             or fp32."
        );
        return Ok(());
    }
    Err(format!(
        "kv_mode={kv_mode} is deprecated. hipfire is retiring KV storage down to two \
         families: kvarn (kvarn2 / kvarn / kvarn4 / kvarn8) and unquantized (fp32). \
         Set HIPFIRE_KV_ALLOW_DEPRECATED=1 to run it during migration, or pick a \
         supported mode."
    ))
}

/// KV modes whose cache TriAttention/CASK eviction can actually compact.
///
/// `EvictionCtx::maybe_evict` dispatches on the `KvCache` flags `quant_asym3`,
/// `quant_asym4`, `quant_asym2` and `quant_q8`, and has no arm for anything
/// else. These are the `kv_mode` tokens that set one of those four:
/// `asym2/3/4` and `fwht2/3/4` (which are asym with a rotation, so they set
/// `quant_asym*` too) and `q8`.
///
/// Everything else — `fp32`/unquantized and every `kvarn*` — reaches no arm.
/// This is deliberately a WHITELIST: an unrecognised or newly added mode
/// declines the sidecar (losing eviction, still serving) rather than failing
/// the request the moment the cache reaches `budget + beta`.
pub(crate) fn triattn_can_evict_kv_mode(kv_mode: &str) -> bool {
    matches!(
        kv_mode,
        "q8" | "asym2" | "asym3" | "asym4" | "fwht2" | "fwht3" | "fwht4"
    )
}

pub(crate) fn kvarn_bits_from_mode(kv_mode: &str) -> usize {
    match kv_mode {
        "kvarn8" => 8,
        "kvarn2" => 2,
        _ => 4, // "kvarn" | "kvarn4"
    }
}

fn cap_gemma3_stopgap_max_seq(max_seq: usize, arch_id: u32, kv_mode: &str) -> usize {
    let is_gemma3 = arch_id == ARCH_ID_GEMMA3_TEXT || arch_id == ARCH_ID_GEMMA3_VL;
    if !is_gemma3 {
        return max_seq;
    }
    // q8/int8/kvarn global KV is ~4× smaller than F32, so with sliding-window
    // local layers the full trained context fits — no gemma3-specific cap needed
    // (the model-context clamp still applies). Only F32 global KV needs the cap.
    let quantized_global = matches!(kv_mode, "q8" | "int8" | "kvarn");
    let cap = if quantized_global {
        max_seq
    } else {
        GEMMA3_STOPGAP_MAX_SEQ
    };
    if max_seq <= cap {
        return max_seq;
    }
    if std::env::var("HIPFIRE_MAX_SEQ_ALLOW_OVERRIDE")
        .ok()
        .as_deref()
        == Some("1")
    {
        tracing::warn!(
            "gemma3 max_seq={max_seq} — the global-layer F32 KV alone may OOM the \
             shared GTT pool at this context; HIPFIRE_MAX_SEQ_ALLOW_OVERRIDE=1 set — proceeding \
             (use kv_mode=q8 for full 128K)."
        );
        max_seq
    } else {
        tracing::warn!(
            "capping gemma3 max_seq {max_seq} -> {GEMMA3_STOPGAP_MAX_SEQ} to fit the GTT \
             pool (sliding-window KV bounds local layers; global F32 KV still scales with context \
             — use kv_mode=q8 for full 128K). Override: HIPFIRE_MAX_SEQ_ALLOW_OVERRIDE=1."
        );
        GEMMA3_STOPGAP_MAX_SEQ
    }
}

// Auto-upgrade DeltaNet state to FP32 for low-redundancy models when the caller
// has not made an explicit non-default choice. Q8/Q4 state accumulates quality
// drift on long outputs; the recurrent state is the model's numerical anchor
// (quant error compounds across the sequence), and low-redundancy models lack
// the head capacity to absorb it. Gate keys on DeltaNet state redundancy
// (`linear_key_head_dim × linear_num_value_heads`) rather than parameter count
// — this directly measures state capacity (0.8B=2048, 9B=4096, 27B=6144) and
// scales with the actual sensitivity. Threshold: HIPFIRE_DN_STATE_FP32_BELOW
// (default usize::MAX ⇒ FP32 for all current models; see TODO.md / NEXT-STEPS).
// An explicit override (state_quant="q8"/"q4" in JSON params) is honoured.
/// Bring-up helper for the tiny smoke-test models: resolve their abbreviated
/// config into the concrete state needed to load them.
pub fn resolve_tiny_model_state(
    _hfq: &HfqFile,
    _override_str: Option<&str>,
    q: hipfire_arch_qwen35::qwen35::StateQuant,
) -> hipfire_arch_qwen35::qwen35::StateQuant {
    // Pass-through. This used to second-guess the caller via the DeltaNet
    // redundancy threshold, promoting an unspecified state to Q8 above a size
    // cutoff. Q8 is gone, the threshold with it, and the remaining precisions
    // are both valid everywhere — so there is nothing left to resolve.
    //
    // It was also the reason FP16 could not be selected: with `q = FP16` and no
    // explicit override, the old body fell through to the threshold and handed
    // back FP32, silently. `HIPFIRE_DN_STATE_FP16=1` reached
    // `default_state_quant`, produced FP16, and this function threw it away.
    q
}

/// `MemAvailable` from `/proc/meminfo`, in bytes.
///
/// `MemAvailable`, not `MemFree`: reclaimable page cache is genuinely available
/// to a load, and on this box the cache is routinely tens of GiB. Reading
/// `MemFree` would refuse loads that fit comfortably.
fn mem_available_bytes() -> Option<u64> {
    let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
    meminfo.lines().find_map(|line| {
        let kb: u64 = line
            .strip_prefix("MemAvailable:")?
            .split_whitespace()
            .next()?
            .parse()
            .ok()?;
        Some(kb * 1024)
    })
}

/// Decide whether a load of `need` bytes may proceed. Split out from
/// [`check_load_headroom`] so the arithmetic is testable without `/proc`.
fn load_headroom_verdict(need: u64, available: u64, reserve: u64) -> Result<(), String> {
    if need.saturating_add(reserve) <= available {
        return Ok(());
    }
    let gib = |b: u64| b as f64 / (1 << 30) as f64;
    Err(format!(
        "refusing to load: this model needs about {:.1} GiB resident and only {:.1} GiB is \
         available (MemAvailable), leaving no room for the {:.1} GiB system reserve. \
         This machine has unified memory -- GPU allocations come out of the same \
         pool as everything else, so overcommitting here does not fail the loader, \
         it invokes the OOM killer on whatever else is running. \
         Free memory, or set HIPFIRE_LOAD_MEM_RESERVE_GIB=<n> to lower the reserve, \
         or HIPFIRE_LOAD_MEM_CHECK=0 to skip this check entirely.",
        gib(need),
        gib(available),
        gib(reserve),
    ))
}

/// Refuse a load that cannot fit in available system memory.
///
/// Loading is the one place that commits tens of GiB in a single unattended
/// step, and on a UMA APU there is no separate VRAM pool to run out of first:
/// an over-large load walks `MemAvailable` to zero and the kernel reaps whatever
/// it likes. Observed on halo: a 68 GB artifact against ~40 GB of headroom took
/// out dbus, pipewire, `systemd --user`, and both agent processes -- the loader
/// itself was not even the first victim, which is exactly why the loader has to
/// be the one to refuse.
///
/// The estimate is NOT the artifact's on-disk size. That was the first version
/// of this guard and it was wrong by 3.5x on exactly the models that need it:
/// it admitted the 122B, which then exhausted GTT at layer 35 of 48. Routed MoE
/// experts cost far more resident than stored, for two compounding reasons that
/// [`estimated_module_resident_bytes`] prices -- a compact-to-Oq8 expansion on
/// load, and the driver's 2 MiB GTT rounding applied per expert tensor.
///
/// Still approximate: it omits the KV cache and scratch (the reserve absorbs
/// those, ~6 GiB measured across three models) and over-counts when
/// `paged_experts` is on and most experts stay on host. A guard against the
/// catastrophic case, not a precise admission test. Anything unreadable (no
/// `/proc`, no `stat`, an index that will not parse) skips the check: never
/// block a load because a diagnostic could not be read.
fn check_load_headroom(path: &str) -> Result<(), String> {
    if std::env::var("HIPFIRE_LOAD_MEM_CHECK").as_deref() == Ok("0") {
        return Ok(());
    }
    if !hipfire_config::load_config_bundle().config.load_mem_check {
        return Ok(());
    }
    let Ok(meta) = std::fs::metadata(path) else {
        return Ok(());
    };
    let Some(available) = mem_available_bytes() else {
        return Ok(());
    };
    // Price the routed-expert modules at what they will actually occupy, and the
    // rest of the file at its disk size. An artifact with no module table (any
    // dense model) leaves this equal to the file length, which the dense
    // measurements support: 1.37x on a 27B, where the excess is the fixed
    // overhead the reserve already covers.
    let need = match HfqFile::open_index_only(Path::new(path)) {
        Ok(index) => {
            let (resident, on_disk) =
                hipfire_runtime::weight_pager::estimated_module_resident_bytes(&index);
            meta.len().saturating_sub(on_disk).saturating_add(resident)
        }
        Err(_) => meta.len(),
    };
    let reserve = std::env::var("HIPFIRE_LOAD_MEM_RESERVE_GIB")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .map(|gib| gib << 30)
        .unwrap_or_else(|| {
            (hipfire_config::load_config_bundle()
                .config
                .load_mem_reserve_gib as u64)
                << 30
        });
    load_headroom_verdict(need, available, reserve)
}

#[cfg(test)]
mod load_headroom_tests {
    use super::load_headroom_verdict;

    const GIB: u64 = 1 << 30;

    #[test]
    fn fits_with_reserve_to_spare() {
        assert!(load_headroom_verdict(60 * GIB, 100 * GIB, 4 * GIB).is_ok());
    }

    #[test]
    fn exactly_meeting_the_reserve_is_allowed() {
        assert!(load_headroom_verdict(96 * GIB, 100 * GIB, 4 * GIB).is_ok());
    }

    #[test]
    fn eating_into_the_reserve_is_refused() {
        // The halo case: a 68 GiB artifact against 40 GiB of headroom.
        let err = load_headroom_verdict(68 * GIB, 40 * GIB, 4 * GIB).unwrap_err();
        assert!(err.contains("68.0 GiB"), "names the artifact size: {err}");
        assert!(err.contains("40.0 GiB"), "names what is available: {err}");
        assert!(
            err.contains("HIPFIRE_LOAD_MEM_CHECK=0"),
            "names the override: {err}"
        );
    }

    #[test]
    fn a_huge_estimate_does_not_wrap_into_a_pass() {
        assert!(load_headroom_verdict(u64::MAX, 100 * GIB, 4 * GIB).is_err());
    }
}

/// Load a model from an HFQ path + load-message params into a [`LoadedModel`]:
/// detect the arch, parse its config, upload weights and allocate the forward
/// scratch/KV/state for that family, resolve the chat template and eviction
/// policy, and wire any optional DFlash drafter. The single-GPU entry point
/// (multi-GPU goes through [`load_model_pp`]).
pub fn load_model(
    path: &str,
    max_seq: usize,
    requested_physical_cap: Option<usize>,
    draft_path: Option<&str>,
    dflash_mode: Option<&str>,
    kv_mode_override: Option<&str>,
    state_quant_override: Option<&str>,
    cask: &CaskConfig,
    pp: usize,
    gpu: &mut hipfire_rdna::Gpu,
) -> Result<LoadedModel, String> {
    check_load_headroom(path)?;
    // The serving path takes hipfire containers only. A HuggingFace
    // safetensors directory is an external format: converting it is offline
    // tooling's job (AGENTS.md — "conversion and compatibility concerns are
    // offline tooling"), and routing it here silently misclassified every
    // non-ParoQuant checkpoint, failing deep in the loader with the unrelated
    // "ParoQuant model must have quantization_config".
    if Path::new(path).is_dir() {
        return Err(format!(
            "{path} is a directory; the daemon loads .hfq containers only. \
             Convert it first: `hipfire-quantize --input {path} --output <model>--bf16.hfq --format bf16` \
             (or another --format). An .hfa archive unpacks with `hipfire-coexistence repack`."
        ));
    }
    if pp > 1 {
        // Refusal contracts (DFlash, CASK sidecar) are enforced upstream in
        // the "load" event handler so the operator gets a structured error
        // before any HFQ open / weight allocation. By the time we get here
        // with pp>1, draft_path is None and cask.sidecar is None.
        let index = HfqFile::open_index_only(Path::new(path)).map_err(|error| error.to_string())?;
        let manifest = compose_manifest_from_metadata(&index.metadata_json)
            .map_err(|error| format!("embedded component manifest: {error}"))?;
        if dflash_mode != Some("off") && manifest_has_role(manifest.as_ref(), "dflash") {
            return Err("embedded DFLASH requires pp=1".to_string());
        }
        if std::env::var("HIPFIRE_CASK_OFF").ok().as_deref() != Some("1")
            && manifest_has_role(manifest.as_ref(), "triattn")
        {
            return Err("embedded CASK/TriAttention requires pp=1".to_string());
        }
        let _ = (draft_path, cask);
        return load_model_pp(
            path,
            max_seq,
            kv_mode_override,
            state_quant_override,
            pp,
            gpu,
        );
    }
    // Per-load kv_mode (sent in load message params) overrides the env var.
    // Lets the CLI set size-aware defaults — e.g. Qwen3.5-27B prefers asym4
    // since layer-count compounding of asym3 noise flips argmax at decision
    // boundaries on deep stacks.
    let mut kv_mode = kv_mode_override
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .unwrap_or_else(|| std::env::var("HIPFIRE_KV_MODE").unwrap_or_default());
    // `auto` (the config default) and unset both mean "the loader chooses".
    // KVarN: ~8x smaller K than fp32 with no token dropped, and batched-prefill
    // eligible. fp32 is an oracle/debug mode, not a production path — ask for it
    // explicitly (kv_cache=fp32) if you want it.
    if kv_mode.is_empty() || kv_mode == "auto" {
        kv_mode = "kvarn".to_string();
    }
    let mut hfq = HfqFile::open(Path::new(path)).map_err(|e| format!("{e}"))?;
    let compose_manifest = compose_manifest_from_metadata(&hfq.metadata_json)
        .map_err(|error| format!("embedded component manifest: {error}"))?;
    let embedded_dflash = manifest_has_role(compose_manifest.as_ref(), "dflash");
    let embedded_triattn = manifest_has_role(compose_manifest.as_ref(), "triattn");
    let dflash_off = dflash_mode.is_some_and(|mode| mode.eq_ignore_ascii_case("off"));
    let dflash_sibling = if draft_path.is_none() && !embedded_dflash && !dflash_off {
        hipfire_model::discover_dflash_draft_for_model(Path::new(path))
    } else {
        None
    };
    let resolved_dflash =
        resolve_component_source(draft_path, embedded_dflash, dflash_sibling, dflash_off);
    if dflash_mode.is_some_and(|mode| mode.eq_ignore_ascii_case("on")) && resolved_dflash.is_none()
    {
        return Err(
            "dflash_mode=on but no explicit, embedded, or sibling DFLASH component was found"
                .to_string(),
        );
    }
    let cask_off = std::env::var("HIPFIRE_CASK_OFF").ok().as_deref() == Some("1");
    let triattn_sibling = if cask.sidecar.is_none() && !embedded_triattn && !cask_off {
        hipfire_model::discover_triattn_for_model(Path::new(path))
    } else {
        None
    };
    let resolved_triattn = resolve_component_source(
        cask.sidecar.as_deref(),
        embedded_triattn,
        triattn_sibling,
        cask_off,
    );
    let dflash_requested = resolved_dflash.is_some();
    // TriAttention/CASK eviction compacts the KV by gathering the retained
    // positions, a per-position copy, and `maybe_evict` has an arm only for
    // asym2/3/4 and q8 (see `triattn_can_evict_kv_mode`). Every other mode used
    // to reach a `panic!` there — and because a sidecar makes the KV allocate at
    // `physical_cap` (budget+beta+256) instead of `max_seq` precisely so
    // eviction can bound VRAM, `maybe_evict` fires the moment a prompt passes
    // that. So under `fp32` or `kvarn`, ANY prompt longer than ~physical_cap
    // killed the daemon mid-generation.
    //
    // KVarN in particular can never be compacted this way: its K is 4-bit
    // var-norm records covering 128-token BLOCKS, so dropping arbitrary
    // positions means re-encoding every surviving block — new kernels, plus a
    // re-quantization of already-quantized values that would compound error on
    // every eviction cycle.
    //
    // So decline the sidecar for anything not known-evictable. Nothing is lost
    // under KVarN, which already bounds KV cost on K by ~8x WITHOUT dropping a
    // token — that is what eviction was there to do. The cache is then sized for
    // the full window; `HIPFIRE_KV_PHYSICAL_CAP` still caps the allocation for
    // operators who need a smaller one.
    let resolved_triattn = match resolved_triattn {
        Some(source) if !triattn_can_evict_kv_mode(&kv_mode) => {
            eprintln!(
                "  TriAttention component ({}) declined: eviction cannot compact \
                 kv_mode={kv_mode} — KV sized for the full context window instead",
                source.label()
            );
            None
        }
        other => other,
    };
    let cask_requested = resolved_triattn.is_some();
    if matches!(resolved_dflash, Some(ResolvedComponentSource::Embedded)) {
        let verify_started = std::time::Instant::now();
        let manifest = compose_manifest
            .as_ref()
            .ok_or("embedded DFLASH selected without a manifest")?;
        let view = file_component_view(&hfq, manifest, "dflash")
            .map_err(|error| format!("embedded DFLASH view: {error}"))?
            .ok_or("embedded DFLASH role disappeared from manifest")?;
        view.verify_digest()
            .map_err(|error| format!("embedded DFLASH digest: {error}"))?;
        DflashConfig::from_source(&view).ok_or("embedded DFLASH metadata/config is invalid")?;
        tracing::info!(
            "embedded DFLASH verified: encoding={} bytes={} sha256={} elapsed={:.2}s",
            view.source_format(),
            view.original_byte_len(),
            view.sha256(),
            verify_started.elapsed().as_secs_f64(),
        );
    }
    let resolved_triattn_centers = resolved_triattn
        .as_ref()
        .map(|source| load_resolved_triattn(source, &hfq, compose_manifest.as_ref()))
        .transpose()?;
    let max_seq = clamp_max_seq_to_model_context(max_seq, &hfq.metadata_json);
    let max_seq = cap_gemma3_stopgap_max_seq(max_seq, hfq.arch_id, &kv_mode);
    let model_memory = hfq_model_memory(path, &hfq);
    warn_if_unoptimized(path, &hfq);
    // KV precision policy:
    //   * BF16-DOMINANT model (full-precision artifact) -> force fp32 KV (mixing
    //     a quantized KV under bf16 weights is a precision mismatch).
    //   * Quantized model (MQ4/Q8 weights, only norms BF16) -> honor an explicit
    //     kv_mode; otherwise default to fp32 for now. A quantized KV (q8/asym/
    //     KVarN) makes the model batched-prefill eligible (~32x prefill); the
    //     prior rule wrongly force-fp32'd these via the BF16 norms, locking them
    //     to the per-token path. That flip is done: the default is now KVarN.
    // KV precision: respect an explicit kv_mode (config/CLI/JSON) for ALL
    // models, including BF16-dominant ones. Default to fp32 only when the
    // caller left it unspecified. (Previously BF16-dominant artifacts were
    // force-overridden to fp32 even when the operator asked for q8/asym/KVarN
    // — that silently discarded the requested KV quant. fp32 is reserved for
    // oracle/debug runs; quantizing KV under bf16 weights is an opt-in the
    // operator owns.)
    reject_deprecated_kv_mode(&kv_mode)?;
    let tokenizer = hipfire_model::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
        .map_err(|e| format!("tokenizer not found: {e}"))?;

    // DFlash speculative-decode requires the target's lm_head to have a
    // batched-GEMM kernel (used for verify and DDTree top-K). Only
    // Q8_0 (qt=3) / HFQ4G256 (qt=6) / MQ4G256 (qt=13) are wired into
    // speculative.rs's `try_batched` predicate (lines 2083-2087,
    // 2606-2609); every other dtype falls through to a per-row sequential
    // GEMV path that hangs spec verify (observed: 1 token in 240 s on
    // 27B MQ3 + mq4.dflash draft).
    //
    // Refuse fast at the HFQ-index level — BEFORE any weight upload, KV
    // alloc, or scratch alloc — so we don't strand ~12 GB of VRAM in the
    // pool when the operator passed a draft against an unsupported target.
    // Read the lm_head tensor's `quant_type` byte directly from the index
    // (no GPU work). lm_head can be a separate tensor or tied to
    // embed_tokens, and the tensor names differ by arch:
    //   - Qwen3.5/3.6 separate: "lm_head.weight" or "model.language_model.lm_head.weight"
    //   - Qwen3.5/3.6 tied:     "model.language_model.embed_tokens.weight"
    //   - LLaMA separate:       "lm_head.weight"
    //   - LLaMA tied:           "model.embed_tokens.weight"
    // Cover all four; the order mirrors what qwen35::load_weights /
    // hfq::load_weights_hfq do at runtime, so the qt we read here is the
    // qt that will end up driving `weights.output.gpu_dtype`.
    if dflash_requested {
        if hfq.arch_id == ARCH_ID_LFM2_MOE {
            #[cfg(not(feature = "arch-lfm2moe"))]
            {
                return Err(
                    "LFM2 DFlash draft requested but arch-lfm2moe is not compiled in".to_string(),
                );
            }
            if std::env::var("HIPFIRE_LFM2_DFLASH").ok().as_deref() != Some("1") {
                return Err(
                    "LFM2 DFlash is experimental; set HIPFIRE_LFM2_DFLASH=1 to load a draft"
                        .to_string(),
                );
            }
            tracing::warn!(
                "LFM2 DFlash is experimental; admission is gated by HIPFIRE_LFM2_DFLASH=1"
            );
        } else {
            // Arch-level capability gate FIRST (matrix-backed). DFlash spec-decode
            // only runs on archs whose matrix marks it Full — the generate() router
            // requires it. Without this an operator could attach a draft to a
            // non-DFlash arch, pass the lm_head dtype check below, then silently get
            // plain AR decode (a no-op draft). Refuse up front with the matrix reason.
            require_arch_feature(
                hfq.arch_id,
                "DFlash spec-decode",
                arch_features(hfq.arch_id).dflash,
            )?;

            // `find_tensor_info`, not `tensor_data`: this only wants the quant
            // type, and the borrowing accessor returns None for a losslessly
            // recoded (LUT3/Huffman) table. That made the whole chain yield None
            // for a tied embed stored compressed, SILENTLY dropping the lm_head
            // out of the batched-kernel whitelist below rather than failing.
            let lm_qt = hfq
                .find_tensor_info("lm_head.weight")
                .or_else(|| hfq.find_tensor_info("model.language_model.lm_head.weight"))
                .or_else(|| hfq.find_tensor_info("model.language_model.embed_tokens.weight"))
                .or_else(|| hfq.find_tensor_info("model.embed_tokens.weight"))
                .map(|info| info.quant_type);
            // MQ3 (qt=17) batched lm_head + WMMA prefill kernels exist on gfx11
            // only (`gemm_hfq3g256_batched_lmhead` + `is_batchable_la` admits MQ3
            // for gfx1100/1101/1102/1150/1151). On other archs, MQ3 lm_head still
            // falls through to per-row GEMV that hangs verify. Whitelist:
            //   - Always: Q8_0=3, HFQ4G256=6, MQ4G256=13
            //   - gfx11 only: MQ3G256=17
            // MQ2 (qt=18) is not yet wired into speculative.rs match arms.
            // MQ3 WMMA family is ported to gfx11 (RDNA3) and gfx12 (RDNA4).
            // Keep them grouped under the same flag — the builtin name differs
            // (_w32 vs _w32_gfx12) but the dispatch wrappers route per-arch.
            let arch_is_gfx11 = matches!(
                gpu.arch.as_str(),
                "gfx1100" | "gfx1101" | "gfx1102" | "gfx1150" | "gfx1151" | "gfx1200" | "gfx1201"
            );
            // Opus Quant (qt 33 OqPlusG256, 34 Oq4G256, 35 Oq8G256, 36
            // OqPlusCompact, 38 Oq3G256, 52 OqPlusCompactG128) all resolve
            // through `oq8_arch_load` to Oq8G256 / OqCompactG256 / G128, and
            // `dflash_enqueue_verify_lm_head` now carries arms for those three
            // (AWQ-aware rotate + quantize_act_oq8 + grouped-iu8 WMMA, on
            // VerifyScratch buffers so it stays graph-capturable). Without
            // these codes every oq* target — i.e. every model we now induct —
            // refused to load with a draft attached.
            // NOTE ON OPUS (qt 33/34/35/36/38/52): the batched lm_head half is
            // DONE, and so is the correctness half. The dense miscompute that used
            // to justify this gate was root-caused on 2026-08-20 and fixed in
            // `b7b7a9ae5` — it was never about Opus. `forward_scratch_layers`
            // skipped the lowered executor whenever a hidden-state ring buffer was
            // requested, so spec-decode verify ran the hand arms in
            // decode_layers.rs, which miscompute on DENSE qwen3.5-family models.
            // The lowered path extracts per-layer hidden itself now, and dense
            // DFlash commits byte-identical text to plain decode.
            //
            // The gate STAYS, for the one reason that survived: measured, DFlash
            // does not pay on this family. On Qwen3.8-27B oq4.25++ + CASK,
            // 128 tokens greedy (docs/experiments/2026-08-20-dflash2-qwen38-27b-performance.md):
            //
            //     baseline          7.50 tok/s
            //     DFlash1 B=16      0.79 tok/s   tau 1.098
            //     DFlash2 B=8       2.01 tok/s   tau 1.778   (q8 KV, batched verify)
            //
            // and the ceiling is arithmetic: verify costs ~6.8 serial decodes to
            // check 9 positions, so even at 100 % acceptance the best case is
            // 9.2 tok/s — 1.22x. 48 of 64 layers are DeltaNet and its recurrence is
            // sequential per token, so a wider verify batch amortizes only the
            // other quarter of the stack. Same shape as the MoE result
            // (Qwen3.5-35B-A3B: 8.17 vs 35.7 tok/s, expert bytes rather than a
            // recurrence). Speculation on this family needs the DeltaNet recurrence
            // batched across the verify block before admitting it by default makes
            // anyone faster.
            //
            // That measurement is why the gate existed; it is kept above as the
            // record of where speculation does NOT pay on this family. It is no
            // longer a load-time refusal: `dflash_mode` decides whether to try,
            // and a per-model override picks the drafter.
            let supported = match lm_qt {
                Some(3 | 6 | 13) => true,
                Some(17) => arch_is_gfx11,
                // Opus Quant. Batched lm_head and correctness are both done —
                // see the note above — and it is measured FASTER here, not
                // slower: Qwen3.5-9B oq4.25++ on gfx1103, greedy, 13.2 -> 29.8
                // tok/s repetitive and 12.8 -> 30.2 prose. The old opt-in
                // generalised a 27B/35B-A3B result to the whole family; those
                // are MoE and DeltaNet-heavy, where a wider verify batch
                // amortizes only part of the stack. A dense target is a
                // different shape.
                Some(33 | 34 | 35 | 36 | 38 | 52) => true,
                // Unquantized heads. BF16 (16) resolves to DType::BF16; the
                // losslessly recoded pair (Bf16Lut3=49, Bf16Huff=50) keeps the
                // head PACKED as DType::Bf16L3. `dflash_enqueue_verify_lm_head`
                // has arms for both. This is what makes a bf16 dense target
                // runnable, which is the control that separates "dense" from
                // "Opus" in the dense-Opus miscompute.
                Some(16 | 49 | 50) => true,
                _ => false,
            };
            if !supported {
                let qt_desc = match lm_qt {
                    Some(qt) => format!("quant_type={qt}"),
                    None => "no lm_head/embed_tokens tensor found at any known name".to_string(),
                };
                return Err(format!(
                    "DFlash draft requested but target lm_head {} is not \
                     supported by speculative.rs's batched GEMM paths on this arch \
                     ({}). Supported: Q8_0 (qt=3), HFQ4G256 (qt=6), MQ4G256 (qt=13), \
                     BF16 (qt=16), Bf16Lut3/Bf16Huff (qt=49/50), Opus oq* \
                     (qt=33/34/35/36/38/52) always; MQ3G256 (qt=17) on gfx11 \
                     only. Other dtypes \
                     (MQ2 qt=18, MQ6/MQ8, HFQ3/HFQ2, HFQ4G128, HFQ6, F16, …) fall \
                     through to a per-row GEMV that hangs verify. Reload without a \
                     draft, or use an MQ4 / HFQ4 / Q8 target. (PRD Phase 2: extend \
                     speculative.rs match arms + add gemm_*_batched_lmhead kernels \
                     for the remaining dtypes.)",
                    qt_desc, gpu.arch
                ));
            }

            // Defense-in-depth: refuse if any body weight is MQ2 (qt=18). MQ3
            // is now allowed on gfx11 dense (arch_id=5) because the WMMA prefill
            // family (qkvza/qkv/gate_up/residual hfq3) and
            // `gemm_hfq3g256_batched_lmhead` are wired. MQ3 is REFUSED on:
            //   - non-gfx11 archs (no batched WMMA prefill kernels)
            //   - MoE/A3B targets (arch_id=6) — the MoE LA/FA prefill branches
            //     and `moe_ffn_all_mq4` predicate are MQ4-only; MQ3 weights
            //     would silently fall through to HFQ4 kernels with the wrong
            //     104-vs-136 byte stride. (Future: wire MQ3 into the MoE
            //     batched branches and the MoE FFN expert kernels.)
            // MQ2 body still has no batched WMMA kernels anywhere.
            let arch_is_dense_qwen35 = is_qwen35_dense_arch_id(hfq.arch_id);
            let mq3_supported = arch_is_gfx11 && arch_is_dense_qwen35;
            let mq_unsupported = hfq
                .first_tensor_with_quant_type(18)
                .map(|n| ("MQ2 (qt=18)", n));
            let mq_unsupported = mq_unsupported.or_else(|| {
                if !mq3_supported {
                    hfq.first_tensor_with_quant_type(17)
                        .map(|n| ("MQ3 (qt=17)", n))
                } else {
                    None
                }
            });
            if let Some((qt_label, name)) = mq_unsupported {
                let arch_reason = if !arch_is_dense_qwen35 && qt_label.starts_with("MQ3") {
                    format!(
                        "arch_id={} (MoE/A3B-class) has no MQ3 MoE kernels",
                        hfq.arch_id
                    )
                } else {
                    format!(
                        "arch={} lacks the corresponding batched WMMA prefill family",
                        gpu.arch
                    )
                };
                return Err(format!(
                    "DFlash draft requested but model contains {qt_label} weight \
                     `{name}` and {arch_reason}. The prefill fast-path falls back \
                     to per-token `forward_scratch` for every spec verify cycle \
                     (or worse, a kernel-stride mismatch on MoE) — defeating \
                     DFlash's speedup. Reload without a draft, or use an MQ4 / \
                     HFQ4 / Q8 target. (Future: port MQ3/MQ2 to MoE branches and \
                     additional archs.)"
                ));
            }
        }
    }

    // Derive physical_cap. With eviction selected, the physical
    // buffer only needs to hold budget+beta+safety slots; max_seq is the
    // advertised window the client targets. Without eviction, the server may
    // still request a smaller initial allocation and reload a larger worker
    // on demand; max_seq remains the logical context-window limit.
    //
    // The `HIPFIRE_KV_PHYSICAL_CAP` env var is an explicit operator override —
    // useful for ablations or reproducing dflash_spec_demo settings.
    let physical_cap = if cask_requested {
        let env_override = std::env::var("HIPFIRE_KV_PHYSICAL_CAP")
            .ok()
            .and_then(|s| s.parse::<usize>().ok());
        let safety = 256usize;
        let floor = cask.budget + cask.beta + 4;
        let derived = cask.budget + cask.beta + safety;
        env_override.unwrap_or(derived).clamp(floor, max_seq)
    } else {
        let requested = requested_physical_cap
            .or_else(|| {
                std::env::var("HIPFIRE_KV_PHYSICAL_CAP")
                    .ok()
                    .and_then(|s| s.parse::<usize>().ok())
            })
            .unwrap_or(max_seq);
        requested.clamp(512.min(max_seq), max_seq)
    };

    let embedding_metadata =
        hipfire_model::embedding::EmbeddingMetadata::from_hfq_metadata_json(&hfq.metadata_json)?;
    let embedding_runtime = classify_embedding_workload(hfq.arch_id, embedding_metadata.as_ref())?;

    if embedding_runtime == Some(EmbeddingRuntimeKind::Qwen3) {
        if dflash_requested {
            return Err(
                "DFlash is not supported by Qwen3 embedding workloads; reload without a draft"
                    .into(),
            );
        }
        if cask_requested {
            return Err(
                "CASK eviction is not supported by Qwen3 embedding workloads; reload without --cask-sidecar"
                    .into(),
            );
        }
        let metadata = embedding_metadata.expect("classified Qwen3 embedding has metadata");
        let config = hipfire_runtime::hfq::config_from_hfq(&hfq)
            .ok_or_else(|| "Qwen3 embedding: failed to parse Qwen3 config".to_string())?;
        if config.arch != llama::ModelArch::Qwen3 || !config.has_qk_norm {
            return Err(
                "Qwen3 embedding requires model_type=qwen3 and per-head Q/K norm tensors".into(),
            );
        }
        let state = Qwen3EmbeddingState::load(&hfq, config, metadata)?;
        hfq.drop_mmap();
        tracing::info!(
            "qwen3 embedding: hidden={}, layers={}, heads={}, kv_heads={}, head_dim={}, intermediate={}, backend=xdna-only",
            state.config.dim,
            state.config.n_layers,
            state.config.n_heads,
            state.config.n_kv_heads,
            state.config.head_dim,
            state.config.hidden_dim,
        );
        let chat_template = resolve_chat_template(&hfq, path);
        let (chat_template, chat_template_profile) =
            profile_chat_template(chat_template, Some(&tokenizer));
        return Ok(LoadedModel {
            arch_id: hfq.arch_id,
            registered_backend: None,
            pp: 1,
            pp_gpus: None,
            pp_scratch_set: None,
            pp_dn_la_to_device: None,
            q35_config: None,
            q35_weights: None,
            q35_scratch: None,
            q35_kv_mode: None,
            q35_state_quant: None,
            q35_registry: SessionRegistry::default(),
            llama_config: None,
            llama_weights: None,
            llama_scratch: None,
            llama_kv: None,
            llama_backend: None,
            nemotron_backend: None,
            zaya_backend: None,
            qwen2_config: None,
            qwen2_weights: None,
            qwen2_state: None,
            deepseek4_config: None,
            deepseek4_weights: None,
            deepseek4_state: None,
            deepseek4_pbs: None,
            deepseek4_eos_tok: 0,
            mtp_mode: "auto".to_string(),
            mtp_k: 3,
            mtp_weights_present: false,
            minimax_config: None,
            minimax_weights: None,
            minimax_state: None,
            minimax_eos_tok: 0,
            #[cfg(feature = "arch-lfm2moe")]
            lfm2moe_config: None,
            #[cfg(feature = "arch-lfm2moe")]
            lfm2moe_weights: None,
            #[cfg(feature = "arch-lfm2moe")]
            lfm2_registry: SessionRegistry::default(),
            #[cfg(feature = "arch-lfm2moe")]
            lfm2moe_eos_tok: 0,
            dots_ocr_config: None,
            dots_ocr_weights: None,
            vision_config: None,
            vision_weights: None,
            gemma3_vl: None,
            gemma3_text: None,
            embeddinggemma: None,
            qwen3_embedding: Some(state),
            tokenizer: Some(tokenizer),
            active: ResidentSession::default(),
            max_seq: 2048,
            physical_cap: 2048,
            eviction: None,
            asst_turn_cache: std::collections::HashMap::new(),
            decoded_vocab: None,
            model_path: path.to_string(),
            memory: model_memory,
            #[cfg(feature = "arch-lfm2moe")]
            lfm2_dflash: None,
            dflash: None,
            // Opt-in; the daemon fills this in from `ngram_spec` after load.
            ngram: None,
            dspark: None,
            chat_template,
            chat_template_profile,
        });
    }

    if hfq.arch_id == ARCH_ID_EMBEDDINGGEMMA {
        // embeddinggemma is a non-autoregressive encoder: no KV cache, no
        // decode loop, and no speculative drafter/eviction state.
        if dflash_requested {
            return Err(
                "DFlash not supported on arch_id=19 (embeddinggemma). Reload without a draft."
                    .to_string(),
            );
        }
        if cask_requested {
            return Err(
                "CASK eviction not supported on arch_id=19 (embeddinggemma). \
                 Reload without --cask-sidecar."
                    .to_string(),
            );
        }
        let _ = (kv_mode.as_str(), state_quant_override);
        let cfg = hipfire_arch_embeddinggemma::config_from_metadata_json(&hfq.metadata_json)
            .ok_or("embeddinggemma: failed to parse config from HFQ metadata")?;
        let embedding_metadata = embedding_metadata;
        tracing::info!(
            "embeddinggemma: hidden={}, layers={}, heads={}, kv_heads={}, vocab={}, embedding_dim={}, matryoshka={:?}",
            cfg.hidden_size,
            cfg.num_hidden_layers,
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            cfg.vocab_size,
            cfg.embedding_dim,
            cfg.matryoshka_dims,
        );
        let storage = embeddinggemma_storage_contract(hfq.tensors())?;
        let metadata_requires_npu = embedding_metadata
            .as_ref()
            .and_then(|metadata| metadata.npu.as_ref())
            .is_some_and(|npu| npu.required);
        let requires_npu = metadata_requires_npu || storage.requires_npu();
        if requires_npu {
            tracing::info!(
                "embeddinggemma: artifact contract requires XDNA{}",
                if !metadata_requires_npu {
                    " (legacy implicit qt=35; requantize to explicit qt=43)"
                } else {
                    ""
                }
            );
        }
        #[cfg(target_os = "linux")]
        let npu_projector = load_embeddinggemma_npu_projector(&hfq, &cfg, requires_npu);
        #[cfg(target_os = "linux")]
        let weights = if requires_npu {
            if std::env::var_os("HIPFIRE_EMBED_GPU_FALLBACK_MODEL").is_some() {
                load_embeddinggemma_gpu_fallback_weights(&cfg, gpu)
            } else if npu_projector.is_some() {
                hipfire_arch_embeddinggemma::EmbeddingGemmaWeights::load_resident_npu(
                    &mut hfq, &cfg, gpu,
                )
            } else {
                Err(
                    "row-padded OQ8 artifact requires a complete XDNA resident-layer cache; \
                     set HIPFIRE_EMBED_GPU_FALLBACK_MODEL for an explicit GPU fallback"
                        .to_string(),
                )
            }
        } else {
            hipfire_arch_embeddinggemma::EmbeddingGemmaWeights::load(&mut hfq, &cfg, gpu)
        }
        .map_err(|e| format!("embeddinggemma weights: {e}"))?;
        #[cfg(not(target_os = "linux"))]
        let weights = {
            if requires_npu {
                return Err(
                    "NPU-only embedding artifact requires the Linux XDNA backend".to_string(),
                );
            }
            hipfire_arch_embeddinggemma::EmbeddingGemmaWeights::load(&mut hfq, &cfg, gpu)
                .map_err(|e| format!("embeddinggemma weights: {e}"))?
        };
        let chat_template = resolve_chat_template(&hfq, path);
        let (chat_template, chat_template_profile) =
            profile_chat_template(chat_template, Some(&tokenizer));
        return Ok(LoadedModel {
            arch_id: hfq.arch_id,
            registered_backend: None,
            pp: 1,
            pp_gpus: None,
            pp_scratch_set: None,
            pp_dn_la_to_device: None,
            q35_config: None,
            q35_weights: None,
            q35_scratch: None,
            q35_kv_mode: None,
            q35_state_quant: None,
            q35_registry: SessionRegistry::default(),
            llama_config: None,
            llama_weights: None,
            llama_scratch: None,
            llama_kv: None,
            llama_backend: None,
            nemotron_backend: None,
            zaya_backend: None,
            qwen2_config: None,
            qwen2_weights: None,
            qwen2_state: None,
            deepseek4_config: None,
            deepseek4_weights: None,
            deepseek4_state: None,
            deepseek4_pbs: None,
            deepseek4_eos_tok: 0,
            mtp_mode: "auto".to_string(),
            mtp_k: 3,
            mtp_weights_present: false,
            minimax_config: None,
            minimax_weights: None,
            minimax_state: None,
            minimax_eos_tok: 0,
            #[cfg(feature = "arch-lfm2moe")]
            lfm2moe_config: None,
            #[cfg(feature = "arch-lfm2moe")]
            lfm2moe_weights: None,
            #[cfg(feature = "arch-lfm2moe")]
            lfm2_registry: SessionRegistry::default(),
            #[cfg(feature = "arch-lfm2moe")]
            lfm2moe_eos_tok: 0,
            dots_ocr_config: None,
            dots_ocr_weights: None,
            vision_config: None,
            vision_weights: None,
            gemma3_vl: None,
            gemma3_text: None,
            embeddinggemma: Some(EmbeddingGemmaState {
                config: cfg,
                embedding_metadata,
                weights,
                #[cfg(target_os = "linux")]
                npu_projector,
            }),
            qwen3_embedding: None,
            tokenizer: Some(tokenizer),
            active: ResidentSession::default(),
            max_seq,
            physical_cap: max_seq,
            eviction: None,
            asst_turn_cache: std::collections::HashMap::new(),
            decoded_vocab: None,
            model_path: path.to_string(),
            memory: model_memory,
            #[cfg(feature = "arch-lfm2moe")]
            lfm2_dflash: None,
            dflash: None,
            // Opt-in; the daemon fills this in from `ngram_spec` after load.
            ngram: None,
            dspark: None,
            chat_template,
            chat_template_profile,
        });
    }

    // Force-link serving-heavy architecture registrations through the aggregate,
    // then resolve by data rather than growing the central arch-id ladder.
    let _ = hipfire_archs::registry();
    if let Some(factory) = hipfire_runtime::arch::serving_factory(hfq.arch_id)? {
        if dflash_requested {
            return Err(format!(
                "DFlash is not supported by the registered {} backend; reload without a draft",
                factory.family()
            ));
        }
        let registered_triattn = if cask_requested {
            Some(
                resolved_triattn_centers
                    .as_ref()
                    .and_then(LoadedTriAttn::layered)
                    .ok_or_else(|| {
                        format!(
                            "registered {} backend requires a heterogeneous TriAttention HFQM package; TRIA v1 is not architecture-safe",
                            factory.family()
                        )
                    })?,
            )
        } else {
            None
        };
        let _ = state_quant_override;
        let registered_backend = factory.load(
            &mut hfq,
            gpu,
            &hipfire_runtime::arch::ServingFactoryOptions {
                max_seq,
                kv_mode: &kv_mode,
                triattn: registered_triattn,
                cask_budget: cask.budget,
                cask_beta: cask.beta,
                physical_cap: cask_requested.then_some(physical_cap),
            },
        )?;
        let physical_cap = registered_backend.physical_cap;
        let chat_template = resolve_chat_template(&hfq, path);
        let (chat_template, chat_template_profile) =
            profile_chat_template(chat_template, Some(&tokenizer));
        return Ok(LoadedModel {
            arch_id: hfq.arch_id,
            registered_backend: Some(registered_backend),
            pp: 1,
            pp_gpus: None,
            pp_scratch_set: None,
            pp_dn_la_to_device: None,
            q35_config: None,
            q35_weights: None,
            q35_scratch: None,
            q35_kv_mode: None,
            q35_state_quant: None,
            q35_registry: SessionRegistry::default(),
            llama_config: None,
            llama_weights: None,
            llama_scratch: None,
            llama_kv: None,
            llama_backend: None,
            nemotron_backend: None,
            zaya_backend: None,
            qwen2_config: None,
            qwen2_weights: None,
            qwen2_state: None,
            deepseek4_config: None,
            deepseek4_weights: None,
            deepseek4_state: None,
            deepseek4_pbs: None,
            deepseek4_eos_tok: 0,
            mtp_mode: "auto".to_string(),
            mtp_k: 3,
            mtp_weights_present: false,
            minimax_config: None,
            minimax_weights: None,
            minimax_state: None,
            minimax_eos_tok: 0,
            #[cfg(feature = "arch-lfm2moe")]
            lfm2moe_config: None,
            #[cfg(feature = "arch-lfm2moe")]
            lfm2moe_weights: None,
            #[cfg(feature = "arch-lfm2moe")]
            lfm2_registry: SessionRegistry::default(),
            #[cfg(feature = "arch-lfm2moe")]
            lfm2moe_eos_tok: 0,
            dots_ocr_config: None,
            dots_ocr_weights: None,
            vision_config: None,
            vision_weights: None,
            gemma3_vl: None,
            gemma3_text: None,
            embeddinggemma: None,
            qwen3_embedding: None,
            tokenizer: Some(tokenizer),
            active: ResidentSession::default(),
            max_seq: physical_cap,
            physical_cap,
            eviction: None,
            asst_turn_cache: std::collections::HashMap::new(),
            decoded_vocab: None,
            model_path: path.to_string(),
            memory: model_memory,
            #[cfg(feature = "arch-lfm2moe")]
            lfm2_dflash: None,
            dflash: None,
            // Opt-in; the daemon fills this in from `ngram_spec` after load.
            ngram: None,
            dspark: None,
            chat_template,
            chat_template_profile,
        });
    }

    if hfq.arch_id == ARCH_ID_ZAYA {
        // ZAYA1 (CCA attention + EDA/MoD-routed MoE). Served through the shared
        // ServingBackend seam on ZayaModel (O(1) KV-cache decode).
        if dflash_requested {
            return Err("DFlash not supported on arch_id=16 (zaya).".to_string());
        }
        let _ = kv_mode;
        let _ = state_quant_override;
        let meta: serde_json::Value = serde_json::from_str(&hfq.metadata_json)
            .map_err(|e| format!("zaya metadata parse: {e}"))?;
        let cfg_json = meta
            .get("config")
            .ok_or("zaya: metadata_json missing 'config'")?;
        let cfg = hipfire_arch_zaya::ZayaConfig::from_json(cfg_json)
            .map_err(|e| format!("zaya config: {e}"))?;
        tracing::info!(
            "zaya: hidden={}, blocks={}, experts={}, vocab={}, eos={}",
            cfg.hidden_size,
            cfg.num_blocks,
            cfg.moe.num_experts,
            cfg.vocab_size,
            cfg.eos_token_id,
        );
        let model = hipfire_arch_zaya::arch::ZayaModel::from_hfq(gpu, &hfq, cfg, max_seq)
            .map_err(|e| format!("ZayaModel::from_hfq: {e}"))?;
        let chat_template = resolve_chat_template(&hfq, path);
        let (chat_template, chat_template_profile) =
            profile_chat_template(chat_template, Some(&tokenizer));
        return Ok(LoadedModel {
            arch_id: hfq.arch_id,
            registered_backend: None,
            pp: 1,
            pp_gpus: None,
            pp_scratch_set: None,
            pp_dn_la_to_device: None,
            q35_config: None,
            q35_weights: None,
            q35_scratch: None,
            q35_kv_mode: None,
            q35_state_quant: None,
            q35_registry: SessionRegistry::default(),
            llama_config: None,
            llama_weights: None,
            llama_scratch: None,
            llama_kv: None,
            llama_backend: None,
            nemotron_backend: None,
            zaya_backend: Some(model),
            qwen2_config: None,
            qwen2_weights: None,
            qwen2_state: None,
            deepseek4_config: None,
            deepseek4_weights: None,
            deepseek4_state: None,
            deepseek4_pbs: None,
            deepseek4_eos_tok: 0,
            mtp_mode: "auto".to_string(),
            mtp_k: 3,
            mtp_weights_present: false,
            minimax_config: None,
            minimax_weights: None,
            minimax_state: None,
            minimax_eos_tok: 0,
            #[cfg(feature = "arch-lfm2moe")]
            lfm2moe_config: None,
            #[cfg(feature = "arch-lfm2moe")]
            lfm2moe_weights: None,
            #[cfg(feature = "arch-lfm2moe")]
            lfm2_registry: SessionRegistry::default(),
            #[cfg(feature = "arch-lfm2moe")]
            lfm2moe_eos_tok: 0,
            dots_ocr_config: None,
            dots_ocr_weights: None,
            vision_config: None,
            vision_weights: None,
            gemma3_vl: None,
            gemma3_text: None,
            embeddinggemma: None,
            qwen3_embedding: None,
            tokenizer: Some(tokenizer),
            active: ResidentSession::default(),
            max_seq,
            physical_cap: max_seq,
            eviction: None,
            asst_turn_cache: std::collections::HashMap::new(),
            decoded_vocab: None,
            model_path: path.to_string(),
            memory: model_memory,
            #[cfg(feature = "arch-lfm2moe")]
            lfm2_dflash: None,
            dflash: None,
            // Opt-in; the daemon fills this in from `ngram_spec` after load.
            ngram: None,
            dspark: None,
            chat_template,
            chat_template_profile,
        });
    }

    if hfq.arch_id == ARCH_ID_NEMOTRON_H || hfq.arch_id == ARCH_ID_MAMBA2 {
        // nemotron_h (hybrid Mamba-2 + attention/MLP/MoE) and pure Mamba-2
        // from quantized (or bf16) .hfq artifacts, driven through the same
        // Mamba-capable ServingBackend seam.
        if dflash_requested {
            return Err(format!(
                "DFlash not supported on arch_id={} ({}). Reload without a draft.",
                hfq.arch_id,
                if hfq.arch_id == ARCH_ID_MAMBA2 {
                    "mamba2"
                } else {
                    "nemotron_h"
                }
            ));
        }
        let _ = kv_mode;
        let _ = state_quant_override;
        let meta: serde_json::Value = serde_json::from_str(&hfq.metadata_json)
            .map_err(|e| format!("nemotron metadata parse: {e}"))?;
        let cfg_json = meta
            .get("config")
            .ok_or("nemotron: metadata_json missing 'config'")?;
        let mut cfg = if hfq.arch_id == ARCH_ID_MAMBA2 {
            hipfire_arch_nemotron::NemotronHConfig::from_mamba2_json(cfg_json)
                .map_err(|e| format!("mamba2 config: {e}"))?
        } else {
            hipfire_arch_nemotron::NemotronHConfig::from_json(cfg_json)
                .map_err(|e| format!("nemotron config: {e}"))?
        };
        if hfq.arch_id == ARCH_ID_MAMBA2 {
            if let Some(eot) = tokenizer.special_token_id("<|endoftext|>") {
                cfg.eos_token_id = eot;
            }
        } else if let Some(im_end) = tokenizer.special_token_id("<|im_end|>") {
            // Chat serving stops on the ChatML turn delimiter `<|im_end|>`, not
            // the base `eos_token_id` (`</s>` = 2 for Nano).
            cfg.eos_token_id = im_end;
        }
        tracing::info!(
            "{}: hidden={}, layers={} ({} M / {} * / {} - / {} E), vocab={}, eos={}",
            if hfq.arch_id == ARCH_ID_MAMBA2 {
                "mamba2"
            } else {
                "nemotron_h"
            },
            cfg.hidden_size,
            cfg.num_layers,
            cfg.count(hipfire_arch_nemotron::BlockKind::Mamba2),
            cfg.count(hipfire_arch_nemotron::BlockKind::Attention),
            cfg.count(hipfire_arch_nemotron::BlockKind::Mlp),
            cfg.count(hipfire_arch_nemotron::BlockKind::Moe),
            cfg.vocab_size,
            cfg.eos_token_id,
        );
        let model = hipfire_arch_nemotron::model::NemotronModel::from_hfq(gpu, &hfq, cfg, max_seq)
            .map_err(|e| format!("mamba-capable NemotronModel::from_hfq: {e}"))?;
        let chat_template = resolve_chat_template(&hfq, path);
        let (chat_template, chat_template_profile) =
            profile_chat_template(chat_template, Some(&tokenizer));
        return Ok(LoadedModel {
            arch_id: hfq.arch_id,
            registered_backend: None,
            pp: 1,
            pp_gpus: None,
            pp_scratch_set: None,
            pp_dn_la_to_device: None,
            q35_config: None,
            q35_weights: None,
            q35_scratch: None,
            q35_kv_mode: None,
            q35_state_quant: None,
            q35_registry: SessionRegistry::default(),
            llama_config: None,
            llama_weights: None,
            llama_scratch: None,
            llama_kv: None,
            llama_backend: None,
            nemotron_backend: Some(model),
            zaya_backend: None,
            qwen2_config: None,
            qwen2_weights: None,
            qwen2_state: None,
            deepseek4_config: None,
            deepseek4_weights: None,
            deepseek4_state: None,
            deepseek4_pbs: None,
            deepseek4_eos_tok: 0,
            mtp_mode: "auto".to_string(),
            mtp_k: 3,
            mtp_weights_present: false,
            minimax_config: None,
            minimax_weights: None,
            minimax_state: None,
            minimax_eos_tok: 0,
            #[cfg(feature = "arch-lfm2moe")]
            lfm2moe_config: None,
            #[cfg(feature = "arch-lfm2moe")]
            lfm2moe_weights: None,
            #[cfg(feature = "arch-lfm2moe")]
            lfm2_registry: SessionRegistry::default(),
            #[cfg(feature = "arch-lfm2moe")]
            lfm2moe_eos_tok: 0,
            dots_ocr_config: None,
            dots_ocr_weights: None,
            vision_config: None,
            vision_weights: None,
            gemma3_vl: None,
            gemma3_text: None,
            embeddinggemma: None,
            qwen3_embedding: None,
            tokenizer: Some(tokenizer),
            active: ResidentSession::default(),
            max_seq,
            physical_cap: max_seq,
            eviction: None,
            asst_turn_cache: std::collections::HashMap::new(),
            decoded_vocab: None,
            model_path: path.to_string(),
            memory: model_memory,
            #[cfg(feature = "arch-lfm2moe")]
            lfm2_dflash: None,
            dflash: None,
            // Opt-in; the daemon fills this in from `ngram_spec` after load.
            ngram: None,
            dspark: None,
            chat_template,
            chat_template_profile,
        });
    }

    if hfq.arch_id == ARCH_ID_GEMMA3_TEXT {
        // Gemma3 text (medgemma-*-text). Plain dense-AR: the gemma3 decoder +
        // its own decode state in `Gemma3Backend`, served via the same
        // `ServingBackend::serve` seam (delegates to `run_simple_ar`). No
        // vision, eviction, DFlash, CASK, or PP.
        if dflash_requested {
            return Err(
                "DFlash not supported on arch_id=12 (gemma3 text). Reload without a draft."
                    .to_string(),
            );
        }
        if cask_requested {
            return Err("CASK eviction not supported on arch_id=12 (gemma3 text). \
                       Reload without --cask-sidecar."
                .to_string());
        }
        // gemma3 KV: F32 by default; honor an explicit q8/int8/kvarn kv_mode
        // (all ~4x smaller than F32, letting larger contexts fit). Other quant
        // modes (asym/fwht) have no gemma3 kernel yet and fall back to F32.
        let (kv_mode_g3, kvarn_bits_g3) = gemma3_kv_mode(&kv_mode);
        let _ = state_quant_override;
        let cfg = hipfire_arch_gemma3::config_from_hfq(&hfq)
            .ok_or_else(|| "gemma3: failed to parse Gemma3Config".to_string())?;
        let weights = hipfire_arch_gemma3::load_weights(&mut hfq, &cfg, gpu)
            .map_err(|e| format!("gemma3: load_weights failed: {e:?}"))?;
        let state = Gemma3State::new_with_max_seq(gpu, &cfg, max_seq, kv_mode_g3, kvarn_bits_g3)
            .map_err(|e| format!("gemma3: Gemma3State::new_with_max_seq failed: {e:?}"))?;
        let backend = Gemma3Backend::new(cfg, weights, state);
        let chat_template = resolve_chat_template(&hfq, path);
        let (chat_template, chat_template_profile) =
            profile_chat_template(chat_template, Some(&tokenizer));
        return Ok(LoadedModel {
            arch_id: hfq.arch_id,
            registered_backend: None,
            pp: 1,
            pp_gpus: None,
            pp_scratch_set: None,
            pp_dn_la_to_device: None,
            q35_config: None,
            q35_weights: None,
            q35_scratch: None,
            q35_kv_mode: None,
            q35_state_quant: None,
            q35_registry: SessionRegistry::default(),
            llama_config: None,
            llama_weights: None,
            llama_scratch: None,
            llama_kv: None,
            llama_backend: None,
            nemotron_backend: None,
            zaya_backend: None,
            qwen2_config: None,
            qwen2_weights: None,
            qwen2_state: None,
            deepseek4_config: None,
            deepseek4_weights: None,
            deepseek4_state: None,
            deepseek4_pbs: None,
            deepseek4_eos_tok: 0,
            mtp_mode: "auto".to_string(),
            mtp_k: 3,
            mtp_weights_present: false,
            minimax_config: None,
            minimax_weights: None,
            minimax_state: None,
            minimax_eos_tok: 0,
            #[cfg(feature = "arch-lfm2moe")]
            lfm2moe_config: None,
            #[cfg(feature = "arch-lfm2moe")]
            lfm2moe_weights: None,
            #[cfg(feature = "arch-lfm2moe")]
            lfm2_registry: SessionRegistry::default(),
            #[cfg(feature = "arch-lfm2moe")]
            lfm2moe_eos_tok: 0,
            dots_ocr_config: None,
            dots_ocr_weights: None,
            vision_config: None,
            vision_weights: None,
            gemma3_vl: None,
            gemma3_text: Some(backend),
            embeddinggemma: None,
            qwen3_embedding: None,
            tokenizer: Some(tokenizer),
            active: ResidentSession::default(),
            max_seq,
            physical_cap: max_seq,
            eviction: None,
            asst_turn_cache: std::collections::HashMap::new(),
            decoded_vocab: None,
            model_path: path.to_string(),
            memory: model_memory,
            #[cfg(feature = "arch-lfm2moe")]
            lfm2_dflash: None,
            dflash: None,
            // Opt-in; the daemon fills this in from `ngram_spec` after load.
            ngram: None,
            dspark: None,
            chat_template,
            chat_template_profile,
        });
    }

    if hfq.arch_id == ARCH_ID_GEMMA3_VL {
        // Gemma3-VL (medgemma). Self-contained multimodal backend: the gemma3
        // text decoder (loaded from the `language_model.` prefix) + the SigLIP
        // vision tower + the projector, plus its own decode state — all owned by
        // `Gemma3VlBackend`, which serves via `ServingBackend::serve` →
        // `decode_loop` (greedy). No eviction / DFlash / CASK / PP, and not the
        // qwen35-VL `vision_config` splice path (that field stays None for 13;
        // the `has_vl` gate keys off `gemma3_vl.is_some()`).
        if dflash_requested {
            return Err(
                "DFlash not supported on arch_id=13 (gemma3-vl). Reload without a draft."
                    .to_string(),
            );
        }
        if cask_requested {
            return Err("CASK eviction not supported on arch_id=13 (gemma3-vl). \
                       Reload without --cask-sidecar."
                .to_string());
        }
        // gemma3-vl KV: F32 by default; honor an explicit q8/int8/kvarn kv_mode.
        // Lets medgemma run a much larger context before exhausting the GTT pool.
        let (kv_mode_g3, kvarn_bits_g3) = gemma3_kv_mode(&kv_mode);
        let _ = state_quant_override;
        let LoadedVl {
            text_cfg,
            vl_cfg,
            weights,
            vision_tier,
            vision_source_id,
        } = load_vl(&mut hfq, gpu)?;
        let state =
            Gemma3State::new_with_max_seq(gpu, &text_cfg, max_seq, kv_mode_g3, kvarn_bits_g3)
                .map_err(|e| format!("gemma3-vl: Gemma3State::new_with_max_seq failed: {e:?}"))?;
        let backend = Gemma3VlBackend::new(
            text_cfg,
            vl_cfg,
            weights,
            state,
            vision_tier,
            vision_source_id,
        );
        let chat_template = resolve_chat_template(&hfq, path);
        let (chat_template, chat_template_profile) =
            profile_chat_template(chat_template, Some(&tokenizer));
        return Ok(LoadedModel {
            arch_id: hfq.arch_id,
            registered_backend: None,
            pp: 1,
            pp_gpus: None,
            pp_scratch_set: None,
            pp_dn_la_to_device: None,
            q35_config: None,
            q35_weights: None,
            q35_scratch: None,
            q35_kv_mode: None,
            q35_state_quant: None,
            q35_registry: SessionRegistry::default(),
            llama_config: None,
            llama_weights: None,
            llama_scratch: None,
            llama_kv: None,
            llama_backend: None,
            nemotron_backend: None,
            zaya_backend: None,
            qwen2_config: None,
            qwen2_weights: None,
            qwen2_state: None,
            deepseek4_config: None,
            deepseek4_weights: None,
            deepseek4_state: None,
            deepseek4_pbs: None,
            deepseek4_eos_tok: 0,
            mtp_mode: "auto".to_string(),
            mtp_k: 3,
            mtp_weights_present: false,
            minimax_config: None,
            minimax_weights: None,
            minimax_state: None,
            minimax_eos_tok: 0,
            #[cfg(feature = "arch-lfm2moe")]
            lfm2moe_config: None,
            #[cfg(feature = "arch-lfm2moe")]
            lfm2moe_weights: None,
            #[cfg(feature = "arch-lfm2moe")]
            lfm2_registry: SessionRegistry::default(),
            #[cfg(feature = "arch-lfm2moe")]
            lfm2moe_eos_tok: 0,
            dots_ocr_config: None,
            dots_ocr_weights: None,
            vision_config: None,
            vision_weights: None,
            gemma3_vl: Some(backend),
            gemma3_text: None,
            embeddinggemma: None,
            qwen3_embedding: None,
            tokenizer: Some(tokenizer),
            active: ResidentSession::default(),
            max_seq,
            physical_cap: max_seq,
            eviction: None,
            asst_turn_cache: std::collections::HashMap::new(),
            decoded_vocab: None,
            model_path: path.to_string(),
            memory: model_memory,
            #[cfg(feature = "arch-lfm2moe")]
            lfm2_dflash: None,
            dflash: None,
            // Opt-in; the daemon fills this in from `ngram_spec` after load.
            ngram: None,
            dspark: None,
            chat_template,
            chat_template_profile,
        });
    }

    if hfq.arch_id == ARCH_ID_DOTS_OCR {
        // dots.ocr (Qwen2-VL family). Text decoder is Qwen2; vision tower
        // is the 42-block DotsVisionTransformer. Both load side-by-side in
        // DotsOcrWeights and stay resident. Single-image, greedy decode at
        // bring-up — no eviction, DFlash, CASK, or PP.
        if dflash_requested {
            return Err(
                "DFlash not supported on arch_id=8 (dots.ocr). Reload without a draft.".to_string(),
            );
        }
        if cask_requested {
            return Err("CASK eviction not supported on arch_id=8 (dots.ocr). Reload without --cask-sidecar.".to_string());
        }
        if pp > 1 {
            return Err(
                "pipeline-parallel (pp>1) not supported on arch_id=8 (dots.ocr).".to_string(),
            );
        }
        let _ = kv_mode;
        let _ = state_quant_override;
        use hipfire_arch_dots_ocr::DotsOcr;
        use hipfire_runtime::arch::Architecture;
        let config = <DotsOcr as Architecture>::config_from_hfq(&hfq)?;
        let weights = <DotsOcr as Architecture>::load_weights(&mut hfq, &config, gpu)?;
        // Size the decode KV cache to the requested window (the trait's
        // new_state uses a default max_seq; OCR prompts are long).
        let state = qwen2::Qwen2State::new_with_max_seq(gpu, &config.text, max_seq)
            .map_err(|e| format!("dots-ocr: Qwen2State::new_with_max_seq failed: {e:?}"))?;
        let chat_template = resolve_chat_template(&hfq, path);
        let (chat_template, chat_template_profile) =
            profile_chat_template(chat_template, Some(&tokenizer));
        return Ok(LoadedModel {
            arch_id: hfq.arch_id,
            registered_backend: None,
            pp: 1,
            pp_gpus: None,
            pp_scratch_set: None,
            pp_dn_la_to_device: None,
            q35_config: None,
            q35_weights: None,
            q35_scratch: None,
            q35_kv_mode: None,
            q35_state_quant: None,
            q35_registry: SessionRegistry::default(),
            llama_config: None,
            llama_weights: None,
            llama_scratch: None,
            llama_kv: None,
            llama_backend: None,
            nemotron_backend: None,
            zaya_backend: None,
            qwen2_config: None,
            qwen2_weights: None,
            qwen2_state: Some(state),
            deepseek4_config: None,
            deepseek4_weights: None,
            deepseek4_state: None,
            deepseek4_pbs: None,
            deepseek4_eos_tok: 0,
            mtp_mode: "auto".to_string(),
            mtp_k: 3,
            mtp_weights_present: false,
            minimax_config: None,
            minimax_weights: None,
            minimax_state: None,
            minimax_eos_tok: 0,
            #[cfg(feature = "arch-lfm2moe")]
            lfm2moe_config: None,
            #[cfg(feature = "arch-lfm2moe")]
            lfm2moe_weights: None,
            #[cfg(feature = "arch-lfm2moe")]
            lfm2_registry: SessionRegistry::default(),
            #[cfg(feature = "arch-lfm2moe")]
            lfm2moe_eos_tok: 0,
            dots_ocr_config: Some(config),
            dots_ocr_weights: Some(weights),
            vision_config: None,
            vision_weights: None,
            gemma3_vl: None,
            gemma3_text: None,
            embeddinggemma: None,
            qwen3_embedding: None,
            tokenizer: Some(tokenizer),
            active: ResidentSession::default(),
            max_seq,
            physical_cap: max_seq,
            eviction: None,
            asst_turn_cache: std::collections::HashMap::new(),
            decoded_vocab: None,
            model_path: path.to_string(),
            memory: model_memory,
            #[cfg(feature = "arch-lfm2moe")]
            lfm2_dflash: None,
            dflash: None,
            // Opt-in; the daemon fills this in from `ngram_spec` after load.
            ngram: None,
            dspark: None,
            chat_template,
            chat_template_profile,
        });
    }

    if hfq.arch_id == ARCH_ID_DEEPSEEK4_FLASH {
        // DeepSeek V4 Flash (hipfire-arch-deepseek4). Standalone bring-up —
        // no eviction, no DFlash drafter, no PFlash, no VL. The
        // Architecture trait gives us config + weights + state in three
        // calls; forward goes through `deepseek4::forward::forward_prefill_*` /
        // `decode_step` in the generate hot path.
        if dflash_requested {
            return Err("DFlash not supported on arch_id=9 (DeepSeek V4 Flash). \
                       Reload without a draft."
                .to_string());
        }
        if cask_requested {
            return Err(
                "CASK eviction not supported on arch_id=9 (DeepSeek V4 Flash). \
                       Reload without --cask-sidecar."
                    .to_string(),
            );
        }
        let _ = kv_mode;
        let _ = state_quant_override;
        use hipfire_runtime::arch::Architecture;
        let config = <deepseek4::DeepseekV4 as Architecture>::config_from_hfq(&hfq)?;
        let weights =
            <deepseek4::DeepseekV4 as Architecture>::load_weights(&mut hfq, &config, gpu)?;
        let state = deepseek4::DeepseekV4State::new(&config)?;
        // Pre-allocate PrefillBatchScratch. Default B=1024 (bumped from 64
        // on 2026-05-26). PP_BATCH sweep on the 2.1k-tok bench (3 trials/cell):
        //   PP=256: 46.4 tps   PP=512: 48.3 tps
        //   PP=1024: 49.3 tps  PP=2048: 49.0 tps
        // 1024 captures the L2-amortization peak; 2048 plateaus from PBS
        // memory footprint exceeding effective L2/Inf-cache reuse window.
        // PBS sits in (UMA) GPU memory for the model's lifetime — ~600 MB
        // at B=1024 on V4-Flash, well within 128 GB. Override via
        // HIPFIRE_DEEPSEEK4_PP_BATCH.
        let pbs_max_batch: usize = std::env::var("HIPFIRE_DEEPSEEK4_PP_BATCH")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1024);
        let pbs = deepseek4::forward::PrefillBatchScratch::new(gpu, &config, pbs_max_batch)?;
        // Cache EOS token id. DeepSeek family uses `<｜end▁of▁sentence｜>`;
        // fall back to 1 if tokenizer lacks the entry.
        let eos_tok: u32 = {
            let ids = tokenizer.encode("<｜end▁of▁sentence｜>");
            if ids.len() == 1 {
                ids[0]
            } else {
                1
            }
        };
        let chat_template = resolve_chat_template(&hfq, path);
        let (chat_template, chat_template_profile) =
            profile_chat_template(chat_template, Some(&tokenizer));
        return Ok(LoadedModel {
            arch_id: hfq.arch_id,
            registered_backend: None,
            pp: 1,
            pp_gpus: None,
            pp_scratch_set: None,
            pp_dn_la_to_device: None,
            q35_config: None,
            q35_weights: None,
            q35_scratch: None,
            q35_kv_mode: None,
            q35_state_quant: None,
            q35_registry: SessionRegistry::default(),
            llama_config: None,
            llama_weights: None,
            llama_scratch: None,
            llama_kv: None,
            llama_backend: None,
            nemotron_backend: None,
            zaya_backend: None,
            qwen2_config: None,
            qwen2_weights: None,
            qwen2_state: None,
            deepseek4_config: Some(config),
            deepseek4_weights: Some(weights),
            deepseek4_state: Some(state),
            deepseek4_pbs: Some(pbs),
            deepseek4_eos_tok: eos_tok,
            mtp_mode: "auto".to_string(),
            mtp_k: 3,
            mtp_weights_present: false,
            minimax_config: None,
            minimax_weights: None,
            minimax_state: None,
            minimax_eos_tok: 0,
            #[cfg(feature = "arch-lfm2moe")]
            lfm2moe_config: None,
            #[cfg(feature = "arch-lfm2moe")]
            lfm2moe_weights: None,
            #[cfg(feature = "arch-lfm2moe")]
            lfm2_registry: SessionRegistry::default(),
            #[cfg(feature = "arch-lfm2moe")]
            lfm2moe_eos_tok: 0,
            dots_ocr_config: None,
            dots_ocr_weights: None,
            vision_config: None,
            vision_weights: None,
            gemma3_vl: None,
            gemma3_text: None,
            embeddinggemma: None,
            qwen3_embedding: None,
            tokenizer: Some(tokenizer),
            active: ResidentSession::default(),
            max_seq,
            physical_cap: max_seq,
            eviction: None,
            asst_turn_cache: std::collections::HashMap::new(),
            decoded_vocab: None,
            model_path: path.to_string(),
            memory: model_memory,
            #[cfg(feature = "arch-lfm2moe")]
            lfm2_dflash: None,
            dflash: None,
            // Opt-in; the daemon fills this in from `ngram_spec` after load.
            ngram: None,
            dspark: None,
            chat_template,
            chat_template_profile,
        });
    }

    if hfq.arch_id == ARCH_ID_MINIMAX_M2 {
        // MiniMax-M2 (hipfire-arch-minimax). Standalone bring-up — no
        // eviction, no DFlash drafter, no PFlash, no VL, no spec-decode.
        // The Architecture trait gives us config + weights + state in three
        // calls; prefill + decode both go through the per-token
        // `minimax::forward::decode_step` in the generate hot path. There
        // is NO PrefillBatchScratch (no batched prefill kernel).
        if dflash_requested {
            return Err("DFlash not supported on arch_id=10 (MiniMax-M2). \
                       Reload without a draft."
                .to_string());
        }
        if cask_requested {
            return Err("CASK eviction not supported on arch_id=10 (MiniMax-M2). \
                       Reload without --cask-sidecar."
                .to_string());
        }
        if pp > 1 {
            return Err(
                "pipeline-parallel (pp>1) not supported on arch_id=10 (MiniMax-M2).".to_string(),
            );
        }
        let _ = kv_mode;
        let _ = state_quant_override;
        use hipfire_runtime::arch::Architecture;
        let config = <minimax::MiniMaxM2 as Architecture>::config_from_hfq(&hfq)?;
        let weights = <minimax::MiniMaxM2 as Architecture>::load_weights(&mut hfq, &config, gpu)?;
        // Size the KV cache to the requested window (the trait's new_state
        // caps at 8192; honour the caller's max_seq when it's larger/smaller).
        let state = minimax::MiniMaxState::new_with_max_seq(gpu, &config, max_seq)
            .map_err(|e| format!("minimax: MiniMaxState::new_with_max_seq failed: {e}"))?;
        // Resolve EOS via the tokenizer. MiniMax-M2 does NOT use ChatML —
        // its end-of-turn marker is the added token `[e~[` (id 200020 in the
        // 200k vocab; tokenizer_config.json eos_token = `[e~[`,
        // generation_config.json eos_token_id = 200020). The earlier ChatML
        // probes (`<|im_end|>` etc.) are absent from this vocab and silently
        // fell back to token 1, so generate_minimax never hit EOS: every turn
        // ran to max_tokens and the model spammed `[e~[` trying to end the
        // turn. Probe the real marker first; keep the ChatML fallbacks for
        // safety on any future tokenizer variant.
        let eos_tok: u32 = {
            let try_one = |s: &str| -> Option<u32> {
                let ids = tokenizer.encode(s);
                if ids.len() == 1 {
                    Some(ids[0])
                } else {
                    None
                }
            };
            try_one("[e~[")
                .or_else(|| try_one("<|im_end|>"))
                .or_else(|| try_one("</s>"))
                .or_else(|| try_one("<|endoftext|>"))
                .unwrap_or(1)
        };
        let chat_template = resolve_chat_template(&hfq, path);
        let (chat_template, chat_template_profile) =
            profile_chat_template(chat_template, Some(&tokenizer));
        return Ok(LoadedModel {
            arch_id: hfq.arch_id,
            registered_backend: None,
            pp: 1,
            pp_gpus: None,
            pp_scratch_set: None,
            pp_dn_la_to_device: None,
            q35_config: None,
            q35_weights: None,
            q35_scratch: None,
            q35_kv_mode: None,
            q35_state_quant: None,
            q35_registry: SessionRegistry::default(),
            llama_config: None,
            llama_weights: None,
            llama_scratch: None,
            llama_kv: None,
            llama_backend: None,
            nemotron_backend: None,
            zaya_backend: None,
            qwen2_config: None,
            qwen2_weights: None,
            qwen2_state: None,
            deepseek4_config: None,
            deepseek4_weights: None,
            deepseek4_state: None,
            deepseek4_pbs: None,
            deepseek4_eos_tok: 0,
            mtp_mode: "auto".to_string(),
            mtp_k: 3,
            mtp_weights_present: false,
            minimax_config: Some(config),
            minimax_weights: Some(weights),
            minimax_state: Some(state),
            minimax_eos_tok: eos_tok,
            #[cfg(feature = "arch-lfm2moe")]
            lfm2moe_config: None,
            #[cfg(feature = "arch-lfm2moe")]
            lfm2moe_weights: None,
            #[cfg(feature = "arch-lfm2moe")]
            lfm2_registry: SessionRegistry::default(),
            #[cfg(feature = "arch-lfm2moe")]
            lfm2moe_eos_tok: 0,
            dots_ocr_config: None,
            dots_ocr_weights: None,
            vision_config: None,
            vision_weights: None,
            gemma3_vl: None,
            gemma3_text: None,
            embeddinggemma: None,
            qwen3_embedding: None,
            tokenizer: Some(tokenizer),
            active: ResidentSession::default(),
            max_seq,
            physical_cap: max_seq,
            eviction: None,
            asst_turn_cache: std::collections::HashMap::new(),
            decoded_vocab: None,
            model_path: path.to_string(),
            memory: model_memory,
            #[cfg(feature = "arch-lfm2moe")]
            lfm2_dflash: None,
            dflash: None,
            // Opt-in; the daemon fills this in from `ngram_spec` after load.
            ngram: None,
            dspark: None,
            chat_template,
            chat_template_profile,
        });
    }

    if hfq.arch_id == ARCH_ID_LFM2_MOE {
        // LFM2.5-8B-A1B (hipfire-arch-lfm2moe). Standalone bring-up — no
        // DFlash drafter, no PFlash, no VL, no spec-decode. CASK/TriAttention
        // is wired through the shared KvCache using attention-ordinal sidecar
        // indices (0..num_attention_layers), because LFM2 stores KV only for
        // attention layers rather than allocating one KV slot per model layer.
        // Hybrid LIV short-conv + GQA attention feeding a top-4 MoE FFN.
        // config + weights + state come from the crate's direct API
        // (it does not implement the Architecture trait); generate prefill
        // routes through `lfm2moe::forward::prefill_batch` and decode through
        // `lfm2moe::forward::decode_step`. Structurally mirrors MiniMax (10)
        // while carrying arch-local prefill scratch in the forward module.
        #[cfg(not(feature = "arch-lfm2moe"))]
        {
            let _ = (
                &mut hfq,
                path,
                max_seq,
                draft_path,
                kv_mode,
                state_quant_override,
                &cask,
                pp,
                gpu,
                tokenizer,
            );
            return Err(
                "lfm2moe arch (id 11) not compiled in (enable feature arch-lfm2moe)".to_string(),
            );
        }
        #[cfg(feature = "arch-lfm2moe")]
        {
            if pp > 1 {
                return Err(
                    "pipeline-parallel (pp>1) not supported on arch_id=11 (LFM2.5-MoE)."
                        .to_string(),
                );
            }
            if dflash_requested && cask_requested {
                return Err(
                    "LFM2 DFlash does not support CASK/TriAttention eviction yet; \
                     reload without the draft or without the CASK sidecar"
                        .to_string(),
                );
            }
            let _ = kv_mode;
            let _ = state_quant_override;
            let config = lfm2moe::config::Lfm2MoeConfig::from_hfq(&hfq)?;
            let weights = lfm2moe::lfm2moe::Lfm2MoeWeights::load(&mut hfq, &config, gpu)?;
            // Size the KV + conv-state cache to the requested window.
            let state = lfm2moe::lfm2moe::Lfm2MoeState::new_with_physical_cap(
                gpu,
                &config,
                max_seq,
                physical_cap,
            )
            .map_err(|e| format!("lfm2moe: Lfm2MoeState::new_with_physical_cap failed: {e}"))?;
            let lfm2_dflash = match resolved_dflash.as_ref() {
                Some(source) if source.path().is_some() => Some(load_lfm2_dflash_state(
                    &source.path().unwrap().to_string_lossy(),
                    physical_cap,
                    &config,
                    &state,
                    gpu,
                )?),
                Some(ResolvedComponentSource::Embedded) => {
                    let manifest = compose_manifest
                        .as_ref()
                        .ok_or("embedded DFLASH selected without a manifest")?;
                    let view = file_component_view(&hfq, manifest, "dflash")
                        .map_err(|error| format!("embedded DFLASH view: {error}"))?
                        .ok_or("embedded DFLASH role disappeared from manifest")?;
                    Some(load_lfm2_dflash_state_source(
                        &view,
                        physical_cap,
                        &config,
                        &state,
                        gpu,
                    )?)
                }
                None => None,
                Some(_) => unreachable!("path-backed DFLASH handled above"),
            };
            let eviction = if let Some(source) = resolved_triattn.as_ref() {
                let centers = resolved_triattn_centers
                    .as_ref()
                    .expect("resolved TriAttention source has parsed centers")
                    .uniform_centers()?;
                tracing::info!("TriAttention component source: {}", source.label());
                let n_attn = config.num_attention_layers();
                if centers.n_layers != n_attn
                    || centers.n_heads != config.num_attention_heads
                    || centers.head_dim != config.head_dim
                {
                    return Err(format!(
                        "lfm2moe cask sidecar shape mismatch: sidecar layers={} heads={} head_dim={} but model attention_layers={} heads={} head_dim={}. \
                         LFM2 sidecars must be calibrated with attention-ordinal layer indices.",
                        centers.n_layers,
                        centers.n_heads,
                        centers.head_dim,
                        n_attn,
                        config.num_attention_heads,
                        config.head_dim
                    ));
                }
                if (centers.rope_theta - config.rope_theta).abs()
                    > (config.rope_theta.abs().max(1.0) * 1e-4)
                {
                    return Err(format!(
                        "lfm2moe cask sidecar rope_theta mismatch: sidecar={} model={}",
                        centers.rope_theta, config.rope_theta
                    ));
                }
                let fa_layer_ids = lfm2_triattn_kv_layer_ids(&config);
                let base = EvictionCtx::new(
                    gpu,
                    &centers,
                    fa_layer_ids,
                    cask.budget,
                    cask.beta,
                    config.num_attention_heads,
                    config.num_key_value_heads,
                    config.head_dim,
                    config.head_dim,
                    config.rope_theta,
                    physical_cap,
                )
                .map_err(|e| format!("lfm2moe build EvictionCtx: {e}"))?;
                if cask.cask_m_folding {
                    tracing::info!(
                        "lfm2moe eviction: CASK α={:.2} m={} budget={} β={} physical_cap={}",
                        cask.core_frac,
                        cask.fold_m,
                        cask.budget,
                        cask.beta,
                        physical_cap,
                    );
                    Some(Eviction::Cask(CaskCtx::new(
                        base,
                        cask.core_frac,
                        cask.fold_m,
                    )))
                } else {
                    tracing::info!(
                        "lfm2moe eviction: TriAttention (plain drop) budget={} β={} physical_cap={}",
                        cask.budget, cask.beta, physical_cap,
                    );
                    Some(Eviction::Plain(base))
                }
            } else {
                None
            };
            // Resolve EOS via the tokenizer. LFM2.5 uses the standard
            // ChatML-ish `<|im_end|>`; fall back to common alternates, then 1.
            let eos_tok: u32 = {
                let try_one = |s: &str| -> Option<u32> {
                    let ids = tokenizer.encode(s);
                    if ids.len() == 1 {
                        Some(ids[0])
                    } else {
                        None
                    }
                };
                try_one("<|im_end|>")
                    .or_else(|| try_one("</s>"))
                    .or_else(|| try_one("<|endoftext|>"))
                    .unwrap_or(1)
            };
            let chat_template = resolve_chat_template(&hfq, path);
            let (chat_template, chat_template_profile) =
                profile_chat_template(chat_template, Some(&tokenizer));
            return Ok(LoadedModel {
                arch_id: hfq.arch_id,
                registered_backend: None,
                pp: 1,
                pp_gpus: None,
                pp_scratch_set: None,
                pp_dn_la_to_device: None,
                q35_config: None,
                q35_weights: None,
                q35_scratch: None,
                q35_kv_mode: None,
                q35_state_quant: None,
                q35_registry: SessionRegistry::default(),
                llama_config: None,
                llama_weights: None,
                llama_scratch: None,
                llama_kv: None,
                llama_backend: None,
                nemotron_backend: None,
                zaya_backend: None,
                qwen2_config: None,
                qwen2_weights: None,
                qwen2_state: None,
                deepseek4_config: None,
                deepseek4_weights: None,
                deepseek4_state: None,
                deepseek4_pbs: None,
                deepseek4_eos_tok: 0,
                mtp_mode: "auto".to_string(),
                mtp_k: 3,
                mtp_weights_present: false,
                minimax_config: None,
                minimax_weights: None,
                minimax_state: None,
                minimax_eos_tok: 0,
                lfm2moe_config: Some(config),
                lfm2moe_weights: Some(weights),
                lfm2_registry: SessionRegistry {
                    sessions: std::collections::HashMap::new(),
                    active_session_id: Some(crate::session::LFM2_LEGACY_SESSION_ID.to_string()),
                    allocation_epoch: next_qwen35_state_allocation_epoch(),
                },
                lfm2moe_eos_tok: eos_tok,
                dots_ocr_config: None,
                dots_ocr_weights: None,
                vision_config: None,
                vision_weights: None,
                gemma3_vl: None,
                gemma3_text: None,
                embeddinggemma: None,
                qwen3_embedding: None,
                tokenizer: Some(tokenizer),
                active: ResidentSession {
                    lfm2moe_state: Some(state),
                    ..Default::default()
                },
                max_seq,
                physical_cap,
                eviction,
                asst_turn_cache: std::collections::HashMap::new(),
                decoded_vocab: None,
                model_path: path.to_string(),
                memory: model_memory,
                #[cfg(feature = "arch-lfm2moe")]
                lfm2_dflash,
                dflash: None,
                // Opt-in; the daemon fills this in from `ngram_spec` after load.
                ngram: None,
                dspark: None,
                chat_template,
                chat_template_profile,
            });
        }
    }

    if is_qwen35_family_arch_id(hfq.arch_id) {
        // Qwen3.5 DeltaNet (arch=5 dense, arch=6 MoE/A3B). PR 8: dispatch
        // through the `Architecture` trait for the bring-up triple
        // (config → load → state). Forward passes below still call
        // `qwen35::*` directly — see crates/hipfire-arch-qwen35/src/arch.rs
        // for why static dispatch wins for the hot path.
        use hipfire_arch_qwen35::Qwen35;
        use hipfire_arch_qwen35_vl::Qwen35Vl;
        use hipfire_runtime::arch::Architecture;
        let config = <Qwen35 as Architecture>::config_from_hfq(&hfq).map_err(|e| e.to_string())?;
        if let Some(source) = resolved_dflash.as_ref() {
            let draft = load_resolved_dflash_config(source, &hfq, compose_manifest.as_ref())?;
            draft.validate_target_geometry(
                config.n_layers,
                config.dim,
                config.vocab_size,
                config.rope_theta,
            )?;
        }

        // Detect VL model: vision_config presence (from HFQ metadata) AND
        // actual vision tensors are required. Text-only Qwen3.5 models can
        // have vision_config in metadata without the patch_embed weights.
        // PR 9: bring-up triple now goes through the Qwen35Vl trait impl;
        // forward (`qwen35_vl::vision_forward`) stays a direct static call.
        let has_vision_tensors = hfq
            .find_tensor_info("model.visual.patch_embed.proj.weight")
            .is_some();
        let vision_config = <Qwen35Vl as Architecture>::config_from_hfq(&hfq).ok();
        let (vision_config, vision_weights) = if let Some(vc) = vision_config {
            if has_vision_tensors {
                let vw = <Qwen35Vl as Architecture>::load_weights(&mut hfq, &vc, gpu)
                    .map_err(|e| e.to_string())?;
                tracing::info!(
                    "VL model: vision encoder (hidden={}, layers={})",
                    vc.hidden_size,
                    vc.num_layers
                );
                (Some(vc), Some(vw))
            } else {
                (None, None) // text-only model, no vision tensors
            }
        } else {
            (None, None)
        };

        let weights = <Qwen35 as Architecture>::load_weights(&mut hfq, &config, gpu)?;

        // MMQ per-weight screening (#87): pre-screen all weight matrices at
        // load time so the first prefill doesn't pay the screening overhead.
        // Results are cached by device pointer in gpu.mmq_screen_cache.
        // Disabled by default on all arches; opt-in via mmq_screen=true or
        // HIPFIRE_MMQ_SCREEN=1. gfx906 is included for the opt-in case so
        // its ~700 µs/weight screening-reference dispatch doesn't surprise
        // first prefill if a user enables it.
        if gpu.mmq_screen
            && matches!(
                gpu.arch.as_str(),
                "gfx906"
                    | "gfx1100"
                    | "gfx1101"
                    | "gfx1102"
                    | "gfx1103"
                    | "gfx1150"
                    | "gfx1151"
                    | "gfx1152"
            )
        {
            let t0 = std::time::Instant::now();
            let (n_safe, n_unsafe) = screen_weights_qwen35(&weights, gpu);
            let elapsed = t0.elapsed();
            tracing::info!(
                "MMQ screening: {n_safe} safe, {n_unsafe} unsafe (threshold={:.2}, {:.1}ms)",
                gpu.mmq_screen_threshold,
                elapsed.as_secs_f64() * 1000.0,
            );
        }

        // KV cache modes (RotorQuant-style asymmetric: K rotated + V Q8):
        //   asym3 (default) — K at 3-bit rotated, V at Q8_0. 5.5× vs fp32.
        //                     Best quality/compression tradeoff — RotorQuant "planar3".
        //   asym4 — K at 4-bit rotated, V at Q8_0. 5.1× (slightly safer).
        //   asym2 — K at 2-bit rotated, V at Q8_0. 6.0× (loses rare-token tail).
        //   q8    — K+V both Q8_0. 3.76× (reference quality).
        //
        // Legacy "turbo{2,3,4}" aliases map to asym{2,3,4} for backward compat.
        //
        // All allocators go through the `_capped` entry points with
        // physical_cap derived above. Without eviction, physical_cap may still
        // be smaller than max_seq; the server reloads a larger worker when a
        // request needs more physical rows.
        let is_kv_layer = crate::session::qwen35_mixer_profile(&config.layer_types).kv_layer_mask();
        // Only the full-attention layers carry KV on these hybrids (16 of 64 on
        // Qwen3.8-27B); the rest are linear-attention and get a 1-element
        // placeholder. Every mode with a `_capped_filtered` constructor uses it —
        // the unfiltered ones allocated `n_layers` real buffers, ~4x what the
        // model can address. asym2/asym4/fwht* have a `_filtered` but no
        // `_capped_filtered`, so they still over-allocate; they are deprecated in
        // favour of kvarn (see the kv_cache schema field), so that is left alone.
        let kv = match kv_mode.as_str() {
            "fp32" | "f32" => kv::KvCache::new_gpu_capped_filtered(
                gpu,
                &is_kv_layer,
                config.n_kv_heads,
                config.head_dim,
                max_seq,
                physical_cap,
            )
            .map_err(|e| format!("{e}"))?,
            "q8" => {
                tracing::info!("KV cache: Q8");
                kv::KvCache::new_gpu_q8_capped_filtered(
                    gpu,
                    &is_kv_layer,
                    config.n_kv_heads,
                    config.head_dim,
                    max_seq,
                    physical_cap,
                )
                .map_err(|e| format!("{e}"))?
            }
            "asym4" | "turbo4" => kv::KvCache::new_gpu_asym4_capped(
                gpu,
                config.n_layers,
                config.n_kv_heads,
                config.head_dim,
                max_seq,
                physical_cap,
            )
            .map_err(|e| format!("{e}"))?,
            m @ ("kvarn" | "kvarn2" | "kvarn4" | "kvarn8") => {
                kv::KvCache::new_gpu_kvarn_capped_filtered(
                    gpu,
                    &is_kv_layer,
                    config.n_kv_heads,
                    config.head_dim,
                    max_seq,
                    physical_cap,
                    kvarn_bits_from_mode(m),
                )
                .map_err(|e| format!("{e}"))?
            }
            "asym2" | "turbo2" => kv::KvCache::new_gpu_asym2_capped(
                gpu,
                config.n_layers,
                config.n_kv_heads,
                config.head_dim,
                max_seq,
                physical_cap,
            )
            .map_err(|e| format!("{e}"))?,
            "asym3" | "turbo3" | "turbo" => kv::KvCache::new_gpu_asym3_capped_filtered(
                gpu,
                &is_kv_layer,
                config.n_kv_heads,
                config.head_dim,
                max_seq,
                physical_cap,
            )
            .map_err(|e| format!("{e}"))?,
            other => {
                tracing::warn!("KV cache: unrecognized '{other}', defaulting to asym3");
                kv::KvCache::new_gpu_asym3_capped_filtered(
                    gpu,
                    &is_kv_layer,
                    config.n_kv_heads,
                    config.head_dim,
                    max_seq,
                    physical_cap,
                )
                .map_err(|e| format!("{e}"))?
            }
        };
        // Q8 DeltaNet state can accumulate quality drift on long generation.
        // The load-time override exists for coherence A/B probes.
        // `parse_state_quant` decides, for bf16 and quantized artifacts alike.
        // This used to short-circuit to FP32 when `is_bf16_artifact`, which is
        // `hfq_has_bf16_weights()` — TRUE for QUANTIZED artifacts too, because
        // their norm tensors are BF16. The comment on the KV policy below
        // records the same trap being fixed there ("the prior rule wrongly
        // force-fp32'd these via the BF16 norms"); the DeltaNet-state branch
        // kept it, so no state precision could ever be selected for a
        // quantized model.
        //
        // Default behaviour is unchanged: with no override and no
        // HIPFIRE_DN_STATE_FP16, `parse_state_quant` returns FP32 for every
        // model. What changes is that an explicit request is now honoured.
        let dn_quant = {
            let parsed = parse_state_quant(state_quant_override, &config)?;
            resolve_tiny_model_state(&hfq, state_quant_override, parsed)
        };
        tracing::info!("DeltaNet state: {}", state_quant_label(dn_quant));
        let dn =
            DeltaNetState::new_with_quant(gpu, &config, dn_quant).map_err(|e| format!("{e}"))?;
        // Flash partials size with physical_cap (bounds the max_tiles the
        // flash kernel must address). When physical_cap == max_seq this is
        // identical to sizing-by-max_seq; otherwise it follows the worker's
        // current physical allocation.
        // Keep the request default at 128, but allocate enough history for
        // clients that explicitly ask for a wider repeat / OpenAI penalty
        // window.
        let scratch = qwen35::Qwen35Scratch::new_with_kv_max(gpu, &config, 2048, physical_cap)
            .map_err(|e| format!("{e}"))?;

        // Build eviction policy if the operator supplied a sidecar. Qwen3 (arch_id < 5)
        // lacks the FA/LA hybrid wiring TriAttention needs, so sidecars only take
        // effect on arch_id 5/6 — see the cask.rs docs for why CASK targets full-
        // attention layers only.
        let eviction = if let Some(source) = resolved_triattn.as_ref() {
            let centers = resolved_triattn_centers
                .as_ref()
                .expect("resolved TriAttention source has parsed centers")
                .uniform_centers()?;
            tracing::info!("TriAttention component source: {}", source.label());
            let fa_layer_ids =
                crate::session::qwen35_mixer_profile(&config.layer_types).kv_layer_indices();
            if fa_layer_ids.is_empty() {
                tracing::warn!("cask_sidecar set but model has no FullAttention layers — ignoring");
                None
            } else {
                let n_rot = (config.head_dim as f32 * config.partial_rotary_factor) as usize;
                let base = EvictionCtx::new(
                    gpu,
                    &centers,
                    fa_layer_ids,
                    cask.budget,
                    cask.beta,
                    config.n_heads,
                    config.n_kv_heads,
                    config.head_dim,
                    n_rot,
                    config.rope_theta,
                    physical_cap,
                )
                .map_err(|e| format!("build EvictionCtx: {e}"))?;
                if cask.cask_m_folding {
                    tracing::info!(
                        "eviction: CASK α={:.2} m={} budget={} β={} physical_cap={}",
                        cask.core_frac,
                        cask.fold_m,
                        cask.budget,
                        cask.beta,
                        physical_cap,
                    );
                    Some(Eviction::Cask(CaskCtx::new(
                        base,
                        cask.core_frac,
                        cask.fold_m,
                    )))
                } else {
                    tracing::info!(
                        "eviction: TriAttention (plain drop) budget={} β={} physical_cap={}",
                        cask.budget,
                        cask.beta,
                        physical_cap,
                    );
                    Some(Eviction::Plain(base))
                }
            }
        } else {
            None
        };
        // Optional DFlash draft: load the draft model's weights + a fresh set
        // of per-cycle scratch buffers (hidden ring, verify scratch, GdnTape,
        // DeltaNetSnapshot) sized for the target's max_seq. Once a source wins
        // precedence, an invalid component fails closed rather than falling
        // through to AR or a lower-precedence artifact.
        let dflash = if let Some(source) = resolved_dflash.as_ref() {
            // DFlash state (hidden_rb + target_hidden_host) sizes linearly with
            // the ctx_capacity argument. Pass `physical_cap` instead of
            // `max_seq` so eviction's smaller buffer caps VRAM: a 128K-advertised
            // model with physical_cap=896 allocates an 896-slot ring, not 128K.
            // Without eviction, callers may now choose physical_cap < max_seq;
            // pass the actual allocation size so the draft ring matches.
            let state = match source {
                ResolvedComponentSource::Explicit(dp) => {
                    load_dflash_state(dp, physical_cap, &config, &dn, weights.output.k, gpu)?
                }
                ResolvedComponentSource::Sibling(dp) => load_dflash_state(
                    &dp.to_string_lossy(),
                    physical_cap,
                    &config,
                    &dn,
                    weights.output.k,
                    gpu,
                )?,
                ResolvedComponentSource::Embedded => {
                    let manifest = compose_manifest
                        .as_ref()
                        .ok_or("embedded DFLASH selected without a manifest")?;
                    let view = file_component_view(&hfq, manifest, "dflash")
                        .map_err(|error| format!("embedded DFLASH view: {error}"))?
                        .ok_or("embedded DFLASH role disappeared from manifest")?;
                    load_dflash_state_source(
                        &view,
                        physical_cap,
                        &config,
                        &dn,
                        weights.output.k,
                        gpu,
                    )?
                }
            };
            tracing::info!(
                "DFlash draft loaded: source={} layers={} hidden={} block={}",
                source.label(),
                state.draft_config.as_ref().map_or(0, |c| c.n_layers),
                state.draft_config.as_ref().map_or(0, |c| c.hidden),
                state.draft_config.as_ref().map_or(0, |c| c.block_size),
            );
            Some(state)
        } else {
            None
        };

        let chat_template = resolve_chat_template(&hfq, path);
        let (chat_template, chat_template_profile) =
            profile_chat_template(chat_template, Some(&tokenizer));
        let sequence_state = Some(SequenceState::new(
            crate::session::qwen35_mixer_profile(&config.layer_types),
            Some(kv),
            Some(Box::new(dn)),
        ));
        Ok(LoadedModel {
            arch_id: hfq.arch_id,
            registered_backend: None,
            pp: 1,
            pp_gpus: None,
            pp_scratch_set: None,
            pp_dn_la_to_device: None,
            q35_config: Some(config),
            q35_weights: Some(weights),
            q35_scratch: Some(scratch),
            q35_kv_mode: Some(kv_mode.clone()),
            q35_state_quant: Some(dn_quant),
            q35_registry: SessionRegistry {
                sessions: std::collections::HashMap::new(),
                active_session_id: Some(QWEN35_LEGACY_SESSION_ID.to_string()),
                allocation_epoch: next_qwen35_state_allocation_epoch(),
            },
            llama_config: None,
            llama_weights: None,
            llama_scratch: None,
            llama_kv: None,
            llama_backend: None,
            nemotron_backend: None,
            zaya_backend: None,
            qwen2_config: None,
            qwen2_weights: None,
            qwen2_state: None,
            deepseek4_config: None,
            deepseek4_weights: None,
            deepseek4_state: None,
            deepseek4_pbs: None,
            deepseek4_eos_tok: 0,
            mtp_mode: "auto".to_string(),
            mtp_k: 3,
            mtp_weights_present: false,
            minimax_config: None,
            minimax_weights: None,
            minimax_state: None,
            minimax_eos_tok: 0,
            #[cfg(feature = "arch-lfm2moe")]
            lfm2moe_config: None,
            #[cfg(feature = "arch-lfm2moe")]
            lfm2moe_weights: None,
            #[cfg(feature = "arch-lfm2moe")]
            lfm2_registry: SessionRegistry::default(),
            #[cfg(feature = "arch-lfm2moe")]
            lfm2moe_eos_tok: 0,
            dots_ocr_config: None,
            dots_ocr_weights: None,
            vision_config,
            vision_weights,
            gemma3_vl: None,
            gemma3_text: None,
            embeddinggemma: None,
            qwen3_embedding: None,
            tokenizer: Some(tokenizer),
            active: ResidentSession {
                sequence_state,
                ..Default::default()
            },
            max_seq,
            physical_cap,
            eviction,
            asst_turn_cache: std::collections::HashMap::new(),
            decoded_vocab: None,
            model_path: path.to_string(),
            memory: model_memory,
            #[cfg(feature = "arch-lfm2moe")]
            lfm2_dflash: None,
            dflash,
            // Opt-in; the daemon fills this in from `ngram_spec` after load.
            ngram: None,
            dspark: None,
            chat_template,
            chat_template_profile,
        })
    } else {
        // Qwen3 / LLaMA — no eviction supported on this path (TriAttention needs
        // the FA/LA hybrid wiring from arch_id 5/6). physical_cap == max_seq.
        // PR 11: dispatch through the `Architecture` trait for the bring-up
        // triple (config → load → scratch). Forward passes below still call
        // `llama::*` directly — see crates/hipfire-arch-llama/src/arch.rs
        // for why static dispatch wins for the hot path.
        use hipfire_runtime::arch::Architecture;
        let config = <Llama as Architecture>::config_from_hfq(&hfq).map_err(|e| e.to_string())?;
        let weights = <Llama as Architecture>::load_weights(&mut hfq, &config, gpu)?;
        // Honor the requested kv_mode. LLaMA-family head_dim is typically 64/128,
        // so the rotated-K asym{2,3,4} caches (head_dim ∈ {128,256}) do not apply
        // here; surface that as an explicit error rather than silently building a
        // Q8 cache under an asym label. Supported menu: fp32 | q8.
        let kv = match kv_mode.as_str() {
            "fp32" | "f32" => {
                tracing::info!("KV cache: FP32");
                kv::KvCache::new_gpu(
                    gpu,
                    config.n_layers,
                    config.n_kv_heads,
                    config.head_dim,
                    max_seq,
                )
                .map_err(|e| format!("{e}"))?
            }
            "q8" | "int8" | "auto" | "" => {
                tracing::info!("KV cache: Q8");
                kv::KvCache::new_gpu_q8(
                    gpu,
                    config.n_layers,
                    config.n_kv_heads,
                    config.head_dim,
                    max_seq,
                )
                .map_err(|e| format!("{e}"))?
            }
            // The LLaMA arch has no rotated-K (asym) or KVarN attention path —
            // those need head_dim ∈ {128,256} and a distinct kernel, and llama
            // head_dim is frequently 64 (e.g. Llama-3.2-1B). Since the global
            // default kv_mode is now `kvarn`, an unqualified llama load would
            // otherwise FAIL rather than serve. Fall back to Q8 (with a clear
            // note) so llama stays servable under the kvarn default; an operator
            // who explicitly wants FP32 still gets it via the fp32 arm.
            "kvarn" | "kvarn2" | "kvarn4" | "kvarn8" | "asym2" | "asym3" | "asym4" | "turbo"
            | "turbo2" | "turbo3" | "turbo4" => {
                tracing::warn!(
                    "KV cache: Q8 (llama has no kvarn/asym path at head_dim={}; \
                     '{kv_mode}' → Q8)",
                    config.head_dim
                );
                kv::KvCache::new_gpu_q8(
                    gpu,
                    config.n_layers,
                    config.n_kv_heads,
                    config.head_dim,
                    max_seq,
                )
                .map_err(|e| format!("{e}"))?
            }
            other => {
                return Err(format!(
                    "kv_mode '{other}' is not supported for the LLaMA arch \
                     (head_dim={}); rotated-K asym{{2,3,4}} require head_dim ∈ {{128,256}}. \
                     Use fp32 or q8.",
                    config.head_dim
                ));
            }
        };
        let scratch = <Llama as Architecture>::new_state(gpu, &config)?;
        // P3.2: assemble the ServingBackend (owns config/weights/scratch/kv); the
        // separate llama_* fields stay None. (HFQ load path — mirrors the
        // safetensors path below.)
        let mut llama_backend =
            hipfire_arch_llama::LlamaBackend::new(hfq.arch_id, config, weights, scratch, kv);
        // DSpark sidecar discovery (dense LLaMA/Qwen3, arch 0/1). Mirrors the
        // DFlash draft wiring: look for a `<stem>-<quant>.dspark.hfq` next to the
        // target and, when found, load the drafter + build the greedy speculator.
        // `None` (no sidecar / disabled / non-0-1 arch) leaves the AR path
        // byte-unchanged.
        let dspark = maybe_load_dspark(&mut llama_backend, hfq.arch_id, path, max_seq, gpu);
        let chat_template = resolve_chat_template(&hfq, path);
        let (chat_template, chat_template_profile) =
            profile_chat_template(chat_template, Some(&tokenizer));
        Ok(LoadedModel {
            arch_id: hfq.arch_id,
            registered_backend: None,
            pp: 1,
            pp_gpus: None,
            pp_scratch_set: None,
            pp_dn_la_to_device: None,
            q35_config: None,
            q35_weights: None,
            q35_scratch: None,
            q35_kv_mode: None,
            q35_state_quant: None,
            q35_registry: SessionRegistry::default(),
            llama_config: None,
            llama_weights: None,
            llama_scratch: None,
            llama_kv: None,
            llama_backend: Some(llama_backend),
            nemotron_backend: None,
            zaya_backend: None,
            qwen2_config: None,
            qwen2_weights: None,
            qwen2_state: None,
            deepseek4_config: None,
            deepseek4_weights: None,
            deepseek4_state: None,
            deepseek4_pbs: None,
            deepseek4_eos_tok: 0,
            mtp_mode: "auto".to_string(),
            mtp_k: 3,
            mtp_weights_present: false,
            minimax_config: None,
            minimax_weights: None,
            minimax_state: None,
            minimax_eos_tok: 0,
            #[cfg(feature = "arch-lfm2moe")]
            lfm2moe_config: None,
            #[cfg(feature = "arch-lfm2moe")]
            lfm2moe_weights: None,
            #[cfg(feature = "arch-lfm2moe")]
            lfm2_registry: SessionRegistry::default(),
            #[cfg(feature = "arch-lfm2moe")]
            lfm2moe_eos_tok: 0,
            dots_ocr_config: None,
            dots_ocr_weights: None,
            vision_config: None,
            vision_weights: None,
            gemma3_vl: None,
            gemma3_text: None,
            embeddinggemma: None,
            qwen3_embedding: None,
            tokenizer: Some(tokenizer),
            active: ResidentSession::default(),
            max_seq,
            physical_cap: max_seq,
            eviction: None,
            asst_turn_cache: std::collections::HashMap::new(),
            decoded_vocab: None,
            model_path: path.to_string(),
            memory: model_memory,
            #[cfg(feature = "arch-lfm2moe")]
            lfm2_dflash: None,
            dflash: None,
            // Opt-in; the daemon fills this in from `ngram_spec` after load.
            ngram: None,
            dspark,
            chat_template,
            chat_template_profile,
        })
    }
}

#[cfg(target_os = "linux")]
fn load_embeddinggemma_npu_projector(
    hfq: &HfqFile,
    config: &hipfire_arch_embeddinggemma::EmbeddingGemmaConfig,
    storage_requires_npu: bool,
) -> Option<std::sync::Mutex<hipfire_arch_embeddinggemma::NpuOpusProjector>> {
    let requested = match std::env::var("HIPFIRE_EMBED_RESIDENT_LAYER") {
        Ok(value) => value != "0",
        Err(_) => storage_requires_npu,
    };
    if !requested {
        return None;
    }
    let batch = match std::env::var("HIPFIRE_EMBED_NPU_BATCH") {
        Ok(value) => match value.parse::<usize>() {
            Ok(batch) if batch > 0 => batch,
            _ => {
                tracing::warn!(
                    "embeddinggemma NPU disabled: HIPFIRE_EMBED_NPU_BATCH must be positive"
                );
                return None;
            }
        },
        Err(_) => 1,
    };
    let cache_root = std::env::var_os("HIPFIRE_EMBED_NPU_CACHE")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".hipfire/npu")));
    let Some(cache_root) = cache_root else {
        tracing::warn!(
            "embeddinggemma NPU disabled: set HIPFIRE_EMBED_NPU_CACHE or HOME to locate caches"
        );
        return None;
    };
    match hipfire_arch_embeddinggemma::NpuOpusProjector::load_cached_for_batch(
        hfq,
        config,
        &cache_root,
        batch,
    ) {
        Ok(projector) if !storage_requires_npu || projector.resident_layer_enabled() => {
            tracing::info!(
                "embeddinggemma: resident NPU enabled, batch={batch}, cache={}",
                cache_root.display()
            );
            Some(std::sync::Mutex::new(projector))
        }
        Ok(_) => {
            tracing::warn!(
                "embeddinggemma NPU unavailable (row-padded OQ8 requires the complete \
                 resident-layer cache); serving will use an explicit GPU fallback if configured"
            );
            None
        }
        Err(error) => {
            tracing::warn!(
                "embeddinggemma NPU unavailable ({error}); serving will use an explicit GPU fallback if configured"
            );
            None
        }
    }
}

#[cfg(target_os = "linux")]
fn load_embeddinggemma_gpu_fallback_weights(
    config: &hipfire_arch_embeddinggemma::EmbeddingGemmaConfig,
    gpu: &mut hipfire_rdna::Gpu,
) -> Result<hipfire_arch_embeddinggemma::EmbeddingGemmaWeights, String> {
    let fallback_path = std::env::var_os("HIPFIRE_EMBED_GPU_FALLBACK_MODEL")
        .map(PathBuf::from)
        .ok_or_else(|| "HIPFIRE_EMBED_GPU_FALLBACK_MODEL is not set".to_string())?;
    if !fallback_path.is_file() {
        return Err(format!(
            "embeddinggemma NPU serving requires a GPU fallback model at {} (set HIPFIRE_EMBED_GPU_FALLBACK_MODEL)",
            fallback_path.display()
        ));
    }
    let mut fallback_hfq = HfqFile::open(&fallback_path)
        .map_err(|error| format!("open EmbeddingGemma GPU fallback: {error}"))?;
    let fallback_config =
        hipfire_arch_embeddinggemma::config_from_metadata_json(&fallback_hfq.metadata_json)
            .ok_or_else(|| "embeddinggemma GPU fallback has no valid config".to_string())?;
    if fallback_config.hidden_size != config.hidden_size
        || fallback_config.num_hidden_layers != config.num_hidden_layers
        || fallback_config.num_attention_heads != config.num_attention_heads
        || fallback_config.num_key_value_heads != config.num_key_value_heads
        || fallback_config.intermediate_size != config.intermediate_size
        || fallback_config.embedding_dim != config.embedding_dim
    {
        return Err(format!(
            "embeddinggemma GPU fallback geometry does not match {}",
            fallback_path.display()
        ));
    }
    tracing::info!(
        "embeddinggemma: GPU fallback weights={}",
        fallback_path.display()
    );
    hipfire_arch_embeddinggemma::EmbeddingGemmaWeights::load(
        &mut fallback_hfq,
        &fallback_config,
        gpu,
    )
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct EmbeddingGemmaStorageContract {
    explicit_row_padded_oq8: bool,
    legacy_implicit: bool,
}

impl EmbeddingGemmaStorageContract {
    fn requires_npu(self) -> bool {
        self.explicit_row_padded_oq8 || self.legacy_implicit
    }
}

/// Validate the on-disk OQ8 geometry before either backend sees the payload.
/// New artifacts use qt=43 explicitly. Legacy qt=35 artifacts are recognized
/// only when logical K is ragged and the payload has one padded group sequence
/// per row; this keeps existing NPU artifacts diagnosable without making the
/// filename part of the storage contract.
fn embeddinggemma_storage_contract(
    tensors: &[HfqTensorInfo],
) -> Result<EmbeddingGemmaStorageContract, String> {
    let mut contract = EmbeddingGemmaStorageContract::default();
    for info in tensors {
        let explicit = info.quant_type == QuantType::Oq8G256RowPadded.code();
        let legacy_candidate = info.quant_type == QuantType::Oq8G256.code()
            && info.shape.len() == 2
            && info.shape[1] % 256 != 0;
        if !explicit && !legacy_candidate {
            continue;
        }
        if info.shape.len() != 2 {
            return Err(format!(
                "embeddinggemma: row-padded OQ8 tensor {} must be rank two, got {:?}",
                info.name, info.shape
            ));
        }
        let rows = info.shape[0] as usize;
        let cols = info.shape[1] as usize;
        if cols % 256 == 0 {
            return Err(format!(
                "embeddinggemma: tensor {} uses row-padded OQ8 with aligned K={cols}",
                info.name
            ));
        }
        let expected = QuantType::Oq8G256RowPadded
            .matrix_tensor_bytes(rows, cols)
            .ok_or_else(|| {
                format!(
                    "embeddinggemma: OQ8 byte geometry overflow for {}",
                    info.name
                )
            })?;
        if info.data_size != expected {
            return Err(format!(
                "embeddinggemma: row-padded OQ8 tensor {} has {} bytes; expected {} for [{rows},{cols}]",
                info.name, info.data_size, expected
            ));
        }
        contract.explicit_row_padded_oq8 |= explicit;
        contract.legacy_implicit |= legacy_candidate;
    }
    Ok(contract)
}

/// Multi-GPU pipeline-parallel load path (Stage 7 of #58). Refuses VL,
/// non-Qwen3.5 architectures and (transitively, via the upstream "load"
/// handler) DFlash, CASK and PFlash. Returns a `LoadedModel` with `pp_gpus`,
/// `pp_scratch_set` and `pp_dn_la_to_device` populated; the daemon's primary
/// `gpu` parameter is unused on this path. Eviction is refused at this layer
/// because TriAttention/CASK/PFlash live on a single device and are not v1
/// targets for pp>1 — physical_cap == max_seq accordingly.
pub fn load_model_pp(
    path: &str,
    max_seq: usize,
    kv_mode_override: Option<&str>,
    state_quant_override: Option<&str>,
    pp: usize,
    _gpu: &mut hipfire_rdna::Gpu,
) -> Result<LoadedModel, String> {
    let mut kv_mode = kv_mode_override
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .unwrap_or_else(|| std::env::var("HIPFIRE_KV_MODE").unwrap_or_default());
    // `auto` (the config default) and unset both mean "the loader chooses".
    // KVarN: ~8x smaller K than fp32 with no token dropped, and batched-prefill
    // eligible. fp32 is an oracle/debug mode, not a production path — ask for it
    // explicitly (kv_cache=fp32) if you want it.
    if kv_mode.is_empty() || kv_mode == "auto" {
        kv_mode = "kvarn".to_string();
    }
    let hfq = HfqFile::open(Path::new(path)).map_err(|e| format!("{e}"))?;
    let max_seq = clamp_max_seq_to_model_context(max_seq, &hfq.metadata_json);
    let model_memory = hfq_model_memory(path, &hfq);
    warn_if_unoptimized(path, &hfq);
    // KV precision policy:
    //   * BF16-DOMINANT model (full-precision artifact) -> force fp32 KV (mixing
    //     a quantized KV under bf16 weights is a precision mismatch).
    //   * Quantized model (MQ4/Q8 weights, only norms BF16) -> honor an explicit
    //     kv_mode; otherwise default to fp32 for now. A quantized KV (q8/asym/
    //     KVarN) makes the model batched-prefill eligible (~32x prefill); the
    //     prior rule wrongly force-fp32'd these via the BF16 norms, locking them
    //     to the per-token path. That flip is done: the default is now KVarN.
    // KV precision: respect an explicit kv_mode (config/CLI/JSON) for ALL
    // models, including BF16-dominant ones. Default to fp32 only when the
    // caller left it unspecified. (Previously BF16-dominant artifacts were
    // force-overridden to fp32 even when the operator asked for q8/asym/KVarN
    // — that silently discarded the requested KV quant. fp32 is reserved for
    // oracle/debug runs; quantizing KV under bf16 weights is an opt-in the
    // operator owns.)
    reject_deprecated_kv_mode(&kv_mode)?;
    let tokenizer = hipfire_model::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
        .map_err(|e| format!("tokenizer not found: {e}"))?;

    if !is_qwen35_family_arch_id(hfq.arch_id) {
        return Err(format!(
            "pp>1 supports Qwen3.5 dense (arch_id=5) and Qwen3.5-MoE / \
             Qwen3.6-A3B (arch_id=6) only; got arch_id={}. LLaMA / Qwen3 \
             dense (arch_id<5) is pp=1 only.",
            hfq.arch_id
        ));
    }
    if qwen35_vl::vision_config_from_hfq(&hfq).is_some()
        && hfq
            .find_tensor_info("model.visual.patch_embed.proj.weight")
            .is_some()
    {
        return Err(
            "pp>1 does not support VL models in v1; see issue #58 v1.1 roadmap".to_string(),
        );
    }

    let config = qwen35::config_from_hfq(&hfq).ok_or("failed to read Qwen3.5 config")?;

    // HIPFIRE_PP_LAYERS="a,b,..." overrides uniform split. Length must equal
    // pp; sum must equal n_layers; each entry >= 1. Used to shift layers off
    // dev 0 when token_embd asymmetry caps max_seq under uniform split.
    let mut gpus = match std::env::var("HIPFIRE_PP_LAYERS")
        .ok()
        .filter(|s| !s.is_empty())
    {
        Some(spec) => {
            let counts: Result<Vec<usize>, _> =
                spec.split(',').map(|s| s.trim().parse::<usize>()).collect();
            let counts = counts.map_err(|e| format!("HIPFIRE_PP_LAYERS parse: {e}"))?;
            if counts.len() != pp {
                return Err(format!(
                    "HIPFIRE_PP_LAYERS has {} entries, expected pp={}",
                    counts.len(),
                    pp
                ));
            }
            let sum: usize = counts.iter().sum();
            if sum != config.n_layers {
                return Err(format!(
                    "HIPFIRE_PP_LAYERS sum={} != n_layers={}",
                    sum, config.n_layers
                ));
            }
            tracing::info!("HIPFIRE_PP_LAYERS override: {:?}", counts);
            Gpus::init_layers(&counts).map_err(|e| format!("{e}"))?
        }
        None => Gpus::init_uniform(pp, config.n_layers).map_err(|e| format!("{e}"))?,
    };

    let weights =
        qwen35::load_weights_multi(&hfq, &config, &mut gpus).map_err(|e| format!("{e}"))?;

    // KV cache (asym3 default, q8/asym4/asym2/fwht{4,3,2} selectable).
    // physical_cap == max_seq on this path — eviction is refused at load.
    let kv = match kv_mode.as_str() {
        "fp32" | "f32" => kv::KvCache::new_gpu_multi(
            &mut gpus,
            config.n_layers,
            config.n_kv_heads,
            config.head_dim,
            max_seq,
        )
        .map_err(|e| format!("{e}"))?,
        "q8" => kv::KvCache::new_gpu_q8_capped_multi(
            &mut gpus,
            config.n_layers,
            config.n_kv_heads,
            config.head_dim,
            max_seq,
            max_seq,
        )
        .map_err(|e| format!("{e}"))?,
        "asym4" | "turbo4" => kv::KvCache::new_gpu_asym4_capped_multi(
            &mut gpus,
            config.n_layers,
            config.n_kv_heads,
            config.head_dim,
            max_seq,
            max_seq,
        )
        .map_err(|e| format!("{e}"))?,
        "asym2" | "turbo2" => kv::KvCache::new_gpu_asym2_capped_multi(
            &mut gpus,
            config.n_layers,
            config.n_kv_heads,
            config.head_dim,
            max_seq,
            max_seq,
        )
        .map_err(|e| format!("{e}"))?,
        "asym3" | "turbo3" | "turbo" => kv::KvCache::new_gpu_asym3_capped_multi(
            &mut gpus,
            config.n_layers,
            config.n_kv_heads,
            config.head_dim,
            max_seq,
            max_seq,
        )
        .map_err(|e| format!("{e}"))?,
        "fwht4" => kv::KvCache::new_gpu_fwht4_capped_multi(
            &mut gpus,
            config.n_layers,
            config.n_kv_heads,
            config.head_dim,
            max_seq,
            max_seq,
        )
        .map_err(|e| format!("{e}"))?,
        "fwht3" => kv::KvCache::new_gpu_fwht3_capped_multi(
            &mut gpus,
            config.n_layers,
            config.n_kv_heads,
            config.head_dim,
            max_seq,
            max_seq,
        )
        .map_err(|e| format!("{e}"))?,
        "fwht2" => kv::KvCache::new_gpu_fwht2_capped_multi(
            &mut gpus,
            config.n_layers,
            config.n_kv_heads,
            config.head_dim,
            max_seq,
            max_seq,
        )
        .map_err(|e| format!("{e}"))?,
        other => {
            tracing::warn!("KV cache: unrecognized '{other}', defaulting to asym3");
            kv::KvCache::new_gpu_asym3_capped_multi(
                &mut gpus,
                config.n_layers,
                config.n_kv_heads,
                config.head_dim,
                max_seq,
                max_seq,
            )
            .map_err(|e| format!("{e}"))?
        }
    };

    // Mirror the pp=1 state-mode parser so pp parity probes can force the
    // same DeltaNet state representation.
    // `parse_state_quant` decides, for bf16 and quantized artifacts alike.
    // This used to short-circuit to FP32 when `is_bf16_artifact`, which is
    // `hfq_has_bf16_weights()` — TRUE for QUANTIZED artifacts too, because
    // their norm tensors are BF16. The comment on the KV policy below
    // records the same trap being fixed there ("the prior rule wrongly
    // force-fp32'd these via the BF16 norms"); the DeltaNet-state branch
    // kept it, so no state precision could ever be selected for a
    // quantized model.
    //
    // Default behaviour is unchanged: with no override and no
    // HIPFIRE_DN_STATE_FP16, `parse_state_quant` returns FP32 for every
    // model. What changes is that an explicit request is now honoured.
    let dn_quant = {
        let parsed = parse_state_quant(state_quant_override, &config)?;
        resolve_tiny_model_state(&hfq, state_quant_override, parsed)
    };
    tracing::info!("DeltaNet state: {}", state_quant_label(dn_quant));
    let (dn, la_to_device) = DeltaNetState::new_with_quant_multi(&mut gpus, &config, dn_quant)
        .map_err(|e| format!("{e}"))?;

    let scratch_set = Qwen35ScratchSet::new_with_kv_max_multi(&mut gpus, &config, 2048, max_seq)
        .map_err(|e| format!("{e}"))?;

    // ROCm 6.4.3 gotcha: enable_peer_access AFTER all allocations are live.
    // See multi_gpu.rs::enable_peer_all docstring for the silent-success bug
    // when the call precedes hipMalloc.
    let _peer = gpus
        .enable_peer_all()
        .map_err(|e| format!("enable_peer_all: {e}"))?;

    tracing::info!(
        "pp={pp} loaded: layer_to_device={:?}, output_device={}, peer_access={}",
        gpus.layer_to_device,
        gpus.output_device,
        gpus.peer_access_enabled,
    );

    let chat_template = resolve_chat_template(&hfq, path);
    let (chat_template, chat_template_profile) =
        profile_chat_template(chat_template, Some(&tokenizer));

    let sequence_state = Some(SequenceState::new(
        crate::session::qwen35_mixer_profile(&config.layer_types),
        Some(kv),
        Some(Box::new(dn)),
    ));
    Ok(LoadedModel {
        arch_id: hfq.arch_id,
        registered_backend: None,
        pp,
        pp_gpus: Some(gpus),
        pp_scratch_set: Some(scratch_set),
        pp_dn_la_to_device: Some(la_to_device),
        q35_config: Some(config),
        q35_weights: Some(weights),
        q35_scratch: None,
        q35_kv_mode: None,
        q35_state_quant: None,
        q35_registry: SessionRegistry::default(),
        llama_config: None,
        llama_weights: None,
        llama_scratch: None,
        llama_kv: None,
        llama_backend: None,
        nemotron_backend: None,
        zaya_backend: None,
        qwen2_config: None,
        qwen2_weights: None,
        qwen2_state: None,
        deepseek4_config: None,
        deepseek4_weights: None,
        deepseek4_state: None,
        deepseek4_pbs: None,
        deepseek4_eos_tok: 0,
        mtp_mode: "auto".to_string(),
        mtp_k: 3,
        mtp_weights_present: false,
        minimax_config: None,
        minimax_weights: None,
        minimax_state: None,
        minimax_eos_tok: 0,
        #[cfg(feature = "arch-lfm2moe")]
        lfm2moe_config: None,
        #[cfg(feature = "arch-lfm2moe")]
        lfm2moe_weights: None,
        #[cfg(feature = "arch-lfm2moe")]
        lfm2_registry: SessionRegistry::default(),
        #[cfg(feature = "arch-lfm2moe")]
        lfm2moe_eos_tok: 0,
        dots_ocr_config: None,
        dots_ocr_weights: None,
        vision_config: None,
        vision_weights: None,
        gemma3_vl: None,
        gemma3_text: None,
        embeddinggemma: None,
        qwen3_embedding: None,
        tokenizer: Some(tokenizer),
        active: ResidentSession {
            sequence_state,
            ..Default::default()
        },
        max_seq,
        physical_cap: max_seq,
        eviction: None,
        asst_turn_cache: std::collections::HashMap::new(),
        decoded_vocab: None,
        model_path: path.to_string(),
        memory: model_memory,
        #[cfg(feature = "arch-lfm2moe")]
        lfm2_dflash: None,
        dflash: None,
        // Opt-in; the daemon fills this in from `ngram_spec` after load.
        ngram: None,
        dspark: None,
        chat_template,
        chat_template_profile,
    })
}

/// Pre-screen all Qwen3.5/3.6 weight matrices for MMQ safety (#87).
/// Returns (n_safe, n_unsafe). Results are cached in gpu.mmq_screen_cache.
pub fn screen_weights_qwen35(
    weights: &qwen35::Qwen35Weights,
    gpu: &mut hipfire_rdna::Gpu,
) -> (usize, usize) {
    use hipfire_arch_qwen35::qwen35::LayerWeights;
    let mut n_safe = 0usize;
    let mut n_unsafe = 0usize;

    for layer in &weights.layers {
        // Collect all weight tensors for this layer that could use MMQ
        let wts: Vec<(&hipfire_runtime::weights::WeightTensor, &str)> = match layer {
            LayerWeights::DeltaNet(l) => vec![
                (&l.wqkv, "qkvza.qkv"),
                (&l.wz, "qkvza.z"),
                (&l.w_beta, "qkvza.beta"),
                (&l.w_alpha, "qkvza.alpha"),
                (&l.w_gate, "gate_up.gate"),
                (&l.w_up, "gate_up.up"),
                (&l.wo, "residual"),
            ],
            LayerWeights::FullAttn(l) => vec![
                (&l.wq, "qkv.q"),
                (&l.wk, "qkv.k"),
                (&l.wv, "qkv.v"),
                (&l.w_gate, "gate_up.gate"),
                (&l.w_up, "gate_up.up"),
                (&l.wo, "residual"),
            ],
            LayerWeights::DeltaNetMoe(l) => vec![
                (&l.wqkv, "qkvza.qkv"),
                (&l.wz, "qkvza.z"),
                (&l.w_beta, "qkvza.beta"),
                (&l.w_alpha, "qkvza.alpha"),
                (&l.wo, "residual"),
            ],
            LayerWeights::FullAttnMoe(l) => vec![
                (&l.wq, "qkv.q"),
                (&l.wk, "qkv.k"),
                (&l.wv, "qkv.v"),
                (&l.wo, "residual"),
            ],
        };

        for (wt, _name) in wts {
            // MMQ kernels only operate on HFQ4G256 weights. Other formats
            // (MQ3, MQ2, HFQ6, etc.) use different dispatch paths and must
            // not be fed to the HFQ4-specific screening kernels — buffer
            // layout mismatch would read past the end. See PR #106.
            if !matches!(
                wt.gpu_dtype,
                hipfire_rdna::DType::HFQ4G256 | hipfire_rdna::DType::MQ4G256
            ) {
                continue;
            }
            if gpu.mmq_screen_weight(&wt.buf, wt.m, wt.k) {
                n_safe += 1;
            } else {
                n_unsafe += 1;
            }
        }
    }

    (n_safe, n_unsafe)
}

/// Free all GPU resources held by a loaded model (weights, scratch, KV/state,
/// eviction scratch, DFlash drafter) by consuming it. Per-arch teardown mirrors
/// whichever Option fields are populated.
pub fn unload_model(mut m: LoadedModel, gpu: &mut hipfire_rdna::Gpu) {
    // Multi-GPU branch (Stage 7 of #58). Frees per-device tensors through the
    // Gpus orchestrator, then invalidates per-device caches so the next load
    // can't inherit stale verdicts at recycled device addresses. Order
    // matches the alloc order in load_model_pp reversed: scratch → kv → dn →
    // weights, so each free targets a still-live owner.
    if m.pp > 1 {
        let mut gpus = m.pp_gpus.expect("pp>1 must carry pp_gpus");
        if let Some(scratch_set) = m.pp_scratch_set {
            scratch_set.free_gpu_multi(&mut gpus);
        }
        if let Some(ss) = m.active.sequence_state.take() {
            let (kv, recurrent) = ss.into_parts();
            if let Some(kv) = kv {
                kv.free_gpu_multi(&mut gpus);
            }
            if let Some(r) = recurrent {
                let dn = *r
                    .into_any()
                    .downcast::<DeltaNetState>()
                    .expect("qwen35 recurrent state is DeltaNetState");
                let la_to_device = m.pp_dn_la_to_device.expect("pp>1 must carry la_to_device");
                dn.free_gpu_multi(&mut gpus, &la_to_device);
            }
        }
        if let Some(w) = m.q35_weights {
            w.free_gpu_multi(&mut gpus);
        }
        for g in gpus.devices.iter_mut() {
            g.invalidate_weight_caches();
            g.invalidate_graph_state();
            g.drain_pool();
        }
        let _ = gpu;
        return;
    }
    // DFlash state: free every GPU-resident component explicitly. No `Drop` on
    // GpuTensor/DeviceBuffer, so the ring buffer, verify scratch, snapshot, tape,
    // and (optional) DDTree state must each be returned to the pool — otherwise a
    // mid-session load/unload cycle strands them until daemon exit.
    if let Some(df) = m.dflash {
        // The drafter half is absent on an n-gram-only load; the target-side
        // buffers below are always present and always need returning.
        if let Some(draft_weights) = df.draft_weights {
            draft_weights.free_gpu(gpu);
        }
        if let Some(draft_scratch) = df.draft_scratch {
            draft_scratch.free_gpu(gpu);
        }
        if let Some(hidden_rb) = df.hidden_rb {
            hidden_rb.free_gpu(gpu);
        }
        df.verify_scratch.free_gpu(gpu);
        df.target_snap.free_gpu(gpu);
        df.gdn_tape.free_gpu(gpu);
        if let Some(ddtree) = df.ddtree {
            ddtree.free_gpu(gpu);
        }
    }
    // DSpark speculator: release every GPU buffer the drafter owns (drafter body
    // weights/scratch + block KV + main-hidden cache). `Speculator::free` is a
    // required trait method, so a forgotten buffer is a compile error, not a leak.
    if let Some(ds) = m.dspark {
        ds.speculator.free(gpu);
    }
    #[cfg(feature = "arch-lfm2moe")]
    if let Some(df) = m.lfm2_dflash {
        df.draft_weights.free_gpu(gpu);
        df.draft_scratch.free_gpu(gpu);
        df.target_snap.free_gpu(gpu);
    }
    // Free eviction context (centers + scratch tensors) if active.
    if let Some(ev) = m.eviction {
        ev.free_gpu(gpu);
    }
    // Free KV cache + DeltaNet state + scratch first (small fraction of VRAM).
    if let Some(ss) = m.active.sequence_state.take() {
        let (kv, recurrent) = ss.into_parts();
        if let Some(kv) = kv {
            kv.free_gpu(gpu);
        }
        if let Some(r) = recurrent {
            let dn = *r
                .into_any()
                .downcast::<DeltaNetState>()
                .expect("qwen35 recurrent state is DeltaNetState");
            dn.free_gpu(gpu);
        }
    }
    if let Some(s) = m.q35_scratch {
        s.free_gpu(gpu);
    }
    if let Some(kv) = m.llama_kv {
        kv.free_gpu(gpu);
    }
    if let Some(s) = m.llama_scratch {
        s.free_gpu(gpu);
    }
    // Qwen2 state holds both the per-step scratch AND the KV cache — one
    // free_gpu call handles both. (Compare LLaMA where ForwardScratch and
    // KvCache are separate fields.)
    if let Some(s) = m.qwen2_state {
        s.free_gpu(gpu);
    }
    // V4F (arch_id=9) per-session scratch + per-layer SWA/indexer/
    // compressor caches. Without these `unload_model` would leak ~tens
    // of MB of state buffers per load/unload cycle, defeating idle
    // eviction.
    if let Some(s) = m.deepseek4_state {
        s.free_gpu(gpu);
    }
    if let Some(pbs) = m.deepseek4_pbs {
        pbs.free_gpu(gpu);
    }
    // MiniMax-M2 (arch_id=10): state (KV + scratch + device pos scalar) then
    // weights (the VRAM bulk). Both expose free_gpu so a load/unload cycle
    // returns their device buffers to the pool instead of leaking.
    if let Some(s) = m.minimax_state {
        s.free_gpu(gpu);
    }
    if let Some(w) = m.minimax_weights {
        w.free_gpu(gpu);
    }
    // LFM2.5-MoE (arch_id=11): state (KV + conv ring + scratch + pos scalar)
    // then weights. Same explicit-free contract as minimax.
    #[cfg(feature = "arch-lfm2moe")]
    if let Some(s) = m.active.lfm2moe_state {
        s.free_gpu(gpu);
    }
    #[cfg(feature = "arch-lfm2moe")]
    if let Some(w) = m.lfm2moe_weights {
        w.free_gpu(gpu);
    }
    // Weights are the bulk of VRAM (~80%). Free them too so idle eviction
    // actually returns VRAM to the system, not just the cache.
    if let Some(w) = m.q35_weights {
        w.free_gpu(gpu);
    }
    if let Some(w) = m.llama_weights {
        w.free_gpu(gpu);
    }
    if let Some(w) = m.qwen2_weights {
        w.free_gpu(gpu);
    }
    if let Some(w) = m.vision_weights {
        w.free_gpu(gpu);
    }
    // Gemma3-VL (arch_id=13): the backend owns the text/vision/projector weights
    // and its decode state — free both (mirrors Gemma3VlBackend::unload).
    if let Some(b) = m.gemma3_vl {
        b.weights.free_gpu(gpu);
        b.state.free_gpu(gpu);
    }
    // Gemma3 text (arch_id=12): backend owns the decoder weights + decode state.
    if let Some(b) = m.gemma3_text {
        b.weights.free_gpu(gpu);
        b.state.free_gpu(gpu);
    }
    // embeddinggemma (arch_id=19): owns a Gemma3 backbone plus host Dense heads.
    if let Some(e) = m.embeddinggemma {
        e.weights.free_gpu(gpu);
    }
    if let Some(w) = m.deepseek4_weights {
        w.free_gpu(gpu);
    }
    // Assembled ServingBackend wrappers (arch_id 0/1 llama, 7 qwen2, 14/15
    // nemotron, 16 zaya) own their GPU weights + decode state internally and
    // have no Drop. Unlike the loose-slot paths above, dropping the LoadedModel
    // does NOT return their device buffers to the pool — so without an explicit
    // unload they leak across a load/unload cycle. For a bf16 reference (~34 GB)
    // that OOMs the very next load (the multi-model quality-battery failure).
    // Each backend's `ServingBackend::unload` frees exactly what it allocated;
    // the loose `*_weights`/`*_state` frees above are no-ops on these paths
    // (those fields stay None when the backend wrapper is populated).
    {
        use hipfire_runtime::arch::ServingBackend;
        if let Some(b) = m.zaya_backend {
            Box::new(b).unload(gpu);
        }
        if let Some(b) = m.nemotron_backend {
            Box::new(b).unload(gpu);
        }
        if let Some(loaded) = m.registered_backend {
            loaded.backend.unload(gpu);
        }
        if let Some(b) = m.llama_backend {
            Box::new(b).unload(gpu);
        }
    }
    // Drop pointer-keyed caches whose keys point at weight buffers that are
    // about to be returned to the pool. Without this, the next model loaded
    // can land at the same device address and silently inherit stale
    // verdicts (mmq_screen_cache) or leaked FP16 shadows (fp16_shadow_cache).
    gpu.invalidate_weight_caches();
    // Tear down any captured hipGraphs (single-slot AR forward graph plus
    // DFlash verify and replay graph caches). These bake KV-cache, scratch,
    // and draft-weight pointers into kernarg memory at capture time; the
    // tensors backing those pointers are freed above, so replaying after
    // a model swap would dispatch against dangling or wrong-content
    // memory.
    gpu.invalidate_graph_state();
    gpu.drain_pool();
}

/// Load the experimental LFM2 DFlash drafter. Unlike Qwen DFlash, this path
/// does not allocate DeltaNet/ring-buffer/DDTree state; LFM2 verifies through
/// its arch-local batched prefill and snapshots KV + short-conv target state.
#[cfg(feature = "arch-lfm2moe")]
pub fn load_lfm2_dflash_state(
    draft_path: &str,
    ctx_capacity: usize,
    target_config: &lfm2moe::config::Lfm2MoeConfig,
    target_state: &lfm2moe::lfm2moe::Lfm2MoeState,
    gpu: &mut hipfire_rdna::Gpu,
) -> Result<Lfm2DflashState, String> {
    let hfq = HfqFile::open(Path::new(draft_path)).map_err(|e| format!("open LFM2 draft: {e}"))?;
    load_lfm2_dflash_state_source(&hfq, ctx_capacity, target_config, target_state, gpu)
}

#[cfg(feature = "arch-lfm2moe")]
fn load_lfm2_dflash_state_source(
    source: &(impl DflashSource + ?Sized),
    ctx_capacity: usize,
    target_config: &lfm2moe::config::Lfm2MoeConfig,
    target_state: &lfm2moe::lfm2moe::Lfm2MoeState,
    gpu: &mut hipfire_rdna::Gpu,
) -> Result<Lfm2DflashState, String> {
    let draft_config = DflashConfig::from_source(source).ok_or("parse LFM2 DflashConfig")?;
    lfm2moe::validate_dflash_contract(target_config, &draft_config)
        .map_err(|e| format!("LFM2 DFlash draft contract: {e}"))?;
    let use_f16_weights = lfm2moe::lfm2_dflash_use_f16_weights();
    let draft_weights =
        DflashWeights::load_source_with_f16(gpu, source, &draft_config, use_f16_weights)
            .map_err(|e| format!("load LFM2 draft weights: {e}"))?;
    let sync_gemm = lfm2moe::lfm2_dflash_sync_gemm();
    let draft_scratch = DflashScratch::new_with_mq_and_sync(
        gpu,
        &draft_config,
        draft_config.block_size,
        ctx_capacity,
        draft_weights.has_mq,
        sync_gemm,
    )
    .map_err(|e| format!("LFM2 draft scratch: {e}"))?;
    let target_snap =
        lfm2moe::Lfm2DflashTargetSnapshot::new_for(gpu, target_state, draft_config.block_size)
            .map_err(|e| format!("LFM2 target snapshot: {e:?}"))?;
    let target_hidden_host: Vec<f32> =
        Vec::with_capacity(ctx_capacity * draft_config.num_extract() * draft_config.hidden);
    let block_size = draft_config.block_size;
    tracing::info!(
        "LFM2 DFlash draft loaded: block={} extract_layers={:?} hidden={} ctx_capacity={} f16_weights={} sync_gemm={}",
        block_size, draft_config.target_layer_ids, draft_config.hidden, ctx_capacity, use_f16_weights, sync_gemm
    );

    Ok(Lfm2DflashState {
        draft_config,
        draft_weights,
        draft_scratch,
        target_snap,
        target_hidden_host,
        ctx_capacity,
        block_size,
    })
}

/// DSpark analog of the DFlash `draft_path.is_some()` gate: discover a
/// `<stem>-<quant>.dspark.hfq` sidecar next to a dense LLaMA/Qwen3 target
/// (arch 0/1) and, when present, load it + build the greedy speculator. Returns
/// `None` for any non-0/1 arch, when `HIPFIRE_DSPARK=0`, when no sidecar exists,
/// or when the sidecar fails to load (logged, non-fatal — the model still loads
/// AR-only). Keeps non-DSpark loads byte-identical.
fn maybe_load_dspark(
    backend: &mut hipfire_arch_llama::LlamaBackend,
    arch_id: u32,
    target_path: &str,
    ctx_capacity: usize,
    gpu: &mut hipfire_rdna::Gpu,
) -> Option<DsparkState> {
    if !matches!(arch_id, 0 | 1) {
        return None;
    }
    if std::env::var("HIPFIRE_DSPARK").ok().as_deref() == Some("0") {
        return None;
    }
    let sidecar = hipfire_model::discover_dspark_draft_for_model(Path::new(target_path))?;
    let sidecar = sidecar.to_string_lossy().to_string();
    tracing::info!("llama: DSpark sidecar discovered: {sidecar}");
    match load_dspark_state(&sidecar, backend, ctx_capacity, gpu) {
        Ok(state) => {
            tracing::info!(
                "llama DSpark speculator enabled (sidecar, block={})",
                state.speculator.block_size()
            );
            Some(state)
        }
        Err(e) => {
            tracing::warn!("llama: DSpark sidecar load failed: {e}");
            None
        }
    }
}

/// Load a DSpark drafter sidecar and build the arch-generic greedy speculator.
///
/// Mirrors [`load_dflash_state`] for the dense LLaMA/Qwen3 arch: opens the
/// sidecar HFQ, loads the Qwen3 drafter body + DSpark globals via
/// `hipfire_arch_llama::dspark_body::load_qwen3_dspark`, arms the target's
/// extract-layer capture (`set_dflash_extract_layers`), then builds the
/// `Box<dyn Speculator>` (`build_qwen3_dspark_body` + `build_dspark_speculator`).
///
/// `stage_norm` / `lm_head` are NON-OWNING aliases of the drafter's
/// `output_norm` / `output` tensors: the speculator body owns the primaries and
/// frees them in `Speculator::free`; `DsparkDrafter::mtp_free` never frees these
/// aliases, so there is no double-free (this matches the source carrier's
/// `shallow_clone` contract, expressed here via `GpuTensor::sub_offset`).
pub fn load_dspark_state(
    sidecar_path: &str,
    backend: &mut hipfire_arch_llama::LlamaBackend,
    ctx_capacity: usize,
    gpu: &mut hipfire_rdna::Gpu,
) -> Result<DsparkState, String> {
    let mut hfq =
        HfqFile::open(Path::new(sidecar_path)).map_err(|e| format!("open dspark sidecar: {e}"))?;
    // The loader reads via the pread-backed `tensor_data_vec`; drop the mmap
    // first (mirrors the load-smoke example + the source carrier).
    hfq.drop_mmap();
    let (dspark_weights, dspark_assets) =
        hipfire_arch_llama::dspark_body::load_qwen3_dspark(&hfq, gpu)?
            .ok_or("dspark sidecar has no dspark_* metadata")?;

    // Stash on the target then take back to build — the LlamaBackend fields are
    // the canonical home for the discovered sidecar (mirrors the source
    // carrier's `bundle.dspark_weights = Some(..)` / `.take()` two-phase shape),
    // and arm the target's extract-layer capture with the drafter's
    // `target_layer_ids`.
    let target_layers = dspark_weights.cfg.target_layer_ids.clone();
    backend.set_dflash_extract_layers(target_layers);
    backend.dspark_weights = Some(dspark_weights);
    backend.dspark_assets = Some(dspark_assets);
    let dspark_weights = backend.dspark_weights.take().unwrap();
    let dspark_assets = backend.dspark_assets.take().unwrap();

    let block = dspark_weights.cfg.block_size;
    // conf_threshold ladder: env > 0.1 default (source's sweep-tuned default).
    let conf_threshold = std::env::var("HIPFIRE_QWEN3_DSPARK_CONF_THRESHOLD")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.1f32);
    let vocab = dspark_assets.config.vocab_size;

    // Non-owning aliases (body owns/frees the primaries). `stage_norm` keeps the
    // drafter final-norm dtype/shape; `lm_head` is re-tagged F16/[vocab] because
    // `output.buf` was uploaded raw but its layout is F16 and `run_heads`
    // dispatches on `GpuTensor.dtype` + reads `lm_head.shape[0]` as the vocab.
    let stage_norm = {
        let n = &dspark_assets.weights.output_norm;
        n.sub_offset(0, n.numel())
    };
    let mut lm_head = {
        let o = &dspark_assets.weights.output.buf;
        o.sub_offset(0, o.numel())
    };
    lm_head.dtype = hipfire_rdna::DType::F16;
    lm_head.shape = vec![vocab];

    let body = hipfire_arch_llama::dspark_body::build_qwen3_dspark_body(
        dspark_assets,
        &dspark_weights.cfg,
        gpu,
    )
    .map_err(|e| format!("build dspark body: {e}"))?;

    // Greedy-only MVP: `supports_temp=false` ⇒ the speculator advertises
    // `requires_greedy()`, and `generate_llama` only drives it when temp≈0.
    let speculator = hipfire_specdecode_dspark::dspark_core::build_dspark_speculator(
        body,
        dspark_weights,
        stage_norm,
        lm_head,
        block,
        ctx_capacity,
        conf_threshold,
        false,
    );
    Ok(DsparkState { speculator })
}

/// Load the optional DFlash speculative-decoding drafter for a model: the draft
/// weights/config/scratch, the hidden-state ring buffer + verify scratch, and
/// (when `HIPFIRE_DDTREE_BUDGET` is set) the DDTree tree-verify side state.
/// Build the target-side speculative state for n-gram decode with NO drafter.
///
/// Every field here comes from the TARGET — verify scratch, the DeltaNet
/// snapshot and the GDN tape are what the verify step needs regardless of where
/// the draft came from. The four drafter fields are None, which makes
/// `spec_step_dflash` skip the drafter branch, the per-layer hidden extraction
/// on every verify, and the staging that feeds the next drafter cycle.
///
/// `max_spine` sizes the verify buffers the way `draft_cfg.block_size` does for
/// DFlash: one slot for the seed plus the longest spine n-gram can supply.
pub fn ngram_only_state(
    max_spine: usize,
    ctx_capacity: usize,
    target_config: &qwen35::Qwen35Config,
    target_dn: &DeltaNetState,
    lm_head_k: usize,
    gpu: &mut hipfire_rdna::Gpu,
) -> Result<DflashState, String> {
    let scratch_max_n = max_spine.max(1) + 1;
    let target_snap =
        DeltaNetSnapshot::new_for(gpu, target_dn).map_err(|e| format!("target_snap: {e}"))?;
    let gdn_tape = GdnTape::new_for_config(gpu, target_config, scratch_max_n)
        .map_err(|e| format!("gdn_tape: {e}"))?;
    let verify_scratch = VerifyScratch::with_prefill(
        gpu,
        scratch_max_n,
        target_config.dim,
        target_config.vocab_size,
        lm_head_k,
        target_config,
    )
    .map_err(|e| format!("verify_scratch: {e}"))?;
    Ok(DflashState {
        draft_config: None,
        draft_weights: None,
        draft_scratch: None,
        hidden_rb: None,
        verify_scratch,
        target_snap,
        gdn_tape,
        // Only the drafter reads the host hidden shadow.
        target_hidden_host: Vec::new(),
        ctx_capacity,
        block_size: max_spine.max(1),
        // ON here, unlike the drafter path below, and the asymmetry is the
        // point. There the controller is off because `max_block` is clamped to
        // the TRAINED block, which already is the in-range optimum — no upside
        // to find, and the ramp costs ~25% decode. Here there is no trained
        // block: `max_spine` is a config ceiling nobody measured, so the search
        // space genuinely contains a win. Measured on gfx1103/qwen3.5-0.8b,
        // fixed widths either side of that ceiling: b=16 gave 397/557 tok/s and
        // b=17 gave 328/397, so the optimum is strictly BELOW the ceiling and
        // the ceiling is not it.
        //
        // This used to read "Adaptive-B tunes a DRAFTER's block; the spine
        // length is n-gram's", which conflates two things: spine length is the
        // n-gram's SUPPLY, but `b` is how many of those the target verifies per
        // forward — a cost/benefit trade, which is exactly what the controller
        // models, and it is drafter-agnostic by construction (it consumes only
        // accept depth, proposal count and wall time).
        // Fallback only; the daemon overwrites this from `spec_adaptive_block`.
        adaptive_b: false,
        spec_block: None,
        // DDTree is a tree over a drafter's candidates.
        ddtree: None,
    })
}

pub fn load_dflash_state(
    draft_path: &str,
    ctx_capacity: usize,
    target_config: &qwen35::Qwen35Config,
    target_dn: &DeltaNetState,
    lm_head_k: usize,
    gpu: &mut hipfire_rdna::Gpu,
) -> Result<DflashState, String> {
    let hfq = HfqFile::open(Path::new(draft_path)).map_err(|e| format!("open draft: {e}"))?;
    load_dflash_state_source(&hfq, ctx_capacity, target_config, target_dn, lm_head_k, gpu)
}

fn load_dflash_state_source(
    source: &(impl DflashSource + ?Sized),
    ctx_capacity: usize,
    target_config: &qwen35::Qwen35Config,
    target_dn: &DeltaNetState,
    // The target lm_head's inner dimension (`weights.output.k`). A compact
    // Opus head can pad this past `target_config.dim`; sizing VerifyScratch
    // from `dim` then trips the "verify_scratch undersized for compact Opus
    // lm_head" assert in speculative.rs.
    lm_head_k: usize,
    gpu: &mut hipfire_rdna::Gpu,
) -> Result<DflashState, String> {
    let draft_config = DflashConfig::from_source(source).ok_or("parse DflashConfig")?;
    let weights_started = std::time::Instant::now();
    let draft_weights = DflashWeights::load_source(gpu, source, &draft_config)
        .map_err(|e| format!("load weights: {e}"))?;
    tracing::info!(
        "DFlash weight upload: {:.2}s",
        weights_started.elapsed().as_secs_f64()
    );
    let draft_scratch = DflashScratch::new_with_mq(
        gpu,
        &draft_config,
        draft_config.block_size,
        ctx_capacity,
        draft_weights.has_mq,
    )
    .map_err(|e| format!("draft scratch: {e}"))?;

    // Hidden ring: one row per target-layer selected by the draft config,
    // captured during each target forward. Sized so the whole context plus
    // one block fits without aliasing. Cheap (< 100 MB) next to the draft
    // weights themselves.
    //
    // Use the layer ids the DRAFTER DECLARES, not a re-derived spread. They
    // are what it was trained against; the derived spread only coincides with
    // the declared one for the shipped 40-layer drafter, and a mismatch feeds
    // the drafter hidden states from layers it never saw (measured on
    // Qwen3.8-27B/gfx1103 as tau 5.9 -> 2.24; see
    // docs/plans/2026-08-27-dflash-demo-merge-scope.md). Mirrors
    // dflash_spec_demo and the LFM2/dspark loaders, which already use the
    // declared ids.
    let hidden_rb = if draft_config.target_layer_ids.is_empty() {
        HiddenStateRingBuffer::new(
            gpu,
            target_config.n_layers,
            draft_config.num_extract(),
            draft_config.hidden,
            ctx_capacity + draft_config.block_size,
            hipfire_arch_qwen35::qwen35::PREFILL_MAX_BATCH.max(draft_config.block_size),
        )
    } else {
        HiddenStateRingBuffer::new_for_layers(
            gpu,
            target_config.n_layers,
            draft_config.target_layer_ids.clone(),
            draft_config.hidden,
            ctx_capacity + draft_config.block_size,
            hipfire_arch_qwen35::qwen35::PREFILL_MAX_BATCH.max(draft_config.block_size),
        )
    }
    .map_err(|e| format!("hidden_rb: {e}"))?;
    debug_assert!(
        draft_config.target_layer_ids.is_empty()
            || hidden_rb.extract_layers == draft_config.target_layer_ids
    );

    let target_snap =
        DeltaNetSnapshot::new_for(gpu, target_dn).map_err(|e| format!("target_snap: {e}"))?;

    // Read DDTree budget env-var BEFORE sizing GdnTape / VerifyScratch.
    // When DDTree is enabled, both must be sized for `1 + budget` nodes
    // per cycle (the linearized tree includes one root slot plus all
    // tree nodes), not just `block_size`. Reading the env-var here keeps
    // a single source of truth and avoids re-allocating these scratches
    // after the model is on GPU.
    //
    // DdtreeScratch::attn_bias is sized `max_n²` (max_n = 1 + budget),
    // so the allocation is quadratic in budget. The paper's Algorithm 1
    // typically uses budget ≤ 22; we cap at 256 to leave huge headroom
    // while killing the OOM cliff from a typo'd budget value (`=10000`
    // would request 400 MB just for attn_bias; `=100000` would OOM most
    // GPUs). Invalid / out-of-range values warn loudly and disable
    // DDTree rather than silently falling through.
    const DDTREE_BUDGET_MAX: usize = 256;
    let ddtree_budget_env: usize = match std::env::var("HIPFIRE_DDTREE_BUDGET").ok() {
        None => 0,
        Some(s) if s.is_empty() => 0,
        Some(s) => match s.parse::<usize>() {
            Ok(0) => 0,
            Ok(n) if n <= DDTREE_BUDGET_MAX => n,
            Ok(n) => {
                tracing::warn!(
                    "HIPFIRE_DDTREE_BUDGET={} exceeds cap {DDTREE_BUDGET_MAX} \
                     (attn_bias is O(budget²); typical values are 12-22). Disabling DDTree.",
                    n
                );
                0
            }
            Err(_) => {
                tracing::warn!(
                    "HIPFIRE_DDTREE_BUDGET={:?} is not a non-negative integer. \
                     Disabling DDTree.",
                    s
                );
                0
            }
        },
    };
    let scratch_max_n = if ddtree_budget_env > 0 {
        std::cmp::max(draft_config.block_size, 1 + ddtree_budget_env)
    } else {
        draft_config.block_size
    };

    let gdn_tape = GdnTape::new_for_config(gpu, target_config, scratch_max_n)
        .map_err(|e| format!("gdn_tape: {e}"))?;
    let verify_scratch = VerifyScratch::with_prefill(
        gpu,
        scratch_max_n,
        target_config.dim,
        target_config.vocab_size,
        lm_head_k,
        target_config,
    )
    .map_err(|e| format!("verify_scratch: {e}"))?;

    let target_hidden_host: Vec<f32> =
        Vec::with_capacity(ctx_capacity * draft_config.num_extract() * draft_config.hidden);
    let block_size = draft_config.block_size;

    // Optional DDTree allocation. `HIPFIRE_DDTREE_BUDGET=<n>` (positive
    // integer) wires the decode loop to `spec_step_ddtree_batched` instead
    // of `spec_step_dflash`. `HIPFIRE_DDTREE_TOPK=<k>` controls the
    // per-position top-K (default 4). Anything else, or budget=0, leaves
    // the existing DFlash chain-mode path untouched.
    let ddtree = match Some(ddtree_budget_env).filter(|&n| n > 0) {
        Some(budget) => {
            // topk caps the per-position branching factor in the tree
            // builder. Algorithm 1's typical setting is 4; the active
            // kernel `run_dflash_draft_for_topk_gpu` (called by both
            // `spec_step_ddtree_batched` and `spec_step_ddtree_path_c`)
            // asserts `k >= 1 && k <= 8` at speculative.rs:3302 and panics
            // outside that range. Take the kernel's bound as authoritative;
            // anything looser would let env-var values pass daemon
            // validation but blow up at the first decode cycle.
            //
            // Two upper bounds:
            //   - DDTREE_TOPK_KERNEL_MAX = 8 — kernel's hardcoded assert.
            //   - vocab_size — extra correctness cap for tiny-vocab /
            //     character-level targets where vocab can be < 8.
            //
            // Effective cap = min(8, vocab_size). Default = min(4, vocab_size).
            const DDTREE_TOPK_KERNEL_MAX: usize = 8;
            let vocab = target_config.vocab_size;
            let effective_topk_max = std::cmp::min(DDTREE_TOPK_KERNEL_MAX, vocab);
            let default_topk = std::cmp::min(4usize, vocab.max(1));
            let topk = match std::env::var("HIPFIRE_DDTREE_TOPK").ok() {
                None => default_topk,
                Some(s) if s.is_empty() => default_topk,
                Some(s) => match s.parse::<usize>() {
                    Ok(k) if k >= 1 && k <= effective_topk_max => k,
                    Ok(k) => {
                        tracing::warn!(
                            "HIPFIRE_DDTREE_TOPK={k} out of range [1, {effective_topk_max}] \
                             (vocab_size={vocab}). Falling back to default topk={default_topk}."
                        );
                        default_topk
                    }
                    Err(_) => {
                        tracing::warn!(
                            "HIPFIRE_DDTREE_TOPK={:?} is not a positive integer. \
                             Falling back to default topk={default_topk}.",
                            s
                        );
                        default_topk
                    }
                },
            };
            let post_seed_snap = DeltaNetSnapshot::new_for(gpu, target_dn)
                .map_err(|e| format!("ddtree post_seed_snap: {e}"))?;
            let path_c_parent_pre_snap = DeltaNetSnapshot::new_for(gpu, target_dn)
                .map_err(|e| format!("ddtree path_c_parent_pre_snap: {e}"))?;
            let path_c_main_end_snap = DeltaNetSnapshot::new_for(gpu, target_dn)
                .map_err(|e| format!("ddtree path_c_main_end_snap: {e}"))?;
            let n_fa_layers = target_config
                .layer_types
                .iter()
                .filter(|t| **t == LayerType::FullAttention)
                .count();
            // qkv_dim mirrors GdnTape::new_for_config: linear-attention
            // qkv row width (k_dim × 2 + v_dim).
            let k_dim = target_config.linear_num_key_heads * target_config.linear_key_head_dim;
            let v_dim = target_config.linear_num_value_heads * target_config.linear_value_head_dim;
            let qkv_dim = k_dim * 2 + v_dim;
            let scratch = DdtreeScratch::new(
                gpu,
                budget,
                target_config.n_kv_heads,
                target_config.head_dim,
                qkv_dim,
                n_fa_layers,
            )
            .map_err(|e| format!("ddtree scratch: {e}"))?;
            tracing::info!(
                "DDTree enabled: budget={budget}, topk={topk}, n_fa_layers={n_fa_layers}"
            );
            Some(DdtreeState {
                post_seed_snap,
                scratch,
                budget,
                topk,
                path_c_parent_pre_snap,
                path_c_main_end_snap,
            })
        }
        None => None,
    };

    Ok(DflashState {
        draft_config: Some(draft_config),
        draft_weights: Some(draft_weights),
        draft_scratch: Some(draft_scratch),
        hidden_rb: Some(hidden_rb),
        verify_scratch,
        target_snap,
        gdn_tape,
        target_hidden_host,
        ctx_capacity,
        block_size,
        // Fallback only; the daemon overwrites both from `spec_adaptive_block`
        // and `spec_block` after the load. Measured on gfx1103/Qwen3.8-27B: with
        // max_block clamped to the trained block the controller has no upside to
        // find (the trained block IS the in-range optimum) and the ramp costs
        // ~25% decode. Revisit once scratches are sized for B above trained
        // (scope doc item 4) and the search space contains a win.
        adaptive_b: false,
        spec_block: None,
        ddtree,
    })
}

#[cfg(test)]
mod admission_tests {
    use super::*;

    /// Every KV mode TriAttention/CASK eviction can compact is ALSO a
    /// deprecated mode. That is not a coincidence to be discovered later: it
    /// means eviction is reachable only behind HIPFIRE_KV_ALLOW_DEPRECATED=1,
    /// and that deleting the deprecated KV constructors deletes eviction's last
    /// live caller with them. Pinned here so the coupling surfaces as a failing
    /// test rather than as dead code nobody noticed.
    #[test]
    fn every_evictable_kv_mode_is_deprecated() {
        let evictable = ["q8", "asym2", "asym3", "asym4", "fwht2", "fwht3", "fwht4"];
        for m in evictable {
            assert!(
                triattn_can_evict_kv_mode(m),
                "{m} should be evictable — keep this list in sync with \
                 triattn_can_evict_kv_mode"
            );
            assert!(
                hipfire_config::DEPRECATED_KV_MODES.contains(&m),
                "{m} is evictable but no longer deprecated: TriAttention eviction \
                 now has a supported caller, so revisit the load-path filter that \
                 declines the sidecar"
            );
        }
        // The two supported families reach no eviction arm.
        for m in ["fp32", "kvarn", "kvarn2", "kvarn4", "kvarn8"] {
            assert!(
                !triattn_can_evict_kv_mode(m),
                "{m} must decline the sidecar"
            );
        }
    }

    #[test]
    fn component_resolution_precedence_and_off_are_stable() {
        let sibling = PathBuf::from("sibling.hfq");
        assert_eq!(
            resolve_component_source(Some("explicit.hfq"), true, Some(sibling.clone()), false),
            Some(ResolvedComponentSource::Explicit("explicit.hfq"))
        );
        assert_eq!(
            resolve_component_source(None, true, Some(sibling.clone()), false),
            Some(ResolvedComponentSource::Embedded)
        );
        assert_eq!(
            resolve_component_source(None, false, Some(sibling.clone()), false),
            Some(ResolvedComponentSource::Sibling(sibling))
        );
        assert_eq!(
            resolve_component_source(Some("explicit.hfq"), true, None, true),
            None
        );
    }

    fn tensor_info(name: &str, quant_type: QuantType, rows: u32, cols: u32) -> HfqTensorInfo {
        let data_size = if quant_type == QuantType::Oq8G256RowPadded {
            quant_type
                .matrix_tensor_bytes(rows as usize, cols as usize)
                .unwrap()
        } else {
            rows as usize * cols.div_ceil(256) as usize * 258
        };
        HfqTensorInfo {
            name: name.to_string(),
            quant_type: quant_type.code(),
            shape: vec![rows, cols],
            group_size: 256,
            data_offset: 0,
            data_size,
        }
    }

    #[test]
    fn embeddinggemma_storage_contract_is_driven_by_tensor_layout() {
        let aligned = tensor_info("aligned", QuantType::Oq8G256, 2, 512);
        assert_eq!(
            embeddinggemma_storage_contract(&[aligned]).unwrap(),
            EmbeddingGemmaStorageContract::default()
        );

        let explicit = tensor_info(
            "model.layers.0.mlp.down_proj.weight",
            QuantType::Oq8G256RowPadded,
            768,
            1152,
        );
        let contract = embeddinggemma_storage_contract(&[explicit]).unwrap();
        assert!(contract.requires_npu());
        assert!(contract.explicit_row_padded_oq8);
        assert!(!contract.legacy_implicit);

        let legacy = tensor_info(
            "model.layers.0.mlp.down_proj.weight",
            QuantType::Oq8G256,
            768,
            1152,
        );
        let contract = embeddinggemma_storage_contract(&[legacy]).unwrap();
        assert!(contract.requires_npu());
        assert!(contract.legacy_implicit);
    }

    #[test]
    fn embeddinggemma_storage_contract_rejects_invalid_explicit_geometry() {
        let aligned = tensor_info("aligned", QuantType::Oq8G256RowPadded, 2, 512);
        assert!(embeddinggemma_storage_contract(&[aligned])
            .unwrap_err()
            .contains("aligned K"));

        let mut truncated = tensor_info("truncated", QuantType::Oq8G256RowPadded, 2, 384);
        truncated.data_size -= 1;
        assert!(embeddinggemma_storage_contract(&[truncated])
            .unwrap_err()
            .contains("expected"));
    }

    /// The DFlash load gate is driven by the generated capability matrix, not a
    /// hard-coded arch list: qwen3.5 (5/6) is admitted; archs whose matrix marks
    /// dflash != full are refused with an operator-facing, matrix-derived reason.
    /// Guards the silent-no-op gap (draft attached to a non-DFlash arch → load
    /// succeeds, then generate() falls through to plain AR).
    #[test]
    fn dflash_admission_is_matrix_backed() {
        for arch in [5u32, 6] {
            assert!(
                require_arch_feature(arch, "DFlash spec-decode", arch_features(arch).dflash)
                    .is_ok(),
                "qwen3.5 arch {arch} must admit DFlash"
            );
        }
        // llama (0) and gemma3 (12) have dflash=none → refused.
        let e = require_arch_feature(0, "DFlash spec-decode", arch_features(0).dflash)
            .expect_err("llama must be refused");
        assert!(
            e.contains("llama") && e.contains("does not support"),
            "msg: {e}"
        );
        assert!(require_arch_feature(12, "DFlash spec-decode", arch_features(12).dflash).is_err());
    }

    #[cfg(feature = "arch-lfm2moe")]
    #[test]
    fn lfm2_triattn_sidecars_use_attention_ordinal_indices() {
        let config = lfm2moe::config::Lfm2MoeConfig::from_config_value(&serde_json::json!({
            "vocab_size": 256,
            "hidden_size": 64,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "intermediate_size": 128,
            "layer_types": ["conv", "full_attention", "conv", "full_attention"]
        }))
        .unwrap();

        assert_eq!(config.num_attention_layers(), 2);
        assert_eq!(lfm2_triattn_kv_layer_ids(&config), vec![0, 1]);
    }
}
