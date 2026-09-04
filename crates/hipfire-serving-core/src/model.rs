// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The daemon's in-memory model representation and its satellites.
//!
//! `LoadedModel` is the per-loaded-model state the request loop drives — an
//! arch-tagged bundle of the typed config/weights/state for whichever family is
//! resident (the Option-soup the E2 `ServingBackend` seam will eventually
//! collapse). Extracted verbatim from the former `main.rs` monolith (no behavior
//! change); fields are `pub` so the load/generate/memory/session modules
//! that still live in `main.rs` (and sibling modules) can construct and read it.

use hip_bridge::HipResult;
use hipfire_arch_deepseek4 as deepseek4;
use hipfire_arch_dots_ocr::dots_ocr;
use hipfire_arch_embeddinggemma as embeddinggemma;
use hipfire_arch_gemma3::Gemma3Backend;
use hipfire_arch_gemma3_vl::Gemma3VlBackend;
#[cfg(feature = "arch-lfm2moe")]
use hipfire_arch_lfm2moe as lfm2moe;
use hipfire_arch_minimax as minimax;
use hipfire_arch_qwen2::qwen2;
use hipfire_arch_qwen35::qwen35;
use hipfire_arch_qwen35::qwen35::{DeltaNetState, Qwen35ScratchSet};
use hipfire_arch_qwen35::speculative::{
    DdtreeScratch, DeltaNetSnapshot, GdnTape, HiddenStateRingBuffer, VerifyScratch,
};
use hipfire_arch_qwen35_vl::qwen35_vl;
use hipfire_prompt as prompt_frame;
use hipfire_runtime::arch::FactoryLoadedBackend;
use hipfire_runtime::cask::CaskCtx;
use hipfire_runtime::dflash::{DflashConfig, DflashScratch, DflashWeights};
use hipfire_runtime::kv;
use hipfire_runtime::llama;
use hipfire_runtime::multi_gpu::Gpus;
use hipfire_runtime::sequence_state::SequenceState;
use hipfire_runtime::triattn::EvictionCtx;
use hipfire_state::ModelArtifactMemory;

use crate::qwen3_embedding::Qwen3EmbeddingState;
#[cfg(feature = "arch-lfm2moe")]
use crate::session::Lfm2RequestSessionState;
use crate::session::Qwen35RequestSessionState;
use crate::session::SessionCursor;
use crate::session::SessionRegistry;

/// CASK/TriAttention params forwarded by the CLI at load time. Zero-initialized
/// CaskConfig{sidecar: None, ..} means no eviction — matches 0.1.7-alpha behavior.
#[derive(Default)]
pub struct CaskConfig {
    pub sidecar: Option<String>,
    /// true = CASK m-folding; false = plain TriAttention drop-eviction.
    pub cask_m_folding: bool,
    pub budget: usize,
    pub beta: usize,
    pub core_frac: f32,
    pub fold_m: usize,
}

/// Eviction policy wrapper — dispatches to plain TriAttention or CASK m-folding.
pub enum Eviction {
    Plain(EvictionCtx),
    Cask(CaskCtx),
}

impl Eviction {
    /// Run one eviction pass when the physical KV occupancy warrants it,
    /// dispatching to the active policy; `Ok(None)` when nothing was evicted.
    pub fn maybe_evict(
        &self,
        gpu: &mut hipfire_rdna::Gpu,
        kv: &mut kv::KvCache,
        physical: usize,
    ) -> HipResult<Option<hipfire_runtime::triattn::EvictionResult>> {
        match self {
            Eviction::Plain(c) => c.maybe_evict(gpu, kv, physical),
            Eviction::Cask(c) => c.maybe_evict(gpu, kv, physical),
        }
    }
    /// Protected-core budget (slots kept before eviction begins).
    pub fn budget(&self) -> usize {
        match self {
            Eviction::Plain(c) => c.budget,
            Eviction::Cask(c) => c.base.budget,
        }
    }
    /// Recency window (`beta`) preserved alongside the budget.
    pub fn beta(&self) -> usize {
        match self {
            Eviction::Plain(c) => c.beta,
            Eviction::Cask(c) => c.base.beta,
        }
    }
    /// Release the policy's GPU-side scratch on unload.
    pub fn free_gpu(self, gpu: &mut hipfire_rdna::Gpu) {
        match self {
            Eviction::Plain(c) => c.free_gpu(gpu),
            Eviction::Cask(c) => c.free_gpu(gpu),
        }
    }
}

/// Optional DFlash speculative-decoding state. Populated when `load` supplies
/// a matching draft (.hfq arch=20) via `params.draft`. Used by the daemon's
/// `generate` fast path when temperature == 0 — falls back to AR sampling
/// otherwise (DFlash is greedy-only in this integration).
pub struct DflashState {
    // The DRAFTER. `None` when this state exists only to carry n-gram
    // speculative decode, which drafts from statistics and needs no drafter
    // model: `spec_step_dflash` takes all four as Option and skips the drafter
    // branch, the hidden-state extraction and the staging that feeds it.
    pub draft_config: Option<DflashConfig>,
    pub draft_weights: Option<DflashWeights>,
    pub draft_scratch: Option<DflashScratch>,
    pub hidden_rb: Option<HiddenStateRingBuffer>,
    pub verify_scratch: VerifyScratch,
    pub target_snap: DeltaNetSnapshot,
    pub gdn_tape: GdnTape,
    /// CPU-side ring of target hidden states (num_extract × hidden per pos)
    /// — seeded from the prompt, extended by each verify's accepted rows.
    /// Drives the draft's diffusion forward.
    pub target_hidden_host: Vec<f32>,
    /// Max ctx the draft was initialized for (ring buffer cap).
    pub ctx_capacity: usize,
    /// Block size the draft was trained at.
    pub block_size: usize,
    /// Adaptive block sizing (cost-model argmax over [2, block_size]) in the
    /// chain-mode decode loop. Set from the `spec_adaptive_block` setting by the
    /// daemon after load; default OFF — see the load-site comments for the
    /// measurement on both the drafter and drafter-free paths. A `spec_block`
    /// pin overrides it.
    pub adaptive_b: bool,
    /// The adaptive block controller, when `adaptive_b` is on. Lives HERE, not
    /// in the decode loop: it was constructed per `generate` call, so its
    /// calibration -- documented in `reset()` as "a thermal-invariant hardware
    /// cost: calibrate once, reuse across requests" -- was thrown away every
    /// request and `reset()` was unreachable. Two generates produced two
    /// different fitted cost curves for the same hardware.
    pub block_controller:
        Option<hipfire_specdecode_dspark::dspark_block_controller::BlockController>,
    /// Fixed verify block, or `None` for auto. From the `spec_block` setting;
    /// was the raw `HIPFIRE_DFLASH_BLOCK` env read, which outranked config
    /// silently. Auto means the trained block with a drafter, and the spine's
    /// own length without one.
    pub spec_block: Option<usize>,
    /// Optional DDTree state. Populated only when `HIPFIRE_DDTREE_BUDGET` is
    /// set to a positive integer at daemon startup. None = DDTree disabled,
    /// the decode loop falls through to `spec_step_dflash` (chain mode).
    /// See `spec_step_ddtree_batched` for the tree-verify path.
    pub ddtree: Option<DdtreeState>,
}

/// Drafter-free n-gram speculative decode, owned by the MODEL.
///
/// It used to live on [`DflashState`], which was the wrong home twice over: the
/// feature needs no drafter, and a `DflashState` is the one place it could be
/// stored, so on a model without a drafter the setup was built and then dropped
/// on the floor. Hanging it off `LoadedModel` instead means any decode path can
/// reach it — today only the qwen35-shaped `generate()` body does, but that is
/// now a wiring gap rather than a type-level impossibility.
///
/// It does NOT make n-gram spec decode arch-generic on its own. Verification
/// still runs through `spec_step_dflash`, which wants DFlash's verify scratch,
/// snapshot and tape; making THAT generic needs a multi-token verify seam on
/// `ServingBackend`, which has none today.
pub struct NgramState {
    /// Static configuration, from the `ngram_spec*` settings at load.
    pub setup: NgramSetup,
    /// Live tables, kept across requests for the life of the load.
    ///
    /// This has to outlive a single request or the feature is decorative: the
    /// hot tier carries ~95% of the measured value and grams only reach disk
    /// after `promote_count` observations, so rebuilding per request resets
    /// every counter and almost nothing is ever promoted. The `String` is the
    /// scope key (user + session type); a request with a different key swaps
    /// the state out rather than reading another user's table.
    pub live: Option<(String, hipfire_specdecode_ngram::NgramSpec)>,
}

impl NgramState {
    pub fn new(setup: NgramSetup) -> Self {
        Self { setup, live: None }
    }

    /// Take the live tables if they belong to `key`, otherwise drop them.
    ///
    /// This is the privacy boundary of the feature, not a cache policy: the
    /// tables are built from one scope's decoded tokens, so handing them to a
    /// request with a different scope key would let one user's text draft
    /// another's. Returning `None` costs a cold start; returning the wrong
    /// state leaks. A dropped state is merged first so what it staged still
    /// reaches disk.
    pub fn take_live_for(&mut self, key: &str) -> Option<hipfire_specdecode_ngram::NgramSpec> {
        match self.live.take() {
            Some((k, prev)) if k == key => Some(prev),
            Some((_, mut prev)) => {
                let _ = prev.merge();
                None
            }
            None => None,
        }
    }
}

/// Merge the live tables when the load goes away.
///
/// `take_live_for` already merges on a scope swap; this is the other half, and
/// without it a single-user, single-topic daemon never merged at all — the
/// backlog was simply dropped with the `NgramSpec`, and every gram that had lost
/// an in-place insert was lost with it.
///
/// On `Drop` rather than in `unload_model` because `unload_model` is not the only
/// way a model goes away: it has seven call sites, a worker swap moves models in
/// and out of `resident_models`, and the next unload path added would have to
/// remember. A drop cannot be forgotten.
///
/// The merge also rebalances — `insert_in_place` writes reach disk without
/// passing through the backlog, so an empty backlog does not mean an unchanged
/// file — which is what leaves the store tidy for the next load. Cost is one
/// full-file rewrite, measured ~7 s/GiB against a 256 MiB default store; it is
/// the same cost the scope-swap path has always paid. A RAM-only store has no
/// write store and merges nothing.
impl Drop for NgramState {
    fn drop(&mut self) {
        if let Some((_, spec)) = self.live.as_mut() {
            if let Err(e) = spec.merge() {
                eprintln!("[ngram] merge on unload failed, staged grams not persisted: {e}");
            }
        }
    }
}

/// Per-request identity used to pick n-gram tables.
///
/// Threaded rather than read from a global, for the same reason `raw_override`
/// and `sampler_seed` are — and here the stakes are higher than a wrong
/// sampling seed: `next`/`next2` are stored in plaintext, so a request writing
/// into the wrong user's table would hand that user's text to whoever reads it
/// next.
#[derive(Debug, Clone, Copy, Default)]
pub struct NgramRequestScope<'a> {
    /// Owner of the writable table. `None` = daemon-local (single-tenant).
    pub user_id: Option<&'a str>,
    /// Subject label ("python-coding"). Selects a topic table *under this
    /// user*, so it stays private.
    pub session_type: Option<&'a str>,
}

/// Where a request's n-gram tables live, resolved at load time.
///
/// `scope` names the **tokenizer**, not the model: records are token ids, so
/// every quant variant of one base may share a table and two tokenizers never
/// can. It defaults to the model filename, which never wrongly shares.
#[derive(Debug, Clone)]
pub struct NgramSetup {
    /// Root directory for the tables, or a RAM-only sentinel.
    ///
    /// Empty, `ram`, `none` or `off` (any case) all mean *do not persist*: the
    /// hot tier still runs, nothing touches disk. The word forms exist because
    /// "leave this blank" is not expressible in a settings UI — an operator
    /// cannot tell an unset field from one deliberately set to RAM-only.
    pub store_root: std::path::PathBuf,
    pub scope: String,
    /// Blocks in a newly created store; the file is allocated in full and never
    /// grows, so this is the budget. 65536 blocks x 4 KiB = 256 MiB.
    pub blocks: usize,
    /// Probe orders, longest first.
    pub orders: Vec<u8>,
    /// Minimum winning order to keep extending a chain; 0 disables the gate.
    pub chain_floor: u8,
    pub max_spine: usize,
    /// Observations before a gram is persisted. Gates writes, never drafting.
    pub promote_count: u16,
    /// Which store the write path feeds.
    pub write_target: NgramWriteTarget,
    /// Confidence-bounded acceptance floor below which a gram stops drafting.
    /// 0 disables. See `ngram_spec_min_acceptance`.
    pub min_acceptance: f32,
    /// Proposals required before that floor may act.
    pub min_acceptance_proposals: u32,
}

impl NgramSetup {
    /// Whether tables should be written to disk at all.
    ///
    /// The RAM sentinels come from the `ngram_store_root` schema field
    /// (`NGRAM_STORE_ROOT_RAM`) rather than a list repeated here. They used to
    /// be written twice — once in that field's prose description and once as a
    /// `matches!` in this function — with nothing keeping them in step.
    pub fn persists(&self) -> bool {
        let raw = self.store_root.as_os_str().to_string_lossy();
        let t = raw.trim().to_ascii_lowercase();
        !hipfire_config::NGRAM_STORE_ROOT_RAM.contains(&t.as_str())
    }
}

/// Which n-gram store the write path feeds. Only a store private to its scope
/// may be written; a shared one is opened read-only and refuses writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NgramWriteTarget {
    User,
    Topic,
    None,
}

impl NgramWriteTarget {
    /// Parse an operator string; anything unrecognised falls back to `User`,
    /// the safe default (private by construction).
    pub fn parse(s: &str) -> Self {
        match s.trim().to_ascii_lowercase().as_str() {
            "topic" => Self::Topic,
            "none" | "off" => Self::None,
            _ => Self::User,
        }
    }
}

/// Optional DSpark speculative-decoding state for the dense LLaMA/Qwen3 arch
/// (arch_id 0/1). Populated when `load_model` discovers a `<stem>-<quant>.dspark.hfq`
/// sidecar next to the target and loads it via `load_dspark_state`.
///
/// The drafter+verifier is built once at load and stored behind the arch-generic
/// `Box<dyn Speculator>` seam (drafter arch id [`ARCH_ID_DSPARK_DRAFT`] = 22).
/// `generate_llama` drives it (greedy only for the MVP) when this is `Some`; the
/// speculator's GPU buffers are released in `unload_model` via `Speculator::free`.
pub struct DsparkState {
    pub speculator: Box<dyn hipfire_specdecode_dspark::spec::Speculator>,
}

/// Resident non-autoregressive embedding model state (arch_id=19).
pub struct EmbeddingGemmaState {
    pub config: embeddinggemma::EmbeddingGemmaConfig,
    pub embedding_metadata: Option<hipfire_model::embedding::EmbeddingMetadata>,
    pub weights: embeddinggemma::EmbeddingGemmaWeights,
    #[cfg(target_os = "linux")]
    pub npu_projector: Option<std::sync::Mutex<embeddinggemma::NpuOpusProjector>>,
}

/// Optional LFM2 DFlash speculative-decoding state. LFM2 has no DeltaNet
/// recurrent state, so it carries the generic DFlash draft plus an arch-local
/// target snapshot and host hidden-history rows.
#[cfg(feature = "arch-lfm2moe")]
pub struct Lfm2DflashState {
    pub draft_config: DflashConfig,
    pub draft_weights: DflashWeights,
    pub draft_scratch: DflashScratch,
    pub target_snap: lfm2moe::Lfm2DflashTargetSnapshot,
    pub target_hidden_host: Vec<f32>,
    pub ctx_capacity: usize,
    pub block_size: usize,
}

/// Side state for DDTree-mode speculative decoding. Allocated alongside
/// the rest of `DflashState` at model-load time when DDTree is enabled,
/// reused across all decode cycles.
pub struct DdtreeState {
    /// Second DeltaNetSnapshot used by `spec_step_ddtree_batched`: snap0 =
    /// pre-seed (lives in `DflashState::target_snap`), snap1 = post-seed.
    /// The batched verify forward uses both to bracket the tree-verify pass.
    pub post_seed_snap: DeltaNetSnapshot,
    /// Persistent tree-verify scratch (attn_bias, parent_indices, kv-gather
    /// staging, pre-RoPE K capture). Sized for `budget` non-root nodes.
    pub scratch: DdtreeScratch,
    /// Maximum non-root tree nodes per cycle. Read once at daemon startup
    /// from `HIPFIRE_DDTREE_BUDGET` (positive integer required to enable).
    pub budget: usize,
    /// Per-position top-K width fed into the DDTree builder. Read from
    /// `HIPFIRE_DDTREE_TOPK` (default 4 — matches paper Algorithm 1's
    /// typical setting on dense Qwen targets).
    pub topk: usize,
    /// Path C Phase 2 auxiliary snapshots. Used only when
    /// `HIPFIRE_DDTREE_PATH_C=phase2`. Allocated unconditionally when DDTree
    /// is enabled — DN state buffers are small (a few KB each on 27B) and
    /// avoiding the gate keeps allocation deterministic at session start.
    /// See `speculative::Phase2Snapshots` for what each snapshot holds.
    pub path_c_parent_pre_snap: DeltaNetSnapshot,
    pub path_c_main_end_snap: DeltaNetSnapshot,
}

impl DdtreeState {
    /// Return every GPU-resident snapshot + scratch buffer to the pool.
    /// Consumes self; called from `unload_model`'s DFlash teardown.
    pub fn free_gpu(self, gpu: &mut hipfire_rdna::Gpu) {
        self.post_seed_snap.free_gpu(gpu);
        self.scratch.free_gpu(gpu);
        self.path_c_parent_pre_snap.free_gpu(gpu);
        self.path_c_main_end_snap.free_gpu(gpu);
    }
}

/// Effective raw-prompt flag for prompt framing. An explicit request override
/// wins; otherwise default to raw for base/completion models (no chat_template)
/// and framed for chat models (has a chat_template).
///
/// `override_` used to be a `thread_local RAW_OVERRIDE` cell whose doc claimed
/// "reset every generate request, so no cross-request leak". That held only for
/// `generate`: the **batch prefill path never set it and never cleared it**, so a
/// batch prefill inherited whatever the last plain `generate` left behind.
///
/// Measured on gfx1103 before the fix, one identical `prefix_hash_preflight`
/// session: a fresh daemon materialised a 15-token chat-framed prompt with three
/// semantic boundaries; after a single unrelated `generate` carrying
/// `"raw": true`, the same request materialised **7 tokens** with one `full`
/// boundary. Those hashes are the KV-reuse cache keys, so the leak did not just
/// reframe a prompt — it changed what a later request matched in the prefix cache.
pub fn effective_raw(m: &LoadedModel, override_: Option<bool>) -> bool {
    override_
        .or_else(|| {
            m.registered_backend
                .as_ref()
                .and_then(|loaded| loaded.profile.prompt.raw)
        })
        .unwrap_or(m.chat_template.is_none())
}

/// The currently-resident session as one cohesive struct (C2a). Groups the four
/// formerly-decomposed `LoadedModel` working-copy fields — the generation cursor,
/// the qwen35 unified KV+DeltaNet `sequence_state`, its prefilled-suffix
/// bookkeeping, and the lfm2 recurrent/KV `lfm2moe_state` — so the forward reads
/// `m.active.cursor` / `m.active.sequence_state` as one resident unit instead of
/// spreading/recomposing them on every session swap.
///
/// Field grouping (not an enum, not accessors): `m.active.cursor` and
/// `m.active.sequence_state` are plain field paths, so Rust still grants the
/// disjoint borrows the generation loop relies on (the C1a finding — accessors
/// borrow all of `*m`; an enum can't yield `cursor` + `sequence_state` disjointly
/// without a threaded `match`). C2b will let the registry session structs hold a
/// `ResidentSession` so activate/save become a single move; C2c lifts this to N
/// concurrent residents.
#[derive(Default)]
pub struct ResidentSession {
    /// The active session's generation cursor (absolute position +
    /// conversation-token history). See [`SessionCursor`].
    ///
    /// `seq_pos` is the *physical* write position in the KV cache (the value
    /// passed to `forward_scratch(..., pos, ...)`). With no eviction, physical
    /// == absolute, so it simply grows. Under eviction, it is bounded to
    /// `physical_cap`; absolute position = `seq_pos + kv.compact_offset`.
    pub cursor: SessionCursor,
    /// Active session's live qwen35 decode state (KV cache + DeltaNet recurrent
    /// state) as one unified container. `None` when no session is resident or the
    /// arch carries no such state.
    pub sequence_state: Option<SequenceState>,
    /// qwen35 prefilled-then-generated suffix length carried with the active
    /// session's `sequence_state`.
    pub q35_active_prefilled_generated_suffix_len: usize,
    /// Active session's live LFM2.5-MoE decode state (KV + conv-state cache).
    /// `None` when no lfm2 session is resident.
    #[cfg(feature = "arch-lfm2moe")]
    pub lfm2moe_state: Option<lfm2moe::lfm2moe::Lfm2MoeState>,
}

/// Everything resident for the currently-loaded model: an `arch_id` tag plus the
/// per-family typed config/weights/state behind `Option` fields (only the active
/// arch's are `Some`), the KV cache + eviction policy, multi-turn conversation
/// bookkeeping, and the optional DFlash/vision/chat-template side state. The
/// request loop drives whichever arch is populated. The E2 `ServingBackend` seam
/// will eventually collapse this Option-soup into one boxed backend.
pub struct LoadedModel {
    pub arch_id: u32,
    /// Factory-loaded text backends share this one coarse-grained slot together
    /// with their prompt/generation policy and host-only shape metadata.
    pub registered_backend: Option<FactoryLoadedBackend>,
    /// Pipeline-parallel degree. 1 = single-GPU (all existing fields below in
    /// use, q35_scratch populated). >1 = multi-GPU (pp_gpus + pp_scratch_set
    /// populated; q35_scratch stays None; kv_cache + dn_state still hold the
    /// per-layer-routed tensors since the struct types are the same as
    /// single-GPU). Refusal contracts in load_model_pp keep DFlash, CASK,
    /// PFlash, VL and arch_id < 5 out of this branch.
    pub pp: usize,
    /// Owned multi-GPU orchestrator when `pp > 1`. The single-GPU path
    /// continues to use the daemon's main `Gpu` directly.
    pub pp_gpus: Option<Gpus>,
    /// Per-device scratch when `pp > 1`. Replaces `q35_scratch`.
    pub pp_scratch_set: Option<Qwen35ScratchSet>,
    /// LA-layer → device map returned by `DeltaNetState::new_with_quant_multi`,
    /// kept so `unload_model` and the reset handler can route per-layer
    /// memsets to the correct device.
    pub pp_dn_la_to_device: Option<Vec<u8>>,
    // Qwen3.5 state
    pub q35_config: Option<qwen35::Qwen35Config>,
    pub q35_weights: Option<qwen35::Qwen35Weights>,
    pub q35_scratch: Option<qwen35::Qwen35Scratch>,
    /// The currently-resident session (cursor + qwen35/lfm2 decode state) as one
    /// cohesive unit. P2c/C2a: replaces the former separate `sequence_state` +
    /// `q35_active_prefilled_generated_suffix_len` + `lfm2moe_state` + `cursor`
    /// working-copy fields. See [`ResidentSession`].
    pub active: ResidentSession,
    pub q35_kv_mode: Option<String>,
    pub q35_state_quant: Option<hipfire_arch_qwen35::qwen35::StateQuant>,
    pub q35_registry: SessionRegistry<Qwen35RequestSessionState>,
    // Qwen3 state
    pub llama_config: Option<llama::LlamaConfig>,
    pub llama_weights: Option<llama::LlamaWeights>,
    pub llama_scratch: Option<llama::ForwardScratch>,
    pub llama_kv: Option<kv::KvCache>,
    /// Assembled LLaMA/Qwen3 serving backend (arch_id 0/1), driven through the
    /// shared `ServingBackend::serve` seam — mirrors `qwen2_backend`/`gemma3_text`.
    /// Owns its own config/weights/scratch/KV; the separate `llama_*` fields above
    /// stay `None` on the backend path. P3.2.
    pub llama_backend: Option<hipfire_arch_llama::LlamaBackend>,
    /// Assembled Mamba-capable backend (nemotron_h arch_id 14 or pure Mamba-2
    /// arch_id 15), driven through the shared `ServingBackend::serve` seam.
    /// `NemotronModel` owns its own weights + per-block recurrent/KV state;
    /// there are no separate `nemotron_*` Option fields. N5b.
    pub nemotron_backend: Option<hipfire_arch_nemotron::model::NemotronModel>,
    /// Assembled ZAYA1 serving backend (arch_id 16 — hipfire-arch-zaya). CCA
    /// attention + EDA/MoD-routed MoE; owns its GPU weights + rolling sequence.
    /// Driven through the shared `ServingBackend::serve` seam. None on other archs.
    pub zaya_backend: Option<hipfire_arch_zaya::arch::ZayaModel>,
    // Qwen2 state (arch_id=7 — hipfire-arch-qwen2 standalone). The
    // KV cache lives inside Qwen2State, so there's no separate
    // qwen2_kv field. None on every other arch path.
    pub qwen2_config: Option<qwen2::Qwen2Config>,
    pub qwen2_weights: Option<qwen2::Qwen2Weights>,
    pub qwen2_state: Option<qwen2::Qwen2State>,
    // The separate qwen2 fields above remain only for dots-ocr's text tower.
    // Standalone Qwen2 is factory-loaded into `registered_backend`.
    // DeepSeek V4 Flash state (arch_id=9 — hipfire-arch-deepseek4).
    // Hyper-Connections + compressed-KV indexer + tail-only RoPE + raw
    // SWA cache. KV cache lives inside DeepseekV4State; no separate
    // deepseek4_kv field. None on every other arch path.
    pub deepseek4_config: Option<deepseek4::DeepseekV4Config>,
    pub deepseek4_weights: Option<deepseek4::DeepseekV4Weights>,
    pub deepseek4_state: Option<deepseek4::DeepseekV4State>,
    /// Pre-allocated PrefillBatchScratch sized to `HIPFIRE_DEEPSEEK4_PP_BATCH`
    /// (default 64). Used by both batched prefill and the MTP spec-decode
    /// verify pass. Lazy-allocated on first arch_id=9 load — None on every
    /// other arch path.
    pub deepseek4_pbs: Option<deepseek4::forward::PrefillBatchScratch>,
    /// Cached `<｜end▁of▁sentence｜>` token id resolved at load time.
    /// Falls back to 1 (DeepSeek family default) if the tokenizer lacks
    /// the special-token entry.
    pub deepseek4_eos_tok: u32,
    /// MTP config — parsed from load-message params, read at generate time.
    /// Arch-agnostic: currently only DeepSeek V4 (arch_id=9) evaluates these,
    /// but the namespace is intentionally not deepseek4-specific.
    pub mtp_mode: String,
    /// Draft tokens per spec-decode window (1-10, default 3).
    pub mtp_k: usize,
    /// Whether MTP head weights were found at load time. Set by the sibling-
    /// file scan (e.g. `<stem>-mtp.*`) or bundled MTP detection. Used by
    /// `mtp_mode = "auto"` to decide whether to enable spec-decode.
    pub mtp_weights_present: bool,
    // MiniMax-M2 state (arch_id=10 — hipfire-arch-minimax). Mixtral-style
    // MoE: GQA + per-layer QK-norm + partial RoPE + sigmoid-bias top-k
    // routing, no shared expert. KV cache lives inside MiniMaxState; no
    // separate field. NO PrefillBatchScratch — prefill is the per-token
    // `decode_step` loop. None on every other arch path.
    pub minimax_config: Option<minimax::MiniMaxConfig>,
    pub minimax_weights: Option<minimax::MiniMaxWeights>,
    pub minimax_state: Option<minimax::MiniMaxState>,
    /// Cached EOS token id resolved at load time. Falls back to 1 if the
    /// tokenizer lacks the special-token entry.
    pub minimax_eos_tok: u32,
    // LFM2.5-8B-A1B state (arch_id=11 — hipfire-arch-lfm2moe). Hybrid:
    // double-gated LIV short-conv mixers interleaved with GQA+QK-norm
    // attention, feeding a DeepSeek-style sigmoid-bias top-4 MoE FFN (or
    // dense SwiGLU on the first num_dense_layers). KV cache + conv-state
    // cache both live inside Lfm2MoeState; no separate field. NO
    // PrefillBatchScratch — prefill is the per-token `decode_step` loop.
    // None on every other arch path. Structurally mirrors MiniMax (10).
    #[cfg(feature = "arch-lfm2moe")]
    pub lfm2moe_config: Option<lfm2moe::config::Lfm2MoeConfig>,
    #[cfg(feature = "arch-lfm2moe")]
    pub lfm2moe_weights: Option<lfm2moe::lfm2moe::Lfm2MoeWeights>,
    #[cfg(feature = "arch-lfm2moe")]
    pub lfm2_registry: SessionRegistry<Lfm2RequestSessionState>,
    /// Cached EOS token id resolved at load time. Falls back to 1 if the
    /// tokenizer lacks the special-token entry.
    #[cfg(feature = "arch-lfm2moe")]
    pub lfm2moe_eos_tok: u32,
    #[cfg(feature = "arch-lfm2moe")]
    pub lfm2_dflash: Option<Lfm2DflashState>,
    // dots.ocr state (arch_id=8 — Qwen2-VL family). The text decoder is
    // Qwen2: `dots_ocr_config.text` / `dots_ocr_weights.text` feed
    // `qwen2::forward_step*`, and the per-step decode state reuses the
    // `qwen2_state` field above. `dots_ocr_weights.vision` holds the
    // resident vision-tower weights for `dots_ocr::vision_forward`.
    pub dots_ocr_config: Option<dots_ocr::DotsOcrConfig>,
    pub dots_ocr_weights: Option<dots_ocr::DotsOcrWeights>,
    // Vision state (VL models only)
    pub vision_config: Option<qwen35_vl::VisionConfig>,
    pub vision_weights: Option<qwen35_vl::VisionWeights>,
    // Gemma3-VL (medgemma, arch_id=13). Self-contained multimodal backend
    // (gemma3 text decoder + SigLIP tower + projector + decode state) behind
    // the object-safe `ServingBackend::serve` seam — its own KV/decode state
    // lives inside, so there is no separate kv_cache/scratch field. The
    // `has_vl` gate keys off `gemma3_vl.is_some()` for arch 13 (rather than the
    // qwen35-typed `vision_config`). None on every other arch path.
    pub gemma3_vl: Option<Gemma3VlBackend>,
    // Gemma3 text (medgemma-*-text, arch_id=12). The splice-free text decoder
    // behind the same `ServingBackend::serve` seam (delegates to `run_simple_ar`);
    // its KV/decode state lives inside `Gemma3Backend`. None on every other arch.
    pub gemma3_text: Option<Gemma3Backend>,
    // embeddinggemma (arch_id=19). Bidirectional encoder for embeddings/rerank;
    // no KV cache and no autoregressive decode state.
    pub embeddinggemma: Option<EmbeddingGemmaState>,
    /// Qwen3 (arch_id=1) embedding workload. Holds no lm_head, generation
    /// scratch, or persistent KV state; all encoder execution is XDNA-only.
    pub qwen3_embedding: Option<Qwen3EmbeddingState>,
    // Shared
    pub tokenizer: Option<hipfire_model::tokenizer::Tokenizer>,
    /// Advertised context window — client-facing capacity, the upper bound on
    /// absolute conversation length. Without eviction this equals
    /// `physical_cap` (the buffer size); under eviction it can be much larger.
    pub max_seq: usize,
    /// Physical KV buffer capacity, in slots. Allocators size per-layer K/V
    /// for this many tokens. Under eviction, budget+beta <= physical_cap;
    /// without eviction, physical_cap may be lower than max_seq and grows by
    /// loading a larger worker.
    pub physical_cap: usize,
    /// When Some(_), the daemon calls `maybe_evict` after every prefill-chunk
    /// and every decode-forward so the physical cache stays bounded by
    /// `physical_cap` even when `max_seq` advertises a much larger window.
    pub eviction: Option<Eviction>,

    /// Per-turn token cache for V4F prefix-cache stability.
    ///
    /// Maps a stable fingerprint of an assistant message — `(role,
    /// content_text, tool_calls_canonical_json)` — to the token IDs the
    /// model ACTUALLY emitted for that turn. When the next request
    /// replays the same assistant message in its `messages` history, the
    /// V4F render loop uses these cached tokens verbatim instead of
    /// re-encoding via `render_assistant_tool_calls` + tokenizer.encode.
    ///
    /// Why this matters: BPE is not bijective. The model can emit a
    /// 2-token DSML tool_call (multi-char special tokens picked
    /// greedily); our re-encode of the same text via Jinja-style
    /// rendering may produce 67 tokens covering the same string. The
    /// resulting prompt diverges from the prior turn's KV slots at
    /// the assistant-turn boundary, capping the prefix-cache LCP at
    /// the divergence point. Caching the emitted tokens restores
    /// byte-identical replay and lets LCP extend through all prior
    /// assistant turns.
    ///
    /// Cleared on model unload (LoadedModel destruction). Bounded by
    /// the natural lifetime of a session — entries that never come
    /// back in a `messages` history will linger but never affect
    /// correctness (worst case: VRAM-free Vec<u32> memory growth on
    /// the host).
    pub asst_turn_cache: std::collections::HashMap<u64, Vec<u32>>,

    /// Lazily-built decoded-vocab cache for grammar-guided sampling.
    /// `tokenizer.decode(&[id])` for every id ∈ `0..vocab_size`. Built
    /// once on first tool-using V4F request, reused for every subsequent
    /// request on the same model. Without this cache, each generate
    /// rebuilt all ~129k entries at request entry (one tokenizer.decode
    /// allocation per id), adding tens of milliseconds of pure overhead
    /// to every tool-using turn. `None` until first build; cleared by
    /// `unload_model` via `LoadedModel` drop.
    pub decoded_vocab: Option<std::sync::Arc<Vec<String>>>,
    // Target model file path — cached so the DFlash fast path can reopen the
    // HfqFile mmap to construct a transient ModelSlot without reloading
    // weights. `HfqFile::open` is a cheap mmap operation.
    pub model_path: String,
    pub memory: ModelArtifactMemory,
    // DFlash speculative decoding state (populated when load supplied a draft).
    pub dflash: Option<DflashState>,
    // Drafter-free n-gram speculative decode (`ngram_spec`). `None` = off.
    // Deliberately NOT inside `dflash`: it needs no drafter. See `NgramState`.
    pub ngram: Option<NgramState>,
    // DSpark speculative decoding state for the dense LLaMA/Qwen3 arch (0/1),
    // populated when a `.dspark.hfq` sidecar is discovered next to the target.
    pub dspark: Option<DsparkState>,
    // Upstream HF Jinja chat_template, extracted from the HFQ
    // `tokenizer_config.chat_template` at load time. `None` when the source
    // model didn't ship one (rare for instruct models). Consumed only when
    // `chat_prompt` resolves to Jinja; otherwise the daemon's hand-rolled
    // `prompt_frame::ChatFrame::Plain` scaffolding renders the prompt.
    pub chat_template: Option<String>,
    pub chat_template_profile: Option<prompt_frame::ChatTemplateProfile>,
    /// How this model's prompts are framed, resolved once at load by
    /// [`hipfire_model::chat_prompt_policy`] from the arch's own default plus the
    /// operator's `jinja_chat` setting (`auto` | `on` | `off`, settable per model under
    /// `model_overrides`, with `HIPFIRE_JINJA_CHAT` still winning when set). One
    /// resolved answer on the model rather than the decision itself re-derived at each
    /// render site: the sites used to spell it as two *opposite* environment
    /// comparisons, so a Qwen-only defect read as a global one.
    pub chat_prompt: hipfire_model::ChatPromptPolicy,
}

impl LoadedModel {
    /// Does this model's prompt render through its Jinja `chat_template`?
    ///
    /// `true` implies `chat_template.is_some()` — [`hipfire_model::chat_prompt_policy`]
    /// resolves a template-less model to the scaffold — so a render site may unwrap it.
    pub fn renders_jinja(&self) -> bool {
        self.chat_prompt == hipfire_model::ChatPromptPolicy::Jinja
    }

    /// Active session's KV cache, if any. Replaces the former `kv_cache.as_ref()`
    /// on the unified `sequence_state`. Sites needing KV **and** DeltaNet
    /// simultaneously bind `sequence_state.as_mut()` once, then borrow its
    /// disjoint `.kv` / `.recurrent` fields.
    pub fn kv_cache(&self) -> Option<&kv::KvCache> {
        self.active.sequence_state.as_ref().and_then(|s| s.kv())
    }
    /// Mutable active KV cache (single-access only — see [`Self::kv_cache`]).
    pub fn kv_cache_mut(&mut self) -> Option<&mut kv::KvCache> {
        self.active.sequence_state.as_mut().and_then(|s| s.kv_mut())
    }
    /// Active session's DeltaNet recurrent state, if any.
    pub fn dn_state(&self) -> Option<&DeltaNetState> {
        self.active
            .sequence_state
            .as_ref()
            .and_then(|s| s.recurrent_as::<DeltaNetState>())
    }
    /// Mutable active DeltaNet recurrent state (single-access only).
    pub fn dn_state_mut(&mut self) -> Option<&mut DeltaNetState> {
        self.active
            .sequence_state
            .as_mut()
            .and_then(|s| s.recurrent_as_mut::<DeltaNetState>())
    }
}

#[cfg(test)]
mod ngram_setup_tests {
    use super::*;

    fn setup() -> NgramSetup {
        NgramSetup {
            store_root: std::path::PathBuf::from("ram"),
            scope: "s".into(),
            blocks: 1,
            orders: vec![2],
            chain_floor: 0,
            min_acceptance: 0.0,
            min_acceptance_proposals: 8,
            max_spine: 1,
            promote_count: 1,
            write_target: NgramWriteTarget::User,
        }
    }

    /// A request must never inherit live tables staged under a different scope
    /// key. The key is user + session type, so a mismatch here is one user's
    /// decoded text drafting another user's request — a leak, not a cache miss.
    #[test]
    fn live_tables_never_cross_a_scope_key() {
        let mut st = NgramState::new(setup());
        let cfg = hipfire_specdecode_ngram::NgramConfig::default();

        st.live = Some((
            "alice\u{1}chat".into(),
            hipfire_specdecode_ngram::NgramSpec::new(cfg.clone()),
        ));
        assert!(
            st.take_live_for("bob\u{1}chat").is_none(),
            "bob must not receive alice's tables"
        );
        assert!(
            st.live.is_none(),
            "the mismatched state must be dropped, not left for the next caller"
        );

        st.live = Some((
            "alice\u{1}chat".into(),
            hipfire_specdecode_ngram::NgramSpec::new(cfg),
        ));
        assert!(
            st.take_live_for("alice\u{1}chat").is_some(),
            "the same scope must carry its own tables across requests, or the hot \
             tier is rebuilt per request and the feature is decorative"
        );
    }

    /// A dropped load must flush what it staged. Before this, `merge_backlog` went
    /// out of scope with the `NgramSpec` and every gram that had lost an in-place
    /// insert was lost with it — which, for a single-user single-topic daemon that
    /// never swaps scope, was every merge the store would ever get.
    ///
    /// Asserts that dropping rewrote the store, not that a particular gram landed:
    /// `merge` bumps and persists the epoch, so the file cannot come out identical
    /// if it ran, and cannot differ if it did not. What `merge` itself keeps is
    /// `cold_store_roundtrips_and_rebalances`'s job, in the crate that owns it.
    #[test]
    fn dropping_a_load_flushes_the_staged_grams_to_disk() {
        use hipfire_specdecode_ngram::{NgramConfig, NgramSpec};

        let dir = std::env::temp_dir().join(format!("hng-unload-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("t.hng");
        let _ = std::fs::remove_file(&path);

        // Fewer blocks than distinct keys, so keys touched late cannot be handed
        // one and spill to the backlog — the state this is about.
        let mut ng = NgramSpec::new(NgramConfig {
            promote_count: 1,
            orders: vec![2],
            ..Default::default()
        });
        ng.attach_user(&path, 256, 8).unwrap();
        let period = 64u32;
        let stream: Vec<u32> = (0..period * 16).map(|i| i % period).collect();
        ng.observe(&stream);
        assert!(
            ng.merge_backlog_len() > 0,
            "test needs a non-empty backlog to prove anything was flushed"
        );

        // In-place inserts are already in the mmap, so this snapshot differs from
        // the next one only if the drop merged.
        let before = std::fs::read(&path).unwrap();

        let mut st = NgramState::new(setup());
        st.live = Some(("alice\u{1}chat".into(), ng));
        drop(st);

        let after = std::fs::read(&path).unwrap();
        assert_ne!(
            before, after,
            "dropping a load left the store untouched — the staged grams went with it"
        );
        let _ = std::fs::remove_file(&path);
    }

    /// `persists()` reads the `ngram_store_root` schema field's arm rather than
    /// its own list, so the two cannot drift. This is what binds them: a
    /// sentinel added to the schema and not handled here fails the build's
    /// tests, not a user's config.
    #[test]
    fn every_schema_sentinel_means_ram() {
        for sentinel in hipfire_config::NGRAM_STORE_ROOT_RAM {
            let setup = NgramSetup {
                store_root: std::path::PathBuf::from(sentinel),
                scope: "s".into(),
                blocks: 1,
                orders: vec![2],
                chain_floor: 0,
                max_spine: 1,
                promote_count: 1,
                min_acceptance: 0.0,
                min_acceptance_proposals: 8,
                write_target: NgramWriteTarget::User,
            };
            assert!(
                !setup.persists(),
                "schema declares {sentinel:?} a RAM sentinel but persists() disagrees"
            );
        }
    }

    /// The RAM-only case has to be expressible as a *value*, not just as "leave
    /// the field blank" — a settings UI cannot distinguish an unset field from
    /// one deliberately set to RAM-only.
    #[test]
    fn ram_sentinels_disable_persistence() {
        let mk = |root: &str| NgramSetup {
            store_root: std::path::PathBuf::from(root),
            scope: "s".into(),
            blocks: 1,
            orders: vec![2],
            chain_floor: 0,
            max_spine: 1,
            promote_count: 1,
            min_acceptance: 0.0,
            min_acceptance_proposals: 8,
            write_target: NgramWriteTarget::User,
        };
        for off in [
            "", "  ", "ram", "RAM", "Ram", "none", "NONE", "off", " off ",
        ] {
            assert!(!mk(off).persists(), "{off:?} should mean RAM-only");
        }
        for on in [
            "/srv/ngram",
            "./tables",
            "/tmp/ram-disk",
            "ramdisk",
            "/var/ram",
        ] {
            assert!(mk(on).persists(), "{on:?} is a path and must persist");
        }
    }

    #[test]
    fn write_target_parses_and_defaults_to_private() {
        assert_eq!(NgramWriteTarget::parse("topic"), NgramWriteTarget::Topic);
        assert_eq!(NgramWriteTarget::parse("none"), NgramWriteTarget::None);
        assert_eq!(NgramWriteTarget::parse("off"), NgramWriteTarget::None);
        assert_eq!(NgramWriteTarget::parse("user"), NgramWriteTarget::User);
        assert_eq!(NgramWriteTarget::parse("USER"), NgramWriteTarget::User);
        // Anything unrecognised must fall back to the private store, never to a
        // shared one.
        assert_eq!(NgramWriteTarget::parse("typo"), NgramWriteTarget::User);
        assert_eq!(NgramWriteTarget::parse(""), NgramWriteTarget::User);
    }
}
