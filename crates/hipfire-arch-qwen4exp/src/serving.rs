// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! Serving seam for `qwen4_exp` (arch_id 26): `SimpleAr` + `ServingBackend` on
//! [`Qwen4ExpBackend`], registered through [`ServingFactory`] so the daemon does
//! a data lookup rather than growing another `arch_id` branch.
//!
//! **Prefill is per-token.** The trunk has no batched prefill: `decode_step_into`
//! advances exactly one position, and the recurrent halves (Gated DeltaNet state,
//! the PLE conv ring) are sequential by construction. `prefill` therefore replays
//! the prompt one token at a time, which is correct but O(prompt) launches. A
//! batched path is the obvious next perf item and is NOT a correctness gap.
//!
//! ## The two tables the trunk reads on the host
//!
//! `decode_step_into` gathers the token embedding and the PLE n-gram rows from
//! host memory. They are very different problems:
//!
//! * the token embedding is `[vocab, hidden]` — 2.5 GB at the shipped geometry,
//!   large but holdable, and read once per token;
//! * the n-gram table is **102 GB**, 41% of the parameters, and one token touches
//!   exactly `heads_per_ngram` rows of it.
//!
//! The second is handled by reading rows out of the artifact on demand
//! ([`crate::ngram_rows`]), chosen by SIZE rather than configuration: a fixture's
//! table is held, the shipped one is streamed. `examples/serve_fixture` forces the
//! streamed path on a small artifact and requires BIT-IDENTICAL logits, because a
//! wrong shard split or row offset yields a perfectly plausible embedding.
//!
//! Still outstanding: weights are dequantised to f32 at load, so an oq4 model
//! costs what an f32 one does. Fitting the shipped checkpoint needs them to stay
//! quantised behind `hipfire_runtime::weights::WeightTensor`.

use crate::arch::{load_ngram_table, HfqTensorReader, Qwen4Exp};
use crate::config::Qwen4ExpConfig;
use crate::ngram::NgramHasher;
use crate::ngram_rows::{HfqShardRows, NgramRows, ResidentRows};
use crate::prefill_gpu::{prefill_chunk_into, BatchScratch};
use crate::trunk_gpu::{decode_step_into, TensorReader, TrunkScratch, TrunkState, TrunkWeights};
use hipfire_arch_api::ARCH_ID_QWEN4EXP;
use hipfire_rdna::{Gpu, GpuTensor};
use hipfire_runtime::arch::{
    run_simple_ar, ArchCaps, Architecture, FactoryLoadedBackend, GenerateCtx, ModelShapeProfile,
    OutputProtocol, PromptGenerationProfile, ServeOutcome, ServingBackend, ServingFactory,
    ServingFactoryOptions, SimpleAr,
};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::tokenizer::Tokenizer;

/// Above this, n-gram rows are STREAMED from the artifact instead of held.
///
/// 8 GiB is far above any fixture and far below the shipped 102 GB, so it
/// separates the two cases without needing to know which model this is. It is a
/// performance choice, not a correctness one: both paths are differenced and
/// produce identical rows.
const NGRAM_HOST_BUDGET_BYTES: u64 = 8 << 30;

/// How this backend gets n-gram embedding rows.
///
/// The choice is made by SIZE, not by configuration: a fixture's table is a few
/// MB and reading it once is cheaper than a pread per row, while the shipped
/// model's is 102 GB and holding it is impossible. Both paths produce identical
/// rows — `examples/serve_fixture` differences them on the same artifact.
enum NgramSource {
    /// No PLE block in this model.
    None,
    /// The whole table on the host.
    Resident(Vec<f32>),
    /// Rows read out of the artifact's shard tensors on demand. Owns its OWN
    /// `HfqFile` handle: the factory only borrows the caller's, and the loader's
    /// copy is gone long before the first token.
    Streamed {
        hfq: Box<HfqFile>,
        hasher: Box<NgramHasher>,
        layer: usize,
    },
}

/// A loaded Qwen3.8-Flash-Next model: GPU-resident trunk weights and per-layer
/// state (KV cache for the sparse-attention layers, DeltaNet recurrent state and
/// conv ring for the rest), plus the two host-side tables the trunk still reads.
pub struct Qwen4ExpBackend {
    cfg: Qwen4ExpConfig,
    weights: TrunkWeights,
    state: TrunkState,
    scratch: TrunkScratch,
    /// `[vocab, hidden]`, host-side — `decode_step_into` gathers one row per step.
    embed: Vec<f32>,
    /// Where the PLE n-gram rows come from — resident for a small table, streamed
    /// from the artifact for one that cannot be held.
    ngram: NgramSource,
    /// Every token committed so far. The trunk takes the full history because the
    /// PLE hash reads an n-gram window, not just the current token.
    history: Vec<u32>,
    eos: u32,
    /// The routed-expert pager, when this artifact carries a module table.
    ///
    /// Held so `unload` can free what it paged in. The expert stacks hold their
    /// own `Arc` clones, so this is not what keeps it alive — it is the handle
    /// for the teardown the stacks deliberately do not do.
    pager: Option<crate::moe_gpu::ExpertPager>,
    /// Chunk-wide scratch for the batched prefill, allocated on first use so a
    /// backend that only ever decodes never pays for it.
    batch: Option<BatchScratch>,
    /// Upper bound on a prefill chunk (`PREFILL_CHUNK`, capped by `max_seq`).
    chunk_rows: usize,
    /// Sequence capacity the state was built for (KV caches, positions).
    max_seq: usize,
    /// MTP speculative decode, when a head is attached (`attach_mtp`).
    mtp: Option<Box<MtpState>>,
}

/// Counters for MTP speculation. A `verify` is one 2-row trunk forward; each
/// one is followed by exactly one `hit` (the draft was the next token, so that
/// step cost no forward) or one `miss` (rolled back), except a verify still
/// pending when generation stopped.
#[derive(Clone, Copy, Debug, Default)]
pub struct MtpStats {
    pub verifies: u64,
    pub hits: u64,
    pub misses: u64,
}

/// A verify's second row: the trunk has already run `draft` at `pos`.
struct Pending {
    pos: usize,
    draft: u32,
}

/// Everything MTP speculation holds. Boxed: it is ~3 GB of head on the GPU and
/// a backend without one should not carry the fields.
struct MtpState {
    head: crate::mtp_gpu::MtpWeightsGpu,
    scratch: crate::mtp_gpu::MtpScratchGpu,
    /// The head's OWN attention cache — it is a full QSA layer.
    cache: crate::attn_gpu::QsaCache,
    /// `[vocab]` — the head's draft logits through the shared `lm_head`.
    logits: GpuTensor,
    /// Chunk-wide scratch for writing the head cache over a prompt.
    hb: BatchScratch,
    /// Two-row scratch for verifies (a separate one: the prefill scratch's
    /// all-position logits would be 512 x vocab for a 2-row forward).
    vb: BatchScratch,
    ckpt: crate::prefill_gpu::Row0Checkpoint,
    pending: Option<Pending>,
    /// Test hook: drafts by absolute position instead of the head's.
    script: Option<Vec<u32>>,
    stats: MtpStats,
}

impl MtpState {
    fn free(self, gpu: &mut Gpu) {
        let Self {
            head,
            scratch,
            cache,
            logits,
            hb,
            vb,
            ckpt,
            pending: _,
            script: _,
            stats: _,
        } = self;
        head.free(gpu);
        scratch.free(gpu);
        cache.free(gpu);
        let _ = gpu.free_tensor(logits);
        hb.free(gpu);
        vb.free(gpu);
        ckpt.free(gpu);
    }
}

/// Which forward a step takes.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Rows {
    /// The rows go through the batched prefill scratch.
    Batch { all_logits: bool },
    /// `[token, draft]` through the verify scratch, with a row-0 checkpoint.
    Verify,
}

/// Tokens per batched prefill chunk. Every routed expert hit in a chunk is read
/// once for the whole chunk, so bigger amortises the expert sweep further; the
/// scratch is ~25 MB at this size (logits excluded — those are only allocated
/// when a scorer asks for every position).
const PREFILL_CHUNK: usize = 512;

/// How the head folds the token embedding into the trunk's streams: the
/// reading that measured 60.9% acceptance on the shipped 180B.
const MTP_FUSION: crate::mtp::Fusion = crate::mtp::Fusion::BroadcastAllStreams;

/// `HIPFIRE_QWEN4EXP_PER_TOKEN_PREFILL=1` — DEBUG A/B only: prefill through the
/// per-token decode step, the pre-batching path, to difference against it.
fn per_token_prefill() -> bool {
    std::env::var("HIPFIRE_QWEN4EXP_PER_TOKEN_PREFILL").as_deref() == Ok("1")
}

impl Qwen4ExpBackend {
    /// Build a backend from an artifact. The [`ServingFactory`] below is a thin
    /// wrapper over this, so the daemon path and any direct driver construct the
    /// SAME object — a test that exercises one is exercising the other.
    pub fn load(gpu: &mut Gpu, hfq: &mut HfqFile, max_seq: usize) -> Result<Self, String> {
        Self::load_with_ngram_budget(gpu, hfq, max_seq, NGRAM_HOST_BUDGET_BYTES)
    }

    /// [`Self::load`] with the resident/streamed threshold overridden.
    ///
    /// Exists so a test can force the STREAMED path on a small artifact and
    /// difference it against the resident one. Without that, the streaming row
    /// addressing (shard split, row-in-shard, element offset) would only ever run
    /// on the 102 GB model, where a wrong row is an indistinguishable-looking
    /// embedding rather than a failure.
    /// Routed-expert paging counters, or `None` when this model is not paged.
    ///
    /// Exists so a check can prove it is not passing VACUOUSLY: "paged and
    /// resident agree" means nothing if the paged run never actually paged.
    pub fn expert_pager_stats(&self) -> Option<hipfire_runtime::weight_pager::ModulePagerStats> {
        // Registered is NOT the same as used. A bf16 artifact carries a module
        // table like any other, but bf16 has no per-expert device form, so every
        // layer declines the pager and loads resident. Reporting stats then would
        // read as "paged, 0 cold loads" — which is what a silent fallback bug
        // looks like, so the honest answer is `None`.
        if !self.weights.layers.iter().any(|l| l.moe.gate_up.is_paged()) {
            return None;
        }
        let p = self.pager.as_ref()?;
        Some(p.lock().ok()?.module_stats())
    }

    /// Open a routed-expert pager over `hfq`, or `None` when it has no module
    /// table to page from.
    ///
    /// `PerExpertNative` rather than the indexed-MoE default: this arch's
    /// `moe_forward` calls `weight_gemv` once per routed expert with a
    /// `WeightTensor`, so it needs each expert in the layout that dispatch reads
    /// — not the block-interleaved form the indexed kernels want.
    fn build_expert_pager(hfq: &HfqFile) -> Result<Option<crate::moe_gpu::ExpertPager>, String> {
        use hipfire_runtime::weight_pager::{
            ExpertResidentLayout, PagerConfig, ResidencyPolicy, WeightPager,
        };
        if hfq.modules().is_empty() {
            return Ok(None);
        }
        // Debug escape hatch, not a user knob. Paged and resident must produce
        // identical logits, and that is only checkable if the resident path can
        // still be reached on an artifact that carries a module table — which,
        // since the quantizer emits one, is now every artifact.
        if matches!(
            std::env::var("HIPFIRE_QWEN4EXP_PAGED_EXPERTS")
                .ok()
                .as_deref(),
            Some("0" | "off" | "false" | "no")
        ) {
            return Ok(None);
        }
        // LRU eviction by default: the routed experts outsize host memory, so
        // pinning on first touch OOMs mid-conversation. The env var is a debug
        // override of the budget.
        let residency = ResidencyPolicy::expert_lru(
            std::env::var("HIPFIRE_QWEN4EXP_EXPERT_CACHE_BYTES")
                .ok()
                .and_then(|v| v.parse::<u64>().ok())
                .filter(|b| *b > 0),
        );
        let mut pager = WeightPager::with_env_transport(
            hfq.path(),
            PagerConfig {
                expert_layout: ExpertResidentLayout::PerExpertNative,
                residency,
                trace: matches!(
                    std::env::var("HIPFIRE_QWEN4EXP_EXPERT_CACHE_TRACE")
                        .ok()
                        .as_deref(),
                    Some("1" | "true" | "on" | "yes")
                ),
            },
        )
        .map_err(|e| format!("qwen4_exp: open expert module pager: {e}"))?;
        let n = pager
            .register_expert_modules(hfq.modules().iter().cloned())
            .map_err(|e| format!("qwen4_exp: register expert modules: {e}"))?;
        if n == 0 {
            return Ok(None);
        }
        // Registered, not necessarily used — whether a layer takes the pager
        // depends on its quant type having a per-expert device form. Worded so it
        // does not claim more than it knows.
        eprintln!("qwen4_exp: module table: {n} routed expert modules available to page");
        Ok(Some(std::sync::Arc::new(std::sync::Mutex::new(pager))))
    }

    pub fn load_with_ngram_budget(
        gpu: &mut Gpu,
        hfq: &mut HfqFile,
        max_seq: usize,
        ngram_host_budget: u64,
    ) -> Result<Self, String> {
        let cfg = Qwen4Exp::config_from_hfq(hfq)?;
        // PAGED ROUTED EXPERTS, when the artifact was repacked with a module
        // table. Presence of the table IS the switch: routed experts are 97.3%
        // of this trunk, so an artifact that carries the table was repacked in
        // order to be paged, and one that does not cannot be. There is nothing
        // for a flag to choose between.
        let pager = Self::build_expert_pager(hfq)?;
        let weights = {
            let reader = HfqTensorReader { hfq };
            TrunkWeights::upload_paged(gpu, &cfg, &reader, pager.as_ref())
                .map_err(|e| format!("qwen4_exp: {e:?}"))?
        };
        let max_seq = max_seq.min(cfg.max_position).max(1);
        let state =
            TrunkState::new(gpu, &cfg, max_seq).map_err(|e| format!("qwen4_exp state: {e:?}"))?;
        let scratch = TrunkScratch::new(gpu, &cfg, max_seq)
            .map_err(|e| format!("qwen4_exp scratch: {e:?}"))?;

        let reader = HfqTensorReader { hfq };
        let embed = reader.read("model.language_model.embed_tokens.weight")?;
        // Resident when the table fits the host budget, streamed when it does not.
        // This is what lets the shipped checkpoint load at all: its table is 102 GB.
        let ngram = match cfg.ngram.as_ref() {
            None => NgramSource::None,
            Some(n) if Self::ngram_bytes(n) <= ngram_host_budget => {
                NgramSource::Resident(load_ngram_table(&reader, &cfg)?)
            }
            Some(n) => {
                // Re-open the artifact so the backend owns a handle that outlives
                // the factory's borrow. `HfqShardRows::new` validates the shard
                // encoding up front, so an unsupported one is refused here rather
                // than on the first token.
                let path = hfq.path().to_path_buf();
                let own = HfqFile::open(&path).map_err(|e| {
                    format!(
                        "qwen4_exp: reopening {} for streamed n-gram rows: {e}",
                        path.display()
                    )
                })?;
                let hasher = NgramHasher::from_config(n, cfg.vocab as u64, cfg.eos_token_id);
                HfqShardRows::new(&own, &hasher, n.layer_idx)?;
                NgramSource::Streamed {
                    hfq: Box::new(own),
                    hasher: Box::new(hasher),
                    layer: n.layer_idx,
                }
            }
        };
        let eos = cfg.eos_token_id;
        Ok(Self {
            cfg,
            weights,
            state,
            scratch,
            embed,
            ngram,
            history: Vec::new(),
            eos,
            pager,
            batch: None,
            chunk_rows: PREFILL_CHUNK.min(max_seq),
            max_seq,
            mtp: None,
        })
    }

    /// Host bytes the n-gram table would occupy if held resident.
    fn ngram_bytes(n: &crate::config::NgramConfig) -> u64 {
        let (_, _, padded) = hipfire_arch_qwen4exp_spec::ngram_head_layout(
            n.vocab_size_base,
            n.heads(),
            n.divisible_by,
        );
        padded * n.head_dim() as u64 * 4
    }

    pub fn config(&self) -> &Qwen4ExpConfig {
        &self.cfg
    }

    /// Device dtype the ROUTED EXPERTS are resident in.
    ///
    /// Worth surfacing because the difference is invisible from the logits: an
    /// artifact whose experts were dequantised to f32 serves exactly as well and
    /// costs ~8x the memory. On the shipped geometry the experts are 97.3% of the
    /// trunk, so this one value decides whether the model fits.
    /// One row of the trunk's embedding table, `[hidden]`.
    ///
    /// The MTP head shares it (`mtp_use_dedicated_embeddings` is false), so the
    /// probe and any future drafter need the same rows the trunk gathers.
    pub fn embed_row(&self, token: u32) -> &[f32] {
        let h = self.cfg.hidden;
        let o = token as usize * h;
        &self.embed[o..o + h]
    }

    /// The uploaded trunk weights, for building a calibration capture map.
    pub fn weights(&self) -> &TrunkWeights {
        &self.weights
    }

    /// Apply the trunk's `lm_head` to an arbitrary collapsed hidden.
    ///
    /// The MTP head shares this head (`mtp_use_dedicated_embeddings` is false),
    /// so a draft's logits must come from the same matrix the trunk uses — a
    /// separate copy would drift and make acceptance meaningless.
    pub fn logits_of(&self, gpu: &mut Gpu, hidden: &GpuTensor) -> Result<Vec<f32>, String> {
        hipfire_runtime::weights::weight_gemv(
            gpu,
            &self.weights.lm_head,
            hidden,
            self.scratch.logits(),
        )
        .map_err(|e| format!("qwen4_exp lm_head: {e:?}"))?;
        gpu.download_f32(self.scratch.logits())
            .map_err(|e| format!("qwen4_exp logits download: {e:?}"))
    }

    /// Last-position logits, `[vocab]`.
    pub fn trunk_logits(&self) -> &GpuTensor {
        self.scratch.logits()
    }

    /// The trunk's wide residual and collapsed hidden from the last step.
    ///
    /// Exposed for the MTP probe: the head reads the wide stream, and its own
    /// output is compared against the collapsed one.
    pub fn trunk_states(&self) -> (&GpuTensor, &GpuTensor) {
        (self.scratch.wide(), self.scratch.collapsed())
    }

    pub fn routed_expert_dtype(&self) -> Option<hipfire_rdna::DType> {
        self.weights.layers.first().map(|l| l.moe.gate_up.dtype())
    }

    /// Run `history[pos0..]` through the trunk, leaving the last position's
    /// logits on the GPU. One row takes the decode step; more take the batched
    /// layer-major forward (`prefill_gpu`). `all_logits` forces the batched path
    /// and also keeps every row's logits in the batch scratch.
    ///
    /// `self` is destructured so the n-gram provider (which borrows `ngram`) and
    /// the mutable trunk state are disjoint borrows rather than two of `self`.
    fn forward(&mut self, gpu: &mut Gpu, pos0: usize, all_logits: bool) -> Result<(), String> {
        self.run_rows(gpu, pos0, Rows::Batch { all_logits })
    }

    fn run_rows(&mut self, gpu: &mut Gpu, pos0: usize, mode: Rows) -> Result<(), String> {
        let all_logits = matches!(mode, Rows::Batch { all_logits: true } | Rows::Verify);
        let Self {
            cfg,
            weights,
            state,
            scratch,
            embed,
            ngram,
            history,
            eos,
            // Reached through the expert stacks, not from here.
            pager: _,
            batch,
            chunk_rows,
            max_seq: _,
            mtp,
        } = self;
        let resident;
        let streamed;
        let rows: Option<&dyn NgramRows> = match ngram {
            NgramSource::None => None,
            NgramSource::Resident(t) => {
                resident = ResidentRows { table: t };
                Some(&resident)
            }
            NgramSource::Streamed { hfq, hasher, layer } => {
                streamed = HfqShardRows::new(hfq, hasher, *layer)?;
                Some(&streamed)
            }
        };
        let n = history.len() - pos0;
        let _trace_rows = hipfire_runtime::dispatch::moe_telemetry::trace_rows(pos0..pos0 + n);
        if mode == Rows::Verify {
            let m = mtp.as_mut().ok_or("qwen4_exp verify without an MTP head")?;
            return prefill_chunk_into(
                gpu,
                cfg,
                weights,
                state,
                scratch,
                &mut m.vb,
                embed,
                rows,
                history,
                pos0,
                *eos,
                true,
                Some(&m.ckpt),
            )
            .map_err(|e| format!("qwen4_exp verify: {e:?}"));
        }
        if n == 1 && !all_logits {
            return decode_step_into(
                gpu, cfg, weights, state, scratch, embed, rows, history, pos0, *eos,
            )
            .map_err(|e| format!("qwen4_exp decode: {e:?}"));
        }
        if batch.is_none() {
            *batch = Some(
                BatchScratch::new(gpu, cfg, *chunk_rows)
                    .map_err(|e| format!("qwen4_exp prefill scratch: {e:?}"))?,
            );
        }
        let b = batch.as_mut().expect("just allocated");
        prefill_chunk_into(
            gpu, cfg, weights, state, scratch, b, embed, rows, history, pos0, *eos, all_logits,
            None,
        )
        .map_err(|e| format!("qwen4_exp prefill: {e:?}"))
    }

    /// Attach an MTP head (read from `sidecar`, a `<Model>--mtp.<quant>.hfq`).
    /// From then on `decode_step` speculates: see [`Self::spec_decode_step`].
    pub fn attach_mtp(&mut self, gpu: &mut Gpu, sidecar: &std::path::Path) -> Result<(), String> {
        let mut hfq = HfqFile::open(sidecar)
            .map_err(|e| format!("qwen4_exp mtp: open {}: {e}", sidecar.display()))?;
        let head = {
            let reader = HfqTensorReader { hfq: &mut hfq };
            crate::mtp_gpu::MtpWeightsGpu::upload(gpu, &self.cfg, &reader)
                .map_err(|e| format!("qwen4_exp mtp: upload head: {e:?}"))?
        };
        self.attach_mtp_weights(gpu, head)
    }

    /// [`Self::attach_mtp`] for a head already on the GPU (tests build one from
    /// synthetic weights).
    pub fn attach_mtp_weights(
        &mut self,
        gpu: &mut Gpu,
        head: crate::mtp_gpu::MtpWeightsGpu,
    ) -> Result<(), String> {
        let e = |e: hipfire_rdna::HipError| format!("qwen4_exp mtp state: {e:?}");
        let cfg = &self.cfg;
        let state = MtpState {
            head,
            scratch: crate::mtp_gpu::MtpScratchGpu::new(gpu, cfg, self.max_seq).map_err(e)?,
            cache: crate::attn_gpu::QsaCache::new(gpu, cfg, self.max_seq).map_err(e)?,
            logits: gpu
                .zeros(&[cfg.vocab], hipfire_rdna::DType::F32)
                .map_err(e)?,
            hb: BatchScratch::new(gpu, cfg, self.chunk_rows).map_err(e)?,
            vb: BatchScratch::new(gpu, cfg, crate::prefill_gpu::Row0Checkpoint::ROWS).map_err(e)?,
            ckpt: crate::prefill_gpu::Row0Checkpoint::new(gpu, cfg, &self.state).map_err(e)?,
            pending: None,
            script: None,
            stats: MtpStats::default(),
        };
        if let Some(old) = self.mtp.replace(Box::new(state)) {
            old.free(gpu);
        }
        Ok(())
    }

    /// Speculation counters, when a head is attached.
    pub fn mtp_stats(&self) -> Option<MtpStats> {
        self.mtp.as_ref().map(|m| m.stats)
    }

    /// TEST HOOK: draft `script[pos]` for absolute position `pos` instead of
    /// asking the head. A script of the true continuation makes every verify a
    /// hit, a wrong one makes every verify a miss — the two paths a random
    /// test head would otherwise exercise only by chance.
    #[doc(hidden)]
    pub fn set_draft_script(&mut self, script: Option<Vec<u32>>) {
        if let Some(m) = self.mtp.as_mut() {
            m.script = script;
        }
    }

    /// Write the head's cache over the prompt rows `pos0 .. pos0+n` of the
    /// chunk just run. Row t pairs the trunk's wide state at t with the
    /// embedding of token t+1, so the LAST prompt row is left for the first
    /// draft, which is the first point that token is known.
    fn mtp_prefill_rows(&mut self, gpu: &mut Gpu, pos0: usize, n: usize) -> Result<(), String> {
        let total = self.history.len();
        let k = n.min(total.saturating_sub(1).saturating_sub(pos0));
        let Some(m) = self.mtp.as_mut() else {
            return Ok(());
        };
        if k == 0 {
            return Ok(());
        }
        let width = self.cfg.gated_residual.count * self.cfg.hidden;
        // A one-row chunk took the decode path, which leaves its wide state in
        // the trunk scratch rather than the batch scratch.
        let wide = match (n, self.batch.as_ref()) {
            (1, _) | (_, None) => self.scratch.wide().sub_offset(0, width),
            (_, Some(b)) => b.wide_rows(&self.cfg, k),
        };
        let mut embeds = Vec::with_capacity(k * self.cfg.hidden);
        for &t in &self.history[pos0 + 1..pos0 + 1 + k] {
            let h = self.cfg.hidden;
            embeds.extend_from_slice(&self.embed[t as usize * h..(t as usize + 1) * h]);
        }
        crate::prefill_gpu::mtp_cache_rows(
            gpu,
            &self.cfg,
            &m.head,
            &m.hb,
            &mut m.cache,
            &wide,
            &embeds,
            pos0,
            k,
            MTP_FUSION,
        )
        .map_err(|e| format!("qwen4_exp mtp prefill: {e:?}"))
    }

    /// One generated token with speculation. Exact under ANY sampler: every
    /// logit row the caller sees comes from the trunk over the true history.
    ///
    /// * VERIFY — draft `d` for `pos+1` from the head, run `[token, d]` as one
    ///   2-row forward, hand back row 0's logits, keep row 1 pending.
    /// * HIT — the caller's next token IS `d`: row 1 already ran, so its logits
    ///   are ready and this step costs no trunk forward at all.
    /// * MISS — restore the state checkpointed after row 0, then verify the
    ///   real token.
    fn spec_decode_step(&mut self, gpu: &mut Gpu, token: u32, pos: usize) -> Result<(), String> {
        let vocab = self.cfg.vocab;
        let width = self.cfg.gated_residual.count * self.cfg.hidden;
        let hidden = self.cfg.hidden;
        let ge = |e: hipfire_rdna::HipError| format!("qwen4_exp mtp: {e:?}");
        let m = self.mtp.as_mut().expect("spec_decode_step needs a head");

        if let Some(p) = m.pending.take() {
            if p.pos == pos && p.draft == token {
                m.stats.hits += 1;
                let lg = m.vb.all_logits().expect("a verify leaves logits");
                gpu.memcpy_dtod_auto(
                    &self.scratch.logits().buf,
                    &lg.sub_offset(vocab, vocab).buf,
                    vocab * 4,
                )
                .map_err(ge)?;
                // The head never drafted from row 0's position; fill its cache
                // there so the head's attention stays complete.
                let row0 = m.vb.wide_row(&self.cfg, 0);
                let e = self.embed[token as usize * hidden..(token as usize + 1) * hidden].to_vec();
                return crate::prefill_gpu::mtp_cache_rows(
                    gpu,
                    &self.cfg,
                    &m.head,
                    &m.hb,
                    &mut m.cache,
                    &row0,
                    &e,
                    pos - 1,
                    1,
                    MTP_FUSION,
                )
                .map_err(|e| format!("qwen4_exp mtp fill: {e:?}"));
            }
            m.stats.misses += 1;
            m.ckpt.restore(gpu, &mut self.state, pos).map_err(ge)?;
            self.history.truncate(pos);
            gpu.memcpy_dtod_auto(
                &self.scratch.wide().buf,
                &m.vb.wide_row(&self.cfg, 0).buf,
                width * 4,
            )
            .map_err(ge)?;
        }

        // Draft pos+1 from the head at pos-1: the trunk's wide state there plus
        // the embedding of `token`, exactly the pairing the head was trained on.
        let draft = match m.script.as_ref() {
            Some(s) => s.get(pos + 1).copied().unwrap_or(0),
            None => {
                let g_emb = gpu
                    .upload_f32(
                        &self.embed[token as usize * hidden..(token as usize + 1) * hidden],
                        &[hidden],
                    )
                    .map_err(ge)?;
                let r = crate::mtp_gpu::mtp_decode_step(
                    gpu,
                    &self.cfg,
                    &m.head,
                    &mut m.cache,
                    &mut m.scratch,
                    self.scratch.wide(),
                    &g_emb,
                    pos - 1,
                    MTP_FUSION,
                );
                gpu.free_tensor(g_emb).map_err(ge)?;
                r.map_err(ge)?;
                hipfire_runtime::weights::weight_gemv(
                    gpu,
                    &self.weights.lm_head,
                    &m.scratch.collapsed,
                    &m.logits,
                )
                .map_err(ge)?;
                let lg = gpu.download_f32(&m.logits).map_err(ge)?;
                lg.iter()
                    .enumerate()
                    .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &v)| {
                        if v > bv {
                            (i, v)
                        } else {
                            (bi, bv)
                        }
                    })
                    .0 as u32
            }
        };
        if m.script.is_some() {
            // The scripted path skips the head, so keep its cache complete here.
            let e = self.embed[token as usize * hidden..(token as usize + 1) * hidden].to_vec();
            let wide = self.scratch.wide().sub_offset(0, width);
            crate::prefill_gpu::mtp_cache_rows(
                gpu,
                &self.cfg,
                &m.head,
                &m.hb,
                &mut m.cache,
                &wide,
                &e,
                pos - 1,
                1,
                MTP_FUSION,
            )
            .map_err(|e| format!("qwen4_exp mtp fill: {e:?}"))?;
        }
        m.stats.verifies += 1;
        m.pending = Some(Pending {
            pos: pos + 1,
            draft,
        });
        self.history.push(token);
        self.history.push(draft);
        self.run_rows(gpu, pos, Rows::Verify)?;
        // The caller wants THIS position's logits: row 0.
        let m = self.mtp.as_ref().expect("still attached");
        let lg = m.vb.all_logits().expect("a verify leaves logits");
        gpu.memcpy_dtod_auto(
            &self.scratch.logits().buf,
            &lg.sub_offset(0, vocab).buf,
            vocab * 4,
        )
        .map_err(ge)
    }

    /// Drop speculation state at a sequence boundary.
    fn mtp_reset(&mut self) {
        if let Some(m) = self.mtp.as_mut() {
            m.pending = None;
            m.cache.reset();
        }
    }

    /// Override the prefill chunk size (capped at `PREFILL_CHUNK`). For tests: a
    /// small chunk puts a chunk boundary inside a short prompt.
    pub fn set_prefill_chunk(&mut self, gpu: &mut Gpu, rows: usize) {
        self.chunk_rows = rows.clamp(1, PREFILL_CHUNK);
        if let Some(b) = self.batch.take() {
            b.free(gpu);
        }
    }

    /// Restart the sequence and run `tokens` from position 0, in chunks.
    /// `at_chunk(pos0, n, backend)` runs after each chunk — the scorer reads the
    /// chunk's logits there, before the next chunk overwrites them.
    fn prefill_chunks(
        &mut self,
        gpu: &mut Gpu,
        tokens: &[u32],
        all_logits: bool,
        at_chunk: &mut dyn FnMut(&mut Gpu, usize, usize, &Self) -> Result<(), String>,
    ) -> Result<(), String> {
        if tokens.is_empty() {
            return Err("qwen4_exp prefill: empty prompt".to_string());
        }
        // Replay from scratch: the recurrent state is a function of the whole
        // prefix, so a prefill must start from a reset trunk, not from whatever
        // the previous request left behind.
        self.history.clear();
        self.mtp_reset();
        self.state
            .reset(gpu)
            .map_err(|e| format!("qwen4_exp reset: {e:?}"))?;
        let chunk = if per_token_prefill() && !all_logits {
            1
        } else {
            self.chunk_rows
        };
        // The head's prompt cache pairs row t with token t+1, so the whole
        // prompt must be in `history` before a chunk's head rows are written.
        self.history.extend_from_slice(tokens);
        for (c, part) in tokens.chunks(chunk).enumerate() {
            let pos0 = c * chunk;
            let end = pos0 + part.len();
            // The trunk sees exactly the history up to this chunk's end.
            let rest = self.history.split_off(end);
            let r = self.forward(gpu, pos0, all_logits);
            self.history.extend(rest);
            r?;
            if !all_logits {
                self.mtp_prefill_rows(gpu, pos0, part.len())?;
            }
            at_chunk(gpu, pos0, part.len(), self)?;
        }
        Ok(())
    }
}

impl SimpleAr for Qwen4ExpBackend {
    fn prefill(&mut self, gpu: &mut Gpu, tokens: &[u32]) -> Result<(), String> {
        self.prefill_chunks(gpu, tokens, false, &mut |_, _, _, _| Ok(()))
    }

    fn decode_step(&mut self, gpu: &mut Gpu, token: u32, pos: usize) -> Result<(), String> {
        // Speculate when a head is attached and the verify's second row fits.
        if self.mtp.is_some() && pos >= 1 && pos + 1 < self.max_seq {
            return self.spec_decode_step(gpu, token, pos);
        }
        if let Some(m) = self.mtp.as_mut() {
            // Out of room to speculate: a pending draft can no longer be
            // honoured, so undo it before taking the plain step.
            if let Some(p) = m.pending.take() {
                m.ckpt
                    .restore(gpu, &mut self.state, p.pos)
                    .map_err(|e| format!("qwen4_exp mtp: {e:?}"))?;
                self.history.truncate(p.pos);
                if !(p.pos == pos && p.draft == token) {
                    m.stats.misses += 1;
                } else {
                    m.stats.hits += 1;
                }
            }
        }
        self.history.push(token);
        self.forward(gpu, pos, false)
    }

    fn logits(&self) -> &GpuTensor {
        self.scratch.logits()
    }

    fn vocab_size(&self) -> usize {
        self.cfg.vocab
    }
}

impl ServingBackend for Qwen4ExpBackend {
    fn arch_id(&self) -> u32 {
        ARCH_ID_QWEN4EXP
    }

    fn caps(&self) -> ArchCaps {
        // Everything off by design: no DFlash drafter, no CASK, no VL or MTP
        // serving yet. Prefill IS batched, but inside `SimpleAr::prefill` — the
        // arch-api `batched_prefill` seam (cross-request batching) is not
        // implemented, so it is not claimed. `docs/model-support.toml` says the same, and a
        // capability claimed here that the trunk cannot honour is worse than one
        // that is simply absent.
        ArchCaps {
            mtp: self.mtp.is_some(),
            ..ArchCaps::default()
        }
    }

    fn eos_token(&self) -> u32 {
        self.eos
    }

    fn serve(
        &mut self,
        gpu: &mut Gpu,
        tok: &Tokenizer,
        ctx: &mut GenerateCtx,
    ) -> Result<ServeOutcome, String> {
        let eos = self.eos;
        run_simple_ar(gpu, self, tok, eos, ctx)
    }

    fn expert_pager_stats(&self) -> Option<hipfire_runtime::weight_pager::ModulePagerStats> {
        Qwen4ExpBackend::expert_pager_stats(self)
    }

    /// Makes arch 26 scorable by the daemon's `kld_eval` (the multi-arch PPL/KLD
    /// path `hipfire eval` drives). Without this the daemon reported "this
    /// architecture cannot produce logits" for a model that decodes fine.
    fn kld_forward(&mut self) -> Option<&mut dyn hipfire_runtime::kld_eval::ChunkScoredForward> {
        if per_token_prefill() {
            // The blanket `SimpleAr` walk: one decode step per position. Kept as
            // the A/B reference for the batched scorer below.
            return Some(self);
        }
        // SAFETY: `BatchedScorer` is `#[repr(transparent)]` over this exact type,
        // so the two references have the same layout and the same lifetime.
        Some(unsafe { &mut *(self as *mut Self as *mut BatchedScorer) })
    }

    fn reset_session(&mut self, gpu: &mut Gpu, _session_id: &str) -> Result<(), String> {
        self.history.clear();
        self.mtp_reset();
        self.state
            .reset(gpu)
            .map_err(|e| format!("qwen4_exp reset: {e:?}"))
    }

    fn unload(self: Box<Self>, gpu: &mut Gpu) {
        let m = *self;
        m.weights.free(gpu);
        m.state.free(gpu);
        m.scratch.free(gpu);
        if let Some(b) = m.batch {
            b.free(gpu);
        }
        if let Some(s) = m.mtp {
            s.free(gpu);
        }
        // The expert stacks deliberately free nothing — the pager owns those
        // buffers, and it outlives any one of them.
        if let Some(pager) = m.pager {
            if let Ok(mut p) = pager.lock() {
                p.free_all(gpu);
            }
        }
    }
}

/// Teacher-forced scoring through the BATCHED forward: every row's logits from
/// one layer-major pass per chunk, instead of the blanket `SimpleAr` walk's one
/// decode step per position.
///
/// A newtype because the blanket `impl<T: SimpleAr> ChunkScoredForward for T`
/// already covers the backend itself and cannot be specialised.
#[repr(transparent)]
pub struct BatchedScorer(Qwen4ExpBackend);

impl hipfire_runtime::kld_eval::ChunkScoredForward for BatchedScorer {
    fn forward_chunk_scored(
        &mut self,
        gpu: &mut Gpu,
        chunk: &[u32],
        scoring_start: usize,
        at_scored: &mut dyn FnMut(&hipfire_runtime::kld_eval::ScoredWindow<'_>),
    ) -> Result<(), String> {
        let n = chunk.len();
        if n < 2 {
            return Ok(());
        }
        let vocab = self.0.cfg.vocab;
        // Row p predicts chunk[p + 1], so the last token is a label only.
        self.0
            .prefill_chunks(gpu, &chunk[..n - 1], true, &mut |gpu, pos0, rows, be| {
                let (lo, hi) = (scoring_start.max(pos0), pos0 + rows);
                if lo >= hi {
                    return Ok(());
                }
                let all = be
                    .batch
                    .as_ref()
                    .and_then(|b| b.all_logits())
                    .ok_or("qwen4_exp scorer: batched forward left no logits")?;
                let view = all.sub_offset((lo - pos0) * vocab, (hi - lo) * vocab);
                let logits = gpu
                    .download_f32(&view)
                    .map_err(|e| format!("qwen4_exp scorer: download logits: {e:?}"))?;
                at_scored(&hipfire_runtime::kld_eval::ScoredWindow {
                    j0: lo - scoring_start,
                    logits: &logits,
                    vocab,
                    nexts: &chunk[lo + 1..hi + 1],
                });
                Ok(())
            })
    }

    fn kld_vocab_size(&self) -> usize {
        self.0.cfg.vocab
    }

    fn final_position_logits(&mut self, gpu: &mut Gpu, tokens: &[u32]) -> Result<Vec<f32>, String> {
        self.0.prefill(gpu, tokens)?;
        gpu.download_f32(self.0.logits())
            .map_err(|e| format!("qwen4_exp final logits: {e:?}"))
    }
}

/// The MTP sidecar for `model`: a sibling `<Model>--mtp.<quant>.hfq`, where
/// `<Model>` is the model's human-readable name (left of its `--`). Several
/// quants of the head resolve to the lexically first, so the pick is stable.
pub fn mtp_sidecar(model: &std::path::Path) -> Option<std::path::PathBuf> {
    let name = model.file_name()?.to_str()?;
    let stem = name.split_once("--")?.0;
    let prefix = format!("{stem}--mtp.");
    let mut found: Vec<std::path::PathBuf> = std::fs::read_dir(model.parent()?)
        .ok()?
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with(&prefix) && n.ends_with(".hfq"))
        })
        .collect();
    found.sort();
    found.into_iter().next()
}

pub struct Qwen4ExpServingFactory;

pub static QWEN4EXP_SERVING_FACTORY: Qwen4ExpServingFactory = Qwen4ExpServingFactory;

impl ServingFactory for Qwen4ExpServingFactory {
    fn arch_id(&self) -> u32 {
        Qwen4Exp::arch_id()
    }

    fn family(&self) -> &'static str {
        Qwen4Exp::name()
    }

    fn load(
        &self,
        hfq: &mut HfqFile,
        gpu: &mut Gpu,
        options: &ServingFactoryOptions<'_>,
    ) -> Result<FactoryLoadedBackend, String> {
        let max_seq = options.max_seq;
        let mut backend = Qwen4ExpBackend::load(gpu, hfq, max_seq)?;
        // `auto` does NOT attach a found head here: speculation is exact but
        // measured SLOWER on gfx1151 (2026-11-27, 180B oq4: 70.7% acceptance,
        // 1.71 tokens/verify, yet 7.24 -> 5.95 tok/s) because a 2-row verify
        // costs 2.08 decode steps. Only an explicit `on` opts in until the
        // verify is cheaper than the 1.71x break-even.
        match (options.mtp_mode, mtp_sidecar(hfq.path())) {
            ("on", Some(side)) => {
                backend.attach_mtp(gpu, &side)?;
                eprintln!("qwen4_exp: MTP speculative decode ON ({})", side.display());
            }
            ("on", None) => {
                return Err(format!(
                    "mtp_mode=on but no MTP sidecar next to {} (expected <Model>--mtp.<quant>.hfq)",
                    hfq.path().display()
                ));
            }
            ("auto", Some(side)) => eprintln!(
                "qwen4_exp: MTP sidecar {} found but NOT enabled under mtp_mode=auto \
                 (measured slower than plain decode on this family); set mtp_mode=on to use it",
                side.display()
            ),
            _ => {}
        }
        let cfg = backend.config();
        let shape = ModelShapeProfile {
            hidden_size: cfg.hidden,
            num_layers: cfg.layers,
            vocab_size: cfg.vocab,
            intermediate_size: cfg.moe.intermediate,
        };
        let profile = PromptGenerationProfile {
            output_protocol: OutputProtocol::Plain,
            ..Default::default()
        };
        let physical_cap = max_seq.min(cfg.max_position).max(1);
        let _ = (options.kv_mode, options.triattn);
        Ok(FactoryLoadedBackend {
            backend: Box::new(backend),
            family: self.family(),
            shape,
            profile,
            physical_cap,
        })
    }
}

hipfire_runtime::register_serving_factory!(QWEN4EXP_SERVING_FACTORY);

#[cfg(test)]
mod tests {
    use super::mtp_sidecar;

    #[test]
    fn mtp_sidecar_is_the_canonical_role_sibling() {
        let dir = std::env::temp_dir().join(format!("q4x-mtp-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let touch = |n: &str| std::fs::write(dir.join(n), b"").unwrap();
        touch("Flash-180B--oq4.hfq");
        touch("Flash-180B--oq4.25++.hfq");
        touch("Other-9B--mtp.oq8.hfq"); // another model's head
        touch("Flash-180B.mtp.hfq"); // legacy dotted form: not ours to read
        let base = dir.join("Flash-180B--oq4.hfq");
        assert_eq!(mtp_sidecar(&base), None, "no canonical sidecar yet");
        touch("Flash-180B--mtp.oq8.hfq");
        touch("Flash-180B--mtp.oq4.hfq");
        assert_eq!(
            mtp_sidecar(&base),
            Some(dir.join("Flash-180B--mtp.oq4.hfq"))
        );
        // A dotted-name model has no `--` boundary, so nothing is guessed.
        assert_eq!(mtp_sidecar(&dir.join("Flash-180B.oq4.hfq")), None);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
