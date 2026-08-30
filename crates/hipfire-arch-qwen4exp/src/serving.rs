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

/// Above this, routed experts are fetched on demand rather than uploaded at load.
///
/// 16 GiB sits above anything that comfortably fits eagerly and far below the
/// shipped model's expert set, so it separates the two cases without naming a
/// model. It is a LOAD-TIME and FOOTPRINT choice, not a correctness one: both
/// residencies serve identical weights.
const LAZY_EXPERT_THRESHOLD_BYTES: u64 = 16 << 30;

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
    pub fn load_with_ngram_budget(
        gpu: &mut Gpu,
        hfq: &mut HfqFile,
        max_seq: usize,
        ngram_host_budget: u64,
    ) -> Result<Self, String> {
        let cfg = Qwen4Exp::config_from_hfq(hfq)?;
        // LAZY EXPERTS above a threshold. One token routes to `experts_per_tok` of
        // `num_experts`, so a short interaction needs a few percent of the expert
        // bytes; uploading all of them costs ~85 GiB and ~65 s on the shipped model
        // before the first token exists. Small models stay eager — the fetch
        // machinery is pure overhead when everything fits anyway.
        let expert_bytes = Self::routed_expert_bytes(&cfg);
        let lazy_experts = expert_bytes > LAZY_EXPERT_THRESHOLD_BYTES;
        if lazy_experts {
            eprintln!(
                "[qwen4_exp] routed experts are ~{:.1} GiB — loading them LAZILY \
                 ({} experts, top-{} per token)",
                expert_bytes as f64 / (1u64 << 30) as f64,
                cfg.moe.num_experts,
                cfg.moe.experts_per_tok,
            );
        }
        let reader0 = HfqTensorReader { hfq };
        let weights = TrunkWeights::upload_with(gpu, &cfg, &reader0, lazy_experts)
            .map_err(|e| format!("qwen4_exp: {e:?}"))?;
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
        })
    }

    /// Device bytes every routed expert would occupy, at the source encoding's
    /// own rate — 4-bit Opus is ~0.5 B/weight plus scales, so this is deliberately
    /// an over-estimate rather than a guess at the codec.
    fn routed_expert_bytes(cfg: &Qwen4ExpConfig) -> u64 {
        let (h, mi) = (cfg.hidden as u64, cfg.moe.intermediate as u64);
        let per_expert = (2 * mi * h + h * mi) / 2; // 4-bit
        per_expert * cfg.moe.num_experts as u64 * cfg.layers as u64
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
    pub fn routed_expert_dtype(&self) -> Option<hipfire_rdna::DType> {
        self.weights.layers.first().map(|l| l.moe.gate_up.dtype())
    }

    /// Expert elements actually uploaded so far, across every layer. For a lazy
    /// model this GROWS as tokens route to new experts, and is the number the
    /// lazy path exists to keep small.
    pub fn resident_expert_elems(&self) -> usize {
        self.weights
            .layers
            .iter()
            .map(|l| l.moe.gate_up.resident_elems() + l.moe.down.resident_elems())
            .sum()
    }

    /// One trunk step at absolute position `pos`, leaving logits on the GPU.
    ///
    /// `self` is destructured so the n-gram provider (which borrows `ngram`) and
    /// the mutable trunk state are disjoint borrows rather than two of `self`.
    fn step(&mut self, gpu: &mut Gpu, pos: usize) -> Result<(), String> {
        let Self {
            cfg,
            weights,
            state,
            scratch,
            embed,
            ngram,
            history,
            eos,
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
        decode_step_into(
            gpu, cfg, weights, state, scratch, embed, rows, history, pos, *eos,
        )
        .map_err(|e| format!("qwen4_exp decode: {e:?}"))
    }
}

impl SimpleAr for Qwen4ExpBackend {
    fn prefill(&mut self, gpu: &mut Gpu, tokens: &[u32]) -> Result<(), String> {
        if tokens.is_empty() {
            return Err("qwen4_exp prefill: empty prompt".to_string());
        }
        // Replay from scratch: the recurrent state is a function of the whole
        // prefix, so a prefill must start from a reset trunk, not from whatever
        // the previous request left behind.
        self.history.clear();
        self.state
            .reset(gpu)
            .map_err(|e| format!("qwen4_exp reset: {e:?}"))?;
        for (pos, &tok) in tokens.iter().enumerate() {
            self.history.push(tok);
            self.step(gpu, pos)?;
        }
        Ok(())
    }

    fn decode_step(&mut self, gpu: &mut Gpu, token: u32, pos: usize) -> Result<(), String> {
        self.history.push(token);
        self.step(gpu, pos)
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
        // Everything off by design: no batched prefill, no DFlash drafter, no
        // CASK, no VL wiring yet. `docs/model-support.toml` says the same, and a
        // capability claimed here that the trunk cannot honour is worse than one
        // that is simply absent.
        ArchCaps::default()
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

    fn reset_session(&mut self, gpu: &mut Gpu, _session_id: &str) -> Result<(), String> {
        self.history.clear();
        self.state
            .reset(gpu)
            .map_err(|e| format!("qwen4_exp reset: {e:?}"))
    }

    fn unload(self: Box<Self>, gpu: &mut Gpu) {
        let m = *self;
        m.weights.free(gpu);
        m.state.free(gpu);
        m.scratch.free(gpu);
    }
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
        let backend = Qwen4ExpBackend::load(gpu, hfq, max_seq)?;
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
