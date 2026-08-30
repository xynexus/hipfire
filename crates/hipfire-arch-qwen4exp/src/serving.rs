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
//! ## What this seam does not do yet, and why it refuses rather than degrades
//!
//! Two tables are read on the HOST here, because `decode_step_into` takes them as
//! `&[f32]`:
//!
//! * the token embedding — `[vocab, hidden]`, 2.5 GB at the shipped geometry;
//! * the PLE n-gram table — **102 GB**, 41% of the model's parameters.
//!
//! The n-gram table must never be materialised: `crate::ngram_store` exists to
//! read single rows out of the sharded file, and the row indices are a pure
//! function of already-committed token ids, so the reads are issuable before the
//! forward starts. Until `decode_step_into` consumes that reader, this factory
//! REFUSES a model whose table exceeds a conservative budget instead of trying
//! and being OOM-killed. A named refusal at load is debuggable; a 128 GB machine
//! dying under page pressure is not.

use crate::arch::{load_ngram_table, HfqTensorReader, Qwen4Exp};
use crate::config::Qwen4ExpConfig;
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

/// Host bytes this bring-up seam is willing to spend on the n-gram table.
///
/// 8 GiB is far above any fixture and far below the shipped 102 GB, so it
/// separates "small artifact, load it" from "you need the streaming reader"
/// without needing to know which model this is.
const NGRAM_HOST_BUDGET_BYTES: u64 = 8 << 30;

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
    /// Present only for artifacts small enough to hold it (see the module note).
    ngram: Option<Vec<f32>>,
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
        let cfg = Qwen4Exp::config_from_hfq(hfq)?;
        Self::check_ngram_budget(&cfg)?;

        let weights = Qwen4Exp::load_weights(hfq, &cfg, gpu)?;
        let max_seq = max_seq.min(cfg.max_position).max(1);
        let state =
            TrunkState::new(gpu, &cfg, max_seq).map_err(|e| format!("qwen4_exp state: {e:?}"))?;
        let scratch = TrunkScratch::new(gpu, &cfg, max_seq)
            .map_err(|e| format!("qwen4_exp scratch: {e:?}"))?;

        let reader = HfqTensorReader { hfq };
        let embed = reader.read("model.language_model.embed_tokens.weight")?;
        let ngram = match cfg.ngram.as_ref() {
            Some(_) => Some(load_ngram_table(&reader, &cfg)?),
            None => None,
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

    /// Refuse a model whose n-gram table cannot be held on the host, BEFORE
    /// allocating anything, and say which table and how big.
    fn check_ngram_budget(cfg: &Qwen4ExpConfig) -> Result<(), String> {
        let Some(n) = cfg.ngram.as_ref() else {
            return Ok(());
        };
        let (_, _, padded) = hipfire_arch_qwen4exp_spec::ngram_head_layout(
            n.vocab_size_base,
            n.heads(),
            n.divisible_by,
        );
        let bytes = padded * n.head_dim() as u64 * 4;
        if bytes > NGRAM_HOST_BUDGET_BYTES {
            return Err(format!(
                "qwen4_exp: this model's n-gram table is {:.1} GiB at f32 ({padded} rows x {} \
                 dims), over the {} GiB host budget this bring-up seam allows. It is meant to be \
                 READ FROM DISK a row at a time (`crate::ngram_store`), which the trunk's decode \
                 does not consume yet. Serving the shipped checkpoint needs that wiring first — \
                 see docs/plans/2026-08-29-qwen4exp-flash-next-scope.md.",
                bytes as f64 / (1u64 << 30) as f64,
                n.head_dim(),
                NGRAM_HOST_BUDGET_BYTES >> 30,
            ));
        }
        Ok(())
    }

    pub fn config(&self) -> &Qwen4ExpConfig {
        &self.cfg
    }

    /// One trunk step at absolute position `pos`, leaving logits on the GPU.
    fn step(&mut self, gpu: &mut Gpu, pos: usize) -> Result<(), String> {
        decode_step_into(
            gpu,
            &self.cfg,
            &self.weights,
            &mut self.state,
            &mut self.scratch,
            &self.embed,
            self.ngram.as_deref(),
            &self.history,
            pos,
            self.eos,
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
