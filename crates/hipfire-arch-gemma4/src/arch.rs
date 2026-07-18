// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! Runtime factory and boxed dense-text serving backend for Gemma 4.

use crate::config::Gemma4Config;
use crate::forward::{forward_step, lower_dense_forward, Gemma4DenseState};
use crate::weights::{load_dense_weights, Gemma4DenseWeights};
use hipfire_dispatch::pipeline::superop::LoweredForward;
use hipfire_rdna::{Gpu, GpuTensor};
use hipfire_runtime::arch::{
    run_simple_ar_with_terminators, ArchCaps, Architecture, EosFilterOverrides,
    FactoryLoadedBackend, GenerateCtx, ModelShapeProfile, OutputProtocol, PromptFrameOverrides,
    PromptGenerationProfile, SamplerOverrides, ServeOutcome, ServingBackend, ServingFactory,
    ServingFactoryOptions, SimpleAr,
};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::kv::KvQuantMode;
use hipfire_runtime::tokenizer::Tokenizer;

/// Map an operator `kv_mode` string to the gemma4 KV cache mode + KVarN bit width.
/// gemma4 wires KVarN (variance-normalized K + Q8 V) on the global/full-context
/// layers at `kvarn2`/`kvarn`(=4)/`kvarn8` (head_dim 256; local sliding-window
/// layers stay F32). Q8 KV is intentionally not offered (deprecated per the mq*/Q8
/// direction); every other token → unquantized F32. The `usize` is the KVarN K bit
/// width (meaningful only for the Kvarn mode).
fn gemma4_kv_mode(kv_mode: &str) -> (KvQuantMode, usize) {
    match kv_mode {
        "kvarn2" => (KvQuantMode::Kvarn, 2),
        "kvarn" | "kvarn4" => (KvQuantMode::Kvarn, 4),
        "kvarn8" => (KvQuantMode::Kvarn, 8),
        _ => (KvQuantMode::Unquantized, 4),
    }
}

pub struct Gemma4;

fn config_from_hfq(hfq: &HfqFile) -> Result<Gemma4Config, String> {
    let metadata: serde_json::Value = serde_json::from_str(&hfq.metadata_json)
        .map_err(|error| format!("Gemma 4 metadata JSON: {error}"))?;
    Gemma4Config::from_value(metadata.get("config").unwrap_or(&metadata))
}

fn generation_eos_ids(metadata: &serde_json::Value) -> Vec<u32> {
    let config = metadata.get("config").unwrap_or(metadata);
    let text = config.get("text_config").unwrap_or(config);
    let value = metadata
        .get("generation_config")
        .and_then(|generation| generation.get("eos_token_id"))
        .or_else(|| config.get("eos_token_id"))
        .or_else(|| text.get("eos_token_id"));
    let ids = match value {
        Some(serde_json::Value::Array(values)) => values
            .iter()
            .filter_map(|value| value.as_u64().map(|value| value as u32))
            .collect(),
        Some(value) => value
            .as_u64()
            .map(|value| vec![value as u32])
            .unwrap_or_default(),
        None => Vec::new(),
    };
    if ids.is_empty() {
        vec![1]
    } else {
        ids
    }
}

pub fn generation_eos_ids_from_hfq(hfq: &HfqFile) -> Vec<u32> {
    serde_json::from_str::<serde_json::Value>(&hfq.metadata_json)
        .map(|metadata| generation_eos_ids(&metadata))
        .unwrap_or_else(|_| vec![1])
}

fn bounded_physical_cap(
    requested: usize,
    trained: usize,
    bringup_limit: usize,
) -> Result<usize, String> {
    let cap = requested.min(trained).min(bringup_limit);
    if cap == 0 {
        return Err("Gemma 4 physical context must be nonzero".to_string());
    }
    Ok(cap)
}

impl Architecture for Gemma4 {
    type Weights = Gemma4DenseWeights;
    type State = Gemma4DenseState;
    type Config = Gemma4Config;

    fn arch_id() -> u32 {
        24
    }

    fn name() -> &'static str {
        "gemma4"
    }

    fn config_from_hfq(hfq: &HfqFile) -> Result<Self::Config, String> {
        config_from_hfq(hfq)
    }

    fn load_weights(
        hfq: &mut HfqFile,
        cfg: &Self::Config,
        gpu: &mut Gpu,
    ) -> Result<Self::Weights, String> {
        load_dense_weights(hfq, gpu, cfg).map_err(|error| format!("Gemma 4 weights: {error:?}"))
    }

    fn new_state(gpu: &mut Gpu, cfg: &Self::Config) -> Result<Self::State, String> {
        let cap = cfg.max_position_embeddings.min(2048);
        Gemma4DenseState::new(gpu, cfg, cap).map_err(|error| format!("Gemma 4 state: {error:?}"))
    }

    fn prompt_frame_overrides(_cfg: &Self::Config) -> PromptFrameOverrides {
        PromptFrameOverrides::default()
    }

    fn eos_filter_overrides(_cfg: &Self::Config) -> EosFilterOverrides {
        EosFilterOverrides {
            stop_at: Vec::new(),
            holdback_prefixes: Vec::new(),
            strip_think: Some(false),
        }
    }

    fn sampler_overrides(_cfg: &Self::Config) -> SamplerOverrides {
        SamplerOverrides {
            temperature: Some(1.0),
            top_p: Some(0.95),
            top_k: Some(64),
            ..SamplerOverrides::default()
        }
    }
}

pub struct Gemma4Backend {
    config: Gemma4Config,
    weights: Gemma4DenseWeights,
    state: Gemma4DenseState,
    lowered: LoweredForward,
    eos_token_ids: Vec<u32>,
}

impl Gemma4Backend {
    pub fn new(
        config: Gemma4Config,
        weights: Gemma4DenseWeights,
        state: Gemma4DenseState,
        mut eos_token_ids: Vec<u32>,
    ) -> Self {
        if eos_token_ids.is_empty() {
            eos_token_ids.push(1);
        }
        let lowered = lower_dense_forward(&config, &state);
        Self {
            config,
            weights,
            state,
            lowered,
            eos_token_ids,
        }
    }
}

impl SimpleAr for Gemma4Backend {
    fn prefill(&mut self, gpu: &mut Gpu, tokens: &[u32]) -> Result<(), String> {
        self.state.reset();
        for &token in tokens {
            forward_step(
                gpu,
                &self.weights,
                &self.config,
                &mut self.state,
                &self.lowered,
                token,
            )
            .map_err(|error| format!("Gemma 4 prefill: {error:?}"))?;
        }
        Ok(())
    }

    fn decode_step(&mut self, gpu: &mut Gpu, token: u32, pos: usize) -> Result<(), String> {
        if pos != self.state.next_pos() {
            return Err(format!(
                "Gemma 4 decode position {pos} != cache cursor {}",
                self.state.next_pos()
            ));
        }
        forward_step(
            gpu,
            &self.weights,
            &self.config,
            &mut self.state,
            &self.lowered,
            token,
        )
        .map_err(|error| format!("Gemma 4 decode: {error:?}"))
    }

    fn logits(&self) -> &GpuTensor {
        self.state.logits_tensor()
    }

    fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }
}

impl ServingBackend for Gemma4Backend {
    fn arch_id(&self) -> u32 {
        Gemma4::arch_id()
    }

    fn caps(&self) -> ArchCaps {
        ArchCaps::default()
    }

    fn eos_token(&self) -> u32 {
        self.eos_token_ids[0]
    }

    fn serve(
        &mut self,
        gpu: &mut Gpu,
        tok: &Tokenizer,
        ctx: &mut GenerateCtx,
    ) -> Result<ServeOutcome, String> {
        // Copy the tiny metadata-declared set so `self` can be borrowed mutably
        // by the shared decode loop. Gemma 4 instruction checkpoints declare
        // `[1, 106, 50]`; none may leak into visible output.
        let terminators = self.eos_token_ids.clone();
        run_simple_ar_with_terminators(gpu, self, tok, &terminators, ctx)
    }

    fn reset_session(&mut self, _gpu: &mut Gpu, _session_id: &str) -> Result<(), String> {
        self.state.reset();
        Ok(())
    }

    fn unload(self: Box<Self>, gpu: &mut Gpu) {
        let backend = *self;
        backend.state.free_gpu(gpu);
        backend.weights.free_gpu(gpu);
    }

    fn kld_forward(&mut self) -> Option<&mut dyn hipfire_runtime::kld_eval::ChunkScoredForward> {
        Some(self)
    }
}

pub struct Gemma4ServingFactory;

pub static GEMMA4_SERVING_FACTORY: Gemma4ServingFactory = Gemma4ServingFactory;

impl ServingFactory for Gemma4ServingFactory {
    fn arch_id(&self) -> u32 {
        Gemma4::arch_id()
    }

    fn family(&self) -> &'static str {
        Gemma4::name()
    }

    fn load(
        &self,
        hfq: &mut HfqFile,
        gpu: &mut Gpu,
        options: &ServingFactoryOptions<'_>,
    ) -> Result<FactoryLoadedBackend, String> {
        let config = config_from_hfq(hfq)?;
        let default_cap = std::env::var("HIPFIRE_GEMMA4_MAX_SEQ")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|&value| value > 0)
            .unwrap_or(2048);
        let physical_cap =
            bounded_physical_cap(options.max_seq, config.max_position_embeddings, default_cap)?;
        let eos_token_ids = generation_eos_ids_from_hfq(hfq);
        let instruction_tuned = eos_token_ids.len() > 1;
        let weights = load_dense_weights(hfq, gpu, &config)
            .map_err(|error| format!("Gemma 4 weights: {error:?}"))?;
        let (kv_mode, kvarn_bits) = gemma4_kv_mode(options.kv_mode);
        let state =
            Gemma4DenseState::new_with_kv_mode(gpu, &config, physical_cap, kv_mode, kvarn_bits)
                .map_err(|error| format!("Gemma 4 state: {error:?}"))?;
        let shape = ModelShapeProfile {
            hidden_size: config.hidden_size,
            num_layers: config.num_hidden_layers,
            vocab_size: config.vocab_size,
            intermediate_size: config.intermediate_size,
        };
        let profile = PromptGenerationProfile {
            prompt: PromptFrameOverrides {
                raw: Some(!instruction_tuned),
            },
            sampler: Gemma4::sampler_overrides(&config),
            loop_guard: Gemma4::loop_guard_overrides(&config),
            eos_filter: Gemma4::eos_filter_overrides(&config),
            output_protocol: OutputProtocol::Gemma4Native,
            bos_token: Some("<bos>"),
            require_official_template: instruction_tuned,
        };
        Ok(FactoryLoadedBackend {
            backend: Box::new(Gemma4Backend::new(config, weights, state, eos_token_ids)),
            family: self.family(),
            shape,
            profile,
            physical_cap,
        })
    }
}

hipfire_runtime::register_serving_factory!(GEMMA4_SERVING_FACTORY);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn factory_identity_and_bounded_default() {
        assert_eq!(GEMMA4_SERVING_FACTORY.arch_id(), 24);
        assert_eq!(GEMMA4_SERVING_FACTORY.family(), "gemma4");
        assert_eq!(bounded_physical_cap(262_144, 262_144, 2048), Ok(2048));
        assert_eq!(bounded_physical_cap(1025, 262_144, 2048), Ok(1025));
        assert!(bounded_physical_cap(0, 262_144, 2048).is_err());
        assert_eq!(
            generation_eos_ids(&serde_json::json!({
                "generation_config": {"eos_token_id": [1, 106, 50]}
            })),
            vec![1, 106, 50]
        );
        assert_eq!(
            generation_eos_ids(&serde_json::json!({
                "generation_config": {"eos_token_id": 1}
            })),
            vec![1]
        );
    }
}
