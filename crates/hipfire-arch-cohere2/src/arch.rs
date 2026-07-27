// SPDX-License-Identifier: Apache-2.0
//! Boxed autoregressive serving backend for Cohere2-MoE/BLS (arch 25).

use crate::calibration_stream::{
    forward_resident_token, Cohere2ResidentState, Cohere2ResidentWeights,
};
use crate::config::Cohere2Config;
use hipfire_rdna::{Gpu, GpuTensor};
use hipfire_runtime::arch::{
    run_simple_ar, ArchCaps, FactoryLoadedBackend, GenerateCtx, ModelShapeProfile,
    PromptFrameOverrides, PromptGenerationProfile, SamplerOverrides, ServeOutcome, ServingBackend,
    ServingFactory, ServingFactoryOptions, SimpleAr,
};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::tokenizer::Tokenizer;
use hipfire_runtime::triattn::LayeredEvictionCtx;

fn eos_token_from_hfq(hfq: &HfqFile) -> u32 {
    serde_json::from_str::<serde_json::Value>(&hfq.metadata_json)
        .ok()
        .and_then(|metadata| {
            metadata
                .get("generation_config")
                .and_then(|generation| generation.get("eos_token_id"))
                .or_else(|| {
                    metadata
                        .get("config")
                        .and_then(|config| config.get("eos_token_id"))
                })
                .or_else(|| metadata.get("eos_token_id"))
                .and_then(serde_json::Value::as_u64)
        })
        .map(|value| value as u32)
        .unwrap_or(255_001)
}

fn has_official_chat_template(hfq: &HfqFile) -> bool {
    let Ok(metadata) = serde_json::from_str::<serde_json::Value>(&hfq.metadata_json) else {
        return false;
    };
    metadata
        .get("tokenizer_config")
        .and_then(|value| {
            value.as_object().cloned().or_else(|| {
                value.as_str().and_then(|encoded| {
                    serde_json::from_str::<serde_json::Map<String, serde_json::Value>>(encoded).ok()
                })
            })
        })
        .and_then(|config| config.get("chat_template").cloned())
        .is_some_and(|template| !template.is_null())
}

pub struct Cohere2Backend {
    config: Cohere2Config,
    weights: Cohere2ResidentWeights,
    state: Cohere2ResidentState,
    eos_token: u32,
    eviction: Option<LayeredEvictionCtx>,
}

impl Cohere2Backend {
    fn maybe_evict(&mut self, gpu: &mut Gpu) -> Result<(), String> {
        if let Some(eviction) = &self.eviction {
            self.state
                .maybe_evict(gpu, eviction)
                .map_err(|error| format!("Cohere2 CASK eviction: {error}"))?;
        }
        Ok(())
    }
}

impl SimpleAr for Cohere2Backend {
    fn prefill(&mut self, gpu: &mut Gpu, tokens: &[u32]) -> Result<(), String> {
        self.state.reset();
        for &token in tokens {
            forward_resident_token(gpu, &self.config, &self.weights, &mut self.state, token)
                .map_err(|error| format!("Cohere2 prefill: {error}"))?;
            self.maybe_evict(gpu)?;
        }
        Ok(())
    }

    fn decode_step(&mut self, gpu: &mut Gpu, token: u32, pos: usize) -> Result<(), String> {
        if pos != self.state.next_pos() {
            return Err(format!(
                "Cohere2 decode position {pos} != cache cursor {}",
                self.state.next_pos()
            ));
        }
        forward_resident_token(gpu, &self.config, &self.weights, &mut self.state, token)
            .map_err(|error| format!("Cohere2 decode: {error}"))?;
        self.maybe_evict(gpu)
    }

    fn logits(&self) -> &GpuTensor {
        self.state.logits()
    }

    fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }
}

impl ServingBackend for Cohere2Backend {
    fn arch_id(&self) -> u32 {
        25
    }

    fn caps(&self) -> ArchCaps {
        ArchCaps::default()
    }

    fn eos_token(&self) -> u32 {
        self.eos_token
    }

    fn serve(
        &mut self,
        gpu: &mut Gpu,
        tok: &Tokenizer,
        ctx: &mut GenerateCtx,
    ) -> Result<ServeOutcome, String> {
        let eos_token = self.eos_token;
        run_simple_ar(gpu, self, tok, eos_token, ctx)
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

pub struct Cohere2ServingFactory;

pub static COHERE2_SERVING_FACTORY: Cohere2ServingFactory = Cohere2ServingFactory;

impl ServingFactory for Cohere2ServingFactory {
    fn arch_id(&self) -> u32 {
        25
    }

    fn family(&self) -> &'static str {
        "cohere2-moe"
    }

    fn load(
        &self,
        hfq: &mut HfqFile,
        gpu: &mut Gpu,
        options: &ServingFactoryOptions<'_>,
    ) -> Result<FactoryLoadedBackend, String> {
        let config = Cohere2Config::from_json_str(&hfq.metadata_json)?;
        let default_cap = std::env::var("HIPFIRE_COHERE2_MAX_SEQ")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|&value| value > 0)
            .unwrap_or(2048);
        let requested_max = options.max_seq.min(config.max_position_embeddings);
        let (logical_max, physical_cap) = if options.triattn.is_some() {
            let physical = options
                .physical_cap
                .ok_or("Cohere2 CASK load requires an explicit physical KV capacity")?;
            (requested_max, physical)
        } else {
            let bounded = requested_max.min(default_cap);
            (bounded, bounded)
        };
        if logical_max == 0 || physical_cap == 0 || physical_cap > logical_max {
            return Err("Cohere2 physical context must be nonzero".into());
        }
        let eos_token = eos_token_from_hfq(hfq);
        let instruction_tuned = has_official_chat_template(hfq);
        let weights = Cohere2ResidentWeights::load(hfq, gpu, &config)
            .map_err(|error| format!("Cohere2 weights: {error}"))?;
        let state = match Cohere2ResidentState::new(gpu, &config, logical_max, physical_cap) {
            Ok(state) => state,
            Err(error) => {
                weights.free_gpu(gpu);
                return Err(format!("Cohere2 state: {error}"));
            }
        };
        let eviction_result = options
            .triattn
            .map(|artifact| {
                if artifact.metadata.model_arch_id != 25
                    || artifact.metadata.model_layers as usize != config.num_hidden_layers
                {
                    return Err(format!(
                        "Cohere2 CASK identity mismatch: arch={} layers={}, expected arch=25 layers={}",
                        artifact.metadata.model_arch_id,
                        artifact.metadata.model_layers,
                        config.num_hidden_layers,
                    ));
                }
                state.build_eviction(
                    gpu,
                    artifact,
                    options.cask_budget,
                    options.cask_beta,
                )
            })
            .transpose();
        let eviction = match eviction_result {
            Ok(eviction) => eviction,
            Err(error) => {
                state.free_gpu(gpu);
                weights.free_gpu(gpu);
                return Err(error);
            }
        };
        let shape = ModelShapeProfile {
            hidden_size: config.hidden_size,
            num_layers: config.num_hidden_layers,
            vocab_size: config.vocab_size,
            intermediate_size: config.expert_intermediate,
        };
        Ok(FactoryLoadedBackend {
            backend: Box::new(Cohere2Backend {
                config,
                weights,
                state,
                eos_token,
                eviction,
            }),
            family: self.family(),
            shape,
            profile: PromptGenerationProfile {
                prompt: PromptFrameOverrides {
                    raw: Some(!instruction_tuned),
                },
                sampler: SamplerOverrides {
                    temperature: Some(0.6),
                    top_p: Some(0.95),
                    top_k: Some(50),
                    ..SamplerOverrides::default()
                },
                bos_token: Some("<BOS_TOKEN>"),
                require_official_template: instruction_tuned,
                ..PromptGenerationProfile::default()
            },
            physical_cap,
        })
    }
}

hipfire_runtime::register_serving_factory!(COHERE2_SERVING_FACTORY);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn factory_identity_is_stable() {
        assert_eq!(COHERE2_SERVING_FACTORY.arch_id(), 25);
        assert_eq!(COHERE2_SERVING_FACTORY.family(), "cohere2-moe");
    }
}
