// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! Gemma 4 family-owned core tensor names over the shared loader mechanics.

use crate::config::{FfnPlan, Gemma4Config, ValueProjection};
use hip_bridge::HipResult;
use hipfire_rdna::{Gpu, GpuTensor};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::transformer_loader::TransformerLoader;
use hipfire_runtime::weights::{EmbeddingFormat, WeightTensor};

const EMBEDDING: &str = "model.language_model.embed_tokens.weight";
const FINAL_NORM: &str = "model.language_model.norm.weight";
const LM_HEAD: &str = "lm_head.weight";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Gemma4CoreShape {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub tie_word_embeddings: bool,
}

pub struct Gemma4CoreWeights {
    pub token_embd: GpuTensor,
    pub embd_format: EmbeddingFormat,
    /// The loader expands direct BF16 embeddings to F32 GPU storage. Preserve
    /// their source dtype so the forward can reproduce upstream's BF16-scaled
    /// embedding boundary instead of silently retaining extra F32 precision.
    pub embedding_source_bf16: bool,
    pub output_norm: GpuTensor,
    pub output: WeightTensor,
    pub tied_lm_head: bool,
}

pub struct Gemma4DenseLayerWeights {
    pub input_norm: GpuTensor,
    pub q_norm: GpuTensor,
    pub k_norm: GpuTensor,
    pub wq: WeightTensor,
    pub wk: WeightTensor,
    pub wv: Option<WeightTensor>,
    pub wo: WeightTensor,
    pub post_attn_norm: GpuTensor,
    pub pre_ffn_norm: GpuTensor,
    pub post_ffn_norm: GpuTensor,
    pub w_gate: WeightTensor,
    pub w_up: WeightTensor,
    pub w_down: WeightTensor,
    /// Source-precision scalar resolved once at load time.
    pub layer_scalar: f32,
    /// Per-Layer-Embedding merge weights (E2B/E4B); `None` when the model has no PLE
    /// (`hidden_size_per_layer_input == 0`).
    pub ple: Option<Gemma4PleLayerWeights>,
}

/// PLE per-layer merge weights: `h += post_norm(projection(act(gate·h) ⊙ ple[L]))`.
pub struct Gemma4PleLayerWeights {
    /// `[ple_dim, hidden]` — hidden → ple_dim gate.
    pub input_gate: WeightTensor,
    /// `[hidden, ple_dim]` — ple_dim → hidden projection.
    pub projection: WeightTensor,
    /// `[hidden]` RMSNorm.
    pub post_norm: GpuTensor,
}

/// PLE model-level weights: the per-layer embedding table + its projection from the
/// token embedding, computed once per token into `state.per_layer_inputs`.
pub struct Gemma4PleWeights {
    /// `[vocab, num_layers * ple_dim]` Q8 embedding table (row-lookup per token).
    pub embed_per_layer: WeightTensor,
    /// `[num_layers * ple_dim, hidden]` projection of the token embedding.
    pub model_projection: WeightTensor,
    /// `[ple_dim]` RMSNorm over each per-layer slice.
    pub projection_norm: GpuTensor,
    pub ple_dim: usize,
    pub num_layers: usize,
}

pub struct Gemma4DenseWeights {
    pub core: Gemma4CoreWeights,
    pub layers: Vec<Gemma4DenseLayerWeights>,
    /// `Some` for E2B/E4B (PLE models); `None` for plain-dense gemma4 (31B).
    pub ple: Option<Gemma4PleWeights>,
}

impl Gemma4CoreWeights {
    pub fn free_gpu(self, gpu: &mut Gpu) {
        let _ = gpu.free_tensor(self.token_embd);
        let _ = gpu.free_tensor(self.output_norm);
        self.output.free_all(gpu);
    }
}

impl Gemma4DenseWeights {
    pub fn free_gpu(self, gpu: &mut Gpu) {
        self.core.free_gpu(gpu);
        for layer in self.layers {
            for norm in [
                layer.input_norm,
                layer.q_norm,
                layer.k_norm,
                layer.post_attn_norm,
                layer.pre_ffn_norm,
                layer.post_ffn_norm,
            ] {
                let _ = gpu.free_tensor(norm);
            }
            layer.wq.free_all(gpu);
            layer.wk.free_all(gpu);
            if let Some(wv) = layer.wv {
                wv.free_all(gpu);
            }
            layer.wo.free_all(gpu);
            layer.w_gate.free_all(gpu);
            layer.w_up.free_all(gpu);
            layer.w_down.free_all(gpu);
            if let Some(ple) = layer.ple {
                ple.input_gate.free_all(gpu);
                ple.projection.free_all(gpu);
                let _ = gpu.free_tensor(ple.post_norm);
            }
        }
        if let Some(ple) = self.ple {
            ple.embed_per_layer.free_all(gpu);
            ple.model_projection.free_all(gpu);
            let _ = gpu.free_tensor(ple.projection_norm);
        }
    }
}

/// Load the vocabulary table, final direct RMSNorm, and tied/untied output
/// projection. Layer-specific required/optional policy remains in this crate as
/// the later layer plans land; no tensor mechanics are copied here.
pub fn load_core_weights(
    hfq: &mut HfqFile,
    gpu: &mut Gpu,
    shape: Gemma4CoreShape,
) -> HipResult<Gemma4CoreWeights> {
    let embedding_source_bf16 = hfq
        .find_tensor_info(EMBEDDING)
        .map(|info| info.quant_type == 16)
        .unwrap_or(false);
    #[cfg(unix)]
    hfq.drop_mmap();
    let loader = TransformerLoader::new(hfq, "gemma4");
    let (token_embd, embd_format) =
        loader.load_embedding(gpu, EMBEDDING, shape.vocab_size, shape.hidden_size)?;
    let output_norm = loader.load_direct_f32(gpu, FINAL_NORM, &[shape.hidden_size])?;
    let (output, tied_lm_head) = loader.load_lm_head(
        gpu,
        EMBEDDING,
        LM_HEAD,
        shape.tie_word_embeddings,
        shape.vocab_size,
        shape.hidden_size,
    )?;
    Ok(Gemma4CoreWeights {
        token_embd,
        embd_format,
        embedding_source_bf16,
        output_norm,
        output,
        tied_lm_head,
    })
}

/// Load the resident dense decoder used by the Phase 4 reference/lowered
/// forward and Phase 5 31B bring-up. PLE, sharing, and routed experts remain
/// explicit later-phase paths rather than optional fields on this structure.
pub fn load_dense_weights(
    hfq: &mut HfqFile,
    gpu: &mut Gpu,
    config: &Gemma4Config,
) -> HipResult<Gemma4DenseWeights> {
    // PLE (per-layer embeddings) IS supported (E2B/E4B). The KV-sharing forward is
    // implemented too, but E2B currently has a coherence bug (semantic next-token
    // prediction degrades — see docs/plans/2026-07-18-gemma4-effective-ple-kvshare.md),
    // so KV-sharing stays gated until that is isolated + fixed. Routed experts (MoE)
    // are not yet supported. Flip this to enable E2B/E4B once coherence is fixed.
    if config.layers.iter().any(|layer| {
        !matches!(layer.kv_producer, crate::config::KvProducer::Own)
            || !matches!(layer.ffn, FfnPlan::Dense { .. })
    }) {
        return Err(hip_bridge::HipError::new(
            0,
            "Gemma 4 dense loader: KV sharing pending a coherence fix; MoE unsupported",
        ));
    }

    let core = load_core_weights(
        hfq,
        gpu,
        Gemma4CoreShape {
            vocab_size: config.vocab_size,
            hidden_size: config.hidden_size,
            tie_word_embeddings: config.tie_word_embeddings,
        },
    )?;
    let loader = TransformerLoader::new(hfq, "gemma4");
    let mut layers = Vec::with_capacity(config.layers.len());
    for (layer_idx, plan) in config.layers.iter().enumerate() {
        let prefix = format!("model.language_model.layers.{layer_idx}");
        let attn = format!("{prefix}.self_attn");
        let q_dim = plan.attention.q_heads * plan.attention.head_dim;
        let kv_dim = plan.attention.kv_heads * plan.attention.head_dim;
        let intermediate = match plan.ffn {
            FfnPlan::Dense { intermediate } => intermediate,
            FfnPlan::DensePlusMoe { .. } => unreachable!("rejected above"),
        };
        let scalar_tensor = loader.load_direct_f32(gpu, &format!("{prefix}.layer_scalar"), &[1])?;
        let layer_scalar = gpu.download_f32(&scalar_tensor)?[0];
        let _ = gpu.free_tensor(scalar_tensor);
        layers.push(Gemma4DenseLayerWeights {
            input_norm: loader.load_direct_f32(
                gpu,
                &format!("{prefix}.input_layernorm.weight"),
                &[config.hidden_size],
            )?,
            q_norm: loader.load_direct_f32(
                gpu,
                &format!("{attn}.q_norm.weight"),
                &[plan.attention.head_dim],
            )?,
            k_norm: loader.load_direct_f32(
                gpu,
                &format!("{attn}.k_norm.weight"),
                &[plan.attention.head_dim],
            )?,
            wq: loader.load_weight(
                gpu,
                &format!("{attn}.q_proj.weight"),
                q_dim,
                config.hidden_size,
            )?,
            wk: loader.load_weight(
                gpu,
                &format!("{attn}.k_proj.weight"),
                kv_dim,
                config.hidden_size,
            )?,
            wv: match plan.value_projection {
                ValueProjection::Separate => Some(loader.load_weight(
                    gpu,
                    &format!("{attn}.v_proj.weight"),
                    kv_dim,
                    config.hidden_size,
                )?),
                ValueProjection::FromPreNormKey => None,
            },
            wo: loader.load_weight(
                gpu,
                &format!("{attn}.o_proj.weight"),
                config.hidden_size,
                q_dim,
            )?,
            post_attn_norm: loader.load_direct_f32(
                gpu,
                &format!("{prefix}.post_attention_layernorm.weight"),
                &[config.hidden_size],
            )?,
            pre_ffn_norm: loader.load_direct_f32(
                gpu,
                &format!("{prefix}.pre_feedforward_layernorm.weight"),
                &[config.hidden_size],
            )?,
            post_ffn_norm: loader.load_direct_f32(
                gpu,
                &format!("{prefix}.post_feedforward_layernorm.weight"),
                &[config.hidden_size],
            )?,
            w_gate: loader.load_weight(
                gpu,
                &format!("{prefix}.mlp.gate_proj.weight"),
                intermediate,
                config.hidden_size,
            )?,
            w_up: loader.load_weight(
                gpu,
                &format!("{prefix}.mlp.up_proj.weight"),
                intermediate,
                config.hidden_size,
            )?,
            w_down: loader.load_weight(
                gpu,
                &format!("{prefix}.mlp.down_proj.weight"),
                config.hidden_size,
                intermediate,
            )?,
            layer_scalar,
            ple: if config.hidden_size_per_layer_input != 0 {
                let ple_dim = config.hidden_size_per_layer_input;
                Some(Gemma4PleLayerWeights {
                    input_gate: loader.load_weight(
                        gpu,
                        &format!("{prefix}.per_layer_input_gate.weight"),
                        ple_dim,
                        config.hidden_size,
                    )?,
                    projection: loader.load_weight(
                        gpu,
                        &format!("{prefix}.per_layer_projection.weight"),
                        config.hidden_size,
                        ple_dim,
                    )?,
                    post_norm: loader.load_direct_f32(
                        gpu,
                        &format!("{prefix}.post_per_layer_input_norm.weight"),
                        &[config.hidden_size],
                    )?,
                })
            } else {
                None
            },
        });
    }
    let ple = if config.hidden_size_per_layer_input != 0 {
        let ple_dim = config.hidden_size_per_layer_input;
        let num_layers = config.layers.len();
        let p = Gemma4PleWeights {
            embed_per_layer: loader.load_weight(
                gpu,
                "model.language_model.embed_tokens_per_layer.weight",
                config.vocab_size,
                num_layers * ple_dim,
            )?,
            model_projection: loader.load_weight(
                gpu,
                "model.language_model.per_layer_model_projection.weight",
                num_layers * ple_dim,
                config.hidden_size,
            )?,
            projection_norm: loader.load_direct_f32(
                gpu,
                "model.language_model.per_layer_projection_norm.weight",
                &[ple_dim],
            )?,
            ple_dim,
            num_layers,
        };
        Some(p)
    } else {
        None
    };
    Ok(Gemma4DenseWeights { core, layers, ple })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn core_names_match_released_standard_and_unified_text_layout() {
        assert_eq!(EMBEDDING, "model.language_model.embed_tokens.weight");
        assert_eq!(FINAL_NORM, "model.language_model.norm.weight");
    }
}
