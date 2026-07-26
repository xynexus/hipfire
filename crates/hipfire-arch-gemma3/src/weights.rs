// SPDX-License-Identifier: Apache-2.0
// hipfire — Gemma3 weight loading. See LICENSE / NOTICE.

//! GPU-resident Gemma3 weights + family-specific tensor assembly over the
//! shared runtime transformer loader. Gemma3 contributes its layout: **4 norms
//! per layer** (`input_layernorm`,
//! `post_attention_layernorm`, `pre_feedforward_layernorm`,
//! `post_feedforward_layernorm`), **per-head `q_norm`/`k_norm`** (over
//! `head_dim`), **GeGLU** (`gate`/`up`/`down`), **tied embeddings**, and **no
//! QKV bias** (`attention_bias=false`). Norm weights ship `(1+w)`-baked from the
//! quantizer, so they load as plain F32 and need no runtime offset.
//!
//! Prefix rules, required tensors, and the per-layer structure stay here;
//! lookup, exact-shape validation, upload/quant mapping, embeddings, tied
//! heads, and direct norms live in `hipfire_runtime::transformer_loader`.

use hip_bridge::HipResult;
use hipfire_rdna::{DType, Gpu, GpuTensor};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::transformer_loader::TransformerLoader;
use hipfire_runtime::weights::{EmbeddingFormat, WeightTensor};

use crate::config::Gemma3Config;

/// Per-layer Gemma3 weights. No biases (attention_bias=false). The four norms
/// and the two qk-norms are F32 on GPU (qk-norm shape `[head_dim]`, the rest
/// `[hidden_size]`).
pub struct Gemma3LayerWeights {
    pub input_norm: GpuTensor,     // input_layernorm.weight  [hidden]
    pub q_norm: GpuTensor,         // self_attn.q_norm.weight  [head_dim]
    pub k_norm: GpuTensor,         // self_attn.k_norm.weight  [head_dim]
    pub wq: WeightTensor,          // self_attn.q_proj.weight  [n_heads*head_dim, hidden]
    pub wk: WeightTensor,          // self_attn.k_proj.weight  [n_kv*head_dim, hidden]
    pub wv: WeightTensor,          // self_attn.v_proj.weight
    pub wo: WeightTensor,          // self_attn.o_proj.weight
    pub post_attn_norm: GpuTensor, // post_attention_layernorm.weight  [hidden]
    pub pre_ffn_norm: GpuTensor,   // pre_feedforward_layernorm.weight  [hidden]
    pub post_ffn_norm: GpuTensor,  // post_feedforward_layernorm.weight [hidden]
    pub w_gate: WeightTensor,      // mlp.gate_proj.weight  [intermediate, hidden]
    pub w_up: WeightTensor,        // mlp.up_proj.weight
    pub w_down: WeightTensor,      // mlp.down_proj.weight  [hidden, intermediate]
}

/// GPU-resident Gemma3 model weights.
pub struct Gemma3Weights {
    pub token_embd: GpuTensor,
    pub embd_format: EmbeddingFormat,
    pub output_norm: GpuTensor, // model.norm.weight, F32
    pub output: WeightTensor,   // lm_head (tied → re-uploaded embedding bytes)
    pub layers: Vec<Gemma3LayerWeights>,
    /// Gemma3 ties lm_head to the embedding table; `output` is a separate
    /// allocation of the same bytes (GpuTensor is not Clone).
    pub tied_lm_head: bool,
}

impl Gemma3Weights {
    /// Load every tensor from `hfq` to GPU.
    pub fn load(hfq: &mut HfqFile, cfg: &Gemma3Config, gpu: &mut Gpu) -> Result<Self, String> {
        load_weights(hfq, cfg, gpu).map_err(|e| format!("gemma3: load_weights failed: {e:?}"))
    }

    /// Release every GPU buffer back to the pool. Consumes self.
    pub fn free_gpu(self, gpu: &mut Gpu) {
        let _ = gpu.free_tensor(self.token_embd);
        let _ = gpu.free_tensor(self.output_norm);
        free_weight_tensor(gpu, self.output);
        for l in self.layers {
            let _ = gpu.free_tensor(l.input_norm);
            let _ = gpu.free_tensor(l.q_norm);
            let _ = gpu.free_tensor(l.k_norm);
            free_weight_tensor(gpu, l.wq);
            free_weight_tensor(gpu, l.wk);
            free_weight_tensor(gpu, l.wv);
            free_weight_tensor(gpu, l.wo);
            let _ = gpu.free_tensor(l.post_attn_norm);
            let _ = gpu.free_tensor(l.pre_ffn_norm);
            let _ = gpu.free_tensor(l.post_ffn_norm);
            free_weight_tensor(gpu, l.w_gate);
            free_weight_tensor(gpu, l.w_up);
            free_weight_tensor(gpu, l.w_down);
        }
    }
}

/// Free-function loader; takes a borrowed `Gpu` so the `Architecture` impl can
/// pass the runtime-provided handle.
pub fn load_weights(
    hfq: &mut HfqFile,
    cfg: &Gemma3Config,
    gpu: &mut Gpu,
) -> HipResult<Gemma3Weights> {
    load_weights_prefixed(hfq, cfg, gpu, "")
}

/// Load the gemma3 text decoder with a tensor-name `prefix`. Pure-text gemma3
/// uses `""` (`model.*` / `lm_head.weight`); the gemma3 multimodal wrapper nests
/// the decoder under `"language_model."` (`language_model.model.*`).
pub fn load_weights_prefixed(
    hfq: &mut HfqFile,
    cfg: &Gemma3Config,
    gpu: &mut Gpu,
    prefix: &str,
) -> HipResult<Gemma3Weights> {
    #[cfg(unix)]
    hfq.drop_mmap();
    let loader = TransformerLoader::new(hfq, "gemma3");

    eprintln!("gemma3: loading token_embd...");
    let (token_embd, embd_format) = loader.load_embedding(
        gpu,
        &format!("{prefix}model.embed_tokens.weight"),
        cfg.vocab_size,
        cfg.hidden_size,
    )?;

    eprintln!("gemma3: loading model.norm...");
    let output_norm = loader.load_direct_f32(
        gpu,
        &format!("{prefix}model.norm.weight"),
        &[cfg.hidden_size],
    )?;

    eprintln!(
        "gemma3: loading lm_head (tied={})...",
        cfg.tie_word_embeddings
    );
    let (output, tied_lm_head) = loader.load_lm_head(
        gpu,
        &format!("{prefix}model.embed_tokens.weight"),
        &format!("{prefix}lm_head.weight"),
        cfg.tie_word_embeddings,
        cfg.vocab_size,
        cfg.hidden_size,
    )?;

    let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
    for i in 0..cfg.num_hidden_layers {
        hipfire_runtime::load_progress::report(
            i as u32 + 1,
            cfg.num_hidden_layers as u32,
            "weights",
        );
        layers.push(load_layer(&loader, gpu, cfg, i, prefix)?);
    }

    Ok(Gemma3Weights {
        token_embd,
        embd_format,
        output_norm,
        output,
        layers,
        tied_lm_head,
    })
}

/// Load only the Gemma3 transformer encoder weights. The token embedding table
/// and tied LM head are replaced with tiny unused placeholders so encoder-only
/// callers can provide embeddings through another seam without allocating the
/// full vocabulary table twice.
pub fn load_encoder_weights_prefixed(
    hfq: &mut HfqFile,
    cfg: &Gemma3Config,
    gpu: &mut Gpu,
    prefix: &str,
) -> HipResult<Gemma3Weights> {
    #[cfg(unix)]
    hfq.drop_mmap();
    let loader = TransformerLoader::new(hfq, "gemma3");

    eprintln!("gemma3: loading encoder without token_embd/lm_head...");
    let token_embd = gpu.zeros(&[1], DType::F32)?;
    let output_norm = loader.load_direct_f32(
        gpu,
        &format!("{prefix}model.norm.weight"),
        &[cfg.hidden_size],
    )?;
    let output = dummy_weight_tensor(gpu.zeros(&[1], DType::F32)?, 1, 1);

    let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
    for i in 0..cfg.num_hidden_layers {
        hipfire_runtime::load_progress::report(
            i as u32 + 1,
            cfg.num_hidden_layers as u32,
            "weights",
        );
        layers.push(load_layer(&loader, gpu, cfg, i, prefix)?);
    }

    Ok(Gemma3Weights {
        token_embd,
        embd_format: EmbeddingFormat::F32,
        output_norm,
        output,
        layers,
        tied_lm_head: false,
    })
}

/// Load only normalization tensors for a backend that owns every projection.
/// Projection fields retain their logical shapes but contain one-element dummy
/// buffers and must never reach the GPU fallback path.
pub fn load_resident_encoder_scaffold(
    hfq: &HfqFile,
    cfg: &Gemma3Config,
    gpu: &mut Gpu,
) -> HipResult<Gemma3Weights> {
    let loader = TransformerLoader::new(hfq, "gemma3");
    let dummy_weight = |gpu: &mut Gpu, m: usize, k: usize| -> HipResult<WeightTensor> {
        Ok(dummy_weight_tensor(gpu.zeros(&[1], DType::F32)?, m, k))
    };
    let output_norm = loader.load_direct_f32(gpu, "model.norm.weight", &[cfg.hidden_size])?;
    let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
    for i in 0..cfg.num_hidden_layers {
        let p = format!("model.layers.{i}");
        let input_norm = loader.load_direct_f32(
            gpu,
            &format!("{p}.input_layernorm.weight"),
            &[cfg.hidden_size],
        )?;
        let q_norm = loader.load_direct_f32(
            gpu,
            &format!("{p}.self_attn.q_norm.weight"),
            &[cfg.head_dim],
        )?;
        let prescale = cfg.q_prescale();
        if (prescale - 1.0).abs() > 1e-6 {
            gpu.scale_f32(&q_norm, prescale)?;
        }
        let k_norm = loader.load_direct_f32(
            gpu,
            &format!("{p}.self_attn.k_norm.weight"),
            &[cfg.head_dim],
        )?;
        let post_attn_norm = loader.load_direct_f32(
            gpu,
            &format!("{p}.post_attention_layernorm.weight"),
            &[cfg.hidden_size],
        )?;
        let pre_ffn_norm = loader.load_direct_f32(
            gpu,
            &format!("{p}.pre_feedforward_layernorm.weight"),
            &[cfg.hidden_size],
        )?;
        let post_ffn_norm = loader.load_direct_f32(
            gpu,
            &format!("{p}.post_feedforward_layernorm.weight"),
            &[cfg.hidden_size],
        )?;
        let q_dim = cfg.num_attention_heads * cfg.head_dim;
        let kv_dim = cfg.num_key_value_heads * cfg.head_dim;
        layers.push(Gemma3LayerWeights {
            input_norm,
            q_norm,
            k_norm,
            wq: dummy_weight(gpu, q_dim, cfg.hidden_size)?,
            wk: dummy_weight(gpu, kv_dim, cfg.hidden_size)?,
            wv: dummy_weight(gpu, kv_dim, cfg.hidden_size)?,
            wo: dummy_weight(gpu, cfg.hidden_size, q_dim)?,
            post_attn_norm,
            pre_ffn_norm,
            post_ffn_norm,
            w_gate: dummy_weight(gpu, cfg.intermediate_size, cfg.hidden_size)?,
            w_up: dummy_weight(gpu, cfg.intermediate_size, cfg.hidden_size)?,
            w_down: dummy_weight(gpu, cfg.hidden_size, cfg.intermediate_size)?,
        });
    }
    Ok(Gemma3Weights {
        token_embd: gpu.zeros(&[1], DType::F32)?,
        embd_format: EmbeddingFormat::F32,
        output_norm,
        output: dummy_weight(gpu, 1, 1)?,
        layers,
        tied_lm_head: false,
    })
}

fn load_layer(
    loader: &TransformerLoader<'_>,
    gpu: &mut Gpu,
    cfg: &Gemma3Config,
    i: usize,
    prefix: &str,
) -> HipResult<Gemma3LayerWeights> {
    let p = format!("{prefix}model.layers.{i}");
    let q_dim = cfg.num_attention_heads * cfg.head_dim;
    let kv_dim = cfg.num_key_value_heads * cfg.head_dim;

    let input_norm = loader.load_direct_f32(
        gpu,
        &format!("{p}.input_layernorm.weight"),
        &[cfg.hidden_size],
    )?;
    // Per-head QK-norm: RMSNorm over head_dim. Bake the Q pre-scale into
    // q_norm so the attention kernel's built-in 1/√head_dim becomes Gemma's
    // 1/√query_pre_attn_scalar (no per-step scale launch). No-op when
    // q_prescale == 1.0 (query_pre_attn_scalar == head_dim, e.g. gemma3-4b).
    let q_norm = loader.load_direct_f32(
        gpu,
        &format!("{p}.self_attn.q_norm.weight"),
        &[cfg.head_dim],
    )?;
    let prescale = cfg.q_prescale();
    if (prescale - 1.0).abs() > 1e-6 {
        gpu.scale_f32(&q_norm, prescale)?;
    }
    let k_norm = loader.load_direct_f32(
        gpu,
        &format!("{p}.self_attn.k_norm.weight"),
        &[cfg.head_dim],
    )?;

    let wq = loader.load_weight(
        gpu,
        &format!("{p}.self_attn.q_proj.weight"),
        q_dim,
        cfg.hidden_size,
    )?;
    let wk = loader.load_weight(
        gpu,
        &format!("{p}.self_attn.k_proj.weight"),
        kv_dim,
        cfg.hidden_size,
    )?;
    let wv = loader.load_weight(
        gpu,
        &format!("{p}.self_attn.v_proj.weight"),
        kv_dim,
        cfg.hidden_size,
    )?;
    let wo = loader.load_weight(
        gpu,
        &format!("{p}.self_attn.o_proj.weight"),
        cfg.hidden_size,
        q_dim,
    )?;

    let post_attn_norm = loader.load_direct_f32(
        gpu,
        &format!("{p}.post_attention_layernorm.weight"),
        &[cfg.hidden_size],
    )?;
    let pre_ffn_norm = loader.load_direct_f32(
        gpu,
        &format!("{p}.pre_feedforward_layernorm.weight"),
        &[cfg.hidden_size],
    )?;
    let post_ffn_norm = loader.load_direct_f32(
        gpu,
        &format!("{p}.post_feedforward_layernorm.weight"),
        &[cfg.hidden_size],
    )?;

    let w_gate = loader.load_weight(
        gpu,
        &format!("{p}.mlp.gate_proj.weight"),
        cfg.intermediate_size,
        cfg.hidden_size,
    )?;
    let w_up = loader.load_weight(
        gpu,
        &format!("{p}.mlp.up_proj.weight"),
        cfg.intermediate_size,
        cfg.hidden_size,
    )?;
    let w_down = loader.load_weight(
        gpu,
        &format!("{p}.mlp.down_proj.weight"),
        cfg.hidden_size,
        cfg.intermediate_size,
    )?;

    Ok(Gemma3LayerWeights {
        input_norm,
        q_norm,
        k_norm,
        wq,
        wk,
        wv,
        wo,
        post_attn_norm,
        pre_ffn_norm,
        post_ffn_norm,
        w_gate,
        w_up,
        w_down,
    })
}

fn free_weight_tensor(gpu: &mut Gpu, wt: WeightTensor) {
    let _ = gpu.free_tensor(wt.buf);
    if let Some(awq) = wt.awq_scale {
        let _ = gpu.free_tensor(awq);
    }
}

fn dummy_weight_tensor(buf: GpuTensor, m: usize, k: usize) -> WeightTensor {
    WeightTensor {
        buf,
        gpu_dtype: DType::F32,
        m,
        k,
        row_stride: 0,
        paro: None,
        awq_scale: None,
    }
}
