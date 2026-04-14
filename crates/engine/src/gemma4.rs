//! Gemma 4 model: hybrid sliding-window + full attention, dense FFN (SwiGLU + gelu_pytorch_tanh).
//!
//! Architectural features vs. Qwen3.5:
//!   • Sliding-window attention on 5 of every 6 layers (window=1024).
//!   • Full attention layers use head_dim=512 (global_head_dim) with
//!     attention_k_eq_v: V is the pre-k_norm output of k_proj (no v_proj).
//!   • Partial proportional RoPE on full layers (first 64 of 512 dims rotate,
//!     rope_theta=1e6; sliding uses default RoPE with theta=10000).
//!   • Sandwich RMSNorm: input + post-attn + pre-FFN + post-FFN per layer,
//!     plus a learned per-layer `layer_scalar [1]` at layer end.
//!   • Attention scale = 1.0 (not 1/√d); Q/K norms absorb scaling.
//!   • Final logit softcap: `tanh(logits/30) * 30` before sampling.
//!   • MLP: SwiGLU with `gelu_pytorch_tanh` activation.
//!   • Tied LM head (embed_tokens.weight aliased).
//!   • Embed scale: sqrt(hidden_size) multiplied onto every embedding row lookup.
//!
//! Phase 1 (this file): scaffolding only — config parser + stub weight/scratch
//! structs. Real forward pass lands in Phase 3.

use crate::hfq::HfqFile;
use hip_bridge::HipResult;
use rdna_compute::{Gpu, GpuTensor};

// ─── Config ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LayerType {
    /// Sliding-window causal attention (window=1024 on 31B).
    Sliding,
    /// Full causal attention (global).
    Full,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RopeType {
    /// Standard RoPE: all head_dim positions rotate.
    Default,
    /// Proportional RoPE (Gemma 4 full layers): only the first
    /// `partial_rotary_factor × head_dim` positions rotate; rest are NoPE.
    Proportional,
}

#[derive(Debug)]
pub struct Gemma4Config {
    // Common
    pub dim: usize,                        // hidden_size, e.g. 5376 on 31B
    pub n_layers: usize,                   // 60 on 31B
    pub vocab_size: usize,                 // 262144 on Gemma 4
    pub norm_eps: f32,                     // 1e-6
    pub bos_token: u32,                    // 2
    pub eos_token: u32,                    // 1
    pub pad_token: u32,                    // 0

    // Attention heads (same count for sliding + full)
    pub n_heads: usize,                    // 32 on 31B

    // Sliding-window attention
    pub sliding_head_dim: usize,           // 256 on 31B
    pub sliding_n_kv_heads: usize,         // 16 on 31B
    pub sliding_rope_theta: f32,           // 10000.0
    pub sliding_window: usize,             // 1024

    // Full attention (global)
    pub full_head_dim: usize,              // 512 on 31B (= global_head_dim)
    pub full_n_kv_heads: usize,            // 4 on 31B
    pub full_rope_theta: f32,              // 1_000_000.0
    pub full_rope_type: RopeType,          // Proportional on 31B
    pub full_partial_rotary_factor: f32,   // 0.25
    pub attention_k_eq_v: bool,            // true on 31B — V = pre-k_norm output

    // FFN (SwiGLU, gelu_pytorch_tanh)
    pub hidden_dim: usize,                 // intermediate_size = 21504 on 31B

    // Output
    pub final_logit_softcapping: f32,      // 30.0 — tanh(x/30)*30
    pub tie_word_embeddings: bool,         // true — lm_head aliases embed_tokens
    pub embed_scale: f32,                  // sqrt(dim), applied at embed lookup

    // Per-layer dispatch (len == n_layers)
    pub layer_types: Vec<LayerType>,

    // Vision integration (present even on text-only 31B since config ships it)
    pub has_vision: bool,
    pub image_token_id: u32,               // 258880
    pub boi_token_id: u32,                 // 255999
    pub eoi_token_id: u32,                 // 258882
    pub audio_token_id: u32,               // 258881 (reserved, unused on dense 31B)
    pub video_token_id: u32,               // 258884 (reserved)
}

pub fn config_from_hfq(hfq: &HfqFile) -> Option<Gemma4Config> {
    let meta: serde_json::Value = serde_json::from_str(&hfq.metadata_json).ok()?;
    let config = meta.get("config")?;
    let tc = config.get("text_config").unwrap_or(config);

    let dim = tc.get("hidden_size")?.as_u64()? as usize;
    let n_layers = tc.get("num_hidden_layers")?.as_u64()? as usize;
    let vocab_size = tc.get("vocab_size")?.as_u64()? as usize;
    let norm_eps = tc.get("rms_norm_eps").and_then(|v| v.as_f64()).unwrap_or(1e-6) as f32;
    let bos_token = tc.get("bos_token_id").and_then(|v| v.as_u64()).unwrap_or(2) as u32;
    let eos_token = tc.get("eos_token_id").and_then(|v| v.as_u64()).unwrap_or(1) as u32;
    let pad_token = tc.get("pad_token_id").and_then(|v| v.as_u64()).unwrap_or(0) as u32;

    let n_heads = tc.get("num_attention_heads")?.as_u64()? as usize;

    // Sliding attention params
    let sliding_head_dim = tc.get("head_dim").and_then(|v| v.as_u64()).map(|v| v as usize)
        .unwrap_or(dim / n_heads);
    let sliding_n_kv_heads = tc.get("num_key_value_heads").and_then(|v| v.as_u64())
        .unwrap_or(n_heads as u64) as usize;
    let sliding_window = tc.get("sliding_window").and_then(|v| v.as_u64()).unwrap_or(1024) as usize;

    // Full attention params (may differ from sliding)
    let full_head_dim = tc.get("global_head_dim").and_then(|v| v.as_u64()).map(|v| v as usize)
        .unwrap_or(sliding_head_dim);
    let full_n_kv_heads = tc.get("num_global_key_value_heads").and_then(|v| v.as_u64())
        .unwrap_or(sliding_n_kv_heads as u64) as usize;
    let attention_k_eq_v = tc.get("attention_k_eq_v").and_then(|v| v.as_bool()).unwrap_or(false);

    // rope_parameters is a dict with "sliding_attention" and "full_attention" sub-dicts
    // per the Gemma 4 config schema. Parse both independently.
    let rope_params = tc.get("rope_parameters");
    let sliding_rope = rope_params.and_then(|r| r.get("sliding_attention"));
    let full_rope = rope_params.and_then(|r| r.get("full_attention"));

    let sliding_rope_theta = sliding_rope.and_then(|r| r.get("rope_theta"))
        .and_then(|v| v.as_f64()).unwrap_or(10_000.0) as f32;
    let full_rope_theta = full_rope.and_then(|r| r.get("rope_theta"))
        .and_then(|v| v.as_f64()).unwrap_or(1_000_000.0) as f32;
    let full_rope_type = match full_rope.and_then(|r| r.get("rope_type")).and_then(|v| v.as_str()) {
        Some("proportional") => RopeType::Proportional,
        _ => RopeType::Default,
    };
    let full_partial_rotary_factor = full_rope.and_then(|r| r.get("partial_rotary_factor"))
        .and_then(|v| v.as_f64()).unwrap_or(1.0) as f32;

    let hidden_dim = tc.get("intermediate_size")?.as_u64()? as usize;

    let final_logit_softcapping = tc.get("final_logit_softcapping")
        .and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
    let tie_word_embeddings = tc.get("tie_word_embeddings").and_then(|v| v.as_bool())
        .or_else(|| config.get("tie_word_embeddings").and_then(|v| v.as_bool()))
        .unwrap_or(true);

    let embed_scale = (dim as f32).sqrt();

    let layer_types: Vec<LayerType> = tc.get("layer_types")
        .and_then(|v| v.as_array())
        .map(|arr| arr.iter().map(|v| match v.as_str().unwrap_or("sliding_attention") {
            "full_attention" => LayerType::Full,
            _ => LayerType::Sliding,
        }).collect())
        .unwrap_or_else(|| vec![LayerType::Sliding; n_layers]);

    // Multimodal token IDs (top-level in config, not under text_config)
    let has_vision = config.get("vision_config").map(|v| !v.is_null()).unwrap_or(false);
    let image_token_id = config.get("image_token_id").and_then(|v| v.as_u64()).unwrap_or(258880) as u32;
    let boi_token_id = config.get("boi_token_id").and_then(|v| v.as_u64()).unwrap_or(255999) as u32;
    let eoi_token_id = config.get("eoi_token_id").and_then(|v| v.as_u64()).unwrap_or(258882) as u32;
    let audio_token_id = config.get("audio_token_id").and_then(|v| v.as_u64()).unwrap_or(258881) as u32;
    let video_token_id = config.get("video_token_id").and_then(|v| v.as_u64()).unwrap_or(258884) as u32;

    Some(Gemma4Config {
        dim, n_layers, vocab_size, norm_eps,
        bos_token, eos_token, pad_token,
        n_heads,
        sliding_head_dim, sliding_n_kv_heads, sliding_rope_theta, sliding_window,
        full_head_dim, full_n_kv_heads, full_rope_theta, full_rope_type,
        full_partial_rotary_factor, attention_k_eq_v,
        hidden_dim,
        final_logit_softcapping, tie_word_embeddings, embed_scale,
        layer_types,
        has_vision,
        image_token_id, boi_token_id, eoi_token_id, audio_token_id, video_token_id,
    })
}

// ─── Weights ────────────────────────────────────────────────────────────
//
// Phase 1 skeleton: we track what needs to land in Phase 3 (forward) but don't
// populate any GPU tensors yet. Real loader lives in Phase 5 (quantizer) and
// runtime-load in Phase 3.

/// Per-layer weights for a SLIDING layer (head_dim=256, 16 KV heads, full RoPE).
pub struct SlidingLayerWeights {
    pub input_layernorm: GpuTensor,           // [dim]
    pub post_attention_layernorm: GpuTensor,  // [dim]
    pub pre_feedforward_layernorm: GpuTensor, // [dim]
    pub post_feedforward_layernorm: GpuTensor,// [dim]
    pub layer_scalar: GpuTensor,              // [1]

    // Attention (sliding — head_dim=256)
    pub q_proj: GpuTensor,   // [n_heads * 256, dim]
    pub k_proj: GpuTensor,   // [16 * 256, dim]
    pub v_proj: GpuTensor,   // [16 * 256, dim]
    pub o_proj: GpuTensor,   // [dim, n_heads * 256]
    pub q_norm: GpuTensor,   // [256]
    pub k_norm: GpuTensor,   // [256]

    // MLP (SwiGLU)
    pub gate_proj: GpuTensor, // [hidden_dim, dim]
    pub up_proj: GpuTensor,   // [hidden_dim, dim]
    pub down_proj: GpuTensor, // [dim, hidden_dim]
}

/// Per-layer weights for a FULL layer (head_dim=512, 4 KV heads, K=V shared).
///
/// Note: no `v_proj` — V is the pre-k_norm output of k_proj, renormed by
/// weight-less `v_norm`. No `v_norm` tensor either (no_scale — the `with_scale=False`
/// RMSNorm applies only the divide, no learned gain).
pub struct FullLayerWeights {
    pub input_layernorm: GpuTensor,
    pub post_attention_layernorm: GpuTensor,
    pub pre_feedforward_layernorm: GpuTensor,
    pub post_feedforward_layernorm: GpuTensor,
    pub layer_scalar: GpuTensor,

    // Attention (full — head_dim=512, K=V)
    pub q_proj: GpuTensor,   // [n_heads * 512, dim]
    pub k_proj: GpuTensor,   // [4 * 512, dim]
    // no v_proj — V = pre-k_norm output of k_proj
    pub o_proj: GpuTensor,   // [dim, n_heads * 512]
    pub q_norm: GpuTensor,   // [512]
    pub k_norm: GpuTensor,   // [512]
    // no v_norm weight — v_norm is no-scale (divide only)

    // MLP (SwiGLU, same shape as sliding)
    pub gate_proj: GpuTensor,
    pub up_proj: GpuTensor,
    pub down_proj: GpuTensor,
}

pub enum LayerWeights {
    Sliding(SlidingLayerWeights),
    Full(FullLayerWeights),
}

pub struct Gemma4Weights {
    /// Token embedding [vocab_size, dim], Q8F16 to keep the 262144×5376 table manageable.
    /// Aliased as lm_head when tie_word_embeddings is true.
    pub embed_tokens: GpuTensor,
    /// Model-final RMSNorm scale [dim].
    pub final_norm: GpuTensor,
    /// Per-layer weights indexed by layer ordinal.
    pub layers: Vec<LayerWeights>,
}

impl Gemma4Weights {
    pub fn free_gpu(self, gpu: &mut Gpu) {
        let _ = gpu.free_tensor(self.embed_tokens);
        let _ = gpu.free_tensor(self.final_norm);
        for l in self.layers {
            match l {
                LayerWeights::Sliding(s) => {
                    for t in [s.input_layernorm, s.post_attention_layernorm,
                              s.pre_feedforward_layernorm, s.post_feedforward_layernorm,
                              s.layer_scalar, s.q_proj, s.k_proj, s.v_proj, s.o_proj,
                              s.q_norm, s.k_norm, s.gate_proj, s.up_proj, s.down_proj] {
                        let _ = gpu.free_tensor(t);
                    }
                }
                LayerWeights::Full(f) => {
                    for t in [f.input_layernorm, f.post_attention_layernorm,
                              f.pre_feedforward_layernorm, f.post_feedforward_layernorm,
                              f.layer_scalar, f.q_proj, f.k_proj, f.o_proj,
                              f.q_norm, f.k_norm, f.gate_proj, f.up_proj, f.down_proj] {
                        let _ = gpu.free_tensor(t);
                    }
                }
            }
        }
    }
}

/// Stub: real loader lands in Phase 3 + 5.
pub fn load_weights(_hfq: &HfqFile, _config: &Gemma4Config, _gpu: &mut Gpu)
    -> HipResult<Gemma4Weights>
{
    Err(hip_bridge::HipError::new(0, "gemma4::load_weights not implemented (Phase 3)"))
}

// ─── Scratch ────────────────────────────────────────────────────────────

/// Per-decode scratch buffers, sized once at model-load time.
/// Phase 3 will fill in the concrete layout. For now this is a placeholder
/// so the daemon can reference the type.
pub struct Gemma4Scratch {
    pub _placeholder: (),
}

impl Gemma4Scratch {
    pub fn new(_gpu: &mut Gpu, _config: &Gemma4Config, _max_prefill: usize) -> HipResult<Self> {
        Ok(Self { _placeholder: () })
    }
}

// ─── Forward pass stubs ─────────────────────────────────────────────────

/// Single-token decode. Phase 3 fills this in.
pub fn forward_scratch(
    _gpu: &mut Gpu,
    _weights: &Gemma4Weights,
    _config: &Gemma4Config,
    _token_id: u32,
    _pos: usize,
    _kv_sliding: &mut crate::llama::KvCache,
    _kv_full: &mut crate::llama::KvCache,
    _scratch: &Gemma4Scratch,
) -> HipResult<GpuTensor> {
    Err(hip_bridge::HipError::new(0, "gemma4::forward_scratch not implemented (Phase 3)"))
}

/// Batched prefill. Phase 4 fills this in.
pub fn forward_prefill_batch(
    _gpu: &mut Gpu,
    _weights: &Gemma4Weights,
    _config: &Gemma4Config,
    _tokens: &[u32],
    _start_pos: usize,
    _kv_sliding: &mut crate::llama::KvCache,
    _kv_full: &mut crate::llama::KvCache,
    _scratch: &Gemma4Scratch,
) -> HipResult<GpuTensor> {
    Err(hip_bridge::HipError::new(0, "gemma4::forward_prefill_batch not implemented (Phase 4)"))
}
