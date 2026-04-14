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

use crate::hfq::HfqFile;
use crate::llama::{self, weight_gemv, WeightTensor, EmbeddingFormat};
use hip_bridge::HipResult;
use rdna_compute::{DType, Gpu, GpuTensor};

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

/// Per-layer weights for a SLIDING layer (head_dim=256, 16 KV heads, full RoPE).
pub struct SlidingLayerWeights {
    pub input_layernorm: GpuTensor,           // [dim]
    pub post_attention_layernorm: GpuTensor,  // [dim]
    pub pre_feedforward_layernorm: GpuTensor, // [dim]
    pub post_feedforward_layernorm: GpuTensor,// [dim]
    pub layer_scalar: GpuTensor,              // [1]

    // Attention (sliding — head_dim=256)
    pub q_proj: WeightTensor,   // [n_heads * 256, dim]
    pub k_proj: WeightTensor,   // [16 * 256, dim]
    pub v_proj: WeightTensor,   // [16 * 256, dim]
    pub o_proj: WeightTensor,   // [dim, n_heads * 256]
    pub q_norm: GpuTensor,      // [256]
    pub k_norm: GpuTensor,      // [256]

    // MLP (SwiGLU)
    pub gate_proj: WeightTensor, // [hidden_dim, dim]
    pub up_proj: WeightTensor,   // [hidden_dim, dim]
    pub down_proj: WeightTensor, // [dim, hidden_dim]
}

/// Per-layer weights for a FULL layer (head_dim=512, 4 KV heads, K=V shared).
///
/// Note: no `v_proj` — V is the pre-k_norm output of k_proj, renormed by
/// weight-less `v_norm`. No `v_norm` tensor either (no_scale — the `with_scale=False`
/// RMSNorm applies only the divide, no learned gain). We reuse the existing
/// rmsnorm kernel with a ones-filled `v_norm_ones` buffer (shared across
/// full-attn layers) to preserve the no-scale semantics.
pub struct FullLayerWeights {
    pub input_layernorm: GpuTensor,
    pub post_attention_layernorm: GpuTensor,
    pub pre_feedforward_layernorm: GpuTensor,
    pub post_feedforward_layernorm: GpuTensor,
    pub layer_scalar: GpuTensor,

    // Attention (full — head_dim=512, K=V)
    pub q_proj: WeightTensor,   // [n_heads * 512, dim]
    pub k_proj: WeightTensor,   // [4 * 512, dim]
    // no v_proj — V = pre-k_norm output of k_proj
    pub o_proj: WeightTensor,   // [dim, n_heads * 512]
    pub q_norm: GpuTensor,      // [512]
    pub k_norm: GpuTensor,      // [512]
    // no v_norm weight — v_norm is no-scale (divide only)

    // MLP (SwiGLU, same shape as sliding)
    pub gate_proj: WeightTensor,
    pub up_proj: WeightTensor,
    pub down_proj: WeightTensor,
}

pub enum LayerWeights {
    Sliding(SlidingLayerWeights),
    Full(FullLayerWeights),
}

pub struct Gemma4Weights {
    /// Token embedding [vocab_size, dim], Q8F16 to keep the 262144×5376 table manageable.
    /// Aliased as lm_head when tie_word_embeddings is true.
    pub embed_tokens: GpuTensor,
    /// Embed/LM-head format tag for dispatch.
    pub embd_format: EmbeddingFormat,
    /// LM-head projection (shares bytes with embed_tokens when tied).
    pub lm_head: WeightTensor,
    /// Model-final RMSNorm scale [dim].
    pub final_norm: GpuTensor,
    /// Per-layer weights indexed by layer ordinal.
    pub layers: Vec<LayerWeights>,
}

impl Gemma4Weights {
    pub fn free_gpu(self, gpu: &mut Gpu) {
        let _ = gpu.free_tensor(self.embed_tokens);
        let _ = gpu.free_tensor(self.final_norm);
        // lm_head may alias embed_tokens — skip if so (we rely on the loader
        // to set `lm_head.buf` to an alias and not a separate allocation).
        for l in self.layers {
            match l {
                LayerWeights::Sliding(s) => {
                    for t in [s.input_layernorm, s.post_attention_layernorm,
                              s.pre_feedforward_layernorm, s.post_feedforward_layernorm,
                              s.layer_scalar, s.q_norm, s.k_norm] {
                        let _ = gpu.free_tensor(t);
                    }
                    for wt in [s.q_proj.buf, s.k_proj.buf, s.v_proj.buf, s.o_proj.buf,
                               s.gate_proj.buf, s.up_proj.buf, s.down_proj.buf] {
                        let _ = gpu.free_tensor(wt);
                    }
                }
                LayerWeights::Full(f) => {
                    for t in [f.input_layernorm, f.post_attention_layernorm,
                              f.pre_feedforward_layernorm, f.post_feedforward_layernorm,
                              f.layer_scalar, f.q_norm, f.k_norm] {
                        let _ = gpu.free_tensor(t);
                    }
                    for wt in [f.q_proj.buf, f.k_proj.buf, f.o_proj.buf,
                               f.gate_proj.buf, f.up_proj.buf, f.down_proj.buf] {
                        let _ = gpu.free_tensor(wt);
                    }
                }
            }
        }
    }
}

/// Stub: real loader lands in Phase 5 (quantizer produces HFQ → this consumes).
pub fn load_weights(_hfq: &HfqFile, _config: &Gemma4Config, _gpu: &mut Gpu)
    -> HipResult<Gemma4Weights>
{
    Err(hip_bridge::HipError::new(0, "gemma4::load_weights not implemented (Phase 5)"))
}

// ─── Scratch ────────────────────────────────────────────────────────────

use hip_bridge::DeviceBuffer;

/// Per-decode scratch, sized once at model-load time against the MAX of
/// sliding and full attention dimensions so a single buffer works across
/// layer types. 31B target shapes: sliding Q=[32*256]=8192, full Q=[32*512]=16384
/// → size Q at 16384. Sliding KV=[16*256]=4096, full KV=[4*512]=2048 → size at 4096.
pub struct Gemma4Scratch {
    pub x: GpuTensor,           // [dim] — hidden state
    pub residual: GpuTensor,    // [dim] — saved for sandwich residual
    pub tmp: GpuTensor,         // [dim] — norm output scratch

    /// Position buffer (single i32 on device, updated per decode step).
    pub pos_buf: DeviceBuffer,

    // Attention scratch — sized for max(sliding, full)
    pub q: GpuTensor,           // [max(n_heads*head_dim_sliding, n_heads*head_dim_full)]
    pub k: GpuTensor,           // [max(n_kv_heads*head_dim for each layer type)]
    pub v: GpuTensor,           // [same as k]
    pub attn_out: GpuTensor,    // [same as q]

    // MLP scratch
    pub gate_ffn: GpuTensor,    // [hidden_dim]
    pub up_ffn: GpuTensor,      // [hidden_dim]
    pub ffn_hidden: GpuTensor,  // [hidden_dim]
    pub ffn_out: GpuTensor,     // [dim]

    // Output
    pub logits: GpuTensor,      // [vocab_size]
    pub sample_buf: GpuTensor,  // [2] — (token_id, new_rng_state) for GPU sampling
    pub repeat_buf: GpuTensor,  // [1024] — rolling window for repeat penalty

    // Flash attention tile partials. Sized for the LARGER of the two
    // cache shapes: full-attn uses head_dim=512, max_tiles=max_seq/128.
    // Sliding uses head_dim=256, max_tiles=sliding_window/128 (much smaller).
    pub flash_partials: GpuTensor,

    // Pre-computed RoPE cos/sin tables per layer type.
    // Sliding: default RoPE, head_dim=256, theta=10000, n_rot = head_dim.
    pub sliding_cos: GpuTensor, // [max_seq, head_dim]
    pub sliding_sin: GpuTensor,
    // Full: proportional RoPE, head_dim=512, theta=1e6, rotated_dims = 64 of 256-half
    pub full_cos: GpuTensor,
    pub full_sin: GpuTensor,

    // No-scale v_norm ones buffer (full-attn layers compute v_norm without
    // a learned weight — we pass this ones-filled tensor to the existing
    // rmsnorm kernel to get no-scale RMS semantics).
    pub v_norm_ones_full: GpuTensor, // [full_head_dim]
}

impl Gemma4Scratch {
    pub fn new(gpu: &mut Gpu, config: &Gemma4Config, _max_prefill: usize) -> HipResult<Self> {
        let dim = config.dim;
        let q_dim = (config.n_heads * config.sliding_head_dim).max(config.n_heads * config.full_head_dim);
        let kv_dim = (config.sliding_n_kv_heads * config.sliding_head_dim)
            .max(config.full_n_kv_heads * config.full_head_dim);

        let x = gpu.zeros(&[dim], DType::F32)?;
        let residual = gpu.zeros(&[dim], DType::F32)?;
        let tmp = gpu.zeros(&[dim], DType::F32)?;

        let pos_buf = gpu.hip.malloc(4)?;

        let q = gpu.zeros(&[q_dim], DType::F32)?;
        let k = gpu.zeros(&[kv_dim], DType::F32)?;
        let v = gpu.zeros(&[kv_dim], DType::F32)?;
        let attn_out = gpu.zeros(&[q_dim], DType::F32)?;

        let gate_ffn = gpu.zeros(&[config.hidden_dim], DType::F32)?;
        let up_ffn = gpu.zeros(&[config.hidden_dim], DType::F32)?;
        let ffn_hidden = gpu.zeros(&[config.hidden_dim], DType::F32)?;
        let ffn_out = gpu.zeros(&[dim], DType::F32)?;

        let logits = gpu.zeros(&[config.vocab_size], DType::F32)?;
        let sample_buf = gpu.zeros(&[2], DType::F32)?;
        let repeat_buf = gpu.zeros(&[1024], DType::F32)?;

        // Flash partials sizing. Assumes max_seq <= 32768 (typical daemon max).
        // Per-head × max_tiles × (2 + head_dim).
        // Sized for FULL attn (larger head_dim=512, larger max_tiles).
        const MAX_CTX_DEFAULT: usize = 32768;
        const TILE_SIZE: usize = 128;
        let max_tiles_full = (MAX_CTX_DEFAULT + TILE_SIZE - 1) / TILE_SIZE;
        let flash_partials_sz = config.n_heads * max_tiles_full * (2 + config.full_head_dim);
        let flash_partials = gpu.zeros(&[flash_partials_sz], DType::F32)?;

        // RoPE tables. The actual sin/cos values are computed host-side and
        // uploaded once per model load. For now allocate and zero; the loader
        // will populate them.
        // Size: max_seq * head_dim (enough for every (position, rotary_dim) pair).
        // TODO: make max_seq configurable — using 32k default.
        let sliding_cos = gpu.zeros(&[MAX_CTX_DEFAULT * config.sliding_head_dim], DType::F32)?;
        let sliding_sin = gpu.zeros(&[MAX_CTX_DEFAULT * config.sliding_head_dim], DType::F32)?;
        let full_cos = gpu.zeros(&[MAX_CTX_DEFAULT * config.full_head_dim], DType::F32)?;
        let full_sin = gpu.zeros(&[MAX_CTX_DEFAULT * config.full_head_dim], DType::F32)?;

        // v_norm ones — populated on first use in the forward pass.
        // (Allocated to the full head_dim because only full-attn layers
        // apply no-scale v_norm.)
        let v_norm_ones_full = gpu.zeros(&[config.full_head_dim], DType::F32)?;

        Ok(Gemma4Scratch {
            x, residual, tmp, pos_buf,
            q, k, v, attn_out,
            gate_ffn, up_ffn, ffn_hidden, ffn_out,
            logits, sample_buf, repeat_buf,
            flash_partials,
            sliding_cos, sliding_sin, full_cos, full_sin,
            v_norm_ones_full,
        })
    }
}

// ─── Forward pass ───────────────────────────────────────────────────────

/// Single-token decode. Phase 3 implementation.
///
/// Precondition: `scratch.sliding_cos/sin` + `scratch.full_cos/sin` +
/// `scratch.v_norm_ones_full` must be populated by the loader before the
/// first forward call (one-time init).
pub fn forward_scratch(
    gpu: &mut Gpu,
    weights: &Gemma4Weights,
    config: &Gemma4Config,
    token: u32,
    pos: usize,
    kv_sliding: &mut llama::KvCache,
    kv_full: &mut llama::KvCache,
    scratch: &Gemma4Scratch,
) -> HipResult<()> {
    let dim = config.dim;

    // 1) Embedding lookup + sqrt(dim) scale.
    //
    // Gemma 4 multiplies the embedding row by sqrt(hidden_size) (bf16-cast
    // in the reference — we do it in fp32 here; the absolute magnitude
    // difference is sub-epsilon for our MQ4 quality target).
    match weights.embd_format {
        EmbeddingFormat::HFQ4G256 => gpu.embedding_lookup_hfq4g256(&weights.embed_tokens, &scratch.x, token, dim)?,
        EmbeddingFormat::HFQ4G128 => gpu.embedding_lookup_hfq4g128(&weights.embed_tokens, &scratch.x, token, dim)?,
        EmbeddingFormat::Q8_0    => gpu.embedding_lookup_q8(&weights.embed_tokens, &scratch.x, token, dim)?,
        EmbeddingFormat::F32     => gpu.embedding_lookup(&weights.embed_tokens, &scratch.x, token, dim)?,
        _ => return Err(hip_bridge::HipError::new(0, "unsupported Gemma 4 embed format")),
    }
    gpu.scale_f32(&scratch.x, config.embed_scale)?;

    // 2) Update device pos_buf.
    let pos_i32 = pos as i32;
    gpu.hip.memcpy_htod(&scratch.pos_buf, &pos_i32.to_ne_bytes())?;

    // 3) Per-layer forward.
    let mut sliding_kv_idx = 0usize;
    let mut full_kv_idx = 0usize;
    for (layer_idx, layer_type) in config.layer_types.iter().copied().enumerate() {
        match (layer_type, &weights.layers[layer_idx]) {
            (LayerType::Sliding, LayerWeights::Sliding(lw)) => {
                sliding_layer_decode(gpu, config, lw, pos, kv_sliding, sliding_kv_idx, scratch)?;
                sliding_kv_idx += 1;
            }
            (LayerType::Full, LayerWeights::Full(lw)) => {
                full_layer_decode(gpu, config, lw, pos, kv_full, full_kv_idx, scratch)?;
                full_kv_idx += 1;
            }
            _ => return Err(hip_bridge::HipError::new(
                0,
                &format!("Gemma 4 layer {} type/weights mismatch", layer_idx),
            )),
        }
    }

    // 4) Final RMSNorm.
    gpu.rmsnorm_f32(&scratch.x, &weights.final_norm, &scratch.tmp, config.norm_eps)?;

    // 5) LM head → logits (reads tied embed bytes via lm_head.buf alias).
    weight_gemv(gpu, &weights.lm_head, &scratch.tmp, &scratch.logits)?;

    // 6) Final logit softcap (Gemma 4): logits = tanh(logits / cap) * cap.
    if config.final_logit_softcapping > 0.0 {
        gpu.logit_softcap_f32(&scratch.logits, config.vocab_size, config.final_logit_softcapping)?;
    }

    Ok(())
}

/// Single sliding-window attention layer.
///
/// Order matches HF modeling_gemma4.py:
///   residual = x
///   x = input_layernorm(x)              — RMSNorm (sandwich pre-attn)
///   q = q_proj(x); q_normed = q_norm(q) — RMSNorm over head_dim, no weight on v
///   k = k_proj(x); k_normed = k_norm(k)
///   v = v_proj(x)                        — no v_norm weight on sliding either
///   apply RoPE(q, k) with theta=10000, full head_dim=256
///   write K, V to KV cache at position `pos`
///   attn = flash_attention(q, K_cache, V_cache, window_size=1024)
///   x = o_proj(attn)
///   x = post_attention_layernorm(x)     — RMSNorm (sandwich post-attn)
///   x = residual + x                     — residual
///   residual = x
///   x = pre_feedforward_layernorm(x)    — RMSNorm (sandwich pre-FFN)
///   gate = gate_proj(x); up = up_proj(x)
///   ffn_hidden = gelu_tanh(gate) * up    — SwiGLU with gelu_pytorch_tanh
///   x = down_proj(ffn_hidden)
///   x = post_feedforward_layernorm(x)   — RMSNorm (sandwich post-FFN)
///   x = residual + x                     — residual
///   x = x * layer_scalar[0]              — learned per-layer scalar
fn sliding_layer_decode(
    _gpu: &mut Gpu,
    _config: &Gemma4Config,
    _lw: &SlidingLayerWeights,
    _pos: usize,
    _kv_cache: &mut llama::KvCache,
    _kv_layer_idx: usize,
    _scratch: &Gemma4Scratch,
) -> HipResult<()> {
    // TODO Phase 3b: implement body. Uses existing kernels:
    //   rmsnorm_f32, rmsnorm_batched, weight_gemv (MQ4 dispatch),
    //   rope_f32 or rope_batched_f32 (full 256-dim rotation),
    //   kv_cache_write_asym3_fused / q8_0 (format-specific),
    //   attention_flash_asym3 with window_size=1024 (or matching format),
    //   gelu_tanh_f32, mul_f32, scale_f32 (layer_scalar), add_inplace_f32.
    Err(hip_bridge::HipError::new(
        0,
        "gemma4::sliding_layer_decode not implemented (Phase 3b)",
    ))
}

/// Single full (global) attention layer with K=V weight sharing.
///
/// Key differences from sliding:
///   • head_dim = 512 (global_head_dim), 4 KV heads (vs sliding's 256 / 16).
///   • V is the *pre*-k_norm output of k_proj — CRITICAL ordering (line 1214
///     of modeling_gemma4.py). Get this wrong and V is silently mangled.
///   • v_norm is `no_scale=true` RMSNorm — divide only, no learned gain.
///     We pass `scratch.v_norm_ones_full` as the "weight" to the existing
///     rmsnorm kernel to preserve no-scale semantics.
///   • RoPE is partial_rotary_factor=0.25 proportional:
///     first 64 dims rotate with cos/sin from inv_freq[0..64];
///     dims 64..256 are NoPE (pass-through);
///     the rotate_half partner at 256..319 sees sin=0, cos=1 (no-op).
///     Net effect implemented via rope_partial_interleaved_f32(n_rot=64).
///   • No sliding window (window_size=0 = full causal).
fn full_layer_decode(
    _gpu: &mut Gpu,
    _config: &Gemma4Config,
    _lw: &FullLayerWeights,
    _pos: usize,
    _kv_cache: &mut llama::KvCache,
    _kv_layer_idx: usize,
    _scratch: &Gemma4Scratch,
) -> HipResult<()> {
    // TODO Phase 3b: implement body. Same kernel set as sliding layer minus:
    //   • no v_proj — v_src buffer borrows bytes from post-k_proj output
    //     BEFORE k_norm is applied. Clone k_raw → v_raw before k_norm.
    //   • rope_partial_interleaved_f32 with n_rot=64, head_dim=512.
    //   • attention kernel called with window_size=0.
    Err(hip_bridge::HipError::new(
        0,
        "gemma4::full_layer_decode not implemented (Phase 3b)",
    ))
}

/// Batched prefill. Phase 4.
pub fn forward_prefill_batch(
    _gpu: &mut Gpu,
    _weights: &Gemma4Weights,
    _config: &Gemma4Config,
    _tokens: &[u32],
    _start_pos: usize,
    _kv_sliding: &mut llama::KvCache,
    _kv_full: &mut llama::KvCache,
    _scratch: &Gemma4Scratch,
) -> HipResult<()> {
    Err(hip_bridge::HipError::new(0, "gemma4::forward_prefill_batch not implemented (Phase 4)"))
}
