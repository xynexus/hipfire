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
use crate::llama::{self, f16_to_f32, weight_gemv, WeightTensor, EmbeddingFormat};
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
    /// Host-side mirror of layer_scalar. Populated at load time so decode can
    /// call `gpu.scale_f32(x, layer_scalar_host)` without a D2H round-trip.
    pub layer_scalar_host: f32,

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
    /// Host-side mirror of layer_scalar. See SlidingLayerWeights for rationale.
    pub layer_scalar_host: f32,

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

// ─── Loading helpers ───────────────────────────────────────────────────

/// Decode a shape-[n] F16 or F32 tensor from HFQ into an F32 host Vec.
fn load_f32_vec(hfq: &HfqFile, name: &str, expected_n: usize) -> HipResult<Vec<f32>> {
    let (info, data) = hfq.tensor_data(name).ok_or_else(|| {
        hip_bridge::HipError::new(0, &format!("tensor not found: {name}"))
    })?;
    let n: usize = info.shape.iter().map(|&s| s as usize).product();
    if n != expected_n {
        return Err(hip_bridge::HipError::new(
            0, &format!("shape mismatch for {name}: expected {expected_n}, got {n}"),
        ));
    }
    let f32_data = match info.quant_type {
        1 => data.chunks_exact(2)
                 .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                 .collect(),
        2 => data.chunks_exact(4)
                 .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                 .collect(),
        qt => return Err(hip_bridge::HipError::new(
            0, &format!("expected F16/F32 for {name}, got qt={qt}"),
        )),
    };
    Ok(f32_data)
}

/// Load a Gemma 4 RMSNorm weight — `x * weight` form, NO +1 shift.
///
/// Distinct from qwen35::load_norm_weight which shifts by +1 for HF Gemma
/// 2/3-style `x * (1 + weight)`. Gemma 4 uses plain `x * weight` with weights
/// initialized to 1.0 (see modeling_gemma4.py::Gemma4RMSNorm line 157).
fn load_gemma4_norm(hfq: &HfqFile, gpu: &mut Gpu, name: &str, dim: usize)
    -> HipResult<GpuTensor>
{
    let f32_data = load_f32_vec(hfq, name, dim)?;
    gpu.upload_f32(&f32_data, &[dim])
}

/// Load a 256-element head-dim Q/K RMSNorm weight. Same semantics as
/// `load_gemma4_norm` but scoped to the attention head_dim (256 on sliding,
/// 512 on full).
fn load_gemma4_head_norm(hfq: &HfqFile, gpu: &mut Gpu, name: &str, head_dim: usize)
    -> HipResult<GpuTensor>
{
    load_gemma4_norm(hfq, gpu, name, head_dim)
}

/// Load the per-layer `layer_scalar` — shape-[1] BF16/F16 tensor — returning
/// both a GPU-resident [1]-tensor (for potential batched use) and its host-side
/// f32 value (used by the decode path to call `scale_f32(x, cpu_scalar)`).
fn load_layer_scalar(hfq: &HfqFile, gpu: &mut Gpu, name: &str)
    -> HipResult<(GpuTensor, f32)>
{
    let data = load_f32_vec(hfq, name, 1)?;
    let host_val = data[0];
    let gpu_tensor = gpu.upload_f32(&data, &[1])?;
    Ok((gpu_tensor, host_val))
}

/// Load a quantized projection weight. Mirrors qwen35::load_weight_tensor_raw
/// but uses the Gemma 4 tensor-name convention (`model.language_model.<name>`).
fn load_gemma4_weight(hfq: &HfqFile, gpu: &mut Gpu, name: &str, m: usize, k: usize)
    -> HipResult<WeightTensor>
{
    let (info, data) = hfq.tensor_data(name).ok_or_else(|| {
        hip_bridge::HipError::new(0, &format!("tensor not found: {name}"))
    })?;
    let dtype = match info.quant_type {
        1 => {
            // F16 → upload as f32
            let f32_data: Vec<f32> = data.chunks_exact(2)
                .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect();
            let bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(f32_data.as_ptr() as *const u8, f32_data.len() * 4)
            };
            let buf = gpu.upload_raw(bytes, &[m, k])?;
            return Ok(WeightTensor { buf, gpu_dtype: DType::F32, m, k, row_stride: 0 });
        }
        3  => DType::Q8_0,
        4  => DType::Q4K,
        6  => DType::HFQ4G256,
        7  => DType::HFQ4G128,
        8  => DType::HFQ6G256,
        9  => DType::HFQ2G256,
        10 => DType::HFQ2G128,
        11 => DType::HFQ3G256,
        12 => DType::HFQ3G128,
        13 => DType::MQ4G256,
        14 => DType::MQ8G256,
        15 => DType::MQ6G256,
        qt => return Err(hip_bridge::HipError::new(
            0, &format!("unsupported quant_type {qt} for {name}"),
        )),
    };
    let buf = gpu.upload_raw(data, &[data.len()])?;
    Ok(WeightTensor { buf, gpu_dtype: dtype, m, k, row_stride: 0 })
}

/// Load Gemma 4 text model weights from an HFQ file.
///
/// Design notes:
///   - `lm_head` aliases the `embed_tokens` GPU bytes (tied weights). We upload
///     the embed data once and create a second WeightTensor whose DeviceBuffer
///     points at the same allocation via `buf.alias()`. `Gemma4Weights::free_gpu`
///     skips freeing the LM head to avoid a double-free.
///   - Vision tensors are skipped here — Phase 7 `gemma4_vision::load_weights`
///     picks those up from the same HFQ file in a separate pass.
///   - The `v_norm_ones_full` ones-filled scratch buffer is populated here so
///     the forward pass never has to manage one-time init state.
pub fn load_weights(hfq: &HfqFile, config: &Gemma4Config, gpu: &mut Gpu)
    -> HipResult<Gemma4Weights>
{
    eprintln!("gemma4: loading embed_tokens...");
    let embed_name = "model.language_model.embed_tokens.weight";
    let (embed_info, embed_data) = hfq.tensor_data(embed_name).ok_or_else(|| {
        hip_bridge::HipError::new(0, "embed_tokens not found in HFQ")
    })?;
    let (embed_tokens, embd_format) = match embed_info.quant_type {
        3 => {
            eprintln!("  (Q8_0 / Q8F16, {} MB)", embed_data.len() / 1_000_000);
            (gpu.upload_raw(embed_data, &[embed_data.len()])?, EmbeddingFormat::Q8_0)
        }
        6 => {
            eprintln!("  (HFQ4-G256, {} MB)", embed_data.len() / 1_000_000);
            (gpu.upload_raw(embed_data, &[embed_data.len()])?, EmbeddingFormat::HFQ4G256)
        }
        7 => {
            eprintln!("  (HFQ4-G128, {} MB)", embed_data.len() / 1_000_000);
            (gpu.upload_raw(embed_data, &[embed_data.len()])?, EmbeddingFormat::HFQ4G128)
        }
        1 => {
            eprintln!("  (F16 → F32)");
            let f32_data: Vec<f32> = embed_data.chunks_exact(2)
                .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect();
            (gpu.upload_f32(&f32_data, &[config.vocab_size, config.dim])?, EmbeddingFormat::F32)
        }
        qt => return Err(hip_bridge::HipError::new(
            0, &format!("unsupported embed quant_type {qt}"),
        )),
    };

    // Tied LM head: WeightTensor whose buffer aliases the embed allocation.
    // free_gpu skips freeing this — embed_tokens owns the bytes.
    let lm_head = {
        let alias_buf = unsafe { embed_tokens.buf.alias() };
        let dtype = match embd_format {
            EmbeddingFormat::Q8_0    => DType::Q8_0,
            EmbeddingFormat::HFQ4G256 => DType::HFQ4G256,
            EmbeddingFormat::HFQ4G128 => DType::HFQ4G128,
            EmbeddingFormat::F32     => DType::F32,
            EmbeddingFormat::Q4K     => DType::Q4K,
        };
        let alias_tensor = GpuTensor {
            buf: alias_buf,
            shape: embed_tokens.shape.clone(),
            dtype,
        };
        WeightTensor { buf: alias_tensor, gpu_dtype: dtype, m: config.vocab_size, k: config.dim, row_stride: 0 }
    };

    eprintln!("gemma4: loading final norm...");
    let final_norm = load_gemma4_norm(hfq, gpu, "model.language_model.norm.weight", config.dim)?;

    eprintln!("gemma4: loading {} layers...", config.n_layers);
    let mut layers = Vec::with_capacity(config.n_layers);
    for i in 0..config.n_layers {
        let p = format!("model.language_model.layers.{i}");
        match config.layer_types[i] {
            LayerType::Sliding => {
                let hd = config.sliding_head_dim;
                let kv_dim = config.sliding_n_kv_heads * hd;
                let q_dim = config.n_heads * hd;
                let (layer_scalar, layer_scalar_host) =
                    load_layer_scalar(hfq, gpu, &format!("{p}.layer_scalar"))?;
                layers.push(LayerWeights::Sliding(SlidingLayerWeights {
                    input_layernorm: load_gemma4_norm(hfq, gpu,
                        &format!("{p}.input_layernorm.weight"), config.dim)?,
                    post_attention_layernorm: load_gemma4_norm(hfq, gpu,
                        &format!("{p}.post_attention_layernorm.weight"), config.dim)?,
                    pre_feedforward_layernorm: load_gemma4_norm(hfq, gpu,
                        &format!("{p}.pre_feedforward_layernorm.weight"), config.dim)?,
                    post_feedforward_layernorm: load_gemma4_norm(hfq, gpu,
                        &format!("{p}.post_feedforward_layernorm.weight"), config.dim)?,
                    layer_scalar,
                    layer_scalar_host,
                    q_proj: load_gemma4_weight(hfq, gpu,
                        &format!("{p}.self_attn.q_proj.weight"), q_dim, config.dim)?,
                    k_proj: load_gemma4_weight(hfq, gpu,
                        &format!("{p}.self_attn.k_proj.weight"), kv_dim, config.dim)?,
                    v_proj: load_gemma4_weight(hfq, gpu,
                        &format!("{p}.self_attn.v_proj.weight"), kv_dim, config.dim)?,
                    o_proj: load_gemma4_weight(hfq, gpu,
                        &format!("{p}.self_attn.o_proj.weight"), config.dim, q_dim)?,
                    q_norm: load_gemma4_head_norm(hfq, gpu,
                        &format!("{p}.self_attn.q_norm.weight"), hd)?,
                    k_norm: load_gemma4_head_norm(hfq, gpu,
                        &format!("{p}.self_attn.k_norm.weight"), hd)?,
                    gate_proj: load_gemma4_weight(hfq, gpu,
                        &format!("{p}.mlp.gate_proj.weight"), config.hidden_dim, config.dim)?,
                    up_proj: load_gemma4_weight(hfq, gpu,
                        &format!("{p}.mlp.up_proj.weight"), config.hidden_dim, config.dim)?,
                    down_proj: load_gemma4_weight(hfq, gpu,
                        &format!("{p}.mlp.down_proj.weight"), config.dim, config.hidden_dim)?,
                }));
            }
            LayerType::Full => {
                let hd = config.full_head_dim;
                let kv_dim = config.full_n_kv_heads * hd;
                let q_dim = config.n_heads * hd;
                let (layer_scalar, layer_scalar_host) =
                    load_layer_scalar(hfq, gpu, &format!("{p}.layer_scalar"))?;
                layers.push(LayerWeights::Full(FullLayerWeights {
                    input_layernorm: load_gemma4_norm(hfq, gpu,
                        &format!("{p}.input_layernorm.weight"), config.dim)?,
                    post_attention_layernorm: load_gemma4_norm(hfq, gpu,
                        &format!("{p}.post_attention_layernorm.weight"), config.dim)?,
                    pre_feedforward_layernorm: load_gemma4_norm(hfq, gpu,
                        &format!("{p}.pre_feedforward_layernorm.weight"), config.dim)?,
                    post_feedforward_layernorm: load_gemma4_norm(hfq, gpu,
                        &format!("{p}.post_feedforward_layernorm.weight"), config.dim)?,
                    layer_scalar,
                    layer_scalar_host,
                    q_proj: load_gemma4_weight(hfq, gpu,
                        &format!("{p}.self_attn.q_proj.weight"), q_dim, config.dim)?,
                    k_proj: load_gemma4_weight(hfq, gpu,
                        &format!("{p}.self_attn.k_proj.weight"), kv_dim, config.dim)?,
                    // no v_proj on full layers — V reuses k_proj's pre-norm output.
                    o_proj: load_gemma4_weight(hfq, gpu,
                        &format!("{p}.self_attn.o_proj.weight"), config.dim, q_dim)?,
                    q_norm: load_gemma4_head_norm(hfq, gpu,
                        &format!("{p}.self_attn.q_norm.weight"), hd)?,
                    k_norm: load_gemma4_head_norm(hfq, gpu,
                        &format!("{p}.self_attn.k_norm.weight"), hd)?,
                    // no v_norm weight — v_norm is no-scale (ones buffer passed at decode time).
                    gate_proj: load_gemma4_weight(hfq, gpu,
                        &format!("{p}.mlp.gate_proj.weight"), config.hidden_dim, config.dim)?,
                    up_proj: load_gemma4_weight(hfq, gpu,
                        &format!("{p}.mlp.up_proj.weight"), config.hidden_dim, config.dim)?,
                    down_proj: load_gemma4_weight(hfq, gpu,
                        &format!("{p}.mlp.down_proj.weight"), config.dim, config.hidden_dim)?,
                }));
            }
        }
    }
    eprintln!("gemma4: loaded all {} layers", config.n_layers);

    Ok(Gemma4Weights {
        embed_tokens,
        embd_format,
        lm_head,
        final_norm,
        layers,
    })
}

/// One-time init for the scratch buffers that must hold a constant value
/// across forward passes (notably the ones-filled `v_norm_ones_full`).
/// Call once after `Gemma4Scratch::new` before the first forward pass.
pub fn init_scratch_constants(gpu: &mut Gpu, scratch: &Gemma4Scratch, full_head_dim: usize)
    -> HipResult<()>
{
    let ones: Vec<f32> = vec![1.0; full_head_dim];
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(ones.as_ptr() as *const u8, ones.len() * 4)
    };
    gpu.hip.memcpy_htod(&scratch.v_norm_ones_full.buf, bytes)?;
    Ok(())
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
/// Order matches HF modeling_gemma4.py::Gemma4TextDecoderLayer +
/// Gemma4TextAttention (sliding branch):
///   residual = x
///   x = input_layernorm(x)              — RMSNorm (sandwich pre-attn)
///   q = q_proj(x); q = q_norm(q)        — RMSNorm over head_dim=256
///   k = k_proj(x); k = k_norm(k)
///   v = v_proj(x)                        — sliding has its own v_proj
///   RoPE(q, k) with rotate_half, theta=10000, full head_dim=256
///   write K, V to KV cache at position `pos`
///   attn = flash_attention(q, K, V, window_size=1024, scale=1.0 effective)
///   x = o_proj(attn)
///   x = post_attention_layernorm(x)     — RMSNorm (sandwich post-attn)
///   x = residual + x
///   residual = x
///   x = pre_feedforward_layernorm(x)    — RMSNorm (sandwich pre-FFN)
///   gate = gate_proj(x); up = up_proj(x)
///   ffn = gelu_pytorch_tanh(gate) * up  — SwiGLU
///   x = down_proj(ffn)
///   x = post_feedforward_layernorm(x)   — RMSNorm (sandwich post-FFN)
///   x = residual + x
///   x = x * layer_scalar                — learned per-layer scalar
///
/// Gemma 4 attention uses `scaling=1.0` in HF (see modeling_gemma4.py line 1143).
/// Our flash kernels bake in `scale = 1/sqrt(head_dim)`; we compensate by
/// pre-scaling Q by sqrt(head_dim) so the effective scale is 1.0.
fn sliding_layer_decode(
    gpu: &mut Gpu,
    config: &Gemma4Config,
    lw: &SlidingLayerWeights,
    pos: usize,
    kv_cache: &mut llama::KvCache,
    kv_layer_idx: usize,
    scratch: &Gemma4Scratch,
) -> HipResult<()> {
    let dim = config.dim;
    let head_dim = config.sliding_head_dim;
    let n_heads = config.n_heads;
    let n_kv = config.sliding_n_kv_heads;
    let dim_bytes = dim * 4;

    // residual = x
    gpu.hip.memcpy_dtod(&scratch.residual.buf, &scratch.x.buf, dim_bytes)?;

    // tmp = input_layernorm(x)
    gpu.rmsnorm_f32(&scratch.x, &lw.input_layernorm, &scratch.tmp, config.norm_eps)?;

    // Q/K/V projections: q[n_heads*head_dim], k/v[n_kv*head_dim].
    weight_gemv(gpu, &lw.q_proj, &scratch.tmp, &scratch.q)?;
    weight_gemv(gpu, &lw.k_proj, &scratch.tmp, &scratch.k)?;
    weight_gemv(gpu, &lw.v_proj, &scratch.tmp, &scratch.v)?;

    // q_norm + k_norm across head_dim (in-place).
    gpu.rmsnorm_batched(&scratch.q, &lw.q_norm, &scratch.q, n_heads, head_dim, config.norm_eps)?;
    gpu.rmsnorm_batched(&scratch.k, &lw.k_norm, &scratch.k, n_kv, head_dim, config.norm_eps)?;

    // Pre-scale Q by sqrt(head_dim) so the flash-attn kernel's internal
    // 1/sqrt(head_dim) cancels, leaving the effective Gemma 4 scale of 1.0.
    // Only the first n_heads*head_dim elements of scratch.q are live.
    gpu.scale_f32(&scratch.q, (head_dim as f32).sqrt())?;

    // Full rotate_half RoPE, theta=10000, head_dim=256 (all dims rotate).
    gpu.rope_f32(&scratch.q, &scratch.k, &scratch.pos_buf,
        n_heads, n_kv, head_dim, config.sliding_rope_theta)?;

    // KV cache write + flash attention with window_size=1024.
    // Branch on cache quant mode, same as qwen35::run_fa_layer_body.
    if kv_cache.quant_asym3 {
        let ct = kv_cache.givens_cos.as_ref().unwrap();
        let st = kv_cache.givens_sin.as_ref().unwrap();
        gpu.kv_cache_write_asym3_fused(
            &kv_cache.k_gpu[kv_layer_idx], &kv_cache.v_gpu[kv_layer_idx],
            &scratch.k, &scratch.v, &scratch.pos_buf, ct, st, n_kv, head_dim)?;
        gpu.attention_flash_asym3(
            &scratch.q, &kv_cache.k_gpu[kv_layer_idx], &kv_cache.v_gpu[kv_layer_idx],
            &scratch.attn_out, &scratch.pos_buf, ct, st, pos + 1,
            n_heads, n_kv, head_dim, kv_cache.max_seq,
            &scratch.flash_partials,
            config.sliding_window as u32,
        )?;
    } else if kv_cache.quant_asym4 {
        let ct = kv_cache.givens_cos.as_ref().unwrap();
        let st = kv_cache.givens_sin.as_ref().unwrap();
        gpu.kv_cache_write_asym4_fused(
            &kv_cache.k_gpu[kv_layer_idx], &kv_cache.v_gpu[kv_layer_idx],
            &scratch.k, &scratch.v, &scratch.pos_buf, ct, st, n_kv, head_dim)?;
        gpu.attention_flash_asym4(
            &scratch.q, &kv_cache.k_gpu[kv_layer_idx], &kv_cache.v_gpu[kv_layer_idx],
            &scratch.attn_out, &scratch.pos_buf, ct, st, pos + 1,
            n_heads, n_kv, head_dim, kv_cache.max_seq,
            &scratch.flash_partials,
            config.sliding_window as u32,
        )?;
    } else if kv_cache.quant_asym2 {
        let ct = kv_cache.givens_cos.as_ref().unwrap();
        let st = kv_cache.givens_sin.as_ref().unwrap();
        gpu.kv_cache_write_asym2_fused(
            &kv_cache.k_gpu[kv_layer_idx], &kv_cache.v_gpu[kv_layer_idx],
            &scratch.k, &scratch.v, &scratch.pos_buf, ct, st, n_kv, head_dim)?;
        gpu.attention_flash_asym2(
            &scratch.q, &kv_cache.k_gpu[kv_layer_idx], &kv_cache.v_gpu[kv_layer_idx],
            &scratch.attn_out, &scratch.pos_buf, ct, st, pos + 1,
            n_heads, n_kv, head_dim, kv_cache.max_seq,
            &scratch.flash_partials,
            config.sliding_window as u32,
        )?;
    } else if kv_cache.quant_q8 {
        gpu.kv_cache_write_q8_0(&kv_cache.k_gpu[kv_layer_idx], &scratch.k, &scratch.pos_buf, n_kv, head_dim)?;
        gpu.kv_cache_write_q8_0(&kv_cache.v_gpu[kv_layer_idx], &scratch.v, &scratch.pos_buf, n_kv, head_dim)?;
        gpu.attention_flash_q8_0(
            &scratch.q, &kv_cache.k_gpu[kv_layer_idx], &kv_cache.v_gpu[kv_layer_idx],
            &scratch.attn_out, &scratch.pos_buf, pos + 1,
            n_heads, n_kv, head_dim, kv_cache.max_seq,
            &scratch.flash_partials,
            config.sliding_window as u32,
        )?;
    } else {
        // Plain FP32 KV path (kvf16 / kvfp32).
        let kv_dim = n_kv * head_dim;
        gpu.kv_cache_write(&kv_cache.k_gpu[kv_layer_idx], &scratch.k, &scratch.pos_buf, kv_dim)?;
        gpu.kv_cache_write(&kv_cache.v_gpu[kv_layer_idx], &scratch.v, &scratch.pos_buf, kv_dim)?;
        // No sliding-window support in the plain attention_f32 kernel; this
        // path is used only for debugging (mostly Qwen3.5 kvf16 mode).
        return Err(hip_bridge::HipError::new(
            0,
            "gemma4 requires a quantized KV cache (asym2/asym3/asym4/q8); kvf16 lacks sliding-window support",
        ));
    }

    // o_proj → tmp (reuse tmp, overwriting input_layernorm output).
    weight_gemv(gpu, &lw.o_proj, &scratch.attn_out, &scratch.tmp)?;

    // Sandwich post-attn norm (in-place on tmp).
    gpu.rmsnorm_f32(&scratch.tmp, &lw.post_attention_layernorm, &scratch.tmp, config.norm_eps)?;

    // x = residual + tmp. (Reset x first since earlier ops mutated it.)
    gpu.hip.memcpy_dtod(&scratch.x.buf, &scratch.residual.buf, dim_bytes)?;
    gpu.add_inplace_f32(&scratch.x, &scratch.tmp)?;

    // residual = x (for the FFN residual stream).
    gpu.hip.memcpy_dtod(&scratch.residual.buf, &scratch.x.buf, dim_bytes)?;

    // Pre-FFN norm.
    gpu.rmsnorm_f32(&scratch.x, &lw.pre_feedforward_layernorm, &scratch.tmp, config.norm_eps)?;

    // SwiGLU(gelu_pytorch_tanh): gate_proj, up_proj, gelu_tanh(gate) * up → down_proj.
    weight_gemv(gpu, &lw.gate_proj, &scratch.tmp, &scratch.gate_ffn)?;
    weight_gemv(gpu, &lw.up_proj, &scratch.tmp, &scratch.up_ffn)?;
    gpu.gelu_tanh_f32(&scratch.gate_ffn, &scratch.ffn_hidden, config.hidden_dim)?;
    gpu.mul_f32(&scratch.ffn_hidden, &scratch.up_ffn, &scratch.ffn_hidden)?;
    weight_gemv(gpu, &lw.down_proj, &scratch.ffn_hidden, &scratch.ffn_out)?;

    // Sandwich post-FFN norm.
    gpu.rmsnorm_f32(&scratch.ffn_out, &lw.post_feedforward_layernorm, &scratch.tmp, config.norm_eps)?;

    // x = residual + tmp (again, reset x from saved residual).
    gpu.hip.memcpy_dtod(&scratch.x.buf, &scratch.residual.buf, dim_bytes)?;
    gpu.add_inplace_f32(&scratch.x, &scratch.tmp)?;

    // Learned per-layer scalar multiplier.
    gpu.scale_f32(&scratch.x, lw.layer_scalar_host)?;

    Ok(())
}

/// Single full (global) attention layer with K=V weight sharing.
///
/// Key differences from sliding:
///   • head_dim = 512 (global_head_dim), 4 KV heads (vs sliding's 256 / 16).
///   • V is the *pre*-k_norm output of k_proj — CRITICAL ordering (line 1214
///     of modeling_gemma4.py). In Python:
///         key_states = k_proj(x)
///         value_states = v_proj(x) if v_proj else key_states   # bound BEFORE norm
///         key_states   = k_norm(key_states)                    # rebind, value_states holds pre-norm
///         value_states = v_norm(value_states)
///     Our translation: write k_proj output into `scratch.k`, memcpy into
///     `scratch.v`, then apply k_norm in-place on scratch.k.
///   • v_norm is `no_scale=true` RMSNorm — divide only, no learned gain.
///     We call the existing `rmsnorm_batched` with the ones-filled
///     `scratch.v_norm_ones_full` as the weight vector.
///   • RoPE is partial_rotary_factor=0.25 proportional:
///     pairs (i, i+head_dim/2) for i in [0, 64) rotate with theta=1e6;
///     pairs [64, 256) are NoPE (identity). See `rope_partial_halved_f32`.
///   • No sliding window (window_size=0 = full causal).
///   • Attention scale = 1.0 (same as sliding — Gemma 4 sets
///     `self.scaling = 1.0`; we compensate by pre-scaling Q by sqrt(head_dim)).
fn full_layer_decode(
    gpu: &mut Gpu,
    config: &Gemma4Config,
    lw: &FullLayerWeights,
    pos: usize,
    kv_cache: &mut llama::KvCache,
    kv_layer_idx: usize,
    scratch: &Gemma4Scratch,
) -> HipResult<()> {
    let dim = config.dim;
    let head_dim = config.full_head_dim;
    let n_heads = config.n_heads;
    let n_kv = config.full_n_kv_heads;
    let dim_bytes = dim * 4;
    let kv_bytes = n_kv * head_dim * 4;

    // residual = x
    gpu.hip.memcpy_dtod(&scratch.residual.buf, &scratch.x.buf, dim_bytes)?;

    // tmp = input_layernorm(x)
    gpu.rmsnorm_f32(&scratch.x, &lw.input_layernorm, &scratch.tmp, config.norm_eps)?;

    // Q + K projections. V is derived from K's pre-norm output below.
    weight_gemv(gpu, &lw.q_proj, &scratch.tmp, &scratch.q)?;
    weight_gemv(gpu, &lw.k_proj, &scratch.tmp, &scratch.k)?;

    // CRITICAL: capture pre-k_norm bytes as V before applying k_norm.
    gpu.hip.memcpy_dtod(&scratch.v.buf, &scratch.k.buf, kv_bytes)?;

    // q_norm, k_norm, and no-scale v_norm (all head_dim = 512).
    gpu.rmsnorm_batched(&scratch.q, &lw.q_norm, &scratch.q, n_heads, head_dim, config.norm_eps)?;
    gpu.rmsnorm_batched(&scratch.k, &lw.k_norm, &scratch.k, n_kv, head_dim, config.norm_eps)?;
    gpu.rmsnorm_batched(&scratch.v, &scratch.v_norm_ones_full, &scratch.v,
        n_kv, head_dim, config.norm_eps)?;

    // Pre-scale Q by sqrt(head_dim=512) so the flash kernel's 1/sqrt(head_dim)
    // cancels (Gemma 4 attention scaling is 1.0).
    gpu.scale_f32(&scratch.q, (head_dim as f32).sqrt())?;

    // Proportional RoPE: rotate_half of the first 64 pairs of every 512-dim head.
    let n_rot_pairs = ((head_dim as f32) * config.full_partial_rotary_factor * 0.5) as usize;
    gpu.rope_partial_halved_f32(&scratch.q, &scratch.k, &scratch.pos_buf,
        n_heads, n_kv, head_dim, n_rot_pairs, config.full_rope_theta)?;

    // KV cache write + flash attention with window_size=0 (full causal).
    if kv_cache.quant_asym3 {
        let ct = kv_cache.givens_cos.as_ref().unwrap();
        let st = kv_cache.givens_sin.as_ref().unwrap();
        gpu.kv_cache_write_asym3_fused(
            &kv_cache.k_gpu[kv_layer_idx], &kv_cache.v_gpu[kv_layer_idx],
            &scratch.k, &scratch.v, &scratch.pos_buf, ct, st, n_kv, head_dim)?;
        gpu.attention_flash_asym3(
            &scratch.q, &kv_cache.k_gpu[kv_layer_idx], &kv_cache.v_gpu[kv_layer_idx],
            &scratch.attn_out, &scratch.pos_buf, ct, st, pos + 1,
            n_heads, n_kv, head_dim, kv_cache.max_seq,
            &scratch.flash_partials,
            0u32,
        )?;
    } else if kv_cache.quant_asym4 {
        let ct = kv_cache.givens_cos.as_ref().unwrap();
        let st = kv_cache.givens_sin.as_ref().unwrap();
        gpu.kv_cache_write_asym4_fused(
            &kv_cache.k_gpu[kv_layer_idx], &kv_cache.v_gpu[kv_layer_idx],
            &scratch.k, &scratch.v, &scratch.pos_buf, ct, st, n_kv, head_dim)?;
        gpu.attention_flash_asym4(
            &scratch.q, &kv_cache.k_gpu[kv_layer_idx], &kv_cache.v_gpu[kv_layer_idx],
            &scratch.attn_out, &scratch.pos_buf, ct, st, pos + 1,
            n_heads, n_kv, head_dim, kv_cache.max_seq,
            &scratch.flash_partials,
            0u32,
        )?;
    } else if kv_cache.quant_asym2 {
        let ct = kv_cache.givens_cos.as_ref().unwrap();
        let st = kv_cache.givens_sin.as_ref().unwrap();
        gpu.kv_cache_write_asym2_fused(
            &kv_cache.k_gpu[kv_layer_idx], &kv_cache.v_gpu[kv_layer_idx],
            &scratch.k, &scratch.v, &scratch.pos_buf, ct, st, n_kv, head_dim)?;
        gpu.attention_flash_asym2(
            &scratch.q, &kv_cache.k_gpu[kv_layer_idx], &kv_cache.v_gpu[kv_layer_idx],
            &scratch.attn_out, &scratch.pos_buf, ct, st, pos + 1,
            n_heads, n_kv, head_dim, kv_cache.max_seq,
            &scratch.flash_partials,
            0u32,
        )?;
    } else if kv_cache.quant_q8 {
        gpu.kv_cache_write_q8_0(&kv_cache.k_gpu[kv_layer_idx], &scratch.k, &scratch.pos_buf, n_kv, head_dim)?;
        gpu.kv_cache_write_q8_0(&kv_cache.v_gpu[kv_layer_idx], &scratch.v, &scratch.pos_buf, n_kv, head_dim)?;
        gpu.attention_flash_q8_0(
            &scratch.q, &kv_cache.k_gpu[kv_layer_idx], &kv_cache.v_gpu[kv_layer_idx],
            &scratch.attn_out, &scratch.pos_buf, pos + 1,
            n_heads, n_kv, head_dim, kv_cache.max_seq,
            &scratch.flash_partials,
            0u32,
        )?;
    } else {
        return Err(hip_bridge::HipError::new(
            0,
            "gemma4 full-attn layer requires a quantized KV cache (asym2/asym3/asym4/q8)",
        ));
    }

    // o_proj → tmp.
    weight_gemv(gpu, &lw.o_proj, &scratch.attn_out, &scratch.tmp)?;

    // Sandwich post-attn norm.
    gpu.rmsnorm_f32(&scratch.tmp, &lw.post_attention_layernorm, &scratch.tmp, config.norm_eps)?;

    // x = residual + tmp.
    gpu.hip.memcpy_dtod(&scratch.x.buf, &scratch.residual.buf, dim_bytes)?;
    gpu.add_inplace_f32(&scratch.x, &scratch.tmp)?;

    // Save new residual.
    gpu.hip.memcpy_dtod(&scratch.residual.buf, &scratch.x.buf, dim_bytes)?;

    // Pre-FFN norm.
    gpu.rmsnorm_f32(&scratch.x, &lw.pre_feedforward_layernorm, &scratch.tmp, config.norm_eps)?;

    // SwiGLU with gelu_pytorch_tanh activation.
    weight_gemv(gpu, &lw.gate_proj, &scratch.tmp, &scratch.gate_ffn)?;
    weight_gemv(gpu, &lw.up_proj, &scratch.tmp, &scratch.up_ffn)?;
    gpu.gelu_tanh_f32(&scratch.gate_ffn, &scratch.ffn_hidden, config.hidden_dim)?;
    gpu.mul_f32(&scratch.ffn_hidden, &scratch.up_ffn, &scratch.ffn_hidden)?;
    weight_gemv(gpu, &lw.down_proj, &scratch.ffn_hidden, &scratch.ffn_out)?;

    // Sandwich post-FFN norm.
    gpu.rmsnorm_f32(&scratch.ffn_out, &lw.post_feedforward_layernorm, &scratch.tmp, config.norm_eps)?;

    // x = residual + tmp.
    gpu.hip.memcpy_dtod(&scratch.x.buf, &scratch.residual.buf, dim_bytes)?;
    gpu.add_inplace_f32(&scratch.x, &scratch.tmp)?;

    // Learned per-layer scalar multiplier.
    gpu.scale_f32(&scratch.x, lw.layer_scalar_host)?;

    Ok(())
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
