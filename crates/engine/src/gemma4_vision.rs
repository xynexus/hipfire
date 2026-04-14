//! Gemma 4 vision tower: SigLIP-style ViT with factored 2D positional embedding.
//!
//! Architectural features:
//!   • Patch embedder is `Linear(3*patch*patch → hidden)` (not Conv2D).
//!     Pixels are pre-unfolded `(C,H,W) → (Hp,Wp,ph,pw,C) → permute → flat`.
//!     In-model: pixels are remapped `2*(x-0.5)` from [0,1] to [-1,1].
//!   • 2D factored position embedding: `position_embedding_table[2, 10240, 1152]`,
//!     one-hot on (x,y) positions → sum the two axes.
//!   • Sandwich RMSNorm per layer (same as text model).
//!   • 2D RoPE on Q/K per patch (rope_theta=100, head_dim halves rotate per axis).
//!   • `standardize`: `(h - std_bias) * std_scale` with [1152] buffers.
//!   • CPU-side avg-pool k=3 after the encoder → 280 soft tokens / image.
//!   • Multimodal embedder: `RMSNorm(no_scale) → Linear(1152 → 5376)`.
//!
//! Vision tensor names carry an extra `.linear.` segment that text layers don't
//! (e.g. `q_proj.linear.weight` vs text `q_proj.weight`). Also, vision layers
//! use `Gemma4ClippableLinear` wrappers — on 31B `use_clipped_linears=False`
//! so they behave as plain Linear.
//!
//! Phase 1 (this file): config parser + stub weight struct. Real forward pass
//! and weight loader land in Phase 7.

use crate::hfq::HfqFile;
use hip_bridge::HipResult;
use rdna_compute::{Gpu, GpuTensor};

// ─── Config ─────────────────────────────────────────────────────────────

#[derive(Debug)]
pub struct Gemma4VisionConfig {
    pub hidden_size: usize,                  // 1152
    pub num_layers: usize,                   // 27
    pub num_heads: usize,                    // 16
    pub head_dim: usize,                     // 72
    pub intermediate_size: usize,            // 4304
    pub patch_size: usize,                   // 16
    pub position_embedding_size: usize,      // 10240 — per-axis one-hot table depth
    pub pooling_kernel_size: usize,          // 3 — avg-pool kernel
    pub default_output_length: usize,        // 280 — soft tokens per image
    pub rope_theta: f32,                     // 100.0
    pub norm_eps: f32,                       // 1e-6
    pub standardize: bool,                   // true on 31B
    /// When true (rare), the ClippableLinear wrappers apply output clamps.
    /// 31B sets false. Reserved so future E-variants parse cleanly.
    pub use_clipped_linears: bool,
    /// Output projection dim (= text hidden_size). 5376 for 31B.
    pub out_hidden_size: usize,
}

pub fn vision_config_from_hfq(hfq: &HfqFile) -> Option<Gemma4VisionConfig> {
    let meta: serde_json::Value = serde_json::from_str(&hfq.metadata_json).ok()?;
    let config = meta.get("config")?;
    let vc = config.get("vision_config")?;
    if vc.is_null() { return None; }

    let hidden_size = vc.get("hidden_size")?.as_u64()? as usize;
    let num_layers = vc.get("num_hidden_layers").and_then(|v| v.as_u64()).unwrap_or(27) as usize;
    let num_heads = vc.get("num_attention_heads").and_then(|v| v.as_u64()).unwrap_or(16) as usize;
    let head_dim = vc.get("head_dim").and_then(|v| v.as_u64())
        .unwrap_or((hidden_size / num_heads) as u64) as usize;
    let intermediate_size = vc.get("intermediate_size").and_then(|v| v.as_u64()).unwrap_or(4304) as usize;
    let patch_size = vc.get("patch_size").and_then(|v| v.as_u64()).unwrap_or(16) as usize;
    let position_embedding_size = vc.get("position_embedding_size")
        .and_then(|v| v.as_u64()).unwrap_or(10240) as usize;
    let pooling_kernel_size = vc.get("pooling_kernel_size")
        .and_then(|v| v.as_u64()).unwrap_or(3) as usize;
    let default_output_length = vc.get("default_output_length")
        .and_then(|v| v.as_u64()).unwrap_or(280) as usize;
    let rope_theta = vc.get("rope_parameters")
        .and_then(|rp| rp.get("rope_theta"))
        .and_then(|v| v.as_f64()).unwrap_or(100.0) as f32;
    let norm_eps = vc.get("rms_norm_eps").and_then(|v| v.as_f64()).unwrap_or(1e-6) as f32;
    let standardize = vc.get("standardize").and_then(|v| v.as_bool()).unwrap_or(true);
    let use_clipped_linears = vc.get("use_clipped_linears").and_then(|v| v.as_bool()).unwrap_or(false);

    let out_hidden_size = config.get("text_config")
        .and_then(|tc| tc.get("hidden_size")).and_then(|v| v.as_u64())
        .unwrap_or(5376) as usize;

    Some(Gemma4VisionConfig {
        hidden_size, num_layers, num_heads, head_dim,
        intermediate_size, patch_size,
        position_embedding_size, pooling_kernel_size, default_output_length,
        rope_theta, norm_eps,
        standardize, use_clipped_linears,
        out_hidden_size,
    })
}

// ─── Weights ────────────────────────────────────────────────────────────

pub struct Gemma4VisionLayerWeights {
    pub input_layernorm: GpuTensor,
    pub post_attention_layernorm: GpuTensor,
    pub pre_feedforward_layernorm: GpuTensor,
    pub post_feedforward_layernorm: GpuTensor,
    pub q_proj: GpuTensor,
    pub k_proj: GpuTensor,
    pub v_proj: GpuTensor,
    pub o_proj: GpuTensor,
    pub q_norm: GpuTensor,
    pub k_norm: GpuTensor,
    // no v_norm.weight (no_scale — divide only, no learned gain)
    pub gate_proj: GpuTensor,
    pub up_proj: GpuTensor,
    pub down_proj: GpuTensor,
}

pub struct Gemma4VisionWeights {
    /// Patch embedder: Linear(3 * patch_size² → hidden). [hidden, 3*patch²]
    pub patch_embedder_input_proj: GpuTensor,
    /// Factored 2D position embedding table. [2, position_embedding_size, hidden]
    pub position_embedding_table: GpuTensor,
    /// Standardize buffers (applied at vision tower exit, before multimodal embedder).
    pub std_bias: GpuTensor,   // [hidden]
    pub std_scale: GpuTensor,  // [hidden]
    /// Per-layer encoder weights.
    pub layers: Vec<Gemma4VisionLayerWeights>,
    /// Multimodal embedder: Linear(hidden → out_hidden_size). [out_hidden, hidden]
    /// (RMSNorm before this is weight-less — no tensor to load.)
    pub embedding_projection: GpuTensor,
}

impl Gemma4VisionWeights {
    pub fn free_gpu(self, gpu: &mut Gpu) {
        let _ = gpu.free_tensor(self.patch_embedder_input_proj);
        let _ = gpu.free_tensor(self.position_embedding_table);
        let _ = gpu.free_tensor(self.std_bias);
        let _ = gpu.free_tensor(self.std_scale);
        for l in self.layers {
            for t in [l.input_layernorm, l.post_attention_layernorm,
                      l.pre_feedforward_layernorm, l.post_feedforward_layernorm,
                      l.q_proj, l.k_proj, l.v_proj, l.o_proj,
                      l.q_norm, l.k_norm,
                      l.gate_proj, l.up_proj, l.down_proj] {
                let _ = gpu.free_tensor(t);
            }
        }
        let _ = gpu.free_tensor(self.embedding_projection);
    }
}

/// Stub: Phase 7.
pub fn load_vision_weights(
    _hfq: &HfqFile,
    _config: &Gemma4VisionConfig,
    _gpu: &mut Gpu,
) -> HipResult<Gemma4VisionWeights> {
    Err(hip_bridge::HipError::new(
        0,
        "gemma4_vision::load_vision_weights not implemented (Phase 7)",
    ))
}

/// Encode a single image to `default_output_length` (280) soft tokens of
/// dimension `out_hidden_size` (5376 on 31B). Phase 7.
pub fn encode_image(
    _gpu: &mut Gpu,
    _weights: &Gemma4VisionWeights,
    _config: &Gemma4VisionConfig,
    _image_rgb_planar: &[f32],   // [3, H, W] in [0,1]
    _height: usize,
    _width: usize,
) -> HipResult<GpuTensor> {
    Err(hip_bridge::HipError::new(
        0,
        "gemma4_vision::encode_image not implemented (Phase 7)",
    ))
}
