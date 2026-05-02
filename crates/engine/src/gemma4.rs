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

    // Per-layer embedding (Gemma 3n / Gemma 4 E-series). 0 when absent
    // (31B and 26B-A4B have no per-layer embedding; E2B/E4B have 256).
    // Adds a side-channel signal injected after each layer's FFN+residual,
    // before the learned `layer_scalar` multiply. See `forward_scratch` for
    // the pre-loop projection and `apply_per_layer_inject` for the per-layer
    // application. References:
    //   • llama.cpp/src/models/gemma4-iswa.cpp lines 33-38, 202-224, 263-313
    //   • HF tensor names: `embed_tokens_per_layer`, `per_layer_model_projection`,
    //     `per_layer_projection_norm`, `per_layer_input_gate`,
    //     `per_layer_projection`, `post_per_layer_input_norm`
    pub n_embd_per_layer: usize,

    /// KV-sharing layer count for Gemma 4 E-series (per HF
    /// `num_kv_shared_layers`). When non-zero, the LAST `num_kv_shared_layers`
    /// layers do NOT compute their own K/V projection — they reuse an earlier
    /// layer's KV cache. See llama-model.cpp `n_layer_kv_from_start`:
    ///   • Layers `[0, n_layers - num_kv_shared_layers)` compute own KV.
    ///   • Layers `[n_layers - num_kv_shared_layers, n_layers)` are KV-shared.
    /// Sliding shared layers read from layer-index `(n_layers - shared) - 2`;
    /// full shared layers read from `(n_layers - shared) - 1`. Per-type cache
    /// slot mapping is built in `kv_share_plan()`.
    /// Values for the official lineup: E2B=20, E4B=18, 26B-A4B=0, 31B=0.
    pub num_kv_shared_layers: usize,

    /// `use_double_wide_mlp` flag (E2B only). When true, the LAST
    /// `num_kv_shared_layers` layers use `intermediate_size * 2` for the
    /// FFN width (gate / up / down projections double in their
    /// hidden-dim direction) — see llama-model.cpp lines 4730-4735.
    pub use_double_wide_mlp: bool,

    /// MoE block enable (Gemma 4 26B-A4B only). When true, every layer
    /// has the standard SwiGLU FFN PLUS a parallel routed-experts branch:
    /// router projects to N experts, top-K activated, weighted sum.
    /// Requires distinct sandwich norms (pre_feedforward_layernorm_2,
    /// post_feedforward_layernorm_1, post_feedforward_layernorm_2) and
    /// 3D-stacked expert tensors (experts.gate_up_proj / experts.down_proj).
    pub enable_moe_block: bool,
    /// Total number of routed experts (128 on 26B-A4B).
    pub num_experts: usize,
    /// Top-K experts activated per token (8 on 26B-A4B).
    pub top_k_experts: usize,
    /// Per-expert FFN intermediate size (704 on 26B-A4B). Distinct from
    /// the shared-MLP `hidden_dim` (= intermediate_size = 2112).
    pub moe_intermediate_size: usize,

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

/// Per-layer KV-cache routing for Gemma 4 E-series KV-sharing.
///
/// Built once at model load time from `Gemma4Config`. Indexed by layer
/// ordinal; tells the forward pass two things per layer:
///
/// 1. `owns_kv[il]` — whether this layer should run its own K/V projection
///    + RoPE-on-K + KV-cache write. False on shared layers.
/// 2. `kv_slot[il]` — which slot in the per-type KV cache to read (and write
///    if `owns_kv`). For own layers this is the count of same-type own-kv
///    layers seen so far; for shared layers it's the anchor slot derived
///    from llama-model.cpp's `(start - 2)` / `(start - 1)` rule.
///
/// On 31B / 26B-A4B (no shared layers) this collapses to the trivial
/// "every layer owns its KV, slot = sequential count per type" plan.
#[derive(Debug, Clone)]
pub struct KvSharePlan {
    /// `n_layers - num_kv_shared_layers` — first index where shared layers start.
    pub n_layer_kv_from_start: usize,
    /// Number of sliding layers in `[0, n_layer_kv_from_start)` (= kv_sliding cache size).
    pub n_sliding_own: usize,
    /// Number of full layers in `[0, n_layer_kv_from_start)` (= kv_full cache size).
    pub n_full_own: usize,
    /// True iff layer `il` writes its own K/V into the cache.
    pub owns_kv: Vec<bool>,
    /// Per-type cache slot to use for layer `il` (read + write if owns_kv).
    pub kv_slot: Vec<usize>,
}

impl Gemma4Config {
    /// Per-layer FFN hidden_dim. On most Gemma 4 models this is constant
    /// (config.hidden_dim across all layers). E2B sets `use_double_wide_mlp`
    /// which doubles the FFN width on the LAST `num_kv_shared_layers` layers
    /// (intermediate_size*2). Per llama-model.cpp lines 4730-4735.
    pub fn ffn_hidden_dim_for_layer(&self, layer_idx: usize) -> usize {
        if self.use_double_wide_mlp {
            let n_kv_start = self.n_layers.saturating_sub(self.num_kv_shared_layers);
            if layer_idx >= n_kv_start {
                return self.hidden_dim * 2;
            }
        }
        self.hidden_dim
    }

    /// Maximum FFN hidden_dim across all layers — used to size scratch
    /// buffers (`gate_ffn`, `up_ffn`, `ffn_hidden`) once per model load.
    pub fn max_ffn_hidden_dim(&self) -> usize {
        if self.use_double_wide_mlp { self.hidden_dim * 2 } else { self.hidden_dim }
    }

    /// Compute the KV-cache routing plan for this config. Cheap; call once
    /// at load time and stash on the model object.
    ///
    /// Architectural assumption (validated for Gemma 4 E2B / E4B):
    /// the layer at index `n_layer_kv_from_start - 2` MUST be a sliding
    /// layer (it's the anchor for all shared sliding layers), and the layer
    /// at index `n_layer_kv_from_start - 1` MUST be a full layer (anchor
    /// for shared full). If those slot types don't match, the per-type
    /// cache layouts wouldn't align (different head_dim / n_kv_heads), so
    /// we panic loudly in that case with a diagnostic.
    pub fn kv_share_plan(&self) -> KvSharePlan {
        let n_kv_start = self.n_layers.saturating_sub(self.num_kv_shared_layers);
        let mut owns_kv = Vec::with_capacity(self.n_layers);
        let mut kv_slot = Vec::with_capacity(self.n_layers);
        let mut n_sliding_own = 0usize;
        let mut n_full_own = 0usize;
        let mut s = 0usize;
        let mut f = 0usize;
        let mut anchor_sliding: Option<usize> = None;
        let mut anchor_full: Option<usize> = None;
        for (il, &lt) in self.layer_types.iter().enumerate() {
            let owns = il < n_kv_start;
            owns_kv.push(owns);
            if owns {
                let slot = match lt { LayerType::Sliding => s, LayerType::Full => f };
                kv_slot.push(slot);
                match lt {
                    LayerType::Sliding => {
                        if self.num_kv_shared_layers > 0 && il + 2 == n_kv_start {
                            anchor_sliding = Some(s);
                        }
                        s += 1;
                        n_sliding_own += 1;
                    }
                    LayerType::Full => {
                        if self.num_kv_shared_layers > 0 && il + 1 == n_kv_start {
                            anchor_full = Some(f);
                        }
                        f += 1;
                        n_full_own += 1;
                    }
                }
            } else {
                kv_slot.push(usize::MAX);  // anchor filled below
            }
        }
        if self.num_kv_shared_layers > 0 {
            assert!(n_kv_start >= 2,
                "Gemma 4 KV-sharing assumes n_layer_kv_from_start >= 2 (got {})", n_kv_start);
            let sliding_anchor_layer = n_kv_start - 2;
            let full_anchor_layer = n_kv_start - 1;
            assert_eq!(
                self.layer_types[sliding_anchor_layer], LayerType::Sliding,
                "Gemma 4 KV-sharing expects layer {sliding_anchor_layer} to be Sliding",
            );
            assert_eq!(
                self.layer_types[full_anchor_layer], LayerType::Full,
                "Gemma 4 KV-sharing expects layer {full_anchor_layer} to be Full",
            );
            let asl = anchor_sliding.expect("sliding anchor slot");
            let afl = anchor_full.expect("full anchor slot");
            for (il, &lt) in self.layer_types.iter().enumerate() {
                if il >= n_kv_start {
                    kv_slot[il] = match lt { LayerType::Sliding => asl, LayerType::Full => afl };
                }
            }
        }
        KvSharePlan {
            n_layer_kv_from_start: n_kv_start,
            n_sliding_own,
            n_full_own,
            owns_kv,
            kv_slot,
        }
    }
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

    // Per-layer embedding feature (E2B / E4B / Gemma 3n). 0 ≙ disabled.
    let n_embd_per_layer = tc.get("hidden_size_per_layer_input")
        .and_then(|v| v.as_u64()).unwrap_or(0) as usize;
    // KV-sharing (E-series). 0 on 31B / 26B-A4B; 18-20 on E4B / E2B.
    let num_kv_shared_layers = tc.get("num_kv_shared_layers")
        .and_then(|v| v.as_u64()).unwrap_or(0) as usize;
    // Double-wide-MLP (E2B only). When true, shared layers have
    // intermediate_size * 2.
    let use_double_wide_mlp = tc.get("use_double_wide_mlp")
        .and_then(|v| v.as_bool()).unwrap_or(false);

    // MoE block (26B-A4B only).
    let enable_moe_block = tc.get("enable_moe_block")
        .and_then(|v| v.as_bool()).unwrap_or(false);
    let num_experts = tc.get("num_experts")
        .and_then(|v| v.as_u64()).unwrap_or(0) as usize;
    let top_k_experts = tc.get("top_k_experts")
        .and_then(|v| v.as_u64()).unwrap_or(0) as usize;
    let moe_intermediate_size = tc.get("moe_intermediate_size")
        .and_then(|v| v.as_u64()).unwrap_or(0) as usize;

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
        n_embd_per_layer,
        num_kv_shared_layers,
        use_double_wide_mlp,
        enable_moe_block,
        num_experts,
        top_k_experts,
        moe_intermediate_size,
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

    // MLP (SwiGLU). `ffn_hidden_dim` is per-layer to support
    // `use_double_wide_mlp` on E2B (last 20 layers have intermediate_size*2).
    pub ffn_hidden_dim: usize,
    pub gate_proj: WeightTensor, // [ffn_hidden_dim, dim]
    pub up_proj: WeightTensor,   // [ffn_hidden_dim, dim]
    pub down_proj: WeightTensor, // [dim, ffn_hidden_dim]

    // Per-layer embedding side-channel (Some on E2B / E4B; None on 31B / 26B-A4B).
    // Applied at layer end after FFN+residual but BEFORE the layer_scalar.
    pub per_layer_input_gate: Option<WeightTensor>,    // [n_embd_per_layer, dim]
    pub per_layer_projection: Option<WeightTensor>,    // [dim, n_embd_per_layer]
    pub post_per_layer_input_norm: Option<GpuTensor>,  // [dim]

    // MoE branch — Some on 26B-A4B (where every layer is MoE), None elsewhere.
    pub moe: Option<MoeLayerExtras>,
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

    // Attention (full — head_dim=512)
    pub q_proj: WeightTensor,   // [n_heads * 512, dim]
    pub k_proj: WeightTensor,   // [n_kv * 512, dim]
    /// Separate V projection — present when `attention_k_eq_v == false`
    /// (Gemma 4 E2B / E4B). Absent (None) when V is captured from K's
    /// pre-norm output (Gemma 4 31B / 26B-A4B with `attention_k_eq_v: true`).
    pub v_proj: Option<WeightTensor>, // [n_kv * 512, dim] when Some
    pub o_proj: WeightTensor,   // [dim, n_heads * 512]
    pub q_norm: GpuTensor,      // [512]
    pub k_norm: GpuTensor,      // [512]
    // no v_norm weight — v_norm is no-scale (divide only)

    // MLP (SwiGLU). `ffn_hidden_dim` is per-layer (same use_double_wide_mlp
    // behavior as sliding layers).
    pub ffn_hidden_dim: usize,
    pub gate_proj: WeightTensor,
    pub up_proj: WeightTensor,
    pub down_proj: WeightTensor,

    // Per-layer embedding side-channel — same as on sliding layers when the
    // model uses this feature. See `SlidingLayerWeights` for shapes.
    pub per_layer_input_gate: Option<WeightTensor>,
    pub per_layer_projection: Option<WeightTensor>,
    pub post_per_layer_input_norm: Option<GpuTensor>,

    // MoE branch — Some on 26B-A4B, None elsewhere.
    pub moe: Option<MoeLayerExtras>,
}

/// Per-routed-expert weights for a Gemma 4 MoE layer. Each layer has
/// `n_experts` of these (128 on 26B-A4B), of which top_k_experts (8) are
/// activated per token.
pub struct MoeExpertWeights {
    /// gate + up projection fused: `[2 * moe_intermediate, dim]`. The
    /// first `moe_intermediate` rows are gate; the last are up.
    pub gate_up_proj: WeightTensor,
    /// `[dim, moe_intermediate]`. Projects per-expert FFN hidden back to dim.
    pub down_proj: WeightTensor,
}

/// MoE branch weights for a Gemma 4 MoE layer (26B-A4B). Present on
/// every layer when `config.enable_moe_block` is set.
pub struct MoeLayerExtras {
    /// `[n_experts, dim]` — projects router input to expert logits.
    pub router_proj: WeightTensor,
    /// `[dim]` — multiplicative scale on router input (`router.scale` in HF).
    pub router_scale: GpuTensor,
    /// `[n_experts]` — per-expert post-`down_proj` scale (`router.per_expert_scale`).
    pub per_expert_scale: GpuTensor,
    /// Host mirror of `per_expert_scale` for fast top-K weight composition.
    pub per_expert_scale_host: Vec<f32>,
    /// `[dim]` — RMSNorm applied to attn_out before the MoE branch.
    pub pre_feedforward_layernorm_2: GpuTensor,
    /// `[dim]` — RMSNorm applied to cur_mlp (the standard SwiGLU output) BEFORE summing.
    pub post_feedforward_layernorm_1: GpuTensor,
    /// `[dim]` — RMSNorm applied to cur_moe (the MoE branch output) BEFORE summing.
    pub post_feedforward_layernorm_2: GpuTensor,
    /// Pooled allocation for all `n_experts` gate_up_proj tensors of this layer
    /// (concatenated). Freed once per layer; `experts[i].gate_up_proj.buf`
    /// aliases into this buffer at offset `i * gate_up_bytes_per_expert`.
    /// Replaces the previous per-expert hipMalloc that caused 128×30 = 3840
    /// small allocations and fragmented the HIP heap to OOM at layer 25.
    pub experts_gate_up_pool: GpuTensor,
    /// Pooled allocation for all `n_experts` down_proj tensors of this layer.
    /// Same aliasing scheme as the gate_up pool.
    pub experts_down_pool: GpuTensor,
    /// Per-expert WeightTensor views (aliases into the pools above).
    /// `free_gpu` SKIPS freeing these — the pools own the bytes.
    pub experts: Vec<MoeExpertWeights>,
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
    // ─── Per-layer embedding (Gemma 4 E-series) — Some when n_embd_per_layer > 0 ───
    // Per-token embedding into the per-layer side channel.
    /// `embed_tokens_per_layer.weight` [vocab_size, n_embd_per_layer * n_layers].
    /// Looked up per token, then split into n_layers slots of n_embd_per_layer.
    /// Forced to Q8F16 in the loader so row-lookup uses the existing
    /// embedding_lookup_q8 dispatch with `dim = n_embd_per_layer * n_layers`.
    pub per_layer_tok_embd: Option<GpuTensor>,
    pub per_layer_tok_embd_format: EmbeddingFormat,
    /// `per_layer_model_projection.weight` [n_embd_per_layer * n_layers, dim].
    /// Mixes the regular post-embed hidden state into the per-layer slots.
    pub per_layer_model_proj: Option<WeightTensor>,
    /// `per_layer_projection_norm.weight` [n_embd_per_layer].
    /// Applied per-slot via rmsnorm_batched(batch=n_layers, n=n_embd_per_layer).
    pub per_layer_proj_norm: Option<GpuTensor>,
    /// Per-layer weights indexed by layer ordinal.
    pub layers: Vec<LayerWeights>,
}

impl Gemma4Weights {
    pub fn free_gpu(self, gpu: &mut Gpu) {
        let _ = gpu.free_tensor(self.embed_tokens);
        let _ = gpu.free_tensor(self.final_norm);
        // lm_head may alias embed_tokens — skip if so (we rely on the loader
        // to set `lm_head.buf` to an alias and not a separate allocation).
        // Per-layer embedding model-level tensors (Some on E-series only).
        if let Some(t) = self.per_layer_tok_embd { let _ = gpu.free_tensor(t); }
        if let Some(wt) = self.per_layer_model_proj { let _ = gpu.free_tensor(wt.buf); }
        if let Some(t) = self.per_layer_proj_norm { let _ = gpu.free_tensor(t); }
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
                    if let Some(wt) = s.per_layer_input_gate { let _ = gpu.free_tensor(wt.buf); }
                    if let Some(wt) = s.per_layer_projection { let _ = gpu.free_tensor(wt.buf); }
                    if let Some(t) = s.post_per_layer_input_norm { let _ = gpu.free_tensor(t); }
                    if let Some(moe) = s.moe { Self::free_moe(gpu, moe); }
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
                    if let Some(wt) = f.v_proj { let _ = gpu.free_tensor(wt.buf); }
                    if let Some(wt) = f.per_layer_input_gate { let _ = gpu.free_tensor(wt.buf); }
                    if let Some(wt) = f.per_layer_projection { let _ = gpu.free_tensor(wt.buf); }
                    if let Some(t) = f.post_per_layer_input_norm { let _ = gpu.free_tensor(t); }
                    if let Some(moe) = f.moe { Self::free_moe(gpu, moe); }
                }
            }
        }
    }

    fn free_moe(gpu: &mut Gpu, moe: MoeLayerExtras) {
        let _ = gpu.free_tensor(moe.router_proj.buf);
        let _ = gpu.free_tensor(moe.router_scale);
        let _ = gpu.free_tensor(moe.per_expert_scale);
        let _ = gpu.free_tensor(moe.pre_feedforward_layernorm_2);
        let _ = gpu.free_tensor(moe.post_feedforward_layernorm_1);
        let _ = gpu.free_tensor(moe.post_feedforward_layernorm_2);
        // Per-expert WeightTensors are aliases into the two pool buffers;
        // freeing them would double-free. Drop the Vec without freeing
        // each, then free the pools once.
        std::mem::drop(moe.experts);
        let _ = gpu.free_tensor(moe.experts_gate_up_pool);
        let _ = gpu.free_tensor(moe.experts_down_pool);
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
///
/// HIPFIRE_GEMMA4_NORM_PLUS_ONE=1 forces the Gemma 3 +1 shift at load time
/// (diagnostic only — for testing whether 26B-A4B's MoE-norm weights need
/// the alternate convention even though convert_hf_to_gguf.py says they
/// don't).
fn load_gemma4_norm(hfq: &HfqFile, gpu: &mut Gpu, name: &str, dim: usize)
    -> HipResult<GpuTensor>
{
    let mut f32_data = load_f32_vec(hfq, name, dim)?;
    if std::env::var("HIPFIRE_GEMMA4_NORM_PLUS_ONE").ok().as_deref() == Some("1") {
        for v in f32_data.iter_mut() { *v += 1.0; }
    }
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
        // MG4G256 (Magnum-Gemma 4-bit, qt=19) reads back as MQ4G256: identical
        // 136 B/group binary layout, same FWHT signs, same `q * scale + min`
        // decode. Only the per-block scale calibration (percentile-clip vs
        // true min..max) differs at quantize time — engine GEMV is unchanged.
        19 => DType::MQ4G256,
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

    // Per-layer embedding model-level tensors (only present when
    // n_embd_per_layer > 0). E2B / E4B have these; 31B / 26B-A4B don't.
    // The per-token embedding table `embed_tokens_per_layer` is forced to
    // Q8F16 by the quantizer (same is_embed treatment as `embed_tokens`),
    // so engine-side row lookup uses the existing embedding_lookup_q8 path
    // with `dim = n_embd_per_layer * n_layers`.
    let (per_layer_tok_embd, per_layer_tok_embd_format,
         per_layer_model_proj, per_layer_proj_norm) = if config.n_embd_per_layer > 0 {
        eprintln!("gemma4: loading per-layer embedding tensors (n_embd_per_layer={})...",
                  config.n_embd_per_layer);
        let pl_dim = config.n_embd_per_layer * config.n_layers;
        // embed_tokens_per_layer: row-lookup table.
        let pl_embd_name = "model.language_model.embed_tokens_per_layer.weight";
        let (pl_embd_info, pl_embd_data) = hfq.tensor_data(pl_embd_name).ok_or_else(|| {
            hip_bridge::HipError::new(0, "embed_tokens_per_layer not found in HFQ")
        })?;
        let (pl_embd, pl_fmt) = match pl_embd_info.quant_type {
            3 => (gpu.upload_raw(pl_embd_data, &[pl_embd_data.len()])?, EmbeddingFormat::Q8_0),
            6 => (gpu.upload_raw(pl_embd_data, &[pl_embd_data.len()])?, EmbeddingFormat::HFQ4G256),
            7 => (gpu.upload_raw(pl_embd_data, &[pl_embd_data.len()])?, EmbeddingFormat::HFQ4G128),
            1 => {
                let f32_data: Vec<f32> = pl_embd_data.chunks_exact(2)
                    .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect();
                (gpu.upload_f32(&f32_data, &[config.vocab_size, pl_dim])?, EmbeddingFormat::F32)
            }
            qt => return Err(hip_bridge::HipError::new(
                0, &format!("unsupported per_layer_tok_embd quant_type {qt}"),
            )),
        };
        // per_layer_model_projection: standard 2D weight, projects [dim] → [pl_dim].
        let model_proj = load_gemma4_weight(hfq, gpu,
            "model.language_model.per_layer_model_projection.weight", pl_dim, config.dim)?;
        // per_layer_projection_norm: shape [n_embd_per_layer], applied per-slot.
        let proj_norm = load_gemma4_norm(hfq, gpu,
            "model.language_model.per_layer_projection_norm.weight", config.n_embd_per_layer)?;
        (Some(pl_embd), pl_fmt, Some(model_proj), Some(proj_norm))
    } else {
        (None, EmbeddingFormat::F32, None, None)
    };

    eprintln!("gemma4: loading {} layers...", config.n_layers);
    let mut layers = Vec::with_capacity(config.n_layers);
    for i in 0..config.n_layers {
        if i % 5 == 0 {
            eprintln!("  layer {i}/{}", config.n_layers);
        }
        let p = format!("model.language_model.layers.{i}");
        match config.layer_types[i] {
            LayerType::Sliding => {
                let hd = config.sliding_head_dim;
                let kv_dim = config.sliding_n_kv_heads * hd;
                let q_dim = config.n_heads * hd;
                let (layer_scalar, layer_scalar_host) =
                    load_layer_scalar(hfq, gpu, &format!("{p}.layer_scalar"))?;
                let (pl_gate, pl_proj, pl_post_norm) = if config.n_embd_per_layer > 0 {
                    (Some(load_gemma4_weight(hfq, gpu,
                        &format!("{p}.per_layer_input_gate.weight"),
                        config.n_embd_per_layer, config.dim)?),
                     Some(load_gemma4_weight(hfq, gpu,
                        &format!("{p}.per_layer_projection.weight"),
                        config.dim, config.n_embd_per_layer)?),
                     Some(load_gemma4_norm(hfq, gpu,
                        &format!("{p}.post_per_layer_input_norm.weight"), config.dim)?))
                } else { (None, None, None) };
                let ffn_hd = config.ffn_hidden_dim_for_layer(i);
                let moe = if config.enable_moe_block {
                    Some(load_moe_layer_extras(hfq, gpu, &p, config)?)
                } else { None };
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
                    ffn_hidden_dim: ffn_hd,
                    gate_proj: load_gemma4_weight(hfq, gpu,
                        &format!("{p}.mlp.gate_proj.weight"), ffn_hd, config.dim)?,
                    up_proj: load_gemma4_weight(hfq, gpu,
                        &format!("{p}.mlp.up_proj.weight"), ffn_hd, config.dim)?,
                    down_proj: load_gemma4_weight(hfq, gpu,
                        &format!("{p}.mlp.down_proj.weight"), config.dim, ffn_hd)?,
                    per_layer_input_gate: pl_gate,
                    per_layer_projection: pl_proj,
                    post_per_layer_input_norm: pl_post_norm,
                    moe,
                }));
            }
            LayerType::Full => {
                let hd = config.full_head_dim;
                let kv_dim = config.full_n_kv_heads * hd;
                let q_dim = config.n_heads * hd;
                let (layer_scalar, layer_scalar_host) =
                    load_layer_scalar(hfq, gpu, &format!("{p}.layer_scalar"))?;
                let (pl_gate, pl_proj, pl_post_norm) = if config.n_embd_per_layer > 0 {
                    (Some(load_gemma4_weight(hfq, gpu,
                        &format!("{p}.per_layer_input_gate.weight"),
                        config.n_embd_per_layer, config.dim)?),
                     Some(load_gemma4_weight(hfq, gpu,
                        &format!("{p}.per_layer_projection.weight"),
                        config.dim, config.n_embd_per_layer)?),
                     Some(load_gemma4_norm(hfq, gpu,
                        &format!("{p}.post_per_layer_input_norm.weight"), config.dim)?))
                } else { (None, None, None) };
                let ffn_hd = config.ffn_hidden_dim_for_layer(i);
                let moe = if config.enable_moe_block {
                    Some(load_moe_layer_extras(hfq, gpu, &p, config)?)
                } else { None };
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
                    v_proj: if config.attention_k_eq_v {
                        // 31B / 26B-A4B: V is captured from K's pre-norm output;
                        // no separate v_proj tensor.
                        None
                    } else {
                        // E2B / E4B: separate v_proj on full layers.
                        Some(load_gemma4_weight(hfq, gpu,
                            &format!("{p}.self_attn.v_proj.weight"), kv_dim, config.dim)?)
                    },
                    o_proj: load_gemma4_weight(hfq, gpu,
                        &format!("{p}.self_attn.o_proj.weight"), config.dim, q_dim)?,
                    q_norm: load_gemma4_head_norm(hfq, gpu,
                        &format!("{p}.self_attn.q_norm.weight"), hd)?,
                    k_norm: load_gemma4_head_norm(hfq, gpu,
                        &format!("{p}.self_attn.k_norm.weight"), hd)?,
                    // no v_norm weight — v_norm is no-scale (ones buffer passed at decode time).
                    ffn_hidden_dim: ffn_hd,
                    gate_proj: load_gemma4_weight(hfq, gpu,
                        &format!("{p}.mlp.gate_proj.weight"), ffn_hd, config.dim)?,
                    up_proj: load_gemma4_weight(hfq, gpu,
                        &format!("{p}.mlp.up_proj.weight"), ffn_hd, config.dim)?,
                    down_proj: load_gemma4_weight(hfq, gpu,
                        &format!("{p}.mlp.down_proj.weight"), config.dim, ffn_hd)?,
                    per_layer_input_gate: pl_gate,
                    per_layer_projection: pl_proj,
                    post_per_layer_input_norm: pl_post_norm,
                    moe,
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
        per_layer_tok_embd,
        per_layer_tok_embd_format,
        per_layer_model_proj,
        per_layer_proj_norm,
        layers,
    })
}

/// Load the MoE branch tensors for one layer of Gemma 4 26B-A4B.
/// Caller has already verified `config.enable_moe_block`. Tensor names per
/// 26B-A4B safetensors (`<layer_path>.experts.{X}.{base}.weight` after our
/// quantizer's 3D-stacked split):
///
///     <p>.router.proj.weight                  [n_experts, dim]
///     <p>.router.scale.weight                 [dim]
///     <p>.router.per_expert_scale.weight      [n_experts]
///     <p>.pre_feedforward_layernorm_2.weight  [dim]
///     <p>.post_feedforward_layernorm_1.weight [dim]
///     <p>.post_feedforward_layernorm_2.weight [dim]
///     <p>.experts.{X}.gate_up_proj.weight     [2 * moe_intermediate, dim]
///     <p>.experts.{X}.down_proj.weight        [dim, moe_intermediate]
///
/// Per-expert scale is mirrored to host (n_experts floats) so we can fold
/// it into the top-K weight composition without a D2H per token.
fn load_moe_layer_extras(hfq: &HfqFile, gpu: &mut Gpu, p: &str, config: &Gemma4Config)
    -> HipResult<MoeLayerExtras>
{
    let n_exp = config.num_experts;
    let dim = config.dim;
    let mi = config.moe_intermediate_size;

    let router_proj = load_gemma4_weight(hfq, gpu,
        &format!("{p}.router.proj.weight"), n_exp, dim)?;
    // NOTE: `router.scale` and `router.per_expert_scale` ship WITHOUT the
    // `.weight` suffix in HF's 26B-A4B safetensors (so `should_quantize`
    // returns false → stored as F16). Loader uses bare paths.
    let router_scale = load_gemma4_norm(hfq, gpu,
        &format!("{p}.router.scale"), dim)?;
    let per_expert_scale_host = load_f32_vec(hfq,
        &format!("{p}.router.per_expert_scale"), n_exp)?;
    let per_expert_scale = gpu.upload_f32(&per_expert_scale_host, &[n_exp])?;
    let pre_feedforward_layernorm_2 = load_gemma4_norm(hfq, gpu,
        &format!("{p}.pre_feedforward_layernorm_2.weight"), dim)?;
    let post_feedforward_layernorm_1 = load_gemma4_norm(hfq, gpu,
        &format!("{p}.post_feedforward_layernorm_1.weight"), dim)?;
    let post_feedforward_layernorm_2 = load_gemma4_norm(hfq, gpu,
        &format!("{p}.post_feedforward_layernorm_2.weight"), dim)?;

    // Pool the per-expert weights into ONE GPU allocation per kind. With
    // 128 experts × 2 kinds × 30 layers, the previous "1 hipMalloc per
    // expert tensor" path created 7680 allocations and fragmented HIP's
    // heap to OOM at layer 25/30 even though total memory fit. Pooling
    // collapses that to 60 allocations (2 per layer).
    //
    // Approach: read all 128 experts of each kind, validate they share
    // the same byte layout (same quant_type / size — guaranteed because
    // the quantizer wrote them with identical (m, k)), concat into a
    // single CPU buffer, upload once, then build per-expert WeightTensor
    // views via `sub_offset` into the pool. The pool owns the bytes;
    // per-expert views are non-owning aliases (free_gpu skips them).
    let load_pool = |gpu: &mut Gpu, base: &str, m: usize, k: usize|
        -> HipResult<(GpuTensor, DType, usize)>
    {
        // First pass: read first expert to learn quant_type + bytes-per-expert.
        let first_name = format!("{p}.experts.0.{base}.weight");
        let (first_info, first_data) = hfq.tensor_data(&first_name).ok_or_else(|| {
            hip_bridge::HipError::new(0, &format!("MoE expert tensor not found: {first_name}"))
        })?;
        let bytes_per_expert = first_data.len();
        let dtype = match first_info.quant_type {
            1 => DType::F32, // dequant from F16 → F32 expansion (rare for experts)
            3 => DType::Q8_0, 4 => DType::Q4K,
            6 => DType::HFQ4G256, 7 => DType::HFQ4G128, 8 => DType::HFQ6G256,
            9 => DType::HFQ2G256, 10 => DType::HFQ2G128,
            11 => DType::HFQ3G256, 12 => DType::HFQ3G128,
            // MG4G256 (qt=19) and MQ4G256 (qt=13) both use the MQ4 dispatch.
            13 | 19 => DType::MQ4G256,
            14 => DType::MQ8G256, 15 => DType::MQ6G256,
            17 => DType::MQ3G256, 18 => DType::MQ2G256,
            qt => return Err(hip_bridge::HipError::new(
                0, &format!("unsupported MoE expert quant_type {qt} for {first_name}"),
            )),
        };
        // Concat 128 experts' bytes into one CPU buffer, then upload once.
        let mut concat = Vec::with_capacity(bytes_per_expert * n_exp);
        concat.extend_from_slice(first_data);
        for x in 1..n_exp {
            let name = format!("{p}.experts.{x}.{base}.weight");
            let (info, data) = hfq.tensor_data(&name).ok_or_else(|| {
                hip_bridge::HipError::new(0, &format!("MoE expert tensor not found: {name}"))
            })?;
            if data.len() != bytes_per_expert {
                return Err(hip_bridge::HipError::new(
                    0, &format!("MoE expert {name} byte size mismatch ({} vs {bytes_per_expert})", data.len()),
                ));
            }
            if info.quant_type != first_info.quant_type {
                return Err(hip_bridge::HipError::new(
                    0, &format!("MoE expert {name} quant_type mismatch ({} vs {})", info.quant_type, first_info.quant_type),
                ));
            }
            concat.extend_from_slice(data);
        }
        // For F16-source path we'd need dequant before upload — Gemma 4 MoE
        // never ships experts at F16 (always quantized) so panic if hit.
        if first_info.quant_type == 1 {
            return Err(hip_bridge::HipError::new(
                0, "MoE expert pool path doesn't dequant F16 — quantizer should have produced MG4/MQ4/HFQ",
            ));
        }
        let pool = gpu.upload_raw(&concat, &[concat.len()])?;
        let _ = (m, k);
        Ok((pool, dtype, bytes_per_expert))
    };

    let (gate_up_pool, gate_up_dtype, gate_up_bytes) = load_pool(gpu, "gate_up_proj", 2 * mi, dim)?;
    let (down_pool, down_dtype, down_bytes) = load_pool(gpu, "down_proj", dim, mi)?;

    // Build per-expert WeightTensor views. Each view aliases into the pool
    // at offset `expert_idx * bytes_per_expert`. Per the GpuTensor
    // contract, sub_offset returns a non-owning view — free_gpu must
    // NOT free these (the pool owns the bytes).
    let mut experts = Vec::with_capacity(n_exp);
    for x in 0..n_exp {
        let gu_view = gate_up_pool.sub_offset(x * gate_up_bytes, gate_up_bytes);
        let dn_view = down_pool.sub_offset(x * down_bytes, down_bytes);
        experts.push(MoeExpertWeights {
            gate_up_proj: WeightTensor {
                buf: gu_view, gpu_dtype: gate_up_dtype, m: 2 * mi, k: dim, row_stride: 0,
            },
            down_proj: WeightTensor {
                buf: dn_view, gpu_dtype: down_dtype, m: dim, k: mi, row_stride: 0,
            },
        });
    }

    Ok(MoeLayerExtras {
        router_proj,
        router_scale,
        per_expert_scale,
        per_expert_scale_host,
        pre_feedforward_layernorm_2,
        post_feedforward_layernorm_1,
        post_feedforward_layernorm_2,
        experts_gate_up_pool: gate_up_pool,
        experts_down_pool: down_pool,
        experts,
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

    // (Removed: sliding_cos/sin and full_cos/sin pre-computed RoPE tables.
    // gpu.rope_f32 / rope_partial_halved_f32 compute cos/sin in-kernel from
    // pos_buf + theta, so the tables were never read. Allocating them at
    // 4× max_seq × head_dim was wasted VRAM — at the freshly-fixed
    // max_seq=131072 default for Gemma 4 this was ~768 MB just for dead
    // tables. Removed entirely; restoring requires actually wiring a
    // table-using rope kernel.)

    // No-scale v_norm ones buffer (full-attn layers compute v_norm without
    // a learned weight — we pass this ones-filled tensor to the existing
    // rmsnorm kernel to get no-scale RMS semantics).
    pub v_norm_ones_full: GpuTensor, // [full_head_dim]

    // ─── Per-layer embedding scratch (E-series only — zero-sized when n_embd_per_layer == 0) ───
    /// Stages the per-layer side-channel signal: looked up from
    /// `per_layer_tok_embd` once per token, then `per_layer_model_proj @ x`
    /// is added in. Sliced as `[il*n_embd_per_layer..(il+1)*n_embd_per_layer]`
    /// inside each layer. Shape: [n_embd_per_layer * n_layers] flat.
    pub per_layer_inp: GpuTensor,
    /// Scratch for the projection branch (`per_layer_model_proj @ x`) before
    /// it gets added into `per_layer_inp`. Same shape as `per_layer_inp`.
    pub per_layer_inp_proj: GpuTensor,
    /// Per-layer-decode scratch: holds `per_layer_input_gate @ x` then
    /// `gelu(...)` then `... * per_layer_inp_slice`. [n_embd_per_layer].
    pub per_layer_tmp: GpuTensor,
    /// Per-layer-decode scratch: holds `per_layer_projection @ tmp` (before
    /// rmsnorm + residual add). [dim].
    pub per_layer_out: GpuTensor,
    /// Pre-saved residual at the per-layer inject site (after FFN+residual,
    /// before the side-channel additions). [dim].
    pub per_layer_pe_in: GpuTensor,

    // ─── MoE scratch (26B-A4B only — zero-sized when enable_moe_block==false) ───
    /// `[dim]` — `rmsnorm(attn_out, router_scale) * 1/sqrt(dim)` router input.
    pub moe_router_in: GpuTensor,
    /// `[n_experts]` — router logits (post-gemv, pre-softmax).
    pub moe_router_logits: GpuTensor,
    /// `[top_k]` — i32 expert indices written by `moe_softmax_topk_renorm_k8`.
    pub moe_topk_indices: GpuTensor,
    /// `[top_k]` — top-K softmax weights (renormalized).
    pub moe_topk_weights: GpuTensor,
    /// `[dim]` — `pre_feedforward_layernorm_2(attn_out)`, the MoE branch input.
    pub moe_pre2: GpuTensor,
    /// `[dim]` — `post_feedforward_layernorm_1(SwiGLU output)`.
    pub moe_cur_mlp: GpuTensor,
    /// `[dim]` — accumulated `Σ_k weight[k] * expert[k](moe_pre2)`.
    pub moe_cur_moe: GpuTensor,
    /// `[2 * moe_intermediate]` — per-expert gate_up_proj output.
    pub moe_expert_gate_up: GpuTensor,
    /// `[moe_intermediate]` — per-expert `gelu_tanh(gate) * up`.
    pub moe_expert_hidden: GpuTensor,
    /// `[dim]` — per-expert `down_proj(hidden)` output.
    pub moe_expert_out: GpuTensor,
}

impl Gemma4Scratch {
    /// `max_seq` is the per-cache capacity used by `KvCache::new_gpu*` calls
    /// the caller will make alongside this scratch. Must match: the flash
    /// kernels write `partials[h * max_tiles + tile_id]` with `max_tiles =
    /// ceil(max_seq / 128)`, and the RoPE tables are indexed by absolute
    /// position. Sizing the scratch to a smaller value than the kv_cache
    /// passed at dispatch time silently overruns these buffers — caller
    /// must keep them in lockstep.
    pub fn new(gpu: &mut Gpu, config: &Gemma4Config, max_seq: usize) -> HipResult<Self> {
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

        // Size FFN scratch by the MAX layer hidden_dim — on E2B that's
        // intermediate_size*2 (for the use_double_wide_mlp shared layers);
        // every other model has a single hidden_dim across all layers.
        let max_ffn_hd = config.max_ffn_hidden_dim();
        let gate_ffn = gpu.zeros(&[max_ffn_hd], DType::F32)?;
        let up_ffn = gpu.zeros(&[max_ffn_hd], DType::F32)?;
        let ffn_hidden = gpu.zeros(&[max_ffn_hd], DType::F32)?;
        let ffn_out = gpu.zeros(&[dim], DType::F32)?;

        let logits = gpu.zeros(&[config.vocab_size], DType::F32)?;
        let sample_buf = gpu.zeros(&[2], DType::F32)?;
        let repeat_buf = gpu.zeros(&[1024], DType::F32)?;

        // Flash partials sizing — must accommodate the largest max_tiles the
        // dispatch path will compute. Sized for FULL attn (larger head_dim=512,
        // larger max_tiles). Previously hardcoded to MAX_CTX_DEFAULT=32768
        // which silently overran when the caller allocated a kv_cache at
        // larger max_seq (Gemma 4 advertises max_position_embeddings=131072).
        const TILE_SIZE: usize = 128;
        let max_tiles_full = (max_seq + TILE_SIZE - 1) / TILE_SIZE;
        let flash_partials_sz = config.n_heads * max_tiles_full * (2 + config.full_head_dim);
        let flash_partials = gpu.zeros(&[flash_partials_sz], DType::F32)?;

        // (sliding_cos/sin and full_cos/sin tables removed — see struct
        // comment. The rope_f32 and rope_partial_halved_f32 kernels compute
        // cos/sin in-kernel from pos+theta and never indexed into these
        // tables; allocating them at max_seq × head_dim per layer-type was
        // pure VRAM waste that scaled badly when max_seq was lifted from
        // the previous 32k constant to the caller's actual value.)

        // v_norm ones — populated on first use in the forward pass.
        // (Allocated to the full head_dim because only full-attn layers
        // apply no-scale v_norm.)
        let v_norm_ones_full = gpu.zeros(&[config.full_head_dim], DType::F32)?;

        // Per-layer embedding scratch. Allocated as size-1 placeholder when
        // the model doesn't use this feature (31B / 26B-A4B) so the GpuTensor
        // fields are valid; the forward path branches on
        // `config.n_embd_per_layer > 0` and never touches them in that case.
        let pl_dim = config.n_embd_per_layer * config.n_layers;
        let pl_dim_alloc = pl_dim.max(1);
        let pl_w_alloc = config.n_embd_per_layer.max(1);
        let per_layer_inp      = gpu.zeros(&[pl_dim_alloc], DType::F32)?;
        let per_layer_inp_proj = gpu.zeros(&[pl_dim_alloc], DType::F32)?;
        let per_layer_tmp      = gpu.zeros(&[pl_w_alloc],   DType::F32)?;
        let per_layer_out      = gpu.zeros(&[dim],          DType::F32)?;
        let per_layer_pe_in    = gpu.zeros(&[dim],          DType::F32)?;

        // MoE scratch — sized for the active config when enable_moe_block is
        // set; otherwise size-1 placeholders so the GpuTensor fields are
        // valid but unused. The forward pass branches on lw.moe.is_some()
        // and only touches these buffers in the MoE path.
        let moe_active = config.enable_moe_block;
        let n_exp_alloc = if moe_active { config.num_experts.max(1) } else { 1 };
        let topk_alloc = if moe_active { config.top_k_experts.max(1) } else { 1 };
        let mi_alloc = if moe_active { config.moe_intermediate_size.max(1) } else { 1 };
        let dim_alloc_moe = if moe_active { dim } else { 1 };
        let moe_router_in       = gpu.zeros(&[dim_alloc_moe], DType::F32)?;
        let moe_router_logits   = gpu.zeros(&[n_exp_alloc],   DType::F32)?;
        let moe_topk_indices    = gpu.zeros(&[topk_alloc],    DType::F32)?;  // bytes interp as i32
        let moe_topk_weights    = gpu.zeros(&[topk_alloc],    DType::F32)?;
        let moe_pre2            = gpu.zeros(&[dim_alloc_moe], DType::F32)?;
        let moe_cur_mlp         = gpu.zeros(&[dim_alloc_moe], DType::F32)?;
        let moe_cur_moe         = gpu.zeros(&[dim_alloc_moe], DType::F32)?;
        let moe_expert_gate_up  = gpu.zeros(&[2 * mi_alloc],  DType::F32)?;
        let moe_expert_hidden   = gpu.zeros(&[mi_alloc],      DType::F32)?;
        let moe_expert_out      = gpu.zeros(&[dim_alloc_moe], DType::F32)?;

        Ok(Gemma4Scratch {
            x, residual, tmp, pos_buf,
            q, k, v, attn_out,
            gate_ffn, up_ffn, ffn_hidden, ffn_out,
            logits, sample_buf, repeat_buf,
            flash_partials,
            v_norm_ones_full,
            per_layer_inp, per_layer_inp_proj, per_layer_tmp,
            per_layer_out, per_layer_pe_in,
            moe_router_in, moe_router_logits, moe_topk_indices, moe_topk_weights,
            moe_pre2, moe_cur_mlp, moe_cur_moe,
            moe_expert_gate_up, moe_expert_hidden, moe_expert_out,
        })
    }

    /// Release every GPU allocation owned by this scratch. Mirrors the
    /// Qwen35Scratch / LlamaScratch pattern so `unload_model` in the daemon
    /// can reclaim VRAM on idle eviction.
    pub fn free_gpu(self, gpu: &mut Gpu) {
        let _ = gpu.free_tensor(self.x);
        let _ = gpu.free_tensor(self.residual);
        let _ = gpu.free_tensor(self.tmp);
        // pos_buf is a DeviceBuffer, not a GpuTensor; rely on Drop.
        let _ = gpu.free_tensor(self.q);
        let _ = gpu.free_tensor(self.k);
        let _ = gpu.free_tensor(self.v);
        let _ = gpu.free_tensor(self.attn_out);
        let _ = gpu.free_tensor(self.gate_ffn);
        let _ = gpu.free_tensor(self.up_ffn);
        let _ = gpu.free_tensor(self.ffn_hidden);
        let _ = gpu.free_tensor(self.ffn_out);
        let _ = gpu.free_tensor(self.logits);
        let _ = gpu.free_tensor(self.sample_buf);
        let _ = gpu.free_tensor(self.repeat_buf);
        let _ = gpu.free_tensor(self.flash_partials);
        let _ = gpu.free_tensor(self.v_norm_ones_full);
        let _ = gpu.free_tensor(self.per_layer_inp);
        let _ = gpu.free_tensor(self.per_layer_inp_proj);
        let _ = gpu.free_tensor(self.per_layer_tmp);
        let _ = gpu.free_tensor(self.per_layer_out);
        let _ = gpu.free_tensor(self.per_layer_pe_in);
        let _ = gpu.free_tensor(self.moe_router_in);
        let _ = gpu.free_tensor(self.moe_router_logits);
        let _ = gpu.free_tensor(self.moe_topk_indices);
        let _ = gpu.free_tensor(self.moe_topk_weights);
        let _ = gpu.free_tensor(self.moe_pre2);
        let _ = gpu.free_tensor(self.moe_cur_mlp);
        let _ = gpu.free_tensor(self.moe_cur_moe);
        let _ = gpu.free_tensor(self.moe_expert_gate_up);
        let _ = gpu.free_tensor(self.moe_expert_hidden);
        let _ = gpu.free_tensor(self.moe_expert_out);
    }
}

// ─── Forward pass ───────────────────────────────────────────────────────

/// Run the Gemma 4 MoE routed-experts branch. Caller has already filled
/// `scratch.ffn_out` with the standard SwiGLU output for the same layer.
/// On exit, `scratch.tmp` holds `post_feedforward_layernorm(cur_mlp + cur_moe)`
/// — ready to be added into the residual by the caller.
///
/// Mirrors llama.cpp gemma4-iswa.cpp lines 125-179:
///
///     cur_mlp = post_feedforward_layernorm_1(SwiGLU output)
///     pre2    = pre_feedforward_layernorm_2(attn_out)
///     router_in = rmsnorm(attn_out, router_scale) * 1/sqrt(dim)
///     logits   = router_proj @ router_in
///     (idx, w) = top_k_softmax_renorm(logits)
///     cur_moe  = Σ_k (per_expert_scale[idx[k]] * w[k]) * expert[idx[k]](pre2)
///     cur_moe  = post_feedforward_layernorm_2(cur_moe)
///     combined = cur_mlp + cur_moe
///     out      = post_feedforward_layernorm(combined)
fn apply_moe_branch(
    gpu: &mut Gpu,
    config: &Gemma4Config,
    scratch: &Gemma4Scratch,
    moe: &MoeLayerExtras,
    post_ffn_norm: &GpuTensor,
    attn_out: &GpuTensor,
) -> HipResult<()> {
    let dim = config.dim;
    let dim_bytes = dim * 4;
    let mi = config.moe_intermediate_size;
    let n_exp = config.num_experts;
    let k_top = config.top_k_experts;

    // 1) cur_mlp = post_feedforward_layernorm_1(scratch.ffn_out)
    gpu.rmsnorm_f32(&scratch.ffn_out, &moe.post_feedforward_layernorm_1,
        &scratch.moe_cur_mlp, config.norm_eps)?;
    // HIPFIRE_DUMP_MOE_VALS=1 prints magnitude (rms, min, max, abs_mean) of
    // each of attn_out / ffn_out / cur_mlp / cur_moe / final_combined at the
    // first MoE layer of the first decode. Helps spot a single-branch
    // dominating or a NaN-adjacent buffer without full reference values.
    //
    // Only the first invocation per process emits — prevents 30 layers ×
    // N tokens × 6 lines from drowning the log. Uses an AtomicBool with
    // acquire-release ordering so the first dispatcher to compare-exchange
    // wins; subsequent calls see DUMPED=true and skip. Replaces the prior
    // `std::env::set_var("__HIPFIRE_MOE_VALS_DUMPED", "1")` hack which is
    // process-wide and unsafe in modern Rust (set_var races with reads).
    use std::sync::atomic::{AtomicBool, Ordering};
    static DUMPED: AtomicBool = AtomicBool::new(false);
    let dump_vals = std::env::var("HIPFIRE_DUMP_MOE_VALS").ok().as_deref() == Some("1")
        && DUMPED.compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire).is_ok();
    let stat = |gpu: &mut Gpu, t: &GpuTensor, label: &str| -> HipResult<()> {
        let v = gpu.download_f32(t)?;
        let n = v.len();
        let rms = (v.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>() / n as f64).sqrt();
        let mn = v.iter().cloned().fold(f32::INFINITY, f32::min);
        let mx = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let abs_mean = v.iter().map(|x| x.abs() as f64).sum::<f64>() / n as f64;
        let nonfinite = v.iter().filter(|x| !x.is_finite()).count();
        eprintln!("[moe-vals] {label:30}  n={n:6}  rms={rms:.4}  min={mn:.4}  max={mx:.4}  abs_mean={abs_mean:.4}  nonfinite={nonfinite}");
        Ok(())
    };
    if dump_vals {
        stat(gpu, attn_out, "attn_out (input to MoE)")?;
        stat(gpu, &scratch.ffn_out, "ffn_out (post-SwiGLU)")?;
        stat(gpu, &scratch.moe_cur_mlp, "cur_mlp (post_norm_1)")?;
        // Also dump the norm WEIGHTS to see their magnitude — large peak
        // values would amplify post-rmsnorm outputs nonlinearly.
        stat(gpu, &moe.post_feedforward_layernorm_1, "WEIGHT post_norm_1")?;
        stat(gpu, &moe.pre_feedforward_layernorm_2, "WEIGHT pre_norm_2")?;
        stat(gpu, &moe.post_feedforward_layernorm_2, "WEIGHT post_norm_2")?;
        stat(gpu, post_ffn_norm, "WEIGHT post_norm")?;
        stat(gpu, &moe.router_scale, "WEIGHT router_scale")?;
    }

    // 2) MoE branch input: pre2 = pre_feedforward_layernorm_2(attn_out)
    gpu.rmsnorm_f32(attn_out, &moe.pre_feedforward_layernorm_2,
        &scratch.moe_pre2, config.norm_eps)?;

    // 3) Router input: rmsnorm(attn_out, router_scale) * 1/sqrt(dim).
    //    rmsnorm_f32(x, w) = w * x / sqrt(mean(x²) + eps); identical to
    //    the ref `rms_norm + mul(router_scale)` factoring since they're
    //    elementwise-commutative.
    gpu.rmsnorm_f32(attn_out, &moe.router_scale,
        &scratch.moe_router_in, config.norm_eps)?;
    gpu.scale_f32(&scratch.moe_router_in, 1.0 / (dim as f32).sqrt())?;

    // 4) Router GEMV → logits [n_exp]
    weight_gemv(gpu, &moe.router_proj, &scratch.moe_router_in, &scratch.moe_router_logits)?;

    // 5) Top-K softmax + renorm on device. The shared kernel is hardcoded
    //    to k_top=8 (matches Qwen3.5-MoE A3B and Gemma 4 26B-A4B).
    if k_top != 8 {
        return Err(hip_bridge::HipError::new(
            0, &format!("MoE top_k_experts={k_top} unsupported (kernel hardcoded to 8)"),
        ));
    }
    gpu.moe_softmax_topk_renorm_k8(
        &scratch.moe_router_logits,
        &scratch.moe_topk_indices,
        &scratch.moe_topk_weights,
        n_exp,
        true, // renormalize selected probs
    )?;

    // 6) D2H the top-K dispatch table (8×i32 + 8×f32 = 64B per token, per layer).
    //    The router is data-dependent so we can't avoid this sync entirely
    //    without a fully GPU-side indexed expert dispatch — that's a v2
    //    optimization equivalent to Qwen3.5-MoE's `gemv_indexed_*` kernels.
    let topk_idx_bytes = gpu.download_f32(&scratch.moe_topk_indices)?;
    let topk_indices: Vec<usize> = unsafe {
        std::slice::from_raw_parts(topk_idx_bytes.as_ptr() as *const i32, k_top)
    }.iter().map(|&i| i as usize).collect();
    let topk_weights = gpu.download_f32(&scratch.moe_topk_weights)?;
    if std::env::var("HIPFIRE_DUMP_MOE").ok().as_deref() == Some("1") {
        let logits = gpu.download_f32(&scratch.moe_router_logits)?;
        let mut top5: Vec<(usize, f32)> = logits.iter().enumerate().map(|(i,&v)|(i,v)).collect();
        top5.select_nth_unstable_by(4, |a,b| b.1.partial_cmp(&a.1).unwrap());
        top5[..5].sort_by(|a,b| b.1.partial_cmp(&a.1).unwrap());
        let scale_sample: Vec<f32> = topk_indices.iter().take(8).map(|&e| moe.per_expert_scale_host[e]).collect();
        eprintln!("[moe] logit_top5={:?} | topk_idx={:?} | topk_w={:?} | per_expert_scale={:?}",
            &top5[..5], topk_indices, &topk_weights[..k_top], scale_sample);
    }

    // 7) Zero accumulator
    gpu.hip.memset(&scratch.moe_cur_moe.buf, 0, dim_bytes)?;

    // Debug isolation: HIPFIRE_MOE_ZERO=1 skips per-expert dispatch entirely
    // (cur_moe stays zero post the post_norm_2). Test whether output gets
    // closer to coherent when the broken MoE branch is muted.
    let moe_zero = std::env::var("HIPFIRE_MOE_ZERO").ok().as_deref() == Some("1");
    if moe_zero {
        // Skip the loop. cur_moe = 0 → cur_mlp + 0 = standard MLP path.
        // Note: post_feedforward_layernorm_2(0) is still 0 since rmsnorm
        // of all-zeros yields all-zeros (with eps in the denom, x/sqrt(eps)=0).
        gpu.rmsnorm_f32(&scratch.moe_cur_moe, &moe.post_feedforward_layernorm_2,
            &scratch.moe_cur_moe, config.norm_eps)?;
        gpu.add_f32(&scratch.moe_cur_mlp, &scratch.moe_cur_moe, &scratch.tmp)?;
        gpu.rmsnorm_f32(&scratch.tmp, post_ffn_norm, &scratch.tmp, config.norm_eps)?;
        return Ok(());
    }

    // 8) Per-expert dispatch: for each selected expert e,
    //    out_e = down_proj_e @ (gelu_tanh(gate_up_proj_e[..mi]) * gate_up_proj_e[mi..])
    //    cur_moe += (topk_weights[k] * per_expert_scale[e]) * out_e
    for ki in 0..k_top {
        let e = topk_indices[ki];
        // Defensive bound check — invalid indices would silently OOB here,
        // and the kernel-side top-K is supposed to clamp to [0, n_exp).
        if e >= n_exp {
            return Err(hip_bridge::HipError::new(
                0, &format!("MoE topk index {e} out of range (n_exp={n_exp})"),
            ));
        }
        let weight = topk_weights[ki] * moe.per_expert_scale_host[e];
        let expert = &moe.experts[e];

        // gate_up = gate_up_proj @ moe_pre2 → [2 * mi]
        weight_gemv(gpu, &expert.gate_up_proj, &scratch.moe_pre2, &scratch.moe_expert_gate_up)?;
        // Split: gate (first mi), up (second mi). HIPFIRE_MOE_SWAP_GATE_UP=1
        // tries the opposite layout (up first, gate second) for diagnosis.
        let (gate, up) = if std::env::var("HIPFIRE_MOE_SWAP_GATE_UP").ok().as_deref() == Some("1") {
            (scratch.moe_expert_gate_up.sub_offset(mi, mi),
             scratch.moe_expert_gate_up.sub_offset(0, mi))
        } else {
            (scratch.moe_expert_gate_up.sub_offset(0, mi),
             scratch.moe_expert_gate_up.sub_offset(mi, mi))
        };
        // Per Gemma 4 config `hidden_activation: 'gelu_pytorch_tanh'`, FFN
        // uses tanh-approximation. HIPFIRE_MOE_GELU_ERF=1 swaps to exact-erf
        // GELU for diagnosis (matches what fixed the per-layer-embed inject
        // on E4B — the per-layer block uses `nn.GELU()` not gelu_pytorch_tanh).
        if std::env::var("HIPFIRE_MOE_GELU_ERF").ok().as_deref() == Some("1") {
            gpu.gelu_erf_f32(&gate, &scratch.moe_expert_hidden, mi)?;
        } else {
            gpu.gelu_tanh_f32(&gate, &scratch.moe_expert_hidden, mi)?;
        }
        gpu.mul_f32(&scratch.moe_expert_hidden, &up, &scratch.moe_expert_hidden)?;
        // out = down_proj @ hidden → [dim]
        weight_gemv(gpu, &expert.down_proj, &scratch.moe_expert_hidden, &scratch.moe_expert_out)?;
        // cur_moe += weight * out
        gpu.scaled_add_inplace_cpu_scalar_f32(&scratch.moe_cur_moe, &scratch.moe_expert_out, weight)?;
    }

    if dump_vals {
        stat(gpu, &scratch.moe_cur_moe, "cur_moe (raw expert sum)")?;
    }

    // 9) cur_moe = post_feedforward_layernorm_2(cur_moe), in-place
    gpu.rmsnorm_f32(&scratch.moe_cur_moe, &moe.post_feedforward_layernorm_2,
        &scratch.moe_cur_moe, config.norm_eps)?;

    // 10) combined = cur_mlp + cur_moe → scratch.tmp
    gpu.add_f32(&scratch.moe_cur_mlp, &scratch.moe_cur_moe, &scratch.tmp)?;

    // 11) tmp = post_feedforward_layernorm(combined)
    gpu.rmsnorm_f32(&scratch.tmp, post_ffn_norm, &scratch.tmp, config.norm_eps)?;

    if dump_vals {
        stat(gpu, &scratch.moe_cur_moe, "cur_moe (post_norm_2)")?;
        stat(gpu, &scratch.tmp, "combined+post_norm")?;
        // The AtomicBool above (DUMPED) was already flipped by the
        // compare_exchange at function entry, so subsequent layers see
        // dump_vals=false and skip. No additional latch needed here.
    }

    let _ = dim_bytes;
    Ok(())
}

/// Apply the Gemma 4 E-series per-layer-embedding side channel at the end of
/// a layer's forward pass — between FFN+residual and `layer_scalar`. Mirrors
/// llama.cpp gemma4-iswa.cpp lines 202-224:
///
///     pe_in = cur                                  # save the residual stream
///     cur = per_layer_input_gate @ cur             # [n_embd_per_layer]
///     cur = gelu(cur)
///     cur = cur * inp_per_layer[il]                # [n_embd_per_layer], element-wise
///     cur = per_layer_projection @ cur             # [dim]
///     cur = rmsnorm(cur, post_per_layer_input_norm)
///     cur = pe_in + cur                            # residual back into x
///
/// Caller is responsible for the `n_embd_per_layer == 0` skip — no-op layers
/// don't pay any cost (zero-sized scratch buffers, no kernel launches).
///
/// We use `gelu_tanh_f32` here because Gemma 4 follows the gemma-pytorch-tanh
/// convention everywhere else (FFN SwiGLU activation). The exact GELU vs
/// tanh-approximation difference is sub-1% per element; if a future test
/// shows divergence vs HF reference, swap to an exact-erf GELU kernel.
fn apply_per_layer_inject(
    gpu: &mut Gpu,
    config: &Gemma4Config,
    scratch: &Gemma4Scratch,
    inp_gate: &WeightTensor,
    proj: &WeightTensor,
    post_norm: &GpuTensor,
    layer_idx: usize,
) -> HipResult<()> {
    let pl_w = config.n_embd_per_layer;
    let dim = config.dim;
    let dim_bytes = dim * 4;

    // Save current x → pe_in (residual stream snapshot before the inject).
    gpu.hip.memcpy_dtod(&scratch.per_layer_pe_in.buf, &scratch.x.buf, dim_bytes)?;

    // tmp_pl = per_layer_input_gate @ x → [n_embd_per_layer]
    weight_gemv(gpu, inp_gate, &scratch.x, &scratch.per_layer_tmp)?;
    // Empirical: erf-GELU produces materially better E4B/E2B output than
    // tanh-GELU, even though HF Gemma4TextDecoderLayer technically wires
    // `act_fn = ACT2FN[config.hidden_activation]` (= gelu_pytorch_tanh).
    // Tested 2026-05-02 with gelu_tanh: E4B emits pure-digit gibberish
    // ("The capital of France is 3417"); reverted. The HF code may be
    // inconsistent with the published checkpoint, or our pre-/post-norm
    // path interacts with the activation differently than HF's. Either
    // way, gelu_erf is what makes the E-series coherent at MG4 — keep
    // until the divergence is rigorously diagnosed against a Python
    // reference forward of layer 0 with both activations.
    // HIPFIRE_GEMMA4_PLE_GELU_TANH=1 swaps to tanh approx for diagnosis.
    if std::env::var("HIPFIRE_GEMMA4_PLE_GELU_TANH").ok().as_deref() == Some("1") {
        gpu.gelu_tanh_f32(&scratch.per_layer_tmp, &scratch.per_layer_tmp, pl_w)?;
    } else {
        gpu.gelu_erf_f32(&scratch.per_layer_tmp, &scratch.per_layer_tmp, pl_w)?;
    }

    // tmp_pl *= inp_per_layer[layer_idx*pl_w .. (layer_idx+1)*pl_w]
    let inp_slice = scratch.per_layer_inp.sub_offset(layer_idx * pl_w, pl_w);
    gpu.mul_f32(&scratch.per_layer_tmp, &inp_slice, &scratch.per_layer_tmp)?;

    // x_pl = per_layer_projection @ tmp_pl → [dim]
    weight_gemv(gpu, proj, &scratch.per_layer_tmp, &scratch.per_layer_out)?;
    // rmsnorm in place with post_per_layer_input_norm weight.
    gpu.rmsnorm_f32(&scratch.per_layer_out, post_norm, &scratch.per_layer_out, config.norm_eps)?;

    // x = pe_in + x_pl. Restore pe_in into x then add the projected channel.
    gpu.hip.memcpy_dtod(&scratch.x.buf, &scratch.per_layer_pe_in.buf, dim_bytes)?;
    gpu.add_inplace_f32(&scratch.x, &scratch.per_layer_out)?;

    Ok(())
}

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

    // 2b) Per-layer embedding pre-loop (E2B / E4B only — NOOP on 31B / 26B-A4B
    // where n_embd_per_layer == 0). Builds scratch.per_layer_inp = the per-token
    // per-layer side-channel signal that gets injected after each layer's
    // FFN+residual but before layer_scalar. Mirrors llama.cpp gemma4-iswa.cpp
    // build_inp_per_layer + project_per_layer_inputs (lines 263-322):
    //
    //     a = embed_tokens_per_layer[T]                       # [pl_dim]
    //     a *= sqrt(n_embd_per_layer)
    //     b = per_layer_model_proj @ x_embed                  # [pl_dim] = [n_embd_per_layer * n_layers]
    //     b *= 1 / sqrt(n_embd)
    //     b = rmsnorm_batched(b, per_layer_proj_norm, batch=n_layers, n=n_embd_per_layer)
    //     per_layer_inp = (a + b) * (1 / sqrt(2))
    if config.n_embd_per_layer > 0 {
        let pl_dim = config.n_embd_per_layer * config.n_layers;
        let pl_embd = weights.per_layer_tok_embd.as_ref().expect("per_layer_tok_embd loaded");
        let model_proj = weights.per_layer_model_proj.as_ref().expect("per_layer_model_proj loaded");
        let proj_norm = weights.per_layer_proj_norm.as_ref().expect("per_layer_proj_norm loaded");

        // (a) Per-token row lookup → scratch.per_layer_inp.
        // The per-layer embed table is forced to Q8F16 by the quantizer (see
        // `should_quantize` + the `embed_tokens_per_layer.weight` carve-out
        // in hipfire-quantize), so this dispatch always works regardless of
        // the model-level weight quant.
        match weights.per_layer_tok_embd_format {
            EmbeddingFormat::Q8_0 => gpu.embedding_lookup_q8(pl_embd, &scratch.per_layer_inp, token, pl_dim)?,
            EmbeddingFormat::F32  => gpu.embedding_lookup(pl_embd, &scratch.per_layer_inp, token, pl_dim)?,
            other => return Err(hip_bridge::HipError::new(
                0, &format!("unsupported per_layer_tok_embd format {other:?}"),
            )),
        }
        gpu.scale_f32(&scratch.per_layer_inp, (config.n_embd_per_layer as f32).sqrt())?;

        // (b) Project the regular post-embed hidden state into the per-layer
        // slots. weight_gemv reads scratch.x (already scaled by sqrt(dim)
        // above — that scaling is the standard embed_scale, NOT specific to
        // the per-layer branch), produces [pl_dim] into per_layer_inp_proj.
        weight_gemv(gpu, model_proj, &scratch.x, &scratch.per_layer_inp_proj)?;
        gpu.scale_f32(&scratch.per_layer_inp_proj, 1.0 / (config.dim as f32).sqrt())?;
        // Per-slot RMSNorm: weight is [n_embd_per_layer], applied identically
        // to each of n_layers consecutive slots.
        gpu.rmsnorm_batched(&scratch.per_layer_inp_proj, proj_norm, &scratch.per_layer_inp_proj,
            config.n_layers, config.n_embd_per_layer, config.norm_eps)?;

        // (c) Combine: per_layer_inp = (per_layer_inp + per_layer_inp_proj) * (1/sqrt(2)).
        gpu.add_inplace_f32(&scratch.per_layer_inp, &scratch.per_layer_inp_proj)?;
        gpu.scale_f32(&scratch.per_layer_inp, 1.0 / 2.0_f32.sqrt())?;
    }

    // 3) Per-layer forward. KV-sharing routing comes from `kv_share_plan`:
    // own-KV layers compute K/V and write to their own cache slot; shared
    // layers (E2B / E4B last 18-20 layers) skip K/V projection and read
    // from the anchor slot. On 31B / 26B-A4B (no shared layers) every
    // layer owns its KV and the plan trivially returns sequential slots.
    //
    // Note: kv_share_plan() rebuilds per-call which is fine for decode
    // (negligible cost vs the 60-layer forward); the daemon would lift
    // this to load-time once we wire CLI integration.
    let plan = config.kv_share_plan();
    for (layer_idx, layer_type) in config.layer_types.iter().copied().enumerate() {
        let owns = plan.owns_kv[layer_idx];
        let slot = plan.kv_slot[layer_idx];
        match (layer_type, &weights.layers[layer_idx]) {
            (LayerType::Sliding, LayerWeights::Sliding(lw)) => {
                sliding_layer_decode(gpu, config, lw, pos, kv_sliding, slot, owns, layer_idx, scratch)?;
            }
            (LayerType::Full, LayerWeights::Full(lw)) => {
                full_layer_decode(gpu, config, lw, pos, kv_full, slot, owns, layer_idx, scratch)?;
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
    kv_slot: usize,
    owns_kv: bool,
    layer_idx: usize,
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

    // Q is always projected (every layer needs its own Q, regardless of
    // KV ownership). K/V are skipped on shared layers — their k_proj /
    // v_proj weights still ship in HF safetensors but are stale/unused;
    // the actual K/V data is read from the KV cache slot of the anchor
    // layer (sliding anchor = layer n_layer_kv_from_start - 2).
    weight_gemv(gpu, &lw.q_proj, &scratch.tmp, &scratch.q)?;
    if owns_kv {
        weight_gemv(gpu, &lw.k_proj, &scratch.tmp, &scratch.k)?;
        weight_gemv(gpu, &lw.v_proj, &scratch.tmp, &scratch.v)?;
    }

    // q_norm + (own-only) k_norm + v_norm across head_dim (in-place).
    // V uses the no-scale RMS pattern (ones buffer as weight) — same as
    // full-attn layers. Matches llama.cpp gemma4-iswa.cpp line 92:
    //   Vcur = ggml_rms_norm(ctx0, Vcur, hparams.f_norm_rms_eps);
    gpu.rmsnorm_batched(&scratch.q, &lw.q_norm, &scratch.q, n_heads, head_dim, config.norm_eps)?;
    if owns_kv {
        gpu.rmsnorm_batched(&scratch.k, &lw.k_norm, &scratch.k, n_kv, head_dim, config.norm_eps)?;
        gpu.rmsnorm_batched(&scratch.v, &scratch.v_norm_ones_full, &scratch.v,
            n_kv, head_dim, config.norm_eps)?;
    }

    // Pre-scale Q by sqrt(head_dim) so the flash-attn kernel's internal
    // 1/sqrt(head_dim) cancels, leaving the effective Gemma 4 scale of 1.0.
    gpu.scale_f32(&scratch.q, (head_dim as f32).sqrt())?;

    // RoPE: always rotate Q. K is rotated only on own-kv layers (we pass
    // n_heads_k=0 to skip the K loop in the kernel for shared layers; the
    // existing K cache contents stay untouched). theta=10000, head_dim=256
    // (all dims rotate).
    let n_kv_for_rope = if owns_kv { n_kv } else { 0 };
    gpu.rope_f32(&scratch.q, &scratch.k, &scratch.pos_buf,
        n_heads, n_kv_for_rope, head_dim, config.sliding_rope_theta)?;

    // KV cache write + flash attention with window_size=1024.
    // - Own-KV layers: write current K/V into kv_slot, then attend.
    // - Shared layers: skip write (cache slot already populated by an
    //   earlier own-KV layer); just attend against the existing data.
    // Branch on cache quant mode, same as qwen35::run_fa_layer_body.
    if kv_cache.quant_asym3 {
        let ct = kv_cache.givens_cos.as_ref().unwrap();
        let st = kv_cache.givens_sin.as_ref().unwrap();
        if owns_kv {
            gpu.kv_cache_write_asym3_fused(
                &kv_cache.k_gpu[kv_slot], &kv_cache.v_gpu[kv_slot],
                &scratch.k, &scratch.v, &scratch.pos_buf, ct, st, n_kv, head_dim)?;
        }
        gpu.attention_flash_asym3(
            &scratch.q, &kv_cache.k_gpu[kv_slot], &kv_cache.v_gpu[kv_slot],
            &scratch.attn_out, &scratch.pos_buf, ct, st, pos + 1,
            n_heads, n_kv, head_dim, kv_cache.max_seq,
            &scratch.flash_partials,
            config.sliding_window as u32,
        )?;
    } else if kv_cache.quant_asym4 {
        let ct = kv_cache.givens_cos.as_ref().unwrap();
        let st = kv_cache.givens_sin.as_ref().unwrap();
        if owns_kv {
            gpu.kv_cache_write_asym4_fused(
                &kv_cache.k_gpu[kv_slot], &kv_cache.v_gpu[kv_slot],
                &scratch.k, &scratch.v, &scratch.pos_buf, ct, st, n_kv, head_dim)?;
        }
        gpu.attention_flash_asym4(
            &scratch.q, &kv_cache.k_gpu[kv_slot], &kv_cache.v_gpu[kv_slot],
            &scratch.attn_out, &scratch.pos_buf, ct, st, pos + 1,
            n_heads, n_kv, head_dim, kv_cache.max_seq,
            &scratch.flash_partials,
            config.sliding_window as u32,
        )?;
    } else if kv_cache.quant_asym2 {
        let ct = kv_cache.givens_cos.as_ref().unwrap();
        let st = kv_cache.givens_sin.as_ref().unwrap();
        if owns_kv {
            gpu.kv_cache_write_asym2_fused(
                &kv_cache.k_gpu[kv_slot], &kv_cache.v_gpu[kv_slot],
                &scratch.k, &scratch.v, &scratch.pos_buf, ct, st, n_kv, head_dim)?;
        }
        gpu.attention_flash_asym2(
            &scratch.q, &kv_cache.k_gpu[kv_slot], &kv_cache.v_gpu[kv_slot],
            &scratch.attn_out, &scratch.pos_buf, ct, st, pos + 1,
            n_heads, n_kv, head_dim, kv_cache.max_seq,
            &scratch.flash_partials,
            config.sliding_window as u32,
        )?;
    } else if kv_cache.quant_q8 {
        if owns_kv {
            gpu.kv_cache_write_q8_0(&kv_cache.k_gpu[kv_slot], &scratch.k, &scratch.pos_buf, n_kv, head_dim)?;
            gpu.kv_cache_write_q8_0(&kv_cache.v_gpu[kv_slot], &scratch.v, &scratch.pos_buf, n_kv, head_dim)?;
        }
        gpu.attention_flash_q8_0(
            &scratch.q, &kv_cache.k_gpu[kv_slot], &kv_cache.v_gpu[kv_slot],
            &scratch.attn_out, &scratch.pos_buf, pos + 1,
            n_heads, n_kv, head_dim, kv_cache.max_seq,
            &scratch.flash_partials,
            config.sliding_window as u32,
        )?;
    } else {
        // Plain FP32 KV path (kvf16 / kvfp32).
        let kv_dim = n_kv * head_dim;
        gpu.kv_cache_write(&kv_cache.k_gpu[kv_slot], &scratch.k, &scratch.pos_buf, kv_dim)?;
        gpu.kv_cache_write(&kv_cache.v_gpu[kv_slot], &scratch.v, &scratch.pos_buf, kv_dim)?;
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

    // Diagnostic: dump magnitudes at sliding-layer 0 only.
    if layer_idx == 0 && std::env::var("HIPFIRE_DUMP_LAYER0").ok().as_deref() == Some("1") {
        let v = gpu.download_f32(&scratch.x)?;
        let n = v.len();
        let rms = (v.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>() / n as f64).sqrt();
        let mn = v.iter().cloned().fold(f32::INFINITY, f32::min);
        let mx = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        eprintln!("[layer0] post-attn-residual  n={n}  rms={rms:.4}  min={mn:.4}  max={mx:.4}");
    }

    // Pre-FFN norm.
    gpu.rmsnorm_f32(&scratch.x, &lw.pre_feedforward_layernorm, &scratch.tmp, config.norm_eps)?;

    if layer_idx == 0 && std::env::var("HIPFIRE_DUMP_LAYER0").ok().as_deref() == Some("1") {
        let v = gpu.download_f32(&scratch.tmp)?;
        let n = v.len();
        let rms = (v.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>() / n as f64).sqrt();
        let mn = v.iter().cloned().fold(f32::INFINITY, f32::min);
        let mx = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        eprintln!("[layer0] pre_ffn_norm output n={n}  rms={rms:.4}  min={mn:.4}  max={mx:.4}");
    }

    // SwiGLU(gelu_pytorch_tanh): gate_proj, up_proj, gelu_tanh(gate) * up → down_proj.
    // Use lw.ffn_hidden_dim — varies per layer on E2B (use_double_wide_mlp).
    weight_gemv(gpu, &lw.gate_proj, &scratch.tmp, &scratch.gate_ffn)?;
    weight_gemv(gpu, &lw.up_proj, &scratch.tmp, &scratch.up_ffn)?;
    gpu.gelu_tanh_f32(&scratch.gate_ffn, &scratch.ffn_hidden, lw.ffn_hidden_dim)?;
    gpu.mul_f32(&scratch.ffn_hidden, &scratch.up_ffn, &scratch.ffn_hidden)?;
    weight_gemv(gpu, &lw.down_proj, &scratch.ffn_hidden, &scratch.ffn_out)?;

    // Post-FFN norm. On MoE layers (26B-A4B), apply the parallel routed-
    // experts branch and combine via the sandwich norms before the outer
    // post_feedforward_layernorm. On dense layers, just the standard
    // post_feedforward_layernorm.
    let moe_bypass = std::env::var("HIPFIRE_MOE_BYPASS").ok().as_deref() == Some("1");
    match (lw.moe.as_ref(), moe_bypass) {
        (Some(moe), false) => apply_moe_branch(gpu, config, scratch, moe,
            &lw.post_feedforward_layernorm, &scratch.residual)?,
        _ => gpu.rmsnorm_f32(&scratch.ffn_out, &lw.post_feedforward_layernorm,
            &scratch.tmp, config.norm_eps)?,
    }

    // x = residual + tmp (again, reset x from saved residual).
    gpu.hip.memcpy_dtod(&scratch.x.buf, &scratch.residual.buf, dim_bytes)?;
    gpu.add_inplace_f32(&scratch.x, &scratch.tmp)?;

    // Per-layer embedding inject (E2B / E4B only — None on 31B / 26B-A4B).
    if let (Some(g), Some(p), Some(n)) = (
        lw.per_layer_input_gate.as_ref(),
        lw.per_layer_projection.as_ref(),
        lw.post_per_layer_input_norm.as_ref(),
    ) {
        apply_per_layer_inject(gpu, config, scratch, g, p, n, layer_idx)?;
    }

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
    kv_slot: usize,
    owns_kv: bool,
    layer_idx: usize,
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

    // Q always projected. K + V only on own-KV layers — shared layers reuse
    // the anchor full layer's KV slot.
    //
    // V handling depends on `attention_k_eq_v`:
    //   • true  (31B / 26B-A4B): V is captured from K's PRE-norm output —
    //     a memcpy from scratch.k after k_proj. lw.v_proj is None.
    //   • false (E2B / E4B):     V comes from a separate v_proj weight.
    //     lw.v_proj is Some.
    weight_gemv(gpu, &lw.q_proj, &scratch.tmp, &scratch.q)?;
    if owns_kv {
        weight_gemv(gpu, &lw.k_proj, &scratch.tmp, &scratch.k)?;
        match &lw.v_proj {
            Some(vw) => {
                weight_gemv(gpu, vw, &scratch.tmp, &scratch.v)?;
            }
            None => {
                // CRITICAL: capture pre-k_norm bytes as V before applying k_norm.
                gpu.hip.memcpy_dtod(&scratch.v.buf, &scratch.k.buf, kv_bytes)?;
            }
        }
    }

    // q_norm always; k_norm + v_norm only on own-KV layers (head_dim = 512).
    gpu.rmsnorm_batched(&scratch.q, &lw.q_norm, &scratch.q, n_heads, head_dim, config.norm_eps)?;
    if owns_kv {
        gpu.rmsnorm_batched(&scratch.k, &lw.k_norm, &scratch.k, n_kv, head_dim, config.norm_eps)?;
        gpu.rmsnorm_batched(&scratch.v, &scratch.v_norm_ones_full, &scratch.v,
            n_kv, head_dim, config.norm_eps)?;
    }

    // Pre-scale Q by sqrt(head_dim=512).
    gpu.scale_f32(&scratch.q, (head_dim as f32).sqrt())?;

    // Proportional RoPE: rotate Q always; K only on own-KV layers
    // (n_heads_k=0 in the kernel skips the K loop for shared layers).
    let n_rot_pairs = ((head_dim as f32) * config.full_partial_rotary_factor * 0.5) as usize;
    let n_kv_for_rope = if owns_kv { n_kv } else { 0 };
    gpu.rope_partial_halved_f32(&scratch.q, &scratch.k, &scratch.pos_buf,
        n_heads, n_kv_for_rope, head_dim, n_rot_pairs, config.full_rope_theta)?;

    // KV cache write + attention. Full-attn layers (head_dim=512) used to
    // require an FP32 KV path because all quantized flash kernels hardcoded
    // a single-warp × 8-dims-per-thread layout that truncated past 256.
    // The `*_hd512` variants (commit landing this comment) extend asym3 to
    // 32 threads × 16 dims (= 2 chunks of 8) and unblock quantized KV on
    // full layers. asym4/asym2/q8 are still hd=256-only; those paths
    // continue to require FP32 here.
    //
    // No sliding window on full layers (window_size=0 = full causal).
    if kv_cache.quant_asym3 {
        let ct = kv_cache.givens_cos.as_ref().unwrap();
        let st = kv_cache.givens_sin.as_ref().unwrap();
        if owns_kv {
            gpu.kv_cache_write_asym3_fused(
                &kv_cache.k_gpu[kv_slot], &kv_cache.v_gpu[kv_slot],
                &scratch.k, &scratch.v, &scratch.pos_buf, ct, st, n_kv, head_dim)?;
        }
        gpu.attention_flash_asym3(
            &scratch.q, &kv_cache.k_gpu[kv_slot], &kv_cache.v_gpu[kv_slot],
            &scratch.attn_out, &scratch.pos_buf, ct, st, pos + 1,
            n_heads, n_kv, head_dim, kv_cache.max_seq,
            &scratch.flash_partials,
            0u32, // full-attn layer — no sliding window
        )?;
    } else if kv_cache.quantized {
        return Err(hip_bridge::HipError::new(
            0,
            "gemma4 full-attn layer: only asym3 quantized KV is supported at \
             head_dim=512 today (asym4/asym2/q8 truncate). Use new_gpu_asym3 \
             or fall back to FP32 (KvCache::new_gpu).",
        ));
    } else {
        let kv_dim = n_kv * head_dim;
        if owns_kv {
            gpu.kv_cache_write(&kv_cache.k_gpu[kv_slot], &scratch.k, &scratch.pos_buf, kv_dim)?;
            gpu.kv_cache_write(&kv_cache.v_gpu[kv_slot], &scratch.v, &scratch.pos_buf, kv_dim)?;
        }
        gpu.attention_f32(
            &scratch.q, &kv_cache.k_gpu[kv_slot], &kv_cache.v_gpu[kv_slot],
            &scratch.attn_out, &scratch.pos_buf, pos + 1,
            n_heads, n_kv, head_dim, kv_cache.max_seq,
        )?;
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

    // SwiGLU with gelu_pytorch_tanh activation. lw.ffn_hidden_dim is per-layer
    // (use_double_wide_mlp on E2B doubles it for shared-KV layers).
    weight_gemv(gpu, &lw.gate_proj, &scratch.tmp, &scratch.gate_ffn)?;
    weight_gemv(gpu, &lw.up_proj, &scratch.tmp, &scratch.up_ffn)?;
    gpu.gelu_tanh_f32(&scratch.gate_ffn, &scratch.ffn_hidden, lw.ffn_hidden_dim)?;
    gpu.mul_f32(&scratch.ffn_hidden, &scratch.up_ffn, &scratch.ffn_hidden)?;
    weight_gemv(gpu, &lw.down_proj, &scratch.ffn_hidden, &scratch.ffn_out)?;

    // Post-FFN norm + (optional) MoE branch — same dispatch as sliding.
    let moe_bypass = std::env::var("HIPFIRE_MOE_BYPASS").ok().as_deref() == Some("1");
    match (lw.moe.as_ref(), moe_bypass) {
        (Some(moe), false) => apply_moe_branch(gpu, config, scratch, moe,
            &lw.post_feedforward_layernorm, &scratch.residual)?,
        _ => gpu.rmsnorm_f32(&scratch.ffn_out, &lw.post_feedforward_layernorm,
            &scratch.tmp, config.norm_eps)?,
    }

    // x = residual + tmp.
    gpu.hip.memcpy_dtod(&scratch.x.buf, &scratch.residual.buf, dim_bytes)?;
    gpu.add_inplace_f32(&scratch.x, &scratch.tmp)?;

    // Per-layer embedding inject (E-series only). See sliding decoder for
    // the rationale; same call site (after FFN+residual, before layer_scalar).
    if let (Some(g), Some(p), Some(n)) = (
        lw.per_layer_input_gate.as_ref(),
        lw.per_layer_projection.as_ref(),
        lw.post_per_layer_input_norm.as_ref(),
    ) {
        apply_per_layer_inject(gpu, config, scratch, g, p, n, layer_idx)?;
    }

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
