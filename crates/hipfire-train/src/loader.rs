// SPDX-License-Identifier: Apache-2.0
//! Load a dense LLaMA safetensors checkpoint into fp32 GPU tensors.
//!
//! Uses `SafetensorsSource` purely as a name→raw-bytes mmap (no quantizer
//! involvement, per Phase 0 plan §2). Every weight is converted to fp32 on
//! upload — Supra-50M ships bf16. Weights are the *frozen base*; the trainable
//! LoRA adapters are created separately.

use crate::config::LlamaConfig;
use hipfire_model::ModelSource;
use hipfire_rdna::{Gpu, GpuTensor};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::safetensors_source::SafetensorsSource;
use std::path::Path;

/// Per-layer frozen weights (HF row-major `[out, in]`, ready for
/// `gemm_f32_train` with `trans_b=true`).
pub struct LlamaLayerF32 {
    pub input_layernorm: GpuTensor, // [hidden]
    /// qwen3-style QK-norm, `[head_dim]`. `None` on llama-shaped models.
    pub q_norm: Option<GpuTensor>,
    pub k_norm: Option<GpuTensor>,
    /// True when `q_proj` emits `2*q_dim` (qwen3.5 `attn_output_gate`).
    /// DERIVED from the tensor's own shape, never from config: the runtime
    /// does the same (`infer_attn_output_gate_from_hfq`), because some routed
    /// Qwen3 artifacts set the config flag while storing plain Q.
    pub attn_out_gate: bool,
    pub q_proj: GpuTensor,                   // [q_dim, hidden]
    pub k_proj: GpuTensor,                   // [kv_dim, hidden]
    pub v_proj: GpuTensor,                   // [kv_dim, hidden]
    pub o_proj: GpuTensor,                   // [hidden, q_dim]
    pub post_attention_layernorm: GpuTensor, // [hidden]
    pub gate_proj: GpuTensor,                // [inter, hidden]
    pub up_proj: GpuTensor,                  // [inter, hidden]
    pub down_proj: GpuTensor,                // [hidden, inter]
}

pub struct LlamaWeightsF32 {
    pub embed_tokens: GpuTensor, // [vocab, hidden]
    pub layers: Vec<LlamaLayerF32>,
    pub final_norm: GpuTensor, // [hidden]
    /// `None` when `tie_word_embeddings` — logits use `embed_tokens`.
    pub lm_head: Option<GpuTensor>, // [vocab, hidden]
}

/// Open `dir`, parse config, and upload all weights as fp32.
pub fn load_llama_fp32(
    gpu: &mut Gpu,
    dir: &Path,
) -> Result<(LlamaConfig, LlamaWeightsF32), String> {
    let cfg = LlamaConfig::from_dir(dir)?;
    let src = SafetensorsSource::open(dir).map_err(|e| format!("open safetensors: {e}"))?;

    let load = |gpu: &mut Gpu, name: &str, want: &[usize]| -> Result<GpuTensor, String> {
        load_tensor_f32(gpu, &src, name, want)
    };

    let h = cfg.hidden_size;
    let q = cfg.q_dim();
    let kv = cfg.kv_dim();
    let inter = cfg.intermediate_size;

    let embed_tokens = load(gpu, "model.embed_tokens.weight", &[cfg.vocab_size, h])?;
    let final_norm = load(gpu, "model.norm.weight", &[h])?;
    let lm_head = if cfg.tie_word_embeddings {
        None
    } else {
        Some(load(gpu, "lm_head.weight", &[cfg.vocab_size, h])?)
    };

    let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
    for i in 0..cfg.num_hidden_layers {
        let p = format!("model.layers.{i}");
        layers.push(LlamaLayerF32 {
            input_layernorm: load(gpu, &format!("{p}.input_layernorm.weight"), &[h])?,
            q_norm: None,
            k_norm: None,
            attn_out_gate: false,
            q_proj: load(gpu, &format!("{p}.self_attn.q_proj.weight"), &[q, h])?,
            k_proj: load(gpu, &format!("{p}.self_attn.k_proj.weight"), &[kv, h])?,
            v_proj: load(gpu, &format!("{p}.self_attn.v_proj.weight"), &[kv, h])?,
            o_proj: load(gpu, &format!("{p}.self_attn.o_proj.weight"), &[h, q])?,
            post_attention_layernorm: load(
                gpu,
                &format!("{p}.post_attention_layernorm.weight"),
                &[h],
            )?,
            gate_proj: load(gpu, &format!("{p}.mlp.gate_proj.weight"), &[inter, h])?,
            up_proj: load(gpu, &format!("{p}.mlp.up_proj.weight"), &[inter, h])?,
            down_proj: load(gpu, &format!("{p}.mlp.down_proj.weight"), &[h, inter])?,
        });
    }

    Ok((
        cfg,
        LlamaWeightsF32 {
            embed_tokens,
            layers,
            final_norm,
            lm_head,
        },
    ))
}

/// Load an fp32 LLaMA from a `.hfq` artifact instead of a safetensors directory.
///
/// Exists because the models we actually measure are `.hfq` — the HF snapshots on
/// this box ship Meta `.pth`, and `hipfire-coexistence export safetensors`
/// implements only `zaya`. Config comes from the artifact's own metadata via
/// [`LlamaConfig::from_hfq_metadata`], so the base is the exact served artifact.
///
/// Weights are widened to f32 on load. That is EXACT for a bf16 or f16 source —
/// both are strict subsets of f32 — so it costs memory, not fidelity: ~4 GB for
/// a 1B model. It does not scale, and deliberately so: a 35B would want bf16
/// STORAGE through the forward and backward, which is a real change (the block
/// keeps f32 master weights and the backward is f32 by design; note
/// `HIPFIRE_TRAIN_LOWP=bf16` is bf16 COMPUTE over f32 storage and does not help
/// here). This path is for models that fit.
pub fn load_llama_fp32_hfq(
    gpu: &mut Gpu,
    path: &Path,
) -> Result<(LlamaConfig, LlamaWeightsF32), String> {
    let hfq = HfqFile::open(path).map_err(|e| format!("open hfq {}: {e}", path.display()))?;
    let cfg = LlamaConfig::from_hfq_metadata(hfq.metadata_json())?;

    let load = |gpu: &mut Gpu, name: &str, want: &[usize]| -> Result<GpuTensor, String> {
        load_tensor_f32_hfq(gpu, &hfq, name, want)
    };

    let h = cfg.hidden_size;
    let q = cfg.q_dim();
    let kv = cfg.kv_dim();
    let inter = cfg.intermediate_size;

    let embed_tokens = load(gpu, "model.embed_tokens.weight", &[cfg.vocab_size, h])?;
    let final_norm = load(gpu, "model.norm.weight", &[h])?;
    let lm_head = if cfg.tie_word_embeddings {
        None
    } else {
        Some(load(gpu, "lm_head.weight", &[cfg.vocab_size, h])?)
    };

    let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
    for i in 0..cfg.num_hidden_layers {
        let p = format!("model.layers.{i}");
        layers.push(LlamaLayerF32 {
            input_layernorm: load(gpu, &format!("{p}.input_layernorm.weight"), &[h])?,
            q_norm: None,
            k_norm: None,
            attn_out_gate: false,
            q_proj: load(gpu, &format!("{p}.self_attn.q_proj.weight"), &[q, h])?,
            k_proj: load(gpu, &format!("{p}.self_attn.k_proj.weight"), &[kv, h])?,
            v_proj: load(gpu, &format!("{p}.self_attn.v_proj.weight"), &[kv, h])?,
            o_proj: load(gpu, &format!("{p}.self_attn.o_proj.weight"), &[h, q])?,
            post_attention_layernorm: load(
                gpu,
                &format!("{p}.post_attention_layernorm.weight"),
                &[h],
            )?,
            gate_proj: load(gpu, &format!("{p}.mlp.gate_proj.weight"), &[inter, h])?,
            up_proj: load(gpu, &format!("{p}.mlp.up_proj.weight"), &[inter, h])?,
            down_proj: load(gpu, &format!("{p}.mlp.down_proj.weight"), &[h, inter])?,
        });
    }

    Ok((
        cfg,
        LlamaWeightsF32 {
            embed_tokens,
            layers,
            final_norm,
            lm_head,
        },
    ))
}

/// Load ONE layer's weights from a `.hfq`, for the layer-streamed gamma walk.
///
/// The per-model loader holds every layer at once, which is what caps
/// `calib_gamma` at models that fit f32 (~5 GB for a 1B, ~140 GB for a 35B
/// against 128 GB of UMA). Streaming one layer at a time makes peak residency
/// independent of depth: one layer's weights plus the boundary activations,
/// which are seq*hidden floats per layer and negligible beside them.
///
/// The caller is responsible for freeing the returned tensors before loading
/// the next layer — that is the entire point.
pub fn load_llama_layer_fp32_hfq(
    gpu: &mut Gpu,
    hfq: &HfqFile,
    cfg: &LlamaConfig,
    layer: usize,
) -> Result<LlamaLayerF32, String> {
    load_llama_layer_fp32_hfq_pfx(gpu, hfq, "model.", cfg, layer, true)
}

/// As [`load_llama_layer_fp32_hfq`], with an explicit name prefix and the
/// option to skip the dense MLP.
///
/// A routed layer has no `mlp.{gate,up,down}_proj` at all — its MLP is the
/// experts — so `dense_mlp = false` substitutes 1-element placeholders. Nothing
/// reads them: `block_forward_attn_only` and `block_backward_from_dxn2` skip the
/// dense MLP entirely.
pub fn load_llama_layer_fp32_hfq_pfx<S: WeightSource + ?Sized>(
    gpu: &mut Gpu,
    src: &S,
    prefix: &str,
    cfg: &LlamaConfig,
    layer: usize,
    dense_mlp: bool,
) -> Result<LlamaLayerF32, String> {
    load_llama_layer_fp32_pfx_off(gpu, src, prefix, cfg, layer, dense_mlp, false)
}

/// As [`load_llama_layer_fp32_hfq_pfx`], with the GemmaRMSNorm unit offset.
#[allow(clippy::too_many_arguments)]
pub fn load_llama_layer_fp32_pfx_off<S: WeightSource + ?Sized>(
    gpu: &mut Gpu,
    src: &S,
    prefix: &str,
    cfg: &LlamaConfig,
    layer: usize,
    dense_mlp: bool,
    unit_offset: bool,
) -> Result<LlamaLayerF32, String> {
    let h = cfg.hidden_size;
    let q = cfg.q_dim();
    let kv = cfg.kv_dim();
    let inter = cfg.intermediate_size;
    let p = format!("{prefix}layers.{layer}");
    // Sequential rather than a closure: a `|..| load_tensor_f32_hfq(gpu, ..)`
    // closure holds `&mut gpu` for its whole life, which blocks the placeholder
    // allocations below.
    let input_layernorm = upload_tensor(gpu, src, &format!("{p}.input_layernorm.weight"), &[h])?;
    let q_rows = src
        .shape_of(&format!("{p}.self_attn.q_proj.weight"))
        .ok_or_else(|| format!("layer {layer}: no q_proj"))?[0];
    let attn_out_gate = match q_rows {
        r if r == q => false,
        r if r == 2 * q => true,
        r => {
            return Err(format!(
                "layer {layer}: q_proj rows {r} are neither q_dim {q} nor 2*q_dim {}",
                2 * q
            ))
        }
    };
    let q_proj = upload_tensor(
        gpu,
        src,
        &format!("{p}.self_attn.q_proj.weight"),
        &[q_rows, h],
    )?;
    let k_proj = upload_tensor(gpu, src, &format!("{p}.self_attn.k_proj.weight"), &[kv, h])?;
    let v_proj = upload_tensor(gpu, src, &format!("{p}.self_attn.v_proj.weight"), &[kv, h])?;
    let o_proj = upload_tensor(gpu, src, &format!("{p}.self_attn.o_proj.weight"), &[h, q])?;
    let post_attention_layernorm = upload_tensor(
        gpu,
        src,
        &format!("{p}.post_attention_layernorm.weight"),
        &[h],
    )?;

    let (gate_proj, up_proj, down_proj) = if dense_mlp {
        (
            upload_tensor(gpu, src, &format!("{p}.mlp.gate_proj.weight"), &[inter, h])?,
            upload_tensor(gpu, src, &format!("{p}.mlp.up_proj.weight"), &[inter, h])?,
            upload_tensor(gpu, src, &format!("{p}.mlp.down_proj.weight"), &[h, inter])?,
        )
    } else {
        let z = |g: &mut Gpu| {
            g.zeros(&[1], hipfire_rdna::DType::F32)
                .map_err(|e| e.to_string())
        };
        (z(gpu)?, z(gpu)?, z(gpu)?)
    };

    // qwen3-style QK-norm, per HEAD_DIM not per hidden. Absent on llama.
    let hd = cfg.head_dim;
    let q_norm = match src.has(&format!("{p}.self_attn.q_norm.weight")) {
        true => Some(upload_tensor(
            gpu,
            src,
            &format!("{p}.self_attn.q_norm.weight"),
            &[hd],
        )?),
        false => None,
    };
    let k_norm = match src.has(&format!("{p}.self_attn.k_norm.weight")) {
        true => Some(upload_tensor(
            gpu,
            src,
            &format!("{p}.self_attn.k_norm.weight"),
            &[hd],
        )?),
        false => None,
    };

    if unit_offset {
        apply_unit_offset(gpu, &input_layernorm)?;
        apply_unit_offset(gpu, &post_attention_layernorm)?;
    }
    Ok(LlamaLayerF32 {
        input_layernorm,
        q_norm,
        k_norm,
        attn_out_gate,
        q_proj,
        k_proj,
        v_proj,
        o_proj,
        post_attention_layernorm,
        gate_proj,
        up_proj,
        down_proj,
    })
}

/// One MoE layer's frozen weights: router plus every expert's SwiGLU.
pub struct MoeLayerF32 {
    pub router: GpuTensor,
    /// Per expert, in expert order: (gate, up, down).
    pub experts: Vec<(GpuTensor, GpuTensor, GpuTensor)>,
    /// The always-on shared branch, when the architecture has one.
    pub shared: Option<SharedLayerF32>,
}

/// One layer's shared-expert weights. `scalar_gate` is `[1, h]` and is NOT the
/// SwiGLU gate — see `ops::moe::SharedExpert`.
pub struct SharedLayerF32 {
    pub scalar_gate: GpuTensor,
    pub gate: GpuTensor,
    pub up: GpuTensor,
    pub down: GpuTensor,
    pub inter: usize,
}

/// A name→fp32 weight source, so the MoE loaders work against either an `.hfq`
/// artifact or a raw safetensors directory.
///
/// This exists because the stacked/fused expert layout could otherwise only be
/// tested by inspection: the 35B target is `.hfq`, but the only fixture that
/// has that layout (`qwen3_5_moe-tiny`) is safetensors, and
/// `import safetensors` implements 'zaya' alone. One trait means the fixture
/// exercises the SAME slicing code the artifact will, rather than a parallel
/// copy that can drift.
pub trait WeightSource {
    fn shape_of(&self, name: &str) -> Option<Vec<usize>>;
    fn fetch_f32(&self, name: &str) -> Result<(Vec<usize>, Vec<f32>), String>;
    fn has(&self, name: &str) -> bool {
        self.shape_of(name).is_some()
    }
}

impl WeightSource for HfqFile {
    fn shape_of(&self, name: &str) -> Option<Vec<usize>> {
        self.find_tensor_info(name)
            .map(|i| i.shape.iter().map(|&d| d as usize).collect())
    }
    fn fetch_f32(&self, name: &str) -> Result<(Vec<usize>, Vec<f32>), String> {
        fetch_f32_hfq(self, name)
    }
}

impl WeightSource for SafetensorsSource {
    fn shape_of(&self, name: &str) -> Option<Vec<usize>> {
        self.tensor_data(name).map(|(i, _)| i.shape.clone())
    }
    fn fetch_f32(&self, name: &str) -> Result<(Vec<usize>, Vec<f32>), String> {
        let (info, bytes) = self
            .tensor_data(name)
            .ok_or_else(|| format!("missing tensor {name}"))?;
        let f32s = bytes_to_f32(&info.dtype, bytes).map_err(|e| format!("tensor {name}: {e}"))?;
        Ok((info.shape.clone(), f32s))
    }
}

/// Fetch, validate shape, upload.
fn upload_tensor<S: WeightSource + ?Sized>(
    gpu: &mut Gpu,
    src: &S,
    name: &str,
    want_shape: &[usize],
) -> Result<GpuTensor, String> {
    let (shape, f32s) = src.fetch_f32(name)?;
    if shape != want_shape {
        return Err(format!(
            "tensor {name}: shape {shape:?} != expected {want_shape:?}"
        ));
    }
    gpu.upload_f32(&f32s, want_shape)
        .map_err(|e| format!("upload {name}: {e}"))
}

/// One `linear_attn` layer's fp32 weights.
///
/// The four small tensors stay on the host — see `la_block::
/// LinearAttnBlockWeights`, whose core consumes them as slices.
pub struct LinearAttnLayerF32 {
    pub input_layernorm: GpuTensor,
    pub post_attention_layernorm: GpuTensor,
    pub in_proj_qkv: GpuTensor,
    pub in_proj_a: GpuTensor,
    pub in_proj_b: GpuTensor,
    pub in_proj_z: GpuTensor,
    pub out_proj: GpuTensor,
    pub conv1d: Vec<f32>,
    pub a_log: Vec<f32>,
    pub dt_bias: Vec<f32>,
    pub norm: Vec<f32>,
    pub n_heads: usize,
    pub hd_k: usize,
    pub hd_v: usize,
    pub conv_k: usize,
}

/// Load a dense MLP's `(gate, up, down)` for one layer, with `inter` taken
/// from the tensor rather than config — a hybrid's linear_attn layers can
/// carry a different MLP width than its attention layers.
pub fn load_dense_mlp_fp32<S: WeightSource + ?Sized>(
    gpu: &mut Gpu,
    src: &S,
    prefix: &str,
    layer: usize,
    h: usize,
) -> Result<(GpuTensor, GpuTensor, GpuTensor, usize), String> {
    let p = format!("{prefix}layers.{layer}.mlp");
    let inter = src
        .shape_of(&format!("{p}.gate_proj.weight"))
        .ok_or_else(|| format!("layer {layer}: no mlp.gate_proj"))?[0];
    Ok((
        upload_tensor(gpu, src, &format!("{p}.gate_proj.weight"), &[inter, h])?,
        upload_tensor(gpu, src, &format!("{p}.up_proj.weight"), &[inter, h])?,
        upload_tensor(gpu, src, &format!("{p}.down_proj.weight"), &[h, inter])?,
        inter,
    ))
}

/// A [`WeightSource`] that DEQUANTIZES on read.
///
/// `HfqFile`'s own impl deliberately refuses quantized tensors: training on
/// dequantized weights is a different thing from training on the source, and
/// that guard should stay. This wrapper is the explicit opt-in, and it exists
/// for one job — running the hybrid assembly against a REAL model when no
/// unquantized hybrid is on disk.
///
/// **Gamma captured through this is not production gamma.** It measures the
/// quantized model's sensitivities, not the source's, which is the wrong input
/// for deciding where to spend bits. It is a validation tool.
pub struct DequantHfq<'a>(pub &'a HfqFile);

impl WeightSource for DequantHfq<'_> {
    fn shape_of(&self, name: &str) -> Option<Vec<usize>> {
        self.0.shape_of(name)
    }
    fn fetch_f32(&self, name: &str) -> Result<(Vec<usize>, Vec<f32>), String> {
        let (info, bytes) = self
            .0
            .tensor_data_cow(name)
            .ok_or_else(|| format!("missing tensor {name}"))?;
        let shape: Vec<usize> = info.shape.iter().map(|&d| d as usize).collect();
        let n: usize = shape.iter().product();
        let mut w = match info.quant_type {
            1 | 2 | 16 => {
                let dt = match info.quant_type {
                    1 => "F16",
                    2 => "F32",
                    _ => "BF16",
                };
                bytes_to_f32(dt, bytes.as_ref()).map_err(|e| format!("tensor {name}: {e}"))?
            }
            3 => hipfire_runtime::quant::dequant_q8f16(bytes.as_ref(), n),
            34 => hipfire_runtime::quant::dequant_oq4g256(bytes.as_ref(), n),
            35 => hipfire_runtime::quant::dequant_oq8g256(bytes.as_ref(), n),
            other => {
                return Err(format!(
                    "tensor {name}: quant_type {other} has no host dequantizer here; \
                     Oq4G256 (34), Oq8G256 (35) and Q8F16 (3) are wired"
                ))
            }
        };
        if w.len() != n {
            return Err(format!("tensor {name}: decoded {} != {n}", w.len()));
        }

        // AWQ fold. The artifact stores W*s and the runtime computes
        // (W*s)*(x/s) — see hfq.rs::load_awq_scale. This forward takes plain x,
        // so the scale has to come back out of the WEIGHT, per input column.
        // Getting the axis wrong here is silent: the shapes still line up.
        if shape.len() == 2 {
            let stem = name.strip_suffix(".weight").unwrap_or(name);
            let sidecar = format!("{stem}.awq_scale.weight");
            if let Some((si, sb)) = self.0.tensor_data_cow(&sidecar) {
                let k = shape[1];
                let sdt = match si.quant_type {
                    1 => "F16",
                    2 => "F32",
                    16 => "BF16",
                    o => return Err(format!("{sidecar}: unexpected quant_type {o}")),
                };
                let sc = bytes_to_f32(sdt, sb.as_ref()).map_err(|e| format!("{sidecar}: {e}"))?;
                if sc.len() != k {
                    return Err(format!("{sidecar}: len {} != K {k}", sc.len()));
                }
                for r in 0..shape[0] {
                    for c in 0..k {
                        w[r * k + c] /= sc[c];
                    }
                }
            }
        }
        Ok((shape, w))
    }
}

/// Does this model use GemmaRMSNorm — `rmsnorm(x) * (1 + w)` — for its block
/// and final norms?
///
/// Qwen3.5/3.6 do. The weights are stored as deviations from 1, which is why
/// they centre near 0 (layer-0 `input_layernorm` means +0.24) where a plain
/// RMSNorm weight would centre near 1. Applying them plainly is silent: shapes
/// match, everything stays differentiable, and the model merely gets quietly
/// worse. Measured against the runtime's own prefill logits at one token, the
/// difference is cos 0.7896 plain vs 0.9994 with the offset.
///
/// It does NOT apply to `q_norm`, `k_norm` or `linear_attn.norm` — those are
/// stored as ordinary weights (linear_attn.norm centres near +0.96), and
/// offsetting them too drops the same measurement back to 0.7840.
pub fn uses_unit_offset_norm(model_type: &str) -> bool {
    model_type.starts_with("qwen3_5") || model_type.starts_with("qwen3_next")
}

/// Add the GemmaRMSNorm unit offset to a loaded norm weight, in place.
fn apply_unit_offset(gpu: &mut Gpu, t: &GpuTensor) -> Result<(), String> {
    let mut v = gpu.download_f32(t).map_err(|e| format!("{e}"))?;
    for x in v.iter_mut() {
        *x += 1.0;
    }
    let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_ne_bytes()).collect();
    gpu.memcpy_htod_auto(&t.buf, &bytes)
        .map_err(|e| format!("unit-offset upload: {e}"))
}

/// Load `embed_tokens` as fp32 from any weight source.
pub fn load_embed_f32<S: WeightSource + ?Sized>(
    gpu: &mut Gpu,
    src: &S,
    prefix: &str,
    vocab: usize,
    h: usize,
) -> Result<GpuTensor, String> {
    upload_tensor(
        gpu,
        src,
        &format!("{prefix}embed_tokens.weight"),
        &[vocab, h],
    )
}

/// Load the final RMSNorm weight as fp32 from any weight source.
pub fn load_final_norm_f32<S: WeightSource + ?Sized>(
    gpu: &mut Gpu,
    src: &S,
    prefix: &str,
    h: usize,
    unit_offset: bool,
) -> Result<GpuTensor, String> {
    let t = upload_tensor(gpu, src, &format!("{prefix}norm.weight"), &[h])?;
    if unit_offset {
        apply_unit_offset(gpu, &t)?;
    }
    Ok(t)
}

/// Is layer `l` linear-attention rather than self-attention? qwen3.5/3.6 are
/// HYBRID — the 35B is 30 linear_attn layers to 10 full-attention — so this is
/// probed per layer, never inferred from the model.
pub fn layer_is_linear_attn<S: WeightSource + ?Sized>(src: &S, prefix: &str, layer: usize) -> bool {
    src.has(&format!(
        "{prefix}layers.{layer}.linear_attn.in_proj_qkv.weight"
    ))
}

/// Load one `linear_attn` layer.
///
/// Every head dimension is DERIVED from the tensors, not from config, and then
/// cross-checked: `hd_v` comes from `in_proj_z` and must equal `norm`'s length,
/// and `hd_k` falls out of `in_proj_qkv`'s `[Q|K|V]` width. A config that
/// disagreed with the checkpoint would otherwise reshape the recurrence
/// silently.
pub fn load_linear_attn_layer_fp32<S: WeightSource + ?Sized>(
    gpu: &mut Gpu,
    src: &S,
    prefix: &str,
    layer: usize,
    h: usize,
    unit_offset: bool,
) -> Result<LinearAttnLayerF32, String> {
    let p = format!("{prefix}layers.{layer}.linear_attn");
    let (_, a_log) = src.fetch_f32(&format!("{p}.A_log"))?;
    let n_heads = a_log.len();
    if n_heads == 0 {
        return Err(format!("layer {layer}: empty A_log"));
    }
    let (_, norm) = src.fetch_f32(&format!("{p}.norm.weight"))?;
    let hd_v = norm.len();

    let z_shape = src
        .shape_of(&format!("{p}.in_proj_z.weight"))
        .ok_or_else(|| format!("layer {layer}: no in_proj_z"))?;
    if z_shape[0] != n_heads * hd_v {
        return Err(format!(
            "layer {layer}: in_proj_z out {} != n_heads {n_heads} * hd_v {hd_v}",
            z_shape[0]
        ));
    }
    let qkv_shape = src
        .shape_of(&format!("{p}.in_proj_qkv.weight"))
        .ok_or_else(|| format!("layer {layer}: no in_proj_qkv"))?;
    let per_head = qkv_shape[0] / n_heads;
    if per_head * n_heads != qkv_shape[0] || per_head < hd_v || (per_head - hd_v) % 2 != 0 {
        return Err(format!(
            "layer {layer}: in_proj_qkv out {} is not n_heads*(2*hd_k + hd_v) with hd_v {hd_v}",
            qkv_shape[0]
        ));
    }
    let hd_k = (per_head - hd_v) / 2;

    let conv_shape = src
        .shape_of(&format!("{p}.conv1d.weight"))
        .ok_or_else(|| format!("layer {layer}: no conv1d"))?;
    if conv_shape[0] != qkv_shape[0] {
        return Err(format!(
            "layer {layer}: conv1d channels {} != qkv width {}",
            conv_shape[0], qkv_shape[0]
        ));
    }
    let conv_k = *conv_shape.last().unwrap();
    let (_, conv1d) = src.fetch_f32(&format!("{p}.conv1d.weight"))?;
    let (_, dt_bias) = src.fetch_f32(&format!("{p}.dt_bias"))?;

    let lp = format!("{prefix}layers.{layer}");
    // `norm` (the gated per-head one) is NOT offset — see uses_unit_offset_norm.
    let input_layernorm = upload_tensor(gpu, src, &format!("{lp}.input_layernorm.weight"), &[h])?;
    let post_attention_layernorm = upload_tensor(
        gpu,
        src,
        &format!("{lp}.post_attention_layernorm.weight"),
        &[h],
    )?;
    if unit_offset {
        apply_unit_offset(gpu, &input_layernorm)?;
        apply_unit_offset(gpu, &post_attention_layernorm)?;
    }
    Ok(LinearAttnLayerF32 {
        input_layernorm,
        post_attention_layernorm,
        in_proj_qkv: upload_tensor(
            gpu,
            src,
            &format!("{p}.in_proj_qkv.weight"),
            &[qkv_shape[0], h],
        )?,
        in_proj_a: upload_tensor(gpu, src, &format!("{p}.in_proj_a.weight"), &[n_heads, h])?,
        in_proj_b: upload_tensor(gpu, src, &format!("{p}.in_proj_b.weight"), &[n_heads, h])?,
        in_proj_z: upload_tensor(
            gpu,
            src,
            &format!("{p}.in_proj_z.weight"),
            &[n_heads * hd_v, h],
        )?,
        out_proj: upload_tensor(
            gpu,
            src,
            &format!("{p}.out_proj.weight"),
            &[h, n_heads * hd_v],
        )?,
        conv1d,
        a_log,
        dt_bias,
        norm,
        n_heads,
        hd_k,
        hd_v,
        conv_k,
    })
}

pub fn free_linear_attn_layer_fp32(gpu: &mut Gpu, l: LinearAttnLayerF32) -> Result<(), String> {
    for t in [
        l.input_layernorm,
        l.post_attention_layernorm,
        l.in_proj_qkv,
        l.in_proj_a,
        l.in_proj_b,
        l.in_proj_z,
        l.out_proj,
    ] {
        gpu.free_tensor(t)
            .map_err(|e| format!("free linear_attn tensor: {e}"))?;
    }
    Ok(())
}

/// Is layer `l` routed? Detected from the artifact rather than the config,
/// because hybrid models exist — BLS-Mini-Code-1.0 has a DENSE layer 0 and
/// routed layers 1..49, so a per-model flag would be wrong for it.
pub fn layer_is_moe<S: WeightSource + ?Sized>(src: &S, prefix: &str, layer: usize) -> bool {
    let p = format!("{prefix}layers.{layer}.mlp.experts");
    // Two on-disk shapes: per-expert tensors (BLS, Mixtral) and the stacked
    // form qwen3.5/3.6 uses, where all experts live in one 3-D tensor.
    src.has(&format!("{p}.0.down_proj.weight"))
        || src.has(&format!("{p}.down_proj"))
        || src.has(&format!("{p}.down_proj.weight"))
}

/// Tensor-name prefix an artifact uses: `model.` or `model.language_model.`
/// (the multimodal wrapper). Probed once rather than guessed.
pub fn detect_prefix<S: WeightSource + ?Sized>(src: &S) -> &'static str {
    // Probe input_layernorm, not self_attn: a hybrid can have linear_attn at
    // layer 0 and no self_attn there at all, which made the old probe report
    // the wrong prefix for exactly the models this crate now targets.
    if src.has("model.language_model.layers.0.input_layernorm.weight") {
        "model.language_model."
    } else {
        "model."
    }
}

/// Load one MoE layer's router and experts.
///
/// `inter` here is the EXPERT intermediate size, which differs from the dense
/// `intermediate_size` in config (BLS: 768 per expert against a 3072 dense
/// layer 0), so it is taken from the expert tensor's own shape rather than the
/// config.
pub fn load_moe_layer_fp32<S: WeightSource + ?Sized>(
    gpu: &mut Gpu,
    src: &S,
    prefix: &str,
    layer: usize,
    h: usize,
    n_experts: usize,
) -> Result<(MoeLayerF32, usize), String> {
    let p = format!("{prefix}layers.{layer}.mlp");
    let router = upload_tensor(gpu, src, &format!("{p}.gate.weight"), &[n_experts, h])?;

    // Three layouts in the wild, all seen on real artifacts:
    //   experts.N.gate_proj + up_proj      per-expert, unfused (BLS, Mixtral)
    //   experts.N.gate_up_proj  [2*I, H]   per-expert, FUSED   (what
    //                                      hipfire-quantize emits for MoE)
    //   experts.gate_up_proj    [E, 2I, H] stacked + fused     (HF export)
    let (experts, inter) = if src.has(&format!("{p}.experts.0.gate_proj.weight")) {
        load_experts_per_tensor(gpu, src, &p, layer, h, n_experts)?
    } else if src.has(&format!("{p}.experts.0.gate_up_proj.weight")) {
        load_experts_per_expert_fused(gpu, src, &p, layer, h, n_experts)?
    } else {
        load_experts_stacked(gpu, src, &p, layer, h, n_experts)?
    };

    // Shared branch is optional: qwen3.5/3.6 has one, BLS does not.
    let sp = format!("{p}.shared_expert");
    let shared = match src.shape_of(&format!("{sp}.down_proj.weight")) {
        None => None,
        Some(shape) => {
            // [h, shared_inter] — the shared intermediate is its own size and
            // need not match the routed experts'.
            let si = shape[1];
            Some(SharedLayerF32 {
                scalar_gate: upload_tensor(
                    gpu,
                    src,
                    &format!("{p}.shared_expert_gate.weight"),
                    &[1, h],
                )?,
                gate: upload_tensor(gpu, src, &format!("{sp}.gate_proj.weight"), &[si, h])?,
                up: upload_tensor(gpu, src, &format!("{sp}.up_proj.weight"), &[si, h])?,
                down: upload_tensor(gpu, src, &format!("{sp}.down_proj.weight"), &[h, si])?,
                inter: si,
            })
        }
    };

    Ok((
        MoeLayerF32 {
            router,
            experts,
            shared,
        },
        inter,
    ))
}

type ExpertTriples = Vec<(GpuTensor, GpuTensor, GpuTensor)>;

/// BLS/Mixtral shape: one tensor per expert per projection.
fn load_experts_per_tensor<S: WeightSource + ?Sized>(
    gpu: &mut Gpu,
    src: &S,
    p: &str,
    layer: usize,
    h: usize,
    n_experts: usize,
) -> Result<(ExpertTriples, usize), String> {
    let inter = src
        .shape_of(&format!("{p}.experts.0.gate_proj.weight"))
        .ok_or_else(|| format!("layer {layer}: no experts.0.gate_proj"))?[0];
    let mut experts = Vec::with_capacity(n_experts);
    for e in 0..n_experts {
        let ep = format!("{p}.experts.{e}");
        experts.push((
            upload_tensor(gpu, src, &format!("{ep}.gate_proj.weight"), &[inter, h])?,
            upload_tensor(gpu, src, &format!("{ep}.up_proj.weight"), &[inter, h])?,
            upload_tensor(gpu, src, &format!("{ep}.down_proj.weight"), &[h, inter])?,
        ));
    }
    Ok((experts, inter))
}

/// Per-expert but FUSED: `experts.N.gate_up_proj.weight` is `[2*inter, h]`.
///
/// This is what `hipfire-quantize` emits for a routed MoE — it splits the
/// stacked HF tensor per expert but leaves gate and up fused, deferring the
/// halving to the runtime. Same `(gate || up)` order as the stacked form
/// (`qwen35/layout.rs`, and `moe_gate_up_unscatter_k8.hip` reads rows
/// `0..mi` as gate).
fn load_experts_per_expert_fused<S: WeightSource + ?Sized>(
    gpu: &mut Gpu,
    src: &S,
    p: &str,
    layer: usize,
    h: usize,
    n_experts: usize,
) -> Result<(ExpertTriples, usize), String> {
    let gu0 = src
        .shape_of(&format!("{p}.experts.0.gate_up_proj.weight"))
        .ok_or_else(|| format!("layer {layer}: no experts.0.gate_up_proj"))?;
    if gu0.len() != 2 || gu0[1] != h || gu0[0] % 2 != 0 {
        return Err(format!(
            "layer {layer}: experts.0.gate_up_proj shape {gu0:?} is not [2*inter, {h}]"
        ));
    }
    let inter = gu0[0] / 2;

    let mut experts = Vec::with_capacity(n_experts);
    for e in 0..n_experts {
        let ep = format!("{p}.experts.{e}");
        let (shape, gu) = src.fetch_f32(&format!("{ep}.gate_up_proj.weight"))?;
        if shape != vec![2 * inter, h] {
            return Err(format!(
                "layer {layer} expert {e}: gate_up shape {shape:?} != [{}, {h}]",
                2 * inter
            ));
        }
        experts.push((
            gpu.upload_f32(&gu[..inter * h], &[inter, h])
                .map_err(|r| format!("upload expert {e} gate: {r}"))?,
            gpu.upload_f32(&gu[inter * h..], &[inter, h])
                .map_err(|r| format!("upload expert {e} up: {r}"))?,
            upload_tensor(gpu, src, &format!("{ep}.down_proj.weight"), &[h, inter])?,
        ));
    }
    Ok((experts, inter))
}

/// qwen3.5/3.6 shape: all experts stacked into one 3-D tensor, with gate and up
/// fused into `gate_up_proj`.
///
/// `gate_up_proj` is `[n_experts, 2*inter, h]` and the layout comment in
/// `qwen35/layout.rs` pins the fusion as `(gate || up)` — gate is the FIRST
/// `inter` rows, not interleaved. `inter` is derived from that tensor rather
/// than from config, and `down_proj` is then required to agree, so a
/// transposed or interleaved variant fails loudly here instead of training
/// silently on scrambled weights.
fn load_experts_stacked<S: WeightSource + ?Sized>(
    gpu: &mut Gpu,
    src: &S,
    p: &str,
    layer: usize,
    h: usize,
    n_experts: usize,
) -> Result<(ExpertTriples, usize), String> {
    let gu_name = [
        format!("{p}.experts.gate_up_proj"),
        format!("{p}.experts.gate_up_proj.weight"),
    ]
    .into_iter()
    .find(|n| src.has(n))
    .ok_or_else(|| format!("layer {layer}: no per-expert or stacked gate_up_proj"))?;
    let dn_name = gu_name.replace("gate_up_proj", "down_proj");

    let (gu_shape, gu) = src.fetch_f32(&gu_name)?;
    let (dn_shape, dn) = src.fetch_f32(&dn_name)?;
    if gu_shape.len() != 3 || gu_shape[0] != n_experts || gu_shape[2] != h {
        return Err(format!(
            "layer {layer}: {gu_name} shape {gu_shape:?} is not [{n_experts}, 2*inter, {h}]"
        ));
    }
    if gu_shape[1] % 2 != 0 {
        return Err(format!(
            "layer {layer}: fused gate_up rows {} are odd, so it is not (gate || up)",
            gu_shape[1]
        ));
    }
    let inter = gu_shape[1] / 2;
    if dn_shape != vec![n_experts, h, inter] {
        return Err(format!(
            "layer {layer}: {dn_name} shape {dn_shape:?} != [{n_experts}, {h}, {inter}]"
        ));
    }

    let mut experts = Vec::with_capacity(n_experts);
    for e in 0..n_experts {
        let base = e * 2 * inter * h;
        let g = &gu[base..base + inter * h];
        let u = &gu[base + inter * h..base + 2 * inter * h];
        let d = &dn[e * h * inter..(e + 1) * h * inter];
        experts.push((
            gpu.upload_f32(g, &[inter, h])
                .map_err(|e| format!("upload expert gate: {e}"))?,
            gpu.upload_f32(u, &[inter, h])
                .map_err(|e| format!("upload expert up: {e}"))?,
            gpu.upload_f32(d, &[h, inter])
                .map_err(|e| format!("upload expert down: {e}"))?,
        ));
    }
    Ok((experts, inter))
}

/// Free one streamed MoE layer.
pub fn free_moe_layer_fp32(gpu: &mut Gpu, l: MoeLayerF32) -> Result<(), String> {
    gpu.free_tensor(l.router)
        .map_err(|e| format!("free router: {e}"))?;
    for (g, u, d) in l.experts {
        for t in [g, u, d] {
            gpu.free_tensor(t)
                .map_err(|e| format!("free expert tensor: {e}"))?;
        }
    }
    if let Some(sh) = l.shared {
        for t in [sh.scalar_gate, sh.gate, sh.up, sh.down] {
            gpu.free_tensor(t)
                .map_err(|e| format!("free shared expert tensor: {e}"))?;
        }
    }
    Ok(())
}

/// Free one streamed layer's tensors.
pub fn free_llama_layer_fp32(gpu: &mut Gpu, l: LlamaLayerF32) -> Result<(), String> {
    for t in [
        l.input_layernorm,
        l.q_proj,
        l.k_proj,
        l.v_proj,
        l.o_proj,
        l.post_attention_layernorm,
        l.gate_proj,
        l.up_proj,
        l.down_proj,
    ] {
        gpu.free_tensor(t)
            .map_err(|e| format!("free layer tensor: {e}"))?;
    }
    for t in [l.q_norm, l.k_norm].into_iter().flatten() {
        gpu.free_tensor(t)
            .map_err(|e| format!("free qk-norm tensor: {e}"))?;
    }
    Ok(())
}

/// `load_tensor_f32` against an HFQ artifact.
///
/// Uses `tensor_data_cow`, not `tensor_data`: `--bf16-codec` defaults to `huff`,
/// so a bf16 artifact's tensors may be losslessly recoded, and the borrowing
/// accessor returns None for those — reporting a present tensor as missing.
/// Decode one hfq tensor to fp32 on the host, returning its shape alongside.
///
/// Split out from [`load_tensor_f32_hfq`] because the stacked MoE tensors have
/// to be sliced per expert (and the fused `gate_up` halved) before upload, and
/// doing that through the upload path would mean decoding the whole stack once
/// per expert.
fn fetch_f32_hfq(hfq: &HfqFile, name: &str) -> Result<(Vec<usize>, Vec<f32>), String> {
    let (info, bytes) = hfq
        .tensor_data_cow(name)
        .ok_or_else(|| format!("missing tensor {name}"))?;
    let shape: Vec<usize> = info.shape.iter().map(|&d| d as usize).collect();
    let dtype = match info.quant_type {
        1 => "F16",
        2 => "F32",
        16 => "BF16",
        other => {
            return Err(format!(
                "tensor {name}: quant_type {other} is not an unquantized float;                  the fp32 training base needs an f16/f32/bf16 artifact"
            ))
        }
    };
    let f32s = bytes_to_f32(dtype, bytes.as_ref()).map_err(|e| format!("tensor {name}: {e}"))?;
    Ok((shape, f32s))
}

fn load_tensor_f32_hfq(
    gpu: &mut Gpu,
    hfq: &HfqFile,
    name: &str,
    want_shape: &[usize],
) -> Result<GpuTensor, String> {
    let (info, bytes) = hfq
        .tensor_data_cow(name)
        .ok_or_else(|| format!("missing tensor {name}"))?;
    let shape: Vec<usize> = info.shape.iter().map(|&d| d as usize).collect();
    if shape != want_shape {
        return Err(format!(
            "tensor {name}: shape {shape:?} != expected {want_shape:?}"
        ));
    }
    // Only unquantized sources make sense as a training base.
    let dtype = match info.quant_type {
        1 => "F16",
        2 => "F32",
        16 => "BF16",
        other => {
            return Err(format!(
                "tensor {name}: quant_type {other} is not an unquantized float;                  the fp32 training base needs an f16/f32/bf16 artifact"
            ))
        }
    };
    let f32s = bytes_to_f32(dtype, bytes.as_ref()).map_err(|e| format!("tensor {name}: {e}"))?;
    let expected: usize = want_shape.iter().product();
    if f32s.len() != expected {
        return Err(format!(
            "tensor {name}: {} elements != expected {expected}",
            f32s.len()
        ));
    }
    gpu.upload_f32(&f32s, want_shape)
        .map_err(|e| format!("upload {name}: {e}"))
}

/// Fetch a tensor's raw bytes, convert to fp32, validate shape, upload.
fn load_tensor_f32(
    gpu: &mut Gpu,
    src: &SafetensorsSource,
    name: &str,
    want_shape: &[usize],
) -> Result<GpuTensor, String> {
    let (info, bytes) = src
        .tensor_data(name)
        .ok_or_else(|| format!("missing tensor {name}"))?;
    if info.shape != want_shape {
        return Err(format!(
            "tensor {name}: shape {:?} != expected {:?}",
            info.shape, want_shape
        ));
    }
    let f32s = bytes_to_f32(&info.dtype, bytes).map_err(|e| format!("tensor {name}: {e}"))?;
    let expected: usize = want_shape.iter().product();
    if f32s.len() != expected {
        return Err(format!(
            "tensor {name}: {} elems != {} from shape",
            f32s.len(),
            expected
        ));
    }
    gpu.upload_f32(&f32s, want_shape)
        .map_err(|e| format!("upload {name}: {e:?}"))
}

/// Decode an HFQM tensor's bytes (by `quant_type`) to fp32 — the layer-1 runtime
/// unification: training loads its base from the *exact served artifact*.
/// Handles BF16(16)/F32(2)/Q8F16(3) now; Qtip3G256(31) is a clear TODO (qtip2-sim
/// `.hfq` is all bf16, so this covers the 2-bit path today).
fn decode_hfq_tensor(quant_type: u8, data: &[u8], n: usize) -> Result<Vec<f32>, String> {
    match quant_type {
        2 => Ok(data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()),
        16 => Ok(data
            .chunks_exact(2)
            .map(|c| crate::hfq_patch::bf16_bits_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect()),
        3 => Ok(hipfire_runtime::quant::dequant_q8f16(data, n)),
        31 => Err(
            "Qtip3G256 .hfq decode not yet implemented in hipfire-train \
                   (use a bf16/qtip2-sim .hfq, or load from the source safetensors)"
                .to_string(),
        ),
        other => Err(format!("unsupported quant_type {other} for hfq decode")),
    }
}

/// Parallel HFQ Q8F16 dequant (34-byte blocks: f16 scale + 32×i8). The gemma3
/// embedding table is ~671M elements and the serial `dequant_q8f16` is a
/// multi-minute single-threaded startup tax on every training run; blocks are
/// independent, so split the block stream across cores. Falls back to the serial
/// path if the byte length doesn't match the 34-B block layout (layout guard).
fn dequant_q8f16_parallel(data: &[u8], n: usize) -> Vec<f32> {
    const BYTES_PER_BLOCK: usize = 34;
    if n % 32 != 0 || data.len() != (n / 32) * BYTES_PER_BLOCK {
        return hipfire_runtime::quant::dequant_q8f16(data, n);
    }
    let total_blocks = n / 32;
    let threads = std::thread::available_parallelism()
        .map(|x| x.get())
        .unwrap_or(8)
        .clamp(1, total_blocks.max(1));
    let blocks_per = total_blocks.div_ceil(threads);
    let mut out = vec![0.0f32; n];
    std::thread::scope(|s| {
        let mut rest = out.as_mut_slice();
        let mut b0 = 0usize;
        while b0 < total_blocks {
            let nb = blocks_per.min(total_blocks - b0);
            let (chunk, tail) = rest.split_at_mut(nb * 32);
            rest = tail;
            let dslice = &data[b0 * BYTES_PER_BLOCK..(b0 + nb) * BYTES_PER_BLOCK];
            s.spawn(move || {
                chunk.copy_from_slice(&hipfire_runtime::quant::dequant_q8f16(dslice, nb * 32));
            });
            b0 += nb;
        }
    });
    out
}

/// Load a dense LLaMA model's base weights directly from a `.hfq` artifact,
/// decoded to fp32 — so the training "student" IS the served model (no
/// re-quantize / format-matching). Config comes from the HFQM metadata.
pub fn load_llama_from_hfq(
    gpu: &mut Gpu,
    path: &Path,
) -> Result<(LlamaConfig, LlamaWeightsF32), String> {
    use std::collections::HashMap;
    let bytes = std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let (entries, meta) = crate::hfq_patch::parse_hfq(&bytes)?;
    let cfg = LlamaConfig::from_hfq_metadata(&meta)?;
    let map: HashMap<&str, &crate::hfq_patch::HfqEntry> =
        entries.iter().map(|e| (e.name.as_str(), e)).collect();

    let load = |gpu: &mut Gpu, name: &str, want: &[usize]| -> Result<GpuTensor, String> {
        let e = map
            .get(name)
            .ok_or_else(|| format!("missing tensor {name}"))?;
        let data = &bytes[e.data_offset..e.data_offset + e.data_size];
        let n: usize = want.iter().product();
        let f32s = decode_hfq_tensor(e.quant_type, data, n).map_err(|x| format!("{name}: {x}"))?;
        if f32s.len() != n {
            return Err(format!("{name}: {} elems != {n}", f32s.len()));
        }
        gpu.upload_f32(&f32s, want)
            .map_err(|x| format!("upload {name}: {x:?}"))
    };

    let (h, q, kv, inter) = (
        cfg.hidden_size,
        cfg.q_dim(),
        cfg.kv_dim(),
        cfg.intermediate_size,
    );
    let embed_tokens = load(gpu, "model.embed_tokens.weight", &[cfg.vocab_size, h])?;
    let final_norm = load(gpu, "model.norm.weight", &[h])?;
    let lm_head = if map.contains_key("lm_head.weight") {
        Some(load(gpu, "lm_head.weight", &[cfg.vocab_size, h])?)
    } else {
        None
    };
    let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
    for i in 0..cfg.num_hidden_layers {
        let p = format!("model.layers.{i}");
        layers.push(LlamaLayerF32 {
            input_layernorm: load(gpu, &format!("{p}.input_layernorm.weight"), &[h])?,
            q_norm: None,
            k_norm: None,
            attn_out_gate: false,
            q_proj: load(gpu, &format!("{p}.self_attn.q_proj.weight"), &[q, h])?,
            k_proj: load(gpu, &format!("{p}.self_attn.k_proj.weight"), &[kv, h])?,
            v_proj: load(gpu, &format!("{p}.self_attn.v_proj.weight"), &[kv, h])?,
            o_proj: load(gpu, &format!("{p}.self_attn.o_proj.weight"), &[h, q])?,
            post_attention_layernorm: load(
                gpu,
                &format!("{p}.post_attention_layernorm.weight"),
                &[h],
            )?,
            gate_proj: load(gpu, &format!("{p}.mlp.gate_proj.weight"), &[inter, h])?,
            up_proj: load(gpu, &format!("{p}.mlp.up_proj.weight"), &[inter, h])?,
            down_proj: load(gpu, &format!("{p}.mlp.down_proj.weight"), &[h, inter])?,
        });
    }
    Ok((
        cfg,
        LlamaWeightsF32 {
            embed_tokens,
            layers,
            final_norm,
            lm_head,
        },
    ))
}

/// Load a **gemma3** target's shared embedding (+ tied lm-head) as fp32 for
/// DSpark training. The trainer only reads `embed_tokens` (and `lm_head`, which
/// gemma3 ties → `None` ⇒ the loop falls back to `embed_tokens`); the `layers` /
/// `final_norm` fields are unused by the drafter training and left empty/zero.
///
/// The returned [`LlamaConfig`] carries gemma3's decoder dims so
/// `init_dspark_model` shapes the 5-layer drafter body as a dense GQA version of
/// the gemma3 block (h, n_heads, n_kv, head_dim, inter). gemma3's **global**
/// `rope_theta` is used (the drafter body is all-global dense; it does not model
/// the sliding-window local layers). The multimodal wrapper (`architecture ==
/// "gemma3"`) nests the decoder under `text_config` / `language_model.`.
pub fn load_gemma3_target_f32(
    gpu: &mut Gpu,
    path: &Path,
) -> Result<(LlamaConfig, LlamaWeightsF32), String> {
    use std::collections::HashMap;
    let bytes = std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let (entries, meta) = crate::hfq_patch::parse_hfq(&bytes)?;
    let v: serde_json::Value =
        serde_json::from_str(&meta).map_err(|e| format!("gemma3 meta json: {e}"))?;
    let arch = v.get("architecture").and_then(|a| a.as_str()).unwrap_or("");
    let config = v.get("config").ok_or("gemma3 meta: no config")?;
    let tc = config.get("text_config").unwrap_or(config);
    let u = |k: &str| tc.get(k).and_then(|x| x.as_u64()).map(|x| x as usize);

    let hidden_size = u("hidden_size").ok_or("gemma3: no hidden_size")?;
    let num_attention_heads = u("num_attention_heads").ok_or("gemma3: no num_attention_heads")?;
    let num_key_value_heads = u("num_key_value_heads").unwrap_or(num_attention_heads);
    let head_dim = u("head_dim").unwrap_or(hidden_size / num_attention_heads);
    let intermediate_size = u("intermediate_size").ok_or("gemma3: no intermediate_size")?;
    let vocab_size = u("vocab_size").ok_or("gemma3: no vocab_size")?;
    let num_hidden_layers = u("num_hidden_layers").unwrap_or(1);
    let max_position_embeddings = u("max_position_embeddings").unwrap_or(131072);
    let rms_norm_eps = tc
        .get("rms_norm_eps")
        .and_then(|x| x.as_f64())
        .unwrap_or(1e-6) as f32;
    let rope_theta = tc
        .get("rope_theta")
        .and_then(|x| x.as_f64())
        .unwrap_or(1_000_000.0) as f32;

    let cfg = LlamaConfig {
        hidden_size,
        intermediate_size,
        num_hidden_layers,
        num_attention_heads,
        num_key_value_heads,
        head_dim,
        vocab_size,
        rms_norm_eps,
        rope_theta,
        tie_word_embeddings: true,
        max_position_embeddings,
    };

    // Dequantize the (possibly Q8/HFQ4-packed) embedding table to fp32.
    let prefix = if arch == "gemma3_text" {
        ""
    } else {
        "language_model."
    };
    let embed_name = format!("{prefix}model.embed_tokens.weight");
    let map: HashMap<&str, &crate::hfq_patch::HfqEntry> =
        entries.iter().map(|e| (e.name.as_str(), e)).collect();
    let e = map
        .get(embed_name.as_str())
        .ok_or_else(|| format!("gemma3: missing tensor {embed_name}"))?;
    let data = &bytes[e.data_offset..e.data_offset + e.data_size];
    let n = vocab_size * hidden_size;
    // Q8F16 (the common embed format) uses the parallel dequant to avoid a
    // multi-minute single-threaded startup; other formats go through the shared
    // decoder.
    let f32s = if e.quant_type == 3 {
        dequant_q8f16_parallel(data, n)
    } else {
        decode_hfq_tensor(e.quant_type, data, n).map_err(|x| format!("{embed_name}: {x}"))?
    };
    if f32s.len() != n {
        return Err(format!("{embed_name}: {} elems != {n}", f32s.len()));
    }
    let embed_tokens = gpu
        .upload_f32(&f32s, &[vocab_size, hidden_size])
        .map_err(|x| format!("upload embed: {x:?}"))?;
    let final_norm = gpu
        .zeros(&[hidden_size], hipfire_rdna::DType::F32)
        .map_err(|x| format!("gemma3 final_norm: {x:?}"))?;

    Ok((
        cfg,
        LlamaWeightsF32 {
            embed_tokens,
            layers: Vec::new(),
            final_norm,
            lm_head: None,
        },
    ))
}

/// Load a DSpark target's shared embed/lm-head + decoder dims from a `.hfq`,
/// dispatching on the artifact's `architecture`: `gemma3*` →
/// [`load_gemma3_target_f32`], everything else → [`load_llama_from_hfq`].
pub fn load_target_f32(
    gpu: &mut Gpu,
    path: &Path,
) -> Result<(LlamaConfig, LlamaWeightsF32), String> {
    let bytes = std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let (_entries, meta) = crate::hfq_patch::parse_hfq(&bytes)?;
    let arch = serde_json::from_str::<serde_json::Value>(&meta)
        .ok()
        .and_then(|v| {
            v.get("architecture")
                .and_then(|a| a.as_str())
                .map(String::from)
        })
        .unwrap_or_default();
    if arch.starts_with("gemma3") {
        load_gemma3_target_f32(gpu, path)
    } else {
        load_llama_from_hfq(gpu, path)
    }
}

/// Convert little-endian safetensors bytes of the given dtype to fp32.
fn bytes_to_f32(dtype: &str, bytes: &[u8]) -> Result<Vec<f32>, String> {
    match dtype {
        "F32" => {
            if !bytes.len().is_multiple_of(4) {
                return Err("F32 byte len not /4".into());
            }
            Ok(bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect())
        }
        "F16" => Ok(bytes
            .chunks_exact(2)
            .map(|b| half::f16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32())
            .collect()),
        "BF16" => Ok(bytes
            .chunks_exact(2)
            .map(|b| half::bf16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32())
            .collect()),
        other => Err(format!("unsupported dtype {other} for fp32 training load")),
    }
}
