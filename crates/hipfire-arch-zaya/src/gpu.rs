// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! ZAYA1 GPU forward. Loads each big linear from the HFQ in its quant format via
//! [`LinearWeight`] (f32 / Q8 / MQ4 / MQ6) and runs the forward op-for-op against
//! `cpu.rs`, reusing `gemv_f32`/the dispatched gemv/`rmsnorm_f32`/`silu_mul_f32`
//! plus the custom CCA/EDA kernels in `hipfire-rdna/src/dispatch/zaya_cca.rs`.
//! Batch 1. `gpu_forward_serve` prefills the whole prompt and primes a
//! [`ZayaDecodeState`] (KV cache + conv ring + delayed value); `gpu_decode` then
//! advances one token at a time (O(1) per token). `gpu_forward_prefill` is a
//! per-block-trace variant used only by the `gpu_golden` validation example.

use crate::ZayaConfig;
use hipfire_dispatch::context::DispatchCtx;
use hipfire_dispatch::pipeline::{execute_steps, GemvInput, Step};
use hipfire_rdna::{DType, Gpu, GpuTensor, OwnedTensor};
use hipfire_runtime::calibration::{logsumexp, topk_logits};
use hipfire_runtime::hfq::{load_awq_scale, oq4_arch_load, oq8_arch_load, HfqFile};
use hipfire_runtime::kv::KvCache;
use hipfire_runtime::weights::WeightTensor;

/// quant_type byte → quantized linear `DType` (None ⇒ a plain precision handled
/// by `dequant_qt`). Matches the hipfire-quantize QuantType discriminants.
fn linear_dtype(qt: u8) -> Option<DType> {
    match qt {
        3 => Some(DType::Q8_0), // Q8F16
        6 => Some(DType::HFQ4G256),
        7 => Some(DType::HFQ4G128),
        13 => Some(DType::MQ4G256), // MagnumQuant 4-bit
        15 => Some(DType::MQ6G256), // MagnumQuant 6-bit (zaya experts under --format mq4)
        // BF16 (16) / F16 (1): upload verbatim (2 bytes/elem) instead of widening to
        // F32. The gather (embedding_lookup_bf16) and lm_head gemv (bf16×f32 portable)
        // convert to f32 in-kernel — bit-identical to the source, HALF the VRAM read.
        16 => Some(DType::BF16),
        1 => Some(DType::F16),
        _ => None,
    }
}

// ─── Opus-Quant (OQ4/OQ8) repack-on-load ─────────────────────────────────────
// The OQ formats are WMMA-gated (halo/gfx1151). Unlike the verbatim-upload quant
// formats above, the on-disk OQ block layout must be transformed to the kernel's
// combined buffer at load time. These helpers mirror `hipfire-arch-gemma3`'s
// `weights.rs` byte-for-byte so the layout cannot drift between arches.

/// Repack an OQ-family on-disk tensor to its kernel buffer + gpu_dtype, or `None`
/// for non-OQ quant_types (handled by `linear_dtype` verbatim upload / f32).
/// The OQ8 int8-activation codes (33=OQ+ W4A8, 35=OQ8 W8A8, 36=OQ+ compact) route
/// through the shared `oq8_arch_load`; the OQ4 pair (34/37) through the shared
/// `oq4_arch_load`. Both resolve to a generic iu8/iu4 GEMM dtype — no zaya-local
/// expansion, so a new OQ code lights up here for free.
fn oq_repack(qt: u8, data: &[u8], m: usize, k: usize) -> Option<(Vec<u8>, DType)> {
    oq8_arch_load(qt, data, m, k)
        .or_else(|| oq4_arch_load(qt, data, m, k).map(|(bytes, dt)| (bytes.into_owned(), dt)))
}

/// A `[out, in]` linear weight: plain f32 (bf16/f16 hfq dequant) or a quantized
/// `WeightTensor` (mq4/mq6/q8). One `gemv` entry dispatches both — mirrors
/// `hipfire_arch_nemotron::weight::LinearWeight`.
pub enum LinearWeight {
    F32(GpuTensor),
    Quant(Box<WeightTensor>),
}

impl LinearWeight {
    /// `out[m] = W · x[k]`. F32 → `gemv_f32`; Quant → the dispatched gemv
    /// (auto-rotates for MQ-family, plain for HFQ/Q8).
    pub fn gemv(&self, gpu: &mut Gpu, x: &GpuTensor, out: &GpuTensor) -> Result<(), String> {
        match self {
            LinearWeight::F32(w) => gpu
                .gemv_f32(w, x, out)
                .map_err(|e| format!("zaya f32 gemv: {e:?}")),
            // Source-precision (bf16/f16) weight × f32 activation gemv — the decode
            // lm_head path. Routes to the bf16/f16-weight×f32 gemv directly (the
            // generic dispatch mis-selects a WMMA bf16×bf16 path that needs a bf16
            // activation). Bit-identical to the source; half the VRAM read vs F32.
            LinearWeight::Quant(wt) if wt.gpu_dtype == DType::BF16 => {
                // `gemv_bf16_f32` is bf16×bf16 (F32 accumulate) — cast the f32
                // activation to bf16 first. Weight stays bf16 (source, lossless);
                // the bf16 activation is a negligible loss for the output logits.
                let xb = gpu
                    .alloc_tensor(&[wt.k], DType::BF16)
                    .map_err(|e| format!("zaya bf16 x: {e:?}"))?;
                gpu.cast_f32_to_bf16(x, &xb)
                    .map_err(|e| format!("zaya cast bf16: {e:?}"))?;
                let r = gpu
                    .gemv_bf16_f32(&wt.buf, &xb, out, wt.m, wt.k)
                    .map_err(|e| format!("zaya bf16 gemv: {e:?}"));
                let _ = gpu.free_tensor(xb);
                r
            }
            LinearWeight::Quant(wt) if wt.gpu_dtype == DType::F16 => gpu
                .gemv_f16_xf32(&wt.buf, x, out, wt.m, wt.k)
                .map_err(|e| format!("zaya f16 gemv: {e:?}")),
            LinearWeight::Quant(wt) => {
                let ctx = DispatchCtx::new(gpu);
                execute_steps(
                    gpu,
                    &ctx,
                    &[Step::Gemv {
                        w: &wt.dispatch_ref(),
                        input: GemvInput::Raw(x),
                        out,
                    }],
                )
                .map_err(|e| format!("zaya quant gemv: {e}"))
            }
        }
    }

    /// Opus Quant **W8A8** GEMV: consumes a shared, pre-quantized int8 activation
    /// (`xq`/`xs` from one `quantize_act_oq8` per rmsnorm) instead of self-rotating
    /// + f32-dequant per call. Only valid for Oq8G256 (planar `[int8 M*K | f32
    /// scales M*ng]`); `ws` is the scale plane at byte offset `m*k`.
    fn gemv_w8a8(
        &self,
        gpu: &mut Gpu,
        xq: &GpuTensor,
        xs: &GpuTensor,
        out: &GpuTensor,
    ) -> Result<(), String> {
        match self {
            LinearWeight::Quant(wt) if wt.gpu_dtype == DType::Oq8G256 => {
                let (m, k) = (wt.m, wt.k);
                let ng = k / 256;
                let ws = wt.buf.sub_offset(m * k, m * ng * 4);
                gpu.gemv_oq8_w8a8_grouped(&wt.buf, &ws, xq, xs, out, m, k, 256)
                    .map_err(|e| format!("zaya w8a8 gemv: {e:?}"))
            }
            _ => Err("zaya w8a8 gemv: expected Oq8G256 weight".to_string()),
        }
    }

    /// Planar Oq8G256 weight `[int8 M*K | f32 scales M*ng]` + `(m, k)`, for the
    /// fused multi-projection W8A16 GEMV. `None` for non-Oq8G256 weights.
    fn quant_mk(&self) -> Option<(&GpuTensor, usize, usize)> {
        match self {
            LinearWeight::Quant(wt) if wt.gpu_dtype == DType::Oq8G256 => {
                Some((&wt.buf, wt.m, wt.k))
            }
            _ => None,
        }
    }

    /// `(buf, m, k)` for any quantized weight (vs `quant_mk` which is Oq8-only).
    /// Used by the two-stage lm_head to reach the bf16 embed for coarse-quant build.
    fn wt_mk(&self) -> Option<(&GpuTensor, usize, usize)> {
        match self {
            LinearWeight::Quant(wt) => Some((&wt.buf, wt.m, wt.k)),
            LinearWeight::F32(_) => None,
        }
    }

    /// The backing GPU buffer (for activation-capture name mapping). During
    /// calibration every linear is `F32` (bf16 model), but the `Quant` arm is
    /// handled for completeness.
    fn buf(&self) -> &GpuTensor {
        match self {
            LinearWeight::F32(t) => t,
            LinearWeight::Quant(wt) => &wt.buf,
        }
    }

    /// Kernel dispatch dtype when quantized (`None` for f32 calibration weights).
    /// Used to decide whether the indexed on-device MoE decode path applies.
    fn quant_dtype(&self) -> Option<DType> {
        match self {
            LinearWeight::Quant(wt) => Some(wt.gpu_dtype),
            LinearWeight::F32(_) => None,
        }
    }

    /// Device address of the weight buffer as a `u64`, for the per-expert pointer
    /// tables the indexed MoE GEMV kernels dereference. `None` for f32 weights.
    fn quant_buf_ptr(&self) -> Option<u64> {
        match self {
            LinearWeight::Quant(wt) => Some(wt.buf.buf.as_ptr() as u64),
            LinearWeight::F32(_) => None,
        }
    }

    /// Whether this weight carries a per-channel AWQ sidecar (OQ `+`/`++`). The
    /// on-device indexed MoE path uses plain `rotate_x_mq` and does NOT apply a
    /// per-expert AWQ scale, so AWQ-bearing experts must take the AWQ-aware host
    /// decode path (`oq8_gemv_into` → `rotate_x_mq_awq`) instead.
    fn quant_has_awq(&self) -> bool {
        matches!(self, LinearWeight::Quant(wt) if wt.awq_scale.is_some())
    }

    /// Release the GPU storage.
    fn free(self, gpu: &mut Gpu) {
        match self {
            LinearWeight::F32(w) => {
                let _ = gpu.free_tensor(w);
            }
            LinearWeight::Quant(wt) => wt.free_all(gpu),
        }
    }

    /// Embedding-row lookup (the weight doubles as the tied embedding table).
    /// The quantizer stores `embed_tokens.weight` as Q8 for every quant format;
    /// reject anything else rather than silently mis-dequantizing.
    fn embed_lookup(
        &self,
        gpu: &mut Gpu,
        out: &GpuTensor,
        token: u32,
        dim: usize,
    ) -> Result<(), String> {
        match self {
            LinearWeight::F32(t) => gpu
                .embedding_lookup(t, out, token, dim)
                .map_err(|e| format!("zaya embed lookup: {e:?}")),
            LinearWeight::Quant(wt) => match wt.gpu_dtype {
                DType::Q8_0 => gpu
                    .embedding_lookup_q8(&wt.buf, out, token, dim)
                    .map_err(|e| format!("zaya q8 embed lookup: {e:?}")),
                // BF16/F16 source-precision embed (no F32 widening): gather converts
                // to f32 in-kernel, bit-identical to the source.
                DType::BF16 => gpu
                    .embedding_lookup_bf16(&wt.buf, out, token, dim)
                    .map_err(|e| format!("zaya bf16 embed lookup: {e:?}")),
                DType::F16 => gpu
                    .embedding_lookup_f16(&wt.buf, out, token, dim)
                    .map_err(|e| format!("zaya f16 embed lookup: {e:?}")),
                other => Err(format!(
                    "zaya embed: unsupported quantized embedding dtype {other:?}"
                )),
            },
        }
    }
}

/// Load a `[m, k]` linear as quantized (`WeightTensor`) when the hfq stores it
/// quantized, else an f32 upload.
fn load_linear(hfq: &HfqFile, gpu: &mut Gpu, name: &str) -> Result<LinearWeight, String> {
    let info = hfq
        .find_tensor_info(name)
        .ok_or_else(|| format!("zaya gpu: missing tensor {name:?}"))?;
    let shape: Vec<usize> = info.shape.iter().map(|&x| x as usize).collect();
    let (m, k) = (shape[0], shape.get(1).copied().unwrap_or(1));
    let qt = info.quant_type;
    let (_, data) = hfq
        .tensor_data_vec(name)
        .ok_or_else(|| format!("zaya gpu: no data for {name:?}"))?;
    // OQ-family (33/34/35/36/37) repacks on load; the verbatim formats upload as-is
    // (no clone — the large Q8 embedding uploads straight from `data`).
    let buf_dtype: Option<(GpuTensor, DType)> =
        if let Some((bytes, dtype)) = oq_repack(qt, &data, m, k) {
            let buf = gpu
                .upload_raw(&bytes, &[bytes.len()])
                .map_err(|e| format!("zaya upload {name}: {e:?}"))?;
            Some((buf, dtype))
        } else if let Some(dtype) = linear_dtype(qt) {
            let buf = gpu
                .upload_raw(&data, &[data.len()])
                .map_err(|e| format!("zaya upload {name}: {e:?}"))?;
            Some((buf, dtype))
        } else {
            None
        };
    if let Some((buf, dtype)) = buf_dtype {
        let mut wt = WeightTensor {
            buf,
            gpu_dtype: dtype,
            m,
            k,
            row_stride: 0,
            paro: None,
            awq_scale: None,
        };
        // The OQ `+`/`++` calibration writes a per-channel `<name>.awq_scale`
        // sidecar; the gemv (`dispatch_ref`) divides x by it before the rotation.
        if wt.gpu_dtype.supports_awq_sidecar() {
            wt.awq_scale = load_awq_scale(hfq, gpu, name, k);
        }
        Ok(LinearWeight::Quant(Box::new(wt)))
    } else {
        let f = dequant_qt(qt, &data)?;
        let g = gpu
            .upload_f32(&f, &[m, k])
            .map_err(|e| format!("zaya f32 {name}: {e:?}"))?;
        Ok(LinearWeight::F32(g))
    }
}

fn dequant_qt(qt: u8, bytes: &[u8]) -> Result<Vec<f32>, String> {
    match qt {
        1 => Ok(bytes
            .chunks_exact(2)
            .map(|c| {
                let h = u16::from_le_bytes([c[0], c[1]]);
                let s = (h >> 15) & 1;
                let e = (h >> 10) & 0x1f;
                let m = h & 0x3ff;
                let v = match e {
                    0 => (m as f32) * 2f32.powi(-24),
                    0x1f => {
                        if m == 0 {
                            f32::INFINITY
                        } else {
                            f32::NAN
                        }
                    }
                    _ => (1.0 + (m as f32) / 1024.0) * 2f32.powi(e as i32 - 15),
                };
                if s == 1 {
                    -v
                } else {
                    v
                }
            })
            .collect()),
        2 => Ok(bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()),
        16 => Ok(bytes
            .chunks_exact(2)
            .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
            .collect()),
        other => Err(format!("zaya gpu: unsupported quant_type {other}")),
    }
}

/// Upload an HFQ tensor (bf16/f16/f32) as an f32 GpuTensor with its stored shape.
fn up(hfq: &HfqFile, gpu: &mut Gpu, name: &str) -> Result<GpuTensor, String> {
    let info = hfq
        .find_tensor_info(name)
        .ok_or_else(|| format!("zaya gpu: missing tensor {name:?}"))?;
    let shape: Vec<usize> = info.shape.iter().map(|&x| x as usize).collect();
    let qt = info.quant_type;
    let (_, data) = hfq
        .tensor_data_vec(name)
        .ok_or_else(|| format!("zaya gpu: no data for {name:?}"))?;
    let f = dequant_qt(qt, &data)?;
    let shape = if shape.is_empty() {
        vec![f.len()]
    } else {
        shape
    };
    gpu.upload_f32(&f, &shape)
        .map_err(|e| format!("zaya gpu upload {name}: {e:?}"))
}

struct GpuExpert {
    gate_up: LinearWeight, // [2*moe_int, hidden]
    down: LinearWeight,    // [hidden, moe_int]
}

/// Device-side state for the indexed on-device MoE decode path. Present only
/// when every expert is `Oq8G256` (what oq8/oq8++ artifacts expand to). Lets
/// `gpu_decode` select and run the top-1 expert entirely on device, replacing
/// the per-block `download_f32(rlogits)` host readback.
struct ZayaMoeIndexed {
    /// `[2*n_route]` f32 storage of `n_route` u64 device pointers, one per
    /// expert's `gate_up` buffer. The last (`n_exp`) slot is the null expert and
    /// aliases expert 0 so the indexed GEMV can dereference it safely; its output
    /// is gated to zero by [`Gpu::zaya_router_select_f32`].
    gate_up_ptrs: GpuTensor,
    down_ptrs: GpuTensor,
    /// `[n_route]` f32 balancing biases uploaded for the on-device argmax.
    balancing_biases_dev: GpuTensor,
}

struct GpuLayer {
    input_ln: GpuTensor,
    post_attn_ln: GpuTensor,
    q_proj: LinearWeight,
    k_proj: LinearWeight,
    v_cur: LinearWeight,
    v_del: LinearWeight,
    conv_dw_w: GpuTensor,
    conv_dw_b: GpuTensor,
    conv_gr_w: GpuTensor,
    conv_gr_b: GpuTensor,
    qk_temp: GpuTensor,
    o_proj: LinearWeight,
    // router
    down_proj_w: LinearWeight,
    down_proj_b: GpuTensor,
    router_states_scale: Option<GpuTensor>,
    rnorm_w: GpuTensor,
    fc1_w: LinearWeight,
    fc1_b: GpuTensor,
    fc2_w: LinearWeight,
    fc2_b: GpuTensor,
    out_proj_w: LinearWeight,
    balancing_biases: Vec<f32>,
    experts: Vec<GpuExpert>,
    /// On-device indexed MoE decode state (`Some` when experts are Oq8G256).
    moe_indexed: Option<ZayaMoeIndexed>,
    // residual scales (each: hidden_states_scale, hidden_states_bias, residual_scale, residual_bias)
    pa_rs: [GpuTensor; 4],
    pm_rs: [GpuTensor; 4],
}

impl GpuExpert {
    fn free(self, gpu: &mut Gpu) {
        self.gate_up.free(gpu);
        self.down.free(gpu);
    }
}

impl GpuLayer {
    fn free(self, gpu: &mut Gpu) {
        for lw in [
            self.q_proj,
            self.k_proj,
            self.v_cur,
            self.v_del,
            self.o_proj,
            self.down_proj_w,
            self.fc1_w,
            self.fc2_w,
            self.out_proj_w,
        ] {
            lw.free(gpu);
        }
        for e in self.experts {
            e.free(gpu);
        }
        let mut ts = vec![
            self.input_ln,
            self.post_attn_ln,
            self.conv_dw_w,
            self.conv_dw_b,
            self.conv_gr_w,
            self.conv_gr_b,
            self.qk_temp,
            self.down_proj_b,
            self.rnorm_w,
            self.fc1_b,
            self.fc2_b,
        ];
        if let Some(idx) = self.moe_indexed {
            ts.push(idx.gate_up_ptrs);
            ts.push(idx.down_ptrs);
            ts.push(idx.balancing_biases_dev);
        }
        ts.extend(self.pa_rs);
        ts.extend(self.pm_rs);
        ts.extend(self.router_states_scale);
        for t in ts {
            let _ = gpu.free_tensor(t);
        }
    }
}

/// All ZAYA weights resident on the GPU. Big linears (projections, experts,
/// tied lm_head/embed) keep their hfq quant format via [`LinearWeight`]; small
/// protected tensors (norms, biases, conv filters, scales) are f32.
pub struct ZayaGpuWeights {
    embed: LinearWeight,
    in_scale: GpuTensor,
    in_bias: GpuTensor,
    layers: Vec<GpuLayer>,
    norm: GpuTensor,
}

impl ZayaGpuWeights {
    /// Release every GPU buffer (weights). Consumes self.
    pub fn free(self, gpu: &mut Gpu) {
        self.embed.free(gpu);
        for l in self.layers {
            l.free(gpu);
        }
        for t in [self.in_scale, self.in_bias, self.norm] {
            let _ = gpu.free_tensor(t);
        }
    }
}

impl ZayaGpuWeights {
    pub fn load(hfq: &HfqFile, gpu: &mut Gpu, cfg: &ZayaConfig) -> Result<Self, String> {
        let embed = load_linear(hfq, gpu, "model.embed_tokens.weight")?;
        let in_scale = up(hfq, gpu, "model.input_hidden_states_scale")?;
        let in_bias = up(hfq, gpu, "model.input_hidden_states_bias")?;
        let norm = up(hfq, gpu, "model.norm.weight")?;
        let mut layers = Vec::with_capacity(cfg.num_blocks);
        for l in 0..cfg.num_blocks {
            hipfire_runtime::load_progress::report(l as u32 + 1, cfg.num_blocks as u32, "weights");
            let p = format!("model.layers.{l}");
            let qkv = format!("{p}.self_attn.qkv_proj");
            let gate = format!("{p}.mlp.gate");
            let rmlp = format!("{gate}.router_mlp");
            let rs = |gpu: &mut Gpu, base: &str| -> Result<[GpuTensor; 4], String> {
                Ok([
                    up(hfq, gpu, &format!("{base}.hidden_states_scale"))?,
                    up(hfq, gpu, &format!("{base}.hidden_states_bias"))?,
                    up(hfq, gpu, &format!("{base}.residual_scale"))?,
                    up(hfq, gpu, &format!("{base}.residual_bias"))?,
                ])
            };
            let mut experts = Vec::with_capacity(cfg.moe.num_experts);
            for e in 0..cfg.moe.num_experts {
                experts.push(GpuExpert {
                    gate_up: load_linear(
                        hfq,
                        gpu,
                        &format!("{p}.mlp.experts.{e}.gate_up_proj.weight"),
                    )?,
                    down: load_linear(hfq, gpu, &format!("{p}.mlp.experts.{e}.down_proj.weight"))?,
                });
            }
            let router_states_scale = if l == 0 {
                None
            } else {
                Some(up(hfq, gpu, &format!("{gate}.router_states_scale"))?)
            };
            // balancing_biases stays on host (top-1 done host-side).
            let (_, bb_bytes) = hfq
                .tensor_data_vec(&format!("{gate}.balancing_biases"))
                .ok_or("zaya gpu: missing balancing_biases")?;
            let bb_qt = hfq
                .find_tensor_info(&format!("{gate}.balancing_biases"))
                .unwrap()
                .quant_type;
            let balancing_biases = dequant_qt(bb_qt, &bb_bytes)?;

            // Build the on-device indexed MoE decode state when every expert is
            // Oq8G256 (oq8/oq8++ artifacts). The per-expert device pointer tables
            // + device balancing biases let `gpu_decode` route + run the top-1
            // expert without the per-block host readback. Other quant formats
            // (f32 calibration, OQ4, …) keep the host reference decode path.
            let n_route = cfg.moe.num_router_experts();
            // On-device path requires plain Oq8G256 experts with NO per-expert AWQ
            // sidecar — the indexed planar kernels use `rotate_x_mq` and cannot apply
            // a per-expert AWQ scale, so oq8+/oq8++ experts fall back to the host path.
            let all_oq8 = experts.iter().all(|e| {
                e.gate_up.quant_dtype() == Some(DType::Oq8G256)
                    && e.down.quant_dtype() == Some(DType::Oq8G256)
                    && !e.gate_up.quant_has_awq()
                    && !e.down.quant_has_awq()
            });
            let moe_indexed = if all_oq8 {
                // n_route entries: experts 0..n_exp, then a null slot aliasing
                // expert 0 (safe deref; gated to 0 weight at select time).
                let mut gu: Vec<u64> = experts
                    .iter()
                    .map(|e| e.gate_up.quant_buf_ptr().unwrap())
                    .collect();
                let mut dn: Vec<u64> = experts
                    .iter()
                    .map(|e| e.down.quant_buf_ptr().unwrap())
                    .collect();
                gu.push(gu[0]);
                dn.push(dn[0]);
                debug_assert_eq!(gu.len(), n_route);
                let upload_table = |gpu: &mut Gpu, v: &[u64]| -> Result<GpuTensor, String> {
                    let bytes: Vec<u8> = v.iter().flat_map(|p| p.to_ne_bytes()).collect();
                    // 8 B per pointer = 2 f32 slots; kernel reads them via u64 cast.
                    let t = gpu
                        .alloc_tensor(&[2 * v.len()], DType::F32)
                        .map_err(|e| format!("zaya moe ptr table alloc: {e:?}"))?;
                    gpu.hip
                        .memcpy_htod(&t.buf, &bytes)
                        .map_err(|e| format!("zaya moe ptr table htod: {e:?}"))?;
                    Ok(t)
                };
                let gate_up_ptrs = upload_table(gpu, &gu)?;
                let down_ptrs = upload_table(gpu, &dn)?;
                let balancing_biases_dev = gpu
                    .upload_f32(&balancing_biases, &[balancing_biases.len()])
                    .map_err(|e| format!("zaya balancing_biases upload: {e:?}"))?;
                Some(ZayaMoeIndexed {
                    gate_up_ptrs,
                    down_ptrs,
                    balancing_biases_dev,
                })
            } else {
                None
            };

            layers.push(GpuLayer {
                input_ln: up(hfq, gpu, &format!("{p}.input_layernorm.weight"))?,
                post_attn_ln: up(hfq, gpu, &format!("{p}.post_attention_layernorm.weight"))?,
                q_proj: load_linear(hfq, gpu, &format!("{qkv}.q_proj.weight"))?,
                k_proj: load_linear(hfq, gpu, &format!("{qkv}.k_proj.weight"))?,
                v_cur: load_linear(hfq, gpu, &format!("{qkv}.v_proj_current.weight"))?,
                v_del: load_linear(hfq, gpu, &format!("{qkv}.v_proj_delayed.weight"))?,
                conv_dw_w: up(hfq, gpu, &format!("{qkv}.conv_qk_depthwise.weight"))?,
                conv_dw_b: up(hfq, gpu, &format!("{qkv}.conv_qk_depthwise.bias"))?,
                conv_gr_w: up(hfq, gpu, &format!("{qkv}.conv_qk_grouped.weight"))?,
                conv_gr_b: up(hfq, gpu, &format!("{qkv}.conv_qk_grouped.bias"))?,
                qk_temp: up(hfq, gpu, &format!("{p}.self_attn.qk_norm.temp"))?,
                o_proj: load_linear(hfq, gpu, &format!("{p}.self_attn.o_proj.weight"))?,
                down_proj_w: load_linear(hfq, gpu, &format!("{gate}.down_proj.weight"))?,
                down_proj_b: up(hfq, gpu, &format!("{gate}.down_proj.bias"))?,
                router_states_scale,
                rnorm_w: up(hfq, gpu, &format!("{rmlp}.norm.weight"))?,
                fc1_w: load_linear(hfq, gpu, &format!("{rmlp}.fc1.weight"))?,
                fc1_b: up(hfq, gpu, &format!("{rmlp}.fc1.bias"))?,
                fc2_w: load_linear(hfq, gpu, &format!("{rmlp}.fc2.weight"))?,
                fc2_b: up(hfq, gpu, &format!("{rmlp}.fc2.bias"))?,
                out_proj_w: load_linear(hfq, gpu, &format!("{rmlp}.out_proj.weight"))?,
                balancing_biases,
                experts,
                moe_indexed,
                pa_rs: rs(gpu, &format!("{p}.post_attention_residual_scale"))?,
                pm_rs: rs(gpu, &format!("{p}.post_mlp_residual_scale"))?,
            });
        }
        Ok(Self {
            embed,
            in_scale,
            in_bias,
            layers,
            norm,
        })
    }
}

/// Per-block hidden states + logits, downloaded to host for golden comparison.
pub struct GpuTrace {
    pub embed_scaled: Vec<f32>,
    pub block: Vec<Vec<f32>>,
    pub final_norm: Vec<f32>,
    pub logits: Vec<f32>,
    pub seq: usize,
}

/// gemv over a sequence: `y[s,m] = x[s,k] @ w[m,k]^T` (per-token gemv loop,
/// f32 or quantized via [`LinearWeight`]).
fn gemv_seq(
    gpu: &mut Gpu,
    w: &LinearWeight,
    x: &GpuTensor,
    y: &GpuTensor,
    s: usize,
    m: usize,
    k: usize,
) -> Result<(), String> {
    for t in 0..s {
        let xt = x.sub_offset(t * k, k);
        let yt = y.sub_offset(t * m, m);
        w.gemv(gpu, &xt, &yt)?;
    }
    Ok(())
}

fn z(gpu: &mut Gpu, n: usize) -> Result<GpuTensor, String> {
    gpu.zeros(&[n], DType::F32)
        .map_err(|e| format!("zaya alloc: {e:?}"))
}

/// RAII (`OwnedTensor`) analog of `z`: a uniquely-pooled 1D scratch buffer that
/// returns itself to the deferred-free mailbox on drop (see `OwnedTensor`).
fn zo(gpu: &mut Gpu, n: usize) -> Result<OwnedTensor, String> {
    gpu.zeros_owned(&[n], DType::F32)
        .map_err(|e| format!("zaya alloc: {e:?}"))
}

/// RAII analog of `z` for 2D `[rows, d]` scratch — required for tensors fed to
/// `rmsnorm_f32`, which reads `batch = shape[0]` and `n = shape.last()`.
fn z2o(gpu: &mut Gpu, rows: usize, d: usize) -> Result<OwnedTensor, String> {
    gpu.zeros_owned(&[rows, d], DType::F32)
        .map_err(|e| format!("zaya alloc: {e:?}"))
}

/// Per-layer decode state: KV cache (post-rope key, composed value), the two
/// last raw qk-stream columns (conv ring), and the previous token's delayed-value
/// projection. Carried across decode steps so each step processes one token.
/// KVarN KV cache (opt-in via `HIPFIRE_ZAYA_KVARN=2|4|8`): variance-normalized
/// low-bit K (block-tiled records + f32 recent-window) + Q8_0 V, consumed by the
/// fused `kvarn_attend`. Replaces the f32 `k_cache`/`v_cache` when present; both
/// prefill (`n=s`) and decode (`n=1`) route through the same primitive.
struct ZayaKvarn {
    cache: KvCache,
    tiles: GpuTensor,
    flash_partials: GpuTensor,
    /// Prefill positions `[max_seq]` (F32-typed, i32 bytes); decode uses `pos_buf`.
    positions: GpuTensor,
    bits: usize,
}

pub struct ZayaDecodeState {
    pub pos: usize,
    max_seq: usize,
    // Empty when `kvarn` is Some — the KVarN cache replaces the f32 rings.
    k_cache: Vec<GpuTensor>, // [layer] [max_seq * kvdim]
    v_cache: Vec<GpuTensor>,
    /// Opt-in low-bit KV cache. `None` ⇒ the f32 `attention_f32` path above.
    kvarn: Option<ZayaKvarn>,
    conv_ring: Vec<GpuTensor>, // [layer] [conv_ch * pad]
    delayed_v: Vec<GpuTensor>, // [layer] [v_half]
    /// Device-resident decode position (4 bytes, `seq_len = pos_buf[0] + 1`) for
    /// the parallel flash-decode `attention_f32` + `kv_cache_write` path.
    pos_buf: hip_bridge::DeviceBuffer,
    /// Persistent residual-stream buffer `[hidden]` produced by the per-token
    /// prologue (embed + input affine) and evolved by the captured layer body. A
    /// stable address is required so the hipGraph body can bake it once and read
    /// it on every replay. Lazily allocated on first decode.
    decode_hidden: Option<GpuTensor>,
    // hipGraph MVP (HIPFIRE_ZAYA_GRAPH): warm the pool on step 1, capture the
    // decode body on step 2, replay it on step 3+. THROWAWAY timing harness —
    // the captured graph bakes token/pos, so replay output is garbage; it only
    // measures the launch-overhead payoff before the device-`pos` retrofit.
    graph_warmed: bool,
    graph_ready: bool,
    graph_disabled: bool,
    /// Optional F16 copy of the F32 lm_head (tied embed), built once for the
    /// untied output projection (`HIPFIRE_ZAYA_F16_LMHEAD`). The input gather
    /// keeps the F32 table; only the 2.15 GB output read is halved to ~1.07 GB.
    lmhead_f16: Option<GpuTensor>,
    /// Two-stage lm_head coarse scorer (`HIPFIRE_ZAYA_LMHEAD_SHORTLIST`): row-wise
    /// L2-normalized Q4 copy of the embed (optionally projected H→r), built once.
    /// The coarse pass shortlists candidates; the fine pass rescores them at bf16.
    lmhead_coarse: Option<LmheadCoarse>,
}

/// Coarse lm_head scorer: `q4` [V, kdim/2] (row-wise-normalized 3σ-clipped Q4),
/// `scales` [V] (= L2 norm × global unit scale), optional random projection `proj`
/// [kdim, H] (None ⇒ full-H coarse, kdim = H), `kdim` = r (reduced) or H.
struct LmheadCoarse {
    q4: GpuTensor,
    scales: GpuTensor,
    proj: Option<GpuTensor>,
    kdim: usize,
    bits: usize, // 2 or 4
    // Stage-3 low-rank residual correction (HIPFIRE_ZAYA_SHORTLIST_CORRECT=r): the
    // coarse score gets += A[V,r]·(B[r,H]·h), recovering the low-rank part of the
    // W − Qrecon residual. None when disabled.
    corr_a: Option<GpuTensor>,
    corr_b: Option<GpuTensor>,
    corr_r: usize,
}

impl ZayaDecodeState {
    pub fn new(gpu: &mut Gpu, cfg: &ZayaConfig, max_seq: usize) -> Result<Self, String> {
        let kvdim = cfg.attn.num_kv_heads * cfg.attn.head_dim;
        let conv_ch = (cfg.attn.num_heads + cfg.attn.num_kv_heads) * cfg.attn.head_dim;
        let pad = (cfg.attn.conv_depthwise_kernel - 1) + (cfg.attn.conv_grouped_kernel - 1);
        let v_half = kvdim / 2;
        let nkv = cfg.attn.num_kv_heads;
        let nq = cfg.attn.num_heads;
        let hd = cfg.attn.head_dim;
        // Opt-in low-bit KV: HIPFIRE_ZAYA_KVARN=2|4|8. head_dim=128 → no rotation
        // (FWHT is head_dim=256-only). Falls back to the f32 rings when unset.
        let kvarn_bits = std::env::var("HIPFIRE_ZAYA_KVARN")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|b| matches!(b, 2 | 4 | 8));
        let mut k_cache = Vec::with_capacity(cfg.num_blocks);
        let mut v_cache = Vec::with_capacity(cfg.num_blocks);
        let mut conv_ring = Vec::with_capacity(cfg.num_blocks);
        let mut delayed_v = Vec::with_capacity(cfg.num_blocks);
        for _ in 0..cfg.num_blocks {
            // The f32 KV rings are unused (and unallocated) under KVarN.
            if kvarn_bits.is_none() {
                k_cache.push(z(gpu, max_seq * kvdim)?);
                v_cache.push(z(gpu, max_seq * kvdim)?);
            }
            conv_ring.push(z(gpu, conv_ch * pad)?);
            delayed_v.push(z(gpu, v_half)?);
        }
        let kvarn = match kvarn_bits {
            Some(bits) => {
                let cache = KvCache::new_gpu_kvarn(gpu, cfg.num_blocks, nkv, hd, max_seq, bits)
                    .map_err(|e| format!("zaya kvarn cache: {e:?}"))?;
                let tiles = gpu
                    .alloc_tensor(&[nkv * hd * KvCache::KVARN_GROUP], DType::F32)
                    .map_err(|e| format!("zaya kvarn tiles: {e:?}"))?;
                let max_tiles = max_seq.div_ceil(KvCache::KVARN_GROUP);
                // 16 batch slots cover prefill fan-out; the flash kernel sub-batches
                // for larger prompts, and decode only ever needs one.
                let flash_partials = gpu
                    .alloc_tensor(&[16 * nq * max_tiles * (2 + hd)], DType::F32)
                    .map_err(|e| format!("zaya kvarn partials: {e:?}"))?;
                let positions = gpu
                    .alloc_tensor(&[max_seq], DType::F32)
                    .map_err(|e| format!("zaya kvarn positions: {e:?}"))?;
                Some(ZayaKvarn {
                    cache,
                    tiles,
                    flash_partials,
                    positions,
                    bits,
                })
            }
            None => None,
        };
        let pos_buf = gpu
            .hip
            .malloc(4)
            .map_err(|e| format!("zaya pos_buf alloc: {e:?}"))?;
        Ok(Self {
            pos: 0,
            max_seq,
            k_cache,
            v_cache,
            kvarn,
            conv_ring,
            delayed_v,
            pos_buf,
            decode_hidden: None,
            graph_warmed: false,
            graph_ready: false,
            graph_disabled: false,
            lmhead_f16: None,
            lmhead_coarse: None,
        })
    }

    pub fn reset(&mut self) {
        self.pos = 0;
        self.graph_warmed = false;
        self.graph_ready = false;
        self.graph_disabled = false;
    }

    /// Release the KV cache + conv-ring + delayed-value buffers. Consumes self.
    pub fn free(self, gpu: &mut Gpu) {
        for v in [self.k_cache, self.v_cache, self.conv_ring, self.delayed_v] {
            for t in v {
                let _ = gpu.free_tensor(t);
            }
        }
        if let Some(kv) = self.kvarn {
            let _ = gpu.free_tensor(kv.tiles);
            let _ = gpu.free_tensor(kv.flash_partials);
            let _ = gpu.free_tensor(kv.positions);
            kv.cache.free_gpu(gpu);
        }
        if let Some(hidden) = self.decode_hidden {
            let _ = gpu.free_tensor(hidden);
        }
        if let Some(t) = self.lmhead_f16 {
            let _ = gpu.free_tensor(t);
        }
        if let Some(c) = self.lmhead_coarse {
            let _ = gpu.free_tensor(c.q4);
            let _ = gpu.free_tensor(c.scales);
            if let Some(p) = c.proj {
                let _ = gpu.free_tensor(p);
            }
            if let Some(a) = c.corr_a {
                let _ = gpu.free_tensor(a);
            }
            if let Some(b) = c.corr_b {
                let _ = gpu.free_tensor(b);
            }
        }
    }
}

/// Serving forward: prefill the whole prompt, write the **last position's** logits
/// into `logits_out` (`[vocab]`), and **prime `state`** (KV cache + conv ring +
/// delayed value for every layer) so subsequent tokens go through `gpu_decode`.
#[allow(clippy::needless_range_loop)]
pub fn gpu_forward_serve(
    gpu: &mut Gpu,
    w: &ZayaGpuWeights,
    cfg: &ZayaConfig,
    ids: &[u32],
    state: &mut ZayaDecodeState,
    logits_out: &GpuTensor,
) -> Result<(), String> {
    if ids.len() > state.max_seq {
        return Err(format!(
            "zaya prefill: prompt {} exceeds max_seq {}",
            ids.len(),
            state.max_seq
        ));
    }
    let s = ids.len();
    let h = cfg.hidden_size;
    let a = &cfg.attn;
    let (nq, nkv, hd) = (a.num_heads, a.num_kv_heads, a.head_dim);
    let q_dim = nq * hd;
    let k_dim = nkv * hd;
    let v_half = k_dim / 2;
    let conv_ch = q_dim + k_dim;
    let rh = cfg.moe.router_hidden_size;
    let n_route = cfg.moe.num_router_experts();
    let n_exp = cfg.moe.num_experts;
    let moe_int = cfg.moe.moe_intermediate_size;
    let eps = cfg.rms_norm_eps;
    let pad = (a.conv_depthwise_kernel - 1) + (a.conv_grouped_kernel - 1);
    let dw_len = s + pad - a.conv_depthwise_kernel + 1;
    let attn_scale = 1.0 / (hd as f32).sqrt();
    let l2_scale = (hd as f32).sqrt();

    let hidden = z2o(gpu, s, h)?;
    for t in 0..s {
        let row = hidden.sub_offset(t * h, h);
        w.embed.embed_lookup(gpu, &row, ids[t], h)?;
    }
    // global input residual affine, in place.
    gpu.zaya_affine_input_f32(&hidden, &hidden, &w.in_scale, &w.in_bias, h, s * h)
        .map_err(|e| format!("{e:?}"))?;

    // KVarN prefill positions [0, s): identical for every layer, uploaded once.
    if let Some(kv) = state.kvarn.as_ref() {
        let pos_host: Vec<i32> = (0..s as i32).collect();
        let pos_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(pos_host.as_ptr() as *const u8, s * 4) };
        gpu.hip
            .memcpy_htod(&kv.positions.buf, pos_bytes)
            .map_err(|e| format!("zaya kvarn prefill positions: {e:?}"))?;
    }

    let normed = z2o(gpu, s, h)?;
    let q = zo(gpu, s * q_dim)?;
    let k = zo(gpu, s * k_dim)?;
    let vcur = zo(gpu, s * v_half)?;
    let vdel = zo(gpu, s * v_half)?;
    let qres = zo(gpu, s * nq * hd)?;
    let kres = zo(gpu, s * nkv * hd)?;
    let stream = zo(gpu, conv_ch * (s + pad))?;
    let dw = zo(gpu, conv_ch * dw_len)?;
    let gw = zo(gpu, conv_ch * s)?;
    let query = zo(gpu, s * nq * hd)?;
    let key = zo(gpu, s * nkv * hd)?;
    let value = zo(gpu, s * nkv * hd)?;
    let ctx = zo(gpu, s * q_dim)?;
    let attn_out = zo(gpu, s * h)?;
    let g_res2 = z2o(gpu, s, h)?;
    let rhid = z2o(gpu, s, rh)?;
    let rnormed = z2o(gpu, s, rh)?;
    let a1 = z2o(gpu, s, rh)?;
    let a2 = z2o(gpu, s, rh)?;
    let rlogits = zo(gpu, s * n_route)?;
    let moe_out = zo(gpu, s * h)?;
    let gate_up = zo(gpu, 2 * moe_int)?;
    let act = zo(gpu, moe_int)?;
    let down_t = zo(gpu, h)?;
    let router_state = zo(gpu, s * rh)?;
    let fnorm = z2o(gpu, s, h)?;

    for (li, lw) in w.layers.iter().enumerate() {
        gpu.rmsnorm_f32(&hidden, &lw.input_ln, &normed, eps)
            .map_err(|e| format!("{e:?}"))?;
        gemv_seq(gpu, &lw.q_proj, &normed, &q, s, q_dim, h)?;
        gemv_seq(gpu, &lw.k_proj, &normed, &k, s, k_dim, h)?;
        gemv_seq(gpu, &lw.v_cur, &normed, &vcur, s, v_half, h)?;
        gemv_seq(gpu, &lw.v_del, &normed, &vdel, s, v_half, h)?;
        gpu.zaya_qk_residual_f32(&qres, &kres, &q, &k, s, nq, nkv, hd, 0)
            .map_err(|e| format!("{e:?}"))?;
        gpu.zaya_qk_residual_f32(&qres, &kres, &q, &k, s, nq, nkv, hd, 1)
            .map_err(|e| format!("{e:?}"))?;
        gpu.fill_f32(&stream, 0.0).map_err(|e| format!("{e:?}"))?;
        gpu.zaya_qk_stream_f32(&stream, &q, &k, s, q_dim, k_dim, pad)
            .map_err(|e| format!("{e:?}"))?;
        gpu.zaya_conv1d_valid_f32(
            &dw,
            &stream,
            &lw.conv_dw_w,
            &lw.conv_dw_b,
            conv_ch,
            conv_ch,
            a.conv_depthwise_kernel,
            s + pad,
            dw_len,
        )
        .map_err(|e| format!("{e:?}"))?;
        gpu.zaya_conv1d_valid_f32(
            &gw,
            &dw,
            &lw.conv_gr_w,
            &lw.conv_gr_b,
            conv_ch,
            nq + nkv,
            a.conv_grouped_kernel,
            dw_len,
            s,
        )
        .map_err(|e| format!("{e:?}"))?;
        gpu.zaya_add_conv_residual_f32(&query, &gw, &qres, s, nq, hd, q_dim, 0)
            .map_err(|e| format!("{e:?}"))?;
        gpu.zaya_add_conv_residual_f32(&key, &gw, &kres, s, nkv, hd, q_dim, 1)
            .map_err(|e| format!("{e:?}"))?;
        gpu.zaya_value_compose_f32(&value, &vcur, &vdel, s, nkv, hd)
            .map_err(|e| format!("{e:?}"))?;
        gpu.zaya_qk_l2norm_temp_f32(&query, None, s, nq, hd, l2_scale, f32::EPSILON)
            .map_err(|e| format!("{e:?}"))?;
        gpu.zaya_qk_l2norm_temp_f32(&key, Some(&lw.qk_temp), s, nkv, hd, l2_scale, f32::EPSILON)
            .map_err(|e| format!("{e:?}"))?;
        gpu.zaya_rope_partial_f32(&query, s, nq, hd, a.n_rot, a.rope_theta, 0)
            .map_err(|e| format!("{e:?}"))?;
        gpu.zaya_rope_partial_f32(&key, s, nkv, hd, a.n_rot, a.rope_theta, 0)
            .map_err(|e| format!("{e:?}"))?;
        if let Some(kv) = state.kvarn.as_ref() {
            // Fused KVarN prefill (n=s): quantize+store the whole K/V run and
            // flash-attend into `ctx` in one call. Positions [0, s) uploaded above.
            gpu.kvarn_attend(
                &kv.cache.k_gpu[li],
                &kv.cache.k_window[li],
                &kv.cache.v_gpu[li],
                &query,
                &key,
                &value,
                &kv.positions,
                &ctx,
                &kv.flash_partials,
                &kv.tiles,
                s,
                0,
                nq,
                nkv,
                hd,
                kv.cache.physical_cap,
                None,
                0,
                0,
                kv.bits,
            )
            .map_err(|e| format!("{e:?}"))?;
        } else {
            gpu.zaya_gqa_attn_f32(&ctx, &query, &key, &value, s, nq, nkv, hd, attn_scale)
                .map_err(|e| format!("{e:?}"))?;
        }
        gemv_seq(gpu, &lw.o_proj, &ctx, &attn_out, s, h, q_dim)?;
        // Prime decode state for this layer: KV (post-rope key / composed value),
        // conv ring (last `pad` raw qk-stream columns), delayed value (last token).
        // Under KVarN the KV was already written inside `kvarn_attend` above.
        if state.kvarn.is_none() {
            let kvdim = k_dim;
            gpu.zaya_write_at_f32(&state.k_cache[li], &key, 0, s * kvdim)
                .map_err(|e| format!("{e:?}"))?;
            gpu.zaya_write_at_f32(&state.v_cache[li], &value, 0, s * kvdim)
                .map_err(|e| format!("{e:?}"))?;
        }
        gpu.zaya_strided_copy_f32(&state.conv_ring[li], &stream, conv_ch, pad, s + pad, s, pad)
            .map_err(|e| format!("{e:?}"))?;
        let vdel_last = vdel.sub_offset((s - 1) * v_half, v_half);
        gpu.zaya_write_at_f32(&state.delayed_v[li], &vdel_last, 0, v_half)
            .map_err(|e| format!("{e:?}"))?;
        // residual stays on-device: `hidden` is still the block input here.
        gpu.zaya_affine_residual_f32(
            &g_res2,
            &attn_out,
            &hidden,
            &lw.pa_rs[0],
            &lw.pa_rs[1],
            &lw.pa_rs[2],
            &lw.pa_rs[3],
            h,
            s * h,
        )
        .map_err(|e| format!("{e:?}"))?;
        gpu.rmsnorm_f32(&g_res2, &lw.post_attn_ln, &normed, eps)
            .map_err(|e| format!("{e:?}"))?;
        // MoE
        gemv_seq(gpu, &lw.down_proj_w, &normed, &rhid, s, rh, h)?;
        gpu.zaya_bias_add_f32(&rhid, &lw.down_proj_b, rh, s * rh)
            .map_err(|e| format!("{e:?}"))?;
        if li != 0 {
            if let Some(scale) = lw.router_states_scale.as_ref() {
                gpu.zaya_eda_add_f32(&rhid, &router_state, scale, rh, s * rh)
                    .map_err(|e| format!("{e:?}"))?;
            }
        }
        // save next router state via an on-device copy (no host round-trip).
        gpu.zaya_write_at_f32(&router_state, &rhid, 0, s * rh)
            .map_err(|e| format!("{e:?}"))?;
        gpu.rmsnorm_f32(&rhid, &lw.rnorm_w, &rnormed, eps)
            .map_err(|e| format!("{e:?}"))?;
        gemv_seq(gpu, &lw.fc1_w, &rnormed, &a1, s, rh, rh)?;
        gpu.zaya_bias_add_f32(&a1, &lw.fc1_b, rh, s * rh)
            .map_err(|e| format!("{e:?}"))?;
        gpu.zaya_gelu_exact_f32(&a1, s * rh)
            .map_err(|e| format!("{e:?}"))?;
        gemv_seq(gpu, &lw.fc2_w, &a1, &a2, s, rh, rh)?;
        gpu.zaya_bias_add_f32(&a2, &lw.fc2_b, rh, s * rh)
            .map_err(|e| format!("{e:?}"))?;
        gpu.zaya_gelu_exact_f32(&a2, s * rh)
            .map_err(|e| format!("{e:?}"))?;
        gemv_seq(gpu, &lw.out_proj_w, &a2, &rlogits, s, n_route, rh)?;
        let logit_host = gpu.download_f32(&rlogits).map_err(|e| format!("{e:?}"))?;
        gpu.fill_f32(&moe_out, 0.0).map_err(|e| format!("{e:?}"))?;
        for t in 0..s {
            let row = &logit_host[t * n_route..(t + 1) * n_route];
            let maxv = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut probs = vec![0f32; n_route];
            let mut denom = 0f32;
            for e in 0..n_route {
                probs[e] = (row[e] - maxv).exp();
                denom += probs[e];
            }
            for p in probs.iter_mut() {
                *p /= denom;
            }
            let mut best = 0usize;
            let mut bestv = f32::NEG_INFINITY;
            for e in 0..n_route {
                let v = probs[e] + lw.balancing_biases[e];
                if v > bestv {
                    bestv = v;
                    best = e;
                }
            }
            if best == n_exp {
                continue;
            }
            let weight = probs[best];
            let xt = normed.sub_offset(t * h, h);
            let ex = &lw.experts[best];
            ex.gate_up.gemv(gpu, &xt, &gate_up)?;
            let g = gate_up.sub_offset(0, moe_int);
            let u = gate_up.sub_offset(moe_int, moe_int);
            gpu.silu_mul_f32(&g, &u, &act)
                .map_err(|e| format!("{e:?}"))?;
            ex.down.gemv(gpu, &act, &down_t)?;
            let ot = moe_out.sub_offset(t * h, h);
            gpu.scaled_add_inplace_cpu_scalar_f32(&ot, &down_t, weight)
                .map_err(|e| format!("{e:?}"))?;
        }
        gpu.zaya_affine_residual_f32(
            &hidden,
            &moe_out,
            &g_res2,
            &lw.pm_rs[0],
            &lw.pm_rs[1],
            &lw.pm_rs[2],
            &lw.pm_rs[3],
            h,
            s * h,
        )
        .map_err(|e| format!("{e:?}"))?;
        gpu.reclaim_pending();
    }

    // final norm on the last row → tied lm_head → logits_out [vocab]
    gpu.rmsnorm_f32(&hidden, &w.norm, &fnorm, eps)
        .map_err(|e| format!("{e:?}"))?;
    let last = fnorm.sub_offset((s - 1) * h, h);
    w.embed
        .gemv(gpu, &last, logits_out)
        .map_err(|e| format!("zaya lm_head: {e}"))?;
    state.pos = s;
    gpu.reclaim_pending();
    Ok(())
}

/// Map each dense linear's GPU buffer address → its canonical HFQ tensor name
/// (sans `.weight`), for activation capture during calibration. Routed experts
/// are intentionally excluded (LDLQ targets the dense projections; experts can
/// be captured imatrix-only in a later pass). The tied `embed` doubles as the
/// `lm_head`, keyed by `model.embed_tokens` (the name the quantizer sees).
pub fn build_capture_names(w: &ZayaGpuWeights) -> std::collections::HashMap<usize, String> {
    let mut m = std::collections::HashMap::new();
    let mut put = |lw: &LinearWeight, name: String| {
        m.insert(lw.buf().buf.as_ptr() as usize, name);
    };
    put(&w.embed, "model.embed_tokens".to_string());
    for (l, lw) in w.layers.iter().enumerate() {
        let p = format!("model.layers.{l}");
        let qkv = format!("{p}.self_attn.qkv_proj");
        let rmlp = format!("{p}.mlp.gate.router_mlp");
        put(&lw.q_proj, format!("{qkv}.q_proj"));
        put(&lw.k_proj, format!("{qkv}.k_proj"));
        put(&lw.v_cur, format!("{qkv}.v_proj_current"));
        put(&lw.v_del, format!("{qkv}.v_proj_delayed"));
        put(&lw.o_proj, format!("{p}.self_attn.o_proj"));
        put(&lw.down_proj_w, format!("{p}.mlp.gate.down_proj"));
        put(&lw.fc1_w, format!("{rmlp}.fc1"));
        put(&lw.fc2_w, format!("{rmlp}.fc2"));
        put(&lw.out_proj_w, format!("{rmlp}.out_proj"));
        // Routed experts (imatrix-only — sparse under top-1, so no full Hessian).
        for (e, ex) in lw.experts.iter().enumerate() {
            put(&ex.gate_up, format!("{p}.mlp.experts.{e}.gate_up_proj"));
            put(&ex.down, format!("{p}.mlp.experts.{e}.down_proj"));
        }
    }
    m
}

/// Calibration forward: run the full prompt and call `maybe_capture_activation`
/// before each dense gemv so the active `CalibCollector` accumulates `H = XᵀX`
/// (+ diag) keyed by the names in [`build_capture_names`]. No KV/decode state,
/// no logits output — just drives activations. The caller must set
/// `gpu.capture_names`/`gpu.active_capture` around this.
#[allow(clippy::needless_range_loop)]
/// Calibration forward over the whole prompt. Captures per-tensor Hessian +
/// imatrix via the active capture hook. When `kldref_topk` is `Some(k)`, also
/// computes the tied lm-head logits and returns the per-position `(logZ, top-k)`
/// KLDREF reference (the bf16 teacher signal for KL-divergence eval); otherwise
/// returns an empty vec and skips the lm-head gemv.
pub fn gpu_forward_calib(
    gpu: &mut Gpu,
    w: &ZayaGpuWeights,
    cfg: &ZayaConfig,
    ids: &[u32],
    kldref_topk: Option<usize>,
) -> Result<Vec<(f32, Vec<(u32, f32)>)>, String> {
    let s = ids.len();
    let h = cfg.hidden_size;
    let a = &cfg.attn;
    let (nq, nkv, hd) = (a.num_heads, a.num_kv_heads, a.head_dim);
    let q_dim = nq * hd;
    let k_dim = nkv * hd;
    let v_half = k_dim / 2;
    let conv_ch = q_dim + k_dim;
    let rh = cfg.moe.router_hidden_size;
    let n_route = cfg.moe.num_router_experts();
    let n_exp = cfg.moe.num_experts;
    let moe_int = cfg.moe.moe_intermediate_size;
    let eps = cfg.rms_norm_eps;
    let pad = (a.conv_depthwise_kernel - 1) + (a.conv_grouped_kernel - 1);
    let dw_len = s + pad - a.conv_depthwise_kernel + 1;
    let attn_scale = 1.0 / (hd as f32).sqrt();
    let l2_scale = (hd as f32).sqrt();

    let hidden = z2o(gpu, s, h)?;
    for t in 0..s {
        let row = hidden.sub_offset(t * h, h);
        w.embed.embed_lookup(gpu, &row, ids[t], h)?;
    }
    gpu.zaya_affine_input_f32(&hidden, &hidden, &w.in_scale, &w.in_bias, h, s * h)
        .map_err(|e| format!("{e:?}"))?;

    let normed = z2o(gpu, s, h)?;
    let q = zo(gpu, s * q_dim)?;
    let k = zo(gpu, s * k_dim)?;
    let vcur = zo(gpu, s * v_half)?;
    let vdel = zo(gpu, s * v_half)?;
    let qres = zo(gpu, s * nq * hd)?;
    let kres = zo(gpu, s * nkv * hd)?;
    let stream = zo(gpu, conv_ch * (s + pad))?;
    let dw = zo(gpu, conv_ch * dw_len)?;
    let gw = zo(gpu, conv_ch * s)?;
    let query = zo(gpu, s * nq * hd)?;
    let key = zo(gpu, s * nkv * hd)?;
    let value = zo(gpu, s * nkv * hd)?;
    let ctx = zo(gpu, s * q_dim)?;
    let attn_out = zo(gpu, s * h)?;
    let g_res2 = z2o(gpu, s, h)?;
    let rhid = z2o(gpu, s, rh)?;
    let rnormed = z2o(gpu, s, rh)?;
    let a1 = z2o(gpu, s, rh)?;
    let a2 = z2o(gpu, s, rh)?;
    let rlogits = zo(gpu, s * n_route)?;
    let moe_out = zo(gpu, s * h)?;
    let gate_up = zo(gpu, 2 * moe_int)?;
    let act = zo(gpu, moe_int)?;
    let down_t = zo(gpu, h)?;
    let router_state = zo(gpu, s * rh)?;
    let fnorm = z2o(gpu, s, h)?;

    for (li, lw) in w.layers.iter().enumerate() {
        gpu.rmsnorm_f32(&hidden, &lw.input_ln, &normed, eps)
            .map_err(|e| format!("{e:?}"))?;
        // capture the q/k/v projection inputs (all = post-input-norm hidden).
        gpu.maybe_capture_activation(lw.q_proj.buf(), &normed, s, h);
        gpu.maybe_capture_activation(lw.k_proj.buf(), &normed, s, h);
        gpu.maybe_capture_activation(lw.v_cur.buf(), &normed, s, h);
        gpu.maybe_capture_activation(lw.v_del.buf(), &normed, s, h);
        gemv_seq(gpu, &lw.q_proj, &normed, &q, s, q_dim, h)?;
        gemv_seq(gpu, &lw.k_proj, &normed, &k, s, k_dim, h)?;
        gemv_seq(gpu, &lw.v_cur, &normed, &vcur, s, v_half, h)?;
        gemv_seq(gpu, &lw.v_del, &normed, &vdel, s, v_half, h)?;
        gpu.zaya_qk_residual_f32(&qres, &kres, &q, &k, s, nq, nkv, hd, 0)
            .map_err(|e| format!("{e:?}"))?;
        gpu.zaya_qk_residual_f32(&qres, &kres, &q, &k, s, nq, nkv, hd, 1)
            .map_err(|e| format!("{e:?}"))?;
        gpu.zaya_qk_stream_f32(&stream, &q, &k, s, q_dim, k_dim, pad)
            .map_err(|e| format!("{e:?}"))?;
        gpu.zaya_conv1d_valid_f32(
            &dw,
            &stream,
            &lw.conv_dw_w,
            &lw.conv_dw_b,
            conv_ch,
            conv_ch,
            a.conv_depthwise_kernel,
            s + pad,
            dw_len,
        )
        .map_err(|e| format!("{e:?}"))?;
        gpu.zaya_conv1d_valid_f32(
            &gw,
            &dw,
            &lw.conv_gr_w,
            &lw.conv_gr_b,
            conv_ch,
            nq + nkv,
            a.conv_grouped_kernel,
            dw_len,
            s,
        )
        .map_err(|e| format!("{e:?}"))?;
        gpu.zaya_add_conv_residual_f32(&query, &gw, &qres, s, nq, hd, q_dim, 0)
            .map_err(|e| format!("{e:?}"))?;
        gpu.zaya_add_conv_residual_f32(&key, &gw, &kres, s, nkv, hd, q_dim, 1)
            .map_err(|e| format!("{e:?}"))?;
        gpu.zaya_value_compose_f32(&value, &vcur, &vdel, s, nkv, hd)
            .map_err(|e| format!("{e:?}"))?;
        gpu.zaya_qk_l2norm_temp_f32(&query, None, s, nq, hd, l2_scale, f32::EPSILON)
            .map_err(|e| format!("{e:?}"))?;
        gpu.zaya_qk_l2norm_temp_f32(&key, Some(&lw.qk_temp), s, nkv, hd, l2_scale, f32::EPSILON)
            .map_err(|e| format!("{e:?}"))?;
        gpu.zaya_rope_partial_f32(&query, s, nq, hd, a.n_rot, a.rope_theta, 0)
            .map_err(|e| format!("{e:?}"))?;
        gpu.zaya_rope_partial_f32(&key, s, nkv, hd, a.n_rot, a.rope_theta, 0)
            .map_err(|e| format!("{e:?}"))?;
        gpu.zaya_gqa_attn_f32(&ctx, &query, &key, &value, s, nq, nkv, hd, attn_scale)
            .map_err(|e| format!("{e:?}"))?;
        gpu.maybe_capture_activation(lw.o_proj.buf(), &ctx, s, q_dim);
        gemv_seq(gpu, &lw.o_proj, &ctx, &attn_out, s, h, q_dim)?;
        gpu.zaya_affine_residual_f32(
            &g_res2,
            &attn_out,
            &hidden,
            &lw.pa_rs[0],
            &lw.pa_rs[1],
            &lw.pa_rs[2],
            &lw.pa_rs[3],
            h,
            s * h,
        )
        .map_err(|e| format!("{e:?}"))?;
        gpu.rmsnorm_f32(&g_res2, &lw.post_attn_ln, &normed, eps)
            .map_err(|e| format!("{e:?}"))?;
        // MoE: capture the router projections (dense); experts run but aren't captured.
        gpu.maybe_capture_activation(lw.down_proj_w.buf(), &normed, s, h);
        gemv_seq(gpu, &lw.down_proj_w, &normed, &rhid, s, rh, h)?;
        gpu.zaya_bias_add_f32(&rhid, &lw.down_proj_b, rh, s * rh)
            .map_err(|e| format!("{e:?}"))?;
        if li != 0 {
            if let Some(scale) = lw.router_states_scale.as_ref() {
                gpu.zaya_eda_add_f32(&rhid, &router_state, scale, rh, s * rh)
                    .map_err(|e| format!("{e:?}"))?;
            }
        }
        gpu.zaya_write_at_f32(&router_state, &rhid, 0, s * rh)
            .map_err(|e| format!("{e:?}"))?;
        gpu.rmsnorm_f32(&rhid, &lw.rnorm_w, &rnormed, eps)
            .map_err(|e| format!("{e:?}"))?;
        gpu.maybe_capture_activation(lw.fc1_w.buf(), &rnormed, s, rh);
        gemv_seq(gpu, &lw.fc1_w, &rnormed, &a1, s, rh, rh)?;
        gpu.zaya_bias_add_f32(&a1, &lw.fc1_b, rh, s * rh)
            .map_err(|e| format!("{e:?}"))?;
        gpu.zaya_gelu_exact_f32(&a1, s * rh)
            .map_err(|e| format!("{e:?}"))?;
        gpu.maybe_capture_activation(lw.fc2_w.buf(), &a1, s, rh);
        gemv_seq(gpu, &lw.fc2_w, &a1, &a2, s, rh, rh)?;
        gpu.zaya_bias_add_f32(&a2, &lw.fc2_b, rh, s * rh)
            .map_err(|e| format!("{e:?}"))?;
        gpu.zaya_gelu_exact_f32(&a2, s * rh)
            .map_err(|e| format!("{e:?}"))?;
        gpu.maybe_capture_activation(lw.out_proj_w.buf(), &a2, s, rh);
        gemv_seq(gpu, &lw.out_proj_w, &a2, &rlogits, s, n_route, rh)?;
        let logit_host = gpu.download_f32(&rlogits).map_err(|e| format!("{e:?}"))?;
        gpu.fill_f32(&moe_out, 0.0).map_err(|e| format!("{e:?}"))?;
        for t in 0..s {
            let row = &logit_host[t * n_route..(t + 1) * n_route];
            let maxv = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut probs = vec![0f32; n_route];
            let mut denom = 0f32;
            for e in 0..n_route {
                probs[e] = (row[e] - maxv).exp();
                denom += probs[e];
            }
            for p in probs.iter_mut() {
                *p /= denom;
            }
            let mut best = 0usize;
            let mut bestv = f32::NEG_INFINITY;
            for e in 0..n_route {
                let v = probs[e] + lw.balancing_biases[e];
                if v > bestv {
                    bestv = v;
                    best = e;
                }
            }
            if best == n_exp {
                continue;
            }
            let weight = probs[best];
            let xt = normed.sub_offset(t * h, h);
            let ex = &lw.experts[best];
            gpu.maybe_capture_activation(ex.gate_up.buf(), &xt, 1, h); // no-op unless calibrating
            ex.gate_up.gemv(gpu, &xt, &gate_up)?;
            let g = gate_up.sub_offset(0, moe_int);
            let u = gate_up.sub_offset(moe_int, moe_int);
            gpu.silu_mul_f32(&g, &u, &act)
                .map_err(|e| format!("{e:?}"))?;
            gpu.maybe_capture_activation(ex.down.buf(), &act, 1, moe_int); // no-op unless calibrating
            ex.down.gemv(gpu, &act, &down_t)?;
            let ot = moe_out.sub_offset(t * h, h);
            gpu.scaled_add_inplace_cpu_scalar_f32(&ot, &down_t, weight)
                .map_err(|e| format!("{e:?}"))?;
        }
        gpu.zaya_affine_residual_f32(
            &hidden,
            &moe_out,
            &g_res2,
            &lw.pm_rs[0],
            &lw.pm_rs[1],
            &lw.pm_rs[2],
            &lw.pm_rs[3],
            h,
            s * h,
        )
        .map_err(|e| format!("{e:?}"))?;
        gpu.reclaim_pending();
    }

    gpu.rmsnorm_f32(&hidden, &w.norm, &fnorm, eps)
        .map_err(|e| format!("{e:?}"))?;
    // capture the tied lm_head input (no gemv needed for the imatrix/Hessian).
    gpu.maybe_capture_activation(w.embed.buf(), &fnorm, s, h);

    // Optional KLDREF: run the tied lm-head to get logits [s, vocab], then keep
    // only the per-position logZ + top-k (a compact bf16 reference). The full
    // [s, vocab] host buffer is dropped here, so peak host memory is one row.
    let kldref = if let Some(topk) = kldref_topk {
        let logits = zo(gpu, s * cfg.vocab_size)?;
        gemv_seq(gpu, &w.embed, &fnorm, &logits, s, cfg.vocab_size, h)?;
        let host = gpu.download_f32(&logits).map_err(|e| format!("{e:?}"))?;
        let v = cfg.vocab_size;
        (0..s)
            .map(|p| {
                let row = &host[p * v..(p + 1) * v];
                (logsumexp(row), topk_logits(row, topk))
            })
            .collect()
    } else {
        Vec::new()
    };

    gpu.reclaim_pending();
    Ok(kldref)
}

/// Single-token decode at `state.pos`: O(1) per-layer compute using the KV cache,
/// conv ring, and delayed-value state. Writes the new token's logits into
/// `logits_out` and advances the state by one position.
/// Serving decode entry. Normally runs the body directly; under
/// `HIPFIRE_ZAYA_GRAPH` (and only on the capture-clean on-device OQ8 path) it
/// runs the warmup→capture→replay hipGraph MVP harness. See [`ZayaDecodeState`].
pub fn gpu_decode(
    gpu: &mut Gpu,
    w: &ZayaGpuWeights,
    cfg: &ZayaConfig,
    token: u32,
    state: &mut ZayaDecodeState,
    logits_out: &GpuTensor,
) -> Result<(), String> {
    let pos = state.pos;
    if pos >= state.max_seq {
        return Err(format!(
            "zaya decode: position {pos} exceeds max_seq {}",
            state.max_seq
        ));
    }
    let h = cfg.hidden_size;
    // EXP-19 diagnostic: measure grid.sync() cost at the megakernel launch shape.
    // ~11 grid.syncs/block × num_blocks layers/token — is that the flat-tok/s cause?
    static SYNCBENCH: std::sync::Once = std::sync::Once::new();
    if std::env::var("HIPFIRE_ZAYA_SYNCBENCH").is_ok() && !SYNCBENCH.is_completed() {
        SYNCBENCH.call_once(|| {});
        for gb in [160u32, 80, 40, 20] {
            match gpu.zaya_bench_grid_sync(gb, 2000) {
                Ok(us) => {
                    let per_tok = us * 11.0 * cfg.num_blocks as f64;
                    eprintln!(
                        "[zaya-syncbench] grid={gb} blocks: {us:.3} us/grid.sync → \
                         {per_tok:.0} us/token (11 syncs/block × {} blocks)",
                        cfg.num_blocks
                    );
                }
                Err(e) => eprintln!("[zaya-syncbench] grid={gb} failed: {e:?}"),
            }
        }
    }
    // Persistent residual-stream buffer — a stable device address is required so
    // the captured graph body can bake it once and read/evolve it every replay.
    if state.decode_hidden.is_none() {
        state.decode_hidden = Some(
            gpu.zeros(&[h], DType::F32)
                .map_err(|e| format!("zaya decode_hidden alloc: {e:?}"))?,
        );
    }
    // Non-owning view so `hidden` can be passed alongside `&mut state`.
    let hidden = {
        let ptr = state.decode_hidden.as_ref().unwrap().buf.as_ptr();
        GpuTensor {
            buf: unsafe { hip_bridge::DeviceBuffer::from_raw(ptr, h * 4) },
            shape: vec![h],
            dtype: DType::F32,
        }
    };

    // The captured attention sizes its scores LDS by a fixed cap (see
    // dispatch::attention GRAPH_CTX_CAP = 8192); beyond it the baked shared-mem
    // would overflow, so tear the graph down and run pure non-graph for the rest.
    const GRAPH_CTX_CAP: usize = 8192;
    if pos >= GRAPH_CTX_CAP && gpu.active_stream.is_some() {
        let _ = gpu.end_graph_capture();
        gpu.drop_captured_graph();
        gpu.active_stream = None;
        state.graph_disabled = true;
    }

    // Graph-capture eligibility: on-device (plain-oq8) MoE only (the AWQ host
    // fallback does a per-block `download_f32`), NOT kvarn (kvarn_attend bakes
    // `start_pos` as an immediate), and within the fixed LDS cap. The f32
    // attention reads pos_buf, so the body is position-independent.
    let capturable = !state.graph_disabled
        && pos < GRAPH_CTX_CAP
        && state.kvarn.is_none()
        && w.layers.first().is_some_and(|l| l.moe_indexed.is_some())
        && std::env::var("HIPFIRE_ZAYA_GRAPH").is_ok();

    if !capturable {
        if std::env::var("HIPFIRE_ZAYA_LAUNCHSTATS").is_ok() {
            use hip_bridge::launch_counters as lc;
            lc::reset();
            let t = std::time::Instant::now();
            let r = (|| -> Result<(), String> {
                gpu_decode_prologue(gpu, w, token, pos, h, &hidden, &state.pos_buf)?;
                gpu_decode_body(gpu, w, cfg, state, &hidden, logits_out)
            })();
            let wall = t.elapsed().as_micros();
            eprintln!(
                "[zaya-stats] pos={pos} wall={wall}us | launch n={} cpu={}us | dtoh n={} t={}us | dtod n={} htod n={} memset n={} | stream_sync n={} t={}us | device_sync n={} t={}us | event_sync n={}",
                lc::launch_kernel::count(),
                lc::launch_kernel::time_ns() / 1000,
                lc::memcpy_dtoh::count(),
                lc::memcpy_dtoh::time_ns() / 1000,
                lc::memcpy_dtod::count(),
                lc::memcpy_htod::count(),
                lc::memset::count(),
                lc::stream_sync::count(),
                lc::stream_sync::time_ns() / 1000,
                lc::device_sync::count(),
                lc::device_sync::time_ns() / 1000,
                lc::event_sync::count(),
            );
            r?;
            state.pos = pos + 1;
            return Ok(());
        }
        gpu_decode_prologue(gpu, w, token, pos, h, &hidden, &state.pos_buf)?;
        gpu_decode_body(gpu, w, cfg, state, &hidden, logits_out)?;
        state.pos = pos + 1;
        return Ok(());
    }

    // ── hipGraph path (HIPFIRE_ZAYA_GRAPH) ──
    // The token/pos-dependent prologue (pos_buf write + embed + input affine) runs
    // EVERY token outside the captured region; only the position-independent layer
    // body is captured (RoPE/attention read pos_buf, hidden is a stable address),
    // so replay produces CORRECT output — unlike the earlier whole-body MVP.
    gpu_decode_prologue(gpu, w, token, pos, h, &hidden, &state.pos_buf)?;

    if state.graph_ready {
        if std::env::var("HIPFIRE_ZAYA_LAUNCHSTATS").is_ok() {
            let t = std::time::Instant::now();
            gpu.graph_launch()
                .map_err(|e| format!("zaya graph replay: {e:?}"))?;
            let submit_us = t.elapsed().as_micros();
            // Synchronize to isolate true GPU body execution time (roofline check).
            let _ = gpu.hip.device_synchronize();
            eprintln!(
                "[zaya-graph-replay] pos={pos} submit={submit_us}us gpu_body={}us",
                t.elapsed().as_micros(),
            );
        } else {
            gpu.graph_launch()
                .map_err(|e| format!("zaya graph replay: {e:?}"))?;
        }
        state.pos = pos + 1;
        return Ok(());
    }

    // A capturable stream is required (the null stream cannot be captured).
    if gpu.active_stream.is_none() {
        gpu.active_stream = Some(
            gpu.hip
                .stream_create()
                .map_err(|e| format!("zaya graph stream: {e:?}"))?,
        );
    }

    // Warmup: run the body direct once to prime the scratch pool (capture cannot
    // hipMalloc); reclaim afterwards so the capture step reuses the free-list.
    if !state.graph_warmed {
        gpu_decode_body(gpu, w, cfg, state, &hidden, logits_out)?;
        gpu.reclaim_pending();
        state.graph_warmed = true;
        state.pos = pos + 1;
        return Ok(());
    }

    // Capture the position-independent body, then replay once for this token.
    let captured = (|| -> Result<(), String> {
        gpu.begin_graph_capture()
            .map_err(|e| format!("zaya begin capture: {e:?}"))?;
        gpu_decode_body(gpu, w, cfg, state, &hidden, logits_out)?;
        gpu.end_graph_capture()
            .map_err(|e| format!("zaya end capture: {e:?}"))?;
        gpu.graph_launch()
            .map_err(|e| format!("zaya graph launch: {e:?}"))
    })();
    match captured {
        Ok(()) => {
            state.graph_ready = true;
            state.pos = pos + 1;
            Ok(())
        }
        Err(e) => {
            eprintln!("[zaya-graph] capture failed, disabling graph + falling back to direct: {e}");
            // Always close the stream's capture state (resets capture_mode and is a
            // HIP requirement even after a mid-capture error) before falling back —
            // otherwise the direct re-run below inherits capture_mode and 906s too.
            let _ = gpu.end_graph_capture();
            gpu.drop_captured_graph();
            gpu.active_stream = None;
            state.graph_disabled = true;
            // Capture records without executing, so re-run the body directly to
            // emit this token's logits.
            gpu_decode_body(gpu, w, cfg, state, &hidden, logits_out)?;
            state.pos = pos + 1;
            Ok(())
        }
    }
}

/// Per-token decode prologue (never captured — token/pos vary): write the device
/// position, embed the token, and apply the input-residual affine into `hidden`.
fn gpu_decode_prologue(
    gpu: &mut Gpu,
    w: &ZayaGpuWeights,
    token: u32,
    pos: usize,
    h: usize,
    hidden: &GpuTensor,
    pos_buf: &hip_bridge::DeviceBuffer,
) -> Result<(), String> {
    gpu.hip
        .memcpy_htod(pos_buf, &(pos as i32).to_ne_bytes())
        .map_err(|e| format!("zaya pos_buf write: {e:?}"))?;
    w.embed.embed_lookup(gpu, hidden, token, h)?;
    gpu.zaya_affine_input_f32(hidden, hidden, &w.in_scale, &w.in_bias, h, h)
        .map_err(|e| format!("{e:?}"))?;
    Ok(())
}

#[allow(clippy::needless_range_loop)]
fn gpu_decode_body(
    gpu: &mut Gpu,
    w: &ZayaGpuWeights,
    cfg: &ZayaConfig,
    state: &mut ZayaDecodeState,
    hidden: &GpuTensor,
    logits_out: &GpuTensor,
) -> Result<(), String> {
    // `hidden` (residual stream) and `state.pos_buf` are produced by the prologue.
    // This body is position-independent (RoPE + attention read pos_buf), so it is
    // the region captured into the decode hipGraph.
    let pos = state.pos;
    // Timing-ablation hooks (HIPFIRE_ZAYA_ABLATE, comma list; output is garbage —
    // for gpu_body decomposition only): "moe" skips the two big expert weight GEMVs.
    let ablate = std::env::var("HIPFIRE_ZAYA_ABLATE").unwrap_or_default();
    let ablate_moe = ablate.contains("moe");
    // "glue" skips the CCA attention elementwise glue (conv/residual/value/l2/rope).
    let ablate_glue = ablate.contains("glue");
    // Timing probe: process only the first N blocks (gpu_body slope vs N gives the
    // per-block cost; intercept = embed+lm_head+norm). Output garbage — timing only.
    let nblocks = std::env::var("HIPFIRE_ZAYA_NBLOCKS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(usize::MAX);
    let h = cfg.hidden_size;
    let a = &cfg.attn;
    let (nq, nkv, hd) = (a.num_heads, a.num_kv_heads, a.head_dim);
    let q_dim = nq * hd;
    let k_dim = nkv * hd;
    let kvdim = k_dim;
    let v_half = k_dim / 2;
    let conv_ch = q_dim + k_dim;
    let rh = cfg.moe.router_hidden_size;
    let n_route = cfg.moe.num_router_experts();
    let n_exp = cfg.moe.num_experts;
    let moe_int = cfg.moe.moe_intermediate_size;
    let eps = cfg.rms_norm_eps;
    let pad = (a.conv_depthwise_kernel - 1) + (a.conv_grouped_kernel - 1);
    // attention_f32 applies scale = 1/sqrt(head_dim) internally.
    let l2_scale = (hd as f32).sqrt();

    // single-token scratch (s = 1)
    let normed = z2o(gpu, 1, h)?;
    let q = zo(gpu, q_dim)?;
    let k = zo(gpu, k_dim)?;
    let vcur = zo(gpu, v_half)?;
    let vdel = zo(gpu, v_half)?;
    let qres = zo(gpu, nq * hd)?;
    let kres = zo(gpu, nkv * hd)?;
    let cur_qk = zo(gpu, conv_ch)?;
    let window = zo(gpu, conv_ch * (pad + 1))?;
    let dw = zo(gpu, conv_ch * (pad + 1 - a.conv_depthwise_kernel + 1))?;
    let gw = zo(gpu, conv_ch)?;
    let query = zo(gpu, nq * hd)?;
    let key = zo(gpu, nkv * hd)?;
    let value = zo(gpu, nkv * hd)?;
    let ctx = zo(gpu, q_dim)?;
    let attn_out = zo(gpu, h)?;
    let g_res2 = z2o(gpu, 1, h)?;
    let rhid = z2o(gpu, 1, rh)?;
    let rnormed = z2o(gpu, 1, rh)?;
    let a1 = z2o(gpu, 1, rh)?;
    let a2 = z2o(gpu, 1, rh)?;
    let rlogits = zo(gpu, n_route)?;
    let moe_out = zo(gpu, h)?;
    let gate_up = zo(gpu, 2 * moe_int)?;
    let act = zo(gpu, moe_int)?;
    let down_t = zo(gpu, h)?;
    // On-device top-1 route outputs (indexed MoE path): winning expert id
    // (i32-in-f32) and its unbiased softmax gate.
    let sel_idx = zo(gpu, 1)?;
    let sel_gate = zo(gpu, 1)?;
    // FWHT-rotated activations for the OQ8 experts (Opus Quant weights live in
    // the rotated basis, so decode must rotate x before the indexed GEMVs).
    let xr_norm = zo(gpu, h)?;
    let xr_act = zo(gpu, moe_int)?;
    // Stage-12 scratch for the cooperative megakernel-B (rmsnormed pre-FWHT).
    let mk_norm = zo(gpu, h)?;
    // Phase 2 scratch: FWHT-rotated ctx for the folded o_proj (megakernel-B head).
    let ctx_rot = zo(gpu, q_dim)?;
    let router_state = zo(gpu, rh)?;
    // ZAYA decode cooperative megakernel (HIPFIRE_ZAYA_MEGAKERNEL): fuse the MLP
    // half (stages 12–17) into one cooperative launch. "validate" also runs the
    // reference path and logs the per-layer cosine of the fused `hidden`.
    let mega_env = std::env::var("HIPFIRE_ZAYA_MEGAKERNEL").unwrap_or_default();
    let mega_on = !mega_env.is_empty();
    let mega_validate = mega_env == "validate";
    let hidden2 = if mega_validate {
        Some(zo(gpu, h)?)
    } else {
        None
    };
    // Front-half (megakernel-A) validation scratch: parallel query/key/value.
    let (query2, key2, value2) = if mega_validate {
        (
            Some(zo(gpu, q_dim)?),
            Some(zo(gpu, k_dim)?),
            Some(zo(gpu, k_dim)?),
        )
    } else {
        (None, None, None)
    };
    let dw_len = pad + 1 - a.conv_depthwise_kernel + 1;
    let fnorm = z2o(gpu, 1, h)?;
    // W8A8 QKV: one fused rmsnorm+rotate → one int8-quantize → shared across the
    // four in-projections (dedups the 4 redundant per-gemv rotations of `normed`).
    let qkv_xrot = z2o(gpu, 1, h)?;
    let _qkv_xq = gpu
        .zeros_owned(&[h], DType::Raw)
        .map_err(|e| format!("zaya alloc qkv_xq: {e:?}"))?;
    let _qkv_xs = zo(gpu, h / 256)?;
    // Post-attention rmsnorm shared across down_proj (W8A8) AND the MoE gate_up
    // (reuses the f32 rotated form) — one fused rmsnorm+rotate feeds both.
    let pa_xrot = z2o(gpu, 1, h)?;
    let pa_xq = gpu
        .zeros_owned(&[h], DType::Raw)
        .map_err(|e| format!("zaya alloc pa_xq: {e:?}"))?;
    let pa_xs = zo(gpu, h / 256)?;
    // W8A8 MoE (HIPFIRE_ZAYA_MOE_W8A8): int8-quantize the down-proj activation (gate_up
    // reuses pa_xq/pa_xs). The f32 activation is ~80% of the MoE gemv load traffic;
    // int8 cuts it 4× + enables the signed V_DOT4 int8 dot. Requires the W8A8 post-attn
    // path (pa_xq valid). Small quality cost (activation quant) — not bit-exact.
    let moe_w8a8 = std::env::var("HIPFIRE_ZAYA_MOE_W8A8").is_ok();
    let moe_xq_dn = gpu
        .zeros_owned(&[moe_int], DType::Raw)
        .map_err(|e| format!("zaya alloc moe_xq_dn: {e:?}"))?;
    let moe_xs_dn = zo(gpu, moe_int / 256)?;

    // EXP-20 diagnostic: device-sync section timers (body loop vs lm_head).
    let sectiontime = std::env::var("HIPFIRE_ZAYA_SECTIONTIME").is_ok();
    let sec_t0 = if sectiontime {
        let _ = gpu.hip.device_synchronize();
        Some(std::time::Instant::now())
    } else {
        None
    };
    for (li, lw) in w.layers.iter().take(nblocks).enumerate() {
        if li == 0 && std::env::var("HIPFIRE_ZAYA_DTYPES").is_ok() {
            eprintln!(
                "[zaya-dtypes] down_proj={:?} fc1={:?} fc2={:?} out_proj={:?} rnorm_is_gt qk_temp",
                lw.down_proj_w.quant_dtype(),
                lw.fc1_w.quant_dtype(),
                lw.fc2_w.quant_dtype(),
                lw.out_proj_w.quant_dtype(),
            );
        }
        // W8A8 in-projection path when the dense weights are plain Oq8G256 (no AWQ
        // sidecar, which this path doesn't apply); else the W8A16 fallback.
        let w8a8_qkv =
            lw.q_proj.quant_dtype() == Some(DType::Oq8G256) && !lw.q_proj.quant_has_awq();
        // ── ZAYA cooperative megakernel-A (front half, stages 1–8) ──
        // Fuse input rmsnorm+rotate → qkv gemv → conv glue → l2norm → RoPE into one
        // cooperative launch, producing query/key/value ready for attention.
        // `validate` runs A on a snapshot of the conv-ring + delayed-v state so the
        // reference run below re-advances them from the same base (authoritative).
        let front_mega = mega_on && w8a8_qkv && !ablate_glue;
        if front_mega {
            let (qw, _, _) = lw.q_proj.quant_mk().unwrap();
            let (kw, _, _) = lw.k_proj.quant_mk().unwrap();
            let (vcw, _, _) = lw.v_cur.quant_mk().unwrap();
            let (vdw, _, _) = lw.v_del.quant_mk().unwrap();
            let (out_q, out_k, out_v) = if mega_validate {
                (
                    query2.as_ref().unwrap(),
                    key2.as_ref().unwrap(),
                    value2.as_ref().unwrap(),
                )
            } else {
                (&query, &key, &value)
            };
            let state_save = if mega_validate {
                let r = gpu
                    .download_f32(&state.conv_ring[li])
                    .map_err(|e| format!("{e:?}"))?;
                let d = gpu
                    .download_f32(&state.delayed_v[li])
                    .map_err(|e| format!("{e:?}"))?;
                Some((r, d))
            } else {
                None
            };
            gpu.zaya_decode_megakernel_a(
                hidden,
                &lw.input_ln,
                &qkv_xrot,
                qw,
                kw,
                vcw,
                vdw,
                &q,
                &k,
                &vcur,
                &vdel,
                &qres,
                &kres,
                &cur_qk,
                &lw.conv_dw_w,
                &lw.conv_dw_b,
                &lw.conv_gr_w,
                &lw.conv_gr_b,
                &state.conv_ring[li],
                &window,
                &dw,
                &gw,
                &state.delayed_v[li],
                out_q,
                out_k,
                out_v,
                &lw.qk_temp,
                &state.pos_buf,
                h,
                nq,
                nkv,
                hd,
                q_dim,
                k_dim,
                v_half,
                conv_ch,
                pad,
                a.conv_depthwise_kernel,
                a.conv_grouped_kernel,
                dw_len,
                a.n_rot,
                a.rope_theta,
                l2_scale,
                f32::EPSILON,
            )
            .map_err(|e| format!("{e:?}"))?;
            if let Some((r, d)) = state_save {
                let rb: Vec<u8> = r.iter().flat_map(|f| f.to_ne_bytes()).collect();
                let db: Vec<u8> = d.iter().flat_map(|f| f.to_ne_bytes()).collect();
                gpu.memcpy_htod_auto(&state.conv_ring[li].buf, &rb)
                    .map_err(|e| format!("{e:?}"))?;
                gpu.memcpy_htod_auto(&state.delayed_v[li].buf, &db)
                    .map_err(|e| format!("{e:?}"))?;
            }
        }
        // Reference front half runs unless the megakernel replaced it (still runs
        // in validate, as the authoritative path for the comparison).
        let run_ref_front = !front_mega || mega_validate;
        if run_ref_front {
            if w8a8_qkv {
                // Fused rmsnorm+rotate → one W8A16 GEMV computing all 4 projections
                // (q/k/vcur/vdel) in a single launch, reading the rotated f32 activation
                // once. Replaces quantize_act_oq8 + 4× W8A8 GEMV (5 kernels → 2).
                gpu.fused_rmsnorm_rotate_mq_plain(hidden, &lw.input_ln, &qkv_xrot, &normed, h, eps)
                    .map_err(|e| format!("{e:?}"))?;
                let (qw, qm, _) = lw.q_proj.quant_mk().unwrap();
                let (kw, km, _) = lw.k_proj.quant_mk().unwrap();
                let (vcw, vcm, _) = lw.v_cur.quant_mk().unwrap();
                let (vdw, vdm, kk) = lw.v_del.quant_mk().unwrap();
                gpu.fused_qkvza_oq8_gemv(
                    qw, kw, vcw, vdw, &qkv_xrot, &q, &k, &vcur, &vdel, qm, km, vcm, vdm, kk, 256,
                )
                .map_err(|e| format!("{e:?}"))?;
            } else {
                gpu.rmsnorm_f32(hidden, &lw.input_ln, &normed, eps)
                    .map_err(|e| format!("{e:?}"))?;
                lw.q_proj.gemv(gpu, &normed, &q)?;
                lw.k_proj.gemv(gpu, &normed, &k)?;
                lw.v_cur.gemv(gpu, &normed, &vcur)?;
                lw.v_del.gemv(gpu, &normed, &vdel)?;
            }
            if !ablate_glue {
                // Fused qk_residual (both modes) + qk-stream column (3 launches → 1).
                gpu.zaya_qk_prep_decode_f32(
                    &q, &k, &qres, &kres, &cur_qk, nq, nkv, hd, q_dim, k_dim,
                )
                .map_err(|e| format!("{e:?}"))?;
                // conv window from the ring (advances ring).
                gpu.zaya_conv_window_f32(&window, &state.conv_ring[li], &cur_qk, conv_ch, pad)
                    .map_err(|e| format!("{e:?}"))?;
                gpu.zaya_conv1d_valid_f32(
                    &dw,
                    &window,
                    &lw.conv_dw_w,
                    &lw.conv_dw_b,
                    conv_ch,
                    conv_ch,
                    a.conv_depthwise_kernel,
                    pad + 1,
                    dw_len,
                )
                .map_err(|e| format!("{e:?}"))?;
                gpu.zaya_conv1d_valid_f32(
                    &gw,
                    &dw,
                    &lw.conv_gr_w,
                    &lw.conv_gr_b,
                    conv_ch,
                    nq + nkv,
                    a.conv_grouped_kernel,
                    dw_len,
                    1,
                )
                .map_err(|e| format!("{e:?}"))?;
                // Fused q+k add-conv-residual (one launch, both modes).
                gpu.zaya_add_conv_residual_qk_f32(
                    &query, &key, &gw, &qres, &kres, 1, nq, nkv, hd, q_dim,
                )
                .map_err(|e| format!("{e:?}"))?;
                // Fused value assembly: head0=current v, head1=previous delayed v, then
                // advance delayed_v (one launch, replaces 3× write_at).
                gpu.zaya_value_assemble_decode_f32(
                    &value,
                    &vcur,
                    &state.delayed_v[li],
                    &vdel,
                    v_half,
                )
                .map_err(|e| format!("{e:?}"))?;
                // Fused q+k L2-norm+scale (query no temp, key per-head temp) in one launch.
                gpu.zaya_qk_l2norm_qk_f32(
                    &query,
                    &key,
                    &lw.qk_temp,
                    1,
                    nq,
                    nkv,
                    hd,
                    l2_scale,
                    f32::EPSILON,
                )
                .map_err(|e| format!("{e:?}"))?;
                // Fused q+k partial-RoPE (device position from pos_buf → capture-safe).
                gpu.zaya_rope_partial_qk_posbuf_f32(
                    &query,
                    &key,
                    &state.pos_buf,
                    1,
                    nq,
                    nkv,
                    hd,
                    a.n_rot,
                    a.rope_theta,
                )
                .map_err(|e| format!("{e:?}"))?;
            } // end ablate_glue
        } // end run_ref_front
        if front_mega && mega_validate {
            let cmp: [(&str, &GpuTensor, &GpuTensor, usize); 3] = [
                ("query", &query, query2.as_ref().unwrap(), q_dim),
                ("key", &key, key2.as_ref().unwrap(), k_dim),
                ("value", &value, value2.as_ref().unwrap(), k_dim),
            ];
            for (name, aref, bref, n) in cmp {
                let av = gpu.download_f32(aref).map_err(|e| format!("{e:?}"))?;
                let bv = gpu.download_f32(bref).map_err(|e| format!("{e:?}"))?;
                let (mut dot, mut na, mut nb, mut mx) = (0f64, 0f64, 0f64, 0f64);
                for i in 0..n {
                    let (x, y) = (av[i] as f64, bv[i] as f64);
                    dot += x * y;
                    na += x * x;
                    nb += y * y;
                    mx = mx.max((x - y).abs());
                }
                let cos = dot / (na.sqrt() * nb.sqrt()).max(1e-30);
                eprintln!(
                    "[zaya-mega-a-validate] pos={pos} li={li} {name} cos={cos:.6} maxabs={mx:.3e}"
                );
            }
        }
        // Megakernel fold decisions (hoisted so the reference stage-9 attention can
        // be skipped when Phase 3 folds it). fold_oproj: Phase 2/3 fold o_proj +
        // attn affine into megakernel-B. fold_attn: Phase 3 (=3) also folds the KV
        // write + flash attention into B (f32-KV path only, not KVarN).
        let mega_fold_pre = std::env::var("HIPFIRE_ZAYA_HOST_MOE").is_err()
            && lw.moe_indexed.is_some()
            && lw.down_proj_w.quant_dtype() == Some(DType::Oq8G256)
            && !lw.down_proj_w.quant_has_awq()
            && lw.o_proj.quant_dtype() == Some(DType::Oq8G256)
            && !lw.o_proj.quant_has_awq();
        let fold_oproj = (mega_env == "2" || mega_env == "3") && mega_fold_pre;
        // Phase 3: fold the KV write + flash-decode attention (stage 9) into
        // megakernel-B's head (f32-KV path only, not KVarN). Byte-identical output.
        let fold_attn = mega_env == "3" && mega_fold_pre && state.kvarn.is_none();
        // append to KV cache at `pos`, then attend over 0..=pos.
        // Append the current K/V at the device position, then flash-decode
        // attention (one block per head, threads split the context + LDS reduce,
        // online softmax) — replaces the 1-thread-per-head serial `zaya_gqa_decode`.
        if let Some(kv) = state.kvarn.as_ref() {
            // Fused KVarN: low-bit var-norm K + Q8 V write (n=1) + flash read over
            // [0..=pos]. The KV kernels take positions from a GpuTensor; wrap the
            // raw 4-byte i32 pos_buf as a non-owning [1] view.
            let pos_view = GpuTensor {
                buf: unsafe { hip_bridge::DeviceBuffer::from_raw(state.pos_buf.as_ptr(), 4) },
                shape: vec![1],
                dtype: DType::F32,
            };
            gpu.kvarn_attend(
                &kv.cache.k_gpu[li],
                &kv.cache.k_window[li],
                &kv.cache.v_gpu[li],
                &query,
                &key,
                &value,
                &pos_view,
                &ctx,
                &kv.flash_partials,
                &kv.tiles,
                1,
                pos,
                nq,
                nkv,
                hd,
                kv.cache.physical_cap,
                None,
                0,
                0,
                kv.bits,
            )
            .map_err(|e| format!("{e:?}"))?;
        } else if !(fold_attn && !mega_validate) {
            // Reference stage 9 (skipped when Phase 3 folds it into megakernel-B,
            // except in validate where the reference is authoritative).
            gpu.kv_cache_write(&state.k_cache[li], &key, &state.pos_buf, kvdim)
                .map_err(|e| format!("{e:?}"))?;
            gpu.kv_cache_write(&state.v_cache[li], &value, &state.pos_buf, kvdim)
                .map_err(|e| format!("{e:?}"))?;
            gpu.attention_f32(
                &query,
                &state.k_cache[li],
                &state.v_cache[li],
                &ctx,
                &state.pos_buf,
                pos + 1,
                nq,
                nkv,
                hd,
                state.max_seq,
            )
            .map_err(|e| format!("{e:?}"))?;
        }
        // fold_oproj / fold_attn were decided above (hoisted). The reference o_proj
        // + attn affine (stage 10–11) are skipped when folded, except in validate.
        let skip_ref_oproj = fold_oproj && !mega_validate;
        if !skip_ref_oproj {
            lw.o_proj.gemv(gpu, &ctx, &attn_out)?;
            gpu.zaya_affine_residual_f32(
                &g_res2,
                &attn_out,
                hidden,
                &lw.pa_rs[0],
                &lw.pa_rs[1],
                &lw.pa_rs[2],
                &lw.pa_rs[3],
                h,
                h,
            )
            .map_err(|e| format!("{e:?}"))?;
        }
        // Post-attention rmsnorm: fused rmsnorm+rotate when down_proj is plain
        // Oq8G256, producing `pa_xrot` shared with the MoE gate_up below.
        let w8a8_pa =
            lw.down_proj_w.quant_dtype() == Some(DType::Oq8G256) && !lw.down_proj_w.quant_has_awq();
        let force_host_moe = std::env::var("HIPFIRE_ZAYA_HOST_MOE").is_ok();
        let eda_scale = if li != 0 {
            lw.router_states_scale.as_ref()
        } else {
            None
        };
        // ── ZAYA cooperative megakernel-B (stages 12–17, the MLP half) ──
        // Fuse post-attn rmsnorm+rotate → router MLP+select → MoE gate_up →
        // silu_mul+rotate → MoE down + affine residual into ONE cooperative
        // launch. Same preconditions as the fused router path (plain Oq8 W8A8
        // down_proj + indexed MoE). `validate` runs the megakernel on a snapshot
        // of the EDA state, then falls through to the reference path and logs the
        // per-layer cosine of `hidden`.
        let mega_eligible = mega_on && w8a8_pa && !force_host_moe && lw.moe_indexed.is_some();
        if mega_eligible {
            let idx = lw.moe_indexed.as_ref().unwrap();
            let (dpw, _, _) = lw.down_proj_w.quant_mk().unwrap();
            let (f1w, _, _) = lw.fc1_w.quant_mk().unwrap();
            let (f2w, _, _) = lw.fc2_w.quant_mk().unwrap();
            let (ow, _, _) = lw.out_proj_w.quant_mk().unwrap();
            let out_hidden = if mega_validate {
                hidden2.as_ref().unwrap()
            } else {
                hidden
            };
            let rs_save = if mega_validate {
                Some(
                    gpu.download_f32(&router_state)
                        .map_err(|e| format!("{e:?}"))?,
                )
            } else {
                None
            };
            let op_w = if fold_oproj {
                lw.o_proj.quant_mk().unwrap().0
            } else {
                dpw
            };
            // Phase 3 attention buffers (real f32 KV cache when folding; dummy
            // otherwise — fold_attn is false unless the f32-KV path is active).
            let (kc_t, vc_t) = if state.kvarn.is_none() {
                (&state.k_cache[li], &state.v_cache[li])
            } else {
                (hidden, hidden)
            };
            let attn_scale = 1.0f32 / (hd as f32).sqrt();
            gpu.zaya_decode_megakernel_b(
                &g_res2,
                out_hidden,
                &ctx,
                op_w,
                &ctx_rot,
                &lw.pa_rs,
                q_dim,
                fold_oproj,
                &query,
                &key,
                &value,
                kc_t,
                vc_t,
                &state.pos_buf,
                nkv,
                hd,
                state.max_seq,
                attn_scale,
                fold_attn,
                &lw.post_attn_ln,
                &pa_xrot,
                &mk_norm,
                dpw,
                &lw.down_proj_b,
                &router_state,
                eda_scale,
                &lw.rnorm_w,
                f1w,
                &lw.fc1_b,
                f2w,
                &lw.fc2_b,
                ow,
                &idx.balancing_biases_dev,
                &sel_idx,
                &sel_gate,
                &idx.gate_up_ptrs,
                &idx.down_ptrs,
                &gate_up,
                &xr_act,
                &lw.pm_rs,
                h,
                rh,
                n_route,
                moe_int,
                eps,
            )
            .map_err(|e| format!("{e:?}"))?;
            if let Some(rs) = rs_save {
                // Restore the EDA router_state so the reference run below advances
                // it from the same base (the reference is authoritative).
                let bytes: Vec<u8> = rs.iter().flat_map(|f| f.to_ne_bytes()).collect();
                gpu.memcpy_htod_auto(&router_state.buf, &bytes)
                    .map_err(|e| format!("{e:?}"))?;
            }
            if !mega_validate {
                gpu.reclaim_pending();
                continue;
            }
            // validate: fall through to the reference path, compare after it writes `hidden`.
        }
        // Fused router MLP megakernel (HIPFIRE_ZAYA_ROUTER_FUSED): down_proj + prep +
        // rmsnorm + FWHT/fc1/gelu + FWHT/fc2/gelu + FWHT/out + select → ~9 kernels → 1.
        let router_fused = std::env::var("HIPFIRE_ZAYA_ROUTER_FUSED").is_ok()
            && w8a8_pa
            && !force_host_moe
            && lw.moe_indexed.is_some();
        if router_fused {
            gpu.fused_rmsnorm_rotate_mq_plain(&g_res2, &lw.post_attn_ln, &pa_xrot, &normed, h, eps)
                .map_err(|e| format!("{e:?}"))?;
            let idx = lw.moe_indexed.as_ref().unwrap();
            let (dpw, _, _) = lw.down_proj_w.quant_mk().unwrap();
            let (f1w, _, _) = lw.fc1_w.quant_mk().unwrap();
            let (f2w, _, _) = lw.fc2_w.quant_mk().unwrap();
            let (ow, _, _) = lw.out_proj_w.quant_mk().unwrap();
            gpu.zaya_router_mlp_fused(
                &pa_xrot,
                dpw,
                &lw.down_proj_b,
                &router_state,
                eda_scale,
                &lw.rnorm_w,
                f1w,
                &lw.fc1_b,
                f2w,
                &lw.fc2_b,
                ow,
                &idx.balancing_biases_dev,
                &sel_idx,
                &sel_gate,
                h,
                rh,
                n_route,
                eps,
            )
            .map_err(|e| format!("{e:?}"))?;
        } else {
            if w8a8_pa {
                gpu.fused_rmsnorm_rotate_mq_plain(
                    &g_res2,
                    &lw.post_attn_ln,
                    &pa_xrot,
                    &normed,
                    h,
                    eps,
                )
                .map_err(|e| format!("{e:?}"))?;
                gpu.quantize_act_oq8(&pa_xrot, &pa_xq, &pa_xs, 1, h, 256)
                    .map_err(|e| format!("{e:?}"))?;
                lw.down_proj_w.gemv_w8a8(gpu, &pa_xq, &pa_xs, &rhid)?;
            } else {
                gpu.rmsnorm_f32(&g_res2, &lw.post_attn_ln, &normed, eps)
                    .map_err(|e| format!("{e:?}"))?;
                lw.down_proj_w.gemv(gpu, &normed, &rhid)?;
            }
            gpu.zaya_router_prep_f32(&rhid, &lw.down_proj_b, &router_state, eda_scale, rh, rh)
                .map_err(|e| format!("{e:?}"))?;
            gpu.rmsnorm_f32(&rhid, &lw.rnorm_w, &rnormed, eps)
                .map_err(|e| format!("{e:?}"))?;
            lw.fc1_w.gemv(gpu, &rnormed, &a1)?;
            gpu.zaya_bias_gelu_f32(&a1, &lw.fc1_b, rh, rh)
                .map_err(|e| format!("{e:?}"))?;
            lw.fc2_w.gemv(gpu, &a1, &a2)?;
            gpu.zaya_bias_gelu_f32(&a2, &lw.fc2_b, rh, rh)
                .map_err(|e| format!("{e:?}"))?;
            lw.out_proj_w.gemv(gpu, &a2, &rlogits)?;
        }
        gpu.fill_f32(&moe_out, 0.0).map_err(|e| format!("{e:?}"))?;
        if let Some(idx) = lw.moe_indexed.as_ref().filter(|_| !force_host_moe) {
            // On-device path: select the top-1 expert and run it via the indexed
            // OQ8 GEMV kernels — no per-block GPU→host router readback. The null
            // slot (`n_exp`) is dereference-safe (aliases expert 0) and gated to
            // zero weight by `zaya_router_select_f32`, so the combine adds nothing.
            if !router_fused {
                gpu.zaya_router_select_f32(
                    &rlogits,
                    &idx.balancing_biases_dev,
                    &sel_idx,
                    &sel_gate,
                    n_route,
                )
                .map_err(|e| format!("{e:?}"))?;
            }
            let g = gate_up.sub_offset(0, moe_int);
            let u = gate_up.sub_offset(moe_int, moe_int);
            // Reuse the rotated post-attn activation from the down_proj step when
            // available (same `normed`, already FWHT-rotated); else rotate here.
            if !w8a8_pa {
                gpu.rotate_x_mq(&normed, &xr_norm, h)
                    .map_err(|e| format!("{e:?}"))?;
            }
            let gate_up_x = if w8a8_pa { &pa_xrot } else { &xr_norm };
            if !ablate_moe {
                if moe_w8a8 && w8a8_pa {
                    // gate_up_x == pa_xrot here; reuse its int8 quant (pa_xq/pa_xs).
                    gpu.zaya_moe_gate_up_oq8_planar_indexed_w8a8(
                        &idx.gate_up_ptrs,
                        &sel_idx,
                        &pa_xq,
                        &pa_xs,
                        &g,
                        &u,
                        2 * moe_int,
                        h,
                    )
                    .map_err(|e| format!("{e:?}"))?;
                } else {
                    gpu.zaya_moe_gate_up_oq8_planar_indexed(
                        &idx.gate_up_ptrs,
                        &sel_idx,
                        gate_up_x,
                        &g,
                        &u,
                        2 * moe_int,
                        h,
                    )
                    .map_err(|e| format!("{e:?}"))?;
                }
            }
            // Fused silu(g)*u + MQ-rotate → xr_act (one launch, replaces
            // silu_mul_f32 + rotate_x_mq). down_proj weights live in the MQ-rotated
            // basis, so the fused rotate feeds the indexed OQ8 down GEMV directly.
            gpu.fused_silu_mul_rotate_mq(&g, &u, &xr_act, moe_int)
                .map_err(|e| format!("{e:?}"))?;
            if !ablate_moe {
                if moe_w8a8 && w8a8_pa {
                    gpu.quantize_act_oq8(&xr_act, &moe_xq_dn, &moe_xs_dn, 1, moe_int, 256)
                        .map_err(|e| format!("{e:?}"))?;
                    gpu.zaya_moe_down_oq8_planar_indexed_w8a8(
                        &idx.down_ptrs,
                        &sel_idx,
                        &moe_xq_dn,
                        &moe_xs_dn,
                        &down_t,
                        h,
                        moe_int,
                    )
                    .map_err(|e| format!("{e:?}"))?;
                } else {
                    gpu.zaya_moe_down_oq8_planar_indexed(
                        &idx.down_ptrs,
                        &sel_idx,
                        &xr_act,
                        &down_t,
                        h,
                        moe_int,
                    )
                    .map_err(|e| format!("{e:?}"))?;
                }
            }
            // moe_out += sel_gate * down_t (top-1 → single term).
            gpu.moe_down_combine_k8_batched(&down_t, &sel_gate, &moe_out, h, 1, 1)
                .map_err(|e| format!("{e:?}"))?;
        } else {
            // Host reference path (f32 calibration / non-Oq8G256 experts).
            let row = gpu.download_f32(&rlogits).map_err(|e| format!("{e:?}"))?;
            let maxv = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut probs = vec![0f32; n_route];
            let mut denom = 0f32;
            for e in 0..n_route {
                probs[e] = (row[e] - maxv).exp();
                denom += probs[e];
            }
            for p in probs.iter_mut() {
                *p /= denom;
            }
            let mut best = 0usize;
            let mut bestv = f32::NEG_INFINITY;
            for e in 0..n_route {
                let v = probs[e] + lw.balancing_biases[e];
                if v > bestv {
                    bestv = v;
                    best = e;
                }
            }
            if best != n_exp {
                let weight = probs[best];
                let ex = &lw.experts[best];
                ex.gate_up.gemv(gpu, &normed, &gate_up)?;
                let g = gate_up.sub_offset(0, moe_int);
                let u = gate_up.sub_offset(moe_int, moe_int);
                gpu.silu_mul_f32(&g, &u, &act)
                    .map_err(|e| format!("{e:?}"))?;
                ex.down.gemv(gpu, &act, &down_t)?;
                gpu.scaled_add_inplace_cpu_scalar_f32(&moe_out, &down_t, weight)
                    .map_err(|e| format!("{e:?}"))?;
            }
        }
        gpu.zaya_affine_residual_f32(
            hidden,
            &moe_out,
            &g_res2,
            &lw.pm_rs[0],
            &lw.pm_rs[1],
            &lw.pm_rs[2],
            &lw.pm_rs[3],
            h,
            h,
        )
        .map_err(|e| format!("{e:?}"))?;
        // Megakernel-B validation: compare the fused `hidden` (in hidden2) to the
        // reference `hidden` just written, per layer.
        if mega_validate && w8a8_pa && !force_host_moe && lw.moe_indexed.is_some() {
            let a = gpu.download_f32(hidden).map_err(|e| format!("{e:?}"))?;
            let b = gpu
                .download_f32(hidden2.as_ref().unwrap())
                .map_err(|e| format!("{e:?}"))?;
            let (mut dot, mut na, mut nb, mut maxabs) = (0f64, 0f64, 0f64, 0f64);
            for i in 0..h {
                let (x, y) = (a[i] as f64, b[i] as f64);
                dot += x * y;
                na += x * x;
                nb += y * y;
                maxabs = maxabs.max((x - y).abs());
            }
            let cos = dot / (na.sqrt() * nb.sqrt()).max(1e-30);
            eprintln!("[zaya-mega-validate] pos={pos} li={li} cos={cos:.6} maxabs={maxabs:.3e}");
        }
        gpu.reclaim_pending();
    }

    let sec_body = sec_t0.map(|t| {
        let _ = gpu.hip.device_synchronize();
        t.elapsed().as_micros()
    });
    let sec_t1 = if sectiontime {
        Some(std::time::Instant::now())
    } else {
        None
    };
    gpu.rmsnorm_f32(hidden, &w.norm, &fnorm, eps)
        .map_err(|e| format!("{e:?}"))?;
    if !ablate.contains("lmhead") {
        // Two-stage lm_head (HIPFIRE_ZAYA_LMHEAD=q4|q4c|q2|q2c): coarse row-norm
        // Q-scorer shortlists the top-K, an exact bf16 fine pass rescores just those
        // and scatters into a -inf-masked logit vector. Greedy-exact at the measured
        // K; ~1.6–2.5 ms vs the ~6.3 ms full bf16 lm_head. See docs/kernel_work.
        if let Some((bits, corr_r, topk)) = parse_lmhead_twostage() {
            lmhead_twostage_serve(gpu, w, &fnorm, logits_out, state, bits, corr_r, topk)?;
        } else if std::env::var("HIPFIRE_ZAYA_F16_LMHEAD").is_ok() {
            // Untied F16 lm_head (HIPFIRE_ZAYA_F16_LMHEAD): the tied embed is F32
            // (2.15 GB) and dominates decode (~41%). The input gather keeps F32; the
            // output projection reads a one-time F16 copy (~1.07 GB) → ~2× less
            // lm_head bandwidth at negligible quality cost. Demonstrates plan
            // workstream B (the untied lower-precision output projection) without a
            // requantize; the production path is oq4 LDLQ in the quantizer.
            if state.lmhead_f16.is_none() {
                let embuf = w.embed.buf();
                let numel: usize = embuf.shape.iter().product();
                let f16 = gpu
                    .alloc_tensor(&[numel], DType::F16)
                    .map_err(|e| format!("zaya lmhead_f16 alloc: {e:?}"))?;
                gpu.convert_f32_to_f16_into(embuf, &f16, numel)
                    .map_err(|e| format!("zaya lmhead f32→f16: {e:?}"))?;
                state.lmhead_f16 = Some(f16);
            }
            let f16 = state.lmhead_f16.as_ref().unwrap();
            let vocab = f16.shape.iter().product::<usize>() / h;
            gpu.gemv_f16_xf32(f16, &fnorm, logits_out, vocab, h)
                .map_err(|e| format!("zaya f16 lm_head: {e:?}"))?;
        } else {
            w.embed
                .gemv(gpu, &fnorm, logits_out)
                .map_err(|e| format!("zaya lm_head: {e}"))?;
        }
    }
    if let (Some(b), Some(t1)) = (sec_body, sec_t1) {
        let _ = gpu.hip.device_synchronize();
        let lm = t1.elapsed().as_micros();
        eprintln!("[zaya-sectiontime] pos={pos} body={b}us lmhead={lm}us");
    }
    // EXP-21: does the PRODUCTION lm_head gemv ramp in a tight back-to-back loop?
    // If it hits ~200 GB/s here but ~50 in decode → idle-induced clock drop; if it
    // stays ~50 → the kernel's access pattern is the ceiling.
    static LMLOOP: std::sync::Once = std::sync::Once::new();
    if std::env::var("HIPFIRE_ZAYA_LMHEAD_LOOP").is_ok() && !LMLOOP.is_completed() {
        LMLOOP.call_once(|| {});
        let iters = 300;
        let _ = gpu.hip.device_synchronize();
        let t = std::time::Instant::now();
        for _ in 0..iters {
            w.embed
                .gemv(gpu, &fnorm, logits_out)
                .map_err(|e| format!("zaya lm_head loop: {e}"))?;
        }
        let _ = gpu.hip.device_synchronize();
        let us = t.elapsed().as_micros() as f64 / iters as f64;
        // Q8 lm_head ≈ 537 MB (vocab 262272 × hidden 2048, ~1 byte/weight + scales).
        let mb = 537.0;
        eprintln!(
            "[zaya-lmhead-loop] embed_dtype={:?} {us:.1} us/gemv over {iters} back-to-back iters → ~{:.0} GB/s (~{mb} MB)",
            w.embed.quant_dtype(),
            mb / us * 1000.0
        );
    }
    if std::env::var("HIPFIRE_ZAYA_LMHEAD_SHORTLIST").is_ok() && !ablate.contains("lmhead") {
        lmhead_shortlist_measure(gpu, w, &fnorm, logits_out, state, pos)?;
    }
    gpu.reclaim_pending();
    Ok(())
}

/// Top-r eigenvectors of a symmetric PSD gram `G [n×n]` via a randomized range
/// finder (Halko et al.): returns `proj [r, n]` row-major (row j = eigenvector j =
/// right singular direction j). Avoids the full O(n³) eigendecomp — one small l×l
/// Jacobi + a couple of gram matmuls. l = r + oversample.
fn randomized_topr_eigvecs(gram: &[f64], n: usize, r: usize) -> Vec<f32> {
    use rayon::prelude::*;
    let l = (r + 32).min(n);
    // C[rows×cols] = A[rows×k] · B[k×cols] (rayon over rows).
    let mm = |a: &[f64], b: &[f64], rows: usize, k: usize, cols: usize| -> Vec<f64> {
        (0..rows)
            .into_par_iter()
            .flat_map(|i| {
                let ai = &a[i * k..i * k + k];
                (0..cols)
                    .map(|j| {
                        let mut s = 0.0f64;
                        for t in 0..k {
                            s += ai[t] * b[t * cols + j];
                        }
                        s
                    })
                    .collect::<Vec<f64>>()
            })
            .collect()
    };
    // Random Ω [n, l].
    let mut omega = vec![0.0f64; n * l];
    let mut s: u64 = 0xDEAD_BEEF_1234_5678;
    for x in omega.iter_mut() {
        s = s.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^= z >> 31;
        let u1 = (((z >> 11) as f64) / ((1u64 << 53) as f64)).max(1e-12);
        let u2 = ((z & 0xFFFFF) as f64) / ((1u64 << 20) as f64);
        *x = (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
    }
    // Range: Q ← orthonormal cols of (G² Ω) (one power iteration for accuracy).
    let mut q = mm(gram, &omega, n, n, l);
    q = mm(gram, &q, n, n, l);
    // Modified Gram-Schmidt on the l columns of q [n, l].
    for j in 0..l {
        let mut nrm = 0.0f64;
        for i in 0..n {
            nrm += q[i * l + j] * q[i * l + j];
        }
        nrm = nrm.sqrt();
        if nrm > 1e-12 {
            let inv = 1.0 / nrm;
            for i in 0..n {
                q[i * l + j] *= inv;
            }
        }
        for k2 in (j + 1)..l {
            let mut dot = 0.0f64;
            for i in 0..n {
                dot += q[i * l + j] * q[i * l + k2];
            }
            for i in 0..n {
                q[i * l + k2] -= dot * q[i * l + j];
            }
        }
    }
    // T = Qᵀ G Q [l, l]; eigendecompose the small T.
    let gq = mm(gram, &q, n, n, l);
    let mut t = vec![0.0f64; l * l];
    for a in 0..l {
        for b in 0..l {
            let mut acc = 0.0f64;
            for i in 0..n {
                acc += q[i * l + a] * gq[i * l + b];
            }
            t[a * l + b] = acc;
        }
    }
    let (_ev, tv) = hipfire_kvquant::lowrank::jacobi_eig(&t, l); // eigvals desc, eigvecs = cols
                                                                 // V_r = Q · tv[:, :r]; proj row j = V_r col j.
    let mut proj = vec![0.0f32; r * n];
    for j in 0..r {
        for h in 0..n {
            let mut acc = 0.0f64;
            for c in 0..l {
                acc += q[h * l + c] * tv[c * l + j];
            }
            proj[j * n + h] = acc as f32;
        }
    }
    proj
}

/// Build the two-stage lm_head coarse scorer once: a row-wise L2-normalized,
/// 3σ-clipped Q`bits` copy of the bf16 embed, optionally H→`proj_r` projected
/// (`use_svd` = top-r right singular basis, else Gaussian JL), plus an optional
/// rank-`correct_r` low-rank residual correction (full-H only). Parallel host build.
fn build_lmhead_coarse(
    gpu: &mut Gpu,
    w: &ZayaGpuWeights,
    bits: usize,
    proj_r: usize,
    use_svd: bool,
    correct_r: usize,
) -> Result<LmheadCoarse, String> {
    use rayon::prelude::*;
    let (vocab, hidden) = match w.embed.wt_mk() {
        Some((_, m, k)) => (m, k),
        None => return Err("zaya coarse: embed is not quant-backed".into()),
    };
    let r = if proj_r > 0 && proj_r < hidden && proj_r % 2 == 0 {
        proj_r
    } else {
        0
    };
    // The low-rank residual correction is defined against the full-H quantizer only.
    let correct_r = if r > 0 { 0 } else { correct_r };
    let embuf = w.embed.wt_mk().unwrap().0;
    let bytes = gpu
        .download_raw(embuf, vocab * hidden * 2)
        .map_err(|e| format!("zaya coarse download: {e:?}"))?;
    let build_t = std::time::Instant::now();
    let (kdim, proj) = if r > 0 {
        let p = if use_svd {
            let svd_t = std::time::Instant::now();
            // Subsample the unit directions into a COLUMN-major matrix Us [H, S]
            // (us_col[a*S + s] = û_s[a]), then gram G[a,b] = Σ_s Us[a,s]·Us[b,s]
            // via parallel dot products over the H rows of G (no huge-accumulator
            // fold, which thrashes). S=10k is plenty for the top-r directions.
            let s_sub = 10000usize.min(vocab);
            let stride = (vocab / s_sub).max(1);
            let mut us_col = vec![0f32; hidden * s_sub];
            for si in 0..s_sub {
                let v = si * stride;
                let row = &bytes[v * hidden * 2..(v + 1) * hidden * 2];
                let mut nrm = 0f32;
                let mut u = vec![0f32; hidden];
                for i in 0..hidden {
                    let f = f32::from_bits(
                        (u16::from_le_bytes([row[2 * i], row[2 * i + 1]]) as u32) << 16,
                    );
                    u[i] = f;
                    nrm += f * f;
                }
                let ni = if nrm > 0.0 { 1.0 / nrm.sqrt() } else { 0.0 };
                for a in 0..hidden {
                    us_col[a * s_sub + si] = u[a] * ni;
                }
            }
            let gram: Vec<f64> = (0..hidden)
                .into_par_iter()
                .flat_map(|a| {
                    let ra = &us_col[a * s_sub..a * s_sub + s_sub];
                    (0..hidden)
                        .map(|b| {
                            let rb = &us_col[b * s_sub..b * s_sub + s_sub];
                            let mut acc = 0f64;
                            for s in 0..s_sub {
                                acc += (ra[s] * rb[s]) as f64;
                            }
                            acc
                        })
                        .collect::<Vec<f64>>()
                })
                .collect();
            // Top-r eigenvectors of the gram (= right singular basis V_r) via a
            // randomized range finder — avoids a full 2048² eigendecomp.
            let p = randomized_topr_eigvecs(&gram, hidden, r);
            eprintln!(
                "[zaya-shortlist] SVD proj r={r} from {s_sub}-row gram: {}ms",
                svd_t.elapsed().as_millis()
            );
            p
        } else {
            let mut p = vec![0f32; r * hidden];
            let mut s: u64 = 0x9E3779B97F4A7C15;
            for x in p.iter_mut() {
                s = s.wrapping_add(0x9E3779B97F4A7C15);
                let mut z = s;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
                z ^= z >> 31;
                let u1 = (((z >> 11) as f64) / ((1u64 << 53) as f64)).max(1e-12);
                let u2 = ((z & 0xFFFFF) as f64) / ((1u64 << 20) as f64);
                *x = ((-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()) as f32;
            }
            p
        };
        (r, Some(p))
    } else {
        (hidden, None)
    };
    // Coarse bit-width (HIPFIRE_ZAYA_SHORTLIST_BITS = 4 default | 2). Global
    let (lo, hi, max_mag, per_byte) = if bits == 2 {
        (-2.0f32, 1.0f32, 2.0f32, 4usize)
    } else {
        (-7.0f32, 7.0f32, 7.0f32, 2usize)
    };
    let unit_scale = 3.0f32 / (max_mag * (kdim as f32).sqrt());
    let inv = 1.0 / unit_scale;
    let kb = kdim / per_byte;
    let rows: Vec<(Vec<u8>, f32)> = (0..vocab)
        .into_par_iter()
        .map(|v| {
            let row = &bytes[v * hidden * 2..(v + 1) * hidden * 2];
            let mut wv = vec![0f32; hidden];
            for i in 0..hidden {
                let u = u16::from_le_bytes([row[2 * i], row[2 * i + 1]]);
                wv[i] = f32::from_bits((u as u32) << 16);
            }
            let u: Vec<f32> = match &proj {
                Some(p) => (0..kdim)
                    .map(|i| {
                        let pr = &p[i * hidden..(i + 1) * hidden];
                        let mut acc = 0f32;
                        for h in 0..hidden {
                            acc += pr[h] * wv[h];
                        }
                        acc
                    })
                    .collect(),
                None => wv,
            };
            let norm = u.iter().map(|&x| x * x).sum::<f32>().sqrt();
            let mut q = vec![0u8; kb];
            if norm > 0.0 {
                let ni = inv / norm;
                let qz = |d: usize| ((u[d] * ni).round().clamp(lo, hi) as i32) as u8;
                if bits == 2 {
                    for i in 0..kb {
                        q[i] = (qz(4 * i) & 0x3)
                            | ((qz(4 * i + 1) & 0x3) << 2)
                            | ((qz(4 * i + 2) & 0x3) << 4)
                            | ((qz(4 * i + 3) & 0x3) << 6);
                    }
                } else {
                    for i in 0..kb {
                        q[i] = (qz(2 * i) & 0xF) | ((qz(2 * i + 1) & 0xF) << 4);
                    }
                }
            }
            (q, norm * unit_scale)
        })
        .collect();
    let mut q4 = vec![0u8; vocab * kb];
    let mut scales = vec![0f32; vocab];
    for (v, (q, s)) in rows.into_iter().enumerate() {
        q4[v * kb..(v + 1) * kb].copy_from_slice(&q);
        scales[v] = s;
    }
    let q4buf = gpu
        .upload_raw(&q4, &[q4.len()])
        .map_err(|e| format!("{e:?}"))?;
    let scbuf = gpu
        .upload_f32(&scales, &[vocab])
        .map_err(|e| format!("{e:?}"))?;
    let projbuf = match proj {
        Some(p) => Some(
            gpu.upload_f32(&p, &[kdim, hidden])
                .map_err(|e| format!("{e:?}"))?,
        ),
        None => None,
    };
    eprintln!(
            "[zaya-shortlist] built coarse V={vocab} kdim={kdim} proj={} rownorm+3σ-clip Q{bits}: {}MB (bf16 fine={}MB) build={}ms",
            r > 0,
            vocab * kb / 1_000_000,
            vocab * hidden * 2 / 1_000_000,
            build_t.elapsed().as_millis()
        );
    let (corr_a, corr_b) = if correct_r > 0 {
        let ct = std::time::Instant::now();
        // Residual row D[h] = norm·(û[h] − unit_scale·qint[h]), recomputed from bytes.
        let resid = |v: usize| -> Vec<f32> {
            let row = &bytes[v * hidden * 2..(v + 1) * hidden * 2];
            let mut u = vec![0f32; hidden];
            let mut nrm = 0f32;
            for i in 0..hidden {
                let f =
                    f32::from_bits((u16::from_le_bytes([row[2 * i], row[2 * i + 1]]) as u32) << 16);
                u[i] = f;
                nrm += f * f;
            }
            nrm = nrm.sqrt();
            let mut d = vec![0f32; hidden];
            if nrm > 0.0 {
                let ni = inv / nrm;
                for h in 0..hidden {
                    let q = (u[h] * ni).round().clamp(lo, hi);
                    d[h] = nrm * (u[h] / nrm - unit_scale * q);
                }
            }
            d
        };
        let s_sub = 10000usize.min(vocab);
        let stride = (vocab / s_sub).max(1);
        let mut ds_col = vec![0f32; hidden * s_sub];
        for si in 0..s_sub {
            let d = resid(si * stride);
            for h in 0..hidden {
                ds_col[h * s_sub + si] = d[h];
            }
        }
        let gram: Vec<f64> = (0..hidden)
            .into_par_iter()
            .flat_map(|a| {
                let ra = &ds_col[a * s_sub..a * s_sub + s_sub];
                (0..hidden)
                    .map(|b| {
                        let rb = &ds_col[b * s_sub..b * s_sub + s_sub];
                        let mut acc = 0f64;
                        for s in 0..s_sub {
                            acc += (ra[s] * rb[s]) as f64;
                        }
                        acc
                    })
                    .collect::<Vec<f64>>()
            })
            .collect();
        let b = randomized_topr_eigvecs(&gram, hidden, correct_r); // [r, H]
        let a: Vec<f32> = (0..vocab)
            .into_par_iter()
            .flat_map(|v| {
                let d = resid(v);
                (0..correct_r)
                    .map(|j| {
                        let bj = &b[j * hidden..j * hidden + hidden];
                        let mut acc = 0f32;
                        for h in 0..hidden {
                            acc += d[h] * bj[h];
                        }
                        acc
                    })
                    .collect::<Vec<f32>>()
            })
            .collect();
        eprintln!(
            "[zaya-shortlist] Stage-3 correction r={correct_r}: A[{vocab},{correct_r}] build={}ms",
            ct.elapsed().as_millis()
        );
        let abuf = gpu
            .upload_f32(&a, &[vocab, correct_r])
            .map_err(|e| format!("{e:?}"))?;
        let bbuf = gpu
            .upload_f32(&b, &[correct_r, hidden])
            .map_err(|e| format!("{e:?}"))?;
        (Some(abuf), Some(bbuf))
    } else {
        (None, None)
    };
    Ok(LmheadCoarse {
        q4: q4buf,
        scales: scbuf,
        proj: projbuf,
        kdim,
        bits,
        corr_a,
        corr_b,
        corr_r: correct_r,
    })
}

/// Score every vocab row with the coarse scorer (+ optional low-rank correction),
/// returning the host score vector and the on-GPU coarse latency (µs).
fn coarse_score_gpu(
    gpu: &mut Gpu,
    c: &LmheadCoarse,
    fnorm: &GpuTensor,
    vocab: usize,
    hidden: usize,
) -> Result<GpuTensor, String> {
    let kdim = c.kdim;
    let bits = c.bits;
    let corr_r = c.corr_r;
    let corr_av = c.corr_a.as_ref().map(|a| GpuTensor {
        buf: unsafe { hip_bridge::DeviceBuffer::from_raw(a.buf.as_ptr(), vocab * corr_r * 4) },
        shape: vec![vocab, corr_r],
        dtype: DType::F32,
    });
    let corr_bv = c.corr_b.as_ref().map(|b| GpuTensor {
        buf: unsafe { hip_bridge::DeviceBuffer::from_raw(b.buf.as_ptr(), corr_r * hidden * 4) },
        shape: vec![corr_r, hidden],
        dtype: DType::F32,
    });
    let q4bytes = vocab * kdim * bits / 8;
    let q4v = GpuTensor {
        buf: unsafe { hip_bridge::DeviceBuffer::from_raw(c.q4.buf.as_ptr(), q4bytes) },
        shape: vec![q4bytes],
        dtype: DType::Raw,
    };
    let scv = GpuTensor {
        buf: unsafe { hip_bridge::DeviceBuffer::from_raw(c.scales.buf.as_ptr(), vocab * 4) },
        shape: vec![vocab],
        dtype: DType::F32,
    };
    let proj_view = c.proj.as_ref().map(|p| GpuTensor {
        buf: unsafe { hip_bridge::DeviceBuffer::from_raw(p.buf.as_ptr(), kdim * hidden * 4) },
        shape: vec![kdim, hidden],
        dtype: DType::F32,
    });
    let coarse = gpu
        .zeros(&[vocab], DType::F32)
        .map_err(|e| format!("{e:?}"))?;
    // Project h → h_r if dimensionality-reduced, then the coarse Q4/Q2 gemv.
    let hin = match &proj_view {
        Some(pv) => {
            let hr = gpu
                .zeros(&[kdim], DType::F32)
                .map_err(|e| format!("{e:?}"))?;
            gpu.gemv_f32(pv, fnorm, &hr)
                .map_err(|e| format!("zaya coarse proj: {e:?}"))?;
            hr
        }
        None => GpuTensor {
            buf: unsafe { hip_bridge::DeviceBuffer::from_raw(fnorm.buf.as_ptr(), hidden * 4) },
            shape: vec![hidden],
            dtype: DType::F32,
        },
    };
    if bits == 2 {
        gpu.gemv_q2sym_f32(&q4v, &scv, &hin, &coarse, vocab, kdim)
            .map_err(|e| format!("zaya coarse gemv q2: {e:?}"))?;
    } else {
        gpu.gemv_q4sym_f32(&q4v, &scv, &hin, &coarse, vocab, kdim)
            .map_err(|e| format!("zaya coarse gemv: {e:?}"))?;
    }
    // Stage-3 correction added ON-DEVICE: coarse += A·(B·fnorm).
    if let (Some(av), Some(bv)) = (&corr_av, &corr_bv) {
        let bh = gpu
            .zeros(&[corr_r], DType::F32)
            .map_err(|e| format!("{e:?}"))?;
        gpu.gemv_f32(bv, fnorm, &bh)
            .map_err(|e| format!("zaya corr B·h: {e:?}"))?;
        let corr = gpu
            .zeros(&[vocab], DType::F32)
            .map_err(|e| format!("{e:?}"))?;
        gpu.gemv_f32(av, &bh, &corr)
            .map_err(|e| format!("zaya corr A·bh: {e:?}"))?;
        gpu.add_inplace_f32(&coarse, &corr)
            .map_err(|e| format!("zaya corr add: {e:?}"))?;
        let _ = gpu.free_tensor(bh);
        let _ = gpu.free_tensor(corr);
    }
    if proj_view.is_some() {
        let _ = gpu.free_tensor(hin);
    }
    Ok(coarse)
}

/// Coarse scores on the HOST (diagnostic path): score on GPU then download. Returns the
/// host score vector + the on-GPU coarse latency (µs).
fn coarse_scores_host(
    gpu: &mut Gpu,
    c: &LmheadCoarse,
    fnorm: &GpuTensor,
    vocab: usize,
    hidden: usize,
) -> Result<(Vec<f32>, u128), String> {
    let _ = gpu.hip.device_synchronize();
    let t0 = std::time::Instant::now();
    let coarse = coarse_score_gpu(gpu, c, fnorm, vocab, hidden)?;
    let _ = gpu.hip.device_synchronize();
    let coarse_us = t0.elapsed().as_micros();
    let cv = gpu.download_f32(&coarse).map_err(|e| format!("{e:?}"))?;
    let _ = gpu.free_tensor(coarse);
    Ok((cv, coarse_us))
}

/// Device top-K over `coarse` [V]: min/max → histogram → threshold scan → compact.
/// Returns (idx device buffer, count) — a SUPERSET of the exact top-`kk` (the fine bf16
/// pass rescores exactly, so extra candidates are harmless). Replaces the host download
/// + `select_nth`: only three tiny scalars (min/max, histogram, count) cross to the host.
fn gpu_topk(
    gpu: &mut Gpu,
    coarse: &GpuTensor,
    vocab: usize,
    kk: usize,
) -> Result<(GpuTensor, usize), String> {
    const NBINS: usize = 4096;
    let kk = kk.min(vocab).max(1);
    // Folded stats buffer: [0..NBINS) histogram bins (zeroed) | [NBINS] min key | [NBINS+1]
    // max key. min/max writes the tail; the histogram reads lo/hi from it on-device — so
    // the whole top-K needs ONE host download (the histogram), not a round-trip per pass.
    let stats = gpu
        .zeros(&[NBINS + 2], DType::F32)
        .map_err(|e| format!("{e:?}"))?;
    // Init the min slot to 0xFFFFFFFF so atomicMin reduces it (max slot stays 0 from zeros).
    let lo_slot = stats.sub_offset(NBINS, 1);
    gpu.hip
        .memset(&lo_slot.buf, 0xFF, 4)
        .map_err(|e| format!("{e:?}"))?;
    gpu.lmhead_coarse_minmax(coarse, &stats, vocab, NBINS)
        .map_err(|e| format!("zaya topk minmax: {e:?}"))?;
    gpu.lmhead_coarse_hist(coarse, &stats, vocab, NBINS)
        .map_err(|e| format!("zaya topk hist: {e:?}"))?;
    let _ = gpu.hip.device_synchronize();
    let sb = gpu
        .download_raw(&stats, (NBINS + 2) * 4)
        .map_err(|e| format!("{e:?}"))?;
    let _ = gpu.free_tensor(stats);
    let rd =
        |i: usize| u32::from_le_bytes([sb[i * 4], sb[i * 4 + 1], sb[i * 4 + 2], sb[i * 4 + 3]]);
    let lo = rd(NBINS);
    let hi = rd(NBINS + 1);
    // scan bins top-down until the cumulative count reaches kk → threshold τ.
    let mut acc = 0usize;
    let mut boundary = 0usize;
    for b in (0..NBINS).rev() {
        acc += rd(b) as usize;
        if acc >= kk {
            boundary = b;
            break;
        }
    }
    let range = (hi as u64) - (lo as u64) + 1;
    let tau = (lo as u64 + (boundary as u64) * range / (NBINS as u64)) as u32;
    let cap = acc + 512; // count == acc up to boundary integer-division rounding; +slack.
                         // Sentinel-fill idx (0xFFFFFFFF), compact key≥τ rows into it, and let the fine gather
                         // run over all `cap` slots skipping sentinels — so no count round-trip is needed.
    let idxbuf = gpu
        .zeros(&[cap], DType::F32)
        .map_err(|e| format!("{e:?}"))?;
    gpu.hip
        .memset(&idxbuf.buf, 0xFF, cap * 4)
        .map_err(|e| format!("{e:?}"))?;
    let counter = gpu.zeros(&[1], DType::F32).map_err(|e| format!("{e:?}"))?; // device write-cursor for compact (not read back)
    gpu.lmhead_coarse_compact(coarse, &idxbuf, &counter, vocab, tau, cap)
        .map_err(|e| format!("zaya topk compact: {e:?}"))?;
    let _ = gpu.free_tensor(counter);
    Ok((idxbuf, cap))
}

/// Parse `HIPFIRE_ZAYA_LMHEAD` → (bits, correction rank, top-K) for the two-stage
/// serving lm_head, or None for the default full bf16 path. Presets: `q4` (Q4
/// coarse, K=32), `q4c` (+rank-64 correction), `q2` (Q2 coarse, K=2048), `q2c`
/// (Q2 + correction, K=2048). `HIPFIRE_ZAYA_LMHEAD_K` / `_CORR` override the defaults.
fn parse_lmhead_twostage() -> Option<(usize, usize, usize)> {
    let v = std::env::var("HIPFIRE_ZAYA_LMHEAD")
        .ok()?
        .to_ascii_lowercase();
    let bits = if v.contains("q2") {
        2
    } else if v.contains("q4") {
        4
    } else {
        return None;
    };
    let corr_r = if v.contains('c') {
        std::env::var("HIPFIRE_ZAYA_LMHEAD_CORR")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(64)
    } else {
        0
    };
    let default_k = if bits == 4 { 32 } else { 2048 };
    let topk = std::env::var("HIPFIRE_ZAYA_LMHEAD_K")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(default_k);
    Some((bits, corr_r, topk))
}

/// Two-stage lm_head SERVING path. Coarse-score all V rows with the row-norm
/// Q`bits` (+ optional rank-`corr_r` correction) scorer, host-select the top-`topk`,
/// then rescore exactly those rows at bf16 and scatter into a -inf-masked
/// `logits_out`. Greedy-exact when the coarse recall@1 = 100% at `topk` (measured).
fn lmhead_twostage_serve(
    gpu: &mut Gpu,
    w: &ZayaGpuWeights,
    fnorm: &GpuTensor,
    logits_out: &GpuTensor,
    state: &mut ZayaDecodeState,
    bits: usize,
    corr_r: usize,
    topk: usize,
) -> Result<(), String> {
    let (vocab, hidden) = match w.embed.wt_mk() {
        Some((_, m, k)) => (m, k),
        // Non-quant (f32) embed: fall back to the exact full lm_head.
        None => {
            return w
                .embed
                .gemv(gpu, fnorm, logits_out)
                .map_err(|e| format!("zaya lm_head: {e}"))
        }
    };
    if state.lmhead_coarse.is_none() {
        state.lmhead_coarse = Some(build_lmhead_coarse(gpu, w, bits, 0, false, corr_r)?);
    }
    let c = state.lmhead_coarse.as_ref().unwrap();
    // GPU top-K serving path (default): the coarse score + correction stay on-device and
    // a device histogram-select produces the shortlist — no 1MB score download, no host
    // `select_nth`. The host packed-key select is kept behind HIPFIRE_ZAYA_LMHEAD_HOSTSELECT
    // for A/B comparison. Both are greedy-exact (the fine bf16 gather rescores the shortlist).
    if std::env::var("HIPFIRE_ZAYA_LMHEAD_HOSTSELECT").is_err() {
        let kk = topk.min(vocab).max(1);
        let coarse = coarse_score_gpu(gpu, c, fnorm, vocab, hidden)?;
        let (idxbuf, count) = gpu_topk(gpu, &coarse, vocab, kk)?;
        let _ = gpu.free_tensor(coarse);
        let xb = gpu
            .alloc_tensor(&[hidden], DType::BF16)
            .map_err(|e| format!("zaya fine xb: {e:?}"))?;
        gpu.cast_f32_to_bf16(fnorm, &xb)
            .map_err(|e| format!("zaya fine cast: {e:?}"))?;
        gpu.fill_f32(logits_out, f32::NEG_INFINITY)
            .map_err(|e| format!("zaya mask: {e:?}"))?;
        let embuf = w.embed.wt_mk().unwrap().0;
        gpu.gemv_bf16_gather_f32(embuf, &idxbuf, &xb, logits_out, count, hidden)
            .map_err(|e| format!("zaya fine gather: {e:?}"))?;
        let _ = gpu.free_tensor(idxbuf);
        let _ = gpu.free_tensor(xb);
        return Ok(());
    }
    let timing = std::env::var("HIPFIRE_ZAYA_LMHEAD_TIMING").is_ok();
    let tc = std::time::Instant::now();
    let (cv, gpu_us) = coarse_scores_host(gpu, c, fnorm, vocab, hidden)?;
    let t_coarse = tc.elapsed().as_micros();
    // Host top-K over the coarse scores -> shortlist row indices. Pack each score
    // into an order-preserving u64 key `(monotone_f32_bits << 32) | idx` and select
    // on the u64 directly — no comparator closure / double-indirection into `cv`
    // (that indexed compare cost ~468µs over V=262k; the packed select is ~5× faster).
    let ts = std::time::Instant::now();
    let kk = topk.min(vocab).max(1);
    // Host top-K over the coarse scores -> shortlist row indices. Pack each score into
    // an order-preserving u64 key `(monotone_f32_bits << 32) | idx` and select on the
    // u64 directly — no comparator closure / double-indirection into `cv`. Kept SERIAL:
    // a chunked-rayon variant (per-chunk partial top-K + merge) added run-to-run jitter
    // (310–553µs vs a stable ~303µs) and regressed at large K, since post-coarse-gemv
    // fix this select is memory-bound and no longer the critical cost.
    let mut keyed: Vec<u64> = (0..vocab)
        .map(|i| {
            let b = cv[i].to_bits();
            let mono = if b & 0x8000_0000 != 0 {
                !b
            } else {
                b | 0x8000_0000
            };
            ((mono as u64) << 32) | i as u64
        })
        .collect();
    // Ascending select: the top-kk are the largest kk keys (the tail).
    let split = vocab - kk;
    keyed.select_nth_unstable(split);
    let t_select = ts.elapsed().as_micros();
    if timing {
        eprintln!("[zaya-lmhead-timing] gpu_coarse_gemv={gpu_us}us wall_incl_bodydrain={t_coarse}us host_select={t_select}us kk={kk}");
    }
    let idx_bytes: Vec<u8> = keyed[split..]
        .iter()
        .flat_map(|&k| (k as u32).to_le_bytes())
        .collect();
    let idxbuf = gpu
        .upload_raw(&idx_bytes, &[kk])
        .map_err(|e| format!("zaya shortlist idx: {e:?}"))?;
    // Cast fnorm->bf16 to match the full bf16 lm_head's arithmetic for the fine pass.
    let xb = gpu
        .alloc_tensor(&[hidden], DType::BF16)
        .map_err(|e| format!("zaya fine xb: {e:?}"))?;
    gpu.cast_f32_to_bf16(fnorm, &xb)
        .map_err(|e| format!("zaya fine cast: {e:?}"))?;
    // Mask all vocab to -inf, then scatter exact bf16 logits for the shortlist rows.
    gpu.fill_f32(logits_out, f32::NEG_INFINITY)
        .map_err(|e| format!("zaya mask: {e:?}"))?;
    let embuf = w.embed.wt_mk().unwrap().0;
    gpu.gemv_bf16_gather_f32(embuf, &idxbuf, &xb, logits_out, kk, hidden)
        .map_err(|e| format!("zaya fine gather: {e:?}"))?;
    let _ = gpu.free_tensor(idxbuf);
    let _ = gpu.free_tensor(xb);
    Ok(())
}

/// Two-stage lm_head DIAGNOSTIC (`HIPFIRE_ZAYA_LMHEAD_SHORTLIST`): build the coarse
/// scorer (env-tuned), score all V rows, and report recall@1 + captured mass at
/// several K vs the exact full bf16 logits. The serving path is `lmhead_twostage_serve`.
fn lmhead_shortlist_measure(
    gpu: &mut Gpu,
    w: &ZayaGpuWeights,
    fnorm: &GpuTensor,
    logits_out: &GpuTensor,
    state: &mut ZayaDecodeState,
    pos: usize,
) -> Result<(), String> {
    let (vocab, hidden) = match w.embed.wt_mk() {
        Some((_, m, k)) => (m, k),
        None => return Ok(()),
    };
    // Offline quant-format experiment dump (HIPFIRE_ZAYA_DUMP=<dir>): the bf16 lm_head W
    // once + each decode step's post-norm hidden h, so a numpy oracle can compare per-row
    // vs per-group int8 (fine tier) and the high-nibble coarse without re-running the GPU.
    if let Ok(dir) = std::env::var("HIPFIRE_ZAYA_DUMP") {
        use std::io::Write;
        let wpath = format!("{dir}/W.bf16");
        if !std::path::Path::new(&wpath).exists() {
            let (buf, m, k) = w.embed.wt_mk().unwrap();
            let wb = gpu
                .download_raw(buf, m * k * 2)
                .map_err(|e| format!("{e:?}"))?;
            std::fs::write(&wpath, &wb).map_err(|e| e.to_string())?;
            eprintln!("[zaya-dump] wrote W [{m},{k}] bf16 -> {wpath}");
        }
        let h = gpu.download_f32(fnorm).map_err(|e| format!("{e:?}"))?;
        let mut hb = Vec::with_capacity(h.len() * 4);
        for x in &h {
            hb.extend_from_slice(&x.to_le_bytes());
        }
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(format!("{dir}/h.f32"))
            .and_then(|mut f| f.write_all(&hb))
            .map_err(|e| e.to_string())?;
    }
    if state.lmhead_coarse.is_none() {
        let r = std::env::var("HIPFIRE_ZAYA_SHORTLIST_R")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(0);
        let use_svd = std::env::var("HIPFIRE_ZAYA_SHORTLIST_SVD").is_ok();
        let bits = std::env::var("HIPFIRE_ZAYA_SHORTLIST_BITS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&b| b == 2 || b == 4)
            .unwrap_or(4);
        let correct_r = std::env::var("HIPFIRE_ZAYA_SHORTLIST_CORRECT")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(0);
        state.lmhead_coarse = Some(build_lmhead_coarse(gpu, w, bits, r, use_svd, correct_r)?);
    }
    let c = state.lmhead_coarse.as_ref().unwrap();
    let (cv, coarse_us) = coarse_scores_host(gpu, c, fnorm, vocab, hidden)?;
    let fv = gpu.download_f32(logits_out).map_err(|e| format!("{e:?}"))?;
    // Full-distribution reference: argmax + softmax normaliser.
    let mut true_am = 0usize;
    let mut fmax = f32::NEG_INFINITY;
    for (i, &f) in fv.iter().enumerate() {
        if f > fmax {
            fmax = f;
            true_am = i;
        }
    }
    let z: f64 = fv.iter().map(|&f| ((f - fmax) as f64).exp()).sum();

    // Coarse top-K by partial select; report recall@1 + captured mass per K.
    let mut idx: Vec<u32> = (0..vocab as u32).collect();
    let mut out = format!("[zaya-shortlist] pos={pos} coarse={coarse_us}us true_am={true_am}");
    for &kk in &[1usize, 8, 32, 128, 512, 2048] {
        let kk = kk.min(vocab);
        idx.select_nth_unstable_by(kk - 1, |&a, &b| {
            cv[b as usize].partial_cmp(&cv[a as usize]).unwrap()
        });
        let topk = &idx[..kk];
        let recall1 = topk.iter().any(|&i| i as usize == true_am);
        let captured: f64 = topk
            .iter()
            .map(|&i| ((fv[i as usize] - fmax) as f64).exp())
            .sum::<f64>()
            / z;
        out.push_str(&format!(
            " | K={kk}: recall1={} mass={:.5}",
            recall1 as u8, captured
        ));
    }
    eprintln!("{out}");
    Ok(())
}

/// Run the GPU prefill forward, capturing per-block hidden states.
#[allow(clippy::needless_range_loop)]
pub fn gpu_forward_prefill(
    gpu: &mut Gpu,
    w: &ZayaGpuWeights,
    cfg: &ZayaConfig,
    ids: &[u32],
) -> Result<GpuTrace, String> {
    let s = ids.len();
    let h = cfg.hidden_size;
    let a = &cfg.attn;
    let (nq, nkv, hd) = (a.num_heads, a.num_kv_heads, a.head_dim);
    let q_dim = nq * hd;
    let k_dim = nkv * hd;
    let v_half = k_dim / 2;
    let conv_ch = q_dim + k_dim;
    let rh = cfg.moe.router_hidden_size;
    let n_route = cfg.moe.num_router_experts();
    let n_exp = cfg.moe.num_experts;
    let moe_int = cfg.moe.moe_intermediate_size;
    let eps = cfg.rms_norm_eps;
    let pad = (a.conv_depthwise_kernel - 1) + (a.conv_grouped_kernel - 1);

    // ids → device i32, embed gather → input affine.
    let hidden = z2o(gpu, s, h)?;
    for t in 0..s {
        let row = hidden.sub_offset(t * h, h);
        w.embed.embed_lookup(gpu, &row, ids[t], h)?;
    }
    // global input residual affine, in place.
    gpu.zaya_affine_input_f32(&hidden, &hidden, &w.in_scale, &w.in_bias, h, s * h)
        .map_err(|e| format!("{e:?}"))?;
    let trace_embed = gpu.download_f32(&hidden).map_err(|e| format!("{e:?}"))?;

    let mut block_traces = Vec::with_capacity(cfg.num_blocks);

    // reusable scratch
    let normed = z2o(gpu, s, h)?;
    let q = zo(gpu, s * q_dim)?;
    let k = zo(gpu, s * k_dim)?;
    let vcur = zo(gpu, s * v_half)?;
    let vdel = zo(gpu, s * v_half)?;
    let qres = zo(gpu, s * nq * hd)?;
    let kres = zo(gpu, s * nkv * hd)?;
    let stream = zo(gpu, conv_ch * (s + pad))?;
    let dw = zo(gpu, conv_ch * (s + pad - a.conv_depthwise_kernel + 1))?;
    let gw = zo(gpu, conv_ch * s)?;
    let query = zo(gpu, s * nq * hd)?;
    let key = zo(gpu, s * nkv * hd)?;
    let value = zo(gpu, s * nkv * hd)?;
    let ctx = zo(gpu, s * q_dim)?;
    let attn_out = zo(gpu, s * h)?;
    let rhid = z2o(gpu, s, rh)?;
    let rnormed = z2o(gpu, s, rh)?;
    let a1 = zo(gpu, s * rh)?;
    let a2 = zo(gpu, s * rh)?;
    let rlogits = zo(gpu, s * n_route)?;
    let moe_out = zo(gpu, s * h)?;
    let gate_up = zo(gpu, 2 * moe_int)?;
    let act = zo(gpu, moe_int)?;
    let down_t = zo(gpu, h)?;
    // Hoisted out of the layer loop (matches gpu_forward_serve/_calib): the
    // post-attention residual scratch and the cross-layer router state are fully
    // overwritten each layer, so one allocation reused across blocks suffices —
    // no per-layer alloc churn, no leak.
    let g_res2 = z2o(gpu, s, h)?;
    let router_state = zo(gpu, s * rh)?;
    let attn_scale = 1.0 / (hd as f32).sqrt();
    let l2_scale = (hd as f32).sqrt();
    let dw_len = s + pad - a.conv_depthwise_kernel + 1;
    let fnorm = z2o(gpu, s, h)?;
    let logits = zo(gpu, s * cfg.vocab_size)?;

    for (li, lw) in w.layers.iter().enumerate() {
        gpu.rmsnorm_f32(&hidden, &lw.input_ln, &normed, eps)
            .map_err(|e| format!("{e:?}"))?;
        // CCA projections
        gemv_seq(gpu, &lw.q_proj, &normed, &q, s, q_dim, h)?;
        gemv_seq(gpu, &lw.k_proj, &normed, &k, s, k_dim, h)?;
        gemv_seq(gpu, &lw.v_cur, &normed, &vcur, s, v_half, h)?;
        gemv_seq(gpu, &lw.v_del, &normed, &vdel, s, v_half, h)?;
        // q/k residual paths
        gpu.zaya_qk_residual_f32(&qres, &kres, &q, &k, s, nq, nkv, hd, 0)
            .map_err(|e| format!("{e:?}"))?;
        gpu.zaya_qk_residual_f32(&qres, &kres, &q, &k, s, nq, nkv, hd, 1)
            .map_err(|e| format!("{e:?}"))?;
        // conv stream (zero pad region) → depthwise → grouped
        gpu.fill_f32(&stream, 0.0).map_err(|e| format!("{e:?}"))?;
        gpu.zaya_qk_stream_f32(&stream, &q, &k, s, q_dim, k_dim, pad)
            .map_err(|e| format!("{e:?}"))?;
        gpu.zaya_conv1d_valid_f32(
            &dw,
            &stream,
            &lw.conv_dw_w,
            &lw.conv_dw_b,
            conv_ch,
            conv_ch,
            a.conv_depthwise_kernel,
            s + pad,
            dw_len,
        )
        .map_err(|e| format!("{e:?}"))?;
        gpu.zaya_conv1d_valid_f32(
            &gw,
            &dw,
            &lw.conv_gr_w,
            &lw.conv_gr_b,
            conv_ch,
            nq + nkv,
            a.conv_grouped_kernel,
            dw_len,
            s,
        )
        .map_err(|e| format!("{e:?}"))?;
        // add residuals → head-major q,k
        gpu.zaya_add_conv_residual_f32(&query, &gw, &qres, s, nq, hd, q_dim, 0)
            .map_err(|e| format!("{e:?}"))?;
        gpu.zaya_add_conv_residual_f32(&key, &gw, &kres, s, nkv, hd, q_dim, 1)
            .map_err(|e| format!("{e:?}"))?;
        // value compose
        gpu.zaya_value_compose_f32(&value, &vcur, &vdel, s, nkv, hd)
            .map_err(|e| format!("{e:?}"))?;
        // qk-norm (+ temp on key), rope
        gpu.zaya_qk_l2norm_temp_f32(&query, None, s, nq, hd, l2_scale, f32::EPSILON)
            .map_err(|e| format!("{e:?}"))?;
        gpu.zaya_qk_l2norm_temp_f32(&key, Some(&lw.qk_temp), s, nkv, hd, l2_scale, f32::EPSILON)
            .map_err(|e| format!("{e:?}"))?;
        gpu.zaya_rope_partial_f32(&query, s, nq, hd, a.n_rot, a.rope_theta, 0)
            .map_err(|e| format!("{e:?}"))?;
        gpu.zaya_rope_partial_f32(&key, s, nkv, hd, a.n_rot, a.rope_theta, 0)
            .map_err(|e| format!("{e:?}"))?;
        // attention + o_proj
        gpu.zaya_gqa_attn_f32(&ctx, &query, &key, &value, s, nq, nkv, hd, attn_scale)
            .map_err(|e| format!("{e:?}"))?;
        gemv_seq(gpu, &lw.o_proj, &ctx, &attn_out, s, h, q_dim)?;
        // residual = post_attention_residual_scale(attn_out, residual). `hidden`
        // is still the block input here (only overwritten by the post-MLP affine
        // below), so it is the residual source — no host round-trip needed.
        gpu.zaya_affine_residual_f32(
            &g_res2,
            &attn_out,
            &hidden,
            &lw.pa_rs[0],
            &lw.pa_rs[1],
            &lw.pa_rs[2],
            &lw.pa_rs[3],
            h,
            s * h,
        )
        .map_err(|e| format!("{e:?}"))?;
        // normed = post_attention_layernorm(residual)
        gpu.rmsnorm_f32(&g_res2, &lw.post_attn_ln, &normed, eps)
            .map_err(|e| format!("{e:?}"))?;

        // ── MoE ──
        gemv_seq(gpu, &lw.down_proj_w, &normed, &rhid, s, rh, h)?;
        gpu.zaya_bias_add_f32(&rhid, &lw.down_proj_b, rh, s * rh)
            .map_err(|e| format!("{e:?}"))?;
        if li != 0 {
            if let Some(scale) = lw.router_states_scale.as_ref() {
                gpu.zaya_eda_add_f32(&rhid, &router_state, scale, rh, s * rh)
                    .map_err(|e| format!("{e:?}"))?;
            }
        }
        // save next router state via an on-device copy (no host round-trip).
        gpu.zaya_write_at_f32(&router_state, &rhid, 0, s * rh)
            .map_err(|e| format!("{e:?}"))?;
        // router_mlp
        gpu.rmsnorm_f32(&rhid, &lw.rnorm_w, &rnormed, eps)
            .map_err(|e| format!("{e:?}"))?;
        gemv_seq(gpu, &lw.fc1_w, &rnormed, &a1, s, rh, rh)?;
        gpu.zaya_bias_add_f32(&a1, &lw.fc1_b, rh, s * rh)
            .map_err(|e| format!("{e:?}"))?;
        gpu.zaya_gelu_exact_f32(&a1, s * rh)
            .map_err(|e| format!("{e:?}"))?;
        gemv_seq(gpu, &lw.fc2_w, &a1, &a2, s, rh, rh)?;
        gpu.zaya_bias_add_f32(&a2, &lw.fc2_b, rh, s * rh)
            .map_err(|e| format!("{e:?}"))?;
        gpu.zaya_gelu_exact_f32(&a2, s * rh)
            .map_err(|e| format!("{e:?}"))?;
        gemv_seq(gpu, &lw.out_proj_w, &a2, &rlogits, s, n_route, rh)?;
        // host top-1 of (softmax + balancing_biases); MoD skip if idx==n_exp
        let logit_host = gpu.download_f32(&rlogits).map_err(|e| format!("{e:?}"))?;
        gpu.fill_f32(&moe_out, 0.0).map_err(|e| format!("{e:?}"))?;
        for t in 0..s {
            let row = &logit_host[t * n_route..(t + 1) * n_route];
            let maxv = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut probs = vec![0f32; n_route];
            let mut denom = 0f32;
            for e in 0..n_route {
                probs[e] = (row[e] - maxv).exp();
                denom += probs[e];
            }
            for p in probs.iter_mut() {
                *p /= denom;
            }
            let mut best = 0usize;
            let mut bestv = f32::NEG_INFINITY;
            for e in 0..n_route {
                let v = probs[e] + lw.balancing_biases[e];
                if v > bestv {
                    bestv = v;
                    best = e;
                }
            }
            if best == n_exp {
                continue; // MoD skip
            }
            let weight = probs[best];
            let xt = normed.sub_offset(t * h, h);
            let ex = &lw.experts[best];
            ex.gate_up.gemv(gpu, &xt, &gate_up)?;
            let g = gate_up.sub_offset(0, moe_int);
            let u = gate_up.sub_offset(moe_int, moe_int);
            gpu.silu_mul_f32(&g, &u, &act)
                .map_err(|e| format!("{e:?}"))?;
            ex.down.gemv(gpu, &act, &down_t)?;
            let ot = moe_out.sub_offset(t * h, h);
            gpu.scaled_add_inplace_cpu_scalar_f32(&ot, &down_t, weight)
                .map_err(|e| format!("{e:?}"))?;
        }
        // hidden = post_mlp_residual_scale(moe_out, residual)
        gpu.zaya_affine_residual_f32(
            &hidden,
            &moe_out,
            &g_res2,
            &lw.pm_rs[0],
            &lw.pm_rs[1],
            &lw.pm_rs[2],
            &lw.pm_rs[3],
            h,
            s * h,
        )
        .map_err(|e| format!("{e:?}"))?;
        block_traces.push(gpu.download_f32(&hidden).map_err(|e| format!("{e:?}"))?);
        let _ = li;
        gpu.reclaim_pending();
    }

    // final norm + tied lm_head
    gpu.rmsnorm_f32(&hidden, &w.norm, &fnorm, eps)
        .map_err(|e| format!("{e:?}"))?;
    let final_norm = gpu.download_f32(&fnorm).map_err(|e| format!("{e:?}"))?;
    gemv_seq(gpu, &w.embed, &fnorm, &logits, s, cfg.vocab_size, h)?;
    let logits_host = gpu.download_f32(&logits).map_err(|e| format!("{e:?}"))?;

    gpu.reclaim_pending();
    Ok(GpuTrace {
        embed_scaled: trace_embed,
        block: block_traces,
        final_norm,
        logits: logits_host,
        seq: s,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // f16 1.0 = 0x3C00; little-endian on-disk.
    const F16_ONE: [u8; 2] = [0x00, 0x3C];

    #[test]
    fn oq_repack_dispatch_and_dtype() {
        let mut data = Vec::new();
        data.extend_from_slice(&F16_ONE);
        data.extend_from_slice(&[0u8; 128]);
        assert!(matches!(
            oq_repack(34, &data, 1, 256),
            Some((_, DType::Oq4G256))
        ));
        assert!(matches!(
            oq_repack(33, &data, 1, 256),
            Some((_, DType::Oq8G256))
        ));
        // qt 35 needs the 258-byte block; 33/34 use the 130-byte block.
        let mut data8 = Vec::new();
        data8.extend_from_slice(&F16_ONE);
        data8.extend_from_slice(&vec![0u8; 256]);
        assert!(matches!(
            oq_repack(35, &data8, 1, 256),
            Some((_, DType::Oq8G256))
        ));
        // qt 36 (OqPlusCompact, `oq4.125`+): block = [f16 scale][128 nibbles]
        // [N_out·(u8 idx, i8 val)]. This is the code that PREVIOUSLY fell through
        // to `dequant_qt` → "unsupported quant_type 36"; it now routes through the
        // shared `oq8_arch_load` and expands to Oq8G256, overlaying the outlier.
        let mut data36 = Vec::new();
        data36.extend_from_slice(&F16_ONE); // scale
        data36.extend_from_slice(&[0u8; 128]); // int4 bulk = all zero
        data36.push(5u8); // outlier at in-group index 5
        data36.push(42i8 as u8); // outlier value 42
        let (combined, dt) = oq_repack(36, &data36, 1, 256).expect("qt 36 is OQ8-family");
        assert_eq!(dt, DType::Oq8G256);
        // Combined = [int8 m*k=256][f32 scale·ng=4]; the overlay lands at index 5.
        assert_eq!(combined.len(), 256 + 4);
        assert_eq!(combined[5] as i8, 42);
        assert_eq!(combined[4], 0); // a non-overlaid bulk slot stays zero

        // Non-OQ quant_types fall through to linear_dtype / dequant.
        assert!(oq_repack(13, &data, 1, 256).is_none());
        assert!(oq_repack(3, &data, 1, 256).is_none());
    }
}
