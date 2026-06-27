// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! ZAYA1 GPU prefill forward (bring-up). Loads bf16/f16 weights from the HFQ as
//! f32 GPU tensors and runs the full forward op-for-op against `cpu.rs`, reusing
//! `gemv_f32`/`rmsnorm_f32`/`silu_mul_f32` plus the custom CCA/EDA kernels in
//! `rdna-compute/src/dispatch/zaya_cca.rs`. Batch 1; prefill over the whole
//! prompt at once (matches the golden). Decode + flash-attention + serving seam
//! follow once this validates.

use crate::ZayaConfig;
use hipfire_runtime::hfq::HfqFile;
use rdna_compute::{DType, Gpu, GpuTensor};

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
    let shape = if shape.is_empty() { vec![f.len()] } else { shape };
    gpu.upload_f32(&f, &shape)
        .map_err(|e| format!("zaya gpu upload {name}: {e:?}"))
}

struct GpuExpert {
    gate_up: GpuTensor, // [2*moe_int, hidden]
    down: GpuTensor,    // [hidden, moe_int]
}

struct GpuLayer {
    input_ln: GpuTensor,
    post_attn_ln: GpuTensor,
    q_proj: GpuTensor,
    k_proj: GpuTensor,
    v_cur: GpuTensor,
    v_del: GpuTensor,
    conv_dw_w: GpuTensor,
    conv_dw_b: GpuTensor,
    conv_gr_w: GpuTensor,
    conv_gr_b: GpuTensor,
    qk_temp: GpuTensor,
    o_proj: GpuTensor,
    // router
    down_proj_w: GpuTensor,
    down_proj_b: GpuTensor,
    router_states_scale: Option<GpuTensor>,
    rnorm_w: GpuTensor,
    fc1_w: GpuTensor,
    fc1_b: GpuTensor,
    fc2_w: GpuTensor,
    fc2_b: GpuTensor,
    out_proj_w: GpuTensor,
    balancing_biases: Vec<f32>,
    experts: Vec<GpuExpert>,
    // residual scales (each: hidden_states_scale, hidden_states_bias, residual_scale, residual_bias)
    pa_rs: [GpuTensor; 4],
    pm_rs: [GpuTensor; 4],
}

/// All ZAYA weights resident on the GPU (f32).
pub struct ZayaGpuWeights {
    embed: GpuTensor,
    in_scale: GpuTensor,
    in_bias: GpuTensor,
    layers: Vec<GpuLayer>,
    norm: GpuTensor,
}

impl ZayaGpuWeights {
    pub fn load(hfq: &HfqFile, gpu: &mut Gpu, cfg: &ZayaConfig) -> Result<Self, String> {
        let embed = up(hfq, gpu, "model.embed_tokens.weight")?;
        let in_scale = up(hfq, gpu, "model.input_hidden_states_scale")?;
        let in_bias = up(hfq, gpu, "model.input_hidden_states_bias")?;
        let norm = up(hfq, gpu, "model.norm.weight")?;
        let mut layers = Vec::with_capacity(cfg.num_blocks);
        for l in 0..cfg.num_blocks {
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
                    gate_up: up(hfq, gpu, &format!("{p}.mlp.experts.{e}.gate_up_proj.weight"))?,
                    down: up(hfq, gpu, &format!("{p}.mlp.experts.{e}.down_proj.weight"))?,
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
            let bb_qt = hfq.find_tensor_info(&format!("{gate}.balancing_biases")).unwrap().quant_type;
            let balancing_biases = dequant_qt(bb_qt, &bb_bytes)?;

            layers.push(GpuLayer {
                input_ln: up(hfq, gpu, &format!("{p}.input_layernorm.weight"))?,
                post_attn_ln: up(hfq, gpu, &format!("{p}.post_attention_layernorm.weight"))?,
                q_proj: up(hfq, gpu, &format!("{qkv}.q_proj.weight"))?,
                k_proj: up(hfq, gpu, &format!("{qkv}.k_proj.weight"))?,
                v_cur: up(hfq, gpu, &format!("{qkv}.v_proj_current.weight"))?,
                v_del: up(hfq, gpu, &format!("{qkv}.v_proj_delayed.weight"))?,
                conv_dw_w: up(hfq, gpu, &format!("{qkv}.conv_qk_depthwise.weight"))?,
                conv_dw_b: up(hfq, gpu, &format!("{qkv}.conv_qk_depthwise.bias"))?,
                conv_gr_w: up(hfq, gpu, &format!("{qkv}.conv_qk_grouped.weight"))?,
                conv_gr_b: up(hfq, gpu, &format!("{qkv}.conv_qk_grouped.bias"))?,
                qk_temp: up(hfq, gpu, &format!("{p}.self_attn.qk_norm.temp"))?,
                o_proj: up(hfq, gpu, &format!("{p}.self_attn.o_proj.weight"))?,
                down_proj_w: up(hfq, gpu, &format!("{gate}.down_proj.weight"))?,
                down_proj_b: up(hfq, gpu, &format!("{gate}.down_proj.bias"))?,
                router_states_scale,
                rnorm_w: up(hfq, gpu, &format!("{rmlp}.norm.weight"))?,
                fc1_w: up(hfq, gpu, &format!("{rmlp}.fc1.weight"))?,
                fc1_b: up(hfq, gpu, &format!("{rmlp}.fc1.bias"))?,
                fc2_w: up(hfq, gpu, &format!("{rmlp}.fc2.weight"))?,
                fc2_b: up(hfq, gpu, &format!("{rmlp}.fc2.bias"))?,
                out_proj_w: up(hfq, gpu, &format!("{rmlp}.out_proj.weight"))?,
                balancing_biases,
                experts,
                pa_rs: rs(gpu, &format!("{p}.post_attention_residual_scale"))?,
                pm_rs: rs(gpu, &format!("{p}.post_mlp_residual_scale"))?,
            });
        }
        Ok(Self { embed, in_scale, in_bias, layers, norm })
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

/// gemv over a sequence: `y[s,m] = x[s,k] @ w[m,k]^T` (per-token gemv loop).
fn gemv_seq(gpu: &mut Gpu, w: &GpuTensor, x: &GpuTensor, y: &GpuTensor, s: usize, m: usize, k: usize) -> Result<(), String> {
    for t in 0..s {
        let xt = x.sub_offset(t * k, k);
        let yt = y.sub_offset(t * m, m);
        gpu.gemv_f32(w, &xt, &yt).map_err(|e| format!("zaya gemv: {e:?}"))?;
    }
    Ok(())
}

fn z(gpu: &mut Gpu, n: usize) -> Result<GpuTensor, String> {
    gpu.zeros(&[n], DType::F32).map_err(|e| format!("zaya alloc: {e:?}"))
}

/// 2D `[rows, d]` allocation — required for tensors fed to `rmsnorm_f32`, which
/// reads `batch = shape[0]` and `n = shape.last()`.
fn z2(gpu: &mut Gpu, rows: usize, d: usize) -> Result<GpuTensor, String> {
    gpu.zeros(&[rows, d], DType::F32).map_err(|e| format!("zaya alloc: {e:?}"))
}

/// Serving forward: run the full forward over `ids` and write the **last
/// position's** logits into `logits_out` (`[vocab]`). No per-block downloads;
/// residual stays on-device. (Bring-up: re-prefills the whole sequence each call.)
#[allow(clippy::needless_range_loop)]
pub fn gpu_forward_serve(
    gpu: &mut Gpu,
    w: &ZayaGpuWeights,
    cfg: &ZayaConfig,
    ids: &[u32],
    logits_out: &GpuTensor,
) -> Result<(), String> {
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

    let id_bytes: Vec<u8> = ids.iter().flat_map(|&x| (x as i32).to_le_bytes()).collect();
    let g_ids = gpu.upload_raw(&id_bytes, &[s]).map_err(|e| format!("zaya ids: {e:?}"))?;
    let mut hidden = z2(gpu, s, h)?;
    gpu.zaya_embed_gather_f32(&hidden, &w.embed, &g_ids, h, s * h).map_err(|e| format!("{e:?}"))?;
    let embed_scaled = z2(gpu, s, h)?;
    gpu.zaya_affine_input_f32(&embed_scaled, &hidden, &w.in_scale, &w.in_bias, h, s * h).map_err(|e| format!("{e:?}"))?;
    hidden = embed_scaled;

    let normed = z2(gpu, s, h)?;
    let q = z(gpu, s * q_dim)?;
    let k = z(gpu, s * k_dim)?;
    let vcur = z(gpu, s * v_half)?;
    let vdel = z(gpu, s * v_half)?;
    let qres = z(gpu, s * nq * hd)?;
    let kres = z(gpu, s * nkv * hd)?;
    let stream = z(gpu, conv_ch * (s + pad))?;
    let dw = z(gpu, conv_ch * dw_len)?;
    let gw = z(gpu, conv_ch * s)?;
    let query = z(gpu, s * nq * hd)?;
    let key = z(gpu, s * nkv * hd)?;
    let value = z(gpu, s * nkv * hd)?;
    let ctx = z(gpu, s * q_dim)?;
    let attn_out = z(gpu, s * h)?;
    let g_res2 = z2(gpu, s, h)?;
    let rhid = z2(gpu, s, rh)?;
    let rnormed = z2(gpu, s, rh)?;
    let a1 = z2(gpu, s, rh)?;
    let a2 = z2(gpu, s, rh)?;
    let rlogits = z(gpu, s * n_route)?;
    let moe_out = z(gpu, s * h)?;
    let gate_up = z(gpu, 2 * moe_int)?;
    let act = z(gpu, moe_int)?;
    let down_t = z(gpu, h)?;
    let mut router_state = z(gpu, s * rh)?;

    for (li, lw) in w.layers.iter().enumerate() {
        gpu.rmsnorm_f32(&hidden, &lw.input_ln, &normed, eps).map_err(|e| format!("{e:?}"))?;
        gemv_seq(gpu, &lw.q_proj, &normed, &q, s, q_dim, h)?;
        gemv_seq(gpu, &lw.k_proj, &normed, &k, s, k_dim, h)?;
        gemv_seq(gpu, &lw.v_cur, &normed, &vcur, s, v_half, h)?;
        gemv_seq(gpu, &lw.v_del, &normed, &vdel, s, v_half, h)?;
        gpu.zaya_qk_residual_f32(&qres, &kres, &q, &k, s, nq, nkv, hd, 0).map_err(|e| format!("{e:?}"))?;
        gpu.zaya_qk_residual_f32(&qres, &kres, &q, &k, s, nq, nkv, hd, 1).map_err(|e| format!("{e:?}"))?;
        gpu.fill_f32(&stream, 0.0).map_err(|e| format!("{e:?}"))?;
        gpu.zaya_qk_stream_f32(&stream, &q, &k, s, q_dim, k_dim, pad).map_err(|e| format!("{e:?}"))?;
        gpu.zaya_conv1d_valid_f32(&dw, &stream, &lw.conv_dw_w, &lw.conv_dw_b, conv_ch, conv_ch, a.conv_depthwise_kernel, s + pad, dw_len).map_err(|e| format!("{e:?}"))?;
        gpu.zaya_conv1d_valid_f32(&gw, &dw, &lw.conv_gr_w, &lw.conv_gr_b, conv_ch, nq + nkv, a.conv_grouped_kernel, dw_len, s).map_err(|e| format!("{e:?}"))?;
        gpu.zaya_add_conv_residual_f32(&query, &gw, &qres, s, nq, hd, q_dim, 0).map_err(|e| format!("{e:?}"))?;
        gpu.zaya_add_conv_residual_f32(&key, &gw, &kres, s, nkv, hd, q_dim, 1).map_err(|e| format!("{e:?}"))?;
        gpu.zaya_value_compose_f32(&value, &vcur, &vdel, s, nkv, hd).map_err(|e| format!("{e:?}"))?;
        gpu.zaya_qk_l2norm_temp_f32(&query, None, s, nq, hd, l2_scale, f32::EPSILON).map_err(|e| format!("{e:?}"))?;
        gpu.zaya_qk_l2norm_temp_f32(&key, Some(&lw.qk_temp), s, nkv, hd, l2_scale, f32::EPSILON).map_err(|e| format!("{e:?}"))?;
        gpu.zaya_rope_partial_f32(&query, s, nq, hd, a.n_rot, a.rope_theta).map_err(|e| format!("{e:?}"))?;
        gpu.zaya_rope_partial_f32(&key, s, nkv, hd, a.n_rot, a.rope_theta).map_err(|e| format!("{e:?}"))?;
        gpu.zaya_gqa_attn_f32(&ctx, &query, &key, &value, s, nq, nkv, hd, attn_scale).map_err(|e| format!("{e:?}"))?;
        gemv_seq(gpu, &lw.o_proj, &ctx, &attn_out, s, h, q_dim)?;
        // residual stays on-device: `hidden` is still the block input here.
        gpu.zaya_affine_residual_f32(&g_res2, &attn_out, &hidden, &lw.pa_rs[0], &lw.pa_rs[1], &lw.pa_rs[2], &lw.pa_rs[3], h, s * h).map_err(|e| format!("{e:?}"))?;
        gpu.rmsnorm_f32(&g_res2, &lw.post_attn_ln, &normed, eps).map_err(|e| format!("{e:?}"))?;
        // MoE
        gemv_seq(gpu, &lw.down_proj_w, &normed, &rhid, s, rh, h)?;
        gpu.zaya_bias_add_f32(&rhid, &lw.down_proj_b, rh, s * rh).map_err(|e| format!("{e:?}"))?;
        if li != 0 {
            if let Some(scale) = lw.router_states_scale.as_ref() {
                gpu.zaya_eda_add_f32(&rhid, &router_state, scale, rh, s * rh).map_err(|e| format!("{e:?}"))?;
            }
        }
        // save next router state (device copy via re-upload of the host snapshot)
        let rhid_host = gpu.download_f32(&rhid).map_err(|e| format!("{e:?}"))?;
        router_state = gpu.upload_f32(&rhid_host, &[s, rh]).map_err(|e| format!("{e:?}"))?;
        gpu.rmsnorm_f32(&rhid, &lw.rnorm_w, &rnormed, eps).map_err(|e| format!("{e:?}"))?;
        gemv_seq(gpu, &lw.fc1_w, &rnormed, &a1, s, rh, rh)?;
        gpu.zaya_bias_add_f32(&a1, &lw.fc1_b, rh, s * rh).map_err(|e| format!("{e:?}"))?;
        gpu.zaya_gelu_exact_f32(&a1, s * rh).map_err(|e| format!("{e:?}"))?;
        gemv_seq(gpu, &lw.fc2_w, &a1, &a2, s, rh, rh)?;
        gpu.zaya_bias_add_f32(&a2, &lw.fc2_b, rh, s * rh).map_err(|e| format!("{e:?}"))?;
        gpu.zaya_gelu_exact_f32(&a2, s * rh).map_err(|e| format!("{e:?}"))?;
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
            gpu.gemv_f32(&ex.gate_up, &xt, &gate_up).map_err(|e| format!("{e:?}"))?;
            let g = gate_up.sub_offset(0, moe_int);
            let u = gate_up.sub_offset(moe_int, moe_int);
            gpu.silu_mul_f32(&g, &u, &act).map_err(|e| format!("{e:?}"))?;
            gpu.gemv_f32(&ex.down, &act, &down_t).map_err(|e| format!("{e:?}"))?;
            let ot = moe_out.sub_offset(t * h, h);
            gpu.scaled_add_inplace_cpu_scalar_f32(&ot, &down_t, weight).map_err(|e| format!("{e:?}"))?;
        }
        gpu.zaya_affine_residual_f32(&hidden, &moe_out, &g_res2, &lw.pm_rs[0], &lw.pm_rs[1], &lw.pm_rs[2], &lw.pm_rs[3], h, s * h).map_err(|e| format!("{e:?}"))?;
    }

    // final norm on the last row → tied lm_head → logits_out [vocab]
    let fnorm = z2(gpu, s, h)?;
    gpu.rmsnorm_f32(&hidden, &w.norm, &fnorm, eps).map_err(|e| format!("{e:?}"))?;
    let last = fnorm.sub_offset((s - 1) * h, h);
    gpu.gemv_f32(&w.embed, &last, logits_out).map_err(|e| format!("zaya lm_head: {e:?}"))?;
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
    let id_bytes: Vec<u8> = ids.iter().flat_map(|&x| (x as i32).to_le_bytes()).collect();
    let g_ids = gpu.upload_raw(&id_bytes, &[s]).map_err(|e| format!("zaya ids: {e:?}"))?;
    let mut hidden = z2(gpu, s, h)?;
    gpu.zaya_embed_gather_f32(&hidden, &w.embed, &g_ids, h, s * h).map_err(|e| format!("{e:?}"))?;
    let embed_scaled = z2(gpu, s, h)?;
    gpu.zaya_affine_input_f32(&embed_scaled, &hidden, &w.in_scale, &w.in_bias, h, s * h).map_err(|e| format!("{e:?}"))?;
    hidden = embed_scaled;
    let trace_embed = gpu.download_f32(&hidden).map_err(|e| format!("{e:?}"))?;

    let mut block_traces = Vec::with_capacity(cfg.num_blocks);
    let mut router_state: Option<GpuTensor> = None;

    // reusable scratch
    let normed = z2(gpu, s, h)?;
    let q = z(gpu, s * q_dim)?;
    let k = z(gpu, s * k_dim)?;
    let vcur = z(gpu, s * v_half)?;
    let vdel = z(gpu, s * v_half)?;
    let qres = z(gpu, s * nq * hd)?;
    let kres = z(gpu, s * nkv * hd)?;
    let stream = z(gpu, conv_ch * (s + pad))?;
    let dw = z(gpu, conv_ch * (s + pad - a.conv_depthwise_kernel + 1))?;
    let gw = z(gpu, conv_ch * s)?;
    let query = z(gpu, s * nq * hd)?;
    let key = z(gpu, s * nkv * hd)?;
    let value = z(gpu, s * nkv * hd)?;
    let ctx = z(gpu, s * q_dim)?;
    let attn_out = z(gpu, s * h)?;
    let rhid = z2(gpu, s, rh)?;
    let rnormed = z2(gpu, s, rh)?;
    let a1 = z(gpu, s * rh)?;
    let a2 = z(gpu, s * rh)?;
    let rlogits = z(gpu, s * n_route)?;
    let moe_out = z(gpu, s * h)?;
    let gate_up = z(gpu, 2 * moe_int)?;
    let act = z(gpu, moe_int)?;
    let down_t = z(gpu, h)?;
    let attn_scale = 1.0 / (hd as f32).sqrt();
    let l2_scale = (hd as f32).sqrt();
    let dw_len = s + pad - a.conv_depthwise_kernel + 1;

    for (li, lw) in w.layers.iter().enumerate() {
        let residual = gpu.download_f32(&hidden).map_err(|e| format!("{e:?}"))?; // keep host copy for affine residual source
        let g_residual = gpu.upload_f32(&residual, &[s * h]).map_err(|e| format!("{e:?}"))?;

        gpu.rmsnorm_f32(&hidden, &lw.input_ln, &normed, eps).map_err(|e| format!("{e:?}"))?;
        // CCA projections
        gemv_seq(gpu, &lw.q_proj, &normed, &q, s, q_dim, h)?;
        gemv_seq(gpu, &lw.k_proj, &normed, &k, s, k_dim, h)?;
        gemv_seq(gpu, &lw.v_cur, &normed, &vcur, s, v_half, h)?;
        gemv_seq(gpu, &lw.v_del, &normed, &vdel, s, v_half, h)?;
        // q/k residual paths
        gpu.zaya_qk_residual_f32(&qres, &kres, &q, &k, s, nq, nkv, hd, 0).map_err(|e| format!("{e:?}"))?;
        gpu.zaya_qk_residual_f32(&qres, &kres, &q, &k, s, nq, nkv, hd, 1).map_err(|e| format!("{e:?}"))?;
        // conv stream (zero pad region) → depthwise → grouped
        gpu.fill_f32(&stream, 0.0).map_err(|e| format!("{e:?}"))?;
        gpu.zaya_qk_stream_f32(&stream, &q, &k, s, q_dim, k_dim, pad).map_err(|e| format!("{e:?}"))?;
        gpu.zaya_conv1d_valid_f32(&dw, &stream, &lw.conv_dw_w, &lw.conv_dw_b, conv_ch, conv_ch, a.conv_depthwise_kernel, s + pad, dw_len).map_err(|e| format!("{e:?}"))?;
        gpu.zaya_conv1d_valid_f32(&gw, &dw, &lw.conv_gr_w, &lw.conv_gr_b, conv_ch, nq + nkv, a.conv_grouped_kernel, dw_len, s).map_err(|e| format!("{e:?}"))?;
        // add residuals → head-major q,k
        gpu.zaya_add_conv_residual_f32(&query, &gw, &qres, s, nq, hd, q_dim, 0).map_err(|e| format!("{e:?}"))?;
        gpu.zaya_add_conv_residual_f32(&key, &gw, &kres, s, nkv, hd, q_dim, 1).map_err(|e| format!("{e:?}"))?;
        // value compose
        gpu.zaya_value_compose_f32(&value, &vcur, &vdel, s, nkv, hd).map_err(|e| format!("{e:?}"))?;
        // qk-norm (+ temp on key), rope
        gpu.zaya_qk_l2norm_temp_f32(&query, None, s, nq, hd, l2_scale, f32::EPSILON).map_err(|e| format!("{e:?}"))?;
        gpu.zaya_qk_l2norm_temp_f32(&key, Some(&lw.qk_temp), s, nkv, hd, l2_scale, f32::EPSILON).map_err(|e| format!("{e:?}"))?;
        gpu.zaya_rope_partial_f32(&query, s, nq, hd, a.n_rot, a.rope_theta).map_err(|e| format!("{e:?}"))?;
        gpu.zaya_rope_partial_f32(&key, s, nkv, hd, a.n_rot, a.rope_theta).map_err(|e| format!("{e:?}"))?;
        // attention + o_proj
        gpu.zaya_gqa_attn_f32(&ctx, &query, &key, &value, s, nq, nkv, hd, attn_scale).map_err(|e| format!("{e:?}"))?;
        gemv_seq(gpu, &lw.o_proj, &ctx, &attn_out, s, h, q_dim)?;
        // residual = post_attention_residual_scale(attn_out, residual)
        let g_res2 = z2(gpu, s, h)?;
        gpu.zaya_affine_residual_f32(&g_res2, &attn_out, &g_residual, &lw.pa_rs[0], &lw.pa_rs[1], &lw.pa_rs[2], &lw.pa_rs[3], h, s * h).map_err(|e| format!("{e:?}"))?;
        // normed = post_attention_layernorm(residual)
        gpu.rmsnorm_f32(&g_res2, &lw.post_attn_ln, &normed, eps).map_err(|e| format!("{e:?}"))?;

        // ── MoE ──
        gemv_seq(gpu, &lw.down_proj_w, &normed, &rhid, s, rh, h)?;
        gpu.zaya_bias_add_f32(&rhid, &lw.down_proj_b, rh, s * rh).map_err(|e| format!("{e:?}"))?;
        if li != 0 {
            if let (Some(scale), Some(prev)) = (lw.router_states_scale.as_ref(), router_state.as_ref()) {
                gpu.zaya_eda_add_f32(&rhid, prev, scale, rh, s * rh).map_err(|e| format!("{e:?}"))?;
            }
        }
        // save next router state (copy of rhid)
        let rhid_host = gpu.download_f32(&rhid).map_err(|e| format!("{e:?}"))?;
        router_state = Some(gpu.upload_f32(&rhid_host, &[s * rh]).map_err(|e| format!("{e:?}"))?);
        // router_mlp
        gpu.rmsnorm_f32(&rhid, &lw.rnorm_w, &rnormed, eps).map_err(|e| format!("{e:?}"))?;
        gemv_seq(gpu, &lw.fc1_w, &rnormed, &a1, s, rh, rh)?;
        gpu.zaya_bias_add_f32(&a1, &lw.fc1_b, rh, s * rh).map_err(|e| format!("{e:?}"))?;
        gpu.zaya_gelu_exact_f32(&a1, s * rh).map_err(|e| format!("{e:?}"))?;
        gemv_seq(gpu, &lw.fc2_w, &a1, &a2, s, rh, rh)?;
        gpu.zaya_bias_add_f32(&a2, &lw.fc2_b, rh, s * rh).map_err(|e| format!("{e:?}"))?;
        gpu.zaya_gelu_exact_f32(&a2, s * rh).map_err(|e| format!("{e:?}"))?;
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
            gpu.gemv_f32(&ex.gate_up, &xt, &gate_up).map_err(|e| format!("{e:?}"))?;
            let g = gate_up.sub_offset(0, moe_int);
            let u = gate_up.sub_offset(moe_int, moe_int);
            gpu.silu_mul_f32(&g, &u, &act).map_err(|e| format!("{e:?}"))?;
            gpu.gemv_f32(&ex.down, &act, &down_t).map_err(|e| format!("{e:?}"))?;
            let ot = moe_out.sub_offset(t * h, h);
            gpu.scaled_add_inplace_cpu_scalar_f32(&ot, &down_t, weight).map_err(|e| format!("{e:?}"))?;
        }
        // hidden = post_mlp_residual_scale(moe_out, residual)
        gpu.zaya_affine_residual_f32(&hidden, &moe_out, &g_res2, &lw.pm_rs[0], &lw.pm_rs[1], &lw.pm_rs[2], &lw.pm_rs[3], h, s * h).map_err(|e| format!("{e:?}"))?;
        block_traces.push(gpu.download_f32(&hidden).map_err(|e| format!("{e:?}"))?);
        let _ = li;
    }

    // final norm + tied lm_head
    let fnorm = z2(gpu, s, h)?;
    gpu.rmsnorm_f32(&hidden, &w.norm, &fnorm, eps).map_err(|e| format!("{e:?}"))?;
    let final_norm = gpu.download_f32(&fnorm).map_err(|e| format!("{e:?}"))?;
    let logits = z(gpu, s * cfg.vocab_size)?;
    gemv_seq(gpu, &w.embed, &fnorm, &logits, s, cfg.vocab_size, h)?;
    let logits_host = gpu.download_f32(&logits).map_err(|e| format!("{e:?}"))?;

    Ok(GpuTrace {
        embed_scaled: trace_embed,
        block: block_traces,
        final_norm,
        logits: logits_host,
        seq: s,
    })
}
