#![allow(clippy::too_many_arguments)]
// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! Model-level assembly for a qwen3.5/3.6 hybrid stack, and a layer-streamed
//! gamma pass over it.
//!
//! The 35B is 30 `linear_attn` layers to 10 full-attention ones, and the MLP is
//! routed on some layers and dense on others. Every one of those is probed PER
//! LAYER from the artifact rather than inferred from config: hybrid models with
//! irregular patterns exist (BLS-Mini-Code is dense at layer 0 and routed
//! above), so a per-model flag is wrong in principle, not just in practice.
//!
//! Streaming matters here for the same reason it does in
//! [`crate::model::model_gamma_streamed`]: a 35B fp32 forward would need the
//! whole model resident. Only each layer's INPUT is kept, on the host, and the
//! reverse walk pages the layer back in and recomputes its internals.

use hipfire_rdna::{DType, Gpu, GpuTensor};
use std::collections::HashMap;

use crate::block::BlockDims;
use crate::la_block::{
    free_la_block_acts, la_block_backward, la_block_forward, LaBlockAdjoints,
    LinearAttnBlockWeights,
};
use crate::loader::{
    free_linear_attn_layer_fp32, free_llama_layer_fp32, free_moe_layer_fp32, layer_is_linear_attn,
    layer_is_moe, load_dense_mlp_fp32, load_linear_attn_layer_fp32, load_llama_layer_fp32_pfx_off,
    load_moe_layer_fp32, LinearAttnLayerF32, WeightSource,
};
use crate::model::GammaAccum;
use crate::ops::cross_entropy::cross_entropy;
use crate::ops::deltanet::LinearAttnDims;
use crate::ops::linear::{linear_backward_x, linear_forward};
use crate::ops::moe::{free_moe_acts, moe_backward, moe_forward, MoeDims};
use crate::ops::rmsnorm::{rmsnorm_backward, rmsnorm_forward};
use crate::ops::swiglu::{swiglu_backward, swiglu_forward};

/// What each layer of the stack turned out to be. Recorded on the forward pass
/// so the reverse walk reloads the same thing rather than re-probing.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LayerKind {
    LinearAttnMoe,
    LinearAttnDense,
    AttnMoe,
    AttnDense,
}

impl LayerKind {
    fn probe<S: WeightSource + ?Sized>(
        src: &S,
        prefix: &str,
        layer: usize,
        n_experts: usize,
    ) -> Self {
        let la = layer_is_linear_attn(src, prefix, layer);
        let moe = n_experts > 0 && layer_is_moe(src, prefix, layer);
        match (la, moe) {
            (true, true) => LayerKind::LinearAttnMoe,
            (true, false) => LayerKind::LinearAttnDense,
            (false, true) => LayerKind::AttnMoe,
            (false, false) => LayerKind::AttnDense,
        }
    }
    fn is_linear_attn(self) -> bool {
        matches!(self, LayerKind::LinearAttnMoe | LayerKind::LinearAttnDense)
    }
    fn is_moe(self) -> bool {
        matches!(self, LayerKind::LinearAttnMoe | LayerKind::AttnMoe)
    }
}

fn la_dims(l: &LinearAttnLayerF32, seq: usize, h: usize, eps: f32) -> LinearAttnDims {
    LinearAttnDims {
        seq,
        h,
        n_heads: l.n_heads,
        hd_k: l.hd_k,
        hd_v: l.hd_v,
        conv_k: l.conv_k,
        eps,
    }
}

fn la_weights<'a>(l: &'a LinearAttnLayerF32) -> LinearAttnBlockWeights<'a> {
    LinearAttnBlockWeights {
        norm1: &l.input_layernorm,
        in_proj_qkv: &l.in_proj_qkv,
        in_proj_a: &l.in_proj_a,
        in_proj_b: &l.in_proj_b,
        in_proj_z: &l.in_proj_z,
        out_proj: &l.out_proj,
        norm2: &l.post_attention_layernorm,
        conv1d: &l.conv1d,
        a_log: &l.a_log,
        dt_bias: &l.dt_bias,
        norm: &l.norm,
    }
}

/// Fold one `linear_attn` layer's adjoints into the gamma accumulator.
///
/// Keys are the artifact's own tensor names so the table joins to weights the
/// same way the dense and routed ones do. `conv1d`, `A_log` and `dt_bias` get
/// no entry: they are not linear projections, so there is no output-gradient
/// energy to integrate over and a quantizer has nothing to allocate bits to.
pub fn accumulate_gamma_linear_attn(
    acc: &mut GammaAccum,
    layer: usize,
    a: &LaBlockAdjoints,
    seq: usize,
    h: usize,
    n_heads: usize,
    hd_k: usize,
    hd_v: usize,
) {
    let p = format!("model.layers.{layer}.linear_attn");
    for (name, d, width) in [
        (
            format!("{p}.in_proj_qkv"),
            &a.d_qkv,
            n_heads * (2 * hd_k + hd_v),
        ),
        (format!("{p}.in_proj_a"), &a.d_a_raw, n_heads),
        (format!("{p}.in_proj_b"), &a.d_b_raw, n_heads),
        (format!("{p}.in_proj_z"), &a.d_z, n_heads * hd_v),
        (format!("{p}.out_proj"), &a.d_out_proj, h),
    ] {
        if width == 0 || d.len() < seq * width {
            continue;
        }
        let mut tot = 0.0f64;
        for r in 0..seq {
            let row = &d[r * width..(r + 1) * width];
            let ss: f64 = row.iter().map(|&x| (x as f64) * (x as f64)).sum();
            tot += ss / width as f64;
        }
        *acc.sum.entry(name).or_insert(0.0) += tot / seq.max(1) as f64;
    }
}

/// One layer-streamed forward+backward over a hybrid stack, accumulating gamma.
///
/// Returns the summed cross-entropy, the per-layer kinds (so a caller can
/// report what the stack actually was rather than assuming), and each layer's
/// INPUT hidden state — which is what a comparison against the runtime's own
/// `dump_qwen35_hidden_states` needs to locate a divergence.
pub fn gamma_hybrid_streamed<S: WeightSource + ?Sized>(
    gpu: &mut Gpu,
    src: &S,
    prefix: &str,
    cfg: &crate::config::LlamaConfig,
    dims: &BlockDims,
    embed: &GpuTensor,
    lm_head: Option<&GpuTensor>,
    final_norm: &GpuTensor,
    token_ids: &[u32],
    pos_host: &[f32],
    targets: &[f32],
    ignore_index: i32,
    n_experts: usize,
    top_k: usize,
    acc: &mut GammaAccum,
    // GemmaRMSNorm: block and final norms are `1 + w`. See
    // `loader::uses_unit_offset_norm` — wrong here is silent, not loud.
    unit_offset: bool,
) -> Result<(f32, Vec<LayerKind>, Vec<Vec<f32>>), String> {
    let (seq, h) = (dims.seq, dims.h);
    let vocab = cfg.vocab_size;
    let n_layers = cfg.num_hidden_layers;
    let e = |r: hipfire_rdna::HipError| format!("{r}");

    let kinds: Vec<LayerKind> = (0..n_layers)
        .map(|i| LayerKind::probe(src, prefix, i, n_experts))
        .collect();

    // Zero LoRA so the attention blocks run exactly at the base weights.
    let r = dims.lora_rank.max(1);
    let zl = |gpu: &mut Gpu, n: usize| gpu.zeros(&[n], DType::F32).map_err(e);
    let lora = crate::model::LayerLora {
        aq: zl(gpu, r * h)?,
        bq: zl(gpu, dims.q_dim() * r)?,
        av: zl(gpu, r * h)?,
        bv: zl(gpu, dims.kv_dim() * r)?,
    };

    let mut x = gpu.zeros(&[seq * h], DType::F32).map_err(e)?;
    for (t, &tok) in token_ids.iter().enumerate() {
        gpu.strided_copy_2d(embed, tok as usize * h, h, &x, t * h, h, 1, h, false)
            .map_err(e)?;
    }

    // Forward, keeping only each layer's input on the host.
    let mut layer_inputs: Vec<Vec<f32>> = Vec::with_capacity(n_layers);
    for (i, &kind) in kinds.iter().enumerate() {
        layer_inputs.push(gpu.download_f32(&x).map_err(e)?);
        let x_out = run_layer_forward(
            gpu,
            src,
            prefix,
            cfg,
            dims,
            &lora,
            &x,
            i,
            kind,
            n_experts,
            top_k,
            pos_host,
            unit_offset,
        )?;
        gpu.free_tensor(x).map_err(e)?;
        x = x_out;
    }

    // Head.
    let x_last = x;
    let xn = gpu.zeros(&[seq * h], DType::F32).map_err(e)?;
    let rinv = gpu.zeros(&[seq], DType::F32).map_err(e)?;
    rmsnorm_forward(gpu, &x_last, final_norm, &xn, &rinv, seq, h, dims.eps).map_err(e)?;
    let logits = gpu.zeros(&[seq * vocab], DType::F32).map_err(e)?;
    let out_proj = lm_head.unwrap_or(embed);
    linear_forward(gpu, &xn, out_proj, &logits, seq, h, vocab).map_err(e)?;

    let tgt = gpu.upload_f32(targets, &[seq]).map_err(e)?;
    let loss = gpu.zeros(&[seq], DType::F32).map_err(e)?;
    let d_logits = gpu.zeros(&[seq * vocab], DType::F32).map_err(e)?;
    cross_entropy(
        gpu,
        &logits,
        &tgt,
        &loss,
        &d_logits,
        seq,
        vocab,
        ignore_index,
    )
    .map_err(e)?;
    let loss_sum: f32 = gpu.download_f32(&loss).map_err(e)?.iter().sum();

    let d_xf = gpu.zeros(&[seq * h], DType::F32).map_err(e)?;
    linear_backward_x(gpu, &d_logits, out_proj, &d_xf, seq, h, vocab, false).map_err(e)?;
    let mut d_x = gpu.zeros(&[seq * h], DType::F32).map_err(e)?;
    let dw_dummy = gpu.zeros(&[h], DType::F32).map_err(e)?;
    rmsnorm_backward(
        gpu, &d_xf, &x_last, final_norm, &rinv, &d_x, &dw_dummy, seq, h,
    )
    .map_err(e)?;
    for t in [
        x_last, xn, rinv, logits, tgt, loss, d_logits, d_xf, dw_dummy,
    ] {
        gpu.free_tensor(t).map_err(e)?;
    }

    // Reverse walk.
    for i in (0..n_layers).rev() {
        let x_in = gpu.upload_f32(&layer_inputs[i], &[seq * h]).map_err(e)?;
        let d_in = run_layer_backward(
            gpu,
            src,
            prefix,
            cfg,
            dims,
            &lora,
            &x_in,
            &d_x,
            i,
            kinds[i],
            n_experts,
            top_k,
            pos_host,
            acc,
            unit_offset,
        )?;
        gpu.free_tensor(x_in).map_err(e)?;
        gpu.free_tensor(d_x).map_err(e)?;
        d_x = d_in;
    }
    gpu.free_tensor(d_x).map_err(e)?;
    for t in [lora.aq, lora.bq, lora.av, lora.bv] {
        gpu.free_tensor(t).map_err(e)?;
    }

    acc.n += 1;
    Ok((loss_sum, kinds, layer_inputs))
}

/// One layer forward. Returns the layer output; drops everything else.
fn run_layer_forward<S: WeightSource + ?Sized>(
    gpu: &mut Gpu,
    src: &S,
    prefix: &str,
    cfg: &crate::config::LlamaConfig,
    dims: &BlockDims,
    lora: &crate::model::LayerLora,
    x: &GpuTensor,
    i: usize,
    kind: LayerKind,
    n_experts: usize,
    top_k: usize,
    pos_host: &[f32],
    unit_offset: bool,
) -> Result<GpuTensor, String> {
    let (seq, h) = (dims.seq, dims.h);
    let e = |r: hipfire_rdna::HipError| format!("{r}");

    if kind.is_linear_attn() {
        let l = load_linear_attn_layer_fp32(gpu, src, prefix, i, h, unit_offset)?;
        let d = la_dims(&l, seq, h, dims.eps);
        let acts = la_block_forward(gpu, x, &la_weights(&l), &d).map_err(e)?;
        let x_out = gpu.zeros(&[seq * h], DType::F32).map_err(e)?;
        if kind.is_moe() {
            let (ml, inter) = load_moe_layer_fp32(gpu, src, prefix, i, h, n_experts)?;
            let mw = crate::model::moe_weights_of(&ml);
            let md = MoeDims {
                seq,
                h,
                inter,
                n_experts,
                top_k,
            };
            let (moe_out, macts) = moe_forward(gpu, &acts.xn2, &mw, &md).map_err(e)?;
            gpu.add_f32(&acts.x_mid, &moe_out, &x_out).map_err(e)?;
            free_moe_acts(gpu, macts).map_err(e)?;
            gpu.free_tensor(moe_out).map_err(e)?;
            drop(mw);
            free_moe_layer_fp32(gpu, ml)?;
        } else {
            // linear_attn + dense SwiGLU MLP — the Qwen3.5-0.8B shape.
            let (wg, wu, wd, inter) = load_dense_mlp_fp32(gpu, src, prefix, i, h)?;
            let mlp = dense_mlp_forward(gpu, &acts.xn2, &wg, &wu, &wd, seq, h, inter)?;
            gpu.add_f32(&acts.x_mid, &mlp.out, &x_out).map_err(e)?;
            free_dense_mlp_acts(gpu, mlp)?;
            for t in [wg, wu, wd] {
                gpu.free_tensor(t).map_err(e)?;
            }
        }
        free_la_block_acts(gpu, acts).map_err(e)?;
        free_linear_attn_layer_fp32(gpu, l)?;
        return Ok(x_out);
    }

    // Full-attention layer, dense or routed — the existing block handles both.
    let lw = load_llama_layer_fp32_pfx_off(gpu, src, prefix, cfg, i, !kind.is_moe(), unit_offset)?;
    let bw = crate::model::block_of(&lw);
    let out = if kind.is_moe() {
        let (ml, inter) = load_moe_layer_fp32(gpu, src, prefix, i, h, n_experts)?;
        let mw = crate::model::moe_weights_of(&ml);
        let md = MoeDims {
            seq,
            h,
            inter,
            n_experts,
            top_k,
        };
        let (x_out, acts, macts) = crate::block::moe_block_forward(
            gpu,
            x,
            &bw,
            &mw,
            &lora.as_block(),
            dims,
            &md,
            pos_host,
            i,
        )
        .map_err(e)?;
        free_moe_acts(gpu, macts).map_err(e)?;
        crate::block::free_block_acts(gpu, acts).map_err(e)?;
        drop(mw);
        free_moe_layer_fp32(gpu, ml)?;
        x_out
    } else {
        let (x_out, acts) =
            crate::block::block_forward(gpu, x, &bw, &lora.as_block(), dims, pos_host, i)
                .map_err(e)?;
        crate::block::free_block_acts(gpu, acts).map_err(e)?;
        x_out
    };
    drop(bw);
    free_llama_layer_fp32(gpu, lw)?;
    Ok(out)
}

/// One layer backward: reload, recompute from the saved input, capture gamma.
fn run_layer_backward<S: WeightSource + ?Sized>(
    gpu: &mut Gpu,
    src: &S,
    prefix: &str,
    cfg: &crate::config::LlamaConfig,
    dims: &BlockDims,
    lora: &crate::model::LayerLora,
    x_in: &GpuTensor,
    d_x: &GpuTensor,
    i: usize,
    kind: LayerKind,
    n_experts: usize,
    top_k: usize,
    pos_host: &[f32],
    acc: &mut GammaAccum,
    unit_offset: bool,
) -> Result<GpuTensor, String> {
    let (seq, h) = (dims.seq, dims.h);
    let e = |r: hipfire_rdna::HipError| format!("{r}");

    if kind.is_linear_attn() {
        let l = load_linear_attn_layer_fp32(gpu, src, prefix, i, h, unit_offset)?;
        let d = la_dims(&l, seq, h, dims.eps);
        let w = la_weights(&l);
        let acts = la_block_forward(gpu, x_in, &w, &d).map_err(e)?;

        let d_xn2 = if kind.is_moe() {
            let (ml, inter) = load_moe_layer_fp32(gpu, src, prefix, i, h, n_experts)?;
            let mw = crate::model::moe_weights_of(&ml);
            let md = MoeDims {
                seq,
                h,
                inter,
                n_experts,
                top_k,
            };
            let (moe_out, macts) = moe_forward(gpu, &acts.xn2, &mw, &md).map_err(e)?;
            let (dx2, moe_adj) = moe_backward(gpu, d_x, &mw, &macts, &md).map_err(e)?;
            crate::model::accumulate_gamma_moe(acc, i, &moe_adj, h, n_experts);
            free_moe_acts(gpu, macts).map_err(e)?;
            gpu.free_tensor(moe_out).map_err(e)?;
            drop(mw);
            free_moe_layer_fp32(gpu, ml)?;
            dx2
        } else {
            let (wg, wu, wd, inter) = load_dense_mlp_fp32(gpu, src, prefix, i, h)?;
            let mlp = dense_mlp_forward(gpu, &acts.xn2, &wg, &wu, &wd, seq, h, inter)?;
            let (dx2, d_down) = dense_mlp_backward(gpu, d_x, &wg, &wu, &wd, &mlp, seq, h, inter)?;
            accumulate_gamma_dense_mlp(acc, i, &d_down, seq, h);
            free_dense_mlp_acts(gpu, mlp)?;
            for t in [wg, wu, wd] {
                gpu.free_tensor(t).map_err(e)?;
            }
            dx2
        };
        let (d_in, la_adj) = la_block_backward(gpu, d_x, &d_xn2, x_in, &w, &acts, &d).map_err(e)?;
        accumulate_gamma_linear_attn(acc, i, &la_adj, seq, h, l.n_heads, l.hd_k, l.hd_v);
        free_la_block_acts(gpu, acts).map_err(e)?;
        gpu.free_tensor(d_xn2).map_err(e)?;
        free_linear_attn_layer_fp32(gpu, l)?;
        return Ok(d_in);
    }

    let lw = load_llama_layer_fp32_pfx_off(gpu, src, prefix, cfg, i, !kind.is_moe(), unit_offset)?;
    let bw = crate::model::block_of(&lw);
    let (qd, kvd) = (dims.q_dim(), dims.kv_dim());
    let d_in = if kind.is_moe() {
        let (ml, inter) = load_moe_layer_fp32(gpu, src, prefix, i, h, n_experts)?;
        let mw = crate::model::moe_weights_of(&ml);
        let md = MoeDims {
            seq,
            h,
            inter,
            n_experts,
            top_k,
        };
        let (x_out, acts, macts) = crate::block::moe_block_forward(
            gpu,
            x_in,
            &bw,
            &mw,
            &lora.as_block(),
            dims,
            &md,
            pos_host,
            i,
        )
        .map_err(e)?;
        gpu.free_tensor(x_out).map_err(e)?;
        let (d_in, adj, moe_adj) = crate::block::moe_block_backward_capture(
            gpu,
            d_x,
            x_in,
            &bw,
            &mw,
            &lora.as_block(),
            &acts,
            &macts,
            dims,
            &md,
        )
        .map_err(e)?;
        // inter = 0: this layer has no dense gate/up to key.
        crate::model::accumulate_gamma(acc, i, &adj, seq, h, qd, kvd, 0);
        crate::model::accumulate_gamma_moe(acc, i, &moe_adj, h, n_experts);
        free_moe_acts(gpu, macts).map_err(e)?;
        crate::block::free_block_acts(gpu, acts).map_err(e)?;
        drop(mw);
        free_moe_layer_fp32(gpu, ml)?;
        d_in
    } else {
        let (x_out, acts) =
            crate::block::block_forward(gpu, x_in, &bw, &lora.as_block(), dims, pos_host, i)
                .map_err(e)?;
        gpu.free_tensor(x_out).map_err(e)?;
        let d_down = gpu.download_f32(d_x).map_err(e)?;
        let (d_in, _g, mut adj) = crate::block::block_backward_capture(
            gpu,
            d_x,
            x_in,
            &bw,
            &lora.as_block(),
            &acts,
            dims,
        )
        .map_err(e)?;
        adj.d_down = d_down;
        crate::model::accumulate_gamma(acc, i, &adj, seq, h, qd, kvd, dims.inter);
        crate::block::free_block_acts(gpu, acts).map_err(e)?;
        d_in
    };
    drop(bw);
    free_llama_layer_fp32(gpu, lw)?;
    Ok(d_in)
}

/// Saved dense-MLP state for one layer.
pub struct DenseMlpActs {
    pub gate: GpuTensor,
    pub up: GpuTensor,
    pub act: GpuTensor,
    pub out: GpuTensor,
}

/// `down(swiglu(gate(x), up(x)))` — the dense half of a hybrid layer whose
/// attention side is linear_attn. `crate::block` only offers this fused into a
/// full self-attention block, which a linear_attn layer cannot use.
fn dense_mlp_forward(
    gpu: &mut Gpu,
    xn2: &GpuTensor,
    wg: &GpuTensor,
    wu: &GpuTensor,
    wd: &GpuTensor,
    seq: usize,
    h: usize,
    inter: usize,
) -> Result<DenseMlpActs, String> {
    let e = |r: hipfire_rdna::HipError| format!("{r}");
    let gate = gpu.zeros(&[seq * inter], DType::F32).map_err(e)?;
    let up = gpu.zeros(&[seq * inter], DType::F32).map_err(e)?;
    let act = gpu.zeros(&[seq * inter], DType::F32).map_err(e)?;
    let out = gpu.zeros(&[seq * h], DType::F32).map_err(e)?;
    linear_forward(gpu, xn2, wg, &gate, seq, h, inter).map_err(e)?;
    linear_forward(gpu, xn2, wu, &up, seq, h, inter).map_err(e)?;
    swiglu_forward(gpu, &gate, &up, &act, seq * inter).map_err(e)?;
    linear_forward(gpu, &act, wd, &out, seq, inter, h).map_err(e)?;
    Ok(DenseMlpActs { gate, up, act, out })
}

/// Returns `(d_xn2, d_down)` — the MLP's input gradient, and `down_proj`'s
/// OUTPUT adjoint, which is where every path in this crate takes MLP gamma.
fn dense_mlp_backward(
    gpu: &mut Gpu,
    d_out: &GpuTensor,
    wg: &GpuTensor,
    wu: &GpuTensor,
    wd: &GpuTensor,
    a: &DenseMlpActs,
    seq: usize,
    h: usize,
    inter: usize,
) -> Result<(GpuTensor, Vec<f32>), String> {
    let e = |r: hipfire_rdna::HipError| format!("{r}");
    // x_out = x_mid + mlp(xn2), so down_proj's output adjoint IS d_out.
    let d_down = gpu.download_f32(d_out).map_err(e)?;
    let d_act = gpu.zeros(&[seq * inter], DType::F32).map_err(e)?;
    linear_backward_x(gpu, d_out, wd, &d_act, seq, inter, h, false).map_err(e)?;
    let d_gate = gpu.zeros(&[seq * inter], DType::F32).map_err(e)?;
    let d_up = gpu.zeros(&[seq * inter], DType::F32).map_err(e)?;
    swiglu_backward(gpu, &d_act, &a.gate, &a.up, &d_gate, &d_up, seq * inter).map_err(e)?;
    // gate and up both read xn2, so the second call ACCUMULATES (true).
    let d_xn2 = gpu.zeros(&[seq * h], DType::F32).map_err(e)?;
    linear_backward_x(gpu, &d_gate, wg, &d_xn2, seq, h, inter, false).map_err(e)?;
    linear_backward_x(gpu, &d_up, wu, &d_xn2, seq, h, inter, true).map_err(e)?;
    for t in [d_act, d_gate, d_up] {
        gpu.free_tensor(t).map_err(e)?;
    }
    Ok((d_xn2, d_down))
}

fn free_dense_mlp_acts(gpu: &mut Gpu, a: DenseMlpActs) -> Result<(), String> {
    for t in [a.gate, a.up, a.act, a.out] {
        gpu.free_tensor(t).map_err(|r| format!("{r}"))?;
    }
    Ok(())
}

/// `down_proj`'s output-gradient energy, keyed exactly as the dense and routed
/// paths key theirs so the table joins uniformly.
fn accumulate_gamma_dense_mlp(
    acc: &mut GammaAccum,
    layer: usize,
    d_down: &[f32],
    seq: usize,
    h: usize,
) {
    let mut tot = 0.0f64;
    for r in 0..seq {
        let row = &d_down[r * h..(r + 1) * h];
        tot += row.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>() / h as f64;
    }
    *acc.sum
        .entry(format!("model.layers.{layer}.mlp.down_proj"))
        .or_insert(0.0) += tot / seq.max(1) as f64;
}

/// Group a finished gamma table by layer, for reporting.
pub fn gamma_by_layer(table: &HashMap<String, f32>) -> Vec<(usize, Vec<(String, f32)>)> {
    let mut by: HashMap<usize, Vec<(String, f32)>> = HashMap::new();
    for (k, &v) in table {
        let layer = k
            .strip_prefix("model.layers.")
            .and_then(|r| r.split('.').next())
            .and_then(|n| n.parse::<usize>().ok());
        if let Some(l) = layer {
            by.entry(l).or_default().push((k.clone(), v));
        }
    }
    let mut out: Vec<_> = by.into_iter().collect();
    out.sort_by_key(|(l, _)| *l);
    for (_, v) in out.iter_mut() {
        v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    }
    out
}
