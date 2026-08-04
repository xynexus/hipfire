// SPDX-License-Identifier: Apache-2.0
//! Full dense LLaMA model (fp32) for training: embedding → N pre-norm blocks →
//! final RMSNorm → tied-logit projection → cross-entropy. LoRA on every layer's
//! q_proj/v_proj; all base weights (incl. embedding / tied lm_head) frozen.
//!
//! Built on the gradchecked `block` module — this layer is the embed/head/loss
//! bookends plus the block loop and its reverse.

use crate::block::{
    block_backward, block_backward_capture, block_forward, BlockActivations, BlockAdjoints,
    BlockDims, BlockLora, BlockLoraGrad, BlockWeights,
};
use crate::config::LlamaConfig;
use crate::loader::LlamaWeightsF32;
use crate::ops::cross_entropy::cross_entropy;
use crate::ops::distill::distill_kl;
use crate::ops::linear::{linear_backward_x, linear_forward};
use crate::ops::rmsnorm::{rmsnorm_backward, rmsnorm_forward};
use hipfire_rdna::{DType, Gpu, GpuTensor, HipResult};
use hipfire_runtime::calibration::CalibCollector;

/// Owned frozen weights for one layer.
pub struct LayerWeights {
    pub norm1: GpuTensor,
    pub wq: GpuTensor,
    pub wk: GpuTensor,
    pub wv: GpuTensor,
    pub wo: GpuTensor,
    pub norm2: GpuTensor,
    pub wgate: GpuTensor,
    pub wup: GpuTensor,
    pub wdown: GpuTensor,
}

impl LayerWeights {
    fn as_block(&self) -> BlockWeights<'_> {
        BlockWeights {
            norm1: &self.norm1,
            wq: &self.wq,
            wk: &self.wk,
            wv: &self.wv,
            wo: &self.wo,
            norm2: &self.norm2,
            wgate: &self.wgate,
            wup: &self.wup,
            wdown: &self.wdown,
        }
    }
}

/// Owned trainable LoRA params for one layer.
pub struct LayerLora {
    pub aq: GpuTensor,
    pub bq: GpuTensor,
    pub av: GpuTensor,
    pub bv: GpuTensor,
}

impl LayerLora {
    fn as_block(&self) -> BlockLora<'_> {
        BlockLora {
            aq: &self.aq,
            bq: &self.bq,
            av: &self.av,
            bv: &self.bv,
        }
    }
}

pub struct LlamaModel {
    pub embed: GpuTensor, // [vocab, h] (input embedding; also lm_head when tied)
    pub lm_head: Option<GpuTensor>, // [vocab, h] separate output proj; None ⇒ tied (use embed)
    pub final_norm: GpuTensor, // [h]
    pub layers: Vec<(LayerWeights, LayerLora)>,
    pub dims: BlockDims,
    pub vocab: usize,
}

impl LlamaModel {
    /// Build a trainable model from loaded fp32 base weights. Base weights are
    /// frozen; LoRA adapters are created on q_proj/v_proj of every layer with
    /// `A` small-random and `B = 0` (so the initial LoRA contribution is zero
    /// and the model starts exactly at the base). `seq` is the (fixed) training
    /// sequence length used to populate `BlockDims`. Requires tied embeddings.
    pub fn from_f32_weights(
        gpu: &mut Gpu,
        cfg: &LlamaConfig,
        w: LlamaWeightsF32,
        seq: usize,
        lora_rank: usize,
        lora_alpha: f32,
    ) -> HipResult<Self> {
        // Untied lm_head is carried for the forward logit path (inference / probes);
        // the backward functions only support tied embeddings (guarded there).
        let h = cfg.hidden_size;
        let qd = cfg.q_dim();
        let kvd = cfg.kv_dim();
        let r = lora_rank;

        // Deterministic small-random A init (Kaiming-ish, scaled); B = 0.
        let a_init = |n: usize, seed: usize| -> Vec<f32> {
            (0..n)
                .map(|i| {
                    let x =
                        ((i.wrapping_mul(2654435761).wrapping_add(seed)) % 1000) as f32 / 1000.0;
                    (x - 0.5) * (2.0 / h as f32).sqrt()
                })
                .collect()
        };

        let mut layers = Vec::with_capacity(w.layers.len());
        for (li, lw) in w.layers.into_iter().enumerate() {
            let weights = LayerWeights {
                norm1: lw.input_layernorm,
                wq: lw.q_proj,
                wk: lw.k_proj,
                wv: lw.v_proj,
                wo: lw.o_proj,
                norm2: lw.post_attention_layernorm,
                wgate: lw.gate_proj,
                wup: lw.up_proj,
                wdown: lw.down_proj,
            };
            let lora = LayerLora {
                aq: gpu.upload_f32(&a_init(r * h, li * 7 + 1), &[r * h])?,
                bq: gpu.zeros(&[qd * r], DType::F32)?,
                av: gpu.upload_f32(&a_init(r * h, li * 7 + 3), &[r * h])?,
                bv: gpu.zeros(&[kvd * r], DType::F32)?,
            };
            layers.push((weights, lora));
        }

        let dims = BlockDims {
            seq,
            h,
            n_heads: cfg.num_attention_heads,
            n_kv: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            inter: cfg.intermediate_size,
            rope_base: cfg.rope_theta,
            eps: cfg.rms_norm_eps,
            lora_scale: lora_alpha / r as f32,
            lora_rank: r,
        };

        Ok(Self {
            embed: w.embed_tokens,
            lm_head: w.lm_head,
            final_norm: w.final_norm,
            layers,
            dims,
            vocab: cfg.vocab_size,
        })
    }

    /// Flat list of trainable LoRA params, layer-major `[aq,bq,av,bv]` — the
    /// order the optimizer and `model_loss_backward`'s grads use.
    pub fn lora_params(&self) -> Vec<&GpuTensor> {
        let mut v = Vec::with_capacity(self.layers.len() * 4);
        for (_, l) in &self.layers {
            v.push(&l.aq);
            v.push(&l.bq);
            v.push(&l.av);
            v.push(&l.bv);
        }
        v
    }

    /// Element counts of `lora_params()`, for `AdamW::new`.
    pub fn lora_param_sizes(&self) -> Vec<usize> {
        self.lora_params()
            .iter()
            .map(|t| t.shape.iter().product())
            .collect()
    }

    /// Trainable params for QTIP recovery FT: per layer `[aq,bq,av,bv,norm1,
    /// norm2]`, then the final norm. Matches `flatten_recovery_grads` order.
    pub fn recovery_params(&self) -> Vec<&GpuTensor> {
        let mut v = Vec::with_capacity(self.layers.len() * 6 + 1);
        for (w, l) in &self.layers {
            v.push(&l.aq);
            v.push(&l.bq);
            v.push(&l.av);
            v.push(&l.bv);
            v.push(&w.norm1);
            v.push(&w.norm2);
        }
        v.push(&self.final_norm);
        v
    }

    pub fn recovery_param_sizes(&self) -> Vec<usize> {
        self.recovery_params()
            .iter()
            .map(|t| t.shape.iter().product())
            .collect()
    }

    /// Layernorm-only trainable params (faithful QTIP recovery, no LoRA): per
    /// layer `[norm1, norm2]`, then the final norm. Matches `flatten_norm_grads`.
    /// This is the Path-A export set — tuning only the fp norms leaves the
    /// trellis codes exportable as-is (lossless servable export).
    pub fn norm_params(&self) -> Vec<&GpuTensor> {
        let mut v = Vec::with_capacity(self.layers.len() * 2 + 1);
        for (w, _) in &self.layers {
            v.push(&w.norm1);
            v.push(&w.norm2);
        }
        v.push(&self.final_norm);
        v
    }

    pub fn norm_param_sizes(&self) -> Vec<usize> {
        self.norm_params()
            .iter()
            .map(|t| t.shape.iter().product())
            .collect()
    }
}

/// Flatten norm grads to match `norm_params()`: per layer `[dnorm1, dnorm2]`,
/// then `d_final_norm`.
pub fn flatten_norm_grads<'a>(
    grads: &'a [BlockLoraGrad],
    d_final_norm: &'a GpuTensor,
) -> Vec<&'a GpuTensor> {
    let mut v = Vec::with_capacity(grads.len() * 2 + 1);
    for g in grads {
        v.push(&g.dnorm1);
        v.push(&g.dnorm2);
    }
    v.push(d_final_norm);
    v
}

/// Flatten recovery grads to match `recovery_params()`: per layer
/// `[daq,dbq,dav,dbv,dnorm1,dnorm2]`, then `d_final_norm`.
pub fn flatten_recovery_grads<'a>(
    grads: &'a [BlockLoraGrad],
    d_final_norm: &'a GpuTensor,
) -> Vec<&'a GpuTensor> {
    let mut v = Vec::with_capacity(grads.len() * 6 + 1);
    for g in grads {
        v.push(&g.daq);
        v.push(&g.dbq);
        v.push(&g.dav);
        v.push(&g.dbv);
        v.push(&g.dnorm1);
        v.push(&g.dnorm2);
    }
    v.push(d_final_norm);
    v
}

/// Saved forward state for the backward pass.
pub struct ModelActivations {
    pub layer_inputs: Vec<GpuTensor>, // input to each block (layer_inputs[0] = embedding)
    pub layer_acts: Vec<BlockActivations>,
    pub x_last: GpuTensor, // output of last block (input to final norm)
    pub rinv_final: GpuTensor,
    pub xf: GpuTensor,     // final-norm output (logit input)
    pub logits: GpuTensor, // [seq, vocab]
}

/// Return a forward's activations to the pool — GpuTensor has no Drop, so a loop
/// of `model_forward` calls (e.g. capturing labels over a corpus) climbs ~2 GB/
/// forward and OOMs without this.
pub fn free_model_acts(gpu: &mut Gpu, a: ModelActivations) -> HipResult<()> {
    let ModelActivations {
        layer_inputs,
        layer_acts,
        x_last,
        rinv_final,
        xf,
        logits,
    } = a;
    for t in layer_inputs {
        gpu.free_tensor(t)?;
    }
    for b in layer_acts {
        let BlockActivations {
            xn1,
            rinv1,
            hq,
            hv,
            q_r,
            k_r,
            v,
            p_all,
            ctx,
            x_mid,
            xn2,
            rinv2,
            gate,
            up,
            act,
            pos,
        } = b;
        for t in [
            xn1, rinv1, hq, hv, q_r, k_r, v, p_all, ctx, x_mid, xn2, rinv2, gate, up, act, pos,
        ] {
            gpu.free_tensor(t)?;
        }
    }
    for t in [x_last, rinv_final, xf, logits] {
        gpu.free_tensor(t)?;
    }
    Ok(())
}

/// Partial forward: embed + blocks `0..=up_to`, returning each block's saved
/// activations (incl. `k_r`, the post-rope K). Skips the final norm + logit GEMM
/// — cheap when you only need intermediate K (e.g. PFlash-style shallow-layer
/// block scoring; the teacher only needs the target's first few layers).
pub fn model_block_activations(
    gpu: &mut Gpu,
    model: &LlamaModel,
    token_ids: &[u32],
    pos_host: &[f32],
    up_to: usize,
) -> HipResult<Vec<BlockActivations>> {
    let (seq, h) = (model.dims.seq, model.dims.h);
    assert_eq!(token_ids.len(), seq);
    let x0 = gpu.zeros(&[seq * h], DType::F32)?;
    for (t, &tok) in token_ids.iter().enumerate() {
        gpu.strided_copy_2d(
            &model.embed,
            tok as usize * h,
            h,
            &x0,
            t * h,
            h,
            1,
            h,
            false,
        )?;
    }
    let last = up_to.min(model.layers.len() - 1);
    let mut out = Vec::with_capacity(last + 1);
    let mut x = x0;
    for i in 0..=last {
        let (lw, ll) = &model.layers[i];
        let (x_out, acts) = block_forward(
            gpu,
            &x,
            &lw.as_block(),
            &ll.as_block(),
            &model.dims,
            pos_host,
            i,
        )?;
        out.push(acts);
        x = x_out;
    }
    Ok(out)
}

/// Forward through logits (no loss). `token_ids.len()` must equal `dims.seq`.
pub fn model_forward(
    gpu: &mut Gpu,
    model: &LlamaModel,
    token_ids: &[u32],
    pos_host: &[f32],
) -> HipResult<ModelActivations> {
    let (seq, h) = (model.dims.seq, model.dims.h);
    assert_eq!(token_ids.len(), seq);

    // Embedding lookup: gather row token_ids[t] of embed into x0[t].
    let x0 = gpu.zeros(&[seq * h], DType::F32)?;
    for (t, &tok) in token_ids.iter().enumerate() {
        gpu.strided_copy_2d(
            &model.embed,
            tok as usize * h,
            h,
            &x0,
            t * h,
            h,
            1,
            h,
            false,
        )?;
    }

    let mut layer_inputs = Vec::with_capacity(model.layers.len());
    let mut layer_acts = Vec::with_capacity(model.layers.len());
    let mut x = x0;
    for (i, (lw, ll)) in model.layers.iter().enumerate() {
        layer_inputs.push(clone_tensor(gpu, &x)?);
        let (x_out, acts) = block_forward(
            gpu,
            &x,
            &lw.as_block(),
            &ll.as_block(),
            &model.dims,
            pos_host,
            i,
        )?;
        layer_acts.push(acts);
        x = x_out;
    }
    let x_last = x;

    // final norm → logits = xf · embedᵀ (tied)
    let xf = gpu.zeros(&[seq * h], DType::F32)?;
    let rinv_final = gpu.zeros(&[seq], DType::F32)?;
    rmsnorm_forward(
        gpu,
        &x_last,
        &model.final_norm,
        &xf,
        &rinv_final,
        seq,
        h,
        model.dims.eps,
    )?;
    let logits = gpu.zeros(&[seq * model.vocab], DType::F32)?;
    let out_proj = model.lm_head.as_ref().unwrap_or(&model.embed);
    linear_forward(gpu, &xf, out_proj, &logits, seq, h, model.vocab)?;

    Ok(ModelActivations {
        layer_inputs,
        layer_acts,
        x_last,
        rinv_final,
        xf,
        logits,
    })
}

/// Cross-entropy loss (summed over non-ignored tokens) + full backward.
/// Returns the summed loss and per-layer LoRA gradients. `targets` are
/// integer-valued f32 (or `ignore_index`).
pub fn model_loss_backward(
    gpu: &mut Gpu,
    model: &LlamaModel,
    acts: &ModelActivations,
    targets: &[f32],
    ignore_index: i32,
) -> HipResult<(f32, Vec<BlockLoraGrad>)> {
    let (seq, h, vocab) = (model.dims.seq, model.dims.h, model.vocab);

    // Loss + d_logits (sum-reduction).
    let tgt = gpu.upload_f32(targets, &[seq])?;
    let loss = gpu.zeros(&[seq], DType::F32)?;
    let d_logits = gpu.zeros(&[seq * vocab], DType::F32)?;
    cross_entropy(
        gpu,
        &acts.logits,
        &tgt,
        &loss,
        &d_logits,
        seq,
        vocab,
        ignore_index,
    )?;
    let loss_sum: f32 = gpu.download_f32(&loss)?.iter().sum();

    // d_xf = d_logits · out_proj. Base weights are frozen, so the only use of the
    // output matrix in backward is this term — `out_proj = lm_head ?? embed`
    // (mirrors model_forward), which handles both tied and untied heads (the
    // latter produced by rotation::apply_r1).
    let out_proj = model.lm_head.as_ref().unwrap_or(&model.embed);
    let d_xf = gpu.zeros(&[seq * h], DType::F32)?;
    linear_backward_x(gpu, &d_logits, out_proj, &d_xf, seq, h, vocab, false)?;
    // final norm backward → d_x_last
    let d_x_last = gpu.zeros(&[seq * h], DType::F32)?;
    let dw_dummy = gpu.zeros(&[h], DType::F32)?;
    rmsnorm_backward(
        gpu,
        &d_xf,
        &acts.x_last,
        &model.final_norm,
        &acts.rinv_final,
        &d_x_last,
        &dw_dummy,
        seq,
        h,
    )?;

    // Walk blocks in reverse.
    let mut grads: Vec<BlockLoraGrad> = Vec::with_capacity(model.layers.len());
    let mut d_x = d_x_last;
    for i in (0..model.layers.len()).rev() {
        let (lw, ll) = &model.layers[i];
        let (d_in, g) = block_backward(
            gpu,
            &d_x,
            &acts.layer_inputs[i],
            &lw.as_block(),
            &ll.as_block(),
            &acts.layer_acts[i],
            &model.dims,
        )?;
        grads.push(g);
        d_x = d_in;
    }
    grads.reverse(); // align with layer order
    Ok((loss_sum, grads))
}

/// Forward+backward that returns the per-layer per-linear OUTPUT adjoints (∂ℓ/∂z)
/// instead of the LoRA grads — the raw material for GuidedQuant Fisher weights. Same
/// reverse walk as [`model_loss_backward`] but via [`block_backward_capture`]. Layer
/// order. `acts` must be the forward of the SAME model.
pub fn model_guided_adjoints(
    gpu: &mut Gpu,
    model: &LlamaModel,
    acts: &ModelActivations,
    targets: &[f32],
    ignore_index: i32,
) -> HipResult<(f32, Vec<BlockAdjoints>)> {
    let (seq, h, vocab) = (model.dims.seq, model.dims.h, model.vocab);
    let tgt = gpu.upload_f32(targets, &[seq])?;
    let loss = gpu.zeros(&[seq], DType::F32)?;
    let d_logits = gpu.zeros(&[seq * vocab], DType::F32)?;
    cross_entropy(
        gpu,
        &acts.logits,
        &tgt,
        &loss,
        &d_logits,
        seq,
        vocab,
        ignore_index,
    )?;
    let loss_sum: f32 = gpu.download_f32(&loss)?.iter().sum();

    let out_proj = model.lm_head.as_ref().unwrap_or(&model.embed);
    let d_xf = gpu.zeros(&[seq * h], DType::F32)?;
    linear_backward_x(gpu, &d_logits, out_proj, &d_xf, seq, h, vocab, false)?;
    let d_x_last = gpu.zeros(&[seq * h], DType::F32)?;
    let dw_dummy = gpu.zeros(&[h], DType::F32)?;
    rmsnorm_backward(
        gpu,
        &d_xf,
        &acts.x_last,
        &model.final_norm,
        &acts.rinv_final,
        &d_x_last,
        &dw_dummy,
        seq,
        h,
    )?;

    let mut adj: Vec<BlockAdjoints> = Vec::with_capacity(model.layers.len());
    let mut d_x = d_x_last;
    for i in (0..model.layers.len()).rev() {
        let (lw, ll) = &model.layers[i];
        // down_proj's output adjoint is the block's INCOMING d_x — its output is
        // the block output pre-residual — so it has to be read before the
        // backward consumes it. This is the same fact `down_guided_capture`
        // relies on; without it down_proj is absent from the adjoint set.
        let d_down = gpu.download_f32(&d_x)?;
        let (d_in, _lora_g, mut a) = block_backward_capture(
            gpu,
            &d_x,
            &acts.layer_inputs[i],
            &lw.as_block(),
            &ll.as_block(),
            &acts.layer_acts[i],
            &model.dims,
        )?;
        a.d_down = d_down;
        adj.push(a);
        d_x = d_in;
    }
    adj.reverse();
    Ok((loss_sum, adj))
}

/// GuidedQuant down_proj capture: form the per-token Fisher weight
/// `w[n] = mean_c (∂ℓ/∂z)²` from the down output-grad `d_out [seq,h]`, normalize
/// so `mean(w)=1` (scale-invariant for LDLQ, keeps H̄ ~ plain-H magnitude), and
/// accumulate the weighted Hessian `H̄ = Σ wₙ·actₙactₙᵀ` for this layer's
/// `down_proj` into the collector (`act_in [seq,inter]` is down's input).
#[allow(clippy::too_many_arguments)]
fn down_guided_capture(
    gpu: &mut Gpu,
    collector: &CalibCollector,
    layer: usize,
    d_out: &GpuTensor,
    act_in: &GpuTensor,
    seq: usize,
    h: usize,
    inter: usize,
    fisher: bool,
) -> HipResult<()> {
    let w = if fisher {
        let w = gpu.zeros(&[seq], DType::F32)?;
        gpu.calib_row_meansq_f32(d_out, &w, seq, h)?;
        let mean: f32 = gpu.download_f32(&w)?.iter().sum::<f32>() / seq.max(1) as f32;
        if mean > 0.0 {
            gpu.scale_f32(&w, 1.0 / mean)?;
        }
        w
    } else {
        // Control: w≡1 ⇒ capture_weighted produces the plain XᵀX over the same
        // tokens — the apples-to-apples baseline for the guided-vs-plain compare.
        gpu.upload_f32(&vec![1.0f32; seq], &[seq])?
    };
    let name = format!("model.layers.{layer}.mlp.down_proj");
    collector.capture_weighted(gpu, &name, act_in, &w, seq, inter);
    gpu.free_tensor(w)?;
    Ok(())
}

/// Like [`model_loss_backward`], but during the reverse block walk it captures
/// the **GuidedQuant Fisher-weighted Hessian** for each layer's `down_proj` into
/// `collector` (down's output adjoint is the loop's `d_x`, its input is
/// `acts.layer_acts[i].act`). The LoRA grads are discarded — this path exists
/// only to drive the weighted-Hessian accumulation, not an optimizer step.
/// First move of the calibration-backward path (down_proj only).
pub fn model_calib_down_backward(
    gpu: &mut Gpu,
    model: &LlamaModel,
    acts: &ModelActivations,
    targets: &[f32],
    ignore_index: i32,
    collector: &CalibCollector,
    fisher: bool,
) -> HipResult<f32> {
    let (seq, h, vocab) = (model.dims.seq, model.dims.h, model.vocab);
    let inter = model.dims.inter;

    let tgt = gpu.upload_f32(targets, &[seq])?;
    let loss = gpu.zeros(&[seq], DType::F32)?;
    let d_logits = gpu.zeros(&[seq * vocab], DType::F32)?;
    cross_entropy(
        gpu,
        &acts.logits,
        &tgt,
        &loss,
        &d_logits,
        seq,
        vocab,
        ignore_index,
    )?;
    let loss_sum: f32 = gpu.download_f32(&loss)?.iter().sum();

    // out_proj = lm_head ?? embed (frozen); handles tied + untied heads.
    let out_proj = model.lm_head.as_ref().unwrap_or(&model.embed);
    let d_xf = gpu.zeros(&[seq * h], DType::F32)?;
    linear_backward_x(gpu, &d_logits, out_proj, &d_xf, seq, h, vocab, false)?;
    let d_x_last = gpu.zeros(&[seq * h], DType::F32)?;
    let dw_dummy = gpu.zeros(&[h], DType::F32)?;
    rmsnorm_backward(
        gpu,
        &d_xf,
        &acts.x_last,
        &model.final_norm,
        &acts.rinv_final,
        &d_x_last,
        &dw_dummy,
        seq,
        h,
    )?;

    let mut d_x = d_x_last;
    for i in (0..model.layers.len()).rev() {
        // Capture down_proj BEFORE block_backward consumes d_x: here d_x is the
        // grad w.r.t. this block's output = down_proj's output adjoint.
        down_guided_capture(
            gpu,
            collector,
            i,
            &d_x,
            &acts.layer_acts[i].act,
            seq,
            h,
            inter,
            fisher,
        )?;
        let (lw, ll) = &model.layers[i];
        let (d_in, _g) = block_backward(
            gpu,
            &d_x,
            &acts.layer_inputs[i],
            &lw.as_block(),
            &ll.as_block(),
            &acts.layer_acts[i],
            &model.dims,
        )?;
        d_x = d_in;
    }
    Ok(loss_sum)
}

/// Mean squared OUTPUT gradient per linear, accumulated over calibration
/// sequences — the `gamma` in the K-FAC factorization
/// `dL ~= 1/2 tr(dW^T G dW H)` under the approximation `G ~= gamma*I`.
///
/// Why this is not already available: [`down_guided_capture`] computes exactly
/// this quantity and then NORMALIZES it away (`w /= mean`) before handing the
/// weights to `capture_weighted`. That is right for GuidedQuant, which wants
/// which TOKENS matter within a layer and needs guided/plain Hessians on a
/// comparable scale. But it discards the per-tensor MAGNITUDE, and the
/// magnitude is the whole cross-tensor signal — it is what says o_proj matters
/// more than k_proj.
///
/// Keyed by HFQ tensor name without `.weight`, matching the imatrix/hessian
/// convention so the quantizer joins them the same way.
#[derive(Debug, Default, Clone)]
pub struct GammaAccum {
    pub sum: std::collections::HashMap<String, f64>,
    pub n: usize,
}

impl GammaAccum {
    /// Mean over accumulated sequences.
    pub fn finish(&self) -> std::collections::HashMap<String, f32> {
        let d = self.n.max(1) as f64;
        self.sum
            .iter()
            .map(|(k, v)| (k.clone(), (v / d) as f32))
            .collect()
    }
}

/// Forward+backward accumulating per-linear output-gradient energy for every
/// projection, via [`block_backward_capture`].
///
/// The plain [`model_calib_down_backward`] path uses `block_backward` and can
/// only see `down_proj`, because down_proj's output IS the block output
/// pre-residual and its adjoint is `d_x` before the block consumes it. Every
/// other projection's adjoint exists only inside the backward, which is what
/// [`BlockAdjoints`] exposes.
pub fn model_gamma_backward(
    gpu: &mut Gpu,
    model: &LlamaModel,
    acts: &ModelActivations,
    targets: &[f32],
    ignore_index: i32,
    acc: &mut GammaAccum,
) -> HipResult<f32> {
    let (loss_sum, adjoints) = model_guided_adjoints(gpu, model, acts, targets, ignore_index)?;
    let (seq, h) = (model.dims.seq, model.dims.h);
    // q is n_heads*head_dim, k/v are n_kv*head_dim (GQA).
    let qd = model.dims.n_heads * model.dims.head_dim;
    let kvd = model.dims.n_kv * model.dims.head_dim;
    let inter = model.dims.inter;

    for (i, a) in adjoints.iter().enumerate() {
        let p = format!("model.layers.{i}");
        // (name, adjoint rows, per-row width). Mean over rows of the
        // mean-over-channels square — the same statistic
        // `calib_row_meansq_f32` forms, kept UNNORMALISED.
        for (name, d, width) in [
            (format!("{p}.self_attn.q_proj"), &a.d_q, qd),
            (format!("{p}.self_attn.k_proj"), &a.d_k, kvd),
            (format!("{p}.self_attn.v_proj"), &a.d_v, kvd),
            (format!("{p}.self_attn.o_proj"), &a.d_attn, h),
            (format!("{p}.mlp.gate_proj"), &a.d_gate, inter),
            (format!("{p}.mlp.up_proj"), &a.d_up, inter),
            (format!("{p}.mlp.down_proj"), &a.d_down, h),
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
    acc.n += 1;
    Ok(loss_sum)
}

/// Layer-STREAMED gamma capture: one layer resident at a time.
///
/// [`model_gamma_backward`] needs the whole model in f32 — ~5 GB for a 1B, but
/// ~140 GB for a 35B against 128 GB of UMA, so it is a small-model instrument.
/// Here each layer's weights are paged in, used, and dropped, so peak residency
/// is the embedding table plus ONE layer plus the boundary activations
/// (seq*h floats per layer, negligible). Depth becomes free.
///
/// Two passes over the layers, which is the standard gradient-checkpointing
/// trade: the forward saves only each layer's INPUT, and the reverse walk
/// recomputes that layer's internals from it rather than storing every
/// activation for every layer. Costs one extra forward, saves O(depth) memory.
///
/// DENSE ONLY. `block_forward`/`block_backward_capture` are a pre-norm LLaMA
/// block with no router or routed experts, so this cannot run a MoE — see
/// docs/npu/scope-gamma-layer-streamed.md.
#[allow(clippy::too_many_arguments)]
pub fn model_gamma_streamed(
    gpu: &mut Gpu,
    hfq: &hipfire_runtime::hfq::HfqFile,
    cfg: &crate::config::LlamaConfig,
    embed: &GpuTensor,
    lm_head: Option<&GpuTensor>,
    final_norm: &GpuTensor,
    dims: &BlockDims,
    token_ids: &[u32],
    pos_host: &[f32],
    targets: &[f32],
    ignore_index: i32,
    // 0 for a dense model. Routed-ness is still probed PER LAYER — hybrid
    // models exist (BLS-Mini-Code-1.0 is dense at layer 0, routed above).
    n_experts: usize,
    top_k: usize,
    acc: &mut GammaAccum,
) -> Result<f32, String> {
    use crate::loader::{
        detect_prefix, free_llama_layer_fp32, free_moe_layer_fp32, layer_is_moe,
        load_llama_layer_fp32_hfq_pfx, load_moe_layer_fp32_hfq,
    };
    let prefix = detect_prefix(hfq);
    let (seq, h) = (dims.seq, dims.h);
    let vocab = cfg.vocab_size;
    let n_layers = cfg.num_hidden_layers;
    let e = |r: hipfire_rdna::HipError| format!("{r}");

    // Zero LoRA (rank 1, B = 0) so the block runs exactly at the base weights.
    let r = dims.lora_rank.max(1);
    let qd = dims.q_dim();
    let kvd = dims.kv_dim();
    let zl = |gpu: &mut Gpu, n: usize| gpu.zeros(&[n], DType::F32).map_err(e);
    let lora = LayerLora {
        aq: zl(gpu, r * h)?,
        bq: zl(gpu, qd * r)?,
        av: zl(gpu, r * h)?,
        bv: zl(gpu, kvd * r)?,
    };

    // Embedding lookup.
    let mut x = gpu.zeros(&[seq * h], DType::F32).map_err(e)?;
    for (t, &tok) in token_ids.iter().enumerate() {
        // (src, src_off, src_stride, dst, dst_off, dst_stride, rows, cols, accumulate)
        gpu.strided_copy_2d(embed, tok as usize * h, h, &x, t * h, h, 1, h, false)
            .map_err(e)?;
    }

    // Forward, keeping only each layer's INPUT on the host.
    let mut layer_inputs: Vec<Vec<f32>> = Vec::with_capacity(n_layers);
    for i in 0..n_layers {
        layer_inputs.push(gpu.download_f32(&x).map_err(e)?);
        let lw = load_llama_layer_fp32_hfq_pfx(gpu, hfq, prefix, cfg, i, true)?;
        let (x_out, acts) =
            block_forward(gpu, &x, &block_of(&lw), &lora.as_block(), dims, pos_host, i)
                .map_err(e)?;
        crate::block::free_block_acts(gpu, acts).map_err(e)?;
        gpu.free_tensor(x).map_err(e)?;
        x = x_out;
        free_llama_layer_fp32(gpu, lw)?;
    }

    // Head: final norm -> logits -> CE -> back to the last block's output.
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

    // Reverse walk: page the layer back in, recompute its internals from the
    // saved input, capture the adjoints, drop it.
    for i in (0..n_layers).rev() {
        let moe = n_experts > 0 && layer_is_moe(hfq, prefix, i);
        let lw = load_llama_layer_fp32_hfq_pfx(gpu, hfq, prefix, cfg, i, !moe)?;
        let x_in = gpu.upload_f32(&layer_inputs[i], &[seq * h]).map_err(e)?;

        let d_in = if moe {
            let (ml, inter) = load_moe_layer_fp32_hfq(gpu, hfq, prefix, i, h, n_experts)?;
            let mw = moe_weights_of(&ml);
            let md = crate::ops::moe::MoeDims {
                seq,
                h,
                inter,
                n_experts,
                top_k,
            };
            let (x_out, acts, macts) = crate::block::moe_block_forward(
                gpu,
                &x_in,
                &block_of(&lw),
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
                &d_x,
                &x_in,
                &block_of(&lw),
                &mw,
                &lora.as_block(),
                &acts,
                &macts,
                dims,
                &md,
            )
            .map_err(e)?;
            // Attention-side gamma is identical to the dense case. The MLP side
            // comes from the experts rather than one down_proj, so `d_down` is
            // left empty and `accumulate_gamma` is passed inter = 0 to skip the
            // dense gate/up entries that do not exist on this layer.
            accumulate_gamma(acc, i, &adj, seq, h, qd, kvd, 0);
            accumulate_gamma_moe(acc, i, &moe_adj, h, n_experts);
            crate::block::free_block_acts(gpu, acts).map_err(e)?;
            crate::ops::moe::free_moe_acts(gpu, macts).map_err(e)?;
            drop(mw);
            free_moe_layer_fp32(gpu, ml)?;
            d_in
        } else {
            let (x_out, acts) = block_forward(
                gpu,
                &x_in,
                &block_of(&lw),
                &lora.as_block(),
                dims,
                pos_host,
                i,
            )
            .map_err(e)?;
            gpu.free_tensor(x_out).map_err(e)?;
            let d_down = gpu.download_f32(&d_x).map_err(e)?;
            let (d_in, _g, mut adj) = block_backward_capture(
                gpu,
                &d_x,
                &x_in,
                &block_of(&lw),
                &lora.as_block(),
                &acts,
                dims,
            )
            .map_err(e)?;
            adj.d_down = d_down;
            accumulate_gamma(acc, i, &adj, seq, h, qd, kvd, dims.inter);
            crate::block::free_block_acts(gpu, acts).map_err(e)?;
            d_in
        };

        gpu.free_tensor(x_in).map_err(e)?;
        gpu.free_tensor(d_x).map_err(e)?;
        d_x = d_in;
        free_llama_layer_fp32(gpu, lw)?;
    }
    acc.n += 1;
    Ok(loss_sum)
}

/// `BlockWeights` borrowed from a streamed `LlamaLayerF32`, so the streamed walk
/// needs no owned `LayerWeights` copy of a layer it is about to drop.
fn block_of(l: &crate::loader::LlamaLayerF32) -> BlockWeights<'_> {
    BlockWeights {
        norm1: &l.input_layernorm,
        wq: &l.q_proj,
        wk: &l.k_proj,
        wv: &l.v_proj,
        wo: &l.o_proj,
        norm2: &l.post_attention_layernorm,
        wgate: &l.gate_proj,
        wup: &l.up_proj,
        wdown: &l.down_proj,
    }
}

/// `MoeWeights` borrowed from a streamed layer, so nothing is copied for a
/// layer that is about to be dropped.
fn moe_weights_of(l: &crate::loader::MoeLayerF32) -> crate::ops::moe::MoeWeights<'_> {
    crate::ops::moe::MoeWeights {
        router: &l.router,
        experts: l
            .experts
            .iter()
            .map(|(g, u, d)| crate::ops::moe::ExpertWeights {
                wgate: g,
                wup: u,
                wdown: d,
            })
            .collect(),
    }
}

/// Fold one layer's adjoints into the accumulator. Shared with
/// [`model_gamma_backward`] so the streamed and whole-model paths cannot drift.
pub fn accumulate_gamma(
    acc: &mut GammaAccum,
    layer: usize,
    a: &BlockAdjoints,
    seq: usize,
    h: usize,
    qd: usize,
    kvd: usize,
    inter: usize,
) {
    let p = format!("model.layers.{layer}");
    for (name, d, width) in [
        (format!("{p}.self_attn.q_proj"), &a.d_q, qd),
        (format!("{p}.self_attn.k_proj"), &a.d_k, kvd),
        (format!("{p}.self_attn.v_proj"), &a.d_v, kvd),
        (format!("{p}.self_attn.o_proj"), &a.d_attn, h),
        (format!("{p}.mlp.gate_proj"), &a.d_gate, inter),
        (format!("{p}.mlp.up_proj"), &a.d_up, inter),
        (format!("{p}.mlp.down_proj"), &a.d_down, h),
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

/// Fold one MoE layer's per-expert adjoints into the accumulator.
///
/// Keys match the HFQ tensor naming a routed MoE artifact uses, so the table
/// joins to weights exactly as the dense one does.
///
/// **Normalised by ROUTED tokens, not by sequence length.** An expert only sees
/// the tokens the router sent it, so dividing by `seq` would report a rarely
/// routed expert as insensitive when that is a routing fact, not a sensitivity
/// fact — and the allocator would then under-promote precisely the experts that
/// are sharp on the few tokens they serve. An expert that received nothing
/// contributes nothing rather than a zero, so it does not drag its own mean
/// down across calibration sequences.
pub fn accumulate_gamma_moe(
    acc: &mut GammaAccum,
    layer: usize,
    adj: &crate::ops::moe::MoeAdjoints,
    h: usize,
    n_experts: usize,
) {
    for e in 0..n_experts {
        let d = &adj.d_expert_out[e];
        let rows = d.len() / h.max(1);
        if rows == 0 {
            continue;
        }
        let mut tot = 0.0f64;
        for r in 0..rows {
            let row = &d[r * h..(r + 1) * h];
            let ss: f64 = row.iter().map(|&x| (x as f64) * (x as f64)).sum();
            tot += ss / h as f64;
        }
        // The routed-expert down_proj is where the expert's error leaves for the
        // residual stream, matching where the dense path takes d_down.
        let name = format!("model.layers.{layer}.mlp.experts.{e}.down_proj");
        *acc.sum.entry(name).or_insert(0.0) += tot / rows as f64;
    }
}

/// Flatten per-layer LoRA grads into the same order as `LlamaModel::lora_params`.
pub fn flatten_lora_grads(grads: &[BlockLoraGrad]) -> Vec<&GpuTensor> {
    let mut v = Vec::with_capacity(grads.len() * 4);
    for g in grads {
        v.push(&g.daq);
        v.push(&g.dbq);
        v.push(&g.dav);
        v.push(&g.dbv);
    }
    v
}

/// Distillation forward-loss + backward against a teacher distribution
/// `teacher_p` `[seq*vocab]` (probabilities). Returns the summed KL, per-layer
/// grads (LoRA + layernorms), and the final-norm grad. The student's base
/// linears stay frozen.
pub fn model_distill_backward(
    gpu: &mut Gpu,
    model: &LlamaModel,
    acts: &ModelActivations,
    teacher_p: &GpuTensor,
) -> HipResult<(f32, Vec<BlockLoraGrad>, GpuTensor)> {
    let (seq, h, vocab) = (model.dims.seq, model.dims.h, model.vocab);

    let loss = gpu.zeros(&[seq], DType::F32)?;
    let d_logits = gpu.zeros(&[seq * vocab], DType::F32)?;
    distill_kl(gpu, &acts.logits, teacher_p, &loss, &d_logits, seq, vocab)?;
    let loss_sum: f32 = gpu.download_f32(&loss)?.iter().sum();

    // out_proj = lm_head ?? embed (frozen); handles tied + untied heads.
    let out_proj = model.lm_head.as_ref().unwrap_or(&model.embed);
    let d_xf = gpu.zeros(&[seq * h], DType::F32)?;
    linear_backward_x(gpu, &d_logits, out_proj, &d_xf, seq, h, vocab, false)?;
    let d_x_last = gpu.zeros(&[seq * h], DType::F32)?;
    let d_final_norm = gpu.zeros(&[h], DType::F32)?;
    rmsnorm_backward(
        gpu,
        &d_xf,
        &acts.x_last,
        &model.final_norm,
        &acts.rinv_final,
        &d_x_last,
        &d_final_norm,
        seq,
        h,
    )?;

    let mut grads: Vec<BlockLoraGrad> = Vec::with_capacity(model.layers.len());
    let mut d_x = d_x_last;
    for i in (0..model.layers.len()).rev() {
        let (lw, ll) = &model.layers[i];
        let (d_in, g) = block_backward(
            gpu,
            &d_x,
            &acts.layer_inputs[i],
            &lw.as_block(),
            &ll.as_block(),
            &acts.layer_acts[i],
            &model.dims,
        )?;
        grads.push(g);
        d_x = d_in;
    }
    grads.reverse();
    Ok((loss_sum, grads, d_final_norm))
}

fn clone_tensor(gpu: &mut Gpu, t: &GpuTensor) -> HipResult<GpuTensor> {
    let out = gpu.zeros(&t.shape, DType::F32)?;
    let bytes: usize = t.shape.iter().product::<usize>() * 4;
    gpu.memcpy_dtod_auto(&out.buf, &t.buf, bytes)?;
    Ok(out)
}

/// Like `model_distill_backward` but scores ONLY the last `n_score` query
/// positions (zeros the loss/gradient on earlier rows). Used by the KVarN+CASK
/// retest: in a full-sequence forward, only tail queries read merged cold keys
/// that are strictly in their causal past, so scoring the tail is leak-free
/// (an earlier query inside a merged fold would otherwise attend to a key
/// blended with its own future — an acausal leak that confounds the measurement).
pub fn model_distill_backward_tail(
    gpu: &mut Gpu,
    model: &LlamaModel,
    acts: &ModelActivations,
    teacher_p: &GpuTensor,
    n_score: usize,
) -> HipResult<(f32, Vec<BlockLoraGrad>, GpuTensor)> {
    let (seq, h, vocab) = (model.dims.seq, model.dims.h, model.vocab);
    let n_score = n_score.min(seq).max(1);
    let first_scored = seq - n_score;

    let loss = gpu.zeros(&[seq], DType::F32)?;
    let d_logits = gpu.zeros(&[seq * vocab], DType::F32)?;
    distill_kl(gpu, &acts.logits, teacher_p, &loss, &d_logits, seq, vocab)?;
    // Mask to the last n_score rows (host zero — correctness over speed).
    let mut dl = gpu.download_f32(&d_logits)?;
    for v in dl[..first_scored * vocab].iter_mut() {
        *v = 0.0;
    }
    gpu.free_tensor(d_logits)?;
    let d_logits = gpu.upload_f32(&dl, &[seq * vocab])?;
    let loss_sum: f32 = gpu.download_f32(&loss)?[first_scored..].iter().sum();

    // out_proj = lm_head ?? embed (frozen); handles tied + untied heads.
    let out_proj = model.lm_head.as_ref().unwrap_or(&model.embed);
    let d_xf = gpu.zeros(&[seq * h], DType::F32)?;
    linear_backward_x(gpu, &d_logits, out_proj, &d_xf, seq, h, vocab, false)?;
    let d_x_last = gpu.zeros(&[seq * h], DType::F32)?;
    let d_final_norm = gpu.zeros(&[h], DType::F32)?;
    rmsnorm_backward(
        gpu,
        &d_xf,
        &acts.x_last,
        &model.final_norm,
        &acts.rinv_final,
        &d_x_last,
        &d_final_norm,
        seq,
        h,
    )?;

    let mut grads: Vec<BlockLoraGrad> = Vec::with_capacity(model.layers.len());
    let mut d_x = d_x_last;
    for i in (0..model.layers.len()).rev() {
        let (lw, ll) = &model.layers[i];
        let (d_in, g) = block_backward(
            gpu,
            &d_x,
            &acts.layer_inputs[i],
            &lw.as_block(),
            &ll.as_block(),
            &acts.layer_acts[i],
            &model.dims,
        )?;
        grads.push(g);
        d_x = d_in;
    }
    grads.reverse();
    gpu.free_tensor(d_logits)?;
    gpu.free_tensor(loss)?;
    gpu.free_tensor(d_xf)?;
    Ok((loss_sum, grads, d_final_norm))
}
