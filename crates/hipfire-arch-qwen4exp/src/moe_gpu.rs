// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! The routed MoE block on the GPU. Mirrors [`crate::moe::MoeLayer`].
//!
//! Routing is `moe_softmax_topk_renorm`, which takes `k_top` at RUNTIME. It used
//! to be a compile-time `#define K_TOP 8`; this family routes top-10 of 512, so a
//! fixed 8 silently dropped two experts per token.
//!
//! ponytail: experts are dispatched one GEMV at a time over `sub_offset` views of
//! the stacked weights, which costs a device sync per token to read the selected
//! indices. That is the correctness path. The fused/indexed expert GEMVs already
//! in this tree are the performance answer, and they want the `mi % 256` and
//! `k_top` admission work (M17/M18) before they accept this geometry.

use crate::config::Qwen4ExpConfig;
use hipfire_rdna::{DType, Gpu, GpuTensor, HipResult};
use hipfire_runtime::weights::{weight_gemv, WeightTensor};

/// The routed experts for ONE projection, in a single allocation.
///
/// One buffer, not `n_experts` of them, and that is a hardware decision rather
/// than tidiness: gfx1151 rounds every GTT allocation up to 2 MiB, so a 512-expert
/// model allocated per expert pays that rounding 512 times per projection per
/// layer. The measured cost of getting this wrong on a comparable model was
/// 105 GB against 66 GB for the same weights.
///
/// The experts sit back to back, each region SELF-CONTAINED so a per-expert view
/// is one offset. For a quantised Opus dtype that region is the combined
/// `[int8 weights | f32 scales]` form the kernels expect — which is why the
/// experts cannot simply be a stride into one flat weight plane: `weight_gemv`
/// finds a tensor's scales immediately after its own weights.
pub struct ExpertStack {
    pub buf: GpuTensor,
    pub dtype: DType,
    /// Per-expert output rows.
    pub rows: usize,
    /// Per-expert input dim (K).
    pub cols: usize,
    /// Elements between the start of one expert's region and the next.
    pub stride: usize,
}

impl ExpertStack {
    /// A `WeightTensor` view of expert `e`. Cheap — a sub-offset, no copy.
    ///
    /// The 2-D shape has to be restored explicitly. `sub_offset` returns a flat
    /// `[len]` view, and the F32 GEMV reads `shape[1]` for its inner dimension —
    /// leaving it 1-D indexes out of bounds. Quantised paths take `m`/`k` from the
    /// `WeightTensor` instead and do not care, so this only has to be right where
    /// the shape is actually read.
    pub fn expert(&self, e: usize) -> WeightTensor {
        let mut buf = self.buf.sub_offset(e * self.stride, self.stride);
        if self.dtype == DType::F32 {
            buf.shape = vec![self.rows, self.cols];
        }
        WeightTensor {
            buf,
            gpu_dtype: self.dtype,
            m: self.rows,
            k: self.cols,
            row_stride: 0,
            paro: None,
            awq_scale: None,
        }
    }

    pub fn free(self, gpu: &mut Gpu) {
        let Self {
            buf,
            dtype: _,
            rows: _,
            cols: _,
            stride: _,
        } = self;
        let _ = gpu.free_tensor(buf);
    }
}

pub struct MoeWeights {
    /// `[n_experts, hidden]`
    pub router: WeightTensor,
    /// `[n_experts, 2 * mi, hidden]`, gate first.
    pub gate_up: ExpertStack,
    /// `[n_experts, hidden, mi]`.
    pub down: ExpertStack,
    /// `[shared_mi, hidden]` each.
    pub shared_gate: WeightTensor,
    pub shared_up: WeightTensor,
    /// `[hidden, shared_mi]`
    pub shared_down: WeightTensor,
    /// `[1, hidden]`
    pub shared_expert_gate: WeightTensor,
}

pub struct MoeScratch {
    logits: GpuTensor,
    topk_idx: GpuTensor,
    topk_w: GpuTensor,
    gu: GpuTensor,
    inter: GpuTensor,
    expert_out: GpuTensor,
    sg: GpuTensor,
    su: GpuTensor,
    sinter: GpuTensor,
    sout: GpuTensor,
    sgate: GpuTensor,
}

impl MoeScratch {
    pub fn new(gpu: &mut Gpu, cfg: &Qwen4ExpConfig) -> HipResult<Self> {
        let m = &cfg.moe;
        let z = |g: &mut Gpu, n: usize| g.zeros(&[n], DType::F32);
        Ok(Self {
            logits: z(gpu, m.num_experts)?,
            // No I32 dtype in this tree: integer buffers ride f32 storage and are
            // read back through `to_bits`, the same convention the QSA slot list uses.
            topk_idx: z(gpu, m.experts_per_tok)?,
            topk_w: z(gpu, m.experts_per_tok)?,
            gu: z(gpu, 2 * m.intermediate)?,
            inter: z(gpu, m.intermediate)?,
            expert_out: z(gpu, cfg.hidden)?,
            sg: z(gpu, m.shared_intermediate)?,
            su: z(gpu, m.shared_intermediate)?,
            sinter: z(gpu, m.shared_intermediate)?,
            sout: z(gpu, cfg.hidden)?,
            sgate: z(gpu, 1)?,
        })
    }
}

impl MoeScratch {
    /// The routed expert indices from the last call, for tests that need to check
    /// the SET and not only the output.
    pub fn topk_idx_view(&self) -> GpuTensor {
        self.topk_idx.sub_offset(0, self.topk_idx.numel())
    }
}

/// One token through the block. `x` and `out` are `[hidden]`.
pub fn moe_forward(
    gpu: &mut Gpu,
    cfg: &Qwen4ExpConfig,
    w: &MoeWeights,
    s: &mut MoeScratch,
    x: &GpuTensor,
    out: &GpuTensor,
) -> HipResult<()> {
    let m = &cfg.moe;
    let (hidden, mi) = (cfg.hidden, m.intermediate);

    weight_gemv(gpu, &w.router, x, &s.logits)?;
    gpu.moe_softmax_topk_renorm(
        &s.logits,
        &s.topk_idx,
        &s.topk_w,
        m.num_experts,
        m.norm_topk_prob,
        m.experts_per_tok,
    )?;
    let idx: Vec<i32> = gpu
        .download_f32(&s.topk_idx)?
        .iter()
        .map(|v| v.to_bits() as i32)
        .collect();
    for (slot, &e) in idx.iter().enumerate() {
        let e = e.max(0) as usize;
        // `weight_gemv` dispatches on the stack's dtype, so a quantised expert
        // runs its own kernel rather than being dequantised into scratch.
        let gu = w.gate_up.expert(e);
        let dn = w.down.expert(e);
        weight_gemv(gpu, &gu, x, &s.gu)?;
        // Contiguous halves of the projection output, gate FIRST.
        let gate = s.gu.sub_offset(0, mi);
        let up = s.gu.sub_offset(mi, mi);
        gpu.silu_mul_f32(&gate, &up, &s.inter)?;
        weight_gemv(gpu, &dn, &s.inter, &s.expert_out)?;
        // The routing weight scales the expert's OUTPUT, after `down`.
        // Slot 0 overwrites; the rest accumulate. Saves a zero-fill pass.
        gpu.moe_accum_scaled(
            out,
            &s.expert_out,
            &s.topk_w,
            slot as i32,
            hidden as i32,
            slot > 0,
        )?;
    }

    // The shared expert is always on, gated by a scalar sigmoid.
    weight_gemv(gpu, &w.shared_gate, x, &s.sg)?;
    weight_gemv(gpu, &w.shared_up, x, &s.su)?;
    gpu.silu_mul_f32(&s.sg, &s.su, &s.sinter)?;
    weight_gemv(gpu, &w.shared_down, &s.sinter, &s.sout)?;
    weight_gemv(gpu, &w.shared_expert_gate, x, &s.sgate)?;
    gpu.moe_shared_gate(&s.sout, &s.sgate, &s.sout, hidden as i32)?;
    gpu.add_inplace_f32(out, &s.sout)
}

// ── GPU teardown ────────────────────────────────────────────────────────────
//
// Every `free` below DESTRUCTURES its struct exhaustively rather than naming
// fields to free. That is deliberate: a field added later fails to compile until
// someone decides what happens to it, where a `self.a; self.b;` list would just
// silently leak the new tensor. `unload` on a 360 GB model has no test that would
// catch that.

impl MoeWeights {
    pub fn free(self, gpu: &mut Gpu) {
        let Self {
            router,
            gate_up,
            down,
            shared_gate,
            shared_up,
            shared_down,
            shared_expert_gate,
        } = self;
        gate_up.free(gpu);
        down.free(gpu);
        for t in [
            router,
            shared_gate,
            shared_up,
            shared_down,
            shared_expert_gate,
        ] {
            let _ = gpu.free_tensor(t.buf);
        }
    }
}

impl MoeScratch {
    pub fn free(self, gpu: &mut Gpu) {
        let Self {
            logits,
            topk_idx,
            topk_w,
            gu,
            inter,
            expert_out,
            sg,
            su,
            sinter,
            sout,
            sgate,
        } = self;
        for t in [
            logits, topk_idx, topk_w, gu, inter, expert_out, sg, su, sinter, sout, sgate,
        ] {
            let _ = gpu.free_tensor(t);
        }
    }
}
