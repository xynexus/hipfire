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

pub struct MoeWeights {
    /// `[n_experts, hidden]`
    pub router: GpuTensor,
    /// `[n_experts, 2 * mi, hidden]` stacked, gate first.
    pub gate_up: GpuTensor,
    /// `[n_experts, hidden, mi]` stacked.
    pub down: GpuTensor,
    /// `[shared_mi, hidden]` each.
    pub shared_gate: GpuTensor,
    pub shared_up: GpuTensor,
    /// `[hidden, shared_mi]`
    pub shared_down: GpuTensor,
    /// `[1, hidden]`
    pub shared_expert_gate: GpuTensor,
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

/// A 2-D view into a stacked expert tensor.
///
/// `sub_offset` returns a flat `[len]` view, but `gemv_f32` reads `shape[1]` for
/// its inner dimension — a 1-D view makes it index out of bounds. The shape has to
/// be restored explicitly.
fn view2d(t: &GpuTensor, offset: usize, rows: usize, cols: usize) -> GpuTensor {
    let mut v = t.sub_offset(offset, rows * cols);
    v.shape = vec![rows, cols];
    v
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

    gpu.gemv_f32(&w.router, x, &s.logits)?;
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
    let (gu_sz, dn_sz) = (2 * mi * hidden, hidden * mi);
    for (slot, &e) in idx.iter().enumerate() {
        let e = e.max(0) as usize;
        let gu = view2d(&w.gate_up, e * gu_sz, 2 * mi, hidden);
        let dn = view2d(&w.down, e * dn_sz, hidden, mi);
        gpu.gemv_f32(&gu, x, &s.gu)?;
        // Contiguous halves of the projection output, gate FIRST.
        let gate = s.gu.sub_offset(0, mi);
        let up = s.gu.sub_offset(mi, mi);
        gpu.silu_mul_f32(&gate, &up, &s.inter)?;
        gpu.gemv_f32(&dn, &s.inter, &s.expert_out)?;
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
    gpu.gemv_f32(&w.shared_gate, x, &s.sg)?;
    gpu.gemv_f32(&w.shared_up, x, &s.su)?;
    gpu.silu_mul_f32(&s.sg, &s.su, &s.sinter)?;
    gpu.gemv_f32(&w.shared_down, &s.sinter, &s.sout)?;
    gpu.gemv_f32(&w.shared_expert_gate, x, &s.sgate)?;
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
        for t in [
            router,
            gate_up,
            down,
            shared_gate,
            shared_up,
            shared_down,
            shared_expert_gate,
        ] {
            let _ = gpu.free_tensor(t);
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
