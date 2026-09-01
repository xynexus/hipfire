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
use hipfire_rdna::{DType, Gpu, GpuTensor, HipError, HipResult};
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
/// Where one projection's routed experts live.
///
/// Two residencies, and the choice is about MEMORY, not correctness — both serve
/// identical weights:
///
/// * [`Resident`](Self::Resident) uploads every expert at load, in ONE allocation
///   per projection. gfx1151 rounds each GTT allocation up to 2 MiB, so a
///   512-expert model allocated per expert would pay that rounding 512 times per
///   projection per layer.
/// * [`Paged`](Self::Paged) holds nothing and asks the weight pager per routed
///   expert, which evicts under a budget.
pub enum ExpertStack {
    Resident {
        buf: GpuTensor,
        dtype: DType,
        /// Per-expert output rows.
        rows: usize,
        /// Per-expert input dim (K).
        cols: usize,
        /// Elements between the start of one expert's region and the next.
        stride: usize,
    },
    Paged(Box<PagedExperts>),
}

impl ExpertStack {
    pub fn is_paged(&self) -> bool {
        matches!(self, ExpertStack::Paged(_))
    }

    pub fn dtype(&self) -> DType {
        match self {
            ExpertStack::Resident { dtype, .. } => *dtype,
            ExpertStack::Paged(p) => p.dtype,
        }
    }

    /// Elements this projection currently holds ITSELF.
    ///
    /// Zero when paged: the pager accounts for the whole model in one budget, so
    /// asking a single projection what it holds would double-count across layers
    /// and roles. `WeightPager::module_stats()` is the honest number there.
    pub fn resident_elems(&self) -> usize {
        match self {
            ExpertStack::Resident { buf, .. } => buf.numel(),
            ExpertStack::Paged(_) => 0,
        }
    }

    /// A `WeightTensor` view of expert `e`, paging it in first if needed.
    ///
    /// The 2-D shape has to be restored explicitly for F32. `sub_offset` returns
    /// a flat `[len]` view and the F32 GEMV reads `shape[1]` for its inner
    /// dimension, so leaving it 1-D indexes out of bounds. Quantised paths take
    /// `m`/`k` from the `WeightTensor` and never read the shape.
    pub fn expert(&self, gpu: &mut Gpu, e: usize) -> HipResult<WeightTensor> {
        let (buf, dtype, rows, cols) = match self {
            ExpertStack::Resident {
                buf,
                dtype,
                rows,
                cols,
                stride,
            } => (buf.sub_offset(e * *stride, *stride), *dtype, *rows, *cols),
            ExpertStack::Paged(p) => {
                use hipfire_runtime::weight_pager::{ExpertModuleKey, ExpertRole};
                let key = ExpertModuleKey {
                    layer: p.layer,
                    expert: e as u16,
                };
                let mut pager = p
                    .pager
                    .lock()
                    .map_err(|_| HipError::new(0, "qwen4_exp: expert pager mutex poisoned"))?;
                pager
                    .ensure_expert_module_resident(key, gpu)
                    .map_err(|err| {
                        HipError::new(
                            0,
                            &format!("qwen4_exp paged expert l{} e{e}: {err}", p.layer),
                        )
                    })?;
                let views = pager.resident_expert_views(key).ok_or_else(|| {
                    HipError::new(
                        0,
                        &format!(
                            "qwen4_exp paged expert l{} e{e}: resident but no view",
                            p.layer
                        ),
                    )
                })?;
                // One module holds BOTH projections back to back, so each role is
                // an offset into the same buffer — which is also why the GTT
                // rounding is paid once per expert rather than twice.
                let rel = match p.role {
                    ExpertRole::GateUp => views.gate_up_rel,
                    ExpertRole::Down => views.down_rel,
                };
                let len = views.buf.numel().saturating_sub(rel);
                (views.buf.sub_offset(rel, len), p.dtype, p.rows, p.cols)
            }
        };
        let mut buf = buf;
        if dtype == DType::F32 {
            buf.shape = vec![rows, cols];
        }
        Ok(WeightTensor {
            buf,
            gpu_dtype: dtype,
            m: rows,
            k: cols,
            row_stride: 0,
            paro: None,
            awq_scale: None,
        })
    }

    /// Both projections of expert `e`, resolved in ONE pager round-trip.
    ///
    /// `gate_up.expert()` + `down.expert()` each take the mutex, hash the key and
    /// call `ensure_expert_module_resident` — **for the same module**, since one
    /// module holds both projections. The second is a cache hit but still pays
    /// lock, hash and an LRU touch. At 48 layers x top-10 that is ~960 round
    /// trips per token where ~480 suffice.
    ///
    /// That overhead is the cost, not fetching: a 48 GiB budget with ZERO
    /// evictions measured no faster (0.27 s/tok) than an 8 GiB budget thrashing
    /// 15670 of them (0.25) — so the paged-vs-eager gap is fixed per-access work.
    ///
    /// Residency is still ensured immediately before use, so nothing may be
    /// evicted between ensuring and reading it. That is why this resolves ONE
    /// expert rather than pre-ensuring the whole top-k: with a tight budget,
    /// ensuring expert k can evict expert 1.
    pub fn expert_pair(
        gate_up: &ExpertStack,
        down: &ExpertStack,
        gpu: &mut Gpu,
        e: usize,
    ) -> HipResult<(WeightTensor, WeightTensor)> {
        // Only the paged/paged case can share a round-trip; anything else falls
        // back to the independent path, which is already lock-free.
        let (ExpertStack::Paged(gp), ExpertStack::Paged(dp)) = (gate_up, down) else {
            return Ok((gate_up.expert(gpu, e)?, down.expert(gpu, e)?));
        };
        debug_assert_eq!(
            gp.layer, dp.layer,
            "gate_up and down must page from the same layer"
        );
        use hipfire_runtime::weight_pager::ExpertModuleKey;
        let key = ExpertModuleKey {
            layer: gp.layer,
            expert: e as u16,
        };
        let mut pager = gp
            .pager
            .lock()
            .map_err(|_| HipError::new(0, "qwen4_exp: expert pager mutex poisoned"))?;
        pager
            .ensure_expert_module_resident(key, gpu)
            .map_err(|err| {
                HipError::new(
                    0,
                    &format!("qwen4_exp paged expert l{} e{e}: {err}", gp.layer),
                )
            })?;
        let views = pager.resident_expert_views(key).ok_or_else(|| {
            HipError::new(
                0,
                &format!(
                    "qwen4_exp paged expert l{} e{e}: resident but no view",
                    gp.layer
                ),
            )
        })?;
        let mk = |rel: usize, p: &PagedExperts| -> WeightTensor {
            let len = views.buf.numel().saturating_sub(rel);
            let mut buf = views.buf.sub_offset(rel, len);
            if p.dtype == DType::F32 {
                buf.shape = vec![p.rows, p.cols];
            }
            WeightTensor {
                buf,
                gpu_dtype: p.dtype,
                m: p.rows,
                k: p.cols,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            }
        };
        Ok((mk(views.gate_up_rel, gp), mk(views.down_rel, dp)))
    }

    pub fn free(self, gpu: &mut Gpu) {
        match self {
            ExpertStack::Resident { buf, .. } => {
                let _ = gpu.free_tensor(buf);
            }
            // The pager owns paged buffers and frees them on eviction/unload.
            ExpertStack::Paged(_) => {}
        }
    }
}

/// Routed experts served by [`hipfire_runtime::weight_pager::WeightPager`].
///
/// Holds no weights: it names the layer and asks the pager for each expert as it
/// is routed to. Residency, LRU eviction, GTT-aware accounting and the transport
/// are the pager's, not this crate's — an earlier version of this feature grew
/// its own cache and got only the easy half (a HashMap, no eviction).
///
/// The pager is shared across every layer, so its budget is one budget for the
/// whole model rather than a per-layer allowance that cannot see the others.
/// The shared expert pager, as the loader and the MoE forward pass hold it.
pub type ExpertPager = std::sync::Arc<std::sync::Mutex<hipfire_runtime::weight_pager::WeightPager>>;

pub struct PagedExperts {
    /// Shared across every layer and both roles, so the eviction budget is ONE
    /// budget for the model. `Arc<Mutex<_>>` rather than `Rc<RefCell<_>>` because
    /// `ServingBackend` is `Send`; serving is single-threaded, so the lock is
    /// uncontended.
    pub pager: ExpertPager,
    pub layer: u16,
    pub role: hipfire_runtime::weight_pager::ExpertRole,
    pub dtype: DType,
    pub rows: usize,
    pub cols: usize,
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
        let (gu, dn) = ExpertStack::expert_pair(&w.gate_up, &w.down, gpu, e)?;
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
