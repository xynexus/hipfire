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

/// The routed experts for ONE projection.
///
/// Two residencies, and which one a model gets is a memory decision:
///
/// * [`Resident`](Self::Resident) uploads every expert at load. One allocation,
///   not `n_experts` of them — gfx1151 rounds every GTT allocation up to 2 MiB, so
///   a 512-expert model allocated per expert pays that rounding 512 times per
///   projection per layer (a comparable model measured 105 GB against 66 GB).
/// * [`Lazy`](Self::Lazy) uploads NOTHING at load and fetches a GROUP of experts
///   the first time one of them is routed to.
///
/// The experts sit back to back with each region SELF-CONTAINED, because
/// `weight_gemv` finds a tensor's scales immediately after its own weights — which
/// is why experts cannot be a stride into one shared weight plane once quantised.
pub enum ExpertStack {
    Resident {
        buf: GpuTensor,
        dtype: DType,
        rows: usize,
        cols: usize,
        /// Elements between the start of one expert's region and the next.
        stride: usize,
    },
    /// Fetched on demand. See [`ExpertStack::expert`].
    Lazy(Box<LazyExperts>),
}

/// Per-expert weights fetched from the artifact the first time they are routed to.
///
/// ONE TOKEN TOUCHES AT MOST `experts_per_tok` OF `num_experts` — 10 of 512 on the
/// shipped model — so a forward pass needs ~2% of the expert bytes. Loading all of
/// them to serve a short interaction is the waste this removes.
///
/// Fetches expert by expert. Grouping was tried and MEASURED WORSE — see the
/// `group` field — because it multiplies coverage faster than it saves
/// allocations.
///
/// The trade is footprint against latency, and it is not subtle: on the shipped
/// model, 8 tokens leave 8.5 GiB of experts resident instead of 56.2 GiB, at
/// 0.37 s/tok instead of 0.08. Right when VRAM-bound, wrong when time-bound.
///
/// NO EVICTION. A group, once fetched, stays. That is the right trade for a
/// bounded interaction and the wrong one for a long context that eventually routes
/// to everything — at which point this degrades to the resident case plus fetch
/// overhead, never to something incorrect. Eviction is what
/// `hipfire_runtime::weight_pager` is for.
pub struct LazyExperts {
    /// The backend's OWN handle: the loader's is long gone by the first token.
    hfq: hipfire_runtime::hfq::HfqFile,
    /// `model.language_model.layers.<n>.mlp.experts.` — `<e>.<which>.weight`.
    prefix: String,
    which: String,
    pub dtype: DType,
    rows: usize,
    cols: usize,
    stride: usize,
    n_experts: usize,
    group: usize,
    /// Group index -> its uploaded buffer. `RefCell` because the forward holds
    /// `&MoeWeights` while needing to record a fetch; serving is single-threaded.
    resident: std::cell::RefCell<std::collections::HashMap<usize, GpuTensor>>,
}

impl LazyExperts {
    /// `stride` is one expert's converted byte length, taken from expert 0 so the
    /// layout is known before any fetch rather than discovered mid-forward.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        hfq: hipfire_runtime::hfq::HfqFile,
        prefix: String,
        which: String,
        dtype: DType,
        rows: usize,
        cols: usize,
        stride: usize,
        n_experts: usize,
    ) -> Self {
        Self {
            hfq,
            prefix,
            which,
            dtype,
            rows,
            cols,
            stride,
            n_experts,
            // ONE expert per fetch. The intuition says group them — gfx1151 rounds a
            // GTT allocation up to 2 MiB and one `down_proj` is ~870 KB, a 2.4x tax
            // — but MEASURED on the shipped model that reasoning is wrong for an
            // on-demand fetch, because read volume dominates the rounding:
            //
            //   group=4:  29.6 GiB resident after 8 tokens, 1.07 s/tok
            //   group=1:   8.5 GiB resident after 8 tokens, 0.37 s/tok
            //
            // Grouping multiplies COVERAGE: every routed expert drags in 3 more
            // that may never be used, and with top-10 routing across 48 layers the
            // union grows fast enough that the extra bytes swamp the saved
            // allocations. The rounding tax is real; it is just smaller than
            // fetching 4x the data.
            group: 1,
            resident: std::cell::RefCell::new(std::collections::HashMap::new()),
        }
    }

    /// How many expert groups have actually been fetched — the number this whole
    /// mechanism exists to keep small.
    pub fn resident_groups(&self) -> usize {
        self.resident.borrow().len()
    }
    pub fn total_groups(&self) -> usize {
        self.n_experts.div_ceil(self.group)
    }
}

impl ExpertStack {
    /// Elements currently uploaded. For a lazy stack this GROWS as groups are
    /// fetched, which is the number the whole mechanism exists to keep small.
    pub fn resident_elems(&self) -> usize {
        match self {
            ExpertStack::Resident { buf, .. } => buf.numel(),
            ExpertStack::Lazy(l) => l.resident.borrow().values().map(|b| b.numel()).sum(),
        }
    }

    pub fn dtype(&self) -> DType {
        match self {
            ExpertStack::Resident { dtype, .. } => *dtype,
            ExpertStack::Lazy(l) => l.dtype,
        }
    }

    /// A `WeightTensor` view of expert `e`, fetching its group if this is a lazy
    /// stack and the group is not yet resident.
    ///
    /// The 2-D shape has to be restored explicitly. `sub_offset` returns a flat
    /// `[len]` view, and the F32 GEMV reads `shape[1]` for its inner dimension —
    /// leaving it 1-D indexes out of bounds. Quantised paths take `m`/`k` from the
    /// `WeightTensor` instead and do not read the shape.
    pub fn expert(&self, gpu: &mut Gpu, e: usize) -> HipResult<WeightTensor> {
        let (buf, dtype, rows, cols, stride, off) = match self {
            ExpertStack::Resident {
                buf,
                dtype,
                rows,
                cols,
                stride,
            } => (
                buf.sub_offset(e * *stride, *stride),
                *dtype,
                *rows,
                *cols,
                *stride,
                0usize,
            ),
            ExpertStack::Lazy(l) => {
                let g = e / l.group;
                if !l.resident.borrow().contains_key(&g) {
                    let lo = g * l.group;
                    let hi = (lo + l.group).min(l.n_experts);
                    let mut bytes = Vec::with_capacity((hi - lo) * l.stride);
                    for ex in lo..hi {
                        let name = format!("{}{}.{}.weight", l.prefix, ex, l.which);
                        let (qt, raw) = l.hfq.tensor_data_logical(&name).map_err(|err| {
                            hipfire_rdna::HipError::new(
                                0,
                                &format!("qwen4_exp lazy expert `{name}`: {err:?}"),
                            )
                        })?;
                        let (conv, d) = hipfire_runtime::oq8_arch::oq8_arch_load_allow_compact(
                            qt, &raw, l.rows, l.cols,
                        )
                        .or_else(|| {
                            if hipfire_runtime::oq4_arch::oq4_arch_unsupported_reason(
                                l.rows, l.cols,
                            )
                            .is_some()
                            {
                                None
                            } else {
                                hipfire_runtime::oq4_arch::oq4_arch_load(qt, &raw, l.rows, l.cols)
                                    .map(|(b, d)| (b.into_owned(), d))
                            }
                        })
                        .ok_or_else(|| {
                            hipfire_rdna::HipError::new(
                                0,
                                &format!(
                                    "qwen4_exp lazy expert `{name}`: quant type {qt} has no \
                                     device form"
                                ),
                            )
                        })?;
                        // A group whose experts disagree would misalign every one
                        // after the first.
                        if d != l.dtype || conv.len() != l.stride {
                            return Err(hipfire_rdna::HipError::new(
                                0,
                                &format!(
                                    "qwen4_exp lazy expert `{name}`: {d:?}/{} does not match the \
                                     projection's {:?}/{}",
                                    conv.len(),
                                    l.dtype,
                                    l.stride
                                ),
                            ));
                        }
                        bytes.extend_from_slice(&conv);
                    }
                    let buf = gpu.upload_raw(&bytes, &[bytes.len()])?;
                    l.resident.borrow_mut().insert(g, buf);
                }
                let borrowed = l.resident.borrow();
                let base = borrowed.get(&g).expect("just inserted");
                let within = (e % l.group) * l.stride;
                (
                    base.sub_offset(within, l.stride),
                    l.dtype,
                    l.rows,
                    l.cols,
                    l.stride,
                    within,
                )
            }
        };
        let _ = (stride, off);
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

    pub fn free(self, gpu: &mut Gpu) {
        match self {
            ExpertStack::Resident { buf, .. } => {
                let _ = gpu.free_tensor(buf);
            }
            ExpertStack::Lazy(l) => {
                for (_, b) in l.resident.into_inner() {
                    let _ = gpu.free_tensor(b);
                }
            }
        }
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
        let gu = w.gate_up.expert(gpu, e)?;
        let dn = w.down.expert(gpu, e)?;
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
