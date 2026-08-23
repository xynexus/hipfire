// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Qwen3.5 in-memory weight layout: per-layer weight structs, the MoE FFN
//! weight bundle, the `Qwen35Weights` model container, and GPU-storage /
//! free helpers.

use super::*;

/// Weights for a DeltaNet (linear attention) layer.
pub struct DeltaNetLayerWeights {
    pub attn_norm: GpuTensor,   // input_layernorm [dim]
    pub wqkv: WeightTensor,     // in_proj_qkv [6144, dim] → Q+K+V concat
    pub wz: WeightTensor,       // in_proj_z [2048, dim] → gate Z
    pub w_alpha: WeightTensor,  // in_proj_a [n_heads, dim] → decay
    pub w_beta: WeightTensor,   // in_proj_b [n_heads, dim] → update
    pub a_log: GpuTensor,       // A_log [n_heads] — learnable log-decay
    pub dt_bias: GpuTensor,     // dt_bias [n_heads]
    pub conv_weight: GpuTensor, // conv1d.weight [conv_channels, 1, 4] → F32
    pub norm_weight: GpuTensor, // norm.weight [head_dim] — gated output norm
    pub wo: WeightTensor,       // out_proj [dim, d_inner]
    pub ffn_norm: GpuTensor,    // post_attention_layernorm [dim]
    pub w_gate: WeightTensor,   // mlp.gate_proj
    pub w_up: WeightTensor,     // mlp.up_proj
    pub w_down: WeightTensor,   // mlp.down_proj
    pub bf16_down_shadow: Option<Bf16DownShadow>,
}

/// Weights for a full attention (gated) layer — similar to Qwen3 but with q+gate split.
pub struct FullAttnLayerWeights {
    pub attn_norm: GpuTensor,
    pub wq: WeightTensor,  // q_proj [4096, dim] — 2x wide (query + gate)
    pub wk: WeightTensor,  // k_proj
    pub wv: WeightTensor,  // v_proj
    pub wo: WeightTensor,  // o_proj
    pub q_norm: GpuTensor, // q_norm [head_dim]
    pub k_norm: GpuTensor, // k_norm [head_dim]
    pub ffn_norm: GpuTensor,
    pub w_gate: WeightTensor,
    pub w_up: WeightTensor,
    pub w_down: WeightTensor,
    pub bf16_down_shadow: Option<Bf16DownShadow>,
}

// ─── MoE FFN weights (Qwen3.5-MoE / A3B) ────────────────────────────────
//
// Replaces the dense (w_gate, w_up, w_down) triple with N+1 expert FFNs
// gated by a router, plus a shared always-on expert.
//
// A3B specifics:
//   num_experts = 256, top_k = 8, moe_intermediate = 512, hidden = 2048
//   shared_expert_intermediate = 512 (same as routed)
//
// Per-layer storage:
//   router:               [num_experts, hidden]  MQ4G256 / Q8
//   shared_expert_gate:   [1, hidden]            MQ4G256 / Q8 — projects to scalar
//   experts[X].gate_up:   [2*moe_intermediate, hidden]  MQ4G256
//   experts[X].down:      [hidden, moe_intermediate]    MQ4G256
//   shared_expert.gate:   [shared_expert_intermediate, hidden]   MQ4G256
//   shared_expert.up:     [shared_expert_intermediate, hidden]   MQ4G256
//   shared_expert.down:   [hidden, shared_expert_intermediate]   MQ4G256
//
// The quantizer (hipfire-quantize) splits the safetensors 3D
// `mlp.experts.gate_up_proj` / `down_proj` tensors per-expert into
// `mlp.experts.{X}.gate_up_proj.weight` / `down_proj.weight` so the loader
// can fish them out by index. The shared expert is stored with separate
// gate_proj + up_proj + down_proj (it is not fused in safetensors either).

pub struct ExpertWeights {
    pub gate_up: WeightTensor, // [2 * moe_intermediate, hidden] — fused (gate || up)
    pub down: WeightTensor,    // [hidden, moe_intermediate]
}

/// Shared expert storage — unlike routed experts, gate_proj and up_proj are
/// NOT fused in the safetensors, so we keep them separate here too. The
/// forward path does two GEMVs + silu_mul + down GEMV.
pub struct SharedExpertWeights {
    pub gate: WeightTensor, // [shared_expert_intermediate, hidden]
    pub up: WeightTensor,   // [shared_expert_intermediate, hidden]
    pub down: WeightTensor, // [hidden, shared_expert_intermediate]
}

pub struct MoeFfnWeights {
    pub router: WeightTensor, // [num_experts, hidden]
    /// Routed expert weights. Populated when this layer is fully resident
    /// (`paged_experts == false`); **empty `Vec`** when `paged_experts == true`
    /// (the [`hipfire_runtime::weight_pager::WeightPager`] owns the buffers, and the
    /// indexed kernels read pointers from `expert_*_ptrs`).
    ///
    /// **The paged half of that is NOT wired (verified 2026-08-10).** This comment
    /// used to say the pager "patches per-token via `patch_expert_ptr_table`". It
    /// does not: `patch_expert_ptr_table` has zero call sites workspace-wide, so
    /// under paged residency `expert_*_ptrs` stays all-zero and the indexed
    /// kernels dereference null — which wedges the GPU (see BUGS.md, MES hang on
    /// `gemv_oq4g256_moe_gate_up_k8_indexed_batched`). Treat paged + indexed MoE
    /// as unimplemented, not as working machinery.
    pub experts: Vec<ExpertWeights>, // num_experts (= 256 for A3B); empty in paged mode
    pub shared_expert: SharedExpertWeights,
    pub shared_expert_gate: WeightTensor, // [1, hidden] — row-vector projecting to scalar
    /// Device-side array of `unsigned long long` pointers, one per
    /// expert's `gate_up.buf`. Indexed at runtime by the GPU top-K
    /// kernel's output so the indexed MoE GEMV can stay capture-safe.
    pub expert_gate_up_ptrs: GpuTensor, // [num_experts * 2] f32 slots = num_experts × u64
    pub expert_down_ptrs: GpuTensor,      // [num_experts * 2] f32 slots = num_experts × u64

    /// Device-side per-expert compact block stride (`130 + 2*N_out`), or **0**
    /// where that expert is `Oq8G256`. `[num_experts]` i32.
    ///
    /// This is what lets ONE indexed launch serve a layer that mixes compact and
    /// Oq8 routed experts, which mixed-precision promotion produces routinely --
    /// the 122B mixes them across 37 of its 48 layers. Without it the layout is
    /// a launch-wide constant, and the only way to dispatch a mixed layer is to
    /// expand its compact experts at load until the layer is uniform, at 1.80x
    /// the bytes.
    ///
    /// 0 as the Oq8 sentinel rather than 260: a compact stride is `130 + 2*N_out`
    /// and 260 satisfies that at N_out=65, so a stride cannot identify its own
    /// layout.
    /// `None` on paths that never dispatch the compact GEMVs (paged residency,
    /// calibration) -- an absent table is the honest encoding there, and the
    /// dispatch asserts on it rather than reading a table of zeros that would
    /// silently mean "every expert is Oq8".
    pub expert_gate_up_strides: Option<GpuTensor>,
    pub expert_down_strides: Option<GpuTensor>,

    /// Device-side per-expert AWQ scale pointers, same shape and construction
    /// as `expert_gate_up_ptrs` (a 0 entry means that expert has no sidecar).
    ///
    /// The indexed MoE path rotates x per (token, krank) via
    /// `rotate_x_mq_awq_indexed_batched`, which needs EACH expert's scale:
    /// routed experts do NOT share one (measured on a 35B-A3B oq4.25++ —
    /// expert0/expert1/expert7/shared all differ), because routing gives each a
    /// different token subset and therefore a different imatrix.
    ///
    /// `None` when no routed expert carries a sidecar; the plain rotation is
    /// then already correct and byte-identical.
    pub expert_gate_up_awq_ptrs: Option<GpuTensor>,
    pub expert_down_awq_ptrs: Option<GpuTensor>,

    /// Routed-expert AWQ scales for **paged** layers, indexed by expert id.
    ///
    /// Resident layers keep theirs inside `experts[i].{gate_up,down}.awq_scale`,
    /// which is what holds those allocations alive. A paged layer has no
    /// `ExpertWeights` at all, so the scales need an owner here or they drop at
    /// the end of the loader and `expert_*_awq_ptrs` is left holding dangling
    /// device pointers — which the indexed kernels dereference with no
    /// validation. Same reason `paro_shared` exists.
    ///
    /// Indexed by expert rather than compacted, because the paged decode loop is
    /// bucketed BY EXPERT and looks its scale up directly
    /// (`run_paged_mixed_routed_decode`); the `expert_*_awq_ptrs` tables are
    /// built from these for the indexed kernels. `None` at a slot means that
    /// expert has no sidecar and takes the plain rotation.
    ///
    /// Empty for resident layers, and for artifacts with no routed sidecars.
    ///
    /// The scales are NOT paged. They are tiny next to the weights — f32 on
    /// device, `dim + mi` per expert, ~201 MB for the 122B's 12,288 experts
    /// against a 9.3 GB paged footprint — and paging them would mean a residency
    /// transition on the rotation input as well as the weights.
    pub expert_gate_up_awq: Vec<Option<GpuTensor>>,
    pub expert_down_awq: Vec<Option<GpuTensor>>,

    /// Layer index. Stable identity used to key
    /// [`hipfire_runtime::weight_pager::WeightId::Expert`] entries.
    pub layer_idx: u16,

    /// Per-expert tensor shapes. `None` in non-paged mode (shapes are read
    /// from `experts[i].gate_up.{m, k}` etc.); `Some` in paged mode where
    /// `experts` is empty but kernels still need m/k for kernel-arg setup.
    /// Qwen3.5-MoE-A3B has uniform per-expert shape so one descriptor per
    /// layer suffices for v0.1.
    pub expert_shape: Option<hipfire_runtime::weight_pager::ExpertShape>,
    pub expert_gate_up_dtype: Option<DType>,
    pub expert_down_dtype: Option<DType>,
    /// Per-expert dtype metadata used when experts are paged and therefore do
    /// not have resident [`ExpertWeights`] records to inspect.  These vectors
    /// are also populated for resident layers so admission/provenance checks
    /// can use one representation.  Mixed-precision induction may preserve an
    /// undercovered expert at BF16/F16 while quantizing its siblings.
    pub expert_gate_up_dtypes: Vec<DType>,
    pub expert_down_dtypes: Vec<DType>,

    /// ParoQuant only: shared per-layer rotation sidecars for the routed
    /// experts. shisa-ai's PARO checkpoint quantizes all 256 experts with
    /// one rotation tuple per projection-group (gate||up vs down), so we
    /// upload the sidecars ONCE per layer and broadcast a non-owning
    /// `ParoRotation` (built via `DeviceBuffer::from_raw`) into every
    /// `ExpertWeights.gate_up.paro` / `ExpertWeights.down.paro`. The
    /// owning storage lives here so the aliases stay valid for the
    /// lifetime of the layer. `None` for HFQ MoE (per-tensor PARO sidecars
    /// or no PARO at all).
    pub paro_shared: Option<MoeParoSidecars>,
    /// Owning storage for source safetensors whose routed experts are stacked
    /// as `[E, M, K]`. `experts` then contains non-owning slice aliases used by
    /// the existing executor; the two backing allocations are freed once here.
    pub raw_expert_storage: Option<RawExpertStorage>,
}

pub struct RawExpertStorage {
    pub gate_up: GpuTensor,
    pub down: GpuTensor,
}

/// Owning storage for the per-layer shared ParoQuant rotation sidecars.
/// One tuple per projection-group:
///   - `gate_up_*`: applied to the post-RMSNorm hidden activation (K = hidden_dim).
///     Shared by all 256 experts' gate AND up projections, and by the fused
///     gate_up `WeightTensor`'s `paro` alias.
///   - `down_*`: applied to the post-SiLU intermediate activation (K = mi).
///     Shared by all 256 experts' down projection.
pub struct MoeParoSidecars {
    pub gate_up_pairs: GpuTensor,
    pub gate_up_theta: GpuTensor,
    pub gate_up_channel_scales: GpuTensor,
    pub down_pairs: GpuTensor,
    pub down_theta: GpuTensor,
    pub down_channel_scales: GpuTensor,
    pub krot: u32,
    pub group_size: u32,
}

pub struct DeltaNetMoeLayerWeights {
    pub attn_norm: GpuTensor,
    pub wqkv: WeightTensor,
    pub wz: WeightTensor,
    pub w_alpha: WeightTensor,
    pub w_beta: WeightTensor,
    pub a_log: GpuTensor,
    pub dt_bias: GpuTensor,
    pub conv_weight: GpuTensor,
    pub norm_weight: GpuTensor,
    pub wo: WeightTensor,
    pub ffn_norm: GpuTensor,
    pub ffn: MoeFfnWeights,
}

pub struct FullAttnMoeLayerWeights {
    pub attn_norm: GpuTensor,
    pub wq: WeightTensor,
    pub wk: WeightTensor,
    pub wv: WeightTensor,
    pub wo: WeightTensor,
    pub q_norm: GpuTensor,
    pub k_norm: GpuTensor,
    pub ffn_norm: GpuTensor,
    pub ffn: MoeFfnWeights,
}

pub enum LayerWeights {
    DeltaNet(DeltaNetLayerWeights),
    FullAttn(FullAttnLayerWeights),
    // A3B / qwen3_5_moe: same attention as above, MoE FFN instead of dense.
    // Loader + forward path TODO — adding the variants now so the enum is
    // forward-compatible and downstream code that pattern-matches gets a
    // compile-time hint to handle the new case.
    DeltaNetMoe(DeltaNetMoeLayerWeights),
    FullAttnMoe(FullAttnMoeLayerWeights),
}

/// RoughQuant real-format protected-channel correction (one per quantized
/// residual projection that carries a `.rqcorr` sidecar). Applied at GEMV time as
/// `y += R_S · x_S`, restoring the protected channels to bf16 precision over the
/// mq4 bulk. See `docs/roughquant/phase3-real-format-scope.md`.
pub enum RqCorr {
    /// Residual reader (protected input COLUMNS): gather `xs = x_normed[S]` (padded
    /// to power-of-2 `np`), `out += corr[m×np] · xs`.
    Reader {
        corr: GpuTensor, // [m × np] f32
        idx: GpuTensor,  // [n_idx] i32 — residual channel ids (gather source)
        m: usize,
        n_idx: usize,
        np: usize,
    },
    /// Residual writer (protected output ROWS): `c = corr[|S|×k] · input`, then
    /// scatter-add `out[S[j]] += c[j]`.
    Writer {
        corr: GpuTensor, // [n_s × k] f32
        idx: GpuTensor,  // [n_s] i32 — residual channel ids (scatter target)
        n_s: usize,
        k: usize,
    },
}

/// Which projection within a layer a correction targets (keys the side-map).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum RqProj {
    Wqkv,
    Wz,
    Walpha,
    Wbeta,
    Wq,
    Wk,
    Wv,
    Wgate,
    Wup,
    Wo,
    Wdown,
}

pub struct Qwen35Weights {
    pub token_embd: GpuTensor,
    pub embd_format: EmbeddingFormat,
    pub output_norm: GpuTensor,
    pub output: WeightTensor,
    pub layers: Vec<LayerWeights>,
    pub slab_storage: Option<ModelGpuStorage>,
    /// RoughQuant real-format corrections, keyed by `(layer_idx, projection)`.
    /// Empty for all non-roughquant models (backward-compatible: the forward
    /// applies nothing). Populated by `load_rq_corrections` from the
    /// `metadata["roughquant_sidecar"]` index + `<name>.rqcorr` tensors.
    pub rq_corrections: std::collections::HashMap<(u32, RqProj), RqCorr>,

    /// Weight pager (MAD-93 v0.1). `Some` only when the model was loaded
    /// with `Qwen35Config::paged_experts == true`. `None` means the model is
    /// fully resident — no behavior change vs main.
    ///
    /// **Reachability, corrected 2026-08-10.** This used to claim "the forward
    /// path uses interior mutability (`borrow_mut`) at the MoE dispatch site to
    /// call `ensure_resident` / `patch_expert_ptr_table`". Only half of that is
    /// true, and only on paths that are not the default:
    /// - `ensure_paged_experts_resident` is called from `moe_decode.rs` (the qwen35
    ///   HAND decode path) and `prefill_chunk.rs`. The **lowered super-op pipeline
    ///   is default-ON and calls neither**, and its `MoeParams` carries no pager at
    ///   all, so it cannot.
    /// - `patch_expert_ptr_table` is called from **nowhere**.
    ///
    /// Consequence: on the default path the expert pointer table is never written.
    /// Do not assume this field being `Some` means paging is active at dispatch.
    pub pager: Option<std::cell::RefCell<hipfire_runtime::weight_pager::WeightPager>>,
}

impl Qwen35Weights {
    /// Return all GPU buffers to the pool (drained on unload). Consumes self.
    pub fn free_gpu(self, gpu: &mut Gpu) {
        let Qwen35Weights {
            token_embd,
            output_norm,
            output,
            layers,
            pager,
            slab_storage,
            ..
        } = self;
        let slabs = slab_storage.as_ref();
        free_tensor_maybe_slab(gpu, slabs, token_embd);
        free_tensor_maybe_slab(gpu, slabs, output_norm);
        free_weight_tensor_maybe_slab(gpu, slabs, output);
        for layer in layers {
            match layer {
                LayerWeights::DeltaNet(l) => {
                    free_tensor_maybe_slab(gpu, slabs, l.attn_norm);
                    free_weight_tensor_maybe_slab(gpu, slabs, l.wqkv);
                    free_weight_tensor_maybe_slab(gpu, slabs, l.wz);
                    free_weight_tensor_maybe_slab(gpu, slabs, l.w_alpha);
                    free_weight_tensor_maybe_slab(gpu, slabs, l.w_beta);
                    free_tensor_maybe_slab(gpu, slabs, l.a_log);
                    free_tensor_maybe_slab(gpu, slabs, l.dt_bias);
                    free_tensor_maybe_slab(gpu, slabs, l.conv_weight);
                    free_tensor_maybe_slab(gpu, slabs, l.norm_weight);
                    free_weight_tensor_maybe_slab(gpu, slabs, l.wo);
                    free_tensor_maybe_slab(gpu, slabs, l.ffn_norm);
                    free_weight_tensor_maybe_slab(gpu, slabs, l.w_gate);
                    free_weight_tensor_maybe_slab(gpu, slabs, l.w_up);
                    free_weight_tensor_maybe_slab(gpu, slabs, l.w_down);
                }
                LayerWeights::FullAttn(l) => {
                    free_tensor_maybe_slab(gpu, slabs, l.attn_norm);
                    free_weight_tensor_maybe_slab(gpu, slabs, l.wq);
                    free_weight_tensor_maybe_slab(gpu, slabs, l.wk);
                    free_weight_tensor_maybe_slab(gpu, slabs, l.wv);
                    free_weight_tensor_maybe_slab(gpu, slabs, l.wo);
                    free_tensor_maybe_slab(gpu, slabs, l.q_norm);
                    free_tensor_maybe_slab(gpu, slabs, l.k_norm);
                    free_tensor_maybe_slab(gpu, slabs, l.ffn_norm);
                    free_weight_tensor_maybe_slab(gpu, slabs, l.w_gate);
                    free_weight_tensor_maybe_slab(gpu, slabs, l.w_up);
                    free_weight_tensor_maybe_slab(gpu, slabs, l.w_down);
                }
                LayerWeights::DeltaNetMoe(l) => {
                    free_tensor_maybe_slab(gpu, slabs, l.attn_norm);
                    free_weight_tensor_maybe_slab(gpu, slabs, l.wqkv);
                    free_weight_tensor_maybe_slab(gpu, slabs, l.wz);
                    free_weight_tensor_maybe_slab(gpu, slabs, l.w_alpha);
                    free_weight_tensor_maybe_slab(gpu, slabs, l.w_beta);
                    free_tensor_maybe_slab(gpu, slabs, l.a_log);
                    free_tensor_maybe_slab(gpu, slabs, l.dt_bias);
                    free_tensor_maybe_slab(gpu, slabs, l.conv_weight);
                    free_tensor_maybe_slab(gpu, slabs, l.norm_weight);
                    free_weight_tensor_maybe_slab(gpu, slabs, l.wo);
                    free_tensor_maybe_slab(gpu, slabs, l.ffn_norm);
                    free_moe_ffn_maybe_slab(gpu, slabs, l.ffn);
                }
                LayerWeights::FullAttnMoe(l) => {
                    free_tensor_maybe_slab(gpu, slabs, l.attn_norm);
                    free_weight_tensor_maybe_slab(gpu, slabs, l.wq);
                    free_weight_tensor_maybe_slab(gpu, slabs, l.wk);
                    free_weight_tensor_maybe_slab(gpu, slabs, l.wv);
                    free_weight_tensor_maybe_slab(gpu, slabs, l.wo);
                    free_tensor_maybe_slab(gpu, slabs, l.q_norm);
                    free_tensor_maybe_slab(gpu, slabs, l.k_norm);
                    free_tensor_maybe_slab(gpu, slabs, l.ffn_norm);
                    free_moe_ffn_maybe_slab(gpu, slabs, l.ffn);
                }
            }
        }
        // MAD-93 v0.1: in paged mode, the pager owns expert weight allocations
        // (the per-layer `free_moe_ffn` loops ran no-ops since `ffn.experts`
        // was empty). Drain the pager's resident set back to the GPU pool here.
        if let Some(pager_cell) = pager {
            pager_cell.into_inner().free_all(gpu);
        }
        if let Some(storage) = slab_storage {
            storage.free_gpu(gpu);
        }
    }

    /// Multi-GPU companion to `free_gpu`. Each layer freed on its
    /// band-owning device per `gpus.device_for_layer(i)`; `token_embd`
    /// freed on dev 0; `output_norm + output` on `gpus.output_device`.
    /// Mirror of `load_weights_multi` placement. The `pager` field is
    /// always `None` on the multi path (paged-experts is not wired into
    /// pp>1 yet); a non-None pager would need its own per-band drain
    /// strategy and is rejected at load.
    pub fn free_gpu_multi(self, gpus: &mut Gpus) {
        debug_assert!(
            self.pager.is_none(),
            "free_gpu_multi: pager must be None on pp>1 path"
        );
        // A HARD assert, not a debug_assert like its neighbour above, because
        // the failure it prevents is silent weight corruption rather than a
        // leak — and a panic is strictly better than that.
        //
        // Every free below is a bare `free_tensor`: this whole path has no slab
        // awareness at all, unlike `free_gpu`, which routes through
        // `free_tensor_maybe_slab`. A slab-backed tensor here would put a
        // mid-slab pointer on `GpuPool`'s free list and the next `pool.alloc`
        // would hand it out as scratch, writing over live weights. That exact
        // bug was live in `shard_moe_experts` until it was fixed.
        //
        // Safe TODAY only because `load_weights_multi` hardcodes
        // `slab_storage: None` — safe by DATA, not by type, and one loader
        // change from being live. Guarding the two `free_moe_ffn` calls alone
        // would be worse than this: it would look like the path had been made
        // slab-safe while ~40 other frees stayed exposed. Making the path
        // genuinely slab-safe means threading `slabs` through all of them,
        // which is a real change and wants its own reasoning; asserting the
        // assumption is the honest interim.
        assert!(
            self.slab_storage.is_none(),
            "free_gpu_multi: slab-backed weights on the pp>1 path. Every free here \
             is unguarded, so freeing a slab alias would corrupt live weights. \
             Either route this path through free_tensor_maybe_slab (threading \
             slabs through every free), or keep load_weights_multi's \
             slab_storage: None."
        );
        let _ = gpus.devices[0].free_tensor(self.token_embd);
        let out_dev = gpus.output_device;
        let _ = gpus.devices[out_dev].free_tensor(self.output_norm);
        let _ = gpus.devices[out_dev].free_tensor(self.output.buf);
        for (i, layer) in self.layers.into_iter().enumerate() {
            let dev_idx = gpus.device_for_layer(i);
            let gpu = &mut gpus.devices[dev_idx];
            match layer {
                LayerWeights::DeltaNet(l) => {
                    let _ = gpu.free_tensor(l.attn_norm);
                    let _ = gpu.free_tensor(l.wqkv.buf);
                    let _ = gpu.free_tensor(l.wz.buf);
                    let _ = gpu.free_tensor(l.w_alpha.buf);
                    let _ = gpu.free_tensor(l.w_beta.buf);
                    let _ = gpu.free_tensor(l.a_log);
                    let _ = gpu.free_tensor(l.dt_bias);
                    let _ = gpu.free_tensor(l.conv_weight);
                    let _ = gpu.free_tensor(l.norm_weight);
                    let _ = gpu.free_tensor(l.wo.buf);
                    let _ = gpu.free_tensor(l.ffn_norm);
                    let _ = gpu.free_tensor(l.w_gate.buf);
                    let _ = gpu.free_tensor(l.w_up.buf);
                    let _ = gpu.free_tensor(l.w_down.buf);
                }
                LayerWeights::FullAttn(l) => {
                    let _ = gpu.free_tensor(l.attn_norm);
                    let _ = gpu.free_tensor(l.wq.buf);
                    let _ = gpu.free_tensor(l.wk.buf);
                    let _ = gpu.free_tensor(l.wv.buf);
                    let _ = gpu.free_tensor(l.wo.buf);
                    let _ = gpu.free_tensor(l.q_norm);
                    let _ = gpu.free_tensor(l.k_norm);
                    let _ = gpu.free_tensor(l.ffn_norm);
                    let _ = gpu.free_tensor(l.w_gate.buf);
                    let _ = gpu.free_tensor(l.w_up.buf);
                    let _ = gpu.free_tensor(l.w_down.buf);
                }
                LayerWeights::DeltaNetMoe(l) => {
                    let _ = gpu.free_tensor(l.attn_norm);
                    let _ = gpu.free_tensor(l.wqkv.buf);
                    let _ = gpu.free_tensor(l.wz.buf);
                    let _ = gpu.free_tensor(l.w_alpha.buf);
                    let _ = gpu.free_tensor(l.w_beta.buf);
                    let _ = gpu.free_tensor(l.a_log);
                    let _ = gpu.free_tensor(l.dt_bias);
                    let _ = gpu.free_tensor(l.conv_weight);
                    let _ = gpu.free_tensor(l.norm_weight);
                    let _ = gpu.free_tensor(l.wo.buf);
                    let _ = gpu.free_tensor(l.ffn_norm);
                    free_moe_ffn(gpu, l.ffn);
                }
                LayerWeights::FullAttnMoe(l) => {
                    let _ = gpu.free_tensor(l.attn_norm);
                    let _ = gpu.free_tensor(l.wq.buf);
                    let _ = gpu.free_tensor(l.wk.buf);
                    let _ = gpu.free_tensor(l.wv.buf);
                    let _ = gpu.free_tensor(l.wo.buf);
                    let _ = gpu.free_tensor(l.q_norm);
                    let _ = gpu.free_tensor(l.k_norm);
                    let _ = gpu.free_tensor(l.ffn_norm);
                    free_moe_ffn(gpu, l.ffn);
                }
            }
        }
    }
}

pub fn validate_paged_moe_decode_expert_cache(
    weights: &Qwen35Weights,
    config: &Qwen35Config,
) -> Result<(), String> {
    if !config.paged_experts || config.num_experts == 0 {
        return Ok(());
    }
    let pager = weights
        .pager
        .as_ref()
        .ok_or_else(|| "paged Qwen35-MoE decode requires a weight pager".to_string())?;
    let layer = weights
        .layers
        .iter()
        .find_map(|layer| match layer {
            LayerWeights::DeltaNetMoe(layer) => Some(layer.ffn.layer_idx),
            LayerWeights::FullAttnMoe(layer) => Some(layer.ffn.layer_idx),
            _ => None,
        })
        .ok_or_else(|| "paged Qwen35-MoE decode requires a MoE layer".to_string())?;
    pager
        .borrow()
        .would_fit_largest_expert_module_set(layer, config.num_experts_per_tok)
        .map_err(|e| format!("paged Qwen35-MoE decode expert cache too small: {e}"))
}

/// Unguarded MoE-FFN teardown for the `pp > 1` path ONLY.
///
/// The guarded version is [`free_moe_ffn_maybe_slab`]; use that one unless you
/// are `free_gpu_multi`, which asserts `slab_storage.is_none()` precisely so
/// this is sound. Two aliasing sources make the distinction matter, and neither
/// is checked here:
///
/// * slab-backed tensors — non-owning aliases into a weight slab;
/// * `raw_expert_storage` — expert buffers that are interior slices of one
///   stacked allocation, which `free_moe_ffn_maybe_slab` `mem::forget`s.
///
/// `paro_shared` IS handled below, which is the tell that the other two were
/// oversights rather than deliberate: the same function already knows that
/// freeing a non-owning view is wrong.
fn free_moe_ffn(gpu: &mut Gpu, ffn: MoeFfnWeights) {
    debug_assert!(
        ffn.raw_expert_storage.is_none(),
        "free_moe_ffn: raw_expert_storage aliases would be double-freed here; \
         use free_moe_ffn_maybe_slab"
    );
    let _ = gpu.free_tensor(ffn.router.buf);
    let _ = gpu.free_tensor(ffn.shared_expert_gate.buf);
    let _ = gpu.free_tensor(ffn.shared_expert.gate.buf);
    let _ = gpu.free_tensor(ffn.shared_expert.up.buf);
    let _ = gpu.free_tensor(ffn.shared_expert.down.buf);
    if let Some(t) = ffn.expert_gate_up_strides {
        let _ = gpu.free_tensor(t);
    }
    if let Some(t) = ffn.expert_down_strides {
        let _ = gpu.free_tensor(t);
    }
    let _ = gpu.free_tensor(ffn.expert_gate_up_ptrs);
    let _ = gpu.free_tensor(ffn.expert_down_ptrs);
    for e in ffn.experts {
        let _ = gpu.free_tensor(e.gate_up.buf);
        let _ = gpu.free_tensor(e.down.buf);
    }
    // ParoQuant MoE: free the owning shared sidecars (per-expert `paro` fields
    // alias these and must NOT be freed separately — they're non-owning views).
    if let Some(s) = ffn.paro_shared {
        let _ = gpu.free_tensor(s.gate_up_pairs);
        let _ = gpu.free_tensor(s.gate_up_theta);
        let _ = gpu.free_tensor(s.gate_up_channel_scales);
        let _ = gpu.free_tensor(s.down_pairs);
        let _ = gpu.free_tensor(s.down_theta);
        let _ = gpu.free_tensor(s.down_channel_scales);
    }
}

pub struct ModelGpuStorage {
    pub(crate) slabs: Vec<GpuTensor>,
    pub(crate) bytes: usize,
}

impl ModelGpuStorage {
    pub(crate) fn new(slabs: Vec<GpuTensor>, bytes: usize) -> Self {
        Self { slabs, bytes }
    }

    fn contains_tensor(&self, tensor: &GpuTensor) -> bool {
        let ptr = tensor.buf.as_ptr() as usize;
        self.slabs.iter().any(|slab| {
            let start = slab.buf.as_ptr() as usize;
            let end = start.saturating_add(slab.buf.size());
            ptr >= start && ptr < end
        })
    }

    fn free_gpu(self, gpu: &mut Gpu) {
        for slab in self.slabs {
            let _ = gpu.free_tensor(slab);
        }
    }
}

pub(super) fn free_tensor_maybe_slab(
    gpu: &mut Gpu,
    slabs: Option<&ModelGpuStorage>,
    tensor: GpuTensor,
) {
    if slabs.is_some_and(|s| s.contains_tensor(&tensor)) {
        std::mem::forget(tensor);
    } else {
        let _ = gpu.free_tensor(tensor);
    }
}

pub(super) fn free_weight_tensor_maybe_slab(
    gpu: &mut Gpu,
    slabs: Option<&ModelGpuStorage>,
    wt: WeightTensor,
) {
    if let Some(scale) = wt.awq_scale {
        let _ = gpu.free_tensor(scale);
    }
    free_tensor_maybe_slab(gpu, slabs, wt.buf);
}

fn free_moe_ffn_maybe_slab(gpu: &mut Gpu, slabs: Option<&ModelGpuStorage>, ffn: MoeFfnWeights) {
    let raw_expert_storage = ffn.raw_expert_storage;
    free_weight_tensor_maybe_slab(gpu, slabs, ffn.router);
    free_weight_tensor_maybe_slab(gpu, slabs, ffn.shared_expert_gate);
    free_weight_tensor_maybe_slab(gpu, slabs, ffn.shared_expert.gate);
    free_weight_tensor_maybe_slab(gpu, slabs, ffn.shared_expert.up);
    free_weight_tensor_maybe_slab(gpu, slabs, ffn.shared_expert.down);
    let _ = gpu.free_tensor(ffn.expert_gate_up_ptrs);
    let _ = gpu.free_tensor(ffn.expert_down_ptrs);
    for e in ffn.experts {
        if raw_expert_storage.is_some() {
            // These buffers alias slices of the owning stacked allocations.
            std::mem::forget(e.gate_up.buf);
            std::mem::forget(e.down.buf);
        } else {
            free_weight_tensor_maybe_slab(gpu, slabs, e.gate_up);
            free_weight_tensor_maybe_slab(gpu, slabs, e.down);
        }
    }
    if let Some(storage) = raw_expert_storage {
        let _ = gpu.free_tensor(storage.gate_up);
        let _ = gpu.free_tensor(storage.down);
    }
}

pub(crate) fn free_streamed_layer_weights(gpu: &mut Gpu, layer: LayerWeights) {
    match layer {
        LayerWeights::DeltaNet(l) => {
            let _ = gpu.free_tensor(l.attn_norm);
            let _ = gpu.free_tensor(l.wqkv.buf);
            let _ = gpu.free_tensor(l.wz.buf);
            let _ = gpu.free_tensor(l.w_alpha.buf);
            let _ = gpu.free_tensor(l.w_beta.buf);
            let _ = gpu.free_tensor(l.a_log);
            let _ = gpu.free_tensor(l.dt_bias);
            let _ = gpu.free_tensor(l.conv_weight);
            let _ = gpu.free_tensor(l.norm_weight);
            let _ = gpu.free_tensor(l.wo.buf);
            let _ = gpu.free_tensor(l.ffn_norm);
            let _ = gpu.free_tensor(l.w_gate.buf);
            let _ = gpu.free_tensor(l.w_up.buf);
            let _ = gpu.free_tensor(l.w_down.buf);
        }
        LayerWeights::FullAttn(l) => {
            let _ = gpu.free_tensor(l.attn_norm);
            let _ = gpu.free_tensor(l.wq.buf);
            let _ = gpu.free_tensor(l.wk.buf);
            let _ = gpu.free_tensor(l.wv.buf);
            let _ = gpu.free_tensor(l.wo.buf);
            let _ = gpu.free_tensor(l.q_norm);
            let _ = gpu.free_tensor(l.k_norm);
            let _ = gpu.free_tensor(l.ffn_norm);
            let _ = gpu.free_tensor(l.w_gate.buf);
            let _ = gpu.free_tensor(l.w_up.buf);
            let _ = gpu.free_tensor(l.w_down.buf);
        }
        LayerWeights::DeltaNetMoe(l) => {
            let _ = gpu.free_tensor(l.attn_norm);
            let _ = gpu.free_tensor(l.wqkv.buf);
            let _ = gpu.free_tensor(l.wz.buf);
            let _ = gpu.free_tensor(l.w_alpha.buf);
            let _ = gpu.free_tensor(l.w_beta.buf);
            let _ = gpu.free_tensor(l.a_log);
            let _ = gpu.free_tensor(l.dt_bias);
            let _ = gpu.free_tensor(l.conv_weight);
            let _ = gpu.free_tensor(l.norm_weight);
            let _ = gpu.free_tensor(l.wo.buf);
            let _ = gpu.free_tensor(l.ffn_norm);
            free_moe_ffn_maybe_slab(gpu, None, l.ffn);
        }
        LayerWeights::FullAttnMoe(l) => {
            let _ = gpu.free_tensor(l.attn_norm);
            let _ = gpu.free_tensor(l.wq.buf);
            let _ = gpu.free_tensor(l.wk.buf);
            let _ = gpu.free_tensor(l.wv.buf);
            let _ = gpu.free_tensor(l.wo.buf);
            let _ = gpu.free_tensor(l.q_norm);
            let _ = gpu.free_tensor(l.k_norm);
            let _ = gpu.free_tensor(l.ffn_norm);
            free_moe_ffn_maybe_slab(gpu, None, l.ffn);
        }
    }
}
