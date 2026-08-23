// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Qwen3.5 prefill chunk executor: the batched per-chunk prefill layer loop
//! (`forward_prefill_chunk`) plus its batched MoE FFN body
//! (`prefill_moe_ffn_body_batched`) and full-attention layer body
//! (`run_fa_layer_body`). On the prefill hot path.

use super::prefill_batch::*;
use super::*;

/// Activation precision for one oq4 prefill projection site (W4A4 experiments).
///
/// `HIPFIRE_OQ4_PREFILL_ACT_BITS_<SITE>` (QKV | GATEUP | O | DOWN) overrides the
/// global `HIPFIRE_OQ4_PREFILL_ACT_BITS` for that site alone; with both unset the
/// production routing is unchanged. Values: `4` (int4 activation), `8` (int8-MMQ),
/// `16` (f16 activation); all four sites accept all three.
///
/// Note which sites are actually int4 by default at prefill batch sizes: **QKV
/// and GATEUP both route to int8 MMQ at n>=64**, so only O and DOWN run int4.
/// That means a global `=4` does NOT produce an all-int4 prefill — per-site `=4`
/// on QKV/GATEUP is what forces those (plan §13i). A global `=8` gives a fully-A8
/// prefill.
///
/// The per-site form started as a way to hold ONE projection at A16 while the
/// rest ran A4 — an upper bound on what mixed precision at that site could buy,
/// measurable before building anything. That sweep found `o_proj` to be the
/// activation-sensitive site (plan §13c), and `8` is the real lever it pointed at.
/// The site names are spelled out rather than interpolated so the env-doc
/// scanner (and anyone grepping) can see them, and so the lookup does not
/// allocate on the prefill path.
pub(crate) fn oq4_act_bits(site: &str) -> Option<String> {
    let per_site = match site {
        "QKV" => std::env::var("HIPFIRE_OQ4_PREFILL_ACT_BITS_QKV"),
        "GATEUP" => std::env::var("HIPFIRE_OQ4_PREFILL_ACT_BITS_GATEUP"),
        "O" => std::env::var("HIPFIRE_OQ4_PREFILL_ACT_BITS_O"),
        "DOWN" => std::env::var("HIPFIRE_OQ4_PREFILL_ACT_BITS_DOWN"),
        other => panic!("oq4_act_bits: unknown site {other}"),
    };
    per_site
        .ok()
        .or_else(|| std::env::var("HIPFIRE_OQ4_PREFILL_ACT_BITS").ok())
}

/// Batched MoE FFN for `forward_prefill_chunk`. Takes the post-attention
/// residual stream in `pbs.x_batch` ([N × dim]) and writes the FFN output
/// residual back into the same buffer in-place.
///
/// Preconditions (caller must guarantee):
/// - `moe_ffn_batched_admissible(ffn, arch)` returns true: router +
///   shared_expert_gate may be MQ4G256 or Q8_0; all other MoE weights must
///   use an arch-supported MoE quant family.
/// - `pbs.moe_*_batch` tensors are allocated (num_experts > 0 at scratch
///   construction time) and sized to max_batch ≥ N
/// - `config.num_experts_per_tok` is an admitted routed shape (currently K=8
///   or K=10) and `config.num_experts <= 1024`; K=8 uses the GPU top-K reducer
///   while K=10 uses the deterministic host merge/upload path below
///
/// Sequence mirrors `moe_ffn_decode_impl`'s GPU fast path, with every
/// per-token launch replaced by its N-batched equivalent. Byte-exact
/// except for atomicAdd nondeterminism in the routed-down accumulation
/// (same as the single-token indexed kernel it replaces).
#[allow(clippy::too_many_arguments)]
pub(crate) fn prefill_moe_ffn_body_batched(
    gpu: &mut Gpu,
    pager: Option<&RefCell<hipfire_runtime::weight_pager::WeightPager>>,
    ffn: &MoeFfnWeights,
    ffn_norm: &GpuTensor,
    config: &Qwen35Config,
    pbs: &PrefillBatchScratch,
    n: usize,
    layer_idx: usize,
    ctx: &DispatchCtx,
    // EP (Ship 6 substrate-EP prefill): when `Some`, the routed combine writes
    // into this zeroed `[n × dim]` partial instead of `pbs.x_batch` (the EP
    // driver all-reduce-sums it across ranks and adds into x_batch). The shared
    // expert (step 5) stays in `pbs.x_batch` — replicated per rank, not
    // redirected. `None` = byte-identical single-GPU behavior.
    routed_out: Option<&GpuTensor>,
    capture: Option<&dyn hipfire_dispatch::families::moe::MoePrefillCapture>,
) -> HipResult<()> {
    let dim = config.dim;
    let mi = config.moe_intermediate_size;
    let smi = config.shared_expert_intermediate_size;
    let k_top = config.num_experts_per_tok;
    let n_exp = config.num_experts;

    let router_logits = pbs.moe_router_logits_batch.as_ref().expect("moe scratch");
    let shared_scalar = pbs.moe_shared_scalar_batch.as_ref().expect("moe scratch");
    let shared_gate = pbs.moe_shared_gate_batch.as_ref().expect("moe scratch");
    let shared_up = pbs.moe_shared_up_batch.as_ref().expect("moe scratch");
    let shared_rot = pbs.moe_shared_rot_batch.as_ref().expect("moe scratch");
    let topk_indices = pbs.moe_topk_indices_batch.as_ref().expect("moe scratch");
    let topk_weights = pbs.moe_topk_weights_batch.as_ref().expect("moe scratch");
    let gate_batch = pbs.moe_gate_batch.as_ref().expect("moe scratch");
    let up_batch = pbs.moe_up_batch.as_ref().expect("moe scratch");
    let hidden_batch = pbs.moe_hidden_batch.as_ref().expect("moe scratch");
    let rot_batch = pbs.moe_rot_batch.as_ref().expect("moe scratch");
    let down_expanded = pbs.moe_down_expanded_batch.as_ref().expect("moe scratch");
    let dtypes = MoePrefillDtypes::from_ffn(ffn)
        .ok_or_else(|| HipError::new(0, "missing MoE expert dtype metadata for batched prefill"))?;
    if (!ffn.expert_gate_up_dtypes.is_empty() && ffn.expert_gate_up_dtypes.len() != n_exp)
        || (!ffn.expert_down_dtypes.is_empty() && ffn.expert_down_dtypes.len() != n_exp)
    {
        return Err(HipError::new(
            0,
            "MoE per-expert dtype metadata length differs from configured expert count",
        ));
    }
    let expert_shape = moe_expert_shape(ffn)
        .ok_or_else(|| HipError::new(0, "missing MoE expert shape metadata for batched prefill"))?;

    // ── 1. Split rmsnorm vs FWHT rotate ──
    //
    // A3B (and every other MoE here) leaves router + shared_expert_gate
    // as Q8_0 in the quantizer — these tiny tensors lose too much
    // accuracy at 4-bit, so the engine never reduces them. Q8 weights
    // are quantized against the un-rotated rmsnorm output, while the
    // MQ-family siblings (shared_expert.{gate,up,down} +
    // experts.{gate_up,down}) expect FWHT(rmsnorm(x) / awq_scale). Populate both:
    //   x_norm_batch ← rmsnorm(x_batch)
    //   x_rot_batch  ← FWHT(x_norm_batch / awq_scale)  (only if any
    //                  downstream MQ weight is present)
    //
    // Pick `shared_expert.gate` as the AWQ representative (instead of
    // the previous `ffn.router`). Per the F1 imatrix scope every gate-side
    // MQ4 sibling shares the same input basis and therefore an identical
    // awq_scale, but the router itself is excluded from F1 (it stays Q8).
    // Reading awq_scale from router would silently drop AWQ rotation in
    // v3 AWQ runs — latent until this predicate widened.
    gpu.rmsnorm_batched(
        &pbs.x_batch,
        ffn_norm,
        &pbs.x_norm_batch,
        n,
        dim,
        config.norm_eps,
    )?;
    // PARO mode (shared_expert.gate is ParoQ4G128): each weight carries its
    // own Givens rotation table (paro.pairs / theta / channel_scales). The
    // shared MQ4-style FWHT pre-rotation here would be wrong — skip it. The
    // ParoQ4G128 dispatch arms below run per-weight Givens rotation in-place
    // before each GEMM, using pbs.x_rot_batch as the rotation destination.
    let paro_mode =
        config.has_shared_expert && matches!(ffn.shared_expert.gate.gpu_dtype, DType::ParoQ4G128);
    if !paro_mode {
        if config.has_shared_expert {
            rotate_x_mq_batched_for(
                gpu,
                &ffn.shared_expert.gate,
                &pbs.x_norm_batch,
                &pbs.x_rot_batch,
                dim,
                n,
            )?;
        } else {
            gpu.rotate_x_mq_batched(&pbs.x_norm_batch, &pbs.x_rot_batch, dim, n)?;
        }
    }

    // ── 2. Router + shared-gate + shared.gate + shared.up (4 batched GEMMs) ──
    //
    // Per-dtype dispatch — Q8 reads `x_norm_batch`, MQ4 reads
    // `x_rot_batch`. The natural 4-way fuse via `gemm_qkvza_hfq4g256`
    // is not applicable when router/shared_expert_gate are Q8 (mixed
    // strides). Four separate launches; +3 per MoE layer over the fused
    // ideal, acceptable for the structural unlock.
    let raw_router = match ffn.router.gpu_dtype {
        DType::F16 => gpu
            .gemm_f16_x_f32_wmma(
                &ffn.router.buf,
                &pbs.x_norm_batch,
                router_logits,
                ffn.router.m,
                ffn.router.k,
                n,
            )
            .map(|()| true)?,
        DType::BF16 => gpu
            .gemm_bf16_x_bf16_wmma(
                &ffn.router.buf,
                &pbs.x_norm_batch,
                router_logits,
                ffn.router.m,
                ffn.router.k,
                n,
            )
            .map(|()| true)?,
        _ => false,
    };
    if config.has_shared_expert {
        match ffn.shared_expert_gate.gpu_dtype {
            DType::F16 => gpu.gemm_f16_x_f32_wmma(
                &ffn.shared_expert_gate.buf,
                &pbs.x_norm_batch,
                shared_scalar,
                ffn.shared_expert_gate.m,
                ffn.shared_expert_gate.k,
                n,
            )?,
            DType::BF16 => gpu.gemm_bf16_x_bf16_wmma(
                &ffn.shared_expert_gate.buf,
                &pbs.x_norm_batch,
                shared_scalar,
                ffn.shared_expert_gate.m,
                ffn.shared_expert_gate.k,
                n,
            )?,
            _ => {}
        }
    }
    // #397 Ship 5.2 PILOT: route the router GEMM through GemmFamily::run_key.
    // Each arm uses the *dispatcher-entry* KernelKey (GemmQ8_0BatchedChunked /
    // GemmHfq4G256 / GemmF32Batched) so run_key dispatches to the IDENTICAL
    // gpu.gemm_* method the prior direct call used — preserving each method's
    // own internal arch routing (RDNA4-WMMA / gfx906-dp4a / CDNA-rocBLAS / …)
    // byte-for-byte. The x input still differs per dtype (Q8/F32 read
    // x_norm_batch; MQ4 reads x_rot_batch), exactly as before. The three keys
    // are registered ArchPredicate::Always, so run_key never rejects.
    if !raw_router {
        use hipfire_dispatch::families::gemm::GemmParams;
        let ctx = DispatchCtx::new(gpu);
        let (key, x_in): (hipfire_dispatch::types::KernelKey, &GpuTensor) =
            match ffn.router.gpu_dtype {
                DType::Q8_0 => (
                    hipfire_dispatch::types::KernelKey::GemmQ8_0BatchedChunked,
                    &pbs.x_norm_batch,
                ),
                DType::MQ4G256 => (
                    hipfire_dispatch::types::KernelKey::GemmHfq4G256,
                    &pbs.x_rot_batch,
                ),
                DType::F32 => (
                    hipfire_dispatch::types::KernelKey::GemmF32Batched,
                    &pbs.x_norm_batch,
                ),
                other => panic!(
                    "prefill_moe_ffn_body_batched: unexpected router dtype {other:?} \
                         — moe_ffn_batched_admitted admits MQ4G256, Q8_0, F32"
                ),
            };
        let w = WeightRef {
            buf: &ffn.router.buf,
            dtype: ffn.router.gpu_dtype,
            m: ffn.router.m,
            k: ffn.router.k,
            row_stride: ffn.router.k,
            rotation: None,
            awq_scale: None,
        };
        let params = GemmParams {
            w: &w,
            x: x_in,
            y: router_logits,
            batch_size: n,
        };
        hipfire_runtime::dispatch::gemm_family()
            .run_key(key, &ctx, gpu, &params)
            .map_err(HipError::from)?;
    }
    // DIAG: dump MoE router logits (batched)
    dump_hidden_localize(gpu, router_logits, n, 0, ffn.router.m, 0, "router_b");
    // #397 Ship 5.2 slice1: route the shared-expert-gate GEMM through
    // GemmFamily::run_key. Same dtype-routed dispatcher-entry keys as the router
    // match above (Q8/F32 read x_norm_batch, MQ4 reads x_rot_batch) → identical
    // gpu.gemm_* method, byte-for-byte.
    if !matches!(ffn.shared_expert_gate.gpu_dtype, DType::F16 | DType::BF16) {
        use hipfire_dispatch::types::KernelKey;
        let (key, x_in): (KernelKey, &GpuTensor) = match ffn.shared_expert_gate.gpu_dtype {
            DType::Q8_0 => (KernelKey::GemmQ8_0BatchedChunked, &pbs.x_norm_batch),
            DType::MQ4G256 => (KernelKey::GemmHfq4G256, &pbs.x_rot_batch),
            DType::F32 => (KernelKey::GemmF32Batched, &pbs.x_norm_batch),
            other => panic!(
                "prefill_moe_ffn_body_batched: unexpected shared_expert_gate dtype {other:?} \
                         — moe_ffn_batched_admissible admits MQ4G256, Q8_0, F32"
            ),
        };
        run_plain_gemm_key(
            gpu,
            key,
            &ffn.shared_expert_gate.buf,
            ffn.shared_expert_gate.gpu_dtype,
            x_in,
            shared_scalar,
            ffn.shared_expert_gate.m,
            ffn.shared_expert_gate.k,
            n,
        )?;
    }
    // Fused gate+up dispatch for the shared expert — halves the kernel
    // launch count vs back-to-back gemm_hfq*g256 (~75µs/launch × 40
    // MoE layers = ~3ms saved on R9700 A3B prefill at bs=256).
    // Per-projection dispatch: gate AND up share the same dtype (predicate
    // enforces). MQ4/MQ3/MQ6 route to their HFQ-layout fused kernels.
    if config.has_shared_expert {
        match ffn.shared_expert.gate.gpu_dtype {
            // #397 Ship 5.2 slice 2: shared-expert fused gate+up → FusedQkvFamily
            // (batched-prefill gate+up variant). Same batched kernel, behavior-preserving.
            DType::MQ4G256 => run_fused_gate_up_key(
                gpu,
                hipfire_dispatch::types::KernelKey::FusedGateUpHfq4G256,
                &ffn.shared_expert.gate.buf,
                &ffn.shared_expert.up.buf,
                &pbs.x_rot_batch,
                shared_gate,
                shared_up,
                ffn.shared_expert.gate.m,
                ffn.shared_expert.up.m,
                ffn.shared_expert.gate.k,
                n,
            )?,
            DType::MQ6G256 => run_fused_gate_up_key(
                gpu,
                hipfire_dispatch::types::KernelKey::FusedGateUpHfq6G256,
                &ffn.shared_expert.gate.buf,
                &ffn.shared_expert.up.buf,
                &pbs.x_rot_batch,
                shared_gate,
                shared_up,
                ffn.shared_expert.gate.m,
                ffn.shared_expert.up.m,
                ffn.shared_expert.gate.k,
                n,
            )?,
            // Phase 2: PARO shared_expert.gate + up. Each weight has its own
            // Givens rotation table — rotate x_norm_batch into x_rot_batch using
            // gate's tables, GEMM, then re-rotate using up's tables, GEMM. Total
            // 4 dispatches vs the MQ4 path's 1 fused gemm_gate_up — acceptable
            // overhead for the per-token-loop elimination win. Phase 4 could
            // collapse this into a single fused kernel
            // (gemm_gate_up_paro_q4g128_batched) if measurement shows it matters.
            DType::ParoQ4G128 => {
                let paro_gate = ffn
                    .shared_expert
                    .gate
                    .paro
                    .as_ref()
                    .expect("ParoQ4G128 shared_expert.gate missing paro metadata");
                let paro_up = ffn
                    .shared_expert
                    .up
                    .paro
                    .as_ref()
                    .expect("ParoQ4G128 shared_expert.up missing paro metadata");
                // Gate: rotate x_norm by gate's Givens → x_rot, then HFQ4G128 GEMM
                gpu.givens_rotate_to(
                    &pbs.x_norm_batch,
                    &pbs.x_rot_batch,
                    &paro_gate.pairs,
                    &paro_gate.theta,
                    &paro_gate.channel_scales,
                    n,
                    dim,
                    paro_gate.krot as usize,
                )?;
                run_plain_gemm_key(
                    gpu,
                    hipfire_dispatch::types::KernelKey::GemmHfq4G128,
                    &ffn.shared_expert.gate.buf,
                    ffn.shared_expert.gate.gpu_dtype,
                    &pbs.x_rot_batch,
                    shared_gate,
                    ffn.shared_expert.gate.m,
                    ffn.shared_expert.gate.k,
                    n,
                )?;
                // Up: re-rotate x_norm by up's Givens → x_rot (overwrite), GEMM
                gpu.givens_rotate_to(
                    &pbs.x_norm_batch,
                    &pbs.x_rot_batch,
                    &paro_up.pairs,
                    &paro_up.theta,
                    &paro_up.channel_scales,
                    n,
                    dim,
                    paro_up.krot as usize,
                )?;
                run_plain_gemm_key(
                    gpu,
                    hipfire_dispatch::types::KernelKey::GemmHfq4G128,
                    &ffn.shared_expert.up.buf,
                    ffn.shared_expert.up.gpu_dtype,
                    &pbs.x_rot_batch,
                    shared_up,
                    ffn.shared_expert.up.m,
                    ffn.shared_expert.up.k,
                    n,
                )?;
            }
            DType::F16 => {
                debug_assert_eq!(
                    ffn.shared_expert.up.gpu_dtype,
                    DType::F16,
                    "shared_expert.gate/up dtype predicate should keep F16 paired"
                );
                gpu.gemm_f16_x_f32_wmma(
                    &ffn.shared_expert.gate.buf,
                    &pbs.x_norm_batch,
                    shared_gate,
                    ffn.shared_expert.gate.m,
                    ffn.shared_expert.gate.k,
                    n,
                )?;
                gpu.gemm_f16_x_f32_wmma(
                    &ffn.shared_expert.up.buf,
                    &pbs.x_norm_batch,
                    shared_up,
                    ffn.shared_expert.up.m,
                    ffn.shared_expert.up.k,
                    n,
                )?;
            }
            DType::BF16 => {
                debug_assert_eq!(
                    ffn.shared_expert.up.gpu_dtype,
                    DType::BF16,
                    "shared_expert.gate/up dtype predicate should keep BF16 paired"
                );
                gpu.gemm_bf16_x_bf16_wmma(
                    &ffn.shared_expert.gate.buf,
                    &pbs.x_norm_batch,
                    shared_gate,
                    ffn.shared_expert.gate.m,
                    ffn.shared_expert.gate.k,
                    n,
                )?;
                gpu.gemm_bf16_x_bf16_wmma(
                    &ffn.shared_expert.up.buf,
                    &pbs.x_norm_batch,
                    shared_up,
                    ffn.shared_expert.up.m,
                    ffn.shared_expert.up.k,
                    n,
                )?;
            }
            // Opus Quant shared expert: fused gate+up via the dense OQ WMMA keys
            // (x_rot_batch is FWHT-rotated above, same as the MQ4 arm).
            DType::Oq4G256 => run_fused_gate_up_key(
                gpu,
                hipfire_dispatch::types::KernelKey::FusedGateUpOq4G256,
                &ffn.shared_expert.gate.buf,
                &ffn.shared_expert.up.buf,
                &pbs.x_rot_batch,
                shared_gate,
                shared_up,
                ffn.shared_expert.gate.m,
                ffn.shared_expert.up.m,
                ffn.shared_expert.gate.k,
                n,
            )?,
            DType::Oq8G256 => run_fused_gate_up_key(
                gpu,
                hipfire_dispatch::types::KernelKey::FusedGateUpOq8G256,
                &ffn.shared_expert.gate.buf,
                &ffn.shared_expert.up.buf,
                &pbs.x_rot_batch,
                shared_gate,
                shared_up,
                ffn.shared_expert.gate.m,
                ffn.shared_expert.up.m,
                ffn.shared_expert.gate.k,
                n,
            )?,
            DType::OqCompactG256 => {
                // Compact-resident Opus shared expert. There is no
                // FusedGateUpOqCompact kernel key -- compact appears in dispatch
                // only as decode-side GEMV entries -- so run gate and up as two
                // compact GEMMs off ONE quantize of the shared rotated
                // activation. Same shape as the dense compact arms.
                gpu.quantize_act_oq8_batched_interleaved(
                    &pbs.x_rot_batch,
                    ffn.shared_expert.gate.m,
                    ffn.shared_expert.gate.k,
                    n,
                )?;
                for (w, y) in [
                    (&ffn.shared_expert.gate, shared_gate),
                    (&ffn.shared_expert.up, shared_up),
                ] {
                    let bs = super::prefill_batch::oq_compact_block_stride(w)?;
                    gpu.gemm_oq_compact_grouped_prequant(&w.buf, y, w.m, w.k, n, bs)?;
                }
            }
            other => panic!(
                "prefill_moe_ffn_body_batched: unsupported shared_expert.gate dtype {other:?} \
                             — admit predicate should have rejected this layer"
            ),
        }
    }

    // ── 3. GPU softmax + top-K + renorm, batched over N tokens ──
    //
    // Same Path B split as the decode call site: split the fused
    // softmax+topk+renorm into gpu.softmax_f32 + moe_topk_renorm_k8_batched
    // so prefill activations match the CPU-reference softmax math
    // exactly. router_logits is allocated 1D as [n × n_exp]; alias it
    // into a 2D view so gpu.softmax_f32 takes rows = n.
    let router_logits_2d = GpuTensor {
        buf: unsafe { router_logits.buf.alias() },
        shape: vec![n, n_exp],
        dtype: DType::F32,
    };
    gpu.softmax_f32(&router_logits_2d)?;
    let cpu_topk = if k_top == 8 {
        gpu.moe_topk_renorm_k8_batched(
            router_logits,
            topk_indices,
            topk_weights,
            n_exp,
            config.norm_topk_prob,
            n,
        )?;
        None
    } else {
        let probs = gpu.download_f32(router_logits)?;
        let (indices, weights) =
            cpu_topk_from_softmaxed_rows(&probs, n, n_exp, k_top, config.norm_topk_prob)?;
        upload_cpu_topk_to_device(gpu, &indices, &weights, topk_indices, topk_weights)?;
        Some((indices, weights))
    };
    if moe_router_histogram_active() {
        let (indices, weights) = if let Some((indices, weights)) = cpu_topk.as_ref() {
            (indices.clone(), weights.clone())
        } else {
            (
                download_i32_tensor(gpu, topk_indices, n * k_top)?
                    .into_iter()
                    .map(router_index_i32_to_usize)
                    .collect::<Vec<_>>(),
                gpu.download_f32(topk_weights)?,
            )
        };
        for token_idx in 0..n {
            let start = token_idx * k_top;
            let end = start + k_top;
            record_moe_router_selection(layer_idx, &indices[start..end], &weights[start..end]);
        }
    }
    let bucketed_routed_experts = ffn.experts.is_empty() || dtypes.routed_profile.is_mixed();
    if bucketed_routed_experts && capture.is_some() {
        return Err(HipError::new(
            0,
            "bucketed mixed/paged routed-expert prefill capture is not supported",
        ));
    }
    let bucket_topk_indices = if bucketed_routed_experts {
        let indices = if let Some((indices, _)) = cpu_topk.as_ref() {
            indices.clone()
        } else {
            download_i32_tensor(gpu, topk_indices, n * k_top)?
                .into_iter()
                .map(router_index_i32_to_usize)
                .collect::<Vec<_>>()
        };
        Some(indices)
    } else {
        None
    };
    let routed_expert_buckets = if let Some(indices) = bucket_topk_indices.as_ref() {
        Some(build_paged_moe_expert_buckets(indices, n, k_top, n_exp)?)
    } else {
        None
    };

    // ── 4. Shared-expert SwiGLU + FWHT, batched over N tokens ──
    //
    // fused_silu_mul_rotate_mq_batched expects [batch × k] gate/up with
    // batch on grid.y and writes FWHT(silu(gate) * up) into x_rot. Here
    // batch=N, k=smi; the shared-rot output buffer is [N × smi].
    // F2: AWQ-aware silu_mul+rotate for the batched shared-expert down input.
    // PARO: shared_expert.down has its own Givens rotation tables (paro.*);
    // use the dedicated fused kernel (commit 50198daa). It takes a per-weight
    // (pairs, theta, channel_scales, krot) tuple instead of the MQ4 FWHT
    // convention. Same shape: gate/up [N × smi] → shared_rot [N × smi].
    if config.has_shared_expert {
        if paro_mode {
            let paro_down = ffn
                .shared_expert
                .down
                .paro
                .as_ref()
                .expect("ParoQ4G128 shared_expert.down missing paro metadata");
            gpu.fused_silu_mul_givens_rotate_f32(
                shared_gate,
                shared_up,
                shared_rot,
                &paro_down.pairs,
                &paro_down.theta,
                &paro_down.channel_scales,
                n,
                smi,
                paro_down.krot as usize,
            )?;
        } else if matches!(ffn.shared_expert.down.gpu_dtype, DType::F16 | DType::BF16) {
            gpu.silu_mul_f32(shared_gate, shared_up, shared_rot)?;
        } else {
            fused_silu_mul_rotate_mq_batched_for(
                gpu,
                &ffn.shared_expert.down,
                shared_gate,
                shared_up,
                shared_rot,
                smi,
                n,
            )?;
        }
    }

    // ── 5. Shared-expert down with sigmoid-scaled residual, batched ──
    //
    // Reads shared_scalar[token] as the pre-sigmoid logit, applies sigmoid
    // internally, and += sigmoid(scalar) × (W_down · rot) into
    // pbs.x_batch[token × dim + row]. (Note: HFQ4 sister uses += not
    // atomicAdd; each (bid, row) writes a unique cell.)
    // Per-projection dispatch: MQ4/MQ3/MQ6 route to their HFQ-layout sisters.
    if config.has_shared_expert {
        match ffn.shared_expert.down.gpu_dtype {
            DType::MQ4G256 => gpu.gemv_hfq4g256_residual_sigmoid_scaled_gpu_batched(
                &ffn.shared_expert.down.buf,
                shared_rot,
                &pbs.x_batch,
                shared_scalar,
                ffn.shared_expert.down.m,
                ffn.shared_expert.down.k,
                n,
            )?,
            DType::MQ6G256 => gpu.gemv_hfq6g256_residual_sigmoid_scaled_gpu_batched(
                &ffn.shared_expert.down.buf,
                shared_rot,
                &pbs.x_batch,
                shared_scalar,
                ffn.shared_expert.down.m,
                ffn.shared_expert.down.k,
                n,
            )?,
            DType::MQ3G256 => gpu.gemv_hfq3g256_residual_sigmoid_scaled_gpu_batched(
                &ffn.shared_expert.down.buf,
                shared_rot,
                &pbs.x_batch,
                shared_scalar,
                ffn.shared_expert.down.m,
                ffn.shared_expert.down.k,
                n,
            )?,
            DType::MQ2G256 => gpu.gemv_hfq2g256_residual_sigmoid_scaled_gpu_batched(
                &ffn.shared_expert.down.buf,
                shared_rot,
                &pbs.x_batch,
                shared_scalar,
                ffn.shared_expert.down.m,
                ffn.shared_expert.down.k,
                n,
            )?,
            DType::MQ8G256 => gpu.gemv_hfq8g256_residual_sigmoid_scaled_gpu_batched(
                &ffn.shared_expert.down.buf,
                shared_rot,
                &pbs.x_batch,
                shared_scalar,
                ffn.shared_expert.down.m,
                ffn.shared_expert.down.k,
                n,
            )?,
            DType::MQ2G256Lloyd => gpu.gemv_mq2g256_lloyd_residual_sigmoid_scaled_gpu_batched(
                &ffn.shared_expert.down.buf,
                shared_rot,
                &pbs.x_batch,
                shared_scalar,
                ffn.shared_expert.down.m,
                ffn.shared_expert.down.k,
                n,
            )?,
            DType::MQ3G256Lloyd => gpu.gemv_mq3g256_lloyd_residual_sigmoid_scaled_gpu_batched(
                &ffn.shared_expert.down.buf,
                shared_rot,
                &pbs.x_batch,
                shared_scalar,
                ffn.shared_expert.down.m,
                ffn.shared_expert.down.k,
                n,
            )?,
            // Phase 2: HFQ4G128 batched residual+sigmoid-scaled kernel. Single
            // launch, same semantics as the HFQ4G256 sister — reads shared_rot
            // (already silu-mul-rotated by the PARO fused kernel above), GEMVs
            // against W_down, applies sigmoid(shared_scalar[token]) × output,
            // accumulates into pbs.x_batch.
            DType::ParoQ4G128 => gpu.gemv_hfq4g128_residual_sigmoid_scaled_gpu_batched(
                &ffn.shared_expert.down.buf,
                shared_rot,
                &pbs.x_batch,
                shared_scalar,
                ffn.shared_expert.down.m,
                ffn.shared_expert.down.k,
                n,
            )?,
            DType::F16 => {
                let shared_down_scratch =
                    pbs.x_rot_batch.sub_offset(0, n * ffn.shared_expert.down.m);
                gpu.gemm_f16_x_f32_wmma(
                    &ffn.shared_expert.down.buf,
                    shared_rot,
                    &shared_down_scratch,
                    ffn.shared_expert.down.m,
                    ffn.shared_expert.down.k,
                    n,
                )?;
                let x_n = pbs.x_batch.sub_offset(0, n * ffn.shared_expert.down.m);
                gpu.scaled_add_inplace_gpu_sigmoid_rows_f32(
                    &x_n,
                    &shared_down_scratch,
                    shared_scalar,
                    ffn.shared_expert.down.m,
                    n,
                )?;
            }
            DType::BF16 => {
                let shared_down_scratch =
                    pbs.x_rot_batch.sub_offset(0, n * ffn.shared_expert.down.m);
                gpu.gemm_bf16_x_bf16_wmma(
                    &ffn.shared_expert.down.buf,
                    shared_rot,
                    &shared_down_scratch,
                    ffn.shared_expert.down.m,
                    ffn.shared_expert.down.k,
                    n,
                )?;
                let x_n = pbs.x_batch.sub_offset(0, n * ffn.shared_expert.down.m);
                gpu.scaled_add_inplace_gpu_sigmoid_rows_f32(
                    &x_n,
                    &shared_down_scratch,
                    shared_scalar,
                    ffn.shared_expert.down.m,
                    n,
                )?;
            }
            // Opus Quant shared-expert down: fused sigmoid-scaled residual GEMV.
            // W = down.buf (dense OQ combined layout); scales are the sub-offset
            // view after the packed weights (OQ4: M·K/2 bytes of nibbles; OQ8:
            // M·K bytes of int8). shared_rot is FWHT(silu·mul) from step 4.
            DType::Oq4G256 => {
                let dm = ffn.shared_expert.down.m;
                let dk = ffn.shared_expert.down.k;
                let ws = ffn
                    .shared_expert
                    .down
                    .buf
                    .sub_offset(dm * (dk / 2), dm * (dk / 256) * 4);
                gpu.gemv_oq4g256_residual_sigmoid_scaled_gpu_batched(
                    &ffn.shared_expert.down.buf,
                    &ws,
                    shared_rot,
                    &pbs.x_batch,
                    shared_scalar,
                    dm,
                    dk,
                    256,
                    n,
                )?;
            }
            DType::Oq8G256 => {
                let dm = ffn.shared_expert.down.m;
                let dk = ffn.shared_expert.down.k;
                let ws = ffn
                    .shared_expert
                    .down
                    .buf
                    .sub_offset(dm * dk, dm * (dk / 256) * 4);
                gpu.gemv_oq8g256_residual_sigmoid_scaled_gpu_batched(
                    &ffn.shared_expert.down.buf,
                    &ws,
                    shared_rot,
                    &pbs.x_batch,
                    shared_scalar,
                    dm,
                    dk,
                    256,
                    n,
                )?;
            }
            DType::OqCompactG256 => {
                // Compact shared-expert down. The Oq4/Oq8 arms above use a FUSED
                // residual+sigmoid GEMV; compact has no such kernel, so decompose
                // exactly as the F16/BF16 arms below already do: plain GEMM into
                // the x_rot_batch scratch, then the shared
                // `scaled_add_inplace_gpu_sigmoid_rows_f32`, which is
                // y[row,col] += sigmoid(shared_scalar[row]) * x[row,col].
                let dm = ffn.shared_expert.down.m;
                let dk = ffn.shared_expert.down.k;
                let shared_down_scratch = pbs.x_rot_batch.sub_offset(0, n * dm);
                let bs =
                    super::prefill_batch::oq_compact_block_stride(&ffn.shared_expert.down)?;
                gpu.quantize_act_oq8_batched_interleaved(shared_rot, dm, dk, n)?;
                gpu.gemm_oq_compact_grouped_prequant(
                    &ffn.shared_expert.down.buf,
                    &shared_down_scratch,
                    dm,
                    dk,
                    n,
                    bs,
                )?;
                let x_n = pbs.x_batch.sub_offset(0, n * dm);
                gpu.scaled_add_inplace_gpu_sigmoid_rows_f32(
                    &x_n,
                    &shared_down_scratch,
                    shared_scalar,
                    dm,
                    n,
                )?;
            }
            other => panic!(
                "prefill_moe_ffn_body_batched: unsupported shared_expert.down dtype {other:?} \
                         — admit predicate should have rejected this layer"
            ),
        }
    }

    // Paged experts retain page-in orchestration here. Mixed resident experts
    // use the same one-expert buckets so each launch interprets its weight
    // pointer with the correct low-bit or raw layout.
    // Path 2 (SGLang-style scatter + grouped-WMMA-GEMM) — default ON for gfx11/
    // gfx12, where the grouped-WMMA kernel is validated. Empirical lift on
    // Qwen3.5-A3B mq4 prefill=256: gfx1100 7900 XTX 1396 -> 2983 tok/s (+114%);
    // gfx1201 R9700 1016 -> 2966 tok/s (+192%). CDNA wave64 (gfx9*) and pre-WMMA
    // RDNA (gfx10*) stay on the per-token indexed GEMV. `HIPFIRE_MOE_GROUPED_GEMM=0`
    // opts out.
    //
    // HOISTED out of the bucketed block so a RESIDENT, uniform-dtype model can
    // reach it. Buckets are built only for paged (`ffn.experts.is_empty()`) or
    // mixed-profile models, so a resident model previously fell through to
    // section 6 and the per-token GEMV however eligible its dtype was. The
    // path-2 core needs no buckets: the scatter and unscatter are GPU-side off
    // topk_indices, and the CPU bucket download exists for PAGING.
    static USE_PATH2_GATE_UP: OnceLock<bool> = OnceLock::new();
    let use_path2 = *USE_PATH2_GATE_UP.get_or_init(|| {
        moe_grouped_gemm_path2_enabled_from_env(
            std::env::var("HIPFIRE_MOE_GROUPED_GEMM").ok().as_deref(),
        )
    });
    let path2_eligible = moe_grouped_gemm_path2_eligible_for_dtype(
        dtypes.expert_gate_up,
        &gpu.arch,
        (use_path2 || dtypes.routed_profile.is_mixed())
            && (!ffn.experts.is_empty() || routed_expert_buckets.is_some()),
    );
    // Report the DECISION, not one of its inputs. An earlier version of this
    // line keyed on `routed_expert_buckets` and kept printing "not reachable
    // here" after the hoist had made it reachable -- the exact failure the rule
    // in feature_report.rs warns about.
    if super::feature_report::wanted() {
        super::feature_report::note(
            "moe_routed",
            if path2_eligible {
                format!(
                    "GROUPED path-2 (scatter + grouped WMMA) gate_up={:?} buckets={}",
                    dtypes.expert_gate_up,
                    routed_expert_buckets.is_some()
                )
            } else {
                format!(
                    "indexed GEMV path-1 (dtype {:?} not path-2 eligible on {}) buckets={}",
                    dtypes.expert_gate_up,
                    gpu.arch,
                    routed_expert_buckets.is_some()
                )
            },
        );
    }
    if routed_expert_buckets.is_some() || path2_eligible {
        // ── 6. Routed experts: batched gate_up → SwiGLU+FWHT → down ──
        //
        // Gate/up for top-K experts (per token) → [N × K_TOP × mi]. Each
        // output row reads topk_indices[token × K_TOP + krank] to pick its
        // expert weight base from the device-side expert_gate_up_ptrs table.
        let down_m = expert_shape.down_m;
        let down_k = expert_shape.down_k;
        let gate_up_k = expert_shape.gate_up_k;

        // m_total — computed during gate_up scatter, reused for down. Avoids
        // a second dtoh sync per MoE layer.
        let mut path2_m_total: usize = 0;
        let path2_shape = moe_grouped_path2_shape(n, k_top, n_exp);
        // Same reasoning as the FA gate: when the grouped path is declined the
        // only outward sign is a GEMV-dominated histogram — measured 1279
        // dispatches/token on Qwen3.6-35B-A3B, with MoE routing at one dispatch
        // per token per layer. Name the reason instead.
        if !path2_eligible && hipfire_rdna::kernel_trace::enabled() {
            eprintln!(
                "[kernel-trace] MoE grouped GEMM declined: expert_gate_up={:?} arch={}                  use_path2={} mixed={} experts_empty={} buckets={} n={} k_top={} n_exp={}",
                dtypes.expert_gate_up,
                gpu.arch,
                use_path2,
                dtypes.routed_profile.is_mixed(),
                ffn.experts.is_empty(),
                routed_expert_buckets.is_some(),
                n,
                k_top,
                n_exp
            );
        }
        if routed_expert_buckets.is_some() && !path2_eligible {
            return Err(HipError::new(
                0,
                "paged grouped-MoE prefill requires grouped GEMM path2 support",
            ));
        }
        moe_prefill_prepare_routed_gate_up_input(
            gpu,
            ffn,
            &dtypes,
            &pbs.x_norm_batch,
            &pbs.x_rot_batch,
            dim,
            n,
        )?;
        if path2_eligible {
            // Stage 1 scatter pipeline. The scratch buffers are sized for
            // worst-case max_batch. Runtime launch bounds use the tighter live
            // slot upper bound below. Block size 16 (the WMMA tile row count).
            const BLOCK_M: usize = MOE_GROUPED_BLOCK_M;
            let grouped_scratch = pbs.grouped_moe_scratch.as_ref().expect("path2 scratch");
            let counts = &grouped_scratch.expert_token_counts;
            let offsets = &grouped_scratch.expert_offsets;
            let sorted = &grouped_scratch.sorted_slot_index;
            let inverse_perm = &grouped_scratch.inverse_perm;
            let tile_ids = &grouped_scratch.expert_tile_ids;
            let y_gu_grouped = &grouped_scratch.y_gate_up_grouped;
            if let Some(buckets) = routed_expert_buckets.as_ref() {
                // Load all active experts once before the per-bucket loops so the
                // down phase doesn't need a second round of page-ins.
                let active_experts: Vec<usize> =
                    buckets.iter().map(|b| b.expert as usize).collect();
                ensure_paged_experts_resident(gpu, pager, ffn, &active_experts)?;
                for bucket in buckets {
                    let expert = bucket.expert as usize;
                    let dtype = moe_expert_gate_up_dtype(ffn, expert).ok_or_else(|| {
                        HipError::new(
                            0,
                            &format!("missing gate_up dtype for routed expert {expert}"),
                        )
                    })?;
                    if !mixed_routed_quant_dtype_supported(dtype)
                        && !matches!(dtype, DType::F16 | DType::BF16)
                    {
                        return Err(HipError::new(
                            0,
                            &format!(
                                "bucketed grouped-MoE gate_up does not support expert {expert} dtype {dtype:?}"
                            ),
                        ));
                    }
                    let x_source = if matches!(dtype, DType::F16 | DType::BF16) {
                        &pbs.x_norm_batch
                    } else {
                        &pbs.x_rot_batch
                    };
                    upload_paged_moe_expert_bucket(gpu, bucket, sorted, inverse_perm, tile_ids)?;
                    hipfire_dispatch::pipeline::run_grouped_moe_gemm(
                        gpu,
                        dtype,
                        &ffn.expert_gate_up_ptrs,
                        tile_ids,
                        sorted,
                        x_source,
                        y_gu_grouped,
                        2 * mi,
                        gate_up_k,
                        path2_shape.gate_up_x_row_div,
                        bucket.m_total,
                        path2_shape.gate_up_source_rows,
                        false,
                        false,
                        // Resident routed experts sit in the oq4_arch combined
                        // layout unless oq_moe repacked them (indexed opt-in).
                        !hipfire_dispatch::families::moe::oq_indexed_decode_active(
                            config.dim, mi, k_top,
                        ),
                    )
                    .map_err(HipError::from)?;
                    gpu.moe_gate_up_unscatter_k8(
                        y_gu_grouped,
                        sorted,
                        gate_batch,
                        up_batch,
                        mi,
                        k_top,
                        bucket.m_total,
                    )?;
                }
            } else {
                // m_total upper bound — scratch is sized in PrefillBatchScratch::new
                // with the all-experts worst case, while this launch only needs slots
                // plus padding for experts that can be non-empty at this N.
                // The scatter fused kernel pre-fills every tile id in this aligned
                // bound with -1; grouped GEMM early-returns on sentinel tiles, so we
                // can skip the m_total dtoh sync entirely. Saves ~50µs/layer.
                let m_total_max = path2_shape.m_total_bound;

                // Fused scatter pipeline: one launch replaces histogram + offsets
                // + permute. Saves 2 launches × ~75µs × MoE layers.
                gpu.moe_scatter_fused_k8(
                    topk_indices,
                    counts,
                    offsets,
                    sorted,
                    tile_ids,
                    inverse_perm,
                    path2_shape.total_slots,
                    n_exp,
                    m_total_max,
                    BLOCK_M,
                )?;

                // Use m_total_max as the upper bound for grid sizing — the kernel
                // early-returns on expert_tile_ids[tile_y] == -1 for the
                // pre-sentinel'd unused-tile range.
                path2_m_total = m_total_max;
                let m_total = m_total_max;

                // Stage 2 grouped GEMM (gate_up). Writes Y_grouped[m_total × 2*mi] direct.
                // x_src = x_rot_batch [N × dim], x_row_div = K_TOP.
                // Per-dtype dispatch: experts uniform per layer (admit predicate
                // enforces). MQ4/MQ3/MQ6 route to their HFQ-layout grouped WMMA
                // sisters.
                match dtypes.expert_gate_up {
                    DType::MQ4G256 => gpu.gemm_hfq4g256_moe_grouped_wmma_k2(
                        &ffn.expert_gate_up_ptrs,
                        tile_ids,
                        sorted,
                        &pbs.x_rot_batch,
                        y_gu_grouped,
                        2 * mi,
                        gate_up_k,
                        path2_shape.gate_up_x_row_div,
                        m_total,
                        path2_shape.gate_up_source_rows,
                    )?,
                    // Opus OQ8 routed experts. The path-1 GEMV this replaces re-reads an
                    // expert's weights once per (token, expert) pair -- ~16x redundant at
                    // n=512 / k_top=8 / E=256, and 47.8% of MoE prefill in a kernel trace.
                    // The grouped path reads each expert once per tile instead.
                    // weight_byte_offset is 0: resident OQ8 experts point straight at
                    // interleaved 260 B [f32 scale | 256 int8] blocks (OQ8_BLK), exactly
                    // as the path-1 kernel addresses them.
                    DType::Oq8G256 => gpu.gemm_oq8g256_moe_grouped_wmma(
                        &ffn.expert_gate_up_ptrs,
                        tile_ids,
                        sorted,
                        &pbs.x_rot_batch,
                        y_gu_grouped,
                        2 * mi,
                        gate_up_k,
                        path2_shape.gate_up_x_row_div,
                        m_total,
                        path2_shape.gate_up_source_rows,
                        0,
                    )?,
                    DType::MQ6G256 => gpu.gemm_hfq6g256_moe_grouped_wmma(
                        &ffn.expert_gate_up_ptrs,
                        tile_ids,
                        sorted,
                        &pbs.x_rot_batch,
                        y_gu_grouped,
                        2 * mi,
                        gate_up_k,
                        path2_shape.gate_up_x_row_div,
                        m_total,
                        path2_shape.gate_up_source_rows,
                    )?,
                    DType::MQ3G256 => gpu.gemm_hfq3g256_moe_grouped_wmma(
                        &ffn.expert_gate_up_ptrs,
                        tile_ids,
                        sorted,
                        &pbs.x_rot_batch,
                        y_gu_grouped,
                        2 * mi,
                        gate_up_k,
                        path2_shape.gate_up_x_row_div,
                        m_total,
                        path2_shape.gate_up_source_rows,
                    )?,
                    DType::MQ2G256Lloyd => {
                        if mq2_lloyd_n32_gfx1151_enabled(&gpu.arch, path2_shape.total_slots) {
                            gpu.gemm_mq2g256_lloyd_moe_grouped_wmma_4w_k2_n32(
                                &ffn.expert_gate_up_ptrs,
                                tile_ids,
                                sorted,
                                &pbs.x_rot_batch,
                                y_gu_grouped,
                                2 * mi,
                                gate_up_k,
                                path2_shape.gate_up_x_row_div,
                                m_total,
                                path2_shape.gate_up_source_rows,
                            )?
                        } else {
                            gpu.gemm_mq2g256_lloyd_moe_grouped_wmma_k2(
                                &ffn.expert_gate_up_ptrs,
                                tile_ids,
                                sorted,
                                &pbs.x_rot_batch,
                                y_gu_grouped,
                                2 * mi,
                                gate_up_k,
                                path2_shape.gate_up_x_row_div,
                                m_total,
                                path2_shape.gate_up_source_rows,
                            )?
                        }
                    }
                    DType::F16 => gpu.gemm_f16_moe_grouped_wmma_gfx1151(
                        &ffn.expert_gate_up_ptrs,
                        tile_ids,
                        sorted,
                        &pbs.x_norm_batch,
                        y_gu_grouped,
                        2 * mi,
                        gate_up_k,
                        path2_shape.gate_up_x_row_div,
                        m_total,
                        path2_shape.gate_up_source_rows,
                    )?,
                    DType::BF16 => gpu.gemm_bf16_moe_grouped_wmma_gfx1151(
                        &ffn.expert_gate_up_ptrs,
                        tile_ids,
                        sorted,
                        &pbs.x_norm_batch,
                        y_gu_grouped,
                        2 * mi,
                        gate_up_k,
                        path2_shape.gate_up_x_row_div,
                        m_total,
                        path2_shape.gate_up_source_rows,
                    )?,
                    // Phase 4: Path 2 ParoQ4G128 grouped-WMMA. All 256 routed
                    // experts at this layer share one gate_up Givens rotation
                    // sidecar (ffn.paro_shared.gate_up_*); rotate x_norm into
                    // x_rot ONCE, then dispatch the HFQ4G128 grouped WMMA. The
                    // kernel auto-converts the F32 x_rot to F16 internally via
                    // ensure_fp16_x, same as the G256 sister.
                    //
                    // gfx1151 i8 MMQ opt-in (HIPFIRE_MOE_PARO_I8=1): routes to the
                    // HFQ4G128 i8 MMQ kernel which doubles compute throughput on
                    // Strix Halo (~140 vs ~71 TFLOPS). Compute-bound regime per
                    // Phase 4 attribution (gemm_paro_q4g128_moe_grouped_wmma_k2
                    // = 68.5% GPU time, 25.8 GiB/s — far from BW roof).
                    DType::ParoQ4G128 => {
                        let paro = ffn
                            .paro_shared
                            .as_ref()
                            .expect("ParoQ4G128 routed experts require paro_shared sidecars");
                        gpu.givens_rotate_to(
                            &pbs.x_norm_batch,
                            &pbs.x_rot_batch,
                            &paro.gate_up_pairs,
                            &paro.gate_up_theta,
                            &paro.gate_up_channel_scales,
                            n,
                            dim,
                            paro.krot as usize,
                        )?;
                        // Default-on for gfx1151 since 2026-05-21: i8 MMQ +6.3% over
                        // FP16 WMMA, k8 +2.5% over k2, both validated via PARO gen 100
                        // (clean decode, finite logits) + coherence-gate (MQ4 paths
                        // unchanged). Opt-out via HIPFIRE_MOE_PARO_I8=0 or _K8=0.
                        let use_paro_i8 = paro_moe_i8_enabled_for_arch_from_env(
                            gpu.arch.as_str(),
                            std::env::var("HIPFIRE_MOE_PARO_I8").ok().as_deref(),
                        );
                        let use_paro_i8_k8 = paro_moe_i8_k8_enabled_from_env(
                            use_paro_i8,
                            std::env::var("HIPFIRE_MOE_PARO_I8_K8").ok().as_deref(),
                        );
                        if use_paro_i8_k8 {
                            gpu.gemm_paro_q4g128_moe_grouped_mmq_k8_gfx1151(
                                &ffn.expert_gate_up_ptrs,
                                tile_ids,
                                sorted,
                                &pbs.x_rot_batch,
                                y_gu_grouped,
                                2 * mi,
                                gate_up_k,
                                path2_shape.gate_up_x_row_div,
                                m_total,
                                path2_shape.gate_up_source_rows,
                            )?;
                        } else if use_paro_i8 {
                            gpu.gemm_paro_q4g128_moe_grouped_mmq_gfx1151(
                                &ffn.expert_gate_up_ptrs,
                                tile_ids,
                                sorted,
                                &pbs.x_rot_batch,
                                y_gu_grouped,
                                2 * mi,
                                gate_up_k,
                                path2_shape.gate_up_x_row_div,
                                m_total,
                                path2_shape.gate_up_source_rows,
                            )?;
                        } else {
                            gpu.gemm_paro_q4g128_moe_grouped_wmma_k2(
                                &ffn.expert_gate_up_ptrs,
                                tile_ids,
                                sorted,
                                &pbs.x_rot_batch,
                                y_gu_grouped,
                                2 * mi,
                                gate_up_k,
                                path2_shape.gate_up_x_row_div,
                                m_total,
                                path2_shape.gate_up_source_rows,
                            )?;
                        }
                    }
                    other => panic!(
                    "prefill_moe_ffn_body_batched: unsupported experts[0].gate_up dtype {other:?} \
                             — admit predicate should have rejected this layer"
                ),
                }

                // Stage 3 unscatter combine. Fans Y_grouped → gate_batch + up_batch.
                gpu.moe_gate_up_unscatter_k8(
                    y_gu_grouped,
                    sorted,
                    gate_batch,
                    up_batch,
                    mi,
                    k_top,
                    m_total,
                )?;
            }
        } else {
            // Path 1 fallback (CDNA/gfx10): per-token indexed GEMV, batched
            // over the N tokens via grid.z. The dispatch is dtype-keyed because
            // the kernel reads the weight nibble layout directly (HFQ4G256:
            // 136 B/group; HFQ4G128/PARO: 72 B/group).
            match dtypes.expert_gate_up {
                DType::MQ4G256 => gpu.gemv_hfq4g256_moe_gate_up_k8_indexed_batched(
                    &ffn.expert_gate_up_ptrs,
                    topk_indices,
                    &pbs.x_rot_batch,
                    gate_batch,
                    up_batch,
                    2 * mi,
                    gate_up_k,
                    k_top,
                    n,
                )?,
                DType::MQ6G256 => gpu.gemv_hfq6g256_moe_gate_up_k8_indexed_batched(
                    &ffn.expert_gate_up_ptrs,
                    topk_indices,
                    &pbs.x_rot_batch,
                    gate_batch,
                    up_batch,
                    2 * mi,
                    gate_up_k,
                    k_top,
                    n,
                )?,
                DType::MQ2G256 => gpu.gemv_hfq2g256_moe_gate_up_k8_indexed_batched(
                    &ffn.expert_gate_up_ptrs,
                    topk_indices,
                    &pbs.x_rot_batch,
                    gate_batch,
                    up_batch,
                    2 * mi,
                    gate_up_k,
                    k_top,
                    n,
                )?,
                DType::MQ8G256 => gpu.gemv_hfq8g256_moe_gate_up_k8_indexed_batched(
                    &ffn.expert_gate_up_ptrs,
                    topk_indices,
                    &pbs.x_rot_batch,
                    gate_batch,
                    up_batch,
                    2 * mi,
                    gate_up_k,
                    k_top,
                    n,
                )?,
                DType::MQ2G256Lloyd => gpu.gemv_mq2g256_lloyd_moe_gate_up_k8_indexed_batched(
                    &ffn.expert_gate_up_ptrs,
                    topk_indices,
                    &pbs.x_rot_batch,
                    gate_batch,
                    up_batch,
                    2 * mi,
                    gate_up_k,
                    k_top,
                    n,
                )?,
                DType::MQ3G256Lloyd => gpu.gemv_mq3g256_lloyd_moe_gate_up_k8_indexed_batched(
                    &ffn.expert_gate_up_ptrs,
                    topk_indices,
                    &pbs.x_rot_batch,
                    gate_batch,
                    up_batch,
                    2 * mi,
                    gate_up_k,
                    k_top,
                    n,
                )?,
                // Phase 3 PARO routed-expert: apply the layer's shared gate_up
                // Givens rotation to x_norm_batch into x_rot_batch ONCE, then
                // dispatch the HFQ4G128 indexed batched kernel. All 256 experts
                // at this layer share the same gate_up rotation sidecar
                // (ffn.paro_shared, populated by paro_load_moe_shared_sidecars).
                DType::ParoQ4G128 => {
                    let paro = ffn
                        .paro_shared
                        .as_ref()
                        .expect("ParoQ4G128 routed experts require paro_shared sidecars");
                    gpu.givens_rotate_to(
                        &pbs.x_norm_batch,
                        &pbs.x_rot_batch,
                        &paro.gate_up_pairs,
                        &paro.gate_up_theta,
                        &paro.gate_up_channel_scales,
                        n,
                        dim,
                        paro.krot as usize,
                    )?;
                    gpu.gemv_paro_q4g128_moe_gate_up_k8_indexed_batched(
                        &ffn.expert_gate_up_ptrs,
                        topk_indices,
                        &pbs.x_rot_batch,
                        gate_batch,
                        up_batch,
                        2 * mi,
                        gate_up_k,
                        k_top,
                        n,
                    )?;
                }
                // Opus Quant routed experts (indexed Path 1; no grouped-WMMA
                // kernel). These read x PER SLOT, not the shared
                // `x_rot_batch`: routed experts carry DIFFERENT AWQ scales
                // (each sees a different token subset, hence a different
                // imatrix), and the divide must precede the FWHT, so one
                // rotation cannot serve them all. Expand x_norm_batch into
                // [N × K_TOP × dim], each slot divided by its own expert's
                // scale. See `rotate_x_mq_awq_indexed_batched`; with no
                // sidecars (paged mode included) this is the plain rotation
                // replicated per slot, which the kernels still require.
                dt @ (DType::Oq4G256 | DType::Oq8G256) => {
                    let x_rot_slots = pbs.moe_x_rot_expanded_batch.as_ref().expect("moe scratch");
                    gpu.rotate_x_mq_awq_indexed_batched(
                        &pbs.x_norm_batch,
                        ffn.expert_gate_up_awq_ptrs.as_ref(),
                        topk_indices,
                        x_rot_slots,
                        gate_up_k,
                        k_top,
                        n,
                    )?;
                    if dt == DType::Oq4G256 {
                        gpu.gemv_oq4g256_moe_gate_up_k8_indexed_batched(
                            &ffn.expert_gate_up_ptrs,
                            topk_indices,
                            x_rot_slots,
                            gate_batch,
                            up_batch,
                            2 * mi,
                            gate_up_k,
                            k_top,
                            n,
                            true, // x_rot_slots is [N x K_TOP x dim]
                        )?;
                    } else {
                        gpu.gemv_oq8g256_moe_gate_up_k8_indexed_batched(
                            &ffn.expert_gate_up_ptrs,
                            topk_indices,
                            x_rot_slots,
                            gate_batch,
                            up_batch,
                            2 * mi,
                            gate_up_k,
                            k_top,
                            n,
                            true, // x_rot_slots is [N x K_TOP x dim]
                        )?;
                    }
                }
                other => panic!(
                    "prefill_moe_ffn_body_batched: Path 1 fallback unsupported \
                             experts[0].gate_up dtype {other:?} — admit predicate should \
                             have rejected this layer"
                ),
            }
        }

        // SwiGLU + FWHT over [N*K_TOP × mi] — batch flatten across tokens and
        // expert ranks, k=mi is per-row width.
        // F2: AWQ-aware silu_mul+rotate; experts[0].down is representative (all
        // experts at this layer share imatrix at the same residual basis).
        // PARO branch (Phase 3): the layer-shared `down` rotation sidecar lives
        // on ffn.paro_shared (not per-expert; all 256 experts alias the same
        // tuple). Apply via fused_silu_mul_givens_rotate_f32 over the flattened
        // [n*k_top × mi] grid.
        if paro_mode {
            let paro = ffn
                .paro_shared
                .as_ref()
                .expect("ParoQ4G128 routed experts require paro_shared sidecars");
            gpu.fused_silu_mul_givens_rotate_f32(
                gate_batch,
                up_batch,
                rot_batch,
                &paro.down_pairs,
                &paro.down_theta,
                &paro.down_channel_scales,
                n * k_top,
                mi,
                paro.krot as usize,
            )?;
        } else if dtypes.routed_profile.is_mixed() {
            // Raw fallback experts and low-bit siblings share gate/up results,
            // but their down projections consume different activation bases.
            // Materialize the unrotated SwiGLU once, then derive the quantized
            // basis into the existing rotation buffer.
            gpu.silu_mul_f32(gate_batch, up_batch, hidden_batch)?;
            if ffn.experts.is_empty() {
                gpu.rotate_x_mq_batched(hidden_batch, rot_batch, mi, n * k_top)?;
            } else {
                let representative = ffn
                    .experts
                    .iter()
                    .find(|expert| expert.down.gpu_dtype == dtypes.expert_down)
                    .ok_or_else(|| {
                        HipError::new(0, "mixed MoE layer has no quantized down representative")
                    })?;
                rotate_x_mq_batched_for(
                    gpu,
                    &representative.down,
                    hidden_batch,
                    rot_batch,
                    mi,
                    n * k_top,
                )?;
            }
        } else if matches!(dtypes.expert_down, DType::F16 | DType::BF16) {
            gpu.silu_mul_f32(gate_batch, up_batch, rot_batch)?;
        } else if ffn.experts.is_empty() {
            gpu.fused_silu_mul_rotate_mq_batched(gate_batch, up_batch, rot_batch, mi, n * k_top)?;
        } else if let Some(down_awq) = ffn.expert_down_awq_ptrs.as_ref() {
            // Per-expert AWQ on the down input. `experts[0].down` is NOT
            // representative: routed experts carry different scales for the
            // same reason they do on the gate_up side. `rot_batch` is already
            // one row per (token, krank), so only the scale lookup changes —
            // select it on device from topk_indices.
            gpu.fused_silu_mul_rotate_mq_awq_indexed(
                gate_batch,
                up_batch,
                Some(down_awq),
                topk_indices,
                rot_batch,
                mi,
                n * k_top,
            )?;
        } else {
            fused_silu_mul_rotate_mq_batched_for(
                gpu,
                &ffn.experts[0].down,
                gate_batch,
                up_batch,
                rot_batch,
                mi,
                n * k_top,
            )?;
        }

        // Down projection. Three paths:
        //   Path 2 (HIPFIRE_MOE_GROUPED_GEMM=1, RDNA): grouped-WMMA-GEMM
        //     reusing the gate_up scatter + inverse_perm + a non-atomic combine.
        //   Path 1 (RDNA, default): atomic-free expanded GEMV write + combine.
        //   Path 0 (CDNA wave64 fallback): residual_scaled atomic GEMV.
        //
        // Path 1: K_TOP-way atomicAdd contention per output cell — 387 GiB/s
        // observed vs 954 on the sister gate_up. Path 2 amortizes weights via
        // WMMA across the m_total tokens routed to each expert; ~67ms saved on
        // the down kernel for A3B prefill at batch 256 (R9700).
        // CDNA (wave64, HBM2/3) stays on Path 0 — cheap HBM atomics +
        // expanded scratch cost makes the GEMV pattern competitive.
        if path2_eligible {
            let grouped_scratch = pbs.grouped_moe_scratch.as_ref().expect("path2 scratch");
            let y_down_grouped = &grouped_scratch.y_down_grouped;
            let inverse_perm = &grouped_scratch.inverse_perm;
            let sorted = &grouped_scratch.sorted_slot_index;
            let tile_ids = &grouped_scratch.expert_tile_ids;
            if let Some(buckets) = routed_expert_buckets.as_ref() {
                let routed_target = routed_out.unwrap_or(&pbs.x_batch);
                for bucket in buckets {
                    let expert = bucket.expert as usize;
                    let dtype = moe_expert_down_dtype(ffn, expert).ok_or_else(|| {
                        HipError::new(0, &format!("missing down dtype for routed expert {expert}"))
                    })?;
                    if !mixed_routed_quant_dtype_supported(dtype)
                        && !matches!(dtype, DType::F16 | DType::BF16)
                    {
                        return Err(HipError::new(
                            0,
                            &format!(
                                "bucketed grouped-MoE down does not support expert {expert} dtype {dtype:?}"
                            ),
                        ));
                    }
                    let down_source = if matches!(dtype, DType::F16 | DType::BF16) {
                        hidden_batch
                    } else {
                        rot_batch
                    };
                    upload_paged_moe_expert_bucket(gpu, bucket, sorted, inverse_perm, tile_ids)?;
                    hipfire_dispatch::pipeline::run_grouped_moe_gemm(
                        gpu,
                        dtype,
                        &ffn.expert_down_ptrs,
                        tile_ids,
                        sorted,
                        down_source,
                        y_down_grouped,
                        down_m,
                        down_k,
                        path2_shape.down_x_row_div,
                        bucket.m_total,
                        path2_shape.down_source_rows,
                        false,
                        false,
                        // Resident routed experts sit in the oq4_arch combined
                        // layout unless oq_moe repacked them (indexed opt-in).
                        !hipfire_dispatch::families::moe::oq_indexed_decode_active(
                            config.dim, mi, k_top,
                        ),
                    )
                    .map_err(HipError::from)?;
                    gpu.moe_down_combine_grouped_k8(
                        y_down_grouped,
                        inverse_perm,
                        topk_weights,
                        routed_target,
                        down_m,
                        k_top,
                        n,
                    )?;
                }
            } else {
                // m_total already computed during gate_up scatter — reuse to skip
                // a second dtoh sync per MoE layer (~50µs each × 40 layers = 2ms).
                let m_total = path2_m_total;

                // Grouped GEMM on down: x_src = rot_batch [N*K_TOP × mi], x_row_div = 1
                // (sorted_slot_index[slot] directly indexes the source row).
                // Per-dtype dispatch: experts uniform per layer. MQ4 → HFQ4-layout;
                // MQ6 → HFQ6 sister (shipped via feat/hfq6-moe-grouped-wmma).
                match dtypes.expert_down {
                    DType::MQ4G256 => gpu.gemm_hfq4g256_moe_grouped_wmma_k2(
                        &ffn.expert_down_ptrs,
                        tile_ids,
                        sorted,
                        rot_batch,
                        y_down_grouped,
                        down_m,
                        down_k,
                        path2_shape.down_x_row_div,
                        m_total,
                        path2_shape.down_source_rows,
                    )?,
                    // Opus OQ8 routed down. See the gate_up arm for why the grouped
                    // path matters here; offset 0 for the same reason.
                    DType::Oq8G256 => gpu.gemm_oq8g256_moe_grouped_wmma(
                        &ffn.expert_down_ptrs,
                        tile_ids,
                        sorted,
                        rot_batch,
                        y_down_grouped,
                        down_m,
                        down_k,
                        path2_shape.down_x_row_div,
                        m_total,
                        path2_shape.down_source_rows,
                        0,
                    )?,
                    DType::MQ6G256 => gpu.gemm_hfq6g256_moe_grouped_wmma(
                        &ffn.expert_down_ptrs,
                        tile_ids,
                        sorted,
                        rot_batch,
                        y_down_grouped,
                        down_m,
                        down_k,
                        path2_shape.down_x_row_div,
                        m_total,
                        path2_shape.down_source_rows,
                    )?,
                    DType::MQ3G256 => gpu.gemm_hfq3g256_moe_grouped_wmma(
                        &ffn.expert_down_ptrs,
                        tile_ids,
                        sorted,
                        rot_batch,
                        y_down_grouped,
                        down_m,
                        down_k,
                        path2_shape.down_x_row_div,
                        m_total,
                        path2_shape.down_source_rows,
                    )?,
                    DType::MQ2G256Lloyd => {
                        if mq2_lloyd_n32_gfx1151_enabled(&gpu.arch, path2_shape.total_slots) {
                            gpu.gemm_mq2g256_lloyd_moe_grouped_wmma_4w_k2_n32(
                                &ffn.expert_down_ptrs,
                                tile_ids,
                                sorted,
                                rot_batch,
                                y_down_grouped,
                                down_m,
                                down_k,
                                path2_shape.down_x_row_div,
                                m_total,
                                path2_shape.down_source_rows,
                            )?
                        } else {
                            gpu.gemm_mq2g256_lloyd_moe_grouped_wmma_k2(
                                &ffn.expert_down_ptrs,
                                tile_ids,
                                sorted,
                                rot_batch,
                                y_down_grouped,
                                down_m,
                                down_k,
                                path2_shape.down_x_row_div,
                                m_total,
                                path2_shape.down_source_rows,
                            )?
                        }
                    }
                    DType::F16 => gpu.gemm_f16_moe_grouped_wmma_gfx1151(
                        &ffn.expert_down_ptrs,
                        tile_ids,
                        sorted,
                        rot_batch,
                        y_down_grouped,
                        down_m,
                        down_k,
                        path2_shape.down_x_row_div,
                        m_total,
                        path2_shape.down_source_rows,
                    )?,
                    DType::BF16 => gpu.gemm_bf16_moe_grouped_wmma_gfx1151(
                        &ffn.expert_down_ptrs,
                        tile_ids,
                        sorted,
                        rot_batch,
                        y_down_grouped,
                        down_m,
                        down_k,
                        path2_shape.down_x_row_div,
                        m_total,
                        path2_shape.down_source_rows,
                    )?,
                    // Phase 4: Path 2 ParoQ4G128 down grouped-WMMA (with i8 MMQ
                    // opt-in for gfx1151 — see gate_up arm above). rot_batch was
                    // already Givens-rotated by paro_shared.down_* via the PARO
                    // fused_silu_mul_givens_rotate_f32 step above; the kernel is
                    // rotation-agnostic. Same kernel for gate_up + down — only
                    // shape parameters and x_row_div differ.
                    DType::ParoQ4G128 => {
                        // Default-on for gfx1151 since 2026-05-21: i8 MMQ +6.3% over
                        // FP16 WMMA, k8 +2.5% over k2, both validated via PARO gen 100
                        // (clean decode, finite logits) + coherence-gate (MQ4 paths
                        // unchanged). Opt-out via HIPFIRE_MOE_PARO_I8=0 or _K8=0.
                        let use_paro_i8 = paro_moe_i8_enabled_for_arch_from_env(
                            gpu.arch.as_str(),
                            std::env::var("HIPFIRE_MOE_PARO_I8").ok().as_deref(),
                        );
                        let use_paro_i8_k8 = paro_moe_i8_k8_enabled_from_env(
                            use_paro_i8,
                            std::env::var("HIPFIRE_MOE_PARO_I8_K8").ok().as_deref(),
                        );
                        if use_paro_i8_k8 {
                            gpu.gemm_paro_q4g128_moe_grouped_mmq_k8_gfx1151(
                                &ffn.expert_down_ptrs,
                                tile_ids,
                                sorted,
                                rot_batch,
                                y_down_grouped,
                                down_m,
                                down_k,
                                path2_shape.down_x_row_div,
                                m_total,
                                path2_shape.down_source_rows,
                            )?;
                        } else if use_paro_i8 {
                            gpu.gemm_paro_q4g128_moe_grouped_mmq_gfx1151(
                                &ffn.expert_down_ptrs,
                                tile_ids,
                                sorted,
                                rot_batch,
                                y_down_grouped,
                                down_m,
                                down_k,
                                path2_shape.down_x_row_div,
                                m_total,
                                path2_shape.down_source_rows,
                            )?;
                        } else {
                            gpu.gemm_paro_q4g128_moe_grouped_wmma_k2(
                                &ffn.expert_down_ptrs,
                                tile_ids,
                                sorted,
                                rot_batch,
                                y_down_grouped,
                                down_m,
                                down_k,
                                path2_shape.down_x_row_div,
                                m_total,
                                path2_shape.down_source_rows,
                            )?;
                        }
                    }
                    other => panic!(
                    "prefill_moe_ffn_body_batched: unsupported experts[0].down dtype {other:?} \
                             — admit predicate should have rejected this layer"
                ),
                }
                // Non-atomic combine via inverse_perm + topk_weights.
                gpu.moe_down_combine_grouped_k8(
                    y_down_grouped,
                    inverse_perm,
                    topk_weights,
                    &pbs.x_batch,
                    down_m,
                    k_top,
                    n,
                )?;
            }
        } else {
            let use_atomic_free_down = !gpu.arch.starts_with("gfx9");
            if use_atomic_free_down {
                // Path 1 expanded-down: per-token-per-rank GEMV writes to a
                // [N × K_TOP × M] scratch, then a separate combine kernel folds
                // it back into pbs.x_batch with topk weights. The expanded
                // kernel is dtype-keyed; the combine kernel is dtype-agnostic.
                match dtypes.expert_down {
                    DType::MQ4G256 => gpu.gemv_hfq4g256_moe_down_k8_indexed_batched_expanded(
                        &ffn.expert_down_ptrs,
                        topk_indices,
                        rot_batch,
                        down_expanded,
                        down_m,
                        down_k,
                        k_top,
                        n,
                    )?,
                    DType::MQ6G256 => gpu.gemv_hfq6g256_moe_down_k8_indexed_batched_expanded(
                        &ffn.expert_down_ptrs,
                        topk_indices,
                        rot_batch,
                        down_expanded,
                        down_m,
                        down_k,
                        k_top,
                        n,
                    )?,
                    DType::MQ2G256 => gpu.gemv_hfq2g256_moe_down_k8_indexed_batched_expanded(
                        &ffn.expert_down_ptrs,
                        topk_indices,
                        rot_batch,
                        down_expanded,
                        down_m,
                        down_k,
                        k_top,
                        n,
                    )?,
                    DType::MQ8G256 => gpu.gemv_hfq8g256_moe_down_k8_indexed_batched_expanded(
                        &ffn.expert_down_ptrs,
                        topk_indices,
                        rot_batch,
                        down_expanded,
                        down_m,
                        down_k,
                        k_top,
                        n,
                    )?,
                    DType::MQ2G256Lloyd => gpu
                        .gemv_mq2g256_lloyd_moe_down_k8_indexed_batched_expanded(
                            &ffn.expert_down_ptrs,
                            topk_indices,
                            rot_batch,
                            down_expanded,
                            down_m,
                            down_k,
                            k_top,
                            n,
                        )?,
                    DType::MQ3G256Lloyd => gpu
                        .gemv_mq3g256_lloyd_moe_down_k8_indexed_batched_expanded(
                            &ffn.expert_down_ptrs,
                            topk_indices,
                            rot_batch,
                            down_expanded,
                            down_m,
                            down_k,
                            k_top,
                            n,
                        )?,
                    // Phase 3 PARO down: the layer-shared `down` Givens rotation
                    // has already been applied to rot_batch by the
                    // fused_silu_mul_givens_rotate_f32 call above. The HFQ4G128
                    // indexed kernel (existing, shipped in 7c00970d) is
                    // rotation-agnostic; same dispatch shape as G256 sister.
                    DType::ParoQ4G128 => gpu.gemv_paro_q4g128_moe_down_k8_indexed_batched(
                        &ffn.expert_down_ptrs,
                        topk_indices,
                        rot_batch,
                        down_expanded,
                        down_m,
                        down_k,
                        k_top,
                        n,
                    )?,
                    DType::Oq4G256 => gpu.gemv_oq4g256_moe_down_k8_indexed_batched_expanded(
                        &ffn.expert_down_ptrs,
                        topk_indices,
                        rot_batch,
                        down_expanded,
                        down_m,
                        down_k,
                        k_top,
                        n,
                    )?,
                    DType::Oq8G256 => gpu.gemv_oq8g256_moe_down_k8_indexed_batched_expanded(
                        &ffn.expert_down_ptrs,
                        topk_indices,
                        rot_batch,
                        down_expanded,
                        down_m,
                        down_k,
                        k_top,
                        n,
                    )?,
                    other => panic!(
                        "prefill_moe_ffn_body_batched: Path 1 fallback unsupported \
                                 experts[0].down dtype {other:?} — admit predicate should \
                                 have rejected this layer"
                    ),
                }
                gpu.moe_down_combine_k8_batched(
                    down_expanded,
                    topk_weights,
                    &pbs.x_batch,
                    down_m,
                    k_top,
                    n,
                )?;
            } else {
                gpu.gemv_hfq4g256_moe_down_residual_scaled_k8_indexed_batched(
                    &ffn.expert_down_ptrs,
                    topk_indices,
                    topk_weights,
                    rot_batch,
                    &pbs.x_batch,
                    down_m,
                    down_k,
                    k_top,
                    n,
                )?;
            }
        }
        return Ok(());
    }
    // ── 6. Routed experts: delegated to MoeFamily::run_prefill (Ship 4.2) ──
    let down_m = ffn.experts[0].down.m;
    let down_k = ffn.experts[0].down.k;
    let gate_up_k = ffn.experts[0].gate_up.k;
    let total_slots = n * k_top;
    let m_total_max = moe_grouped_m_total_bound(total_slots, n_exp);

    // Same correction as the decode path: read the routed dtypes from the
    // per-expert dtype tables rather than `ffn.experts`, which paged residency
    // leaves empty. Here the old form was worse than a wrong default — a bare
    // `ffn.experts[0]` index panics outright on a paged model.
    let routed_gate_up_dtype = moe_expert_gate_up_dtype(ffn, 0).ok_or_else(|| {
        HipError::new(
            0,
            "moe prefill: routed gate_up dtype unknown (no per-expert dtype table \
             and no resident experts) — cannot resolve a routed dispatch path",
        )
    })?;
    let routed_down_dtype = moe_expert_down_dtype(ffn, 0).ok_or_else(|| {
        HipError::new(
            0,
            "moe prefill: routed down dtype unknown (no per-expert dtype table \
             and no resident experts) — cannot resolve a routed dispatch path",
        )
    })?;
    let moe_dtypes = hipfire_dispatch::families::moe::MoeDtypes {
        router: ffn.router.gpu_dtype,
        shared_gate: ffn.shared_expert_gate.gpu_dtype,
        shared_expert_gate: ffn.shared_expert.gate.gpu_dtype,
        shared_expert_up: ffn.shared_expert.up.gpu_dtype,
        // Empty `.all()` is vacuously true — see the decode-side note.
        experts_all_gate_up_mq4: if !ffn.expert_gate_up_dtypes.is_empty() {
            ffn.expert_gate_up_dtypes
                .iter()
                .all(|d| *d == DType::MQ4G256)
        } else if !ffn.experts.is_empty() {
            ffn.experts
                .iter()
                .all(|e| e.gate_up.gpu_dtype == DType::MQ4G256)
        } else {
            false
        },
        routed_gate_up: routed_gate_up_dtype,
        routed_down: routed_down_dtype,
        has_paro_shared: ffn.paro_shared.is_some(),
    };

    let paro_gate_up =
        ffn.paro_shared
            .as_ref()
            .map(|paro| hipfire_dispatch::families::gemv::GivensRef {
                pairs: &paro.gate_up_pairs,
                theta: &paro.gate_up_theta,
                scales: &paro.gate_up_channel_scales,
                krot: paro.krot as usize,
            });
    let paro_down =
        ffn.paro_shared
            .as_ref()
            .map(|paro| hipfire_dispatch::families::gemv::GivensRef {
                pairs: &paro.down_pairs,
                theta: &paro.down_theta,
                scales: &paro.down_channel_scales,
                krot: paro.krot as usize,
            });
    let down_awq_scale = ffn.experts[0].down.awq_scale.as_ref();
    let grouped_scratch = pbs.grouped_moe_scratch.as_ref().expect("moe scratch");

    let moe_prefill_params = hipfire_dispatch::families::moe::MoePrefillParams {
        layer: layer_idx,
        capture,
        dtypes: moe_dtypes,
        batch_size: n,
        mi,
        down_m,
        down_k,
        gate_up_k,
        k_top,
        n_exp,
        m_total_max,
        topk_indices,
        topk_weights,
        x_batch: &pbs.x_batch,
        x_norm_batch: &pbs.x_norm_batch,
        x_rot_batch: &pbs.x_rot_batch,
        expert_gate_up_ptrs: &ffn.expert_gate_up_ptrs,
        expert_down_ptrs: &ffn.expert_down_ptrs,
        expert_gate_up_awq_ptrs: ffn.expert_gate_up_awq_ptrs.as_ref(),
        expert_down_awq_ptrs: ffn.expert_down_awq_ptrs.as_ref(),
        routed_oq_arch_combined: !hipfire_dispatch::families::moe::oq_indexed_decode_active(
            config.dim, mi, k_top,
        ),
        gate_batch,
        up_batch,
        rot_batch,
        down_expanded,
        x_rot_expanded: pbs.moe_x_rot_expanded_batch.as_ref().expect("moe scratch"),
        expert_token_counts: &grouped_scratch.expert_token_counts,
        expert_offsets: &grouped_scratch.expert_offsets,
        sorted_slot_index: &grouped_scratch.sorted_slot_index,
        expert_tile_ids: &grouped_scratch.expert_tile_ids,
        inverse_perm: &grouped_scratch.inverse_perm,
        y_gate_up_grouped: &grouped_scratch.y_gate_up_grouped,
        y_down_grouped: &grouped_scratch.y_down_grouped,
        paro_gate_up,
        paro_down,
        down_awq_scale,
        routed_out,
    };
    hipfire_runtime::dispatch::moe_family()
        .run_prefill(ctx, gpu, &moe_prefill_params)
        .map_err(HipError::from)?;

    Ok(())
}

/// Band view for `forward_prefill_chunk`. `None` (the default) means the
/// chunk processes the whole stack: embedding → all layers → final norm
/// + lm_head. `Some(b)` restricts the chunk to layers `b.layer_start..
/// b.layer_end`, skips the embedding when `!b.is_first_band` (input is
/// already in `pbs.x_batch` from a prior peer-copy), and skips the final
/// norm + lm_head when `!b.is_last_band` (output activation stays in
/// `pbs.x_batch` for the next band's peer-copy).
///
/// Counter offsets seed the running per-LA / per-KV / per-FA counters so
/// the band's first DeltaNet/FullAttn layer indexes the correct
/// `dn_state.s_matrices[i]` / `kv_cache.k_caches[i]` slot.
pub(crate) struct PrefillBandCtx<'a> {
    pub layer_start: usize,
    pub layer_end: usize,
    pub delta_layer_offset: usize,
    pub fa_layer_offset: usize,
    pub is_first_band: bool,
    pub is_last_band: bool,
    /// Per-device asym{2,3,4} givens replicas. When `Some`, the chunk's
    /// FA-layer batched KV writers use these instead of `kv_cache.givens_*`
    /// (which is `None` in multi-GPU mode by design — each device needs its
    /// own copy of the rotation tables).
    pub givens_cos: Option<&'a GpuTensor>,
    pub givens_sin: Option<&'a GpuTensor>,
}

#[allow(clippy::too_many_arguments)]
/// Debug localization hook (no-op unless `HIPFIRE_DUMP_HIDDEN` is set to a file
/// prefix). Appends the post-layer hidden row for the target absolute position
/// to `{HIPFIRE_DUMP_HIDDEN}.{tag}` as `u32 layer_idx` followed by `dim`
/// little-endian f32. The target absolute position is `HIPFIRE_DUMP_HIDDEN_POS`
/// (default 0); `abs_pos_of_row0` is the absolute sequence position of row 0 of
/// `x` (`start_pos` for the batched residual `pbs.x_batch`, `pos` for the
/// single-row per-token `s.x`). Used to localize the PARO batched-prefill
/// divergence by diffing `.batched` vs `.pertoken` per layer. Requires
/// `HIPFIRE_GRAPH=0` (does a synchronous D2H readback, which is illegal under
/// graph capture).
pub(crate) fn dump_hidden_localize(
    gpu: &Gpu,
    x: &GpuTensor,
    n_rows: usize,
    abs_pos_of_row0: usize,
    dim: usize,
    layer_idx: usize,
    tag: &str,
) {
    let prefix = match std::env::var("HIPFIRE_DUMP_HIDDEN") {
        Ok(p) => p,
        Err(_) => return,
    };
    use std::io::Write;
    let path = format!("{prefix}.{tag}");
    // Activation-capture mode (HIPFIRE_DUMP_HIDDEN_ALL=1): dump EVERY row of `x`
    // as raw [dim] f32 each (no per-row header) — so one prefill yields n_rows
    // real-activation samples for an offline rotation/quant study, AND a per-token
    // decode appends its single row each call → the file accumulates the whole
    // sequence. This mode IGNORES the single-position POS gate (which only makes
    // sense for the localize path below); it must fire at every position.
    if std::env::var("HIPFIRE_DUMP_HIDDEN_ALL").as_deref() == Ok("1") {
        // Two sub-modes:
        //  - default: restrict to one target layer (HIPFIRE_DUMP_HIDDEN_LAYER,
        //    default 0), one file `{prefix}.{tag}` (the kv-compression study path).
        //  - HIPFIRE_DUMP_HIDDEN_ALLLAYERS=1: capture EVERY layer to per-layer
        //    files `{prefix}.{tag}.L{layer_idx}` — the Phase-A block-local
        //    recovery capture (residual-stream in/mid/out for all blocks).
        let all_layers = std::env::var("HIPFIRE_DUMP_HIDDEN_ALLLAYERS").as_deref() == Ok("1");
        let layer_path = if all_layers {
            format!("{path}.L{layer_idx}")
        } else {
            let want_layer: usize = std::env::var("HIPFIRE_DUMP_HIDDEN_LAYER")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0);
            if layer_idx != want_layer {
                return;
            }
            path.clone()
        };
        if gpu.hip.device_synchronize().is_err() {
            return;
        }
        let all = match gpu.download_f32(x) {
            Ok(v) => v,
            Err(_) => return,
        };
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&layer_path)
        {
            let take = (n_rows * dim).min(all.len());
            let bytes: Vec<u8> = all[..take].iter().flat_map(|v| v.to_le_bytes()).collect();
            let _ = f.write_all(&bytes);
        }
        return;
    }
    // Single-position localize path (PARO batched-vs-pertoken diff).
    let target: usize = std::env::var("HIPFIRE_DUMP_HIDDEN_POS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    if target < abs_pos_of_row0 {
        return;
    }
    let row = target - abs_pos_of_row0;
    if row >= n_rows {
        return;
    }
    if gpu.hip.device_synchronize().is_err() {
        return;
    }
    let all = match gpu.download_f32(x) {
        Ok(v) => v,
        Err(_) => return,
    };
    let off = row * dim;
    if off + dim > all.len() {
        return;
    }
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
    {
        let _ = f.write_all(&(layer_idx as u32).to_le_bytes());
        let mut bytes = Vec::with_capacity(dim * 4);
        for v in &all[off..off + dim] {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let _ = f.write_all(&bytes);
    }
}

pub(crate) fn forward_prefill_chunk(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    tokens: &[u32],
    start_pos: usize,
    kv_cache: &mut kv::KvCache,
    dn_state: &mut DeltaNetState,
    s: &Qwen35Scratch,
    pbs: &PrefillBatchScratch,
    hidden_rb: Option<&HiddenStateRingBuffer>,
    per_token_hidden_out: Option<(&GpuTensor, usize)>,
    mut gdn_tape: Option<&mut crate::speculative::GdnTape>,
    tape_offset: usize,
    tree_verify: Option<TreeVerifyCtx<'_>>,
    pre_uploaded: bool,
    band: Option<&PrefillBandCtx<'_>>,
    mask_override: Option<MaskEmbedOverride<'_>>,
    positions_override: Option<&[usize]>,
    needs_last_token_logits: bool,
    max_layer: Option<usize>,
    force_q8_gdn_per_token: bool,
    // EP (Ship 6 substrate-EP prefill): per-MoE-layer routed partial. ONLY set
    // by the EP driver, which calls this with a SINGLE-layer band so the routed
    // combine of that one MoE layer lands in the zeroed partial (all-reduced by
    // the driver after the call). Always `None` for multi-layer bands (PP /
    // single-GPU full stack) — a shared partial across >1 MoE layer would be wrong.
    routed_out: Option<&GpuTensor>,
) -> HipResult<()> {
    let n = tokens.len();
    debug_assert!(n > 0);
    debug_assert!(n <= pbs.max_batch);
    debug_assert!(
        routed_out.is_none()
            || band
                .map(|b| b.layer_end - b.layer_start <= 1)
                .unwrap_or(false),
        "forward_prefill_chunk: routed_out requires a single-layer band (EP driver invariant)",
    );

    let dim = config.dim;
    let k_dim = config.linear_num_key_heads * config.linear_key_head_dim;
    let v_dim = config.linear_num_value_heads * config.linear_value_head_dim;
    let n_v_heads = config.linear_num_value_heads;
    let hd = config.linear_key_head_dim;
    let dim_row_bytes = dim * 4;
    let do_embed = band.map(|b| b.is_first_band).unwrap_or(true);
    let layer_start = band.map(|b| b.layer_start).unwrap_or(0);
    // `max_layer = Some(N)` early-exits at layer N (exclusive). pflash uses
    // this with N = score_layer_idx + 1: the drafter forward only needs to
    // populate the K cache through the scoring layer (the shallowest
    // FullAttention layer, typically layer 3 of 24 in Qwen3.5 hybrid),
    // since `pflash_score_q8_kv` reads exactly that layer's K. Layers
    // beyond it and the final norm + lm_head are wasted compute for
    // pflash. Saves ~80% of drafter forward time on hybrid drafters.
    let layer_end = band
        .map(|b| b.layer_end)
        .unwrap_or(config.n_layers)
        .min(max_layer.unwrap_or(usize::MAX));
    // Skip final norm + lm_head when the caller early-exits — they produce
    // logits the caller doesn't read, and require running through the full
    // layer stack anyway.
    let do_lm_head = band.map(|b| b.is_last_band).unwrap_or(true) && max_layer.is_none();
    let debug_stop_after_la_layer = std::env::var("HIPFIRE_PREFILL_STOP_AFTER_LA_LAYER")
        .ok()
        .and_then(|v| v.parse::<usize>().ok());
    let debug_stop_stage_layer = std::env::var("HIPFIRE_PREFILL_STOP_STAGE_LAYER")
        .ok()
        .and_then(|v| v.parse::<usize>().ok());
    let debug_stop_stage = std::env::var("HIPFIRE_PREFILL_STOP_STAGE").ok();
    macro_rules! debug_stop_after {
        ($stage:literal, $layer_idx:expr) => {
            if debug_stop_stage_layer == Some($layer_idx)
                && debug_stop_stage.as_deref() == Some($stage)
            {
                return Ok(());
            }
        };
    }
    // Per-call-site `givens_cos_view` / `givens_sin_view` macros below
    // resolve to either the band-supplied per-device replica (multi-GPU
    // mode where `kv_cache.givens_*` is `None` by design) or the
    // kv_cache's own table (single-GPU). Held as macros, not top-level
    // bindings, so the immutable borrow on `kv_cache.givens_*` doesn't
    // outlive the kernel-call statement and conflict with later
    // mutable borrows of `kv_cache` (e.g. inside `run_fa_layer_body`).
    macro_rules! givens_cos_view {
        () => {
            band.and_then(|b| b.givens_cos)
                .or(kv_cache.givens_cos.as_ref())
        };
    }
    macro_rules! givens_sin_view {
        () => {
            band.and_then(|b| b.givens_sin)
                .or(kv_cache.givens_sin.as_ref())
        };
    }

    // ── 1. Embed tokens into pbs.x_batch ─────────────────────────────────
    //
    // Fast path for HFQ4G256 (all MQ4-quantized Qwen3.5 models + friends):
    // upload token ids to a device buffer and dispatch one batched kernel
    // that dequantizes N rows directly into `pbs.x_batch`. This collapses
    // 2N launches (N embed + N memcpy_dtod_at) into 1 upload + 1 launch
    // AND is hipGraph-captureable — the kernel reads token ids from a
    // device pointer instead of taking them as a baked-in scalar arg.
    //
    // Other formats fall back to the per-token loop (kept for correctness
    // breadth; the MQ4-quantized hot path doesn't hit them).
    //
    // Multi-GPU band-mode: skip embedding when this is not the first band.
    // The activation already lives in `pbs.x_batch` from a peer-copy of
    // the previous band's `pbs.x_batch`.
    if do_embed
        && matches!(
            weights.embd_format,
            EmbeddingFormat::HFQ4G256
                | EmbeddingFormat::Q8_0
                | EmbeddingFormat::F32
                | EmbeddingFormat::BF16
                | EmbeddingFormat::F16
        )
    {
        if !pre_uploaded {
            let tokens_host: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
            let tokens_bytes: &[u8] =
                unsafe { std::slice::from_raw_parts(tokens_host.as_ptr() as *const u8, n * 4) };
            gpu.hip.memcpy_htod(&pbs.tokens.buf, tokens_bytes)?;
        }
        match weights.embd_format {
            EmbeddingFormat::HFQ4G256 => {
                gpu.embedding_lookup_hfq4g256_batched(
                    &weights.token_embd,
                    &pbs.x_batch,
                    &pbs.tokens,
                    n,
                    dim,
                )?;
            }
            EmbeddingFormat::Q8_0 => {
                gpu.embedding_lookup_q8_batched(
                    &weights.token_embd,
                    &pbs.x_batch,
                    &pbs.tokens,
                    n,
                    dim,
                )?;
            }
            EmbeddingFormat::F32 => {
                gpu.embedding_lookup_f32_batched(
                    &weights.token_embd,
                    &pbs.x_batch,
                    &pbs.tokens,
                    n,
                    dim,
                )?;
            }
            EmbeddingFormat::BF16 => {
                gpu.embedding_lookup_bf16_batched(
                    &weights.token_embd,
                    &pbs.x_batch,
                    &pbs.tokens,
                    n,
                    dim,
                )?;
            }
            EmbeddingFormat::F16 => {
                gpu.embedding_lookup_f16_batched(
                    &weights.token_embd,
                    &pbs.x_batch,
                    &pbs.tokens,
                    n,
                    dim,
                )?;
            }
            _ => unreachable!(),
        }
    } else if do_embed {
        for (i, &tok) in tokens.iter().enumerate() {
            match weights.embd_format {
                EmbeddingFormat::HFQ4G256 => unreachable!(),
                EmbeddingFormat::HFQ4G128 => {
                    gpu.embedding_lookup_hfq4g128(&weights.token_embd, &s.x, tok, dim)?
                }
                EmbeddingFormat::Q8_0 => {
                    gpu.embedding_lookup_q8(&weights.token_embd, &s.x, tok, dim)?
                }
                EmbeddingFormat::BF16 => {
                    gpu.embedding_lookup_bf16(&weights.token_embd, &s.x, tok, dim)?
                }
                EmbeddingFormat::F16 => {
                    gpu.embedding_lookup_f16(&weights.token_embd, &s.x, tok, dim)?
                }
                EmbeddingFormat::F32 => {
                    gpu.embedding_lookup(&weights.token_embd, &s.x, tok, dim)?
                }
                _ => panic!("unsupported embedding format"),
            }
            gpu.memcpy_dtod_at_auto(
                &pbs.x_batch.buf,
                i * dim_row_bytes,
                &s.x.buf,
                0,
                dim_row_bytes,
            )?;
        }
    }

    // ── 1a. Apply MaskEmbedOverride (MTP probe hook) ─────────────────────
    //
    // Overwrite a single batch slot's embedding row in `pbs.x_batch` after
    // the embedding-lookup kernel populated it but BEFORE the layer loop
    // (or any subsequent kernel) reads it. The Qualcomm MTP probe uses this
    // to replace the embedding-table value at a "mask token" position with
    // a prompt-mean vector. Default callers pass `None` → zero overhead.
    //
    // Multi-GPU band-mode: skip on non-first bands; pbs.x_batch already
    // holds the peer-copied activation from the previous band, so an
    // override applied at band 0 has already propagated through the layer
    // stack on that device — re-applying here would clobber the partial
    // forward state.
    if do_embed {
        if let Some(ovr) = mask_override {
            assert!(
                ovr.slot < n,
                "MaskEmbedOverride.slot ({}) must be < n ({})",
                ovr.slot,
                n,
            );
            assert_eq!(
                ovr.embed.len(),
                dim,
                "MaskEmbedOverride.embed.len() ({}) must equal config.dim ({})",
                ovr.embed.len(),
                dim,
            );
            let bytes: &[u8] =
                unsafe { std::slice::from_raw_parts(ovr.embed.as_ptr() as *const u8, dim * 4) };
            let offset = ovr.slot * dim_row_bytes;
            gpu.hip
                .memcpy_htod_offset(&pbs.x_batch.buf, offset, bytes)?;
        }
    }

    // ── 1b. Upload positions array ────────────────────────────────────────
    //
    // Positions is the per-row RoPE angle AND the physical KV cache slot (the
    // batched kv_write kernels use the same index for both). Default callers
    // use flat linear `start_pos .. start_pos + n`; the dense server-prefill
    // session-batch worker can pass explicit per-row positions for independent
    // sessions. Siblings in DDTree mode get DISTINCT slots via the default
    // linear path — no write race — and the stored K carries a RoPE angle that
    // matches the physical slot, which keeps subsequent cycles' attention
    // reads consistent.
    //
    // Semantic trade vs. the original depth-based scheme (paper): tree
    // siblings that represent "alternative futures at the same time step"
    // now see a RoPE distance of 1 (or more) instead of 0. Empirically that
    // slight distance shift costs little — the attn_bias mask still gates
    // ancestor visibility exactly, and the Q·K dot products stay consistent
    // across the whole cache (prompt + tree block). In exchange we get
    // DDTree correctness for topk>1 without needing a tree-local KV scratch
    // or a scatter-kernel for commit. `ctx.positions` is accepted for API
    // compatibility but ignored — the DdNode depths it carries are only
    // used by `linearize_tree` to build the attn_bias mask.
    if !pre_uploaded {
        let positions_host: Vec<i32> = if let Some(positions) = positions_override {
            assert_eq!(
                positions.len(),
                n,
                "positions_override length {} must equal tokens.len() {}",
                positions.len(),
                n,
            );
            positions.iter().map(|&p| p as i32).collect()
        } else {
            (0..n).map(|i| (start_pos + i) as i32).collect()
        };
        let positions_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(positions_host.as_ptr() as *const u8, n * 4) };
        gpu.hip.memcpy_htod(&pbs.positions.buf, positions_bytes)?;
    }

    // Decide whether the FA layers can take the batched path. Requires
    // (a) all FA weights to be MQ4G256 or HFQ4G256 (the batched gemm_qkv
    // + wo GEMMs are dtype-agnostic; the rmsnorm+rotate / silu_mul kernels
    // differ by dtype and we branch on that at each layer) and (b) a Q8_0
    // or givens KV cache. If the check fails, FA layers fall back to
    // per-token gather/scatter via run_fa_layer_body.
    let fa_arch = gpu.arch.as_str();
    // Q8 WMMA gate: the fused Q8 WMMA family (gemm_qkv/qkvza/gate_up/residual
    // _q8_0_wmma) uses the gfx11 `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32`
    // builtin; the sibling `*.gfx12.hip` kernels use the `_w32_gfx12` variant
    // (silicon-validated on R9700, 2026-05-14, 4/4 unit tests PASS). Each
    // call site below selects the right variant via an `arch.starts_with`
    // branch. On non-WMMA archs we keep the Tier 2 chunked-substrate path.
    // Split the two halves so a failure can be NAMED. When FA layers fall back
    // they go per-token through `run_fa_layer_body`, and the only outward sign
    // is a GEMV-dominated kernel histogram — measured 43:1 against GEMM on
    // Qwen3.5-0.8B--mq4. Which of the two conditions failed is what turns that
    // into something actionable, and it is invisible from outside.
    let fa_kv_ok = !kv_cache.quantized
        || kv_cache.quant_q8
        // KVarN belongs here: prefill_chunk handles it explicitly (kvarn_attend
        // owns the batched write). Omitting it sends every FullAttn layer down
        // the per-token run_fa_layer_body fallback, which on Qwen3.8-27B (16 FA
        // layers) measured 136k GEMV dispatches and 54 tok/s against 301.
        || kv_cache.quant_kvarn
        || kv_cache.quant_asym4
        || kv_cache.quant_asym3
        || kv_cache.quant_asym2;
    let fa_batched_ok = fa_kv_ok
        && weights.layers.iter().all(|lw| match lw {
            LayerWeights::FullAttn(l) => {
                is_batchable_la(l.wq.gpu_dtype, fa_arch, gdn_tape.is_none())
                    && is_batchable_la(l.wk.gpu_dtype, fa_arch, gdn_tape.is_none())
                    && is_batchable_la(l.wv.gpu_dtype, fa_arch, gdn_tape.is_none())
                    && is_batchable_la(l.wo.gpu_dtype, fa_arch, gdn_tape.is_none())
                    && is_batchable_la(l.w_gate.gpu_dtype, fa_arch, gdn_tape.is_none())
                    && is_batchable_la(l.w_up.gpu_dtype, fa_arch, gdn_tape.is_none())
                    && is_batchable_la(l.w_down.gpu_dtype, fa_arch, gdn_tape.is_none())
            }
            // MoE variant: attention weights must be MQ4-class (FFN is
            // checked separately by moe_ffn_batched_admissible in the eligibility gate).
            LayerWeights::FullAttnMoe(l) => {
                is_batchable_la(l.wq.gpu_dtype, fa_arch, gdn_tape.is_none())
                    && is_batchable_la(l.wk.gpu_dtype, fa_arch, gdn_tape.is_none())
                    && is_batchable_la(l.wv.gpu_dtype, fa_arch, gdn_tape.is_none())
                    && is_batchable_la(l.wo.gpu_dtype, fa_arch, gdn_tape.is_none())
            }
            _ => true, // LA layers don't gate this check
        });

    // Reported under HIPFIRE_KERNEL_TRACE so it lands beside the histogram that
    // motivates the question, and costs nothing otherwise.
    if !fa_batched_ok && hipfire_rdna::kernel_trace::enabled() {
        let bad_dtype = weights.layers.iter().find_map(|lw| match lw {
            LayerWeights::FullAttn(l) => {
                (!is_batchable_la(l.wq.gpu_dtype, fa_arch, gdn_tape.is_none())).then(|| format!("{:?}", l.wq.gpu_dtype))
            }
            _ => None,
        });
        eprintln!(
            "[kernel-trace] FA layers take the PER-TOKEN fallback: kv_ok={fa_kv_ok}              (quantized={}, q8={}, asym4={}, asym3={}, asym2={}), first non-batchable              FA weight dtype={}",
            kv_cache.quantized,
            kv_cache.quant_q8,
            kv_cache.quant_asym4,
            kv_cache.quant_asym3,
            kv_cache.quant_asym2,
            bad_dtype.as_deref().unwrap_or("<none — weights all batchable>")
        );
    }
    // Under hipGraph capture, scalar kernargs get BAKED into the kernarg blob
    // at capture time. `max_ctx_len = start_pos + n` grows per cycle, so the
    // captured value would be stale on replay — the attention kernel would
    // allocate too-small LDS for `scores[]` and over-read. Bake the physical
    // cap instead (LDS sized for the worst case). The kernel still iterates
    // over the actual `positions[b] + 1` per-row seq_len from a device buffer,
    // so correctness is preserved; only the LDS allocation is over-provisioned.
    let max_ctx_len = if gpu.capture_mode {
        kv_cache.physical_cap
    } else if let Some(positions) = positions_override {
        positions.iter().copied().max().unwrap_or(start_pos) + 1
    } else {
        start_pos + n
    };
    let position_at_row = |row: usize| -> usize {
        positions_override
            .map(|p| p[row])
            .unwrap_or(start_pos + row)
    };

    // ── 2. Per-layer loop ────────────────────────────────────────────────
    // Multi-GPU band-mode: counters seed from the band's running offsets so
    // the band's first DeltaNet/FullAttn layer reads the correct
    // `dn_state.s_matrices[i]` / `kv_cache.k_caches[i]` slot. Single-GPU
    // (band==None) seeds zeros — original behavior.
    let mut delta_layer_idx = band.map(|b| b.delta_layer_offset).unwrap_or(0);
    // Path B: per-FA-layer counter, drives the index into
    // tree_verify.pre_rope_k_capture[]. Increments alongside each
    // FullAttention layer iteration regardless of MoE/non-MoE variant.
    let mut fa_layer_idx = band.map(|b| b.fa_layer_offset).unwrap_or(0);
    let use_gdn_per_token =
        force_q8_gdn_per_token || (gdn_tape.is_some() && q8_gdn_verify_per_token_enabled());
    let q8_gdn_serial_frame_base = if use_gdn_per_token
        && q8_gdn_verify_serial_frames_enabled()
        && gdn_tape.is_some()
        && tree_verify.is_none()
        && band.is_none()
    {
        Some(gpu.debug_gdn_requant_frame())
    } else {
        None
    };
    let q8_gdn_serial_frame_layers = config
        .layer_types
        .iter()
        .filter(|&&lt| lt == LayerType::LinearAttention)
        .count();
    if let Some(frame_base) = q8_gdn_serial_frame_base {
        if let Some(tape) = gdn_tape.as_deref_mut() {
            tape.q8_requant_frame_base = Some(frame_base);
            tape.q8_requant_frame_layers = q8_gdn_serial_frame_layers;
        }
    }
    let ctx = DispatchCtx::new(gpu); // hoisted — arch-constant, safe to reuse per-layer
                                     // §M2a2 — one decision per chunk, and the trace reports the DECISION.
    let take_prefill_lowered = prefill_lowered_enabled();
    if prefill_backend_trace_enabled() {
        eprintln!(
            "  [prefill] batched FA layers → {} path{}",
            if take_prefill_lowered {
                "lowered"
            } else {
                "direct"
            },
            if take_prefill_lowered {
                ""
            } else {
                " (HIPFIRE_PREFILL_LOWERED=0)"
            }
        );
    }
    // Built once per chunk: `lower_variant` is pure and returns the same five
    // ops for every FullAttn layer.
    let fa_program = lower_variant(Q35Variant::FullAttn);
    let dn_program = lower_variant(Q35Variant::DeltaNet);

    for layer_idx in layer_start..layer_end {
        match (&weights.layers[layer_idx], config.layer_types[layer_idx]) {
            (LayerWeights::DeltaNet(layer), LayerType::LinearAttention) => {
                // §M2a4 — the seven super-ops of `lower_variant(DeltaNet)`,
                // executed over `n` rows. Bodies moved to `prefill_lowered.rs`
                // unchanged, cut where this arm's own comments named the
                // boundaries. DeltaNet came AFTER the dense FullAttn arm on
                // purpose (plan §M2a4): it carries the sequential recurrent
                // scan, in-place `dn_state.s_matrices` mutation, and the
                // `gdn_tape` capture interaction, so doing it first would have
                // meant debugging the extraction and those together.
                {
                    let mut bind = Qwen35PrefillDnBindings {
                        layer,
                        pbs,
                        config,
                        dn_state: &*dn_state,
                        gdn_tape: gdn_tape.as_deref(),
                        tree_verify,
                        use_gdn_per_token,
                        n,
                        tape_offset,
                        delta_layer_idx,
                    };
                    if take_prefill_lowered {
                        superop::run_layer_program(gpu, &ctx, &dn_program, &mut bind)
                            .map_err(|e| HipError::new(0, &e.to_string()))?;
                    } else {
                        bind.proj_qkvza(gpu)?;
                        bind.attend_dn_prep(gpu)?;
                        bind.recur_gdn(gpu)?;
                        bind.norm_gated(gpu)?;
                        bind.resid_wo(gpu)?;
                        bind.proj_gate_up(gpu)?;
                        bind.resid_down_swiglu(gpu)?;
                    }
                }

                if let Some(tape) = gdn_tape.as_ref() {
                    let hidden_row_bytes = tape.x_in_dim * 4;
                    let off_hidden = tape_offset * hidden_row_bytes;
                    gpu.memcpy_dtod_at_auto(
                        &tape.layer_out_bufs[delta_layer_idx].buf,
                        off_hidden,
                        &pbs.x_batch.buf,
                        0,
                        n * hidden_row_bytes,
                    )?;
                }

                // Post-layer hidden extract for the DFlash draft path.
                if let Some(rb) = hidden_rb {
                    if let Some(slot) = rb.extract_slot(layer_idx) {
                        rb.write_rows_to_staging(gpu, slot, &pbs.x_batch, n)?;
                    }
                }

                delta_layer_idx += 1;
            }

            (LayerWeights::FullAttn(layer), LayerType::FullAttention) if fa_batched_ok => {
                // Fully batched FA layer. Mirrors the FA branch of
                // forward_scratch_layers kernel-for-kernel, but every
                // §M2a3 — the five super-ops of `lower_variant(FullAttn)`,
                // executed over `n` rows. The kernel sequences that used to sit
                // inline here now live in `prefill_lowered.rs`, moved unchanged
                // and cut on this arm's own numbered phase boundaries (1-2 →
                // PROJ_QKV, 4-8 → ATTEND_FULL, 9 → RESID_WO, 10a → PROJ_GATE_UP,
                // 10b → RESID_DOWN_SWIGLU).
                //
                // The row count travels in the bindings struct, NOT in
                // `DispatchCtx` — that has three arch-constant fields resolved
                // once at `Gpu::init()` and 42 `::new()` sites across 13 crates
                // (plan §M2a3, §6).
                {
                    let mut bind = Qwen35PrefillBindings {
                        layer,
                        s,
                        pbs,
                        config,
                        kv_cache: &mut *kv_cache,
                        gdn_tape: gdn_tape.as_deref(),
                        tree_verify,
                        n,
                        start_pos,
                        tape_offset,
                        delta_layer_idx,
                        layer_idx,
                        fa_layer_idx,
                        max_ctx_len,
                        positions_override,
                        band_givens_cos: band.and_then(|b| b.givens_cos),
                        band_givens_sin: band.and_then(|b| b.givens_sin),
                    };
                    if take_prefill_lowered {
                        superop::run_layer_program(gpu, &ctx, &fa_program, &mut bind)
                            .map_err(|e| HipError::new(0, &e.to_string()))?;
                    } else {
                        // Same five ops, called in order instead of sequenced by
                        // the executor. The rollback exists to isolate the
                        // executor from the kernels, so it must not be a second
                        // copy of the kernels.
                        bind.proj_qkv(gpu)?;
                        bind.attend_full(gpu, &ctx)?;
                        bind.resid_wo(gpu)?;
                        bind.proj_gate_up(gpu)?;
                        bind.resid_down_swiglu(gpu)?;
                    }
                }
                if let Some(tape) = gdn_tape.as_ref() {
                    if delta_layer_idx < tape.fa_bridge_valid.len()
                        && tape.fa_bridge_valid[delta_layer_idx]
                    {
                        let hidden_row_bytes = tape.x_in_dim * 4;
                        let off_hidden = tape_offset * hidden_row_bytes;
                        gpu.memcpy_dtod_at_auto(
                            &tape.fa_bridge_layer_out_bufs[delta_layer_idx].buf,
                            off_hidden,
                            &pbs.x_batch.buf,
                            0,
                            n * hidden_row_bytes,
                        )?;
                    }
                }

                // Post-layer hidden extract for the DFlash draft path.
                if let Some(rb) = hidden_rb {
                    if let Some(slot) = rb.extract_slot(layer_idx) {
                        rb.write_rows_to_staging(gpu, slot, &pbs.x_batch, n)?;
                    }
                }

                fa_layer_idx += 1;
            }

            (LayerWeights::FullAttn(_layer), LayerType::FullAttention) => {
                // Per-token gather/scatter fallback for FA layers that don't
                // qualify for batched FA (non-MQ4 weights, non-Q8_0 KV, etc).
                //
                // The body is the LOWERED FullAttn program — the same five
                // super-ops the decode path runs — not a second hand-written
                // copy of it. It used to call `run_fa_layer_body`, 661 lines
                // whose own doc said it was "byte-exact with the FA branch of
                // forward_scratch_layers": the very hand arm that
                // `lower_variant(Q35Variant::FullAttn)` already replaced and was
                // validated against. Two copies of one arm is one copy too many,
                // and the second was the one nothing gated.
                //
                // `run_layer_program` directly, NOT
                // `forward_scratch_layers_lowered` — that loops all layers and
                // appends final-norm + lm_head, which this arm must not do.
                let program = lower_variant(Q35Variant::FullAttn);
                for i in 0..n {
                    let pos = start_pos + i;
                    gpu.hip.memcpy_dtod_at(
                        &s.x.buf,
                        0,
                        &pbs.x_batch.buf,
                        i * dim_row_bytes,
                        dim_row_bytes,
                    )?;
                    let pos_i32 = pos as i32;
                    gpu.memcpy_htod_auto(&s.pos_buf, &pos_i32.to_ne_bytes())?;
                    {
                        let mut bind = Qwen35Bindings {
                            pager: weights.pager.as_ref(),
                            layer: &weights.layers[layer_idx],
                            s,
                            config,
                            kv_cache: &mut *kv_cache,
                            dn_state: &*dn_state,
                            pos,
                            layer_idx,
                            delta_layer_idx,
                            k_dim,
                            v_dim,
                            n_v_heads,
                            hd,
                        };
                        superop::run_layer_program(gpu, &ctx, &program, &mut bind)
                            .map_err(|e| HipError::new(0, &e.to_string()))?;
                    }
                    gpu.hip.memcpy_dtod_at(
                        &pbs.x_batch.buf,
                        i * dim_row_bytes,
                        &s.x.buf,
                        0,
                        dim_row_bytes,
                    )?;
                }

                // Post-layer hidden extract for the DFlash draft path. After
                // the per-token loop, pbs.x_batch has the full layer output
                // for all N tokens (last copy-back finishes each row).
                if let Some(rb) = hidden_rb {
                    if let Some(slot) = rb.extract_slot(layer_idx) {
                        rb.write_rows_to_staging(gpu, slot, &pbs.x_batch, n)?;
                    }
                }

                fa_layer_idx += 1;
            }

            (LayerWeights::DeltaNetMoe(layer), LayerType::LinearAttention) => {
                // Batched MoE LA layer. LA body is the same as DeltaNet
                // (rmsnorm + qkvza + sigmoid_alpha + conv1d + L2norm +
                // repeat_interleave + GDN + gated_norm + wo+residual);
                // only the FFN differs. Duplicated inline for now — can
                // be factored into a `prefill_la_body_batched` helper
                // when dense and MoE LA paths are proven byte-exact.
                // This body is unreachable for MQ3 / MQ3-Lloyd weights —
                // the upstream `mq3_in_moe` guard at the top of
                // `forward_prefill_batch_with_pbs` rejects any MoE layer
                // with MQ3/Lloyd-MQ3 weights anywhere (attention OR FFN),
                // mirroring the captured-path guard at line 3367+. So
                // `layer.wqkv.gpu_dtype` is restricted here to MQ4G256 /
                // HFQ4G256 / MQ6G256 / HFQ6G256 / Q8_0. Q8 admit landed
                // alongside the moe_ffn router/gate Q8 unlock (A3B's LA
                // attention weights are Q8 — engine quantizer keeps q/k/v/o
                // at Q8 alongside the Q8 router + shared_expert_gate).
                // Opus (Oq4/Oq8/OqCompact) weights are FWHT(+AWQ)-rotated OFFLINE, so the
                // activation must be rotated to match. Leaving them out of this
                // predicate sends an UNROTATED x into an Opus GEMM -- the dense path
                // records that outcome as "garbage: PPL 3.5e6".
                let is_mq = matches!(
                    layer.wqkv.gpu_dtype,
                    DType::MQ4G256
                        | DType::MQ6G256
                        | DType::Oq4G256
                        | DType::Oq8G256
                        | DType::OqCompactG256
                );
                let is_6bit = matches!(layer.wqkv.gpu_dtype, DType::MQ6G256 | DType::HFQ6G256);
                let is_q8 = matches!(layer.wqkv.gpu_dtype, DType::Q8_0);
                // Phase 1.5: PARO mode for DeltaNetMoe — wqkv/wz are
                // ParoQ4G128 (each with its own Givens rotation tables);
                // w_alpha/w_beta are F32 (no rotation, no quantization).
                // Dispatch is unfused: rotate+gemm_hfq4g128 for wqkv and wz,
                // direct gemm_f32_batched for w_alpha and w_beta. Same shape
                // outputs as the Q8/MQ4 paths (dn_qkv_batch, dn_z_batch,
                // dn_alpha_batch, dn_beta_batch).
                let is_paro = matches!(layer.wqkv.gpu_dtype, DType::ParoQ4G128);
                let q8_wmma_arch = gpu.arch_caps.has_wmma();

                if is_mq {
                    // AWQ-aware: next linear is LA's fused wqkv.
                    fused_rmsnorm_rotate_mq_batched_for(
                        gpu,
                        &pbs.x_batch,
                        &layer.attn_norm,
                        &layer.wqkv,
                        &pbs.x_rot_batch,
                        dim,
                        config.norm_eps,
                        n,
                    )?;
                } else if is_paro {
                    // PARO: need un-rotated x_norm available for per-weight
                    // Givens rotation. Write rmsnorm into x_norm_batch (the
                    // dedicated normalized buffer); x_rot_batch becomes the
                    // per-weight rotation scratch (overwritten per GEMM).
                    gpu.rmsnorm_batched(
                        &pbs.x_batch,
                        &layer.attn_norm,
                        &pbs.x_norm_batch,
                        n,
                        dim,
                        config.norm_eps,
                    )?;
                } else {
                    gpu.rmsnorm_batched(
                        &pbs.x_batch,
                        &layer.attn_norm,
                        &pbs.x_rot_batch,
                        n,
                        dim,
                        config.norm_eps,
                    )?;
                }
                debug_stop_after!("attn_norm", layer_idx);
                if is_paro {
                    // PARO 4-way unfused dispatch. wqkv and wz are
                    // ParoQ4G128 with their own Givens rotation tables;
                    // w_alpha and w_beta are F32 with no rotation.
                    let paro_wqkv = layer.wqkv.paro.as_ref().unwrap_or_else(|| {
                        panic!(
                            "ParoQ4G128 wqkv missing paro metadata at LA layer {layer_idx} \
                             — paro_load_wt() loader regression?"
                        )
                    });
                    let paro_wz = layer.wz.paro.as_ref().unwrap_or_else(|| {
                        panic!("ParoQ4G128 wz missing paro metadata at LA layer {layer_idx}")
                    });
                    // wqkv: rotate x_norm → x_rot, then HFQ4G128 GEMM.
                    gpu.givens_rotate_to(
                        &pbs.x_norm_batch,
                        &pbs.x_rot_batch,
                        &paro_wqkv.pairs,
                        &paro_wqkv.theta,
                        &paro_wqkv.channel_scales,
                        n,
                        dim,
                        paro_wqkv.krot as usize,
                    )?;
                    run_plain_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmHfq4G128,
                        &layer.wqkv.buf,
                        layer.wqkv.gpu_dtype,
                        &pbs.x_rot_batch,
                        &pbs.dn_qkv_batch,
                        layer.wqkv.m,
                        layer.wqkv.k,
                        n,
                    )?;
                    // wz: re-rotate x_norm → x_rot (overwrite), then GEMM.
                    gpu.givens_rotate_to(
                        &pbs.x_norm_batch,
                        &pbs.x_rot_batch,
                        &paro_wz.pairs,
                        &paro_wz.theta,
                        &paro_wz.channel_scales,
                        n,
                        dim,
                        paro_wz.krot as usize,
                    )?;
                    run_plain_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmHfq4G128,
                        &layer.wz.buf,
                        layer.wz.gpu_dtype,
                        &pbs.x_rot_batch,
                        &pbs.dn_z_batch,
                        layer.wz.m,
                        layer.wz.k,
                        n,
                    )?;
                    // w_alpha / w_beta: F32, no rotation, direct batched GEMM.
                    run_plain_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmF32Batched,
                        &layer.w_alpha.buf,
                        layer.w_alpha.gpu_dtype,
                        &pbs.x_norm_batch,
                        &pbs.dn_alpha_batch,
                        layer.w_alpha.m,
                        layer.w_alpha.k,
                        n,
                    )?;
                    run_plain_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmF32Batched,
                        &layer.w_beta.buf,
                        layer.w_beta.gpu_dtype,
                        &pbs.x_norm_batch,
                        &pbs.dn_beta_batch,
                        layer.w_beta.m,
                        layer.w_beta.k,
                        n,
                    )?;
                } else if is_6bit {
                    run_fused_qkvza_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::FusedQkvzaHfq6G256,
                        &layer.wqkv.buf,
                        &layer.wz.buf,
                        &layer.w_beta.buf,
                        &layer.w_alpha.buf,
                        &pbs.x_rot_batch,
                        &pbs.dn_qkv_batch,
                        &pbs.dn_z_batch,
                        &pbs.dn_beta_batch,
                        &pbs.dn_alpha_batch,
                        layer.wqkv.m,
                        layer.wz.m,
                        layer.w_beta.m,
                        layer.w_alpha.m,
                        layer.wqkv.k,
                        n,
                    )?;
                } else if is_q8 && q8_wmma_arch {
                    // Fused Q8 QKVZA WMMA — assumes all 4 weights share Q8_0
                    // stride; mixed Q8/other layers within DNMoe are rejected
                    // upstream by `moe_ffn_batched_admissible` (router/gate Q8 OK, but
                    // shared_expert + experts must be MQ4) and would otherwise
                    // re-introduce Tier-1 stride corruption.
                    debug_assert!(
                        matches!(layer.wz.gpu_dtype, DType::Q8_0)
                        && matches!(layer.w_beta.gpu_dtype, DType::Q8_0)
                        && matches!(layer.w_alpha.gpu_dtype, DType::Q8_0),
                        "DNMoe LA qkvza Q8 WMMA dispatch requires all of wqkv/wz/w_beta/w_alpha to be Q8_0",
                    );
                    run_fused_qkvza_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::FusedQkvzaQ8_0,
                        &layer.wqkv.buf,
                        &layer.wz.buf,
                        &layer.w_beta.buf,
                        &layer.w_alpha.buf,
                        &pbs.x_rot_batch,
                        &pbs.dn_qkv_batch,
                        &pbs.dn_z_batch,
                        &pbs.dn_beta_batch,
                        &pbs.dn_alpha_batch,
                        layer.wqkv.m,
                        layer.wz.m,
                        layer.w_beta.m,
                        layer.w_alpha.m,
                        layer.wqkv.k,
                        n,
                    )?;
                } else if is_q8 {
                    // #397 Ship 5.2 slice1: four plain Q8 batched GEMMs
                    // (wqkv/wz/w_beta/w_alpha), sibling DeltaNet QKVZA path.
                    run_plain_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmQ8_0BatchedChunked,
                        &layer.wqkv.buf,
                        layer.wqkv.gpu_dtype,
                        &pbs.x_rot_batch,
                        &pbs.dn_qkv_batch,
                        layer.wqkv.m,
                        layer.wqkv.k,
                        n,
                    )?;
                    run_plain_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmQ8_0BatchedChunked,
                        &layer.wz.buf,
                        layer.wz.gpu_dtype,
                        &pbs.x_rot_batch,
                        &pbs.dn_z_batch,
                        layer.wz.m,
                        layer.wz.k,
                        n,
                    )?;
                    run_plain_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmQ8_0BatchedChunked,
                        &layer.w_beta.buf,
                        layer.w_beta.gpu_dtype,
                        &pbs.x_rot_batch,
                        &pbs.dn_beta_batch,
                        layer.w_beta.m,
                        layer.w_beta.k,
                        n,
                    )?;
                    run_plain_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmQ8_0BatchedChunked,
                        &layer.w_alpha.buf,
                        layer.w_alpha.gpu_dtype,
                        &pbs.x_rot_batch,
                        &pbs.dn_alpha_batch,
                        layer.w_alpha.m,
                        layer.w_alpha.k,
                        n,
                    )?;
                } else if matches!(layer.wqkv.gpu_dtype, DType::OqCompactG256) {
                    // Compact-resident Opus: the SAME W8A8 math as the oq8 arm
                    // above, decoding OqPlusCompact blocks in-kernel.
                    //
                    // Without this arm compact fell through to the final `else`,
                    // which hard-codes `FusedQkvzaHfq4G256` — so a 136-byte
                    // [f16 scale][128 nibbles][3x(u8 idx, i8 val)] block was read
                    // as an HFQ4G256 block. Two layouts sharing no field, and no
                    // error: just wrong numbers. That is what made compact's
                    // BATCHED prefill unusable, which is why the batched
                    // spec-decode verify had to exclude compact entirely and why
                    // the tape-free rollback replay collapsed tau 3.00 -> 0.63.
                    // Measured: 48 fall-throughs in a 16-token spec run.
                    gpu.quantize_act_oq8_batched_interleaved(
                        &pbs.x_rot_batch,
                        layer.wqkv.m,
                        layer.wqkv.k,
                        n,
                    )?;
                    for (w, y) in [
                        (&layer.wqkv, &pbs.dn_qkv_batch),
                        (&layer.wz, &pbs.dn_z_batch),
                        (&layer.w_beta, &pbs.dn_beta_batch),
                        (&layer.w_alpha, &pbs.dn_alpha_batch),
                    ] {
                        let bs = super::prefill_batch::oq_compact_block_stride(w)?;
                        gpu.gemm_oq_compact_grouped_prequant(&w.buf, y, w.m, w.k, n, bs)?;
                    }
                } else {
                    run_fused_qkvza_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::FusedQkvzaHfq4G256,
                        &layer.wqkv.buf,
                        &layer.wz.buf,
                        &layer.w_beta.buf,
                        &layer.w_alpha.buf,
                        &pbs.x_rot_batch,
                        &pbs.dn_qkv_batch,
                        &pbs.dn_z_batch,
                        &pbs.dn_beta_batch,
                        &pbs.dn_alpha_batch,
                        layer.wqkv.m,
                        layer.wz.m,
                        layer.w_beta.m,
                        layer.w_alpha.m,
                        layer.wqkv.k,
                        n,
                    )?;
                }
                debug_stop_after!("qkvza", layer_idx);
                if let Some(tape) = gdn_tape.as_ref() {
                    let x_in_row_bytes = tape.x_in_dim * 4;
                    let alpha_row_bytes = n_v_heads * 4;
                    let off_x = tape_offset * x_in_row_bytes;
                    let off_a = tape_offset * alpha_row_bytes;
                    let copy_x = n * x_in_row_bytes;
                    let copy_a = n * alpha_row_bytes;
                    gpu.memcpy_dtod_at_auto(
                        &tape.x_in_bufs[delta_layer_idx].buf,
                        off_x,
                        &pbs.x_rot_batch.buf,
                        0,
                        copy_x,
                    )?;
                    gpu.memcpy_dtod_at_auto(
                        &tape.alpha_raw_bufs[delta_layer_idx].buf,
                        off_a,
                        &pbs.dn_alpha_batch.buf,
                        0,
                        copy_a,
                    )?;
                    gpu.memcpy_dtod_at_auto(
                        &tape.beta_raw_bufs[delta_layer_idx].buf,
                        off_a,
                        &pbs.dn_beta_batch.buf,
                        0,
                        copy_a,
                    )?;
                }
                gpu.fused_sigmoid_alpha_gate_f32_batched(
                    &pbs.dn_beta_batch,
                    &pbs.dn_alpha_batch,
                    &layer.dt_bias,
                    &layer.a_log,
                    n_v_heads,
                    n,
                )?;
                debug_stop_after!("sigmoid_alpha", layer_idx);
                if let Some(tape) = gdn_tape.as_ref() {
                    let qkv_row_bytes = tape.qkv_dim * 4;
                    let alpha_row_bytes = n_v_heads * 4;
                    let off_qkv = tape_offset * qkv_row_bytes;
                    let off_a = tape_offset * alpha_row_bytes;
                    let copy_qkv = n * qkv_row_bytes;
                    let copy_a = n * alpha_row_bytes;
                    gpu.memcpy_dtod_at_auto(
                        &tape.qkv_bufs[delta_layer_idx].buf,
                        off_qkv,
                        &pbs.dn_qkv_batch.buf,
                        0,
                        copy_qkv,
                    )?;
                    gpu.memcpy_dtod_at_auto(
                        &tape.alpha_bufs[delta_layer_idx].buf,
                        off_a,
                        &pbs.dn_alpha_batch.buf,
                        0,
                        copy_a,
                    )?;
                    gpu.memcpy_dtod_at_auto(
                        &tape.beta_bufs[delta_layer_idx].buf,
                        off_a,
                        &pbs.dn_beta_batch.buf,
                        0,
                        copy_a,
                    )?;
                }
                // Same tree-aware dispatch gate as dense LA branch above.
                let tree_parents = tree_verify.as_ref().and_then(|c| c.parent_indices);
                if let Some(parents) = tree_parents {
                    gpu.conv1d_silu_split_tree_f32_n(
                        &pbs.dn_q_raw_batch,
                        &pbs.dn_k_raw_batch,
                        &pbs.dn_v_batch,
                        &pbs.dn_qkv_batch,
                        &layer.conv_weight,
                        &dn_state.conv_states[delta_layer_idx],
                        parents,
                        k_dim,
                        v_dim,
                        n,
                    )?;
                } else {
                    gpu.conv1d_silu_split_f32_n(
                        &pbs.dn_q_raw_batch,
                        &pbs.dn_k_raw_batch,
                        &pbs.dn_v_batch,
                        &pbs.dn_qkv_batch,
                        &layer.conv_weight,
                        &dn_state.conv_states[delta_layer_idx],
                        k_dim,
                        v_dim,
                        n,
                    )?;
                }
                debug_stop_after!("conv", layer_idx);
                if let Some(tape) = gdn_tape.as_ref() {
                    let q_raw_row_bytes = tape.k_dim * 4;
                    let v_row_bytes = tape.v_dim * 4;
                    let off_q_raw = tape_offset * q_raw_row_bytes;
                    let off_v = tape_offset * v_row_bytes;
                    gpu.memcpy_dtod_at_auto(
                        &tape.q_raw_bufs[delta_layer_idx].buf,
                        off_q_raw,
                        &pbs.dn_q_raw_batch.buf,
                        0,
                        n * q_raw_row_bytes,
                    )?;
                    gpu.memcpy_dtod_at_auto(
                        &tape.k_raw_bufs[delta_layer_idx].buf,
                        off_q_raw,
                        &pbs.dn_k_raw_batch.buf,
                        0,
                        n * q_raw_row_bytes,
                    )?;
                    gpu.memcpy_dtod_at_auto(
                        &tape.v_bufs[delta_layer_idx].buf,
                        off_v,
                        &pbs.dn_v_batch.buf,
                        0,
                        n * v_row_bytes,
                    )?;
                }
                gpu.fused_qk_l2_norm_scale_f32_batched(
                    &pbs.dn_q_raw_batch,
                    &pbs.dn_k_raw_batch,
                    config.linear_num_key_heads,
                    hd,
                    1.0 / (hd as f32).sqrt(),
                    config.norm_eps,
                    n,
                )?;
                if config.linear_num_key_heads < n_v_heads {
                    let ratio = n_v_heads / config.linear_num_key_heads;
                    gpu.repeat_interleave_qk_f32_batched(
                        &pbs.dn_q_raw_batch,
                        &pbs.dn_k_raw_batch,
                        &pbs.dn_q_batch,
                        &pbs.dn_k_batch,
                        config.linear_num_key_heads,
                        ratio,
                        hd,
                        n,
                    )?;
                } else {
                    gpu.memcpy_dtod_auto(
                        &pbs.dn_q_batch.buf,
                        &pbs.dn_q_raw_batch.buf,
                        n * k_dim * 4,
                    )?;
                    gpu.memcpy_dtod_auto(
                        &pbs.dn_k_batch.buf,
                        &pbs.dn_k_raw_batch.buf,
                        n * k_dim * 4,
                    )?;
                }
                debug_stop_after!("qk_repeat", layer_idx);
                if let Some(tape) = gdn_tape.as_ref() {
                    let q_row_bytes = tape.v_dim * 4;
                    let off_q = tape_offset * q_row_bytes;
                    gpu.memcpy_dtod_at_auto(
                        &tape.q_bufs[delta_layer_idx].buf,
                        off_q,
                        &pbs.dn_q_batch.buf,
                        0,
                        n * q_row_bytes,
                    )?;
                    gpu.memcpy_dtod_at_auto(
                        &tape.k_bufs[delta_layer_idx].buf,
                        off_q,
                        &pbs.dn_k_batch.buf,
                        0,
                        n * q_row_bytes,
                    )?;
                }
                // DIAG: dump GDN inputs (batched, MoE branch)
                if layer_idx == 0 {
                    let qk_dim = n_v_heads * hd;
                    dump_hidden_localize(gpu, &pbs.dn_q_batch, n, start_pos, qk_dim, 0, "q_b");
                    dump_hidden_localize(gpu, &pbs.dn_k_batch, n, start_pos, qk_dim, 0, "k_b");
                    dump_hidden_localize(gpu, &pbs.dn_v_batch, n, start_pos, v_dim, 0, "v_b");
                    dump_hidden_localize(
                        gpu,
                        &pbs.dn_alpha_batch,
                        n,
                        start_pos,
                        n_v_heads,
                        0,
                        "alpha_b",
                    );
                    dump_hidden_localize(
                        gpu,
                        &pbs.dn_beta_batch,
                        n,
                        start_pos,
                        n_v_heads,
                        0,
                        "beta_b",
                    );
                }
                if let Some(parents) = tree_parents {
                    let tape = pbs
                        .dn_s_tape
                        .as_ref()
                        .expect("tree-aware LA requires dn_s_tape scratch");
                    let tree = match dn_state.quant {
                        StateQuant::FP32 => Gpu::gated_delta_net_f32_tree_batch_seq,
                        StateQuant::FP16 => Gpu::gated_delta_net_f16_tree_batch_seq,
                    };
                    tree(
                        gpu,
                        &pbs.dn_q_batch,
                        &pbs.dn_k_batch,
                        &pbs.dn_v_batch,
                        &pbs.dn_alpha_batch,
                        &pbs.dn_beta_batch,
                        &dn_state.s_matrices[delta_layer_idx],
                        tape,
                        parents,
                        &pbs.dn_attn_out_batch,
                        n,
                        n_v_heads,
                        config.linear_value_head_dim,
                    )?;
                } else if matches!(dn_state.quant, StateQuant::FP32)
                    && hipfire_rdna::gdn_chunk::chunk_enabled()
                {
                    gpu.gated_delta_net_f32_chunk(
                        &pbs.dn_q_batch,
                        &pbs.dn_k_batch,
                        &pbs.dn_v_batch,
                        &pbs.dn_alpha_batch,
                        &pbs.dn_beta_batch,
                        &dn_state.s_matrices[delta_layer_idx],
                        &pbs.dn_attn_out_batch,
                        n,
                        n_v_heads,
                        config.linear_value_head_dim,
                    )?;
                } else if matches!(dn_state.quant, StateQuant::FP32) {
                    // Was MISSING: this MoE branch fell straight through to the
                    // Q8 kernels regardless of state precision, reading FP32
                    // state as int8 and a [n_heads] scales vector as
                    // [n_heads × head_dim]. Only unreached because batched
                    // prefill is declined for MoE models (`pbs_eligible`).
                    gpu.gated_delta_net_f32_batch_seq(
                        &pbs.dn_q_batch,
                        &pbs.dn_k_batch,
                        &pbs.dn_v_batch,
                        &pbs.dn_alpha_batch,
                        &pbs.dn_beta_batch,
                        &dn_state.s_matrices[delta_layer_idx],
                        &pbs.dn_attn_out_batch,
                        n,
                        n_v_heads,
                        config.linear_value_head_dim,
                    )?;
                } else if use_gdn_per_token {
                    for step in 0..n {
                        let q = pbs.dn_q_batch.sub_offset(step * v_dim, v_dim);
                        let k = pbs.dn_k_batch.sub_offset(step * v_dim, v_dim);
                        let v = pbs.dn_v_batch.sub_offset(step * v_dim, v_dim);
                        let alpha = pbs.dn_alpha_batch.sub_offset(step * n_v_heads, n_v_heads);
                        let beta = pbs.dn_beta_batch.sub_offset(step * n_v_heads, n_v_heads);
                        let out = pbs.dn_attn_out_batch.sub_offset(step * v_dim, v_dim);
                        gpu.gated_delta_net_f16_batch_seq(
                            &q,
                            &k,
                            &v,
                            &alpha,
                            &beta,
                            &dn_state.s_matrices[delta_layer_idx],
                            &out,
                            1,
                            n_v_heads,
                            config.linear_value_head_dim,
                        )?;
                    }
                } else {
                    gpu.gated_delta_net_f16_batch_seq(
                        &pbs.dn_q_batch,
                        &pbs.dn_k_batch,
                        &pbs.dn_v_batch,
                        &pbs.dn_alpha_batch,
                        &pbs.dn_beta_batch,
                        &dn_state.s_matrices[delta_layer_idx],
                        &pbs.dn_attn_out_batch,
                        n,
                        n_v_heads,
                        config.linear_value_head_dim,
                    )?;
                }
                debug_stop_after!("gdn", layer_idx);
                if let Some(tape) = gdn_tape.as_ref() {
                    let v_row_bytes = tape.v_dim * 4;
                    let off_v = tape_offset * v_row_bytes;
                    gpu.memcpy_dtod_at_auto(
                        &tape.attn_out_bufs[delta_layer_idx].buf,
                        off_v,
                        &pbs.dn_attn_out_batch.buf,
                        0,
                        n * v_row_bytes,
                    )?;
                    // #18: the GDN recurrence for this layer already ran above and
                    // advanced `dn_state.s_matrices[delta_layer_idx]` IN PLACE. The
                    // former re-dispatch here ran the same recurrence a SECOND time
                    // over the same tokens, double-advancing the state and clobbering
                    // `dn_attn_out_batch` with a value computed from the doubly-
                    // advanced state. The tape copy above must stay; the re-dispatch
                    // must not.
                    // DIAG: dump GDN attention output at layer 0
                    if layer_idx == 0 {
                        dump_hidden_localize(
                            gpu,
                            &pbs.dn_attn_out_batch,
                            n,
                            start_pos,
                            n_v_heads * config.linear_value_head_dim,
                            0,
                            "gdn_b",
                        );
                    }
                }
                gpu.gated_norm_f32_batched(
                    &pbs.dn_attn_out_batch,
                    &pbs.dn_z_batch,
                    &layer.norm_weight,
                    &pbs.dn_normed_batch,
                    n_v_heads,
                    config.linear_value_head_dim,
                    config.norm_eps,
                    n,
                )?;
                debug_stop_after!("gated_norm", layer_idx);
                if let Some(tape) = gdn_tape.as_ref() {
                    let v_row_bytes = tape.v_dim * 4;
                    let off_v = tape_offset * v_row_bytes;
                    gpu.memcpy_dtod_at_auto(
                        &tape.normed_bufs[delta_layer_idx].buf,
                        off_v,
                        &pbs.dn_normed_batch.buf,
                        0,
                        n * v_row_bytes,
                    )?;
                }
                if let Some(tape) = gdn_tape.as_ref() {
                    let hidden_row_bytes = tape.x_in_dim * 4;
                    let off_hidden = tape_offset * hidden_row_bytes;
                    gpu.memcpy_dtod_at_auto(
                        &tape.wo_residual_in_bufs[delta_layer_idx].buf,
                        off_hidden,
                        &pbs.x_batch.buf,
                        0,
                        n * hidden_row_bytes,
                    )?;
                }
                // wo + residual. Q8 wo lands un-rotated (Q8 weights were
                // quantized against un-rotated activations); MQ4/MQ6 wo
                // require FWHT(awq_scale-adjusted) rotation. Mirrors the
                // dense LA wo dispatch (qwen35.rs:5000-5043) — the MQ6
                // branch is required for AWQ A3B where 4/40 LA layers
                // ship MQ6 wo and would otherwise corrupt the residual
                // stream when dispatched through the HFQ4 kernel against
                // 200 B/group MQ6-layout bytes.
                let dn_wo_is_q8 = matches!(layer.wo.gpu_dtype, DType::Q8_0);
                let dn_wo_is_6bit = matches!(layer.wo.gpu_dtype, DType::MQ6G256 | DType::HFQ6G256);
                let dn_wo_is_paro = matches!(layer.wo.gpu_dtype, DType::ParoQ4G128);
                let dn_wo_input = if dn_wo_is_q8 {
                    &pbs.dn_normed_batch
                } else if dn_wo_is_paro {
                    // PARO wo: rotate dn_normed by wo's own Givens tables
                    // into dn_normed_rot_batch. Same scratch layout as MQ4
                    // (since dn_normed_rot_batch is unused on the Q8 path).
                    let paro_wo = layer.wo.paro.as_ref().unwrap_or_else(|| {
                        panic!("ParoQ4G128 wo missing paro metadata at LA layer {layer_idx}")
                    });
                    gpu.givens_rotate_to(
                        &pbs.dn_normed_batch,
                        &pbs.dn_normed_rot_batch,
                        &paro_wo.pairs,
                        &paro_wo.theta,
                        &paro_wo.channel_scales,
                        n,
                        layer.wo.k,
                        paro_wo.krot as usize,
                    )?;
                    &pbs.dn_normed_rot_batch
                } else {
                    // F2: AWQ-aware rotate for linear_attn wo (out_proj) input.
                    rotate_x_mq_batched_for(
                        gpu,
                        &layer.wo,
                        &pbs.dn_normed_batch,
                        &pbs.dn_normed_rot_batch,
                        layer.wo.k,
                        n,
                    )?;
                    &pbs.dn_normed_rot_batch
                };
                if let Some(tape) = gdn_tape.as_ref() {
                    let v_row_bytes = tape.v_dim * 4;
                    let off_v = tape_offset * v_row_bytes;
                    gpu.memcpy_dtod_at_auto(
                        &tape.wo_input_bufs[delta_layer_idx].buf,
                        off_v,
                        &dn_wo_input.buf,
                        0,
                        n * v_row_bytes,
                    )?;
                }
                if dn_wo_is_6bit {
                    run_residual_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmHfq6G256Residual,
                        &layer.wo.buf,
                        layer.wo.gpu_dtype,
                        dn_wo_input,
                        &pbs.x_batch,
                        layer.wo.m,
                        layer.wo.k,
                        n,
                    )?;
                } else if dn_wo_is_q8 && q8_wmma_arch {
                    let x_n = pbs.x_batch.sub_offset(0, n * layer.wo.m);
                    run_residual_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmQ8_0ResidualWmma,
                        &layer.wo.buf,
                        layer.wo.gpu_dtype,
                        dn_wo_input,
                        &x_n,
                        layer.wo.m,
                        layer.wo.k,
                        n,
                    )?;
                } else if dn_wo_is_q8 {
                    // Non-WMMA Q8: gemm into a scratch then add into x_batch.
                    // Reuse `dn_normed_rot_batch` (free since the MQ4 rotate
                    // path didn't run here) as the GEMM scratch.
                    let scratch = pbs.dn_normed_rot_batch.sub_offset(0, n * layer.wo.m);
                    run_plain_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmQ8_0BatchedChunked,
                        &layer.wo.buf,
                        layer.wo.gpu_dtype,
                        dn_wo_input,
                        &scratch,
                        layer.wo.m,
                        layer.wo.k,
                        n,
                    )?;
                    let x_n = pbs.x_batch.sub_offset(0, n * layer.wo.m);
                    gpu.add_inplace_f32(&x_n, &scratch)?;
                } else if dn_wo_is_paro {
                    // PARO wo residual: HFQ4G128 batched GEMM into scratch,
                    // then add into x_batch. Reuse x_norm_batch (free at
                    // this point — used earlier for the QKVZA stage; not
                    // needed for the rest of this layer) as the scratch.
                    let scratch = pbs.x_norm_batch.sub_offset(0, n * layer.wo.m);
                    run_plain_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmHfq4G128,
                        &layer.wo.buf,
                        layer.wo.gpu_dtype,
                        dn_wo_input,
                        &scratch,
                        layer.wo.m,
                        layer.wo.k,
                        n,
                    )?;
                    let x_n = pbs.x_batch.sub_offset(0, n * layer.wo.m);
                    gpu.add_inplace_f32(&x_n, &scratch)?;
                } else if matches!(layer.wo.gpu_dtype, DType::OqCompactG256) {
                    // Compact-resident Opus o_proj: identical W8A8 math to the oq8
                    // arm below, decoding the OqPlusCompact blocks in-kernel.
                    let bs = super::prefill_batch::oq_compact_block_stride(&layer.wo)?;
                    gpu.gemm_oq_compact_residual_act_batched(
                        &layer.wo.buf,
                        dn_wo_input,
                        &pbs.x_batch,
                        layer.wo.m,
                        layer.wo.k,
                        n,
                        bs,
                    )?;
                } else {
                    run_residual_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmHfq4G256Residual,
                        &layer.wo.buf,
                        layer.wo.gpu_dtype,
                        dn_wo_input,
                        &pbs.x_batch,
                        layer.wo.m,
                        layer.wo.k,
                        n,
                    )?;
                }
                debug_stop_after!("wo", layer_idx);
                if let Some(tape) = gdn_tape.as_ref() {
                    let hidden_row_bytes = tape.x_in_dim * 4;
                    let off_hidden = tape_offset * hidden_row_bytes;
                    gpu.memcpy_dtod_at_auto(
                        &tape.attn_residual_bufs[delta_layer_idx].buf,
                        off_hidden,
                        &pbs.x_batch.buf,
                        0,
                        n * hidden_row_bytes,
                    )?;
                }

                // Batched MoE FFN replaces the dense (rmsnorm + gate+up +
                // silu_mul + w_down) block. Takes pbs.x_batch as input AND
                // accumulates the FFN output residual back into it via the
                // batched indexed down kernel's atomicAdd path.
                if debug_stop_after_la_layer == Some(layer_idx) {
                    return Ok(());
                }
                if let Some(tape) = gdn_tape.as_ref() {
                    let hidden_row_bytes = tape.x_in_dim * 4;
                    let off_hidden = tape_offset * hidden_row_bytes;
                    gpu.memcpy_dtod_at_auto(
                        &tape.ffn_input_bufs[delta_layer_idx].buf,
                        off_hidden,
                        &pbs.x_batch.buf,
                        0,
                        n * hidden_row_bytes,
                    )?;
                }
                prefill_moe_ffn_body_batched(
                    gpu,
                    weights.pager.as_ref(),
                    &layer.ffn,
                    &layer.ffn_norm,
                    config,
                    pbs,
                    n,
                    layer_idx,
                    &ctx,
                    routed_out,
                    None,
                )?;

                // Post-layer hidden extract for the DFlash draft path.
                if let Some(rb) = hidden_rb {
                    if let Some(slot) = rb.extract_slot(layer_idx) {
                        rb.write_rows_to_staging(gpu, slot, &pbs.x_batch, n)?;
                    }
                }
                delta_layer_idx += 1;
            }

            (LayerWeights::FullAttnMoe(layer), LayerType::FullAttention) if fa_batched_ok => {
                // Batched MoE FA layer. FA body is the same as FullAttn
                // (rmsnorm + qkv + deinterleave + q/k norm + RoPE +
                // kv_write + attention + sigmoid_mul + wo+residual);
                // only the FFN differs. Duplicated inline — will be
                // consolidated with the dense FA batched body once the
                // MoE path is proven byte-exact.
                let kv_dim = config.n_kv_heads * config.head_dim;
                let q_dim = config.n_heads * config.head_dim;
                // This body is unreachable for MQ3 / MQ3-Lloyd weights —
                // the upstream `mq3_in_moe` guard at the top of
                // `forward_prefill_batch_with_pbs` rejects any MoE layer
                // with MQ3/Lloyd-MQ3 weights anywhere (attention OR FFN),
                // mirroring the captured-path guard at line 3367+. So
                // `layer.wq.gpu_dtype` is restricted to MQ4G256 / HFQ4G256
                // / MQ6G256 / HFQ6G256 here. Adding MQ3 to the matcher AND
                // the QKV dispatch is insufficient — the wo path below
                // (line 5320) is hardcoded MQ4 too — so the all-or-nothing
                // wiring lives in a separate PR (see followup issue).
                // Opus (Oq4/Oq8/OqCompact) weights are FWHT(+AWQ)-rotated OFFLINE, so the
                // activation must be rotated to match. Leaving them out of this
                // predicate sends an UNROTATED x into an Opus GEMM -- the dense path
                // records that outcome as "garbage: PPL 3.5e6".
                let qkv_is_mq = matches!(
                    layer.wq.gpu_dtype,
                    DType::MQ4G256
                        | DType::MQ6G256
                        | DType::Oq4G256
                        | DType::Oq8G256
                        | DType::OqCompactG256
                );
                let qkv_is_6bit = matches!(layer.wq.gpu_dtype, DType::MQ6G256 | DType::HFQ6G256);
                let qkv_is_q8 = matches!(layer.wq.gpu_dtype, DType::Q8_0);
                // Phase 1.6 (PARO FullAttnMoe): wq/wk/wv are ParoQ4G128
                // (each with its own Givens rotation tables). The fused-QKV
                // kernels can't handle this — they assume one shared
                // rotation. Unfused 3-way dispatch (rotate + gemm_hfq4g128
                // per projection) matches the LA QKVZA Phase 1.5 pattern.
                let qkv_is_paro = matches!(layer.wq.gpu_dtype, DType::ParoQ4G128);
                // Fused QKV requires uniform dtype — see issue #249 for
                // the dense FA variant. Gate the same way here.
                let q8_wmma_arch = gpu.arch_caps.has_wmma();
                let qkv_same_dtype = layer.wk.gpu_dtype == layer.wq.gpu_dtype
                    && layer.wv.gpu_dtype == layer.wq.gpu_dtype;

                if qkv_is_mq {
                    // AWQ-aware: next linear is wq (Q/K/V share input → same AWQ scale).
                    fused_rmsnorm_rotate_mq_batched_for(
                        gpu,
                        &pbs.x_batch,
                        &layer.attn_norm,
                        &layer.wq,
                        &pbs.x_rot_batch,
                        dim,
                        config.norm_eps,
                        n,
                    )?;
                } else if qkv_is_paro {
                    // PARO: rmsnorm into x_norm_batch (un-rotated). x_rot_batch
                    // is reused as the per-weight rotation scratch.
                    gpu.rmsnorm_batched(
                        &pbs.x_batch,
                        &layer.attn_norm,
                        &pbs.x_norm_batch,
                        n,
                        dim,
                        config.norm_eps,
                    )?;
                } else {
                    gpu.rmsnorm_batched(
                        &pbs.x_batch,
                        &layer.attn_norm,
                        &pbs.x_rot_batch,
                        n,
                        dim,
                        config.norm_eps,
                    )?;
                }
                if qkv_is_paro {
                    // PARO 3-way unfused dispatch (wq, wk, wv each with own
                    // Givens rotation). Same shape outputs as the fused
                    // paths: fa_q_full_batch, fa_k_batch, fa_v_batch.
                    let paro_wq = layer.wq.paro.as_ref().unwrap_or_else(|| {
                        panic!("ParoQ4G128 wq missing paro metadata at FA layer {layer_idx}")
                    });
                    let paro_wk = layer.wk.paro.as_ref().unwrap_or_else(|| {
                        panic!("ParoQ4G128 wk missing paro metadata at FA layer {layer_idx}")
                    });
                    let paro_wv = layer.wv.paro.as_ref().unwrap_or_else(|| {
                        panic!("ParoQ4G128 wv missing paro metadata at FA layer {layer_idx}")
                    });
                    // wq
                    gpu.givens_rotate_to(
                        &pbs.x_norm_batch,
                        &pbs.x_rot_batch,
                        &paro_wq.pairs,
                        &paro_wq.theta,
                        &paro_wq.channel_scales,
                        n,
                        dim,
                        paro_wq.krot as usize,
                    )?;
                    run_plain_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmHfq4G128,
                        &layer.wq.buf,
                        layer.wq.gpu_dtype,
                        &pbs.x_rot_batch,
                        &pbs.fa_q_full_batch,
                        layer.wq.m,
                        layer.wq.k,
                        n,
                    )?;
                    // wk
                    gpu.givens_rotate_to(
                        &pbs.x_norm_batch,
                        &pbs.x_rot_batch,
                        &paro_wk.pairs,
                        &paro_wk.theta,
                        &paro_wk.channel_scales,
                        n,
                        dim,
                        paro_wk.krot as usize,
                    )?;
                    run_plain_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmHfq4G128,
                        &layer.wk.buf,
                        layer.wk.gpu_dtype,
                        &pbs.x_rot_batch,
                        &pbs.fa_k_batch,
                        layer.wk.m,
                        layer.wk.k,
                        n,
                    )?;
                    // wv
                    gpu.givens_rotate_to(
                        &pbs.x_norm_batch,
                        &pbs.x_rot_batch,
                        &paro_wv.pairs,
                        &paro_wv.theta,
                        &paro_wv.channel_scales,
                        n,
                        dim,
                        paro_wv.krot as usize,
                    )?;
                    run_plain_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmHfq4G128,
                        &layer.wv.buf,
                        layer.wv.gpu_dtype,
                        &pbs.x_rot_batch,
                        &pbs.fa_v_batch,
                        layer.wv.m,
                        layer.wv.k,
                        n,
                    )?;
                } else if qkv_is_6bit && qkv_same_dtype {
                    run_fused_qkv_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::FusedQkvHfq6G256,
                        &layer.wq.buf,
                        &layer.wk.buf,
                        &layer.wv.buf,
                        &pbs.x_rot_batch,
                        &pbs.fa_q_full_batch,
                        &pbs.fa_k_batch,
                        &pbs.fa_v_batch,
                        layer.wq.m,
                        layer.wk.m,
                        layer.wv.m,
                        layer.wq.k,
                        n,
                    )?;
                } else if qkv_is_q8 && q8_wmma_arch && qkv_same_dtype {
                    debug_assert!(
                        matches!(layer.wk.gpu_dtype, DType::Q8_0)
                            && matches!(layer.wv.gpu_dtype, DType::Q8_0),
                        "FAMoe qkv Q8 WMMA dispatch requires all of wq/wk/wv to be Q8_0",
                    );
                    run_fused_qkv_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::FusedQkvQ8_0,
                        &layer.wq.buf,
                        &layer.wk.buf,
                        &layer.wv.buf,
                        &pbs.x_rot_batch,
                        &pbs.fa_q_full_batch,
                        &pbs.fa_k_batch,
                        &pbs.fa_v_batch,
                        layer.wq.m,
                        layer.wk.m,
                        layer.wv.m,
                        layer.wq.k,
                        n,
                    )?;
                } else if qkv_is_q8 && qkv_same_dtype {
                    run_plain_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmQ8_0BatchedChunked,
                        &layer.wq.buf,
                        layer.wq.gpu_dtype,
                        &pbs.x_rot_batch,
                        &pbs.fa_q_full_batch,
                        layer.wq.m,
                        layer.wq.k,
                        n,
                    )?;
                    run_plain_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmQ8_0BatchedChunked,
                        &layer.wk.buf,
                        layer.wk.gpu_dtype,
                        &pbs.x_rot_batch,
                        &pbs.fa_k_batch,
                        layer.wk.m,
                        layer.wk.k,
                        n,
                    )?;
                    run_plain_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmQ8_0BatchedChunked,
                        &layer.wv.buf,
                        layer.wv.gpu_dtype,
                        &pbs.x_rot_batch,
                        &pbs.fa_v_batch,
                        layer.wv.m,
                        layer.wv.k,
                        n,
                    )?;
                } else if matches!(layer.wq.gpu_dtype, DType::OqCompactG256) && qkv_same_dtype {
                    // Compact-resident Opus: one quantize of the shared rotated
                    // activation, then one compact GEMM per projection.
                    gpu.quantize_act_oq8_batched_interleaved(
                        &pbs.x_rot_batch,
                        layer.wq.m,
                        layer.wq.k,
                        n,
                    )?;
                    for (w, y) in [
                        (&layer.wq, &pbs.fa_q_full_batch),
                        (&layer.wk, &pbs.fa_k_batch),
                        (&layer.wv, &pbs.fa_v_batch),
                    ] {
                        let bs = super::prefill_batch::oq_compact_block_stride(w)?;
                        gpu.gemm_oq_compact_grouped_prequant(&w.buf, y, w.m, w.k, n, bs)?;
                    }
                } else if qkv_same_dtype {
                    run_fused_qkv_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::FusedQkvHfq4G256,
                        &layer.wq.buf,
                        &layer.wk.buf,
                        &layer.wv.buf,
                        &pbs.x_rot_batch,
                        &pbs.fa_q_full_batch,
                        &pbs.fa_k_batch,
                        &pbs.fa_v_batch,
                        layer.wq.m,
                        layer.wk.m,
                        layer.wv.m,
                        layer.wq.k,
                        n,
                    )?;
                } else {
                    // Mixed-format fallback (issue #249). batched_gemm_single_weight
                    // covers MQ4/HFQ4 + MQ6/HFQ6 + Q8_0; mixed-Q8/MQ4 within FAMoe
                    // routes here.
                    batched_gemm_single_weight(
                        gpu,
                        &layer.wq,
                        &pbs.x_rot_batch,
                        &pbs.fa_q_full_batch,
                        n,
                    )?;
                    batched_gemm_single_weight(
                        gpu,
                        &layer.wk,
                        &pbs.x_rot_batch,
                        &pbs.fa_k_batch,
                        n,
                    )?;
                    batched_gemm_single_weight(
                        gpu,
                        &layer.wv,
                        &pbs.x_rot_batch,
                        &pbs.fa_v_batch,
                        n,
                    )?;
                }
                qwen35_materialize_fa_q(
                    gpu,
                    config,
                    &pbs.fa_q_full_batch,
                    &pbs.fa_q_batch,
                    &pbs.fa_gate_batch,
                    n,
                )?;
                gpu.rmsnorm_batched(
                    &pbs.fa_q_batch,
                    &layer.q_norm,
                    &pbs.fa_q_batch,
                    n * config.n_heads,
                    config.head_dim,
                    config.norm_eps,
                )?;
                gpu.rmsnorm_batched(
                    &pbs.fa_k_batch,
                    &layer.k_norm,
                    &pbs.fa_k_batch,
                    n * config.n_kv_heads,
                    config.head_dim,
                    config.norm_eps,
                )?;
                if hipfire_runtime::triattn::tap_enabled() {
                    let gpu_handled =
                        hipfire_runtime::triattn::record_prerope_q_batch_gpu_if_applicable(
                            gpu,
                            layer_idx,
                            &pbs.fa_q_batch.buf,
                            n,
                            config.n_heads,
                            config.head_dim,
                        )?;
                    if !gpu_handled {
                        let n_q = config.n_heads * config.head_dim;
                        let q_cpu = gpu.download_f32(&pbs.fa_q_batch)?;
                        if hipfire_runtime::triattn::tap_needs_k() {
                            let n_k = config.n_kv_heads * config.head_dim;
                            let k_cpu = gpu.download_f32(&pbs.fa_k_batch)?;
                            for b in 0..n {
                                hipfire_runtime::triattn::record_prerope_qk(
                                    layer_idx,
                                    &q_cpu[b * n_q..(b + 1) * n_q],
                                    Some(&k_cpu[b * n_k..(b + 1) * n_k]),
                                );
                            }
                        } else {
                            for b in 0..n {
                                hipfire_runtime::triattn::record_prerope_q(
                                    layer_idx,
                                    &q_cpu[b * n_q..(b + 1) * n_q],
                                );
                            }
                        }
                    }
                }
                // Path B pre-RoPE K capture (MoE FA variant). See same
                // block in the FullAttn branch for rationale.
                if let Some(slots) = tree_verify.as_ref().and_then(|c| c.pre_rope_k_capture) {
                    if let Some(slot) = slots.get(fa_layer_idx) {
                        let kv_dim = config.n_kv_heads * config.head_dim;
                        let n_bytes = n * kv_dim * 4;
                        gpu.memcpy_dtod_at_auto(&slot.buf, 0, &pbs.fa_k_batch.buf, 0, n_bytes)?;
                    }
                }
                let n_rot = (config.head_dim as f32 * config.partial_rotary_factor) as usize;
                // pbs.positions stays physical for the KV write below; the
                // offset rotates new Q/K at absolute phase after compaction.
                gpu.rope_partial_interleaved_f32_batched(
                    &pbs.fa_q_batch,
                    &pbs.fa_k_batch,
                    &pbs.positions,
                    config.n_heads,
                    config.n_kv_heads,
                    config.head_dim,
                    n_rot,
                    n_rot,
                    config.rope_theta,
                    n,
                    kv_cache.compact_offset as i32,
                )?;

                let use_kld_direct_f16kv_attention = kld_direct_f16kv_attention_eligible(
                    gpu,
                    kv_cache,
                    config,
                    start_pos,
                    tree_verify.as_ref(),
                );
                let use_kld_fp32_gqa4_attention = kld_fp32_gqa4_attention_eligible(
                    gpu,
                    kv_cache,
                    config,
                    start_pos,
                    tree_verify.as_ref(),
                    n,
                );

                if kv_cache.quant_kvarn {
                    // KVarN K = 4-bit block records + fp16 window (not a contiguous
                    // buffer), so the generic batched writes below would fault.
                    // kvarn_attend owns the batched write (window append + 128-block
                    // flush) AND the fused causal flash together — the same entry
                    // point decode uses (mod.rs), which already supports n>1.
                    // Rotation + tiles scratch mirror the decode path. The paired
                    // attention step below is a no-op for kvarn (done here).
                    static PREFILL_KVARN_ROTATE: std::sync::OnceLock<bool> =
                        std::sync::OnceLock::new();
                    let kvarn_rotate = *PREFILL_KVARN_ROTATE.get_or_init(|| {
                        std::env::var("HIPFIRE_KVARN_ROTATE").ok().as_deref() != Some("0")
                    });
                    if kvarn_rotate && config.head_dim == 256 {
                        gpu.rotate_x_mq_batched(
                            &pbs.fa_k_batch,
                            &pbs.fa_k_batch,
                            config.n_kv_heads * config.head_dim,
                            n,
                        )?;
                        gpu.rotate_x_mq_batched(
                            &pbs.fa_q_batch,
                            &pbs.fa_q_batch,
                            config.n_heads * config.head_dim,
                            n,
                        )?;
                    }
                    if kv_cache.kvarn_tiles.is_none() {
                        let tiles = gpu.alloc_tensor(
                            &[config.n_kv_heads * config.head_dim * 128],
                            DType::F32,
                        )?;
                        kv_cache.kvarn_tiles = Some(tiles);
                    }
                    let kvarn_tree_bias = tree_verify.as_ref().map(|c| c.attn_bias);
                    let (kvarn_block_start, kvarn_block_cols) = match tree_verify.as_ref() {
                        Some(_) => (start_pos, n),
                        None => (0, 0),
                    };
                    gpu.kvarn_attend(
                        &kv_cache.k_gpu[layer_idx],
                        &kv_cache.k_window[layer_idx],
                        &kv_cache.v_gpu[layer_idx],
                        &pbs.fa_q_batch,
                        &pbs.fa_k_batch,
                        &pbs.fa_v_batch,
                        &pbs.positions,
                        &pbs.fa_attn_out_batch,
                        &s.flash_partials,
                        kv_cache.kvarn_tiles.as_ref().unwrap(),
                        n,
                        start_pos,
                        config.n_heads,
                        config.n_kv_heads,
                        config.head_dim,
                        kv_cache.physical_cap,
                        kvarn_tree_bias,
                        kvarn_block_start,
                        kvarn_block_cols,
                        kv_cache.kvarn_bits,
                    )?;
                } else if kv_cache.quant_asym4 {
                    let ct = givens_cos_view!().unwrap();
                    let st = givens_sin_view!().unwrap();
                    if kv_cache.quant_fwht {
                        gpu.kv_cache_write_fwht4_batched(
                            &kv_cache.k_gpu[layer_idx],
                            &kv_cache.v_gpu[layer_idx],
                            &pbs.fa_k_batch,
                            &pbs.fa_v_batch,
                            &pbs.positions,
                            ct,
                            st,
                            config.n_kv_heads,
                            config.head_dim,
                            n,
                            0,
                        )?;
                    } else {
                        gpu.kv_cache_write_asym4_batched(
                            &kv_cache.k_gpu[layer_idx],
                            &kv_cache.v_gpu[layer_idx],
                            &pbs.fa_k_batch,
                            &pbs.fa_v_batch,
                            &pbs.positions,
                            ct,
                            st,
                            config.n_kv_heads,
                            config.head_dim,
                            n,
                        )?;
                    }
                } else if kv_cache.quant_asym3 {
                    let ct = givens_cos_view!().unwrap();
                    let st = givens_sin_view!().unwrap();
                    if kv_cache.quant_fwht {
                        gpu.kv_cache_write_fwht3_batched(
                            &kv_cache.k_gpu[layer_idx],
                            &kv_cache.v_gpu[layer_idx],
                            &pbs.fa_k_batch,
                            &pbs.fa_v_batch,
                            &pbs.positions,
                            ct,
                            st,
                            config.n_kv_heads,
                            config.head_dim,
                            n,
                            0,
                        )?;
                    } else {
                        gpu.kv_cache_write_asym3_batched(
                            &kv_cache.k_gpu[layer_idx],
                            &kv_cache.v_gpu[layer_idx],
                            &pbs.fa_k_batch,
                            &pbs.fa_v_batch,
                            &pbs.positions,
                            ct,
                            st,
                            config.n_kv_heads,
                            config.head_dim,
                            n,
                        )?;
                    }
                } else if kv_cache.quant_asym2 {
                    let ct = givens_cos_view!().unwrap();
                    let st = givens_sin_view!().unwrap();
                    if kv_cache.quant_fwht {
                        gpu.kv_cache_write_fwht2_batched(
                            &kv_cache.k_gpu[layer_idx],
                            &kv_cache.v_gpu[layer_idx],
                            &pbs.fa_k_batch,
                            &pbs.fa_v_batch,
                            &pbs.positions,
                            ct,
                            st,
                            config.n_kv_heads,
                            config.head_dim,
                            n,
                            0,
                        )?;
                    } else {
                        gpu.kv_cache_write_asym2_batched(
                            &kv_cache.k_gpu[layer_idx],
                            &kv_cache.v_gpu[layer_idx],
                            &pbs.fa_k_batch,
                            &pbs.fa_v_batch,
                            &pbs.positions,
                            ct,
                            st,
                            config.n_kv_heads,
                            config.head_dim,
                            n,
                        )?;
                    }
                } else if kv_cache.quant_q8 {
                    gpu.kv_cache_write_q8_0_batched(
                        &kv_cache.k_gpu[layer_idx],
                        &pbs.fa_k_batch,
                        &pbs.positions,
                        config.n_kv_heads,
                        config.head_dim,
                        n,
                    )?;
                    gpu.kv_cache_write_q8_0_batched(
                        &kv_cache.v_gpu[layer_idx],
                        &pbs.fa_v_batch,
                        &pbs.positions,
                        config.n_kv_heads,
                        config.head_dim,
                        n,
                    )?;
                } else if !use_kld_direct_f16kv_attention && !use_kld_fp32_gqa4_attention {
                    gpu.kv_cache_write_f32_batched(
                        &kv_cache.k_gpu[layer_idx],
                        &pbs.fa_k_batch,
                        &pbs.positions,
                        config.n_kv_heads * config.head_dim,
                        n,
                    )?;
                    gpu.kv_cache_write_f32_batched(
                        &kv_cache.v_gpu[layer_idx],
                        &pbs.fa_v_batch,
                        &pbs.positions,
                        config.n_kv_heads * config.head_dim,
                        n,
                    )?;
                }
                const LDS_CTX_LIMIT: usize = 15000;
                let tree_bias = tree_verify.as_ref().map(|c| c.attn_bias);
                // Batched KV write + flash attention (via dispatch).
                let is_tree = tree_verify.is_some();
                let (block_start, block_cols) = match tree_verify.as_ref() {
                    Some(_) => (start_pos, n),
                    None => (0, 0),
                };
                if kv_cache.quant_kvarn {
                    // KVarN did the fused write + causal flash in the write step above.
                } else if use_kld_direct_f16kv_attention {
                    gpu.attention_dflash_wmma_causal_f32(
                        &pbs.fa_q_batch,
                        &pbs.fa_k_batch,
                        &pbs.fa_v_batch,
                        &pbs.fa_attn_out_batch,
                        n,
                        n,
                        config.n_heads,
                        config.n_kv_heads,
                        config.head_dim,
                    )?;
                } else if use_kld_fp32_gqa4_attention {
                    gpu.attention_f32_batched_gqa4(
                        &pbs.fa_q_batch,
                        &pbs.fa_k_batch,
                        &pbs.fa_v_batch,
                        &pbs.fa_attn_out_batch,
                        &pbs.positions,
                        config.n_heads,
                        config.n_kv_heads,
                        config.head_dim,
                        n,
                        n,
                    )?;
                } else if kv_cache.quant_asym4 {
                    let ct = givens_cos_view!().unwrap();
                    let st = givens_sin_view!().unwrap();
                    if kv_cache.quant_fwht {
                        gpu.attention_flash_fwht4_batched_masked(
                            &pbs.fa_q_batch,
                            &kv_cache.k_gpu[layer_idx],
                            &kv_cache.v_gpu[layer_idx],
                            &pbs.fa_attn_out_batch,
                            &pbs.positions,
                            ct,
                            st,
                            config.n_heads,
                            config.n_kv_heads,
                            config.head_dim,
                            kv_cache.physical_cap,
                            max_ctx_len,
                            n,
                            &s.flash_partials,
                            tree_bias,
                            block_start,
                            block_cols,
                            0,
                        )?;
                    } else {
                        gpu.attention_flash_asym4_batched_masked(
                            &pbs.fa_q_batch,
                            &kv_cache.k_gpu[layer_idx],
                            &kv_cache.v_gpu[layer_idx],
                            &pbs.fa_attn_out_batch,
                            &pbs.positions,
                            ct,
                            st,
                            config.n_heads,
                            config.n_kv_heads,
                            config.head_dim,
                            kv_cache.physical_cap,
                            max_ctx_len,
                            n,
                            &s.flash_partials,
                            tree_bias,
                            block_start,
                            block_cols,
                        )?;
                    }
                } else if kv_cache.quant_asym3 {
                    let ct = givens_cos_view!().unwrap();
                    let st = givens_sin_view!().unwrap();
                    if kv_cache.quant_fwht {
                        gpu.attention_flash_fwht3_batched_masked(
                            &pbs.fa_q_batch,
                            &kv_cache.k_gpu[layer_idx],
                            &kv_cache.v_gpu[layer_idx],
                            &pbs.fa_attn_out_batch,
                            &pbs.positions,
                            ct,
                            st,
                            config.n_heads,
                            config.n_kv_heads,
                            config.head_dim,
                            kv_cache.physical_cap,
                            max_ctx_len,
                            n,
                            &s.flash_partials,
                            tree_bias,
                            block_start,
                            block_cols,
                            0,
                        )?;
                    } else {
                        gpu.attention_flash_asym3_batched_masked(
                            &pbs.fa_q_batch,
                            &kv_cache.k_gpu[layer_idx],
                            &kv_cache.v_gpu[layer_idx],
                            &pbs.fa_attn_out_batch,
                            &pbs.positions,
                            ct,
                            st,
                            config.n_heads,
                            config.n_kv_heads,
                            config.head_dim,
                            kv_cache.physical_cap,
                            max_ctx_len,
                            n,
                            &s.flash_partials,
                            tree_bias,
                            block_start,
                            block_cols,
                        )?;
                    }
                } else if kv_cache.quant_asym2 {
                    assert!(
                        tree_verify.is_none(),
                        "tree-verify mode not supported on asym2 KV (use asym3)",
                    );
                    let ct = givens_cos_view!().unwrap();
                    let st = givens_sin_view!().unwrap();
                    if kv_cache.quant_fwht {
                        gpu.attention_flash_fwht2_batched(
                            &pbs.fa_q_batch,
                            &kv_cache.k_gpu[layer_idx],
                            &kv_cache.v_gpu[layer_idx],
                            &pbs.fa_attn_out_batch,
                            &pbs.positions,
                            ct,
                            st,
                            config.n_heads,
                            config.n_kv_heads,
                            config.head_dim,
                            kv_cache.physical_cap,
                            max_ctx_len,
                            n,
                            &s.flash_partials,
                            0,
                        )?;
                    } else {
                        gpu.attention_flash_asym2_batched(
                            &pbs.fa_q_batch,
                            &kv_cache.k_gpu[layer_idx],
                            &kv_cache.v_gpu[layer_idx],
                            &pbs.fa_attn_out_batch,
                            &pbs.positions,
                            ct,
                            st,
                            config.n_heads,
                            config.n_kv_heads,
                            config.head_dim,
                            kv_cache.physical_cap,
                            max_ctx_len,
                            n,
                            &s.flash_partials,
                        )?;
                    }
                } else if kv_cache.quant_q8 && max_ctx_len > LDS_CTX_LIMIT {
                    assert!(
                        tree_verify.is_none(),
                        "tree-verify mode hits the long-context Q8 fallback \
                         at max_ctx_len={} > {}; tree blocks should stay small",
                        max_ctx_len,
                        LDS_CTX_LIMIT,
                    );
                    // See dense FullAttn branch above for the i32-vs-f32 slot
                    // rationale; reconstruct positions from the host-side row
                    // position directly.
                    let q_dim_local = config.n_heads * config.head_dim;
                    let pos_buf_tmp = gpu.hip.malloc(4)?;
                    let pos_buf_result = (|| -> HipResult<()> {
                        for b in 0..n {
                            let pos_b = position_at_row(b);
                            let seq_len_b = pos_b + 1;
                            let pos_i32 = pos_b as i32;
                            gpu.hip.memcpy_htod(&pos_buf_tmp, &pos_i32.to_ne_bytes())?;
                            let q_b = pbs.fa_q_batch.sub_offset(b * q_dim_local, q_dim_local);
                            let out_b = pbs
                                .fa_attn_out_batch
                                .sub_offset(b * q_dim_local, q_dim_local);
                            gpu.attention_flash_q8_0(
                                &q_b,
                                &kv_cache.k_gpu[layer_idx],
                                &kv_cache.v_gpu[layer_idx],
                                &out_b,
                                &pos_buf_tmp,
                                seq_len_b,
                                config.n_heads,
                                config.n_kv_heads,
                                config.head_dim,
                                kv_cache.physical_cap,
                                &s.flash_partials,
                            )?;
                        }
                        Ok(())
                    })();
                    let _ = gpu.hip.free(pos_buf_tmp);
                    pos_buf_result?;
                } else if kv_cache.quant_q8 {
                    gpu.attention_q8_0_kv_batched_masked(
                        &pbs.fa_q_batch,
                        &kv_cache.k_gpu[layer_idx],
                        &kv_cache.v_gpu[layer_idx],
                        &pbs.fa_attn_out_batch,
                        &pbs.positions,
                        config.n_heads,
                        config.n_kv_heads,
                        config.head_dim,
                        kv_cache.physical_cap,
                        max_ctx_len,
                        n,
                        tree_bias,
                        block_start,
                        block_cols,
                    )?;
                } else {
                    gpu.attention_f32_batched_masked(
                        &pbs.fa_q_batch,
                        &kv_cache.k_gpu[layer_idx],
                        &kv_cache.v_gpu[layer_idx],
                        &pbs.fa_attn_out_batch,
                        &pbs.positions,
                        config.n_heads,
                        config.n_kv_heads,
                        config.head_dim,
                        kv_cache.physical_cap,
                        max_ctx_len,
                        n,
                        tree_bias,
                        block_start,
                        block_cols,
                    )?;
                }
                qwen35_apply_fa_gate(gpu, config, &pbs.fa_attn_out_batch, &pbs.fa_gate_batch)?;
                // KVarN already did the fused write + causal flash in kvarn_attend
                // above, and KvTierPlan has no KVarN tier -- it falls through to an
                // F32 write that has no batched form and resolves as
                // `no implementation for KvWriteF32`. Pairs with admitting
                // quant_kvarn into `fa_kv_ok`; the dense FullAttn path carries the
                // same guard for the same reason.
                if !kv_cache.quant_kvarn {
                    let tree_bias = tree_verify.as_ref().map(|c| c.attn_bias);
                    let plan = KvTierPlan::derive(KvTierInputs {
                        quant_asym4: kv_cache.quant_asym4,
                        quant_asym3: kv_cache.quant_asym3,
                        quant_asym2: kv_cache.quant_asym2,
                        quant_q8: kv_cache.quant_q8,
                        quant_fwht: kv_cache.quant_fwht,
                        quant_hfq4: false,
                        quant_q4: false,
                        v_mode_bits: 0,
                        pos: start_pos,
                        flash_mode: s.flash_mode as usize,
                        capture_mode: gpu.capture_mode,
                        batch_size: n,
                        is_tree,
                        // TODO: boundary producer not yet populated. Matches the
                        // serial path (qwen35/mod.rs:3191) — `layer_is_boundary` is
                        // `vec![]` at every KvCache constructor and never filled, so
                        // `KvCache::is_boundary()` is always false. Threading it here
                        // would be a no-op AND would imply boundary layers work.
                        // Wire all three sites together when the producer lands.
                        is_boundary: false,
                    })
                    .map_err(|e| HipError::new(0, &e.to_string()))?;
                    let io = AttnParams {
                        q: &pbs.fa_q_batch,
                        k: &pbs.fa_k_batch,
                        v: &pbs.fa_v_batch,
                        k_cache: &kv_cache.k_gpu[layer_idx],
                        v_cache: &kv_cache.v_gpu[layer_idx],
                        k_scales: None,
                        v_scales: None,
                        pos_buf: &s.pos_buf,
                        pos: start_pos,
                        positions: Some(&pbs.positions),
                        n_heads: config.n_heads,
                        n_kv_heads: config.n_kv_heads,
                        head_dim: config.head_dim,
                        physical_cap: kv_cache.physical_cap,
                        batch_size: n,
                        max_ctx_len,
                        flash_partials: Some(&s.flash_partials),
                        givens_cos: kv_cache.givens_cos.as_ref(),
                        givens_sin: kv_cache.givens_sin.as_ref(),
                        tree_bias,
                        block_start,
                        block_cols,
                        output: &pbs.fa_attn_out_batch,
                    };
                    execute_steps(gpu, &ctx, &[Step::Attend { plan, io }])
                        .map_err(|e| HipError::new(0, &e.to_string()))?;
                }
                gpu.sigmoid_mul_f32(&pbs.fa_attn_out_batch, &pbs.fa_gate_batch)?;
                // wo + residual. Mirrors the dense FA wo dispatch at
                // qwen35.rs:5591-5623 — Q8 wo skips rotation (un-rotated
                // input expected); MQ4/MQ6 wo apply FWHT(awq_scale-adjusted).
                // MQ6 wo has its own branch: feeding MQ6 bytes to the MQ4
                // residual kernel would read 200 B/group data as 136 B/group
                // HFQ4 layout and catastrophically mis-stride.
                let fa_wo_is_q8 = matches!(layer.wo.gpu_dtype, DType::Q8_0);
                let fa_wo_is_6bit = matches!(layer.wo.gpu_dtype, DType::MQ6G256 | DType::HFQ6G256);
                // Phase 1.6 (PARO FullAttnMoe wo): own Givens rotation table,
                // 72 B/group HFQ4G128 layout. Rotate fa_attn_out_batch by wo's
                // paro into fa_attn_out_rot_batch, then HFQ4G128 GEMM into a
                // scratch, then add into x_batch.
                let fa_wo_is_paro = matches!(layer.wo.gpu_dtype, DType::ParoQ4G128);
                let fa_wo_input = if fa_wo_is_q8 {
                    &pbs.fa_attn_out_batch
                } else if fa_wo_is_paro {
                    let paro_wo = layer.wo.paro.as_ref().unwrap_or_else(|| {
                        panic!("ParoQ4G128 wo missing paro metadata at FA layer {layer_idx}")
                    });
                    gpu.givens_rotate_to(
                        &pbs.fa_attn_out_batch,
                        &pbs.fa_attn_out_rot_batch,
                        &paro_wo.pairs,
                        &paro_wo.theta,
                        &paro_wo.channel_scales,
                        n,
                        layer.wo.k,
                        paro_wo.krot as usize,
                    )?;
                    &pbs.fa_attn_out_rot_batch
                } else {
                    // F2: AWQ-aware rotate for FullAttention wo (o_proj) input.
                    rotate_x_mq_batched_for(
                        gpu,
                        &layer.wo,
                        &pbs.fa_attn_out_batch,
                        &pbs.fa_attn_out_rot_batch,
                        layer.wo.k,
                        n,
                    )?;
                    &pbs.fa_attn_out_rot_batch
                };
                if fa_wo_is_6bit {
                    run_residual_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmHfq6G256Residual,
                        &layer.wo.buf,
                        layer.wo.gpu_dtype,
                        fa_wo_input,
                        &pbs.x_batch,
                        layer.wo.m,
                        layer.wo.k,
                        n,
                    )?;
                } else if fa_wo_is_q8 && q8_wmma_arch {
                    let x_n = pbs.x_batch.sub_offset(0, n * layer.wo.m);
                    run_residual_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmQ8_0ResidualWmma,
                        &layer.wo.buf,
                        layer.wo.gpu_dtype,
                        fa_wo_input,
                        &x_n,
                        layer.wo.m,
                        layer.wo.k,
                        n,
                    )?;
                } else if fa_wo_is_q8 {
                    // Non-WMMA Q8: GEMM into a scratch then add into x_batch.
                    // Reuse `fa_attn_out_rot_batch` (free since MQ4 rotate
                    // didn't run here) as scratch.
                    let scratch = pbs.fa_attn_out_rot_batch.sub_offset(0, n * layer.wo.m);
                    run_plain_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmQ8_0BatchedChunked,
                        &layer.wo.buf,
                        layer.wo.gpu_dtype,
                        fa_wo_input,
                        &scratch,
                        layer.wo.m,
                        layer.wo.k,
                        n,
                    )?;
                    let x_n = pbs.x_batch.sub_offset(0, n * layer.wo.m);
                    gpu.add_inplace_f32(&x_n, &scratch)?;
                } else if fa_wo_is_paro {
                    // PARO wo residual: HFQ4G128 batched GEMM into scratch,
                    // then add into x_batch. Reuse x_norm_batch (free since
                    // QKVZA is done — the MoE FFN body below rewrites it
                    // as its first action) as the gemm output scratch.
                    let scratch = pbs.x_norm_batch.sub_offset(0, n * layer.wo.m);
                    run_plain_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmHfq4G128,
                        &layer.wo.buf,
                        layer.wo.gpu_dtype,
                        fa_wo_input,
                        &scratch,
                        layer.wo.m,
                        layer.wo.k,
                        n,
                    )?;
                    let x_n = pbs.x_batch.sub_offset(0, n * layer.wo.m);
                    gpu.add_inplace_f32(&x_n, &scratch)?;
                } else if matches!(layer.wo.gpu_dtype, DType::OqCompactG256) {
                    // Compact-resident Opus: same W8A8 math as the oq8 arm,
                    // decoding OqPlusCompact blocks in-kernel.
                    let bs = super::prefill_batch::oq_compact_block_stride(&layer.wo)?;
                    gpu.gemm_oq_compact_residual_act_batched(
                        &layer.wo.buf,
                        fa_wo_input,
                        &pbs.x_batch,
                        layer.wo.m,
                        layer.wo.k,
                        n,
                        bs,
                    )?;
                } else {
                    run_residual_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmHfq4G256Residual,
                        &layer.wo.buf,
                        layer.wo.gpu_dtype,
                        fa_wo_input,
                        &pbs.x_batch,
                        layer.wo.m,
                        layer.wo.k,
                        n,
                    )?;
                }

                // Batched MoE FFN.
                prefill_moe_ffn_body_batched(
                    gpu,
                    weights.pager.as_ref(),
                    &layer.ffn,
                    &layer.ffn_norm,
                    config,
                    pbs,
                    n,
                    layer_idx,
                    &ctx,
                    routed_out,
                    None,
                )?;

                // Post-layer hidden extract for the DFlash draft path.
                if let Some(rb) = hidden_rb {
                    if let Some(slot) = rb.extract_slot(layer_idx) {
                        rb.write_rows_to_staging(gpu, slot, &pbs.x_batch, n)?;
                    }
                }

                let _ = kv_dim;
                let _ = q_dim;
                fa_layer_idx += 1;
            }

            _ => panic!("layer type mismatch at layer {layer_idx}"),
        }
        dump_hidden_localize(gpu, &pbs.x_batch, n, start_pos, dim, layer_idx, "batched");
        // Block-boundary steering/abliteration hook (no-op unless active).
        // Prefill convention: capture folds the last position, apply hits all.
        // `pbs.x_batch` holds the settled per-layer residual for all n rows of
        // this chunk (same tensor the DFlash extract sites write).
        hipfire_steer::maybe_steer_block_batched(gpu, &pbs.x_batch, layer_idx, n, dim)?;
    }

    // ── 3. Final output norm + logits ───────────────────────────────────
    // Multi-GPU band-mode: skip when this is not the last band — the
    // running activation in `pbs.x_batch` is what the next band's
    // peer-copy reads. `weights.output_norm` and `weights.output` only
    // live on the last band's device anyway.
    if do_lm_head {
        // If the caller requested per-token hidden output (DFlash verify path),
        // run rmsnorm over all N rows into their buffer. Otherwise use the
        // legacy last-token-only path.
        if let Some((dst, offset_rows)) = per_token_hidden_out {
            let dst_view = dst.sub_offset(offset_rows * dim, n * dim);
            gpu.rmsnorm_batched(
                &pbs.x_batch,
                &weights.output_norm,
                &dst_view,
                n,
                dim,
                config.norm_eps,
            )?;
            if prefill_should_emit_last_token_logits(true, needs_last_token_logits) {
                // Still populate s.logits with the last-token logits for
                // callers that rely on it (the legacy prefill post-condition).
                let last = n - 1;
                let last_view = dst.sub_offset((offset_rows + last) * dim, dim);
                {
                    let wr = weights.output.dispatch_ref();
                    let step = Step::Gemv {
                        w: &wr,
                        input: GemvInput::Raw(&last_view),
                        out: &s.logits,
                    };
                    execute_steps(gpu, &ctx, &[step])
                        .map_err(|e| hip_bridge::HipError::new(0, &e.to_string()))?;
                }
            }
        } else {
            // Legacy path: only last-token logits.
            // Use _auto so the D→D copy routes through the active stream
            // during hipGraph capture (bare memcpy_dtod_at uses the legacy
            // null stream and breaks capture: HIP error 906).
            let last = n - 1;
            gpu.memcpy_dtod_at_auto(
                &s.x.buf,
                0,
                &pbs.x_batch.buf,
                last * dim_row_bytes,
                dim_row_bytes,
            )?;
            gpu.rmsnorm_f32(&s.x, &weights.output_norm, &s.tmp, config.norm_eps)?;
            {
                let wr = weights.output.dispatch_ref();
                let step = Step::Gemv {
                    w: &wr,
                    input: GemvInput::Raw(&s.tmp),
                    out: &s.logits,
                };
                execute_steps(gpu, &ctx, &[step])
                    .map_err(|e| hip_bridge::HipError::new(0, &e.to_string()))?;
            }
        }
    }

    // Flushed at the END of the first chunk: the MoE and dtype decisions are
    // recorded during the body, so flushing on entry would print a half report.
    super::feature_report::flush_once();
    Ok(())
}
