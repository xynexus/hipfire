// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.

//! §M2a3 — the batched-prefill `ForwardBindings` impl.
//!
//! [`Qwen35PrefillBindings`] executes the SAME five super-ops as
//! `lower_variant(Q35Variant::FullAttn)` — the program the decode path already
//! runs — but over `n` rows instead of one. The bodies are the batched kernel
//! sequences that used to sit inline in `prefill_chunk.rs`'s
//! `(FullAttn, FullAttention) if fa_batched_ok` arm, MOVED here unchanged and
//! cut on the arm's own numbered phase boundaries, which fall exactly on the op
//! boundaries:
//!
//! | super-op | phases | was |
//! |---|---|---|
//! | `PROJ_QKV` | 1–2 rmsnorm(+rotate), 3-way QKV | `prefill_chunk.rs:4081-4550` |
//! | `ATTEND_FULL` | 4–8 q/k norm, RoPE, KV write, attention, gate | `:4551-5383` |
//! | `RESID_WO` | 9 wo residual | `:5384-5579` |
//! | `PROJ_GATE_UP` | 10a FFN norm(+rotate), gate+up | `:5580-5830` |
//! | `RESID_DOWN_SWIGLU` | 10b silu_mul(+rotate), w_down residual | `:5831-6010` |
//!
//! **The move is deliberately text-preserving.** Each method re-binds the
//! enclosing locals it used (`layer`, `pbs`, `config`, `n`, …) at the top and
//! leaves the body otherwise untouched, so this stage cannot change numerics by
//! construction — which matters because §4 established that the automatic tier
//! could not have told us if it had.
//!
//! **Shape travels in the impl, not in `DispatchCtx`.** `self.n` is the row
//! count; `DispatchCtx` has three arch-constant fields resolved once at
//! `Gpu::init()` and 42 `::new()` sites across 13 crates, so widening it for a
//! per-call value is the wrong seam (plan §M2a3, §6).
//!
//! Scope: dense `FullAttn` only. The DeltaNet and MoE arms are §M2a4 and still
//! live inline in `prefill_chunk.rs`.

use super::*;

use hipfire_dispatch::pipeline::superop::EscapeKind;

/// Per-layer execution context for the LOWERED BATCHED prefill. Mirrors
/// `Qwen35Bindings` (decode, n=1) field-for-field in spirit, but carries the row
/// count and the batched scratch instead of a single-token cursor.
pub(crate) struct Qwen35PrefillBindings<'a> {
    pub(crate) layer: &'a FullAttnLayerWeights,
    pub(crate) s: &'a Qwen35Scratch,
    pub(crate) pbs: &'a PrefillBatchScratch,
    pub(crate) config: &'a Qwen35Config,
    pub(crate) kv_cache: &'a mut kv::KvCache,
    /// Read-only here: this arm only ever takes `.as_ref()` on the tape.
    pub(crate) gdn_tape: Option<&'a crate::speculative::GdnTape>,
    pub(crate) tree_verify: Option<TreeVerifyCtx<'a>>,
    /// The BAND half of the KV rotation-table override. The moved body's
    /// `band_givens_cos.or(kv_cache.givens_cos.as_ref())` was a function-local macro over `band` and
    /// `kv_cache`; `band` does not exist here, so only that half becomes a
    /// field and the shim below rebuilds the identical expression. The
    /// `kv_cache` half stays a live read, because pre-resolving it would need a
    /// shared borrow that outlives the struct's `&mut kv_cache`.
    pub(crate) band_givens_cos: Option<&'a GpuTensor>,
    pub(crate) band_givens_sin: Option<&'a GpuTensor>,
    /// Rows in this chunk. The whole point of the impl — see the module note.
    pub(crate) n: usize,
    pub(crate) start_pos: usize,
    pub(crate) tape_offset: usize,
    pub(crate) delta_layer_idx: usize,
    pub(crate) layer_idx: usize,
    /// FA-layer ordinal within the band; indexes the per-FA-layer TriAttention
    /// slot table.
    pub(crate) fa_layer_idx: usize,
    /// `gpu.capture_mode`-dependent attention context bound, resolved by the
    /// caller because it is a property of the whole chunk, not of one layer.
    pub(crate) max_ctx_len: usize,
    /// Per-row logical positions when the caller overrides the linear default
    /// (tree verify, MTP). Backs the `position_at_row` helper the moved body
    /// calls, which was a closure over this same slice.
    pub(crate) positions_override: Option<&'a [usize]>,
}

impl<'a> Qwen35PrefillBindings<'a> {
    pub(crate) fn proj_qkv(&mut self, gpu: &mut Gpu) -> HipResult<()> {
        let layer = self.layer;
        let pbs = self.pbs;
        let config = self.config;
        let n = self.n;
        let dim = config.dim;
        let gdn_tape = self.gdn_tape;
        let tape_offset = self.tape_offset;
        let delta_layer_idx = self.delta_layer_idx;
        let q8_wmma_arch = gpu.arch_caps.has_wmma();
        let f16_prefill_wmma = qwen35_f16_prefill_wmma_enabled(gpu);

        // launch covers all N tokens at once.
        let _q_dim = config.n_heads * config.head_dim;
        let qkv_is_mq = matches!(
            layer.wq.gpu_dtype,
            DType::MQ4G256
                | DType::MQ6G256
                | DType::MQ3G256
                | DType::MQ3G256Lloyd
                | DType::MFP4G32
                | DType::Oq4G256
                // Opus W8A8 needs the SAME FWHT rotation as W4A4 — its
                // weights are rotated offline too. Omitting it here fed the
                // oq8 GEMM an unrotated activation (garbage: PPL 3.5e6).
                | DType::Oq8G256
                | DType::OqCompactG256
        );
        let qkv_is_6bit = matches!(layer.wq.gpu_dtype, DType::MQ6G256 | DType::HFQ6G256);
        let qkv_is_mq3 = matches!(layer.wq.gpu_dtype, DType::MQ3G256);
        let qkv_is_mq3_lloyd = matches!(layer.wq.gpu_dtype, DType::MQ3G256Lloyd);
        let qkv_is_fp4 = matches!(layer.wq.gpu_dtype, DType::HFP4G32 | DType::MFP4G32);
        let qkv_is_oq4 = matches!(layer.wq.gpu_dtype, DType::Oq4G256);
        let qkv_is_oq8 = matches!(layer.wq.gpu_dtype, DType::Oq8G256);
        let qkv_is_q8 = matches!(layer.wq.gpu_dtype, DType::Q8_0);
        let qkv_is_f32 = matches!(layer.wq.gpu_dtype, DType::F32);
        let qkv_is_f16 = matches!(layer.wq.gpu_dtype, DType::F16 | DType::BF16);
        // Fused QKV kernels require all three weights to share a
        // dtype — they treat wq/wk/wv as same-stride byte arrays.
        // When kmap mode 2 promotes only `v_proj` (issue #249), the
        // fused HFQ4 path reads `wv` as MQ6 with HFQ4's 136-B stride
        // and produces silent NaN. Gate the fused kernels here.
        //
        // The Q8 substrate path (gemm_q8_0_batched_chunked × 3) also
        // dispatches a Q8-stride kernel per weight, so it needs the
        // same gate when wk/wv aren't Q8.
        let qkv_same_dtype =
            layer.wk.gpu_dtype == layer.wq.gpu_dtype && layer.wv.gpu_dtype == layer.wq.gpu_dtype;
        let fa_bridge_tape_active = gdn_tape.as_ref().is_some_and(|tape| {
            delta_layer_idx < tape.fa_bridge_valid.len() && tape.fa_bridge_valid[delta_layer_idx]
        });
        if let Some(tape) = gdn_tape.as_ref() {
            if delta_layer_idx < tape.fa_bridge_valid.len() && tape.fa_bridge_valid[delta_layer_idx]
            {
                let hidden_row_bytes = tape.x_in_dim * 4;
                let off_hidden = tape_offset * hidden_row_bytes;
                gpu.memcpy_dtod_at_auto(
                    &tape.fa_bridge_input_bufs[delta_layer_idx].buf,
                    off_hidden,
                    &pbs.x_batch.buf,
                    0,
                    n * hidden_row_bytes,
                )?;
            }
        }

        // 1. rmsnorm (+ rotate for MQ) for the attn preamble.
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
        if let Some(tape) = gdn_tape.as_ref() {
            if delta_layer_idx < tape.fa_bridge_valid.len() && tape.fa_bridge_valid[delta_layer_idx]
            {
                let hidden_row_bytes = tape.x_in_dim * 4;
                let off_hidden = tape_offset * hidden_row_bytes;
                gpu.memcpy_dtod_at_auto(
                    &tape.fa_bridge_x_bufs[delta_layer_idx].buf,
                    off_hidden,
                    &pbs.x_rot_batch.buf,
                    0,
                    n * hidden_row_bytes,
                )?;
            }
        }

        // 2. Batched 3-way QKV projection (wq+wk+wv).
        if qkv_is_6bit && qkv_same_dtype {
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
        } else if qkv_is_mq3_lloyd && qkv_same_dtype {
            run_fused_qkv_key(
                gpu,
                hipfire_dispatch::types::KernelKey::FusedQkvMq3G256Lloyd,
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
        } else if qkv_is_mq3 && qkv_same_dtype {
            // X is already FWHT-rotated by fused_rmsnorm_rotate_mq_batched
            // above; call the bare HFQ3 GEMM (no second rotation). The
            // FusedQkvHfq3G256 run-arm replicates the call-site WMMA-vs-base
            // arch split internally (gemm_qkv_hfq3g256_wmma on has_wmma()
            // else the base cross-arch ladder), so the same kernel runs.
            run_fused_qkv_key(
                gpu,
                hipfire_dispatch::types::KernelKey::FusedQkvHfq3G256,
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
        } else if qkv_is_fp4 && qkv_same_dtype {
            // HFP4G32 / MFP4G32 FP4 batched WMMA. X is already
            // rotated above for MFP4 (is_mq path) — same kernel
            // covers both unrotated HFP4 and rotated MFP4 inputs.
            run_fused_qkv_key(
                gpu,
                hipfire_dispatch::types::KernelKey::FusedQkvHfp4G32,
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
        } else if qkv_is_oq8 && qkv_same_dtype {
            // Opus W8A8 FA QKV: one grouped int8-WMMA GEMM per projection
            // off the shared FWHT-rotated activation.
            debug_assert!(
                matches!(layer.wk.gpu_dtype, DType::Oq8G256)
                    && matches!(layer.wv.gpu_dtype, DType::Oq8G256),
                "FA qkv Oq8 dispatch requires all of wq/wk/wv to be Oq8G256",
            );
            gpu.quantize_act_oq8_batched_interleaved(&pbs.x_rot_batch, layer.wq.m, layer.wq.k, n)?;
            for (w, y) in [
                (&layer.wq, &pbs.fa_q_full_batch),
                (&layer.wk, &pbs.fa_k_batch),
                (&layer.wv, &pbs.fa_v_batch),
            ] {
                gpu.gemm_oq8_grouped_prequant(&w.buf, y, w.m, w.k, n)?;
            }
        } else if qkv_is_oq4 && qkv_same_dtype {
            // OQ4+ batched prefill FA QKV: int8-WMMA MMQ (n>=64) quantizing
            // the shared FWHT(+AWQ)-rotated activation to q8_1 ONCE across
            // q/k/v; gemm_oq4_qkv_mmq falls back to the f16 grouped path for
            // tiny batches internally via gemm_oq4_grouped_act_batched.
            debug_assert!(
                matches!(layer.wk.gpu_dtype, DType::Oq4G256)
                    && matches!(layer.wv.gpu_dtype, DType::Oq4G256),
                "FA qkv Oq4 dispatch requires all of wq/wk/wv to be Oq4G256",
            );
            // A4 int4-act gate: HIPFIRE_OQ4_PREFILL_ACT_BITS=4 forces TRUE
            // W4A4 (int4 activations) on qkv even at n>=64, where the default
            // routes to W4A8-MMQ. qkv@n>=64 is the ONLY non-W4A4 site in oq4
            // prefill (gate_up/o/down are already W4A4), so this makes a
            // fully-W4A4 scored prefill reachable for the A4 KLD gate. Default
            // (unset) keeps W4A8-MMQ, the shipped incumbent. See plan doc §9a.
            let act_bits = oq4_act_bits("QKV");
            let force_a4 = act_bits.as_deref() == Some("4");
            if act_bits.as_deref() == Some("16") {
                // A4 KLD act16 baseline: W4A16 per-projection qkv (no fused f16
                // qkv kernel), into the same q/k/v buffers as the W4A4 path.
                gpu.gemm_oq4_grouped_f16_wmma(
                    &layer.wq.buf,
                    &pbs.x_rot_batch,
                    &pbs.fa_q_full_batch,
                    layer.wq.m,
                    layer.wq.k,
                    n,
                    256,
                )?;
                gpu.gemm_oq4_grouped_f16_wmma(
                    &layer.wk.buf,
                    &pbs.x_rot_batch,
                    &pbs.fa_k_batch,
                    layer.wk.m,
                    layer.wk.k,
                    n,
                    256,
                )?;
                gpu.gemm_oq4_grouped_f16_wmma(
                    &layer.wv.buf,
                    &pbs.x_rot_batch,
                    &pbs.fa_v_batch,
                    layer.wv.m,
                    layer.wv.k,
                    n,
                    256,
                )?;
            } else if n >= 64 && !force_a4 {
                gpu.gemm_oq4_qkv_mmq(
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
                gpu.gemm_oq4_grouped_act_batched(
                    &layer.wq.buf,
                    &pbs.x_rot_batch,
                    &pbs.fa_q_full_batch,
                    layer.wq.m,
                    layer.wq.k,
                    n,
                )?;
                gpu.gemm_oq4_grouped_act_batched(
                    &layer.wk.buf,
                    &pbs.x_rot_batch,
                    &pbs.fa_k_batch,
                    layer.wk.m,
                    layer.wk.k,
                    n,
                )?;
                gpu.gemm_oq4_grouped_act_batched(
                    &layer.wv.buf,
                    &pbs.x_rot_batch,
                    &pbs.fa_v_batch,
                    layer.wv.m,
                    layer.wv.k,
                    n,
                )?;
            }
        } else if qkv_is_q8 && q8_wmma_arch && qkv_same_dtype {
            debug_assert!(
                matches!(layer.wk.gpu_dtype, DType::Q8_0)
                    && matches!(layer.wv.gpu_dtype, DType::Q8_0),
                "FA qkv Q8 WMMA dispatch requires all of wq/wk/wv to be Q8_0",
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
        } else if qkv_is_f32 && qkv_same_dtype {
            debug_assert!(
                matches!(layer.wk.gpu_dtype, DType::F32)
                    && matches!(layer.wv.gpu_dtype, DType::F32),
                "FA qkv F32 dispatch requires all of wq/wk/wv to be F32",
            );
        } else if qkv_is_f16 && qkv_same_dtype {
            if f16_prefill_wmma {
                gemm_raw_x_f32_auto(
                    gpu,
                    &layer.wq.buf,
                    &pbs.x_rot_batch,
                    &pbs.fa_q_full_batch,
                    layer.wq.m,
                    layer.wq.k,
                    n,
                )?;
                gemm_raw_x_f32_auto(
                    gpu,
                    &layer.wk.buf,
                    &pbs.x_rot_batch,
                    &pbs.fa_k_batch,
                    layer.wk.m,
                    layer.wk.k,
                    n,
                )?;
                gemm_raw_x_f32_auto(
                    gpu,
                    &layer.wv.buf,
                    &pbs.x_rot_batch,
                    &pbs.fa_v_batch,
                    layer.wv.m,
                    layer.wv.k,
                    n,
                )?;
            } else {
                gpu.fused_qkvza_f16_xf32_batched(
                    &layer.wq.buf,
                    &layer.wk.buf,
                    &layer.wv.buf,
                    &layer.wv.buf,
                    &pbs.x_rot_batch,
                    &pbs.fa_q_full_batch,
                    &pbs.fa_k_batch,
                    &pbs.fa_v_batch,
                    &pbs.dn_alpha_batch,
                    layer.wq.m,
                    layer.wk.m,
                    layer.wv.m,
                    0,
                    layer.wq.k,
                    n,
                )?;
            }
        } else if qkv_same_dtype {
            if fa_bridge_tape_active {
                gpu.gemm_qkv_hfq4g256_exact(
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
                gpu.gemm_qkv_hfq4g256(
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
            }
        } else {
            // Mixed-format fallback (issue #249): wq/wk/wv don't all
            // share a dtype. Dispatch each weight to its own
            // single-weight batched GEMM, dropping the fused-kernel
            // launch-overhead optimization for correctness.
            batched_gemm_single_weight(gpu, &layer.wq, &pbs.x_rot_batch, &pbs.fa_q_full_batch, n)?;
            batched_gemm_single_weight(gpu, &layer.wk, &pbs.x_rot_batch, &pbs.fa_k_batch, n)?;
            batched_gemm_single_weight(gpu, &layer.wv, &pbs.x_rot_batch, &pbs.fa_v_batch, n)?;
        }
        if let Some(tape) = gdn_tape.as_ref() {
            if delta_layer_idx < tape.fa_bridge_valid.len() && tape.fa_bridge_valid[delta_layer_idx]
            {
                let q_full_row_bytes = tape.fa_q_full_dim * 4;
                gpu.memcpy_dtod_at_auto(
                    &tape.fa_bridge_q_full_bufs[delta_layer_idx].buf,
                    tape_offset * q_full_row_bytes,
                    &pbs.fa_q_full_batch.buf,
                    0,
                    n * q_full_row_bytes,
                )?;
            }
        }

        qwen35_materialize_fa_q(
            gpu,
            config,
            &pbs.fa_q_full_batch,
            &pbs.fa_q_batch,
            &pbs.fa_gate_batch,
            n,
        )?;

        Ok(())
    }

    pub(crate) fn attend_full(&mut self, gpu: &mut Gpu, ctx: &DispatchCtx) -> HipResult<()> {
        let positions_override = self.positions_override;
        let start_pos_for_rows = self.start_pos;
        let position_at_row = |row: usize| -> usize {
            positions_override
                .map(|p| p[row])
                .unwrap_or(start_pos_for_rows + row)
        };
        // `forward_prefill_chunk` wrote these as a function-local
        // `band_givens_cos.or(kv_cache.givens_cos.as_ref())` macro over `band` + `kv_cache`. A macro cannot
        // come along: its body would resolve `kv_cache` in the wrong syntax
        // context here. The call sites below carry the macro's own expansion
        // instead, with the `band` half supplied by the caller.
        let band_givens_cos = self.band_givens_cos;
        let band_givens_sin = self.band_givens_sin;
        let layer = self.layer;
        let pbs = self.pbs;
        let config = self.config;
        let n = self.n;
        let gdn_tape = self.gdn_tape;
        let tape_offset = self.tape_offset;
        let delta_layer_idx = self.delta_layer_idx;
        let s = self.s;
        let start_pos = self.start_pos;
        let layer_idx = self.layer_idx;
        let tree_verify = self.tree_verify;
        let fa_layer_idx = self.fa_layer_idx;
        let max_ctx_len = self.max_ctx_len;
        let kv_cache = &mut *self.kv_cache;

        // 4. Per-head Q/K rmsnorm. rmsnorm_batched uses batch =
        // number of "rows" of head_dim. For [N × n_heads × head_dim]
        // that's batch = N * n_heads.
        gpu.rmsnorm_batched(
            &pbs.fa_q_batch,
            &layer.q_norm,
            &pbs.fa_q_batch,
            n * config.n_heads,
            config.head_dim,
            config.norm_eps,
        )?;
        if let Some(tape) = gdn_tape.as_ref() {
            if delta_layer_idx < tape.fa_bridge_valid.len() && tape.fa_bridge_valid[delta_layer_idx]
            {
                let q_row_bytes = tape.fa_q_dim * 4;
                gpu.memcpy_dtod_at_auto(
                    &tape.fa_bridge_q_norm_bufs[delta_layer_idx].buf,
                    tape_offset * q_row_bytes,
                    &pbs.fa_q_batch.buf,
                    0,
                    n * q_row_bytes,
                )?;
            }
        }
        gpu.rmsnorm_batched(
            &pbs.fa_k_batch,
            &layer.k_norm,
            &pbs.fa_k_batch,
            n * config.n_kv_heads,
            config.head_dim,
            config.norm_eps,
        )?;

        if hipfire_runtime::triattn::tap_enabled() {
            // Try GPU path first: dispatches a reduce kernel on the
            // device-resident Q tensor, zero PCIe transfer. Only
            // succeeds when install_tap_gpu() was used. Falls through
            // to CPU path otherwise.
            let gpu_handled = hipfire_runtime::triattn::record_prerope_q_batch_gpu_if_applicable(
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

        // Path B pre-RoPE K capture (slow-path-kill, WIP).
        // The next line mutates pbs.fa_k_batch in place — capture
        // BEFORE so the slow path has the unrotated K available
        // and can apply RoPE for the COMMITTED slot phases instead
        // of these linearization-slot phases. Capture is None
        // unless the env gate + the per-FA-layer scratch are both
        // wired through TreeVerifyCtx.
        if let Some(slots) = tree_verify.as_ref().and_then(|c| c.pre_rope_k_capture) {
            if let Some(slot) = slots.get(fa_layer_idx) {
                let kv_dim = config.n_kv_heads * config.head_dim;
                let n_bytes = n * kv_dim * 4;
                // Use _auto so the memcpy is recorded onto the
                // active stream when one exists (matches the
                // existing GdnTape capture pattern at line ~3193).
                // Plain gpu.hip.memcpy_dtod_at runs on the null
                // stream and sync-blocks pending async kernels,
                // changing kernel-launch order in ways that
                // perturb DDTree's ksplit-atomic nondeterminism
                // — output diverges even though no data is
                // actually changed.
                gpu.memcpy_dtod_at_auto(&slot.buf, 0, &pbs.fa_k_batch.buf, 0, n_bytes)?;
            }
        }

        // 5. Batched partial-interleaved RoPE (per-row positions).
        // pbs.positions stays physical for the KV write below; the
        // offset rotates new Q/K at absolute phase after compaction.
        let n_rot = (config.head_dim as f32 * config.partial_rotary_factor) as usize;
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
        // KV-compression study capture: post-RoPE FA Q/K/V for a target FA
        // layer (HIPFIRE_DUMP_HIDDEN_ALL=1 + HIPFIRE_DUMP_HIDDEN_LAYER).
        dump_hidden_localize(
            gpu,
            &pbs.fa_q_batch,
            n,
            start_pos,
            config.n_heads * config.head_dim,
            layer_idx,
            "faq",
        );
        dump_hidden_localize(
            gpu,
            &pbs.fa_k_batch,
            n,
            start_pos,
            config.n_kv_heads * config.head_dim,
            layer_idx,
            "fak",
        );
        dump_hidden_localize(
            gpu,
            &pbs.fa_v_batch,
            n,
            start_pos,
            config.n_kv_heads * config.head_dim,
            layer_idx,
            "fav",
        );
        if let Some(tape) = gdn_tape.as_ref() {
            if delta_layer_idx < tape.fa_bridge_valid.len() && tape.fa_bridge_valid[delta_layer_idx]
            {
                let q_row_bytes = tape.fa_q_dim * 4;
                let kv_row_bytes = tape.fa_kv_dim * 4;
                gpu.memcpy_dtod_at_auto(
                    &tape.fa_bridge_q_bufs[delta_layer_idx].buf,
                    tape_offset * q_row_bytes,
                    &pbs.fa_q_batch.buf,
                    0,
                    n * q_row_bytes,
                )?;
                gpu.memcpy_dtod_at_auto(
                    &tape.fa_bridge_k_bufs[delta_layer_idx].buf,
                    tape_offset * kv_row_bytes,
                    &pbs.fa_k_batch.buf,
                    0,
                    n * kv_row_bytes,
                )?;
                gpu.memcpy_dtod_at_auto(
                    &tape.fa_bridge_v_bufs[delta_layer_idx].buf,
                    tape_offset * kv_row_bytes,
                    &pbs.fa_v_batch.buf,
                    0,
                    n * kv_row_bytes,
                )?;
            }
        }

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

        // 6. Batched KV cache writes (per-row positions).
        if kv_cache.quant_kvarn {
            // KVarN K = 4-bit block records + fp16 window (not a contiguous
            // buffer), so the generic batched writes below would fault.
            // kvarn_attend owns the batched write (window append + 128-block
            // flush) AND the fused causal flash together — the same entry
            // point decode uses (mod.rs), which already supports n>1.
            // Rotation + tiles scratch mirror the decode path. The paired
            // attention step below is a no-op for kvarn (done here).
            static PREFILL_KVARN_ROTATE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            let kvarn_rotate = *PREFILL_KVARN_ROTATE
                .get_or_init(|| std::env::var("HIPFIRE_KVARN_ROTATE").ok().as_deref() != Some("0"));
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
                let tiles =
                    gpu.alloc_tensor(&[config.n_kv_heads * config.head_dim * 128], DType::F32)?;
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
            let ct = band_givens_cos.or(kv_cache.givens_cos.as_ref()).unwrap();
            let st = band_givens_sin.or(kv_cache.givens_sin.as_ref()).unwrap();
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
            let ct = band_givens_cos.or(kv_cache.givens_cos.as_ref()).unwrap();
            let st = band_givens_sin.or(kv_cache.givens_sin.as_ref()).unwrap();
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
            let ct = band_givens_cos.or(kv_cache.givens_cos.as_ref()).unwrap();
            let st = band_givens_sin.or(kv_cache.givens_sin.as_ref()).unwrap();
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
        } else if kv_cache.quant_q8 && q8_fa_attention_serial_kv_loop_enabled() {
            assert!(
                tree_verify.is_none(),
                "HIPFIRE_Q8_FA_ATTENTION_SERIAL_KV_LOOP is a causal Q8 FA diagnostic; tree-verify masking is not supported",
            );
            // Diagnostic: defer KV writes to the row-serial attention
            // loop below so write/read ordering matches serial decode.
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

        // 7. Batched causal attention (or tree-attention if tree_verify is set).
        // asym{4,3,2}: batched flash (K rotated-quantized + V Q8 in normal space).
        // Q8: batched kernel unless ctx > 15K (LDS overflow), then per-position flash.
        //
        // Tree-verify mode: `block_start = start_pos`, `block_cols = n`.
        // The bias buffer is `[n × n]`; each query row applies its
        // corresponding bias row to in-block keys. Long-context Q8
        // tiled fallback isn't supported in tree mode (we caught
        // that as an assert above — tree blocks are small).
        const LDS_CTX_LIMIT: usize = 15000;
        let tree_bias = tree_verify.as_ref().map(|c| c.attn_bias);
        // 6–7. Batched KV write + flash attention (via dispatch).
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
            let ct = band_givens_cos.or(kv_cache.givens_cos.as_ref()).unwrap();
            let st = band_givens_sin.or(kv_cache.givens_sin.as_ref()).unwrap();
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
            let ct = band_givens_cos.or(kv_cache.givens_cos.as_ref()).unwrap();
            let st = band_givens_sin.or(kv_cache.givens_sin.as_ref()).unwrap();
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
            let ct = band_givens_cos.or(kv_cache.givens_cos.as_ref()).unwrap();
            let st = band_givens_sin.or(kv_cache.givens_sin.as_ref()).unwrap();
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
        } else if kv_cache.quant_q8 && q8_fa_attention_serial_kv_loop_enabled() {
            assert!(
                tree_verify.is_none(),
                "HIPFIRE_Q8_FA_ATTENTION_SERIAL_KV_LOOP is a causal Q8 FA diagnostic; tree-verify masking is not supported",
            );
            let q_dim = config.n_heads * config.head_dim;
            let kv_dim = config.n_kv_heads * config.head_dim;
            let pos_buf_tmp = gpu.hip.malloc(4)?;
            let pos_buf_result = (|| -> HipResult<()> {
                for b in 0..n {
                    let pos_b = position_at_row(b);
                    let pos_i32 = pos_b as i32;
                    gpu.hip.memcpy_htod(&pos_buf_tmp, &pos_i32.to_ne_bytes())?;
                    let q_b = pbs.fa_q_batch.sub_offset(b * q_dim, q_dim);
                    let k_b = pbs.fa_k_batch.sub_offset(b * kv_dim, kv_dim);
                    let v_b = pbs.fa_v_batch.sub_offset(b * kv_dim, kv_dim);
                    let out_b = pbs.fa_attn_out_batch.sub_offset(b * q_dim, q_dim);
                    gpu.kv_cache_write_q8_0(
                        &kv_cache.k_gpu[layer_idx],
                        &k_b,
                        &pos_buf_tmp,
                        config.n_kv_heads,
                        config.head_dim,
                    )?;
                    gpu.kv_cache_write_q8_0(
                        &kv_cache.v_gpu[layer_idx],
                        &v_b,
                        &pos_buf_tmp,
                        config.n_kv_heads,
                        config.head_dim,
                    )?;
                    gpu.attention_q8_0_kv(
                        &q_b,
                        &kv_cache.k_gpu[layer_idx],
                        &kv_cache.v_gpu[layer_idx],
                        &out_b,
                        &pos_buf_tmp,
                        pos_b + 1,
                        config.n_heads,
                        config.n_kv_heads,
                        config.head_dim,
                        kv_cache.physical_cap,
                    )?;
                }
                Ok(())
            })();
            let _ = gpu.hip.free(pos_buf_tmp);
            pos_buf_result?;
        } else if kv_cache.quant_q8 && max_ctx_len > LDS_CTX_LIMIT {
            assert!(
                tree_verify.is_none(),
                "tree-verify mode hits the long-context Q8 fallback \
                 at max_ctx_len={} > {}; tree blocks should stay small",
                max_ctx_len,
                LDS_CTX_LIMIT,
            );
            // Per-position flash Q8 attention for long-context prefill.
            //
            // `pbs.positions` is raw i32 bits in an F32 slot
            // (slot-cosmetic, see PrefillBatchScratch::new).
            // `download_f32` would reinterpret those bytes as floats —
            // i32 15000 = 0x3A98 round-trips through f32 as ~1e-3
            // subnormal, which casts to 0. Reconstruct from the
            // host-side row position directly.
            let q_dim = config.n_heads * config.head_dim;
            let pos_buf_tmp = gpu.hip.malloc(4)?;
            let pos_buf_result = (|| -> HipResult<()> {
                for b in 0..n {
                    let pos_b = position_at_row(b);
                    let seq_len_b = pos_b + 1;
                    let pos_i32 = pos_b as i32;
                    gpu.hip.memcpy_htod(&pos_buf_tmp, &pos_i32.to_ne_bytes())?;
                    let q_b = pbs.fa_q_batch.sub_offset(b * q_dim, q_dim);
                    let out_b = pbs.fa_attn_out_batch.sub_offset(b * q_dim, q_dim);
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
        } else if kv_cache.quant_q8 && q8_fa_attention_scalar_loop_enabled() {
            let q_dim = config.n_heads * config.head_dim;
            let pos_buf_tmp = gpu.hip.malloc(4)?;
            let pos_buf_result = (|| -> HipResult<()> {
                for b in 0..n {
                    let pos_b = position_at_row(b);
                    let pos_i32 = pos_b as i32;
                    gpu.hip.memcpy_htod(&pos_buf_tmp, &pos_i32.to_ne_bytes())?;
                    let q_b = pbs.fa_q_batch.sub_offset(b * q_dim, q_dim);
                    let out_b = pbs.fa_attn_out_batch.sub_offset(b * q_dim, q_dim);
                    gpu.attention_q8_0_kv(
                        &q_b,
                        &kv_cache.k_gpu[layer_idx],
                        &kv_cache.v_gpu[layer_idx],
                        &out_b,
                        &pos_buf_tmp,
                        pos_b + 1,
                        config.n_heads,
                        config.n_kv_heads,
                        config.head_dim,
                        kv_cache.physical_cap,
                    )?;
                }
                Ok(())
            })();
            let _ = gpu.hip.free(pos_buf_tmp);
            pos_buf_result?;
        } else if kv_cache.quant_q8 && q8_fa_attention_row_loop_enabled() {
            let q8_tree_bias = if q8_fa_attention_ignore_tree_bias_enabled() {
                None
            } else {
                tree_bias
            };
            let q_dim = config.n_heads * config.head_dim;
            for b in 0..n {
                let q_b = pbs.fa_q_batch.sub_offset(b * q_dim, q_dim);
                let out_b = pbs.fa_attn_out_batch.sub_offset(b * q_dim, q_dim);
                let pos_b = pbs.positions.sub_offset(b, 1);
                let bias_b = q8_tree_bias.map(|bias| bias.sub_offset(b * block_cols, block_cols));
                gpu.attention_q8_0_kv_batched_masked(
                    &q_b,
                    &kv_cache.k_gpu[layer_idx],
                    &kv_cache.v_gpu[layer_idx],
                    &out_b,
                    &pos_b,
                    config.n_heads,
                    config.n_kv_heads,
                    config.head_dim,
                    kv_cache.physical_cap,
                    max_ctx_len,
                    1,
                    bias_b.as_ref(),
                    block_start,
                    block_cols,
                )?;
            }
        } else if kv_cache.quant_q8 {
            let q8_tree_bias = if q8_fa_attention_ignore_tree_bias_enabled() {
                None
            } else {
                tree_bias
            };
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
                q8_tree_bias,
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
        if let Some(tape) = gdn_tape.as_ref() {
            if delta_layer_idx < tape.fa_bridge_valid.len() && tape.fa_bridge_valid[delta_layer_idx]
            {
                let q_row_bytes = tape.fa_q_dim * 4;
                gpu.memcpy_dtod_at_auto(
                    &tape.fa_bridge_attn_raw_bufs[delta_layer_idx].buf,
                    tape_offset * q_row_bytes,
                    &pbs.fa_attn_out_batch.buf,
                    0,
                    n * q_row_bytes,
                )?;
            }
        }
        // KVarN already did the fused write + causal flash in step 6
        // (`kvarn_attend`), and KvTierPlan has no KVarN tier -- it would fall
        // through to the F32 fallback, which has no batched write, and resolve
        // as `no implementation for KvWriteF32`. Skip. Pairs with admitting
        // quant_kvarn into `fa_kv_ok`; admitting it without this guard is
        // exactly how that error reappears.
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

        qwen35_apply_fa_gate(gpu, config, &pbs.fa_attn_out_batch, &pbs.fa_gate_batch)?;
        if let Some(tape) = gdn_tape.as_ref() {
            if delta_layer_idx < tape.fa_bridge_valid.len() && tape.fa_bridge_valid[delta_layer_idx]
            {
                let hidden_row_bytes = tape.x_in_dim * 4;
                let off_hidden = tape_offset * hidden_row_bytes;
                gpu.memcpy_dtod_at_auto(
                    &tape.fa_bridge_attn_out_bufs[delta_layer_idx].buf,
                    off_hidden,
                    &pbs.fa_attn_out_batch.buf,
                    0,
                    n * hidden_row_bytes,
                )?;
            }
        }

        Ok(())
    }

    pub(crate) fn resid_wo(&mut self, gpu: &mut Gpu) -> HipResult<()> {
        let layer = self.layer;
        let pbs = self.pbs;
        let n = self.n;
        let gdn_tape = self.gdn_tape;
        let tape_offset = self.tape_offset;
        let delta_layer_idx = self.delta_layer_idx;
        let q8_wmma_arch = gpu.arch_caps.has_wmma();
        let f16_prefill_wmma = qwen35_f16_prefill_wmma_enabled(gpu);

        // 9. wo residual: x_batch += wo · (optional rotate)(fa_attn_out_batch).
        // Same MQ rotation requirement as the LA wo path.
        let fa_wo_is_mq = matches!(
            layer.wo.gpu_dtype,
            DType::MQ4G256
                | DType::MQ6G256
                | DType::MQ3G256
                | DType::MQ3G256Lloyd
                | DType::MFP4G32
                | DType::Oq4G256
                // Opus W8A8 needs the SAME FWHT rotation as W4A4 — its
                // weights are rotated offline too. Omitting it here fed the
                // oq8 GEMM an unrotated activation (garbage: PPL 3.5e6).
                | DType::Oq8G256
                | DType::OqCompactG256
        );
        let fa_wo_is_6bit = matches!(layer.wo.gpu_dtype, DType::MQ6G256 | DType::HFQ6G256);
        let fa_wo_is_mq3 = matches!(layer.wo.gpu_dtype, DType::MQ3G256);
        let fa_wo_is_mq3_lloyd = matches!(layer.wo.gpu_dtype, DType::MQ3G256Lloyd);
        let fa_wo_is_fp4 = matches!(layer.wo.gpu_dtype, DType::HFP4G32 | DType::MFP4G32);
        let fa_wo_is_oq4 = matches!(layer.wo.gpu_dtype, DType::Oq4G256);
        let fa_wo_is_oq8 = matches!(layer.wo.gpu_dtype, DType::Oq8G256);
        let fa_wo_is_q8 = matches!(layer.wo.gpu_dtype, DType::Q8_0);
        let fa_wo_is_f32 = matches!(layer.wo.gpu_dtype, DType::F32);
        let fa_wo_is_f16 = matches!(layer.wo.gpu_dtype, DType::F16 | DType::BF16);
        let fa_wo_input = if fa_wo_is_mq {
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
        } else {
            &pbs.fa_attn_out_batch
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
        } else if fa_wo_is_oq8 {
            // Opus W8A8: fa_wo_input is FWHT-rotated above.
            gpu.gemm_oq8_grouped_residual_act_batched(
                &layer.wo.buf,
                fa_wo_input,
                &pbs.x_batch,
                layer.wo.m,
                layer.wo.k,
                n,
            )?;
        } else if fa_wo_is_oq4 {
            // Opus W4A4: fa_wo_input is FWHT(+AWQ)-rotated above.
            gpu.gemm_oq4_grouped_residual_act_batched(
                &layer.wo.buf,
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
            let scratch = pbs.x_rot_batch.sub_offset(0, n * layer.wo.m);
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
        } else if fa_wo_is_f32 {
            gemm_f32_residual_batched(
                gpu,
                &layer.wo.buf,
                fa_wo_input,
                &pbs.x_batch,
                &pbs.x_rot_batch,
                layer.wo.m,
                layer.wo.k,
                n,
            )?;
        } else if fa_wo_is_f16 {
            if f16_prefill_wmma {
                gemm_raw_x_f32_residual_batched_auto(
                    gpu,
                    &layer.wo.buf,
                    fa_wo_input,
                    &pbs.x_batch,
                    &pbs.x_rot_batch,
                    layer.wo.m,
                    layer.wo.k,
                    n,
                )?;
            } else {
                gpu.gemv_f16_xf32_residual_batched(
                    &layer.wo.buf,
                    fa_wo_input,
                    &pbs.x_batch,
                    layer.wo.m,
                    layer.wo.k,
                    n,
                )?;
            }
        } else if fa_wo_is_mq3_lloyd {
            run_residual_gemm_key(
                gpu,
                hipfire_dispatch::types::KernelKey::GemmMq3G256LloydResidual,
                &layer.wo.buf,
                layer.wo.gpu_dtype,
                fa_wo_input,
                &pbs.x_batch,
                layer.wo.m,
                layer.wo.k,
                n,
            )?;
        } else if fa_wo_is_mq3 {
            run_residual_gemm_key(
                gpu,
                hipfire_dispatch::types::KernelKey::GemmHfq3G256Residual,
                &layer.wo.buf,
                layer.wo.gpu_dtype,
                fa_wo_input,
                &pbs.x_batch,
                layer.wo.m,
                layer.wo.k,
                n,
            )?;
        } else if fa_wo_is_fp4 {
            run_residual_gemm_key(
                gpu,
                hipfire_dispatch::types::KernelKey::GemmHfp4G32Residual,
                &layer.wo.buf,
                layer.wo.gpu_dtype,
                fa_wo_input,
                &pbs.x_batch,
                layer.wo.m,
                layer.wo.k,
                n,
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
        if let Some(tape) = gdn_tape.as_ref() {
            if delta_layer_idx < tape.fa_bridge_valid.len() && tape.fa_bridge_valid[delta_layer_idx]
            {
                let hidden_row_bytes = tape.x_in_dim * 4;
                let off_hidden = tape_offset * hidden_row_bytes;
                gpu.memcpy_dtod_at_auto(
                    &tape.fa_bridge_wo_residual_bufs[delta_layer_idx].buf,
                    off_hidden,
                    &pbs.x_batch.buf,
                    0,
                    n * hidden_row_bytes,
                )?;
            }
        }

        Ok(())
    }

    pub(crate) fn proj_gate_up(&mut self, gpu: &mut Gpu) -> HipResult<()> {
        let layer = self.layer;
        let pbs = self.pbs;
        let config = self.config;
        let n = self.n;
        let dim = config.dim;
        let q8_wmma_arch = gpu.arch_caps.has_wmma();
        let f16_prefill_wmma = qwen35_f16_prefill_wmma_enabled(gpu);

        // 10. FFN: rmsnorm (+ rotate for MQ), gate+up, silu_mul
        // (+ rotate for MQ), w_down residual.
        let fa_ffn_is_mq = matches!(
            layer.w_gate.gpu_dtype,
            DType::MQ4G256
                | DType::MQ6G256
                | DType::MQ3G256
                | DType::MQ3G256Lloyd
                | DType::MFP4G32
                | DType::Oq4G256
                // Opus W8A8 needs the SAME FWHT rotation as W4A4 — its
                // weights are rotated offline too. Omitting it here fed the
                // oq8 GEMM an unrotated activation (garbage: PPL 3.5e6).
                | DType::Oq8G256
                | DType::OqCompactG256
        );
        let fa_ffn_is_6bit = matches!(layer.w_gate.gpu_dtype, DType::MQ6G256 | DType::HFQ6G256);
        let fa_ffn_is_mq3 = matches!(layer.w_gate.gpu_dtype, DType::MQ3G256);
        let fa_ffn_is_mq3_lloyd = matches!(layer.w_gate.gpu_dtype, DType::MQ3G256Lloyd);
        let fa_ffn_is_fp4 = matches!(layer.w_gate.gpu_dtype, DType::HFP4G32 | DType::MFP4G32);
        let fa_ffn_is_oq4 = matches!(layer.w_gate.gpu_dtype, DType::Oq4G256);
        let fa_ffn_is_oq8 = matches!(layer.w_gate.gpu_dtype, DType::Oq8G256);
        let fa_ffn_is_q8 = matches!(layer.w_gate.gpu_dtype, DType::Q8_0);
        let fa_ffn_is_f32 = matches!(layer.w_gate.gpu_dtype, DType::F32);
        let fa_ffn_is_f16 = matches!(layer.w_gate.gpu_dtype, DType::F16 | DType::BF16);
        if fa_ffn_is_mq {
            // AWQ-aware: next linear is w_gate (FA-FFN, gate/up share input).
            fused_rmsnorm_rotate_mq_batched_for(
                gpu,
                &pbs.x_batch,
                &layer.ffn_norm,
                &layer.w_gate,
                &pbs.x_rot_batch,
                dim,
                config.norm_eps,
                n,
            )?;
        } else {
            gpu.rmsnorm_batched(
                &pbs.x_batch,
                &layer.ffn_norm,
                &pbs.x_rot_batch,
                n,
                dim,
                config.norm_eps,
            )?;
        }
        // #397 Ship 5.2 slice 2: FA-FFN fused gate+up → FusedQkvFamily
        // (batched-prefill gate+up variant), mirroring the LA-FFN block
        // above. Q8-non-WMMA stays as two plain GEMMs; HFQ3 WMMA-vs-base
        // is folded into the FusedGateUpHfq3G256 run-arm.
        if fa_ffn_is_6bit {
            run_fused_gate_up_key(
                gpu,
                hipfire_dispatch::types::KernelKey::FusedGateUpHfq6G256,
                &layer.w_gate.buf,
                &layer.w_up.buf,
                &pbs.x_rot_batch,
                &pbs.gate_ffn_batch,
                &pbs.up_batch,
                layer.w_gate.m,
                layer.w_up.m,
                layer.w_gate.k,
                n,
            )?;
                } else if matches!(layer.w_gate.gpu_dtype, DType::OqCompactG256) {
                    // Compact-resident Opus: one quantize of the shared rotated
                    // activation, then one compact GEMM per projection.
                    gpu.quantize_act_oq8_batched_interleaved(
                        &pbs.x_rot_batch,
                        layer.w_gate.m,
                        layer.w_gate.k,
                        n,
                    )?;
                    for (w, y) in [
                        (&layer.w_gate, &pbs.gate_ffn_batch),
                        (&layer.w_up, &pbs.up_batch),
                    ] {
                        let bs = super::prefill_batch::oq_compact_block_stride(w)?;
                        gpu.gemm_oq_compact_grouped_prequant(&w.buf, y, w.m, w.k, n, bs)?;
                    }
        } else if fa_ffn_is_oq8 {
            // Opus W8A8 gate+up: one grouped int8-WMMA GEMM per projection.
            gpu.quantize_act_oq8_batched_interleaved(&pbs.x_rot_batch, layer.w_gate.m, layer.w_gate.k, n)?;
            for (w, y) in [
                (&layer.w_gate, &pbs.gate_ffn_batch),
                (&layer.w_up, &pbs.up_batch),
            ] {
                gpu.gemm_oq8_grouped_prequant(&w.buf, y, w.m, w.k, n)?;
            }
        } else if fa_ffn_is_oq4 {
            // Opus W4A4: x_rot_batch is FWHT(+AWQ)-rotated above (fa_ffn_is_mq).
            run_fused_gate_up_key(
                gpu,
                hipfire_dispatch::types::KernelKey::FusedGateUpOq4G256,
                &layer.w_gate.buf,
                &layer.w_up.buf,
                &pbs.x_rot_batch,
                &pbs.gate_ffn_batch,
                &pbs.up_batch,
                layer.w_gate.m,
                layer.w_up.m,
                layer.w_gate.k,
                n,
            )?;
        } else if fa_ffn_is_q8 && q8_wmma_arch {
            debug_assert!(
                matches!(layer.w_up.gpu_dtype, DType::Q8_0),
                "FA FFN Q8 WMMA dispatch requires both w_gate and w_up to be Q8_0",
            );
            run_fused_gate_up_key(
                gpu,
                hipfire_dispatch::types::KernelKey::FusedGateUpQ8_0,
                &layer.w_gate.buf,
                &layer.w_up.buf,
                &pbs.x_rot_batch,
                &pbs.gate_ffn_batch,
                &pbs.up_batch,
                layer.w_gate.m,
                layer.w_up.m,
                layer.w_gate.k,
                n,
            )?;
        } else if fa_ffn_is_q8 {
            run_plain_gemm_key(
                gpu,
                hipfire_dispatch::types::KernelKey::GemmQ8_0BatchedChunked,
                &layer.w_gate.buf,
                layer.w_gate.gpu_dtype,
                &pbs.x_rot_batch,
                &pbs.gate_ffn_batch,
                layer.w_gate.m,
                layer.w_gate.k,
                n,
            )?;
            run_plain_gemm_key(
                gpu,
                hipfire_dispatch::types::KernelKey::GemmQ8_0BatchedChunked,
                &layer.w_up.buf,
                layer.w_up.gpu_dtype,
                &pbs.x_rot_batch,
                &pbs.up_batch,
                layer.w_up.m,
                layer.w_up.k,
                n,
            )?;
        } else if fa_ffn_is_f32 {
            debug_assert!(
                matches!(layer.w_up.gpu_dtype, DType::F32),
                "FA FFN F32 dispatch requires both w_gate and w_up to be F32",
            );
            gpu.gemm_f32_register_tiled(
                &layer.w_gate.buf,
                &pbs.x_rot_batch,
                &pbs.gate_ffn_batch,
                layer.w_gate.m,
                layer.w_gate.k,
                n,
            )?;
            gpu.gemm_f32_register_tiled(
                &layer.w_up.buf,
                &pbs.x_rot_batch,
                &pbs.up_batch,
                layer.w_up.m,
                layer.w_up.k,
                n,
            )?;
        } else if fa_ffn_is_f16 {
            debug_assert!(
                matches!(layer.w_up.gpu_dtype, DType::F16 | DType::BF16),
                "FA FFN F16/BF16 dispatch requires both w_gate and w_up to be F16",
            );
            if f16_prefill_wmma {
                gemm_raw_x_f32_auto(
                    gpu,
                    &layer.w_gate.buf,
                    &pbs.x_rot_batch,
                    &pbs.gate_ffn_batch,
                    layer.w_gate.m,
                    layer.w_gate.k,
                    n,
                )?;
                gemm_raw_x_f32_auto(
                    gpu,
                    &layer.w_up.buf,
                    &pbs.x_rot_batch,
                    &pbs.up_batch,
                    layer.w_up.m,
                    layer.w_up.k,
                    n,
                )?;
            } else {
                gpu.fused_gate_up_f16_xf32_batched(
                    &layer.w_gate.buf,
                    &layer.w_up.buf,
                    &pbs.x_rot_batch,
                    &pbs.gate_ffn_batch,
                    &pbs.up_batch,
                    layer.w_gate.m,
                    layer.w_up.m,
                    layer.w_gate.k,
                    n,
                )?;
            }
        } else if fa_ffn_is_mq3_lloyd {
            run_fused_gate_up_key(
                gpu,
                hipfire_dispatch::types::KernelKey::FusedGateUpMq3G256Lloyd,
                &layer.w_gate.buf,
                &layer.w_up.buf,
                &pbs.x_rot_batch,
                &pbs.gate_ffn_batch,
                &pbs.up_batch,
                layer.w_gate.m,
                layer.w_up.m,
                layer.w_gate.k,
                n,
            )?;
        } else if fa_ffn_is_mq3 {
            run_fused_gate_up_key(
                gpu,
                hipfire_dispatch::types::KernelKey::FusedGateUpHfq3G256,
                &layer.w_gate.buf,
                &layer.w_up.buf,
                &pbs.x_rot_batch,
                &pbs.gate_ffn_batch,
                &pbs.up_batch,
                layer.w_gate.m,
                layer.w_up.m,
                layer.w_gate.k,
                n,
            )?;
        } else if fa_ffn_is_fp4 {
            run_fused_gate_up_key(
                gpu,
                hipfire_dispatch::types::KernelKey::FusedGateUpHfp4G32,
                &layer.w_gate.buf,
                &layer.w_up.buf,
                &pbs.x_rot_batch,
                &pbs.gate_ffn_batch,
                &pbs.up_batch,
                layer.w_gate.m,
                layer.w_up.m,
                layer.w_gate.k,
                n,
            )?;
        } else {
            run_fused_gate_up_key(
                gpu,
                hipfire_dispatch::types::KernelKey::FusedGateUpHfq4G256,
                &layer.w_gate.buf,
                &layer.w_up.buf,
                &pbs.x_rot_batch,
                &pbs.gate_ffn_batch,
                &pbs.up_batch,
                layer.w_gate.m,
                layer.w_up.m,
                layer.w_gate.k,
                n,
            )?;
        }
        Ok(())
    }

    pub(crate) fn resid_down_swiglu(&mut self, gpu: &mut Gpu) -> HipResult<()> {
        let layer = self.layer;
        let pbs = self.pbs;
        let config = self.config;
        let n = self.n;
        let q8_wmma_arch = gpu.arch_caps.has_wmma();
        let f16_prefill_wmma = qwen35_f16_prefill_wmma_enabled(gpu);
        let hidden_dim = config.hidden_dim;

        let fa_w_down_is_mq = matches!(
            layer.w_down.gpu_dtype,
            DType::MQ4G256
                | DType::MQ6G256
                | DType::MQ3G256
                | DType::MQ3G256Lloyd
                | DType::MFP4G32
                | DType::Oq4G256
                // Opus W8A8 needs the SAME FWHT rotation as W4A4 — its
                // weights are rotated offline too. Omitting it here fed the
                // oq8 GEMM an unrotated activation (garbage: PPL 3.5e6).
                | DType::Oq8G256
                | DType::OqCompactG256
        );
        let fa_w_down_is_6bit = matches!(layer.w_down.gpu_dtype, DType::MQ6G256 | DType::HFQ6G256);
        let fa_w_down_is_mq3 = matches!(layer.w_down.gpu_dtype, DType::MQ3G256);
        let fa_w_down_is_mq3_lloyd = matches!(layer.w_down.gpu_dtype, DType::MQ3G256Lloyd);
        let fa_w_down_is_fp4 = matches!(layer.w_down.gpu_dtype, DType::HFP4G32 | DType::MFP4G32);
        let fa_w_down_is_oq4 = matches!(layer.w_down.gpu_dtype, DType::Oq4G256);
        let fa_w_down_is_oq8 = matches!(layer.w_down.gpu_dtype, DType::Oq8G256);
        let fa_w_down_is_q8 = matches!(layer.w_down.gpu_dtype, DType::Q8_0);
        let fa_w_down_is_f32 = matches!(layer.w_down.gpu_dtype, DType::F32);
        let fa_w_down_is_f16 = matches!(layer.w_down.gpu_dtype, DType::F16 | DType::BF16);
        if fa_w_down_is_mq {
            // F2: AWQ-aware silu_mul+rotate for FullAttention w_down input.
            fused_silu_mul_rotate_mq_batched_for(
                gpu,
                &layer.w_down,
                &pbs.gate_ffn_batch,
                &pbs.up_batch,
                &pbs.ffn_hidden_batch,
                hidden_dim,
                n,
            )?;
        } else {
            gpu.silu_mul_f32(&pbs.gate_ffn_batch, &pbs.up_batch, &pbs.ffn_hidden_batch)?;
        }
        if fa_w_down_is_6bit {
            run_residual_gemm_key(
                gpu,
                hipfire_dispatch::types::KernelKey::GemmHfq6G256Residual,
                &layer.w_down.buf,
                layer.w_down.gpu_dtype,
                &pbs.ffn_hidden_batch,
                &pbs.x_batch,
                layer.w_down.m,
                layer.w_down.k,
                n,
            )?;
                } else if matches!(layer.w_down.gpu_dtype, DType::OqCompactG256) {
                    // Compact-resident Opus: same W8A8 math as the oq8 arm,
                    // decoding OqPlusCompact blocks in-kernel.
                    let bs = super::prefill_batch::oq_compact_block_stride(&layer.w_down)?;
                    gpu.gemm_oq_compact_residual_act_batched(
                        &layer.w_down.buf,
                        &pbs.ffn_hidden_batch,
                        &pbs.x_batch,
                        layer.w_down.m,
                        layer.w_down.k,
                        n,
                        bs,
                    )?;
        } else if fa_w_down_is_oq8 {
            // Opus W8A8: ffn_hidden_batch is FWHT-rotated above.
            gpu.gemm_oq8_grouped_residual_act_batched(
                &layer.w_down.buf,
                &pbs.ffn_hidden_batch,
                &pbs.x_batch,
                layer.w_down.m,
                layer.w_down.k,
                n,
            )?;
        } else if fa_w_down_is_oq4 {
            // Opus W4A4: ffn_hidden_batch is FWHT(+AWQ)-rotated above.
            gpu.gemm_oq4_grouped_residual_act_batched(
                &layer.w_down.buf,
                &pbs.ffn_hidden_batch,
                &pbs.x_batch,
                layer.w_down.m,
                layer.w_down.k,
                n,
            )?;
        } else if fa_w_down_is_q8 && q8_wmma_arch {
            let x_n = pbs.x_batch.sub_offset(0, n * layer.w_down.m);
            run_residual_gemm_key(
                gpu,
                hipfire_dispatch::types::KernelKey::GemmQ8_0ResidualWmma,
                &layer.w_down.buf,
                layer.w_down.gpu_dtype,
                &pbs.ffn_hidden_batch,
                &x_n,
                layer.w_down.m,
                layer.w_down.k,
                n,
            )?;
        } else if fa_w_down_is_q8 {
            let scratch = pbs.x_rot_batch.sub_offset(0, n * layer.w_down.m);
            run_plain_gemm_key(
                gpu,
                hipfire_dispatch::types::KernelKey::GemmQ8_0BatchedChunked,
                &layer.w_down.buf,
                layer.w_down.gpu_dtype,
                &pbs.ffn_hidden_batch,
                &scratch,
                layer.w_down.m,
                layer.w_down.k,
                n,
            )?;
            let x_n = pbs.x_batch.sub_offset(0, n * layer.w_down.m);
            gpu.add_inplace_f32(&x_n, &scratch)?;
        } else if fa_w_down_is_f32 {
            gemm_f32_residual_batched(
                gpu,
                &layer.w_down.buf,
                &pbs.ffn_hidden_batch,
                &pbs.x_batch,
                &pbs.x_rot_batch,
                layer.w_down.m,
                layer.w_down.k,
                n,
            )?;
        } else if fa_w_down_is_f16 {
            if f16_prefill_wmma {
                gemm_raw_x_f32_residual_batched_auto(
                    gpu,
                    &layer.w_down.buf,
                    &pbs.ffn_hidden_batch,
                    &pbs.x_batch,
                    &pbs.x_rot_batch,
                    layer.w_down.m,
                    layer.w_down.k,
                    n,
                )?;
            } else {
                gpu.gemv_f16_xf32_residual_batched(
                    &layer.w_down.buf,
                    &pbs.ffn_hidden_batch,
                    &pbs.x_batch,
                    layer.w_down.m,
                    layer.w_down.k,
                    n,
                )?;
            }
        } else if fa_w_down_is_mq3_lloyd {
            run_residual_gemm_key(
                gpu,
                hipfire_dispatch::types::KernelKey::GemmMq3G256LloydResidual,
                &layer.w_down.buf,
                layer.w_down.gpu_dtype,
                &pbs.ffn_hidden_batch,
                &pbs.x_batch,
                layer.w_down.m,
                layer.w_down.k,
                n,
            )?;
        } else if fa_w_down_is_mq3 {
            run_residual_gemm_key(
                gpu,
                hipfire_dispatch::types::KernelKey::GemmHfq3G256Residual,
                &layer.w_down.buf,
                layer.w_down.gpu_dtype,
                &pbs.ffn_hidden_batch,
                &pbs.x_batch,
                layer.w_down.m,
                layer.w_down.k,
                n,
            )?;
        } else if fa_w_down_is_fp4 {
            run_residual_gemm_key(
                gpu,
                hipfire_dispatch::types::KernelKey::GemmHfp4G32Residual,
                &layer.w_down.buf,
                layer.w_down.gpu_dtype,
                &pbs.ffn_hidden_batch,
                &pbs.x_batch,
                layer.w_down.m,
                layer.w_down.k,
                n,
            )?;
        } else {
            run_residual_gemm_key(
                gpu,
                hipfire_dispatch::types::KernelKey::GemmHfq4G256Residual,
                &layer.w_down.buf,
                layer.w_down.gpu_dtype,
                &pbs.ffn_hidden_batch,
                &pbs.x_batch,
                layer.w_down.m,
                layer.w_down.k,
                n,
            )?;
        }
        Ok(())
    }
}

impl<'a> ForwardBindings for Qwen35PrefillBindings<'a> {
    fn run_proj(
        &mut self,
        gpu: &mut Gpu,
        _ctx: &DispatchCtx,
        op: &OpBinding,
    ) -> Result<(), DispatchError> {
        let res: HipResult<()> = match op_code(op) {
            q35_op::PROJ_QKV => self.proj_qkv(gpu),
            q35_op::PROJ_GATE_UP => self.proj_gate_up(gpu),
            other => Err(HipError::new(0, &format!("prefill PROJ opcode {other}"))),
        };
        res.map_err(|e| DispatchError::Hip(e.to_string()))
    }

    fn run_attend(
        &mut self,
        gpu: &mut Gpu,
        ctx: &DispatchCtx,
        op: &OpBinding,
    ) -> Result<(), DispatchError> {
        let res: HipResult<()> = match op_code(op) {
            q35_op::ATTEND_FULL => self.attend_full(gpu, ctx),
            other => Err(HipError::new(0, &format!("prefill ATTEND opcode {other}"))),
        };
        res.map_err(|e| DispatchError::Hip(e.to_string()))
    }

    fn run_residual_gemv(
        &mut self,
        gpu: &mut Gpu,
        _ctx: &DispatchCtx,
        op: &OpBinding,
    ) -> Result<(), DispatchError> {
        let res: HipResult<()> = match op_code(op) {
            q35_op::RESID_WO => self.resid_wo(gpu),
            q35_op::RESID_DOWN_SWIGLU => self.resid_down_swiglu(gpu),
            other => Err(HipError::new(0, &format!("prefill RESID opcode {other}"))),
        };
        res.map_err(|e| DispatchError::Hip(e.to_string()))
    }

    // The dense FullAttn program contains no Norm / Moe / Recurrent / Conv op.
    // These are NOT stubs to be filled in later by a generic per-token body:
    // reaching one means a non-FullAttn program was handed to these bindings,
    // and the plan's §6 stop-line is explicit that turning such a case into
    // numerics is the accept-and-miscompute failure M2a exists to prevent. Fail
    // loudly instead.
    fn run_norm(
        &mut self,
        _gpu: &mut Gpu,
        _ctx: &DispatchCtx,
        _op: &OpBinding,
    ) -> Result<(), DispatchError> {
        Err(DispatchError::Hip(
            "prefill bindings: Norm super-op on a dense FullAttn program (§M2a4 owns DeltaNet)"
                .into(),
        ))
    }

    fn run_moe(
        &mut self,
        _gpu: &mut Gpu,
        _ctx: &DispatchCtx,
        _op: &OpBinding,
    ) -> Result<(), DispatchError> {
        Err(DispatchError::Hip(
            "prefill bindings: Moe super-op on a dense FullAttn program (§M2a4 owns the MoE arms)"
                .into(),
        ))
    }

    fn run_recurrent(
        &mut self,
        _gpu: &mut Gpu,
        _ctx: &DispatchCtx,
        _op: &OpBinding,
    ) -> Result<(), DispatchError> {
        Err(DispatchError::Hip(
            "prefill bindings: Recurrent super-op on a dense FullAttn program (§M2a4)".into(),
        ))
    }

    fn run_conv(
        &mut self,
        _gpu: &mut Gpu,
        _ctx: &DispatchCtx,
        _op: &OpBinding,
    ) -> Result<(), DispatchError> {
        Err(DispatchError::Hip(
            "prefill bindings: Conv super-op on a dense FullAttn program (§M2a4)".into(),
        ))
    }

    /// `lower_variant` emits no `Escape` for any qwen35 variant, so reaching
    /// this means the program was not one of the four it produces.
    fn run_escape(
        &mut self,
        _gpu: &mut Gpu,
        _ctx: &DispatchCtx,
        _op: &OpBinding,
        kind: EscapeKind,
    ) -> Result<(), DispatchError> {
        Err(DispatchError::Hip(format!(
            "prefill bindings: unexpected Escape super-op {kind:?}"
        )))
    }
}

/// §M2a4 — the DeltaNet (LinearAttention) batched-prefill bindings.
///
/// Same shape and the same argument as [`Qwen35PrefillBindings`], for the seven
/// super-ops of `lower_variant(Q35Variant::DeltaNet)`. The bodies are the
/// batched kernel sequences that sat inline in `prefill_chunk.rs`'s
/// `(DeltaNet, LinearAttention)` arm, moved unchanged and cut where that arm's
/// own comments already named the boundaries:
///
/// | super-op | was |
/// |---|---|
/// | `PROJ_QKVZA` | rmsnorm(+FWHT) preamble, 4-way LA projection |
/// | `ATTEND_DN_PREP` | sigmoid/alpha gate, conv1d, L2-norm + repeat-interleave |
/// | `RECUR_GDN` | the gated-delta-net recurrent scan |
/// | `NORM_GATED` | `gated_norm_f32_batched` |
/// | `RESID_WO` | wo residual |
/// | `PROJ_GATE_UP` | FFN rmsnorm(+rotate), gate+up |
/// | `RESID_DOWN_SWIGLU` | silu_mul(+rotate), w_down residual |
///
/// A SEPARATE struct from the FullAttn one because the two carry different
/// layer-weight types (`DeltaNetLayerWeights` vs `FullAttnLayerWeights`); a
/// single struct would have to re-match the variant in all seven methods and
/// duplicate the bodies anyway.
///
/// `dn_state` is a SHARED reference even though the scan mutates the S
/// matrices: they are `GpuTensor` device buffers, written by kernels, exactly as
/// the decode `Qwen35Bindings` documents at `lowered.rs:139`.
pub(crate) struct Qwen35PrefillDnBindings<'a> {
    pub(crate) layer: &'a DeltaNetLayerWeights,
    pub(crate) tree_verify: Option<TreeVerifyCtx<'a>>,
    /// Chunk-level decision (`prefill_chunk.rs`), not a per-layer one.
    pub(crate) use_gdn_per_token: bool,
    pub(crate) pbs: &'a PrefillBatchScratch,
    pub(crate) config: &'a Qwen35Config,
    pub(crate) dn_state: &'a DeltaNetState,
    pub(crate) gdn_tape: Option<&'a crate::speculative::GdnTape>,
    pub(crate) n: usize,
    pub(crate) tape_offset: usize,
    pub(crate) delta_layer_idx: usize,
}

impl<'a> Qwen35PrefillDnBindings<'a> {
    pub(crate) fn proj_qkvza(&mut self, gpu: &mut Gpu) -> HipResult<()> {
        let layer = self.layer;
        let pbs = self.pbs;
        let config = self.config;
        let n = self.n;
        let dim = config.dim;
        let gdn_tape = self.gdn_tape;
        let tape_offset = self.tape_offset;
        let delta_layer_idx = self.delta_layer_idx;
        let n_v_heads = config.linear_num_value_heads;
        let q8_wmma_arch = gpu.arch_caps.has_wmma();
        let f16_prefill_wmma = qwen35_f16_prefill_wmma_enabled(gpu);
        // Defined in the DN_PREP segment upstream and read here; re-derived
        // rather than threaded, since it is a pure projection of `tree_verify`.

        // Per-layer dtype branch: MQ4 needs FWHT-rotation on the
        // activation to match its pre-rotated weights; HFQ4 uses
        // plain rmsnormed activations. The GEMM kernels themselves
        // are dtype-agnostic — they just consume whatever [N × K]
        // activation buffer we point them at.
        // GAP NOTE: this matcher (and the 7 sibling dense LA/FA
        // matchers in this file) wires MQ3G256Lloyd through the
        // gemm_*_mq3g256_lloyd_wmma family. MQ2G256Lloyd remains
        // unwired — to add it, update is_batchable_la, ALL 8 is_mq*
        // matchers, AND add a Lloyd-MQ2-specific GEMM dispatch arm
        // together (the all-together corruption-prevention rule from
        // docs/plans/mq-lloyd-batched-prefill-followup.md). MQ4-Lloyd
        // is wired in a separate PR (issue #182).
        let is_mq = matches!(
            layer.wqkv.gpu_dtype,
            DType::MQ4G256
                | DType::MQ6G256
                | DType::MQ3G256
                | DType::MQ3G256Lloyd
                | DType::MFP4G32
                // Opus W4A4: weights FWHT-rotated offline → x must be
                // FWHT(+AWQ)-rotated by the shared mq rotate path before
                // the int4 activation quantize (decode parity:
                // rotate_x_mq[_awq] → quantize_act_oq4).
                | DType::Oq4G256
                // Opus W8A8 needs the SAME FWHT rotation as W4A4 — its
                // weights are rotated offline too. Omitting it here fed the
                // oq8 GEMM an unrotated activation (garbage: PPL 3.5e6).
                | DType::Oq8G256
                | DType::OqCompactG256
        );
        let is_6bit = matches!(layer.wqkv.gpu_dtype, DType::MQ6G256 | DType::HFQ6G256);
        let is_mq3 = matches!(layer.wqkv.gpu_dtype, DType::MQ3G256);
        let is_mq3_lloyd = matches!(layer.wqkv.gpu_dtype, DType::MQ3G256Lloyd);
        let is_fp4 = matches!(layer.wqkv.gpu_dtype, DType::HFP4G32 | DType::MFP4G32);
        let is_oq4 = matches!(layer.wqkv.gpu_dtype, DType::Oq4G256);
        let is_oq8 = matches!(layer.wqkv.gpu_dtype, DType::Oq8G256);
        let is_q8 = matches!(layer.wqkv.gpu_dtype, DType::Q8_0);
        let is_f32 = matches!(layer.wqkv.gpu_dtype, DType::F32);
        let is_f16 = matches!(layer.wqkv.gpu_dtype, DType::F16 | DType::BF16);

        // Batched rmsnorm (+ FWHT for MQ) for the LA preamble.
        // x_batch / x_rot_batch are [N × dim] contiguous. For HFQ
        // we reuse x_rot_batch as the "normed, unrotated" output
        // so the subsequent GEMM can read it the same way.
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

        // Batched 4-way LA projection (wqkv + wz + w_beta + w_alpha).
        if is_6bit {
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
            // `is_q8` only inspects `wqkv` (the routing anchor). The fused
            // kernel assumes ALL four weights share the Q8_0 stride; a
            // mixed-dtype layer would silently re-introduce the Tier-1
            // kernel-vs-stride corruption mode.
            debug_assert!(
                matches!(layer.wz.gpu_dtype, DType::Q8_0)
                    && matches!(layer.w_beta.gpu_dtype, DType::Q8_0)
                    && matches!(layer.w_alpha.gpu_dtype, DType::Q8_0),
                "LA qkvza Q8 WMMA dispatch requires all of wqkv/wz/w_beta/w_alpha to be Q8_0",
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
            // (wqkv/wz/w_beta/w_alpha) → GemmFamily::run_key with the
            // GemmQ8_0BatchedChunked dispatcher-entry key → identical
            // gpu.gemm_q8_0_batched_chunked method, byte-for-byte.
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
        } else if is_f32 {
            debug_assert!(
                matches!(layer.wz.gpu_dtype, DType::F32)
                    && matches!(layer.w_beta.gpu_dtype, DType::F32)
                    && matches!(layer.w_alpha.gpu_dtype, DType::F32),
                "LA qkvza F32 dispatch requires all of wqkv/wz/w_beta/w_alpha to be F32",
            );
            gpu.gemm_f32_register_tiled(
                &layer.wqkv.buf,
                &pbs.x_rot_batch,
                &pbs.dn_qkv_batch,
                layer.wqkv.m,
                layer.wqkv.k,
                n,
            )?;
            gpu.gemm_f32_register_tiled(
                &layer.wz.buf,
                &pbs.x_rot_batch,
                &pbs.dn_z_batch,
                layer.wz.m,
                layer.wz.k,
                n,
            )?;
            gpu.gemm_f32_register_tiled(
                &layer.w_beta.buf,
                &pbs.x_rot_batch,
                &pbs.dn_beta_batch,
                layer.w_beta.m,
                layer.w_beta.k,
                n,
            )?;
            gpu.gemm_f32_register_tiled(
                &layer.w_alpha.buf,
                &pbs.x_rot_batch,
                &pbs.dn_alpha_batch,
                layer.w_alpha.m,
                layer.w_alpha.k,
                n,
            )?;
        } else if is_f16 {
            debug_assert!(
                matches!(layer.wz.gpu_dtype, DType::F16 | DType::BF16)
                    && matches!(layer.w_beta.gpu_dtype, DType::F16 | DType::BF16)
                    && matches!(layer.w_alpha.gpu_dtype, DType::F16 | DType::BF16),
                "LA qkvza F16/BF16 dispatch requires all of wqkv/wz/w_beta/w_alpha to be F16",
            );
            if f16_prefill_wmma {
                gemm_raw_x_f32_auto(
                    gpu,
                    &layer.wqkv.buf,
                    &pbs.x_rot_batch,
                    &pbs.dn_qkv_batch,
                    layer.wqkv.m,
                    layer.wqkv.k,
                    n,
                )?;
                gemm_raw_x_f32_auto(
                    gpu,
                    &layer.wz.buf,
                    &pbs.x_rot_batch,
                    &pbs.dn_z_batch,
                    layer.wz.m,
                    layer.wz.k,
                    n,
                )?;
                gemm_raw_x_f32_auto(
                    gpu,
                    &layer.w_beta.buf,
                    &pbs.x_rot_batch,
                    &pbs.dn_beta_batch,
                    layer.w_beta.m,
                    layer.w_beta.k,
                    n,
                )?;
                gemm_raw_x_f32_auto(
                    gpu,
                    &layer.w_alpha.buf,
                    &pbs.x_rot_batch,
                    &pbs.dn_alpha_batch,
                    layer.w_alpha.m,
                    layer.w_alpha.k,
                    n,
                )?;
            } else {
                gpu.fused_qkvza_f16_xf32_batched(
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
        } else if is_mq3_lloyd {
            // 112 B/group Lloyd-MQ3 stride; X is already FWHT-rotated.
            run_fused_qkvza_key(
                gpu,
                hipfire_dispatch::types::KernelKey::FusedQkvzaMq3G256Lloyd,
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
        } else if is_mq3 {
            // 104 B/group HFQ3-stride; X is already FWHT-rotated by
            // fused_rmsnorm_rotate_mq_batched above. The FusedQkvzaHfq3G256
            // run-arm replicates the call-site WMMA-vs-base arch split
            // internally (gemm_qkvza_hfq3g256_wmma on has_wmma() else the
            // base cross-arch ladder), so the same kernel runs.
            run_fused_qkvza_key(
                gpu,
                hipfire_dispatch::types::KernelKey::FusedQkvzaHfq3G256,
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
        } else if is_fp4 {
            // HFP4G32: 17-B blocks (vs HFQ4's 136-B groups), per-row 16-B header.
            // MFP4G32: same storage as HFP4 + offline-FWHT weights; X is already
            // rotated above when is_mq, so this branch handles both unrotated
            // (HFP4) and post-rotation (MFP4) activations identically.
            run_fused_qkvza_key(
                gpu,
                hipfire_dispatch::types::KernelKey::FusedQkvzaHfp4G32,
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
        } else if is_oq8 {
            // Opus W8A8: x_rot_batch is FWHT-rotated above (is_mq covers
            // Oq8G256 — types.rs maps it to RotationPlan::FwhtG256). No
            // fused oq8 PREFILL arm exists (FusedQkvzaOq8G256 resolves to
            // the decode GEMV), so run one grouped int8-WMMA GEMM per
            // projection. Each shares the same int8 activation quantize
            // via the batched scratch, so the redundancy is the quantize
            // launch, not a re-read of x.
            gpu.quantize_act_oq8_batched_interleaved(&pbs.x_rot_batch, layer.wqkv.m, layer.wqkv.k, n)?;
            for (w, y) in [
                (&layer.wqkv, &pbs.dn_qkv_batch),
                (&layer.wz, &pbs.dn_z_batch),
                (&layer.w_beta, &pbs.dn_beta_batch),
                (&layer.w_alpha, &pbs.dn_alpha_batch),
            ] {
                gpu.gemm_oq8_grouped_prequant(&w.buf, y, w.m, w.k, n)?;
            }
        } else if is_oq4 {
            // Opus W4A4: x_rot_batch is FWHT(+AWQ)-rotated above (is_mq).
            // The FusedQkvzaOq4G256 run-arm int4-quantizes it once then
            // runs the batched grouped-WMMA fused kernel.
            run_fused_qkvza_key(
                gpu,
                hipfire_dispatch::types::KernelKey::FusedQkvzaOq4G256,
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
        } else if gdn_tape.is_some() {
            gpu.gemm_qkvza_hfq4g256_exact(
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

        Ok(())
    }

    pub(crate) fn attend_dn_prep(&mut self, gpu: &mut Gpu) -> HipResult<()> {
        let layer = self.layer;
        let pbs = self.pbs;
        let config = self.config;
        let n = self.n;
        let gdn_tape = self.gdn_tape;
        let tape_offset = self.tape_offset;
        let delta_layer_idx = self.delta_layer_idx;
        let dn_state = self.dn_state;
        let n_v_heads = config.linear_num_value_heads;
        let k_dim = config.linear_num_key_heads * config.linear_key_head_dim;
        let v_dim = config.linear_num_value_heads * config.linear_value_head_dim;
        let hd = config.linear_key_head_dim;
        let tree_verify = self.tree_verify;
        // Defined in the DN_PREP segment upstream and read here; re-derived
        // rather than threaded, since it is a pure projection of `tree_verify`.

        // Fused sigmoid(beta) + alpha_gate(alpha) — [N × n_v_heads] each.
        gpu.fused_sigmoid_alpha_gate_f32_batched(
            &pbs.dn_beta_batch,
            &pbs.dn_alpha_batch,
            &layer.dt_bias,
            &layer.a_log,
            n_v_heads,
            n,
        )?;

        // DFlash tape capture: snap pre-conv1d qkv + post-sigmoid α/β
        // for this layer into the per-layer tape slots. The next LA
        // layer's fused_qkvza / fused_sigmoid_alpha_gate will overwrite
        // dn_qkv_batch / dn_{alpha,beta}_batch, so capture must happen
        // now (after sigmoid_alpha_gate, before conv1d consumes qkv).
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

        // Tree-aware dispatch gate: when the caller provides
        // parent_indices (Phase 3b+ of Task #101), swap the linear
        // conv1d + GDN for tree-walking variants that eliminate
        // sibling-subtree state cross-contamination. The tree
        // kernels are READ-ONLY on dn_state (don't advance it) —
        // caller runs linear replay on the accepted spine
        // post-acceptance to commit the trajectory.
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

        // Fused L2-norm(Q) + scale(Q) + L2-norm(K) + repeat-interleave
        // when n_key_heads < n_v_heads. One launch instead of two —
        // ~200µs saved per LA layer × ~30 LA layers ≈ 6ms per prefill
        // on A3B (R9700/gfx1201).
        //
        // The fused kernel reads q_raw/k_raw (unchanged on exit), so
        // the conv1d output is preserved if downstream readers need it
        // (no current consumer reads _raw after this).
        if config.linear_num_key_heads < n_v_heads {
            let ratio = n_v_heads / config.linear_num_key_heads;
            gpu.fused_qk_l2_norm_scale_interleave_f32_batched(
                &pbs.dn_q_raw_batch,
                &pbs.dn_k_raw_batch,
                &pbs.dn_q_batch,
                &pbs.dn_k_batch,
                config.linear_num_key_heads,
                ratio,
                hd,
                1.0 / (hd as f32).sqrt(),
                config.norm_eps,
                n,
            )?;
        } else {
            // n_key_heads == n_v_heads → no replication; keep the
            // original sequence (norm in place, then memcpy).
            gpu.fused_qk_l2_norm_scale_f32_batched(
                &pbs.dn_q_raw_batch,
                &pbs.dn_k_raw_batch,
                config.linear_num_key_heads,
                hd,
                1.0 / (hd as f32).sqrt(),
                config.norm_eps,
                n,
            )?;
            gpu.memcpy_dtod_auto(&pbs.dn_q_batch.buf, &pbs.dn_q_raw_batch.buf, n * k_dim * 4)?;
            gpu.memcpy_dtod_auto(&pbs.dn_k_batch.buf, &pbs.dn_k_raw_batch.buf, n * k_dim * 4)?;
        }

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

        Ok(())
    }

    pub(crate) fn recur_gdn(&mut self, gpu: &mut Gpu) -> HipResult<()> {
        let pbs = self.pbs;
        let config = self.config;
        let n = self.n;
        let gdn_tape = self.gdn_tape;
        let tape_offset = self.tape_offset;
        let delta_layer_idx = self.delta_layer_idx;
        let dn_state = self.dn_state;
        let n_v_heads = config.linear_num_value_heads;
        let v_dim = config.linear_num_value_heads * config.linear_value_head_dim;
        let tree_verify = self.tree_verify;
        let use_gdn_per_token = self.use_gdn_per_token;
        // Defined in the DN_PREP segment upstream and read here; re-derived
        // rather than threaded, since it is a pure projection of `tree_verify`.
        let tree_parents = tree_verify.as_ref().and_then(|c| c.parent_indices);

        // Gated Delta Net — tree variant reads per-token S from
        // s_tape[parent] (or pre-block s_init at root); linear
        // variant advances dn_state.s_matrices in place.
        if let Some(parents) = tree_parents {
            let tape = pbs.dn_s_tape.as_ref().expect(
                "tree-aware LA requires dn_s_tape scratch (check PrefillBatchScratch::new)",
            );
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
            // Chunkwise-parallel: the tokens in this batch are resolved
            // together instead of one at a time. Same recurrence (see
            // `hipfire_rdna::gdn_chunk`), different summation order, so
            // it is NOT bit-identical to the serial arm below.
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
            // FP16 only — the FP32 arm above is batched, and batched
            // vs per-token is identical there (no narrowing). f16
            // narrows once per launch, so per-token matters.
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
            // EXPERIMENT (not #417): mirror the state-quant dispatch the
            // decode siblings already do (forward_scratch_layers:13194),
            // so the captured/eager batched prefill honours FP32/Q4 state
            // instead of forcing the Q8 kernel onto non-Q8 buffers.
            // #18: the GDN recurrence for this layer already ran above and
            // advanced `dn_state.s_matrices[delta_layer_idx]` IN PLACE. The
            // former re-dispatch here ran the same recurrence a SECOND time
            // over the same tokens, double-advancing the state and clobbering
            // `dn_attn_out_batch` with a value computed from the doubly-
            // advanced state. The tape copy above must stay; the re-dispatch
            // must not.
        }

        // Batched gated output norm.
        Ok(())
    }

    pub(crate) fn norm_gated(&mut self, gpu: &mut Gpu) -> HipResult<()> {
        let layer = self.layer;
        let pbs = self.pbs;
        let config = self.config;
        let n = self.n;
        let gdn_tape = self.gdn_tape;
        let tape_offset = self.tape_offset;
        let delta_layer_idx = self.delta_layer_idx;
        let n_v_heads = config.linear_num_value_heads;
        // Defined in the DN_PREP segment upstream and read here; re-derived
        // rather than threaded, since it is a pure projection of `tree_verify`.

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

        // Batched wo + residual.
        //
        // For MQ weights, the decode path's weight_gemv_residual
        // internally FWHT-rotates dn_normed into mq_x_rot before
        // calling gemv_hfq{4,6}g256_residual (MQ weights are pre-rotated
        // at quant time; math requires dot(rot(W), rot(x)) = dot(W,x)).
        // For HFQ weights no rotation is needed — the activation
        // feeds gemm_hfq{4,6}g256_residual directly.
        Ok(())
    }

    pub(crate) fn resid_wo(&mut self, gpu: &mut Gpu) -> HipResult<()> {
        let layer = self.layer;
        let pbs = self.pbs;
        let n = self.n;
        let gdn_tape = self.gdn_tape;
        let tape_offset = self.tape_offset;
        let delta_layer_idx = self.delta_layer_idx;
        let q8_wmma_arch = gpu.arch_caps.has_wmma();
        let f16_prefill_wmma = qwen35_f16_prefill_wmma_enabled(gpu);
        // Defined in the DN_PREP segment upstream and read here; re-derived
        // rather than threaded, since it is a pure projection of `tree_verify`.

        let wo_is_mq = matches!(
            layer.wo.gpu_dtype,
            DType::MQ4G256
                | DType::MQ6G256
                | DType::MQ3G256
                | DType::MQ3G256Lloyd
                | DType::MFP4G32
                | DType::Oq4G256
                // Opus W8A8 needs the SAME FWHT rotation as W4A4 — its
                // weights are rotated offline too. Omitting it here fed the
                // oq8 GEMM an unrotated activation (garbage: PPL 3.5e6).
                | DType::Oq8G256
                | DType::OqCompactG256
        );
        let wo_is_6bit = matches!(layer.wo.gpu_dtype, DType::MQ6G256 | DType::HFQ6G256);
        let wo_is_mq3 = matches!(layer.wo.gpu_dtype, DType::MQ3G256);
        let wo_is_mq3_lloyd = matches!(layer.wo.gpu_dtype, DType::MQ3G256Lloyd);
        let wo_is_fp4 = matches!(layer.wo.gpu_dtype, DType::HFP4G32 | DType::MFP4G32);
        let wo_is_oq4 = matches!(layer.wo.gpu_dtype, DType::Oq4G256);
        let wo_is_oq8 = matches!(layer.wo.gpu_dtype, DType::Oq8G256);
        let wo_is_q8 = matches!(layer.wo.gpu_dtype, DType::Q8_0);
        let wo_is_f32 = matches!(layer.wo.gpu_dtype, DType::F32);
        let wo_is_f16 = matches!(layer.wo.gpu_dtype, DType::F16 | DType::BF16);
        let wo_input = if wo_is_mq {
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
        } else {
            &pbs.dn_normed_batch
        };
        if let Some(tape) = gdn_tape.as_ref() {
            let v_row_bytes = tape.v_dim * 4;
            let off_v = tape_offset * v_row_bytes;
            gpu.memcpy_dtod_at_auto(
                &tape.wo_input_bufs[delta_layer_idx].buf,
                off_v,
                &wo_input.buf,
                0,
                n * v_row_bytes,
            )?;
        }
        if wo_is_6bit {
            run_residual_gemm_key(
                gpu,
                hipfire_dispatch::types::KernelKey::GemmHfq6G256Residual,
                &layer.wo.buf,
                layer.wo.gpu_dtype,
                wo_input,
                &pbs.x_batch,
                layer.wo.m,
                layer.wo.k,
                n,
            )?;
                } else if matches!(layer.wo.gpu_dtype, DType::OqCompactG256) {
                    // Compact-resident Opus o_proj: identical W8A8 math to the oq8
                    // arm below, decoding the OqPlusCompact blocks in-kernel.
                    let bs = super::prefill_batch::oq_compact_block_stride(&layer.wo)?;
                    gpu.gemm_oq_compact_residual_act_batched(
                        &layer.wo.buf,
                        wo_input,
                        &pbs.x_batch,
                        layer.wo.m,
                        layer.wo.k,
                        n,
                        bs,
                    )?;
        } else if wo_is_oq8 {
            // Opus W8A8 o_proj: grouped int8-WMMA GEMM into scratch +
            // residual add (no fused oq8 residual kernel), mirroring the
            // oq4 arm below.
            gpu.gemm_oq8_grouped_residual_act_batched(
                &layer.wo.buf,
                wo_input,
                &pbs.x_batch,
                layer.wo.m,
                layer.wo.k,
                n,
            )?;
        } else if wo_is_oq4 {
            // Opus W4A4: wo_input is FWHT(+AWQ)-rotated above (wo_is_mq).
            // No fused oq4 residual kernel → grouped-WMMA GEMM into scratch
            // + add into the residual stream (pbs.x_batch).
            // A4 KLD gate: HIPFIRE_OQ4_PREFILL_ACT_BITS[_O]=16 uses the
            // W4A16 residual variant (act16 baseline), =8 the int8-MMQ
            // residual variant; default = W4A4. o_proj is the most
            // activation-sensitive oq4 site (plan §13c), so A8 here is
            // the cheapest real mixed-precision lever.
            let o_bits = oq4_act_bits("O");
            if o_bits.as_deref() == Some("16") {
                gpu.gemm_oq4_grouped_residual_f16_batched(
                    &layer.wo.buf,
                    wo_input,
                    &pbs.x_batch,
                    layer.wo.m,
                    layer.wo.k,
                    n,
                )?;
            } else if o_bits.as_deref() == Some("8") {
                let x_n = pbs.x_batch.sub_offset(0, n * layer.wo.m);
                gpu.gemm_oq4_residual_mmq(
                    &layer.wo.buf,
                    wo_input,
                    &x_n,
                    layer.wo.m,
                    layer.wo.k,
                    n,
                    true,
                )?;
            } else {
                gpu.gemm_oq4_grouped_residual_act_batched(
                    &layer.wo.buf,
                    wo_input,
                    &pbs.x_batch,
                    layer.wo.m,
                    layer.wo.k,
                    n,
                )?;
            }
        } else if wo_is_q8 && q8_wmma_arch {
            let x_n = pbs.x_batch.sub_offset(0, n * layer.wo.m);
            run_residual_gemm_key(
                gpu,
                hipfire_dispatch::types::KernelKey::GemmQ8_0ResidualWmma,
                &layer.wo.buf,
                layer.wo.gpu_dtype,
                wo_input,
                &x_n,
                layer.wo.m,
                layer.wo.k,
                n,
            )?;
        } else if wo_is_q8 {
            // Tier 2 fallback (non-WMMA archs): GEMM into x_rot_batch as
            // scratch (safe — next consumer is the FFN rmsnorm), then
            // add into residual.
            let scratch = pbs.x_rot_batch.sub_offset(0, n * layer.wo.m);
            run_plain_gemm_key(
                gpu,
                hipfire_dispatch::types::KernelKey::GemmQ8_0BatchedChunked,
                &layer.wo.buf,
                layer.wo.gpu_dtype,
                wo_input,
                &scratch,
                layer.wo.m,
                layer.wo.k,
                n,
            )?;
            let x_n = pbs.x_batch.sub_offset(0, n * layer.wo.m);
            gpu.add_inplace_f32(&x_n, &scratch)?;
        } else if wo_is_f32 {
            gemm_f32_residual_batched(
                gpu,
                &layer.wo.buf,
                wo_input,
                &pbs.x_batch,
                &pbs.x_rot_batch,
                layer.wo.m,
                layer.wo.k,
                n,
            )?;
        } else if wo_is_f16 {
            if f16_prefill_wmma {
                gemm_raw_x_f32_residual_batched_auto(
                    gpu,
                    &layer.wo.buf,
                    wo_input,
                    &pbs.x_batch,
                    &pbs.x_rot_batch,
                    layer.wo.m,
                    layer.wo.k,
                    n,
                )?;
            } else {
                gpu.gemv_f16_xf32_residual_batched(
                    &layer.wo.buf,
                    wo_input,
                    &pbs.x_batch,
                    layer.wo.m,
                    layer.wo.k,
                    n,
                )?;
            }
        } else if wo_is_mq3_lloyd {
            run_residual_gemm_key(
                gpu,
                hipfire_dispatch::types::KernelKey::GemmMq3G256LloydResidual,
                &layer.wo.buf,
                layer.wo.gpu_dtype,
                wo_input,
                &pbs.x_batch,
                layer.wo.m,
                layer.wo.k,
                n,
            )?;
        } else if wo_is_mq3 {
            run_residual_gemm_key(
                gpu,
                hipfire_dispatch::types::KernelKey::GemmHfq3G256Residual,
                &layer.wo.buf,
                layer.wo.gpu_dtype,
                wo_input,
                &pbs.x_batch,
                layer.wo.m,
                layer.wo.k,
                n,
            )?;
        } else if wo_is_fp4 {
            run_residual_gemm_key(
                gpu,
                hipfire_dispatch::types::KernelKey::GemmHfp4G32Residual,
                &layer.wo.buf,
                layer.wo.gpu_dtype,
                wo_input,
                &pbs.x_batch,
                layer.wo.m,
                layer.wo.k,
                n,
            )?;
        } else if gdn_tape.is_some() {
            gpu.gemm_hfq4g256_residual_exact(
                &layer.wo.buf,
                wo_input,
                &pbs.x_batch,
                layer.wo.m,
                layer.wo.k,
                n,
            )?;
        } else {
            run_residual_gemm_key(
                gpu,
                hipfire_dispatch::types::KernelKey::GemmHfq4G256Residual,
                &layer.wo.buf,
                layer.wo.gpu_dtype,
                wo_input,
                &pbs.x_batch,
                layer.wo.m,
                layer.wo.k,
                n,
            )?;
        }

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

        Ok(())
    }

    pub(crate) fn proj_gate_up(&mut self, gpu: &mut Gpu) -> HipResult<()> {
        let layer = self.layer;
        let pbs = self.pbs;
        let config = self.config;
        let n = self.n;
        let dim = config.dim;
        let gdn_tape = self.gdn_tape;
        let tape_offset = self.tape_offset;
        let delta_layer_idx = self.delta_layer_idx;
        let q8_wmma_arch = gpu.arch_caps.has_wmma();
        let f16_prefill_wmma = qwen35_f16_prefill_wmma_enabled(gpu);
        // Defined in the DN_PREP segment upstream and read here; re-derived
        // rather than threaded, since it is a pure projection of `tree_verify`.

        // FFN: rmsnorm (+ rotate for MQ).
        let ffn_is_mq = matches!(
            layer.w_gate.gpu_dtype,
            DType::MQ4G256
                | DType::MQ6G256
                | DType::MQ3G256
                | DType::MQ3G256Lloyd
                | DType::MFP4G32
                | DType::Oq4G256
                // Opus W8A8 needs the SAME FWHT rotation as W4A4 — its
                // weights are rotated offline too. Omitting it here fed the
                // oq8 GEMM an unrotated activation (garbage: PPL 3.5e6).
                | DType::Oq8G256
                | DType::OqCompactG256
        );
        let ffn_is_6bit = matches!(layer.w_gate.gpu_dtype, DType::MQ6G256 | DType::HFQ6G256);
        let ffn_is_mq3 = matches!(layer.w_gate.gpu_dtype, DType::MQ3G256);
        let ffn_is_mq3_lloyd = matches!(layer.w_gate.gpu_dtype, DType::MQ3G256Lloyd);
        let ffn_is_fp4 = matches!(layer.w_gate.gpu_dtype, DType::HFP4G32 | DType::MFP4G32);
        let ffn_is_oq4 = matches!(layer.w_gate.gpu_dtype, DType::Oq4G256);
        let ffn_is_oq8 = matches!(layer.w_gate.gpu_dtype, DType::Oq8G256);
        let ffn_is_q8 = matches!(layer.w_gate.gpu_dtype, DType::Q8_0);
        let ffn_is_f32 = matches!(layer.w_gate.gpu_dtype, DType::F32);
        let ffn_is_f16 = matches!(layer.w_gate.gpu_dtype, DType::F16 | DType::BF16);
        if ffn_is_mq {
            // AWQ-aware: next linear is w_gate (gate/up share input → same AWQ scale).
            fused_rmsnorm_rotate_mq_batched_for(
                gpu,
                &pbs.x_batch,
                &layer.ffn_norm,
                &layer.w_gate,
                &pbs.x_rot_batch,
                dim,
                config.norm_eps,
                n,
            )?;
        } else {
            gpu.rmsnorm_batched(
                &pbs.x_batch,
                &layer.ffn_norm,
                &pbs.x_rot_batch,
                n,
                dim,
                config.norm_eps,
            )?;
        }
        if let Some(tape) = gdn_tape.as_ref() {
            let hidden_row_bytes = tape.x_in_dim * 4;
            let off_hidden = tape_offset * hidden_row_bytes;
            gpu.memcpy_dtod_at_auto(
                &tape.ffn_input_bufs[delta_layer_idx].buf,
                off_hidden,
                &pbs.x_rot_batch.buf,
                0,
                n * hidden_row_bytes,
            )?;
        }

        // Batched gate+up projection.
        // #397 Ship 5.2 slice 2: fused gate+up dtypes → FusedQkvFamily
        // (batched-prefill gate+up variant) via run_fused_gate_up_key.
        // The Q8-non-WMMA case stays as two plain GemmQ8_0BatchedChunked
        // GEMMs (not a fused kernel — slice 1). The HFQ3 WMMA-vs-base
        // split is folded into the FusedGateUpHfq3G256 run-arm, which
        // re-derives it from gpu.arch_caps.has_wmma() (== arch_has_wmma).
        if ffn_is_6bit {
            run_fused_gate_up_key(
                gpu,
                hipfire_dispatch::types::KernelKey::FusedGateUpHfq6G256,
                &layer.w_gate.buf,
                &layer.w_up.buf,
                &pbs.x_rot_batch,
                &pbs.gate_ffn_batch,
                &pbs.up_batch,
                layer.w_gate.m,
                layer.w_up.m,
                layer.w_gate.k,
                n,
            )?;
        } else if ffn_is_q8 && q8_wmma_arch {
            debug_assert!(
                matches!(layer.w_up.gpu_dtype, DType::Q8_0),
                "LA FFN Q8 WMMA dispatch requires both w_gate and w_up to be Q8_0",
            );
            run_fused_gate_up_key(
                gpu,
                hipfire_dispatch::types::KernelKey::FusedGateUpQ8_0,
                &layer.w_gate.buf,
                &layer.w_up.buf,
                &pbs.x_rot_batch,
                &pbs.gate_ffn_batch,
                &pbs.up_batch,
                layer.w_gate.m,
                layer.w_up.m,
                layer.w_gate.k,
                n,
            )?;
        } else if ffn_is_q8 {
            run_plain_gemm_key(
                gpu,
                hipfire_dispatch::types::KernelKey::GemmQ8_0BatchedChunked,
                &layer.w_gate.buf,
                layer.w_gate.gpu_dtype,
                &pbs.x_rot_batch,
                &pbs.gate_ffn_batch,
                layer.w_gate.m,
                layer.w_gate.k,
                n,
            )?;
            run_plain_gemm_key(
                gpu,
                hipfire_dispatch::types::KernelKey::GemmQ8_0BatchedChunked,
                &layer.w_up.buf,
                layer.w_up.gpu_dtype,
                &pbs.x_rot_batch,
                &pbs.up_batch,
                layer.w_up.m,
                layer.w_up.k,
                n,
            )?;
        } else if ffn_is_f32 {
            debug_assert!(
                matches!(layer.w_up.gpu_dtype, DType::F32),
                "LA FFN F32 dispatch requires both w_gate and w_up to be F32",
            );
            gpu.gemm_f32_register_tiled(
                &layer.w_gate.buf,
                &pbs.x_rot_batch,
                &pbs.gate_ffn_batch,
                layer.w_gate.m,
                layer.w_gate.k,
                n,
            )?;
            gpu.gemm_f32_register_tiled(
                &layer.w_up.buf,
                &pbs.x_rot_batch,
                &pbs.up_batch,
                layer.w_up.m,
                layer.w_up.k,
                n,
            )?;
        } else if ffn_is_f16 {
            debug_assert!(
                matches!(layer.w_up.gpu_dtype, DType::F16 | DType::BF16),
                "LA FFN F16/BF16 dispatch requires both w_gate and w_up to be F16",
            );
            if f16_prefill_wmma {
                gemm_raw_x_f32_auto(
                    gpu,
                    &layer.w_gate.buf,
                    &pbs.x_rot_batch,
                    &pbs.gate_ffn_batch,
                    layer.w_gate.m,
                    layer.w_gate.k,
                    n,
                )?;
                gemm_raw_x_f32_auto(
                    gpu,
                    &layer.w_up.buf,
                    &pbs.x_rot_batch,
                    &pbs.up_batch,
                    layer.w_up.m,
                    layer.w_up.k,
                    n,
                )?;
            } else {
                gpu.fused_gate_up_f16_xf32_batched(
                    &layer.w_gate.buf,
                    &layer.w_up.buf,
                    &pbs.x_rot_batch,
                    &pbs.gate_ffn_batch,
                    &pbs.up_batch,
                    layer.w_gate.m,
                    layer.w_up.m,
                    layer.w_gate.k,
                    n,
                )?;
            }
        } else if ffn_is_mq3_lloyd {
            run_fused_gate_up_key(
                gpu,
                hipfire_dispatch::types::KernelKey::FusedGateUpMq3G256Lloyd,
                &layer.w_gate.buf,
                &layer.w_up.buf,
                &pbs.x_rot_batch,
                &pbs.gate_ffn_batch,
                &pbs.up_batch,
                layer.w_gate.m,
                layer.w_up.m,
                layer.w_gate.k,
                n,
            )?;
        } else if ffn_is_mq3 {
            run_fused_gate_up_key(
                gpu,
                hipfire_dispatch::types::KernelKey::FusedGateUpHfq3G256,
                &layer.w_gate.buf,
                &layer.w_up.buf,
                &pbs.x_rot_batch,
                &pbs.gate_ffn_batch,
                &pbs.up_batch,
                layer.w_gate.m,
                layer.w_up.m,
                layer.w_gate.k,
                n,
            )?;
        } else if ffn_is_fp4 {
            run_fused_gate_up_key(
                gpu,
                hipfire_dispatch::types::KernelKey::FusedGateUpHfp4G32,
                &layer.w_gate.buf,
                &layer.w_up.buf,
                &pbs.x_rot_batch,
                &pbs.gate_ffn_batch,
                &pbs.up_batch,
                layer.w_gate.m,
                layer.w_up.m,
                layer.w_gate.k,
                n,
            )?;
                } else if matches!(layer.w_gate.gpu_dtype, DType::OqCompactG256) {
                    // Compact-resident Opus: one quantize of the shared rotated
                    // activation, then one compact GEMM per projection.
                    gpu.quantize_act_oq8_batched_interleaved(
                        &pbs.x_rot_batch,
                        layer.w_gate.m,
                        layer.w_gate.k,
                        n,
                    )?;
                    for (w, y) in [
                        (&layer.w_gate, &pbs.gate_ffn_batch),
                        (&layer.w_up, &pbs.up_batch),
                    ] {
                        let bs = super::prefill_batch::oq_compact_block_stride(w)?;
                        gpu.gemm_oq_compact_grouped_prequant(&w.buf, y, w.m, w.k, n, bs)?;
                    }
        } else if ffn_is_oq8 {
            // Opus W8A8 gate+up: two grouped int8-WMMA GEMMs into the
            // same buffers the fused kernel writes; downstream silu_mul
            // is unchanged.
            gpu.quantize_act_oq8_batched_interleaved(&pbs.x_rot_batch, layer.w_gate.m, layer.w_gate.k, n)?;
            for (w, y) in [
                (&layer.w_gate, &pbs.gate_ffn_batch),
                (&layer.w_up, &pbs.up_batch),
            ] {
                gpu.gemm_oq8_grouped_prequant(&w.buf, y, w.m, w.k, n)?;
            }
        } else if ffn_is_oq4 {
            // Opus W4A4: x_rot_batch is FWHT(+AWQ)-rotated above (ffn_is_mq).
            // CAREFUL — the default here is NOT int4 activation. The
            // `FusedGateUpOq4G256` dispatch key routes to
            // `gemm_oq4_gate_up_mmq` (int8 MMQ) whenever n >= 64, falling
            // back to f16-WMMA below that; the int4 activation path is
            // never taken at prefill batch sizes. So gate_up has always
            // run at A8, including under a global `=4` (plan §13i).
            //
            // =16 unfuses to two W4A16 GEMMs, =8 pins the int8-MMQ pair
            // explicitly at any n, and =4 forces the TRUE int4-activation
            // path (two grouped-act GEMMs) — which nothing reached before,
            // so "full W4A4" numbers predating §13i all had gate_up at A8.
            // Default (unset) keeps the existing routing untouched.
            // The downstream silu_mul is identical in every case.
            let gate_up_bits = oq4_act_bits("GATEUP");
            if gate_up_bits.as_deref() == Some("4") {
                gpu.gemm_oq4_grouped_act_batched(
                    &layer.w_gate.buf,
                    &pbs.x_rot_batch,
                    &pbs.gate_ffn_batch,
                    layer.w_gate.m,
                    layer.w_gate.k,
                    n,
                )?;
                gpu.gemm_oq4_grouped_act_batched(
                    &layer.w_up.buf,
                    &pbs.x_rot_batch,
                    &pbs.up_batch,
                    layer.w_up.m,
                    layer.w_up.k,
                    n,
                )?;
            } else if gate_up_bits.as_deref() == Some("8") {
                gpu.gemm_oq4_gate_up_mmq(
                    &layer.w_gate.buf,
                    &layer.w_up.buf,
                    &pbs.x_rot_batch,
                    &pbs.gate_ffn_batch,
                    &pbs.up_batch,
                    layer.w_gate.m,
                    layer.w_up.m,
                    layer.w_gate.k,
                    n,
                )?;
            } else if gate_up_bits.as_deref() == Some("16") {
                gpu.gemm_oq4_grouped_f16_wmma(
                    &layer.w_gate.buf,
                    &pbs.x_rot_batch,
                    &pbs.gate_ffn_batch,
                    layer.w_gate.m,
                    layer.w_gate.k,
                    n,
                    256,
                )?;
                gpu.gemm_oq4_grouped_f16_wmma(
                    &layer.w_up.buf,
                    &pbs.x_rot_batch,
                    &pbs.up_batch,
                    layer.w_up.m,
                    layer.w_up.k,
                    n,
                    256,
                )?;
            } else {
                run_fused_gate_up_key(
                    gpu,
                    hipfire_dispatch::types::KernelKey::FusedGateUpOq4G256,
                    &layer.w_gate.buf,
                    &layer.w_up.buf,
                    &pbs.x_rot_batch,
                    &pbs.gate_ffn_batch,
                    &pbs.up_batch,
                    layer.w_gate.m,
                    layer.w_up.m,
                    layer.w_gate.k,
                    n,
                )?;
            }
        } else if gdn_tape.is_some() {
            gpu.gemm_gate_up_hfq4g256_exact(
                &layer.w_gate.buf,
                &layer.w_up.buf,
                &pbs.x_rot_batch,
                &pbs.gate_ffn_batch,
                &pbs.up_batch,
                layer.w_gate.m,
                layer.w_up.m,
                layer.w_gate.k,
                n,
            )?;
        } else {
            run_fused_gate_up_key(
                gpu,
                hipfire_dispatch::types::KernelKey::FusedGateUpHfq4G256,
                &layer.w_gate.buf,
                &layer.w_up.buf,
                &pbs.x_rot_batch,
                &pbs.gate_ffn_batch,
                &pbs.up_batch,
                layer.w_gate.m,
                layer.w_up.m,
                layer.w_gate.k,
                n,
            )?;
        }
        if let Some(tape) = gdn_tape.as_ref() {
            let ffn_row_bytes = tape.ffn_dim * 4;
            let off_ffn = tape_offset * ffn_row_bytes;
            gpu.memcpy_dtod_at_auto(
                &tape.ffn_gate_bufs[delta_layer_idx].buf,
                off_ffn,
                &pbs.gate_ffn_batch.buf,
                0,
                n * ffn_row_bytes,
            )?;
            gpu.memcpy_dtod_at_auto(
                &tape.ffn_up_bufs[delta_layer_idx].buf,
                off_ffn,
                &pbs.up_batch.buf,
                0,
                n * ffn_row_bytes,
            )?;
        }

        Ok(())
    }

    pub(crate) fn resid_down_swiglu(&mut self, gpu: &mut Gpu) -> HipResult<()> {
        let layer = self.layer;
        let pbs = self.pbs;
        let config = self.config;
        let n = self.n;
        let hidden_dim = config.hidden_dim;
        let gdn_tape = self.gdn_tape;
        let tape_offset = self.tape_offset;
        let delta_layer_idx = self.delta_layer_idx;
        let q8_wmma_arch = gpu.arch_caps.has_wmma();
        let f16_prefill_wmma = qwen35_f16_prefill_wmma_enabled(gpu);
        // Defined in the DN_PREP segment upstream and read here; re-derived
        // rather than threaded, since it is a pure projection of `tree_verify`.

        // SwiGLU activation feeding w_down. For MQ, we need the
        // output FWHT-rotated so it matches the pre-rotated w_down
        // weights. For HFQ, plain silu_mul is enough. silu_mul_f32
        // is purely element-wise and uses numel() as its length,
        // so a [N × hidden_dim] tensor processes all rows in one
        // launch with no batch offset needed.
        let w_down_is_mq = matches!(
            layer.w_down.gpu_dtype,
            DType::MQ4G256
                | DType::MQ6G256
                | DType::MQ3G256
                | DType::MQ3G256Lloyd
                | DType::MFP4G32
                | DType::Oq4G256
                // Opus W8A8 needs the SAME FWHT rotation as W4A4 — its
                // weights are rotated offline too. Omitting it here fed the
                // oq8 GEMM an unrotated activation (garbage: PPL 3.5e6).
                | DType::Oq8G256
                | DType::OqCompactG256
        );
        let w_down_is_6bit = matches!(layer.w_down.gpu_dtype, DType::MQ6G256 | DType::HFQ6G256);
        let w_down_is_mq3 = matches!(layer.w_down.gpu_dtype, DType::MQ3G256);
        let w_down_is_mq3_lloyd = matches!(layer.w_down.gpu_dtype, DType::MQ3G256Lloyd);
        let w_down_is_fp4 = matches!(layer.w_down.gpu_dtype, DType::HFP4G32 | DType::MFP4G32);
        let w_down_is_oq4 = matches!(layer.w_down.gpu_dtype, DType::Oq4G256);
        let w_down_is_oq8 = matches!(layer.w_down.gpu_dtype, DType::Oq8G256);
        let w_down_is_q8 = matches!(layer.w_down.gpu_dtype, DType::Q8_0);
        let w_down_is_f32 = matches!(layer.w_down.gpu_dtype, DType::F32);
        let w_down_is_f16 = matches!(layer.w_down.gpu_dtype, DType::F16 | DType::BF16);
        if w_down_is_mq {
            // F2: AWQ-aware silu_mul+rotate for w_down input.
            fused_silu_mul_rotate_mq_batched_for(
                gpu,
                &layer.w_down,
                &pbs.gate_ffn_batch,
                &pbs.up_batch,
                &pbs.ffn_hidden_batch,
                hidden_dim,
                n,
            )?;
        } else {
            gpu.silu_mul_f32(&pbs.gate_ffn_batch, &pbs.up_batch, &pbs.ffn_hidden_batch)?;
        }
        if let Some(tape) = gdn_tape.as_ref() {
            let hidden_row_bytes = tape.x_in_dim * 4;
            let off_hidden = tape_offset * hidden_row_bytes;
            gpu.memcpy_dtod_at_auto(
                &tape.w_down_residual_in_bufs[delta_layer_idx].buf,
                off_hidden,
                &pbs.x_batch.buf,
                0,
                n * hidden_row_bytes,
            )?;
            let ffn_row_bytes = tape.ffn_dim * 4;
            let off_ffn = tape_offset * ffn_row_bytes;
            gpu.memcpy_dtod_at_auto(
                &tape.w_down_input_bufs[delta_layer_idx].buf,
                off_ffn,
                &pbs.ffn_hidden_batch.buf,
                0,
                n * ffn_row_bytes,
            )?;
        }

        // Batched w_down + residual.
        if w_down_is_6bit {
            run_residual_gemm_key(
                gpu,
                hipfire_dispatch::types::KernelKey::GemmHfq6G256Residual,
                &layer.w_down.buf,
                layer.w_down.gpu_dtype,
                &pbs.ffn_hidden_batch,
                &pbs.x_batch,
                layer.w_down.m,
                layer.w_down.k,
                n,
            )?;
                } else if matches!(layer.w_down.gpu_dtype, DType::OqCompactG256) {
                    // Compact-resident Opus: same W8A8 math as the oq8 arm,
                    // decoding OqPlusCompact blocks in-kernel.
                    let bs = super::prefill_batch::oq_compact_block_stride(&layer.w_down)?;
                    gpu.gemm_oq_compact_residual_act_batched(
                        &layer.w_down.buf,
                        &pbs.ffn_hidden_batch,
                        &pbs.x_batch,
                        layer.w_down.m,
                        layer.w_down.k,
                        n,
                        bs,
                    )?;
        } else if w_down_is_oq8 {
            // Opus W8A8 down: grouped int8-WMMA GEMM + residual add.
            gpu.gemm_oq8_grouped_residual_act_batched(
                &layer.w_down.buf,
                &pbs.ffn_hidden_batch,
                &pbs.x_batch,
                layer.w_down.m,
                layer.w_down.k,
                n,
            )?;
        } else if w_down_is_oq4 {
            // Opus W4A4: ffn_hidden_batch is FWHT(+AWQ)-rotated above
            // (fused_silu_mul_rotate_mq, w_down_is_mq). grouped-WMMA GEMM
            // into scratch + residual add into the hidden stream.
            // A4 KLD gate: [_DOWN]=16 uses the W4A16 residual variant,
            // =8 the int8-MMQ one (down is the 2nd most act-sensitive
            // oq4 site after o_proj — plan §13c).
            let down_bits = oq4_act_bits("DOWN");
            if down_bits.as_deref() == Some("16") {
                gpu.gemm_oq4_grouped_residual_f16_batched(
                    &layer.w_down.buf,
                    &pbs.ffn_hidden_batch,
                    &pbs.x_batch,
                    layer.w_down.m,
                    layer.w_down.k,
                    n,
                )?;
            } else if down_bits.as_deref() == Some("8") {
                let x_n = pbs.x_batch.sub_offset(0, n * layer.w_down.m);
                gpu.gemm_oq4_residual_mmq(
                    &layer.w_down.buf,
                    &pbs.ffn_hidden_batch,
                    &x_n,
                    layer.w_down.m,
                    layer.w_down.k,
                    n,
                    true,
                )?;
            } else {
                gpu.gemm_oq4_grouped_residual_act_batched(
                    &layer.w_down.buf,
                    &pbs.ffn_hidden_batch,
                    &pbs.x_batch,
                    layer.w_down.m,
                    layer.w_down.k,
                    n,
                )?;
            }
        } else if w_down_is_q8 && q8_wmma_arch {
            let x_n = pbs.x_batch.sub_offset(0, n * layer.w_down.m);
            run_residual_gemm_key(
                gpu,
                hipfire_dispatch::types::KernelKey::GemmQ8_0ResidualWmma,
                &layer.w_down.buf,
                layer.w_down.gpu_dtype,
                &pbs.ffn_hidden_batch,
                &x_n,
                layer.w_down.m,
                layer.w_down.k,
                n,
            )?;
        } else if w_down_is_q8 {
            let scratch = pbs.x_rot_batch.sub_offset(0, n * layer.w_down.m);
            run_plain_gemm_key(
                gpu,
                hipfire_dispatch::types::KernelKey::GemmQ8_0BatchedChunked,
                &layer.w_down.buf,
                layer.w_down.gpu_dtype,
                &pbs.ffn_hidden_batch,
                &scratch,
                layer.w_down.m,
                layer.w_down.k,
                n,
            )?;
            let x_n = pbs.x_batch.sub_offset(0, n * layer.w_down.m);
            gpu.add_inplace_f32(&x_n, &scratch)?;
        } else if w_down_is_f32 {
            gemm_f32_residual_batched(
                gpu,
                &layer.w_down.buf,
                &pbs.ffn_hidden_batch,
                &pbs.x_batch,
                &pbs.x_rot_batch,
                layer.w_down.m,
                layer.w_down.k,
                n,
            )?;
        } else if w_down_is_f16 {
            if f16_prefill_wmma {
                gemm_raw_x_f32_residual_batched_auto(
                    gpu,
                    &layer.w_down.buf,
                    &pbs.ffn_hidden_batch,
                    &pbs.x_batch,
                    &pbs.x_rot_batch,
                    layer.w_down.m,
                    layer.w_down.k,
                    n,
                )?;
            } else {
                gpu.gemv_f16_xf32_residual_batched(
                    &layer.w_down.buf,
                    &pbs.ffn_hidden_batch,
                    &pbs.x_batch,
                    layer.w_down.m,
                    layer.w_down.k,
                    n,
                )?;
            }
        } else if w_down_is_mq3_lloyd {
            run_residual_gemm_key(
                gpu,
                hipfire_dispatch::types::KernelKey::GemmMq3G256LloydResidual,
                &layer.w_down.buf,
                layer.w_down.gpu_dtype,
                &pbs.ffn_hidden_batch,
                &pbs.x_batch,
                layer.w_down.m,
                layer.w_down.k,
                n,
            )?;
        } else if w_down_is_mq3 {
            run_residual_gemm_key(
                gpu,
                hipfire_dispatch::types::KernelKey::GemmHfq3G256Residual,
                &layer.w_down.buf,
                layer.w_down.gpu_dtype,
                &pbs.ffn_hidden_batch,
                &pbs.x_batch,
                layer.w_down.m,
                layer.w_down.k,
                n,
            )?;
        } else if w_down_is_fp4 {
            run_residual_gemm_key(
                gpu,
                hipfire_dispatch::types::KernelKey::GemmHfp4G32Residual,
                &layer.w_down.buf,
                layer.w_down.gpu_dtype,
                &pbs.ffn_hidden_batch,
                &pbs.x_batch,
                layer.w_down.m,
                layer.w_down.k,
                n,
            )?;
        } else if gdn_tape.is_some() {
            gpu.gemm_hfq4g256_residual_exact(
                &layer.w_down.buf,
                &pbs.ffn_hidden_batch,
                &pbs.x_batch,
                layer.w_down.m,
                layer.w_down.k,
                n,
            )?;
        } else {
            run_residual_gemm_key(
                gpu,
                hipfire_dispatch::types::KernelKey::GemmHfq4G256Residual,
                &layer.w_down.buf,
                layer.w_down.gpu_dtype,
                &pbs.ffn_hidden_batch,
                &pbs.x_batch,
                layer.w_down.m,
                layer.w_down.k,
                n,
            )?;
        }
        Ok(())
    }
}

impl<'a> ForwardBindings for Qwen35PrefillDnBindings<'a> {
    fn run_proj(
        &mut self,
        gpu: &mut Gpu,
        _ctx: &DispatchCtx,
        op: &OpBinding,
    ) -> Result<(), DispatchError> {
        let res: HipResult<()> = match op_code(op) {
            q35_op::PROJ_QKVZA => self.proj_qkvza(gpu),
            q35_op::PROJ_GATE_UP => self.proj_gate_up(gpu),
            other => Err(HipError::new(0, &format!("prefill DN PROJ opcode {other}"))),
        };
        res.map_err(|e| DispatchError::Hip(e.to_string()))
    }

    fn run_attend(
        &mut self,
        gpu: &mut Gpu,
        _ctx: &DispatchCtx,
        op: &OpBinding,
    ) -> Result<(), DispatchError> {
        let res: HipResult<()> = match op_code(op) {
            q35_op::ATTEND_DN_PREP => self.attend_dn_prep(gpu),
            other => Err(HipError::new(
                0,
                &format!("prefill DN ATTEND opcode {other}"),
            )),
        };
        res.map_err(|e| DispatchError::Hip(e.to_string()))
    }

    fn run_recurrent(
        &mut self,
        gpu: &mut Gpu,
        _ctx: &DispatchCtx,
        op: &OpBinding,
    ) -> Result<(), DispatchError> {
        let res: HipResult<()> = match op_code(op) {
            q35_op::RECUR_GDN => self.recur_gdn(gpu),
            other => Err(HipError::new(
                0,
                &format!("prefill DN RECUR opcode {other}"),
            )),
        };
        res.map_err(|e| DispatchError::Hip(e.to_string()))
    }

    fn run_norm(
        &mut self,
        gpu: &mut Gpu,
        _ctx: &DispatchCtx,
        op: &OpBinding,
    ) -> Result<(), DispatchError> {
        let res: HipResult<()> = match op_code(op) {
            q35_op::NORM_GATED => self.norm_gated(gpu),
            other => Err(HipError::new(0, &format!("prefill DN NORM opcode {other}"))),
        };
        res.map_err(|e| DispatchError::Hip(e.to_string()))
    }

    fn run_residual_gemv(
        &mut self,
        gpu: &mut Gpu,
        _ctx: &DispatchCtx,
        op: &OpBinding,
    ) -> Result<(), DispatchError> {
        let res: HipResult<()> = match op_code(op) {
            q35_op::RESID_WO => self.resid_wo(gpu),
            q35_op::RESID_DOWN_SWIGLU => self.resid_down_swiglu(gpu),
            other => Err(HipError::new(
                0,
                &format!("prefill DN RESID opcode {other}"),
            )),
        };
        res.map_err(|e| DispatchError::Hip(e.to_string()))
    }

    // The dense DeltaNet program contains no Moe or Conv op — conv1d is inside
    // ATTEND_DN_PREP, matching the decode lowering. Reaching either means a
    // program that is not `lower_variant(DeltaNet)`; fail loudly rather than
    // improvise numerics (plan §6).
    fn run_moe(
        &mut self,
        _gpu: &mut Gpu,
        _ctx: &DispatchCtx,
        _op: &OpBinding,
    ) -> Result<(), DispatchError> {
        Err(DispatchError::Hip(
            "prefill DN bindings: Moe super-op on a dense DeltaNet program".into(),
        ))
    }

    fn run_conv(
        &mut self,
        _gpu: &mut Gpu,
        _ctx: &DispatchCtx,
        _op: &OpBinding,
    ) -> Result<(), DispatchError> {
        Err(DispatchError::Hip(
            "prefill DN bindings: Conv super-op (conv1d lives in ATTEND_DN_PREP)".into(),
        ))
    }

    fn run_escape(
        &mut self,
        _gpu: &mut Gpu,
        _ctx: &DispatchCtx,
        _op: &OpBinding,
        kind: EscapeKind,
    ) -> Result<(), DispatchError> {
        Err(DispatchError::Hip(format!(
            "prefill DN bindings: unexpected Escape super-op {kind:?}"
        )))
    }
}
