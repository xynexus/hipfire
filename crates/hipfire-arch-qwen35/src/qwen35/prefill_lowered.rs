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
        let qkv_same_dtype = layer.wk.gpu_dtype == layer.wq.gpu_dtype
            && layer.wv.gpu_dtype == layer.wq.gpu_dtype;
        let fa_bridge_tape_active = gdn_tape.as_ref().is_some_and(|tape| {
            delta_layer_idx < tape.fa_bridge_valid.len()
                && tape.fa_bridge_valid[delta_layer_idx]
        });
        if let Some(tape) = gdn_tape.as_ref() {
            if delta_layer_idx < tape.fa_bridge_valid.len()
                && tape.fa_bridge_valid[delta_layer_idx]
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
            if delta_layer_idx < tape.fa_bridge_valid.len()
                && tape.fa_bridge_valid[delta_layer_idx]
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
        } else if qkv_is_oq8 && qkv_same_dtype {
            // Opus W8A8 FA QKV: one grouped int8-WMMA GEMM per projection
            // off the shared FWHT-rotated activation.
            debug_assert!(
                matches!(layer.wk.gpu_dtype, DType::Oq8G256)
                    && matches!(layer.wv.gpu_dtype, DType::Oq8G256),
                "FA qkv Oq8 dispatch requires all of wq/wk/wv to be Oq8G256",
            );
            gpu.quantize_act_oq8_batched(&pbs.x_rot_batch, layer.wq.m, layer.wq.k, n)?;
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
        if let Some(tape) = gdn_tape.as_ref() {
            if delta_layer_idx < tape.fa_bridge_valid.len()
                && tape.fa_bridge_valid[delta_layer_idx]
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
            if delta_layer_idx < tape.fa_bridge_valid.len()
                && tape.fa_bridge_valid[delta_layer_idx]
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
            if delta_layer_idx < tape.fa_bridge_valid.len()
                && tape.fa_bridge_valid[delta_layer_idx]
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
                let bias_b =
                    q8_tree_bias.map(|bias| bias.sub_offset(b * block_cols, block_cols));
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
            if delta_layer_idx < tape.fa_bridge_valid.len()
                && tape.fa_bridge_valid[delta_layer_idx]
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

        qwen35_apply_fa_gate(gpu, config, &pbs.fa_attn_out_batch, &pbs.fa_gate_batch)?;
        if let Some(tape) = gdn_tape.as_ref() {
            if delta_layer_idx < tape.fa_bridge_valid.len()
                && tape.fa_bridge_valid[delta_layer_idx]
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
            if delta_layer_idx < tape.fa_bridge_valid.len()
                && tape.fa_bridge_valid[delta_layer_idx]
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
        );
        let fa_ffn_is_6bit =
            matches!(layer.w_gate.gpu_dtype, DType::MQ6G256 | DType::HFQ6G256);
        let fa_ffn_is_mq3 = matches!(layer.w_gate.gpu_dtype, DType::MQ3G256);
        let fa_ffn_is_mq3_lloyd = matches!(layer.w_gate.gpu_dtype, DType::MQ3G256Lloyd);
        let fa_ffn_is_fp4 =
            matches!(layer.w_gate.gpu_dtype, DType::HFP4G32 | DType::MFP4G32);
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
        } else if fa_ffn_is_oq8 {
            // Opus W8A8 gate+up: one grouped int8-WMMA GEMM per projection.
            gpu.quantize_act_oq8_batched(
                &pbs.x_rot_batch,
                layer.w_gate.m,
                layer.w_gate.k,
                n,
            )?;
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
        );
        let fa_w_down_is_6bit =
            matches!(layer.w_down.gpu_dtype, DType::MQ6G256 | DType::HFQ6G256);
        let fa_w_down_is_mq3 = matches!(layer.w_down.gpu_dtype, DType::MQ3G256);
        let fa_w_down_is_mq3_lloyd = matches!(layer.w_down.gpu_dtype, DType::MQ3G256Lloyd);
        let fa_w_down_is_fp4 =
            matches!(layer.w_down.gpu_dtype, DType::HFP4G32 | DType::MFP4G32);
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
