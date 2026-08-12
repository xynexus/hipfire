// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Qwen3.5 DeltaNet recurrent state: `DeltaNetState`, its quantization mode,
//! and the sizing helpers that pick redundancy / precision from config.

use super::*;

/// Persistent state for DeltaNet layers across tokens.
/// State quantization mode for DeltaNet S matrix.
/// DeltaNet recurrent-state storage precision.
///
/// Q8 and Q4 were REMOVED 2026-08-09 — not gated, removed. Quantized recurrent
/// state produced three separate silent failures: long-decode attractors on
/// low-redundancy models, a stochastic-rounding seed that leaked execution
/// history into target numerics (issue #17), and an `s_ef_residual` accumulator
/// `DeltaNetSnapshot` never saved, breaking spec-decode losslessness (#22).
/// Each degraded output rather than failing, which is why they persisted.
///
/// FP32 is the default and the numerical reference. FP16 is opt-in.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum StateQuant {
    FP32,
    /// Half-precision storage, FP32 arithmetic. Halves per-sequence state; no
    /// scales, no error-feedback buffer, no stochastic rounding, so it is
    /// deterministic and spec-decode rollback is lossless.
    FP16,
}

/// Redundancy proxy for DeltaNet recurrent-state precision:
/// `linear_key_head_dim × linear_num_value_heads`. The recurrent state is the
/// most precision-sensitive tensor in the model — its quant error compounds
/// across the sequence — so low-redundancy models (small `head_dim × n_heads`)
/// lack the slack to absorb Q8 noise and develop attractors on long decode
/// (observed on 0.8B/2B). State traffic is only ~1–3% of per-token bandwidth,
/// so paying FP32 here is nearly free.
pub fn deltanet_state_redundancy(config: &Qwen35Config) -> usize {
    config.linear_key_head_dim * config.linear_num_value_heads
}

/// Redundancy threshold below which the DeltaNet state defaults to FP32.
/// Env-tunable via `HIPFIRE_DN_STATE_FP32_BELOW`; defaults to `usize::MAX`
/// ⇒ **FP32 for all current models**.
///
/// # POLICY (2026-07-19): DeltaNet state must NEVER be Q8.
///
/// FP32 is the intended default, via this gate. The only sanctioned future
/// alternatives are **FP16** or a **purpose-built DeltaNet-state codec** (not yet
/// implemented). Q8 is not one of them.
///
/// This supersedes the earlier suggestion to lower the threshold (e.g. to 3000)
/// so 9B/27B would use Q8 — **do not do that.** Q8 state has produced, in this
/// repo's own history: long-decode attractors on low-redundancy models
/// (2026-06-15), a stochastic-rounding seed that leaked execution history into
/// target numerics (issue #17), and a rollback hazard where `s_ef_residual` — the
/// Q8-only error-feedback accumulator — is never saved or restored by
/// `DeltaNetSnapshot` (issue #22), which breaks spec-decode losslessness the
/// moment Q8 is enabled. All three are Q8-specific.
///
/// # Sizing (measured 2026-07-19) — why a codec is a CONCURRENCY question
///
/// State is **per-sequence**, unlike weights. FP32 totals (S matrices + conv):
/// 0.8B/2B 19.3 MB · 4B/9B 50.2 MB · 35B-A3B 62.8 MB · 27B and 122B-A10B
/// ~149 MB · 397B-A17B 186 MB. Spec decode **doubles** it (the rollback snapshot
/// holds a second copy), so 27B costs ~300 MB per spec-decoding session.
/// One session is negligible; 32 concurrent sessions on 27B is ~4.8 GB and 64 is
/// ~9.6 GB. FP16 halves it. A real codec only earns its complexity for
/// high-concurrency serving of 27B-class models.
///
/// # Formerly a known conflict
///
/// The tree DeltaNet replay path used to be Q8-only, so DDTree spec-decode could
/// not run under this policy. Resolved 2026-08-09:
/// `gated_delta_net_{f32,f16}_tree_batch_seq` are the tree kernels now and the
/// Q8 ones are deleted, so tree replay runs at whichever precision the state is.
/// Retained only so older references resolve; the redundancy threshold that
/// once selected Q8 above a size cutoff is gone with Q8 itself. State precision
/// is FP16 by default since 2026-08-12; `HIPFIRE_DN_STATE_FP16=0` opts out.
#[deprecated(note = "Q8 state was removed; precision is FP32 or opt-in FP16")]
pub fn deltanet_state_fp32_below() -> usize {
    usize::MAX
}

/// Default DeltaNet state precision, gated on redundancy (`head_dim × n_heads`)
/// rather than parameter count. Below the threshold → FP32 (the numerical
/// anchor); at/above → Q8.
///
/// **With the default threshold (`usize::MAX`) this always returns FP32, which is
/// the intended and only sanctioned behaviour — see the policy note on
/// [`deltanet_state_fp32_below`]. The Q8 arm is reachable only by lowering that
/// threshold, which is now explicitly disallowed.** The intended non-FP32 future
/// is FP16 or a DeltaNet-state codec, neither of which exists yet.
///
/// The `HIPFIRE_QWEN35_STATE_QUANT` override referenced in earlier revisions of
/// this comment does **not** exist — no code reads that variable.
pub fn default_state_quant(config: &Qwen35Config) -> StateQuant {
    let _ = config;
    // `.flag()`, NOT `.parse_or(false)`: `parse_or` goes through Rust's
    // `FromStr for bool`, which accepts ONLY "true"/"false", so `=1` parses as
    // Err and falls back silently. That cost a 24-minute KLD run which reported
    // "FP16" numbers identical to FP32 to the last digit because it had quietly
    // run FP32 twice.
    //
    // FP16 is the DEFAULT as of 2026-08-09, with FP32 as the opt-out. Evidence:
    // on Qwen3.5-35B-A3B, +0.68% mean KLD (0.039176 vs 0.038913), ~40x smaller
    // than the CI half-width; and on the low-redundancy 2B — where Q8 broke
    // first — greedy decode tracked FP32 EXACTLY for 720 tokens with
    // degeneration metrics unchanged (unique_ratio 0.3325 vs 0.3317, max_freq
    // 0.0450 vs 0.0458), against Q8's recorded signature of 0.625 -> 0.555 and
    // 0.055 -> 0.078.
    //
    // That evidence is one prompt on one model, which is why FP32 stays one
    // flag away rather than being deleted: it is the oracle, and losing the
    // ability to diff against it is how quantized state hid for months.
    //
    // FP16 was made the default and REVERTED the same day (2026-08-09): the
    // Q8 dispatch functions had never been deleted, only the `StateQuant`
    // variants that selected them, so surviving `else` arms still reached
    // them unconditionally and half-size FP16 state faulted with
    //   Memory Fault ... kernel: gated_delta_net_q8
    // The kernels, their dispatch entry points and every caller are now gone,
    // so that blocker is cleared.
    //
    // Made the default again 2026-08-12, on a CAPACITY argument rather than an
    // accuracy one. Per-session DeltaNet state is 30 layers x 2 MiB = 60 MiB on
    // the 35B-A3B, ~9x its KV cost, and it is what bounds concurrency: at width
    // 64 the FP32 state OOMs the GTT pool mid-prefill and only 19/64 sessions
    // complete. FP16 storage takes that to 64/64 with throughput monotonic in
    // width (5.96 / 6.25 / 6.31 tok/s at 16 / 32 / 64). See BUGS.md.
    //
    // The accuracy evidence is still one prompt on one model, so FP32 remains
    // one flag away and stays the oracle to diff against.
    // `is_off()`, not `!flag()`: unset must mean "not explicitly off", which is
    // the whole point of the opt-out spelling.
    if hipfire_env::DN_STATE_FP16.is_off() {
        StateQuant::FP32
    } else {
        StateQuant::FP16
    }
}

pub struct DeltaNetState {
    /// S matrix storage — FP32 or Q8 depending on quant mode
    pub s_matrices: Vec<GpuTensor>,
    /// Per-head scale factors (only used for Q8 mode)
    pub s_scales: Vec<GpuTensor>,
    /// Conv ring buffer: [n_deltanet_layers × conv_channels × (kernel_size-1)] FP32
    pub conv_states: Vec<GpuTensor>,
    /// Per-element f16 error-feedback residual for Q8 state requant (sigma-delta
    /// noise-shaping). Empty unless Q8 + `HIPFIRE_DN_STATE_EF`. Same element count
    /// as `s_matrices`; carries the previous step's quant error so the next
    /// requant cancels it — DeltaNet's contractive decay damps the shaped noise,
    /// yielding ~FP32-grade state at Q8's byte container.
    pub s_ef_residual: Vec<GpuTensor>,
    /// Current quantization mode
    pub quant: StateQuant,
}

impl DeltaNetState {
    /// EF residual for a delta-layer, if error-feedback is active (Q8 + flag).
    /// `None` ⇒ callers pass null ⇒ kernel uses the legacy stochastic-rounding requant.
    #[inline]
    pub fn ef_residual(&self, idx: usize) -> Option<&GpuTensor> {
        self.s_ef_residual.get(idx)
    }

    pub fn new(gpu: &mut Gpu, config: &Qwen35Config) -> HipResult<Self> {
        Self::new_with_quant(gpu, config, default_state_quant(config))
    }

    pub fn new_with_quant(
        gpu: &mut Gpu,
        config: &Qwen35Config,
        quant: StateQuant,
    ) -> HipResult<Self> {
        let n_delta_layers = config
            .layer_types
            .iter()
            .filter(|t| **t == LayerType::LinearAttention)
            .count();
        let s_dim = config.linear_key_head_dim; // 128
        let n_heads = config.linear_num_value_heads; // 16
        let s_size = n_heads * s_dim * s_dim; // 16 * 128 * 128 = 262144

        let conv_channels = config.linear_num_key_heads * config.linear_key_head_dim * 2
            + config.linear_num_value_heads * config.linear_value_head_dim;
        let conv_state_size = conv_channels * (config.conv_kernel_dim - 1);

        // Error-feedback (sigma-delta) requant for Q8 state — DEFAULT ON as of
        // 2026-06-08. q8_ef ≈ FP32 coherence at −0.7% decode vs FP32's −4.5% (best
        // spec-decode τ too), and far better than stochastic Q8 — DFlash 27b-prose
        // unique_ratio 0.625 vs 0.555, max_freq 0.055 vs 0.078. Also makes the DN
        // state DETERMINISTIC (no stochastic dither). Opt OUT with
        // HIPFIRE_DN_STATE_EF=0. Q8-only (FP32 has no requant; Q4 EF is future
        // work; the multi-GPU band split is still stochastic — new_with_quant_multi
        // leaves s_ef_residual empty). Residual is f16 per-element.
        // DISABLED 2026-07-19 with the Q8/Q4 state arms below — error feedback is
        // Q8-only, so this can no longer fire. Preserved as REFERENCE for the
        // DeltaNet-state codec: sigma-delta noise shaping is the technique that
        // made Q8 state viable at all (see the measurements above), and a codec
        // will likely want the same trick.
        //
        //   let ef_enabled = quant == StateQuant::Q8
        //       && std::env::var("HIPFIRE_DN_STATE_EF")
        //           .map(|v| v != "0")
        //           .unwrap_or(true);
        //
        // KNOWN DEFECT if this is ever restored: `s_ef_residual` is NOT saved or
        // restored by `DeltaNetSnapshot` (which covers only s_matrices, s_scales
        // and conv_states). A per-token recurrent buffer surviving rollback makes
        // committed output depend on accept length, hence on the drafter —
        // breaking spec-decode losslessness. Fix the snapshot first.
        let ef_enabled = false;

        let mut s_matrices = Vec::with_capacity(n_delta_layers);
        let mut s_scales = Vec::with_capacity(n_delta_layers);
        let mut conv_states = Vec::with_capacity(n_delta_layers);
        let mut s_ef_residual = Vec::with_capacity(if ef_enabled { n_delta_layers } else { 0 });
        for _ in 0..n_delta_layers {
            match quant {
                StateQuant::FP32 => {
                    s_matrices.push(gpu.zeros(&[s_size], DType::F32)?);
                    s_scales.push(gpu.zeros(&[n_heads], DType::F32)?);
                }
                // ── Q8 / Q4 DeltaNet state: DISABLED 2026-07-19 ──────────────
                //
                // Quantized recurrent state is disallowed by policy (see the
                // note on `deltanet_state_fp32_below`). The allocation bodies
                // are preserved verbatim below as REFERENCE for the eventual
                // DeltaNet-state codec — the layout decisions (byte-container
                // sizing, per-row vs per-head scales, nibble packing) are the
                // part worth keeping. Do NOT re-enable them as-is:
                // `s_ef_residual` is still missing from `DeltaNetSnapshot`, so
                // Q8 breaks spec-decode losslessness the moment it is restored.
                //
                //   StateQuant::Q8 => {
                //       // int8 state: s_size bytes (1 byte each), per-row scales
                //       let buf = gpu.hip.malloc(s_size)?;
                //       gpu.hip.memset(&buf, 0, s_size)?;
                //       s_matrices.push(GpuTensor {
                //           buf,
                //           shape: vec![s_size],
                //           dtype: DType::F32,
                //       });
                //       s_scales.push(gpu.zeros(&[n_heads * s_dim], DType::F32)?);
                //   }
                //   StateQuant::Q4 => {
                //       // 4-bit nibble-packed: s_size/2 bytes, per-row scales
                //       let buf = gpu.hip.malloc(s_size / 2)?;
                //       gpu.hip.memset(&buf, 0, s_size / 2)?;
                //       s_matrices.push(GpuTensor {
                //           buf,
                //           shape: vec![s_size / 2],
                //           dtype: DType::F32,
                //       });
                //       s_scales.push(gpu.zeros(&[n_heads * s_dim], DType::F32)?);
                //   }
                //
                // DDTree tree-replay no longer depends on any of this: it runs
                // on `gated_delta_net_{f32,f16}_tree_batch_seq`, picked from
                // the state precision. The Q8 tree kernel is deleted.
                StateQuant::FP16 => {
                    // Half the bytes. Raw because this is f16 storage the
                    // kernels widen on load — there is no f16 DType at this
                    // layer, and labelling it F32 would make every downstream
                    // size calculation wrong.
                    let buf = gpu.hip.malloc(s_size * 2)?;
                    gpu.hip.memset(&buf, 0, s_size * 2)?;
                    s_matrices.push(GpuTensor {
                        buf,
                        shape: vec![s_size],
                        dtype: DType::Raw,
                    });
                    // f16 needs no scales (per-element exponent); the vector
                    // survives only because callers still index it.
                    s_scales.push(gpu.zeros(&[n_heads], DType::F32)?);
                }
            }
            if ef_enabled {
                s_ef_residual.push(gpu.zeros(&[s_size], DType::F16)?);
            }
            conv_states.push(gpu.zeros(&[conv_state_size], DType::F32)?);
        }
        Ok(Self {
            s_matrices,
            s_scales,
            conv_states,
            s_ef_residual,
            quant,
        })
    }

    /// Free all GPU tensors. Call before drop to return VRAM.
    pub fn free_gpu(self, gpu: &mut Gpu) {
        for t in self.s_matrices {
            let _ = gpu.free_tensor(t);
        }
        for t in self.s_scales {
            let _ = gpu.free_tensor(t);
        }
        for t in self.conv_states {
            let _ = gpu.free_tensor(t);
        }
        for t in self.s_ef_residual {
            let _ = gpu.free_tensor(t);
        }
    }

    /// Reset all DeltaNet recurrent buffers to zero in place. Lets callers
    /// reuse a single `DeltaNetState` across independent chunks/sequences
    /// without allocating per chunk (which leaks since DeltaNetState has no
    /// Drop). Mirrors `ModelSlot::reset_state` in speculative.rs.
    pub fn reset(&mut self, gpu: &mut Gpu) {
        match gpu.active_stream.as_ref() {
            Some(stream) => {
                for s in &self.s_matrices {
                    let _ = gpu.hip.memset_async(&s.buf, 0, s.buf.size(), stream);
                }
                for s in &self.s_scales {
                    let _ = gpu.hip.memset_async(&s.buf, 0, s.buf.size(), stream);
                }
                for s in &self.conv_states {
                    let _ = gpu.hip.memset_async(&s.buf, 0, s.buf.size(), stream);
                }
                for s in &self.s_ef_residual {
                    let _ = gpu.hip.memset_async(&s.buf, 0, s.buf.size(), stream);
                }
            }
            None => {
                for s in &self.s_matrices {
                    let _ = gpu.hip.memset(&s.buf, 0, s.buf.size());
                }
                for s in &self.s_scales {
                    let _ = gpu.hip.memset(&s.buf, 0, s.buf.size());
                }
                for s in &self.conv_states {
                    let _ = gpu.hip.memset(&s.buf, 0, s.buf.size());
                }
                for s in &self.s_ef_residual {
                    let _ = gpu.hip.memset(&s.buf, 0, s.buf.size());
                }
            }
        }
    }

    /// Multi-GPU companion to `new_with_quant`. Each LA-layer's state is
    /// allocated on the device that owns the layer in the multi-GPU band
    /// split: `gpus.devices[gpus.device_for_layer(orig_layer_idx)]` for the
    /// `orig_layer_idx` of the LA-layer. Returns the state alongside the
    /// `la_to_device` mapping the daemon needs to route reset memsets to
    /// the correct device.
    pub fn new_with_quant_multi(
        gpus: &mut Gpus,
        config: &Qwen35Config,
        quant: StateQuant,
    ) -> HipResult<(Self, Vec<u8>)> {
        let s_dim = config.linear_key_head_dim;
        let n_heads = config.linear_num_value_heads;
        let s_size = n_heads * s_dim * s_dim;
        let conv_channels = config.linear_num_key_heads * config.linear_key_head_dim * 2
            + config.linear_num_value_heads * config.linear_value_head_dim;
        let conv_state_size = conv_channels * (config.conv_kernel_dim - 1);

        let mut s_matrices = Vec::new();
        let mut s_scales = Vec::new();
        let mut conv_states = Vec::new();
        let mut la_to_device: Vec<u8> = Vec::new();

        for (orig_layer_idx, lt) in config.layer_types.iter().enumerate() {
            if *lt != LayerType::LinearAttention {
                continue;
            }
            let dev_idx = gpus.device_for_layer(orig_layer_idx);
            la_to_device.push(dev_idx as u8);
            let g = &mut gpus.devices[dev_idx];
            // g.hip.malloc/memset bypass the Stage 2 bind_thread audit
            // (HipRuntime methods don't carry a device id). Bind explicitly
            // before any raw HIP ops so allocations land on the right device.
            g.bind_thread()?;
            match quant {
                StateQuant::FP32 => {
                    s_matrices.push(g.zeros(&[s_size], DType::F32)?);
                    s_scales.push(g.zeros(&[n_heads], DType::F32)?);
                }
                StateQuant::FP16 => {
                    let buf = g.hip.malloc(s_size * 2)?;
                    g.hip.memset(&buf, 0, s_size * 2)?;
                    s_matrices.push(GpuTensor {
                        buf,
                        shape: vec![s_size],
                        dtype: DType::Raw,
                    });
                    s_scales.push(g.zeros(&[n_heads], DType::F32)?);
                }
            }
            conv_states.push(g.zeros(&[conv_state_size], DType::F32)?);
        }
        Ok((
            Self {
                s_matrices,
                s_scales,
                conv_states,
                // EF residual not wired for the multi-GPU band split (would need
                // per-device residual alloc routed by device_for_layer); empty ⇒
                // ef_residual() returns None ⇒ kernel uses the stochastic path.
                s_ef_residual: Vec::new(),
                quant,
            },
            la_to_device,
        ))
    }

    /// Free per-LA-layer tensors on the devices listed in `la_to_device`
    /// (the second tuple element returned by `new_with_quant_multi`).
    pub fn free_gpu_multi(self, gpus: &mut Gpus, la_to_device: &[u8]) {
        for (i, t) in self.s_matrices.into_iter().enumerate() {
            let _ = gpus.devices[la_to_device[i] as usize].free_tensor(t);
        }
        for (i, t) in self.s_scales.into_iter().enumerate() {
            let _ = gpus.devices[la_to_device[i] as usize].free_tensor(t);
        }
        for (i, t) in self.conv_states.into_iter().enumerate() {
            let _ = gpus.devices[la_to_device[i] as usize].free_tensor(t);
        }
    }
}

/// Plug `DeltaNetState` into the neutral family-seam state container
/// (`hipfire_runtime::sequence_state`). The serving layer holds the recurrent
/// state as `Box<dyn RecurrentMixerState>` inside a `SequenceState`, and
/// recovers the concrete `&DeltaNetState` for its monomorphized hot path via
/// `SequenceState::recurrent_as::<DeltaNetState>()` — no per-token dyn cost.
/// See docs/plans/2026-06-23-seam-finish-and-mamba2.md (P2c, Slice 1).
impl hipfire_runtime::sequence_state::RecurrentMixerState for DeltaNetState {
    fn reset(&mut self, gpu: &mut Gpu) -> HipResult<()> {
        // The inherent `reset` zeros every recurrent buffer (s_matrices /
        // s_scales / conv_states / s_ef_residual) and swallows memset errors,
        // so it is infallible — always report Ok.
        DeltaNetState::reset(self, gpu);
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any> {
        self
    }
}
