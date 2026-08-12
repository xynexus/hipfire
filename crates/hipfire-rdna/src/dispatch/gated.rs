// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Gated-norm + gated-delta-net (DeltaNet/GLA) dispatch. Pure move (Phase 1 M2).

use super::{Gpu, GpuTensor};
use crate::kernels;
use hip_bridge::HipResult;
use std::ffi::c_void;

impl Gpu {
    /// Gated output norm: rmsnorm(x) * silu(z). Fused kernel.
    #[cfg(feature = "deltanet")]
    pub fn gated_norm_f32(
        &mut self,
        x: &GpuTensor,
        z: &GpuTensor,
        weight: &GpuTensor,
        out: &GpuTensor,
        n_heads: usize,
        head_dim: usize,
        eps: f32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("gated_norm", kernels::GATED_NORM_SRC, "gated_norm_f32")?;
        let xp = x.buf.as_ptr();
        let zp = z.buf.as_ptr();
        let wp = weight.buf.as_ptr();
        let op = out.buf.as_ptr();
        let nh = n_heads as i32;
        let hd = head_dim as i32;
        let ep = eps;
        let bytes = crate::profile::gated_norm_bytes(n_heads * head_dim);
        let timer = crate::profile::begin_timer(&self.hip, "rmsnorm", "gated_norm_f32", bytes);
        let result = self.launch_kernargs(
            "gated_norm_f32",
            [n_heads as u32, 1, 1],
            [32, 1, 1],
            0,
            &kernargs![ptr xp, ptr zp, ptr wp, ptr op, i32 nh, i32 hd, f32 ep],
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// Batched `gated_norm_f32`. Grid.y is the batch dim.
    #[cfg(feature = "deltanet")]
    pub fn gated_norm_f32_batched(
        &mut self,
        x: &GpuTensor,
        z: &GpuTensor,
        weight: &GpuTensor,
        out: &GpuTensor,
        n_heads: usize,
        head_dim: usize,
        eps: f32,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("gated_norm", kernels::GATED_NORM_SRC, "gated_norm_f32")?;
        let xp = x.buf.as_ptr();
        let zp = z.buf.as_ptr();
        let wp = weight.buf.as_ptr();
        let op = out.buf.as_ptr();
        let nh = n_heads as i32;
        let hd = head_dim as i32;
        let ep = eps;
        let bytes = crate::profile::gated_norm_bytes(n_heads * head_dim) * batch_size;
        let timer =
            crate::profile::begin_timer(&self.hip, "rmsnorm", "gated_norm_f32_batched", bytes);
        let result = self.launch_kernargs(
            "gated_norm_f32",
            [n_heads as u32, batch_size as u32, 1],
            [32, 1, 1],
            0,
            &kernargs![ptr xp, ptr zp, ptr wp, ptr op, i32 nh, i32 hd, f32 ep],
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    /// Gated Delta Net recurrence. S matrix in LDS. Processes all tokens sequentially.
    #[cfg(feature = "deltanet")]
    pub fn gated_delta_net_f32(
        &mut self,
        q: &GpuTensor,
        k: &GpuTensor,
        v: &GpuTensor,
        gate: &GpuTensor,
        beta: &GpuTensor,
        state: &GpuTensor,
        output: &GpuTensor,
        n_tokens: usize,
        n_heads: usize,
        head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        Self::ensure_gdn_hd128(head_dim)?;
        self.ensure_kernel(
            "gated_delta_net",
            kernels::GATED_DELTA_NET_SRC,
            "gated_delta_net_f32",
        )?;
        let func = &self.functions["gated_delta_net_f32"];
        let mut qp = q.buf.as_ptr();
        let mut kp = k.buf.as_ptr();
        let mut vp = v.buf.as_ptr();
        let mut gp = gate.buf.as_ptr();
        let mut bp = beta.buf.as_ptr();
        let mut sp = state.buf.as_ptr();
        let mut op = output.buf.as_ptr();
        let mut nt = n_tokens as i32;
        let mut nh = n_heads as i32;
        let mut hd = head_dim as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut qp as *mut _ as *mut c_void,
            &mut kp as *mut _ as *mut c_void,
            &mut vp as *mut _ as *mut c_void,
            &mut gp as *mut _ as *mut c_void,
            &mut bp as *mut _ as *mut c_void,
            &mut sp as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut nt as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
        ];
        // 32 threads, tiled S in LDS (4KB per tile). Grid: [n_heads, 128/8=16].
        let n_tiles = (128 / 4) as u32;
        unsafe {
            self.hip.launch_kernel(
                func,
                [n_heads as u32, n_tiles, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// Batched FP32-state Gated Delta Net recurrence. Processes all tokens
    /// sequentially inside the kernel and advances the FP32 state in place.
    #[cfg(feature = "deltanet")]
    pub fn gated_delta_net_f32_batch_seq(
        &mut self,
        q: &GpuTensor,
        k: &GpuTensor,
        v: &GpuTensor,
        gate: &GpuTensor,
        beta: &GpuTensor,
        state: &GpuTensor,
        output: &GpuTensor,
        n_tokens: usize,
        n_heads: usize,
        head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        Self::ensure_gdn_hd128(head_dim)?;
// Oracle swap, same as the routed sibling: identical signature and lane
        // mapping, `double` tile and arithmetic. Single-session decode takes
        // THIS kernel while batched decode takes the routed one, so both need
        // the swap or an oracle run silently measures nothing.
        static F64_ORACLE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let f64_oracle = *F64_ORACLE.get_or_init(|| {
            matches!(
                std::env::var("HIPFIRE_DN_STATE_F64_ORACLE").ok().as_deref(),
                Some("1") | Some("on") | Some("true") | Some("yes")
            )
        });
        let (cache_key, src, entry) = if f64_oracle {
            (
                "gated_delta_net_f64acc",
                kernels::GATED_DELTA_NET_F64ACC_SRC,
                "gated_delta_net_f64acc",
            )
        } else {
            (
                "gated_delta_net",
                kernels::GATED_DELTA_NET_SRC,
                "gated_delta_net_f32",
            )
        };
        self.ensure_kernel(cache_key, src, entry)?;
        let qp = q.buf.as_ptr();
        let kp = k.buf.as_ptr();
        let vp = v.buf.as_ptr();
        let gp = gate.buf.as_ptr();
        let bp = beta.buf.as_ptr();
        let sp = state.buf.as_ptr();
        let op = output.buf.as_ptr();
        let nt = n_tokens as i32;
        let nh = n_heads as i32;
        let hd = head_dim as i32;
        let bytes = crate::profile::gated_delta_net_f32_bytes(n_tokens, n_heads, head_dim);
        let timer = crate::profile::begin_timer(
            &self.hip,
            "deltanet",
            "gated_delta_net_f32_batch_seq",
            bytes,
        );
        let n_tiles = (128 / 4) as u32;
        let result = self.launch_kernargs(
            entry,
            [n_heads as u32, n_tiles, 1],
            [32, 1, 1],
            0,
            &kernargs![ptr qp, ptr kp, ptr vp, ptr gp, ptr bp, ptr sp, ptr op, i32 nt, i32 nh, i32 hd],
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
    #[cfg(feature = "deltanet")]
    pub fn gated_delta_net_f32_routed_batch_seq(
        &mut self,
        q: &GpuTensor,
        k: &GpuTensor,
        v: &GpuTensor,
        gate: &GpuTensor,
        beta: &GpuTensor,
        state_ptrs: &GpuTensor,
        row_session_indices: &GpuTensor,
        output: &GpuTensor,
        ptr_layer_stride: usize,
        delta_layer_index: usize,
        n_tokens: usize,
        n_heads: usize,
        head_dim: usize,
        n_sessions: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        Self::ensure_gdn_hd128(head_dim)?;
        // Oracle swap: identical signature, routing and lane mapping; `double` tile
        // and arithmetic. Off by default and slow by design — it exists to
        // measure how far this f32 path drifts, since every FP16-vs-FP32 state
        // figure is quoted against it.
        static F64_ORACLE_ROUTED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let f64_oracle = *F64_ORACLE_ROUTED.get_or_init(|| {
            matches!(
                std::env::var("HIPFIRE_DN_STATE_F64_ORACLE").ok().as_deref(),
                Some("1") | Some("on") | Some("true") | Some("yes")
            )
        });
        let kname = if f64_oracle {
            "gated_delta_net_f64acc_routed_batch_seq"
        } else {
            "gated_delta_net_f32_routed_batch_seq"
        };
        self.ensure_kernel(
            kname,
            if f64_oracle {
                kernels::GATED_DELTA_NET_F64ACC_ROUTED_BATCH_SEQ_SRC
            } else {
                kernels::GATED_DELTA_NET_F32_ROUTED_BATCH_SEQ_SRC
            },
            kname,
        )?;
        let qp = q.buf.as_ptr();
        let kp = k.buf.as_ptr();
        let vp = v.buf.as_ptr();
        let gp = gate.buf.as_ptr();
        let bp = beta.buf.as_ptr();
        let spp = state_ptrs.buf.as_ptr();
        let rsp = row_session_indices.buf.as_ptr();
        let op = output.buf.as_ptr();
        let ptr_stride = ptr_layer_stride as i32;
        let layer = delta_layer_index as i32;
        let nt = n_tokens as i32;
        let nh = n_heads as i32;
        let hd = head_dim as i32;
        let bytes = crate::profile::gated_delta_net_f32_bytes(n_tokens, n_heads, head_dim);
        let timer = crate::profile::begin_timer(
            &self.hip,
            "deltanet",
            "gated_delta_net_f32_routed_batch_seq",
            bytes,
        );
        let n_tiles = (128 / 4) as u32;
        let result = self.launch_kernargs(
            kname,
            [n_heads as u32, n_tiles, n_sessions as u32],
            [32, 1, 1],
            0,
            &kernargs![
                ptr qp, ptr kp, ptr vp, ptr gp, ptr bp, ptr spp, ptr rsp, ptr op,
                i32 ptr_stride, i32 layer, i32 nt, i32 nh, i32 hd
            ],
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// FP32 tree-aware DeltaNet replay — the tree path for FP32 state.
    ///
    /// DFS order, each token reading its parent's tape slot (or `s_init` at a
    /// root, `parent_indices[t] < 0`), persist-writing its post-update tile so
    /// children see the parent rather than the previous sibling. `s_init` is
    /// READ-ONLY; the caller replays the accepted spine linearly afterwards to
    /// advance the persistent state.
    ///
    /// Tape layout (caller responsibility):
    /// - `s_tape`:         `[n_tokens × n_heads × HD × HD]` f32 (scratch)
    /// - `parent_indices`: `[n_tokens]` i32 (`ddtree::linearize_tree`; spine is
    ///   `[-1, 0, 1, 2, ...]`)
    ///
    /// The tape is 4x the Q8 tape. [`Self::gated_delta_net_f16_tree_batch_seq`]
    /// halves it when the tree is wide enough for that to matter.
    #[cfg(feature = "deltanet")]
    #[allow(clippy::too_many_arguments)]
    pub fn gated_delta_net_f32_tree_batch_seq(
        &mut self,
        q_batch: &GpuTensor,
        k_batch: &GpuTensor,
        v_batch: &GpuTensor,
        gate_batch: &GpuTensor,
        beta_batch: &GpuTensor,
        s_init: &GpuTensor,
        s_tape: &GpuTensor,
        parent_indices: &GpuTensor,
        output_batch: &GpuTensor,
        n_tokens: usize,
        n_heads: usize,
        head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        Self::ensure_gdn_hd128(head_dim)?;
        self.ensure_kernel(
            "gated_delta_net_f32_tree",
            kernels::GATED_DELTA_NET_F32_TREE_SRC,
            "gated_delta_net_f32_tree",
        )?;

        let n_tiles = (128 / 4) as u32;

        let qp = q_batch.buf.as_ptr();
        let kp = k_batch.buf.as_ptr();
        let vp = v_batch.buf.as_ptr();
        let gp = gate_batch.buf.as_ptr();
        let bp = beta_batch.buf.as_ptr();
        let sip = s_init.buf.as_ptr();
        let stp = s_tape.buf.as_ptr();
        let pp = parent_indices.buf.as_ptr();
        let op = output_batch.buf.as_ptr();
        let nt = n_tokens as i32;
        let nh = n_heads as i32;
        let hd = head_dim as i32;

        let bytes = crate::profile::gated_delta_net_f32_bytes(n_tokens, n_heads, head_dim);
        let timer = crate::profile::begin_timer(
            &self.hip,
            "deltanet",
            "gated_delta_net_f32_tree_batch_seq",
            bytes,
        );
        let result = self.launch_kernargs(
            "gated_delta_net_f32_tree",
            [n_heads as u32, n_tiles, 1],
            [32, 1, 1],
            0,
            &kernargs![
                ptr qp, ptr kp, ptr vp, ptr gp, ptr bp, ptr sip, ptr stp,
                ptr pp, ptr op, i32 nt, i32 nh, i32 hd
            ],
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// FP16 tree-aware DeltaNet replay — the tree half of the f16 state path.
    ///
    /// Identical to [`Self::gated_delta_net_f32_tree_batch_seq`] except BOTH
    /// `s_init` and the tape are `_Float16`, matching
    /// [`Self::gated_delta_net_f16_batch_seq`]: under FP16 state the persistent
    /// S matrix is itself f16. Storage f16, arithmetic FP32.
    ///
    /// Layout: `s_init` is `[n_heads × HD × HD]` f16 and `s_tape` is
    /// `[n_tokens × n_heads × HD × HD]` f16 — HALF the bytes of the f32 entry
    /// point in both cases, which the caller must size for.
    ///
    /// Costs one f16 rounding per tape round-trip, accumulating with tree DEPTH.
    /// Not byte-exact against the linear FP32 kernel; against the f16 LINEAR
    /// kernel on a spine it is exact.
    #[cfg(feature = "deltanet")]
    #[allow(clippy::too_many_arguments)]
    pub fn gated_delta_net_f16_tree_batch_seq(
        &mut self,
        q_batch: &GpuTensor,
        k_batch: &GpuTensor,
        v_batch: &GpuTensor,
        gate_batch: &GpuTensor,
        beta_batch: &GpuTensor,
        s_init: &GpuTensor,
        s_tape: &GpuTensor,
        parent_indices: &GpuTensor,
        output_batch: &GpuTensor,
        n_tokens: usize,
        n_heads: usize,
        head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        Self::ensure_gdn_hd128(head_dim)?;
        self.ensure_kernel(
            "gated_delta_net_f16_tree",
            kernels::GATED_DELTA_NET_F16_TREE_SRC,
            "gated_delta_net_f16_tree",
        )?;

        let n_tiles = (128 / 4) as u32;

        let qp = q_batch.buf.as_ptr();
        let kp = k_batch.buf.as_ptr();
        let vp = v_batch.buf.as_ptr();
        let gp = gate_batch.buf.as_ptr();
        let bp = beta_batch.buf.as_ptr();
        let sip = s_init.buf.as_ptr();
        let stp = s_tape.buf.as_ptr();
        let pp = parent_indices.buf.as_ptr();
        let op = output_batch.buf.as_ptr();
        let nt = n_tokens as i32;
        let nh = n_heads as i32;
        let hd = head_dim as i32;

        let bytes = crate::profile::gated_delta_net_f32_bytes(n_tokens, n_heads, head_dim);
        let timer = crate::profile::begin_timer(
            &self.hip,
            "deltanet",
            "gated_delta_net_f16_tree_batch_seq",
            bytes,
        );
        let result = self.launch_kernargs(
            "gated_delta_net_f16_tree",
            [n_heads as u32, n_tiles, 1],
            [32, 1, 1],
            0,
            &kernargs![
                ptr qp, ptr kp, ptr vp, ptr gp, ptr bp, ptr sip, ptr stp,
                ptr pp, ptr op, i32 nt, i32 nh, i32 hd
            ],
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// Routed FP16-state DeltaNet for independent sessions — the f16
    /// counterpart of [`Self::gated_delta_net_f32_routed_batch_seq`], with the
    /// same argument order.
    ///
    /// This is where f16 earns its keep. State is per-SEQUENCE, so its footprint
    /// scales with concurrency, and this is the cross-session batched path: one
    /// state per session behind `state_ptrs`, so halving each halves the whole
    /// fleet's resident state.
    ///
    /// The pointers in `state_ptrs` must address **f16** buffers — half the f32
    /// stride. Passing f32 state here reads every element at the wrong offset,
    /// which is why the state's precision and this kernel are chosen together.
    #[cfg(feature = "deltanet")]
    #[allow(clippy::too_many_arguments)]
    pub fn gated_delta_net_f16_routed_batch_seq(
        &mut self,
        q: &GpuTensor,
        k: &GpuTensor,
        v: &GpuTensor,
        gate: &GpuTensor,
        beta: &GpuTensor,
        state_ptrs: &GpuTensor,
        row_session_indices: &GpuTensor,
        output: &GpuTensor,
        ptr_layer_stride: usize,
        delta_layer_index: usize,
        n_tokens: usize,
        n_heads: usize,
        head_dim: usize,
        n_sessions: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        Self::ensure_gdn_hd128(head_dim)?;
        self.ensure_kernel(
            "gated_delta_net_f16_routed_batch_seq",
            kernels::GATED_DELTA_NET_F16_ROUTED_BATCH_SEQ_SRC,
            "gated_delta_net_f16_routed_batch_seq",
        )?;
        let qp = q.buf.as_ptr();
        let kp = k.buf.as_ptr();
        let vp = v.buf.as_ptr();
        let gp = gate.buf.as_ptr();
        let bp = beta.buf.as_ptr();
        let spp = state_ptrs.buf.as_ptr();
        let rsp = row_session_indices.buf.as_ptr();
        let op = output.buf.as_ptr();
        let ptr_stride = ptr_layer_stride as i32;
        let layer = delta_layer_index as i32;
        let nt = n_tokens as i32;
        let nh = n_heads as i32;
        let hd = head_dim as i32;
        // Same env as the single-session f16 path; read here so no caller changes.
        static SR_ROUTED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let sr = *SR_ROUTED.get_or_init(|| {
            matches!(
                std::env::var("HIPFIRE_DN_STATE_FP16_DITHER").ok().as_deref(),
                Some("1") | Some("on") | Some("true") | Some("yes")
            )
        }) as i32;
        let bytes = crate::profile::gated_delta_net_f32_bytes(n_tokens, n_heads, head_dim);
        let timer = crate::profile::begin_timer(
            &self.hip,
            "deltanet",
            "gated_delta_net_f16_routed_batch_seq",
            bytes,
        );
        let n_tiles = (128 / 4) as u32;
        let result = self.launch_kernargs(
            "gated_delta_net_f16_routed_batch_seq",
            [n_heads as u32, n_tiles, n_sessions as u32],
            [32, 1, 1],
            0,
            &kernargs![
                ptr qp, ptr kp, ptr vp, ptr gp, ptr bp, ptr spp, ptr rsp, ptr op,
                i32 ptr_stride, i32 layer, i32 nt, i32 nh, i32 hd, i32 sr
            ],
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// Linear DeltaNet recurrence with the persistent S state stored as f16 —
    /// the sanctioned half-precision state path.
    ///
    /// Drop-in for [`Self::gated_delta_net_f32_batch_seq`] with one difference
    /// the caller must honour: `state` is `[n_heads × HD × HD]` **f16**, half
    /// the bytes. Everything else — argument order, output layout, arithmetic
    /// precision — is identical, because only the storage format changes.
    ///
    /// Unlike the Q8 state path this replaces, there is no scales tensor and no
    /// `s_ef_residual` accumulator, so it is deterministic and a spec-decode
    /// snapshot restores exactly what it saved.
    ///
    /// `HIPFIRE_DN_STATE_FP16_DITHER=1` (default off) narrows with a dither
    /// instead of round-to-nearest, to break the bias that compounds across
    /// decode steps — the state is re-narrowed once per call and a decode call
    /// carries one token. The dither hashes the value's own bits with the
    /// element index, NOT a counter or a carried RNG, so the store stays a pure
    /// function of its input and the snapshot property above still holds. That
    /// is the difference from the Q8 path's stochastic rounding, which was
    /// removed precisely because it broke it.
    #[cfg(feature = "deltanet")]
    #[allow(clippy::too_many_arguments)]
    pub fn gated_delta_net_f16_batch_seq(
        &mut self,
        q_batch: &GpuTensor,
        k_batch: &GpuTensor,
        v_batch: &GpuTensor,
        gate_batch: &GpuTensor,
        beta_batch: &GpuTensor,
        state: &GpuTensor,
        output_batch: &GpuTensor,
        n_tokens: usize,
        n_heads: usize,
        head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        Self::ensure_gdn_hd128(head_dim)?;
        self.ensure_kernel(
            "gated_delta_net_f16",
            kernels::GATED_DELTA_NET_F16_SRC,
            "gated_delta_net_f16",
        )?;

        let n_tiles = (128 / 4) as u32;

        let qp = q_batch.buf.as_ptr();
        let kp = k_batch.buf.as_ptr();
        let vp = v_batch.buf.as_ptr();
        let gp = gate_batch.buf.as_ptr();
        let bp = beta_batch.buf.as_ptr();
        let sp = state.buf.as_ptr();
        let op = output_batch.buf.as_ptr();
        let nt = n_tokens as i32;
        let nh = n_heads as i32;
        let hd = head_dim as i32;
        // Read here rather than threading a flag through four call sites in
        // three modules. Cached: this is on the per-layer decode path.
        static SR: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let sr = *SR.get_or_init(|| {
            matches!(
                std::env::var("HIPFIRE_DN_STATE_FP16_DITHER").ok().as_deref(),
                Some("1") | Some("on") | Some("true") | Some("yes")
            )
        }) as i32;

        let bytes = crate::profile::gated_delta_net_f32_bytes(n_tokens, n_heads, head_dim);
        let timer = crate::profile::begin_timer(
            &self.hip,
            "deltanet",
            "gated_delta_net_f16_batch_seq",
            bytes,
        );
        let result = self.launch_kernargs(
            "gated_delta_net_f16",
            [n_heads as u32, n_tiles, 1],
            [32, 1, 1],
            0,
            &kernargs![
                ptr qp, ptr kp, ptr vp, ptr gp, ptr bp, ptr sp, ptr op,
                i32 nt, i32 nh, i32 hd, i32 sr
            ],
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// Gated linear-recurrence scan forward (fp32). `g`,`u`,`h_out`: `[seq*D]`
    /// row-major (time-major: index `t*D+c`). `h[t]=g[t]*h[t-1]+(1-g[t])*u[t]`,
    /// `h[-1]=0`. One thread per channel `c`, sequential over time; no shared mem.
    pub fn gated_scan_fwd(
        &mut self,
        g: &GpuTensor,
        u: &GpuTensor,
        h_out: &GpuTensor,
        seq: usize,
        d: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gated_scan_fwd",
            kernels::GATED_SCAN_TRAIN_SRC,
            "gated_scan_fwd",
        )?;
        let func = &self.functions["gated_scan_fwd"];
        let mut gp = g.buf.as_ptr();
        let mut up = u.buf.as_ptr();
        let mut hp = h_out.buf.as_ptr();
        let mut s = seq as i32;
        let mut dd = d as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut gp as *mut _ as *mut c_void,
            &mut up as *mut _ as *mut c_void,
            &mut hp as *mut _ as *mut c_void,
            &mut s as *mut _ as *mut c_void,
            &mut dd as *mut _ as *mut c_void,
        ];
        let blocks = (d as u32).div_ceil(256);
        unsafe {
            self.hip.launch_kernel(
                func,
                [blocks, 1, 1],
                [256, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
    /// Gated linear-recurrence scan backward (fp32). Given `d_hout`=dL/dh[t] for
    /// every t, produces `d_g`,`d_u` (`[seq*D]`). Reverse scan, one thread per
    /// channel; `h_out` is the forward output (needed for `h[t-1]`). No shared mem.
    #[allow(clippy::too_many_arguments)]
    pub fn gated_scan_bwd(
        &mut self,
        g: &GpuTensor,
        u: &GpuTensor,
        h_out: &GpuTensor,
        d_hout: &GpuTensor,
        d_g: &GpuTensor,
        d_u: &GpuTensor,
        seq: usize,
        d: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "gated_scan_bwd",
            kernels::GATED_SCAN_TRAIN_SRC,
            "gated_scan_bwd",
        )?;
        let func = &self.functions["gated_scan_bwd"];
        let mut gp = g.buf.as_ptr();
        let mut up = u.buf.as_ptr();
        let mut hp = h_out.buf.as_ptr();
        let mut dhp = d_hout.buf.as_ptr();
        let mut dgp = d_g.buf.as_ptr();
        let mut dup = d_u.buf.as_ptr();
        let mut s = seq as i32;
        let mut dd = d as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut gp as *mut _ as *mut c_void,
            &mut up as *mut _ as *mut c_void,
            &mut hp as *mut _ as *mut c_void,
            &mut dhp as *mut _ as *mut c_void,
            &mut dgp as *mut _ as *mut c_void,
            &mut dup as *mut _ as *mut c_void,
            &mut s as *mut _ as *mut c_void,
            &mut dd as *mut _ as *mut c_void,
        ];
        let blocks = (d as u32).div_ceil(256);
        unsafe {
            self.hip.launch_kernel(
                func,
                [blocks, 1, 1],
                [256, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
}
