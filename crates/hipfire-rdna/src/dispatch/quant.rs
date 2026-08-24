// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Reference-layer activation quant kernels (W4A4 oq4, W4A8 oq8, per-token
//! int8). Split out of `dispatch/mod.rs` (Phase 1 of
//! docs/plans/2026-06-23-dispatch-refactor.md) — a pure behavior-preserving
//! move: the methods stay on `Gpu` via this child `impl` block and reach
//! `Gpu`'s module-private fields as a descendant of `dispatch`.

use super::{DType, Gpu, GpuTensor};
use crate::kernels;
use hip_bridge::HipResult;
use std::ffi::c_void;

/// Activation group size for the batched W4A4 prefill path (`HIPFIRE_OQ4_ACT_GROUP`,
/// default 256 = the Oq4G256 weight codec's group, i.e. production unchanged).
///
/// A finer activation group means a tighter absmax per scale, so fewer values in
/// each group are crushed by an outlier — the cheapest quality knob on the
/// activation side that needs no new math. Legal sizes are constrained from three
/// directions: the LDS GEMM flushes on a BK=64 K-strip boundary, the wave32
/// quantizer needs `group/32` even, and it must divide the weight group. That
/// leaves 64 / 128 / 256 on wave32; a wave64 quantizer (`group/64` even) would
/// start at 128, so **128 and 256 are the only sizes portable across both wave
/// widths**. Anything below 256 goes through `gemm_oq4_grouped_wmma_lds_gx`.
pub fn oq4_act_group() -> usize {
    match std::env::var("HIPFIRE_OQ4_ACT_GROUP") {
        Ok(v) => {
            let g: usize = v.parse().expect("HIPFIRE_OQ4_ACT_GROUP must be an integer");
            assert!(
                g == 64 || g == 128 || g == 256,
                "HIPFIRE_OQ4_ACT_GROUP must be 64, 128 or 256"
            );
            g
        }
        Err(_) => 256,
    }
}

impl Gpu {
    /// Ensure reusable plain-basis DFLASH activation staging for `n` rows.
    /// Capacity only grows; sequential projections safely reuse it on the same
    /// stream. `k` is G256-aligned for the staged W4A8/W8A8 path.
    fn ensure_dflash_oq_scratch(&mut self, n: usize, k: usize) -> HipResult<()> {
        let need_xq = n * k;
        let need_xs = n * (k / 256);
        if self
            .dflash_oq_xq_batch
            .as_ref()
            .map(|t| t.numel() < need_xq)
            .unwrap_or(true)
        {
            self.dflash_oq_xq_batch = Some(self.alloc_tensor(&[need_xq], DType::Raw)?);
        }
        if self
            .dflash_oq_xs_batch
            .as_ref()
            .map(|t| t.numel() < need_xs)
            .unwrap_or(true)
        {
            self.dflash_oq_xs_batch = Some(self.alloc_tensor(&[need_xs], DType::F32)?);
        }
        Ok(())
    }

    /// Packed plain-basis DFLASH Opus production dispatch. G256-aligned rows use
    /// the fused A8+dot4 path; ragged rows retain the exact scalar-F16 fallback
    /// because checkpoint blocks can cross logical row boundaries there.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_dflash_oq_plain(
        &mut self,
        dtype: DType,
        w_blocks: &GpuTensor,
        x_f32: &GpuTensor,
        y_f32: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
        block_stride: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        const QUANT_CHUNK_ROWS: usize = 64;
        let aligned = k % 256 == 0;
        let (kernel_name, expected_stride) = dflash_plain_kernel(dtype, aligned);
        validate_dflash_plain_stride(kernel_name, block_stride, expected_stride);
        if aligned {
            self.ensure_dflash_oq_scratch(batch_size.min(QUANT_CHUNK_ROWS), k)?;
            self.ensure_kernel(
                "quantize_dflash_act_g256",
                kernels::GEMM_DFLASH_OQ_PLAIN_REF_SRC,
                "quantize_dflash_act_g256",
            )?;
        }
        self.ensure_kernel(
            kernel_name,
            kernels::GEMM_DFLASH_OQ_PLAIN_REF_SRC,
            kernel_name,
        )?;
        if !aligned {
            let wp = w_blocks.buf.as_ptr();
            let xp = x_f32.buf.as_ptr();
            let yp = y_f32.buf.as_ptr();
            let mut mi = m as i32;
            let mut ki = k as i32;
            let mut bi = batch_size as i32;
            let mut stride = block_stride as i32;
            let mut params: Vec<*mut c_void> = vec![
                &wp as *const _ as *mut c_void,
                &xp as *const _ as *mut c_void,
                &yp as *const _ as *mut c_void,
                &mut mi as *mut _ as *mut c_void,
                &mut ki as *mut _ as *mut c_void,
                &mut bi as *mut _ as *mut c_void,
                &mut stride as *mut _ as *mut c_void,
            ];
            let func = &self.functions[kernel_name];
            return unsafe {
                self.hip.launch_kernel(
                    func,
                    [m.div_ceil(8) as u32, batch_size as u32, 1],
                    [256, 1, 1],
                    0,
                    self.stream_ref(),
                    &mut params,
                )
            };
        }

        let groups = k / 256;
        let mut row = 0;
        while row < batch_size {
            let n = (batch_size - row).min(QUANT_CHUNK_ROWS);
            let x_chunk = x_f32.sub_offset(row * k, n * k);
            let y_chunk = y_f32.sub_offset(row * m, n * m);
            let xq = GpuTensor {
                // SAFETY: bounded view of persistent scratch, used only by
                // stream-ordered launches before the next projection reuses it.
                buf: unsafe { self.dflash_oq_xq_batch.as_ref().unwrap().buf.alias() },
                shape: vec![n * k],
                dtype: DType::Raw,
            };
            let xs = GpuTensor {
                // SAFETY: same lifetime/order contract as `xq`.
                buf: unsafe { self.dflash_oq_xs_batch.as_ref().unwrap().buf.alias() },
                shape: vec![n * groups],
                dtype: DType::F32,
            };

            let xp = x_chunk.buf.as_ptr();
            let xqp = xq.buf.as_ptr();
            let xsp = xs.buf.as_ptr();
            let mut ki = k as i32;
            let mut ni = n as i32;
            let mut quant_params: Vec<*mut c_void> = vec![
                &xp as *const _ as *mut c_void,
                &xqp as *const _ as *mut c_void,
                &xsp as *const _ as *mut c_void,
                &mut ki as *mut _ as *mut c_void,
                &mut ni as *mut _ as *mut c_void,
            ];
            let quant_func = &self.functions["quantize_dflash_act_g256"];
            unsafe {
                self.hip.launch_kernel(
                    quant_func,
                    [groups as u32, n as u32, 1],
                    [256, 1, 1],
                    0,
                    self.stream_ref(),
                    &mut quant_params,
                )?;
            }

            let wp = w_blocks.buf.as_ptr();
            let yp = y_chunk.buf.as_ptr();
            let mut mi = m as i32;
            let mut stride = block_stride as i32;
            let mut gemm_params: Vec<*mut c_void> = vec![
                &wp as *const _ as *mut c_void,
                &xqp as *const _ as *mut c_void,
                &xsp as *const _ as *mut c_void,
                &yp as *const _ as *mut c_void,
                &mut mi as *mut _ as *mut c_void,
                &mut ki as *mut _ as *mut c_void,
                &mut ni as *mut _ as *mut c_void,
                &mut stride as *mut _ as *mut c_void,
            ];
            // Multi-column when an exact-NB instantiation exists: one grid
            // column instead of `n`, because the kernel loops the columns
            // internally and reads each weight row once for all of them.
            let (gemm_name, grid_y) = match dflash_plain_multicol_kernel(dtype, n) {
                Some(name) => (name, 1u32),
                None => (kernel_name, n as u32),
            };
            if !std::ptr::eq(gemm_name, kernel_name) {
                self.ensure_kernel(
                    gemm_name,
                    kernels::GEMM_DFLASH_OQ_PLAIN_REF_SRC,
                    gemm_name,
                )?;
            }
            let gemm_func = &self.functions[gemm_name];
            unsafe {
                self.hip.launch_kernel(
                    gemm_func,
                    [m.div_ceil(8) as u32, grid_y, 1],
                    [256, 1, 1],
                    0,
                    self.stream_ref(),
                    &mut gemm_params,
                )?;
            }
            row += n;
        }
        Ok(())
    }

    /// Native packed reference GEMM for DFLASH qt=45/46/47 plain-basis Opus
    /// blocks. This is intentionally separate from the primary-model
    /// `Oq{4,8}G256` paths: those consume split f32 scales after FWHT, while
    /// DFLASH preserves interleaved f16-scale NPU blocks and unrotated X.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_dflash_oq_plain_ref(
        &mut self,
        dtype: DType,
        w_blocks: &GpuTensor,
        x_f32: &GpuTensor,
        y_f32: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
        block_stride: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let (kernel_name, expected_stride) = match dtype {
            DType::DflashOq8Plain => ("gemm_dflash_oq8_plain_ref", Some(258usize)),
            DType::DflashOq4Plain => ("gemm_dflash_oq4_plain_ref", Some(130usize)),
            DType::DflashOq4MixedPlain => ("gemm_dflash_oq4_mixed_plain_ref", None),
            other => panic!("gemm_dflash_oq_plain_ref: unsupported dtype {other:?}"),
        };
        validate_dflash_plain_stride(kernel_name, block_stride, expected_stride);
        self.ensure_kernel(
            kernel_name,
            kernels::GEMM_DFLASH_OQ_PLAIN_REF_SRC,
            kernel_name,
        )?;
        let wp = w_blocks.buf.as_ptr();
        let xp = x_f32.buf.as_ptr();
        let yp = y_f32.buf.as_ptr();
        let mut mi = m as i32;
        let mut ki = k as i32;
        let mut bi = batch_size as i32;
        let mut stride = block_stride as i32;
        let mut params: Vec<*mut c_void> = vec![
            &wp as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &yp as *const _ as *mut c_void,
            &mut mi as *mut _ as *mut c_void,
            &mut ki as *mut _ as *mut c_void,
            &mut bi as *mut _ as *mut c_void,
            &mut stride as *mut _ as *mut c_void,
        ];
        let func = &self.functions[kernel_name];
        unsafe {
            self.hip.launch_kernel(
                func,
                [m as u32, batch_size as u32, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// Opus Quant W4A4: dynamic per-token/group INT4 activation quantizer.
    /// `x_f32` [B,K] → `xq_i4` [B,K/2] (packed signed int4) + `xs` [B,K/group]
    /// (f32 scales). gfx1103 wave32, zero LDS. `group % 32 == 0`, `k % group == 0`.
    pub fn quantize_act_oq4(
        &mut self,
        x_f32: &GpuTensor,
        xq_i4: &GpuTensor,
        xs: &GpuTensor,
        batch_size: usize,
        k: usize,
        group: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        // Multiple of 64, not 32: both quantizer kernels give each of the 32 lanes
        // a contiguous run of `group/32` elements and nibble-pack it in PAIRS, so
        // an odd run (group = 32, 96, …) makes a lane read its neighbour's first
        // element and drop its own last one — silent corruption, not a crash.
        // (A wave64 port would need group % 128 == 0 for the same reason.)
        assert_eq!(
            group % 64,
            0,
            "quantize_act_oq4: group must be a multiple of 64 (group/32 must be even to nibble-pack)"
        );
        assert_eq!(
            k % group,
            0,
            "quantize_act_oq4: K must be a multiple of group"
        );
        // Stream B lever: HIPFIRE_OQ4_ACT_CLIP=1 swaps in the per-group clip-search
        // activation quantizer (MSE-optimal clip vs plain absmax). Same output
        // format, so callers/GEMM are unchanged; default = plain absmax.
        let (entry, src): (&str, &str) =
            if std::env::var("HIPFIRE_OQ4_ACT_CLIP").as_deref() == Ok("1") {
                ("quantize_act_oq4_clip", kernels::QUANTIZE_ACT_OQ4_CLIP_SRC)
            } else {
                ("quantize_act_oq4", kernels::QUANTIZE_ACT_OQ4_SRC)
            };
        self.ensure_kernel(entry, src, entry)?;
        let xp = x_f32.buf.as_ptr();
        let xqp = xq_i4.buf.as_ptr();
        let xsp = xs.buf.as_ptr();
        let mut bi = batch_size as i32;
        let mut ki = k as i32;
        let mut gi = group as i32;
        let mut params: Vec<*mut c_void> = vec![
            &xp as *const _ as *mut c_void,
            &xqp as *const _ as *mut c_void,
            &xsp as *const _ as *mut c_void,
            &mut bi as *mut _ as *mut c_void,
            &mut ki as *mut _ as *mut c_void,
            &mut gi as *mut _ as *mut c_void,
        ];
        let grid_g = (k / group) as u32;
        let grid_b = batch_size as u32;
        let func = &self.functions[entry];
        unsafe {
            self.hip.launch_kernel(
                func,
                [grid_g, grid_b, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// Opus Quant W8A8 grouped int8×int8 GEMM (the int8 generalization of
    /// [`Self::gemm_oq4_grouped_wmma`]). `w_i8`/`x_i8` are [M,K]/[B,K] signed int8
    /// rows; `w_scales`/`x_scales` are per-group f32; `y_f32` is [B,M].
    #[allow(clippy::too_many_arguments)]
    /// Default rows-per-launch target for the M-slab split.
    ///
    /// Measured on gfx1151 against COLD weights (`bench_oq8_gemm_small_n --cold`),
    /// gate/up [17408, 5120] at B=9:
    ///
    /// ```text
    ///   rows/launch   1024   2176   4352   5120   8704   17408(one launch)
    ///   GB/s           101    133    132    136    108    18
    /// ```
    ///
    /// A broad plateau from ~2K to ~5K rows, and a 7.5x cliff if the whole M goes
    /// in one launch. Cold matters: this part has a 32 MB MALL, so a warm loop
    /// over a sub-32 MiB shape reports cache bandwidth (o_proj measured 288 GB/s
    /// warm against 140 cold — above the ~256 GB/s LPDDR5X peak, which is the
    /// tell). An earlier version of this constant was justified by that warm
    /// number; the value survived re-tuning, the reasoning did not.
    const SLAB_TARGET_DEFAULT: usize = 5120;

    pub fn gemm_oq8_grouped_wmma(
        &mut self,
        w_i8: &GpuTensor,
        w_scales: &GpuTensor,
        x_i8: &GpuTensor,
        x_scales: &GpuTensor,
        y_f32: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
        group: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert_eq!(
            k % group,
            0,
            "gemm_oq8_grouped_wmma: K must be a multiple of group"
        );
        assert_eq!(
            group % 16,
            0,
            "gemm_oq8_grouped_wmma: group must be a multiple of 16"
        );
        // Multi-wave: several 16-row WMMA tiles per block, so there are that many
        // independent WMMA chains in flight to hide the scattered weight read's
        // latency. Same reads, same math, composes with the row-slab below.
        //
        // ponytail: 8 waves, from a cold sweep on gfx1151 (gate/up [17408, 5120],
        // B=9): 141.0 at 1 wave, 148.8 / 150.3 / **158.4** / 154.5 at 2/4/8/16.
        // The other two production shapes are flat within noise, so this is a
        // free +12% on the tall-thin one. `HIPFIRE_OQ8_GEMM_MW` overrides;
        // 0 selects the original one-wave kernel.
        static MW: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
        let mw_waves = *MW.get_or_init(|| {
            std::env::var("HIPFIRE_OQ8_GEMM_MW")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .filter(|w| *w <= 16)
                .unwrap_or(8)
        });
        let entry = if mw_waves > 0 {
            "gemm_oq8_grouped_wmma_mw"
        } else {
            "gemm_oq8_grouped_wmma"
        };
        let (block_threads, rows_per_block) = if mw_waves > 0 {
            ((mw_waves * 32) as u32, mw_waves * 16)
        } else {
            (32u32, 16usize)
        };

        self.ensure_kernel(
            "gemm_oq8_grouped_wmma",
            kernels::GEMM_OQ8_GROUPED_WMMA_SRC,
            entry,
        )?;
        let wp = w_i8.buf.as_ptr();
        let wsp = w_scales.buf.as_ptr();
        let xp = x_i8.buf.as_ptr();
        let xsp = x_scales.buf.as_ptr();
        let yp = y_f32.buf.as_ptr();
        // Row-slab the M dimension. One launch covering all of M collapses on
        // tall-thin shapes — 18 GB/s at [17408, 5120] B=9 on cold weights, where
        // the same kernel does 169 GB/s at [5120, 17408] with the identical byte
        // count and per-block work. Slabbing recovers it to 135 GB/s, a 7.3x
        // speedup, with no change to the math: W/Ws are advanced per slab and the
        // kernel writes Y at `m_base + out_row`, so every output word is computed
        // by the same block arithmetic as the single launch (checked exactly by
        // `parity_oq8_gemm`'s row-placement case). See `bench_oq8_gemm_small_n`.
        //
        // Split EVENLY rather than into fixed slabs plus a runt: a trailing
        // 1024-row launch costs more than it computes, and shapes already at full
        // bandwidth (M <= the target) must keep taking a single launch — slabbing
        // those measured ~20 % SLOWER.
        //
        // ponytail: tuned on gfx1151 against COLD weights — see the note in
        // `bench_oq8_gemm_small_n` about the 32 MB MALL, which makes any shape
        // under ~32 MiB report cache bandwidth when timed in a loop. Override to
        // re-tune on another arch.
        let slab_target: usize = std::env::var("HIPFIRE_OQ8_GEMM_SLAB_ROWS")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&v: &usize| v >= 16)
            .unwrap_or(Self::SLAB_TARGET_DEFAULT);
        // Escape hatch, and what the slab/no-slab equivalence check drives.
        // Slabbing changes only which launch computes a row, never the row's
        // arithmetic, so `HIPFIRE_OQ8_GEMM_SLAB=0` must be BIT-identical.
        static SLAB_ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let slab_on = *SLAB_ON.get_or_init(|| {
            !matches!(
                std::env::var("HIPFIRE_OQ8_GEMM_SLAB").ok().as_deref(),
                Some("0" | "off" | "false" | "no")
            )
        });
        let n_slabs = if slab_on {
            m.div_ceil(slab_target).max(1)
        } else {
            1
        };
        let slab_rows = m.div_ceil(n_slabs).next_multiple_of(16);
        let ky = k as i32;
        let mut ki = ky;
        let mut bi = batch_size as i32;
        let mut gi = group as i32;
        let mut ma = m as i32;
        let grid_b = batch_size.div_ceil(16) as u32;
        let n_groups = k / group;
        let mut base = 0usize;
        while base < m {
            let rows = slab_rows.min(m - base);
            let wp_s = unsafe { (wp as *const u8).add(base * k) } as *mut c_void;
            let wsp_s = unsafe { (wsp as *const f32).add(base * n_groups) } as *mut c_void;
            let mut mi = rows as i32;
            let mut mb = base as i32;
            let mut params: Vec<*mut c_void> = vec![
                &wp_s as *const _ as *mut c_void,
                &wsp_s as *const _ as *mut c_void,
                &xp as *const _ as *mut c_void,
                &xsp as *const _ as *mut c_void,
                &yp as *const _ as *mut c_void,
                &mut mi as *mut _ as *mut c_void,
                &mut ki as *mut _ as *mut c_void,
                &mut bi as *mut _ as *mut c_void,
                &mut gi as *mut _ as *mut c_void,
                &mut ma as *mut _ as *mut c_void,
                &mut mb as *mut _ as *mut c_void,
            ];
            let grid_m = rows.div_ceil(rows_per_block) as u32;
            let func = &self.functions[entry];
            unsafe {
                self.hip.launch_kernel(
                    func,
                    [grid_m, grid_b, 1],
                    [block_threads, 1, 1],
                    0,
                    self.stream_ref(),
                    &mut params,
                )?
            };
            base += rows;
        }
        Ok(())
    }

    /// Compact-resident twin of [`Self::gemm_oq8_grouped_residual_act_batched`]:
    /// `residual[N x M] += W_compact . x_rot`. GEMMs into the persistent batched
    /// f32 scratch then one elementwise add — there is no fused compact residual
    /// kernel, mirroring the oq8 and oq4 arms.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_oq_compact_residual_act_batched(
        &mut self,
        w_blocks: &GpuTensor,
        x_rot: &GpuTensor,
        residual: &GpuTensor,
        m: usize,
        k: usize,
        n: usize,
        block_stride: usize,
    ) -> HipResult<()> {
        self.ensure_oq8_scratch_batched(n, k, m)?;
        let tmp = GpuTensor {
            buf: unsafe { self.oq8_ytmp_batch.as_ref().unwrap().buf.alias() },
            shape: vec![n * m],
            dtype: DType::F32,
        };
        self.gemm_oq_compact_act_batched(w_blocks, x_rot, &tmp, m, k, n, block_stride)?;
        let res_n = residual.sub_offset(0, n * m);
        self.add_inplace_f32(&res_n, &tmp)
    }

    /// Compact-resident twin of [`Self::gemm_oq8_grouped_act_batched`]: quantize
    /// the rotated activation into the shared oq8 scratch, then run the compact
    /// GEMM against it. The scratch is Gpu-owned, so callers outside this module
    /// use this rather than assembling `x_i8`/`x_scales` themselves.
    #[allow(clippy::too_many_arguments)]
    /// Route one compact batched GEMM against an activation already quantized
    /// into the oq8 scratch.
    ///
    /// Exact W4A8 via two iu4 WMMA passes plus the overlay as a K-major
    /// accumulate is the DEFAULT: it is the same arithmetic as the iu8 core --
    /// iu8 spends half its weight lanes on sign-extended 4-bit values the
    /// compact format never had -- and ~1.4x faster on the 27B projection
    /// shapes. `HIPFIRE_OQ_COMPACT_IU4X2=0` falls back to the iu8 core.
    ///
    /// Both `gemm_oq_compact_act_batched` and `gemm_oq_compact_grouped_prequant`
    /// come through here, so the route is decided in exactly one place. That
    /// matters: routing only the first left the prequant path (which serves
    /// gate+up and q/k/v off a shared quantize) on iu8, and a kernel trace put
    /// it at 54.8% of prefill kernel time against iu4x2's 15.1%.
    #[allow(clippy::too_many_arguments)]
    fn compact_batched_route(
        &mut self,
        w_blocks: &GpuTensor,
        xq: &GpuTensor,
        xs: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        n: usize,
        ng: usize,
        group: usize,
        block_stride: usize,
    ) -> HipResult<()> {
        // SMALL BATCH -> multi-column GEMV, the same routing
        // `gemm_oq_compact_grouped_wmma` already makes and for the same reason.
        // The tiled GEMMs below are N-heavy by design (the wave64 recipe is
        // tuned at B=128..512); at spec-decode's verify width they read every
        // weight byte to produce a handful of output columns. Measured on
        // Qwen3.8-27B/DFlash2: verify was 310ms at B=2 and 322ms at B=8 --
        // CONSTANT in B, ~4.5 weight sweeps where one batched pass should cost
        // one. `gemv_oq_compact_multicol` reads each weight row once and
        // accumulates B columns.
        //
        // `xq`/`xs` are already the int8 activation and its scales, which is
        // exactly multicol's contract -- the int4 path below derives its own
        // nibbles from them, so nothing upstream has to change.
        //
        // HIPFIRE_OQ_COMPACT_SMALL_N=0 restores the GEMM for A/B.
        if n <= 16
            && group == 256
            && std::env::var("HIPFIRE_OQ_COMPACT_SMALL_N").as_deref() != Ok("0")
        {
            return self.gemv_oq_compact_multicol(w_blocks, xq, xs, y, m, k, n, block_stride);
        }
        if std::env::var("HIPFIRE_OQ_COMPACT_IU4X2").as_deref() != Ok("0") {
            // Wave64 twin, where it actually wins. Benched at the 27B shapes vs
            // the wave32 two-pass: gate/up 1.26x, B=512 1.26x, B=128 1.47x,
            // down 1.05x, qkv 1.05x -- but wo (K=4096) 0.75x. The wave64 recipe
            // is N-heavy, and the second i32 accumulator set forces WNt=4 rather
            // than the 1-pass twin's 8, which is why this lands at 1.26x and not
            // the 1.56x the 1-pass sees. Route on K: the small-K shape loses.
            let use_w64 = group == 256
                && k >= 5120
                && std::env::var("HIPFIRE_OQ_COMPACT_W64").as_deref() != Ok("0");
            let xt = GpuTensor {
                buf: unsafe { self.oq_xt_batch.as_ref().unwrap().buf.alias() },
                shape: vec![n * k],
                dtype: DType::Raw,
            };
            let xst = GpuTensor {
                buf: unsafe { self.oq_xst_batch.as_ref().unwrap().buf.alias() },
                shape: vec![n * ng],
                dtype: DType::F32,
            };
            if oq_compact_a4() && group == 256 && oq4_act_group() == 256 {
                // W4A4. Weights are the SAME compact 4.25-bit blocks -- only the
                // activation narrows -- so the bits/weight floor is untouched.
                // The bulk nibble under each overlay entry is zeroed by the
                // loader, so those positions contribute 0 here and the sparse
                // overlay below corrects them, exactly as in the W4A8 twin.
                let x4 = GpuTensor {
                    buf: unsafe { self.oq4_xq_batch.as_ref().unwrap().buf.alias() },
                    shape: vec![n * k / 2],
                    dtype: DType::Raw,
                };
                let s4 = GpuTensor {
                    buf: unsafe { self.oq4_xs_batch.as_ref().unwrap().buf.alias() },
                    shape: vec![n * ng],
                    dtype: DType::F32,
                };
                self.gemm_oq_compact_iu4_w64(w_blocks, &x4, &s4, y, m, k, n, block_stride)?;
                if std::env::var("HIPFIRE_OQ_COMPACT_NO_CORRECT").as_deref() == Ok("1") {
                    return Ok(());
                }
                // The overlay must see the SAME int4 activation the GEMM used:
                // the correction replaces a bulk nibble the GEMM read as zero, so
                // it has to be scaled on the int4 grid, not the int8 one.
                self.oq_compact_x4_transpose(&x4, &s4, &xt, &xst, n, k, ng)?;
                return self.oq_compact_overlay_correct_t(
                    w_blocks,
                    &xt,
                    &xst,
                    y,
                    m,
                    k,
                    n,
                    group,
                    block_stride,
                );
            }
            if use_w64 {
                // The wave64 kernel consumes fragment-interleaved nibble pairs,
                // not int8. The permutation is done ONCE per activation, beside
                // the quantize -- doing it here instead re-permuted the same
                // activation for every projection (gate, up, q, k, v, ...) and
                // cost 7% end to end.
                let xilv = GpuTensor {
                    buf: unsafe { self.oq_xilv_batch.as_ref().unwrap().buf.alias() },
                    shape: vec![n * k],
                    dtype: DType::Raw,
                };
                self.gemm_oq_compact_iu4x2_w64(w_blocks, &xilv, xs, y, m, k, n, block_stride)?;
            } else {
                self.gemm_oq_compact_iu4x2_wmma(w_blocks, xq, xs, y, m, k, n, group, block_stride)?;
            }
            // TIMING-ONLY ABLATION. Skips the transpose + overlay correction so
            // the GEMM's uncorrected speed can be measured directly. Output is
            // NUMERICALLY WRONG (the sparse overlay is simply not applied); this
            // exists to bound what any correction scheme is competing against.
            if std::env::var("HIPFIRE_OQ_COMPACT_NO_CORRECT").as_deref() == Ok("1") {
                return Ok(());
            }
            // Skip when the hoisted quantize already built XT for THIS
            // activation. Keyed on the generation counter plus ng: a group other
            // than 256 gives XsT a different layout, and the hoisted path only
            // ever emits ng = k/256.
            let hoisted = self.oq_xt_gen == self.oq_act_gen
                && self.oq_xt_ng == ng
                && self.oq_xt_n == n
                && std::env::var("HIPFIRE_OQ_XT_HOIST").as_deref() != Ok("0");
            if !hoisted {
                self.oq_compact_x8_transpose(xq, xs, &xt, &xst, n, k, ng)?;
            }
            // ACCUMULATES into y, so it must follow the GEMM on the same Y.
            return self.oq_compact_overlay_correct_t(
                w_blocks,
                &xt,
                &xst,
                y,
                m,
                k,
                n,
                group,
                block_stride,
            );
        }
        self.gemm_oq_compact_grouped_wmma(w_blocks, xq, xs, y, m, k, n, group, block_stride)
    }

    pub fn gemm_oq_compact_act_batched(
        &mut self,
        w_blocks: &GpuTensor,
        x_rot: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        n: usize,
        block_stride: usize,
    ) -> HipResult<()> {
        const GROUP: usize = 256;
        self.quantize_act_oq8_batched_interleaved(x_rot, m, k, n)?;
        let ng = k / GROUP;
        let xq = GpuTensor {
            buf: unsafe { self.oq8_xq_batch.as_ref().unwrap().buf.alias() },
            shape: vec![n * k],
            dtype: DType::Raw,
        };
        let xs = GpuTensor {
            buf: unsafe { self.oq8_xs_batch.as_ref().unwrap().buf.alias() },
            shape: vec![n * ng],
            dtype: DType::F32,
        };
        self.compact_batched_route(w_blocks, &xq, &xs, y, m, k, n, ng, GROUP, block_stride)
    }

    /// Compact-resident Opus W8A8 GEMM: the [`Self::gemm_oq8_grouped_wmma`] core
    /// reading OqPlusCompact (qt=36) blocks DIRECTLY instead of a pre-expanded
    /// dense int8 plane, so oq4.25++ stays ~4.25 bits/weight in VRAM rather than
    /// 8. Same FWHT-rotated W8A8 math and the same int8 activations from
    /// `quantize_act_oq8`, so results are bit-identical to the expanded path
    /// (see `examples/parity_gemm_oq_compact.rs`).
    ///
    /// `w_blocks` is `[M, K/group]` blocks of `block_stride` bytes, each
    /// `[f16 scale | 128 packed int4 | N_out * (u8 idx, i8 val)]`, so
    /// `block_stride == 130 + 2 * N_out`. There is no separate weight-scale
    /// plane — the scale lives in the block.
    /// Small-batch twin of [`Self::gemm_oq_compact_grouped_wmma`]: same inputs
    /// and same output layout, but reads each weight row ONCE and accumulates B
    /// columns instead of tiling B by 16. For B <= 16 (spec-decode verify) the
    /// WMMA path cannot amortize the int4 decode; this can.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_oq_compact_multicol(
        &mut self,
        w_blocks: &GpuTensor,
        x_i8: &GpuTensor,
        x_scales: &GpuTensor,
        y_f32: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
        block_stride: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        // B == 0 reached here before (the router admits `batch_size <= 16`) and
        // launched a kernel whose column loop ran zero times. Keep that a no-op
        // rather than letting the tightened assert below turn it into a panic.
        if batch_size == 0 {
            return Ok(());
        }
        assert!(
            (1..=16).contains(&batch_size),
            "gemv_oq_compact_multicol: B must be in 1..=16"
        );
        // B is a COMPILE-TIME parameter of the kernel: with a runtime B the
        // column loop cannot unroll and `facc[b]` degrades to indexed-register
        // access (`m0` + `v_movrels`/`v_movreld` per column) plus one
        // `s_waitcnt lgkmcnt(0)` per column for its `Xs` load. One entry point
        // per reachable B, so that path no longer exists.
        // WIDE variant: 8 lanes per group and dwordx4 on both streams, so a
        // group-round moves 4x the weights for the same instruction count. Needs
        // ng % 4 == 0 (K % 1024 == 0); the narrow kernel has no such constraint
        // and stays the fallback. HIPFIRE_OQ_COMPACT_MULTICOL_WIDE=1 to enable
        // while it is being proven against the narrow one.
        let ng_ok = (k / 256) % 4 == 0;
        let wide = ng_ok && std::env::var("HIPFIRE_OQ_COMPACT_MULTICOL_WIDE").as_deref() == Ok("1");
        if wide {
            let entry: &str = match batch_size {
                1 => "gemv_oq_compact_multicol_w1",
                2 => "gemv_oq_compact_multicol_w2",
                3 => "gemv_oq_compact_multicol_w3",
                4 => "gemv_oq_compact_multicol_w4",
                5 => "gemv_oq_compact_multicol_w5",
                6 => "gemv_oq_compact_multicol_w6",
                7 => "gemv_oq_compact_multicol_w7",
                8 => "gemv_oq_compact_multicol_w8",
                9 => "gemv_oq_compact_multicol_w9",
                10 => "gemv_oq_compact_multicol_w10",
                11 => "gemv_oq_compact_multicol_w11",
                12 => "gemv_oq_compact_multicol_w12",
                13 => "gemv_oq_compact_multicol_w13",
                14 => "gemv_oq_compact_multicol_w14",
                15 => "gemv_oq_compact_multicol_w15",
                _ => "gemv_oq_compact_multicol_w16",
            };
            self.ensure_kernel(
                "gemv_oq_compact_multicol_wide",
                kernels::GEMV_OQ_COMPACT_MULTICOL_WIDE_SRC,
                entry,
            )?;
            let (wp, xp, xsp, yp) = (
                w_blocks.buf.as_ptr(),
                x_i8.buf.as_ptr(),
                x_scales.buf.as_ptr(),
                y_f32.buf.as_ptr(),
            );
            let (mi, ki, bi, bs) = (m as i32, k as i32, batch_size as i32, block_stride as i32);
            let grid = ((m as u32).div_ceil(3 * 8)).clamp(1, 2048);
            return self.launch_kernargs(
                entry,
                [grid, 1, 1],
                [256, 1, 1],
                0,
                &kernargs![ptr wp, ptr xp, ptr xsp, ptr yp, i32 mi, i32 ki, i32 bi, i32 bs],
            );
        }
        let entry: &str = match batch_size {
            1 => "gemv_oq_compact_multicol_b1",
            2 => "gemv_oq_compact_multicol_b2",
            3 => "gemv_oq_compact_multicol_b3",
            4 => "gemv_oq_compact_multicol_b4",
            5 => "gemv_oq_compact_multicol_b5",
            6 => "gemv_oq_compact_multicol_b6",
            7 => "gemv_oq_compact_multicol_b7",
            8 => "gemv_oq_compact_multicol_b8",
            9 => "gemv_oq_compact_multicol_b9",
            10 => "gemv_oq_compact_multicol_b10",
            11 => "gemv_oq_compact_multicol_b11",
            12 => "gemv_oq_compact_multicol_b12",
            13 => "gemv_oq_compact_multicol_b13",
            14 => "gemv_oq_compact_multicol_b14",
            15 => "gemv_oq_compact_multicol_b15",
            _ => "gemv_oq_compact_multicol_b16",
        };
        self.ensure_kernel(
            "gemv_oq_compact_multicol",
            kernels::GEMV_OQ_COMPACT_MULTICOL_SRC,
            entry,
        )?;
        let (wp, xp, xsp, yp) = (
            w_blocks.buf.as_ptr(),
            x_i8.buf.as_ptr(),
            x_scales.buf.as_ptr(),
            y_f32.buf.as_ptr(),
        );
        let (mi, ki, bi, bs) = (m as i32, k as i32, batch_size as i32, block_stride as i32);
        // Each wave carries RW rows (mirrors the RW choice in the .hip entry
        // macro), so the grid shrinks by RW or most waves launch with no rows.
        let rw: u32 = if batch_size <= 8 { 4 } else { 2 };
        let grid = ((m as u32).div_ceil(8 * rw)).clamp(1, 2048);
        self.launch_kernargs(
            entry,
            [grid, 1, 1],
            [256, 1, 1],
            0,
            &kernargs![ptr wp, ptr xp, ptr xsp, ptr yp, i32 mi, i32 ki, i32 bi, i32 bs],
        )
    }

    #[allow(clippy::too_many_arguments)]
    /// Transpose the packed int4 activation to K-major for the overlay
    /// correction: `[B, K/2]` packed -> `[K, B]` int8, and `[B, ng]` -> `[ng, B]`.
    pub fn oq_compact_x4_transpose(
        &mut self,
        x_i4: &GpuTensor,
        x_scales: &GpuTensor,
        xt: &GpuTensor,
        xst: &GpuTensor,
        b: usize,
        k: usize,
        n_groups: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "oq_compact_x4_transpose",
            kernels::OQ_COMPACT_OVERLAY_CORRECT_T_SRC,
            "oq_compact_x4_transpose",
        )?;
        let (xp, xsp, xtp, xstp) = (
            x_i4.buf.as_ptr(),
            x_scales.buf.as_ptr(),
            xt.buf.as_ptr(),
            xst.buf.as_ptr(),
        );
        let (bi, ki, ngi) = (b as i32, k as i32, n_groups as i32);
        self.launch_kernargs(
            "oq_compact_x4_transpose",
            [(b as u32).div_ceil(256), k as u32, 1],
            [256, 1, 1],
            0,
            &kernargs![ptr xp, ptr xsp, ptr xtp, ptr xstp, i32 bi, i32 ki, i32 ngi],
        )
    }

    /// int8 twin of [`Self::oq_compact_x4_transpose`], for the exact W4A8 path
    /// whose activation is already one byte per element.
    pub fn oq_compact_x8_transpose(
        &mut self,
        x_i8: &GpuTensor,
        x_scales: &GpuTensor,
        xt: &GpuTensor,
        xst: &GpuTensor,
        b: usize,
        k: usize,
        n_groups: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "oq_compact_x8_transpose",
            kernels::OQ_COMPACT_OVERLAY_CORRECT_T_SRC,
            "oq_compact_x8_transpose",
        )?;
        let (xp, xsp, xtp, xstp) = (
            x_i8.buf.as_ptr(),
            x_scales.buf.as_ptr(),
            xt.buf.as_ptr(),
            xst.buf.as_ptr(),
        );
        let (bi, ki, ngi) = (b as i32, k as i32, n_groups as i32);
        self.launch_kernargs(
            "oq_compact_x8_transpose",
            [(b as u32).div_ceil(256), k as u32, 1],
            [256, 1, 1],
            0,
            &kernargs![ptr xp, ptr xsp, ptr xtp, ptr xstp, i32 bi, i32 ki, i32 ngi],
        )
    }

    /// K-major sparse overlay correction. ACCUMULATES into `y_f32`.
    #[allow(clippy::too_many_arguments)]
    pub fn oq_compact_overlay_correct_t(
        &mut self,
        w_blocks: &GpuTensor,
        xt: &GpuTensor,
        xst: &GpuTensor,
        y_f32: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
        group: usize,
        block_stride: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        // Row-coalesced write variant. The `_t` kernel's store is 67% of its
        // runtime (ablation in the kernel header): Y is [B, M], a wave owns one
        // row and 128 b, so 128 stores hit 128 lines at 4 bytes each. `_tr`
        // keeps the same gather and transposes through LDS so one store covers
        // 32 consecutive rows = a full line. Opt out with =0.
        let row_coalesced = std::env::var("HIPFIRE_OQ_OVERLAY_ROWC").as_deref() != Ok("0");
        if row_coalesced {
            return self.oq_compact_overlay_correct_tr(
                w_blocks,
                xt,
                xst,
                y_f32,
                m,
                k,
                batch_size,
                group,
                block_stride,
            );
        }
        self.ensure_kernel(
            "oq_compact_overlay_correct_t",
            kernels::OQ_COMPACT_OVERLAY_CORRECT_T_SRC,
            "oq_compact_overlay_correct_t",
        )?;
        let (wp, xtp, xstp, yp) = (
            w_blocks.buf.as_ptr(),
            xt.buf.as_ptr(),
            xst.buf.as_ptr(),
            y_f32.buf.as_ptr(),
        );
        let (mi, ki, bi) = (m as i32, k as i32, batch_size as i32);
        let (gi, si) = (group as i32, block_stride as i32);
        let waves = 8u32;
        // The kernel holds its b-slice in MAXQ=4 dword register blocks = 512
        // columns per launch. Tile anything wider: letting its `nblk` run past
        // MAXQ indexes acc[]/isum[] out of bounds and faults the GPU (observed
        // at HIPFIRE_PREFILL_MAX_BATCH=1024). XT is [K, B] k-major so a b-tile
        // is strided, not contiguous -- hence a b_off argument rather than a
        // sub-view.
        const B_TILE: usize = 512;
        let mut b_off = 0usize;
        while b_off < batch_size {
            let boi = b_off as i32;
            self.launch_kernargs(
                "oq_compact_overlay_correct_t",
                [(m as u32).div_ceil(waves), 1, 1],
                [waves * 32, 1, 1],
                0,
                &kernargs![ptr wp, ptr xtp, ptr xstp, ptr yp, i32 mi, i32 ki, i32 bi, i32 gi, i32 si, i32 boi],
            )?;
            b_off += B_TILE;
        }
        Ok(())
    }

    /// Row-coalesced twin of [`Self::oq_compact_overlay_correct_t`]: same
    /// gather, but the accumulator is transposed through LDS so each store
    /// covers 32 consecutive rows of `Y` instead of 32 separate cache lines.
    /// Needs no b-tiling -- the b-block is a grid dimension, not a register
    /// array, so there is no `nblk` to overrun.
    #[allow(clippy::too_many_arguments)]
    pub fn oq_compact_overlay_correct_tr(
        &mut self,
        w_blocks: &GpuTensor,
        xt: &GpuTensor,
        xst: &GpuTensor,
        y_f32: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
        group: usize,
        block_stride: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "oq_compact_overlay_correct_tr",
            kernels::OQ_COMPACT_OVERLAY_CORRECT_T_SRC,
            "oq_compact_overlay_correct_tr",
        )?;
        let (wp, xtp, xstp, yp) = (
            w_blocks.buf.as_ptr(),
            xt.buf.as_ptr(),
            xst.buf.as_ptr(),
            y_f32.buf.as_ptr(),
        );
        let (mi, ki, bi) = (m as i32, k as i32, batch_size as i32);
        let (gi, si) = (group as i32, block_stride as i32);
        // Must match OQCO_ROWS / OQCO_BB in the kernel.
        const ROWS: usize = 32;
        const BB: usize = 128;
        self.launch_kernargs(
            "oq_compact_overlay_correct_tr",
            [
                (m as u32).div_ceil(ROWS as u32),
                (batch_size as u32).div_ceil(BB as u32),
                1,
            ],
            [256, 1, 1],
            0,
            &kernargs![ptr wp, ptr xtp, ptr xstp, ptr yp, i32 mi, i32 ki, i32 bi, i32 gi, i32 si],
        )
    }

    /// Sparse overlay correction for the compact W4A4 path. ACCUMULATES into
    /// `y_f32`, so it must run AFTER `gemm_oq_compact_iu4_wmma` on the same Y.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_oq_compact_overlay_correct(
        &mut self,
        w_blocks: &GpuTensor,
        x_i4: &GpuTensor,
        x_scales: &GpuTensor,
        y_f32: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
        group: usize,
        block_stride: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert_eq!(k % group, 0, "overlay_correct: K % group != 0");
        let header = 2 + group / 2;
        let overlays = (block_stride - header) / 2;
        assert!(
            overlays <= 16,
            "overlay_correct: {overlays} overlays exceeds the 16 the kernel keeps"
        );
        self.ensure_kernel(
            "gemv_oq_compact_overlay_correct",
            kernels::GEMV_OQ_COMPACT_OVERLAY_CORRECT_SRC,
            "gemv_oq_compact_overlay_correct",
        )?;
        let wp = w_blocks.buf.as_ptr();
        let xp = x_i4.buf.as_ptr();
        let xsp = x_scales.buf.as_ptr();
        let yp = y_f32.buf.as_ptr();
        let (mut mi, mut ki, mut bi) = (m as i32, k as i32, batch_size as i32);
        let (mut gi, mut si) = (group as i32, block_stride as i32);
        let mut params: Vec<*mut c_void> = vec![
            &wp as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &xsp as *const _ as *mut c_void,
            &yp as *const _ as *mut c_void,
            &mut mi as *mut _ as *mut c_void,
            &mut ki as *mut _ as *mut c_void,
            &mut bi as *mut _ as *mut c_void,
            &mut gi as *mut _ as *mut c_void,
            &mut si as *mut _ as *mut c_void,
        ];
        let waves = 8u32; // 256 threads, one wave per row
        let grid_m = (m as u32).div_ceil(waves);
        let grid_b = 1u32;
        let func = &self.functions["gemv_oq_compact_overlay_correct"];
        unsafe {
            self.hip.launch_kernel(
                func,
                [grid_m, grid_b, 1],
                [waves * 32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// Compact-resident Opus W4 with 8-bit activations carried as TWO iu4
    /// passes. `x_hi` is packed SIGNED int4, `x_lo` packed UNSIGNED int4, both
    /// `[B, K/2]`; together they span int8 exactly and are recombined as
    /// `16*hi + lo` in i32 before scaling.
    ///
    /// Does NOT apply the sparse weight overlay — same contract as the 1-pass
    /// `gemm_oq_compact_iu4_wmma`.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_oq_compact_iu4x2_wmma(
        &mut self,
        w_blocks: &GpuTensor,
        x_i8: &GpuTensor,
        x_scales: &GpuTensor,
        y_f32: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
        group: usize,
        block_stride: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert_eq!(k % group, 0, "gemm_oq_compact_iu4x2_wmma: K % group != 0");
        assert!(
            group == 256 || group == 128,
            "gemm_oq_compact_iu4x2_wmma: compact group must be 256 or 128 (got {group})"
        );
        self.ensure_kernel(
            "gemm_oq_compact_iu4x2_wmma",
            kernels::GEMM_OQ_COMPACT_IU4X2_WMMA_SRC,
            "gemm_oq_compact_iu4x2_wmma",
        )?;
        let wp = w_blocks.buf.as_ptr();
        let xqp = x_i8.buf.as_ptr();
        let xsp = x_scales.buf.as_ptr();
        let yp = y_f32.buf.as_ptr();
        let (mut mi, mut ki, mut bi) = (m as i32, k as i32, batch_size as i32);
        let (mut gi, mut si) = (group as i32, block_stride as i32);
        let mut params: Vec<*mut c_void> = vec![
            &wp as *const _ as *mut c_void,
            &xqp as *const _ as *mut c_void,
            &xsp as *const _ as *mut c_void,
            &yp as *const _ as *mut c_void,
            &mut mi as *mut _ as *mut c_void,
            &mut ki as *mut _ as *mut c_void,
            &mut bi as *mut _ as *mut c_void,
            &mut gi as *mut _ as *mut c_void,
            &mut si as *mut _ as *mut c_void,
        ];
        // Must match OQC4X2_NB / OQC4X2_MW. NB is HALVED against the 1-pass
        // kernel because two live i32 accumulator sets double the register cost.
        const NB: usize = 4;
        const MW: usize = 16;
        let grid_m = m.div_ceil(16 * MW) as u32;
        let grid_b = batch_size.div_ceil(16 * NB) as u32;
        let func = &self.functions["gemm_oq_compact_iu4x2_wmma"];
        unsafe {
            self.hip.launch_kernel(
                func,
                [grid_m, grid_b, 1],
                [(32 * MW) as u32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// Launch one of the pure iu4 WMMA issue-rate probes. `chains` selects the
    /// number of independent accumulator chains (1/2/4/8/16/32); `wave64`
    /// selects which family. No memory is touched by the kernel.
    /// Launch one iu4 WMMA OPERAND-SUPPLY probe. Same shape as
    /// [`Self::wmma_iu4_noop`], but every WMMA re-reads A and B from LDS, so the
    /// gap between the two curves is the cost of operand supply alone.
    pub fn wmma_iu4_lds_probe(
        &mut self,
        out: &GpuTensor,
        blocks: u32,
        iters: i32,
        chains: u32,
        fold: bool,
        stage: Option<(&GpuTensor, usize)>,
        dbuf: bool,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let name = if stage.is_some() && dbuf {
            format!("wmma_iu4_dbuf_w32_c{chains}")
        } else if stage.is_some() {
            format!("wmma_iu4_stage_w32_c{chains}")
        } else if fold {
            format!("wmma_iu4_fold_w32_c{chains}")
        } else {
            format!("wmma_iu4_lds_w32_c{chains}")
        };
        self.ensure_kernel("wmma_iu4_lds_probe", kernels::WMMA_IU4_LDS_PROBE_SRC, &name)?;
        let op = out.buf.as_ptr();
        let mut it = iters;
        let mut params: Vec<*mut c_void> = vec![
            &op as *const _ as *mut c_void,
            &mut it as *mut _ as *mut c_void,
        ];
        // The staging variant takes (src, mask) after `out`; keep the arg order
        // matching the kernel signature (out, src, iters, srcmask).
        let sp;
        let mut smask;
        if let Some((src, words)) = stage {
            sp = src.buf.as_ptr();
            smask = (words - 1) as i32;
            params = vec![
                &op as *const _ as *mut c_void,
                &sp as *const _ as *mut c_void,
                &mut it as *mut _ as *mut c_void,
                &mut smask as *mut _ as *mut c_void,
            ];
        }
        let func = &self.functions[&name];
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

    /// Rung 5 probe: real GEMM tiling. `kstrips` BK=64 strips.
    pub fn wmma_iu4_tiled(
        &mut self,
        out: &GpuTensor,
        src: &GpuTensor,
        blocks: u32,
        kstrips: i32,
        src_words: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "wmma_iu4_tiled_probe",
            kernels::WMMA_IU4_TILED_PROBE_SRC,
            "wmma_iu4_tiled_w32",
        )?;
        let op = out.buf.as_ptr();
        let sp = src.buf.as_ptr();
        let mut ks = kstrips;
        let mut sm = (src_words - 1) as i32;
        let mut params: Vec<*mut c_void> = vec![
            &op as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &mut ks as *mut _ as *mut c_void,
            &mut sm as *mut _ as *mut c_void,
        ];
        let func = &self.functions["wmma_iu4_tiled_w32"];
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

    pub fn wmma_iu4_noop(
        &mut self,
        out: &GpuTensor,
        blocks: u32,
        iters: i32,
        chains: u32,
        wave64: bool,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let (file, src) = if wave64 {
            ("wmma_iu4_noop_w64", kernels::WMMA_IU4_NOOP_W64_SRC)
        } else {
            ("wmma_iu4_noop_w32", kernels::WMMA_IU4_NOOP_W32_SRC)
        };
        let name = format!("{file}_c{chains}");
        self.ensure_kernel(file, src, &name)?;
        let op = out.buf.as_ptr();
        let mut it = iters;
        let mut params: Vec<*mut c_void> = vec![
            &op as *const _ as *mut c_void,
            &mut it as *mut _ as *mut c_void,
        ];
        let func = &self.functions[&name];
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

    /// Compact-resident Opus W4A4 on the tuned **wave64** structure. Same block
    /// bytes as `gemm_oq_compact_iu4_wmma`; the A operand is the compact nibble
    /// plane read directly, which is already a dense `[M, K/2]` int4 array.
    ///
    /// Does NOT apply the sparse overlay. Requires `K % 256 == 0`.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_oq_compact_iu4_w64(
        &mut self,
        w_blocks: &GpuTensor,
        x_i4: &GpuTensor,
        x_scales: &GpuTensor,
        y_f32: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
        block_stride: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert_eq!(
            k % 256,
            0,
            "gemm_oq_compact_iu4_w64: K must be a multiple of 256"
        );
        self.ensure_kernel(
            "gemm_oq_compact_iu4_w64",
            kernels::GEMM_OQ_COMPACT_IU4_W64_SRC,
            "gemm_oq_compact_iu4_w64",
        )?;
        let wp = w_blocks.buf.as_ptr();
        let xp = x_i4.buf.as_ptr();
        let xsp = x_scales.buf.as_ptr();
        let yp = y_f32.buf.as_ptr();
        let (mut mi, mut ki, mut bi) = (m as i32, k as i32, batch_size as i32);
        let mut si = block_stride as i32;
        let mut params: Vec<*mut c_void> = vec![
            &wp as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &xsp as *const _ as *mut c_void,
            &yp as *const _ as *mut c_void,
            &mut mi as *mut _ as *mut c_void,
            &mut ki as *mut _ as *mut c_void,
            &mut bi as *mut _ as *mut c_void,
            &mut si as *mut _ as *mut c_void,
        ];
        // Must match BM / BN in the kernel.
        const BM: usize = 64;
        const BN: usize = 256;
        let func = &self.functions["gemm_oq_compact_iu4_w64"];
        unsafe {
            self.hip.launch_kernel(
                func,
                [m.div_ceil(BM) as u32, batch_size.div_ceil(BN) as u32, 1],
                [256, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// Wave64 EXACT W4A8 compact GEMM: the tuned wave64 structure
    /// (BK=64 strip, register-staged double buffer, b128 fragments) carrying the
    /// two-pass activation split `x = 16*hi + lo`. Consumes int8 activations
    /// directly and splits the digit planes into LDS. WNt is halved against the
    /// 1-pass twin to pay for the second i32 accumulator set.
    ///
    /// Does NOT apply the sparse overlay -- separate pass, as for both twins.
    #[allow(clippy::too_many_arguments)]
    /// Tiled wave32 exact-W4A8 compact GEMM: BM=64 x BN=128, BK=64, 8 wave32
    /// waves as 2x4, double-buffered LDS. Same contract as
    /// [`Self::gemm_oq_compact_iu4x2_wmma`] (int8 activations, no overlay).
    #[allow(clippy::too_many_arguments)]
    /// First-principles ladder kernel. Launch geometry is read from the two
    /// consts below, which must track the rung's tile shape.
    #[allow(clippy::too_many_arguments)]
    /// int8 activations -> fragment-interleaved nibble pairs, in place of the
    /// per-GEMM in-kernel split. Cross-process verified at 1.184x on the ladder
    /// kernel; see `.agents/skills/hipfire-kernel-tuning/levers.md` §9.
    pub fn act_interleave_nibbles(
        &mut self,
        src: &GpuTensor,
        dst: &GpuTensor,
        n: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert_eq!(k % 16, 0, "act_interleave_nibbles: K % 16 != 0");
        self.ensure_kernel(
            "act_interleave_nibbles",
            kernels::ACT_INTERLEAVE_NIBBLES_SRC,
            "act_interleave_nibbles",
        )?;
        let sp = src.buf.as_ptr();
        let dp = dst.buf.as_ptr();
        let (mut ni, mut ki) = (n as i32, k as i32);
        let mut params: Vec<*mut c_void> = vec![
            &sp as *const _ as *mut c_void,
            &dp as *const _ as *mut c_void,
            &mut ni as *mut _ as *mut c_void,
            &mut ki as *mut _ as *mut c_void,
        ];
        let total = n * (k / 16);
        let func = &self.functions["act_interleave_nibbles"];
        unsafe {
            self.hip.launch_kernel(
                func,
                [total.div_ceil(256) as u32, 1, 1],
                [256, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// Diagnostic (HIPFIRE_HIPASS_STATS=1): count how often the exact-W4A8 hi
    /// pass would be a no-op. `x = 16*x_hi + x_lo`, so x_hi is zero exactly when
    /// |x| < 16, and an H-pass WMMA is dead iff every value in its 16x16 fragment
    /// is. Reports BOTH granularities that matter: per fragment, and per BN=128
    /// column block (a wave issues WNt=4 column tiles together, so a
    /// per-fragment decision would diverge across `j`).
    fn hipass_stats(&mut self, x_i8: &GpuTensor, n: usize, k: usize) -> HipResult<()> {
        use std::sync::atomic::{AtomicU64, Ordering};
        static FRAG: AtomicU64 = AtomicU64::new(0);
        static FRAG_DEAD: AtomicU64 = AtomicU64::new(0);
        static BLK: AtomicU64 = AtomicU64::new(0);
        static BLK_DEAD: AtomicU64 = AtomicU64::new(0);
        static CALLS: AtomicU64 = AtomicU64::new(0);
        static SUMMAX: AtomicU64 = AtomicU64::new(0);
        if n < 16 || k < 16 {
            return Ok(());
        }
        let (tn, tk) = (n / 16, k / 16);
        let out = self.alloc_tensor(&[tn * tk], DType::Raw)?;
        self.act_hipass_tilemax(x_i8, &out, n, k)?;
        let v = self.download_raw(&out, tn * tk)?;
        let mut fd = 0u64;
        let mut bd = 0u64;
        let mut sm = 0u64;
        for m in &v {
            sm += *m as u64;
            if *m < 16 {
                fd += 1;
            }
        }
        // BN=128 column block == 8 consecutive n-tiles at one k-tile.
        let nblk = tn / 8;
        for b in 0..nblk {
            for t in 0..tk {
                if (0..8).all(|i| v[(b * 8 + i) * tk + t] < 16) {
                    bd += 1;
                }
            }
        }
        FRAG.fetch_add((tn * tk) as u64, Ordering::Relaxed);
        FRAG_DEAD.fetch_add(fd, Ordering::Relaxed);
        BLK.fetch_add((nblk * tk) as u64, Ordering::Relaxed);
        BLK_DEAD.fetch_add(bd, Ordering::Relaxed);
        SUMMAX.fetch_add(sm, Ordering::Relaxed);
        let c = CALLS.fetch_add(1, Ordering::Relaxed) + 1;
        if c % 256 == 0 {
            let (f, fdd) = (
                FRAG.load(Ordering::Relaxed),
                FRAG_DEAD.load(Ordering::Relaxed),
            );
            let (bl, bld) = (
                BLK.load(Ordering::Relaxed),
                BLK_DEAD.load(Ordering::Relaxed),
            );
            let sx = SUMMAX.load(Ordering::Relaxed);
            eprintln!(
                "hipass[{c} calls]: fragments {fdd}/{f} = {:.4}% dead | BN=128 blocks {bld}/{bl} = {:.4}% dead | mean fragment max|x| = {:.1}",
                fdd as f64 / f as f64 * 100.0,
                bld as f64 / bl as f64 * 100.0,
                sx as f64 / f as f64
            );
        }
        Ok(())
    }

    /// Diagnostic: fill `out` with max|x| of each 16x16 fragment of `x_i8`.
    /// `out` must hold at least (n/16)*(k/16) bytes. Used to count how often the
    /// exact-W4A8 hi pass would be a no-op; see the kernel header.
    pub fn act_hipass_tilemax(
        &mut self,
        x_i8: &GpuTensor,
        out: &GpuTensor,
        n: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "act_hipass_tilemax",
            kernels::ACT_HIPASS_TILEMAX_SRC,
            "act_hipass_tilemax",
        )?;
        let xp = x_i8.buf.as_ptr();
        let op = out.buf.as_ptr();
        let (ni, ki) = (n as i32, k as i32);
        let total = (n / 16) * (k / 16);
        self.launch_kernargs(
            "act_hipass_tilemax",
            [total.div_ceil(256) as u32, 1, 1],
            [256, 1, 1],
            0,
            &kernargs![ptr xp, ptr op, i32 ni, i32 ki],
        )
    }

    pub fn gemm_oq_compact_ladder(
        &mut self,
        w_blocks: &GpuTensor,
        x_i8: &GpuTensor,
        x_scales: &GpuTensor,
        y_f32: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
        block_stride: usize,
        bm: usize,
        bn: usize,
        threads: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert_eq!(k % 256, 0, "gemm_oq_compact_ladder: K % 256 != 0");
        self.ensure_kernel(
            "gemm_oq_compact_ladder",
            kernels::GEMM_OQ_COMPACT_LADDER_SRC,
            "gemm_oq_compact_ladder",
        )?;
        let wp = w_blocks.buf.as_ptr();
        let xp = x_i8.buf.as_ptr();
        let xsp = x_scales.buf.as_ptr();
        let yp = y_f32.buf.as_ptr();
        let (mut mi, mut ki, mut bi) = (m as i32, k as i32, batch_size as i32);
        let mut si = block_stride as i32;
        let mut params: Vec<*mut c_void> = vec![
            &wp as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &xsp as *const _ as *mut c_void,
            &yp as *const _ as *mut c_void,
            &mut mi as *mut _ as *mut c_void,
            &mut ki as *mut _ as *mut c_void,
            &mut bi as *mut _ as *mut c_void,
            &mut si as *mut _ as *mut c_void,
        ];
        let func = &self.functions["gemm_oq_compact_ladder"];
        unsafe {
            self.hip.launch_kernel(
                func,
                [m.div_ceil(bm) as u32, batch_size.div_ceil(bn) as u32, 1],
                [threads as u32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    pub fn gemm_oq_compact_iu4x2_tiled(
        &mut self,
        w_blocks: &GpuTensor,
        x_i8: &GpuTensor,
        x_scales: &GpuTensor,
        y_f32: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
        block_stride: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert_eq!(k % 256, 0, "gemm_oq_compact_iu4x2_tiled: K % 256 != 0");
        self.ensure_kernel(
            "gemm_oq_compact_iu4x2_tiled",
            kernels::GEMM_OQ_COMPACT_IU4X2_TILED_SRC,
            "gemm_oq_compact_iu4x2_tiled",
        )?;
        let wp = w_blocks.buf.as_ptr();
        let xp = x_i8.buf.as_ptr();
        let xsp = x_scales.buf.as_ptr();
        let yp = y_f32.buf.as_ptr();
        let (mut mi, mut ki, mut bi) = (m as i32, k as i32, batch_size as i32);
        let mut si = block_stride as i32;
        let mut params: Vec<*mut c_void> = vec![
            &wp as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &xsp as *const _ as *mut c_void,
            &yp as *const _ as *mut c_void,
            &mut mi as *mut _ as *mut c_void,
            &mut ki as *mut _ as *mut c_void,
            &mut bi as *mut _ as *mut c_void,
            &mut si as *mut _ as *mut c_void,
        ];
        const BM: usize = 64;
        const BN: usize = 128;
        let func = &self.functions["gemm_oq_compact_iu4x2_tiled"];
        unsafe {
            self.hip.launch_kernel(
                func,
                [m.div_ceil(BM) as u32, batch_size.div_ceil(BN) as u32, 1],
                [256, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    pub fn gemm_oq_compact_iu4x2_w64(
        &mut self,
        w_blocks: &GpuTensor,
        x_i8: &GpuTensor,
        x_scales: &GpuTensor,
        y_f32: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
        block_stride: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert_eq!(
            k % 256,
            0,
            "gemm_oq_compact_iu4x2_w64: K must be a multiple of 256"
        );
        self.ensure_kernel(
            "gemm_oq_compact_iu4x2_w64",
            kernels::GEMM_OQ_COMPACT_IU4X2_W64_SRC,
            "gemm_oq_compact_iu4x2_w64",
        )?;
        let wp = w_blocks.buf.as_ptr();
        let xp = x_i8.buf.as_ptr();
        let xsp = x_scales.buf.as_ptr();
        let yp = y_f32.buf.as_ptr();
        let (mut mi, mut ki, mut bi) = (m as i32, k as i32, batch_size as i32);
        let mut si = block_stride as i32;
        let mut params: Vec<*mut c_void> = vec![
            &wp as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &xsp as *const _ as *mut c_void,
            &yp as *const _ as *mut c_void,
            &mut mi as *mut _ as *mut c_void,
            &mut ki as *mut _ as *mut c_void,
            &mut bi as *mut _ as *mut c_void,
            &mut si as *mut _ as *mut c_void,
        ];
        // Must match BM / BN / BLOCK in the kernel.
        const WARPS_M: usize = 2;
        const W_MT: usize = 2;
        const W_NT: usize = 4;
        const BM: usize = WARPS_M * W_MT * 16;
        const WARPS_N: usize = 2;
        const BN: usize = WARPS_N * W_NT * 16;
        const BLOCK: usize = WARPS_M * WARPS_N * 64;
        let func = &self.functions["gemm_oq_compact_iu4x2_w64"];
        unsafe {
            self.hip.launch_kernel(
                func,
                [batch_size.div_ceil(BN) as u32, m.div_ceil(BM) as u32, 1],
                [BLOCK as u32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// Compact-resident Opus **W4A4** GEMM. Same block bytes as
    /// `gemm_oq_compact_grouped_wmma`, but the bulk nibbles go to
    /// `v_wmma_i32_16x16x16_iu4` raw and `x_i4` is packed signed int4
    /// (`[B, K/2]`, byte = k_even | k_odd<<4).
    ///
    /// Does NOT apply the sparse overlay — the loader zeroes the bulk nibble
    /// under each entry, so those positions contribute 0 here and the caller
    /// must add the `val * x[idx]` correction separately.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_oq_compact_iu4_wmma(
        &mut self,
        w_blocks: &GpuTensor,
        x_i4: &GpuTensor,
        x_scales: &GpuTensor,
        y_f32: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
        group: usize,
        block_stride: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert_eq!(k % group, 0, "gemm_oq_compact_iu4_wmma: K % group != 0");
        assert!(
            group == 256 || group == 128,
            "gemm_oq_compact_iu4_wmma: compact group must be 256 or 128 (got {group})"
        );
        self.ensure_kernel(
            "gemm_oq_compact_iu4_wmma",
            kernels::GEMM_OQ_COMPACT_IU4_WMMA_SRC,
            "gemm_oq_compact_iu4_wmma",
        )?;
        let wp = w_blocks.buf.as_ptr();
        let xp = x_i4.buf.as_ptr();
        let xsp = x_scales.buf.as_ptr();
        let yp = y_f32.buf.as_ptr();
        let (mut mi, mut ki, mut bi) = (m as i32, k as i32, batch_size as i32);
        let (mut gi, mut si) = (group as i32, block_stride as i32);
        let mut params: Vec<*mut c_void> = vec![
            &wp as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &xsp as *const _ as *mut c_void,
            &yp as *const _ as *mut c_void,
            &mut mi as *mut _ as *mut c_void,
            &mut ki as *mut _ as *mut c_void,
            &mut bi as *mut _ as *mut c_void,
            &mut gi as *mut _ as *mut c_void,
            &mut si as *mut _ as *mut c_void,
        ];
        // Must match OQC4_NB / OQC4_MW in the kernel.
        const OQC4_NB: usize = 8;
        const OQC4_MW: usize = 16;
        let grid_m = m.div_ceil(16 * OQC4_MW) as u32;
        let grid_b = batch_size.div_ceil(16 * OQC4_NB) as u32;
        let func = &self.functions["gemm_oq_compact_iu4_wmma"];
        unsafe {
            self.hip.launch_kernel(
                func,
                [grid_m, grid_b, 1],
                [(32 * OQC4_MW) as u32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    pub fn gemm_oq_compact_grouped_wmma(
        &mut self,
        w_blocks: &GpuTensor,
        x_i8: &GpuTensor,
        x_scales: &GpuTensor,
        y_f32: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
        group: usize,
        block_stride: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert_eq!(
            k % group,
            0,
            "gemm_oq_compact_grouped_wmma: K must be a multiple of group"
        );
        // 256 and 128 are the two defined compact groups. Larger groups are not
        // representable: the overlay index is a u8, so it cannot address a
        // position >= 256 (see
        // docs/experiments/2026-08-06-oq-compact-group-size.md).
        assert!(
            group == 256 || group == 128,
            "gemm_oq_compact_grouped_wmma: compact group must be 256 or 128 (got {group})"
        );
        // Mirrors the host oracle's contract (`oqplus_compact_to_oq8_combined`):
        // a valid block carries at least one overlay, so the minimum stride is
        // header + 2, where header = f16 scale + group/2 nibble bytes.
        let header = 2 + group / 2;
        assert!(
            block_stride >= header + 2 && (block_stride - header) % 2 == 0,
            "gemm_oq_compact_grouped_wmma: block_stride {block_stride} invalid (expected {header} + 2*N_out, N_out >= 1)"
        );
        // The kernel holds the overlay table in registers; refuse rather than
        // silently clip a block carrying more outliers than it can keep.
        // SMALL BATCH -> multi-column GEMV. This kernel tiles B by 16, so at
        // spec-decode's verify width (B = K+1, typically 8) it computes a 16-wide
        // WMMA tile for 8 useful columns with no N to amortize the int4 decode
        // over — measured at 64 us per draft token against plain decode's 66.
        // `gemv_oq_compact_multicol` reads each weight row ONCE and accumulates
        // B columns, so verify costs one weight sweep for all B tokens.
        // RE-TESTED 2026-08-21 and this routing HOLDS. multicol is now 80.7% of a
        // spec-decode profile and sustains only ~35 GB/s against a 233 GB/s
        // ceiling, which makes it look like the thing to route around. It is not:
        // sending B <= 16 to the WMMA GEMM instead measured 6.72 tok/s against
        // multicol's 11.84 on Qwen3.8-27B/DFlash2 (tau 4.875, identical both
        // ways). The half-empty 16-wide tile at B=8 costs more than multicol's
        // low bandwidth. The lever is making THIS kernel faster, not replacing it.
        if batch_size <= 16 && group == 256 {
            return self.gemv_oq_compact_multicol(
                w_blocks,
                x_i8,
                x_scales,
                y_f32,
                m,
                k,
                batch_size,
                block_stride,
            );
        }

        // Must track OQC_MAX_OVERLAYS in the kernel. Raising it costs REGISTERS
        // there, not just an unused bound: 32 measured 9.41 TOPS against 16's
        // 10.24. 16 admits oq4.25++ (3), oq4.5++ (7) and the parity suite's
        // N_out=16 cell at no cost over 8.
        const MAX_OVERLAYS: usize = 16;
        let overlays = (block_stride - header) / 2;
        assert!(
            overlays <= MAX_OVERLAYS,
            "gemm_oq_compact_grouped_wmma: {overlays} overlays exceeds the {MAX_OVERLAYS} the kernel keeps in registers"
        );
        self.ensure_kernel(
            "gemm_oq_compact_grouped_wmma",
            kernels::GEMM_OQ_COMPACT_GROUPED_WMMA_SRC,
            "gemm_oq_compact_grouped_wmma",
        )?;
        let wp = w_blocks.buf.as_ptr();
        let xp = x_i8.buf.as_ptr();
        let xsp = x_scales.buf.as_ptr();
        let yp = y_f32.buf.as_ptr();
        let mut mi = m as i32;
        let mut ki = k as i32;
        let mut bi = batch_size as i32;
        let mut gi = group as i32;
        let mut bs = block_stride as i32;
        let mut params: Vec<*mut c_void> = vec![
            &wp as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &xsp as *const _ as *mut c_void,
            &yp as *const _ as *mut c_void,
            &mut mi as *mut _ as *mut c_void,
            &mut ki as *mut _ as *mut c_void,
            &mut bi as *mut _ as *mut c_void,
            &mut gi as *mut _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
        ];
        // Must match OQC_MW in the kernel: one workgroup covers OQC_MW row-tiles,
        // one per wave, so they share the same X columns through L1.
        const OQC_MW: usize = 16;
        let grid_m = m.div_ceil(16 * OQC_MW) as u32;
        // Must match OQC_NB in the kernel: one workgroup covers OQC_NB b-tiles
        // so the decoded weight tile is reused instead of re-read per b-tile.
        const OQC_NB: usize = 8;
        let grid_b = batch_size.div_ceil(16 * OQC_NB) as u32;
        let func = &self.functions["gemm_oq_compact_grouped_wmma"];
        unsafe {
            self.hip.launch_kernel(
                func,
                [grid_m, grid_b, 1],
                [(32 * OQC_MW) as u32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// Register-tiled unified Opus-Quant GEMM for W4A8 (`w_bits == 4`, packed
    /// int4 weight `[M,K/2]`) and W8A8 (`w_bits == 8`, int8 weight `[M,K]`).
    /// Dynamic-int8 activation (`X` int8 `[B,K]`, `Xs` `[B,K/group]`), iu8 WMMA,
    /// per-group rescale → `Y` f32 `[B,M]`. The weight fetch is the only per-width
    /// difference. `(mb, nb)` ∈ {(2,2),(2,4)}.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_opus_tiled_wmma(
        &mut self,
        w_bits: usize,
        w_packed: &GpuTensor,
        w_scales: &GpuTensor,
        x_i8: &GpuTensor,
        x_scales: &GpuTensor,
        y_f32: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
        group: usize,
        mb: usize,
        nb: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert_eq!(
            k % group,
            0,
            "gemm_opus_tiled_wmma: K must be a multiple of group"
        );
        assert_eq!(
            group % 16,
            0,
            "gemm_opus_tiled_wmma: group must be a multiple of 16"
        );
        let kname = match (w_bits, mb, nb) {
            (8, 2, 2) => "gemm_opus_w8a8_tiled_wmma_2x2",
            (8, 2, 4) => "gemm_opus_w8a8_tiled_wmma_2x4",
            (4, 2, 2) => "gemm_opus_w4a8_tiled_wmma_2x2",
            (4, 2, 4) => "gemm_opus_w4a8_tiled_wmma_2x4",
            _ => panic!("gemm_opus_tiled_wmma: unsupported w{w_bits} tiling {mb}x{nb}"),
        };
        self.ensure_kernel(kname, kernels::GEMM_OPUS_TILED_WMMA_SRC, kname)?;
        let wp = w_packed.buf.as_ptr();
        let wsp = w_scales.buf.as_ptr();
        let xp = x_i8.buf.as_ptr();
        let xsp = x_scales.buf.as_ptr();
        let yp = y_f32.buf.as_ptr();
        let mut mi = m as i32;
        let mut ki = k as i32;
        let mut bi = batch_size as i32;
        let mut gi = group as i32;
        let mut params: Vec<*mut c_void> = vec![
            &wp as *const _ as *mut c_void,
            &wsp as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &xsp as *const _ as *mut c_void,
            &yp as *const _ as *mut c_void,
            &mut mi as *mut _ as *mut c_void,
            &mut ki as *mut _ as *mut c_void,
            &mut bi as *mut _ as *mut c_void,
            &mut gi as *mut _ as *mut c_void,
        ];
        let grid_m = m.div_ceil(16 * mb) as u32;
        let grid_b = batch_size.div_ceil(16 * nb) as u32;
        let func = &self.functions[kname];
        unsafe {
            self.hip.launch_kernel(
                func,
                [grid_m, grid_b, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// Mixed-precision Opus GEMM (unsigned weight codes + zero-point fold).
    /// Consumes W{8,4,2,1}A8: dense unsigned codes, the WMMA weight operand flagged
    /// unsigned, and the symmetric zero-point folded out per group using `x_sum`
    /// (`Σ_{k∈g} x[b,k]`). One kernel body per width; see `gemm_opus_tiled_wmma.hip`
    /// and the `hipfire_quantize::opus_lowbit` CPU reference.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_opus_tiled_wmma_u(
        &mut self,
        w_bits: usize,
        w_packed: &GpuTensor,
        w_scales: &GpuTensor,
        x_i8: &GpuTensor,
        x_scales: &GpuTensor,
        x_sum: &GpuTensor,
        y_f32: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
        group: usize,
        mb: usize,
        nb: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert_eq!(
            k % group,
            0,
            "gemm_opus_tiled_wmma_u: K must be a multiple of group"
        );
        assert_eq!(
            group % 16,
            0,
            "gemm_opus_tiled_wmma_u: group must be a multiple of 16"
        );
        let kname = match (w_bits, mb, nb) {
            (8, 2, 2) => "gemm_opus_w8a8u_tiled_wmma_2x2",
            (8, 2, 4) => "gemm_opus_w8a8u_tiled_wmma_2x4",
            (4, 2, 2) => "gemm_opus_w4a8u_tiled_wmma_2x2",
            (4, 2, 4) => "gemm_opus_w4a8u_tiled_wmma_2x4",
            (2, 2, 2) => "gemm_opus_w2a8u_tiled_wmma_2x2",
            (2, 2, 4) => "gemm_opus_w2a8u_tiled_wmma_2x4",
            (1, 2, 2) => "gemm_opus_w1a8u_tiled_wmma_2x2",
            (1, 2, 4) => "gemm_opus_w1a8u_tiled_wmma_2x4",
            _ => panic!("gemm_opus_tiled_wmma_u: unsupported w{w_bits} tiling {mb}x{nb}"),
        };
        self.ensure_kernel(kname, kernels::GEMM_OPUS_TILED_WMMA_SRC, kname)?;
        let wp = w_packed.buf.as_ptr();
        let wsp = w_scales.buf.as_ptr();
        let xp = x_i8.buf.as_ptr();
        let xsp = x_scales.buf.as_ptr();
        let xsump = x_sum.buf.as_ptr();
        let yp = y_f32.buf.as_ptr();
        let mut mi = m as i32;
        let mut ki = k as i32;
        let mut bi = batch_size as i32;
        let mut gi = group as i32;
        let mut params: Vec<*mut c_void> = vec![
            &wp as *const _ as *mut c_void,
            &wsp as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &xsp as *const _ as *mut c_void,
            &xsump as *const _ as *mut c_void,
            &yp as *const _ as *mut c_void,
            &mut mi as *mut _ as *mut c_void,
            &mut ki as *mut _ as *mut c_void,
            &mut bi as *mut _ as *mut c_void,
            &mut gi as *mut _ as *mut c_void,
        ];
        let grid_m = m.div_ceil(16 * mb) as u32;
        let grid_b = batch_size.div_ceil(16 * nb) as u32;
        let func = &self.functions[kname];
        unsafe {
            self.hip.launch_kernel(
                func,
                [grid_m, grid_b, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// Opus Quant W8A8 dynamic int8 activation quantizer (f32 → signed int8 +
    /// per-group f32 scales). `xq_i8` is [B,K] int8; `xs` is [B,K/group] f32.
    pub fn quantize_act_oq8(
        &mut self,
        x_f32: &GpuTensor,
        xq_i8: &GpuTensor,
        xs: &GpuTensor,
        batch_size: usize,
        k: usize,
        group: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert_eq!(
            group % 32,
            0,
            "quantize_act_oq8: group must be a multiple of 32"
        );
        assert_eq!(
            k % group,
            0,
            "quantize_act_oq8: K must be a multiple of group"
        );
        self.ensure_kernel(
            "quantize_act_oq8",
            kernels::QUANTIZE_ACT_OQ8_SRC,
            "quantize_act_oq8",
        )?;
        let xp = x_f32.buf.as_ptr();
        let xqp = xq_i8.buf.as_ptr();
        let xsp = xs.buf.as_ptr();
        let mut bi = batch_size as i32;
        let mut ki = k as i32;
        let mut gi = group as i32;
        let mut params: Vec<*mut c_void> = vec![
            &xp as *const _ as *mut c_void,
            &xqp as *const _ as *mut c_void,
            &xsp as *const _ as *mut c_void,
            &mut bi as *mut _ as *mut c_void,
            &mut ki as *mut _ as *mut c_void,
            &mut gi as *mut _ as *mut c_void,
        ];
        let grid_g = (k / group) as u32;
        let grid_b = batch_size as u32;
        let func = &self.functions["quantize_act_oq8"];
        unsafe {
            self.hip.launch_kernel(
                func,
                [grid_g, grid_b, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// As [`Self::quantize_act_oq8`], but also emits the per-group signed sum
    /// `x_sum` [B, K/group] (int32) that the mixed-precision fold GEMM
    /// ([`Self::gemm_opus_tiled_wmma_u`]) uses to cancel the weight zero-point.
    #[allow(clippy::too_many_arguments)]
    pub fn quantize_act_oq8_sum(
        &mut self,
        x_f32: &GpuTensor,
        xq_i8: &GpuTensor,
        xs: &GpuTensor,
        x_sum: &GpuTensor,
        batch_size: usize,
        k: usize,
        group: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert_eq!(
            group % 32,
            0,
            "quantize_act_oq8_sum: group must be a multiple of 32"
        );
        assert_eq!(
            k % group,
            0,
            "quantize_act_oq8_sum: K must be a multiple of group"
        );
        self.ensure_kernel(
            "quantize_act_oq8_sum",
            kernels::QUANTIZE_ACT_OQ8_SRC,
            "quantize_act_oq8_sum",
        )?;
        let xp = x_f32.buf.as_ptr();
        let xqp = xq_i8.buf.as_ptr();
        let xsp = xs.buf.as_ptr();
        let xsump = x_sum.buf.as_ptr();
        let mut bi = batch_size as i32;
        let mut ki = k as i32;
        let mut gi = group as i32;
        let mut params: Vec<*mut c_void> = vec![
            &xp as *const _ as *mut c_void,
            &xqp as *const _ as *mut c_void,
            &xsp as *const _ as *mut c_void,
            &xsump as *const _ as *mut c_void,
            &mut bi as *mut _ as *mut c_void,
            &mut ki as *mut _ as *mut c_void,
            &mut gi as *mut _ as *mut c_void,
        ];
        let grid_g = (k / group) as u32;
        let grid_b = batch_size as u32;
        let func = &self.functions["quantize_act_oq8_sum"];
        unsafe {
            self.hip.launch_kernel(
                func,
                [grid_g, grid_b, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// Batched W4A4 (Opus oq4) GEMM for prefill: int4-quantize the FWHT-rotated
    /// activation `x_rot` [N×K] ONCE into the shared batched scratch, then a
    /// grouped WMMA GEMM into `y` [N×M]. `w_combined` is the loader's
    /// `[nibbles M*K/2 | f32 scales M*ng]` Raw buffer; the scale view is derived
    /// via `sub_offset`. group is fixed at 256 (oq4 codec). This is the batched
    /// counterpart of the decode `gemv_oq4_grouped` quantize+GEMV.
    pub fn gemm_oq4_grouped_act_batched(
        &mut self,
        w_combined: &GpuTensor,
        x_rot: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        n: usize,
    ) -> HipResult<()> {
        const GROUP: usize = 256;
        let group_x = oq4_act_group().min(k);
        self.ensure_oq4_scratch_batched(n, k, m)?;
        let ng = k / GROUP;
        let ngx = k / group_x;
        let xq = GpuTensor {
            buf: unsafe { self.oq4_xq_batch.as_ref().unwrap().buf.alias() },
            shape: vec![n * (k / 2)],
            dtype: DType::Raw,
        };
        let xs = GpuTensor {
            buf: unsafe { self.oq4_xs_batch.as_ref().unwrap().buf.alias() },
            shape: vec![n * ngx],
            dtype: DType::F32,
        };
        self.quantize_act_oq4(x_rot, &xq, &xs, n, k, group_x)?;
        let ws = w_combined.sub_offset(m * (k / 2), m * ng * 4);
        if group_x != GROUP {
            // Finer activation group than the weight codec's 256 — the decoupled
            // kernel. Needs the LDS tiling (BK-boundary flush), so only for n>=128.
            if n >= 128 {
                return self.gemm_oq4_grouped_wmma_lds_gx(
                    w_combined, &ws, &xq, &xs, y, m, k, n, GROUP, group_x,
                );
            }
            panic!("HIPFIRE_OQ4_ACT_GROUP != 256 needs the LDS path (n>=128), got n={n}");
        }
        // For prefill-sized batches, the LDS-staged kernel is bit-identical to the
        // zero-LDS original but ~1.8× faster (beats W4A8-MMQ; see the A3 gate in
        // docs/plans/2026-07-30-oq4-w4a4-near-lossless.md §9). It needs K%64==0
        // (group=256 guarantees it) and a full BN=128 N-tile to be worth it, so the
        // original stays for decode/small batches. Bit-exact ⇒ no correctness risk.
        if n >= 128 {
            self.gemm_oq4_grouped_wmma_lds(w_combined, &ws, &xq, &xs, y, m, k, n, GROUP)
        } else {
            self.gemm_oq4_grouped_wmma(w_combined, &ws, &xq, &xs, y, m, k, n, GROUP)
        }
    }

    /// Batched W8A8 (Opus oq8) GEMM for prefill: int8-quantize the FWHT-rotated
    /// activation `x_rot` [N×K] once into the shared batched scratch, then the
    /// grouped int8 WMMA GEMM into `y` [N×M]. The oq8 counterpart of
    /// [`Self::gemm_oq4_grouped_act_batched`]; `group` is fixed at 256 (oq8
    /// codec). Same rotate-then-quantize-then-GEMM sequence that
    /// `weight_gemm`'s `DType::Oq8G256` arm already uses, hoisted here so the
    /// arch prefill sites can reach it without per-call scratch alloc/free.
    pub fn gemm_oq8_grouped_act_batched(
        &mut self,
        w_combined: &GpuTensor,
        x_rot: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        n: usize,
    ) -> HipResult<()> {
        self.quantize_act_oq8_batched(x_rot, m, k, n)?;
        self.gemm_oq8_grouped_prequant(w_combined, y, m, k, n)
    }

    /// Quantize the rotated activation into the shared oq8 batched scratch.
    /// Split out from [`Self::gemm_oq8_grouped_act_batched`] so a multi-projection
    /// site (qkvza, gate+up) can quantize ONCE and then issue one
    /// [`Self::gemm_oq8_grouped_prequant`] per projection — the oq8 analogue of
    /// what the fused oq4 kernels do internally.
    /// Quantize, then emit the fragment-interleaved nibble form alongside the
    /// int8 one. Called once per activation; every compact projection off that
    /// activation then reads the interleaved buffer with no in-kernel split.
    pub fn quantize_act_oq8_batched_interleaved(
        &mut self,
        x_rot: &GpuTensor,
        m_max: usize,
        k: usize,
        n: usize,
    ) -> HipResult<()> {
        self.quantize_act_oq8_batched(x_rot, m_max, k, n)?;
        let xq = GpuTensor {
            buf: unsafe { self.oq8_xq_batch.as_ref().unwrap().buf.alias() },
            shape: vec![n * k],
            dtype: DType::Raw,
        };
        let xilv = GpuTensor {
            buf: unsafe { self.oq_xilv_batch.as_ref().unwrap().buf.alias() },
            shape: vec![n * k],
            dtype: DType::Raw,
        };
        self.act_interleave_nibbles(&xq, &xilv, n, k)?;
        // W4A4: same activation, int4 grid. Produced here for the same reason as
        // the interleave -- it is a function of the ACTIVATION, so every compact
        // projection off it reuses one quantize.
        if oq_compact_a4() && k % 256 == 0 && oq4_act_group() == 256 {
            let x4 = GpuTensor {
                buf: unsafe { self.oq4_xq_batch.as_ref().unwrap().buf.alias() },
                shape: vec![n * k / 2],
                dtype: DType::Raw,
            };
            let s4 = GpuTensor {
                buf: unsafe { self.oq4_xs_batch.as_ref().unwrap().buf.alias() },
                shape: vec![n * (k / 256)],
                dtype: DType::F32,
            };
            self.quantize_act_oq4(x_rot, &x4, &s4, n, k, 256)?;
        }
        // Also produce the k-major twin the sparse overlay reads. Same argument
        // as the interleave above: it is a function of the ACTIVATION, so doing
        // it here rather than per projection stops gate/up/q/k/v each redoing it.
        if k % 256 != 0 {
            return Ok(());
        }
        let ng = k / 256;
        let xs = GpuTensor {
            buf: unsafe { self.oq8_xs_batch.as_ref().unwrap().buf.alias() },
            shape: vec![n * ng],
            dtype: DType::F32,
        };
        let xt = GpuTensor {
            buf: unsafe { self.oq_xt_batch.as_ref().unwrap().buf.alias() },
            shape: vec![n * k],
            dtype: DType::Raw,
        };
        let xst = GpuTensor {
            buf: unsafe { self.oq_xst_batch.as_ref().unwrap().buf.alias() },
            shape: vec![n * ng],
            dtype: DType::F32,
        };
        self.oq_compact_x8_transpose(&xq, &xs, &xt, &xst, n, k, ng)?;
        if std::env::var("HIPFIRE_HIPASS_STATS").as_deref() == Ok("1") {
            self.hipass_stats(&xq, n, k)?;
        }
        self.oq_xt_gen = self.oq_act_gen;
        self.oq_xt_ng = ng;
        self.oq_xt_n = n;
        Ok(())
    }

    pub fn quantize_act_oq8_batched(
        &mut self,
        x_rot: &GpuTensor,
        m_max: usize,
        k: usize,
        n: usize,
    ) -> HipResult<()> {
        // Any re-quantize invalidates the k-major twin above.
        self.oq_act_gen = self.oq_act_gen.wrapping_add(1);
        const GROUP: usize = 256;
        self.ensure_oq8_scratch_batched(n, k, m_max)?;
        let ng = k / GROUP;
        let xq = GpuTensor {
            buf: unsafe { self.oq8_xq_batch.as_ref().unwrap().buf.alias() },
            shape: vec![n * k],
            dtype: DType::Raw,
        };
        let xs = GpuTensor {
            buf: unsafe { self.oq8_xs_batch.as_ref().unwrap().buf.alias() },
            shape: vec![n * ng],
            dtype: DType::F32,
        };
        self.quantize_act_oq8(x_rot, &xq, &xs, n, k, GROUP)
    }

    /// One grouped int8-WMMA GEMM against the activation already quantized into
    /// the oq8 scratch by [`Self::quantize_act_oq8_batched`]. The caller must have
    /// quantized with the SAME `n`/`k` — the scratch is only grown, never shrunk,
    /// so a later call with a larger `m` cannot invalidate the quantized data.
    /// Compact-resident twin of [`Self::gemm_oq8_grouped_prequant`]: run the
    /// compact GEMM against an activation ALREADY quantized into the shared oq8
    /// batch scratch by `quantize_act_oq8_batched`. Lets several projections off
    /// one activation (gate+up, q/k/v) share a single quantize pass instead of
    /// re-quantizing per output the way `gemm_oq_compact_act_batched` would.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_oq_compact_grouped_prequant(
        &mut self,
        w_blocks: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        n: usize,
        block_stride: usize,
    ) -> HipResult<()> {
        const GROUP: usize = 256;
        self.ensure_oq8_scratch_batched(n, k, m)?;
        let ng = k / GROUP;
        let xq = GpuTensor {
            buf: unsafe { self.oq8_xq_batch.as_ref().unwrap().buf.alias() },
            shape: vec![n * k],
            dtype: DType::Raw,
        };
        let xs = GpuTensor {
            buf: unsafe { self.oq8_xs_batch.as_ref().unwrap().buf.alias() },
            shape: vec![n * ng],
            dtype: DType::F32,
        };
        self.compact_batched_route(w_blocks, &xq, &xs, y, m, k, n, ng, GROUP, block_stride)
    }

    pub fn gemm_oq8_grouped_prequant(
        &mut self,
        w_combined: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        n: usize,
    ) -> HipResult<()> {
        const GROUP: usize = 256;
        self.ensure_oq8_scratch_batched(n, k, m)?;
        let ng = k / GROUP;
        let xq = GpuTensor {
            buf: unsafe { self.oq8_xq_batch.as_ref().unwrap().buf.alias() },
            shape: vec![n * k],
            dtype: DType::Raw,
        };
        let xs = GpuTensor {
            buf: unsafe { self.oq8_xs_batch.as_ref().unwrap().buf.alias() },
            shape: vec![n * ng],
            dtype: DType::F32,
        };
        let ws = w_combined.sub_offset(m * k, m * ng * 4);
        self.gemm_oq8_grouped_wmma(w_combined, &ws, &xq, &xs, y, m, k, n, GROUP)
    }

    /// Batched W8A8 oq8 GEMM with residual add: `residual[N×M] += W·x_rot`.
    /// GEMMs into the persistent batched f32 scratch then one elementwise add —
    /// there is no fused oq8 residual kernel, mirroring the oq4 arm.
    pub fn gemm_oq8_grouped_residual_act_batched(
        &mut self,
        w_combined: &GpuTensor,
        x_rot: &GpuTensor,
        residual: &GpuTensor,
        m: usize,
        k: usize,
        n: usize,
    ) -> HipResult<()> {
        self.ensure_oq8_scratch_batched(n, k, m)?;
        let tmp = GpuTensor {
            buf: unsafe { self.oq8_ytmp_batch.as_ref().unwrap().buf.alias() },
            shape: vec![n * m],
            dtype: DType::F32,
        };
        self.gemm_oq8_grouped_act_batched(w_combined, x_rot, &tmp, m, k, n)?;
        let res_n = residual.sub_offset(0, n * m);
        self.add_inplace_f32(&res_n, &tmp)
    }

    /// Batched W4A4 oq4 GEMM with residual add: `residual[N×M] += W·x_rot`.
    /// GEMMs into the persistent batched f32 scratch (`oq4_ytmp_batch`, sized for
    /// M*N here) then a single elementwise add — there is no fused oq4 residual
    /// kernel, mirroring the decode `GemvResidual` Oq4 arm.
    pub fn gemm_oq4_grouped_residual_act_batched(
        &mut self,
        w_combined: &GpuTensor,
        x_rot: &GpuTensor,
        residual: &GpuTensor,
        m: usize,
        k: usize,
        n: usize,
    ) -> HipResult<()> {
        self.ensure_oq4_scratch_batched(n, k, m)?;
        let tmp = GpuTensor {
            buf: unsafe { self.oq4_ytmp_batch.as_ref().unwrap().buf.alias() },
            shape: vec![n * m],
            dtype: DType::F32,
        };
        self.gemm_oq4_grouped_act_batched(w_combined, x_rot, &tmp, m, k, n)?;
        let res_n = residual.sub_offset(0, n * m);
        self.add_inplace_f32(&res_n, &tmp)
    }

    /// W4A16 (act16) residual variant of [`Self::gemm_oq4_grouped_residual_act_batched`]:
    /// `residual[N×M] += W·x_rot` with the int4 weight dequantized to f16 against the
    /// full-precision (f16) activation — no activation quantization. Used only by the
    /// A4 KLD harness to build a same-batched-path W4A16 baseline for o_proj/down;
    /// production keeps the W4A4 residual path. `group` is fixed at 256 (oq4 codec).
    pub fn gemm_oq4_grouped_residual_f16_batched(
        &mut self,
        w_combined: &GpuTensor,
        x_rot: &GpuTensor,
        residual: &GpuTensor,
        m: usize,
        k: usize,
        n: usize,
    ) -> HipResult<()> {
        self.ensure_oq4_scratch_batched(n, k, m)?;
        let tmp = GpuTensor {
            buf: unsafe { self.oq4_ytmp_batch.as_ref().unwrap().buf.alias() },
            shape: vec![n * m],
            dtype: DType::F32,
        };
        self.gemm_oq4_grouped_f16_wmma(w_combined, x_rot, &tmp, m, k, n, 256)?;
        let res_n = residual.sub_offset(0, n * m);
        self.add_inplace_f32(&res_n, &tmp)
    }

    /// Generic kernel library: WMMA GEMM, signed INT8 inputs → INT32 output.
    /// `a_i8` [M,K], `x_i8` [B,K] (int8), `y_i32` [B,M] (int32).
    /// gfx1103/RDNA3 wave32, zero LDS. Requires `k % 16 == 0` and wave32 WMMA.
    pub fn gemm_iu8_i32_wmma(
        &mut self,
        a_i8: &GpuTensor,
        x_i8: &GpuTensor,
        y_i32: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert_eq!(k % 16, 0, "gemm_iu8_i32_wmma: K must be a multiple of 16");
        self.ensure_kernel(
            "gemm_iu8_i32_wmma",
            kernels::GEMM_IU8_I32_WMMA_SRC,
            "gemm_iu8_i32_wmma",
        )?;
        let ap = a_i8.buf.as_ptr();
        let xp = x_i8.buf.as_ptr();
        let yp = y_i32.buf.as_ptr();
        let mut mi = m as i32;
        let mut ki = k as i32;
        let mut bi = batch_size as i32;
        let mut params: Vec<*mut c_void> = vec![
            &ap as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &yp as *const _ as *mut c_void,
            &mut mi as *mut _ as *mut c_void,
            &mut ki as *mut _ as *mut c_void,
            &mut bi as *mut _ as *mut c_void,
        ];
        let grid_m = ((m + 15) / 16) as u32;
        let grid_b = ((batch_size + 15) / 16) as u32;
        let func = &self.functions["gemm_iu8_i32_wmma"];
        unsafe {
            self.hip.launch_kernel(
                func,
                [grid_m, grid_b, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// Reference kernel layer: per-token symmetric int8 activation quant.
    /// `x_f32` [B,K] → `xq_i8` [B,K] int8 + `xs_f32` [B] per-row scale. Zero LDS.
    pub fn quantize_act_int8_per_token(
        &mut self,
        x_f32: &GpuTensor,
        xq_i8: &GpuTensor,
        xs_f32: &GpuTensor,
        batch_size: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "quantize_act_int8_per_token",
            kernels::QUANTIZE_ACT_INT8_PER_TOKEN_SRC,
            "quantize_act_int8_per_token",
        )?;
        let xp = x_f32.buf.as_ptr();
        let qp = xq_i8.buf.as_ptr();
        let sp = xs_f32.buf.as_ptr();
        let mut bi = batch_size as i32;
        let mut ki = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &xp as *const _ as *mut c_void,
            &qp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &mut bi as *mut _ as *mut c_void,
            &mut ki as *mut _ as *mut c_void,
        ];
        let func = &self.functions["quantize_act_int8_per_token"];
        unsafe {
            self.hip.launch_kernel(
                func,
                [batch_size as u32, 1, 1],
                [32, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// Reference kernel layer: int32 → f32 dequant by per-row × per-col scales.
    /// `y_i32` [B,M] · `x_scale` [B] · `w_scale` [M] → `y_f32` [B,M].
    pub fn dequant_i32_rowcol(
        &mut self,
        y_i32: &GpuTensor,
        x_scale: &GpuTensor,
        w_scale: &GpuTensor,
        y_f32: &GpuTensor,
        batch_size: usize,
        m: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "dequant_i32_rowcol",
            kernels::DEQUANT_I32_ROWCOL_SRC,
            "dequant_i32_rowcol",
        )?;
        let yp = y_i32.buf.as_ptr();
        let xsp = x_scale.buf.as_ptr();
        let wsp = w_scale.buf.as_ptr();
        let op = y_f32.buf.as_ptr();
        let mut bi = batch_size as i32;
        let mut mi = m as i32;
        let mut params: Vec<*mut c_void> = vec![
            &yp as *const _ as *mut c_void,
            &xsp as *const _ as *mut c_void,
            &wsp as *const _ as *mut c_void,
            &op as *const _ as *mut c_void,
            &mut bi as *mut _ as *mut c_void,
            &mut mi as *mut _ as *mut c_void,
        ];
        let n = batch_size * m;
        let grid = ((n + 255) / 256) as u32;
        let func = &self.functions["dequant_i32_rowcol"];
        unsafe {
            self.hip.launch_kernel(
                func,
                [grid, 1, 1],
                [256, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
}

fn dflash_plain_kernel(dtype: DType, aligned: bool) -> (&'static str, Option<usize>) {
    match (dtype, aligned) {
        (DType::DflashOq8Plain, true) => ("gemm_dflash_oq8_plain_dp4a_staged_8w", Some(258)),
        (DType::DflashOq4Plain, true) => ("gemm_dflash_oq4_plain_dp4a_staged_8w", Some(130)),
        (DType::DflashOq4MixedPlain, true) => ("gemm_dflash_oq4_mixed_plain_dp4a_staged_8w", None),
        (DType::DflashOq8Plain, false) => ("gemm_dflash_oq8_plain_8w", Some(258)),
        (DType::DflashOq4Plain, false) => ("gemm_dflash_oq4_plain_8w", Some(130)),
        (DType::DflashOq4MixedPlain, false) => ("gemm_dflash_oq4_mixed_plain_8w", None),
        other => panic!("gemm_dflash_oq_plain: unsupported dtype {other:?}"),
    }
}

/// Multi-column drafter GEMV name for an exact column count, when one exists.
///
/// The per-column kernels put the batch on `blockIdx.y`, so the drafter's whole
/// weight set is re-read B times — at B=8 that is 9.4 GiB per spec cycle for a
/// 1.18 GiB drafter, ~40% of the cycle's bytes, and it is what makes the draft
/// phase scale linearly with B (~6.3 ms per position = one weight sweep each).
/// These read the row ONCE and accumulate all B columns.
///
/// NB is compile-time in the kernel so the accumulators stay in registers, so
/// only exact matches route here; everything else keeps the per-column kernel.
fn dflash_plain_multicol_kernel(dtype: DType, batch: usize) -> Option<&'static str> {
    if !(2..=8).contains(&batch) {
        return None;
    }
    if std::env::var("HIPFIRE_DFLASH_MULTICOL").as_deref() == Ok("0") {
        return None;
    }
    Some(match (dtype, batch) {
        (DType::DflashOq8Plain, 2) => "gemm_dflash_oq8_plain_multicol_w2",
        (DType::DflashOq8Plain, 3) => "gemm_dflash_oq8_plain_multicol_w3",
        (DType::DflashOq8Plain, 4) => "gemm_dflash_oq8_plain_multicol_w4",
        (DType::DflashOq8Plain, 5) => "gemm_dflash_oq8_plain_multicol_w5",
        (DType::DflashOq8Plain, 6) => "gemm_dflash_oq8_plain_multicol_w6",
        (DType::DflashOq8Plain, 7) => "gemm_dflash_oq8_plain_multicol_w7",
        (DType::DflashOq8Plain, 8) => "gemm_dflash_oq8_plain_multicol_w8",
        (DType::DflashOq4Plain, 2) => "gemm_dflash_oq4_plain_multicol_w2",
        (DType::DflashOq4Plain, 3) => "gemm_dflash_oq4_plain_multicol_w3",
        (DType::DflashOq4Plain, 4) => "gemm_dflash_oq4_plain_multicol_w4",
        (DType::DflashOq4Plain, 5) => "gemm_dflash_oq4_plain_multicol_w5",
        (DType::DflashOq4Plain, 6) => "gemm_dflash_oq4_plain_multicol_w6",
        (DType::DflashOq4Plain, 7) => "gemm_dflash_oq4_plain_multicol_w7",
        (DType::DflashOq4Plain, 8) => "gemm_dflash_oq4_plain_multicol_w8",
        (DType::DflashOq4MixedPlain, 2) => "gemm_dflash_oq4_mixed_plain_multicol_w2",
        (DType::DflashOq4MixedPlain, 3) => "gemm_dflash_oq4_mixed_plain_multicol_w3",
        (DType::DflashOq4MixedPlain, 4) => "gemm_dflash_oq4_mixed_plain_multicol_w4",
        (DType::DflashOq4MixedPlain, 5) => "gemm_dflash_oq4_mixed_plain_multicol_w5",
        (DType::DflashOq4MixedPlain, 6) => "gemm_dflash_oq4_mixed_plain_multicol_w6",
        (DType::DflashOq4MixedPlain, 7) => "gemm_dflash_oq4_mixed_plain_multicol_w7",
        (DType::DflashOq4MixedPlain, 8) => "gemm_dflash_oq4_mixed_plain_multicol_w8",
        _ => return None,
    })
}

fn validate_dflash_plain_stride(
    kernel_name: &str,
    block_stride: usize,
    expected_stride: Option<usize>,
) {
    if let Some(expected) = expected_stride {
        assert_eq!(
            block_stride, expected,
            "{kernel_name}: invalid block stride"
        );
    } else {
        assert!(
            block_stride >= 132 && (block_stride - 130) % 2 == 0,
            "{kernel_name}: mixed block stride must be 130 + 2*N"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dflash_plain_aligned_routes_all_formats_through_staged_a8() {
        for (dtype, expected, stride) in [
            (
                DType::DflashOq8Plain,
                "gemm_dflash_oq8_plain_dp4a_staged_8w",
                Some(258),
            ),
            (
                DType::DflashOq4Plain,
                "gemm_dflash_oq4_plain_dp4a_staged_8w",
                Some(130),
            ),
            (
                DType::DflashOq4MixedPlain,
                "gemm_dflash_oq4_mixed_plain_dp4a_staged_8w",
                None,
            ),
        ] {
            assert_eq!(dflash_plain_kernel(dtype, true), (expected, stride));
        }
    }

    #[test]
    fn dflash_plain_ragged_keeps_exact_f16_activation_path() {
        for (dtype, expected, stride) in [
            (DType::DflashOq8Plain, "gemm_dflash_oq8_plain_8w", Some(258)),
            (DType::DflashOq4Plain, "gemm_dflash_oq4_plain_8w", Some(130)),
            (
                DType::DflashOq4MixedPlain,
                "gemm_dflash_oq4_mixed_plain_8w",
                None,
            ),
        ] {
            assert_eq!(dflash_plain_kernel(dtype, false), (expected, stride));
        }
    }
}

/// W4A4 opt-in. Weights stay compact 4.25-bit -- only the ACTIVATION narrows to
/// int4, which turns the exact radix-16 pair (x = 16*x_hi + x_lo, two iu4 WMMA
/// passes) into a single iu4 pass. The bits/weight floor is untouched because
/// nothing about the weight encoding changes. Kernel-level 1.66x at the 27B
/// prefill shapes (gate/up B=512: 2.87 ms W4A8 -> 1.733 ms W4A4).
/// Both call sites additionally require `oq4_act_group() == 256`: the compact
/// GEMM's fold indexes the activation scale by WEIGHT group, so a finer
/// activation group would silently misalign the two.
fn oq_compact_a4() -> bool {
    std::env::var("HIPFIRE_OQ_COMPACT_A4").as_deref() == Ok("1")
}
