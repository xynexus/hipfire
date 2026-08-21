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
            let gemm_func = &self.functions[kernel_name];
            unsafe {
                self.hip.launch_kernel(
                    gemm_func,
                    [m.div_ceil(8) as u32, n as u32, 1],
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
        self.quantize_act_oq8_batched(x_rot, m, k, n)?;
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
        self.gemm_oq_compact_grouped_wmma(w_blocks, &xq, &xs, y, m, k, n, GROUP, block_stride)
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
        assert!(
            batch_size <= 16,
            "gemv_oq_compact_multicol: B must be <= 16"
        );
        self.ensure_kernel(
            "gemv_oq_compact_multicol",
            kernels::GEMV_OQ_COMPACT_MULTICOL_SRC,
            "gemv_oq_compact_multicol",
        )?;
        let (wp, xp, xsp, yp) = (
            w_blocks.buf.as_ptr(),
            x_i8.buf.as_ptr(),
            x_scales.buf.as_ptr(),
            y_f32.buf.as_ptr(),
        );
        let (mi, ki, bi, bs) = (m as i32, k as i32, batch_size as i32, block_stride as i32);
        let grid = ((m as u32).div_ceil(8)).clamp(1, 2048);
        self.launch_kernargs(
            "gemv_oq_compact_multicol",
            [grid, 1, 1],
            [256, 1, 1],
            0,
            &kernargs![ptr wp, ptr xp, ptr xsp, ptr yp, i32 mi, i32 ki, i32 bi, i32 bs],
        )
    }

    #[allow(clippy::too_many_arguments)]
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
        x_hi: &GpuTensor,
        x_lo: &GpuTensor,
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
        let xhp = x_hi.buf.as_ptr();
        let xlp = x_lo.buf.as_ptr();
        let xsp = x_scales.buf.as_ptr();
        let yp = y_f32.buf.as_ptr();
        let (mut mi, mut ki, mut bi) = (m as i32, k as i32, batch_size as i32);
        let (mut gi, mut si) = (group as i32, block_stride as i32);
        let mut params: Vec<*mut c_void> = vec![
            &wp as *const _ as *mut c_void,
            &xhp as *const _ as *mut c_void,
            &xlp as *const _ as *mut c_void,
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
    pub fn quantize_act_oq8_batched(
        &mut self,
        x_rot: &GpuTensor,
        m_max: usize,
        k: usize,
        n: usize,
    ) -> HipResult<()> {
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
        self.gemm_oq_compact_grouped_wmma(w_blocks, &xq, &xs, y, m, k, n, GROUP, block_stride)
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
