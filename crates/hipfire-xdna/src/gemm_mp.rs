//! `NpuGemmMp` — the M-parallel W-broadcast W4A8/W8A8 GEMM primitive (productionized
//! `r6_gen_mp.py` array + `r6_gemm_ts.cc` tensor-stream kernel). This is the best
//! runtime-callable NPU GEMM path: one xclbin handles any M, weights are packed and loaded
//! ONCE and broadcast to all cores, and A/C move ROW-MAJOR (the kernel's tensor streams
//! tile in-core — no CPU marshaling). Each dispatch computes `COLS` distinct M-blocks over
//! full N; [`Self::run`] tiles M over blocking dispatches (reliable — no pipelined-readback
//! coherence hazard). Measured ~1.45 TOPS e2e on halo, flat across batch (weight-bandwidth-
//! bound). See benchmarks/npu_gemm_tuning/r6/README.md for the topology + ceiling analysis.
//!
//! Shape contract: `K == k()` (single K-chunk), `N == n()`, `M % rows_per_dispatch() == 0`.
//! Linux-only (amdxdna).
#![cfg(target_os = "linux")]

use crate::{DeviceBuffer, NpuKernel, XdnaError};

const MR: usize = 4; // mmul M
const MK: usize = 16; // mmul K
const MN: usize = 16; // mmul N
const NT: usize = 4; // N-blocks per slab (r6_gemm_ts.cc accumulator count)

/// A loaded M-parallel W-broadcast R6 kernel specialized for (COLS, MT, KCHUNK, NB), with
/// reusable arg buffers and the broadcast weights resident.
pub struct NpuGemmMp {
    kernel: NpuKernel,
    cols: usize,
    mt: usize,
    kchunk: usize,
    nb: usize,
    mr: usize,
    mk: usize,
    mn: usize,
    weight_bits: usize,
    a_buf: DeviceBuffer, // COLS M-blocks (row-major), one per core, per dispatch
    w_buf: DeviceBuffer, // NB broadcast weight slabs (tile-major W4/W8), loaded once
    c_buf: DeviceBuffer, // COLS*NB output blocks (row-major)
    w_loaded: bool,
}

impl NpuGemmMp {
    /// M rows computed per dispatch (`COLS` M-blocks of `MT·MR` rows).
    pub fn rows_per_dispatch(&self) -> usize {
        self.cols * self.mt * self.mr
    }
    /// N this kernel computes (full N in one dispatch): `NB·NT·MN`.
    pub fn n(&self) -> usize {
        self.nb * NT * self.mn
    }
    /// K contracted per dispatch (single chunk): `KCHUNK·MK`.
    pub fn k(&self) -> usize {
        self.kchunk * self.mk
    }
    /// Weight precision this xclbin consumes: 4 for packed signed int4, 8 for signed int8.
    pub fn weight_bits(&self) -> usize {
        self.weight_bits
    }

    fn aw(&self) -> usize {
        self.mt * self.kchunk * self.mr * self.mk
    }
    fn ww(&self) -> usize {
        let elems = NT * self.kchunk * self.mk * self.mn;
        match self.weight_bits {
            4 => elems / 2,
            8 => elems,
            bits => panic!("unsupported NpuGemmMp weight bits: {bits}"),
        }
    }
    fn cw(&self) -> usize {
        self.mt * NT * self.mr * self.mn
    }

    /// Load an M-parallel xclbin built with `r6_gen_mp.py` (COLS cores, ROUNDS=1) for
    /// (mt, kchunk, nb) and allocate its arg buffers. Call [`Self::load_weights`] before
    /// [`Self::run`].
    pub fn load(
        xclbin: &[u8],
        insts: &[u8],
        cols: usize,
        mt: usize,
        kchunk: usize,
        nb: usize,
    ) -> Result<Self, XdnaError> {
        Self::load_with_tile(xclbin, insts, cols, mt, kchunk, nb, 4, MR, MK, MN)
    }

    /// Load an M-parallel xclbin with an explicit weight width. `weight_bits=4` matches
    /// the original W4A8 R6 kernel; `weight_bits=8` matches the W8A8 tensor-stream kernel.
    pub fn load_with_weight_bits(
        xclbin: &[u8],
        insts: &[u8],
        cols: usize,
        mt: usize,
        kchunk: usize,
        nb: usize,
        weight_bits: usize,
    ) -> Result<Self, XdnaError> {
        Self::load_with_tile(xclbin, insts, cols, mt, kchunk, nb, weight_bits, MR, MK, MN)
    }

    fn load_with_tile(
        xclbin: &[u8],
        insts: &[u8],
        cols: usize,
        mt: usize,
        kchunk: usize,
        nb: usize,
        weight_bits: usize,
        mr: usize,
        mk: usize,
        mn: usize,
    ) -> Result<Self, XdnaError> {
        assert!(
            weight_bits == 4 || weight_bits == 8,
            "NpuGemmMp weight_bits must be 4 or 8"
        );
        let kernel = NpuKernel::load(xclbin, insts)?;
        let aw = mt * kchunk * mr * mk;
        let ww = match weight_bits {
            4 => NT * kchunk * mk * mn / 2,
            8 => NT * kchunk * mk * mn,
            _ => unreachable!(),
        };
        let cw = mt * NT * mr * mn;
        let a_buf = kernel.alloc_arg(cols * aw)?;
        let w_buf = kernel.alloc_arg(nb * ww)?;
        let c_buf = kernel.alloc_arg(cols * nb * cw * 4)?;
        Ok(Self {
            kernel,
            cols,
            mt,
            kchunk,
            nb,
            mr,
            mk,
            mn,
            weight_bits,
            a_buf,
            w_buf,
            c_buf,
            w_loaded: false,
        })
    }

    /// Load from a standard `r6_cache.sh` cache dir, parsing (COLS, MT, KCHUNK, NB) from its
    /// name (`..._{MT}x{NT}x{KCHUNK}_c{COLS}_nb{NB}`) so the config can't silently mismatch
    /// the xclbin. A `_w8` token selects the W8A8 variant; otherwise W4A8 is assumed.
    /// Rejects whole-GEMM `_r{ROUNDS}` builds (different layout) and any NT≠4.
    pub fn load_cached(dir: &str) -> Result<Self, XdnaError> {
        let xclbin = std::fs::read(format!("{dir}/final.xclbin")).map_err(XdnaError::Open)?;
        let insts = std::fs::read(format!("{dir}/insts.bin")).map_err(XdnaError::Open)?;
        let base = std::path::Path::new(dir)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("");
        let toks: Vec<&str> = base.split('_').collect();
        let bad = || XdnaError::BadCacheName(base.to_string());
        // A `_r{N}` token means a whole-GEMM (ROUNDS) build — not this per-dispatch primitive.
        if toks.iter().any(|t| {
            t.strip_prefix('r')
                .is_some_and(|r| !r.is_empty() && r.bytes().all(|b| b.is_ascii_digit()))
        }) {
            return Err(bad());
        }
        let pfx = |p: &str| {
            toks.iter()
                .find_map(|t| t.strip_prefix(p).and_then(|r| r.parse().ok()))
        };
        let nb: usize = pfx("nb").ok_or_else(bad)?;
        let cols: usize = pfx("c").ok_or_else(bad)?;
        let dims = toks
            .iter()
            .find(|t| t.split('x').count() == 3)
            .ok_or_else(bad)?;
        let d: Vec<usize> = dims.split('x').filter_map(|s| s.parse().ok()).collect();
        if d.len() != 3 || d[1] != NT {
            return Err(bad());
        }
        let weight_bits = if toks.contains(&"w8") { 8 } else { 4 };
        let (mr, mk, mn) = if toks.contains(&"m8k8") {
            (8, 8, 16)
        } else {
            (MR, MK, MN)
        };
        Self::load_with_tile(
            &xclbin,
            &insts,
            cols,
            d[0],
            d[2],
            nb,
            weight_bits,
            mr,
            mk,
            mn,
        )
    }

    /// Pack a full `K×N` weight matrix into the broadcast slab layout. For W4 xclbins,
    /// values must fit `-8..=7` and are packed as two signed nibbles per byte. For W8
    /// xclbins, values are copied as signed int8. This slow packing is an offline/load-time
    /// step; weights are static once loaded.
    pub fn prepack_weights(&self, k: usize, n: usize, weights: &[i8]) -> Vec<u8> {
        assert_eq!(k, self.k(), "K");
        assert_eq!(n, self.n(), "N");
        assert_eq!(weights.len(), k * n, "weight element count");
        let (kc, nb, ww) = (self.kchunk, self.nb, self.ww());
        let mut out = vec![0u8; nb * ww];
        match self.weight_bits {
            4 => {
                assert_eq!((self.mr, self.mk, self.mn), (MR, MK, MN));
                for j in 0..nb {
                    for nt in 0..NT {
                        for ki in 0..kc {
                            for kk in 0..self.mk {
                                for nn in 0..self.mn {
                                    let kg = ki * self.mk + kk;
                                    let ng = j * NT * self.mn + nt * self.mn + nn;
                                    let idx =
                                        (nt * kc + ki) * (self.mk * self.mn) + kk * self.mn + nn;
                                    let w = weights[kg * n + ng];
                                    assert!((-8..=7).contains(&w), "W4 value {w} outside -8..=7");
                                    let u = (w & 0xf) as u8;
                                    out[j * ww + idx / 2] |= if idx % 2 == 0 { u } else { u << 4 };
                                }
                            }
                        }
                    }
                }
            }
            8 => {
                for j in 0..nb {
                    for nt in 0..NT {
                        for ki in 0..kc {
                            if self.mk == 16 {
                                for k_half in 0..2 {
                                    for n_half in 0..2 {
                                        for kk in 0..8 {
                                            for nn in 0..8 {
                                                let kg = ki * self.mk + k_half * 8 + kk;
                                                let ng = j * NT * self.mn
                                                    + nt * self.mn
                                                    + n_half * 8
                                                    + nn;
                                                let idx =
                                                    ((nt * kc + ki) * 4 + k_half * 2 + n_half) * 64
                                                        + kk * 8
                                                        + nn;
                                                out[j * ww + idx] = weights[kg * n + ng] as u8;
                                            }
                                        }
                                    }
                                }
                            } else {
                                assert_eq!((self.mr, self.mk, self.mn), (8, 8, 16));
                                for n_half in 0..2 {
                                    for kk in 0..8 {
                                        for nn in 0..8 {
                                            let kg = ki * self.mk + kk;
                                            let ng =
                                                j * NT * self.mn + nt * self.mn + n_half * 8 + nn;
                                            let idx =
                                                ((nt * kc + ki) * 2 + n_half) * 64 + kk * 8 + nn;
                                            out[j * ww + idx] = weights[kg * n + ng] as u8;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            bits => panic!("unsupported NpuGemmMp weight bits: {bits}"),
        }
        out
    }

    /// Load packed weights (from [`Self::prepack_weights`]) into the resident broadcast
    /// buffer once; every [`Self::run`] dispatch reuses them (fanned to all cores).
    pub fn load_weights(&mut self, packed_w: &[u8]) {
        self.w_buf.as_mut_slice().copy_from_slice(packed_w);
        self.w_loaded = true;
    }

    /// Full GEMM `C[M,N] = A[M,K] · W[K,N]` (W4A8/W8A8), tiling M over blocking dispatches. `a`
    /// row-major `M×K` int8, `c` row-major `M×N` int32. Requires `load_weights` first,
    /// `K == k()`, `N == n()`, and `M % rows_per_dispatch() == 0`.
    pub fn run(
        &mut self,
        m: usize,
        k: usize,
        n: usize,
        a: &[i8],
        c: &mut [i32],
    ) -> Result<(), XdnaError> {
        assert!(self.w_loaded, "call load_weights() before run()");
        assert_eq!(k, self.k(), "K");
        assert_eq!(n, self.n(), "N");
        let rows_per = self.rows_per_dispatch();
        assert!(m % rows_per == 0, "M must be a multiple of {rows_per}");
        let (cols, mt, aw) = (self.cols, self.mt, self.aw());
        for d in 0..(m / rows_per) {
            let row0 = d * rows_per;
            self.copy_a_tile(row0, k, a, cols, mt, aw);
            self.kernel
                .dispatch(&[&self.a_buf, &self.w_buf, &self.c_buf])?;
            self.read_c_tile(row0, n, c); // de-block c_buf -> rows [row0,+) of row-major c
        }
        Ok(())
    }

    fn copy_a_tile(&mut self, row0: usize, k: usize, a: &[i8], cols: usize, mt: usize, aw: usize) {
        let s = self.a_buf.as_mut_slice();
        match self.weight_bits {
            4 => {
                // COLS row-major M-blocks -> a_buf (the W4 kernel's A tensor stream tiles in-core).
                for ci in 0..cols {
                    for lr in 0..mt * self.mr {
                        let src = (row0 + ci * mt * self.mr + lr) * k;
                        for kk in 0..k {
                            s[ci * aw + lr * k + kk] = a[src + kk] as u8;
                        }
                    }
                }
            }
            8 => {
                // Dense AIE2P W8 stores each A tile as contiguous 8-wide K halves.
                for ci in 0..cols {
                    for mi in 0..mt {
                        for ki in 0..self.kchunk {
                            let k_halves = self.mk / 8;
                            for half in 0..k_halves {
                                for r in 0..self.mr {
                                    let src_row = row0 + ci * mt * self.mr + mi * self.mr + r;
                                    for kk in 0..8 {
                                        let src = src_row * k + ki * self.mk + half * 8 + kk;
                                        let dst = ci * aw
                                            + (mi * self.kchunk + ki) * self.mr * self.mk
                                            + half * self.mr * 8
                                            + r * 8
                                            + kk;
                                        s[dst] = a[src] as u8;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            bits => panic!("unsupported NpuGemmMp weight bits: {bits}"),
        }
    }

    // De-block the current c_buf (COLS*NB blocks, each (MT·MR)×(NT·MN) row-major) into rows
    // [row0, row0+rows_per_dispatch()) of a row-major `c` `M×N`. This host copy is exactly
    // what zero-copy avoids — a GPU consumer reads the block layout from the shared buffer
    // directly (see `run_into_shared` + `c_block_offset`).
    fn read_c_tile(&self, row0: usize, n: usize, c: &mut [i32]) {
        let (cols, mt, nb, cw) = (self.cols, self.mt, self.nb, self.cw());
        let out: &[i32] = unsafe {
            std::slice::from_raw_parts(self.c_buf.as_slice().as_ptr() as *const i32, cols * nb * cw)
        };
        for ci in 0..cols {
            for j in 0..nb {
                for lr in 0..mt * self.mr {
                    let base = (ci * nb + j) * cw + lr * (NT * self.mn);
                    let dst = (row0 + ci * mt * self.mr + lr) * n + j * NT * self.mn;
                    c[dst..dst + NT * self.mn].copy_from_slice(&out[base..base + NT * self.mn]);
                }
            }
        }
    }

    /// Byte size the output buffer must be for one dispatch's C: `COLS·NB·(MT·NT·MR·MN)·4`.
    pub fn c_buf_bytes(&self) -> usize {
        self.cols * self.nb * self.cw() * 4
    }

    /// Replace the SHMEM output buffer with an imported GPU dma-buf (zero-copy). After this,
    /// [`Self::run_into_shared`] writes C straight into the GPU-shared pages — no host copy.
    /// `size` must be [`Self::c_buf_bytes`] (one dispatch's C). The dma-buf is typically an
    /// amdgpu GTT BO exported via `PRIME_HANDLE_TO_FD`; the driver `dma_buf_get`s the fd.
    pub fn attach_output_dmabuf(&mut self, fd: i32, size: usize) -> Result<(), XdnaError> {
        assert_eq!(size, self.c_buf_bytes(), "output dma-buf size");
        self.c_buf = self.kernel.import_dmabuf(fd, size, true)?;
        Ok(())
    }

    /// Byte offset (into the output buffer) of the C block for (`core`, `slab`): a
    /// (MT·MR)×(NT·MN) row-major int32 tile covering global rows
    /// [`core*MT*MR`, +), cols [`slab*NT*MN`, +) of this dispatch's M-tile. Lets a GPU
    /// consumer index the block-layout C in the shared buffer directly.
    pub fn c_block_offset_i32(&self, core: usize, slab: usize) -> usize {
        (core * self.nb + slab) * self.cw()
    }

    /// Run ONE M-block (`a` = `rows_per_dispatch()×K` row-major int8) with C written directly
    /// into the attached output dma-buf — **no host readback**. The result lands in the
    /// GPU-shared pages in the NPU block layout (see [`Self::c_block_offset_i32`]); the GPU
    /// reads it with zero host involvement. Requires [`Self::attach_output_dmabuf`] +
    /// [`Self::load_weights`]. For full M, drive this per M-tile and consume between calls
    /// (the single output buffer is reused each dispatch).
    pub fn run_into_shared(&mut self, k: usize, n: usize, a: &[i8]) -> Result<(), XdnaError> {
        assert!(
            self.w_loaded,
            "call load_weights() before run_into_shared()"
        );
        assert_eq!(k, self.k(), "K");
        assert_eq!(n, self.n(), "N");
        let rows_per = self.rows_per_dispatch();
        assert_eq!(a.len(), rows_per * k, "A must be exactly one M-block");
        let (cols, mt, aw) = (self.cols, self.mt, self.aw());
        self.copy_a_tile(0, k, a, cols, mt, aw);
        self.kernel
            .dispatch(&[&self.a_buf, &self.w_buf, &self.c_buf])?;
        Ok(()) // C is now in the shared dma-buf; no host copy
    }

    /// De-block the shared/output buffer's current C into a row-major `rows_per × N` host
    /// buffer — for validation or a host (non-GPU) consumer of [`Self::run_into_shared`].
    pub fn read_shared_rowmajor(&self, n: usize, c: &mut [i32]) {
        self.read_c_tile(0, n, c);
    }

    /// Byte size the input buffer must be for one dispatch's A: `rows_per_dispatch()·K`
    /// int8. The `a_buf` block layout is *exactly* row-major `A[rows_per][K]` (the COLS
    /// M-blocks are contiguous M-rows), so a producer writes A with no reshuffle.
    pub fn a_buf_bytes(&self) -> usize {
        self.rows_per_dispatch() * self.k()
    }

    /// Replace the SHMEM input buffer with an imported GPU dma-buf (zero-copy input). After
    /// this, a producer (the GPU, or the CPU on this UMA APU) writes one M-block of A —
    /// row-major `A[rows_per][K]` int8 — into the shared pages, and [`Self::run_shared`]
    /// dispatches with no host A-copy. `size` must be [`Self::a_buf_bytes`].
    pub fn attach_input_dmabuf(&mut self, fd: i32, size: usize) -> Result<(), XdnaError> {
        assert_eq!(size, self.a_buf_bytes(), "input dma-buf size");
        self.a_buf = self.kernel.import_dmabuf(fd, size, true)?;
        Ok(())
    }

    /// Fully zero-copy dispatch of ONE M-block: A is already in the attached input dma-buf
    /// (written by the producer) and C goes to the attached output dma-buf — **no host
    /// copies at all**. Requires [`Self::attach_input_dmabuf`] + [`Self::attach_output_dmabuf`]
    /// + [`Self::load_weights`]. (`submit` still flushes the arg BOs, which reconciles the
    /// shared pages across engines.)
    pub fn run_shared(&mut self, k: usize, n: usize) -> Result<(), XdnaError> {
        assert!(self.w_loaded, "call load_weights() before run_shared()");
        assert_eq!(k, self.k(), "K");
        assert_eq!(n, self.n(), "N");
        self.kernel
            .dispatch(&[&self.a_buf, &self.w_buf, &self.c_buf])
    }

    /// Host view of the input dma-buf as one row-major `rows_per × K` int8 M-block — for a
    /// CPU producer / validation to fill A directly on this UMA APU.
    pub fn input_slice_mut(&mut self) -> &mut [i8] {
        let s = self.a_buf.as_mut_slice();
        unsafe { std::slice::from_raw_parts_mut(s.as_mut_ptr() as *mut i8, s.len()) }
    }
}
