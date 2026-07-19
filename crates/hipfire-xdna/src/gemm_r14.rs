//! `NpuGemmR14` — the 4x4 whole-array broadcast W4A8 GEMM (npu1/aie2).
//!
//! Drives the `r14_gen.py` array: output C is a 4x4 grid of `LM x LN` base-tile
//! blocks, block (i, j) computed by core (col = j, row = 2 + i). Column j's shim
//! feeds W-stripe j (in-column broadcast); block-row i's shim feeds A-stripe i
//! (cross-column broadcast); each column joins its 4 cores' C back out its shim.
//!
//! One dispatch runs `NBLK` **independent** block-iterations. Each iteration is a
//! complete little GEMM of shape
//!
//! ```text
//!   M_TILE = 4 * LM * 4        rows   (4 A-stripes x LM base-tiles x mmul M=4)
//!   N_TILE = 4 * LN * 8        cols   (4 W-stripes x LN base-tiles x mmul N=8)
//!   K_CHUNK = KT * 16                  (KT base-tiles x mmul K=16)
//! ```
//!
//! so a caller covers a full `M x K x N` GEMM by enumerating
//! `(m_block, k_chunk, n_tile)` triples into iteration slots and summing the
//! per-k-chunk int32 partials on the host. The kernel does no scaling, so the
//! host owns dequant — which means the weight group size is pinned to `K_CHUNK`.
//!
//! Base-tile layouts follow `r11_gemm.cc` (`aie::mmul<4, 16, 8, int8, int4>`):
//! A tile = row-major 4x16 int8, W tile = row-major 16x8 int4 (2 per byte, low
//! nibble first), C tile = row-major 4x8 int32. Stripes are tile-major:
//! A `[LM][KT]`, W `[LN][KT]`, C `[LM][LN]`.
//!
//! Linux-only (amdxdna).
#![cfg(target_os = "linux")]

use crate::{DeviceBuffer, NpuKernel, XdnaError};

/// mmul M (rows per A base-tile).
pub const MR: usize = 4;
/// mmul K (contraction per base-tile).
pub const MK: usize = 16;
/// mmul N (cols per W base-tile).
pub const MN: usize = 8;
/// Core-array edge: 4 columns x 4 block-rows.
pub const GRID: usize = 4;

/// Shape of one r14 build, parsed from its cache-dir name.
#[derive(Clone, Copy, Debug)]
pub struct R14Geometry {
    pub lm: usize,
    pub ln: usize,
    pub kt: usize,
    pub nblk: usize,
}

impl R14Geometry {
    /// Rows one iteration covers.
    pub fn m_tile(&self) -> usize {
        GRID * self.lm * MR
    }
    /// Output columns one iteration covers.
    pub fn n_tile(&self) -> usize {
        GRID * self.ln * MN
    }
    /// K contracted by one iteration.
    pub fn k_chunk(&self) -> usize {
        self.kt * MK
    }
    /// A-stripe bytes (one block-row, one iteration).
    pub fn ab(&self) -> usize {
        self.lm * self.kt * MR * MK
    }
    /// W-stripe bytes (one column, one iteration) — int4, 2 per byte.
    pub fn wb(&self) -> usize {
        self.ln * self.kt * MK * MN / 2
    }
    /// Per-core C int32 count.
    pub fn cb(&self) -> usize {
        self.lm * self.ln * MR * MN
    }
    /// Per-column A region bytes (all iterations).
    pub fn at(&self) -> usize {
        self.nblk * self.ab()
    }
    /// Per-column W region bytes (all iterations).
    pub fn wt(&self) -> usize {
        self.nblk * self.wb()
    }
    /// Per-column C region int32 count (all iterations).
    pub fn ct(&self) -> usize {
        self.nblk * GRID * self.cb()
    }
    pub fn a_bytes(&self) -> usize {
        GRID * self.at()
    }
    pub fn w_bytes(&self) -> usize {
        GRID * self.wt()
    }
    pub fn c_bytes(&self) -> usize {
        GRID * self.ct() * 4
    }

    /// Parse `r14_{LM}x{LN}x{KT}_nb{NBLK}` (the `r14_cache.sh` dir name).
    pub fn from_dir_name(base: &str) -> Option<Self> {
        let mut lm = None;
        let mut nblk = None;
        let (mut ln, mut kt) = (0, 0);
        for tok in base.split('_') {
            if let Some(rest) = tok.strip_prefix("nb") {
                nblk = rest.parse().ok();
            } else if tok.split('x').count() == 3 {
                let d: Vec<usize> = tok.split('x').filter_map(|s| s.parse().ok()).collect();
                if d.len() == 3 {
                    lm = Some(d[0]);
                    ln = d[1];
                    kt = d[2];
                }
            }
        }
        Some(Self {
            lm: lm?,
            ln,
            kt,
            nblk: nblk?,
        })
    }

    // ── packers ─────────────────────────────────────────────────────────────
    /// Pack one iteration's W stripes. `codes` is the int4 weight matrix laid out
    /// **`[N][K]` row-major** (output-major, matching the DFlash `w_*.i8` dumps),
    /// values in `-8..=7`. `dst` must be [`Self::w_bytes`] and pre-zeroed for this
    /// slot (this ORs nibbles in).
    pub fn pack_w_slot(
        &self,
        dst: &mut [u8],
        slot: usize,
        codes: &[i8],
        k_total: usize,
        k0: usize,
        n0: usize,
    ) {
        let (ln, kt) = (self.ln, self.kt);
        for j in 0..GRID {
            let base = j * self.wt() + slot * self.wb();
            for jn in 0..ln {
                for ktile in 0..kt {
                    let toff = base + (jn * kt + ktile) * (MK * MN / 2);
                    for kk in 0..MK {
                        let kg = k0 + ktile * MK + kk;
                        for nn in 0..MN {
                            let idx = kk * MN + nn;
                            let ng = n0 + j * ln * MN + jn * MN + nn;
                            let v = (codes[ng * k_total + kg] & 0xf) as u8;
                            let b = &mut dst[toff + idx / 2];
                            *b = if idx % 2 == 0 { v } else { *b | (v << 4) };
                        }
                    }
                }
            }
        }
    }

    /// Pack one iteration's A stripes from a row-major `[rows][K]` int8 activation.
    pub fn pack_a_slot(
        &self,
        dst: &mut [u8],
        slot: usize,
        a: &[i8],
        k_total: usize,
        m0: usize,
        k0: usize,
    ) {
        let (lm, kt) = (self.lm, self.kt);
        for i in 0..GRID {
            let base = i * self.at() + slot * self.ab();
            for im in 0..lm {
                for ktile in 0..kt {
                    let toff = base + (im * kt + ktile) * (MR * MK);
                    for r in 0..MR {
                        let row = m0 + i * lm * MR + im * MR + r;
                        let src = row * k_total + k0 + ktile * MK;
                        // int8 -> u8 is a pure reinterpret; copy the whole MK run
                        // as one memcpy instead of MK bounds-checked byte stores.
                        let s: &[u8] = unsafe {
                            std::slice::from_raw_parts(a[src..src + MK].as_ptr() as *const u8, MK)
                        };
                        dst[toff + r * MK..toff + r * MK + MK].copy_from_slice(s);
                    }
                }
            }
        }
    }

    /// Read one iteration's C grid as contiguous `MN`-wide runs, calling
    /// `f(row, col0, &values[..MN])`. Same traversal order as [`Self::each_c`]
    /// — only the innermost `MN` loop is handed to the caller as a slice, so a
    /// consumer can vectorize it without changing accumulation order.
    pub fn each_c_run(&self, c: &[i32], slot: usize, mut f: impl FnMut(usize, usize, &[i32])) {
        let (lm, ln, cb) = (self.lm, self.ln, self.cb());
        for j in 0..GRID {
            for i in 0..GRID {
                let base = j * self.ct() + slot * GRID * cb + i * cb;
                for im in 0..lm {
                    for jn in 0..ln {
                        let toff = base + (im * ln + jn) * (MR * MN);
                        for r in 0..MR {
                            let row = i * lm * MR + im * MR + r;
                            let col0 = j * ln * MN + jn * MN;
                            let o = toff + r * MN;
                            f(row, col0, &c[o..o + MN]);
                        }
                    }
                }
            }
        }
    }

    /// Read one iteration's C grid, calling `f(row, col, value)` for each element.
    pub fn each_c(&self, c: &[i32], slot: usize, mut f: impl FnMut(usize, usize, i32)) {
        let (lm, ln, cb) = (self.lm, self.ln, self.cb());
        for j in 0..GRID {
            for i in 0..GRID {
                let base = j * self.ct() + slot * GRID * cb + i * cb;
                for im in 0..lm {
                    for jn in 0..ln {
                        let toff = base + (im * ln + jn) * (MR * MN);
                        for r in 0..MR {
                            let row = i * lm * MR + im * MR + r;
                            for nn in 0..MN {
                                let col = j * ln * MN + jn * MN + nn;
                                f(row, col, c[toff + r * MN + nn]);
                            }
                        }
                    }
                }
            }
        }
    }
}

/// A loaded r14 array kernel with its three argument buffers.
pub struct NpuGemmR14 {
    kernel: NpuKernel,
    geom: R14Geometry,
    a_buf: DeviceBuffer,
    c_buf: DeviceBuffer,
}

impl NpuGemmR14 {
    /// Load from an `r14_cache.sh` dir, taking the geometry from its name.
    /// (`NpuGemmMp::load_cached` cannot parse these names — its `_r{N}` whole-GEMM
    /// guard matches the `r14` token itself, and there is no `c{COLS}` token.)
    pub fn load_dir(dir: &str) -> Result<Self, XdnaError> {
        let (xclbin, insts, geom) = Self::read_dir(dir)?;
        Self::wrap(NpuKernel::load(&xclbin, &insts)?, geom)
    }

    /// Same, but sharing an existing kernel's DRM file + device heap.
    pub fn load_peer_dir(peer: &NpuKernel, dir: &str) -> Result<Self, XdnaError> {
        let (xclbin, insts, geom) = Self::read_dir(dir)?;
        Self::wrap(NpuKernel::load_peer(peer, &xclbin, &insts)?, geom)
    }

    fn read_dir(dir: &str) -> Result<(Vec<u8>, Vec<u8>, R14Geometry), XdnaError> {
        let base = std::path::Path::new(dir)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("");
        let geom =
            R14Geometry::from_dir_name(base).ok_or_else(|| XdnaError::BadCacheName(base.into()))?;
        let xclbin = std::fs::read(format!("{dir}/final.xclbin")).map_err(XdnaError::Open)?;
        let insts = std::fs::read(format!("{dir}/insts.bin")).map_err(XdnaError::Open)?;
        Ok((xclbin, insts, geom))
    }

    fn wrap(kernel: NpuKernel, geom: R14Geometry) -> Result<Self, XdnaError> {
        let a_buf = kernel.alloc_arg(geom.a_bytes())?;
        let c_buf = kernel.alloc_arg(geom.c_bytes())?;
        Ok(Self {
            kernel,
            geom,
            a_buf,
            c_buf,
        })
    }

    pub fn geom(&self) -> R14Geometry {
        self.geom
    }
    pub fn kernel(&self) -> &NpuKernel {
        &self.kernel
    }

    /// Allocate a resident, pre-packed weight buffer for one dispatch's `NBLK`
    /// iterations ([`R14Geometry::w_bytes`]).
    pub fn alloc_weights(&self) -> Result<DeviceBuffer, XdnaError> {
        self.kernel.alloc_arg(self.geom.w_bytes())
    }

    /// Flush a weight buffer to the device once, after packing.
    pub fn sync_weights(&self, w: &DeviceBuffer) -> Result<(), XdnaError> {
        self.kernel.sync_to_device(w)
    }

    /// Host view of the A argument buffer.
    pub fn a_mut(&mut self) -> &mut [u8] {
        self.a_buf.as_mut_slice()
    }

    /// Dispatch one `NBLK`-iteration batch against a resident weight buffer.
    /// A is synced (it changes every call); W is **not** re-flushed.
    pub fn dispatch(&self, weights: &DeviceBuffer) -> Result<(), XdnaError> {
        self.kernel
            .dispatch_synced(&[&self.a_buf, weights, &self.c_buf], &[true, false, false])
    }

    /// Sync C back and view it as int32.
    pub fn read_c(&self) -> Result<&[i32], XdnaError> {
        self.kernel.sync_output(&self.c_buf)?;
        let s = self.c_buf.as_slice();
        Ok(unsafe { std::slice::from_raw_parts(s.as_ptr() as *const i32, s.len() / 4) })
    }
}
