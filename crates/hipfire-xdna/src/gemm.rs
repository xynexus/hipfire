//! Wire-in step 2 — `NpuGemm`: a W4A8 GEMM primitive over the R6-TS kernel
//! (`r6_gemm_ts.cc`). Turns a standard row-major GEMM into R6 dispatches. A and C move
//! ROW-MAJOR: the kernel's in-core tensor buffer streams do the tile reshuffle via the
//! address generators (no CPU marshaling, no strided DMA), so `a_buf`/`c_buf` are plain
//! row-major blocks. Only W is pre-packed into the kernel's tile-major int4 layout — a
//! static, once-per-load cost (see [`Self::prepack_weights`]). Validated by
//! `r6_ts_verify` (0 mismatches at MT=8 and the MT=24 peak).
//!
//! `groups` = how many `NT·MN`-wide N-slabs one dispatch computes: 1 for the single
//! core (`r6_cache.sh` COLS=1), or `COLS·NB` for the array (COLS cores × NB streamed
//! blocks) — the latter is where the 20.7-TOPS throughput lives. Each dispatch does
//! one M-block (`MT·MR` rows, shared A) × one K-chunk (`KCHUNK·MK`) × `groups` N-slabs;
//! [`Self::run`] tiles the full GEMM over that (M/N independent, K accumulated).
//! Linux-only (amdxdna).
#![cfg(target_os = "linux")]

use crate::{DeviceBuffer, NpuKernel, XdnaError};

const MR: usize = 4; // mmul M
const MK: usize = 16; // mmul K
const MN: usize = 16; // mmul N

/// Dispatches kept in flight by [`NpuGemm::run_resident`]. Each needs its own A and C
/// buffer. 1 is submit-latency-bound; 2 hides the ~78 µs submit behind the previous
/// dispatch's weight streaming and is where the gain is (18.4 -> 28.0 GB/s on
/// K=2048 N=8192). Measured deeper: 4 is slightly WORSE (27.4 GB/s, and 28.9 vs 32.8
/// on K=8192) — two in flight already saturates the queue, so extra slots only add
/// buffer footprint and readback pressure.
const PIPE: usize = 2;

/// A loaded R6 kernel specialized for (MT, NT, KCHUNK, groups) with reusable buffers.
pub struct NpuGemm {
    kernel: NpuKernel,
    mt: usize,
    nt: usize,
    kchunk: usize,
    groups: usize,
    /// N-slabs per column (`r6_cache.sh` NB). `groups == cols * nb`. Only affects the
    /// W/C region ordering when `rounds > 1`; with `rounds == 1` the (col, nb) walk
    /// collapses to plain group order and any value tiles identically.
    nb: usize,
    /// M-blocks streamed per core in ONE dispatch (`r6_gen.py` ROUNDS). The cores loop
    /// over ROUNDS resident A-blocks without an inter-dispatch host stall, so this is
    /// the main lever against the ~78 µs dispatch latency.
    rounds: usize,
    // PIPE-deep ring: with resident weights the only thing stopping N dispatches being
    // in flight is A/C being overwritten under a running one, so both are ringed.
    a_buf: Vec<DeviceBuffer>, // rounds * MT*KCHUNK tiles of MR*MK int8 (broadcast to cores)
    w_buf: DeviceBuffer,      // groups * rounds * NT*KCHUNK tiles of MK*MN int4 (2/byte)
    c_buf: Vec<DeviceBuffer>, // one dispatch writes while the host reads an earlier
                              // one back (pipelined run_packed / run_resident).
}

impl NpuGemm {
    /// M rows computed per dispatch (`rounds` M-blocks of `MT·MR`).
    pub fn block_m(&self) -> usize {
        self.rounds * self.mt * MR
    }
    /// N cols computed per dispatch (`groups` N-slabs).
    pub fn block_n(&self) -> usize {
        self.groups * self.nt * MN
    }
    /// K contracted per dispatch (one chunk).
    pub fn block_k(&self) -> usize {
        self.kchunk * MK
    }

    /// Load an R6 xclbin built for (mt, nt, kchunk) with `groups` = COLS·NB N-slabs
    /// (1 for the single-core cache) and allocate its arg buffers.
    pub fn load(
        xclbin: &[u8],
        insts: &[u8],
        mt: usize,
        nt: usize,
        kchunk: usize,
        groups: usize,
    ) -> Result<Self, XdnaError> {
        Self::load_rounds(xclbin, insts, mt, nt, kchunk, groups, groups, 1)
    }

    /// Load an R6 xclbin built with `R6_ROUNDS=rounds` (see `r6_gen.py`): each core
    /// streams `rounds` M-blocks per dispatch, cutting dispatch count — and therefore
    /// the ~78 µs per-dispatch latency — by the same factor. `nb` is the builder's NB
    /// (N-slabs per column); `groups` must equal `cols * nb`.
    #[allow(clippy::too_many_arguments)]
    pub fn load_rounds(
        xclbin: &[u8],
        insts: &[u8],
        mt: usize,
        nt: usize,
        kchunk: usize,
        groups: usize,
        nb: usize,
        rounds: usize,
    ) -> Result<Self, XdnaError> {
        assert!(
            rounds >= 1 && nb >= 1 && groups % nb == 0,
            "groups must be cols*nb"
        );
        let kernel = NpuKernel::load(xclbin, insts)?;
        let asz = rounds * mt * kchunk * MR * MK;
        let mut a_buf = Vec::with_capacity(PIPE);
        for _ in 0..PIPE {
            a_buf.push(kernel.alloc_arg(asz)?);
        }
        let w_buf = kernel.alloc_arg(groups * rounds * nt * kchunk * MK * MN / 2)?;
        let csz = groups * rounds * mt * nt * MR * MN * 4;
        let mut c_buf = Vec::with_capacity(PIPE);
        for _ in 0..PIPE {
            c_buf.push(kernel.alloc_arg(csz)?);
        }
        Ok(Self {
            kernel,
            mt,
            nt,
            kchunk,
            groups,
            nb,
            rounds,
            a_buf,
            w_buf,
            c_buf,
        })
    }

    /// Bytes of one group's W slab (`NT·KCHUNK` tiles of `MK·MN` int4).
    fn slab_bytes(&self) -> usize {
        self.nt * self.kchunk * MK * MN / 2
    }

    /// Stride between `(ko, no)` blocks in a [`Self::prepack_weights`] buffer: ONE
    /// round's worth (`groups` slabs in group order). The per-round replication the
    /// kernel wants is expanded at copy time, so the prepacked array is not `rounds`×
    /// larger than the weights themselves.
    fn packed_stride(&self) -> usize {
        self.groups * self.slab_bytes()
    }

    /// Fill `w_buf` from one `(ko, no)` block of a prepacked buffer, replicating each
    /// group's slab across rounds in the core-major order the MLIR streams:
    /// `(col, round, nb)` — core `c` owns the contiguous region `c*rounds*nb` slabs.
    fn fill_w_replicated(&mut self, block: &[u8]) {
        let (g, nb, rounds, sb) = (self.groups, self.nb, self.rounds, self.slab_bytes());
        let cols = g / nb;
        let dst = self.w_buf.as_mut_slice();
        Self::replicate_into(dst, block, cols, rounds, nb, sb);
    }

    /// Same expansion, into an arbitrary destination (used to fill resident per-block
    /// weight buffers once at load).
    fn replicate_into(
        dst: &mut [u8],
        block: &[u8],
        cols: usize,
        rounds: usize,
        nb: usize,
        sb: usize,
    ) {
        for c in 0..cols {
            for r in 0..rounds {
                for j in 0..nb {
                    let gi = c * nb + j; // group whose N-slab this is
                    let d = ((c * rounds + r) * nb + j) * sb;
                    dst[d..d + sb].copy_from_slice(&block[gi * sb..(gi + 1) * sb]);
                }
            }
        }
    }

    /// Upload the whole `K×N` weight matrix into DEVICE-RESIDENT per-block buffers, one
    /// per `(ko, no)` dispatch block, already in the kernel's replicated tile-major form.
    /// After this, a GEMM costs ZERO host weight traffic — [`Self::run_resident`] just
    /// binds the right buffer per dispatch.
    ///
    /// This is what makes decode viable. `run_packed` re-fills a shared `w_buf` from host
    /// memory, which for GEMV means copying the entire weight matrix per token — at 8.4 MB
    /// for one llama-3.2-1B projection that is milliseconds of pure memcpy per linear per
    /// token, dwarfing the compute. Weights are static, so they belong on the device once.
    /// Costs `K*N/2 * (rounds)` bytes of device memory per matrix; a 1B at oq4 is well
    /// inside the 128 GB UMA budget.
    pub fn upload_weights(
        &self,
        k: usize,
        n: usize,
        w_int4: &[i8],
    ) -> Result<Vec<DeviceBuffer>, XdnaError> {
        let (bn, bk) = (self.block_n(), self.block_k());
        assert!(k % bk == 0 && n % bn == 0, "K/N must tile evenly");
        let (nks, nns) = (k / bk, n / bn);
        let (cols, rounds, nb, sb) = (
            self.groups / self.nb,
            self.rounds,
            self.nb,
            self.slab_bytes(),
        );
        let wl = self.packed_stride();
        let mut block = vec![0u8; wl];
        let mut w_sub = vec![0i8; bk * bn];
        let mut out = Vec::with_capacity(nks * nns);
        for ko_i in 0..nks {
            for no_i in 0..nns {
                for i in 0..bk {
                    let src = (ko_i * bk + i) * n + no_i * bn;
                    w_sub[i * bn..(i + 1) * bn].copy_from_slice(&w_int4[src..src + bn]);
                }
                self.pack_w_slab(&w_sub, &mut block);
                let mut buf = self.kernel.alloc_arg(self.groups * rounds * sb)?;
                Self::replicate_into(buf.as_mut_slice(), &block, cols, rounds, nb, sb);
                self.kernel.sync_to_device(&buf)?;
                out.push(buf);
            }
        }
        Ok(out)
    }

    /// GEMM against device-resident weights from [`Self::upload_weights`]. Identical
    /// tiling to `run_packed`, but the per-dispatch W cost is binding a buffer instead of
    /// copying one, and W never needs re-syncing.
    pub fn run_resident(
        &mut self,
        m: usize,
        k: usize,
        n: usize,
        a: &[i8],
        weights: &[DeviceBuffer],
        c: &mut [i32],
    ) -> Result<(), XdnaError> {
        let (bm, bn, bk) = (self.block_m(), self.block_n(), self.block_k());
        assert!(
            m % bm == 0 && n % bn == 0 && k % bk == 0,
            "M/N/K must tile evenly (block {bm}x{bn}x{bk})"
        );
        let (nms, nns, nks) = (m / bm, n / bn, k / bk);
        assert_eq!(weights.len(), nks * nns, "weight block count");
        let ndisp = nms * nns * nks;
        if ndisp == 0 {
            return Ok(());
        }
        let mut a_sub = vec![0i8; bm * bk];
        // mo innermost: A for a given (ko) is reused across no, and C accumulates in place.
        let coord = |i: usize| {
            let per_n = nks * nms;
            let (no_i, r) = (i / per_n, i % per_n);
            (r % nms, no_i, r / nms)
        };
        // Keep up to PIPE dispatches in flight. Nothing serialises them: A and C come
        // from PIPE-deep rings and every weight block is its own resident buffer, so
        // dispatch i shares no buffer with the PIPE-1 before it. Submitting i only has
        // to wait for i-PIPE, whose slot it reuses — and retiring that one in order also
        // makes its C safe to read. This is what moves decode from submit-latency-bound
        // to weight-streaming-bound.
        let mut inflight: std::collections::VecDeque<(u64, usize)> = Default::default();
        let retire = |g: &Self,
                      q: &mut std::collections::VecDeque<(u64, usize)>,
                      c: &mut [i32]|
         -> Result<(), XdnaError> {
            if let Some((pseq, pi)) = q.pop_front() {
                g.kernel.wait(pseq)?;
                g.kernel.sync_output(&g.c_buf[pi % PIPE])?;
                let (pm, pn, pk) = coord(pi);
                g.readback(pi % PIPE, pm, pn, pk, n, c);
            }
            Ok(())
        };
        for i in 0..ndisp {
            if inflight.len() >= PIPE {
                retire(self, &mut inflight, c)?; // frees slot i % PIPE
            }
            let (mo_i, no_i, ko_i) = coord(i);
            for r in 0..bm {
                let src = (mo_i * bm + r) * k + ko_i * bk;
                a_sub[r * bk..(r + 1) * bk].copy_from_slice(&a[src..src + bk]);
            }
            self.load_a_slot(&a_sub, i % PIPE);
            let seq = self.kernel.submit_synced(
                &[
                    &self.a_buf[i % PIPE],
                    &weights[ko_i * nns + no_i],
                    &self.c_buf[i % PIPE],
                ],
                Some(&[true, false, true]), // W is already on-device and never dirtied
            )?;
            inflight.push_back((seq, i));
        }
        while !inflight.is_empty() {
            retire(self, &mut inflight, c)?;
        }
        Ok(())
    }

    /// Full GEMM `C[M,N] = A[M,K] · W[K,N]` (W4A8) by tiling over R6 dispatches: M and N
    /// split into blocks, K accumulated. `a` row-major `M×K` int8, `w_int4` row-major
    /// `K×N` int4 values (`-8..=7`, one per byte), `c` row-major `M×N` int32.
    /// M/N/K must be multiples of `block_m()`/`block_n()`/`block_k()`.
    pub fn run(
        &mut self,
        m: usize,
        k: usize,
        n: usize,
        a: &[i8],
        w_int4: &[i8],
        c: &mut [i32],
    ) -> Result<(), XdnaError> {
        let (bm, bn, bk) = (self.block_m(), self.block_n(), self.block_k());
        assert!(
            m % bm == 0 && n % bn == 0 && k % bk == 0,
            "M/N/K must tile evenly (block {bm}x{bn}x{bk})"
        );
        let mut a_sub = vec![0i8; bm * bk];
        let mut w_sub = vec![0i8; bk * bn];
        let mut c_blk = vec![0i32; bm * bn];
        let mut c_acc = vec![0i32; bm * bn];
        for mo in (0..m).step_by(bm) {
            for no in (0..n).step_by(bn) {
                c_acc.iter_mut().for_each(|x| *x = 0);
                for ko in (0..k).step_by(bk) {
                    for i in 0..bm {
                        a_sub[i * bk..(i + 1) * bk]
                            .copy_from_slice(&a[(mo + i) * k + ko..(mo + i) * k + ko + bk]);
                    }
                    for i in 0..bk {
                        w_sub[i * bn..(i + 1) * bn]
                            .copy_from_slice(&w_int4[(ko + i) * n + no..(ko + i) * n + no + bn]);
                    }
                    self.run_slab(&a_sub, &w_sub, &mut c_blk)?;
                    for (acc, &v) in c_acc.iter_mut().zip(c_blk.iter()) {
                        *acc += v;
                    }
                }
                for i in 0..bm {
                    c[(mo + i) * n + no..(mo + i) * n + no + bn]
                        .copy_from_slice(&c_acc[i * bn..(i + 1) * bn]);
                }
            }
        }
        Ok(())
    }

    /// One dispatch: `a` row-major `(MT·MR) × (KCHUNK·MK)` int8; `w_int4` row-major
    /// `(KCHUNK·MK) × (groups·NT·MN)` int4 values; `c` gets row-major
    /// `(MT·MR) × (groups·NT·MN)` int32. A/C are copied row-major (the kernel's tensor
    /// streams tile them in-core); W is pre-packed tile-major. Dispatches, copies C out.
    pub fn run_slab(&mut self, a: &[i8], w_int4: &[i8], c: &mut [i32]) -> Result<(), XdnaError> {
        let (nt, kc, g) = (self.nt, self.kchunk, self.groups);
        let k = kc * MK;
        let n = g * nt * MN; // full N of this slab
        let bm = self.block_m();
        assert_eq!(a.len(), bm * k, "A shape");
        assert_eq!(w_int4.len(), k * n, "W shape");
        assert_eq!(c.len(), bm * n, "C shape");

        self.load_a_slot(a, 0); // row-major A blocks -> a_buf (kernel tensor-streams the tiling)
                                // W -> per-group tile-major + int4 pack, then replicated per round.
        let mut block = vec![0u8; self.packed_stride()];
        self.pack_w_slab(w_int4, &mut block);
        self.fill_w_replicated(&block);

        self.kernel
            .dispatch(&[&self.a_buf[0], &self.w_buf, &self.c_buf[0]])?;
        self.unpack_c_block(0, c, 0, n); // c is exactly the bm×bn (=n-wide) block
        Ok(())
    }

    /// Marshal one `(KCHUNK·MK) × (groups·NT·MN)` int4 W slab into `out` (a
    /// `wbuf_len()`-sized tile-major, int4-packed buffer). Pure CPU; no dispatch.
    fn pack_w_slab(&self, w_int4: &[i8], out: &mut [u8]) {
        let (nt, kc, g) = (self.nt, self.kchunk, self.groups);
        let n = g * nt * MN;
        out.fill(0);
        for gi in 0..g {
            let wbase = gi * (nt * kc) * (MK * MN);
            let ncol0 = gi * nt * MN;
            for nti in 0..nt {
                for ki in 0..kc {
                    for kk in 0..MK {
                        for nn in 0..MN {
                            let v =
                                (w_int4[(ki * MK + kk) * n + ncol0 + nti * MN + nn] & 0xf) as u8;
                            let idx = wbase + (nti * kc + ki) * (MK * MN) + kk * MN + nn;
                            out[idx / 2] |= if idx % 2 == 0 { v } else { v << 4 };
                        }
                    }
                }
            }
        }
    }

    // Copy c_buf[slot] into `dst` at `dst[base + row*row_stride + col]`. The R6-TS kernel
    // writes each group's block ROW-MAJOR (its C tensor stream de-tiles in-core), so this
    // is a straight per-group block copy — no reshuffle. `base`/`row_stride` place the
    // (MT·MR)×(groups·NT·MN) result anywhere in a larger output (e.g. the (mo,no) block of
    // the full GEMM), so no intermediate copy is needed on the single-K-chunk path.
    fn unpack_c_block(&self, slot: usize, dst: &mut [i32], base: usize, row_stride: usize) {
        self.unpack_c_inner(slot, dst, base, row_stride, false)
    }

    /// Add this dispatch's C into `dst` instead of overwriting it — the K-accumulation
    /// path. Fused so a multi-K-chunk GEMM makes ONE pass over M·N per chunk instead of
    /// unpacking into scratch and then adding (two passes over the same int32s).
    fn accum_c_block(&self, slot: usize, dst: &mut [i32], base: usize, row_stride: usize) {
        self.unpack_c_inner(slot, dst, base, row_stride, true)
    }

    fn unpack_c_inner(
        &self,
        slot: usize,
        dst: &mut [i32],
        base: usize,
        row_stride: usize,
        accumulate: bool,
    ) {
        let (mt, nt, g, nb, rounds) = (self.mt, self.nt, self.groups, self.nb, self.rounds);
        let bw = nt * MN; // group width (cols per group)
        let bh = mt * MR; // rows per M-block
        let cols = g / nb;
        let out: &[i32] = unsafe {
            std::slice::from_raw_parts(
                self.c_buf[slot].as_slice().as_ptr() as *const i32,
                g * rounds * bh * bw,
            )
        };
        // C mirrors the W walk: core c's region holds `rounds*nb` blocks in (round, nb)
        // order, and round r is M-block r of this dispatch (row offset r*bh).
        for c in 0..cols {
            for r in 0..rounds {
                for j in 0..nb {
                    let cbase = ((c * rounds + r) * nb + j) * bh * bw;
                    let gcol0 = (c * nb + j) * bw; // N-slab column offset
                    for row in 0..bh {
                        let src = cbase + row * bw;
                        let d = base + (r * bh + row) * row_stride + gcol0;
                        if accumulate {
                            for (o, &v) in dst[d..d + bw].iter_mut().zip(&out[src..src + bw]) {
                                *o += v;
                            }
                        } else {
                            dst[d..d + bw].copy_from_slice(&out[src..src + bw]);
                        }
                    }
                }
            }
        }
    }

    /// Pre-marshal a full `K×N` weight matrix ONCE into the tile-major, int4-packed
    /// form the kernel consumes — the slow bit-packing that must NOT happen per
    /// inference (weights are static). The result is indexed by (K-chunk, N-slab):
    /// block `(ko, no)` at `(ko*n_slabs + no) * wbuf_len()`. Pass to [`Self::run_packed`].
    pub fn prepack_weights(&self, k: usize, n: usize, w_int4: &[i8]) -> Vec<u8> {
        let (bn, bk) = (self.block_n(), self.block_k());
        assert!(k % bk == 0 && n % bn == 0, "K/N must tile evenly");
        let (nks, nns, wl) = (k / bk, n / bn, self.packed_stride());
        let mut packed = vec![0u8; nks * nns * wl];
        let mut w_sub = vec![0i8; bk * bn];
        for ko_i in 0..nks {
            for no_i in 0..nns {
                for i in 0..bk {
                    let src = (ko_i * bk + i) * n + no_i * bn;
                    w_sub[i * bn..(i + 1) * bn].copy_from_slice(&w_int4[src..src + bn]);
                }
                let off = (ko_i * nns + no_i) * wl;
                self.pack_w_slab(&w_sub, &mut packed[off..off + wl]);
            }
        }
        packed
    }

    /// Full GEMM using pre-marshaled weights (from [`Self::prepack_weights`]): the
    /// per-dispatch weight cost is a `memcpy`, not a re-pack — the whole point of the
    /// hot path. Only `a` (activations) is marshaled per inference.
    pub fn run_packed(
        &mut self,
        m: usize,
        k: usize,
        n: usize,
        a: &[i8],
        packed_w: &[u8],
        c: &mut [i32],
    ) -> Result<(), XdnaError> {
        let (bm, bn, bk) = (self.block_m(), self.block_n(), self.block_k());
        assert!(
            m % bm == 0 && n % bn == 0 && k % bk == 0,
            "M/N/K must tile evenly"
        );
        let (nms, nns, nks) = (m / bm, n / bn, k / bk);
        let wl = self.packed_stride();
        let ndisp = nms * nns * nks;
        if ndisp == 0 {
            return Ok(());
        }

        let mut a_sub = vec![0i8; bm * bk];

        // Flat dispatch order: no outer, then ko, with mo INNERMOST. The W slab is a
        // function of (ko, no) only, so walking all M-blocks inside it copies W
        // `nns*nks` times instead of `nms*nns*nks` — on a 2048x2048x8192 prefill that
        // is 8 W fills instead of 128, and the W buffer is `groups*rounds` slabs, so
        // the copies dominate everything else the host does. K-partials then land
        // straight in the output block (overwrite at ko==0, accumulate after), so the
        // scratch accumulator and its extra pass over M*N disappear too.
        let coord = |i: usize| {
            let per_n = nks * nms;
            let (no_i, r) = (i / per_n, i % per_n);
            (r % nms, no_i, r / nms) // (mo_i, no_i, ko_i)
        };

        // Software-pipelined: submit dispatch i (into c_buf[i%2]) BEFORE reading dispatch
        // i-1 back, so the host read-back overlaps dispatch i's execution on the NPU.
        // a_buf/w_buf are single-buffered but only refilled after wait(i-1), so the
        // in-flight dispatch never sees torn inputs; c_buf is double-buffered so dispatch
        // i's output can't clobber i-1's before we read it. W copy is skipped when the
        // slab is unchanged from the previous dispatch (same (ko,no)).
        let mut prev: Option<(u64, usize)> = None; // (timeline seq, dispatch index)
        let mut last_w_off = usize::MAX;
        for i in 0..ndisp {
            if let Some((seq, _)) = prev {
                self.kernel.wait(seq)?; // i-1 done: a_buf/w_buf free, its C readable
            }
            let (mo_i, no_i, ko_i) = coord(i);
            for r in 0..bm {
                let src = (mo_i * bm + r) * k + ko_i * bk;
                a_sub[r * bk..(r + 1) * bk].copy_from_slice(&a[src..src + bk]);
            }
            self.load_a_slot(&a_sub, i % 2);
            let off = (ko_i * nns + no_i) * wl;
            let w_changed = off != last_w_off;
            if w_changed {
                self.fill_w_replicated(&packed_w[off..off + wl]);
                last_w_off = off;
            }
            // Flush A (rewritten every dispatch) and W (only when re-copied). C must
            // still be synced: the to-device flush also manages the CPU cache so the
            // post-dispatch read-back sees the kernel's DMA writes, not stale lines.
            let seq = self.kernel.submit_synced(
                &[&self.a_buf[i % 2], &self.w_buf, &self.c_buf[i % 2]],
                Some(&[true, w_changed, true]),
            )?;
            if let Some((_, pi)) = prev {
                self.kernel.sync_output(&self.c_buf[pi % 2])?; // invalidate before read
                let (pm, pn, pk) = coord(pi);
                self.readback(pi % 2, pm, pn, pk, n, c);
            }
            prev = Some((seq, i));
        }
        if let Some((seq, pi)) = prev {
            self.kernel.wait(seq)?;
            self.kernel.sync_output(&self.c_buf[pi % 2])?;
            let (pm, pn, pk) = coord(pi);
            self.readback(pi % 2, pm, pn, pk, n, c);
        }
        Ok(())
    }

    /// Read one completed dispatch's C directly into its `(mo, no)` block of the output.
    /// The first K-chunk overwrites and later ones accumulate in place, so a multi-chunk
    /// GEMM needs no scratch buffer, no zero-fill, and exactly one pass over M·N per
    /// chunk. Relies on the `coord` walk visiting every `ko` for a given `(mo, no)`.
    fn readback(
        &self,
        slot: usize,
        mo_i: usize,
        no_i: usize,
        ko_i: usize,
        n: usize,
        c: &mut [i32],
    ) {
        let (bm, bn) = (self.block_m(), self.block_n());
        let base = mo_i * bm * n + no_i * bn;
        if ko_i == 0 {
            self.unpack_c_block(slot, c, base, n);
        } else {
            self.accum_c_block(slot, c, base, n);
        }
    }

    // Copy `a` `(MT·MR) × (KCHUNK·MK)` int8 row-major into a_buf. The R6-TS kernel's A
    // tensor stream tiles it in-core, so no CPU reshuffle — just an int8->u8 block copy.
    fn load_a_slot(&mut self, a: &[i8], slot: usize) {
        let s = self.a_buf[slot].as_mut_slice();
        debug_assert_eq!(s.len(), a.len());
        for (d, &v) in s.iter_mut().zip(a.iter()) {
            *d = v as u8;
        }
    }
}
