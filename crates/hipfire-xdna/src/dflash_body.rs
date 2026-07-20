//! DFlash NPU block body as a callable library struct (Phase 1 reachability).
//!
//! Lifts the 5-layer DFlash draft block-forward out of the harness example
//! `examples/dflash_body_native.rs` into a struct the runtime spec-decode loop
//! can call as a *serial* substitute for the GPU draft. It is an addition, not
//! a replacement: the harness stays the instrumented parity reference, this is
//! the lean hot-path version.
//!
//! Fixed to the **validated body** (`docs/plans/2026-07-19-dflash-phase0-brief.md`):
//! multi-core W4A8 r14 GEMM + flash attention + CPU primitives. Unlike the
//! harness it does NOT keep a cross-call context cache — the runtime feeds a
//! sliding L-row window of `target_hidden`, so every call recomputes the context
//! projection. That is the slower no-cache path (~185 ms/block) but correct for
//! a moving window; Phase 1 makes no performance claim.
//!
//! Layout contract (must match the GPU draft buffers so the seam is a drop-in):
//!   * input `target_hidden` : `[l_ctx, num_extract * hidden]` f32
//!   * input `noise`         : `[block, hidden]` f32 (the target-embedded block)
//!   * output `block_hidden` : `[block, hidden]` f32
//!
//! Correctness note: spec decode is lossless — the GPU target verifies every
//! drafted token — so the *committed* sequence is byte-identical regardless of
//! this body's numerics. Phase 1 proves the seam (`02e621bd56b5`); acceptance
//! rate (the numeric quality) is a later phase.

#![cfg(target_os = "linux")]

use crate::gemm_r14::{NpuGemmR14, R14Geometry, GRID};
use crate::{DeviceBuffer, NpuKernel, XdnaError};
use std::collections::HashMap;

const QMAX: f32 = 127.0;

// ── bf16 <-> f32 (round-to-nearest-even, matching ml_dtypes) ────────────────
#[inline]
pub fn f32_to_bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    if x.is_nan() {
        return ((bits >> 16) as u16) | 0x0040;
    }
    let bias = 0x7fff + ((bits >> 16) & 1);
    ((bits + bias) >> 16) as u16
}

#[inline]
pub fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

// ── int8 per-(row, k-chunk) symmetric activation quant ──────────────────────
/// Per-(row, k-chunk) symmetric int8 activation quant (the r14 array contracts
/// one K_CHUNK at a time).
fn quantize_row_chunked(x: &[f32], rows: usize, k: usize, kc: usize, q: &mut [i8], scale: &mut [f32]) {
    let chunks = k / kc;
    for r in 0..rows {
        for c in 0..chunks {
            let seg = &x[r * k + c * kc..r * k + (c + 1) * kc];
            let absmax = seg.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
            let s = if absmax > 0.0 { absmax / QMAX } else { 1.0 };
            scale[r * chunks + c] = s;
            let inv = 1.0 / s;
            for (i, &v) in seg.iter().enumerate() {
                q[r * k + c * kc + i] = (v * inv).round_ties_even().clamp(-QMAX, QMAX) as i8;
            }
        }
    }
}

// ── rope cos/sin table ──────────────────────────────────────────────────────
fn cs_buf(hd: usize, pos: f64, theta: f64, out: &mut [f32]) {
    let half = hd / 2;
    for i in 0..half {
        let freq = 1.0 / theta.powf(2.0 * i as f64 / hd as f64);
        let ang = pos * freq;
        out[i] = ang.cos() as f32;
        out[half + i] = ang.sin() as f32;
    }
}

// ── CPU primitives (f32, mirror the numpy reference) ────────────────────────
fn cpu_rmsnorm(x: &[f32], gamma: &[f32], rows: usize, h: usize, eps: f32, out: &mut [f32]) {
    for r in 0..rows {
        let row = &x[r * h..(r + 1) * h];
        let ss: f32 = row.iter().map(|&v| v * v).sum();
        let inv = 1.0 / (ss / h as f32 + eps).sqrt();
        let o = &mut out[r * h..(r + 1) * h];
        for i in 0..h {
            o[i] = row[i] * inv * gamma[i];
        }
    }
}

fn cpu_headnorm(x: &[f32], gamma: &[f32], rows: usize, heads: usize, hd: usize, eps: f32, out: &mut [f32]) {
    for t in 0..rows * heads {
        let head = &x[t * hd..(t + 1) * hd];
        let ss: f32 = head.iter().map(|&v| v * v).sum();
        let inv = 1.0 / (ss / hd as f32 + eps).sqrt();
        let o = &mut out[t * hd..(t + 1) * hd];
        for i in 0..hd {
            o[i] = head[i] * inv * gamma[i];
        }
    }
}

fn cpu_rope(x: &[f32], rows: usize, heads: usize, hd: usize, pos0: usize, theta: f64, out: &mut [f32]) {
    let half = hd / 2;
    let mut cs = vec![0f32; hd];
    for r in 0..rows {
        cs_buf(hd, (pos0 + r) as f64, theta, &mut cs);
        let (c, s) = cs.split_at(half);
        for hh in 0..heads {
            let base = (r * heads + hh) * hd;
            let (xin, xout) = (&x[base..base + hd], &mut out[base..base + hd]);
            for i in 0..half {
                let (xi, yi) = (xin[i], xin[half + i]);
                xout[i] = xi * c[i] - yi * s[i];
                xout[half + i] = yi * c[i] + xi * s[i];
            }
        }
    }
}

fn cpu_swiglu(gateup: &[f32], m_gu: usize, i_dim: usize, rows: usize, out: &mut [f32]) {
    for r in 0..rows {
        for i in 0..i_dim {
            let g = gateup[r * m_gu + i];
            let u = gateup[r * m_gu + i_dim + i];
            out[r * i_dim + i] = (g / (1.0 + (-g).exp())) * u;
        }
    }
}

// ── multi-core (r14) W4A8 GEMM path ─────────────────────────────────────────
/// One weight matrix staged for the r14 array: int4 codes packed into resident
/// per-dispatch buffers plus the per-(out-row, k-chunk) dequant scale.
struct R14Matrix {
    m: usize,
    k: usize,
    rows: usize,
    k_chunks: usize,
    scale4t: Vec<f32>,
    wbufs: Vec<DeviceBuffer>,
    plan: Vec<(usize, usize, usize)>,
}

impl R14Matrix {
    fn build(g: &NpuGemmR14, raw: &[i8], row_scale: &[f32], m: usize, k: usize, rows: usize) -> Result<Self, XdnaError> {
        let geom = g.geom();
        let (mt, nt, kc) = (geom.m_tile(), geom.n_tile(), geom.k_chunk());
        assert_eq!(k % kc, 0, "K={k} must be a multiple of K_CHUNK={kc}");
        assert_eq!(m % nt, 0, "N={m} must be a multiple of N_TILE={nt}");
        assert_eq!(rows % mt, 0, "rows={rows} must be a multiple of M_TILE={mt}");
        let (k_chunks, n_tiles, m_blocks) = (k / kc, m / nt, rows / mt);

        let mut codes = vec![0i8; m * k];
        let mut scale4 = vec![0f32; m * k_chunks];
        for n in 0..m {
            for c in 0..k_chunks {
                let seg = &raw[n * k + c * kc..n * k + (c + 1) * kc];
                let amax = seg.iter().map(|v| (*v as i32).abs()).max().unwrap_or(0);
                if amax == 0 {
                    continue;
                }
                scale4[n * k_chunks + c] = row_scale[n] * amax as f32 / 7.0;
                let inv = 7.0 / amax as f32;
                for (i, &v) in seg.iter().enumerate() {
                    codes[n * k + c * kc + i] = (v as f32 * inv).round_ties_even().clamp(-7.0, 7.0) as i8;
                }
            }
        }

        let mut plan = Vec::with_capacity(m_blocks * k_chunks * n_tiles);
        for mb in 0..m_blocks {
            for c in 0..k_chunks {
                for t in 0..n_tiles {
                    plan.push((mb, c, t));
                }
            }
        }
        let nblk = geom.nblk;
        let ndisp = plan.len().div_ceil(nblk);
        let mut wbufs = Vec::with_capacity(ndisp);
        for d in 0..ndisp {
            let mut buf = g.alloc_weights()?;
            buf.as_mut_slice().fill(0);
            for s in 0..nblk {
                let Some(&(_, c, t)) = plan.get(d * nblk + s) else {
                    break;
                };
                geom.pack_w_slot(buf.as_mut_slice(), s, &codes, k, c * kc, t * nt);
            }
            g.sync_weights(&buf)?;
            wbufs.push(buf);
        }
        let mut scale4t = vec![0f32; m * k_chunks];
        for n in 0..m {
            for c in 0..k_chunks {
                scale4t[c * m + n] = scale4[n * k_chunks + c];
            }
        }
        Ok(Self { m, k, rows, k_chunks, scale4t, wbufs, plan })
    }
}

/// Run one full GEMM on the r14 array (device ns discarded — no measurement).
fn run_r14(g: &mut NpuGemmR14, mx: &R14Matrix, x: &[f32], out: &mut [f32], qa: &mut [i8], sx: &mut [f32]) {
    let geom: R14Geometry = g.geom();
    let (mt, nt, kc) = (geom.m_tile(), geom.n_tile(), geom.k_chunk());
    let (rows, m, k, nblk) = (mx.rows, mx.m, mx.k, geom.nblk);
    quantize_row_chunked(x, rows, k, kc, qa, sx);
    out[..rows * m].fill(0.0);

    let a_state = &mut None::<(usize, usize)>;
    for (d, wbuf) in mx.wbufs.iter().enumerate() {
        let lo = d * nblk;
        let hi = (lo + nblk).min(mx.plan.len());
        let span = &mx.plan[lo..hi];
        let uniform = span
            .first()
            .map(|&(mb, c, _)| span.iter().all(|&(m2, c2, _)| m2 == mb && c2 == c).then_some((mb, c)))
            .flatten();
        let skip = uniform.is_some() && uniform == *a_state;
        if !skip {
            let abuf = g.a_mut();
            let mut prev: Option<(usize, usize, usize)> = None;
            for (s, &(mb, c, _)) in mx.plan[lo..hi].iter().enumerate() {
                if let Some((pmb, pc, ps)) = prev {
                    if pmb == mb && pc == c {
                        for i in 0..GRID {
                            let (base, off) = (i * geom.at(), geom.ab());
                            let (src, dst) = (base + ps * off, base + s * off);
                            abuf.copy_within(src..src + off, dst);
                        }
                        continue;
                    }
                }
                geom.pack_a_slot(abuf, s, qa, k, mb * mt, c * kc);
                prev = Some((mb, c, s));
            }
            *a_state = uniform;
        }
        g.dispatch(wbuf).expect("r14 dispatch");
        let c32 = g.read_c().expect("r14 read C");
        for (s, &(mb, kchunk, tile)) in mx.plan[lo..hi].iter().enumerate() {
            let row0 = mb * mt;
            let col0 = tile * nt;
            let scol = &mx.scale4t[kchunk * mx.m..(kchunk + 1) * mx.m];
            geom.each_c_run(c32, s, |lr, lc0, vals| {
                let r = row0 + lr;
                let sxr = sx[r * mx.k_chunks + kchunk];
                let n0 = col0 + lc0;
                let sc = &scol[n0..n0 + vals.len()];
                let o = &mut out[r * m + n0..r * m + n0 + vals.len()];
                for t in 0..vals.len() {
                    o[t] += sc[t] * sxr * vals[t] as f32;
                }
            });
        }
    }
}

// ── kernel cache (anchor + LRU) under the 6-hwctx budget ────────────────────
struct KernelCache {
    anchor: NpuKernel,
    anchor_name: String,
    artifacts: HashMap<String, (Vec<u8>, Vec<u8>)>,
    live: Vec<(String, NpuKernel)>,
    capacity: usize,
    clock: u64,
    last_use: HashMap<String, u64>,
    mru: bool,
}

impl KernelCache {
    const ANCHOR: usize = usize::MAX;

    fn new(artifacts: HashMap<String, (Vec<u8>, Vec<u8>)>, anchor_name: &str, capacity: usize) -> Result<Self, XdnaError> {
        let (x, i) = artifacts.get(anchor_name).ok_or_else(|| XdnaError::BadCacheName(anchor_name.into()))?;
        let anchor = NpuKernel::load(x, i)?;
        Ok(Self {
            anchor,
            anchor_name: anchor_name.to_string(),
            artifacts,
            live: Vec::new(),
            capacity,
            clock: 0,
            last_use: HashMap::new(),
            mru: true,
        })
    }

    fn get(&mut self, name: &str) -> usize {
        if name == self.anchor_name {
            return Self::ANCHOR;
        }
        self.clock += 1;
        self.last_use.insert(name.to_string(), self.clock);
        if let Some(idx) = self.live.iter().position(|(n, _)| n == name) {
            return idx;
        }
        if self.live.len() >= self.capacity {
            let key = |n: &String| self.last_use.get(n).copied().unwrap_or(0);
            let victim = if self.mru {
                self.live.iter().enumerate().max_by_key(|(_, (n, _))| key(n)).map(|(i, _)| i)
            } else {
                self.live.iter().enumerate().min_by_key(|(_, (n, _))| key(n)).map(|(i, _)| i)
            }
            .expect("non-empty cache");
            self.live.remove(victim);
        }
        let (x, i) = self.artifacts.get(name).unwrap_or_else(|| panic!("no artifact {name}"));
        let k = NpuKernel::load_peer(&self.anchor, x, i).unwrap_or_else(|e| panic!("load_peer {name}: {e:?}"));
        self.live.push((name.to_string(), k));
        self.live.len() - 1
    }

    fn at(&self, idx: usize) -> &NpuKernel {
        if idx == Self::ANCHOR {
            return &self.anchor;
        }
        &self.live[idx].1
    }
}

// ── config + weight metadata ────────────────────────────────────────────────
#[derive(Clone, Copy)]
struct Cfg {
    h: usize,
    i_dim: usize,
    nh: usize,
    nkv: usize,
    hd: usize,
    nl: usize,
    b_rows: usize,
    l_ctx: usize,
    ne: usize,
    tot: usize,
    groups: usize,
    theta: f64,
    kernel_eps: f32,
}

/// One layer's five projections (r14) plus its four gammas.
struct Layer {
    qkv: R14Matrix,
    kv: R14Matrix,
    o: R14Matrix,
    gateup: R14Matrix,
    down: R14Matrix,
    m_qkv: usize,
    sizes_qkv: Vec<usize>,
    m_kv: usize,
    m_gu: usize,
    g_input: Vec<f32>,
    g_post: Vec<f32>,
    g_qnorm: Vec<f32>,
    g_knorm: Vec<f32>,
}

/// The loaded DFlash NPU block body, callable per draft cycle.
pub struct DflashNpuBody {
    cfg: Cfg,
    cache: KernelCache,
    r14: NpuGemmR14,
    // fc context projection
    fc: R14Matrix,
    g_hidden: Vec<f32>,
    g_final: Vec<f32>,
    layers: Vec<Layer>,
    // attention (flash) geometry + kernel name
    attn_name: String,
    fl_q_len: usize,
    fl_kv_tile: usize,
    fl_n_tiles: usize,
    fl_n_iters: usize,
    // resident scratch device buffers
    attn_q: DeviceBuffer,
    attn_kv: DeviceBuffer,
    attn_o: DeviceBuffer,
    // host staging
    qbuf: Vec<i8>,
    sxbuf: Vec<f32>,
}

impl DflashNpuBody {
    /// Load the body from the on-disk harness artifacts:
    ///   * `weights_dir` — the `--weights` dir (index.json + w_*.i8/.scale + g_*.f32)
    ///   * `manifest_path` — the `--manifest` json (xclbin/insts per op)
    ///   * `r14_dir` — the packed r14 array dir (e.g. `~/.hipfire/npu/r14_1x2x128_nb128`)
    ///
    /// Fixed to multicore + flash + CPU primitives. Loads ~3 hardware contexts
    /// (anchor + r14 + flash attn); primitives run on the host.
    pub fn load(weights_dir: &str, manifest_path: &str, r14_dir: &str) -> Result<Self, XdnaError> {
        let index: serde_json::Value =
            serde_json::from_slice(&std::fs::read(format!("{weights_dir}/index.json")).map_err(XdnaError::Open)?)
                .map_err(|e| XdnaError::BadCacheName(format!("index.json: {e}")))?;
        let c = &index["cfg"];
        let g = |k: &str| c[k].as_u64().unwrap() as usize;
        let cfg = Cfg {
            h: g("H"),
            i_dim: g("I"),
            nh: g("NH"),
            nkv: g("NKV"),
            hd: g("HD"),
            nl: g("NL"),
            b_rows: g("B"),
            l_ctx: g("L"),
            ne: g("NE"),
            tot: g("tot"),
            groups: g("groups"),
            theta: c["THETA"].as_f64().unwrap(),
            kernel_eps: c["KERNEL_EPS"].as_f64().unwrap_or(1e-5) as f32,
        };

        let manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(manifest_path).map_err(XdnaError::Open)?)
                .map_err(|e| XdnaError::BadCacheName(format!("manifest: {e}")))?;
        let mkernels = manifest["kernels"].as_object().expect("manifest kernels");
        // Multicore resolves every GEMM through the r14 array, not named kernels.

        // Flash attention kernel + geometry from the manifest.
        let attn_name = mkernels
            .keys()
            .find(|n| n.starts_with("dflash_attn_flash"))
            .expect("flash attn kernel in manifest")
            .clone();
        let ca = &mkernels[&attn_name]["compile_args"];
        let ga = |k: &str| ca[k].as_u64().unwrap_or_else(|| panic!("compile_args.{k}")) as usize;
        let (fl_q_len, fl_kv_tile, fl_n_tiles, fl_n_iters) = (ga("q_len"), ga("kv_tile"), ga("n_tiles"), ga("n_iters"));
        assert_eq!(fl_q_len % cfg.b_rows, 0, "flash q_len must be a multiple of B");
        assert!(fl_n_tiles * fl_kv_tile >= cfg.tot, "flash tiles cannot cover tot");

        let artifacts: HashMap<String, (Vec<u8>, Vec<u8>)> = mkernels
            .iter()
            .map(|(n, s)| {
                (
                    n.clone(),
                    (
                        std::fs::read(s["xclbin"].as_str().unwrap()).expect("xclbin"),
                        std::fs::read(s["insts"].as_str().unwrap()).expect("insts"),
                    ),
                )
            })
            .collect();

        // Anchor kernel: rmsnorm-b16 (pinned; owns the DRM heap even though CPU
        // primitives never dispatch it).
        let rms16 = format!("qwen35-rmsnorm-{}-b{}", cfg.h, cfg.b_rows);
        let cache = KernelCache::new(artifacts, &rms16, 4)?;
        let r14 = NpuGemmR14::load_peer_dir(&cache.anchor, r14_dir)?;

        // Weight loaders (int8 + row-scale -> r14 int4 repack).
        let load_r14 = |r14: &NpuGemmR14, key: &str, rows: usize| -> (R14Matrix, usize, Vec<usize>) {
            let spec = &index["gemms"][key];
            let (m, k) = (spec["M"].as_u64().unwrap() as usize, spec["K"].as_u64().unwrap() as usize);
            let raw = std::fs::read(format!("{weights_dir}/w_{key}.i8")).expect("weight");
            assert_eq!(raw.len(), m * k, "{key} weight size");
            let sraw = std::fs::read(format!("{weights_dir}/w_{key}.scale")).expect("scale");
            let scale: Vec<f32> = sraw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
            let sizes: Vec<usize> = spec["sizes"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
            let raw_i8 = unsafe { std::slice::from_raw_parts(raw.as_ptr() as *const i8, raw.len()) };
            let mx = R14Matrix::build(r14, raw_i8, &scale, m, k, rows).unwrap_or_else(|e| panic!("r14 build {key}: {e:?}"));
            (mx, m, sizes)
        };
        let load_gamma = |key: &str| -> Vec<f32> {
            std::fs::read(format!("{weights_dir}/g_{key}.f32"))
                .expect("gamma")
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        };

        let (fc, _fc_m, _fc_sizes) = load_r14(&r14, "fc", cfg.l_ctx);
        let g_hidden = load_gamma("hidden_norm");
        let g_final = load_gamma("final_norm");
        let mut layers = Vec::with_capacity(cfg.nl);
        for li in 0..cfg.nl {
            let (qkv, m_qkv, sizes_qkv) = load_r14(&r14, &format!("l{li}_qkv"), cfg.b_rows);
            let (kv, m_kv, _) = load_r14(&r14, &format!("l{li}_kv"), cfg.l_ctx);
            let (o, _, _) = load_r14(&r14, &format!("l{li}_o"), cfg.b_rows);
            let (gateup, m_gu, _) = load_r14(&r14, &format!("l{li}_gateup"), cfg.b_rows);
            let (down, _, _) = load_r14(&r14, &format!("l{li}_down"), cfg.b_rows);
            layers.push(Layer {
                qkv,
                kv,
                o,
                gateup,
                down,
                m_qkv,
                sizes_qkv,
                m_kv,
                m_gu,
                g_input: load_gamma(&format!("l{li}_input")),
                g_post: load_gamma(&format!("l{li}_post")),
                g_qnorm: load_gamma(&format!("l{li}_qnorm")),
                g_knorm: load_gamma(&format!("l{li}_knorm")),
            });
        }

        // Resident attention scratch (flash sizes).
        let (h, hd) = (cfg.h, cfg.hd);
        let _ = h;
        let attn_q_elems = fl_n_iters * fl_q_len * hd;
        let attn_kv_elems = fl_n_iters * fl_n_tiles * (2 * fl_kv_tile * hd + 2 * fl_kv_tile);
        let attn_q = cache.anchor.alloc_arg(attn_q_elems * 2)?;
        let attn_kv = cache.anchor.alloc_arg(attn_kv_elems * 2)?;
        let attn_o = cache.anchor.alloc_arg(attn_q_elems * 2)?;

        let max_rows = cfg.tot.max(cfg.l_ctx).max(cfg.b_rows);
        let qbuf = vec![0i8; max_rows * (cfg.ne * cfg.h).max(cfg.i_dim).max(cfg.h)];
        let sxbuf = vec![0f32; max_rows * 64];

        Ok(Self {
            cfg,
            cache,
            r14,
            fc,
            g_hidden,
            g_final,
            layers,
            attn_name,
            fl_q_len,
            fl_kv_tile,
            fl_n_tiles,
            fl_n_iters,
            attn_q,
            attn_kv,
            attn_o,
            qbuf,
            sxbuf,
        })
    }

    pub fn l_ctx(&self) -> usize {
        self.cfg.l_ctx
    }
    pub fn block_size(&self) -> usize {
        self.cfg.b_rows
    }
    pub fn hidden(&self) -> usize {
        self.cfg.h
    }
    pub fn num_extract(&self) -> usize {
        self.cfg.ne
    }

    /// One draft block forward.
    ///   * `target_hidden` : `[l_ctx, ne*h]` f32 (the committed-context window)
    ///   * `noise`         : `[block, h]` f32 (target-embedded block tokens)
    ///   * `out`           : `[block, h]` f32 (final block hidden), caller-sized
    pub fn forward_block(&mut self, target_hidden: &[f32], noise: &[f32], out: &mut [f32]) {
        let cfg = self.cfg;
        let (h, hd, nh, nkv, i_dim) = (cfg.h, cfg.hd, cfg.nh, cfg.nkv, cfg.i_dim);
        let (b_rows, l_ctx, tot, groups) = (cfg.b_rows, cfg.l_ctx, cfg.tot, cfg.groups);
        let nkd = nkv * hd;
        let eps = cfg.kernel_eps;
        let theta = cfg.theta;
        assert_eq!(target_hidden.len(), l_ctx * cfg.ne * h, "target_hidden shape");
        assert!(noise.len() >= b_rows * h, "noise too small");
        assert!(out.len() >= b_rows * h, "out too small");

        // Context projection: thp = rmsnorm(fc(target_hidden)) over l_ctx rows.
        let mut thp_raw = vec![0f32; l_ctx * h];
        run_r14(&mut self.r14, &self.fc, target_hidden, &mut thp_raw, &mut self.qbuf, &mut self.sxbuf);
        let mut thp = vec![0f32; l_ctx * h];
        cpu_rmsnorm(&thp_raw, &self.g_hidden, l_ctx, h, eps, &mut thp);

        let mut hidden = noise[..b_rows * h].to_vec();
        let mut x_norm = vec![0f32; b_rows * h];
        let mut qkv = vec![0f32; b_rows * (nh * hd + 2 * nkv * hd)];
        let mut kv_ctx = vec![0f32; l_ctx * 2 * nkv * hd];
        let mut q = vec![0f32; b_rows * nh * hd];
        let mut k_all = vec![0f32; tot * nkv * hd];
        let mut v_all = vec![0f32; tot * nkv * hd];
        let mut q_tmp = vec![0f32; b_rows * nh * hd];
        let mut k_tmp = vec![0f32; tot * nkv * hd];
        let mut ctx = vec![0f32; b_rows * nh * hd];
        let mut attn_proj = vec![0f32; b_rows * h];
        let mut gateup = vec![0f32; b_rows * 2 * i_dim];
        let mut swig = vec![0f32; b_rows * i_dim];
        let mut down = vec![0f32; b_rows * h];

        for li in 0..cfg.nl {
            let residual = hidden.clone();
            let (nq, nk) = (self.layers[li].sizes_qkv[0], self.layers[li].sizes_qkv[1]);
            let m_qkv = self.layers[li].m_qkv;
            let m_kv = self.layers[li].m_kv;
            let m_gu = self.layers[li].m_gu;

            // input rmsnorm -> concat qkv projection.
            cpu_rmsnorm(&hidden[..b_rows * h], &self.layers[li].g_input, b_rows, h, eps, &mut x_norm);
            run_r14(&mut self.r14, &self.layers[li].qkv, &x_norm, &mut qkv, &mut self.qbuf, &mut self.sxbuf);

            // context k/v from thp (recomputed every cycle — sliding window).
            run_r14(&mut self.r14, &self.layers[li].kv, &thp, &mut kv_ctx, &mut self.qbuf, &mut self.sxbuf);
            for r in 0..l_ctx {
                k_all[r * nkd..(r + 1) * nkd].copy_from_slice(&kv_ctx[r * m_kv..r * m_kv + nkd]);
                v_all[r * nkd..(r + 1) * nkd].copy_from_slice(&kv_ctx[r * m_kv + nkd..r * m_kv + 2 * nkd]);
            }

            // q rows + noise k/v rows (new every cycle).
            for r in 0..b_rows {
                q[r * nq..(r + 1) * nq].copy_from_slice(&qkv[r * m_qkv..r * m_qkv + nq]);
            }
            for r in 0..b_rows {
                let d = (l_ctx + r) * nkd;
                k_all[d..d + nkd].copy_from_slice(&qkv[r * m_qkv + nq..r * m_qkv + nq + nk]);
                v_all[d..d + nkd].copy_from_slice(&qkv[r * m_qkv + nq + nk..r * m_qkv + nq + 2 * nk]);
            }

            // headnorm + rope (CPU).
            cpu_headnorm(&q, &self.layers[li].g_qnorm, b_rows, nh, hd, eps, &mut q_tmp);
            cpu_rope(&q_tmp, b_rows, nh, hd, l_ctx, theta, &mut q);
            cpu_headnorm(&k_all, &self.layers[li].g_knorm, tot, nkv, hd, eps, &mut k_tmp);
            cpu_rope(&k_tmp, tot, nkv, hd, 0, theta, &mut k_all);

            // attention (flash) — one dispatch, kv-heads streamed.
            self.flash_attention(&q, &k_all, &v_all, &mut ctx);

            run_r14(&mut self.r14, &self.layers[li].o, &ctx, &mut attn_proj, &mut self.qbuf, &mut self.sxbuf);
            for i in 0..b_rows * h {
                hidden[i] = residual[i] + attn_proj[i];
            }

            let residual2 = hidden.clone();
            cpu_rmsnorm(&hidden[..b_rows * h], &self.layers[li].g_post, b_rows, h, eps, &mut x_norm);
            run_r14(&mut self.r14, &self.layers[li].gateup, &x_norm, &mut gateup, &mut self.qbuf, &mut self.sxbuf);
            cpu_swiglu(&gateup, m_gu, i_dim, b_rows, &mut swig[..b_rows * i_dim]);
            run_r14(&mut self.r14, &self.layers[li].down, &swig, &mut down, &mut self.qbuf, &mut self.sxbuf);
            for i in 0..b_rows * h {
                hidden[i] = residual2[i] + down[i];
            }
        }

        cpu_rmsnorm(&hidden[..b_rows * h], &self.g_final, b_rows, h, eps, &mut out[..b_rows * h]);
        let _ = groups;
    }

    /// Flash attention for one layer, ported from the harness (flash ABI).
    fn flash_attention(&mut self, q: &[f32], k_all: &[f32], v_all: &[f32], ctx: &mut [f32]) {
        let cfg = self.cfg;
        let (h, hd, nh, nkv, groups, tot, b_rows) = (cfg.h, cfg.hd, cfg.nh, cfg.nkv, cfg.groups, cfg.tot, cfg.b_rows);
        let _ = h;
        let nkd = nkv * hd;
        let (fl_q_len, fl_kv_tile, fl_n_tiles, fl_n_iters) = (self.fl_q_len, self.fl_kv_tile, self.fl_n_tiles, self.fl_n_iters);
        const MR: usize = 4;
        const MS: usize = 8;
        const MT: usize = 4;
        const MASK_NEG: f32 = -3.0e30;

        let hpi = fl_q_len / b_rows;
        // Q: per iteration an A-layout [fl_q_len, hd] of the iteration's q-head rows.
        {
            let dst = self.attn_q.as_mut_slice();
            for it in 0..fl_n_iters {
                let qbase = it * fl_q_len * hd;
                for i in 0..hpi {
                    let head = it * hpi + i;
                    for r in 0..b_rows {
                        let (qb, qi) = ((i * b_rows + r) / MR, (i * b_rows + r) % MR);
                        for d in 0..hd {
                            let (db, si) = (d / MS, d % MS);
                            let o = (qbase + ((qb * (hd / MS) + db) * MR + qi) * MS + si) * 2;
                            let src = q[r * nh * hd + head * hd + d];
                            dst[o..o + 2].copy_from_slice(&f32_to_bf16(src).to_le_bytes());
                        }
                    }
                }
            }
        }
        // KV: per iteration n_tiles of [ Kᵀ | V | mask ].
        {
            let dst = self.attn_kv.as_mut_slice();
            let tile_elems = 2 * fl_kv_tile * hd + 2 * fl_kv_tile;
            let (kb_n, ob_n) = (fl_kv_tile / MT, hd / MT);
            for it in 0..fl_n_iters {
                let kvh = (it * hpi) / groups;
                for t in 0..fl_n_tiles {
                    let base = (it * fl_n_tiles + t) * tile_elems;
                    for db in 0..(hd / MS) {
                        for kb in 0..kb_n {
                            for si in 0..MS {
                                for ti in 0..MT {
                                    let krow = t * fl_kv_tile + kb * MT + ti;
                                    let val = if krow < tot { k_all[krow * nkd + kvh * hd + db * MS + si] } else { 0.0 };
                                    let o = (base + ((db * kb_n + kb) * MS + si) * MT + ti) * 2;
                                    dst[o..o + 2].copy_from_slice(&f32_to_bf16(val).to_le_bytes());
                                }
                            }
                        }
                    }
                    let vbase = base + fl_kv_tile * hd;
                    for vb in 0..(fl_kv_tile / MS) {
                        for ob in 0..ob_n {
                            for si in 0..MS {
                                for ti in 0..MT {
                                    let krow = t * fl_kv_tile + vb * MS + si;
                                    let val = if krow < tot { v_all[krow * nkd + kvh * hd + ob * MT + ti] } else { 0.0 };
                                    let o = (vbase + ((vb * ob_n + ob) * MS + si) * MT + ti) * 2;
                                    dst[o..o + 2].copy_from_slice(&f32_to_bf16(val).to_le_bytes());
                                }
                            }
                        }
                    }
                    let mbase = (base + 2 * fl_kv_tile * hd) * 2;
                    for j in 0..fl_kv_tile {
                        let m = if t * fl_kv_tile + j < tot { 0.0f32 } else { MASK_NEG };
                        dst[mbase + j * 4..mbase + j * 4 + 4].copy_from_slice(&m.to_le_bytes());
                    }
                }
            }
        }
        let idx = self.cache.get(&self.attn_name);
        let kern = self.cache.at(idx);
        kern.dispatch(&[&self.attn_q, &self.attn_kv, &self.attn_o]).expect("attn dispatch");
        kern.sync_output(&self.attn_o).expect("sync attn");
        let src = self.attn_o.as_slice();
        // O is C-layout: (fl_q_len/MR) x (hd/MT) tiles of MR x MT.
        for it in 0..fl_n_iters {
            let obase = it * fl_q_len * hd;
            for i in 0..hpi {
                let head = it * hpi + i;
                for r in 0..b_rows {
                    let (qb, qi) = ((i * b_rows + r) / MR, (i * b_rows + r) % MR);
                    for d in 0..hd {
                        let (ob, ti) = (d / MT, d % MT);
                        let o = (obase + ((qb * (hd / MT) + ob) * MR + qi) * MT + ti) * 2;
                        ctx[r * nh * hd + head * hd + d] = bf16_to_f32(u16::from_le_bytes([src[o], src[o + 1]]));
                    }
                }
            }
        }
    }
}
