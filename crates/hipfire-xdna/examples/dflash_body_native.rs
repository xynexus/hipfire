//! DFlash native NPU driver — the 5-layer block body, dispatched from Rust.
//!
//! Replaces the Python/XRT harness (`tools/npu/dflash_body_npu.py`) on the hot
//! path with direct amdxdna DRM submission, and measures the real block
//! wall-clock. Python stays the parity reference; this is an addition.
//!
//! Inputs, all produced by the harness so both sides run identical numerics:
//!   * `--manifest`  artifact manifest (resolved xclbin/insts per op, incl. the
//!                   hash-keyed `@iron.jit` cache entries)   [--dump-manifest]
//!   * `--weights`   pre-quantized int8 GEMM weights + gammas [--dump-weights]
//!   * `--golden`    Phase-A golden dir (inputs + f16 golden block_hidden)
//!   * `--ref`       int8/bf16 precision reference             [--dump-ref]
//!
//! Two facts shape the design, both measured on nix1:
//!   1. npu1 (Phoenix) admits only SIX concurrent hardware contexts, and the
//!      body uses 12 distinct kernels per layer. So kernels are cached with a
//!      pinned anchor + LRU; a miss costs ~205 us via `load_peer` (vs ~19.5 ms
//!      via `load`, which re-opens the DRM file and a 64 MiB heap).
//!   2. Weights are ~1.09 GB of int8. They are uploaded ONCE into buffers
//!      allocated on the shared device and stay resident across every dispatch
//!      and every block — they outlive the kernels that consume them, which is
//!      what makes the LRU affordable.
//!
//! Usage (hold the hipfire lock):
//!   dflash_body_native --manifest M.json --weights DIR --golden GDIR \
//!                      --ref REF.npy [--blocks N]

#[cfg(target_os = "linux")]
#[path = "common/npy.rs"]
mod npy;

#[cfg(target_os = "linux")]
mod body {
    use hipfire_xdna::{DeviceBuffer, NpuKernel, XdnaError};
    use std::collections::HashMap;

    pub const HEAD_DIM: usize = 128;
    const QMAX: f32 = 127.0;

    // ── bf16 <-> f32 ────────────────────────────────────────────────────────
    // Round-to-nearest-even, matching ml_dtypes' bfloat16 so the native path
    // rounds identically to the numpy reference.
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

    pub fn write_bf16(buf: &mut DeviceBuffer, src: &[f32]) {
        let dst = buf.as_mut_slice();
        assert!(dst.len() >= src.len() * 2, "bf16 dst too small");
        for (i, &v) in src.iter().enumerate() {
            dst[i * 2..i * 2 + 2].copy_from_slice(&f32_to_bf16(v).to_le_bytes());
        }
    }

    pub fn read_bf16(buf: &DeviceBuffer, out: &mut [f32]) {
        let src = buf.as_slice();
        for (i, o) in out.iter_mut().enumerate() {
            *o = bf16_to_f32(u16::from_le_bytes([src[i * 2], src[i * 2 + 1]]));
        }
    }

    // ── int8 per-row symmetric quant (mirrors quantize_row_symmetric) ───────
    /// One scale per activation row over the whole K. `round_ties_even` matches
    /// numpy's half-to-even, so the int8 codes agree with the Python path.
    pub fn quantize_row(x: &[f32], rows: usize, k: usize, q: &mut [i8], scale: &mut [f32]) {
        for r in 0..rows {
            let row = &x[r * k..(r + 1) * k];
            let absmax = row.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
            let s = if absmax > 0.0 { absmax / QMAX } else { 1.0 };
            scale[r] = s;
            let inv = 1.0 / s;
            for (i, &v) in row.iter().enumerate() {
                q[r * k + i] = (v * inv).round_ties_even().clamp(-QMAX, QMAX) as i8;
            }
        }
    }

    /// rope cos/sin table for one position: [cos_0..cos_{hd/2-1}, sin_...].
    pub fn cs_buf(hd: usize, pos: f64, theta: f64, out: &mut [f32]) {
        let half = hd / 2;
        for i in 0..half {
            let freq = 1.0 / theta.powf(2.0 * i as f64 / hd as f64);
            let ang = pos * freq;
            out[i] = ang.cos() as f32;
            out[half + i] = ang.sin() as f32;
        }
    }

    pub fn cosine(a: &[f32], b: &[f32]) -> f64 {
        let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
        for i in 0..a.len().min(b.len()) {
            let (x, y) = (a[i] as f64, b[i] as f64);
            dot += x * y;
            na += x * x;
            nb += y * y;
        }
        if na == 0.0 || nb == 0.0 {
            0.0
        } else {
            dot / (na.sqrt() * nb.sqrt())
        }
    }

    // ── kernel cache ────────────────────────────────────────────────────────
    /// Kernels under a hardware-context budget. The anchor is pinned (it owns
    /// the DRM file and device heap every peer shares, and every argument
    /// buffer is allocated against it), so it must never be evicted.
    pub struct KernelCache {
        pub anchor: NpuKernel,
        anchor_name: String,
        artifacts: HashMap<String, (Vec<u8>, Vec<u8>)>,
        live: Vec<(String, NpuKernel)>,
        capacity: usize,
        clock: u64,
        last_use: HashMap<String, u64>,
        pub misses: u64,
        pub miss_ns: u64,
    }

    impl KernelCache {
        pub fn new(
            artifacts: HashMap<String, (Vec<u8>, Vec<u8>)>,
            anchor_name: &str,
            capacity: usize,
        ) -> Result<Self, XdnaError> {
            let (x, i) = artifacts.get(anchor_name).expect("anchor artifact");
            let anchor = NpuKernel::load(x, i)?;
            Ok(Self {
                anchor,
                anchor_name: anchor_name.to_string(),
                artifacts,
                live: Vec::new(),
                capacity,
                clock: 0,
                last_use: HashMap::new(),
                misses: 0,
                miss_ns: 0,
            })
        }

        /// Borrow a loaded kernel, loading (and evicting) as needed. Returns an
        /// index into `live`, not a reference, so the caller can still touch
        /// the cache; use [`Self::at`] to dispatch.
        pub fn get(&mut self, name: &str) -> usize {
            self.clock += 1;
            self.last_use.insert(name.to_string(), self.clock);
            if let Some(idx) = self.live.iter().position(|(n, _)| n == name) {
                return idx;
            }
            if self.live.len() >= self.capacity {
                // Evict least-recently-used. Dropping the kernel destroys its
                // hwctx, freeing a slot; its argument buffers are unaffected
                // (they belong to the shared device, not the context).
                let victim = self
                    .live
                    .iter()
                    .enumerate()
                    .min_by_key(|(_, (n, _))| self.last_use.get(n).copied().unwrap_or(0))
                    .map(|(i, _)| i)
                    .expect("non-empty cache");
                self.live.remove(victim);
            }
            let t0 = std::time::Instant::now();
            let (x, i) = self.artifacts.get(name).unwrap_or_else(|| panic!("no artifact {name}"));
            let k = NpuKernel::load_peer(&self.anchor, x, i)
                .unwrap_or_else(|e| panic!("load_peer {name}: {e:?}"));
            self.miss_ns += t0.elapsed().as_nanos() as u64;
            self.misses += 1;
            self.live.push((name.to_string(), k));
            self.live.len() - 1
        }

        pub fn at(&self, idx: usize) -> &NpuKernel {
            &self.live[idx].1
        }

        pub fn is_anchor(&self, name: &str) -> bool {
            name == self.anchor_name
        }
    }
}

#[cfg(target_os = "linux")]
fn main() {
    use body::*;
    use hipfire_xdna::{DeviceBuffer, NpuKernel};
    use std::collections::HashMap;
    use std::time::Instant;

    // ── args ────────────────────────────────────────────────────────────────
    let argv: Vec<String> = std::env::args().collect();
    let arg = |k: &str| -> Option<String> {
        argv.iter().position(|a| a == k).and_then(|i| argv.get(i + 1)).cloned()
    };
    let manifest_path = arg("--manifest").expect("--manifest");
    let wdir = arg("--weights").expect("--weights");
    let gdir = arg("--golden").expect("--golden");
    let refpath = arg("--ref");
    let blocks: usize = arg("--blocks").and_then(|v| v.parse().ok()).unwrap_or(3);
    let capacity: usize = arg("--ctx-budget").and_then(|v| v.parse().ok()).unwrap_or(5);

    // ── config + weights ────────────────────────────────────────────────────
    let index: serde_json::Value =
        serde_json::from_slice(&std::fs::read(format!("{wdir}/index.json")).expect("index.json"))
            .expect("parse index");
    let c = &index["cfg"];
    let g = |k: &str| c[k].as_u64().unwrap() as usize;
    let (h, i_dim, nh, nkv, hd) = (g("H"), g("I"), g("NH"), g("NKV"), g("HD"));
    let (nl, b_rows, l_ctx, ne, tot, groups) =
        (g("NL"), g("B"), g("L"), g("NE"), g("tot"), g("groups"));
    let theta = c["THETA"].as_f64().unwrap();
    let q_len = groups * b_rows;
    println!(
        "[dflash_body_native] H={h} I={i_dim} NH={nh} NKV={nkv} NL={nl} B={b_rows} L={l_ctx} tot={tot}"
    );

    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&manifest_path).expect("manifest")).expect("parse");
    let mkernels = manifest["kernels"].as_object().expect("kernels");

    // Map each body op to its manifest kernel name. GEMMs are identified by
    // their CompileTime shape, exactly as the JIT cache keys them.
    let gemm_kernel = |m: usize, k: usize, n: usize| -> String {
        mkernels
            .iter()
            .find(|(_, s)| {
                let ca = &s["compile_args"];
                ca["M"].as_u64() == Some(m as u64)
                    && ca["K"].as_u64() == Some(k as u64)
                    && ca["N"].as_u64() == Some(n as u64)
            })
            .map(|(n, _)| n.clone())
            .unwrap_or_else(|| panic!("no GEMM kernel M={m} K={k} N={n}"))
    };
    let attn_name = mkernels
        .keys()
        .find(|n| n.starts_with("dflash_attn_all"))
        .expect("attn kernel")
        .clone();

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

    // rmsnorm-b16 runs 11x per block (2/layer + the final norm) — pin it.
    let rms16 = format!("qwen35-rmsnorm-{h}-b{b_rows}");
    let rms32 = format!("qwen35-rmsnorm-{h}-b{l_ctx}");
    let hn_q = format!("qwen35-headnorm-q-{nh}h{hd}d-b{b_rows}");
    let hn_k = format!("qwen35-headnorm-k-{nkv}h{hd}d-b{tot}");
    let rope_q = format!("dflash-rope-q-{nh}h{hd}d-b{b_rows}");
    let rope_k = format!("dflash-rope-k-{nkv}h{hd}d-b{tot}");
    let swiglu = format!("qwen35-swiglu-{i_dim}-b{b_rows}");

    let t_setup = Instant::now();
    let mut cache = KernelCache::new(artifacts, &rms16, capacity).expect("kernel cache");

    // ── resident weights: uploaded ONCE, reused across dispatches + blocks ──
    struct Gemm {
        w: DeviceBuffer,
        scale: Vec<f32>,
        m: usize,
        k: usize,
        sizes: Vec<usize>,
    }
    let load_gemm = |anchor: &NpuKernel, key: &str| -> Gemm {
        let spec = &index["gemms"][key];
        let (m, k) = (
            spec["M"].as_u64().unwrap() as usize,
            spec["K"].as_u64().unwrap() as usize,
        );
        let raw = std::fs::read(format!("{wdir}/w_{key}.i8")).expect("weight");
        assert_eq!(raw.len(), m * k, "{key} weight size");
        let mut w = anchor.alloc_arg(m * k).expect("alloc weight");
        w.as_mut_slice().copy_from_slice(&raw);
        // Flush once here; the dispatch path then skips re-flushing this
        // buffer, which is the point of keeping it resident.
        anchor.sync_to_device(&w).expect("sync weight");
        let sraw = std::fs::read(format!("{wdir}/w_{key}.scale")).expect("scale");
        let scale: Vec<f32> = sraw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        Gemm {
            w,
            scale,
            m,
            k,
            sizes: spec["sizes"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect(),
        }
    };
    let load_gamma = |key: &str| -> Vec<f32> {
        std::fs::read(format!("{wdir}/g_{key}.f32"))
            .expect("gamma")
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    };

    let w_fc = load_gemm(&cache.anchor, "fc");
    let g_hidden = load_gamma("hidden_norm");
    let g_final = load_gamma("final_norm");
    let layers: Vec<(Gemm, Gemm, Gemm, Gemm, Gemm, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> = (0
        ..nl)
        .map(|li| {
            (
                load_gemm(&cache.anchor, &format!("l{li}_qkv")),
                load_gemm(&cache.anchor, &format!("l{li}_kv")),
                load_gemm(&cache.anchor, &format!("l{li}_o")),
                load_gemm(&cache.anchor, &format!("l{li}_gateup")),
                load_gemm(&cache.anchor, &format!("l{li}_down")),
                load_gamma(&format!("l{li}_input")),
                load_gamma(&format!("l{li}_post")),
                load_gamma(&format!("l{li}_qnorm")),
                load_gamma(&format!("l{li}_knorm")),
            )
        })
        .collect();
    println!(
        "  [setup] weights resident in {:.1} s",
        t_setup.elapsed().as_secs_f64()
    );

    // ── scratch argument buffers (allocated once, reused) ───────────────────
    // Every buffer belongs to the shared device via the pinned anchor, so it
    // stays valid no matter which kernels the LRU evicts.
    let a = &cache.anchor;
    let mk = |n: usize| a.alloc_arg(n).expect("alloc scratch");
    let max_gemm_m = layers.iter().map(|l| l.3.m).max().unwrap().max(w_fc.m);
    let max_rows = tot.max(l_ctx).max(b_rows);

    let mut gemm_b = mk(max_rows * (ne * h).max(i_dim).max(h)); // int8 activation
    let mut gemm_c = mk(max_gemm_m * max_rows * 4); // int32 result
    let mut norm_in = mk(max_rows * h * 2);
    let mut norm_w = mk(max_rows * h * 2);
    let mut norm_out = mk(max_rows * h * 2);
    let mut hn_in = mk(tot * nh * hd * 2);
    let mut hn_out = mk(tot * nh * hd * 2);
    let mut hn_w = mk(hd * 2);
    let mut rope_in = mk(tot * nh * hd * 2);
    let mut rope_cs = mk(tot * nh * hd * 2);
    let mut rope_out = mk(tot * nh * hd * 2);
    let mut sw_gate = mk(b_rows * i_dim * 2);
    let mut sw_up = mk(b_rows * i_dim * 2);
    let mut sw_out = mk(b_rows * i_dim * 2);
    let mut attn_q = mk(nkv * q_len * hd * 2);
    let mut attn_kv = mk(nkv * 2 * tot * hd * 2);
    let mut attn_o = mk(nkv * q_len * hd * 2);

    // Host-side staging (f32, mirroring the numpy harness exactly).
    let mut qbuf = vec![0i8; max_rows * (ne * h).max(i_dim).max(h)];
    let mut sxbuf = vec![0f32; max_rows];
    let mut csrow = vec![0f32; hd];

    // ── inputs + validation targets ─────────────────────────────────────────
    let noise = npy::read(&format!("{gdir}/noise_embedding.npy")).expect("noise").to_f32();
    let target_hidden = npy::read(&format!("{gdir}/target_hidden.npy")).expect("th").to_f32();
    let golden = npy::read(&format!("{gdir}/rust/rust_final_block_hidden.npy"))
        .expect("golden")
        .to_f32();
    let precision_ref = refpath.as_ref().map(|p| npy::read(p).expect("ref").to_f32());

    // ── probe: steady-state dispatch cost with NO context churn ─────────────
    // Separates the per-dispatch floor from the cost of re-establishing a
    // ~1 GB resident weight in a freshly created hardware context. Dispatches
    // one GEMM repeatedly on a kernel that is never evicted.
    if let Some(key) = arg("--probe-gemm") {
        let n: usize = arg("--probe-iters").and_then(|v| v.parse().ok()).unwrap_or(50);
        let gm = load_gemm(&cache.anchor, &key);
        let rows = if gm.k == ne * h { l_ctx } else { b_rows };
        let name = gemm_kernel(gm.m, gm.k, rows);
        let idx = cache.get(&name);
        let k = cache.at(idx);
        let x = vec![0.5f32; rows * gm.k];
        quantize_row(&x, rows, gm.k, &mut qbuf, &mut sxbuf);
        gemm_b.as_mut_slice()[..rows * gm.k].copy_from_slice(unsafe {
            std::slice::from_raw_parts(qbuf.as_ptr() as *const u8, rows * gm.k)
        });
        // Warm: the first dispatch on a fresh context pays the residency map.
        k.dispatch_synced(&[&gm.w, &gemm_b, &gemm_c], &[false, true, false])
            .expect("probe warm");
        let t0 = Instant::now();
        for _ in 0..n {
            k.dispatch_synced(&[&gm.w, &gemm_b, &gemm_c], &[false, true, false])
                .expect("probe");
        }
        let us = t0.elapsed().as_secs_f64() * 1e6 / n as f64;
        println!(
            "  probe {key}: M={} K={} rows={rows} weight={:.0} MB  \
             steady-state dispatch = {us:.0} us  (n={n}, no ctx churn)",
            gm.m,
            gm.k,
            (gm.m * gm.k) as f64 / 1e6
        );
        return;
    }

    // Steady-state cost of the whole-layer attention dispatch (small buffers,
    // so this isolates compute from the GEMMs' weight streaming).
    if arg("--probe-attn").is_some() {
        let n: usize = arg("--probe-iters").and_then(|v| v.parse().ok()).unwrap_or(50);
        let idx = cache.get(&attn_name);
        let k = cache.at(idx);
        k.dispatch(&[&attn_q, &attn_kv, &attn_o]).expect("warm");
        let t0 = Instant::now();
        for _ in 0..n {
            k.dispatch(&[&attn_q, &attn_kv, &attn_o]).expect("probe");
        }
        println!(
            "  probe {attn_name}: Q/KV/O = {:.0}/{:.0}/{:.0} KB  \
             steady-state dispatch = {:.0} us  (n={n}, no ctx churn)",
            (nkv * q_len * hd * 2) as f64 / 1e3,
            (nkv * 2 * tot * hd * 2) as f64 / 1e3,
            (nkv * q_len * hd * 2) as f64 / 1e3,
            t0.elapsed().as_secs_f64() * 1e6 / n as f64
        );
        return;
    }

    // ── the block body ──────────────────────────────────────────────────────
    let mut npu_ns_total: u64 = 0;
    let mut dispatches: u64 = 0;
    // Per-op accounting: which kernels the dispatch time actually goes to, and
    // how much of it lands on a freshly-(re)loaded context.
    let mut per_op: HashMap<String, (u64, u64, u64)> = HashMap::new(); // n, ns, ns_after_miss
    let mut block_out = vec![0f32; b_rows * h];

    // One GEMM: quantize rows, dispatch, rescale into `out` [rows, M].
    macro_rules! gemm {
        ($cache:expr, $gm:expr, $x:expr, $rows:expr, $out:expr) => {{
            let gm: &Gemm = $gm;
            let rows: usize = $rows;
            quantize_row($x, rows, gm.k, &mut qbuf, &mut sxbuf);
            gemm_b.as_mut_slice()[..rows * gm.k].copy_from_slice(unsafe {
                std::slice::from_raw_parts(qbuf.as_ptr() as *const u8, rows * gm.k)
            });
            let name = gemm_kernel(gm.m, gm.k, rows);
            let miss0 = $cache.misses;
            let idx = $cache.get(&name);
            let was_miss = $cache.misses > miss0;
            let k = $cache.at(idx);
            let t = Instant::now();
            // The weight was flushed once at upload; only the activation needs
            // a host->device sync, and the output is written by the NPU.
            k.dispatch_synced(&[&gm.w, &gemm_b, &gemm_c], &[false, true, false])
                .expect("gemm dispatch");
            let dt = t.elapsed().as_nanos() as u64;
            npu_ns_total += dt;
            dispatches += 1;
            let e = per_op.entry(name.clone()).or_insert((0, 0, 0));
            e.0 += 1;
            e.1 += dt;
            if was_miss {
                e.2 += dt;
            }
            k.sync_output(&gemm_c).expect("sync C");
            let c32 = unsafe {
                std::slice::from_raw_parts(gemm_c.as_slice().as_ptr() as *const i32, gm.m * rows)
            };
            // Y[r, n] = sw[n] * sx[r] * C[n, r]   (C is [M, rows])
            let out: &mut [f32] = $out;
            for n in 0..gm.m {
                let sw = gm.scale[n];
                for r in 0..rows {
                    out[r * gm.m + n] = sw * sxbuf[r] * c32[n * rows + r] as f32;
                }
            }
        }};
    }

    // rmsnorm: gamma is a TILED input, so replicate it per row.
    macro_rules! rmsnorm {
        ($cache:expr, $name:expr, $x:expr, $gamma:expr, $rows:expr, $out:expr) => {{
            let rows: usize = $rows;
            let gamma: &[f32] = $gamma;
            write_bf16(&mut norm_in, &$x[..rows * h]);
            {
                let dst = norm_w.as_mut_slice();
                for r in 0..rows {
                    for (i, &gv) in gamma.iter().enumerate() {
                        let o = (r * h + i) * 2;
                        dst[o..o + 2].copy_from_slice(&f32_to_bf16(gv).to_le_bytes());
                    }
                }
            }
            let miss0 = $cache.misses;
            let idx = $cache.get($name);
            let was_miss = $cache.misses > miss0;
            let k = $cache.at(idx);
            let t = Instant::now();
            k.dispatch(&[&norm_in, &norm_w, &norm_out]).expect("rmsnorm");
            let dt = t.elapsed().as_nanos() as u64;
            npu_ns_total += dt;
            dispatches += 1;
            let e = per_op.entry($name.to_string()).or_insert((0, 0, 0));
            e.0 += 1;
            e.1 += dt;
            if was_miss {
                e.2 += dt;
            }
            k.sync_output(&norm_out).expect("sync norm");
            read_bf16(&norm_out, &mut $out[..rows * h]);
        }};
    }

    // headnorm: arg order is in, out, weight (differs from rmsnorm!).
    macro_rules! headnorm {
        ($cache:expr, $name:expr, $x:expr, $gamma:expr, $n:expr, $out:expr) => {{
            let n: usize = $n;
            write_bf16(&mut hn_in, &$x[..n]);
            write_bf16(&mut hn_w, $gamma);
            let miss0 = $cache.misses;
            let idx = $cache.get($name);
            let was_miss = $cache.misses > miss0;
            let k = $cache.at(idx);
            let t = Instant::now();
            k.dispatch(&[&hn_in, &hn_out, &hn_w]).expect("headnorm");
            let dt = t.elapsed().as_nanos() as u64;
            npu_ns_total += dt;
            dispatches += 1;
            let e = per_op.entry($name.to_string()).or_insert((0, 0, 0));
            e.0 += 1;
            e.1 += dt;
            if was_miss {
                e.2 += dt;
            }
            k.sync_output(&hn_out).expect("sync hn");
            read_bf16(&hn_out, &mut $out[..n]);
        }};
    }

    // rope: cs is a tiled second input — each head-tile gets its row's table.
    macro_rules! rope {
        ($cache:expr, $name:expr, $x:expr, $rows:expr, $heads:expr, $pos0:expr, $out:expr) => {{
            let rows: usize = $rows;
            let heads: usize = $heads;
            let n = rows * heads * hd;
            write_bf16(&mut rope_in, &$x[..n]);
            {
                let dst = rope_cs.as_mut_slice();
                for r in 0..rows {
                    cs_buf(hd, ($pos0 + r) as f64, theta, &mut csrow);
                    for hh in 0..heads {
                        for (i, &cv) in csrow.iter().enumerate() {
                            let o = ((r * heads + hh) * hd + i) * 2;
                            dst[o..o + 2].copy_from_slice(&f32_to_bf16(cv).to_le_bytes());
                        }
                    }
                }
            }
            let miss0 = $cache.misses;
            let idx = $cache.get($name);
            let was_miss = $cache.misses > miss0;
            let k = $cache.at(idx);
            let t = Instant::now();
            k.dispatch(&[&rope_in, &rope_cs, &rope_out]).expect("rope");
            let dt = t.elapsed().as_nanos() as u64;
            npu_ns_total += dt;
            dispatches += 1;
            let e = per_op.entry($name.to_string()).or_insert((0, 0, 0));
            e.0 += 1;
            e.1 += dt;
            if was_miss {
                e.2 += dt;
            }
            k.sync_output(&rope_out).expect("sync rope");
            read_bf16(&rope_out, &mut $out[..n]);
        }};
    }

    let mut wall_cold = 0f64;
    let mut walls = Vec::new();

    for blk in 0..blocks {
        let t_block = Instant::now();
        let (d0, m0) = (dispatches, cache.misses);
        let n0 = npu_ns_total;

        // ── one-time context projection: thp = hidden_norm(fc(target_hidden))
        let mut thp_raw = vec![0f32; l_ctx * h];
        gemm!(cache, &w_fc, &target_hidden, l_ctx, &mut thp_raw);
        let mut thp = vec![0f32; l_ctx * h];
        rmsnorm!(cache, &rms32, thp_raw, &g_hidden, l_ctx, thp);

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

        for li in 0..nl {
            let (gm_qkv, gm_kv, gm_o, gm_gu, gm_dn, g_in, g_post, g_qn, g_kn) = (
                &layers[li].0,
                &layers[li].1,
                &layers[li].2,
                &layers[li].3,
                &layers[li].4,
                &layers[li].5,
                &layers[li].6,
                &layers[li].7,
                &layers[li].8,
            );
            let residual = hidden.clone();

            // input_layernorm -> concat q/k/v projection (one GEMM)
            rmsnorm!(cache, &rms16, hidden, g_in, b_rows, x_norm);
            gemm!(cache, gm_qkv, &x_norm, b_rows, &mut qkv);
            // k_ctx/v_ctx from thp (one GEMM)
            gemm!(cache, gm_kv, &thp, l_ctx, &mut kv_ctx);

            // split + assemble k/v as [ctx rows ; noise rows]
            let (nq, nk) = (gm_qkv.sizes[0], gm_qkv.sizes[1]);
            let m_qkv = gm_qkv.m;
            for r in 0..b_rows {
                q[r * nq..(r + 1) * nq].copy_from_slice(&qkv[r * m_qkv..r * m_qkv + nq]);
            }
            let m_kv = gm_kv.m;
            let nkd = nkv * hd;
            for r in 0..l_ctx {
                k_all[r * nkd..(r + 1) * nkd].copy_from_slice(&kv_ctx[r * m_kv..r * m_kv + nkd]);
                v_all[r * nkd..(r + 1) * nkd]
                    .copy_from_slice(&kv_ctx[r * m_kv + nkd..r * m_kv + 2 * nkd]);
            }
            for r in 0..b_rows {
                let d = (l_ctx + r) * nkd;
                k_all[d..d + nkd].copy_from_slice(&qkv[r * m_qkv + nq..r * m_qkv + nq + nk]);
                v_all[d..d + nkd]
                    .copy_from_slice(&qkv[r * m_qkv + nq + nk..r * m_qkv + nq + 2 * nk]);
            }

            // headnorm + rope, q and k
            headnorm!(cache, &hn_q, q, g_qn, b_rows * nh * hd, q_tmp);
            rope!(cache, &rope_q, q_tmp, b_rows, nh, l_ctx, q);
            headnorm!(cache, &hn_k, k_all, g_kn, tot * nkv * hd, k_tmp);
            rope!(cache, &rope_k, k_tmp, tot, nkv, 0, k_all);

            // attention: whole layer in ONE dispatch, kv-heads streamed.
            // Pack Q per kv-head as the `groups` q-heads' rows stacked.
            {
                let dst = attn_q.as_mut_slice();
                for kvh in 0..nkv {
                    for i in 0..groups {
                        let head = kvh * groups + i;
                        for r in 0..b_rows {
                            for d in 0..hd {
                                let src = q[r * nh * hd + head * hd + d];
                                let o = ((kvh * q_len + i * b_rows + r) * hd + d) * 2;
                                dst[o..o + 2].copy_from_slice(&f32_to_bf16(src).to_le_bytes());
                            }
                        }
                    }
                }
                let dst = attn_kv.as_mut_slice();
                for kvh in 0..nkv {
                    let base = kvh * 2 * tot * hd;
                    for t in 0..tot {
                        for d in 0..hd {
                            let o = (base + t * hd + d) * 2;
                            dst[o..o + 2].copy_from_slice(
                                &f32_to_bf16(k_all[t * nkd + kvh * hd + d]).to_le_bytes(),
                            );
                            let o2 = (base + tot * hd + t * hd + d) * 2;
                            dst[o2..o2 + 2].copy_from_slice(
                                &f32_to_bf16(v_all[t * nkd + kvh * hd + d]).to_le_bytes(),
                            );
                        }
                    }
                }
                let miss0 = cache.misses;
                let idx = cache.get(&attn_name);
                let was_miss = cache.misses > miss0;
                let kern = cache.at(idx);
                let t = Instant::now();
                kern.dispatch(&[&attn_q, &attn_kv, &attn_o]).expect("attn");
                let dt = t.elapsed().as_nanos() as u64;
                npu_ns_total += dt;
                dispatches += 1;
                let e = per_op.entry(attn_name.clone()).or_insert((0, 0, 0));
                e.0 += 1;
                e.1 += dt;
                if was_miss {
                    e.2 += dt;
                }
                kern.sync_output(&attn_o).expect("sync attn");
                let src = attn_o.as_slice();
                for kvh in 0..nkv {
                    for i in 0..groups {
                        let head = kvh * groups + i;
                        for r in 0..b_rows {
                            for d in 0..hd {
                                let o = ((kvh * q_len + i * b_rows + r) * hd + d) * 2;
                                ctx[r * nh * hd + head * hd + d] =
                                    bf16_to_f32(u16::from_le_bytes([src[o], src[o + 1]]));
                            }
                        }
                    }
                }
            }

            gemm!(cache, gm_o, &ctx, b_rows, &mut attn_proj);
            for i in 0..b_rows * h {
                hidden[i] = residual[i] + attn_proj[i];
            }

            let residual2 = hidden.clone();
            rmsnorm!(cache, &rms16, hidden, g_post, b_rows, x_norm);
            gemm!(cache, gm_gu, &x_norm, b_rows, &mut gateup);
            // swiglu over the [gate | up] halves of the concat GEMM output
            {
                let m_gu = gm_gu.m;
                let dg = sw_gate.as_mut_slice();
                for r in 0..b_rows {
                    for i in 0..i_dim {
                        let o = (r * i_dim + i) * 2;
                        dg[o..o + 2]
                            .copy_from_slice(&f32_to_bf16(gateup[r * m_gu + i]).to_le_bytes());
                    }
                }
                let du = sw_up.as_mut_slice();
                for r in 0..b_rows {
                    for i in 0..i_dim {
                        let o = (r * i_dim + i) * 2;
                        du[o..o + 2].copy_from_slice(
                            &f32_to_bf16(gateup[r * m_gu + i_dim + i]).to_le_bytes(),
                        );
                    }
                }
                let miss0 = cache.misses;
                let idx = cache.get(&swiglu);
                let was_miss = cache.misses > miss0;
                let kern = cache.at(idx);
                let t = Instant::now();
                kern.dispatch(&[&sw_gate, &sw_up, &sw_out]).expect("swiglu");
                let dt = t.elapsed().as_nanos() as u64;
                npu_ns_total += dt;
                dispatches += 1;
                let e = per_op.entry(swiglu.clone()).or_insert((0, 0, 0));
                e.0 += 1;
                e.1 += dt;
                if was_miss {
                    e.2 += dt;
                }
                kern.sync_output(&sw_out).expect("sync swiglu");
                read_bf16(&sw_out, &mut swig[..b_rows * i_dim]);
            }
            gemm!(cache, gm_dn, &swig, b_rows, &mut down);
            for i in 0..b_rows * h {
                hidden[i] = residual2[i] + down[i];
            }
        }

        rmsnorm!(cache, &rms16, hidden, &g_final, b_rows, block_out);

        let wall = t_block.elapsed().as_secs_f64();
        let (nd, nm) = (dispatches - d0, cache.misses - m0);
        let npu_ms = (npu_ns_total - n0) as f64 / 1e6;
        println!(
            "  block {blk}: wall={:.1} ms  dispatches={nd}  npu_busy={npu_ms:.1} ms  ctx_misses={nm}",
            wall * 1e3
        );
        if blk == 0 {
            wall_cold = wall;
        } else {
            walls.push(wall);
        }
    }

    // ── validation ──────────────────────────────────────────────────────────
    let cos_golden = cosine(&block_out, &golden);
    println!("\n  final block_hidden:");
    println!("    cos vs golden        = {cos_golden:.6}");
    let cos_ref = precision_ref.as_ref().map(|r| cosine(&block_out, r));
    if let Some(cr) = cos_ref {
        println!("    cos vs int8/bf16 ref = {cr:.6}");
    }

    let warm = walls.iter().cloned().fold(f64::INFINITY, f64::min);
    let per_block_dispatches = dispatches as f64 / blocks as f64;
    println!("\n  wall (cold) = {:.1} ms", wall_cold * 1e3);
    if warm.is_finite() {
        println!(
            "  wall (warm) = {:.1} ms   [budget: <57 ms/block]",
            warm * 1e3
        );
        println!(
            "    dispatches/block = {per_block_dispatches:.0}   per-dispatch mean = {:.0} us",
            warm * 1e6 / per_block_dispatches
        );
    }
    println!(
        "    ctx misses = {} total, {:.1} ms in load_peer",
        cache.misses,
        cache.miss_ns as f64 / 1e6
    );

    println!("\n  per-op dispatch time (total over {blocks} blocks):");
    let mut ops: Vec<_> = per_op.iter().collect();
    ops.sort_by_key(|(_, v)| std::cmp::Reverse(v.1));
    for (name, (n, ns, ns_miss)) in ops {
        println!(
            "    {:44} n={n:3}  mean={:8.2} ms  (post-ctx-miss share {:.0}%)",
            name,
            *ns as f64 / 1e6 / *n as f64,
            if *ns > 0 { *ns_miss as f64 / *ns as f64 * 100.0 } else { 0.0 }
        );
    }

    let gate = cos_golden > 0.99 && cos_ref.map(|c| c > 0.99).unwrap_or(true);
    println!(
        "\n=== NATIVE BODY: {} (need cos_golden>0.99 AND cos_ref>0.99) ===",
        if gate { "PARITY MET" } else { "PARITY NOT MET" }
    );
    if !gate {
        std::process::exit(1);
    }
}

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("Linux-only");
}
