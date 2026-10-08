// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! Should the 4.25-bit compact format stay a UNIFORM int4 grid, now that the
//! prefill GEMM may decode weights at staging time?
//!
//! On gfx1151 an f16/bf16 WMMA costs the same 32 cycles as iu8 (only iu4 is
//! faster), so a staging-decode GEMM can feed f16 weights decoded through a
//! lookup table at no extra matrix cost. That frees the weight grid: a
//! non-uniform codebook (NF4, a table learned per tensor, FP4 E2M1) becomes as
//! cheap in prefill as uniform int4, and decode already pays a lookup per byte
//! when it wants one. This scores those grids against the shipped overlay at
//! matched bits, after the same FWHT-256 the packer applies.
//!
//! Arms (bits/weight):
//!   int4 G256, clip-searched scale                         4.0625
//!   int4 + overlay-3 (shipped; scale searched jointly)      4.25
//!   int4 + overlay-1 + per-row W5 promotion                 ~4.25
//!   NF4 (Gaussian Lloyd-Max) G256                           4.0625
//!   NF4 + overlay-3                                         4.25
//!   learned CB16 per tensor G256                            4.0625 (+ a 64 B table)
//!   learned CB16 + overlay-3                                4.25
//!   learned CB16 + overlay-1 + per-row CB32 promotion       ~4.25
//!   FP4 E2M1, E4M3 scale per 32 (NVFP4-like, tensor f32)    4.25
//!   int4, E4M3 scale per 32                                 4.25
//! Overlay entries are int8 at the grid's int4-equivalent step (range repair, as
//! shipped). Weight SSE relative to row energy; capture = share of the shipped
//! overlay's reduction from plain int4. SSE is not KLD: a winner here earns a
//! fake-quant KLD run, nothing more.
//!
//!   cargo run --release -p hipfire-quantize --example codebook_format_study \
//!     -- <model.safetensors> [rows]

use hipfire_quantize::{cpu_fwht_256, gen_fwht_signs};

const G: usize = 256;

/// Scale multipliers searched for every arm (of a per-arm base scale).
fn grid() -> impl Iterator<Item = f32> {
    (0..46).map(|i| 0.30 + 0.02 * i as f32)
}

fn q_uniform(v: f32, s: f32, lim: f32) -> f32 {
    (v / s).round().clamp(-lim, lim) * s
}

/// Nearest codebook entry (cb sorted ascending), scaled.
fn q_cb(v: f32, s: f32, cb: &[f32]) -> f32 {
    let x = v / s;
    let i = cb.partition_point(|&c| c < x);
    let best = if i == 0 {
        cb[0]
    } else if i == cb.len() {
        cb[cb.len() - 1]
    } else if (x - cb[i - 1]).abs() <= (cb[i] - x).abs() {
        cb[i - 1]
    } else {
        cb[i]
    };
    best * s
}

/// Indices of the `n` largest |v|.
fn top(g: &[f32], n: usize) -> Vec<usize> {
    let mut o: Vec<usize> = (0..g.len()).collect();
    o.sort_by(|&i, &j| g[j].abs().partial_cmp(&g[i].abs()).unwrap());
    o.truncate(n);
    o
}

/// Best SSE over the scale grid for a group quantized by `q(v, s)`, with the
/// `n_ov` largest positions carried as int8 at step `s * step_of_s`.
fn search(g: &[f32], base: f32, n_ov: usize, step_of_s: f32, q: impl Fn(f32, f32) -> f32) -> f64 {
    let ov = top(g, n_ov);
    let mut on = [false; G];
    for &p in &ov {
        on[p] = true;
    }
    let mut best = f64::INFINITY;
    for m in grid() {
        let s = base * m;
        if s <= 0.0 {
            continue;
        }
        let mut e = 0f64;
        for (i, &v) in g.iter().enumerate() {
            let r = if on[i] {
                q_uniform(v, s * step_of_s, 127.0)
            } else {
                q(v, s)
            };
            e += ((v - r) as f64).powi(2);
        }
        best = best.min(e);
    }
    best
}

/// Lloyd-Max levels for a 1-D sample.
fn lloyd(sample: &[f32], n: usize) -> Vec<f32> {
    let mut s = sample.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut cb: Vec<f32> = (0..n).map(|i| s[(2 * i + 1) * s.len() / (2 * n)]).collect();
    for _ in 0..30 {
        let mut sum = vec![0f64; n];
        let mut cnt = vec![0usize; n];
        for &v in &s {
            let i = cb.partition_point(|&c| c < v);
            let k = if i == 0 {
                0
            } else if i == n || (v - cb[i - 1]).abs() <= (cb[i] - v).abs() {
                i - 1
            } else {
                i
            };
            sum[k] += v as f64;
            cnt[k] += 1;
        }
        for k in 0..n {
            if cnt[k] > 0 {
                cb[k] = (sum[k] / cnt[k] as f64) as f32;
            }
        }
        cb.sort_by(|a, b| a.partial_cmp(b).unwrap());
    }
    cb
}

/// Round a positive scale to FP8 E4M3 (bias 7, max 448, subnormal step 2^-9).
fn e4m3(x: f32) -> f32 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 448.0 {
        return 448.0;
    }
    let e = x.log2().floor() as i32;
    if e < -6 {
        return (x / 2f32.powi(-9)).round() * 2f32.powi(-9);
    }
    let m = (x / 2f32.powi(e) * 8.0).round() / 8.0;
    m * 2f32.powi(e)
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let Some(path) = a.get(1).cloned() else {
        eprintln!("usage: codebook_format_study <model.safetensors> [rows]");
        std::process::exit(2);
    };
    let rows_cap: usize = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(256);
    let file = std::fs::File::open(&path).expect("open");
    let mmap = unsafe { memmap2::Mmap::map(&file).expect("mmap") };
    let hlen = u64::from_le_bytes(mmap[0..8].try_into().unwrap()) as usize;
    let header: serde_json::Value = serde_json::from_slice(&mmap[8..8 + hlen]).expect("hdr");
    let base = 8 + hlen;
    let obj = header.as_object().expect("obj");
    let (s1, s2) = (gen_fwht_signs(42, G), gen_fwht_signs(1042, G));

    // Gaussian Lloyd-Max levels (NF4/NF5), from a fixed Box-Muller sample.
    let mut st = 0x1234_5678u64;
    let mut u = move || {
        st = st
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((st >> 11) as f64 / (1u64 << 53) as f64).max(1e-12)
    };
    let gauss: Vec<f32> = (0..400_000)
        .map(|_| ((-2.0 * u().ln()).sqrt() * (std::f64::consts::TAU * u()).cos()) as f32)
        .collect();
    let norm = |cb: Vec<f32>| {
        let m = cb.iter().fold(0f32, |m, v| m.max(v.abs()));
        cb.into_iter().map(|v| v / m).collect::<Vec<f32>>()
    };
    let nf4 = norm(lloyd(&gauss, 16));
    let nf5 = norm(lloyd(&gauss, 32));
    const E2M1: [f32; 15] = [
        -6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    ];

    println!("codebook format study: G={G}, <= {rows_cap} rows, after FWHT-256");
    println!("capture = share of the shipped overlay's SSE reduction from int4\n");
    for sfx in [
        "mlp.gate_proj.weight",
        "mlp.down_proj.weight",
        "self_attn.o_proj.weight",
        "linear_attn.out_proj.weight",
    ] {
        let Some(name) = obj.keys().filter(|n| n.ends_with(sfx)).min() else {
            continue;
        };
        let info = &obj[name];
        let sh: Vec<usize> = info["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        if sh.len() != 2 || sh[1] % G != 0 || info["dtype"].as_str() != Some("BF16") {
            continue;
        }
        let (nrows, k) = (sh[0].min(rows_cap), sh[1]);
        let ng = k / G;
        let start = base + info["data_offsets"][0].as_u64().unwrap() as usize;
        let rows: Vec<Vec<f32>> = (0..nrows)
            .map(|r| {
                let mut row: Vec<f32> = (0..k)
                    .map(|c| {
                        let i = start + (r * k + c) * 2;
                        f32::from_bits((u16::from_le_bytes([mmap[i], mmap[i + 1]]) as u32) << 16)
                    })
                    .collect();
                for ch in row.chunks_mut(G) {
                    cpu_fwht_256(ch, &s1, &s2);
                }
                row
            })
            .collect();
        let energy: f64 = rows
            .iter()
            .flatten()
            .map(|&v| (v as f64) * (v as f64))
            .sum();
        let amax = |g: &[f32]| g.iter().fold(0f32, |m, v| m.max(v.abs()));
        // Learned tables: Lloyd on group-absmax-normalized values of this tensor.
        let normed: Vec<f32> = rows
            .iter()
            .flat_map(|r| {
                r.chunks(G).flat_map(move |g| {
                    let a = amax(g).max(1e-30);
                    g.iter().map(move |v| v / a)
                })
            })
            .collect();
        let cb16 = norm(lloyd(&normed, 16));
        let cb32 = norm(lloyd(&normed, 32));
        // E4M3 per-32 scales need a per-tensor f32 so the block scales fit E4M3.
        let gmax = rows.iter().flatten().fold(0f32, |m, v| m.max(v.abs()));
        let tensor_e2m1 = gmax / 6.0 / 448.0;
        let tensor_i4 = gmax / 7.0 / 448.0;

        // Per row, per group: SSE per arm, and per-row promotion gains.
        let mut tot = [0f64; 13];
        for row in &rows {
            let mut g_i4o1 = Vec::with_capacity(ng); // base: int4 + overlay-1
            let mut g_w5 = Vec::with_capacity(ng); // promoted: W5
            let mut g_cbo1 = Vec::with_capacity(ng); // base: CB16 + overlay-1
            let mut g_cb32 = Vec::with_capacity(ng); // promoted: CB32
            let mut g_nf4 = Vec::with_capacity(ng); // base: NF4, no overlay
            let mut g_nf5 = Vec::with_capacity(ng); // promoted: NF5
            for g in row.chunks(G) {
                let a = amax(g);
                if a <= 0.0 {
                    for v in [
                        &mut g_i4o1,
                        &mut g_w5,
                        &mut g_cbo1,
                        &mut g_cb32,
                        &mut g_nf4,
                        &mut g_nf5,
                    ] {
                        v.push(0.0);
                    }
                    continue;
                }
                let s4 = a / 7.0;
                let u4 = |v: f32, s: f32| q_uniform(v, s, 7.0);
                tot[0] += search(g, s4, 0, 1.0, u4);
                tot[1] += search(g, s4, 3, 1.0, u4);
                g_i4o1.push(search(g, s4, 1, 1.0, u4));
                g_w5.push(search(g, a / 15.0, 0, 1.0, |v, s| q_uniform(v, s, 15.0)));
                let nf4_sse = search(g, a, 0, 1.0, |v, s| q_cb(v, s, &nf4));
                tot[3] += nf4_sse;
                g_nf4.push(nf4_sse);
                g_nf5.push(search(g, a, 0, 1.0, |v, s| q_cb(v, s, &nf5)));
                // NF4 with a scale per 128 (FWHT stays 256).
                for h in g.chunks(128) {
                    let ha = amax(h);
                    if ha > 0.0 {
                        tot[12] += search(h, ha, 0, 1.0, |v, s| q_cb(v, s, &nf4));
                    }
                }
                tot[4] += search(g, a, 3, 1.0 / 7.0, |v, s| q_cb(v, s, &nf4));
                tot[5] += search(g, a, 0, 1.0, |v, s| q_cb(v, s, &cb16));
                tot[6] += search(g, a, 3, 1.0 / 7.0, |v, s| q_cb(v, s, &cb16));
                g_cbo1.push(search(g, a, 1, 1.0 / 7.0, |v, s| q_cb(v, s, &cb16)));
                g_cb32.push(search(g, a, 0, 1.0, |v, s| q_cb(v, s, &cb32)));
                // Microscaled: per 32-block E4M3 scale (searched, then rounded).
                for blk in g.chunks(32) {
                    let ba = amax(blk);
                    if ba <= 0.0 {
                        continue;
                    }
                    let fp4 = |v: f32, s: f32| {
                        let s = e4m3(s / tensor_e2m1) * tensor_e2m1;
                        if s <= 0.0 {
                            0.0
                        } else {
                            q_cb(v, s, &E2M1)
                        }
                    };
                    tot[8] += search(blk, ba / 6.0, 0, 1.0, fp4);
                    let i4 = |v: f32, s: f32| {
                        let s = e4m3(s / tensor_i4) * tensor_i4;
                        if s <= 0.0 {
                            0.0
                        } else {
                            q_uniform(v, s, 7.0)
                        }
                    };
                    tot[9] += search(blk, ba / 7.0, 0, 1.0, i4);
                }
            }
            // Per-row promotion of the R groups with the largest gain.
            let promote = |base: &[f64], prom: &[f64], r_n: usize| -> f64 {
                let mut gain: Vec<f64> = base.iter().zip(prom).map(|(b, p)| b - p).collect();
                gain.sort_by(|x, y| y.partial_cmp(x).unwrap());
                base.iter().sum::<f64>() - gain[..r_n.min(gain.len())].iter().sum::<f64>()
            };
            let r_n = ((0.125 * k as f64 / 256.0).round() as usize).max(1);
            tot[2] += promote(&g_i4o1, &g_w5, r_n);
            tot[7] += promote(&g_cbo1, &g_cb32, r_n);
            // Gather-free: NF4 bulk, the overlay's bits spent on per-row NF5 groups.
            let r_full = ((0.1875 * k as f64 / 256.0).round() as usize).max(1);
            tot[10] += promote(&g_nf4, &g_nf5, r_full);
            tot[11] += promote(&g_nf4, &g_nf5, 2 * r_full);
        }
        let r_n = ((0.125 * k as f64 / 256.0).round() as usize).max(1);
        let promo_bw = 4.0625 + 0.0625 + r_n as f64 * 256.0 / k as f64;
        let r_full = ((0.1875 * k as f64 / 256.0).round() as usize).max(1);
        let nf_bw = |r: usize| 4.0625 + r as f64 * 256.0 / k as f64;
        let cap = |x: f64| 100.0 * (tot[0] - x) / (tot[0] - tot[1]);
        println!(
            "  {} (K={k}, {nrows} rows)",
            sfx.trim_end_matches(".weight")
        );
        println!(
            "    {:<40} {:>7} {:>10} {:>8}",
            "arm", "b/w", "rel SSE", "capture"
        );
        let labels = [
            ("int4 G256", 4.0625),
            ("int4 + overlay-3 (shipped)", 4.25),
            ("int4 + overlay-1 + per-row W5", promo_bw),
            ("NF4 G256", 4.0625),
            ("NF4 + overlay-3", 4.25),
            ("learned CB16 G256", 4.0625),
            ("learned CB16 + overlay-3", 4.25),
            ("learned CB16 + overlay-1 + per-row CB32", promo_bw),
            ("FP4 E2M1, E4M3 scale /32", 4.25),
            ("int4, E4M3 scale /32", 4.25),
            ("NF4 + per-row NF5 (no overlay, gather-free)", nf_bw(r_full)),
            ("NF4 + per-row NF5, 2x the promotions", nf_bw(2 * r_full)),
            ("NF4, scale per 128", 4.125),
        ];
        for (i, (l, bw)) in labels.iter().enumerate() {
            println!(
                "    {:<40} {:>7.3} {:>10.3e} {:>7.1}%",
                l,
                bw,
                tot[i] / energy,
                cap(tot[i])
            );
        }
    }
}
