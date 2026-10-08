// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! Fake-quantize a bf16 safetensors checkpoint in a candidate WEIGHT FORMAT, so
//! two formats can be scored by KLD through the same bf16 path before either is
//! built: every tensor the compact 4.25-bit quantizer quantizes (the layers'
//! linear weights and lm_head) is FWHT-rotated per 256-group, quantized and
//! dequantized in the chosen format, rotated back, and written as bf16. Everything
//! else is copied. Import the result with `hipfire import safetensors`.
//!
//! Formats (both ~4.25 bits/weight, both with the packer's FWHT-256 and a scale
//! searched per 256-group; neither applies AWQ or LDLQ, so absolute KLD is worse
//! than the shipped artifact's -- the comparison between them is the point):
//!   shipped   int4 [-7,7] + the 3 largest positions as int8 at the group's step,
//!             scale searched jointly with them (the oq4.25 overlay)
//!   nf4nf5    NF4 (Gaussian Lloyd-Max levels), plus per row the R groups whose
//!             error a 32-level NF5 table cuts most, R = round(0.1875 K / 256):
//!             no overlay, no index, gather-free
//!
//!   cargo run --release -p hipfire-quantize --example fakequant_format -- \
//!     <in_safetensors_dir> <out_dir> <shipped|nf4nf5>

use hipfire_quantize::{cpu_fwht_256, gen_fwht_signs};
use rayon::prelude::*;

const G: usize = 256;
const PROJ: [&str; 12] = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_a",
    "in_proj_b",
    "out_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
];

fn quantized(name: &str) -> bool {
    name == "lm_head.weight"
        || (name.contains("language_model.layers.")
            && name.ends_with(".weight")
            && PROJ.iter().any(|p| name.ends_with(&format!(".{p}.weight"))))
}

fn bf16_to_f32(b: [u8; 2]) -> f32 {
    f32::from_bits((u16::from_le_bytes(b) as u32) << 16)
}

fn f32_to_bf16(v: f32) -> [u8; 2] {
    let x = v.to_bits();
    let r = x.wrapping_add(0x7fff + ((x >> 16) & 1)); // round to nearest even
    ((r >> 16) as u16).to_le_bytes()
}

fn grid() -> impl Iterator<Item = f32> {
    (0..46).map(|i| 0.30 + 0.02 * i as f32)
}

fn q_cb(x: f32, cb: &[f32]) -> f32 {
    let i = cb.partition_point(|&c| c < x);
    if i == 0 {
        cb[0]
    } else if i == cb.len() || (x - cb[i - 1]).abs() <= (cb[i] - x).abs() {
        cb[i - 1]
    } else {
        cb[i]
    }
}

/// Best reconstruction of `g` over the scale grid: (sse, recon).
fn best(g: &[f32], base: f32, q: impl Fn(f32, f32, usize) -> f32) -> (f64, Vec<f32>) {
    let mut out = (f64::INFINITY, vec![0f32; g.len()]);
    for m in grid() {
        let s = base * m;
        let r: Vec<f32> = g.iter().enumerate().map(|(i, &v)| q(v, s, i)).collect();
        let e: f64 = g
            .iter()
            .zip(&r)
            .map(|(a, b)| ((a - b) as f64).powi(2))
            .sum();
        if e < out.0 {
            out = (e, r);
        }
    }
    out
}

fn lloyd(sample: &[f32], n: usize) -> Vec<f32> {
    let mut s = sample.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut cb: Vec<f32> = (0..n).map(|i| s[(2 * i + 1) * s.len() / (2 * n)]).collect();
    for _ in 0..30 {
        let (mut sum, mut cnt) = (vec![0f64; n], vec![0usize; n]);
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
    let m = cb.iter().fold(0f32, |m, v| m.max(v.abs()));
    cb.into_iter().map(|v| v / m).collect()
}

/// One row (already rotated, f32) quantized and dequantized in place.
fn fake_row(row: &mut [f32], arm: &str, nf4: &[f32], nf5: &[f32]) {
    let amax = |g: &[f32]| g.iter().fold(0f32, |m, v| m.max(v.abs()));
    match arm {
        "shipped" => {
            for g in row.chunks_mut(G) {
                let a = amax(g);
                if a <= 0.0 {
                    continue;
                }
                let mut ord: Vec<usize> = (0..G).collect();
                ord.sort_by(|&i, &j| g[j].abs().partial_cmp(&g[i].abs()).unwrap());
                let mut on = [false; G];
                for &p in &ord[..3] {
                    on[p] = true;
                }
                let (_, r) = best(g, a / 7.0, |v, s, i| {
                    let lim = if on[i] { 127.0 } else { 7.0 };
                    (v / s).round().clamp(-lim, lim) * s
                });
                g.copy_from_slice(&r);
            }
        }
        "nf4nf5" => {
            let k = row.len();
            let r_n = ((0.1875 * k as f64 / G as f64).round() as usize).max(1);
            let mut cand: Vec<(f64, Vec<f32>, Vec<f32>)> = row
                .chunks(G)
                .map(|g| {
                    let a = amax(g);
                    if a <= 0.0 {
                        return (0.0, g.to_vec(), g.to_vec());
                    }
                    let (e4, r4) = best(g, a, |v, s, _| q_cb(v / s, nf4) * s);
                    let (e5, r5) = best(g, a, |v, s, _| q_cb(v / s, nf5) * s);
                    (e4 - e5, r4, r5)
                })
                .collect();
            let mut ord: Vec<usize> = (0..cand.len()).collect();
            ord.sort_by(|&i, &j| cand[j].0.partial_cmp(&cand[i].0).unwrap());
            let promoted: std::collections::HashSet<usize> =
                ord[..r_n.min(ord.len())].iter().copied().collect();
            for (gi, g) in row.chunks_mut(G).enumerate() {
                let c = &mut cand[gi];
                g.copy_from_slice(if promoted.contains(&gi) { &c.2 } else { &c.1 });
            }
        }
        other => panic!("unknown arm {other}"),
    }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() != 4 {
        eprintln!("usage: fakequant_format <in_safetensors_dir> <out_dir> <shipped|nf4nf5>");
        std::process::exit(2);
    }
    let (indir, outdir, arm) = (
        std::path::Path::new(&a[1]),
        std::path::Path::new(&a[2]),
        a[3].as_str(),
    );
    std::fs::create_dir_all(outdir).expect("out dir");

    // Gaussian Lloyd-Max levels, from a fixed Box-Muller sample.
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
    let (nf4, nf5) = (lloyd(&gauss, 16), lloyd(&gauss, 32));
    let (s1, s2) = (gen_fwht_signs(42, G), gen_fwht_signs(1042, G));

    let mut entries: Vec<_> = std::fs::read_dir(indir)
        .expect("in dir")
        .map(|e| e.unwrap().path())
        .collect();
    entries.sort();
    let (mut n_q, mut n_w) = (0usize, 0usize);
    for src in entries {
        let dst = outdir.join(src.file_name().unwrap());
        std::fs::copy(&src, &dst).expect("copy");
        if src.extension().and_then(|e| e.to_str()) != Some("safetensors") {
            continue;
        }
        let f = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&dst)
            .expect("open rw");
        let mut mm = unsafe { memmap2::MmapMut::map_mut(&f).expect("mmap") };
        let hlen = u64::from_le_bytes(mm[0..8].try_into().unwrap()) as usize;
        let header: serde_json::Value = serde_json::from_slice(&mm[8..8 + hlen]).expect("hdr");
        let base = 8 + hlen;
        for (name, info) in header.as_object().unwrap() {
            if name == "__metadata__" || !quantized(name) {
                continue;
            }
            let shape: Vec<usize> = info["shape"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect();
            if shape.len() != 2 || shape[1] % G != 0 || info["dtype"].as_str() != Some("BF16") {
                eprintln!("  skip {name} {shape:?} {}", info["dtype"]);
                continue;
            }
            let (s, e) = (
                base + info["data_offsets"][0].as_u64().unwrap() as usize,
                base + info["data_offsets"][1].as_u64().unwrap() as usize,
            );
            let k = shape[1];
            mm[s..e].par_chunks_mut(k * 2).for_each(|bytes| {
                let mut row: Vec<f32> =
                    bytes.chunks(2).map(|b| bf16_to_f32([b[0], b[1]])).collect();
                for g in row.chunks_mut(G) {
                    cpu_fwht_256(g, &s1, &s2);
                }
                fake_row(&mut row, arm, &nf4, &nf5);
                for g in row.chunks_mut(G) {
                    cpu_fwht_256(g, &s2, &s1); // inverse: signs swapped
                }
                for (b, v) in bytes.chunks_mut(2).zip(&row) {
                    b.copy_from_slice(&f32_to_bf16(*v));
                }
            });
            n_q += 1;
            n_w += shape[0] * k;
        }
        mm.flush().expect("flush");
        eprintln!("  {} done", dst.display());
    }
    println!(
        "{arm}: fake-quantized {n_q} tensors, {:.2}B weights",
        n_w as f64 / 1e9
    );
}
