//! What does Kairic's "one long K segment" contract cost in weight error?
//!
//! `ROCmFPX/ggml/src/ggml-cuda/promptforge_iu4.cuh` (MIT) ships
//! `constexpr int kSegments = 1` with the rationale, verbatim:
//!
//!   "One long K segment is the performance-first contract. It keeps scale and
//!    zero-point correction out of the WMMA inner loop and leaves quality
//!    recovery to offline weight optimization and bounded keeper corrections."
//!
//! That is ONE f32 scale per output row across the whole K, against our G=256.
//! It is the reason their IU4 inner loop has no fold-and-rescale, which is in
//! turn why our wave64 port had to spend registers on a second accumulator set
//! (WNt=4, 1.26x) instead of the 1-pass twin's shape (WNt=8, 1.56x) -- and the
//! per-group contract is also what makes the sparse overlay necessary at 4.25
//! bits, costing a further 23.9% of prefill.
//!
//! So before any kernel work: how much weight error does coarsening the scale
//! actually cost? Everything here is offline weight math; no GPU, no kernel.
//!
//!     cargo run --release -p hipfire-quantize --example coarse_scale_study \
//!         -- <model.safetensors> [max_rows_per_tensor]
//!
//! Reports relative reconstruction error (SSE / sum w^2) per tensor for a
//! sweep of scale granularities, all int4 symmetric, all on the SAME
//! FWHT-rotated weights so only granularity varies.

use hipfire_quantize::{cpu_fwht_256, gen_fwht_signs};

const SUFFIXES: [&str; 4] = [
    "down_proj.weight",
    "o_proj.weight",
    "gate_proj.weight",
    "q_proj.weight",
];

/// Signed FWHT of arbitrary power-of-two length, orthonormal.
/// Kairic rotates at `kHadamardBlock = 1024`; our shipped pipeline uses 256. A
/// longer segment wants a longer rotation, because what makes one scale viable
/// is variance being equalized across the whole span it covers.
fn signed_fwht_n(x: &mut [f32], signs: &[f32]) {
    let n = x.len();
    for (v, s) in x.iter_mut().zip(signs) {
        *v *= *s;
    }
    let mut len = 1;
    while len < n {
        let mut i = 0;
        while i < n {
            for j in i..i + len {
                let (a, b) = (x[j], x[j + len]);
                x[j] = a + b;
                x[j + len] = a - b;
            }
            i += len << 1;
        }
        len <<= 1;
    }
    let inv = 1.0 / (n as f32).sqrt();
    for v in x.iter_mut() {
        *v *= inv;
    }
}

/// Deterministic +-1 signs, same generator shape as `gen_fwht_signs`.
fn signs_n(seed: u64, n: usize) -> Vec<f32> {
    let mut st = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..n)
        .map(|_| {
            st = st
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            if (st >> 33) & 1 == 0 {
                -1.0
            } else {
                1.0
            }
        })
        .collect()
}

/// int4 symmetric quantize/dequantize one span with a single scale; returns SSE.
fn sse_span(w: &[f32]) -> f64 {
    let amax = w.iter().fold(0f32, |m, v| m.max(v.abs()));
    if amax <= 0.0 {
        return 0.0;
    }
    // 4-bit signed: codes -8..7, so the positive edge is 7.
    let scale = amax / 7.0;
    let inv = 1.0 / scale;
    w.iter()
        .map(|&v| {
            let q = (v * inv).round().clamp(-8.0, 7.0);
            let d = v - q * scale;
            (d as f64) * (d as f64)
        })
        .sum()
}

/// SSE for a whole row split into spans of `g` (g == 0 means one span over K).
fn sse_rowwise(row: &[f32], g: usize) -> f64 {
    let g = if g == 0 { row.len() } else { g };
    row.chunks(g).map(sse_span).sum()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let Some(path) = args.get(1).cloned() else {
        eprintln!("usage: coarse_scale_study <model.safetensors> [max_rows]");
        std::process::exit(2);
    };
    let max_rows: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(256);
    const G: usize = 256;

    let file = std::fs::File::open(&path).expect("open safetensors");
    let mmap = unsafe { memmap2::Mmap::map(&file).expect("mmap") };
    let hlen = u64::from_le_bytes(mmap[0..8].try_into().unwrap()) as usize;
    let header: serde_json::Value =
        serde_json::from_slice(&mmap[8..8 + hlen]).expect("header json");
    let base = 8 + hlen;
    let signs1 = gen_fwht_signs(42, G);
    let signs2 = gen_fwht_signs(1042, G);

    println!("coarse-scale study: int4 symmetric, FWHT-rotated, <= {max_rows} rows/tensor");
    println!("relative reconstruction error (SSE / sum w^2), lower is better\n");
    println!(
        "  {:<20} {:>7} {:>10} {:>10} {:>10} {:>11} {:>11} {:>8}",
        "tensor",
        "K",
        "r256/G256",
        "r256/G1k",
        "r256/1seg",
        "r1024/1seg",
        "r4096/1seg",
        "best/base"
    );
    let s1k = signs_n(42, 1024);
    let s4k = signs_n(42, 4096);

    let obj = header.as_object().expect("header object");
    let mut names: Vec<&String> = obj
        .keys()
        .filter(|k| SUFFIXES.iter().any(|s| k.ends_with(s)))
        .collect();
    names.sort();
    // One representative tensor per suffix keeps the run short.
    let mut seen: Vec<&str> = Vec::new();
    let mut tot = [0.0f64; 5];
    let mut totw = 0.0f64;

    for name in names {
        let Some(sfx) = SUFFIXES.iter().find(|s| name.ends_with(*s)) else {
            continue;
        };
        if seen.contains(sfx) {
            continue;
        }
        let info = &obj[name];
        let shape: Vec<usize> = info["shape"]
            .as_array()
            .expect("shape")
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        if shape.len() != 2 || shape[1] % G != 0 {
            continue;
        }
        let dtype = info["dtype"].as_str().unwrap_or("");
        let off = info["data_offsets"].as_array().expect("offsets");
        let start = base + off[0].as_u64().unwrap() as usize;
        let (rows, k) = (shape[0], shape[1]);
        let take = rows.min(max_rows);
        let mut acc = [0.0f64; 5];
        let mut energy = 0.0f64;

        for r in 0..take {
            let raw: Vec<f32> = (0..k)
                .map(|c| {
                    let i = start + (r * k + c) * 2;
                    let raw = u16::from_le_bytes([mmap[i], mmap[i + 1]]);
                    match dtype {
                        "BF16" => f32::from_bits((raw as u32) << 16),
                        _ => half_to_f32(raw),
                    }
                })
                .collect();
            let mut row: Vec<f32> = raw.clone();
            // Same rotation the shipped pipeline applies before quantizing.
            for chunk in row.chunks_mut(G) {
                cpu_fwht_256(chunk, &signs1, &signs2);
            }
            energy += row.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>();
            acc[0] += sse_rowwise(&row, 256);
            acc[1] += sse_rowwise(&row, 1024);
            acc[2] += sse_rowwise(&row, 0); // one segment, OUR 256 rotation
                                            // Re-rotate the ORIGINAL row at 1024 and at 4096, then one segment.
            let mut r1k: Vec<f32> = raw.clone();
            for c in r1k.chunks_mut(1024) {
                if c.len() == 1024 {
                    signed_fwht_n(c, &s1k);
                }
            }
            acc[3] += sse_rowwise(&r1k, 0);
            let mut r4k: Vec<f32> = raw.clone();
            for c in r4k.chunks_mut(4096) {
                if c.len() == 4096 {
                    signed_fwht_n(c, &s4k);
                }
            }
            acc[4] += sse_rowwise(&r4k, 0);
        }
        if energy <= 0.0 {
            continue;
        }
        let rel: Vec<f64> = acc.iter().map(|a| a / energy).collect();
        let best = rel[2].min(rel[3]).min(rel[4]);
        println!(
            "  {:<20} {:>7} {:>10.3e} {:>10.3e} {:>10.3e} {:>11.3e} {:>11.3e} {:>7.2}x",
            sfx,
            k,
            rel[0],
            rel[1],
            rel[2],
            rel[3],
            rel[4],
            best / rel[0].max(1e-30)
        );
        for i in 0..5 {
            tot[i] += acc[i];
        }
        totw += energy;
        seen.push(sfx);
    }
    if totw > 0.0 {
        let rel: Vec<f64> = tot.iter().map(|a| a / totw).collect();
        let best = rel[2].min(rel[3]).min(rel[4]);
        println!(
            "\n  {:<20} {:>7} {:>10.3e} {:>10.3e} {:>10.3e} {:>11.3e} {:>11.3e} {:>7.2}x",
            "ALL",
            "",
            rel[0],
            rel[1],
            rel[2],
            rel[3],
            rel[4],
            best / rel[0].max(1e-30)
        );
    }
}

fn half_to_f32(h: u16) -> f32 {
    let s = ((h >> 15) & 1) as u32;
    let e = ((h >> 10) & 0x1f) as u32;
    let m = (h & 0x3ff) as u32;
    let bits = match e {
        0 if m == 0 => s << 31,
        0 => {
            let mut e2 = 127 - 15 + 1;
            let mut m2 = m;
            while m2 & 0x400 == 0 {
                m2 <<= 1;
                e2 -= 1;
            }
            (s << 31) | (e2 << 23) | ((m2 & 0x3ff) << 13)
        }
        31 => (s << 31) | (0xff << 23) | (m << 13),
        _ => (s << 31) | ((e + 127 - 15) << 23) | (m << 13),
    };
    f32::from_bits(bits)
}
