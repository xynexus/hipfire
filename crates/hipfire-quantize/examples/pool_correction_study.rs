//! Correction as a POST-WMMA dense add over a per-group position pool.
//!
//! The shipped overlay is `Y += S*X_g` with `S` sparse and per-row indexed, so
//! applying it is a gather -- 23.9% of prefill for 1.2% of the arithmetic.
//! Making the widening row-uniform kills the gather but recovers only 22% of the
//! overlay's benefit (tile_promote_study).
//!
//! Middle ground: let the OFFSETS define a per-group POOL of P candidate
//! positions, shared by every row of the group. Then `S` is a dense `M x P`
//! matrix, `X_g` compacts to those P rows ONCE per group, and the correction is
//! a dense GEMM of inner dimension P -- i.e. P/256 extra WMMA cycles, no gather,
//! no per-entry index. At P=16 that is exactly one extra K-tile: +6.25%.
//!
//! Unlike exact position sharing, every row still gets corrected at all P pooled
//! positions; only WHICH positions are candidates is shared. This measures what
//! that costs.
//!
//!     cargo run --release -p hipfire-quantize --example pool_correction_study \
//!         -- <model.safetensors> [rows]

use hipfire_quantize::codecs::symmetric_clipsearch;
use hipfire_quantize::{cpu_fwht_256, gen_fwht_signs};

const G: usize = 256;
const N_OUT: usize = 3;

fn sse_at(g: &[f32], scale: f32, wide: &[bool]) -> f64 {
    let inv = 1.0 / scale;
    g.iter()
        .enumerate()
        .map(|(i, &v)| {
            let q = if wide[i] {
                (v * inv).round().clamp(-127.0, 127.0)
            } else {
                (v * inv).round().clamp(-7.0, 7.0)
            };
            let d = (v - q * scale) as f64;
            d * d
        })
        .sum()
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let Some(path) = a.get(1).cloned() else {
        eprintln!("usage: pool_correction_study <model.safetensors> [rows]");
        std::process::exit(2);
    };
    let cap: usize = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(128);

    let file = std::fs::File::open(&path).expect("open");
    let mmap = unsafe { memmap2::Mmap::map(&file).expect("mmap") };
    let hlen = u64::from_le_bytes(mmap[0..8].try_into().unwrap()) as usize;
    let hdr: serde_json::Value = serde_json::from_slice(&mmap[8..8 + hlen]).expect("hdr");
    let base = 8 + hlen;
    let obj = hdr.as_object().unwrap();
    let s1 = gen_fwht_signs(42, G);
    let s2 = gen_fwht_signs(1042, G);

    println!("pool-correction study: G={G}, <= {cap} rows, pool shared per 256-group\n");
    println!(
        "  {:<12} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "tensor", "int4", "overlay-3", "pool P=16", "P=32", "P=64", "P=16/ov"
    );

    for sfx in ["gate_proj.weight", "down_proj.weight", "q_proj.weight"] {
        let Some(name) = obj.keys().find(|n| n.ends_with(sfx)) else {
            continue;
        };
        let info = &obj[name];
        let sh: Vec<usize> = info["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        if sh.len() != 2 || sh[1] % G != 0 {
            continue;
        }
        let (rows, k) = (sh[0], sh[1]);
        let start = base
            + info["data_offsets"].as_array().unwrap()[0]
                .as_u64()
                .unwrap() as usize;
        let take = rows.min(cap);

        let mut rowsv: Vec<Vec<f32>> = Vec::with_capacity(take);
        for r in 0..take {
            let mut row: Vec<f32> = (0..k)
                .map(|c| {
                    let i = start + (r * k + c) * 2;
                    f32::from_bits((u16::from_le_bytes([mmap[i], mmap[i + 1]]) as u32) << 16)
                })
                .collect();
            for ch in row.chunks_mut(G) {
                cpu_fwht_256(ch, &s1, &s2);
            }
            rowsv.push(row);
        }
        let energy: f64 = rowsv
            .iter()
            .flat_map(|r| r.iter())
            .map(|&v| (v as f64) * (v as f64))
            .sum();
        let ng = k / G;
        let mut acc = [0.0f64; 5];

        for gi in 0..ng {
            let scales: Vec<f32> = rowsv
                .iter()
                .map(|r| symmetric_clipsearch(&r[gi * G..(gi + 1) * G], 7.0))
                .collect();
            // Per-position benefit, summed over rows: how much error int4 leaves
            // at this position that int8 would remove. This ranks the pool.
            let mut benefit = vec![0.0f64; G];
            for (r, &sc) in rowsv.iter().zip(&scales) {
                let grp = &r[gi * G..(gi + 1) * G];
                let inv = 1.0 / sc;
                for (p, &v) in grp.iter().enumerate() {
                    let q4 = (v * inv).round().clamp(-7.0, 7.0);
                    let q8 = (v * inv).round().clamp(-127.0, 127.0);
                    let e4 = (v - q4 * sc) as f64;
                    let e8 = (v - q8 * sc) as f64;
                    benefit[p] += e4 * e4 - e8 * e8;
                }
            }
            let mut order: Vec<usize> = (0..G).collect();
            order.sort_by(|&i, &j| benefit[j].partial_cmp(&benefit[i]).unwrap());

            for (r, &sc) in rowsv.iter().zip(&scales) {
                let grp = &r[gi * G..(gi + 1) * G];
                let none = vec![false; G];
                acc[0] += sse_at(grp, sc, &none);
                // shipped: this row's own top-N_OUT positions (gather)
                let mut own: Vec<usize> = (0..G).collect();
                own.sort_by(|&i, &j| grp[j].abs().partial_cmp(&grp[i].abs()).unwrap());
                let mut w = vec![false; G];
                for &p in &own[..N_OUT] {
                    w[p] = true;
                }
                acc[1] += sse_at(grp, sc, &w);
                // pooled: EVERY row corrected at the group's P pooled positions
                for (slot, p) in [(2usize, 16usize), (3, 32), (4, 64)] {
                    let mut w = vec![false; G];
                    for &q in &order[..p] {
                        w[q] = true;
                    }
                    acc[slot] += sse_at(grp, sc, &w);
                }
            }
        }
        let rel: Vec<f64> = acc.iter().map(|v| v / energy).collect();
        println!(
            "  {:<12} {:>10.3e} {:>10.3e} {:>10.3e} {:>10.3e} {:>10.3e} {:>9.2}x",
            sfx.trim_end_matches(".weight"),
            rel[0],
            rel[1],
            rel[2],
            rel[3],
            rel[4],
            rel[2] / rel[1]
        );
    }
    println!("\ncost of the post-WMMA dense add (inner dim P, no gather, no index):");
    for p in [16usize, 32, 64] {
        println!(
            "  P={p:<3} -> +{:.2}% WMMA cycles, +{} B/group at int4 corrections",
            p as f64 / 256.0 * 100.0,
            p / 2
        );
    }
    println!("  shipped overlay -> +23.9% of PREFILL as a separate gather pass");
}
