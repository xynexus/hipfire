// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! Can the sparse overlay be replaced by promoting WHOLE GROUPS, at the same bits?
//!
//! The proposal: per row, append R extra 256-wide groups holding nothing but the
//! high digits of R promoted groups, plus a map of which groups they are (w1, w2,
//! w3a, w3b, w4, ...). A promoted group is then int8 (hi*16 + lo) at a scale
//! re-chosen for 8 bits, and the extra group is a plain dense WMMA pass against
//! the promoted group's OWN activation slice -- no gather, no index per weight.
//!
//! The catch is the same one tile promotion hit: a WMMA A-tile is 16 rows x 16 K
//! against ONE activation layout, so the 16 rows of a tile must promote the SAME
//! groups (or the extra pass runs for every group any of them promoted). So this
//! scores both: per-row choice (an upper bound, not implementable) and choice
//! shared by each 16-row band (implementable).
//!
//! Bits are matched to the shipped overlay's 0.1875 b/w: R = round(0.1875 * K /
//! 1024) promoted groups per row (an extra group is 256 x 4 bits = 1024 bits).
//! Weight SSE relative to row energy, after the same FWHT and clip-search scale
//! the packer uses. SSE is not KLD; the gaps this is meant to resolve are large.
//!
//!   cargo run --release -p hipfire-quantize --example group_promote_study \
//!     -- <model.safetensors> [rows]

use hipfire_quantize::{cpu_fwht_256, gen_fwht_signs};

const G: usize = 256;
const N_OUT: usize = 3;

fn sse_int4(g: &[f32], scale: f32) -> f64 {
    let inv = 1.0 / scale;
    g.iter()
        .map(|&v| {
            let d = (v - (v * inv).round().clamp(-7.0, 7.0) * scale) as f64;
            d * d
        })
        .sum()
}

/// The shipped overlay: the N_OUT largest positions as int8 at the group's own
/// (clip-searched) scale, the rest int4.
fn sse_overlay(g: &[f32], scale: f32) -> f64 {
    let mut order: Vec<usize> = (0..G).collect();
    order.sort_by(|&i, &j| g[j].abs().partial_cmp(&g[i].abs()).unwrap());
    let inv = 1.0 / scale;
    let mut on = [false; G];
    for &p in &order[..N_OUT] {
        on[p] = true;
    }
    g.iter()
        .enumerate()
        .map(|(i, &v)| {
            let lim = if on[i] { 127.0 } else { 7.0 };
            let d = (v - (v * inv).round().clamp(-lim, lim) * scale) as f64;
            d * d
        })
        .sum()
}

/// A promoted group at `bits` (4 + the extra plane's width), its scale
/// re-chosen by the same clip search for the wider grid -- the format may store
/// a promoted group's own scale.
fn sse_promoted(g: &[f32], bits: u32) -> f64 {
    let lim = ((1u32 << (bits - 1)) - 1) as f32;
    let scale = hipfire_quantize::codecs::symmetric_clipsearch(g, lim);
    let inv = 1.0 / scale;
    g.iter()
        .map(|&v| {
            let d = (v - (v * inv).round().clamp(-lim, lim) * scale) as f64;
            d * d
        })
        .sum()
}

/// The overlay with only its single largest position.
fn sse_overlay1(g: &[f32], scale: f32) -> f64 {
    let top = (0..G)
        .max_by(|&i, &j| g[i].abs().partial_cmp(&g[j].abs()).unwrap())
        .unwrap();
    let inv = 1.0 / scale;
    g.iter()
        .enumerate()
        .map(|(i, &v)| {
            let lim = if i == top { 127.0 } else { 7.0 };
            let d = (v - (v * inv).round().clamp(-lim, lim) * scale) as f64;
            d * d
        })
        .sum()
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let Some(path) = a.get(1).cloned() else {
        eprintln!("usage: group_promote_study <model.safetensors> [rows]");
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

    println!("group-promotion study: G={G}, <= {rows_cap} rows, promotions shared per 16-row band");
    println!(
        "capture = share of the shipped overlay's SSE reduction from int4, at ~matched bits\n"
    );
    for sfx in [
        "mlp.gate_proj.weight",
        "mlp.down_proj.weight",
        "self_attn.q_proj.weight",
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
        let (rows, k) = (sh[0], sh[1]);
        let ng = k / G;
        let start = base + info["data_offsets"][0].as_u64().unwrap() as usize;
        // Per row, per group: [int4, overlay-1, promoted W8, W6, W5].
        let mut per_row: Vec<Vec<[f64; 5]>> = Vec::new();
        let (mut e, mut s4, mut sov) = (0f64, 0f64, 0f64);
        for r in 0..rows.min(rows_cap) {
            let mut row: Vec<f32> = (0..k)
                .map(|c| {
                    let i = start + (r * k + c) * 2;
                    f32::from_bits((u16::from_le_bytes([mmap[i], mmap[i + 1]]) as u32) << 16)
                })
                .collect();
            for ch in row.chunks_mut(G) {
                cpu_fwht_256(ch, &s1, &s2);
            }
            e += row.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>();
            let mut gs = Vec::with_capacity(ng);
            for grp in row.chunks(G) {
                if grp.iter().all(|v| *v == 0.0) {
                    gs.push([0.0; 5]);
                    continue;
                }
                let sc = hipfire_quantize::codecs::symmetric_clipsearch(grp, 7.0);
                let q4 = sse_int4(grp, sc);
                s4 += q4;
                sov += sse_overlay(grp, sc);
                gs.push([
                    q4,
                    sse_overlay1(grp, sc),
                    sse_promoted(grp, 8),
                    sse_promoted(grp, 6),
                    sse_promoted(grp, 5),
                ]);
            }
            per_row.push(gs);
        }
        // An arm: base column, promoted column, R promoted groups per row, shared
        // by each 16-row band (the band promotes the R groups with the largest
        // summed gain).
        let arm = |bcol: usize, pcol: usize, r_n: usize, band_rows: usize| -> f64 {
            let mut tot = 0f64;
            for band in per_row.chunks(band_rows) {
                let mut gain = vec![0f64; ng];
                for row in band {
                    for (gi, v) in row.iter().enumerate() {
                        gain[gi] += v[bcol] - v[pcol];
                    }
                }
                let mut order: Vec<usize> = (0..ng).collect();
                order.sort_by(|&i, &j| gain[j].partial_cmp(&gain[i]).unwrap());
                let chosen = &order[..r_n.min(ng)];
                for row in band {
                    for (gi, v) in row.iter().enumerate() {
                        tot += if chosen.contains(&gi) {
                            v[pcol]
                        } else {
                            v[bcol]
                        };
                    }
                }
            }
            tot
        };
        let cap = |x: f64| 100.0 * (s4 - x) / (s4 - sov);
        let kf = k as f64;
        println!("  {} (K={k}, ng={ng})", sfx.trim_end_matches(".weight"));
        println!(
            "    {:<34} {:>7} {:>10} {:>8}",
            "arm", "b/w", "rel SSE", "capture"
        );
        println!(
            "    {:<34} {:>7.3} {:>10.3e} {:>7.1}%",
            "int4",
            4.0625,
            s4 / e,
            0.0
        );
        println!(
            "    {:<34} {:>7.3} {:>10.3e} {:>7.1}%",
            "overlay-3 (shipped)",
            4.25,
            sov / e,
            100.0
        );
        for (label, bcol, pcol, extra_bits, budget, ov_bits, band_rows) in [
            (
                "promote W8, shared/16 rows",
                0usize,
                2usize,
                4.0,
                0.1875,
                0.0,
                16usize,
            ),
            ("promote W6, shared/16 rows", 0, 3, 2.0, 0.1875, 0.0, 16),
            ("promote W5, shared/16 rows", 0, 4, 1.0, 0.1875, 0.0, 16),
            (
                "overlay-1 + promote W6, shared/16",
                1,
                3,
                2.0,
                0.125,
                0.0625,
                16,
            ),
            ("promote W8, per row", 0, 2, 4.0, 0.1875, 0.0, 1),
            ("promote W6, per row", 0, 3, 2.0, 0.1875, 0.0, 1),
            ("promote W5, per row", 0, 4, 1.0, 0.1875, 0.0, 1),
            (
                "overlay-1 + promote W6, per row",
                1,
                3,
                2.0,
                0.125,
                0.0625,
                1,
            ),
            (
                "overlay-1 + promote W5, per row",
                1,
                4,
                1.0,
                0.125,
                0.0625,
                1,
            ),
        ] {
            let r_n = ((budget * kf / (256.0 * extra_bits)).round() as usize).max(1);
            let bw = 4.0625 + ov_bits + r_n as f64 * 256.0 * extra_bits / kf;
            let x = arm(bcol, pcol, r_n, band_rows);
            println!(
                "    {:<34} {:>7.3} {:>10.3e} {:>7.1}%   R={r_n}",
                label,
                bw,
                x / e,
                cap(x)
            );
        }
    }
}
