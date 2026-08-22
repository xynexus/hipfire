//! Does forcing the compact overlay to SHARE outlier positions across rows cost
//! quality — and how much?
//!
//! Why ask: the sparse overlay correction is only ~1.2% of the GEMM's MACs but
//! costs 30-100% of its time, because `idx` varies per (row, group) so every
//! lookup is a gather. If all rows of a group used the SAME n_out positions, the
//! correction collapses to a dense rank-n_out update — no gather — and the index
//! bytes disappear from the format (4.25 -> 4.156 bits/weight as a bonus).
//!
//! The catch, and the reason this is a measurement and not a patch: positions are
//! chosen AFTER the FWHT rotation (`signed_fwht` then `mixed_clipsearch`). The
//! AWQ intuition that outliers are shared activation channels lives in the
//! ORIGINAL basis; the rotation exists precisely to destroy that structure. So
//! the shared positions may be close to arbitrary.
//!
//!     cargo run --release -p hipfire-quantize --example opus_shared_overlay_study \
//!         -- <model.safetensors> [max_rows_per_tensor]
//!
//! Reports per tensor: weight SSE of per-row selection (shipped) vs shared
//! selection, at identical n_out and identical scales.

use hipfire_quantize::codecs::mixed_clipsearch;
use hipfire_quantize::{cpu_fwht_256, gen_fwht_signs};

const SUFFIXES: [&str; 4] = [
    "down_proj.weight",
    "o_proj.weight",
    "gate_proj.weight",
    "q_proj.weight",
];

/// Sum of squared reconstruction error for one group under a given overlay set.
fn sse_for(group: &[f32], scale: f32, idx: &[usize]) -> f64 {
    let inv = 1.0 / scale.max(1e-12);
    let mut on = vec![false; group.len()];
    for &p in idx {
        on[p] = true;
    }
    let mut sse = 0.0f64;
    for (i, &v) in group.iter().enumerate() {
        let q = if on[i] {
            (v * inv).round().clamp(-127.0, 127.0)
        } else {
            (v * inv).round().clamp(-7.0, 7.0)
        };
        let e = (v - q * scale) as f64;
        sse += e * e;
    }
    sse
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let Some(path) = args.get(1).cloned() else {
        eprintln!("usage: opus_shared_overlay_study <model.safetensors> [max_rows]");
        std::process::exit(2);
    };
    let max_rows: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(512);
    const G: usize = 256;
    const N_OUT: usize = 3;

    let file = std::fs::File::open(&path).expect("open safetensors");
    let mmap = unsafe { memmap2::Mmap::map(&file).expect("mmap") };
    let hlen = u64::from_le_bytes(mmap[0..8].try_into().unwrap()) as usize;
    let header: serde_json::Value =
        serde_json::from_slice(&mmap[8..8 + hlen]).expect("header json");
    let base = 8 + hlen;
    let signs1 = gen_fwht_signs(42, G);
    let signs2 = gen_fwht_signs(1042, G);

    println!("shared-position overlay study: n_out={N_OUT}, G={G}, <= {max_rows} rows/tensor");
    println!(
        "  {:<24} {:>11} {:>12} {:>11} {:>11} {:>8} {:>8}",
        "tensor", "int4 G256", "overlay 4.25", "shared", "G64 no-ov", "G64vsOv", "capture"
    );

    let mut tot_row = 0.0f64;
    let mut tot_shared = 0.0f64;
    for suffix in SUFFIXES {
        let mut names: Vec<&String> = header
            .as_object()
            .unwrap()
            .keys()
            .filter(|k| k.ends_with(suffix))
            .collect();
        if names.is_empty() {
            continue;
        }
        names.sort();
        let name = names[names.len() / 2];
        let shape: Vec<usize> = header[name]["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        if shape.len() != 2 || shape[1] % G != 0 {
            continue;
        }
        let (rows, k) = (shape[0].min(max_rows), shape[1]);
        let ngroups = k / G;
        let off: Vec<usize> = header[name]["data_offsets"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        let bytes = &mmap[base + off[0]..base + off[1]];

        // Rotate every (row, group) once; keep scales from the shipped per-row search.
        let mut rot: Vec<Vec<f32>> = Vec::with_capacity(rows * ngroups);
        let mut scales: Vec<f32> = Vec::with_capacity(rows * ngroups);
        let mut per_row_idx: Vec<Vec<usize>> = Vec::with_capacity(rows * ngroups);
        for r in 0..rows {
            for g in 0..ngroups {
                let start = (r * k + g * G) * 2;
                let mut grp = vec![0.0f32; G];
                for i in 0..G {
                    let b = u16::from_le_bytes([bytes[start + 2 * i], bytes[start + 2 * i + 1]]);
                    grp[i] = f32::from_bits((b as u32) << 16); // bf16 -> f32
                }
                cpu_fwht_256(&mut grp, &signs1, &signs2);
                let (s, idx) = mixed_clipsearch(&grp, N_OUT);
                scales.push(s);
                per_row_idx.push(idx[..N_OUT].to_vec());
                rot.push(grp);
            }
        }

        // (A one-off residual dump lived here while ranking the low-rank
        // alternative — see docs/experiments/2026-08-22-overlay-alternatives-and-qat-scope.md
        // for the result. Removed: hipfire-quantize forbids raw std::env::var,
        // and registering an env var for a throwaway probe is not worth it.)

        // Shared selection: for each group column g, rank positions by the TOTAL
        // int4->int8 error reduction summed over every row, then take the top n_out.
        // Same scales, so the two arms differ only in WHICH positions get 8 bits.
        let mut shared_idx: Vec<Vec<usize>> = Vec::with_capacity(ngroups);
        for g in 0..ngroups {
            let mut gain = vec![0.0f64; G];
            for r in 0..rows {
                let gi = r * ngroups + g;
                let s = scales[gi];
                let inv = 1.0 / s.max(1e-12);
                for p in 0..G {
                    let v = rot[gi][p];
                    let e4 = v - (v * inv).round().clamp(-7.0, 7.0) * s;
                    let e8 = v - (v * inv).round().clamp(-127.0, 127.0) * s;
                    gain[p] += (e4 * e4 - e8 * e8) as f64;
                }
            }
            let mut order: Vec<usize> = (0..G).collect();
            order.sort_by(|&a, &b| gain[b].partial_cmp(&gain[a]).unwrap());
            shared_idx.push(order[..N_OUT].to_vec());
        }

        // ALTERNATIVE AT MATCHED BITS: G=64, no overlay. Scale cost 16 b / 64 w
        // = 0.25 b/w against G=256's 0.0625 + 3 overlays' 0.1875 = 0.25. Same
        // 4.25 bits/weight, but no side table -> no gather at all.
        let s1_64 = gen_fwht_signs(44, 64);
        let s2_64 = gen_fwht_signs(1044, 64);
        let mut sse_g64 = 0.0f64;
        for r in 0..rows {
            for g in 0..ngroups {
                for sub in 0..4 {
                    let start = (r * k + g * G + sub * 64) * 2;
                    let mut grp = vec![0.0f32; 64];
                    for i in 0..64 {
                        let b =
                            u16::from_le_bytes([bytes[start + 2 * i], bytes[start + 2 * i + 1]]);
                        grp[i] = f32::from_bits((b as u32) << 16);
                    }
                    hipfire_primitives::fwht::signed_fwht(&mut grp, &s1_64, &s2_64);
                    let (sc, _) = mixed_clipsearch(&grp, 1);
                    sse_g64 += sse_for(&grp, sc, &[]); // no overlay
                }
            }
        }

        let mut sse_row = 0.0f64;
        let mut sse_shared = 0.0f64;
        let mut sse_none = 0.0f64; // pure int4, same scales: what the overlay buys
        for r in 0..rows {
            for g in 0..ngroups {
                let gi = r * ngroups + g;
                sse_row += sse_for(&rot[gi], scales[gi], &per_row_idx[gi]);
                sse_shared += sse_for(&rot[gi], scales[gi], &shared_idx[g]);
                sse_none += sse_for(&rot[gi], scales[gi], &[]);
            }
        }
        tot_row += sse_row;
        tot_shared += sse_shared;
        // capture = fraction of the per-row overlay's benefit that shared keeps
        let capture = (sse_none - sse_shared) / (sse_none - sse_row) * 100.0;
        println!(
            "  {:<24} {:>11.4e} {:>11.4e} {:>11.4e} {:>11.4e} {:>7.1}% {:>8.1}%",
            name.rsplit('.').take(3).collect::<Vec<_>>().join("."),
            sse_none,
            sse_row,
            sse_shared,
            sse_g64,
            (sse_g64 / sse_row - 1.0) * 100.0,
            capture
        );
    }
    println!(
        "\n  TOTAL  per-row {tot_row:.4e}   shared {tot_shared:.4e}   ratio {:.2}x",
        tot_shared / tot_row
    );
    println!("  (ratio 1.00 = free; higher = shared positions cost quality)");
}
