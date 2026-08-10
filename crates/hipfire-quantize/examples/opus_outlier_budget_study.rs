// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! Per-layer outlier budget study for the compact mixed Opus format — the CPU
//! half of the `HIPFIRE_OUTLIERS_BY_LAYER` sweep.
//!
//! The allocation question is: at a FIXED byte budget, should every tensor get
//! the same `N_out` overlay slots per 256-group, or should some layer types get
//! more? Commit d77fa637a answered "uniform wins" against the OLD selector,
//! which seeded the group scale from an int4-only clip-search and then
//! alternated. That selector systematically under-rewarded extra outliers, so
//! this re-scores the same question against the joint `mixed_clipsearch`.
//!
//! Why per-group SSE is the right currency here: both the gain and the cost of
//! one more overlay slot scale with a tensor's group count, so the group count
//! cancels. The optimal bit-matched allocation is the one that EQUALISES the
//! per-group marginal SSE reduction across layer types — no parameter-share
//! weighting needed to rank them.
//!
//! CAVEAT, and it is not a small one: weight SSE is not KLD, and these two
//! demonstrably disagreed in this exact experiment before — d77fa637a found
//! oq4.5++ (N=7) scoring WORSE KLD than N=3 while spending more bits, which no
//! weight-SSE curve would predict. Treat this as a config narrower for the GPU
//! run, never as its replacement.
//!
//!   cargo run --release -p hipfire-quantize --example opus_outlier_budget_study \
//!     -- <model.safetensors> [max_groups_per_type]

use hipfire_quantize::codecs::{mixed_clipsearch, mixed_overlay_error, mixed_overlay_indices};
use hipfire_quantize::{cpu_fwht_256, gen_fwht_signs};

/// Scale grid shared by the historical and joint searches.
const CLIP_GRID: [f32; 14] = [
    1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35,
];

/// The pre-P2 selector, verbatim: int4-only seed, then two rounds of
/// set-at-fixed-scale / scale-at-fixed-set. Kept local — it exists only as the
/// baseline this study measures against, and must not drift back into the lib.
fn old_alternating(group: &[f32; 256], n_out: usize) -> (f32, Vec<usize>) {
    fn int4_seed(group: &[f32; 256]) -> f32 {
        const SEED_GRID: [f32; 9] = [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6];
        let amax = group.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let (mut best_scale, mut best_err) = ((amax / 7.0).max(1e-12), f32::INFINITY);
        for c in SEED_GRID {
            let scale = (c * amax / 7.0).max(1e-12);
            let inv = 1.0 / scale;
            let err: f32 = group
                .iter()
                .map(|&v| {
                    let d = v - (v * inv).round().clamp(-7.0, 7.0) * scale;
                    d * d
                })
                .sum();
            if err < best_err {
                best_err = err;
                best_scale = scale;
            }
        }
        best_scale
    }
    fn refit(group: &[f32; 256], idx: &[usize], n_out: usize, fallback: f32) -> f32 {
        let amax = group.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let mut best_scale = fallback.max(1e-12);
        let mut best_err = mixed_overlay_error(group, best_scale, idx, n_out);
        for clip in CLIP_GRID {
            let scale = (clip * amax / 7.0).max(1e-12);
            let err = mixed_overlay_error(group, scale, idx, n_out);
            if err < best_err {
                best_scale = scale;
                best_err = err;
            }
        }
        best_scale
    }
    let s0 = int4_seed(group);
    let i0 = mixed_overlay_indices(group, s0, n_out);
    let s1 = refit(group, &i0, n_out, s0);
    let i1 = mixed_overlay_indices(group, s1, n_out);
    let s2 = refit(group, &i1, n_out, s1);
    (s2, i1)
}

/// Layer types the budget is allocated across, matched by name suffix exactly
/// as `outliers_per_group_for` matches `HIPFIRE_OUTLIERS_BY_LAYER` keys.
const TYPES: [&str; 7] = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
];
const BUDGETS: [usize; 7] = [1, 3, 5, 7, 9, 15, 31];

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).cloned().unwrap_or_else(|| {
        "/srv/huggingface/models--Qwen--Qwen3.5-0.8B/snapshots/\
         2fc06364715b967f1860aea9cf38778875588b17/model.safetensors-00001-of-00001.safetensors"
            .to_string()
    });
    let cap: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(2048);

    let file = std::fs::File::open(&path).expect("open safetensors");
    let mmap = unsafe { memmap2::Mmap::map(&file).expect("mmap") };
    let hlen = u64::from_le_bytes(mmap[0..8].try_into().unwrap()) as usize;
    let header: serde_json::Value =
        serde_json::from_slice(&mmap[8..8 + hlen]).expect("parse header");
    let base = 8 + hlen;
    let signs1 = gen_fwht_signs(42, 256);
    let signs2 = gen_fwht_signs(1042, 256);

    println!("model: {path}");
    println!("cap:   {cap} groups per layer type\n");

    // Per type: total params (for the bit-rate arithmetic) and the sampled groups.
    let mut rows: Vec<(String, usize, Vec<[f32; 256]>)> = Vec::new();
    for ty in TYPES {
        let mut params = 0usize;
        let mut groups: Vec<[f32; 256]> = Vec::new();
        let mut names: Vec<&String> = header
            .as_object()
            .unwrap()
            .keys()
            .filter(|n| n.contains(ty) && n.ends_with(".weight"))
            .collect();
        names.sort();
        for name in &names {
            let meta = &header[*name];
            if meta["dtype"].as_str() != Some("BF16") {
                continue;
            }
            let off: Vec<usize> = meta["data_offsets"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect();
            params += (off[1] - off[0]) / 2;
            if groups.len() >= cap {
                continue;
            }
            let bytes = &mmap[base + off[0]..base + off[1]];
            for chunk in bytes.chunks_exact(2).collect::<Vec<_>>().chunks(256) {
                if chunk.len() < 256 || groups.len() >= cap {
                    break;
                }
                let mut g = [0.0f32; 256];
                for (i, c) in chunk.iter().enumerate() {
                    g[i] = f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16);
                }
                cpu_fwht_256(&mut g, &signs1, &signs2);
                groups.push(g);
            }
        }
        if !groups.is_empty() {
            rows.push((ty.to_string(), params, groups));
        }
    }

    let total_params: usize = rows.iter().map(|r| r.1).sum();

    // Mean per-group SSE at each budget, both selectors.
    let mut old_tab: Vec<Vec<f64>> = Vec::new();
    let mut new_tab: Vec<Vec<f64>> = Vec::new();
    for (_, _, groups) in &rows {
        let mut o = Vec::new();
        let mut n = Vec::new();
        for &b in &BUDGETS {
            let (mut oa, mut na) = (0.0f64, 0.0f64);
            for g in groups {
                let (os, oi) = old_alternating(g, b);
                let (ns, ni) = mixed_clipsearch(g, b);
                oa += mixed_overlay_error(g, os, &oi, b) as f64;
                na += mixed_overlay_error(g, ns, &ni, b) as f64;
            }
            o.push(oa / groups.len() as f64);
            n.push(na / groups.len() as f64);
        }
        old_tab.push(o);
        new_tab.push(n);
    }

    let head: String = BUDGETS.iter().map(|b| format!(" N={b:<9}")).collect();
    for (label, tab) in [("OLD (alternating)", &old_tab), ("NEW (joint)", &new_tab)] {
        println!("== mean per-group SSE — {label} ==");
        println!("{:<11}{head}", "layer");
        for (i, (ty, params, groups)) in rows.iter().enumerate() {
            let cells: String = tab[i].iter().map(|v| format!(" {v:<10.5}")).collect();
            println!(
                "{ty:<11}{cells}   ({:.1}% of params, {} groups)",
                100.0 * *params as f64 / total_params as f64,
                groups.len()
            );
        }
        println!();
    }

    // Marginal value of one more overlay slot, in SSE reduced per EXTRA BYTE per
    // group. A bit-matched allocation should equalise this across layer types;
    // where the columns are flat, uniform is already optimal.
    println!("== marginal SSE reduction per extra byte/group, NEW selector ==");
    println!("(budget moves toward whichever layer's number is largest)");
    let mhead: String = BUDGETS
        .windows(2)
        .map(|w| format!(" {}->{:<7}", w[0], w[1]))
        .collect();
    println!("{:<11}{mhead}", "layer");
    for (i, (ty, _, _)) in rows.iter().enumerate() {
        let cells: String = BUDGETS
            .windows(2)
            .enumerate()
            .map(|(j, w)| {
                let d = new_tab[i][j] - new_tab[i][j + 1];
                format!(" {:<10.6}", d / (2.0 * (w[1] - w[0]) as f64))
            })
            .collect();
        println!("{ty:<11}{cells}");
    }
    println!(
        "\nbit rates: N=1 4.125  N=3 4.250  N=5 4.375  N=7 4.500  \
         N=9 4.625  N=15 5.000  N=31 6.000 b/w"
    );

    // ---- Bit-matched optimum -------------------------------------------------
    //
    // minimise  Σ_t G_t·sse_t(N_t)   s.t.  Σ_t G_t·(130 + 2·N_t) fixed.
    // The Lagrange condition is sse_t'(N_t)/2 = λ, and G_t cancels out of it —
    // so RANKING is by per-group marginal alone, while the BUDGET is charged by
    // parameter share. Greedy in unit steps is exact here because each sse_t(N)
    // curve is convex in N.
    const NMAX: usize = 16;
    let dense: Vec<Vec<f64>> = rows
        .iter()
        .map(|(_, _, groups)| {
            (1..=NMAX)
                .map(|n| {
                    groups
                        .iter()
                        .map(|g| {
                            let (s, i) = mixed_clipsearch(g, n);
                            mixed_overlay_error(g, s, &i, n) as f64
                        })
                        .sum::<f64>()
                        / groups.len() as f64
                })
                .collect()
        })
        .collect();

    let share: Vec<f64> = rows
        .iter()
        .map(|(_, p, _)| *p as f64 / total_params as f64)
        .collect();
    let target: f64 = 3.0; // Σ share·N == 3  ⇒  4.25 b/w, matching uniform N=3
    let mut alloc = vec![1usize; rows.len()];
    loop {
        let spent: f64 = alloc.iter().zip(&share).map(|(n, w)| *n as f64 * w).sum();
        // Best next unit step by per-group marginal, that still fits the budget.
        let mut best: Option<(usize, f64)> = None;
        for t in 0..rows.len() {
            if alloc[t] >= NMAX || spent + share[t] > target + 1e-9 {
                continue;
            }
            let gain = dense[t][alloc[t] - 1] - dense[t][alloc[t]];
            if best.is_none_or(|(_, g)| gain > g) {
                best = Some((t, gain));
            }
        }
        match best {
            Some((t, _)) => alloc[t] += 1,
            None => break,
        }
    }

    let spent: f64 = alloc.iter().zip(&share).map(|(n, w)| *n as f64 * w).sum();
    let opt_sse: f64 = (0..rows.len())
        .map(|t| share[t] * dense[t][alloc[t] - 1])
        .sum();
    let uni_sse: f64 = (0..rows.len()).map(|t| share[t] * dense[t][2]).sum();

    println!("\n== bit-matched optimal allocation (NEW selector, greedy water-fill) ==");
    println!("target Σ share·N = {target:.3}  achieved {spent:.3}");
    for (t, (ty, _, _)) in rows.iter().enumerate() {
        println!("  {ty:<11} N={:<3} (uniform was 3)", alloc[t]);
    }
    let spec: Vec<String> = rows
        .iter()
        .enumerate()
        .map(|(t, (ty, _, _))| format!("{ty}:{}", alloc[t]))
        .collect();
    println!("\n  HIPFIRE_OUTLIERS_BY_LAYER={}", spec.join(","));
    println!(
        "\n  param-weighted mean per-group SSE: uniform N=3 {uni_sse:.7}  \
         optimal {opt_sse:.7}  ({:.2}% better, both {:.4} b/w)",
        100.0 * (uni_sse - opt_sse) / uni_sse,
        4.0625 + spent / 16.0
    );
    println!(
        "\nNOTE: weight SSE is a PROXY. d77fa637a measured oq4.5++ (N=7) as worse \n\
         KLD than N=3 despite more bits — no SSE curve predicts that. Confirm any \n\
         allocation this suggests with a real KLD run before believing it."
    );
}
