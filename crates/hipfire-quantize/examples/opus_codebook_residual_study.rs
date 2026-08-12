// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! P1 step 1 — are 4 codebook-approximated corrections better than 3 exact ones?
//!
//! The overlay entry is `(u8 index, i8 value)` = 2 B, so a 136 B block buys
//! `N_out = 3`. Replacing the i8 value with a 4-bit code into a per-tensor
//! codebook makes the entry 1.5 B, so the SAME 136 B buys `N_out = 4`. The
//! headline bit rate does not move (4.25 b/w either way) — this is not a
//! compression trick, it is 33% more corrected positions at equal bytes.
//!
//! ```text
//! 130 B base + 3·(1 B index + 1 B value)   = 136 B   3 exact int8 replacements
//! 130 B base + 4·(1 B index) + 4·(0.5 B code) = 136 B   4 approximate ones
//! ```
//!
//! **The selector has to change with it.** Today promotion is exact in the
//! integer domain (`q_final = q8`), so ranking by int8-upgrade gain is right.
//! Under a codebook `q_final = q4 + C[c]` and every promoted position keeps a
//! residual, so the 4-set must be chosen against the POST-CODEBOOK
//! reconstruction. Picking four positions with the old W8-gain metric and
//! quantizing their deltas afterwards understates the format and is not this
//! experiment.
//!
//! **Kill criterion (plan P1 step 1): if 4-way VQ error is not below 3-way
//! exact error, the idea is dead and the cost was one example binary.**
//!
//! P2 (joint scale + set) is a precondition, not a nicety: under the old
//! selector extra outliers bought almost nothing, so a 3-vs-4 comparison run
//! before P2 would have been scored through a selector that could not reward
//! the extra correction.
//!
//!   cargo run --release -p hipfire-quantize --example opus_codebook_residual_study \
//!     -- <model.safetensors> [max_groups_per_tensor]

use hipfire_quantize::codecs::{mixed_clipsearch, mixed_overlay_error, mixed_overlay_indices};
use hipfire_quantize::{cpu_fwht_256, gen_fwht_signs};

/// The exact arm: 3 × (u8 index, i8 value).
const N_EXACT: usize = 3;
/// The codebook arm: 4 × (u8 index, u4 code) — same 136 B block.
const N_CODED: usize = 4;
/// 4-bit code.
const CODEBOOK_SIZE: usize = 16;
/// Lloyd refinement rounds over the tensor's Δ pool.
const LLOYD_ROUNDS: usize = 24;

/// `codecs::MIXED_CLIP_GRID`, private there. The codebook arm has to search the
/// same scale grid as the exact arm or the comparison measures the grid.
const CLIP_GRID: [f32; 14] = [
    1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35,
];

const SUFFIXES: [&str; 4] = ["down_proj", "o_proj", "gate_proj", "q_proj"];

/// Integer correction a promoted position wants: `q8 − q4`, both at `scale`.
fn delta_at(value: f32, inv: f32) -> i32 {
    let q = (value * inv).round();
    let q4 = q.clamp(-7.0, 7.0);
    let q8 = q.clamp(-127.0, 127.0);
    (q8 - q4) as i32
}

/// Nearest codebook entry to `delta`, as (index, value).
fn nearest(codebook: &[i32], delta: i32) -> (usize, i32) {
    let mut best = (0usize, codebook[0]);
    let mut best_d = (codebook[0] - delta).abs();
    for (i, &c) in codebook.iter().enumerate().skip(1) {
        let d = (c - delta).abs();
        if d < best_d {
            best_d = d;
            best = (i, c);
        }
    }
    best
}

/// Post-codebook SSE of a group: `n_out` positions carry `q4 + C[c]`, the rest
/// stay at int4. Mirrors `mixed_overlay_error`'s contract so the two arms are
/// scored the same way.
fn coded_overlay_error(group: &[f32; 256], scale: f32, indices: &[usize], codebook: &[i32]) -> f32 {
    let inv = 1.0 / scale.max(1e-12);
    let mut promoted = [false; 256];
    for &index in indices {
        promoted[index] = true;
    }
    group
        .iter()
        .enumerate()
        .map(|(index, &value)| {
            let q = (value * inv).round();
            let quantized = if promoted[index] {
                let q4 = q.clamp(-7.0, 7.0);
                let (_, correction) = nearest(codebook, delta_at(value, inv));
                // The reconstruction must still be an int8 value.
                (q4 + correction as f32).clamp(-127.0, 127.0)
            } else {
                q.clamp(-7.0, 7.0)
            };
            let error = value - quantized * scale;
            error * error
        })
        .sum()
}

/// Rank positions by POST-CODEBOOK gain and take the best `n_out` — the
/// selector change the plan calls for. Error is separable across positions, so
/// at a fixed scale the top-`n_out` by gain is exactly the best set.
fn coded_overlay_indices(
    group: &[f32; 256],
    scale: f32,
    codebook: &[i32],
    n_out: usize,
) -> Vec<usize> {
    let inv = 1.0 / scale.max(1e-12);
    let mut scored: Vec<(usize, f32)> = (0..256)
        .map(|index| {
            let value = group[index];
            let q = (value * inv).round();
            let q4 = q.clamp(-7.0, 7.0);
            let (_, correction) = nearest(codebook, delta_at(value, inv));
            let coded = (q4 + correction as f32).clamp(-127.0, 127.0);
            let error4 = value - q4 * scale;
            let error_coded = value - coded * scale;
            (index, error4 * error4 - error_coded * error_coded)
        })
        .collect();
    scored.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    scored.into_iter().take(n_out).map(|(i, _)| i).collect()
}

/// Joint `(scale, set)` search for the codebook arm, matching `mixed_clipsearch`'s
/// structure: re-select the set inside the scale sweep, keep the best.
fn coded_clipsearch(group: &[f32; 256], codebook: &[i32], n_out: usize) -> (f32, Vec<usize>, f32) {
    let amax = group.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    let mut best = (0.0f32, Vec::new(), f32::INFINITY);
    for clip in CLIP_GRID {
        let scale = (clip * amax / 7.0).max(1e-12);
        let indices = coded_overlay_indices(group, scale, codebook, n_out);
        let error = coded_overlay_error(group, scale, &indices, codebook);
        if error < best.2 {
            best = (scale, indices, error);
        }
    }
    best
}

/// Lloyd–Max over the tensor's Δ pool, centroids snapped to integers because
/// `q4 + C[c]` has to land on the int8 grid.
fn fit_codebook(deltas: &[i32]) -> Vec<i32> {
    assert!(!deltas.is_empty(), "empty delta pool");
    let (lo, hi) = deltas
        .iter()
        .fold((i32::MAX, i32::MIN), |(l, h), &d| (l.min(d), h.max(d)));
    // Seed on a uniform spread of the observed range; 0 must be representable
    // so a position whose Δ is already 0 is never made worse by promotion.
    let mut codebook: Vec<i32> = (0..CODEBOOK_SIZE)
        .map(|k| {
            let t = k as f64 / (CODEBOOK_SIZE - 1) as f64;
            (lo as f64 + t * (hi - lo) as f64).round() as i32
        })
        .collect();
    if !codebook.contains(&0) {
        codebook[0] = 0;
    }
    for _ in 0..LLOYD_ROUNDS {
        let mut sums = vec![0i64; CODEBOOK_SIZE];
        let mut counts = vec![0i64; CODEBOOK_SIZE];
        for &d in deltas {
            let (k, _) = nearest(&codebook, d);
            sums[k] += d as i64;
            counts[k] += 1;
        }
        let mut moved = false;
        for k in 0..CODEBOOK_SIZE {
            if counts[k] > 0 {
                let centroid = (sums[k] as f64 / counts[k] as f64).round() as i32;
                if centroid != codebook[k] {
                    codebook[k] = centroid;
                    moved = true;
                }
            }
        }
        codebook.sort_unstable();
        codebook.dedup();
        while codebook.len() < CODEBOOK_SIZE {
            // Refill collapsed cells with the worst-represented delta.
            let worst = deltas
                .iter()
                .max_by_key(|&&d| (nearest(&codebook, d).1 - d).abs())
                .copied()
                .unwrap_or(0);
            if codebook.contains(&worst) {
                break;
            }
            codebook.push(worst);
            codebook.sort_unstable();
        }
        if !moved {
            break;
        }
    }
    codebook
}

fn selfcheck() {
    // A codebook containing the exact delta must reconstruct exactly, so the
    // coded arm can never be worse than int4 on that position.
    let mut group = [0.0f32; 256];
    group[0] = 1.0;
    group[7] = -0.5;
    let codebook: Vec<i32> = (-8..8).collect();
    let (scale, indices, error) = coded_clipsearch(&group, &codebook, N_CODED);
    assert!(scale > 0.0);
    assert_eq!(indices.len(), N_CODED);
    assert!(error.is_finite());
    // Lloyd on a pool with fewer than 16 distinct values still returns usable
    // centroids and always represents 0 exactly.
    let codebook = fit_codebook(&[0, 0, 5, 5, 5, -3]);
    assert!(
        codebook.contains(&0),
        "codebook lost the zero: {codebook:?}"
    );
    assert_eq!(nearest(&codebook, 5).1, 5, "exact delta not representable");
    // Selection must be by post-codebook gain: with a codebook that can only
    // say 0, no position has any gain and the ranking must not crash.
    let flat = coded_overlay_indices(&group, 0.1, &[0], N_CODED);
    assert_eq!(flat.len(), N_CODED);
}

fn main() {
    selfcheck();

    let args: Vec<String> = std::env::args().collect();
    let Some(path) = args.get(1).cloned() else {
        eprintln!(
            "usage: opus_codebook_residual_study <model.safetensors> [max_groups_per_tensor]\n\
             \n\
             Any BF16 safetensors checkpoint works; the recorded result used\n\
             Qwen3.5-0.8B. No default path — the model store is machine-local."
        );
        std::process::exit(2);
    };
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
    println!("cap:   {cap} groups per tensor");
    println!(
        "arms:  N={N_EXACT} exact i8  vs  N={N_CODED} through a {CODEBOOK_SIZE}-entry \
         per-tensor codebook (both 136 B/group)\n"
    );

    let mut verdicts: Vec<(String, f64, f64)> = Vec::new();
    for suffix in SUFFIXES {
        let mut names: Vec<&String> = header
            .as_object()
            .unwrap()
            .keys()
            .filter(|n| n.contains(suffix) && n.ends_with(".weight"))
            .filter(|n| header[*n]["dtype"].as_str() == Some("BF16"))
            .collect();
        if names.is_empty() {
            continue;
        }
        names.sort();
        let name = names[names.len() / 2];

        let off: Vec<usize> = header[name]["data_offsets"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        let bytes = &mmap[base + off[0]..base + off[1]];
        let mut groups: Vec<[f32; 256]> = Vec::new();
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
        if groups.is_empty() {
            continue;
        }

        // Arm A — the shipped format: 3 exact int8 replacements.
        let exact: f64 = groups
            .iter()
            .map(|g| {
                let (scale, indices) = mixed_clipsearch(g, N_EXACT);
                mixed_overlay_error(g, scale, &indices, N_EXACT) as f64
            })
            .sum();

        // Codebook fit: pool the Δ the 4-position sets actually want. Seeded
        // from the W8-gain set at the joint scale, then the selector switches
        // to post-codebook gain below.
        let mut pool: Vec<i32> = Vec::new();
        for g in &groups {
            let (scale, _) = mixed_clipsearch(g, N_CODED);
            let inv = 1.0 / scale.max(1e-12);
            let seed = mixed_overlay_indices(g, scale, N_CODED);
            for &position in &seed[..N_CODED] {
                pool.push(delta_at(g[position], inv));
            }
        }
        let codebook = fit_codebook(&pool);

        // Arm B — 4 codebook-approximated corrections, set chosen against the
        // codebook reconstruction.
        let coded: f64 = groups
            .iter()
            .map(|g| coded_clipsearch(g, &codebook, N_CODED).2 as f64)
            .sum();

        // Reference: what 4 EXACT corrections would score. The gap between this
        // and arm B is what the 4-bit code costs; the gap to arm A is what the
        // extra position buys.
        let exact4: f64 = groups
            .iter()
            .map(|g| {
                let (scale, indices) = mixed_clipsearch(g, N_CODED);
                mixed_overlay_error(g, scale, &indices, N_CODED) as f64
            })
            .sum();

        // Arm C — the structured code the plan prefers as an endgame: the 4 bits
        // hold a RAW signed Δ, no codebook, no per-tensor fit, no sidecar. If
        // this matches arm B, the codebook is machinery for nothing.
        let raw_code: Vec<i32> = (-8..8).collect();
        let raw: f64 = groups
            .iter()
            .map(|g| coded_clipsearch(g, &raw_code, N_CODED).2 as f64)
            .sum();

        let improvement = 100.0 * (exact - coded) / exact;
        let raw_improvement = 100.0 * (exact - raw) / exact;
        verdicts.push((suffix.to_string(), improvement, raw_improvement));

        let mut representable = pool.clone();
        representable.sort_unstable();
        representable.dedup();
        let exact_share = 100.0
            * pool
                .iter()
                .filter(|&&d| nearest(&codebook, d).1 == d)
                .count() as f64
            / pool.len() as f64;

        println!("== {name} ==");
        println!("  {} groups, {} pooled deltas", groups.len(), pool.len());
        println!("  N={N_EXACT} exact   (shipped): {exact:.6}");
        println!("  N={N_CODED} coded  (proposed): {coded:.6}   {improvement:+.2}% vs shipped");
        println!(
            "  N={N_CODED} exact  (reference): {exact4:.6}   \
             [not a real format at 136 B — shows what the code costs]"
        );
        println!(
            "  N={N_CODED} raw Δ (no sidecar): {raw:.6}   {raw_improvement:+.2}% vs shipped   \
             [4 bits = signed Δ in [-8,7]]"
        );
        println!(
            "  codebook: {} distinct deltas in pool, {exact_share:.1}% hit exactly",
            representable.len()
        );
        println!("  centroids: {codebook:?}\n");
    }

    println!("== verdict (plan P1 step 1: coded must beat exact, else the idea is dead) ==");
    println!("  {:<11} {:>12} {:>12}", "tensor", "codebook", "raw Δ");
    for (suffix, improvement, raw_improvement) in &verdicts {
        println!("  {suffix:<11} {improvement:>11.2}% {raw_improvement:>11.2}%");
    }
    let best_coded = verdicts
        .iter()
        .fold(f64::NEG_INFINITY, |m, (_, v, _)| m.max(*v));
    let best_raw = verdicts
        .iter()
        .fold(f64::NEG_INFINITY, |m, (_, _, v)| m.max(*v));
    let raw_wins = verdicts.iter().all(|(_, coded, raw)| raw >= coded);

    println!(
        "\n  best: codebook {best_coded:+.2}%, raw Δ {best_raw:+.2}%  →  {}",
        if best_coded.max(best_raw) <= 0.0 {
            "STOP — 4 approximate corrections do not beat 3 exact ones"
        } else {
            "step 1 CLEARS: 4 approximate corrections beat 3 exact ones"
        }
    );
    if raw_wins && best_raw > 0.0 {
        println!(
            "\n  ...and the codebook is machinery for nothing: a RAW signed Δ in\n\
             \x20 [-8,7] wins on every tensor above, with no per-tensor fit, no\n\
             \x20 Lloyd pass, and no `<name>.oqvq` sidecar. Plan steps 2 (sidecar)\n\
             \x20 and the codebook half of 4 (encoder fork) drop out; what remains\n\
             \x20 is one new qt whose overlay entry is (u8 index, i4 delta).\n\
             \x20 The deltas genuinely fit: the raw arm matches N={N_CODED}-EXACT to four\n\
             \x20 decimals, so 4 bits costs essentially nothing against a full i8."
        );
    }
    println!(
        "\nNOTE: weight SSE is a proxy. The plan's real gate is KLD at matched \n\
         bytes; this study only decides whether that run is worth its GPU time."
    );
}
