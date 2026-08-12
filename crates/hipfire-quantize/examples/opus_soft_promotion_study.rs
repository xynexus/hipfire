// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! E3 gate — is a LOW base with many soft promotions better than W4 with three
//! hard ones, at the same bytes?
//!
//! The platform premise is that weight bytes are the decode lever (the W3A4
//! result), so the plan's E3 proposes promoting to W5/W6 instead of W8, and
//! pairing it with a lower base: "a W3 base with a W5 overlay is a smaller
//! artifact than W4+W8 at comparable error".
//!
//! That costs a W5 grid in `opus_lowbit.rs` and a W3 decode GEMV before a single
//! number exists. This study produces the number first. Every arm is scored at
//! the SAME byte budget as the shipped `oq4.25++` block, so the comparison is
//! the one the format decision actually turns on:
//!
//! ```text
//! bits = 16 (f16 scale) + 256·base + n_out·(8 index + overlay)
//! ```
//!
//! Dropping the base from W4 to W3 frees 32 B per group, which at 2 B an entry
//! buys ~16 more promoted positions — so the arms differ wildly in `n_out` and
//! that is the point. The selector is the P2 joint `(scale, set)` search
//! generalised to arbitrary base/overlay widths: error is separable across
//! positions, so at a fixed scale the best set is the top `n_out` by promotion
//! gain, and sweeping the scale grid around that is the true joint minimum.
//!
//! Reported against the shipped arm (W4 base, W8 overlay, n_out=3). A win here
//! justifies the kernel work; a loss kills E3 for the cost of one example.
//!
//!   cargo run --release -p hipfire-quantize --example opus_soft_promotion_study \
//!     -- <model.safetensors> [max_groups_per_tensor]

use hipfire_quantize::{cpu_fwht_256, gen_fwht_signs};

/// Byte budget per 256-group — the shipped `oq4.25++` block (4.25 b/w).
const BUDGET_BYTES: usize = 136;

/// `codecs::MIXED_CLIP_GRID`, private there. Every arm searches the same grid.
const CLIP_GRID: [f32; 14] = [
    1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35,
];

const SUFFIXES: [&str; 4] = ["down_proj", "o_proj", "gate_proj", "q_proj"];

/// (base bits, overlay bits). The overlay stores an absolute signed value, so
/// `overlay <= base` would be pointless and is excluded.
const ARMS: [(u32, u32); 10] = [
    (4, 8), // the shipped arm
    (4, 6),
    (4, 5),
    (3, 8),
    (3, 6),
    (3, 5),
    (3, 4),
    (2, 8),
    (2, 6),
    (2, 5),
];

/// Largest signed magnitude at `bits` (W4 → 7, W3 → 3, W2 → 1).
fn qmax(bits: u32) -> f32 {
    ((1u32 << (bits - 1)) - 1) as f32
}

/// Promoted positions the budget affords: `16 + 256·base + n·(8 + overlay)` bits.
fn affordable_n_out(base: u32, overlay: u32) -> usize {
    let fixed_bits = 16 + 256 * base as usize;
    let budget_bits = BUDGET_BYTES * 8;
    if fixed_bits >= budget_bits {
        return 0;
    }
    (budget_bits - fixed_bits) / (8 + overlay as usize)
}

/// Actual bytes an arm occupies at `n_out` — never above the budget, and worth
/// printing because the arms round off differently.
fn arm_bytes(base: u32, overlay: u32, n_out: usize) -> usize {
    (16 + 256 * base as usize + n_out * (8 + overlay as usize)).div_ceil(8)
}

/// SSE of a group where `n_out` positions clamp to the overlay grid and the
/// rest to the base grid. Generalises `codecs::mixed_overlay_error`, which is
/// this function with `(base, overlay) = (4, 8)`.
fn arm_error(
    group: &[f32; 256],
    scale: f32,
    promoted: &[bool; 256],
    base: f32,
    overlay: f32,
) -> f32 {
    let inv = 1.0 / scale.max(1e-12);
    group
        .iter()
        .enumerate()
        .map(|(index, &value)| {
            let limit = if promoted[index] { overlay } else { base };
            let error = value - (value * inv).round().clamp(-limit, limit) * scale;
            error * error
        })
        .sum()
}

/// Top-`n_out` positions by promotion gain at a fixed scale — exact, since the
/// error is separable across positions.
fn arm_indices(
    group: &[f32; 256],
    scale: f32,
    n_out: usize,
    base: f32,
    overlay: f32,
) -> [bool; 256] {
    let inv = 1.0 / scale.max(1e-12);
    let mut scored: Vec<(usize, f32)> = (0..256)
        .map(|index| {
            let value = group[index];
            let q = value * inv;
            let e_base = value - q.round().clamp(-base, base) * scale;
            let e_ovl = value - q.round().clamp(-overlay, overlay) * scale;
            (index, e_base * e_base - e_ovl * e_ovl)
        })
        .collect();
    scored.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    let mut promoted = [false; 256];
    for &(index, _) in scored.iter().take(n_out) {
        promoted[index] = true;
    }
    promoted
}

/// Joint `(scale, set)` minimum for one arm — P2's search, widened.
fn arm_clipsearch(group: &[f32; 256], n_out: usize, base_bits: u32, overlay_bits: u32) -> f32 {
    let (base, overlay) = (qmax(base_bits), qmax(overlay_bits));
    let amax = group.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    let mut best = f32::INFINITY;
    for clip in CLIP_GRID {
        let scale = (clip * amax / base).max(1e-12);
        let promoted = arm_indices(group, scale, n_out, base, overlay);
        best = best.min(arm_error(group, scale, &promoted, base, overlay));
    }
    best
}

fn selfcheck() {
    assert_eq!(qmax(4), 7.0);
    assert_eq!(qmax(3), 3.0);
    assert_eq!(qmax(2), 1.0);
    // The shipped arm must reproduce the known point: W4 base, W8 overlay, 3
    // promotions, 136 B exactly.
    assert_eq!(affordable_n_out(4, 8), 3);
    assert_eq!(arm_bytes(4, 8, 3), 136);
    // Dropping to a W3 base frees 32 B, which at 2 B an entry buys 16 more.
    assert_eq!(affordable_n_out(3, 8), 19);
    assert!(arm_bytes(3, 8, 19) <= BUDGET_BYTES);
    // A narrower overlay buys more entries still, and none may exceed budget.
    for (base, overlay) in ARMS {
        let n = affordable_n_out(base, overlay);
        assert!(
            arm_bytes(base, overlay, n) <= BUDGET_BYTES,
            "W{base}+W{overlay} n={n} overruns the budget"
        );
        assert!(
            arm_bytes(base, overlay, n + 1) > BUDGET_BYTES,
            "W{base}+W{overlay} could afford one more than n={n}"
        );
    }
    // Promoting every position to the same width as the base must be a no-op.
    let mut group = [0.0f32; 256];
    for (i, g) in group.iter_mut().enumerate() {
        *g = (i as f32 * 0.017).sin();
    }
    let none = arm_clipsearch(&group, 0, 4, 8);
    let same = arm_clipsearch(&group, 8, 4, 4);
    assert!(
        (none - same).abs() / none < 1e-5,
        "W4 overlay on a W4 base changed the error: {none} vs {same}"
    );
}

fn main() {
    selfcheck();

    let args: Vec<String> = std::env::args().collect();
    let Some(path) = args.get(1).cloned() else {
        eprintln!(
            "usage: opus_soft_promotion_study <model.safetensors> [max_groups_per_tensor]\n\
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
    let base_off = 8 + hlen;
    let signs1 = gen_fwht_signs(42, 256);
    let signs2 = gen_fwht_signs(1042, 256);

    println!("model: {path}");
    println!("cap:   {cap} groups per tensor");
    println!("budget: {BUDGET_BYTES} B/group — the shipped oq4.25++ block\n");

    println!("== what the budget buys each arm ==");
    println!("  {:<12}{:>8}{:>10}{:>12}", "arm", "n_out", "bytes", "b/w");
    for (base, overlay) in ARMS {
        let n = affordable_n_out(base, overlay);
        let bytes = arm_bytes(base, overlay, n);
        println!(
            "  W{base}+W{overlay:<9}{n:>8}{bytes:>10}{:>12.4}",
            8.0 * bytes as f64 / 256.0
        );
    }
    println!();

    let mut totals: Vec<(String, Vec<f64>)> = Vec::new();
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
        let bytes = &mmap[base_off + off[0]..base_off + off[1]];
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

        let sums: Vec<f64> = ARMS
            .iter()
            .map(|&(base, overlay)| {
                let n = affordable_n_out(base, overlay);
                groups
                    .iter()
                    .map(|g| arm_clipsearch(g, n, base, overlay) as f64)
                    .sum::<f64>()
            })
            .collect();
        totals.push((suffix.to_string(), sums));
    }

    println!("== SSE vs the shipped W4+W8 arm, at equal bytes (negative = worse) ==");
    let head: String = ARMS
        .iter()
        .map(|(b, o)| format!("{:>10}", format!("W{b}+W{o}")))
        .collect();
    println!("  {:<12}{head}", "tensor");
    for (suffix, sums) in &totals {
        let shipped = sums[0];
        let cells: String = sums
            .iter()
            .map(|s| format!("{:>9.2}%", 100.0 * (shipped - s) / shipped))
            .collect();
        println!("  {suffix:<12}{cells}");
    }

    let best_arm = (0..ARMS.len())
        .map(|i| {
            let worst = totals
                .iter()
                .map(|(_, s)| 100.0 * (s[0] - s[i]) / s[0])
                .fold(f64::INFINITY, f64::min);
            (i, worst)
        })
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .expect("at least one arm");

    let (index, margin) = best_arm;
    let (base, overlay) = ARMS[index];
    println!(
        "\n== verdict ==\n  best arm by WORST-tensor margin: W{base}+W{overlay} \
         at n_out={}, {margin:+.2}%",
        affordable_n_out(base, overlay)
    );
    println!(
        "  {}",
        if index == 0 || margin <= 0.0 {
            "E3 does not clear — a lower base with soft promotions does not beat\n\
             \x20 W4+W8 on every tensor, so the W3 GEMV and W5 grid are not worth\n\
             \x20 building on this evidence."
        } else {
            "E3 CLEARS on weight SSE — the kernel work has a measured reason.\n\
             \x20 Confirm with KLD at matched bytes before committing to a format."
        }
    );
    println!(
        "\nNOTE: weight SSE is a proxy, and a lower base changes the ACTIVATION\n\
         story too (W3A4 needed a learned rotation, per opus-quant.md §7). This\n\
         study bounds the weight side only."
    );
}
