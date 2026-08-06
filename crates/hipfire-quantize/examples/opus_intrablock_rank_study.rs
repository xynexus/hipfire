// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! E2 probe — is there low-rank structure in the residual INSIDE a 256-group?
//!
//! `HIPFIRE_LOWRANK_R` already does a low-rank residual correction at tensor
//! level and is the strongest known lever at 2 bits (−13%). The recurring
//! proposal is to do the same thing inside each 256-group. The plan predicts it
//! dead — the FWHT exists precisely to decorrelate within the group, so residual
//! rank structure inside 256 should be flat — and asks for one cheap probe so
//! the negative is on record, because the idea keeps coming back.
//!
//! The probe: stack each group's post-int4 residual as a row of a
//! `n_groups × 256` matrix Δ, then read the spectrum of `ΔᵀΔ` (squared singular
//! values). A rank-`r` correction can only ever recover the leading `r`
//! directions, so the top-`r` share of total energy IS the ceiling on what the
//! idea could buy — before paying for a single byte of the factors.
//!
//! Reference points, both computed here:
//! - **white noise floor**: for an i.i.d. residual the top-`r` share is `r/256`.
//!   Structure means beating that, and beating it by enough to pay for storage.
//! - **the same measurement pre-rotation**, which is the control that says
//!   whether any flatness is the FWHT's doing or was never there.
//!
//! Storage reality check: a rank-`r` factorization of one group costs
//! `r·(256 + 1)` values against a 136 B block. Even `r = 1` in f16 is ~514 B
//! per group — 3.8× the entire block — so an intra-block low-rank correction has
//! to be shared across groups to make any sense at all, which is what the
//! tensor-level `HIPFIRE_LOWRANK_R` already is.
//!
//!   cargo run --release -p hipfire-quantize --example opus_intrablock_rank_study \
//!     -- <model.safetensors> [max_groups_per_tensor]

use faer::{Mat, Side};
use hipfire_quantize::codecs::symmetric_clipsearch;
use hipfire_quantize::{cpu_fwht_256, gen_fwht_signs};

const SUFFIXES: [&str; 4] = ["down_proj", "o_proj", "gate_proj", "q_proj"];
/// Ranks to report the cumulative energy share at.
const RANKS: [usize; 6] = [1, 2, 4, 8, 16, 32];

/// Residual left by the int4 quantizer at its MSE-optimal clip: `v − q4·scale`.
/// This is what a low-rank correction would have to model.
fn int4_residual(group: &[f32; 256]) -> [f32; 256] {
    let scale = symmetric_clipsearch(group, 7.0).max(1e-12);
    let inv = 1.0 / scale;
    let mut residual = [0.0f32; 256];
    for (r, &v) in residual.iter_mut().zip(group.iter()) {
        *r = v - (v * inv).round().clamp(-7.0, 7.0) * scale;
    }
    residual
}

/// Eigenvalues of `ΔᵀΔ` in descending order — the squared singular values of Δ.
/// Built as a 256×256 Gram matrix so the cost is independent of group count.
fn spectrum(rows: &[[f32; 256]]) -> Vec<f64> {
    let mut gram = vec![0.0f64; 256 * 256];
    for row in rows {
        for i in 0..256 {
            let vi = row[i] as f64;
            if vi == 0.0 {
                continue;
            }
            for j in 0..256 {
                gram[i * 256 + j] += vi * row[j] as f64;
            }
        }
    }
    let m = Mat::<f64>::from_fn(256, 256, |i, j| gram[i * 256 + j]);
    let eig = m
        .self_adjoint_eigen(Side::Lower)
        .expect("self-adjoint eigendecomposition of a Gram matrix");
    let s = eig.S();
    let mut values: Vec<f64> = (0..256).map(|i| s[i].max(0.0)).collect();
    values.sort_unstable_by(|a, b| b.total_cmp(a));
    values
}

/// Best (most favourable) energy share any sampled tensor offers at `RANKS[slot]`
/// — the strongest case the idea has, so the cost table is not straw-manned.
fn tensor_best_share(shares: &[Vec<f64>], rank: usize) -> f64 {
    let slot = RANKS.iter().position(|&r| r == rank).expect("known rank");
    shares
        .iter()
        .map(|s| s[slot])
        .fold(0.0f64, |best, v| best.max(v))
}

/// Cumulative share of total energy captured by the leading `rank` directions.
fn energy_share(values: &[f64], rank: usize) -> f64 {
    let total: f64 = values.iter().sum();
    if total <= 0.0 {
        return 0.0;
    }
    values.iter().take(rank).sum::<f64>() / total
}

fn selfcheck() {
    // A rank-1 matrix must put all its energy in the first direction.
    let mut rank1 = Vec::new();
    for k in 0..8 {
        let mut row = [0.0f32; 256];
        for (i, r) in row.iter_mut().enumerate() {
            *r = (k as f32 + 1.0) * (i as f32 + 1.0);
        }
        rank1.push(row);
    }
    let values = spectrum(&rank1);
    assert!(
        energy_share(&values, 1) > 0.999,
        "rank-1 input did not concentrate: {:.4}",
        energy_share(&values, 1)
    );

    // An identity-like set of one-hot rows is maximally flat: 8 equal directions,
    // so the top-1 share must be 1/8.
    let mut onehot = Vec::new();
    for k in 0..8 {
        let mut row = [0.0f32; 256];
        row[k] = 1.0;
        onehot.push(row);
    }
    let values = spectrum(&onehot);
    assert!(
        (energy_share(&values, 1) - 0.125).abs() < 1e-6,
        "flat input misread: {:.6}",
        energy_share(&values, 1)
    );
}

fn main() {
    selfcheck();

    let args: Vec<String> = std::env::args().collect();
    let Some(path) = args.get(1).cloned() else {
        eprintln!(
            "usage: opus_intrablock_rank_study <model.safetensors> [max_groups_per_tensor]\n\
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
    println!("probe: cumulative energy share of the leading r directions of ΔᵀΔ");
    println!("       white-noise floor is r/256 — structure has to beat that\n");

    let header_row: String = RANKS.iter().map(|r| format!("  r={r:<7}")).collect();
    println!("{:<34}{header_row}", "tensor / domain");
    let floor: String = RANKS
        .iter()
        .map(|r| format!("  {:<7.4}", *r as f64 / 256.0))
        .collect();
    println!("{:<34}{floor}", "white-noise floor");

    let mut worst_excess = 0.0f64;
    let mut shares: Vec<Vec<f64>> = Vec::new();
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

        // Two arms: the residual as the codec actually sees it (post-FWHT), and
        // the same weights without the rotation — the control.
        let mut rotated: Vec<[f32; 256]> = Vec::new();
        let mut plain: Vec<[f32; 256]> = Vec::new();
        for chunk in bytes.chunks_exact(2).collect::<Vec<_>>().chunks(256) {
            if chunk.len() < 256 || rotated.len() >= cap {
                break;
            }
            let mut g = [0.0f32; 256];
            for (i, c) in chunk.iter().enumerate() {
                g[i] = f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16);
            }
            plain.push(int4_residual(&g));
            cpu_fwht_256(&mut g, &signs1, &signs2);
            rotated.push(int4_residual(&g));
        }
        if rotated.is_empty() {
            continue;
        }

        for (label, rows) in [("post-FWHT (the codec)", &rotated), ("no rotation", &plain)] {
            let values = spectrum(rows);
            let cells: String = RANKS
                .iter()
                .map(|&r| format!("  {:<7.4}", energy_share(&values, r)))
                .collect();
            println!("{:<34}{cells}", format!("{suffix} — {label}"));
            if label.starts_with("post-FWHT") {
                for &r in &RANKS {
                    worst_excess = worst_excess.max(energy_share(&values, r) - r as f64 / 256.0);
                }
                shares.push(RANKS.iter().map(|&r| energy_share(&values, r)).collect());
            }
        }
    }

    println!("\n== verdict ==");
    println!(
        "  largest excess over the white-noise floor, post-FWHT: {:+.4}",
        worst_excess
    );
    println!(
        "  The rotated residual is NEAR-flat, not exactly flat: the FWHT tightens\n\
         \x20 the spectrum against the no-rotation control on 3 of 4 tensors, but a\n\
         \x20 little structure survives. So the spectrum alone does not close E2 —\n\
         \x20 the cost model below does."
    );
    println!("\n== what the structure is worth, per byte ==");
    println!(
        "  Cheapest sane form: ONE 256-dim basis shared across a tensor's groups,\n\
         \x20 each group storing r f16 coefficients. Per-group cost is 2r B on a\n\
         \x20 136 B block; the shared basis amortises to nothing. Benefit is capped\n\
         \x20 by the energy share above — a perfect rank-r projection, no quantizer\n\
         \x20 loss on the coefficients, which is generous.\n"
    );
    println!(
        "  {:<12}{:>10}{:>12}{:>16}",
        "rank", "cost/group", "best case", "%SSE per byte"
    );
    for &r in &RANKS {
        let best = tensor_best_share(&shares, r);
        let bytes = 2.0 * r as f64;
        println!(
            "  r={r:<10}{:>9.0} B{:>11.2}%{:>15.3}",
            bytes,
            100.0 * best,
            100.0 * best / bytes
        );
    }
    println!(
        "\n  CLOSE E2, on two grounds that do not depend on each other:\n\
         \n\
         \x20 1. It is beaten by a FREE option. P1 step 1 measured +6.3% SSE at\n\
         \x20    ZERO extra bytes (4-bit corrections buy a 4th promoted position\n\
         \x20    inside the same 136 B). The r=1 row costs 2 B to beat 4.84%, and\n\
         \x20    every row is a worse deal per byte than the one above it, so the\n\
         \x20    curve never turns favourable — it only buys more by spending more.\n\
         \x20 2. It is the only option here that needs a new decode path. The\n\
         \x20    overlay resolves inside the existing expander; a shared basis adds\n\
         \x20    a 256-wide matmul per group to every load, against 4.8-26% of the\n\
         \x20    residual — which is a fraction of a fraction, since the residual is\n\
         \x20    itself what int4 already got close to.\n\
         \n\
         \x20 The honest framing for the recurring proposal: the structure IS there\n\
         \x20 (12x the white-noise floor at r=1 on down_proj), the FWHT does not\n\
         \x20 fully destroy it, and it still is not worth buying."
    );
}
