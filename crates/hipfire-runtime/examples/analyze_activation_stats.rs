// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! Turn the calibration's `*.actstats` records into the 4-bit ACTIVATION
//! decision they exist to support.
//!
//! Until these were captured, calibration measured only second moments — Σx²
//! per channel (imatrix) and Σxxᵀ (Hessian). Those describe channel ENERGY,
//! which is what AWQ-style weight scaling wants and which cannot distinguish a
//! channel whose mass is spread evenly (fine in int4) from one whose mass sits
//! in a spike (destroyed by int4). Both have the same Σx².
//!
//! What decides it is the CREST FACTOR, max|x| / rms. Signed int4 has 15 levels,
//! so a symmetric per-group scale set to max|x| leaves rms sitting at
//! 7.5 / crest levels. Rule of thumb: crest ~3 keeps ~2.5 levels of rms
//! resolution and is comfortable; crest ~30 leaves 0.25 and clips or flattens.
//!
//!   cargo run --release -p hipfire-runtime --example analyze_activation_stats -- <artifact.calib.hfq>

use hipfire_runtime::hfq::HfqFile;
use std::path::Path;

fn as_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: <artifact.calib.hfq>");
    let hfq = HfqFile::open(Path::new(&path)).expect("open");

    let names: Vec<String> = hfq
        .tensors()
        .iter()
        .filter(|t| t.name.ends_with(".actstats"))
        .map(|t| t.name.clone())
        .collect();
    if names.is_empty() {
        eprintln!("no .actstats records — this artifact predates activation shape capture");
        std::process::exit(2);
    }
    println!("activation shape statistics from {} tensors\n", names.len());
    println!(
        "  {:<58} {:>7} {:>9} {:>8} {:>9}",
        "tensor", "crest", "kurtosis", "asym", "int4?"
    );

    let mut worst: Vec<(f32, String)> = Vec::new();
    for name in &names {
        let (info, bytes) = hfq.tensor_data(name).expect("data");
        let v = as_f32(&bytes);
        let k = *info.shape.last().unwrap() as usize;
        if v.len() < 4 * k {
            continue;
        }
        let (sum, sumabs, sum4, absmax) =
            (&v[..k], &v[k..2 * k], &v[2 * k..3 * k], &v[3 * k..4 * k]);

        // Σ|x| and Σx⁴ are per-token means; Σx² is not in this record, so rms is
        // recovered from the fourth and first absolute moments only where valid.
        // Use E[x⁴]^(1/4) as a tail-weighted scale and E|x| as the bulk scale;
        // their ratio is a peakedness proxy that needs no second moment.
        let mut max_crest = 0f32;
        let mut sum_kurt = 0f64;
        let mut max_asym = 0f32;
        let mut n = 0usize;
        for c in 0..k {
            let l1 = sumabs[c];
            if l1 <= 1e-12 {
                continue;
            }
            let m4 = sum4[c].max(0.0) as f64;
            let rms_like = m4.powf(0.25) as f32; // tail-weighted scale
            let crest = if rms_like > 1e-12 {
                absmax[c] / rms_like
            } else {
                0.0
            };
            // E[x⁴] / E[|x|]⁴ — 3.0 for a Gaussian-ish bulk, higher = heavy tail.
            let kurt = m4 / (l1 as f64).powi(4).max(1e-30);
            let asym = (sum[c] / l1).abs(); // 0 = symmetric, 1 = one-sided
            max_crest = max_crest.max(crest);
            sum_kurt += kurt.min(1e6);
            max_asym = max_asym.max(asym);
            n += 1;
        }
        if n == 0 {
            continue;
        }
        let kurt = sum_kurt / n as f64;
        let verdict = if max_crest < 6.0 {
            "ok"
        } else if max_crest < 20.0 {
            "tight"
        } else {
            "CLIPS"
        };
        let short = name.trim_end_matches(".actstats");
        let short = short.rsplit_once("layers.").map(|x| x.1).unwrap_or(short);
        println!("  {short:<58} {max_crest:>7.2} {kurt:>9.2} {max_asym:>8.2} {verdict:>9}");
        worst.push((max_crest, short.to_string()));
    }

    worst.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    println!("\n  worst channels by crest factor (these decide whether A4 is viable):");
    for (c, n) in worst.iter().take(8) {
        println!("    {c:>8.2}  {n}");
    }
}
