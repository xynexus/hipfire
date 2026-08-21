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
//! rms comes from the SIBLING `.imatrix` record (Σx²/N), not from the actstats
//! fourth moment. Using `E[x⁴]^(1/4)` as an rms stand-in is tempting and wrong:
//! on heavy-tailed data it sits well ABOVE the true rms, which deflates the
//! crest factor exactly where the tail is worst — i.e. it under-reports risk
//! precisely on the channels the metric exists to find.
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
        "  {:<50} {:>8} {:>8} {:>8} {:>9} {:>8}",
        "tensor", "corpus", "grp-mean", "grp-max", "kurtosis", "int4?"
    );
    println!(
        "  {:<50} {:>8} {:>8} {:>8}",
        "", "(crest)", "(crest)", "(crest)"
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

        // Pair with the imatrix (Σx²/N) for the true rms and true kurtosis.
        let im_name = name.replace(".actstats", ".imatrix");
        let imatrix = hfq.tensor_data(&im_name).map(|(_, b)| as_f32(&b));
        // And with the per-(token, group) crest, which is what the quantizer
        // actually experiences. The corpus-wide column above reduces over
        // tokens and so cannot see that window.
        let gc_name = name.replace(".actstats", ".groupcrest");
        let gc = hfq.tensor_data(&gc_name).map(|(_, b)| as_f32(&b));
        let (gmean, gmax) = match &gc {
            Some(v) if v.len() >= 2 => {
                let h = v.len() / 2;
                let m = v[..h].iter().sum::<f32>() / h as f32;
                let x = v[h..].iter().cloned().fold(0f32, f32::max);
                (m, x)
            }
            _ => (0.0, 0.0),
        };
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
            let m2 = imatrix
                .as_ref()
                .map(|v| v[c].max(0.0) as f64)
                .unwrap_or(0.0);
            let rms = m2.sqrt() as f32;
            let crest = if rms > 1e-12 { absmax[c] / rms } else { 0.0 };
            // True kurtosis E[x⁴]/E[x²]²: 3.0 is Gaussian, higher = heavy tail.
            let kurt = if m2 > 1e-30 { m4 / (m2 * m2) } else { 0.0 };
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
        // Judge on the PER-GROUP max where available — that is the window the
        // scale is chosen in. Signed int4 has 7 positive levels, so rms sits at
        // 7/crest levels: <4 is comfortable, <8 workable, beyond that int4
        // flattens the bulk of the group.
        let judged = if gmax > 0.0 { gmax } else { max_crest };
        let verdict = if judged < 4.0 {
            "ok"
        } else if judged < 8.0 {
            "tight"
        } else {
            "CLIPS"
        };
        let short = name.trim_end_matches(".actstats");
        let short = short.rsplit_once("layers.").map(|x| x.1).unwrap_or(short);
        println!(
            "  {short:<50} {max_crest:>8.1} {gmean:>8.2} {gmax:>8.2} {kurt:>9.1} {verdict:>8}"
        );
        worst.push((judged, short.to_string()));
    }

    worst.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    println!("\n  worst by the PER-GROUP crest the quantizer actually sees:");
    for (c, n) in worst.iter().take(8) {
        println!("    {c:>8.2}  {n}");
    }
}
