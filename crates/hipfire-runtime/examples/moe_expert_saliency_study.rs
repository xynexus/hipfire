// SPDX-License-Identifier: Apache-2.0
//! Q1 of the routed-expert Hessian question: do routed experts actually see
//! DIFFERENT input distributions?
//!
//! Routed experts are captured imatrix-only, so `oq*++` silently RTN-quantizes
//! them (`ldlq: skip`). Fixing that means either a per-expert Hessian (~44 GB
//! stored for qwen3.6-35b-a3b, or a fused per-layer pass at ~1.1 GB) or a
//! single LAYER-POOLED Hessian shared by every expert in a layer (~168 MB, no
//! new pipeline). Pooling trades per-expert specialisation for ~E x more
//! samples.
//!
//! This decides whether specialisation exists at all, using only the per-expert
//! imatrix ALREADY in every MoE calib — zero GPU, zero new capture. The imatrix
//! is diag(XᵀX) = per-channel energy, i.e. the diagonal of the very Hessian
//! under discussion. If experts' diagonals are near-identical, their full
//! Hessians are near-identical too and pooling is strictly better (same
//! structure, E x the samples). If they differ sharply, specialisation is real
//! and a per-expert Hessian could carry information pooling destroys.
//!
//! Run:
//!   cargo run --release -p hipfire-runtime --example moe_expert_saliency_study -- \
//!     --calib ~/.hipfire/calib/zaya1-8b-resident-2ktok.calib.hfq [--layer N]

use hipfire_runtime::hfq::HfqFile;
use std::collections::BTreeMap;
use std::path::Path;

fn arg(flag: &str, default: Option<String>) -> Option<String> {
    let a: Vec<String> = std::env::args().collect();
    a.iter()
        .position(|x| x == flag)
        .and_then(|i| a.get(i + 1).cloned())
        .or(default)
}

fn read_f32(hfq: &HfqFile, name: &str) -> Option<Vec<f32>> {
    let (info, bytes) = hfq.tensor_data_vec(name)?;
    // imatrix tensors are written as plain F32 (quant_type 2).
    if info.quant_type != 2 {
        return None;
    }
    Some(
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
    )
}

/// Cosine similarity between two nonnegative energy vectors.
fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b) {
        dot += *x as f64 * *y as f64;
        na += *x as f64 * *x as f64;
        nb += *y as f64 * *y as f64;
    }
    if na == 0.0 || nb == 0.0 {
        return f64::NAN;
    }
    dot / (na.sqrt() * nb.sqrt())
}

/// L1-normalise so vectors are compared by SHAPE, not by how many rows the
/// expert happened to receive. Without this a hot expert and a cold expert look
/// different purely from row count, which is not specialisation.
fn normalise(v: &[f32]) -> Vec<f32> {
    let s: f64 = v.iter().map(|x| *x as f64).sum();
    if s <= 0.0 {
        return v.to_vec();
    }
    v.iter().map(|x| (*x as f64 / s) as f32).collect()
}

fn main() {
    let calib = arg("--calib", None).expect("--calib <artifact.calib.hfq> required");
    let want_layer: Option<usize> = arg("--layer", None).and_then(|s| s.parse().ok());
    let hfq = HfqFile::open(Path::new(&calib)).expect("open calib");

    // Group per-expert imatrix tensors by (layer, projection).
    // name: <prefix>.layers.<L>.<proj-with-experts.<E>>.imatrix
    let mut groups: BTreeMap<(usize, String), Vec<(usize, String)>> = BTreeMap::new();
    for t in hfq.tensors() {
        let Some(base) = t.name.strip_suffix(".imatrix") else {
            continue;
        };
        let Some((_, rest)) = base.split_once(".layers.") else {
            continue;
        };
        let Some((layer_s, proj)) = rest.split_once('.') else {
            continue;
        };
        let Ok(layer) = layer_s.parse::<usize>() else {
            continue;
        };
        // Only routed experts: `...experts.<N>....`
        let Some((head, tail)) = proj.split_once("experts.") else {
            continue;
        };
        let Some((expert_s, leaf)) = tail.split_once('.') else {
            continue;
        };
        let Ok(expert) = expert_s.parse::<usize>() else {
            continue;
        };
        groups
            .entry((layer, format!("{head}experts.*.{leaf}")))
            .or_default()
            .push((expert, t.name.clone()));
    }

    if groups.is_empty() {
        eprintln!("no routed-expert imatrix tensors in {calib}");
        eprintln!("(this artefact has no MoE experts, or they were not captured)");
        return;
    }

    println!("calib: {calib}");
    println!(
        "routed-expert imatrix groups: {} across {} layer(s)",
        groups.len(),
        groups
            .keys()
            .map(|(l, _)| *l)
            .collect::<std::collections::BTreeSet<_>>()
            .len()
    );
    println!();
    println!("Per-layer, per-projection: how similar are experts' input-energy profiles?");
    println!("cos=1.000 means identical shape (pooling loses nothing).");
    println!();
    println!(
        "  {:<34} {:>7} {:>9} {:>9} {:>9} {:>9}",
        "layer / projection", "experts", "cos_mean", "cos_min", "vs_pooled", "top1_frac"
    );

    let mut all_cos: Vec<f64> = Vec::new();
    let mut all_vs_pooled: Vec<f64> = Vec::new();

    for ((layer, proj), members) in &groups {
        if let Some(w) = want_layer {
            if *layer != w {
                continue;
            }
        }
        let mut vecs: Vec<Vec<f32>> = Vec::new();
        for (_, name) in members {
            if let Some(v) = read_f32(&hfq, name) {
                if v.iter().any(|x| *x > 0.0) {
                    vecs.push(normalise(&v));
                }
            }
        }
        if vecs.len() < 2 {
            continue;
        }
        let k = vecs[0].len();

        // Pooled profile = mean over experts (each already L1-normalised, so
        // this is the shape a layer-pooled Hessian's diagonal would have).
        let mut pooled = vec![0.0f32; k];
        for v in &vecs {
            for (p, x) in pooled.iter_mut().zip(v) {
                *p += *x / vecs.len() as f32;
            }
        }

        // Pairwise cosine across experts.
        let mut cos: Vec<f64> = Vec::new();
        for i in 0..vecs.len() {
            for j in (i + 1)..vecs.len() {
                let c = cosine(&vecs[i], &vecs[j]);
                if c.is_finite() {
                    cos.push(c);
                }
            }
        }
        if cos.is_empty() {
            continue;
        }
        let cos_mean = cos.iter().sum::<f64>() / cos.len() as f64;
        let cos_min = cos.iter().cloned().fold(f64::INFINITY, f64::min);
        let vs_pooled: f64 =
            vecs.iter().map(|v| cosine(v, &pooled)).sum::<f64>() / vecs.len() as f64;

        // Concentration: what share of total energy sits in the single biggest
        // channel? A near-flat profile has little for any Hessian to exploit.
        let top1: f64 = vecs
            .iter()
            .map(|v| v.iter().cloned().fold(0.0f32, f32::max) as f64)
            .sum::<f64>()
            / vecs.len() as f64;

        all_cos.push(cos_mean);
        all_vs_pooled.push(vs_pooled);
        println!(
            "  {:<34} {:>7} {:>9.4} {:>9.4} {:>9.4} {:>9.4}",
            format!("L{layer} {proj}"),
            vecs.len(),
            cos_mean,
            cos_min,
            vs_pooled,
            top1
        );
    }

    if all_cos.is_empty() {
        eprintln!("no group had >=2 readable experts");
        return;
    }
    let m = all_cos.iter().sum::<f64>() / all_cos.len() as f64;
    let worst = all_cos.iter().cloned().fold(f64::INFINITY, f64::min);
    let p = all_vs_pooled.iter().sum::<f64>() / all_vs_pooled.len() as f64;
    println!();
    println!("summary over {} group(s):", all_cos.len());
    println!("  mean pairwise cosine between experts : {m:.4}");
    println!("  worst group mean                     : {worst:.4}");
    println!("  mean cosine of expert vs pooled      : {p:.4}");
    println!();
    println!("Reading: cosine near 1 means experts share an input-energy profile,");
    println!("so a layer-pooled Hessian carries the same structure with E x the");
    println!("samples and per-expert capture buys little. Cosine well below 1");
    println!("means specialisation is real and pooling would destroy it.");
}
