// SPDX-License-Identifier: Apache-2.0
//! Q2 of the routed-expert Hessian question: at a realistic sample:dimension
//! ratio, does an `XᵀX` Hessian generalise off its own calibration corpus?
//!
//! Context. Routed MoE experts are captured imatrix-only, so `oq*++` silently
//! RTN-quantizes them. Giving them Hessians costs either ~44 GB stored or a
//! fused per-layer pass. Q1 (`moe_expert_saliency_study`) showed the two expert
//! projections are different problems: `gate_up` inputs are near-shared across
//! experts (poolable), `down_proj` inputs are near-orthogonal (NOT poolable, so
//! per-expert or nothing).
//!
//! That makes the deciding question for `down_proj`: a per-expert Hessian is
//! built from only the rows routed to that expert — order `n/K ~ 2` for a
//! top-1 16-expert layer at a realistic corpus size. Is that enough samples for
//! `XᵀX` to help on data it was not fitted on?
//!
//! This measures it on the DENSE path, where rows are cheap (every token feeds
//! every dense tensor) and `K` varies 1024..3072 across projections, so one
//! capture sweeps the ratio range. If LDLQ's benefit only appears at ratios far
//! above what a routed expert can reach, per-expert Hessians are not worth
//! their cost regardless of how they are stored.
//!
//! Metric is the quantity LDLQ actually minimises — the H-weighted proxy loss
//!     E(Ŵ; H) = Σ_m (w_m − ŵ_m)ᵀ H (w_m − ŵ_m) / Σ_m w_mᵀ H w_m
//! evaluated on a HELD-OUT `H_test` from a disjoint corpus half, and on the
//! fitting `H_fit` for the in-sample comparison. The gap between the two IS the
//! generalisation question.
//!
//! CAVEAT, carried from `opus_outlier_budget_study`: weight-space proxy loss is
//! not KLD, and the two have demonstrably disagreed in this codebase before.
//! This ranks a mechanism, it does not certify a format.
//!
//! Run:
//!   cargo run --release -p hipfire-quantize --example hessian_generalisation_study -- \
//!     --fits a.calib.hfq,b.calib.hfq --test t.calib.hfq [--safetensors PATH]

use hipfire_quantize::codecs::{dequant_oq4g256, quantize_oq4g256};
use hipfire_quantize::gen_fwht_signs;
use hipfire_quantize::hessian_io::HessianSidecar;
use hipfire_quantize::ldlq;
use rayon::prelude::*;
use std::path::Path;

fn arg(flag: &str, default: Option<String>) -> Option<String> {
    let a: Vec<String> = std::env::args().collect();
    a.iter()
        .position(|x| x == flag)
        .and_then(|i| a.get(i + 1).cloned())
        .or(default)
}

/// Dense row-major [K,K] f32 from a calib package's compact Hessian.
fn dense_hessian(pkg: &HessianSidecar, name: &str, k: usize) -> Option<Vec<f32>> {
    let h = pkg.get(name, 0)?;
    let mut out = vec![0.0f32; k * k];
    for i in 0..k {
        for j in 0..k {
            out[i * k + j] = h.at(i, j) as f32;
        }
    }
    Some(out)
}

/// Σ_m d_mᵀ H d_m for D = [M,K] row-major.
fn h_weighted_energy(d: &[f32], m: usize, k: usize, h: &[f32]) -> f64 {
    (0..m)
        .into_par_iter()
        .map(|row| {
            let dr = &d[row * k..(row + 1) * k];
            let mut acc = 0.0f64;
            for i in 0..k {
                let di = dr[i] as f64;
                if di == 0.0 {
                    continue;
                }
                let hrow = &h[i * k..(i + 1) * k];
                let mut inner = 0.0f64;
                for j in 0..k {
                    inner += hrow[j] as f64 * dr[j] as f64;
                }
                acc += di * inner;
            }
            acc
        })
        .sum()
}

struct Tensor {
    name: String,
    m: usize,
    k: usize,
    w: Vec<f32>,
}

fn load_bf16_tensors(path: &str, want: &[&str]) -> Vec<Tensor> {
    let file = std::fs::File::open(path).expect("open safetensors");
    let mmap = unsafe { memmap2::Mmap::map(&file).expect("mmap") };
    let hlen = u64::from_le_bytes(mmap[0..8].try_into().unwrap()) as usize;
    let header: serde_json::Value =
        serde_json::from_slice(&mmap[8..8 + hlen]).expect("parse header");
    let base = 8 + hlen;
    let mut out = Vec::new();
    for name in want {
        let key = format!("{name}.weight");
        let meta = &header[&key];
        if meta.is_null() || meta["dtype"].as_str() != Some("BF16") {
            continue;
        }
        let shape: Vec<usize> = meta["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        if shape.len() != 2 {
            continue;
        }
        let off: Vec<usize> = meta["data_offsets"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        let bytes = &mmap[base + off[0]..base + off[1]];
        let w: Vec<f32> = bytes
            .chunks_exact(2)
            .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
            .collect();
        out.push(Tensor {
            name: name.to_string(),
            m: shape[0],
            k: shape[1],
            w,
        });
    }
    out
}

fn main() {
    let fits = arg("--fits", None).expect("--fits a.calib.hfq,b.calib.hfq required");
    let test = arg("--test", None).expect("--test t.calib.hfq required");
    let st = arg(
        "--safetensors",
        Some(
            "/srv/huggingface/models--Qwen--Qwen3.5-0.8B/snapshots/\
             2fc06364715b967f1860aea9cf38778875588b17/model.safetensors-00001-of-00001.safetensors"
                .to_string(),
        ),
    )
    .unwrap();
    let layer = arg("--layer", Some("12".into())).unwrap();

    // Three projections whose K spans 3.5x (1024 / 2048 / 3584 on qwen3.5-0.8b),
    // so a single capture sweeps the sample:dimension ratio without needing
    // proportionally more tokens. Layer 12 is a linear-attn layer on this model,
    // hence `linear_attn.out_proj` rather than `self_attn.o_proj`.
    let names: Vec<String> = ["mlp.gate_proj", "linear_attn.out_proj", "mlp.down_proj"]
        .iter()
        .map(|p| format!("model.language_model.layers.{layer}.{p}"))
        .collect();
    let refs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
    let tensors = load_bf16_tensors(&st, &refs);
    if tensors.is_empty() {
        eprintln!("no BF16 2-D tensors matched in {st}");
        eprintln!("tried: {refs:?}");
        return;
    }

    let signs1 = gen_fwht_signs(42, 256);
    let signs2 = gen_fwht_signs(1042, 256);
    let test_pkg = HessianSidecar::open(Path::new(&test)).expect("open --test calib");

    println!("held-out corpus half: {test}");
    println!("weights: {st}\n");
    println!(
        "  {:<34} {:>6} {:>7} {:>7} {:>11} {:>11} {:>9}",
        "tensor / fit-n", "K", "n", "n/K", "in-sample", "held-out", "vs RTN"
    );

    for t in &tensors {
        if !t.k.is_multiple_of(256) {
            println!("  {:<34} K={} not a multiple of 256, skipped", t.name, t.k);
            continue;
        }
        let Some(h_test) = dense_hessian(&test_pkg, &t.name, t.k) else {
            println!("  {:<34} absent from held-out calib", t.name);
            continue;
        };
        let denom_test = h_weighted_energy(&t.w, t.m, t.k, &h_test);

        // Baseline: RTN, no Hessian at all.
        let rtn = dequant_oq4g256(
            &quantize_oq4g256(&t.w, &signs1, &signs2),
            t.w.len(),
            &signs1,
            &signs2,
        );
        let d_rtn: Vec<f32> = t.w.iter().zip(&rtn).map(|(a, b)| a - b).collect();
        let rtn_test = h_weighted_energy(&d_rtn, t.m, t.k, &h_test) / denom_test;
        println!(
            "  {:<34} {:>6} {:>7} {:>7} {:>11} {:>11.6} {:>9}",
            format!("{} [RTN]", t.name.rsplit('.').next().unwrap()),
            t.k,
            "-",
            "-",
            "-",
            rtn_test,
            "1.000x"
        );

        for fit in fits.split(',') {
            let Ok(fit_pkg) = HessianSidecar::open(Path::new(fit)) else {
                continue;
            };
            let Some(h_fit) = dense_hessian(&fit_pkg, &t.name, t.k) else {
                continue;
            };
            // Token count is recorded per artefact; use it for the ratio.
            let n = fit
                .rsplit('.')
                .nth(2)
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(0);

            let diag_sum: f64 = (0..t.k).map(|i| h_fit[i * t.k + i] as f64).sum();
            let damp = 0.01 * (diag_sum / t.k as f64).max(1e-12);
            let Some(packed) = ldlq::oq4_ldlq_pack(&t.w, t.m, t.k, &h_fit, &signs1, &signs2, damp)
            else {
                println!("  {:<34} ldlq failed (singular H?)", t.name);
                continue;
            };
            let ld = dequant_oq4g256(&packed, t.w.len(), &signs1, &signs2);
            let d_ld: Vec<f32> = t.w.iter().zip(&ld).map(|(a, b)| a - b).collect();

            let denom_fit = h_weighted_energy(&t.w, t.m, t.k, &h_fit);
            let in_sample = h_weighted_energy(&d_ld, t.m, t.k, &h_fit) / denom_fit;
            let held_out = h_weighted_energy(&d_ld, t.m, t.k, &h_test) / denom_test;

            println!(
                "  {:<34} {:>6} {:>7} {:>7.2} {:>11.6} {:>11.6} {:>8.3}x",
                format!("  ldlq n={n}"),
                t.k,
                n,
                n as f64 / t.k as f64,
                in_sample,
                held_out,
                held_out / rtn_test
            );
        }
        println!();
    }

    println!("Reading: 'vs RTN' below 1.000 means the Hessian helped on data it");
    println!("was not fitted on. Compare in-sample against held-out: a large gap");
    println!("is LDLQ fitting corpus noise. A routed expert can only reach the");
    println!("LOW n/K rows of this table.");
}
