// SPDX-License-Identifier: Apache-2.0
//! Is activation quant better split across the rotation — one pass before, one
//! after — than done once on either side?
//!
//! The motivating tension. Activation outliers are **sparse in the channel
//! basis**: a few large channels per group. A rotation Gaussianizes them, which
//! is what lets one shared int4 scale work — but it also destroys the sparsity,
//! so after rotating there is nothing left for a cheap sparse overlay to catch.
//! The two mechanisms attack the same defect from opposite sides:
//!
//!   * promote-then-quantize (unrotated) exploits sparsity, needs a position mask
//!   * rotate-then-quantize    (rotated)  needs no mask, but spends the outlier
//!                                        energy across all 256 codes
//!
//! A two-pass scheme tries to have both: capture the sparse outliers in the
//! ORIGINAL basis where they are cheap, then quantize the (now outlier-free)
//! residual in the ROTATED basis where uniform int4 is near-optimal:
//!
//!     x ≈ s + r,   s = top-k int8 (channel basis),   r = x − s
//!     y = s·Wᵀ + Q4(r Rᵀ)·(W Rᵀ)ᵀ
//!
//! ⚠️ Measured END TO END through the real weight, never as reconstruction SNR of
//! the activation alone. `a4_quant`'s own test header explains why: the raw
//! Frobenius norm is dominated by the outliers, which quantize *well*, so a raw
//! metric rewards a scheme for preserving them while the bulk is crushed.
//!
//! ⚠️ The kernel cost is NOT symmetric between the schemes, and the numbers here
//! do not price it. `y = s·Wᵀ + Q4(rRᵀ)·(WRᵀ)ᵀ` needs the weight in TWO bases.
//! Storing both doubles weight traffic and is fatal. The only way it pays is if
//! `s·Wᵀ` is a gather of a few UNROTATED weight columns — which requires the
//! outlier channel positions to be static enough to fix offline. That is exactly
//! what the last section measures, and it is the go/no-go for the whole idea.
//!
//! Run:
//!   hipfire lock acquire "two-pass-probe"
//!   ./target/release/examples/two_pass_act_probe [model_dir]
//!   hipfire lock release

use hipfire_model::tokenizer::Tokenizer;
use hipfire_rdna::Gpu;
use hipfire_train::a4_quant::{simquant_bits, simquant_outlier, snr_db, GROUP};
use hipfire_train::loader::load_llama_fp32;
use hipfire_train::model::{model_forward, LlamaModel};
use hipfire_train::rotation::{apply_r1, rotate_rows, Rotation};
use rayon::prelude::*;
use std::collections::HashMap;
use std::path::Path;

const DEFAULT_DIR: &str = "/srv/huggingface/models--meta-llama--Llama-3.2-1B/snapshots/main";
const SEQ: usize = 64;
/// Promoted values per 256-group. `HIPFIRE_PROBE_NOUT` sweeps the operating
/// point — a scheme that loses at one k might win at another, and "we measured
/// one k" is not an answer to "is two-pass better".
fn n_out() -> usize {
    std::env::var("HIPFIRE_PROBE_NOUT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8)
}

/// `y = x·Wᵀ`, `x [rows,k]`, `w [out,k]`, both row-major.
fn matmul_t(x: &[f32], w: &[f32], rows: usize, k: usize, out: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; rows * out];
    y.par_chunks_mut(out).enumerate().for_each(|(r, yr)| {
        let xr = &x[r * k..r * k + k];
        for (o, dst) in yr.iter_mut().enumerate() {
            let wr = &w[o * k..o * k + k];
            let mut acc = 0.0f32;
            for (a, b) in xr.iter().zip(wr.iter()) {
                acc += a * b;
            }
            *dst = acc;
        }
    });
    y
}

/// Top-`k` magnitudes per 256-group captured at int8; everything else exactly 0.
/// Returns the dequantized sparse tensor, so `x − s` is the true residual the
/// second pass would see (it accounts for `s`'s own quantization error).
fn sparse_top_k_int8(x: &[f32], rows: usize, feat: usize, k: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * feat];
    for r in 0..rows {
        let row = &x[r * feat..r * feat + feat];
        let dst = &mut out[r * feat..r * feat + feat];
        let mut g = 0;
        while g < feat {
            let end = (g + GROUP).min(feat);
            let grp = &row[g..end];
            let kk = k.min(grp.len());
            let mut idx: Vec<usize> = (0..grp.len()).collect();
            idx.sort_unstable_by(|&a, &b| {
                grp[b]
                    .abs()
                    .partial_cmp(&grp[a].abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let amax = idx[..kk].iter().fold(0.0f32, |a, &i| a.max(grp[i].abs()));
            let scale = (amax / 127.0).max(1e-12);
            for &i in &idx[..kk] {
                dst[g + i] = (grp[i] / scale).round().clamp(-127.0, 127.0) * scale;
            }
            g = end;
        }
    }
    out
}

fn sub(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b).map(|(x, y)| x - y).collect()
}

/// What fraction of the dynamic top-`k` outlier slots would a STATIC per-group
/// mask catch? 1.0 means the outlier channels never move across tokens, so the
/// mask can be fixed offline from calibration and `s·Wᵀ` becomes a gather of a
/// few fixed weight columns. Low values mean the mask must ride the data, which
/// is what makes the two-pass scheme expensive.
fn static_mask_coverage(x: &[f32], rows: usize, feat: usize, k: usize) -> f32 {
    let (mut hit, mut total) = (0usize, 0usize);
    let mut g = 0;
    while g < feat {
        let end = (g + GROUP).min(feat);
        let width = end - g;
        let kk = k.min(width);
        let mut counts: HashMap<usize, usize> = HashMap::new();
        for r in 0..rows {
            let grp = &x[r * feat + g..r * feat + end];
            let mut idx: Vec<usize> = (0..width).collect();
            idx.sort_unstable_by(|&a, &b| {
                grp[b]
                    .abs()
                    .partial_cmp(&grp[a].abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            for &i in &idx[..kk] {
                *counts.entry(i).or_insert(0) += 1;
            }
        }
        // The static mask is the kk channels selected most often.
        let mut by_count: Vec<(usize, usize)> = counts.into_iter().collect();
        by_count.sort_unstable_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        hit += by_count.iter().take(kk).map(|&(_, c)| c).sum::<usize>();
        total += rows * kk;
        g = end;
    }
    hit as f32 / total.max(1) as f32
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| DEFAULT_DIR.to_string());
    let dir = Path::new(&dir);
    if !dir.exists() {
        return Err(format!("model dir not found: {}", dir.display()).into());
    }
    let mut gpu = Gpu::init().expect("Gpu::init failed");
    println!("arch: {}  model: {}", gpu.arch, dir.display());

    let (cfg, w) = load_llama_fp32(&mut gpu, dir)?;
    let h = cfg.hidden_size;
    if !h.is_power_of_two() {
        return Err(format!("hidden {h} is not a power of two (Hadamard needs one)").into());
    }
    let mut model = LlamaModel::from_f32_weights(&mut gpu, &cfg, w, SEQ, 2, 1.0)?;
    // Fold-only: makes the captured activations the ones the deployed graph sees.
    apply_r1(&mut gpu, &mut model, &Rotation::identity(h))?;
    let inter = model.dims.inter;
    // Real text, not synthetic ids: outlier structure is a property of the
    // activations real text produces, and a probe on uniform-random tokens would
    // not transfer -- the same trap this repo's QAT loop already hit.
    let corpus = std::env::var("HIPFIRE_PROBE_CORPUS")
        .unwrap_or_else(|_| "benchmarks/calib/calib-1m.txt".to_string());
    let tokens: Vec<u32> = {
        let tok = Tokenizer::from_tokenizer_json(&dir.join("tokenizer.json"))?
            .ok_or("model dir has no tokenizer.json")?;
        let raw = std::fs::read_to_string(&corpus)?;
        let mut end = 64 * 1024;
        while !raw.is_char_boundary(end) {
            end += 1;
        }
        let ids: Vec<u32> = tok
            .encode(&raw[..end])
            .into_iter()
            .filter(|&i| (i as usize) < cfg.vocab_size)
            .collect();
        assert!(
            ids.len() >= SEQ,
            "corpus {corpus} yielded {} tokens",
            ids.len()
        );
        ids[..SEQ].to_vec()
    };
    println!("tokens: real text from {corpus}");
    let pos: Vec<f32> = (0..SEQ).map(|t| t as f32).collect();
    let acts = model_forward(&mut gpu, &model, &tokens, &pos)?;

    let mut cap: Vec<(&str, Vec<f32>, Vec<f32>, usize, usize)> = Vec::new();
    for (i, (lw, _)) in model.layers.iter().enumerate() {
        cap.push((
            "xn1→q_proj",
            gpu.download_f32(&acts.layer_acts[i].xn1)?,
            gpu.download_f32(&lw.wq)?,
            h,
            model.dims.q_dim(),
        ));
        cap.push((
            "xn2→gate_proj",
            gpu.download_f32(&acts.layer_acts[i].xn2)?,
            gpu.download_f32(&lw.wgate)?,
            h,
            inter,
        ));
        // The two HIGH-CREST sites -- measured crest factor ~8 here, versus ~2 on
        // the residual reads. Omitting them would characterise the mechanism on
        // the two sites where it matters least.
        cap.push((
            "ctx→o_proj",
            gpu.download_f32(&acts.layer_acts[i].ctx)?,
            gpu.download_f32(&lw.wo)?,
            model.dims.q_dim(),
            h,
        ));
        cap.push((
            "act→down_proj",
            gpu.download_f32(&acts.layer_acts[i].act)?,
            gpu.download_f32(&lw.wdown)?,
            inter,
            h,
        ));
    }
    let n_out = n_out();
    println!("n_out = {n_out} promoted per 256-group");
    let mut rots: HashMap<usize, (Rotation, Rotation)> = HashMap::new();
    for (_, _, _, feat, _) in cap.iter() {
        rots.entry(*feat).or_insert_with(|| {
            // `hadamard` is a dense Sylvester matrix with random column signs;
            // `block_fwht` is the Oq4G256 codec's own per-256 signed FWHT. They
            // are DIFFERENT rotations and only the second is what deploys.
            (Rotation::hadamard(*feat, 1), Rotation::block_fwht(*feat))
        });
    }
    println!(
        "captured {} (act, weight) pairs over {} layers, SEQ={SEQ}, h={h}\n",
        cap.len(),
        model.layers.len()
    );

    // Each scheme returns the end-to-end output SNR in dB, higher is better.
    let mut sums: Vec<(String, f32)> = vec![
        ("a4  (uniform, no rotation)".to_string(), 0.0),
        ("a4  + Hadamard".to_string(), 0.0),
        ("a4  + codec block-FWHT".to_string(), 0.0),
        (format!("a4o{n_out} (promote, no rotation)"), 0.0),
        (format!("a4o{n_out} + Hadamard"), 0.0),
        (format!("a4o{n_out} + codec block-FWHT"), 0.0),
        (
            format!("2-pass: int8 top-{n_out} pre-rot, a4 residual post-rot"),
            0.0,
        ),
        ("a8  (uniform, no rotation) [ceiling]".to_string(), 0.0),
    ];
    for (_, x, w, k, out) in cap.iter() {
        let (k, out) = (*k, *out);
        let y = matmul_t(x, w, SEQ, k, out);
        let (had, bf) = &rots[&k];
        let xr = rotate_rows(x, had, SEQ);
        let wr = rotate_rows(w, had, out);
        let xb = rotate_rows(x, bf, SEQ);
        let wb = rotate_rows(w, bf, out);
        let snr = |xq: &[f32], wu: &[f32]| snr_db(&y, &matmul_t(xq, wu, SEQ, k, out));

        sums[0].1 += snr(&simquant_bits(x, SEQ, k, 4), w);
        sums[1].1 += snr(&simquant_bits(&xr, SEQ, k, 4), &wr);
        sums[2].1 += snr(&simquant_bits(&xb, SEQ, k, 4), &wb);
        sums[3].1 += snr(&simquant_outlier(x, SEQ, k, 4, n_out), w);
        sums[4].1 += snr(&simquant_outlier(&xr, SEQ, k, 4, n_out), &wr);
        sums[5].1 += snr(&simquant_outlier(&xb, SEQ, k, 4, n_out), &wb);

        // Two-pass: sparse int8 in the channel basis, int4 residual in the
        // rotated basis. y ≈ s·Wᵀ + Q4(r Rᵀ)·(W Rᵀ)ᵀ — two GEMMs, two bases.
        let s = sparse_top_k_int8(x, SEQ, k, n_out);
        let r_rot = rotate_rows(&sub(x, &s), had, SEQ);
        let y_lo = matmul_t(&s, w, SEQ, k, out);
        let y_hi = matmul_t(&simquant_bits(&r_rot, SEQ, k, 4), &wr, SEQ, k, out);
        let y2: Vec<f32> = y_lo.iter().zip(&y_hi).map(|(a, b)| a + b).collect();
        sums[6].1 += snr_db(&y, &y2);

        sums[7].1 += snr(&simquant_bits(x, SEQ, k, 8), w);
    }
    let n = cap.len() as f32;
    println!("  ── end-to-end output SNR through the real weight (dB, higher better) ──");
    for (name, s) in sums.iter() {
        println!("  {:52} {:7.2}", name, s / n);
    }

    println!("\n  ── can the outlier mask be fixed offline? ──");
    println!("  fraction of dynamic top-{n_out} slots a STATIC per-group mask would catch:");
    for label in ["xn1→q_proj", "xn2→gate_proj", "ctx→o_proj", "act→down_proj"] {
        let mut sum = 0.0f32;
        let mut cnt = 0.0f32;
        for (nm, x, _, k, _) in cap.iter() {
            if *nm == label {
                sum += static_mask_coverage(x, SEQ, *k, n_out);
                cnt += 1.0;
            }
        }
        println!("  {label:16} {:6.1}%", 100.0 * sum / cnt.max(1.0));
    }
    println!(
        "\n  (A static mask makes s·Wᵀ a gather of {n_out} fixed weight columns per\n   \
         group. If coverage is low the mask must ride the data, and the two-pass\n   \
         scheme needs the weight in two bases — which is fatal.)"
    );
    Ok(())
}
