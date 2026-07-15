// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! BF16-first diffusion admission.
//!
//! Generates a frozen prompt/seed batch with a candidate and its baseline, then
//! gates on metrics that survive a chaotic sampler rather than on exact pixel
//! identity. Flow-matching samplers are trajectory-fragile: a near-lossless
//! weight perturbation (e.g. oq8, ~0.7% rel-L2 on a handful of tensors) reaches
//! a *different-but-valid* mode — same content, repositioned — which blows up an
//! exact per-pixel comparison while the model is entirely healthy. So we gate on:
//!
//!   1. Coherence guard (hard): the candidate image must be non-degenerate
//!      (finite, not blank/constant). Catches true corruption.
//!   2. Early-latent fidelity (hard): the step-1 latent (captured from the
//!      progress callback) diverges little between baseline and candidate. This
//!      is the numerical quant-fidelity signal, taken *before* trajectory chaos
//!      compounds — both runs share the same seed and (bf16) conditioning, so the
//!      only difference at step 1 is the quantized weights.
//!   3. Structural similarity (hard, generous floor): final-image SSIM catches
//!      structured-noise garbage that passes the coherence guard, without failing
//!      on legitimate repositioning.
//!
//! Exact pixel MAE/max are still recorded as telemetry (non-gating). LPIPS is an
//! opt-in perceptual cross-check computed out-of-band over the saved PNGs by
//! `scripts/flux2_lpips.py`.

use base64::Engine;
use hipfire_diffusion::{
    DiffusionBatchRequest, DiffusionGenerationRuntimeOptions, DiffusionPipeline, DiffusionProgress,
    DiffusionPrompt, LatentBatch,
};
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::time::Instant;

use crate::*;

const CASES: [(&str, i64); 3] = [
    ("benchmarks/prompts/flux2_image_admission_object.txt", 7),
    ("benchmarks/prompts/flux2_image_admission_scene.txt", 23),
    ("benchmarks/prompts/flux2_image_admission_texture.txt", 101),
];

/// Denoise step whose latent is used for the numerical-fidelity gate. Step 1 is
/// after a single integration — perturbed by quantization but not yet amplified
/// by the remaining trajectory.
const EARLY_LATENT_STEP: usize = 1;
/// Default cap on the step-1 latent relative RMSE (‖cand-base‖ / ‖base‖). First-
/// cut; tune against known-good/known-bad pairs. Overridable per run.
const EARLY_LATENT_REL_RMSE_LIMIT: f64 = 0.35;
/// Generous SSIM floor on the final image: a garbage-catcher, not a similarity
/// requirement. Legitimately repositioned images sit well above this.
const SSIM_MIN: f64 = 0.15;

pub(crate) fn diffusion_battery_rows(config: &EvalConfig, ctx: &EvalContext) -> Vec<EvalResult> {
    match run_diffusion_battery(config, ctx) {
        Ok(rows) => rows,
        Err(error) => vec![row(
            BatteryId::Diffusion,
            None,
            "diffusion_baseline_compare",
            None,
            EvalStatus::Fail,
            Some(error),
            BTreeMap::from([("implemented".to_string(), json!(true))]),
            config,
            ctx,
            None,
            0,
        )],
    }
}

fn run_diffusion_battery(config: &EvalConfig, ctx: &EvalContext) -> Result<Vec<EvalResult>, String> {
    let baseline_path = config
        .baseline
        .as_deref()
        .ok_or_else(|| "diffusion battery requires --baseline <bf16.hfq>".to_string())?;
    let baseline = DiffusionPipeline::open_hfq(baseline_path)
        .map_err(|error| format!("open diffusion baseline {baseline_path:?}: {error}"))?;
    let candidate = DiffusionPipeline::open_hfq(&config.model)
        .map_err(|error| format!("open diffusion candidate {:?}: {error}", config.model))?;
    let sefi = baseline.metadata().pipeline.sefi;
    if candidate.metadata().pipeline.sefi != sefi {
        return Err("candidate and baseline disagree on the SeFi pipeline marker".to_string());
    }

    const OVERRIDES: [&str; 7] = [
        "HIPFIRE_DIFFUSION_EVAL_WIDTH",
        "HIPFIRE_DIFFUSION_EVAL_HEIGHT",
        "HIPFIRE_DIFFUSION_EVAL_STEPS",
        "HIPFIRE_DIFFUSION_EVAL_DEVICE",
        "HIPFIRE_DIFFUSION_EVAL_CASES",
        "HIPFIRE_DIFFUSION_EVAL_EARLY_RMSE",
        "HIPFIRE_DIFFUSION_EVAL_SSIM_MIN",
    ];
    if config.fail_on_admission && OVERRIDES.iter().any(|name| std::env::var_os(name).is_some()) {
        return Err(
            "diffusion diagnostic overrides are forbidden with --fail-on-admission".to_string(),
        );
    }
    let case_count = eval_u32("HIPFIRE_DIFFUSION_EVAL_CASES", CASES.len() as u32)? as usize;
    if case_count == 0 || case_count > CASES.len() {
        return Err(format!(
            "HIPFIRE_DIFFUSION_EVAL_CASES must be in 1..={}, got {case_count}",
            CASES.len()
        ));
    }
    let cases = &CASES[..case_count];

    let prompts = cases
        .iter()
        .map(|(path, seed)| {
            let resolved = resolve_repo_path(path)
                .ok_or_else(|| format!("diffusion admission prompt is missing: {path}"))?;
            let text = fs::read_to_string(&resolved)
                .map_err(|error| format!("read {}: {error}", resolved.display()))?;
            Ok(DiffusionPrompt {
                prompt: text.trim_end().to_string(),
                negative_prompt: String::new(),
                seed: *seed,
                subseed: None,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    let width = eval_u32("HIPFIRE_DIFFUSION_EVAL_WIDTH", 64)?;
    let height = eval_u32("HIPFIRE_DIFFUSION_EVAL_HEIGHT", 64)?;
    let steps = eval_u32("HIPFIRE_DIFFUSION_EVAL_STEPS", 4)?;
    let device_id = eval_i32("HIPFIRE_DIFFUSION_EVAL_DEVICE", 0)?;
    let early_rmse_limit = eval_f64("HIPFIRE_DIFFUSION_EVAL_EARLY_RMSE", EARLY_LATENT_REL_RMSE_LIMIT)?;
    let ssim_min = eval_f64("HIPFIRE_DIFFUSION_EVAL_SSIM_MIN", SSIM_MIN)?;
    let cfg_scale = if sefi { 1.0 } else { 4.0 };
    let request = DiffusionBatchRequest {
        prompts,
        conditioning: None,
        width,
        height,
        original_width: None,
        original_height: None,
        target_width: None,
        target_height: None,
        seed_resize_from_width: None,
        seed_resize_from_height: None,
        crop_x: 0,
        crop_y: 0,
        steps,
        cfg_scale,
        distilled_guidance_scale: None,
        scheduler: "Euler".to_string(),
        subseed_strength: 0.0,
        send_images: true,
        save_images: false,
    };
    let runtime = DiffusionGenerationRuntimeOptions::rocm_hybrid(device_id);
    let baseline_run = generate_images(&baseline, &request, runtime, "baseline")?;
    let candidate_run = generate_images(&candidate, &request, runtime, "candidate")?;

    // Persist the rendered PNGs so rejections are eyeball-able and LPIPS can be
    // computed out-of-band. Best-effort: a write failure is logged, not fatal.
    let image_dir = config.out_dir.join("artifacts").join("diffusion");
    let _ = fs::create_dir_all(&image_dir);

    let mut rows = Vec::with_capacity(cases.len());
    for (index, (path, seed)) in cases.iter().enumerate() {
        let reference = decode_rgb_png(&baseline_run.images[index])?;
        let actual = decode_rgb_png(&candidate_run.images[index])?;
        if reference.dimensions != actual.dimensions {
            return Err(format!(
                "diffusion image {index} dimensions {:?} != {:?}",
                actual.dimensions, reference.dimensions
            ));
        }
        let _ = save_png(&baseline_run.images[index], &image_dir.join(format!("case{index}_seed{seed}_baseline.png")));
        let _ = save_png(&candidate_run.images[index], &image_dir.join(format!("case{index}_seed{seed}_candidate.png")));

        // --- telemetry: exact pixel drift (non-gating) ---
        let mut abs_sum = 0u64;
        let mut max_error = 0u8;
        for (&expected, &observed) in reference.rgb.iter().zip(&actual.rgb) {
            let error = expected.abs_diff(observed);
            abs_sum += u64::from(error);
            max_error = max_error.max(error);
        }
        let mae_u8 = abs_sum as f64 / reference.rgb.len() as f64;

        // --- gate 1: coherence guard (hard) ---
        let coherence = image_coherence(&actual);

        // --- gate 2: early-latent fidelity (hard) ---
        let early = match (baseline_run.early.get(index), candidate_run.early.get(index)) {
            (Some(b), Some(c)) if !b.is_empty() && b.len() == c.len() => Some(rel_rmse(b, c)),
            _ => None,
        };

        // --- gate 3: structural similarity (hard, generous floor) ---
        let ssim = ssim_luma(&reference, &actual);

        let mut fail_reasons: Vec<String> = Vec::new();
        if !coherence.ok {
            fail_reasons.push(format!(
                "candidate image is degenerate (std={:.3}, finite={})",
                coherence.std, coherence.finite
            ));
        }
        match early {
            Some(rel) if rel > early_rmse_limit => fail_reasons.push(format!(
                "step-{EARLY_LATENT_STEP} latent rel-RMSE {rel:.4} exceeds {early_rmse_limit:.4}"
            )),
            None => fail_reasons.push(
                "early-latent unavailable (no preview_latents captured); cannot gate fidelity"
                    .to_string(),
            ),
            _ => {}
        }
        if ssim < ssim_min {
            fail_reasons.push(format!("final SSIM {ssim:.4} below floor {ssim_min:.4}"));
        }
        let passed = fail_reasons.is_empty();

        let mut metrics = BTreeMap::from([
            ("implemented".to_string(), json!(true)),
            ("early_latent_rel_rmse".to_string(), json!(early)),
            ("early_latent_rel_rmse_limit".to_string(), json!(early_rmse_limit)),
            ("early_latent_step".to_string(), json!(EARLY_LATENT_STEP)),
            ("ssim".to_string(), json!(ssim)),
            ("ssim_min".to_string(), json!(ssim_min)),
            ("candidate_image_std".to_string(), json!(coherence.std)),
            ("candidate_image_finite".to_string(), json!(coherence.finite)),
            // telemetry (non-gating): kept for continuity / dashboards.
            ("rgb_mae_u8".to_string(), json!(mae_u8)),
            ("rgb_max_error_u8".to_string(), json!(max_error)),
            ("width".to_string(), json!(width)),
            ("height".to_string(), json!(height)),
            ("steps".to_string(), json!(steps)),
            ("cfg_scale".to_string(), json!(cfg_scale)),
            ("seed".to_string(), json!(seed)),
            ("baseline_elapsed_ms".to_string(), json!(baseline_run.elapsed_ms)),
            ("candidate_elapsed_ms".to_string(), json!(candidate_run.elapsed_ms)),
        ]);
        metrics.insert("candidate_png".to_string(), json!(format!("artifacts/diffusion/case{index}_seed{seed}_candidate.png")));
        metrics.insert("baseline_png".to_string(), json!(format!("artifacts/diffusion/case{index}_seed{seed}_baseline.png")));

        rows.push(row(
            BatteryId::Diffusion,
            None,
            &format!("rgb_baseline_{index}"),
            Some(format!("seed-{seed}")),
            if passed { EvalStatus::Pass } else { EvalStatus::Fail },
            (!passed).then(|| format!("diffusion admission failed: {}", fail_reasons.join("; "))),
            metrics,
            config,
            ctx,
            prompt(path),
            candidate_run.elapsed_ms,
        ));
    }
    Ok(rows)
}

struct GeneratedRun {
    images: Vec<String>,
    /// Per-case step-`EARLY_LATENT_STEP` latent (flattened `channels*height*width`).
    /// Empty entry when no preview latent was captured for that case.
    early: Vec<Vec<f32>>,
    elapsed_ms: u128,
}

/// Slice a captured batch latent into per-item flat vectors (`channels*h*w` each).
fn split_latent_items(latent: &LatentBatch) -> Vec<Vec<f32>> {
    let per_item = latent.channels * latent.height * latent.width;
    if per_item == 0 || latent.batch == 0 {
        return Vec::new();
    }
    (0..latent.batch)
        .map(|b| latent.data[b * per_item..(b + 1) * per_item].to_vec())
        .collect()
}

fn generate_images(
    pipeline: &DiffusionPipeline,
    request: &DiffusionBatchRequest,
    runtime: DiffusionGenerationRuntimeOptions,
    label: &str,
) -> Result<GeneratedRun, String> {
    let started = Instant::now();
    let batched = pipeline.metadata().batch.max_batch as usize >= request.prompts.len();

    let (images, early) = if batched {
        let captured: RefCell<Option<LatentBatch>> = RefCell::new(None);
        let mut progress = |p: DiffusionProgress| {
            if p.completed_steps >= EARLY_LATENT_STEP && captured.borrow().is_none() {
                *captured.borrow_mut() = p.preview_latents.clone();
            }
            eprintln!(
                "[diffusion-eval] {label} batch step {}/{} timestep {}",
                p.completed_steps, p.total_steps, p.timestep
            );
            Ok(())
        };
        let images = pipeline
            .generate_batch_with_progress_and_runtime_options(request.clone(), runtime, &mut progress)
            .map_err(|error| format!("generate diffusion {label}: {error}"))?
            .images;
        let early = captured
            .into_inner()
            .map(|l| split_latent_items(&l))
            .unwrap_or_default();
        // Pad/truncate to one entry per prompt.
        let early = (0..request.prompts.len())
            .map(|i| early.get(i).cloned().unwrap_or_default())
            .collect();
        (images, early)
    } else {
        let mut images = Vec::with_capacity(request.prompts.len());
        let mut early = Vec::with_capacity(request.prompts.len());
        for (index, prompt) in request.prompts.iter().enumerate() {
            let mut single = request.clone();
            single.prompts = vec![prompt.clone()];
            let captured: RefCell<Option<LatentBatch>> = RefCell::new(None);
            let mut progress = |p: DiffusionProgress| {
                if p.completed_steps >= EARLY_LATENT_STEP && captured.borrow().is_none() {
                    *captured.borrow_mut() = p.preview_latents.clone();
                }
                eprintln!(
                    "[diffusion-eval] {label} case {}/{} step {}/{} timestep {}",
                    index + 1,
                    request.prompts.len(),
                    p.completed_steps,
                    p.total_steps,
                    p.timestep
                );
                Ok(())
            };
            let output = pipeline
                .generate_batch_with_progress_and_runtime_options(single, runtime, &mut progress)
                .map_err(|error| format!("generate diffusion {label}: {error}"))?;
            if output.images.len() != 1 {
                return Err(format!(
                    "diffusion {label} single-item request returned {} images",
                    output.images.len()
                ));
            }
            images.extend(output.images);
            early.push(
                captured
                    .into_inner()
                    .map(|l| split_latent_items(&l).into_iter().next().unwrap_or_default())
                    .unwrap_or_default(),
            );
        }
        (images, early)
    };

    if images.len() != request.prompts.len() {
        return Err(format!(
            "diffusion {label} returned {} images; expected {}",
            images.len(),
            request.prompts.len()
        ));
    }
    Ok(GeneratedRun {
        images,
        early,
        elapsed_ms: started.elapsed().as_millis(),
    })
}

struct DecodedRgb {
    dimensions: (u32, u32),
    rgb: Vec<u8>,
}

struct Coherence {
    ok: bool,
    std: f64,
    finite: bool,
}

/// Reject degenerate candidate images: non-finite (defensive), or near-constant
/// (blank/flat) output. A healthy render — even a badly repositioned one — has
/// real per-channel spread.
fn image_coherence(img: &DecodedRgb) -> Coherence {
    let n = img.rgb.len().max(1) as f64;
    let mean = img.rgb.iter().map(|&v| v as f64).sum::<f64>() / n;
    let var = img.rgb.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n;
    let std = var.sqrt();
    let finite = std.is_finite();
    // u8 pixels are always finite; the std floor (≈ <1 LSB spread) is the real
    // degenerate-image guard.
    Coherence { ok: finite && std > 1.0, std, finite }
}

fn rel_rmse(a: &[f32], b: &[f32]) -> f64 {
    let mut sq = 0.0f64;
    let mut ref_sq = 0.0f64;
    for (&x, &y) in a.iter().zip(b) {
        let d = (x - y) as f64;
        sq += d * d;
        ref_sq += (x as f64) * (x as f64);
    }
    let ref_rms = (ref_sq / a.len().max(1) as f64).sqrt();
    let rms = (sq / a.len().max(1) as f64).sqrt();
    if ref_rms > 0.0 {
        rms / ref_rms
    } else {
        0.0
    }
}

fn rgb_to_luma(img: &DecodedRgb) -> Vec<f64> {
    img.rgb
        .chunks_exact(3)
        .map(|p| 0.299 * p[0] as f64 + 0.587 * p[1] as f64 + 0.114 * p[2] as f64)
        .collect()
}

/// Windowed SSIM over luma (8x8 window, stride 4) on the [0,255] scale. Structure-
/// aware similarity: near 1 for identical images, near 0 for structured noise.
fn ssim_luma(a: &DecodedRgb, b: &DecodedRgb) -> f64 {
    if a.dimensions != b.dimensions {
        return 0.0;
    }
    let (w, h) = (a.dimensions.0 as usize, a.dimensions.1 as usize);
    let la = rgb_to_luma(a);
    let lb = rgb_to_luma(b);
    if w < 8 || h < 8 {
        return global_ssim(&la, &lb);
    }
    const WIN: usize = 8;
    const STRIDE: usize = 4;
    const C1: f64 = 6.5025; // (0.01*255)^2
    const C2: f64 = 58.5225; // (0.03*255)^2
    let mut acc = 0.0;
    let mut count = 0.0;
    let mut y = 0;
    while y + WIN <= h {
        let mut x = 0;
        while x + WIN <= w {
            let (mut ma, mut mb) = (0.0, 0.0);
            for j in 0..WIN {
                for i in 0..WIN {
                    let idx = (y + j) * w + (x + i);
                    ma += la[idx];
                    mb += lb[idx];
                }
            }
            let np = (WIN * WIN) as f64;
            ma /= np;
            mb /= np;
            let (mut va, mut vb, mut cov) = (0.0, 0.0, 0.0);
            for j in 0..WIN {
                for i in 0..WIN {
                    let idx = (y + j) * w + (x + i);
                    let da = la[idx] - ma;
                    let db = lb[idx] - mb;
                    va += da * da;
                    vb += db * db;
                    cov += da * db;
                }
            }
            va /= np - 1.0;
            vb /= np - 1.0;
            cov /= np - 1.0;
            let s = ((2.0 * ma * mb + C1) * (2.0 * cov + C2))
                / ((ma * ma + mb * mb + C1) * (va + vb + C2));
            acc += s;
            count += 1.0;
            x += STRIDE;
        }
        y += STRIDE;
    }
    if count > 0.0 {
        acc / count
    } else {
        global_ssim(&la, &lb)
    }
}

fn global_ssim(la: &[f64], lb: &[f64]) -> f64 {
    const C1: f64 = 6.5025;
    const C2: f64 = 58.5225;
    let n = la.len().max(1) as f64;
    let ma = la.iter().sum::<f64>() / n;
    let mb = lb.iter().sum::<f64>() / n;
    let (mut va, mut vb, mut cov) = (0.0, 0.0, 0.0);
    for (&a, &b) in la.iter().zip(lb) {
        va += (a - ma).powi(2);
        vb += (b - mb).powi(2);
        cov += (a - ma) * (b - mb);
    }
    va /= n;
    vb /= n;
    cov /= n;
    ((2.0 * ma * mb + C1) * (2.0 * cov + C2)) / ((ma * ma + mb * mb + C1) * (va + vb + C2))
}

fn decode_rgb_png(encoded: &str) -> Result<DecodedRgb, String> {
    let bytes = decode_png_bytes(encoded)?;
    let image = image::load_from_memory_with_format(&bytes, image::ImageFormat::Png)
        .map_err(|error| format!("decode diffusion PNG: {error}"))?
        .to_rgb8();
    Ok(DecodedRgb {
        dimensions: image.dimensions(),
        rgb: image.into_raw(),
    })
}

fn decode_png_bytes(encoded: &str) -> Result<Vec<u8>, String> {
    let payload = encoded
        .split_once(',')
        .filter(|(prefix, _)| prefix.starts_with("data:image/"))
        .map(|(_, payload)| payload)
        .unwrap_or(encoded);
    base64::engine::general_purpose::STANDARD
        .decode(payload)
        .map_err(|error| format!("decode diffusion PNG base64: {error}"))
}

fn save_png(encoded: &str, path: &std::path::Path) -> Result<(), String> {
    let bytes = decode_png_bytes(encoded)?;
    fs::write(path, bytes).map_err(|error| format!("write {}: {error}", path.display()))
}

fn eval_u32(name: &str, default: u32) -> Result<u32, String> {
    std::env::var(name)
        .map(|raw| raw.parse().map_err(|error| format!("{name}={raw:?}: {error}")))
        .unwrap_or(Ok(default))
}

fn eval_i32(name: &str, default: i32) -> Result<i32, String> {
    std::env::var(name)
        .map(|raw| raw.parse().map_err(|error| format!("{name}={raw:?}: {error}")))
        .unwrap_or(Ok(default))
}

fn eval_f64(name: &str, default: f64) -> Result<f64, String> {
    std::env::var(name)
        .map(|raw| raw.parse().map_err(|error| format!("{name}={raw:?}: {error}")))
        .unwrap_or(Ok(default))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn img(w: u32, h: u32, rgb: Vec<u8>) -> DecodedRgb {
        DecodedRgb { dimensions: (w, h), rgb }
    }

    fn checkerboard(w: usize, h: usize, invert: bool) -> DecodedRgb {
        let mut rgb = Vec::with_capacity(w * h * 3);
        for y in 0..h {
            for x in 0..w {
                let on = ((x / 4 + y / 4) % 2 == 0) ^ invert;
                let v = if on { 220 } else { 30 };
                rgb.extend_from_slice(&[v, v, v]);
            }
        }
        img(w as u32, h as u32, rgb)
    }

    #[test]
    fn ssim_identical_is_one() {
        let a = checkerboard(32, 32, false);
        assert!((ssim_luma(&a, &a) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn ssim_inverted_is_low() {
        let a = checkerboard(32, 32, false);
        let b = checkerboard(32, 32, true);
        // Anti-correlated structure => SSIM well below the garbage floor.
        assert!(ssim_luma(&a, &b) < SSIM_MIN, "inverted ssim = {}", ssim_luma(&a, &b));
    }

    #[test]
    fn coherence_rejects_flat_accepts_textured() {
        let flat = img(16, 16, vec![128; 16 * 16 * 3]);
        assert!(!image_coherence(&flat).ok);
        assert!(image_coherence(&checkerboard(16, 16, false)).ok);
    }

    #[test]
    fn rel_rmse_zero_when_identical_and_scales_with_error() {
        let a = vec![1.0f32, -2.0, 3.0, -4.0];
        assert!(rel_rmse(&a, &a) < 1e-9);
        let b: Vec<f32> = a.iter().map(|v| v + 0.1).collect();
        let r = rel_rmse(&a, &b);
        assert!(r > 0.0 && r < 0.1, "rel_rmse = {r}");
    }
}
