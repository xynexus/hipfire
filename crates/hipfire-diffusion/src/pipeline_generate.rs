// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `DiffusionPipeline::generate` — extracted from the pipeline
//! god-impl in lib.rs into its own `impl` block (3.8 Part 2). Uses `super::*`
//! so the pipeline's helpers + types resolve unchanged; the struct's fields are
//! pub(crate) so this block can read them.

use super::*;

/// Debug: print summary stats for a `[batch, seq, hidden]` conditioning tensor
/// (or any tensor) — mean/std/min/max, finite fraction, and per-token L2 norm
/// spread. Used by the HIPFIRE_DUMP_COND hook to spot dead/constant/NaN text
/// conditioning that would make the denoiser emit structured noise.
fn dump_conditioning_stats(label: &str, t: &CpuTensor) {
    let n = t.data.len().max(1);
    let finite = t.data.iter().filter(|v| v.is_finite()).count();
    let mean = t.data.iter().copied().map(|v| v as f64).sum::<f64>() / n as f64;
    let var = t
        .data
        .iter()
        .map(|&v| (v as f64 - mean).powi(2))
        .sum::<f64>()
        / n as f64;
    let min = t.data.iter().copied().fold(f32::INFINITY, f32::min);
    let max = t.data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    // Per-token L2 norm over the last (hidden) dim.
    let hidden = *t.shape.last().unwrap_or(&n).max(&1);
    let mut norms = Vec::new();
    let mut zero_rows = 0usize;
    for row in t.data.chunks(hidden) {
        let l2 = (row.iter().map(|&v| (v as f64).powi(2)).sum::<f64>()).sqrt();
        if l2 == 0.0 {
            zero_rows += 1;
        }
        norms.push(l2);
    }
    let (nmin, nmax, nmean) = if norms.is_empty() {
        (0.0, 0.0, 0.0)
    } else {
        let s: f64 = norms.iter().sum();
        (
            norms.iter().copied().fold(f64::INFINITY, f64::min),
            norms.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            s / norms.len() as f64,
        )
    };
    eprintln!(
        "[cond] {label}: shape={:?} mean={mean:+.5} std={:.5} min={min:+.3} max={max:+.3} \
         finite={finite}/{n} | per-token L2 mean={nmean:.3} min={nmin:.3} max={nmax:.3} \
         zero_rows={zero_rows}/{}",
        t.shape,
        var.sqrt(),
        norms.len()
    );
}

impl DiffusionPipeline {
    pub fn generate_batch(
        &self,
        request: DiffusionBatchRequest,
    ) -> DiffusionResult<DiffusionBatchOutput> {
        self.generate_batch_inner(request, DiffusionGenerationRuntimeOptions::default(), None)
            .map(|(output, _latent)| output)
    }

    pub fn generate_batch_with_progress(
        &self,
        request: DiffusionBatchRequest,
        progress: &mut dyn FnMut(DiffusionProgress) -> DiffusionResult<()>,
    ) -> DiffusionResult<DiffusionBatchOutput> {
        self.generate_batch_inner(
            request,
            DiffusionGenerationRuntimeOptions::default(),
            Some(progress),
        )
        .map(|(output, _latent)| output)
    }

    pub fn generate_batch_with_runtime_options(
        &self,
        request: DiffusionBatchRequest,
        runtime_options: DiffusionGenerationRuntimeOptions,
    ) -> DiffusionResult<DiffusionBatchOutput> {
        self.generate_batch_inner(request, runtime_options, None)
            .map(|(output, _latent)| output)
    }

    pub fn generate_batch_with_progress_and_runtime_options(
        &self,
        request: DiffusionBatchRequest,
        runtime_options: DiffusionGenerationRuntimeOptions,
        progress: &mut dyn FnMut(DiffusionProgress) -> DiffusionResult<()>,
    ) -> DiffusionResult<DiffusionBatchOutput> {
        self.generate_batch_inner(request, runtime_options, Some(progress))
            .map(|(output, _latent)| output)
    }

    /// As [`Self::generate_batch_with_progress_and_runtime_options`] but also
    /// returns the complete model latent (pre-VAE-slice) for staged-sampling
    /// (MrFlow / draft) Stage-2, which refines in latent space so model-specific
    /// channels (e.g. SeFi's semantic stream) are carried, not reconstructed.
    pub fn generate_batch_capturing_latent(
        &self,
        request: DiffusionBatchRequest,
        runtime_options: DiffusionGenerationRuntimeOptions,
        progress: Option<&mut dyn FnMut(DiffusionProgress) -> DiffusionResult<()>>,
    ) -> DiffusionResult<(DiffusionBatchOutput, LatentBatch)> {
        self.generate_batch_inner(request, runtime_options, progress)
    }

    /// Generic latent-space Stage-2 refine — the foundation of draft mode.
    ///
    /// Takes the FULL Stage-1 latent (every channel, e.g. SeFi's 144 = 16
    /// semantic + 128 texture), upscales it in latent space to the target
    /// request's resolution, injects flow-match refine noise at `first_sigma`,
    /// and re-denoises with the model's OWN denoiser — SeFi's dual-stream split
    /// when the model is SeFi, the standard denoiser otherwise. It never
    /// re-encodes from pixels, so model-specific latent channels (SeFi
    /// semantics) are *carried* through the refine instead of being lost and
    /// reconstructed. The (texture) latent is then decoded to RGB.
    ///
    /// This replaces the old pixel-re-encode-via-img2img Stage-2, which could
    /// only handle fully pixel-derivable latents and hung on SeFi.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_draft_refine(
        &self,
        request: DiffusionBatchRequest,
        stage1_full_latents: LatentBatch,
        first_sigma: f32,
        refine_steps: u32,
        shifted: bool,
        runtime_options: DiffusionGenerationRuntimeOptions,
        progress: Option<&mut dyn FnMut(DiffusionProgress) -> DiffusionResult<()>>,
    ) -> DiffusionResult<DiffusionBatchOutput> {
        validate_batch_request(&self.metadata, &request)?;
        let runtime = self.native_runtime()?;
        let mut runtime_context = DiffusionGenerationRuntimeContext::new(runtime_options);
        let plan = self.prepare_run_plan_with_runtime_context(&request, &mut runtime_context)?;
        let positive_embeddings = plan
            .conditioning
            .prompt_cross_attention_embeddings
            .as_ref()
            .or(plan.conditioning.prompt_embeddings.as_ref())
            .ok_or_else(|| {
                DiffusionError::BackendUnavailable(
                    "diffusion HFQ does not contain usable native text conditioning".to_string(),
                )
            })?;
        let negative_embeddings = plan
            .conditioning
            .negative_cross_attention_embeddings
            .as_ref()
            .or(plan.conditioning.negative_embeddings.as_ref())
            .ok_or_else(|| {
                DiffusionError::BackendUnavailable(
                    "diffusion HFQ does not contain usable native text conditioning".to_string(),
                )
            })?;
        let sdxl_time_ids = sdxl_time_ids_for_request(&request)?;
        let positive_sdxl_conditioning =
            build_sdxl_denoise_conditioning(&plan.conditioning, &sdxl_time_ids, true)?;
        let negative_sdxl_conditioning =
            build_sdxl_denoise_conditioning(&plan.conditioning, &sdxl_time_ids, false)?;

        let is_sefi = plan.sefi_dual_schedule.is_some();
        let semantic_channels = self.metadata.pipeline.semantic_channels.unwrap_or(0) as usize;

        // Upscale the full Stage-1 latent to the target latent grid in latent
        // space — channel-agnostic, so SeFi's semantic+texture stack is carried
        // untouched. Bilinear (not nearest): nearest replicates each source cell
        // into a k×k block, which the VAE decoder amplifies into a weave that a
        // light refine cannot erase. Batch/channels must already match the arch.
        let mut latents = resize_latent_batch_bilinear(
            &stage1_full_latents,
            plan.latent_shape.height,
            plan.latent_shape.width,
        )?;
        if latents.batch != plan.latent_shape.batch
            || latents.channels != plan.latent_shape.channels
        {
            return Err(DiffusionError::InvalidRequest(format!(
                "Stage-1 latent [{}x{}x{}x{}] is incompatible with the target latent shape \
                 [{}x{}x{}x{}] (batch/channels must match for a latent-space refine)",
                latents.batch,
                latents.channels,
                latents.height,
                latents.width,
                plan.latent_shape.batch,
                plan.latent_shape.channels,
                plan.latent_shape.height,
                plan.latent_shape.width
            )));
        }

        // Fresh refine noise at the target grid, seeds salted off the request
        // seeds so the refine injection is decorrelated from Stage-1 noise.
        let request_seeds = request
            .prompts
            .iter()
            .map(|prompt| prompt.seed)
            .collect::<Vec<_>>();
        let refine_seeds = vae_encode_seeds(&request_seeds, DRAFT_REFINE_NOISE_SEED_SALT);
        let noise = LatentBatch::seeded_normal(
            plan.latent_shape.batch,
            plan.latent_shape.channels,
            plan.latent_shape.height,
            plan.latent_shape.width,
            &refine_seeds,
        );

        let mut generation_runtime_kind = runtime.kind;
        // Diagnostic: HIPFIRE_DRAFT_DECODE_UPSCALED decodes the upscaled Stage-1
        // latent WITHOUT the refine, isolating "is the upscale-decode clean?"
        // from "does the refine step corrupt?".
        if let Ok(mode) = std::env::var("HIPFIRE_DRAFT_DECODE_UPSCALED") {
            if !mode.is_empty() {
                // "raw" decodes the captured Stage-1 latent at its NATIVE
                // resolution (pre-resize); anything else decodes the upscaled
                // latent. Splitting these isolates a bad captured latent from a
                // bad resize.
                let source = if mode == "raw" {
                    &stage1_full_latents
                } else {
                    &latents
                };
                let decode_latents = if is_sefi {
                    slice_latent_channels(source, semantic_channels)?
                } else {
                    source.clone()
                };
            let images = if request.send_images {
                let (rgb, image_runtime_kind) = decode_to_rgb8_with_runtime_context(
                    runtime.decoder.as_ref(),
                    &decode_latents,
                    &mut runtime_context,
                )?;
                generation_runtime_kind =
                    merge_runtime_kind(generation_runtime_kind, image_runtime_kind);
                encode_rgb_batch_png_base64(&rgb)?
            } else {
                Vec::new()
            };
            let mut info = diffusion_generation_info(
                self.summary(),
                generation_runtime_kind,
                &request,
                &plan.latent_shape,
            );
            if let Value::Object(map) = &mut info {
                map.insert(
                    "mode".to_string(),
                    Value::String("draft-upscaled-nodenoise".to_string()),
                );
            }
            return Ok(DiffusionBatchOutput { images, info });
            }
        }
        let denoise_output = if is_sefi {
            // Dual-stream refine: the semantic and texture streams resume from
            // `first_sigma` with the metadata `delta_t` offset preserved.
            let delta_t = self.metadata.pipeline.delta_t.ok_or_else(|| {
                DiffusionError::InvalidRequest(
                    "SeFi pipeline metadata is missing delta_t".to_string(),
                )
            })?;
            let schedule = DiffusionSchedule::sefi_dual_refine(
                first_sigma,
                refine_steps as usize,
                delta_t,
                1.0,
            )?;
            schedule.add_refine_noise(&mut latents, &noise.data, semantic_channels)?;
            runtime.noise.denoise_sefi_latents_with_runtime_context(
                latents,
                &schedule,
                semantic_channels,
                request.cfg_scale,
                positive_embeddings,
                negative_embeddings,
                plan.conditioning.prompt_attention_mask.as_ref(),
                plan.conditioning.negative_attention_mask.as_ref(),
                &mut runtime_context,
                progress,
            )?
        } else {
            let schedule =
                plan.schedule
                    .refine_direct_sigma(first_sigma, refine_steps, shifted)?;
            schedule.add_flow_match_refine_noise(&mut latents, &noise.data)?;
            runtime.noise.denoise_latents_with_runtime_context(
                latents,
                &schedule,
                request.cfg_scale,
                positive_embeddings,
                negative_embeddings,
                plan.conditioning.prompt_attention_mask.as_ref(),
                plan.conditioning.negative_attention_mask.as_ref(),
                positive_sdxl_conditioning.as_ref(),
                negative_sdxl_conditioning.as_ref(),
                None,
                None,
                &mut runtime_context,
                progress,
            )?
        };
        generation_runtime_kind =
            merge_runtime_kind(generation_runtime_kind, denoise_output.runtime_kind);
        let full_latents = denoise_output.latents;
        let decode_latents = if is_sefi {
            slice_latent_channels(&full_latents, semantic_channels)?
        } else {
            full_latents
        };
        let images = if request.send_images {
            let (rgb, image_runtime_kind) = decode_to_rgb8_with_runtime_context(
                runtime.decoder.as_ref(),
                &decode_latents,
                &mut runtime_context,
            )?;
            generation_runtime_kind =
                merge_runtime_kind(generation_runtime_kind, image_runtime_kind);
            encode_rgb_batch_png_base64(&rgb)?
        } else {
            Vec::new()
        };
        let mut info = diffusion_generation_info(
            self.summary(),
            generation_runtime_kind,
            &request,
            &plan.latent_shape,
        );
        if let Value::Object(map) = &mut info {
            map.insert("mode".to_string(), Value::String("draft-refine".to_string()));
            map.insert("refine_first_sigma".to_string(), json!(first_sigma));
            map.insert("refine_steps".to_string(), json!(refine_steps));
            map.insert("sefi_dual_refine".to_string(), json!(is_sefi));
        }
        Ok(DiffusionBatchOutput { images, info })
    }

    fn generate_batch_inner(
        &self,
        request: DiffusionBatchRequest,
        runtime_options: DiffusionGenerationRuntimeOptions,
        progress: Option<&mut dyn FnMut(DiffusionProgress) -> DiffusionResult<()>>,
    ) -> DiffusionResult<(DiffusionBatchOutput, LatentBatch)> {
        validate_batch_request(&self.metadata, &request)?;
        let runtime = self.native_runtime()?;
        let mut runtime_context = DiffusionGenerationRuntimeContext::new(runtime_options);
        let plan = self.prepare_run_plan_with_runtime_context(&request, &mut runtime_context)?;
        let positive_embeddings = plan
            .conditioning
            .prompt_cross_attention_embeddings
            .as_ref()
            .or(plan.conditioning.prompt_embeddings.as_ref())
            .ok_or_else(|| {
                DiffusionError::BackendUnavailable(
                    "diffusion HFQ does not contain usable native text conditioning".to_string(),
                )
            })?;
        let negative_embeddings = plan
            .conditioning
            .negative_cross_attention_embeddings
            .as_ref()
            .or(plan.conditioning.negative_embeddings.as_ref())
            .ok_or_else(|| {
                DiffusionError::BackendUnavailable(
                    "diffusion HFQ does not contain usable native text conditioning".to_string(),
                )
            })?;
        // Debug hook: HIPFIRE_DUMP_COND prints stats for the text conditioning
        // tensors fed to the denoiser. A dead/constant/NaN conditioning tensor
        // makes the denoiser produce structured noise. Also reports whether the
        // positive and negative streams are (nearly) identical, which would mean
        // CFG has nothing to steer with.
        if std::env::var("HIPFIRE_DUMP_COND").is_ok_and(|v| !v.is_empty()) {
            dump_conditioning_stats("positive", positive_embeddings);
            dump_conditioning_stats("negative", negative_embeddings);
            if positive_embeddings.shape == negative_embeddings.shape {
                let n = positive_embeddings.data.len().max(1);
                let mad: f64 = positive_embeddings
                    .data
                    .iter()
                    .zip(&negative_embeddings.data)
                    .map(|(a, b)| (a - b).abs() as f64)
                    .sum::<f64>()
                    / n as f64;
                eprintln!(
                    "[cond] pos-vs-neg mean|Δ|={mad:.6} (near 0 => CFG has nothing to steer)"
                );
            }
        }
        let _primary_positive_embeddings = plan
            .conditioning
            .prompt_embeddings
            .as_ref()
            .ok_or_else(|| {
                DiffusionError::BackendUnavailable(
                    "diffusion HFQ does not contain a usable native CLIP text encoder".to_string(),
                )
            })?;
        let _primary_negative_embeddings = plan
            .conditioning
            .negative_embeddings
            .as_ref()
            .ok_or_else(|| {
                DiffusionError::BackendUnavailable(
                    "diffusion HFQ does not contain a usable native CLIP text encoder".to_string(),
                )
            })?;
        let sdxl_time_ids = sdxl_time_ids_for_request(&request)?;
        let positive_sdxl_conditioning =
            build_sdxl_denoise_conditioning(&plan.conditioning, &sdxl_time_ids, true)?;
        let negative_sdxl_conditioning =
            build_sdxl_denoise_conditioning(&plan.conditioning, &sdxl_time_ids, false)?;
        if is_sdxl_pipeline_class(&self.config.pipeline_class)
            && (positive_sdxl_conditioning.is_none() || negative_sdxl_conditioning.is_none())
        {
            return Err(DiffusionError::BackendUnavailable(
                "SDXL generation requires dual text encoders, pooled text embeddings, and time IDs"
                    .to_string(),
            ));
        }
        let is_sefi = plan.sefi_dual_schedule.is_some();
        let semantic_channels = self.metadata.pipeline.semantic_channels.unwrap_or(0) as usize;
        let denoise_output = if let Some(schedule) = plan.sefi_dual_schedule.as_ref() {
            runtime.noise.denoise_sefi_latents_with_runtime_context(
                plan.latents,
                schedule,
                semantic_channels,
                request.cfg_scale,
                positive_embeddings,
                negative_embeddings,
                plan.conditioning.prompt_attention_mask.as_ref(),
                plan.conditioning.negative_attention_mask.as_ref(),
                &mut runtime_context,
                progress,
            )?
        } else {
            runtime.noise.denoise_latents_with_runtime_context(
                plan.latents,
                &plan.schedule,
                request.cfg_scale,
                positive_embeddings,
                negative_embeddings,
                plan.conditioning.prompt_attention_mask.as_ref(),
                plan.conditioning.negative_attention_mask.as_ref(),
                positive_sdxl_conditioning.as_ref(),
                negative_sdxl_conditioning.as_ref(),
                None,
                None,
                &mut runtime_context,
                progress,
            )?
        };
        // `full_latents` is the complete model latent (e.g. SeFi's 144 = 16
        // semantic + 128 texture); `latents` is what the VAE decodes (texture-only
        // for SeFi). The generic MrFlow/draft Stage-2 carries `full_latents`
        // forward so model-specific channels (SeFi semantics) survive the refine.
        let full_latents = denoise_output.latents;
        let latents = if is_sefi {
            slice_latent_channels(&full_latents, semantic_channels)?
        } else {
            full_latents.clone()
        };
        let mut generation_runtime_kind =
            merge_runtime_kind(runtime.kind, denoise_output.runtime_kind);
        let images = if request.send_images {
            let (rgb, image_runtime_kind) = decode_to_rgb8_with_runtime_context(
                runtime.decoder.as_ref(),
                &latents,
                &mut runtime_context,
            )?;
            generation_runtime_kind =
                merge_runtime_kind(generation_runtime_kind, image_runtime_kind);
            encode_rgb_batch_png_base64(&rgb)?
        } else {
            Vec::new()
        };
        let output = DiffusionBatchOutput {
            images,
            info: diffusion_generation_info(
                self.summary(),
                generation_runtime_kind,
                &request,
                &plan.latent_shape,
            ),
        };
        Ok((output, full_latents))
    }

    pub fn generate_img2img_batch(
        &self,
        request: DiffusionImg2ImgRequest,
    ) -> DiffusionResult<DiffusionBatchOutput> {
        self.generate_img2img_batch_inner(
            request,
            DiffusionGenerationRuntimeOptions::default(),
            None,
        )
    }

    pub fn generate_img2img_batch_with_progress(
        &self,
        request: DiffusionImg2ImgRequest,
        progress: &mut dyn FnMut(DiffusionProgress) -> DiffusionResult<()>,
    ) -> DiffusionResult<DiffusionBatchOutput> {
        self.generate_img2img_batch_inner(
            request,
            DiffusionGenerationRuntimeOptions::default(),
            Some(progress),
        )
    }

    pub fn generate_img2img_batch_with_runtime_options(
        &self,
        request: DiffusionImg2ImgRequest,
        runtime_options: DiffusionGenerationRuntimeOptions,
    ) -> DiffusionResult<DiffusionBatchOutput> {
        self.generate_img2img_batch_inner(request, runtime_options, None)
    }

    pub fn generate_img2img_batch_with_progress_and_runtime_options(
        &self,
        request: DiffusionImg2ImgRequest,
        runtime_options: DiffusionGenerationRuntimeOptions,
        progress: &mut dyn FnMut(DiffusionProgress) -> DiffusionResult<()>,
    ) -> DiffusionResult<DiffusionBatchOutput> {
        self.generate_img2img_batch_inner(request, runtime_options, Some(progress))
    }

    fn generate_img2img_batch_inner(
        &self,
        request: DiffusionImg2ImgRequest,
        runtime_options: DiffusionGenerationRuntimeOptions,
        progress: Option<&mut dyn FnMut(DiffusionProgress) -> DiffusionResult<()>>,
    ) -> DiffusionResult<DiffusionBatchOutput> {
        validate_img2img_request(&self.metadata, &request)?;
        let runtime = self.native_runtime()?;
        let mut runtime_context = DiffusionGenerationRuntimeContext::new(runtime_options);
        let plan =
            self.prepare_run_plan_with_runtime_context(&request.batch, &mut runtime_context)?;
        let positive_embeddings = plan
            .conditioning
            .prompt_cross_attention_embeddings
            .as_ref()
            .or(plan.conditioning.prompt_embeddings.as_ref())
            .ok_or_else(|| {
                DiffusionError::BackendUnavailable(
                    "diffusion HFQ does not contain usable native text conditioning".to_string(),
                )
            })?;
        let negative_embeddings = plan
            .conditioning
            .negative_cross_attention_embeddings
            .as_ref()
            .or(plan.conditioning.negative_embeddings.as_ref())
            .ok_or_else(|| {
                DiffusionError::BackendUnavailable(
                    "diffusion HFQ does not contain usable native text conditioning".to_string(),
                )
            })?;
        // Debug hook: HIPFIRE_DUMP_COND prints stats for the text conditioning
        // tensors fed to the denoiser. A dead/constant/NaN conditioning tensor
        // makes the denoiser produce structured noise. Also reports whether the
        // positive and negative streams are (nearly) identical, which would mean
        // CFG has nothing to steer with.
        if std::env::var("HIPFIRE_DUMP_COND").is_ok_and(|v| !v.is_empty()) {
            dump_conditioning_stats("positive", positive_embeddings);
            dump_conditioning_stats("negative", negative_embeddings);
            if positive_embeddings.shape == negative_embeddings.shape {
                let n = positive_embeddings.data.len().max(1);
                let mad: f64 = positive_embeddings
                    .data
                    .iter()
                    .zip(&negative_embeddings.data)
                    .map(|(a, b)| (a - b).abs() as f64)
                    .sum::<f64>()
                    / n as f64;
                eprintln!(
                    "[cond] pos-vs-neg mean|Δ|={mad:.6} (near 0 => CFG has nothing to steer)"
                );
            }
        }
        let _primary_positive_embeddings = plan
            .conditioning
            .prompt_embeddings
            .as_ref()
            .ok_or_else(|| {
                DiffusionError::BackendUnavailable(
                    "diffusion HFQ does not contain a usable native CLIP text encoder".to_string(),
                )
            })?;
        let _primary_negative_embeddings = plan
            .conditioning
            .negative_embeddings
            .as_ref()
            .ok_or_else(|| {
                DiffusionError::BackendUnavailable(
                    "diffusion HFQ does not contain a usable native CLIP text encoder".to_string(),
                )
            })?;
        let sdxl_time_ids = sdxl_time_ids_for_request(&request.batch)?;
        let positive_sdxl_conditioning =
            build_sdxl_denoise_conditioning(&plan.conditioning, &sdxl_time_ids, true)?;
        let negative_sdxl_conditioning =
            build_sdxl_denoise_conditioning(&plan.conditioning, &sdxl_time_ids, false)?;
        if is_sdxl_pipeline_class(&self.config.pipeline_class)
            && (positive_sdxl_conditioning.is_none() || negative_sdxl_conditioning.is_none())
        {
            return Err(DiffusionError::BackendUnavailable(
                "SDXL generation requires dual text encoders, pooled text embeddings, and time IDs"
                    .to_string(),
            ));
        }
        let encoder = runtime.encoder.as_ref().ok_or_else(|| {
            DiffusionError::BackendUnavailable(
                "diffusion HFQ does not contain a usable native VAE encoder".to_string(),
            )
        })?;
        let mut generation_runtime_kind = runtime.kind;
        let expanded_init_image =
            expand_rgb_batch_for_prompts(&request.init_image, request.batch.prompts.len())?;
        let init_image = match request.resize_mode {
            DiffusionImg2ImgResizeMode::Image => resize_rgb_batch_nearest(
                &expanded_init_image,
                request.batch.width,
                request.batch.height,
            )?,
            DiffusionImg2ImgResizeMode::Latent => expanded_init_image,
        };
        let request_seeds = request
            .batch
            .prompts
            .iter()
            .map(|prompt| prompt.seed)
            .collect::<Vec<_>>();
        let init_encode_seeds = vae_encode_seeds(&request_seeds, VAE_INIT_ENCODE_SEED_SALT);
        let (encoded_init_latents, init_encode_kind) = encode_to_latents_with_runtime_context(
            encoder,
            &init_image,
            Some(&init_encode_seeds),
            &mut runtime_context,
        )?;
        generation_runtime_kind = merge_runtime_kind(generation_runtime_kind, init_encode_kind);
        let init_latents = match request.resize_mode {
            DiffusionImg2ImgResizeMode::Image => encoded_init_latents,
            DiffusionImg2ImgResizeMode::Latent => resize_latent_batch_nearest(
                &encoded_init_latents,
                plan.latent_shape.height,
                plan.latent_shape.width,
            )?,
        };
        let mut denoise_init_latents = init_latents.clone();
        if denoise_init_latents.batch != plan.latent_shape.batch
            || denoise_init_latents.channels != plan.latent_shape.channels
            || denoise_init_latents.height != plan.latent_shape.height
            || denoise_init_latents.width != plan.latent_shape.width
        {
            return Err(DiffusionError::InvalidRequest(format!(
                "encoded init latent shape [{}x{}x{}x{}] != requested latent shape [{}x{}x{}x{}]",
                denoise_init_latents.batch,
                denoise_init_latents.channels,
                denoise_init_latents.height,
                denoise_init_latents.width,
                plan.latent_shape.batch,
                plan.latent_shape.channels,
                plan.latent_shape.height,
                plan.latent_shape.width
            )));
        }
        // MrFlow staged-sampling refine: an explicit direct-sigma schedule
        // replaces the strength-derived slice of the base schedule. `start_step`
        // is 0 because the refine schedule already starts at `first_sigma`.
        let (schedule, start_step) = if let Some(refine) = request.refine_sigma.as_ref() {
            let refine_schedule = plan.schedule.refine_direct_sigma(
                refine.first_sigma,
                refine.steps,
                refine.shifted,
            )?;
            (refine_schedule, 0usize)
        } else {
            let strength = request.denoising_strength.clamp(0.0, 1.0);
            let denoise_steps = ((plan.schedule.timesteps.len() as f32) * strength).ceil() as usize;
            let start_step = plan.schedule.timesteps.len().saturating_sub(denoise_steps);
            (plan.schedule.slice_from_step(start_step)?, start_step)
        };
        let expanded_mask = if let Some(mask) = request.mask.as_ref() {
            let mask = expand_rgb_batch_for_prompts(mask, request.batch.prompts.len())?;
            let target_width = match request.resize_mode {
                DiffusionImg2ImgResizeMode::Image => request.batch.width,
                DiffusionImg2ImgResizeMode::Latent => {
                    u32::try_from(init_image.width).map_err(|_| {
                        DiffusionError::InvalidRequest(
                            "init image width is out of range".to_string(),
                        )
                    })?
                }
            };
            let target_height = match request.resize_mode {
                DiffusionImg2ImgResizeMode::Image => request.batch.height,
                DiffusionImg2ImgResizeMode::Latent => {
                    u32::try_from(init_image.height).map_err(|_| {
                        DiffusionError::InvalidRequest(
                            "init image height is out of range".to_string(),
                        )
                    })?
                }
            };
            Some(resize_rgb_batch_nearest(
                &mask,
                target_width,
                target_height,
            )?)
        } else {
            None
        };
        let mask_weights = if let Some(mask) = expanded_mask.as_ref() {
            let (weights, mask_kind) = latent_mask_weights_with_runtime_context(
                mask,
                &denoise_init_latents,
                &mut runtime_context,
            )?;
            generation_runtime_kind = merge_runtime_kind(generation_runtime_kind, mask_kind);
            Some(weights)
        } else {
            None
        };
        let inpaint_conditioning = if let Some(mask) = expanded_mask.as_ref() {
            let (conditioning, inpaint_kind) = build_inpaint_conditioning_if_supported(
                runtime.noise.as_ref(),
                encoder,
                &init_image,
                mask,
                &denoise_init_latents,
                mask_weights.as_deref(),
                &request_seeds,
                &mut runtime_context,
            )?;
            generation_runtime_kind = merge_runtime_kind(generation_runtime_kind, inpaint_kind);
            conditioning
        } else {
            None
        };
        let inpainting_fill = request.inpainting_fill.unwrap_or(0);
        let applied_inpainting_fill = if let Some(mask_weights) = mask_weights.as_ref() {
            apply_inpainting_fill_to_latents(
                &mut denoise_init_latents,
                &plan.latents,
                mask_weights,
                inpainting_fill,
            )?
        } else {
            false
        };
        let mut latents = denoise_init_latents.clone();
        if !schedule.timesteps.is_empty() {
            let noise = plan.latents;
            if request.refine_sigma.is_some() {
                // Flow-match interpolation noising: the re-encoded super-resolved
                // image is a clean x0, so inject (1 - sigma) * x0 + sigma * noise.
                schedule.add_flow_match_refine_noise(&mut latents, &noise.data)?;
            } else {
                plan.schedule
                    .add_noise_to_latents(&mut latents, &noise.data, start_step)?;
            }
            let masked_reference =
                mask_weights
                    .as_ref()
                    .map(|mask_weights| MaskedDenoiseReference {
                        init_latents: &denoise_init_latents,
                        noise: &noise.data,
                        mask_weights,
                        source_schedule: &plan.schedule,
                        start_step,
                    });
            let denoise_output = runtime.noise.denoise_latents_with_runtime_context(
                latents,
                &schedule,
                request.batch.cfg_scale,
                positive_embeddings,
                negative_embeddings,
                plan.conditioning.prompt_attention_mask.as_ref(),
                plan.conditioning.negative_attention_mask.as_ref(),
                positive_sdxl_conditioning.as_ref(),
                negative_sdxl_conditioning.as_ref(),
                inpaint_conditioning.as_ref(),
                masked_reference.as_ref(),
                &mut runtime_context,
                progress,
            )?;
            latents = denoise_output.latents;
            generation_runtime_kind =
                merge_runtime_kind(generation_runtime_kind, denoise_output.runtime_kind);
        }
        let masked = if let Some(mask_weights) = mask_weights.as_ref() {
            let blend_kind = blend_latents_with_mask_with_runtime_context(
                &mut latents,
                &init_latents,
                mask_weights,
                &mut runtime_context,
            )?;
            generation_runtime_kind = merge_runtime_kind(generation_runtime_kind, blend_kind);
            true
        } else {
            false
        };
        let images = if request.batch.send_images {
            let (rgb, image_runtime_kind) = decode_to_rgb8_with_runtime_context(
                runtime.decoder.as_ref(),
                &latents,
                &mut runtime_context,
            )?;
            generation_runtime_kind =
                merge_runtime_kind(generation_runtime_kind, image_runtime_kind);
            encode_rgb_batch_png_base64(&rgb)?
        } else {
            Vec::new()
        };
        let mut info = diffusion_generation_info(
            self.summary(),
            generation_runtime_kind,
            &request.batch,
            &plan.latent_shape,
        );
        if let Value::Object(map) = &mut info {
            map.insert("mode".to_string(), Value::String("img2img".to_string()));
            map.insert(
                "denoising_strength".to_string(),
                json!(request.denoising_strength),
            );
            map.insert("start_step".to_string(), json!(start_step));
            map.insert("denoise_steps".to_string(), json!(schedule.timesteps.len()));
            map.insert("masked".to_string(), json!(masked));
            if request.resize_mode == DiffusionImg2ImgResizeMode::Latent {
                map.insert(
                    "resize_mode".to_string(),
                    Value::String("latent".to_string()),
                );
                map.insert("latent_resize".to_string(), json!(true));
            }
            if request.inpainting_fill.is_some() || applied_inpainting_fill {
                map.insert("inpainting_fill".to_string(), json!(inpainting_fill));
            }
            if applied_inpainting_fill {
                let masked_content = match inpainting_fill {
                    2 => "latent noise",
                    3 => "latent nothing",
                    _ => unreachable!("only latent inpaint fill modes are applied"),
                };
                map.insert(
                    "masked_content".to_string(),
                    Value::String(masked_content.to_string()),
                );
            }
        }
        Ok(DiffusionBatchOutput { images, info })
    }
}
