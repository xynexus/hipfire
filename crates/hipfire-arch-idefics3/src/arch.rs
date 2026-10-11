// SPDX-License-Identifier: Apache-2.0
// hipfire — Idefics3 ServingBackend + factory. See LICENSE / NOTICE.

//! Text-only requests run the plain LLaMA path ([`run_simple_ar`]). With images,
//! each `<image>` placeholder in the rendered prompt is expanded into the
//! processor's tile layout, every frame is encoded (SigLIP → pixel shuffle →
//! one batched connector GEMM), and the rows are spliced into the LLaMA batched
//! prefill before the shared [`decode_loop`].

use hipfire_arch_gemma3_vl::{read_f32, upload_bf16, vision_forward, SigLipWeights};
use hipfire_arch_llama::{Llama, LlamaBackend};
use hipfire_rdna::{DType, Gpu, GpuTensor};
use hipfire_runtime::arch::{
    decode_loop, run_simple_ar, ArchCaps, Architecture, FactoryLoadedBackend, GenerateCtx,
    ModelShapeProfile, PromptGenerationProfile, ServeOutcome, ServingBackend, ServingFactory,
    ServingFactoryOptions, IMAGE_PLACEHOLDER,
};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::kv::KvCache;
use hipfire_runtime::llama;
use hipfire_runtime::tokenizer::{Tokenizer, LITERAL_ESC};

use crate::config::Idefics3Config;
use crate::image::{image_prompt, pixel_shuffle, preprocess_bytes};

pub struct Idefics3Backend {
    pub cfg: Idefics3Config,
    pub text: LlamaBackend,
    pub vision: SigLipWeights,
    /// `modality_projection.proj` as BF16 `[text_hidden, vision_hidden·scale²]`.
    pub proj_w: GpuTensor,
}

impl Idefics3Backend {
    /// Encode every image and expand its placeholder. Returns the expanded
    /// prompt and the projected rows `[n_rows, text_hidden]` (F32, prompt order).
    fn encode(
        &self,
        gpu: &mut Gpu,
        prompt: &str,
        images: &[&[u8]],
    ) -> Result<(String, GpuTensor, usize), String> {
        let parts = split_placeholders(prompt);
        if parts.len() - 1 != images.len() {
            return Err(format!(
                "idefics3: prompt has {} `{IMAGE_PLACEHOLDER}` placeholders for {} images",
                parts.len() - 1,
                images.len()
            ));
        }
        let vc = &self.cfg.vision;
        let mut expanded = String::from(parts[0]);
        let mut shuffled = Vec::new();
        for (bytes, tail) in images.iter().zip(&parts[1..]) {
            let tiled = preprocess_bytes(bytes, &self.cfg)?;
            for frame in &tiled.frames {
                let feats = vision_forward(gpu, &self.vision, vc, frame)
                    .map_err(|e| format!("idefics3: vision_forward: {e:?}"))?;
                let host = gpu.download_f32(&feats);
                let _ = gpu.free_tensor(feats);
                let host = host.map_err(|e| format!("idefics3: download vision: {e:?}"))?;
                shuffled.extend(pixel_shuffle(
                    &host,
                    vc.grid_side(),
                    vc.hidden_size,
                    self.cfg.scale_factor,
                ));
            }
            expanded.push_str(&image_prompt(
                tiled.rows,
                tiled.cols,
                self.cfg.image_seq_len(),
            ));
            expanded.push_str(tail);
        }
        let k = self.cfg.connector_in();
        let th = self.cfg.text_hidden;
        let n_rows = shuffled.len() / k;
        let x = gpu
            .upload_owned_f32(&shuffled, &[shuffled.len()])
            .map_err(|e| format!("idefics3: upload connector input: {e:?}"))?;
        let rows = gpu
            .alloc_tensor(&[n_rows * th], DType::F32)
            .map_err(|e| format!("idefics3: alloc rows: {e:?}"))?;
        if let Err(e) = gpu.gemm_bf16_x_bf16_wmma(&self.proj_w, &x, &rows, th, k, n_rows) {
            let _ = gpu.free_tensor(rows);
            return Err(format!("idefics3: connector gemm: {e:?}"));
        }
        Ok((expanded, rows, n_rows))
    }

    fn serve_images(
        &mut self,
        gpu: &mut Gpu,
        tok: &Tokenizer,
        ctx: &mut GenerateCtx,
    ) -> Result<ServeOutcome, String> {
        let (prompt, rows, n_rows) = self.encode(gpu, ctx.prompt, ctx.images)?;
        let ids = tok.encode(&prompt);
        let n_img = ids
            .iter()
            .filter(|&&t| t == self.cfg.image_token_id)
            .count();
        let result = if n_img != n_rows {
            Err(format!(
                "idefics3: {n_img} `{IMAGE_PLACEHOLDER}` tokens after tokenize, {n_rows} image rows"
            ))
        } else {
            let t = &mut self.text;
            llama::prefill_forward_spliced(
                gpu,
                &t.weights,
                &t.config,
                &ids,
                Some((self.cfg.image_token_id, &rows)),
                &mut t.kv_cache,
            )
            .map_err(|e| format!("idefics3: prefill: {e:?}"))
        };
        let _ = gpu.free_tensor(rows);
        let logits = result?;
        let bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(logits.as_ptr() as *const u8, logits.len() * 4) };
        gpu.hip
            .memcpy_htod(&self.text.scratch.logits.buf, bytes)
            .map_err(|e| format!("idefics3: logits upload: {e:?}"))?;
        let eos = self.text.config.eos_token;
        decode_loop(gpu, &mut self.text, tok, eos, ctx, ids.len(), ids.len())
    }
}

impl ServingBackend for Idefics3Backend {
    fn arch_id(&self) -> u32 {
        hipfire_arch_api::ARCH_ID_IDEFICS3
    }

    fn caps(&self) -> ArchCaps {
        ArchCaps {
            vision: true,
            ..ArchCaps::default()
        }
    }

    fn eos_token(&self) -> u32 {
        self.text.config.eos_token
    }

    fn serve(
        &mut self,
        gpu: &mut Gpu,
        tok: &Tokenizer,
        ctx: &mut GenerateCtx,
    ) -> Result<ServeOutcome, String> {
        if ctx.images.is_empty() {
            let eos = self.text.config.eos_token;
            return run_simple_ar(gpu, &mut self.text, tok, eos, ctx);
        }
        self.serve_images(gpu, tok, ctx)
    }

    fn reset_session(&mut self, gpu: &mut Gpu, session_id: &str) -> Result<(), String> {
        self.text.reset_session(gpu, session_id)
    }

    fn unload(self: Box<Self>, gpu: &mut Gpu) {
        let b = *self;
        b.vision.free_gpu(gpu);
        let _ = gpu.free_tensor(b.proj_w);
        Box::new(b.text).unload(gpu);
    }

    fn kld_forward(&mut self) -> Option<&mut dyn hipfire_runtime::kld_eval::ChunkScoredForward> {
        Some(&mut self.text)
    }
}

/// Split on [`IMAGE_PLACEHOLDER`]s, skipping escaped (literal-text) ones.
fn split_placeholders(prompt: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut start = 0;
    for (i, _) in prompt.match_indices(IMAGE_PLACEHOLDER) {
        if !prompt[..i].ends_with(LITERAL_ESC) {
            parts.push(&prompt[start..i]);
            start = i + IMAGE_PLACEHOLDER.len();
        }
    }
    parts.push(&prompt[start..]);
    parts
}

pub struct Idefics3ServingFactory;

pub static IDEFICS3_SERVING_FACTORY: Idefics3ServingFactory = Idefics3ServingFactory;

impl ServingFactory for Idefics3ServingFactory {
    fn arch_id(&self) -> u32 {
        hipfire_arch_api::ARCH_ID_IDEFICS3
    }

    fn family(&self) -> &'static str {
        "idefics3"
    }

    fn load(
        &self,
        hfq: &mut HfqFile,
        gpu: &mut Gpu,
        options: &ServingFactoryOptions<'_>,
    ) -> Result<FactoryLoadedBackend, String> {
        let cfg = Idefics3Config::from_metadata_json(&hfq.metadata_json)
            .ok_or("idefics3: failed to parse vision/connector config")?;
        let config = <Llama as Architecture>::config_from_hfq(hfq)?;
        let weights = <Llama as Architecture>::load_weights(hfq, &config, gpu)?;
        let (layers, kv_heads, head_dim) = (config.n_layers, config.n_kv_heads, config.head_dim);
        // LLaMA has no kvarn/asym path at head_dim 64; everything but fp32 is Q8.
        let kv = match options.kv_mode {
            "fp32" | "f32" => KvCache::new_gpu(gpu, layers, kv_heads, head_dim, options.max_seq),
            _ => KvCache::new_gpu_q8(gpu, layers, kv_heads, head_dim, options.max_seq),
        }
        .map_err(|e| format!("idefics3: kv cache: {e}"))?;
        let scratch = <Llama as Architecture>::new_state(gpu, &config)?;
        let shape = ModelShapeProfile {
            hidden_size: config.dim,
            num_layers: config.n_layers,
            vocab_size: config.vocab_size,
            intermediate_size: config.hidden_dim,
        };
        let text = LlamaBackend::new(hfq.arch_id, config, weights, scratch, kv);

        let vision = SigLipWeights::load_prefixed(hfq, &cfg.vision, gpu, "model.vision_model")?;
        let proj = read_f32(hfq, "model.connector.modality_projection.proj.weight");
        if proj.len() != cfg.text_hidden * cfg.connector_in() {
            return Err(format!(
                "idefics3: connector proj has {} elems, expected {}x{}",
                proj.len(),
                cfg.text_hidden,
                cfg.connector_in()
            ));
        }
        let proj_w =
            upload_bf16(gpu, &proj).map_err(|e| format!("idefics3: upload connector: {e:?}"))?;

        Ok(FactoryLoadedBackend {
            backend: Box::new(Idefics3Backend {
                cfg,
                text,
                vision,
                proj_w,
            }),
            family: self.family(),
            shape,
            profile: PromptGenerationProfile::default(),
            physical_cap: options.max_seq,
        })
    }
}

hipfire_runtime::register_serving_factory!(IDEFICS3_SERVING_FACTORY);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn escaped_placeholder_stays_text() {
        let p = format!("a<image>b{LITERAL_ESC}<image>c<image>");
        assert_eq!(
            split_placeholders(&p),
            ["a", &format!("b{LITERAL_ESC}<image>c"), ""]
        );
    }
}
