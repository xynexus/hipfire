// SPDX-License-Identifier: Apache-2.0
// hipfire — Idefics3 config parsing. See LICENSE / NOTICE.

use hipfire_arch_gemma3_vl::SigLipConfig;

/// Vision tower + connector + tiling parameters. The text decoder is a plain
/// `LlamaConfig` parsed from `text_config` by the shared LLaMA parser.
#[derive(Debug, Clone, PartialEq)]
pub struct Idefics3Config {
    pub vision: SigLipConfig,
    /// Pixel-shuffle factor per side (4: 32×32 patches → 8×8 = 64 tokens).
    pub scale_factor: usize,
    /// `<image>` placeholder id in the text stream.
    pub image_token_id: u32,
    pub text_hidden: usize,
    /// Processor `size.longest_edge`: the image's long side is resized to this
    /// before tiling (2048 for Granite-Docling → at most 4×4 tiles).
    pub longest_edge: usize,
}

impl Idefics3Config {
    /// Image tokens per frame: `(grid / scale)²`.
    pub fn image_seq_len(&self) -> usize {
        let side = self.vision.grid_side() / self.scale_factor;
        side * side
    }

    /// Connector input width: `vision_hidden · scale²`.
    pub fn connector_in(&self) -> usize {
        self.vision.hidden_size * self.scale_factor * self.scale_factor
    }

    pub fn from_metadata_json(metadata_json: &str) -> Option<Self> {
        let meta: serde_json::Value = serde_json::from_str(metadata_json).ok()?;
        let config = meta.get("config")?;
        let vc = config.get("vision_config")?;
        let u = |v: &serde_json::Value, k: &str| v.get(k).and_then(|x| x.as_u64());
        let vision = SigLipConfig {
            hidden_size: u(vc, "hidden_size")? as usize,
            num_hidden_layers: u(vc, "num_hidden_layers")? as usize,
            num_attention_heads: u(vc, "num_attention_heads")? as usize,
            intermediate_size: u(vc, "intermediate_size")? as usize,
            patch_size: u(vc, "patch_size")? as usize,
            // The processor's tile edge is what the tower actually sees.
            image_size: meta
                .pointer("/preprocessor_config/max_image_size/longest_edge")
                .and_then(|x| x.as_u64())
                .or_else(|| u(vc, "image_size"))? as usize,
            num_channels: u(vc, "num_channels").unwrap_or(3) as usize,
            layer_norm_eps: vc
                .get("layer_norm_eps")
                .and_then(|x| x.as_f64())
                .unwrap_or(1e-6) as f32,
        };
        // ponytail: absent preprocessor_config → idefics3's 4-tiles-per-side default.
        let longest_edge = meta
            .pointer("/preprocessor_config/size/longest_edge")
            .and_then(|x| x.as_u64())
            .map_or(4 * vision.image_size, |x| x as usize);
        Some(Self {
            vision,
            scale_factor: u(config, "scale_factor").unwrap_or(2) as usize,
            image_token_id: u(config, "image_token_id")? as u32,
            text_hidden: config
                .get("text_config")
                .and_then(|t| u(t, "hidden_size"))? as usize,
            longest_edge,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_granite_docling() {
        let meta = serde_json::json!({
            "config": {
                "model_type": "idefics3", "scale_factor": 4, "image_token_id": 100270,
                "text_config": { "hidden_size": 576 },
                "vision_config": { "hidden_size": 768, "num_hidden_layers": 12,
                    "num_attention_heads": 12, "intermediate_size": 3072,
                    "patch_size": 16, "image_size": 512 }
            },
            "preprocessor_config": { "size": {"longest_edge": 2048},
                "max_image_size": {"longest_edge": 512} }
        });
        let c = Idefics3Config::from_metadata_json(&meta.to_string()).unwrap();
        assert_eq!(c.vision.num_patches(), 1024);
        assert_eq!(c.image_seq_len(), 64);
        assert_eq!(c.connector_in(), 12288);
        assert_eq!(c.longest_edge, 2048);
    }
}
