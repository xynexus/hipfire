// SPDX-License-Identifier: Apache-2.0
// hipfire — tokenizer-free, multi-arch tiny-model probe harness.
//
//! Loads a tiny random-init `.hfq` for ANY supported dense/MoE arch, feeds it a
//! FIXED synthetic token-ID stream (no tokenizer), and either:
//!   - **kld**: computes KL(ref || candidate) over the per-position logits of two
//!     models (a bf16 reference vs a quantized candidate of the same arch), or
//!   - **collect**: arms the model-agnostic [`CalibCollector`], runs the forward,
//!     and drains a `<name>.hessian`/`<name>.imatrix` `.calib.hfq` (HFQM).
//!
//! This is the multi-family generalization of `fixture_golden` (qwen35-only,
//! logit-hash) and of the daemon `kld_eval`/`collect` ops (qwen35-only +
//! tokenizer-bound). Activation capture works for every arch because the shared
//! `hipfire_runtime::weights::weight_gemv` chokepoint now calls
//! `gpu.maybe_capture_activation` (no-op unless a collector is armed).
//!
//! GPU-only: every forward requires `hipfire_rdna::Gpu`. Used by the
//! `tiny_quant_probe` example and the hipfire-eval `tiny_quant` battery.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use hipfire_dispatch::pipeline::superop::LoweredForward;
use hipfire_rdna::{Gpu, GpuTensor};

use hipfire_runtime::arch::{Architecture, SimpleAr};
use hipfire_runtime::calibration::CalibCollector;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::kv::KvCache;
use hipfire_runtime::weights::WeightTensor;

use hipfire_arch_deepseek4 as deepseek4;
use hipfire_arch_dots_ocr as dots_ocr_arch;
use hipfire_arch_dots_ocr::image as dots_image;
use hipfire_arch_gemma3 as gemma3;
use hipfire_arch_gemma3_vl as gemma3_vl;
use hipfire_arch_gemma4 as gemma4;
#[cfg(feature = "arch-lfm2moe")]
use hipfire_arch_lfm2moe as lfm2moe;
use hipfire_arch_minimax as minimax;
use hipfire_arch_nemotron::{model::NemotronModel, NemotronHConfig};
use hipfire_arch_qwen2::qwen2;
use hipfire_arch_qwen35::qwen35::{self, DeltaNetState, Qwen35Scratch};
use hipfire_arch_qwen35_vl as qwen35_vl_arch;
use hipfire_arch_zaya::{arch::ZayaModel, calibration as zaya_calib, gpu as zaya_gpu, ZayaConfig};
// LLaMA/Mistral (arch 0/1) live in the runtime crate (HFQ config + loader +
// forward), surfaced via the hipfire-arch-llama Architecture impl.
use hipfire_runtime::llama::{self, ForwardScratch, LlamaConfig, LlamaWeights};

/// Which arch family a fixture belongs to. Parsed from the `--arch` flag (the
/// caller always knows it — it emitted the fixture), so no arch-id sniffing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TinyArch {
    Qwen35,
    Qwen35Vl,
    Qwen35Moe,
    Deepseek4,
    Deepseek4Compressed,
    Deepseek4Mtp,
    DotsOcr,
    Qwen2,
    Gemma3,
    Gemma3Vl,
    Gemma4Dense,
    Gemma4Ple,
    Gemma4Moe,
    #[cfg(feature = "arch-lfm2moe")]
    Lfm2Moe,
    MiniMax,
    NemotronH,
    Mamba2,
    Zaya,
    Llama,
}

impl TinyArch {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s
            .trim()
            .to_ascii_lowercase()
            .replace(['-', '.'], "_")
            .as_str()
        {
            "qwen3_5" | "qwen35" | "qwen3_5_text" => Ok(Self::Qwen35),
            "qwen3_5_vl" | "qwen35_vl" | "qwen3_5_vision_language" => Ok(Self::Qwen35Vl),
            // `_indexed` is the same arch with a wider fixture (top-8,
            // moe_inter 768) so the indexed routed-expert path is admitted.
            "qwen3_5_moe" | "qwen35moe" | "qwen3_5_moe_text" | "qwen3_5_moe_indexed"
            | "qwen35moe_indexed" => Ok(Self::Qwen35Moe),
            "deepseek4" | "deepseek_v4" | "deepseek4_flash" | "deepseek_v4_flash" => {
                Ok(Self::Deepseek4)
            }
            "deepseek4_compressed" | "deepseek4_compressed_kv" => Ok(Self::Deepseek4Compressed),
            "deepseek4_mtp" | "deepseek4_mtp_draft" | "deepseek_v4_mtp" => {
                Ok(Self::Deepseek4Mtp)
            }
            "dots_ocr" | "dotsocr" | "dots_ocr_text" => Ok(Self::DotsOcr),
            "qwen2" => Ok(Self::Qwen2),
            "gemma3" | "gemma3_text" => Ok(Self::Gemma3),
            "gemma3_vl" | "gemma3_vl_text" => Ok(Self::Gemma3Vl),
            "gemma4" | "gemma4_dense" | "gemma4_text" => Ok(Self::Gemma4Dense),
            "gemma4_ple" | "gemma4_ple_sharing" | "gemma4_ple_text" => Ok(Self::Gemma4Ple),
            "gemma4_moe" | "gemma4_dense_moe" | "gemma4_moe_text" => Ok(Self::Gemma4Moe),
            #[cfg(feature = "arch-lfm2moe")]
            "lfm2" | "lfm2_moe" | "lfm2moe" | "lfm2_moe_text" => Ok(Self::Lfm2Moe),
            "minimax" | "minimax_m2" => Ok(Self::MiniMax),
            "nemotron_h" | "nemotron" => Ok(Self::NemotronH),
            "mamba2" | "mamba_2" => Ok(Self::Mamba2),
            "zaya" | "zaya1" | "zaya1_text" => Ok(Self::Zaya),
            "llama" | "mistral" | "qwen3_legacy" | "qwen3_legacy_text" | "qwen3" => {
                Ok(Self::Llama)
            }
            other => Err(format!(
                "unknown --arch '{other}' (qwen3_5|qwen3_5_vl|qwen3_5_moe|deepseek4|deepseek4_compressed|deepseek4_mtp|dots_ocr|qwen2|gemma3|gemma3_vl|gemma4_dense|gemma4_ple|gemma4_moe|lfm2_moe|minimax|nemotron_h|mamba2|zaya|llama)"
            )),
        }
    }

    /// The `.hfq` arch_id this family writes (for the HFQM calib metadata).
    pub fn arch_id(self) -> u32 {
        match self {
            Self::Llama => 0,
            Self::Qwen35 | Self::Qwen35Vl => 5,
            Self::Qwen35Moe => 6,
            Self::Deepseek4 | Self::Deepseek4Compressed | Self::Deepseek4Mtp => 9,
            Self::DotsOcr => 8,
            Self::Qwen2 => 7,
            Self::MiniMax => 10,
            #[cfg(feature = "arch-lfm2moe")]
            Self::Lfm2Moe => 11,
            Self::Gemma3 => 12,
            Self::Gemma3Vl => 13,
            Self::NemotronH => 14,
            Self::Mamba2 => 15,
            Self::Zaya => 16,
            Self::Gemma4Dense | Self::Gemma4Ple | Self::Gemma4Moe => 24,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Qwen35 => "qwen3_5",
            Self::Qwen35Vl => "qwen3_5_vl",
            Self::Qwen35Moe => "qwen3_5_moe",
            Self::Deepseek4 => "deepseek4",
            Self::Deepseek4Compressed => "deepseek4_compressed",
            Self::Deepseek4Mtp => "deepseek4_mtp",
            Self::DotsOcr => "dots_ocr",
            Self::Qwen2 => "qwen2",
            Self::Gemma3 => "gemma3",
            Self::Gemma3Vl => "gemma3_vl",
            Self::Gemma4Dense => "gemma4_dense",
            Self::Gemma4Ple => "gemma4_ple",
            Self::Gemma4Moe => "gemma4_moe",
            #[cfg(feature = "arch-lfm2moe")]
            Self::Lfm2Moe => "lfm2_moe",
            Self::MiniMax => "minimax",
            Self::NemotronH => "nemotron_h",
            Self::Mamba2 => "mamba2",
            Self::Zaya => "zaya",
            Self::Llama => "llama",
        }
    }
}

/// A loaded tiny model + everything its forward needs, behind one enum so the
/// `kld`/`collect` drivers stay arch-agnostic.
pub enum TinyModel {
    Qwen35 {
        config: qwen35::Qwen35Config,
        weights: qwen35::Qwen35Weights,
        kv: KvCache,
        dn: DeltaNetState,
        scratch: Qwen35Scratch,
    },
    Qwen35Vl {
        config: qwen35::Qwen35Config,
        weights: qwen35::Qwen35Weights,
        kv: KvCache,
        dn: DeltaNetState,
        scratch: Qwen35Scratch,
        vision_config: qwen35_vl_arch::qwen35_vl::VisionConfig,
        vision_weights: qwen35_vl_arch::qwen35_vl::VisionWeights,
        visual_tokens: Option<Vec<f32>>,
    },
    Qwen2 {
        config: qwen2::Qwen2Config,
        weights: qwen2::Qwen2Weights,
        state: qwen2::Qwen2State,
    },
    DotsOcr {
        config: dots_ocr_arch::dots_ocr::DotsOcrConfig,
        weights: dots_ocr_arch::dots_ocr::DotsOcrWeights,
        state: qwen2::Qwen2State,
        visual_tokens: Option<Vec<f32>>,
    },
    Deepseek4 {
        config: deepseek4::DeepseekV4Config,
        weights: deepseek4::DeepseekV4Weights,
        state: deepseek4::DeepseekV4State,
    },
    Deepseek4Mtp {
        config: deepseek4::DeepseekV4Config,
        weights: deepseek4::DeepseekV4Weights,
        state: deepseek4::DeepseekV4State,
    },
    Gemma3 {
        config: gemma3::Gemma3Config,
        weights: gemma3::Gemma3Weights,
        state: gemma3::Gemma3State,
    },
    Gemma3Vl {
        loaded: gemma3_vl::LoadedVl,
        state: gemma3::Gemma3State,
        image_embeddings: Option<Vec<f32>>,
    },
    Gemma4 {
        config: gemma4::Gemma4Config,
        weights: gemma4::Gemma4DenseWeights,
        state: gemma4::Gemma4DenseState,
        lowered: LoweredForward,
    },
    MiniMax {
        config: minimax::MiniMaxConfig,
        weights: minimax::MiniMaxWeights,
        state: minimax::MiniMaxState,
    },
    #[cfg(feature = "arch-lfm2moe")]
    Lfm2Moe {
        config: lfm2moe::Lfm2MoeConfig,
        weights: lfm2moe::Lfm2MoeWeights,
        state: lfm2moe::Lfm2MoeState,
    },
    NemotronH {
        model: NemotronModel,
    },
    Mamba2 {
        model: NemotronModel,
    },
    Zaya {
        model: ZayaModel,
    },
    Llama {
        config: LlamaConfig,
        weights: LlamaWeights,
        kv: KvCache,
        scratch: ForwardScratch,
    },
}

fn qwen35_vl_synthetic_patches(config: &qwen35_vl_arch::qwen35_vl::VisionConfig) -> Vec<f32> {
    let grid_h = config.spatial_merge_size;
    let grid_w = config.spatial_merge_size;
    let patch_dim = 3 * config.temporal_patch_size * config.patch_size * config.patch_size;
    let n = grid_h * grid_w * patch_dim;
    (0..n)
        .map(|i| {
            let x = ((i as u32)
                .wrapping_mul(1_664_525)
                .wrapping_add(1_013_904_223)
                & 0xffff) as f32
                / 65_535.0;
            x * 0.2 - 0.1
        })
        .collect()
}

fn dots_ocr_synthetic_image_patches(
    config: &dots_ocr_arch::dots_ocr::DotsVisionConfig,
) -> Result<(Vec<f32>, usize, usize), String> {
    if config.patch_size != dots_image::PATCH_SIZE
        || config.spatial_merge_size != dots_image::SPATIAL_MERGE_SIZE
        || config.temporal_patch_size != dots_image::TEMPORAL_PATCH_SIZE
        || config.num_channels != 3
    {
        return Err(format!(
            "dots-ocr tiny image path requires production preprocessing dims \
             patch={} sm={} temporal={} channels=3; got patch={} sm={} temporal={} channels={}",
            dots_image::PATCH_SIZE,
            dots_image::SPATIAL_MERGE_SIZE,
            dots_image::TEMPORAL_PATCH_SIZE,
            config.patch_size,
            config.spatial_merge_size,
            config.temporal_patch_size,
            config.num_channels
        ));
    }

    let h = dots_image::PATCH_SIZE * dots_image::SPATIAL_MERGE_SIZE * 2;
    let w = h;
    let mut rgb = vec![0u8; h * w * 3];
    for y in 0..h {
        for x in 0..w {
            let base = (y * w + x) * 3;
            rgb[base] = ((x * 9 + y * 3) & 0xff) as u8;
            rgb[base + 1] = ((x * 5 + y * 11 + 17) & 0xff) as u8;
            rgb[base + 2] = ((x * 13 + y * 7 + 29) & 0xff) as u8;
        }
    }
    let chw = dots_image::clip_normalise(&rgb, h, w);
    let patches = dots_image::extract_patches(&chw, h, w);
    Ok((
        patches,
        h / dots_image::PATCH_SIZE,
        w / dots_image::PATCH_SIZE,
    ))
}

fn gemma3_vl_synthetic_png_bytes() -> &'static [u8] {
    &[
        137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13, 73, 72, 68, 82, 0, 0, 0, 1, 0, 0, 0, 1, 8, 2,
        0, 0, 0, 144, 119, 83, 222, 0, 0, 0, 12, 73, 68, 65, 84, 120, 156, 99, 248, 207, 208, 0, 0,
        3, 129, 1, 128, 162, 173, 150, 129, 0, 0, 0, 0, 73, 69, 78, 68, 174, 66, 96, 130,
    ]
}

impl TinyModel {
    /// Load `path` as `arch`, sizing per-arch state for `max_seq` positions.
    pub fn load(
        arch: TinyArch,
        path: &Path,
        gpu: &mut Gpu,
        max_seq: usize,
    ) -> Result<Self, String> {
        let mut hfq = HfqFile::open(path).map_err(|e| format!("open {path:?}: {e:?}"))?;
        match arch {
            TinyArch::Qwen35 | TinyArch::Qwen35Moe | TinyArch::Qwen35Vl => {
                let config =
                    qwen35::config_from_hfq(&hfq).ok_or("qwen35: config_from_hfq failed")?;
                let vision = if arch == TinyArch::Qwen35Vl {
                    let vc = qwen35_vl_arch::qwen35_vl::vision_config_from_hfq(&hfq)
                        .ok_or("qwen35-vl: vision_config_from_hfq failed")?;
                    let vw = qwen35_vl_arch::qwen35_vl::load_vision_weights(&mut hfq, &vc, gpu)
                        .map_err(|e| format!("qwen35-vl load_vision_weights: {e:?}"))?;
                    Some((vc, vw))
                } else {
                    None
                };
                let weights = qwen35::load_weights(&mut hfq, &config, gpu)
                    .map_err(|e| format!("qwen35 load_weights: {e:?}"))?;
                let kv = KvCache::new_gpu_q8(
                    gpu,
                    config.n_layers,
                    config.n_kv_heads,
                    config.head_dim,
                    max_seq + 16,
                )
                .map_err(|e| format!("qwen35 kv: {e:?}"))?;
                let dn =
                    DeltaNetState::new(gpu, &config).map_err(|e| format!("qwen35 dn: {e:?}"))?;
                let scratch = Qwen35Scratch::new(gpu, &config, 64)
                    .map_err(|e| format!("qwen35 scratch: {e:?}"))?;
                if let Some((vision_config, vision_weights)) = vision {
                    Ok(Self::Qwen35Vl {
                        config,
                        weights,
                        kv,
                        dn,
                        scratch,
                        vision_config,
                        vision_weights,
                        visual_tokens: None,
                    })
                } else {
                    Ok(Self::Qwen35 {
                        config,
                        weights,
                        kv,
                        dn,
                        scratch,
                    })
                }
            }
            TinyArch::Qwen2 => {
                let config = qwen2::config_from_hfq(&hfq).ok_or("qwen2: config_from_hfq failed")?;
                let weights = qwen2::load_weights(&mut hfq, &config, gpu)
                    .map_err(|e| format!("qwen2 load_weights: {e:?}"))?;
                let state = qwen2::Qwen2State::new_with_max_seq(gpu, &config, max_seq)
                    .map_err(|e| format!("qwen2 state: {e:?}"))?;
                Ok(Self::Qwen2 {
                    config,
                    weights,
                    state,
                })
            }
            TinyArch::DotsOcr => {
                let config = dots_ocr_arch::DotsOcr::config_from_hfq(&hfq)?;
                let weights = dots_ocr_arch::DotsOcr::load_weights(&mut hfq, &config, gpu)
                    .map_err(|e| format!("dots-ocr load_weights: {e:?}"))?;
                let state = qwen2::Qwen2State::new_with_max_seq(gpu, &config.text, max_seq)
                    .map_err(|e| format!("dots-ocr state: {e:?}"))?;
                Ok(Self::DotsOcr {
                    config,
                    weights,
                    state,
                    visual_tokens: None,
                })
            }
            TinyArch::Deepseek4 | TinyArch::Deepseek4Compressed | TinyArch::Deepseek4Mtp => {
                let config = deepseek4::DeepseekV4Config::from_hfq(&hfq)?;
                let weights = deepseek4::DeepseekV4::load_weights(&mut hfq, &config, gpu)
                    .map_err(|e| format!("deepseek4 load_weights: {e:?}"))?;
                let state = deepseek4::DeepseekV4State::new(&config)
                    .map_err(|e| format!("deepseek4 state: {e:?}"))?;
                match arch {
                    TinyArch::Deepseek4Mtp => Ok(Self::Deepseek4Mtp {
                        config,
                        weights,
                        state,
                    }),
                    _ => Ok(Self::Deepseek4 {
                        config,
                        weights,
                        state,
                    }),
                }
            }
            TinyArch::Gemma3 => {
                let config =
                    gemma3::config_from_hfq(&hfq).ok_or("gemma3: config_from_hfq failed")?;
                let weights = gemma3::load_weights(&mut hfq, &config, gpu)
                    .map_err(|e| format!("gemma3 load_weights: {e:?}"))?;
                let state = gemma3::Gemma3State::new_with_max_seq(
                    gpu,
                    &config,
                    max_seq,
                    hipfire_runtime::kv::KvQuantMode::Unquantized,
                    4,
                )
                .map_err(|e| format!("gemma3 state: {e:?}"))?;
                Ok(Self::Gemma3 {
                    config,
                    weights,
                    state,
                })
            }
            TinyArch::Gemma3Vl => {
                let loaded = gemma3_vl::load_vl(&mut hfq, gpu)?;
                let state = gemma3::Gemma3State::new_with_max_seq(
                    gpu,
                    &loaded.text_cfg,
                    max_seq,
                    hipfire_runtime::kv::KvQuantMode::Unquantized,
                    4,
                )
                .map_err(|e| format!("gemma3-vl state: {e:?}"))?;
                Ok(Self::Gemma3Vl {
                    loaded,
                    state,
                    image_embeddings: None,
                })
            }
            TinyArch::Gemma4Dense | TinyArch::Gemma4Ple | TinyArch::Gemma4Moe => {
                let config = gemma4::Gemma4::config_from_hfq(&hfq)
                    .map_err(|e| format!("gemma4 config_from_hfq: {e}"))?;
                let weights = gemma4::load_dense_weights(&mut hfq, gpu, &config)
                    .map_err(|e| format!("gemma4 load_dense_weights: {e:?}"))?;
                let state = gemma4::Gemma4DenseState::new(gpu, &config, max_seq)
                    .map_err(|e| format!("gemma4 state: {e:?}"))?;
                let lowered = gemma4::lower_dense_forward(&config, &state);
                Ok(Self::Gemma4 {
                    config,
                    weights,
                    state,
                    lowered,
                })
            }
            TinyArch::MiniMax => {
                let config = minimax::MiniMaxConfig::from_hfq(&hfq)?;
                let weights = minimax::MiniMaxWeights::load(&mut hfq, &config, gpu, None)?;
                let state = minimax::MiniMaxState::new_with_max_seq(gpu, &config, max_seq)?;
                Ok(Self::MiniMax {
                    config,
                    weights,
                    state,
                })
            }
            #[cfg(feature = "arch-lfm2moe")]
            TinyArch::Lfm2Moe => {
                let config = lfm2moe::Lfm2MoeConfig::from_hfq(&hfq)?;
                let weights = lfm2moe::Lfm2MoeWeights::load(&mut hfq, &config, gpu)?;
                let state = lfm2moe::Lfm2MoeState::new_with_max_seq(gpu, &config, max_seq)?;
                Ok(Self::Lfm2Moe {
                    config,
                    weights,
                    state,
                })
            }
            TinyArch::NemotronH => {
                let meta: serde_json::Value = serde_json::from_str(&hfq.metadata_json)
                    .map_err(|e| format!("nemotron_h metadata parse: {e}"))?;
                let cfg_json = meta
                    .get("config")
                    .ok_or("nemotron_h: metadata_json missing 'config'")?;
                let config = NemotronHConfig::from_json(cfg_json)
                    .map_err(|e| format!("nemotron_h config: {e}"))?;
                let model = NemotronModel::from_hfq(gpu, &hfq, config, max_seq)
                    .map_err(|e| format!("nemotron_h NemotronModel::from_hfq: {e}"))?;
                Ok(Self::NemotronH { model })
            }
            TinyArch::Mamba2 => {
                let meta: serde_json::Value = serde_json::from_str(&hfq.metadata_json)
                    .map_err(|e| format!("mamba2 metadata parse: {e}"))?;
                let cfg_json = meta
                    .get("config")
                    .ok_or("mamba2: metadata_json missing 'config'")?;
                let config = NemotronHConfig::from_mamba2_json(cfg_json)
                    .map_err(|e| format!("mamba2 config: {e}"))?;
                let model = NemotronModel::from_hfq(gpu, &hfq, config, max_seq)
                    .map_err(|e| format!("mamba2 NemotronModel::from_hfq: {e}"))?;
                Ok(Self::Mamba2 { model })
            }
            TinyArch::Zaya => {
                let meta: serde_json::Value = serde_json::from_str(&hfq.metadata_json)
                    .map_err(|e| format!("zaya metadata parse: {e}"))?;
                let cfg_json = meta
                    .get("config")
                    .ok_or("zaya: metadata_json missing 'config'")?;
                let config =
                    ZayaConfig::from_json(cfg_json).map_err(|e| format!("zaya config: {e}"))?;
                let model = ZayaModel::from_hfq(gpu, &hfq, config, max_seq)
                    .map_err(|e| format!("zaya ZayaModel::from_hfq: {e}"))?;
                Ok(Self::Zaya { model })
            }
            TinyArch::Llama => {
                let config = hipfire_runtime::hfq::config_from_hfq(&hfq)
                    .ok_or("llama: config_from_hfq failed")?;
                let weights = hipfire_runtime::hfq::load_weights_hfq(&hfq, &config, gpu)
                    .map_err(|e| format!("llama load_weights_hfq: {e:?}"))?;
                let kv = KvCache::new_gpu_q8(
                    gpu,
                    config.n_layers,
                    config.n_kv_heads,
                    config.head_dim,
                    max_seq + 16,
                )
                .map_err(|e| format!("llama kv: {e:?}"))?;
                let scratch = ForwardScratch::new(gpu, &config)
                    .map_err(|e| format!("llama scratch: {e:?}"))?;
                Ok(Self::Llama {
                    config,
                    weights,
                    kv,
                    scratch,
                })
            }
        }
    }

    pub fn vocab(&self) -> usize {
        match self {
            Self::Qwen35 { config, .. } => config.vocab_size,
            Self::Qwen35Vl { config, .. } => config.vocab_size,
            Self::Qwen2 { config, .. } => config.vocab_size,
            Self::DotsOcr { config, .. } => config.text.vocab_size,
            Self::Deepseek4 { config, .. } => config.vocab_size,
            Self::Deepseek4Mtp { config, .. } => config.vocab_size,
            Self::Gemma3 { config, .. } => config.vocab_size,
            Self::Gemma3Vl { loaded, .. } => loaded.text_cfg.vocab_size,
            Self::Gemma4 { config, .. } => config.vocab_size,
            Self::MiniMax { config, .. } => config.vocab_size,
            #[cfg(feature = "arch-lfm2moe")]
            Self::Lfm2Moe { config, .. } => config.vocab_size,
            Self::NemotronH { model } => model.config().vocab_size,
            Self::Mamba2 { model } => model.config().vocab_size,
            Self::Zaya { model } => model.config().vocab_size,
            Self::Llama { config, .. } => config.vocab_size,
        }
    }

    /// Forward one token, returning the host logits (`[vocab]`). `pos` is honored
    /// by qwen35/minimax; qwen2/gemma3 self-increment their own position counter,
    /// so callers MUST feed tokens strictly in order from a fresh state (which
    /// `run_logits`/`run_collect` do) — this is not a random-access scorer.
    pub fn forward_logits(
        &mut self,
        gpu: &mut Gpu,
        token: u32,
        pos: usize,
    ) -> Result<Vec<f32>, String> {
        match self {
            Self::Qwen35 {
                config,
                weights,
                kv,
                dn,
                scratch,
            } => {
                qwen35::forward_scratch(gpu, weights, config, token, pos, kv, dn, scratch)
                    .map_err(|e| format!("qwen35 forward: {e:?}"))?;
                gpu.download_f32(&scratch.logits)
                    .map_err(|e| format!("dl logits: {e:?}"))
            }
            Self::Qwen35Vl {
                config,
                weights,
                kv,
                dn,
                scratch,
                vision_config,
                vision_weights,
                visual_tokens,
            } => {
                if pos == 0 {
                    if visual_tokens.is_none() {
                        let patches = qwen35_vl_synthetic_patches(vision_config);
                        let tokens = qwen35_vl_arch::qwen35_vl::vision_forward(
                            gpu,
                            vision_weights,
                            vision_config,
                            &patches,
                            vision_config.spatial_merge_size,
                            vision_config.spatial_merge_size,
                        )
                        .map_err(|e| format!("qwen35-vl vision forward: {e:?}"))?;
                        if tokens.len() != config.dim {
                            return Err(format!(
                                "qwen35-vl synthetic vision emitted {} floats, expected one token of dim {}",
                                tokens.len(),
                                config.dim
                            ));
                        }
                        *visual_tokens = Some(tokens);
                    }
                    let emb = visual_tokens.as_ref().unwrap();
                    qwen35::forward_scratch_embed(gpu, weights, config, emb, pos, kv, dn, scratch)
                        .map_err(|e| format!("qwen35-vl forward_scratch_embed: {e:?}"))?;
                } else {
                    qwen35::forward_scratch(gpu, weights, config, token, pos, kv, dn, scratch)
                        .map_err(|e| format!("qwen35-vl forward: {e:?}"))?;
                }
                gpu.download_f32(&scratch.logits)
                    .map_err(|e| format!("dl logits: {e:?}"))
            }
            Self::Qwen2 {
                config,
                weights,
                state,
            } => {
                qwen2::forward_step(gpu, weights, config, state, token)
                    .map_err(|e| format!("qwen2 forward: {e:?}"))?;
                gpu.download_f32(&state.logits)
                    .map_err(|e| format!("dl logits: {e:?}"))
            }
            Self::DotsOcr {
                config,
                weights,
                state,
                visual_tokens,
            } => {
                let dim = config.text.hidden_size;
                if visual_tokens.is_none() && pos == 0 {
                    let (patches, grid_h, grid_w) =
                        dots_ocr_synthetic_image_patches(&config.vision)?;
                    let n_patches = grid_h * grid_w;
                    let n_visual = n_patches
                        / (config.vision.spatial_merge_size * config.vision.spatial_merge_size);
                    let patch_dim = patches.len() / n_patches;
                    let patches_gpu = gpu
                        .upload_f32(&patches, &[n_patches, patch_dim])
                        .map_err(|e| format!("dots-ocr upload synthetic patches: {e:?}"))?;
                    let merged_gpu = dots_ocr_arch::dots_ocr::vision_forward(
                        gpu,
                        &weights.vision,
                        &config.vision,
                        &patches_gpu,
                        grid_h,
                        grid_w,
                    )
                    .map_err(|e| format!("dots-ocr vision forward: {e:?}"))?;
                    gpu.free_tensor(patches_gpu)
                        .map_err(|e| format!("dots-ocr free patches: {e:?}"))?;
                    let merged = gpu
                        .download_f32(&merged_gpu)
                        .map_err(|e| format!("dots-ocr dl visual tokens: {e:?}"))?;
                    gpu.free_tensor(merged_gpu)
                        .map_err(|e| format!("dots-ocr free visual tokens: {e:?}"))?;
                    if merged.len() != n_visual * dim {
                        return Err(format!(
                            "dots-ocr synthetic image emitted {} floats, expected {n_visual} visual tokens of dim {}",
                            merged.len(),
                            dim
                        ));
                    }
                    *visual_tokens = Some(merged);
                }
                let n_visual = visual_tokens.as_ref().map_or(0, |v| v.len() / dim);
                if pos < n_visual {
                    let rows = visual_tokens.as_ref().unwrap();
                    let emb = &rows[pos * dim..(pos + 1) * dim];
                    qwen2::forward_step_with_embed(gpu, &weights.text, &config.text, state, emb)
                        .map_err(|e| format!("dots-ocr image-token splice forward: {e:?}"))?;
                } else {
                    qwen2::forward_step(gpu, &weights.text, &config.text, state, token)
                        .map_err(|e| format!("dots-ocr forward: {e:?}"))?;
                }
                gpu.download_f32(&state.logits)
                    .map_err(|e| format!("dl logits: {e:?}"))
            }
            Self::Deepseek4 {
                config,
                weights,
                state,
            } => deepseek4::forward::decode_step(config, weights, state, gpu, token, pos as u32),
            Self::Deepseek4Mtp {
                config,
                weights,
                state,
            } => {
                deepseek4::forward::decode_step(config, weights, state, gpu, token, pos as u32)?;
                let h_n_ptr = state
                    .mtp_last_hidden
                    .as_ref()
                    .ok_or("deepseek4 MTP: decode_step did not capture mtp_last_hidden")?
                    as *const hipfire_rdna::GpuTensor;
                // `mtp_forward` mutates scratch and refreshes `mtp_last_hidden`,
                // but it only reads the initial hidden. Keep the borrow shape
                // identical to the production speculative decoder.
                let h_n = unsafe { &*h_n_ptr };
                deepseek4::forward::mtp_forward(config, weights, state, gpu, h_n, token, pos as u32)
            }
            Self::Gemma3 {
                config,
                weights,
                state,
            } => {
                gemma3::forward::forward_step(gpu, weights, config, state, token)
                    .map_err(|e| format!("gemma3 forward: {e:?}"))?;
                gpu.download_f32(&state.logits)
                    .map_err(|e| format!("dl logits: {e:?}"))
            }
            Self::Gemma3Vl {
                loaded,
                state,
                image_embeddings,
            } => {
                let dim = loaded.text_cfg.hidden_size;
                if image_embeddings.is_none() && pos == 0 {
                    let patches = gemma3_vl::preprocess_image_bytes(
                        gemma3_vl_synthetic_png_bytes(),
                        &loaded.vl_cfg.vision,
                    )?;
                    let vis = gemma3_vl::vision_forward(
                        gpu,
                        &loaded.weights.vision,
                        &loaded.vl_cfg.vision,
                        &patches,
                    )
                    .map_err(|e| format!("gemma3-vl vision forward: {e:?}"))?;
                    let projected = match gemma3_vl::project(
                        gpu,
                        &loaded.weights.projector,
                        &loaded.vl_cfg,
                        &vis,
                    ) {
                        Ok(t) => t,
                        Err(e) => {
                            let _ = gpu.free_tensor(vis);
                            return Err(format!("gemma3-vl projector: {e:?}"));
                        }
                    };
                    gpu.free_tensor(vis)
                        .map_err(|e| format!("gemma3-vl free vision output: {e:?}"))?;
                    let embeds = gpu
                        .download_f32(&projected)
                        .map_err(|e| format!("gemma3-vl dl image embeddings: {e:?}"))?;
                    gpu.free_tensor(projected)
                        .map_err(|e| format!("gemma3-vl free image embeddings: {e:?}"))?;
                    let expected = loaded.vl_cfg.mm_tokens_per_image * dim;
                    if embeds.len() != expected {
                        return Err(format!(
                            "gemma3-vl synthetic image emitted {} floats, expected {} image tokens of dim {}",
                            embeds.len(),
                            loaded.vl_cfg.mm_tokens_per_image,
                            dim
                        ));
                    }
                    *image_embeddings = Some(embeds);
                }
                let n_image_tokens = image_embeddings.as_ref().map_or(0, |v| v.len() / dim);
                if pos < n_image_tokens {
                    let rows = image_embeddings.as_ref().unwrap();
                    let emb = &rows[pos * dim..(pos + 1) * dim];
                    gemma3::forward::forward_step_with_embed(
                        gpu,
                        &loaded.weights.text,
                        &loaded.text_cfg,
                        state,
                        emb,
                    )
                    .map_err(|e| format!("gemma3-vl image-token splice forward: {e:?}"))?;
                } else {
                    gemma3::forward::forward_step(
                        gpu,
                        &loaded.weights.text,
                        &loaded.text_cfg,
                        state,
                        token,
                    )
                    .map_err(|e| format!("gemma3-vl forward: {e:?}"))?;
                }
                gpu.download_f32(&state.logits)
                    .map_err(|e| format!("dl logits: {e:?}"))
            }
            Self::Gemma4 {
                config,
                weights,
                state,
                lowered,
            } => {
                gemma4::forward_step(gpu, weights, config, state, lowered, token)
                    .map_err(|e| format!("gemma4 forward: {e:?}"))?;
                gemma4::logits(gpu, state).map_err(|e| format!("dl logits: {e:?}"))
            }
            Self::MiniMax {
                config,
                weights,
                state,
            } => minimax::forward::decode_step(config, weights, state, gpu, token, pos as u32),
            #[cfg(feature = "arch-lfm2moe")]
            Self::Lfm2Moe {
                config,
                weights,
                state,
            } => lfm2moe::forward::decode_step(config, weights, state, gpu, token, pos as u32),
            Self::NemotronH { model } => model
                .forward(gpu, token, pos)
                .map_err(|e| format!("nemotron_h forward: {e:?}")),
            Self::Mamba2 { model } => model
                .forward(gpu, token, pos)
                .map_err(|e| format!("mamba2 forward: {e:?}")),
            Self::Zaya { model } => model
                .decode_step(gpu, token, pos)
                .and_then(|_| {
                    gpu.download_f32(model.logits())
                        .map_err(|e| format!("dl logits: {e:?}"))
                })
                .map_err(|e| format!("zaya forward: {e:?}")),
            Self::Llama {
                config,
                weights,
                kv,
                scratch,
            } => {
                // forward_scratch computes logits into scratch.logits, THEN samples.
                // We pass greedy/no-op sampling params and read the raw pre-sample
                // logits (the sampled-token return value is discarded).
                llama::forward_scratch(
                    gpu, weights, config, token, pos, kv, scratch, 0.0, 1.0, 0, 0, 1.0,
                )
                .map_err(|e| format!("llama forward: {e:?}"))?;
                gpu.download_f32(&scratch.logits)
                    .map_err(|e| format!("dl logits: {e:?}"))
            }
        }
    }

    /// Map each captured linear's device-buffer pointer → its checkpoint tensor
    /// name (minus `.weight`), so the collector keys Hessians by the same name
    /// the quantizer matches `--hessian`/`HIPFIRE_QTIP_HESSIAN` against. MoE
    /// routed experts (indexed-GEMV, not `weight_gemv`) are intentionally absent.
    pub fn capture_names(&self) -> HashMap<usize, String> {
        match self {
            // qwen35 ships a typed walker, but it labels tensors with the
            // real-checkpoint `model.language_model.` prefix. The tiny fixtures
            // use the short `model.` prefix, and the quantizer keys the Hessian
            // sidecar by the .hfq weight name (short) — so normalize the prefix
            // here, else qtip3-sim LDLQ matches 0 tensors and silently falls back
            // to plain QTIP. (The real collect path keeps the long prefix; both
            // sides agree there.)
            Self::Qwen35 { weights, .. } | Self::Qwen35Vl { weights, .. } => {
                qwen35::build_capture_names(weights)
                    .into_iter()
                    .map(|(ptr, name)| {
                        let short = name
                            .strip_prefix("model.language_model.")
                            .map(|rest| format!("model.{rest}"))
                            .unwrap_or(name);
                        (ptr, short)
                    })
                    .collect()
            }
            Self::Qwen2 { weights, .. } => {
                let mut m = HashMap::new();
                let mut put = |w: &WeightTensor, n: String| {
                    m.insert(w.buf.buf.as_ptr() as usize, n);
                };
                for (i, l) in weights.layers.iter().enumerate() {
                    let p = format!("model.layers.{i}");
                    put(&l.wq, format!("{p}.self_attn.q_proj"));
                    put(&l.wk, format!("{p}.self_attn.k_proj"));
                    put(&l.wv, format!("{p}.self_attn.v_proj"));
                    put(&l.wo, format!("{p}.self_attn.o_proj"));
                    put(&l.w_gate, format!("{p}.mlp.gate_proj"));
                    put(&l.w_up, format!("{p}.mlp.up_proj"));
                    put(&l.w_down, format!("{p}.mlp.down_proj"));
                }
                m
            }
            Self::DotsOcr { weights, .. } => {
                let mut m = HashMap::new();
                let mut put = |w: &WeightTensor, n: String| {
                    m.insert(w.buf.buf.as_ptr() as usize, n);
                };
                for (i, l) in weights.text.layers.iter().enumerate() {
                    let p = format!("model.layers.{i}");
                    put(&l.wq, format!("{p}.self_attn.q_proj"));
                    put(&l.wk, format!("{p}.self_attn.k_proj"));
                    put(&l.wv, format!("{p}.self_attn.v_proj"));
                    put(&l.wo, format!("{p}.self_attn.o_proj"));
                    put(&l.w_gate, format!("{p}.mlp.gate_proj"));
                    put(&l.w_up, format!("{p}.mlp.up_proj"));
                    put(&l.w_down, format!("{p}.mlp.down_proj"));
                }
                m
            }
            Self::Deepseek4 { weights, .. } | Self::Deepseek4Mtp { weights, .. } => {
                let mut m = HashMap::new();
                let mut put = |w: &hipfire_rdna::GpuTensor, n: String| {
                    m.insert(w.buf.as_ptr() as usize, n);
                };
                if let Some(w) = &weights.head {
                    put(w, "head".to_string());
                }
                for (i, l) in weights.layers.iter().enumerate() {
                    let p = format!("layers.{i}");
                    let attn = format!("{p}.attn");
                    if let Some(w) = &l.wq_a {
                        put(w, format!("{attn}.wq_a"));
                    }
                    if let Some(w) = &l.wq_b {
                        put(w, format!("{attn}.wq_b"));
                    }
                    if let Some(w) = &l.wkv {
                        put(w, format!("{attn}.wkv"));
                    }
                    if let Some(w) = &l.wo_a {
                        put(w, format!("{attn}.wo_a"));
                    }
                    if let Some(w) = &l.wo_b {
                        put(w, format!("{attn}.wo_b"));
                    }
                    if let Some(w) = &l.compressor_wkv {
                        put(w, format!("{attn}.compressor.wkv"));
                    }
                    if let Some(w) = &l.compressor_wgate {
                        put(w, format!("{attn}.compressor.wgate"));
                    }
                    if let Some(w) = &l.indexer_wq_b {
                        put(w, format!("{attn}.indexer.wq_b"));
                    }
                    if let Some(w) = &l.indexer_weights_proj {
                        put(w, format!("{attn}.indexer.weights_proj"));
                    }
                    if let Some(w) = &l.indexer_compressor_wkv {
                        put(w, format!("{attn}.indexer.compressor.wkv"));
                    }
                    if let Some(w) = &l.indexer_compressor_wgate {
                        put(w, format!("{attn}.indexer.compressor.wgate"));
                    }
                    if let Some(w) = &l.gate_weight {
                        put(w, format!("{p}.ffn.gate"));
                    }
                    if let Some(w) = &l.shared_w1 {
                        put(w, format!("{p}.ffn.shared_experts.w1"));
                    }
                    if let Some(w) = &l.shared_w2 {
                        put(w, format!("{p}.ffn.shared_experts.w2"));
                    }
                    if let Some(w) = &l.shared_w3 {
                        put(w, format!("{p}.ffn.shared_experts.w3"));
                    }
                }
                if let Some(l) = &weights.mtp_layer {
                    let p = "mtp.0";
                    let attn = format!("{p}.attn");
                    if let Some(w) = &l.wq_a {
                        put(w, format!("{attn}.wq_a"));
                    }
                    if let Some(w) = &l.wq_b {
                        put(w, format!("{attn}.wq_b"));
                    }
                    if let Some(w) = &l.wkv {
                        put(w, format!("{attn}.wkv"));
                    }
                    if let Some(w) = &l.wo_a {
                        put(w, format!("{attn}.wo_a"));
                    }
                    if let Some(w) = &l.wo_b {
                        put(w, format!("{attn}.wo_b"));
                    }
                    if let Some(w) = &l.mtp_e_proj {
                        put(w, format!("{p}.e_proj"));
                    }
                    if let Some(w) = &l.mtp_h_proj {
                        put(w, format!("{p}.h_proj"));
                    }
                    if let Some(w) = &l.gate_weight {
                        put(w, format!("{p}.ffn.gate"));
                    }
                    if let Some(w) = &l.shared_w1 {
                        put(w, format!("{p}.ffn.shared_experts.w1"));
                    }
                    if let Some(w) = &l.shared_w2 {
                        put(w, format!("{p}.ffn.shared_experts.w2"));
                    }
                    if let Some(w) = &l.shared_w3 {
                        put(w, format!("{p}.ffn.shared_experts.w3"));
                    }
                }
                m
            }
            Self::Gemma3 { weights, .. } => {
                let mut m = HashMap::new();
                let mut put = |w: &WeightTensor, n: String| {
                    m.insert(w.buf.buf.as_ptr() as usize, n);
                };
                for (i, l) in weights.layers.iter().enumerate() {
                    let p = format!("model.layers.{i}");
                    put(&l.wq, format!("{p}.self_attn.q_proj"));
                    put(&l.wk, format!("{p}.self_attn.k_proj"));
                    put(&l.wv, format!("{p}.self_attn.v_proj"));
                    put(&l.wo, format!("{p}.self_attn.o_proj"));
                    put(&l.w_gate, format!("{p}.mlp.gate_proj"));
                    put(&l.w_up, format!("{p}.mlp.up_proj"));
                    put(&l.w_down, format!("{p}.mlp.down_proj"));
                }
                m
            }
            Self::Gemma3Vl { loaded, .. } => {
                let mut m = HashMap::new();
                let mut put = |w: &WeightTensor, n: String| {
                    m.insert(w.buf.buf.as_ptr() as usize, n);
                };
                for (i, l) in loaded.weights.text.layers.iter().enumerate() {
                    let p = format!("language_model.model.layers.{i}");
                    put(&l.wq, format!("{p}.self_attn.q_proj"));
                    put(&l.wk, format!("{p}.self_attn.k_proj"));
                    put(&l.wv, format!("{p}.self_attn.v_proj"));
                    put(&l.wo, format!("{p}.self_attn.o_proj"));
                    put(&l.w_gate, format!("{p}.mlp.gate_proj"));
                    put(&l.w_up, format!("{p}.mlp.up_proj"));
                    put(&l.w_down, format!("{p}.mlp.down_proj"));
                }
                m
            }
            // The arch's own walker, not a copy of it. The copy this replaces
            // was identical except that it omitted `lm_head`, which the real
            // walker captures — and lm_head is LDLQ-eligible, so the omission
            // showed up as a permanent `missing=1` in the calibrated cells: the
            // harness was scoring `++` formats against a Hessian set it had
            // silently narrowed. That is the whole argument against hand-rolled
            // capture maps in one instance, so use the walker.
            Self::MiniMax { weights, .. } => minimax::calibration::build_capture_names(weights),
            Self::NemotronH { model } | Self::Mamba2 { model } => model.build_capture_names(),
            Self::Zaya { model } => zaya_gpu::build_capture_names(model.weights()),
            #[cfg(feature = "arch-lfm2moe")]
            Self::Lfm2Moe { weights, .. } => lfm2moe::calibration::build_capture_names(weights),
            Self::Gemma4 { weights, .. } => {
                let mut m = HashMap::new();
                let mut put = |w: &WeightTensor, n: String| {
                    m.insert(w.buf.buf.as_ptr() as usize, n);
                };
                for (i, l) in weights.layers.iter().enumerate() {
                    let p = format!("model.language_model.layers.{i}");
                    let attn = format!("{p}.self_attn");
                    put(&l.wq, format!("{attn}.q_proj"));
                    put(&l.wk, format!("{attn}.k_proj"));
                    if let Some(wv) = &l.wv {
                        put(wv, format!("{attn}.v_proj"));
                    }
                    put(&l.wo, format!("{attn}.o_proj"));
                    put(&l.w_gate, format!("{p}.mlp.gate_proj"));
                    put(&l.w_up, format!("{p}.mlp.up_proj"));
                    put(&l.w_down, format!("{p}.mlp.down_proj"));
                    if let Some(ple) = &l.ple {
                        put(&ple.input_gate, format!("{p}.per_layer_input_gate"));
                        put(&ple.projection, format!("{p}.per_layer_projection"));
                    }
                    if let Some(moe) = &l.moe {
                        put(&moe.router, format!("{p}.router.proj"));
                        for (expert, weights) in moe.experts.iter().enumerate() {
                            let ep = format!("{p}.experts.{expert}");
                            put(&weights.gate, format!("{ep}.gate_proj"));
                            put(&weights.up, format!("{ep}.up_proj"));
                            put(&weights.down, format!("{ep}.down_proj"));
                        }
                    }
                }
                if let Some(ple) = &weights.ple {
                    put(
                        &ple.embed_per_layer,
                        "model.language_model.embed_tokens_per_layer".to_string(),
                    );
                    put(
                        &ple.model_projection,
                        "model.language_model.per_layer_model_projection".to_string(),
                    );
                }
                m
            }
            // The arch's own walker over the full layer range. The copy this
            // replaces was textually identical to it — no drift had happened
            // yet, which is the argument for switching before it does rather
            // than after.
            Self::Llama { weights, .. } => {
                hipfire_runtime::llama_calibration::build_capture_names_for_layers(
                    weights,
                    0,
                    weights.layers.len(),
                )
            }
        }
    }
}

/// A fixed synthetic token-ID stream valid for any tiny fixture vocab (mod 100).
/// Mirrors `fixture_golden`'s generator so streams are comparable.
pub fn synthetic_tokens(len: usize, seed: u64) -> Vec<u32> {
    let mut st = seed ^ 0x5DEE_CE66_D8A1_0001u64;
    let mut next = || {
        st = st.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = st;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    (0..len).map(|_| (next() % 100) as u32).collect()
}

/// Numerically-stable log-softmax in place → returns log-probs.
fn log_softmax(logits: &[f32]) -> Vec<f32> {
    let m = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f64;
    for &v in logits {
        sum += ((v - m) as f64).exp();
    }
    let lse = m as f64 + sum.ln();
    logits.iter().map(|&v| (v as f64 - lse) as f32).collect()
}

/// Result of a [`run_kld`] pass.
pub struct KldOut {
    pub mean_kld: f64,
    pub max_kld: f64,
    pub n_scored: usize,
    pub finite: bool,
    pub first_nonfinite: Option<String>,
}

/// Result of an autoregressive tiny forward pass.
pub struct ArHashOut {
    pub logit_hash: u64,
    pub token_hash: u64,
    pub n_steps: usize,
    pub prompt_len: usize,
    pub last_token: u32,
}

fn hash_mix(mut h: u64, x: u64) -> u64 {
    h ^= x;
    h = h.wrapping_mul(0x1000_0000_01B3);
    h ^ (h >> 32)
}

fn argmax(logits: &[f32]) -> u32 {
    let mut best_i = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_i = i;
            best_v = v;
        }
    }
    best_i as u32
}

/// Free-running greedy decode over a tiny fixture, hashing the full logit vector
/// and the generated token stream. Unlike [`run_kld`], this grows KV/state using
/// the model's own argmax outputs after a short deterministic prompt, so it is a
/// cheap tripwire for position/state/KV/long-tail decode drift.
pub fn run_ar_hash(
    arch: TinyArch,
    model_path: &Path,
    gpu: &mut Gpu,
    len: usize,
    prompt_len: usize,
    seed: u64,
) -> Result<ArHashOut, String> {
    if len == 0 {
        return Err("ar_hash: --len must be > 0".into());
    }
    if prompt_len == 0 || prompt_len > len {
        return Err("ar_hash: --prompt-len must be in 1..=len".into());
    }

    let mut model = TinyModel::load(arch, model_path, gpu, len + 16)?;
    let vocab = model.vocab().max(1) as u32;
    let prompt = synthetic_tokens(prompt_len, seed);
    let mut next_token = prompt[0] % vocab;
    let mut logit_hash = 0xcbf2_9ce4_8422_2325u64;
    let mut token_hash = 0x9e37_79b9_7f4a_7c15u64;

    for pos in 0..len {
        let token = if pos < prompt_len {
            prompt[pos] % vocab
        } else {
            next_token
        };
        token_hash = hash_mix(token_hash, token as u64);
        let logits = model.forward_logits(gpu, token, pos)?;
        for &v in &logits {
            logit_hash = hash_mix(logit_hash, v.to_bits() as u64);
        }
        next_token = argmax(&logits) % vocab;
    }

    Ok(ArHashOut {
        logit_hash,
        token_hash,
        n_steps: len,
        prompt_len,
        last_token: next_token,
    })
}

/// Which KV cache the prefill probe builds. The choice decides which FA arm the
/// batched prefill takes, so it is the difference between covering
/// `prefill_chunk.rs`'s batched FA layer and covering its per-token fallback.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrefillKvMode {
    /// Q8_0 — `fa_kv_ok` holds, FA layers take the batched arm.
    Q8,
    /// KVarN — fails `fa_kv_ok`, FA layers take the per-token fallback. This is
    /// the default KV mode in production.
    Kvarn,
}

impl PrefillKvMode {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "q8" | "q8_0" => Ok(Self::Q8),
            "kvarn" => Ok(Self::Kvarn),
            other => Err(format!("prefill probe: unknown --kv mode {other:?} (q8|kvarn)")),
        }
    }
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Q8 => "q8",
            Self::Kvarn => "kvarn",
        }
    }
}

/// Result of a [`run_prefill_hash`] pass.
///
/// **Why this is differential, and why it is not a hash comparison.** Every
/// other tiny probe records a baseline and diffs against it, which is how a
/// prefill change the gate never executes still reports COVERED. This probe
/// carries its own in-tree oracle instead: the same tokens are driven through
/// `forward_prefill_batch` and through the per-token `forward_scratch` path —
/// the branch `qwen35_prefill.rs:150-166` already takes when
/// `replay_as_generated_suffix || hier_enabled`, and which
/// `qwen35_prefill_owned_session_serial_segment` takes unconditionally — and the
/// two must agree.
///
/// They cannot agree *bit-exactly*: the batched path runs GEMM kernels over `n`
/// rows and the reference runs GEMV per token, so float reassociation differs by
/// construction. Hashing the two and requiring equality is the parity-oracle
/// trap; this compares them numerically instead, on the two measures that
/// separate reassociation noise from a real state bug — KL divergence between
/// the post-prefill decode distributions, and whether the two paths pick the
/// same argmax at every decoded position.
pub struct PrefillHashOut {
    /// Max KL(ref || batched) over the decoded positions, nats.
    pub max_kld: f64,
    pub mean_kld: f64,
    /// Decoded positions where both paths chose the same argmax token.
    pub argmax_agree: usize,
    /// Largest absolute logit difference seen, any position.
    pub max_abs_diff: f32,
    /// Byte hash of the DeltaNet recurrent state after each path's prefill.
    /// Never compared for EQUALITY as a correctness signal — the two paths
    /// reassociate differently, so they are expected to differ. They are
    /// compared for INEQUALITY instead: see [`PrefillHashOut::distinct_paths`].
    pub batched_state_hash: u64,
    pub ref_state_hash: u64,
    pub n_prefill: usize,
    pub n_decode: usize,
}

impl PrefillHashOut {
    /// True when the batched prefill left the model in the same state as the
    /// per-token reference, to within `max_kld_tol` and with every decoded
    /// argmax matching.
    pub fn agrees(&self, max_kld_tol: f64) -> bool {
        self.max_kld <= max_kld_tol && self.argmax_agree == self.n_decode
    }

    /// True when the two runs actually took different code paths.
    ///
    /// **A false here invalidates the cell — it is not a pass.** The batched and
    /// per-token paths write bit-different recurrent state (GEMM vs GEMV
    /// reassociation), so equal state hashes mean the "batched" run silently
    /// fell back to the reference and the probe compared it against itself.
    ///
    /// Measured, not assumed: `HIPFIRE_PREFILL_BATCHED=0` collapses the qwen3_5
    /// cell's hashes to equal and its max KLD to exactly 0.0, which is the
    /// fallback's signature. Both tiny MoE fixtures show that signature with the
    /// flag UNSET — they never reach `forward_prefill_batch` at all — so without
    /// this check they would report a clean pass for a path they never ran,
    /// which is the exact false-coverage failure this milestone exists to end.
    pub fn distinct_paths(&self) -> bool {
        self.batched_state_hash != self.ref_state_hash
    }
}

/// Hash a device buffer's full contents. Safe for the DeltaNet recurrent state
/// (`s_matrices`, `conv_states`): the scan writes every element, so there is no
/// uninitialized tail. **Not** safe for KV buffers — `Gpu::alloc_tensor` hands
/// out pool-recycled memory, so a KV ring's unwritten positions hold whatever
/// the last tenant left there. KV is covered by the decode comparison instead,
/// which reads all of it.
fn hash_tensor_bytes(gpu: &Gpu, t: &GpuTensor, mut h: u64) -> Result<u64, String> {
    let bytes = gpu
        .download_raw(t, t.byte_size())
        .map_err(|e| format!("prefill probe: download_raw: {e:?}"))?;
    for c in bytes.chunks(8) {
        let mut w = [0u8; 8];
        w[..c.len()].copy_from_slice(c);
        h = hash_mix(h, u64::from_le_bytes(w));
    }
    Ok(h)
}

/// Prefill `n_prefill` tokens (batched or per-token), then decode the rest of
/// `tokens` teacher-forced, returning the recurrent-state hash and the
/// per-position decode logits.
///
/// The decode is what covers KV: with causal attention, step `n_prefill` reads
/// every position the prefill wrote, so a prefill that gets position 0 wrong and
/// position n−1 right — the exact failure `smoke-generate-batch-prefill.sh` is
/// blind to, since it compares only the final-position logit — shows up here.
fn qwen35_prefill_then_decode(
    arch: TinyArch,
    path: &Path,
    gpu: &mut Gpu,
    tokens: &[u32],
    n_prefill: usize,
    batched: bool,
    corrupt_kv_prefix: bool,
    kv_mode: PrefillKvMode,
) -> Result<(u64, Vec<Vec<f32>>), String> {
    let mut model = TinyModel::load(arch, path, gpu, tokens.len() + 16)?;

    let state_hash = {
        let (config, weights, kv, dn, scratch) = match &mut model {
            TinyModel::Qwen35 {
                config,
                weights,
                kv,
                dn,
                scratch,
            } => (config, weights, kv, dn, scratch),
            _ => {
                return Err(
                    "prefill probe: only the qwen3_5 arch has a batched prefill to check".into(),
                )
            }
        };
        // `TinyModel::load` builds a Q8 KV, for which `fa_kv_ok` holds and the
        // FA layers take the BATCHED arm. KVarN is `quantized` but is none of
        // q8/asym{2,3,4}, so it fails `fa_kv_ok` (`prefill_chunk.rs:2313`) and
        // routes FA to the per-token fallback instead. KVarN is also the default
        // KV mode, so that fallback is live in production — and it is the arm
        // M2a1 rewrote, which nothing else here executes.
        //
        // ponytail: the replaced Q8 buffers leak (GpuTensor has no pool-returning
        // Drop). Bounded and fine for a probe process that exits immediately;
        // free them explicitly if this ever moves somewhere long-lived.
        if kv_mode == PrefillKvMode::Kvarn {
            *kv = KvCache::new_gpu_kvarn(
                gpu,
                config.n_layers,
                config.n_kv_heads,
                config.head_dim,
                tokens.len() + 16,
                4,
            )
            .map_err(|e| format!("prefill probe: kvarn kv: {e:?}"))?;
        }

        let vocab = config.vocab_size.max(1) as u32;
        let prompt: Vec<u32> = tokens[..n_prefill].iter().map(|t| t % vocab).collect();

        if batched {
            qwen35::forward_prefill_batch(
                gpu, weights, config, &prompt, 0, kv, dn, scratch, None, None, None, None,
            )
            .map_err(|e| format!("qwen35 forward_prefill_batch: {e:?}"))?;
        } else {
            for (pos, &t) in prompt.iter().enumerate() {
                qwen35::forward_scratch(gpu, weights, config, t, pos, kv, dn, scratch)
                    .map_err(|e| format!("qwen35 forward_scratch: {e:?}"))?;
            }
        }

        // Falsifiability injection (see `--corrupt-kv-pos0`): scribble over the
        // start of layer 0's K buffer, which is position 0's row under any
        // token-major layout. A probe that still agrees after this is not
        // reading the prefill's KV at all, and every stage gated on it would be
        // green by default.
        if corrupt_kv_prefix {
            // Reproduce the exact failure this milestone exists to catch: KV
            // correct at the final position and garbage everywhere before it.
            // That is what `smoke-generate-batch-prefill.sh` cannot see, because
            // it compares only the final-position logit — so a lowered prefill
            // with this bug passes it and then degrades on turn 2.
            //
            // Zeroing position 0 ALONE was tried first and rejected on
            // measurement: on a random-weight fixture attention is near-uniform,
            // so one row of 64 moves max KLD by only 2x-44x over the noise floor
            // and sometimes lands under any tolerance with headroom. A check
            // that passes or fails on the token seed is not a check.
            //
            // Every layer, not just layer 0: qwen3.5 interleaves LinearAttention
            // (DeltaNet, writes no KV) with FullAttention, so scribbling on
            // layer 0 alone is silently a no-op on a fixture whose first layer
            // is recurrent — which is how a falsifiability check becomes false
            // reassurance.
            //
            // The per-layer K buffer is token-major (`physical_cap x per-token
            // blocks`, see `KvCache::new_gpu_q8_capped`), so positions
            // `0..n_prefill-1` own the leading `(n_prefill-1) * numel /
            // physical_cap` elements.
            if kv.k_gpu.is_empty() {
                return Err("prefill probe: KV cache has no layers to corrupt".into());
            }
            let cap = kv.physical_cap.max(1);
            for k in kv.k_gpu.iter() {
                let per_pos = (k.numel() / cap).max(1);
                let span = (per_pos * n_prefill.saturating_sub(1)).min(k.numel());
                if span > 0 {
                    gpu.fill_f32(&k.sub_offset(0, span), 0.0)
                        .map_err(|e| format!("prefill probe: corrupt fill: {e:?}"))?;
                }
            }
            // KVarN keeps the not-yet-quantized trailing block in a separate f32
            // recent-window ring and only flushes into `k_gpu` once a full
            // GROUP=128 block accumulates. A 64-token prefill therefore lives
            // ENTIRELY in `k_window`, and scribbling on `k_gpu` alone is a no-op
            // — which is what the self-check caught: under `--kv kvarn` the
            // corrupted run still agreed, to eight decimals. Each window buffer
            // is `[GROUP x kv_dim]` f32, so position p owns `kv_dim` elements at
            // `p * kv_dim`.
            let kv_dim = kv.kv_dim.max(1);
            for w in kv.k_window.iter() {
                let span = (kv_dim * n_prefill.saturating_sub(1)).min(w.numel());
                if span > 0 {
                    gpu.fill_f32(&w.sub_offset(0, span), 0.0)
                        .map_err(|e| format!("prefill probe: corrupt window fill: {e:?}"))?;
                }
            }
        }

        let mut h = 0xcbf2_9ce4_8422_2325u64;
        for t in dn.s_matrices.iter().chain(dn.conv_states.iter()) {
            h = hash_tensor_bytes(gpu, t, h)?;
        }
        h
    };

    let vocab = model.vocab().max(1) as u32;
    let mut decoded = Vec::new();
    for pos in n_prefill..tokens.len() {
        decoded.push(model.forward_logits(gpu, tokens[pos] % vocab, pos)?);
    }

    // `TinyModel` has no Drop that returns device memory, and `run_prefill_hash`
    // loads a SECOND model right after this one. On a tiny fixture the leak is
    // invisible; on a real MoE it is fatal — the 35B-A3B's second load died at
    // `hipMalloc(1.03 MiB), free=15.1 MiB of total=43008 MiB`, which reads like
    // a model too big for the box and is actually the first copy still resident.
    // Freeing here is what lets the probe run against real artifacts at all,
    // which is the only oracle the MoE arms have.
    if let TinyModel::Qwen35 {
        weights,
        kv,
        dn,
        scratch,
        ..
    } = model
    {
        weights.free_gpu(gpu);
        kv.free_gpu(gpu);
        dn.free_gpu(gpu);
        scratch.free_gpu(gpu);
        // `free_gpu` returns buffers to the internal pool's free-list, which the
        // next load's `upload_raw` cannot see (`dispatch/mod.rs:2171` calls that
        // out as a monotonic leak). Without the drain the second load still
        // OOMs, just later — it died at layer 30 of 40 instead of 29. This is
        // what `load.rs:4121` does at unload.
        gpu.drain_pool();
    }

    Ok((state_hash, decoded))
}

/// M2a0's probe: does the batched prefill leave the same KV and DeltaNet state
/// as the per-token reference?
///
/// `corrupt_kv_prefix` deliberately damages the batched run's KV at every position
/// except the last — the "correct at n-1, garbage at 0..n-2" failure this
/// milestone exists to catch, position 0 included — so the probe's own
/// falsifiability can be demonstrated. With it set, `agrees()` MUST be false.
/// That is the exit criterion for this stage.
pub fn run_prefill_hash(
    arch: TinyArch,
    model_path: &Path,
    gpu: &mut Gpu,
    n_prefill: usize,
    n_decode: usize,
    seed: u64,
    corrupt_kv_prefix: bool,
    kv_mode: PrefillKvMode,
) -> Result<PrefillHashOut, String> {
    if n_prefill < 2 {
        return Err("prefill probe: --prefill must be >= 2".into());
    }
    if n_decode == 0 {
        return Err("prefill probe: --decode must be > 0 (it is what covers KV)".into());
    }
    let tokens = synthetic_tokens(n_prefill + n_decode, seed);

    let (batched_state_hash, batched) = qwen35_prefill_then_decode(
        arch,
        model_path,
        gpu,
        &tokens,
        n_prefill,
        true,
        corrupt_kv_prefix,
        kv_mode,
    )?;
    let (ref_state_hash, reference) = qwen35_prefill_then_decode(
        arch, model_path, gpu, &tokens, n_prefill, false, false, kv_mode,
    )?;

    if batched.len() != reference.len() {
        return Err(format!(
            "prefill probe: decoded {} positions batched vs {} reference",
            batched.len(),
            reference.len()
        ));
    }

    let mut max_kld = 0.0f64;
    let mut sum_kld = 0.0f64;
    let mut argmax_agree = 0usize;
    let mut max_abs_diff = 0.0f32;
    for (b, r) in batched.iter().zip(reference.iter()) {
        let lb = log_softmax(b);
        let lr = log_softmax(r);
        let mut kld = 0.0f64;
        for (&pb, &pr) in lb.iter().zip(lr.iter()) {
            // KL(ref || batched): the reference is the trusted distribution.
            kld += (pr as f64).exp() * (pr as f64 - pb as f64);
        }
        let kld = kld.max(0.0);
        sum_kld += kld;
        if kld > max_kld {
            max_kld = kld;
        }
        if argmax(b) == argmax(r) {
            argmax_agree += 1;
        }
        for (&vb, &vr) in b.iter().zip(r.iter()) {
            let d = (vb - vr).abs();
            if d > max_abs_diff {
                max_abs_diff = d;
            }
        }
    }

    Ok(PrefillHashOut {
        max_kld,
        mean_kld: sum_kld / batched.len().max(1) as f64,
        argmax_agree,
        max_abs_diff,
        batched_state_hash,
        ref_state_hash,
        n_prefill,
        n_decode,
    })
}

/// Run `model` over `tokens`, returning per-position logits for pos >= warmup.
fn run_logits(
    arch: TinyArch,
    path: &Path,
    gpu: &mut Gpu,
    tokens: &[u32],
    warmup: usize,
) -> Result<Vec<Vec<f32>>, String> {
    let mut model = TinyModel::load(arch, path, gpu, tokens.len() + 16)?;
    let mut out = Vec::new();
    for (pos, &tok) in tokens.iter().enumerate() {
        let lg = model.forward_logits(gpu, tok, pos)?;
        if pos >= warmup {
            out.push(lg);
        }
    }
    Ok(out)
}

/// KL(ref || cand) averaged over positions, feeding both models the identical
/// fixed synthetic stream (teacher-forced — inputs never depend on output, so
/// the two runs are independent and comparable position-by-position).
pub fn run_kld(
    arch: TinyArch,
    ref_path: &Path,
    cand_path: &Path,
    gpu: &mut Gpu,
    len: usize,
    warmup: usize,
    seed: u64,
) -> Result<KldOut, String> {
    let tokens = synthetic_tokens(len, seed);
    let refs = run_logits(arch, ref_path, gpu, &tokens, warmup)?;
    let cands = run_logits(arch, cand_path, gpu, &tokens, warmup)?;
    if refs.len() != cands.len() || refs.is_empty() {
        return Err(format!(
            "kld: position mismatch ref={} cand={}",
            refs.len(),
            cands.len()
        ));
    }
    let mut sum = 0.0f64;
    let mut max = 0.0f64;
    let mut finite = true;
    let mut first_nonfinite = None;
    for (pos, (rp, qp)) in refs.iter().zip(cands.iter()).enumerate() {
        if first_nonfinite.is_none() {
            if let Some((i, v)) = rp.iter().enumerate().find(|(_, v)| !v.is_finite()) {
                first_nonfinite = Some(format!("ref position {pos} logit {i} = {v}"));
            } else if let Some((i, v)) = qp.iter().enumerate().find(|(_, v)| !v.is_finite()) {
                first_nonfinite = Some(format!("cand position {pos} logit {i} = {v}"));
            }
        }
        let lr = log_softmax(rp);
        let lq = log_softmax(qp);
        // KL = Σ p·(log p − log q), p = exp(lr).
        let mut kl = 0.0f64;
        for (a, b) in lr.iter().zip(lq.iter()) {
            let p = (*a as f64).exp();
            if p > 0.0 {
                kl += p * (*a as f64 - *b as f64);
            }
        }
        if !kl.is_finite() {
            finite = false;
        }
        sum += kl;
        if kl > max {
            max = kl;
        }
    }
    let n = refs.len();
    Ok(KldOut {
        mean_kld: sum / n as f64,
        max_kld: max,
        n_scored: n,
        finite,
        first_nonfinite,
    })
}

/// Result of a [`run_collect`] pass.
pub struct CollectOut {
    pub n_tensors: usize,
    pub consistency: f32,
    pub out_path: String,
}

/// Arm the model-agnostic [`CalibCollector`], run the bf16 forward over the
/// synthetic stream (capturing per-linear input activations at the shared
/// `weight_gemv` chokepoint), and drain a `<name>.hessian`/`.imatrix`
/// `.calib.hfq` (HFQM) the quantizer can consume via `HIPFIRE_QTIP_HESSIAN`.
pub fn run_collect(
    arch: TinyArch,
    model_path: &Path,
    out_path: &Path,
    gpu: &mut Gpu,
    len: usize,
    seed: u64,
) -> Result<CollectOut, String> {
    let tokens = synthetic_tokens(len, seed);
    let mut model = TinyModel::load(arch, model_path, gpu, tokens.len() + 16)?;

    if let TinyModel::Zaya { model } = &model {
        let opts = zaya_calib::CalibOpts {
            kldref: false,
            kldref_topk: 64,
        };
        let provenance = vec![
            ("source", serde_json::json!("tiny_quant_probe")),
            ("arch", serde_json::json!(arch.as_str())),
            ("n_calib_tokens", serde_json::json!(tokens.len())),
        ];
        let summary = zaya_calib::collect_calibration_artifacts(
            gpu,
            model.weights(),
            model.config(),
            &tokens,
            &opts,
            out_path,
            &provenance,
        )?;
        return Ok(CollectOut {
            n_tensors: summary.n_hessian + summary.n_imatrix,
            consistency: summary.max_consistency,
            out_path: out_path.display().to_string(),
        });
    }

    // Routed experts (if any) would be imatrix-only, but we don't name them, so
    // a plain collector suffices — only the named dense linears are captured.
    let collector = Arc::new(CalibCollector::new());
    gpu.capture_names = model.capture_names();
    gpu.active_capture = Some(collector.clone());

    // Run the capturing forward, then ALWAYS disarm before propagating — leaving
    // `active_capture` armed would silently capture into this stale collector on
    // any later forward through the shared `&mut Gpu`.
    let fwd = (|| {
        for (pos, &tok) in tokens.iter().enumerate() {
            model.forward_logits(gpu, tok, pos)?;
        }
        Ok::<(), String>(())
    })();
    gpu.active_capture = None;
    gpu.capture_names = HashMap::new();
    fwd?;

    let n_tensors = collector.len();
    if n_tensors == 0 {
        return Err(
            "collect: no tensors captured (capture_names empty or weight_gemv not hit)".into(),
        );
    }
    let meta = serde_json::json!({
        "artifact_kind": "calibration",
        "source": "tiny_quant_probe",
        "arch": arch.as_str(),
        "n_calib_tokens": tokens.len(),
    })
    .to_string();
    let consistency = collector
        .write_streaming(gpu, out_path, arch.arch_id(), &meta, &[])
        .map_err(|e| format!("collect: write {out_path:?}: {e}"))?;
    Ok(CollectOut {
        n_tensors,
        consistency,
        out_path: out_path.display().to_string(),
    })
}
