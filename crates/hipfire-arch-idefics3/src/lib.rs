// SPDX-License-Identifier: Apache-2.0
// hipfire — Idefics3 (Granite-Docling, SmolVLM). See LICENSE / NOTICE.

//! Idefics3 vision-language models — `arch_id = 27`.
//!
//! image → Lanczos tiling into `max_image_size²` frames (row-major tiles + one
//! global view) → the **SigLIP** tower shared with gemma3-vl (tensors under
//! `model.vision_model.`) → **pixel shuffle** (`scale_factor²` neighbouring
//! patches concatenated) → `model.connector.modality_projection.proj` (no bias)
//! → rows spliced at the `<image>` placeholders of the plain **LLaMA** decoder
//! (`model.text_model.*`, resolved by the shared LLaMA loader).

pub mod arch;
pub mod config;
pub mod image;

pub use arch::{Idefics3Backend, IDEFICS3_SERVING_FACTORY};
pub use config::Idefics3Config;
