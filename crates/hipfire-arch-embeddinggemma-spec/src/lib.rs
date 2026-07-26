// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! Lean offline spec for **embeddinggemma** (arch_id 19): identity + the `Ingest`
//! quant-policy, no runtime/kernel deps (the quantizer links this without the GPU
//! stack). The serving crate (`hipfire-arch-embeddinggemma`) declares runtime
//! capabilities on the same [`ArchId`]; the registry merges them.
//!
//! embeddinggemma-300m is a **bidirectional** Gemma-3 encoder: llama-shaped tensor
//! names (plus per-head `q_norm`/`k_norm` and the 4 gemma norms/layer), followed by
//! the sentence-transformers post-processing — mean pooling and one or two `Dense`
//! projection heads (the Matryoshka bottleneck). It emits a pooled sentence vector,
//! not autoregressive logits, so there is **no `lm_head`** and the embedding table is
//! never used as an output projection.
//!
//! Like the gemma3 spec, `importance` is a STRUCTURAL PRIOR, not a tuned bit
//! assignment. The tiny, sensitive Dense heads sit directly on the output vector, so
//! they get the same high-precision floor as attention/norms.

use hipfire_arch_api::{register_arch, Arch, ArchId, CapReq, Ingest, TensorRole};

/// embeddinggemma family header id.
pub const EMBEDDINGGEMMA_ARCH_ID: ArchId = ArchId(19);

/// Lean identity marker for the embeddinggemma family's offline spec.
pub struct EmbeddingGemmaSpec;

impl Arch for EmbeddingGemmaSpec {
    fn id(&self) -> ArchId {
        EMBEDDINGGEMMA_ARCH_ID
    }
    fn family(&self) -> &'static str {
        "embeddinggemma"
    }
}

impl Ingest for EmbeddingGemmaSpec {
    fn role(&self, tensor: &str) -> TensorRole {
        if tensor.contains("embed_tokens") {
            TensorRole::Embed
        } else if tensor.contains("q_proj")
            || tensor.contains("k_proj")
            || tensor.contains("v_proj")
            || tensor.contains("o_proj")
        {
            TensorRole::AttnProj
        } else if tensor.contains("gate_proj")
            || tensor.contains("up_proj")
            || tensor.contains("down_proj")
        {
            TensorRole::Mlp
        } else if tensor.contains("norm") {
            // Includes gemma's per-head q_norm/k_norm and the layer RMSNorms.
            TensorRole::Norm
        } else {
            // Sentence-transformers `dense.*` projection heads (+ any pooling
            // params) land here — treated as high-precision Other below.
            TensorRole::Other
        }
    }

    fn importance(&self, tensor: &str) -> u8 {
        // Structural prior: protect the gather-indexed embedding table, attention,
        // the tiny norms, and the Dense projection heads (they sit on the final
        // output vector — quantizing them corrupts every embedding). Compress the
        // MLP bulk. Refined by the quantizer.
        match self.role(tensor) {
            TensorRole::Embed => 255,
            TensorRole::Norm => 255,
            TensorRole::AttnProj => 255,
            TensorRole::Mlp => 128,
            // `dense.*` heads: keep them near-lossless.
            _ => 255,
        }
    }

    fn requires(&self, tensor: &str) -> CapReq {
        match self.role(tensor) {
            TensorRole::Embed => CapReq::RANDOM_ACCESS,
            _ => CapReq::NONE,
        }
    }
}

static EMBEDDINGGEMMA_SPEC: EmbeddingGemmaSpec = EmbeddingGemmaSpec;
register_arch!(EMBEDDINGGEMMA_SPEC, Ingest);

#[cfg(test)]
mod tests {
    use super::*;
    use hipfire_arch_api::ArchRegistry;

    #[test]
    fn embeddinggemma_spec_registers_ingest() {
        let reg = ArchRegistry::build();
        let a = reg
            .get(EMBEDDINGGEMMA_ARCH_ID)
            .expect("embeddinggemma spec registered");
        assert_eq!(a.family, "embeddinggemma");
        let ing = a.caps.ingest.expect("Ingest declared");
        assert_eq!(
            ing.requires("model.embed_tokens.weight"),
            CapReq::RANDOM_ACCESS
        );
        // Attention and the Dense heads outrank the MLP bulk.
        assert!(
            ing.importance("model.layers.0.self_attn.q_proj.weight")
                > ing.importance("model.layers.0.mlp.up_proj.weight")
        );
        assert!(
            ing.importance("dense.0.weight") > ing.importance("model.layers.0.mlp.up_proj.weight")
        );
    }
}
