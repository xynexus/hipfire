// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! Lean offline spec for the Krea2 diffusion denoiser (MMDiT, arch_id 17):
//! identity + the `Diffusion` modality marker (so routing tells it apart from
//! LLMs without a magic id) + the `Ingest` quant-policy (shared MMDiT role prior).
//! Deps only `hipfire-arch-api`.

use hipfire_arch_api::{
    default_importance, default_precision_class, mmdit_role, register_arch, Arch, ArchId, CapReq,
    Diffusion, Ingest, PrecisionClass, TensorRole, ARCH_ID_KREA2,
};

/// Krea2 denoiser header id.
pub const KREA2_ARCH_ID: ArchId = ArchId(ARCH_ID_KREA2 as u16);

/// Lean identity marker for the Krea2 offline spec.
pub struct Krea2Spec;

impl Krea2Spec {
    const TRANSFORMER_LAYERS: usize = 28;

    fn transformer_block_index(tensor: &str) -> Option<usize> {
        let rest = tensor.split_once("transformer_blocks.")?.1;
        rest.split('.').next()?.parse().ok()
    }

    fn is_boundary_block(tensor: &str) -> bool {
        Self::transformer_block_index(tensor)
            .is_some_and(|layer| layer == 0 || layer + 1 == Self::TRANSFORMER_LAYERS)
    }

    fn is_protected_resident(tensor: &str) -> bool {
        Self::transformer_block_index(tensor).is_some_and(|layer| layer < Self::TRANSFORMER_LAYERS)
            && (tensor.contains(".attn.") || tensor.ends_with(".ff.down.weight"))
    }

    /// Krea2-specific promotion priority for the resident DiT linears.
    ///
    /// The shared MMDiT role prior is intentionally broad. Mixed precision needs
    /// enough resolution to protect complete boundary transforms first, followed
    /// by the attention gate/output writers, QK logits, values, and finally the
    /// FF residual writer. FF expansion tensors remain the compressible bulk.
    fn tensor_importance(tensor: &str) -> u8 {
        let in_main_block = Self::transformer_block_index(tensor)
            .is_some_and(|layer| layer < Self::TRANSFORMER_LAYERS);
        if in_main_block
            && Self::is_boundary_block(tensor)
            && (tensor.contains(".attn.") || tensor.ends_with(".ff.down.weight"))
        {
            255
        } else if in_main_block
            && (tensor.ends_with(".attn.to_gate.weight")
                || tensor.ends_with(".attn.to_out.0.weight"))
        {
            254
        } else if in_main_block
            && (tensor.ends_with(".attn.to_q.weight") || tensor.ends_with(".attn.to_k.weight"))
        {
            253
        } else if in_main_block && tensor.ends_with(".attn.to_v.weight") {
            252
        } else if in_main_block && tensor.ends_with(".ff.down.weight") {
            251
        } else {
            default_importance(mmdit_role(tensor))
        }
    }
}

impl Arch for Krea2Spec {
    fn id(&self) -> ArchId {
        KREA2_ARCH_ID
    }
    fn family(&self) -> &'static str {
        "krea2"
    }
}

impl Diffusion for Krea2Spec {
    fn denoiser_family(&self) -> &'static str {
        "krea2-mmdit"
    }
}

impl Ingest for Krea2Spec {
    fn role(&self, tensor: &str) -> TensorRole {
        mmdit_role(tensor)
    }
    fn importance(&self, tensor: &str) -> u8 {
        Self::tensor_importance(tensor)
    }
    /// Diffusion has no gather-indexed tables. Rank-4 conv weights that need a
    /// random-access (ungrouped) codec are steered by shape at encode time, not
    /// by name here — so the name-only requirement is `NONE`.
    fn requires(&self, _tensor: &str) -> CapReq {
        CapReq::NONE
    }
    fn precision_class(&self, tensor: &str) -> PrecisionClass {
        if Self::is_protected_resident(tensor) {
            PrecisionClass::High
        } else {
            default_precision_class(mmdit_role(tensor))
        }
    }
}

static KREA2_SPEC: Krea2Spec = Krea2Spec;
register_arch!(KREA2_SPEC, Diffusion, Ingest);

#[cfg(test)]
mod tests {
    use super::*;
    use hipfire_arch_api::ArchRegistry;

    #[test]
    fn registers_as_diffusion_arch() {
        let reg = ArchRegistry::build();
        assert!(reg.is_diffusion(KREA2_ARCH_ID));
        assert_eq!(reg.diffusion_family(KREA2_ARCH_ID), Some("krea2-mmdit"));
        assert!(reg.get(KREA2_ARCH_ID).unwrap().caps.ingest.is_some());
    }

    #[test]
    fn mmdit_role_prior_protects_and_compresses() {
        // Non-block entry/output tensors stay maximally protected.
        assert_eq!(Krea2Spec.importance("final_layer.linear.weight"), 255);
        assert_eq!(Krea2Spec.importance("img_in.weight"), 255);

        // Boundary blocks are the highest-leverage residual transforms.
        assert_eq!(
            Krea2Spec.importance("transformer_blocks.0.attn.to_q.weight"),
            255
        );
        assert_eq!(
            Krea2Spec.importance("transformer_blocks.27.ff.down.weight"),
            255
        );

        // Interior attention is ranked by how directly it controls or writes the
        // residual stream; FF down is protected above the FF expansion bulk.
        assert_eq!(
            Krea2Spec.importance("transformer_blocks.3.attn.to_gate.weight"),
            254
        );
        assert_eq!(
            Krea2Spec.importance("transformer_blocks.3.attn.to_out.0.weight"),
            254
        );
        assert_eq!(
            Krea2Spec.importance("transformer_blocks.3.attn.to_q.weight"),
            253
        );
        assert_eq!(
            Krea2Spec.importance("transformer_blocks.3.attn.to_k.weight"),
            253
        );
        assert_eq!(
            Krea2Spec.importance("transformer_blocks.3.attn.to_v.weight"),
            252
        );
        assert_eq!(
            Krea2Spec.importance("transformer_blocks.3.ff.down.weight"),
            251
        );
        assert_eq!(
            Krea2Spec.precision_class("transformer_blocks.3.ff.down.weight"),
            hipfire_arch_api::PrecisionClass::High
        );

        // FF expansion/gating remains the compressible bulk.
        assert_eq!(
            Krea2Spec.importance("transformer_blocks.3.ff.up.weight"),
            128
        );
        assert_eq!(
            Krea2Spec.importance("transformer_blocks.3.ff.gate.weight"),
            128
        );

        // Modulation is copied at source precision by the diffusion packer, but
        // remains high-importance in the architecture contract.
        assert_eq!(
            Krea2Spec.importance("transformer_blocks.3.img_mod.linear.weight"),
            255
        );
        // Diffusion never needs random access.
        assert!(!Krea2Spec.requires("img_in.weight").random_access);
    }
}
