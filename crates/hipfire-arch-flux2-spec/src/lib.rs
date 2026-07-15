// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! Lean offline spec for the FLUX.2 MMDiT denoiser (arch_id 23).
//!
//! This covers the shared Klein/SeFi transformer family. Pipeline metadata owns
//! the SeFi dual-time distinction; the on-disk tensor role and precision policy
//! remains a property of the common FLUX.2 weight topology.

use hipfire_arch_api::{
    default_importance, default_precision_class, mmdit_role, register_arch, Arch, ArchId, CapReq,
    Diffusion, Ingest, PrecisionClass, TensorRole, ARCH_ID_FLUX2,
};

pub const FLUX2_ARCH_ID: ArchId = ArchId(ARCH_ID_FLUX2 as u16);

pub struct Flux2Spec;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Stack {
    Double,
    Single,
}

impl Flux2Spec {
    fn block(tensor: &str) -> Option<(Stack, usize)> {
        for (prefix, stack) in [
            ("single_transformer_blocks.", Stack::Single),
            ("transformer_blocks.", Stack::Double),
        ] {
            if let Some(rest) = tensor.split_once(prefix).map(|(_, rest)| rest) {
                return Some((stack, rest.split('.').next()?.parse().ok()?));
            }
        }
        None
    }

    fn is_target_boundary(tensor: &str) -> bool {
        // Targeted variants are Klein-4B (5/20) and SeFi-2B (4/16). Protect the
        // first and both possible terminal blocks; the small over-protection is
        // deliberate until artifact geometry can participate in allocation.
        matches!(
            Self::block(tensor),
            Some((Stack::Double, 0 | 3 | 4)) | Some((Stack::Single, 0 | 15 | 19))
        )
    }

    fn is_attention(tensor: &str) -> bool {
        tensor.contains(".attn.") || tensor.contains(".img_attn.") || tensor.contains(".txt_attn.")
    }

    fn is_residual_writer(tensor: &str) -> bool {
        tensor.ends_with(".attn.to_out.0.weight")
            || tensor.ends_with(".attn.to_add_out.weight")
            || tensor.ends_with(".attn.to_out.weight")
            || tensor.ends_with(".linear2.weight")
            || tensor.ends_with(".ff.linear_out.weight")
            || tensor.ends_with(".ff_context.linear_out.weight")
    }

    fn is_qk(tensor: &str) -> bool {
        tensor.ends_with(".attn.to_q.weight")
            || tensor.ends_with(".attn.to_k.weight")
            || tensor.ends_with(".attn.add_q_proj.weight")
            || tensor.ends_with(".attn.add_k_proj.weight")
    }

    fn is_v(tensor: &str) -> bool {
        tensor.ends_with(".attn.to_v.weight") || tensor.ends_with(".attn.add_v_proj.weight")
    }

    fn is_top_level_protected(tensor: &str) -> bool {
        [
            "x_embedder.weight",
            "context_embedder.weight",
            "proj_out.weight",
            "norm_out.linear.weight",
            "double_stream_modulation_img.linear.weight",
            "double_stream_modulation_txt.linear.weight",
            "single_stream_modulation.linear.weight",
            "time_guidance_embed.timestep_embedder.linear_1.weight",
            "time_guidance_embed.timestep_embedder.linear_2.weight",
            "dual_time_embed.semantic_embedder.linear_1.weight",
            "dual_time_embed.semantic_embedder.linear_2.weight",
            "dual_time_embed.texture_embedder.linear_1.weight",
            "dual_time_embed.texture_embedder.linear_2.weight",
        ]
        .iter()
        .any(|suffix| tensor.ends_with(suffix))
    }

    fn is_fused_single_qkv_mlp(tensor: &str) -> bool {
        tensor.ends_with(".attn.to_qkv_mlp_proj.weight")
    }

    fn importance(tensor: &str) -> u8 {
        if Self::is_top_level_protected(tensor) {
            255
        } else if Self::block(tensor).is_some()
            && Self::is_target_boundary(tensor)
            && (Self::is_attention(tensor) || Self::is_residual_writer(tensor))
        {
            255
        } else if Self::is_residual_writer(tensor) {
            254
        } else if Self::is_qk(tensor) || Self::is_fused_single_qkv_mlp(tensor) {
            253
        } else if Self::is_v(tensor) {
            252
        } else {
            default_importance(mmdit_role(tensor))
        }
    }
}

impl Arch for Flux2Spec {
    fn id(&self) -> ArchId {
        FLUX2_ARCH_ID
    }

    fn family(&self) -> &'static str {
        "flux2"
    }
}

impl Diffusion for Flux2Spec {
    fn denoiser_family(&self) -> &'static str {
        "flux2-mmdit"
    }
}

impl Ingest for Flux2Spec {
    fn role(&self, tensor: &str) -> TensorRole {
        mmdit_role(tensor)
    }

    fn importance(&self, tensor: &str) -> u8 {
        Self::importance(tensor)
    }

    fn requires(&self, _tensor: &str) -> CapReq {
        CapReq::NONE
    }

    fn precision_class(&self, tensor: &str) -> PrecisionClass {
        if Self::is_top_level_protected(tensor)
            || Self::is_attention(tensor)
            || Self::is_residual_writer(tensor)
        {
            PrecisionClass::High
        } else {
            default_precision_class(mmdit_role(tensor))
        }
    }
}

static FLUX2_SPEC: Flux2Spec = Flux2Spec;
register_arch!(FLUX2_SPEC, Diffusion, Ingest);

#[cfg(test)]
mod tests {
    use super::*;
    use hipfire_arch_api::ArchRegistry;

    #[test]
    fn registers_as_diffusion_arch() {
        let registry = ArchRegistry::build();
        assert!(registry.is_diffusion(FLUX2_ARCH_ID));
        assert_eq!(
            registry.diffusion_family(FLUX2_ARCH_ID),
            Some("flux2-mmdit")
        );
        assert!(registry.get(FLUX2_ARCH_ID).unwrap().caps.ingest.is_some());
    }

    #[test]
    fn protects_both_stack_boundaries_and_residual_writers() {
        for tensor in [
            "x_embedder.weight",
            "proj_out.weight",
            "double_stream_modulation_img.linear.weight",
            "dual_time_embed.texture_embedder.linear_2.weight",
        ] {
            assert_eq!(Flux2Spec::importance(tensor), 255, "{tensor}");
        }
        assert_eq!(
            Flux2Spec::importance("transformer_blocks.0.attn.to_q.weight"),
            255
        );
        assert_eq!(
            Flux2Spec::importance("single_transformer_blocks.19.attn.to_out.weight"),
            255
        );
        assert_eq!(
            Flux2Spec::importance("single_transformer_blocks.7.attn.to_out.weight"),
            254
        );
        assert_eq!(
            Flux2Spec::importance("transformer_blocks.2.attn.to_q.weight"),
            253
        );
        assert_eq!(
            Flux2Spec::importance("transformer_blocks.2.attn.to_v.weight"),
            252
        );
        assert_eq!(
            Flux2Spec::importance("single_transformer_blocks.7.attn.to_qkv_mlp_proj.weight"),
            253
        );
        assert_eq!(
            Flux2Spec::importance("transformer_blocks.2.ff.linear_in.weight"),
            128
        );
        assert_eq!(
            Flux2Spec.precision_class("proj_out.weight"),
            PrecisionClass::High
        );
        assert_eq!(
            Flux2Spec.precision_class("transformer_blocks.2.ff.linear_in.weight"),
            PrecisionClass::Compressed
        );
    }
}
