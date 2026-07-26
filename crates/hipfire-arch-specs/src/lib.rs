// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! # hipfire-arch-specs — the offline arch aggregation point
//!
//! Force-links every arch `-spec` crate so their `register_arch!` `Ingest`
//! registrations survive rlib pruning. The quantizer links THIS crate (all deps are
//! lean, CPU-only `-spec` crates) and builds its `ArchRegistry` from the
//! registrations, so a migrated family's quant policy is available offline without
//! the GPU/serving stack.
//!
//! Adding a migrated family = add its `-spec` crate to this crate's `Cargo.toml`
//! and one `use … as _;` line below. No quantizer edit.
//!
//! One `-spec` per UNIQUE `arch_id`. Variants that reuse a base id do NOT get their
//! own `-spec` — e.g. `hipfire-arch-qwen35-vl` is "Qwen3.5 dense + ViT" and ships
//! under `arch_id` 5/6, so it is already covered by `hipfire-arch-qwen35-spec`. A
//! `qwen35-vl-spec` would register `ArchId(5)` a second time and collide in the
//! registry (which merges by `ArchId`). Its absence here is intentional, not a gap.

#![allow(unused_imports)]

use hipfire_arch_cohere2_spec as _;
use hipfire_arch_deepseek4_spec as _;
use hipfire_arch_dots_ocr_spec as _;
use hipfire_arch_embeddinggemma_spec as _;
use hipfire_arch_flux2_spec as _;
use hipfire_arch_gemma3_spec as _;
use hipfire_arch_gemma3_vl_spec as _;
use hipfire_arch_gemma4_spec as _;
use hipfire_arch_krea2_spec as _;
use hipfire_arch_lfm2moe_spec as _;
use hipfire_arch_llama_spec as _;
use hipfire_arch_minimax_spec as _;
use hipfire_arch_nemotron_spec as _;
use hipfire_arch_qwen2_spec as _;
use hipfire_arch_qwen35_spec as _;
use hipfire_arch_qwenimage_spec as _;
use hipfire_arch_zaya_spec as _;

#[cfg(test)]
mod tests {
    use hipfire_arch_api::{ArchId, ArchRegistry, ExpertLayout, Role};

    /// The body of one `## Heading` section, up to the next `## `.
    ///
    /// Checks are scoped per-section on purpose: a whole-file substring search
    /// passes when a family is missing from the Families table but still
    /// mentioned in the legacy detection table, which is exactly the drift this
    /// test exists to catch.
    fn section<'a>(doc: &'a str, heading: &str) -> &'a str {
        let start = doc
            .find(heading)
            .unwrap_or_else(|| panic!("docs/architecture-ids.md has no `{heading}` section"));
        let rest = &doc[start + heading.len()..];
        match rest.find("\n## ") {
            Some(end) => &rest[..end],
            None => rest,
        }
    }

    /// `docs/architecture-ids.md` is the human-readable rendering of the frozen
    /// identity vocabulary. Nothing breaks when it drifts, which is exactly why
    /// it drifts — the file previously advertised a four-step onboarding
    /// checklist for a change that touches seventy files. Assert the registry
    /// and the doc agree, so adding an arch without documenting it fails here
    /// rather than misleading the next reader.
    #[test]
    fn identity_vocabulary_matches_the_docs() {
        let doc = std::fs::read_to_string(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs/architecture-ids.md"),
        )
        .expect("docs/architecture-ids.md");

        let roles = section(&doc, "\n## Roles\n");
        for role in Role::ALL {
            assert!(
                roles.contains(&format!("`{}`", role.as_str())),
                "role `{}` is in the frozen vocabulary but absent from the \
                 Roles table in docs/architecture-ids.md",
                role.as_str(),
            );
        }

        let families = section(&doc, "\n## Families\n");
        let variants = section(&doc, "\n## Variants\n");
        for arch in ArchRegistry::build().iter() {
            let family = arch.base.family();
            assert!(
                families.contains(&format!("`{family}`")),
                "family `{family}` ({}) is registered but absent from the \
                 Families table in docs/architecture-ids.md",
                arch.id,
            );
            for variant in arch.base.variants() {
                assert!(
                    variants.contains(&format!("`{family}/{variant}`")),
                    "variant `{family}/{variant}` is declared but absent from \
                     the Variants table in docs/architecture-ids.md",
                );
            }
        }
    }

    #[test]
    fn all_specs_present_and_declare_ingest() {
        let reg = ArchRegistry::build();
        // Every migrated arch id must be reachable with an Ingest policy through the
        // bundle (the path the quantizer uses).
        for id in [
            0u16, 1, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 23, 24, 25,
        ] {
            let a = reg
                .get(ArchId(id))
                .unwrap_or_else(|| panic!("arch id {id} not registered in the specs bundle"));
            assert!(
                a.caps.ingest.is_some(),
                "arch id {id} ({}) declares no Ingest",
                a.family
            );
        }
    }

    #[test]
    fn gemma4_spec_is_force_linked_with_identity_ingest_and_named_toys() {
        let reg = ArchRegistry::build();
        let arch = reg.get(ArchId(24)).expect("Gemma 4 arch id 24");
        assert_eq!(arch.family, "gemma4");
        assert_eq!(
            arch.caps.ingest.expect("Gemma 4 ingest").expert_layout(),
            ExpertLayout::StackedGateUpDown
        );
        let toy = arch.caps.toy_model.expect("Gemma 4 named toy models");
        assert_eq!(toy.fixture_names(), &["dense", "ple-sharing", "dense-moe"]);
        for model_type in [
            "gemma4",
            "gemma4_text",
            "gemma4_unified",
            "gemma4_unified_text",
        ] {
            assert_eq!(
                reg.find_by_model_type(model_type)
                    .unwrap_or_else(|| panic!("missing model type {model_type}"))
                    .id,
                ArchId(24)
            );
        }
    }
}
