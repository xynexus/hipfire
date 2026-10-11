// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! Lean offline spec for the Idefics3 vision-language family (arch_id 27,
//! IBM Granite-Docling, SmolVLM): identity + the `Ingest` quant policy. The text
//! decoder is plain LLaMA, so the shared transformer prior applies; the quantizer
//! keeps `model.vision_model.*` / `model.connector.*` at source precision.

use hipfire_arch_api::{
    default_importance, default_requires, register_arch, transformer_role, Arch, ArchId, CapReq,
    Ingest, TensorRole, ARCH_ID_IDEFICS3,
};

pub const IDEFICS3_ARCH_ID: ArchId = ArchId(ARCH_ID_IDEFICS3 as u16);

pub struct Idefics3Spec;

impl Arch for Idefics3Spec {
    fn id(&self) -> ArchId {
        IDEFICS3_ARCH_ID
    }

    fn family(&self) -> &'static str {
        "idefics3"
    }

    fn model_types(&self) -> &'static [&'static str] {
        &["idefics3"]
    }
}

impl Ingest for Idefics3Spec {
    fn role(&self, tensor: &str) -> TensorRole {
        transformer_role(tensor)
    }

    fn importance(&self, tensor: &str) -> u8 {
        default_importance(self.role(tensor))
    }

    fn requires(&self, tensor: &str) -> CapReq {
        default_requires(self.role(tensor))
    }
}

static IDEFICS3_SPEC: Idefics3Spec = Idefics3Spec;
register_arch!(IDEFICS3_SPEC, Ingest);
