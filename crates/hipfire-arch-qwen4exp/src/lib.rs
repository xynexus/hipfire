// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! Qwen3.8-Flash-Next (`qwen4_exp`) serving architecture, arch id 26.
//!
//! The offline half — identity, quant policy, fixtures — lives in the sibling
//! `hipfire-arch-qwen4exp-spec`, which the quantizer links without the GPU stack.
//! This crate is the serving half. Config parsing is in place; weights and the
//! forward pass are not yet.
//!
//! See `docs/plans/2026-08-29-qwen4exp-flash-next-scope.md` for the architecture
//! and the staged plan, and `third_party/transformers-qwen4_exp/` for the reference.

pub mod arch;
pub mod attn;
pub mod attn_gpu;
pub mod config;
pub mod gdn;
pub mod gdn_cpu;
pub mod hc;
pub mod hc_gpu;
pub mod moe;
pub mod moe_gpu;
pub mod mtp;
pub mod ngram;
pub mod ngram_store;
pub mod ple;
pub mod ple_gpu;
pub mod qsa;
pub mod rope;
pub mod trunk;
pub mod trunk_gpu;
pub mod vision;
pub mod vision_gpu;
pub mod weights;

pub use config::{
    DeltaNetConfig, GatedResidualConfig, IndexerConfig, LayerType, MoeConfig, NgramConfig,
    Qwen4ExpConfig,
};
pub use gdn::{gdn_decode_step, GdnScratch, GdnState, GdnWeights};
pub use hc::{gated_rmsnorm_sigmoid, grouped_rmsnorm, GatedResidual, Read as GatedResidualRead};
pub use hipfire_arch_qwen4exp_spec::{
    ngram_head_layout, ngram_head_layout_at, nth_prime_after, QWEN4EXP_ARCH_ID,
};
pub use moe::{Expert, MoeLayer, Routing};
pub use ngram::{build_layer_multipliers, NgramHasher, RowLocation};
pub use ngram_store::{pack, Codec, NgramStore, BLOCK};
pub use ple::dilated_conv_silu_step;
pub use qsa::{pool_block, score_block, select as qsa_select, topk_by_threshold, QsaParams};
pub use weights::{plan, Expect, Mismatch, Plan, TEXT_PREFIX};
