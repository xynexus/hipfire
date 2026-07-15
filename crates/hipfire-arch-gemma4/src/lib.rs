// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! Gemma 4 text runtime, built phase-by-phase under the canonical support plan.
//!
//! Phase 2 contributes the family-owned core tensor layout and consumes the
//! shared runtime transformer loader. Config lowering, layer assembly, forward,
//! and serving registration land only at their later frozen gates.

pub mod arch;
pub mod config;
pub mod forward;
pub mod weights;

pub use arch::{generation_eos_ids_from_hfq, Gemma4, Gemma4Backend};

pub use config::{Gemma4Config, Gemma4LayerPlan};
pub use forward::{
    forward_step, forward_step_lowered, forward_step_reference, logits, lower_dense_forward,
    Gemma4DenseState, Gemma4ForwardCapture,
};
pub use weights::{
    load_core_weights, load_dense_weights, Gemma4CoreShape, Gemma4CoreWeights,
    Gemma4DenseLayerWeights, Gemma4DenseWeights,
};
