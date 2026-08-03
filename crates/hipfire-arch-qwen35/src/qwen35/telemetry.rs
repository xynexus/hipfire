// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! MoE router selection histogram — re-exported from `hipfire-dispatch`.
//!
//! The implementation moved to [`hipfire_dispatch::moe_telemetry`] because two
//! MoE decode implementations must feed one histogram: this crate's
//! `moe_decode::moe_ffn_decode_impl`, and `hipfire_dispatch::pipeline`'s
//! executor, which ports that body verbatim. Arch crates depend on
//! `hipfire-dispatch` and not the reverse, so the dispatch crate is the lowest
//! one both paths can reach; keeping the state here left the dispatch path
//! unable to record, and its routing telemetry silently empty.
//!
//! Re-exported rather than moved wholesale so `qwen35::reset_moe_router_histogram`
//! and friends stay valid for existing callers (`loading.rs`, the daemon
//! evidence guard, `hipfire-runtime`'s `run` example).

pub use hipfire_dispatch::moe_telemetry::{
    moe_router_histogram_active, record_moe_router_selection, reset_moe_router_histogram,
    router_index_i32_to_usize, take_moe_router_histogram, MoeRouterHistogram,
    MoeRouterLayerHistogram,
};
