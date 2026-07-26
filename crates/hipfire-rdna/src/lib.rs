// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! hipfire-rdna: Kernel compilation, caching, and dispatch for RDNA GPUs.

pub mod arch_caps;
mod compiler;
mod dispatch;
pub mod feature_flags;
pub mod generic_warn;
pub mod gtt_share;
mod kernels;
pub mod pool;
pub mod profile;
pub mod profile_rocprof;
pub mod profiler;

pub use compiler::KernelCompiler;
pub use dispatch::{
    gen_fwht_signs, ActivationCapture, DType, Gpu, GpuTensor, OpusNpuIoLayout, OwnedTensor,
    LLOYD_MQ4_GROUP_BYTES, MMQ_CURRENT_LAYER,
};
pub use feature_flags::FeatureFlags;
pub use gtt_share::{ImportedTensor, SharedGttBuffer};
pub use kernels::GEMV_SRC;
// Re-export the result/error types of `Gpu`'s public methods so downstream
// crates (e.g. hipfire-train) can name them without depending on hip-bridge.
pub use hip_bridge::{HipError, HipResult};

/// Physical row tile used by the grouped-MoE scatter and GEMM kernels.
///
/// Routing planners and higher-level dispatch must use this value when they
/// allocate padded expert buckets and tile-id tables. Keep the kernel contract
/// in the lowest shared layer so model families cannot redeclare it differently.
pub const GROUPED_MOE_BLOCK_ROWS: usize = 16;
