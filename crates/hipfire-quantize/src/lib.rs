// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! Library surface for the hipfire quantizer.
//!
//! Historically this crate was binary-only (`main.rs` owned every module). The
//! pure quantization codecs, the LDLQ/GPTQ calibration machinery, and the
//! Hessian sidecar I/O are useful to other crates — notably `hipfire-diffusion`,
//! which reuses the oq4/oq8 packers and decoders for activation-calibrated
//! diffusion weight quantization. Those modules now live here and the
//! `hipfire-quantize` binary (`main.rs`) consumes this same library via
//! `use hipfire_quantize::…`.
//!
//! Crate-root helpers (`cpu_fwht_256`, `gen_fwht_signs`, `f16_to_f32`,
//! `f32_to_f16`) are re-exported from `hipfire-primitives` so the in-crate
//! `crate::{…}` references inside the modules below keep resolving unchanged.

// `cli` and `tools` were separate crate roots (src/main.rs, src/bin/*.rs) and
// still spell their imports `hipfire_quantize::…`. A crate cannot refer to
// itself by name in Rust 2018+ without this alias, so it is what lets those
// modules move in here untouched.
extern crate self as hipfire_quantize;

use std::sync::OnceLock;

pub use hipfire_kvquant::{kv_compact, kvarn};
pub use hipfire_primitives::conv::{f16_to_f32, f32_to_f16};
pub use hipfire_primitives::fwht::{cpu_fwht_256, gen_fwht_signs};

// Force-link the offline arch `-spec` bundle so every family's `register_arch!`
// (Ingest quant-policy + ToyModel fixture) survives rlib pruning. The `fixture` and
// quant-policy modules build their `ArchRegistry` from these registrations. This
// MUST live in the lib (not only `main.rs`) so lib tests see the registrations too.
use hipfire_arch_specs as _;

pub mod codecs;
pub mod gguf_import;
pub mod gptq;
pub mod hessian_io;
/// `.hfa` archive reading moved to `hipfire-quant-format` so BOTH the writer
/// (this crate) and the readers (`hipfire-runtime`) can see it. Re-exported so
/// existing `hipfire_quantize::hfa::...` paths keep resolving.
pub use hipfire_quant_format::hfa;
#[allow(dead_code)]
pub mod hfhs_diag;
pub mod hfq_out;
#[allow(dead_code)]
pub mod ldlq;
pub mod mixed_precision;
/// Opus mixed-precision low-bit weight codec (unsigned codes + offset fold).
pub mod opus_lowbit;
pub mod quant_plan;
// QTIP encoder core: some helpers are not yet wired into the dispatch.
pub mod fixture;
#[allow(dead_code)]
pub mod qtip;
/// Weight-rotation helpers. Lived in `main.rs` when that was a crate root; now
/// in the lib so both `cli` and any other consumer can reach it.
pub mod rotate;
pub mod roughquant;

/// The `hipfire-quantize` command line, as a module rather than a crate root.
pub mod cli;
/// The former standalone conversion binaries.
pub mod tools;

// Process-global toggle for the `mqN+` clip-search codec variant. Lives in the
// library so the codecs (which read it via `crate::mq_clipsearch_enabled`) and
// the binary (which arms it from a CLI flag via `set_mq_clipsearch`) share one
// source of truth.
static MQ_CLIPSEARCH: OnceLock<bool> = OnceLock::new();

/// Whether the `mqN+` clip-search variant is active for MQ codecs.
pub fn mq_clipsearch_enabled() -> bool {
    MQ_CLIPSEARCH.get().copied().unwrap_or(false)
}

/// Arm the `mqN+` clip-search variant (idempotent; first set wins).
pub fn set_mq_clipsearch(enabled: bool) {
    let _ = MQ_CLIPSEARCH.set(enabled);
}
