//! Library facade for `hipfire-quantize`.
//!
//! The crate started as a binary-only quantizer (`main.rs` — invoked as
//! `quantize`) plus a handful of private modules for GGUF input, GPTQ
//! Cholesky, and the HFHS Hessian sidecar reader. The Tier 1 BF16
//! calibration pipeline introduces *writers* for the same artifacts, and
//! those writers must be reachable from `hipfire-runtime`'s
//! `collect_imatrix` and `collect_hessian` binaries.
//!
//! This `lib.rs` exposes the shared modules to other crates in the
//! workspace. The existing `main.rs` is unaffected — Cargo compiles it as a
//! separate crate root that continues to declare its private `mod`s for
//! the quantizer's own use.

pub mod awq_compute;
pub mod awq_scales_io;
pub mod gguf_imatrix_writer;
pub mod gguf_input;
pub mod hessian_io;
pub mod hfhs_writer;
pub mod imq_reader;
pub mod imq_writer;
