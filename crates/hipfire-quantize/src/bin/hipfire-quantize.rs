// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! Standalone `hipfire-quantize` executable.
//!
//! The command line lives in the crate library so `hipfire quantize` can call
//! the same entry point. This shim keeps the separate binary working.

fn main() {
    hipfire_quantize::cli::main();
}
