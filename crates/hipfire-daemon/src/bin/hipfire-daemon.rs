// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Standalone `hipfire-daemon` executable.
//!
//! The daemon itself lives in the crate's library so `hipfire daemon` can call
//! the same entry point in-tree. This shim exists so the separate binary keeps
//! working for anything that still spawns it by name.

fn main() {
    hipfire_daemon::main();
}
