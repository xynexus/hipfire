// SPDX-License-Identifier: Apache-2.0
//! Offline compatibility and model-induction orchestration.

// `cli` was a separate crate root (src/main.rs) and spells its 14 dispatch
// targets `hipfire_coexistence::…`. A crate cannot name itself in Rust 2018+
// without this alias, so it is what lets that file move in untouched.
extern crate self as hipfire_coexistence;

/// The `hipfire-coexistence` command line, as a module rather than a crate root.
pub mod cli;

/// How this process was invoked, for user-facing text.
///
/// The same code is reachable as the standalone `hipfire-coexistence` binary
/// and as `hipfire convert …`. Usage strings and — more importantly — the GPU
/// lock's holder line must name the command the user actually ran: AGENTS.md
/// calls out that a stale or wrong holder line makes a real contention error
/// point at the wrong thing.
pub fn invoked_as() -> &'static str {
    match std::env::current_exe()
        .ok()
        .as_deref()
        .and_then(|p| p.file_stem())
        .and_then(|s| s.to_str())
    {
        Some("hipfire") => "hipfire convert",
        _ => "hipfire-coexistence",
    }
}

pub mod artifact;
pub mod calibrate;
pub mod calibration_audit;
pub mod calibration_compare;
pub mod export_safetensors;
pub mod hub_archive;
pub mod import_safetensors;
pub mod induction;
pub mod repack;
pub mod residual_compare;
pub mod router_profile;
