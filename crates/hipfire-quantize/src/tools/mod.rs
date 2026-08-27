// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! Artefact-conversion tools that used to be their own binaries.
//!
//! Each was a standalone `src/bin/*.rs` with its own `fn main`. They are
//! modules now so `hipfire` can call them directly; the thin shims under
//! `src/bin/` keep the separate executables building.

pub mod dflash_convert;
pub mod draft_to_mq4;
pub mod dspark_convert;
pub mod mq4_merge_mtp;
pub mod mtp_extract;

use std::sync::OnceLock;

/// argv these tools parse, which is NOT always the process's own.
///
/// Each tool used to be its own executable and reads argv positionally —
/// rejecting anything it does not recognise. Called as `hipfire convert
/// mtp-extract …` the real argv carries two leading tokens the tool has never
/// heard of, and it exits with `unknown arg: convert`. So `hipfire` installs
/// the argv it wants the tool to see, and the standalone binaries fall through
/// to the process's own.
///
/// ponytail: process-global, set once before a one-shot tool runs. Matches the
/// existing `MQ_CLIPSEARCH` plumbing in this crate. If tools ever need to run
/// concurrently in one process, thread argv through their entry points instead.
static TOOL_ARGV: OnceLock<Vec<String>> = OnceLock::new();

/// The argv the current tool should parse.
pub fn argv() -> Vec<String> {
    TOOL_ARGV
        .get()
        .cloned()
        .unwrap_or_else(|| std::env::args().collect())
}

/// Install the argv a tool will parse. First set wins; index 0 is the tool name.
pub fn set_argv(args: Vec<String>) {
    let _ = TOOL_ARGV.set(args);
}
