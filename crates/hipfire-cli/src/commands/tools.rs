// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! The small former binaries: monitor, atlas, steer, hfq.
//!
//! Each parses argv by hand and indexes it absolutely, so none can read the
//! process argv directly — `hipfire atlas read x` would select on "atlas".
//! Every one is handed the argv it would have had as its own binary.

use anyhow::Result;
use clap::Args;

/// Build the argv a folded tool expects: its own name, then the forwarded args.
fn argv_for(name: &str, args: &[std::ffi::OsString]) -> Vec<String> {
    let mut argv = vec![name.to_string()];
    argv.extend(args.iter().map(|a| a.to_string_lossy().into_owned()));
    argv
}

#[derive(Debug, Args)]
#[command(disable_help_flag = true, trailing_var_arg = true)]
pub struct PassthroughArgs {
    /// Arguments forwarded verbatim to the tool.
    #[arg(allow_hyphen_values = true)]
    pub args: Vec<std::ffi::OsString>,
}

/// `hipfire monitor` — the live TUI. Takes no arguments, and never did.
pub fn run_monitor() -> Result<()> {
    hipfire_monitor::run_standalone()
}

/// `hipfire atlas` — Kernel Atlas row inspection and rendering.
pub fn run_atlas(args: PassthroughArgs) -> Result<()> {
    hipfire_atlas::cli::main_with_args(&argv_for("hipfire-atlas", &args.args));
    Ok(())
}

/// `hipfire steer` — the steering-vector harness.
pub fn run_steer(args: PassthroughArgs) -> Result<()> {
    hipfire_steer_harness::cli::main_with_args(&argv_for("hipfire-steer", &args.args))
        .map_err(|e| anyhow::anyhow!("{e}"))
}

/// `hipfire hneurons-probe` — the harmful-neuron probe.
pub fn run_hneurons_probe(args: PassthroughArgs) -> Result<()> {
    hipfire_steer_harness::hneurons_probe::main_with_args(&argv_for(
        "hipfire-hneurons-probe",
        &args.args,
    ));
    Ok(())
}

/// `hipfire hfq` — inspect/verify/extract inside a `.hfq` artefact.
pub fn run_hfq(args: PassthroughArgs) -> Result<()> {
    hipfire_runtime::hfq_cli::main_with_args(&argv_for("hfq", &args.args));
    Ok(())
}
