// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! `hipfire quantize` and `hipfire convert` — the former `hipfire-quantize`
//! binary and its five satellite executables, called in-process.

use anyhow::Result;
use clap::{Args, Subcommand};

#[derive(Debug, Args)]
#[command(
    // The quantizer scans `std::env::args()` by hand rather than using clap, so
    // flags must reach it untouched. Same passthrough shape as `hipfire eval`.
    disable_help_flag = true,
    trailing_var_arg = true,
    after_help = "Examples:\n  hipfire quantize --input model.hfa --output model--oq4.hfq --quant oq4\n  hipfire quantize --help\n"
)]
pub struct QuantizeArgs {
    /// Arguments forwarded verbatim to the quantizer.
    #[arg(allow_hyphen_values = true)]
    pub args: Vec<std::ffi::OsString>,
}

/// Run the quantizer in this process.
///
/// It reads `std::env::args()` itself, so the captured `args` are deliberately
/// unused — capturing them is only what stops clap claiming the flags first.
/// The quantizer exits the process when it finishes or fails, which is correct:
/// quantizing is the whole job of the invocation.
pub fn run(_args: QuantizeArgs) -> Result<()> {
    hipfire_quantize::cli::main();
    Ok(())
}

/// Artefact conversions that used to be five separate executables.
///
/// These live under `convert` rather than under `quantize` because clap cannot
/// disambiguate sibling subcommands from a `trailing_var_arg` passthrough — and
/// because grouping conversion tooling in one place is what AGENTS.md asks for.
/// Step 6 of the merge plan adds coexistence's offline half here too.
#[derive(Debug, Subcommand)]
pub enum ConvertCommand {
    /// Build a DFlash drafter sidecar from a model
    Dflash(ToolArgs),
    /// Build a DSpark drafter sidecar from a model
    Dspark(ToolArgs),
    /// Convert a drafter to mq4
    DraftMq4(ToolArgs),
    /// Extract an MTP head into its own artefact
    MtpExtract(ToolArgs),
    /// Merge an MTP head into an mq4 artefact
    MtpMerge(ToolArgs),
}

#[derive(Debug, Args)]
#[command(disable_help_flag = true, trailing_var_arg = true)]
pub struct ToolArgs {
    /// Arguments forwarded verbatim to the tool.
    #[arg(allow_hyphen_values = true)]
    pub args: Vec<std::ffi::OsString>,
}

pub fn run_convert(cmd: ConvertCommand) -> Result<()> {
    // These tools read argv positionally and reject tokens they do not know, so
    // they cannot simply see the process argv: `hipfire convert mtp-extract`
    // would reach the tool as `unknown arg: convert`. Hand each one the argv it
    // would have had as its own binary — its name, then the forwarded args.
    let (name, args, run): (&str, &ToolArgs, fn()) = match &cmd {
        ConvertCommand::Dflash(a) => (
            "dflash_convert",
            a,
            hipfire_quantize::tools::dflash_convert::main,
        ),
        ConvertCommand::Dspark(a) => (
            "dspark_convert",
            a,
            hipfire_quantize::tools::dspark_convert::main,
        ),
        ConvertCommand::DraftMq4(a) => (
            "draft_to_mq4",
            a,
            hipfire_quantize::tools::draft_to_mq4::main,
        ),
        ConvertCommand::MtpExtract(a) => {
            ("mtp_extract", a, hipfire_quantize::tools::mtp_extract::main)
        }
        ConvertCommand::MtpMerge(a) => (
            "mq4_merge_mtp",
            a,
            hipfire_quantize::tools::mq4_merge_mtp::main,
        ),
    };
    let mut argv = vec![name.to_string()];
    argv.extend(args.args.iter().map(|a| a.to_string_lossy().into_owned()));
    hipfire_quantize::tools::set_argv(argv);
    run();
    Ok(())
}
