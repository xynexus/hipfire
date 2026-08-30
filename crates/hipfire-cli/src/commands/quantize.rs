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
#[command(
    after_help = "Also reachable here, forwarded verbatim to the offline conversion tools:\n          artifact   inspect | audit-calibration | compare-calibration | moe-router-profile\n          import     gguf | safetensors\n          export     safetensors\n          repack     <hf_dir> <-> <archive.hfa>, or --check   (NOT `optimize`, which is a layout pass)\n          lora       export | merge | convert\n          calibrate  activation capture -> .calib.hfq\n          two-pass   |  induct  |  npu pair-hfp\n\n        These arrive as an external subcommand, so clap cannot enumerate them and\n        `gen-docs` cannot render them -- this list is the only place they appear.\n        Promote one to a real subcommand (as `hipfire download` now is) and it\n        documents itself."
)]
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
    /// Everything hipfire-coexistence dispatches: calibrate, artifact, import,
    /// export, repack, lora, hub, download, npu, induct, two-pass.
    ///
    /// Captured as an external subcommand rather than enumerated, because that
    /// crate routes on `args[0]`/`args[1]` itself and its flag bags reject any
    /// token they do not recognise — so the vector has to arrive verbatim.
    #[command(external_subcommand)]
    Coexistence(Vec<String>),
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
    if let ConvertCommand::Coexistence(argv) = &cmd {
        return hipfire_coexistence::cli::run(argv).map_err(|e| anyhow::anyhow!("{e}"));
    }
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
        ConvertCommand::Coexistence(_) => unreachable!("handled above"),
    };
    let mut argv = vec![name.to_string()];
    argv.extend(args.args.iter().map(|a| a.to_string_lossy().into_owned()));
    hipfire_quantize::tools::set_argv(argv);
    run();
    Ok(())
}
