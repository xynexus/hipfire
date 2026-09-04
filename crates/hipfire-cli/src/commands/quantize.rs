// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! `hipfire quantize` and `hipfire convert` — the former `hipfire-quantize`
//! binary and its five satellite executables, called in-process.

use anyhow::{Context, Result};
use clap::{Args, Subcommand};

#[derive(Debug, Args)]
#[command(
    // The quantizer scans `std::env::args()` by hand rather than using clap, so
    // flags must reach it untouched. Same passthrough shape as `hipfire eval`.
    disable_help_flag = true,
    trailing_var_arg = true,
    after_help = "Examples:\n  hipfire quantize --input model.hfa --output model--oq4.hfq --quant oq4\n  hipfire quantize --detach --input model.hfa --output model--oq4.hfq --quant oq4\n  hipfire quantize --help\n"
)]
pub struct QuantizeArgs {
    /// Arguments forwarded verbatim to the quantizer.
    ///
    /// `--detach` is the one flag read here rather than forwarded: it queues the
    /// run as a service job instead of doing it now.
    #[arg(allow_hyphen_values = true)]
    pub args: Vec<std::ffi::OsString>,
}

/// Run the quantizer in this process, or hand it to the service.
///
/// In the foreground it reads `std::env::args()` itself, so the captured `args`
/// are deliberately unused — capturing them is only what stops clap claiming the
/// flags first. The quantizer exits the process when it finishes or fails, which
/// is correct: quantizing is the whole job of the invocation.
///
/// With `--detach` the args ARE used: they become the job spec the server's
/// deferred runner replays, which is what gets a build queued behind serving
/// instead of racing it for `hip-gpu-0`.
pub fn run(args: QuantizeArgs) -> Result<()> {
    if let Some(forwarded) = detached_args(&args.args) {
        // `output` duplicates what is inside `args`; it is there so a QUEUED job
        // lists the artefact it will build. The runner ignores it.
        let id = crate::commands::jobs::submit(serde_json::json!({
            "kind": "quantize",
            "output": flag_value(&forwarded, "--output"),
            "args": forwarded,
        }))?;
        println!("queued quantize job {id}");
        println!("  hipfire jobs watch {id}");
        return Ok(());
    }
    hipfire_quantize::cli::main();
    Ok(())
}

/// The value following `flag`, for the job's display label.
fn flag_value(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

/// `Some(args without --detach)` when `--detach` was passed, else `None`.
fn detached_args(args: &[std::ffi::OsString]) -> Option<Vec<String>> {
    args.iter().any(|a| a == "--detach").then(|| {
        args.iter()
            .filter(|a| *a != "--detach")
            .map(|a| a.to_string_lossy().into_owned())
            .collect()
    })
}

#[derive(Debug, Args)]
#[command(
    after_help = "Light QAT: block-local RMSNorm recovery against a quantized artefact's own weights.\n\nFeed the result back with:\n  hipfire quantize --input <bf16.hfq> --output <recovered.hfq> --norm-patch <tuned_norms.json>\n\nNeeds the teacher's residual captures on disk first (see the hipfire-qat header for the capture command). qwen3.5 only today.\n"
)]
pub struct QatArgs {
    /// Quantized artefact whose norms are recovered.
    pub quantized: String,
    /// bf16 teacher artefact.
    pub bf16: String,
    /// Where to write the tuned-norms JSON.
    pub output: String,
    /// Queue it as a service job instead of running it now.
    #[arg(long)]
    pub detach: bool,
}

pub fn run_qat(args: QatArgs) -> Result<()> {
    if args.detach {
        let id = crate::commands::jobs::submit(serde_json::json!({
            "kind": "qat",
            "quantized": args.quantized,
            "bf16": args.bf16,
            "output": args.output,
        }))?;
        println!("queued qat job {id}");
        println!("  hipfire jobs watch {id}");
        return Ok(());
    }
    // Foreground: a sibling binary rather than an in-process call, because the
    // CLI does not (and should not) link the training crate. Under the GPU lock,
    // since hipfire-qat is a non-daemon GPU binary and does not self-lock.
    let bin = crate::commands::induct::sibling_binary("hipfire-qat", "HIPFIRE_QAT_BIN")
        .context("hipfire-qat not found (cargo build --release -p hipfire-train)")?;
    let hipfire = std::env::current_exe().unwrap_or_else(|_| "hipfire".into());
    crate::commands::induct::run_tool(
        &hipfire,
        &[
            "lock".into(),
            "run".into(),
            "qat".into(),
            "--".into(),
            bin.into_os_string(),
            args.quantized.into(),
            args.bf16.into(),
            args.output.into(),
        ],
        "cargo build --release -p hipfire-train",
    )
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
