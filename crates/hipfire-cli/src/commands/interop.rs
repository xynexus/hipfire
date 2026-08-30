// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.
//! `hipfire import`, `hipfire export`, `hipfire repack` — real subcommands for
//! the three user-facing interop groups.
//!
//! Promoted from the `hipfire convert <group>` argv passthrough for the reason
//! `download` was: while a group arrives through `#[command(external_subcommand)]`
//! clap sees a bare `Vec<String>`, so neither `--help` nor `gen-docs` can
//! describe it. Promotion is what makes them self-documenting — the man pages
//! appear on their own.
//!
//! Flag names mirror the hand-rolled bags EXACTLY, including their
//! inconsistency: `import gguf` takes `--in`/`--out` while every other group
//! takes `--input`/`--output`. Scripts already use both spellings, so the names
//! are preserved and the inconsistency is recorded in
//! `docs/todo/2026-08-30-coexistence-subcommand-promotion.md` rather than fixed
//! silently here.

use clap::{Args, Subcommand};
use std::path::PathBuf;

fn to_argv(pairs: Vec<(&str, Option<String>)>, flags: Vec<(&str, bool)>) -> Vec<String> {
    let mut argv = Vec::new();
    for (k, v) in pairs {
        if let Some(v) = v {
            argv.push(k.to_string());
            argv.push(v);
        }
    }
    for (k, on) in flags {
        if on {
            argv.push(k.to_string());
        }
    }
    argv
}

// ── import ───────────────────────────────────────────────────────────────

#[derive(Debug, Args)]
pub struct ImportArgs {
    #[command(subcommand)]
    command: ImportCommand,
}

#[derive(Debug, Subcommand)]
enum ImportCommand {
    /// Import a GGUF checkpoint into a `.hfq`.
    Gguf(ImportGgufArgs),
    /// Import a HuggingFace safetensors directory into a `.hfq`.
    Safetensors(ImportSafetensorsArgs),
}

#[derive(Debug, Args)]
pub struct ImportGgufArgs {
    /// Source `.gguf`. (Spelled `--in`, not `--input`, to match the existing tool.)
    #[arg(long = "in")]
    input: PathBuf,
    /// Destination `.hfq`.
    #[arg(long = "out")]
    output: PathBuf,
    /// Target quant format token.
    #[arg(long)]
    format: String,
    /// Disable the k-map, quantizing uniformly.
    #[arg(long, alias = "uniform")]
    no_kmap: bool,
    /// Dense k-map.
    #[arg(long)]
    kmap_dense: bool,
    /// k-map mode: `full`, `alternating`/`alt`, or `typed`.
    #[arg(long, default_value = "alternating")]
    kmap_mode: String,
}

#[derive(Debug, Args)]
pub struct ImportSafetensorsArgs {
    /// Source HuggingFace directory.
    #[arg(long)]
    input: PathBuf,
    /// Destination `.hfq`.
    #[arg(long)]
    output: PathBuf,
    /// Architecture family override.
    #[arg(long)]
    arch: Option<String>,
}

pub fn run_import(args: ImportArgs) -> anyhow::Result<()> {
    let err = |e: Box<dyn std::error::Error>| anyhow::anyhow!("{e}");
    match args.command {
        ImportCommand::Gguf(a) => {
            let argv = to_argv(
                vec![
                    ("--in", Some(a.input.display().to_string())),
                    ("--out", Some(a.output.display().to_string())),
                    ("--format", Some(a.format)),
                    ("--kmap-mode", Some(a.kmap_mode)),
                ],
                vec![("--no-kmap", a.no_kmap), ("--kmap-dense", a.kmap_dense)],
            );
            // gguf import's entry point is private to coexistence's dispatcher,
            // so it goes through the public `cli::run` with the group tokens
            // rebuilt. The other three call their `run_cli` directly.
            let mut full = vec!["import".to_string(), "gguf".to_string()];
            full.extend(argv);
            hipfire_coexistence::cli::run(&full).map_err(err)
        }
        ImportCommand::Safetensors(a) => {
            let argv = to_argv(
                vec![
                    ("--input", Some(a.input.display().to_string())),
                    ("--output", Some(a.output.display().to_string())),
                    ("--arch", a.arch),
                ],
                vec![],
            );
            hipfire_coexistence::import_safetensors::run_cli(&argv).map_err(err)
        }
    }
}

// ── export ───────────────────────────────────────────────────────────────

#[derive(Debug, Args)]
pub struct ExportArgs {
    #[command(subcommand)]
    command: ExportCommand,
}

#[derive(Debug, Subcommand)]
enum ExportCommand {
    /// Export a `.hfq` back to a HuggingFace safetensors directory.
    Safetensors(ExportSafetensorsArgs),
}

#[derive(Debug, Args)]
pub struct ExportSafetensorsArgs {
    /// Source `.hfq`.
    #[arg(long)]
    input: PathBuf,
    /// Destination directory.
    #[arg(long)]
    output: PathBuf,
    /// Architecture family override.
    #[arg(long)]
    arch: Option<String>,
    /// Shard size, e.g. `5G`.
    #[arg(long)]
    shard_size: Option<String>,
}

pub fn run_export(args: ExportArgs) -> anyhow::Result<()> {
    let ExportCommand::Safetensors(a) = args.command;
    let argv = to_argv(
        vec![
            ("--input", Some(a.input.display().to_string())),
            ("--output", Some(a.output.display().to_string())),
            ("--arch", a.arch),
            ("--shard-size", a.shard_size),
        ],
        vec![],
    );
    hipfire_coexistence::export_safetensors::run_cli(&argv).map_err(|e| anyhow::anyhow!("{e}"))
}

// ── repack ───────────────────────────────────────────────────────────────

/// NOT `optimize`. This is the HF-dir <-> `.hfa` archive round-trip; `optimize`
/// is an arch-optimal weight-layout pass over a `.hfq`. They were one `repack`
/// alias away from colliding, which is why that alias is gone.
#[derive(Debug, Args)]
#[command(after_help = "Examples:\n  \
        hipfire repack --input <hf_dir> --output <archive.hfa>   # pack, lossless\n  \
        hipfire repack --input <archive.hfa> --output <hf_dir>   # restore, byte-identical\n  \
        hipfire repack --input <archive.hfa> --check             # verify stored checksums\n\
        \nNot to be confused with `hipfire optimize`, which rewrites a .hfq into\n\
        an arch-optimal weight layout.")]
pub struct RepackArgs {
    /// Source: a HuggingFace directory to pack, or a `.hfa` to restore/check.
    #[arg(long)]
    input: PathBuf,
    /// Destination. Omit with `--check`.
    #[arg(long)]
    output: Option<PathBuf>,
    /// Verify the restored tree against this directory.
    #[arg(long)]
    verify: Option<PathBuf>,
    /// Verify stored checksums without writing anything.
    #[arg(long)]
    check: bool,
    /// Upgrade an older archive in place.
    #[arg(long)]
    upgrade: bool,
}

pub fn run_repack(args: RepackArgs) -> anyhow::Result<()> {
    let argv = to_argv(
        vec![
            ("--input", Some(args.input.display().to_string())),
            ("--output", args.output.map(|p| p.display().to_string())),
            ("--verify", args.verify.map(|p| p.display().to_string())),
        ],
        vec![("--check", args.check), ("--upgrade", args.upgrade)],
    );
    hipfire_coexistence::repack::run_cli(&argv).map_err(|e| anyhow::anyhow!("{e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The bridge only emits flags whose value is present, so an omitted
    /// `Option` must not reach the tool as a bare flag with a missing value --
    /// those bags read the NEXT token as the value and would silently consume
    /// an unrelated flag.
    #[test]
    fn absent_options_emit_no_flag() {
        let argv = to_argv(
            vec![("--input", Some("a".into())), ("--arch", None)],
            vec![("--check", false), ("--upgrade", true)],
        );
        assert_eq!(argv, vec!["--input", "a", "--upgrade"]);
    }
}

// ── lora ─────────────────────────────────────────────────────────────────

#[derive(Debug, Args)]
pub struct LoraArgs {
    #[command(subcommand)]
    command: LoraCommand,
}

#[derive(Debug, Subcommand)]
enum LoraCommand {
    /// Derive a steering adapter from contrastive prompt sets.
    Export(LoraExportArgs),
    /// Merge an adapter into a base `.hfq`.
    Merge(LoraMergeArgs),
    /// Convert an adapter between `.hfq` and `.json` forms.
    Convert(LoraConvertArgs),
}

#[derive(Debug, Args)]
pub struct LoraExportArgs {
    /// Base model to derive the adapter against.
    #[arg(long)]
    hfq: PathBuf,
    /// Directory holding `good_prompts.txt` and `bad_prompts.txt`.
    #[arg(long)]
    data_dir: PathBuf,
    /// Destination adapter (`.lora.hfq` or `.lora.json`).
    #[arg(long)]
    out: PathBuf,
    /// Prompts to read from each set.
    #[arg(long, default_value_t = 16)]
    limit: usize,
    /// Steering strength.
    #[arg(long, default_value_t = 0.2)]
    strength: f32,
    /// Max sequence length during capture.
    #[arg(long, default_value_t = 2048)]
    max_seq: usize,
    /// Skip orthogonalisation of the derived directions.
    #[arg(long)]
    no_orthogonalize: bool,
}

#[derive(Debug, Args)]
pub struct LoraMergeArgs {
    /// Base `.hfq`.
    #[arg(long)]
    hfq: PathBuf,
    /// Adapter to merge.
    #[arg(long)]
    adapter: PathBuf,
    /// Destination merged `.hfq`.
    #[arg(long)]
    out: PathBuf,
}

#[derive(Debug, Args)]
pub struct LoraConvertArgs {
    /// Source adapter. (Spelled `--in`, matching the existing tool.)
    #[arg(long = "in")]
    input: PathBuf,
    /// Destination adapter.
    #[arg(long)]
    out: PathBuf,
}

pub fn run_lora(args: LoraArgs) -> anyhow::Result<()> {
    let err = |e: Box<dyn std::error::Error>| anyhow::anyhow!("{e}");
    let (op, argv) = match args.command {
        LoraCommand::Export(a) => (
            "export",
            to_argv(
                vec![
                    ("--hfq", Some(a.hfq.display().to_string())),
                    ("--data-dir", Some(a.data_dir.display().to_string())),
                    ("--out", Some(a.out.display().to_string())),
                    ("--limit", Some(a.limit.to_string())),
                    ("--strength", Some(a.strength.to_string())),
                    ("--max-seq", Some(a.max_seq.to_string())),
                ],
                vec![("--no-orthogonalize", a.no_orthogonalize)],
            ),
        ),
        LoraCommand::Merge(a) => (
            "merge",
            to_argv(
                vec![
                    ("--hfq", Some(a.hfq.display().to_string())),
                    ("--adapter", Some(a.adapter.display().to_string())),
                    ("--out", Some(a.out.display().to_string())),
                ],
                vec![],
            ),
        ),
        LoraCommand::Convert(a) => (
            "convert",
            to_argv(
                vec![
                    ("--in", Some(a.input.display().to_string())),
                    ("--out", Some(a.out.display().to_string())),
                ],
                vec![],
            ),
        ),
    };
    // These three entry points are private to coexistence's dispatcher, so they
    // go through the public `cli::run` with the group tokens rebuilt.
    let mut full = vec!["lora".to_string(), op.to_string()];
    full.extend(argv);
    hipfire_coexistence::cli::run(&full).map_err(err)
}

// ── artifact ─────────────────────────────────────────────────────────────

/// The five read-only artifact reporters.
///
/// `inspect` is deliberately absent: it folded into `hipfire inspect`, which
/// already reported a superset. The one thing it had that `inspect` did not was
/// a second, independently-computed fingerprint — since unified, so there is
/// nothing left to promote.
#[derive(Debug, Args)]
pub struct ArtifactArgs {
    #[command(subcommand)]
    command: ArtifactCommand,
}

#[derive(Debug, Subcommand)]
enum ArtifactCommand {
    /// Check a `.calib.hfq` for structural and coverage problems.
    AuditCalibration(AuditCalibrationArgs),
    /// Compare two calibration artifacts numerically.
    CompareCalibration(CompareCalibrationArgs),
    /// Compare a lower-capacity calibration against a higher-capacity one.
    CompareCalibrationStability(ComparePairArgs),
    /// Compare two residual-probe artifacts.
    CompareResiduals(CompareResidualsArgs),
    /// Report routed-expert activation distribution from a calibration artifact.
    MoeRouterProfile(MoeRouterProfileArgs),
}

#[derive(Debug, Args)]
pub struct AuditCalibrationArgs {
    /// Calibration artifact to audit.
    #[arg(long)]
    input: PathBuf,
}

#[derive(Debug, Args)]
pub struct CompareCalibrationArgs {
    /// Reference artifact.
    #[arg(long)]
    reference: PathBuf,
    /// Candidate artifact.
    #[arg(long)]
    candidate: PathBuf,
    /// Absolute tolerance.
    #[arg(long)]
    atol: Option<f64>,
    /// Relative tolerance.
    #[arg(long)]
    rtol: Option<f64>,
    /// Cap on reported mismatches.
    #[arg(long)]
    max_reports: Option<usize>,
    /// Compare even when provenance cannot be proven equal.
    #[arg(long)]
    allow_unproven_provenance: bool,
}

#[derive(Debug, Args)]
pub struct ComparePairArgs {
    /// Reference (higher-capacity) artifact.
    #[arg(long)]
    reference: PathBuf,
    /// Candidate (lower-capacity) artifact.
    #[arg(long)]
    candidate: PathBuf,
}

#[derive(Debug, Args)]
pub struct CompareResidualsArgs {
    /// Reference residuals artifact.
    #[arg(long)]
    reference: PathBuf,
    /// Candidate residuals artifact.
    #[arg(long)]
    candidate: PathBuf,
    /// Absolute tolerance.
    #[arg(long)]
    atol: Option<f64>,
    /// Relative tolerance.
    #[arg(long)]
    rtol: Option<f64>,
    /// Cap on reported mismatches.
    #[arg(long)]
    max_reports: Option<usize>,
}

#[derive(Debug, Args)]
pub struct MoeRouterProfileArgs {
    /// Calibration artifact to profile.
    #[arg(long)]
    input: PathBuf,
    /// Restrict to one layer.
    #[arg(long)]
    layer: Option<usize>,
    /// Report the top N experts.
    #[arg(long)]
    top: Option<usize>,
    /// Ignore experts below this activation count.
    #[arg(long)]
    min_activations: Option<u64>,
    /// Tokenizer for naming, when available.
    #[arg(long)]
    tokenizer: Option<PathBuf>,
    /// Emit JSON instead of a table.
    #[arg(long)]
    json: bool,
}

pub fn run_artifact(args: ArtifactArgs) -> anyhow::Result<()> {
    let err = |e: Box<dyn std::error::Error>| anyhow::anyhow!("{e}");
    let (op, argv) = match args.command {
        ArtifactCommand::AuditCalibration(a) => (
            "audit-calibration",
            to_argv(
                vec![("--input", Some(a.input.display().to_string()))],
                vec![],
            ),
        ),
        ArtifactCommand::CompareCalibration(a) => (
            "compare-calibration",
            to_argv(
                vec![
                    ("--reference", Some(a.reference.display().to_string())),
                    ("--candidate", Some(a.candidate.display().to_string())),
                    ("--atol", a.atol.map(|v| v.to_string())),
                    ("--rtol", a.rtol.map(|v| v.to_string())),
                    ("--max-reports", a.max_reports.map(|v| v.to_string())),
                ],
                vec![("--allow-unproven-provenance", a.allow_unproven_provenance)],
            ),
        ),
        ArtifactCommand::CompareCalibrationStability(a) => (
            "compare-calibration-stability",
            to_argv(
                vec![
                    ("--reference", Some(a.reference.display().to_string())),
                    ("--candidate", Some(a.candidate.display().to_string())),
                ],
                vec![],
            ),
        ),
        ArtifactCommand::CompareResiduals(a) => (
            "compare-residuals",
            to_argv(
                vec![
                    ("--reference", Some(a.reference.display().to_string())),
                    ("--candidate", Some(a.candidate.display().to_string())),
                    ("--atol", a.atol.map(|v| v.to_string())),
                    ("--rtol", a.rtol.map(|v| v.to_string())),
                    ("--max-reports", a.max_reports.map(|v| v.to_string())),
                ],
                vec![],
            ),
        ),
        ArtifactCommand::MoeRouterProfile(a) => (
            "moe-router-profile",
            to_argv(
                vec![
                    ("--input", Some(a.input.display().to_string())),
                    ("--layer", a.layer.map(|v| v.to_string())),
                    ("--top", a.top.map(|v| v.to_string())),
                    (
                        "--min-activations",
                        a.min_activations.map(|v| v.to_string()),
                    ),
                    ("--tokenizer", a.tokenizer.map(|p| p.display().to_string())),
                ],
                vec![("--json", a.json)],
            ),
        ),
    };
    let mut full = vec!["artifact".to_string(), op.to_string()];
    full.extend(argv);
    hipfire_coexistence::cli::run(&full).map_err(err)
}
