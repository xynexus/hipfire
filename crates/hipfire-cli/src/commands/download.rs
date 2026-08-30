// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.
//! `hipfire download <org/name>` — fetch a model repository.
//!
//! A REAL subcommand, not an argv passthrough. That distinction is the whole
//! point: while this reached the user only as `hipfire convert download …`
//! through `#[command(external_subcommand)]`, clap saw a bare `Vec<String>`, so
//! `--help` could not list it and `gen-docs` could not render it — the generated
//! `docs/CLI.md` and `man/hipfire-convert.1` described five drafter tools and
//! never mentioned that downloading was possible at all.

use clap::Args;
use hipfire_coexistence::download::{run, DownloadOptions};
use std::path::PathBuf;

#[derive(Debug, Args)]
#[command(after_help = "Examples:\n  \
        hipfire download Qwen/Qwen3.5-9B\n  \
        hipfire download Qwen/Qwen3.5-9B --revision <sha>\n  \
        hipfire download Zyphra/ZAYA1-8B --include '*.safetensors'\n  \
        hipfire download Qwen/Qwen3.5-9B --raw          # HuggingFace cache tree\n\
        \nStreams into ~/.hipfire/models/models--Org--Name.hfa, encoding as it\n\
        downloads so the raw checkpoint is never staged. An interrupted run\n\
        leaves <archive>.hfa.part and resumes on the next download.")]
pub struct DownloadArgs {
    /// Repository to fetch, as `org/name`.
    ///
    /// HuggingFace is the only source today. When a second one exists it joins
    /// as `--source <name>` rather than a new subcommand.
    pub repo: String,
    /// Revision to pin: a commit sha, or `main`.
    #[arg(long, default_value = "main")]
    pub revision: String,
    /// Only fetch paths matching this glob.
    #[arg(long)]
    pub include: Option<String>,
    /// Destination root. Defaults to `~/.hipfire/models` (or `$HF_HOME` with `--raw`).
    #[arg(long)]
    pub dest: Option<PathBuf>,
    /// Write the archive to this exact path instead of deriving it from the repo.
    #[arg(long)]
    pub output: Option<PathBuf>,
    /// Replace an existing archive. Without this an existing file is never
    /// overwritten — these are routinely the only copy of a model on an array
    /// with no redundancy.
    #[arg(long)]
    pub force: bool,
    /// Fetch a HuggingFace cache tree instead of encoding to a `.hfa` archive.
    #[arg(long)]
    pub raw: bool,
    /// Parallel connections: whole files in raw mode, ranged windows within a
    /// file in archive mode.
    #[arg(long, default_value_t = 4)]
    pub jobs: usize,
    /// Queue the fetch as a background job instead of downloading here, and
    /// return its id. Monitor it with `hipfire jobs watch <id>`.
    ///
    /// The job is a file in `~/.hipfire/jobs/deferred/queued`, so this works
    /// whether or not the server is running — an unclaimed job simply waits.
    #[arg(long)]
    pub detach: bool,
}

pub fn run_download(args: DownloadArgs) -> anyhow::Result<()> {
    if args.detach {
        let mut spec = serde_json::json!({
            "kind": "download",
            "repo": args.repo,
            "revision": args.revision,
            "force": args.force,
            "raw": args.raw,
            "jobs": args.jobs,
        });
        // Only send the optional paths that were actually given, so the job
        // file records the request rather than this command's defaults.
        if let Some(v) = args.include {
            spec["include"] = serde_json::json!(v);
        }
        if let Some(v) = args.dest {
            spec["dest"] = serde_json::json!(v.display().to_string());
        }
        if let Some(v) = args.output {
            spec["output"] = serde_json::json!(v.display().to_string());
        }
        let id = crate::commands::jobs::submit(spec)?;
        println!("queued download job {id}");
        println!("  hipfire jobs watch {id}");
        return Ok(());
    }
    let opts = DownloadOptions {
        repo: args.repo,
        revision: args.revision,
        include: args.include,
        dest: args.dest,
        output: args.output,
        force: args.force,
        raw: args.raw,
        jobs: args.jobs,
    };
    run(&opts).map_err(|e| anyhow::anyhow!("{e}"))
}
