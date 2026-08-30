// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.
//! Fetch a model repository, as a typed call rather than an argv shape.
//!
//! Extracted from `cli.rs`'s hand-rolled flag bag so `hipfire download` can be a
//! REAL subcommand: clap owns the arguments, `--help` can describe them, and
//! `gen-docs` can render them. While the groups arrived through
//! `#[command(external_subcommand)]` none of that was possible — clap saw a bare
//! `Vec<String>`, so the generated `docs/CLI.md` and `man/hipfire-convert.1`
//! documented five drafter tools and never mentioned that `download` existed.
//!
//! Offline tooling: the runtime never links this.

use std::error::Error;
use std::path::PathBuf;

/// Where archives land by default: `~/.hipfire/models`, derived from `$HOME` the
/// way every other crate locates `~/.hipfire`. Deliberately not an env var —
/// `--dest` already overrides it.
pub fn archive_root() -> PathBuf {
    match std::env::var("HOME") {
        Ok(home) => PathBuf::from(home).join(".hipfire").join("models"),
        Err(_) => PathBuf::from(".hipfire/models"),
    }
}

/// Canonical archive filename for a repo id.
pub fn archive_name(repo: &str) -> String {
    format!("models--{}.hfa", repo.replace('/', "--"))
}

/// What to fetch and where to put it.
#[derive(Debug, Clone)]
pub struct DownloadOptions {
    /// `org/name` on the source hub.
    pub repo: String,
    /// Revision to pin: a sha, or `main`.
    pub revision: String,
    /// Only fetch paths matching this glob.
    pub include: Option<String>,
    /// Override the destination root.
    pub dest: Option<PathBuf>,
    /// Override the archive path outright.
    pub output: Option<PathBuf>,
    /// Replace an existing archive.
    pub force: bool,
    /// Fetch a HuggingFace cache tree instead of encoding to `.hfa`.
    pub raw: bool,
    /// Parallel connections: whole files in raw mode, ranged windows within a
    /// file in archive mode.
    pub jobs: usize,
}

impl DownloadOptions {
    /// Defaults matching the documented CLI behaviour.
    pub fn new(repo: impl Into<String>) -> Self {
        Self {
            repo: repo.into(),
            revision: "main".to_string(),
            include: None,
            dest: None,
            output: None,
            force: false,
            raw: false,
            jobs: 4,
        }
    }

    /// Destination root, honouring `--dest` and the raw-vs-archive default.
    pub fn root(&self) -> PathBuf {
        match &self.dest {
            Some(d) => d.clone(),
            None if self.raw => PathBuf::from(
                std::env::var("HF_HOME").unwrap_or_else(|_| "/srv/huggingface".to_string()),
            ),
            None => archive_root(),
        }
    }

    /// Archive path, honouring `--output`.
    pub fn archive(&self) -> PathBuf {
        match &self.output {
            Some(o) => o.clone(),
            None => self.root().join(archive_name(&self.repo)),
        }
    }
}

/// Fetch `opts.repo`, streaming into a `.hfa` archive unless `--raw`.
///
/// Builds its own tokio runtime because this is an offline tool with no ambient
/// one, and `hipfire`'s `main` is deliberately not `#[tokio::main]`.
pub fn run(opts: &DownloadOptions) -> Result<(), Box<dyn Error>> {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|e| format!("download: runtime: {e}"))?;
    rt.block_on(run_async(opts))
}

/// The async half, for a caller that already has a runtime.
pub async fn run_async(opts: &DownloadOptions) -> Result<(), Box<dyn Error>> {
    let root = opts.root();
    if opts.raw {
        let n = hipfire_hub::run::fetch(
            &root,
            &opts.repo,
            &opts.revision,
            opts.include.as_deref(),
            opts.jobs,
        )
        .await?;
        eprintln!("download: {n} file(s) present and verified");
        return Ok(());
    }

    let archive = opts.archive();
    // These archives are routinely the only copy of their model on an array with
    // no redundancy, so overwriting one is never the silent default.
    if archive.exists() && !opts.force {
        return Err(format!(
            "download: {} already exists — pass --force to replace it, \
             or `repack --check` it first",
            archive.display()
        )
        .into());
    }
    // Written under a `.part` marker and only renamed into place once complete
    // and digest-verified, so the final name never holds a truncated file. A
    // leftover marker is from an interrupted run: `fetch_to_archive` resumes it
    // at the last completed file via the `.manifest` sidecar (rechecking the
    // kept bytes first), or restarts if that is missing, stale, or fails.
    let part = PathBuf::from(format!("{}.part", archive.display()));
    if let Some(p) = archive.parent() {
        std::fs::create_dir_all(p)?;
    }
    let files = hipfire_hub::list_files(&opts.repo, &opts.revision).await?;
    crate::hub_archive::fetch_to_archive(
        &part,
        &opts.repo,
        &opts.revision,
        opts.include.as_deref(),
        files,
        opts.jobs,
    )
    .await?;
    std::fs::rename(&part, &archive)?;
    eprintln!("download: wrote {}", archive.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn archive_name_flattens_the_repo_id() {
        assert_eq!(
            archive_name("Qwen/Qwen3.5-9B"),
            "models--Qwen--Qwen3.5-9B.hfa"
        );
    }

    #[test]
    fn output_overrides_dest_and_dest_overrides_the_default_root() {
        let mut o = DownloadOptions::new("Org/Name");
        assert_eq!(o.archive(), archive_root().join("models--Org--Name.hfa"));

        o.dest = Some(PathBuf::from("/tmp/models"));
        assert_eq!(
            o.archive(),
            PathBuf::from("/tmp/models/models--Org--Name.hfa")
        );

        o.output = Some(PathBuf::from("/tmp/explicit.hfa"));
        assert_eq!(o.archive(), PathBuf::from("/tmp/explicit.hfa"));
    }

    /// Raw mode targets the HuggingFace cache root, not the archive root — a
    /// raw fetch produces a cache tree that other tools expect to find there.
    #[test]
    fn raw_mode_defaults_to_the_hf_cache_root() {
        let mut o = DownloadOptions::new("Org/Name");
        o.raw = true;
        let root = o.root();
        assert_ne!(root, archive_root());
        assert!(o.dest.is_none());
    }
}
