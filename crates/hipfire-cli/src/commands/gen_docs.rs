// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! `hipfire gen-docs` (hidden) — render the CLI's clap definitions into
//! committed user-facing docs, so the `///` doc comments on the arg structs are
//! the single source of truth for `--help`, `docs/CLI.md`, and the man pages.
//!
//! Writes `docs/CLI.md` (via `clap-markdown`) and `man/hipfire*.1` (via
//! `clap_mangen`, one page for the root command and one per subcommand). With
//! `--check` it renders to memory and diffs against what is on disk instead of
//! writing, exiting non-zero on drift — the freshness gate `tests/no-gpu-ci.sh`
//! runs so the docs can't silently fall out of sync with the code. Only
//! `docs/CLI.md` is tracked; `/man/` is gitignored, so its absence is not drift.
//! See `check_file`.

use std::path::Path;

use clap::CommandFactory;

use crate::Cli;

#[derive(Debug, clap::Args)]
pub struct GenDocsArgs {
    /// Directory for the generated Markdown reference (`CLI.md`).
    #[arg(long, default_value = "docs")]
    pub docs_dir: String,
    /// Directory for the generated man pages (`hipfire*.1`).
    #[arg(long, default_value = "man")]
    pub man_dir: String,
    /// Verify the committed docs match the current CLI without writing; exit
    /// non-zero on any drift (for CI).
    #[arg(long)]
    pub check: bool,
}

pub fn run(args: GenDocsArgs) -> anyhow::Result<()> {
    let markdown = render_markdown();
    let man_pages = render_man_pages()?;

    if args.check {
        let mut stale = Vec::new();
        check_file(
            Path::new(&args.docs_dir).join("CLI.md"),
            &markdown,
            true,
            &mut stale,
        );
        // `/man/` is gitignored, so a clean checkout has no man pages at all.
        // Requiring them made this gate unsatisfiable off a developer's own
        // tree; drift detection rides on the tracked `docs/CLI.md` above.
        for (name, bytes) in &man_pages {
            check_file(
                Path::new(&args.man_dir).join(name),
                bytes,
                false,
                &mut stale,
            );
        }
        if !stale.is_empty() {
            anyhow::bail!(
                "CLI docs are stale ({} file(s)): {}\n\
                 regenerate with `cargo run -p hipfire-cli -- gen-docs` and commit.",
                stale.len(),
                stale.join(", "),
            );
        }
        eprintln!("gen-docs: CLI docs are up to date");
        return Ok(());
    }

    std::fs::create_dir_all(&args.docs_dir)?;
    std::fs::create_dir_all(&args.man_dir)?;
    let cli_md = Path::new(&args.docs_dir).join("CLI.md");
    std::fs::write(&cli_md, markdown.as_bytes())?;
    eprintln!("gen-docs: wrote {}", cli_md.display());
    for (name, bytes) in &man_pages {
        let p = Path::new(&args.man_dir).join(name);
        std::fs::write(&p, bytes)?;
        eprintln!("gen-docs: wrote {}", p.display());
    }
    Ok(())
}

/// The full Markdown command reference, from the clap `Cli` definition.
fn render_markdown() -> String {
    clap_markdown::help_markdown::<Cli>()
}

/// One roff man page for the root command and one per subcommand, keyed by file
/// name (`hipfire.1`, `hipfire-<sub>.1`).
fn render_man_pages() -> anyhow::Result<Vec<(String, Vec<u8>)>> {
    // The binary's runtime `--version` is a dynamic Git-derived identity
    // (`hipfire_build_info::VERSION`, e.g. `v0.3.0-957-g...`). Pin the *man page*
    // to the static crate version so the docs freshness gate stays
    // deterministic — otherwise every commit would render a new `.TH`/VERSION
    // line and the gate could never be satisfied.
    let cmd = Cli::command().version(env!("CARGO_PKG_VERSION"));
    let mut out = Vec::new();

    let mut root = Vec::new();
    clap_mangen::Man::new(cmd.clone()).render(&mut root)?;
    out.push(("hipfire.1".to_string(), trim_trailing_line_ws(root)?));

    for sub in cmd.get_subcommands() {
        if sub.is_hide_set() {
            continue;
        }
        let file = format!("hipfire-{}.1", sub.get_name());
        // `Command::name` wants `Into<Str>` (≈ `&'static str`); leak the short
        // per-subcommand title — this is a one-shot maintenance command.
        let title: &'static str = Box::leak(format!("hipfire-{}", sub.get_name()).into_boxed_str());
        let titled = sub.clone().name(title);
        let mut buf = Vec::new();
        clap_mangen::Man::new(titled).render(&mut buf)?;
        out.push((file, trim_trailing_line_ws(buf)?));
    }
    Ok(out)
}

fn trim_trailing_line_ws(bytes: Vec<u8>) -> anyhow::Result<Vec<u8>> {
    let text = String::from_utf8(bytes)?;
    let mut out = String::with_capacity(text.len());
    for line in text.split_inclusive('\n') {
        let (body, newline) = line
            .strip_suffix('\n')
            .map(|body| (body, "\n"))
            .unwrap_or((line, ""));
        out.push_str(body.trim_end_matches([' ', '\t']));
        out.push_str(newline);
    }
    Ok(out.into_bytes())
}

/// Record `path` as stale unless it matches freshly generated `expected`.
///
/// `required` separates the two outputs. `docs/CLI.md` is tracked, so a missing
/// file is real drift. The man pages under `/man/` are generated and gitignored,
/// so their absence is the normal state of a clean checkout — but one that is
/// present still has to match, or a stale local copy would quietly mislead
/// whoever read it. Only `NotFound` is forgiven; an unreadable file still fails.
fn check_file(
    path: std::path::PathBuf,
    expected: impl AsRef<[u8]>,
    required: bool,
    stale: &mut Vec<String>,
) {
    match std::fs::read(&path) {
        Ok(got) if got == expected.as_ref() => {}
        Err(error) if !required && error.kind() == std::io::ErrorKind::NotFound => {}
        _ => stale.push(path.display().to_string()),
    }
}
