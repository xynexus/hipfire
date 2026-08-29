# The doc-freshness gates could only fail locally, never in CI

Status: found and **FIXED** 2026-08-29, master `0c9e3d252`, nix1. Fix is in the
working tree (`crates/hipfire-cli/src/commands/{mod,gen_docs,gen_env_docs}.rs`,
`tests/no-gpu-ci.sh`), not yet committed.

## Symptom

`./tests/no-gpu-ci.sh` — the gate AGENTS.md requires before handing off
workflow-only changes — failed on a **clean, up-to-date `master` worktree** with
`git status` reporting nothing to commit:

```
Error: env docs are stale (1 file(s)): /home/sadara/hipfire/docs/env-vars.md
regenerate with `cargo run -p hipfire-cli -- gen-env-docs`.
```

Regenerating cleared it and revealed a second failure immediately behind it:

```
Error: CLI docs are stale (1 file(s)): man/hipfire.1
regenerate with `cargo run -p hipfire-cli -- gen-docs` and commit.
```

Regenerating that too made the whole gate pass. `git status` stayed clean
throughout — **git never saw any of it.**

## Cause

Both checks are presence-based. `check_file` treats an *absent* file as fine and
a *present-but-different* file as stale:

```rust
match std::fs::read(path) {
    Ok(got) if got == expected => {}
    Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
    _ => stale.push(path.display().to_string()),
}
```

For a **tracked** output that is exactly right: CI has the file, so CI enforces
it. For a **gitignored** output it is inverted, and both of these are gitignored:

| output | ignore rule |
|---|---|
| `docs/env-vars.md` | `.gitignore:169` |
| `man/` (all pages) | `.gitignore:20` (`/man/`) |

- **CI clones fresh, so the file is absent, so nothing is ever enforced.** The
  check had zero enforcement value where it was supposed to have all of it.
- **Every developer worktree has the file**, so the first commit that changes the
  generator's input fails the gate for everyone — pointing at a path no commit
  can carry the fix for. `gen-docs`'s message even said *"regenerate … and
  commit"*, which is impossible for `/man/`.

The drift was genuine, not spurious: `4ddf67218` added `HIPFIRE_QAT_TIER` and
renamed `qat_w3_kvarn.rs` → `qat_opus_kvarn.rs`, `e4bc6e837` added
`HIPFIRE_ORACLE_DUMP`, and `2b6247d53` folded six subcommands into the CLI and
committed the regenerated `docs/CLI.md` — the man pages generated from that same
clap definition could not ride along.

Half of this was already understood. `gen_docs.rs` carried
*"`/man/` is gitignored, so its absence is not drift"* and passed `required:
false` for the man pages. That fixed the absent case. The stale-present case —
the one that actually fires — was left.

## Fix

One shared predicate in `crates/hipfire-cli/src/commands/mod.rs`, used by both
`check_file`s:

```rust
pub(crate) fn is_git_ignored(path: &Path) -> bool {
    let Some(name) = path.file_name() else { return false };
    let dir = path.parent().filter(|p| !p.as_os_str().is_empty());
    Command::new("git")
        .arg("-C").arg(dir.unwrap_or(Path::new(".")))
        .args(["check-ignore", "-q", "--"]).arg(name)
        .output()
        .is_ok_and(|out| out.status.code() == Some(0))
}
```

Exit 0 = ignored, 1 = not, 128 = no repo. Only a clean 0 skips, so a missing or
broken git keeps the old strict behavior. `git check-ignore` consults the index,
so a tracked path is never reported ignored.

Two things fell out of the change rather than being added to it:

- **`gen_docs`'s `required` bool is gone.** The skip handles the missing-man-page
  case earlier, leaving one live value — and `gen_env_docs` already carried a
  note from someone removing the same parameter there: *"a bool with one live
  value is a trap, not a feature."*
- **`tests/no-gpu-ci.sh` now runs `cargo test -p hipfire-cli --bin hipfire`.**
  See below.

## The second finding: 92 tests nothing ran

`hipfire-cli` is bin-only (`[[bin]] name = "hipfire"`, no `src/lib.rs`). CI's
workspace test gate is `cargo test --lib --workspace --locked`
(`.github/workflows/ci.yml:71`), which selects **zero targets** from a package
with no lib and skips it in silence. `no-gpu-ci.sh` invoked the crate's binary
for the `gen-*` checks but never ran its tests.

So the crate's 92 unit tests had never run in any automated path. This is the
same trap `no-gpu-ci.sh` already documents for `hipfire-daemon` — *"both
spellings exit 0, only one of them runs anything"* — still live one crate over.

## Verification

Discrimination, not just a green run:

| case | expected | result |
|---|---|---|
| ignored `man/hipfire.1` stale | pass | exit 0 |
| tracked `docs/CLI.md` stale | **fail** | exit 1 |
| ignored `docs/env-vars.md` stale | pass | exit 0 |
| undocumented `HIPFIRE_*` in README.md | **fail** | exit 1 |

The last row matters most: it proves the check was not made vacuous.
`coverage_gaps` — the half CI can actually enforce, since it reads only tracked
docs — still bites. The full gate then ran green *with both ignored artifacts
deliberately stale*, and the drift markers were still in the files afterward,
confirming they were skipped rather than quietly regenerated.

## Left alone

`gen_config_schema` and `gen_model_support` use the same check shape, but their
outputs are tracked, so they behave correctly. Add the guard there if one ever
becomes gitignored.
