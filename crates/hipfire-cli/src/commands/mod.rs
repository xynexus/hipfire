pub mod admin;
pub mod bench;
pub mod chat;
pub mod daemon;
pub mod detect;
pub mod diffusion;
pub mod doctor;
pub mod download;
pub mod env;
pub mod forward;
pub mod gen_config_schema;
pub mod gen_docs;
pub mod gen_env_docs;
pub mod gen_model_support;
pub mod induct;
pub mod inspect;
pub mod interop;
pub mod jobs;
pub mod list;
pub mod lock;
pub mod model;
pub mod quantize;
pub mod serve;
pub mod tools;

use std::path::Path;
use std::process::Command;

/// True when `path` is a gitignored generated output — one no commit can carry.
///
/// A freshness `--check` on such a file is one-way. CI clones fresh and never
/// has it, so nothing is enforced there; every developer worktree *does* have
/// it, so the first commit that changes the generator's input fails the gate
/// for everyone, naming a file they cannot commit the fix for. `docs/env-vars.md`
/// (`.gitignore`) and all of `/man/` sit in exactly that position — the gate
/// only ever fired as a local false alarm. Skip them; the tracked siblings
/// (`docs/CLI.md`, `MODEL-SUPPORT.md`, `docs/config-schema.*`) are what actually
/// catch drift, because CI has those.
///
/// `git check-ignore` exits 0 when ignored, 1 when not, 128 with no repo. Only a
/// clean 0 skips, so a missing or broken git keeps the old strict behavior.
pub(crate) fn is_git_ignored(path: &Path) -> bool {
    // Run from the nearest ANCESTOR THAT EXISTS, passing the path unchanged.
    //
    // `check-ignore` is a pattern query — it answers for paths that do not exist,
    // which is the case that matters: in CI `/man/` is absent entirely, and the
    // whole point is to recognise it as ignored anyway. An earlier version ran
    // `git -C <path's parent>`, which exits 128 ("cannot change to 'man'") when
    // that parent is missing, so every man page was reported NOT ignored and the
    // freshness gate failed on a fresh clone. That is the same local-vs-CI
    // asymmetry this helper exists to remove, reintroduced by the helper itself.
    let mut dir = path.parent().unwrap_or(Path::new("."));
    while !dir.as_os_str().is_empty() && !dir.is_dir() {
        dir = dir.parent().unwrap_or(Path::new("."));
    }
    let dir = if dir.as_os_str().is_empty() {
        Path::new(".")
    } else {
        dir
    };
    Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(["check-ignore", "-q", "--"])
        .arg(path)
        .output()
        .is_ok_and(|out| out.status.code() == Some(0))
}

#[cfg(test)]
mod tests {
    use super::is_git_ignored;

    /// Hermetic: builds its own repo so it does not depend on hipfire's own
    /// `.gitignore` or on being run from inside a checkout.
    #[test]
    fn only_gitignored_outputs_are_skipped() {
        let dir = std::env::temp_dir().join(format!("hipfire-ignore-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("man")).unwrap();
        std::fs::create_dir_all(dir.join("docs")).unwrap();
        std::fs::write(dir.join(".gitignore"), "/man/\n").unwrap();
        std::fs::write(dir.join("man/hipfire.1"), "x").unwrap();
        std::fs::write(dir.join("docs/CLI.md"), "x").unwrap();
        let ok = std::process::Command::new("git")
            .arg("-C")
            .arg(&dir)
            .args(["init", "-q"])
            .status()
            .unwrap()
            .success();
        assert!(ok, "git init failed");

        assert!(
            is_git_ignored(&dir.join("man/hipfire.1")),
            "a gitignored generated output must be skipped by --check"
        );
        assert!(
            !is_git_ignored(&dir.join("docs/CLI.md")),
            "a committable output must still be checked"
        );

        // THE CI CASE, and the one that escaped a first version of this helper:
        // a fresh clone has no `/man/` at all. `check-ignore` is a pattern query
        // and answers for paths that do not exist, so an ignored output must be
        // recognised as ignored even when its directory is missing. Running git
        // from the path's own parent made this exit 128 and report NOT ignored,
        // which failed the docs gate on every fresh clone while passing locally.
        std::fs::remove_dir_all(dir.join("man")).unwrap();
        assert!(
            is_git_ignored(&dir.join("man/hipfire.1")),
            "an ignored output must be recognised even when its directory is absent"
        );
        assert!(
            is_git_ignored(&dir.join("man/deeper/nested.1")),
            "…including through several missing levels"
        );

        std::fs::remove_dir_all(&dir).unwrap();
    }
}

/// Is `pid` a live process on this host?
///
/// `kill(pid, 0)` delivers nothing and only probes for existence: `0` means
/// alive, `EPERM` means alive but not ours to signal (another uid), and `ESRCH`
/// means gone. Dropping the EPERM case reports another user's process as dead —
/// which is what the third copy of this in `hipfire-hub`'s cache did, reclaiming
/// `.part` downloads that were still being written.
///
/// `pid <= 1` is refused: `kill` reads 0 as "this process group" and negatives
/// as a group id, so neither names a process, and pid 1 is never a hipfire
/// process worth waiting on.
pub(crate) fn pid_alive(pid: impl Into<i64>) -> bool {
    let pid: i64 = pid.into();
    if pid <= 1 {
        return false;
    }
    let Ok(pid) = i32::try_from(pid) else {
        return false;
    };
    // SAFETY: signal 0 is a pure existence probe.
    unsafe { libc::kill(pid, 0) == 0 || *libc::__errno_location() == libc::EPERM }
}
