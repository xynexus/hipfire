pub mod admin;
pub mod bench;
pub mod chat;
pub mod daemon;
pub mod detect;
pub mod diffusion;
pub mod doctor;
pub mod env;
pub mod forward;
pub mod gen_config_schema;
pub mod gen_docs;
pub mod gen_env_docs;
pub mod gen_model_support;
pub mod induct;
pub mod inspect;
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
    let Some(name) = path.file_name() else {
        return false;
    };
    let dir = path.parent().filter(|p| !p.as_os_str().is_empty());
    Command::new("git")
        .arg("-C")
        .arg(dir.unwrap_or(Path::new(".")))
        .args(["check-ignore", "-q", "--"])
        .arg(name)
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

        std::fs::remove_dir_all(&dir).unwrap();
    }
}
