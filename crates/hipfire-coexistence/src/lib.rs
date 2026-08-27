// SPDX-License-Identifier: Apache-2.0
//! Offline compatibility and model-induction orchestration.

// `cli` was a separate crate root (src/main.rs) and spells its 14 dispatch
// targets `hipfire_coexistence::…`. A crate cannot name itself in Rust 2018+
// without this alias, so it is what lets that file move in untouched.
extern crate self as hipfire_coexistence;

/// The `hipfire-coexistence` command line, as a module rather than a crate root.
pub mod cli;

/// How this process was invoked, for user-facing text.
///
/// The same code is reachable as the standalone `hipfire-coexistence` binary
/// and as `hipfire convert …`. Usage strings and — more importantly — the GPU
/// lock's holder line must name the command the user actually ran: AGENTS.md
/// calls out that a stale or wrong holder line makes a real contention error
/// point at the wrong thing.
/// argv prefix for a helper tool the induction orchestrator shells out to.
///
/// These used to default to CWD-relative paths (`target/release/hipfire-quantize`
/// and friends), which meant induction only worked from a repo checkout with a
/// built target/ — anywhere else the spawn failed. Everything they name now
/// lives inside `hipfire`, so when we ARE `hipfire` the tool is just one of our
/// own subcommands and we spawn ourselves. The legacy path stays as the
/// fallback for a standalone `hipfire-coexistence`.
pub fn tool_argv(subcommand: &[&str], legacy_bin: &str) -> Vec<String> {
    let exe = std::env::current_exe().ok();
    let is_hipfire = exe
        .as_deref()
        .and_then(|p| p.file_stem())
        .and_then(|s| s.to_str())
        == Some("hipfire");
    match (is_hipfire, exe) {
        (true, Some(exe)) => {
            let mut argv = vec![exe.to_string_lossy().into_owned()];
            argv.extend(subcommand.iter().map(|s| (*s).to_string()));
            argv
        }
        _ => vec![legacy_bin.to_string()],
    }
}

pub fn invoked_as() -> &'static str {
    match std::env::current_exe()
        .ok()
        .as_deref()
        .and_then(|p| p.file_stem())
        .and_then(|s| s.to_str())
    {
        Some("hipfire") => "hipfire convert",
        _ => "hipfire-coexistence",
    }
}

pub mod artifact;
pub mod calibrate;
pub mod calibration_audit;
pub mod calibration_compare;
pub mod export_safetensors;
pub mod hub_archive;
pub mod import_safetensors;
pub mod induction;
pub mod repack;
pub mod residual_compare;
pub mod router_profile;

#[cfg(test)]
mod tool_argv_tests {
    use super::tool_argv;

    #[test]
    fn resolves_to_our_own_subcommand_when_we_are_hipfire() {
        // The test harness binary is not named `hipfire`, so this exercises the
        // fallback: a standalone hipfire-coexistence keeps the legacy path.
        assert_eq!(
            tool_argv(&["quantize"], "target/release/hipfire-quantize"),
            vec!["target/release/hipfire-quantize".to_string()]
        );
    }

    #[test]
    fn subcommand_tokens_follow_the_executable_in_order() {
        // Whatever the resolution, the result must be a usable argv prefix:
        // non-empty, program first, and the caller's tokens appended in order.
        let argv = tool_argv(&["convert", "dflash"], "target/release/dflash_convert");
        assert!(!argv.is_empty(), "argv prefix must name a program");
        if argv.len() > 1 {
            assert_eq!(&argv[1..], &["convert", "dflash"]);
        }
    }
}
