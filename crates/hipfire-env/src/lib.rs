// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! The single declaration point for every hipfire environment variable.
//!
//! This is the Rust equivalent of a C header that declares `NAME` + a real
//! description and is consumed twice: once by the code that reads the variable,
//! once by `hipfire help env`. [`env_vars!`] expands each entry into both a
//! named constant (used at the read site) and an element of [`ALL`] (walked by
//! the CLI), so the two can never disagree.
//!
//! # Why a central table and not `inventory`
//!
//! The workspace already uses `inventory` for architecture registration, where
//! collecting only what is linked into the binary is exactly right. Env-var
//! documentation wants the opposite: `hipfire help env` must list every
//! variable regardless of which feature-gated arch crates a given build pulled
//! in. A distributed slice would silently shorten the list per build config, so
//! the table is central.
//!
//! # Adding a variable
//!
//! Add a line to [`env_vars!`] with a description a user can act on — what it
//! changes, accepted values, and the default — then read it through the
//! constant. `std::env::var` is denied by `clippy.toml` outside this file; the
//! lint message points here.
//!
//! # Migration status
//!
//! Crates opt into enforcement by adding `[lints] workspace = true` to their
//! `Cargo.toml`, which is only valid once all of that crate's reads go through
//! this table. So the set of opted-in crates is the migration progress, and
//! `rg -L 'workspace = true' crates/*/Cargo.toml` shows what is left.

use std::str::FromStr;

/// Who a variable is for. `hipfire help env` shows [`Tier::User`] by default
/// and requires `--all` for [`Tier::Developer`], which keeps bench/probe knobs
/// out of the user-facing listing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    /// Documented, supported, safe for an operator to set.
    User,
    /// Diagnostics, benchmarks, and bring-up switches. Unsupported.
    Developer,
}

/// One declared environment variable.
#[derive(Debug, Clone, Copy)]
pub struct EnvVar {
    pub name: &'static str,
    pub description: &'static str,
    pub tier: Tier,
}

impl EnvVar {
    /// Raw value, or `None` when unset.
    //
    // The one sanctioned `std::env::var` call in the workspace. Anything else
    // adding this `allow` is visible to `rg 'allow\(clippy::disallowed_methods\)'`.
    #[allow(clippy::disallowed_methods)]
    pub fn get(&self) -> Option<String> {
        std::env::var(self.name).ok()
    }

    /// Whether the variable is present at all, regardless of value. Use for
    /// presence-only switches where `FOO=0` should still count as set.
    #[allow(clippy::disallowed_methods)]
    pub fn is_set(&self) -> bool {
        std::env::var_os(self.name).is_some()
    }

    /// Common on/off spelling: `1` / `true` / `on` / `yes`, case-insensitive.
    /// Anything else — including unset — is false.
    pub fn flag(&self) -> bool {
        self.get().is_some_and(|v| {
            matches!(
                v.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "on" | "yes"
            )
        })
    }

    /// Inverse of [`Self::flag`]'s spelling: `0` / `false` / `off` / `no`.
    /// Distinct from `!flag()` because unset must mean "not explicitly off".
    pub fn is_off(&self) -> bool {
        self.get().is_some_and(|v| {
            matches!(
                v.trim().to_ascii_lowercase().as_str(),
                "0" | "false" | "off" | "no"
            )
        })
    }

    /// Parse the value, falling back to `default` when unset or unparseable.
    /// Centralises the `.and_then(|v| v.parse().ok()).unwrap_or(..)` chain that
    /// was previously copy-pasted at every read site.
    pub fn parse_or<T: FromStr>(&self, default: T) -> T {
        self.get()
            .and_then(|v| v.trim().parse::<T>().ok())
            .unwrap_or(default)
    }

    /// Parse the value, or `None` when unset or unparseable.
    pub fn parse<T: FromStr>(&self) -> Option<T> {
        self.get()?.trim().parse::<T>().ok()
    }
}

/// Declare env vars once; expand into constants **and** the [`ALL`] table.
///
/// The two expansions are the point: a read site that names the constant and a
/// CLI that walks `ALL` cannot drift apart, and a variable cannot be documented
/// without being declared or vice versa.
macro_rules! env_vars {
    ($(
        $(#[$meta:meta])*
        $konst:ident = $name:literal, $tier:ident, $desc:literal;
    )*) => {
        $(
            $(#[$meta])*
            #[doc = concat!("`", $name, "` — ", $desc)]
            pub const $konst: EnvVar = EnvVar {
                name: $name,
                description: $desc,
                tier: Tier::$tier,
            };
        )*

        /// Every declared variable, in declaration order.
        pub const ALL: &[EnvVar] = &[$($konst),*];
    };
}

env_vars! {
    // ── Tokenizer / prompt (hipfire-model) ──────────────────────────────────
    NORMALIZE_PROMPT = "HIPFIRE_NORMALIZE_PROMPT", User,
        "Normalise prompts before tokenizing: CRLF to LF, NBSP to space, strip \
         trailing line whitespace, collapse 3+ blank lines. Set 0/false/off/no \
         to disable. Default on.";

    PROMPT_HEAT_JSON = "HIPFIRE_PROMPT_HEAT_JSON", Developer,
        "Emit the prompt heat-class dump as JSON instead of a table. Set 1. \
         Diagnostic for BPE merge-rank distribution of a prompt.";

    PROMPT_HEAT_LIMIT = "HIPFIRE_PROMPT_HEAT_LIMIT", Developer,
        "Maximum tokens listed in the prompt heat dump. Default 64.";

    // ── DeltaNet state (hipfire-arch-qwen35) ────────────────────────────────
    DN_STATE_FP32_BELOW = "HIPFIRE_DN_STATE_FP32_BELOW", Developer,
        "DeltaNet state stays FP32 when head_dim x n_value_heads is below this \
         threshold. Defaults to usize::MAX, so FP32 always. Quantized (Q8/Q4) \
         DeltaNet state is refused by policy — lowering this errors out rather \
         than degrading quality.";

    // ── lm_head (hipfire-runtime, hipfire-quantize) ─────────────────────────
    LMHEAD_TWOSTAGE = "HIPFIRE_LMHEAD_TWOSTAGE", Developer,
        "Two-stage lm_head decode: coarse Q4 shortlist then bf16 rescore. \
         Presets `q4` or `1`; or `<bits>,<topk>` explicitly. Unset disables it \
         and the exact full-precision gemv runs.";

    NO_COARSE_LMHEAD = "HIPFIRE_NO_COARSE_LMHEAD", Developer,
        "Set to any value to stop the quantizer emitting the `<embed>.coarse.weight` \
         shortlist tier. Equivalent to `--no-coarse-lmhead`. Default is to emit it.";

    // ── Dispatch diagnostics (hipfire-dispatch) ─────────────────────────────
    DUMP_HIDDEN = "HIPFIRE_DUMP_HIDDEN", Developer,
        "Path prefix for hidden-state and router-logit dumps written during \
         decode. Unset disables dumping. Diagnostic only — it forces device \
         synchronisation and is slow.";
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_are_unique_and_prefixed() {
        let mut seen = std::collections::BTreeSet::new();
        for v in ALL {
            assert!(
                v.name.starts_with("HIPFIRE_"),
                "{} is not HIPFIRE_-prefixed",
                v.name
            );
            assert!(seen.insert(v.name), "duplicate declaration for {}", v.name);
        }
    }

    #[test]
    fn descriptions_say_more_than_the_name() {
        for v in ALL {
            // Guards against the generated-boilerplate problem the scanner had:
            // 704 of 705 entries restated the variable name as a sentence.
            assert!(
                v.description.len() > 40,
                "{} needs a description a user can act on",
                v.name
            );
            assert!(
                !v.description.contains(v.name),
                "{} description just restates the name",
                v.name
            );
        }
    }

    #[test]
    fn flag_and_is_off_spellings() {
        // Exercised through a variable that is certainly unset.
        const ABSENT: EnvVar = EnvVar {
            name: "HIPFIRE_DEFINITELY_NOT_SET_IN_TESTS",
            description: "test-only",
            tier: Tier::Developer,
        };
        assert!(!ABSENT.flag());
        assert!(!ABSENT.is_off(), "unset must not read as explicitly off");
        assert!(!ABSENT.is_set());
        assert_eq!(ABSENT.parse_or(7usize), 7);
        assert_eq!(ABSENT.parse::<usize>(), None);
    }
}
