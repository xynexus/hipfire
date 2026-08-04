// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! `hipfire env` — list the environment variables hipfire reads.
//!
//! Walks [`hipfire_env::ALL`], the same table the read sites use, so a variable
//! cannot appear here without being readable in the code or vice versa. This is
//! the consumer half of the declare-once design in `crates/hipfire-env`.

use clap::Args;
use hipfire_env::{EnvVar, Tier};

#[derive(Args, Debug)]
pub struct EnvArgs {
    /// Show developer/diagnostic variables too. Default lists only the
    /// supported, user-facing ones.
    #[arg(long)]
    pub all: bool,

    /// Case-insensitive substring filter over name and description.
    pub filter: Option<String>,
}

pub fn run(args: EnvArgs) -> anyhow::Result<()> {
    let needle = args.filter.as_ref().map(|f| f.to_ascii_lowercase());
    let matches: Vec<&EnvVar> = hipfire_env::ALL
        .iter()
        .filter(|v| args.all || v.tier == Tier::User)
        .filter(|v| {
            needle.as_ref().is_none_or(|n| {
                v.name.to_ascii_lowercase().contains(n)
                    || v.description.to_ascii_lowercase().contains(n)
            })
        })
        .collect();

    if matches.is_empty() {
        match (&args.filter, args.all) {
            (Some(f), _) => println!("no environment variable matches {f:?}"),
            (None, false) => println!("no user-facing environment variables declared"),
            (None, true) => println!("no environment variables declared"),
        }
        return Ok(());
    }

    // Wrap descriptions against the terminal so long entries stay readable
    // rather than relying on the terminal's own hard wrap mid-word.
    let width = std::env::var("COLUMNS")
        .ok()
        .and_then(|c| c.parse::<usize>().ok())
        .unwrap_or(100)
        .clamp(60, 160);

    for v in &matches {
        let tag = match v.tier {
            Tier::User => "",
            Tier::Developer => "  [developer]",
        };
        println!("{}{}", v.name, tag);
        for line in wrap(v.description, width.saturating_sub(4)) {
            println!("    {line}");
        }
        println!();
    }

    let total = hipfire_env::ALL.len();
    let hidden = total
        - hipfire_env::ALL
            .iter()
            .filter(|v| v.tier == Tier::User)
            .count();
    if !args.all && hidden > 0 {
        println!(
            "{} shown; {hidden} developer variable(s) hidden — pass --all to include them.",
            matches.len()
        );
    } else {
        println!("{} of {total} declared variable(s) shown.", matches.len());
    }
    Ok(())
}

/// Greedy word wrap. Deliberately not a dependency — descriptions are short and
/// this is the only place that needs it.
fn wrap(text: &str, width: usize) -> Vec<String> {
    let mut out = Vec::new();
    let mut line = String::new();
    for word in text.split_whitespace() {
        if !line.is_empty() && line.len() + 1 + word.len() > width {
            out.push(std::mem::take(&mut line));
        }
        if !line.is_empty() {
            line.push(' ');
        }
        line.push_str(word);
    }
    if !line.is_empty() {
        out.push(line);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wrap_respects_width_and_keeps_all_words() {
        let text = "alpha beta gamma delta epsilon zeta eta theta";
        let lines = wrap(text, 20);
        assert!(lines.iter().all(|l| l.len() <= 20), "{lines:?}");
        assert_eq!(lines.join(" "), text, "wrapping must not drop or reorder");
    }

    #[test]
    fn wrap_does_not_split_a_word_longer_than_width() {
        let lines = wrap("short verylongsingletokenthatexceeds", 10);
        assert!(lines.iter().any(|l| l.contains("verylongsingletoken")));
    }

    #[test]
    fn every_declared_var_is_listable() {
        // Guards the consumer against a table entry it cannot render.
        for v in hipfire_env::ALL {
            assert!(
                !wrap(v.description, 96).is_empty(),
                "{} renders empty",
                v.name
            );
        }
    }
}
