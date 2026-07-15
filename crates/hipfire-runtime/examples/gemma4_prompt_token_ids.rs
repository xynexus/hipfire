// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! Validate Hipfire tokenization against pinned Gemma 4 prompt fixtures.
//!
//! Usage:
//! `gemma4_prompt_token_ids TOKENIZER.json FIXTURE.json [FIXTURE.json ...]`

use hipfire_runtime::tokenizer::Tokenizer;
use std::path::{Path, PathBuf};

fn first_difference(expected: &[u32], actual: &[u32]) -> Option<usize> {
    let shared = expected.len().min(actual.len());
    expected[..shared]
        .iter()
        .zip(&actual[..shared])
        .position(|(a, b)| a != b)
        .or_else(|| (expected.len() != actual.len()).then_some(shared))
}

fn validate_fixture(tokenizer: &Tokenizer, path: &Path) -> Result<(usize, usize), String> {
    let raw = std::fs::read_to_string(path)
        .map_err(|error| format!("read {}: {error}", path.display()))?;
    let fixture: serde_json::Value =
        serde_json::from_str(&raw).map_err(|error| format!("parse {}: {error}", path.display()))?;
    let cases = fixture
        .get("cases")
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| format!("{} has no object-valued `cases`", path.display()))?;
    if cases.is_empty() {
        return Err(format!("{} contains zero prompt cases", path.display()));
    }

    let mut token_count = 0usize;
    for (name, case) in cases {
        let rendered = case
            .get("rendered")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| format!("{} case {name}: missing rendered text", path.display()))?;
        let expected = case
            .get("token_ids")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| format!("{} case {name}: missing token_ids", path.display()))?
            .iter()
            .map(|value| {
                value
                    .as_u64()
                    .and_then(|value| u32::try_from(value).ok())
                    .ok_or_else(|| {
                        format!("{} case {name}: invalid token id {value}", path.display())
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let actual = tokenizer.encode(rendered);
        if let Some(index) = first_difference(&expected, &actual) {
            let start = index.saturating_sub(4);
            let expected_end = (index + 5).min(expected.len());
            let actual_end = (index + 5).min(actual.len());
            return Err(format!(
                "{} case {name}: token IDs differ at index {index} (expected {:?}, actual {:?}; lengths {} vs {}); expected[{}..{}]={:?} -> {:?}; actual[{}..{}]={:?} -> {:?}",
                path.display(),
                expected.get(index),
                actual.get(index),
                expected.len(),
                actual.len(),
                start,
                expected_end,
                &expected[start..expected_end],
                tokenizer.decode(&expected[start..expected_end]),
                start,
                actual_end,
                &actual[start..actual_end],
                tokenizer.decode(&actual[start..actual_end]),
            ));
        }
        token_count += actual.len();
    }
    Ok((cases.len(), token_count))
}

fn main() -> Result<(), String> {
    let args = std::env::args_os()
        .skip(1)
        .map(PathBuf::from)
        .collect::<Vec<_>>();
    if args.len() < 2 {
        return Err(
            "usage: gemma4_prompt_token_ids TOKENIZER.json FIXTURE.json [FIXTURE.json ...]"
                .to_string(),
        );
    }
    let tokenizer = Tokenizer::from_tokenizer_json(&args[0])
        .map_err(|error| format!("load {}: {error}", args[0].display()))?
        .ok_or_else(|| format!("tokenizer file {} was not found", args[0].display()))?;

    let mut cases = 0usize;
    let mut tokens = 0usize;
    for fixture in &args[1..] {
        let (fixture_cases, fixture_tokens) = validate_fixture(&tokenizer, fixture)?;
        cases += fixture_cases;
        tokens += fixture_tokens;
    }
    println!(
        "gemma4_prompt_token_ids: PASS (fixtures={} cases={cases} tokens={tokens})",
        args.len() - 1
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::first_difference;

    #[test]
    fn reports_value_and_length_mismatches() {
        assert_eq!(first_difference(&[1, 2, 3], &[1, 9, 3]), Some(1));
        assert_eq!(first_difference(&[1, 2], &[1, 2, 3]), Some(2));
        assert_eq!(first_difference(&[1, 2], &[1, 2]), None);
    }
}
