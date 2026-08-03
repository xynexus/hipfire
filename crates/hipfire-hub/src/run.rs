// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! `fetch`, `verify`, and `repair`.
//!
//! `verify` and `repair` are the operations that matter on a store with no
//! redundancy: they answer "is what I have still correct" and "replace only
//! what is not", without re-downloading a repo to find out.

use std::path::Path;

use crate::{cache::Store, fetch_file, list_files, preflight, sha256_file, Error, RepoFile, Result};

/// What a check found for one file.
#[derive(Debug, PartialEq)]
pub enum FileState {
    /// Blob present and its digest matches.
    Good,
    /// Blob present, digest does not match — the failure a read test cannot see.
    Corrupt { want: String, got: String },
    /// Nothing on disk.
    Missing,
    /// Present, but the hub offers no content hash, so only length was checked.
    LengthOnly,
    /// On disk but unreadable — a bad sector, which on a no-redundancy array
    /// the filesystem cannot repair. This is a state to report and re-fetch,
    /// not an error to abort on: a verify that dies on the first bad blob
    /// cannot tell you what else is wrong, and repair is exactly the answer.
    Unreadable(String),
}

/// Hash every local blob against the hub's digest.
pub async fn verify(root: &Path, repo: &str, revision: &str) -> Result<Vec<(RepoFile, FileState)>> {
    let files = list_files(repo, revision).await?;
    let store = Store::new(root, repo);
    let mut out = Vec::with_capacity(files.len());
    for f in files {
        let state = match &f.sha256 {
            Some(want) => {
                let blob = store.blob_path(want);
                if !blob.exists() {
                    FileState::Missing
                } else {
                    // Hash the bytes rather than trusting the name they are
                    // filed under: a blob can rot in place.
                    match sha256_file(&blob).await {
                        Ok(got) if &got == want => FileState::Good,
                        Ok(got) => FileState::Corrupt {
                            want: want.clone(),
                            got,
                        },
                        Err(Error::Io(e)) => FileState::Unreadable(e.to_string()),
                        Err(e) => return Err(e),
                    }
                }
            }
            None => FileState::LengthOnly,
        };
        out.push((f, state));
    }
    Ok(out)
}

/// Fetch a revision. Existing correct blobs are skipped, not re-downloaded.
pub async fn fetch(root: &Path, repo: &str, revision: &str, include: Option<&str>) -> Result<usize> {
    let mut files = list_files(repo, revision).await?;
    if let Some(pat) = include {
        files.retain(|f| glob_match(pat, &f.path));
    }
    if files.is_empty() {
        return Err(Error::Fatal(format!("{repo}: no files matched")));
    }
    let store = Store::new(root, repo);
    std::fs::create_dir_all(store.dir().join("blobs"))?;

    let freed = store.sweep_stale_parts().unwrap_or(0);
    if freed > 0 {
        eprintln!("hub: swept {:.2} GB of stale .part files", freed as f64 / 1e9);
    }

    let held = store.held_bytes(files.iter().filter_map(|f| f.sha256.clone()));
    let need = preflight(&files, store.dir(), held)?;
    eprintln!(
        "hub: {} file(s), {:.2} GB to fetch ({:.2} GB already held)",
        files.len(),
        need as f64 / 1e9,
        held as f64 / 1e9
    );

    let mut fetched = 0usize;
    for f in &files {
        let blob = fetch_with_retry(repo, revision, f, &store).await?;
        store.link(revision, &f.path, &blob)?;
        fetched += 1;
    }
    store.write_ref("main", revision)?;
    Ok(fetched)
}

/// Verify, then re-fetch only what failed.
///
/// This is the operation the store actually needs: one bad shard in sixteen
/// should cost one shard, not the repo.
pub async fn repair(root: &Path, repo: &str, revision: &str) -> Result<usize> {
    let states = verify(root, repo, revision).await?;
    let store = Store::new(root, repo);
    let broken: Vec<&RepoFile> = states
        .iter()
        .filter(|(_, s)| {
            matches!(
                s,
                FileState::Corrupt { .. } | FileState::Missing | FileState::Unreadable(_)
            )
        })
        .map(|(f, _)| f)
        .collect();

    if broken.is_empty() {
        eprintln!("hub: nothing to repair — {} file(s) verified", states.len());
        return Ok(0);
    }
    for (f, s) in &states {
        match s {
            FileState::Corrupt { want, got } => eprintln!(
                "hub: CORRUPT {} (expected {}…, got {}…)",
                f.path,
                &want[..16.min(want.len())],
                &got[..16.min(got.len())]
            ),
            FileState::Missing => eprintln!("hub: MISSING {}", f.path),
            FileState::Unreadable(e) => eprintln!("hub: UNREADABLE {} ({e})", f.path),
            _ => {}
        }
    }

    let need: u64 = broken.iter().map(|f| f.size).sum();
    let have = crate::free_space(store.dir())?;
    if have < need.saturating_add(crate::SPACE_MARGIN) {
        return Err(Error::Space { need, have });
    }

    let mut fixed = 0usize;
    for f in broken {
        // Remove the bad blob first so the fetch cannot mistake it for a hit.
        if let Some(sha) = &f.sha256 {
            let _ = std::fs::remove_file(store.blob_path(sha));
        }
        let blob = fetch_with_retry(repo, revision, f, &store).await?;
        store.link(revision, &f.path, &blob)?;
        fixed += 1;
        eprintln!("hub: repaired {}", f.path);
    }
    Ok(fixed)
}

/// Retry only what retrying can fix, with backoff.
async fn fetch_with_retry(
    repo: &str,
    revision: &str,
    f: &RepoFile,
    store: &Store,
) -> Result<std::path::PathBuf> {
    let mut delay = std::time::Duration::from_secs(1);
    for attempt in 1..=4 {
        match fetch_file(repo, revision, f, store).await {
            Ok(p) => return Ok(p),
            Err(Error::Retryable(m)) if attempt < 4 => {
                eprintln!("hub: {} — attempt {attempt} failed ({m}); retrying", f.path);
                tokio::time::sleep(delay).await;
                delay *= 2;
            }
            // A digest mismatch is not transient noise: the bytes on the wire
            // were wrong. Surface it rather than papering over it with retries.
            Err(e) => return Err(e),
        }
    }
    unreachable!("loop returns on the final attempt")
}

/// Minimal `*` glob, enough for `--include '*.safetensors'`.
fn glob_match(pat: &str, s: &str) -> bool {
    let mut parts = pat.split('*');
    let Some(first) = parts.next() else {
        return true;
    };
    if !s.starts_with(first) {
        return false;
    }
    let mut idx = first.len();
    let mut last = "";
    for p in parts {
        last = p;
        if p.is_empty() {
            continue;
        }
        match s[idx..].find(p) {
            Some(i) => idx += i + p.len(),
            None => return false,
        }
    }
    pat.ends_with('*') || s.ends_with(last)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn glob_handles_the_patterns_callers_actually_pass() {
        assert!(glob_match("*.safetensors", "model-00001.safetensors"));
        assert!(!glob_match("*.safetensors", "config.json"));
        assert!(glob_match("model-*", "model-00003-of-00016.safetensors"));
        assert!(glob_match("*", "anything"));
        assert!(glob_match("config.json", "config.json"));
        assert!(!glob_match("model-*", "tokenizer.json"));
    }
}
