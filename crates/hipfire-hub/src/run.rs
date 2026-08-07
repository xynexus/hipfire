// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! `fetch`, `verify`, and `repair`.
//!
//! `verify` and `repair` are the operations that matter on a store with no
//! redundancy: they answer "is what I have still correct" and "replace only
//! what is not", without re-downloading a repo to find out.

use std::path::Path;

use crate::{
    cache::Store, fetch_file, list_files, preflight, sha256_file, Error, RepoFile, Result,
};

/// What a check found for one file.
#[derive(Debug, PartialEq)]
pub enum FileState {
    /// Blob present and its digest matches.
    Good,
    /// Blob present, digest does not match — the failure a read test cannot see.
    Corrupt { want: String, got: String },
    /// Nothing on disk.
    Missing,
    /// Verified, but against the weaker git blob SHA-1 rather than a SHA-256.
    GoodGitOid,
    /// Present, and the hub offers no content hash at all.
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
            // No LFS digest: fall back to the git blob hash, which is still a
            // content hash and still catches a wrong file of the right size.
            None => match (&f.git_oid, store.blob_path_for(&f)) {
                (Some(want), Some(blob)) if blob.exists() => {
                    match crate::git_blob_sha1_file(&blob).await {
                        Ok(got) if &got == want => FileState::GoodGitOid,
                        Ok(got) => FileState::Corrupt {
                            want: want.clone(),
                            got,
                        },
                        Err(Error::Io(e)) => FileState::Unreadable(e.to_string()),
                        Err(e) => return Err(e),
                    }
                }
                (Some(_), Some(_)) => FileState::Missing,
                _ => FileState::LengthOnly,
            },
        };
        out.push((f, state));
    }
    Ok(out)
}

/// Fetch a revision. Existing correct blobs are skipped, not re-downloaded.
pub async fn fetch(
    root: &Path,
    repo: &str,
    revision: &str,
    include: Option<&str>,
) -> Result<usize> {
    let mut files = list_files(repo, revision).await?;
    if let Some(pat) = include {
        files.retain(|f| glob_match(pat, &f.path));
    }
    if files.is_empty() {
        return Err(Error::Fatal(format!("{repo}: no files matched")));
    }
    let store = Store::new(root, repo);
    std::fs::create_dir_all(store.dir().join("blobs"))?;

    // Claim any partial left by a run that died before sweeping the rest:
    // an interrupted transfer is progress, not litter.
    let adopted = store.adopt_orphan_parts().unwrap_or(0);
    if adopted > 0 {
        eprintln!(
            "hub: resuming {:.2} GB left by an interrupted run",
            adopted as f64 / 1e9
        );
    }
    let freed = store.sweep_stale_parts().unwrap_or(0);
    if freed > 0 {
        eprintln!(
            "hub: swept {:.2} GB of unusable .part files",
            freed as f64 / 1e9
        );
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

/// Retry only what retrying can fix, and judge attempts by progress.
///
/// A fixed attempt budget cannot finish a large file on a lossy link: measured
/// here, a 4 GB shard dropped every ~0.3 GB, so four tries exhausted the budget
/// at 0.79 GB however well resume worked. What matters is not how many attempts
/// were spent but whether they are still advancing — so an attempt that grew
/// the partial file resets the budget, and only consecutive *stalled* attempts
/// count toward giving up.
async fn fetch_with_retry(
    repo: &str,
    revision: &str,
    f: &RepoFile,
    store: &Store,
) -> Result<std::path::PathBuf> {
    const MAX_STALLED: u32 = 5;
    /// Backstop against an unbounded loop if the server trickles a byte at a time.
    const MAX_ATTEMPTS: u32 = 200;

    let part = store.part_path(&f.path);
    let progress = || std::fs::metadata(&part).map(|m| m.len()).unwrap_or(0);

    let mut stalled = 0u32;
    let mut delay = std::time::Duration::from_secs(1);
    for attempt in 1..=MAX_ATTEMPTS {
        let before = progress();
        match fetch_file(repo, revision, f, store).await {
            Ok(p) => return Ok(p),
            Err(Error::Retryable(m)) => {
                let after = progress();
                if after > before {
                    // Still advancing: this is a lossy link, not a broken one.
                    stalled = 0;
                    delay = std::time::Duration::from_secs(1);
                } else {
                    stalled += 1;
                    if stalled >= MAX_STALLED {
                        return Err(Error::Retryable(format!(
                            "{}: {MAX_STALLED} consecutive attempts made no progress ({m})",
                            f.path
                        )));
                    }
                    delay = (delay * 2).min(std::time::Duration::from_secs(30));
                }
                if attempt % 5 == 0 || after == before {
                    eprintln!(
                        "hub: {} at {:.2} GB — attempt {attempt} ({m})",
                        f.path,
                        after as f64 / 1e9
                    );
                }
                tokio::time::sleep(delay).await;
            }
            // A digest mismatch is not transient noise: the bytes on the wire
            // were wrong. Surface it rather than papering over it with retries.
            Err(e) => return Err(e),
        }
    }
    Err(Error::Retryable(format!(
        "{}: gave up after {MAX_ATTEMPTS} attempts",
        f.path
    )))
}

/// [`fetch_with_retry`] for a streamed transfer.
///
/// Same rule — only consecutive *stalled* attempts count toward giving up — but
/// progress is measured from the resume state rather than a `.part` on disk,
/// because a streamed transfer leaves no file to measure.
pub async fn fetch_file_streamed_with_retry(
    repo: &str,
    revision: &str,
    f: &RepoFile,
    st: &mut crate::StreamProgress,
    sink: &mut dyn crate::ByteSink,
) -> Result<()> {
    const MAX_STALLED: u32 = 5;
    const MAX_ATTEMPTS: u32 = 200;

    let mut stalled = 0u32;
    let mut delay = std::time::Duration::from_secs(1);
    for attempt in 1..=MAX_ATTEMPTS {
        let before = st.consumed();
        match crate::fetch_file_streamed(crate::HUB, repo, revision, f, st, sink).await {
            Ok(()) => return Ok(()),
            Err(Error::Retryable(m)) => {
                let after = st.consumed();
                if after > before {
                    stalled = 0;
                    delay = std::time::Duration::from_secs(1);
                } else {
                    stalled += 1;
                    if stalled >= MAX_STALLED {
                        return Err(Error::Retryable(format!(
                            "{}: {MAX_STALLED} consecutive attempts made no progress ({m})",
                            f.path
                        )));
                    }
                    delay = (delay * 2).min(std::time::Duration::from_secs(30));
                }
                if attempt % 5 == 0 || after == before {
                    eprintln!(
                        "hub: {} at {:.2} GB — attempt {attempt} ({m})",
                        f.path,
                        after as f64 / 1e9
                    );
                }
                tokio::time::sleep(delay).await;
            }
            Err(e) => return Err(e),
        }
    }
    Err(Error::Retryable(format!(
        "{}: gave up after {MAX_ATTEMPTS} attempts",
        f.path
    )))
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
