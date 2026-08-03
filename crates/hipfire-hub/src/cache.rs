// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! HuggingFace-compatible cache layout.
//!
//! `blobs/<sha256>` holds content, `snapshots/<rev>/<path>` are symlinks into
//! it, and `refs/main` names the current revision. Matching the layout is not
//! cosmetic: everything already pointing at the store keeps working, and the
//! content-addressed blobs give file-level dedup for free — the same file
//! appearing in two revisions is stored once.
//!
//! That is the only dedup worth having here. Chunk-level dedup was evaluated
//! and rejected: sibling finetunes share their tokenizer and nothing else —
//! measured at one 4.6 MB blob between LFM2.5-1.2B-Instruct and -Thinking, with
//! no weight bytes in common.

use std::path::{Path, PathBuf};

/// A model's directory inside the cache root.
pub struct Store {
    root: PathBuf,
}

impl Store {
    /// `root` is the cache root (e.g. `/srv/huggingface`); `repo` is `org/name`.
    pub fn new(root: &Path, repo: &str) -> Self {
        Store {
            root: root.join(format!("models--{}", repo.replace('/', "--"))),
        }
    }

    pub fn dir(&self) -> &Path {
        &self.root
    }

    pub fn blob_path(&self, sha256: &str) -> PathBuf {
        self.root.join("blobs").join(sha256)
    }

    /// PID-scoped so two runs against the same store cannot adopt each other's
    /// partial writes. A `.part` from a dead process is inert rather than
    /// dangerous: it will never be renamed into place by anyone else.
    pub fn part_path(&self, rel: &str) -> PathBuf {
        let flat = rel.replace('/', "__");
        self.root
            .join("blobs")
            .join(format!(".{flat}.{}.part", std::process::id()))
    }

    pub fn snapshot_dir(&self, revision: &str) -> PathBuf {
        self.root.join("snapshots").join(revision)
    }

    /// Point `snapshots/<rev>/<rel>` at a blob, relative so the tree stays
    /// movable.
    pub fn link(&self, revision: &str, rel: &str, blob: &Path) -> std::io::Result<()> {
        let link = self.snapshot_dir(revision).join(rel);
        if let Some(p) = link.parent() {
            std::fs::create_dir_all(p)?;
        }
        let depth = Path::new(rel).components().count();
        let mut up = PathBuf::new();
        for _ in 0..depth {
            up.push("..");
        }
        let target = up.join("..").join("blobs").join(
            blob.file_name()
                .unwrap_or_else(|| std::ffi::OsStr::new("unknown")),
        );
        // Replace rather than fail: a re-fetch of a repaired file must be able
        // to re-point the link.
        if link.exists() || std::fs::symlink_metadata(&link).is_ok() {
            std::fs::remove_file(&link)?;
        }
        std::os::unix::fs::symlink(target, link)
    }

    pub fn write_ref(&self, name: &str, revision: &str) -> std::io::Result<()> {
        let refs = self.root.join("refs");
        std::fs::create_dir_all(&refs)?;
        std::fs::write(refs.join(name), revision)
    }

    /// Bytes already held as blobs, so a fetch can size only what is missing.
    pub fn held_bytes(&self, shas: impl Iterator<Item = String>) -> u64 {
        shas.filter_map(|s| std::fs::metadata(self.blob_path(&s)).ok())
            .map(|m| m.len())
            .sum()
    }

    /// Remove `.part` files left by processes that are no longer running.
    ///
    /// They are already harmless — nothing renames another PID's partial into
    /// place — but they occupy space until something sweeps them.
    pub fn sweep_stale_parts(&self) -> std::io::Result<u64> {
        let blobs = self.root.join("blobs");
        let Ok(rd) = std::fs::read_dir(&blobs) else {
            return Ok(0);
        };
        let mut freed = 0u64;
        for e in rd.flatten() {
            let name = e.file_name();
            let Some(name) = name.to_str() else { continue };
            if !name.starts_with('.') || !name.ends_with(".part") {
                continue;
            }
            let Some(pid) = name
                .trim_end_matches(".part")
                .rsplit('.')
                .next()
                .and_then(|p| p.parse::<i32>().ok())
            else {
                continue;
            };
            // `kill` treats 0 as "this process group" and negatives as a
            // group id, so only a positive pid is a process to probe. Without
            // this guard a `.part` tagged 0 looks permanently alive and is
            // never reclaimed.
            // SAFETY: signal 0 only probes for existence, it delivers nothing.
            let alive = pid > 0 && unsafe { libc::kill(pid, 0) } == 0;
            if !alive {
                if let Ok(m) = e.metadata() {
                    freed += m.len();
                }
                let _ = std::fs::remove_file(e.path());
            }
        }
        Ok(freed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_matches_the_hugging_face_cache() {
        let s = Store::new(Path::new("/srv/huggingface"), "Qwen/Qwen3-30B");
        assert_eq!(
            s.dir(),
            Path::new("/srv/huggingface/models--Qwen--Qwen3-30B")
        );
        assert_eq!(
            s.blob_path("abc"),
            Path::new("/srv/huggingface/models--Qwen--Qwen3-30B/blobs/abc")
        );
    }

    /// Two runs must not collide on the same partial file.
    #[test]
    fn part_paths_are_pid_scoped_and_flatten_subdirs() {
        let s = Store::new(Path::new("/tmp"), "org/name");
        let p = s.part_path("a/b/model.safetensors");
        let f = p.file_name().unwrap().to_str().unwrap();
        assert!(f.starts_with(".a__b__model.safetensors."), "{f}");
        assert!(f.ends_with(&format!(".{}.part", std::process::id())), "{f}");
    }
}
