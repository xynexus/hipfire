// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! Source-keyed chunk tables: BLAKE3-256 over fixed windows of a file.
//!
//! The point is repair granularity. A whole-file digest answers "is this file
//! correct", which on a store with no redundancy leaves only one remedy —
//! refetch all of it. A chunk table answers *which bytes*, and those map
//! straight onto a `Range` request against the origin.
//!
//! # Why the window is keyed to the source
//!
//! Windows are offsets into the file as the hub serves it. That is the only
//! keying a repair can act on, and it survives re-encoding: an `.hfa` that
//! re-packs the same checkpoint under different codec settings still describes
//! the same source file, where a table over stored payloads would be
//! invalidated wholesale.
//!
//! # Why BLAKE3 rather than a fast non-cryptographic hash
//!
//! Detecting a bad sector needs only an error-detecting hash, and the archive's
//! per-payload XXH3 is exactly that. This table has a second job: it is the
//! basis for trusting a *range* fetched from elsewhere before the whole file
//! has been seen, and later for a signed index that authenticates payloads
//! transitively. That needs collision resistance.
//!
//! Measured on halo (Strix Halo, 2 GiB of real BF16): XXH3 37.0 GB/s, BLAKE3
//! 9.2 GB/s single-threaded. BLAKE3 is 4x slower and it does not matter — it is
//! 99x the array's 93 MB/s and 18x the bf16 encoder, so it cannot become the
//! bottleneck in any stage that consumes it.

use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

/// Window size for new tables.
///
/// 4 MiB keeps a repair request far below the size at which this link drops a
/// body, while keeping the table small — a 17.7 GB model is ~4.2k hashes, or
/// 132 KB once base64'd.
pub const CHUNK: usize = 4 << 20;

/// One window that does not match its recorded hash.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkMiss {
    pub index: usize,
    /// Byte offset in the source file — the `from` of a `Range` request.
    pub at: u64,
    /// Bytes to fetch. Zero when the file ends before this window starts, i.e.
    /// the file is truncated and the whole window is missing.
    pub len: u64,
}

/// Rolling BLAKE3-256 over fixed windows of a file's bytes.
///
/// Callers must supply the file's bytes in order and in full: a window boundary
/// is a file offset, not a boundary of whatever buffer happened to arrive.
pub struct ChunkHasher {
    size: usize,
    cur: blake3::Hasher,
    filled: usize,
    hashes: Vec<[u8; 32]>,
    total: u64,
}

impl Default for ChunkHasher {
    fn default() -> Self {
        Self::new()
    }
}

impl ChunkHasher {
    pub fn new() -> Self {
        Self::with_size(CHUNK)
    }

    /// Verification must re-hash at the window the table was WRITTEN with, not
    /// whatever [`CHUNK`] happens to be today. Otherwise changing the constant
    /// silently invalidates every table already on the array.
    pub fn with_size(size: usize) -> Self {
        Self {
            size: size.max(1),
            cur: blake3::Hasher::new(),
            filled: 0,
            hashes: Vec::new(),
            total: 0,
        }
    }

    pub fn update(&mut self, mut bytes: &[u8]) {
        self.total += bytes.len() as u64;
        while !bytes.is_empty() {
            let take = (self.size - self.filled).min(bytes.len());
            self.cur.update(&bytes[..take]);
            self.filled += take;
            bytes = &bytes[take..];
            if self.filled == self.size {
                self.hashes.push(*self.cur.finalize().as_bytes());
                self.cur = blake3::Hasher::new();
                self.filled = 0;
            }
        }
    }

    /// Flush the trailing partial window. A file whose length is not a multiple
    /// of the window ends in a short chunk, hashed over its real length rather
    /// than padded — padding would let two different files collide.
    pub fn finish(mut self) -> ChunkTable {
        if self.filled > 0 {
            self.hashes.push(*self.cur.finalize().as_bytes());
        }
        ChunkTable {
            size: self.size,
            src_len: self.total,
            hashes: self.hashes,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkTable {
    size: usize,
    src_len: u64,
    hashes: Vec<[u8; 32]>,
}

impl ChunkTable {
    pub fn size(&self) -> usize {
        self.size
    }

    /// Bytes the table covers. Callers that build one incrementally check this
    /// against the file's real length; a short table means the producer failed
    /// to feed every byte, and every window after the gap is at a wrong offset.
    pub fn src_len(&self) -> u64 {
        self.src_len
    }

    pub fn hashes(&self) -> &[[u8; 32]] {
        &self.hashes
    }

    /// Hash a file that already exists, in one pass.
    pub fn of_file(path: &Path, size: usize) -> std::io::Result<ChunkTable> {
        let mut f = std::fs::File::open(path)?;
        let mut h = ChunkHasher::with_size(size);
        let mut buf = vec![0u8; size.min(8 << 20)];
        loop {
            let n = f.read(&mut buf)?;
            if n == 0 {
                break;
            }
            h.update(&buf[..n]);
        }
        Ok(h.finish())
    }

    /// Windows of `path` that do not match this table.
    ///
    /// A short final window is compared over its real length, so a truncated
    /// file reports its tail as missing rather than reading past the end.
    pub fn mismatched(&self, path: &Path) -> std::io::Result<Vec<ChunkMiss>> {
        let mut f = std::fs::File::open(path)?;
        let total = f.metadata()?.len();
        let mut out = Vec::new();
        let mut buf = vec![0u8; self.size];
        for (index, want) in self.hashes.iter().enumerate() {
            let at = (index as u64) * self.size as u64;
            if at >= total {
                out.push(ChunkMiss { index, at, len: 0 });
                continue;
            }
            let n = (total - at).min(self.size as u64) as usize;
            f.seek(SeekFrom::Start(at))?;
            f.read_exact(&mut buf[..n])?;
            if blake3::hash(&buf[..n]).as_bytes() != want {
                out.push(ChunkMiss {
                    index,
                    at,
                    len: n as u64,
                });
            }
        }
        Ok(out)
    }

    /// The byte range a window occupies, clamped to the file's recorded length.
    pub fn range(&self, index: usize) -> (u64, u64) {
        let at = (index as u64) * self.size as u64;
        (
            at,
            (self.src_len - at.min(self.src_len)).min(self.size as u64),
        )
    }

    pub fn to_json(&self) -> serde_json::Value {
        use base64::Engine as _;
        let mut flat = Vec::with_capacity(self.hashes.len() * 32);
        for h in &self.hashes {
            flat.extend_from_slice(h);
        }
        serde_json::json!({
            "algo": "blake3-256",
            "size": self.size,
            "src_len": self.src_len,
            // Packed then base64'd rather than an array of hex strings: 4.2k
            // hashes cost 132 KB this way against roughly 270 KB as JSON text,
            // and the decoded blob slices by index without parsing.
            "hashes": base64::engine::general_purpose::STANDARD.encode(&flat),
        })
    }

    /// `None` for anything this build cannot act on — a missing table, a
    /// different algorithm, a malformed blob. Not an error: it just means the
    /// caller falls back to whole-file verification.
    pub fn from_json(v: &serde_json::Value) -> Option<ChunkTable> {
        use base64::Engine as _;
        if v.get("algo").and_then(|a| a.as_str())? != "blake3-256" {
            return None;
        }
        let size = v.get("size").and_then(|s| s.as_u64())? as usize;
        let src_len = v.get("src_len").and_then(|s| s.as_u64())?;
        let raw = base64::engine::general_purpose::STANDARD
            .decode(v.get("hashes").and_then(|h| h.as_str())?)
            .ok()?;
        if size == 0 || raw.len() % 32 != 0 {
            return None;
        }
        Some(ChunkTable {
            size,
            src_len,
            hashes: raw
                .chunks_exact(32)
                .map(|c| c.try_into().expect("chunks_exact(32)"))
                .collect(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data(n: usize, seed: u64) -> Vec<u8> {
        let mut s = seed | 1;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (s >> 24) as u8
            })
            .collect()
    }

    /// Windows are file offsets, not buffer boundaries: the on-disk packer feeds
    /// a length prefix then a header then pieces, while a socket feeds whatever
    /// arrived. Both must produce the same table.
    #[test]
    fn feeding_order_does_not_change_the_table() {
        let d = data(40_000, 3);
        let whole = {
            let mut h = ChunkHasher::with_size(4096);
            h.update(&d);
            h.finish()
        };
        let mut odd = ChunkHasher::with_size(4096);
        let (mut i, mut step) = (0usize, 1usize);
        while i < d.len() {
            let n = step.min(d.len() - i);
            odd.update(&d[i..i + n]);
            i += n;
            step = step * 3 + 1;
        }
        assert!(whole.hashes().len() > 1, "fixture is a single window");
        assert_eq!(whole, odd.finish());
    }

    #[test]
    fn json_round_trips() {
        let d = data(9_000, 5);
        let mut h = ChunkHasher::with_size(1024);
        h.update(&d);
        let t = h.finish();
        assert_eq!(ChunkTable::from_json(&t.to_json()).expect("round trip"), t);
        // A table this build cannot act on is declined, not guessed at.
        assert!(ChunkTable::from_json(&serde_json::json!({"algo": "xxh3-64"})).is_none());
    }
}
