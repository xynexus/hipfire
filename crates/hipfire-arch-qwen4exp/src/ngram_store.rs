// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! On-disk reader for the n-gram embedding table.
//!
//! The table is ~102 GB at source width — 41% of the model's parameters — and one
//! token reads exactly `heads` rows of it. Keeping it on drive rather than resident
//! is what makes the model fit, and it costs nothing: the row indices are a pure
//! function of already-committed token ids (see [`crate::ngram`]), so the reads are
//! issuable before the forward pass starts and overlap it entirely.
//!
//! # Why 4 KiB blocks
//!
//! A row read costs one filesystem sector however few bytes it wants. Measured on
//! a Samsung 990 PRO at queue depth 16, over the real archive: a 320-byte read and
//! a 4096-byte block-aligned read cost the same (10.5 vs 11.0 µs), and even 16 KiB
//! is 11.7 µs. So the block IS the I/O unit, and any packing that fits inside it is
//! free.
//!
//! Making the block the FORMAT unit buys three things, in increasing order of
//! importance: disk space; a block cache that holds more rows per byte; and — the
//! real prize — `O_DIRECT`, because 4096-aligned, 4096-sized reads are exactly what
//! it requires. On a unified-memory APU the page cache and GPU memory are one pool,
//! so keeping 102 GB of scattered reads out of the page cache is not a nicety.
//!
//! # Layout
//!
//! ```text
//! index    u64 x (n_blocks + 1) — first row of each block, then the row total.
//!                                 Monotonic, so a row resolves by binary search.
//! blocks   4096 B each, 4096-aligned, VARIABLE row count:
//!            [ 2 B] u16 n_rows
//!            [ 2 B] u16 codec         (0 = raw)
//!            [4n B] u32 byte offset of each row, from the block start
//!            [   ] row payloads, packed until the next row would not fit
//! ```
//!
//! Rows per block is variable and an index maps row → block. That is what removes
//! the overflow case a fixed count would need: a block whose rows encode badly
//! simply holds fewer of them, so the format cannot fail on unexpected data, and no
//! measurement of the payload's compressibility is needed up front. A row never
//! straddles a block, so a lookup is always exactly one aligned read.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::Result as IoResult;
use std::path::Path;

/// The I/O and format unit.
pub const BLOCK: usize = 4096;
const HDR: usize = 4;

/// Intra-block row coding. Only `Raw` exists today; the field is present so a
/// packed coding can be added without moving the addressing, which is the part
/// that is expensive to get wrong.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u16)]
pub enum Codec {
    Raw = 0,
}

impl Codec {
    fn from_u16(v: u16) -> Option<Self> {
        match v {
            0 => Some(Codec::Raw),
            _ => None,
        }
    }
}

/// Pack rows into 4 KiB blocks, returning `(blocks, index)`.
///
/// `index` has `n_blocks + 1` entries: the first row of each block, then the total
/// row count, so `index[b + 1] - index[b]` is block `b`'s row count without a
/// special case for the last block.
pub fn pack(rows: &[Vec<u8>]) -> (Vec<u8>, Vec<u64>) {
    let mut blocks: Vec<u8> = Vec::new();
    let mut index: Vec<u64> = vec![0];
    let mut i = 0usize;
    while i < rows.len() {
        let start = i;
        // Grow the block while the next row still fits alongside its offset entry.
        let mut payload = 0usize;
        let mut n = 0usize;
        while i < rows.len() {
            let need = HDR + (n + 1) * 4 + payload + rows[i].len();
            if need > BLOCK {
                break;
            }
            payload += rows[i].len();
            n += 1;
            i += 1;
        }
        assert!(
            n > 0,
            "a single row must fit in one block (row {start} is too large)"
        );
        let mut b = vec![0u8; BLOCK];
        b[0..2].copy_from_slice(&(n as u16).to_le_bytes());
        b[2..4].copy_from_slice(&(Codec::Raw as u16).to_le_bytes());
        let mut off = HDR + n * 4;
        for (slot, r) in rows[start..start + n].iter().enumerate() {
            let o = HDR + slot * 4;
            b[o..o + 4].copy_from_slice(&(off as u32).to_le_bytes());
            b[off..off + r.len()].copy_from_slice(r);
            off += r.len();
        }
        blocks.extend_from_slice(&b);
        index.push((start + n) as u64);
    }
    (blocks, index)
}

/// A block-addressed table on disk.
pub struct NgramStore {
    file: File,
    /// Byte offset of block 0 within the file. Must be 4096-aligned for `O_DIRECT`.
    base: u64,
    /// First row of each block, plus the row total. Monotonic.
    index: Vec<u64>,
    row_bytes: usize,
}

impl NgramStore {
    /// `index` is the packer's output; `base` must be 4096-aligned.
    pub fn new(file: File, base: u64, index: Vec<u64>, row_bytes: usize) -> Result<Self, String> {
        if base % BLOCK as u64 != 0 {
            return Err(format!("base offset {base} is not {BLOCK}-aligned"));
        }
        if index.len() < 2 {
            return Err("index needs at least one block".into());
        }
        if index.windows(2).any(|w| w[1] <= w[0]) {
            return Err("index must be strictly increasing".into());
        }
        Ok(Self {
            file,
            base,
            index,
            row_bytes,
        })
    }

    pub fn open(path: &Path, base: u64, index: Vec<u64>, row_bytes: usize) -> IoResult<Self> {
        let f = File::open(path)?;
        Self::new(f, base, index, row_bytes)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }

    pub fn n_blocks(&self) -> usize {
        self.index.len() - 1
    }
    pub fn n_rows(&self) -> u64 {
        *self.index.last().unwrap()
    }

    /// The block holding `row`.
    ///
    /// Rows per block barely varies, so the search is seeded with the uniform
    /// estimate and corrected — which lands within a block or two in practice,
    /// turning ~24 dependent memory accesses into ~2. Falls back to a bisection
    /// when the estimate is far off, so the bound is still logarithmic.
    pub fn block_of(&self, row: u64) -> Option<usize> {
        if row >= self.n_rows() {
            return None;
        }
        let n = self.n_blocks();
        let mut guess = ((row as u128 * n as u128) / self.n_rows().max(1) as u128) as usize;
        guess = guess.min(n - 1);
        // Walk at most a couple of steps before giving up on the estimate.
        for _ in 0..3 {
            if row >= self.index[guess] && row < self.index[guess + 1] {
                return Some(guess);
            }
            if row < self.index[guess] {
                if guess == 0 {
                    break;
                }
                guess -= 1;
            } else {
                if guess + 1 >= n {
                    break;
                }
                guess += 1;
            }
        }
        // partition_point gives the first block whose START exceeds `row`.
        Some(self.index.partition_point(|&s| s <= row) - 1)
    }

    /// Read one row's bytes.
    pub fn read_row(&self, row: u64) -> Result<Vec<u8>, String> {
        let mut out = self.read_rows(&[row])?;
        out.remove(&row)
            .ok_or_else(|| format!("row {row} not produced"))
    }

    /// Read many rows, one aligned block read per DISTINCT block.
    ///
    /// Rows sharing a block are served from one read, and the reads are issued in
    /// block order — which is what makes a prefill sweep sequentialish instead of
    /// a scatter of independent seeks.
    pub fn read_rows(&self, rows: &[u64]) -> Result<BTreeMap<u64, Vec<u8>>, String> {
        // Group by block first so a block is never read twice.
        let mut by_block: BTreeMap<usize, Vec<u64>> = BTreeMap::new();
        for &r in rows {
            let b = self
                .block_of(r)
                .ok_or_else(|| format!("row {r} is past the end of the table"))?;
            by_block.entry(b).or_default().push(r);
        }
        let mut out = BTreeMap::new();
        let mut buf = vec![0u8; BLOCK];
        for (b, want) in by_block {
            self.read_block(b, &mut buf)?;
            let n = u16::from_le_bytes([buf[0], buf[1]]) as usize;
            let codec = u16::from_le_bytes([buf[2], buf[3]]);
            if Codec::from_u16(codec).is_none() {
                return Err(format!("block {b}: unknown codec {codec}"));
            }
            let first = self.index[b];
            for r in want {
                let slot = (r - first) as usize;
                if slot >= n {
                    return Err(format!(
                        "block {b} holds {n} rows but row {r} maps to slot {slot} \
                         (index and blocks disagree)"
                    ));
                }
                let o = HDR + slot * 4;
                let off = u32::from_le_bytes([buf[o], buf[o + 1], buf[o + 2], buf[o + 3]]) as usize;
                if off + self.row_bytes > BLOCK {
                    return Err(format!("block {b} slot {slot}: row runs past the block"));
                }
                out.insert(r, buf[off..off + self.row_bytes].to_vec());
            }
        }
        Ok(out)
    }

    fn read_block(&self, b: usize, buf: &mut [u8]) -> Result<(), String> {
        let off = self.base + (b as u64) * BLOCK as u64;
        read_exact_at(&self.file, buf, off).map_err(|e| format!("block {b} at {off}: {e}"))
    }
}

fn read_exact_at(file: &File, dst: &mut [u8], offset: u64) -> IoResult<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileExt;
        file.read_exact_at(dst, offset)
    }
    #[cfg(not(unix))]
    {
        use std::io::{Read, Seek, SeekFrom};
        let mut f = file.try_clone()?;
        f.seek(SeekFrom::Start(offset))?;
        f.read_exact(dst)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// 320-byte rows, the real table's width at source precision (160 x bf16).
    const ROW: usize = 320;

    fn make_rows(n: usize) -> Vec<Vec<u8>> {
        (0..n)
            .map(|i| (0..ROW).map(|j| (i.wrapping_mul(31) ^ j) as u8).collect())
            .collect()
    }

    fn store_of(rows: &[Vec<u8>]) -> (NgramStore, tempfile_shim::Temp) {
        let (blocks, index) = pack(rows);
        let t = tempfile_shim::Temp::new();
        {
            let mut f = File::create(t.path()).unwrap();
            f.write_all(&blocks).unwrap();
            f.flush().unwrap();
        }
        let f = File::open(t.path()).unwrap();
        (NgramStore::new(f, 0, index, ROW).unwrap(), t)
    }

    /// Minimal temp-file helper — no dev-dependency for four lines of work.
    mod tempfile_shim {
        use std::path::{Path, PathBuf};
        pub struct Temp(PathBuf);
        impl Temp {
            pub fn new() -> Self {
                let n = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos();
                Self(std::env::temp_dir().join(format!(
                    "hipfire-ngram-{n}-{:?}.bin",
                    std::thread::current().id()
                )))
            }
            pub fn path(&self) -> &Path {
                &self.0
            }
        }
        impl Drop for Temp {
            fn drop(&mut self) {
                let _ = std::fs::remove_file(&self.0);
            }
        }
    }

    #[test]
    fn every_row_round_trips_byte_for_byte() {
        let rows = make_rows(500);
        let (s, _t) = store_of(&rows);
        assert_eq!(s.n_rows(), 500);
        for (i, want) in rows.iter().enumerate() {
            assert_eq!(&s.read_row(i as u64).unwrap(), want, "row {i}");
        }
    }

    /// Blocks are exactly 4096 and the file is a whole number of them — the
    /// precondition for `O_DIRECT`, which is the whole reason for this layout.
    #[test]
    fn blocks_are_whole_and_aligned() {
        let (blocks, index) = pack(&make_rows(500));
        assert_eq!(blocks.len() % BLOCK, 0);
        assert_eq!(blocks.len() / BLOCK, index.len() - 1);
        // A non-aligned base must be refused rather than silently mis-read.
        let t = tempfile_shim::Temp::new();
        File::create(t.path()).unwrap().write_all(&blocks).unwrap();
        let f = File::open(t.path()).unwrap();
        assert!(NgramStore::new(f, 1, index, ROW).is_err());
    }

    /// Rows must never straddle: the packer stops before overflowing, so a lookup
    /// is always exactly one block read.
    #[test]
    fn rows_never_straddle_a_block() {
        let (blocks, index) = pack(&make_rows(500));
        for b in 0..index.len() - 1 {
            let blk = &blocks[b * BLOCK..(b + 1) * BLOCK];
            let n = u16::from_le_bytes([blk[0], blk[1]]) as usize;
            assert_eq!(
                n as u64,
                index[b + 1] - index[b],
                "block {b} count vs index"
            );
            for slot in 0..n {
                let o = HDR + slot * 4;
                let off = u32::from_le_bytes([blk[o], blk[o + 1], blk[o + 2], blk[o + 3]]) as usize;
                assert!(off + ROW <= BLOCK, "block {b} slot {slot} overruns");
            }
        }
    }

    /// The index resolves every row to the block that actually holds it.
    #[test]
    fn block_of_agrees_with_the_index_everywhere() {
        let rows = make_rows(500);
        let (s, _t) = store_of(&rows);
        for r in 0..s.n_rows() {
            let b = s.block_of(r).unwrap();
            assert!(
                r >= s.index[b] && r < s.index[b + 1],
                "row {r} -> block {b}"
            );
        }
        assert_eq!(s.block_of(s.n_rows()), None, "past the end must be None");
    }

    /// A token needs `heads` scattered rows; rows sharing a block must cost one
    /// read, not several, and all requested rows must come back.
    #[test]
    fn batched_read_dedups_blocks_and_returns_all_rows() {
        let rows = make_rows(500);
        let (s, _t) = store_of(&rows);
        // 16 scattered rows, the real per-token count, plus two that share a block.
        let mut want: Vec<u64> = (0..16).map(|i| (i * 29) % 500).collect();
        want.push(want[0] + 1);
        let got = s.read_rows(&want).unwrap();
        for r in &want {
            assert_eq!(&got[r], &rows[*r as usize], "row {r}");
        }
        assert_eq!(
            got.len(),
            want.iter().collect::<std::collections::BTreeSet<_>>().len()
        );
    }

    /// Variable packing is the point: a block holds as many rows as fit, and the
    /// count is derived from the index rather than assumed.
    #[test]
    fn packing_is_variable_and_dense() {
        let (_, index) = pack(&make_rows(500));
        let counts: Vec<u64> = index.windows(2).map(|w| w[1] - w[0]).collect();
        // 4096 - 4 header, with 4 B of offset per row alongside a 320 B row.
        let expect = (BLOCK - HDR) / (ROW + 4);
        assert!(counts[..counts.len() - 1]
            .iter()
            .all(|&c| c as usize == expect));
        assert!(
            counts.last().unwrap() <= &(expect as u64),
            "tail block may be short"
        );
    }

    /// A corrupt index must fail loudly rather than return a neighbouring row.
    #[test]
    fn index_and_blocks_disagreeing_is_an_error() {
        let rows = make_rows(60);
        let (blocks, mut index) = pack(&rows);
        let t = tempfile_shim::Temp::new();
        File::create(t.path()).unwrap().write_all(&blocks).unwrap();
        // Claim block 0 holds more rows than it does, while keeping the index
        // MONOTONIC — an index that is merely non-increasing is already refused by
        // the constructor, so that would not exercise this path.
        assert!(index.len() > 2, "need at least two blocks");
        index[1] = index[2] - 1;
        let f = File::open(t.path()).unwrap();
        let s = NgramStore::new(f, 0, index.clone(), ROW).unwrap();
        // A row inside block 0's CLAIMED span but past its actual count.
        let bad = index[1] - 1;
        assert_eq!(s.block_of(bad), Some(0));
        assert!(
            s.read_row(bad).is_err(),
            "a row mapping past its block's real count must error, not alias a \
             neighbouring row"
        );
    }

    #[test]
    fn strictly_increasing_index_is_enforced() {
        let t = tempfile_shim::Temp::new();
        File::create(t.path())
            .unwrap()
            .write_all(&[0u8; BLOCK])
            .unwrap();
        let f = File::open(t.path()).unwrap();
        assert!(NgramStore::new(f, 0, vec![0, 5, 5], ROW).is_err());
    }
}
